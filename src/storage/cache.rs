use super::layer::*;
use crate::layer::*;
use crate::proof::LayerCommitment;
use async_trait::async_trait;
use std::collections::HashMap;
use std::io;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::{Arc, Mutex, RwLock, Weak};
use tdb_succinct::{StringDict, TypedDict};

const DEFAULT_PROOF_CACHE_ENTRIES: usize = 64;
const DEFAULT_PROOF_CACHE_BYTES: usize = 256 * 1024 * 1024;
const MAX_INFLIGHT_PROOF_LOADS: usize = 2;
// LRU node/key, OnceCell, Arc control blocks, and allocator bookkeeping that
// live outside LayerCommitment's own conservative accounting weight.
const PROOF_CACHE_ENTRY_ACCOUNTING_OVERHEAD: usize = 512;

fn proof_cache_weight(proof: &LayerCommitment) -> usize {
    proof
        .estimated_cache_bytes()
        .saturating_add(PROOF_CACHE_ENTRY_ACCOUNTING_OVERHEAD)
}

pub trait LayerCache: 'static + Send + Sync {
    fn get_layer_from_cache(&self, name: [u32; 5]) -> Option<Arc<InternalLayer>>;
    fn cache_layer(&self, layer: Arc<InternalLayer>);

    fn invalidate(&self, name: [u32; 5]);

    /// Returns statistics about the cache: (total_entries, live_entries, dead_entries)
    /// Default implementation returns (0, 0, 0) for caches that don't track this.
    fn cache_stats(&self) -> (usize, usize, usize) {
        (0, 0, 0)
    }

    /// Remove stale entries from the cache. Returns number of entries removed.
    /// Default implementation does nothing.
    fn cleanup_stale_entries(&self) -> usize {
        0
    }

    /// Returns bytes of backing data: (total_bytes, live_bytes, dead_bytes).
    /// Default implementation returns (0, 0, 0).
    fn cache_memory_bytes(&self) -> (usize, usize, usize) {
        (0, 0, 0)
    }
}

pub struct NoCache;

impl LayerCache for NoCache {
    fn get_layer_from_cache(&self, _name: [u32; 5]) -> Option<Arc<InternalLayer>> {
        None
    }

    fn cache_layer(&self, _layer: Arc<InternalLayer>) {}

    fn invalidate(&self, _name: [u32; 5]) {}
}

lazy_static! {
    pub static ref NOCACHE: Arc<dyn LayerCache> = Arc::new(NoCache);
}

// locking isn't really ideal but the lock window will be relatively small so it shouldn't hurt performance too much except on heavy updates.
// ideally we should be using some concurrent hashmap implementation instead.

/// Threshold for automatic cleanup: when cache size exceeds this, trigger cleanup on next cache_layer call.
const CACHE_CLEANUP_THRESHOLD: usize = 100;

/// Only cleanup if dead entries exceed this percentage of total.
const DEAD_ENTRY_PERCENTAGE_THRESHOLD: usize = 20;

#[derive(Default)]
pub struct LockingHashMapLayerCache {
    cache: RwLock<HashMap<[u32; 5], Weak<InternalLayer>>>,
}

impl LockingHashMapLayerCache {
    pub fn new() -> Self {
        Default::default()
    }

    /// Returns statistics about the cache: (total_entries, live_entries, dead_entries)
    /// Live entries are those where the Weak reference can still be upgraded.
    /// Dead entries are stale weak references that should be cleaned up.
    pub fn stats(&self) -> (usize, usize, usize) {
        let cache = self
            .cache
            .read()
            .expect("rwlock read should always succeed, but got poisoned");
        let total = cache.len();
        let live = cache.values().filter(|w| w.strong_count() > 0).count();
        let dead = total - live;
        (total, live, dead)
    }

    /// Remove all stale (dead) weak references from the cache.
    /// Returns the number of entries removed.
    pub fn cleanup_stale(&self) -> usize {
        let mut cache = self
            .cache
            .write()
            .expect("rwlock write should always succeed, but got poisoned");
        let before = cache.len();
        cache.retain(|_, weak| weak.strong_count() > 0);
        let after = cache.len();
        before - after
    }

    /// Returns (total_bytes, live_bytes, dead_bytes) of stored_size() across cache entries.
    pub fn memory_bytes(&self) -> (usize, usize, usize) {
        let cache = self
            .cache
            .read()
            .expect("rwlock read should always succeed, but got poisoned");
        let mut live = 0usize;
        let mut dead = 0usize;
        for w in cache.values() {
            if let Some(layer) = w.upgrade() {
                live += layer.stored_size();
            } else {
                // Dead entries have no accessible size, count as 0
                dead += 0;
            }
        }
        (live + dead, live, dead)
    }
}

impl LayerCache for LockingHashMapLayerCache {
    fn get_layer_from_cache(&self, name: [u32; 5]) -> Option<Arc<InternalLayer>> {
        let cache = self
            .cache
            .read()
            .expect("rwlock read should always succeed");

        let result = cache.get(&name).map(|c| c.to_owned());
        std::mem::drop(cache);

        match result {
            None => None,
            Some(weak) => match weak.upgrade() {
                None => {
                    self.cache
                        .write()
                        .expect("rwlock write should always succeed")
                        .remove(&name);
                    None
                }
                Some(result) => Some(result),
            },
        }
    }

    fn cache_layer(&self, layer: Arc<InternalLayer>) {
        let mut cache = self
            .cache
            .write()
            .expect("rwlock write should always succeed");

        // Automatic cleanup: when cache exceeds threshold, check for stale entries
        let cache_size = cache.len();
        if cache_size >= CACHE_CLEANUP_THRESHOLD {
            // Count dead entries
            let dead_count = cache.values().filter(|w| w.strong_count() == 0).count();
            let dead_percentage = (dead_count * 100) / cache_size.max(1);

            // Only cleanup if dead entries exceed threshold percentage
            if dead_percentage >= DEAD_ENTRY_PERCENTAGE_THRESHOLD {
                cache.retain(|_, weak| weak.strong_count() > 0);
            }
        }

        cache.insert(layer.name(), Arc::downgrade(&layer));
    }

    fn invalidate(&self, name: [u32; 5]) {
        // the dumb way - we just delete the thing from cache forcing a refresh
        let mut cache = self
            .cache
            .write()
            .expect("rwlock read should always succeed");

        cache.remove(&name);
    }

    fn cache_stats(&self) -> (usize, usize, usize) {
        self.stats()
    }

    fn cleanup_stale_entries(&self) -> usize {
        self.cleanup_stale()
    }

    fn cache_memory_bytes(&self) -> (usize, usize, usize) {
        self.memory_bytes()
    }
}

struct ProofLayerCache {
    entries: lru::LruCache<[u32; 5], Arc<tokio::sync::OnceCell<Option<Arc<LayerCommitment>>>>>,
    bytes: usize,
    max_entries: usize,
    max_bytes: usize,
}

impl ProofLayerCache {
    fn new(max_entries: usize, max_bytes: usize, max_inflight: usize) -> Self {
        // Unresolved single-flight cells are never eviction candidates. Leave
        // bounded room for them in addition to the configured resident set so
        // LruCache's own capacity eviction cannot discard an active load.
        let physical_entries = max_entries.saturating_add(max_inflight).max(1);
        Self {
            entries: lru::LruCache::new(NonZeroUsize::new(physical_entries).unwrap()),
            bytes: 0,
            max_entries,
            max_bytes,
        }
    }

    fn recompute_and_evict(&mut self) {
        loop {
            self.bytes = self
                .entries
                .iter()
                .filter_map(|(_, cell)| cell.get().and_then(|proof| proof.as_ref()))
                .map(|proof| proof_cache_weight(proof))
                .fold(0usize, usize::saturating_add);
            let settled = self
                .entries
                .iter()
                .filter(|(_, cell)| cell.get().is_some())
                .count();
            if settled <= self.max_entries && self.bytes <= self.max_bytes {
                break;
            }
            let victim = self
                .entries
                .iter()
                .filter_map(|(name, cell)| cell.get().is_some().then_some(*name))
                .last();
            let Some(victim) = victim else {
                // Every remaining cell is an unresolved load and must stay
                // resident until its semaphore-protected initialization ends.
                break;
            };
            self.entries.pop(&victim);
        }
    }

    fn settle_loaded(
        &mut self,
        name: [u32; 5],
        cell: &Arc<tokio::sync::OnceCell<Option<Arc<LayerCommitment>>>>,
        load_succeeded: bool,
    ) {
        let resident = self
            .entries
            .peek(&name)
            .map_or(false, |candidate| Arc::ptr_eq(candidate, cell));
        if resident {
            let exceeds_limit = !load_succeeded
                || self.max_entries == 0
                || cell
                    .get()
                    .and_then(|proof| proof.as_ref())
                    .map_or(false, |proof| proof_cache_weight(proof) > self.max_bytes);
            if exceeds_limit {
                self.entries.pop(&name);
            }
            self.recompute_and_evict();
        }
    }
}

#[derive(Clone)]
pub struct CachedLayerStore {
    pub(crate) inner: Arc<dyn LayerStore>,
    pub(crate) cache: Arc<dyn LayerCache>,
    proof_cache: Arc<Mutex<ProofLayerCache>>,
    proof_loads: Arc<tokio::sync::Semaphore>,
}

impl CachedLayerStore {
    pub fn new<S: LayerStore, C: LayerCache>(inner: S, cache: C) -> CachedLayerStore {
        Self::new_with_proof_cache_limits(
            inner,
            cache,
            DEFAULT_PROOF_CACHE_ENTRIES,
            DEFAULT_PROOF_CACHE_BYTES,
        )
    }

    pub fn new_with_proof_cache_limits<S: LayerStore, C: LayerCache>(
        inner: S,
        cache: C,
        max_entries: usize,
        max_bytes: usize,
    ) -> CachedLayerStore {
        // max_entries and the derived in-flight semaphore are hard caps.
        // max_bytes bounds a conservative logical accounting weight, including
        // container/capacity/fixed allocator overhead; it is not an OS RSS cap.
        let max_inflight = max_entries.max(1).min(MAX_INFLIGHT_PROOF_LOADS);
        CachedLayerStore {
            inner: Arc::new(inner),
            cache: Arc::new(cache),
            proof_cache: Arc::new(Mutex::new(ProofLayerCache::new(
                max_entries,
                max_bytes,
                max_inflight,
            ))),
            proof_loads: Arc::new(tokio::sync::Semaphore::new(max_inflight)),
        }
    }

    pub fn invalidate(&self, name: [u32; 5]) {
        self.cache.invalidate(name);
        let mut proof_cache = self.proof_cache.lock().unwrap();
        proof_cache.entries.pop(&name);
        proof_cache.recompute_and_evict();
    }

    /// Returns cache statistics: (total_entries, live_entries, dead_entries)
    pub fn cache_stats(&self) -> (usize, usize, usize) {
        self.cache.cache_stats()
    }

    /// Remove stale entries from the cache. Returns number of entries removed.
    pub fn cleanup_stale_entries(&self) -> usize {
        self.cache.cleanup_stale_entries()
    }

    /// Returns bytes of backing data: (total_bytes, live_bytes, dead_bytes).
    pub fn cache_memory_bytes(&self) -> (usize, usize, usize) {
        self.cache.cache_memory_bytes()
    }
}

#[async_trait]
impl LayerStore for CachedLayerStore {
    async fn layers(&self) -> io::Result<Vec<[u32; 5]>> {
        self.inner.layers().await
    }

    async fn get_layer(&self, name: [u32; 5]) -> io::Result<Option<Arc<InternalLayer>>> {
        self.inner
            .get_layer_with_cache(name, self.cache.clone())
            .await
    }

    async fn get_layer_with_cache(
        &self,
        name: [u32; 5],
        cache: Arc<dyn LayerCache>,
    ) -> io::Result<Option<Arc<InternalLayer>>> {
        self.inner.get_layer_with_cache(name, cache).await
    }

    async fn get_unrolled_layer(&self, name: [u32; 5]) -> io::Result<Option<Arc<InternalLayer>>> {
        self.inner.get_unrolled_layer(name).await
    }

    async fn finalize_layer(&self, name: [u32; 5]) -> io::Result<()> {
        self.inner.finalize_layer(name).await
    }

    async fn abort_layer(&self, name: [u32; 5]) -> io::Result<()> {
        {
            let mut proof_cache = self.proof_cache.lock().unwrap();
            proof_cache.entries.pop(&name);
            proof_cache.recompute_and_evict();
        }
        self.inner.abort_layer(name).await
    }

    async fn proof_layer(&self, name: [u32; 5]) -> io::Result<Option<Arc<LayerCommitment>>> {
        let existing = {
            let mut cache = self.proof_cache.lock().unwrap();
            cache.entries.get(&name).cloned()
        };
        let (cell, _permit) = if let Some(cell) = existing {
            (cell, None)
        } else {
            let permit = self
                .proof_loads
                .clone()
                .acquire_owned()
                .await
                .map_err(|_| io::Error::new(io::ErrorKind::Other, "proof load gate closed"))?;
            let mut cache = self.proof_cache.lock().unwrap();
            if let Some(cell) = cache.entries.get(&name) {
                (cell.clone(), None)
            } else {
                let cell = Arc::new(tokio::sync::OnceCell::new());
                cache.entries.put(name, cell.clone());
                cache.recompute_and_evict();
                (cell, Some(permit))
            }
        };
        let result = cell
            .get_or_try_init(|| self.inner.proof_layer(name))
            .await
            .cloned();
        let mut cache = self.proof_cache.lock().unwrap();
        cache.settle_loaded(name, &cell, result.is_ok());
        result
    }

    async fn write_proof_layer(&self, name: [u32; 5], proof: &LayerCommitment) -> io::Result<()> {
        self.inner.write_proof_layer(name, proof).await?;
        let cell = Arc::new(tokio::sync::OnceCell::new());
        cell.set(Some(Arc::new(proof.clone()))).map_err(|_| {
            io::Error::new(io::ErrorKind::Other, "new proof cache cell was already set")
        })?;
        let mut cache = self.proof_cache.lock().unwrap();
        if cache.max_entries != 0 {
            if cache.entries.len() == cache.entries.cap().get() {
                if let Some(victim) = cache
                    .entries
                    .iter()
                    .filter_map(|(candidate, value)| value.get().is_some().then_some(*candidate))
                    .last()
                {
                    cache.entries.pop(&victim);
                }
            }
            cache.entries.put(name, cell.clone());
            cache.settle_loaded(name, &cell, true);
        }
        Ok(())
    }

    async fn get_layer_parent_name(&self, name: [u32; 5]) -> io::Result<Option<[u32; 5]>> {
        // is layer in cache? if so, we can use the cached version
        if let Some(layer) = self.cache.get_layer_from_cache(name) {
            Ok(layer.parent_name())
        } else {
            self.inner.get_layer_parent_name(name).await
        }
    }

    async fn get_node_dictionary(&self, name: [u32; 5]) -> io::Result<Option<StringDict>> {
        // is layer in cache? if so, we can use the cached version
        if let Some(layer) = self.cache.get_layer_from_cache(name) {
            // unless it is a rollup
            if !layer.is_rollup() {
                return Ok(Some(layer.node_dictionary().clone()));
            }
        }

        self.inner.get_node_dictionary(name).await
    }

    async fn get_predicate_dictionary(&self, name: [u32; 5]) -> io::Result<Option<StringDict>> {
        // is layer in cache? if so, we can use the cached version
        if let Some(layer) = self.cache.get_layer_from_cache(name) {
            // unless it is a rollup
            if !layer.is_rollup() {
                return Ok(Some(layer.predicate_dictionary().clone()));
            }
        }

        self.inner.get_predicate_dictionary(name).await
    }

    async fn get_value_dictionary(&self, name: [u32; 5]) -> io::Result<Option<TypedDict>> {
        // is layer in cache? if so, we can use the cached version
        if let Some(layer) = self.cache.get_layer_from_cache(name) {
            // unless it is a rollup
            if !layer.is_rollup() {
                return Ok(Some(layer.value_dictionary().clone()));
            }
        }

        self.inner.get_value_dictionary(name).await
    }

    async fn get_node_count(&self, name: [u32; 5]) -> io::Result<Option<u64>> {
        // is layer in cache? if so, we can use the cached version
        if let Some(layer) = self.cache.get_layer_from_cache(name) {
            // unless it is a rollup
            if !layer.is_rollup() {
                return Ok(Some(layer.node_dictionary().num_entries() as u64));
            }
        }

        self.inner.get_node_count(name).await
    }

    async fn get_predicate_count(&self, name: [u32; 5]) -> io::Result<Option<u64>> {
        // is layer in cache? if so, we can use the cached version
        if let Some(layer) = self.cache.get_layer_from_cache(name) {
            // unless it is a rollup
            if !layer.is_rollup() {
                return Ok(Some(layer.predicate_dictionary().num_entries() as u64));
            }
        }

        self.inner.get_value_count(name).await
    }

    async fn get_value_count(&self, name: [u32; 5]) -> io::Result<Option<u64>> {
        // is layer in cache? if so, we can use the cached version
        if let Some(layer) = self.cache.get_layer_from_cache(name) {
            // unless it is a rollup
            if !layer.is_rollup() {
                return Ok(Some(layer.value_dictionary().num_entries() as u64));
            }
        }

        self.inner.get_value_count(name).await
    }

    async fn get_node_value_idmap(&self, name: [u32; 5]) -> io::Result<Option<IdMap>> {
        // is layer in cache? if so, we can use the cached version
        if let Some(layer) = self.cache.get_layer_from_cache(name) {
            // unless it is a rollup
            if !layer.is_rollup() {
                return Ok(Some(layer.node_value_id_map().clone()));
            }
        }

        self.inner.get_node_value_idmap(name).await
    }

    async fn get_predicate_idmap(&self, name: [u32; 5]) -> io::Result<Option<IdMap>> {
        // is layer in cache? if so, we can use the cached version
        if let Some(layer) = self.cache.get_layer_from_cache(name) {
            // unless it is a rollup
            if !layer.is_rollup() {
                return Ok(Some(layer.predicate_id_map().clone()));
            }
        }

        self.inner.get_predicate_idmap(name).await
    }

    async fn create_base_layer(&self) -> io::Result<Box<dyn LayerBuilder>> {
        self.inner.create_base_layer().await
    }

    async fn create_child_layer(&self, parent: [u32; 5]) -> io::Result<Box<dyn LayerBuilder>> {
        self.inner
            .create_child_layer_with_cache(parent, self.cache.clone())
            .await
    }

    async fn create_child_layer_with_cache(
        &self,
        parent: [u32; 5],
        cache: Arc<dyn LayerCache>,
    ) -> io::Result<Box<dyn LayerBuilder>> {
        self.inner
            .create_child_layer_with_cache(parent, cache)
            .await
    }

    async fn perform_rollup(&self, layer: Arc<InternalLayer>) -> io::Result<[u32; 5]> {
        self.inner.perform_rollup(layer).await
    }

    async fn perform_rollup_upto_with_cache(
        &self,
        layer: Arc<InternalLayer>,
        upto: [u32; 5],
        cache: Arc<dyn LayerCache>,
    ) -> io::Result<[u32; 5]> {
        self.inner
            .perform_rollup_upto_with_cache(layer, upto, cache)
            .await
    }

    async fn perform_rollup_upto(
        &self,
        layer: Arc<InternalLayer>,
        upto: [u32; 5],
    ) -> io::Result<[u32; 5]> {
        self.inner
            .perform_rollup_upto_with_cache(layer, upto, self.cache.clone())
            .await
    }

    async fn perform_imprecise_rollup_upto_with_cache(
        &self,
        layer: Arc<InternalLayer>,
        upto: [u32; 5],
        cache: Arc<dyn LayerCache>,
    ) -> io::Result<[u32; 5]> {
        self.inner
            .perform_imprecise_rollup_upto_with_cache(layer, upto, cache)
            .await
    }

    async fn perform_imprecise_rollup_upto(
        &self,
        layer: Arc<InternalLayer>,
        upto: [u32; 5],
    ) -> io::Result<[u32; 5]> {
        self.inner
            .perform_imprecise_rollup_upto_with_cache(layer, upto, self.cache.clone())
            .await
    }

    async fn register_rollup(&self, layer: [u32; 5], rollup: [u32; 5]) -> io::Result<()> {
        // when registering a rollup layer, we need to make sure that
        // the cached version is updated as well.

        // Get the entire parent chain before registering the rollup
        // so we can invalidate all old layers from the cache
        let layer_stack = self.inner.retrieve_layer_stack_names(layer).await?;

        self.inner.register_rollup(layer, rollup).await?;

        // Invalidate the entire parent chain from cache, not just the rolled-up layer.
        // This ensures old layers become "dead" entries that can be cleaned up.
        // Without this, the old parent chain stays "live" because Arc references
        // are held by child layers.
        for old_layer in layer_stack {
            self.cache.invalidate(old_layer);
        }

        Ok(())
    }

    async fn rollup_upto(&self, layer: Arc<InternalLayer>, upto: [u32; 5]) -> io::Result<[u32; 5]> {
        let cache = self.cache.clone();
        self.rollup_upto_with_cache(layer, upto, cache).await
    }

    async fn squash(&self, layer: Arc<InternalLayer>) -> io::Result<[u32; 5]> {
        self.inner.squash(layer).await
    }

    async fn squash_upto(&self, layer: Arc<InternalLayer>, upto: [u32; 5]) -> io::Result<[u32; 5]> {
        self.inner.squash_upto(layer, upto).await
    }

    async fn merge_base_layer(&self, layers: &[[u32; 5]], temp_dir: &Path) -> io::Result<[u32; 5]> {
        self.inner.merge_base_layer(layers, temp_dir).await
    }

    async fn layer_is_ancestor_of(
        &self,
        descendant: [u32; 5],
        ancestor: [u32; 5],
    ) -> io::Result<bool> {
        self.inner.layer_is_ancestor_of(descendant, ancestor).await
    }

    async fn triple_addition_exists(
        &self,
        layer: [u32; 5],
        subject: u64,
        predicate: u64,
        object: u64,
    ) -> io::Result<bool> {
        if let Some(cached) = self.cache.get_layer_from_cache(layer) {
            if !cached.is_rollup() {
                return Ok(cached.internal_triple_addition_exists(subject, predicate, object));
            }
        }

        self.inner
            .triple_addition_exists(layer, subject, predicate, object)
            .await
    }

    async fn triple_removal_exists(
        &self,
        layer: [u32; 5],
        subject: u64,
        predicate: u64,
        object: u64,
    ) -> io::Result<bool> {
        if let Some(cached) = self.cache.get_layer_from_cache(layer) {
            if !cached.is_rollup() {
                return Ok(cached.internal_triple_removal_exists(subject, predicate, object));
            }
        }

        self.inner
            .triple_removal_exists(layer, subject, predicate, object)
            .await
    }

    async fn triple_additions(
        &self,
        layer: [u32; 5],
    ) -> io::Result<OptInternalLayerTripleSubjectIterator> {
        if let Some(cached) = self.cache.get_layer_from_cache(layer) {
            if !cached.is_rollup() {
                return Ok(cached.internal_triple_additions());
            }
        }

        self.inner.triple_additions(layer).await
    }

    async fn triple_removals(
        &self,
        layer: [u32; 5],
    ) -> io::Result<OptInternalLayerTripleSubjectIterator> {
        if let Some(cached) = self.cache.get_layer_from_cache(layer) {
            if !cached.is_rollup() {
                return Ok(cached.internal_triple_removals());
            }
        }

        self.inner.triple_removals(layer).await
    }

    async fn triple_additions_s(
        &self,
        layer: [u32; 5],
        subject: u64,
    ) -> io::Result<Box<dyn Iterator<Item = IdTriple> + Send>> {
        if let Some(cached) = self.cache.get_layer_from_cache(layer) {
            if !cached.is_rollup() {
                return Ok(cached.internal_triple_additions_s(subject));
            }
        }

        self.inner.triple_additions_s(layer, subject).await
    }

    async fn triple_removals_s(
        &self,
        layer: [u32; 5],
        subject: u64,
    ) -> io::Result<Box<dyn Iterator<Item = IdTriple> + Send>> {
        if let Some(cached) = self.cache.get_layer_from_cache(layer) {
            if !cached.is_rollup() {
                return Ok(cached.internal_triple_removals_s(subject));
            }
        }

        self.inner.triple_removals_s(layer, subject).await
    }

    async fn triple_additions_sp(
        &self,
        layer: [u32; 5],
        subject: u64,
        predicate: u64,
    ) -> io::Result<Box<dyn Iterator<Item = IdTriple> + Send>> {
        if let Some(cached) = self.cache.get_layer_from_cache(layer) {
            if !cached.is_rollup() {
                return Ok(cached.internal_triple_additions_sp(subject, predicate));
            }
        }

        self.inner
            .triple_additions_sp(layer, subject, predicate)
            .await
    }

    async fn triple_removals_sp(
        &self,
        layer: [u32; 5],
        subject: u64,
        predicate: u64,
    ) -> io::Result<Box<dyn Iterator<Item = IdTriple> + Send>> {
        if let Some(cached) = self.cache.get_layer_from_cache(layer) {
            if !cached.is_rollup() {
                return Ok(cached.internal_triple_removals_sp(subject, predicate));
            }
        }

        self.inner
            .triple_removals_sp(layer, subject, predicate)
            .await
    }

    async fn triple_additions_p(
        &self,
        layer: [u32; 5],
        predicate: u64,
    ) -> io::Result<Box<dyn Iterator<Item = IdTriple> + Send>> {
        if let Some(cached) = self.cache.get_layer_from_cache(layer) {
            if !cached.is_rollup() {
                return Ok(Box::new(cached.internal_triple_additions_p(predicate))
                    as Box<dyn Iterator<Item = _> + Send>);
            }
        }

        self.inner.triple_additions_p(layer, predicate).await
    }

    async fn triple_removals_p(
        &self,
        layer: [u32; 5],
        predicate: u64,
    ) -> io::Result<Box<dyn Iterator<Item = IdTriple> + Send>> {
        if let Some(cached) = self.cache.get_layer_from_cache(layer) {
            if !cached.is_rollup() {
                return Ok(Box::new(cached.internal_triple_removals_p(predicate))
                    as Box<dyn Iterator<Item = _> + Send>);
            }
        }

        self.inner.triple_removals_p(layer, predicate).await
    }

    async fn triple_additions_o(
        &self,
        layer: [u32; 5],
        object: u64,
    ) -> io::Result<Box<dyn Iterator<Item = IdTriple> + Send>> {
        if let Some(cached) = self.cache.get_layer_from_cache(layer) {
            if !cached.is_rollup() {
                return Ok(cached.internal_triple_additions_o(object));
            }
        }

        self.inner.triple_additions_o(layer, object).await
    }

    async fn triple_removals_o(
        &self,
        layer: [u32; 5],
        object: u64,
    ) -> io::Result<Box<dyn Iterator<Item = IdTriple> + Send>> {
        if let Some(cached) = self.cache.get_layer_from_cache(layer) {
            if !cached.is_rollup() {
                return Ok(cached.internal_triple_removals_o(object));
            }
        }

        self.inner.triple_removals_o(layer, object).await
    }

    async fn triple_layer_addition_count(&self, layer: [u32; 5]) -> io::Result<usize> {
        if let Some(cached) = self.cache.get_layer_from_cache(layer) {
            if !cached.is_rollup() {
                return Ok(cached.internal_triple_layer_addition_count());
            }
        }

        self.inner.triple_layer_addition_count(layer).await
    }

    async fn triple_layer_removal_count(&self, layer: [u32; 5]) -> io::Result<usize> {
        if let Some(cached) = self.cache.get_layer_from_cache(layer) {
            if !cached.is_rollup() {
                return Ok(cached.internal_triple_layer_removal_count());
            }
        }

        self.inner.triple_layer_removal_count(layer).await
    }

    async fn retrieve_layer_stack_names(&self, name: [u32; 5]) -> io::Result<Vec<[u32; 5]>> {
        self.inner.retrieve_layer_stack_names(name).await
    }

    async fn retrieve_layer_stack_names_upto(
        &self,
        name: [u32; 5],
        upto: [u32; 5],
    ) -> io::Result<Vec<[u32; 5]>> {
        self.inner.retrieve_layer_stack_names_upto(name, upto).await
    }

    fn layer_cache_stats(&self) -> (usize, usize, usize) {
        self.cache.cache_stats()
    }

    fn cleanup_layer_cache(&self) -> usize {
        self.cache.cleanup_stale_entries()
    }

    fn layer_cache_memory_bytes(&self) -> (usize, usize, usize) {
        self.cache.cache_memory_bytes()
    }

    fn proof_cache_memory_bytes(&self) -> usize {
        self.proof_cache.lock().unwrap().bytes
    }

    fn invalidate(&self, name: [u32; 5]) {
        self.cache.invalidate(name);
        let mut proof_cache = self.proof_cache.lock().unwrap();
        proof_cache.entries.pop(&name);
        proof_cache.recompute_and_evict();
        std::mem::drop(proof_cache);
        self.inner.invalidate(name);
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::storage::directory::*;
    use crate::storage::memory::*;
    use crate::storage::{FileLoad, FileStore};
    use tempfile::tempdir;

    fn cached_layer_eq(layer1: &dyn Layer, layer2: &dyn Layer) -> bool {
        // a trait object consists of two parts, a pointer to the concrete data, followed by a vtable.
        // we consider two layers equal if that first part, the pointer to the concrete data, is equal.
        unsafe {
            let ptr1 = *(layer1 as *const dyn Layer as *const usize);
            let ptr2 = *(layer2 as *const dyn Layer as *const usize);
            ptr1 == ptr2
        }
    }

    #[tokio::test]
    async fn cached_memory_layer_store_returns_same_layer_multiple_times() {
        let store = CachedLayerStore::new(MemoryLayerStore::new(), LockingHashMapLayerCache::new());
        let mut builder = store.create_base_layer().await.unwrap();
        let base_name = builder.name();

        builder.add_value_triple(ValueTriple::new_string_value("cow", "says", "moo"));
        builder.add_value_triple(ValueTriple::new_string_value("pig", "says", "oink"));
        builder.add_value_triple(ValueTriple::new_string_value("duck", "says", "quack"));

        builder.commit_boxed().await.unwrap();

        builder = store.create_child_layer(base_name).await.unwrap();
        let child_name = builder.name();

        builder.remove_value_triple(ValueTriple::new_string_value("duck", "says", "quack"));
        builder.add_value_triple(ValueTriple::new_node("cow", "likes", "pig"));

        builder.commit_boxed().await.unwrap();

        let layer1 = store.get_layer(child_name).await.unwrap().unwrap();
        let layer2 = store.get_layer(child_name).await.unwrap().unwrap();

        let base_layer = store.cache.get_layer_from_cache(base_name).unwrap();
        let base_layer_2 = store.get_layer(base_name).await.unwrap().unwrap();

        assert!(cached_layer_eq(&*layer1, &*layer2));
        assert!(cached_layer_eq(&*base_layer, &*base_layer_2));
    }

    #[tokio::test]
    async fn cached_directory_layer_store_returns_same_layer_multiple_times() {
        let dir = tempdir().unwrap();
        let store = CachedLayerStore::new(
            DirectoryLayerStore::new(dir.path()),
            LockingHashMapLayerCache::new(),
        );
        let mut builder = store.create_base_layer().await.unwrap();
        let base_name = builder.name();

        builder.add_value_triple(ValueTriple::new_string_value("cow", "says", "moo"));
        builder.add_value_triple(ValueTriple::new_string_value("pig", "says", "oink"));
        builder.add_value_triple(ValueTriple::new_string_value("duck", "says", "quack"));

        builder.commit_boxed().await.unwrap();

        builder = store.create_child_layer(base_name).await.unwrap();
        let child_name = builder.name();

        builder.remove_value_triple(ValueTriple::new_string_value("duck", "says", "quack"));
        builder.add_value_triple(ValueTriple::new_node("cow", "likes", "pig"));

        builder.commit_boxed().await.unwrap();

        let layer1 = store.get_layer(child_name).await.unwrap().unwrap();
        let layer2 = store.get_layer(child_name).await.unwrap().unwrap();

        let base_layer = store.cache.get_layer_from_cache(base_name).unwrap();
        let base_layer_2 = store.get_layer(base_name).await.unwrap().unwrap();

        assert!(cached_layer_eq(&*layer1, &*layer2));
        assert!(cached_layer_eq(&*base_layer, &*base_layer_2));
    }

    #[tokio::test]
    async fn cached_layer_store_forgets_entries_when_they_are_dropped() {
        let store = CachedLayerStore::new(MemoryLayerStore::new(), LockingHashMapLayerCache::new());
        let mut builder = store.create_base_layer().await.unwrap();
        let base_name = builder.name();

        builder.add_value_triple(ValueTriple::new_string_value("cow", "says", "moo"));
        builder.add_value_triple(ValueTriple::new_string_value("pig", "says", "oink"));
        builder.add_value_triple(ValueTriple::new_string_value("duck", "says", "quack"));

        builder.commit_boxed().await.unwrap();

        let layer = store.get_layer(base_name).await.unwrap().unwrap();
        let weak = Arc::downgrade(&layer);

        // we expect 2 weak pointers, the one we made above and the one stored in cache
        assert_eq!(2, Arc::weak_count(&layer));

        // forget the layers
        std::mem::drop(layer);

        // according to our weak reference, there's no longer any strong reference around
        assert!(weak.upgrade().is_none());

        // retrieving the same layer again works just fine
        let layer = store.get_layer(base_name).await.unwrap().unwrap();

        // and only has one weak pointer pointing to it, the newly cached one
        assert_eq!(1, Arc::weak_count(&layer));
    }

    #[tokio::test]
    async fn proof_layers_are_decoded_once_until_explicit_invalidation() {
        let store = CachedLayerStore::new(MemoryLayerStore::new(), LockingHashMapLayerCache::new());
        let mut builder = store.create_base_layer().await.unwrap();
        let name = builder.name();
        builder.add_value_triple(ValueTriple::new_node("a", "p", "b"));
        builder.commit_boxed().await.unwrap();
        store.prepare_proof_layer(name).await.unwrap();

        let first = store.proof_layer(name).await.unwrap().unwrap();
        let second = store.proof_layer(name).await.unwrap().unwrap();
        assert!(Arc::ptr_eq(&first, &second));

        LayerStore::invalidate(&store, name);
        let (reloaded, concurrent) = tokio::join!(store.proof_layer(name), store.proof_layer(name));
        let reloaded = reloaded.unwrap().unwrap();
        let concurrent = concurrent.unwrap().unwrap();
        assert!(Arc::ptr_eq(&reloaded, &concurrent));
        assert!(!Arc::ptr_eq(&first, &reloaded));
        assert_eq!(*first, *reloaded);
    }

    async fn memory_store_with_two_proof_layers() -> (MemoryLayerStore, [u32; 5], [u32; 5]) {
        let store = MemoryLayerStore::new();
        let mut first = store.create_base_layer().await.unwrap();
        let first_name = first.name();
        first.add_value_triple(ValueTriple::new_node("first", "p", "value"));
        first.commit_boxed().await.unwrap();
        store.prepare_proof_layer(first_name).await.unwrap();
        let mut second = store.create_base_layer().await.unwrap();
        let second_name = second.name();
        second.add_value_triple(ValueTriple::new_node("second", "p", "value"));
        second.commit_boxed().await.unwrap();
        store.prepare_proof_layer(second_name).await.unwrap();
        (store, first_name, second_name)
    }

    #[derive(Clone)]
    struct SlowProofFile {
        inner: MemoryBackedStore,
        slow: bool,
        loads: Arc<std::sync::atomic::AtomicUsize>,
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Semaphore>,
    }

    #[async_trait]
    impl FileStore for SlowProofFile {
        type Write = <MemoryBackedStore as FileStore>::Write;

        async fn open_write(&self) -> io::Result<Self::Write> {
            self.inner.open_write().await
        }
    }

    #[async_trait]
    impl FileLoad for SlowProofFile {
        type Read = <MemoryBackedStore as FileLoad>::Read;

        async fn exists(&self) -> io::Result<bool> {
            self.inner.exists().await
        }

        async fn size(&self) -> io::Result<usize> {
            self.inner.size().await
        }

        async fn open_read_from(&self, offset: usize) -> io::Result<Self::Read> {
            self.inner.open_read_from(offset).await
        }

        async fn map(&self) -> io::Result<bytes::Bytes> {
            if self.slow {
                self.loads.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                // Retain a notification permit if the test observer has not
                // started waiting yet; notify_waiters would make that race
                // capable of hanging the regression test.
                self.started.notify_one();
                self.release
                    .acquire()
                    .await
                    .map_err(|_| io::Error::new(io::ErrorKind::Other, "test gate closed"))?
                    .forget();
            }
            self.inner.map().await
        }
    }

    #[derive(Clone)]
    struct SlowProofStore {
        inner: MemoryLayerStore,
        loads: Arc<std::sync::atomic::AtomicUsize>,
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Semaphore>,
    }

    impl SlowProofStore {
        fn new(inner: MemoryLayerStore) -> Self {
            Self {
                inner,
                loads: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                started: Arc::new(tokio::sync::Notify::new()),
                release: Arc::new(tokio::sync::Semaphore::new(0)),
            }
        }

        async fn wait_for_loads(&self, count: usize) {
            loop {
                if self.loads.load(std::sync::atomic::Ordering::SeqCst) >= count {
                    return;
                }
                self.started.notified().await;
            }
        }
    }

    #[async_trait]
    impl PersistentLayerStore for SlowProofStore {
        type File = SlowProofFile;

        async fn directories(&self) -> io::Result<Vec<[u32; 5]>> {
            self.inner.directories().await
        }

        async fn create_named_directory(&self, id: [u32; 5]) -> io::Result<[u32; 5]> {
            self.inner.create_named_directory(id).await
        }

        async fn directory_exists(&self, id: [u32; 5]) -> io::Result<bool> {
            self.inner.directory_exists(id).await
        }

        async fn get_file(&self, id: [u32; 5], name: &str) -> io::Result<Self::File> {
            Ok(SlowProofFile {
                inner: self.inner.get_file(id, name).await?,
                slow: name == crate::storage::consts::FILENAMES.proof_state,
                loads: self.loads.clone(),
                started: self.started.clone(),
                release: self.release.clone(),
            })
        }

        async fn file_exists(&self, id: [u32; 5], name: &str) -> io::Result<bool> {
            self.inner.file_exists(id, name).await
        }

        async fn abort_directory(&self, id: [u32; 5]) -> io::Result<()> {
            self.inner.abort_directory(id).await
        }

        async fn publish_imported_layers_shared(
            &self,
            layers: Vec<([u32; 5], HashMap<String, bytes::Bytes>)>,
        ) -> io::Result<()> {
            self.inner.publish_imported_layers_shared(layers).await
        }

        async fn write_proof_state_bytes(&self, id: [u32; 5], bytes: &[u8]) -> io::Result<()> {
            self.inner.write_proof_state_bytes(id, bytes).await
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn real_slow_proof_loads_share_same_key_and_queue_different_keys() {
        let (inner, first_name, second_name) = memory_store_with_two_proof_layers().await;

        let slow = SlowProofStore::new(inner.clone());
        let same_key = CachedLayerStore::new_with_proof_cache_limits(
            slow.clone(),
            LockingHashMapLayerCache::new(),
            8,
            usize::MAX,
        );
        let first_store = same_key.clone();
        let first = tokio::spawn(async move { first_store.proof_layer(first_name).await });
        slow.wait_for_loads(1).await;
        let second_store = same_key.clone();
        let second = tokio::spawn(async move { second_store.proof_layer(first_name).await });
        for _ in 0..8 {
            tokio::task::yield_now().await;
        }
        assert_eq!(slow.loads.load(std::sync::atomic::Ordering::SeqCst), 1);
        slow.release.add_permits(1);
        let first = first.await.unwrap().unwrap().unwrap();
        let second = second.await.unwrap().unwrap().unwrap();
        assert!(Arc::ptr_eq(&first, &second));

        let queued = SlowProofStore::new(inner);
        let different_keys = CachedLayerStore::new_with_proof_cache_limits(
            queued.clone(),
            LockingHashMapLayerCache::new(),
            1,
            usize::MAX,
        );
        let first_store = different_keys.clone();
        let first = tokio::spawn(async move { first_store.proof_layer(first_name).await });
        queued.wait_for_loads(1).await;
        let second_store = different_keys.clone();
        let second = tokio::spawn(async move { second_store.proof_layer(second_name).await });
        for _ in 0..8 {
            tokio::task::yield_now().await;
        }
        assert_eq!(queued.loads.load(std::sync::atomic::Ordering::SeqCst), 1);
        queued.release.add_permits(1);
        first.await.unwrap().unwrap().unwrap();
        queued.wait_for_loads(2).await;
        queued.release.add_permits(1);
        second.await.unwrap().unwrap().unwrap();
        assert_eq!(queued.loads.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn proof_cache_evicts_lru_entry_at_entry_capacity() {
        let (inner, first_name, second_name) = memory_store_with_two_proof_layers().await;
        let store = CachedLayerStore::new_with_proof_cache_limits(
            inner,
            LockingHashMapLayerCache::new(),
            1,
            usize::MAX,
        );
        let first = store.proof_layer(first_name).await.unwrap().unwrap();
        store.proof_layer(second_name).await.unwrap().unwrap();
        assert_eq!(store.proof_cache.lock().unwrap().entries.len(), 1);
        let reloaded = store.proof_layer(first_name).await.unwrap().unwrap();
        assert!(!Arc::ptr_eq(&first, &reloaded));
    }

    #[tokio::test]
    async fn proof_cache_evicts_by_bytes_and_reports_an_upper_bound() {
        let (inner, first_name, second_name) = memory_store_with_two_proof_layers().await;
        let first_proof = inner.proof_layer(first_name).await.unwrap().unwrap();
        let second_proof = inner.proof_layer(second_name).await.unwrap().unwrap();
        let first_bytes = proof_cache_weight(&first_proof);
        let second_bytes = proof_cache_weight(&second_proof);
        let limit = first_bytes.max(second_bytes);
        let store = CachedLayerStore::new_with_proof_cache_limits(
            inner,
            LockingHashMapLayerCache::new(),
            8,
            limit,
        );
        store.proof_layer(first_name).await.unwrap().unwrap();
        store.proof_layer(second_name).await.unwrap().unwrap();
        let cache = store.proof_cache.lock().unwrap();
        assert!(cache.entries.len() <= 1);
        assert!(cache.bytes <= limit);
        let retained = cache.bytes;
        std::mem::drop(cache);
        assert_eq!(retained, LayerStore::proof_cache_memory_bytes(&store));
    }

    #[tokio::test]
    async fn oversized_proof_is_served_but_not_retained() {
        let (inner, first_name, _) = memory_store_with_two_proof_layers().await;
        let proof = inner.proof_layer(first_name).await.unwrap().unwrap();
        let store = CachedLayerStore::new_with_proof_cache_limits(
            inner,
            LockingHashMapLayerCache::new(),
            8,
            proof_cache_weight(&proof).saturating_sub(1),
        );
        assert_eq!(
            store.proof_layer(first_name).await.unwrap().unwrap().state,
            proof.state
        );
        let cache = store.proof_cache.lock().unwrap();
        assert_eq!(cache.entries.len(), 0);
        assert_eq!(cache.bytes, 0);
    }

    #[tokio::test]
    async fn zero_entry_proof_cache_preserves_single_call_but_retains_nothing() {
        let (inner, first_name, _) = memory_store_with_two_proof_layers().await;
        let store = CachedLayerStore::new_with_proof_cache_limits(
            inner,
            LockingHashMapLayerCache::new(),
            0,
            usize::MAX,
        );
        assert!(store.proof_layer(first_name).await.unwrap().is_some());
        assert_eq!(store.proof_cache.lock().unwrap().entries.len(), 0);
    }

    #[test]
    fn unresolved_proof_cells_are_not_evicted_and_loads_are_bounded() {
        let mut cache = ProofLayerCache::new(0, 0, 1);
        let unresolved = Arc::new(tokio::sync::OnceCell::new());
        let name = [7, 7, 7, 7, 7];
        cache.entries.put(name, unresolved.clone());
        cache.recompute_and_evict();
        assert!(cache
            .entries
            .peek(&name)
            .map_or(false, |cell| Arc::ptr_eq(cell, &unresolved)));

        let store = CachedLayerStore::new_with_proof_cache_limits(
            MemoryLayerStore::new(),
            LockingHashMapLayerCache::new(),
            100,
            usize::MAX,
        );
        let first = store.proof_loads.clone().try_acquire_owned().unwrap();
        let second = store.proof_loads.clone().try_acquire_owned().unwrap();
        assert!(store.proof_loads.clone().try_acquire_owned().is_err());
        drop((first, second));
    }

    #[tokio::test]
    async fn failed_proof_load_can_retry_after_storage_is_repaired() {
        let directory = tempdir().unwrap();
        let inner = DirectoryLayerStore::new(directory.path());
        let mut builder = inner.create_base_layer().await.unwrap();
        let name = builder.name();
        builder.add_value_triple(ValueTriple::new_node("a", "p", "b"));
        builder.commit_boxed().await.unwrap();
        inner
            .write_proof_state_bytes(name, b"invalid")
            .await
            .unwrap();
        let store = CachedLayerStore::new(inner.clone(), LockingHashMapLayerCache::new());
        assert!(store.proof_layer(name).await.is_err());

        let name_text = name_to_string(name);
        let proof_path = directory
            .path()
            .join(&name_text[..3])
            .join(&name_text)
            .join(crate::storage::consts::FILENAMES.proof_state);
        tokio::fs::remove_file(proof_path).await.unwrap();
        inner.prepare_proof_layer(name).await.unwrap();

        assert!(store.proof_layer(name).await.unwrap().is_some());
        assert_eq!(store.proof_cache.lock().unwrap().entries.len(), 1);
    }

    #[test]
    fn retrieve_layer_stack_names_retrieves_correctly() {
        //let store = CachedLayerStore::new(MemoryLayerStore::new());
        //let builder = store.create_base_layer().wait().unwrap();
    }
}
