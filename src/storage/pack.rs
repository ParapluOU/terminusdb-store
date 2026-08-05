use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use std::fmt::Display;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{Arc, RwLock};
use std::task::{Context, Poll};
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use futures::FutureExt;

use super::cache::*;
use super::consts::*;
use super::file::*;
use super::layer::*;

use crate::proof::ProofHash;
use bytes::Bytes;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use tar::*;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};

const PACK_MANIFEST_PATH: &str = ".terminus-store-pack-v2";
const PACK_MANIFEST: &[u8] = b"terminus-store-pack\nversion=2\nproof-state=pf4\n";
// Conservative ownership/bookkeeping allowances in the aggregate peak model.
// Entry backing bytes are charged separately at their exact declared size.
const PACK_LAYER_HEAP_OVERHEAD: usize = 512;
const PACK_ENTRY_HEAP_OVERHEAD: usize = 512;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackImportLimits {
    pub max_compressed_bytes: usize,
    pub max_layers: usize,
    pub max_entries: usize,
    pub max_entry_bytes: usize,
    pub max_uncompressed_bytes: usize,
    pub max_ancestry_layers: usize,
    pub max_ancestry_bytes: usize,
    /// Peak bytes retained by pack input, decoded requested entries, copied
    /// target ancestry, one proof decode, and publication conversion.
    pub max_peak_bytes: usize,
}

impl Default for PackImportLimits {
    fn default() -> Self {
        Self {
            max_compressed_bytes: 512 * 1024 * 1024,
            max_layers: 512,
            max_entries: 16_384,
            max_entry_bytes: crate::proof::MAX_PROOF_LAYER_BYTES,
            max_uncompressed_bytes: 512 * 1024 * 1024,
            max_ancestry_layers: 1024,
            max_ancestry_bytes: 512 * 1024 * 1024,
            max_peak_bytes: 1024 * 1024 * 1024,
        }
    }
}

#[async_trait]
pub trait Packable {
    /// Export the given layers by creating a pack, a Vec<u8> that can later be used with `import_layers` on a different store.
    async fn export_layers(
        &self,
        layer_ids: Box<dyn Iterator<Item = [u32; 5]> + Send>,
    ) -> io::Result<Vec<u8>>;

    /// Export a proof-capable pack under caller-negotiated limits. A pack
    /// emitted successfully with these limits is accepted by an importer using
    /// the same limits, subject to proof-root admission and the destination's
    /// separately bounded pre-existing ancestry fitting `max_ancestry_*` and
    /// the negotiated aggregate `max_peak_bytes`.
    async fn export_layers_with_limits(
        &self,
        layer_ids: Box<dyn Iterator<Item = [u32; 5]> + Send>,
        limits: PackImportLimits,
    ) -> io::Result<Vec<u8>>;

    /// Export a v1-compatible pack without authenticated proof sidecars.
    /// This route exists only for legacy consumers and is not proof-capable.
    async fn export_layers_legacy(
        &self,
        layer_ids: Box<dyn Iterator<Item = [u32; 5]> + Send>,
    ) -> io::Result<Vec<u8>>;

    /// Import the specified layers from the given pack, a byte slice that was previously generated with `export_layers`, on another store, and possibly even another machine).
    ///
    /// On success every requested layer is retrievable. Missing requested
    /// layers reject the entire import before any layer becomes visible.
    async fn import_layers(
        &self,
        pack: &[u8],
        layer_ids: Box<dyn Iterator<Item = [u32; 5]> + Send>,
    ) -> io::Result<()>;

    /// Import proof-enabled layers while anchoring every admitted proof state
    /// to a root obtained out of band from the pack producer.
    async fn import_layers_with_expected_roots(
        &self,
        pack: &[u8],
        layer_ids: Box<dyn Iterator<Item = [u32; 5]> + Send>,
        expected_roots: HashMap<[u32; 5], ProofHash>,
    ) -> io::Result<()>;

    /// Import with explicit parser and staging budgets. Supplying an empty root
    /// map keeps the operation legacy-only.
    async fn import_layers_with_limits(
        &self,
        pack: &[u8],
        layer_ids: Box<dyn Iterator<Item = [u32; 5]> + Send>,
        expected_roots: HashMap<[u32; 5], ProofHash>,
        limits: PackImportLimits,
    ) -> io::Result<()>;
}

#[async_trait]
impl<T: PersistentLayerStore> Packable for T {
    async fn export_layers(
        &self,
        layer_ids: Box<dyn Iterator<Item = [u32; 5]> + Send>,
    ) -> io::Result<Vec<u8>> {
        export_pack(self, layer_ids, true, PackImportLimits::default()).await
    }

    async fn export_layers_with_limits(
        &self,
        layer_ids: Box<dyn Iterator<Item = [u32; 5]> + Send>,
        limits: PackImportLimits,
    ) -> io::Result<Vec<u8>> {
        export_pack(self, layer_ids, true, limits).await
    }

    async fn export_layers_legacy(
        &self,
        layer_ids: Box<dyn Iterator<Item = [u32; 5]> + Send>,
    ) -> io::Result<Vec<u8>> {
        export_pack(self, layer_ids, false, PackImportLimits::default()).await
    }

    async fn import_layers(
        &self,
        pack: &[u8],
        layer_ids: Box<dyn Iterator<Item = [u32; 5]> + Send>,
    ) -> io::Result<()> {
        import_pack(
            self,
            pack,
            layer_ids,
            HashMap::new(),
            PackImportLimits::default(),
        )
        .await
    }

    async fn import_layers_with_expected_roots(
        &self,
        pack: &[u8],
        layer_ids: Box<dyn Iterator<Item = [u32; 5]> + Send>,
        expected_roots: HashMap<[u32; 5], ProofHash>,
    ) -> io::Result<()> {
        import_pack(
            self,
            pack,
            layer_ids,
            expected_roots,
            PackImportLimits::default(),
        )
        .await
    }

    async fn import_layers_with_limits(
        &self,
        pack: &[u8],
        layer_ids: Box<dyn Iterator<Item = [u32; 5]> + Send>,
        expected_roots: HashMap<[u32; 5], ProofHash>,
        limits: PackImportLimits,
    ) -> io::Result<()> {
        import_pack(self, pack, layer_ids, expected_roots, limits).await
    }
}

fn collect_layer_ids(
    layer_ids: Box<dyn Iterator<Item = [u32; 5]> + Send>,
    max_layers: usize,
) -> io::Result<Vec<[u32; 5]>> {
    let mut result = Vec::with_capacity(max_layers.min(1024));
    for id in layer_ids {
        if result.len() == max_layers {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "too many layers requested for one pack operation",
            ));
        }
        result.push(id);
    }
    Ok(result)
}

struct LimitedVecWriter {
    bytes: Vec<u8>,
    max_bytes: usize,
    written: Arc<std::sync::atomic::AtomicUsize>,
}

impl LimitedVecWriter {
    fn new(max_bytes: usize) -> (Self, Arc<std::sync::atomic::AtomicUsize>) {
        let written = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        (
            Self {
                bytes: Vec::new(),
                max_bytes,
                written: written.clone(),
            },
            written,
        )
    }

    fn into_inner(self) -> Vec<u8> {
        self.bytes
    }
}

impl Write for LimitedVecWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let next = self
            .bytes
            .len()
            .checked_add(bytes.len())
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "pack size overflow"))?;
        if next > self.max_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "compressed layer pack exceeds negotiated limit",
            ));
        }
        self.bytes
            .try_reserve(bytes.len())
            .map_err(io::Error::other)?;
        self.bytes.extend_from_slice(bytes);
        self.written
            .store(next, std::sync::atomic::Ordering::Relaxed);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

struct PeakBudget {
    retained: usize,
    transient: usize,
    max: usize,
}

impl PeakBudget {
    fn new(initial: usize, max: usize) -> io::Result<Self> {
        if initial > max {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "pack peak-memory budget exceeded",
            ));
        }
        Ok(Self {
            retained: initial,
            transient: 0,
            max,
        })
    }

    fn retain(&mut self, bytes: usize) -> io::Result<()> {
        let retained = self
            .retained
            .checked_add(bytes)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "pack peak overflow"))?;
        if retained
            .checked_add(self.transient)
            .map_or(true, |peak| peak > self.max)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "pack peak-memory budget exceeded",
            ));
        }
        self.retained = retained;
        Ok(())
    }

    fn set_transient(&mut self, bytes: usize) -> io::Result<()> {
        if self
            .retained
            .checked_add(bytes)
            .map_or(true, |peak| peak > self.max)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "pack peak-memory budget exceeded",
            ));
        }
        self.transient = bytes;
        Ok(())
    }
}

async fn export_pack<S: PersistentLayerStore>(
    store: &S,
    layer_ids: Box<dyn Iterator<Item = [u32; 5]> + Send>,
    proof_capable: bool,
    limits: PackImportLimits,
) -> io::Result<Vec<u8>> {
    let mtime = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| io::Error::new(io::ErrorKind::Other, error))?
        .as_secs();
    let layer_ids = collect_layer_ids(layer_ids, limits.max_layers)?;

    let (writer, compressed_bytes) = LimitedVecWriter::new(limits.max_compressed_bytes);
    let mut budget = PackBudget::new(limits, compressed_bytes);
    let mut enc = GzEncoder::new(writer, Compression::default());
    {
        let mut tar = tar::Builder::new(&mut enc);
        if proof_capable {
            budget.add_entry(PACK_MANIFEST.len())?;
            let mut header = Header::new_gnu();
            header.set_mode(0o644);
            header.set_size(PACK_MANIFEST.len() as u64);
            header.set_mtime(mtime);
            header.set_cksum();
            tar.append_data(
                &mut header,
                PACK_MANIFEST_PATH,
                io::Cursor::new(PACK_MANIFEST),
            )?;
        }
        for id in layer_ids {
            tar_append_layer(&mut tar, store, id, mtime, proof_capable, &mut budget).await?;
        }
        tar.finish()?;
    }
    Ok(enc.finish()?.into_inner())
}

struct PackBudget {
    entries: usize,
    bytes: usize,
    limits: PackImportLimits,
    compressed_bytes: Arc<std::sync::atomic::AtomicUsize>,
}

impl PackBudget {
    fn new(
        limits: PackImportLimits,
        compressed_bytes: Arc<std::sync::atomic::AtomicUsize>,
    ) -> Self {
        Self {
            entries: 0,
            bytes: 0,
            limits,
            compressed_bytes,
        }
    }

    fn add_entry(&mut self, bytes: usize) -> io::Result<()> {
        self.entries = self
            .entries
            .checked_add(1)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "pack entry overflow"))?;
        if self.entries > self.limits.max_entries || bytes > self.limits.max_entry_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "exported layer pack exceeds negotiated entry limit",
            ));
        }
        self.bytes = self
            .bytes
            .checked_add(bytes)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "pack size overflow"))?;
        if self.bytes > self.limits.max_uncompressed_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "exported layer pack exceeds negotiated uncompressed limit",
            ));
        }
        Ok(())
    }

    fn before_map(&self, bytes: usize) -> io::Result<()> {
        let compressed = self
            .compressed_bytes
            .load(std::sync::atomic::Ordering::Relaxed);
        if compressed
            .checked_add(bytes)
            .map_or(true, |peak| peak > self.limits.max_peak_bytes)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "export pack peak-memory budget exceeded",
            ));
        }
        Ok(())
    }
}

fn decode_pack(
    pack: &[u8],
    requested: &HashSet<[u32; 5]>,
    limits: PackImportLimits,
    peak: &mut PeakBudget,
) -> io::Result<(bool, HashMap<[u32; 5], HashMap<String, Bytes>>)> {
    if pack.len() > limits.max_compressed_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "compressed layer pack exceeds limit",
        ));
    }
    let cursor = io::Cursor::new(pack);
    let tar = GzDecoder::new(cursor);
    let mut archive = Archive::new(tar);
    let mut result: HashMap<[u32; 5], HashMap<String, Bytes>> = HashMap::new();
    let mut manifest_seen = false;
    let mut entry_count = 0usize;
    let mut total_bytes = 0usize;

    for item in archive.entries()? {
        entry_count = entry_count.checked_add(1).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "pack entry count overflow")
        })?;
        if entry_count > limits.max_entries {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "layer pack contains too many entries",
            ));
        }
        let mut entry = item?;
        let path = entry.path()?.into_owned();
        let components: Vec<_> = path.iter().collect();
        if components.len() == 1 && components[0].to_str() == Some(PACK_MANIFEST_PATH) {
            if manifest_seen || !entry.header().entry_type().is_file() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid or duplicate pack manifest",
                ));
            }
            let size = checked_pack_entry_size(entry.header().size()?, &mut total_bytes, limits)?;
            peak.set_transient(size)?;
            let mut bytes = vec![0; size];
            entry.read_exact(&mut bytes)?;
            peak.set_transient(0)?;
            if bytes != PACK_MANIFEST {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "unsupported layer pack manifest",
                ));
            }
            manifest_seen = true;
            continue;
        }
        if components.is_empty() || components.len() > 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid layer pack path",
            ));
        }
        let layer_component = components[0].to_str().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "non-utf8 layer pack path")
        })?;
        let layer_id = string_to_name(layer_component)?;
        if components.len() == 1 {
            if !entry.header().entry_type().is_dir() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "layer pack top-level entry is not a directory",
                ));
            }
            continue;
        }
        if !entry.header().entry_type().is_file() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "layer pack contains a non-regular structure",
            ));
        }
        let size = checked_pack_entry_size(entry.header().size()?, &mut total_bytes, limits)?;
        if !requested.contains(&layer_id) {
            io::copy(&mut entry, &mut io::sink())?;
            continue;
        }
        let file_name = components[1]
            .to_str()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "non-utf8 pack filename"))?;
        if !FILENAME_ENUM_MAP.contains_key(file_name) || file_name == FILENAMES.rollup {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "pack contains an unknown or non-portable layer structure",
            ));
        }
        peak.retain(size.checked_add(PACK_ENTRY_HEAP_OVERHEAD).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "pack entry heap overflow")
        })?)?;
        let mut bytes = vec![0; size];
        entry.read_exact(&mut bytes)?;
        if entry.read(&mut [0; 1])? != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "layer pack entry exceeds its declared size",
            ));
        }
        if result
            .entry(layer_id)
            .or_default()
            .insert(file_name.to_owned(), Bytes::from(bytes))
            .is_some()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "pack contains a duplicate layer structure",
            ));
        }
    }
    Ok((manifest_seen, result))
}

fn checked_pack_entry_size(
    raw: u64,
    total: &mut usize,
    limits: PackImportLimits,
) -> io::Result<usize> {
    let size = usize::try_from(raw)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "pack entry is too large"))?;
    if size > limits.max_entry_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "layer pack entry exceeds limit",
        ));
    }
    *total = total
        .checked_add(size)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "pack size overflow"))?;
    if *total > limits.max_uncompressed_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "uncompressed layer pack exceeds limit",
        ));
    }
    Ok(size)
}

fn parent_from_files(files: &HashMap<String, Bytes>) -> io::Result<Option<[u32; 5]>> {
    match files.get(FILENAMES.parent) {
        None => Ok(None),
        Some(bytes) => {
            if bytes.len() != 40 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "imported parent id is not canonical",
                ));
            }
            let parent = std::str::from_utf8(&bytes[..40])
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
            Ok(Some(string_to_name(parent)?))
        }
    }
}

fn validate_layer_file_set(files: &HashMap<String, Bytes>) -> io::Result<()> {
    let required =
        SHARED_REQUIRED_FILES
            .iter()
            .copied()
            .chain(if files.contains_key(FILENAMES.parent) {
                CHILD_LAYER_REQUIRED_FILES.iter().copied()
            } else {
                BASE_LAYER_REQUIRED_FILES.iter().copied()
            });
    if let Some(missing) = required.into_iter().find(|name| !files.contains_key(*name)) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("imported layer is missing required structure {missing}"),
        ));
    }
    Ok(())
}

async fn load_staged_layer_without_panicking(
    staging: &PackValidationStore,
    id: [u32; 5],
) -> io::Result<()> {
    std::panic::AssertUnwindSafe(staging.get_layer(id))
        .catch_unwind()
        .await
        .map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "malformed imported succinct structure",
            )
        })??
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "incomplete imported layer"))?;
    Ok(())
}

async fn validate_staged_proof_without_panicking(
    staging: &PackValidationStore,
    id: [u32; 5],
) -> io::Result<crate::proof::ProofLayerState> {
    std::panic::AssertUnwindSafe(staging.validate_proof_chain_against_store(id))
        .catch_unwind()
        .await
        .map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "malformed imported proof or native layer structure",
            )
        })?
}

async fn copy_layer_files<S: PersistentLayerStore>(
    store: &S,
    id: [u32; 5],
    ancestry_bytes: &mut usize,
    max_ancestry_bytes: usize,
    peak: &mut PeakBudget,
) -> io::Result<HashMap<String, Bytes>> {
    if !store.directory_exists(id).await? {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "imported layer parent is unavailable",
        ));
    }
    let mut files = HashMap::new();
    for name in FILENAME_ENUM_MAP.keys().copied() {
        if name == FILENAMES.rollup {
            continue;
        }
        if store.file_exists(id, name).await? {
            let file = store.get_file(id, name).await?;
            let size = file.size().await?;
            let next = ancestry_bytes.checked_add(size).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "ancestry byte count overflow")
            })?;
            if next > max_ancestry_bytes {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "imported layer ancestry exceeds byte limit",
                ));
            }
            *ancestry_bytes = next;
            peak.retain(size.checked_add(PACK_ENTRY_HEAP_OVERHEAD).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "ancestry entry heap overflow")
            })?)?;
            let bytes = file.map().await?;
            if bytes.len() != size {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "ancestry layer changed while it was copied",
                ));
            }
            files.insert(name.to_owned(), bytes);
        }
    }
    Ok(files)
}

#[derive(Clone)]
struct PackFile(Option<Bytes>);

struct PackFileReader {
    bytes: Bytes,
    position: usize,
}

impl AsyncRead for PackFileReader {
    fn poll_read(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let remaining = &self.bytes[self.position..];
        let length = remaining.len().min(buffer.remaining());
        buffer.put_slice(&remaining[..length]);
        self.position += length;
        Poll::Ready(Ok(()))
    }
}

struct UnsupportedPackWriter;

impl AsyncWrite for UnsupportedPackWriter {
    fn poll_write(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        _bytes: &[u8],
    ) -> Poll<io::Result<usize>> {
        Poll::Ready(Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "pack validation store is read-only",
        )))
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "pack validation store is read-only",
        )))
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        self.poll_flush(cx)
    }
}

#[async_trait]
impl SyncableFile for UnsupportedPackWriter {
    async fn sync_all(self) -> io::Result<()> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "pack validation store is read-only",
        ))
    }
}

#[async_trait]
impl FileStore for PackFile {
    type Write = UnsupportedPackWriter;

    async fn open_write(&self) -> io::Result<Self::Write> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "pack validation store is read-only",
        ))
    }
}

#[async_trait]
impl FileLoad for PackFile {
    type Read = PackFileReader;

    async fn exists(&self) -> io::Result<bool> {
        Ok(self.0.is_some())
    }

    async fn size(&self) -> io::Result<usize> {
        Ok(self.0.as_ref().map_or(0, Bytes::len))
    }

    async fn open_read_from(&self, offset: usize) -> io::Result<Self::Read> {
        let bytes = self.0.clone().unwrap_or_default();
        if offset > bytes.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "pack file read offset exceeds entry",
            ));
        }
        Ok(PackFileReader {
            bytes,
            position: offset,
        })
    }

    async fn map(&self) -> io::Result<Bytes> {
        Ok(self.0.clone().unwrap_or_default())
    }
}

type PackLayerFiles = HashMap<[u32; 5], HashMap<String, Bytes>>;

#[derive(Clone)]
struct PackValidationStore {
    layers: Arc<RwLock<PackLayerFiles>>,
}

impl PackValidationStore {
    fn new(layers: PackLayerFiles) -> Self {
        Self {
            layers: Arc::new(RwLock::new(layers)),
        }
    }

    fn insert_layer(&self, id: [u32; 5], files: HashMap<String, Bytes>) -> io::Result<()> {
        if self.layers.write().unwrap().insert(id, files).is_some() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "duplicate pack validation layer",
            ));
        }
        Ok(())
    }

    fn into_layers(self) -> io::Result<PackLayerFiles> {
        Arc::try_unwrap(self.layers)
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "pack validation handles remain"))?
            .into_inner()
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "pack validation lock poisoned"))
    }
}

#[async_trait]
impl PersistentLayerStore for PackValidationStore {
    type File = PackFile;

    async fn directories(&self) -> io::Result<Vec<[u32; 5]>> {
        Ok(self.layers.read().unwrap().keys().copied().collect())
    }

    async fn create_named_directory(&self, _id: [u32; 5]) -> io::Result<[u32; 5]> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "pack validation store is read-only",
        ))
    }

    async fn directory_exists(&self, id: [u32; 5]) -> io::Result<bool> {
        Ok(self.layers.read().unwrap().contains_key(&id))
    }

    async fn get_file(&self, id: [u32; 5], name: &str) -> io::Result<Self::File> {
        let guard = self.layers.read().unwrap();
        let files = guard
            .get(&id)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "pack layer not found"))?;
        // Layer loading asks for handles to optional structures and then calls
        // `exists` on them. Preserve that contract without mutating staging.
        Ok(PackFile(files.get(name).cloned()))
    }

    async fn file_exists(&self, id: [u32; 5], name: &str) -> io::Result<bool> {
        Ok(self
            .layers
            .read()
            .unwrap()
            .get(&id)
            .map_or(false, |files| files.contains_key(name)))
    }

    async fn abort_directory(&self, _id: [u32; 5]) -> io::Result<()> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "pack validation store is read-only",
        ))
    }

    async fn publish_imported_layers_shared(
        &self,
        _layers: Vec<([u32; 5], HashMap<String, Bytes>)>,
    ) -> io::Result<()> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "pack validation store is read-only",
        ))
    }

    async fn write_proof_state_bytes(&self, _directory: [u32; 5], _bytes: &[u8]) -> io::Result<()> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "pack validation store is read-only",
        ))
    }
}

async fn import_pack<S: PersistentLayerStore>(
    store: &S,
    pack: &[u8],
    layer_ids: Box<dyn Iterator<Item = [u32; 5]> + Send>,
    expected_roots: HashMap<[u32; 5], ProofHash>,
    limits: PackImportLimits,
) -> io::Result<()> {
    if pack.len() > limits.max_compressed_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "compressed layer pack exceeds limit",
        ));
    }
    let mut peak = PeakBudget::new(pack.len(), limits.max_peak_bytes)?;
    let layer_ids = collect_layer_ids(layer_ids, limits.max_layers)?;
    peak.retain(
        layer_ids
            .len()
            .checked_mul(PACK_LAYER_HEAP_OVERHEAD)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "layer heap overflow"))?,
    )?;
    let requested: HashSet<_> = layer_ids.iter().copied().collect();
    if requested.len() != layer_ids.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "duplicate requested layer id",
        ));
    }
    if expected_roots.keys().any(|id| !requested.contains(id)) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "expected proof root supplied for an unrequested layer",
        ));
    }
    for id in &layer_ids {
        if store.directory_exists(*id).await? {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "refusing to import over an existing layer",
            ));
        }
    }

    // Decode directly from the caller's bounded slice. Copying it into a
    // `'static` blocking task would double the largest attacker-controlled
    // allocation before any uncompressed-entry limit could take effect.
    let (v2, imported) = decode_pack(pack, &requested, limits, &mut peak)?;
    if imported.len() != requested.len() || layer_ids.iter().any(|id| !imported.contains_key(id)) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "pack does not contain every requested layer",
        ));
    }
    for files in imported.values() {
        validate_layer_file_set(files)?;
    }

    let mut pending: Vec<_> = imported
        .values()
        .map(parent_from_files)
        .collect::<io::Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect();
    let staging = PackValidationStore::new(imported);
    let mut staged_ids = requested.clone();
    let mut ancestry_bytes = 0usize;
    let mut ancestry_layers = 0usize;
    while let Some(parent) = pending.pop() {
        if staged_ids.contains(&parent) {
            continue;
        }
        ancestry_layers = ancestry_layers.checked_add(1).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "ancestry layer count overflow")
        })?;
        if ancestry_layers > limits.max_ancestry_layers {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "imported layer ancestry exceeds limit",
            ));
        }
        peak.retain(PACK_LAYER_HEAP_OVERHEAD)?;
        let files = copy_layer_files(
            store,
            parent,
            &mut ancestry_bytes,
            limits.max_ancestry_bytes,
            &mut peak,
        )
        .await?;
        validate_layer_file_set(&files)?;
        if let Some(grandparent) = parent_from_files(&files)? {
            pending.push(grandparent);
        }
        staging.insert_layer(parent, files)?;
        staged_ids.insert(parent);
    }

    let mut saw_proof = false;
    for id in &layer_ids {
        load_staged_layer_without_panicking(&staging, *id).await?;
        if staging.file_exists(*id, FILENAMES.proof_state).await? {
            saw_proof = true;
            if !v2 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "proof state requires a versioned proof-capable pack",
                ));
            }
            let expected = expected_roots.get(id).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "proof import requires an out-of-band expected state root",
                )
            })?;
            // Proof decoding has its own hard heap limit. Reserve that entire
            // allowance against the aggregate pack peak before validation.
            peak.set_transient(crate::proof::MAX_DECODED_PROOF_BYTES)?;
            let validated = validate_staged_proof_without_panicking(&staging, *id).await?;
            peak.set_transient(0)?;
            if &validated.state_root != expected {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "imported proof state does not match expected root",
                ));
            }
        } else if expected_roots.contains_key(id) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "expected proof root supplied for a legacy layer",
            ));
        }
    }
    if !saw_proof && !expected_roots.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "expected proof roots were not consumed",
        ));
    }

    let mut imported = staging.into_layers()?;
    let publish = layer_ids
        .into_iter()
        .map(|id| {
            imported
                .remove(&id)
                .map(|files| (id, files))
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "requested layer disappeared")
                })
        })
        .collect::<io::Result<Vec<_>>>()?;
    let publication_bytes = publish.iter().try_fold(0usize, |total, (_, files)| {
        let contents = files.values().try_fold(0usize, |layer_total, bytes| {
            layer_total.checked_add(bytes.len()).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "publication size overflow")
            })
        })?;
        // Archive publication builds one encoded aggregate per layer. This
        // conservative allowance covers its presence header, offset table,
        // and allocator rounding in addition to the file contents.
        let overhead = files
            .len()
            .checked_mul(32)
            .and_then(|value| value.checked_add(4096))
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "publication size overflow")
            })?;
        total
            .checked_add(contents)
            .and_then(|value| value.checked_add(overhead))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "publication size overflow"))
    })?;
    peak.set_transient(publication_bytes)?;
    // Pre-existing ancestry was copied solely for independent validation and
    // must never be republished with the requested layers.
    drop(imported);
    store.publish_imported_layers_shared(publish).await
}

async fn tar_append_file<S: PersistentLayerStore, W: io::Write>(
    store: &S,
    tar: &mut tar::Builder<W>,
    layer: [u32; 5],
    layer_path: &PathBuf,
    file_name: &str,
    mtime: u64,
    budget: &mut PackBudget,
) -> io::Result<()> {
    if store.file_exists(layer, file_name).await? {
        let file = store.get_file(layer, file_name).await?;
        let size = file.size().await?;
        budget.add_entry(size)?;
        budget.before_map(size)?;
        let contents = file.map().await?;
        if contents.len() != size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "layer file changed while it was exported",
            ));
        }
        let cursor = io::Cursor::new(&contents);

        let path = layer_path.join(file_name);

        let mut header = Header::new_gnu();
        header.set_mode(0o644);
        header.set_size(
            u64::try_from(contents.len()).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidData, "pack file is too large")
            })?,
        );
        header.set_mtime(mtime);
        header.set_cksum();
        tar.append_data(&mut header, path, cursor)?;

        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::NotFound,
            "file does not exist",
        ))
    }
}

async fn tar_append_file_if_exists<S: PersistentLayerStore, W: io::Write>(
    store: &S,
    tar: &mut tar::Builder<W>,
    layer: [u32; 5],
    layer_path: &PathBuf,
    file_name: &str,
    mtime: u64,
    budget: &mut PackBudget,
) -> io::Result<()> {
    if store.file_exists(layer, file_name).await? {
        let file = store.get_file(layer, file_name).await?;
        let size = file.size().await?;
        budget.add_entry(size)?;
        budget.before_map(size)?;
        let contents = file.map().await?;
        if contents.len() != size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "layer file changed while it was exported",
            ));
        }
        let cursor = io::Cursor::new(&contents);

        let path = layer_path.join(file_name);

        let mut header = Header::new_gnu();
        header.set_mode(0o644);
        header.set_size(
            u64::try_from(contents.len()).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidData, "pack file is too large")
            })?,
        );
        header.set_mtime(mtime);
        header.set_cksum();
        tar.append_data(&mut header, path, cursor)?;
    }

    Ok(())
}

async fn tar_append_layer<W: io::Write, S: PersistentLayerStore>(
    tar: &mut tar::Builder<W>,
    store: &S,
    layer: [u32; 5],
    mtime: u64,
    include_proof_state: bool,
    budget: &mut PackBudget,
) -> io::Result<()> {
    budget.add_entry(0)?;
    let mut header = Header::new_gnu();
    header.set_mode(0o755);
    header.set_entry_type(EntryType::Directory);
    header.set_mtime(mtime);
    header.set_size(0);
    header.set_cksum();
    let layer_name = name_to_string(layer);
    let mut path = PathBuf::new();
    path.push(layer_name);
    tar.append_data(&mut header, &path, std::io::empty())?;

    for f in &SHARED_REQUIRED_FILES {
        tar_append_file(store, tar, layer, &path, f, mtime, budget).await?;
    }
    for f in &SHARED_OPTIONAL_FILES {
        if f == &FILENAMES.rollup {
            // skip the rollup file. It will not be resolvable remotely.
            continue;
        }
        if f == &FILENAMES.proof_state && !include_proof_state {
            continue;
        }
        tar_append_file_if_exists(store, tar, layer, &path, f, mtime, budget).await?;
    }
    if store.file_exists(layer, FILENAMES.parent).await? {
        // this is a child layer
        for f in &CHILD_LAYER_REQUIRED_FILES {
            tar_append_file(store, tar, layer, &path, f, mtime, budget).await?;
        }
        for f in &CHILD_LAYER_OPTIONAL_FILES {
            tar_append_file_if_exists(store, tar, layer, &path, f, mtime, budget).await?;
        }
    } else {
        // this is a base layer
        for f in &BASE_LAYER_REQUIRED_FILES {
            tar_append_file(store, tar, layer, &path, f, mtime, budget).await?;
        }
        for f in &BASE_LAYER_OPTIONAL_FILES {
            tar_append_file_if_exists(store, tar, layer, &path, f, mtime, budget).await?;
        }
    }

    Ok(())
}

#[derive(Debug)]
pub enum PackError {
    LayerNotFound,
    Io(io::Error),
    Utf8Error(std::str::Utf8Error),
}

impl Display for PackError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(formatter, "{:?}", self)
    }
}

impl From<io::Error> for PackError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}
impl From<std::str::Utf8Error> for PackError {
    fn from(err: std::str::Utf8Error) -> Self {
        Self::Utf8Error(err)
    }
}

pub fn pack_layer_parents<R: io::Read>(
    readable: R,
) -> Result<HashMap<[u32; 5], Option<[u32; 5]>>, PackError> {
    let tar = GzDecoder::new(readable);
    let mut archive = Archive::new(tar);

    // build a set out of the layer ids for easy retrieval
    let mut result_map = HashMap::new();
    let limits = PackImportLimits::default();
    let mut entry_count = 0usize;
    let mut total_bytes = 0usize;

    for e in archive.entries()? {
        entry_count = entry_count
            .checked_add(1)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "pack entry overflow"))?;
        if entry_count > limits.max_entries {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "layer pack contains too many entries",
            )
            .into());
        }
        let mut entry = e?;
        let path = entry.path()?;
        checked_pack_entry_size(entry.header().size()?, &mut total_bytes, limits)?;

        if path.as_ref() == std::path::Path::new(PACK_MANIFEST_PATH) {
            continue;
        }

        let mut components = path.iter();
        let layer_name = components
            .next()
            .and_then(|component| component.to_str())
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid pack layer path"))?;
        if components.next().is_some() && components.next().is_some() {
            return Err(
                io::Error::new(io::ErrorKind::InvalidData, "nested pack layer path").into(),
            );
        }
        let id = string_to_name(layer_name)?;

        if path.file_name().and_then(|name| name.to_str()) == Some(FILENAMES.parent) {
            // this is an element we want to know the parent of
            // lets read it
            let mut parent_id_bytes = [0u8; 40];
            entry.read_exact(&mut parent_id_bytes)?;
            let parent_id_str = std::str::from_utf8(&parent_id_bytes)?;
            let parent_id = string_to_name(parent_id_str)?;

            result_map.insert(id, Some(parent_id));
        } else {
            // Ensure that an entry for this layer exists
            // If we encounter the parent file later on, this'll be overwritten with the parent id.
            // If not, it can be assumed to not have a parent.
            result_map.entry(id).or_insert(None);
        }
    }

    Ok(result_map)
}

#[async_trait]
impl Packable for CachedLayerStore {
    async fn export_layers(
        &self,
        layer_ids: Box<dyn Iterator<Item = [u32; 5]> + Send>,
    ) -> io::Result<Vec<u8>> {
        self.inner.export_layers(layer_ids).await
    }

    async fn export_layers_with_limits(
        &self,
        layer_ids: Box<dyn Iterator<Item = [u32; 5]> + Send>,
        limits: PackImportLimits,
    ) -> io::Result<Vec<u8>> {
        self.inner
            .export_layers_with_limits(layer_ids, limits)
            .await
    }

    async fn export_layers_legacy(
        &self,
        layer_ids: Box<dyn Iterator<Item = [u32; 5]> + Send>,
    ) -> io::Result<Vec<u8>> {
        self.inner.export_layers_legacy(layer_ids).await
    }

    async fn import_layers(
        &self,
        pack: &[u8],
        layer_ids: Box<dyn Iterator<Item = [u32; 5]> + Send>,
    ) -> io::Result<()> {
        self.inner.import_layers(pack, layer_ids).await
    }

    async fn import_layers_with_expected_roots(
        &self,
        pack: &[u8],
        layer_ids: Box<dyn Iterator<Item = [u32; 5]> + Send>,
        expected_roots: HashMap<[u32; 5], ProofHash>,
    ) -> io::Result<()> {
        self.inner
            .import_layers_with_expected_roots(pack, layer_ids, expected_roots)
            .await
    }

    async fn import_layers_with_limits(
        &self,
        pack: &[u8],
        layer_ids: Box<dyn Iterator<Item = [u32; 5]> + Send>,
        expected_roots: HashMap<[u32; 5], ProofHash>,
        limits: PackImportLimits,
    ) -> io::Result<()> {
        self.inner
            .import_layers_with_limits(pack, layer_ids, expected_roots, limits)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::*;
    use crate::storage::directory::*;
    use crate::store::{open_archive_store, open_memory_store};
    use std::io::Cursor;
    use std::sync::Arc;
    use tempfile::tempdir;

    fn remove_pack_entry(pack: &[u8], target: &str) -> Vec<u8> {
        let mut source = Archive::new(GzDecoder::new(Cursor::new(pack)));
        let encoder = GzEncoder::new(Vec::new(), Compression::default());
        let mut destination = tar::Builder::new(encoder);
        for item in source.entries().unwrap() {
            let mut entry = item.unwrap();
            let path = entry.path().unwrap().into_owned();
            if path.to_string_lossy() == target {
                continue;
            }
            let mut bytes = Vec::new();
            entry.read_to_end(&mut bytes).unwrap();
            let mut header = entry.header().clone();
            header.set_size(bytes.len() as u64);
            header.set_cksum();
            destination
                .append_data(&mut header, path, Cursor::new(bytes))
                .unwrap();
        }
        let encoder = destination.into_inner().unwrap();
        encoder.finish().unwrap()
    }

    fn replace_pack_entry(pack: &[u8], target: &str, replacement: &[u8]) -> Vec<u8> {
        let mut source = Archive::new(GzDecoder::new(Cursor::new(pack)));
        let encoder = GzEncoder::new(Vec::new(), Compression::default());
        let mut destination = tar::Builder::new(encoder);
        for item in source.entries().unwrap() {
            let mut entry = item.unwrap();
            let path = entry.path().unwrap().into_owned();
            let mut bytes = Vec::new();
            entry.read_to_end(&mut bytes).unwrap();
            if path.to_string_lossy() == target {
                bytes.clear();
                bytes.extend_from_slice(replacement);
            }
            let mut header = entry.header().clone();
            header.set_size(bytes.len() as u64);
            header.set_cksum();
            destination
                .append_data(&mut header, path, Cursor::new(bytes))
                .unwrap();
        }
        let encoder = destination.into_inner().unwrap();
        encoder.finish().unwrap()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn export_import_layer_with_rollup() {
        let dir1 = tempdir().unwrap();
        let store1 = Arc::new(DirectoryLayerStore::new(dir1.path()));
        let dir2 = tempdir().unwrap();
        let store2 = Arc::new(DirectoryLayerStore::new(dir2.path()));

        let mut builder = store1.create_base_layer().await.unwrap();
        let base_name = builder.name();

        builder.add_value_triple(ValueTriple::new_node("cow", "likes", "duck"));
        builder.add_value_triple(ValueTriple::new_node("duck", "hates", "cow"));

        builder.commit_boxed().await.unwrap();

        let mut builder = store1.create_child_layer(base_name).await.unwrap();
        let child_name = builder.name();

        builder.remove_value_triple(ValueTriple::new_node("duck", "hates", "cow"));
        builder.add_value_triple(ValueTriple::new_node("duck", "likes", "cow"));

        builder.commit_boxed().await.unwrap();

        let unrolled_layer = store1.get_layer(child_name).await.unwrap().unwrap();

        store1.clone().rollup(unrolled_layer).await.unwrap();

        let export = store1
            .export_layers(Box::new(vec![base_name, child_name].into_iter()))
            .await
            .unwrap();

        store2
            .import_layers(&export, Box::new(vec![base_name, child_name].into_iter()))
            .await
            .unwrap();

        let imported_layer = store2.get_layer(child_name).await.unwrap().unwrap();
        let triples: Vec<_> = imported_layer
            .triples()
            .map(|t| imported_layer.id_triple_to_string(&t).unwrap())
            .collect();
        assert_eq!(
            vec![
                ValueTriple::new_node("cow", "likes", "duck"),
                ValueTriple::new_node("duck", "likes", "cow")
            ],
            triples
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn duplicate_request_and_incomplete_pack_publish_nothing() {
        let source = open_memory_store();
        let base_builder = source.create_base_layer().await.unwrap();
        base_builder
            .add_value_triple(ValueTriple::new_node("a", "p", "b"))
            .unwrap();
        let base = base_builder.commit().await.unwrap();
        let child_builder = base.open_write().await.unwrap();
        child_builder
            .add_value_triple(ValueTriple::new_node("b", "p", "c"))
            .unwrap();
        let child = child_builder.commit().await.unwrap();
        let ids = vec![base.name(), child.name()];
        let pack = source
            .export_layers(Box::new(ids.clone().into_iter()))
            .await
            .unwrap();

        let duplicate_target = open_memory_store();
        assert!(duplicate_target
            .import_layers(&pack, Box::new(vec![base.name(), base.name()].into_iter()),)
            .await
            .is_err());
        assert!(duplicate_target
            .get_layer_from_id(base.name())
            .await
            .unwrap()
            .is_none());

        let missing_path = format!(
            "{}/{}",
            name_to_string(child.name()),
            FILENAMES.pos_s_p_adjacency_list_bits
        );
        let incomplete = remove_pack_entry(&pack, &missing_path);
        let target_dir = tempdir().unwrap();
        let target = crate::store::open_directory_store(target_dir.path());
        assert!(target
            .import_layers(&incomplete, Box::new(ids.clone().into_iter()))
            .await
            .is_err());
        for id in ids {
            assert!(target.get_layer_from_id(id).await.unwrap().is_none());
        }

        let archive_dir = tempdir().unwrap();
        let archive = open_archive_store(archive_dir.path(), 8);
        let archive_ids = vec![base.name(), child.name()];
        assert!(archive
            .import_layers(&incomplete, Box::new(archive_ids.clone().into_iter()))
            .await
            .is_err());
        for id in archive_ids {
            assert!(archive.get_layer_from_id(id).await.unwrap().is_none());
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn proof_pack_requires_matching_external_root_and_legacy_route_drops_proof() {
        let source = open_memory_store();
        let builder = source.create_base_layer().await.unwrap();
        builder
            .add_value_triple(ValueTriple::new_node("a", "p", "b"))
            .unwrap();
        let (layer, state) = builder.commit_with_proof().await.unwrap();
        let id = layer.name();
        let proof_pack = source
            .export_layers(Box::new(vec![id].into_iter()))
            .await
            .unwrap();

        let mismatch = open_memory_store();
        assert!(mismatch
            .import_layers_with_expected_roots(
                &proof_pack,
                Box::new(vec![id].into_iter()),
                HashMap::from([(id, [0x55; 32])]),
            )
            .await
            .is_err());
        assert!(mismatch.get_layer_from_id(id).await.unwrap().is_none());

        let anchored = open_memory_store();
        anchored
            .import_layers_with_expected_roots(
                &proof_pack,
                Box::new(vec![id].into_iter()),
                HashMap::from([(id, state.state_root)]),
            )
            .await
            .unwrap();
        assert_eq!(
            anchored
                .get_layer_from_id(id)
                .await
                .unwrap()
                .unwrap()
                .proof_state()
                .await
                .unwrap()
                .unwrap()
                .state_root,
            state.state_root
        );

        let legacy_pack = source
            .export_layers_legacy(Box::new(vec![id].into_iter()))
            .await
            .unwrap();
        let legacy = open_memory_store();
        legacy
            .import_layers(&legacy_pack, Box::new(vec![id].into_iter()))
            .await
            .unwrap();
        assert!(legacy
            .get_layer_from_id(id)
            .await
            .unwrap()
            .unwrap()
            .proof_state()
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn archive_same_id_import_race_has_one_atomic_winner() {
        let source = open_memory_store();
        let builder = source.create_base_layer().await.unwrap();
        builder
            .add_value_triple(ValueTriple::new_node("a", "p", "b"))
            .unwrap();
        let layer = builder.commit().await.unwrap();
        let id = layer.name();
        let pack = source
            .export_layers(Box::new(vec![id].into_iter()))
            .await
            .unwrap();

        let target_dir = tempdir().unwrap();
        let target = open_archive_store(target_dir.path(), 8);
        let target_a = target.clone();
        let target_b = target.clone();
        let (a, b) = tokio::join!(
            target_a.import_layers(&pack, Box::new(vec![id].into_iter())),
            target_b.import_layers(&pack, Box::new(vec![id].into_iter()))
        );
        assert_ne!(a.is_ok(), b.is_ok());
        let imported = target.get_layer_from_id(id).await.unwrap().unwrap();
        assert!(imported.value_triple_exists(&ValueTriple::new_node("a", "p", "b")));
    }

    #[test]
    fn declared_entry_size_is_rejected_before_accounting_or_allocation() {
        let mut total = 0usize;
        let limits = PackImportLimits {
            max_entry_bytes: 8,
            ..PackImportLimits::default()
        };
        assert!(checked_pack_entry_size(u64::MAX, &mut total, limits).is_err());
        assert_eq!(total, 0);
    }

    #[tokio::test]
    async fn oversized_compressed_input_is_rejected_before_collecting_requested_ids() {
        let target = open_memory_store();
        let limits = PackImportLimits {
            max_compressed_bytes: 8,
            ..PackImportLimits::default()
        };
        let endless = std::iter::repeat([0u32; 5]);
        let result = target
            .import_layers_with_limits(&[0u8; 9], Box::new(endless), HashMap::new(), limits)
            .await;
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::InvalidData);
    }

    #[tokio::test]
    async fn requested_layer_iterators_are_bounded_before_pack_processing() {
        let target = open_memory_store();
        let limits = PackImportLimits {
            max_layers: 1,
            ..PackImportLimits::default()
        };
        let import = target
            .import_layers_with_limits(
                &[],
                Box::new(std::iter::repeat([0u32; 5])),
                HashMap::new(),
                limits,
            )
            .await;
        assert_eq!(import.unwrap_err().kind(), io::ErrorKind::InvalidInput);

        let export = target
            .export_layers_with_limits(Box::new(std::iter::repeat([0u32; 5])), limits)
            .await;
        assert_eq!(export.unwrap_err().kind(), io::ErrorKind::InvalidInput);
    }

    #[tokio::test]
    async fn aggregate_peak_is_rejected_before_entry_allocation_or_publication() {
        let source = open_memory_store();
        let builder = source.create_base_layer().await.unwrap();
        builder
            .add_value_triple(ValueTriple::new_node("a", "p", "b"))
            .unwrap();
        let layer = builder.commit().await.unwrap();
        let id = layer.name();
        let pack = source
            .export_layers(Box::new(vec![id].into_iter()))
            .await
            .unwrap();
        let limits = PackImportLimits {
            max_peak_bytes: pack.len(),
            ..PackImportLimits::default()
        };
        let target = open_memory_store();
        let result = target
            .import_layers_with_limits(
                &pack,
                Box::new(vec![id].into_iter()),
                HashMap::new(),
                limits,
            )
            .await;
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::InvalidData);
        assert!(target.get_layer_from_id(id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn compressed_export_is_bounded_during_encoding() {
        let source = open_memory_store();
        let builder = source.create_base_layer().await.unwrap();
        builder
            .add_value_triple(ValueTriple::new_node("a", "p", "b"))
            .unwrap();
        let layer = builder.commit().await.unwrap();
        let limits = PackImportLimits {
            max_compressed_bytes: 1,
            ..PackImportLimits::default()
        };
        let result = source
            .export_layers_with_limits(Box::new(vec![layer.name()].into_iter()), limits)
            .await;
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::InvalidData);
    }

    #[tokio::test]
    async fn malformed_succinct_structure_is_an_import_error_not_a_process_panic() {
        let source = open_memory_store();
        let builder = source.create_base_layer().await.unwrap();
        builder
            .add_value_triple(ValueTriple::new_node("a", "p", "b"))
            .unwrap();
        let layer = builder.commit().await.unwrap();
        let id = layer.name();
        let pack = source
            .export_layers(Box::new(vec![id].into_iter()))
            .await
            .unwrap();
        let path = format!(
            "{}/{}",
            name_to_string(id),
            FILENAMES.base_s_p_adjacency_list_bits
        );
        let malformed = replace_pack_entry(&pack, &path, &[]);
        let target = open_memory_store();
        let result = target
            .import_layers(&malformed, Box::new(vec![id].into_iter()))
            .await;
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::InvalidData);
        assert!(target.get_layer_from_id(id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn export_and_import_enforce_the_same_negotiated_limits() {
        let source = open_memory_store();
        let builder = source.create_base_layer().await.unwrap();
        builder
            .add_value_triple(ValueTriple::new_node("a", "p", "b"))
            .unwrap();
        let layer = builder.commit().await.unwrap();
        let id = layer.name();
        let limits = PackImportLimits {
            max_layers: 1,
            ..PackImportLimits::default()
        };
        let pack = source
            .export_layers_with_limits(Box::new(vec![id].into_iter()), limits)
            .await
            .unwrap();
        let target = open_memory_store();
        target
            .import_layers_with_limits(
                &pack,
                Box::new(vec![id].into_iter()),
                HashMap::new(),
                limits,
            )
            .await
            .unwrap();
        assert!(target.get_layer_from_id(id).await.unwrap().is_some());

        let lower = PackImportLimits {
            max_layers: 0,
            ..limits
        };
        assert!(source
            .export_layers_with_limits(Box::new(vec![id].into_iter()), lower)
            .await
            .is_err());
    }

    #[tokio::test]
    async fn copied_ancestry_bytes_are_bounded_before_child_publication() {
        let source = open_memory_store();
        let base_builder = source.create_base_layer().await.unwrap();
        base_builder
            .add_value_triple(ValueTriple::new_node("base", "p", "value"))
            .unwrap();
        let base = base_builder.commit().await.unwrap();
        let child_builder = base.open_write().await.unwrap();
        child_builder
            .add_value_triple(ValueTriple::new_node("child", "p", "value"))
            .unwrap();
        let child = child_builder.commit().await.unwrap();
        let base_pack = source
            .export_layers(Box::new(vec![base.name()].into_iter()))
            .await
            .unwrap();
        let child_pack = source
            .export_layers(Box::new(vec![child.name()].into_iter()))
            .await
            .unwrap();
        let target = open_memory_store();
        target
            .import_layers(&base_pack, Box::new(vec![base.name()].into_iter()))
            .await
            .unwrap();
        let limits = PackImportLimits {
            max_ancestry_bytes: 0,
            ..PackImportLimits::default()
        };
        assert!(target
            .import_layers_with_limits(
                &child_pack,
                Box::new(vec![child.name()].into_iter()),
                HashMap::new(),
                limits,
            )
            .await
            .is_err());
        assert!(target
            .get_layer_from_id(child.name())
            .await
            .unwrap()
            .is_none());
    }
}
