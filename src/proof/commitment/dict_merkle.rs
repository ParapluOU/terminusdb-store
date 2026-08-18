//! Id-indexed SHA-256 Merkle trees over the dictionary catalogs — succinct id->term
//! resolution (increment I1).
//!
//! # Why an id-indexed tree
//!
//! A layer's dictionary ids are DENSE over the interval it introduces,
//! `(parent_count, count]` (a bijection the catalog validation already enforces). So the
//! position of an id in the tree is STRUCTURAL: `position = id - parent_count - 1`. Building
//! one Merkle tree per catalog (node/value and predicate, separate id namespaces) makes
//! id->term resolution a direct positional opening in `O(log N)`, and id-uniqueness free
//! (exactly `count - parent_count` real leaves, bound by the root's counts; a padded slot is a
//! domain-separated EMPTY sentinel that can never be a real id).
//!
//! # Leaf, node, and the sentinel
//!
//! * A real leaf commits `(id, datatype_tag, term_bytes)`:
//!   `SHA256(LEAF_DOMAIN ‖ id_be(8) ‖ datatype_tag(1) ‖ len(term_bytes)_be(8) ‖ term_bytes)`.
//!   `datatype_tag`/`term_bytes` are the SAME canonical decomposition the `hash_object`
//!   constant-matching path uses (`super::super::dict`): a `Node`/predicate IRI is
//!   `(NODE_DATATYPE_TAG, iri_bytes)`; a `Value` is `(datatype_discriminant, lexical_bytes)`.
//!   The one-byte tag separates node from value (no `CanonicalDatatype` is `255`) and the
//!   length prefix makes the pair unambiguously parseable, so distinct terms never collide.
//! * A padded slot is the fixed sentinel `SHA256(LEAF_DOMAIN ‖ "empty")` — its preimage can
//!   never equal a real leaf's (which begins with an 8-byte id), so a sentinel can never
//!   masquerade as an id.
//! * An internal node is `SHA256(NODE_DOMAIN ‖ left ‖ right)`, domain-separated from leaves.
//!
//! # Anchor: the two-hash [`resolution_root`] WRAPPER
//!
//! `resolution_root = SHA256(RESOLUTION_DOMAIN ‖ commitment_root ‖ nv_merkle_root ‖
//! pred_merkle_root)` mirrors the existing `database_root` wrapper in [`super::root`]. It leaves
//! `commitment_root` BYTE-IDENTICAL, so every existing gadget golden survives; the two Merkle
//! roots are a deterministic function of the already-committed catalog (recomputed and checked
//! against the root-bound catalog by `validate`), so trusting `commitment_root` transitively
//! trusts them through the wrapper.

use super::*;

/// Domain for a dictionary Merkle LEAF preimage (and, with the `b"empty"` suffix, the
/// padded-slot sentinel).
pub(super) const LEAF_DOMAIN: &[u8] = b"terminus-store/pow/dict-leaf/v1";
/// Domain for an internal Merkle node, separated from the leaf domain.
const NODE_DOMAIN: &[u8] = b"terminus-store/pow/dict-merkle-node/v1";
/// Suffix appended to [`LEAF_DOMAIN`] for the empty padded-slot sentinel.
const EMPTY_SUFFIX: &[u8] = b"empty";
/// Domain for the [`resolution_root`] two-hash wrapper.
const RESOLUTION_DOMAIN: &[u8] = b"terminus-store/pow/dict-merkle/v1";

/// `SHA256(LEAF_DOMAIN ‖ id_be(8) ‖ datatype_tag(1) ‖ len(term_bytes)_be(8) ‖ term_bytes)`.
fn leaf_hash(id: u64, datatype_tag: u8, term_bytes: &[u8]) -> ProofHash {
    let mut hasher = Sha256::new();
    hasher.update(LEAF_DOMAIN);
    hasher.update(id.to_be_bytes());
    hasher.update([datatype_tag]);
    hasher.update((term_bytes.len() as u64).to_be_bytes());
    hasher.update(term_bytes);
    hasher.finalize().into()
}

/// The fixed domain-separated sentinel filling padded slots. Its preimage
/// (`LEAF_DOMAIN ‖ "empty"`) cannot coincide with any real leaf's (which begins
/// with an 8-byte id), so a padded slot can never be opened as a real id.
fn empty_leaf() -> ProofHash {
    let mut hasher = Sha256::new();
    hasher.update(LEAF_DOMAIN);
    hasher.update(EMPTY_SUFFIX);
    hasher.finalize().into()
}

/// `SHA256(NODE_DOMAIN ‖ left ‖ right)`.
fn node_hash(left: &ProofHash, right: &ProofHash) -> ProofHash {
    let mut hasher = Sha256::new();
    hasher.update(NODE_DOMAIN);
    hasher.update(left);
    hasher.update(right);
    hasher.finalize().into()
}

/// The two-hash resolution anchor binding `commitment_root` to the two dictionary
/// Merkle roots without perturbing `commitment_root` itself.
pub fn resolution_root(
    commitment_root: &ProofHash,
    nv_merkle_root: &ProofHash,
    pred_merkle_root: &ProofHash,
) -> ProofHash {
    let mut hasher = Sha256::new();
    hasher.update(RESOLUTION_DOMAIN);
    hasher.update(commitment_root);
    hasher.update(nv_merkle_root);
    hasher.update(pred_merkle_root);
    hasher.finalize().into()
}

/// Canonical `(datatype_tag, term_bytes)` decomposition of a node/value object — the
/// SAME split `hash_object` fingerprints (`Node -> (NODE tag, iri)`, `Value ->
/// (datatype, lexical)`), so Merkle resolution and constant-matching agree on identity.
fn object_leaf_parts(object: &CanonicalObject) -> (u8, &[u8]) {
    match object {
        CanonicalObject::Node(iri) => (NODE_DATATYPE_TAG as u8, iri.as_slice()),
        CanonicalObject::Value { datatype, lexical } => (*datatype as u8, lexical.as_slice()),
    }
}

/// Inverse of [`object_leaf_parts`]: reconstruct the canonical term an opened leaf
/// resolves to. A predicate leaf (always the node tag) reconstructs as a `Node`, exactly
/// as the legacy predicate resolver wraps `record.triple.predicate`.
fn object_from_leaf_parts(datatype_tag: u8, term_bytes: Vec<u8>) -> io::Result<CanonicalObject> {
    if datatype_tag as u64 == NODE_DATATYPE_TAG {
        Ok(CanonicalObject::Node(term_bytes))
    } else {
        Ok(CanonicalObject::Value {
            datatype: CanonicalDatatype::try_from(datatype_tag)?,
            lexical: term_bytes,
        })
    }
}

fn leaf_count(parent_count: u64, count: u64) -> io::Result<usize> {
    if count < parent_count {
        return Err(invalid("dict merkle count is below the parent count"));
    }
    Ok((count - parent_count) as usize)
}

/// Padded, power-of-two leaf domain (min 1) for `leaf_count` real leaves.
fn padded_leaves(leaf_count: usize) -> io::Result<usize> {
    leaf_count
        .max(1)
        .checked_next_power_of_two()
        .ok_or_else(|| invalid("dict merkle leaf count overflow"))
}

/// A retained leaf preimage — what a positional opening re-hashes and reconstructs.
#[derive(Clone, Debug)]
struct LeafPreimage {
    id: u64,
    datatype_tag: u8,
    term_bytes: Vec<u8>,
}

/// One id-indexed dictionary Merkle tree over a layer's introduced interval.
pub(crate) struct DictMerkleTree {
    /// `levels[0]` is the padded leaf row; `levels.last()` is the single-element root row.
    levels: Vec<Vec<ProofHash>>,
    /// Retained real-leaf preimages by position (`None` = padded sentinel slot).
    preimages: Vec<Option<LeafPreimage>>,
    parent_count: u64,
    count: u64,
}

/// Place each `(id, datatype_tag, term_bytes)` at `position = id - parent_count - 1`,
/// padding to a power of two with the empty sentinel. Optionally retains the preimages so
/// the tree can later serve openings. Rejects an id outside the introduced interval, a
/// duplicate position, or an unfilled position (the catalog must cover the interval exactly).
fn place_leaves<'a>(
    parent_count: u64,
    count: u64,
    entries: impl Iterator<Item = (u64, u8, &'a [u8])>,
    mut preimages: Option<&mut Vec<Option<LeafPreimage>>>,
) -> io::Result<Vec<ProofHash>> {
    let real = leaf_count(parent_count, count)?;
    let padded = padded_leaves(real)?;
    let mut leaves = vec![empty_leaf(); padded];
    if let Some(sink) = preimages.as_deref_mut() {
        sink.clear();
        sink.resize_with(padded, || None);
    }
    let mut seen = vec![false; real];
    let mut filled = 0usize;
    for (id, datatype_tag, term_bytes) in entries {
        if id <= parent_count || id > count {
            return Err(invalid("dict entry id lies outside the introduced interval"));
        }
        let position = (id - parent_count - 1) as usize;
        if seen[position] {
            return Err(invalid("duplicate dict id in merkle tree construction"));
        }
        seen[position] = true;
        filled += 1;
        leaves[position] = leaf_hash(id, datatype_tag, term_bytes);
        if let Some(sink) = preimages.as_deref_mut() {
            sink[position] = Some(LeafPreimage {
                id,
                datatype_tag,
                term_bytes: term_bytes.to_vec(),
            });
        }
    }
    if filled != real {
        return Err(invalid("dict merkle tree has an unfilled id position"));
    }
    Ok(leaves)
}

/// Fold a padded leaf row bottom-up into `levels` (`[leaves, .., [root]]`).
fn build_levels(leaves: Vec<ProofHash>) -> Vec<Vec<ProofHash>> {
    let mut levels = vec![leaves];
    while levels.last().map(Vec::len).unwrap_or(0) > 1 {
        let current = levels.last().unwrap();
        let next: Vec<ProofHash> = current
            .chunks(2)
            .map(|pair| node_hash(&pair[0], &pair[1]))
            .collect();
        levels.push(next);
    }
    levels
}

impl DictMerkleTree {
    fn build<'a>(
        parent_count: u64,
        count: u64,
        entries: impl Iterator<Item = (u64, u8, &'a [u8])>,
    ) -> io::Result<Self> {
        let mut preimages = Vec::new();
        let leaves = place_leaves(parent_count, count, entries, Some(&mut preimages))?;
        Ok(Self {
            levels: build_levels(leaves),
            preimages,
            parent_count,
            count,
        })
    }

    pub(crate) fn root(&self) -> ProofHash {
        // `build_levels` always yields a nonempty top row of exactly one hash.
        self.levels.last().and_then(|row| row.first()).copied().unwrap()
    }

    /// Sibling hashes from the leaf level up to (excluding) the root.
    fn siblings(&self, mut position: usize) -> Vec<ProofHash> {
        let height = self.levels.len().saturating_sub(1);
        let mut out = Vec::with_capacity(height);
        for level in &self.levels[..height] {
            out.push(level[position ^ 1]);
            position >>= 1;
        }
        out
    }

    /// Build a positional opening for `queried_id` from the retained preimage.
    pub(crate) fn open(&self, queried_id: u64) -> io::Result<DictOpening> {
        if queried_id <= self.parent_count || queried_id > self.count {
            return Err(invalid("queried dict id outside the layer's introduced interval"));
        }
        let position = (queried_id - self.parent_count - 1) as usize;
        let preimage = self
            .preimages
            .get(position)
            .and_then(|slot| slot.as_ref())
            .ok_or_else(|| invalid("dict merkle position has no committed leaf"))?;
        if preimage.id != queried_id {
            return Err(invalid("dict merkle leaf id does not match its position"));
        }
        Ok(DictOpening {
            id: preimage.id,
            datatype_tag: preimage.datatype_tag,
            term_bytes: preimage.term_bytes.clone(),
            position,
            siblings: self.siblings(position),
        })
    }
}

/// Build the node/value dictionary tree over `node_value_catalog` (ids in
/// `(parent_count, count]`, positioned by `object_id`).
pub(crate) fn build_node_value_tree(
    node_value_catalog: &[ProofRecord],
    parent_count: u64,
    count: u64,
) -> io::Result<DictMerkleTree> {
    DictMerkleTree::build(
        parent_count,
        count,
        node_value_catalog.iter().map(|record| {
            let (tag, bytes) = object_leaf_parts(&record.triple.object);
            (record.object_id, tag, bytes)
        }),
    )
}

/// Build the predicate dictionary tree over `predicate_catalog` (ids in
/// `(parent_count, count]`, positioned by `predicate_id`).
pub(crate) fn build_predicate_tree(
    predicate_catalog: &[ProofRecord],
    parent_count: u64,
    count: u64,
) -> io::Result<DictMerkleTree> {
    DictMerkleTree::build(
        parent_count,
        count,
        predicate_catalog
            .iter()
            .map(|record| (record.predicate_id, NODE_DATATYPE_TAG as u8, record.triple.predicate.as_slice())),
    )
}

/// Node/value Merkle root without retaining preimages (commit/validate path).
pub(super) fn node_value_merkle_root(
    node_value_catalog: &[ProofRecord],
    parent_count: u64,
    count: u64,
) -> io::Result<ProofHash> {
    let leaves = place_leaves(
        parent_count,
        count,
        node_value_catalog.iter().map(|record| {
            let (tag, bytes) = object_leaf_parts(&record.triple.object);
            (record.object_id, tag, bytes)
        }),
        None,
    )?;
    Ok(build_levels(leaves).last().and_then(|row| row.first()).copied().unwrap())
}

/// Predicate Merkle root without retaining preimages (commit/validate path).
pub(super) fn predicate_merkle_root(
    predicate_catalog: &[ProofRecord],
    parent_count: u64,
    count: u64,
) -> io::Result<ProofHash> {
    let leaves = place_leaves(
        parent_count,
        count,
        predicate_catalog
            .iter()
            .map(|record| (record.predicate_id, NODE_DATATYPE_TAG as u8, record.triple.predicate.as_slice())),
        None,
    )?;
    Ok(build_levels(leaves).last().and_then(|row| row.first()).copied().unwrap())
}

/// A positional Merkle opening of one dictionary id: the leaf preimage plus the sibling
/// path up to the root.
///
/// Merkle-dict I2b S5: this is now a PUBLIC (but UNCOMMITTED) proof-wire witness — a term-less
/// [`LayerBgpProof`](crate::proof::query::LayerBgpProof) ships one per distinct matched-constant /
/// resolved-result id. It is never folded into `commitment_root`/`resolution_root`; every consumer
/// re-runs [`verify_opening`] against the trusted root, so a forged/absent opening fails closed.
#[derive(Clone, Debug)]
pub struct DictOpening {
    pub id: u64,
    pub datatype_tag: u8,
    pub term_bytes: Vec<u8>,
    pub position: usize,
    pub siblings: Vec<ProofHash>,
}

/// Verify a positional opening against a trusted (root-bound) Merkle root and return the
/// reconstructed canonical term.
///
/// Soundness: the opening's `id` must equal the queried id, lie in the introduced interval
/// `(parent_count, count]`, and sit at the STRUCTURAL position `id - parent_count - 1`; the
/// leaf re-hashed from `(id, datatype_tag, term_bytes)` must walk the siblings to exactly
/// `bound_root`. Because the honest root fixes the tree height, a short/long/forged path or a
/// sentinel slot (whose preimage can never equal a real leaf) cannot reach it except by a
/// SHA-256 collision. A padded slot can never be reached: its position is `>= leaf_count`, so
/// `id > count` and the interval check rejects it.
pub(crate) fn verify_opening(
    bound_root: &ProofHash,
    parent_count: u64,
    count: u64,
    queried_id: u64,
    opening: &DictOpening,
) -> io::Result<CanonicalObject> {
    if opening.id != queried_id {
        return Err(invalid("dict opening id does not match the queried id"));
    }
    if queried_id <= parent_count || queried_id > count {
        return Err(invalid("queried dict id outside the layer's introduced interval"));
    }
    let expected_position = (queried_id - parent_count - 1) as usize;
    if opening.position != expected_position {
        return Err(invalid("dict opening position is not id - parent_count - 1"));
    }
    let mut hash = leaf_hash(opening.id, opening.datatype_tag, &opening.term_bytes);
    let mut position = opening.position;
    for sibling in &opening.siblings {
        hash = if position & 1 == 0 {
            node_hash(&hash, sibling)
        } else {
            node_hash(sibling, &hash)
        };
        position >>= 1;
    }
    if position != 0 {
        return Err(invalid("dict opening position exceeds the tree height"));
    }
    if &hash != bound_root {
        return Err(invalid(
            "dict opening does not authenticate against the bound merkle root",
        ));
    }
    object_from_leaf_parts(opening.datatype_tag, opening.term_bytes.clone())
}

/// Prover-side, UNCOMMITTED resolution witness for a TERM-LESS artifact (Merkle-dict I2b S3).
///
/// A term-less [`VerifierCommitment`] dropped its two clear catalogs, so the verifier can no
/// longer REBUILD the id-indexed dictionary tree to open an id. Instead the prover ships one
/// positional [`DictOpening`] per DISTINCT resolved id — node/value and predicate ids kept in
/// separate namespaces — and term-less resolution verifies each against the root-bound
/// `nv_merkle_root`/`pred_merkle_root` (β re-anchor), taking the term from the opening's OWN
/// authenticated `term_bytes` ([`object_from_leaf_parts`]). This is EXACTLY analogous to the
/// I2a filter-path `id_hints`: a proof-side witness, NEVER folded into
/// `commitment_root`/`resolution_root`, bound solely by [`verify_opening`] (which re-forces the
/// interval `(parent,count]` + the structural position `id-parent-1` against the trusted root,
/// so a fabricated tree or a wrong-id opening cannot authenticate). A MISSING opening is an
/// `Err` (fail-closed), never a silent empty result.
#[derive(Clone, Debug, Default)]
pub struct ResolutionOpenings {
    node_value: std::collections::BTreeMap<u64, DictOpening>,
    predicate: std::collections::BTreeMap<u64, DictOpening>,
}

impl ResolutionOpenings {
    /// The shipped opening for a node/value id, or `Err` (fail-closed) if the prover shipped
    /// none. The caller still runs [`verify_opening`] against the root-bound root, so a present
    /// but WRONG opening is rejected there; this only turns a MISSING opening into an error
    /// rather than an empty resolution.
    pub(crate) fn node_value_opening(&self, id: u64) -> io::Result<&DictOpening> {
        self.node_value.get(&id).ok_or_else(|| {
            invalid("term-less resolution: prover shipped no node/value Merkle opening for this id")
        })
    }

    /// The shipped opening for a predicate id, or `Err` (fail-closed) if the prover shipped none.
    pub(crate) fn predicate_opening(&self, id: u64) -> io::Result<&DictOpening> {
        self.predicate.get(&id).ok_or_else(|| {
            invalid("term-less resolution: prover shipped no predicate Merkle opening for this id")
        })
    }

    /// Insert a node/value opening (keyed by its own id). Prover/wire constructor. Merkle-dict
    /// I2b S5: un-gated — the shipping term-less producer now mints openings (bound at verify by
    /// [`verify_opening`], never committed).
    pub(crate) fn insert_node_value(&mut self, opening: DictOpening) {
        self.node_value.insert(opening.id, opening);
    }

    /// Insert a predicate opening (keyed by its own id). Prover/wire constructor.
    pub(crate) fn insert_predicate(&mut self, opening: DictOpening) {
        self.predicate.insert(opening.id, opening);
    }

    /// Merkle-dict I2b S5 (wire): the node/value openings in deterministic id order (BTreeMap
    /// iteration) — the encoder's view. Uncommitted; re-verified per id at consume time.
    pub(crate) fn node_value_openings(&self) -> impl Iterator<Item = &DictOpening> {
        self.node_value.values()
    }

    /// Merkle-dict I2b S5 (wire): the predicate openings in deterministic id order.
    pub(crate) fn predicate_openings(&self) -> impl Iterator<Item = &DictOpening> {
        self.predicate.values()
    }

    /// Merkle-dict I2b S5 (wire): reconstruct from decoded opening lists, each keyed by its own
    /// `id`. A duplicate id in either list is malformed (rejected). This is pure PARSING — it
    /// confers NO trust; every opening is re-checked against the trusted root by [`verify_opening`]
    /// at consume time, and a MISSING opening fails closed there.
    pub(crate) fn from_wire(
        node_value: Vec<DictOpening>,
        predicate: Vec<DictOpening>,
    ) -> io::Result<Self> {
        let mut out = Self::default();
        for opening in node_value {
            if out.node_value.insert(opening.id, opening).is_some() {
                return Err(invalid("duplicate node/value id in resolution openings"));
            }
        }
        for opening in predicate {
            if out.predicate.insert(opening.id, opening).is_some() {
                return Err(invalid("duplicate predicate id in resolution openings"));
            }
        }
        Ok(out)
    }
}

/// Prover-side builder: assemble the [`ResolutionOpenings`] for the given DISTINCT node/value
/// and predicate ids from TERM-BEARING layer view(s) (the prover still holds the clear catalogs
/// and rebuilds each tree). Each id is ROUTED to the unique view whose introduced interval
/// `(parent_count, count]` contains it — the same public routing the resolver re-forces — and
/// opened positionally. A single-element slice serves the base path; the N-element slice serves
/// the chain. Every supplied view MUST be term-bearing (a term-less view has no catalog to open
/// from); an id outside every interval (or in more than one) is an `Err`. The produced openings
/// are an UNCOMMITTED witness the prover ships alongside the term-less artifact.
///
/// Merkle-dict I2b S5: un-gated (production). The shipping term-less BGP/ReadDocument producer
/// calls this to mint the uncommitted opening witness it ships alongside the term-less artifact;
/// the coupling guard against the I2b object-constant forge is now S4's mandatory per-constant
/// belt-hash (not this compile-time gate).
pub(crate) fn build_resolution_openings(
    term_bearing_views: &[VerifierCommitmentRef<'_>],
    node_value_ids: &[u64],
    predicate_ids: &[u64],
) -> io::Result<ResolutionOpenings> {
    let mut nv_trees: Vec<Option<DictMerkleTree>> =
        (0..term_bearing_views.len()).map(|_| None).collect();
    let mut pred_trees: Vec<Option<DictMerkleTree>> =
        (0..term_bearing_views.len()).map(|_| None).collect();
    let mut out = ResolutionOpenings::default();

    for &id in node_value_ids {
        if out.node_value.contains_key(&id) {
            continue;
        }
        let i = nv_interval_layer(term_bearing_views, id)?;
        let view = term_bearing_views[i];
        view.require_terms_present("building node/value resolution openings")?;
        if nv_trees[i].is_none() {
            nv_trees[i] = Some(build_node_value_tree(
                view.node_value_catalog,
                view.state.parent_node_value_count,
                view.state.node_value_count,
            )?);
        }
        out.insert_node_value(nv_trees[i].as_ref().unwrap().open(id)?);
    }

    for &id in predicate_ids {
        if out.predicate.contains_key(&id) {
            continue;
        }
        let i = pred_interval_layer(term_bearing_views, id)?;
        let view = term_bearing_views[i];
        view.require_terms_present("building predicate resolution openings")?;
        if pred_trees[i].is_none() {
            pred_trees[i] = Some(build_predicate_tree(
                view.predicate_catalog,
                view.state.parent_predicate_count,
                view.state.predicate_count,
            )?);
        }
        out.insert_predicate(pred_trees[i].as_ref().unwrap().open(id)?);
    }

    Ok(out)
}

/// The unique view index whose node/value interval `(parent, count]` contains `id`.
fn nv_interval_layer(views: &[VerifierCommitmentRef<'_>], id: u64) -> io::Result<usize> {
    let mut found = None;
    for (i, view) in views.iter().enumerate() {
        if id > view.state.parent_node_value_count && id <= view.state.node_value_count {
            if found.is_some() {
                return Err(invalid("node/value id lies in more than one layer's interval"));
            }
            found = Some(i);
        }
    }
    found.ok_or_else(|| invalid("node/value id lies outside every layer's introduced interval"))
}

/// The unique view index whose predicate interval `(parent, count]` contains `id`.
fn pred_interval_layer(views: &[VerifierCommitmentRef<'_>], id: u64) -> io::Result<usize> {
    let mut found = None;
    for (i, view) in views.iter().enumerate() {
        if id > view.state.parent_predicate_count && id <= view.state.predicate_count {
            if found.is_some() {
                return Err(invalid("predicate id lies in more than one layer's interval"));
            }
            found = Some(i);
        }
    }
    found.ok_or_else(|| invalid("predicate id lies outside every layer's introduced interval"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &ProofHash) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    fn node_record(id: u64, iri: &[u8]) -> ProofRecord {
        ProofRecord {
            subject_id: 0,
            predicate_id: 0,
            object_id: id,
            triple: CanonicalTriple {
                subject: Vec::new(),
                predicate: Vec::new(),
                object: CanonicalObject::Node(iri.to_vec()),
            },
        }
    }

    fn value_record(id: u64, datatype: CanonicalDatatype, lexical: &[u8]) -> ProofRecord {
        ProofRecord {
            subject_id: 0,
            predicate_id: 0,
            object_id: id,
            triple: CanonicalTriple {
                subject: Vec::new(),
                predicate: Vec::new(),
                object: CanonicalObject::Value {
                    datatype,
                    lexical: lexical.to_vec(),
                },
            },
        }
    }

    fn predicate_record(id: u64, iri: &[u8]) -> ProofRecord {
        ProofRecord {
            subject_id: 0,
            predicate_id: id,
            object_id: 0,
            triple: CanonicalTriple::node(Vec::new(), iri.to_vec(), Vec::new()),
        }
    }

    // Independently computed with `sha256sum` over the exact preimages (see the module doc /
    // I1 report). If the leaf/sentinel/node/wrapper preimage layout ever drifts, these fail —
    // and Merkle resolution would diverge from the `hash_object` constant-matching path.
    #[test]
    fn leaf_and_sentinel_preimage_golden() {
        // leaf(id=1, tag=NODE=255, "a")
        assert_eq!(
            hex(&leaf_hash(1, NODE_DATATYPE_TAG as u8, b"a")),
            "6984eb968bf80d6652dd2937a5de5ab6724b525a2a9ef81883cdae342a3a6ef5"
        );
        // leaf(id=2, tag=String=0, "lit")
        assert_eq!(
            hex(&leaf_hash(2, CanonicalDatatype::String as u8, b"lit")),
            "4accf92f5e8313513cb7dac3c1252298bbf170200b29690c7c1ef5c7487477e4"
        );
        assert_eq!(
            hex(&empty_leaf()),
            "b5ec163d105dfa14c6eaf763a983cda631e4fd2360f9715079b609d92b407dcd"
        );
    }

    #[test]
    fn small_tree_and_resolution_root_golden() {
        // Node a->1, Value(String,"lit")->2: a 2-leaf tree, root = node_hash(leaf1, leaf2).
        let nv = vec![
            node_record(1, b"a"),
            value_record(2, CanonicalDatatype::String, b"lit"),
        ];
        let root = node_value_merkle_root(&nv, 0, 2).unwrap();
        assert_eq!(
            hex(&root),
            "7ffdf1cbd3c262456a70a08e5d75001c8c68bb8d024002c822aee29636dbe1a0"
        );

        // resolution_root wrapper over fixed inputs (independent sha256sum oracle).
        let anchor = resolution_root(&[0x11; 32], &[0x22; 32], &[0x33; 32]);
        assert_eq!(
            hex(&anchor),
            "e158d6eee64a8325da10876863867da3c3bea6dbb87f55007bb12101624f78ae"
        );
    }

    #[test]
    fn open_and_verify_roundtrip_for_node_value_and_predicate() {
        let nv = vec![
            node_record(1, b"a"),
            value_record(2, CanonicalDatatype::String, b"lit"),
            node_record(3, b"c"),
        ];
        let tree = build_node_value_tree(&nv, 0, 3).unwrap();
        assert_eq!(tree.root(), node_value_merkle_root(&nv, 0, 3).unwrap());
        for record in &nv {
            let opening = tree.open(record.object_id).unwrap();
            let object = verify_opening(&tree.root(), 0, 3, record.object_id, &opening).unwrap();
            assert_eq!(object, record.triple.object);
        }

        let preds = vec![predicate_record(1, b"p"), predicate_record(2, b"q")];
        let ptree = build_predicate_tree(&preds, 0, 2).unwrap();
        let opening = ptree.open(2).unwrap();
        let object = verify_opening(&ptree.root(), 0, 2, 2, &opening).unwrap();
        // Predicate leaves reconstruct as `Node`, matching the legacy predicate resolver.
        assert_eq!(object, CanonicalObject::Node(b"q".to_vec()));
    }

    #[test]
    fn delta_interval_positions_are_offset_by_parent_count() {
        // Introduced interval (10, 13]: ids 11,12,13 at positions 0,1,2.
        let nv = vec![node_record(11, b"x"), node_record(12, b"y"), node_record(13, b"z")];
        let tree = build_node_value_tree(&nv, 10, 13).unwrap();
        let opening = tree.open(12).unwrap();
        assert_eq!(opening.position, 1);
        assert_eq!(
            verify_opening(&tree.root(), 10, 13, 12, &opening).unwrap(),
            CanonicalObject::Node(b"y".to_vec())
        );
    }

    #[test]
    fn tampered_sibling_is_rejected() {
        let nv = vec![node_record(1, b"a"), node_record(2, b"b"), node_record(3, b"c"), node_record(4, b"d")];
        let tree = build_node_value_tree(&nv, 0, 4).unwrap();
        let mut opening = tree.open(3).unwrap();
        opening.siblings[0][0] ^= 0x01;
        assert!(verify_opening(&tree.root(), 0, 4, 3, &opening).is_err());
    }

    #[test]
    fn wrong_position_for_an_id_is_rejected() {
        let nv = vec![node_record(1, b"a"), node_record(2, b"b"), node_record(3, b"c"), node_record(4, b"d")];
        let tree = build_node_value_tree(&nv, 0, 4).unwrap();
        let mut opening = tree.open(3).unwrap();
        opening.position = 0; // id 3 must sit at position 2
        assert!(verify_opening(&tree.root(), 0, 4, 3, &opening).is_err());
    }

    #[test]
    fn leaf_whose_id_differs_from_the_queried_id_is_rejected() {
        let nv = vec![node_record(1, b"a"), node_record(2, b"b")];
        let tree = build_node_value_tree(&nv, 0, 2).unwrap();
        let opening = tree.open(1).unwrap();
        assert!(verify_opening(&tree.root(), 0, 2, 2, &opening).is_err());
    }

    #[test]
    fn empty_slot_sentinel_cannot_be_passed_off_as_an_id() {
        // Three real leaves -> padded to 4; position 3 is the empty sentinel. An attacker
        // presenting the sentinel's authentic sibling path but claiming it is an id fails,
        // because any id at position 3 would be id=4 > count=3 (interval check), and no leaf
        // preimage hashes to the sentinel anyway.
        let nv = vec![node_record(1, b"a"), node_record(2, b"b"), node_record(3, b"c")];
        let tree = build_node_value_tree(&nv, 0, 3).unwrap();
        // Sibling path for the sentinel slot at position 3.
        let siblings = tree.siblings(3);
        let forged = DictOpening {
            id: 4,
            datatype_tag: NODE_DATATYPE_TAG as u8,
            term_bytes: b"forged".to_vec(),
            position: 3,
            siblings,
        };
        assert!(verify_opening(&tree.root(), 0, 3, 4, &forged).is_err());
    }

    #[test]
    fn construction_rejects_a_gap_in_the_introduced_interval() {
        // (0, 3] needs ids 1,2,3; supplying 1,2,2 leaves position 2 unfilled.
        let nv = vec![node_record(1, b"a"), node_record(2, b"b"), node_record(2, b"b")];
        assert!(build_node_value_tree(&nv, 0, 3).is_err());
    }
}
