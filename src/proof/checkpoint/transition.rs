//! The write-transition proof (`doc/proof-of-woql/checkpoint-proofs.md` §4, step 3).
//!
//! A catalog write names its base head `X_{k-1}`, a public diff and its candidate `X_k`. The
//! verifier recomputes `X_k` itself from the latest checkpoint `C` it trusts, the deltas since
//! `C` (term-bearing views, chained by their roots from `C`'s head to the base), the diff and a
//! witness the proposer supplies:
//!
//! 1. **Ids.** Every distinct term of the diff carries one path in `C`'s keyed term index. A
//!    membership path gives its id (and the term must not be re-minted by a delta since); a
//!    non-membership path sends the lookup to the deltas' clear new terms; a term in neither is
//!    NEW and gets the next id in canonical term order, `base_count + 1 ..` (nodes before values,
//!    each sorted; predicates sorted), exactly as the store's child builder numbers it.
//! 2. **Delta invariant.** Every diff triple whose ids all resolve carries one path in `C`'s
//!    triple index (presence at `C`); the deltas since, whose clear id triples are re-committed
//!    against their root-bound column commitments, then override it in chain order. Each removal
//!    must end present and each addition absent (an addition with a new term is absent).
//! 3. **Root.** The delta's add/remove records and incremental dictionary are then known, so the
//!    verifier builds the candidate's commitments ([`LayerCommitment::from_native_delta`], an
//!    `O(delta)` MSM) and its `resolution_root` exactly as the store does.
//!
//! Cost: `O(diff · log n)` hashing for the paths plus `O(diff)` MSM for the candidate, plus the
//! deltas since (`O(deltas since)`, bounded by the checkpoint interval). Nothing depends on the
//! catalog size `n` except the path depth.

use std::collections::{BTreeMap, BTreeSet};
use std::io;

use super::genesis::{node_values_by_id, predicates_by_id};
use super::inductive::check_delta_records;
use super::smt::{verify_membership, verify_non_membership, SmtPath, SmtTerminal, NS_NV_INDEX, NS_PRED_INDEX, NS_TRIPLE_INDEX};
use super::state::{triple_key, CheckpointArtifact, CheckpointState, TRIPLE_PRESENT};
use super::IdTriple;
use crate::proof::commitment::{node_value_sort_key, predicate_sort_key, LayerCommitment, VerifierCommitmentRef};
use crate::proof::{invalid, CanonicalObject, CanonicalTriple, ProofHash, ProofRecord};

/// Most deltas since the checkpoint a transition may span (a checkpoint is due well before).
pub const MAX_TRANSITION_DELTAS: usize = 256;
/// Most triples (additions plus removals) one write's diff may carry.
pub const MAX_TRANSITION_DIFF_TRIPLES: usize = 100_000;
/// Hard cap on an encoded [`WriteTransition`] (checkpoint artifact, deltas since, diff, witness).
pub const MAX_WRITE_TRANSITION_BYTES: usize = 32 * 1024 * 1024;

/// A write's public diff in canonical form: additions and removals each strictly sorted (so
/// duplicate-free) and disjoint.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WriteDiff {
    pub additions: Vec<CanonicalTriple>,
    pub removals: Vec<CanonicalTriple>,
}

impl WriteDiff {
    /// The canonical form of `additions` / `removals` (sorted); rejects duplicates and a triple
    /// both added and removed.
    pub fn new(mut additions: Vec<CanonicalTriple>, mut removals: Vec<CanonicalTriple>) -> io::Result<Self> {
        additions.sort();
        removals.sort();
        let diff = WriteDiff { additions, removals };
        diff.check_canonical()?;
        Ok(diff)
    }

    /// Strictly sorted, disjoint and within [`MAX_TRANSITION_DIFF_TRIPLES`].
    pub fn check_canonical(&self) -> io::Result<()> {
        if self.additions.len() + self.removals.len() > MAX_TRANSITION_DIFF_TRIPLES {
            return Err(invalid("write diff exceeds the transition triple cap"));
        }
        for table in [&self.additions, &self.removals] {
            if table.windows(2).any(|w| w[0] >= w[1]) {
                return Err(invalid("write diff is not strictly sorted (unsorted or duplicate triple)"));
            }
        }
        let removals: BTreeSet<&CanonicalTriple> = self.removals.iter().collect();
        if self.additions.iter().any(|t| removals.contains(t)) {
            return Err(invalid("write diff adds and removes the same triple"));
        }
        Ok(())
    }

    /// The diff's distinct node/value terms (subjects and objects) and predicates, sorted.
    fn terms(&self) -> (BTreeSet<CanonicalObject>, BTreeSet<Vec<u8>>) {
        let mut nv = BTreeSet::new();
        let mut pred = BTreeSet::new();
        for triple in self.additions.iter().chain(&self.removals) {
            nv.insert(CanonicalObject::Node(triple.subject.clone()));
            nv.insert(triple.object.clone());
            pred.insert(triple.predicate.clone());
        }
        (nv, pred)
    }
}

/// One delta since the checkpoint: its clear id triples, in the layer's committed SPO order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DeltaTriples {
    pub additions: Vec<IdTriple>,
    pub removals: Vec<IdTriple>,
}

/// The proposer's witness for a write transition.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WriteTransitionWitness {
    /// One entry per delta since the checkpoint, in chain order.
    pub deltas: Vec<DeltaTriples>,
    /// One checkpoint term-index path (membership or non-membership) per distinct node/value
    /// term of the diff, in canonical term order.
    pub nv_term_paths: Vec<SmtPath>,
    /// Likewise per distinct predicate of the diff, in byte order.
    pub pred_term_paths: Vec<SmtPath>,
    /// One checkpoint triple-index path per diff triple whose ids all resolve: additions first,
    /// then removals, each in the diff's order.
    pub triple_paths: Vec<SmtPath>,
}

/// What a write transition ships: the checkpoint artifact (must carry the trusted root), the
/// deltas since (term-bearing views, chain order), the diff and the witness.
#[derive(Clone, Debug)]
pub struct WriteTransition {
    pub checkpoint: CheckpointArtifact,
    pub deltas: Vec<crate::proof::commitment::VerifierCommitment>,
    pub diff: WriteDiff,
    pub witness: WriteTransitionWitness,
}

/// The base a transition applies to: the head of the deltas since (or the checkpoint's head).
struct Base {
    root: ProofHash,
    commitment_root: ProofHash,
    node_value_count: u64,
    predicate_count: u64,
}

/// The dictionary and present relation the deltas since add to the checkpoint's.
struct SinceCheckpoint {
    base: Base,
    nv_ids: BTreeMap<CanonicalObject, u64>,
    pred_ids: BTreeMap<Vec<u8>, u64>,
    /// The last delta's word on a triple: `true` = added, `false` = removed.
    touched: BTreeMap<IdTriple, bool>,
}

/// Walk the deltas since the checkpoint: authentic, term-bearing, chained by root from the
/// checkpoint's head, their new ids in sorted term order and never re-minted among themselves;
/// their clear triples are their committed records.
fn verified_since(
    checkpoint: &CheckpointArtifact,
    deltas: &[VerifierCommitmentRef<'_>],
    records: &[DeltaTriples],
) -> io::Result<SinceCheckpoint> {
    if deltas.len() > MAX_TRANSITION_DELTAS {
        return Err(invalid("write transition spans more deltas than the cap since its checkpoint"));
    }
    if records.len() != deltas.len() {
        return Err(invalid("write transition witness covers another number of deltas"));
    }
    let mut base = Base {
        root: checkpoint.head_root,
        commitment_root: checkpoint.head_commitment_root,
        node_value_count: checkpoint.node_value_count,
        predicate_count: checkpoint.predicate_count,
    };
    let mut nv_ids = BTreeMap::new();
    let mut pred_ids = BTreeMap::new();
    let mut touched = BTreeMap::new();
    for (view, delta) in deltas.iter().zip(records) {
        view.require_terms_present("write transition delta")?;
        view.validate()?;
        let state = view.state;
        if state.parent_state_root != Some(base.commitment_root) {
            return Err(invalid(
                "write transition deltas do not chain from the checkpoint's head (spliced, reordered, skipped or foreign layer)",
            ));
        }
        if state.parent_node_value_count != base.node_value_count || state.parent_predicate_count != base.predicate_count {
            return Err(invalid("write transition delta dictionary counts do not abut"));
        }
        check_delta_records(view, &delta.additions, &delta.removals)?;
        let (nv, pred) = super::inductive::new_terms(state, view.node_value_catalog, view.predicate_catalog)?;
        for (i, term) in nv.into_iter().enumerate() {
            if nv_ids.insert(term, state.parent_node_value_count + i as u64 + 1).is_some() {
                return Err(invalid("a delta since the checkpoint re-mints a node/value term"));
            }
        }
        for (i, term) in pred.into_iter().enumerate() {
            if pred_ids.insert(term, state.parent_predicate_count + i as u64 + 1).is_some() {
                return Err(invalid("a delta since the checkpoint re-mints a predicate"));
            }
        }
        for t in &delta.removals {
            touched.insert(*t, false);
        }
        for t in &delta.additions {
            touched.insert(*t, true);
        }
        base = Base {
            root: view.resolution_root(),
            commitment_root: state.commitment_root,
            node_value_count: state.node_value_count,
            predicate_count: state.predicate_count,
        };
    }
    Ok(SinceCheckpoint { base, nv_ids, pred_ids, touched })
}

/// How the verifier resolves one term: the checkpoint's index path decides between the
/// checkpoint's id and "not in the checkpoint".
fn resolve_term<K: Ord>(
    namespace: u8,
    index_root: &ProofHash,
    key: &[u8; 32],
    path: &SmtPath,
    checkpoint_count: u64,
    since: &BTreeMap<K, u64>,
    term: &K,
) -> io::Result<Option<u64>> {
    let member = matches!(&path.terminal, SmtTerminal::Leaf { key: found, .. } if found == key);
    if member {
        let id = verify_membership(namespace, index_root, key, path)?;
        if id == 0 || id > checkpoint_count {
            return Err(invalid("checkpoint term index maps a term outside the checkpoint's dictionary"));
        }
        if since.contains_key(term) {
            return Err(invalid("a delta since the checkpoint re-mints a checkpoint term"));
        }
        Ok(Some(id))
    } else {
        verify_non_membership(namespace, index_root, key, path)?;
        Ok(since.get(term).copied())
    }
}

/// Verify a write transition and return the candidate's commitment (its `resolution_root` is
/// the recomputed `X_k`). `checkpoint` must be the trusted checkpoint (callers check its root;
/// [`decode_and_verify_write_transition`] does), `deltas` the term-bearing views of the deltas
/// since it, in chain order, and `base_root` the head the write names.
pub fn verify_write_transition(
    checkpoint: &CheckpointArtifact,
    deltas: &[VerifierCommitmentRef<'_>],
    base_root: ProofHash,
    diff: &WriteDiff,
    witness: &WriteTransitionWitness,
) -> io::Result<LayerCommitment> {
    checkpoint.validate()?;
    diff.check_canonical()?;
    let since = verified_since(checkpoint, deltas, &witness.deltas)?;
    if since.base.root != base_root {
        return Err(invalid("write transition deltas do not end at the write's base"));
    }

    // 1. Ids.
    let (nv_terms, pred_terms) = diff.terms();
    if witness.nv_term_paths.len() != nv_terms.len() || witness.pred_term_paths.len() != pred_terms.len() {
        return Err(invalid("write transition term paths disagree with the diff's terms"));
    }
    let mut nv_ids: BTreeMap<&CanonicalObject, u64> = BTreeMap::new();
    let mut new_nv = Vec::new();
    for (term, path) in nv_terms.iter().zip(&witness.nv_term_paths) {
        let key = node_value_sort_key(term);
        match resolve_term(NS_NV_INDEX, &checkpoint.nv_term_index_root, &key, path, checkpoint.node_value_count, &since.nv_ids, term)? {
            Some(id) => {
                nv_ids.insert(term, id);
            }
            None => new_nv.push(term),
        }
    }
    let mut pred_ids: BTreeMap<&Vec<u8>, u64> = BTreeMap::new();
    let mut new_pred = Vec::new();
    for (term, path) in pred_terms.iter().zip(&witness.pred_term_paths) {
        let key = predicate_sort_key(term);
        match resolve_term(NS_PRED_INDEX, &checkpoint.pred_term_index_root, &key, path, checkpoint.predicate_count, &since.pred_ids, term)? {
            Some(id) => {
                pred_ids.insert(term, id);
            }
            None => new_pred.push(term),
        }
    }
    // New terms are numbered after the base in canonical term order (BTreeSet order).
    let mut nv_mappings = Vec::with_capacity(new_nv.len());
    for (i, term) in new_nv.into_iter().enumerate() {
        let id = since.base.node_value_count + i as u64 + 1;
        nv_ids.insert(term, id);
        nv_mappings.push((term.clone(), id));
    }
    let mut pred_mappings = Vec::with_capacity(new_pred.len());
    for (i, term) in new_pred.into_iter().enumerate() {
        let id = since.base.predicate_count + i as u64 + 1;
        pred_ids.insert(term, id);
        pred_mappings.push((term.clone(), id));
    }
    let node_value_count = since.base.node_value_count + nv_mappings.len() as u64;
    let predicate_count = since.base.predicate_count + pred_mappings.len() as u64;
    let ids_of = |triple: &CanonicalTriple| -> IdTriple {
        [
            nv_ids[&CanonicalObject::Node(triple.subject.clone())],
            pred_ids[&triple.predicate],
            nv_ids[&triple.object],
        ]
    };
    let is_new = |ids: &IdTriple| {
        ids[0] > since.base.node_value_count || ids[2] > since.base.node_value_count || ids[1] > since.base.predicate_count
    };

    // 2. The delta invariant: removals present, additions absent, at the base.
    let mut paths = witness.triple_paths.iter();
    let mut present_at_base = |ids: &IdTriple| -> io::Result<bool> {
        let path = paths.next().ok_or_else(|| invalid("write transition lacks a triple-index path"))?;
        let key = triple_key(ids);
        let member = matches!(&path.terminal, SmtTerminal::Leaf { key: found, .. } if *found == key);
        let at_checkpoint = if member {
            if verify_membership(NS_TRIPLE_INDEX, &checkpoint.triple_index_root, &key, path)? != TRIPLE_PRESENT {
                return Err(invalid("checkpoint triple index holds a non-presence value"));
            }
            true
        } else {
            verify_non_membership(NS_TRIPLE_INDEX, &checkpoint.triple_index_root, &key, path)?;
            false
        };
        Ok(since.touched.get(ids).copied().unwrap_or(at_checkpoint))
    };
    let mut additions = Vec::with_capacity(diff.additions.len());
    for triple in &diff.additions {
        let ids = ids_of(triple);
        if !is_new(&ids) && present_at_base(&ids)? {
            return Err(invalid("write adds a triple that is already present"));
        }
        additions.push(ProofRecord { subject_id: ids[0], predicate_id: ids[1], object_id: ids[2], triple: triple.clone() });
    }
    let mut removals = Vec::with_capacity(diff.removals.len());
    for triple in &diff.removals {
        let ids = ids_of(triple);
        if is_new(&ids) || !present_at_base(&ids)? {
            return Err(invalid("write removes a triple that is not present"));
        }
        removals.push(ProofRecord { subject_id: ids[0], predicate_id: ids[1], object_id: ids[2], triple: triple.clone() });
    }
    if paths.next().is_some() {
        return Err(invalid("write transition carries surplus triple-index paths"));
    }

    // 3. The candidate's commitments, exactly as the store builds a child layer's.
    LayerCommitment::from_native_delta(
        Some(since.base.commitment_root),
        additions,
        removals,
        Vec::new(),
        since.base.node_value_count,
        since.base.predicate_count,
        node_value_count,
        predicate_count,
        nv_mappings,
        pred_mappings,
    )
}

/// The prover's side: the witness for applying `diff` to the head of `deltas` (the deltas since
/// `checkpoint`, chain order). Fails if the diff breaks the delta invariant.
pub fn prove_write_transition(
    checkpoint: &CheckpointState,
    deltas: &[&LayerCommitment],
    diff: &WriteDiff,
) -> io::Result<WriteTransitionWitness> {
    prove_write_transition_inner(checkpoint, deltas, diff, true)
}

/// [`prove_write_transition`]; with `enforce = false` (tests) it mints the honest paths of a diff
/// that breaks the delta invariant, so the verifier's own checks can be exercised.
pub(super) fn prove_write_transition_inner(
    checkpoint: &CheckpointState,
    deltas: &[&LayerCommitment],
    diff: &WriteDiff,
    enforce: bool,
) -> io::Result<WriteTransitionWitness> {
    if enforce {
        diff.check_canonical()?;
    }
    let records: Vec<DeltaTriples> = deltas
        .iter()
        .map(|d| DeltaTriples {
            additions: d.additions.iter().map(|r| [r.subject_id, r.predicate_id, r.object_id]).collect(),
            removals: d.removals.iter().map(|r| [r.subject_id, r.predicate_id, r.object_id]).collect(),
        })
        .collect();
    let mut nv_since = BTreeMap::new();
    let mut pred_since = BTreeMap::new();
    let mut touched = BTreeMap::new();
    let (mut base_nv, mut base_pred) = (checkpoint.artifact.node_value_count, checkpoint.artifact.predicate_count);
    for (delta, rec) in deltas.iter().zip(&records) {
        let s = &delta.state;
        let nv = node_values_by_id(&delta.node_value_catalog, s.parent_node_value_count + 1, s.node_value_count)?;
        let pred = predicates_by_id(&delta.predicate_catalog, s.parent_predicate_count + 1, s.predicate_count)?;
        for (i, term) in nv.into_iter().enumerate() {
            nv_since.insert(term, s.parent_node_value_count + i as u64 + 1);
        }
        for (i, term) in pred.into_iter().enumerate() {
            pred_since.insert(term, s.parent_predicate_count + i as u64 + 1);
        }
        for t in &rec.removals {
            touched.insert(*t, false);
        }
        for t in &rec.additions {
            touched.insert(*t, true);
        }
        base_nv = s.node_value_count;
        base_pred = s.predicate_count;
    }
    let (nv_terms, pred_terms) = diff.terms();
    let mut nv_ids = BTreeMap::new();
    let mut nv_term_paths = Vec::with_capacity(nv_terms.len());
    let mut fresh = 0u64;
    for term in &nv_terms {
        let key = node_value_sort_key(term);
        let path = checkpoint.nv_index.path(&key);
        let id = match &path.terminal {
            SmtTerminal::Leaf { key: found, value } if *found == key => Some(*value),
            _ => nv_since.get(term).copied(),
        };
        nv_ids.insert(term.clone(), id.unwrap_or_else(|| {
            fresh += 1;
            base_nv + fresh
        }));
        nv_term_paths.push(path);
    }
    let mut pred_ids = BTreeMap::new();
    let mut pred_term_paths = Vec::with_capacity(pred_terms.len());
    let mut fresh = 0u64;
    for term in &pred_terms {
        let key = predicate_sort_key(term);
        let path = checkpoint.pred_index.path(&key);
        let id = match &path.terminal {
            SmtTerminal::Leaf { key: found, value } if *found == key => Some(*value),
            _ => pred_since.get(term).copied(),
        };
        pred_ids.insert(term.clone(), id.unwrap_or_else(|| {
            fresh += 1;
            base_pred + fresh
        }));
        pred_term_paths.push(path);
    }
    let mut triple_paths = Vec::new();
    for (triple, adding) in diff.additions.iter().map(|t| (t, true)).chain(diff.removals.iter().map(|t| (t, false))) {
        let ids = [
            nv_ids[&CanonicalObject::Node(triple.subject.clone())],
            pred_ids[&triple.predicate],
            nv_ids[&triple.object],
        ];
        if ids[0] > base_nv || ids[2] > base_nv || ids[1] > base_pred {
            if !adding && enforce {
                return Err(invalid("write removes a triple that is not present"));
            }
            continue;
        }
        let key = triple_key(&ids);
        let path = checkpoint.triple_index.path(&key);
        let at_checkpoint = matches!(&path.terminal, SmtTerminal::Leaf { key: found, .. } if *found == key);
        let present = touched.get(&ids).copied().unwrap_or(at_checkpoint);
        if present == adding && enforce {
            return Err(invalid(if adding {
                "write adds a triple that is already present"
            } else {
                "write removes a triple that is not present"
            }));
        }
        triple_paths.push(path);
    }
    Ok(WriteTransitionWitness { deltas: records, nv_term_paths, pred_term_paths, triple_paths })
}

impl WriteTransition {
    /// Assemble what a proposer ships for applying `diff` to the head of `deltas`.
    pub fn prove(checkpoint: &CheckpointState, deltas: &[&LayerCommitment], diff: WriteDiff) -> io::Result<Self> {
        let witness = prove_write_transition(checkpoint, deltas, &diff)?;
        let deltas = deltas.iter().map(|d| d.verifier_commitment()).collect::<io::Result<Vec<_>>>()?;
        Ok(WriteTransition { checkpoint: checkpoint.artifact.clone(), deltas, diff, witness })
    }

    /// Verify against the trusted checkpoint root, the write's base and its candidate; returns
    /// the candidate's commitment.
    pub fn verify(
        &self,
        trusted_checkpoint_root: ProofHash,
        base_root: ProofHash,
        candidate_root: ProofHash,
    ) -> io::Result<LayerCommitment> {
        use crate::proof::commitment::VerifierLayer;
        if self.checkpoint.checkpoint_root != trusted_checkpoint_root {
            return Err(invalid("write transition is bound to another checkpoint"));
        }
        let refs: Vec<_> = self.deltas.iter().map(|d| d.verifier_view()).collect();
        let candidate = verify_write_transition(&self.checkpoint, &refs, base_root, &self.diff, &self.witness)?;
        if candidate.resolution_root() != candidate_root {
            return Err(invalid("write's candidate root does not recompute from its base and diff"));
        }
        Ok(candidate)
    }
}

/// DB-free byte-in verification of a write transition (the ledger's check): decode, verify
/// against the chain-recorded `trusted_checkpoint_root`, the write's `base_root` and its
/// `candidate_root`. Returns the verified diff.
pub fn decode_and_verify_write_transition(
    trusted_checkpoint_root: ProofHash,
    base_root: ProofHash,
    candidate_root: ProofHash,
    bytes: &[u8],
) -> io::Result<WriteDiff> {
    let transition = WriteTransition::decode(bytes)?;
    transition.verify(trusted_checkpoint_root, base_root, candidate_root)?;
    Ok(transition.diff)
}
