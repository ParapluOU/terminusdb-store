//! Proven checkpoints — a provable `optimize()` (`doc/proof-of-woql/checkpoint-proofs.md`, step 1).
//!
//! A checkpoint `C_k` commits the materialized state of a layer chain at head `X_k` (the head
//! layer's `resolution_root`), in the CHAIN's id space:
//!
//! | component | structure | `C_j -> C_k` |
//! |---|---|---|
//! | present S, P, O (SPO-sorted) | Dory columns ([`present_domain_log`]) | re-committed, proven |
//! | node/value `hash`, `value`, `datatype`; predicate `hash` (row `id - 1`) | Dory, fixed layout | homomorphic append ([`fixed_commit`]) |
//! | term trees (`id -> term`) | append-only Merkle ([`term_tree`]) | frontier + appends |
//! | term indexes (`hash(term) -> id`) | keyed sparse Merkle ([`smt`]) | insert paths |
//! | triple index (present triples) | keyed sparse Merkle | delete / insert paths |
//! | facts, counts | small record | recomputed |
//!
//! and `checkpoint_root = H(domain, X_k, head commitment root, previous checkpoint root, counts,
//! commitments, Merkle roots, facts)`.
//!
//! # Why it is incremental
//!
//! A child layer numbers its new terms `parent_count + 1 ..` in sorted order and ids never
//! change, so the dictionary, the term tree and the term index only grow at the end; only the
//! present columns change in place. [`prove_checkpoint`] proves `C_k` from `C_j` and the deltas
//! `j+1..k`; [`verify_checkpoint`] needs only `C_j`'s artifact, the deltas' term-bearing verifier
//! views (whose catalogs ARE the new dictionary entries) and the proof — never the history before
//! `j`. Its work is the sumcheck/GKR/Dory verification plus `O(delta)` MSM (the deltas' records
//! re-committed, the new dictionary segments committed at their offsets) and `O(delta · log n)`
//! Merkle hashing.
//!
//! # What the proof shows
//!
//! * The present relation: the signed LogUp-GKR link of the reconcile core over four blocks,
//!   `present(C_k) ⊎ removals ≡ present(C_j) ⊎ additions`, in id space (no dictionary lookups —
//!   both sides share the chain's ids); a zero-check that `C_k`'s padding rows are zero; and
//!   strict sortedness of the packed `(S, P, O)` key, so `C_k`'s columns are the unique SPO-sorted
//!   set. The additions and removals are shipped clear (they are `O(delta)`), re-committed against
//!   each delta's root-bound column commitments, and evaluated directly by the verifier.
//! * The triple index: each removal deletes a member and each addition inserts a non-member,
//!   delta by delta — this also checks the store's delta invariant (add absent, remove present).
//! * The dictionary: the verifier appends the deltas' new terms to the previous commitments,
//!   term trees and term indexes itself; a term index insert is a non-membership proof, so no
//!   term is ever minted twice.

use std::io;

use dory_pcs::backends::arkworks::{ArkFr, ArkGT};

use crate::proof::argument::{ColumnOpening, SumcheckProof};
use crate::proof::logup_gkr::GkrFractionProof;
use crate::proof::sort::SortedProof;
use crate::proof::ProofHash;

pub mod fixed_commit;
mod genesis;
mod inductive;
pub mod smt;
mod state;
pub mod term_tree;
pub mod transition;
mod record;
#[cfg(test)]
mod record_tests;
#[cfg(test)]
mod transition_tests;
#[cfg(test)]
mod query_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod audit_inductive_tests;
#[cfg(test)]
mod audit_smt_tests;
#[cfg(test)]
mod audit_view_tests;
#[cfg(test)]
mod audit_transition_tests;
pub mod view;

pub use genesis::{genesis_checkpoint, genesis_checkpoint_with_context, verify_genesis_checkpoint};
pub use inductive::{prove_checkpoint, prove_checkpoint_with_context, verify_checkpoint};
pub use transition::{
    decode_and_verify_write_transition, prove_write_transition, verify_write_transition, DeltaTriples, WriteDiff,
    WriteTransition, WriteTransitionWitness, MAX_TRANSITION_DELTAS, MAX_TRANSITION_DIFF_TRIPLES,
    MAX_WRITE_TRANSITION_BYTES,
};
pub use record::{
    decode_and_verify_checkpoint_record, CheckpointRecord, VerifiedCheckpointRecord, MAX_CHECKPOINT_RECORD_BYTES,
    MAX_CHECKPOINT_RECORD_DELTAS,
};
pub use view::{
    prove_checkpoint_view, verify_checkpoint_view, CheckpointQueryLayer, CheckpointViewProof,
    VerifiedCheckpointView,
};
pub use state::{
    nv_term_leaf, pred_term_leaf, present_domain_log, triple_key, CheckpointArtifact,
    CheckpointState, MAX_CHECKPOINT_TRIPLES, TRIPLE_PRESENT,
};

/// A triple of chain ids `(subject, predicate, object)`.
pub type IdTriple = [u64; 3];

/// One delta's part of a checkpoint proof: its clear id triples (in the layer's committed SPO
/// order) and the triple-index update witnesses.
#[derive(Clone, Debug)]
pub struct DeltaUpdateProof {
    pub additions: Vec<IdTriple>,
    pub removals: Vec<IdTriple>,
    /// One delete witness per removal, applied first, in order.
    pub removal_paths: Vec<smt::SmtDeleteProof>,
    /// One insert (non-membership) witness per addition, applied after the removals, in order.
    pub addition_paths: Vec<smt::SmtPath>,
}

/// The zero-check that `C_k`'s present columns are zero beyond the present count.
#[derive(Clone, Debug)]
pub struct PaddingProof {
    pub sumcheck: SumcheckProof,
    pub opening: ColumnOpening,
}

/// The inductive proof `C_j -> C_k`.
#[derive(Clone, Debug)]
pub struct CheckpointProof {
    pub deltas: Vec<DeltaUpdateProof>,
    /// Frontiers of `C_j`'s term trees.
    pub nv_tree_frontier: Vec<ProofHash>,
    pub pred_tree_frontier: Vec<ProofHash>,
    /// One insert witness per new term, in id order.
    pub nv_index_paths: Vec<smt::SmtPath>,
    pub pred_index_paths: Vec<smt::SmtPath>,
    /// `log2` of the link's per-block domain.
    pub working_log: usize,
    /// The signed four-block LogUp-GKR link.
    pub link: GkrFractionProof,
    /// `C_j`'s present S, P, O at the link point (empty when `C_j` has no triples).
    pub prev_present_evals: Vec<ArkFr>,
    pub prev_present_opening: Option<ColumnOpening>,
    /// `C_k`'s present S, P, O at the link point (empty when `C_k` has no triples).
    pub next_present_evals: Vec<ArkFr>,
    pub next_present_opening: Option<ColumnOpening>,
    /// Present iff `C_k` has triples.
    pub padding: Option<PaddingProof>,
    /// Present iff `C_k` has two or more triples.
    pub sorted: Option<SortedProof>,
}

/// The genesis proof: the base layer's present triples, re-committed by the verifier against the
/// base layer's own materialized column commitments.
#[derive(Clone, Debug)]
pub struct GenesisProof {
    pub present: Vec<IdTriple>,
}

/// DB-free byte-in verification of an inductive checkpoint: decode `C_j`'s artifact (which must
/// carry the chain-recorded `trusted_prev_checkpoint_root`), the deltas' term-bearing verifier
/// commitments (chain order), `C_k`'s artifact and the proof, then [`verify_checkpoint`]. Returns
/// the verified `C_k`.
pub fn decode_and_verify_checkpoint(
    trusted_prev_checkpoint_root: ProofHash,
    prev_artifact: &[u8],
    delta_views: &[&[u8]],
    trusted_head_root: ProofHash,
    next_artifact: &[u8],
    proof: &[u8],
) -> io::Result<CheckpointArtifact> {
    use crate::proof::commitment::{VerifierCommitment, VerifierLayer};
    let prev = CheckpointArtifact::decode(prev_artifact)?;
    if prev.checkpoint_root != trusted_prev_checkpoint_root {
        return Err(crate::proof::invalid("previous checkpoint is not the trusted one"));
    }
    let views = delta_views
        .iter()
        .map(|bytes| VerifierCommitment::decode(bytes))
        .collect::<io::Result<Vec<_>>>()?;
    let refs: Vec<_> = views.iter().map(|v| v.verifier_view()).collect();
    let next = CheckpointArtifact::decode(next_artifact)?;
    let proof = CheckpointProof::decode(proof)?;
    verify_checkpoint(&prev, &refs, trusted_head_root, &next, &proof)?;
    Ok(next)
}

/// DB-free byte-in verification of a genesis checkpoint against the base layer's term-bearing
/// verifier commitment and its trusted root. Returns the verified `C_0`.
pub fn decode_and_verify_genesis_checkpoint(
    base_view: &[u8],
    trusted_head_root: ProofHash,
    artifact: &[u8],
    proof: &[u8],
) -> io::Result<CheckpointArtifact> {
    use crate::proof::commitment::{VerifierCommitment, VerifierLayer};
    let base = VerifierCommitment::decode(base_view)?;
    let artifact = CheckpointArtifact::decode(artifact)?;
    let proof = GenesisProof::decode(proof)?;
    verify_genesis_checkpoint(base.verifier_view(), trusted_head_root, &artifact, &proof)?;
    Ok(artifact)
}

/// Commit an id column the way a layer commits its delta columns (`next_pow2(len)` padding,
/// identity when empty).
pub(crate) fn commit_layer_id_column(ids: &[u64]) -> io::Result<ArkGT> {
    use dory_pcs::primitives::arithmetic::{Field, Group};
    if ids.is_empty() {
        return Ok(<ArkGT as Group>::identity());
    }
    let mut column = vec![ArkFr::zero(); ids.len().next_power_of_two()];
    for (slot, id) in column.iter_mut().zip(ids) {
        *slot = ArkFr::from_u64(*id);
    }
    crate::proof::argument::commit_column(&column)
}
