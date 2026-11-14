//! Recording a proven checkpoint on chain (`doc/proof-of-woql/checkpoint-proofs.md` §6 step 4).
//!
//! A supernode that has proven `C_k` from the chain's recorded checkpoint `C_j` and the accepted
//! deltas `j+1..k` ships a [`CheckpointRecord`]: `C_j`'s artifact, the deltas' TERM-BEARING
//! verifier commitments (chain order), `C_k`'s artifact and the inductive proof. The chain checks
//! it with [`decode_and_verify_checkpoint_record`] against the checkpoint root it recorded and the
//! accepted head the record names, and on success records `(X_k, checkpoint_root(C_k))`.
//!
//! The record also gives the chain each delta's `resolution_root`, so it can check the deltas are
//! exactly its accepted heads since `C_j` (the proof already binds them by their commitment-root
//! hash chain from `C_j`'s head to `X_k`; the roots make the check explicit and cheap).
//!
//! Also here: the two helpers a node needs to keep a checkpoint and cut write transitions against
//! it — [`CheckpointState::of_layer`] (rebuild a recorded checkpoint from the store's
//! materialized view of its head, e.g. after a restart) and [`WriteDiff::of_delta`] (the diff a
//! child layer spells, for writes whose candidate is regenerated rather than given as a diff).

use std::io;

use super::inductive::prove_checkpoint_with_context;
use super::{verify_checkpoint, CheckpointArtifact, CheckpointProof, CheckpointState, IdTriple, WriteDiff};
use crate::layer::Layer;
use crate::proof::argument::StoreProverContext;
use crate::proof::commitment::{LayerCommitment, VerifierCommitment, VerifierLayer};
use crate::proof::{invalid, CanonicalObject, ProofHash};
use crate::store::sync::SyncStoreLayer;

/// Most deltas one checkpoint record may span. Comfortably above the 256-delta write-transition
/// window ([`super::MAX_TRANSITION_DELTAS`]), so a checkpoint can always be recorded even after
/// the window filled up — recording never depends on writes, so writes cannot deadlock it.
pub const MAX_CHECKPOINT_RECORD_DELTAS: usize = 1024;
/// Hard cap on an encoded [`CheckpointRecord`].
pub const MAX_CHECKPOINT_RECORD_BYTES: usize = 64 * 1024 * 1024;

/// What a prover ships to have the chain record `C_k`: the recorded `C_j`, the deltas since it
/// (term-bearing views, chain order), `C_k` and the inductive proof.
#[derive(Clone, Debug)]
pub struct CheckpointRecord {
    pub prev: CheckpointArtifact,
    pub deltas: Vec<VerifierCommitment>,
    pub next: CheckpointArtifact,
    pub proof: CheckpointProof,
}

/// A checkpoint record the verifier accepted: the new checkpoint and the `resolution_root` of
/// every delta it spans, in chain order (the last is the checkpoint's head).
#[derive(Clone, Debug)]
pub struct VerifiedCheckpointRecord {
    pub artifact: CheckpointArtifact,
    pub delta_roots: Vec<ProofHash>,
}

impl CheckpointRecord {
    /// Prove `C_k` from `prev = C_j` and the deltas since it (chain order) and assemble the record.
    pub fn prove(prev: &CheckpointState, deltas: &[&LayerCommitment]) -> io::Result<(CheckpointState, Self)> {
        Self::prove_with_context(&StoreProverContext::cpu(), prev, deltas)
    }

    /// [`Self::prove`] with explicit prover arithmetic.
    pub fn prove_with_context(
        context: &StoreProverContext<'_>,
        prev: &CheckpointState,
        deltas: &[&LayerCommitment],
    ) -> io::Result<(CheckpointState, Self)> {
        if deltas.len() > MAX_CHECKPOINT_RECORD_DELTAS {
            return Err(invalid("checkpoint record spans too many deltas"));
        }
        let (next, proof) = prove_checkpoint_with_context(context, prev, deltas)?;
        let views = deltas.iter().map(|d| d.verifier_commitment()).collect::<io::Result<Vec<_>>>()?;
        let record = CheckpointRecord { prev: prev.artifact.clone(), deltas: views, next: next.artifact.clone(), proof };
        Ok((next, record))
    }

    /// Verify against the chain-recorded `trusted_prev_root` and the accepted head
    /// `trusted_head_root` the record names.
    pub fn verify(&self, trusted_prev_root: ProofHash, trusted_head_root: ProofHash) -> io::Result<VerifiedCheckpointRecord> {
        if self.prev.checkpoint_root != trusted_prev_root {
            return Err(invalid("checkpoint record is bound to another previous checkpoint"));
        }
        if self.deltas.len() > MAX_CHECKPOINT_RECORD_DELTAS {
            return Err(invalid("checkpoint record spans too many deltas"));
        }
        let refs: Vec<_> = self.deltas.iter().map(|d| d.verifier_view()).collect();
        verify_checkpoint(&self.prev, &refs, trusted_head_root, &self.next, &self.proof)?;
        Ok(VerifiedCheckpointRecord {
            artifact: self.next.clone(),
            delta_roots: self.deltas.iter().map(|d| d.resolution_root()).collect(),
        })
    }
}

/// DB-free byte-in verification of a checkpoint record (the ledger's check): decode under
/// [`MAX_CHECKPOINT_RECORD_BYTES`], then [`CheckpointRecord::verify`].
pub fn decode_and_verify_checkpoint_record(
    trusted_prev_root: ProofHash,
    trusted_head_root: ProofHash,
    bytes: &[u8],
) -> io::Result<VerifiedCheckpointRecord> {
    CheckpointRecord::decode(bytes)?.verify(trusted_prev_root, trusted_head_root)
}

impl CheckpointState {
    /// The checkpoint of `layer`'s materialized state built from scratch (`O(n)` commitments and
    /// Merkle trees), chained to `prev_checkpoint_root`. Deterministic: for the head of a proven
    /// checkpoint and that checkpoint's `prev_checkpoint_root` it reproduces the proven artifact
    /// exactly (callers compare `checkpoint_root` with the recorded one), so a node need not
    /// persist the prover's state across a restart.
    pub fn of_layer(
        context: &StoreProverContext<'_>,
        layer: &SyncStoreLayer,
        prev_checkpoint_root: Option<ProofHash>,
    ) -> io::Result<Self> {
        let commitment = layer.proof_commitment()?;
        let present: Vec<IdTriple> = layer.triples().map(|t| [t.subject, t.predicate, t.object]).collect();
        let nv = (1..=layer.node_and_value_count() as u64)
            .map(|id| {
                layer
                    .id_object(id)
                    .map(|o| CanonicalObject::from_store_object(&o))
                    .ok_or_else(|| invalid("layer dictionary has a gap"))
            })
            .collect::<io::Result<Vec<_>>>()?;
        let pred = (1..=layer.predicate_count() as u64)
            .map(|id| layer.id_predicate(id).map(String::into_bytes).ok_or_else(|| invalid("layer predicate dictionary has a gap")))
            .collect::<io::Result<Vec<_>>>()?;
        CheckpointState::from_parts(
            context,
            commitment.resolution_root(),
            commitment.state.commitment_root,
            prev_checkpoint_root,
            present,
            nv,
            pred,
        )
    }
}

impl WriteDiff {
    /// The diff a child layer spells: its added and removed triples, canonical. A layer's records
    /// carry their clear triples, so no dictionary lookup is needed.
    pub fn of_delta(layer: &LayerCommitment) -> io::Result<Self> {
        if layer.state.parent_state_root.is_none() {
            return Err(invalid("a base layer is not a delta"));
        }
        WriteDiff::new(
            layer.additions.iter().map(|r| r.triple.clone()).collect(),
            layer.removals.iter().map(|r| r.triple.clone()).collect(),
        )
    }
}
