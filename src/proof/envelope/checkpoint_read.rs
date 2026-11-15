//! Single-layer query proofs over a PROVEN CHECKPOINT (`doc/proof-of-woql/checkpoint-proofs.md`
//! §5): a [`RecursiveRelationEnvelope`] over a checkpoint's query view plus the resolution
//! witness for exactly the ids and constants its verification reads.
//!
//! # Trust
//!
//! The anchor is a `checkpoint_root` the caller trusts because the chain recorded it after
//! verifying the checkpoint's inductive proof — no quorum and no `Attested` assumption. The
//! verifier builds the view itself ([`verify_checkpoint_view`]): present commitments, facts and
//! term structures from the artifact the root binds, dictionary columns relocated by the view
//! proof. Every result id is opened against the checkpoint's term tree, every constant looked up
//! (membership or absence) in its keyed term index, each re-verified against the root-bound root;
//! a missing one fails closed at the lookup, a forged one at its verification, an extra one at the
//! exact-consumption check.
//!
//! # Checkpoint + deltas since
//!
//! When deltas were written after the checkpoint `C_k`, the reader proves a checkpoint `C'` of the
//! head from `C_k` and those deltas (`prove_checkpoint`: the signed four-block reconcile link
//! `present(C') ⊎ removals ≡ present(C_k) ⊎ additions`, the triple index updated by the delta
//! invariant, the dictionary appended from the deltas' new terms) and the query over `C'`.
//! [`decode_and_verify_checkpoint_delta_read`] verifies `C'` against the trusted `C_k` root, the
//! deltas' verifier commitments and the trusted head root — work bounded by the deltas — and the
//! query over the view of `C'`. `C'` is ephemeral: nothing records it.
//!
//! # Scope
//!
//! Every statement the attested single-layer path admits ([`require_attested_termless_plan`]).

use std::convert::TryInto;
use std::io;

use sha2::{Digest, Sha256};

use super::plan::*;
use super::recursive::RecursiveRelationEnvelope;
use super::termless::require_attested_termless_plan;
use super::wire::*;
use crate::proof::checkpoint::view::{CheckpointQueryView, VerifiedCheckpointView};
use crate::proof::checkpoint::{
    decode_and_verify_checkpoint, verify_checkpoint_view, CheckpointArtifact, CheckpointQueryLayer,
    CheckpointViewProof,
};
use crate::proof::checkpoint::smt::SmtPath;
use crate::proof::commitment::{ResolutionOpenings, SortedNamespace};
use crate::proof::ProofHash;

/// Framing magic of a [`CheckpointRelationEnvelope`] — permanent; never change these bytes.
pub(super) const CHECKPOINT_READ_MAGIC: &[u8; 12] = b"TDBWQCKPQRY\0";
/// Wire-format version of the checkpoint relation envelope (accepted exactly).
pub(super) const CHECKPOINT_READ_VERSION: u32 = 1;

/// A single-layer relation proof over a checkpoint's query view plus its resolution witness.
#[derive(Clone, Debug)]
pub struct CheckpointRelationEnvelope {
    /// The operator-tree proof, proven over the view's base sidecar.
    pub envelope: RecursiveRelationEnvelope,
    /// One term-tree opening per distinct id the verifier resolves and one keyed-index path per
    /// constant it looks up — no more, no fewer.
    pub openings: ResolutionOpenings,
}

impl CheckpointRelationEnvelope {
    /// PROVER: wrap an envelope (proven over `layer.layer()`) with the exact witness, by running
    /// the real verifier over the view with a lazy opener over the checkpoint's term structures.
    pub fn from_envelope(
        envelope: RecursiveRelationEnvelope,
        layer: &CheckpointQueryLayer,
        plan: &RelationProofPlan,
    ) -> io::Result<Self> {
        require_attested_termless_plan(plan)?;
        let oracle = layer.oracle();
        envelope.verify(
            &CheckpointQueryView { commitment: layer.view().commitment(), openings: &oracle },
            layer.commitment_root(),
            plan,
        )?;
        let openings = oracle.consumed_witness()?;
        Ok(Self { envelope, openings })
    }

    /// VERIFIER: check this envelope against a verified checkpoint view and the caller's OWN plan.
    pub fn verify(&self, view: &VerifiedCheckpointView, plan: &RelationProofPlan) -> io::Result<()> {
        require_attested_termless_plan(plan)?;
        if self.openings.is_oracle() {
            return Err(invalid("an oracle carrier is never a shipped witness"));
        }
        let openings = self.openings.fresh_tracked();
        self.envelope.verify(
            &CheckpointQueryView { commitment: view.commitment(), openings: &openings },
            view.commitment_root(),
            plan,
        )?;
        openings.require_exactly_consumed()
    }

    /// Frame as BYTES: magic ‖ version ‖ (node-bounded envelope ‖ id openings ‖ keyed lookups) ‖
    /// SHA-256 checksum.
    pub fn encode(&self) -> io::Result<Vec<u8>> {
        let mut out = Vec::new();
        out.extend_from_slice(CHECKPOINT_READ_MAGIC);
        out.extend_from_slice(&CHECKPOINT_READ_VERSION.to_le_bytes());
        let mut nodes = 0;
        self.envelope.put_bounded(&mut out, 1, &mut nodes)?;
        put_id_openings(&self.openings, &mut out)?;
        let lookups: Vec<_> = self.openings.keyed_lookups().collect();
        if lookups.len() > MAX_ITEMS {
            return Err(invalid("too many keyed term-index lookups"));
        }
        lookups.len().put(&mut out)?;
        for ((namespace, key), path) in lookups {
            match namespace {
                SortedNamespace::NodeValue => 0u8.put(&mut out)?,
                SortedNamespace::Predicate => 1u8.put(&mut out)?,
            }
            key.put(&mut out)?;
            path.put(&mut out)?;
        }
        if out.len().checked_add(CHECKSUM_BYTES).map_or(true, |len| len > MAX_QUERY_ENVELOPE_BYTES) {
            return Err(invalid("checkpoint relation envelope too large"));
        }
        let checksum = Sha256::digest(&out);
        out.extend_from_slice(&checksum);
        Ok(out)
    }

    /// Decode a framed checkpoint relation envelope (parsing only — no trust).
    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < CHECKPOINT_READ_MAGIC.len() + 4 + CHECKSUM_BYTES
            || bytes.len() > MAX_QUERY_ENVELOPE_BYTES
        {
            return Err(invalid("invalid checkpoint relation envelope size"));
        }
        let (body, checksum) = bytes.split_at(bytes.len() - CHECKSUM_BYTES);
        if Sha256::digest(body).as_slice() != checksum {
            return Err(invalid("checkpoint relation envelope checksum mismatch"));
        }
        let mut reader = Reader { bytes: body, pos: 0, decoded_items: 0 };
        if reader.take(CHECKPOINT_READ_MAGIC.len())? != CHECKPOINT_READ_MAGIC {
            return Err(invalid("unknown checkpoint relation envelope magic"));
        }
        if u32::from_le_bytes(reader.take(4)?.try_into().unwrap()) != CHECKPOINT_READ_VERSION {
            return Err(invalid("unsupported checkpoint relation envelope version"));
        }
        let mut nodes = 0;
        let envelope = RecursiveRelationEnvelope::get_bounded(&mut reader, 1, &mut nodes)?;
        let mut openings = ResolutionOpenings::get(&mut reader)?;
        let count = usize::get(&mut reader)?;
        if count > MAX_ITEMS {
            return Err(invalid("too many keyed term-index lookups"));
        }
        for _ in 0..count {
            let namespace = match u8::get(&mut reader)? {
                0 => SortedNamespace::NodeValue,
                1 => SortedNamespace::Predicate,
                _ => return Err(invalid("unknown keyed term-index namespace")),
            };
            let key: ProofHash = Wire::get(&mut reader)?;
            let path: SmtPath = Wire::get(&mut reader)?;
            openings.insert_keyed_from_wire(namespace, key, path)?;
        }
        reader.finish()?;
        Ok(Self { envelope, openings })
    }
}

/// DB-free byte-in verification of a query over a checkpoint with NO deltas since: decode the
/// checkpoint artifact (which must carry the chain-recorded `trusted_checkpoint_root`), its view
/// proof and the envelope, verify the view, then the envelope against the caller's OWN `plan`.
/// Returns the verified envelope (its result is the answer).
pub fn decode_and_verify_checkpoint_read(
    envelope: &[u8],
    artifact: &[u8],
    view_proof: &[u8],
    trusted_checkpoint_root: ProofHash,
    plan: &RelationProofPlan,
) -> io::Result<CheckpointRelationEnvelope> {
    let artifact = CheckpointArtifact::decode(artifact)?;
    let view = verify_checkpoint_view(&artifact, trusted_checkpoint_root, &CheckpointViewProof::decode(view_proof)?)?;
    let envelope = CheckpointRelationEnvelope::decode(envelope)?;
    envelope.verify(&view, plan)?;
    Ok(envelope)
}

/// DB-free byte-in verification of a query over a checkpoint PLUS the deltas written since it:
/// verify the head checkpoint `C'` from the trusted `C_k` (`trusted_checkpoint_root`), the deltas'
/// TERM-BEARING verifier commitments (chain order) and the trusted head root, then its view proof,
/// then the envelope over that view against the caller's OWN `plan`. Every step but the last is
/// bounded by the deltas (plus the succinct checkpoint and view verifications).
#[allow(clippy::too_many_arguments)]
pub fn decode_and_verify_checkpoint_delta_read(
    envelope: &[u8],
    checkpoint_artifact: &[u8],
    trusted_checkpoint_root: ProofHash,
    delta_views: &[&[u8]],
    trusted_head_root: ProofHash,
    head_artifact: &[u8],
    head_checkpoint_proof: &[u8],
    view_proof: &[u8],
    plan: &RelationProofPlan,
) -> io::Result<CheckpointRelationEnvelope> {
    let head = decode_and_verify_checkpoint(
        trusted_checkpoint_root,
        checkpoint_artifact,
        delta_views,
        trusted_head_root,
        head_artifact,
        head_checkpoint_proof,
    )?;
    let view = verify_checkpoint_view(&head, head.checkpoint_root, &CheckpointViewProof::decode(view_proof)?)?;
    let envelope = CheckpointRelationEnvelope::decode(envelope)?;
    envelope.verify(&view, plan)?;
    Ok(envelope)
}

/// Verify, ONCE PER HEAD, the checkpoint view every query at that head reads — the part of
/// [`decode_and_verify_checkpoint_read`] / [`decode_and_verify_checkpoint_delta_read`] that does not
/// depend on the query — so a verifier can cache the result per `(trusted_checkpoint_root,
/// trusted_head_root)` and check each query with [`decode_and_verify_checkpoint_read_over_view`].
///
/// With no `head` (no deltas since): the checkpoint artifact must carry the trusted checkpoint root
/// and be OF the trusted head, and `delta_views` must be empty. With `head = (artifact, proof)`: the
/// head checkpoint `C'` is verified from the trusted checkpoint, the deltas' term-bearing verifier
/// commitments (chain order) and the trusted head root. Then the view proof of the head checkpoint.
pub fn verify_checkpoint_read_view(
    checkpoint_artifact: &[u8],
    trusted_checkpoint_root: ProofHash,
    delta_views: &[&[u8]],
    trusted_head_root: ProofHash,
    head: Option<(&[u8], &[u8])>,
    view_proof: &[u8],
) -> io::Result<VerifiedCheckpointView> {
    let head = match head {
        None => {
            if !delta_views.is_empty() {
                return Err(invalid("a checkpoint read with deltas since needs the head checkpoint proof"));
            }
            let artifact = CheckpointArtifact::decode(checkpoint_artifact)?;
            if artifact.checkpoint_root != trusted_checkpoint_root {
                return Err(invalid("checkpoint is not the trusted one"));
            }
            if artifact.head_root != trusted_head_root {
                return Err(invalid("checkpoint is not of the trusted head"));
            }
            artifact
        }
        Some((head_artifact, head_proof)) => decode_and_verify_checkpoint(
            trusted_checkpoint_root,
            checkpoint_artifact,
            delta_views,
            trusted_head_root,
            head_artifact,
            head_proof,
        )?,
    };
    verify_checkpoint_view(&head, head.checkpoint_root, &CheckpointViewProof::decode(view_proof)?)
}

/// Verify one query's envelope against a view already verified by [`verify_checkpoint_read_view`]
/// (e.g. from a per-head cache), against the caller's OWN `plan`.
pub fn decode_and_verify_checkpoint_read_over_view(
    envelope: &[u8],
    view: &VerifiedCheckpointView,
    plan: &RelationProofPlan,
) -> io::Result<CheckpointRelationEnvelope> {
    let envelope = CheckpointRelationEnvelope::decode(envelope)?;
    envelope.verify(view, plan)?;
    Ok(envelope)
}
