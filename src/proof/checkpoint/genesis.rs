//! The genesis checkpoint `C_0` of a base layer, proven against the base layer's own commitments.
//!
//! Genesis is a one-time `O(n)` step: the verifier holds the base layer's TERM-BEARING verifier
//! view (its complete, root-bound dictionary catalogs) and receives the base layer's clear present
//! id triples, which it re-commits against the base layer's root-bound materialized column
//! commitments. From those it rebuilds every component of `C_0` and requires the artifact to equal
//! it. Every later checkpoint is proven inductively.

use std::io;

use super::state::{commit_present, CheckpointArtifact, CheckpointState};
use super::{GenesisProof, IdTriple};
use crate::proof::argument::StoreProverContext;
use crate::proof::commitment::{LayerCommitment, VerifierCommitmentRef};
use crate::proof::{invalid, CanonicalObject, ProofHash, ProofRecord};

/// The base layer's dictionary in id order (ids must be exactly `1..=count`).
pub(super) fn dictionary_by_id<T: Clone>(
    catalog: &[ProofRecord],
    first: u64,
    count: u64,
    select: impl Fn(&ProofRecord) -> (u64, T),
) -> io::Result<Vec<T>> {
    let mut entries: Vec<(u64, T)> = catalog.iter().map(&select).collect();
    entries.sort_by_key(|(id, _)| *id);
    let expected = count
        .checked_sub(first - 1)
        .ok_or_else(|| invalid("checkpoint dictionary interval is inverted"))?;
    if entries.len() as u64 != expected
        || entries.iter().enumerate().any(|(i, (id, _))| *id != first + i as u64)
    {
        return Err(invalid("checkpoint dictionary ids are not the layer's dense id interval"));
    }
    Ok(entries.into_iter().map(|(_, term)| term).collect())
}

pub(super) fn node_values_by_id(catalog: &[ProofRecord], first: u64, count: u64) -> io::Result<Vec<CanonicalObject>> {
    dictionary_by_id(catalog, first, count, |r| (r.object_id, r.triple.object.clone()))
}

pub(super) fn predicates_by_id(catalog: &[ProofRecord], first: u64, count: u64) -> io::Result<Vec<Vec<u8>>> {
    dictionary_by_id(catalog, first, count, |r| (r.predicate_id, r.triple.predicate.clone()))
}

/// The genesis checkpoint of a BASE layer (no parent).
pub fn genesis_checkpoint(base: &LayerCommitment) -> io::Result<(CheckpointState, GenesisProof)> {
    genesis_checkpoint_with_context(&StoreProverContext::cpu(), base)
}

/// [`genesis_checkpoint`] with explicit prover arithmetic.
pub fn genesis_checkpoint_with_context(
    context: &StoreProverContext<'_>,
    base: &LayerCommitment,
) -> io::Result<(CheckpointState, GenesisProof)> {
    if base.state.parent_state_root.is_some() {
        return Err(invalid("a genesis checkpoint is taken of a base layer"));
    }
    let present: Vec<IdTriple> =
        base.materialized.iter().map(|r| [r.subject_id, r.predicate_id, r.object_id]).collect();
    let nv_terms = node_values_by_id(&base.node_value_catalog, 1, base.state.node_value_count)?;
    let pred_terms = predicates_by_id(&base.predicate_catalog, 1, base.state.predicate_count)?;
    let state = CheckpointState::from_parts(
        context,
        base.resolution_root(),
        base.state.commitment_root,
        None,
        present.clone(),
        nv_terms,
        pred_terms,
    )?;
    Ok((state, GenesisProof { present }))
}

/// Verify a genesis checkpoint against the base layer's term-bearing verifier view and the
/// trusted base root `X_0`.
pub fn verify_genesis_checkpoint(
    base: VerifierCommitmentRef<'_>,
    trusted_head_root: ProofHash,
    artifact: &CheckpointArtifact,
    proof: &GenesisProof,
) -> io::Result<()> {
    base.require_terms_present("genesis checkpoint")?;
    base.validate()?;
    if base.state.parent_state_root.is_some() {
        return Err(invalid("a genesis checkpoint is taken of a base layer"));
    }
    if base.resolution_root() != trusted_head_root {
        return Err(invalid("genesis base layer does not match the trusted root"));
    }
    artifact.validate()?;
    if artifact.prev_checkpoint_root.is_some() {
        return Err(invalid("a genesis checkpoint has no previous checkpoint"));
    }
    let present = &proof.present;
    if present.len() as u64 != base.state.materialized_count {
        return Err(invalid("genesis present triples disagree with the base layer's count"));
    }
    if present.windows(2).any(|w| w[0] >= w[1]) {
        return Err(invalid("genesis present triples are not strictly SPO-sorted"));
    }
    let nv_count = base.state.node_value_count;
    let pred_count = base.state.predicate_count;
    if present
        .iter()
        .any(|t| t[0] == 0 || t[1] == 0 || t[2] == 0 || t[0] > nv_count || t[2] > nv_count || t[1] > pred_count)
    {
        return Err(invalid("genesis present triple id lies outside the base dictionary"));
    }
    let context = StoreProverContext::cpu();
    if &commit_present(&context, present)? != base.materialized_column_commitments {
        return Err(invalid("genesis present triples are not the base layer's materialized triples"));
    }
    let nv_terms = node_values_by_id(base.node_value_catalog, 1, nv_count)?;
    let pred_terms = predicates_by_id(base.predicate_catalog, 1, pred_count)?;
    let expected = CheckpointState::from_parts(
        &context,
        trusted_head_root,
        base.state.commitment_root,
        None,
        present.clone(),
        nv_terms,
        pred_terms,
    )?
    .artifact;
    if &expected != artifact {
        return Err(invalid("genesis checkpoint does not commit the base layer's state"));
    }
    Ok(())
}
