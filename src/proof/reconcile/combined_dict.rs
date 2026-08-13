//! Cross-layer dictionary authentication: one combined id<->term table for the whole chain.
//!
//! Each layer commits only the dictionary entries IT introduced, so a term's id is minted in
//! exactly one layer. To resolve any id anywhere in the stack in a single root-authenticated
//! lookup, this module builds a COMBINED dictionary as the exact bag union of every layer's
//! incremental dict (node/value and predicate), each branch bound to its own layer
//! `commitment_root`, and relocated into one working domain. Reconciliation runs in this
//! combined-hash space, so a term hashes identically regardless of which layer minted it. It
//! also builds the combined `(id, value, datatype)` RANGE dictionary the present-BGP range
//! gadget compares against.

use super::*;

// -- Combined dictionary (bag union of every contributing layer's dict, or direct) --

pub(super) struct CombinedDictBuilt {
    pub(super) proof: CombinedDictProof,
    pub(super) hash_at_k: Vec<ArkFr>,
    pub(super) id_at_k: Vec<ArkFr>,
    pub(super) com_hash_at_k: ArkGT,
    pub(super) com_id_at_k: ArkGT,
    pub(super) n: usize,
}

pub(super) fn prove_combined_dict(
    context: &StoreProverContext<'_>,
    branches: &[RelationWitness],
    working_log: usize,
    transcript: &mut Transcript254,
) -> io::Result<CombinedDictBuilt> {
    if branches.is_empty() {
        return Err(invalid("reconcile: combined dict needs at least one contributing layer"));
    }
    let (union, hash_data, id_data, com_hash, com_id, log, n) = if branches.len() >= 2 {
        let res = prove_bag_union_with_context(context, branches, transcript)?;
        (
            Some(res.proof),
            res.columns[0].clone(),
            res.columns[1].clone(),
            res.commitments[0],
            res.commitments[1],
            res.log_domain,
            res.n,
        )
    } else {
        let b = &branches[0];
        (None, b.columns[0].clone(), b.columns[1].clone(), b.commitments[0], b.commitments[1], b.log_domain, b.n)
    };
    let hash_at_k = pad_to(&hash_data, working_log);
    let id_at_k = pad_to(&id_data, working_log);
    let com_hash_at_k = context.commit_column(&hash_at_k)?;
    let com_id_at_k = context.commit_column(&id_at_k)?;
    let reloc_hash = prove_column_relocation_with_context(
        context, &hash_data, &com_hash, &hash_at_k, &com_hash_at_k, transcript,
    )?;
    let reloc_id = prove_column_relocation_with_context(
        context, &id_data, &com_id, &id_at_k, &com_id_at_k, transcript,
    )?;
    Ok(CombinedDictBuilt {
        proof: CombinedDictProof {
            union,
            com_hash,
            com_id,
            log,
            n,
            reloc_hash,
            reloc_id,
            com_hash_at_k,
            com_id_at_k,
        },
        hash_at_k,
        id_at_k,
        com_hash_at_k,
        com_id_at_k,
        n,
    })
}

pub(super) fn verify_combined_dict(
    branches: &[RelationCommit],
    expected_n: usize,
    working_log: usize,
    proof: &CombinedDictProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    if branches.is_empty() {
        return Err(invalid("reconcile: combined dict needs at least one contributing layer"));
    }
    if proof.n != expected_n {
        return Err(invalid("reconcile: combined dict length mismatch"));
    }
    if branches.len() >= 2 {
        let union = proof
            .union
            .as_ref()
            .ok_or_else(|| invalid("reconcile: missing bag-union proof for a multi-layer dict"))?;
        verify_bag_union(
            branches,
            &[proof.com_hash, proof.com_id],
            expected_n,
            proof.log,
            union,
            transcript,
        )?;
    } else {
        if proof.union.is_some() {
            return Err(invalid("reconcile: unexpected bag-union proof for a single-layer dict"));
        }
        let b = &branches[0];
        if proof.com_hash != b.commitments[0]
            || proof.com_id != b.commitments[1]
            || proof.log != b.log_domain
            || proof.n != b.n
        {
            return Err(invalid("reconcile: single-layer combined dict is not the layer's root-bound dict"));
        }
    }
    verify_column_relocation(
        &proof.com_hash, proof.log, &proof.com_hash_at_k, working_log, &proof.reloc_hash, transcript,
    )?;
    verify_column_relocation(
        &proof.com_id, proof.log, &proof.com_id_at_k, working_log, &proof.reloc_id, transcript,
    )?;
    Ok(())
}

// -- Combined node/value RANGE dictionary `(id, value, datatype)` (Wave 6 range) --
//
// The audited object-range gadget authenticates each object id's ORDER-PRESERVING VALUE
// and DATATYPE from a committed `(id, value, datatype)` dict. Each chain layer already
// commits its incremental `node_value_dict_range_commitments = (Com(value), Com(datatype))`
// AND its id column (`node_value_dict_commitments.1`), all root-bound (`compute_commitment_root`
// binds the six dictionary commitments; `view.validate()` re-checks it). So the combined
// range dict is the bag union of every layer's incremental `(id, value, datatype)` rows —
// the SAME machinery as the `(hash, id)` combined dict — relocated to the working domain.
// (Sound under the same no-re-mint invariant the reconciliation already relies on: an id is
// defined in exactly one layer's incremental range, so it appears once in the union.)

pub(super) struct CombinedRangeDictBuilt {
    pub(super) proof: CombinedRangeDictProof,
    pub(super) id_at_k: Vec<ArkFr>,
    pub(super) value_at_k: Vec<ArkFr>,
    pub(super) datatype_at_k: Vec<ArkFr>,
    pub(super) com_id_at_k: ArkGT,
    pub(super) com_value_at_k: ArkGT,
    pub(super) com_datatype_at_k: ArkGT,
}

/// A combined `(id, value, datatype)` node/value range dictionary authenticated across the
/// chain and relocated to the working domain (parallels [`CombinedDictProof`]).
#[derive(Clone, Debug)]
pub struct CombinedRangeDictProof {
    /// Present iff two or more layers contribute a dict (else a single-layer direct dict).
    pub union: Option<BagUnionProof>,
    pub com_id: ArkGT,
    pub com_value: ArkGT,
    pub com_datatype: ArkGT,
    pub log: usize,
    pub n: usize,
    pub reloc_id: RelocationProof,
    pub reloc_value: RelocationProof,
    pub reloc_datatype: RelocationProof,
    pub com_id_at_k: ArkGT,
    pub com_value_at_k: ArkGT,
    pub com_datatype_at_k: ArkGT,
}

pub(super) fn prove_combined_range_dict(
    context: &StoreProverContext<'_>,
    branches: &[RelationWitness],
    working_log: usize,
    transcript: &mut Transcript254,
) -> io::Result<CombinedRangeDictBuilt> {
    if branches.is_empty() {
        return Err(invalid("reconcile range: combined dict needs at least one contributing layer"));
    }
    let (union, id_data, value_data, datatype_data, com_id, com_value, com_datatype, log, n) =
        if branches.len() >= 2 {
            let res = prove_bag_union_with_context(context, branches, transcript)?;
            (
                Some(res.proof),
                res.columns[0].clone(),
                res.columns[1].clone(),
                res.columns[2].clone(),
                res.commitments[0],
                res.commitments[1],
                res.commitments[2],
                res.log_domain,
                res.n,
            )
        } else {
            let b = &branches[0];
            (
                None,
                b.columns[0].clone(),
                b.columns[1].clone(),
                b.columns[2].clone(),
                b.commitments[0],
                b.commitments[1],
                b.commitments[2],
                b.log_domain,
                b.n,
            )
        };
    let id_at_k = pad_to(&id_data, working_log);
    let value_at_k = pad_to(&value_data, working_log);
    let datatype_at_k = pad_to(&datatype_data, working_log);
    let com_id_at_k = context.commit_column(&id_at_k)?;
    let com_value_at_k = context.commit_column(&value_at_k)?;
    let com_datatype_at_k = context.commit_column(&datatype_at_k)?;
    let reloc_id = prove_column_relocation_with_context(
        context, &id_data, &com_id, &id_at_k, &com_id_at_k, transcript,
    )?;
    let reloc_value = prove_column_relocation_with_context(
        context, &value_data, &com_value, &value_at_k, &com_value_at_k, transcript,
    )?;
    let reloc_datatype = prove_column_relocation_with_context(
        context, &datatype_data, &com_datatype, &datatype_at_k, &com_datatype_at_k, transcript,
    )?;
    Ok(CombinedRangeDictBuilt {
        proof: CombinedRangeDictProof {
            union,
            com_id,
            com_value,
            com_datatype,
            log,
            n,
            reloc_id,
            reloc_value,
            reloc_datatype,
            com_id_at_k,
            com_value_at_k,
            com_datatype_at_k,
        },
        id_at_k,
        value_at_k,
        datatype_at_k,
        com_id_at_k,
        com_value_at_k,
        com_datatype_at_k,
    })
}

pub(super) fn verify_combined_range_dict(
    branches: &[RelationCommit],
    expected_n: usize,
    working_log: usize,
    proof: &CombinedRangeDictProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    if branches.is_empty() {
        return Err(invalid("reconcile range: combined dict needs at least one contributing layer"));
    }
    if proof.n != expected_n {
        return Err(invalid("reconcile range: combined dict length mismatch"));
    }
    if branches.len() >= 2 {
        let union = proof
            .union
            .as_ref()
            .ok_or_else(|| invalid("reconcile range: missing bag-union proof for a multi-layer dict"))?;
        verify_bag_union(
            branches,
            &[proof.com_id, proof.com_value, proof.com_datatype],
            expected_n,
            proof.log,
            union,
            transcript,
        )?;
    } else {
        if proof.union.is_some() {
            return Err(invalid("reconcile range: unexpected bag-union proof for a single-layer dict"));
        }
        let b = &branches[0];
        if proof.com_id != b.commitments[0]
            || proof.com_value != b.commitments[1]
            || proof.com_datatype != b.commitments[2]
            || proof.log != b.log_domain
            || proof.n != b.n
        {
            return Err(invalid("reconcile range: single-layer combined dict is not the layer's root-bound dict"));
        }
    }
    verify_column_relocation(
        &proof.com_id, proof.log, &proof.com_id_at_k, working_log, &proof.reloc_id, transcript,
    )?;
    verify_column_relocation(
        &proof.com_value, proof.log, &proof.com_value_at_k, working_log, &proof.reloc_value, transcript,
    )?;
    verify_column_relocation(
        &proof.com_datatype, proof.log, &proof.com_datatype_at_k, working_log, &proof.reloc_datatype, transcript,
    )?;
    Ok(())
}

/// Build the prover-side combined range-dict branches: per layer with a non-empty
/// incremental node/value dict, its `(id, value, datatype)` columns + root-bound commitments.
pub(super) fn range_dict_branches(layers: &[&LayerCommitment]) -> io::Result<Vec<RelationWitness>> {
    let mut branches = Vec::new();
    for layer in layers {
        if layer.node_value_catalog.is_empty() {
            continue;
        }
        let (_h, id) = layer.node_value_dictionary_columns()?;
        let value = layer.node_value_dictionary_value_column()?;
        let datatype = layer.node_value_dictionary_datatype_column()?;
        let n = layer.node_value_catalog.len();
        branches.push(RelationWitness {
            columns: vec![id, value, datatype],
            commitments: vec![
                layer.node_value_dict_commitments.1,
                layer.node_value_dict_range_commitments.0,
                layer.node_value_dict_range_commitments.1,
            ],
            log_domain: dict_log(n)?,
            n,
        });
    }
    Ok(branches)
}

/// Build the verifier-side combined range-dict branches (commitments only, from the roots).
pub(super) fn range_dict_branch_commits(views: &[VerifierCommitmentRef<'_>]) -> io::Result<Vec<RelationCommit>> {
    let mut branches = Vec::new();
    for view in views {
        let nv_incr = (view.state.node_value_count - view.state.parent_node_value_count) as usize;
        if nv_incr == 0 {
            continue;
        }
        branches.push(RelationCommit {
            commitments: vec![
                view.node_value_dict_commitments.1,
                view.node_value_dict_range_commitments.0,
                view.node_value_dict_range_commitments.1,
            ],
            log_domain: dict_log(nv_incr)?,
            n: nv_incr,
        });
    }
    Ok(branches)
}

