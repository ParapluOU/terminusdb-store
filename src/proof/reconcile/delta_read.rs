//! Delta reads: `AddedTriple` / `DeletedTriple` over the HEAD delta layer.
//!
//! Not every WOQL read is over the reconciled current set — `AddedTriple`/`DeletedTriple` ask
//! what a specific commit ADDED or REMOVED. This module runs the same authenticated BGP
//! machinery, but sourced from the head layer's committed additions or removals group
//! (selected by [`DeltaSource`]) instead of `present`. It reuses the combined node/value +
//! predicate dictionaries to authenticate and resolve ids, on one transcript domain-separated
//! from the reconciliation envelope, so the read is bound to the trusted head root.

use super::*;

// ===========================================================================
// DELTA-READS — `AddedTriple` / `DeletedTriple` over the HEAD delta layer.
//
// `AddedTriple(S,P,O)` reads THIS commit's ADDITIONS file (the head layer's `additions`,
// "no deeper"); `DeletedTriple(S,P,O)` reads its `removals` file. Wave 0's commitment diet
// stopped committing a materialized head, but each layer STILL commits its additions/removals
// id columns root-bound (`column_commitments`: additions S/P/O at `[0..3]`, removals S/P/O at
// `[3..6]`) AND its INCREMENTAL dictionary. An added/removed triple may reference an
// ANCESTOR-defined term (e.g. a predicate declared in the base), so — exactly like the
// reconciliation — we authenticate its canonical hashes against the COMBINED chain dictionary
// (the bag-union of every layer's incremental dict), which resolves any id minted anywhere in
// the chain in one root-authenticated table.
//
// This is SIMPLER than reconciliation: no multiset union/difference. The answer is a
// conjunctive BGP filtered over ONE root-committed group (the head's additions OR removals). We:
//   1. build the combined node/value + predicate dictionaries across the chain, so ancestor
//      constants resolve (reused `prove_combined_dict`);
//   2. authenticate the head group's S/O/P id columns → canonical HASH columns via the audited
//      `prove_group_hash` (relocate each ROOT-committed `column_commitments` column to the
//      working domain, then an id→hash lookup against the combined dict) — this TIES the BGP's
//      source relation to the trusted head root, so a fabricated added/removed row cannot be
//      proved and an inherited/materialized row (absent from this commit's group) never appears;
//   3. run the audited `present` BGP over those authenticated hash/id columns — the head group
//      slots in exactly where `present` does — delivering resolved canonical IRIs/typed literals.
//
// SCOPE (mirrors the `present` BGP path): the group must have `>= 2` rows (the relocatable
// domain floor), every atom's predicate is a constant, each atom keeps `>= 1` variable, and the
// result columns are combined-dict node/value ids resolved against the chain catalogs. Cross-
// source joins (additions ⋈ materialized) are out of scope. All sub-proofs share ONE transcript,
// domain-separated from the reconciliation envelope, so the read is bound to the head root.
// ===========================================================================

pub(super) const DELTA_READ_DOMAIN: &[u8] = b"parture/proof-of-woql/delta-read/v1";
pub(super) const DELTA_READ_TRANSCRIPT: &[u8] = b"parture-delta-read-envelope";

/// Which committed delta file of the HEAD layer a delta-read reads: `Additions`
/// (`AddedTriple`) or `Removals` (`DeletedTriple`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeltaSource {
    /// `AddedTriple`: the head commit's additions file (`column_commitments[0..3]`).
    Additions,
    /// `DeletedTriple`: the head commit's removals file (`column_commitments[3..6]`).
    Removals,
}

impl DeltaSource {
    /// The `column_commitments` offset of this source's S column (S/P/O follow).
    fn column_offset(self) -> usize {
        match self {
            DeltaSource::Additions => 0,
            DeltaSource::Removals => 3,
        }
    }
    fn transcript_tag(self) -> &'static [u8] {
        match self {
            DeltaSource::Additions => b"additions",
            DeltaSource::Removals => b"removals",
        }
    }
}

/// A delta-read proof: the combined chain dictionaries, the head group's three authenticated
/// canonical-hash-column bindings (each tied to a root-committed additions/removals column), and
/// the BGP over them. Bound to the trusted head root through the per-layer roots + group hashes.
#[derive(Clone, Debug)]
pub struct DeltaReadProof {
    pub combined_nv: CombinedDictProof,
    pub combined_pred: CombinedDictProof,
    pub working_log: usize,
    /// The head group's S/O/P id→hash authentications (S,O vs the combined node/value dict, P vs
    /// the combined predicate dict), each relocated from the root-committed group column.
    pub group: CommittedGroupProof,
    pub group_len: usize,
    pub bgp: PresentBgpProof,
}

/// The working power-of-two domain a delta-read runs at: large enough for the combined
/// node/value + predicate dictionaries and the chosen group's column domain. Recomputable by
/// both prover and verifier from public counts alone.
pub(super) fn delta_working_log(head_nv: usize, head_pred: usize, group_len: usize) -> io::Result<usize> {
    Ok(dict_log(head_nv)?.max(dict_log(head_pred)?).max(group_col_log(group_len)?))
}

/// Prover-side combined-dict branches (node/value, predicate): one per layer with a non-empty
/// incremental dict, its `(hash, id)` columns + root-bound commitments. Identical to the inline
/// construction in [`prove_reconcile_core`].
pub(super) fn nv_pred_branches(
    layers: &[&LayerCommitment],
) -> io::Result<(Vec<RelationWitness>, Vec<RelationWitness>)> {
    let mut nv = Vec::new();
    let mut pred = Vec::new();
    for layer in layers {
        if !layer.node_value_catalog.is_empty() {
            let (h, id) = layer.node_value_dictionary_columns()?;
            let n = layer.node_value_catalog.len();
            nv.push(RelationWitness {
                columns: vec![h, id],
                commitments: vec![layer.node_value_dict_commitments.0, layer.node_value_dict_commitments.1],
                log_domain: dict_log(n)?,
                n,
            });
        }
        if !layer.predicate_catalog.is_empty() {
            let (h, id) = layer.predicate_dictionary_columns()?;
            let n = layer.predicate_catalog.len();
            pred.push(RelationWitness {
                columns: vec![h, id],
                commitments: vec![layer.predicate_dict_commitments.0, layer.predicate_dict_commitments.1],
                log_domain: dict_log(n)?,
                n,
            });
        }
    }
    Ok((nv, pred))
}

/// Verifier-side combined-dict branch commitments (node/value, predicate), from the roots.
/// Identical to the inline construction in [`verify_reconcile_core`].
pub(super) fn nv_pred_branch_commits(
    views: &[VerifierCommitmentRef<'_>],
) -> io::Result<(Vec<RelationCommit>, Vec<RelationCommit>)> {
    let mut nv = Vec::new();
    let mut pred = Vec::new();
    for view in views {
        let nv_incr = (view.state.node_value_count - view.state.parent_node_value_count) as usize;
        if nv_incr > 0 {
            nv.push(RelationCommit {
                commitments: vec![view.node_value_dict_commitments.0, view.node_value_dict_commitments.1],
                log_domain: dict_log(nv_incr)?,
                n: nv_incr,
            });
        }
        let pred_incr = (view.state.predicate_count - view.state.parent_predicate_count) as usize;
        if pred_incr > 0 {
            pred.push(RelationCommit {
                commitments: vec![view.predicate_dict_commitments.0, view.predicate_dict_commitments.1],
                log_domain: dict_log(pred_incr)?,
                n: pred_incr,
            });
        }
    }
    Ok((nv, pred))
}

pub(super) fn prove_delta_read_core(
    context: &StoreProverContext<'_>,
    layers: &[&LayerCommitment],
    source: DeltaSource,
    atoms: &[PresentBgpAtom],
    joins: &[JoinStep],
    transcript: &mut Transcript254,
) -> io::Result<(DeltaReadProof, PresentReadResult)> {
    let states: Vec<&crate::proof::commitment::CommitmentState> = layers.iter().map(|l| &l.state).collect();
    validate_chain_states(&states)?;
    let head = layers.last().unwrap();
    let head_nv = head.state.node_value_count as usize;
    let head_pred = head.state.predicate_count as usize;

    // The chosen HEAD group (this commit's added or removed rows), root-committed.
    let records: &[ProofRecord] = match source {
        DeltaSource::Additions => head.additions.as_slice(),
        DeltaSource::Removals => head.removals.as_slice(),
    };
    let group_len = records.len();
    if group_len < 2 {
        return Err(invalid(
            "delta-read: the head delta group must have >= 2 rows (single-row groups unsupported this pass)",
        ));
    }
    let col_offset = source.column_offset();
    let col_log = group_col_log(group_len)?;
    let working_log = delta_working_log(head_nv, head_pred, group_len)?;
    let k_size = 1usize << working_log;

    transcript.append_bytes(b"pow/delta-read/domain", DELTA_READ_DOMAIN);
    transcript.append_bytes(b"pow/delta-read/source", source.transcript_tag());
    for layer in layers {
        transcript.append_bytes(b"pow/delta-read/layer-root", &layer.state.commitment_root);
    }

    // Combined node/value + predicate dictionaries (bag union across the whole chain), so an
    // ancestor-defined constant term resolves against one root-authenticated table.
    let (nv_branches, pred_branches) = nv_pred_branches(layers)?;
    let combined_nv = prove_combined_dict(context, &nv_branches, working_log, transcript)?;
    let combined_pred = prove_combined_dict(context, &pred_branches, working_log, transcript)?;
    let combined_nv_proof = combined_nv.proof.clone();
    let combined_pred_proof = combined_pred.proof.clone();

    // Authenticate the head group's S/O/P hash columns against the combined dicts, each tied to
    // its ROOT-committed `column_commitments` column (S,O vs node/value; P vs predicate).
    let cc = &head.column_commitments;
    let s = prove_group_hash(
        context, records, |r| r.subject_id, &cc[col_offset], col_log, group_len, working_log,
        &combined_nv, transcript,
    )?;
    let o = prove_group_hash(
        context, records, |r| r.object_id, &cc[col_offset + 2], col_log, group_len, working_log,
        &combined_nv, transcript,
    )?;
    let p = prove_group_hash(
        context, records, |r| r.predicate_id, &cc[col_offset + 1], col_log, group_len, working_log,
        &combined_pred, transcript,
    )?;

    // The group's combined-dict id columns at 2^K (identical to `prove_group_hash`'s relocated
    // columns; recomputed here to feed the BGP's kept/join columns).
    let mut present_id_s = vec![f0(); k_size];
    let mut present_id_p = vec![f0(); k_size];
    let mut present_id_o = vec![f0(); k_size];
    for (i, r) in records.iter().enumerate() {
        present_id_s[i] = ark(r.subject_id);
        present_id_p[i] = ark(r.predicate_id);
        present_id_o[i] = ark(r.object_id);
    }

    // The head group slots in exactly where `present` does: its authenticated hash columns are
    // the equality-filter source, its id columns the kept/join source.
    let artifacts = ReconcileArtifacts {
        present_s: s.1,
        present_p: p.1,
        present_o: o.1,
        com_present_s: s.0.lookup.com_v_hash,
        com_present_p: p.0.lookup.com_v_hash,
        com_present_o: o.0.lookup.com_v_hash,
        present_id_s,
        present_id_o,
        present_id_p,
        working_log,
        present_len: group_len,
        nv_dict: combined_nv,
        pred_dict: combined_pred,
        present_rows: Vec::new(),
    };

    let (bgp, result) = prove_present_bgp(context, layers, &artifacts, atoms, joins, transcript)?;

    Ok((
        DeltaReadProof {
            combined_nv: combined_nv_proof,
            combined_pred: combined_pred_proof,
            working_log,
            group: CommittedGroupProof { s: s.0, o: o.0, p: p.0 },
            group_len,
            bgp,
        },
        result,
    ))
}

/// Prove a conjunctive `AddedTriple`/`DeletedTriple` BGP over the HEAD delta layer's committed
/// additions (`DeltaSource::Additions`) or removals (`DeltaSource::Removals`), with every
/// constant/term authenticated to the trusted head root via the combined chain dictionaries.
/// `layers` is ordered base..head (`len >= 2`). The delivered result columns are combined-dict
/// node/value ids ([`verify_and_resolve_delta_read`] resolves them to canonical answers).
pub fn prove_delta_read(
    layers: &[&LayerCommitment],
    source: DeltaSource,
    atoms: &[PresentBgpAtom],
    joins: &[JoinStep],
) -> io::Result<(DeltaReadProof, PresentReadResult)> {
    let context = StoreProverContext::cpu();
    let mut transcript = Transcript254::new(DELTA_READ_TRANSCRIPT);
    prove_delta_read_core(&context, layers, source, atoms, joins, &mut transcript)
}

pub(super) fn verify_delta_read_core(
    layers: &[&LayerCommitment],
    trusted_head_root: ProofHash,
    source: DeltaSource,
    atoms: &[PresentBgpAtom],
    joins: &[JoinStep],
    proof: &DeltaReadProof,
    transcript: &mut Transcript254,
) -> io::Result<(Vec<ArkGT>, usize, usize)> {
    let views: Vec<VerifierCommitmentRef<'_>> = layers.iter().map(|l| l.verifier_view()).collect();
    // (1) Each layer authenticated against ITS OWN commitment root.
    for view in &views {
        view.validate()?;
    }
    // (2) Whole parent chain + trusted head anchor (splice/reorder/drop/foreign-proof).
    let states: Vec<&crate::proof::commitment::CommitmentState> = views.iter().map(|v| v.state).collect();
    validate_chain_states(&states)?;
    if states.last().unwrap().commitment_root != trusted_head_root {
        return Err(invalid("delta-read: head root does not match the trusted anchor"));
    }
    let head = *views.last().unwrap();
    let head_nv = head.state.node_value_count as usize;
    let head_pred = head.state.predicate_count as usize;
    let group_len = match source {
        DeltaSource::Additions => head.state.addition_count as usize,
        DeltaSource::Removals => head.state.removal_count as usize,
    };
    if group_len < 2 {
        return Err(invalid(
            "delta-read: the head delta group must have >= 2 rows (single-row groups unsupported this pass)",
        ));
    }
    if group_len != proof.group_len {
        return Err(invalid("delta-read: claimed group length disagrees with the head delta count"));
    }
    let col_offset = source.column_offset();
    let col_log = group_col_log(group_len)?;
    let working_log = delta_working_log(head_nv, head_pred, group_len)?;
    if working_log != proof.working_log {
        return Err(invalid("delta-read: working domain mismatch"));
    }

    transcript.append_bytes(b"pow/delta-read/domain", DELTA_READ_DOMAIN);
    transcript.append_bytes(b"pow/delta-read/source", source.transcript_tag());
    for view in &views {
        transcript.append_bytes(b"pow/delta-read/layer-root", &view.state.commitment_root);
    }

    // Combined dictionaries (verifier side).
    let (nv_branches, pred_branches) = nv_pred_branch_commits(&views)?;
    verify_combined_dict(&nv_branches, head_nv, working_log, &proof.combined_nv, transcript)?;
    verify_combined_dict(&pred_branches, head_pred, working_log, &proof.combined_pred, transcript)?;

    // Authenticate the head group's S/O/P hash columns against the ROOT-committed group columns.
    let cc = head.column_commitments;
    verify_group_hash(&cc[col_offset], col_log, group_len, working_log, &proof.combined_nv, &proof.group.s, transcript)?;
    verify_group_hash(&cc[col_offset + 2], col_log, group_len, working_log, &proof.combined_nv, &proof.group.o, transcript)?;
    verify_group_hash(&cc[col_offset + 1], col_log, group_len, working_log, &proof.combined_pred, &proof.group.p, transcript)?;

    // The head group as the authenticated `present`-shaped relation.
    let verified = ReconcileVerified {
        com_present_s: proof.group.s.lookup.com_v_hash,
        com_present_p: proof.group.p.lookup.com_v_hash,
        com_present_o: proof.group.o.lookup.com_v_hash,
        working_log,
        present_len: group_len,
    };

    verify_present_bgp(
        layers, &proof.combined_nv, &proof.combined_pred, &verified, head_nv, atoms, joins,
        &proof.bgp, transcript,
    )
}

/// Verify a delta-read BGP against the trusted head root. On success returns the authenticated
/// result row count. `atoms`/`joins` MUST be the VERIFIER's own compiled query (anti-substitution
/// is enforced one level up in the compiler routing).
pub fn verify_delta_read(
    layers: &[&LayerCommitment],
    trusted_head_root: ProofHash,
    source: DeltaSource,
    atoms: &[PresentBgpAtom],
    joins: &[JoinStep],
    proof: &DeltaReadProof,
) -> io::Result<usize> {
    let mut transcript = Transcript254::new(DELTA_READ_TRANSCRIPT);
    let (_coms, result_m, _log) =
        verify_delta_read_core(layers, trusted_head_root, source, atoms, joins, proof, &mut transcript)?;
    Ok(result_m)
}

/// Verify a delta-read BGP AND resolve its result bindings to `CanonicalObject`s (real IRIs /
/// typed literals) against the chain catalogs — the readable answer, no re-run. Every result
/// column is a combined-dict node/value id (predicates are constants).
pub fn verify_and_resolve_delta_read(
    layers: &[&LayerCommitment],
    trusted_head_root: ProofHash,
    source: DeltaSource,
    atoms: &[PresentBgpAtom],
    joins: &[JoinStep],
    proof: &DeltaReadProof,
    claimed_columns: &[Vec<ArkFr>],
) -> io::Result<Vec<Vec<CanonicalObject>>> {
    let mut transcript = Transcript254::new(DELTA_READ_TRANSCRIPT);
    let (result_coms, result_m, result_log_domain) =
        verify_delta_read_core(layers, trusted_head_root, source, atoms, joins, proof, &mut transcript)?;
    let views: Vec<VerifierCommitmentRef<'_>> = layers.iter().map(|l| l.verifier_view()).collect();
    let kinds = vec![ResolveKind::NodeValueId; result_coms.len()];
    resolve_result_columns(&views, claimed_columns, &result_coms, result_m, result_log_domain, &kinds)
}

