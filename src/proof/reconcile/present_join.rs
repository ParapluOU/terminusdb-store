//! Two-atom self-join read over `present`:  `Triple(?x, p, ?y) & Triple(?y, q, ?z)`.
//!
//! Builds on the single-atom filter: it authenticates `present`'s id columns against the
//! reconciliation hashes, filters each atom by its constant predicate, then proves the join on
//! the shared variable — all on the one reconciliation transcript, so the joined answer stays
//! bound to the trusted head root.

use super::*;

// ===========================================================================
// Wave 3 — two-atom JOIN over present:  Triple(?x, p, ?y) & Triple(?y, q, ?z).
//
// The join key ?y is a term shared between atom-1's OBJECT and atom-2's SUBJECT. In hash
// space the key is a ~254-bit field element (no magnitude-sort join). We therefore bridge
// to the COMBINED-DICT id space — a small, dense, chain-wide canonical id per term — and
// reuse the audited magnitude-sort inner join:
//   * present's subject/object COMBINED ids are authenticated by an `id → hash` lookup
//     against the combined node/value dict whose result column is required to EQUAL the
//     reconciliation-authenticated hash column (`com_present_s`/`_o`). This pins each id to
//     the correct term (unique id ↔ hash; `hash_object` collision-resistant), so the ids
//     faithfully represent `present` without leaving the authenticated relation.
//   * each atom is the audited equality filter on the hash predicate column, compacting the
//     authenticated id columns to `R1 = [id_x, id_y]` (predicate p) and `R2 = [id_y, id_z]`
//     (predicate q).
//   * the audited complete-and-sound inner join joins `R1.col1 == R2.col0` on the small ids
//     (no 254-bit cap), delivering exactly the `[id_x, id_y, id_z]` solutions.
// Result ids resolve to canonical IRIs/values against the chain catalogs — deferred (report).
// ===========================================================================

/// `present`'s id-column authentication: the committed id column and the lookup pinning it to
/// the reconciliation-authenticated hash column.
#[derive(Clone, Debug)]
pub struct PresentIdAuth {
    pub com_id: ArkGT,
    pub lookup: HashLookupInnerProof,
}

/// A two-atom join proof over `present`.
#[derive(Clone, Debug)]
pub struct PresentJoinProof {
    pub auth_s: PresentIdAuth,
    pub auth_o: PresentIdAuth,
    pub filter1: PresentFilterProof,
    pub filter2: PresentFilterProof,
    pub inner: InnerJoinProof,
    pub com_join_out: Vec<ArkGT>,
    pub join_out_len: usize,
    pub join_log: usize,
}

/// Authenticate a `present` id column (already at 2^K) against the combined node/value dict:
/// the lookup's `com_v_hash` is later required to equal the reconciliation-authenticated hash.
pub(super) fn prove_id_auth(
    context: &StoreProverContext<'_>,
    id_col: &[ArkFr],
    real_n: usize,
    dict: &CombinedDictBuilt,
    transcript: &mut Transcript254,
) -> io::Result<PresentIdAuth> {
    let com_id = context.commit_column(id_col)?;
    let lookup = prove_hash_lookup_inner(
        context, id_col, &com_id, &dict.id_at_k, &dict.hash_at_k, &dict.com_id_at_k, &dict.com_hash_at_k,
        real_n, dict.n, transcript,
    )?;
    Ok(PresentIdAuth { com_id, lookup })
}

pub(super) fn verify_id_auth(
    reconcile_nv: &CombinedDictProof,
    real_n: usize,
    n_dict: usize,
    expected_hash_com: ArkGT,
    auth: &PresentIdAuth,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    verify_hash_lookup_inner(
        &auth.com_id, &reconcile_nv.com_id_at_k, &reconcile_nv.com_hash_at_k, real_n, n_dict, &auth.lookup, transcript,
    )?;
    if auth.lookup.com_v_hash != expected_hash_com {
        return Err(invalid("join: present id column is not authenticated to the reconciled hashes"));
    }
    Ok(())
}

pub(super) fn prove_present_join(
    context: &StoreProverContext<'_>,
    artifacts: &ReconcileArtifacts,
    const_p: ArkFr,
    const_q: ArkFr,
    transcript: &mut Transcript254,
) -> io::Result<(PresentJoinProof, PresentReadResult)> {
    let m = artifacts.present_len;
    // (1) Authenticate present's subject/object combined-dict ids.
    let auth_s = prove_id_auth(context, &artifacts.present_id_s, m, &artifacts.nv_dict, transcript)?;
    let auth_o = prove_id_auth(context, &artifacts.present_id_o, m, &artifacts.nv_dict, transcript)?;

    // (2) Two predicate atoms → id relations. atom1 Triple(?x,p,?y): x=subject, y=object.
    //     atom2 Triple(?y,q,?z): y=subject, z=object. Both compact [id_s, id_o].
    let id_sources = vec![artifacts.present_id_s.clone(), artifacts.present_id_o.clone()];
    let id_source_coms = vec![auth_s.com_id, auth_o.com_id];
    let (filter1, r1_cols) = prove_filter_part(
        context, &artifacts.present_p, &artifacts.com_present_p, const_p, m, &id_sources, &id_source_coms, transcript,
    )?;
    let (filter2, r2_cols) = prove_filter_part(
        context, &artifacts.present_p, &artifacts.com_present_p, const_q, m, &id_sources, &id_source_coms, transcript,
    )?;

    // (3) Inner join on the shared ?y: R1.col1 (id_y = object of atom1) == R2.col0 (id_y = subject of atom2).
    let r1 = RelationWitness {
        columns: r1_cols,
        commitments: filter1.com_out.clone(),
        log_domain: artifacts.working_log,
        n: filter1.out_len,
    };
    let r2 = RelationWitness {
        columns: r2_cols,
        commitments: filter2.com_out.clone(),
        log_domain: artifacts.working_log,
        n: filter2.out_len,
    };
    let join = prove_inner_join_with_keys_and_context(context, &r1, &r2, &[(1, 0)], transcript)?;
    let out_cols: Vec<Vec<ArkFr>> = join.columns.iter().map(|c| c[..join.n].to_vec()).collect();

    Ok((
        PresentJoinProof {
            auth_s,
            auth_o,
            filter1,
            filter2,
            inner: join.proof,
            com_join_out: join.commitments,
            join_out_len: join.n,
            join_log: join.log_domain,
        },
        PresentReadResult { columns: out_cols, len: join.n },
    ))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn verify_present_join(
    reconcile: &ReconcileProof,
    verified: &ReconcileVerified,
    head_nv: usize,
    const_p: ArkFr,
    const_q: ArkFr,
    proof: &PresentJoinProof,
    transcript: &mut Transcript254,
) -> io::Result<usize> {
    let m = verified.present_len;
    // (1) present id authentication against the reconciled hashes.
    verify_id_auth(&reconcile.combined_nv, m, head_nv, verified.com_present_s, &proof.auth_s, transcript)?;
    verify_id_auth(&reconcile.combined_nv, m, head_nv, verified.com_present_o, &proof.auth_o, transcript)?;

    // (2) The two predicate atoms.
    let id_source_coms = [proof.auth_s.com_id, proof.auth_o.com_id];
    let m1 = verify_filter_part(&verified.com_present_p, const_p, m, &id_source_coms, &proof.filter1, transcript)?;
    let m2 = verify_filter_part(&verified.com_present_p, const_q, m, &id_source_coms, &proof.filter2, transcript)?;

    // (3) The inner join.
    let r1 = RelationCommit { commitments: proof.filter1.com_out.clone(), log_domain: verified.working_log, n: m1 };
    let r2 = RelationCommit { commitments: proof.filter2.com_out.clone(), log_domain: verified.working_log, n: m2 };
    verify_inner_join_with_keys(
        &r1, &r2, &[(1, 0)], &proof.com_join_out, proof.join_out_len, proof.join_log, &proof.inner, transcript,
    )?;
    Ok(proof.join_out_len)
}

/// Prove a two-atom join read `Triple(?x, p, ?y) & Triple(?y, q, ?z)` over the reconciled
/// N-layer `present`. The delivered result is the `[id_x, id_y, id_z]` solutions as combined-dict
/// ids (resolution to canonical IRIs/values deferred).
pub fn prove_read_two_atom_join(
    layers: &[&LayerCommitment],
    const_p: ArkFr,
    const_q: ArkFr,
) -> io::Result<(ReconcileProof, PresentJoinProof, PresentReadResult)> {
    let context = StoreProverContext::cpu();
    let mut transcript = Transcript254::new(b"parture-delta-reconcile-envelope");
    let (reconcile, artifacts) = prove_reconcile_core(&context, layers, &mut transcript)?;
    let (join, result) = prove_present_join(&context, &artifacts, const_p, const_q, &mut transcript)?;
    Ok((reconcile, join, result))
}

/// Verify a two-atom join read. On success returns the authenticated solution count.
pub fn verify_read_two_atom_join(
    layers: &[&LayerCommitment],
    trusted_head_root: ProofHash,
    const_p: ArkFr,
    const_q: ArkFr,
    reconcile: &ReconcileProof,
    join: &PresentJoinProof,
) -> io::Result<usize> {
    let mut transcript = Transcript254::new(b"parture-delta-reconcile-envelope");
    let verified = verify_reconcile_core(layers, trusted_head_root, reconcile, &mut transcript)?;
    let head_nv = layers.last().unwrap().verifier_view().state.node_value_count as usize;
    verify_present_join(reconcile, &verified, head_nv, const_p, const_q, join, &mut transcript)
}

