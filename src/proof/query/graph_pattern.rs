//! Q2 — proving a 2-atom "friend of a friend" pattern: `t(?x,<P>,?y), t(?y,<P>,?z)`.
//!
//! This proves the complete result of a two-hop self-join on one predicate: every `(x,y,z)`
//! where both `(x,y)` and `(y,z)` are `<P>` triples. It reuses Q1's dictionary+filter to get
//! the predicate-`P` pair relation, then adds two "relocation" proofs (moving the filter
//! output into the join's coordinate system) and one complete self-join proof — all under a
//! single shared transcript anchored to `commitment_root`. The result is the joined `(X,Y,Z)`
//! columns and a proof the verifier checks without ever running the join itself.

use super::*;

const GRAPH2_DOMAIN: &[u8] = b"parture/proof-of-woql/q2-graph-pattern2-by-iri/v0";

/// A complete 2-atom graph-pattern proof, composed under one shared transcript and
/// anchored to `commitment_root`:
///   * `membership` — Q3 dictionary binding of the predicate IRI to its id.
///   * `filter` (+ `com_sout`/`com_oout`) — Q1 complete filter of the layer by that
///     id into `F = (Sf, Of)` = the predicate-P `(subject, object)` pairs, committed
///     at the TABLE domain and root-anchored via the filter argument.
///   * `relocate_s`/`relocate_o` — cross-domain equality tying the filter output
///     (table domain) to the self-join input `Sf`/`Of` (join domain), positionally.
///   * `join` (+ `join_commitments`) — the complete self-join of `F` on `Of == Sf`
///     into `(X, Y, Z)`.
/// `n_f`/`n_u` are the (sub-proof-pinned) filter-result count and distinct-key count,
/// carried so the verifier can re-derive the join domain.
#[derive(Clone, Debug)]
pub struct GraphPattern2Proof {
    pub membership: DictProof,
    pub filter: FullFilterProof,
    pub com_sout: ArkGT,
    pub com_oout: ArkGT,
    pub relocate_s: RelocationProof,
    pub relocate_o: RelocationProof,
    pub join: SelfJoinCompleteProof,
    pub join_commitments: SelfJoinCommitments,
    pub n_f: usize,
    pub n_u: usize,
}

/// Prove the COMPLETE result of the 2-atom graph pattern
/// `t(?x, <predicate_term>, ?y), t(?y, <predicate_term>, ?z)` against a materialized
/// base layer: every `(x, y, z)` such that `(x, y)` and `(y, z)` are both triples
/// with predicate `<predicate_term>`. Composes, under one shared transcript anchored
/// to `commitment_root`: Q3 dict binding (IRI -> id) + Q1 complete filter (id -> F) +
/// cross-domain relocation (F -> join input) + the complete self-join (F -> result).
/// Returns the combined proof, the result commitments `(Com X, Com Y, Com Z)`, the
/// result length `m_J`, and the REVEALED resolved id (authenticated by membership).
pub fn prove_layer_graph_pattern2(
    layer: &LayerCommitment,
    predicate_term: &[u8],
) -> io::Result<(GraphPattern2Proof, ArkGT, ArkGT, ArkGT, usize, u64)> {
    prove_layer_graph_pattern2_with_context(&StoreProverContext::cpu(), layer, predicate_term)
}

pub fn prove_layer_graph_pattern2_with_context(
    context: &StoreProverContext<'_>,
    layer: &LayerCommitment,
    predicate_term: &[u8],
) -> io::Result<(GraphPattern2Proof, ArkGT, ArkGT, ArkGT, usize, u64)> {
    prove_layer_graph_pattern2_with_source(
        context,
        layer,
        ColumnSource::Materialized,
        predicate_term,
    )
}

/// Prove the 2-hop self-join over the columns selected by `column_source`. Materialized is
/// the byte-identical legacy behavior; a delta source reads the layer's own addition/removal
/// columns (a 2-hop join WITHIN one delta set).
pub fn prove_layer_graph_pattern2_with_source(
    context: &StoreProverContext<'_>,
    layer: &LayerCommitment,
    column_source: ColumnSource,
    predicate_term: &[u8],
) -> io::Result<(GraphPattern2Proof, ArkGT, ArkGT, ArkGT, usize, u64)> {
    let (s, p, o, n) = source_layer_columns(layer, column_source)?;
    let predicate_id = resolve_predicate_id(layer, predicate_term)?;

    let (dict_hash, dict_id) = layer.predicate_dictionary_columns()?;
    let com_dict_hash = layer.predicate_dict_commitments.0;
    let com_dict_id = layer.predicate_dict_commitments.1;

    let mut transcript = Transcript254::new(GRAPH2_DOMAIN);
    bind_statement(&mut transcript, layer, column_source, predicate_term, predicate_id);

    // (1) Q3 membership: (hash_term(IRI), id) is a genuine dictionary row.
    let q_hash = hash_term(predicate_term);
    let id = ArkFr::from_u64(predicate_id);
    let membership = prove_dict_membership_with_context(
        context, &dict_hash,
        &dict_id,
        &com_dict_hash,
        &com_dict_id,
        q_hash,
        id,
        &mut transcript,
    )?;

    // (2) Q1 complete filter: F = (Sf, Of) = the predicate-P (subject, object) pairs.
    let [com_s, com_p, com_o] = source_column_commitments(layer.verifier_view(), column_source);
    let pred = ArkFr::from_u64(predicate_id);
    let (filter, com_sout, com_oout, m) =
        prove_filter_with_context(
            context, &s, &p, &o, pred, n, &com_s, &com_p, &com_o, &mut transcript,
        )?;
    // Recompute the dense filter output (same witness prove_filter used) to obtain the
    // F columns; `s_out`/`o_out` are length `2^a` (table domain), dense in `[0, m)`.
    let (sel, _w) = honest_witness(&p, pred, n);
    let (s_out, o_out, _m) = compact_selected(&s, &o, &sel);
    let sf = &s_out[..m];
    let of = &o_out[..m];

    // (3) Complete self-join of F on Of == Sf, committed at the join domain.
    let (x, y, z, rho_l, rho_r, u, w_l, w_r, m_j) = compute_self_join(sf, of);
    let n_u = u.len();
    let n_join = self_join_domain(m, m_j, n_u)?;
    let jcom = self_join_commitments_with_context(
        context, sf, of, &x, &y, &z, &rho_l, &rho_r, &u, &w_l, &w_r,
    )?;

    // (4) Relocate F from the table domain to the join domain (positional equality).
    let target_sf = pad(sf, n_join);
    let target_of = pad(of, n_join);
    let relocate_s = prove_column_relocation_with_context(
        context, &s_out, &com_sout, &target_sf, &jcom.sf, &mut transcript,
    )?;
    let relocate_o = prove_column_relocation_with_context(
        context, &o_out, &com_oout, &target_of, &jcom.of, &mut transcript,
    )?;

    // (5) The complete self-join argument over the relocated F.
    let join = prove_self_join_complete_with_context(
        context, sf,
        of,
        &x,
        &y,
        &z,
        &rho_l,
        &rho_r,
        &u,
        &w_l,
        &w_r,
        m_j,
        &jcom,
        &mut transcript,
    )?;

    let (com_x, com_y, com_z) = (jcom.x, jcom.y, jcom.z);
    let proof = GraphPattern2Proof {
        membership,
        filter,
        com_sout,
        com_oout,
        relocate_s,
        relocate_o,
        join,
        join_commitments: jcom,
        n_f: m,
        n_u,
    };
    Ok((proof, com_x, com_y, com_z, m_j, predicate_id))
}

/// Verify a 2-atom graph-pattern proof. Anchors to `commitment_root`; re-derives the
/// join domain from the proof's pinned `n_F`/`n_U` and the public `m_J`; checks the
/// caller's result commitments are the proof's join output; then re-runs, under the
/// shared transcript in prover order: dict membership (IRI -> revealed id), the
/// complete filter (id -> F), the two cross-domain relocations (F -> join input), and
/// the complete self-join (F -> result). Concludes `(Com X, Com Y, Com Z)` is EXACTLY
/// the complete result of `t(?x, <IRI>, ?y), t(?y, <IRI>, ?z)`, without rerunning it.
#[allow(clippy::too_many_arguments)]
pub fn verify_layer_graph_pattern2(
    source: &impl VerifierLayer,
    trusted_resolution_root: ProofHash,
    predicate_term: &[u8],
    predicate_id: u64,
    com_x: &ArkGT,
    com_y: &ArkGT,
    com_z: &ArkGT,
    m_j: usize,
    proof: &GraphPattern2Proof,
) -> io::Result<()> {
    verify_layer_graph_pattern2_with_source(
        source,
        trusted_resolution_root,
        ColumnSource::Materialized,
        predicate_term,
        predicate_id,
        com_x,
        com_y,
        com_z,
        m_j,
        proof,
    )
}

/// Verify a 2-hop self-join proof over the columns selected by `column_source`.
#[allow(clippy::too_many_arguments)]
pub fn verify_layer_graph_pattern2_with_source(
    source: &impl VerifierLayer,
    trusted_resolution_root: ProofHash,
    column_source: ColumnSource,
    predicate_term: &[u8],
    predicate_id: u64,
    com_x: &ArkGT,
    com_y: &ArkGT,
    com_z: &ArkGT,
    m_j: usize,
    proof: &GraphPattern2Proof,
) -> io::Result<()> {
    let layer = source.verifier_view();
    // β re-anchor (FIX 4): pin to an EXTERNALLY-trusted `resolution_root` FIRST — it binds
    // `commitment_root` AND the two dictionary Merkle roots directly.
    if layer.resolution_root() != trusted_resolution_root {
        return Err(invalid(
            "layer resolution_root does not match the externally-trusted resolution root",
        ));
    }
    if layer.recomputed_commitment_root()? != layer.state.commitment_root {
        return Err(invalid(
            "layer commitment_root does not bind its committed columns/dictionary",
        ));
    }
    // Merkle-dict I2b (interim S4) fail-close: as in `verify_layer_bgp`, a TERM-LESS artifact's
    // committed predicate-dict HASH column is prover-chosen (validate() skips the catalog re-MSM),
    // so the `verify_dict_membership` predicate constant-match below is forgeable. This legacy
    // 2-hop entry is OFF the compact-envelope release path but is a pub verify surface — fence
    // term-less out of it too. (No-op on a term-bearing sidecar/artifact.)
    layer.require_terms_present("layer graph-pattern predicate constant-match")?;
    let n = source_count(layer.state, column_source) as usize;
    let padded = padded_len(n)?;
    let a = padded.trailing_zeros() as usize;
    if proof.filter.sumcheck.round_polys.len() != a {
        return Err(invalid(
            "proof dimension does not match the layer's committed columns",
        ));
    }

    // The caller's result commitments must be the proof's committed join output.
    if proof.join_commitments.x != *com_x
        || proof.join_commitments.y != *com_y
        || proof.join_commitments.z != *com_z
    {
        return Err(invalid(
            "result commitments do not match the proof's join output",
        ));
    }

    // Re-derive the join domain from the pinned filter-result / distinct-key counts.
    let n_f = proof.n_f;
    let n_u = proof.n_u;
    if n_f > padded {
        return Err(invalid("filter result length exceeds the table domain"));
    }
    let n_join = self_join_domain(n_f, m_j, n_u)?;
    let b = n_join.trailing_zeros() as usize;

    let mut transcript = Transcript254::new(GRAPH2_DOMAIN);
    bind_statement(&mut transcript, &layer, column_source, predicate_term, predicate_id);

    // (1) Membership: the revealed id is authentic for this IRI.
    let q_hash = hash_term(predicate_term);
    let id = ArkFr::from_u64(predicate_id);
    verify_dict_membership(
        &layer.predicate_dict_commitments.0,
        &layer.predicate_dict_commitments.1,
        q_hash,
        id,
        &proof.membership,
        &mut transcript,
    )?;

    // (2) Filter: F = (S_out, O_out) is the complete predicate-P result (length n_F).
    let [com_s, com_p, com_o] = source_column_commitments(layer, column_source);
    let pred = ArkFr::from_u64(predicate_id);
    verify_filter(
        &com_s,
        &com_p,
        &com_o,
        pred,
        n,
        &proof.com_sout,
        &proof.com_oout,
        n_f,
        &proof.filter,
        &mut transcript,
    )?;

    // (3) Relocate: the filter output equals the join input Sf/Of, positionally.
    verify_column_relocation(
        &proof.com_sout,
        a,
        &proof.join_commitments.sf,
        b,
        &proof.relocate_s,
        &mut transcript,
    )?;
    verify_column_relocation(
        &proof.com_oout,
        a,
        &proof.join_commitments.of,
        b,
        &proof.relocate_o,
        &mut transcript,
    )?;

    // (4) Complete self-join of F -> (X, Y, Z).
    verify_self_join_complete(
        &proof.join_commitments,
        n_f,
        n_u,
        m_j,
        &proof.join,
        &mut transcript,
    )
}
