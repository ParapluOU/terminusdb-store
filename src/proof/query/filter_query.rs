//! Q1 — proving a single triple pattern by predicate IRI: `t(?s, <P>, ?o)`.
//!
//! This is the simplest end-to-end query: name a predicate IRI, get back the complete set
//! of `(subject, object)` pairs carrying that predicate, with a proof. Under one shared
//! transcript it composes two sub-proofs: (1) the predicate IRI really resolves to the
//! revealed dictionary id (dictionary membership), and (2) the returned columns are exactly
//! the rows whose predicate equals that id (filter/compaction). Both are anchored to the
//! layer's `commitment_root`, so the verifier trusts the result without re-scanning the
//! table. The `*_with_source` entry points let the same machinery run over a delta file
//! (additions/removals) instead of the materialized head.

use super::*;

const QUERY_DOMAIN: &[u8] = b"parture/proof-of-woql/q1-query-by-iri/v0";

/// A complete query-by-IRI proof: the predicate-dictionary membership proof
/// (IRI -> id) and the filter/compaction proof (id -> result), both produced
/// under one shared transcript.
#[derive(Clone, Debug)]
pub struct CombinedFilterProof {
    pub membership: DictProof,
    pub filter: FullFilterProof,
}

/// Bind the public statement into the shared transcript before either sub-proof:
/// the trust anchor (`commitment_root`, which binds the column + dictionary
/// commitments), the public predicate IRI bytes, and the revealed id.
pub(super) fn bind_statement(
    transcript: &mut Transcript254,
    source: &impl VerifierLayer,
    column_source: ColumnSource,
    predicate_term: &[u8],
    predicate_id: u64,
) {
    let layer = source.verifier_view();
    transcript.append_bytes(b"pow/q1-iri/commitment-root", &layer.state.commitment_root);
    // Domain-separate the column source. `Materialized` emits NOTHING, so pre-delta
    // Q1/Q2 transcripts are byte-identical; a delta source emits a distinct tag.
    match column_source {
        ColumnSource::Materialized => {}
        ColumnSource::Additions => transcript.append_bytes(b"pow/q1-iri/source-additions", &[]),
        ColumnSource::Removals => transcript.append_bytes(b"pow/q1-iri/source-removals", &[]),
    }
    transcript.append_bytes(b"pow/q1-iri/predicate-term", predicate_term);
    transcript.append_bytes(b"pow/q1-iri/predicate-id", &predicate_id.to_le_bytes());
}

/// Prove `t(?s, <predicate_term>, ?o)` against a complete materialized head view. Resolves
/// the predicate IRI to its committed dictionary id, proves that binding (Q3
/// membership), then proves the filtered result for that id (Q1). Returns the
/// combined proof, the output commitments, the result length `m`, and the REVEALED
/// resolved id (authenticated by the membership proof — not secret).
pub fn prove_layer_filter(
    layer: &LayerCommitment,
    predicate_term: &[u8],
) -> io::Result<(CombinedFilterProof, ArkGT, ArkGT, usize, u64)> {
    prove_layer_filter_with_context(&StoreProverContext::cpu(), layer, predicate_term)
}

pub fn prove_layer_filter_with_context(
    context: &StoreProverContext<'_>,
    layer: &LayerCommitment,
    predicate_term: &[u8],
) -> io::Result<(CombinedFilterProof, ArkGT, ArkGT, usize, u64)> {
    prove_layer_filter_with_source(context, layer, ColumnSource::Materialized, predicate_term)
}

/// Prove `t(?s, <predicate_term>, ?o)` over the layer columns selected by `column_source`
/// (materialized head, or the layer's own addition/removal delta files). Materialized is the
/// byte-identical legacy behavior; a delta source reads the delta rows + delta column
/// commitments + delta count and adds a source tag to the transcript.
pub fn prove_layer_filter_with_source(
    context: &StoreProverContext<'_>,
    layer: &LayerCommitment,
    column_source: ColumnSource,
    predicate_term: &[u8],
) -> io::Result<(CombinedFilterProof, ArkGT, ArkGT, usize, u64)> {
    require_base_layer_for_query(&layer.state)?;
    let (s, p, o, n) = source_layer_columns(layer, column_source)?;
    let predicate_id = resolve_predicate_id(layer, predicate_term)?;

    let (dict_hash, dict_id) = layer.predicate_dictionary_columns()?;
    let com_dict_hash = layer.predicate_dict_commitments.0;
    let com_dict_id = layer.predicate_dict_commitments.1;

    let mut transcript = Transcript254::new(QUERY_DOMAIN);
    bind_statement(&mut transcript, layer, column_source, predicate_term, predicate_id);

    // Q3 membership: (hash_term(IRI), id) is a genuine dictionary row.
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

    // Q1 filter: the result for that id.
    let [com_s, com_p, com_o] = source_column_commitments(layer.verifier_view(), column_source);
    let pred = ArkFr::from_u64(predicate_id);
    let (filter, com_sout, com_oout, m) =
        prove_filter_with_context(
            context, &s, &p, &o, pred, n, &com_s, &com_p, &com_o, &mut transcript,
        )?;

    Ok((
        CombinedFilterProof { membership, filter },
        com_sout,
        com_oout,
        m,
        predicate_id,
    ))
}

/// Verify a query-by-IRI proof. Anchors to `commitment_root` (which binds both the
/// column and predicate-dictionary commitments), independently computes
/// `q_hash = hash_term(predicate_term)`, checks the revealed `predicate_id` is the
/// genuine dictionary id for that IRI (membership), then checks the filtered
/// result for that id (filter) — all under the shared transcript. The verifier
/// never trusts the id out of band and never rescans the table.
pub fn verify_layer_filter(
    source: &impl VerifierLayer,
    expected_root: ProofHash,
    predicate_term: &[u8],
    predicate_id: u64,
    com_sout: &ArkGT,
    com_oout: &ArkGT,
    m: usize,
    proof: &CombinedFilterProof,
) -> io::Result<()> {
    verify_layer_filter_with_source(
        source,
        expected_root,
        ColumnSource::Materialized,
        predicate_term,
        predicate_id,
        com_sout,
        com_oout,
        m,
        proof,
    )
}

/// Verify a query-by-IRI proof over the columns selected by `column_source`. The source is a
/// PUBLIC input to verification (the verifier chooses it from its own trusted statement), so a
/// prover cannot prove additions and have the proof accepted as materialized.
#[allow(clippy::too_many_arguments)]
pub fn verify_layer_filter_with_source(
    source: &impl VerifierLayer,
    expected_root: ProofHash,
    column_source: ColumnSource,
    predicate_term: &[u8],
    predicate_id: u64,
    com_sout: &ArkGT,
    com_oout: &ArkGT,
    m: usize,
    proof: &CombinedFilterProof,
) -> io::Result<()> {
    let layer = source.verifier_view();
    require_base_layer_for_query(layer.state)?;
    // FIX 4: pin to an EXTERNALLY-trusted root FIRST (mirroring import_pack's
    // external-root check), so the proof is bound to a root the verifier already
    // trusts rather than whatever LayerCommitment the prover supplied.
    if layer.state.commitment_root != expected_root {
        return Err(invalid(
            "layer commitment_root does not match the externally-trusted expected root",
        ));
    }
    if layer.recomputed_commitment_root()? != layer.state.commitment_root {
        return Err(invalid(
            "layer commitment_root does not bind its committed columns/dictionary",
        ));
    }
    let n = source_count(layer.state, column_source) as usize;
    let padded = padded_len(n)?;
    let expected_k = padded.trailing_zeros() as usize;
    if proof.filter.sumcheck.round_polys.len() != expected_k {
        return Err(invalid(
            "proof dimension does not match the layer's committed columns",
        ));
    }

    let mut transcript = Transcript254::new(QUERY_DOMAIN);
    bind_statement(&mut transcript, &layer, column_source, predicate_term, predicate_id);

    // Membership: the revealed id is authentic for this IRI.
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

    // Filter: the result for that id.
    let [com_s, com_p, com_o] = source_column_commitments(layer, column_source);
    let pred = ArkFr::from_u64(predicate_id);
    verify_filter(
        &com_s,
        &com_p,
        &com_o,
        pred,
        n,
        com_sout,
        com_oout,
        m,
        &proof.filter,
        &mut transcript,
    )
}
