//! End-to-end single-pattern query BY PREDICATE IRI against a REAL committed
//! store layer, anchored to the layer's `commitment_root`.
//!
//! A query names a predicate IRI (term bytes), not a raw dictionary id. The proof
//! combines two arguments under ONE shared transcript:
//!   1. Q3 dictionary membership ([`super::dict`]): the revealed `predicate_id` is
//!      the genuine dictionary id for `hash_term(predicate_term)`, authenticated by
//!      the layer's committed predicate dictionary (`predicate_dict_commitments`).
//!   2. Q1 filter compaction ([`super::filter`]): `(S_out, O_out)[0..m]` is exactly
//!      the compaction of the rows whose predicate id equals that id.
//!
//! Both the column commitments and the predicate-dictionary commitments are bound
//! by `commitment_root` (1b + 3b-1). The verifier gets only the public IRI bytes
//! plus a revealed id (checked via membership — never trusted out of band), and
//! concludes `(S_out, O_out)` is exactly the complete result of `t(?s, <IRI>, ?o)`
//! without rerunning the query. Node/value (subject/object) dictionary binding is
//! deferred; absent-predicate (non-membership) proofs are a later feature — an IRI
//! not in the layer fails closed.

use std::io;

use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::transcript::Transcript;

use super::argument::Transcript254;
use super::commitment::LayerCommitment;
use super::dict::{hash_term, prove_dict_membership, verify_dict_membership, DictProof};
use super::filter::{prove_filter, verify_filter, FullFilterProof};
use super::invalid;

const QUERY_DOMAIN: &[u8] = b"parture/proof-of-woql/q1-query-by-iri/v0";

#[inline]
fn f0() -> ArkFr {
    ArkFr::zero()
}

/// A complete query-by-IRI proof: the predicate-dictionary membership proof
/// (IRI -> id) and the filter/compaction proof (id -> result), both produced
/// under one shared transcript.
#[derive(Clone, Debug)]
pub struct CombinedFilterProof {
    pub membership: DictProof,
    pub filter: FullFilterProof,
}

/// The padded hypercube size for a materialized base layer's columns, matching
/// `commit_id_column`'s `next_power_of_two(additions.len())`. Requires `>= 2` (a
/// one-row layer has `k = 0`, below the sumcheck's minimum of one variable).
fn padded_len(real_rows: usize) -> io::Result<usize> {
    let padded = real_rows
        .checked_next_power_of_two()
        .ok_or_else(|| invalid("layer row count overflow"))?;
    if padded < 2 {
        return Err(invalid(
            "layer has too few rows for a query proof (need >= 2)",
        ));
    }
    Ok(padded)
}

/// Extract the `(S, P, O)` `ArkFr` columns and real-row count `n` from a
/// materialized base layer, in the exact stored (SPO-sorted) order and padding
/// that produced `column_commitments[0..3]`.
fn base_layer_columns(
    layer: &LayerCommitment,
) -> io::Result<(Vec<ArkFr>, Vec<ArkFr>, Vec<ArkFr>, usize)> {
    if !layer.removals.is_empty() {
        return Err(invalid(
            "layer filter requires a materialized base layer (no removals)",
        ));
    }
    let n = layer.additions.len();
    let padded = padded_len(n)?;
    let mut s = vec![f0(); padded];
    let mut p = vec![f0(); padded];
    let mut o = vec![f0(); padded];
    for (i, record) in layer.additions.iter().enumerate() {
        s[i] = ArkFr::from_u64(record.subject_id);
        p[i] = ArkFr::from_u64(record.predicate_id);
        o[i] = ArkFr::from_u64(record.object_id);
    }
    Ok((s, p, o, n))
}

/// Resolve a predicate IRI (term bytes) to its dictionary id in this layer, or
/// fail closed if the term is not present (non-membership proofs are deferred).
fn resolve_predicate_id(layer: &LayerCommitment, predicate_term: &[u8]) -> io::Result<u64> {
    layer
        .predicate_catalog
        .iter()
        .find(|record| record.triple.predicate == predicate_term)
        .map(|record| record.predicate_id)
        .ok_or_else(|| invalid("predicate not in layer dictionary"))
}

/// Bind the public statement into the shared transcript before either sub-proof:
/// the trust anchor (`commitment_root`, which binds the column + dictionary
/// commitments), the public predicate IRI bytes, and the revealed id.
fn bind_statement(
    transcript: &mut Transcript254,
    layer: &LayerCommitment,
    predicate_term: &[u8],
    predicate_id: u64,
) {
    transcript.append_bytes(b"pow/q1-iri/commitment-root", &layer.state.commitment_root);
    transcript.append_bytes(b"pow/q1-iri/predicate-term", predicate_term);
    transcript.append_bytes(b"pow/q1-iri/predicate-id", &predicate_id.to_le_bytes());
}

/// Prove `t(?s, <predicate_term>, ?o)` against a materialized base layer. Resolves
/// the predicate IRI to its committed dictionary id, proves that binding (Q3
/// membership), then proves the filtered result for that id (Q1). Returns the
/// combined proof, the output commitments, the result length `m`, and the REVEALED
/// resolved id (authenticated by the membership proof — not secret).
pub fn prove_layer_filter(
    layer: &LayerCommitment,
    predicate_term: &[u8],
) -> io::Result<(CombinedFilterProof, ArkGT, ArkGT, usize, u64)> {
    let (s, p, o, n) = base_layer_columns(layer)?;
    let predicate_id = resolve_predicate_id(layer, predicate_term)?;

    let (dict_hash, dict_id) = layer.predicate_dictionary_columns()?;
    let com_dict_hash = layer.predicate_dict_commitments.0;
    let com_dict_id = layer.predicate_dict_commitments.1;

    let mut transcript = Transcript254::new(QUERY_DOMAIN);
    bind_statement(&mut transcript, layer, predicate_term, predicate_id);

    // Q3 membership: (hash_term(IRI), id) is a genuine dictionary row.
    let q_hash = hash_term(predicate_term);
    let id = ArkFr::from_u64(predicate_id);
    let membership = prove_dict_membership(
        &dict_hash,
        &dict_id,
        &com_dict_hash,
        &com_dict_id,
        q_hash,
        id,
        &mut transcript,
    )?;

    // Q1 filter: the result for that id.
    let com_s = layer.column_commitments[0];
    let com_p = layer.column_commitments[1];
    let com_o = layer.column_commitments[2];
    let pred = ArkFr::from_u64(predicate_id);
    let (filter, com_sout, com_oout, m) =
        prove_filter(&s, &p, &o, pred, n, &com_s, &com_p, &com_o, &mut transcript)?;

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
    layer: &LayerCommitment,
    predicate_term: &[u8],
    predicate_id: u64,
    com_sout: &ArkGT,
    com_oout: &ArkGT,
    m: usize,
    proof: &CombinedFilterProof,
) -> io::Result<()> {
    if layer.recomputed_commitment_root()? != layer.state.commitment_root {
        return Err(invalid(
            "layer commitment_root does not bind its committed columns/dictionary",
        ));
    }
    if layer.state.removal_count != 0 {
        return Err(invalid(
            "layer filter requires a materialized base layer (no removals)",
        ));
    }
    let n = layer.state.addition_count as usize;
    let padded = padded_len(n)?;
    let expected_k = padded.trailing_zeros() as usize;
    if proof.filter.sumcheck.round_polys.len() != expected_k {
        return Err(invalid(
            "proof dimension does not match the layer's committed columns",
        ));
    }

    let mut transcript = Transcript254::new(QUERY_DOMAIN);
    bind_statement(&mut transcript, layer, predicate_term, predicate_id);

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
    let com_s = layer.column_commitments[0];
    let com_p = layer.column_commitments[1];
    let com_o = layer.column_commitments[2];
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::ValueTriple;
    use crate::proof::argument::commit_column;
    use crate::store::sync::open_sync_memory_store;

    const P1: &[u8] = b"http://ex/p1";
    const P2: &[u8] = b"http://ex/p2";

    fn iri(suffix: &str) -> String {
        format!("http://ex/{suffix}")
    }

    // Build a materialized base layer from (subject, predicate, object) IRI triples.
    fn build_layer(triples: &[(&str, &str, &str)]) -> std::sync::Arc<LayerCommitment> {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for (s, p, o) in triples {
            builder
                .add_value_triple(ValueTriple::new_node(&iri(s), &iri(p), &iri(o)))
                .unwrap();
        }
        let (layer, _state) = builder.commit_with_proof().unwrap();
        layer.proof_commitment().unwrap().unwrap()
    }

    // Independently recompute the expected dense result for a predicate id.
    fn expected_result(layer: &LayerCommitment, pred_id: u64) -> (Vec<ArkFr>, Vec<ArkFr>, usize) {
        let padded = padded_len(layer.additions.len()).unwrap();
        let mut s_out = vec![f0(); padded];
        let mut o_out = vec![f0(); padded];
        let mut m = 0usize;
        for record in &layer.additions {
            if record.predicate_id == pred_id {
                s_out[m] = ArkFr::from_u64(record.subject_id);
                o_out[m] = ArkFr::from_u64(record.object_id);
                m += 1;
            }
        }
        (s_out, o_out, m)
    }

    fn sample_layer() -> std::sync::Arc<LayerCommitment> {
        build_layer(&[
            ("a1", "p1", "b1"),
            ("a2", "p1", "b2"),
            ("a3", "p1", "b3"),
            ("a4", "p1", "b4"),
            ("c1", "p2", "d1"),
            ("c2", "p2", "d2"),
        ])
    }

    #[test]
    fn query_commitment_convention_matches_layer_commitment_columns() {
        let lc = build_layer(&[("a1", "p1", "b1"), ("a2", "p1", "b2"), ("c1", "p2", "d1")]);
        let (s, p, o, _n) = base_layer_columns(&lc).unwrap();
        assert_eq!(commit_column(&s).unwrap(), lc.column_commitments[0]);
        assert_eq!(commit_column(&p).unwrap(), lc.column_commitments[1]);
        assert_eq!(commit_column(&o).unwrap(), lc.column_commitments[2]);
    }

    #[test]
    fn prove_and_verify_by_iri_with_exact_result() {
        let lc = sample_layer();
        let (proof, com_sout, com_oout, m, resolved_id) = prove_layer_filter(&lc, P1).unwrap();
        // The verifier is given ONLY the IRI bytes + the revealed id (authenticated
        // by membership) -- never an out-of-band id.
        verify_layer_filter(&lc, P1, resolved_id, &com_sout, &com_oout, m, &proof).unwrap();

        // Result is EXACTLY the p1 rows (count + contents).
        assert_eq!(m, 4);
        let (exp_s, exp_o, exp_m) = expected_result(&lc, resolved_id);
        assert_eq!(exp_m, m);
        assert_eq!(commit_column(&exp_s).unwrap(), com_sout);
        assert_eq!(commit_column(&exp_o).unwrap(), com_oout);
    }

    #[test]
    fn wrong_revealed_id_for_the_iri_is_rejected() {
        let lc = sample_layer();
        let (proof, com_sout, com_oout, m, resolved_id) = prove_layer_filter(&lc, P1).unwrap();
        // Claim a different id for the same IRI -> membership fails.
        let wrong = resolved_id.wrapping_add(1);
        assert!(verify_layer_filter(&lc, P1, wrong, &com_sout, &com_oout, m, &proof).is_err());
    }

    #[test]
    fn substituted_id_from_a_different_predicate_is_rejected() {
        let lc = sample_layer();
        let (proof, com_sout, com_oout, m, _p1_id) = prove_layer_filter(&lc, P1).unwrap();
        // p2's genuine id, but paired with p1's IRI: (hash_term(p1), id_p2) is not a
        // dictionary row -> membership fails.
        let p2_id = resolve_predicate_id(&lc, P2).unwrap();
        assert!(verify_layer_filter(&lc, P1, p2_id, &com_sout, &com_oout, m, &proof).is_err());
    }

    #[test]
    fn absent_predicate_iri_fails_closed() {
        let lc = sample_layer();
        let err = prove_layer_filter(&lc, b"http://ex/absent").unwrap_err();
        assert!(err.to_string().contains("predicate not in layer dictionary"));
    }

    #[test]
    fn tampered_result_is_rejected_under_the_combined_transcript() {
        let lc = sample_layer();
        let (proof, com_sout, com_oout, m, resolved_id) = prove_layer_filter(&lc, P1).unwrap();
        // Wrong S_out commitment (use the O_out one) -> filter opening fails.
        assert!(
            verify_layer_filter(&lc, P1, resolved_id, &com_oout, &com_oout, m, &proof).is_err()
        );
        // Honest still verifies.
        verify_layer_filter(&lc, P1, resolved_id, &com_sout, &com_oout, m, &proof).unwrap();
    }

    #[test]
    fn non_materialized_layer_fails_closed() {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for (s, p, o) in [("a1", "p1", "b1"), ("a2", "p1", "b2")] {
            builder
                .add_value_triple(ValueTriple::new_node(&iri(s), &iri(p), &iri(o)))
                .unwrap();
        }
        let (base, _) = builder.commit_with_proof().unwrap();
        let child = base.open_write().unwrap();
        child
            .remove_value_triple(ValueTriple::new_node(&iri("a1"), &iri("p1"), &iri("b1")))
            .unwrap();
        let (head, _) = child.commit_with_proof().unwrap();
        let lc = head.proof_commitment().unwrap().unwrap();
        assert!(prove_layer_filter(&lc, P1).is_err());
    }
}
