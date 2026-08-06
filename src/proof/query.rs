//! Vertical slice: prove/verify a Q1 single-pattern filter `t(?s, P, ?o)` against
//! a REAL committed store layer's columns, anchored to the layer's `commitment_root`.
//!
//! For a materialized base layer (no removals), 1b's [`LayerCommitment`] holds the
//! full table's `S/P/O` id columns in `additions` and their Dory commitments in
//! `column_commitments[0..3]` (built by `commit_id_column`). Because the query
//! argument (2a/2b) commits/opens columns with the SAME `matrix_dims`, global
//! deterministic setup, `ArkFr::from_u64` encoding, and `next_power_of_two`
//! padding that `commit_id_column` uses, a column committed by 1b opens correctly
//! under the filter argument — so we can prove/verify a query directly against the
//! layer's committed state without rerunning it. (A test here asserts the two
//! commitments are byte-identical.)
//!
//! Predicate is a raw dictionary id (`ArkFr::from_u64`); IRI->id binding is Q3.

use std::io;

use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;

use super::argument::Transcript254;
use super::commitment::LayerCommitment;
use super::filter::{prove_filter, verify_filter, FullFilterProof};
use super::invalid;

const QUERY_DOMAIN: &[u8] = b"parture/proof-of-woql/q1-layer-filter/v0";

#[inline]
fn f0() -> ArkFr {
    ArkFr::zero()
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

/// Prove `t(?s, predicate_id, ?o)` against a materialized base layer: the dense
/// output `(S_out, O_out)[0..m]` is EXACTLY the compaction of the layer rows whose
/// predicate id equals `predicate_id`. Returns the proof, the output commitments,
/// and the result length `m`.
pub fn prove_layer_filter(
    layer: &LayerCommitment,
    predicate_id: u64,
) -> io::Result<(FullFilterProof, ArkGT, ArkGT, usize)> {
    let (s, p, o, n) = base_layer_columns(layer)?;
    let com_s = layer.column_commitments[0];
    let com_p = layer.column_commitments[1];
    let com_o = layer.column_commitments[2];
    let pred = ArkFr::from_u64(predicate_id);
    let mut transcript = Transcript254::new(QUERY_DOMAIN);
    prove_filter(&s, &p, &o, pred, n, &com_s, &com_p, &com_o, &mut transcript)
}

/// Verify a Q1 layer-filter proof. The trust anchor is the layer's
/// `commitment_root`: this recomputes the root binding from the layer's state
/// counts and `column_commitments` and checks it matches, so the `Com(S/P/O)` it
/// opens are exactly the ones the root authenticates. Then it runs the filter
/// verification. The verifier never rescans the table.
pub fn verify_layer_filter(
    layer: &LayerCommitment,
    predicate_id: u64,
    com_sout: &ArkGT,
    com_oout: &ArkGT,
    m: usize,
    proof: &FullFilterProof,
) -> io::Result<()> {
    // Anchor: the column commitments (and the counts) must be the ones bound in
    // commitment_root. (In deployment the trusted root also comes from the
    // authenticated layer chain; here we confirm the binding logic holds.)
    if layer.recomputed_commitment_root()? != layer.state.commitment_root {
        return Err(invalid(
            "layer commitment_root does not bind its column commitments",
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
    if proof.sumcheck.round_polys.len() != expected_k {
        return Err(invalid(
            "proof dimension does not match the layer's committed columns",
        ));
    }
    let com_s = layer.column_commitments[0];
    let com_p = layer.column_commitments[1];
    let com_o = layer.column_commitments[2];
    let pred = ArkFr::from_u64(predicate_id);
    let mut transcript = Transcript254::new(QUERY_DOMAIN);
    verify_filter(
        &com_s, &com_p, &com_o, pred, n, com_sout, com_oout, m, proof, &mut transcript,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::ValueTriple;
    use crate::proof::argument::commit_column;
    use crate::store::sync::open_sync_memory_store;

    // Build a materialized base layer with a known set of triples and return its
    // LayerCommitment plus the authenticated CommitmentState.
    fn build_layer(
        triples: &[(&str, &str, &str)],
    ) -> (
        std::sync::Arc<LayerCommitment>,
        crate::proof::CommitmentState,
    ) {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for (s, p, o) in triples {
            builder
                .add_value_triple(ValueTriple::new_node(*s, *p, *o))
                .unwrap();
        }
        let (layer, state) = builder.commit_with_proof().unwrap();
        let lc = layer.proof_commitment().unwrap().unwrap();
        (lc, state)
    }

    fn predicate_id(layer: &LayerCommitment, predicate: &str) -> u64 {
        layer
            .additions
            .iter()
            .find(|r| r.triple.predicate == predicate.as_bytes())
            .map(|r| r.predicate_id)
            .unwrap_or(u64::MAX) // absent predicate -> an id that matches nothing
    }

    // Independently recompute the expected dense result (subject/object id pairs)
    // for a predicate, in the same stored order the prover compacts.
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

    #[test]
    fn query_commitment_convention_matches_layer_commitment_columns() {
        // THE integration check: the query side's commitment of a column extracted
        // from the layer equals the 1b column commitment stored in the layer.
        let (lc, _) = build_layer(&[
            ("a1", "p1", "b1"),
            ("a2", "p1", "b2"),
            ("c1", "p2", "d1"),
        ]);
        let (s, p, o, _n) = base_layer_columns(&lc).unwrap();
        assert_eq!(commit_column(&s).unwrap(), lc.column_commitments[0]);
        assert_eq!(commit_column(&p).unwrap(), lc.column_commitments[1]);
        assert_eq!(commit_column(&o).unwrap(), lc.column_commitments[2]);
    }

    #[test]
    fn prove_and_verify_q1_over_a_real_layer_with_exact_result() {
        let triples = [
            ("a1", "p1", "b1"),
            ("a2", "p1", "b2"),
            ("a3", "p1", "b3"),
            ("a4", "p1", "b4"),
            ("c1", "p2", "d1"),
            ("c2", "p2", "d2"),
        ];
        let (lc, state) = build_layer(&triples);
        // The layer commitment we prove against carries the store's authenticated
        // commitment_root.
        assert_eq!(lc.state.commitment_root, state.commitment_root);

        let p1 = predicate_id(&lc, "p1");
        let (proof, com_sout, com_oout, m) = prove_layer_filter(&lc, p1).unwrap();
        verify_layer_filter(&lc, p1, &com_sout, &com_oout, m, &proof).unwrap();

        // The proven result is EXACTLY the p1 rows (count + contents): the output
        // commitments equal an independent commitment of the expected dense rows.
        assert_eq!(m, 4);
        let (exp_s, exp_o, exp_m) = expected_result(&lc, p1);
        assert_eq!(exp_m, m);
        assert_eq!(commit_column(&exp_s).unwrap(), com_sout);
        assert_eq!(commit_column(&exp_o).unwrap(), com_oout);
    }

    #[test]
    fn tampered_result_commitment_is_rejected() {
        let (lc, _) = build_layer(&[
            ("a1", "p1", "b1"),
            ("a2", "p1", "b2"),
            ("c1", "p2", "d1"),
        ]);
        let p1 = predicate_id(&lc, "p1");
        let (proof, com_sout, com_oout, m) = prove_layer_filter(&lc, p1).unwrap();
        // Swap in a wrong S_out commitment (use the O_out one) -> rejected.
        assert!(verify_layer_filter(&lc, p1, &com_oout, &com_oout, m, &proof).is_err());
        // Honest still verifies.
        verify_layer_filter(&lc, p1, &com_sout, &com_oout, m, &proof).unwrap();
    }

    #[test]
    fn absent_predicate_yields_empty_result_and_verifies() {
        let (lc, _) = build_layer(&[
            ("a1", "p1", "b1"),
            ("a2", "p1", "b2"),
            ("c1", "p2", "d1"),
        ]);
        // A predicate id that is not present in the layer.
        let absent = predicate_id(&lc, "does-not-exist");
        let (proof, com_sout, com_oout, m) = prove_layer_filter(&lc, absent).unwrap();
        assert_eq!(m, 0);
        verify_layer_filter(&lc, absent, &com_sout, &com_oout, m, &proof).unwrap();
    }

    #[test]
    fn non_materialized_layer_fails_closed() {
        // A base layer with removals is rejected (cross-layer live-set is later).
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for (s, p, o) in [("a1", "p1", "b1"), ("a2", "p1", "b2")] {
            builder
                .add_value_triple(ValueTriple::new_node(s, p, o))
                .unwrap();
        }
        let (base, _) = builder.commit_with_proof().unwrap();
        let child = base.open_write().unwrap();
        child
            .remove_value_triple(ValueTriple::new_node("a1", "p1", "b1"))
            .unwrap();
        let (head, _) = child.commit_with_proof().unwrap();
        let lc = head.proof_commitment().unwrap().unwrap();
        assert!(prove_layer_filter(&lc, 1).is_err());
    }
}
