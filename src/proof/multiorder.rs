//! Exact complete-row lexicographic order for two or three numeric Store-ID columns.
//!
//! WHAT IT PROVES: this is a full `ORDER BY` over a 2- or 3-column relation of Store ids.
//! Given a committed input relation and a claimed output relation, it certifies that the
//! output is (a) an exact full-row PERMUTATION of the input — same rows, same
//! multiplicities, no fabricated or dropped row — AND (b) sorted by the requested key
//! columns and directions. `key_indices` is a permutation of the columns giving the sort
//! priority (so a subset key plus a tie-break column can express `ORDER BY key` with a
//! deterministic total order).
//!
//! HOW (composition of two audited gadgets):
//!   1. [`super::permutation`] proves output-is-a-permutation-of-input (row multiset
//!      equality), yielding the reordered columns and their commitments.
//!   2. The ids are resolved to their authenticated fixed-width numeric VALUES through the
//!      layer's root-bound node/value dictionary (rejecting any id that is not the exact
//!      planned datatype), and [`super::lexsort`] proves those value columns are
//!      nondecreasing in the requested order.
//! Permutation alone would allow any reordering; lexsort alone would allow fabricated
//! rows. Together they pin the output to the one correct sorted relation. Only fixed-width
//! order-preserving datatypes (the integer/float/boolean family) are accepted, because the
//! order proof compares the committed value encodings as magnitudes.

use std::{cmp::Ordering, io};
use ark_ff::PrimeField;
use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::transcript::Transcript;
use super::argument::{commit_column, StoreProverContext, Transcript254};
use super::bgp::{RelationCommit, RelationWitness};
use super::commitment::{node_value_op_value, LayerCommitment, VerifierLayer};
use super::lexsort::{prove_lexicographically_nondecreasing_with_context, verify_lexicographically_nondecreasing, LexicographicSortedProof};
use super::order::OrderDirection;
use super::permutation::{prove_relation_permutation_with_context, verify_relation_permutation, RelationPermutationProof};
use super::{invalid, CanonicalDatatype, CanonicalObject};

/// An exact multi-key order proof: the output-is-a-permutation-of-input subproof plus the
/// lexicographic-order subproof over the resolved value columns.
#[derive(Clone, Debug)]
pub struct ExactMultiOrderProof {
    pub permutation: RelationPermutationProof,
    pub lexicographic: LexicographicSortedProof,
}

/// Prover result: the proof plus the sorted output relation (its id columns, their
/// commitments, the shared power-of-two `log_domain`, and the real row count `n`).
pub struct ExactMultiOrderResult {
    pub proof: ExactMultiOrderProof, pub commitments: Vec<ArkGT>,
    pub columns: Vec<Vec<ArkFr>>, pub log_domain: usize, pub n: usize,
}

fn validate_shape(arity: usize, datatypes: &[CanonicalDatatype], key_indices: &[usize], directions: &[OrderDirection]) -> io::Result<()> {
    if !(2..=3).contains(&arity) || datatypes.len() != arity || directions.len() != arity || key_indices.len() != arity {
        return Err(invalid("exact multi-order requires two or three complete-row keys"));
    }
    let mut sorted = key_indices.to_vec(); sorted.sort_unstable();
    if sorted != (0..arity).collect::<Vec<_>>() { return Err(invalid("multi-order key indices must be an exact column permutation")); }
    // `Float32`/`Float64` are DELIBERATELY excluded (defense-in-depth): the IEEE total-order
    // float key diverges from WOQL numeric order on `-0.0`/`+0.0`/`NaN`, and the woql2 compiler
    // already reject-closes float ORDER before a float key can reach this gadget.
    if datatypes.iter().any(|datatype| !matches!(datatype,
        CanonicalDatatype::Boolean | CanonicalDatatype::UInt8 | CanonicalDatatype::UInt16 |
        CanonicalDatatype::UInt32 | CanonicalDatatype::UInt64 | CanonicalDatatype::Int8 |
        CanonicalDatatype::Int16 | CanonicalDatatype::Int32 | CanonicalDatatype::Int64)) {
        return Err(invalid("exact multi-order datatype is not fixed-width order-preserving"));
    }
    Ok(())
}

fn resolve_values(source: &impl VerifierLayer, columns: &[Vec<ArkFr>], n: usize, datatypes: &[CanonicalDatatype], domain: usize) -> io::Result<Vec<Vec<ArkFr>>> {
    let view = source.verifier_view();
    // Merkle-dict I2b S2 fail-close: multi-column OrderBy resolves each key id to its dict VALUE
    // (`node_value_op_value`) for the lexicographic comparison — tied to real terms only by the
    // catalog re-MSM a TERM-LESS artifact drops. Reject before reading the (empty) catalog.
    view.require_terms_present("exact multi-column OrderBy (dictionary-value sort keys)")?;
    view.validate()?;
    columns.iter().zip(datatypes).map(|(column, datatype)| {
        if column.len() < n { return Err(invalid("multi-order output column too short")); }
        let mut values = Vec::with_capacity(domain);
        for id in &column[..n] {
            let record = view.node_value_catalog.iter().find(|record| ArkFr::from_u64(record.object_id) == *id)
                .ok_or_else(|| invalid("multi-order ID absent from root catalog"))?;
            let actual = match &record.triple.object { CanonicalObject::Value { datatype, .. } => *datatype, _ => return Err(invalid("multi-order key is not a value")) };
            if actual != *datatype { return Err(invalid("multi-order datatype mask is not complete")); }
            values.push(node_value_op_value(&record.triple.object)?);
        }
        values.resize(domain, ArkFr::zero()); Ok(values)
    }).collect()
}

fn bind(transcript: &mut Transcript254, datatypes: &[CanonicalDatatype], key_indices: &[usize], directions: &[OrderDirection], n: usize) {
    transcript.append_bytes(b"pow/multi-order/domain", b"parture/exact-multi-order/v0");
    transcript.append_bytes(b"pow/multi-order/rows", &(n as u64).to_le_bytes());
    for (index, direction) in key_indices.iter().zip(directions) {
        transcript.append_bytes(b"pow/multi-order/key", &[(*index) as u8, match direction { OrderDirection::Ascending => 0, OrderDirection::Descending => 1 }]);
    }
    for datatype in datatypes { transcript.append_bytes(b"pow/multi-order/datatype", &[*datatype as u8]); }
}

/// Prove a full 2- or 3-column `ORDER BY` of `input` against the root-bound `layer`
/// dictionary. `datatypes` gives each column's expected fixed-width datatype;
/// `key_indices` is a permutation of the columns giving sort priority; `directions` is the
/// per-key ascending/descending sense. Returns the sorted output relation and the proof
/// that it is the input's rows in exactly that order.
pub fn prove_exact_multi_order_with_context(context: &StoreProverContext<'_>, layer: &LayerCommitment, input: &RelationWitness, datatypes: &[CanonicalDatatype], key_indices: &[usize], directions: &[OrderDirection], transcript: &mut Transcript254) -> io::Result<ExactMultiOrderResult> {
    validate_shape(input.columns.len(), datatypes, key_indices, directions)?;
    let domain = 1usize << input.log_domain;
    let values = resolve_values(layer, &input.columns, input.n, datatypes, domain)?;
    let mut order = (0..input.n).collect::<Vec<_>>();
    order.sort_by(|left, right| {
        for (index, direction) in key_indices.iter().zip(directions) { let column = &values[*index];
            let cmp = column[*left].0.into_bigint().cmp(&column[*right].0.into_bigint());
            if cmp != Ordering::Equal { return if *direction == OrderDirection::Ascending { cmp } else { cmp.reverse() }; }
        } Ordering::Equal
    });
    bind(transcript, datatypes, key_indices, directions, input.n);
    let permutation = prove_relation_permutation_with_context(context, input, &order, transcript)?;
    let ordered_values = resolve_values(layer, &permutation.columns, input.n, datatypes, domain)?;
    let value_commitments = ordered_values.iter().map(|column| context.commit_column(column)).collect::<io::Result<Vec<_>>>()?;
    let mut chi = vec![ArkFr::zero(); domain]; chi[..input.n].fill(ArkFr::one());
    let chi_commitment = context.commit_column(&chi)?;
    let ordered_key_values = key_indices.iter().map(|i| ordered_values[*i].clone()).collect::<Vec<_>>();
    let ordered_key_commitments = key_indices.iter().map(|i| value_commitments[*i]).collect::<Vec<_>>();
    let lexicographic = prove_lexicographically_nondecreasing_with_context(context, &ordered_key_values, &ordered_key_commitments, directions, &chi_commitment, input.n, transcript)?;
    Ok(ExactMultiOrderResult { proof: ExactMultiOrderProof { permutation: permutation.proof, lexicographic }, commitments: permutation.commitments, columns: permutation.columns, log_domain: permutation.log_domain, n: permutation.n })
}

/// Verify an exact multi-key order proof: the claimed output relation (its `output_columns`
/// / `output_commitments`, length, and domain) is a full-row permutation of `input` AND is
/// sorted by `key_indices`/`directions` under the `source` layer's authenticated
/// dictionary values.
pub fn verify_exact_multi_order(source: &impl VerifierLayer, input: &RelationCommit, output_columns: &[Vec<ArkFr>], datatypes: &[CanonicalDatatype], key_indices: &[usize], directions: &[OrderDirection], output_commitments: &[ArkGT], output_len: usize, output_log_domain: usize, proof: &ExactMultiOrderProof, transcript: &mut Transcript254) -> io::Result<()> {
    validate_shape(input.commitments.len(), datatypes, key_indices, directions)?; bind(transcript, datatypes, key_indices, directions, input.n);
    verify_relation_permutation(input, output_commitments, output_len, output_log_domain, &proof.permutation, transcript)?;
    if output_len != input.n || output_log_domain != input.log_domain || output_columns.len() != datatypes.len() { return Err(invalid("multi-order output shape mismatch")); }
    let domain = 1usize << output_log_domain;
    let values = resolve_values(source, output_columns, output_len, datatypes, domain)?;
    let commitments = values.iter().map(|column| commit_column(column)).collect::<io::Result<Vec<_>>>()?;
    let mut chi = vec![ArkFr::zero(); domain]; chi[..output_len].fill(ArkFr::one());
    let key_commitments = key_indices.iter().map(|i| commitments[*i]).collect::<Vec<_>>();
    verify_lexicographically_nondecreasing(&key_commitments, directions, &commit_column(&chi)?, output_len, &proof.lexicographic, transcript)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{store::sync::open_sync_memory_store, ValueTriple};
    use tdb_succinct::TdbDataType;

    fn fixture() -> std::sync::Arc<LayerCommitment> {
        let store = open_sync_memory_store(); let builder = store.create_base_layer().unwrap();
        for value in 1u32..=9 { builder.add_value_triple(ValueTriple::new_value(&format!("s{value}"), "p", u32::make_entry(&value))).unwrap(); }
        builder.commit().unwrap().proof_commitment().unwrap()
    }
    fn id(layer: &LayerCommitment, value: u32) -> ArkFr {
        let entry = CanonicalObject::from_store_object(&ValueTriple::new_value("", "", u32::make_entry(&value)).object);
        ArkFr::from_u64(layer.node_value_catalog.iter().find(|r| r.triple.object == entry).unwrap().object_id)
    }
    /// Merkle-dict I2b S2 (N1): a valid multi-column order proof rejects on a TERM-LESS view of
    /// the same base layer — `resolve_values` decodes each key id's dictionary VALUE, untied to
    /// real terms without the catalog re-MSM. Term-bearing verify still succeeds.
    #[test]
    fn multi_order_verify_fails_closed_on_termless_view() {
        let layer = fixture();
        let input = RelationWitness::commit(vec![vec![id(&layer,2),id(&layer,1),id(&layer,1)], vec![id(&layer,8),id(&layer,3),id(&layer,9)]], 3).unwrap();
        let types = [CanonicalDatatype::UInt32, CanonicalDatatype::UInt32];
        let dirs = [OrderDirection::Descending, OrderDirection::Ascending];
        let result = prove_exact_multi_order_with_context(&StoreProverContext::cpu(), &layer, &input, &types, &[1,0], &dirs, &mut Transcript254::new(b"multiorder-test")).unwrap();
        verify_exact_multi_order(&layer, &input.view(), &result.columns, &types, &[1,0], &dirs, &result.commitments, result.n, result.log_domain, &result.proof, &mut Transcript254::new(b"multiorder-test")).unwrap();
        let termless = layer.verifier_commitment_termless().unwrap();
        let err = verify_exact_multi_order(&termless, &input.view(), &result.columns, &types, &[1,0], &dirs, &result.commitments, result.n, result.log_domain, &result.proof, &mut Transcript254::new(b"multiorder-test")).unwrap_err();
        assert!(err.to_string().contains("terms-bearing"), "{err}");
    }

    #[test]
    fn exact_two_key_mixed_order_and_direction_tamper() {
        let layer = fixture();
        let input = RelationWitness::commit(vec![vec![id(&layer,2),id(&layer,1),id(&layer,1)], vec![id(&layer,8),id(&layer,3),id(&layer,9)]], 3).unwrap();
        let types = [CanonicalDatatype::UInt32, CanonicalDatatype::UInt32];
        let dirs = [OrderDirection::Descending, OrderDirection::Ascending];
        let result = prove_exact_multi_order_with_context(&StoreProverContext::cpu(), &layer, &input, &types, &[1,0], &dirs, &mut Transcript254::new(b"multiorder-test")).unwrap();
        assert_eq!(&result.columns[0][..3], &[id(&layer,1),id(&layer,2),id(&layer,1)]);
        assert_eq!(&result.columns[1][..3], &[id(&layer,9),id(&layer,8),id(&layer,3)]);
        verify_exact_multi_order(&layer, &input.view(), &result.columns, &types, &[1,0], &dirs, &result.commitments, result.n, result.log_domain, &result.proof, &mut Transcript254::new(b"multiorder-test")).unwrap();
        let wrong = [OrderDirection::Ascending, OrderDirection::Ascending];
        assert!(verify_exact_multi_order(&layer, &input.view(), &result.columns, &types, &[1,0], &wrong, &result.commitments, result.n, result.log_domain, &result.proof, &mut Transcript254::new(b"multiorder-test")).is_err());
        assert!(verify_exact_multi_order(&layer, &input.view(), &result.columns, &types, &[0,1], &dirs, &result.commitments, result.n, result.log_domain, &result.proof, &mut Transcript254::new(b"multiorder-test")).is_err());
        for invalid in [&[0,0][..], &[0,2][..], &[0][..]] {
            assert!(prove_exact_multi_order_with_context(&StoreProverContext::cpu(), &layer, &input, &types, invalid, &dirs, &mut Transcript254::new(b"invalid-key-map")).is_err());
        }
    }
    #[test]
    fn exact_three_key_order_and_invalid_rows_reject() {
        let layer = fixture();
        let input = RelationWitness::commit(vec![
            vec![id(&layer,1),id(&layer,1),id(&layer,1)],
            vec![id(&layer,2),id(&layer,2),id(&layer,3)],
            vec![id(&layer,3),id(&layer,4),id(&layer,1)],
        ], 3).unwrap();
        let types = [CanonicalDatatype::UInt32; 3];
        let dirs = [OrderDirection::Descending, OrderDirection::Ascending, OrderDirection::Ascending];
        let result = prove_exact_multi_order_with_context(&StoreProverContext::cpu(), &layer, &input, &types, &[2,0,1], &dirs, &mut Transcript254::new(b"multiorder-3-test")).unwrap();
        assert_eq!(&result.columns[2][..3], &[id(&layer,4),id(&layer,3),id(&layer,1)]);
        verify_exact_multi_order(&layer, &input.view(), &result.columns, &types, &[2,0,1], &dirs, &result.commitments, result.n, result.log_domain, &result.proof, &mut Transcript254::new(b"multiorder-3-test")).unwrap();

        let duplicate = RelationWitness::commit(vec![vec![id(&layer,1);2], vec![id(&layer,2);2]], 2).unwrap();
        let duplicate_result = prove_exact_multi_order_with_context(&StoreProverContext::cpu(), &layer, &duplicate, &types[..2], &[0,1], &dirs[..2], &mut Transcript254::new(b"duplicate")).unwrap();
        assert_eq!(&duplicate_result.columns[0][..2], &[id(&layer,1);2]);
        assert_eq!(&duplicate_result.columns[1][..2], &[id(&layer,2);2]);
        verify_exact_multi_order(&layer, &duplicate.view(), &duplicate_result.columns, &types[..2], &[0,1], &dirs[..2], &duplicate_result.commitments, duplicate_result.n, duplicate_result.log_domain, &duplicate_result.proof, &mut Transcript254::new(b"duplicate")).unwrap();
        let mut changed_columns = duplicate_result.columns.clone();
        changed_columns[0][1] = id(&layer,3);
        let changed_commitments = changed_columns.iter().map(|column| commit_column(column)).collect::<io::Result<Vec<_>>>().unwrap();
        assert!(verify_exact_multi_order(&layer, &duplicate.view(), &changed_columns, &types[..2], &[0,1], &dirs[..2], &changed_commitments, duplicate_result.n, duplicate_result.log_domain, &duplicate_result.proof, &mut Transcript254::new(b"duplicate")).is_err());
        let node_id = ArkFr::from_u64(layer.node_value_catalog.iter().find(|r| matches!(r.triple.object, CanonicalObject::Node(_))).unwrap().object_id);
        let mixed = RelationWitness::commit(vec![vec![id(&layer,1),node_id], vec![id(&layer,2),id(&layer,3)]], 2).unwrap();
        assert!(prove_exact_multi_order_with_context(&StoreProverContext::cpu(), &layer, &mixed, &types[..2], &[0,1], &dirs[..2], &mut Transcript254::new(b"mixed")).is_err());
    }

    #[test]
    fn subset_order_ascending_tiebreak_binds_total_order_and_rejects_tamper() {
        // Models the compiler's `OrderBy([asc(key)])` over a two-column BGP result: the named key is
        // column 0 and the UNNAMED column 1 is appended as an ascending total-order tie-break, so the
        // full-row `key_indices = [0, 1]` are both ascending. Column 0 repeats (1, 1, 2), so the
        // tie-break is load-bearing: it fixes the order WITHIN the column-0 == 1 group.
        let layer = fixture();
        let input = RelationWitness::commit(vec![
            vec![id(&layer, 1), id(&layer, 1), id(&layer, 2)],
            vec![id(&layer, 5), id(&layer, 3), id(&layer, 9)],
        ], 3).unwrap();
        let types = [CanonicalDatatype::UInt32, CanonicalDatatype::UInt32];
        let dirs = [OrderDirection::Ascending, OrderDirection::Ascending];
        let result = prove_exact_multi_order_with_context(
            &StoreProverContext::cpu(), &layer, &input, &types, &[0, 1], &dirs,
            &mut Transcript254::new(b"subset-tiebreak"),
        ).unwrap();
        // Sorted by column 0 ascending, ties on column 0 broken by column 1 ascending.
        assert_eq!(&result.columns[0][..3], &[id(&layer, 1), id(&layer, 1), id(&layer, 2)]);
        assert_eq!(&result.columns[1][..3], &[id(&layer, 3), id(&layer, 5), id(&layer, 9)]);
        verify_exact_multi_order(
            &layer, &input.view(), &result.columns, &types, &[0, 1], &dirs,
            &result.commitments, result.n, result.log_domain, &result.proof,
            &mut Transcript254::new(b"subset-tiebreak"),
        ).unwrap();

        let reject = |columns: Vec<Vec<ArkFr>>, n: usize| {
            let commitments = columns.iter().map(|c| commit_column(c)).collect::<io::Result<Vec<_>>>().unwrap();
            verify_exact_multi_order(
                &layer, &input.view(), &columns, &types, &[0, 1], &dirs,
                &commitments, n, result.log_domain, &result.proof,
                &mut Transcript254::new(b"subset-tiebreak"),
            ).is_err()
        };

        // TIE-BREAK SWAP: exchange the two column-0 == 1 rows. Still a permutation and still sorted on
        // column 0 (1, 1, 2), but column 1 within the tie group becomes descending (5, 3) — rejected.
        let mut tie_swapped = result.columns.clone();
        for column in tie_swapped.iter_mut() { column.swap(0, 1); }
        assert!(reject(tie_swapped, result.n));

        // REORDER across the primary key: swap the first and last rows — column 0 is no longer sorted.
        let mut reordered = result.columns.clone();
        for column in reordered.iter_mut() { column.swap(0, 2); }
        assert!(reject(reordered, result.n));

        // NON-PERMUTATION: overwrite a tie-break value with another catalog id absent from the input
        // multiset (row (2, 9) -> (2, 8)); the order still holds but the multiset check rejects.
        let mut altered = result.columns.clone();
        altered[1][2] = id(&layer, 8);
        assert!(reject(altered, result.n));

        // TRUNCATION: claiming one fewer row is not a full-row permutation of the three input rows.
        let truncated = result.columns.iter().map(|c| { let mut c = c.clone(); c[2] = ArkFr::zero(); c }).collect::<Vec<_>>();
        assert!(reject(truncated, result.n - 1));
    }
}
