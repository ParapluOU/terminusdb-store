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

use std::collections::BTreeMap;
use std::{cmp::Ordering, io};
use ark_ff::PrimeField;
use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::transcript::Transcript;
use super::argument::{commit_column, StoreProverContext, Transcript254};
use super::bgp::{RelationCommit, RelationWitness};
use super::commitment::{
    node_value_op_value, reject_nan_objects, LayerCommitment, ResolutionOpenings,
    VerifierCommitmentRef, VerifierLayer,
};
use super::query::{
    build_resolution_openings, canonical_dictionary_id, require_base_layer_for_query,
    resolve_node_value_id_via_merkle, verify_dict_value_terms,
};
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
    /// Merkle-dict I2b step 3 (value↔term consistency, TERM-LESS multi-key order). One
    /// positional Merkle opening per DISTINCT id touched across ALL key columns (node/value
    /// namespace). `None` on a TERM-BEARING artifact (the catalog re-MSM ties each key's
    /// value/datatype to real terms). On a TERM-LESS artifact the verifier rederives each key
    /// column's sort values from these openings and ties them to the root-bound output-id
    /// commitments per column ([`verify_dict_value_terms`]); the per-column `oid` vector is the
    /// shipped `result_columns[k]` (already bound to `result_commitments[k]`), so no separate
    /// `oid` field is needed. UNCOMMITTED — a missing opening (or recommit mismatch) fails closed.
    pub value_openings: Option<ResolutionOpenings>,
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

/// Merkle-dict I2b step 3 (TERM-LESS value↔term resolve). For each key column, decode its sort
/// values from the Merkle-authenticated terms at the shipped output ids (`output_columns[k]`) and
/// tie them to the root-bound `output_commitments[k]` via the audited [`verify_dict_value_terms`].
///
/// The per-column `oid` is the shipped `output_columns[k]` itself (already bound to
/// `output_commitments[k]` by the enclosing envelope's `verify_result_columns`); the tie re-forces
/// `Com(oid) == output_commitments[k]` HERE too, so a forged `output_columns[k]` recommits
/// differently and rejects. The locally-built `values`/`dt_col` are consistency-checked by the tie
/// (`Com(values) == com_v_val` etc.), so the returned per-row value columns are exactly
/// OP-encode(real term) — the columns `lexsort` consumes. Missing openings / non-value ids / a
/// datatype mismatch all fail closed.
fn resolve_values_termless(
    view: VerifierCommitmentRef<'_>,
    output_columns: &[Vec<ArkFr>],
    output_commitments: &[ArkGT],
    n: usize,
    datatypes: &[CanonicalDatatype],
    domain: usize,
    proof: &ExactMultiOrderProof,
) -> io::Result<Vec<Vec<ArkFr>>> {
    view.validate()?;
    require_base_layer_for_query(view.state)?;
    let value_openings = proof.value_openings.as_ref().ok_or_else(|| {
        invalid("term-less exact multi-column OrderBy requires the prover-shipped value↔term Merkle openings")
    })?;
    if output_commitments.len() != output_columns.len() {
        return Err(invalid("multi-order term-less output/commitment shape mismatch"));
    }
    let trusted_root = view.resolution_root();
    output_columns.iter().zip(datatypes).zip(output_commitments).map(|((column, datatype), com)| {
        if column.len() < n { return Err(invalid("multi-order output column too short")); }
        let mut oid: Vec<u64> = Vec::with_capacity(n);
        let mut values = vec![ArkFr::zero(); domain];
        let mut dt_col = vec![ArkFr::zero(); domain];
        let mut cache: BTreeMap<u64, (ArkFr, ArkFr)> = BTreeMap::new();
        for (i, id) in column[..n].iter().enumerate() {
            let raw = canonical_dictionary_id(id)?;
            if raw == 0 { return Err(invalid("multi-order key id must be a nonzero dictionary id")); }
            oid.push(raw);
            let (val, dt) = if let Some(entry) = cache.get(&raw) {
                *entry
            } else {
                let opened = resolve_node_value_id_via_merkle(id, view, None, Some(value_openings))?;
                let actual = match &opened {
                    CanonicalObject::Value { datatype, .. } => *datatype,
                    _ => return Err(invalid("multi-order key is not a value")),
                };
                if actual != *datatype { return Err(invalid("multi-order datatype mask is not complete")); }
                let val = node_value_op_value(&opened)?;
                let dt = ArkFr::from_u64(actual as u8 as u64);
                cache.insert(raw, (val, dt));
                (val, dt)
            };
            values[i] = val;
            dt_col[i] = dt;
        }
        // Audited tie: binds this column's ids to the root-bound output_commitments[k] and
        // re-verifies value/datatype == the Merkle terms (consistency-checking the resolve above).
        let com_v_val = commit_column(&values)?;
        let com_v_dt = commit_column(&dt_col)?;
        let touched = verify_dict_value_terms(com, &com_v_val, &com_v_dt, n, domain, &oid, value_openings, view, trusted_root)?;
        reject_nan_objects(&touched, ArkFr::from_u64(*datatype as u64))?;
        Ok(values)
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
    // Merkle-dict I2b step 3: term-bearing by default (the catalog re-MSM ties value/datatype to
    // terms). `build_termless_multi_order_openings` fills `value_openings` for a TERM-LESS artifact.
    Ok(ExactMultiOrderResult { proof: ExactMultiOrderProof { permutation: permutation.proof, lexicographic, value_openings: None }, commitments: permutation.commitments, columns: permutation.columns, log_domain: permutation.log_domain, n: permutation.n })
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
    let view = source.verifier_view();
    // Merkle-dict I2b step 3 (value↔term consistency). Each key column's sort values are its ids'
    // order-preserving dict VALUEs. TERM-BEARING: `resolve_values` decodes them from the root-bound
    // catalog (re-MSM tie). TERM-LESS: the catalog is dropped ⇒ decode from the Merkle-authenticated
    // terms via the prover-shipped openings and tie each column to its root-bound output-id
    // commitment. Missing openings ⇒ Err (fail-closed). DELTA order stays fail-closed.
    let values = if view.terms_present {
        resolve_values(source, output_columns, output_len, datatypes, domain)?
    } else {
        resolve_values_termless(view, output_columns, output_commitments, output_len, datatypes, domain, proof)?
    };
    let commitments = values.iter().map(|column| commit_column(column)).collect::<io::Result<Vec<_>>>()?;
    let mut chi = vec![ArkFr::zero(); domain]; chi[..output_len].fill(ArkFr::one());
    let key_commitments = key_indices.iter().map(|i| commitments[*i]).collect::<Vec<_>>();
    verify_lexicographically_nondecreasing(&key_commitments, directions, &commit_column(&chi)?, output_len, &proof.lexicographic, transcript)
}

/// Merkle-dict I2b step 3 (producer): mint the TERM-LESS value↔term witness for a proven
/// base/rollup exact-multi-order proof. `output_columns` are the SORTED output id columns
/// (`result.columns`); `n` is the real row count. Opens each DISTINCT id touched across ALL key
/// columns against the node/value tree, populating `proof.value_openings` IN PLACE. The per-column
/// `oid` is the shipped `output_columns[k]` itself, so no `oid` field is minted. UNCOMMITTED
/// witness; BASE/ROLLUP single-layer ONLY (delta multi-order stays fail-closed).
pub fn build_termless_multi_order_openings(
    layer: &LayerCommitment,
    proof: &mut ExactMultiOrderProof,
    output_columns: &[Vec<ArkFr>],
    n: usize,
) -> io::Result<()> {
    require_base_layer_for_query(layer.verifier_view().state)?;
    let mut nv_ids: Vec<u64> = Vec::new();
    for column in output_columns {
        for value in column.iter().take(n) {
            let raw = canonical_dictionary_id(value)?;
            if raw != 0 { nv_ids.push(raw); }
        }
    }
    // `build_resolution_openings` dedups + routes each id to the layer's interval.
    let openings = build_resolution_openings(&[layer.verifier_view()], &nv_ids, &[])?;
    proof.value_openings = Some(openings);
    Ok(())
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
        // Merkle-dict I2b step 3: a term-less multi-order proof WITHOUT the value openings still
        // fails closed, now with the step-3 "value↔term Merkle openings" message.
        assert!(err.to_string().contains("Merkle openings"), "{err}");
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

    // =====================================================================================
    // Merkle-dict I2b step 3 (value↔term consistency, MULTI-KEY ORDER). A TERM-LESS base/rollup
    // multi-order proof rederives each key column's sort values from the Merkle-authenticated terms
    // and ties them (per column) to the root-bound output-id commitments. Term-bearing unchanged.
    // =====================================================================================
    fn prove_attach_multi(layer: &LayerCommitment) -> (RelationWitness, ExactMultiOrderResult, [CanonicalDatatype; 2], [usize; 2], [OrderDirection; 2]) {
        let input = RelationWitness::commit(vec![
            vec![id(layer,2),id(layer,1),id(layer,1)],
            vec![id(layer,8),id(layer,3),id(layer,9)],
        ], 3).unwrap();
        let types = [CanonicalDatatype::UInt32, CanonicalDatatype::UInt32];
        let keys = [1usize, 0];
        let dirs = [OrderDirection::Descending, OrderDirection::Ascending];
        let mut result = prove_exact_multi_order_with_context(&StoreProverContext::cpu(), layer, &input, &types, &keys, &dirs, &mut Transcript254::new(b"mo-step3")).unwrap();
        let n = result.n;
        build_termless_multi_order_openings(layer, &mut result.proof, &result.columns, n).unwrap();
        assert!(result.proof.value_openings.is_some(), "multi-order value openings attached");
        (input, result, types, keys, dirs)
    }

    /// Differential positive: a TERM-LESS multi-key order proof verifies EXACTLY like the
    /// TERM-BEARING one — same accept (honest) and same reject (wrong key priority).
    #[test]
    fn i2b_step3_multi_order_termless_differential() {
        let layer = fixture();
        let (input, result, types, keys, dirs) = prove_attach_multi(&layer);
        let termless = layer.verifier_commitment_termless().unwrap();
        verify_exact_multi_order(&layer, &input.view(), &result.columns, &types, &keys, &dirs, &result.commitments, result.n, result.log_domain, &result.proof, &mut Transcript254::new(b"mo-step3")).unwrap();
        verify_exact_multi_order(&termless, &input.view(), &result.columns, &types, &keys, &dirs, &result.commitments, result.n, result.log_domain, &result.proof, &mut Transcript254::new(b"mo-step3")).unwrap();
        // Same reject: a different key priority rejects on BOTH artifacts.
        assert!(verify_exact_multi_order(&layer, &input.view(), &result.columns, &types, &[0,1], &dirs, &result.commitments, result.n, result.log_domain, &result.proof, &mut Transcript254::new(b"mo-step3")).is_err());
        assert!(verify_exact_multi_order(&termless, &input.view(), &result.columns, &types, &[0,1], &dirs, &result.commitments, result.n, result.log_domain, &result.proof, &mut Transcript254::new(b"mo-step3")).is_err());
    }

    /// A forged output-id column (`output_columns[k]`) no longer recommits to `output_commitments[k]`
    /// ⇒ REJECT (the per-column tie binds it; the gadget is self-contained, not relying on the
    /// enclosing `verify_result_columns`).
    #[test]
    fn i2b_step3_multi_order_termless_rejects_forged_output_ids() {
        let layer = fixture();
        let (input, result, types, keys, dirs) = prove_attach_multi(&layer);
        let termless = layer.verifier_commitment_termless().unwrap();
        let mut forged = result.columns.clone();
        // Forge column 0 row 0 to a DIFFERENT real dict id that IS present in the shipped openings
        // (id(8) appears in column 1), so the resolve opens it fine and the TIE's `oid` recommit is
        // the load-bearing check: Com(oid) != output_commitments[0] ⇒ REJECT.
        forged[0][0] = id(&layer, 8);
        let err = verify_exact_multi_order(&termless, &input.view(), &forged, &types, &keys, &dirs, &result.commitments, result.n, result.log_domain, &result.proof, &mut Transcript254::new(b"mo-step3")).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("oid column"), "GOT: {}", msg);
    }

    /// A term-less multi-order proof WITHOUT the value openings fails closed; term-bearing verifies.
    #[test]
    fn i2b_step3_multi_order_termless_rejects_missing_openings() {
        let layer = fixture();
        let (input, mut result, types, keys, dirs) = prove_attach_multi(&layer);
        result.proof.value_openings = None;
        let termless = layer.verifier_commitment_termless().unwrap();
        let err = verify_exact_multi_order(&termless, &input.view(), &result.columns, &types, &keys, &dirs, &result.commitments, result.n, result.log_domain, &result.proof, &mut Transcript254::new(b"mo-step3")).unwrap_err();
        assert!(err.to_string().contains("Merkle openings"), "{err}");
        verify_exact_multi_order(&layer, &input.view(), &result.columns, &types, &keys, &dirs, &result.commitments, result.n, result.log_domain, &result.proof, &mut Transcript254::new(b"mo-step3")).unwrap();
    }
}
