//! Verifier-recomputed derived columns for the generated / aggregate operators.
//!
//! Some operators do not need a bespoke cryptographic gadget: once the child relation and the
//! root-bound dictionary are authenticated and public to the verifier, the verifier can simply
//! RE-DERIVE the output column itself and compare. That is the "verifier-recompute" pattern,
//! and this module is where those recomputations live:
//!
//! * generated-decimal row filters ([`generated_u64_filtered_columns`] /
//!   [`generated_i64_filtered_columns`]) — keep the child rows whose generated value satisfies
//!   the comparison;
//! * grouped and whole-collection aggregates — `GROUP BY` / `findall`-style sum, count and
//!   min/max over fixed-width integral Store values (`derived_group_*`, `derived_collection_*`,
//!   `derived_cardinality_column`).
//!
//! [`GroupSumValue`] carries a signed/unsigned accumulator wide enough (i128) that no
//! intermediate overflow is possible within the envelope's row cap, while still requiring the
//! final result to fit its declared generated domain. Floating types are refused here as
//! defense-in-depth, because their total-order encoding places NaN differently from Prolog's
//! comparison.
#![allow(unused_imports)]

use super::plan::{integral_op_datatype, CollectionSumTemplate, GeneratedU64ComparisonOp};
use super::recursive::RecursiveRelationEnvelope;
use super::wire::invalid;
use std::convert::{TryFrom, TryInto};
use std::io::{self, Cursor};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_ff::PrimeField;
use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::transcript::Transcript;
use sha2::{Digest, Sha256};

use crate::proof::anti::{
    prove_global_left_outer_join_with_context, prove_global_not_with_context,
    prove_inner_join_with_keys_and_context, prove_left_anti_join_with_keys_and_context, prove_left_outer_join_with_keys_and_context,
    verify_global_left_outer_join, verify_global_not, verify_left_anti_join_with_keys,
    verify_inner_join_with_keys, verify_left_outer_join_with_keys, InnerJoinProof, InnerJoinResult,
    LeftAntiJoinProof, LeftOuterJoinProof,
};
use crate::proof::argument::{commit_column, ColumnOpening, StoreProverContext, SumcheckProof, Transcript254};
use crate::proof::bgp::{BgpChainProof, BgpStageProof, JoinStep, RelationCommit, RelationWitness};
use crate::proof::commitment::{LayerCommitment, VerifierLayer};
use crate::proof::count::CountProof;
use crate::proof::dict::DictProof;
use crate::proof::dicthash::{
    prove_dict_hash_lookup, verify_dict_hash_lookup, DictHashLookupProof, HashLookupInnerProof,
};
use crate::proof::semijoin::{
    prove_hash_semijoin, verify_hash_semijoin, HashSemiJoinProof, SelMaskProof,
};
use crate::proof::distinct::{
    prove_relation_distinct_with_context, verify_relation_distinct,
    RelationDistinctEntryProof, RelationDistinctProof, RelationDistinctResult,
};
use crate::proof::disequality::{
    prove_tuple_disequality_with_context, verify_tuple_disequality, DisequalityNode,
    DisequalityPair, TupleDisequalityProof, TupleDisequalityStageProof,
};
use crate::proof::equality::{
    prove_relation_equality_with_context, verify_relation_equality, RelationEqualityConstraint,
    RelationEqualityProof,
};
use crate::proof::filter::{
    AtomFilterProof, BooleanProductProof, CompactionOnlyProof, FilterProof, FullFilterProof,
    SelectionOnlyProof,
};
use crate::proof::group::{GroupExtremaEntryProof, GroupExtremaKind, GroupExtremaProof};
use crate::proof::join2::{BinaryJoinCommitments, BinaryJoinProof, BinaryPairsProof};
use crate::proof::membership::MembershipProof;
use crate::proof::multiorder::{prove_exact_multi_order_with_context, verify_exact_multi_order, ExactMultiOrderProof};
use crate::proof::lexsort::LexicographicSortedProof;
use crate::proof::order::{
    prove_exact_scalar_order_with_context, verify_exact_scalar_order, ExactScalarOrderProof,
    OrderDirection,
};
use crate::proof::permutation::RelationPermutationProof;
use crate::proof::query::{
    has_ground_result_unit, prove_layer_bgp_with_result_and_context,
    resolve_typed_result_objects, result_id_namespaces,
    resolve_result_values, verify_layer_bgp, verify_layer_bgp_result_log_domain,
    verify_result_columns, generated_decimal_i64_field, generated_decimal_u64_value,
    AtomTerm, CanonicalResultValue, ColumnSource,
    DictionaryAbsence,
    DictionaryAbsencePosition, LayerBgpAtom,
    LayerBgpAtomProof, LayerBgpPlan, LayerBgpProof, ObjectKindFilter, ObjectRangeBound,
    ObjectRangeSpec, RangeAtomSubproof, ResultColumnDescriptor, ResultIdNamespace,
    SecondRangeSubproof,
};
use crate::proof::range::{
    DictLookupProof, ObjectRangeCombineProof, ObjectRangeProof, RangeOp, RangeSelectProof,
};
use crate::proof::union::{prove_bag_union_with_context, BagUnionResult};
use crate::proof::relocate::{prove_column_relocation_with_context, verify_column_relocation, RelocationProof};
use crate::proof::sort::SortedProof;
use crate::proof::union::{verify_bag_union, BagUnionProof};
use crate::proof::{CanonicalDatatype, CanonicalObject, ProofHash};
pub(super) fn generated_u64_comparison(
    value: u64,
    op: GeneratedU64ComparisonOp,
    constant: u64,
    constant_on_left: bool,
) -> bool {
    let (left, right) = if constant_on_left {
        (constant, value)
    } else {
        (value, constant)
    };
    match op {
        GeneratedU64ComparisonOp::Eq => left == right,
        GeneratedU64ComparisonOp::Lt => left < right,
        GeneratedU64ComparisonOp::Le => left <= right,
        GeneratedU64ComparisonOp::Gt => left > right,
        GeneratedU64ComparisonOp::Ge => left >= right,
    }
}

pub(super) fn generated_i64_comparison(
    value: i64,
    op: GeneratedU64ComparisonOp,
    constant: i64,
    constant_on_left: bool,
) -> bool {
    let (left, right) = if constant_on_left {
        (constant, value)
    } else {
        (value, constant)
    };
    match op {
        GeneratedU64ComparisonOp::Eq => left == right,
        GeneratedU64ComparisonOp::Lt => left < right,
        GeneratedU64ComparisonOp::Le => left <= right,
        GeneratedU64ComparisonOp::Gt => left > right,
        GeneratedU64ComparisonOp::Ge => left >= right,
    }
}

pub(super) fn generated_u64_filtered_columns(
    input: &RecursiveRelationEnvelope,
    column: usize,
    op: GeneratedU64ComparisonOp,
    constant: u64,
    constant_on_left: bool,
) -> io::Result<Vec<Vec<ArkFr>>> {
    let values = input
        .result_values
        .get(column)
        .ok_or_else(|| invalid("generated-u64 filter value column out of range"))?;
    if values.len() != input.result_len {
        return Err(invalid("generated-u64 filter input value length mismatch"));
    }
    let mut keep = Vec::new();
    for (row, value) in values.iter().enumerate() {
        let CanonicalResultValue::GeneratedDecimalU64(value) = value else {
            return Err(invalid("generated-u64 filter input is not a generated decimal"));
        };
        if generated_u64_comparison(*value, op, constant, constant_on_left) {
            keep.push(row);
        }
    }
    input
        .result_columns
        .iter()
        .map(|input_column| {
            keep.iter()
                .map(|row| {
                    input_column
                        .get(*row)
                        .copied()
                        .ok_or_else(|| invalid("generated-u64 filter input column is short"))
                })
                .collect()
        })
        .collect()
}

pub(super) fn generated_i64_filtered_columns(
    input: &RecursiveRelationEnvelope,
    column: usize,
    op: GeneratedU64ComparisonOp,
    constant: i64,
    constant_on_left: bool,
) -> io::Result<Vec<Vec<ArkFr>>> {
    let values = input
        .result_values
        .get(column)
        .ok_or_else(|| invalid("generated-i64 filter value column out of range"))?;
    if values.len() != input.result_len {
        return Err(invalid("generated-i64 filter input value length mismatch"));
    }
    let mut keep = Vec::new();
    for (row, value) in values.iter().enumerate() {
        let CanonicalResultValue::GeneratedDecimalI64(value) = value else {
            return Err(invalid(
                "generated-i64 filter input is not a signed generated decimal",
            ));
        };
        if generated_i64_comparison(*value, op, constant, constant_on_left) {
            keep.push(row);
        }
    }
    input
        .result_columns
        .iter()
        .map(|input_column| {
            keep.iter()
                .map(|row| {
                    input_column
                        .get(*row)
                        .copied()
                        .ok_or_else(|| invalid("generated-i64 filter input column is short"))
                })
                .collect()
        })
        .collect()
}

#[derive(Clone, Copy)]
pub(super) enum GroupSumValue {
    Unsigned(u64),
    // Signed Prolog arithmetic is unbounded. The public envelope caps rows at four
    // million, so i128 can accumulate every possible i64 input without intermediate
    // overflow; only the final result is required to fit the generated i64 domain.
    Signed(i128),
}

impl GroupSumValue {
    fn zero(datatype: CanonicalDatatype) -> io::Result<Self> {
        match datatype {
            CanonicalDatatype::UInt8
            | CanonicalDatatype::UInt16
            | CanonicalDatatype::UInt32
            | CanonicalDatatype::UInt64 => Ok(Self::Unsigned(0)),
            CanonicalDatatype::Int8
            | CanonicalDatatype::Int16
            | CanonicalDatatype::Int32
            | CanonicalDatatype::Int64 => Ok(Self::Signed(0)),
            _ => Err(invalid("group-sum datatype is outside the integral subset")),
        }
    }

    fn checked_add(self, rhs: Self) -> io::Result<Self> {
        match (self, rhs) {
            (Self::Unsigned(left), Self::Unsigned(right)) => left
                .checked_add(right)
                .map(Self::Unsigned)
                .ok_or_else(|| invalid("group-sum unsigned arithmetic exceeds u64")),
            (Self::Signed(left), Self::Signed(right)) => left
                .checked_add(right)
                .map(Self::Signed)
                .ok_or_else(|| invalid("group-sum signed arithmetic exceeds i64")),
            _ => Err(invalid("group-sum mixes signed and unsigned values")),
        }
    }

    fn field(self) -> io::Result<ArkFr> {
        match self {
            Self::Unsigned(value) => Ok(ArkFr::from_u64(value)),
            Self::Signed(value) => Ok(generated_decimal_i64_field(
                i64::try_from(value)
                    .map_err(|_| invalid("group-sum signed result exceeds i64"))?,
            )),
        }
    }
}

pub(super) fn integral_catalog_value(
    object: &CanonicalObject,
    datatype: CanonicalDatatype,
) -> io::Result<GroupSumValue> {
    let CanonicalObject::Value {
        datatype: actual,
        lexical,
    } = object
    else {
        return Err(invalid("group-sum value is not a typed literal"));
    };
    if *actual != datatype {
        return Err(invalid("group-sum value has the wrong exact datatype"));
    }
    Ok(match datatype {
        CanonicalDatatype::UInt8 if lexical.len() == 1 => GroupSumValue::Unsigned(u64::from(lexical[0])),
        CanonicalDatatype::UInt16 if lexical.len() == 2 => {
            GroupSumValue::Unsigned(u64::from(u16::from_be_bytes(lexical.as_slice().try_into().unwrap())))
        }
        CanonicalDatatype::UInt32 if lexical.len() == 4 => {
            GroupSumValue::Unsigned(u64::from(u32::from_be_bytes(lexical.as_slice().try_into().unwrap())))
        }
        CanonicalDatatype::UInt64 if lexical.len() == 8 => {
            GroupSumValue::Unsigned(u64::from_be_bytes(lexical.as_slice().try_into().unwrap()))
        }
        CanonicalDatatype::Int8 if lexical.len() == 1 => {
            GroupSumValue::Signed(i128::from((lexical[0] ^ 0x80) as i8))
        }
        CanonicalDatatype::Int16 if lexical.len() == 2 => {
            let encoded = u16::from_be_bytes(lexical.as_slice().try_into().unwrap());
            GroupSumValue::Signed(i128::from((encoded ^ (1 << 15)) as i16))
        }
        CanonicalDatatype::Int32 if lexical.len() == 4 => {
            let encoded = u32::from_be_bytes(lexical.as_slice().try_into().unwrap());
            GroupSumValue::Signed(i128::from((encoded ^ (1 << 31)) as i32))
        }
        CanonicalDatatype::Int64 if lexical.len() == 8 => {
            let encoded = u64::from_be_bytes(lexical.as_slice().try_into().unwrap());
            GroupSumValue::Signed(i128::from((encoded ^ (1 << 63)) as i64))
        }
        CanonicalDatatype::UInt8
        | CanonicalDatatype::UInt16
        | CanonicalDatatype::UInt32
        | CanonicalDatatype::UInt64
        | CanonicalDatatype::Int8
        | CanonicalDatatype::Int16
        | CanonicalDatatype::Int32
        | CanonicalDatatype::Int64 => {
            return Err(invalid("group-sum value has a noncanonical lexical width"))
        }
        _ => return Err(invalid("group-sum datatype is outside the integral subset")),
    })
}

pub(super) fn grouped_node_rows(
    layer: &impl VerifierLayer,
    input: &RecursiveRelationEnvelope,
    key_columns: &[usize],
) -> io::Result<std::collections::BTreeMap<Vec<Vec<u8>>, (Vec<u64>, Vec<usize>)>> {
    if key_columns.is_empty()
        || key_columns.len() > 2
        || (key_columns.len() == 2 && key_columns[0] == key_columns[1])
    {
        return Err(invalid("group key tuple is invalid"));
    }
    let view = layer.verifier_view();
    view.validate()?;
    let keys = key_columns
        .iter()
        .map(|key_column| {
            let descriptor = input
                .result_descriptors
                .get(*key_column)
                .copied()
                .ok_or_else(|| invalid("group key descriptor out of range"))?;
            if descriptor.nullable || descriptor.namespace != ResultIdNamespace::NodeValue {
                return Err(invalid("group key is not a nonnullable node"));
            }
            let column = input
                .result_columns
                .get(*key_column)
                .ok_or_else(|| invalid("group key column out of range"))?;
            if column.len() != input.result_len {
                return Err(invalid("group key length mismatch"));
            }
            Ok(column)
        })
        .collect::<io::Result<Vec<_>>>()?;
    let mut groups = std::collections::BTreeMap::<
        Vec<Vec<u8>>,
        (Vec<u64>, Vec<usize>),
    >::new();
    for row in 0..input.result_len {
        let resolved = keys
            .iter()
            .map(|column| {
                let field = column[row];
                let record = view
                    .node_value_catalog
                    .iter()
                    .find(|record| ArkFr::from_u64(record.object_id) == field)
                    .ok_or_else(|| invalid("group key is absent from the node dictionary"))?;
                let CanonicalObject::Node(iri) = &record.triple.object else {
                    return Err(invalid("group key is not a node IRI"));
                };
                Ok((iri.clone(), record.object_id))
            })
            .collect::<io::Result<Vec<_>>>()?;
        let canonical_key = resolved.iter().map(|(iri, _)| iri.clone()).collect::<Vec<_>>();
        let key_ids = resolved.iter().map(|(_, id)| *id).collect::<Vec<_>>();
        let entry = groups
            .entry(canonical_key)
            .or_insert((key_ids.clone(), Vec::new()));
        if entry.0 != key_ids {
            return Err(invalid("canonical node tuple maps to multiple Store IDs"));
        }
        entry.1.push(row);
    }
    Ok(groups)
}

pub(super) fn derived_group_sum_columns(
    layer: &impl VerifierLayer,
    input: &RecursiveRelationEnvelope,
    key_columns: &[usize],
    value_column: usize,
    datatype: CanonicalDatatype,
) -> io::Result<Vec<Vec<ArkFr>>> {
    let values = input
        .result_columns
        .get(value_column)
        .ok_or_else(|| invalid("group-sum value column out of range"))?;
    if values.len() != input.result_len {
        return Err(invalid("group-sum input column length mismatch"));
    }
    let view = layer.verifier_view();
    view.validate()?;
    let resolve_value = |field: ArkFr| -> io::Result<GroupSumValue> {
        let record = view
            .node_value_catalog
            .iter()
            .find(|record| ArkFr::from_u64(record.object_id) == field)
            .ok_or_else(|| invalid("group-sum value id is absent from the authenticated dictionary"))?;
        integral_catalog_value(&record.triple.object, datatype)
    };
    if key_columns.is_empty() {
        if input.result_len == 0 {
            return Ok(vec![Vec::new()]);
        }
        let mut sum = GroupSumValue::zero(datatype)?;
        for field in values {
            sum = sum.checked_add(resolve_value(*field)?)?;
        }
        return Ok(vec![vec![sum.field()?]]);
    }
    if key_columns.iter().any(|key| *key == value_column) {
        return Err(invalid("group-sum key tuple is invalid"));
    }
    // SWI bagof orders free-variable tuples lexicographically in the declared
    // `group_by` variable order. Resolve authenticated expanded IRI bytes for every
    // component; never sort by child-schema position or append-order Store IDs.
    let groups = grouped_node_rows(layer, input, key_columns)?;
    let mut output = vec![Vec::with_capacity(groups.len()); key_columns.len() + 1];
    for (key_ids, rows) in groups.values() {
        let mut sum = GroupSumValue::zero(datatype)?;
        for row in rows {
            sum = sum.checked_add(resolve_value(values[*row])?)?;
        }
        for (column, key_id) in output.iter_mut().zip(key_ids) {
            column.push(ArkFr::from_u64(*key_id));
        }
        output[key_columns.len()].push(sum.field()?);
    }
    Ok(output)
}

pub(super) fn derived_group_count_columns(
    layer: &impl VerifierLayer,
    input: &RecursiveRelationEnvelope,
    key_columns: &[usize],
) -> io::Result<Vec<Vec<ArkFr>>> {
    if key_columns.is_empty() {
        return derived_cardinality_column(input.result_len, false);
    }
    let groups = grouped_node_rows(layer, input, key_columns)?;
    let mut output = vec![Vec::with_capacity(groups.len()); key_columns.len() + 1];
    for (key_ids, rows) in groups.values() {
        for (column, key_id) in output.iter_mut().zip(key_ids) {
            column.push(ArkFr::from_u64(*key_id));
        }
        output[key_columns.len()].push(ArkFr::from_u64(
            u64::try_from(rows.len())
                .map_err(|_| invalid("group-count cardinality exceeds u64"))?,
        ));
    }
    Ok(output)
}

pub(super) fn derived_cardinality_column(result_len: usize, emit_zero_for_empty: bool) -> io::Result<Vec<Vec<ArkFr>>> {
    if result_len == 0 && !emit_zero_for_empty {
        return Ok(vec![Vec::new()]);
    }
    Ok(vec![vec![ArkFr::from_u64(
        u64::try_from(result_len).map_err(|_| invalid("collection cardinality exceeds u64"))?,
    )]])
}

pub(super) fn derived_collection_sum_column(
    layer: &impl VerifierLayer,
    input: &RecursiveRelationEnvelope,
    template: CollectionSumTemplate,
    datatype: CanonicalDatatype,
) -> io::Result<Vec<Vec<ArkFr>>> {
    let mut sum = GroupSumValue::zero(datatype)?;
    match template {
        CollectionSumTemplate::Column(column) => {
            let values = input
                .result_columns
                .get(column)
                .ok_or_else(|| invalid("collection-sum value column out of range"))?;
            if values.len() != input.result_len {
                return Err(invalid("collection-sum input column length mismatch"));
            }
            let view = layer.verifier_view();
            view.validate()?;
            for field in values.iter().take(input.result_len) {
                let record = view
                    .node_value_catalog
                    .iter()
                    .find(|record| ArkFr::from_u64(record.object_id) == *field)
                    .ok_or_else(|| {
                        invalid("collection-sum value id is absent from the authenticated dictionary")
                    })?;
                sum = sum.checked_add(integral_catalog_value(&record.triple.object, datatype)?)?;
            }
        }
        CollectionSumTemplate::UnsignedConstant(value) => {
            for _ in 0..input.result_len {
                sum = sum.checked_add(GroupSumValue::Unsigned(value))?;
            }
        }
        CollectionSumTemplate::SignedConstant(value) => {
            for _ in 0..input.result_len {
                sum = sum.checked_add(GroupSumValue::Signed(i128::from(value)))?;
            }
        }
    }
    Ok(vec![vec![sum.field()?]])
}

pub(super) fn derived_collection_extrema_column(
    layer: &impl VerifierLayer,
    input: &RecursiveRelationEnvelope,
    value_column: usize,
    datatype: CanonicalDatatype,
    kind: GroupExtremaKind,
) -> io::Result<Vec<Vec<ArkFr>>> {
    // Defense-in-depth: `node_value_op_value` also encodes Float32/64, whose total-order NaN
    // placement diverges from Prolog's IEEE comparison. Assert the integral OP subset here so a
    // future loosening of the `shape_bounded` gate cannot silently admit floating extrema.
    if !integral_op_datatype(datatype) {
        return Err(invalid(
            "collection-extrema datatype must be a fixed-width integral OP type",
        ));
    }
    let values = input
        .result_columns
        .get(value_column)
        .ok_or_else(|| invalid("collection-extrema value column out of range"))?;
    if values.len() != input.result_len {
        return Err(invalid("collection-extrema input column length mismatch"));
    }
    let view = layer.verifier_view();
    view.validate()?;
    let mut selected: Option<(ArkFr, ark_bn254::Fr)> = None;
    for field in values.iter().take(input.result_len) {
        let record = view
            .node_value_catalog
            .iter()
            .find(|record| ArkFr::from_u64(record.object_id) == *field)
            .ok_or_else(|| invalid("collection-extrema value is absent from the catalog"))?;
        let CanonicalObject::Value {
            datatype: actual, ..
        } = &record.triple.object
        else {
            return Err(invalid("collection-extrema value is not a typed literal"));
        };
        if *actual != datatype {
            return Err(invalid("collection-extrema value has the wrong exact datatype"));
        }
        let encoding = crate::proof::commitment::node_value_op_value(&record.triple.object)?.0;
        let replace = selected.as_ref().is_none_or(|(_, current)| match kind {
            GroupExtremaKind::Min => encoding.into_bigint() < current.into_bigint(),
            GroupExtremaKind::Max => encoding.into_bigint() > current.into_bigint(),
        });
        if replace {
            selected = Some((*field, encoding));
        }
    }
    Ok(vec![selected.map(|(field, _)| vec![field]).unwrap_or_default()])
}

pub(super) fn derived_group_extrema_columns(
    layer: &impl VerifierLayer,
    input: &RecursiveRelationEnvelope,
    key_columns: &[usize],
    value_column: usize,
    datatype: CanonicalDatatype,
    kind: GroupExtremaKind,
) -> io::Result<Vec<Vec<ArkFr>>> {
    // Defense-in-depth: reject floating (NaN-divergent) extrema at derivation time, independent
    // of the `shape_bounded` integral gate. `integral_catalog_value` also enforces this per value.
    if !integral_op_datatype(datatype) {
        return Err(invalid(
            "group-extrema datatype must be a fixed-width integral OP type",
        ));
    }
    let values = input
        .result_columns
        .get(value_column)
        .ok_or_else(|| invalid("group-extrema value column out of range"))?;
    if values.len() != input.result_len || key_columns.iter().any(|key| *key == value_column) {
        return Err(invalid("group-extrema input shape mismatch"));
    }
    let view = layer.verifier_view();
    view.validate()?;
    let resolve = |field: ArkFr| -> io::Result<ark_bn254::Fr> {
        let record = view
            .node_value_catalog
            .iter()
            .find(|record| ArkFr::from_u64(record.object_id) == field)
            .ok_or_else(|| invalid("group-extrema value is absent from the catalog"))?;
        // This enforces both the exact integral datatype and its canonical width.
        integral_catalog_value(&record.triple.object, datatype)?;
        Ok(crate::proof::commitment::node_value_op_value(&record.triple.object)?.0)
    };
    let choose = |rows: &[usize]| -> io::Result<Option<ArkFr>> {
        let mut selected: Option<(ArkFr, ark_bn254::Fr)> = None;
        for row in rows {
            let field = values[*row];
            let encoding = resolve(field)?;
            let replace = match selected.as_ref() {
                None => true,
                Some((_, current)) => match kind {
                    GroupExtremaKind::Min => encoding.into_bigint() < current.into_bigint(),
                    GroupExtremaKind::Max => encoding.into_bigint() > current.into_bigint(),
                },
            };
            if replace {
                selected = Some((field, encoding));
            }
        }
        Ok(selected.map(|(field, _)| field))
    };
    if key_columns.is_empty() {
        let rows = (0..input.result_len).collect::<Vec<_>>();
        return Ok(vec![choose(&rows)?.into_iter().collect()]);
    }
    let groups = grouped_node_rows(layer, input, key_columns)?;
    let mut output = vec![Vec::with_capacity(groups.len()); key_columns.len() + 1];
    for (key_ids, rows) in groups.values() {
        let selected = choose(rows)?
            .ok_or_else(|| invalid("group-extrema canonical group is unexpectedly empty"))?;
        for (column, key_id) in output.iter_mut().zip(key_ids) {
            column.push(ArkFr::from_u64(*key_id));
        }
        output[key_columns.len()].push(selected);
    }
    Ok(output)
}
