//! Exact scalar-key `GroupBy` followed immediately by `RangeMin`/`RangeMax`.
//!
//! Each public output `(key, extremum-id)` defines an exact equality-compacted
//! subgroup of the authenticated child relation.  Distinct keys plus the sum of
//! subgroup cardinalities being the child cardinality prove an exact, disjoint
//! partition (including duplicate multiplicity).  A second equality proof proves
//! that the chosen id occurs in its subgroup.  Finally the subgroup value-id
//! column is dictionary-authenticated and compared against the chosen value; the
//! resulting range-selection commitment must be the whole real-row prefix.

use std::collections::BTreeSet;
use std::io;

use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::transcript::Transcript;

use super::argument::{commit_column, Transcript254};
use super::bgp::{RelationCommit, RelationWitness};
use super::commitment::{LayerCommitment, VerifierLayer};
use super::equality::{
    prove_relation_equality, verify_relation_equality, RelationEqualityConstraint,
    RelationEqualityProof,
};
use super::range::{
    prove_object_range_select, verify_object_range_select, ObjectRangeProof, RangeOp,
    OBJECT_RANGE_B_BITS,
};
use super::relocate::{prove_column_relocation, verify_column_relocation, RelocationProof};
use super::{invalid, CanonicalDatatype, CanonicalObject};

const DOMAIN: &[u8] = b"parture/proof-of-woql/group-extrema/v0";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupExtremaKind {
    Min,
    Max,
}

impl GroupExtremaKind {
    fn op(self) -> RangeOp {
        match self {
            Self::Min => RangeOp::Ge,
            Self::Max => RangeOp::Le,
        }
    }
    fn tag(self) -> u8 {
        match self { Self::Min => 0, Self::Max => 1 }
    }
}

#[derive(Clone, Debug)]
pub struct GroupExtremaEntryProof {
    pub group: RelationEqualityProof,
    pub group_commitments: Vec<ArkGT>,
    pub group_len: usize,
    pub membership: RelationEqualityProof,
    pub membership_commitments: Vec<ArkGT>,
    pub membership_len: usize,
    pub relocate_value: RelocationProof,
    pub relocate_dict_id: RelocationProof,
    pub relocate_dict_value: RelocationProof,
    pub relocate_dict_datatype: RelocationProof,
    pub com_value_at_d: ArkGT,
    pub com_dict_id_at_d: ArkGT,
    pub com_dict_value_at_d: ArkGT,
    pub com_dict_datatype_at_d: ArkGT,
    pub range: ObjectRangeProof,
    pub com_range_selection: ArkGT,
}

#[derive(Clone, Debug)]
pub struct GroupExtremaProof {
    pub entries: Vec<GroupExtremaEntryProof>,
}

pub struct GroupExtremaResult {
    pub proof: GroupExtremaProof,
    pub columns: Vec<Vec<ArkFr>>,
    pub commitments: Vec<ArkGT>,
    pub log_domain: usize,
    pub n: usize,
}

fn supported(datatype: CanonicalDatatype) -> bool {
    matches!(datatype, CanonicalDatatype::UInt8 | CanonicalDatatype::Int8
        | CanonicalDatatype::UInt16 | CanonicalDatatype::Int16
        | CanonicalDatatype::UInt32 | CanonicalDatatype::Int32 | CanonicalDatatype::Float32
        | CanonicalDatatype::UInt64 | CanonicalDatatype::Int64 | CanonicalDatatype::Float64)
}

fn padded(values: &[ArkFr], log: usize) -> Vec<ArkFr> {
    let mut out = values.to_vec();
    out.resize(1usize << log, ArkFr::zero());
    out
}

fn bind(
    transcript: &mut Transcript254,
    input: &RelationCommit,
    key_column: usize,
    value_column: usize,
    datatype: CanonicalDatatype,
    kind: GroupExtremaKind,
    rows: &[(u64, u64)],
) {
    transcript.append_bytes(b"pow/group/domain", DOMAIN);
    for v in [input.log_domain, input.n, input.commitments.len(), key_column, value_column] {
        transcript.append_bytes(b"pow/group/meta", &(v as u64).to_le_bytes());
    }
    transcript.append_bytes(b"pow/group/datatype", &[datatype as u8]);
    transcript.append_bytes(b"pow/group/kind", &[kind.tag()]);
    for commitment in &input.commitments { transcript.append_group(b"pow/group/input", commitment); }
    transcript.append_bytes(b"pow/group/count", &(rows.len() as u64).to_le_bytes());
    for (key, value) in rows {
        transcript.append_bytes(b"pow/group/key", &key.to_le_bytes());
        transcript.append_bytes(b"pow/group/value", &value.to_le_bytes());
    }
}

fn validate_statement(input: &RelationCommit, key: usize, value: usize, datatype: CanonicalDatatype, rows: &[(u64,u64)]) -> io::Result<()> {
    if input.log_domain == 0 || input.log_domain >= usize::BITS as usize || input.n > (1usize << input.log_domain) {
        return Err(invalid("group-extrema input domain out of range"));
    }
    if input.commitments.is_empty() || key >= input.commitments.len() || value >= input.commitments.len() || key == value {
        return Err(invalid("group-extrema column out of range or aliased"));
    }
    if !supported(datatype) { return Err(invalid("group-extrema datatype is not fixed-width order-preserving")); }
    if rows.len() > input.n { return Err(invalid("group-extrema has more groups than input rows")); }
    let mut keys = BTreeSet::new();
    for (k, v) in rows {
        if *k == 0 || *v == 0 { return Err(invalid("group-extrema Store ids must be nonzero")); }
        if !keys.insert(*k) { return Err(invalid("group-extrema result keys are not distinct")); }
    }
    Ok(())
}

fn extremum_encoding(source: &impl VerifierLayer, id: u64, datatype: CanonicalDatatype) -> io::Result<ArkFr> {
    let layer = source.verifier_view();
    let record = layer.node_value_catalog.iter().find(|r| r.object_id == id)
        .ok_or_else(|| invalid("group-extrema result id absent from node/value dictionary"))?;
    match &record.triple.object {
        CanonicalObject::Value { datatype: actual, .. } if *actual == datatype => {}
        _ => return Err(invalid("group-extrema result does not have the exact planned datatype")),
    }
    super::commitment::node_value_op_value(&record.triple.object)
}

#[allow(clippy::too_many_arguments)]
pub fn prove_group_extrema(
    input: &RelationWitness,
    key_column: usize,
    value_column: usize,
    datatype: CanonicalDatatype,
    kind: GroupExtremaKind,
    result_rows: &[(u64, u64)],
    layer: &LayerCommitment,
    transcript: &mut Transcript254,
) -> io::Result<GroupExtremaResult> {
    validate_statement(&input.view(), key_column, value_column, datatype, result_rows)?;
    super::commitment::reject_nan_catalog(
        &layer.node_value_catalog,
        ArkFr::from_u64(datatype as u64),
    )?;
    bind(transcript, &input.view(), key_column, value_column, datatype, kind, result_rows);
    if input.n == 0 {
        let log_domain = 1;
        let columns = vec![vec![ArkFr::zero(); 2], vec![ArkFr::zero(); 2]];
        let commitments = columns.iter().map(|c| commit_column(c)).collect::<io::Result<Vec<_>>>()?;
        return Ok(GroupExtremaResult { proof: GroupExtremaProof { entries: Vec::new() }, columns, commitments, log_domain, n: 0 });
    }
    let (dict_id, dict_value, dict_datatype) = {
        let (_, id) = layer.node_value_dictionary_columns()?;
        (id, layer.node_value_dictionary_value_column()?, layer.node_value_dictionary_datatype_column()?)
    };
    let n_dict = layer.state.node_value_count as usize;
    if n_dict != layer.node_value_catalog.len() || n_dict == 0 { return Err(invalid("group-extrema dictionary count mismatch or empty")); }
    let dict_log = dict_id.len().trailing_zeros() as usize;
    let mut entries = Vec::with_capacity(result_rows.len());
    for &(key, extremum) in result_rows {
        let group_constraint = RelationEqualityConstraint::ColumnConstant { column: key_column, value: ArkFr::from_u64(key) };
        let group = prove_relation_equality(input, &group_constraint, transcript)?;
        if group.n == 0 { return Err(invalid("group-extrema output names an empty group")); }
        let group_witness = RelationWitness { columns: group.columns.clone(), commitments: group.commitments.clone(), log_domain: group.log_domain, n: group.n };
        let member_constraint = RelationEqualityConstraint::ColumnConstant { column: value_column, value: ArkFr::from_u64(extremum) };
        let membership = prove_relation_equality(&group_witness, &member_constraint, transcript)?;
        if membership.n == 0 { return Err(invalid("group-extrema result is not a member of its group")); }

        let d_log = group.log_domain.max(dict_log);
        let value_at_d = padded(&group.columns[value_column], d_log);
        let dict_id_at_d = padded(&dict_id, d_log);
        let dict_value_at_d = padded(&dict_value, d_log);
        let dict_datatype_at_d = padded(&dict_datatype, d_log);
        let com_value_at_d = commit_column(&value_at_d)?;
        let com_dict_id_at_d = commit_column(&dict_id_at_d)?;
        let com_dict_value_at_d = commit_column(&dict_value_at_d)?;
        let com_dict_datatype_at_d = commit_column(&dict_datatype_at_d)?;
        let relocate_value = prove_column_relocation(&group.columns[value_column], &group.commitments[value_column], &value_at_d, &com_value_at_d, transcript)?;
        let relocate_dict_id = prove_column_relocation(&dict_id, &layer.node_value_dict_commitments.1, &dict_id_at_d, &com_dict_id_at_d, transcript)?;
        let relocate_dict_value = prove_column_relocation(&dict_value, &layer.node_value_dict_range_commitments.0, &dict_value_at_d, &com_dict_value_at_d, transcript)?;
        let relocate_dict_datatype = prove_column_relocation(&dict_datatype, &layer.node_value_dict_range_commitments.1, &dict_datatype_at_d, &com_dict_datatype_at_d, transcript)?;
        let c = extremum_encoding(layer, extremum, datatype)?;
        let (range, com_range_selection) = prove_object_range_select(
            &value_at_d, &com_value_at_d, &dict_id_at_d, &dict_value_at_d, &dict_datatype_at_d,
            &com_dict_id_at_d, &com_dict_value_at_d, &com_dict_datatype_at_d, n_dict,
            ArkFr::from_u64(datatype as u64), kind.op(), c, OBJECT_RANGE_B_BITS, group.n, transcript)?;
        let mut whole = vec![ArkFr::zero(); 1usize << d_log];
        whole[..group.n].fill(ArkFr::one());
        if com_range_selection != commit_column(&whole)? { return Err(invalid("group-extrema candidate does not dominate its group")); }
        entries.push(GroupExtremaEntryProof {
            group: group.proof, group_commitments: group.commitments, group_len: group.n,
            membership: membership.proof, membership_commitments: membership.commitments, membership_len: membership.n,
            relocate_value, relocate_dict_id, relocate_dict_value, relocate_dict_datatype,
            com_value_at_d, com_dict_id_at_d, com_dict_value_at_d, com_dict_datatype_at_d,
            range, com_range_selection,
        });
    }
    if entries.iter().try_fold(0usize, |n, e| n.checked_add(e.group_len)).ok_or_else(|| invalid("group-extrema cardinality overflow"))? != input.n {
        return Err(invalid("group-extrema groups do not exactly cover the input"));
    }
    let out_log = result_rows.len().max(1).next_power_of_two().max(2).trailing_zeros() as usize;
    let columns = vec![padded(&result_rows.iter().map(|r| ArkFr::from_u64(r.0)).collect::<Vec<_>>(), out_log), padded(&result_rows.iter().map(|r| ArkFr::from_u64(r.1)).collect::<Vec<_>>(), out_log)];
    let commitments = columns.iter().map(|c| commit_column(c)).collect::<io::Result<Vec<_>>>()?;
    Ok(GroupExtremaResult { proof: GroupExtremaProof { entries }, columns, commitments, log_domain: out_log, n: result_rows.len() })
}

#[allow(clippy::too_many_arguments)]
pub fn verify_group_extrema(
    input: &RelationCommit, key_column: usize, value_column: usize,
    datatype: CanonicalDatatype, kind: GroupExtremaKind, result_rows: &[(u64,u64)],
    layer: &impl VerifierLayer, proof: &GroupExtremaProof, transcript: &mut Transcript254,
) -> io::Result<()> {
    validate_statement(input, key_column, value_column, datatype, result_rows)?;
    if proof.entries.len() != result_rows.len() { return Err(invalid("group-extrema proof/result count mismatch")); }
    bind(transcript, input, key_column, value_column, datatype, kind, result_rows);
    let view = layer.verifier_view();
    view.validate()?;
    super::commitment::reject_nan_catalog(
        &view.node_value_catalog,
        ArkFr::from_u64(datatype as u64),
    )?;
    let n_dict = view.state.node_value_count as usize;
    if n_dict != view.node_value_catalog.len() || (!result_rows.is_empty() && n_dict == 0) { return Err(invalid("group-extrema dictionary count mismatch or empty")); }
    if result_rows.is_empty() { return Ok(()); }
    let dict_len = n_dict.next_power_of_two().max(2);
    let dict_log = dict_len.trailing_zeros() as usize;
    let mut total = 0usize;
    for (&(key, extremum), entry) in result_rows.iter().zip(&proof.entries) {
        if entry.group_len == 0 { return Err(invalid("group-extrema empty group")); }
        total = total.checked_add(entry.group_len).ok_or_else(|| invalid("group-extrema cardinality overflow"))?;
        let gc = RelationEqualityConstraint::ColumnConstant { column: key_column, value: ArkFr::from_u64(key) };
        verify_relation_equality(input, &gc, &entry.group_commitments, entry.group_len, input.log_domain, &entry.group, transcript)?;
        let group = RelationCommit { commitments: entry.group_commitments.clone(), log_domain: input.log_domain, n: entry.group_len };
        let mc = RelationEqualityConstraint::ColumnConstant { column: value_column, value: ArkFr::from_u64(extremum) };
        if entry.membership_len == 0 { return Err(invalid("group-extrema missing member")); }
        verify_relation_equality(&group, &mc, &entry.membership_commitments, entry.membership_len, input.log_domain, &entry.membership, transcript)?;
        let d_log = input.log_domain.max(dict_log);
        verify_column_relocation(&entry.group_commitments[value_column], input.log_domain, &entry.com_value_at_d, d_log, &entry.relocate_value, transcript)?;
        verify_column_relocation(&view.node_value_dict_commitments.1, dict_log, &entry.com_dict_id_at_d, d_log, &entry.relocate_dict_id, transcript)?;
        verify_column_relocation(&view.node_value_dict_range_commitments.0, dict_log, &entry.com_dict_value_at_d, d_log, &entry.relocate_dict_value, transcript)?;
        verify_column_relocation(&view.node_value_dict_range_commitments.1, dict_log, &entry.com_dict_datatype_at_d, d_log, &entry.relocate_dict_datatype, transcript)?;
        let c = extremum_encoding(layer, extremum, datatype)?;
        verify_object_range_select(&entry.com_value_at_d, &entry.com_dict_id_at_d, &entry.com_dict_value_at_d, &entry.com_dict_datatype_at_d,
            n_dict, ArkFr::from_u64(datatype as u64), kind.op(), c, OBJECT_RANGE_B_BITS, entry.group_len,
            &entry.com_range_selection, &entry.range, transcript)?;
        let mut whole = vec![ArkFr::zero(); 1usize << d_log]; whole[..entry.group_len].fill(ArkFr::one());
        if entry.com_range_selection != commit_column(&whole)? { return Err(invalid("group-extrema candidate does not dominate its group")); }
    }
    if total != input.n { return Err(invalid("group-extrema groups do not exactly cover input")); }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::TryInto;
    use crate::layer::ValueTriple;
    use crate::store::sync::open_sync_memory_store;
    use tdb_succinct::TdbDataType;

    fn fixture() -> (std::sync::Arc<LayerCommitment>, Vec<(u64, u64)>, Vec<u64>) {
        let iri = |s: &str| format!("http://group.test/{s}");
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for (s, value) in [("a", 30u32), ("b", 10), ("c", 5), ("d", 20)] {
            builder.add_value_triple(ValueTriple::new_value(&iri(s), &iri("age"), u32::make_entry(&value))).unwrap();
        }
        let layer = builder.commit().unwrap().proof_commitment().unwrap();
        let mut values = Vec::new();
        let mut nodes = Vec::new();
        for record in &layer.node_value_catalog {
            match &record.triple.object {
                CanonicalObject::Value { datatype: CanonicalDatatype::UInt32, lexical } => {
                    values.push((u32::from_be_bytes(lexical.as_slice().try_into().unwrap()) as u64, record.object_id));
                }
                CanonicalObject::Node(_) => nodes.push(record.object_id),
                _ => {}
            }
        }
        values.sort_unstable(); nodes.sort_unstable();
        (layer, values, nodes)
    }

    #[test]
    fn scalar_group_min_proves_partition_membership_and_domination() {
        let (layer, values, nodes) = fixture();
        let id = |v| values.iter().find(|x| x.0 == v).unwrap().1;
        let keys = [nodes[0], nodes[0], nodes[1], nodes[1]];
        let vals = [id(30), id(10), id(5), id(20)];
        let input = RelationWitness::commit(vec![
            keys.iter().map(|x| ArkFr::from_u64(*x)).collect(),
            vals.iter().map(|x| ArkFr::from_u64(*x)).collect(),
        ], 4).unwrap();
        let rows = [(nodes[0], id(10)), (nodes[1], id(5))];
        let mut pt = Transcript254::new(b"group-min-test");
        let result = prove_group_extrema(&input, 0, 1, CanonicalDatatype::UInt32, GroupExtremaKind::Min, &rows, &layer, &mut pt).unwrap();
        assert_eq!(result.n, 2);
        let mut vt = Transcript254::new(b"group-min-test");
        verify_group_extrema(&input.view(), 0, 1, CanonicalDatatype::UInt32, GroupExtremaKind::Min, &rows, &layer, &result.proof, &mut vt).unwrap();
    }

    #[test]
    fn duplicate_public_keys_and_unsupported_datatype_fail_closed() {
        let input = RelationWitness::commit(vec![vec![ArkFr::from_u64(1)], vec![ArkFr::from_u64(2)]], 1).unwrap();
        assert!(validate_statement(&input.view(), 0, 1, CanonicalDatatype::UInt32, &[(1,2),(1,2)]).is_err());
        assert!(validate_statement(&input.view(), 0, 1, CanonicalDatatype::Decimal, &[(1,2)]).is_err());
    }
}
