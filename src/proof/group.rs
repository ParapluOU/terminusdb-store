//! Exact scalar-key `GroupBy` followed immediately by `RangeMin`/`RangeMax`.
//!
//! Each public output `(key, extremum-id)` defines an exact equality-compacted
//! subgroup of the authenticated child relation.  Distinct keys plus the sum of
//! subgroup cardinalities being the child cardinality prove an exact, disjoint
//! partition (including duplicate multiplicity).  A second equality proof proves
//! that the chosen id occurs in its subgroup.  Finally the subgroup value-id
//! column is dictionary-authenticated and compared against the chosen value; the
//! resulting range-selection commitment must be the whole real-row prefix.
//!
//! For a developer: this is the group-by (with per-group min/max) [gadget](crate::proof::primer).
//! It adds no new cryptography — it COMPOSES the audited [`equality`](super::equality) gadget
//! (subgroup partition + membership), the [`relocate`](super::relocate) domain tie, and the
//! [`range`](super::range) selection (which the extremum must dominate). It also reads dictionary
//! values, so it is anchored to the layer commitment / dictionary model (primer §9) and it
//! deliberately rejects float datatypes on a fidelity ground (IEEE total order ≠ WOQL numeric
//! order; primer §11 on soundness vs fidelity). See also [gadgets](crate::proof::primer) (§6).

use std::collections::{BTreeMap, BTreeSet};
use std::io;

use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::transcript::Transcript;

use super::argument::{commit_column, StoreProverContext, Transcript254};
use super::bgp::{RelationCommit, RelationWitness};
use super::commitment::{
    node_value_op_value, reject_nan_objects, LayerCommitment, ResolutionOpenings, VerifierLayer,
};
use super::query::{
    build_resolution_openings, canonical_dictionary_id, require_base_layer_for_query,
    resolve_node_value_id_via_merkle, verify_dict_value_terms,
};
use super::equality::{
    prove_relation_equality_with_context, verify_relation_equality, RelationEqualityConstraint,
    RelationEqualityProof,
};
use super::range::{
    prove_object_range_select_with_context, verify_object_range_select, ObjectRangeProof, RangeOp,
    OBJECT_RANGE_B_BITS,
};
use super::relocate::{
    prove_column_relocation_with_context, verify_column_relocation, RelocationProof,
};
use super::{invalid, CanonicalDatatype, CanonicalObject};

const DOMAIN: &[u8] = b"parture/proof-of-woql/group-extrema/v0";

/// Which per-group extremum to prove: the group's minimum value (`Min`) or maximum
/// (`Max`). Selects the range operator used to prove the chosen id dominates its group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupExtremaKind {
    /// Prove the group's minimum value (the chosen id is `<=` every group value, via `RangeOp::Ge`).
    Min,
    /// Prove the group's maximum value (the chosen id is `>=` every group value, via `RangeOp::Le`).
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

/// Per output `(key, extremum-id)` row: the subproofs that (1) carve the key's exact
/// subgroup, (2) prove the chosen id is a member of it, (3) relocate the value + dictionary
/// columns to a common domain, and (4) prove via a range selection that the chosen value
/// dominates (min/max) the whole subgroup.
#[derive(Clone, Debug)]
pub struct GroupExtremaEntryProof {
    /// Equality proof carving the exact subgroup where the key column equals this row's key.
    pub group: RelationEqualityProof,
    /// Commitments to the carved subgroup's columns.
    pub group_commitments: Vec<ArkGT>,
    /// The subgroup's real row count.
    pub group_len: usize,
    /// Equality proof that the chosen extremum id occurs in the subgroup's value column.
    pub membership: RelationEqualityProof,
    /// Commitments to the membership-filtered columns.
    pub membership_commitments: Vec<ArkGT>,
    /// The membership-filtered row count (must be nonzero).
    pub membership_len: usize,
    /// Relocate the subgroup value column to the common domain `2^d`.
    pub relocate_value: RelocationProof,
    /// Relocate the dictionary id column to `2^d`.
    pub relocate_dict_id: RelocationProof,
    /// Relocate the dictionary order-preserving value column to `2^d`.
    pub relocate_dict_value: RelocationProof,
    /// Relocate the dictionary datatype column to `2^d`.
    pub relocate_dict_datatype: RelocationProof,
    /// Commitment to the subgroup value column at domain `2^d`.
    pub com_value_at_d: ArkGT,
    /// Commitment to the dictionary id column at domain `2^d`.
    pub com_dict_id_at_d: ArkGT,
    /// Commitment to the dictionary value column at domain `2^d`.
    pub com_dict_value_at_d: ArkGT,
    /// Commitment to the dictionary datatype column at domain `2^d`.
    pub com_dict_datatype_at_d: ArkGT,
    /// The range selection proving the chosen value dominates (min/max) the whole subgroup.
    pub range: ObjectRangeProof,
    /// Commitment to the range-selection column; must equal the whole real-row prefix of the subgroup.
    pub com_range_selection: ArkGT,
    /// Merkle-dict I2b step 3 (value↔term consistency, TERM-LESS group). The real object ids of
    /// this group's VALUE column (`oid[i]` = the id at compacted row `i`, length = `group_len`,
    /// in input/compaction order), which `com_value_at_d` commits and `range.lookup` gathers
    /// `(v_val, v_dt)` from. EMPTY on a TERM-BEARING artifact. On a TERM-LESS artifact the verifier
    /// recommits `oid` to `com_value_at_d` and rederives `range.lookup.com_v_val`/`com_v_dt` from
    /// the Merkle-authenticated opened terms (`verify_dict_value_terms`) — the SAME columns the
    /// Min/Max range selection consumes. UNCOMMITTED; a missing opening / recommit mismatch fails
    /// closed.
    pub oid: Vec<u64>,
}

/// A group-extrema proof: one entry per public output group row.
#[derive(Clone, Debug)]
pub struct GroupExtremaProof {
    pub entries: Vec<GroupExtremaEntryProof>,
    /// Merkle-dict I2b step 3: one positional Merkle opening per DISTINCT id touched across ALL
    /// groups' value columns AND the extremum ids (node/value namespace). `None` on a TERM-BEARING
    /// artifact (the catalog re-MSM ties value/datatype to real terms). UNCOMMITTED.
    pub value_openings: Option<ResolutionOpenings>,
}

/// Prover result: the proof plus the output relation `(key, extremum-id)` — its columns,
/// their commitments, the output `log_domain`, and the group count `n`.
pub struct GroupExtremaResult {
    /// The group-extrema proof.
    pub proof: GroupExtremaProof,
    /// The output `(key, extremum-id)` columns, zero-padded to `2^log_domain`.
    pub columns: Vec<Vec<ArkFr>>,
    /// Dory commitments to `columns`.
    pub commitments: Vec<ArkGT>,
    /// The output columns' power-of-two log-domain.
    pub log_domain: usize,
    /// The number of output groups (real rows).
    pub n: usize,
}

fn supported(datatype: CanonicalDatatype) -> bool {
    // Integral only — mirrors the LIVE extrema guard `integral_op_datatype`. Float32/64 are
    // EXCLUDED: their committed OP value is IEEE TOTAL order (−0.0 sorts below +0.0; NaN above +∞),
    // which diverges from WOQL numeric Min/Max, so an unfaithful float extremum must never become
    // provable — even if this (currently caller-less) standalone GroupExtrema path is ever wired in.
    matches!(datatype, CanonicalDatatype::UInt8 | CanonicalDatatype::Int8
        | CanonicalDatatype::UInt16 | CanonicalDatatype::Int16
        | CanonicalDatatype::UInt32 | CanonicalDatatype::Int32
        | CanonicalDatatype::UInt64 | CanonicalDatatype::Int64)
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

fn extremum_encoding(
    source: &impl VerifierLayer,
    id: u64,
    datatype: CanonicalDatatype,
    openings: Option<&ResolutionOpenings>,
) -> io::Result<ArkFr> {
    let layer = source.verifier_view();
    // Merkle-dict I2b step 3: encode the chosen extremum id's order-preserving dict VALUE (the
    // range-select constant `c`). TERM-BEARING reads the root-bound catalog; TERM-LESS opens the id
    // against the `resolution_root`-bound Merkle tree (the extremum is a member of its group's value
    // column, so its opening is present in `value_openings`). Either way the datatype must match.
    let object = if layer.terms_present {
        layer
            .node_value_catalog
            .iter()
            .find(|r| r.object_id == id)
            .ok_or_else(|| invalid("group-extrema result id absent from node/value dictionary"))?
            .triple
            .object
            .clone()
    } else {
        let openings = openings
            .ok_or_else(|| invalid("term-less group extremum requires the value↔term Merkle openings"))?;
        resolve_node_value_id_via_merkle(&ArkFr::from_u64(id), layer, None, Some(openings))?
    };
    match &object {
        CanonicalObject::Value { datatype: actual, .. } if *actual == datatype => {}
        _ => return Err(invalid("group-extrema result does not have the exact planned datatype")),
    }
    node_value_op_value(&object)
}

/// Prove a scalar-key `GroupBy(key_column)` followed by `RangeMin`/`RangeMax(value_column)`
/// (`kind`). The public `result_rows` are `(group key id, extremum value id)` pairs; the
/// proof certifies they partition `input` exactly by key and that each chosen id is the
/// true min/max of its group under the root-bound `layer` dictionary values.
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
    prove_group_extrema_with_context(
        &StoreProverContext::cpu(), input, key_column, value_column, datatype, kind, result_rows,
        layer, transcript,
    )
}

/// As [`prove_group_extrema`], with an explicit prover-arithmetic context.
#[allow(clippy::too_many_arguments)]
pub fn prove_group_extrema_with_context(
    context: &StoreProverContext<'_>,
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
        let commitments = columns.iter().map(|c| context.commit_column(c)).collect::<io::Result<Vec<_>>>()?;
        return Ok(GroupExtremaResult { proof: GroupExtremaProof { entries: Vec::new(), value_openings: None }, columns, commitments, log_domain, n: 0 });
    }
    let (dict_id, dict_value, dict_datatype) = {
        let (_, id) = layer.node_value_dictionary_columns()?;
        (id, layer.node_value_dictionary_value_column()?, layer.node_value_dictionary_datatype_column()?)
    };
    let n_dict = layer.state.node_value_count as usize;
    if n_dict != layer.node_value_catalog.len() || n_dict == 0 { return Err(invalid("group-extrema dictionary count mismatch or empty")); }
    let dict_log = dict_id.len().trailing_zeros() as usize;
    // Equality compaction preserves the input domain, so every group relocates the
    // same dictionary columns to the same domain.  Commit those immutable columns
    // once instead of repeating three full MSMs per group.
    let d_log = input.log_domain.max(dict_log);
    let dict_id_at_d = padded(&dict_id, d_log);
    let dict_value_at_d = padded(&dict_value, d_log);
    let dict_datatype_at_d = padded(&dict_datatype, d_log);
    let com_dict_id_at_d = context.commit_column(&dict_id_at_d)?;
    let com_dict_value_at_d = context.commit_column(&dict_value_at_d)?;
    let com_dict_datatype_at_d = context.commit_column(&dict_datatype_at_d)?;
    let mut whole_commitments = BTreeMap::new();
    let mut entries = Vec::with_capacity(result_rows.len());
    for &(key, extremum) in result_rows {
        let group_constraint = RelationEqualityConstraint::ColumnConstant { column: key_column, value: ArkFr::from_u64(key) };
        let group = prove_relation_equality_with_context(context, input, &group_constraint, transcript)?;
        if group.n == 0 { return Err(invalid("group-extrema output names an empty group")); }
        let group_witness = RelationWitness { columns: group.columns.clone(), commitments: group.commitments.clone(), log_domain: group.log_domain, n: group.n };
        let member_constraint = RelationEqualityConstraint::ColumnConstant { column: value_column, value: ArkFr::from_u64(extremum) };
        let membership = prove_relation_equality_with_context(context, &group_witness, &member_constraint, transcript)?;
        if membership.n == 0 { return Err(invalid("group-extrema result is not a member of its group")); }

        if group.log_domain != input.log_domain {
            return Err(invalid("group-extrema equality changed the input domain"));
        }
        let value_at_d = padded(&group.columns[value_column], d_log);
        let com_value_at_d = context.commit_column(&value_at_d)?;
        let relocate_value = prove_column_relocation_with_context(context, &group.columns[value_column], &group.commitments[value_column], &value_at_d, &com_value_at_d, transcript)?;
        let relocate_dict_id = prove_column_relocation_with_context(context, &dict_id, &layer.node_value_dict_commitments.1, &dict_id_at_d, &com_dict_id_at_d, transcript)?;
        let relocate_dict_value = prove_column_relocation_with_context(context, &dict_value, &layer.node_value_dict_range_commitments.0, &dict_value_at_d, &com_dict_value_at_d, transcript)?;
        let relocate_dict_datatype = prove_column_relocation_with_context(context, &dict_datatype, &layer.node_value_dict_range_commitments.1, &dict_datatype_at_d, &com_dict_datatype_at_d, transcript)?;
        let c = extremum_encoding(layer, extremum, datatype, None)?;
        let (range, com_range_selection) = prove_object_range_select_with_context(
            context, &value_at_d, &com_value_at_d, &dict_id_at_d, &dict_value_at_d, &dict_datatype_at_d,
            &com_dict_id_at_d, &com_dict_value_at_d, &com_dict_datatype_at_d, n_dict,
            ArkFr::from_u64(datatype as u64), kind.op(), c, OBJECT_RANGE_B_BITS, group.n, transcript)?;
        let expected_whole = if let Some(commitment) = whole_commitments.get(&group.n) {
            *commitment
        } else {
            let mut whole = vec![ArkFr::zero(); 1usize << d_log];
            whole[..group.n].fill(ArkFr::one());
            let commitment = context.commit_column(&whole)?;
            whole_commitments.insert(group.n, commitment);
            commitment
        };
        if com_range_selection != expected_whole { return Err(invalid("group-extrema candidate does not dominate its group")); }
        entries.push(GroupExtremaEntryProof {
            group: group.proof, group_commitments: group.commitments, group_len: group.n,
            membership: membership.proof, membership_commitments: membership.commitments, membership_len: membership.n,
            relocate_value, relocate_dict_id, relocate_dict_value, relocate_dict_datatype,
            com_value_at_d, com_dict_id_at_d, com_dict_value_at_d, com_dict_datatype_at_d,
            range, com_range_selection,
            // Merkle-dict I2b step 3: term-bearing by default; `build_termless_group_openings`
            // fills the per-group value oid vector for a TERM-LESS artifact.
            oid: Vec::new(),
        });
    }
    if entries.iter().try_fold(0usize, |n, e| n.checked_add(e.group_len)).ok_or_else(|| invalid("group-extrema cardinality overflow"))? != input.n {
        return Err(invalid("group-extrema groups do not exactly cover the input"));
    }
    let out_log = result_rows.len().max(1).next_power_of_two().max(2).trailing_zeros() as usize;
    let columns = vec![padded(&result_rows.iter().map(|r| ArkFr::from_u64(r.0)).collect::<Vec<_>>(), out_log), padded(&result_rows.iter().map(|r| ArkFr::from_u64(r.1)).collect::<Vec<_>>(), out_log)];
    let commitments = columns.iter().map(|c| context.commit_column(c)).collect::<io::Result<Vec<_>>>()?;
    Ok(GroupExtremaResult { proof: GroupExtremaProof { entries, value_openings: None }, columns, commitments, log_domain: out_log, n: result_rows.len() })
}

/// Verify a group-extrema proof: the public `result_rows` are exactly one `(key, min/max)`
/// per group, the groups partition `input`, and each extremum dominates its group under the
/// `layer` dictionary values.
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
    // Merkle-dict I2b step 3 (value↔term consistency). GroupBy + Min/Max ranges over each group's
    // order-preserving dict VALUE column. TERM-BEARING: `validate()`'s catalog re-MSM ties
    // value/datatype to the clear terms, the NaN guard scans the catalog, and the shipped count is
    // cross-checked. TERM-LESS: the re-MSM is dropped ⇒ require the openings (fail-closed) and,
    // per entry, tie the gathered `(v_val, v_dt)` to the Merkle terms (below). Reject a delta layer
    // (base/rollup only). `n_dict` is root-bound (`state.node_value_count`) either way.
    let value_openings: Option<&ResolutionOpenings> = if view.terms_present {
        super::commitment::reject_nan_catalog(
            &view.node_value_catalog,
            ArkFr::from_u64(datatype as u64),
        )?;
        None
    } else {
        require_base_layer_for_query(view.state)?;
        Some(proof.value_openings.as_ref().ok_or_else(|| {
            invalid("term-less group extrema requires the prover-shipped value↔term Merkle openings")
        })?)
    };
    let n_dict = view.state.node_value_count as usize;
    if (view.terms_present && n_dict != view.node_value_catalog.len()) || (!result_rows.is_empty() && n_dict == 0) { return Err(invalid("group-extrema dictionary count mismatch or empty")); }
    if result_rows.is_empty() { return Ok(()); }
    let dict_len = n_dict.next_power_of_two().max(2);
    let dict_log = dict_len.trailing_zeros() as usize;
    let d_log = input.log_domain.max(dict_log);
    let mut whole_commitments = BTreeMap::new();
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
        verify_column_relocation(&entry.group_commitments[value_column], input.log_domain, &entry.com_value_at_d, d_log, &entry.relocate_value, transcript)?;
        verify_column_relocation(&view.node_value_dict_commitments.1, dict_log, &entry.com_dict_id_at_d, d_log, &entry.relocate_dict_id, transcript)?;
        verify_column_relocation(&view.node_value_dict_range_commitments.0, dict_log, &entry.com_dict_value_at_d, d_log, &entry.relocate_dict_value, transcript)?;
        verify_column_relocation(&view.node_value_dict_range_commitments.1, dict_log, &entry.com_dict_datatype_at_d, d_log, &entry.relocate_dict_datatype, transcript)?;
        let c = extremum_encoding(layer, extremum, datatype, value_openings)?;
        // Merkle-dict I2b step 3 (TERM-LESS value↔term tie). The range-select below gathers
        // `range.lookup.com_v_val`/`com_v_dt` at each group VALUE id (from `com_value_at_d`); on a
        // term-less artifact those committed dict columns are prover-chosen. Recommit the shipped
        // `oid` to `com_value_at_d` (binding the group value column) and rederive the value/datatype
        // from the Merkle terms, forcing the Min/Max comparison onto real values. Runs BEFORE the
        // (transcript-consuming) range-select; it is itself transcript-free.
        if let Some(openings) = value_openings {
            let touched = verify_dict_value_terms(
                &entry.com_value_at_d,
                &entry.range.lookup.com_v_val,
                &entry.range.lookup.com_v_dt,
                entry.group_len,
                1usize << d_log,
                &entry.oid,
                openings,
                view,
                view.resolution_root(),
            )?;
            reject_nan_objects(&touched, ArkFr::from_u64(datatype as u64))?;
        }
        verify_object_range_select(&entry.com_value_at_d, &entry.com_dict_id_at_d, &entry.com_dict_value_at_d, &entry.com_dict_datatype_at_d,
            n_dict, ArkFr::from_u64(datatype as u64), kind.op(), c, OBJECT_RANGE_B_BITS, entry.group_len,
            &entry.com_range_selection, &entry.range, transcript)?;
        let expected_whole = if let Some(commitment) = whole_commitments.get(&entry.group_len) {
            *commitment
        } else {
            let mut whole = vec![ArkFr::zero(); 1usize << d_log];
            whole[..entry.group_len].fill(ArkFr::one());
            let commitment = commit_column(&whole)?;
            whole_commitments.insert(entry.group_len, commitment);
            commitment
        };
        if entry.com_range_selection != expected_whole { return Err(invalid("group-extrema candidate does not dominate its group")); }
    }
    if total != input.n { return Err(invalid("group-extrema groups do not exactly cover input")); }
    Ok(())
}

/// Merkle-dict I2b step 3 (producer): mint the TERM-LESS value↔term witness for a proven
/// base/rollup group-extrema proof. Re-derives each group's VALUE column ids by filtering the
/// (prover-held) `input_columns` where `key_column == key` in input/compaction order — the SAME
/// dense compaction `com_value_at_d` commits — records them as `entry.oid`, and opens each DISTINCT
/// touched id (group values + extremum ids) against the node/value tree into `proof.value_openings`.
/// UNCOMMITTED witness; BASE/ROLLUP single-layer ONLY (delta group stays fail-closed).
pub fn build_termless_group_openings(
    layer: &LayerCommitment,
    proof: &mut GroupExtremaProof,
    input_columns: &[Vec<ArkFr>],
    input_n: usize,
    key_column: usize,
    value_column: usize,
    result_rows: &[(u64, u64)],
) -> io::Result<()> {
    require_base_layer_for_query(layer.verifier_view().state)?;
    if proof.entries.len() != result_rows.len() {
        return Err(invalid("group openings: proof/result count mismatch"));
    }
    if key_column >= input_columns.len() || value_column >= input_columns.len() {
        return Err(invalid("group openings: column out of range"));
    }
    if input_columns[key_column].len() < input_n || input_columns[value_column].len() < input_n {
        return Err(invalid("group openings: input column too short"));
    }
    let mut nv_ids: Vec<u64> = Vec::new();
    for (entry, &(key, extremum)) in proof.entries.iter_mut().zip(result_rows) {
        let mut oid: Vec<u64> = Vec::new();
        for i in 0..input_n {
            if canonical_dictionary_id(&input_columns[key_column][i])? == key {
                let raw = canonical_dictionary_id(&input_columns[value_column][i])?;
                oid.push(raw);
                if raw != 0 {
                    nv_ids.push(raw);
                }
            }
        }
        entry.oid = oid;
        // The extremum id (range-select constant `c`) is a member of the group's value column, so
        // it is already gathered above; push it explicitly as belt-and-suspenders (dedup is free).
        if extremum != 0 {
            nv_ids.push(extremum);
        }
    }
    let openings = build_resolution_openings(&[layer.verifier_view()], &nv_ids, &[])?;
    proof.value_openings = Some(openings);
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

    /// Merkle-dict I2b S2 (N1): a valid group-extrema proof rejects on a TERM-LESS view of the
    /// same base layer — GroupBy Min/Max ranges over the dictionary VALUE column, untied to real
    /// terms without the catalog re-MSM. Term-bearing verify still succeeds.
    #[test]
    fn group_extrema_verify_fails_closed_on_termless_view() {
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
        let mut vt = Transcript254::new(b"group-min-test");
        verify_group_extrema(&input.view(), 0, 1, CanonicalDatatype::UInt32, GroupExtremaKind::Min, &rows, &layer, &result.proof, &mut vt).unwrap();

        let termless = layer.verifier_commitment_termless().unwrap();
        let mut tv = Transcript254::new(b"group-min-test");
        let err = verify_group_extrema(&input.view(), 0, 1, CanonicalDatatype::UInt32, GroupExtremaKind::Min, &rows, &termless, &result.proof, &mut tv).unwrap_err();
        // Merkle-dict I2b step 3: a term-less group-extrema proof WITHOUT the value openings still
        // fails closed, now with the step-3 "value↔term Merkle openings" message.
        assert!(err.to_string().contains("Merkle openings"), "{err}");
    }

    #[test]
    fn duplicate_public_keys_and_unsupported_datatype_fail_closed() {
        let input = RelationWitness::commit(vec![vec![ArkFr::from_u64(1)], vec![ArkFr::from_u64(2)]], 1).unwrap();
        assert!(validate_statement(&input.view(), 0, 1, CanonicalDatatype::UInt32, &[(1,2),(1,2)]).is_err());
        assert!(validate_statement(&input.view(), 0, 1, CanonicalDatatype::Decimal, &[(1,2)]).is_err());
    }

    // =====================================================================================
    // Merkle-dict I2b step 3 (value↔term consistency, GROUP). A TERM-LESS base/rollup group-extrema
    // proof ties each group's VALUE column (and the extremum constant) to the Merkle-authenticated
    // terms via the prover-shipped per-group `oid` + value openings. Term-bearing verify unchanged.
    // =====================================================================================
    fn prove_attach_group() -> (std::sync::Arc<LayerCommitment>, RelationWitness, Vec<(u64,u64)>, GroupExtremaResult) {
        let (layer, values, nodes) = fixture();
        let id = |v| values.iter().find(|x| x.0 == v).unwrap().1;
        let keys = [nodes[0], nodes[0], nodes[1], nodes[1]];
        let vals = [id(30), id(10), id(5), id(20)];
        let input = RelationWitness::commit(vec![
            keys.iter().map(|x| ArkFr::from_u64(*x)).collect(),
            vals.iter().map(|x| ArkFr::from_u64(*x)).collect(),
        ], 4).unwrap();
        let rows = vec![(nodes[0], id(10)), (nodes[1], id(5))];
        let mut result = prove_group_extrema(&input, 0, 1, CanonicalDatatype::UInt32, GroupExtremaKind::Min, &rows, &layer, &mut Transcript254::new(b"g-step3")).unwrap();
        build_termless_group_openings(&layer, &mut result.proof, &input.columns, input.n, 0, 1, &rows).unwrap();
        assert!(result.proof.value_openings.is_some(), "group value openings attached");
        assert!(result.proof.entries.iter().all(|e| !e.oid.is_empty()), "group oid vectors minted");
        (layer, input, rows, result)
    }

    /// Differential positive: a TERM-LESS group-extrema proof verifies EXACTLY like the
    /// TERM-BEARING one — same accept (honest Min), same reject (verifying as Max).
    #[test]
    fn i2b_step3_group_termless_differential() {
        let (layer, input, rows, result) = prove_attach_group();
        let termless = layer.verifier_commitment_termless().unwrap();
        verify_group_extrema(&input.view(), 0, 1, CanonicalDatatype::UInt32, GroupExtremaKind::Min, &rows, &*layer, &result.proof, &mut Transcript254::new(b"g-step3")).unwrap();
        verify_group_extrema(&input.view(), 0, 1, CanonicalDatatype::UInt32, GroupExtremaKind::Min, &rows, &termless, &result.proof, &mut Transcript254::new(b"g-step3")).unwrap();
        // Same reject: verifying a Min proof as Max rejects on BOTH artifacts.
        assert!(verify_group_extrema(&input.view(), 0, 1, CanonicalDatatype::UInt32, GroupExtremaKind::Max, &rows, &*layer, &result.proof, &mut Transcript254::new(b"g-step3")).is_err());
        assert!(verify_group_extrema(&input.view(), 0, 1, CanonicalDatatype::UInt32, GroupExtremaKind::Max, &rows, &termless, &result.proof, &mut Transcript254::new(b"g-step3")).is_err());
    }

    /// Forged committed VALUE column (`entry.range.lookup.com_v_val`) ⇒ Merkle-rederived value
    /// disagrees ⇒ REJECT (term-less).
    #[test]
    fn i2b_step3_group_termless_rejects_forged_value_column() {
        let (layer, input, rows, mut result) = prove_attach_group();
        result.proof.entries[0].range.lookup.com_v_val = result.proof.entries[0].range.lookup.com_v_dt;
        let termless = layer.verifier_commitment_termless().unwrap();
        let err = verify_group_extrema(&input.view(), 0, 1, CanonicalDatatype::UInt32, GroupExtremaKind::Min, &rows, &termless, &result.proof, &mut Transcript254::new(b"g-step3")).unwrap_err();
        assert!(err.to_string().contains("com_v_val"), "{err}");
    }

    /// A substituted per-group `oid` no longer recommits to `com_value_at_d` ⇒ REJECT (only the tie
    /// checks `oid`).
    #[test]
    fn i2b_step3_group_termless_rejects_forged_oid() {
        let (layer, input, rows, mut result) = prove_attach_group();
        result.proof.entries[0].oid[0] = result.proof.entries[0].oid[0].wrapping_add(1);
        let termless = layer.verifier_commitment_termless().unwrap();
        let err = verify_group_extrema(&input.view(), 0, 1, CanonicalDatatype::UInt32, GroupExtremaKind::Min, &rows, &termless, &result.proof, &mut Transcript254::new(b"g-step3")).unwrap_err();
        assert!(err.to_string().contains("oid column"), "{err}");
    }

    /// A term-less group proof WITHOUT the value openings fails closed; term-bearing still verifies.
    #[test]
    fn i2b_step3_group_termless_rejects_missing_openings() {
        let (layer, input, rows, mut result) = prove_attach_group();
        result.proof.value_openings = None;
        let termless = layer.verifier_commitment_termless().unwrap();
        let err = verify_group_extrema(&input.view(), 0, 1, CanonicalDatatype::UInt32, GroupExtremaKind::Min, &rows, &termless, &result.proof, &mut Transcript254::new(b"g-step3")).unwrap_err();
        assert!(err.to_string().contains("Merkle openings"), "{err}");
        verify_group_extrema(&input.view(), 0, 1, CanonicalDatatype::UInt32, GroupExtremaKind::Min, &rows, &*layer, &result.proof, &mut Transcript254::new(b"g-step3")).unwrap();
    }

    /// DELTA fail-closed (order + group + multi-order share this root guard). E1: a term-less
    /// artifact is now CONSTRUCTIBLE for a delta layer, but term-less order/group verification over a
    /// delta is STILL unreachable — the `require_base_layer_for_query` belt in every term-less verify
    /// branch (and this direct assertion) fails closed on any delta view.
    #[test]
    fn i2b_step3_delta_termless_artifact_constructible_but_query_fails_closed() {
        let iri = |s: &str| format!("http://group.test/{s}");
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        builder.add_value_triple(ValueTriple::new_value(&iri("a"), &iri("age"), u32::make_entry(&10u32))).unwrap();
        let base = builder.commit().unwrap();
        let child = base.open_write().unwrap();
        child.add_value_triple(ValueTriple::new_value(&iri("b"), &iri("age"), u32::make_entry(&20u32))).unwrap();
        let delta = child.commit().unwrap().proof_commitment().unwrap();
        assert!(delta.state.parent_state_root.is_some(), "constructed a delta layer");
        // E1: the term-less delta artifact now BUILDS + validates (previously rejected).
        let termless = delta.verifier_commitment_termless().unwrap();
        assert!(!termless.terms_present);
        termless.validate().unwrap();
        // ... but the single-layer query belt still fails closed on the delta view.
        let err = require_base_layer_for_query(termless.verifier_view().state).unwrap_err();
        assert!(err.to_string().contains("not directly query-provable"), "{err}");
    }
}
