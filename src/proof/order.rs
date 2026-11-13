//! Authenticated order for a one-column exact numeric relation.
//!
//! This deliberately narrow gadget is the first sound `OrderBy` slice: the sole output
//! column is also the complete sort key, every value is a nonnullable Store ID resolving
//! to one fixed-width order-preserving datatype, the output is an exact full-row
//! permutation, and the authenticated dictionary value column is strictly ascending.
//!
//! For a developer: this [gadget](crate::proof::primer) proves an `ORDER BY` answer is
//! the input's rows in the right order — nothing dropped, duplicated, or invented, and
//! genuinely sorted. Two audited pieces do the work and neither alone suffices: a
//! [permutation](crate::proof::primer) proof ([`super::permutation`]) forces the output
//! to be a row-multiset copy of the input, and a nondecreasing [sort](crate::proof::primer)
//! proof forces it to be ordered. Because ids are not themselves value-ordered, the sort
//! key is the id's dictionary VALUE, authenticated against the root-bound dictionary by a
//! [LogUp](crate::proof::primer) lookup (primer §7) before the sort runs. The datatype
//! restriction and float exclusion are [fidelity](crate::proof::primer) guards (primer
//! §11), not soundness ones. See [the primer](crate::proof::primer) for
//! [commitments](crate::proof::primer) (§3) and [prove → verify](crate::proof::primer) (§10).

use std::io;

use ark_ff::PrimeField;
use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::transcript::Transcript;

use super::argument::{commit_column, StoreProverContext, Transcript254};
use super::bgp::{RelationCommit, RelationWitness};
use super::commitment::{reject_nan_objects, LayerCommitment, ResolutionOpenings, VerifierLayer};
use super::query::{
    build_resolution_openings, canonical_dictionary_id, require_base_layer_for_query,
    verify_dict_value_terms,
};
use super::filter::{
    prove_filter_indicator_with_context, verify_filter_indicator, FilterProof,
};
use super::permutation::{
    prove_relation_permutation_with_context, verify_relation_permutation,
    RelationPermutationProof,
};
use super::range::{
    honest_dict_lookup, prove_dict_value_lookup_with_context, verify_dict_value_lookup,
    DictLookupProof,
};
use super::relocate::{
    prove_column_relocation_with_context, verify_column_relocation, RelocationProof,
};
use super::sort::{
    prove_nondecreasing_bits_with_context, verify_nondecreasing_bits, SortedProof,
};
use super::{invalid, CanonicalDatatype};

const ORDER_DOMAIN: &[u8] = b"parture/proof-of-woql/exact-scalar-order/v0";
const ORDER_BITS: usize = 64;

/// Sort direction for an ordered column: ascending or descending. Descending is proved by
/// sorting an affine `MAX - value` transform ascending, so one nondecreasing gadget serves
/// both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrderDirection {
    /// Sort smallest-first.
    Ascending,
    /// Sort largest-first (proved by sorting `MAX - value` ascending).
    Descending,
}

/// An exact single-column order proof: the permutation subproof (output is the input's rows
/// reordered), the relocations tying the id/dictionary columns to a common domain, the
/// authenticated `id -> (value, datatype)` dictionary lookup, the datatype-completeness
/// filter, and the nondecreasing-order proof over the (possibly complemented) sort key.
#[derive(Clone, Debug)]
pub struct ExactScalarOrderProof {
    /// Output is a full-row permutation of the input (same rows, same multiplicities).
    pub permutation: RelationPermutationProof,
    /// Relocation of the sorted output id column into the common lookup domain.
    pub relocate_output_id: RelocationProof,
    /// Relocation of the dictionary id column into the common lookup domain.
    pub relocate_dict_id: RelocationProof,
    /// Relocation of the dictionary value column into the common lookup domain.
    pub relocate_dict_value: RelocationProof,
    /// Relocation of the dictionary datatype column into the common lookup domain.
    pub relocate_dict_datatype: RelocationProof,
    /// Commitment to the sorted output id column at the common domain `d`.
    pub com_output_id_at_d: ArkGT,
    /// Commitment to the dictionary id column at the common domain `d`.
    pub com_dict_id_at_d: ArkGT,
    /// Commitment to the dictionary value column at the common domain `d`.
    pub com_dict_value_at_d: ArkGT,
    /// Commitment to the dictionary datatype column at the common domain `d`.
    pub com_dict_datatype_at_d: ArkGT,
    /// Authenticates each output id's `(value, datatype)` against the root-bound dictionary.
    pub lookup: DictLookupProof,
    /// Filter proof that every real row's datatype equals the planned sort datatype.
    pub datatype_filter: FilterProof,
    /// Commitment to the datatype filter's selection column.
    pub datatype_selection_commitment: ArkGT,
    /// Commitment to the datatype filter's pseudo-inverse witness column.
    pub datatype_inverse_commitment: ArkGT,
    /// Nondecreasing-order proof over the (possibly complemented) authenticated value key.
    pub sorted: SortedProof,
    /// Merkle-dict I2b step 3 (value↔term consistency, TERM-LESS order). The real object
    /// ids of the SORTED output rows (`oid[i]` = the id at sorted row `i`, length = the
    /// output row count), and one positional Merkle opening per DISTINCT id (node/value
    /// namespace). EMPTY / `None` on a TERM-BEARING artifact (the catalog re-MSM ties the
    /// sort-key value/datatype to real terms; the verifier ignores these). On a TERM-LESS
    /// artifact the verifier recommits `oid` to `com_output_id_at_d` (binding the sort key's
    /// object-id column) and rederives `lookup.com_v_val`/`lookup.com_v_dt` from the Merkle-
    /// authenticated opened terms (`verify_dict_value_terms`). UNCOMMITTED — never folded
    /// into `commitment_root`/`resolution_root`; a missing opening (or recommit mismatch)
    /// fails closed.
    pub oid: Vec<u64>,
    /// One positional Merkle opening per distinct `oid` (node/value namespace) on a
    /// TERM-LESS artifact; `None` on a TERM-BEARING one. Consumed alongside `oid`.
    pub value_openings: Option<ResolutionOpenings>,
}

/// Prover result: the proof plus the ordered output relation (the single id column, its
/// commitment, the `log_domain`, and the real row count `n`).
pub struct ExactScalarOrderResult {
    /// The order proof to hand to [`verify_exact_scalar_order`].
    pub proof: ExactScalarOrderProof,
    /// Commitment(s) to the ordered output relation (the single id column).
    pub commitments: Vec<ArkGT>,
    /// The ordered output column data, padded to `2^log_domain`.
    pub columns: Vec<Vec<ArkFr>>,
    /// `log2` of the output padding domain.
    pub log_domain: usize,
    /// The real output row count.
    pub n: usize,
}

fn supported_datatype(datatype: CanonicalDatatype) -> bool {
    // `Float32`/`Float64` are DELIBERATELY excluded (defense-in-depth, matching the
    // `integral_op_datatype` extrema guard). The IEEE total-order float key `node_value_op_value`
    // commits sorts `-0.0` below `+0.0` and `NaN` above `+∞`, diverging from WOQL numeric order;
    // the woql2 compiler already reject-closes float ORDER, so this gadget never receives a float
    // key in production — this guard ensures it stays that way.
    matches!(
        datatype,
        CanonicalDatatype::Boolean
            | CanonicalDatatype::UInt8
            | CanonicalDatatype::UInt16
            | CanonicalDatatype::UInt32
            | CanonicalDatatype::UInt64
            | CanonicalDatatype::Int8
            | CanonicalDatatype::Int16
            | CanonicalDatatype::Int32
            | CanonicalDatatype::Int64
    )
}

fn pad(column: &[ArkFr], size: usize) -> Vec<ArkFr> {
    let mut result = column.to_vec();
    result.resize(size, ArkFr::zero());
    result
}

fn bind_statement(
    transcript: &mut Transcript254,
    datatype: CanonicalDatatype,
    direction: OrderDirection,
    n: usize,
    common_log_domain: usize,
) {
    transcript.append_bytes(b"pow/order/domain", ORDER_DOMAIN);
    transcript.append_bytes(b"pow/order/datatype", &[datatype as u8]);
    transcript.append_bytes(b"pow/order/direction", &[match direction {
        OrderDirection::Ascending => 0,
        OrderDirection::Descending => 1,
    }]);
    transcript.append_bytes(b"pow/order/rows", &(n as u64).to_le_bytes());
    transcript.append_bytes(
        b"pow/order/common-log-domain",
        &(common_log_domain as u64).to_le_bytes(),
    );
}

/// Prove an exact single-column `OrderBy(direction)` of a one-column id relation whose ids
/// all resolve, through the root-bound `layer` dictionary, to values of one fixed-width
/// numeric `datatype`. Returns the ordered output relation and a proof that it is exactly
/// the input's rows in that order (duplicates preserved, no fabricated rows).
pub fn prove_exact_scalar_order_with_context(
    context: &StoreProverContext<'_>,
    layer: &LayerCommitment,
    input: &RelationWitness,
    datatype: CanonicalDatatype,
    direction: OrderDirection,
    transcript: &mut Transcript254,
) -> io::Result<ExactScalarOrderResult> {
    if !supported_datatype(datatype) || input.columns.len() != 1 || input.commitments.len() != 1 {
        return Err(invalid("exact scalar order requires one supported numeric column"));
    }
    layer.validate()?;
    super::commitment::reject_nan_catalog(
        &layer.node_value_catalog,
        ArkFr::from_u64(datatype as u64),
    )?;
    let n_dict = layer.node_value_catalog.len();
    let (_, dict_id) = layer.node_value_dictionary_columns()?;
    let dict_value = layer.node_value_dictionary_value_column()?;
    let dict_datatype = layer.node_value_dictionary_datatype_column()?;
    if dict_id.is_empty() || !dict_id.len().is_power_of_two() {
        return Err(invalid("exact scalar order requires a nonempty authenticated dictionary"));
    }
    let dict_log_domain = dict_id.len().trailing_zeros() as usize;
    let common_log_domain = input.log_domain.max(dict_log_domain);
    let common_domain = 1usize << common_log_domain;

    let input_id_at_d = pad(&input.columns[0], common_domain);
    let dict_id_at_d = pad(&dict_id, common_domain);
    let dict_value_at_d = pad(&dict_value, common_domain);
    let dict_datatype_at_d = pad(&dict_datatype, common_domain);
    let (input_values, input_datatypes, _) = honest_dict_lookup(
        &input_id_at_d,
        &dict_id_at_d,
        &dict_value_at_d,
        &dict_datatype_at_d,
        input.n,
        n_dict,
    );
    let expected_datatype = ArkFr::from_u64(datatype as u64);
    if input_datatypes[..input.n]
        .iter()
        .any(|actual| *actual != expected_datatype)
    {
        return Err(invalid("exact scalar order input contains another datatype"));
    }
    let mut row_order: Vec<usize> = (0..input.n).collect();
    row_order.sort_by(|left, right| {
        input_values[*left]
            .0
            .into_bigint()
            .cmp(&input_values[*right].0.into_bigint())
    });
    if direction == OrderDirection::Descending {
        row_order.reverse();
    }
    bind_statement(transcript, datatype, direction, input.n, common_log_domain);
    let permutation =
        prove_relation_permutation_with_context(context, input, &row_order, transcript)?;
    let output_id_at_d = pad(&permutation.columns[0], common_domain);
    let com_output_id_at_d = context.commit_column(&output_id_at_d)?;
    let relocate_output_id = prove_column_relocation_with_context(
        context,
        &permutation.columns[0],
        &permutation.commitments[0],
        &output_id_at_d,
        &com_output_id_at_d,
        transcript,
    )?;

    let relocate = |source: &[ArkFr], source_commitment: &ArkGT, target: &[ArkFr], target_commitment: &ArkGT, transcript: &mut Transcript254| {
        prove_column_relocation_with_context(
            context,
            source,
            source_commitment,
            target,
            target_commitment,
            transcript,
        )
    };
    let com_dict_id_at_d = context.commit_column(&dict_id_at_d)?;
    let relocate_dict_id = relocate(
        &dict_id,
        &layer.node_value_dict_commitments.1,
        &dict_id_at_d,
        &com_dict_id_at_d,
        transcript,
    )?;
    let com_dict_value_at_d = context.commit_column(&dict_value_at_d)?;
    let relocate_dict_value = relocate(
        &dict_value,
        &layer.node_value_dict_range_commitments.0,
        &dict_value_at_d,
        &com_dict_value_at_d,
        transcript,
    )?;
    let com_dict_datatype_at_d = context.commit_column(&dict_datatype_at_d)?;
    let relocate_dict_datatype = relocate(
        &dict_datatype,
        &layer.node_value_dict_range_commitments.1,
        &dict_datatype_at_d,
        &com_dict_datatype_at_d,
        transcript,
    )?;

    let lookup = prove_dict_value_lookup_with_context(
        context,
        &output_id_at_d,
        &com_output_id_at_d,
        &dict_id_at_d,
        &dict_value_at_d,
        &dict_datatype_at_d,
        &com_dict_id_at_d,
        &com_dict_value_at_d,
        &com_dict_datatype_at_d,
        input.n,
        n_dict,
        transcript,
    )?;
    let (datatype_filter, datatype_selection_commitment, datatype_inverse_commitment, _) =
        prove_filter_indicator_with_context(
            context,
            &input_datatypes_for_output(
                &output_id_at_d,
                &dict_id_at_d,
                &dict_value_at_d,
                &dict_datatype_at_d,
                input.n,
                n_dict,
            ),
            expected_datatype,
            input.n,
            &lookup.com_v_dt,
            transcript,
        )?;
    let mut chi = vec![ArkFr::zero(); common_domain];
    chi[..input.n].fill(ArkFr::one());
    if datatype_selection_commitment != context.commit_column(&chi)? {
        return Err(invalid("exact scalar order datatype selection is not complete"));
    }
    let output_values = input_values_for_output(
        &output_id_at_d,
        &dict_id_at_d,
        &dict_value_at_d,
        &dict_datatype_at_d,
        input.n,
        n_dict,
    );
    let sort_values = match direction {
        OrderDirection::Ascending => output_values.clone(),
        OrderDirection::Descending => {
            let max = ArkFr::from_u64(u64::MAX);
            output_values
                .iter()
                .enumerate()
                .map(|(index, value)| if index < input.n { max - *value } else { ArkFr::zero() })
                .collect()
        }
    };
    let sort_commitment = match direction {
        OrderDirection::Ascending => lookup.com_v_val,
        OrderDirection::Descending => {
            ArkFr::from_u64(u64::MAX) * datatype_selection_commitment - lookup.com_v_val
        }
    };
    if context.commit_column(&sort_values)? != sort_commitment {
        return Err(invalid("descending affine sort-key commitment mismatch"));
    }
    let sorted = prove_nondecreasing_bits_with_context(
        context,
        &sort_values,
        input.n,
        ORDER_BITS,
        &sort_commitment,
        transcript,
    )?;

    Ok(ExactScalarOrderResult {
        proof: ExactScalarOrderProof {
            permutation: permutation.proof,
            relocate_output_id,
            relocate_dict_id,
            relocate_dict_value,
            relocate_dict_datatype,
            com_output_id_at_d,
            com_dict_id_at_d,
            com_dict_value_at_d,
            com_dict_datatype_at_d,
            lookup,
            datatype_filter,
            datatype_selection_commitment,
            datatype_inverse_commitment,
            sorted,
            // Merkle-dict I2b step 3: term-bearing by default (the catalog re-MSM ties the
            // sort-key value/datatype to terms). `build_termless_order_openings` fills these
            // when a TERM-LESS artifact is shipped for a base/rollup layer.
            oid: Vec::new(),
            value_openings: None,
        },
        commitments: permutation.commitments,
        columns: permutation.columns,
        log_domain: permutation.log_domain,
        n: permutation.n,
    })
}

fn input_values_for_output(
    ids: &[ArkFr],
    dict_id: &[ArkFr],
    dict_value: &[ArkFr],
    dict_datatype: &[ArkFr],
    n: usize,
    n_dict: usize,
) -> Vec<ArkFr> {
    honest_dict_lookup(ids, dict_id, dict_value, dict_datatype, n, n_dict).0
}

fn input_datatypes_for_output(
    ids: &[ArkFr],
    dict_id: &[ArkFr],
    dict_value: &[ArkFr],
    dict_datatype: &[ArkFr],
    n: usize,
    n_dict: usize,
) -> Vec<ArkFr> {
    honest_dict_lookup(ids, dict_id, dict_value, dict_datatype, n, n_dict).1
}

/// Verify an exact single-column order proof: the relation committed by `output_commitments`
/// is the input's rows, of the given `datatype`, in the requested `direction`, under the
/// `source` layer's authenticated dictionary values.
pub fn verify_exact_scalar_order(
    source: &impl VerifierLayer,
    input: &RelationCommit,
    datatype: CanonicalDatatype,
    direction: OrderDirection,
    output_commitments: &[ArkGT],
    output_len: usize,
    output_log_domain: usize,
    proof: &ExactScalarOrderProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    if !supported_datatype(datatype) || input.commitments.len() != 1 {
        return Err(invalid("exact scalar order requires one supported numeric column"));
    }
    let layer = source.verifier_view();
    layer.validate()?;
    // Merkle-dict I2b step 3 (value↔term consistency). The ordering key is each output id's
    // order-preserving dict VALUE (`lookup.com_v_val`) plus its datatype (`lookup.com_v_dt`).
    //   * TERM-BEARING: `validate()`'s catalog re-MSM ties value/datatype to the clear terms and
    //     the NaN guard scans the catalog; read `n_dict` / the dict domain from the catalog.
    //     Unchanged.
    //   * TERM-LESS: the re-MSM is dropped ⇒ the committed value/datatype are prover-chosen. Read
    //     `n_dict` / the dict domain from the root-bound counts, and (once the lookup pins the
    //     per-row value/datatype below) tie them to the Merkle-authenticated terms via the prover-
    //     shipped `oid` + value openings. Missing openings ⇒ Err (fail-closed). DELTA order stays
    //     fail-closed via the `require_base_layer_for_query` belt below (E1 makes a term-less delta
    //     constructible/decodable, so `validate()` no longer rejects it — this belt is the gate).
    let (n_dict, dict_log_domain) = if layer.terms_present {
        super::commitment::reject_nan_catalog(
            layer.node_value_catalog,
            ArkFr::from_u64(datatype as u64),
        )?;
        let n_dict = layer.node_value_catalog.len();
        let (_, dict_id) = layer.node_value_dictionary_columns()?;
        if dict_id.is_empty() || !dict_id.len().is_power_of_two() {
            return Err(invalid("exact scalar order requires a nonempty authenticated dictionary"));
        }
        (n_dict, dict_id.len().trailing_zeros() as usize)
    } else {
        require_base_layer_for_query(layer.state)?;
        let n_dict = layer.state.node_value_count as usize;
        if n_dict == 0 {
            return Err(invalid("exact scalar order requires a nonempty authenticated dictionary"));
        }
        (n_dict, n_dict.next_power_of_two().max(2).trailing_zeros() as usize)
    };
    let common_log_domain = input.log_domain.max(dict_log_domain);
    bind_statement(transcript, datatype, direction, input.n, common_log_domain);
    verify_relation_permutation(
        input,
        output_commitments,
        output_len,
        output_log_domain,
        &proof.permutation,
        transcript,
    )?;
    verify_column_relocation(
        &output_commitments[0],
        output_log_domain,
        &proof.com_output_id_at_d,
        common_log_domain,
        &proof.relocate_output_id,
        transcript,
    )?;
    for (source_commitment, target_commitment, relocation) in [
        (
            layer.node_value_dict_commitments.1,
            proof.com_dict_id_at_d,
            &proof.relocate_dict_id,
        ),
        (
            layer.node_value_dict_range_commitments.0,
            proof.com_dict_value_at_d,
            &proof.relocate_dict_value,
        ),
        (
            layer.node_value_dict_range_commitments.1,
            proof.com_dict_datatype_at_d,
            &proof.relocate_dict_datatype,
        ),
    ] {
        verify_column_relocation(
            &source_commitment,
            dict_log_domain,
            &target_commitment,
            common_log_domain,
            relocation,
            transcript,
        )?;
    }
    // Merkle-dict I2b step 3 (TERM-LESS value↔term tie). The lookup below binds
    // `lookup.com_v_val`/`lookup.com_v_dt` as the dict value/datatype gathered at each SORTED
    // output id (`com_output_id_at_d`); on a term-less artifact those committed dict columns are
    // prover-chosen. Recommit the shipped `oid` to `com_output_id_at_d` (binding the exact sort-
    // key object column) and rederive the value/datatype from the Merkle-authenticated opened
    // terms, forcing `lookup.com_v_val`/`lookup.com_v_dt` == OP-encode(real term) at every row —
    // the SAME columns the sort key (`sort_commitment`) and datatype mask below consume. Runs
    // BEFORE the (transcript-consuming) lookup; it is itself transcript-free.
    if !layer.terms_present && layer.attested {
        // ATTESTED term-less: the root-bound value/datatype columns are authentic (the caller
        // vouched for the root's well-formedness); the whole-catalog NaN guard reads the facts.
        // No `oid`/value openings may ride.
        if !proof.oid.is_empty() || proof.value_openings.is_some() {
            return Err(invalid("an attested scalar OrderBy carries no per-row value openings"));
        }
        layer.reject_nan_whole_catalog(ArkFr::from_u64(datatype as u64))?;
    } else if !layer.terms_present {
        let value_openings = proof.value_openings.as_ref().ok_or_else(|| {
            invalid(
                "term-less exact scalar OrderBy requires the prover-shipped value↔term Merkle openings",
            )
        })?;
        let domain = 1usize << common_log_domain;
        let touched = verify_dict_value_terms(
            &proof.com_output_id_at_d,
            &proof.lookup.com_v_val,
            &proof.lookup.com_v_dt,
            output_len,
            domain,
            &proof.oid,
            value_openings,
            layer,
            layer.resolution_root(),
        )?;
        reject_nan_objects(&touched, ArkFr::from_u64(datatype as u64))?;
    }
    verify_dict_value_lookup(
        &proof.com_output_id_at_d,
        &proof.com_dict_id_at_d,
        &proof.com_dict_value_at_d,
        &proof.com_dict_datatype_at_d,
        output_len,
        n_dict,
        &proof.lookup,
        transcript,
    )?;
    verify_filter_indicator(
        &proof.lookup.com_v_dt,
        ArkFr::from_u64(datatype as u64),
        output_len,
        &proof.datatype_selection_commitment,
        &proof.datatype_inverse_commitment,
        &proof.datatype_filter,
        transcript,
    )?;
    let mut chi = vec![ArkFr::zero(); 1usize << common_log_domain];
    chi[..output_len].fill(ArkFr::one());
    if proof.datatype_selection_commitment != commit_column(&chi)? {
        return Err(invalid("exact scalar order datatype selection is not complete"));
    }
    let sort_commitment = match direction {
        OrderDirection::Ascending => proof.lookup.com_v_val,
        OrderDirection::Descending => {
            ArkFr::from_u64(u64::MAX) * proof.datatype_selection_commitment
                - proof.lookup.com_v_val
        }
    };
    verify_nondecreasing_bits(
        &sort_commitment,
        output_len,
        ORDER_BITS,
        &proof.sorted,
        transcript,
    )
}

/// Merkle-dict I2b step 3 (producer): mint the TERM-LESS value↔term witness for a proven
/// base/rollup exact-scalar-order proof. `output_ids` are the SORTED output id column's real
/// rows (`result.columns[0][..result.n]` — the exact ids `com_output_id_at_d` commits and the
/// lookup gathered `(v_val, v_dt)` from). Records them as `proof.oid` and opens each DISTINCT id
/// against the node/value tree, populating `proof.value_openings` IN PLACE.
///
/// UNCOMMITTED witness (never folded into `commitment_root`/`resolution_root`): at verify,
/// `verify_dict_value_terms` recommits `oid` to `com_output_id_at_d` (binding the sort-key
/// object column) and rederives the sort key value/datatype from the opened terms, failing closed
/// on any mismatch or missing opening. BASE/ROLLUP single-layer ONLY (delta order stays
/// fail-closed).
pub fn build_termless_order_openings(
    layer: &LayerCommitment,
    proof: &mut ExactScalarOrderProof,
    output_ids: &[ArkFr],
) -> io::Result<()> {
    require_base_layer_for_query(layer.verifier_view().state)?;
    let mut oid: Vec<u64> = Vec::with_capacity(output_ids.len());
    let mut nv_ids: Vec<u64> = Vec::new();
    for value in output_ids {
        let raw = canonical_dictionary_id(value)?;
        oid.push(raw);
        // The reserved null id 0 never indexes a dict row (never opened; masked out).
        if raw != 0 {
            nv_ids.push(raw);
        }
    }
    let openings = build_resolution_openings(&[layer.verifier_view()], &nv_ids, &[])?;
    proof.oid = oid;
    proof.value_openings = Some(openings);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::ValueTriple;
    use crate::store::sync::open_sync_memory_store;
    use crate::proof::CanonicalObject;
    use tdb_succinct::TdbDataType;

    fn layer() -> std::sync::Arc<LayerCommitment> {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for (subject, value) in [("a", 30u32), ("b", 10), ("c", 20), ("z", 0), ("m", u32::MAX)] {
            builder
                .add_value_triple(ValueTriple::new_value(
                    subject,
                    "score",
                    u32::make_entry(&value),
                ))
                .unwrap();
        }
        builder.commit().unwrap().proof_commitment().unwrap()
    }

    fn id_for(layer: &LayerCommitment, value: u32) -> u64 {
        let object = CanonicalObject::from_store_object(
            &ValueTriple::new_value("", "", u32::make_entry(&value)).object,
        );
        layer
            .node_value_catalog
            .iter()
            .find(|record| record.triple.object == object)
            .unwrap()
            .object_id
    }

    fn transcript() -> Transcript254 {
        Transcript254::new(b"parture-exact-scalar-order-test")
    }

    #[test]
    fn proves_exact_ascending_numeric_order_and_rejects_tamper() {
        let layer = layer();
        let ids = [id_for(&layer, 30), id_for(&layer, 10), id_for(&layer, 20)];
        let input = RelationWitness::commit(
            vec![ids.iter().map(|id| ArkFr::from_u64(*id)).collect()],
            3,
        )
        .unwrap();
        let mut prover = transcript();
        let result = prove_exact_scalar_order_with_context(
            &StoreProverContext::cpu(),
            &layer,
            &input,
            CanonicalDatatype::UInt32,
            OrderDirection::Ascending,
            &mut prover,
        )
        .unwrap();
        assert_eq!(
            &result.columns[0][..3],
            [
                ArkFr::from_u64(id_for(&layer, 10)),
                ArkFr::from_u64(id_for(&layer, 20)),
                ArkFr::from_u64(id_for(&layer, 30)),
            ]
        );
        let mut verifier = transcript();
        verify_exact_scalar_order(
            &layer,
            &input.view(),
            CanonicalDatatype::UInt32,
            OrderDirection::Ascending,
            &result.commitments,
            result.n,
            result.log_domain,
            &result.proof,
            &mut verifier,
        )
        .unwrap();

        let mut wrong_type = transcript();
        assert!(verify_exact_scalar_order(
            &layer,
            &input.view(),
            CanonicalDatatype::Int32,
            OrderDirection::Ascending,
            &result.commitments,
            result.n,
            result.log_domain,
            &result.proof,
            &mut wrong_type,
        )
        .is_err());
        let mut wrong_len = transcript();
        assert!(verify_exact_scalar_order(
            &layer,
            &input.view(),
            CanonicalDatatype::UInt32,
            OrderDirection::Ascending,
            &result.commitments,
            result.n - 1,
            result.log_domain,
            &result.proof,
            &mut wrong_len,
        )
        .is_err());
    }

    /// Merkle-dict I2b S2 (N1): a valid scalar-order proof rejects on a TERM-LESS view of the
    /// same base layer — the dictionary-VALUE sort key is untied to real terms without the
    /// catalog re-MSM. Term-bearing verify still succeeds (the guard is a no-op there).
    #[test]
    fn scalar_order_verify_fails_closed_on_termless_view() {
        let layer = layer();
        let ids = [id_for(&layer, 30), id_for(&layer, 10), id_for(&layer, 20)];
        let input =
            RelationWitness::commit(vec![ids.iter().map(|id| ArkFr::from_u64(*id)).collect()], 3)
                .unwrap();
        let mut prover = transcript();
        let result = prove_exact_scalar_order_with_context(
            &StoreProverContext::cpu(),
            &layer,
            &input,
            CanonicalDatatype::UInt32,
            OrderDirection::Ascending,
            &mut prover,
        )
        .unwrap();
        let mut verifier = transcript();
        verify_exact_scalar_order(
            &layer,
            &input.view(),
            CanonicalDatatype::UInt32,
            OrderDirection::Ascending,
            &result.commitments,
            result.n,
            result.log_domain,
            &result.proof,
            &mut verifier,
        )
        .unwrap();

        let termless = layer.verifier_commitment_termless().unwrap();
        let mut tv = transcript();
        let err = verify_exact_scalar_order(
            &termless,
            &input.view(),
            CanonicalDatatype::UInt32,
            OrderDirection::Ascending,
            &result.commitments,
            result.n,
            result.log_domain,
            &result.proof,
            &mut tv,
        )
        .unwrap_err();
        // Merkle-dict I2b step 3: a term-less scalar-order proof WITHOUT the value openings still
        // fails closed, now with the step-3 "value↔term Merkle openings" message.
        assert!(err.to_string().contains("Merkle openings"), "{err}");
    }

    #[test]
    fn preserves_duplicate_keys_and_rejects_mixed_datatypes() {
        let layer = layer();
        let ten = id_for(&layer, 10);
        let duplicate = RelationWitness::commit(
            vec![vec![ArkFr::from_u64(ten), ArkFr::from_u64(ten)]],
            2,
        )
        .unwrap();
        let result = prove_exact_scalar_order_with_context(
            &StoreProverContext::cpu(),
            &layer,
            &duplicate,
            CanonicalDatatype::UInt32,
            OrderDirection::Ascending,
            &mut transcript(),
        ).unwrap();
        assert_eq!(&result.columns[0][..2], &[ArkFr::from_u64(ten); 2]);
        verify_exact_scalar_order(
            &layer, &duplicate.view(), CanonicalDatatype::UInt32,
            OrderDirection::Ascending, &result.commitments, result.n,
            result.log_domain, &result.proof, &mut transcript(),
        ).unwrap();

        let node_id = layer
            .node_value_catalog
            .iter()
            .find(|record| matches!(record.triple.object, CanonicalObject::Node(_)))
            .unwrap()
            .object_id;
        let mixed = RelationWitness::commit(
            vec![vec![ArkFr::from_u64(ten), ArkFr::from_u64(node_id)]],
            2,
        )
        .unwrap();
        assert!(prove_exact_scalar_order_with_context(
            &StoreProverContext::cpu(),
            &layer,
            &mixed,
            CanonicalDatatype::UInt32,
            OrderDirection::Ascending,
            &mut transcript(),
        )
        .is_err());
    }

    #[test]
    fn proves_descending_edges_and_rejects_ascending_substitution() {
        let layer = layer();
        let ids = [0u32, u32::MAX, 20]
            .map(|value| ArkFr::from_u64(id_for(&layer, value)));
        let input = RelationWitness::commit(vec![ids.to_vec()], 3).unwrap();
        let result = prove_exact_scalar_order_with_context(
            &StoreProverContext::cpu(), &layer, &input, CanonicalDatatype::UInt32,
            OrderDirection::Descending, &mut transcript(),
        ).unwrap();
        assert_eq!(&result.columns[0][..3], &[
            ArkFr::from_u64(id_for(&layer, u32::MAX)),
            ArkFr::from_u64(id_for(&layer, 20)),
            ArkFr::from_u64(id_for(&layer, 0)),
        ]);
        verify_exact_scalar_order(
            &layer, &input.view(), CanonicalDatatype::UInt32, OrderDirection::Descending,
            &result.commitments, result.n, result.log_domain, &result.proof, &mut transcript(),
        ).unwrap();
        assert!(verify_exact_scalar_order(
            &layer, &input.view(), CanonicalDatatype::UInt32, OrderDirection::Ascending,
            &result.commitments, result.n, result.log_domain, &result.proof, &mut transcript(),
        ).is_err());
    }

    // =====================================================================================
    // Merkle-dict I2b step 3 (value↔term consistency, ORDER). A TERM-LESS base/rollup order
    // proof ties the committed sort-key value/datatype to the Merkle-authenticated terms via the
    // prover-shipped `oid` + per-distinct value openings. Term-bearing verify is unchanged.
    // =====================================================================================
    fn prove_attach_order(
        layer: &LayerCommitment,
        dir: OrderDirection,
    ) -> (RelationWitness, ExactScalarOrderResult) {
        let ids = [id_for(layer, 30), id_for(layer, 10), id_for(layer, 20)];
        let input =
            RelationWitness::commit(vec![ids.iter().map(|id| ArkFr::from_u64(*id)).collect()], 3)
                .unwrap();
        let mut result = prove_exact_scalar_order_with_context(
            &StoreProverContext::cpu(), layer, &input, CanonicalDatatype::UInt32, dir,
            &mut transcript(),
        )
        .unwrap();
        let n = result.n;
        build_termless_order_openings(layer, &mut result.proof, &result.columns[0][..n]).unwrap();
        assert!(result.proof.value_openings.is_some(), "order value openings attached");
        assert!(!result.proof.oid.is_empty(), "order oid vector minted");
        (input, result)
    }

    /// Differential positive: a TERM-LESS scalar-order proof verifies EXACTLY like the
    /// TERM-BEARING one — same accept (honest), same reject (wrong datatype), ascending + descending.
    #[test]
    fn i2b_step3_order_termless_differential() {
        let layer = layer();
        for dir in [OrderDirection::Ascending, OrderDirection::Descending] {
            let (input, result) = prove_attach_order(&layer, dir);
            let termless = layer.verifier_commitment_termless().unwrap();
            verify_exact_scalar_order(&layer, &input.view(), CanonicalDatatype::UInt32, dir,
                &result.commitments, result.n, result.log_domain, &result.proof, &mut transcript()).unwrap();
            verify_exact_scalar_order(&termless, &input.view(), CanonicalDatatype::UInt32, dir,
                &result.commitments, result.n, result.log_domain, &result.proof, &mut transcript()).unwrap();
            // Same reject: a wrong queried datatype rejects on BOTH artifacts.
            assert!(verify_exact_scalar_order(&layer, &input.view(), CanonicalDatatype::Int32, dir,
                &result.commitments, result.n, result.log_domain, &result.proof, &mut transcript()).is_err());
            assert!(verify_exact_scalar_order(&termless, &input.view(), CanonicalDatatype::Int32, dir,
                &result.commitments, result.n, result.log_domain, &result.proof, &mut transcript()).is_err());
        }
    }

    /// Forged committed VALUE column (`lookup.com_v_val`) ⇒ the Merkle-rederived value disagrees ⇒
    /// `Com(ov) != com_v_val` ⇒ REJECT (term-less). Term-bearing is unaffected (its own lookup ties it).
    #[test]
    fn i2b_step3_order_termless_rejects_forged_value_column() {
        let layer = layer();
        let (input, mut result) = prove_attach_order(&layer, OrderDirection::Ascending);
        result.proof.lookup.com_v_val = result.proof.lookup.com_v_dt;
        let termless = layer.verifier_commitment_termless().unwrap();
        let err = verify_exact_scalar_order(&termless, &input.view(), CanonicalDatatype::UInt32,
            OrderDirection::Ascending, &result.commitments, result.n, result.log_domain, &result.proof,
            &mut transcript()).unwrap_err();
        assert!(err.to_string().contains("com_v_val"), "{err}");
    }

    /// A substituted `oid` vector no longer recommits to `com_output_id_at_d` ⇒ REJECT (only the
    /// tie checks `oid`).
    #[test]
    fn i2b_step3_order_termless_rejects_forged_oid() {
        let layer = layer();
        let (input, mut result) = prove_attach_order(&layer, OrderDirection::Ascending);
        result.proof.oid[0] = result.proof.oid[0].wrapping_add(1);
        let termless = layer.verifier_commitment_termless().unwrap();
        let err = verify_exact_scalar_order(&termless, &input.view(), CanonicalDatatype::UInt32,
            OrderDirection::Ascending, &result.commitments, result.n, result.log_domain, &result.proof,
            &mut transcript()).unwrap_err();
        assert!(err.to_string().contains("oid column"), "{err}");
    }

    /// A term-less order proof WITHOUT the value openings fails closed; term-bearing still verifies.
    #[test]
    fn i2b_step3_order_termless_rejects_missing_openings() {
        let layer = layer();
        let (input, mut result) = prove_attach_order(&layer, OrderDirection::Ascending);
        result.proof.value_openings = None;
        let termless = layer.verifier_commitment_termless().unwrap();
        let err = verify_exact_scalar_order(&termless, &input.view(), CanonicalDatatype::UInt32,
            OrderDirection::Ascending, &result.commitments, result.n, result.log_domain, &result.proof,
            &mut transcript()).unwrap_err();
        assert!(err.to_string().contains("Merkle openings"), "{err}");
        verify_exact_scalar_order(&layer, &input.view(), CanonicalDatatype::UInt32,
            OrderDirection::Ascending, &result.commitments, result.n, result.log_domain, &result.proof,
            &mut transcript()).unwrap();
    }
}
