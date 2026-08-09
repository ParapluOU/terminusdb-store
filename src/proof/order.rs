//! Authenticated ascending order for a one-column exact numeric relation.
//!
//! This deliberately narrow gadget is the first sound `OrderBy` slice: the sole output
//! column is also the complete sort key, every value is a nonnullable Store ID resolving
//! to one fixed-width order-preserving datatype, the output is an exact full-row
//! permutation, and the authenticated dictionary value column is strictly ascending.

use std::io;

use ark_ff::PrimeField;
use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::transcript::Transcript;

use super::argument::{commit_column, StoreProverContext, Transcript254};
use super::bgp::{RelationCommit, RelationWitness};
use super::commitment::{LayerCommitment, VerifierLayer};
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
    prove_strictly_sorted_bits_with_context, verify_strictly_sorted_bits, SortedProof,
};
use super::{invalid, CanonicalDatatype};

const ORDER_DOMAIN: &[u8] = b"parture/proof-of-woql/exact-scalar-order/v0";
const ORDER_BITS: usize = 64;

#[derive(Clone, Debug)]
pub struct ExactScalarOrderProof {
    pub permutation: RelationPermutationProof,
    pub relocate_output_id: RelocationProof,
    pub relocate_dict_id: RelocationProof,
    pub relocate_dict_value: RelocationProof,
    pub relocate_dict_datatype: RelocationProof,
    pub com_output_id_at_d: ArkGT,
    pub com_dict_id_at_d: ArkGT,
    pub com_dict_value_at_d: ArkGT,
    pub com_dict_datatype_at_d: ArkGT,
    pub lookup: DictLookupProof,
    pub datatype_filter: FilterProof,
    pub datatype_selection_commitment: ArkGT,
    pub datatype_inverse_commitment: ArkGT,
    pub sorted: SortedProof,
}

pub struct ExactScalarOrderResult {
    pub proof: ExactScalarOrderProof,
    pub commitments: Vec<ArkGT>,
    pub columns: Vec<Vec<ArkFr>>,
    pub log_domain: usize,
    pub n: usize,
}

fn supported_datatype(datatype: CanonicalDatatype) -> bool {
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
            | CanonicalDatatype::Float32
            | CanonicalDatatype::Float64
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
    n: usize,
    common_log_domain: usize,
) {
    transcript.append_bytes(b"pow/order/domain", ORDER_DOMAIN);
    transcript.append_bytes(b"pow/order/datatype", &[datatype as u8]);
    transcript.append_bytes(b"pow/order/rows", &(n as u64).to_le_bytes());
    transcript.append_bytes(
        b"pow/order/common-log-domain",
        &(common_log_domain as u64).to_le_bytes(),
    );
}

pub fn prove_exact_scalar_order_with_context(
    context: &StoreProverContext<'_>,
    layer: &LayerCommitment,
    input: &RelationWitness,
    datatype: CanonicalDatatype,
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
    if row_order
        .windows(2)
        .any(|rows| input_values[rows[0]] == input_values[rows[1]])
    {
        return Err(invalid("exact scalar order key is not unique"));
    }

    bind_statement(transcript, datatype, input.n, common_log_domain);
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
    let sorted = prove_strictly_sorted_bits_with_context(
        context,
        &output_values,
        input.n,
        ORDER_BITS,
        &lookup.com_v_val,
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

pub fn verify_exact_scalar_order(
    source: &impl VerifierLayer,
    input: &RelationCommit,
    datatype: CanonicalDatatype,
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
    super::commitment::reject_nan_catalog(
        layer.node_value_catalog,
        ArkFr::from_u64(datatype as u64),
    )?;
    let n_dict = layer.node_value_catalog.len();
    let (_, dict_id) = layer.node_value_dictionary_columns()?;
    if dict_id.is_empty() || !dict_id.len().is_power_of_two() {
        return Err(invalid("exact scalar order requires a nonempty authenticated dictionary"));
    }
    let dict_log_domain = dict_id.len().trailing_zeros() as usize;
    let common_log_domain = input.log_domain.max(dict_log_domain);
    bind_statement(transcript, datatype, input.n, common_log_domain);
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
    verify_strictly_sorted_bits(
        &proof.lookup.com_v_val,
        output_len,
        ORDER_BITS,
        &proof.sorted,
        transcript,
    )
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
        for (subject, value) in [("a", 30u32), ("b", 10), ("c", 20)] {
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
            &result.commitments,
            result.n - 1,
            result.log_domain,
            &result.proof,
            &mut wrong_len,
        )
        .is_err());
    }

    #[test]
    fn rejects_duplicate_keys_and_mixed_datatypes() {
        let layer = layer();
        let ten = id_for(&layer, 10);
        let duplicate = RelationWitness::commit(
            vec![vec![ArkFr::from_u64(ten), ArkFr::from_u64(ten)]],
            2,
        )
        .unwrap();
        assert!(prove_exact_scalar_order_with_context(
            &StoreProverContext::cpu(),
            &layer,
            &duplicate,
            CanonicalDatatype::UInt32,
            &mut transcript(),
        )
        .is_err());

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
            &mut transcript(),
        )
        .is_err());
    }
}
