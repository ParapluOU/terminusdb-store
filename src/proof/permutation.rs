//! Exact full-row permutation proof for authenticated relations.
//!
//! Input and output rows are fingerprinted across every column with a transcript-drawn
//! RLC challenge. Two reciprocal identities plus one masked LogUp equality prove that
//! the real output rows are exactly the input row multiset, including duplicates and
//! tuple correlation. This gadget intentionally proves no ordering property: composing
//! `OrderBy` must additionally prove the requested ordering and, for SWI-compatible
//! semantics, stable order among equal keys.

use std::io;

use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::transcript::Transcript;

use super::argument::{
    prove_sumcheck, verify_column_openings, verify_sumcheck, ColumnOpening, StoreProverContext,
    SumcheckProof, SumcheckShape, Transcript254,
};
use super::bgp::{RelationCommit, RelationWitness};
use super::invalid;
use super::logup::{masked_multiset_link_claim, reciprocal_claim, reciprocal_witness, rlc_fold};

const PERMUTATION_DOMAIN: &[u8] = b"parture/proof-of-woql/relation-permutation/v0";

#[derive(Clone, Debug)]
pub struct RelationPermutationProof {
    pub input_reciprocal_commitment: ArkGT,
    pub output_reciprocal_commitment: ArkGT,
    pub sumcheck: SumcheckProof,
    pub opening: ColumnOpening,
}

pub struct RelationPermutationResult {
    pub proof: RelationPermutationProof,
    pub commitments: Vec<ArkGT>,
    pub columns: Vec<Vec<ArkFr>>,
    pub log_domain: usize,
    pub n: usize,
}

fn validate_relation(input: &RelationCommit) -> io::Result<usize> {
    if input.commitments.is_empty() {
        return Err(invalid("relation permutation needs nonzero arity"));
    }
    if input.log_domain == 0 || input.log_domain >= usize::BITS as usize {
        return Err(invalid("relation permutation domain out of range"));
    }
    if input.n > (1usize << input.log_domain) {
        return Err(invalid("relation permutation length exceeds its domain"));
    }
    Ok(input.commitments.len())
}

fn validate_witness(input: &RelationWitness) -> io::Result<usize> {
    let arity = validate_relation(&input.view())?;
    let domain = 1usize << input.log_domain;
    if input.columns.len() != arity || input.columns.iter().any(|column| column.len() != domain) {
        return Err(invalid("relation permutation witness shape mismatch"));
    }
    Ok(arity)
}

fn validate_row_order(row_order: &[usize], n: usize) -> io::Result<()> {
    if row_order.len() != n {
        return Err(invalid("relation permutation row-order length mismatch"));
    }
    let mut seen = vec![false; n];
    for &index in row_order {
        if index >= n || seen[index] {
            return Err(invalid("relation permutation row order is not a permutation"));
        }
        seen[index] = true;
    }
    Ok(())
}

fn bind_statement(
    transcript: &mut Transcript254,
    input: &RelationCommit,
    output_commitments: &[ArkGT],
) {
    transcript.append_bytes(b"pow/permutation/domain", PERMUTATION_DOMAIN);
    transcript.append_bytes(b"pow/permutation/arity", &(input.commitments.len() as u64).to_le_bytes());
    transcript.append_bytes(b"pow/permutation/rows", &(input.n as u64).to_le_bytes());
    transcript.append_bytes(b"pow/permutation/log-domain", &(input.log_domain as u64).to_le_bytes());
    for commitment in &input.commitments {
        transcript.append_group(b"pow/permutation/input-com", commitment);
    }
    for commitment in output_commitments {
        transcript.append_group(b"pow/permutation/output-com", commitment);
    }
}

fn permutation_shape(
    log_domain: usize,
    n: usize,
    arity: usize,
    alpha: ArkFr,
    beta: ArkFr,
) -> SumcheckShape {
    let input_columns: Vec<usize> = (0..arity).collect();
    let output_columns: Vec<usize> = (arity..2 * arity).collect();
    let input_reciprocal = 2 * arity;
    let output_reciprocal = input_reciprocal + 1;
    SumcheckShape {
        num_vars: log_domain,
        num_columns: 2 * arity + 2,
        chi_bounds: vec![n, n],
        claims: vec![
            reciprocal_claim(input_reciprocal, &input_columns, alpha, beta),
            reciprocal_claim(output_reciprocal, &output_columns, alpha, beta),
            masked_multiset_link_claim(
                &[super::argument::Operand::Chi(0)],
                input_reciprocal,
                &[super::argument::Operand::Chi(1)],
                output_reciprocal,
            ),
        ],
    }
}

pub fn prove_relation_permutation(
    input: &RelationWitness,
    row_order: &[usize],
    transcript: &mut Transcript254,
) -> io::Result<RelationPermutationResult> {
    prove_relation_permutation_with_context(
        &StoreProverContext::cpu(),
        input,
        row_order,
        transcript,
    )
}

pub fn prove_relation_permutation_with_context(
    context: &StoreProverContext<'_>,
    input: &RelationWitness,
    row_order: &[usize],
    transcript: &mut Transcript254,
) -> io::Result<RelationPermutationResult> {
    let arity = validate_witness(input)?;
    validate_row_order(row_order, input.n)?;
    let domain = 1usize << input.log_domain;
    let mut output_columns = Vec::with_capacity(arity);
    for input_column in &input.columns {
        let mut output: Vec<ArkFr> = row_order.iter().map(|&row| input_column[row]).collect();
        output.resize(domain, ArkFr::zero());
        output_columns.push(output);
    }
    let output_commitments = output_columns
        .iter()
        .map(|column| context.commit_column(column))
        .collect::<io::Result<Vec<_>>>()?;
    let input_view = input.view();
    bind_statement(transcript, &input_view, &output_commitments);
    let beta = transcript.challenge_scalar(b"pow/permutation/beta");
    let alpha = transcript.challenge_scalar(b"pow/permutation/alpha");

    let input_slices: Vec<&[ArkFr]> = input.columns.iter().map(Vec::as_slice).collect();
    let output_slices: Vec<&[ArkFr]> = output_columns.iter().map(Vec::as_slice).collect();
    let input_reciprocal = reciprocal_witness(alpha, &rlc_fold(beta, &input_slices));
    let output_reciprocal = reciprocal_witness(alpha, &rlc_fold(beta, &output_slices));
    let input_reciprocal_commitment = context.commit_column(&input_reciprocal)?;
    let output_reciprocal_commitment = context.commit_column(&output_reciprocal)?;
    transcript.append_group(b"pow/permutation/input-rec-com", &input_reciprocal_commitment);
    transcript.append_group(b"pow/permutation/output-rec-com", &output_reciprocal_commitment);

    let shape = permutation_shape(input.log_domain, input.n, arity, alpha, beta);
    let mut columns = Vec::with_capacity(shape.num_columns);
    columns.extend(input.columns.iter().cloned());
    columns.extend(output_columns.iter().cloned());
    columns.push(input_reciprocal);
    columns.push(output_reciprocal);
    let (sumcheck, point, _) = prove_sumcheck(&shape, &columns, transcript)?;
    let opening = context.prove_column_openings(&columns, &point, transcript)?;

    Ok(RelationPermutationResult {
        proof: RelationPermutationProof {
            input_reciprocal_commitment,
            output_reciprocal_commitment,
            sumcheck,
            opening,
        },
        commitments: output_commitments,
        columns: output_columns,
        log_domain: input.log_domain,
        n: input.n,
    })
}

pub fn verify_relation_permutation(
    input: &RelationCommit,
    output_commitments: &[ArkGT],
    output_len: usize,
    output_log_domain: usize,
    proof: &RelationPermutationProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    let arity = validate_relation(input)?;
    if output_commitments.len() != arity {
        return Err(invalid("relation permutation output arity mismatch"));
    }
    if output_len != input.n || output_log_domain != input.log_domain {
        return Err(invalid("relation permutation output shape mismatch"));
    }
    bind_statement(transcript, input, output_commitments);
    let beta = transcript.challenge_scalar(b"pow/permutation/beta");
    let alpha = transcript.challenge_scalar(b"pow/permutation/alpha");
    transcript.append_group(b"pow/permutation/input-rec-com", &proof.input_reciprocal_commitment);
    transcript.append_group(b"pow/permutation/output-rec-com", &proof.output_reciprocal_commitment);
    if proof.sumcheck.round_polys.len() != input.log_domain {
        return Err(invalid("relation permutation sumcheck domain mismatch"));
    }
    let shape = permutation_shape(input.log_domain, input.n, arity, alpha, beta);
    let (point, evals) = verify_sumcheck(&shape, &proof.sumcheck, transcript)?;
    let mut commitments = Vec::with_capacity(shape.num_columns);
    commitments.extend(input.commitments.iter().copied());
    commitments.extend(output_commitments.iter().copied());
    commitments.push(proof.input_reciprocal_commitment);
    commitments.push(proof.output_reciprocal_commitment);
    verify_column_openings(&commitments, &point, &evals, &proof.opening, transcript)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::argument::commit_column;

    fn frs(values: &[u64]) -> Vec<ArkFr> {
        values.iter().map(|value| ArkFr::from_u64(*value)).collect()
    }

    fn relation() -> RelationWitness {
        RelationWitness::commit(vec![frs(&[2, 1, 1]), frs(&[20, 11, 10])], 3).unwrap()
    }

    fn transcript() -> Transcript254 {
        Transcript254::new(b"parture-relation-permutation-test")
    }

    #[test]
    fn proves_full_row_permutation_with_duplicate_keys() {
        let input = relation();
        let mut prover = transcript();
        let result = prove_relation_permutation(&input, &[2, 1, 0], &mut prover).unwrap();
        assert_eq!(&result.columns[0][..3], frs(&[1, 1, 2]));
        assert_eq!(&result.columns[1][..3], frs(&[10, 11, 20]));
        let mut verifier = transcript();
        verify_relation_permutation(
            &input.view(),
            &result.commitments,
            result.n,
            result.log_domain,
            &result.proof,
            &mut verifier,
        )
        .unwrap();

        let mut spliced = result.columns.clone();
        spliced[1].swap(0, 2);
        let spliced_commitments = spliced
            .iter()
            .map(|column| commit_column(column))
            .collect::<io::Result<Vec<_>>>()
            .unwrap();
        let mut verifier = transcript();
        assert!(verify_relation_permutation(
            &input.view(),
            &spliced_commitments,
            result.n,
            result.log_domain,
            &result.proof,
            &mut verifier,
        )
        .is_err());
    }

    #[test]
    fn permutation_does_not_claim_stable_equal_key_order() {
        let input = relation();
        let mut first_transcript = transcript();
        let first = prove_relation_permutation(&input, &[1, 2, 0], &mut first_transcript).unwrap();
        let mut second_transcript = transcript();
        let second = prove_relation_permutation(&input, &[2, 1, 0], &mut second_transcript).unwrap();
        assert_eq!(&first.columns[0][..3], &second.columns[0][..3]);
        assert_ne!(&first.columns[1][..3], &second.columns[1][..3]);
        for result in [&first, &second] {
            let mut verifier = transcript();
            verify_relation_permutation(
                &input.view(),
                &result.commitments,
                result.n,
                result.log_domain,
                &result.proof,
                &mut verifier,
            )
            .unwrap();
        }
    }

    #[test]
    fn rejects_non_permutation_row_orders_and_forged_shape() {
        let input = relation();
        for order in [vec![0, 1], vec![0, 1, 1], vec![0, 1, 3]] {
            assert!(prove_relation_permutation(&input, &order, &mut transcript()).is_err());
        }
        let result = prove_relation_permutation(&input, &[0, 1, 2], &mut transcript()).unwrap();
        let mut verifier = transcript();
        assert!(verify_relation_permutation(
            &input.view(),
            &result.commitments,
            result.n - 1,
            result.log_domain,
            &result.proof,
            &mut verifier,
        )
        .is_err());
    }
}
