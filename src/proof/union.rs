//! Exact bag union over authenticated relation results.
//!
//! Every branch is independently committed and carries a verifier-pinned real length.
//! Branch columns are relocation-bound to one common domain, rows are fingerprinted by
//! a transcript-challenged RLC, and one LogUp identity proves that the output multiset is
//! exactly the sum of the branch multisets. The public output length is independently
//! required to equal the checked sum of branch lengths, preserving overlap multiplicity.

use std::io;

use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::transcript::Transcript;

use super::argument::{
    prove_sumcheck, verify_column_openings, verify_sumcheck, ClaimShape, ColumnOpening, Expr,
    Operand, SumcheckClaim, SumcheckProof,
    StoreProverContext, SumcheckShape, Term, Transcript254,
};
#[cfg(test)]
use super::argument::commit_column;
use super::bgp::{RelationCommit, RelationWitness};
use super::invalid;
use super::logup::{reciprocal_claim, reciprocal_witness, rlc_fold};
use super::relocate::{
    prove_column_relocation_with_context, verify_column_relocation, RelocationProof,
};

const UNION_DOMAIN: &[u8] = b"parture/proof-of-woql/bag-union/v0";

#[inline]
fn f0() -> ArkFr {
    ArkFr::zero()
}

#[inline]
fn f1() -> ArkFr {
    ArkFr::one()
}

fn padded_len(n: usize) -> io::Result<usize> {
    n.checked_next_power_of_two()
        .ok_or_else(|| invalid("bag-union domain overflow"))
        .map(|size| size.max(2))
}

fn zero_pad(values: &[ArkFr], size: usize) -> Vec<ArkFr> {
    let mut out = values.to_vec();
    out.resize(size, f0());
    out
}

fn checked_output_len(inputs: &[RelationCommit]) -> io::Result<usize> {
    inputs.iter().try_fold(0usize, |total, input| {
        total
            .checked_add(input.n)
            .ok_or_else(|| invalid("bag-union result length overflow"))
    })
}

fn validate_inputs(inputs: &[RelationCommit]) -> io::Result<usize> {
    if inputs.len() < 2 {
        return Err(invalid("bag union needs at least two branches"));
    }
    let arity = inputs[0].commitments.len();
    if arity == 0 {
        return Err(invalid("column bag union needs nonzero arity"));
    }
    for input in inputs {
        if input.commitments.len() != arity {
            return Err(invalid("bag-union branch arity mismatch"));
        }
        if input.log_domain == 0 || input.log_domain >= usize::BITS as usize {
            return Err(invalid("bag-union branch domain out of range"));
        }
        if input.n > (1usize << input.log_domain) {
            return Err(invalid("bag-union branch length exceeds its domain"));
        }
    }
    Ok(arity)
}

/// Proof that `output` is the exact bag union of all branch relations.
#[derive(Clone, Debug)]
pub struct BagUnionProof {
    /// Per branch, per column relocation into the common union domain.
    pub relocations: Vec<Vec<RelocationProof>>,
    /// Commitments consumed by the union LogUp after relocation.
    pub relocated_commitments: Vec<Vec<ArkGT>>,
    /// One reciprocal commitment per branch, followed by the output reciprocal.
    pub reciprocal_commitments: Vec<ArkGT>,
    pub sumcheck: SumcheckProof,
    pub opening: ColumnOpening,
}

/// Prover-side exact union result. Rows are emitted by branch order, while verification
/// intentionally authenticates bag equality and therefore permits result reordering.
pub struct BagUnionResult {
    pub proof: BagUnionProof,
    pub commitments: Vec<ArkGT>,
    pub columns: Vec<Vec<ArkFr>>,
    pub log_domain: usize,
    pub n: usize,
}

fn bind_statement(
    transcript: &mut Transcript254,
    inputs: &[RelationCommit],
    output_commitments: &[ArkGT],
    output_len: usize,
    log_domain: usize,
) {
    transcript.append_bytes(b"pow/union/domain", UNION_DOMAIN);
    transcript.append_bytes(b"pow/union/branches", &(inputs.len() as u64).to_le_bytes());
    transcript.append_bytes(
        b"pow/union/arity",
        &(output_commitments.len() as u64).to_le_bytes(),
    );
    transcript.append_bytes(b"pow/union/output-len", &(output_len as u64).to_le_bytes());
    transcript.append_bytes(b"pow/union/log-domain", &(log_domain as u64).to_le_bytes());
    for input in inputs {
        transcript.append_bytes(b"pow/union/input-domain", &(input.log_domain as u64).to_le_bytes());
        transcript.append_bytes(b"pow/union/input-len", &(input.n as u64).to_le_bytes());
        for commitment in &input.commitments {
            transcript.append_group(b"pow/union/input-com", commitment);
        }
    }
    for commitment in output_commitments {
        transcript.append_group(b"pow/union/output-com", commitment);
    }
}

fn bind_relocated(transcript: &mut Transcript254, commitments: &[Vec<ArkGT>]) {
    for branch in commitments {
        for commitment in branch {
            transcript.append_group(b"pow/union/relocated-com", commitment);
        }
    }
}

fn union_shape(
    num_vars: usize,
    branch_lengths: &[usize],
    output_len: usize,
    arity: usize,
    alpha: ArkFr,
    beta: ArkFr,
) -> SumcheckShape {
    let branches = branch_lengths.len();
    let output_base = branches * arity;
    let reciprocal_base = output_base + arity;
    let mut claims = Vec::with_capacity(branches + 2);
    for branch in 0..branches {
        let columns: Vec<usize> = (0..arity).map(|column| branch * arity + column).collect();
        claims.push(reciprocal_claim(
            reciprocal_base + branch,
            &columns,
            alpha,
            beta,
        ));
    }
    let output_columns: Vec<usize> = (0..arity).map(|column| output_base + column).collect();
    claims.push(reciprocal_claim(
        reciprocal_base + branches,
        &output_columns,
        alpha,
        beta,
    ));

    let mut terms: Vec<Term> = (0..branches)
        .map(|branch| Term {
            coeff: f1(),
            operands: vec![
                Operand::Chi(branch),
                Operand::Column(reciprocal_base + branch),
            ],
        })
        .collect();
    terms.push(Term {
        coeff: f0() - f1(),
        operands: vec![
            Operand::Chi(branches),
            Operand::Column(reciprocal_base + branches),
        ],
    });
    claims.push(SumcheckClaim {
        shape: ClaimShape::ZeroSum,
        expr: Expr { terms },
        target: f0(),
    });

    let mut chi_bounds = branch_lengths.to_vec();
    chi_bounds.push(output_len);
    SumcheckShape {
        num_vars,
        num_columns: (branches + 1) * (arity + 1),
        chi_bounds,
        claims,
    }
}

/// Prove the exact bag union of compatible branch relations.
pub fn prove_bag_union(
    inputs: &[RelationWitness],
    transcript: &mut Transcript254,
) -> io::Result<BagUnionResult> {
    prove_bag_union_with_context(&StoreProverContext::cpu(), inputs, transcript)
}

/// As [`prove_bag_union`], with an explicit prover-arithmetic context.
pub fn prove_bag_union_with_context(
    context: &StoreProverContext<'_>,
    inputs: &[RelationWitness],
    transcript: &mut Transcript254,
) -> io::Result<BagUnionResult> {
    let views: Vec<RelationCommit> = inputs.iter().map(RelationWitness::view).collect();
    let arity = validate_inputs(&views)?;
    for input in inputs {
        if input.columns.len() != arity
            || input.columns.len() != input.commitments.len()
            || input
                .columns
                .iter()
                .any(|column| column.len() != 1usize << input.log_domain)
        {
            return Err(invalid("bag-union witness shape mismatch"));
        }
    }
    let output_len = checked_output_len(&views)?;
    let domain = padded_len(output_len)?;
    let log_domain = domain.trailing_zeros() as usize;

    let mut output_columns = vec![Vec::with_capacity(output_len); arity];
    for input in inputs {
        for (column, output) in input.columns.iter().zip(&mut output_columns) {
            output.extend_from_slice(&column[..input.n]);
        }
    }
    for output in &mut output_columns {
        output.resize(domain, f0());
    }
    let output_commitments: Vec<ArkGT> = output_columns
        .iter()
        .map(|column| context.commit_column(column))
        .collect::<io::Result<_>>()?;

    bind_statement(
        transcript,
        &views,
        &output_commitments,
        output_len,
        log_domain,
    );

    let mut relocated_columns = Vec::with_capacity(inputs.len());
    let mut relocated_commitments = Vec::with_capacity(inputs.len());
    let mut relocations = Vec::with_capacity(inputs.len());
    for input in inputs {
        let mut branch_columns = Vec::with_capacity(arity);
        let mut branch_commitments = Vec::with_capacity(arity);
        let mut branch_relocations = Vec::with_capacity(arity);
        for (column, source_commitment) in input.columns.iter().zip(&input.commitments) {
            let target = zero_pad(&column[..input.n], domain);
            let target_commitment = context.commit_column(&target)?;
            branch_relocations.push(prove_column_relocation_with_context(
                context, column, source_commitment, &target, &target_commitment, transcript,
            )?);
            branch_columns.push(target);
            branch_commitments.push(target_commitment);
        }
        relocated_columns.push(branch_columns);
        relocated_commitments.push(branch_commitments);
        relocations.push(branch_relocations);
    }
    bind_relocated(transcript, &relocated_commitments);
    let beta = transcript.challenge_scalar(b"pow/union/beta");
    let alpha = transcript.challenge_scalar(b"pow/union/alpha");

    let mut reciprocals = Vec::with_capacity(inputs.len() + 1);
    for columns in &relocated_columns {
        let slices: Vec<&[ArkFr]> = columns.iter().map(Vec::as_slice).collect();
        reciprocals.push(reciprocal_witness(alpha, &rlc_fold(beta, &slices)));
    }
    let output_slices: Vec<&[ArkFr]> = output_columns.iter().map(Vec::as_slice).collect();
    reciprocals.push(reciprocal_witness(alpha, &rlc_fold(beta, &output_slices)));
    let reciprocal_commitments: Vec<ArkGT> = reciprocals
        .iter()
        .map(|column| context.commit_column(column))
        .collect::<io::Result<_>>()?;
    for commitment in &reciprocal_commitments {
        transcript.append_group(b"pow/union/reciprocal-com", commitment);
    }

    let branch_lengths: Vec<usize> = views.iter().map(|input| input.n).collect();
    let shape = union_shape(
        log_domain,
        &branch_lengths,
        output_len,
        arity,
        alpha,
        beta,
    );
    let mut columns = Vec::with_capacity(shape.num_columns);
    for branch in &relocated_columns {
        columns.extend(branch.iter().cloned());
    }
    columns.extend(output_columns.iter().cloned());
    columns.extend(reciprocals);
    let (sumcheck, point, _) = prove_sumcheck(&shape, &columns, transcript)?;
    let opening = context.prove_column_openings(&columns, &point, transcript)?;

    Ok(BagUnionResult {
        proof: BagUnionProof {
            relocations,
            relocated_commitments,
            reciprocal_commitments,
            sumcheck,
            opening,
        },
        commitments: output_commitments,
        columns: output_columns,
        log_domain,
        n: output_len,
    })
}

/// Verify an exact bag union against branch result commitments and pinned lengths.
pub fn verify_bag_union(
    inputs: &[RelationCommit],
    output_commitments: &[ArkGT],
    output_len: usize,
    output_log_domain: usize,
    proof: &BagUnionProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    let arity = validate_inputs(inputs)?;
    if output_commitments.len() != arity {
        return Err(invalid("bag-union output arity mismatch"));
    }
    let expected_len = checked_output_len(inputs)?;
    if output_len != expected_len {
        return Err(invalid("bag-union output length is not the branch-length sum"));
    }
    let expected_domain = padded_len(output_len)?;
    let expected_log_domain = expected_domain.trailing_zeros() as usize;
    if output_log_domain != expected_log_domain {
        return Err(invalid("bag-union output domain mismatch"));
    }
    if proof.relocations.len() != inputs.len()
        || proof.relocated_commitments.len() != inputs.len()
        || proof.reciprocal_commitments.len() != inputs.len() + 1
    {
        return Err(invalid("bag-union proof branch shape mismatch"));
    }

    bind_statement(
        transcript,
        inputs,
        output_commitments,
        output_len,
        output_log_domain,
    );
    for (branch, input) in inputs.iter().enumerate() {
        if proof.relocations[branch].len() != arity
            || proof.relocated_commitments[branch].len() != arity
        {
            return Err(invalid("bag-union relocation shape mismatch"));
        }
        for column in 0..arity {
            verify_column_relocation(
                &input.commitments[column],
                input.log_domain,
                &proof.relocated_commitments[branch][column],
                output_log_domain,
                &proof.relocations[branch][column],
                transcript,
            )?;
        }
    }
    bind_relocated(transcript, &proof.relocated_commitments);
    let beta = transcript.challenge_scalar(b"pow/union/beta");
    let alpha = transcript.challenge_scalar(b"pow/union/alpha");
    for commitment in &proof.reciprocal_commitments {
        transcript.append_group(b"pow/union/reciprocal-com", commitment);
    }

    if proof.sumcheck.round_polys.len() != output_log_domain {
        return Err(invalid("bag-union sumcheck domain mismatch"));
    }
    let branch_lengths: Vec<usize> = inputs.iter().map(|input| input.n).collect();
    let shape = union_shape(
        output_log_domain,
        &branch_lengths,
        output_len,
        arity,
        alpha,
        beta,
    );
    let (point, evals) = verify_sumcheck(&shape, &proof.sumcheck, transcript)?;
    let mut commitments = Vec::with_capacity(shape.num_columns);
    for branch in &proof.relocated_commitments {
        commitments.extend(branch.iter().copied());
    }
    commitments.extend(output_commitments.iter().copied());
    commitments.extend(proof.reciprocal_commitments.iter().copied());
    verify_column_openings(&commitments, &point, &evals, &proof.opening, transcript)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frs(values: &[u64]) -> Vec<ArkFr> {
        values.iter().map(|value| ArkFr::from_u64(*value)).collect()
    }

    fn relation(columns: &[&[u64]]) -> RelationWitness {
        let n = columns[0].len();
        RelationWitness::commit(columns.iter().map(|column| frs(column)).collect(), n).unwrap()
    }

    fn transcript() -> Transcript254 {
        Transcript254::new(b"parture-bag-union-test")
    }

    #[test]
    fn exact_bag_union_preserves_overlap_and_tuple_correlation() {
        let inputs = vec![
            relation(&[&[1, 2], &[10, 20]]),
            relation(&[&[2, 3], &[20, 30]]),
            relation(&[&[4], &[40]]),
        ];
        let views: Vec<RelationCommit> = inputs.iter().map(RelationWitness::view).collect();
        let mut pt = transcript();
        let result = prove_bag_union(&inputs, &mut pt).unwrap();
        assert_eq!(result.n, 5);
        assert_eq!(&result.columns[0][..5], frs(&[1, 2, 2, 3, 4]));
        assert_eq!(&result.columns[1][..5], frs(&[10, 20, 20, 30, 40]));
        let mut vt = transcript();
        verify_bag_union(
            &views,
            &result.commitments,
            result.n,
            result.log_domain,
            &result.proof,
            &mut vt,
        )
        .unwrap();

        for forged_len in [4, 6] {
            let mut vt = transcript();
            assert!(verify_bag_union(
                &views,
                &result.commitments,
                forged_len,
                result.log_domain,
                &result.proof,
                &mut vt,
            )
            .is_err());
        }

        // Column splicing destroys row fingerprints even though each individual value
        // still belongs to one of the input columns.
        let mut spliced = result.columns.clone();
        spliced[1].swap(0, 1);
        let spliced_commitments: Vec<ArkGT> = spliced
            .iter()
            .map(|column| commit_column(column).unwrap())
            .collect();
        let mut vt = transcript();
        assert!(verify_bag_union(
            &views,
            &spliced_commitments,
            result.n,
            result.log_domain,
            &result.proof,
            &mut vt,
        )
        .is_err());

        let mut reordered = views.clone();
        reordered.swap(0, 1);
        let mut vt = transcript();
        assert!(verify_bag_union(
            &reordered,
            &result.commitments,
            result.n,
            result.log_domain,
            &result.proof,
            &mut vt,
        )
        .is_err());
    }

    #[test]
    fn empty_branch_contributes_zero_rows() {
        let inputs = vec![relation(&[&[1], &[10]]), relation(&[&[], &[]])];
        let views: Vec<RelationCommit> = inputs.iter().map(RelationWitness::view).collect();
        let mut pt = transcript();
        let result = prove_bag_union(&inputs, &mut pt).unwrap();
        assert_eq!(result.n, 1);
        let mut vt = transcript();
        verify_bag_union(
            &views,
            &result.commitments,
            result.n,
            result.log_domain,
            &result.proof,
            &mut vt,
        )
        .unwrap();
    }
}
