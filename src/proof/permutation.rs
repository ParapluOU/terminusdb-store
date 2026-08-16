//! Exact full-row permutation proof for authenticated relations.
//!
//! Input and output rows are fingerprinted across every column with a transcript-drawn
//! RLC challenge. A single GKR fraction tree ([`super::logup_gkr`]) proves the two row
//! multisets are equal via the log-derivative identity
//! `Sum_i 1/(alpha - in_fold_i) - Sum_j 1/(alpha - out_fold_j) = 0`, over the committed
//! input and output data columns, with NO reciprocal column committed anywhere (the old
//! `input_reciprocal`/`output_reciprocal` LogUp witnesses and their masked link are
//! gone). This gadget intentionally proves no ordering property: composing `OrderBy`
//! must additionally prove the requested ordering and, for SWI-compatible semantics,
//! stable order among equal keys.
//!
//! Block-disjoint leaf layout (LogUp-GKR migration, Inc 2). The fraction leaf vector has
//! length `2*domain` (the next power of two of the two length-`domain = 2^log_domain`
//! blocks): the input rows occupy the LOW half `[0, domain)` and the output rows the
//! HIGH half `[domain, 2*domain)`. The committed columns stay length `domain` (their
//! commitments are UNCHANGED — `commitment_root` is unaffected); the prover feeds the
//! tree their zero-extensions — input columns zero on the high half, output columns zero
//! on the low half. The verifier opens the length-`domain` committed columns at the low
//! `log_domain` bits `r_low` of the GKR leaf point and reconstructs each `2*domain`
//! leaf-MLE via the top-bit factor: `(1 - r_high)*in_col(r_low)` for an input column
//! (low half) and `r_high*out_col(r_low)` for an output column (high half). This
//! STRUCTURALLY forces the input columns to zero on the output block and vice-versa (the
//! same zero-extend enforcement `dict` uses for its target leaf): a prover cannot commit
//! nonzero data outside a block's own half, because the reconstruction from the honest
//! length-`domain` opening would disagree with the tree's leaf claim — the two sides
//! cannot leak into each other.
//!
//! The two leaf groups are combined by an ADDITIVE public block-indicator numerator
//! `p = +[i < n] - [domain <= i < domain+n]` (each side's real rows counted once, signed
//! +1 for input and -1 for output) — NOT a `committed x public` mask — so within-block
//! padding rows self-annihilate (`p = 0 => 0/q = 0`) and no reciprocal is reintroduced.
//! At random `alpha` (drawn AFTER the input/output commitments are bound), the identity
//! holds iff the `beta`-fingerprint multiset of the real output rows equals that of the
//! real input rows, i.e. iff the output is an exact full-row permutation (duplicates and
//! tuple correlation included).

use std::io;

use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::transcript::Transcript;

use super::argument::{
    mle_full_eval, powers, verify_column_openings, ColumnOpening, StoreProverContext, Transcript254,
};
use super::bgp::{RelationCommit, RelationWitness};
use super::invalid;
use super::logup_gkr::{
    bind_and_verify_fraction_tree, check_leaf_tie, prove_fraction_sum_zero, FractionInstance,
    GkrFractionProof, LinearForm, LinearTerm, Src,
};

const PERMUTATION_DOMAIN: &[u8] = b"parture/proof-of-woql/relation-permutation/v0";

#[inline]
fn f0() -> ArkFr {
    ArkFr::zero()
}
#[inline]
fn f1() -> ArkFr {
    ArkFr::one()
}

/// A full-row permutation proof: the GKR fraction-sum-zero proof for
/// `Sum 1/(alpha - in_fold) - Sum 1/(alpha - out_fold) = 0`, the claimed evaluations of
/// the `2*arity` committed data columns (input then output) at `r_low` (the low
/// `log_domain` bits of the GKR leaf point), and one batched Dory opening of those
/// columns at `r_low`. NO reciprocal (`input_reciprocal`/`output_reciprocal`) column
/// exists anymore.
#[derive(Clone, Debug)]
pub struct RelationPermutationProof {
    pub gkr: GkrFractionProof,
    pub col_evals: Vec<ArkFr>,
    pub opening: ColumnOpening,
}

/// Prover result: the proof plus the reordered output relation (its columns, their
/// commitments, the shared `log_domain`, and the real row count `n`).
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

/// The PUBLIC block-disjoint fraction-sum-zero instance for a permutation of `n` real
/// rows over `arity` columns at `2^log_domain`:
///   * committed columns `[0, arity)` are the input columns (low half `[0, domain)`),
///     columns `[arity, 2*arity)` the output columns (high half `[domain, 2*domain)`);
///   * denominator `q_leaf = alpha - Sum_d beta^d*(in_col_d + out_col_d)` — since the
///     blocks are disjoint, this is `alpha - in_fold` on the low half and
///     `alpha - out_fold` on the high half, with the SAME `beta^d` fingerprinting a
///     given column on both sides;
///   * numerator `p_leaf = +[i < n] - [domain <= i < domain+n]` (additive public block
///     indicators): the input real rows contribute `+1`, the output real rows `-1`, and
///     every padding row `0`.
/// The leaf count is `2*domain = 2^(log_domain+1)`. Built IDENTICALLY by prover and
/// verifier from public inputs only.
fn permutation_fraction_instance(
    alpha: ArkFr,
    beta: ArkFr,
    n: usize,
    log_domain: usize,
    arity: usize,
) -> FractionInstance {
    let domain = 1usize << log_domain;
    let beta_powers = powers(beta, arity);
    let mut q_terms = Vec::with_capacity(2 * arity);
    for (d, power) in beta_powers.iter().enumerate() {
        q_terms.push(LinearTerm {
            coeff: f0() - *power,
            src: Src::Column(d),
        });
    }
    for (d, power) in beta_powers.iter().enumerate() {
        q_terms.push(LinearTerm {
            coeff: f0() - *power,
            src: Src::Column(arity + d),
        });
    }
    // p_leaf = +[i<n] - ([i<domain+n] - [i<domain]): +1 on the input rows [0,n) and
    // -1 on the output rows [domain,domain+n); every other (padding) leaf gets 0.
    let p_terms = vec![
        LinearTerm {
            coeff: f1(),
            src: Src::Prefix(n),
        },
        LinearTerm {
            coeff: f0() - f1(),
            src: Src::Prefix(domain + n),
        },
        LinearTerm {
            coeff: f1(),
            src: Src::Prefix(domain),
        },
    ];
    FractionInstance {
        num_vars: log_domain + 1,
        num_cols: 2 * arity,
        p_form: LinearForm {
            constant: f0(),
            terms: p_terms,
        },
        q_form: LinearForm {
            constant: alpha,
            terms: q_terms,
        },
    }
}

/// Prove the output relation obtained by applying `row_order` (a permutation of the real
/// input rows) is an exact full-row permutation of `input`: same rows with the same
/// multiplicities, columns kept tuple-correlated. Returns the reordered output and the
/// proof. No ordering property is claimed (that is `OrderBy`'s job).
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

/// As [`prove_relation_permutation`], with an explicit prover-arithmetic context.
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
    // Bind BOTH sides' commitments, THEN draw beta/alpha. Caller contract (Inc 0
    // audit): alpha MUST be drawn after the column commitments are bound so the
    // fraction leaves are fixed before any per-layer GKR challenge. There is no
    // reciprocal commitment anymore, so this is the last binding before the folds.
    bind_statement(transcript, &input_view, &output_commitments);
    let beta = transcript.challenge_scalar(b"pow/permutation/beta");
    let alpha = transcript.challenge_scalar(b"pow/permutation/alpha");

    // Block-disjoint zero-extension to the 2*domain leaf vector: input columns fill the
    // low half, output columns the high half; the rest is zero.
    let leaf_count = 2 * domain;
    let mut extended: Vec<Vec<ArkFr>> = Vec::with_capacity(2 * arity);
    for column in &input.columns {
        let mut ext = vec![ArkFr::zero(); leaf_count];
        ext[..domain].copy_from_slice(column);
        extended.push(ext);
    }
    for column in &output_columns {
        let mut ext = vec![ArkFr::zero(); leaf_count];
        ext[domain..].copy_from_slice(column);
        extended.push(ext);
    }

    let inst = permutation_fraction_instance(alpha, beta, input.n, input.log_domain, arity);
    let (gkr, r_leaf) = prove_fraction_sum_zero(&inst, &extended, transcript)?;

    // Single batched Dory opening of the length-`domain` data columns at `r_low`, bound
    // to their commitments on this transcript.
    let r_low = &r_leaf[..input.log_domain];
    let mut columns = Vec::with_capacity(2 * arity);
    columns.extend(input.columns.iter().cloned());
    columns.extend(output_columns.iter().cloned());
    let col_evals: Vec<ArkFr> = columns.iter().map(|c| mle_full_eval(c, r_low)).collect();
    let opening = context.prove_column_openings(&columns, r_low, transcript)?;

    Ok(RelationPermutationResult {
        proof: RelationPermutationProof {
            gkr,
            col_evals,
            opening,
        },
        commitments: output_commitments,
        columns: output_columns,
        log_domain: input.log_domain,
        n: input.n,
    })
}

/// Verify a full-row permutation proof: the relation committed by `output_commitments`
/// (with pinned length/domain) is exactly the row multiset of `input`.
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
    if proof.gkr.num_vars() != input.log_domain + 1 {
        return Err(invalid("relation permutation fraction layer count mismatch"));
    }
    if proof.col_evals.len() != 2 * arity {
        return Err(invalid("relation permutation column-evaluation count mismatch"));
    }
    // Same binding order as the prover: BOTH sides' commitments, THEN beta/alpha.
    bind_statement(transcript, input, output_commitments);
    let beta = transcript.challenge_scalar(b"pow/permutation/beta");
    let alpha = transcript.challenge_scalar(b"pow/permutation/alpha");

    // Bind the public instance and verify the GKR tree -> the leaf point + claims.
    let inst = permutation_fraction_instance(alpha, beta, input.n, input.log_domain, arity);
    let (r_leaf, p_claim, q_claim) = bind_and_verify_fraction_tree(&inst, &proof.gkr, transcript)?;

    // Open the length-`domain` committed columns at `r_low`, bound to their commitments.
    let r_low = &r_leaf[..input.log_domain];
    let r_high = r_leaf[input.log_domain];
    let mut commitments = Vec::with_capacity(2 * arity);
    commitments.extend(input.commitments.iter().copied());
    commitments.extend(output_commitments.iter().copied());
    verify_column_openings(&commitments, r_low, &proof.col_evals, &proof.opening, transcript)?;

    // Reconstruct each `2*domain` leaf-MLE from the length-`domain` opening via the
    // top-bit zero-extension factor: input columns live in the LOW half (factor
    // `1 - r_high`), output columns in the HIGH half (factor `r_high`). This is the
    // block-disjoint enforcement: the input columns reconstruct to zero on the output
    // block and vice-versa (a prover cannot commit nonzero data in the other half —
    // the honest opening zero-extends), so neither side can leak into the other.
    let one_minus_high = f1() - r_high;
    let mut leaf_evals = Vec::with_capacity(2 * arity);
    for eval in &proof.col_evals[..arity] {
        leaf_evals.push(one_minus_high * *eval);
    }
    for eval in &proof.col_evals[arity..] {
        leaf_evals.push(r_high * *eval);
    }
    check_leaf_tie(&inst, &leaf_evals, p_claim, q_claim, &r_leaf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::argument::{commit_column, prove_column_openings};

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

    // Tampering the GKR fraction proof (root numerator) is rejected: the root claim
    // `P = 0` is the whole statement, so a nonzero `root_p` fails immediately.
    #[test]
    fn tampered_gkr_root_is_rejected() {
        let input = relation();
        let mut prover = transcript();
        let mut result = prove_relation_permutation(&input, &[2, 1, 0], &mut prover).unwrap();
        result.proof.gkr.root_p = result.proof.gkr.root_p + f1();
        let mut verifier = transcript();
        assert!(verify_relation_permutation(
            &input.view(),
            &result.commitments,
            result.n,
            result.log_domain,
            &result.proof,
            &mut verifier,
        )
        .is_err());
    }

    // Tampering a claimed column evaluation is rejected by the Dory opening / leaf tie:
    // the opening binds `col_evals` to the committed columns at `r_low`.
    #[test]
    fn tampered_column_eval_is_rejected() {
        let input = relation();
        let mut prover = transcript();
        let mut result = prove_relation_permutation(&input, &[2, 1, 0], &mut prover).unwrap();
        result.proof.col_evals[0] = result.proof.col_evals[0] + f1();
        let mut verifier = transcript();
        assert!(verify_relation_permutation(
            &input.view(),
            &result.commitments,
            result.n,
            result.log_domain,
            &result.proof,
            &mut verifier,
        )
        .is_err());
    }

    // Drive the honest prover flow with an EXPLICIT (possibly forged) output relation,
    // then verify. Mirrors the real prover body but takes arbitrary length-`n` output
    // columns so a test can supply a NON-permutation output multiset.
    fn prove_with_explicit_output(
        input: &RelationWitness,
        output_real: &[Vec<ArkFr>],
    ) -> (RelationPermutationProof, Vec<ArkGT>) {
        let arity = input.columns.len();
        let domain = 1usize << input.log_domain;
        let mut output_columns = Vec::with_capacity(arity);
        for column in output_real {
            let mut padded = column.clone();
            padded.resize(domain, ArkFr::zero());
            output_columns.push(padded);
        }
        let output_commitments: Vec<ArkGT> = output_columns
            .iter()
            .map(|column| commit_column(column).unwrap())
            .collect();
        let mut t = transcript();
        bind_statement(&mut t, &input.view(), &output_commitments);
        let beta = t.challenge_scalar(b"pow/permutation/beta");
        let alpha = t.challenge_scalar(b"pow/permutation/alpha");
        let leaf_count = 2 * domain;
        let mut extended: Vec<Vec<ArkFr>> = Vec::with_capacity(2 * arity);
        for column in &input.columns {
            let mut ext = vec![ArkFr::zero(); leaf_count];
            ext[..domain].copy_from_slice(column);
            extended.push(ext);
        }
        for column in &output_columns {
            let mut ext = vec![ArkFr::zero(); leaf_count];
            ext[domain..].copy_from_slice(column);
            extended.push(ext);
        }
        let inst = permutation_fraction_instance(alpha, beta, input.n, input.log_domain, arity);
        let (gkr, r_leaf) = prove_fraction_sum_zero(&inst, &extended, &mut t).unwrap();
        let r_low = &r_leaf[..input.log_domain];
        let mut columns = input.columns.clone();
        columns.extend(output_columns.iter().cloned());
        let col_evals: Vec<ArkFr> = columns.iter().map(|c| mle_full_eval(c, r_low)).collect();
        let opening = prove_column_openings(&columns, r_low, &mut t).unwrap();
        (
            RelationPermutationProof {
                gkr,
                col_evals,
                opening,
            },
            output_commitments,
        )
    }

    // A forged output whose row MULTISET differs from the input (one row dropped, a
    // fresh one added) yields a nonzero fraction sum, so the tree's root `P != 0` and
    // verification rejects. Covers the "added/dropped element" soundness case that a
    // valid `row_order` can never express.
    #[test]
    fn forged_non_permutation_output_is_rejected() {
        let input = relation(); // rows (2,20),(1,11),(1,10)
        // Forged output: keep (2,20),(1,11) but replace (1,10) with (9,99) — the
        // multiset now differs (one dropped, one added).
        let forged = vec![frs(&[2, 1, 9]), frs(&[20, 11, 99])];
        let (proof, output_commitments) = prove_with_explicit_output(&input, &forged);
        // The fraction sum is 1/(a-fold(1,10)) - 1/(a-fold(9,99)) != 0.
        assert!(!proof.gkr.root_p.is_zero(), "forged multiset should not sum to zero");
        let mut verifier = transcript();
        assert!(verify_relation_permutation(
            &input.view(),
            &output_commitments,
            input.n,
            input.log_domain,
            &proof,
            &mut verifier,
        )
        .is_err());
    }

    // A forged output that DUPLICATES one input row and DROPS another (same length,
    // still a non-permutation multiset) is rejected for the same reason.
    #[test]
    fn forged_duplicate_row_output_is_rejected() {
        let input = relation(); // rows (2,20),(1,11),(1,10)
        // (2,20) twice, (1,11) once, drop (1,10): Sum = 1/(a-f(2,20)) - 1/(a-f(1,10)) != 0.
        let forged = vec![frs(&[2, 2, 1]), frs(&[20, 20, 11])];
        let (proof, output_commitments) = prove_with_explicit_output(&input, &forged);
        assert!(!proof.gkr.root_p.is_zero());
        let mut verifier = transcript();
        assert!(verify_relation_permutation(
            &input.view(),
            &output_commitments,
            input.n,
            input.log_domain,
            &proof,
            &mut verifier,
        )
        .is_err());
    }
}
