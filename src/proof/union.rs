//! Exact bag union over authenticated relation results.
//!
//! Every branch is independently committed and carries a verifier-pinned real length.
//! Branch columns are relocation-bound to one common domain, rows are fingerprinted by
//! a transcript-challenged RLC, and one GKR fraction tree ([`super::logup_gkr`]) proves
//! that the output multiset is exactly the sum of the branch multisets — with NO
//! reciprocal column committed anywhere (the old per-branch/output `reciprocal_witness`
//! columns and their masked LogUp link are gone). The public output length is
//! independently required to equal the checked sum of branch lengths, preserving overlap
//! multiplicity.
//!
//! # Multi-block leaf layout (LogUp-GKR migration, Inc 4)
//! The `N` input branches and the output are each RELOCATED into their OWN length-`domain`
//! column set (real rows at `[0, n_b)`, zero-padded), so there is NO single combined input
//! column space — genuinely `N + 1` fraction groups. The fraction leaf vector therefore
//! TILES `G = N + 1` blocks (padded to `G_pad = 2^high_bits` blocks) of `domain = 2^k`
//! rows each into one tree of `num_vars = k + high_bits` variables: block `b < N` is branch
//! `b` at leaves `[b·domain, (b+1)·domain)`, block `N` is the output, and blocks
//! `[N+1, G_pad)` are padding (`p = 0`, self-annihilating). The low `k` bits index within a
//! block, the top `high_bits` select the block.
//!
//! The committed columns stay length `domain` (their commitments are UNCHANGED —
//! `commitment_root` is unaffected; only proof bytes change). The verifier opens each
//! length-`domain` committed column at the low `k` bits `r_low` of the GKR leaf point and
//! reconstructs its `2^num_vars` leaf-MLE via the PUBLIC block selector
//! `eq_at_index(b, r_high)` (the multilinear indicator `[block == b]` over the high bits —
//! the multi-bit generalization of `permutation`/`join_link`'s single top-bit
//! `(1 - r_high)` / `r_high` factor). This STRUCTURALLY forces block `b`'s fold columns to
//! zero in every other block (the honest length-`domain` opening tensor-scaled by
//! `eq_at_index(b, r_high)` cannot leak into another block), so no cross-branch row can be
//! injected/swapped without the leaf tie disagreeing with the tree's reduced claim.
//!
//! The blocks are combined by an ADDITIVE public block-indicator numerator: each branch's
//! real rows count `+1` (as `+([i < b·domain + n_b] - [i < b·domain])`) and the output's
//! real rows count `-1`; every padding row is `0`. All numerators are PUBLIC prefix
//! indicators (there is NO `committed × public-Chi` mask — no GAP-1, no auxiliary Identity),
//! so within-block and padding-block rows self-annihilate and no reciprocal is reintroduced.
//! At random `alpha` (drawn AFTER every relocated/output column commitment is bound), the
//! fraction sum is zero iff the `beta`-fingerprint multiset of the output real rows equals
//! the disjoint union of the branches' real-row multisets, i.e. iff the output is the exact
//! bag union (duplicates and tuple correlation included).
//!
//! For a developer: this is the multiset (bag) union [gadget](crate::proof::primer) — the store
//! side of WOQL `Or`. "Output = disjoint union of the branches" is proved as one
//! [LogUp](crate::proof::primer) rational-sum identity `Σ_branch Σ 1/(alpha - branch_fold) - Σ
//! 1/(alpha - out_fold) = 0`, batched through a [LogUp-GKR](crate::proof::primer) fraction tree
//! (§7); branches are first tied to one domain with the [relocation gadget](super::relocate).
//! `beta`/`alpha` are Fiat–Shamir, drawn after all commitments are bound ([the transcript
//! invariant](crate::proof::primer), §8). See also [gadgets](crate::proof::primer) (§6).

use std::io;

use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::transcript::Transcript;

use super::argument::{
    mle_full_eval, powers, verify_column_openings, ColumnOpening, StoreProverContext, Transcript254,
};
#[cfg(test)]
use super::argument::commit_column;
use super::bgp::{RelationCommit, RelationWitness};
use super::invalid;
use super::logup_gkr::{
    bind_and_verify_fraction_tree, check_leaf_tie, eq_at_index, prove_fraction_sum_zero,
    FractionInstance, GkrFractionProof, LinearForm, LinearTerm, Src,
};
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
    /// Commitments consumed by the union fraction tree after relocation.
    pub relocated_commitments: Vec<Vec<ArkGT>>,
    /// The multi-block GKR fraction-sum-zero proof for
    /// `Σ_branch Σ_i [i<n_b]/(alpha - branch_fold_i) - Σ_j [j<out_len]/(alpha - out_fold_j) = 0`.
    pub gkr: GkrFractionProof,
    /// Evaluations of the `(N+1)·arity` committed columns (branch columns in branch order,
    /// then the output columns) at `r_low` (the low `log_domain` bits of the GKR leaf point).
    pub col_evals: Vec<ArkFr>,
    /// One batched Dory opening binding `col_evals` to those committed columns at `r_low`.
    pub opening: ColumnOpening,
}

/// Prover-side exact union result. Rows are emitted by branch order, while verification
/// intentionally authenticates bag equality and therefore permits result reordering.
pub struct BagUnionResult {
    /// The bag-union proof.
    pub proof: BagUnionProof,
    /// Dory commitments to the output columns.
    pub commitments: Vec<ArkGT>,
    /// The concatenated output columns (all branches' real rows, zero-padded to `2^log_domain`).
    pub columns: Vec<Vec<ArkFr>>,
    /// The output columns' power-of-two log-domain.
    pub log_domain: usize,
    /// The output row count (the sum of the branch real lengths).
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

/// The output relation's `log_domain` (the padded power-of-two width covering `output_len`).
/// Prover and verifier derive it identically; recursive consumers use it to pin the union
/// output's committed-column length.
pub(crate) fn output_log_domain(output_len: usize) -> io::Result<usize> {
    Ok(padded_len(output_len)?.trailing_zeros() as usize)
}

/// The PUBLIC multi-block fraction-sum-zero instance for a bag union of `N` branches (real
/// lengths `branch_lengths`) into an `output_len`-row output, over `arity` columns at
/// `2^log_domain` per block. Built IDENTICALLY by prover and verifier from public inputs.
///
/// Column layout (`num_cols = (N+1)·arity`): branch `b`'s `arity` fold columns at instance
/// indices `[b·arity, (b+1)·arity)`, the output columns at `[N·arity, (N+1)·arity)`. Each
/// column is block-placed (nonzero only in its own leaf block), so the denominator
/// `q_leaf = alpha - Σ_{block b} Σ_d beta^d·col_{b,d}` reduces, on the leaves of block `b`,
/// to `alpha - Σ_d beta^d·col_{b,d}` (= `alpha - branch/out fold`), and to the constant
/// `alpha` on padding blocks. The numerator `p_leaf` sums the additive public block
/// indicators `+([i < b·domain + n_b] - [i < b·domain])` over the branches and
/// `-([i < N·domain + out_len] - [i < N·domain])` for the output — no committed multiplicity,
/// so within-block/padding rows self-annihilate (`p = 0`).
fn union_fraction_instance(
    alpha: ArkFr,
    beta: ArkFr,
    branch_lengths: &[usize],
    output_len: usize,
    arity: usize,
    log_domain: usize,
) -> io::Result<FractionInstance> {
    let branches = branch_lengths.len();
    let groups = branches + 1;
    let group_pad = groups
        .checked_next_power_of_two()
        .ok_or_else(|| invalid("bag-union group count overflow"))?;
    let high_bits = group_pad.trailing_zeros() as usize;
    let num_vars = log_domain
        .checked_add(high_bits)
        .filter(|nv| *nv < usize::BITS as usize)
        .ok_or_else(|| invalid("bag-union leaf domain overflow"))?;
    let domain = 1usize << log_domain;
    let num_cols = groups * arity;
    let beta_powers = powers(beta, arity);

    // q_leaf = alpha - Σ_{block b} Σ_d beta^d·col_{b,d} (blocks disjoint).
    let mut q_terms = Vec::with_capacity(num_cols);
    for block in 0..groups {
        for (d, power) in beta_powers.iter().enumerate() {
            q_terms.push(LinearTerm {
                coeff: f0() - *power,
                src: Src::Column(block * arity + d),
            });
        }
    }

    // p_leaf: each branch's real rows +1, the output's real rows -1, everything else 0.
    let mut p_terms = Vec::with_capacity(2 * groups);
    for (b, &n_b) in branch_lengths.iter().enumerate() {
        let base = b * domain;
        p_terms.push(LinearTerm {
            coeff: f1(),
            src: Src::Prefix(base + n_b),
        });
        p_terms.push(LinearTerm {
            coeff: f0() - f1(),
            src: Src::Prefix(base),
        });
    }
    let out_base = branches * domain;
    p_terms.push(LinearTerm {
        coeff: f0() - f1(),
        src: Src::Prefix(out_base + output_len),
    });
    p_terms.push(LinearTerm {
        coeff: f1(),
        src: Src::Prefix(out_base),
    });

    Ok(FractionInstance {
        num_vars,
        num_cols,
        p_form: LinearForm {
            constant: f0(),
            terms: p_terms,
        },
        q_form: LinearForm {
            constant: alpha,
            terms: q_terms,
        },
    })
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
    // Bind BOTH sides' commitments (output in `bind_statement`, relocated here), THEN draw
    // beta/alpha. Caller contract (Inc 0 audit): alpha MUST be drawn after every data-column
    // commitment so the fraction leaves are fixed before any per-layer GKR challenge. There is
    // no reciprocal commitment anymore, so this is the last binding before the folds.
    bind_relocated(transcript, &relocated_commitments);
    let beta = transcript.challenge_scalar(b"pow/union/beta");
    let alpha = transcript.challenge_scalar(b"pow/union/alpha");

    let branch_lengths: Vec<usize> = views.iter().map(|input| input.n).collect();
    let inst = union_fraction_instance(alpha, beta, &branch_lengths, output_len, arity, log_domain)?;
    let leaf_count = inst.size();

    // Block-disjoint zero-extension into the `leaf_count`-row leaf vector: branch `b`'s
    // columns fill block `b`, the output columns fill block `N`; padding blocks stay zero.
    let mut extended: Vec<Vec<ArkFr>> = Vec::with_capacity(inst.num_cols);
    for (branch_index, branch) in relocated_columns.iter().enumerate() {
        let base = branch_index * domain;
        for column in branch {
            let mut ext = vec![f0(); leaf_count];
            ext[base..base + domain].copy_from_slice(column);
            extended.push(ext);
        }
    }
    let output_base = inputs.len() * domain;
    for column in &output_columns {
        let mut ext = vec![f0(); leaf_count];
        ext[output_base..output_base + domain].copy_from_slice(column);
        extended.push(ext);
    }

    let (gkr, r_leaf) = prove_fraction_sum_zero(&inst, &extended, transcript)?;

    // Single batched Dory opening of the length-`domain` committed columns (branch columns in
    // branch order, then the output columns) at `r_low`, bound to their commitments here.
    let r_low = &r_leaf[..log_domain];
    let mut columns: Vec<Vec<ArkFr>> = Vec::with_capacity(inst.num_cols);
    for branch in &relocated_columns {
        columns.extend(branch.iter().cloned());
    }
    columns.extend(output_columns.iter().cloned());
    let col_evals: Vec<ArkFr> = columns.iter().map(|c| mle_full_eval(c, r_low)).collect();
    let opening = context.prove_column_openings(&columns, r_low, transcript)?;

    Ok(BagUnionResult {
        proof: BagUnionProof {
            relocations,
            relocated_commitments,
            gkr,
            col_evals,
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
        || proof.col_evals.len() != (inputs.len() + 1) * arity
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
    // Same binding order as the prover: both sides' commitments, THEN beta/alpha.
    bind_relocated(transcript, &proof.relocated_commitments);
    let beta = transcript.challenge_scalar(b"pow/union/beta");
    let alpha = transcript.challenge_scalar(b"pow/union/alpha");

    // Bind the public instance and verify the multi-block GKR tree -> leaf point + claims.
    let branch_lengths: Vec<usize> = inputs.iter().map(|input| input.n).collect();
    let inst = union_fraction_instance(alpha, beta, &branch_lengths, output_len, arity, output_log_domain)?;
    if proof.gkr.num_vars() != inst.num_vars {
        return Err(invalid("bag-union fraction layer count mismatch"));
    }
    let (r_leaf, p_claim, q_claim) = bind_and_verify_fraction_tree(&inst, &proof.gkr, transcript)?;

    // Open the length-`domain` committed columns (branch columns in branch order, then the
    // output columns) at `r_low`, bound to their commitments on this transcript.
    let r_low = &r_leaf[..output_log_domain];
    let r_high = &r_leaf[output_log_domain..];
    let mut commitments = Vec::with_capacity(inst.num_cols);
    for branch in &proof.relocated_commitments {
        commitments.extend(branch.iter().copied());
    }
    commitments.extend(output_commitments.iter().copied());
    verify_column_openings(&commitments, r_low, &proof.col_evals, &proof.opening, transcript)?;

    // Reconstruct each block's leaf-MLE from its length-`domain` opening via the PUBLIC block
    // selector `eq_at_index(block, r_high)` (the multilinear `[block == b]` indicator over the
    // high bits). Block `b`'s columns are tensor-scaled into block `b` only, STRUCTURALLY
    // forcing them to zero in every other block (no cross-branch leak).
    let groups = inputs.len() + 1;
    let mut leaf_evals = Vec::with_capacity(inst.num_cols);
    for block in 0..groups {
        let selector = eq_at_index(block, r_high);
        for column in 0..arity {
            leaf_evals.push(selector * proof.col_evals[block * arity + column]);
        }
    }
    check_leaf_tie(&inst, &leaf_evals, p_claim, q_claim, &r_leaf)
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

    // Honest four-branch roundtrip: groups = 5 -> group_pad = 8 (three PADDING blocks), so it
    // exercises the multi-block tree with more selector bits and self-annihilating pad blocks.
    #[test]
    fn four_branch_union_with_padding_blocks_roundtrips() {
        let inputs = vec![
            relation(&[&[1, 5], &[10, 50]]),
            relation(&[&[2], &[20]]),
            relation(&[&[3, 7], &[30, 70]]),
            relation(&[&[4], &[40]]),
        ];
        let views: Vec<RelationCommit> = inputs.iter().map(RelationWitness::view).collect();
        let mut pt = transcript();
        let result = prove_bag_union(&inputs, &mut pt).unwrap();
        assert_eq!(result.n, 6);
        assert_eq!(&result.columns[0][..6], frs(&[1, 5, 2, 3, 7, 4]));
        assert_eq!(&result.columns[1][..6], frs(&[10, 50, 20, 30, 70, 40]));
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

    // Drive the honest prover flow with an EXPLICIT (possibly forged) output relation. Mirrors
    // the real prover body (honest branch relocations, then an arbitrary output multiset), so a
    // test can supply an output whose bag differs from the branch bag-union and check that the
    // new fraction tree rejects it. Returns the proof, output commitments, and public shape.
    fn prove_with_explicit_output(
        inputs: &[RelationWitness],
        output_real: &[Vec<ArkFr>],
    ) -> (BagUnionProof, Vec<ArkGT>, usize, usize) {
        let context = StoreProverContext::cpu();
        let views: Vec<RelationCommit> = inputs.iter().map(RelationWitness::view).collect();
        let arity = validate_inputs(&views).unwrap();
        let output_len = checked_output_len(&views).unwrap();
        let domain = padded_len(output_len).unwrap();
        let log_domain = domain.trailing_zeros() as usize;

        let mut output_columns: Vec<Vec<ArkFr>> = Vec::with_capacity(arity);
        for column in output_real {
            let mut padded = column.clone();
            padded.resize(domain, f0());
            output_columns.push(padded);
        }
        let output_commitments: Vec<ArkGT> = output_columns
            .iter()
            .map(|column| context.commit_column(column).unwrap())
            .collect();

        let mut t = transcript();
        bind_statement(&mut t, &views, &output_commitments, output_len, log_domain);

        let mut relocated_columns = Vec::with_capacity(inputs.len());
        let mut relocated_commitments = Vec::with_capacity(inputs.len());
        let mut relocations = Vec::with_capacity(inputs.len());
        for input in inputs {
            let mut branch_columns = Vec::with_capacity(arity);
            let mut branch_commitments = Vec::with_capacity(arity);
            let mut branch_relocations = Vec::with_capacity(arity);
            for (column, source_commitment) in input.columns.iter().zip(&input.commitments) {
                let target = zero_pad(&column[..input.n], domain);
                let target_commitment = context.commit_column(&target).unwrap();
                branch_relocations.push(
                    prove_column_relocation_with_context(
                        &context, column, source_commitment, &target, &target_commitment, &mut t,
                    )
                    .unwrap(),
                );
                branch_columns.push(target);
                branch_commitments.push(target_commitment);
            }
            relocated_columns.push(branch_columns);
            relocated_commitments.push(branch_commitments);
            relocations.push(branch_relocations);
        }
        bind_relocated(&mut t, &relocated_commitments);
        let beta = t.challenge_scalar(b"pow/union/beta");
        let alpha = t.challenge_scalar(b"pow/union/alpha");

        let branch_lengths: Vec<usize> = views.iter().map(|input| input.n).collect();
        let inst =
            union_fraction_instance(alpha, beta, &branch_lengths, output_len, arity, log_domain)
                .unwrap();
        let leaf_count = inst.size();
        let mut extended: Vec<Vec<ArkFr>> = Vec::with_capacity(inst.num_cols);
        for (branch_index, branch) in relocated_columns.iter().enumerate() {
            let base = branch_index * domain;
            for column in branch {
                let mut ext = vec![f0(); leaf_count];
                ext[base..base + domain].copy_from_slice(column);
                extended.push(ext);
            }
        }
        let output_base = inputs.len() * domain;
        for column in &output_columns {
            let mut ext = vec![f0(); leaf_count];
            ext[output_base..output_base + domain].copy_from_slice(column);
            extended.push(ext);
        }
        let (gkr, r_leaf) = prove_fraction_sum_zero(&inst, &extended, &mut t).unwrap();
        let r_low = &r_leaf[..log_domain];
        let mut columns: Vec<Vec<ArkFr>> = Vec::new();
        for branch in &relocated_columns {
            columns.extend(branch.iter().cloned());
        }
        columns.extend(output_columns.iter().cloned());
        let col_evals: Vec<ArkFr> = columns.iter().map(|c| mle_full_eval(c, r_low)).collect();
        let opening = context.prove_column_openings(&columns, r_low, &mut t).unwrap();

        (
            BagUnionProof {
                relocations,
                relocated_commitments,
                gkr,
                col_evals,
                opening,
            },
            output_commitments,
            output_len,
            log_domain,
        )
    }

    // A forged output that DROPS a branch row (and duplicates another to keep the length) has a
    // different bag, so the fraction sum is nonzero (root P != 0) and verification rejects.
    #[test]
    fn forged_dropped_row_output_is_rejected() {
        let inputs = vec![relation(&[&[1, 2], &[10, 20]]), relation(&[&[3], &[30]])];
        let views: Vec<RelationCommit> = inputs.iter().map(RelationWitness::view).collect();
        // Honest bag = {(1,10),(2,20),(3,30)}; forge to {(1,10),(1,10),(2,20)} (drop (3,30)).
        let forged = vec![frs(&[1, 1, 2]), frs(&[10, 10, 20])];
        let (proof, output_commitments, output_len, log_domain) =
            prove_with_explicit_output(&inputs, &forged);
        assert!(!proof.gkr.root_p.is_zero(), "dropped-row bag should not sum to zero");
        let mut vt = transcript();
        assert!(verify_bag_union(
            &views,
            &output_commitments,
            output_len,
            log_domain,
            &proof,
            &mut vt,
        )
        .is_err());
    }

    // A forged output that INJECTS a fresh row not present in any branch (and drops a real one to
    // keep the length) is rejected: the injected fold value leaves an uncancelled pole.
    #[test]
    fn forged_injected_row_output_is_rejected() {
        let inputs = vec![relation(&[&[1, 2], &[10, 20]]), relation(&[&[3], &[30]])];
        let views: Vec<RelationCommit> = inputs.iter().map(RelationWitness::view).collect();
        // Inject (9,99), drop (2,20): Sum = 1/(a-f(2,20)) - 1/(a-f(9,99)) != 0.
        let forged = vec![frs(&[1, 9, 3]), frs(&[10, 99, 30])];
        let (proof, output_commitments, output_len, log_domain) =
            prove_with_explicit_output(&inputs, &forged);
        assert!(!proof.gkr.root_p.is_zero());
        let mut vt = transcript();
        assert!(verify_bag_union(
            &views,
            &output_commitments,
            output_len,
            log_domain,
            &proof,
            &mut vt,
        )
        .is_err());
    }

    // A CROSS-BRANCH swapped row — one output row mixes a subject from branch 0 with an object
    // from branch 1, a tuple present in NO branch — breaks tuple correlation and is rejected. The
    // block-disjoint reconstruction means neither branch can "lend" its columns across blocks.
    #[test]
    fn forged_cross_branch_swapped_row_is_rejected() {
        let inputs = vec![relation(&[&[1, 2], &[10, 20]]), relation(&[&[3], &[30]])];
        let views: Vec<RelationCommit> = inputs.iter().map(RelationWitness::view).collect();
        // Honest bag = {(1,10),(2,20),(3,30)}; forge (3,30) -> (1,30): subject from branch 0,
        // object from branch 1. Sum = 1/(a-f(3,30)) - 1/(a-f(1,30)) != 0.
        let forged = vec![frs(&[1, 2, 1]), frs(&[10, 20, 30])];
        let (proof, output_commitments, output_len, log_domain) =
            prove_with_explicit_output(&inputs, &forged);
        assert!(!proof.gkr.root_p.is_zero());
        let mut vt = transcript();
        assert!(verify_bag_union(
            &views,
            &output_commitments,
            output_len,
            log_domain,
            &proof,
            &mut vt,
        )
        .is_err());
    }

    // A forged output that DUPLICATES a branch row beyond its multiplicity (dropping a distinct
    // one to keep the length) is rejected — overlap multiplicity is preserved exactly.
    #[test]
    fn forged_duplicated_row_output_is_rejected() {
        let inputs = vec![relation(&[&[1, 2], &[10, 20]]), relation(&[&[3], &[30]])];
        let views: Vec<RelationCommit> = inputs.iter().map(RelationWitness::view).collect();
        // Honest bag = {(1,10),(2,20),(3,30)}; forge to {(1,10),(2,20),(2,20)} (dup (2,20)).
        let forged = vec![frs(&[1, 2, 2]), frs(&[10, 20, 20])];
        let (proof, output_commitments, output_len, log_domain) =
            prove_with_explicit_output(&inputs, &forged);
        assert!(!proof.gkr.root_p.is_zero());
        let mut vt = transcript();
        assert!(verify_bag_union(
            &views,
            &output_commitments,
            output_len,
            log_domain,
            &proof,
            &mut vt,
        )
        .is_err());
    }

    // Tampering the GKR fraction root (P = 0 is the whole statement) is rejected immediately.
    #[test]
    fn tampered_gkr_root_is_rejected() {
        let inputs = vec![relation(&[&[1, 2], &[10, 20]]), relation(&[&[2, 3], &[20, 30]])];
        let views: Vec<RelationCommit> = inputs.iter().map(RelationWitness::view).collect();
        let mut pt = transcript();
        let mut result = prove_bag_union(&inputs, &mut pt).unwrap();
        result.proof.gkr.root_p = result.proof.gkr.root_p + f1();
        let mut vt = transcript();
        assert!(verify_bag_union(
            &views,
            &result.commitments,
            result.n,
            result.log_domain,
            &result.proof,
            &mut vt,
        )
        .is_err());
    }

    // Tampering a claimed column evaluation is rejected by the Dory opening / leaf tie (the
    // opening binds `col_evals` to the committed columns at `r_low`).
    #[test]
    fn tampered_column_eval_is_rejected() {
        let inputs = vec![relation(&[&[1, 2], &[10, 20]]), relation(&[&[2, 3], &[20, 30]])];
        let views: Vec<RelationCommit> = inputs.iter().map(RelationWitness::view).collect();
        let mut pt = transcript();
        let mut result = prove_bag_union(&inputs, &mut pt).unwrap();
        result.proof.col_evals[0] = result.proof.col_evals[0] + f1();
        let mut vt = transcript();
        assert!(verify_bag_union(
            &views,
            &result.commitments,
            result.n,
            result.log_domain,
            &result.proof,
            &mut vt,
        )
        .is_err());
    }
}
