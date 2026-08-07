//! Binary join of two DIFFERENT relations `R`, `S` on a single-column shared key
//! (Phase 7.2). Generalizes the self-join (`proof::pairs` + `proof::join`) to two
//! relations; the self-join is the `R = S` special case. Clean-room; reuses the
//! audited masked LogUp links (source AND universe masked) and the combined-index
//! monotonicity.
//!
//! Model. `R` has `n_R` rows `(r_key[a], r_other_0[a], ..)`; `S` has `n_S` rows
//! `(s_key[b], s_other_0[b], ..)`. The join output, for every `(a, b)` with
//! `r_key[a] == s_key[b]`, is `(r_other[a].., key, s_other[b]..)` with index pointers
//! `rho_l -> a` (into `R`) and `rho_r -> b` (into `S`).
//!
//! Argument (composed under one shared transcript, over a shared domain
//! `N = next_pow2(max(n_R, n_S, m_J, n_U))`):
//!   * distinct valid pairs: LEFT index-augmented membership of
//!     `(r_other_out, key, rho_l)` in `{(r_other, r_key, idx)}` (source masked by
//!     `chi_{n_R}`), RIGHT membership of `(key, s_other_out, rho_r)` in
//!     `{(s_key, s_other, idx)}` (source masked by `chi_{n_S}`). The SHARED committed
//!     `key` column links them (`key = r_key[rho_l] = s_key[rho_r]`, the join
//!     condition). Combined-index `i_eval = C*rho_l + rho_r` with
//!     `C = next_pow2(max(n_R, n_S))` is strictly increasing => distinct pairs.
//!     **Radix (the #1 audit item): `C >= max(n_R, n_S) >= n_S > rho_r`, so the map
//!     `(rho_l, rho_r) -> i_eval` is injective; `C = next_pow2(n_R)` is UNSOUND when
//!     `n_R < n_S`.**
//!   * `U` = sorted distinct union of `{r_key} ∪ {s_key}`; membership forces `r_key`
//!     to be the image of `U` under `w_L` and `s_key` the image under `w_R` (both
//!     sides masked); count `Σ chi_{n_U} w_L*w_R = m_J`.
//! Together: `m_J` distinct valid pairs AND `m_J` = total valid pairs => COMPLETE.

use ark_ff::PrimeField;
use std::io;

use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::{Field, Group};
use dory_pcs::primitives::transcript::Transcript;

use super::argument::{
    commit_column, prove_column_openings, prove_sumcheck, verify_column_openings, verify_sumcheck,
    ClaimShape, ColumnOpening, Expr, Operand, SumcheckClaim, SumcheckProof, SumcheckShape, Term,
    Transcript254,
};
use super::count::{prove_count_identity, verify_count_identity, CountProof};
use super::invalid;
use super::logup::{masked_multiset_link_claim, reciprocal_witness, rlc_fold};
use super::membership::{
    prove_membership_with_mult_with_witness, verify_membership_with_mult, MembershipProof,
};
use super::sort::{prove_strictly_sorted, verify_strictly_sorted, SortedProof};

#[inline]
fn f0() -> ArkFr {
    ArkFr::zero()
}
#[inline]
fn f1() -> ArkFr {
    ArkFr::one()
}
#[inline]
fn fr(v: usize) -> ArkFr {
    ArkFr::from_u64(v as u64)
}

fn padded(column: &[ArkFr], n: usize) -> Vec<ArkFr> {
    let mut out = column.to_vec();
    out.resize(n, f0());
    out
}

// chi_bounds indices for the distinct-valid-pairs shape.
const CHI_MJ: usize = 0;
const CHI_NR: usize = 1;
const CHI_NS: usize = 2;

/// The dynamic column layout of the binary distinct-valid-pairs sumcheck. `p` =
/// #r_other columns, `q` = #s_other columns. Order:
/// `[r_key, r_other.., s_key, s_other.., r_other_out.., key, s_other_out.., rho_l,
///   rho_r, mult_l, mult_r, rec_src_l, rec_cand_l, rec_src_r, rec_cand_r]`.
#[derive(Clone, Copy)]
struct Layout {
    p: usize,
    q: usize,
}

impl Layout {
    fn r_key(&self) -> usize {
        0
    }
    fn r_other(&self, d: usize) -> usize {
        1 + d
    }
    fn s_key(&self) -> usize {
        1 + self.p
    }
    fn s_other(&self, d: usize) -> usize {
        2 + self.p + d
    }
    fn r_other_out(&self, d: usize) -> usize {
        2 + self.p + self.q + d
    }
    fn key(&self) -> usize {
        2 + 2 * self.p + self.q
    }
    fn s_other_out(&self, d: usize) -> usize {
        3 + 2 * self.p + self.q + d
    }
    fn rho_l(&self) -> usize {
        3 + 2 * self.p + 2 * self.q
    }
    fn rho_r(&self) -> usize {
        4 + 2 * self.p + 2 * self.q
    }
    fn mult_l(&self) -> usize {
        5 + 2 * self.p + 2 * self.q
    }
    fn mult_r(&self) -> usize {
        6 + 2 * self.p + 2 * self.q
    }
    fn rec_src_l(&self) -> usize {
        7 + 2 * self.p + 2 * self.q
    }
    fn rec_cand_l(&self) -> usize {
        8 + 2 * self.p + 2 * self.q
    }
    fn rec_src_r(&self) -> usize {
        9 + 2 * self.p + 2 * self.q
    }
    fn rec_cand_r(&self) -> usize {
        10 + 2 * self.p + 2 * self.q
    }
    fn num_columns(&self) -> usize {
        11 + 2 * self.p + 2 * self.q
    }
}

/// The shared power-of-two domain (`>= 2`) onto which every join column is padded.
///
/// Exposed `pub` alongside [`compute_binary_join`] so external consumers (the
/// `woql-proof` compiler's end-to-end tests) can reproduce the exact per-column
/// padding a BGP proof uses when recomputing an independent natural join.
pub fn binary_join_domain(
    n_r: usize,
    n_s: usize,
    m_j: usize,
    n_u: usize,
) -> io::Result<usize> {
    n_r.max(n_s)
        .max(m_j)
        .max(n_u)
        .checked_next_power_of_two()
        .ok_or_else(|| invalid("binary join domain size overflow"))
        .map(|d| d.max(2))
}

/// The injective combined-index radix `C = next_pow2(max(n_R, n_S))` (`>= n_S > rho_r`,
/// `>= n_R > rho_l`). Getting this wrong (e.g. `next_pow2(n_R)` when `n_R < n_S`)
/// reopens the injectivity/substitution bug class the audit closed.
pub(crate) fn combined_index_radix(n_r: usize, n_s: usize) -> io::Result<usize> {
    n_r.max(n_s)
        .checked_next_power_of_two()
        .ok_or_else(|| invalid("combined-index radix overflow"))
        .map(|c| c.max(1))
}

/// `rec_col * (alpha - Σ coeff*operand) - 1 = 0`.
fn reciprocal_fold_claim(rec_col: usize, fold: &[(ArkFr, Operand)], alpha: ArkFr) -> SumcheckClaim {
    let mut terms = vec![Term {
        coeff: alpha,
        operands: vec![Operand::Column(rec_col)],
    }];
    for (coeff, operand) in fold {
        terms.push(Term {
            coeff: f0() - *coeff,
            operands: vec![Operand::Column(rec_col), *operand],
        });
    }
    terms.push(Term {
        coeff: f0() - f1(),
        operands: Vec::new(),
    });
    SumcheckClaim {
        shape: ClaimShape::Identity,
        expr: Expr { terms },
        target: f0(),
    }
}

/// `beta`-power fold `[(1, ops[0]), (beta, ops[1]), ..]`.
fn power_fold(beta: ArkFr, operands: &[Operand]) -> Vec<(ArkFr, Operand)> {
    let mut out = Vec::with_capacity(operands.len());
    let mut power = f1();
    for op in operands {
        out.push((power, *op));
        power = power * beta;
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn binary_pairs_shape(
    layout: &Layout,
    num_vars: usize,
    m_j: usize,
    n_r: usize,
    n_s: usize,
    alpha_l: ArkFr,
    beta_l: ArkFr,
    alpha_r: ArkFr,
    beta_r: ArkFr,
) -> SumcheckShape {
    // LEFT fold order: [r_other_0.., r_key, idx] for the source; the candidate uses
    // [r_other_out_0.., key, rho_l] with the SAME powers.
    let mut left_src_ops: Vec<Operand> =
        (0..layout.p).map(|d| Operand::Column(layout.r_other(d))).collect();
    left_src_ops.push(Operand::Column(layout.r_key()));
    left_src_ops.push(Operand::Idx);
    let mut left_cand_ops: Vec<Operand> =
        (0..layout.p).map(|d| Operand::Column(layout.r_other_out(d))).collect();
    left_cand_ops.push(Operand::Column(layout.key()));
    left_cand_ops.push(Operand::Column(layout.rho_l()));

    // RIGHT fold order: [s_key, s_other_0.., idx] and [key, s_other_out_0.., rho_r].
    let mut right_src_ops: Vec<Operand> = vec![Operand::Column(layout.s_key())];
    right_src_ops.extend((0..layout.q).map(|d| Operand::Column(layout.s_other(d))));
    right_src_ops.push(Operand::Idx);
    let mut right_cand_ops: Vec<Operand> = vec![Operand::Column(layout.key())];
    right_cand_ops.extend((0..layout.q).map(|d| Operand::Column(layout.s_other_out(d))));
    right_cand_ops.push(Operand::Column(layout.rho_r()));

    let src_l = reciprocal_fold_claim(layout.rec_src_l(), &power_fold(beta_l, &left_src_ops), alpha_l);
    let cand_l =
        reciprocal_fold_claim(layout.rec_cand_l(), &power_fold(beta_l, &left_cand_ops), alpha_l);
    // Source side masked by chi_{n_R} (audit fix: don't regress source masking).
    let link_l = masked_multiset_link_claim(
        &[Operand::Chi(CHI_MJ)],
        layout.rec_cand_l(),
        &[Operand::Chi(CHI_NR), Operand::Column(layout.mult_l())],
        layout.rec_src_l(),
    );

    let src_r = reciprocal_fold_claim(layout.rec_src_r(), &power_fold(beta_r, &right_src_ops), alpha_r);
    let cand_r =
        reciprocal_fold_claim(layout.rec_cand_r(), &power_fold(beta_r, &right_cand_ops), alpha_r);
    let link_r = masked_multiset_link_claim(
        &[Operand::Chi(CHI_MJ)],
        layout.rec_cand_r(),
        &[Operand::Chi(CHI_NS), Operand::Column(layout.mult_r())],
        layout.rec_src_r(),
    );

    SumcheckShape {
        num_vars,
        num_columns: layout.num_columns(),
        chi_bounds: vec![m_j, n_r, n_s],
        claims: vec![src_l, cand_l, link_l, src_r, cand_r, link_r],
    }
}

/// Source multiplicity `mult[a] = #{ k < m_j : rho[k] == a }` over `[0, real)`.
fn source_multiplicities(rho: &[ArkFr], real: usize, m_j: usize, n: usize) -> Vec<ArkFr> {
    let mut mult = vec![f0(); n];
    for a in 0..real {
        let a_fr = fr(a);
        mult[a] = fr(rho[..m_j].iter().filter(|v| **v == a_fr).count());
    }
    mult
}

/// The padded data columns of a binary join, all length `N`.
#[derive(Clone)]
pub struct JoinColumns {
    pub r_key: Vec<ArkFr>,
    pub r_other: Vec<Vec<ArkFr>>,
    pub s_key: Vec<ArkFr>,
    pub s_other: Vec<Vec<ArkFr>>,
    pub r_other_out: Vec<Vec<ArkFr>>,
    pub key: Vec<ArkFr>,
    pub s_other_out: Vec<Vec<ArkFr>>,
    pub rho_l: Vec<ArkFr>,
    pub rho_r: Vec<ArkFr>,
}

/// The per-column commitments of a binary join (all committed at the shared domain).
#[derive(Clone, Debug)]
pub struct BinaryJoinCommitments {
    pub r_key: ArkGT,
    pub r_other: Vec<ArkGT>,
    pub s_key: ArkGT,
    pub s_other: Vec<ArkGT>,
    pub r_other_out: Vec<ArkGT>,
    pub key: ArkGT,
    pub s_other_out: Vec<ArkGT>,
    pub rho_l: ArkGT,
    pub rho_r: ArkGT,
    pub u: ArkGT,
    pub w_l: ArkGT,
    pub w_r: ArkGT,
}

/// A distinct-valid-pairs proof for the binary join.
#[derive(Clone, Debug)]
pub struct BinaryPairsProof {
    pub membership_sumcheck: SumcheckProof,
    pub membership_opening: ColumnOpening,
    pub mult_l_commitment: ArkGT,
    pub mult_r_commitment: ArkGT,
    pub rec_src_l_commitment: ArkGT,
    pub rec_cand_l_commitment: ArkGT,
    pub rec_src_r_commitment: ArkGT,
    pub rec_cand_r_commitment: ArkGT,
    pub sorted: SortedProof,
}

fn bind_pairs_data(transcript: &mut Transcript254, com: &BinaryJoinCommitments) {
    transcript.append_group(b"pow/join2/com-r-key", &com.r_key);
    for c in &com.r_other {
        transcript.append_group(b"pow/join2/com-r-other", c);
    }
    transcript.append_group(b"pow/join2/com-s-key", &com.s_key);
    for c in &com.s_other {
        transcript.append_group(b"pow/join2/com-s-other", c);
    }
    for c in &com.r_other_out {
        transcript.append_group(b"pow/join2/com-r-other-out", c);
    }
    transcript.append_group(b"pow/join2/com-key", &com.key);
    for c in &com.s_other_out {
        transcript.append_group(b"pow/join2/com-s-other-out", c);
    }
    transcript.append_group(b"pow/join2/com-rho-l", &com.rho_l);
    transcript.append_group(b"pow/join2/com-rho-r", &com.rho_r);
}

fn slices(cols: &[Vec<ArkFr>]) -> Vec<&[ArkFr]> {
    cols.iter().map(|c| c.as_slice()).collect()
}

/// Prove distinct-valid-pairs for EXPLICIT padded columns + multiplicities + radix.
#[allow(clippy::too_many_arguments)]
fn prove_binary_pairs(
    cols: &JoinColumns,
    mult_l_in: &[ArkFr],
    mult_r_in: &[ArkFr],
    n_r: usize,
    n_s: usize,
    m_j: usize,
    radix: usize,
    com: &BinaryJoinCommitments,
    transcript: &mut Transcript254,
) -> io::Result<BinaryPairsProof> {
    let n = cols.r_key.len();
    let layout = Layout {
        p: cols.r_other.len(),
        q: cols.s_other.len(),
    };
    let num_vars = n.trailing_zeros() as usize;
    let idx: Vec<ArkFr> = (0..n).map(fr).collect();

    let mult_l_commitment = commit_column(mult_l_in)?;
    let mult_r_commitment = commit_column(mult_r_in)?;

    // Round 1: bind the data columns + multiplicities, draw fresh challenges.
    bind_pairs_data(transcript, com);
    transcript.append_group(b"pow/join2/com-mult-l", &mult_l_commitment);
    transcript.append_group(b"pow/join2/com-mult-r", &mult_r_commitment);
    let beta_l = transcript.challenge_scalar(b"pow/join2/left-beta");
    let alpha_l = transcript.challenge_scalar(b"pow/join2/left-alpha");
    let beta_r = transcript.challenge_scalar(b"pow/join2/right-beta");
    let alpha_r = transcript.challenge_scalar(b"pow/join2/right-alpha");

    // Round 2: reciprocal witnesses over the folds.
    let mut left_src: Vec<&[ArkFr]> = slices(&cols.r_other);
    left_src.push(&cols.r_key);
    left_src.push(&idx);
    let mut left_cand: Vec<&[ArkFr]> = slices(&cols.r_other_out);
    left_cand.push(&cols.key);
    left_cand.push(&cols.rho_l);
    let mut right_src: Vec<&[ArkFr]> = vec![cols.s_key.as_slice()];
    right_src.extend(slices(&cols.s_other));
    right_src.push(&idx);
    let mut right_cand: Vec<&[ArkFr]> = vec![cols.key.as_slice()];
    right_cand.extend(slices(&cols.s_other_out));
    right_cand.push(&cols.rho_r);

    let rec_src_l = reciprocal_witness(alpha_l, &rlc_fold(beta_l, &left_src));
    let rec_cand_l = reciprocal_witness(alpha_l, &rlc_fold(beta_l, &left_cand));
    let rec_src_r = reciprocal_witness(alpha_r, &rlc_fold(beta_r, &right_src));
    let rec_cand_r = reciprocal_witness(alpha_r, &rlc_fold(beta_r, &right_cand));
    let rec_src_l_commitment = commit_column(&rec_src_l)?;
    let rec_cand_l_commitment = commit_column(&rec_cand_l)?;
    let rec_src_r_commitment = commit_column(&rec_src_r)?;
    let rec_cand_r_commitment = commit_column(&rec_cand_r)?;
    transcript.append_group(b"pow/join2/com-rec-src-l", &rec_src_l_commitment);
    transcript.append_group(b"pow/join2/com-rec-cand-l", &rec_cand_l_commitment);
    transcript.append_group(b"pow/join2/com-rec-src-r", &rec_src_r_commitment);
    transcript.append_group(b"pow/join2/com-rec-cand-r", &rec_cand_r_commitment);

    let shape = binary_pairs_shape(&layout, num_vars, m_j, n_r, n_s, alpha_l, beta_l, alpha_r, beta_r);
    let mut columns = vec![cols.r_key.clone()];
    columns.extend(cols.r_other.iter().cloned());
    columns.push(cols.s_key.clone());
    columns.extend(cols.s_other.iter().cloned());
    columns.extend(cols.r_other_out.iter().cloned());
    columns.push(cols.key.clone());
    columns.extend(cols.s_other_out.iter().cloned());
    columns.push(cols.rho_l.clone());
    columns.push(cols.rho_r.clone());
    columns.push(mult_l_in.to_vec());
    columns.push(mult_r_in.to_vec());
    columns.push(rec_src_l);
    columns.push(rec_cand_l);
    columns.push(rec_src_r);
    columns.push(rec_cand_r);
    let (membership_sumcheck, r, _evals) = prove_sumcheck(&shape, &columns, transcript)?;
    let membership_opening = prove_column_openings(&columns, &r, transcript)?;

    // Combined-index strict monotonicity, homomorphic com(i_eval).
    let radix_fr = fr(radix);
    let i_eval: Vec<ArkFr> = (0..n)
        .map(|i| radix_fr * cols.rho_l[i] + cols.rho_r[i])
        .collect();
    let com_i_eval = com.rho_l.scale(&radix_fr).add(&com.rho_r);
    let sorted = prove_strictly_sorted(&i_eval, m_j, &com_i_eval, transcript)?;

    Ok(BinaryPairsProof {
        membership_sumcheck,
        membership_opening,
        mult_l_commitment,
        mult_r_commitment,
        rec_src_l_commitment,
        rec_cand_l_commitment,
        rec_src_r_commitment,
        rec_cand_r_commitment,
        sorted,
    })
}

#[allow(clippy::too_many_arguments)]
fn verify_binary_pairs(
    com: &BinaryJoinCommitments,
    n_r: usize,
    n_s: usize,
    m_j: usize,
    radix: usize,
    proof: &BinaryPairsProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    let num_vars = proof.membership_sumcheck.round_polys.len();
    if num_vars == 0 || num_vars >= usize::BITS as usize {
        return Err(invalid("binary pairs proof has an implausible variable count"));
    }
    let n = 1usize << num_vars;
    if n_r > n || n_s > n || m_j > n {
        return Err(invalid("binary join real length exceeds the domain"));
    }
    let layout = Layout {
        p: com.r_other.len(),
        q: com.s_other.len(),
    };
    if com.r_other_out.len() != layout.p || com.s_other_out.len() != layout.q {
        return Err(invalid("binary join output arity mismatch"));
    }

    bind_pairs_data(transcript, com);
    transcript.append_group(b"pow/join2/com-mult-l", &proof.mult_l_commitment);
    transcript.append_group(b"pow/join2/com-mult-r", &proof.mult_r_commitment);
    let beta_l = transcript.challenge_scalar(b"pow/join2/left-beta");
    let alpha_l = transcript.challenge_scalar(b"pow/join2/left-alpha");
    let beta_r = transcript.challenge_scalar(b"pow/join2/right-beta");
    let alpha_r = transcript.challenge_scalar(b"pow/join2/right-alpha");
    transcript.append_group(b"pow/join2/com-rec-src-l", &proof.rec_src_l_commitment);
    transcript.append_group(b"pow/join2/com-rec-cand-l", &proof.rec_cand_l_commitment);
    transcript.append_group(b"pow/join2/com-rec-src-r", &proof.rec_src_r_commitment);
    transcript.append_group(b"pow/join2/com-rec-cand-r", &proof.rec_cand_r_commitment);

    let shape = binary_pairs_shape(&layout, num_vars, m_j, n_r, n_s, alpha_l, beta_l, alpha_r, beta_r);
    let (r, evals) = verify_sumcheck(&shape, &proof.membership_sumcheck, transcript)?;

    let mut coms = vec![com.r_key];
    coms.extend(com.r_other.iter().copied());
    coms.push(com.s_key);
    coms.extend(com.s_other.iter().copied());
    coms.extend(com.r_other_out.iter().copied());
    coms.push(com.key);
    coms.extend(com.s_other_out.iter().copied());
    coms.push(com.rho_l);
    coms.push(com.rho_r);
    coms.push(proof.mult_l_commitment);
    coms.push(proof.mult_r_commitment);
    coms.push(proof.rec_src_l_commitment);
    coms.push(proof.rec_cand_l_commitment);
    coms.push(proof.rec_src_r_commitment);
    coms.push(proof.rec_cand_r_commitment);
    verify_column_openings(&coms, &r, &evals, &proof.membership_opening, transcript)?;

    let radix_fr = fr(radix);
    let com_i_eval = com.rho_l.scale(&radix_fr).add(&com.rho_r);
    verify_strictly_sorted(&com_i_eval, m_j, &proof.sorted, transcript)
}

/// Sorted distinct union of two value columns (the join-key universe `U`).
fn sorted_distinct_union(a: &[ArkFr], b: &[ArkFr]) -> Vec<ArkFr> {
    let mut all: Vec<ArkFr> = a.iter().chain(b.iter()).copied().collect();
    all.sort_by(|p, q| p.0.into_bigint().cmp(&q.0.into_bigint()));
    all.dedup();
    all
}

fn multiplicity_in(u: &[ArkFr], col: &[ArkFr]) -> Vec<ArkFr> {
    u.iter()
        .map(|value| fr(col.iter().filter(|e| *e == value).count()))
        .collect()
}

/// The complete-join witness: output columns + the key universe + multiplicities.
/// `n_u` is the CLAIMED distinct-key count (the chi real length); the honest builder
/// sets it to `u.len()`, but it is decoupled so a malicious prover can stash a
/// duplicate key in `u`'s padding tail `[n_u, domain)` (that attack must be rejected).
#[derive(Clone)]
pub struct BinaryJoinWitness {
    pub r_other_out: Vec<Vec<ArkFr>>,
    pub key: Vec<ArkFr>,
    pub s_other_out: Vec<Vec<ArkFr>>,
    pub rho_l: Vec<ArkFr>,
    pub rho_r: Vec<ArkFr>,
    pub u: Vec<ArkFr>,
    pub w_l: Vec<ArkFr>,
    pub w_r: Vec<ArkFr>,
    pub n_u: usize,
    pub m_j: usize,
}

/// Compute the complete join of `R = (r_key, r_other..)` and `S = (s_key, s_other..)`
/// on `r_key == s_key`, plus the witness columns. Output rows are emitted in `(a, b)`
/// lexicographic order == combined-index `C*a + b` ascending, so `rho`s are strictly
/// increasing.
pub fn compute_binary_join(
    r_key: &[ArkFr],
    r_other: &[&[ArkFr]],
    s_key: &[ArkFr],
    s_other: &[&[ArkFr]],
) -> BinaryJoinWitness {
    let (n_r, n_s) = (r_key.len(), s_key.len());
    let (p, q) = (r_other.len(), s_other.len());
    let mut r_other_out: Vec<Vec<ArkFr>> = vec![Vec::new(); p];
    let mut key = Vec::new();
    let mut s_other_out: Vec<Vec<ArkFr>> = vec![Vec::new(); q];
    let (mut rho_l, mut rho_r) = (Vec::new(), Vec::new());
    for a in 0..n_r {
        for b in 0..n_s {
            if r_key[a] == s_key[b] {
                for d in 0..p {
                    r_other_out[d].push(r_other[d][a]);
                }
                key.push(r_key[a]);
                for d in 0..q {
                    s_other_out[d].push(s_other[d][b]);
                }
                rho_l.push(fr(a));
                rho_r.push(fr(b));
            }
        }
    }
    let m_j = key.len();
    let u = sorted_distinct_union(r_key, s_key);
    let w_l = multiplicity_in(&u, r_key);
    let w_r = multiplicity_in(&u, s_key);
    let n_u = u.len();
    BinaryJoinWitness {
        r_other_out,
        key,
        s_other_out,
        rho_l,
        rho_r,
        u,
        w_l,
        w_r,
        n_u,
        m_j,
    }
}

/// Commit every join column at the shared domain.
pub fn binary_join_commitments(
    r_key: &[ArkFr],
    r_other: &[&[ArkFr]],
    s_key: &[ArkFr],
    s_other: &[&[ArkFr]],
    witness: &BinaryJoinWitness,
) -> io::Result<BinaryJoinCommitments> {
    let n = binary_join_domain(r_key.len(), s_key.len(), witness.m_j, witness.n_u)?;
    let commit_vec = |cols: &[Vec<ArkFr>]| -> io::Result<Vec<ArkGT>> {
        cols.iter().map(|c| commit_column(&padded(c, n))).collect()
    };
    let commit_ref_vec = |cols: &[&[ArkFr]]| -> io::Result<Vec<ArkGT>> {
        cols.iter().map(|c| commit_column(&padded(c, n))).collect()
    };
    Ok(BinaryJoinCommitments {
        r_key: commit_column(&padded(r_key, n))?,
        r_other: commit_ref_vec(r_other)?,
        s_key: commit_column(&padded(s_key, n))?,
        s_other: commit_ref_vec(s_other)?,
        r_other_out: commit_vec(&witness.r_other_out)?,
        key: commit_column(&padded(&witness.key, n))?,
        s_other_out: commit_vec(&witness.s_other_out)?,
        rho_l: commit_column(&padded(&witness.rho_l, n))?,
        rho_r: commit_column(&padded(&witness.rho_r, n))?,
        u: commit_column(&padded(&witness.u, n))?,
        w_l: commit_column(&padded(&witness.w_l, n))?,
        w_r: commit_column(&padded(&witness.w_r, n))?,
    })
}

/// A full sound+complete binary-join proof.
#[derive(Clone, Debug)]
pub struct BinaryJoinProof {
    pub pairs: BinaryPairsProof,
    pub sorted_u: SortedProof,
    pub membership_l: MembershipProof,
    pub membership_r: MembershipProof,
    pub count: CountProof,
    pub n_u: usize,
}

impl BinaryJoinProof {
    /// The number of variables the join's pairs-membership sumcheck operates
    /// over, i.e. `log2` of the join domain `d_j`. The chain verifier
    /// cross-checks this against each stage's relocation TARGET domain so a stage
    /// cannot run its join over a different domain than the one its input columns
    /// were relocated into.
    pub fn num_vars(&self) -> usize {
        self.pairs.membership_sumcheck.round_polys.len()
    }
}

const LABEL_DOMAIN: &[u8] = b"pow/join2/instance-v1";
const LABEL_LEN: &[u8] = b"pow/join2/len";

fn bind_instance(
    transcript: &mut Transcript254,
    com: &BinaryJoinCommitments,
    n_r: usize,
    n_s: usize,
    n_u: usize,
    m_j: usize,
) {
    transcript.append_bytes(LABEL_DOMAIN, b"parture/proof-of-woql/binary-join/v0");
    for v in [n_r, n_s, n_u, m_j] {
        transcript.append_bytes(LABEL_LEN, &(v as u64).to_le_bytes());
    }
    bind_pairs_data(transcript, com);
    transcript.append_group(b"pow/join2/com-u", &com.u);
    transcript.append_group(b"pow/join2/com-w-l", &com.w_l);
    transcript.append_group(b"pow/join2/com-w-r", &com.w_r);
}

/// Prove the complete binary join of `R`, `S` on `r_key == s_key`. `com` must commit
/// every column at `binary_join_domain` (see [`binary_join_commitments`]). Returns the
/// proof, the OUTPUT commitments `[r_other_out.., key, s_other_out..]`, and `m_J`.
pub fn prove_binary_join(
    r_key: &[ArkFr],
    r_other: &[&[ArkFr]],
    s_key: &[ArkFr],
    s_other: &[&[ArkFr]],
    com: &BinaryJoinCommitments,
    transcript: &mut Transcript254,
) -> io::Result<(BinaryJoinProof, Vec<ArkGT>, usize)> {
    let witness = compute_binary_join(r_key, r_other, s_key, s_other);
    let radix = combined_index_radix(r_key.len(), s_key.len())?;
    let proof = prove_binary_join_with_witness(
        r_key, r_other, s_key, s_other, &witness, radix, com, transcript,
    )?;
    let mut out = com.r_other_out.clone();
    out.push(com.key);
    out.extend(com.s_other_out.iter().copied());
    Ok((proof, out, witness.m_j))
}

/// Prove a binary join from an EXPLICIT witness and radix (the honest entry derives
/// them; tests inject malicious output / U / w / a wrong radix). `witness.u`/`w_l`/`w_r`
/// may carry a padding-tail stash; `witness.rho_*`/output may be fabricated.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_binary_join_with_witness(
    r_key: &[ArkFr],
    r_other: &[&[ArkFr]],
    s_key: &[ArkFr],
    s_other: &[&[ArkFr]],
    witness: &BinaryJoinWitness,
    radix: usize,
    com: &BinaryJoinCommitments,
    transcript: &mut Transcript254,
) -> io::Result<BinaryJoinProof> {
    let (n_r, n_s) = (r_key.len(), s_key.len());
    let n_u = witness.n_u;
    let m_j = witness.m_j;
    let n = binary_join_domain(n_r, n_s, m_j, n_u)?;

    bind_instance(transcript, com, n_r, n_s, n_u, m_j);

    let cols = JoinColumns {
        r_key: padded(r_key, n),
        r_other: r_other.iter().map(|c| padded(c, n)).collect(),
        s_key: padded(s_key, n),
        s_other: s_other.iter().map(|c| padded(c, n)).collect(),
        r_other_out: witness.r_other_out.iter().map(|c| padded(c, n)).collect(),
        key: padded(&witness.key, n),
        s_other_out: witness.s_other_out.iter().map(|c| padded(c, n)).collect(),
        rho_l: padded(&witness.rho_l, n),
        rho_r: padded(&witness.rho_r, n),
    };
    let mult_l = source_multiplicities(&cols.rho_l, n_r, m_j, n);
    let mult_r = source_multiplicities(&cols.rho_r, n_s, m_j, n);

    let pairs = prove_binary_pairs(&cols, &mult_l, &mult_r, n_r, n_s, m_j, radix, com, transcript)?;

    // U distinctness over [0, n_U).
    let u_full = padded(&witness.u, n);
    let sorted_u = prove_strictly_sorted(&u_full, n_u, &com.u, transcript)?;
    // r_key is the image of U under w_L, s_key the image under w_R (both sides masked).
    let w_l_full = padded(&witness.w_l, n);
    let w_r_full = padded(&witness.w_r, n);
    let membership_l = prove_membership_with_mult_with_witness(
        &u_full, &cols.r_key, &w_l_full, n_u, n_r, &com.u, &com.r_key, &com.w_l, transcript,
    )?;
    let membership_r = prove_membership_with_mult_with_witness(
        &u_full, &cols.s_key, &w_r_full, n_u, n_s, &com.u, &com.s_key, &com.w_r, transcript,
    )?;
    // Count Σ chi_{n_U} w_L*w_R = m_J (true n_U -- audit fix).
    let count = prove_count_identity(&w_l_full, &w_r_full, n_u, m_j, &com.w_l, &com.w_r, transcript)?;

    Ok(BinaryJoinProof {
        pairs,
        sorted_u,
        membership_l,
        membership_r,
        count,
        n_u,
    })
}

/// Verify a full binary-join proof against the committed instance.
pub fn verify_binary_join(
    com: &BinaryJoinCommitments,
    n_r: usize,
    n_s: usize,
    m_j: usize,
    proof: &BinaryJoinProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    let radix = combined_index_radix(n_r, n_s)?;
    verify_binary_join_inner(com, n_r, n_s, proof.n_u, m_j, radix, proof, transcript)
}

#[allow(clippy::too_many_arguments)]
fn verify_binary_join_inner(
    com: &BinaryJoinCommitments,
    n_r: usize,
    n_s: usize,
    n_u: usize,
    m_j: usize,
    radix: usize,
    proof: &BinaryJoinProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    bind_instance(transcript, com, n_r, n_s, n_u, m_j);
    verify_binary_pairs(com, n_r, n_s, m_j, radix, &proof.pairs, transcript)?;
    verify_strictly_sorted(&com.u, n_u, &proof.sorted_u, transcript)?;
    verify_membership_with_mult(&com.u, &com.r_key, &com.w_l, n_u, n_r, &proof.membership_l, transcript)?;
    verify_membership_with_mult(&com.u, &com.s_key, &com.w_r, n_u, n_s, &proof.membership_r, transcript)?;
    verify_count_identity(&com.w_l, &com.w_r, n_u, m_j, &proof.count, transcript)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frs(vals: &[u64]) -> Vec<ArkFr> {
        vals.iter().map(|v| ArkFr::from_u64(*v)).collect()
    }

    fn transcript() -> Transcript254 {
        Transcript254::new(b"parture-join2-test")
    }

    fn refs(cols: &[Vec<ArkFr>]) -> Vec<&[ArkFr]> {
        cols.iter().map(|c| c.as_slice()).collect()
    }

    // R = (dept, emp); S = (dept, name). Join on dept.
    // R: (10,1),(20,2),(10,3)  -> n_R=3.  S: (10,100),(30,300) -> n_S=2.
    // Matches: (a0,b0)->(emp1,dept10,name100); (a2,b0)->(emp3,dept10,name100). m_J=2.
    fn relations() -> (Vec<ArkFr>, Vec<Vec<ArkFr>>, Vec<ArkFr>, Vec<Vec<ArkFr>>) {
        let r_key = frs(&[10, 20, 10]);
        let r_other = vec![frs(&[1, 2, 3])];
        let s_key = frs(&[10, 30]);
        let s_other = vec![frs(&[100, 300])];
        (r_key, r_other, s_key, s_other)
    }

    fn honest_setup(
        r_key: &[ArkFr],
        r_other: &[Vec<ArkFr>],
        s_key: &[ArkFr],
        s_other: &[Vec<ArkFr>],
    ) -> (BinaryJoinWitness, BinaryJoinCommitments) {
        let ro = refs(r_other);
        let so = refs(s_other);
        let witness = compute_binary_join(r_key, &ro, s_key, &so);
        let com = binary_join_commitments(r_key, &ro, s_key, &so, &witness).unwrap();
        (witness, com)
    }

    #[test]
    fn honest_binary_join_of_two_distinct_relations_verifies_with_exact_result() {
        let (r_key, r_other, s_key, s_other) = relations();
        let (witness, com) = honest_setup(&r_key, &r_other, &s_key, &s_other);
        assert_eq!(witness.m_j, 2);
        // Exact contents (r_other carried through, key, s_other carried through).
        assert_eq!(witness.r_other_out, vec![frs(&[1, 3])]);
        assert_eq!(witness.key, frs(&[10, 10]));
        assert_eq!(witness.s_other_out, vec![frs(&[100, 100])]);
        assert_eq!(witness.rho_l, frs(&[0, 2]));
        assert_eq!(witness.rho_r, frs(&[0, 0]));

        let ro = refs(&r_other);
        let so = refs(&s_other);
        let mut pt = transcript();
        let (proof, out, m) = prove_binary_join(&r_key, &ro, &s_key, &so, &com, &mut pt).unwrap();
        assert_eq!(m, 2);
        // Returned OUTPUT commitments == the committed result columns.
        let mut expected = com.r_other_out.clone();
        expected.push(com.key);
        expected.extend(com.s_other_out.iter().copied());
        assert_eq!(out, expected);
        let mut vt = transcript();
        verify_binary_join(&com, r_key.len(), s_key.len(), m, &proof, &mut vt).unwrap();
    }

    #[test]
    fn honest_join_with_arbitrary_width_other_columns_verifies() {
        // R has TWO other columns, S has TWO other columns.
        let r_key = frs(&[10, 20, 10]);
        let r_other = vec![frs(&[1, 2, 3]), frs(&[7, 8, 9])];
        let s_key = frs(&[10, 30]);
        let s_other = vec![frs(&[100, 300]), frs(&[111, 333])];
        let (witness, com) = honest_setup(&r_key, &r_other, &s_key, &s_other);
        assert_eq!(witness.m_j, 2);
        assert_eq!(witness.r_other_out, vec![frs(&[1, 3]), frs(&[7, 9])]);
        assert_eq!(witness.s_other_out, vec![frs(&[100, 100]), frs(&[111, 111])]);
        let ro = refs(&r_other);
        let so = refs(&s_other);
        let mut pt = transcript();
        let (proof, _out, m) = prove_binary_join(&r_key, &ro, &s_key, &so, &com, &mut pt).unwrap();
        let mut vt = transcript();
        verify_binary_join(&com, r_key.len(), s_key.len(), m, &proof, &mut vt).unwrap();
    }

    #[test]
    fn self_join_as_binary_join_verifies() {
        // The self-join F=(Sf,Of) on Of==Sf is the binary join with R=(key=Of,[Sf]),
        // S=(key=Sf,[Of]). F={(1,5),(2,5),(5,10),(5,20)} -> complete join has m_J=4.
        let sf = frs(&[1, 2, 5, 5]);
        let of = frs(&[5, 5, 10, 20]);
        let r_key = of.clone(); // left joins on Of
        let r_other = vec![sf.clone()];
        let s_key = sf.clone(); // right joins on Sf
        let s_other = vec![of.clone()];
        let (witness, com) = honest_setup(&r_key, &r_other, &s_key, &s_other);
        assert_eq!(witness.m_j, 4);
        // X = r_other_out = Sf[rho_l], Z = s_other_out = Of[rho_r], key = Y.
        assert_eq!(witness.r_other_out, vec![frs(&[1, 1, 2, 2])]);
        assert_eq!(witness.key, frs(&[5, 5, 5, 5]));
        assert_eq!(witness.s_other_out, vec![frs(&[10, 20, 10, 20])]);
        let ro = refs(&r_other);
        let so = refs(&s_other);
        let mut pt = transcript();
        let (proof, _out, m) = prove_binary_join(&r_key, &ro, &s_key, &so, &com, &mut pt).unwrap();
        let mut vt = transcript();
        verify_binary_join(&com, r_key.len(), s_key.len(), m, &proof, &mut vt).unwrap();
    }

    #[test]
    fn empty_join_no_matching_keys_verifies() {
        let r_key = frs(&[1, 2]);
        let r_other = vec![frs(&[10, 20])];
        let s_key = frs(&[3, 4, 5]);
        let s_other = vec![frs(&[30, 40, 50])];
        let (witness, com) = honest_setup(&r_key, &r_other, &s_key, &s_other);
        assert_eq!(witness.m_j, 0);
        let ro = refs(&r_other);
        let so = refs(&s_other);
        let mut pt = transcript();
        let (proof, _out, m) = prove_binary_join(&r_key, &ro, &s_key, &so, &com, &mut pt).unwrap();
        let mut vt = transcript();
        verify_binary_join(&com, r_key.len(), s_key.len(), m, &proof, &mut vt).unwrap();
    }

    // ---- Adversarial matrix (two relations) ----

    // Attempt with an EXPLICIT (possibly malicious) witness + radix, re-committing the
    // (possibly mutated) witness so `com` matches what the prover uses.
    fn attempt_with_witness(
        r_key: &[ArkFr],
        r_other: &[Vec<ArkFr>],
        s_key: &[ArkFr],
        s_other: &[Vec<ArkFr>],
        witness: &BinaryJoinWitness,
        radix: usize,
    ) -> io::Result<()> {
        let ro = refs(r_other);
        let so = refs(s_other);
        let com = binary_join_commitments(r_key, &ro, s_key, &so, witness).unwrap();
        let mut pt = transcript();
        let proof =
            prove_binary_join_with_witness(r_key, &ro, s_key, &so, witness, radix, &com, &mut pt)?;
        let mut vt = transcript();
        verify_binary_join_inner(
            &com,
            r_key.len(),
            s_key.len(),
            witness.n_u,
            witness.m_j,
            radix,
            &proof,
            &mut vt,
        )
    }

    // Commit an explicit set of padded pairs columns (u/w unused by the pairs gadget).
    fn commit_cols(cols: &JoinColumns) -> BinaryJoinCommitments {
        let cv = |c: &[ArkFr]| commit_column(c).unwrap();
        let cvv = |cs: &[Vec<ArkFr>]| cs.iter().map(|c| cv(c)).collect::<Vec<_>>();
        let zero = vec![f0(); cols.r_key.len()];
        BinaryJoinCommitments {
            r_key: cv(&cols.r_key),
            r_other: cvv(&cols.r_other),
            s_key: cv(&cols.s_key),
            s_other: cvv(&cols.s_other),
            r_other_out: cvv(&cols.r_other_out),
            key: cv(&cols.key),
            s_other_out: cvv(&cols.s_other_out),
            rho_l: cv(&cols.rho_l),
            rho_r: cv(&cols.rho_r),
            u: cv(&zero),
            w_l: cv(&zero),
            w_r: cv(&zero),
        }
    }

    #[test]
    fn wrong_radix_next_pow2_of_n_r_is_rejected() {
        // n_R = 2 < n_S = 3, all keys equal -> the join is the full 2x3 grid (m_J=6).
        // Correct C = next_pow2(max(2,3)) = 4 keeps every (rho_l,rho_r) i_eval distinct;
        // the WRONG C = next_pow2(n_R) = 2 collides (0,2) and (1,0) at i_eval=2 -> the
        // combined-index monotonicity cannot strictly order the distinct pairs -> REJECT.
        let r_key = frs(&[5, 5]);
        let r_other = vec![frs(&[1, 2])];
        let s_key = frs(&[5, 5, 5]);
        let s_other = vec![frs(&[10, 20, 30])];
        let (witness, _) = honest_setup(&r_key, &r_other, &s_key, &s_other);
        assert_eq!(witness.m_j, 6);
        let correct = combined_index_radix(2, 3).unwrap();
        assert_eq!(correct, 4);
        assert!(attempt_with_witness(&r_key, &r_other, &s_key, &s_other, &witness, correct).is_ok());
        // Wrong radix (next_pow2(n_R) = 2) rejects.
        assert!(attempt_with_witness(&r_key, &r_other, &s_key, &s_other, &witness, 2).is_err());
    }

    #[test]
    fn rob_peter_omit_and_duplicate_is_rejected() {
        // Full 2x2 grid over key 5 (m_J = 4). Keep the honest count/U/w but omit (1,1)
        // and duplicate (0,0): every row still a valid pair, count still 4, but the
        // duplicated (rho_l,rho_r) makes i_eval repeat -> monotonicity rejects.
        let r_key = frs(&[5, 5]);
        let r_other = vec![frs(&[1, 2])];
        let s_key = frs(&[5, 5]);
        let s_other = vec![frs(&[10, 20])];
        let (mut witness, _) = honest_setup(&r_key, &r_other, &s_key, &s_other);
        assert_eq!(witness.m_j, 4);
        witness.r_other_out = vec![frs(&[1, 1, 1, 2])];
        witness.key = frs(&[5, 5, 5, 5]);
        witness.s_other_out = vec![frs(&[10, 10, 20, 10])];
        witness.rho_l = frs(&[0, 0, 0, 1]);
        witness.rho_r = frs(&[0, 0, 1, 0]); // (0,0) duplicated, (1,1) omitted
        let radix = combined_index_radix(2, 2).unwrap();
        assert!(attempt_with_witness(&r_key, &r_other, &s_key, &s_other, &witness, radix).is_err());
    }

    #[test]
    fn within_group_swap_is_rejected() {
        // Same marginals/count, every row a valid pair, but the swap DUPLICATES a
        // (rho_l,rho_r) pair -> i_eval repeats -> monotonicity rejects.
        let r_key = frs(&[5, 5]);
        let r_other = vec![frs(&[1, 2])];
        let s_key = frs(&[5, 5]);
        let s_other = vec![frs(&[10, 20])];
        let (mut witness, _) = honest_setup(&r_key, &r_other, &s_key, &s_other);
        // {(0,0)x2, (1,1)x2}: same source multiplicities as the honest grid.
        witness.r_other_out = vec![frs(&[1, 1, 2, 2])];
        witness.key = frs(&[5, 5, 5, 5]);
        witness.s_other_out = vec![frs(&[10, 10, 20, 20])];
        witness.rho_l = frs(&[0, 0, 1, 1]);
        witness.rho_r = frs(&[0, 0, 1, 1]);
        let radix = combined_index_radix(2, 2).unwrap();
        assert!(attempt_with_witness(&r_key, &r_other, &s_key, &s_other, &witness, radix).is_err());
    }

    #[test]
    fn omission_via_count_is_rejected() {
        // Drop a genuine pair but keep the honest U/w (Σ w_L*w_R = 2 != claimed 1).
        let (r_key, r_other, s_key, s_other) = relations();
        let (mut witness, _) = honest_setup(&r_key, &r_other, &s_key, &s_other);
        witness.r_other_out = vec![frs(&[1])];
        witness.key = frs(&[10]);
        witness.s_other_out = vec![frs(&[100])];
        witness.rho_l = frs(&[0]);
        witness.rho_r = frs(&[0]);
        witness.m_j = 1;
        let radix = combined_index_radix(r_key.len(), s_key.len()).unwrap();
        assert!(attempt_with_witness(&r_key, &r_other, &s_key, &s_other, &witness, radix).is_err());
    }

    #[test]
    fn duplicate_key_stashed_in_u_padding_is_rejected() {
        // R,S all key=5 (n_R=n_S=2). Honest: U=[5] (n_U=1), w_L=w_R=[2], m_J=4. Attack:
        // claim n_U=1 but commit U=[5,5] (a duplicate 5 STASHED at padding index 1),
        // split the weight, and output 2 rows claiming m_J=2. The count is masked to 2
        // (chi_{n_U=1} hides the stash), but the U side of membership is masked too, so
        // the split undercount for r_key's two 5's -> membership REJECTS.
        let r_key = frs(&[5, 5]);
        let r_other = vec![frs(&[1, 2])];
        let s_key = frs(&[5, 5]);
        let s_other = vec![frs(&[10, 20])];
        let (mut witness, _) = honest_setup(&r_key, &r_other, &s_key, &s_other);
        assert_eq!(witness.u, frs(&[5]));
        witness.u = frs(&[5, 5]); // stash a duplicate 5 at index 1
        witness.n_u = 1; // ... but CLAIM only 1 distinct key
        witness.w_l = frs(&[1, 1]); // split weight
        witness.w_r = frs(&[2, 2]);
        witness.r_other_out = vec![frs(&[1, 2])];
        witness.key = frs(&[5, 5]);
        witness.s_other_out = vec![frs(&[10, 20])];
        witness.rho_l = frs(&[0, 1]);
        witness.rho_r = frs(&[0, 1]);
        witness.m_j = 2;
        let radix = combined_index_radix(2, 2).unwrap();
        assert!(attempt_with_witness(&r_key, &r_other, &s_key, &s_other, &witness, radix).is_err());
    }

    #[test]
    fn substitution_pointing_at_a_padding_r_row_is_rejected() {
        // Distinct-valid-pairs gadget directly. R real = {(key=5, emp=1)} (n_R=1),
        // committed padded with a FAKE R-row (key=5, emp=99) at padding index 1;
        // S = {(5,10)} (n_S=1). A fabricated output row (emp=99, key=5, name=10) points
        // rho_l=1 at the fake padding row, with a full-domain multiplicity that lights
        // source row 1. The source side is masked by chi_{n_R=1}, so the fabricated
        // candidate pole is unmatched -> REJECT (and rho_l is forced < n_R).
        let n = 4usize;
        let mk = |vals: &[(usize, u64)]| {
            let mut v = vec![f0(); n];
            for (i, x) in vals {
                v[*i] = ArkFr::from_u64(*x);
            }
            v
        };
        let cols = JoinColumns {
            r_key: mk(&[(0, 5), (1, 5)]),   // real row 0; fake padding row 1 (key 5)
            r_other: vec![mk(&[(0, 1), (1, 99)])], // emp: real 1; fake 99
            s_key: mk(&[(0, 5)]),
            s_other: vec![mk(&[(0, 10)])],
            r_other_out: vec![mk(&[(0, 1), (1, 99)])], // fabricated emp=99 in row 1
            key: mk(&[(0, 5), (1, 5)]),
            s_other_out: vec![mk(&[(0, 10), (1, 10)])],
            rho_l: mk(&[(0, 0), (1, 1)]), // row 1 points at the padding R-row
            rho_r: mk(&[(0, 0), (1, 0)]),
        };
        let com = commit_cols(&cols);
        let (n_r, n_s, m_j) = (1usize, 1usize, 2usize);
        // Malicious multiplicities over the FULL domain (light source row 1).
        let mult_l = source_multiplicities(&cols.rho_l, n, m_j, n);
        let mult_r = source_multiplicities(&cols.rho_r, n, m_j, n);
        assert_eq!(mult_l[1], ArkFr::from_u64(1)); // padding source row is lit
        let radix = combined_index_radix(n_r, n_s).unwrap();
        let mut pt = transcript();
        let proof =
            prove_binary_pairs(&cols, &mult_l, &mult_r, n_r, n_s, m_j, radix, &com, &mut pt).unwrap();
        let mut vt = transcript();
        assert!(verify_binary_pairs(&com, n_r, n_s, m_j, radix, &proof, &mut vt).is_err());

        // Sanity: the honest single-row output (no padding row lit) verifies.
        let honest_cols = JoinColumns {
            r_other_out: vec![mk(&[(0, 1)])],
            key: mk(&[(0, 5)]),
            s_other_out: vec![mk(&[(0, 10)])],
            rho_l: mk(&[(0, 0)]),
            rho_r: mk(&[(0, 0)]),
            ..cols.clone()
        };
        let honest_com = commit_cols(&honest_cols);
        let hmult_l = source_multiplicities(&honest_cols.rho_l, n_r, 1, n);
        let hmult_r = source_multiplicities(&honest_cols.rho_r, n_s, 1, n);
        let mut pt2 = transcript();
        let honest_proof =
            prove_binary_pairs(&honest_cols, &hmult_l, &hmult_r, n_r, n_s, 1, radix, &honest_com, &mut pt2)
                .unwrap();
        let mut vt2 = transcript();
        assert!(verify_binary_pairs(&honest_com, n_r, n_s, 1, radix, &honest_proof, &mut vt2).is_ok());
    }
}
