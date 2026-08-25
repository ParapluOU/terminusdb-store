//! Binary join of two DIFFERENT relations `R`, `S` on a single-column shared key
//! (Phase 7.2). Generalizes the self-join (`proof::pairs` + `proof::join`) to two
//! relations; the self-join is the `R = S` special case. Clean-room; reuses the
//! audited masked LogUp links (source AND universe masked) and the combined-index
//! monotonicity.
//!
//! For a developer: this proves a natural-join answer is the *complete* join of two
//! committed relations — every matching `(R-row, S-row)` present, none forged or
//! duplicated. Like [`super::join`] it composes audited [gadgets](crate::proof::primer)
//! (two [`super::join_link`] [LogUp](crate::proof::primer) links + a [`super::sort`]
//! strict-increase + two [`super::membership`] + a [`super::count`]) on one Fiat–Shamir
//! [transcript](crate::proof::primer). The #1 soundness subtlety is the combined-index
//! radix `C` used to pack `(rho_l, rho_r)` injectively — see the note on
//! `combined_index_radix`. See [the primer](crate::proof::primer) for
//! [commitments](crate::proof::primer) (§3), [LogUp](crate::proof::primer) (§7), and
//! [prove → verify](crate::proof::primer) (§10).
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

use super::argument::{StoreProverContext, Transcript254};
#[cfg(test)]
use super::argument::commit_column;
use super::count::{prove_count_identity_with_context, verify_count_identity, CountProof};
use super::invalid;
use super::join_link::{prove_join_link_with_context, verify_join_link, JoinLinkProof};
use super::membership::{
    prove_membership_with_mult_with_witness_and_context, verify_membership_with_mult,
    MembershipProof,
};
use super::sort::{
    prove_strictly_sorted_bits_with_context, prove_strictly_sorted_with_context,
    verify_strictly_sorted, verify_strictly_sorted_bits, SortedProof,
};

#[inline]
fn f0() -> ArkFr {
    ArkFr::zero()
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

/// The shared power-of-two domain (`>= 2`) onto which every join column is padded.
///
/// Exposed `pub` alongside [`compute_binary_join`] so external consumers (the
/// `woql-proof` compiler's end-to-end tests) can reproduce the exact per-column
/// padding a BGP proof uses when recomputing an independent natural join.
pub fn binary_join_domain(n_r: usize, n_s: usize, m_j: usize, n_u: usize) -> io::Result<usize> {
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
    /// R's join-key column.
    pub r_key: Vec<ArkFr>,
    /// R's non-key payload columns.
    pub r_other: Vec<Vec<ArkFr>>,
    /// S's join-key column.
    pub s_key: Vec<ArkFr>,
    /// S's non-key payload columns.
    pub s_other: Vec<Vec<ArkFr>>,
    /// Output projection of R's payload (one row per matched pair).
    pub r_other_out: Vec<Vec<ArkFr>>,
    /// Output shared join-key column.
    pub key: Vec<ArkFr>,
    /// Output projection of S's payload (one row per matched pair).
    pub s_other_out: Vec<Vec<ArkFr>>,
    /// LEFT index pointers (each output row's source R-row index).
    pub rho_l: Vec<ArkFr>,
    /// RIGHT index pointers (each output row's source S-row index).
    pub rho_r: Vec<ArkFr>,
}

/// The per-column commitments of a binary join (all committed at the shared domain).
/// Fields mirror [`JoinColumns`] one-for-one, plus the key universe and its
/// multiplicity columns.
#[derive(Clone, Debug)]
pub struct BinaryJoinCommitments {
    /// Commitment to R's join-key column.
    pub r_key: ArkGT,
    /// Commitments to R's payload columns.
    pub r_other: Vec<ArkGT>,
    /// Commitment to S's join-key column.
    pub s_key: ArkGT,
    /// Commitments to S's payload columns.
    pub s_other: Vec<ArkGT>,
    /// Commitments to the output projection of R's payload.
    pub r_other_out: Vec<ArkGT>,
    /// Commitment to the output shared join-key column.
    pub key: ArkGT,
    /// Commitments to the output projection of S's payload.
    pub s_other_out: Vec<ArkGT>,
    /// Commitment to the LEFT index-pointer column.
    pub rho_l: ArkGT,
    /// Commitment to the RIGHT index-pointer column.
    pub rho_r: ArkGT,
    /// Commitment to the join-key universe `U` (sorted distinct union of `r_key`, `s_key`).
    pub u: ArkGT,
    /// Commitment to the LEFT per-key multiplicity column `w_L`.
    pub w_l: ArkGT,
    /// Commitment to the RIGHT per-key multiplicity column `w_R`.
    pub w_r: ArkGT,
}

/// A distinct-valid-pairs proof for the binary join (LogUp-GKR): the two committed
/// source-multiplicity commitments, the LEFT + RIGHT keyed-join links (each a
/// block-disjoint GKR fraction tree + committed-column opening + co-running
/// `(1 - chi_{n_R|n_S})*mult = 0` Identity; see [`JoinLinkProof`]), and the
/// combined-index strict-monotonicity subproof. The four committed reciprocal columns
/// (`rec_src_l`/`rec_cand_l`/`rec_src_r`/`rec_cand_r`) and the batched membership
/// sumcheck are GONE.
#[derive(Clone, Debug)]
pub struct BinaryPairsProof {
    /// Commitment to the LEFT source-multiplicity column (fan-out of each R-row via `rho_l`).
    pub mult_l_commitment: ArkGT,
    /// Commitment to the RIGHT source-multiplicity column (fan-out of each S-row via `rho_r`).
    pub mult_r_commitment: ArkGT,
    /// LEFT keyed-join link: each output row's `(r_other_out.., key)` copies an R-row.
    pub link_l: JoinLinkProof,
    /// RIGHT keyed-join link: each output row's `(key, s_other_out..)` copies an S-row.
    pub link_r: JoinLinkProof,
    /// Strict-monotonicity proof on the combined index `C*rho_l + rho_r` (pairs distinct).
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
fn prove_binary_pairs_with_context(
    context: &StoreProverContext<'_>,
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

    let mult_l_commitment = context.commit_column(mult_l_in)?;
    let mult_r_commitment = context.commit_column(mult_r_in)?;

    // Round 1: bind the data columns + multiplicities, draw fresh challenges. NO reciprocal
    // column is committed anymore (both links are GKR fraction trees over uncommitted
    // intermediates), so alpha/beta are drawn here AFTER every data-column commitment — the
    // primitive's caller contract (leaves fixed before any per-layer GKR challenge).
    bind_pairs_data(transcript, com);
    transcript.append_group(b"pow/join2/com-mult-l", &mult_l_commitment);
    transcript.append_group(b"pow/join2/com-mult-r", &mult_r_commitment);
    let beta_l = transcript.challenge_scalar(b"pow/join2/left-beta");
    let alpha_l = transcript.challenge_scalar(b"pow/join2/left-alpha");
    let beta_r = transcript.challenge_scalar(b"pow/join2/right-beta");
    let alpha_r = transcript.challenge_scalar(b"pow/join2/right-alpha");

    // LEFT link: candidate (r_other_out.., key, rho_l) in source {(r_other.., r_key, idx)}
    // weighted by mult_l (source masked chi_{n_R}, candidate masked chi_{m_J}).
    let mut left_src: Vec<&[ArkFr]> = slices(&cols.r_other);
    left_src.push(&cols.r_key);
    let mut left_cand: Vec<&[ArkFr]> = slices(&cols.r_other_out);
    left_cand.push(&cols.key);
    left_cand.push(&cols.rho_l);
    let link_l = prove_join_link_with_context(
        context, &left_src, mult_l_in, &left_cand, n_r, m_j, alpha_l, beta_l, transcript,
    )?;

    // RIGHT link: candidate (key, s_other_out.., rho_r) in source {(s_key, s_other.., idx)}
    // weighted by mult_r.
    let mut right_src: Vec<&[ArkFr]> = vec![cols.s_key.as_slice()];
    right_src.extend(slices(&cols.s_other));
    let mut right_cand: Vec<&[ArkFr]> = vec![cols.key.as_slice()];
    right_cand.extend(slices(&cols.s_other_out));
    right_cand.push(&cols.rho_r);
    let link_r = prove_join_link_with_context(
        context, &right_src, mult_r_in, &right_cand, n_s, m_j, alpha_r, beta_r, transcript,
    )?;

    // Combined-index strict monotonicity, homomorphic com(i_eval).
    let radix_fr = fr(radix);
    let i_eval: Vec<ArkFr> = (0..n)
        .map(|i| radix_fr * cols.rho_l[i] + cols.rho_r[i])
        .collect();
    let com_i_eval = com.rho_l.scale(&radix_fr).add(&com.rho_r);
    let sorted =
        prove_strictly_sorted_with_context(context, &i_eval, m_j, &com_i_eval, transcript)?;

    Ok(BinaryPairsProof {
        mult_l_commitment,
        mult_r_commitment,
        link_l,
        link_r,
        sorted,
    })
}

#[cfg(test)]
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
    prove_binary_pairs_with_context(
        &StoreProverContext::cpu(),
        cols,
        mult_l_in,
        mult_r_in,
        n_r,
        n_s,
        m_j,
        radix,
        com,
        transcript,
    )
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
    // The domain is re-derived from the LEFT link's GKR tree (k + 1 layers over a
    // 2*domain leaf vector); the RIGHT link's own layer count is cross-checked to `k`
    // inside `verify_join_link`, and both links re-anchor `k` via their column openings.
    let num_vars = proof.link_l.gkr.num_vars();
    if num_vars < 2 || num_vars >= usize::BITS as usize {
        return Err(invalid(
            "binary pairs proof has an implausible variable count",
        ));
    }
    let k = num_vars - 1;
    let n = 1usize << k;
    if n_r > n || n_s > n || m_j > n {
        return Err(invalid("binary join real length exceeds the domain"));
    }
    let p = com.r_other.len();
    let q = com.s_other.len();
    if com.r_other_out.len() != p || com.s_other_out.len() != q {
        return Err(invalid("binary join output arity mismatch"));
    }

    bind_pairs_data(transcript, com);
    transcript.append_group(b"pow/join2/com-mult-l", &proof.mult_l_commitment);
    transcript.append_group(b"pow/join2/com-mult-r", &proof.mult_r_commitment);
    let beta_l = transcript.challenge_scalar(b"pow/join2/left-beta");
    let alpha_l = transcript.challenge_scalar(b"pow/join2/left-alpha");
    let beta_r = transcript.challenge_scalar(b"pow/join2/right-beta");
    let alpha_r = transcript.challenge_scalar(b"pow/join2/right-alpha");

    // LEFT link: candidate (r_other_out.., key, rho_l) in source {(r_other.., r_key, idx)}.
    let mut left_src_coms: Vec<ArkGT> = com.r_other.clone();
    left_src_coms.push(com.r_key);
    let mut left_cand_coms: Vec<ArkGT> = com.r_other_out.clone();
    left_cand_coms.push(com.key);
    left_cand_coms.push(com.rho_l);
    verify_join_link(
        &left_src_coms, &proof.mult_l_commitment, &left_cand_coms, n_r, m_j, alpha_l, beta_l, k,
        &proof.link_l, transcript,
    )?;

    // RIGHT link: candidate (key, s_other_out.., rho_r) in source {(s_key, s_other.., idx)}.
    let mut right_src_coms: Vec<ArkGT> = vec![com.s_key];
    right_src_coms.extend(com.s_other.iter().copied());
    let mut right_cand_coms: Vec<ArkGT> = vec![com.key];
    right_cand_coms.extend(com.s_other_out.iter().copied());
    right_cand_coms.push(com.rho_r);
    verify_join_link(
        &right_src_coms, &proof.mult_r_commitment, &right_cand_coms, n_s, m_j, alpha_r, beta_r, k,
        &proof.link_r, transcript,
    )?;

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
    /// Output projection of R's payload columns.
    pub r_other_out: Vec<Vec<ArkFr>>,
    /// Output shared join-key column.
    pub key: Vec<ArkFr>,
    /// Output projection of S's payload columns.
    pub s_other_out: Vec<Vec<ArkFr>>,
    /// LEFT index pointers (source R-row per output row).
    pub rho_l: Vec<ArkFr>,
    /// RIGHT index pointers (source S-row per output row).
    pub rho_r: Vec<ArkFr>,
    /// The join-key universe (sorted distinct union of `r_key` and `s_key`).
    pub u: Vec<ArkFr>,
    /// LEFT per-key multiplicity `w_L[i] = #{ a : r_key[a] == u[i] }`.
    pub w_l: Vec<ArkFr>,
    /// RIGHT per-key multiplicity `w_R[i] = #{ b : s_key[b] == u[i] }`.
    pub w_r: Vec<ArkFr>,
    /// Claimed distinct-key count (the chi real length; see the type doc).
    pub n_u: usize,
    /// Output row count (number of matched pairs).
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
    binary_join_commitments_with_context(
        &StoreProverContext::cpu(), r_key, r_other, s_key, s_other, witness,
    )
}

/// As [`binary_join_commitments`], with an explicit prover-arithmetic context.
pub fn binary_join_commitments_with_context(
    context: &StoreProverContext<'_>,
    r_key: &[ArkFr],
    r_other: &[&[ArkFr]],
    s_key: &[ArkFr],
    s_other: &[&[ArkFr]],
    witness: &BinaryJoinWitness,
) -> io::Result<BinaryJoinCommitments> {
    let n = binary_join_domain(r_key.len(), s_key.len(), witness.m_j, witness.n_u)?;
    let commit_vec = |cols: &[Vec<ArkFr>]| -> io::Result<Vec<ArkGT>> {
        cols.iter().map(|c| context.commit_column(&padded(c, n))).collect()
    };
    let commit_ref_vec = |cols: &[&[ArkFr]]| -> io::Result<Vec<ArkGT>> {
        cols.iter().map(|c| context.commit_column(&padded(c, n))).collect()
    };
    Ok(BinaryJoinCommitments {
        r_key: context.commit_column(&padded(r_key, n))?,
        r_other: commit_ref_vec(r_other)?,
        s_key: context.commit_column(&padded(s_key, n))?,
        s_other: commit_ref_vec(s_other)?,
        r_other_out: commit_vec(&witness.r_other_out)?,
        key: context.commit_column(&padded(&witness.key, n))?,
        s_other_out: commit_vec(&witness.s_other_out)?,
        rho_l: context.commit_column(&padded(&witness.rho_l, n))?,
        rho_r: context.commit_column(&padded(&witness.rho_r, n))?,
        u: context.commit_column(&padded(&witness.u, n))?,
        w_l: context.commit_column(&padded(&witness.w_l, n))?,
        w_r: context.commit_column(&padded(&witness.w_r, n))?,
    })
}

/// A full sound+complete binary-join proof.
#[derive(Clone, Debug)]
pub struct BinaryJoinProof {
    /// The output is a set of distinct genuine key-matching `(R, S)` pairs.
    pub pairs: BinaryPairsProof,
    /// The key universe `U` is strictly increasing, hence distinct.
    pub sorted_u: SortedProof,
    /// `r_key` is the multiset image of `U` under `w_L`.
    pub membership_l: MembershipProof,
    /// `s_key` is the multiset image of `U` under `w_R`.
    pub membership_r: MembershipProof,
    /// `Sum_i w_L[i]*w_R[i] = m_J`: the pair count equals the output size.
    pub count: CountProof,
    /// The distinct-key count bound into the count/membership chi masks.
    pub n_u: usize,
}

impl BinaryJoinProof {
    /// `log2` of the join domain `d_j`. The chain verifier cross-checks this against
    /// each stage's relocation TARGET domain so a stage cannot run its join over a
    /// different domain than the one its input columns were relocated into. Derived
    /// from the LEFT keyed-join link's GKR tree, which has `k + 1` layers over a
    /// `2*domain = 2^{k+1}` leaf vector, so `num_vars = (k + 1) - 1 = k = log2(domain)`
    /// (the same value the old committed-reciprocal membership sumcheck reported).
    pub fn num_vars(&self) -> usize {
        self.pairs.link_l.gkr.num_vars().saturating_sub(1)
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
    key_bits: usize,
) {
    transcript.append_bytes(LABEL_DOMAIN, b"parture/proof-of-woql/binary-join/v0");
    for v in [n_r, n_s, n_u, m_j] {
        transcript.append_bytes(LABEL_LEN, &(v as u64).to_le_bytes());
    }
    transcript.append_bytes(b"pow/join2/key-bits", &(key_bits as u64).to_le_bytes());
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
    prove_binary_join_with_context(
        &StoreProverContext::cpu(), r_key, r_other, s_key, s_other, com, transcript,
    )
}

/// As [`prove_binary_join`], with an explicit prover-arithmetic context (64-bit keys).
pub fn prove_binary_join_with_context(
    context: &StoreProverContext<'_>,
    r_key: &[ArkFr],
    r_other: &[&[ArkFr]],
    s_key: &[ArkFr],
    s_other: &[&[ArkFr]],
    com: &BinaryJoinCommitments,
    transcript: &mut Transcript254,
) -> io::Result<(BinaryJoinProof, Vec<ArkGT>, usize)> {
    prove_binary_join_with_key_bits_and_context(
        context, r_key, r_other, s_key, s_other, 64, com, transcript,
    )
}

/// Prove a join whose canonical integer key representation has the given bit width.
pub fn prove_binary_join_with_key_bits(
    r_key: &[ArkFr],
    r_other: &[&[ArkFr]],
    s_key: &[ArkFr],
    s_other: &[&[ArkFr]],
    key_bits: usize,
    com: &BinaryJoinCommitments,
    transcript: &mut Transcript254,
) -> io::Result<(BinaryJoinProof, Vec<ArkGT>, usize)> {
    prove_binary_join_with_key_bits_and_context(
        &StoreProverContext::cpu(), r_key, r_other, s_key, s_other, key_bits, com, transcript,
    )
}

/// As [`prove_binary_join_with_key_bits`], with an explicit prover-arithmetic context.
#[allow(clippy::too_many_arguments)]
pub fn prove_binary_join_with_key_bits_and_context(
    context: &StoreProverContext<'_>,
    r_key: &[ArkFr],
    r_other: &[&[ArkFr]],
    s_key: &[ArkFr],
    s_other: &[&[ArkFr]],
    key_bits: usize,
    com: &BinaryJoinCommitments,
    transcript: &mut Transcript254,
) -> io::Result<(BinaryJoinProof, Vec<ArkGT>, usize)> {
    let witness = compute_binary_join(r_key, r_other, s_key, s_other);
    let radix = combined_index_radix(r_key.len(), s_key.len())?;
    let proof = prove_binary_join_with_witness_and_key_bits(
        context, r_key, r_other, s_key, s_other, &witness, radix, key_bits, com, transcript,
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
#[allow(dead_code)]
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
    prove_binary_join_with_witness_and_key_bits(
        &StoreProverContext::cpu(), r_key, r_other, s_key, s_other, witness, radix, 64, com,
        transcript,
    )
}

#[allow(clippy::too_many_arguments)]
fn prove_binary_join_with_witness_and_key_bits(
    context: &StoreProverContext<'_>,
    r_key: &[ArkFr],
    r_other: &[&[ArkFr]],
    s_key: &[ArkFr],
    s_other: &[&[ArkFr]],
    witness: &BinaryJoinWitness,
    radix: usize,
    key_bits: usize,
    com: &BinaryJoinCommitments,
    transcript: &mut Transcript254,
) -> io::Result<BinaryJoinProof> {
    let (n_r, n_s) = (r_key.len(), s_key.len());
    let n_u = witness.n_u;
    let m_j = witness.m_j;
    let n = binary_join_domain(n_r, n_s, m_j, n_u)?;

    bind_instance(transcript, com, n_r, n_s, n_u, m_j, key_bits);

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

    let pairs = prove_binary_pairs_with_context(
        context, &cols, &mult_l, &mult_r, n_r, n_s, m_j, radix, com, transcript,
    )?;

    // U distinctness over [0, n_U).
    let u_full = padded(&witness.u, n);
    let sorted_u = prove_strictly_sorted_bits_with_context(
        context, &u_full, n_u, key_bits, &com.u, transcript,
    )?;
    // r_key is the image of U under w_L, s_key the image under w_R (both sides masked).
    let w_l_full = padded(&witness.w_l, n);
    let w_r_full = padded(&witness.w_r, n);
    let membership_l = prove_membership_with_mult_with_witness_and_context(
        context, &u_full,
        &cols.r_key,
        &w_l_full,
        n_u,
        n_r,
        &com.u,
        &com.r_key,
        &com.w_l,
        transcript,
    )?;
    let membership_r = prove_membership_with_mult_with_witness_and_context(
        context, &u_full,
        &cols.s_key,
        &w_r_full,
        n_u,
        n_s,
        &com.u,
        &com.s_key,
        &com.w_r,
        transcript,
    )?;
    // Count Σ chi_{n_U} w_L*w_R = m_J (true n_U -- audit fix).
    let count = prove_count_identity_with_context(
        context, &w_l_full, &w_r_full, n_u, m_j, &com.w_l, &com.w_r, transcript,
    )?;

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
    verify_binary_join_with_key_bits(com, n_r, n_s, m_j, 64, proof, transcript)
}

/// Verify a join using the public integer width of its canonical key representation.
pub fn verify_binary_join_with_key_bits(
    com: &BinaryJoinCommitments,
    n_r: usize,
    n_s: usize,
    m_j: usize,
    key_bits: usize,
    proof: &BinaryJoinProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    let radix = combined_index_radix(n_r, n_s)?;
    verify_binary_join_inner(
        com, n_r, n_s, proof.n_u, m_j, radix, key_bits, proof, transcript,
    )
}

#[allow(clippy::too_many_arguments)]
fn verify_binary_join_inner(
    com: &BinaryJoinCommitments,
    n_r: usize,
    n_s: usize,
    n_u: usize,
    m_j: usize,
    radix: usize,
    key_bits: usize,
    proof: &BinaryJoinProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    bind_instance(transcript, com, n_r, n_s, n_u, m_j, key_bits);
    verify_binary_pairs(com, n_r, n_s, m_j, radix, &proof.pairs, transcript)
        .map_err(|error| invalid(format!("binary pair verification failed: {error}")))?;
    verify_strictly_sorted_bits(&com.u, n_u, key_bits, &proof.sorted_u, transcript)
        .map_err(|error| invalid(format!("join-key universe sort failed: {error}")))?;
    verify_membership_with_mult(
        &com.u,
        &com.r_key,
        &com.w_l,
        n_u,
        n_r,
        &proof.membership_l,
        transcript,
    )
    .map_err(|error| invalid(format!("left join-key membership failed: {error}")))?;
    verify_membership_with_mult(
        &com.u,
        &com.s_key,
        &com.w_r,
        n_u,
        n_s,
        &proof.membership_r,
        transcript,
    )
    .map_err(|error| invalid(format!("right join-key membership failed: {error}")))?;
    verify_count_identity(&com.w_l, &com.w_r, n_u, m_j, &proof.count, transcript)
        .map_err(|error| invalid(format!("join completeness count failed: {error}")))?;
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
        assert_eq!(
            witness.s_other_out,
            vec![frs(&[100, 100]), frs(&[111, 111])]
        );
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
            64,
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
        assert!(
            attempt_with_witness(&r_key, &r_other, &s_key, &s_other, &witness, correct).is_ok()
        );
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
            r_key: mk(&[(0, 5), (1, 5)]), // real row 0; fake padding row 1 (key 5)
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
            prove_binary_pairs(&cols, &mult_l, &mult_r, n_r, n_s, m_j, radix, &com, &mut pt)
                .unwrap();
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
        let honest_proof = prove_binary_pairs(
            &honest_cols,
            &hmult_l,
            &hmult_r,
            n_r,
            n_s,
            1,
            radix,
            &honest_com,
            &mut pt2,
        )
        .unwrap();
        let mut vt2 = transcript();
        assert!(
            verify_binary_pairs(&honest_com, n_r, n_s, 1, radix, &honest_proof, &mut vt2).is_ok()
        );
    }
}
