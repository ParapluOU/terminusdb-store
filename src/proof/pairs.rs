//! SOUND self-join completeness core (Phase 2c-2f): distinct genuine key-matching
//! pairs. Adopts Proof-of-SQL's production approach (index-pointer membership +
//! combined-index strict monotonicity), which is sound where our 2c-2d gather was
//! not (the gather only pinned MARGINAL usage counts, which do not determine the
//! bipartite pairing; a within-group swap keeps the marginals but changes the
//! matrix). Read `sort_merge_join_exec.rs` / `membership_check.rs` closely;
//! reimplemented here on our primitives. No COSL code.
//!
//! Given `F = (Sf, Of)` (n_F rows) and a claimed self-join output `(X, Y, Z)` with
//! committed INDEX-POINTER columns `rho_l, rho_r` (each in `[0, n_F)`), `m_J` real
//! rows, prove the output is a SET of DISTINCT genuine key-matching
//! (left-F-row, right-F-row) pairs:
//!
//! 1. LEFT index-augmented membership: `(X[k], Y[k], rho_l[k])` is a member of
//!    `{ (Sf_a, Of_a, idx_a) : a<n_F }` (3-column fingerprint with `Operand::Idx`
//!    for the F index; source side weighted by a committed multiplicity
//!    `mult_l_in`, candidate side masked by `chi_{m_J}`). Because `idx` is
//!    distinct, this forces `rho_l[k]` to point to the exact F-row and `(X,Y)` to
//!    equal it.
//! 2. RIGHT index-augmented membership: `(Y[k], Z[k], rho_r[k])` in
//!    `{ (Sf_b, Of_b, idx_b) }`. The SHARED committed `Y` column links the two
//!    (Y = Of[rho_l] = Sf[rho_r]), so each output row is a valid key-match.
//! 3. COMBINED-INDEX strict monotonicity: `i_eval[k] = C*rho_l[k] + rho_r[k]` is
//!    STRICTLY INCREASING over the `m_J` real rows (via 2c-2a
//!    `prove_strictly_sorted`). `C = next_pow2(n_F) >= n_F`, so with `rho_r < n_F
//!    <= C` the map `(rho_l, rho_r) -> i_eval` is INJECTIVE; strict monotonicity
//!    then forces all `(rho_l, rho_r)` pairs DISTINCT -> no duplicate output pairs.
//!    (MVP assumes `n_F <= 2^32` so `i_eval < C*n_F <= 2^64` and the 64-bit range
//!    check applies. The prover orders the output by `i_eval` ascending; the result
//!    multiset is unchanged.) `com(i_eval)` is derived homomorphically as
//!    `C*com(rho_l) + com(rho_r)` -- no extra commitment.
//!
//! This subargument proves the output is a SET of distinct genuine key-matching
//! F-pairs (subsuming 2c-1's plain memberships). `proof::join` composes it with
//! U-distinctness, U-membership, and `count m_J = Sum w_L*w_R` to force these
//! distinct valid pairs to be ALL of them.

use std::io;

use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::{Field, Group};
use dory_pcs::primitives::transcript::Transcript;

use super::argument::{
    prove_sumcheck, verify_column_openings, verify_sumcheck,
    ClaimShape, ColumnOpening, Expr, Operand, StoreProverContext, SumcheckClaim, SumcheckProof,
    SumcheckShape, Term, Transcript254,
};
#[cfg(test)]
use super::argument::commit_column;
use super::invalid;
use super::logup::{masked_multiset_link_claim, reciprocal_witness, rlc_fold};
use super::sort::{prove_strictly_sorted_with_context, verify_strictly_sorted, SortedProof};

const COL_SF: usize = 0;
const COL_OF: usize = 1;
const COL_X: usize = 2;
const COL_Y: usize = 3;
const COL_Z: usize = 4;
const COL_RHO_L: usize = 5;
const COL_RHO_R: usize = 6;
const COL_MULT_L_IN: usize = 7;
const COL_MULT_R_IN: usize = 8;
const COL_REC_SRC_L: usize = 9;
const COL_REC_CAND_L: usize = 10;
const COL_REC_SRC_R: usize = 11;
const COL_REC_CAND_R: usize = 12;
const NUM_COLUMNS: usize = 13;
// chi_bounds indices: chi_{m_J} (candidate/output real length) and chi_{n_F} (source
// real length; masks padding F-rows out of the SOURCE side of each link).
const CHI_MJ: usize = 0;
const CHI_NF: usize = 1;

const LABEL_BETA_L: &[u8] = b"pow/pairs/left-beta";
const LABEL_ALPHA_L: &[u8] = b"pow/pairs/left-alpha";
const LABEL_BETA_R: &[u8] = b"pow/pairs/right-beta";
const LABEL_ALPHA_R: &[u8] = b"pow/pairs/right-alpha";

#[inline]
fn f0() -> ArkFr {
    ArkFr::zero()
}
#[inline]
fn f1() -> ArkFr {
    ArkFr::one()
}

/// A distinct-valid-pairs proof: the batched index-augmented membership sumcheck
/// (left + right) with its batched Dory opening, the multiplicity + reciprocal
/// commitments, and the combined-index strict-monotonicity subproof.
#[derive(Clone, Debug)]
pub struct PairsProof {
    pub membership_sumcheck: SumcheckProof,
    pub membership_opening: ColumnOpening,
    pub mult_l_in_commitment: ArkGT,
    pub mult_r_in_commitment: ArkGT,
    pub rec_src_l_commitment: ArkGT,
    pub rec_cand_l_commitment: ArkGT,
    pub rec_src_r_commitment: ArkGT,
    pub rec_cand_r_commitment: ArkGT,
    pub sorted: SortedProof,
}

/// The common power-of-two domain (`>= 2`) for `n_f` F-rows and `m_j` output rows.
pub(crate) fn pairs_domain(n_f: usize, m_j: usize) -> io::Result<usize> {
    n_f.max(m_j)
        .checked_next_power_of_two()
        .ok_or_else(|| invalid("pairs domain size overflow"))
        .map(|d| d.max(2))
}

/// The injective combined-index radix `C = next_pow2(n_F) >= n_F` (so `rho_r < C`).
pub(crate) fn combined_index_radix(n_f: usize) -> io::Result<usize> {
    n_f.checked_next_power_of_two()
        .ok_or_else(|| invalid("combined-index radix overflow"))
        .map(|c| c.max(1))
}

fn padded(column: &[ArkFr], n: usize) -> Vec<ArkFr> {
    let mut out = column.to_vec();
    out.resize(n, f0());
    out
}

/// `rec_col * (alpha - Sum coeff*operand) - 1 = 0`, an Identity claim pinning a
/// committed reciprocal over an arbitrary multilinear fold (columns and `Idx`).
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

#[allow(clippy::too_many_arguments)]
fn membership_shape(
    k: usize,
    m_j: usize,
    n_f: usize,
    alpha_l: ArkFr,
    beta_l: ArkFr,
    alpha_r: ArkFr,
    beta_r: ArkFr,
) -> SumcheckShape {
    let beta_l2 = beta_l * beta_l;
    let beta_r2 = beta_r * beta_r;
    // LEFT: source (Sf, Of, idx); candidate (X, Y, rho_l).
    let src_l = reciprocal_fold_claim(
        COL_REC_SRC_L,
        &[
            (f1(), Operand::Column(COL_SF)),
            (beta_l, Operand::Column(COL_OF)),
            (beta_l2, Operand::Idx),
        ],
        alpha_l,
    );
    let cand_l = reciprocal_fold_claim(
        COL_REC_CAND_L,
        &[
            (f1(), Operand::Column(COL_X)),
            (beta_l, Operand::Column(COL_Y)),
            (beta_l2, Operand::Column(COL_RHO_L)),
        ],
        alpha_l,
    );
    let link_l = masked_multiset_link_claim(
        &[Operand::Chi(CHI_MJ)],
        COL_REC_CAND_L,
        &[Operand::Chi(CHI_NF), Operand::Column(COL_MULT_L_IN)],
        COL_REC_SRC_L,
    );
    // RIGHT: source (Sf, Of, idx); candidate (Y, Z, rho_r).
    let src_r = reciprocal_fold_claim(
        COL_REC_SRC_R,
        &[
            (f1(), Operand::Column(COL_SF)),
            (beta_r, Operand::Column(COL_OF)),
            (beta_r2, Operand::Idx),
        ],
        alpha_r,
    );
    let cand_r = reciprocal_fold_claim(
        COL_REC_CAND_R,
        &[
            (f1(), Operand::Column(COL_Y)),
            (beta_r, Operand::Column(COL_Z)),
            (beta_r2, Operand::Column(COL_RHO_R)),
        ],
        alpha_r,
    );
    let link_r = masked_multiset_link_claim(
        &[Operand::Chi(CHI_MJ)],
        COL_REC_CAND_R,
        &[Operand::Chi(CHI_NF), Operand::Column(COL_MULT_R_IN)],
        COL_REC_SRC_R,
    );
    SumcheckShape {
        num_vars: k,
        num_columns: NUM_COLUMNS,
        chi_bounds: vec![m_j, n_f],
        claims: vec![src_l, cand_l, link_l, src_r, cand_r, link_r],
    }
}

/// Source multiplicity `mult[a] = #{ k < m_j : rho[k] == a }` over the domain.
fn source_multiplicities(rho: &[ArkFr], n_f: usize, m_j: usize, n: usize) -> Vec<ArkFr> {
    let mut mult = vec![f0(); n];
    for a in 0..n_f {
        let a_fr = ArkFr::from_u64(a as u64);
        let count = rho[..m_j].iter().filter(|v| **v == a_fr).count() as u64;
        mult[a] = ArkFr::from_u64(count);
    }
    mult
}

/// Prove the claimed self-join output is a SET of distinct genuine key-matching
/// F-pairs. Input columns must be committed padded to `pairs_domain(n_F, m_J)`.
#[allow(clippy::too_many_arguments)]
pub fn prove_distinct_valid_pairs(
    sf: &[ArkFr],
    of: &[ArkFr],
    x: &[ArkFr],
    y: &[ArkFr],
    z: &[ArkFr],
    rho_l: &[ArkFr],
    rho_r: &[ArkFr],
    com_sf: &ArkGT,
    com_of: &ArkGT,
    com_x: &ArkGT,
    com_y: &ArkGT,
    com_z: &ArkGT,
    com_rho_l: &ArkGT,
    com_rho_r: &ArkGT,
    transcript: &mut Transcript254,
) -> io::Result<PairsProof> {
    prove_distinct_valid_pairs_with_context(
        &StoreProverContext::cpu(), sf, of, x, y, z, rho_l, rho_r, com_sf, com_of, com_x, com_y,
        com_z, com_rho_l, com_rho_r, transcript,
    )
}

/// As [`prove_distinct_valid_pairs`], with an explicit prover-arithmetic context.
#[allow(clippy::too_many_arguments)]
pub fn prove_distinct_valid_pairs_with_context(
    context: &StoreProverContext<'_>,
    sf: &[ArkFr],
    of: &[ArkFr],
    x: &[ArkFr],
    y: &[ArkFr],
    z: &[ArkFr],
    rho_l: &[ArkFr],
    rho_r: &[ArkFr],
    com_sf: &ArkGT,
    com_of: &ArkGT,
    com_x: &ArkGT,
    com_y: &ArkGT,
    com_z: &ArkGT,
    com_rho_l: &ArkGT,
    com_rho_r: &ArkGT,
    transcript: &mut Transcript254,
) -> io::Result<PairsProof> {
    let domain = pairs_domain(sf.len(), x.len())?;
    prove_distinct_valid_pairs_in_domain_with_context(
        context, sf, of, x, y, z, rho_l, rho_r, com_sf, com_of, com_x, com_y, com_z, com_rho_l,
        com_rho_r, domain, transcript,
    )
}

/// Like [`prove_distinct_valid_pairs`] but pads every column onto a caller-chosen
/// power-of-two `domain` (`>= pairs_domain(n_F, m_J)`) instead of the natural one.
/// The full self-join integration (Phase 2c-2e) uses this so `Sf`/`Of` share ONE
/// commitment with the U-membership sub-proofs (a single committed relation across
/// the whole self-join). Real lengths (`n_F`, `m_J`) still drive the chi mask and
/// the source multiplicities, so padding rows stay inert.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn prove_distinct_valid_pairs_in_domain(
    sf: &[ArkFr],
    of: &[ArkFr],
    x: &[ArkFr],
    y: &[ArkFr],
    z: &[ArkFr],
    rho_l: &[ArkFr],
    rho_r: &[ArkFr],
    com_sf: &ArkGT,
    com_of: &ArkGT,
    com_x: &ArkGT,
    com_y: &ArkGT,
    com_z: &ArkGT,
    com_rho_l: &ArkGT,
    com_rho_r: &ArkGT,
    domain: usize,
    transcript: &mut Transcript254,
) -> io::Result<PairsProof> {
    prove_distinct_valid_pairs_in_domain_with_context(
        &StoreProverContext::cpu(),
        sf,
        of,
        x,
        y,
        z,
        rho_l,
        rho_r,
        com_sf,
        com_of,
        com_x,
        com_y,
        com_z,
        com_rho_l,
        com_rho_r,
        domain,
        transcript,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_distinct_valid_pairs_in_domain_with_context(
    context: &StoreProverContext<'_>,
    sf: &[ArkFr],
    of: &[ArkFr],
    x: &[ArkFr],
    y: &[ArkFr],
    z: &[ArkFr],
    rho_l: &[ArkFr],
    rho_r: &[ArkFr],
    com_sf: &ArkGT,
    com_of: &ArkGT,
    com_x: &ArkGT,
    com_y: &ArkGT,
    com_z: &ArkGT,
    com_rho_l: &ArkGT,
    com_rho_r: &ArkGT,
    domain: usize,
    transcript: &mut Transcript254,
) -> io::Result<PairsProof> {
    let n_f = sf.len();
    let m_j = x.len();
    if of.len() != n_f {
        return Err(invalid("F columns Sf/Of length mismatch"));
    }
    if y.len() != m_j || z.len() != m_j || rho_l.len() != m_j || rho_r.len() != m_j {
        return Err(invalid("output columns X/Y/Z/rho_l/rho_r length mismatch"));
    }
    let natural = pairs_domain(n_f, m_j)?;
    if !domain.is_power_of_two() || domain < natural {
        return Err(invalid(
            "pairs domain must be a power of two >= the natural domain",
        ));
    }
    let n = domain;

    let sf_p = padded(sf, n);
    let of_p = padded(of, n);
    let x_p = padded(x, n);
    let y_p = padded(y, n);
    let z_p = padded(z, n);
    let rho_l_p = padded(rho_l, n);
    let rho_r_p = padded(rho_r, n);

    let mult_l_in = source_multiplicities(&rho_l_p, n_f, m_j, n);
    let mult_r_in = source_multiplicities(&rho_r_p, n_f, m_j, n);

    prove_distinct_valid_pairs_with_witness_and_context(
        context, &sf_p, &of_p, &x_p, &y_p, &z_p, &rho_l_p, &rho_r_p, n_f, m_j, &mult_l_in,
        &mult_r_in, com_sf, com_of, com_x, com_y, com_z, com_rho_l, com_rho_r, transcript,
    )
}

/// Prove distinct-valid-pairs from FULLY EXPLICIT witness columns (each already
/// padded to a common power-of-two domain `n = sf_p.len()`), an explicit real
/// F-length `n_f` and output length `m_j`, and explicit source multiplicities. The
/// honest entry point derives all of these; the security tests use this to inject a
/// malicious `Sf`/`Of` padding tail, out-of-range `rho`, and a full-domain
/// multiplicity that lights up padding source rows.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn prove_distinct_valid_pairs_with_witness(
    sf_p: &[ArkFr],
    of_p: &[ArkFr],
    x_p: &[ArkFr],
    y_p: &[ArkFr],
    z_p: &[ArkFr],
    rho_l_p: &[ArkFr],
    rho_r_p: &[ArkFr],
    n_f: usize,
    m_j: usize,
    mult_l_in: &[ArkFr],
    mult_r_in: &[ArkFr],
    com_sf: &ArkGT,
    com_of: &ArkGT,
    com_x: &ArkGT,
    com_y: &ArkGT,
    com_z: &ArkGT,
    com_rho_l: &ArkGT,
    com_rho_r: &ArkGT,
    transcript: &mut Transcript254,
) -> io::Result<PairsProof> {
    prove_distinct_valid_pairs_with_witness_and_context(
        &StoreProverContext::cpu(),
        sf_p,
        of_p,
        x_p,
        y_p,
        z_p,
        rho_l_p,
        rho_r_p,
        n_f,
        m_j,
        mult_l_in,
        mult_r_in,
        com_sf,
        com_of,
        com_x,
        com_y,
        com_z,
        com_rho_l,
        com_rho_r,
        transcript,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_distinct_valid_pairs_with_witness_and_context(
    context: &StoreProverContext<'_>,
    sf_p: &[ArkFr],
    of_p: &[ArkFr],
    x_p: &[ArkFr],
    y_p: &[ArkFr],
    z_p: &[ArkFr],
    rho_l_p: &[ArkFr],
    rho_r_p: &[ArkFr],
    n_f: usize,
    m_j: usize,
    mult_l_in: &[ArkFr],
    mult_r_in: &[ArkFr],
    com_sf: &ArkGT,
    com_of: &ArkGT,
    com_x: &ArkGT,
    com_y: &ArkGT,
    com_z: &ArkGT,
    com_rho_l: &ArkGT,
    com_rho_r: &ArkGT,
    transcript: &mut Transcript254,
) -> io::Result<PairsProof> {
    let n = sf_p.len();
    if !n.is_power_of_two() || n < 2 {
        return Err(invalid("pairs domain must be a power of two >= 2"));
    }
    for column in [
        of_p, x_p, y_p, z_p, rho_l_p, rho_r_p, mult_l_in, mult_r_in,
    ] {
        if column.len() != n {
            return Err(invalid("pairs witness column length mismatch"));
        }
    }
    if n_f > n || m_j > n {
        return Err(invalid("pairs real length exceeds the domain"));
    }
    let k = n.trailing_zeros() as usize;
    let idx_p: Vec<ArkFr> = (0..n).map(|i| ArkFr::from_u64(i as u64)).collect();

    let mult_l_in_commitment = context.commit_column(mult_l_in)?;
    let mult_r_in_commitment = context.commit_column(mult_r_in)?;

    // Round 1: bind the given columns + multiplicities, draw fresh challenges.
    transcript.append_group(b"pow/pairs/com-sf", com_sf);
    transcript.append_group(b"pow/pairs/com-of", com_of);
    transcript.append_group(b"pow/pairs/com-x", com_x);
    transcript.append_group(b"pow/pairs/com-y", com_y);
    transcript.append_group(b"pow/pairs/com-z", com_z);
    transcript.append_group(b"pow/pairs/com-rho-l", com_rho_l);
    transcript.append_group(b"pow/pairs/com-rho-r", com_rho_r);
    transcript.append_group(b"pow/pairs/com-mult-l-in", &mult_l_in_commitment);
    transcript.append_group(b"pow/pairs/com-mult-r-in", &mult_r_in_commitment);
    let beta_l = transcript.challenge_scalar(LABEL_BETA_L);
    let alpha_l = transcript.challenge_scalar(LABEL_ALPHA_L);
    let beta_r = transcript.challenge_scalar(LABEL_BETA_R);
    let alpha_r = transcript.challenge_scalar(LABEL_ALPHA_R);

    // Round 2: reciprocal witnesses over the 3-column fingerprints.
    let rec_src_l = reciprocal_witness(alpha_l, &rlc_fold(beta_l, &[sf_p, of_p, &idx_p]));
    let rec_cand_l = reciprocal_witness(alpha_l, &rlc_fold(beta_l, &[x_p, y_p, rho_l_p]));
    let rec_src_r = reciprocal_witness(alpha_r, &rlc_fold(beta_r, &[sf_p, of_p, &idx_p]));
    let rec_cand_r = reciprocal_witness(alpha_r, &rlc_fold(beta_r, &[y_p, z_p, rho_r_p]));
    let rec_src_l_commitment = context.commit_column(&rec_src_l)?;
    let rec_cand_l_commitment = context.commit_column(&rec_cand_l)?;
    let rec_src_r_commitment = context.commit_column(&rec_src_r)?;
    let rec_cand_r_commitment = context.commit_column(&rec_cand_r)?;
    transcript.append_group(b"pow/pairs/com-rec-src-l", &rec_src_l_commitment);
    transcript.append_group(b"pow/pairs/com-rec-cand-l", &rec_cand_l_commitment);
    transcript.append_group(b"pow/pairs/com-rec-src-r", &rec_src_r_commitment);
    transcript.append_group(b"pow/pairs/com-rec-cand-r", &rec_cand_r_commitment);

    let shape = membership_shape(k, m_j, n_f, alpha_l, beta_l, alpha_r, beta_r);
    let columns = vec![
        sf_p.to_vec(),
        of_p.to_vec(),
        x_p.to_vec(),
        y_p.to_vec(),
        z_p.to_vec(),
        rho_l_p.to_vec(),
        rho_r_p.to_vec(),
        mult_l_in.to_vec(),
        mult_r_in.to_vec(),
        rec_src_l,
        rec_cand_l,
        rec_src_r,
        rec_cand_r,
    ];
    let (membership_sumcheck, r, _evals) = prove_sumcheck(&shape, &columns, transcript)?;
    let membership_opening = context.prove_column_openings(&columns, &r, transcript)?;

    // Combined-index strict monotonicity: i_eval = C*rho_l + rho_r.
    let radix = ArkFr::from_u64(combined_index_radix(n_f)? as u64);
    let i_eval: Vec<ArkFr> = (0..n)
        .map(|i| radix * rho_l_p[i] + rho_r_p[i])
        .collect();
    let com_i_eval = com_rho_l.scale(&radix).add(com_rho_r);
    let sorted =
        prove_strictly_sorted_with_context(context, &i_eval, m_j, &com_i_eval, transcript)?;

    Ok(PairsProof {
        membership_sumcheck,
        membership_opening,
        mult_l_in_commitment,
        mult_r_in_commitment,
        rec_src_l_commitment,
        rec_cand_l_commitment,
        rec_src_r_commitment,
        rec_cand_r_commitment,
        sorted,
    })
}

/// Verify a distinct-valid-pairs proof.
#[allow(clippy::too_many_arguments)]
pub fn verify_distinct_valid_pairs(
    com_sf: &ArkGT,
    com_of: &ArkGT,
    com_x: &ArkGT,
    com_y: &ArkGT,
    com_z: &ArkGT,
    com_rho_l: &ArkGT,
    com_rho_r: &ArkGT,
    n_f: usize,
    m_j: usize,
    proof: &PairsProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    let k = proof.membership_sumcheck.round_polys.len();
    if k == 0 || k >= usize::BITS as usize {
        return Err(invalid("pairs proof has an implausible variable count"));
    }
    if n_f > (1usize << k) || m_j > (1usize << k) {
        return Err(invalid("real length exceeds the pairs domain"));
    }

    transcript.append_group(b"pow/pairs/com-sf", com_sf);
    transcript.append_group(b"pow/pairs/com-of", com_of);
    transcript.append_group(b"pow/pairs/com-x", com_x);
    transcript.append_group(b"pow/pairs/com-y", com_y);
    transcript.append_group(b"pow/pairs/com-z", com_z);
    transcript.append_group(b"pow/pairs/com-rho-l", com_rho_l);
    transcript.append_group(b"pow/pairs/com-rho-r", com_rho_r);
    transcript.append_group(b"pow/pairs/com-mult-l-in", &proof.mult_l_in_commitment);
    transcript.append_group(b"pow/pairs/com-mult-r-in", &proof.mult_r_in_commitment);
    let beta_l = transcript.challenge_scalar(LABEL_BETA_L);
    let alpha_l = transcript.challenge_scalar(LABEL_ALPHA_L);
    let beta_r = transcript.challenge_scalar(LABEL_BETA_R);
    let alpha_r = transcript.challenge_scalar(LABEL_ALPHA_R);
    transcript.append_group(b"pow/pairs/com-rec-src-l", &proof.rec_src_l_commitment);
    transcript.append_group(b"pow/pairs/com-rec-cand-l", &proof.rec_cand_l_commitment);
    transcript.append_group(b"pow/pairs/com-rec-src-r", &proof.rec_src_r_commitment);
    transcript.append_group(b"pow/pairs/com-rec-cand-r", &proof.rec_cand_r_commitment);

    let shape = membership_shape(k, m_j, n_f, alpha_l, beta_l, alpha_r, beta_r);
    let (r, evals) = verify_sumcheck(&shape, &proof.membership_sumcheck, transcript)?;
    verify_column_openings(
        &[
            *com_sf,
            *com_of,
            *com_x,
            *com_y,
            *com_z,
            *com_rho_l,
            *com_rho_r,
            proof.mult_l_in_commitment,
            proof.mult_r_in_commitment,
            proof.rec_src_l_commitment,
            proof.rec_cand_l_commitment,
            proof.rec_src_r_commitment,
            proof.rec_cand_r_commitment,
        ],
        &r,
        &evals,
        &proof.membership_opening,
        transcript,
    )?;

    // Combined-index strict monotonicity (homomorphic com(i_eval)).
    let radix = ArkFr::from_u64(combined_index_radix(n_f)? as u64);
    let com_i_eval = com_rho_l.scale(&radix).add(com_rho_r);
    verify_strictly_sorted(&com_i_eval, m_j, &proof.sorted, transcript)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fr(v: u64) -> ArkFr {
        ArkFr::from_u64(v)
    }

    fn frs(vals: &[u64]) -> Vec<ArkFr> {
        vals.iter().map(|v| fr(*v)).collect()
    }

    fn transcript() -> Transcript254 {
        Transcript254::new(b"parture-pairs-test")
    }

    fn commit_padded(column: &[ArkFr], n: usize) -> ArkGT {
        commit_column(&padded(column, n)).unwrap()
    }

    // F = {(1,5),(2,5),(5,10),(5,20)}; key 5 has left F-rows {0,1} and right {2,3}.
    fn relation() -> (Vec<ArkFr>, Vec<ArkFr>) {
        (frs(&[1, 2, 5, 5]), frs(&[5, 5, 10, 20]))
    }

    // The complete join output, ordered by (rho_l, rho_r) ascending.
    fn honest_output() -> (Vec<ArkFr>, Vec<ArkFr>, Vec<ArkFr>, Vec<ArkFr>, Vec<ArkFr>) {
        let x = frs(&[1, 1, 2, 2]);
        let y = frs(&[5, 5, 5, 5]);
        let z = frs(&[10, 20, 10, 20]);
        let rho_l = frs(&[0, 0, 1, 1]);
        let rho_r = frs(&[2, 3, 2, 3]);
        (x, y, z, rho_l, rho_r)
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        sf: &[ArkFr],
        of: &[ArkFr],
        x: &[ArkFr],
        y: &[ArkFr],
        z: &[ArkFr],
        rho_l: &[ArkFr],
        rho_r: &[ArkFr],
    ) -> io::Result<()> {
        let n = pairs_domain(sf.len(), x.len()).unwrap();
        let com_sf = commit_padded(sf, n);
        let com_of = commit_padded(of, n);
        let com_x = commit_padded(x, n);
        let com_y = commit_padded(y, n);
        let com_z = commit_padded(z, n);
        let com_rho_l = commit_padded(rho_l, n);
        let com_rho_r = commit_padded(rho_r, n);
        let mut pt = transcript();
        let proof = prove_distinct_valid_pairs(
            sf, of, x, y, z, rho_l, rho_r, &com_sf, &com_of, &com_x, &com_y, &com_z, &com_rho_l,
            &com_rho_r, &mut pt,
        )?;
        let mut vt = transcript();
        verify_distinct_valid_pairs(
            &com_sf, &com_of, &com_x, &com_y, &com_z, &com_rho_l, &com_rho_r, sf.len(), x.len(),
            &proof, &mut vt,
        )
    }

    #[test]
    fn genuine_distinct_pairs_verify() {
        let (sf, of) = relation();
        let (x, y, z, rho_l, rho_r) = honest_output();
        assert!(run(&sf, &of, &x, &y, &z, &rho_l, &rho_r).is_ok());
    }

    #[test]
    fn within_group_swap_the_gather_missed_is_rejected() {
        // THE COUNTEREXAMPLE THE GATHER MISSED. Same marginals (mult_L=[2,2],
        // mult_R=[2,2]) and same count (4), but a within-group swap that DUPLICATES
        // (rho_l,rho_r) pairs: {(0,2)x2, (1,3)x2} instead of the true bijection
        // {(0,2),(0,3),(1,2),(1,3)}. Every row is still a valid pair (memberships
        // pass), so the gather could not see it -- but i_eval = C*rho_l+rho_r
        // repeats (2,2,7,7), so combined-index STRICT MONOTONICITY rejects it.
        let (sf, of) = relation();
        let x = frs(&[1, 1, 2, 2]);
        let y = frs(&[5, 5, 5, 5]);
        let z = frs(&[10, 10, 20, 20]); // (1,5,10)x2, (2,5,20)x2
        let rho_l = frs(&[0, 0, 1, 1]);
        let rho_r = frs(&[2, 2, 3, 3]); // duplicated pairs
        assert!(run(&sf, &of, &x, &y, &z, &rho_l, &rho_r).is_err());
    }

    #[test]
    fn fabricated_left_value_is_rejected() {
        // X[0] != Sf[rho_l[0]]: (99, 5, 0) is not F-row 0 -> left membership fails.
        let (sf, of) = relation();
        let (mut x, y, z, rho_l, rho_r) = honest_output();
        x[0] = fr(99);
        assert!(run(&sf, &of, &x, &y, &z, &rho_l, &rho_r).is_err());
    }

    #[test]
    fn rho_pointing_at_a_nonmatching_key_is_rejected() {
        // rho_r[0] -> F-row 0 (Sf=1), but Y[0]=5 != 1: right membership (5,10,0) is
        // not F-row 0 = (Sf=1, Of=5, 0) -> fails.
        let (sf, of) = relation();
        let (x, y, z, rho_l, mut rho_r) = honest_output();
        rho_r[0] = fr(0);
        assert!(run(&sf, &of, &x, &y, &z, &rho_l, &rho_r).is_err());
    }

    #[test]
    fn rho_out_of_range_is_rejected() {
        // rho_l[0] = 10 (>= n_F): no F-row has idx 10 -> left membership fails.
        let (sf, of) = relation();
        let (x, y, z, mut rho_l, rho_r) = honest_output();
        rho_l[0] = fr(10);
        assert!(run(&sf, &of, &x, &y, &z, &rho_l, &rho_r).is_err());
    }

    #[test]
    fn empty_output_verifies() {
        let (sf, of) = relation();
        assert!(run(&sf, &of, &[], &[], &[], &[], &[]).is_ok());
    }

    // ---- SOUNDNESS regression tests (source-side masking, FIX 1) ----

    // Malicious multiplicity: count rho hits over the FULL domain [0, n) (including
    // padding source rows), not just the honest [0, n_f).
    fn full_multiplicities(rho_full: &[ArkFr], m_j: usize, n: usize) -> Vec<ArkFr> {
        (0..n)
            .map(|a| {
                let a_fr = fr(a as u64);
                fr(rho_full[..m_j].iter().filter(|v| **v == a_fr).count() as u64)
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn run_with_witness(
        sf_p: &[ArkFr],
        of_p: &[ArkFr],
        x_p: &[ArkFr],
        y_p: &[ArkFr],
        z_p: &[ArkFr],
        rho_l_p: &[ArkFr],
        rho_r_p: &[ArkFr],
        n_f: usize,
        m_j: usize,
    ) -> io::Result<()> {
        let n = sf_p.len();
        let mult_l = full_multiplicities(rho_l_p, m_j, n);
        let mult_r = full_multiplicities(rho_r_p, m_j, n);
        let com_sf = commit_column(sf_p).unwrap();
        let com_of = commit_column(of_p).unwrap();
        let com_x = commit_column(x_p).unwrap();
        let com_y = commit_column(y_p).unwrap();
        let com_z = commit_column(z_p).unwrap();
        let com_rho_l = commit_column(rho_l_p).unwrap();
        let com_rho_r = commit_column(rho_r_p).unwrap();
        let mut pt = transcript();
        let proof = prove_distinct_valid_pairs_with_witness(
            sf_p, of_p, x_p, y_p, z_p, rho_l_p, rho_r_p, n_f, m_j, &mult_l, &mult_r, &com_sf,
            &com_of, &com_x, &com_y, &com_z, &com_rho_l, &com_rho_r, &mut pt,
        )?;
        let mut vt = transcript();
        verify_distinct_valid_pairs(
            &com_sf, &com_of, &com_x, &com_y, &com_z, &com_rho_l, &com_rho_r, n_f, m_j, &proof,
            &mut vt,
        )
    }

    #[test]
    fn substitution_pointing_at_a_padding_f_row_is_rejected() {
        // n_f=3, domain=4. Sf/Of committed with a FAKE F-row at padding index 3 =
        // (9,9). A fabricated output row (9,9,9) points rho_l=rho_r=3, and a malicious
        // full-domain multiplicity lights up source row 3. With the source side
        // unmasked this substitution VERIFIES; masking it by chi_{n_F} makes the
        // fabricated candidate pole unmatched -> REJECT (and forces rho < n_F).
        let sf_p = frs(&[1, 2, 5, 9]);
        let of_p = frs(&[5, 5, 10, 9]);
        let x_p = frs(&[1, 2, 9, 0]);
        let y_p = frs(&[5, 5, 9, 0]);
        let z_p = frs(&[10, 10, 9, 0]);
        let rho_l_p = frs(&[0, 1, 3, 0]);
        let rho_r_p = frs(&[2, 2, 3, 0]);
        assert!(
            run_with_witness(&sf_p, &of_p, &x_p, &y_p, &z_p, &rho_l_p, &rho_r_p, 3, 3).is_err()
        );
    }

    #[test]
    fn fabricated_zero_row_out_of_range_is_rejected() {
        // A fabricated (0,0,0) row pointing rho at a ZERO padding row (index 3 >= n_F).
        // With the source unmasked the all-zero padding pole matches; masking rejects.
        let sf_p = frs(&[1, 2, 5, 0]);
        let of_p = frs(&[5, 5, 10, 0]);
        let x_p = frs(&[1, 2, 0, 0]);
        let y_p = frs(&[5, 5, 0, 0]);
        let z_p = frs(&[10, 10, 0, 0]);
        let rho_l_p = frs(&[0, 1, 3, 0]);
        let rho_r_p = frs(&[2, 2, 3, 0]);
        assert!(
            run_with_witness(&sf_p, &of_p, &x_p, &y_p, &z_p, &rho_l_p, &rho_r_p, 3, 3).is_err()
        );
    }

    #[test]
    fn honest_witness_path_still_verifies() {
        // The honest output through the explicit-witness entry (mult only on [0,n_F))
        // must still verify after masking.
        let sf_p = frs(&[1, 2, 5, 5]);
        let of_p = frs(&[5, 5, 10, 20]);
        let x_p = frs(&[1, 1, 2, 2]);
        let y_p = frs(&[5, 5, 5, 5]);
        let z_p = frs(&[10, 20, 10, 20]);
        let rho_l_p = frs(&[0, 0, 1, 1]);
        let rho_r_p = frs(&[2, 3, 2, 3]);
        assert!(
            run_with_witness(&sf_p, &of_p, &x_p, &y_p, &z_p, &rho_l_p, &rho_r_p, 4, 4).is_ok()
        );
    }
}
