//! Join count identity (Phase 2c-2c): `Sum_i w_L[i]*w_R[i] = m_J`.
//!
//! For a sort-merge self-join with per-distinct-key multiplicities `w_L` (left
//! matches) and `w_R` (right matches), each distinct key `i` contributes exactly
//! `w_L[i]*w_R[i]` output rows (the Cartesian product of its matching left/right
//! rows), so the total output count is `Sum_i w_L[i]*w_R[i]`. This gadget proves
//! that sum equals a PUBLIC claimed `m_J`. Clean-room from the math (the
//! `sum w_l*w_r - chi_res = 0` identity in Proof-of-SQL's sort-merge join);
//! reimplemented on our primitives, no COSL code.
//!
//! Gadget: one ZeroSum claim with a product term and a public target
//!   `Sum_i chi_{n_U}[i] * w_L[i]*w_R[i] = m_J`
//! (reusing [`super::argument::SumcheckClaim::target`]; the `chi_{n_U}` mask is
//! belt-and-suspenders since `w` is 0 on padding). One sumcheck; a batched Dory
//! opening of `w_L, w_R`.
//!
//! CRUCIAL SOUNDNESS DEPENDENCY (regression-tested below): this identity does NOT
//! know whether the distinct-key column `U` is actually distinct. If `U` has a
//! duplicated key, the prover can SPLIT that key's `(w_L, w_R)` across the copies
//! and change `Sum w_L*w_R` (e.g. true `3*2 = 6` becomes `2*1 + 1*2 = 4`), and the
//! identity happily verifies the wrong `m_J`. Distinctness (Phase 2c-2a, via
//! `prove_strictly_sorted`) is therefore a HARD prerequisite that 2c-2e composes
//! upstream to collapse duplicates and forbid the split.

use std::io;

use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::transcript::Transcript;

use super::argument::{
    prove_sumcheck, verify_column_openings, verify_sumcheck, ClaimShape,
    ColumnOpening, Expr, Operand, StoreProverContext, SumcheckClaim, SumcheckProof, SumcheckShape,
    Term, Transcript254,
};
use super::invalid;

const COL_WL: usize = 0;
const COL_WR: usize = 1;
const NUM_COLUMNS: usize = 2;
// chi_bounds index for chi_{n_U} (the number of distinct keys).
const CHI_NU: usize = 0;

#[inline]
fn f0() -> ArkFr {
    ArkFr::zero()
}
#[inline]
fn f1() -> ArkFr {
    ArkFr::one()
}

/// A count-identity proof: the single-claim sumcheck and a batched Dory opening
/// of `w_L, w_R` (both commitments are external inputs).
#[derive(Clone, Debug)]
pub struct CountProof {
    pub sumcheck: SumcheckProof,
    pub opening: ColumnOpening,
}

/// The padded power-of-two domain (`>= 2`) for multiplicity columns of `n_u` rows.
pub(crate) fn count_domain(n_u: usize) -> io::Result<usize> {
    n_u.checked_next_power_of_two()
        .ok_or_else(|| invalid("count domain size overflow"))
        .map(|d| d.max(2))
}

fn padded(column: &[ArkFr], n: usize) -> Vec<ArkFr> {
    let mut out = column.to_vec();
    out.resize(n, f0());
    out
}

/// `Sum_i chi_{n_U}[i] * w_L[i]*w_R[i] = m_J` as a single public-target ZeroSum.
fn count_shape(k: usize, n_u: usize, m_j: usize) -> SumcheckShape {
    let claim = SumcheckClaim {
        shape: ClaimShape::ZeroSum,
        expr: Expr {
            terms: vec![Term {
                coeff: f1(),
                operands: vec![
                    Operand::Chi(CHI_NU),
                    Operand::Column(COL_WL),
                    Operand::Column(COL_WR),
                ],
            }],
        },
        target: ArkFr::from_u64(m_j as u64),
    };
    SumcheckShape {
        num_vars: k,
        num_columns: NUM_COLUMNS,
        chi_bounds: vec![n_u],
        claims: vec![claim],
    }
}

/// Prove `Sum_i w_L[i]*w_R[i] = m_J`, summing over the real `[0, n_U)` rows only.
/// `com_wl`/`com_wr` must commit `w_L`/`w_R` padded to `count_domain(w_L.len())`.
/// `n_U` is the TRUE distinct-key count (the chi mask real length); passing the
/// padded domain here would make the mask a no-op and let padding-tail weights
/// inflate the count (audit FIX 3).
pub fn prove_count_identity(
    w_l: &[ArkFr],
    w_r: &[ArkFr],
    n_u: usize,
    m_j: usize,
    com_wl: &ArkGT,
    com_wr: &ArkGT,
    transcript: &mut Transcript254,
) -> io::Result<CountProof> {
    prove_count_identity_with_context(
        &StoreProverContext::cpu(),
        w_l,
        w_r,
        n_u,
        m_j,
        com_wl,
        com_wr,
        transcript,
    )
}

/// As [`prove_count_identity`], with an explicit prover-arithmetic context (the choice of
/// context cannot change the proof bytes or the guarantee).
#[allow(clippy::too_many_arguments)]
pub fn prove_count_identity_with_context(
    context: &StoreProverContext<'_>,
    w_l: &[ArkFr],
    w_r: &[ArkFr],
    n_u: usize,
    m_j: usize,
    com_wl: &ArkGT,
    com_wr: &ArkGT,
    transcript: &mut Transcript254,
) -> io::Result<CountProof> {
    if w_r.len() != w_l.len() {
        return Err(invalid("w_L and w_R must have equal length"));
    }
    let n = count_domain(w_l.len())?;
    if n_u > n {
        return Err(invalid("distinct-key count exceeds the domain"));
    }
    let k = n.trailing_zeros() as usize;
    let wl_p = padded(w_l, n);
    let wr_p = padded(w_r, n);

    transcript.append_group(b"pow/count/com-wl", com_wl);
    transcript.append_group(b"pow/count/com-wr", com_wr);

    let shape = count_shape(k, n_u, m_j);
    let columns = vec![wl_p, wr_p];
    let (sumcheck, r, _evals) = prove_sumcheck(&shape, &columns, transcript)?;
    let opening = context.prove_column_openings(&columns, &r, transcript)?;

    Ok(CountProof { sumcheck, opening })
}

/// Verify a count-identity proof.
pub fn verify_count_identity(
    com_wl: &ArkGT,
    com_wr: &ArkGT,
    n_u: usize,
    m_j: usize,
    proof: &CountProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    let k = proof.sumcheck.round_polys.len();
    if k == 0 || k >= usize::BITS as usize {
        return Err(invalid("count proof has an implausible variable count"));
    }
    if n_u > (1usize << k) {
        return Err(invalid("distinct-key count exceeds the domain"));
    }

    transcript.append_group(b"pow/count/com-wl", com_wl);
    transcript.append_group(b"pow/count/com-wr", com_wr);

    let shape = count_shape(k, n_u, m_j);
    let (r, evals) = verify_sumcheck(&shape, &proof.sumcheck, transcript)?;
    verify_column_openings(&[*com_wl, *com_wr], &r, &evals, &proof.opening, transcript)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof::argument::commit_column;

    fn fr(v: u64) -> ArkFr {
        ArkFr::from_u64(v)
    }

    fn frs(vals: &[u64]) -> Vec<ArkFr> {
        vals.iter().map(|v| fr(*v)).collect()
    }

    fn transcript() -> Transcript254 {
        Transcript254::new(b"parture-count-test")
    }

    fn commit_padded(column: &[ArkFr], n: usize) -> ArkGT {
        commit_column(&padded(column, n)).unwrap()
    }

    fn run(w_l: &[ArkFr], w_r: &[ArkFr], m_j: usize) -> io::Result<()> {
        run_with_n_u(w_l, w_r, w_l.len(), m_j)
    }

    // Prove/verify with an EXPLICIT claimed n_U (the chi real length).
    fn run_with_n_u(w_l: &[ArkFr], w_r: &[ArkFr], n_u: usize, m_j: usize) -> io::Result<()> {
        let n = count_domain(w_l.len()).unwrap();
        let com_wl = commit_padded(w_l, n);
        let com_wr = commit_padded(w_r, n);
        let mut pt = transcript();
        let proof = prove_count_identity(w_l, w_r, n_u, m_j, &com_wl, &com_wr, &mut pt)?;
        let mut vt = transcript();
        verify_count_identity(&com_wl, &com_wr, n_u, m_j, &proof, &mut vt)
    }

    #[test]
    fn padding_tail_weight_is_masked_by_true_n_u() {
        // FIX 3 demonstration. w over domain 8; real n_U = 3. A stashed pair at padding
        // index 5 (w_L=w_R=1) inflates the sum when the chi mask is a no-op (the bug
        // was join.rs passing the DOMAIN as n_U): true sum over [0,3) = 3*1+2*2+1*3 =
        // 10, and the stash adds 1.
        let w_l = frs(&[3, 2, 1, 0, 0, 1, 0, 0]);
        let w_r = frs(&[1, 2, 3, 0, 0, 1, 0, 0]);
        // BUG (n_U = domain = 8, chi no-op): the inflated count 11 verifies.
        assert!(run_with_n_u(&w_l, &w_r, 8, 11).is_ok());
        // FIX (true n_U = 3): the padding-tail weight is masked out.
        assert!(run_with_n_u(&w_l, &w_r, 3, 11).is_err());
        assert!(run_with_n_u(&w_l, &w_r, 3, 10).is_ok());
    }

    #[test]
    fn correct_count_verifies() {
        // 3*1 + 2*2 + 1*3 = 10.
        let w_l = frs(&[3, 2, 1]);
        let w_r = frs(&[1, 2, 3]);
        assert!(run(&w_l, &w_r, 10).is_ok());
    }

    #[test]
    fn wrong_m_j_off_by_one_is_rejected() {
        let w_l = frs(&[3, 2, 1]);
        let w_r = frs(&[1, 2, 3]); // true sum 10
        assert!(run(&w_l, &w_r, 9).is_err());
        assert!(run(&w_l, &w_r, 11).is_err());
    }

    #[test]
    fn distinctness_dependency_regression_split_over_duplicate_u_passes_in_isolation() {
        // HEADLINE REGRESSION TEST. Two of these rows represent the SAME join key
        // (a DUPLICATE in U). Its true counts are (w_L=3, w_R=2) -> true
        // contribution 3*2 = 6 output rows. But with U duplicated, the prover
        // SPLITS the key across two rows -- row A (w_L=2, w_R=1), row B (w_L=1,
        // w_R=2) -- so Sum w_L*w_R over them = 2*1 + 1*2 = 4, not 6. Membership
        // (2c-2b) only pins the SUMS (w_L: 2+1=3, w_R: 1+2=2), so it does not
        // object. The count identity, IN ISOLATION, then happily verifies the
        // WRONG m_J = 4 -- it has no idea U is non-distinct.
        //
        // This is EXACTLY why 2c-2a distinctness is a hard prerequisite: composing
        // prove_strictly_sorted upstream (2c-2e) collapses the duplicate into a
        // single row (w_L=3, w_R=2 -> 6), forbidding the split. Tracked here rather
        // than hidden.
        let w_l_split = frs(&[2, 1]);
        let w_r_split = frs(&[1, 2]);
        assert!(run(&w_l_split, &w_r_split, 4).is_ok());
        // And the TRUE distinct-key encoding (single row) gives the honest count 6.
        assert!(run(&frs(&[3]), &frs(&[2]), 6).is_ok());
    }

    #[test]
    fn all_zero_multiplicities_with_zero_count_verifies() {
        assert!(run(&frs(&[0, 0, 0]), &frs(&[0, 0, 0]), 0).is_ok());
        // A nonzero claimed count against all-zero multiplicities is rejected.
        assert!(run(&frs(&[0, 0, 0]), &frs(&[0, 0, 0]), 1).is_err());
    }
}
