//! Q1 filter selection-indicator (IsZero) gadget — the first real query-correctness
//! proof of the Proof-of-WOQL argument.
//!
//! Given a committed predicate column `Com(P)` over `N = 2^k` rows (`n` of them
//! real, the rest padding) and a PUBLIC constant predicate id `pred`, this gadget
//! proves that a committed selection vector `sel` equals EXACTLY
//! `[ P_i == pred AND i < n ]` for every row — filter correct AND complete (no
//! matching row omitted, no non-matching/padding row selected).
//!
//! Blueprint (IsZero): the prover commits `sel in {0,1}^N` and a pseudo-inverse
//! witness `w`, with `lhs_i = P_i - pred`, and proves two Identity (zero-at-every-
//! row) claims via the batched sumcheck ([`super::argument`]):
//!   * A:  `sel_i * lhs_i = 0`
//!   * B:  `(chi_n_i - sel_i) - lhs_i * w_i = 0`
//! These pin `sel_i = [P_i == pred AND i < n]` uniquely (booleanity subsumed).
//! `chi_n` is PUBLIC (the real-row count `n` lives in the layer root), so it is
//! never committed: the verifier derives `chi_n(r)` closed-form
//! ([`super::argument::chi_n_eval`]), which also masks the padding-collision
//! hazard (a padding row whose `P` equals `pred` must still have `sel = 0`).
//!
//! Honest witnesses:
//!   * `sel_i = 1` iff `i < n && P_i == pred`.
//!   * `w_i = lhs_i^{-1}` iff `i < n && lhs_i != 0`, else `0` (the `i < n` mask is
//!     required: on padding rows B forces `lhs_i * w_i = 0`, so `w` must be 0 on
//!     padding non-matches). On matching rows (`lhs_i = 0`) `w` is unconstrained.
//!
//! Phase 2b-1 only: NO LogUp compaction to a dense output (that is 2b-2).

use std::io;

use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::transcript::Transcript;

use super::argument::{
    commit_column, prove_column_openings, prove_sumcheck, verify_column_openings, verify_sumcheck,
    ClaimShape, ColumnOpening, Expr, Operand, SumcheckClaim, SumcheckProof, SumcheckShape, Term,
    Transcript254,
};
use super::invalid;

// Shared column ordering for [P, sel, w].
const COL_P: usize = 0;
const COL_SEL: usize = 1;
const COL_W: usize = 2;

#[inline]
fn f0() -> ArkFr {
    ArkFr::zero()
}
#[inline]
fn f1() -> ArkFr {
    ArkFr::one()
}

/// A filter-indicator proof: the batched sumcheck over the two Identity claims,
/// plus one batched Dory opening of `[P, sel, w]` at the sumcheck point `r`.
#[derive(Clone, Debug)]
pub struct FilterProof {
    pub sumcheck: SumcheckProof,
    pub opening: ColumnOpening,
}

/// The two Identity claims A and B over columns `[P, sel, w]`, with `chi_n(n)`
/// and the public constant `pred` folded into the expression coefficients.
fn filter_shape(k: usize, real_rows: usize, pred: ArkFr) -> SumcheckShape {
    let neg_one = f0() - f1();
    let neg_pred = f0() - pred;

    // A: sel * (P - pred) = sel*P + (-pred)*sel
    let claim_a = SumcheckClaim {
        shape: ClaimShape::Identity,
        expr: Expr {
            terms: vec![
                Term {
                    coeff: f1(),
                    operands: vec![Operand::Column(COL_SEL), Operand::Column(COL_P)],
                },
                Term {
                    coeff: neg_pred,
                    operands: vec![Operand::Column(COL_SEL)],
                },
            ],
        },
    };

    // B: (chi_n - sel) - (P - pred)*w = chi_n - sel - P*w + pred*w
    let claim_b = SumcheckClaim {
        shape: ClaimShape::Identity,
        expr: Expr {
            terms: vec![
                Term {
                    coeff: f1(),
                    operands: vec![Operand::ChiN],
                },
                Term {
                    coeff: neg_one,
                    operands: vec![Operand::Column(COL_SEL)],
                },
                Term {
                    coeff: neg_one,
                    operands: vec![Operand::Column(COL_P), Operand::Column(COL_W)],
                },
                Term {
                    coeff: pred,
                    operands: vec![Operand::Column(COL_W)],
                },
            ],
        },
    };

    SumcheckShape {
        num_vars: k,
        num_columns: 3,
        chi_n: Some(real_rows),
        claims: vec![claim_a, claim_b],
    }
}

/// Batch-invert: returns `vals[i]^{-1}` for each nonzero entry and `0` for zero
/// entries (Montgomery's trick over the nonzero subset).
fn pseudo_inverse(vals: &[ArkFr]) -> Vec<ArkFr> {
    let indices: Vec<usize> = (0..vals.len()).filter(|&i| !vals[i].is_zero()).collect();
    let m = indices.len();
    let mut out = vec![f0(); vals.len()];
    if m == 0 {
        return out;
    }
    let mut prefix = vec![f1(); m + 1];
    for (t, &i) in indices.iter().enumerate() {
        prefix[t + 1] = prefix[t] * vals[i];
    }
    let mut running = match prefix[m].inv() {
        Some(inv) => inv,
        None => return out, // product of nonzero values is nonzero; unreachable
    };
    for t in (0..m).rev() {
        let i = indices[t];
        out[i] = prefix[t] * running;
        running = running * vals[i];
    }
    out
}

/// Compute the honest `(sel, w)` witnesses for a predicate column.
pub(crate) fn honest_witness(
    predicate_column: &[ArkFr],
    pred: ArkFr,
    real_rows: usize,
) -> (Vec<ArkFr>, Vec<ArkFr>) {
    let size = predicate_column.len();
    let lhs: Vec<ArkFr> = predicate_column.iter().map(|p| *p - pred).collect();
    let sel: Vec<ArkFr> = (0..size)
        .map(|i| {
            if i < real_rows && lhs[i].is_zero() {
                f1()
            } else {
                f0()
            }
        })
        .collect();
    // w is the pseudo-inverse of lhs masked to the real rows [0, n): 0 on padding
    // rows (B forces lhs*w = 0 there) and on matching rows (lhs = 0).
    let masked_lhs: Vec<ArkFr> = (0..size)
        .map(|i| if i < real_rows { lhs[i] } else { f0() })
        .collect();
    let w = pseudo_inverse(&masked_lhs);
    (sel, w)
}

/// Prove the filter indicator for an EXPLICIT witness `(sel, w)`. The honest
/// entry point [`prove_filter_indicator`] derives the witness and calls this;
/// tests use it to inject malicious witnesses.
pub(crate) fn prove_filter_with_witness(
    predicate_column: &[ArkFr],
    pred: ArkFr,
    real_rows: usize,
    sel: &[ArkFr],
    w: &[ArkFr],
    predicate_commitment: &ArkGT,
    transcript: &mut Transcript254,
) -> io::Result<(FilterProof, ArkGT, ArkGT, ArkFr)> {
    let size = predicate_column.len();
    if !size.is_power_of_two() {
        return Err(invalid("predicate column length must be a power of two"));
    }
    if sel.len() != size || w.len() != size {
        return Err(invalid("witness length does not match the predicate column"));
    }
    if real_rows > size {
        return Err(invalid("real-row count exceeds column length"));
    }
    let k = size.trailing_zeros() as usize;

    let sel_commitment = commit_column(sel)?;
    let w_commitment = commit_column(w)?;

    // Bind every column commitment BEFORE the zero-check point is drawn.
    transcript.append_group(b"pow/filter/com-p", predicate_commitment);
    transcript.append_group(b"pow/filter/com-sel", &sel_commitment);
    transcript.append_group(b"pow/filter/com-w", &w_commitment);

    let shape = filter_shape(k, real_rows, pred);
    let columns = vec![predicate_column.to_vec(), sel.to_vec(), w.to_vec()];
    let (sumcheck, r, evals) = prove_sumcheck(&shape, &columns, transcript)?;
    let opening = prove_column_openings(&columns, &r, transcript)?;

    Ok((
        FilterProof { sumcheck, opening },
        sel_commitment,
        w_commitment,
        evals[COL_SEL],
    ))
}

/// Prove that a filter selection vector is correct and complete for the given
/// predicate column and PUBLIC constant. Commits `sel` and `w` internally.
/// Returns the proof, `Com(sel)`, `Com(w)`, and `sel(r)` (for chaining into the
/// 2b-2 compaction).
pub fn prove_filter_indicator(
    predicate_column: &[ArkFr],
    pred: ArkFr,
    real_rows: usize,
    predicate_commitment: &ArkGT,
    transcript: &mut Transcript254,
) -> io::Result<(FilterProof, ArkGT, ArkGT, ArkFr)> {
    let (sel, w) = honest_witness(predicate_column, pred, real_rows);
    prove_filter_with_witness(
        predicate_column,
        pred,
        real_rows,
        &sel,
        &w,
        predicate_commitment,
        transcript,
    )
}

/// Verify a filter-indicator proof. Re-derives the zero-check point, checks the
/// batched sumcheck against the claimed evals + `chi_n(r)` + `(eval(P) - pred)`,
/// and verifies the batched Dory opening at `r`. Returns `sel(r)` for chaining.
pub fn verify_filter_indicator(
    predicate_commitment: &ArkGT,
    pred: ArkFr,
    real_rows: usize,
    sel_commitment: &ArkGT,
    w_commitment: &ArkGT,
    proof: &FilterProof,
    transcript: &mut Transcript254,
) -> io::Result<ArkFr> {
    // k is the number of sumcheck rounds; the columns' size (2^k) is bound by the
    // Dory opening, which verifies the commitments against a length-k point.
    let k = proof.sumcheck.round_polys.len();
    if k == 0 || k >= usize::BITS as usize {
        return Err(invalid("filter proof has an implausible variable count"));
    }
    if real_rows > (1usize << k) {
        return Err(invalid("real-row count exceeds column length"));
    }

    transcript.append_group(b"pow/filter/com-p", predicate_commitment);
    transcript.append_group(b"pow/filter/com-sel", sel_commitment);
    transcript.append_group(b"pow/filter/com-w", w_commitment);

    let shape = filter_shape(k, real_rows, pred);
    let (r, evals) = verify_sumcheck(&shape, &proof.sumcheck, transcript)?;
    verify_column_openings(
        &[*predicate_commitment, *sel_commitment, *w_commitment],
        &r,
        &evals,
        &proof.opening,
        transcript,
    )?;
    Ok(evals[COL_SEL])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fr(v: u64) -> ArkFr {
        ArkFr::from_u64(v)
    }

    fn transcript() -> Transcript254 {
        Transcript254::new(b"parture-filter-test")
    }

    // Build a padded predicate column of length 16 from `n` real ids (the rest
    // padded with `pad`).
    fn predicate_column(real_ids: &[u64], pad: u64) -> Vec<ArkFr> {
        assert!(real_ids.len() <= 16);
        (0..16usize)
            .map(|i| fr(if i < real_ids.len() { real_ids[i] } else { pad }))
            .collect()
    }

    fn commit(column: &[ArkFr]) -> ArkGT {
        commit_column(column).unwrap()
    }

    fn run(
        p_col: &[ArkFr],
        pred: ArkFr,
        n: usize,
        sel: &[ArkFr],
        w: &[ArkFr],
    ) -> io::Result<ArkFr> {
        let com_p = commit(p_col);
        let mut pt = transcript();
        let (proof, com_sel, com_w, _sel_eval) =
            prove_filter_with_witness(p_col, pred, n, sel, w, &com_p, &mut pt)?;
        let mut vt = transcript();
        verify_filter_indicator(&com_p, pred, n, &com_sel, &com_w, &proof, &mut vt)
    }

    #[test]
    fn correct_selection_verifies() {
        // ids: 5,1,5,2,1,9,1,5 real (n=8); const=1 matches rows 1,4,6.
        let p = predicate_column(&[5, 1, 5, 2, 1, 9, 1, 5], 0);
        let n = 8;
        let pred = fr(1);
        let (sel, w) = honest_witness(&p, pred, n);
        // sanity: sel is exactly the in-range matches.
        for i in 0..16 {
            let expected = if i < n && p[i] == pred { f1() } else { f0() };
            assert_eq!(sel[i], expected);
        }
        assert!(run(&p, pred, n, &sel, &w).is_ok());
    }

    #[test]
    fn omission_of_a_true_match_is_rejected() {
        let p = predicate_column(&[5, 1, 5, 2, 1, 9, 1, 5], 0);
        let n = 8;
        let pred = fr(1);
        let (mut sel, w) = honest_witness(&p, pred, n);
        // Flip a genuine match (row 4) from 1 -> 0.
        assert_eq!(sel[4], f1());
        sel[4] = f0();
        assert!(run(&p, pred, n, &sel, &w).is_err());
    }

    #[test]
    fn false_positive_on_a_nonmatch_is_rejected() {
        let p = predicate_column(&[5, 1, 5, 2, 1, 9, 1, 5], 0);
        let n = 8;
        let pred = fr(1);
        let (mut sel, w) = honest_witness(&p, pred, n);
        // Select a non-matching in-range row (row 0, P=5 != 1).
        assert_eq!(sel[0], f0());
        sel[0] = f1();
        assert!(run(&p, pred, n, &sel, &w).is_err());
    }

    #[test]
    fn padding_collision_is_masked_and_a_selected_padding_row_is_rejected() {
        // Row 10 is PADDING (n=8) but its P value equals the queried constant.
        let mut ids = vec![5u64, 1, 5, 2, 1, 9, 1, 5];
        let p = {
            let mut col = predicate_column(&ids, 0);
            col[10] = fr(1); // padding row collides with pred
            col
        };
        let _ = &mut ids;
        let n = 8;
        let pred = fr(1);
        let (mut sel, w) = honest_witness(&p, pred, n);
        // Honest sel MUST be 0 on the colliding padding row (chi_n masks it).
        assert_eq!(sel[10], f0());
        assert!(run(&p, pred, n, &sel, &w).is_ok());
        // A prover that selects the padding row is rejected.
        sel[10] = f1();
        assert!(run(&p, pred, n, &sel, &w).is_err());
    }

    #[test]
    fn wrong_pseudo_inverse_on_a_nonmatch_row_is_rejected() {
        // w is pinned on NON-matching rows (lhs != 0): B forces w = chi_n/lhs.
        // Corrupt w on an in-range non-match row (row 0, P=5): honest w = lhs^{-1},
        // set it to 0 -> B: 1 - 0 - lhs*0 = 1 != 0 -> reject.
        // (On MATCHING rows lhs = 0, so w there is unconstrained and a nonzero
        // value would NOT be a soundness violation.)
        let p = predicate_column(&[5, 1, 5, 2, 1, 9, 1, 5], 0);
        let n = 8;
        let pred = fr(1);
        let (sel, mut w) = honest_witness(&p, pred, n);
        assert_ne!(w[0], f0()); // row 0 is a non-match, so w[0] = (5-1)^{-1}
        w[0] = f0();
        assert!(run(&p, pred, n, &sel, &w).is_err());
    }

    #[test]
    fn nonzero_w_on_a_matching_row_is_harmless() {
        // Documents the gadget's exact guarantee: w is unconstrained on matching
        // rows, so a nonzero value there is NOT a violation and still verifies.
        let p = predicate_column(&[5, 1, 5, 2, 1, 9, 1, 5], 0);
        let n = 8;
        let pred = fr(1);
        let (sel, mut w) = honest_witness(&p, pred, n);
        assert_eq!(sel[1], f1()); // row 1 matches, honest w[1] = 0
        assert_eq!(w[1], f0());
        w[1] = fr(12345);
        assert!(run(&p, pred, n, &sel, &w).is_ok());
    }

    #[test]
    fn empty_result_and_all_match_verify() {
        // Empty: pred absent from the real rows.
        let p = predicate_column(&[5, 1, 5, 2, 1, 9, 1, 5], 0);
        let n = 8;
        let pred = fr(7); // no real row equals 7
        let (sel, w) = honest_witness(&p, pred, n);
        assert!(sel.iter().all(|s| *s == f0()));
        assert!(run(&p, pred, n, &sel, &w).is_ok());

        // All-match: every real row equals pred.
        let p_all = predicate_column(&[3, 3, 3, 3, 3, 3, 3, 3], 0);
        let n2 = 8;
        let pred2 = fr(3);
        let (sel2, w2) = honest_witness(&p_all, pred2, n2);
        for i in 0..16 {
            assert_eq!(sel2[i], if i < n2 { f1() } else { f0() });
        }
        assert!(run(&p_all, pred2, n2, &sel2, &w2).is_ok());
    }
}
