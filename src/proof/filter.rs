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
use super::logup::{
    multiset_link_claim, pseudo_inverse, reciprocal_claim, reciprocal_witness, rlc_fold,
};

// Column ordering. The 2b-1 indicator gadget uses [P, sel, w]; the full Q1
// compaction extends this with the data + output + reciprocal columns.
const COL_P: usize = 0;
const COL_SEL: usize = 1;
const COL_W: usize = 2;
const COL_S: usize = 3;
const COL_O: usize = 4;
const COL_SOUT: usize = 5;
const COL_OOUT: usize = 6;
const COL_CSTAR: usize = 7;
const COL_DSTAR: usize = 8;
// chi_bounds indices: chi_n (real-row count n) and chi_m (output length m).
const CHI_N: usize = 0;
const CHI_M: usize = 1;

const LABEL_BETA: &[u8] = b"pow/q1/logup-beta";
const LABEL_ALPHA: &[u8] = b"pow/q1/logup-alpha";

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
                    operands: vec![Operand::Chi(CHI_N)],
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
        chi_bounds: vec![real_rows],
        claims: vec![claim_a, claim_b],
    }
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

// ===========================================================================
// Full Q1: filter indicator + LogUp compaction to the dense (S_out, O_out).
// ===========================================================================

/// A complete Q1 filter query proof: the batched sumcheck over the two indicator
/// claims (A, B) AND the three LogUp compaction claims (C, D, E), one batched
/// Dory opening of all nine columns at `r`, and the internal witness commitments
/// the verifier binds into the transcript.
#[derive(Clone, Debug)]
pub struct FullFilterProof {
    pub sumcheck: SumcheckProof,
    pub opening: ColumnOpening,
    pub sel_commitment: ArkGT,
    pub w_commitment: ArkGT,
    pub cstar_commitment: ArkGT,
    pub dstar_commitment: ArkGT,
}

/// All nine claims of the full Q1 over columns
/// `[P, sel, w, S, O, S_out, O_out, cstar, dstar]` with `chi_bounds = [n, m]`.
fn full_filter_shape(
    k: usize,
    real_rows: usize,
    output_len: usize,
    pred: ArkFr,
    alpha: ArkFr,
    beta: ArkFr,
) -> SumcheckShape {
    let neg_one = f0() - f1();
    let neg_pred = f0() - pred;

    // A: sel*(P - pred) = 0
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
    // B: (chi_n - sel) - (P - pred)*w = 0
    let claim_b = SumcheckClaim {
        shape: ClaimShape::Identity,
        expr: Expr {
            terms: vec![
                Term {
                    coeff: f1(),
                    operands: vec![Operand::Chi(CHI_N)],
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
    // C: cstar*(alpha - (beta*S + O)) - 1 = 0
    let claim_c = reciprocal_claim(COL_CSTAR, &[COL_O, COL_S], alpha, beta);
    // D: dstar*(alpha - (beta*S_out + O_out)) - 1 = 0
    let claim_d = reciprocal_claim(COL_DSTAR, &[COL_OOUT, COL_SOUT], alpha, beta);
    // E (ZeroSum, the LogUp link): Σ sel*cstar - Σ chi_m*dstar = 0
    let claim_e = multiset_link_claim(
        Operand::Column(COL_SEL),
        COL_CSTAR,
        Operand::Chi(CHI_M),
        COL_DSTAR,
    );

    SumcheckShape {
        num_vars: k,
        num_columns: 9,
        chi_bounds: vec![real_rows, output_len],
        claims: vec![claim_a, claim_b, claim_c, claim_d, claim_e],
    }
}

/// Compact the selected rows into a dense output: for every `i` with `sel_i = 1`
/// (in row order) append `(S_i, O_i)`; pad the tail with zeros. Returns the dense
/// output columns and the match count `m`.
pub(crate) fn compact_selected(
    s: &[ArkFr],
    o: &[ArkFr],
    sel: &[ArkFr],
) -> (Vec<ArkFr>, Vec<ArkFr>, usize) {
    let size = s.len();
    let mut s_out = vec![f0(); size];
    let mut o_out = vec![f0(); size];
    let mut m = 0usize;
    for i in 0..size {
        if sel[i] == f1() {
            s_out[m] = s[i];
            o_out[m] = o[i];
            m += 1;
        }
    }
    (s_out, o_out, m)
}

fn bind_data_commitments(
    transcript: &mut Transcript254,
    com_p: &ArkGT,
    com_sel: &ArkGT,
    com_w: &ArkGT,
    com_s: &ArkGT,
    com_o: &ArkGT,
    com_sout: &ArkGT,
    com_oout: &ArkGT,
) {
    transcript.append_group(b"pow/q1/com-p", com_p);
    transcript.append_group(b"pow/q1/com-sel", com_sel);
    transcript.append_group(b"pow/q1/com-w", com_w);
    transcript.append_group(b"pow/q1/com-s", com_s);
    transcript.append_group(b"pow/q1/com-o", com_o);
    transcript.append_group(b"pow/q1/com-sout", com_sout);
    transcript.append_group(b"pow/q1/com-oout", com_oout);
}

fn bind_reciprocal_commitments(transcript: &mut Transcript254, com_cstar: &ArkGT, com_dstar: &ArkGT) {
    transcript.append_group(b"pow/q1/com-cstar", com_cstar);
    transcript.append_group(b"pow/q1/com-dstar", com_dstar);
}

/// Prove the full Q1 for EXPLICIT witnesses (used by the honest entry point and,
/// in tests, to inject malicious output/witnesses). Commits `sel, w, S_out, O_out`
/// and the reciprocal witnesses; returns the proof and the output commitments.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_filter_with_witness_full(
    s: &[ArkFr],
    p: &[ArkFr],
    o: &[ArkFr],
    pred: ArkFr,
    real_rows: usize,
    sel: &[ArkFr],
    w: &[ArkFr],
    s_out: &[ArkFr],
    o_out: &[ArkFr],
    output_len: usize,
    com_s: &ArkGT,
    com_p: &ArkGT,
    com_o: &ArkGT,
    transcript: &mut Transcript254,
) -> io::Result<(FullFilterProof, ArkGT, ArkGT)> {
    let size = p.len();
    if !size.is_power_of_two() {
        return Err(invalid("predicate column length must be a power of two"));
    }
    for column in [s, o, sel, w, s_out, o_out] {
        if column.len() != size {
            return Err(invalid("Q1 column length mismatch"));
        }
    }
    if real_rows > size || output_len > size {
        return Err(invalid("row count exceeds column length"));
    }
    let k = size.trailing_zeros() as usize;

    let com_sel = commit_column(sel)?;
    let com_w = commit_column(w)?;
    let com_sout = commit_column(s_out)?;
    let com_oout = commit_column(o_out)?;

    // Round 1: bind the data commitments, then draw the fingerprint challenges.
    bind_data_commitments(
        transcript, com_p, &com_sel, &com_w, com_s, com_o, &com_sout, &com_oout,
    );
    let beta = transcript.challenge_scalar(LABEL_BETA);
    let alpha = transcript.challenge_scalar(LABEL_ALPHA);

    // Reciprocal witnesses over the beta-fingerprinted rows (a_i = beta*S+O).
    let input_fold = rlc_fold(beta, &[o, s]);
    let cstar = reciprocal_witness(alpha, &input_fold);
    let output_fold = rlc_fold(beta, &[o_out, s_out]);
    let dstar = reciprocal_witness(alpha, &output_fold);
    let com_cstar = commit_column(&cstar)?;
    let com_dstar = commit_column(&dstar)?;

    // Round 2: bind the reciprocal commitments (drawn AFTER alpha/beta), then run
    // the single batched sumcheck over all five claims and open all nine columns.
    bind_reciprocal_commitments(transcript, &com_cstar, &com_dstar);
    let shape = full_filter_shape(k, real_rows, output_len, pred, alpha, beta);
    let columns = vec![
        p.to_vec(),
        sel.to_vec(),
        w.to_vec(),
        s.to_vec(),
        o.to_vec(),
        s_out.to_vec(),
        o_out.to_vec(),
        cstar,
        dstar,
    ];
    let (sumcheck, r, _evals) = prove_sumcheck(&shape, &columns, transcript)?;
    let opening = prove_column_openings(&columns, &r, transcript)?;

    Ok((
        FullFilterProof {
            sumcheck,
            opening,
            sel_commitment: com_sel,
            w_commitment: com_w,
            cstar_commitment: com_cstar,
            dstar_commitment: com_dstar,
        },
        com_sout,
        com_oout,
    ))
}

/// Prove the complete Q1 filter query: `(S_out, O_out)[0..m]` is EXACTLY the
/// compaction of `{(S_i, O_i) : P_i == pred AND i < real_rows}`. Returns the
/// proof, `Com(S_out)`, `Com(O_out)`, and the result length `m`.
#[allow(clippy::too_many_arguments)]
pub fn prove_filter(
    s: &[ArkFr],
    p: &[ArkFr],
    o: &[ArkFr],
    pred: ArkFr,
    real_rows: usize,
    com_s: &ArkGT,
    com_p: &ArkGT,
    com_o: &ArkGT,
    transcript: &mut Transcript254,
) -> io::Result<(FullFilterProof, ArkGT, ArkGT, usize)> {
    let (sel, w) = honest_witness(p, pred, real_rows);
    let (s_out, o_out, m) = compact_selected(s, o, &sel);
    let (proof, com_sout, com_oout) = prove_filter_with_witness_full(
        s, p, o, pred, real_rows, &sel, &w, &s_out, &o_out, m, com_s, com_p, com_o, transcript,
    )?;
    Ok((proof, com_sout, com_oout, m))
}

/// Verify a complete Q1 filter query proof. The verifier concludes that
/// `(S_out, O_out)[0..m]` is exactly the complete filtered result for
/// `predicate = pred` over the first `real_rows` rows, WITHOUT rerunning the
/// query. `m` is pinned by the LogUp count/residue check.
#[allow(clippy::too_many_arguments)]
pub fn verify_filter(
    com_s: &ArkGT,
    com_p: &ArkGT,
    com_o: &ArkGT,
    pred: ArkFr,
    real_rows: usize,
    com_sout: &ArkGT,
    com_oout: &ArkGT,
    output_len: usize,
    proof: &FullFilterProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    let k = proof.sumcheck.round_polys.len();
    if k == 0 || k >= usize::BITS as usize {
        return Err(invalid("Q1 proof has an implausible variable count"));
    }
    if real_rows > (1usize << k) || output_len > (1usize << k) {
        return Err(invalid("row count exceeds column length"));
    }

    bind_data_commitments(
        transcript,
        com_p,
        &proof.sel_commitment,
        &proof.w_commitment,
        com_s,
        com_o,
        com_sout,
        com_oout,
    );
    let beta = transcript.challenge_scalar(LABEL_BETA);
    let alpha = transcript.challenge_scalar(LABEL_ALPHA);
    bind_reciprocal_commitments(transcript, &proof.cstar_commitment, &proof.dstar_commitment);

    let shape = full_filter_shape(k, real_rows, output_len, pred, alpha, beta);
    let (r, evals) = verify_sumcheck(&shape, &proof.sumcheck, transcript)?;
    verify_column_openings(
        &[
            *com_p,
            proof.sel_commitment,
            proof.w_commitment,
            *com_s,
            *com_o,
            *com_sout,
            *com_oout,
            proof.cstar_commitment,
            proof.dstar_commitment,
        ],
        &r,
        &evals,
        &proof.opening,
        transcript,
    )?;
    Ok(())
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

    // ---- Full Q1 (filter + LogUp compaction) ----

    // S, P, O over 8 real rows (padded to 16). pred=1 matches rows 0,2,4,6.
    fn q1_tables() -> (Vec<ArkFr>, Vec<ArkFr>, Vec<ArkFr>, usize) {
        let s = predicate_column(&[10, 20, 30, 40, 50, 60, 70, 80], 0);
        let p = predicate_column(&[1, 2, 1, 3, 1, 2, 1, 9], 0);
        let o = predicate_column(&[100, 200, 300, 400, 500, 600, 700, 800], 0);
        (s, p, o, 8)
    }

    fn q1_run_honest(
        s: &[ArkFr],
        p: &[ArkFr],
        o: &[ArkFr],
        pred: ArkFr,
        n: usize,
    ) -> io::Result<usize> {
        let (com_s, com_p, com_o) = (commit(s), commit(p), commit(o));
        let mut pt = transcript();
        let (proof, com_sout, com_oout, m) =
            prove_filter(s, p, o, pred, n, &com_s, &com_p, &com_o, &mut pt)?;
        let mut vt = transcript();
        verify_filter(
            &com_s, &com_p, &com_o, pred, n, &com_sout, &com_oout, m, &proof, &mut vt,
        )?;
        Ok(m)
    }

    // Prove/verify with an EXPLICIT (possibly malicious) output.
    #[allow(clippy::too_many_arguments)]
    fn q1_run_with_output(
        s: &[ArkFr],
        p: &[ArkFr],
        o: &[ArkFr],
        pred: ArkFr,
        n: usize,
        s_out: &[ArkFr],
        o_out: &[ArkFr],
        m: usize,
    ) -> io::Result<()> {
        let (sel, w) = honest_witness(p, pred, n);
        let (com_s, com_p, com_o) = (commit(s), commit(p), commit(o));
        let mut pt = transcript();
        let (proof, com_sout, com_oout) = prove_filter_with_witness_full(
            s, p, o, pred, n, &sel, &w, s_out, o_out, m, &com_s, &com_p, &com_o, &mut pt,
        )?;
        let mut vt = transcript();
        verify_filter(
            &com_s, &com_p, &com_o, pred, n, &com_sout, &com_oout, m, &proof, &mut vt,
        )
    }

    #[test]
    fn q1_known_table_dense_output_matches_and_verifies() {
        let (s, p, o, n) = q1_tables();
        let pred = fr(1);
        let (sel, _) = honest_witness(&p, pred, n);
        let (s_out, o_out, m) = compact_selected(&s, &o, &sel);
        assert_eq!(m, 4);
        assert_eq!(&s_out[..4], &[fr(10), fr(30), fr(50), fr(70)]);
        assert_eq!(&o_out[..4], &[fr(100), fr(300), fr(500), fr(700)]);
        assert_eq!(q1_run_honest(&s, &p, &o, pred, n).unwrap(), 4);
    }

    #[test]
    fn q1_fabricated_output_row_is_rejected() {
        let (s, p, o, n) = q1_tables();
        let pred = fr(1);
        let (sel, _) = honest_witness(&p, pred, n);
        let (mut s_out, o_out, m) = compact_selected(&s, &o, &sel);
        s_out[0] = fr(999); // (999, 100) is not a selected row
        assert!(q1_run_with_output(&s, &p, &o, pred, n, &s_out, &o_out, m).is_err());
    }

    #[test]
    fn q1_omitted_output_row_is_rejected() {
        let (s, p, o, n) = q1_tables();
        let pred = fr(1);
        let (sel, _) = honest_witness(&p, pred, n);
        let (mut s_out, mut o_out, _m) = compact_selected(&s, &o, &sel);
        // Drop the 4th selected row and claim m = 3.
        s_out[3] = fr(0);
        o_out[3] = fr(0);
        assert!(q1_run_with_output(&s, &p, &o, pred, n, &s_out, &o_out, 3).is_err());
    }

    #[test]
    fn q1_wrong_m_too_large_or_too_small_is_rejected() {
        let (s, p, o, n) = q1_tables();
        let pred = fr(1);
        let (sel, _) = honest_witness(&p, pred, n);
        let (s_out, o_out, m) = compact_selected(&s, &o, &sel); // m = 4
        // Too large: claim a 5th (padding) output row.
        assert!(q1_run_with_output(&s, &p, &o, pred, n, &s_out, &o_out, m + 1).is_err());
        // Too small: claim only 3 of the 4 real rows.
        assert!(q1_run_with_output(&s, &p, &o, pred, n, &s_out, &o_out, m - 1).is_err());
    }

    #[test]
    fn q1_reordered_output_verifies_but_duplicate_is_rejected() {
        let (s, p, o, n) = q1_tables();
        let pred = fr(1);
        let (sel, _) = honest_witness(&p, pred, n);
        let (s_out, o_out, m) = compact_selected(&s, &o, &sel);

        // Reordering the true rows is a valid multiset -> verifies.
        let mut s_rev = s_out.clone();
        let mut o_rev = o_out.clone();
        s_rev[..m].reverse();
        o_rev[..m].reverse();
        assert!(q1_run_with_output(&s, &p, &o, pred, n, &s_rev, &o_rev, m).is_ok());

        // Duplicating a row (wrong multiplicity) -> rejected.
        let mut s_dup = s_out.clone();
        let mut o_dup = o_out.clone();
        s_dup[3] = s_out[0];
        o_dup[3] = o_out[0];
        assert!(q1_run_with_output(&s, &p, &o, pred, n, &s_dup, &o_dup, m).is_err());
    }

    #[test]
    fn q1_empty_result_and_all_match_verify() {
        let (s, p, o, n) = q1_tables();
        // Empty: pred absent from the real rows -> m = 0.
        assert_eq!(q1_run_honest(&s, &p, &o, fr(42), n).unwrap(), 0);

        // All-match: every real row matches -> m = n.
        let p_all = predicate_column(&[7, 7, 7, 7, 7, 7, 7, 7], 0);
        assert_eq!(q1_run_honest(&s, &p_all, &o, fr(7), n).unwrap(), n);
    }

    #[test]
    fn q1_end_to_end_over_several_predicates() {
        let (s, p, o, n) = q1_tables();
        for (pred_id, expected_m) in [(1u64, 4usize), (2, 2), (3, 1), (9, 1), (42, 0)] {
            let m = q1_run_honest(&s, &p, &o, fr(pred_id), n).unwrap();
            assert_eq!(m, expected_m, "predicate {pred_id}");
        }
    }
}
