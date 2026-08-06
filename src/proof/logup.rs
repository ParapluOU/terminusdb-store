//! Reusable LogUp (log-derivative) reciprocal gadget.
//!
//! The log-derivative identity turns a multiset equality into a single rational
//! equation: two multisets `{a_i}` and `{b_j}` (with multiplicities) are equal
//! iff `Σ_i mult_a(i)/(X - a_i) = Σ_j mult_b(j)/(X - b_j)` as rational functions
//! in `X`. Checked at a random challenge `alpha` (Schwartz–Zippel), this proves
//! multiset equality AND equal count. To fingerprint multi-column rows into one
//! scalar we random-linear-combine them with a second challenge `beta`.
//!
//! This module provides the primitive pieces so that 2b-2 (filter compaction),
//! and later 2c (join) and 2d (dictionary), all reuse the SAME reciprocal gadget:
//!   * [`rlc_fold`] — the `beta` fingerprint of several columns.
//!   * [`reciprocal_witness`] — the honest `1/(alpha - fold)` witness.
//!   * [`reciprocal_claim`] — the Identity claim pinning a committed reciprocal.
//!   * [`multiset_link_claim`] — the ZeroSum claim linking two reciprocals with
//!     their (possibly public) multiplicity masks.
//!
//! Clean-room from the log-derivative math (cf. p3-lookup, MIT/Apache); no COSL
//! code was consulted or copied.

use dory_pcs::backends::arkworks::ArkFr;
use dory_pcs::primitives::arithmetic::Field;

use super::argument::{ClaimShape, Expr, Operand, SumcheckClaim, Term};

#[inline]
fn f0() -> ArkFr {
    ArkFr::zero()
}
#[inline]
fn f1() -> ArkFr {
    ArkFr::one()
}

/// Random-linear-combination fingerprint of several equal-length columns into one
/// scalar per row: `fold[i] = Σ_d beta^d * columns[d][i]` (columns[0] gets
/// `beta^0`). To fingerprint a row `(S, O)` as `beta*S + O`, pass `[O, S]`.
pub(crate) fn rlc_fold(beta: ArkFr, columns: &[&[ArkFr]]) -> Vec<ArkFr> {
    let len = columns.first().map_or(0, |c| c.len());
    let mut fold = vec![f0(); len];
    let mut power = f1();
    for column in columns {
        for (slot, value) in fold.iter_mut().zip(column.iter()) {
            *slot = *slot + power * *value;
        }
        power = power * beta;
    }
    fold
}

/// Batch-invert (Montgomery's trick over the nonzero subset): returns
/// `vals[i]^{-1}` for each nonzero entry and `0` for zero entries.
pub(crate) fn pseudo_inverse(vals: &[ArkFr]) -> Vec<ArkFr> {
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

/// The reciprocal witness `rec[i] = 1/(alpha - folded[i])` (0 where
/// `alpha == folded[i]`, which never happens for a random `alpha` drawn after the
/// folded columns are committed).
pub(crate) fn reciprocal_witness(alpha: ArkFr, folded: &[ArkFr]) -> Vec<ArkFr> {
    let denominators: Vec<ArkFr> = folded.iter().map(|v| alpha - *v).collect();
    pseudo_inverse(&denominators)
}

/// Identity claim `rec_col * (alpha - Σ_d beta^d * value_columns[d]) - 1 = 0`,
/// which pins the committed column `rec_col` to `1/(alpha - fingerprint)` at every
/// row. `value_columns` are given in the same order as [`rlc_fold`].
pub(crate) fn reciprocal_claim(
    rec_col: usize,
    value_columns: &[usize],
    alpha: ArkFr,
    beta: ArkFr,
) -> SumcheckClaim {
    let mut terms = vec![Term {
        coeff: alpha,
        operands: vec![Operand::Column(rec_col)],
    }];
    let mut power = f1();
    for &value_col in value_columns {
        terms.push(Term {
            coeff: f0() - power,
            operands: vec![Operand::Column(rec_col), Operand::Column(value_col)],
        });
        power = power * beta;
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

/// ZeroSum claim `Σ_x [ in_mult(x)*rec_in(x) - out_mult(x)*rec_out(x) ] = 0` — the
/// LogUp multiset equality linking an input reciprocal (masked by the
/// multiplicity operand `in_mult`) to an output reciprocal (masked by `out_mult`).
/// When the reciprocals are pinned to `1/(alpha - fingerprint)`, this proves the
/// two fingerprinted multisets are equal, which at random `alpha`/`beta` forces
/// equal rows AND equal count.
pub(crate) fn multiset_link_claim(
    in_mult: Operand,
    in_rec: usize,
    out_mult: Operand,
    out_rec: usize,
) -> SumcheckClaim {
    SumcheckClaim {
        shape: ClaimShape::ZeroSum,
        expr: Expr {
            terms: vec![
                Term {
                    coeff: f1(),
                    operands: vec![in_mult, Operand::Column(in_rec)],
                },
                Term {
                    coeff: f0() - f1(),
                    operands: vec![out_mult, Operand::Column(out_rec)],
                },
            ],
        },
        target: f0(),
    }
}
