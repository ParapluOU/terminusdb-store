//! Standalone numeric sign/range selection gadget (Phase 7.4, CP-B1).
//!
//! Given a committed value column `V` (`N = 2^k` rows, `n` of them real, the rest
//! padding), a PUBLIC constant `c`, a bit-width `B`, and an operator
//! `op ∈ {<, <=, >, >=}`, this gadget proves that a committed boolean selection
//! column `sel` equals EXACTLY `[ V_i op c AND i < n ]` for every row. It is
//! clean-room from the math (the design mirrors Proof-of-SQL's production sign
//! gadget, but is implemented from our own primitives; no COSL code transcribed).
//!
//! # The math (offset difference + sign bit)
//! To read the sign of a comparison without any field-order magic, form a
//! non-negative offset difference and decompose it:
//!   * `E_i  = V_i − c + 2^B`   (for `>=` and `<`)
//!   * `E'_i = c − V_i + 2^B`   (for `<=` and `>`)
//! If every real `V_i` and `c` lie in `[0, 2^B)` then `E_i, E'_i ∈ (0, 2^{B+1})`,
//! a genuine `(B+1)`-bit non-negative integer — NO field wraparound. Decompose the
//! chosen difference into `B+1` committed bit columns `b_0..b_B`; the TOP bit `b_B`
//! is the sign, because with the exact offset `2^B` a difference is `≥ 2^B` iff it
//! is non-negative:
//!   * `b_B = 1 ⟺ E_i  ≥ 2^B ⟺ V_i ≥ c`
//!   * `b_B = 1 ⟺ E'_i ≥ 2^B ⟺ c ≥ V_i ⟺ V_i ≤ c`
//! The offset is EXACTLY `2^B`, so at the boundary `V_i = c` the difference is
//! exactly `2^B` (`b_B = 1`, all lower bits 0) — see the per-op boundary below.
//!
//! # Operator → (difference, selection) mapping
//! | op   | difference             | sign bit `s = b_B` means | sel (masked by χ_n) |
//! |------|------------------------|--------------------------|---------------------|
//! | `>=` | `E  = V − c + 2^B`     | `V ≥ c`                  | `χ_n · s`           |
//! | `<`  | `E  = V − c + 2^B`     | `V ≥ c`                  | `χ_n · (1 − s)`     |
//! | `<=` | `E' = c − V + 2^B`     | `V ≤ c`                  | `χ_n · s`           |
//! | `>`  | `E' = c − V + 2^B`     | `V ≤ c`                  | `χ_n · (1 − s)`     |
//! Boundary `V = c`: `E = E' = 2^B`, so `s = 1`; thus `>=`/`<=` select it and
//! `<`/`>` do not — exactly right, no off-by-one.
//!
//! # Claims (all `Identity`, one batched sumcheck; see [`super::argument`])
//! Over columns `[V, b_0, .., b_B, sel]`, with `χ_n` the PUBLIC real-row prefix
//! indicator of `[0, n)` ([`super::argument::Operand::Chi`], never committed):
//!   1. **Booleanity** (all `B+1` bits, INCLUDING the sign bit `b_B`, batched with
//!      a fresh Fiat-Shamir `tau`): `χ_n · Σ_j tau^j · b_j·(b_j − 1) = 0`. Missing
//!      the sign bit would let a non-{0,1} `b_B` decode a false sign.
//!   2. **Recomposition** (binds the committed `V` and the public `c`):
//!      `χ_n · (Σ_j 2^j·b_j − V + (c − 2^B)) = 0` for the `E` difference, or
//!      `χ_n · (Σ_j 2^j·b_j + V − (c + 2^B)) = 0` for the `E'` difference.
//!   3. **Sign → sel** (masks padding AND makes `sel` boolean): for `>=`/`<=`
//!      `sel − χ_n·b_B = 0`; for `<`/`>` `sel − χ_n·(1 − b_B) = 0`.
//! Claims 1 & 2 are masked by `χ_n`, so padding rows are unconstrained on the bits;
//! claim 3 forces `sel = 0` on every padding row (`χ_n = 0` there) even if that
//! padding row's `V` would satisfy the comparison.
//!
//! # SOUNDNESS PRECONDITION — the caller MUST guarantee `V_i ∈ [0, 2^B)`
//! The recomposition binds `Σ_j 2^j·b_j = E_i` in the FIELD (mod `r`). With `B+1`
//! boolean bits the left side is a genuine integer in `[0, 2^{B+1})`; the claim can
//! only hold if `E_i mod r` also lands in `[0, 2^{B+1})`. For an honest in-range
//! `V_i` it does, and the unique boolean decomposition IS `binary(E_i)`, so `b_B`
//! is the true sign. But a value column can commit ANY field element: a `V_i` near
//! the modulus (`V_i ≈ r`, canonically HUGE, hence canonically `≥ c`) makes
//! `E_i = V_i − c + 2^B` WRAP to a small residue whose top bit is 0, letting the
//! prover honestly decode `sel = [V < c]` — the WRONG answer. This gadget does NOT
//! range-check `V` (a second `B`-bit decomposition would; see the report). In the
//! real system this precondition is structural: dictionary value encodings are
//! order-preserving integers of ≤ 8 bytes (`< 2^64`), so choosing `B ≥ 64` on an
//! honest root-anchored layer guarantees it. `c ∈ [0, 2^B)` IS checked (public).
//! We also require `2^{B+1} ≤ r`, i.e. `B ≤ 252` on BN254 (`r ≈ 2^253.6`).
//! [`out_of_range_v_breaks_the_sign_precondition`] demonstrates the bound is
//! load-bearing.
//!
//! CP-B1 scope: the STANDALONE gadget only. It is NOT yet wired into the
//! dict/atom/BGP pipeline (CP-B2 / CP-C).

use std::io;

use ark_ff::{BigInteger, PrimeField};
use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::transcript::Transcript;

use super::argument::{
    commit_column, prove_column_openings, prove_sumcheck, verify_column_openings, verify_sumcheck,
    ClaimShape, ColumnOpening, Expr, Operand, SumcheckClaim, SumcheckProof, SumcheckShape, Term,
    Transcript254,
};
use super::invalid;
use super::sort::{booleanity_claim, low_bits};

// Column layout: V, then the B+1 bit columns b_0..b_B, then sel.
const COL_V: usize = 0;
const COL_BIT_BASE: usize = 1;
// chi_bounds index for chi_n (the real-row prefix [0, n)).
const CHI_N: usize = 0;

/// Largest supported bit-width. The offset difference is a `(B+1)`-bit number, so
/// `2^{B+1} ≤ r` (BN254 scalar modulus `r ≈ 2^253.6`) requires `B ≤ 252`.
const MAX_B_BITS: usize = 252;

const LABEL_DOMAIN: &[u8] = b"pow/range/domain-v1";
const LABEL_STMT: &[u8] = b"pow/range/statement";
const LABEL_TAU: &[u8] = b"pow/range/booleanity-tau";

#[inline]
fn f0() -> ArkFr {
    ArkFr::zero()
}
#[inline]
fn f1() -> ArkFr {
    ArkFr::one()
}

/// `2^exp` in the field, by repeated doubling (works for any `exp`, unlike a
/// `1u64 << exp` that overflows past 63 — `B` can be as large as 252).
fn pow2(exp: usize) -> ArkFr {
    let two = f1() + f1();
    let mut acc = f1();
    for _ in 0..exp {
        acc = acc * two;
    }
    acc
}

/// Whether a field element's canonical integer representative is `< 2^num_bits`,
/// i.e. lies in `[0, 2^num_bits)`.
fn fits_in_bits(value: ArkFr, num_bits: usize) -> bool {
    (value.0.into_bigint().num_bits() as usize) <= num_bits
}

/// A numeric comparison operator against a public constant `c`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RangeOp {
    /// `V < c`
    Lt,
    /// `V <= c`
    Le,
    /// `V > c`
    Gt,
    /// `V >= c`
    Ge,
}

impl RangeOp {
    /// `true` iff the offset difference is `E' = c − V + 2^B` (the `<=`/`>` family);
    /// `false` for `E = V − c + 2^B` (the `>=`/`<` family).
    fn use_neg_diff(self) -> bool {
        matches!(self, RangeOp::Le | RangeOp::Gt)
    }

    /// `true` iff `sel = χ_n·(1 − b_B)` (the strict `<`/`>` operators); `false` iff
    /// `sel = χ_n·b_B` (the inclusive `>=`/`<=` operators).
    fn complement_sign(self) -> bool {
        matches!(self, RangeOp::Lt | RangeOp::Gt)
    }

    fn tag(self) -> u8 {
        match self {
            RangeOp::Lt => 0,
            RangeOp::Le => 1,
            RangeOp::Gt => 2,
            RangeOp::Ge => 3,
        }
    }
}

/// A standalone range/sign selection proof: the batched sumcheck over the
/// booleanity + recomposition + sign→sel claims, one batched Dory opening over all
/// committed columns (`V`, the `B+1` bits, and `sel`), and the internal bit
/// commitments the verifier binds. `Com(sel)` is returned separately by the prover
/// and passed back into the verifier (the authoritative output-column commitment).
#[derive(Clone, Debug)]
pub struct RangeSelectProof {
    pub sumcheck: SumcheckProof,
    pub opening: ColumnOpening,
    pub bit_commitments: Vec<ArkGT>,
}

/// The index of the `sel` column for a given bit-width (`V` + `B+1` bits precede it).
fn col_sel(b_bits: usize) -> usize {
    COL_BIT_BASE + b_bits + 1
}

/// `χ_n · (Σ_{j=0}^{B} 2^j·b_j − diff) = 0`, recomposing the committed bits to the
/// offset difference and thereby binding them to the committed `V` and public `c`:
///   * `use_neg_diff = false`: `diff = E  = V − c + 2^B`  ⇒  `Σ 2^j b_j − V + (c − 2^B)`
///   * `use_neg_diff = true` : `diff = E' = c − V + 2^B`  ⇒  `Σ 2^j b_j + V − (c + 2^B)`
fn recomposition_claim(b_bits: usize, c: ArkFr, use_neg_diff: bool) -> SumcheckClaim {
    let two = f1() + f1();
    let two_pow_b = pow2(b_bits);
    let mut terms = Vec::with_capacity(b_bits + 3);
    let mut power = f1();
    for j in 0..=b_bits {
        terms.push(Term {
            coeff: power,
            operands: vec![Operand::Chi(CHI_N), Operand::Column(COL_BIT_BASE + j)],
        });
        power = power * two;
    }
    if use_neg_diff {
        // Σ 2^j b_j + V − (c + 2^B) = 0
        terms.push(Term {
            coeff: f1(),
            operands: vec![Operand::Chi(CHI_N), Operand::Column(COL_V)],
        });
        terms.push(Term {
            coeff: f0() - (c + two_pow_b),
            operands: vec![Operand::Chi(CHI_N)],
        });
    } else {
        // Σ 2^j b_j − V + (c − 2^B) = 0
        terms.push(Term {
            coeff: f0() - f1(),
            operands: vec![Operand::Chi(CHI_N), Operand::Column(COL_V)],
        });
        terms.push(Term {
            coeff: c - two_pow_b,
            operands: vec![Operand::Chi(CHI_N)],
        });
    }
    SumcheckClaim {
        shape: ClaimShape::Identity,
        expr: Expr { terms },
        target: f0(),
    }
}

/// Ties `sel` to the sign bit `b_B` and masks padding:
///   * `complement = false`: `sel − χ_n·b_B = 0`              (`>=`, `<=`)
///   * `complement = true` : `sel − χ_n + χ_n·b_B = 0`       (`<`, `>`)
/// With booleanity of `b_B` this also makes `sel` boolean on real rows, and forces
/// `sel = 0` on every padding row (`χ_n = 0`).
fn sign_to_sel_claim(b_bits: usize, complement: bool) -> SumcheckClaim {
    let sign = Operand::Column(COL_BIT_BASE + b_bits); // b_B, the top bit
    let sel = Operand::Column(col_sel(b_bits));
    let mut terms = vec![Term {
        coeff: f1(),
        operands: vec![sel],
    }];
    if complement {
        terms.push(Term {
            coeff: f0() - f1(),
            operands: vec![Operand::Chi(CHI_N)],
        });
        terms.push(Term {
            coeff: f1(),
            operands: vec![Operand::Chi(CHI_N), sign],
        });
    } else {
        terms.push(Term {
            coeff: f0() - f1(),
            operands: vec![Operand::Chi(CHI_N), sign],
        });
    }
    SumcheckClaim {
        shape: ClaimShape::Identity,
        expr: Expr { terms },
        target: f0(),
    }
}

fn range_shape(
    k: usize,
    real_n: usize,
    b_bits: usize,
    op: RangeOp,
    c: ArkFr,
    tau: ArkFr,
) -> SumcheckShape {
    let num_bits = b_bits + 1;
    SumcheckShape {
        num_vars: k,
        num_columns: col_sel(b_bits) + 1,
        chi_bounds: vec![real_n],
        claims: vec![
            booleanity_claim(CHI_N, COL_BIT_BASE, num_bits, tau),
            recomposition_claim(b_bits, c, op.use_neg_diff()),
            sign_to_sel_claim(b_bits, op.complement_sign()),
        ],
    }
}

/// Bind the PUBLIC statement (op, constant, bit-width, real-row count) so the
/// Fiat-Shamir challenges depend on the whole comparison being proven.
fn bind_statement(transcript: &mut Transcript254, op: RangeOp, c: ArkFr, b_bits: usize, real_n: usize) {
    transcript.append_bytes(LABEL_DOMAIN, b"parture/proof-of-woql/range-select/v0");
    transcript.append_bytes(LABEL_STMT, &[op.tag()]);
    transcript.append_field(LABEL_STMT, &c);
    transcript.append_bytes(LABEL_STMT, &(b_bits as u64).to_le_bytes());
    transcript.append_bytes(LABEL_STMT, &(real_n as u64).to_le_bytes());
}

/// The honest witnesses for `op`: the `B+1` bit columns of the offset difference
/// over the real rows `[0, n)` (zero on padding rows), and the resulting `sel`
/// column. `v` is the full padded value column (length `N = 2^k`).
fn honest_witness(
    v: &[ArkFr],
    real_n: usize,
    op: RangeOp,
    c: ArkFr,
    b_bits: usize,
) -> (Vec<Vec<ArkFr>>, Vec<ArkFr>) {
    let size = v.len();
    let num_bits = b_bits + 1;
    let two_pow_b = pow2(b_bits);
    let use_neg = op.use_neg_diff();
    let complement = op.complement_sign();
    let mut bit_columns = vec![vec![f0(); size]; num_bits];
    let mut sel = vec![f0(); size];
    for i in 0..real_n.min(size) {
        let diff = if use_neg {
            c - v[i] + two_pow_b
        } else {
            v[i] - c + two_pow_b
        };
        let bits = low_bits(diff, num_bits);
        for (j, bit) in bits.iter().enumerate() {
            bit_columns[j][i] = *bit;
        }
        let sign = bits[b_bits];
        sel[i] = if complement { f1() - sign } else { sign };
    }
    (bit_columns, sel)
}

/// Prove a range selection for an EXPLICIT `(bit_columns, sel)` witness. The honest
/// entry [`prove_range_select`] derives the witness and calls this; tests use it to
/// inject malicious witnesses. `v` is the full padded value column; `com_v` must
/// commit `v`. Returns the proof and `Com(sel)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_range_select_with_witness(
    v: &[ArkFr],
    com_v: &ArkGT,
    op: RangeOp,
    c: ArkFr,
    b_bits: usize,
    real_n: usize,
    bit_columns: &[Vec<ArkFr>],
    sel: &[ArkFr],
    transcript: &mut Transcript254,
) -> io::Result<(RangeSelectProof, ArkGT)> {
    let size = v.len();
    if !size.is_power_of_two() || size < 2 {
        return Err(invalid("range value column length must be a power of two >= 2"));
    }
    if !(1..=MAX_B_BITS).contains(&b_bits) {
        return Err(invalid("range bit-width out of range (need 1 <= B <= 252)"));
    }
    if real_n > size {
        return Err(invalid("range real-row count exceeds column length"));
    }
    let num_bits = b_bits + 1;
    if bit_columns.len() != num_bits || bit_columns.iter().any(|col| col.len() != size) {
        return Err(invalid("range bit column layout mismatch"));
    }
    if sel.len() != size {
        return Err(invalid("range selection column length mismatch"));
    }
    if !fits_in_bits(c, b_bits) {
        return Err(invalid("range constant c is not in [0, 2^B)"));
    }
    let k = size.trailing_zeros() as usize;

    // Round 1: commit the B+1 bit columns and sel (raw-data witnesses), bind them
    // (with the given com_v and the public statement) BEFORE drawing tau.
    let mut bit_commitments = Vec::with_capacity(num_bits);
    for col in bit_columns {
        bit_commitments.push(commit_column(col)?);
    }
    let sel_commitment = commit_column(sel)?;

    bind_statement(transcript, op, c, b_bits, real_n);
    transcript.append_group(b"pow/range/com-v", com_v);
    for com in &bit_commitments {
        transcript.append_group(b"pow/range/com-bit", com);
    }
    transcript.append_group(b"pow/range/com-sel", &sel_commitment);
    let tau = transcript.challenge_scalar(LABEL_TAU);

    let shape = range_shape(k, real_n, b_bits, op, c, tau);
    let mut columns = Vec::with_capacity(num_bits + 2);
    columns.push(v.to_vec());
    columns.extend(bit_columns.iter().cloned());
    columns.push(sel.to_vec());

    let (sumcheck, r, _evals) = prove_sumcheck(&shape, &columns, transcript)?;
    let opening = prove_column_openings(&columns, &r, transcript)?;

    Ok((
        RangeSelectProof {
            sumcheck,
            opening,
            bit_commitments,
        },
        sel_commitment,
    ))
}

/// Prove that a committed boolean column `sel` equals EXACTLY `[ V_i op c AND i < n ]`
/// for the value column committed by `com_v`. Commits the bit columns and `sel`
/// internally. Returns the proof and `Com(sel)` (the authoritative output-column
/// commitment the verifier is handed back).
///
/// PRECONDITION: every real `V_i` (and `c`) must be in `[0, 2^B)` — see the module
/// docs. `c ∈ [0, 2^B)` is enforced; the `V` bound is the caller's responsibility.
pub fn prove_range_select(
    v: &[ArkFr],
    com_v: &ArkGT,
    op: RangeOp,
    c: ArkFr,
    b_bits: usize,
    real_n: usize,
    transcript: &mut Transcript254,
) -> io::Result<(RangeSelectProof, ArkGT)> {
    let (bit_columns, sel) = honest_witness(v, real_n, op, c, b_bits);
    prove_range_select_with_witness(
        v, com_v, op, c, b_bits, real_n, &bit_columns, &sel, transcript,
    )
}

/// Verify a range selection proof. Re-derives the challenges, checks the batched
/// sumcheck (booleanity + recomposition + sign→sel) against the claimed evals and
/// `χ_n(r)`, and verifies the batched Dory opening of `[V, b_0..b_B, sel]` at `r`.
/// `com_sel` is the caller's authoritative selection-column commitment: it is bound
/// into the transcript and the opening, so a `com_sel` that does not equal the one
/// the proof was built over makes verification fail.
#[allow(clippy::too_many_arguments)]
pub fn verify_range_select(
    com_v: &ArkGT,
    op: RangeOp,
    c: ArkFr,
    b_bits: usize,
    real_n: usize,
    com_sel: &ArkGT,
    proof: &RangeSelectProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    let k = proof.sumcheck.round_polys.len();
    if k == 0 || k >= usize::BITS as usize {
        return Err(invalid("range proof has an implausible variable count"));
    }
    if !(1..=MAX_B_BITS).contains(&b_bits) {
        return Err(invalid("range bit-width out of range (need 1 <= B <= 252)"));
    }
    let num_bits = b_bits + 1;
    if proof.bit_commitments.len() != num_bits {
        return Err(invalid("range proof has the wrong number of bit commitments"));
    }
    if real_n > (1usize << k) {
        return Err(invalid("range real-row count exceeds column length"));
    }
    if !fits_in_bits(c, b_bits) {
        return Err(invalid("range constant c is not in [0, 2^B)"));
    }

    bind_statement(transcript, op, c, b_bits, real_n);
    transcript.append_group(b"pow/range/com-v", com_v);
    for com in &proof.bit_commitments {
        transcript.append_group(b"pow/range/com-bit", com);
    }
    transcript.append_group(b"pow/range/com-sel", com_sel);
    let tau = transcript.challenge_scalar(LABEL_TAU);

    let shape = range_shape(k, real_n, b_bits, op, c, tau);
    let (r, evals) = verify_sumcheck(&shape, &proof.sumcheck, transcript)?;

    let mut commitments = Vec::with_capacity(num_bits + 2);
    commitments.push(*com_v);
    commitments.extend(proof.bit_commitments.iter().copied());
    commitments.push(*com_sel);
    verify_column_openings(&commitments, &r, &evals, &proof.opening, transcript)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_OPS: [RangeOp; 4] = [RangeOp::Lt, RangeOp::Le, RangeOp::Gt, RangeOp::Ge];

    fn fr(v: u64) -> ArkFr {
        ArkFr::from_u64(v)
    }

    fn transcript() -> Transcript254 {
        Transcript254::new(b"parture-range-test")
    }

    fn commit(column: &[ArkFr]) -> ArkGT {
        commit_column(column).unwrap()
    }

    /// A padded value column of length `size` (a power of two) from `real` ids; the
    /// padding tail is filled with `pad`.
    fn value_column(real: &[u64], size: usize, pad: u64) -> Vec<ArkFr> {
        assert!(size.is_power_of_two() && real.len() <= size);
        (0..size)
            .map(|i| fr(if i < real.len() { real[i] } else { pad }))
            .collect()
    }

    fn cmp_op(op: RangeOp, a: u64, b: u64) -> bool {
        match op {
            RangeOp::Lt => a < b,
            RangeOp::Le => a <= b,
            RangeOp::Gt => a > b,
            RangeOp::Ge => a >= b,
        }
    }

    /// The reference boolean selection column computed directly over the canonical
    /// u64 values, masked to the first `real_n` rows.
    fn expected_sel(real: &[u64], size: usize, real_n: usize, op: RangeOp, c: u64) -> Vec<ArkFr> {
        (0..size)
            .map(|i| {
                if i < real_n && i < real.len() && cmp_op(op, real[i], c) {
                    f1()
                } else {
                    f0()
                }
            })
            .collect()
    }

    // Prove + verify an honest range selection and return the proven Com(sel).
    fn run(
        v: &[ArkFr],
        op: RangeOp,
        c: u64,
        b_bits: usize,
        real_n: usize,
    ) -> io::Result<ArkGT> {
        let com_v = commit(v);
        let mut pt = transcript();
        let (proof, com_sel) = prove_range_select(v, &com_v, op, fr(c), b_bits, real_n, &mut pt)?;
        let mut vt = transcript();
        verify_range_select(&com_v, op, fr(c), b_bits, real_n, &com_sel, &proof, &mut vt)?;
        Ok(com_sel)
    }

    #[test]
    fn all_ops_straddle_c_verify_and_match_expected() {
        let b_bits = 16;
        let c = 100u64;
        // Real values straddling c, INCLUDING the boundary V = c (rows 1, 3, 6).
        let real = [50u64, 100, 150, 100, 200, 30, 100, 175];
        let size = 8;
        let real_n = 8;
        for op in ALL_OPS {
            let v = value_column(&real, size, 0);
            let com_sel = run(&v, op, c, b_bits, real_n).unwrap();
            // Com(sel) equals an INDEPENDENTLY committed expected boolean column.
            let expected = expected_sel(&real, size, real_n, op, c);
            assert_eq!(com_sel, commit(&expected), "op {:?}", op);
        }
    }

    #[test]
    fn boundary_v_equals_c_all_ops() {
        // Every real value is exactly c: inclusive ops select all, strict ops none.
        let b_bits = 12;
        let c = 64u64;
        let real = [64u64, 64, 64];
        let size = 4;
        let real_n = 3;
        for (op, sel_at_boundary) in [
            (RangeOp::Ge, f1()),
            (RangeOp::Le, f1()),
            (RangeOp::Lt, f0()),
            (RangeOp::Gt, f0()),
        ] {
            let v = value_column(&real, size, 0);
            let com_sel = run(&v, op, c, b_bits, real_n).unwrap();
            let mut expected = vec![sel_at_boundary; real_n];
            expected.resize(size, f0()); // padding row -> 0
            assert_eq!(com_sel, commit(&expected), "op {:?}", op);
        }
    }

    #[test]
    fn padding_rows_get_sel_zero_even_if_value_satisfies() {
        let b_bits = 16;
        let c = 100u64;
        let real = [50u64, 150, 30, 200];
        let real_n = 4;
        let size = 8;
        let op = RangeOp::Ge;
        // Padding rows (>= real_n) hold 500, which WOULD satisfy `>= 100`.
        let v = value_column(&real, size, 500);
        let com_sel = run(&v, op, c, b_bits, real_n).unwrap();
        // Honest sel is 0 on every padding row despite V = 500 >= 100.
        let expected = expected_sel(&real, size, real_n, op, c);
        for i in real_n..size {
            assert_eq!(expected[i], f0());
        }
        assert_eq!(com_sel, commit(&expected));

        // A prover that SELECTS a padding row is rejected.
        let com_v = commit(&v);
        let (bits, mut sel) = honest_witness(&v, real_n, op, fr(c), b_bits);
        sel[5] = f1();
        let mut pt = transcript();
        let (proof, com_sel2) =
            prove_range_select_with_witness(&v, &com_v, op, fr(c), b_bits, real_n, &bits, &sel, &mut pt)
                .unwrap();
        let mut vt = transcript();
        assert!(
            verify_range_select(&com_v, op, fr(c), b_bits, real_n, &com_sel2, &proof, &mut vt)
                .is_err()
        );
    }

    #[test]
    fn non_boolean_bit_is_rejected() {
        let b_bits = 16;
        let c = 100u64;
        let real = [50u64, 150, 100, 200];
        let real_n = 4;
        let size = 4;
        let op = RangeOp::Ge;
        let v = value_column(&real, size, 0);
        let com_v = commit(&v);
        let (mut bits, sel) = honest_witness(&v, real_n, op, fr(c), b_bits);
        // A non-{0,1} value in a real row's low bit -> booleanity rejects.
        bits[0][1] = fr(2);
        let mut pt = transcript();
        let (proof, com_sel) =
            prove_range_select_with_witness(&v, &com_v, op, fr(c), b_bits, real_n, &bits, &sel, &mut pt)
                .unwrap();
        let mut vt = transcript();
        assert!(
            verify_range_select(&com_v, op, fr(c), b_bits, real_n, &com_sel, &proof, &mut vt).is_err()
        );
    }

    #[test]
    fn non_boolean_sign_bit_is_rejected() {
        // Specifically the SIGN bit b_B: a non-{0,1} top bit that still hits the
        // recomposition target would decode a false sign if booleanity omitted it.
        let b_bits = 8;
        let c = 100u64;
        let real = [150u64, 50, 100, 200];
        let real_n = 4;
        let size = 4;
        let op = RangeOp::Ge;
        let v = value_column(&real, size, 0);
        let com_v = commit(&v);
        let (mut bits, sel) = honest_witness(&v, real_n, op, fr(c), b_bits);
        // Corrupt b_B on row 0 (a genuine match, honest b_B = 1) to a non-bit value.
        assert_eq!(bits[b_bits][0], f1());
        bits[b_bits][0] = fr(5);
        let mut pt = transcript();
        let (proof, com_sel) =
            prove_range_select_with_witness(&v, &com_v, op, fr(c), b_bits, real_n, &bits, &sel, &mut pt)
                .unwrap();
        let mut vt = transcript();
        assert!(
            verify_range_select(&com_v, op, fr(c), b_bits, real_n, &com_sel, &proof, &mut vt).is_err()
        );
    }

    #[test]
    fn bits_not_recomposing_to_v_is_rejected() {
        // Boolean bits that do NOT recompose to E -> recomposition binds V and rejects.
        let b_bits = 16;
        let c = 100u64;
        let real = [50u64, 150, 100, 200];
        let real_n = 4;
        let size = 4;
        let op = RangeOp::Ge;
        let v = value_column(&real, size, 0);
        let com_v = commit(&v);
        let (mut bits, sel) = honest_witness(&v, real_n, op, fr(c), b_bits);
        // Flip a low bit of a real row: still boolean, but no longer = binary(E).
        bits[0][0] = f1() - bits[0][0];
        let mut pt = transcript();
        let (proof, com_sel) =
            prove_range_select_with_witness(&v, &com_v, op, fr(c), b_bits, real_n, &bits, &sel, &mut pt)
                .unwrap();
        let mut vt = transcript();
        assert!(
            verify_range_select(&com_v, op, fr(c), b_bits, real_n, &com_sel, &proof, &mut vt).is_err()
        );
    }

    #[test]
    fn wrong_sel_contradicting_the_sign_is_rejected() {
        let b_bits = 16;
        let c = 100u64;
        let real = [50u64, 150, 100, 200];
        let real_n = 4;
        let size = 4;
        let v = value_column(&real, size, 0);
        let com_v = commit(&v);
        for op in ALL_OPS {
            let (bits, mut sel) = honest_witness(&v, real_n, op, fr(c), b_bits);
            // Flip sel on a real row -> sign->sel claim rejects.
            sel[1] = f1() - sel[1];
            let mut pt = transcript();
            let (proof, com_sel) = prove_range_select_with_witness(
                &v, &com_v, op, fr(c), b_bits, real_n, &bits, &sel, &mut pt,
            )
            .unwrap();
            let mut vt = transcript();
            assert!(
                verify_range_select(&com_v, op, fr(c), b_bits, real_n, &com_sel, &proof, &mut vt)
                    .is_err(),
                "op {:?}", op
            );
        }
    }

    #[test]
    fn wrong_c_at_verify_is_rejected() {
        let b_bits = 16;
        let c = 100u64;
        let real = [50u64, 150, 100, 200];
        let real_n = 4;
        let size = 4;
        let op = RangeOp::Ge;
        let v = value_column(&real, size, 0);
        let com_v = commit(&v);
        let mut pt = transcript();
        let (proof, com_sel) =
            prove_range_select(&v, &com_v, op, fr(c), b_bits, real_n, &mut pt).unwrap();
        // Verifying against a different public constant rejects (statement + shape
        // both diverge from the transcript the proof was built against).
        let mut vt = transcript();
        assert!(
            verify_range_select(&com_v, op, fr(c + 1), b_bits, real_n, &com_sel, &proof, &mut vt)
                .is_err()
        );
    }

    #[test]
    fn wrong_com_sel_at_verify_is_rejected() {
        // The caller's authoritative Com(sel) is bound in: a mismatched one rejects.
        let b_bits = 16;
        let c = 100u64;
        let real = [50u64, 150, 100, 200];
        let real_n = 4;
        let size = 4;
        let op = RangeOp::Ge;
        let v = value_column(&real, size, 0);
        let com_v = commit(&v);
        let mut pt = transcript();
        let (proof, _com_sel) =
            prove_range_select(&v, &com_v, op, fr(c), b_bits, real_n, &mut pt).unwrap();
        let bogus = commit(&vec![f0(); size]); // not the proven sel
        let mut vt = transcript();
        assert!(
            verify_range_select(&com_v, op, fr(c), b_bits, real_n, &bogus, &proof, &mut vt).is_err()
        );
    }

    #[test]
    fn out_of_range_v_breaks_the_sign_precondition() {
        // DEMONSTRATES that the caller's `V_i ∈ [0, 2^B)` precondition is
        // load-bearing (this gadget does NOT range-check V). We commit a value near
        // the modulus (canonically HUGE, hence canonically >= c). The offset
        // difference V - c + 2^B WRAPS mod r to a small residue whose top bit is 0,
        // so the honest prover decodes sel = [V < c] -- the WRONG answer for the
        // canonical V -- and it VERIFIES. In the real system dict OP values are
        // < 2^64 <= 2^B, so this cannot occur.
        let b_bits = 8;
        let c = 100u64;
        let op = RangeOp::Ge;
        let two_pow_b = fr(1u64 << b_bits); // 256
                                            // t = 50: bit 8 clear, so the decoded sign says "V < c". V = t + c - 2^B (mod r).
        let v0 = fr(50) + fr(c) - two_pow_b; // = r - 106, a huge canonical integer >= c
        assert!(!fits_in_bits(v0, b_bits)); // V is genuinely out of [0, 2^B)
        let size = 4;
        let real_n = 1;
        let v = vec![v0, f0(), f0(), f0()];
        let com_sel = run(&v, op, c, b_bits, real_n).unwrap();
        // It verified, and the proven sel is all-zero ("nothing >= c") -- WRONG,
        // because the canonical v0 = r - 106 is far larger than c = 100.
        assert_eq!(com_sel, commit(&vec![f0(); size]));
    }

    #[test]
    fn rejects_degenerate_bit_widths() {
        let real = [1u64, 2, 3, 4];
        let size = 4;
        let real_n = 4;
        let v = value_column(&real, size, 0);
        let com_v = commit(&v);
        let mut pt = transcript();
        // B = 0 is invalid; B = 253 exceeds the field bound.
        assert!(prove_range_select(&v, &com_v, RangeOp::Ge, fr(2), 0, real_n, &mut pt).is_err());
        assert!(prove_range_select(&v, &com_v, RangeOp::Ge, fr(2), 253, real_n, &mut pt).is_err());
    }

    #[test]
    fn empty_and_all_match_verify() {
        let b_bits = 16;
        let real = [10u64, 20, 30, 40, 50, 60, 70, 80];
        let size = 8;
        let real_n = 8;
        // Empty: nothing is >= 1000.
        let v = value_column(&real, size, 0);
        let com_sel = run(&v, RangeOp::Ge, 1000, b_bits, real_n).unwrap();
        assert_eq!(com_sel, commit(&vec![f0(); size]));
        // All match: everything is >= 0 ... use `< 1000` so every real row is selected.
        let com_sel_all = run(&v, RangeOp::Lt, 1000, b_bits, real_n).unwrap();
        let expected = expected_sel(&real, size, real_n, RangeOp::Lt, 1000);
        assert_eq!(com_sel_all, commit(&expected));
    }
}
