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
//! The object-range composition below authenticates the dictionary lookup and
//! wires this gadget into the atom/BGP query pipeline.

use std::io;

use ark_ff::{BigInteger, PrimeField};
use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::transcript::Transcript;

use super::argument::{
    mle_full_eval, prove_sumcheck, verify_column_openings, verify_sumcheck,
    ClaimShape, ColumnOpening, Expr, Operand, StoreProverContext, SumcheckClaim, SumcheckProof,
    SumcheckShape, Term, Transcript254,
};
#[cfg(test)]
use super::argument::commit_column;
use super::filter::{self, FilterProof};
use super::invalid;
use super::logup_gkr::{
    bind_and_verify_fraction_tree, check_leaf_tie, prove_fraction_sum_zero, FractionInstance,
    GkrFractionProof, LinearForm, LinearTerm, Src,
};
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
#[cfg(test)]
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
    prove_range_select_with_witness_and_context(
        &StoreProverContext::cpu(), v, com_v, op, c, b_bits, real_n, bit_columns, sel, transcript,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_range_select_with_witness_and_context(
    context: &StoreProverContext<'_>,
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
        bit_commitments.push(context.commit_column(col)?);
    }
    let sel_commitment = context.commit_column(sel)?;

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
    let opening = context.prove_column_openings(&columns, &r, transcript)?;

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
    prove_range_select_with_context(
        &StoreProverContext::cpu(), v, com_v, op, c, b_bits, real_n, transcript,
    )
}

#[allow(clippy::too_many_arguments)]
/// As [`prove_range_select`], with an explicit prover-arithmetic context.
pub fn prove_range_select_with_context(
    context: &StoreProverContext<'_>,
    v: &[ArkFr],
    com_v: &ArkGT,
    op: RangeOp,
    c: ArkFr,
    b_bits: usize,
    real_n: usize,
    transcript: &mut Transcript254,
) -> io::Result<(RangeSelectProof, ArkGT)> {
    let (bit_columns, sel) = honest_witness(v, real_n, op, c, b_bits);
    prove_range_select_with_witness_and_context(
        context, v, com_v, op, c, b_bits, real_n, &bit_columns, &sel, transcript,
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

// ===========================================================================
// CP-B2: authenticated node/value dict lookup (obj_id -> value, datatype).
// ===========================================================================
//
// Given a committed OBJECT-ID column `obj_id` (`N = 2^k` rows, `real_n` real) and
// the committed node/value dictionary as three ROW-ALIGNED columns
// `dict_id` / `dict_value` / `dict_datatype` (`n_dict` real rows, padded to the
// SAME `2^k` domain), this proves the prover-committed witness columns `v_val`,
// `v_dt` are the GENUINE dict entries for each row's object id:
//   `(obj_id[i], v_val[i], v_dt[i]) == (dict_id[j], dict_value[j], dict_datatype[j])`
// for the dict row `j` whose id is `obj_id[i]`, for every real row `i < real_n`.
//
// # The lookup (LogUp-GKR, block-disjoint two-side layout + GAP-1 auxiliary Identity)
// Draw a fold challenge `β` (after committing every data column) and fingerprint
// each side's `(id, value, datatype)` triple into ONE scalar per row:
//   * query fold  `q[i]  = obj_id[i]  + β·v_val[i]  + β²·v_dt[i]`
//   * dict  fold  `d[j]  = dict_id[j] + β·dict_value[j] + β²·dict_datatype[j]`
// The log-derivative identity pinning `v_val`/`v_dt` (α drawn after β) is
//   `Σ_{j<n_dict} w[j]/(α − d[j])  −  Σ_{i<real_n} 1/(α − q[i]) = 0`,
// where the committed MULTIPLICITY column `w[j]` = #{ real query rows whose id is dict
// row `j` }. This is a multi-column generalization of [`super::membership`] (the DICT
// side is the "universe" weighted by `w`, the QUERY side is the "image" counted once
// per real row). It is proved by the GKR fraction tree ([`super::logup_gkr`]) over a
// `2·domain` leaf vector laid out BLOCK-DISJOINT — NO reciprocal column (`rec_q`/`rec_d`)
// is committed:
//   * the DICT rows in the LOW half `[0, domain)` — denom `α − d[j]`, numerator the
//     committed multiplicity `+w[j]`;
//   * the QUERY rows in the HIGH half `[domain, 2·domain)` — denom `α − q[i]`, numerator
//     the ADDITIVE public block indicator `−[domain ≤ i < domain + real_n]`.
// The committed DATA columns stay length `domain`; the prover feeds the tree their
// zero-extensions (dict cols + `w` into the low half, query cols into the high half) and
// the verifier reconstructs each `2·domain` leaf-MLE from the length-`domain` opening via
// the top-bit factor (`(1 − r_high)` low, `r_high` high) — structurally forcing each
// side's fold columns to zero in the OTHER block.
//
// # GAP-1 (dict padding tail), closed UNCONDITIONALLY with a co-running Identity
// The reciprocal-era `Σ χ_{n_dict}·w·rec_d` masked the dict padding tail `[n_dict, domain)`
// out of the link. That mask is soundness-critical: without it a prover can STASH a value
// in the dict padding (commit `dict_id`/`dict_value`/`dict_datatype` on a row `j ≥ n_dict`
// crafted to equal a target query fold) and put a nonzero `w[j]` there, letting the padding
// pole `+w[j]/(α − d[j])` cancel a query pole for a NON-member — forging a lookup of a
// WRONG value. The forged pole sits at `α − (matchable value)`, so no "no real row has a
// pole at α" invariant closes it, and the boundary `n_dict` is ARBITRARY (non power of two),
// so a structural zero-extension of `w` cannot reproduce the mask. A `committed × public-Chi`
// mask is not a linear GKR leaf, so `χ_{n_dict}` is NOT put in the leaf; instead GAP-1 is
// closed by a CO-RUNNING auxiliary Identity over the SAME committed `w`:
//   `(1 − χ_{n_dict}(x)) · w(x) = 0   for every x`,
// proved via the existing [`super::argument`] sumcheck (a product term the engine supports
// — NO reciprocal reintroduced). As an Identity it forces `w[x] = 0` at EVERY `x ≥ n_dict`
// for ARBITRARY `n_dict`, unconditionally; its `w` opening binds the SAME `com_w` on the
// SAME transcript as the GKR leaf tie, so the multiplicity the tree used is exactly the one
// pinned to zero on the tail — a forged padding `w` is rejected without any caller invariant.
//
// # WHY this forces v_val/v_dt correct (the security tie)
// At random α the link holds as a rational identity in α iff the query-fold multiset (one
// pole per real query row) equals the dict-fold multiset (dict row `j` weighted by `w[j]`,
// real rows only via the Identity). So every real query pole `q[i]` must be cancelled by
// some real dict pole `d[j]`. With `β` drawn AFTER commit, the fold
// `(id,val,dt) ↦ id + β·val + β²·dt` is injective w.h.p., so `q[i] == d[j]` forces
// `obj_id[i] = dict_id[j]`, `v_val[i] = dict_value[j]`, `v_dt[i] = dict_datatype[j]`.
// Because `dict_id` is UNIQUE per row (the dict is an id↔object bijection),
// `obj_id[i] = dict_id[j]` pins `j` to THE unique dict row for that id — so `v_val[i]`/
// `v_dt[i]` are forced to be that row's TRUE value/datatype. A prover that commits a WRONG
// `v_val[i]` produces a query pole that matches NO real dict pole, leaving an uncancelled
// pole → the root `P ≠ 0` → REJECT.

// Column layout for the lookup fraction instance (one `2^k` domain shared by both sides;
// each committed column is length `domain` and zero-extended into its half). The reciprocal
// witnesses `rec_q`/`rec_d` are GONE (LogUp-GKR migration).
const L_COL_OBJ_ID: usize = 0;
const L_COL_V_VAL: usize = 1;
const L_COL_V_DT: usize = 2;
const L_COL_DICT_ID: usize = 3;
const L_COL_DICT_VALUE: usize = 4;
const L_COL_DICT_DATATYPE: usize = 5;
const L_COL_W: usize = 6;
const L_NUM_COLUMNS: usize = 7;

const LOOKUP_DOMAIN: &[u8] = b"pow/obj-lookup/domain-v1";
const LOOKUP_STMT: &[u8] = b"pow/obj-lookup/statement";
const LOOKUP_BETA: &[u8] = b"pow/obj-lookup/fingerprint-beta";
const LOOKUP_ALPHA: &[u8] = b"pow/obj-lookup/reciprocal-alpha";

/// A batched authenticated dict-lookup proof (LogUp-GKR): the block-disjoint
/// fraction-sum-zero proof for `Σ_{j<n_dict} w/(α − d) − Σ_{i<real_n} 1/(α − q) = 0`, the
/// claimed evaluations of the seven DATA columns at `r_low` (the low `k` bits of the GKR
/// leaf point), one batched Dory opening of those columns, and the co-running GAP-1 Identity
/// `(1 − χ_{n_dict})·w = 0` (its own sumcheck + `w` opening, both bound to `com_w` on the
/// shared transcript). NO reciprocal (`rec_q`/`rec_d`) column exists. `com_v_val` /
/// `com_v_dt` are the AUTHENTICATED value/datatype columns the composing gadget (range on
/// `v_val`, datatype-mask on `v_dt`) reuses as their committed inputs.
#[derive(Clone, Debug)]
pub struct DictLookupProof {
    pub gkr: GkrFractionProof,
    /// Evaluations of `[obj_id, v_val, v_dt, dict_id, dict_value, dict_datatype, w]` at
    /// `r_low`, bound to their commitments by `opening`.
    pub col_evals: Vec<ArkFr>,
    pub opening: ColumnOpening,
    /// GAP-1 auxiliary Identity `(1 − χ_{n_dict})·w = 0` sumcheck + its `w` opening.
    pub id_sumcheck: SumcheckProof,
    pub id_opening: ColumnOpening,
    pub com_v_val: ArkGT,
    pub com_v_dt: ArkGT,
    pub com_w: ArkGT,
}

/// Zero-extend a length-`size` column into the LOW half of a `2·size` leaf vector; the high
/// half stays zero. MLE at `(r_low, r_high)` is `(1 − r_high)·MLE_size(col, r_low)`.
fn low_half(col: &[ArkFr], leaf_count: usize) -> Vec<ArkFr> {
    let mut ext = vec![f0(); leaf_count];
    ext[..col.len()].copy_from_slice(col);
    ext
}

/// Zero-extend a length-`size` column into the HIGH half of a `2·size` leaf vector; the low
/// half stays zero. MLE at `(r_low, r_high)` is `r_high·MLE_size(col, r_low)`.
fn high_half(col: &[ArkFr], leaf_count: usize) -> Vec<ArkFr> {
    let mut ext = vec![f0(); leaf_count];
    ext[leaf_count / 2..].copy_from_slice(col);
    ext
}

/// The PUBLIC block-disjoint fraction-sum-zero instance over a length-`2·domain` leaf vector
/// (`domain = 2^k`): DICT rows in the low half `[0, domain)`, QUERY rows in the high half
/// `[domain, 2·domain)`. Built IDENTICALLY by prover and verifier.
///   * denominator `q_leaf = α − (dict_id + β·dict_value + β²·dict_datatype)` on the low half
///     and `α − (obj_id + β·v_val + β²·v_dt)` on the high half — block-disjoint, single α;
///   * numerator `p_leaf = +w[j]` (committed multiplicity, low half) `− [domain ≤ i < domain
///     + real_n]` (public prefix block indicator, high half); padding rows self-annihilate
///     (`p = 0`), and the dict padding tail is pinned `w = 0` by the co-running Identity.
fn lookup_fraction_instance(alpha: ArkFr, beta: ArkFr, real_n: usize, k: usize) -> FractionInstance {
    let domain = 1usize << k;
    let b1 = beta;
    let b2 = beta * beta;
    // Fold order [id, value, datatype] -> id + β·value + β²·datatype on both sides.
    let q_terms = vec![
        // Dict fold (low half): α − dict_id − β·dict_value − β²·dict_datatype.
        LinearTerm { coeff: f0() - f1(), src: Src::Column(L_COL_DICT_ID) },
        LinearTerm { coeff: f0() - b1, src: Src::Column(L_COL_DICT_VALUE) },
        LinearTerm { coeff: f0() - b2, src: Src::Column(L_COL_DICT_DATATYPE) },
        // Query fold (high half): α − obj_id − β·v_val − β²·v_dt.
        LinearTerm { coeff: f0() - f1(), src: Src::Column(L_COL_OBJ_ID) },
        LinearTerm { coeff: f0() - b1, src: Src::Column(L_COL_V_VAL) },
        LinearTerm { coeff: f0() - b2, src: Src::Column(L_COL_V_DT) },
    ];
    // p = +w (low half) − ([i < domain + real_n] − [i < domain]) = −1 on [domain, domain+real_n).
    let p_terms = vec![
        LinearTerm { coeff: f1(), src: Src::Column(L_COL_W) },
        LinearTerm { coeff: f0() - f1(), src: Src::Prefix(domain + real_n) },
        LinearTerm { coeff: f1(), src: Src::Prefix(domain) },
    ];
    FractionInstance {
        num_vars: k + 1,
        num_cols: L_NUM_COLUMNS,
        p_form: LinearForm { constant: f0(), terms: p_terms },
        q_form: LinearForm { constant: alpha, terms: q_terms },
    }
}

/// The PUBLIC GAP-1 auxiliary Identity `(1 − χ_{n_dict}(x)) · w(x) = 0` over the length-
/// `domain = 2^k` committed multiplicity column (referenced as `Column(0)`), forcing
/// `w[x] = 0` at every `x ≥ n_dict`. Written as `+w − χ_{n_dict}·w` (a product term the
/// [`super::argument`] sumcheck supports — no reciprocal). Built IDENTICALLY by prover and
/// verifier.
fn dict_mask_identity_shape(k: usize, n_dict: usize) -> SumcheckShape {
    SumcheckShape {
        num_vars: k,
        num_columns: 1,
        chi_bounds: vec![n_dict],
        claims: vec![SumcheckClaim {
            shape: ClaimShape::Identity,
            expr: Expr {
                terms: vec![
                    Term { coeff: f1(), operands: vec![Operand::Column(0)] },
                    Term {
                        coeff: f0() - f1(),
                        operands: vec![Operand::Chi(0), Operand::Column(0)],
                    },
                ],
            },
            target: f0(),
        }],
    }
}

/// Bind the PUBLIC lookup statement (real query length, real dict length) so the
/// Fiat-Shamir challenges depend on both masks.
fn bind_lookup_statement(transcript: &mut Transcript254, real_n: usize, n_dict: usize) {
    transcript.append_bytes(LOOKUP_DOMAIN, b"parture/proof-of-woql/object-range-lookup/v0");
    transcript.append_bytes(LOOKUP_STMT, &(real_n as u64).to_le_bytes());
    transcript.append_bytes(LOOKUP_STMT, &(n_dict as u64).to_le_bytes());
}

/// The honest lookup witness: for each row, `v_val[i]`/`v_dt[i]` are the dict entry
/// of the real dict row whose id equals `obj_id[i]` (else the `0` sentinel — only
/// possible on padding rows, which are masked out), and `w[j]` counts the real
/// query rows (`i < real_n`) whose id is dict row `j`. A linear scan (dictionaries
/// are small at the relation level; CP-C can index it).
pub(crate) fn honest_dict_lookup(
    obj_id: &[ArkFr],
    dict_id: &[ArkFr],
    dict_value: &[ArkFr],
    dict_datatype: &[ArkFr],
    real_n: usize,
    n_dict: usize,
) -> (Vec<ArkFr>, Vec<ArkFr>, Vec<ArkFr>) {
    let size = obj_id.len();
    let real_dict = n_dict.min(dict_id.len());
    let mut v_val = vec![f0(); size];
    let mut v_dt = vec![f0(); size];
    for i in 0..size {
        for j in 0..real_dict {
            if dict_id[j] == obj_id[i] {
                v_val[i] = dict_value[j];
                v_dt[i] = dict_datatype[j];
                break;
            }
        }
    }
    let mut w = vec![f0(); size];
    for j in 0..real_dict {
        let mut count = 0u64;
        for id in obj_id.iter().take(real_n.min(size)) {
            if *id == dict_id[j] {
                count += 1;
            }
        }
        w[j] = ArkFr::from_u64(count);
    }
    (v_val, v_dt, w)
}

/// Prove an authenticated lookup for EXPLICIT witness columns `(v_val, v_dt, w)`.
/// The honest entry [`prove_dict_value_lookup`] derives them; the security tests use
/// this to inject a WRONG `v_val`/`v_dt`/`w`. All nine columns must share the same
/// power-of-two domain `2^k` as `obj_id` (the dict columns are pre-padded to it, and
/// their three commitments commit those padded columns).
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn prove_dict_value_lookup_with_witness(
    obj_id: &[ArkFr],
    com_obj_id: &ArkGT,
    v_val: &[ArkFr],
    v_dt: &[ArkFr],
    w: &[ArkFr],
    dict_id: &[ArkFr],
    dict_value: &[ArkFr],
    dict_datatype: &[ArkFr],
    com_dict_id: &ArkGT,
    com_dict_value: &ArkGT,
    com_dict_datatype: &ArkGT,
    real_n: usize,
    n_dict: usize,
    transcript: &mut Transcript254,
) -> io::Result<DictLookupProof> {
    prove_dict_value_lookup_with_witness_and_context(
        &StoreProverContext::cpu(), obj_id, com_obj_id, v_val, v_dt, w, dict_id, dict_value,
        dict_datatype, com_dict_id, com_dict_value, com_dict_datatype, real_n, n_dict, transcript,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_dict_value_lookup_with_witness_and_context(
    context: &StoreProverContext<'_>,
    obj_id: &[ArkFr],
    com_obj_id: &ArkGT,
    v_val: &[ArkFr],
    v_dt: &[ArkFr],
    w: &[ArkFr],
    dict_id: &[ArkFr],
    dict_value: &[ArkFr],
    dict_datatype: &[ArkFr],
    com_dict_id: &ArkGT,
    com_dict_value: &ArkGT,
    com_dict_datatype: &ArkGT,
    real_n: usize,
    n_dict: usize,
    transcript: &mut Transcript254,
) -> io::Result<DictLookupProof> {
    let size = obj_id.len();
    if !size.is_power_of_two() || size < 2 {
        return Err(invalid("lookup object-id column length must be a power of two >= 2"));
    }
    for column in [v_val, v_dt, w, dict_id, dict_value, dict_datatype] {
        if column.len() != size {
            return Err(invalid("lookup column length mismatch (all columns share one domain)"));
        }
    }
    if real_n > size || n_dict > size {
        return Err(invalid("lookup real length exceeds the column domain"));
    }
    let k = size.trailing_zeros() as usize;

    // Round 1: commit the witness DATA columns, bind the statement and every data
    // commitment, THEN draw the fold/reciprocal challenges. α/β are drawn AFTER every column
    // commitment is bound (the primitive's caller contract), so the fraction leaves are fixed
    // before any per-layer GKR challenge. No reciprocal column is committed anymore.
    let com_v_val = context.commit_column(v_val)?;
    let com_v_dt = context.commit_column(v_dt)?;
    let com_w = context.commit_column(w)?;
    bind_lookup_statement(transcript, real_n, n_dict);
    transcript.append_group(b"pow/obj-lookup/com-obj-id", com_obj_id);
    transcript.append_group(b"pow/obj-lookup/com-dict-id", com_dict_id);
    transcript.append_group(b"pow/obj-lookup/com-dict-value", com_dict_value);
    transcript.append_group(b"pow/obj-lookup/com-dict-datatype", com_dict_datatype);
    transcript.append_group(b"pow/obj-lookup/com-v-val", &com_v_val);
    transcript.append_group(b"pow/obj-lookup/com-v-dt", &com_v_dt);
    transcript.append_group(b"pow/obj-lookup/com-w", &com_w);
    let beta = transcript.challenge_scalar(LOOKUP_BETA);
    let alpha = transcript.challenge_scalar(LOOKUP_ALPHA);

    // (1) The block-disjoint LogUp-GKR fraction tree over the 2·domain zero-extensions:
    // dict cols + `w` fill the LOW half (denom α − d, numer +w), query cols the HIGH half
    // (denom α − q, numer additive −[real query row]). The tree lands at `r_leaf` (length
    // k + 1), whose low k bits are the opening point.
    let inst = lookup_fraction_instance(alpha, beta, real_n, k);
    let leaf_count = inst.size();
    let extended = vec![
        high_half(obj_id, leaf_count),
        high_half(v_val, leaf_count),
        high_half(v_dt, leaf_count),
        low_half(dict_id, leaf_count),
        low_half(dict_value, leaf_count),
        low_half(dict_datatype, leaf_count),
        low_half(w, leaf_count),
    ];
    let (gkr, r_leaf) = prove_fraction_sum_zero(&inst, &extended, transcript)?;

    // Single batched Dory opening of the length-`domain` DATA columns at r_low, bound to
    // their (root-anchored / shared) commitments on this transcript.
    let r_low = &r_leaf[..k];
    let columns = vec![
        obj_id.to_vec(),
        v_val.to_vec(),
        v_dt.to_vec(),
        dict_id.to_vec(),
        dict_value.to_vec(),
        dict_datatype.to_vec(),
        w.to_vec(),
    ];
    let col_evals: Vec<ArkFr> = columns.iter().map(|c| mle_full_eval(c, r_low)).collect();
    let opening = context.prove_column_openings(&columns, r_low, transcript)?;

    // (2) The GAP-1 auxiliary Identity (1 − χ_{n_dict})·w = 0, pinning w to zero on the dict
    // padding tail [n_dict, domain) UNCONDITIONALLY (arbitrary n_dict). Proved over the SAME
    // committed w via the argument.rs sumcheck; its opening binds com_w on the SAME transcript
    // as the GKR leaf tie, so the tree and the Identity constrain one identical w.
    let id_shape = dict_mask_identity_shape(k, n_dict);
    let w_col = vec![w.to_vec()];
    let (id_sumcheck, r_id, _evals) = prove_sumcheck(&id_shape, &w_col, transcript)?;
    let id_opening = context.prove_column_openings(&w_col, &r_id, transcript)?;

    Ok(DictLookupProof {
        gkr,
        col_evals,
        opening,
        id_sumcheck,
        id_opening,
        com_v_val,
        com_v_dt,
        com_w,
    })
}

/// Prove that the (internally committed) `v_val`/`v_dt` columns are the genuine dict
/// entries for each `obj_id[i]` (real rows). The dict columns must be pre-padded to
/// `obj_id`'s domain and committed there.
#[allow(clippy::too_many_arguments)]
pub fn prove_dict_value_lookup(
    obj_id: &[ArkFr],
    com_obj_id: &ArkGT,
    dict_id: &[ArkFr],
    dict_value: &[ArkFr],
    dict_datatype: &[ArkFr],
    com_dict_id: &ArkGT,
    com_dict_value: &ArkGT,
    com_dict_datatype: &ArkGT,
    real_n: usize,
    n_dict: usize,
    transcript: &mut Transcript254,
) -> io::Result<DictLookupProof> {
    prove_dict_value_lookup_with_context(
        &StoreProverContext::cpu(), obj_id, com_obj_id, dict_id, dict_value, dict_datatype,
        com_dict_id, com_dict_value, com_dict_datatype, real_n, n_dict, transcript,
    )
}

/// As [`prove_dict_value_lookup`], with an explicit prover-arithmetic context.
#[allow(clippy::too_many_arguments)]
pub fn prove_dict_value_lookup_with_context(
    context: &StoreProverContext<'_>,
    obj_id: &[ArkFr],
    com_obj_id: &ArkGT,
    dict_id: &[ArkFr],
    dict_value: &[ArkFr],
    dict_datatype: &[ArkFr],
    com_dict_id: &ArkGT,
    com_dict_value: &ArkGT,
    com_dict_datatype: &ArkGT,
    real_n: usize,
    n_dict: usize,
    transcript: &mut Transcript254,
) -> io::Result<DictLookupProof> {
    let (v_val, v_dt, w) =
        honest_dict_lookup(obj_id, dict_id, dict_value, dict_datatype, real_n, n_dict);
    prove_dict_value_lookup_with_witness_and_context(
        context, obj_id,
        com_obj_id,
        &v_val,
        &v_dt,
        &w,
        dict_id,
        dict_value,
        dict_datatype,
        com_dict_id,
        com_dict_value,
        com_dict_datatype,
        real_n,
        n_dict,
        transcript,
    )
}

/// Verify a batched authenticated dict-lookup proof. Concludes that the proof's
/// `com_v_val`/`com_v_dt` commit the TRUE dict `value`/`datatype` of each real
/// `obj_id` row (see the module security tie). The `com_v_val`/`com_v_dt` bound and
/// opened here are the SAME commitments the composing range/datatype gadgets consume,
/// so those gadgets range over authenticated values.
#[allow(clippy::too_many_arguments)]
pub fn verify_dict_value_lookup(
    com_obj_id: &ArkGT,
    com_dict_id: &ArkGT,
    com_dict_value: &ArkGT,
    com_dict_datatype: &ArkGT,
    real_n: usize,
    n_dict: usize,
    proof: &DictLookupProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    // The GKR tree has `k + 1` layers (`2·domain` leaves for a length-`domain = 2^k` value
    // domain), so `k = num_vars - 1` and the committed columns have length `2^k`. The column
    // openings below re-anchor `k` to the true commitments.
    let num_vars = proof.gkr.num_vars();
    if num_vars < 2 || num_vars >= usize::BITS as usize {
        return Err(invalid("lookup proof has an implausible variable count"));
    }
    let k = num_vars - 1;
    let domain = 1usize << k;
    if real_n > domain || n_dict > domain {
        return Err(invalid("lookup real length exceeds the column domain"));
    }
    if proof.col_evals.len() != L_NUM_COLUMNS {
        return Err(invalid("lookup proof has the wrong number of column evaluations"));
    }

    bind_lookup_statement(transcript, real_n, n_dict);
    transcript.append_group(b"pow/obj-lookup/com-obj-id", com_obj_id);
    transcript.append_group(b"pow/obj-lookup/com-dict-id", com_dict_id);
    transcript.append_group(b"pow/obj-lookup/com-dict-value", com_dict_value);
    transcript.append_group(b"pow/obj-lookup/com-dict-datatype", com_dict_datatype);
    transcript.append_group(b"pow/obj-lookup/com-v-val", &proof.com_v_val);
    transcript.append_group(b"pow/obj-lookup/com-v-dt", &proof.com_v_dt);
    transcript.append_group(b"pow/obj-lookup/com-w", &proof.com_w);
    let beta = transcript.challenge_scalar(LOOKUP_BETA);
    let alpha = transcript.challenge_scalar(LOOKUP_ALPHA);

    // (1) Bind the public instance and verify the GKR tree -> the leaf point + claims.
    let inst = lookup_fraction_instance(alpha, beta, real_n, k);
    let (r_leaf, p_claim, q_claim) = bind_and_verify_fraction_tree(&inst, &proof.gkr, transcript)?;

    // Open the length-`domain` committed columns at r_low, bound to their commitments.
    let r_low = &r_leaf[..k];
    let r_high = r_leaf[k];
    let commitments = [
        *com_obj_id,
        proof.com_v_val,
        proof.com_v_dt,
        *com_dict_id,
        *com_dict_value,
        *com_dict_datatype,
        proof.com_w,
    ];
    verify_column_openings(&commitments, r_low, &proof.col_evals, &proof.opening, transcript)?;

    // Reconstruct each 2·domain leaf-MLE from the length-`domain` opening via the top-bit
    // zero-extension factor: query cols (obj_id, v_val, v_dt) live in the HIGH half (factor
    // r_high), dict cols + w in the LOW half (factor 1 − r_high). This is the block-disjoint
    // enforcement (neither side's fold columns leak into the other block); then tie to the
    // GKR-reduced leaf claims.
    let one_minus_high = f1() - r_high;
    let leaf_evals = vec![
        r_high * proof.col_evals[L_COL_OBJ_ID],
        r_high * proof.col_evals[L_COL_V_VAL],
        r_high * proof.col_evals[L_COL_V_DT],
        one_minus_high * proof.col_evals[L_COL_DICT_ID],
        one_minus_high * proof.col_evals[L_COL_DICT_VALUE],
        one_minus_high * proof.col_evals[L_COL_DICT_DATATYPE],
        one_minus_high * proof.col_evals[L_COL_W],
    ];
    check_leaf_tie(&inst, &leaf_evals, p_claim, q_claim, &r_leaf)?;

    // (2) Verify the GAP-1 auxiliary Identity (1 − χ_{n_dict})·w = 0 over the SAME com_w. Its
    // `w` opening binds to com_w on this transcript, so the multiplicity the tree used is
    // EXACTLY the one pinned to zero on `[n_dict, domain)` — a forged padding multiplicity is
    // rejected UNCONDITIONALLY (no unmatched-pole invariant needed).
    let id_shape = dict_mask_identity_shape(k, n_dict);
    let (r_id, id_evals) = verify_sumcheck(&id_shape, &proof.id_sumcheck, transcript)?;
    verify_column_openings(&[proof.com_w], &r_id, &id_evals, &proof.id_opening, transcript)?;
    Ok(())
}

// ===========================================================================
// CP-B2: authenticated object-range selection (lookup + datatype mask + range).
// ===========================================================================
//
// Composition: `sel[i] = χ_n[i] · [datatype(obj_id[i]) == T] · [value(obj_id[i]) op c]`,
// where `value`/`datatype` are the DICT-AUTHENTICATED entries for `obj_id[i]`:
//   1. lookup       — `DictLookupProof` binds `com_v_val`/`com_v_dt` to the TRUE dict
//                     value/datatype of each real `obj_id` row.
//   2. datatype mask — `filter::prove_filter_indicator` on the committed `v_dt` with
//                     public constant `T` -> boolean `dt_match = [v_dt == T]` (χ_n-masked).
//   3. range        — `prove_range_select` on the committed `v_val` -> boolean
//                     `range_sel = [v_val op c]` (χ_n-masked; `b_bits = 64` since dict
//                     OP values are < 2^64, an honest root-anchored bound — see below).
//   4. combine      — Identity `sel − dt_match·range_sel = 0`, commit `sel`.
// All four run on ONE shared transcript, so every challenge downstream depends on the
// prior sub-proofs. The datatype mask is MANDATORY for cross-datatype soundness:
// without it a `v_val` of a DIFFERENT datatype than `T` would still be compared to `c`
// (a meaningless OP over a foreign encoding) and could be wrongly selected.
//
// SOUNDNESS PRECONDITION (range): real `v_val < 2^B` with `B = 64`. Structural: the
// dict value column holds order-preserving encodings of ≤ 8-byte numerics (`< 2^64`)
// and `0` sentinels — all `< 2^64`. This relies on the honest, root-anchored dict
// (the lookup authenticates `v_val` AGAINST that dict), so no committed `v_val` on a
// real row can be a huge near-modulus field element.

/// The bit-width the object-range gadget always uses: dict order-preserving VALUE
/// encodings are ≤ 8-byte numerics (`< 2^64`), so 64 bits suffice and the range
/// gadget's `V ∈ [0, 2^B)` precondition holds for every authenticated real value.
pub const OBJECT_RANGE_B_BITS: usize = 64;

const COMBINE_DOMAIN: &[u8] = b"pow/obj-combine/domain-v1";
const CB_COL_DT_MATCH: usize = 0;
const CB_COL_RANGE_SEL: usize = 1;
const CB_COL_SEL: usize = 2;

/// The `sel − dt_match·range_sel = 0` combine proof (Identity sumcheck, arity 2 ->
/// degree 3) plus its batched opening of `[dt_match, range_sel, sel]`.
#[derive(Clone, Debug)]
pub struct ObjectRangeCombineProof {
    pub sumcheck: SumcheckProof,
    pub opening: ColumnOpening,
}

/// The full authenticated object-range selection proof: the lookup sub-proof (which
/// carries `com_v_val`/`com_v_dt`), the datatype-mask indicator, the range sub-proof,
/// and the combine sub-proof. `Com(sel)` is returned separately (the authoritative
/// output-column commitment, passed back into the verifier).
#[derive(Clone, Debug)]
pub struct ObjectRangeProof {
    pub lookup: DictLookupProof,
    pub datatype: FilterProof,
    pub com_dt_match: ArkGT,
    pub com_dt_w: ArkGT,
    pub range: RangeSelectProof,
    pub com_range_sel: ArkGT,
    pub combine: ObjectRangeCombineProof,
}

fn combine_shape(k: usize) -> SumcheckShape {
    // sel − dt_match·range_sel = 0 (both inputs already boolean and χ_n-masked).
    let claim = SumcheckClaim {
        shape: ClaimShape::Identity,
        expr: Expr {
            terms: vec![
                Term {
                    coeff: f1(),
                    operands: vec![Operand::Column(CB_COL_SEL)],
                },
                Term {
                    coeff: f0() - f1(),
                    operands: vec![
                        Operand::Column(CB_COL_DT_MATCH),
                        Operand::Column(CB_COL_RANGE_SEL),
                    ],
                },
            ],
        },
        target: f0(),
    };
    SumcheckShape {
        num_vars: k,
        num_columns: 3,
        chi_bounds: Vec::new(),
        claims: vec![claim],
    }
}

/// Commit `sel = dt_match ⊙ range_sel` and prove the combine Identity over
/// `[dt_match, range_sel, sel]`. Returns the proof and `Com(sel)`.
fn prove_object_range_combine(
    context: &StoreProverContext<'_>,
    dt_match: &[ArkFr],
    range_sel: &[ArkFr],
    com_dt_match: &ArkGT,
    com_range_sel: &ArkGT,
    transcript: &mut Transcript254,
) -> io::Result<(ObjectRangeCombineProof, ArkGT)> {
    let size = dt_match.len();
    if range_sel.len() != size {
        return Err(invalid("combine column length mismatch"));
    }
    let sel: Vec<ArkFr> = dt_match
        .iter()
        .zip(range_sel.iter())
        .map(|(a, b)| *a * *b)
        .collect();
    let com_sel = context.commit_column(&sel)?;
    let k = size.trailing_zeros() as usize;

    transcript.append_bytes(COMBINE_DOMAIN, b"parture/proof-of-woql/object-range-combine/v0");
    transcript.append_group(b"pow/obj-combine/com-dt", com_dt_match);
    transcript.append_group(b"pow/obj-combine/com-range", com_range_sel);
    transcript.append_group(b"pow/obj-combine/com-sel", &com_sel);

    let shape = combine_shape(k);
    let columns = vec![dt_match.to_vec(), range_sel.to_vec(), sel];
    let (sumcheck, r, _evals) = prove_sumcheck(&shape, &columns, transcript)?;
    let opening = context.prove_column_openings(&columns, &r, transcript)?;
    Ok((ObjectRangeCombineProof { sumcheck, opening }, com_sel))
}

fn verify_object_range_combine(
    com_dt_match: &ArkGT,
    com_range_sel: &ArkGT,
    com_sel: &ArkGT,
    proof: &ObjectRangeCombineProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    let k = proof.sumcheck.round_polys.len();
    if k == 0 || k >= usize::BITS as usize {
        return Err(invalid("combine proof has an implausible variable count"));
    }
    transcript.append_bytes(COMBINE_DOMAIN, b"parture/proof-of-woql/object-range-combine/v0");
    transcript.append_group(b"pow/obj-combine/com-dt", com_dt_match);
    transcript.append_group(b"pow/obj-combine/com-range", com_range_sel);
    transcript.append_group(b"pow/obj-combine/com-sel", com_sel);

    let shape = combine_shape(k);
    let (r, evals) = verify_sumcheck(&shape, &proof.sumcheck, transcript)?;
    verify_column_openings(
        &[*com_dt_match, *com_range_sel, *com_sel],
        &r,
        &evals,
        &proof.opening,
        transcript,
    )
}

/// Prove an authenticated OBJECT-RANGE SELECTION: `Com(sel)` where
/// `sel[i] = χ_n[i] · [datatype(obj_id[i]) == T] · [value(obj_id[i]) op c]`, with
/// `value`/`datatype` the dict-authenticated entries of `obj_id[i]`. The dict
/// columns must be pre-padded to `obj_id`'s domain and committed there. Uses
/// `b_bits = 64` internally (dict OP values are `< 2^64`). Returns the proof and
/// `Com(sel)`.
#[allow(clippy::too_many_arguments)]
pub fn prove_object_range_select(
    obj_id: &[ArkFr],
    com_obj_id: &ArkGT,
    dict_id: &[ArkFr],
    dict_value: &[ArkFr],
    dict_datatype: &[ArkFr],
    com_dict_id: &ArkGT,
    com_dict_value: &ArkGT,
    com_dict_datatype: &ArkGT,
    n_dict: usize,
    datatype_t: ArkFr,
    op: RangeOp,
    c: ArkFr,
    b_bits: usize,
    real_n: usize,
    transcript: &mut Transcript254,
) -> io::Result<(ObjectRangeProof, ArkGT)> {
    prove_object_range_select_with_context(
        &StoreProverContext::cpu(), obj_id, com_obj_id, dict_id, dict_value, dict_datatype,
        com_dict_id, com_dict_value, com_dict_datatype, n_dict, datatype_t, op, c, b_bits, real_n,
        transcript,
    )
}

/// As [`prove_object_range_select`], with an explicit prover-arithmetic context.
#[allow(clippy::too_many_arguments)]
pub fn prove_object_range_select_with_context(
    context: &StoreProverContext<'_>,
    obj_id: &[ArkFr],
    com_obj_id: &ArkGT,
    dict_id: &[ArkFr],
    dict_value: &[ArkFr],
    dict_datatype: &[ArkFr],
    com_dict_id: &ArkGT,
    com_dict_value: &ArkGT,
    com_dict_datatype: &ArkGT,
    n_dict: usize,
    datatype_t: ArkFr,
    op: RangeOp,
    c: ArkFr,
    b_bits: usize,
    real_n: usize,
    transcript: &mut Transcript254,
) -> io::Result<(ObjectRangeProof, ArkGT)> {
    // Pin the bit-width: authenticated dict OP values are order-preserving numerics
    // `< 2^64`, so the object-range gadget ALWAYS uses `b_bits = 64`. A too-small
    // `b_bits` cannot forge a sign (the values genuinely fit), but it would make the
    // recomposition fail to complete with a confusing bit-decomposition error; a
    // too-large one wastes bits. Reject anything but the canonical width up front.
    if b_bits != OBJECT_RANGE_B_BITS {
        return Err(invalid(format!(
            "object-range b_bits must be {OBJECT_RANGE_B_BITS} (authenticated dict values are < 2^64); got {b_bits}"
        )));
    }
    // 1) Derive + prove the authenticated (v_val, v_dt) for every obj_id row.
    let (v_val, v_dt, w) =
        honest_dict_lookup(obj_id, dict_id, dict_value, dict_datatype, real_n, n_dict);
    let lookup = prove_dict_value_lookup_with_witness_and_context(
        context, obj_id,
        com_obj_id,
        &v_val,
        &v_dt,
        &w,
        dict_id,
        dict_value,
        dict_datatype,
        com_dict_id,
        com_dict_value,
        com_dict_datatype,
        real_n,
        n_dict,
        transcript,
    )?;

    // 2) Datatype mask over the AUTHENTICATED v_dt column (reuse the filter indicator).
    let (datatype, com_dt_match, com_dt_w, _dt_eval) =
        filter::prove_filter_indicator_with_context(
            context, &v_dt, datatype_t, real_n, &lookup.com_v_dt, transcript,
        )?;

    // 3) Range over the AUTHENTICATED v_val column (reuse CP-B1).
    let (range, com_range_sel) =
        prove_range_select_with_context(
            context, &v_val, &lookup.com_v_val, op, c, b_bits, real_n, transcript,
        )?;

    // 4) Combine sel = dt_match · range_sel. Recompute the two boolean columns via the
    // very same honest witnesses the sub-gadgets committed, so they match the openings.
    let (dt_match, _dt_w_col) = filter::honest_witness(&v_dt, datatype_t, real_n);
    let (_bits, range_sel) = honest_witness(&v_val, real_n, op, c, b_bits);
    let (combine, com_sel) =
        prove_object_range_combine(
            context, &dt_match, &range_sel, &com_dt_match, &com_range_sel, transcript,
        )?;

    Ok((
        ObjectRangeProof {
            lookup,
            datatype,
            com_dt_match,
            com_dt_w,
            range,
            com_range_sel,
            combine,
        },
        com_sel,
    ))
}

/// Verify an authenticated object-range selection proof against the caller's
/// authoritative `com_sel`. Re-runs, on one shared transcript in prover order: the
/// authenticated lookup, the datatype mask (constant `T`), the range (`op`, `c`), and
/// the combine. Concludes `com_sel` commits EXACTLY the object-range selection.
#[allow(clippy::too_many_arguments)]
pub fn verify_object_range_select(
    com_obj_id: &ArkGT,
    com_dict_id: &ArkGT,
    com_dict_value: &ArkGT,
    com_dict_datatype: &ArkGT,
    n_dict: usize,
    datatype_t: ArkFr,
    op: RangeOp,
    c: ArkFr,
    b_bits: usize,
    real_n: usize,
    com_sel: &ArkGT,
    proof: &ObjectRangeProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    // Pin the bit-width to the canonical object-range width (see the prover): the
    // authenticated dict values are `< 2^64`, so `b_bits` MUST be 64. Reject any
    // other value with a clear error rather than letting the range sub-proof fail
    // confusingly downstream.
    if b_bits != OBJECT_RANGE_B_BITS {
        return Err(invalid(format!(
            "object-range b_bits must be {OBJECT_RANGE_B_BITS} (authenticated dict values are < 2^64); got {b_bits}"
        )));
    }
    verify_dict_value_lookup(
        com_obj_id,
        com_dict_id,
        com_dict_value,
        com_dict_datatype,
        real_n,
        n_dict,
        &proof.lookup,
        transcript,
    )?;
    filter::verify_filter_indicator(
        &proof.lookup.com_v_dt,
        datatype_t,
        real_n,
        &proof.com_dt_match,
        &proof.com_dt_w,
        &proof.datatype,
        transcript,
    )?;
    verify_range_select(
        &proof.lookup.com_v_val,
        op,
        c,
        b_bits,
        real_n,
        &proof.com_range_sel,
        &proof.range,
        transcript,
    )?;
    verify_object_range_combine(
        &proof.com_dt_match,
        &proof.com_range_sel,
        com_sel,
        &proof.combine,
        transcript,
    )
}

/// The honest combined object-range selection column
/// `sel[i] = χ_n[i] · [datatype(obj_id[i]) == T] · [value(obj_id[i]) op c]` that
/// [`prove_object_range_select`] commits as its returned `Com(sel)`. Exposed so a
/// composing layer (the BGP) can obtain the SAME column that `Com(sel)` commits — to
/// relocate it across domains and multiply it into a combined selection — WITHOUT
/// re-deriving the private lookup/mask/range witnesses. MUST mirror the combine that
/// [`prove_object_range_select`] runs internally (steps 1, 2, 3, 4 there): this is the
/// elementwise product of the datatype-mask indicator over the authenticated `v_dt`
/// and the range indicator over the authenticated `v_val`.
#[allow(clippy::too_many_arguments)]
pub fn object_range_sel_column(
    obj_id: &[ArkFr],
    dict_id: &[ArkFr],
    dict_value: &[ArkFr],
    dict_datatype: &[ArkFr],
    n_dict: usize,
    datatype_t: ArkFr,
    op: RangeOp,
    c: ArkFr,
    b_bits: usize,
    real_n: usize,
) -> Vec<ArkFr> {
    let (v_val, v_dt, _w) =
        honest_dict_lookup(obj_id, dict_id, dict_value, dict_datatype, real_n, n_dict);
    let (dt_match, _dt_w) = filter::honest_witness(&v_dt, datatype_t, real_n);
    let (_bits, range_sel) = honest_witness(&v_val, real_n, op, c, b_bits);
    dt_match
        .iter()
        .zip(range_sel.iter())
        .map(|(a, b)| *a * *b)
        .collect()
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

// ===========================================================================
// CP-B2 tests: authenticated lookup + authenticated object-range selection.
// ===========================================================================
#[cfg(test)]
mod object_range_tests {
    use super::*;
    use crate::layer::ValueTriple;
    use crate::proof::{CanonicalDatatype, CanonicalObject};
    use crate::store::sync::open_sync_memory_store;
    use std::convert::TryInto;
    use tdb_succinct::TdbDataType;

    const UINT32_TAG: u64 = CanonicalDatatype::UInt32 as u64;
    const BOOLEAN_TAG: u64 = CanonicalDatatype::Boolean as u64;
    const NODE_TAG: u64 = 255;

    fn transcript() -> Transcript254 {
        Transcript254::new(b"parture-object-range-test")
    }

    fn commit(column: &[ArkFr]) -> ArkGT {
        commit_column(column).unwrap()
    }

    fn cmp_op(op: RangeOp, a: u64, b: u64) -> bool {
        match op {
            RangeOp::Lt => a < b,
            RangeOp::Le => a <= b,
            RangeOp::Gt => a > b,
            RangeOp::Ge => a >= b,
        }
    }

    /// A committed node/value dictionary drawn from a REAL base layer carrying a set
    /// of UInt32 values plus a Boolean, a Node object, and a String value (mixed
    /// datatypes), so the range gadget's OP encodings are the genuine store ones.
    struct DictFixture {
        dict_id: Vec<ArkFr>,
        dict_value: Vec<ArkFr>,
        dict_datatype: Vec<ArkFr>,
        com_id: ArkGT,
        com_value: ArkGT,
        com_datatype: ArkGT,
        n_dict: usize,
        domain: usize,
        /// (object_id, u32 value) for each UInt32 catalog entry.
        uint32: Vec<(u64, u64)>,
        /// A Boolean object's id (its value is 1 — we insert `true`).
        bool_id: u64,
        /// A Node object's id (its OP value is the 0 sentinel).
        node_id: u64,
    }

    fn build_dict_fixture() -> DictFixture {
        let iri = |s: &str| format!("http://ex/{s}");
        let p = iri("p");
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        // Five UInt32 values (inserted OUT of numeric order), a Boolean, a Node
        // object, and a String value.
        for (s, v) in [("s1", 300u32), ("s2", 5), ("s3", 7), ("s4", 40), ("s5", 100)] {
            builder
                .add_value_triple(ValueTriple::new_value(&iri(s), &p, u32::make_entry(&v)))
                .unwrap();
        }
        builder
            .add_value_triple(ValueTriple::new_value(&iri("s6"), &p, bool::make_entry(&true)))
            .unwrap();
        builder
            .add_value_triple(ValueTriple::new_node(&iri("s7"), &p, &iri("target")))
            .unwrap();
        builder
            .add_value_triple(ValueTriple::new_string_value(&iri("s8"), &p, "hello"))
            .unwrap();
        let layer = builder.commit().unwrap();
        let layer = layer.proof_commitment().unwrap();

        let dict_id = layer.node_value_dictionary_columns().unwrap().1;
        let dict_value = layer.node_value_dictionary_value_column().unwrap();
        let dict_datatype = layer.node_value_dictionary_datatype_column().unwrap();
        let n_dict = layer.node_value_catalog.len();
        let domain = dict_id.len();

        let mut uint32 = Vec::new();
        let mut bool_id = None;
        let mut node_id = None;
        for record in &layer.node_value_catalog {
            match &record.triple.object {
                CanonicalObject::Value {
                    datatype: CanonicalDatatype::UInt32,
                    lexical,
                } => {
                    let value = u32::from_be_bytes(lexical.as_slice().try_into().unwrap());
                    uint32.push((record.object_id, value as u64));
                }
                CanonicalObject::Value {
                    datatype: CanonicalDatatype::Boolean,
                    ..
                } => bool_id = Some(record.object_id),
                CanonicalObject::Node(_) => {
                    // Pick any node (subject or the node object); its OP value is 0.
                    node_id.get_or_insert(record.object_id);
                }
                _ => {}
            }
        }
        // Sanity: the genuine encoding matches what we asserted (order-preserving u32).
        assert_eq!(uint32.len(), 5, "five UInt32 dictionary entries");

        DictFixture {
            com_id: layer.node_value_dict_commitments.1,
            com_value: layer.node_value_dict_range_commitments.0,
            com_datatype: layer.node_value_dict_range_commitments.1,
            dict_id,
            dict_value,
            dict_datatype,
            n_dict,
            domain,
            uint32,
            bool_id: bool_id.unwrap(),
            node_id: node_id.unwrap(),
        }
    }

    /// A query object-id column of length `domain` from `(id, value, dt)` rows, plus
    /// the reference expected boolean selection for `(datatype_t, op, c)` over the
    /// first `real_n` rows.
    fn query_columns(
        fx: &DictFixture,
        rows: &[(u64, u64, u64)],
    ) -> (Vec<ArkFr>, usize) {
        let real_n = rows.len();
        assert!(real_n <= fx.domain);
        let obj_id: Vec<ArkFr> = (0..fx.domain)
            .map(|i| ArkFr::from_u64(if i < real_n { rows[i].0 } else { 0 }))
            .collect();
        (obj_id, real_n)
    }

    fn expected_sel(
        fx: &DictFixture,
        rows: &[(u64, u64, u64)],
        real_n: usize,
        datatype_t: u64,
        op: RangeOp,
        c: u64,
    ) -> Vec<ArkFr> {
        (0..fx.domain)
            .map(|i| {
                if i < real_n {
                    let (_, value, dt) = rows[i];
                    if dt == datatype_t && cmp_op(op, value, c) {
                        ArkFr::one()
                    } else {
                        ArkFr::zero()
                    }
                } else {
                    ArkFr::zero()
                }
            })
            .collect()
    }

    // A canonical mixed-datatype query: the 5 UInt32 rows, a Boolean row, a Node row.
    fn mixed_rows(fx: &DictFixture) -> Vec<(u64, u64, u64)> {
        let mut rows: Vec<(u64, u64, u64)> = fx
            .uint32
            .iter()
            .map(|(id, v)| (*id, *v, UINT32_TAG))
            .collect();
        rows.push((fx.bool_id, 1, BOOLEAN_TAG)); // Boolean true -> value 1
        rows.push((fx.node_id, 0, NODE_TAG)); // Node -> OP value 0 sentinel
        rows
    }

    fn prove_verify(
        fx: &DictFixture,
        obj_id: &[ArkFr],
        datatype_t: u64,
        op: RangeOp,
        c: u64,
        real_n: usize,
    ) -> io::Result<ArkGT> {
        let com_obj_id = commit(obj_id);
        let t = ArkFr::from_u64(datatype_t);
        let c_fr = ArkFr::from_u64(c);
        let mut pt = transcript();
        let (proof, com_sel) = prove_object_range_select(
            obj_id,
            &com_obj_id,
            &fx.dict_id,
            &fx.dict_value,
            &fx.dict_datatype,
            &fx.com_id,
            &fx.com_value,
            &fx.com_datatype,
            fx.n_dict,
            t,
            op,
            c_fr,
            OBJECT_RANGE_B_BITS,
            real_n,
            &mut pt,
        )?;
        let mut vt = transcript();
        verify_object_range_select(
            &com_obj_id,
            &fx.com_id,
            &fx.com_value,
            &fx.com_datatype,
            fx.n_dict,
            t,
            op,
            c_fr,
            OBJECT_RANGE_B_BITS,
            real_n,
            &com_sel,
            &proof,
            &mut vt,
        )?;
        Ok(com_sel)
    }

    // ---- Standalone authenticated lookup (checkpoint 1) ----

    #[test]
    fn lookup_authenticates_genuine_values() {
        let fx = build_dict_fixture();
        let rows = mixed_rows(&fx);
        let (obj_id, real_n) = query_columns(&fx, &rows);
        let com_obj_id = commit(&obj_id);
        let mut pt = transcript();
        let proof = prove_dict_value_lookup(
            &obj_id,
            &com_obj_id,
            &fx.dict_id,
            &fx.dict_value,
            &fx.dict_datatype,
            &fx.com_id,
            &fx.com_value,
            &fx.com_datatype,
            real_n,
            fx.n_dict,
            &mut pt,
        )
        .unwrap();
        let mut vt = transcript();
        verify_dict_value_lookup(
            &com_obj_id,
            &fx.com_id,
            &fx.com_value,
            &fx.com_datatype,
            real_n,
            fx.n_dict,
            &proof,
            &mut vt,
        )
        .unwrap();

        // The authenticated v_val/v_dt commitments equal an independent commit of the
        // TRUE dict entries for each obj_id row (the lookup's whole point).
        let (v_val, v_dt, _w) =
            honest_dict_lookup(&obj_id, &fx.dict_id, &fx.dict_value, &fx.dict_datatype, real_n, fx.n_dict);
        assert_eq!(proof.com_v_val, commit(&v_val));
        assert_eq!(proof.com_v_dt, commit(&v_dt));
    }

    #[test]
    fn lookup_rejects_wrong_v_val() {
        // A prover that authenticates a WRONG value for a real obj_id (points the row
        // at a DIFFERENT dict value than obj_id's true entry) is rejected: its query
        // pole matches no dict pole.
        let fx = build_dict_fixture();
        let rows = mixed_rows(&fx);
        let (obj_id, real_n) = query_columns(&fx, &rows);
        let com_obj_id = commit(&obj_id);
        let (mut v_val, v_dt, w) =
            honest_dict_lookup(&obj_id, &fx.dict_id, &fx.dict_value, &fx.dict_datatype, real_n, fx.n_dict);
        // Row 0's true value is uint32[0]; overwrite with a DIFFERENT genuine value.
        let wrong = ArkFr::from_u64(fx.uint32[1].1);
        assert_ne!(v_val[0], wrong);
        v_val[0] = wrong;
        let mut pt = transcript();
        let proof = prove_dict_value_lookup_with_witness(
            &obj_id, &com_obj_id, &v_val, &v_dt, &w, &fx.dict_id, &fx.dict_value,
            &fx.dict_datatype, &fx.com_id, &fx.com_value, &fx.com_datatype, real_n, fx.n_dict,
            &mut pt,
        )
        .unwrap();
        let mut vt = transcript();
        assert!(verify_dict_value_lookup(
            &com_obj_id, &fx.com_id, &fx.com_value, &fx.com_datatype, real_n, fx.n_dict, &proof,
            &mut vt,
        )
        .is_err());
    }

    #[test]
    fn lookup_rejects_wrong_v_dt() {
        // Same, but forging the DATATYPE of a real obj_id -> unmatched pole -> reject.
        let fx = build_dict_fixture();
        let rows = mixed_rows(&fx);
        let (obj_id, real_n) = query_columns(&fx, &rows);
        let com_obj_id = commit(&obj_id);
        let (v_val, mut v_dt, w) =
            honest_dict_lookup(&obj_id, &fx.dict_id, &fx.dict_value, &fx.dict_datatype, real_n, fx.n_dict);
        // Claim a UInt32 row is actually Boolean (a cross-datatype forgery attempt).
        assert_eq!(v_dt[0], ArkFr::from_u64(UINT32_TAG));
        v_dt[0] = ArkFr::from_u64(BOOLEAN_TAG);
        let mut pt = transcript();
        let proof = prove_dict_value_lookup_with_witness(
            &obj_id, &com_obj_id, &v_val, &v_dt, &w, &fx.dict_id, &fx.dict_value,
            &fx.dict_datatype, &fx.com_id, &fx.com_value, &fx.com_datatype, real_n, fx.n_dict,
            &mut pt,
        )
        .unwrap();
        let mut vt = transcript();
        assert!(verify_dict_value_lookup(
            &com_obj_id, &fx.com_id, &fx.com_value, &fx.com_datatype, real_n, fx.n_dict, &proof,
            &mut vt,
        )
        .is_err());
    }

    // ---- Full authenticated object-range selection (checkpoint 2) ----

    #[test]
    fn all_ops_select_only_matching_datatype_and_value() {
        let fx = build_dict_fixture();
        let rows = mixed_rows(&fx);
        let (obj_id, real_n) = query_columns(&fx, &rows);
        let c = 40u64; // straddles the UInt32 values 5,7,40,100,300 (boundary at 40).
        for op in [RangeOp::Lt, RangeOp::Le, RangeOp::Gt, RangeOp::Ge] {
            let com_sel = prove_verify(&fx, &obj_id, UINT32_TAG, op, c, real_n).unwrap();
            let expected = expected_sel(&fx, &rows, real_n, UINT32_TAG, op, c);
            assert_eq!(com_sel, commit(&expected), "op {:?}", op);
        }
    }

    #[test]
    fn datatype_mask_excludes_other_datatypes() {
        // c = 0, op = Ge: EVERY row's value satisfies `>= 0` (uint32s, Boolean=1,
        // Node=0). Only the UInt32 rows must be selected; the Boolean and Node rows
        // are dropped purely by the datatype mask.
        let fx = build_dict_fixture();
        let rows = mixed_rows(&fx);
        let (obj_id, real_n) = query_columns(&fx, &rows);
        let com_sel = prove_verify(&fx, &obj_id, UINT32_TAG, RangeOp::Ge, 0, real_n).unwrap();
        let expected = expected_sel(&fx, &rows, real_n, UINT32_TAG, RangeOp::Ge, 0);
        // Exactly the five UInt32 rows are selected (all >= 0); Boolean+Node are 0.
        let ones = expected.iter().filter(|s| **s == ArkFr::one()).count();
        assert_eq!(ones, 5);
        for (i, row) in rows.iter().enumerate() {
            if row.2 != UINT32_TAG {
                assert_eq!(expected[i], ArkFr::zero());
            }
        }
        assert_eq!(com_sel, commit(&expected));

        // And a Boolean-datatype query with the SAME c/op selects ONLY the Boolean row.
        let com_sel_bool = prove_verify(&fx, &obj_id, BOOLEAN_TAG, RangeOp::Ge, 0, real_n).unwrap();
        let expected_bool = expected_sel(&fx, &rows, real_n, BOOLEAN_TAG, RangeOp::Ge, 0);
        assert_eq!(expected_bool.iter().filter(|s| **s == ArkFr::one()).count(), 1);
        assert_eq!(com_sel_bool, commit(&expected_bool));
    }

    #[test]
    fn padding_rows_get_sel_zero() {
        // Real rows are the 5 UInt32; the tail is padding. Even though padding obj_id
        // = 0 authenticates value/datatype 0, the χ_n mask forces sel = 0 there.
        let fx = build_dict_fixture();
        let rows: Vec<(u64, u64, u64)> =
            fx.uint32.iter().map(|(id, v)| (*id, *v, UINT32_TAG)).collect();
        let (obj_id, real_n) = query_columns(&fx, &rows);
        let com_sel = prove_verify(&fx, &obj_id, UINT32_TAG, RangeOp::Ge, 0, real_n).unwrap();
        let expected = expected_sel(&fx, &rows, real_n, UINT32_TAG, RangeOp::Ge, 0);
        for i in real_n..fx.domain {
            assert_eq!(expected[i], ArkFr::zero());
        }
        assert_eq!(com_sel, commit(&expected));
    }

    #[test]
    fn wrong_datatype_t_at_verify_is_rejected() {
        let fx = build_dict_fixture();
        let rows = mixed_rows(&fx);
        let (obj_id, real_n) = query_columns(&fx, &rows);
        let com_obj_id = commit(&obj_id);
        let mut pt = transcript();
        let (proof, com_sel) = prove_object_range_select(
            &obj_id, &com_obj_id, &fx.dict_id, &fx.dict_value, &fx.dict_datatype,
            &fx.com_id, &fx.com_value, &fx.com_datatype, fx.n_dict,
            ArkFr::from_u64(UINT32_TAG), RangeOp::Ge, ArkFr::from_u64(40), OBJECT_RANGE_B_BITS,
            real_n, &mut pt,
        )
        .unwrap();
        let mut vt = transcript();
        assert!(verify_object_range_select(
            &com_obj_id, &fx.com_id, &fx.com_value, &fx.com_datatype, fx.n_dict,
            ArkFr::from_u64(BOOLEAN_TAG), // wrong T
            RangeOp::Ge, ArkFr::from_u64(40), OBJECT_RANGE_B_BITS, real_n, &com_sel, &proof, &mut vt,
        )
        .is_err());
    }

    #[test]
    fn wrong_c_at_verify_is_rejected() {
        let fx = build_dict_fixture();
        let rows = mixed_rows(&fx);
        let (obj_id, real_n) = query_columns(&fx, &rows);
        let com_obj_id = commit(&obj_id);
        let mut pt = transcript();
        let (proof, com_sel) = prove_object_range_select(
            &obj_id, &com_obj_id, &fx.dict_id, &fx.dict_value, &fx.dict_datatype,
            &fx.com_id, &fx.com_value, &fx.com_datatype, fx.n_dict,
            ArkFr::from_u64(UINT32_TAG), RangeOp::Ge, ArkFr::from_u64(40), OBJECT_RANGE_B_BITS,
            real_n, &mut pt,
        )
        .unwrap();
        let mut vt = transcript();
        assert!(verify_object_range_select(
            &com_obj_id, &fx.com_id, &fx.com_value, &fx.com_datatype, fx.n_dict,
            ArkFr::from_u64(UINT32_TAG), RangeOp::Ge,
            ArkFr::from_u64(41), // wrong c
            OBJECT_RANGE_B_BITS, real_n, &com_sel, &proof, &mut vt,
        )
        .is_err());
    }

    #[test]
    fn tampered_com_sel_at_verify_is_rejected() {
        let fx = build_dict_fixture();
        let rows = mixed_rows(&fx);
        let (obj_id, real_n) = query_columns(&fx, &rows);
        let com_obj_id = commit(&obj_id);
        let mut pt = transcript();
        let (proof, _com_sel) = prove_object_range_select(
            &obj_id, &com_obj_id, &fx.dict_id, &fx.dict_value, &fx.dict_datatype,
            &fx.com_id, &fx.com_value, &fx.com_datatype, fx.n_dict,
            ArkFr::from_u64(UINT32_TAG), RangeOp::Ge, ArkFr::from_u64(40), OBJECT_RANGE_B_BITS,
            real_n, &mut pt,
        )
        .unwrap();
        let bogus = commit(&vec![ArkFr::zero(); fx.domain]); // not the proven sel
        let mut vt = transcript();
        assert!(verify_object_range_select(
            &com_obj_id, &fx.com_id, &fx.com_value, &fx.com_datatype, fx.n_dict,
            ArkFr::from_u64(UINT32_TAG), RangeOp::Ge, ArkFr::from_u64(40), OBJECT_RANGE_B_BITS,
            real_n, &bogus, &proof, &mut vt,
        )
        .is_err());
    }

    #[test]
    fn non_canonical_b_bits_is_rejected_by_object_range() {
        // The object-range entry points PIN b_bits to OBJECT_RANGE_B_BITS (64): the
        // authenticated dict values are < 2^64, so any other width is rejected up front
        // with a clear error (rather than failing confusingly inside the range gadget).
        let fx = build_dict_fixture();
        let rows = mixed_rows(&fx);
        let (obj_id, real_n) = query_columns(&fx, &rows);
        let com_obj_id = commit(&obj_id);
        // Prover rejects a too-small width.
        let mut pt = transcript();
        assert!(prove_object_range_select(
            &obj_id, &com_obj_id, &fx.dict_id, &fx.dict_value, &fx.dict_datatype,
            &fx.com_id, &fx.com_value, &fx.com_datatype, fx.n_dict,
            ArkFr::from_u64(UINT32_TAG), RangeOp::Ge, ArkFr::from_u64(40),
            32, // != OBJECT_RANGE_B_BITS
            real_n, &mut pt,
        )
        .is_err());
        // An HONEST proof at 64 bits, but the verifier is asked to check at a different
        // width -> rejected before any sub-proof runs.
        let mut pt2 = transcript();
        let (proof, com_sel) = prove_object_range_select(
            &obj_id, &com_obj_id, &fx.dict_id, &fx.dict_value, &fx.dict_datatype,
            &fx.com_id, &fx.com_value, &fx.com_datatype, fx.n_dict,
            ArkFr::from_u64(UINT32_TAG), RangeOp::Ge, ArkFr::from_u64(40), OBJECT_RANGE_B_BITS,
            real_n, &mut pt2,
        )
        .unwrap();
        let mut vt = transcript();
        assert!(verify_object_range_select(
            &com_obj_id, &fx.com_id, &fx.com_value, &fx.com_datatype, fx.n_dict,
            ArkFr::from_u64(UINT32_TAG), RangeOp::Ge, ArkFr::from_u64(40),
            128, // != OBJECT_RANGE_B_BITS
            real_n, &com_sel, &proof, &mut vt,
        )
        .is_err());
    }

    #[test]
    fn repeated_obj_ids_are_authenticated_with_multiplicity() {
        // The same obj_id appears in several query rows (multiplicity > 1 on its dict
        // row); the lookup's w column must count them and the selection is per-row.
        let fx = build_dict_fixture();
        let (id0, v0) = fx.uint32[0];
        let (id1, v1) = fx.uint32[1];
        let rows = vec![
            (id0, v0, UINT32_TAG),
            (id0, v0, UINT32_TAG),
            (id1, v1, UINT32_TAG),
            (id0, v0, UINT32_TAG),
            (fx.bool_id, 1, BOOLEAN_TAG),
        ];
        let (obj_id, real_n) = query_columns(&fx, &rows);
        let c = v0.max(v1);
        for op in [RangeOp::Lt, RangeOp::Le, RangeOp::Ge] {
            let com_sel = prove_verify(&fx, &obj_id, UINT32_TAG, op, c, real_n).unwrap();
            let expected = expected_sel(&fx, &rows, real_n, UINT32_TAG, op, c);
            assert_eq!(com_sel, commit(&expected), "op {:?}", op);
        }
    }
}
