//! Full-field limb-lexicographic STRICT-distinctness gadget (I4-a).
//!
//! WHAT IT PROVES. Given a committed column `S` (`n` real rows, the rest padding)
//! of canonical hash values (each `< p`, the BN254 scalar modulus), prove that the
//! ALREADY-SORTED candidate `S` is STRICTLY LEXICOGRAPHICALLY INCREASING when each
//! entry is viewed as two 127-bit limbs `(hi, lo)` with `S[i] = hi[i]·2^127 + lo[i]`.
//! Because lexicographic order on `(hi, lo)` (high limb first) is exactly the integer
//! order of the canonical representative, a strictly-lex-increasing column has DISTINCT
//! entries. Paired with a permutation argument (I4-b, NOT this gadget) that `S` is a
//! rearrangement of the real per-layer hash columns, this proves the hashes are
//! DISTINCT — the succinct replacement for the O(dictionary) re-mint recompute.
//!
//! WHY LIMBS. The audited strict-sort ([`super::sort`]) decomposes an adjacent
//! difference into `<= 253` bits (`sort.rs:514`), but a full-field hash sits near the
//! `~253.6`-bit modulus, so its differences do NOT fit. We work at the LIMB level
//! (`<= 127` bits) instead, and reconstruct strict lex order from per-limb comparisons.
//!
//! THE SIX STEPS (all over `[0, n)` unless noted; every piece reuses an audited gadget
//! or the audited sumcheck engine — NO new sumcheck engine, NO in-circuit hash):
//!   1. LIMB DECOMPOSITION + RANGE. Commit `hi`, `lo`; prove each `< 2^127` by a
//!      127-bit booleanity (`super::sort::booleanity_claim`) + recomposition claim.
//!   2. LINEAR TIE. `Com(S) == 2^127·Com(hi) + Com(lo)` — one homomorphic GT equality,
//!      binding the limbs to the committed column (no sumcheck).
//!   3. `hi` NONDECREASING over `[0, n-1)`, via `super::sort::prove_nondecreasing_bits`
//!      at 127 bits. This ALSO commits+binds `hi_shift[i] = hi[i+1]` (its `u_shift`).
//!   4. EQUALITY FLAG `e[i] = [hi[i] == hi[i+1]]` over `[0, n-1)`, via the IsZero
//!      argument (pseudo-inverse `w_e`), COMPLETE both directions.
//!   5. MASKED STRICT-`lo`. `w5[i] := e[i]·(lo_shift[i] - lo[i] - 1) ∈ [0, 2^127)` over
//!      `[0, n-1)`. `lo_shift[i] = lo[i+1]` is bound by a successor lookup
//!      (`super::sort::prove_successor_lookup_with_context`). When `hi` ties (`e=1`)
//!      this forces `lo` to strictly increase; when `hi` differs (`e=0`) it is `0`.
//!   6. ⚠️ CANONICAL `<p` CHECK (LOAD-BEARING; Risk #1). `2×127 = 254 > log2(p)`, so a
//!      value in the lower ~⅓ of the field admits a SECOND in-range split `(hi',lo')`
//!      with `hi'·2^127 + lo' = S + p`. A malicious prover could split two EQUAL hashes
//!      differently so the lex check treats them as distinct — a duplicate slips
//!      through. CLOSED by proving each recomposed row is `< p`, i.e. `(hi[i], lo[i])
//!      <_lex (p_hi, p_lo)`: with `g[i] = [hi[i] == p_hi]` (IsZero) require
//!      `t1[i] := (1-g[i])·(p_hi - hi[i] - 1) ∈ [0, 2^127)` (forces `hi < p_hi` when
//!      `hi != p_hi`) AND `t2[i] := g[i]·(p_lo - lo[i] - 1) ∈ [0, 2^127)` (forces
//!      `lo < p_lo` when `hi == p_hi`). This forces the UNIQUE canonical split, so the
//!      limb representation is injective on field elements.
//!
//! SOUNDNESS. Steps 3+4+5 give a total strict order: step 3 forbids a `hi` decrease;
//! step 4's complete flag + step 5 force a strict `lo` exactly when `hi` ties; a
//! decreasing OR duplicate pair fails step 3 or step 5. Step 6 forces the unique
//! canonical `<p` split, so two equal hashes CANNOT be split to look distinct (without
//! step 6 the gadget is silently forgeable — see the `wrap_split_*` negative test). The
//! range/IsZero claims are batched into ONE sumcheck with all challenges (`tau`, the
//! zero-check point, `rho`) drawn AFTER every column commitment.
//!
//! STANDALONE. This is the I4-a gadget only: it takes the ALREADY-SORTED committed
//! column and proves strict lex order. The permutation (that `S` is the real hashes
//! rearranged) is I4-b; combined-column formation is I4-c; re-mint wiring is I4-d/e.
//!
//! For a developer: this is a lexicographic (multi-limb) distinctness [gadget](crate::proof::primer)
//! — the full-field cousin of [`sort`](super::sort). It batches range/IsZero checks into ONE
//! [sumcheck](crate::proof::primer) (§5) and reuses two audited [`sort`](super::sort) sub-proofs (a
//! nondecreasing-bits proof and a successor [LogUp](crate::proof::primer) lookup, §7). All
//! challenges (`tau`, the zero-check point, `rho`) are Fiat–Shamir, drawn after every column is
//! committed ([the transcript invariant](crate::proof::primer), §8). See also
//! [gadgets](crate::proof::primer) (§6).

use std::io;

use ark_ff::{BigInteger, PrimeField};
use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::{Field, Group};
use dory_pcs::primitives::transcript::Transcript;

use super::argument::{
    prove_sumcheck, verify_column_openings, verify_sumcheck, ClaimShape, ColumnOpening, Expr,
    Operand, StoreProverContext, SumcheckClaim, SumcheckProof, SumcheckShape, Term, Transcript254,
};
use super::invalid;
use super::sort::{
    booleanity_claim, low_bits, prove_nondecreasing_bits_with_context,
    prove_successor_lookup_with_context, verify_nondecreasing_bits, verify_successor_lookup,
    SortedProof, SuccessorProof,
};

/// Limb bit width. `2 × B = 254` covers the full `~253.6`-bit field; each limb is a
/// 127-bit range (well under the `<= 253` bit cap the reused sort/range gadgets impose).
const B: usize = 127;

// --- main-sumcheck column layout (all length `n_domain`) --------------------
const COL_HI: usize = 0;
const COL_LO: usize = 1;
const COL_HI_SHIFT: usize = 2;
const COL_LO_SHIFT: usize = 3;
const COL_E: usize = 4;
const COL_W_E: usize = 5;
const COL_G: usize = 6;
const COL_W_G: usize = 7;
const COL_W5: usize = 8;
const COL_T1: usize = 9;
const COL_T2: usize = 10;
const NUM_SCALAR_COLS: usize = 11;
// Contiguous bit block (five 127-bit groups), so ONE booleanity claim covers all.
const HI_BIT_BASE: usize = NUM_SCALAR_COLS;
const LO_BIT_BASE: usize = HI_BIT_BASE + B;
const W5_BIT_BASE: usize = LO_BIT_BASE + B;
const T1_BIT_BASE: usize = W5_BIT_BASE + B;
const T2_BIT_BASE: usize = T1_BIT_BASE + B;
const NUM_BIT_COLS: usize = 5 * B;
const NUM_COLS: usize = NUM_SCALAR_COLS + NUM_BIT_COLS;
const NUM_AUX: usize = 7; // e, w_e, g, w_g, w5, t1, t2 (com_hi/lo/hi_shift/lo_shift are separate)

// --- chi-bound indices (SumcheckShape::chi_bounds = [n, n-1]) ---------------
const CHI_N: usize = 0; // real-row prefix [0, n)     — per-row (range, g-flag, <p)
const CHI_N1: usize = 1; // adjacent prefix [0, n-1)  — adjacent (e-flag, strict-lo)

const LABEL_DOMAIN: &[u8] = b"pow/lexdistinct/domain";
const LABEL_META: &[u8] = b"pow/lexdistinct/meta";
const LABEL_COM_S: &[u8] = b"pow/lexdistinct/com-s";
const LABEL_MODULUS: &[u8] = b"pow/lexdistinct/modulus";
const LABEL_MAIN_COM: &[u8] = b"pow/lexdistinct/main-com";
const LABEL_TAU: &[u8] = b"pow/lexdistinct/booleanity-tau";

#[inline]
fn f0() -> ArkFr {
    ArkFr::zero()
}
#[inline]
fn f1() -> ArkFr {
    ArkFr::one()
}

/// `2^exp` in the field (by repeated doubling; `1u64 << exp` overflows past 63).
fn pow2(exp: usize) -> ArkFr {
    let two = f1() + f1();
    let mut acc = f1();
    for _ in 0..exp {
        acc = acc * two;
    }
    acc
}

/// Whether a field element's canonical integer representative is `< 2^num_bits`.
fn fits_in_bits(value: ArkFr, num_bits: usize) -> bool {
    (value.0.into_bigint().num_bits() as usize) <= num_bits
}

/// The BN254 scalar modulus `p` split into two 127-bit limbs `(p_hi, p_lo)` with
/// `p = p_hi·2^127 + p_lo`. Built from fixed u64 limbs and then GUARDED against the
/// live field: `p_hi·2^127 + p_lo ≡ 0 (mod p)` with both limbs `< 2^127` and
/// `p_hi != 0` pins `(p_hi, p_lo)` UNIQUELY to the real modulus split — if the field
/// ever changed, this fails closed. (The only `<2^127` limb pair summing to `0 mod p`
/// is `(p>>127, p mod 2^127)`, since any nonzero multiple `k·p` with `k>=2` needs
/// `>= 254` bits and `k=0` needs `p_hi = 0`.)
fn modulus_limbs() -> io::Result<(ArkFr, ArkFr)> {
    let from2 = |l0: u64, l1: u64| ArkFr::from_u64(l0) + ArkFr::from_u64(l1) * pow2(64);
    let p_hi = from2(8_115_639_828_839_837_882, 6_973_996_533_605_941_331);
    let p_lo = from2(4_891_460_686_036_598_785, 2_896_914_383_306_846_353);
    let canonical = fits_in_bits(p_hi, B)
        && fits_in_bits(p_lo, B)
        && p_hi != f0()
        && (pow2(B) * p_hi + p_lo == f0());
    if !canonical {
        return Err(invalid("lexdistinct BN254 modulus-limb guard failed"));
    }
    Ok((p_hi, p_lo))
}

/// Split a field element `< p` into `(hi, lo)` with `v = hi·2^127 + lo`,
/// `hi, lo ∈ [0, 2^127)` — the UNIQUE canonical split (`p < 2^254`).
fn split_canonical(v: ArkFr) -> (ArkFr, ArkFr) {
    let bigint = v.0.into_bigint();
    let mut lo = f0();
    for j in (0..B).rev() {
        lo = lo + lo + if bigint.get_bit(j) { f1() } else { f0() };
    }
    let mut hi = f0();
    for j in (B..2 * B).rev() {
        hi = hi + hi + if bigint.get_bit(j) { f1() } else { f0() };
    }
    (hi, lo)
}

/// The padded power-of-two domain (`>= 2`) for `len` entries with `n` real rows.
fn column_domain(len: usize, n: usize) -> io::Result<usize> {
    len.max(n)
        .checked_next_power_of_two()
        .ok_or_else(|| invalid("lexdistinct domain size overflow"))
        .map(|d| d.max(2))
}

fn padded(column: &[ArkFr], n_domain: usize) -> Vec<ArkFr> {
    let mut out = column.to_vec();
    out.resize(n_domain, f0());
    out
}

/// `col`'s `bits` low bits per real row `[0, rows)` (zero on the rest): a `bits`-wide
/// stack of length-`n_domain` bit columns, `out[j][i] = bit_j(col[i])`.
fn decompose(col: &[ArkFr], rows: usize, bits: usize) -> Vec<Vec<ArkFr>> {
    let n_domain = col.len();
    let mut out = vec![vec![f0(); n_domain]; bits];
    for i in 0..rows.min(n_domain) {
        let b = low_bits(col[i], bits);
        for (j, slot) in b.into_iter().enumerate() {
            out[j][i] = slot;
        }
    }
    out
}

/// The full explicit witness for the main sumcheck. The honest builder
/// ([`honest_witness`]) derives every field from `(hi, lo)`; the negative tests
/// mutate one field before proving to confirm the corresponding claim rejects.
#[derive(Clone, Debug)]
struct LexWitness {
    hi: Vec<ArkFr>,
    lo: Vec<ArkFr>,
    hi_shift: Vec<ArkFr>,
    lo_shift: Vec<ArkFr>,
    e: Vec<ArkFr>,
    w_e: Vec<ArkFr>,
    g: Vec<ArkFr>,
    w_g: Vec<ArkFr>,
    w5: Vec<ArkFr>,
    t1: Vec<ArkFr>,
    t2: Vec<ArkFr>,
    hi_bits: Vec<Vec<ArkFr>>,
    lo_bits: Vec<Vec<ArkFr>>,
    w5_bits: Vec<Vec<ArkFr>>,
    t1_bits: Vec<Vec<ArkFr>>,
    t2_bits: Vec<Vec<ArkFr>>,
}

/// Derive the honest witness from the committed limbs `(hi, lo)` (both length
/// `n_domain`). `hi_shift`/`lo_shift` are the one-step shifts; `e`/`g` the IsZero flags
/// with pseudo-inverses `w_e`/`w_g`; `w5`/`t1`/`t2` the masked-range values (steps 5+6);
/// and the five 127-bit decompositions. For a genuine sorted-distinct input every
/// masked value lands in `[0, 2^127)`; a duplicate / decrease / wrap-split produces an
/// out-of-range value whose 127-bit recomposition (or the tie/flag) then rejects.
fn honest_witness(
    hi: &[ArkFr],
    lo: &[ArkFr],
    n: usize,
    n_domain: usize,
    p_hi: ArkFr,
    p_lo: ArkFr,
) -> io::Result<LexWitness> {
    if hi.len() != n_domain || lo.len() != n_domain {
        return Err(invalid("lexdistinct limb column length mismatch"));
    }
    let last = n.saturating_sub(1);
    let mut hi_shift = vec![f0(); n_domain];
    let mut lo_shift = vec![f0(); n_domain];
    for i in 0..last {
        hi_shift[i] = hi[i + 1];
        lo_shift[i] = lo[i + 1];
    }

    // Adjacent (step 4/5) columns over [0, n-1).
    let mut e = vec![f0(); n_domain];
    let mut w_e = vec![f0(); n_domain];
    let mut w5 = vec![f0(); n_domain];
    for i in 0..last {
        let d = hi_shift[i] - hi[i];
        if d == f0() {
            e[i] = f1();
        } else {
            e[i] = f0();
            w_e[i] = d.inv().ok_or_else(|| invalid("lexdistinct noninvertible hi gap"))?;
        }
        w5[i] = e[i] * (lo_shift[i] - lo[i] - f1());
    }

    // Per-row (step 6) columns over [0, n).
    let mut g = vec![f0(); n_domain];
    let mut w_g = vec![f0(); n_domain];
    let mut t1 = vec![f0(); n_domain];
    let mut t2 = vec![f0(); n_domain];
    for i in 0..n.min(n_domain) {
        let dg = hi[i] - p_hi;
        if dg == f0() {
            g[i] = f1();
        } else {
            g[i] = f0();
            w_g[i] = dg.inv().ok_or_else(|| invalid("lexdistinct noninvertible p_hi gap"))?;
        }
        t1[i] = (f1() - g[i]) * (p_hi - hi[i] - f1());
        t2[i] = g[i] * (p_lo - lo[i] - f1());
    }

    let hi_bits = decompose(hi, n, B);
    let lo_bits = decompose(lo, n, B);
    let w5_bits = decompose(&w5, last, B);
    let t1_bits = decompose(&t1, n, B);
    let t2_bits = decompose(&t2, n, B);

    Ok(LexWitness {
        hi: hi.to_vec(),
        lo: lo.to_vec(),
        hi_shift,
        lo_shift,
        e,
        w_e,
        g,
        w_g,
        w5,
        t1,
        t2,
        hi_bits,
        lo_bits,
        w5_bits,
        t1_bits,
        t2_bits,
    })
}

// ---------------------------------------------------------------------------
// Sumcheck claim builders (Identity zero-checks over the argument engine).
// ---------------------------------------------------------------------------

/// `chi · (Σ_{j<B} 2^j·bit_{bit_base+j} − value_col) = 0`: the committed 127 bits
/// recompose EXACTLY to `value_col` on the masked rows (an out-of-range value cannot
/// be hit by 127 boolean bits, so this doubles as the `< 2^127` range check).
fn recompose_claim(chi: usize, bit_base: usize, value_col: usize) -> SumcheckClaim {
    let mut terms = Vec::with_capacity(B + 1);
    for j in 0..B {
        terms.push(Term {
            coeff: pow2(j),
            operands: vec![Operand::Chi(chi), Operand::Column(bit_base + j)],
        });
    }
    terms.push(Term {
        coeff: f0() - f1(),
        operands: vec![Operand::Chi(chi), Operand::Column(value_col)],
    });
    SumcheckClaim { shape: ClaimShape::Identity, expr: Expr { terms }, target: f0() }
}

/// Step 5 definition: `w5 − e·(lo_shift − lo − 1) = 0` over `[0, n-1)`.
fn w5_definition_claim() -> SumcheckClaim {
    // w5 − e·lo_shift + e·lo + e = 0
    let terms = vec![
        Term { coeff: f1(), operands: vec![Operand::Chi(CHI_N1), Operand::Column(COL_W5)] },
        Term {
            coeff: f0() - f1(),
            operands: vec![Operand::Chi(CHI_N1), Operand::Column(COL_E), Operand::Column(COL_LO_SHIFT)],
        },
        Term {
            coeff: f1(),
            operands: vec![Operand::Chi(CHI_N1), Operand::Column(COL_E), Operand::Column(COL_LO)],
        },
        Term { coeff: f1(), operands: vec![Operand::Chi(CHI_N1), Operand::Column(COL_E)] },
    ];
    SumcheckClaim { shape: ClaimShape::Identity, expr: Expr { terms }, target: f0() }
}

/// Step 6a definition: `t1 − (1−g)·(p_hi − hi − 1) = 0` over `[0, n)`.
fn t1_definition_claim(p_hi: ArkFr) -> SumcheckClaim {
    // t1 + hi + (1 − p_hi) + g·(p_hi − 1) − g·hi = 0
    let terms = vec![
        Term { coeff: f1(), operands: vec![Operand::Chi(CHI_N), Operand::Column(COL_T1)] },
        Term { coeff: f1() - p_hi, operands: vec![Operand::Chi(CHI_N)] },
        Term { coeff: f1(), operands: vec![Operand::Chi(CHI_N), Operand::Column(COL_HI)] },
        Term { coeff: p_hi - f1(), operands: vec![Operand::Chi(CHI_N), Operand::Column(COL_G)] },
        Term {
            coeff: f0() - f1(),
            operands: vec![Operand::Chi(CHI_N), Operand::Column(COL_G), Operand::Column(COL_HI)],
        },
    ];
    SumcheckClaim { shape: ClaimShape::Identity, expr: Expr { terms }, target: f0() }
}

/// Step 6b definition: `t2 − g·(p_lo − lo − 1) = 0` over `[0, n)`.
fn t2_definition_claim(p_lo: ArkFr) -> SumcheckClaim {
    // t2 + g·(1 − p_lo) + g·lo = 0
    let terms = vec![
        Term { coeff: f1(), operands: vec![Operand::Chi(CHI_N), Operand::Column(COL_T2)] },
        Term { coeff: f1() - p_lo, operands: vec![Operand::Chi(CHI_N), Operand::Column(COL_G)] },
        Term {
            coeff: f1(),
            operands: vec![Operand::Chi(CHI_N), Operand::Column(COL_G), Operand::Column(COL_LO)],
        },
    ];
    SumcheckClaim { shape: ClaimShape::Identity, expr: Expr { terms }, target: f0() }
}

/// IsZero over a difference `pred = a − b` (masked by `chi`), pinning `flag = [pred==0]`
/// COMPLETELY: `flag·pred = 0` (flag=1 ⟹ pred=0) AND `(1−flag) − inv·pred = 0`
/// (pred!=0 ⟹ flag=0 with `inv = pred⁻¹`; pred==0 ⟹ flag=1). `a`/`b` are committed
/// columns; either may be a public constant folded into a `Chi`-only term (`b_const`).
fn iszero_claims(
    chi: usize,
    flag: usize,
    inv: usize,
    a: usize,
    b_col: Option<usize>,
    b_const: ArkFr,
) -> [SumcheckClaim; 2] {
    // pred·flag = 0  →  flag·a − flag·b = 0
    let mut prod = vec![Term {
        coeff: f1(),
        operands: vec![Operand::Chi(chi), Operand::Column(flag), Operand::Column(a)],
    }];
    match b_col {
        Some(b) => prod.push(Term {
            coeff: f0() - f1(),
            operands: vec![Operand::Chi(chi), Operand::Column(flag), Operand::Column(b)],
        }),
        None => prod.push(Term {
            coeff: f0() - b_const,
            operands: vec![Operand::Chi(chi), Operand::Column(flag)],
        }),
    }
    // (1 − flag) − inv·pred = 0  →  1 − flag − inv·a + inv·b = 0
    let mut comp = vec![
        Term { coeff: f1(), operands: vec![Operand::Chi(chi)] },
        Term { coeff: f0() - f1(), operands: vec![Operand::Chi(chi), Operand::Column(flag)] },
        Term {
            coeff: f0() - f1(),
            operands: vec![Operand::Chi(chi), Operand::Column(inv), Operand::Column(a)],
        },
    ];
    match b_col {
        Some(b) => comp.push(Term {
            coeff: f1(),
            operands: vec![Operand::Chi(chi), Operand::Column(inv), Operand::Column(b)],
        }),
        None => comp.push(Term {
            coeff: b_const,
            operands: vec![Operand::Chi(chi), Operand::Column(inv)],
        }),
    }
    [
        SumcheckClaim { shape: ClaimShape::Identity, expr: Expr { terms: prod }, target: f0() },
        SumcheckClaim { shape: ClaimShape::Identity, expr: Expr { terms: comp }, target: f0() },
    ]
}

/// The public shape of the main batched sumcheck (identical on both sides).
fn lex_shape(k: usize, n: usize, tau: ArkFr, p_hi: ArkFr, p_lo: ArkFr) -> SumcheckShape {
    let mut claims = Vec::with_capacity(13);
    // Step 1 booleanity — ONE tau-fold over the whole contiguous bit block.
    claims.push(booleanity_claim(CHI_N, HI_BIT_BASE, NUM_BIT_COLS, tau));
    // Step 1 range recompositions (hi, lo, t1, t2 over [0,n); w5 over [0,n-1)).
    claims.push(recompose_claim(CHI_N, HI_BIT_BASE, COL_HI));
    claims.push(recompose_claim(CHI_N, LO_BIT_BASE, COL_LO));
    claims.push(recompose_claim(CHI_N1, W5_BIT_BASE, COL_W5));
    claims.push(recompose_claim(CHI_N, T1_BIT_BASE, COL_T1));
    claims.push(recompose_claim(CHI_N, T2_BIT_BASE, COL_T2));
    // Steps 5+6 masked-value definitions.
    claims.push(w5_definition_claim());
    claims.push(t1_definition_claim(p_hi));
    claims.push(t2_definition_claim(p_lo));
    // Step 4 e-flag = [hi_shift == hi] (both directions).
    let e = iszero_claims(CHI_N1, COL_E, COL_W_E, COL_HI_SHIFT, Some(COL_HI), f0());
    claims.extend(e);
    // Step 6 g-flag = [hi == p_hi] (both directions).
    let g = iszero_claims(CHI_N, COL_G, COL_W_G, COL_HI, None, p_hi);
    claims.extend(g);
    SumcheckShape {
        num_vars: k,
        num_columns: NUM_COLS,
        chi_bounds: vec![n, n.saturating_sub(1)],
        claims,
    }
}

fn bind_statement(transcript: &mut Transcript254, n: usize, com_s: &ArkGT, p_hi: ArkFr, p_lo: ArkFr) {
    transcript.append_bytes(LABEL_DOMAIN, b"parture/proof-of-woql/lexdistinct/v0");
    transcript.append_bytes(LABEL_META, &(n as u64).to_le_bytes());
    transcript.append_bytes(LABEL_META, &(B as u64).to_le_bytes());
    transcript.append_group(LABEL_COM_S, com_s);
    transcript.append_field(LABEL_MODULUS, &p_hi);
    transcript.append_field(LABEL_MODULUS, &p_lo);
}

/// A limb-lexicographic strict-distinctness proof. Two audited sub-proofs plus one
/// batched custom sumcheck, all bound on one transcript: `hi_nondec` (step 3, also
/// binds `hi_shift = hi_nondec.u_shift_commitment`), `lo_succ` (binds `lo_shift`),
/// and the main sumcheck (`sumcheck` + `opening`) carrying steps 1+4+5+6 over
/// `[com_hi, com_lo, hi_shift, com_lo_shift, aux.., bits..]`.
#[derive(Clone, Debug)]
pub struct LexDistinctProof {
    /// Commitment to the high 127-bit limb column `hi`.
    pub com_hi: ArkGT,
    /// Commitment to the low 127-bit limb column `lo`.
    pub com_lo: ArkGT,
    /// Commitment to the one-step shift `lo_shift[i] = lo[i+1]`.
    pub com_lo_shift: ArkGT,
    /// Step 3: `hi` is nondecreasing (also commits + binds `hi_shift` as its `u_shift`).
    pub hi_nondec: SortedProof,
    /// Step 5 support: the successor lookup binding `lo_shift[i] = lo[i+1]`.
    pub lo_succ: SuccessorProof,
    /// `[Com(e), Com(w_e), Com(g), Com(w_g), Com(w5), Com(t1), Com(t2)]`.
    pub aux_commitments: Vec<ArkGT>,
    /// `Com` of the 5×127 contiguous bit columns (hi, lo, w5, t1, t2).
    pub bit_commitments: Vec<ArkGT>,
    /// The batched main sumcheck carrying steps 1+4+5+6.
    pub sumcheck: SumcheckProof,
    /// Batched Dory opening of every main-sumcheck column at the sumcheck's final point.
    pub opening: ColumnOpening,
}

/// Prove that the committed column `s` (real prefix `[0, n)`, `com_s` its commitment)
/// is STRICTLY LEXICOGRAPHICALLY INCREASING over 2×127-bit limbs — hence its entries
/// are DISTINCT. `s` must be the already-sorted candidate (the permutation to the real
/// hashes is I4-b). `com_s` must commit `s` padded to the gadget's power-of-two domain.
pub fn prove_lex_strictly_sorted(
    s: &[ArkFr],
    n: usize,
    com_s: &ArkGT,
    transcript: &mut Transcript254,
) -> io::Result<LexDistinctProof> {
    prove_lex_strictly_sorted_with_context(&StoreProverContext::cpu(), s, n, com_s, transcript)
}

/// As [`prove_lex_strictly_sorted`], with an explicit prover-arithmetic context.
pub fn prove_lex_strictly_sorted_with_context(
    context: &StoreProverContext<'_>,
    s: &[ArkFr],
    n: usize,
    com_s: &ArkGT,
    transcript: &mut Transcript254,
) -> io::Result<LexDistinctProof> {
    if n == 0 {
        return Err(invalid("lexdistinct requires at least one real row"));
    }
    let n_domain = column_domain(s.len(), n)?;
    if s.len() > n_domain {
        return Err(invalid("lexdistinct column longer than its domain"));
    }
    let (p_hi, p_lo) = modulus_limbs()?;
    let s_p = padded(s, n_domain);
    let mut hi = vec![f0(); n_domain];
    let mut lo = vec![f0(); n_domain];
    for i in 0..n {
        let (h, l) = split_canonical(s_p[i]);
        hi[i] = h;
        lo[i] = l;
    }
    let witness = honest_witness(&hi, &lo, n, n_domain, p_hi, p_lo)?;
    prove_with_witness_and_context(context, n, com_s, &witness, transcript)
}

/// Core prover over an EXPLICIT witness (honest entry derives it; tests inject forged
/// columns to exercise each claim). Commits every column, runs the two audited shift
/// sub-proofs, then the batched main sumcheck — all challenges drawn AFTER commitment.
fn prove_with_witness_and_context(
    context: &StoreProverContext<'_>,
    n: usize,
    com_s: &ArkGT,
    w: &LexWitness,
    transcript: &mut Transcript254,
) -> io::Result<LexDistinctProof> {
    let n_domain = w.hi.len();
    if n_domain < 2 || !n_domain.is_power_of_two() || n > n_domain {
        return Err(invalid("lexdistinct witness domain shape"));
    }
    for col in [&w.lo, &w.hi_shift, &w.lo_shift, &w.e, &w.w_e, &w.g, &w.w_g, &w.w5, &w.t1, &w.t2] {
        if col.len() != n_domain {
            return Err(invalid("lexdistinct witness column length mismatch"));
        }
    }
    let bit_groups = [&w.hi_bits, &w.lo_bits, &w.w5_bits, &w.t1_bits, &w.t2_bits];
    if bit_groups.iter().any(|g| g.len() != B || g.iter().any(|c| c.len() != n_domain)) {
        return Err(invalid("lexdistinct witness bit-column shape mismatch"));
    }
    let (p_hi, p_lo) = modulus_limbs()?;
    let k = n_domain.trailing_zeros() as usize;

    // Commit the limbs + the successor shift up front (commit touches no transcript).
    let com_hi = context.commit_column(&w.hi)?;
    let com_lo = context.commit_column(&w.lo)?;
    let com_lo_shift = context.commit_column(&w.lo_shift)?;

    bind_statement(transcript, n, com_s, p_hi, p_lo);

    // Step 3: hi nondecreasing (also commits + binds hi_shift as its u_shift).
    let hi_nondec = prove_nondecreasing_bits_with_context(context, &w.hi, n, B, &com_hi, transcript)?;
    let com_hi_shift = context.commit_column(&w.hi_shift)?;
    if com_hi_shift != hi_nondec.u_shift_commitment {
        return Err(invalid("lexdistinct hi_shift disagrees with the nondecreasing sub-proof"));
    }

    // Step 5 support: lo_shift[i] = lo[i+1] (range-free successor lookup).
    let lo_succ = prove_successor_lookup_with_context(
        context, &w.lo, &w.lo_shift, n, &com_lo, &com_lo_shift, transcript,
    )?;

    // Main sumcheck: commit the remaining columns, bind all, draw tau, prove.
    let com_e = context.commit_column(&w.e)?;
    let com_w_e = context.commit_column(&w.w_e)?;
    let com_g = context.commit_column(&w.g)?;
    let com_w_g = context.commit_column(&w.w_g)?;
    let com_w5 = context.commit_column(&w.w5)?;
    let com_t1 = context.commit_column(&w.t1)?;
    let com_t2 = context.commit_column(&w.t2)?;
    let aux = vec![com_e, com_w_e, com_g, com_w_g, com_w5, com_t1, com_t2];

    let mut columns: Vec<Vec<ArkFr>> = vec![
        w.hi.clone(),
        w.lo.clone(),
        w.hi_shift.clone(),
        w.lo_shift.clone(),
        w.e.clone(),
        w.w_e.clone(),
        w.g.clone(),
        w.w_g.clone(),
        w.w5.clone(),
        w.t1.clone(),
        w.t2.clone(),
    ];
    let mut bit_commitments = Vec::with_capacity(NUM_BIT_COLS);
    for group in bit_groups {
        for col in group {
            bit_commitments.push(context.commit_column(col)?);
            columns.push(col.clone());
        }
    }

    let mut coms = vec![com_hi, com_lo, com_hi_shift, com_lo_shift];
    coms.extend_from_slice(&aux);
    coms.extend_from_slice(&bit_commitments);
    for c in &coms {
        transcript.append_group(LABEL_MAIN_COM, c);
    }
    let tau = transcript.challenge_scalar(LABEL_TAU);
    let shape = lex_shape(k, n, tau, p_hi, p_lo);
    let (sumcheck, r, _evals) = prove_sumcheck(&shape, &columns, transcript)?;
    let opening = context.prove_column_openings(&columns, &r, transcript)?;

    Ok(LexDistinctProof {
        com_hi,
        com_lo,
        com_lo_shift,
        hi_nondec,
        lo_succ,
        aux_commitments: aux,
        bit_commitments,
        sumcheck,
        opening,
    })
}

/// Verify a limb-lexicographic strict-distinctness proof: `com_s`'s column is strictly
/// lex-increasing over 2×127-bit limbs (hence distinct) on its real prefix `[0, n)`.
pub fn verify_lex_strictly_sorted(
    com_s: &ArkGT,
    n: usize,
    proof: &LexDistinctProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    if n == 0 {
        return Err(invalid("lexdistinct requires at least one real row"));
    }
    let (p_hi, p_lo) = modulus_limbs()?;
    let k = proof.sumcheck.round_polys.len();
    if k == 0 || k >= usize::BITS as usize {
        return Err(invalid("lexdistinct proof has an implausible variable count"));
    }
    let n_domain = 1usize << k;
    if n > n_domain {
        return Err(invalid("lexdistinct real length exceeds the domain"));
    }
    if proof.aux_commitments.len() != NUM_AUX {
        return Err(invalid("lexdistinct proof has the wrong aux-commitment count"));
    }
    if proof.bit_commitments.len() != NUM_BIT_COLS {
        return Err(invalid("lexdistinct proof has the wrong bit-commitment count"));
    }

    bind_statement(transcript, n, com_s, p_hi, p_lo);

    // Step 3 + step 5 support (the two audited shift sub-proofs).
    verify_nondecreasing_bits(&proof.com_hi, n, B, &proof.hi_nondec, transcript)?;
    verify_successor_lookup(&proof.com_lo, &proof.com_lo_shift, n, &proof.lo_succ, transcript)?;

    // Step 2: linear tie Com(S) == 2^127·Com(hi) + Com(lo) (homomorphic, O(1)).
    if *com_s != proof.com_hi.scale(&pow2(B)).add(&proof.com_lo) {
        return Err(invalid("lexdistinct limb linear tie mismatch"));
    }

    // Main sumcheck: steps 1+4+5+6. hi_shift's commitment is the sub-proof's u_shift.
    let com_hi_shift = proof.hi_nondec.u_shift_commitment;
    let mut coms = vec![proof.com_hi, proof.com_lo, com_hi_shift, proof.com_lo_shift];
    coms.extend_from_slice(&proof.aux_commitments);
    coms.extend_from_slice(&proof.bit_commitments);
    for c in &coms {
        transcript.append_group(LABEL_MAIN_COM, c);
    }
    let tau = transcript.challenge_scalar(LABEL_TAU);
    let shape = lex_shape(k, n, tau, p_hi, p_lo);
    let (r, evals) = verify_sumcheck(&shape, &proof.sumcheck, transcript)?;
    verify_column_openings(&coms, &r, &evals, &proof.opening, transcript)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transcript() -> Transcript254 {
        Transcript254::new(b"parture-lexdistinct-test")
    }

    fn fr(v: u64) -> ArkFr {
        ArkFr::from_u64(v)
    }

    /// The field element `hi·2^127 + lo` (a value whose canonical split is `(hi, lo)`
    /// when `hi, lo < 2^127` and `hi·2^127 + lo < p`).
    fn compose(hi: ArkFr, lo: ArkFr) -> ArkFr {
        pow2(B) * hi + lo
    }

    fn commit(col: &[ArkFr]) -> ArkGT {
        StoreProverContext::cpu().commit_column(col).unwrap()
    }

    fn com_of(s: &[ArkFr], n: usize) -> (ArkGT, usize) {
        let nd = column_domain(s.len(), n).unwrap();
        (commit(&padded(s, nd)), nd)
    }

    /// Honest end-to-end: prove + verify a strictly-lex-increasing column.
    fn run(s: &[ArkFr], n: usize) -> io::Result<()> {
        let (com_s, _) = com_of(s, n);
        let proof = prove_lex_strictly_sorted(s, n, &com_s, &mut transcript())?;
        verify_lex_strictly_sorted(&com_s, n, &proof, &mut transcript())
    }

    // ---- POSITIVE ---------------------------------------------------------

    #[test]
    fn strictly_increasing_full_field_column_verifies() {
        let (p_hi, p_lo) = modulus_limbs().unwrap();
        // Lower ⅓ (hi=0), equal-hi-differing-lo pairs, and values near p.
        let s = vec![
            fr(1),
            fr(100_000),
            compose(fr(7), fr(3)),
            compose(fr(7), fr(999)),   // equal hi=7, larger lo
            compose(p_hi, fr(5)),      // near p, hi = p_hi
            compose(p_hi, p_lo - fr(1)), // r-1: max canonical value, equal hi, larger lo
        ];
        assert!(run(&s, s.len()).is_ok());
    }

    #[test]
    fn single_row_and_two_row_and_padding_verify() {
        assert!(run(&[fr(42)], 1).is_ok());
        assert!(run(&[fr(3), fr(9)], 2).is_ok());
        // 5 real rows in a domain of 8, with an equal-hi tie in the middle.
        let s = vec![fr(2), compose(fr(4), fr(1)), compose(fr(4), fr(8)), compose(fr(4), fr(9)), fr(0) + compose(fr(9), fr(0))];
        assert!(run(&s, 5).is_ok());
    }

    // ---- helpers for witness-injection negatives --------------------------

    /// Build the honest witness for the given limb columns and prove with it (so tests
    /// can hand-craft `(hi, lo)`, e.g. a wrap split, while everything downstream is
    /// derived honestly from those limbs).
    fn prove_from_limbs(
        com_s: &ArkGT,
        hi: &[ArkFr],
        lo: &[ArkFr],
        n: usize,
    ) -> io::Result<LexDistinctProof> {
        let (p_hi, p_lo) = modulus_limbs().unwrap();
        let w = honest_witness(hi, lo, n, hi.len(), p_hi, p_lo).unwrap();
        prove_with_witness_and_context(&StoreProverContext::cpu(), n, com_s, &w, &mut transcript())
    }

    /// The honest witness for a value column, for tests that mutate one field.
    fn honest_for(s: &[ArkFr], n: usize) -> (LexWitness, ArkGT, usize) {
        let nd = column_domain(s.len(), n).unwrap();
        let s_p = padded(s, nd);
        let (p_hi, p_lo) = modulus_limbs().unwrap();
        let mut hi = vec![f0(); nd];
        let mut lo = vec![f0(); nd];
        for i in 0..n {
            let (h, l) = split_canonical(s_p[i]);
            hi[i] = h;
            lo[i] = l;
        }
        let w = honest_witness(&hi, &lo, n, nd, p_hi, p_lo).unwrap();
        (w, commit(&s_p), nd)
    }

    fn prove_verify(com_s: &ArkGT, n: usize, w: &LexWitness) -> io::Result<()> {
        let proof =
            prove_with_witness_and_context(&StoreProverContext::cpu(), n, com_s, w, &mut transcript())?;
        verify_lex_strictly_sorted(com_s, n, &proof, &mut transcript())
    }

    // ---- N-dup: duplicate hash rejects (both splits) ----------------------

    #[test]
    fn dup_same_canonical_split_rejects_via_masked_lo() {
        // S = [5, 5]: equal hi ⇒ e=1, equal lo ⇒ lo_shift-lo-1 = -1 ∉ [0,2^127):
        // step 5 (w5 recomposition) rejects.
        assert!(run(&[fr(5), fr(5)], 2).is_err());
    }

    #[test]
    fn dup_wrap_split_evasion_rejects_via_canonical_lt_p_check() {
        // THE crux: two EQUAL hashes v (lower ⅓), one split canonically, the other as
        // the SECOND in-range split (hi', lo') with hi'·2^127 + lo' = v + p. The tie
        // still holds (v+p ≡ v), hi is nondecreasing (0 < p_hi), ranges pass, and e=0
        // (hi differ) so step 5 is inactive — ONLY step 6's `<p` check (t2 range,
        // since hi' == p_hi ⇒ g=1 and p_lo - lo' - 1 < 0) rejects it.
        let (p_hi, p_lo) = modulus_limbs().unwrap();
        let v = fr(12_345);
        let (h0, l0) = split_canonical(v); // canonical: (0, 12345)
        // Wrap split of v: (p_hi, p_lo + v), since p_hi·2^127 + (p_lo+v) = p + v ≡ v,
        // and p_lo + v < 2^127 (both limbs in range) — the forgeable duplicate.
        let hi = vec![h0, p_hi];
        let lo = vec![l0, p_lo + v];
        // Sanity: both rows recompose (mod p) to the SAME value v ⇒ the tie holds.
        assert_eq!(compose(hi[0], lo[0]), v);
        assert_eq!(compose(hi[1], lo[1]), v);
        let com_s = commit(&[v, v]);
        let proof = prove_from_limbs(&com_s, &hi, &lo, 2).unwrap();
        assert!(
            verify_lex_strictly_sorted(&com_s, 2, &proof, &mut transcript()).is_err(),
            "wrap-split duplicate MUST be caught by the canonical <p check (step 6)"
        );
    }

    #[test]
    fn within_column_duplicate_deeper_rejects() {
        // A duplicate that is not adjacent-at-0: [1, 7, 7, 20] → the 7,7 tie rejects.
        assert!(run(&[fr(1), fr(7), fr(7), fr(20)], 4).is_err());
    }

    // ---- N-dec: a decreasing adjacent pair rejects ------------------------

    #[test]
    fn hi_decrease_rejects_via_nondecreasing() {
        // hi = [7, 2] decreases ⇒ step 3 (prove_nondecreasing_bits) rejects.
        let s = vec![compose(fr(7), fr(0)), compose(fr(2), fr(0))];
        assert!(run(&s, 2).is_err());
    }

    #[test]
    fn lo_decrease_under_equal_hi_rejects_via_masked_lo() {
        // hi = [3, 3] equal, lo = [9, 2] decreasing ⇒ step 5 rejects.
        let s = vec![compose(fr(3), fr(9)), compose(fr(3), fr(2))];
        assert!(run(&s, 2).is_err());
    }

    // ---- N-flag: a wrong IsZero flag rejects (completeness both ways) ------

    #[test]
    fn wrong_e_flag_claiming_hi_differ_when_equal_rejects() {
        // Honest strictly-increasing input with an equal-hi tie (e[0] should be 1).
        // Force e[0]=0 (and its consistent w5[0]=0) so ONLY the IsZero COMPLETENESS
        // claim `(1-e) - w_e·(hi_shift-hi) = 1 ≠ 0` fires.
        let s = vec![compose(fr(3), fr(1)), compose(fr(3), fr(5))];
        let (mut w, com_s, _) = honest_for(&s, 2);
        assert_eq!(w.e[0], f1(), "precondition: hi ties so honest e[0]=1");
        w.e[0] = f0();
        w.w5[0] = f0();
        for j in 0..B {
            w.w5_bits[j][0] = f0();
        }
        assert!(prove_verify(&com_s, 2, &w).is_err());
    }

    #[test]
    fn wrong_g_flag_claiming_equal_p_hi_rejects() {
        // Honest input where hi != p_hi (g should be 0). Force g[0]=1 ⇒ the IsZero
        // `g·(hi - p_hi) = hi - p_hi ≠ 0` claim fires (equal direction of completeness).
        let s = vec![fr(11), fr(22)];
        let (mut w, com_s, _) = honest_for(&s, 2);
        assert_eq!(w.g[0], f0(), "precondition: hi != p_hi so honest g[0]=0");
        w.g[0] = f1();
        assert!(prove_verify(&com_s, 2, &w).is_err());
    }

    // ---- N-limb: out-of-range / mismatched limb rejects -------------------

    #[test]
    fn corrupted_recomposition_bit_rejects_via_range() {
        // Flip one hi bit: still boolean, but Σ 2^j·bit_j ≠ hi ⇒ step 1 recomposition
        // rejects. A/B/tie stay honest (they never read the main-sumcheck bit columns),
        // isolating the failure to the 127-bit range recomposition.
        let s = vec![fr(4), fr(30), fr(500)];
        let (mut w, com_s, _) = honest_for(&s, 3);
        w.hi_bits[0][1] = f1() - w.hi_bits[0][1];
        assert!(prove_verify(&com_s, 3, &w).is_err());
    }

    #[test]
    fn nonboolean_bit_rejects_via_booleanity() {
        let s = vec![fr(4), fr(30), fr(500)];
        let (mut w, com_s, _) = honest_for(&s, 3);
        w.lo_bits[0][1] = fr(2); // not in {0,1}
        assert!(prove_verify(&com_s, 3, &w).is_err());
    }

    #[test]
    fn limb_not_matching_com_s_rejects_via_linear_tie() {
        // Bump lo[0] by 1 (hi honest ⇒ step 3 passes; lo_shift unchanged at index ≥1 ⇒
        // the successor lookup passes): Com(lo) changes, so 2^127·Com(hi)+Com(lo) ≠
        // Com(S), and the homomorphic linear tie (step 2) rejects the mismatched split.
        let s = vec![fr(10), fr(4000), fr(50_000)];
        let (mut w, com_s, _) = honest_for(&s, 3);
        w.lo[0] = w.lo[0] + f1();
        assert!(prove_verify(&com_s, 3, &w).is_err());
    }

    #[test]
    fn out_of_range_limb_with_preserved_tie_rejects() {
        // hi[0] += 2^127 (out of range) with lo[0] adjusted to preserve the tie
        // (2^127·hi + lo ≡ S). The 127-bit recomposition of the out-of-range hi (or the
        // nondecreasing sub-proof on the now-huge hi[0]) rejects.
        let s = vec![fr(10), compose(fr(1), fr(0)), compose(fr(2), fr(0))];
        let (mut w, com_s, _) = honest_for(&s, 3);
        let bad_hi0 = w.hi[0] + pow2(B);
        // lo0 s.t. 2^127·bad_hi0 + lo0 = S[0]: lo0 = S[0] - 2^127·bad_hi0.
        let s0 = compose(w.hi[0], w.lo[0]);
        let bad_lo0 = s0 - pow2(B) * bad_hi0;
        let (p_hi, p_lo) = modulus_limbs().unwrap();
        let mut hi = w.hi.clone();
        let mut lo = w.lo.clone();
        hi[0] = bad_hi0;
        lo[0] = bad_lo0;
        let w2 = honest_witness(&hi, &lo, 3, hi.len(), p_hi, p_lo).unwrap();
        assert!(prove_verify(&com_s, 3, &w2).is_err());
    }

    // ---- tamper: mutating a committed proof value rejects -----------------

    #[test]
    fn tampered_sumcheck_round_poly_rejects() {
        let s = vec![fr(1), fr(2), fr(3)];
        let (com_s, _) = com_of(&s, 3);
        let mut proof = prove_lex_strictly_sorted(&s, 3, &com_s, &mut transcript()).unwrap();
        proof.sumcheck.round_polys[0][1] = proof.sumcheck.round_polys[0][1] + f1();
        assert!(verify_lex_strictly_sorted(&com_s, 3, &proof, &mut transcript()).is_err());
    }

    #[test]
    fn wrong_n_rejects() {
        let s = vec![fr(1), fr(2), fr(3)];
        let (com_s, _) = com_of(&s, 3);
        let proof = prove_lex_strictly_sorted(&s, 3, &com_s, &mut transcript()).unwrap();
        assert!(verify_lex_strictly_sorted(&com_s, 2, &proof, &mut transcript()).is_err());
    }
}
