//! Strict-sortedness / range primitive (Phase 2c-2a) — the foundation for join
//! completeness (2c-2). Proves a committed column `U` is STRICTLY INCREASING over
//! its real prefix `[0, n)`, hence its entries are DISTINCT. Clean-room from the
//! math (the design mirrors Proof-of-SQL's production shift/range gadgets, but is
//! implemented from our primitives; no COSL code transcribed).
//!
//! Two pieces, batched into one sumcheck:
//!
//! 1. SHIFT (index-augmented LogUp lookup): commit `U_shift` intended as the
//!    one-step shift `U_shift[i] = U[i+1]`. With the PUBLIC index MLE `idx`
//!    (`Operand::Idx`), fold `c_fold[i] = idx[i] + beta*U[i]` and
//!    `d_fold[i] = (idx[i]+1) + beta*U_shift[i]`. Prove `{d_fold[i] : i<n-1}` is a
//!    sub-multiset of `{c_fold[i]}` with a committed multiplicity `mult_c`:
//!    `Sum chi_{n-1}*rec_d - Sum mult_c*rec_c = 0` where `rec_c/rec_d` are pinned
//!    reciprocals. Because `idx` makes the position part DISTINCT, `d_fold[i]`
//!    (position `i+1`) can only match `c_fold[i+1]`, which forces
//!    `U_shift[i] = U[i+1]` positionally.
//!
//! 2. RANGE: the consecutive difference `d_i = U_shift[i] - U[i]` must lie in
//!    `[1, 2^64)` over `[0, n-1)` (this is what forces STRICT increase /
//!    distinctness). Commit 64 bit-columns `bit_j` of `d_i - 1` and prove, masked
//!    by `chi_{n-1}`:
//!      * Booleanity (batched with a fresh `tau`): `Sum_j tau^j * bit_j*(bit_j-1) = 0`.
//!      * Recomposition: `Sum_j bit_j*2^j - (U_shift - U - 1) = 0`.
//!    64 bits is tight for u64 keys; the ~254-bit field has a >2^190 margin, so a
//!    fake decrease-as-increase via field wraparound needs >64 bits and fails.
//!
//! One shared transcript, two-round FS: commit `U, U_shift, mult_c, bit_0..63` ->
//! draw `beta, alpha, tau` -> commit `rec_c, rec_d` -> sumcheck -> batched Dory
//! opening of all committed columns.

use std::io;

use ark_ff::{BigInteger, PrimeField};
use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::transcript::Transcript;

use super::argument::{
    prove_sumcheck, verify_column_openings, verify_sumcheck,
    ClaimShape, ColumnOpening, Expr, Operand, StoreProverContext, SumcheckClaim, SumcheckProof,
    SumcheckShape, Term, Transcript254,
};
#[cfg(test)]
use super::argument::commit_column;
use super::invalid;
use super::logup::{multiset_link_claim, reciprocal_witness, rlc_fold};

const BITS: usize = 64;
const COL_U: usize = 0;
const COL_U_SHIFT: usize = 1;
const COL_MULT_C: usize = 2;
const COL_REC_C: usize = 3;
const COL_REC_D: usize = 4;
const COL_BIT_BASE: usize = 5;
// chi_bounds index for chi_{n-1} (the difference domain).
const CHI_N1: usize = 0;

const LABEL_BETA: &[u8] = b"pow/sort/fingerprint-beta";
const LABEL_ALPHA: &[u8] = b"pow/sort/reciprocal-alpha";
const LABEL_TAU: &[u8] = b"pow/sort/booleanity-tau";
const LABEL_MODE: &[u8] = b"pow/sort/mode";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SortMode {
    Strict,
    NonDecreasing,
}

impl SortMode {
    fn minimum_gap(self) -> ArkFr {
        match self { Self::Strict => f1(), Self::NonDecreasing => f0() }
    }

    fn bind(self, transcript: &mut Transcript254) {
        transcript.append_bytes(LABEL_MODE, &[match self { Self::Strict => 0, Self::NonDecreasing => 1 }]);
    }
}

#[inline]
fn f0() -> ArkFr {
    ArkFr::zero()
}
#[inline]
fn f1() -> ArkFr {
    ArkFr::one()
}
#[inline]
fn pow2(j: usize) -> ArkFr {
    let mut value = f1();
    for _ in 0..j {
        value = value + value;
    }
    value
}

/// A sortedness proof: the batched sumcheck over the shift LogUp + range
/// (booleanity + recomposition) claims, one batched Dory opening over all
/// committed columns, and the internal witness commitments the verifier binds.
#[derive(Clone, Debug)]
pub struct SortedProof {
    pub sumcheck: SumcheckProof,
    pub opening: ColumnOpening,
    pub u_shift_commitment: ArkGT,
    pub mult_c_commitment: ArkGT,
    pub rec_c_commitment: ArkGT,
    pub rec_d_commitment: ArkGT,
    pub bit_commitments: Vec<ArkGT>,
}

/// The padded power-of-two domain for a column with `n` real rows (`>= 2`).
#[allow(dead_code)]
pub(crate) fn sort_domain(n: usize) -> io::Result<usize> {
    column_domain(n, n)
}

/// The padded power-of-two domain (`>= 2`) large enough for a column of `len`
/// entries with `n` real rows. When the caller already padded to a power-of-two
/// `len` (e.g. a shared cross-gadget domain), that `len` is used as-is.
fn column_domain(len: usize, n: usize) -> io::Result<usize> {
    len.max(n)
        .checked_next_power_of_two()
        .ok_or_else(|| invalid("sort domain size overflow"))
        .map(|d| d.max(2))
}

fn padded(column: &[ArkFr], n: usize) -> Vec<ArkFr> {
    let mut out = column.to_vec();
    out.resize(n, f0());
    out
}

/// The low `num_bits` bits of a field element's canonical representation (as 0/1
/// field elements). For a value in `[0, 2^num_bits)` these are its exact bits; a
/// larger value (e.g. a negative difference `= p - d`) has bits that will NOT
/// recompose to it. Shared with [`super::range`] (which uses a variable bit-width
/// `B+1`); this gadget always asks for `BITS` (= 64).
pub(crate) fn low_bits(value: ArkFr, num_bits: usize) -> Vec<ArkFr> {
    let bigint = value.0.into_bigint();
    (0..num_bits)
        .map(|j| if bigint.get_bit(j) { f1() } else { f0() })
        .collect()
}

/// `rec_col * (alpha - offset - Sum coeff*operand) - 1 = 0`, an Identity claim
/// pinning a committed reciprocal over a fold with an index offset.
fn reciprocal_with_offset(
    rec_col: usize,
    offset: ArkFr,
    fold: &[(ArkFr, Operand)],
    alpha: ArkFr,
) -> SumcheckClaim {
    let mut terms = vec![Term {
        coeff: alpha - offset,
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

/// `chi_{chi_idx} * Sum_{j<num_bits} tau^j * bit_j*(bit_j - 1) = 0` over the bit
/// columns `[bit_base, bit_base + num_bits)`. A single fresh `tau` fold makes one
/// sumcheck claim enforce EVERY bit's booleanity. Shared with [`super::range`],
/// which batches all `B+1` bits including the sign bit.
pub(crate) fn booleanity_claim(
    chi_idx: usize,
    bit_base: usize,
    num_bits: usize,
    tau: ArkFr,
) -> SumcheckClaim {
    let mut terms = Vec::with_capacity(2 * num_bits);
    let mut tau_pow = f1();
    for j in 0..num_bits {
        let bit = Operand::Column(bit_base + j);
        terms.push(Term {
            coeff: tau_pow,
            operands: vec![Operand::Chi(chi_idx), bit, bit],
        });
        terms.push(Term {
            coeff: f0() - tau_pow,
            operands: vec![Operand::Chi(chi_idx), bit],
        });
        tau_pow = tau_pow * tau;
    }
    SumcheckClaim {
        shape: ClaimShape::Identity,
        expr: Expr { terms },
        target: f0(),
    }
}

/// `chi_{n-1} * (Sum_j bit_j*2^j - (U_shift - U - 1)) = 0`.
fn recomposition_claim(bits: usize, mode: SortMode) -> SumcheckClaim {
    let mut terms = Vec::with_capacity(bits + 3);
    for j in 0..bits {
        terms.push(Term {
            coeff: pow2(j),
            operands: vec![Operand::Chi(CHI_N1), Operand::Column(COL_BIT_BASE + j)],
        });
    }
    terms.push(Term {
        coeff: f0() - f1(),
        operands: vec![Operand::Chi(CHI_N1), Operand::Column(COL_U_SHIFT)],
    });
    terms.push(Term {
        coeff: f1(),
        operands: vec![Operand::Chi(CHI_N1), Operand::Column(COL_U)],
    });
    terms.push(Term { coeff: mode.minimum_gap(), operands: vec![Operand::Chi(CHI_N1)] });
    SumcheckClaim {
        shape: ClaimShape::Identity,
        expr: Expr { terms },
        target: f0(),
    }
}

fn sort_shape(
    k: usize,
    n: usize,
    bits: usize,
    alpha: ArkFr,
    beta: ArkFr,
    tau: ArkFr,
    mode: SortMode,
) -> SumcheckShape {
    // rec_c: c_fold = idx + beta*U. rec_d: d_fold = (idx+1) + beta*U_shift.
    let rec_c = reciprocal_with_offset(
        COL_REC_C,
        f0(),
        &[(f1(), Operand::Idx), (beta, Operand::Column(COL_U))],
        alpha,
    );
    let rec_d = reciprocal_with_offset(
        COL_REC_D,
        f1(),
        &[(f1(), Operand::Idx), (beta, Operand::Column(COL_U_SHIFT))],
        alpha,
    );
    // Shift lookup: Sum chi_{n-1}*rec_d - Sum mult_c*rec_c = 0.
    let shift = multiset_link_claim(
        Operand::Chi(CHI_N1),
        COL_REC_D,
        Operand::Column(COL_MULT_C),
        COL_REC_C,
    );
    SumcheckShape {
        num_vars: k,
        num_columns: COL_BIT_BASE + bits,
        chi_bounds: vec![n.saturating_sub(1)],
        claims: vec![
            rec_c,
            rec_d,
            shift,
            booleanity_claim(CHI_N1, COL_BIT_BASE, bits, tau),
            recomposition_claim(bits, mode),
        ],
    }
}

/// Prove strict sortedness with an EXPLICIT witness (used by the honest entry
/// point and, in tests, to inject a forged shift/multiplicity/bit witness). All
/// witness columns are length-`n_domain`.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn prove_strictly_sorted_with_witness(
    u: &[ArkFr],
    n: usize,
    com_u: &ArkGT,
    u_shift: &[ArkFr],
    mult_c: &[ArkFr],
    bit_columns: &[Vec<ArkFr>],
    transcript: &mut Transcript254,
) -> io::Result<SortedProof> {
    prove_strictly_sorted_with_witness_and_context(
        &StoreProverContext::cpu(),
        u,
        n,
        com_u,
        u_shift,
        mult_c,
        bit_columns,
        transcript,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_strictly_sorted_with_witness_and_context(
    context: &StoreProverContext<'_>,
    u: &[ArkFr],
    n: usize,
    com_u: &ArkGT,
    u_shift: &[ArkFr],
    mult_c: &[ArkFr],
    bit_columns: &[Vec<ArkFr>],
    transcript: &mut Transcript254,
) -> io::Result<SortedProof> {
    prove_sorted_with_witness_and_context(context, u, n, com_u, u_shift, mult_c, bit_columns, SortMode::Strict, transcript)
}

#[allow(clippy::too_many_arguments)]
fn prove_sorted_with_witness_and_context(
    context: &StoreProverContext<'_>,
    u: &[ArkFr],
    n: usize,
    com_u: &ArkGT,
    u_shift: &[ArkFr],
    mult_c: &[ArkFr],
    bit_columns: &[Vec<ArkFr>],
    mode: SortMode,
    transcript: &mut Transcript254,
) -> io::Result<SortedProof> {
    let n_domain = column_domain(u.len(), n)?;
    if u.len() > n_domain || n > n_domain {
        return Err(invalid("sorted column longer than its domain"));
    }
    if u_shift.len() != n_domain || mult_c.len() != n_domain {
        return Err(invalid("shift/multiplicity column length mismatch"));
    }
    let bits = bit_columns.len();
    if bits == 0 || bits > 253 || bit_columns.iter().any(|c| c.len() != n_domain) {
        return Err(invalid("bit column layout mismatch"));
    }
    let k = n_domain.trailing_zeros() as usize;
    let u_p = padded(u, n_domain);

    // Round 1: commit U_shift, mult_c, and the 64 bit columns (raw-data witnesses),
    // bind them (with the given com_u).
    let u_shift_commitment = context.commit_column(u_shift)?;
    let mult_c_commitment = context.commit_column(mult_c)?;
    let mut bit_commitments = Vec::with_capacity(bits);
    for column in bit_columns {
        bit_commitments.push(context.commit_column(column)?);
    }
    mode.bind(transcript);
    transcript.append_group(b"pow/sort/com-u", com_u);
    transcript.append_group(b"pow/sort/com-u-shift", &u_shift_commitment);
    transcript.append_group(b"pow/sort/com-mult-c", &mult_c_commitment);
    for commitment in &bit_commitments {
        transcript.append_group(b"pow/sort/com-bit", commitment);
    }
    let beta = transcript.challenge_scalar(LABEL_BETA);
    let alpha = transcript.challenge_scalar(LABEL_ALPHA);
    let tau = transcript.challenge_scalar(LABEL_TAU);

    // Round 2: reciprocal witnesses over the fingerprinted (index-augmented) rows.
    let idx_col: Vec<ArkFr> = (0..n_domain).map(|i| ArkFr::from_u64(i as u64)).collect();
    let idx_plus1: Vec<ArkFr> = (0..n_domain)
        .map(|i| ArkFr::from_u64(i as u64 + 1))
        .collect();
    let rec_c = reciprocal_witness(alpha, &rlc_fold(beta, &[&idx_col, &u_p]));
    let rec_d = reciprocal_witness(alpha, &rlc_fold(beta, &[&idx_plus1, u_shift]));
    let rec_c_commitment = context.commit_column(&rec_c)?;
    let rec_d_commitment = context.commit_column(&rec_d)?;
    transcript.append_group(b"pow/sort/com-rec-c", &rec_c_commitment);
    transcript.append_group(b"pow/sort/com-rec-d", &rec_d_commitment);

    let shape = sort_shape(k, n, bits, alpha, beta, tau, mode);
    let mut columns = vec![u_p, u_shift.to_vec(), mult_c.to_vec(), rec_c, rec_d];
    columns.extend(bit_columns.iter().cloned());

    let (sumcheck, r, _evals) = prove_sumcheck(&shape, &columns, transcript)?;
    let opening = context.prove_column_openings(&columns, &r, transcript)?;

    Ok(SortedProof {
        sumcheck,
        opening,
        u_shift_commitment,
        mult_c_commitment,
        rec_c_commitment,
        rec_d_commitment,
        bit_commitments,
    })
}

/// The honest witnesses: `U_shift[i] = U[i+1]`; `mult_c[i] = 1` for `i in [1,n)`;
/// `bit_columns[j][i]` is bit `j` of the adjacent gap minus the mode's minimum
/// gap (`1` for strict, `0` for nondecreasing) for `i < n-1`.
fn honest_witness(
    u: &[ArkFr],
    n: usize,
    n_domain: usize,
    bits: usize,
    mode: SortMode,
) -> (Vec<ArkFr>, Vec<ArkFr>, Vec<Vec<ArkFr>>) {
    let u_p = padded(u, n_domain);
    let mut u_shift = vec![f0(); n_domain];
    for i in 0..n.saturating_sub(1) {
        u_shift[i] = u_p[i + 1];
    }
    let mut mult_c = vec![f0(); n_domain];
    for i in 1..n {
        mult_c[i] = f1();
    }
    let mut bit_columns = vec![vec![f0(); n_domain]; bits];
    for i in 0..n.saturating_sub(1) {
        let value_bits = low_bits(u_shift[i] - u_p[i] - mode.minimum_gap(), bits);
        for j in 0..bits {
            bit_columns[j][i] = value_bits[j];
        }
    }
    (u_shift, mult_c, bit_columns)
}

/// Prove that `com_u`'s column is STRICTLY INCREASING over `[0, n)` (entries
/// distinct). `com_u` must commit `U` padded to `sort_domain(n)`.
pub fn prove_strictly_sorted(
    u: &[ArkFr],
    n: usize,
    com_u: &ArkGT,
    transcript: &mut Transcript254,
) -> io::Result<SortedProof> {
    prove_strictly_sorted_with_context(&StoreProverContext::cpu(), u, n, com_u, transcript)
}

pub fn prove_strictly_sorted_with_context(
    context: &StoreProverContext<'_>,
    u: &[ArkFr],
    n: usize,
    com_u: &ArkGT,
    transcript: &mut Transcript254,
) -> io::Result<SortedProof> {
    let n_domain = column_domain(u.len(), n)?;
    let (u_shift, mult_c, bit_columns) = honest_witness(u, n, n_domain, BITS, SortMode::Strict);
    prove_strictly_sorted_with_witness_and_context(
        context, u, n, com_u, &u_shift, &mult_c, &bit_columns, transcript,
    )
}

/// Prove strict ordering for values whose positive consecutive differences fit in
/// `bits`. Composite two-u64 keys use 128; scalar graph IDs use the 64-bit wrapper.
pub fn prove_strictly_sorted_bits(
    u: &[ArkFr],
    n: usize,
    bits: usize,
    com_u: &ArkGT,
    transcript: &mut Transcript254,
) -> io::Result<SortedProof> {
    prove_strictly_sorted_bits_with_context(
        &StoreProverContext::cpu(), u, n, bits, com_u, transcript,
    )
}

pub fn prove_strictly_sorted_bits_with_context(
    context: &StoreProverContext<'_>,
    u: &[ArkFr],
    n: usize,
    bits: usize,
    com_u: &ArkGT,
    transcript: &mut Transcript254,
) -> io::Result<SortedProof> {
    if bits == 0 || bits > 253 {
        return Err(invalid("sort bit width must be in 1..=253"));
    }
    let n_domain = column_domain(u.len(), n)?;
    let (u_shift, mult_c, bit_columns) = honest_witness(u, n, n_domain, bits, SortMode::Strict);
    prove_strictly_sorted_with_witness_and_context(
        context, u, n, com_u, &u_shift, &mult_c, &bit_columns, transcript,
    )
}

/// Prove nondecreasing order over `[0, n)`. Equal adjacent values are allowed;
/// decreases still cannot fit the bounded difference decomposition.
pub fn prove_nondecreasing_bits_with_context(
    context: &StoreProverContext<'_>,
    u: &[ArkFr],
    n: usize,
    bits: usize,
    com_u: &ArkGT,
    transcript: &mut Transcript254,
) -> io::Result<SortedProof> {
    if bits == 0 || bits > 253 { return Err(invalid("sort bit width must be in 1..=253")); }
    let n_domain = column_domain(u.len(), n)?;
    let (u_shift, mult_c, bit_columns) = honest_witness(u, n, n_domain, bits, SortMode::NonDecreasing);
    prove_sorted_with_witness_and_context(context, u, n, com_u, &u_shift, &mult_c, &bit_columns, SortMode::NonDecreasing, transcript)
}

/// Verify a strict-sortedness proof.
pub fn verify_strictly_sorted(
    com_u: &ArkGT,
    n: usize,
    proof: &SortedProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    verify_strictly_sorted_bits(com_u, n, BITS, proof, transcript)
}

/// Verify the variable-width strict-order argument used by composite keys.
pub fn verify_strictly_sorted_bits(
    com_u: &ArkGT,
    n: usize,
    bits: usize,
    proof: &SortedProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    verify_sorted_bits(com_u, n, bits, proof, SortMode::Strict, transcript)
}

pub fn verify_nondecreasing_bits(
    com_u: &ArkGT,
    n: usize,
    bits: usize,
    proof: &SortedProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    verify_sorted_bits(com_u, n, bits, proof, SortMode::NonDecreasing, transcript)
}

fn verify_sorted_bits(
    com_u: &ArkGT,
    n: usize,
    bits: usize,
    proof: &SortedProof,
    mode: SortMode,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    let k = proof.sumcheck.round_polys.len();
    if k == 0 || k >= usize::BITS as usize {
        return Err(invalid("sort proof has an implausible variable count"));
    }
    if n > (1usize << k) {
        return Err(invalid("real length exceeds the sort domain"));
    }
    if bits == 0 || bits > 253 || proof.bit_commitments.len() != bits {
        return Err(invalid(
            "sort proof has the wrong number of bit commitments",
        ));
    }

    mode.bind(transcript);
    transcript.append_group(b"pow/sort/com-u", com_u);
    transcript.append_group(b"pow/sort/com-u-shift", &proof.u_shift_commitment);
    transcript.append_group(b"pow/sort/com-mult-c", &proof.mult_c_commitment);
    for commitment in &proof.bit_commitments {
        transcript.append_group(b"pow/sort/com-bit", commitment);
    }
    let beta = transcript.challenge_scalar(LABEL_BETA);
    let alpha = transcript.challenge_scalar(LABEL_ALPHA);
    let tau = transcript.challenge_scalar(LABEL_TAU);
    transcript.append_group(b"pow/sort/com-rec-c", &proof.rec_c_commitment);
    transcript.append_group(b"pow/sort/com-rec-d", &proof.rec_d_commitment);

    let shape = sort_shape(k, n, bits, alpha, beta, tau, mode);
    let (r, evals) = verify_sumcheck(&shape, &proof.sumcheck, transcript)?;

    let mut commitments = vec![
        *com_u,
        proof.u_shift_commitment,
        proof.mult_c_commitment,
        proof.rec_c_commitment,
        proof.rec_d_commitment,
    ];
    commitments.extend(proof.bit_commitments.iter().copied());
    verify_column_openings(&commitments, &r, &evals, &proof.opening, transcript)
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
        Transcript254::new(b"parture-sort-test")
    }

    fn commit_u(u: &[ArkFr], n: usize) -> ArkGT {
        let n_domain = sort_domain(n).unwrap();
        commit_column(&padded(u, n_domain)).unwrap()
    }

    fn run(u: &[ArkFr], n: usize) -> io::Result<()> {
        let com_u = commit_u(u, n);
        let mut pt = transcript();
        let proof = prove_strictly_sorted(u, n, &com_u, &mut pt)?;
        let mut vt = transcript();
        verify_strictly_sorted(&com_u, n, &proof, &mut vt)
    }

    fn run_nondecreasing(u: &[ArkFr], n: usize) -> io::Result<()> {
        let com_u = commit_u(u, n);
        let proof = prove_nondecreasing_bits_with_context(
            &StoreProverContext::cpu(), u, n, BITS, &com_u, &mut transcript(),
        )?;
        verify_nondecreasing_bits(&com_u, n, BITS, &proof, &mut transcript())
    }

    #[test]
    fn sorted_distinct_input_verifies() {
        let u = frs(&[3, 7, 10, 42, 1000]);
        assert!(run(&u, u.len()).is_ok());
    }

    #[test]
    fn adjacent_inversion_is_rejected() {
        // U[2] < U[1]: the difference 3 - 7 is negative -> out of [1, 2^64).
        let u = frs(&[1, 7, 3, 100]);
        assert!(run(&u, u.len()).is_err());
    }

    #[test]
    fn duplicate_is_rejected() {
        // U[2] == U[1]: gap 0, and 0 is not in [1, 2^64) (d-1 = -1 can't recompose).
        let u = frs(&[1, 5, 5, 9]);
        assert!(run(&u, u.len()).is_err());
    }

    #[test]
    fn nondecreasing_accepts_duplicates_all_equal_boundaries_and_padding() {
        assert!(run_nondecreasing(&frs(&[1, 5, 5, 9]), 4).is_ok());
        assert!(run_nondecreasing(&frs(&[7, 7, 7]), 3).is_ok());
        assert!(run_nondecreasing(&frs(&[0, u64::MAX]), 2).is_ok());
        assert!(run_nondecreasing(&frs(&[1, 1, 2]), 3).is_ok());
        assert!(run_nondecreasing(&frs(&[10, 2]), 2).is_err());
    }

    #[test]
    fn strict_and_nondecreasing_modes_cannot_be_substituted() {
        let u = frs(&[1, 2, 3]);
        let com_u = commit_u(&u, u.len());
        let strict = prove_strictly_sorted(&u, u.len(), &com_u, &mut transcript()).unwrap();
        assert!(verify_nondecreasing_bits(&com_u, u.len(), BITS, &strict, &mut transcript()).is_err());
        let nondecreasing = prove_nondecreasing_bits_with_context(
            &StoreProverContext::cpu(), &u, u.len(), BITS, &com_u, &mut transcript(),
        ).unwrap();
        assert!(verify_strictly_sorted(&com_u, u.len(), &nondecreasing, &mut transcript()).is_err());
    }

    #[test]
    fn field_wraparound_decrease_is_rejected() {
        // A decrease U=[10, 2]: an attacker cannot pass it as an increase, because
        // the "difference" 2 - 10 - 1 = -9 = p - 9 needs the full ~254-bit field to
        // represent -- the honest 64-bit decomposition recomposes to 2^64 - 9 != p-9.
        let u = frs(&[10, 2]);
        assert!(run(&u, u.len()).is_err());
    }

    #[test]
    fn forged_bit_decomposition_is_rejected() {
        let u = frs(&[2, 9, 20]);
        let n = u.len();
        let n_domain = sort_domain(n).unwrap();
        let com_u = commit_u(&u, n);
        let (u_shift, mult_c, honest_bits) = honest_witness(&u, n, n_domain, BITS, SortMode::Strict);

        // Non-boolean bit at a real row -> booleanity rejects.
        let mut non_boolean = honest_bits.clone();
        non_boolean[0][0] = fr(2);
        let mut pt = transcript();
        let proof = prove_strictly_sorted_with_witness(
            &u,
            n,
            &com_u,
            &u_shift,
            &mult_c,
            &non_boolean,
            &mut pt,
        )
        .unwrap();
        let mut vt = transcript();
        assert!(verify_strictly_sorted(&com_u, n, &proof, &mut vt).is_err());

        // Boolean bits that do NOT recompose to d-1 -> recomposition rejects.
        let mut wrong_recompose = honest_bits;
        wrong_recompose[0][0] = f1() - wrong_recompose[0][0]; // flip the low bit of row 0
        let mut pt2 = transcript();
        let proof2 = prove_strictly_sorted_with_witness(
            &u,
            n,
            &com_u,
            &u_shift,
            &mult_c,
            &wrong_recompose,
            &mut pt2,
        )
        .unwrap();
        let mut vt2 = transcript();
        assert!(verify_strictly_sorted(&com_u, n, &proof2, &mut vt2).is_err());
    }

    #[test]
    fn single_element_and_padded_boundaries_verify() {
        // n = 1: no differences.
        assert!(run(&frs(&[42]), 1).is_ok());
        // Small n with domain padding (n=3 -> domain 4).
        assert!(run(&frs(&[1, 2, 3]), 3).is_ok());
        // Near-full padded capacity: 15 real rows in a domain of 16.
        let u: Vec<ArkFr> = (0..15u64).map(|i| fr(i * 3 + 1)).collect();
        assert!(run(&u, 15).is_ok());
    }
}
