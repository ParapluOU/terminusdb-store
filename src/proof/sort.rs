//! Strict-sortedness / range primitive (Phase 2c-2a) — the foundation for join
//! completeness (2c-2). Proves a committed column `U` is STRICTLY INCREASING over
//! its real prefix `[0, n)`, hence its entries are DISTINCT. Clean-room from the
//! math (the design mirrors Proof-of-SQL's production shift/range gadgets, but is
//! implemented from our primitives; no COSL code transcribed).
//!
//! For a developer: this [gadget](crate::proof::primer) certifies "this committed
//! column is sorted (and therefore its values are distinct)". Distinctness is what
//! lets the join gadgets treat a key column as a set. It combines a
//! [sumcheck](crate::proof::primer) (primer §5) proving each adjacent gap
//! decomposes into 64 committed bits in `[1, 2^64)` — a positive difference, hence a
//! strict increase, with the wide field ruling out wraparound — and a
//! [LogUp](crate::proof::primer) (primer §7) "shift" link binding `U_shift[i]` to
//! `U[i+1]`. Challenges are Fiat–Shamir, drawn after the data is committed ([the
//! transcript invariant](crate::proof::primer), primer §8). See [the
//! primer](crate::proof::primer) for [commitments](crate::proof::primer) (§3) and
//! [prove → verify](crate::proof::primer) (§10).
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
    mle_full_eval, prove_sumcheck, verify_column_openings, verify_sumcheck,
    ClaimShape, ColumnOpening, Expr, Operand, StoreProverContext, SumcheckClaim, SumcheckProof,
    SumcheckShape, Term, Transcript254,
};
#[cfg(test)]
use super::argument::commit_column;
use super::invalid;
use super::logup_gkr::{
    bind_and_verify_fraction_tree, check_leaf_tie, prove_fraction_sum_zero, FractionInstance,
    GkrFractionProof, LinearForm, LinearTerm, Src,
};

const BITS: usize = 64;
// The RANGE sumcheck keeps U/U_shift at columns 0/1 (`recomposition_claim` reuses these);
// the shift LogUp's reciprocal columns (`rec_c`/`rec_d`) and `mult_c`/bit-base column
// indices are gone — both SortedProof and SuccessorProof moved the shift link to the
// block-disjoint LogUp-GKR fraction tree (`sort_link_instance`, LINK_COL_* below).
const COL_U: usize = 0;
const COL_U_SHIFT: usize = 1;
// chi_bounds index for chi_{n-1} (the difference domain).
const CHI_N1: usize = 0;

// Range-only sumcheck column layout (booleanity + recomposition) AFTER the shift LogUp
// moved to the GKR fraction tree: [U, U_shift, bit_0..]. U/U_shift keep indices 0/1
// (= COL_U/COL_U_SHIFT), so `recomposition_claim` reuses those; only the bit base moves.
const RANGE_COL_BIT_BASE: usize = 2;

// GKR shift-link instance columns (length-`domain` committed, block-disjoint
// zero-extended into the `2*domain` leaf vector): U (low half, c-side), U_shift (high
// half, d-side), mult_c (low half, the committed c-side numerator).
const LINK_COL_U: usize = 0;
const LINK_COL_U_SHIFT: usize = 1;
const LINK_COL_MULT_C: usize = 2;

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

/// A sortedness proof. Two sub-proofs bound on one transcript:
///   * the RANGE sumcheck (booleanity + recomposition), pinning the bit decomposition
///     of the consecutive gap, plus one batched Dory opening of `[U, U_shift, bit_0..]`
///     at its point (UNCHANGED by the LogUp-GKR migration);
///   * the SHIFT LogUp LINK — a [`GkrFractionProof`] over an UNCOMMITTED block-disjoint
///     fraction layout proving `U_shift` is the index-augmented one-step shift of `U`
///     with committed multiplicity `mult_c`, plus the link columns' evaluations and one
///     batched Dory opening at the leaf point. This REPLACES the two committed reciprocal
///     columns (`rec_c`/`rec_d`) and the multiset-link sumcheck; no reciprocal is
///     committed anywhere.
#[derive(Clone, Debug)]
pub struct SortedProof {
    /// The RANGE sumcheck: booleanity of the gap bits plus their recomposition to the
    /// consecutive difference.
    pub sumcheck: SumcheckProof,
    /// Batched Dory opening of `[U, U_shift, bit_0..]` at the RANGE sumcheck's point.
    pub opening: ColumnOpening,
    /// Commitment to the one-step shift column `U_shift` (`U_shift[i] = U[i+1]`).
    pub u_shift_commitment: ArkGT,
    /// Commitment to the shift-link multiplicity column `mult_c`.
    pub mult_c_commitment: ArkGT,
    /// GKR fraction tree for the shift LogUp
    /// `Σ [i<n-1]·(α−d_fold)⁻¹ − Σ mult_c·(α−c_fold)⁻¹ = 0`.
    pub link: GkrFractionProof,
    /// Evaluations of `[U, U_shift, mult_c]` at `r_low` (the low `k` bits of the GKR
    /// leaf point), bound to their commitments by `link_opening`.
    pub link_col_evals: Vec<ArkFr>,
    /// Batched Dory opening authenticating `link_col_evals` at the leaf point.
    pub link_opening: ColumnOpening,
    /// Commitments to the gap's bit columns `bit_0..` (one per bit of the difference).
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

/// `chi_{n-1} * (Sum_j bit_j*2^j - (U_shift - U - 1)) = 0`. The bit columns start at
/// `bit_base` (the range-only sumcheck packs them right after `U`/`U_shift`).
fn recomposition_claim(bit_base: usize, bits: usize, mode: SortMode) -> SumcheckClaim {
    let mut terms = Vec::with_capacity(bits + 3);
    for j in 0..bits {
        terms.push(Term {
            coeff: pow2(j),
            operands: vec![Operand::Chi(CHI_N1), Operand::Column(bit_base + j)],
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

/// The RANGE sumcheck shape: booleanity + recomposition over `[U, U_shift, bit_0..]`.
/// This is the old `sort_shape` MINUS the shift LogUp (the two `rec_c`/`rec_d` Identity
/// claims + the multiset link), which the LogUp-GKR migration moved to the block-disjoint
/// fraction tree [`sort_link_instance`]. `num_columns` is `RANGE_COL_BIT_BASE + bits`
/// (= `U`, `U_shift`, and the `bits` bit columns; NO reciprocal or `mult_c` columns).
fn range_shape(k: usize, n: usize, bits: usize, tau: ArkFr, mode: SortMode) -> SumcheckShape {
    SumcheckShape {
        num_vars: k,
        num_columns: RANGE_COL_BIT_BASE + bits,
        chi_bounds: vec![n.saturating_sub(1)],
        claims: vec![
            booleanity_claim(CHI_N1, RANGE_COL_BIT_BASE, bits, tau),
            recomposition_claim(RANGE_COL_BIT_BASE, bits, mode),
        ],
    }
}

/// The PUBLIC block-disjoint fraction-sum-zero instance for the SHIFT LogUp — the
/// LogUp-GKR replacement for the two committed reciprocals `rec_c`/`rec_d` and the
/// [`multiset_link_claim`]. It proves EXACTLY the old link
/// `Σ_x [ Chi_{n-1}(x)·rec_d(x) − mult_c(x)·rec_c(x) ] = 0`, i.e.
/// ```text
///     Σ_i [i<n-1]·(α − d_fold_i)⁻¹   −   Σ_i mult_c_i·(α − c_fold_i)⁻¹   =   0
/// ```
/// with `c_fold_i = i + β·U_i` and `d_fold_i = (i+1) + β·U_shift_i`, over a `2·domain`
/// leaf vector (`domain = 2^k`): the c-side rows fill the LOW half `[0, domain)`, the
/// d-side rows the HIGH half `[domain, 2·domain)`. Column layout (all committed, length
/// `domain`, block-disjoint zero-extended by the prover): `U` (col 0, low), `U_shift`
/// (col 1, high), `mult_c` (col 2, low).
///
///   * denominator
///     `q = α + (domain−1) − Idx − β·U − β·U_shift − (domain−1)·[i<domain]`.
///     The PUBLIC `Idx` ramp supplies the global row index; on the LOW half
///     (`[i<domain]=1`) the `(domain−1)` terms cancel and `q = α − (i + β·U_i)`; on the
///     HIGH half (`[i<domain]=0`, `Idx = domain + i_local`) they add `(domain−1)` back, so
///     `q = α − ((i_local+1) + β·U_shift_i)`. The `β` fingerprints the value on both
///     sides; the `Idx` makes every c_fold pole DISTINCT.
///   * numerator `p = −mult_c (low, committed) + ([i < domain+(n−1)] − [i < domain])
///     (high, public prefix = +Chi_{n-1} on the d block)`: each d-side real row `+1`, each
///     c-side row `−mult_c_i`, every padding row `0`.
///
/// SOUNDNESS. Both numerators are LINEAR (public `Src::Prefix`/`Src::Idx` and `mult_c` as
/// a committed `Src::Column`); there is NO `committed × public-Chi` product — the c-side
/// `mult_c` carries NO Chi mask (matching the reciprocal-era link, where only `rec_d` was
/// Chi-masked) — so no GAP-1 auxiliary Identity is required. `mult_c`'s exact value, its
/// zero-on-padding, and the genuine positional bijection (`U_shift_i = U_{i+1}`, no
/// drops/dups) are all forced by the LINK itself: the `Idx`-augmented c_fold poles are
/// DISTINCT (`i` distinct; `β` post-commit), so `Σ p/q = 0 ⟺` the residue at each c_fold
/// pole vanishes, uniquely pinning `mult_c_i` to the count of d-rows landing there — 1 on
/// `[1,n)`, 0 on padding — and forcing each d-row `d_fold_i` to match the UNIQUE
/// `c_fold_{i+1}`. This is the same distinctness argument the reciprocal era relied on.
/// Built IDENTICALLY by prover and verifier from public inputs only.
fn sort_link_instance(alpha: ArkFr, beta: ArkFr, n: usize, k: usize) -> FractionInstance {
    let domain = 1usize << k;
    let n1 = n.saturating_sub(1);
    let domain_minus_1 = ArkFr::from_u64((domain - 1) as u64);
    let q_terms = vec![
        LinearTerm { coeff: f0() - f1(), src: Src::Idx },
        LinearTerm { coeff: f0() - beta, src: Src::Column(LINK_COL_U) },
        LinearTerm { coeff: f0() - beta, src: Src::Column(LINK_COL_U_SHIFT) },
        LinearTerm { coeff: f0() - domain_minus_1, src: Src::Prefix(domain) },
    ];
    let p_terms = vec![
        LinearTerm { coeff: f0() - f1(), src: Src::Column(LINK_COL_MULT_C) },
        LinearTerm { coeff: f1(), src: Src::Prefix(domain + n1) },
        LinearTerm { coeff: f0() - f1(), src: Src::Prefix(domain) },
    ];
    FractionInstance {
        num_vars: k + 1,
        num_cols: 3,
        p_form: LinearForm { constant: f0(), terms: p_terms },
        q_form: LinearForm { constant: alpha + domain_minus_1, terms: q_terms },
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

    // Commit U_shift, mult_c, and the bit columns (raw-data witnesses), bind them (with
    // the given com_u), then draw beta/alpha/tau. NO reciprocal column is committed
    // anymore — the shift LogUp is a GKR fraction tree over UNCOMMITTED intermediates, so
    // alpha/beta are drawn here, AFTER every data-column commitment (the Inc 0 caller
    // contract: leaves fixed before any per-layer GKR challenge).
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

    // (RANGE) booleanity + recomposition over [U, U_shift, bit_0..] — UNCHANGED math.
    let range = range_shape(k, n, bits, tau, mode);
    let mut range_columns = vec![u_p.clone(), u_shift.to_vec()];
    range_columns.extend(bit_columns.iter().cloned());
    let (sumcheck, r, _evals) = prove_sumcheck(&range, &range_columns, transcript)?;
    let opening = context.prove_column_openings(&range_columns, &r, transcript)?;

    // (SHIFT LINK) the index-augmented multiset link as ONE GKR fraction tree over the
    // block-disjoint 2*domain layout: the c-side (U) fills the low half, the d-side
    // (U_shift) the high half, and mult_c (low) is the committed c-side numerator. The
    // prover feeds the tree the columns' block-disjoint zero-extensions; the committed
    // columns stay length-`domain` (their commitments are UNCHANGED).
    let leaf_count = 2 * n_domain;
    let mut u_ext = vec![f0(); leaf_count];
    u_ext[..n_domain].copy_from_slice(&u_p);
    let mut u_shift_ext = vec![f0(); leaf_count];
    u_shift_ext[n_domain..].copy_from_slice(u_shift);
    let mut mult_c_ext = vec![f0(); leaf_count];
    mult_c_ext[..n_domain].copy_from_slice(mult_c);
    let extended = vec![u_ext, u_shift_ext, mult_c_ext];

    let link_inst = sort_link_instance(alpha, beta, n, k);
    let (link, r_leaf) = prove_fraction_sum_zero(&link_inst, &extended, transcript)?;

    // Single batched Dory opening of the length-`domain` link columns [U, U_shift, mult_c]
    // at r_low, bound to their commitments on this transcript.
    let r_low = &r_leaf[..k];
    let link_columns = vec![u_p, u_shift.to_vec(), mult_c.to_vec()];
    let link_col_evals: Vec<ArkFr> =
        link_columns.iter().map(|c| mle_full_eval(c, r_low)).collect();
    let link_opening = context.prove_column_openings(&link_columns, r_low, transcript)?;

    Ok(SortedProof {
        sumcheck,
        opening,
        u_shift_commitment,
        mult_c_commitment,
        link,
        link_col_evals,
        link_opening,
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

/// As [`prove_strictly_sorted`], with an explicit prover-arithmetic context.
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

/// As [`prove_strictly_sorted_bits`], with an explicit prover-arithmetic context.
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

/// Verify a NONDECREASING-order proof over `[0, n)` at bit width `bits` (equal adjacent
/// values allowed; decreases rejected). The counterpart of the strict-order verifier used
/// by `OrderBy`/lexicographic sorts.
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
    if proof.link.num_vars() != k + 1 {
        return Err(invalid("sort shift-link fraction layer count mismatch"));
    }
    if proof.link_col_evals.len() != 3 {
        return Err(invalid("sort shift-link column-evaluation count mismatch"));
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

    // (RANGE) verify booleanity + recomposition, open [U, U_shift, bit_0..] at r.
    let range = range_shape(k, n, bits, tau, mode);
    let (r, evals) = verify_sumcheck(&range, &proof.sumcheck, transcript)?;
    let mut range_coms = vec![*com_u, proof.u_shift_commitment];
    range_coms.extend(proof.bit_commitments.iter().copied());
    verify_column_openings(&range_coms, &r, &evals, &proof.opening, transcript)?;

    // (SHIFT LINK) bind the public block-disjoint instance and verify the GKR tree.
    let link_inst = sort_link_instance(alpha, beta, n, k);
    let (r_leaf, p_claim, q_claim) =
        bind_and_verify_fraction_tree(&link_inst, &proof.link, transcript)?;

    // Open the length-`domain` link columns [U, U_shift, mult_c] at r_low, bound to their
    // commitments. `mult_c` is a PLAIN committed numerator (NO Chi mask); its exact value,
    // zero-on-padding, and the genuine positional bijection are forced by the link's
    // DISTINCT Idx-augmented c_fold poles — no co-located sub-proof or GAP-1 Identity.
    let r_low = &r_leaf[..k];
    let r_high = r_leaf[k];
    let link_coms = vec![*com_u, proof.u_shift_commitment, proof.mult_c_commitment];
    verify_column_openings(&link_coms, r_low, &proof.link_col_evals, &proof.link_opening, transcript)?;

    // Reconstruct each 2*domain leaf-MLE from the length-`domain` opening via the top-bit
    // zero-extension factor: U + mult_c live in the LOW half (factor 1 − r_high), U_shift
    // in the HIGH half (factor r_high) — structurally forcing each side's columns to zero
    // on the other block (no cross-block leak).
    let one_minus_high = f1() - r_high;
    let leaf_evals = vec![
        one_minus_high * proof.link_col_evals[LINK_COL_U],
        r_high * proof.link_col_evals[LINK_COL_U_SHIFT],
        one_minus_high * proof.link_col_evals[LINK_COL_MULT_C],
    ];
    check_leaf_tie(&link_inst, &leaf_evals, p_claim, q_claim, &r_leaf)
}

// ===========================================================================
// Standalone POSITIONAL SUCCESSOR lookup (range-free reuse of the shift above).
//
// PLAIN-ENGLISH: the sort gadget above proves TWO things at once — (1) the SHIFT
// relation `U_shift[i] = U[i+1]` (the index-augmented LogUp), and (2) that the
// consecutive differences `U[i+1] - U[i]` are positive (the RANGE / booleanity +
// recomposition claims), which is what makes the column strictly sorted. For an
// ordered `rdf:List` walk we want ONLY piece (1): "column `next` is the one-step
// shift of column `base`". The list's cons-cell ids are FRESH RANDOM opaque nodes,
// so they are NOT monotonic — piece (2) would (correctly) reject them. Crucially,
// piece (1) never compares the ids as MAGNITUDES: the distinctness that pins each
// position comes entirely from the PUBLIC index MLE (`Operand::Idx`), and the ids
// themselves are only ever equality-folded through `beta`. So the shift binding is
// sound on arbitrary (unordered) node-id columns, and we expose it here WITHOUT the
// range claims.
//
// SOUNDNESS (why dropping the range check is safe): the check is the single LogUp
// equality `Sum_{i<n-1} rec_d[i] = Sum_i mult[i]*rec_c[i]` with
// `c_fold[i] = idx[i] + beta*base[i]` and `d_fold[i] = (idx[i]+1) + beta*next[i]`.
// Because `idx[i] = i` is distinct across the whole domain, every `c_fold[i]` is a
// DISTINCT pole; matching a `d_fold[i]` to a `c_fold[j]` forces the integer parts
// `i+1 = j` AND `next[i] = base[j] = base[i+1]` (a stray match would need beta to
// hit one exact value, negligible after commitment). The committed multiplicity
// mask `mult` is thereby forced to the honest `1 on [1,n)` — no separate booleanity
// proof on `mult` is needed, exactly as in the sort gadget. The gadget pins
// `next[i] = base[i+1]` for every `i` in `[0, n-1)`; it deliberately says nothing
// about the LAST real row `next[n-1]` or padding (the caller pins the tail, e.g.
// `next[n-1] = rdf:nil`, separately).

const LABEL_SUCC_BETA: &[u8] = b"pow/successor/fingerprint-beta";
const LABEL_SUCC_ALPHA: &[u8] = b"pow/successor/reciprocal-alpha";

/// A positional-successor proof. The range-free reuse of the sort SHIFT link: it proves
/// `next` is the index-augmented one-step shift of `base` with committed multiplicity
/// `mult`, via a single LogUp-GKR fraction tree over an UNCOMMITTED block-disjoint layout
/// (`sort_link_instance`) — the SAME construction the migrated [`SortedProof`] uses, minus
/// the RANGE (booleanity + recomposition) sub-proof (a successor walk over opaque node ids
/// is NOT ordered, so there is no range piece at all). This REPLACES the two committed
/// reciprocal columns (`rec_c`/`rec_d`) and the multiset-link sumcheck; no reciprocal is
/// committed anywhere. `base`/`next` are passed in by the caller, so their commitments are
/// NOT stored here.
///
/// `allow(dead_code)`: consumed by the ordered `rdf:List` walk composite (ReadDocument
/// increment 1) and by this module's tests; the standalone primitive lands first.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct SuccessorProof {
    /// Commitment to the shift-link multiplicity column `mult` (honest mask `1` on `[1, n)`).
    pub mult_commitment: ArkGT,
    /// GKR fraction tree for the shift LogUp
    /// `Σ [i<n-1]·(α−d_fold)⁻¹ − Σ mult·(α−c_fold)⁻¹ = 0` (the exact reciprocal-era link).
    pub link: GkrFractionProof,
    /// Evaluations of `[base, next, mult]` at `r_low` (the low `k` bits of the GKR leaf
    /// point), bound to their commitments by `link_opening`.
    pub link_col_evals: Vec<ArkFr>,
    /// Batched Dory opening authenticating `link_col_evals` at the leaf point.
    pub link_opening: ColumnOpening,
}

/// The honest multiplicity mask: `1` on `[1, n)`, `0` elsewhere — the c-side rows
/// each `d_fold[i]` (i in `[0, n-1)`) must match after the `+1` index offset.
#[allow(dead_code)]
fn successor_mult(n: usize, n_domain: usize) -> Vec<ArkFr> {
    let mut mult = vec![f0(); n_domain];
    for slot in mult.iter_mut().take(n).skip(1) {
        *slot = f1();
    }
    mult
}

/// Prove that committed column `next` is the one-step positional shift of committed
/// column `base` over the real prefix: `next[i] = base[i+1]` for every `i` in
/// `[0, n-1)`. `com_base`/`com_next` must commit the two columns padded to a shared
/// power-of-two domain (`base.len() == next.len()`, both `>= n`). NO ordering or
/// magnitude relation between the entries is assumed or proven — this is the
/// range-free successor binding used by the ordered `rdf:List` walk.
#[allow(dead_code)]
pub(crate) fn prove_successor_lookup_with_context(
    context: &StoreProverContext<'_>,
    base: &[ArkFr],
    next: &[ArkFr],
    n: usize,
    com_base: &ArkGT,
    com_next: &ArkGT,
    transcript: &mut Transcript254,
) -> io::Result<SuccessorProof> {
    let n_domain = column_domain(base.len(), n)?;
    let mult = successor_mult(n, n_domain);
    prove_successor_with_witness_and_context(
        context, base, next, &mult, n, com_base, com_next, transcript,
    )
}

/// As [`prove_successor_lookup_with_context`], but with an EXPLICIT multiplicity
/// witness. The honest entry point above supplies the `1-on-[1,n)` mask; tests use
/// this to inject a FORGED `mult` and confirm the shift binding still rejects — the
/// soundness comes from the public-index pole distinctness, NOT from any separate
/// booleanity check on `mult`.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(crate) fn prove_successor_with_witness_and_context(
    context: &StoreProverContext<'_>,
    base: &[ArkFr],
    next: &[ArkFr],
    mult: &[ArkFr],
    n: usize,
    com_base: &ArkGT,
    com_next: &ArkGT,
    transcript: &mut Transcript254,
) -> io::Result<SuccessorProof> {
    let n_domain = column_domain(base.len(), n)?;
    if base.len() > n_domain || next.len() != n_domain || n > n_domain || mult.len() != n_domain {
        return Err(invalid("successor column length mismatch"));
    }
    let k = n_domain.trailing_zeros() as usize;
    let base_p = padded(base, n_domain);

    // Commit the multiplicity mask (the c-side numerator), bind base/next/mult, then draw
    // beta/alpha AFTER every data-column commitment (the Inc 0 caller contract: the GKR
    // leaves must be fixed before any per-layer challenge). NO reciprocal column is
    // committed — the shift LogUp is a fraction tree over UNCOMMITTED intermediates.
    let mult = mult.to_vec();
    let mult_commitment = context.commit_column(&mult)?;
    transcript.append_group(b"pow/successor/com-base", com_base);
    transcript.append_group(b"pow/successor/com-next", com_next);
    transcript.append_group(b"pow/successor/com-mult", &mult_commitment);
    let beta = transcript.challenge_scalar(LABEL_SUCC_BETA);
    let alpha = transcript.challenge_scalar(LABEL_SUCC_ALPHA);

    // (SHIFT LINK) the index-augmented multiset link as ONE GKR fraction tree over the
    // block-disjoint 2*domain layout — the SAME `sort_link_instance` the sort gadget uses:
    // the c-side (`base`) fills the low half, the d-side (`next`) the high half, and `mult`
    // (low) is the committed c-side numerator. The prover feeds the tree the columns'
    // block-disjoint zero-extensions; the committed columns stay length-`domain`.
    let leaf_count = 2 * n_domain;
    let mut base_ext = vec![f0(); leaf_count];
    base_ext[..n_domain].copy_from_slice(&base_p);
    let mut next_ext = vec![f0(); leaf_count];
    next_ext[n_domain..].copy_from_slice(next);
    let mut mult_ext = vec![f0(); leaf_count];
    mult_ext[..n_domain].copy_from_slice(&mult);
    let extended = vec![base_ext, next_ext, mult_ext];

    let link_inst = sort_link_instance(alpha, beta, n, k);
    let (link, r_leaf) = prove_fraction_sum_zero(&link_inst, &extended, transcript)?;

    // Single batched Dory opening of the length-`domain` link columns [base, next, mult]
    // at r_low, bound to their commitments on this transcript.
    let r_low = &r_leaf[..k];
    let link_columns = vec![base_p, next.to_vec(), mult];
    let link_col_evals: Vec<ArkFr> =
        link_columns.iter().map(|c| mle_full_eval(c, r_low)).collect();
    let link_opening = context.prove_column_openings(&link_columns, r_low, transcript)?;

    Ok(SuccessorProof {
        mult_commitment,
        link,
        link_col_evals,
        link_opening,
    })
}

/// Verify a positional-successor proof: `com_next`'s column is the one-step shift of
/// `com_base`'s column over `[0, n-1)`.
#[allow(dead_code)]
pub(crate) fn verify_successor_lookup(
    com_base: &ArkGT,
    com_next: &ArkGT,
    n: usize,
    proof: &SuccessorProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    // With the shift LogUp now a GKR fraction tree (and NO range sumcheck for the
    // range-free successor), the variable count comes from the link's layering: the leaf
    // vector is `2^{k+1}` (block-disjoint c/d halves), so `link.num_vars() == k + 1`. The
    // committed columns' actual length pins `k` structurally (the r_low opening below fails
    // if the prover lies), exactly as `com_u` pins it in `SortedProof`.
    let layers = proof.link.num_vars();
    if layers < 2 || layers > usize::BITS as usize {
        return Err(invalid("successor proof has an implausible variable count"));
    }
    let k = layers - 1;
    if n > (1usize << k) {
        return Err(invalid("real length exceeds the successor domain"));
    }
    if proof.link_col_evals.len() != 3 {
        return Err(invalid("successor shift-link column-evaluation count mismatch"));
    }
    transcript.append_group(b"pow/successor/com-base", com_base);
    transcript.append_group(b"pow/successor/com-next", com_next);
    transcript.append_group(b"pow/successor/com-mult", &proof.mult_commitment);
    let beta = transcript.challenge_scalar(LABEL_SUCC_BETA);
    let alpha = transcript.challenge_scalar(LABEL_SUCC_ALPHA);

    // (SHIFT LINK) bind the public block-disjoint instance and verify the GKR tree.
    let link_inst = sort_link_instance(alpha, beta, n, k);
    let (r_leaf, p_claim, q_claim) =
        bind_and_verify_fraction_tree(&link_inst, &proof.link, transcript)?;

    // Open the length-`domain` link columns [base, next, mult] at r_low, bound to their
    // commitments. `mult` is a PLAIN committed numerator (NO Chi mask); its exact value,
    // zero-on-padding, and the genuine positional bijection are forced by the link's
    // DISTINCT Idx-augmented c_fold poles — no co-located sub-proof or GAP-1 Identity.
    let r_low = &r_leaf[..k];
    let r_high = r_leaf[k];
    let link_coms = vec![*com_base, *com_next, proof.mult_commitment];
    verify_column_openings(&link_coms, r_low, &proof.link_col_evals, &proof.link_opening, transcript)?;

    // Reconstruct each 2*domain leaf-MLE from the length-`domain` opening via the top-bit
    // zero-extension factor: `base` + `mult` live in the LOW half (factor 1 − r_high),
    // `next` in the HIGH half (factor r_high) — structurally forcing each side's columns to
    // zero on the other block (no cross-block leak).
    let one_minus_high = f1() - r_high;
    let leaf_evals = vec![
        one_minus_high * proof.link_col_evals[LINK_COL_U],
        r_high * proof.link_col_evals[LINK_COL_U_SHIFT],
        one_minus_high * proof.link_col_evals[LINK_COL_MULT_C],
    ];
    check_leaf_tie(&link_inst, &leaf_evals, p_claim, q_claim, &r_leaf)
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

    // ---- Shift LogUp LINK (GKR) soundness negatives -----------------------
    //
    // The RANGE sub-proof (booleanity + recomposition) proves ONLY that the committed
    // consecutive gaps are positive; it never touches `mult_c` and never checks that
    // `U_shift` is the true one-step shift of `U`. That binding is the GKR shift link.
    // These forges keep RANGE honest (bits recompose to the committed `U_shift - U - 1`)
    // so rejection must come from the new fraction tree, not the sumcheck.

    #[test]
    fn forged_shift_multiplicity_is_rejected_via_link() {
        // `mult_c` is a PLAIN committed numerator; its exact value / zero-on-padding are
        // forced ONLY by the link's distinct Idx-augmented c_fold poles. Feed honest
        // `U_shift` + honest bits (RANGE passes) and a spread of adversarial multiplicity
        // masks — an unmatched c-side pole (extra / wrong / padding-row `mult_c`) or a
        // dropped d-side match (all-zero) makes the fraction sum nonzero, so the tree
        // root `P != 0` rejects.
        let u = frs(&[2, 9, 20]); // n=3 -> domain 4, a padding row at index 3
        let n = u.len();
        let n_domain = sort_domain(n).unwrap();
        let com_u = commit_u(&u, n);
        let (u_shift, honest_mult, honest_bits) =
            honest_witness(&u, n, n_domain, BITS, SortMode::Strict);
        // Sanity: the honest multiplicity verifies.
        let ok = prove_strictly_sorted_with_witness(
            &u, n, &com_u, &u_shift, &honest_mult, &honest_bits, &mut transcript(),
        )
        .unwrap();
        assert!(verify_strictly_sorted(&com_u, n, &ok, &mut transcript()).is_ok());

        let mut extra_zero_row = honest_mult.clone();
        extra_zero_row[0] = f1(); // c_fold[0] pole with no d-side match
        let mut padding_pole = honest_mult.clone();
        padding_pole[n_domain - 1] = f1(); // a pole on the padding row [n, domain)
        for forged in [
            vec![f1(); n_domain],  // all ones: several unmatched c-side poles
            vec![f0(); n_domain],  // all zero: both d-side rows unmatched (drop)
            extra_zero_row,
            padding_pole,
        ] {
            let proof = prove_strictly_sorted_with_witness(
                &u, n, &com_u, &u_shift, &forged, &honest_bits, &mut transcript(),
            )
            .unwrap();
            assert!(
                verify_strictly_sorted(&com_u, n, &proof, &mut transcript()).is_err(),
                "forged multiplicity must be rejected by the shift link"
            );
        }
    }

    #[test]
    fn forged_shift_next_is_rejected_via_link() {
        // A `U_shift` that keeps every consecutive gap POSITIVE (so RANGE accepts, with
        // bits recomposing to the committed forged `U_shift - U - 1`) but is NOT the true
        // one-step shift of `U`: only the index-augmented GKR shift link can reject it,
        // because the range gadget proves the gaps positive, NOT `U_shift[i] = U[i+1]`.
        let u = frs(&[1, 10, 100, 1000]);
        let n = u.len(); // n=4 == domain, no padding
        let n_domain = sort_domain(n).unwrap();
        let com_u = commit_u(&u, n);
        let (_honest_shift, honest_mult, _) =
            honest_witness(&u, n, n_domain, BITS, SortMode::Strict);
        // True shift is [10, 100, 1000, 0]; corrupt the middle to 50 — still a positive
        // gap (50 - 10 = 40) so RANGE passes, but 50 != U[2] = 100 so the shift is wrong.
        let forged_shift = frs(&[10, 50, 1000, 0]);
        let mut forged_bits = vec![vec![f0(); n_domain]; BITS];
        for i in 0..n - 1 {
            let value_bits = low_bits(forged_shift[i] - u[i] - f1(), BITS);
            for (j, slot) in forged_bits.iter_mut().enumerate() {
                slot[i] = value_bits[j];
            }
        }
        let proof = prove_strictly_sorted_with_witness(
            &u, n, &com_u, &forged_shift, &honest_mult, &forged_bits, &mut transcript(),
        )
        .unwrap();
        assert!(
            verify_strictly_sorted(&com_u, n, &proof, &mut transcript()).is_err(),
            "a non-shift U_shift with positive gaps must be rejected by the shift link"
        );
    }

    #[test]
    fn fractional_mult_c_is_rejected_via_link() {
        // Belt-and-suspenders for the audit finding that `mult_c` needs NO booleanity/range
        // proof. The c_fold poles are pairwise-DISTINCT (the `Idx` ramp; `beta` post-commit),
        // so `Σ p/q = 0` forces the residue at each c_fold pole to vanish, which UNIQUELY
        // pins `mult_c[i]` to the exact non-negative INTEGER count of d-hits at that pole (1
        // on `[1,n)`, 0 elsewhere). Any value that is not that integer — INCLUDING a genuine
        // field FRACTION like 1/2, which no integer count can equal — leaves an uncancellable
        // residue, so the tree root `P != 0` and verify rejects. Everything else honest, so
        // rejection can ONLY come from the shift link.
        let u = frs(&[2, 9, 20]); // n=3 -> domain 4; honest mult_c is [0,1,1,0]
        let n = u.len();
        let n_domain = sort_domain(n).unwrap();
        let com_u = commit_u(&u, n);
        let (u_shift, honest_mult, honest_bits) =
            honest_witness(&u, n, n_domain, BITS, SortMode::Strict);
        // Contrast: the honest integer multiplicity verifies.
        let ok = prove_strictly_sorted_with_witness(
            &u, n, &com_u, &u_shift, &honest_mult, &honest_bits, &mut transcript(),
        )
        .unwrap();
        assert!(verify_strictly_sorted(&com_u, n, &ok, &mut transcript()).is_ok());

        // Corrupt a REAL c_fold pole (rows 1 and 2, whose honest count is exactly 1) with a
        // field fraction (1/2) and an arbitrary non-count integer (7). Neither is the true
        // count, so each leaves a nonzero residue at its distinct pole.
        let half = ArkFr::from_u64(2).inv().unwrap(); // 1/2: a genuine, non-integer field element
        assert!(half + half == f1());
        for (row, bad) in [(1usize, half), (2usize, fr(7))] {
            let mut forged = honest_mult.clone();
            forged[row] = bad;
            let proof = prove_strictly_sorted_with_witness(
                &u, n, &com_u, &u_shift, &forged, &honest_bits, &mut transcript(),
            )
            .unwrap();
            assert!(
                verify_strictly_sorted(&com_u, n, &proof, &mut transcript()).is_err(),
                "a non-integer / non-count mult_c on a real pole must be rejected by the shift link"
            );
        }
    }

    // ---- Positional successor lookup (range-free shift) -------------------
    //
    // This is the CRUX check for the ordered `rdf:List` walk: the shift binding
    // must hold over cons-cell ids that are FRESH RANDOM opaque nodes (NOT sorted),
    // which the strict/nondecreasing sort verifiers would (correctly) reject.

    fn succ_transcript() -> Transcript254 {
        Transcript254::new(b"parture-successor-test")
    }

    /// Commit `base`/`next` (padded to a shared power-of-two domain) and run the
    /// successor prove/verify round trip with the honest multiplicity mask.
    fn run_successor(base: &[ArkFr], next: &[ArkFr], n: usize) -> io::Result<()> {
        let n_domain = column_domain(base.len().max(next.len()), n).unwrap();
        let base_p = padded(base, n_domain);
        let next_p = padded(next, n_domain);
        let com_base = commit_column(&base_p).unwrap();
        let com_next = commit_column(&next_p).unwrap();
        let proof = prove_successor_lookup_with_context(
            &StoreProverContext::cpu(),
            &base_p,
            &next_p,
            n,
            &com_base,
            &com_next,
            &mut succ_transcript(),
        )?;
        verify_successor_lookup(&com_base, &com_next, n, &proof, &mut succ_transcript())
    }

    #[test]
    fn successor_binds_shift_over_nonmonotonic_node_ids() {
        // A cons chain's cons-cell ids as they appear IN LIST ORDER: deliberately
        // NON-monotonic (up, down, up, down) like fresh random dictionary nodes.
        // `next` is the one-step shift; the LAST real row of `next` is the tail
        // sentinel (here 0 = a stand-in for rdf:nil), which the gadget does NOT
        // constrain — the composite pins it separately.
        let base = frs(&[500, 12, 9999, 3, 77]);
        let next = frs(&[12, 9999, 3, 77, 0]);
        assert!(run_successor(&base, &next, 5).is_ok());

        // A two-cell list and a padded-domain (n=3 -> domain 4) list also bind.
        assert!(run_successor(&frs(&[41, 8]), &frs(&[8, 0]), 2).is_ok());
        assert!(run_successor(&frs(&[9, 4, 6]), &frs(&[4, 6, 0]), 3).is_ok());
    }

    #[test]
    fn successor_rejects_reordered_next() {
        // `next` is a PERMUTATION of the true shift but out of position: the
        // index-augmented lookup must reject (it binds POSITION, not just the set).
        let base = frs(&[500, 12, 9999, 3]);
        let reordered = frs(&[9999, 12, 3, 0]); // 12 and 9999 swapped vs the true shift
        assert!(run_successor(&base, &reordered, 4).is_err());
    }

    #[test]
    fn successor_rejects_wrong_successor_value() {
        // A single successor entry replaced by a fabricated id (an injected cons
        // cell that is not the real rest): must reject.
        let base = frs(&[500, 12, 9999, 3]);
        let wrong = frs(&[12, 424242, 3, 0]); // next[1] should be 9999
        assert!(run_successor(&base, &wrong, 4).is_err());
    }

    #[test]
    fn successor_rejects_forged_multiplicity() {
        // The soundness argument claims `mult` needs NO separate booleanity proof —
        // the public-index pole distinctness forces it to the honest mask. Prove
        // that: feed a WRONG `next` together with an adversarial `mult` and confirm
        // no forged multiplicity can rescue a non-shift `next`.
        let n = 4usize;
        let n_domain = column_domain(4, n).unwrap();
        let base = padded(&frs(&[500, 12, 9999, 3]), n_domain);
        let wrong_next = padded(&frs(&[12, 424242, 3, 0]), n_domain);
        let com_base = commit_column(&base).unwrap();
        let com_next = commit_column(&wrong_next).unwrap();
        // Try a spread of hand-picked forged multiplicity masks; none may verify.
        for mult in [
            vec![f1(); n_domain],
            vec![f0(), f1(), f0(), f1()],
            vec![fr(2), f1(), f1(), f0()],
            vec![f0(), f0(), f1(), f1()],
        ] {
            let proof = prove_successor_with_witness_and_context(
                &StoreProverContext::cpu(),
                &base,
                &wrong_next,
                &mult,
                n,
                &com_base,
                &com_next,
                &mut succ_transcript(),
            )
            .unwrap();
            assert!(
                verify_successor_lookup(&com_base, &com_next, n, &proof, &mut succ_transcript())
                    .is_err()
            );
        }
    }

    #[test]
    fn successor_rejects_statement_substitution() {
        // A valid proof for one (base, next) pair must not verify against a
        // different committed `next` (transcript binds both commitments).
        let base = frs(&[500, 12, 9999, 3]);
        let next = frs(&[12, 9999, 3, 0]);
        let n_domain = column_domain(4, 4).unwrap();
        let base_p = padded(&base, n_domain);
        let next_p = padded(&next, n_domain);
        let com_base = commit_column(&base_p).unwrap();
        let com_next = commit_column(&next_p).unwrap();
        let proof = prove_successor_lookup_with_context(
            &StoreProverContext::cpu(),
            &base_p,
            &next_p,
            4,
            &com_base,
            &com_next,
            &mut succ_transcript(),
        )
        .unwrap();
        let other_next = padded(&frs(&[12, 9999, 999, 0]), n_domain);
        let com_other = commit_column(&other_next).unwrap();
        assert!(
            verify_successor_lookup(&com_base, &com_other, 4, &proof, &mut succ_transcript())
                .is_err()
        );
    }
}
