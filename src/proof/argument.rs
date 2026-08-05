//! Clean-room multilinear sumcheck + Dory evaluation glue.
//!
//! This is the foundation of the Proof-of-WOQL query argument. The sumcheck is a
//! hand-rolled implementation of the standard multilinear sumcheck protocol (the
//! same algorithm arkworks' `ark-linear-sumcheck` implements, MIT/Apache); no
//! COSL-licensed code was consulted or copied. The Dory glue reduces a batched
//! claim to a single random point `r` and opens the referenced committed columns
//! there via `dory-pcs` (BN254), batching multiple openings into one with the
//! commitment's homomorphism.
//!
//! Scope (Phase 2a): sumcheck + Dory glue only. No filter/join/dictionary/LogUp.
//!
//! # Point convention (load-bearing)
//! `dory-pcs` evaluates a committed vector as a multilinear extension with a
//! LSB-first variable/bit mapping (see its `multilinear_lagrange_basis`): flat
//! index `i` has bit `j` controlled by evaluation variable `j`. Because a column
//! is laid out as `2^nu` rows x `2^sigma` columns at flat index
//! `row*2^sigma + col` (see [`super::commitment::matrix_dims`]), the first
//! `sigma` variables of the point are the column bits and the last `nu` are the
//! row bits. Our sumcheck binds variables LSB-first (round `t` binds bit `t`), so
//! the challenge vector `r = (r_0, .., r_{k-1})` we hand to Dory is exactly the
//! point Dory expects. [`mle_full_eval`] uses the same LSB-first convention, and
//! the tests assert `mle_full_eval(col, r) == ArkworksPolynomial::evaluate(col, r)`
//! and that swapping the `[col-bits | row-bits]` halves makes verification FAIL.

use std::io;

use dory_pcs::backends::arkworks::{
    ArkFr, ArkG1, ArkG2, ArkGT, ArkworksPolynomial, Blake2bTranscript, G1Routines, G2Routines,
    BN254,
};
use dory_pcs::primitives::arithmetic::{Field, Group};
use dory_pcs::primitives::poly::Polynomial;
use dory_pcs::primitives::transcript::Transcript;
use dory_pcs::{prove, verify, DoryProof, Transparent};

use super::commitment::{matrix_dims, prover_setup, verifier_setup};
use super::invalid;

/// The concrete Fiat-Shamir transcript shared by our sumcheck and Dory's
/// prove/verify.
pub type Transcript254 = Blake2bTranscript<BN254>;

/// A Dory single-point opening proof over BN254.
pub type ColumnOpening = DoryProof<ArkG1, ArkG2, ArkGT>;

const LABEL_DOMAIN: &[u8] = b"pow/sumcheck/domain";
const LABEL_SHAPE: &[u8] = b"pow/sumcheck/shape";
const LABEL_ZEROCHECK: &[u8] = b"pow/sumcheck/zerocheck-point";
const LABEL_RHO: &[u8] = b"pow/sumcheck/batch-rho";
const LABEL_ROUND_POLY: &[u8] = b"pow/sumcheck/round-poly";
const LABEL_ROUND_CHALLENGE: &[u8] = b"pow/sumcheck/round-challenge";
const LABEL_DORY_GAMMA: &[u8] = b"pow/dory/batch-gamma";

#[inline]
fn f0() -> ArkFr {
    ArkFr::zero()
}
#[inline]
fn f1() -> ArkFr {
    ArkFr::one()
}

/// The two claim shapes the query argument reduces to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClaimShape {
    /// Prove `Sum_{x in {0,1}^k} G(x) = 0` for the claim's expression `G`.
    ZeroSum,
    /// Prove `G(x) = 0` for every `x` on the hypercube, via the standard
    /// zero-check reduction: after the columns are committed, draw a Fiat-Shamir
    /// point `r_zc` and sumcheck `Sum_x eq(r_zc, x) * G(x) = 0`.
    Identity,
}

/// A single multilinear operand in a claim expression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operand {
    /// The MLE of committed column `j` (index into the shared column list).
    Column(usize),
    /// The MLE of the PUBLIC `chi_n` indicator of `[0, n)` (verifier-derivable;
    /// never committed). `n` is carried by [`SumcheckShape::chi_n`].
    ChiN,
}

/// One term of a claim expression: a public constant `coeff` times a product of
/// multilinear operands. An empty `operands` list is a pure constant term.
#[derive(Clone, Debug)]
pub struct Term {
    pub coeff: ArkFr,
    pub operands: Vec<Operand>,
}

/// A claim expression `G(x) = sum_t coeff_t * prod(operands_t)(x)`. This small
/// form is enough for products of committed columns (the 2a claims) as well as
/// the filter claims, which mix committed columns, public constants, and the
/// public `chi_n` MLE with subtraction.
#[derive(Clone, Debug)]
pub struct Expr {
    pub terms: Vec<Term>,
}

impl Expr {
    /// A product of committed columns with coefficient 1 (the 2a claim form).
    pub fn product(factors: &[usize]) -> Self {
        Expr {
            terms: vec![Term {
                coeff: f1(),
                operands: factors.iter().map(|&j| Operand::Column(j)).collect(),
            }],
        }
    }

    fn max_arity(&self) -> usize {
        self.terms
            .iter()
            .map(|term| term.operands.len())
            .max()
            .unwrap_or(0)
    }

    fn uses_chi(&self) -> bool {
        self.terms
            .iter()
            .any(|term| term.operands.iter().any(|op| *op == Operand::ChiN))
    }
}

/// One claim: a shape plus the expression `G` it constrains.
#[derive(Clone, Debug)]
pub struct SumcheckClaim {
    pub shape: ClaimShape,
    pub expr: Expr,
}

impl SumcheckClaim {
    /// Convenience: a claim whose `G` is a product of committed columns.
    pub fn product(shape: ClaimShape, factors: &[usize]) -> Self {
        SumcheckClaim {
            shape,
            expr: Expr::product(factors),
        }
    }
}

/// The public description of a batched sumcheck instance (shared by prover and
/// verifier; contains no witness data). `chi_n` supplies the public real-row
/// count `n` for any [`Operand::ChiN`] used by the claims.
#[derive(Clone, Debug)]
pub struct SumcheckShape {
    pub num_vars: usize,
    pub num_columns: usize,
    pub chi_n: Option<usize>,
    pub claims: Vec<SumcheckClaim>,
}

/// A batched sumcheck proof: one univariate per round (as evaluations at
/// `0..=degree`) plus the claimed evaluation of every column at the final point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckProof {
    pub round_polys: Vec<Vec<ArkFr>>,
    pub claimed_evals: Vec<ArkFr>,
}

impl SumcheckShape {
    fn validate(&self) -> io::Result<()> {
        if self.num_vars == 0 {
            return Err(invalid("sumcheck requires at least one variable"));
        }
        if self.claims.is_empty() {
            return Err(invalid("sumcheck requires at least one claim"));
        }
        for claim in &self.claims {
            if claim.expr.terms.is_empty() {
                return Err(invalid("sumcheck claim needs at least one term"));
            }
            for term in &claim.expr.terms {
                for operand in &term.operands {
                    match operand {
                        Operand::Column(c) if *c >= self.num_columns => {
                            return Err(invalid("sumcheck claim references an unknown column"));
                        }
                        Operand::ChiN if self.chi_n.is_none() => {
                            return Err(invalid("claim uses chi_n but no real-row count is set"));
                        }
                        _ => {}
                    }
                }
            }
        }
        if let Some(n) = self.chi_n {
            if n > (1usize << self.num_vars) {
                return Err(invalid("chi_n real-row count exceeds the hypercube size"));
            }
        }
        Ok(())
    }

    fn has_identity(&self) -> bool {
        self.claims
            .iter()
            .any(|claim| claim.shape == ClaimShape::Identity)
    }

    fn uses_chi(&self) -> bool {
        self.claims.iter().any(|claim| claim.expr.uses_chi())
    }

    /// Per-round univariate degree: the largest term arity across claims,
    /// counting the extra `eq` factor that identity (zero-check) claims carry.
    fn round_degree(&self) -> usize {
        self.claims
            .iter()
            .map(|claim| claim.expr.max_arity() + usize::from(claim.shape == ClaimShape::Identity))
            .max()
            .unwrap_or(1)
            .max(1)
    }
}

// ---------------------------------------------------------------------------
// Multilinear helpers (all LSB-first: variable j controls bit j of the index)
// ---------------------------------------------------------------------------

/// `table[i] = prod_j ( bit_j(i) ? point[j] : 1 - point[j] )`, i.e. the
/// multilinear Lagrange basis at `point`, length `2^point.len()`.
fn eq_table(point: &[ArkFr]) -> Vec<ArkFr> {
    let mut table = vec![f1(); 1usize << point.len()];
    let mut size = 1usize;
    for &p in point {
        let one_minus = f1() - p;
        for i in 0..size {
            let v = table[i];
            table[i] = v * one_minus;
            table[i + size] = v * p;
        }
        size <<= 1;
    }
    table
}

/// Multilinear extension of `evals` (length `2^k`) at `point` (length `k`),
/// using the same LSB-first convention as `dory-pcs`.
pub fn mle_full_eval(evals: &[ArkFr], point: &[ArkFr]) -> ArkFr {
    let basis = eq_table(point);
    evals
        .iter()
        .zip(basis.iter())
        .fold(f0(), |acc, (e, b)| acc + *e * *b)
}

/// `eq(a, b) = prod_j ( a_j b_j + (1-a_j)(1-b_j) )`.
fn eq_eval(a: &[ArkFr], b: &[ArkFr]) -> ArkFr {
    a.iter().zip(b.iter()).fold(f1(), |acc, (x, y)| {
        acc * (*x * *y + (f1() - *x) * (f1() - *y))
    })
}

/// The prover's `chi_n` table: `[1; n] ++ [0; 2^k - n]` (the indicator of the
/// real rows `[0, n)`), whose multilinear extension the verifier reproduces
/// closed-form via [`chi_n_eval`].
fn chi_n_table(k: usize, n: usize) -> Vec<ArkFr> {
    let size = 1usize << k;
    let mut table = vec![f0(); size];
    for slot in table.iter_mut().take(n.min(size)) {
        *slot = f1();
    }
    table
}

/// Closed-form multilinear extension of the PUBLIC `chi_n` indicator of `[0, n)`
/// at `point` (LSB-first). Handles `n` that is NOT a power of two, and the
/// extremes `n = 0` (constant 0) and `n = 2^k` (constant 1). Standard less-than
/// indicator MLE: walk the bits of `n` from MSB to LSB; each set bit `j`
/// contributes the dyadic block of indices whose bit `j` is 0 (with the higher
/// bits fixed to those of `n`), which are all `< n`.
pub fn chi_n_eval(point: &[ArkFr], n: usize) -> ArkFr {
    let k = point.len();
    let size = 1usize << k;
    if n == 0 {
        return f0();
    }
    if n >= size {
        return f1();
    }
    let mut result = f0();
    let mut prefix = f1();
    for j in (0..k).rev() {
        let rj = point[j];
        if (n >> j) & 1 == 1 {
            result = result + prefix * (f1() - rj);
            prefix = prefix * rj;
        } else {
            prefix = prefix * (f1() - rj);
        }
    }
    result
}

/// Fold a table in place at its current lowest variable: `new[i] =
/// t[2i]*(1-r) + t[2i+1]*r`, halving the length.
fn fold_table(table: &mut Vec<ArkFr>, r: ArkFr) {
    let half = table.len() / 2;
    let one_minus = f1() - r;
    for i in 0..half {
        table[i] = table[2 * i] * one_minus + table[2 * i + 1] * r;
    }
    table.truncate(half);
}

/// `[base^0, base^1, .., base^{count-1}]`.
fn powers(base: ArkFr, count: usize) -> Vec<ArkFr> {
    let mut out = Vec::with_capacity(count);
    let mut cur = f1();
    for _ in 0..count {
        out.push(cur);
        cur = cur * base;
    }
    out
}

/// Interpolate a univariate given its evaluations at `0, 1, .., evals.len()-1`
/// and evaluate it at `x` (Lagrange; the integer points are distinct in `Fr`).
fn interpolate(evals: &[ArkFr], x: ArkFr) -> io::Result<ArkFr> {
    let degree = evals.len() - 1;
    let mut result = f0();
    for i in 0..=degree {
        let xi = ArkFr::from_u64(i as u64);
        let mut num = f1();
        let mut den = f1();
        for m in 0..=degree {
            if m == i {
                continue;
            }
            let xm = ArkFr::from_u64(m as u64);
            num = num * (x - xm);
            den = den * (xi - xm);
        }
        let inv = den
            .inv()
            .ok_or_else(|| invalid("sumcheck interpolation basis is singular"))?;
        result = result + evals[i] * (num * inv);
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Fiat-Shamir preamble (identical on both sides)
// ---------------------------------------------------------------------------

fn append_shape(transcript: &mut Transcript254, shape: &SumcheckShape) {
    transcript.append_bytes(LABEL_DOMAIN, b"parture/proof-of-woql/sumcheck/v0");
    transcript.append_bytes(LABEL_SHAPE, &(shape.num_vars as u64).to_le_bytes());
    transcript.append_bytes(LABEL_SHAPE, &(shape.num_columns as u64).to_le_bytes());
    transcript.append_bytes(LABEL_SHAPE, &(shape.chi_n.unwrap_or(usize::MAX) as u64).to_le_bytes());
    transcript.append_bytes(LABEL_SHAPE, &(shape.claims.len() as u64).to_le_bytes());
    for claim in &shape.claims {
        let tag: u8 = match claim.shape {
            ClaimShape::ZeroSum => 0,
            ClaimShape::Identity => 1,
        };
        transcript.append_bytes(LABEL_SHAPE, &[tag]);
        transcript.append_bytes(LABEL_SHAPE, &(claim.expr.terms.len() as u64).to_le_bytes());
        for term in &claim.expr.terms {
            transcript.append_field(LABEL_SHAPE, &term.coeff);
            transcript.append_bytes(LABEL_SHAPE, &(term.operands.len() as u64).to_le_bytes());
            for operand in &term.operands {
                match operand {
                    Operand::Column(c) => {
                        transcript.append_bytes(LABEL_SHAPE, &[0u8]);
                        transcript.append_bytes(LABEL_SHAPE, &(*c as u64).to_le_bytes());
                    }
                    Operand::ChiN => transcript.append_bytes(LABEL_SHAPE, &[1u8]),
                }
            }
        }
    }
}

/// The preamble that both sides run AFTER the caller has appended the column
/// commitments to the transcript: bind the shape (domain tag), then draw the
/// zero-check point `r_zc` (only if some claim is an identity claim), then the
/// batching challenge `rho`. Returns `(r_zc, rho_powers)`.
fn draw_preamble(transcript: &mut Transcript254, shape: &SumcheckShape) -> (Vec<ArkFr>, Vec<ArkFr>) {
    append_shape(transcript, shape);
    let r_zc = if shape.has_identity() {
        (0..shape.num_vars)
            .map(|_| transcript.challenge_scalar(LABEL_ZEROCHECK))
            .collect()
    } else {
        Vec::new()
    };
    let rho = transcript.challenge_scalar(LABEL_RHO);
    (r_zc, powers(rho, shape.claims.len()))
}

// ---------------------------------------------------------------------------
// Sumcheck prover / verifier
// ---------------------------------------------------------------------------

/// Prove a batched sumcheck over `columns` (each an MLE table of length `2^k`).
///
/// The caller MUST have appended the column commitments to `transcript` first
/// (so the zero-check point is bound to them). Returns the proof, the final
/// challenge point `r` (length `k`), and each column's claimed evaluation at `r`.
pub fn prove_sumcheck(
    shape: &SumcheckShape,
    columns: &[Vec<ArkFr>],
    transcript: &mut Transcript254,
) -> io::Result<(SumcheckProof, Vec<ArkFr>, Vec<ArkFr>)> {
    shape.validate()?;
    let k = shape.num_vars;
    let size = 1usize << k;
    if columns.len() != shape.num_columns {
        return Err(invalid("column count does not match sumcheck shape"));
    }
    if columns.iter().any(|c| c.len() != size) {
        return Err(invalid("column length does not match 2^num_vars"));
    }

    let (r_zc, rhos) = draw_preamble(transcript, shape);
    let degree = shape.round_degree();

    let mut tables: Vec<Vec<ArkFr>> = columns.to_vec();
    let mut eq = if shape.has_identity() {
        eq_table(&r_zc)
    } else {
        Vec::new()
    };
    let mut chi = if shape.uses_chi() {
        chi_n_table(k, shape.chi_n.expect("chi_n present when used"))
    } else {
        Vec::new()
    };

    let mut round_polys = Vec::with_capacity(k);
    let mut r = Vec::with_capacity(k);
    for _round in 0..k {
        let half = tables[0].len() / 2;
        let mut poly = vec![f0(); degree + 1];
        for (x, slot) in poly.iter_mut().enumerate() {
            let x_field = ArkFr::from_u64(x as u64);
            let one_minus = f1() - x_field;
            // Fold-to-X value of a table at hypercube pair `i`.
            let at = |table: &[ArkFr], i: usize| table[2 * i] * one_minus + table[2 * i + 1] * x_field;
            let mut acc = f0();
            for i in 0..half {
                for (j, claim) in shape.claims.iter().enumerate() {
                    let eq_factor = if claim.shape == ClaimShape::Identity {
                        at(&eq, i)
                    } else {
                        f1()
                    };
                    let mut expr_val = f0();
                    for term in &claim.expr.terms {
                        let mut prod = term.coeff;
                        for operand in &term.operands {
                            prod = prod
                                * match operand {
                                    Operand::Column(c) => at(&tables[*c], i),
                                    Operand::ChiN => at(&chi, i),
                                };
                        }
                        expr_val = expr_val + prod;
                    }
                    acc = acc + rhos[j] * eq_factor * expr_val;
                }
            }
            *slot = acc;
        }

        for coeff in &poly {
            transcript.append_field(LABEL_ROUND_POLY, coeff);
        }
        let challenge = transcript.challenge_scalar(LABEL_ROUND_CHALLENGE);
        for table in tables.iter_mut() {
            fold_table(table, challenge);
        }
        if shape.has_identity() {
            fold_table(&mut eq, challenge);
        }
        if shape.uses_chi() {
            fold_table(&mut chi, challenge);
        }
        round_polys.push(poly);
        r.push(challenge);
    }

    let claimed_evals: Vec<ArkFr> = tables.iter().map(|t| t[0]).collect();
    Ok((
        SumcheckProof {
            round_polys,
            claimed_evals: claimed_evals.clone(),
        },
        r,
        claimed_evals,
    ))
}

/// Verify a batched sumcheck. Returns the reconstructed point `r` and the
/// prover's claimed column evaluations at `r` (to be checked against Dory
/// openings by the glue). The caller MUST have appended the same column
/// commitments to `transcript` first.
pub fn verify_sumcheck(
    shape: &SumcheckShape,
    proof: &SumcheckProof,
    transcript: &mut Transcript254,
) -> io::Result<(Vec<ArkFr>, Vec<ArkFr>)> {
    shape.validate()?;
    let k = shape.num_vars;
    if proof.round_polys.len() != k {
        return Err(invalid("sumcheck proof has the wrong number of rounds"));
    }
    if proof.claimed_evals.len() != shape.num_columns {
        return Err(invalid("sumcheck proof has the wrong number of claimed evals"));
    }
    let degree = shape.round_degree();

    let (r_zc, rhos) = draw_preamble(transcript, shape);

    // The batched claim starts at 0 (both shapes target a zero sum).
    let mut current = f0();
    let mut r = Vec::with_capacity(k);
    for round_poly in &proof.round_polys {
        if round_poly.len() != degree + 1 {
            return Err(invalid("sumcheck round polynomial has the wrong degree"));
        }
        if round_poly[0] + round_poly[1] != current {
            return Err(invalid("sumcheck round consistency check failed"));
        }
        for coeff in round_poly {
            transcript.append_field(LABEL_ROUND_POLY, coeff);
        }
        let challenge = transcript.challenge_scalar(LABEL_ROUND_CHALLENGE);
        current = interpolate(round_poly, challenge)?;
        r.push(challenge);
    }

    // Final check: the sumcheck's reduced value must equal the batched claim
    // rebuilt from the claimed column evaluations at r, the public chi_n(r), and
    // the public constants baked into the claim expressions.
    let chi_at_r = shape.chi_n.map(|n| chi_n_eval(&r, n));
    let mut expected = f0();
    for (j, claim) in shape.claims.iter().enumerate() {
        let eq_factor = if claim.shape == ClaimShape::Identity {
            eq_eval(&r_zc, &r)
        } else {
            f1()
        };
        let mut expr_val = f0();
        for term in &claim.expr.terms {
            let mut prod = term.coeff;
            for operand in &term.operands {
                prod = prod
                    * match operand {
                        Operand::Column(c) => proof.claimed_evals[*c],
                        Operand::ChiN => chi_at_r.expect("chi_n present when used"),
                    };
            }
            expr_val = expr_val + prod;
        }
        expected = expected + rhos[j] * eq_factor * expr_val;
    }
    if expected != current {
        return Err(invalid("sumcheck final evaluation check failed"));
    }

    Ok((r, proof.claimed_evals.clone()))
}

// ---------------------------------------------------------------------------
// Dory evaluation glue: batch column openings at r into one Dory opening
// ---------------------------------------------------------------------------

/// Commit to a raw column (length `2^k`) with the global deterministic setup,
/// returning its succinct GT (tier-2) commitment. Uses the shared matrix layout.
pub(crate) fn commit_column(column: &[ArkFr]) -> io::Result<ArkGT> {
    let (nu, sigma) = column_dims(column.len())?;
    let poly = ArkworksPolynomial::new(column.to_vec());
    let (tier_2, _rows, _blind) = poly
        .commit::<BN254, Transparent, G1Routines>(nu, sigma, prover_setup())
        .map_err(|error| invalid(format!("column commitment failed: {error}")))?;
    Ok(tier_2)
}

fn column_dims(len: usize) -> io::Result<(usize, usize)> {
    if !len.is_power_of_two() {
        return Err(invalid("column length must be a power of two"));
    }
    Ok(matrix_dims(len.trailing_zeros() as usize))
}

/// Homomorphically combine per-column commitments and claimed evaluations with
/// powers of `gamma` (the verifier's side of the batched opening).
fn combine(commitments: &[ArkGT], evals: &[ArkFr], gamma: ArkFr) -> (ArkGT, ArkFr) {
    let gammas = powers(gamma, commitments.len());
    let mut combined_commitment = <ArkGT as Group>::identity();
    let mut combined_eval = f0();
    for (j, gj) in gammas.iter().enumerate() {
        combined_commitment = combined_commitment.add(&commitments[j].scale(gj));
        combined_eval = combined_eval + evals[j] * *gj;
    }
    (combined_commitment, combined_eval)
}

/// Prover side of the batched opening: draw `gamma`, form the `gamma`-combined
/// column polynomial, and produce ONE Dory opening of it at `r`.
pub fn prove_column_openings(
    column_tables: &[Vec<ArkFr>],
    r: &[ArkFr],
    transcript: &mut Transcript254,
) -> io::Result<ColumnOpening> {
    let size = 1usize << r.len();
    if column_tables.iter().any(|c| c.len() != size) {
        return Err(invalid("column length does not match evaluation point"));
    }
    let (nu, sigma) = column_dims(size)?;
    let gamma = transcript.challenge_scalar(LABEL_DORY_GAMMA);
    let gammas = powers(gamma, column_tables.len());

    let mut combined = vec![f0(); size];
    for (table, gj) in column_tables.iter().zip(gammas.iter()) {
        for (slot, value) in combined.iter_mut().zip(table.iter()) {
            *slot = *slot + *value * *gj;
        }
    }

    let poly = ArkworksPolynomial::new(combined);
    let (_tier_2, row_commitments, blind) = poly
        .commit::<BN254, Transparent, G1Routines>(nu, sigma, prover_setup())
        .map_err(|error| invalid(format!("batched column commitment failed: {error}")))?;
    let (proof, _) = prove::<_, BN254, G1Routines, G2Routines, _, _, Transparent>(
        &poly,
        r,
        row_commitments,
        blind,
        nu,
        sigma,
        prover_setup(),
        transcript,
    )
    .map_err(|error| invalid(format!("dory opening proof failed: {error}")))?;
    Ok(proof)
}

/// Verifier side of the batched opening: draw `gamma`, homomorphically combine
/// the per-column commitments and claimed evals, and verify the single Dory
/// opening at `r`.
pub fn verify_column_openings(
    column_commitments: &[ArkGT],
    r: &[ArkFr],
    claimed_evals: &[ArkFr],
    opening: &ColumnOpening,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    if column_commitments.len() != claimed_evals.len() {
        return Err(invalid("commitment/eval count mismatch in batched opening"));
    }
    let gamma = transcript.challenge_scalar(LABEL_DORY_GAMMA);
    let (combined_commitment, combined_eval) = combine(column_commitments, claimed_evals, gamma);
    verify::<_, BN254, G1Routines, G2Routines, _>(
        combined_commitment,
        combined_eval,
        r,
        opening,
        verifier_setup().clone(),
        transcript,
    )
    .map_err(|error| invalid(format!("dory batched opening rejected: {error}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fr(v: u64) -> ArkFr {
        ArkFr::from_u64(v)
    }

    // A deterministic but non-trivial column of length 2^k.
    fn column(k: usize, seed: u64) -> Vec<ArkFr> {
        (0..(1usize << k))
            .map(|i| fr(seed.wrapping_mul(1_000_003).wrapping_add(i as u64 * 7 + 1)))
            .collect()
    }

    // A column of length 2^k whose entries sum to zero.
    fn zero_sum_column(k: usize, seed: u64) -> Vec<ArkFr> {
        let mut col = column(k, seed);
        let sum = col.iter().fold(f0(), |acc, x| acc + *x);
        col[0] = col[0] - sum; // now the whole column sums to zero
        col
    }

    fn transcript() -> Transcript254 {
        Transcript254::new(b"parture-argument-test")
    }

    #[test]
    fn chi_n_closed_form_matches_table_mle_including_non_power_of_two() {
        // The verifier's closed-form chi_n(r) must equal the MLE of the prover's
        // [1;n]++[0;N-n] table for EVERY n in 0..=N, including non-powers-of-two.
        for k in 1..=6 {
            let n_max = 1usize << k;
            let point: Vec<ArkFr> = (0..k).map(|j| fr(7 * j as u64 + 3)).collect();
            for n in 0..=n_max {
                let table = chi_n_table(k, n);
                assert_eq!(
                    chi_n_eval(&point, n),
                    mle_full_eval(&table, &point),
                    "chi_n mismatch at k={k}, n={n}"
                );
            }
        }
    }

    #[test]
    fn zero_sum_and_identity_roundtrip_for_k_4_to_10() {
        for k in 4..=10 {
            // ZeroSum over a single column that sums to zero, plus a product
            // (degree-2) ZeroSum, plus an Identity (zero-check) on a zero column.
            let a = zero_sum_column(k, 11);
            let b = column(k, 22);
            let c = column(k, 33);
            // Make a*? contribute a valid product-zero-sum: force sum(d*b)=0.
            let mut d = column(k, 44);
            let dot = d
                .iter()
                .zip(b.iter())
                .fold(f0(), |acc, (x, y)| acc + *x * *y);
            let b0_inv = b[0].inv().unwrap();
            d[0] = d[0] - dot * b0_inv; // now sum_i d[i]*b[i] == 0
            let zero_col = vec![f0(); 1usize << k];

            let columns = vec![a, b, c.clone(), d, zero_col];
            let shape = SumcheckShape {
                num_vars: k,
                num_columns: columns.len(),
                chi_n: None,
                claims: vec![
                    SumcheckClaim::product(ClaimShape::ZeroSum, &[0]),
                    SumcheckClaim::product(ClaimShape::ZeroSum, &[3, 1]), // sum_x d*b == 0
                    SumcheckClaim::product(ClaimShape::Identity, &[4]), // zero column
                ],
            };

            let mut pt = transcript();
            let (proof, r, evals) = prove_sumcheck(&shape, &columns, &mut pt).unwrap();

            let mut vt = transcript();
            let (r_v, evals_v) = verify_sumcheck(&shape, &proof, &mut vt).unwrap();
            assert_eq!(r, r_v);
            assert_eq!(evals, evals_v);

            // The claimed evals must be the true MLE evaluations at r.
            for (col, eval) in columns.iter().zip(evals.iter()) {
                assert_eq!(*eval, mle_full_eval(col, &r));
            }
        }
    }

    #[test]
    fn dishonest_round_polynomial_is_rejected() {
        let k = 6;
        let a = zero_sum_column(k, 7);
        let shape = SumcheckShape {
            num_vars: k,
            num_columns: 1,
            chi_n: None,
            claims: vec![SumcheckClaim::product(ClaimShape::ZeroSum, &[0])],
        };
        let mut pt = transcript();
        let (mut proof, _, _) = prove_sumcheck(&shape, &[a], &mut pt).unwrap();
        proof.round_polys[2][1] = proof.round_polys[2][1] + f1();
        let mut vt = transcript();
        assert!(verify_sumcheck(&shape, &proof, &mut vt).is_err());
    }

    #[test]
    fn dishonest_claimed_evaluation_is_rejected() {
        let k = 6;
        let a = zero_sum_column(k, 9);
        let shape = SumcheckShape {
            num_vars: k,
            num_columns: 1,
            chi_n: None,
            claims: vec![SumcheckClaim::product(ClaimShape::ZeroSum, &[0])],
        };
        let mut pt = transcript();
        let (mut proof, _, _) = prove_sumcheck(&shape, &[a], &mut pt).unwrap();
        proof.claimed_evals[0] = proof.claimed_evals[0] + f1();
        let mut vt = transcript();
        assert!(verify_sumcheck(&shape, &proof, &mut vt).is_err());
    }

    #[test]
    fn identity_claim_on_a_nonzero_column_is_rejected() {
        // Soundness: a zero-check on a column that is NOT zero everywhere must
        // fail (the reduced sum eq(r_zc,x)*c(x) = c(r_zc) != 0 whp).
        let k = 6;
        let nonzero = column(k, 123);
        let shape = SumcheckShape {
            num_vars: k,
            num_columns: 1,
            chi_n: None,
            claims: vec![SumcheckClaim::product(ClaimShape::Identity, &[0])],
        };
        let mut pt = transcript();
        let (proof, _, _) = prove_sumcheck(&shape, &[nonzero], &mut pt).unwrap();
        let mut vt = transcript();
        assert!(verify_sumcheck(&shape, &proof, &mut vt).is_err());
    }

    #[test]
    fn homomorphic_combination_matches_direct_commitment() {
        let k = 4;
        let c0 = column(k, 1);
        let c1 = column(k, 2);
        let com0 = commit_column(&c0).unwrap();
        let com1 = commit_column(&c1).unwrap();
        let gamma = fr(999);
        let (combined_com, _) = combine(&[com0, com1], &[f0(), f0()], gamma);
        // Directly commit the gamma-combined polynomial.
        let combined: Vec<ArkFr> = c0
            .iter()
            .zip(c1.iter())
            .map(|(a, b)| *a + gamma * *b)
            .collect();
        let direct = commit_column(&combined).unwrap();
        assert_eq!(combined_com, direct);
    }

    #[test]
    fn dory_glue_happy_path_matches_mle_fold_and_dory_verify() {
        let k = 6;
        let column = zero_sum_column(k, 55);
        let commitment = commit_column(&column).unwrap();
        let shape = SumcheckShape {
            num_vars: k,
            num_columns: 1,
            chi_n: None,
            claims: vec![SumcheckClaim::product(ClaimShape::ZeroSum, &[0])],
        };

        // Prover: bind the commitment, run sumcheck, open at r.
        let mut pt = transcript();
        pt.append_group(b"pow/column-commitment", &commitment);
        let (proof, r, evals) = prove_sumcheck(&shape, &[column.clone()], &mut pt).unwrap();
        let opening = prove_column_openings(&[column.clone()], &r, &mut pt).unwrap();

        // The claimed eval matches our own MLE-fold arithmetic AND Dory's.
        assert_eq!(evals[0], mle_full_eval(&column, &r));
        let reference = ArkworksPolynomial::new(column.clone()).evaluate(&r);
        assert_eq!(evals[0], reference);

        // Verifier: replay, then verify the batched Dory opening at r.
        let mut vt = transcript();
        vt.append_group(b"pow/column-commitment", &commitment);
        let (r_v, evals_v) = verify_sumcheck(&shape, &proof, &mut vt).unwrap();
        assert_eq!(r, r_v);
        verify_column_openings(&[commitment], &r_v, &evals_v, &opening, &mut vt).unwrap();
    }

    #[test]
    fn point_convention_adversarial_swapping_col_and_row_bits_fails() {
        // The KEY test: if we (incorrectly) fed Dory the point with the
        // [col-bits | row-bits] halves transposed, verification of the true
        // evaluation must FAIL -- proving a convention mismatch is caught, not
        // silently accepted.
        let k = 6;
        let (nu, sigma) = matrix_dims(k);
        let column = column(k, 77);
        let commitment = commit_column(&column).unwrap();

        // A random-ish honest point, then its col/row-transposed sibling.
        let r: Vec<ArkFr> = (0..k).map(|i| fr(3 * i as u64 + 5)).collect();
        let mut swapped = Vec::with_capacity(k);
        swapped.extend_from_slice(&r[sigma..]); // row bits first
        swapped.extend_from_slice(&r[..sigma]); // then column bits
        assert_eq!(sigma + nu, k);

        let honest_eval = mle_full_eval(&column, &r);
        // Precondition: the transposed point genuinely evaluates differently, so
        // the test is meaningful.
        assert_ne!(honest_eval, mle_full_eval(&column, &swapped));

        // Honest opening at r verifies.
        let mut pt = transcript();
        let opening = prove_column_openings(&[column.clone()], &r, &mut pt).unwrap();
        let mut vt = transcript();
        assert!(verify_column_openings(&[commitment], &r, &[honest_eval], &opening, &mut vt).is_ok());

        // Same opening + same claimed eval, but the transposed point: rejected.
        let mut pt_bad = transcript();
        let opening_bad = prove_column_openings(&[column.clone()], &r, &mut pt_bad).unwrap();
        let mut vt_bad = transcript();
        assert!(verify_column_openings(
            &[commitment],
            &swapped,
            &[honest_eval],
            &opening_bad,
            &mut vt_bad
        )
        .is_err());
    }
}
