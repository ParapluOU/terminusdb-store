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
//! # Reader's guide (no cryptography assumed)
//! Every other gadget in this directory ultimately reduces the statement it wants to
//! prove — "this output relation is exactly the matching rows", "these two columns are
//! a permutation of each other", and so on — to two machine-checkable facts about a
//! handful of fixed-width number columns:
//!   1. some arithmetic identity holds when it is summed over all the rows, and
//!   2. the columns named in that identity really are the ones the prover committed to
//!      earlier (so a cheat cannot quietly substitute different numbers).
//! This module is the shared engine for BOTH halves; the per-gadget files just describe
//! which columns and which identity to feed it.
//!
//! Three concepts carry the whole design:
//! * A COMMITMENT is a short, binding fingerprint of an entire column. Publishing it
//!   pins the column down without revealing or resending it, and it is *homomorphic*
//!   (the fingerprint of `a + gamma*b` can be computed from the fingerprints of `a`
//!   and `b`). [`commit_column`] produces one.
//! * The SUMCHECK ([`prove_sumcheck`] / [`verify_sumcheck`]) lets a verifier become
//!   convinced that "the sum of this expression over all `2^k` rows equals the claimed
//!   value" while only ever evaluating the expression at ONE random row. A cheating
//!   prover would have to survive a random challenge it cannot predict, which succeeds
//!   only with negligible probability. Two claim shapes exist: [`ClaimShape::ZeroSum`]
//!   ("the sum equals the public target") and [`ClaimShape::Identity`] ("the expression
//!   is zero at EVERY row", reduced to a single sum via a random `eq` weighting — the
//!   standard zero-check). It is made non-interactive with a Fiat-Shamir transcript, so
//!   the "random" challenges are derived by hashing everything committed so far.
//! * The sumcheck finishes by having the prover claim each committed column's value at
//!   that one random row. The DORY OPENING glue
//!   ([`prove_column_openings`] / [`verify_column_openings`]) is what forces those
//!   claimed values to be consistent with the earlier commitments; many column openings
//!   are batched into a single one using the commitment's homomorphism. "Dory" is the
//!   underlying polynomial-commitment scheme (over the BN254 elliptic curve).
//!
//! A typical gadget therefore: commits its columns, appends the commitments to the
//! transcript, builds a [`SumcheckShape`] describing the identity it needs, runs
//! [`prove_sumcheck`], then [`prove_column_openings`] at the point the sumcheck lands
//! on. The verifier replays the same steps. If both pass, the gadget's statement holds
//! except with negligible probability. Nothing in THIS file is specific to any one
//! gadget — see the individual files for what each proves.
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
#[cfg(any(feature = "blitzar-cpu", feature = "blitzar-gpu"))]
use std::sync::Arc;

use dory_pcs::backends::arkworks::{
    ArkFr, ArkG1, ArkG2, ArkGT, ArkworksPolynomial, Blake2bTranscript, G1Routines, G2Routines,
    BN254,
};
#[cfg(any(feature = "blitzar-cpu", feature = "blitzar-gpu"))]
use dory_pcs::backends::arkworks::BlitzarG1Backend;
// `BlitzarRuntime`/`BlitzarRuntimeConfig` appear in the public `StoreProverContext`
// blitzar constructors' signatures, so re-export them here to make them nameable by
// callers (and thus re-exportable through the woql2 high-level prove API) without a
// direct `dory-pcs` dependency.
#[cfg(any(feature = "blitzar-cpu", feature = "blitzar-gpu"))]
pub use dory_pcs::backends::arkworks::{BlitzarRuntime, BlitzarRuntimeConfig};
use dory_pcs::primitives::arithmetic::{Field, Group, ProverBackend};
use dory_pcs::primitives::poly::Polynomial;
use dory_pcs::primitives::transcript::Transcript;
use dory_pcs::{
    prove_with_context, verify, DoryProof, ProverContext, RoutineBackend, Transparent,
};

use super::commitment::{matrix_dims, prover_setup, setup_parameter_id, verifier_setup};
use super::invalid;

/// The concrete Fiat-Shamir transcript shared by our sumcheck and Dory's
/// prove/verify.
pub type Transcript254 = Blake2bTranscript<BN254>;

/// A Dory single-point opening proof over BN254.
pub type ColumnOpening = DoryProof<ArkG1, ArkG2, ArkGT>;

/// Deterministic Dory parameter tier optionally preprocessed by a prover context.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProverSetupTier {
    Log20,
    Log22,
    Log24,
}

impl ProverSetupTier {
    pub const fn log_domain(self) -> usize {
        match self {
            Self::Log20 => 20,
            Self::Log22 => 22,
            Self::Log24 => 24,
        }
    }
}

enum StoreG1Backend<'runtime> {
    Cpu(
        RoutineBackend<G1Routines>,
        std::marker::PhantomData<&'runtime ()>,
    ),
    #[cfg(any(feature = "blitzar-cpu", feature = "blitzar-gpu"))]
    Blitzar(BlitzarG1Backend<'runtime>),
}

impl ProverBackend<ArkG1> for StoreG1Backend<'_> {
    fn msm(&self, bases: &[ArkG1], scalars: &[ArkFr]) -> ArkG1 {
        match self {
            Self::Cpu(backend, _) => backend.msm(bases, scalars),
            #[cfg(any(feature = "blitzar-cpu", feature = "blitzar-gpu"))]
            Self::Blitzar(backend) => backend.msm(bases, scalars),
        }
    }

    fn batch_msm_same_bases(&self, bases: &[ArkG1], batches: &[&[ArkFr]]) -> Vec<ArkG1> {
        match self {
            Self::Cpu(backend, _) => backend.batch_msm_same_bases(bases, batches),
            #[cfg(any(feature = "blitzar-cpu", feature = "blitzar-gpu"))]
            Self::Blitzar(backend) => backend.batch_msm_same_bases(bases, batches),
        }
    }

    fn fixed_base_vector_scalar_mul(&self, base: &ArkG1, scalars: &[ArkFr]) -> Vec<ArkG1> {
        match self {
            Self::Cpu(backend, _) => backend.fixed_base_vector_scalar_mul(base, scalars),
            #[cfg(any(feature = "blitzar-cpu", feature = "blitzar-gpu"))]
            Self::Blitzar(backend) => backend.fixed_base_vector_scalar_mul(base, scalars),
        }
    }

    fn fixed_scalar_mul_bases_then_add(
        &self,
        bases: &[ArkG1],
        values: &mut [ArkG1],
        scalar: &ArkFr,
    ) {
        match self {
            Self::Cpu(backend, _) => backend.fixed_scalar_mul_bases_then_add(bases, values, scalar),
            #[cfg(any(feature = "blitzar-cpu", feature = "blitzar-gpu"))]
            Self::Blitzar(backend) => {
                backend.fixed_scalar_mul_bases_then_add(bases, values, scalar)
            }
        }
    }

    fn fixed_scalar_mul_vs_then_add(
        &self,
        values: &mut [ArkG1],
        addends: &[ArkG1],
        scalar: &ArkFr,
    ) {
        match self {
            Self::Cpu(backend, _) => backend.fixed_scalar_mul_vs_then_add(values, addends, scalar),
            #[cfg(any(feature = "blitzar-cpu", feature = "blitzar-gpu"))]
            Self::Blitzar(backend) => backend.fixed_scalar_mul_vs_then_add(values, addends, scalar),
        }
    }

    fn fold_field_vectors(&self, left: &mut [ArkFr], right: &[ArkFr], scalar: &ArkFr) {
        match self {
            Self::Cpu(backend, _) => backend.fold_field_vectors(left, right, scalar),
            #[cfg(any(feature = "blitzar-cpu", feature = "blitzar-gpu"))]
            Self::Blitzar(backend) => backend.fold_field_vectors(left, right, scalar),
        }
    }
}

enum StoreG2Backend {
    Cpu(RoutineBackend<G2Routines>),
    Provider(Box<dyn ProverBackend<ArkG2> + Send + Sync>),
}

impl ProverBackend<ArkG2> for StoreG2Backend {
    fn msm(&self, bases: &[ArkG2], scalars: &[ArkFr]) -> ArkG2 {
        match self {
            Self::Cpu(backend) => backend.msm(bases, scalars),
            Self::Provider(backend) => backend.msm(bases, scalars),
        }
    }

    fn batch_msm_same_bases(&self, bases: &[ArkG2], batches: &[&[ArkFr]]) -> Vec<ArkG2> {
        match self {
            Self::Cpu(backend) => backend.batch_msm_same_bases(bases, batches),
            Self::Provider(backend) => backend.batch_msm_same_bases(bases, batches),
        }
    }

    fn fixed_base_vector_scalar_mul(&self, base: &ArkG2, scalars: &[ArkFr]) -> Vec<ArkG2> {
        match self {
            Self::Cpu(backend) => backend.fixed_base_vector_scalar_mul(base, scalars),
            Self::Provider(backend) => backend.fixed_base_vector_scalar_mul(base, scalars),
        }
    }

    fn fixed_scalar_mul_bases_then_add(
        &self,
        bases: &[ArkG2],
        values: &mut [ArkG2],
        scalar: &ArkFr,
    ) {
        match self {
            Self::Cpu(backend) => backend.fixed_scalar_mul_bases_then_add(bases, values, scalar),
            Self::Provider(backend) => {
                backend.fixed_scalar_mul_bases_then_add(bases, values, scalar)
            }
        }
    }

    fn fixed_scalar_mul_vs_then_add(
        &self,
        values: &mut [ArkG2],
        addends: &[ArkG2],
        scalar: &ArkFr,
    ) {
        match self {
            Self::Cpu(backend) => backend.fixed_scalar_mul_vs_then_add(values, addends, scalar),
            Self::Provider(backend) => {
                backend.fixed_scalar_mul_vs_then_add(values, addends, scalar)
            }
        }
    }

    fn fold_field_vectors(&self, left: &mut [ArkFr], right: &[ArkFr], scalar: &ArkFr) {
        match self {
            Self::Cpu(backend) => backend.fold_field_vectors(left, right, scalar),
            Self::Provider(backend) => backend.fold_field_vectors(left, right, scalar),
        }
    }
}

/// Caller-created Store proving arithmetic. It is never serialized or consulted
/// by verification, and selecting it cannot change proof bytes.
pub struct StoreProverContext<'runtime> {
    inner: ProverContext<BN254, StoreG1Backend<'runtime>, StoreG2Backend>,
    accelerated_tier: Option<ProverSetupTier>,
}

impl StoreProverContext<'static> {
    /// Explicit stock-Arkworks CPU context used by compatibility conveniences.
    pub const fn cpu() -> Self {
        Self {
            inner: ProverContext::new(
                StoreG1Backend::Cpu(RoutineBackend::new(), std::marker::PhantomData),
                StoreG2Backend::Cpu(RoutineBackend::new()),
            ),
            accelerated_tier: None,
        }
    }

    /// Construct the portable CPU-G1 context with a caller-owned exact BN254
    /// G2 provider. This is an arithmetic-only choice and cannot affect proof
    /// format or verification.
    pub fn cpu_with_g2_backend<B>(g2_backend: B) -> Self
    where
        B: ProverBackend<ArkG2> + Send + Sync + 'static,
    {
        Self {
            inner: ProverContext::new(
                StoreG1Backend::Cpu(RoutineBackend::new(), std::marker::PhantomData),
                StoreG2Backend::Provider(Box::new(g2_backend)),
            ),
            accelerated_tier: None,
        }
    }

    /// A Blitzar (native/GPU) accelerated G1 context at the given setup `tier`, with the
    /// stock Arkworks G2 backend. Arithmetic-only: it cannot change proof bytes.
    #[cfg(any(feature = "blitzar-cpu", feature = "blitzar-gpu"))]
    pub fn blitzar(runtime: &Arc<BlitzarRuntime>, tier: ProverSetupTier) -> io::Result<Self> {
        Self::blitzar_with_g2_backend(runtime, tier, RoutineBackend::<G2Routines>::new())
    }

    /// Construct the native Blitzar-G1 context with a caller-owned exact BN254
    /// G2 provider. Blitzar itself does not provide BN254 G2/Fq2 arithmetic.
    #[cfg(any(feature = "blitzar-cpu", feature = "blitzar-gpu"))]
    pub fn blitzar_with_g2_backend<B>(
        runtime: &Arc<BlitzarRuntime>,
        tier: ProverSetupTier,
        g2_backend: B,
    ) -> io::Result<Self>
    where
        B: ProverBackend<ArkG2> + Send + Sync + 'static,
    {
        let setup = prover_setup(tier.log_domain())?;
        Ok(Self {
            inner: ProverContext::new(
                StoreG1Backend::Blitzar(BlitzarG1Backend::new(
                    Arc::clone(runtime),
                    &setup.g1_vec,
                )),
                StoreG2Backend::Provider(Box::new(g2_backend)),
            ),
            accelerated_tier: Some(tier),
        })
    }

    /// Initialize a shared Blitzar runtime (GPU/native MSM backend) to pass to
    /// [`Self::blitzar`]. One runtime can be reused across many prover contexts.
    #[cfg(any(feature = "blitzar-cpu", feature = "blitzar-gpu"))]
    pub fn initialize_blitzar(config: BlitzarRuntimeConfig) -> Arc<BlitzarRuntime> {
        Arc::new(BlitzarRuntime::initialize(config))
    }
}

impl<'runtime> StoreProverContext<'runtime> {
    /// The single setup tier preprocessed by this context, if any.
    pub const fn accelerated_tier(&self) -> Option<ProverSetupTier> {
        self.accelerated_tier
    }

    /// Commit one proof column using this context.
    pub fn commit_column(&self, column: &[ArkFr]) -> io::Result<ArkGT> {
        commit_column_with_context(self, column)
    }

    /// Open proof columns using this context and the caller's transcript.
    pub fn prove_column_openings(
        &self,
        column_tables: &[Vec<ArkFr>],
        r: &[ArkFr],
        transcript: &mut Transcript254,
    ) -> io::Result<ColumnOpening> {
        prove_column_openings_with_context(self, column_tables, r, transcript)
    }

}

const LABEL_DOMAIN: &[u8] = b"pow/sumcheck/domain";
const LABEL_SHAPE: &[u8] = b"pow/sumcheck/shape";
const LABEL_ZEROCHECK: &[u8] = b"pow/sumcheck/zerocheck-point";
const LABEL_RHO: &[u8] = b"pow/sumcheck/batch-rho";
pub(crate) const LABEL_ROUND_POLY: &[u8] = b"pow/sumcheck/round-poly";
pub(crate) const LABEL_ROUND_CHALLENGE: &[u8] = b"pow/sumcheck/round-challenge";
const LABEL_DORY_GAMMA: &[u8] = b"pow/dory/batch-gamma";
// Fiat-Shamir binding of the batched-opening challenge `gamma` to the exact
// per-column claimed evaluations. These MUST be absorbed (count first, then each
// eval in column order 0..m) immediately BEFORE `gamma` is drawn, symmetrically
// on the prove and verify sides; otherwise `gamma` is independent of the
// individual evals and a prover can forge them on the opening hyperplane
// `Sum_j gamma^j * eval_j = C` while keeping the combined value fixed.
const LABEL_DORY_EVAL_COUNT: &[u8] = b"pow/dory/eval-count";
const LABEL_DORY_EVAL: &[u8] = b"pow/dory/claimed-eval";
// Fiat-Shamir binding of `gamma` to the batched opening's per-column
// COMMITMENTS. These MUST be absorbed (count first, then each commitment in
// column order 0..m) immediately BEFORE the eval block (hence before `gamma`),
// symmetrically on prove and verify. Without this the standalone opener/Dory
// verifier is unsound for adaptively-chosen commitments: `gamma` would be
// independent of the individual commitments and a malicious caller could
// substitute a different commitment set on the combined opening.
const LABEL_DORY_COMMIT_COUNT: &[u8] = b"pow/dory/commit-count";
const LABEL_DORY_COMMIT: &[u8] = b"pow/dory/column-commitment";

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
    /// The MLE of the PUBLIC prefix indicator of `[0, chi_bounds[b])`
    /// (verifier-derivable; never committed). The bound is carried by
    /// [`SumcheckShape::chi_bounds`]`[b]`.
    Chi(usize),
    /// The MLE of the PUBLIC index function `f(i) = i` over `{0,1}^k`
    /// (verifier-derivable; never committed). Its evaluation at `r` is the
    /// closed form `idx(r) = Sum_j r_j * 2^j`.
    Idx,
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
}

/// One claim: a shape, the expression `G` it constrains, and the PUBLIC target
/// sum it asserts. `target` is the value `Sum_x G(x)` (for `ZeroSum`) that the
/// verifier already knows; it is `0` for every `Identity` claim and for a plain
/// zero-sum, but may be a nonzero public scalar (e.g. `1/(alpha - q_fold)` for a
/// LogUp membership claim). See [`SumcheckClaim::product`] / the logup builders,
/// which default it to `0`.
#[derive(Clone, Debug)]
pub struct SumcheckClaim {
    pub shape: ClaimShape,
    pub expr: Expr,
    pub target: ArkFr,
}

impl SumcheckClaim {
    /// Convenience: a zero-target claim whose `G` is a product of committed columns.
    pub fn product(shape: ClaimShape, factors: &[usize]) -> Self {
        SumcheckClaim {
            shape,
            expr: Expr::product(factors),
            target: f0(),
        }
    }
}

/// The public description of a batched sumcheck instance (shared by prover and
/// verifier; contains no witness data). `chi_bounds` supplies the public prefix
/// bounds for any [`Operand::Chi`] used by the claims (e.g. `[n, m]` for the
/// filter's real-row count and the compaction's output length).
#[derive(Clone, Debug)]
pub struct SumcheckShape {
    pub num_vars: usize,
    pub num_columns: usize,
    pub chi_bounds: Vec<usize>,
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
                        Operand::Chi(b) if *b >= self.chi_bounds.len() => {
                            return Err(invalid("claim uses an undefined chi bound"));
                        }
                        _ => {}
                    }
                }
            }
        }
        if self
            .chi_bounds
            .iter()
            .any(|&bound| bound > (1usize << self.num_vars))
        {
            return Err(invalid("chi bound exceeds the hypercube size"));
        }
        Ok(())
    }

    fn has_identity(&self) -> bool {
        self.claims
            .iter()
            .any(|claim| claim.shape == ClaimShape::Identity)
    }

    fn uses_idx(&self) -> bool {
        self.claims.iter().any(|claim| {
            claim
                .expr
                .terms
                .iter()
                .any(|term| term.operands.iter().any(|op| *op == Operand::Idx))
        })
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
pub(crate) fn eq_table(point: &[ArkFr]) -> Vec<ArkFr> {
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
pub(crate) fn eq_eval(a: &[ArkFr], b: &[ArkFr]) -> ArkFr {
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

/// The prover's index table `[0, 1, 2, .., 2^k - 1]` (the values of `f(i) = i`),
/// whose multilinear extension the verifier reproduces closed-form via
/// [`idx_eval`].
fn idx_table(k: usize) -> Vec<ArkFr> {
    (0..(1usize << k))
        .map(|i| ArkFr::from_u64(i as u64))
        .collect()
}

/// Closed-form MLE of the PUBLIC index function `f(i) = i` at `point` (LSB-first):
/// `idx(r) = Sum_j r_j * 2^j` (since `i = Sum_j bit_j(i) * 2^j` and each bit's MLE
/// is `r_j`).
pub fn idx_eval(point: &[ArkFr]) -> ArkFr {
    let mut acc = f0();
    for (j, rj) in point.iter().enumerate() {
        acc = acc + *rj * ArkFr::from_u64(1u64 << j);
    }
    acc
}

/// Fold a table in place at its current lowest variable: `new[i] =
/// t[2i]*(1-r) + t[2i+1]*r`, halving the length.
pub(crate) fn fold_table(table: &mut Vec<ArkFr>, r: ArkFr) {
    let half = table.len() / 2;
    let one_minus = f1() - r;
    for i in 0..half {
        table[i] = table[2 * i] * one_minus + table[2 * i + 1] * r;
    }
    table.truncate(half);
}

/// `[base^0, base^1, .., base^{count-1}]`.
pub(crate) fn powers(base: ArkFr, count: usize) -> Vec<ArkFr> {
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
pub(crate) fn interpolate(evals: &[ArkFr], x: ArkFr) -> io::Result<ArkFr> {
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
    transcript.append_bytes(LABEL_SHAPE, &(shape.chi_bounds.len() as u64).to_le_bytes());
    for bound in &shape.chi_bounds {
        transcript.append_bytes(LABEL_SHAPE, &(*bound as u64).to_le_bytes());
    }
    transcript.append_bytes(LABEL_SHAPE, &(shape.claims.len() as u64).to_le_bytes());
    for claim in &shape.claims {
        let tag: u8 = match claim.shape {
            ClaimShape::ZeroSum => 0,
            ClaimShape::Identity => 1,
        };
        transcript.append_bytes(LABEL_SHAPE, &[tag]);
        transcript.append_field(LABEL_SHAPE, &claim.target);
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
                    Operand::Chi(b) => {
                        transcript.append_bytes(LABEL_SHAPE, &[1u8]);
                        transcript.append_bytes(LABEL_SHAPE, &(*b as u64).to_le_bytes());
                    }
                    Operand::Idx => transcript.append_bytes(LABEL_SHAPE, &[2u8]),
                }
            }
        }
    }
}

/// The preamble that both sides run AFTER the caller has appended the column
/// commitments to the transcript: bind the shape (domain tag), then draw the
/// zero-check point `r_zc` (only if some claim is an identity claim), then the
/// batching challenge `rho`. Returns `(r_zc, rho_powers)`.
fn draw_preamble(
    transcript: &mut Transcript254,
    shape: &SumcheckShape,
) -> (Vec<ArkFr>, Vec<ArkFr>) {
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
    let mut chi_tables: Vec<Vec<ArkFr>> = shape
        .chi_bounds
        .iter()
        .map(|&bound| chi_n_table(k, bound))
        .collect();
    let mut idx = if shape.uses_idx() {
        idx_table(k)
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
            let at =
                |table: &[ArkFr], i: usize| table[2 * i] * one_minus + table[2 * i + 1] * x_field;
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
                                    Operand::Chi(b) => at(&chi_tables[*b], i),
                                    Operand::Idx => at(&idx, i),
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
        for chi in chi_tables.iter_mut() {
            fold_table(chi, challenge);
        }
        if shape.uses_idx() {
            fold_table(&mut idx, challenge);
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
        return Err(invalid(
            "sumcheck proof has the wrong number of claimed evals",
        ));
    }
    let degree = shape.round_degree();

    let (r_zc, rhos) = draw_preamble(transcript, shape);

    // The batched claim starts at the rho-combination of the per-claim PUBLIC
    // target sums (0 for identity/plain-zero-sum claims, a known nonzero scalar
    // for a public-target claim such as LogUp membership).
    let mut current = shape
        .claims
        .iter()
        .zip(rhos.iter())
        .fold(f0(), |acc, (claim, rho)| acc + *rho * claim.target);
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
    // rebuilt from the claimed column evaluations at r, the public chi(r) values,
    // and the public constants baked into the claim expressions.
    let chi_at_r: Vec<ArkFr> = shape
        .chi_bounds
        .iter()
        .map(|&bound| chi_n_eval(&r, bound))
        .collect();
    let idx_at_r = idx_eval(&r);
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
                        Operand::Chi(b) => chi_at_r[*b],
                        Operand::Idx => idx_at_r,
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

/// Commit to a raw column (length `2^k`) with its deterministic parameter tier,
/// returning its succinct GT (tier-2) commitment. Uses the shared matrix layout.
///
/// Exposed `pub` so external consumers (e.g. the `woql-proof` plan compiler's
/// end-to-end tests) can INDEPENDENTLY recompute expected result-column
/// commitments and compare them to a BGP proof's output. Pure and deterministic.
pub fn commit_column(column: &[ArkFr]) -> io::Result<ArkGT> {
    commit_column_with_context(&StoreProverContext::cpu(), column)
}

/// Commit a raw column with explicitly selected prover arithmetic.
pub fn commit_column_with_context(
    context: &StoreProverContext<'_>,
    column: &[ArkFr],
) -> io::Result<ArkGT> {
    let (nu, sigma) = column_dims(column.len())?;
    let num_vars = column.len().trailing_zeros() as usize;
    let poly = ArkworksPolynomial::new(column.to_vec());
    let (tier_2, _rows, _blind) = poly
        .commit_with_backend::<BN254, Transparent, _>(
            nu,
            sigma,
            prover_setup(num_vars)?,
            context.inner.g1(),
        )
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
    prove_column_openings_with_context(&StoreProverContext::cpu(), column_tables, r, transcript)
}

/// Prove batched column openings with explicitly selected prover arithmetic.
pub fn prove_column_openings_with_context(
    context: &StoreProverContext<'_>,
    column_tables: &[Vec<ArkFr>],
    r: &[ArkFr],
    transcript: &mut Transcript254,
) -> io::Result<ColumnOpening> {
    let size = 1usize << r.len();
    if column_tables.iter().any(|c| c.len() != size) {
        return Err(invalid("column length does not match evaluation point"));
    }
    let (nu, sigma) = column_dims(size)?;
    transcript.append_bytes(b"pow/dory/parameter-set", &setup_parameter_id(r.len())?);
    // Bind `gamma` to the per-column COMMITMENTS BEFORE drawing it. The prover
    // RECOMPUTES each column commitment; for any opening that verifies these
    // equal the `column_commitments[j]` the verifier appends (Dory binds the
    // gamma-combined commitment to the gamma-combined polynomial, forcing
    // per-column equality), so honest prove/verify stay byte-identical.
    // The verifier appends the IDENTICAL bytes (see `verify_column_openings`).
    transcript.append_bytes(
        LABEL_DORY_COMMIT_COUNT,
        &(column_tables.len() as u64).to_le_bytes(),
    );
    for table in column_tables.iter() {
        let com = commit_column_with_context(context, table)?;
        transcript.append_group(LABEL_DORY_COMMIT, &com);
    }
    // Bind `gamma` to the exact per-column evaluations at `r` BEFORE drawing it.
    // For a sumcheck-backed caller these equal `proof.claimed_evals[j]`; for a
    // direct caller they equal the honest `col_evals[j]` the verifier passes.
    // The verifier appends the IDENTICAL bytes (see `verify_column_openings`).
    transcript.append_bytes(
        LABEL_DORY_EVAL_COUNT,
        &(column_tables.len() as u64).to_le_bytes(),
    );
    for table in column_tables.iter() {
        let eval_j = mle_full_eval(table, r);
        transcript.append_field(LABEL_DORY_EVAL, &eval_j);
    }
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
        .commit_with_backend::<BN254, Transparent, _>(
            nu,
            sigma,
            prover_setup(r.len())?,
            context.inner.g1(),
        )
        .map_err(|error| invalid(format!("batched column commitment failed: {error}")))?;
    let (proof, _) = prove_with_context::<_, BN254, _, _, _, _, Transparent>(
        &poly,
        r,
        row_commitments,
        blind,
        nu,
        sigma,
        prover_setup(r.len())?,
        &context.inner,
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
    transcript.append_bytes(b"pow/dory/parameter-set", &setup_parameter_id(r.len())?);
    // Bind `gamma` to the per-column COMMITMENTS BEFORE drawing it. MUST append
    // byte-identically to the prover (see `prove_column_openings_with_context`):
    // the count as u64 le-bytes, then each commitment in column order 0..m.
    transcript.append_bytes(
        LABEL_DORY_COMMIT_COUNT,
        &(column_commitments.len() as u64).to_le_bytes(),
    );
    for c in column_commitments.iter() {
        transcript.append_group(LABEL_DORY_COMMIT, c);
    }
    // Bind `gamma` to the exact per-column claimed evaluations BEFORE drawing it.
    // MUST append byte-identically to the prover (see
    // `prove_column_openings_with_context`): the count as u64 le-bytes, then each
    // eval in column order 0..m.
    transcript.append_bytes(
        LABEL_DORY_EVAL_COUNT,
        &(claimed_evals.len() as u64).to_le_bytes(),
    );
    for eval_j in claimed_evals.iter() {
        transcript.append_field(LABEL_DORY_EVAL, eval_j);
    }
    let gamma = transcript.challenge_scalar(LABEL_DORY_GAMMA);
    let (combined_commitment, combined_eval) = combine(column_commitments, claimed_evals, gamma);
    verify::<_, BN254, G1Routines, G2Routines, _>(
        combined_commitment,
        combined_eval,
        r,
        opening,
        verifier_setup(r.len())?.clone(),
        transcript,
    )
    .map_err(|error| invalid(format!("dory batched opening rejected: {error}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_serialize::CanonicalSerialize;
    use dory_pcs::ProfiledBackend;

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
    fn idx_closed_form_matches_table_mle() {
        // idx(r) = Sum_j r_j*2^j must equal the MLE of the table [0,1,..,2^k-1].
        for k in 1..=6 {
            let point: Vec<ArkFr> = (0..k).map(|j| fr(5 * j as u64 + 2)).collect();
            assert_eq!(idx_eval(&point), mle_full_eval(&idx_table(k), &point));
        }
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
                chi_bounds: Vec::new(),
                claims: vec![
                    SumcheckClaim::product(ClaimShape::ZeroSum, &[0]),
                    SumcheckClaim::product(ClaimShape::ZeroSum, &[3, 1]), // sum_x d*b == 0
                    SumcheckClaim::product(ClaimShape::Identity, &[4]),   // zero column
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
            chi_bounds: Vec::new(),
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
            chi_bounds: Vec::new(),
            claims: vec![SumcheckClaim::product(ClaimShape::ZeroSum, &[0])],
        };
        let mut pt = transcript();
        let (mut proof, _, _) = prove_sumcheck(&shape, &[a], &mut pt).unwrap();
        proof.claimed_evals[0] = proof.claimed_evals[0] + f1();
        let mut vt = transcript();
        assert!(verify_sumcheck(&shape, &proof, &mut vt).is_err());
    }

    #[test]
    fn forged_multicolumn_evals_on_opening_hyperplane_are_rejected() {
        // Regression for the Fiat-Shamir soundness bug in the shared batched
        // column opening: `gamma` was drawn BEFORE the per-column claimed
        // evaluations were absorbed into the transcript. With `gamma`
        // independent of the individual evals, the batched Dory opening only
        // pins down the gamma-COMBINATION `Sum_j gamma^j * eval_j = C`, so a
        // prover could replace the honest eval vector with ANY other vector on
        // that same opening hyperplane (same combination, different individual
        // values) and STILL verify -- a false per-column statement accepted.
        //
        // Form used: the "same gamma-combination" regression. The fully
        // adaptive attack (choose the forged evals AFTER seeing `gamma`) is by
        // construction UNCONSTRUCTIBLE against the fixed code -- once the evals
        // are absorbed before `gamma`, the prover can no longer pick them after
        // the challenge. So the faithful residue is: forge a distinct eval
        // vector with the same honest-`gamma` combination and assert rejection.
        //
        // Under the OLD (buggy) code `gamma` did not depend on the evals, so the
        // honest opening and the verifier used the same fixed `gamma` regardless
        // of the eval vector, and this forgery VERIFIED. Under the fix the
        // verifier re-derives `gamma` from the forged vector, gets a DIFFERENT
        // `gamma`, and rejects.
        let k = 5;
        let c0 = column(k, 71);
        let c1 = column(k, 137);
        let coms = [commit_column(&c0).unwrap(), commit_column(&c1).unwrap()];

        // Honest point + honest per-column evaluations at r.
        let r: Vec<ArkFr> = (0..k).map(|i| fr(9 * i as u64 + 4)).collect();
        let e0 = mle_full_eval(&c0, &r);
        let e1 = mle_full_eval(&c1, &r);

        // Honest multi-column opening at r.
        let mut pt = transcript();
        let opening = prove_column_openings(&[c0, c1], &r, &mut pt).unwrap();

        // Positive control: the honest eval vector verifies.
        let mut vt = transcript();
        verify_column_openings(&coms, &r, &[e0, e1], &opening, &mut vt).unwrap();

        // Reconstruct the `gamma` that a REGRESSED, eval-UNBOUND opener would
        // draw -- i.e. the pre-FABLE-1 schedule where the per-column evals are
        // NOT absorbed before `gamma`. We DO absorb the parameter-set and the
        // per-column COMMITMENTS (that binding, FABLE-2, is a separate fix and is
        // still present in such a regression), then draw `gamma` WITHOUT the eval
        // block. This is exactly the challenge such a regressed prove/verify pair
        // would use, so the forged eval vector below lands on the opening
        // hyperplane THAT verifier checks and it would be ACCEPTED there. The
        // FIXED verifier instead absorbs the evals before `gamma`, re-derives a
        // DIFFERENT challenge, and rejects -- which is what makes this test fail
        // if production ever reverted the eval binding.
        let mut gt = transcript();
        gt.append_bytes(b"pow/dory/parameter-set", &setup_parameter_id(r.len()).unwrap());
        gt.append_bytes(LABEL_DORY_COMMIT_COUNT, &(2u64).to_le_bytes());
        gt.append_group(LABEL_DORY_COMMIT, &coms[0]);
        gt.append_group(LABEL_DORY_COMMIT, &coms[1]);
        let gamma = gt.challenge_scalar(LABEL_DORY_GAMMA);

        // Forge a DISTINCT eval vector on the SAME opening hyperplane:
        //   (e0 + gamma) + gamma * (e1 - 1) == e0 + gamma * e1  (the honest C).
        let forged = [e0 + gamma, e1 - f1()];
        assert_ne!(forged[0], e0, "gamma must be nonzero whp");
        assert_ne!(forged[1], e1);
        // Sanity: the forgery really does preserve the gamma-combination.
        assert_eq!(forged[0] + gamma * forged[1], e0 + gamma * e1);

        // The fixed verifier binds `gamma` to the eval vector, so the forged
        // vector yields a different `gamma` and the opening is rejected.
        let mut vt_bad = transcript();
        assert!(
            verify_column_openings(&coms, &r, &forged, &opening, &mut vt_bad).is_err(),
            "forged eval vector on the honest opening hyperplane must be rejected"
        );
    }

    #[test]
    fn altered_column_commitments_are_rejected() {
        // Regression for the Fiat-Shamir soundness bug in the shared batched
        // column opening: the opener drew `gamma` WITHOUT first absorbing the
        // per-column COMMITMENTS. With `gamma` independent of the commitments, a
        // malicious caller could hand `verify_column_openings` a DIFFERENT
        // commitment set than the one the honest proof was produced against and
        // still verify. The fix absorbs the commitments (count + each in column
        // order) before `gamma`, so any change re-derives a different `gamma`
        // (and a different gamma-combined commitment) and the opening is
        // rejected.
        let k = 5;
        let c0 = column(k, 71);
        let c1 = column(k, 137);
        let com0 = commit_column(&c0).unwrap();
        let com1 = commit_column(&c1).unwrap();
        assert_ne!(com0, com1, "distinct columns must have distinct commitments");

        // Honest point + honest per-column evaluations at r.
        let r: Vec<ArkFr> = (0..k).map(|i| fr(9 * i as u64 + 4)).collect();
        let e0 = mle_full_eval(&c0, &r);
        let e1 = mle_full_eval(&c1, &r);

        // Honest multi-column opening at r over columns [c0, c1].
        let mut pt = transcript();
        let opening = prove_column_openings(&[c0.clone(), c1.clone()], &r, &mut pt).unwrap();

        // Positive control: the honest commitment set + evals verify.
        let mut vt = transcript();
        verify_column_openings(&[com0, com1], &r, &[e0, e1], &opening, &mut vt).unwrap();

        // Attack A: swap the two column commitments, keep everything else from
        // the honest proof (same r, same evals, same opening). The commitments
        // are now transcript-bound, so the verifier re-derives a different
        // `gamma` and rejects.
        let mut vt_swap = transcript();
        assert!(
            verify_column_openings(&[com1, com0], &r, &[e0, e1], &opening, &mut vt_swap).is_err(),
            "swapped column commitments must be rejected"
        );

        // Attack B: replace one commitment with a re-commitment of a DIFFERENT
        // column while keeping the honest evals/opening. Also rejected.
        let c2 = column(k, 271);
        let com2 = commit_column(&c2).unwrap();
        let mut vt_repl = transcript();
        assert!(
            verify_column_openings(&[com0, com2], &r, &[e0, e1], &opening, &mut vt_repl).is_err(),
            "substituted column commitment must be rejected"
        );

        // Attack C (the ISOLATING one). Attacks A/B also change the gamma-COMBINED
        // commitment, so a commitment-unbound (pre-FABLE-2) Dory opening would
        // reject them anyway -- they don't prove the commitment BINDING does any
        // work. This attack hands the verifier a DIFFERENT commitment set whose
        // combined commitment is IDENTICAL to the honest one under the challenge a
        // commitment-unbound opener would draw, so the Dory opening alone cannot
        // tell them apart: ONLY the FABLE-2 pre-`gamma` commitment binding catches
        // it. Hence this assertion FLIPS to accept -- and the test FAILS -- if the
        // commitment appends are ever removed from production.
        //
        // `gamma_old` = the challenge a commitment-unbound schedule draws:
        // parameter-set, then the eval block, then `gamma` (NO commitment block).
        let mut gt = transcript();
        gt.append_bytes(b"pow/dory/parameter-set", &setup_parameter_id(r.len()).unwrap());
        gt.append_bytes(LABEL_DORY_EVAL_COUNT, &(2u64).to_le_bytes());
        gt.append_field(LABEL_DORY_EVAL, &e0);
        gt.append_field(LABEL_DORY_EVAL, &e1);
        let gamma_old = gt.challenge_scalar(LABEL_DORY_GAMMA);
        let gamma_old_inv = gamma_old.inv().expect("gamma_old must be nonzero whp");

        // Combined-preserving perturbation: com0' = com0 + delta,
        // com1' = com1 - gamma_old^{-1} * delta, so under gamma_old
        //   com0' + gamma_old*com1' == com0 + gamma_old*com1
        // (the honest combined commitment the opening certifies), while
        // (com0', com1') != (com0, com1).
        let delta = com0; // any nonzero group element
        let com0_prime = com0 + delta;
        let com1_prime = com1 - delta.scale(&gamma_old_inv);
        assert_ne!(com0_prime, com0, "forged commitment 0 must differ");
        assert_ne!(com1_prime, com1, "forged commitment 1 must differ");
        // Sanity: under gamma_old the two sets share the SAME combined commitment.
        let (combined_honest, _) = combine(&[com0, com1], &[f0(), f0()], gamma_old);
        let (combined_forged, _) = combine(&[com0_prime, com1_prime], &[f0(), f0()], gamma_old);
        assert_eq!(
            combined_honest, combined_forged,
            "the forged set must be indistinguishable to a commitment-unbound opening"
        );

        let mut vt_iso = transcript();
        assert!(
            verify_column_openings(&[com0_prime, com1_prime], &r, &[e0, e1], &opening, &mut vt_iso)
                .is_err(),
            "a commitment set only the pre-gamma commitment binding can distinguish must be rejected"
        );
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
            chi_bounds: Vec::new(),
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
            chi_bounds: Vec::new(),
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
    fn caller_owned_g2_provider_preserves_opening_bytes_and_reports_work() {
        let g2 = ProfiledBackend::new(RoutineBackend::<G2Routines>::new());
        let profile = g2.profile();
        let provided = StoreProverContext::cpu_with_g2_backend(g2);
        let cpu = StoreProverContext::cpu();
        let column = column(6, 92);
        let point = (0..6).map(|i| fr(17 * i as u64 + 3)).collect::<Vec<_>>();

        let mut cpu_transcript = Transcript254::new(b"store-g2-provider-equivalence");
        let cpu_proof = cpu
            .prove_column_openings(&[column.clone()], &point, &mut cpu_transcript)
            .unwrap();
        let mut provided_transcript = Transcript254::new(b"store-g2-provider-equivalence");
        let provided_proof = provided
            .prove_column_openings(&[column], &point, &mut provided_transcript)
            .unwrap();
        assert_eq!(cpu_proof, provided_proof);

        let mut cpu_bytes = Vec::new();
        cpu_proof.serialize_compressed(&mut cpu_bytes).unwrap();
        let mut provided_bytes = Vec::new();
        provided_proof
            .serialize_compressed(&mut provided_bytes)
            .unwrap();
        assert_eq!(cpu_bytes, provided_bytes);

        let snapshot = profile.snapshot();
        assert!(snapshot.msm_calls > 0);
        assert!(snapshot.msm_scalar_terms > 0);
        assert!(snapshot.fixed_base_outputs > 0);
        assert!(snapshot.vector_fma_elements > 0);
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
        assert!(
            verify_column_openings(&[commitment], &r, &[honest_eval], &opening, &mut vt).is_ok()
        );

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

    #[cfg(any(feature = "blitzar-cpu", feature = "blitzar-gpu"))]
    #[test]
    fn caller_owned_blitzar_context_preserves_commitments_and_proof_bytes() {
        let runtime = StoreProverContext::initialize_blitzar(BlitzarRuntimeConfig {
            num_precomputed_generators: 0,
        });
        let accelerated = StoreProverContext::blitzar(&runtime, ProverSetupTier::Log20).unwrap();
        drop(runtime);
        let cpu = StoreProverContext::cpu();
        assert_eq!(cpu.accelerated_tier(), None);
        assert_eq!(accelerated.accelerated_tier(), Some(ProverSetupTier::Log20));

        let column = column(6, 91);
        let point = (0..6).map(|i| fr(13 * i as u64 + 2)).collect::<Vec<_>>();
        let cpu_commitment = cpu.commit_column(&column).unwrap();
        let accelerated_commitment = accelerated.commit_column(&column).unwrap();
        assert_eq!(cpu_commitment, accelerated_commitment);

        let mut cpu_transcript = Transcript254::new(b"store-context-equivalence");
        let cpu_proof = cpu
            .prove_column_openings(&[column.clone()], &point, &mut cpu_transcript)
            .unwrap();
        let mut accelerated_transcript = Transcript254::new(b"store-context-equivalence");
        let accelerated_proof = accelerated
            .prove_column_openings(&[column], &point, &mut accelerated_transcript)
            .unwrap();
        assert_eq!(cpu_proof, accelerated_proof);

        let mut cpu_bytes = Vec::new();
        cpu_proof.serialize_compressed(&mut cpu_bytes).unwrap();
        let mut accelerated_bytes = Vec::new();
        accelerated_proof
            .serialize_compressed(&mut accelerated_bytes)
            .unwrap();
        assert_eq!(cpu_bytes, accelerated_bytes);
    }
}
