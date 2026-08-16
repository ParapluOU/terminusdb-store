//! Standalone GKR fraction-tree primitive for log-derivative multiset sums.
//!
//! # What this proves
//! Given a set of fractions `(p_i, q_i)` — a numerator and a nonzero denominator
//! per leaf — this proves the single statement
//! ```text
//!     Σ_i  p_i / q_i  =  0
//! ```
//! WITHOUT ever committing a reciprocal column `1/(alpha - fold)`. The historical
//! LogUp gadget ([`super::logup`]) proved the same multiset facts by committing a
//! per-row reciprocal witness `dstar` and pinning it with an Identity claim; this
//! module instead proves the sum via a **GKR fraction tree over UNCOMMITTED
//! intermediates**, tied at the bottom to the (separately committed) DATA columns.
//! No reciprocal is materialized or committed anywhere.
//!
//! # The fraction tree
//! Each summand is a fraction `(p_i, q_i)`. Fold pairwise up a binary tree; an
//! internal node combines its two children by the standard fraction-add rule
//! ```text
//!     q_out = q_L · q_R
//!     p_out = p_L · q_R + p_R · q_L
//! ```
//! so that `p_out / q_out = p_L/q_L + p_R/q_R`. The ROOT `(P, Q)` therefore
//! satisfies `P/Q = Σ_i p_i/q_i`, and since `Q = Π_i q_i ≠ 0` (whp; see below),
//! the whole statement reduces to the single field equality `P = 0`.
//!
//! A public nonzero target (e.g. the dictionary-membership right-hand side
//! `1/(alpha - q_fold)`) is folded in as ONE extra signed leaf `(−1, alpha − q_fold)`,
//! so `Σ mult_k/(alpha − fold_k) = 1/(alpha − q_fold)` becomes
//! `Σ mult_k/(alpha − fold_k) − 1/(alpha − q_fold) = 0`, i.e. the root claim is
//! ALWAYS `P = 0`.
//!
//! # GKR: proving the tree top-down
//! The proof reveals only the root `(P, Q)` and, per tree layer, a short sumcheck.
//! A running claim about `(p_ℓ, q_ℓ)` at a random point `ρ_ℓ` is reduced, via the
//! fraction-add relation at the next layer,
//! ```text
//!   p_ℓ(ρ) = Σ_z eq(ρ,z)·[ p_{ℓ+1}(z,0)·q_{ℓ+1}(z,1) + p_{ℓ+1}(z,1)·q_{ℓ+1}(z,0) ]
//!   q_ℓ(ρ) = Σ_z eq(ρ,z)·[ q_{ℓ+1}(z,0)·q_{ℓ+1}(z,1) ]
//! ```
//! (an `eq`-weighted zero-check), batched with a per-layer challenge `λ_ℓ` that
//! combines the p-claim and the q-claim into one sum. Each round polynomial has
//! degree 3 (two multilinear wire factors × one `eq` factor). At the end of the
//! layer's sumcheck the prover reveals the four child evaluations
//! `p_{ℓ+1}(ρ',0), p_{ℓ+1}(ρ',1), q_{ℓ+1}(ρ',0), q_{ℓ+1}(ρ',1)`; a fresh challenge
//! `c_ℓ` collapses each `(·,0),(·,1)` pair into a single evaluation at
//! `ρ_{ℓ+1} = (ρ', c_ℓ)`, giving the running claim for the next layer.
//!
//! The `(z, b)` child convention is MSB-split (LSB-first point convention as
//! everywhere else in this crate): the children of node `z ∈ {0,1}^ℓ` are the
//! layer-`(ℓ+1)` indices `z` (new top bit 0) and `z + 2^ℓ` (new top bit 1). The
//! final layer lands at `r_leaf ∈ Fr^K` (LSB-first), and the verifier reconstructs
//! the leaf claims `p_leaf(r_leaf)`, `q_leaf(r_leaf)` from the committed DATA-column
//! evaluations at `r_leaf` plus public constants — see the leaf tie below.
//!
//! # Leaf tie (the load-bearing correctness subtlety)
//! The GKR core reduces the root claim to a claim about the MLEs of the concrete
//! leaf vectors `p_leaf`, `q_leaf` at a single point `r_leaf`. The verifier must
//! recompute those two evaluations from the committed columns. This is exact ONLY
//! because both `p_leaf` and `q_leaf` are **linear forms** — constant-coefficient
//! combinations of single columns/indicators — so that "MLE of a sum = sum of
//! MLEs". Concretely a leaf value is
//! ```text
//!   q_leaf[i] = alpha − Σ_d beta^d·col_{fold_d}[i] − Σ_t q_fold_t·[i = pos_t]
//!   p_leaf[i] =        Σ_g sign_g·mult_g[i]        + Σ_t sign_t·[i = pos_t]
//!              (+ public block indicators, see below)
//! ```
//! and each source has a MATCHED PAIR (`eval_index` ↔ `eval_mle`): a committed
//! column's value `col[i]` pairs with its opening `col(r_leaf)`, a prefix indicator
//! `[i<b]` pairs with the closed form [`super::argument::chi_n_eval`], and a point
//! indicator `[i=pos]` pairs with [`eq_at_index`] (the Lagrange basis). Because
//! every term is a CONSTANT times exactly ONE such source, `eval_mle(r_leaf)` equals
//! the MLE of the `eval_index` vector at `r_leaf`, exactly.
//!
//! IMPORTANT: a numerator that is a *pointwise product* of two non-constant factors
//! (e.g. `chi_len(i)·mult(i)`, as the reciprocal-era masks were written) does NOT
//! reconstruct as `chi_len(r)·mult(r)` — the MLE of a product is not the product of
//! the MLEs. So any real-length masking MUST be baked into the committed
//! multiplicity column (which is already zero on padding, since a padding leaf is a
//! `p = 0` leaf here), OR expressed as an ADDITIVE public block indicator
//! `sign·([i<hi] − [i<lo])`. Both are supported; a multiplicative
//! `committed × public` mask is intentionally NOT, because it would need an extra
//! leaf-level sumcheck (out of scope for this increment).
//!
//! # Padding annihilation
//! Concatenating leaf groups into a power-of-two vector introduces padding leaves.
//! A padding leaf is `(p = 0, q = alpha)`: it contributes `0/alpha = 0` to the sum.
//! In the tree it multiplies both `P` and `Q` of its subtree by its `q = alpha`,
//! which cancels in the ratio `P/Q` — it is annihilated. A padding leaf given a
//! FORGED nonzero numerator would add a real `≠ 0` term, making the sum nonzero and
//! the root `P ≠ 0` (rejected); and the leaf tie forces `p_leaf(padding) = 0`
//! because no committed column or public term supports a padding position.
//!
//! # Q ≠ 0 (soundness of `P = 0 ⟺ Σ = 0`)
//! `Q = Π_i q_i = Π_i (alpha − c_i)` where each `c_i` is a fold value / public
//! `q_fold` / `0` (padding), all FIXED before `alpha` is drawn (the caller draws
//! `alpha`/`beta` after binding the column commitments). As a polynomial in the
//! random `alpha`, `Q` is monic of degree `2^K`, hence a nonzero polynomial even
//! when denominators DUPLICATE (`c_i = c_j` just gives a repeated root); by
//! Schwartz–Zippel `Pr[Q(alpha) = 0] ≤ 2^K/|F|`, negligible. Likewise every
//! individual `q_i = alpha − c_i ≠ 0` whp (union bound). Thus `P = 0 ⟺ Σ p_i/q_i = 0`
//! except with negligible probability, INDEPENDENT of duplicate denominators.
//!
//! Clean-room from the log-derivative / GKR fraction-tree math (cf. Papini–Haböck
//! "Improving logarithmic derivative lookups using GKR", and the layered-circuit
//! sumcheck of Goldwasser–Kalai–Rothblum); no COSL-licensed code was consulted or
//! copied. This module reuses only the crate's own sumcheck engine helpers.

use std::io;

use dory_pcs::backends::arkworks::ArkFr;
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::transcript::Transcript;

use super::argument::{
    chi_n_eval, eq_eval, eq_table, fold_table, idx_eval, interpolate, powers, Transcript254,
    LABEL_ROUND_CHALLENGE, LABEL_ROUND_POLY,
};
use super::invalid;

#[inline]
fn f0() -> ArkFr {
    ArkFr::zero()
}
#[inline]
fn f1() -> ArkFr {
    ArkFr::one()
}

// Per-layer degree of the fraction-add round polynomial: two multilinear wire
// factors (`p·q`) times one `eq` factor => degree 3, i.e. 4 evaluation points.
const LAYER_DEGREE: usize = 3;

const LABEL_GKR_DOMAIN: &[u8] = b"pow/logup-gkr/domain";
const LABEL_GKR_SHAPE: &[u8] = b"pow/logup-gkr/shape";
const LABEL_ROOT_P: &[u8] = b"pow/logup-gkr/root-p";
const LABEL_ROOT_Q: &[u8] = b"pow/logup-gkr/root-q";
const LABEL_LAMBDA: &[u8] = b"pow/logup-gkr/layer-batch";
const LABEL_CHILD_EVAL: &[u8] = b"pow/logup-gkr/child-eval";
const LABEL_COMBINE: &[u8] = b"pow/logup-gkr/child-combine";

/// Closed-form MLE of the PUBLIC single-position indicator `[i = pos]` at `point`
/// (LSB-first). This is the Lagrange basis `eq(bits(pos), point)` and equals the
/// MLE of the vector that is `1` at index `pos` and `0` elsewhere.
pub fn eq_at_index(pos: usize, point: &[ArkFr]) -> ArkFr {
    let mut acc = f1();
    for (j, rj) in point.iter().enumerate() {
        if (pos >> j) & 1 == 1 {
            acc = acc * *rj;
        } else {
            acc = acc * (f1() - *rj);
        }
    }
    acc
}

// ---------------------------------------------------------------------------
// Core GKR fraction tree (operates purely on leaf p/q vectors)
// ---------------------------------------------------------------------------

/// One tree layer's contribution to the proof: the layer's sumcheck round
/// polynomials (as evaluations at `0..=3`) and the four claimed child evaluations
/// at the sumcheck's final point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GkrLayerProof {
    /// `ℓ` round polynomials for the reduction at layer `ℓ` (empty at the root).
    pub round_polys: Vec<Vec<ArkFr>>,
    pub child_p_left: ArkFr,
    pub child_p_right: ArkFr,
    pub child_q_left: ArkFr,
    pub child_q_right: ArkFr,
}

/// A GKR fraction-sum-zero proof: the root `(P, Q)` plus one [`GkrLayerProof`] per
/// tree layer (`layers[ℓ]` is the reduction from layer `ℓ` to layer `ℓ+1`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GkrFractionProof {
    pub root_p: ArkFr,
    pub root_q: ArkFr,
    pub layers: Vec<GkrLayerProof>,
}

impl GkrFractionProof {
    /// Number of leaf variables `K` (so `2^K` leaves) inferred from the layering.
    pub fn num_vars(&self) -> usize {
        self.layers.len()
    }
}

/// Prove `Σ_i p[i]/q[i] = 0` over the leaf fractions `(p[i], q[i])`.
///
/// `p` and `q` must be the same power-of-two length `2^K`, `K ≥ 1`, and every
/// `q[i]` must be nonzero (the fraction is otherwise undefined). Returns the proof
/// and the final leaf point `r_leaf` (so the caller can open the committed columns
/// there; this module never opens columns itself).
///
/// The caller MUST have already bound the DATA-column commitments and drawn
/// `alpha`/`beta` on `transcript` BEFORE calling, so the leaves are fixed before
/// any per-layer challenge is drawn.
pub fn prove_fraction_tree(
    p: &[ArkFr],
    q: &[ArkFr],
    transcript: &mut Transcript254,
) -> io::Result<(GkrFractionProof, Vec<ArkFr>)> {
    let size = p.len();
    if size != q.len() {
        return Err(invalid("fraction tree p/q length mismatch"));
    }
    if size < 2 || !size.is_power_of_two() {
        return Err(invalid("fraction tree needs a power-of-two count of >= 2 leaves"));
    }
    if q.iter().any(|v| v.is_zero()) {
        return Err(invalid("fraction tree denominator is zero"));
    }
    let k = size.trailing_zeros() as usize;

    // Build the tree bottom-up: lp[d]/lq[d] hold the 2^d fractions at depth d.
    let mut lp: Vec<Vec<ArkFr>> = vec![Vec::new(); k + 1];
    let mut lq: Vec<Vec<ArkFr>> = vec![Vec::new(); k + 1];
    lp[k] = p.to_vec();
    lq[k] = q.to_vec();
    for d in (0..k).rev() {
        let half = 1usize << d;
        let mut pv = vec![f0(); half];
        let mut qv = vec![f0(); half];
        for z in 0..half {
            let (pl, pr) = (lp[d + 1][z], lp[d + 1][z + half]);
            let (ql, qr) = (lq[d + 1][z], lq[d + 1][z + half]);
            qv[z] = ql * qr;
            pv[z] = pl * qr + pr * ql;
        }
        lp[d] = pv;
        lq[d] = qv;
    }

    let root_p = lp[0][0];
    let root_q = lq[0][0];
    transcript.append_field(LABEL_ROOT_P, &root_p);
    transcript.append_field(LABEL_ROOT_Q, &root_q);

    let mut rho: Vec<ArkFr> = Vec::new();
    let mut layers: Vec<GkrLayerProof> = Vec::with_capacity(k);
    for d in 0..k {
        let lambda = transcript.challenge_scalar(LABEL_LAMBDA);
        let child = d + 1;
        let half_full = 1usize << d;

        // Wires over z ∈ {0,1}^d: the low/high halves of the child layer, plus eq.
        let mut eq_tab = eq_table(&rho); // length 2^d
        let mut pl = lp[child][..half_full].to_vec();
        let mut pr = lp[child][half_full..].to_vec();
        let mut ql = lq[child][..half_full].to_vec();
        let mut qr = lq[child][half_full..].to_vec();

        let mut round_polys: Vec<Vec<ArkFr>> = Vec::with_capacity(d);
        let mut rprime: Vec<ArkFr> = Vec::with_capacity(d);
        for _round in 0..d {
            let half = pl.len() / 2;
            let mut poly = vec![f0(); LAYER_DEGREE + 1];
            for (x, slot) in poly.iter_mut().enumerate() {
                let xf = ArkFr::from_u64(x as u64);
                let om = f1() - xf;
                let at = |t: &[ArkFr], i: usize| t[2 * i] * om + t[2 * i + 1] * xf;
                let mut acc = f0();
                for i in 0..half {
                    let e = at(&eq_tab, i);
                    let a = at(&pl, i);
                    let b = at(&pr, i);
                    let c = at(&ql, i);
                    let d_ = at(&qr, i);
                    acc = acc + e * (a * d_ + b * c + lambda * c * d_);
                }
                *slot = acc;
            }
            for coeff in &poly {
                transcript.append_field(LABEL_ROUND_POLY, coeff);
            }
            let ch = transcript.challenge_scalar(LABEL_ROUND_CHALLENGE);
            fold_table(&mut eq_tab, ch);
            fold_table(&mut pl, ch);
            fold_table(&mut pr, ch);
            fold_table(&mut ql, ch);
            fold_table(&mut qr, ch);
            round_polys.push(poly);
            rprime.push(ch);
        }

        let (cpl, cpr, cql, cqr) = (pl[0], pr[0], ql[0], qr[0]);
        transcript.append_field(LABEL_CHILD_EVAL, &cpl);
        transcript.append_field(LABEL_CHILD_EVAL, &cpr);
        transcript.append_field(LABEL_CHILD_EVAL, &cql);
        transcript.append_field(LABEL_CHILD_EVAL, &cqr);
        let cc = transcript.challenge_scalar(LABEL_COMBINE);

        rho = rprime;
        rho.push(cc);
        layers.push(GkrLayerProof {
            round_polys,
            child_p_left: cpl,
            child_p_right: cpr,
            child_q_left: cql,
            child_q_right: cqr,
        });
    }

    Ok((
        GkrFractionProof {
            root_p,
            root_q,
            layers,
        },
        rho,
    ))
}

/// Verify a GKR fraction-sum-zero proof. Checks the root statement `P = 0`, the
/// per-layer sumchecks, and returns the derived leaf point `r_leaf` together with
/// the reduced leaf claims `(p_leaf(r_leaf), q_leaf(r_leaf))` for the caller to tie
/// to the committed columns (see [`check_leaf_tie`]).
pub fn verify_fraction_tree(
    proof: &GkrFractionProof,
    transcript: &mut Transcript254,
) -> io::Result<(Vec<ArkFr>, ArkFr, ArkFr)> {
    let k = proof.layers.len();
    if k == 0 || k >= usize::BITS as usize {
        return Err(invalid("fraction proof has an implausible layer count"));
    }
    // The statement.
    if !proof.root_p.is_zero() {
        return Err(invalid("fraction-sum root numerator P is nonzero"));
    }
    // Q = Π q_i is nonzero whp; a zero here is a degenerate/forged proof.
    if proof.root_q.is_zero() {
        return Err(invalid("fraction-sum root denominator Q is zero"));
    }
    transcript.append_field(LABEL_ROOT_P, &proof.root_p);
    transcript.append_field(LABEL_ROOT_Q, &proof.root_q);

    let mut rho: Vec<ArkFr> = Vec::new();
    let mut p_claim = proof.root_p;
    let mut q_claim = proof.root_q;
    for (d, layer) in proof.layers.iter().enumerate() {
        if layer.round_polys.len() != d {
            return Err(invalid("fraction layer has the wrong number of sumcheck rounds"));
        }
        let lambda = transcript.challenge_scalar(LABEL_LAMBDA);
        let mut current = p_claim + lambda * q_claim;
        let mut rprime: Vec<ArkFr> = Vec::with_capacity(d);
        for round_poly in &layer.round_polys {
            if round_poly.len() != LAYER_DEGREE + 1 {
                return Err(invalid("fraction round polynomial has the wrong degree"));
            }
            if round_poly[0] + round_poly[1] != current {
                return Err(invalid("fraction round consistency check failed"));
            }
            for coeff in round_poly {
                transcript.append_field(LABEL_ROUND_POLY, coeff);
            }
            let ch = transcript.challenge_scalar(LABEL_ROUND_CHALLENGE);
            current = interpolate(round_poly, ch)?;
            rprime.push(ch);
        }

        // Final sumcheck check: the fraction-add expression at the child evals.
        let eq_val = eq_eval(&rho, &rprime);
        let (a, b, c, d_) = (
            layer.child_p_left,
            layer.child_p_right,
            layer.child_q_left,
            layer.child_q_right,
        );
        let expected = eq_val * (a * d_ + b * c + lambda * c * d_);
        if expected != current {
            return Err(invalid("fraction layer final evaluation check failed"));
        }

        transcript.append_field(LABEL_CHILD_EVAL, &a);
        transcript.append_field(LABEL_CHILD_EVAL, &b);
        transcript.append_field(LABEL_CHILD_EVAL, &c);
        transcript.append_field(LABEL_CHILD_EVAL, &d_);
        let cc = transcript.challenge_scalar(LABEL_COMBINE);

        // Collapse the (·,0)/(·,1) child evals into the next running claim at
        // ρ_{ℓ+1} = (ρ', c).
        p_claim = (f1() - cc) * a + cc * b;
        q_claim = (f1() - cc) * c + cc * d_;
        rho = rprime;
        rho.push(cc);
    }

    Ok((rho, p_claim, q_claim))
}

// ---------------------------------------------------------------------------
// Leaf model: linear forms over committed columns + public indicators
// ---------------------------------------------------------------------------

/// A single leaf-value source. Each has a MATCHED PAIR of evaluations: the value at
/// a hypercube index (prover side, [`Src::eval_index`]) and the MLE of that source
/// vector at a point (verifier side, [`Src::eval_mle`]). Because leaves are linear
/// combinations of these, the two agree exactly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Src {
    /// A committed column's value at the row (`col[i]` / opening `col(r)`).
    Column(usize),
    /// The PUBLIC prefix indicator `[i < bound]` (closed form `chi_n_eval(r, bound)`).
    Prefix(usize),
    /// The PUBLIC point indicator `[i = pos]` (closed form `eq_at_index(pos, r)`).
    Index(usize),
    /// The PUBLIC identity ramp `i ↦ i` (closed form [`super::argument::idx_eval`],
    /// `Σ_j r_j·2^j`). The matched pair of the ramp table `[0, 1, .., 2^K − 1]`;
    /// `sort.rs`/`pairs.rs`/`join2.rs` fold this into their denominators.
    Idx,
}

impl Src {
    #[inline]
    fn eval_index(self, i: usize, cols: &[Vec<ArkFr>]) -> ArkFr {
        match self {
            Src::Column(c) => cols[c][i],
            Src::Prefix(b) => {
                if i < b {
                    f1()
                } else {
                    f0()
                }
            }
            Src::Index(p) => {
                if i == p {
                    f1()
                } else {
                    f0()
                }
            }
            Src::Idx => ArkFr::from_u64(i as u64),
        }
    }

    #[inline]
    fn eval_mle(self, r: &[ArkFr], col_evals: &[ArkFr]) -> ArkFr {
        match self {
            Src::Column(c) => col_evals[c],
            Src::Prefix(b) => chi_n_eval(r, b),
            Src::Index(p) => eq_at_index(p, r),
            Src::Idx => idx_eval(r),
        }
    }
}

/// One term `coeff · src` of a leaf linear form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinearTerm {
    pub coeff: ArkFr,
    pub src: Src,
}

/// A leaf linear form `constant + Σ coeff·src`. Used for BOTH the numerator
/// (`p_leaf`) and the denominator (`q_leaf`). Being linear (constant coefficients,
/// one source per term) is exactly what makes the leaf tie an exact MLE
/// reconstruction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearForm {
    pub constant: ArkFr,
    pub terms: Vec<LinearTerm>,
}

impl LinearForm {
    /// The value of this form at hypercube index `i` (prover, from raw columns).
    fn eval_index(&self, i: usize, cols: &[Vec<ArkFr>]) -> ArkFr {
        let mut acc = self.constant;
        for t in &self.terms {
            acc = acc + t.coeff * t.src.eval_index(i, cols);
        }
        acc
    }

    /// The MLE of this form's index-vector at `r` (verifier, from column openings).
    fn eval_mle(&self, r: &[ArkFr], col_evals: &[ArkFr]) -> ArkFr {
        let mut acc = self.constant;
        for t in &self.terms {
            acc = acc + t.coeff * t.src.eval_mle(r, col_evals);
        }
        acc
    }
}

/// The PUBLIC shape of a fraction-sum-zero instance: the leaf variable count, the
/// number of committed columns, and the two leaf forms. Contains no witness data;
/// prover and verifier bind an identical copy to the transcript.
///
/// # Migration obligations (guidance for the Inc 1-5 consumers)
/// This primitive proves ONLY `Σ p_leaf/q_leaf = 0` for leaves that are LINEAR forms
/// over committed columns and public indicators. Anything outside that shape is a
/// soundness obligation the CONSUMER must discharge with its own auxiliary
/// constraints; neither this module nor [`check_leaf_tie`] does it for you:
///
/// - A multiplicative `committed × public-Chi` mask is NOT expressible as a linear
///   leaf (the MLE of a product is not the product of the MLEs). Instead the consumer
///   must commit the column ALREADY zeroed on the padding tail `[real_len, domain)`
///   and add an auxiliary Identity constraint `(1 − Chi_{real_len})·col = 0` — a
///   product term the existing `argument.rs` sumcheck supports — WITHOUT
///   reintroducing a reciprocal. (Affects dicthash / range / membership / join2 /
///   pairs.)
/// - The [`FractionBuilder::add_target`] "fold columns are zero at `pos`" premise is
///   the SAME class of obligation: pin it verifier-side with an auxiliary Identity
///   constraint (the fold columns evaluate to zero at each target position).
///   (Affects dict.)
/// - [`FractionBuilder`] is for the simple single-side cases. A consumer with two
///   committed reciprocal sides / distinct-per-block folds builds a
///   [`FractionInstance`] directly, with a block-DISJOINT column layout (each block's
///   fold columns are zero outside its own block).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FractionInstance {
    pub num_vars: usize,
    pub num_cols: usize,
    pub p_form: LinearForm,
    pub q_form: LinearForm,
}

impl FractionInstance {
    /// The leaf count `2^num_vars` (the power-of-two length every committed column
    /// must match / the fraction leaf vector length the caller zero-extends to).
    pub fn size(&self) -> usize {
        1usize << self.num_vars
    }

    fn validate_shape(&self) -> io::Result<()> {
        if self.num_vars == 0 || self.num_vars >= usize::BITS as usize {
            return Err(invalid("fraction instance has an implausible variable count"));
        }
        let size = self.size();
        for form in [&self.p_form, &self.q_form] {
            for term in &form.terms {
                match term.src {
                    Src::Column(c) if c >= self.num_cols => {
                        return Err(invalid("fraction instance references an unknown column"));
                    }
                    Src::Prefix(b) if b > size => {
                        return Err(invalid("fraction instance prefix bound exceeds the domain"));
                    }
                    Src::Index(p) if p >= size => {
                        return Err(invalid("fraction instance index position exceeds the domain"));
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }
}

fn append_form(transcript: &mut Transcript254, form: &LinearForm) {
    transcript.append_field(LABEL_GKR_SHAPE, &form.constant);
    transcript.append_bytes(LABEL_GKR_SHAPE, &(form.terms.len() as u64).to_le_bytes());
    for term in &form.terms {
        transcript.append_field(LABEL_GKR_SHAPE, &term.coeff);
        let (tag, val): (u8, usize) = match term.src {
            Src::Column(c) => (0, c),
            Src::Prefix(b) => (1, b),
            Src::Index(p) => (2, p),
            Src::Idx => (3, 0),
        };
        transcript.append_bytes(LABEL_GKR_SHAPE, &[tag]);
        transcript.append_bytes(LABEL_GKR_SHAPE, &(val as u64).to_le_bytes());
    }
}

/// Bind the public instance shape to the transcript (domain-separated). Called by
/// prover and verifier BEFORE the tree so every per-layer challenge depends on it.
fn append_instance(transcript: &mut Transcript254, inst: &FractionInstance) {
    transcript.append_bytes(LABEL_GKR_DOMAIN, b"parture/proof-of-woql/logup-gkr/v0");
    transcript.append_bytes(LABEL_GKR_SHAPE, &(inst.num_vars as u64).to_le_bytes());
    transcript.append_bytes(LABEL_GKR_SHAPE, &(inst.num_cols as u64).to_le_bytes());
    append_form(transcript, &inst.p_form);
    append_form(transcript, &inst.q_form);
}

/// Prove `Σ_i p_leaf[i]/q_leaf[i] = 0` for the leaf vectors derived from `inst` and
/// the committed `cols`. Returns the proof and the leaf point `r_leaf`.
///
/// The caller MUST have bound the column commitments and drawn `alpha`/`beta`
/// (which live inside the instance forms' coefficients) on `transcript` first. This
/// function does NOT open the columns — the caller opens them at the returned
/// `r_leaf` via its existing batched Dory opening.
pub fn prove_fraction_sum_zero(
    inst: &FractionInstance,
    cols: &[Vec<ArkFr>],
    transcript: &mut Transcript254,
) -> io::Result<(GkrFractionProof, Vec<ArkFr>)> {
    inst.validate_shape()?;
    if cols.len() != inst.num_cols {
        return Err(invalid("fraction column count does not match the instance"));
    }
    let size = inst.size();
    if cols.iter().any(|c| c.len() != size) {
        return Err(invalid("fraction column length does not match 2^num_vars"));
    }

    let p: Vec<ArkFr> = (0..size).map(|i| inst.p_form.eval_index(i, cols)).collect();
    let q: Vec<ArkFr> = (0..size).map(|i| inst.q_form.eval_index(i, cols)).collect();
    if q.iter().any(|v| v.is_zero()) {
        return Err(invalid("fraction leaf denominator is zero (alpha collided with a fold)"));
    }

    append_instance(transcript, inst);
    prove_fraction_tree(&p, &q, transcript)
}

/// The leaf tie: check the GKR-reduced leaf claims against the committed columns.
/// Reconstructs `p_leaf(r_leaf)` / `q_leaf(r_leaf)` from `col_evals` (the columns'
/// openings at `r_leaf`) plus the public constants in `inst`, and checks equality
/// with the claims returned by [`verify_fraction_tree`].
pub fn check_leaf_tie(
    inst: &FractionInstance,
    col_evals: &[ArkFr],
    p_claim: ArkFr,
    q_claim: ArkFr,
    r_leaf: &[ArkFr],
) -> io::Result<()> {
    if col_evals.len() != inst.num_cols {
        return Err(invalid("leaf tie column-eval count does not match the instance"));
    }
    if r_leaf.len() != inst.num_vars {
        return Err(invalid("leaf tie point has the wrong length"));
    }
    if inst.q_form.eval_mle(r_leaf, col_evals) != q_claim {
        return Err(invalid("leaf tie: reconstructed q_leaf disagrees with the proof"));
    }
    if inst.p_form.eval_mle(r_leaf, col_evals) != p_claim {
        return Err(invalid("leaf tie: reconstructed p_leaf disagrees with the proof"));
    }
    Ok(())
}

/// Bind the public instance and verify the GKR tree, returning the derived leaf
/// point and the reduced `(p_leaf, q_leaf)` claims — WITHOUT opening any columns.
///
/// This is the transcript-owning half of [`verify_fraction_sum_zero`], split out
/// for consumers whose committed columns are NOT the same length as the fraction
/// leaf vector (so they cannot open at `r_leaf` directly and must reconstruct the
/// leaf-MLEs with a dimension adjustment before calling [`check_leaf_tie`]). The
/// dictionary membership consumer opens its length-`D` committed columns at the low
/// `k` bits of `r_leaf` and reconstructs the `2D` leaf-MLEs via a `(1 − r_high)`
/// zero-extension factor; doing the Dory opening itself needs `&mut transcript`,
/// which the [`verify_fraction_sum_zero`] `open_cols` closure does not receive.
pub fn bind_and_verify_fraction_tree(
    inst: &FractionInstance,
    proof: &GkrFractionProof,
    transcript: &mut Transcript254,
) -> io::Result<(Vec<ArkFr>, ArkFr, ArkFr)> {
    inst.validate_shape()?;
    if proof.layers.len() != inst.num_vars {
        return Err(invalid("fraction proof layer count does not match the instance"));
    }
    append_instance(transcript, inst);
    verify_fraction_tree(proof, transcript)
}

/// Verify a fraction-sum-zero proof: bind the instance, verify the GKR tree, obtain
/// the committed columns' evaluations at the derived `r_leaf` via `open_cols` (the
/// caller's Dory opening; in tests, a direct MLE evaluation), and check the leaf
/// tie. `open_cols` MUST return one evaluation per instance column, in order.
pub fn verify_fraction_sum_zero<F>(
    inst: &FractionInstance,
    proof: &GkrFractionProof,
    transcript: &mut Transcript254,
    open_cols: F,
) -> io::Result<()>
where
    F: FnOnce(&[ArkFr]) -> io::Result<Vec<ArkFr>>,
{
    inst.validate_shape()?;
    if proof.layers.len() != inst.num_vars {
        return Err(invalid("fraction proof layer count does not match the instance"));
    }
    append_instance(transcript, inst);
    let (r_leaf, p_claim, q_claim) = verify_fraction_tree(proof, transcript)?;
    let col_evals = open_cols(&r_leaf)?;
    check_leaf_tie(inst, &col_evals, p_claim, q_claim, &r_leaf)
}

// ---------------------------------------------------------------------------
// Ergonomic builder for the common leaf-group shapes
// ---------------------------------------------------------------------------

/// Assembles a [`FractionInstance`] from concatenated leaf GROUPS plus public
/// TARGET leaves, folding in `alpha`/`beta`. Used identically by prover and
/// verifier so both derive the same instance. Groups are laid out contiguously in
/// the order added; public targets follow; the whole is padded (with `p = 0`,
/// `q = alpha` leaves) to the next power of two (min 2).
///
/// The denominator is the UNIFORM fold `q[i] = alpha − Σ_d beta^d·col_{fold_d}[i]`
/// over the global `fold_cols`, adjusted at each target position by its public
/// `q_fold`. For a target's denominator to equal the intended `alpha − q_fold` the
/// fold columns MUST be zero at that target position (and at every padding position).
/// This is a SOUNDNESS OBLIGATION ON THE CONSUMER, discharged verifier-side by an
/// auxiliary Identity constraint (the fold columns evaluate to zero at each
/// target/padding position); it is NOT provided by the prover's honesty, and NOTHING
/// in this module validates it ([`check_leaf_tie`] does not). See
/// [`FractionBuilder::add_target`] for the concrete forge vector.
pub struct FractionBuilder {
    beta: ArkFr,
    fold_cols: Vec<usize>,
    num_cols: usize,
    p_terms: Vec<LinearTerm>,
    q_target_terms: Vec<LinearTerm>,
    constant_q: ArkFr,
    offset: usize,
}

impl FractionBuilder {
    /// Start a builder over `num_cols` committed columns, folding the denominator
    /// over `fold_cols` (in `beta`-power order: `fold_cols[0]` gets `beta^0`).
    pub fn new(alpha: ArkFr, beta: ArkFr, fold_cols: Vec<usize>, num_cols: usize) -> Self {
        FractionBuilder {
            beta,
            fold_cols,
            num_cols,
            p_terms: Vec::new(),
            q_target_terms: Vec::new(),
            constant_q: alpha,
            offset: 0,
        }
    }

    /// Add a group whose numerator is a COMMITTED multiplicity column
    /// (`p[i] += sign·mult[i]` over the block). The column must be zero outside the
    /// block `[offset, offset+block_len)` (validated by the prover).
    pub fn add_mult_group(&mut self, sign: ArkFr, mult_col: usize, block_len: usize) {
        self.p_terms.push(LinearTerm {
            coeff: sign,
            src: Src::Column(mult_col),
        });
        self.offset += block_len;
    }

    /// Add a group whose numerator is a PUBLIC block indicator: each of the group's
    /// first `real_len` rows counts once with `sign` (`p[i] += sign` for
    /// `i ∈ [offset, offset+real_len)`), the remaining `block_len − real_len` rows
    /// are within-block padding (`p = 0`). Expressed as `sign·([i<hi] − [i<lo])`.
    pub fn add_block_group(&mut self, sign: ArkFr, real_len: usize, block_len: usize) {
        let lo = self.offset;
        let hi = self.offset + real_len.min(block_len);
        self.p_terms.push(LinearTerm {
            coeff: sign,
            src: Src::Prefix(hi),
        });
        self.p_terms.push(LinearTerm {
            coeff: f0() - sign,
            src: Src::Prefix(lo),
        });
        self.offset += block_len;
    }

    /// Add one PUBLIC target leaf `(sign, alpha − q_fold)`: numerator `sign` at this
    /// position, and `−q_fold·[i = pos]` folded into the denominator.
    ///
    /// SOUNDNESS OBLIGATION (consumer, verifier-side). The denominator actually
    /// assembled at `pos` is `alpha − Σ_d beta^d·col_{fold_d}[pos] − q_fold`, which
    /// equals the intended `alpha − q_fold` ONLY IF every fold column is zero at
    /// `pos`. The prover is the adversary: a prover that commits a nonzero
    /// `col_{fold_d}[pos]` SHIFTS this pole, and can cancel a genuine dictionary pole
    /// to FORGE membership of a non-member. This module does NOT check it and
    /// [`check_leaf_tie`] does NOT check it — the consumer MUST pin the fold columns
    /// to zero at `pos` with its own auxiliary Identity constraint (the same
    /// product-constraint class the `argument.rs` sumcheck supports).
    pub fn add_target(&mut self, sign: ArkFr, q_fold: ArkFr) {
        let pos = self.offset;
        self.p_terms.push(LinearTerm {
            coeff: sign,
            src: Src::Index(pos),
        });
        self.q_target_terms.push(LinearTerm {
            coeff: f0() - q_fold,
            src: Src::Index(pos),
        });
        self.offset += 1;
    }

    /// The next leaf position that would be assigned (i.e. the current real length).
    pub fn current_offset(&self) -> usize {
        self.offset
    }

    /// Finalize into a [`FractionInstance`]. Pads the real leaf count up to the next
    /// power of two (min 2).
    pub fn finish(self) -> FractionInstance {
        let total = self.offset.max(1);
        let size = total.max(2).next_power_of_two();
        let num_vars = size.trailing_zeros() as usize;

        let mut q_terms: Vec<LinearTerm> = Vec::with_capacity(self.fold_cols.len() + self.q_target_terms.len());
        let beta_powers = powers(self.beta, self.fold_cols.len());
        for (d, &col) in self.fold_cols.iter().enumerate() {
            q_terms.push(LinearTerm {
                coeff: f0() - beta_powers[d],
                src: Src::Column(col),
            });
        }
        q_terms.extend(self.q_target_terms);

        FractionInstance {
            num_vars,
            num_cols: self.num_cols,
            p_form: LinearForm {
                constant: f0(),
                terms: self.p_terms,
            },
            q_form: LinearForm {
                constant: self.constant_q,
                terms: q_terms,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof::argument::mle_full_eval;

    fn fr(v: u64) -> ArkFr {
        ArkFr::from_u64(v)
    }

    fn transcript() -> Transcript254 {
        Transcript254::new(b"parture-logup-gkr-test")
    }

    // ------------------------------------------------------------------
    // Core fraction-tree tests (arbitrary p/q vectors)
    // ------------------------------------------------------------------

    // Random-ish leaves whose fraction-sum is exactly zero (last p is corrected).
    fn zero_sum_leaves(kv: usize, seed: u64) -> (Vec<ArkFr>, Vec<ArkFr>) {
        let n = 1usize << kv;
        let q: Vec<ArkFr> = (0..n)
            .map(|i| fr(seed.wrapping_mul(7).wrapping_add(i as u64 * 3 + 1)))
            .collect();
        let mut p: Vec<ArkFr> = (0..n)
            .map(|i| fr(seed.wrapping_mul(5).wrapping_add(i as u64 * 2 + 1)))
            .collect();
        let mut s = f0();
        for i in 0..n - 1 {
            s = s + p[i] * q[i].inv().unwrap();
        }
        // p_{n-1}/q_{n-1} = -s  =>  p_{n-1} = -s * q_{n-1}
        p[n - 1] = (f0() - s) * q[n - 1];
        (p, q)
    }

    #[test]
    fn core_roundtrip_and_leaf_claims_for_kv_1_to_8() {
        for kv in 1..=8 {
            let (p, q) = zero_sum_leaves(kv, 11 + kv as u64);
            let mut pt = transcript();
            let (proof, r_prover) = prove_fraction_tree(&p, &q, &mut pt).unwrap();
            assert_eq!(proof.num_vars(), kv);

            let mut vt = transcript();
            let (r, p_claim, q_claim) = verify_fraction_tree(&proof, &mut vt).unwrap();
            assert_eq!(r, r_prover);
            assert_eq!(r.len(), kv);
            // The reduced leaf claims are exactly the leaf-vector MLEs at r_leaf.
            assert_eq!(p_claim, mle_full_eval(&p, &r));
            assert_eq!(q_claim, mle_full_eval(&q, &r));
        }
    }

    #[test]
    fn core_rejects_nonzero_sum() {
        let kv = 4;
        let (mut p, q) = zero_sum_leaves(kv, 7);
        p[0] = p[0] + f1(); // break the zero-sum
        let mut pt = transcript();
        let (proof, _r) = prove_fraction_tree(&p, &q, &mut pt).unwrap();
        assert!(!proof.root_p.is_zero(), "sum should be nonzero now");
        let mut vt = transcript();
        assert!(verify_fraction_tree(&proof, &mut vt).is_err());
    }

    #[test]
    fn core_tampered_root_p_is_rejected() {
        let (p, q) = zero_sum_leaves(4, 3);
        let mut pt = transcript();
        let (mut proof, _r) = prove_fraction_tree(&p, &q, &mut pt).unwrap();
        proof.root_p = f1();
        let mut vt = transcript();
        assert!(verify_fraction_tree(&proof, &mut vt).is_err());
    }

    #[test]
    fn core_tampered_root_q_is_rejected() {
        let (p, q) = zero_sum_leaves(4, 4);
        let mut pt = transcript();
        let (mut proof, _r) = prove_fraction_tree(&p, &q, &mut pt).unwrap();
        proof.root_q = proof.root_q + f1();
        let mut vt = transcript();
        assert!(verify_fraction_tree(&proof, &mut vt).is_err());
    }

    #[test]
    fn core_tampered_round_poly_is_rejected() {
        let kv = 4;
        let (p, q) = zero_sum_leaves(kv, 5);
        let mut pt = transcript();
        let (mut proof, _r) = prove_fraction_tree(&p, &q, &mut pt).unwrap();
        // The last layer has kv-1 >= 1 round polynomials; corrupt one coefficient.
        let last = proof.layers.len() - 1;
        proof.layers[last].round_polys[0][2] = proof.layers[last].round_polys[0][2] + f1();
        let mut vt = transcript();
        assert!(verify_fraction_tree(&proof, &mut vt).is_err());
    }

    #[test]
    fn core_tampered_child_eval_is_rejected() {
        let (p, q) = zero_sum_leaves(4, 6);
        let mut pt = transcript();
        let (mut proof, _r) = prove_fraction_tree(&p, &q, &mut pt).unwrap();
        proof.layers[0].child_p_left = proof.layers[0].child_p_left + f1();
        let mut vt = transcript();
        assert!(verify_fraction_tree(&proof, &mut vt).is_err());
    }

    #[test]
    fn core_zero_denominator_is_rejected_by_prover() {
        let (p, mut q) = zero_sum_leaves(3, 9);
        q[2] = f0();
        let mut pt = transcript();
        assert!(prove_fraction_tree(&p, &q, &mut pt).is_err());
    }

    // ------------------------------------------------------------------
    // Instance/builder helpers
    // ------------------------------------------------------------------

    fn alpha_beta() -> (ArkFr, ArkFr) {
        (fr(1_000_003), fr(7_919))
    }

    // Verify an instance end-to-end (honest columns), optionally tampering the proof.
    fn verify_honest(
        inst: &FractionInstance,
        cols: &[Vec<ArkFr>],
        proof: &GkrFractionProof,
    ) -> io::Result<()> {
        let mut vt = transcript();
        verify_fraction_sum_zero(inst, proof, &mut vt, |r| {
            Ok(cols.iter().map(|c| mle_full_eval(c, r)).collect())
        })
    }

    // A dictionary-membership instance: cols = [id, hash, mult]; fold = id + beta*hash;
    // group = the dict rows (mult indicator at `match_idx`); target = -1/(alpha-q_fold).
    // `target_qfold` is the (possibly wrong) public target fold.
    fn dict_case(
        ids: &[u64],
        hashes: &[u64],
        mult_positions: &[usize],
        target_qfold: ArkFr,
        alpha: ArkFr,
        beta: ArkFr,
    ) -> (FractionInstance, Vec<Vec<ArkFr>>) {
        let d = ids.len();
        let mut b = FractionBuilder::new(alpha, beta, vec![0, 1], 3);
        b.add_mult_group(f1(), 2, d);
        b.add_target(f0() - f1(), target_qfold);
        let inst = b.finish();
        let size = inst.size();
        let mut idc = vec![f0(); size];
        let mut hc = vec![f0(); size];
        let mut mult = vec![f0(); size];
        for k in 0..d {
            idc[k] = fr(ids[k]);
            hc[k] = fr(hashes[k]);
        }
        for &pos in mult_positions {
            mult[pos] = mult[pos] + f1();
        }
        (inst, vec![idc, hc, mult])
    }

    fn fold_of(id: u64, hash: u64, beta: ArkFr) -> ArkFr {
        fr(id) + beta * fr(hash)
    }

    #[test]
    fn dict_membership_roundtrip_various_sizes() {
        let (alpha, beta) = alpha_beta();
        for d in 2..=12usize {
            let ids: Vec<u64> = (1..=d as u64).collect();
            let hashes: Vec<u64> = (0..d as u64).map(|k| 100 + k * 13).collect();
            for m in 0..d {
                let qfold = fold_of(ids[m], hashes[m], beta);
                let (inst, cols) = dict_case(&ids, &hashes, &[m], qfold, alpha, beta);
                let mut pt = transcript();
                let (proof, _r) = prove_fraction_sum_zero(&inst, &cols, &mut pt).unwrap();
                assert!(
                    verify_honest(&inst, &cols, &proof).is_ok(),
                    "dict d={d} match={m} should verify"
                );
            }
        }
    }

    #[test]
    fn dict_wrong_target_is_rejected() {
        let (alpha, beta) = alpha_beta();
        let ids = [1u64, 2, 3, 4];
        let hashes = [10u64, 20, 30, 40];
        // mult points at row 2, but the public target fold is row 1's fold.
        let wrong = fold_of(ids[1], hashes[1], beta);
        let (inst, cols) = dict_case(&ids, &hashes, &[2], wrong, alpha, beta);
        let mut pt = transcript();
        let (proof, _r) = prove_fraction_sum_zero(&inst, &cols, &mut pt).unwrap();
        assert!(verify_honest(&inst, &cols, &proof).is_err());
    }

    #[test]
    fn dict_absent_target_is_rejected() {
        let (alpha, beta) = alpha_beta();
        let ids = [1u64, 2, 3];
        let hashes = [10u64, 20, 30];
        // A target fold that is in NO dict row; mult still points somewhere.
        let absent = fold_of(99, 99, beta);
        let (inst, cols) = dict_case(&ids, &hashes, &[1], absent, alpha, beta);
        let mut pt = transcript();
        let (proof, _r) = prove_fraction_sum_zero(&inst, &cols, &mut pt).unwrap();
        assert!(verify_honest(&inst, &cols, &proof).is_err());
    }

    #[test]
    fn injected_extra_pole_is_rejected() {
        // mult has TWO ones at DISTINCT-fold rows; the single target only cancels one.
        let (alpha, beta) = alpha_beta();
        let ids = [1u64, 2, 3, 4];
        let hashes = [10u64, 20, 30, 40];
        let qfold = fold_of(ids[0], hashes[0], beta);
        let (inst, cols) = dict_case(&ids, &hashes, &[0, 2], qfold, alpha, beta);
        let mut pt = transcript();
        let (proof, _r) = prove_fraction_sum_zero(&inst, &cols, &mut pt).unwrap();
        // Sum = 1/(a-f0) + 1/(a-f2) - 1/(a-f0) = 1/(a-f2) != 0.
        assert!(!proof.root_p.is_zero());
        assert!(verify_honest(&inst, &cols, &proof).is_err());
    }

    #[test]
    fn split_multiplicity_across_distinct_folds_is_rejected() {
        // A multiplicity-2 claim split across two DISTINCT dict entries cannot pass a
        // single simple-pole target.
        let (alpha, beta) = alpha_beta();
        let ids = [5u64, 6, 7, 8];
        let hashes = [50u64, 60, 70, 80];
        let qfold = fold_of(ids[1], hashes[1], beta);
        // mult = 1 at row1 (the target) AND 1 at row3 (distinct fold).
        let (inst, cols) = dict_case(&ids, &hashes, &[1, 3], qfold, alpha, beta);
        let mut pt = transcript();
        let (proof, _r) = prove_fraction_sum_zero(&inst, &cols, &mut pt).unwrap();
        assert!(verify_honest(&inst, &cols, &proof).is_err());
    }

    #[test]
    fn equal_fold_split_still_proves_membership() {
        // Splitting a residue across two rows with the SAME fold is a legitimate way
        // to represent multiplicity-1 at that fold value (0.5 + 0.5). This must PASS
        // (the pole is still a single simple pole with residue 1).
        let (alpha, beta) = alpha_beta();
        // Two dict rows with identical (id, hash) -> identical fold.
        let ids = [3u64, 3, 9, 12];
        let hashes = [30u64, 30, 90, 120];
        let qfold = fold_of(ids[0], hashes[0], beta);
        let mut b = FractionBuilder::new(alpha, beta, vec![0, 1], 3);
        b.add_mult_group(f1(), 2, ids.len());
        b.add_target(f0() - f1(), qfold);
        let inst = b.finish();
        let size = inst.size();
        let mut idc = vec![f0(); size];
        let mut hc = vec![f0(); size];
        let mut mult = vec![f0(); size];
        for k in 0..ids.len() {
            idc[k] = fr(ids[k]);
            hc[k] = fr(hashes[k]);
        }
        let half = fr(2).inv().unwrap();
        mult[0] = half;
        mult[1] = half; // total residue 1 at fold(row0)==fold(row1)
        let cols = vec![idc, hc, mult];
        let mut pt = transcript();
        let (proof, _r) = prove_fraction_sum_zero(&inst, &cols, &mut pt).unwrap();
        assert!(verify_honest(&inst, &cols, &proof).is_ok());
    }

    #[test]
    fn single_group_signed_duplicate_denominators_roundtrip() {
        // One group, signed multiplicities +1 / -1 over two EQUAL-fold rows: the sum
        // is 1/(a-v) - 1/(a-v) = 0. Exercises duplicate denominators AND a signed
        // multiplicity column with no target leaf.
        let (alpha, beta) = alpha_beta();
        let mut b = FractionBuilder::new(alpha, beta, vec![0], 2); // cols [val, mult]
        b.add_mult_group(f1(), 1, 4);
        let inst = b.finish();
        let size = inst.size();
        let v = fr(4242);
        let mut val = vec![f0(); size];
        let mut mult = vec![f0(); size];
        val[0] = v;
        val[1] = v; // duplicate denominator
        mult[0] = f1();
        mult[1] = f0() - f1();
        let cols = vec![val, mult];
        let mut pt = transcript();
        let (proof, _r) = prove_fraction_sum_zero(&inst, &cols, &mut pt).unwrap();
        assert!(verify_honest(&inst, &cols, &proof).is_ok());
    }

    #[test]
    fn src_idx_matched_pair_and_composes() {
        // Matched pair: the MLE of the ramp vector [0, 1, .., 2^K − 1] (built via
        // Src::Idx.eval_index) equals Src::Idx.eval_mle (idx_eval) at a random point.
        for k in 1..=7usize {
            let size = 1usize << k;
            let ramp: Vec<ArkFr> = (0..size).map(|i| Src::Idx.eval_index(i, &[])).collect();
            let point: Vec<ArkFr> = (0..k).map(|j| fr(3 * j as u64 + 5)).collect();
            assert_eq!(Src::Idx.eval_mle(&point, &[]), mle_full_eval(&ramp, &point));
        }

        // Composes in a LinearForm: a fraction whose DENOMINATOR uses Src::Idx
        // (`q[i] = alpha − i`) proves and verifies round-trip.
        let (alpha, _beta) = alpha_beta();
        let k = 3usize;
        let size = 1usize << k;
        let q_form = LinearForm {
            constant: alpha,
            terms: vec![LinearTerm { coeff: f0() - f1(), src: Src::Idx }],
        };
        let p_form = LinearForm {
            constant: f0(),
            terms: vec![LinearTerm { coeff: f1(), src: Src::Column(0) }],
        };
        let inst = FractionInstance { num_vars: k, num_cols: 1, p_form, q_form };

        // Pick a multiplicity column so that Σ mult[i]/(alpha − i) = 0.
        let q: Vec<ArkFr> = (0..size).map(|i| alpha - fr(i as u64)).collect();
        let mut mult = vec![f0(); size];
        let mut s = f0();
        for i in 0..size - 1 {
            mult[i] = fr(i as u64 + 1);
            s = s + mult[i] * q[i].inv().unwrap();
        }
        mult[size - 1] = (f0() - s) * q[size - 1];
        let cols = vec![mult];

        let mut pt = transcript();
        let (proof, _r) = prove_fraction_sum_zero(&inst, &cols, &mut pt).unwrap();
        assert!(verify_honest(&inst, &cols, &proof).is_ok());
    }

    // Build a reconcile-shaped instance: present(+) ⊎ removals(+) ≡ additions(-),
    // masked by public block indicators; single fold column carries the fold values.
    // `additions_block` may exceed the real additions to exercise within-block padding.
    fn reconcile_case(
        present: &[u64],
        removals: &[u64],
        additions: &[u64],
        additions_block: usize,
        alpha: ArkFr,
        beta: ArkFr,
    ) -> (FractionInstance, Vec<Vec<ArkFr>>) {
        let mut b = FractionBuilder::new(alpha, beta, vec![0], 1); // cols [fold]
        b.add_block_group(f1(), present.len(), present.len());
        b.add_block_group(f1(), removals.len(), removals.len());
        b.add_block_group(f0() - f1(), additions.len(), additions_block);
        let inst = b.finish();
        let size = inst.size();
        let mut fold = vec![f0(); size];
        let mut off = 0usize;
        for &x in present {
            fold[off] = fr(x);
            off += 1;
        }
        for &x in removals {
            fold[off] = fr(x);
            off += 1;
        }
        for &x in additions {
            fold[off] = fr(x);
            off += 1;
        }
        (inst, vec![fold])
    }

    #[test]
    fn reconcile_masked_signed_roundtrip() {
        let (alpha, beta) = alpha_beta();
        // additions multiset == present ⊎ removals.
        let present = [11u64, 22];
        let removals = [33u64];
        let additions = [11u64, 22, 33];
        let (inst, cols) = reconcile_case(&present, &removals, &additions, additions.len(), alpha, beta);
        let mut pt = transcript();
        let (proof, _r) = prove_fraction_sum_zero(&inst, &cols, &mut pt).unwrap();
        assert!(verify_honest(&inst, &cols, &proof).is_ok());
    }

    #[test]
    fn reconcile_with_within_block_padding_roundtrip() {
        let (alpha, beta) = alpha_beta();
        let present = [11u64, 22];
        let removals = [33u64];
        let additions = [11u64, 22, 33];
        // additions block padded to 4 (one within-block padding row, masked out).
        let (inst, cols) = reconcile_case(&present, &removals, &additions, 4, alpha, beta);
        let mut pt = transcript();
        let (proof, _r) = prove_fraction_sum_zero(&inst, &cols, &mut pt).unwrap();
        assert!(verify_honest(&inst, &cols, &proof).is_ok());
    }

    #[test]
    fn reconcile_missing_addition_is_rejected() {
        let (alpha, beta) = alpha_beta();
        let present = [11u64, 22];
        let removals = [33u64];
        let additions = [11u64, 22]; // drops 33 -> sum != 0
        let (inst, cols) = reconcile_case(&present, &removals, &additions, additions.len(), alpha, beta);
        let mut pt = transcript();
        let (proof, _r) = prove_fraction_sum_zero(&inst, &cols, &mut pt).unwrap();
        assert!(!proof.root_p.is_zero());
        assert!(verify_honest(&inst, &cols, &proof).is_err());
    }

    #[test]
    fn padding_leaf_with_forged_numerator_is_rejected() {
        // Build honest dict leaves, then forge a nonzero numerator at a PADDING leaf.
        // A padding leaf must be p=0; a nonzero one injects a real 1/alpha term.
        let (alpha, beta) = alpha_beta();
        let ids = [1u64, 2, 3]; // d=3 -> +1 target = 4 real leaves -> size 4, no padding
        let hashes = [10u64, 20, 30];
        let qfold = fold_of(ids[0], hashes[0], beta);
        // Use d=3 so total_real=4 == size (no padding). Instead force padding with d=2.
        let ids = [1u64, 2];
        let hashes = [10u64, 20];
        let qfold2 = fold_of(ids[0], hashes[0], beta);
        let (inst, cols) = dict_case(&ids, &hashes, &[0], qfold2, alpha, beta);
        // total_real = 2 (dict) + 1 (target) = 3 -> size 4 -> index 3 is padding.
        let size = inst.size();
        assert_eq!(size, 4);
        let p: Vec<ArkFr> = (0..size).map(|i| inst.p_form.eval_index(i, &cols)).collect();
        let q: Vec<ArkFr> = (0..size).map(|i| inst.q_form.eval_index(i, &cols)).collect();
        assert!(p[3].is_zero(), "padding leaf must be p=0");
        let mut forged = p.clone();
        forged[3] = f1(); // forge a nonzero padding numerator
        let mut pt = transcript();
        let (proof, _r) = prove_fraction_tree(&forged, &q, &mut pt).unwrap();
        assert!(!proof.root_p.is_zero());
        let mut vt = transcript();
        assert!(verify_fraction_tree(&proof, &mut vt).is_err());
        let _ = qfold; // silence unused (kept for readability)
    }

    #[test]
    fn mismatched_leaf_data_col_eval_is_rejected() {
        // An honest proof, but the verifier is fed a tampered column opening: the leaf
        // tie must reject (numerator column mult mismatch).
        let (alpha, beta) = alpha_beta();
        let ids = [1u64, 2, 3, 4];
        let hashes = [10u64, 20, 30, 40];
        let qfold = fold_of(ids[2], hashes[2], beta);
        let (inst, cols) = dict_case(&ids, &hashes, &[2], qfold, alpha, beta);
        let mut pt = transcript();
        let (proof, _r) = prove_fraction_sum_zero(&inst, &cols, &mut pt).unwrap();
        let mut vt = transcript();
        let res = verify_fraction_sum_zero(&inst, &proof, &mut vt, |r| {
            let mut evals: Vec<ArkFr> = cols.iter().map(|c| mle_full_eval(c, r)).collect();
            evals[2] = evals[2] + f1(); // tamper the mult column's opening
            Ok(evals)
        });
        assert!(res.is_err());
    }

    #[test]
    fn tampered_leaf_denominator_col_eval_is_rejected() {
        // Tamper a DENOMINATOR (fold) column opening -> q leaf tie mismatch.
        let (alpha, beta) = alpha_beta();
        let ids = [1u64, 2, 3, 4];
        let hashes = [10u64, 20, 30, 40];
        let qfold = fold_of(ids[1], hashes[1], beta);
        let (inst, cols) = dict_case(&ids, &hashes, &[1], qfold, alpha, beta);
        let mut pt = transcript();
        let (proof, _r) = prove_fraction_sum_zero(&inst, &cols, &mut pt).unwrap();
        let mut vt = transcript();
        let res = verify_fraction_sum_zero(&inst, &proof, &mut vt, |r| {
            let mut evals: Vec<ArkFr> = cols.iter().map(|c| mle_full_eval(c, r)).collect();
            evals[0] = evals[0] + f1(); // tamper the id (fold) column's opening
            Ok(evals)
        });
        assert!(res.is_err());
    }

    #[test]
    fn wrong_alpha_beta_ordering_is_rejected() {
        // Prover uses (alpha, beta); verifier reconstructs with beta/alpha swapped.
        let (alpha, beta) = alpha_beta();
        let ids = [1u64, 2, 3, 4];
        let hashes = [10u64, 20, 30, 40];
        let qfold = fold_of(ids[0], hashes[0], beta);
        let (inst, cols) = dict_case(&ids, &hashes, &[0], qfold, alpha, beta);
        let mut pt = transcript();
        let (proof, _r) = prove_fraction_sum_zero(&inst, &cols, &mut pt).unwrap();

        // Rebuild the instance with alpha/beta swapped (a different public shape).
        let qfold_swapped = fold_of(ids[0], hashes[0], alpha);
        let (inst_bad, _cols) = dict_case(&ids, &hashes, &[0], qfold_swapped, beta, alpha);
        let mut vt = transcript();
        let res = verify_fraction_sum_zero(&inst_bad, &proof, &mut vt, |r| {
            Ok(cols.iter().map(|c| mle_full_eval(c, r)).collect())
        });
        assert!(res.is_err());
    }

    #[test]
    fn tampered_root_p_high_level_is_rejected() {
        let (alpha, beta) = alpha_beta();
        let ids = [1u64, 2, 3, 4];
        let hashes = [10u64, 20, 30, 40];
        let qfold = fold_of(ids[3], hashes[3], beta);
        let (inst, cols) = dict_case(&ids, &hashes, &[3], qfold, alpha, beta);
        let mut pt = transcript();
        let (mut proof, _r) = prove_fraction_sum_zero(&inst, &cols, &mut pt).unwrap();
        proof.root_p = f1();
        assert!(verify_honest(&inst, &cols, &proof).is_err());
    }
}
