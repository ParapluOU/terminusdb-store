//! Shared keyed-join LINK gadget (LogUp-GKR). The self-join (`proof::pairs`) and the
//! binary join (`proof::join2`) each tie their claimed output rows to source relation
//! rows with TWO near-identical masked multiset links (LEFT + RIGHT). This module
//! factors the single-link prove/verify both files reuse (four links total: `l`/`r` ×
//! `pairs`/`join2`), each with its OWN `alpha`/`beta` (the two links are NOT fused).
//!
//! # The link (migrated from committed reciprocals to a block-disjoint GKR tree)
//! One link proves the reciprocal-era log-derivative identity (`masked_multiset_link_claim`
//! in [`super::logup`], candidate side masked by `chi_{m_j}`, source side masked by
//! `chi_{n_src}` AND a committed multiplicity `mult`):
//! ```text
//!   Sum_i chi_{m_j}(i) / (alpha - cand_fold_i)  -  Sum_i chi_{n_src}(i)*mult_i / (alpha - src_fold_i) = 0
//! ```
//! with the index-augmented folds (matching `reciprocal_fold_claim`)
//! ```text
//!   src_fold_i  = Sum_{d<s} beta^d * src_col_d[i]  +  beta^s * i     (idx ramp, power beta^s)
//!   cand_fold_i = Sum_{d<c} beta^d * cand_col_d[i]                   (c = s+1 committed cols)
//! ```
//! The candidate's LAST fold column is a COMMITTED index-pointer (`rho`); the source's
//! last fold slot is the PUBLIC row-index ramp `i`. The match `cand_fold_k == src_fold_a`
//! (at random `alpha`/`beta`, drawn AFTER the data commitments) forces the candidate
//! values equal to source row `a`'s values AND the pointer `rho[k] == a` — the join
//! tie. Because the `idx` ramp makes every `src_fold_a` a DISTINCT simple pole, the
//! source-side residue at each pole uniquely PINS `mult_a` to the true fan-out count
//! (how many output rows point to source row `a`); a fractional/negative/phantom/dropped
//! `mult` leaves an uncancelled pole and the tree root `P != 0` rejects. This is the
//! IDENTICAL soundness the committed-reciprocal era relied on (see `proof::pairs`'s
//! within-group-swap / substitution negatives, unchanged).
//!
//! # Block-disjoint two-side layout (range Part B `DictLookupProof` template)
//! Over a `2*domain` leaf vector (`domain = 2^k`, the shared committed-column length):
//!   * SOURCE rows in the LOW half `[0, domain)` — denominator `alpha - src_fold`, numerator
//!     the committed multiplicity `-mult` (NO `chi` in the leaf; the `chi_{n_src}` mask is a
//!     GAP-1 auxiliary Identity, below);
//!   * CANDIDATE rows in the HIGH half `[domain, 2*domain)` — denominator `alpha - cand_fold`,
//!     numerator the ADDITIVE public prefix block indicator `+[domain <= i < domain + m_j]`.
//! Committed columns stay length `domain`; the prover feeds the tree their block-disjoint
//! zero-extensions and the verifier reconstructs each `2*domain` leaf-MLE from the
//! length-`domain` opening via the top-bit factor (`1 - r_high` low, `r_high` high), which
//! STRUCTURALLY forces each side's fold columns to zero in the other block (no cross-leak).
//!
//! ## The source index ramp — a PUBLIC half-scoped pseudo-column
//! `src_fold`'s `beta^s * i` term is the ONE structure range Part B / `membership` lack.
//! The sort gadget's single GLOBAL `Src::Idx` ramp works there ONLY because BOTH its blocks
//! carry an index (`i` and `i+1`); here the CANDIDATE fold uses a COMMITTED pointer instead,
//! so a global `Src::Idx` would pollute the candidate half with an `i`-ramp that no linear
//! form can cancel. The correct object is the "low-half ramp" `L(i) = i*[i<domain]`, whose
//! MLE is `(1 - r_high)*idx_eval(r_low)` — NOT `Src::Idx.eval_mle` (the global ramp). Rather
//! than extend the audited primitive, it is carried as a PUBLIC pseudo-column: the prover
//! feeds `[0,1,..,domain-1, 0,..,0]` into the tree; the verifier reconstructs its leaf-MLE
//! from the public closed form `(1 - r_high)*idx_eval(r_low)` and NEVER commits or opens it
//! (identical in spirit to how `Src::Idx`/`Src::Prefix` are public, matched-pair sources
//! evaluated inline — just routed through the column channel because it needs half-scoping).
//! The leaf tie forces the fed ramp to equal the true ramp (Schwartz-Zippel), so a forged
//! ramp cannot pass; the `(1 - r_high)` factor structurally keeps it out of the candidate
//! block.
//!
//! # GAP-1 (source padding tail), closed with a co-running auxiliary Identity
//! `chi_{n_src}` is an ARBITRARY (non-power-of-two) boundary and `mult` is COMMITTED, so the
//! `chi_{n_src}*mult` mask is a GAP-1 `committed x public-Chi` product — NOT a linear GKR leaf.
//! Exactly as `membership`/`range` Part B, it is closed by a co-running Identity over the SAME
//! committed `mult`:
//! ```text
//!   (1 - chi_{n_src}(x)) * mult(x) = 0   for every x
//! ```
//! proved with the existing [`super::argument`] sumcheck (a product term — NO reciprocal
//! reintroduced), forcing `mult[x] = 0` at every `x >= n_src` UNCONDITIONALLY. Its `mult`
//! opening binds the IDENTICAL `com_mult` the tree opens, on the SAME transcript. Without it
//! a prover could stash fan-out weight in the source padding tail (a substitution forge onto
//! a padding source row); the Identity rejects it (see `pairs`'s padding-substitution tests).
//!
//! Clean-room from the log-derivative / GKR fraction-tree math; no COSL code.

use std::io;

use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;

use super::argument::{
    idx_eval, mle_full_eval, powers, prove_sumcheck, verify_column_openings, verify_sumcheck,
    ClaimShape, ColumnOpening, Expr, Operand, StoreProverContext, SumcheckClaim, SumcheckProof,
    SumcheckShape, Term, Transcript254,
};
use super::invalid;
use super::logup_gkr::{
    bind_and_verify_fraction_tree, check_leaf_tie, prove_fraction_sum_zero, FractionInstance,
    GkrFractionProof, LinearForm, LinearTerm, Src,
};

#[inline]
fn f0() -> ArkFr {
    ArkFr::zero()
}
#[inline]
fn f1() -> ArkFr {
    ArkFr::one()
}

/// One migrated keyed-join link: the block-disjoint GKR fraction tree proving
/// `Sum chi_{m_j}/(alpha - cand_fold) - Sum mult/(alpha - src_fold) = 0`, the committed
/// columns' leaf evaluations `[src.., mult, cand..]` at `r_low` with one batched Dory
/// opening, and the co-running GAP-1 auxiliary Identity `(1 - chi_{n_src})*mult = 0`
/// (its own sumcheck + `mult` opening, both bound to the SAME `com_mult`). NO reciprocal
/// column exists.
#[derive(Clone, Debug)]
pub struct JoinLinkProof {
    pub gkr: GkrFractionProof,
    /// Evaluations of the committed columns `[src_0..src_{s-1}, mult, cand_0..cand_{c-1}]`
    /// (`c = s + 1`; the public index ramp is NOT here — it is verifier-reconstructed).
    pub col_evals: Vec<ArkFr>,
    pub opening: ColumnOpening,
    pub id_sumcheck: SumcheckProof,
    pub id_opening: ColumnOpening,
}

/// Zero-extend a length-`size` column into the LOW half of a `2*size` leaf vector.
/// MLE at `(r_low, r_high)` is `(1 - r_high)*MLE_size(col, r_low)`.
fn low_half(col: &[ArkFr], leaf_count: usize) -> Vec<ArkFr> {
    let mut ext = vec![f0(); leaf_count];
    ext[..col.len()].copy_from_slice(col);
    ext
}

/// Zero-extend a length-`size` column into the HIGH half of a `2*size` leaf vector.
/// MLE at `(r_low, r_high)` is `r_high*MLE_size(col, r_low)`.
fn high_half(col: &[ArkFr], leaf_count: usize) -> Vec<ArkFr> {
    let mut ext = vec![f0(); leaf_count];
    ext[leaf_count / 2..].copy_from_slice(col);
    ext
}

/// The PUBLIC block-disjoint fraction-sum-zero instance for ONE link. Column layout
/// (built IDENTICALLY by prover and verifier): `[src_0..src_{s-1}, mult, ramp,
/// cand_0..cand_{c-1}]` with `c = s + 1`, so `num_cols = 2s + 3`.
///   * `q_leaf = alpha - src_fold` (low half) and `alpha - cand_fold` (high half),
///     block-disjoint (each side's columns are zero in the other block). The source
///     fold folds `[src_0..src_{s-1}, ramp]` at powers `beta^0..beta^s`; the candidate
///     fold folds `[cand_0..cand_{c-1}]` at powers `beta^0..beta^s` (matching slots).
///   * `p_leaf = -mult` (low, committed) `+ ([i < domain+m_j] - [i < domain])` (high,
///     public prefix block); padding rows self-annihilate (`p = 0`).
/// The `ramp` column (index `s+1`) is the PUBLIC low-half index ramp — see the module doc.
fn join_link_instance(alpha: ArkFr, beta: ArkFr, s: usize, m_j: usize, k: usize) -> FractionInstance {
    let domain = 1usize << k;
    let mult_col = s;
    let ramp_col = s + 1;
    let cand_base = s + 2;
    let c = s + 1;
    let num_cols = s + 2 + c; // 2s + 3
    let beta_pows = powers(beta, s + 1); // beta^0 .. beta^s

    let mut q_terms: Vec<LinearTerm> = Vec::with_capacity(s + 1 + c);
    // Source fold (low half): -Sum_{d<s} beta^d * src_col_d - beta^s * ramp.
    for (d, bp) in beta_pows.iter().take(s).enumerate() {
        q_terms.push(LinearTerm { coeff: f0() - *bp, src: Src::Column(d) });
    }
    q_terms.push(LinearTerm { coeff: f0() - beta_pows[s], src: Src::Column(ramp_col) });
    // Candidate fold (high half): -Sum_{d<c} beta^d * cand_col_d.
    for (d, bp) in beta_pows.iter().enumerate() {
        q_terms.push(LinearTerm { coeff: f0() - *bp, src: Src::Column(cand_base + d) });
    }

    // p = -mult (low, committed) + ([i < domain+m_j] - [i < domain]) (high, public block).
    let p_terms = vec![
        LinearTerm { coeff: f0() - f1(), src: Src::Column(mult_col) },
        LinearTerm { coeff: f1(), src: Src::Prefix(domain + m_j) },
        LinearTerm { coeff: f0() - f1(), src: Src::Prefix(domain) },
    ];

    FractionInstance {
        num_vars: k + 1,
        num_cols,
        p_form: LinearForm { constant: f0(), terms: p_terms },
        q_form: LinearForm { constant: alpha, terms: q_terms },
    }
}

/// The PUBLIC GAP-1 auxiliary Identity `(1 - chi_{n_src}(x)) * mult(x) = 0` over the
/// length-`domain = 2^k` committed multiplicity column (referenced as `Column(0)`),
/// forcing `mult[x] = 0` at every `x >= n_src`. Written as `+mult - chi_{n_src}*mult`.
fn mult_mask_identity_shape(k: usize, n_src: usize) -> SumcheckShape {
    SumcheckShape {
        num_vars: k,
        num_columns: 1,
        chi_bounds: vec![n_src],
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

/// Prove one keyed-join link. `src_cols` (`s` committed source value columns, fold powers
/// `beta^0..beta^{s-1}`), `mult` (committed source multiplicity), and `cand_cols`
/// (`c = s + 1` committed candidate columns, fold powers `beta^0..beta^s`) must all share
/// the same power-of-two `domain = mult.len()`. `alpha`/`beta` must already be drawn on
/// `transcript` AFTER every data-column commitment (the primitive's caller contract).
#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_join_link_with_context(
    context: &StoreProverContext<'_>,
    src_cols: &[&[ArkFr]],
    mult: &[ArkFr],
    cand_cols: &[&[ArkFr]],
    n_src: usize,
    m_j: usize,
    alpha: ArkFr,
    beta: ArkFr,
    transcript: &mut Transcript254,
) -> io::Result<JoinLinkProof> {
    let s = src_cols.len();
    let c = cand_cols.len();
    if c != s + 1 {
        return Err(invalid("join link candidate arity must be source arity + 1"));
    }
    let domain = mult.len();
    if !domain.is_power_of_two() || domain < 2 {
        return Err(invalid("join link domain must be a power of two >= 2"));
    }
    for col in src_cols.iter().chain(cand_cols.iter()) {
        if col.len() != domain {
            return Err(invalid("join link column length mismatch"));
        }
    }
    if n_src > domain || m_j > domain {
        return Err(invalid("join link real length exceeds the domain"));
    }
    let k = domain.trailing_zeros() as usize;
    let leaf_count = 2 * domain;

    // Block-disjoint zero-extensions in instance-column order [src.., mult, ramp, cand..].
    let inst = join_link_instance(alpha, beta, s, m_j, k);
    let mut extended: Vec<Vec<ArkFr>> = Vec::with_capacity(inst.num_cols);
    for col in src_cols {
        extended.push(low_half(col, leaf_count));
    }
    extended.push(low_half(mult, leaf_count));
    // The PUBLIC index ramp [0,1,..,domain-1] zero-extended into the LOW half.
    let mut ramp = vec![f0(); leaf_count];
    for (i, slot) in ramp.iter_mut().take(domain).enumerate() {
        *slot = ArkFr::from_u64(i as u64);
    }
    extended.push(ramp);
    for col in cand_cols {
        extended.push(high_half(col, leaf_count));
    }

    let (gkr, r_leaf) = prove_fraction_sum_zero(&inst, &extended, transcript)?;

    // One batched Dory opening of the COMMITTED columns [src.., mult, cand..] at r_low
    // (the public ramp is NOT opened).
    let r_low = &r_leaf[..k];
    let mut columns: Vec<Vec<ArkFr>> = Vec::with_capacity(2 * s + 2);
    for col in src_cols {
        columns.push(col.to_vec());
    }
    columns.push(mult.to_vec());
    for col in cand_cols {
        columns.push(col.to_vec());
    }
    let col_evals: Vec<ArkFr> = columns.iter().map(|col| mle_full_eval(col, r_low)).collect();
    let opening = context.prove_column_openings(&columns, r_low, transcript)?;

    // GAP-1 auxiliary Identity (1 - chi_{n_src})*mult = 0 over the SAME committed mult.
    let id_shape = mult_mask_identity_shape(k, n_src);
    let mult_col = vec![mult.to_vec()];
    let (id_sumcheck, r_id, _evals) = prove_sumcheck(&id_shape, &mult_col, transcript)?;
    let id_opening = context.prove_column_openings(&mult_col, &r_id, transcript)?;

    Ok(JoinLinkProof {
        gkr,
        col_evals,
        opening,
        id_sumcheck,
        id_opening,
    })
}

/// Verify one keyed-join link against the committed columns. `src_coms` (`s`), `com_mult`,
/// and `cand_coms` (`c = s + 1`) commit the length-`domain = 2^k` columns; `alpha`/`beta`
/// must be drawn identically to the prover. The GKR tree's `k + 1` layers re-anchor the
/// column length; the committed-column opening authenticates every value/pointer.
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_join_link(
    src_coms: &[ArkGT],
    com_mult: &ArkGT,
    cand_coms: &[ArkGT],
    n_src: usize,
    m_j: usize,
    alpha: ArkFr,
    beta: ArkFr,
    k: usize,
    proof: &JoinLinkProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    let s = src_coms.len();
    let c = cand_coms.len();
    if c != s + 1 {
        return Err(invalid("join link candidate arity must be source arity + 1"));
    }
    let domain = 1usize << k;
    if n_src > domain || m_j > domain {
        return Err(invalid("join link real length exceeds the domain"));
    }
    if proof.col_evals.len() != 2 * s + 2 {
        return Err(invalid("join link column-evaluation count mismatch"));
    }

    // Bind the public instance and verify the GKR tree (checks its k+1 layers => cross-checks
    // this link's domain against `k`). Then the leaf tie ties it to the committed columns.
    let inst = join_link_instance(alpha, beta, s, m_j, k);
    let (r_leaf, p_claim, q_claim) = bind_and_verify_fraction_tree(&inst, &proof.gkr, transcript)?;

    let r_low = &r_leaf[..k];
    let r_high = r_leaf[k];

    // Open the committed columns [src.., mult, cand..] at r_low, bound to their commitments.
    let mut coms: Vec<ArkGT> = Vec::with_capacity(2 * s + 2);
    coms.extend_from_slice(src_coms);
    coms.push(*com_mult);
    coms.extend_from_slice(cand_coms);
    verify_column_openings(&coms, r_low, &proof.col_evals, &proof.opening, transcript)?;

    // Reconstruct the 2*domain leaf-MLEs from the length-domain openings via the top-bit
    // factor: source cols + mult + ramp in the LOW half (1 - r_high), candidate cols in the
    // HIGH half (r_high). The ramp is the PUBLIC index [0,1,..,domain-1]; its low-half
    // leaf-MLE is (1 - r_high)*idx_eval(r_low), reconstructed WITHOUT a commitment/opening
    // (the leaf tie then forces the prover's fed ramp to equal the true ramp).
    let one_minus_high = f1() - r_high;
    let mut leaf_evals: Vec<ArkFr> = Vec::with_capacity(inst.num_cols);
    for d in 0..s {
        leaf_evals.push(one_minus_high * proof.col_evals[d]);
    }
    leaf_evals.push(one_minus_high * proof.col_evals[s]); // mult (low)
    leaf_evals.push(one_minus_high * idx_eval(r_low)); // ramp (public, low)
    for d in 0..c {
        leaf_evals.push(r_high * proof.col_evals[s + 1 + d]); // candidate (high)
    }
    check_leaf_tie(&inst, &leaf_evals, p_claim, q_claim, &r_leaf)?;

    // GAP-1 auxiliary Identity (1 - chi_{n_src})*mult = 0 over the SAME com_mult. Its mult
    // opening binds to com_mult on this transcript, so the multiplicity the tree used is
    // EXACTLY the one pinned to zero on [n_src, domain) — a forged padding multiplicity is
    // rejected UNCONDITIONALLY.
    let id_shape = mult_mask_identity_shape(k, n_src);
    let (r_id, id_evals) = verify_sumcheck(&id_shape, &proof.id_sumcheck, transcript)?;
    verify_column_openings(&[*com_mult], &r_id, &id_evals, &proof.id_opening, transcript)?;
    Ok(())
}
