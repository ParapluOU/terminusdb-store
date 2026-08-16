//! Membership-with-multiplicity (Phase 2c-2b), the second completeness primitive.
//!
//! Goal: given committed value columns `U` (real length `n_U`) and `Of` (real
//! length `n_F`) and a committed multiplicity column `w` (length `n_U`), prove
//! `Of` is EXACTLY the multiset image of `U` under `w` — every `Of` value appears
//! in `U`, and `w[i] = #{ a : Of_a = U[i] }`. This is completeness claim (i);
//! claim (ii) is the symmetric `Sf/U/w_R` with fresh challenges.
//!
//! # The gadget (LogUp-GKR, block-disjoint two-side layout + GAP-1 auxiliary Identity)
//! The log-derivative identity pinning `w[i]` to the true count of `U[i]` in `Of` is
//! ```text
//!   Sum_{i<n_U} w[i]/(alpha - U[i]) - Sum_{a<n_F} 1/(alpha - Of[a]) = 0
//! ```
//! (Habock's pole/residue identity). Values are single field elements, so no `beta`
//! fingerprint is needed — the fold is ONE value column per side.
//!
//! This is proved with the GKR fraction tree ([`super::logup_gkr`]) over a `2*domain`
//! leaf vector (`domain = 2^k` the shared value-column length) laid out BLOCK-DISJOINT:
//!   * the UNIVERSE rows in the LOW half `[0, domain)` — denominator `alpha - U[i]`,
//!     numerator the committed multiplicity `+w[i]`;
//!   * the IMAGE rows in the HIGH half `[domain, 2*domain)` — denominator
//!     `alpha - Of[a]`, numerator the ADDITIVE public block indicator
//!     `-[domain <= i < domain + n_F]`.
//! NO reciprocal column is committed anywhere (the old `rec_u`/`rec_of` are gone). The
//! committed DATA columns `[U, Of, w]` stay length `domain`; the prover feeds the tree
//! their zero-extensions — `U`/`w` zero on the high half, `Of` zero on the low half —
//! and the verifier reconstructs each `2*domain` leaf-MLE from the length-`domain`
//! opening via the top-bit factor (`(1 - r_high)` for the low half, `r_high` for the
//! high half). This STRUCTURALLY forces each side's fold columns to zero in the OTHER
//! block (the same enforcement `permutation` uses), so the two sides cannot leak.
//!
//! # GAP-1 (universe padding tail), closed UNCONDITIONALLY with a co-running Identity
//! The reciprocal-era `Sum chi_{n_U}*w*rec_u` masked the universe padding tail
//! `[n_U, domain)` out of the link. Its purpose is soundness-critical: without it a
//! prover can STASH a duplicate real key `v` in `U`'s committed padding
//! (`U[i]=v, i >= n_U`) and put a nonzero `w[i]` there, letting the padding pole
//! `+w[i]/(alpha - v)` cover a genuine `Of` pole at `alpha - v` — an UNDERCOUNT forge
//! (documented by `duplicate_key_stashed_in_u_padding_tail_is_rejected`). Unlike the
//! dict-hash padding pole (which sits at `alpha - 0` and is unmatched because every
//! real query fold is nonzero), THIS forged pole is at `alpha - v` with `v` a real,
//! MATCHABLE value, so no "no real row has a pole at alpha" invariant closes it; and
//! the boundary `n_U` is ARBITRARY (not a power of two), so a structural
//! zero-extension of `w` (dict's `mult` trick) cannot reproduce the mask either.
//!
//! So GAP-1 is closed here by a CO-RUNNING auxiliary Identity over the SAME committed
//! `w`:
//! ```text
//!   (1 - chi_{n_U}(x)) * w(x) = 0   for every x
//! ```
//! proved via the existing [`super::argument`] sumcheck (the `chi*w` product is a term
//! that engine already supports — NO reciprocal is reintroduced). As an Identity claim
//! it forces `w[x] = 0` at EVERY `x >= n_U`, for ARBITRARY `n_U`, unconditionally. The
//! Identity's `w` opening binds to the SAME `com_w` on the SAME transcript as the GKR
//! leaf tie, so the multiplicity the tree used is exactly the one pinned to zero on the
//! tail — a forged padding `w` is rejected without any caller invariant. `com_w` stays
//! committed at the full `domain`, in lockstep with the count gadget's shared `com_w`.
//!
//! IMPORTANT (documented by a test): this does NOT assume `U` is distinct. If `U`
//! has a duplicated value, `w` may split its count among the copies and still
//! verify — the residue only pins the SUM of multiplicities over equal `U` values.
//! Distinctness is Phase 2c-2a's job; the count identity (2c-2c), which relies on
//! it, is what closes this. Clean-room from the math; no COSL code.

use std::io;

use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::transcript::Transcript;

use super::argument::{
    mle_full_eval, prove_sumcheck, verify_column_openings, verify_sumcheck, ClaimShape,
    ColumnOpening, Expr, Operand, StoreProverContext, SumcheckClaim, SumcheckProof, SumcheckShape,
    Term, Transcript254,
};
#[cfg(test)]
use super::argument::commit_column;
use super::invalid;
use super::logup_gkr::{
    bind_and_verify_fraction_tree, check_leaf_tie, prove_fraction_sum_zero, FractionInstance,
    GkrFractionProof, LinearForm, LinearTerm, Src,
};

// Column ordering for the batched membership DATA columns (each committed at the shared
// `2^k` domain). The reciprocal witnesses `rec_u`/`rec_of` are GONE: the LogUp-GKR
// migration proves the same log-derivative sum via a block-disjoint fraction tree over
// uncommitted intermediates, so no reciprocal column is committed anywhere.
const COL_U: usize = 0;
const COL_OF: usize = 1;
const COL_W: usize = 2;
const NUM_COLUMNS: usize = 3;

const LABEL_ALPHA: &[u8] = b"pow/membership/reciprocal-alpha";

#[inline]
fn f0() -> ArkFr {
    ArkFr::zero()
}
#[inline]
fn f1() -> ArkFr {
    ArkFr::one()
}

/// A membership-with-multiplicity proof: the block-disjoint GKR fraction-sum-zero proof
/// for `Sum_{i<n_U} w/(alpha - U) - Sum_{a<n_F} 1/(alpha - Of) = 0`, the claimed
/// evaluations of the three DATA columns at `r_low` (the low `k` bits of the GKR leaf
/// point), one batched Dory opening of those columns at `r_low`, and the co-running
/// GAP-1 Identity `(1 - chi_{n_U})*w = 0` (its own sumcheck + `w` opening, both bound to
/// `com_w` on the shared transcript). NO reciprocal (`rec_u`/`rec_of`) column exists.
#[derive(Clone, Debug)]
pub struct MembershipProof {
    pub gkr: GkrFractionProof,
    pub col_evals: Vec<ArkFr>,
    pub opening: ColumnOpening,
    pub id_sumcheck: SumcheckProof,
    pub id_opening: ColumnOpening,
}

/// The common power-of-two domain (`>= 2`) for value columns of `n_u`/`n_of` rows.
pub(crate) fn membership_domain(n_u: usize, n_of: usize) -> io::Result<usize> {
    n_u.max(n_of)
        .checked_next_power_of_two()
        .ok_or_else(|| invalid("membership domain size overflow"))
        .map(|d| d.max(2))
}

fn padded(column: &[ArkFr], n: usize) -> Vec<ArkFr> {
    let mut out = column.to_vec();
    out.resize(n, f0());
    out
}

/// Zero-extend a length-`size` column into the LOW half of a `2*size` leaf vector; the
/// high half stays zero. MLE at `(r_low, r_high)` is `(1 - r_high)*MLE_size(col, r_low)`.
fn low_half(col: &[ArkFr], leaf_count: usize) -> Vec<ArkFr> {
    let mut ext = vec![f0(); leaf_count];
    ext[..col.len()].copy_from_slice(col);
    ext
}

/// Zero-extend a length-`size` column into the HIGH half of a `2*size` leaf vector; the
/// low half stays zero. MLE at `(r_low, r_high)` is `r_high*MLE_size(col, r_low)`.
fn high_half(col: &[ArkFr], leaf_count: usize) -> Vec<ArkFr> {
    let mut ext = vec![f0(); leaf_count];
    ext[leaf_count / 2..].copy_from_slice(col);
    ext
}

/// Honest multiplicities `w[i] = #{ a < n_F : Of_a = U[i] }`. For distinct `U`
/// this is the unique correct `w`; for duplicated `U` the caller must instead
/// choose a `w` whose per-value SUM matches (see the module doc / duplicate test).
// Used by the tests here; the honest-witness form the self-join integration relies
// on (see `proof::join::compute_self_join`).
#[allow(dead_code)]
pub(crate) fn count_multiplicities(u: &[ArkFr], of: &[ArkFr]) -> Vec<ArkFr> {
    u.iter()
        .map(|value| {
            let count = of.iter().filter(|entry| *entry == value).count() as u64;
            ArkFr::from_u64(count)
        })
        .collect()
}

/// The PUBLIC block-disjoint fraction-sum-zero instance over a length-`2*domain` leaf
/// vector (`domain = 2^k`): universe rows in the low half `[0, domain)`, image rows in
/// the high half `[domain, 2*domain)`. Built IDENTICALLY by prover and verifier.
///   * denominator `q_leaf = alpha - U[i] - Of[i]` — block-disjoint, so it is
///     `alpha - U` on the low half and `alpha - Of` on the high half (single `alpha`:
///     values are single field elements, no `beta` fingerprint);
///   * numerator `p_leaf = +w[i] - [domain <= i < domain + n_F]`: each universe row
///     counts `+w` (committed multiplicity, low half), each real image row `-1` (public
///     prefix block indicator, high half); padding rows self-annihilate (`p = 0`).
fn membership_fraction_instance(alpha: ArkFr, n_of: usize, k: usize) -> FractionInstance {
    let domain = 1usize << k;
    let q_terms = vec![
        // Universe side (low half): alpha - U[i].
        LinearTerm { coeff: f0() - f1(), src: Src::Column(COL_U) },
        // Image side (high half): alpha - Of[a].
        LinearTerm { coeff: f0() - f1(), src: Src::Column(COL_OF) },
    ];
    // -[domain <= i < domain + n_of] = -([i < domain + n_of] - [i < domain]).
    let p_terms = vec![
        LinearTerm { coeff: f1(), src: Src::Column(COL_W) },
        LinearTerm { coeff: f0() - f1(), src: Src::Prefix(domain + n_of) },
        LinearTerm { coeff: f1(), src: Src::Prefix(domain) },
    ];
    FractionInstance {
        num_vars: k + 1,
        num_cols: NUM_COLUMNS,
        p_form: LinearForm { constant: f0(), terms: p_terms },
        q_form: LinearForm { constant: alpha, terms: q_terms },
    }
}

/// The PUBLIC GAP-1 auxiliary Identity `(1 - chi_{n_U}(x)) * w(x) = 0` over the length-
/// `domain = 2^k` committed multiplicity column (referenced as `Column(0)`), forcing
/// `w[x] = 0` at every `x >= n_U`. Written as `+w - chi_{n_U}*w` (a product term the
/// [`super::argument`] sumcheck supports — no reciprocal). Built IDENTICALLY by prover
/// and verifier.
fn universe_mask_identity_shape(k: usize, n_u: usize) -> SumcheckShape {
    SumcheckShape {
        num_vars: k,
        num_columns: 1,
        chi_bounds: vec![n_u],
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

/// Prove `Of` is the multiset image of `U` under `w`. `com_u`/`com_of`/`com_w`
/// must commit `U`/`Of`/`w` padded to `membership_domain(n_U, n_F)`.
pub fn prove_membership_with_mult(
    u: &[ArkFr],
    of: &[ArkFr],
    w: &[ArkFr],
    com_u: &ArkGT,
    com_of: &ArkGT,
    com_w: &ArkGT,
    transcript: &mut Transcript254,
) -> io::Result<MembershipProof> {
    prove_membership_with_mult_with_context(
        &StoreProverContext::cpu(), u, of, w, com_u, com_of, com_w, transcript,
    )
}

/// As [`prove_membership_with_mult`], with an explicit prover-arithmetic context.
#[allow(clippy::too_many_arguments)]
pub fn prove_membership_with_mult_with_context(
    context: &StoreProverContext<'_>,
    u: &[ArkFr],
    of: &[ArkFr],
    w: &[ArkFr],
    com_u: &ArkGT,
    com_of: &ArkGT,
    com_w: &ArkGT,
    transcript: &mut Transcript254,
) -> io::Result<MembershipProof> {
    let domain = membership_domain(u.len(), of.len())?;
    prove_membership_with_mult_in_domain_with_context(
        context, u, of, w, com_u, com_of, com_w, domain, transcript,
    )
}

/// Like [`prove_membership_with_mult`] but pads onto a caller-chosen power-of-two
/// `domain` (`>= membership_domain(n_U, n_F)`) instead of the natural one, so the
/// full self-join integration (Phase 2c-2e) can share ONE `Com(U)` /
/// `Com(Of)` / `Com(Sf)` / `Com(w)` with the other sub-proofs. Real lengths
/// `n_U`/`n_F` still drive the block indicators / the GAP-1 Identity, so the residue
/// ranges only over the true `Of` rows and `w` is pinned to zero on `[n_U, domain)`.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn prove_membership_with_mult_in_domain(
    u: &[ArkFr],
    of: &[ArkFr],
    w: &[ArkFr],
    com_u: &ArkGT,
    com_of: &ArkGT,
    com_w: &ArkGT,
    domain: usize,
    transcript: &mut Transcript254,
) -> io::Result<MembershipProof> {
    prove_membership_with_mult_in_domain_with_context(
        &StoreProverContext::cpu(), u, of, w, com_u, com_of, com_w, domain, transcript,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_membership_with_mult_in_domain_with_context(
    context: &StoreProverContext<'_>,
    u: &[ArkFr],
    of: &[ArkFr],
    w: &[ArkFr],
    com_u: &ArkGT,
    com_of: &ArkGT,
    com_w: &ArkGT,
    domain: usize,
    transcript: &mut Transcript254,
) -> io::Result<MembershipProof> {
    let n_u = u.len();
    let n_of = of.len();
    if w.len() != n_u {
        return Err(invalid("multiplicity column length must match U"));
    }
    let natural = membership_domain(n_u, n_of)?;
    if !domain.is_power_of_two() || domain < natural {
        return Err(invalid(
            "membership domain must be a power of two >= the natural domain",
        ));
    }
    let u_p = padded(u, domain);
    let of_p = padded(of, domain);
    let w_p = padded(w, domain);
    prove_membership_with_mult_with_witness_and_context(
        context, &u_p, &of_p, &w_p, n_u, n_of, com_u, com_of, com_w, transcript,
    )
}

/// Prove membership-with-multiplicity from FULLY EXPLICIT witness columns (each
/// already padded to a common power-of-two domain `n = u_p.len()`) and explicit real
/// lengths `n_U`/`n_F`. The honest entry points derive these; the security tests use
/// this to stash a duplicate key in U's padding tail `[n_U, domain)` with a claimed
/// `n_U` smaller than the committed column length.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn prove_membership_with_mult_with_witness(
    u_p: &[ArkFr],
    of_p: &[ArkFr],
    w_p: &[ArkFr],
    n_u: usize,
    n_of: usize,
    com_u: &ArkGT,
    com_of: &ArkGT,
    com_w: &ArkGT,
    transcript: &mut Transcript254,
) -> io::Result<MembershipProof> {
    prove_membership_with_mult_with_witness_and_context(
        &StoreProverContext::cpu(),
        u_p,
        of_p,
        w_p,
        n_u,
        n_of,
        com_u,
        com_of,
        com_w,
        transcript,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_membership_with_mult_with_witness_and_context(
    context: &StoreProverContext<'_>,
    u_p: &[ArkFr],
    of_p: &[ArkFr],
    w_p: &[ArkFr],
    n_u: usize,
    n_of: usize,
    com_u: &ArkGT,
    com_of: &ArkGT,
    com_w: &ArkGT,
    transcript: &mut Transcript254,
) -> io::Result<MembershipProof> {
    let n = u_p.len();
    if !n.is_power_of_two() || n < 2 {
        return Err(invalid("membership domain must be a power of two >= 2"));
    }
    if of_p.len() != n || w_p.len() != n {
        return Err(invalid("membership witness column length mismatch"));
    }
    if n_u > n || n_of > n {
        return Err(invalid("membership real length exceeds the domain"));
    }
    let k = n.trailing_zeros() as usize;

    // Bind the value + multiplicity commitments, THEN draw alpha. Caller contract
    // (Inc 0 audit): alpha MUST be drawn after the column commitments are bound so the
    // fraction leaves are fixed before any per-layer GKR challenge. There is no
    // reciprocal commitment anymore, so this is the last binding before the fold.
    transcript.append_group(b"pow/membership/com-u", com_u);
    transcript.append_group(b"pow/membership/com-of", com_of);
    transcript.append_group(b"pow/membership/com-w", com_w);
    let alpha = transcript.challenge_scalar(LABEL_ALPHA);

    // (1) The block-disjoint LogUp-GKR fraction tree over the 2*domain zero-extensions:
    // U + w fill the LOW half (denom alpha - U, numer +w), Of the HIGH half (denom
    // alpha - Of, numer additive -[real image row]). The tree lands at `r_leaf` (length
    // k + 1), whose low k bits are the opening point.
    let inst = membership_fraction_instance(alpha, n_of, k);
    let leaf_count = inst.size();
    let extended = vec![
        low_half(u_p, leaf_count),
        high_half(of_p, leaf_count),
        low_half(w_p, leaf_count),
    ];
    let (gkr, r_leaf) = prove_fraction_sum_zero(&inst, &extended, transcript)?;

    // Single batched Dory opening of the length-`domain` DATA columns at r_low, bound to
    // their (root-anchored / shared) commitments on this transcript.
    let r_low = &r_leaf[..k];
    let columns = vec![u_p.to_vec(), of_p.to_vec(), w_p.to_vec()];
    let col_evals: Vec<ArkFr> = columns.iter().map(|c| mle_full_eval(c, r_low)).collect();
    let opening = context.prove_column_openings(&columns, r_low, transcript)?;

    // (2) The GAP-1 auxiliary Identity (1 - chi_{n_U})*w = 0, pinning w to zero on the
    // universe padding tail [n_U, domain) UNCONDITIONALLY (arbitrary n_U). Proved over
    // the SAME committed w via the argument.rs sumcheck; its opening below binds to
    // com_w on the SAME transcript, so the tree and the Identity constrain one identical
    // w. No reciprocal is reintroduced (the chi*w product is a supported term).
    let id_shape = universe_mask_identity_shape(k, n_u);
    let w_col = vec![w_p.to_vec()];
    let (id_sumcheck, r_id, _evals) = prove_sumcheck(&id_shape, &w_col, transcript)?;
    let id_opening = context.prove_column_openings(&w_col, &r_id, transcript)?;

    Ok(MembershipProof {
        gkr,
        col_evals,
        opening,
        id_sumcheck,
        id_opening,
    })
}

/// Verify a membership-with-multiplicity proof.
pub fn verify_membership_with_mult(
    com_u: &ArkGT,
    com_of: &ArkGT,
    com_w: &ArkGT,
    n_u: usize,
    n_of: usize,
    proof: &MembershipProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    // The GKR tree has `k + 1` layers (`2*domain` leaves for a length-`domain = 2^k`
    // value domain), so `k = num_vars - 1` and the committed columns have length `2^k`.
    let num_vars = proof.gkr.num_vars();
    if num_vars < 2 || num_vars >= usize::BITS as usize {
        return Err(invalid("membership proof has an implausible variable count"));
    }
    let k = num_vars - 1;
    let domain = 1usize << k;
    if n_u > domain || n_of > domain {
        return Err(invalid("real length exceeds the membership domain"));
    }
    if proof.col_evals.len() != NUM_COLUMNS {
        return Err(invalid("membership proof has the wrong number of column evaluations"));
    }

    transcript.append_group(b"pow/membership/com-u", com_u);
    transcript.append_group(b"pow/membership/com-of", com_of);
    transcript.append_group(b"pow/membership/com-w", com_w);
    let alpha = transcript.challenge_scalar(LABEL_ALPHA);

    // (1) Bind the public instance and verify the GKR tree -> the leaf point + claims.
    let inst = membership_fraction_instance(alpha, n_of, k);
    let (r_leaf, p_claim, q_claim) = bind_and_verify_fraction_tree(&inst, &proof.gkr, transcript)?;

    // Open the length-`domain` committed columns at r_low, bound to their commitments.
    let r_low = &r_leaf[..k];
    let r_high = r_leaf[k];
    verify_column_openings(
        &[*com_u, *com_of, *com_w],
        r_low,
        &proof.col_evals,
        &proof.opening,
        transcript,
    )?;

    // Reconstruct each 2*domain leaf-MLE from the length-`domain` opening via the top-bit
    // zero-extension factor: U + w live in the LOW half (factor 1 - r_high), Of in the
    // HIGH half (factor r_high). This is the block-disjoint enforcement (neither side's
    // fold columns leak into the other block); then tie to the GKR-reduced leaf claims.
    let one_minus_high = f1() - r_high;
    let leaf_evals = vec![
        one_minus_high * proof.col_evals[COL_U],
        r_high * proof.col_evals[COL_OF],
        one_minus_high * proof.col_evals[COL_W],
    ];
    check_leaf_tie(&inst, &leaf_evals, p_claim, q_claim, &r_leaf)?;

    // (2) Verify the GAP-1 auxiliary Identity (1 - chi_{n_U})*w = 0 over the SAME com_w.
    // Its `w` opening binds to com_w on this transcript, so the multiplicity the tree
    // used is EXACTLY the one pinned to zero on `[n_U, domain)` -- a forged padding
    // multiplicity is rejected UNCONDITIONALLY (no unmatched-pole invariant needed).
    let id_shape = universe_mask_identity_shape(k, n_u);
    let (r_id, id_evals) = verify_sumcheck(&id_shape, &proof.id_sumcheck, transcript)?;
    verify_column_openings(&[*com_w], &r_id, &id_evals, &proof.id_opening, transcript)?;
    Ok(())
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
        Transcript254::new(b"parture-membership-test")
    }

    fn commit_padded(column: &[ArkFr], n: usize) -> ArkGT {
        commit_column(&padded(column, n)).unwrap()
    }

    // Run with an EXPLICIT w (so tests can tamper it).
    fn run(u: &[ArkFr], of: &[ArkFr], w: &[ArkFr]) -> io::Result<()> {
        let n = membership_domain(u.len(), of.len()).unwrap();
        let com_u = commit_padded(u, n);
        let com_of = commit_padded(of, n);
        let com_w = commit_padded(w, n);
        let mut pt = transcript();
        let proof = prove_membership_with_mult(u, of, w, &com_u, &com_of, &com_w, &mut pt)?;
        let mut vt = transcript();
        verify_membership_with_mult(&com_u, &com_of, &com_w, u.len(), of.len(), &proof, &mut vt)
    }

    #[test]
    fn correct_multiset_image_verifies() {
        // Of = [1,2,2,3] is the image of U = [1,2,3] with w = [1,2,1].
        let u = frs(&[1, 2, 3]);
        let of = frs(&[1, 2, 2, 3]);
        let w = count_multiplicities(&u, &of);
        assert_eq!(w, frs(&[1, 2, 1]));
        assert!(run(&u, &of, &w).is_ok());
    }

    #[test]
    fn tampered_multiplicity_is_rejected() {
        let u = frs(&[1, 2, 3]);
        let of = frs(&[1, 2, 2, 3]);
        let mut w = count_multiplicities(&u, &of);
        w[0] = w[0] + fr(1); // claim 1 appears twice
        assert!(run(&u, &of, &w).is_err());
    }

    #[test]
    fn of_value_absent_from_u_is_rejected() {
        // 3 in Of is not in U -> its pole cannot be matched by any w*rec_u.
        let u = frs(&[1, 2]);
        let of = frs(&[1, 3]);
        let w = count_multiplicities(&u, &of); // [1, 0]
        assert!(run(&u, &of, &w).is_err());
    }

    #[test]
    fn wrong_claimed_count_is_rejected() {
        // Of has 1 twice; claiming w[U=1] = 1 (should be 2) mismatches the residue.
        let u = frs(&[1, 2]);
        let of = frs(&[1, 1, 2]);
        let w = frs(&[1, 1]); // wrong: should be [2, 1]
        assert!(run(&u, &of, &w).is_err());
    }

    #[test]
    fn duplicate_u_with_split_weight_still_verifies_here() {
        // DOCUMENTED GAP: U has a duplicated value (5). Of has one 5. A w that
        // SPLITS the count across the two copies of 5 ([1,0,1]) still verifies,
        // because the residue only pins the SUM over equal U values. This module
        // does NOT enforce U-distinctness -- that is Phase 2c-2a, and the count
        // identity (2c-2c) relies on it to close this.
        let u = frs(&[5, 5, 7]);
        let of = frs(&[5, 7]);
        assert!(run(&u, &of, &frs(&[1, 0, 1])).is_ok());
        assert!(run(&u, &of, &frs(&[0, 1, 1])).is_ok());
    }

    #[test]
    fn empty_image_verifies() {
        // Of empty (n_F = 0): every multiplicity is 0.
        let u = frs(&[1, 2]);
        assert!(run(&u, &[], &frs(&[0, 0])).is_ok());
    }

    #[test]
    fn duplicate_key_stashed_in_u_padding_tail_is_rejected() {
        // COMPLETENESS regression (FIX 2), now closed by the GAP-1 Identity. Real U =
        // [5] (claimed n_U = 1), but a duplicate 5 is STASHED at padding index 1
        // (>= n_U) of the committed column. Of = [5,5]. A split weight w = [1,1] lets
        // the padding pole cover the second 5, so the GKR fraction sum balances (root
        // P = 0). But w[1] != 0 at index 1 >= n_U = 1 violates (1 - chi_{n_U})*w = 0,
        // so the auxiliary Identity sumcheck REJECTS -- the real U[0]=5 has w=1 but Of
        // has two 5's, an undercount forge.
        let u_p = frs(&[5, 5, 0, 0]);
        let of_p = frs(&[5, 5, 0, 0]);
        let w_p = frs(&[1, 1, 0, 0]);
        let (n_u, n_of) = (1usize, 2usize);
        let com_u = commit_column(&u_p).unwrap();
        let com_of = commit_column(&of_p).unwrap();
        let com_w = commit_column(&w_p).unwrap();
        let mut pt = transcript();
        let proof = prove_membership_with_mult_with_witness(
            &u_p, &of_p, &w_p, n_u, n_of, &com_u, &com_of, &com_w, &mut pt,
        )
        .unwrap();
        let mut vt = transcript();
        assert!(
            verify_membership_with_mult(&com_u, &com_of, &com_w, n_u, n_of, &proof, &mut vt)
                .is_err()
        );
    }

    #[test]
    fn padding_tail_forged_multiplicity_rejected_at_non_pow2_n_u() {
        // The UNCONDITIONAL GAP-1 negative at an ARBITRARY (non-power-of-two) boundary,
        // where a structural zero-extension of w could NOT reproduce the mask. Real U =
        // [1,2,3] (n_U = 3), padded to domain 8; a duplicate of key 1 is STASHED at
        // padding index 3 (>= n_U). Of = [1,1,2,3] (n_F = 4, key 1 twice). A split
        // weight w = [1,1,1, 1, ..] puts the extra 1 at the stashed index 3, so the
        // padding pole +1/(alpha - 1) covers the second Of 1 and the GKR fraction sum
        // is exactly zero (root P = 0). But w[3] != 0 at index 3 >= n_U = 3 violates
        // (1 - chi_3)*w = 0, so the Identity sumcheck REJECTS. Without the mask this is
        // an undercount (claiming key 1 appears once when it appears twice).
        let u_p = frs(&[1, 2, 3, 1, 0, 0, 0, 0]);
        let of_p = frs(&[1, 1, 2, 3, 0, 0, 0, 0]);
        let w_p = frs(&[1, 1, 1, 1, 0, 0, 0, 0]);
        let (n_u, n_of) = (3usize, 4usize);
        let com_u = commit_column(&u_p).unwrap();
        let com_of = commit_column(&of_p).unwrap();
        let com_w = commit_column(&w_p).unwrap();
        let mut pt = transcript();
        // The honest prover flow runs (it does not validate the witness); the forged
        // padding pole makes the GKR sum balance, so it is the Identity that must catch
        // the forge on the shared com_w.
        let proof = prove_membership_with_mult_with_witness(
            &u_p, &of_p, &w_p, n_u, n_of, &com_u, &com_of, &com_w, &mut pt,
        )
        .unwrap();
        assert!(proof.gkr.root_p.is_zero(), "forged padding pole should balance the tree");
        let mut vt = transcript();
        assert!(
            verify_membership_with_mult(&com_u, &com_of, &com_w, n_u, n_of, &proof, &mut vt)
                .is_err(),
            "a forged padding-tail multiplicity at a non-pow2 n_U must reject via the Identity"
        );
    }

    #[test]
    fn tampered_column_eval_is_rejected() {
        // Tampering a claimed column evaluation is rejected by the Dory opening / leaf
        // tie: the opening binds `col_evals` to the committed columns at r_low.
        let u = frs(&[1, 2, 3]);
        let of = frs(&[1, 2, 2, 3]);
        let w = count_multiplicities(&u, &of);
        let n = membership_domain(u.len(), of.len()).unwrap();
        let com_u = commit_padded(&u, n);
        let com_of = commit_padded(&of, n);
        let com_w = commit_padded(&w, n);
        let mut pt = transcript();
        let mut proof =
            prove_membership_with_mult(&u, &of, &w, &com_u, &com_of, &com_w, &mut pt).unwrap();
        proof.col_evals[COL_W] = proof.col_evals[COL_W] + f1();
        let mut vt = transcript();
        assert!(
            verify_membership_with_mult(&com_u, &com_of, &com_w, u.len(), of.len(), &proof, &mut vt)
                .is_err()
        );
    }

    #[test]
    fn tampered_gkr_root_is_rejected() {
        // The root claim `P = 0` is the whole GKR statement; a nonzero root_p fails.
        let u = frs(&[1, 2, 3]);
        let of = frs(&[1, 2, 2, 3]);
        let w = count_multiplicities(&u, &of);
        let n = membership_domain(u.len(), of.len()).unwrap();
        let com_u = commit_padded(&u, n);
        let com_of = commit_padded(&of, n);
        let com_w = commit_padded(&w, n);
        let mut pt = transcript();
        let mut proof =
            prove_membership_with_mult(&u, &of, &w, &com_u, &com_of, &com_w, &mut pt).unwrap();
        proof.gkr.root_p = proof.gkr.root_p + f1();
        let mut vt = transcript();
        assert!(
            verify_membership_with_mult(&com_u, &com_of, &com_w, u.len(), of.len(), &proof, &mut vt)
                .is_err()
        );
    }
}
