//! Membership-with-multiplicity (Phase 2c-2b), the second completeness primitive.
//! A THIN wiring of `proof::logup::multiset_link_claim` + `reciprocal_claim`.
//!
//! Goal: given committed value columns `U` (real length `n_U`) and `Of` (real
//! length `n_F`) and a committed multiplicity column `w` (length `n_U`), prove
//! `Of` is EXACTLY the multiset image of `U` under `w` — every `Of` value appears
//! in `U`, and `w[i] = #{ a : Of_a = U[i] }`. This is completeness claim (i);
//! claim (ii) is the symmetric `Sf/U/w_R` with fresh challenges.
//!
//! Pure value-multiset LogUp (no `idx`): the reciprocals `rec_u[i]=1/(alpha-U[i])`
//! and `rec_of[a]=1/(alpha-Of[a])` are pinned, and
//!   `Sum_i w[i]*rec_u[i] - Sum_a chi_{n_F}[a]*rec_of[a] = 0`
//! (Habock's log-derivative pole/residue identity) pins `w[i]` to the true count
//! of `U[i]` in `Of`. Since values are single field elements, no `beta`
//! fingerprint is needed; `beta` is drawn (via `reciprocal_claim`) but its power
//! on a one-column fold is `1`.
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
    prove_sumcheck, verify_column_openings, verify_sumcheck,
    ColumnOpening, Operand, StoreProverContext, SumcheckProof, SumcheckShape, Transcript254,
};
#[cfg(test)]
use super::argument::commit_column;
use super::invalid;
use super::logup::{masked_multiset_link_claim, reciprocal_claim, reciprocal_witness};

const COL_U: usize = 0;
const COL_OF: usize = 1;
const COL_W: usize = 2;
const COL_REC_U: usize = 3;
const COL_REC_OF: usize = 4;
const NUM_COLUMNS: usize = 5;
// chi_bounds indices: chi_{n_F} (image side real length) and chi_{n_U} (universe side
// real length; masks a duplicate key stashed in U's padding tail out of the link).
const CHI_NF: usize = 0;
const CHI_NU: usize = 1;

const LABEL_BETA: &[u8] = b"pow/membership/fingerprint-beta";
const LABEL_ALPHA: &[u8] = b"pow/membership/reciprocal-alpha";

#[inline]
fn f0() -> ArkFr {
    ArkFr::zero()
}

/// A membership-with-multiplicity proof: the batched sumcheck over the two
/// reciprocal (Identity) claims + the LogUp-link (ZeroSum) claim, one batched
/// Dory opening over the five columns, and the two reciprocal commitments.
#[derive(Clone, Debug)]
pub struct MembershipProof {
    pub sumcheck: SumcheckProof,
    pub opening: ColumnOpening,
    pub rec_u_commitment: ArkGT,
    pub rec_of_commitment: ArkGT,
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

fn membership_shape(k: usize, n_u: usize, n_of: usize, alpha: ArkFr, beta: ArkFr) -> SumcheckShape {
    let rec_u = reciprocal_claim(COL_REC_U, &[COL_U], alpha, beta);
    let rec_of = reciprocal_claim(COL_REC_OF, &[COL_OF], alpha, beta);
    // Mask the UNIVERSE (U) side by chi_{n_U}: a duplicate key stashed in U's padding
    // tail [n_U, domain) is not a valid pole, so its weight cannot cover an Of value.
    let link = masked_multiset_link_claim(
        &[Operand::Chi(CHI_NU), Operand::Column(COL_W)],
        COL_REC_U,
        &[Operand::Chi(CHI_NF)],
        COL_REC_OF,
    );
    SumcheckShape {
        num_vars: k,
        num_columns: NUM_COLUMNS,
        chi_bounds: vec![n_of, n_u],
        claims: vec![rec_u, rec_of, link],
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
/// `n_U`/`n_F` still drive the chi mask, so the residue ranges only over the true
/// `Of` rows (padding poles are excluded).
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

    // Round 1: bind the value + multiplicity commitments, draw the challenges.
    transcript.append_group(b"pow/membership/com-u", com_u);
    transcript.append_group(b"pow/membership/com-of", com_of);
    transcript.append_group(b"pow/membership/com-w", com_w);
    let beta = transcript.challenge_scalar(LABEL_BETA);
    let alpha = transcript.challenge_scalar(LABEL_ALPHA);

    // Round 2: reciprocal witnesses over the values.
    let rec_u = reciprocal_witness(alpha, u_p);
    let rec_of = reciprocal_witness(alpha, of_p);
    let rec_u_commitment = context.commit_column(&rec_u)?;
    let rec_of_commitment = context.commit_column(&rec_of)?;
    transcript.append_group(b"pow/membership/com-rec-u", &rec_u_commitment);
    transcript.append_group(b"pow/membership/com-rec-of", &rec_of_commitment);

    let shape = membership_shape(k, n_u, n_of, alpha, beta);
    let columns = vec![
        u_p.to_vec(),
        of_p.to_vec(),
        w_p.to_vec(),
        rec_u,
        rec_of,
    ];
    let (sumcheck, r, _evals) = prove_sumcheck(&shape, &columns, transcript)?;
    let opening = context.prove_column_openings(&columns, &r, transcript)?;

    Ok(MembershipProof {
        sumcheck,
        opening,
        rec_u_commitment,
        rec_of_commitment,
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
    let k = proof.sumcheck.round_polys.len();
    if k == 0 || k >= usize::BITS as usize {
        return Err(invalid("membership proof has an implausible variable count"));
    }
    if n_u > (1usize << k) || n_of > (1usize << k) {
        return Err(invalid("real length exceeds the membership domain"));
    }

    transcript.append_group(b"pow/membership/com-u", com_u);
    transcript.append_group(b"pow/membership/com-of", com_of);
    transcript.append_group(b"pow/membership/com-w", com_w);
    let beta = transcript.challenge_scalar(LABEL_BETA);
    let alpha = transcript.challenge_scalar(LABEL_ALPHA);
    transcript.append_group(b"pow/membership/com-rec-u", &proof.rec_u_commitment);
    transcript.append_group(b"pow/membership/com-rec-of", &proof.rec_of_commitment);

    let shape = membership_shape(k, n_u, n_of, alpha, beta);
    let (r, evals) = verify_sumcheck(&shape, &proof.sumcheck, transcript)?;
    verify_column_openings(
        &[
            *com_u,
            *com_of,
            *com_w,
            proof.rec_u_commitment,
            proof.rec_of_commitment,
        ],
        &r,
        &evals,
        &proof.opening,
        transcript,
    )
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
        // COMPLETENESS regression (FIX 2). Real U = [5] (claimed n_U = 1), but a
        // duplicate 5 is STASHED at padding index 1 (>= n_U) of the committed column.
        // Of = [5,5]. A split weight w = [1,1] lets the padding pole cover the second
        // 5, so the residue balances. With the U side unmasked this VERIFIES (an
        // undercount vector for the count identity); masking the U side by chi_{n_U}
        // excludes the padding pole -> the real U[0]=5 has w=1 but Of has two 5's ->
        // REJECT.
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
}
