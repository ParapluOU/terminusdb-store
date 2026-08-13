//! Full SOUND & COMPLETE self-join proof (Phase 2c-2e): composes the tested
//! completeness gadgets under ONE shared transcript against ONE set of committed
//! columns. Clean-room composition; no COSL code.
//!
//! A 2-atom self-join `t(?x, P, ?y), t(?y, P, ?z)` reduces (after the Q1 filter by
//! `P`) to self-joining a relation `F = (Sf, Of)` on `Of == Sf` (the shared
//! variable `y`), producing `(X, Y, Z)`. This module proves the claimed output is
//! the COMPLETE join. It replaces the earlier soundness-only gadget (subsumed by
//! `proof::pairs`) and the UNSOUND `proof::gather` (deleted): the gather only pinned
//! MARGINAL usage counts, which do not determine the bipartite pairing.
//!
//! The completeness argument (now sound) chains five tested gadgets:
//!   * (2c-2f) `prove_distinct_valid_pairs`: the output is a SET of `m_J` DISTINCT
//!     genuine key-matching (left-F-row, right-F-row) pairs (index-pointer
//!     membership + combined-index strict monotonicity).
//!   * (2c-2a) `prove_strictly_sorted(U)`: the claimed key column `U` is strictly
//!     increasing, hence DISTINCT.
//!   * (2c-2b) `prove_membership_with_mult(Of, U, w_L)` and `(Sf, U, w_R)`: `Of` is
//!     the image of `U` under `w_L` and `Sf` the image under `w_R`, so `U` contains
//!     EVERY join key and `w_L`/`w_R` are the TRUE per-key group counts.
//!   * (2c-2c) `prove_count_identity(w_L, w_R, m_J)`: `Sum_i w_L[i]*w_R[i] = m_J`,
//!     i.e. the number of valid pairs EQUALS the output count.
//!
//! Together: the output is `m_J` DISTINCT valid pairs AND `m_J` equals the total
//! number of valid pairs, so the output is ALL valid pairs (a size-`m_J` subset of a
//! size-`m_J` universe is the whole universe) => SOUND + COMPLETE.
//!
//! Composition: all ten logical columns are committed ONCE at a single shared
//! `self_join_domain`, and the SAME commitments are threaded into every sub-proof
//! that references them (`Sf`/`Of` link pairs<->membership; `U`/`w_L`/`w_R` link
//! distinctness<->membership<->count). The whole instance is bound up front so each
//! sub-proof's Fiat-Shamir challenges depend on the entire committed instance.

use std::io;

use ark_ff::PrimeField;
use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::transcript::Transcript;

use super::argument::{StoreProverContext, Transcript254};
#[cfg(test)]
use super::argument::commit_column;
use super::count::{prove_count_identity_with_context, verify_count_identity, CountProof};
#[cfg(test)]
use super::count::prove_count_identity;
use super::invalid;
use super::membership::{
    prove_membership_with_mult_in_domain_with_context, verify_membership_with_mult,
    MembershipProof,
};
#[cfg(test)]
use super::membership::prove_membership_with_mult_with_witness;
use super::pairs::{
    prove_distinct_valid_pairs_in_domain_with_context, verify_distinct_valid_pairs, PairsProof,
};
#[cfg(test)]
use super::pairs::prove_distinct_valid_pairs_in_domain;
use super::sort::{prove_strictly_sorted_with_context, verify_strictly_sorted, SortedProof};
#[cfg(test)]
use super::sort::prove_strictly_sorted;

#[inline]
fn f0() -> ArkFr {
    ArkFr::zero()
}

/// The single shared power-of-two domain (`>= 2`) onto which EVERY self-join column
/// is padded, so each logical column has ONE commitment reusable across sub-proofs.
pub(crate) fn self_join_domain(n_f: usize, m_j: usize, n_u: usize) -> io::Result<usize> {
    n_f.max(m_j)
        .max(n_u)
        .checked_next_power_of_two()
        .ok_or_else(|| invalid("self-join domain size overflow"))
        .map(|d| d.max(2))
}

fn padded(column: &[ArkFr], n: usize) -> Vec<ArkFr> {
    let mut out = column.to_vec();
    out.resize(n, f0());
    out
}

/// The ten committed columns of a self-join instance, all committed at
/// `self_join_domain`. The SAME `sf`/`of` feed the pairs proof and the U-membership
/// proofs; the SAME `u`/`w_l`/`w_r` feed distinctness, membership, and count.
#[derive(Clone, Debug)]
pub struct SelfJoinCommitments {
    pub sf: ArkGT,
    pub of: ArkGT,
    pub x: ArkGT,
    pub y: ArkGT,
    pub z: ArkGT,
    pub rho_l: ArkGT,
    pub rho_r: ArkGT,
    pub u: ArkGT,
    pub w_l: ArkGT,
    pub w_r: ArkGT,
}

/// A full sound+complete self-join proof: the five composed sub-proofs.
#[derive(Clone, Debug)]
pub struct SelfJoinCompleteProof {
    pub pairs: PairsProof,
    pub sorted_u: SortedProof,
    pub membership_l: MembershipProof,
    pub membership_r: MembershipProof,
    pub count: CountProof,
}

/// Commit every self-join column at the shared domain. Real callers derive these
/// commitments from the committed store layer; tests build them here.
#[allow(clippy::too_many_arguments)]
pub fn self_join_commitments(
    sf: &[ArkFr],
    of: &[ArkFr],
    x: &[ArkFr],
    y: &[ArkFr],
    z: &[ArkFr],
    rho_l: &[ArkFr],
    rho_r: &[ArkFr],
    u: &[ArkFr],
    w_l: &[ArkFr],
    w_r: &[ArkFr],
) -> io::Result<SelfJoinCommitments> {
    self_join_commitments_with_context(
        &StoreProverContext::cpu(), sf, of, x, y, z, rho_l, rho_r, u, w_l, w_r,
    )
}

/// As [`self_join_commitments`], with an explicit prover-arithmetic context.
#[allow(clippy::too_many_arguments)]
pub fn self_join_commitments_with_context(
    context: &StoreProverContext<'_>,
    sf: &[ArkFr],
    of: &[ArkFr],
    x: &[ArkFr],
    y: &[ArkFr],
    z: &[ArkFr],
    rho_l: &[ArkFr],
    rho_r: &[ArkFr],
    u: &[ArkFr],
    w_l: &[ArkFr],
    w_r: &[ArkFr],
) -> io::Result<SelfJoinCommitments> {
    let n = self_join_domain(sf.len(), x.len(), u.len())?;
    Ok(SelfJoinCommitments {
        sf: context.commit_column(&padded(sf, n))?,
        of: context.commit_column(&padded(of, n))?,
        x: context.commit_column(&padded(x, n))?,
        y: context.commit_column(&padded(y, n))?,
        z: context.commit_column(&padded(z, n))?,
        rho_l: context.commit_column(&padded(rho_l, n))?,
        rho_r: context.commit_column(&padded(rho_r, n))?,
        u: context.commit_column(&padded(u, n))?,
        w_l: context.commit_column(&padded(w_l, n))?,
        w_r: context.commit_column(&padded(w_r, n))?,
    })
}

const LABEL_DOMAIN: &[u8] = b"pow/join-complete/domain-v1";
const LABEL_LEN: &[u8] = b"pow/join-complete/len";

/// Bind the entire committed instance (10 commitments + public lengths) up front so
/// every sub-proof's challenges cross-depend on the whole instance. Mirrored exactly
/// by prover and verifier.
fn bind_instance(
    transcript: &mut Transcript254,
    com: &SelfJoinCommitments,
    n_f: usize,
    n_u: usize,
    m_j: usize,
) {
    transcript.append_bytes(LABEL_DOMAIN, b"parture/proof-of-woql/self-join-complete/v0");
    transcript.append_bytes(LABEL_LEN, &(n_f as u64).to_le_bytes());
    transcript.append_bytes(LABEL_LEN, &(n_u as u64).to_le_bytes());
    transcript.append_bytes(LABEL_LEN, &(m_j as u64).to_le_bytes());
    transcript.append_group(b"pow/join-complete/com-sf", &com.sf);
    transcript.append_group(b"pow/join-complete/com-of", &com.of);
    transcript.append_group(b"pow/join-complete/com-x", &com.x);
    transcript.append_group(b"pow/join-complete/com-y", &com.y);
    transcript.append_group(b"pow/join-complete/com-z", &com.z);
    transcript.append_group(b"pow/join-complete/com-rho-l", &com.rho_l);
    transcript.append_group(b"pow/join-complete/com-rho-r", &com.rho_r);
    transcript.append_group(b"pow/join-complete/com-u", &com.u);
    transcript.append_group(b"pow/join-complete/com-w-l", &com.w_l);
    transcript.append_group(b"pow/join-complete/com-w-r", &com.w_r);
}

/// Prove the claimed self-join output `(X, Y, Z)` (with index pointers `rho_l`,
/// `rho_r` and key witnesses `U`, `w_L`, `w_R`) is the SOUND & COMPLETE self-join of
/// `F = (Sf, Of)` on `Of == Sf`. `com` must commit every column padded to
/// `self_join_domain(n_F, m_J, n_U)` (see [`self_join_commitments`]).
#[allow(clippy::too_many_arguments)]
pub fn prove_self_join_complete(
    sf: &[ArkFr],
    of: &[ArkFr],
    x: &[ArkFr],
    y: &[ArkFr],
    z: &[ArkFr],
    rho_l: &[ArkFr],
    rho_r: &[ArkFr],
    u: &[ArkFr],
    w_l: &[ArkFr],
    w_r: &[ArkFr],
    m_j: usize,
    com: &SelfJoinCommitments,
    transcript: &mut Transcript254,
) -> io::Result<SelfJoinCompleteProof> {
    prove_self_join_complete_with_context(
        &StoreProverContext::cpu(), sf, of, x, y, z, rho_l, rho_r, u, w_l, w_r, m_j, com,
        transcript,
    )
}

/// As [`prove_self_join_complete`], with an explicit prover-arithmetic context.
#[allow(clippy::too_many_arguments)]
pub fn prove_self_join_complete_with_context(
    context: &StoreProverContext<'_>,
    sf: &[ArkFr],
    of: &[ArkFr],
    x: &[ArkFr],
    y: &[ArkFr],
    z: &[ArkFr],
    rho_l: &[ArkFr],
    rho_r: &[ArkFr],
    u: &[ArkFr],
    w_l: &[ArkFr],
    w_r: &[ArkFr],
    m_j: usize,
    com: &SelfJoinCommitments,
    transcript: &mut Transcript254,
) -> io::Result<SelfJoinCompleteProof> {
    let n_f = sf.len();
    let n_u = u.len();
    if of.len() != n_f {
        return Err(invalid("F columns Sf/Of length mismatch"));
    }
    if x.len() != m_j
        || y.len() != m_j
        || z.len() != m_j
        || rho_l.len() != m_j
        || rho_r.len() != m_j
    {
        return Err(invalid("output columns must all have length m_J"));
    }
    if w_l.len() != n_u || w_r.len() != n_u {
        return Err(invalid("multiplicity columns must match U length"));
    }
    let domain = self_join_domain(n_f, m_j, n_u)?;

    bind_instance(transcript, com, n_f, n_u, m_j);

    // (2c-2f) output = m_J DISTINCT genuine key-matching (left, right) F-pairs.
    let pairs = prove_distinct_valid_pairs_in_domain_with_context(
        context, sf, of, x, y, z, rho_l, rho_r, &com.sf, &com.of, &com.x, &com.y, &com.z, &com.rho_l,
        &com.rho_r, domain, transcript,
    )?;
    // (2c-2a) U strictly increasing over [0, n_U): the distinct join-key set.
    let u_padded = padded(u, domain);
    let sorted_u = prove_strictly_sorted_with_context(context, &u_padded, n_u, &com.u, transcript)?;
    // (2c-2b) Of is the image of U under w_L, and Sf the image of U under w_R:
    // U covers every join key and w_L/w_R are the TRUE per-key group counts.
    let membership_l =
        prove_membership_with_mult_in_domain_with_context(context, u, of, w_l, &com.u, &com.of, &com.w_l, domain, transcript)?;
    let membership_r =
        prove_membership_with_mult_in_domain_with_context(context, u, sf, w_r, &com.u, &com.sf, &com.w_r, domain, transcript)?;
    // (2c-2c) Sum_i w_L[i]*w_R[i] = m_J: valid-pair count EQUALS the output count.
    let wl_padded = padded(w_l, domain);
    let wr_padded = padded(w_r, domain);
    // FIX 3: the TRUE n_U (not the padded domain) is the count's chi real length, so a
    // weight stashed in the padding tail [n_U, domain) cannot inflate the count.
    let count =
        prove_count_identity_with_context(context, &wl_padded, &wr_padded, n_u, m_j, &com.w_l, &com.w_r, transcript)?;

    Ok(SelfJoinCompleteProof {
        pairs,
        sorted_u,
        membership_l,
        membership_r,
        count,
    })
}

/// Prove a self-join from EXPLICIT witnesses, with the distinct-key count `n_U`
/// decoupled from the committed `U` column length. The honest path derives `n_U =
/// |U|`; the omission security test uses this to commit a `U`/`w` with a duplicate
/// key STASHED in the padding tail `[n_U, domain)` while claiming a smaller `n_U`.
/// `u_full`/`w_l_full`/`w_r_full` are length `domain`; `sf`/`of` and the output
/// columns are the real (honest) slices.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_self_join_complete_with_witness(
    sf: &[ArkFr],
    of: &[ArkFr],
    x: &[ArkFr],
    y: &[ArkFr],
    z: &[ArkFr],
    rho_l: &[ArkFr],
    rho_r: &[ArkFr],
    u_full: &[ArkFr],
    w_l_full: &[ArkFr],
    w_r_full: &[ArkFr],
    n_f: usize,
    n_u: usize,
    m_j: usize,
    com: &SelfJoinCommitments,
    transcript: &mut Transcript254,
) -> io::Result<SelfJoinCompleteProof> {
    let domain = u_full.len();
    bind_instance(transcript, com, n_f, n_u, m_j);

    let pairs = prove_distinct_valid_pairs_in_domain(
        sf, of, x, y, z, rho_l, rho_r, &com.sf, &com.of, &com.x, &com.y, &com.z, &com.rho_l,
        &com.rho_r, domain, transcript,
    )?;
    let sorted_u = prove_strictly_sorted(u_full, n_u, &com.u, transcript)?;
    let of_p = padded(of, domain);
    let sf_p = padded(sf, domain);
    let membership_l = prove_membership_with_mult_with_witness(
        u_full, &of_p, w_l_full, n_u, n_f, &com.u, &com.of, &com.w_l, transcript,
    )?;
    let membership_r = prove_membership_with_mult_with_witness(
        u_full, &sf_p, w_r_full, n_u, n_f, &com.u, &com.sf, &com.w_r, transcript,
    )?;
    let count =
        prove_count_identity(w_l_full, w_r_full, n_u, m_j, &com.w_l, &com.w_r, transcript)?;

    Ok(SelfJoinCompleteProof {
        pairs,
        sorted_u,
        membership_l,
        membership_r,
        count,
    })
}

/// Verify a full sound+complete self-join proof against the committed instance,
/// re-running each sub-verification on the same shared transcript in prover order.
pub fn verify_self_join_complete(
    com: &SelfJoinCommitments,
    n_f: usize,
    n_u: usize,
    m_j: usize,
    proof: &SelfJoinCompleteProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    bind_instance(transcript, com, n_f, n_u, m_j);

    verify_distinct_valid_pairs(
        &com.sf, &com.of, &com.x, &com.y, &com.z, &com.rho_l, &com.rho_r, n_f, m_j, &proof.pairs,
        transcript,
    )?;
    verify_strictly_sorted(&com.u, n_u, &proof.sorted_u, transcript)?;
    verify_membership_with_mult(&com.u, &com.of, &com.w_l, n_u, n_f, &proof.membership_l, transcript)?;
    verify_membership_with_mult(&com.u, &com.sf, &com.w_r, n_u, n_f, &proof.membership_r, transcript)?;
    // FIX 3: count's chi real length is the TRUE n_U (matches the prover).
    verify_count_identity(&com.w_l, &com.w_r, n_u, m_j, &proof.count, transcript)?;
    Ok(())
}

/// Sorted distinct union of two value columns (the join-key universe `U`). Sorted by
/// canonical field representation, which for `u64`-range dictionary ids is numeric
/// order (the order the strict-sortedness gadget checks).
fn sorted_distinct_union(a: &[ArkFr], b: &[ArkFr]) -> Vec<ArkFr> {
    let mut all: Vec<ArkFr> = a.iter().chain(b.iter()).copied().collect();
    all.sort_by(|p, q| p.0.into_bigint().cmp(&q.0.into_bigint()));
    all.dedup();
    all
}

/// `mult[i] = #{ e in `col` : e == u[i] }` as field elements.
fn multiplicity_in(u: &[ArkFr], col: &[ArkFr]) -> Vec<ArkFr> {
    u.iter()
        .map(|value| ArkFr::from_u64(col.iter().filter(|e| *e == value).count() as u64))
        .collect()
}

/// Compute the complete self-join of `F = (Sf, Of)` on `Of == Sf`, plus every
/// witness column the completeness proof needs. Returns
/// `(X, Y, Z, rho_l, rho_r, U, w_L, w_R, m_J)`:
///   * one output row `(X = Sf[l], Y = Of[l] = Sf[r], Z = Of[r])` with index
///     pointers `rho_l = l`, `rho_r = r` for every `(l, r)` with `Of[l] == Sf[r]`,
///     emitted in `(l, r)` lexicographic order == combined-index `C*l + r` ascending
///     (`C = next_pow2(n_F) >= n_F > r`), so `rho`s are strictly increasing;
///   * `U` = sorted distinct union of `values(Of)` and `values(Sf)` (the join keys);
///   * `w_L[i] = #{ l : Of[l] == U[i] }`, `w_R[i] = #{ r : Sf[r] == U[i] }`;
///   * `m_J = |output| = Sum_i w_L[i]*w_R[i]`.
#[allow(clippy::type_complexity)]
pub fn compute_self_join(
    sf: &[ArkFr],
    of: &[ArkFr],
) -> (
    Vec<ArkFr>,
    Vec<ArkFr>,
    Vec<ArkFr>,
    Vec<ArkFr>,
    Vec<ArkFr>,
    Vec<ArkFr>,
    Vec<ArkFr>,
    Vec<ArkFr>,
    usize,
) {
    let n_f = sf.len();
    let (mut x, mut y, mut z, mut rho_l, mut rho_r) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for l in 0..n_f {
        for r in 0..n_f {
            if of[l] == sf[r] {
                x.push(sf[l]);
                y.push(of[l]);
                z.push(of[r]);
                rho_l.push(ArkFr::from_u64(l as u64));
                rho_r.push(ArkFr::from_u64(r as u64));
            }
        }
    }
    let m_j = x.len();
    let u = sorted_distinct_union(of, sf);
    let w_l = multiplicity_in(&u, of);
    let w_r = multiplicity_in(&u, sf);
    (x, y, z, rho_l, rho_r, u, w_l, w_r, m_j)
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
        Transcript254::new(b"parture-self-join-complete-test")
    }

    // F = {(1,5),(2,5),(5,10),(5,20)}; key 5 has left F-rows {0,1} and right {2,3}.
    fn relation() -> (Vec<ArkFr>, Vec<ArkFr>) {
        (frs(&[1, 2, 5, 5]), frs(&[5, 5, 10, 20]))
    }

    #[allow(clippy::too_many_arguments)]
    fn attempt(
        sf: &[ArkFr],
        of: &[ArkFr],
        x: &[ArkFr],
        y: &[ArkFr],
        z: &[ArkFr],
        rho_l: &[ArkFr],
        rho_r: &[ArkFr],
        u: &[ArkFr],
        w_l: &[ArkFr],
        w_r: &[ArkFr],
        m_j: usize,
    ) -> io::Result<()> {
        let com = self_join_commitments(sf, of, x, y, z, rho_l, rho_r, u, w_l, w_r)?;
        let mut pt = transcript();
        let proof = prove_self_join_complete(
            sf, of, x, y, z, rho_l, rho_r, u, w_l, w_r, m_j, &com, &mut pt,
        )?;
        let mut vt = transcript();
        verify_self_join_complete(&com, sf.len(), u.len(), m_j, &proof, &mut vt)
    }

    // Prove/verify the honestly-computed complete self-join of F.
    fn attempt_honest(sf: &[ArkFr], of: &[ArkFr]) -> io::Result<()> {
        let (x, y, z, rho_l, rho_r, u, w_l, w_r, m_j) = compute_self_join(sf, of);
        attempt(sf, of, &x, &y, &z, &rho_l, &rho_r, &u, &w_l, &w_r, m_j)
    }

    #[test]
    fn compute_self_join_matches_hand_computed_output() {
        let (sf, of) = relation();
        let (x, y, z, rho_l, rho_r, u, w_l, w_r, m_j) = compute_self_join(&sf, &of);
        // Exact contents of the complete join, ordered by (rho_l, rho_r).
        assert_eq!(m_j, 4);
        assert_eq!(x, frs(&[1, 1, 2, 2]));
        assert_eq!(y, frs(&[5, 5, 5, 5]));
        assert_eq!(z, frs(&[10, 20, 10, 20]));
        assert_eq!(rho_l, frs(&[0, 0, 1, 1]));
        assert_eq!(rho_r, frs(&[2, 3, 2, 3]));
        // U = sorted distinct union of {5,10,20} and {1,2,5} = {1,2,5,10,20}.
        assert_eq!(u, frs(&[1, 2, 5, 10, 20]));
        assert_eq!(w_l, frs(&[0, 0, 2, 1, 1])); // key counts in Of
        assert_eq!(w_r, frs(&[1, 1, 2, 0, 0])); // key counts in Sf
    }

    #[test]
    fn genuine_complete_self_join_verifies() {
        let (sf, of) = relation();
        assert!(attempt_honest(&sf, &of).is_ok());
    }

    #[test]
    fn rob_peter_omit_and_duplicate_is_rejected() {
        // THE ROB-PETER ATTACK: keep the honest COUNT (m_J = 4) and the honest key
        // witnesses (U, w_L, w_R) untouched -- so both memberships and the count
        // identity still pass -- but omit the valid pair (1,3) and duplicate (0,2).
        // Every output row is still a genuine key-match, so the memberships cannot
        // see it. It is caught by the pairs' COMBINED-INDEX STRICT MONOTONICITY: the
        // duplicated (rho_l, rho_r) = (0,2) makes i_eval = C*rho_l + rho_r repeat,
        // so i_eval is not strictly increasing.
        let (sf, of) = relation();
        let (_, _, _, _, _, u, w_l, w_r, m_j) = compute_self_join(&sf, &of);
        // {(0,2),(0,2),(0,3),(1,2)}: (1,3) omitted, (0,2) duplicated. Still 4 rows.
        let x = frs(&[1, 1, 1, 2]);
        let y = frs(&[5, 5, 5, 5]);
        let z = frs(&[10, 10, 20, 10]);
        let rho_l = frs(&[0, 0, 0, 1]);
        let rho_r = frs(&[2, 2, 3, 2]);
        assert!(attempt(&sf, &of, &x, &y, &z, &rho_l, &rho_r, &u, &w_l, &w_r, m_j).is_err());
    }

    #[test]
    fn omitting_a_pair_is_caught_by_count_identity() {
        // Drop the valid pair (1,3): 3 DISTINCT valid pairs remain (i_eval strictly
        // increasing, memberships pass), but the honest key witnesses give
        // Sum w_L*w_R = 4 != 3 claimed rows, so the COUNT IDENTITY rejects.
        let (sf, of) = relation();
        let (_, _, _, _, _, u, w_l, w_r, _) = compute_self_join(&sf, &of);
        let x = frs(&[1, 2, 2]);
        let y = frs(&[5, 5, 5]);
        let z = frs(&[10, 10, 20]);
        let rho_l = frs(&[0, 1, 1]);
        let rho_r = frs(&[2, 2, 3]);
        assert!(attempt(&sf, &of, &x, &y, &z, &rho_l, &rho_r, &u, &w_l, &w_r, 3).is_err());
    }

    #[test]
    fn duplicating_a_pair_to_inflate_count_is_rejected() {
        // Claim 5 rows by duplicating (0,2). Honest w gives Sum w_L*w_R = 4 != 5, so
        // the count identity rejects; the duplicate also breaks combined-index
        // monotonicity (checked first). Either way: REJECT.
        let (sf, of) = relation();
        let (_, _, _, _, _, u, w_l, w_r, _) = compute_self_join(&sf, &of);
        let x = frs(&[1, 1, 1, 2, 2]);
        let y = frs(&[5, 5, 5, 5, 5]);
        let z = frs(&[10, 10, 20, 10, 20]);
        let rho_l = frs(&[0, 0, 0, 1, 1]);
        let rho_r = frs(&[2, 2, 3, 2, 3]);
        assert!(attempt(&sf, &of, &x, &y, &z, &rho_l, &rho_r, &u, &w_l, &w_r, 5).is_err());
    }

    #[test]
    fn within_group_swap_is_rejected() {
        // {(0,2)x2,(1,3)x2}: same marginals/count as the honest join, every row a
        // valid pair -> memberships + count pass, but i_eval = [2,2,7,7] repeats ->
        // combined-index strict monotonicity rejects (the case the gather missed).
        let (sf, of) = relation();
        let (_, _, _, _, _, u, w_l, w_r, m_j) = compute_self_join(&sf, &of);
        let x = frs(&[1, 1, 2, 2]);
        let y = frs(&[5, 5, 5, 5]);
        let z = frs(&[10, 10, 20, 20]);
        let rho_l = frs(&[0, 0, 1, 1]);
        let rho_r = frs(&[2, 2, 3, 3]);
        assert!(attempt(&sf, &of, &x, &y, &z, &rho_l, &rho_r, &u, &w_l, &w_r, m_j).is_err());
    }

    #[test]
    fn fabricated_output_row_is_rejected() {
        // X[0] = 99 != Sf[rho_l[0]=0] = 1: the left index-augmented membership fails.
        let (sf, of) = relation();
        let (mut x, y, z, rho_l, rho_r, u, w_l, w_r, m_j) = compute_self_join(&sf, &of);
        x[0] = fr(99);
        assert!(attempt(&sf, &of, &x, &y, &z, &rho_l, &rho_r, &u, &w_l, &w_r, m_j).is_err());
    }

    #[test]
    fn wrong_multiplicity_is_rejected() {
        // Tamper w_L (claim key 5 appears 3x in Of, not 2x): the LEFT membership
        // residue (Of image of U under w_L) no longer holds -> REJECT.
        let (sf, of) = relation();
        let (x, y, z, rho_l, rho_r, u, mut w_l, w_r, m_j) = compute_self_join(&sf, &of);
        w_l[2] = w_l[2] + fr(1);
        assert!(attempt(&sf, &of, &x, &y, &z, &rho_l, &rho_r, &u, &w_l, &w_r, m_j).is_err());
    }

    #[test]
    fn empty_join_verifies() {
        // F with no self-matches: Of and Sf are disjoint value sets.
        let sf = frs(&[1, 2]);
        let of = frs(&[3, 4]);
        let (x, y, z, rho_l, rho_r, u, w_l, w_r, m_j) = compute_self_join(&sf, &of);
        assert_eq!(m_j, 0);
        assert!(attempt(&sf, &of, &x, &y, &z, &rho_l, &rho_r, &u, &w_l, &w_r, m_j).is_ok());
    }

    #[test]
    fn omission_via_duplicate_key_stashed_in_u_padding_is_rejected() {
        // THE REVIEWERS' EXACT OMISSION COUNTEREXAMPLE (audit FIX 2 + FIX 3).
        // F = {(1,5),(2,5),(5,10),(5,20)} has the complete self-join m_J = 4. The
        // attacker commits U = [1,2,5,10,20, 5,0,0] (claimed n_U = 5) with a DUPLICATE
        // key 5 STASHED at padding index 5, and splits its weight: w_L =
        // [0,0,1,1,1, 1,0,0], w_R = [1,1,1,0,0, 1,0,0]. Then Sum w_L*w_R = 2 over the
        // full domain, so the attacker outputs only 2 genuine pairs and claims m_J = 2.
        // BEFORE the fix (U side + count chi unmasked) this VERIFIES an INCOMPLETE
        // result; masking the U side (FIX 2) makes membership reject the split, and the
        // true-n_U count mask (FIX 3) rejects the undercount -> REJECT.
        let sf = frs(&[1, 2, 5, 5]);
        let of = frs(&[5, 5, 10, 20]);
        let (n_f, n_u, m_j, domain) = (4usize, 5usize, 2usize, 8usize);
        // Two GENUINE distinct pairs (so the pairs proof passes): (0,2) and (1,3).
        let x = frs(&[1, 2]);
        let y = frs(&[5, 5]);
        let z = frs(&[10, 20]);
        let rho_l = frs(&[0, 1]);
        let rho_r = frs(&[2, 3]);
        // Malicious witness columns at the domain (stash at index 5).
        let u_full = frs(&[1, 2, 5, 10, 20, 5, 0, 0]);
        let w_l_full = frs(&[0, 0, 1, 1, 1, 1, 0, 0]);
        let w_r_full = frs(&[1, 1, 1, 0, 0, 1, 0, 0]);
        let pad8 = |v: &[ArkFr]| padded(v, domain);
        let com = SelfJoinCommitments {
            sf: commit_column(&pad8(&sf)).unwrap(),
            of: commit_column(&pad8(&of)).unwrap(),
            x: commit_column(&pad8(&x)).unwrap(),
            y: commit_column(&pad8(&y)).unwrap(),
            z: commit_column(&pad8(&z)).unwrap(),
            rho_l: commit_column(&pad8(&rho_l)).unwrap(),
            rho_r: commit_column(&pad8(&rho_r)).unwrap(),
            u: commit_column(&u_full).unwrap(),
            w_l: commit_column(&w_l_full).unwrap(),
            w_r: commit_column(&w_r_full).unwrap(),
        };
        let mut pt = transcript();
        let proof = prove_self_join_complete_with_witness(
            &sf, &of, &x, &y, &z, &rho_l, &rho_r, &u_full, &w_l_full, &w_r_full, n_f, n_u, m_j,
            &com, &mut pt,
        )
        .unwrap();
        let mut vt = transcript();
        assert!(verify_self_join_complete(&com, n_f, n_u, m_j, &proof, &mut vt).is_err());
    }

    #[test]
    fn realistic_multi_key_self_join_verifies() {
        // Several keys with different left/right multiplicities.
        // F = {(1,2),(7,2),(2,9),(2,4),(4,5),(9,1)}:
        //   key 2: left rows with Of=2 -> {0,1} (Sf 1,7); right rows Sf=2 -> {2,3}.
        //   key 4: left Of=4 -> {3}; right Sf=4 -> {4}.
        //   key 9: left Of=9 -> {2}; right Sf=9 -> {5}.
        //   key 1: left Of=1 -> {5}; right Sf=1 -> {0}.
        let sf = frs(&[1, 7, 2, 2, 4, 9]);
        let of = frs(&[2, 2, 9, 4, 5, 1]);
        let (x, y, z, rho_l, rho_r, u, w_l, w_r, m_j) = compute_self_join(&sf, &of);
        // key 2 contributes 2*2 = 4; keys 1,4,9 contribute 1 each => m_J = 7.
        assert_eq!(m_j, 7);
        assert!(attempt(&sf, &of, &x, &y, &z, &rho_l, &rho_r, &u, &w_l, &w_r, m_j).is_ok());
    }
}
