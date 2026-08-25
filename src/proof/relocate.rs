//! Cross-domain column RELOCATION (Phase 2c-3 reconciliation): prove that two Dory
//! commitments to the SAME logical column, made at DIFFERENT power-of-two domains,
//! encode the same zero-padded vector. Clean-room; built on the shared opening glue.
//!
//! Why: the Q1 FILTER commits its dense output `F = (Sf, Of)` at the TABLE domain
//! `2^a = next_pow2(n_table)` (its input columns are the layer's committed columns,
//! pinned by `commitment_root`). The Q2 SELF-JOIN needs `F` committed at
//! `self_join_domain(n_F, m_J, n_U)`, which can EXCEED the table domain under join
//! blow-up (`m_J` up to `n_F^2`). A Dory GT commitment is domain-specific, so one
//! `Com(F)` cannot serve both. This gadget ties a fresh re-commitment of `F` at the
//! join domain to the filter's output commitment, so the whole chain stays anchored
//! to the root.
//!
//! Soundness. Let `source` be committed at domain `2^a` and `target` at `2^b`, each a
//! zero-padded copy of one short vector `v` (`|v| <= 2^min(a,b)`). A zero-padded
//! column's multilinear extension factors, because the padding (high) bits are all
//! zero, as
//!   embed_to_max(source)(x) = MLE_source(x_0..x_{a-1}) * prod_{j=a}^{max-1}(1 - x_j)
//! (max = max(a,b)); likewise for `target`. `source` and `target` encode the same
//! logical column iff their embeddings into the common `2^max` cube are equal as
//! MLEs, i.e. for a random `r`
//!   MLE_source(r[0..a]) * prod_{j=a}^{max-1}(1 - r_j)
//!     == MLE_target(r[0..b]) * prod_{j=b}^{max-1}(1 - r_j).
//! We Fiat-Shamir-derive `r`, BIND both evals with Dory openings (so the prover
//! cannot lie about `MLE_source(r)` / `MLE_target(r)`), and check the scalar
//! identity. By Schwartz-Zippel a mismatched pair (a differing real entry, or any
//! nonzero padding within either domain) fails with overwhelming probability. This
//! also forces the two real prefixes to agree POSITIONALLY, so `Sf`/`Of` keep the
//! filter's row order (which the self-join's `rho` index pointers reference).
//!
//! For a developer: this is the "same column, two hypercube sizes" [gadget](crate::proof::primer).
//! It proves nothing about the *data* beyond "these two commitments are the same padded column";
//! it runs no [sumcheck](crate::proof::primer) of its own, just two Dory openings at one
//! Fiat–Shamir point and a scalar check. Every column here has a power-of-two length `>= 2`
//! (log-domain `>= 1`): that is the [MLE](crate::proof::primer) hypercube requirement — a
//! `2^k`-vector is indexed by the corners of `{0,1}^k`, so it needs at least one variable to open
//! at, and a length-1 column (`k = 0`) has none (see the private `domain_log` helper). See [the
//! primer](crate::proof::primer) for [commitments and their homomorphism](crate::proof::primer)
//! (§3), [MLE + sumcheck](crate::proof::primer) (§5), and the [Fiat–Shamir
//! transcript](crate::proof::primer) (§8).

use std::io;

use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::transcript::Transcript;

use super::argument::{
    mle_full_eval, verify_column_openings, ColumnOpening,
    StoreProverContext, Transcript254,
};
use super::invalid;

const LABEL_DOMAIN: &[u8] = b"pow/relocate/domain-v1";
const LABEL_DIM: &[u8] = b"pow/relocate/dim";
const LABEL_POINT: &[u8] = b"pow/relocate/point";

#[inline]
fn f1() -> ArkFr {
    ArkFr::one()
}

/// A cross-domain relocation proof: the two claimed evals and their Dory openings.
#[derive(Clone, Debug)]
pub struct RelocationProof {
    /// `MLE_source(r[0..a])`: the source column's multilinear extension at the low `a`
    /// coordinates of the shared challenge point `r`.
    pub source_eval: ArkFr,
    /// `MLE_target(r[0..b])`: the target column's multilinear extension at the low `b`
    /// coordinates of `r`.
    pub target_eval: ArkFr,
    /// Dory opening binding `source_eval` to `com_source` (so the prover cannot misreport it).
    pub source_opening: ColumnOpening,
    /// Dory opening binding `target_eval` to `com_target`.
    pub target_opening: ColumnOpening,
}

/// The log-domain `k` of a committed column of length `len = 2^k`. Requires a
/// power-of-two length that is at least 2 (so `k >= 1`): a Dory-committed column is the
/// multilinear extension over the boolean hypercube `{0,1}^k`, so it needs at least one
/// variable to open at — a length-1 column (`k == 0`) has no hypercube variable. Columns
/// are always zero-padded up to the next power of two, hence the power-of-two shape. See
/// the primer's [MLE + sumcheck](crate::proof::primer) section (§5).
fn domain_log(len: usize) -> io::Result<usize> {
    if !len.is_power_of_two() || len < 2 {
        return Err(invalid(
            "relocation column length must be a power of two >= 2",
        ));
    }
    Ok(len.trailing_zeros() as usize)
}

fn bind(
    transcript: &mut Transcript254,
    com_source: &ArkGT,
    com_target: &ArkGT,
    a: usize,
    b: usize,
) {
    transcript.append_bytes(LABEL_DOMAIN, b"parture/proof-of-woql/relocate/v0");
    transcript.append_bytes(LABEL_DIM, &(a as u64).to_le_bytes());
    transcript.append_bytes(LABEL_DIM, &(b as u64).to_le_bytes());
    transcript.append_group(b"pow/relocate/com-source", com_source);
    transcript.append_group(b"pow/relocate/com-target", com_target);
}

fn challenge_point(transcript: &mut Transcript254, len: usize) -> Vec<ArkFr> {
    (0..len)
        .map(|_| transcript.challenge_scalar(LABEL_POINT))
        .collect()
}

/// `prod_j (1 - r_j)` over the high padding coordinates of one domain.
fn padding_factor(high: &[ArkFr]) -> ArkFr {
    high.iter().fold(f1(), |acc, r| acc * (f1() - *r))
}

/// Prove `source` (committed as `com_source` at domain `2^a`) and `target`
/// (committed as `com_target` at domain `2^b`) encode the SAME zero-padded column.
pub fn prove_column_relocation(
    source: &[ArkFr],
    com_source: &ArkGT,
    target: &[ArkFr],
    com_target: &ArkGT,
    transcript: &mut Transcript254,
) -> io::Result<RelocationProof> {
    prove_column_relocation_with_context(
        &StoreProverContext::cpu(),
        source,
        com_source,
        target,
        com_target,
        transcript,
    )
}

/// As [`prove_column_relocation`], with an explicit prover-arithmetic context (the choice
/// of context cannot change the proof bytes or the guarantee).
pub fn prove_column_relocation_with_context(
    context: &StoreProverContext<'_>,
    source: &[ArkFr],
    com_source: &ArkGT,
    target: &[ArkFr],
    com_target: &ArkGT,
    transcript: &mut Transcript254,
) -> io::Result<RelocationProof> {
    let a = domain_log(source.len())?;
    let b = domain_log(target.len())?;
    let max = a.max(b);

    bind(transcript, com_source, com_target, a, b);
    let r = challenge_point(transcript, max);

    let source_eval = mle_full_eval(source, &r[..a]);
    let target_eval = mle_full_eval(target, &r[..b]);
    let source_opening = context.prove_column_openings(&[source.to_vec()], &r[..a], transcript)?;
    let target_opening = context.prove_column_openings(&[target.to_vec()], &r[..b], transcript)?;

    Ok(RelocationProof {
        source_eval,
        target_eval,
        source_opening,
        target_opening,
    })
}

/// Verify a cross-domain relocation proof. `a`/`b` are the log-domains of
/// `com_source`/`com_target` (pinned by the respective column commitments, since the
/// Dory openings verify against a length-`a`/`b` point).
pub fn verify_column_relocation(
    com_source: &ArkGT,
    a: usize,
    com_target: &ArkGT,
    b: usize,
    proof: &RelocationProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    if a == 0 || b == 0 || a >= usize::BITS as usize || b >= usize::BITS as usize {
        return Err(invalid("relocation domain out of range"));
    }
    let max = a.max(b);

    bind(transcript, com_source, com_target, a, b);
    let r = challenge_point(transcript, max);

    // Bind the two evals to the commitments (the prover cannot misreport them).
    verify_column_openings(
        &[*com_source],
        &r[..a],
        &[proof.source_eval],
        &proof.source_opening,
        transcript,
    )?;
    verify_column_openings(
        &[*com_target],
        &r[..b],
        &[proof.target_eval],
        &proof.target_opening,
        transcript,
    )?;

    // Embed-to-max consistency: the two columns are equal as zero-padded vectors.
    let p_source = padding_factor(&r[a..max]);
    let p_target = padding_factor(&r[b..max]);
    if proof.source_eval * p_source != proof.target_eval * p_target {
        return Err(invalid("cross-domain relocation consistency check failed"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof::argument::commit_column;

    fn fr(v: u64) -> ArkFr {
        ArkFr::from_u64(v)
    }

    fn frs(vals: &[u64]) -> Vec<ArkFr> {
        vals.iter().map(|v| fr(*v)).collect()
    }

    fn transcript() -> Transcript254 {
        Transcript254::new(b"parture-relocate-test")
    }

    fn zero_padded(v: &[ArkFr], log_len: usize) -> Vec<ArkFr> {
        let mut out = v.to_vec();
        out.resize(1usize << log_len, ArkFr::zero());
        out
    }

    // Relocate a length-`prefix` vector from domain 2^a to domain 2^b and verify.
    fn run(prefix: &[ArkFr], a: usize, b: usize) -> io::Result<()> {
        let source = zero_padded(prefix, a);
        let target = zero_padded(prefix, b);
        let com_source = commit_column(&source).unwrap();
        let com_target = commit_column(&target).unwrap();
        let mut pt = transcript();
        let proof = prove_column_relocation(&source, &com_source, &target, &com_target, &mut pt)?;
        let mut vt = transcript();
        verify_column_relocation(&com_source, a, &com_target, b, &proof, &mut vt)
    }

    #[test]
    fn same_vector_smaller_to_larger_verifies() {
        assert!(run(&frs(&[1, 2, 3]), 3, 4).is_ok()); // 8 -> 16
    }

    #[test]
    fn same_vector_larger_to_smaller_verifies() {
        assert!(run(&frs(&[7, 8, 9, 10, 11]), 4, 3).is_ok()); // 16 -> 8
    }

    #[test]
    fn same_domain_verifies() {
        assert!(run(&frs(&[42, 43]), 3, 3).is_ok());
    }

    #[test]
    fn single_and_empty_prefix_verify() {
        assert!(run(&frs(&[99]), 1, 4).is_ok());
        assert!(run(&[], 2, 5).is_ok()); // both all-zero
    }

    #[test]
    fn differing_real_entry_is_rejected() {
        // target's real prefix differs at index 1 -> not the same column.
        let source = zero_padded(&frs(&[1, 2, 3]), 3);
        let mut target = zero_padded(&frs(&[1, 2, 3]), 4);
        target[1] = fr(999);
        let com_source = commit_column(&source).unwrap();
        let com_target = commit_column(&target).unwrap();
        let mut pt = transcript();
        let proof =
            prove_column_relocation(&source, &com_source, &target, &com_target, &mut pt).unwrap();
        let mut vt = transcript();
        assert!(verify_column_relocation(&com_source, 3, &com_target, 4, &proof, &mut vt).is_err());
    }

    #[test]
    fn nonzero_padding_in_the_larger_domain_is_rejected() {
        // target has a stray nonzero entry beyond the real prefix (index 10).
        let source = zero_padded(&frs(&[1, 2, 3]), 3);
        let mut target = zero_padded(&frs(&[1, 2, 3]), 4);
        target[10] = fr(5);
        let com_source = commit_column(&source).unwrap();
        let com_target = commit_column(&target).unwrap();
        let mut pt = transcript();
        let proof =
            prove_column_relocation(&source, &com_source, &target, &com_target, &mut pt).unwrap();
        let mut vt = transcript();
        assert!(verify_column_relocation(&com_source, 3, &com_target, 4, &proof, &mut vt).is_err());
    }

    #[test]
    fn wrong_commitment_is_rejected() {
        // Verify against a commitment that does not match the opened column.
        let source = zero_padded(&frs(&[1, 2, 3]), 3);
        let target = zero_padded(&frs(&[1, 2, 3]), 4);
        let com_source = commit_column(&source).unwrap();
        let com_target = commit_column(&target).unwrap();
        let bogus = commit_column(&zero_padded(&frs(&[4, 5, 6]), 4)).unwrap();
        let mut pt = transcript();
        let proof =
            prove_column_relocation(&source, &com_source, &target, &com_target, &mut pt).unwrap();
        let mut vt = transcript();
        assert!(verify_column_relocation(&com_source, 3, &bogus, 4, &proof, &mut vt).is_err());
    }
}
