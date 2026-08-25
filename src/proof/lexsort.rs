//! Nondecreasing lexicographic ordering for two or three authenticated 64-bit key columns.
//!
//! A [gadget](crate::proof::primer) (primer §6) that proves the ordering half of a
//! multi-column `ORDER BY`. It reduces multi-key lexicographic order to a single
//! nondecreasing-range check by packing the keys into one number, so the whole proof is
//! just [`super::sort`]'s bit-range sumcheck run once. See [the developer
//! primer](crate::proof::primer) for [commitments](crate::proof::primer) and their
//! homomorphism (§3), the [MLE/sumcheck](crate::proof::primer) workhorse (§5), and where
//! this sits in [prove → verify](crate::proof::primer) (§10).
//!
//! WHAT IT PROVES: given 2 or 3 committed key columns (each a 64-bit id) with a per-key
//! sort DIRECTION (ascending/descending), the real rows `[0, n)` are already sorted
//! lexicographically — row `i` is `<=` row `i+1` under the requested column priority and
//! directions. It does not reorder anything; it certifies an ordering the caller already
//! produced.
//!
//! HOW: the keys are packed into ONE column by a base-`2^64` positional number
//! (`key0 * 2^64^(k-1) + ... + key_{k-1}`), with descending columns bit-complemented so
//! "smaller packed value" always means "earlier in the requested order". Proving the
//! packed column is nondecreasing (via [`super::sort`]'s bit-range gadget over `64*k`
//! bits) is exactly proving lexicographic order of the originals. The packed column's
//! commitment is rebuilt homomorphically (primer §3) from the individual key commitments
//! and the public real-row mask, so it cannot be a different column than the committed
//! keys.
//!
//! WHERE: the ordering half of exact multi-key `OrderBy` — [`super::multiorder`] pairs
//! this with a permutation proof (rows are the same multiset) to get a full ORDER BY.

use std::io;
use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use super::argument::{StoreProverContext, Transcript254};
use super::order::OrderDirection;
use super::sort::{prove_nondecreasing_bits_with_context, verify_nondecreasing_bits, SortedProof};
use super::invalid;

/// A lexicographic-order proof: the nondecreasing-range proof over the single packed
/// key column (the pack is reconstructed by both sides, so it carries no extra data).
#[derive(Clone, Debug)]
pub struct LexicographicSortedProof { pub sorted: SortedProof }

fn radix() -> ArkFr { ArkFr::from_u64(u64::MAX) + ArkFr::one() }

fn packed_commitment(
    commitments: &[ArkGT], directions: &[OrderDirection], chi: &ArkGT,
) -> io::Result<ArkGT> {
    if !(2..=3).contains(&commitments.len()) || commitments.len() != directions.len() {
        return Err(invalid("lexicographic sort requires two or three keys"));
    }
    let max = ArkFr::from_u64(u64::MAX);
    let r = radix();
    let mut weight = ArkFr::one();
    let mut result = ArkGT::default();
    for (commitment, direction) in commitments.iter().zip(directions).rev() {
        let key = match direction {
            OrderDirection::Ascending => *commitment,
            OrderDirection::Descending => max * *chi - *commitment,
        };
        result = result + weight * key;
        weight = weight * r;
    }
    Ok(result)
}

fn pack(columns: &[Vec<ArkFr>], directions: &[OrderDirection], n: usize) -> io::Result<Vec<ArkFr>> {
    if !(2..=3).contains(&columns.len()) || columns.len() != directions.len() {
        return Err(invalid("lexicographic sort requires two or three keys"));
    }
    let domain = columns[0].len();
    if n > domain || columns.iter().any(|column| column.len() != domain) {
        return Err(invalid("lexicographic key column shape mismatch"));
    }
    let max = ArkFr::from_u64(u64::MAX);
    let r = radix();
    Ok((0..domain).map(|row| {
        if row >= n { return ArkFr::zero(); }
        columns.iter().zip(directions).fold(ArkFr::zero(), |acc, (column, direction)| {
            let value = match direction { OrderDirection::Ascending => column[row], OrderDirection::Descending => max - column[row] };
            acc * r + value
        })
    }).collect())
}

/// Prove the real rows `[0, n)` of the given key columns are nondecreasing in the
/// requested lexicographic order. `columns`/`commitments` are the 2 or 3 key columns and
/// their commitments; `directions` gives each key's ascending/descending sense;
/// `chi_commitment` commits the public real-row mask (1 on `[0, n)`, 0 on padding) used to
/// zero out padding in the descending bit-complement. The prover checks its packed column
/// matches the homomorphically-derived commitment before proving nondecreasing.
pub fn prove_lexicographically_nondecreasing_with_context(
    context: &StoreProverContext<'_>, columns: &[Vec<ArkFr>], commitments: &[ArkGT],
    directions: &[OrderDirection], chi_commitment: &ArkGT, n: usize,
    transcript: &mut Transcript254,
) -> io::Result<LexicographicSortedProof> {
    let packed = pack(columns, directions, n)?;
    let commitment = packed_commitment(commitments, directions, chi_commitment)?;
    if context.commit_column(&packed)? != commitment { return Err(invalid("packed lexicographic commitment mismatch")); }
    let bits = 64 * columns.len();
    Ok(LexicographicSortedProof { sorted: prove_nondecreasing_bits_with_context(context, &packed, n, bits, &commitment, transcript)? })
}

/// Verify a lexicographic-order proof. Rebuilds the packed-column commitment from the key
/// commitments, directions, and the real-row mask, then checks the packed column is
/// nondecreasing over `[0, n)`. On success the committed keys are in the requested order.
pub fn verify_lexicographically_nondecreasing(
    commitments: &[ArkGT], directions: &[OrderDirection], chi_commitment: &ArkGT,
    n: usize, proof: &LexicographicSortedProof, transcript: &mut Transcript254,
) -> io::Result<()> {
    let commitment = packed_commitment(commitments, directions, chi_commitment)?;
    verify_nondecreasing_bits(&commitment, n, 64 * commitments.len(), &proof.sorted, transcript)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::PrimeField;
    use super::super::argument::commit_column;

    fn padded(values: &[u64]) -> Vec<ArkFr> {
        let mut out = values.iter().map(|v| ArkFr::from_u64(*v)).collect::<Vec<_>>();
        out.resize(4, ArkFr::zero()); out
    }

    #[test]
    fn mixed_direction_two_key_nondecreasing_ties_and_tamper() {
        let columns = vec![padded(&[1, 1, 1]), padded(&[9, 9, 3])];
        let commitments = columns.iter().map(|c| commit_column(c).unwrap()).collect::<Vec<_>>();
        let chi = padded(&[1, 1, 1]);
        let chi_commitment = commit_column(&chi).unwrap();
        let directions = [OrderDirection::Ascending, OrderDirection::Descending];
        let mut prover = Transcript254::new(b"lexsort-test");
        let proof = prove_lexicographically_nondecreasing_with_context(&StoreProverContext::cpu(), &columns, &commitments, &directions, &chi_commitment, 3, &mut prover).unwrap();
        verify_lexicographically_nondecreasing(&commitments, &directions, &chi_commitment, 3, &proof, &mut Transcript254::new(b"lexsort-test")).unwrap();
        let wrong = [OrderDirection::Ascending, OrderDirection::Ascending];
        assert!(verify_lexicographically_nondecreasing(&commitments, &wrong, &chi_commitment, 3, &proof, &mut Transcript254::new(b"lexsort-test")).is_err());

        let packed = pack(&columns, &directions, 3).unwrap();
        assert_eq!(packed[0], packed[1]);
        assert!(packed[1].0.into_bigint() < packed[2].0.into_bigint());
        assert_eq!(packed[3], ArkFr::zero(), "padding is excluded by chi");
        let duplicate = vec![padded(&[1, 1]), padded(&[3, 3])];
        let duplicate_packed = pack(&duplicate, &directions, 2).unwrap();
        assert_eq!(duplicate_packed[0], duplicate_packed[1]);
    }
}
