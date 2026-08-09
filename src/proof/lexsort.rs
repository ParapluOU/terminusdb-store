//! Strict lexicographic ordering for two or three authenticated 64-bit key columns.

use std::io;
use ark_ff::PrimeField;
use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use super::argument::{commit_column, StoreProverContext, Transcript254};
use super::order::OrderDirection;
use super::sort::{prove_strictly_sorted_bits_with_context, verify_strictly_sorted_bits, SortedProof};
use super::invalid;

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

pub fn prove_lexicographically_sorted_with_context(
    context: &StoreProverContext<'_>, columns: &[Vec<ArkFr>], commitments: &[ArkGT],
    directions: &[OrderDirection], chi_commitment: &ArkGT, n: usize,
    transcript: &mut Transcript254,
) -> io::Result<LexicographicSortedProof> {
    let packed = pack(columns, directions, n)?;
    let commitment = packed_commitment(commitments, directions, chi_commitment)?;
    if context.commit_column(&packed)? != commitment { return Err(invalid("packed lexicographic commitment mismatch")); }
    let bits = 64 * columns.len();
    Ok(LexicographicSortedProof { sorted: prove_strictly_sorted_bits_with_context(context, &packed, n, bits, &commitment, transcript)? })
}

pub fn verify_lexicographically_sorted(
    commitments: &[ArkGT], directions: &[OrderDirection], chi_commitment: &ArkGT,
    n: usize, proof: &LexicographicSortedProof, transcript: &mut Transcript254,
) -> io::Result<()> {
    let commitment = packed_commitment(commitments, directions, chi_commitment)?;
    verify_strictly_sorted_bits(&commitment, n, 64 * commitments.len(), &proof.sorted, transcript)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn padded(values: &[u64]) -> Vec<ArkFr> {
        let mut out = values.iter().map(|v| ArkFr::from_u64(*v)).collect::<Vec<_>>();
        out.resize(4, ArkFr::zero()); out
    }

    #[test]
    fn mixed_direction_two_key_order_and_tamper() {
        let columns = vec![padded(&[1, 1, 2]), padded(&[9, 3, 8])];
        let commitments = columns.iter().map(|c| commit_column(c).unwrap()).collect::<Vec<_>>();
        let chi = padded(&[1, 1, 1]);
        let chi_commitment = commit_column(&chi).unwrap();
        let directions = [OrderDirection::Ascending, OrderDirection::Descending];
        let mut prover = Transcript254::new(b"lexsort-test");
        let proof = prove_lexicographically_sorted_with_context(&StoreProverContext::cpu(), &columns, &commitments, &directions, &chi_commitment, 3, &mut prover).unwrap();
        verify_lexicographically_sorted(&commitments, &directions, &chi_commitment, 3, &proof, &mut Transcript254::new(b"lexsort-test")).unwrap();
        let wrong = [OrderDirection::Ascending, OrderDirection::Ascending];
        assert!(verify_lexicographically_sorted(&commitments, &wrong, &chi_commitment, 3, &proof, &mut Transcript254::new(b"lexsort-test")).is_err());

        let packed = pack(&columns, &directions, 3).unwrap();
        assert!(packed[0].0.into_bigint() < packed[1].0.into_bigint());
        assert!(packed[1].0.into_bigint() < packed[2].0.into_bigint());
        assert_eq!(packed[3], ArkFr::zero(), "padding is excluded by chi");
        let duplicate = vec![padded(&[1, 1]), padded(&[3, 3])];
        let duplicate_packed = pack(&duplicate, &directions, 2).unwrap();
        assert_eq!(duplicate_packed[0], duplicate_packed[1], "identical full-row ties cannot pass strict sort");
    }
}
