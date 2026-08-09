//! Authenticated correlated negation as a left anti-join.
//!
//! For committed relations `L` and `R`, this operator retains exactly those rows of
//! `L` whose scalar join key has zero matching rows in `R`.  It composes three audited
//! gadgets: the complete binary join authenticates every per-left match multiplicity;
//! the filter IsZero argument pins a selection bit to `multiplicity == 0`; and exact
//! compaction emits the selected left rows.  Thus duplicate right matches cannot change
//! the result, while omitting even one matching key cannot make a left row survive.

use std::io;

use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::transcript::Transcript;

use super::argument::Transcript254;
use super::bgp::{RelationCommit, RelationWitness};
use super::filter::{
    prove_compaction_only, prove_filter_indicator, verify_compaction_only, verify_filter_indicator,
    CompactionOnlyProof, FilterProof,
};
use super::invalid;
use super::join2::{
    binary_join_commitments, binary_join_domain, compute_binary_join, prove_binary_join,
    verify_binary_join, BinaryJoinCommitments, BinaryJoinProof,
};
use super::relocate::{prove_column_relocation, verify_column_relocation, RelocationProof};

const ANTI_DOMAIN: &[u8] = b"parture/proof-of-woql/left-anti-join/v0";

fn f0() -> ArkFr {
    ArkFr::zero()
}

fn pad(values: &[ArkFr], size: usize) -> Vec<ArkFr> {
    let mut result = values.to_vec();
    result.resize(size, f0());
    result
}

fn validate_relation(relation: &RelationCommit, name: &str) -> io::Result<()> {
    if relation.commitments.is_empty() {
        return Err(invalid(format!("{name} anti-join relation has no columns")));
    }
    if relation.log_domain == 0 || relation.log_domain >= usize::BITS as usize {
        return Err(invalid(format!("{name} anti-join domain out of range")));
    }
    if relation.n > (1usize << relation.log_domain) {
        return Err(invalid(format!(
            "{name} anti-join length exceeds its domain"
        )));
    }
    Ok(())
}

fn validate_witness(relation: &RelationWitness, name: &str) -> io::Result<()> {
    validate_relation(&relation.view(), name)?;
    if relation.columns.len() != relation.commitments.len()
        || relation
            .columns
            .iter()
            .any(|column| column.len() != 1usize << relation.log_domain)
    {
        return Err(invalid(format!("{name} anti-join witness shape mismatch")));
    }
    Ok(())
}

fn bind_statement(
    transcript: &mut Transcript254,
    left: &RelationCommit,
    right: &RelationCommit,
    left_key: usize,
    right_key: usize,
    output_log_domain: usize,
) {
    transcript.append_bytes(b"pow/anti/domain", ANTI_DOMAIN);
    for value in [
        left.log_domain,
        left.n,
        right.log_domain,
        right.n,
        left_key,
        right_key,
        output_log_domain,
    ] {
        transcript.append_bytes(b"pow/anti/meta", &(value as u64).to_le_bytes());
    }
    for commitment in &left.commitments {
        transcript.append_group(b"pow/anti/left", commitment);
    }
    for commitment in &right.commitments {
        transcript.append_group(b"pow/anti/right", commitment);
    }
}

fn split_key<'a>(
    columns: &'a [Vec<ArkFr>],
    key: usize,
    n: usize,
) -> (&'a [ArkFr], Vec<&'a [ArkFr]>) {
    let key_column = &columns[key][..n];
    let others = columns
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != key)
        .map(|(_, column)| &column[..n])
        .collect();
    (key_column, others)
}

fn input_commitments(
    com: &BinaryJoinCommitments,
    left: bool,
    key: usize,
    arity: usize,
) -> Vec<ArkGT> {
    let (key_commitment, others) = if left {
        (com.r_key, &com.r_other)
    } else {
        (com.s_key, &com.s_other)
    };
    let mut result = Vec::with_capacity(arity);
    let mut other = 0usize;
    for column in 0..arity {
        if column == key {
            result.push(key_commitment);
        } else {
            result.push(others[other]);
            other += 1;
        }
    }
    result
}

fn multiplicities(rho: &[ArkFr], left_rows: usize, join_rows: usize, size: usize) -> Vec<ArkFr> {
    let mut result = vec![f0(); size];
    for row in 0..left_rows {
        let index = ArkFr::from_u64(row as u64);
        result[row] = ArkFr::from_u64(
            rho[..join_rows]
                .iter()
                .filter(|candidate| **candidate == index)
                .count() as u64,
        );
    }
    result
}

#[derive(Clone, Debug)]
pub struct LeftAntiJoinProof {
    pub left_relocations: Vec<RelocationProof>,
    pub right_relocations: Vec<RelocationProof>,
    pub join_commitments: BinaryJoinCommitments,
    pub join: BinaryJoinProof,
    pub join_len: usize,
    pub zero: FilterProof,
    pub zero_selection_commitment: ArkGT,
    pub zero_inverse_commitment: ArkGT,
    pub compaction: CompactionOnlyProof,
}

pub struct LeftAntiJoinResult {
    pub proof: LeftAntiJoinProof,
    pub commitments: Vec<ArkGT>,
    pub columns: Vec<Vec<ArkFr>>,
    pub log_domain: usize,
    pub n: usize,
}

/// Prove a correlated `Not` for exactly one shared (scalar Store-ID) key column.
pub fn prove_left_anti_join(
    left: &RelationWitness,
    right: &RelationWitness,
    left_key: usize,
    right_key: usize,
    transcript: &mut Transcript254,
) -> io::Result<LeftAntiJoinResult> {
    validate_witness(left, "left")?;
    validate_witness(right, "right")?;
    if left_key >= left.columns.len() || right_key >= right.columns.len() {
        return Err(invalid("anti-join key column out of range"));
    }

    let left_real_key = &left.columns[left_key][..left.n];
    let right_real_key = &right.columns[right_key][..right.n];
    let left_real_other: Vec<&[ArkFr]> = left
        .columns
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != left_key)
        .map(|(_, column)| &column[..left.n])
        .collect();
    let right_real_other: Vec<&[ArkFr]> = right
        .columns
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != right_key)
        .map(|(_, column)| &column[..right.n])
        .collect();
    let witness = compute_binary_join(
        left_real_key,
        &left_real_other,
        right_real_key,
        &right_real_other,
    );
    let size = binary_join_domain(left.n, right.n, witness.m_j, witness.n_u)?;
    let log_domain = size.trailing_zeros() as usize;

    let left_columns: Vec<Vec<ArkFr>> = left
        .columns
        .iter()
        .map(|c| pad(&c[..left.n], size))
        .collect();
    let right_columns: Vec<Vec<ArkFr>> = right
        .columns
        .iter()
        .map(|c| pad(&c[..right.n], size))
        .collect();
    let (left_join_key, left_join_other) = split_key(&left_columns, left_key, left.n);
    let (right_join_key, right_join_other) = split_key(&right_columns, right_key, right.n);
    let join_commitments = binary_join_commitments(
        left_join_key,
        &left_join_other,
        right_join_key,
        &right_join_other,
        &witness,
    )?;
    let left_targets = input_commitments(&join_commitments, true, left_key, left.columns.len());
    let right_targets = input_commitments(&join_commitments, false, right_key, right.columns.len());

    let mut left_relocations = Vec::with_capacity(left.columns.len());
    let mut right_relocations = Vec::with_capacity(right.columns.len());
    // The final output is not known until zero-test compaction, so bind the immutable
    // input statement now; the output commitments are bound by compaction below.
    bind_statement(
        transcript,
        &left.view(),
        &right.view(),
        left_key,
        right_key,
        log_domain,
    );
    for ((source, source_commitment), (target, target_commitment)) in left
        .columns
        .iter()
        .zip(&left.commitments)
        .zip(left_columns.iter().zip(&left_targets))
    {
        left_relocations.push(prove_column_relocation(
            source,
            source_commitment,
            target,
            target_commitment,
            transcript,
        )?);
    }
    for ((source, source_commitment), (target, target_commitment)) in right
        .columns
        .iter()
        .zip(&right.commitments)
        .zip(right_columns.iter().zip(&right_targets))
    {
        right_relocations.push(prove_column_relocation(
            source,
            source_commitment,
            target,
            target_commitment,
            transcript,
        )?);
    }
    let (join, _, join_len) = prove_binary_join(
        left_join_key,
        &left_join_other,
        right_join_key,
        &right_join_other,
        &join_commitments,
        transcript,
    )?;
    let mult = multiplicities(&witness.rho_l, left.n, join_len, size);
    let (zero, zero_selection_commitment, zero_inverse_commitment, _) = prove_filter_indicator(
        &mult,
        f0(),
        left.n,
        &join.pairs.mult_l_commitment,
        transcript,
    )?;
    let selection: Vec<ArkFr> = mult
        .iter()
        .enumerate()
        .map(|(index, value)| {
            if index < left.n && value.is_zero() {
                ArkFr::one()
            } else {
                f0()
            }
        })
        .collect();
    let (compaction, commitments, columns, n) = prove_compaction_only(
        &left_columns,
        &left_targets,
        &selection,
        &zero_selection_commitment,
        transcript,
    )?;

    Ok(LeftAntiJoinResult {
        proof: LeftAntiJoinProof {
            left_relocations,
            right_relocations,
            join_commitments,
            join,
            join_len,
            zero,
            zero_selection_commitment,
            zero_inverse_commitment,
            compaction,
        },
        commitments,
        columns,
        log_domain,
        n,
    })
}

/// Verify the exact scalar-key left anti-join result.
pub fn verify_left_anti_join(
    left: &RelationCommit,
    right: &RelationCommit,
    left_key: usize,
    right_key: usize,
    output_commitments: &[ArkGT],
    output_len: usize,
    output_log_domain: usize,
    proof: &LeftAntiJoinProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    validate_relation(left, "left")?;
    validate_relation(right, "right")?;
    if left_key >= left.commitments.len() || right_key >= right.commitments.len() {
        return Err(invalid("anti-join key column out of range"));
    }
    if output_commitments.len() != left.commitments.len() {
        return Err(invalid("anti-join output arity mismatch"));
    }
    if proof.left_relocations.len() != left.commitments.len()
        || proof.right_relocations.len() != right.commitments.len()
    {
        return Err(invalid("anti-join relocation shape mismatch"));
    }
    if proof.join.num_vars() != output_log_domain {
        return Err(invalid("anti-join output/join domain mismatch"));
    }

    bind_statement(
        transcript,
        left,
        right,
        left_key,
        right_key,
        output_log_domain,
    );
    let left_targets = input_commitments(
        &proof.join_commitments,
        true,
        left_key,
        left.commitments.len(),
    );
    let right_targets = input_commitments(
        &proof.join_commitments,
        false,
        right_key,
        right.commitments.len(),
    );
    for column in 0..left.commitments.len() {
        verify_column_relocation(
            &left.commitments[column],
            left.log_domain,
            &left_targets[column],
            output_log_domain,
            &proof.left_relocations[column],
            transcript,
        )?;
    }
    for column in 0..right.commitments.len() {
        verify_column_relocation(
            &right.commitments[column],
            right.log_domain,
            &right_targets[column],
            output_log_domain,
            &proof.right_relocations[column],
            transcript,
        )?;
    }
    verify_binary_join(
        &proof.join_commitments,
        left.n,
        right.n,
        proof.join_len,
        &proof.join,
        transcript,
    )?;
    verify_filter_indicator(
        &proof.join.pairs.mult_l_commitment,
        f0(),
        left.n,
        &proof.zero_selection_commitment,
        &proof.zero_inverse_commitment,
        &proof.zero,
        transcript,
    )?;
    verify_compaction_only(
        &left_targets,
        &proof.zero_selection_commitment,
        output_commitments,
        output_len,
        &proof.compaction,
        transcript,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn relation(columns: &[&[u64]]) -> RelationWitness {
        let n = columns[0].len();
        RelationWitness::commit(
            columns
                .iter()
                .map(|column| column.iter().map(|value| ArkFr::from_u64(*value)).collect())
                .collect(),
            n,
        )
        .unwrap()
    }

    fn transcript() -> Transcript254 {
        Transcript254::new(b"parture-left-anti-test")
    }

    #[test]
    fn retains_exactly_zero_match_left_rows_with_right_duplicates() {
        // The two key=2 rows have different payloads. Multiplicity is indexed by left
        // row identity, so both must be excluded independently (and each sees both
        // duplicate right matches).
        let left = relation(&[&[1, 2, 2, 3, 4], &[10, 20, 21, 30, 40]]);
        let right = relation(&[&[2, 2, 4], &[200, 201, 400]]);
        let mut pt = transcript();
        let result = prove_left_anti_join(&left, &right, 0, 0, &mut pt).unwrap();
        assert_eq!(result.n, 2);
        assert_eq!(
            &result.columns[0][..2],
            &[ArkFr::from_u64(1), ArkFr::from_u64(3)]
        );
        assert_eq!(
            &result.columns[1][..2],
            &[ArkFr::from_u64(10), ArkFr::from_u64(30)]
        );
        let mut vt = transcript();
        verify_left_anti_join(
            &left.view(),
            &right.view(),
            0,
            0,
            &result.commitments,
            result.n,
            result.log_domain,
            &result.proof,
            &mut vt,
        )
        .unwrap();

        let mut wrong = result.commitments.clone();
        wrong.swap(0, 1);
        let mut vt = transcript();
        assert!(verify_left_anti_join(
            &left.view(),
            &right.view(),
            0,
            0,
            &wrong,
            result.n,
            result.log_domain,
            &result.proof,
            &mut vt,
        )
        .is_err());
        // The verifier must consume the dense compaction commitments, never accept the
        // pre-compaction left branch as the claimed Not result.
        let mut vt = transcript();
        assert!(verify_left_anti_join(
            &left.view(),
            &right.view(),
            0,
            0,
            &left.commitments,
            result.n,
            result.log_domain,
            &result.proof,
            &mut vt,
        )
        .is_err());
        for forged_len in [1, 3] {
            let mut vt = transcript();
            assert!(verify_left_anti_join(
                &left.view(),
                &right.view(),
                0,
                0,
                &result.commitments,
                forged_len,
                result.log_domain,
                &result.proof,
                &mut vt,
            )
            .is_err());
        }
    }

    #[test]
    fn handles_empty_and_all_matching_right_relations() {
        let left = relation(&[&[1, 2], &[10, 20]]);
        for (right, expected) in [
            (relation(&[&[], &[]]), 2usize),
            (relation(&[&[1, 2], &[100, 200]]), 0usize),
        ] {
            let mut pt = transcript();
            let result = prove_left_anti_join(&left, &right, 0, 0, &mut pt).unwrap();
            assert_eq!(result.n, expected);
            let mut vt = transcript();
            verify_left_anti_join(
                &left.view(),
                &right.view(),
                0,
                0,
                &result.commitments,
                result.n,
                result.log_domain,
                &result.proof,
                &mut vt,
            )
            .unwrap();
        }
    }
}
