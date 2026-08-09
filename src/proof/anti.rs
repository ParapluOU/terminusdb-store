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
use dory_pcs::primitives::arithmetic::{Field, Group};
use dory_pcs::primitives::transcript::Transcript;

use super::argument::{commit_column, Transcript254};
use super::bgp::{RelationCommit, RelationWitness};
use super::filter::{
    prove_compaction_only, prove_filter_indicator, verify_compaction_only, verify_filter_indicator,
    CompactionOnlyProof, FilterProof,
};
use super::invalid;
use super::join2::{
    binary_join_commitments, binary_join_domain, compute_binary_join, prove_binary_join,
    prove_binary_join_with_key_bits, verify_binary_join, verify_binary_join_with_key_bits,
    BinaryJoinCommitments, BinaryJoinProof,
};
use super::relocate::{prove_column_relocation, verify_column_relocation, RelocationProof};
use super::union::{prove_bag_union, verify_bag_union, BagUnionProof};

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
    key_pairs: &[(usize, usize)],
    output_log_domain: usize,
) {
    transcript.append_bytes(b"pow/anti/domain", ANTI_DOMAIN);
    for value in [
        left.log_domain,
        left.n,
        right.log_domain,
        right.n,
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
    transcript.append_bytes(b"pow/anti/key-count", &(key_pairs.len() as u64).to_le_bytes());
    for (left_key, right_key) in key_pairs {
        transcript.append_bytes(b"pow/anti/left-key", &(*left_key as u64).to_le_bytes());
        transcript.append_bytes(b"pow/anti/right-key", &(*right_key as u64).to_le_bytes());
    }
}

fn two_word_radix() -> ArkFr {
    let mut radix = ArkFr::one();
    for _ in 0..64 {
        radix = radix + radix;
    }
    radix
}

fn composite_key(columns: &[Vec<ArkFr>], keys: &[usize], n: usize) -> Vec<ArkFr> {
    let radix = two_word_radix();
    (0..n)
        .map(|row| columns[keys[0]][row] + radix * columns[keys[1]][row])
        .collect()
}

fn composite_commitment(commitments: &[ArkGT], keys: &[usize]) -> ArkGT {
    commitments[keys[0]].add(&commitments[keys[1]].scale(&two_word_radix()))
}

fn validate_keys(
    left_arity: usize,
    right_arity: usize,
    key_pairs: &[(usize, usize)],
) -> io::Result<()> {
    if !(1..=2).contains(&key_pairs.len()) {
        return Err(invalid("anti-join needs one or two shared key columns"));
    }
    if key_pairs
        .iter()
        .any(|(left, right)| *left >= left_arity || *right >= right_arity)
    {
        return Err(invalid("anti-join key column out of range"));
    }
    if key_pairs.len() == 2
        && (key_pairs[0].0 == key_pairs[1].0 || key_pairs[0].1 == key_pairs[1].1)
    {
        return Err(invalid("anti-join composite key repeats a column"));
    }
    Ok(())
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
    /// Complete matched inner-join relation in schema `[left.., right-only..]`.
    pub matched_commitments: Vec<ArkGT>,
    pub matched_columns: Vec<Vec<ArkFr>>,
}

#[derive(Clone, Debug)]
pub struct LeftOuterJoinProof {
    pub anti: LeftAntiJoinProof,
    pub unmatched_left_commitments: Vec<ArkGT>,
    pub unmatched_len: usize,
    pub union: BagUnionProof,
}

pub struct LeftOuterJoinResult {
    pub proof: LeftOuterJoinProof,
    pub commitments: Vec<ArkGT>,
    pub columns: Vec<Vec<ArkFr>>,
    pub log_domain: usize,
    pub n: usize,
}

fn matched_projection(
    left_arity: usize,
    right_arity: usize,
    key_pairs: &[(usize, usize)],
    commitments: &BinaryJoinCommitments,
    witness: Option<&super::join2::BinaryJoinWitness>,
) -> io::Result<(Vec<ArkGT>, Option<Vec<Vec<ArkFr>>>)> {
    let composite = key_pairs.len() == 2;
    let right_keys: Vec<usize> = key_pairs.iter().map(|pair| pair.1).collect();
    let mut output_commitments = Vec::with_capacity(left_arity + right_arity - key_pairs.len());
    let mut output_columns = witness.map(|_| Vec::with_capacity(output_commitments.capacity()));

    if composite {
        for index in 0..left_arity {
            output_commitments.push(commitments.r_other_out[index]);
            if let (Some(witness), Some(columns)) = (witness, output_columns.as_mut()) {
                columns.push(witness.r_other_out[index].clone());
            }
        }
        for index in 0..right_arity {
            if !right_keys.contains(&index) {
                output_commitments.push(commitments.s_other_out[index]);
                if let (Some(witness), Some(columns)) = (witness, output_columns.as_mut()) {
                    columns.push(witness.s_other_out[index].clone());
                }
            }
        }
    } else {
        let (left_key, right_key) = key_pairs[0];
        let mut left_other = 0;
        for index in 0..left_arity {
            if index == left_key {
                output_commitments.push(commitments.key);
                if let (Some(witness), Some(columns)) = (witness, output_columns.as_mut()) {
                    columns.push(witness.key.clone());
                }
            } else {
                output_commitments.push(commitments.r_other_out[left_other]);
                if let (Some(witness), Some(columns)) = (witness, output_columns.as_mut()) {
                    columns.push(witness.r_other_out[left_other].clone());
                }
                left_other += 1;
            }
        }
        let mut right_other = 0;
        for index in 0..right_arity {
            if index == right_key {
                continue;
            }
            output_commitments.push(commitments.s_other_out[right_other]);
            if let (Some(witness), Some(columns)) = (witness, output_columns.as_mut()) {
                columns.push(witness.s_other_out[right_other].clone());
            }
            right_other += 1;
        }
    }
    Ok((output_commitments, output_columns))
}

/// Authenticated global negation needs no additional argument beyond the two input
/// relation proofs: the verifier already knows the exact right length. An empty right
/// retains the exact left commitment; a nonempty right yields canonical empty columns.
pub fn prove_global_not(
    left: &RelationWitness,
    right: &RelationWitness,
) -> io::Result<(Vec<ArkGT>, Vec<Vec<ArkFr>>, usize, usize)> {
    validate_witness(left, "left")?;
    validate_witness(right, "right")?;
    if right.n == 0 {
        return Ok((
            left.commitments.clone(),
            left.columns.clone(),
            left.log_domain,
            left.n,
        ));
    }
    let size = 1usize << left.log_domain;
    let columns = vec![vec![f0(); size]; left.columns.len()];
    let commitments = columns
        .iter()
        .map(|column| commit_column(column))
        .collect::<io::Result<_>>()?;
    Ok((commitments, columns, left.log_domain, 0))
}

pub fn verify_global_not(
    left: &RelationCommit,
    right: &RelationCommit,
    output_commitments: &[ArkGT],
    output_len: usize,
    output_log_domain: usize,
) -> io::Result<()> {
    validate_relation(left, "left")?;
    validate_relation(right, "right")?;
    if output_log_domain != left.log_domain || output_commitments.len() != left.commitments.len() {
        return Err(invalid("global Not output shape mismatch"));
    }
    if right.n == 0 {
        if output_len != left.n || output_commitments != left.commitments {
            return Err(invalid("empty global Not did not retain the exact outer relation"));
        }
        return Ok(());
    }
    if output_len != 0 {
        return Err(invalid("nonempty global Not must suppress every outer row"));
    }
    let zero = vec![f0(); 1usize << left.log_domain];
    let expected = commit_column(&zero)?;
    if output_commitments.iter().any(|commitment| *commitment != expected) {
        return Err(invalid("global Not output is not canonically empty"));
    }
    Ok(())
}

/// Prove a correlated `Not` for exactly one shared (scalar Store-ID) key column.
pub fn prove_left_anti_join(
    left: &RelationWitness,
    right: &RelationWitness,
    left_key: usize,
    right_key: usize,
    transcript: &mut Transcript254,
) -> io::Result<LeftAntiJoinResult> {
    prove_left_anti_join_with_keys(left, right, &[(left_key, right_key)], transcript)
}

/// Prove a correlated `Not` over one scalar or two injectively packed Store-ID keys.
pub fn prove_left_anti_join_with_keys(
    left: &RelationWitness,
    right: &RelationWitness,
    key_pairs: &[(usize, usize)],
    transcript: &mut Transcript254,
) -> io::Result<LeftAntiJoinResult> {
    validate_witness(left, "left")?;
    validate_witness(right, "right")?;
    validate_keys(left.columns.len(), right.columns.len(), key_pairs)?;
    let composite = key_pairs.len() == 2;
    let left_keys: Vec<usize> = key_pairs.iter().map(|pair| pair.0).collect();
    let right_keys: Vec<usize> = key_pairs.iter().map(|pair| pair.1).collect();
    let (left_key, right_key) = key_pairs[0];

    let left_key_owned;
    let right_key_owned;
    let (left_real_key, left_real_other): (&[ArkFr], Vec<&[ArkFr]>) = if composite {
        left_key_owned = composite_key(&left.columns, &left_keys, left.n);
        (
            &left_key_owned,
            left.columns.iter().map(|column| &column[..left.n]).collect(),
        )
    } else {
        split_key(&left.columns, left_key, left.n)
    };
    let (right_real_key, right_real_other): (&[ArkFr], Vec<&[ArkFr]>) = if composite {
        right_key_owned = composite_key(&right.columns, &right_keys, right.n);
        (
            &right_key_owned,
            right.columns.iter().map(|column| &column[..right.n]).collect(),
        )
    } else {
        split_key(&right.columns, right_key, right.n)
    };
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
    let left_join_key_owned;
    let right_join_key_owned;
    let (left_join_key, left_join_other): (&[ArkFr], Vec<&[ArkFr]>) = if composite {
        left_join_key_owned = composite_key(&left_columns, &left_keys, left.n);
        (
            &left_join_key_owned,
            left_columns.iter().map(|column| &column[..left.n]).collect(),
        )
    } else {
        split_key(&left_columns, left_key, left.n)
    };
    let (right_join_key, right_join_other): (&[ArkFr], Vec<&[ArkFr]>) = if composite {
        right_join_key_owned = composite_key(&right_columns, &right_keys, right.n);
        (
            &right_join_key_owned,
            right_columns.iter().map(|column| &column[..right.n]).collect(),
        )
    } else {
        split_key(&right_columns, right_key, right.n)
    };
    let join_commitments = binary_join_commitments(
        left_join_key,
        &left_join_other,
        right_join_key,
        &right_join_other,
        &witness,
    )?;
    let left_targets = if composite {
        if composite_commitment(&join_commitments.r_other, &left_keys) != join_commitments.r_key {
            return Err(invalid("anti-join composite left key commitment mismatch"));
        }
        join_commitments.r_other.clone()
    } else {
        input_commitments(&join_commitments, true, left_key, left.columns.len())
    };
    let right_targets = if composite {
        if composite_commitment(&join_commitments.s_other, &right_keys) != join_commitments.s_key {
            return Err(invalid("anti-join composite right key commitment mismatch"));
        }
        join_commitments.s_other.clone()
    } else {
        input_commitments(&join_commitments, false, right_key, right.columns.len())
    };

    let mut left_relocations = Vec::with_capacity(left.columns.len());
    let mut right_relocations = Vec::with_capacity(right.columns.len());
    // The final output is not known until zero-test compaction, so bind the immutable
    // input statement now; the output commitments are bound by compaction below.
    bind_statement(
        transcript,
        &left.view(),
        &right.view(),
        key_pairs,
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
    let (join, _, join_len) = if composite {
        prove_binary_join_with_key_bits(
            left_join_key,
            &left_join_other,
            right_join_key,
            &right_join_other,
            128,
            &join_commitments,
            transcript,
        )?
    } else {
        prove_binary_join(
            left_join_key,
            &left_join_other,
            right_join_key,
            &right_join_other,
            &join_commitments,
            transcript,
        )?
    };
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
    let (matched_commitments, matched_columns) = matched_projection(
        left.columns.len(),
        right.columns.len(),
        key_pairs,
        &join_commitments,
        Some(&witness),
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
        matched_commitments,
        matched_columns: matched_columns.expect("prover supplied a join witness"),
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
    verify_left_anti_join_with_keys(
        left,
        right,
        &[(left_key, right_key)],
        output_commitments,
        output_len,
        output_log_domain,
        proof,
        transcript,
    )
}

pub fn verify_left_anti_join_with_keys(
    left: &RelationCommit,
    right: &RelationCommit,
    key_pairs: &[(usize, usize)],
    output_commitments: &[ArkGT],
    output_len: usize,
    output_log_domain: usize,
    proof: &LeftAntiJoinProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    validate_relation(left, "left")?;
    validate_relation(right, "right")?;
    validate_keys(left.commitments.len(), right.commitments.len(), key_pairs)?;
    let composite = key_pairs.len() == 2;
    let left_keys: Vec<usize> = key_pairs.iter().map(|pair| pair.0).collect();
    let right_keys: Vec<usize> = key_pairs.iter().map(|pair| pair.1).collect();
    let (left_key, right_key) = key_pairs[0];
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
        key_pairs,
        output_log_domain,
    );
    let left_targets = if composite {
        if proof.join_commitments.r_other.len() != left.commitments.len()
            || composite_commitment(&proof.join_commitments.r_other, &left_keys)
                != proof.join_commitments.r_key
        {
            return Err(invalid("anti-join composite left key commitment mismatch"));
        }
        proof.join_commitments.r_other.clone()
    } else {
        input_commitments(&proof.join_commitments, true, left_key, left.commitments.len())
    };
    let right_targets = if composite {
        if proof.join_commitments.s_other.len() != right.commitments.len()
            || composite_commitment(&proof.join_commitments.s_other, &right_keys)
                != proof.join_commitments.s_key
        {
            return Err(invalid("anti-join composite right key commitment mismatch"));
        }
        proof.join_commitments.s_other.clone()
    } else {
        input_commitments(&proof.join_commitments, false, right_key, right.commitments.len())
    };
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
    if composite {
        verify_binary_join_with_key_bits(
            &proof.join_commitments,
            left.n,
            right.n,
            proof.join_len,
            128,
            &proof.join,
            transcript,
        )?;
    } else {
        verify_binary_join(
            &proof.join_commitments,
            left.n,
            right.n,
            proof.join_len,
            &proof.join,
            transcript,
        )?;
    }
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

/// Prove a correlated left outer join over one or two shared key columns. The matched
/// branch is the complete authenticated inner join. The unmatched branch is the exact
/// anti-join result with verifier-known zero columns appended for right-only variables.
pub fn prove_left_outer_join_with_keys(
    left: &RelationWitness,
    right: &RelationWitness,
    key_pairs: &[(usize, usize)],
    transcript: &mut Transcript254,
) -> io::Result<LeftOuterJoinResult> {
    let anti = prove_left_anti_join_with_keys(left, right, key_pairs, transcript)?;
    let size = 1usize << anti.log_domain;
    let right_only = right.columns.len() - key_pairs.len();
    let matched_columns: Vec<Vec<ArkFr>> = anti
        .matched_columns
        .iter()
        .map(|column| pad(column, size))
        .collect();
    let matched = RelationWitness {
        columns: matched_columns,
        commitments: anti.matched_commitments.clone(),
        log_domain: anti.log_domain,
        n: anti.proof.join_len,
    };
    let zero = vec![f0(); size];
    let zero_commitment = commit_column(&zero)?;
    let mut unmatched_columns = anti.columns.clone();
    unmatched_columns.extend((0..right_only).map(|_| zero.clone()));
    let mut unmatched_commitments = anti.commitments.clone();
    unmatched_commitments.extend((0..right_only).map(|_| zero_commitment));
    let unmatched = RelationWitness {
        columns: unmatched_columns,
        commitments: unmatched_commitments,
        log_domain: anti.log_domain,
        n: anti.n,
    };
    let union = prove_bag_union(&[matched, unmatched], transcript)?;
    Ok(LeftOuterJoinResult {
        proof: LeftOuterJoinProof {
            anti: anti.proof,
            unmatched_left_commitments: anti.commitments,
            unmatched_len: anti.n,
            union: union.proof,
        },
        commitments: union.commitments,
        columns: union.columns,
        log_domain: union.log_domain,
        n: union.n,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn verify_left_outer_join_with_keys(
    left: &RelationCommit,
    right: &RelationCommit,
    key_pairs: &[(usize, usize)],
    output_commitments: &[ArkGT],
    output_len: usize,
    output_log_domain: usize,
    proof: &LeftOuterJoinProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    let join_log_domain = proof.anti.join.num_vars();
    verify_left_anti_join_with_keys(
        left,
        right,
        key_pairs,
        &proof.unmatched_left_commitments,
        proof.unmatched_len,
        join_log_domain,
        &proof.anti,
        transcript,
    )?;
    let (matched_commitments, _) = matched_projection(
        left.commitments.len(),
        right.commitments.len(),
        key_pairs,
        &proof.anti.join_commitments,
        None,
    )?;
    let right_only = right.commitments.len() - key_pairs.len();
    let zero = vec![f0(); 1usize << join_log_domain];
    let zero_commitment = commit_column(&zero)?;
    let mut unmatched_commitments = proof.unmatched_left_commitments.clone();
    unmatched_commitments.extend((0..right_only).map(|_| zero_commitment));
    let branches = [
        RelationCommit {
            commitments: matched_commitments,
            log_domain: join_log_domain,
            n: proof.anti.join_len,
        },
        RelationCommit {
            commitments: unmatched_commitments,
            log_domain: join_log_domain,
            n: proof.unmatched_len,
        },
    ];
    verify_bag_union(
        &branches,
        output_commitments,
        output_len,
        output_log_domain,
        &proof.union,
        transcript,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fr(value: u64) -> ArkFr {
        ArkFr::from_u64(value)
    }

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

    #[test]
    fn left_outer_join_preserves_all_matches_and_null_extends_once() {
        let left = relation(&[&[1, 2, 3], &[10, 20, 30]]);
        let right = relation(&[&[1, 1, 2], &[100, 101, 200]]);
        let mut prover = Transcript254::new(b"left-outer-test");
        let result =
            prove_left_outer_join_with_keys(&left, &right, &[(0, 0)], &mut prover).unwrap();
        assert_eq!(result.n, 4);
        assert_eq!(&result.columns[0][..4], &[fr(1), fr(1), fr(2), fr(3)]);
        assert_eq!(&result.columns[1][..4], &[fr(10), fr(10), fr(20), fr(30)]);
        assert_eq!(
            &result.columns[2][..4],
            &[fr(100), fr(101), fr(200), f0()]
        );
        let mut verifier = Transcript254::new(b"left-outer-test");
        verify_left_outer_join_with_keys(
            &left.view(),
            &right.view(),
            &[(0, 0)],
            &result.commitments,
            result.n,
            result.log_domain,
            &result.proof,
            &mut verifier,
        )
        .unwrap();

        let mut forged = result.commitments.clone();
        forged[2] = commit_column(&vec![f0(); 1usize << result.log_domain]).unwrap();
        let mut verifier = Transcript254::new(b"left-outer-test");
        assert!(verify_left_outer_join_with_keys(
            &left.view(),
            &right.view(),
            &[(0, 0)],
            &forged,
            result.n,
            result.log_domain,
            &result.proof,
            &mut verifier,
        )
        .is_err());
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

    #[test]
    fn composite_keys_require_both_words_and_bind_pair_order() {
        let left = relation(&[
            &[1, 1, 2, 2],
            &[10, 11, 10, 10],
            &[100, 101, 200, 201],
        ]);
        // Right key columns are intentionally reversed: (b, a). The final row only
        // matches `a=1` partially and must not suppress the `a=1,b=11` left row.
        let right = relation(&[&[10, 10, 99], &[1, 2, 1], &[900, 901, 902]]);
        let keys = [(0usize, 1usize), (1usize, 0usize)];
        let mut pt = transcript();
        let result = prove_left_anti_join_with_keys(&left, &right, &keys, &mut pt).unwrap();
        assert_eq!(result.n, 1);
        assert_eq!(result.columns[0][0], ArkFr::from_u64(1));
        assert_eq!(result.columns[1][0], ArkFr::from_u64(11));
        assert_eq!(result.columns[2][0], ArkFr::from_u64(101));
        let mut vt = transcript();
        verify_left_anti_join_with_keys(
            &left.view(),
            &right.view(),
            &keys,
            &result.commitments,
            result.n,
            result.log_domain,
            &result.proof,
            &mut vt,
        )
        .unwrap();

        let swapped = [keys[1], keys[0]];
        let mut vt = transcript();
        assert!(verify_left_anti_join_with_keys(
            &left.view(),
            &right.view(),
            &swapped,
            &result.commitments,
            result.n,
            result.log_domain,
            &result.proof,
            &mut vt,
        )
        .is_err());
    }

    #[test]
    fn global_not_is_exactly_the_authenticated_right_nonempty_gate() {
        let left = relation(&[&[1, 2], &[10, 20]]);
        let empty = relation(&[&[], &[]]);
        let (commitments, _, log_domain, n) = prove_global_not(&left, &empty).unwrap();
        assert_eq!(n, 2);
        assert_eq!(commitments, left.commitments);
        verify_global_not(&left.view(), &empty.view(), &commitments, n, log_domain).unwrap();

        let nonempty = relation(&[&[9], &[90]]);
        let (empty_commitments, _, log_domain, n) =
            prove_global_not(&left, &nonempty).unwrap();
        assert_eq!(n, 0);
        verify_global_not(
            &left.view(),
            &nonempty.view(),
            &empty_commitments,
            n,
            log_domain,
        )
        .unwrap();
        assert!(verify_global_not(
            &left.view(),
            &nonempty.view(),
            &left.commitments,
            left.n,
            left.log_domain,
        )
        .is_err());
        assert!(verify_global_not(
            &left.view(),
            &empty.view(),
            &empty_commitments,
            0,
            left.log_domain,
        )
        .is_err());
    }
}
