//! Exact unordered distinct projection for one or two nonnullable relation columns.
//!
//! Every public output tuple defines an exact equality-compacted class of the
//! authenticated input bag. Output tuples must be unique and every class nonempty;
//! the sum of class cardinalities must equal the input cardinality. Because distinct
//! tuples define disjoint equality classes, these checks prove no input class was
//! omitted and no output tuple was duplicated or invented. This statement deliberately
//! does not authenticate SWI-Prolog's first-occurrence output order.
//!
//! For a developer: this is the set-distinct / dedup [gadget](crate::proof::primer). It adds no
//! new cryptography — each output tuple's equality class is carved by the audited
//! [equality gadget](super::equality) (an IsZero indicator + LogUp compaction), and distinctness
//! is argued combinatorially: disjoint nonempty classes whose sizes sum to the input row count
//! exactly cover the input. See the primer for [gadgets](crate::proof::primer) (§6) and the
//! [LogUp](crate::proof::primer) compaction it leans on (§7).

use std::collections::BTreeSet;
use std::io;

use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::transcript::Transcript;

use super::argument::{StoreProverContext, Transcript254};
use super::bgp::{RelationCommit, RelationWitness};
use super::equality::{
    prove_relation_equality_with_context, verify_relation_equality, RelationEqualityConstraint,
    RelationEqualityProof,
};
use super::invalid;

const DOMAIN: &[u8] = b"parture/proof-of-woql/relation-distinct/v0";

/// Per output tuple: the chain of equality-filter proofs (one per key column) that carve
/// out that tuple's equality class of the input, with each stage's commitments and length.
#[derive(Clone, Debug)]
pub struct RelationDistinctEntryProof {
    /// One equality-filter proof per key column: each stage restricts the running relation to
    /// the rows whose key column equals this tuple's value, so the chain carves the tuple's
    /// exact equality class of the input.
    pub filters: Vec<RelationEqualityProof>,
    /// Per stage, the output column commitments the corresponding `filters` entry produced.
    pub commitments: Vec<Vec<ArkGT>>,
    /// Per stage, the surviving (real) row count — the shrinking class cardinality.
    pub lengths: Vec<usize>,
}

/// A distinct-projection proof: one class-carving entry per public output tuple.
#[derive(Clone, Debug)]
pub struct RelationDistinctProof {
    pub entries: Vec<RelationDistinctEntryProof>,
}

/// Prover result: the proof plus the distinct output relation (its key columns, their
/// commitments, the output `log_domain`, and the distinct-tuple count `n`).
pub struct RelationDistinctResult {
    /// The distinct-projection proof.
    pub proof: RelationDistinctProof,
    /// The distinct output key columns, zero-padded to `2^log_domain`.
    pub columns: Vec<Vec<ArkFr>>,
    /// Dory commitments to `columns`.
    pub commitments: Vec<ArkGT>,
    /// The output columns' power-of-two log-domain.
    pub log_domain: usize,
    /// The number of distinct output tuples (real rows).
    pub n: usize,
}

fn bind(
    transcript: &mut Transcript254,
    input: &RelationCommit,
    key_columns: &[usize],
    rows: &[Vec<u64>],
) {
    transcript.append_bytes(b"pow/distinct/domain", DOMAIN);
    for value in [input.log_domain, input.n, input.commitments.len()] {
        transcript.append_bytes(b"pow/distinct/meta", &(value as u64).to_le_bytes());
    }
    for column in key_columns {
        transcript.append_bytes(b"pow/distinct/key-column", &(*column as u64).to_le_bytes());
    }
    for commitment in &input.commitments {
        transcript.append_group(b"pow/distinct/input", commitment);
    }
    transcript.append_bytes(b"pow/distinct/rows", &(rows.len() as u64).to_le_bytes());
    for row in rows {
        for value in row {
            transcript.append_bytes(b"pow/distinct/value", &value.to_le_bytes());
        }
    }
}

fn validate_statement(
    input: &RelationCommit,
    key_columns: &[usize],
    rows: &[Vec<u64>],
) -> io::Result<()> {
    if input.log_domain == 0
        || input.log_domain >= usize::BITS as usize
        || input.n > (1usize << input.log_domain)
        || input.commitments.is_empty()
    {
        return Err(invalid("distinct input relation shape is invalid"));
    }
    if !(1..=2).contains(&key_columns.len()) {
        return Err(invalid("distinct requires one or two key columns"));
    }
    let mut seen_columns = BTreeSet::new();
    for column in key_columns {
        if *column >= input.commitments.len() || !seen_columns.insert(*column) {
            return Err(invalid("distinct key column is out of range or duplicated"));
        }
    }
    if rows.len() > input.n {
        return Err(invalid("distinct result has more rows than its input"));
    }
    let mut seen_rows = BTreeSet::new();
    for row in rows {
        if row.len() != key_columns.len() || row.iter().any(|value| *value == 0) {
            return Err(invalid("distinct result tuple has invalid arity or Store id"));
        }
        if !seen_rows.insert(row.clone()) {
            return Err(invalid("distinct result contains a duplicate tuple"));
        }
    }
    Ok(())
}

fn padded(values: &[ArkFr], log_domain: usize) -> Vec<ArkFr> {
    let mut result = values.to_vec();
    result.resize(1usize << log_domain, ArkFr::zero());
    result
}

/// Prove `result_rows` is EXACTLY the set of distinct tuples of `input` over `key_columns`
/// (1 or 2 columns): each output tuple is unique, names a nonempty equality class, and the
/// class sizes sum to the input row count — so no class is missing and no tuple invented.
/// Unordered (first-occurrence order is not authenticated).
pub fn prove_relation_distinct_with_context(
    context: &StoreProverContext<'_>,
    input: &RelationWitness,
    key_columns: &[usize],
    result_rows: &[Vec<u64>],
    transcript: &mut Transcript254,
) -> io::Result<RelationDistinctResult> {
    validate_statement(&input.view(), key_columns, result_rows)?;
    bind(transcript, &input.view(), key_columns, result_rows);
    let mut entries = Vec::with_capacity(result_rows.len());
    let mut total = 0usize;
    for row in result_rows {
        let mut current = RelationWitness {
            columns: input.columns.clone(),
            commitments: input.commitments.clone(),
            log_domain: input.log_domain,
            n: input.n,
        };
        let mut filters = Vec::with_capacity(key_columns.len());
        let mut commitments = Vec::with_capacity(key_columns.len());
        let mut lengths = Vec::with_capacity(key_columns.len());
        for (&column, &value) in key_columns.iter().zip(row) {
            let constraint = RelationEqualityConstraint::ColumnConstant {
                column,
                value: ArkFr::from_u64(value),
            };
            let filtered = prove_relation_equality_with_context(
                context,
                &current,
                &constraint,
                transcript,
            )?;
            filters.push(filtered.proof);
            commitments.push(filtered.commitments.clone());
            lengths.push(filtered.n);
            current = RelationWitness {
                columns: filtered.columns,
                commitments: filtered.commitments,
                log_domain: filtered.log_domain,
                n: filtered.n,
            };
        }
        if current.n == 0 {
            return Err(invalid("distinct output tuple names an empty equality class"));
        }
        total = total
            .checked_add(current.n)
            .ok_or_else(|| invalid("distinct class cardinality overflow"))?;
        entries.push(RelationDistinctEntryProof {
            filters,
            commitments,
            lengths,
        });
    }
    if total != input.n {
        return Err(invalid("distinct equality classes do not exactly cover the input"));
    }
    let log_domain = result_rows
        .len()
        .max(1)
        .next_power_of_two()
        .max(2)
        .trailing_zeros() as usize;
    let columns = (0..key_columns.len())
        .map(|column| {
            padded(
                &result_rows
                    .iter()
                    .map(|row| ArkFr::from_u64(row[column]))
                    .collect::<Vec<_>>(),
                log_domain,
            )
        })
        .collect::<Vec<_>>();
    let commitments = columns
        .iter()
        .map(|column| context.commit_column(column))
        .collect::<io::Result<Vec<_>>>()?;
    Ok(RelationDistinctResult {
        proof: RelationDistinctProof { entries },
        columns,
        commitments,
        log_domain,
        n: result_rows.len(),
    })
}

/// Verify a distinct-projection proof: the public `result_rows` are exactly the distinct
/// `key_columns` tuples of `input`, with disjoint classes covering every input row.
pub fn verify_relation_distinct(
    input: &RelationCommit,
    key_columns: &[usize],
    result_rows: &[Vec<u64>],
    proof: &RelationDistinctProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    validate_statement(input, key_columns, result_rows)?;
    if proof.entries.len() != result_rows.len() {
        return Err(invalid("distinct proof/result row count mismatch"));
    }
    bind(transcript, input, key_columns, result_rows);
    let mut total = 0usize;
    for (row, entry) in result_rows.iter().zip(&proof.entries) {
        if entry.filters.len() != key_columns.len()
            || entry.commitments.len() != key_columns.len()
            || entry.lengths.len() != key_columns.len()
        {
            return Err(invalid("distinct entry proof arity mismatch"));
        }
        let mut current = input.clone();
        for (((&column, &value), filter), (commitments, &length)) in key_columns
            .iter()
            .zip(row)
            .zip(&entry.filters)
            .zip(entry.commitments.iter().zip(&entry.lengths))
        {
            let constraint = RelationEqualityConstraint::ColumnConstant {
                column,
                value: ArkFr::from_u64(value),
            };
            verify_relation_equality(
                &current,
                &constraint,
                commitments,
                length,
                input.log_domain,
                filter,
                transcript,
            )?;
            current = RelationCommit {
                commitments: commitments.clone(),
                log_domain: input.log_domain,
                n: length,
            };
        }
        if current.n == 0 {
            return Err(invalid("distinct output tuple has an empty equality class"));
        }
        total = total
            .checked_add(current.n)
            .ok_or_else(|| invalid("distinct class cardinality overflow"))?;
    }
    if total != input.n {
        return Err(invalid("distinct equality classes do not exactly cover the input"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn witness() -> RelationWitness {
        RelationWitness::commit(
            vec![
                [2, 1, 2, 3, 1]
                    .iter()
                    .map(|value| ArkFr::from_u64(*value))
                    .collect(),
                [7, 8, 7, 10, 8]
                    .iter()
                    .map(|value| ArkFr::from_u64(*value))
                    .collect(),
            ],
            5,
        )
        .unwrap()
    }

    #[test]
    fn duplicate_heavy_projection_proves_exact_unordered_classes() {
        let input = witness();
        let context = StoreProverContext::cpu();
        for (keys, rows) in [
            (vec![0], vec![vec![2], vec![1], vec![3]]),
            (
                vec![0, 1],
                vec![vec![2, 7], vec![1, 8], vec![3, 10]],
            ),
        ] {
            let mut prover = Transcript254::new(b"distinct-test");
            let result = prove_relation_distinct_with_context(
                &context,
                &input,
                &keys,
                &rows,
                &mut prover,
            )
            .unwrap();
            assert_eq!(result.n, rows.len());
            let mut verifier = Transcript254::new(b"distinct-test");
            verify_relation_distinct(
                &input.view(),
                &keys,
                &rows,
                &result.proof,
                &mut verifier,
            )
            .unwrap();
        }
    }

    #[test]
    fn omissions_extras_duplicates_and_bad_columns_reject() {
        let input = witness();
        let context = StoreProverContext::cpu();
        for rows in [
            vec![vec![1], vec![2]],
            vec![vec![1], vec![2], vec![3], vec![4]],
            vec![vec![1], vec![2], vec![2], vec![3]],
        ] {
            let mut transcript = Transcript254::new(b"distinct-bad");
            assert!(prove_relation_distinct_with_context(
                &context,
                &input,
                &[0],
                &rows,
                &mut transcript,
            )
            .is_err());
        }
        assert!(validate_statement(&input.view(), &[], &[]).is_err());
        assert!(validate_statement(&input.view(), &[0, 0], &[]).is_err());
        assert!(validate_statement(&input.view(), &[2], &[]).is_err());
    }

    #[test]
    fn empty_input_has_one_canonical_empty_distinct_result() {
        let input = RelationWitness::commit(vec![Vec::new()], 0).unwrap();
        let context = StoreProverContext::cpu();
        let mut prover = Transcript254::new(b"distinct-empty");
        let result = prove_relation_distinct_with_context(
            &context,
            &input,
            &[0],
            &[],
            &mut prover,
        )
        .unwrap();
        assert_eq!(result.n, 0);
        let mut verifier = Transcript254::new(b"distinct-empty");
        verify_relation_distinct(&input.view(), &[0], &[], &result.proof, &mut verifier).unwrap();
    }
}
