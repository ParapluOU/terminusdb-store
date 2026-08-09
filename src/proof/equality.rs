//! Exact equality selection over an already-authenticated relation.
//!
//! WOQL `Equals` is unification in the executor.  This gadget deliberately covers
//! only the relationally safe subset where both operands are already public constants
//! or non-null columns of a committed input relation.  It proves the exact equality
//! indicator with the existing IsZero argument, then densely compacts every input
//! column with the existing LogUp compaction argument.

use std::io;

use ark_serialize::CanonicalSerialize;
use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::{Field, Group};
use dory_pcs::primitives::transcript::Transcript;

use super::argument::{commit_column, Transcript254};
use super::bgp::{RelationCommit, RelationWitness};
use super::filter::{
    prove_compaction_only, prove_filter_indicator, verify_compaction_only,
    verify_filter_indicator, CompactionOnlyProof, FilterProof,
};
use super::invalid;

const EQUALITY_DOMAIN: &[u8] = b"parture/proof-of-woql/relation-equality/v0";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RelationEqualityConstraint {
    ColumnConstant { column: usize, value: ArkFr },
    Columns { left: usize, right: usize },
    Constants { left: ArkFr, right: ArkFr },
}

#[derive(Clone, Debug)]
pub struct RelationEqualityProof {
    pub indicator: FilterProof,
    pub selection_commitment: ArkGT,
    pub inverse_commitment: ArkGT,
    pub compaction: CompactionOnlyProof,
}

pub struct RelationEqualityResult {
    pub proof: RelationEqualityProof,
    pub commitments: Vec<ArkGT>,
    pub columns: Vec<Vec<ArkFr>>,
    pub log_domain: usize,
    pub n: usize,
}

fn validate_relation(relation: &RelationCommit) -> io::Result<()> {
    if relation.commitments.is_empty() {
        return Err(invalid("equality relation has no columns"));
    }
    if relation.log_domain == 0 || relation.log_domain >= usize::BITS as usize {
        return Err(invalid("equality relation domain out of range"));
    }
    if relation.n > (1usize << relation.log_domain) {
        return Err(invalid("equality relation length exceeds its domain"));
    }
    Ok(())
}

fn validate_constraint(constraint: &RelationEqualityConstraint, arity: usize) -> io::Result<()> {
    match constraint {
        RelationEqualityConstraint::ColumnConstant { column, .. } if *column >= arity => {
            Err(invalid("equality column out of range"))
        }
        RelationEqualityConstraint::Columns { left, right }
            if *left >= arity || *right >= arity =>
        {
            Err(invalid("equality column out of range"))
        }
        _ => Ok(()),
    }
}

fn bind_statement(
    transcript: &mut Transcript254,
    relation: &RelationCommit,
    constraint: &RelationEqualityConstraint,
) {
    let scalar = |value: &ArkFr| {
        let mut bytes = Vec::new();
        value
            .serialize_compressed(&mut bytes)
            .expect("serializing a field element to memory cannot fail");
        bytes
    };
    transcript.append_bytes(b"pow/equality/domain", EQUALITY_DOMAIN);
    for value in [relation.log_domain, relation.n, relation.commitments.len()] {
        transcript.append_bytes(b"pow/equality/meta", &(value as u64).to_le_bytes());
    }
    for commitment in &relation.commitments {
        transcript.append_group(b"pow/equality/input", commitment);
    }
    match constraint {
        RelationEqualityConstraint::ColumnConstant { column, value } => {
            transcript.append_bytes(b"pow/equality/kind", &[0]);
            transcript.append_bytes(b"pow/equality/column", &(*column as u64).to_le_bytes());
            transcript.append_bytes(b"pow/equality/value", &scalar(value));
        }
        RelationEqualityConstraint::Columns { left, right } => {
            transcript.append_bytes(b"pow/equality/kind", &[1]);
            transcript.append_bytes(b"pow/equality/left", &(*left as u64).to_le_bytes());
            transcript.append_bytes(b"pow/equality/right", &(*right as u64).to_le_bytes());
        }
        RelationEqualityConstraint::Constants { left, right } => {
            transcript.append_bytes(b"pow/equality/kind", &[2]);
            transcript.append_bytes(b"pow/equality/left-value", &scalar(left));
            transcript.append_bytes(b"pow/equality/right-value", &scalar(right));
        }
    }
}

fn predicate_witness(
    relation: &RelationWitness,
    constraint: &RelationEqualityConstraint,
) -> io::Result<(Vec<ArkFr>, ArkGT, ArkFr)> {
    let size = 1usize << relation.log_domain;
    match constraint {
        RelationEqualityConstraint::ColumnConstant { column, value } => Ok((
            relation.columns[*column].clone(),
            relation.commitments[*column],
            *value,
        )),
        RelationEqualityConstraint::Columns { left, right } => {
            let column = relation.columns[*left]
                .iter()
                .zip(&relation.columns[*right])
                .map(|(left, right)| *left - *right)
                .collect::<Vec<_>>();
            let commitment = relation.commitments[*left]
                .add(&relation.commitments[*right].scale(&(-ArkFr::one())));
            Ok((column, commitment, ArkFr::zero()))
        }
        RelationEqualityConstraint::Constants { left, right } => {
            let mut column = vec![*left; relation.n];
            column.resize(size, ArkFr::zero());
            Ok((column.clone(), commit_column(&column)?, *right))
        }
    }
}

fn predicate_commitment(
    relation: &RelationCommit,
    constraint: &RelationEqualityConstraint,
) -> io::Result<(ArkGT, ArkFr)> {
    let size = 1usize << relation.log_domain;
    match constraint {
        RelationEqualityConstraint::ColumnConstant { column, value } => {
            Ok((relation.commitments[*column], *value))
        }
        RelationEqualityConstraint::Columns { left, right } => Ok((
            relation.commitments[*left]
                .add(&relation.commitments[*right].scale(&(-ArkFr::one()))),
            ArkFr::zero(),
        )),
        RelationEqualityConstraint::Constants { left, right } => {
            let mut column = vec![*left; relation.n];
            column.resize(size, ArkFr::zero());
            Ok((commit_column(&column)?, *right))
        }
    }
}

pub fn prove_relation_equality(
    relation: &RelationWitness,
    constraint: &RelationEqualityConstraint,
    transcript: &mut Transcript254,
) -> io::Result<RelationEqualityResult> {
    validate_relation(&relation.view())?;
    validate_constraint(constraint, relation.commitments.len())?;
    bind_statement(transcript, &relation.view(), constraint);
    let (predicate, predicate_commitment, value) = predicate_witness(relation, constraint)?;
    let (indicator, selection_commitment, inverse_commitment, _) = prove_filter_indicator(
        &predicate,
        value,
        relation.n,
        &predicate_commitment,
        transcript,
    )?;
    let selection = predicate
        .iter()
        .enumerate()
        .map(|(index, candidate)| {
            if index < relation.n && *candidate == value {
                ArkFr::one()
            } else {
                ArkFr::zero()
            }
        })
        .collect::<Vec<_>>();
    let (compaction, commitments, columns, n) = prove_compaction_only(
        &relation.columns,
        &relation.commitments,
        &selection,
        &selection_commitment,
        transcript,
    )?;
    Ok(RelationEqualityResult {
        proof: RelationEqualityProof {
            indicator,
            selection_commitment,
            inverse_commitment,
            compaction,
        },
        commitments,
        columns,
        log_domain: relation.log_domain,
        n,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn verify_relation_equality(
    relation: &RelationCommit,
    constraint: &RelationEqualityConstraint,
    output_commitments: &[ArkGT],
    output_len: usize,
    output_log_domain: usize,
    proof: &RelationEqualityProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    validate_relation(relation)?;
    validate_constraint(constraint, relation.commitments.len())?;
    if output_log_domain != relation.log_domain {
        return Err(invalid("equality output domain mismatch"));
    }
    bind_statement(transcript, relation, constraint);
    let (predicate_commitment, value) = predicate_commitment(relation, constraint)?;
    verify_filter_indicator(
        &predicate_commitment,
        value,
        relation.n,
        &proof.selection_commitment,
        &proof.inverse_commitment,
        &proof.indicator,
        transcript,
    )?;
    verify_compaction_only(
        &relation.commitments,
        &proof.selection_commitment,
        output_commitments,
        output_len,
        &proof.compaction,
        transcript,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fr(value: u64) -> ArkFr {
        ArkFr::from_u64(value)
    }

    fn transcript() -> Transcript254 {
        Transcript254::new(b"relation-equality-test")
    }

    #[test]
    fn constant_and_column_equality_are_exact_filters() {
        let relation = RelationWitness::commit(
            vec![vec![fr(1), fr(2), fr(3)], vec![fr(1), fr(9), fr(3)]],
            3,
        )
        .unwrap();
        for (constraint, expected) in [
            (
                RelationEqualityConstraint::ColumnConstant {
                    column: 0,
                    value: fr(2),
                },
                1,
            ),
            (
                RelationEqualityConstraint::Columns { left: 0, right: 1 },
                2,
            ),
        ] {
            let result = prove_relation_equality(&relation, &constraint, &mut transcript()).unwrap();
            verify_relation_equality(
                &relation.view(),
                &constraint,
                &result.commitments,
                result.n,
                result.log_domain,
                &result.proof,
                &mut transcript(),
            )
            .unwrap();
            assert_eq!(result.n, expected);
        }
    }

    #[test]
    fn false_constant_gate_and_statement_substitution_reject() {
        let relation = RelationWitness::commit(vec![vec![fr(1), fr(2)]], 2).unwrap();
        let constraint = RelationEqualityConstraint::Constants {
            left: fr(7),
            right: fr(8),
        };
        let result = prove_relation_equality(&relation, &constraint, &mut transcript()).unwrap();
        assert_eq!(result.n, 0);
        verify_relation_equality(
            &relation.view(),
            &constraint,
            &result.commitments,
            result.n,
            result.log_domain,
            &result.proof,
            &mut transcript(),
        )
        .unwrap();
        assert!(verify_relation_equality(
            &relation.view(),
            &RelationEqualityConstraint::Constants {
                left: fr(7),
                right: fr(7),
            },
            &result.commitments,
            result.n,
            result.log_domain,
            &result.proof,
            &mut transcript(),
        )
        .is_err());
    }
}
