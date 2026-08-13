//! The flat left-outer-join / `OPTIONAL` envelope.
//!
//! [`LeftOuterJoinEnvelope`] proves a left-outer join: every left row is kept, extended with
//! the matching right columns where a match exists and with NULLs where none does. Both sides
//! are independently verified [`LayerBgpEnvelope`]s against the same root; `outer_proof`
//! authenticates the joined output. The output column descriptors (which columns are nullable)
//! are derived from the two child plans by `outer_descriptors`, and the verifier re-derives the
//! delivered canonical values — including the NULLs — from the root, so the padding cannot be
//! forged.
#![allow(unused_imports)]

use super::layer_bgp::LayerBgpEnvelope;
use super::wire::{
    invalid, wire_struct, Reader, Wire, CHECKSUM_BYTES, MAX_QUERY_ENVELOPE_BYTES, OUTER_MAGIC,
    OUTER_VERSION,
};
use std::convert::{TryFrom, TryInto};
use std::io::{self, Cursor};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_ff::PrimeField;
use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::transcript::Transcript;
use sha2::{Digest, Sha256};

use crate::proof::anti::{
    prove_global_left_outer_join_with_context, prove_global_not_with_context,
    prove_inner_join_with_keys_and_context, prove_left_anti_join_with_keys_and_context, prove_left_outer_join_with_keys_and_context,
    verify_global_left_outer_join, verify_global_not, verify_left_anti_join_with_keys,
    verify_inner_join_with_keys, verify_left_outer_join_with_keys, InnerJoinProof, InnerJoinResult,
    LeftAntiJoinProof, LeftOuterJoinProof,
};
use crate::proof::argument::{commit_column, ColumnOpening, StoreProverContext, SumcheckProof, Transcript254};
use crate::proof::bgp::{BgpChainProof, BgpStageProof, JoinStep, RelationCommit, RelationWitness};
use crate::proof::commitment::{LayerCommitment, VerifierLayer};
use crate::proof::count::CountProof;
use crate::proof::dict::DictProof;
use crate::proof::dicthash::{
    prove_dict_hash_lookup, verify_dict_hash_lookup, DictHashLookupProof, HashLookupInnerProof,
};
use crate::proof::semijoin::{
    prove_hash_semijoin, verify_hash_semijoin, HashSemiJoinProof, SelMaskProof,
};
use crate::proof::distinct::{
    prove_relation_distinct_with_context, verify_relation_distinct,
    RelationDistinctEntryProof, RelationDistinctProof, RelationDistinctResult,
};
use crate::proof::disequality::{
    prove_tuple_disequality_with_context, verify_tuple_disequality, DisequalityNode,
    DisequalityPair, TupleDisequalityProof, TupleDisequalityStageProof,
};
use crate::proof::equality::{
    prove_relation_equality_with_context, verify_relation_equality, RelationEqualityConstraint,
    RelationEqualityProof,
};
use crate::proof::filter::{
    AtomFilterProof, BooleanProductProof, CompactionOnlyProof, FilterProof, FullFilterProof,
    SelectionOnlyProof,
};
use crate::proof::group::{GroupExtremaEntryProof, GroupExtremaKind, GroupExtremaProof};
use crate::proof::join2::{BinaryJoinCommitments, BinaryJoinProof, BinaryPairsProof};
use crate::proof::membership::MembershipProof;
use crate::proof::multiorder::{prove_exact_multi_order_with_context, verify_exact_multi_order, ExactMultiOrderProof};
use crate::proof::lexsort::LexicographicSortedProof;
use crate::proof::order::{
    prove_exact_scalar_order_with_context, verify_exact_scalar_order, ExactScalarOrderProof,
    OrderDirection,
};
use crate::proof::permutation::RelationPermutationProof;
use crate::proof::query::{
    has_ground_result_unit, prove_layer_bgp_with_result_and_context,
    resolve_typed_result_objects, result_id_namespaces,
    resolve_result_values, verify_layer_bgp, verify_layer_bgp_result_log_domain,
    verify_result_columns, generated_decimal_i64_field, generated_decimal_u64_value,
    AtomTerm, CanonicalResultValue, ColumnSource,
    DictionaryAbsence,
    DictionaryAbsencePosition, LayerBgpAtom,
    LayerBgpAtomProof, LayerBgpPlan, LayerBgpProof, ObjectKindFilter, ObjectRangeBound,
    ObjectRangeSpec, RangeAtomSubproof, ResultColumnDescriptor, ResultIdNamespace,
    SecondRangeSubproof,
};
use crate::proof::range::{
    DictLookupProof, ObjectRangeCombineProof, ObjectRangeProof, RangeOp, RangeSelectProof,
};
use crate::proof::union::{prove_bag_union_with_context, BagUnionResult};
use crate::proof::relocate::{prove_column_relocation_with_context, verify_column_relocation, RelocationProof};
use crate::proof::sort::SortedProof;
use crate::proof::union::{verify_bag_union, BagUnionProof};
use crate::proof::{CanonicalDatatype, CanonicalObject, ProofHash};
#[derive(Clone, Debug)]
pub struct LeftOuterJoinEnvelope {
    pub layer_root: ProofHash,
    pub query_digest: ProofHash,
    pub left: LayerBgpEnvelope,
    pub right: LayerBgpEnvelope,
    pub key_pairs: Vec<(usize, usize)>,
    pub outer_proof: LeftOuterJoinProof,
    pub result_commitments: Vec<ArkGT>,
    pub result_columns: Vec<Vec<ArkFr>>,
    pub result_log_domain: usize,
    pub result_len: usize,
    pub result_descriptors: Vec<ResultColumnDescriptor>,
    pub result_values: Vec<Vec<CanonicalResultValue>>,
}

wire_struct!(LeftOuterJoinEnvelope {
    layer_root,
    query_digest,
    left,
    right,
    key_pairs,
    outer_proof,
    result_commitments,
    result_columns,
    result_log_domain,
    result_len,
    result_descriptors,
    result_values
});

pub(super) fn outer_descriptors(
    left_plan: &LayerBgpPlan,
    right_plan: &LayerBgpPlan,
    key_pairs: &[(usize, usize)],
) -> io::Result<Vec<ResultColumnDescriptor>> {
    let left = result_id_namespaces(left_plan)?;
    let right = result_id_namespaces(right_plan)?;
    if key_pairs.len() > 2 {
        return Err(invalid("left-outer envelope supports at most two shared keys"));
    }
    let mut shared_right = Vec::with_capacity(key_pairs.len());
    for (left_index, right_index) in key_pairs {
        if *left_index >= left.len() || *right_index >= right.len() {
            return Err(invalid("left-outer envelope key index out of range"));
        }
        if left[*left_index] != right[*right_index] {
            return Err(invalid("left-outer envelope key namespace mismatch"));
        }
        if shared_right.contains(right_index) {
            return Err(invalid("left-outer envelope repeats a right key"));
        }
        shared_right.push(*right_index);
    }
    let mut descriptors: Vec<_> = left
        .into_iter()
        .map(|namespace| ResultColumnDescriptor {
            namespace,
            nullable: false,
        })
        .collect();
    descriptors.extend(right.into_iter().enumerate().filter_map(|(index, namespace)| {
        (!shared_right.contains(&index)).then_some(ResultColumnDescriptor {
            namespace,
            nullable: true,
        })
    }));
    Ok(descriptors)
}

impl LeftOuterJoinEnvelope {
    #[allow(clippy::too_many_arguments)]
    pub fn new_verified(
        layer: &LayerCommitment,
        expected_root: ProofHash,
        left_plan: &LayerBgpPlan,
        right_plan: &LayerBgpPlan,
        query_digest: ProofHash,
        left: LayerBgpEnvelope,
        right: LayerBgpEnvelope,
        key_pairs: Vec<(usize, usize)>,
        outer_proof: LeftOuterJoinProof,
        result_commitments: Vec<ArkGT>,
        result_columns: Vec<Vec<ArkFr>>,
        result_log_domain: usize,
        result_len: usize,
    ) -> io::Result<Self> {
        left.verify(layer, expected_root, left_plan, query_digest)?;
        right.verify(layer, expected_root, right_plan, query_digest)?;
        let mut transcript = crate::proof::argument::Transcript254::new(b"parture-left-outer-envelope");
        transcript.append_bytes(b"pow/outer-envelope/root", &expected_root);
        transcript.append_bytes(b"pow/outer-envelope/query", &query_digest);
        let left_commit = RelationCommit {
                commitments: left.result_commitments.clone(),
                log_domain: left.result_log_domain,
                n: left.result_len,
            };
        let right_commit = RelationCommit {
                commitments: right.result_commitments.clone(),
                log_domain: right.result_log_domain,
                n: right.result_len,
            };
        if key_pairs.is_empty() {
            verify_global_left_outer_join(
                &left_commit, &right_commit, &result_commitments, result_len,
                result_log_domain, &outer_proof, &mut transcript,
            )?;
        } else {
            verify_left_outer_join_with_keys(
                &left_commit, &right_commit, &key_pairs, &result_commitments, result_len,
                result_log_domain, &outer_proof, &mut transcript,
            )?;
        }
        verify_result_columns(
            &result_commitments,
            &result_columns,
            result_len,
            result_log_domain,
        )?;
        let result_descriptors = outer_descriptors(left_plan, right_plan, &key_pairs)?;
        let result_values = resolve_result_values(
            layer,
            expected_root,
            &result_columns,
            &result_descriptors,
        )?;
        Ok(Self {
            layer_root: expected_root,
            query_digest,
            left,
            right,
            key_pairs,
            outer_proof,
            result_commitments,
            result_columns,
            result_log_domain,
            result_len,
            result_descriptors,
            result_values,
        })
    }

    pub fn verify(
        &self,
        layer: &impl VerifierLayer,
        expected_root: ProofHash,
        left_plan: &LayerBgpPlan,
        right_plan: &LayerBgpPlan,
        query_digest: ProofHash,
        key_pairs: &[(usize, usize)],
    ) -> io::Result<()> {
        if self.layer_root != expected_root || self.query_digest != query_digest {
            return Err(invalid("left-outer envelope identity mismatch"));
        }
        if self.key_pairs != key_pairs {
            return Err(invalid("left-outer envelope key mismatch"));
        }
        self.left.verify(layer, expected_root, left_plan, query_digest)?;
        self.right.verify(layer, expected_root, right_plan, query_digest)?;
        let expected_descriptors = outer_descriptors(left_plan, right_plan, key_pairs)?;
        if self.result_descriptors != expected_descriptors {
            return Err(invalid("left-outer result descriptor mismatch"));
        }
        let mut transcript = crate::proof::argument::Transcript254::new(b"parture-left-outer-envelope");
        transcript.append_bytes(b"pow/outer-envelope/root", &expected_root);
        transcript.append_bytes(b"pow/outer-envelope/query", &query_digest);
        let left_commit = RelationCommit {
                commitments: self.left.result_commitments.clone(),
                log_domain: self.left.result_log_domain,
                n: self.left.result_len,
            };
        let right_commit = RelationCommit {
                commitments: self.right.result_commitments.clone(),
                log_domain: self.right.result_log_domain,
                n: self.right.result_len,
            };
        if key_pairs.is_empty() {
            verify_global_left_outer_join(
                &left_commit, &right_commit, &self.result_commitments, self.result_len,
                self.result_log_domain, &self.outer_proof, &mut transcript,
            )?;
        } else {
            verify_left_outer_join_with_keys(
                &left_commit, &right_commit, key_pairs, &self.result_commitments,
                self.result_len, self.result_log_domain, &self.outer_proof, &mut transcript,
            )?;
        }
        verify_result_columns(
            &self.result_commitments,
            &self.result_columns,
            self.result_len,
            self.result_log_domain,
        )?;
        let values = resolve_result_values(
            layer,
            expected_root,
            &self.result_columns,
            &expected_descriptors,
        )?;
        if values != self.result_values {
            return Err(invalid("left-outer canonical result mismatch"));
        }
        Ok(())
    }

    pub fn encode(&self) -> io::Result<Vec<u8>> {
        let mut out = Vec::new();
        out.extend_from_slice(OUTER_MAGIC);
        out.extend_from_slice(&OUTER_VERSION.to_le_bytes());
        self.put(&mut out)?;
        if out.len().checked_add(CHECKSUM_BYTES).map_or(true, |len| len > MAX_QUERY_ENVELOPE_BYTES) {
            return Err(invalid("left-outer envelope too large"));
        }
        let checksum = Sha256::digest(&out);
        out.extend_from_slice(&checksum);
        Ok(out)
    }

    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < OUTER_MAGIC.len() + 4 + CHECKSUM_BYTES
            || bytes.len() > MAX_QUERY_ENVELOPE_BYTES
        {
            return Err(invalid("invalid left-outer envelope size"));
        }
        let body_len = bytes.len() - CHECKSUM_BYTES;
        let (body, checksum) = bytes.split_at(body_len);
        if Sha256::digest(body).as_slice() != checksum {
            return Err(invalid("left-outer envelope checksum mismatch"));
        }
        let mut reader = Reader { bytes: body, pos: 0, decoded_items: 0 };
        if reader.take(OUTER_MAGIC.len())? != OUTER_MAGIC {
            return Err(invalid("unknown left-outer envelope magic"));
        }
        let version = u32::from_le_bytes(reader.take(4)?.try_into().unwrap());
        if version != OUTER_VERSION {
            return Err(invalid("unsupported left-outer envelope version"));
        }
        let envelope = Self::get(&mut reader)?;
        reader.finish()?;
        Ok(envelope)
    }
}
