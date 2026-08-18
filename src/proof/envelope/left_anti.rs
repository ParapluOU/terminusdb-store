//! The flat left-anti-join / correlated-`NOT` envelope.
//!
//! [`LeftAntiJoinEnvelope`] proves "keep the left rows that have NO match on the right". Both
//! sides are independently verified [`LayerBgpEnvelope`]s against the same root. With no key
//! pairs it is a global `NOT` (keep the left rows only when the right relation is empty); with
//! key pairs it is a correlated anti-join (drop each left row whose key tuple appears on the
//! right). `anti_proof` authenticates the exact surviving outer rows; its presence-or-absence
//! must match the keyed-or-global variant, or the envelope is rejected.
#![allow(unused_imports)]

use super::layer_bgp::LayerBgpEnvelope;
use super::wire::{
    invalid, wire_struct, Reader, Wire, ANTI_MAGIC, ANTI_VERSION, CHECKSUM_BYTES,
    MAX_QUERY_ENVELOPE_BYTES,
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
/// Portable correlated-Not envelope. Both BGP inputs remain independently verifiable
/// against the trusted layer root; the anti proof authenticates the exact surviving
/// outer rows.
#[derive(Clone, Debug)]
pub struct LeftAntiJoinEnvelope {
    pub layer_root: ProofHash,
    pub query_digest: ProofHash,
    pub left: LayerBgpEnvelope,
    pub right: LayerBgpEnvelope,
    pub key_pairs: Vec<(usize, usize)>,
    pub anti_proof: Option<LeftAntiJoinProof>,
    pub result_commitments: Vec<ArkGT>,
    pub result_columns: Vec<Vec<ArkFr>>,
    pub result_log_domain: usize,
    pub result_len: usize,
    pub ground_result: bool,
    pub result_objects: Vec<Vec<CanonicalObject>>,
}

wire_struct!(LeftAntiJoinEnvelope {
    layer_root,
    query_digest,
    left,
    right,
    key_pairs,
    anti_proof,
    result_commitments,
    result_columns,
    result_log_domain,
    result_len,
    ground_result,
    result_objects
});

impl LeftAntiJoinEnvelope {
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
        anti_proof: Option<LeftAntiJoinProof>,
        result_commitments: Vec<ArkGT>,
        result_columns: Vec<Vec<ArkFr>>,
        result_log_domain: usize,
        result_len: usize,
        ground_result: bool,
    ) -> io::Result<Self> {
        left.verify(layer, expected_root, left_plan, query_digest)?;
        right.verify(layer, expected_root, right_plan, query_digest)?;
        let mut transcript = crate::proof::argument::Transcript254::new(b"parture-left-anti-envelope");
        transcript.append_bytes(b"pow/anti-envelope/root", &expected_root);
        transcript.append_bytes(b"pow/anti-envelope/query", &query_digest);
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
        match (key_pairs.is_empty(), anti_proof.as_ref()) {
            (true, None) => verify_global_not(
                &left_commit, &right_commit, &result_commitments, result_len, result_log_domain,
            )?,
            (false, Some(proof)) => verify_left_anti_join_with_keys(
                &left_commit, &right_commit, &key_pairs, &result_commitments, result_len,
                result_log_domain, proof, &mut transcript,
            )?,
            _ => return Err(invalid("left-anti envelope key/proof variant mismatch")),
        }
        verify_result_columns(
            &result_commitments,
            &result_columns,
            result_len,
            result_log_domain,
        )?;
        let result_objects = if ground_result {
            Vec::new()
        } else {
            resolve_typed_result_objects(
                layer,
                layer.resolution_root(),
                &result_columns,
                &result_id_namespaces(left_plan)?,
            )?
        };
        Ok(Self {
            layer_root: expected_root,
            query_digest,
            left,
            right,
            key_pairs,
            anti_proof,
            result_commitments,
            result_columns,
            result_log_domain,
            result_len,
            ground_result,
            result_objects,
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
        expected_ground_result: bool,
    ) -> io::Result<()> {
        if self.layer_root != expected_root {
            return Err(invalid(
                "left-anti envelope root does not match trusted root",
            ));
        }
        if self.query_digest != query_digest {
            return Err(invalid("left-anti public-query identity mismatch"));
        }
        if self.key_pairs != key_pairs {
            return Err(invalid("left-anti correlation key mismatch"));
        }
        if self.ground_result != expected_ground_result {
            return Err(invalid("left-anti ground-result interpretation mismatch"));
        }
        self.left
            .verify(layer, expected_root, left_plan, query_digest)?;
        self.right
            .verify(layer, expected_root, right_plan, query_digest)?;
        let mut transcript = crate::proof::argument::Transcript254::new(b"parture-left-anti-envelope");
        transcript.append_bytes(b"pow/anti-envelope/root", &expected_root);
        transcript.append_bytes(b"pow/anti-envelope/query", &query_digest);
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
        match (key_pairs.is_empty(), self.anti_proof.as_ref()) {
            (true, None) => verify_global_not(
                &left_commit, &right_commit, &self.result_commitments, self.result_len,
                self.result_log_domain,
            )?,
            (false, Some(proof)) => verify_left_anti_join_with_keys(
                &left_commit, &right_commit, key_pairs, &self.result_commitments,
                self.result_len, self.result_log_domain, proof, &mut transcript,
            )?,
            _ => return Err(invalid("left-anti envelope key/proof variant mismatch")),
        }
        verify_result_columns(
            &self.result_commitments,
            &self.result_columns,
            self.result_len,
            self.result_log_domain,
        )?;
        if self.ground_result {
            if self.result_commitments.len() != 1 || self.result_columns.len() != 1
                || !self.result_objects.is_empty()
            {
                return Err(invalid("invalid ground left-anti envelope shape"));
            }
            Ok(())
        } else {
            let resolved = resolve_typed_result_objects(
                layer,
                layer.resolution_root(),
                &self.result_columns,
                &result_id_namespaces(left_plan)?,
            )?;
            if resolved != self.result_objects {
                return Err(invalid(
                    "delivered result objects do not match typed authenticated Store IDs",
                ));
            }
            Ok(())
        }
    }

    pub fn encode(&self) -> io::Result<Vec<u8>> {
        let mut out = Vec::new();
        out.extend_from_slice(ANTI_MAGIC);
        out.extend_from_slice(&ANTI_VERSION.to_le_bytes());
        self.put(&mut out)?;
        if out
            .len()
            .checked_add(CHECKSUM_BYTES)
            .map_or(true, |len| len > MAX_QUERY_ENVELOPE_BYTES)
        {
            return Err(invalid("left-anti envelope too large"));
        }
        let checksum = Sha256::digest(&out);
        out.extend_from_slice(&checksum);
        Ok(out)
    }

    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < ANTI_MAGIC.len() + 4 + CHECKSUM_BYTES
            || bytes.len() > MAX_QUERY_ENVELOPE_BYTES
        {
            return Err(invalid("invalid left-anti envelope size"));
        }
        let body_len = bytes.len() - CHECKSUM_BYTES;
        let (body, checksum) = bytes.split_at(body_len);
        if Sha256::digest(body).as_slice() != checksum {
            return Err(invalid("left-anti envelope checksum mismatch"));
        }
        let mut reader = Reader {
            bytes: body,
            pos: 0,
            decoded_items: 0,
        };
        if reader.take(ANTI_MAGIC.len())? != ANTI_MAGIC {
            return Err(invalid("unknown left-anti envelope magic"));
        }
        let version = u32::from_le_bytes(reader.take(4)?.try_into().unwrap());
        if version != ANTI_VERSION {
            return Err(invalid("unsupported left-anti envelope version"));
        }
        let envelope = Self::get(&mut reader)?;
        reader.finish()?;
        Ok(envelope)
    }
}
