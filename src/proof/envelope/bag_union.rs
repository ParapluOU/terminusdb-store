//! The flat bag-union envelope (`UNION` / disjunction of compatible branches).
//!
//! [`BagUnionEnvelope`] wraps two or more independently verifiable [`LayerBgpEnvelope`]
//! branches against the same root and adds a proof that the delivered result is exactly the
//! MULTISET union of the branches' results (duplicates preserved — it is a bag, not a set). Each
//! branch is re-verified in full, then `union_proof` authenticates the combined output against
//! the branch commitments. Branch result schemas (column namespaces) must match, so the union
//! is type-coherent.
#![allow(unused_imports)]

use super::layer_bgp::LayerBgpEnvelope;
use super::wire::{
    invalid, wire_struct, Reader, Wire, CHECKSUM_BYTES, MAX_QUERY_ENVELOPE_BYTES, UNION_MAGIC,
    UNION_VERSION,
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
/// Portable exact-bag-union envelope. Every branch remains a complete independently
/// verifiable layer envelope against the same root; `union_proof` authenticates their
/// compatible-schema result multiset as the public output.
#[derive(Clone, Debug)]
pub struct BagUnionEnvelope {
    pub layer_root: ProofHash,
    pub query_digest: ProofHash,
    pub branches: Vec<LayerBgpEnvelope>,
    pub union_proof: BagUnionProof,
    pub result_commitments: Vec<ArkGT>,
    pub result_columns: Vec<Vec<ArkFr>>,
    pub result_log_domain: usize,
    pub result_len: usize,
    pub ground_result: bool,
    pub result_objects: Vec<Vec<CanonicalObject>>,
}

wire_struct!(BagUnionEnvelope {
    layer_root,
    query_digest,
    branches,
    union_proof,
    result_commitments,
    result_columns,
    result_log_domain,
    result_len,
    ground_result,
    result_objects
});

impl BagUnionEnvelope {
    #[allow(clippy::too_many_arguments)]
    pub fn new_verified(
        layer: &LayerCommitment,
        expected_root: ProofHash,
        branch_plans: &[LayerBgpPlan],
        query_digest: ProofHash,
        branches: Vec<LayerBgpEnvelope>,
        union_proof: BagUnionProof,
        result_commitments: Vec<ArkGT>,
        result_columns: Vec<Vec<ArkFr>>,
        result_log_domain: usize,
        result_len: usize,
        ground_result: bool,
    ) -> io::Result<Self> {
        if branches.len() != branch_plans.len() || branches.len() < 2 {
            return Err(invalid("bag-union envelope branch count mismatch"));
        }
        for (branch, plan) in branches.iter().zip(branch_plans) {
            branch.verify(layer, expected_root, plan, query_digest)?;
        }
        let inputs: Vec<RelationCommit> = branches
            .iter()
            .map(|branch| RelationCommit {
                commitments: branch.result_commitments.clone(),
                log_domain: branch.result_log_domain,
                n: branch.result_len,
            })
            .collect();
        let mut transcript = crate::proof::argument::Transcript254::new(b"parture-union-envelope");
        transcript.append_bytes(b"pow/union-envelope/root", &expected_root);
        transcript.append_bytes(b"pow/union-envelope/query", &query_digest);
        verify_bag_union(
            &inputs,
            &result_commitments,
            result_len,
            result_log_domain,
            &union_proof,
            &mut transcript,
        )?;
        verify_result_columns(
            &result_commitments,
            &result_columns,
            result_len,
            result_log_domain,
        )?;
        let result_objects = if ground_result {
            if result_columns.len() != 1 || result_commitments.len() != 1 {
                return Err(invalid(
                    "ground bag union must carry one internal UNIT column",
                ));
            }
            Vec::new()
        } else {
            let namespaces = result_id_namespaces(&branch_plans[0])?;
            for plan in &branch_plans[1..] {
                if result_id_namespaces(plan)? != namespaces {
                    return Err(invalid("bag-union branch result namespace mismatch"));
                }
            }
            resolve_typed_result_objects(layer, layer.resolution_root(), &result_columns, &namespaces)?
        };
        Ok(Self {
            layer_root: expected_root,
            query_digest,
            branches,
            union_proof,
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
        branch_plans: &[LayerBgpPlan],
        query_digest: ProofHash,
        expected_ground_result: bool,
    ) -> io::Result<()> {
        if self.layer_root != expected_root {
            return Err(invalid(
                "bag-union envelope root does not match trusted root",
            ));
        }
        if self.query_digest != query_digest {
            return Err(invalid("bag-union public-query identity mismatch"));
        }
        if self.ground_result != expected_ground_result {
            return Err(invalid("bag-union ground-result interpretation mismatch"));
        }
        if self.branches.len() != branch_plans.len() || self.branches.len() < 2 {
            return Err(invalid("bag-union envelope branch count mismatch"));
        }
        for (branch, plan) in self.branches.iter().zip(branch_plans) {
            branch.verify(layer, expected_root, plan, query_digest)?;
        }
        let inputs: Vec<RelationCommit> = self
            .branches
            .iter()
            .map(|branch| RelationCommit {
                commitments: branch.result_commitments.clone(),
                log_domain: branch.result_log_domain,
                n: branch.result_len,
            })
            .collect();
        let mut transcript = crate::proof::argument::Transcript254::new(b"parture-union-envelope");
        transcript.append_bytes(b"pow/union-envelope/root", &expected_root);
        transcript.append_bytes(b"pow/union-envelope/query", &query_digest);
        verify_bag_union(
            &inputs,
            &self.result_commitments,
            self.result_len,
            self.result_log_domain,
            &self.union_proof,
            &mut transcript,
        )?;
        verify_result_columns(
            &self.result_commitments,
            &self.result_columns,
            self.result_len,
            self.result_log_domain,
        )?;
        if self.ground_result {
            if self.result_columns.len() != 1
                || self.result_commitments.len() != 1
                || !self.result_objects.is_empty()
            {
                return Err(invalid("invalid ground bag-union envelope shape"));
            }
            Ok(())
        } else {
            let namespaces = result_id_namespaces(&branch_plans[0])?;
            for plan in &branch_plans[1..] {
                if result_id_namespaces(plan)? != namespaces {
                    return Err(invalid("bag-union branch result namespace mismatch"));
                }
            }
            let resolved = resolve_typed_result_objects(
                layer,
                layer.resolution_root(),
                &self.result_columns,
                &namespaces,
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
        out.extend_from_slice(UNION_MAGIC);
        out.extend_from_slice(&UNION_VERSION.to_le_bytes());
        self.put(&mut out)?;
        if out
            .len()
            .checked_add(CHECKSUM_BYTES)
            .map_or(true, |len| len > MAX_QUERY_ENVELOPE_BYTES)
        {
            return Err(invalid("bag-union envelope too large"));
        }
        let checksum = Sha256::digest(&out);
        out.extend_from_slice(&checksum);
        Ok(out)
    }

    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < UNION_MAGIC.len() + 4 + CHECKSUM_BYTES
            || bytes.len() > MAX_QUERY_ENVELOPE_BYTES
        {
            return Err(invalid("invalid bag-union envelope size"));
        }
        let body_len = bytes.len() - CHECKSUM_BYTES;
        let (body, checksum) = bytes.split_at(body_len);
        if Sha256::digest(body).as_slice() != checksum {
            return Err(invalid("bag-union envelope checksum mismatch"));
        }
        let mut reader = Reader {
            bytes: body,
            pos: 0,
            decoded_items: 0,
        };
        if reader.take(UNION_MAGIC.len())? != UNION_MAGIC {
            return Err(invalid("unknown bag-union envelope magic"));
        }
        let version = u32::from_le_bytes(reader.take(4)?.try_into().unwrap());
        if version != UNION_VERSION {
            return Err(invalid("unsupported bag-union envelope version"));
        }
        let envelope = Self::get(&mut reader)?;
        reader.finish()?;
        Ok(envelope)
    }
}
