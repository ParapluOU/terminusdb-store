//! The leaf portable envelope: a proof of ONE basic-graph-pattern read of a single layer.
//!
//! [`LayerBgpEnvelope`] is the base case every other shape builds on. It proves that running
//! the given [`LayerBgpPlan`](crate::proof::query::LayerBgpPlan) against the triples committed
//! under a trusted root yields exactly the delivered result columns, and it re-derives the
//! delivered canonical objects from the root-authenticated dictionary so the caller can trust
//! the human-readable output, not just the internal ids.
//!
//! `new_verified` builds one by checking a freshly produced proof; `verify` re-checks a decoded
//! one against the caller's trusted root, plan digest and query digest; `encode`/`decode` are
//! the framed wire form. Decoding never trusts — `verify` does.
#![allow(unused_imports)]

use super::wire::{
    invalid, plan_digest, wire_struct, Reader, Wire, CHECKSUM_BYTES, MAGIC,
    MAX_QUERY_ENVELOPE_BYTES, VERSION,
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
    resolve_typed_result_objects, resolve_typed_result_objects_with_openings, result_id_namespaces,
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
/// Portable proof plus its public result. Creation authenticates canonical objects;
/// decoding is only parsing, while [`Self::verify`] performs all trust checks.
#[derive(Clone, Debug)]
pub struct LayerBgpEnvelope {
    pub layer_root: ProofHash,
    pub plan_digest: ProofHash,
    /// Caller-domain identity layered over the Store plan (for example WOQL schema,
    /// projection, routing and aggregate metadata).
    pub query_digest: ProofHash,
    pub proof: LayerBgpProof,
    pub result_commitments: Vec<ArkGT>,
    pub result_columns: Vec<Vec<ArkFr>>,
    pub result_log_domain: usize,
    pub result_len: usize,
    pub predicate_ids: Vec<u64>,
    pub result_objects: Vec<Vec<CanonicalObject>>,
}

wire_struct!(LayerBgpEnvelope {
    layer_root,
    plan_digest,
    query_digest,
    proof,
    result_commitments,
    result_columns,
    result_log_domain,
    result_len,
    predicate_ids,
    result_objects
});

impl LayerBgpEnvelope {
    #[allow(clippy::too_many_arguments)]
    pub fn new_verified(
        layer: &LayerCommitment,
        expected_root: ProofHash,
        plan: &LayerBgpPlan,
        query_digest: ProofHash,
        proof: LayerBgpProof,
        result_commitments: Vec<ArkGT>,
        result_columns: Vec<Vec<ArkFr>>,
        result_log_domain: usize,
        result_len: usize,
        predicate_ids: Vec<u64>,
    ) -> io::Result<Self> {
        verify_layer_bgp(
            layer,
            // β re-anchor: the leaf verify anchors on `resolution_root`. Inside the envelope
            // the layer is already bound to the trusted anchor via `self.layer_root`/the outer
            // boundary; derive its resolution_root from the same layer.
            layer.resolution_root(),
            plan,
            &predicate_ids,
            &result_commitments,
            result_len,
            &proof,
        )?;
        verify_layer_bgp_result_log_domain(layer, plan, &proof, result_log_domain)?;
        verify_result_columns(
            &result_commitments,
            &result_columns,
            result_len,
            result_log_domain,
        )?;
        let result_objects = if has_ground_result_unit(plan) {
            if result_columns.len() != 1 || result_commitments.len() != 1 {
                return Err(invalid(
                    "ground result must carry exactly one internal UNIT column",
                ));
            }
            Vec::new()
        } else {
            // Merkle-dict I2b S5: consume the proof's term-less openings (`None` on a term-bearing
            // proof — resolution rebuilds the tree). The proof was just belt-verified above.
            resolve_typed_result_objects_with_openings(
                layer,
                layer.resolution_root(),
                &result_columns,
                &result_id_namespaces(plan)?,
                proof.openings.as_ref(),
            )?
        };
        Ok(Self {
            layer_root: expected_root,
            plan_digest: plan_digest(plan)?,
            query_digest,
            proof,
            result_commitments,
            result_columns,
            result_log_domain,
            result_len,
            predicate_ids,
            result_objects,
        })
    }

    pub fn verify(
        &self,
        layer: &impl VerifierLayer,
        expected_root: ProofHash,
        plan: &LayerBgpPlan,
        query_digest: ProofHash,
    ) -> io::Result<()> {
        if self.layer_root != expected_root {
            return Err(invalid("envelope root does not match trusted root"));
        }
        if self.plan_digest != plan_digest(plan)? {
            return Err(invalid("envelope plan identity mismatch"));
        }
        if self.query_digest != query_digest {
            return Err(invalid("envelope public-query identity mismatch"));
        }
        verify_layer_bgp(
            layer,
            layer.resolution_root(),
            plan,
            &self.predicate_ids,
            &self.result_commitments,
            self.result_len,
            &self.proof,
        )?;
        verify_layer_bgp_result_log_domain(layer, plan, &self.proof, self.result_log_domain)?;
        verify_result_columns(
            &self.result_commitments,
            &self.result_columns,
            self.result_len,
            self.result_log_domain,
        )?;
        if has_ground_result_unit(plan) {
            if self.result_columns.len() != 1
                || self.result_commitments.len() != 1
                || !self.result_objects.is_empty()
            {
                return Err(invalid("invalid ground-result envelope shape"));
            }
            Ok(())
        } else {
            // Merkle-dict I2b S5: term-less resolution consumes the proof's shipped openings.
            let resolved = resolve_typed_result_objects_with_openings(
                layer,
                layer.resolution_root(),
                &self.result_columns,
                &result_id_namespaces(plan)?,
                self.proof.openings.as_ref(),
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
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        self.put(&mut out)?;
        if out
            .len()
            .checked_add(CHECKSUM_BYTES)
            .map_or(true, |len| len > MAX_QUERY_ENVELOPE_BYTES)
        {
            return Err(invalid("query-proof envelope too large"));
        }
        let checksum = Sha256::digest(&out);
        out.extend_from_slice(&checksum);
        Ok(out)
    }

    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < MAGIC.len() + 4 + CHECKSUM_BYTES || bytes.len() > MAX_QUERY_ENVELOPE_BYTES
        {
            return Err(invalid("invalid query-proof envelope size"));
        }
        let body_len = bytes.len() - CHECKSUM_BYTES;
        let (body, checksum) = bytes.split_at(body_len);
        if Sha256::digest(body).as_slice() != checksum {
            return Err(invalid("query-proof envelope checksum mismatch"));
        }
        let mut reader = Reader {
            bytes: body,
            pos: 0,
            decoded_items: 0,
        };
        if reader.take(MAGIC.len())? != MAGIC {
            return Err(invalid("unknown query-proof envelope magic"));
        }
        let version = u32::from_le_bytes(reader.take(4)?.try_into().unwrap());
        if version != VERSION {
            return Err(invalid("unsupported query-proof envelope version"));
        }
        let envelope = Self::get(&mut reader)?;
        reader.finish()?;
        Ok(envelope)
    }
}
