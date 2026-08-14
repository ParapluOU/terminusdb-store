//! Wire format and framing for the relation-proof envelope.
//!
//! This is the low-level serialization layer shared by every envelope shape. It is pure
//! encoding/decoding and resource-bounding — it performs NO trust checks (those live in each
//! envelope's `verify`).
//!
//! Key pieces:
//! * [`Wire`] — a tiny put/get trait implemented for every leaf type that appears in an
//!   envelope (primitives, the arkworks field/group elements, and each operator sub-proof).
//! * [`Reader`] — a cursor over the input with two hard budgets: a byte length it can never
//!   read past, and a GLOBAL decoded-item counter (`claim_items`) shared across all nested
//!   vectors, so an attacker cannot force tens of millions of tiny allocations from a small,
//!   individually valid input.
//! * The `*_MAGIC` tags + `*_VERSION` numbers + `CHECKSUM_BYTES` that frame each encoded
//!   envelope, and the `MAX_*` admission limits.
//! * [`plan_digest`] — the stable hash of the public (variable-blind) layer BGP plan. The
//!   verifier recomputes it from its own compiled query, which is what pins WHICH committed
//!   columns a proof was allowed to read (the anti-substitution guard).
//!
//! The `Wire` implementations for the envelope-internal transitive-closure records live here
//! too, so all serialization sits in one place; those types are defined in the `recursive`
//! module and imported below.
#![allow(unused_imports)]

use super::recursive::{TransitiveClosureFixpoint, TransitiveClosureRound, TransitiveClosureSeed};
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
pub(super) const MAGIC: &[u8; 12] = b"TDBWQPROOF\0\0";
pub(super) const UNION_MAGIC: &[u8; 12] = b"TDBWQUNION\0\0";
pub(super) const ANTI_MAGIC: &[u8; 12] = b"TDBWQANTI\0\0\0";
pub(super) const OUTER_MAGIC: &[u8; 12] = b"TDBWQOUTER\0\0";
pub(super) const RECURSIVE_MAGIC: &[u8; 12] = b"TDBWQTREE\0\0\0";

/// Canonical hidden column for a zero-column relation: one on every real row and zero
/// on commitment padding. This is the same representation used by all-ground BGPs.
pub(super) fn ground_unit_column(n: usize, log_domain: usize) -> Vec<ArkFr> {
    let mut column = vec![ArkFr::zero(); 1usize << log_domain];
    column[..n].fill(ArkFr::one());
    column
}
// v5 adds the explicit transcript/wire-bound object-kind filter. Older envelopes fail
// explicitly rather than guessing whether a variable object meant any value, a node,
// or a literal.
pub(super) const VERSION: u32 = 5;
pub(super) const UNION_VERSION: u32 = 3;
pub(super) const ANTI_VERSION: u32 = 5;
pub(super) const OUTER_VERSION: u32 = 2;
// v21 generalizes grouped extrema to zero through two canonical node keys.
// v22 adds the bounded transitive-closure (`star`/`plus`) composite node.
// v23 adds the ordered `sys:Array` walk composite node (tag 24).
pub(super) const RECURSIVE_VERSION: u32 = 23;
pub(super) const CHECKSUM_BYTES: usize = 32;
pub(super) const MAX_ITEMS: usize = 4_000_000;
pub(super) const MAX_BLOB_BYTES: usize = 64 * 1024 * 1024;
/// Hard cap for an untrusted portable query-proof envelope.
pub const MAX_QUERY_ENVELOPE_BYTES: usize = 256 * 1024 * 1024;
// Keep this conservative: decoding a node includes large cryptographic proof values on
// the stack, so a much larger syntactic limit could exhaust ordinary verifier threads
// before the guard is reached.
pub const MAX_RELATION_PROOF_DEPTH: usize = 64;
pub const MAX_RELATION_PROOF_NODES: usize = 4096;
/// The maximum number of bounded semi-naive closure ROUNDS a [`RelationProofPlanKind::TransitiveClosure`]
/// proof may carry. `K` is prover-chosen (the depth at which the frontier stops growing), NOT declared
/// in the plan; soundness rests entirely on the empty-delta fixpoint, so this bound is purely a COST
/// guard against an unbounded (or pathologically deep) unroll. A closure whose reachable frontier from
/// the seed is deeper than this budget REJECTS CLOSED (there is no committed graph-diameter bound to
/// certify a shorter proof). Each round threads three audited operator proofs, so this also keeps the
/// flat round list well inside [`MAX_RELATION_PROOF_NODES`].
pub const MAX_TRANSITIVE_CLOSURE_ROUNDS: usize = 64;

pub(super) fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

pub(super) struct Reader<'a> {
    pub(super) bytes: &'a [u8],
    pub(super) pos: usize,
    pub(super) decoded_items: usize,
}

impl<'a> Reader<'a> {
    pub(super) fn take(&mut self, len: usize) -> io::Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| invalid("envelope length overflow"))?;
        if end > self.bytes.len() {
            return Err(invalid("truncated query-proof envelope"));
        }
        let value = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(value)
    }

    pub(super) fn finish(self) -> io::Result<()> {
        if self.pos != self.bytes.len() {
            return Err(invalid("trailing bytes in query-proof envelope"));
        }
        Ok(())
    }

    pub(super) fn claim_items(&mut self, len: usize) -> io::Result<()> {
        self.decoded_items = self
            .decoded_items
            .checked_add(len)
            .ok_or_else(|| invalid("decoded envelope item count overflow"))?;
        if self.decoded_items > MAX_ITEMS {
            return Err(invalid("decoded envelope exceeds global item budget"));
        }
        Ok(())
    }
}

pub(super) trait Wire: Sized {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()>;
    fn get(reader: &mut Reader<'_>) -> io::Result<Self>;
}

impl<T: Wire> Wire for Box<T> {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        (**self).put(out)
    }

    #[inline(never)]
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        Ok(Box::new(T::get(reader)?))
    }
}

impl Wire for u8 {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        out.push(*self);
        Ok(())
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        Ok(reader.take(1)?[0])
    }
}
impl Wire for bool {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        (*self as u8).put(out)
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        match u8::get(reader)? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(invalid("invalid boolean")),
        }
    }
}
impl Wire for u64 {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        out.extend_from_slice(&self.to_le_bytes());
        Ok(())
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        Ok(u64::from_le_bytes(reader.take(8)?.try_into().unwrap()))
    }
}
impl Wire for i64 {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        out.extend_from_slice(&self.to_le_bytes());
        Ok(())
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        Ok(i64::from_le_bytes(reader.take(8)?.try_into().unwrap()))
    }
}
impl Wire for usize {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        u64::try_from(*self)
            .map_err(|_| invalid("usize does not fit envelope u64"))?
            .put(out)
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        usize::try_from(u64::get(reader)?).map_err(|_| invalid("envelope u64 does not fit usize"))
    }
}
impl Wire for ProofHash {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        out.extend_from_slice(self);
        Ok(())
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        Ok(reader.take(32)?.try_into().unwrap())
    }
}
impl<T: Wire> Wire for Option<T> {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        match self {
            None => 0u8.put(out),
            Some(value) => {
                1u8.put(out)?;
                value.put(out)
            }
        }
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        match u8::get(reader)? {
            0 => Ok(None),
            1 => Ok(Some(T::get(reader)?)),
            _ => Err(invalid("invalid option tag")),
        }
    }
}
impl<T: Wire> Wire for Vec<T> {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        if self.len() > MAX_ITEMS {
            return Err(invalid("too many envelope items"));
        }
        self.len().put(out)?;
        for value in self {
            value.put(out)?;
        }
        Ok(())
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        let len = usize::get(reader)?;
        if len > MAX_ITEMS {
            return Err(invalid("too many envelope items"));
        }
        // Nested vectors share one budget. Without this, a bounded-size recursive
        // envelope can force tens of millions of decode iterations by distributing
        // individually-valid tiny vectors across many relation nodes.
        reader.claim_items(len)?;
        // `len` is attacker-controlled. Even though it is capped, reserving it up
        // front can allocate far more than the bounded input for a large `T` before
        // the first truncated element is noticed. Grow only after elements decode.
        let mut values = Vec::new();
        for _ in 0..len {
            values.push(T::get(reader)?);
        }
        Ok(values)
    }
}
impl<A: Wire, B: Wire> Wire for (A, B) {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        self.0.put(out)?;
        self.1.put(out)
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        Ok((A::get(reader)?, B::get(reader)?))
    }
}

pub(super) fn put_blob(bytes: &[u8], out: &mut Vec<u8>) -> io::Result<()> {
    if bytes.len() > MAX_BLOB_BYTES {
        return Err(invalid("envelope blob too large"));
    }
    bytes.len().put(out)?;
    out.extend_from_slice(bytes);
    Ok(())
}
pub(super) fn get_blob<'a>(reader: &mut Reader<'a>) -> io::Result<&'a [u8]> {
    let len = usize::get(reader)?;
    if len > MAX_BLOB_BYTES {
        return Err(invalid("envelope blob too large"));
    }
    reader.take(len)
}

macro_rules! canonical_wire {
    ($ty:ty) => {
        impl Wire for $ty {
            fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
                let mut bytes = Vec::new();
                self.serialize_compressed(&mut bytes)
                    .map_err(|_| invalid("canonical proof serialization failed"))?;
                put_blob(&bytes, out)
            }
            fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
                let bytes = get_blob(reader)?;
                let mut cursor = Cursor::new(bytes);
                let value = Self::deserialize_compressed(&mut cursor)
                    .map_err(|_| invalid("invalid canonical proof encoding"))?;
                if cursor.position() as usize != bytes.len() {
                    return Err(invalid("trailing canonical proof bytes"));
                }
                Ok(value)
            }
        }
    };
}
canonical_wire!(ArkFr);
canonical_wire!(ArkGT);
canonical_wire!(ColumnOpening);

macro_rules! wire_struct {
    ($ty:ty { $($field:ident),+ $(,)? }) => {
        impl Wire for $ty {
            fn put(&self, out: &mut Vec<u8>) -> io::Result<()> { $(self.$field.put(out)?;)+ Ok(()) }
            fn get(reader: &mut Reader<'_>) -> io::Result<Self> { Ok(Self { $($field: Wire::get(reader)?,)+ }) }
        }
    };
}
pub(super) use wire_struct;

wire_struct!(SumcheckProof {
    round_polys,
    claimed_evals
});
wire_struct!(DictProof {
    sumcheck,
    opening,
    mult_commitment,
    dstar_commitment
});
wire_struct!(FilterProof { sumcheck, opening });
wire_struct!(FullFilterProof {
    sumcheck,
    opening,
    sel_commitment,
    w_commitment,
    cstar_commitment,
    dstar_commitment
});
wire_struct!(SelectionOnlyProof {
    sumcheck,
    opening,
    w_commitment
});
wire_struct!(BooleanProductProof { sumcheck, opening });
wire_struct!(CompactionOnlyProof {
    sumcheck,
    opening,
    cstar_commitment,
    dstar_commitment
});
wire_struct!(RelationEqualityProof {
    indicator,
    selection_commitment,
    inverse_commitment,
    compaction
});
wire_struct!(TupleDisequalityStageProof {
    indicator,
    selection_commitment,
    inverse_commitment,
    compaction,
    output_commitments,
    output_len
});
wire_struct!(TupleDisequalityProof { stages });
wire_struct!(RelationDistinctEntryProof {
    filters,
    commitments,
    lengths
});
wire_struct!(RelationDistinctProof { entries });
wire_struct!(RelationPermutationProof {
    input_reciprocal_commitment,
    output_reciprocal_commitment,
    sumcheck,
    opening
});
wire_struct!(ExactScalarOrderProof {
    permutation,
    relocate_output_id,
    relocate_dict_id,
    relocate_dict_value,
    relocate_dict_datatype,
    com_output_id_at_d,
    com_dict_id_at_d,
    com_dict_value_at_d,
    com_dict_datatype_at_d,
    lookup,
    datatype_filter,
    datatype_selection_commitment,
    datatype_inverse_commitment,
    sorted
});
wire_struct!(LexicographicSortedProof { sorted });
wire_struct!(ExactMultiOrderProof { permutation, lexicographic });

impl Wire for OrderDirection {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        match self { Self::Ascending => 0u8, Self::Descending => 1u8 }.put(out)
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        match u8::get(reader)? {
            0 => Ok(Self::Ascending),
            1 => Ok(Self::Descending),
            _ => Err(invalid("invalid order direction")),
        }
    }
}
wire_struct!(GroupExtremaEntryProof {
    group,
    group_commitments,
    group_len,
    membership,
    membership_commitments,
    membership_len,
    relocate_value,
    relocate_dict_id,
    relocate_dict_value,
    relocate_dict_datatype,
    com_value_at_d,
    com_dict_id_at_d,
    com_dict_value_at_d,
    com_dict_datatype_at_d,
    range,
    com_range_selection
});
wire_struct!(GroupExtremaProof { entries });

impl Wire for GroupExtremaKind {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        match self { Self::Min => 0u8, Self::Max => 1u8 }.put(out)
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        match u8::get(reader)? {
            0 => Ok(Self::Min),
            1 => Ok(Self::Max),
            _ => Err(invalid("invalid group-extrema kind")),
        }
    }
}
wire_struct!(RelocationProof {
    source_eval,
    target_eval,
    source_opening,
    target_opening
});
wire_struct!(BagUnionProof {
    relocations,
    relocated_commitments,
    reciprocal_commitments,
    sumcheck,
    opening
});
wire_struct!(LeftAntiJoinProof {
    left_relocations,
    right_relocations,
    join_commitments,
    join,
    join_len,
    zero,
    zero_selection_commitment,
    zero_inverse_commitment,
    compaction
});
wire_struct!(InnerJoinProof { anti, unmatched_left_commitments, unmatched_len });
// The positional-successor primitive (`proof::sort`) consumed by the ordered `rdf:List` walk.
wire_struct!(crate::proof::sort::SuccessorProof {
    sumcheck,
    opening,
    mult_commitment,
    rec_c_commitment,
    rec_d_commitment
});
wire_struct!(TransitiveClosureRound {
    inner,
    inner_out,
    union,
    union_out,
    distinct,
    next_rows
});
wire_struct!(TransitiveClosureFixpoint { inner, inner_out, anti, anti_out });
impl Wire for TransitiveClosureSeed {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        match self {
            TransitiveClosureSeed::Reflexive => 0u8.put(out),
            TransitiveClosureSeed::Successor {
                inner,
                inner_out,
                distinct,
                rows,
            } => {
                1u8.put(out)?;
                inner.put(out)?;
                inner_out.put(out)?;
                distinct.put(out)?;
                rows.put(out)
            }
        }
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        match u8::get(reader)? {
            0 => Ok(TransitiveClosureSeed::Reflexive),
            1 => Ok(TransitiveClosureSeed::Successor {
                inner: Wire::get(reader)?,
                inner_out: Wire::get(reader)?,
                distinct: Wire::get(reader)?,
                rows: Wire::get(reader)?,
            }),
            _ => Err(invalid("invalid transitive-closure seed tag")),
        }
    }
}
wire_struct!(LeftOuterJoinProof {
    anti,
    unmatched_left_commitments,
    unmatched_len,
    union
});
wire_struct!(SortedProof {
    sumcheck,
    opening,
    u_shift_commitment,
    mult_c_commitment,
    rec_c_commitment,
    rec_d_commitment,
    bit_commitments
});
wire_struct!(MembershipProof {
    sumcheck,
    opening,
    rec_u_commitment,
    rec_of_commitment
});
wire_struct!(CountProof { sumcheck, opening });
wire_struct!(BinaryJoinCommitments {
    r_key,
    r_other,
    s_key,
    s_other,
    r_other_out,
    key,
    s_other_out,
    rho_l,
    rho_r,
    u,
    w_l,
    w_r
});
wire_struct!(BinaryPairsProof {
    membership_sumcheck,
    membership_opening,
    mult_l_commitment,
    mult_r_commitment,
    rec_src_l_commitment,
    rec_cand_l_commitment,
    rec_src_r_commitment,
    rec_cand_r_commitment,
    sorted
});
wire_struct!(BinaryJoinProof {
    pairs,
    sorted_u,
    membership_l,
    membership_r,
    count,
    n_u
});
wire_struct!(BgpStageProof {
    acc_relocations,
    rel_relocations,
    commitments,
    join,
    m_j
});
wire_struct!(BgpChainProof { stages });
wire_struct!(RangeSelectProof {
    sumcheck,
    opening,
    bit_commitments
});
wire_struct!(DictLookupProof {
    sumcheck,
    opening,
    com_v_val,
    com_v_dt,
    com_w,
    com_rec_q,
    com_rec_d
});
wire_struct!(HashLookupInnerProof {
    sumcheck,
    opening,
    com_v_hash,
    com_w,
    com_rec_q,
    com_rec_d
});
wire_struct!(DictHashLookupProof {
    relocate_dict_hash,
    relocate_dict_id,
    com_dict_hash_at_d,
    com_dict_id_at_d,
    relocate_obj,
    com_obj_at_d,
    lookup
});
wire_struct!(SelMaskProof { sumcheck, opening });
wire_struct!(HashSemiJoinProof {
    sel_mask,
    com_sel,
    compaction,
    com_x_kept,
    com_h_kept,
    com_w,
    membership
});
wire_struct!(ObjectRangeCombineProof { sumcheck, opening });
wire_struct!(ObjectRangeProof {
    lookup,
    datatype,
    com_dt_match,
    com_dt_w,
    range,
    com_range_sel,
    combine
});
wire_struct!(SecondRangeSubproof {
    object_range,
    com_range_sel_at_d,
    relocate_range_sel,
    com_range_sel_at_table,
    intersect,
    com_intersection_sel
});
wire_struct!(RangeAtomSubproof {
    relocate_object,
    relocate_dict_id,
    relocate_dict_value,
    relocate_dict_datatype,
    com_object_at_d,
    com_dict_id_at_d,
    com_dict_value_at_d,
    com_dict_datatype_at_d,
    object_range,
    com_range_sel_at_d,
    relocate_range_sel,
    com_range_sel_at_table,
    second,
    equality,
    com_equality_sel,
    combine,
    com_combined_sel,
    compaction
});
wire_struct!(LayerBgpAtomProof {
    predicate_membership,
    subject_membership,
    object_membership,
    filter,
    range,
    kept_commitments,
    m,
    subject_id,
    object_id
});
wire_struct!(LayerBgpProof {
    atoms,
    chain,
    dictionary_absence
});
wire_struct!(JoinStep {
    acc_key,
    rel_key,
    additional_keys
});
wire_struct!(ObjectRangeBound { op, c });
wire_struct!(ObjectRangeSpec {
    op,
    datatype,
    c,
    b_bits,
    second
});
// `source` is part of each atom's wire image, so `plan_digest` (and therefore the envelope
// anti-substitution check) fixes which committed columns THIS atom read. This is what stops a
// prover relabelling one atom of a cross-source join from additions to materialized (or
// swapping additions↔removals): the verifier reconstructs the same per-atom source and opens
// against the columns it selects.
wire_struct!(LayerBgpAtom {
    subject,
    predicate_term,
    predicate_variable,
    object,
    object_range,
    object_kind,
    subject_object_equal,
    source
});
impl Wire for ColumnSource {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        match self {
            Self::Materialized => 0u8,
            Self::Additions => 1,
            Self::Removals => 2,
        }
        .put(out)
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        match u8::get(reader)? {
            0 => Ok(Self::Materialized),
            1 => Ok(Self::Additions),
            2 => Ok(Self::Removals),
            _ => Err(invalid("invalid column-source tag")),
        }
    }
}
// `source` is part of the plan's wire image, so `plan_digest` (and therefore the envelope
// anti-substitution check) binds which columns the proof read.
wire_struct!(LayerBgpPlan { atoms, joins, source });

wire_struct!(DictionaryAbsence {
    atom_index,
    position
});

impl Wire for DictionaryAbsencePosition {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        match self {
            Self::Predicate => 0u8,
            Self::Subject => 1u8,
            Self::Object => 2u8,
        }
        .put(out)
    }

    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        match u8::get(reader)? {
            0 => Ok(Self::Predicate),
            1 => Ok(Self::Subject),
            2 => Ok(Self::Object),
            _ => Err(invalid("invalid dictionary-absence position tag")),
        }
    }
}

impl Wire for AtomFilterProof {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        match self {
            Self::Identity => 0u8.put(out),
            Self::Compacted(proof) => {
                1u8.put(out)?;
                proof.put(out)
            }
        }
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        match u8::get(reader)? {
            0 => Ok(Self::Identity),
            1 => Ok(Self::Compacted(Wire::get(reader)?)),
            _ => Err(invalid("invalid atom-filter tag")),
        }
    }
}
impl Wire for RangeOp {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        match self {
            Self::Lt => 0u8,
            Self::Le => 1,
            Self::Gt => 2,
            Self::Ge => 3,
        }
        .put(out)
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        match u8::get(reader)? {
            0 => Ok(Self::Lt),
            1 => Ok(Self::Le),
            2 => Ok(Self::Gt),
            3 => Ok(Self::Ge),
            _ => Err(invalid("invalid range-op tag")),
        }
    }
}
impl Wire for ObjectKindFilter {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        match self {
            Self::Any => 0u8,
            Self::Node => 1,
            Self::Literal => 2,
        }
        .put(out)
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        match u8::get(reader)? {
            0 => Ok(Self::Any),
            1 => Ok(Self::Node),
            2 => Ok(Self::Literal),
            _ => Err(invalid("invalid object-kind filter tag")),
        }
    }
}
impl Wire for AtomTerm {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        match self {
            Self::Variable => 0u8.put(out),
            Self::NodeConstant(bytes) => {
                1u8.put(out)?;
                bytes.put(out)
            }
            Self::ValueConstant(datatype, bytes) => {
                2u8.put(out)?;
                (*datatype as u8).put(out)?;
                bytes.put(out)
            }
        }
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        match u8::get(reader)? {
            0 => Ok(Self::Variable),
            1 => Ok(Self::NodeConstant(Wire::get(reader)?)),
            2 => Ok(Self::ValueConstant(
                CanonicalDatatype::try_from(u8::get(reader)?)?,
                Wire::get(reader)?,
            )),
            _ => Err(invalid("invalid atom-term tag")),
        }
    }
}
impl Wire for CanonicalObject {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        match self {
            Self::Node(bytes) => {
                0u8.put(out)?;
                bytes.put(out)
            }
            Self::Value { datatype, lexical } => {
                1u8.put(out)?;
                (*datatype as u8).put(out)?;
                lexical.put(out)
            }
        }
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        match u8::get(reader)? {
            0 => Ok(Self::Node(Wire::get(reader)?)),
            1 => Ok(Self::Value {
                datatype: CanonicalDatatype::try_from(u8::get(reader)?)?,
                lexical: Wire::get(reader)?,
            }),
            _ => Err(invalid("invalid canonical-object tag")),
        }
    }
}
impl Wire for ResultIdNamespace {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        match self {
            Self::NodeValue => 0u8.put(out),
            Self::Predicate => 1u8.put(out),
            Self::GeneratedDecimalU64 => 2u8.put(out),
            Self::GeneratedDecimalI64 => 3u8.put(out),
        }
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        match u8::get(reader)? {
            0 => Ok(Self::NodeValue),
            1 => Ok(Self::Predicate),
            2 => Ok(Self::GeneratedDecimalU64),
            3 => Ok(Self::GeneratedDecimalI64),
            _ => Err(invalid("invalid result namespace tag")),
        }
    }
}
wire_struct!(ResultColumnDescriptor {
    namespace,
    nullable
});
impl Wire for CanonicalResultValue {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        match self {
            Self::Object(object) => {
                0u8.put(out)?;
                object.put(out)
            }
            Self::Null => 1u8.put(out),
            Self::GeneratedDecimalU64(value) => {
                2u8.put(out)?;
                value.put(out)
            }
            Self::GeneratedDecimalI64(value) => {
                3u8.put(out)?;
                value.put(out)
            }
        }
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        match u8::get(reader)? {
            0 => Ok(Self::Object(Wire::get(reader)?)),
            1 => Ok(Self::Null),
            2 => Ok(Self::GeneratedDecimalU64(u64::get(reader)?)),
            3 => Ok(Self::GeneratedDecimalI64(i64::get(reader)?)),
            _ => Err(invalid("invalid canonical result value tag")),
        }
    }
}

/// Stable identity of the variable-blind public plan. Callers independently compile
/// their query and verification rejects a different plan before opening the proof.
pub fn plan_digest(plan: &LayerBgpPlan) -> io::Result<ProofHash> {
    let mut bytes = Vec::new();
    plan.put(&mut bytes)?;
    let mut hasher = Sha256::new();
    hasher.update(b"terminus-store/query-proof-plan/v2");
    hasher.update(&bytes);
    Ok(hasher.finalize().into())
}
