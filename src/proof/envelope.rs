//! Versioned, bounded wire envelope for an explicitly generated layer BGP proof.

use std::convert::{TryFrom, TryInto};
use std::io::{self, Cursor};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_ff::PrimeField;
use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::transcript::Transcript;
use sha2::{Digest, Sha256};

use super::anti::{
    prove_global_left_outer_join_with_context, prove_global_not_with_context,
    prove_inner_join_with_keys_and_context, prove_left_anti_join_with_keys_and_context, prove_left_outer_join_with_keys_and_context,
    verify_global_left_outer_join, verify_global_not, verify_left_anti_join_with_keys,
    verify_inner_join_with_keys, verify_left_outer_join_with_keys, InnerJoinProof, InnerJoinResult,
    LeftAntiJoinProof, LeftOuterJoinProof,
};
use super::argument::{commit_column, ColumnOpening, StoreProverContext, SumcheckProof, Transcript254};
use super::bgp::{BgpChainProof, BgpStageProof, JoinStep, RelationCommit, RelationWitness};
use super::commitment::{LayerCommitment, VerifierLayer};
use super::count::CountProof;
use super::dict::DictProof;
use super::dicthash::{
    prove_dict_hash_lookup, verify_dict_hash_lookup, DictHashLookupProof, HashLookupInnerProof,
};
use super::semijoin::{
    prove_hash_semijoin, verify_hash_semijoin, HashSemiJoinProof, SelMaskProof,
};
use super::distinct::{
    prove_relation_distinct_with_context, verify_relation_distinct,
    RelationDistinctEntryProof, RelationDistinctProof, RelationDistinctResult,
};
use super::disequality::{
    prove_tuple_disequality_with_context, verify_tuple_disequality, DisequalityNode,
    DisequalityPair, TupleDisequalityProof, TupleDisequalityStageProof,
};
use super::equality::{
    prove_relation_equality_with_context, verify_relation_equality, RelationEqualityConstraint,
    RelationEqualityProof,
};
use super::filter::{
    AtomFilterProof, BooleanProductProof, CompactionOnlyProof, FilterProof, FullFilterProof,
    SelectionOnlyProof,
};
use super::group::{GroupExtremaEntryProof, GroupExtremaKind, GroupExtremaProof};
use super::join2::{BinaryJoinCommitments, BinaryJoinProof, BinaryPairsProof};
use super::membership::MembershipProof;
use super::multiorder::{prove_exact_multi_order_with_context, verify_exact_multi_order, ExactMultiOrderProof};
use super::lexsort::LexicographicSortedProof;
use super::order::{
    prove_exact_scalar_order_with_context, verify_exact_scalar_order, ExactScalarOrderProof,
    OrderDirection,
};
use super::permutation::RelationPermutationProof;
use super::query::{
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
use super::range::{
    DictLookupProof, ObjectRangeCombineProof, ObjectRangeProof, RangeOp, RangeSelectProof,
};
use super::union::{prove_bag_union_with_context, BagUnionResult};
use super::relocate::{prove_column_relocation_with_context, verify_column_relocation, RelocationProof};
use super::sort::SortedProof;
use super::union::{verify_bag_union, BagUnionProof};
use super::{CanonicalDatatype, CanonicalObject, ProofHash};

const MAGIC: &[u8; 12] = b"TDBWQPROOF\0\0";
const UNION_MAGIC: &[u8; 12] = b"TDBWQUNION\0\0";
const ANTI_MAGIC: &[u8; 12] = b"TDBWQANTI\0\0\0";
const OUTER_MAGIC: &[u8; 12] = b"TDBWQOUTER\0\0";
const RECURSIVE_MAGIC: &[u8; 12] = b"TDBWQTREE\0\0\0";

/// Canonical hidden column for a zero-column relation: one on every real row and zero
/// on commitment padding. This is the same representation used by all-ground BGPs.
fn ground_unit_column(n: usize, log_domain: usize) -> Vec<ArkFr> {
    let mut column = vec![ArkFr::zero(); 1usize << log_domain];
    column[..n].fill(ArkFr::one());
    column
}
// v5 adds the explicit transcript/wire-bound object-kind filter. Older envelopes fail
// explicitly rather than guessing whether a variable object meant any value, a node,
// or a literal.
const VERSION: u32 = 5;
const UNION_VERSION: u32 = 3;
const ANTI_VERSION: u32 = 5;
const OUTER_VERSION: u32 = 2;
// v21 generalizes grouped extrema to zero through two canonical node keys.
// v22 adds the bounded transitive-closure (`star`/`plus`) composite node.
const RECURSIVE_VERSION: u32 = 22;
const CHECKSUM_BYTES: usize = 32;
const MAX_ITEMS: usize = 4_000_000;
const MAX_BLOB_BYTES: usize = 64 * 1024 * 1024;
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

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
    decoded_items: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, len: usize) -> io::Result<&'a [u8]> {
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

    fn finish(self) -> io::Result<()> {
        if self.pos != self.bytes.len() {
            return Err(invalid("trailing bytes in query-proof envelope"));
        }
        Ok(())
    }

    fn claim_items(&mut self, len: usize) -> io::Result<()> {
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

trait Wire: Sized {
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

fn put_blob(bytes: &[u8], out: &mut Vec<u8>) -> io::Result<()> {
    if bytes.len() > MAX_BLOB_BYTES {
        return Err(invalid("envelope blob too large"));
    }
    bytes.len().put(out)?;
    out.extend_from_slice(bytes);
    Ok(())
}
fn get_blob<'a>(reader: &mut Reader<'a>) -> io::Result<&'a [u8]> {
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
            expected_root,
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
            resolve_typed_result_objects(
                layer,
                expected_root,
                &result_columns,
                &result_id_namespaces(plan)?,
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
            expected_root,
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
            let resolved = resolve_typed_result_objects(
                layer,
                expected_root,
                &self.result_columns,
                &result_id_namespaces(plan)?,
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
        let mut transcript = super::argument::Transcript254::new(b"parture-union-envelope");
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
            resolve_typed_result_objects(layer, expected_root, &result_columns, &namespaces)?
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
        let mut transcript = super::argument::Transcript254::new(b"parture-union-envelope");
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
                expected_root,
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
        let mut transcript = super::argument::Transcript254::new(b"parture-left-anti-envelope");
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
                expected_root,
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
        let mut transcript = super::argument::Transcript254::new(b"parture-left-anti-envelope");
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
                expected_root,
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

fn outer_descriptors(
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
        let mut transcript = super::argument::Transcript254::new(b"parture-left-outer-envelope");
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
        let mut transcript = super::argument::Transcript254::new(b"parture-left-outer-envelope");
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

/// One authoritative recursive read-statement plan. Operator children are complete
/// authenticated relations, so composition never needs to reinterpret a child proof as
/// a leaf BGP. `query_digest` is the caller-domain identity for this exact node.
#[derive(Clone, Debug)]
pub struct RelationProofPlan {
    pub query_digest: ProofHash,
    pub kind: RelationProofPlanKind,
}

#[derive(Clone, Debug)]
pub enum RelationProofPlanKind {
    Bgp(LayerBgpPlan),
    /// One generated nonnegative xsd:decimal equal to the complete logical triple set
    /// visible at this immutable layer head. The trusted root binds materialized_count.
    LayerTripleCount,
    /// Exact row filter for a generated nonnegative integral decimal column. The child
    /// values are already authenticated and public, so the verifier derives the complete
    /// filtered relation without a separate polynomial proof.
    GeneratedU64Filter {
        input: Box<RelationProofPlan>,
        column: usize,
        op: GeneratedU64ComparisonOp,
        constant: u64,
        constant_on_left: bool,
    },
    /// Exact row filter for a generated signed integral decimal column. Signedness is
    /// selected by the authenticated result descriptor, never inferred from the field.
    GeneratedI64Filter {
        input: Box<RelationProofPlan>,
        column: usize,
        op: GeneratedU64ComparisonOp,
        constant: i64,
        constant_on_left: bool,
    },
    /// Exact zero-/one-key grouping and checked u64/i64 sum over fixed-width integral
    /// Store values.
    /// The child rows and root-bound dictionary are public to the verifier, so the
    /// aggregate is derived directly without exposing the private intermediate list.
    GroupSum {
        input: Box<RelationProofPlan>,
        key_columns: Vec<usize>,
        value_column: usize,
        datatype: CanonicalDatatype,
    },
    /// Exact zero-/one-/two-node-key grouping with the private bag consumed by
    /// Prolog `length/2`. The generated nonnegative decimal is the number of
    /// authenticated child solutions in each group, including duplicates.
    GroupCount {
        input: Box<RelationProofPlan>,
        key_columns: Vec<usize>,
    },
    /// Exact cardinality of the private list produced by Prolog `findall/3` and
    /// consumed immediately by `length/2`. Unlike zero-key GroupBy, empty input
    /// emits one generated decimal row containing zero.
    CollectionCount {
        input: Box<RelationProofPlan>,
    },
    /// Exact sum of the private list produced by Prolog `findall/3`, consumed
    /// immediately by `sum_list/2`. Empty input emits one typed generated zero.
    CollectionSum {
        input: Box<RelationProofPlan>,
        template: CollectionSumTemplate,
        datatype: CanonicalDatatype,
    },
    /// Exact min/max of a private list produced by `findall/3`. Empty input
    /// fails like leading Prolog RangeMin/RangeMax and therefore emits no row.
    CollectionExtrema {
        input: Box<RelationProofPlan>,
        value_column: usize,
        datatype: CanonicalDatatype,
        kind: GroupExtremaKind,
    },
    BagUnion {
        branches: Vec<RelationProofPlan>,
        /// Parent output column -> child output column, once per ordered branch.
        mappings: Vec<Vec<usize>>,
    },
    /// Exact bag projection/reorder. Rows, domain, order, and multiplicity are inherited
    /// unchanged from the authenticated child; only a unique subset of columns is kept.
    /// An empty subset converts a non-ground relation to the canonical hidden UNIT
    /// representation of a zero-column relation.
    BagProjection {
        input: Box<RelationProofPlan>,
        indices: Vec<usize>,
    },
    /// Exact unordered set of the selected one or two child columns. This statement
    /// authenticates equivalence classes and multiplicity removal, but not SWI's
    /// first-occurrence output order.
    Distinct {
        input: Box<RelationProofPlan>,
        key_columns: Vec<usize>,
    },
    LeftAnti {
        left: Box<RelationProofPlan>,
        right: Box<RelationProofPlan>,
        key_pairs: Vec<(usize, usize)>,
    },
    InnerJoin {
        left: Box<RelationProofPlan>,
        right: Box<RelationProofPlan>,
        key_pairs: Vec<(usize, usize)>,
    },
    /// CROSS-DICTIONARY two-root hash join — the ONE plan node that reads TWO committed
    /// graphs (INSTANCE + SCHEMA) at once. `left` is proven against the SCHEMA root, `right`
    /// against the INSTANCE root; each side's join-key id column is resolved to an
    /// authenticated `hash_object(IRI)` column (via [`super::dicthash::prove_dict_hash_lookup`])
    /// against ITS OWN root, and the two hash columns are joined by the UNCHANGED audited
    /// [`super::anti::prove_inner_join_with_keys`]. Because `hash_object` is deterministic AND
    /// dictionary-INDEPENDENT, the SAME IRI hashes identically in both graphs, so the hash join
    /// reconciles the two separate dictionaries correctly; a schema-dict id and an instance-dict
    /// id that are equal INTEGERS but different IRIs have DIFFERENT hashes and never match (no
    /// raw cross-dict id comparison anywhere). Both per-graph roots are bound to the trusted
    /// [`super::commitment::database_root`] before either is trusted as an anchor.
    ///
    /// This node is TOP-LEVEL ONLY: it is proven/verified through the dedicated two-layer entries
    /// [`RecursiveRelationEnvelope::prove_cross_root`] / [`RecursiveRelationEnvelope::verify_cross_root`];
    /// the single-layer [`RecursiveRelationEnvelope::prove_bounded`] recursion rejects it closed
    /// (it has only one layer), so every EXISTING single-root plan is byte-identical. `left` is
    /// specialized to a SINGLE key column (the class set, `left_key_col == 0`, delivering nothing to
    /// the output); the output schema is `right`'s columns MINUS `right_key_col` (instance namespace).
    CrossRootHashJoin {
        /// The SCHEMA-graph child (e.g. the downward `sys:inherits` subsumption closure of a class):
        /// a single node/value id column that becomes the join membership set.
        left: Box<RelationProofPlan>,
        /// The INSTANCE-graph child (e.g. `Triple(?x, rdf:type, ?class)`): its `right_key_col` is the
        /// class id column; the OTHER columns (e.g. `?x`) are delivered as the output.
        right: Box<RelationProofPlan>,
        /// The single join-key column of `left` (must be `0`; `left` is single-column).
        left_key_col: usize,
        /// The join-key (class) column of `right`; dropped from the output.
        right_key_col: usize,
    },
    LeftOuter {
        left: Box<RelationProofPlan>,
        right: Box<RelationProofPlan>,
        key_pairs: Vec<(usize, usize)>,
    },
    Equality {
        input: Box<RelationProofPlan>,
        constraint: RelationEqualityPlan,
    },
    /// Exact EDGE-SIMPLE row filter for a finite single-predicate repeated-predicate
    /// property path. Drops the rows in which some hop-pair reuses a stored edge (the
    /// only over-count of a conjunctive bag-join versus Prolog's edge-simple path
    /// semantics); keeps every node-repeating solution. One committed filter runs per
    /// `pairs` entry (a hop-pair), each carrying all columns through, so the schema is
    /// inherited unchanged (the endpoint projection is a separate `BagProjection`).
    ExactTupleDisequality {
        input: Box<RelationProofPlan>,
        pairs: Vec<TupleDisequalityPairPlan>,
    },
    ExactScalarOrder {
        input: Box<RelationProofPlan>,
        datatype: CanonicalDatatype,
        direction: OrderDirection,
    },
    ExactMultiOrder {
        input: Box<RelationProofPlan>,
        datatypes: Vec<CanonicalDatatype>,
        key_indices: Vec<usize>,
        directions: Vec<OrderDirection>,
    },
    /// Exact contiguous slice of an authenticated ordered relation. `None` returns the
    /// complete suffix after `offset`; `Some(limit)` returns at most that many rows.
    ExactOrderedSlice {
        input: Box<RelationProofPlan>,
        offset: u64,
        limit: Option<u64>,
    },
    /// Exact zero-/one-/two-node-key `GroupBy`, with its private bag consumed
    /// immediately by `RangeMin`/`RangeMax`. Declared key priority determines
    /// canonical expanded-IRI tuple order; the value remains an original Store ID.
    GroupExtrema {
        input: Box<RelationProofPlan>,
        key_columns: Vec<usize>,
        value_column: usize,
        datatype: CanonicalDatatype,
        kind: GroupExtremaKind,
    },
    /// Bounded transitive closure (`star`/`plus`) of a SINGLE predicate from a BOUND source,
    /// delivering the exact reachable SET of endpoint nodes as one nonnullable NodeValue column.
    ///
    /// This is a COMPOSITE ORCHESTRATION over the audited `InnerJoin`/`BagUnion`/`Distinct`/`LeftAnti`
    /// operators — NO new cryptographic gadget. The plan node is `K`-INDEPENDENT: it carries only the
    /// closure PATTERN (edge relation, seed node, star-vs-plus). The number of semi-naive rounds `K`
    /// lives in the PROOF, not the plan, so the verifier reconstructs this identity from its own trusted
    /// AST without knowing `K`, then reads `K` from the delivered proof and validates each round.
    ///
    /// Faithful to `path.pl` (`star(P)` = zero-or-more, `plus(P)` = one-or-more) ONLY as a SET closure,
    /// which equals the reference's edge-simple BAG of endpoints exactly when the surrounding query
    /// collapses the bag to a set (an endpoint `Distinct`/`Count`). The compiler therefore emits this
    /// node ONLY under such a gate; bare `star`/`plus` rejects closed.
    TransitiveClosure {
        /// The edge relation `E = Triple(?s, edge_predicate, ?o)`, a 2-column [subject, object] BGP
        /// (both NodeValue). Proven ONCE and reused by commitment across every round and the fixpoint.
        edge: LayerBgpPlan,
        /// The bound seed node (a node IRI); resolved to its dictionary id at prove/verify.
        seed: CanonicalObject,
        /// `true` = `star` (the identity branch `X = Y` unconditionally succeeds, so the seed itself is
        /// in the set); `false` = `plus` (one-or-more hops, so the seed is present only via a cycle).
        reflexive: bool,
        /// Traversal direction over the [subject, object] edge. `false` walks `subject -> object`
        /// (join `A_k.node == E.subject`, take `E.object`); `true` walks the INVERSE edge
        /// `object -> subject` (join `A_k.node == E.object`, take `E.subject`) — this is how a bound
        /// TARGET (or an inverse-predicate closure) is mirrored, `path.pl:114-134`. The ONLY thing it
        /// changes is the InnerJoin key column on `E`; every other operator is direction-agnostic.
        inverted: bool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeneratedU64ComparisonOp {
    Eq,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollectionSumTemplate {
    Column(usize),
    UnsignedConstant(u64),
    SignedConstant(i64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EqualityValueDomain {
    Predicate,
    Node,
    Datatype(CanonicalDatatype),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RelationEqualityPlan {
    ColumnConstant {
        column: usize,
        domain: EqualityValueDomain,
        value: CanonicalObject,
        constant_on_left: bool,
    },
    Columns {
        left: usize,
        right: usize,
        domain: EqualityValueDomain,
    },
    Constants {
        left: CanonicalObject,
        right: CanonicalObject,
    },
}

/// One node position in a tuple-disequality edge fingerprint: a committed input column
/// of the child relation, or a public constant node IRI (a bound path endpoint),
/// resolved through the trusted layer dictionary at prove/verify time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TupleDisequalityNode {
    Column(usize),
    Constant(CanonicalObject),
}

/// One hop-pair disequality: drop the rows where stored edge `A = (a0, a1)` equals
/// stored edge `B = (b0, b1)` under the (single, fixed) path predicate. `a0`/`b0` are
/// the two hops' SUBJECT positions and `a1`/`b1` their OBJECT positions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TupleDisequalityPairPlan {
    pub a0: TupleDisequalityNode,
    pub a1: TupleDisequalityNode,
    pub b0: TupleDisequalityNode,
    pub b1: TupleDisequalityNode,
}

fn numeric_datatype(datatype: CanonicalDatatype) -> bool {
    matches!(
        datatype,
        CanonicalDatatype::UInt8
            | CanonicalDatatype::Int8
            | CanonicalDatatype::UInt16
            | CanonicalDatatype::Int16
            | CanonicalDatatype::UInt32
            | CanonicalDatatype::Int32
            | CanonicalDatatype::Float32
            | CanonicalDatatype::UInt64
            | CanonicalDatatype::Int64
            | CanonicalDatatype::Float64
            | CanonicalDatatype::Decimal
            | CanonicalDatatype::BigInt
            | CanonicalDatatype::PositiveInteger
            | CanonicalDatatype::NonNegativeInteger
            | CanonicalDatatype::NonPositiveInteger
            | CanonicalDatatype::NegativeInteger
    )
}

/// The fixed-width integral OP datatypes whose canonical field encoding is a total order
/// matching Prolog's integer comparison. Floats are deliberately excluded: `node_value_op_value`
/// accepts `Float32`/`Float64`, but their total-order NaN placement diverges from Prolog's IEEE
/// semantics, so any extrema derivation must reject them.
fn integral_op_datatype(datatype: CanonicalDatatype) -> bool {
    matches!(
        datatype,
        CanonicalDatatype::UInt8
            | CanonicalDatatype::UInt16
            | CanonicalDatatype::UInt32
            | CanonicalDatatype::UInt64
            | CanonicalDatatype::Int8
            | CanonicalDatatype::Int16
            | CanonicalDatatype::Int32
            | CanonicalDatatype::Int64
    )
}

/// A Store dictionary ID namespace whose field values are guaranteed `< 2^64`. Composite
/// two-key joins pack the keys as `col0 + 2^64·col1`, which is injective only for such bounded
/// keys; a generated-decimal namespace can reach ~2^254 and would collide, so it must never be
/// packed.
fn is_store_id_namespace(namespace: ResultIdNamespace) -> bool {
    matches!(
        namespace,
        ResultIdNamespace::NodeValue | ResultIdNamespace::Predicate
    )
}

fn ordered_slice_bounds(source_len: usize, offset: u64, limit: Option<u64>) -> (usize, usize) {
    let start = usize::try_from(offset)
        .unwrap_or(usize::MAX)
        .min(source_len);
    let available = source_len - start;
    let take = limit
        .map(|count| usize::try_from(count).unwrap_or(usize::MAX).min(available))
        .unwrap_or(available);
    (start, start + take)
}

fn validate_equality_plan(
    constraint: &RelationEqualityPlan,
    descriptors: &[ResultColumnDescriptor],
    ground: bool,
) -> io::Result<()> {
    let validate_column = |column: usize, domain: EqualityValueDomain| {
        if ground {
            return Err(invalid("equality column constraint cannot target a ground relation"));
        }
        let descriptor = descriptors
            .get(column)
            .ok_or_else(|| invalid("equality column out of range"))?;
        if descriptor.nullable {
            return Err(invalid("equality cannot target a nullable column"));
        }
        let namespace = match domain {
            EqualityValueDomain::Predicate => ResultIdNamespace::Predicate,
            EqualityValueDomain::Node | EqualityValueDomain::Datatype(_) => {
                ResultIdNamespace::NodeValue
            }
        };
        if descriptor.namespace != namespace {
            return Err(invalid("equality column namespace mismatch"));
        }
        Ok(())
    };
    match constraint {
        RelationEqualityPlan::ColumnConstant {
            column,
            domain,
            value,
            ..
        } => {
            validate_column(*column, *domain)?;
            match (domain, value) {
                (EqualityValueDomain::Node, CanonicalObject::Node(_)) => Ok(()),
                (
                    EqualityValueDomain::Datatype(expected),
                    CanonicalObject::Value { datatype, .. },
                ) if expected == datatype => Ok(()),
                _ => Err(invalid("equality constant does not match its exact value domain")),
            }
        }
        RelationEqualityPlan::Columns {
            left,
            right,
            domain,
        } => {
            validate_column(*left, *domain)?;
            validate_column(*right, *domain)
        }
        RelationEqualityPlan::Constants { left, right } => {
            if let (
                CanonicalObject::Value { datatype: left, .. },
                CanonicalObject::Value {
                    datatype: right, ..
                },
            ) = (left, right)
            {
                if left != right && (numeric_datatype(*left) || numeric_datatype(*right)) {
                    return Err(invalid(
                        "cross-datatype numeric Equals has coercive executor semantics",
                    ));
                }
            }
            Ok(())
        }
    }
}

pub fn resolve_relation_equality_constraint(
    source: &impl VerifierLayer,
    constraint: &RelationEqualityPlan,
) -> io::Result<RelationEqualityConstraint> {
    let layer = source.verifier_view();
    match constraint {
        RelationEqualityPlan::ColumnConstant { column, value, .. } => {
            layer.validate()?;
            let id = layer
                .node_value_catalog
                .iter()
                .find(|record| &record.triple.object == value)
                .map(|record| record.object_id)
                .unwrap_or_else(|| layer.state.node_value_count.saturating_add(1));
            Ok(RelationEqualityConstraint::ColumnConstant {
                column: *column,
                value: ArkFr::from_u64(id),
            })
        }
        RelationEqualityPlan::Columns { left, right, .. } => {
            Ok(RelationEqualityConstraint::Columns {
                left: *left,
                right: *right,
            })
        }
        RelationEqualityPlan::Constants { left, right } => {
            let equal = left == right;
            Ok(RelationEqualityConstraint::Constants {
                left: ArkFr::zero(),
                right: if equal { ArkFr::zero() } else { ArkFr::one() },
            })
        }
    }
}

/// Structural validation for a tuple-disequality plan. The child must be non-ground and
/// every referenced column must be a nonnullable node/value column (edges are compared
/// on their node/value ids); a constant node reference must be a node IRI. Rejects
/// closed everything else, mirroring [`validate_equality_plan`]'s discipline.
fn validate_tuple_disequality_plan(
    pairs: &[TupleDisequalityPairPlan],
    descriptors: &[ResultColumnDescriptor],
    ground: bool,
) -> io::Result<()> {
    if ground {
        return Err(invalid("tuple disequality cannot target a ground relation"));
    }
    if pairs.is_empty() {
        return Err(invalid("tuple disequality needs at least one hop-pair"));
    }
    let check = |node: &TupleDisequalityNode| -> io::Result<()> {
        match node {
            TupleDisequalityNode::Column(column) => {
                let descriptor = descriptors
                    .get(*column)
                    .ok_or_else(|| invalid("tuple disequality column out of range"))?;
                if descriptor.nullable {
                    return Err(invalid("tuple disequality cannot target a nullable column"));
                }
                if descriptor.namespace != ResultIdNamespace::NodeValue {
                    return Err(invalid(
                        "tuple disequality column must be a node/value namespace",
                    ));
                }
                Ok(())
            }
            TupleDisequalityNode::Constant(CanonicalObject::Node(_)) => Ok(()),
            TupleDisequalityNode::Constant(_) => {
                Err(invalid("tuple disequality constant must be a node IRI"))
            }
        }
    };
    for pair in pairs {
        check(&pair.a0)?;
        check(&pair.a1)?;
        check(&pair.b0)?;
        check(&pair.b1)?;
    }
    Ok(())
}

/// Resolve the plan-level hop-pairs (canonical objects) into the field-level pairs the
/// gadget consumes: each column reference is carried verbatim; each constant node IRI is
/// resolved to its `object_id` in the trusted layer dictionary (a not-present constant
/// deterministically maps to `node_value_count + 1` — an impossible id → fail-closed),
/// exactly as [`resolve_relation_equality_constraint`] resolves an equality constant.
pub fn resolve_tuple_disequality_pairs(
    source: &impl VerifierLayer,
    pairs: &[TupleDisequalityPairPlan],
) -> io::Result<Vec<DisequalityPair>> {
    let layer = source.verifier_view();
    layer.validate()?;
    let resolve = |node: &TupleDisequalityNode| -> DisequalityNode {
        match node {
            TupleDisequalityNode::Column(column) => DisequalityNode::Column(*column),
            TupleDisequalityNode::Constant(value) => {
                let id = layer
                    .node_value_catalog
                    .iter()
                    .find(|record| &record.triple.object == value)
                    .map(|record| record.object_id)
                    .unwrap_or_else(|| layer.state.node_value_count.saturating_add(1));
                DisequalityNode::Constant(ArkFr::from_u64(id))
            }
        }
    };
    Ok(pairs
        .iter()
        .map(|pair| DisequalityPair {
            a0: resolve(&pair.a0),
            a1: resolve(&pair.a1),
            b0: resolve(&pair.b0),
            b1: resolve(&pair.b1),
        })
        .collect())
}

impl RelationProofPlan {
    fn shape_bounded(
        &self,
        depth: usize,
        nodes: &mut usize,
    ) -> io::Result<(Vec<ResultColumnDescriptor>, bool)> {
        if depth > MAX_RELATION_PROOF_DEPTH || *nodes >= MAX_RELATION_PROOF_NODES {
            return Err(invalid("recursive relation plan exceeds structural limits"));
        }
        *nodes += 1;
        match &self.kind {
            RelationProofPlanKind::Bgp(plan) => Ok((
                super::query::result_column_descriptors(plan)?,
                has_ground_result_unit(plan),
            )),
            RelationProofPlanKind::LayerTripleCount => Ok((
                vec![ResultColumnDescriptor {
                    namespace: ResultIdNamespace::GeneratedDecimalU64,
                    nullable: false,
                }],
                false,
            )),
            RelationProofPlanKind::GeneratedU64Filter { input, column, .. } => {
                let (descriptors, ground) = input.shape_bounded(depth + 1, nodes)?;
                let descriptor = descriptors
                    .get(*column)
                    .ok_or_else(|| invalid("generated-u64 filter column out of range"))?;
                if ground
                    || descriptor.nullable
                    || descriptor.namespace != ResultIdNamespace::GeneratedDecimalU64
                {
                    return Err(invalid(
                        "generated-u64 filter requires a nonnullable generated decimal column",
                    ));
                }
                Ok((descriptors, false))
            }
            RelationProofPlanKind::GeneratedI64Filter { input, column, .. } => {
                let (descriptors, ground) = input.shape_bounded(depth + 1, nodes)?;
                let descriptor = descriptors
                    .get(*column)
                    .ok_or_else(|| invalid("generated-i64 filter column out of range"))?;
                if ground
                    || descriptor.nullable
                    || descriptor.namespace != ResultIdNamespace::GeneratedDecimalI64
                {
                    return Err(invalid(
                        "generated-i64 filter requires a nonnullable signed generated decimal column",
                    ));
                }
                Ok((descriptors, false))
            }
            RelationProofPlanKind::GroupSum {
                input,
                key_columns,
                value_column,
                datatype,
            } => {
                let (descriptors, ground) = input.shape_bounded(depth + 1, nodes)?;
                if ground
                    || key_columns.len() > 2
                    || key_columns.iter().any(|key| *key == *value_column)
                    || (key_columns.len() == 2 && key_columns[0] == key_columns[1])
                {
                    return Err(invalid("group-sum requires zero to two unique key columns distinct from its value column"));
                }
                let value = descriptors
                    .get(*value_column)
                    .copied()
                    .ok_or_else(|| invalid("group-sum value column out of range"))?;
                if value.nullable
                    || value.namespace != ResultIdNamespace::NodeValue
                    || !matches!(
                        datatype,
                        CanonicalDatatype::UInt8
                            | CanonicalDatatype::UInt16
                            | CanonicalDatatype::UInt32
                            | CanonicalDatatype::UInt64
                            | CanonicalDatatype::Int8
                            | CanonicalDatatype::Int16
                            | CanonicalDatatype::Int32
                            | CanonicalDatatype::Int64
                    )
                {
                    return Err(invalid(
                        "group-sum requires an optional nonnullable node key and a fixed-width integral value",
                    ));
                }
                let generated = ResultColumnDescriptor {
                    namespace: if matches!(
                        datatype,
                        CanonicalDatatype::Int8
                            | CanonicalDatatype::Int16
                            | CanonicalDatatype::Int32
                            | CanonicalDatatype::Int64
                    ) {
                        ResultIdNamespace::GeneratedDecimalI64
                    } else {
                        ResultIdNamespace::GeneratedDecimalU64
                    },
                    nullable: false,
                };
                let mut result = key_columns
                    .iter()
                    .map(|key_column| {
                        let key = descriptors
                            .get(*key_column)
                            .copied()
                            .ok_or_else(|| invalid("group-sum key column out of range"))?;
                        if key.nullable || key.namespace != ResultIdNamespace::NodeValue {
                            return Err(invalid("group-sum keys must be nonnullable nodes"));
                        }
                        Ok(key)
                    })
                    .collect::<io::Result<Vec<_>>>()?;
                result.push(generated);
                Ok((result, false))
            }
            RelationProofPlanKind::GroupCount { input, key_columns } => {
                let (descriptors, ground) = input.shape_bounded(depth + 1, nodes)?;
                if ground
                    || key_columns.len() > 2
                    || (key_columns.len() == 2 && key_columns[0] == key_columns[1])
                {
                    return Err(invalid("group-count requires zero to two unique key columns"));
                }
                let mut result = key_columns
                    .iter()
                    .map(|key_column| {
                        let key = descriptors
                            .get(*key_column)
                            .copied()
                            .ok_or_else(|| invalid("group-count key column out of range"))?;
                        if key.nullable || key.namespace != ResultIdNamespace::NodeValue {
                            return Err(invalid("group-count keys must be nonnullable nodes"));
                        }
                        Ok(key)
                    })
                    .collect::<io::Result<Vec<_>>>()?;
                result.push(ResultColumnDescriptor {
                    namespace: ResultIdNamespace::GeneratedDecimalU64,
                    nullable: false,
                });
                Ok((result, false))
            }
            RelationProofPlanKind::CollectionCount { input } => {
                input.shape_bounded(depth + 1, nodes)?;
                Ok((
                    vec![ResultColumnDescriptor {
                        namespace: ResultIdNamespace::GeneratedDecimalU64,
                        nullable: false,
                    }],
                    false,
                ))
            }
            RelationProofPlanKind::CollectionSum {
                input,
                template,
                datatype,
            } => {
                let (descriptors, _) = input.shape_bounded(depth + 1, nodes)?;
                let signed = matches!(
                    datatype,
                    CanonicalDatatype::Int8
                        | CanonicalDatatype::Int16
                        | CanonicalDatatype::Int32
                        | CanonicalDatatype::Int64
                );
                let unsigned = matches!(
                    datatype,
                    CanonicalDatatype::UInt8
                        | CanonicalDatatype::UInt16
                        | CanonicalDatatype::UInt32
                        | CanonicalDatatype::UInt64
                );
                if !signed && !unsigned {
                    return Err(invalid("collection-sum datatype is outside the integral subset"));
                }
                match template {
                    CollectionSumTemplate::Column(column) => {
                        let value = descriptors
                            .get(*column)
                            .copied()
                            .ok_or_else(|| invalid("collection-sum value column out of range"))?;
                        if value.nullable || value.namespace != ResultIdNamespace::NodeValue {
                            return Err(invalid(
                                "collection-sum column must be a nonnullable Store value",
                            ));
                        }
                    }
                    CollectionSumTemplate::UnsignedConstant(value)
                        if unsigned
                            && match datatype {
                                CanonicalDatatype::UInt8 => *value <= u64::from(u8::MAX),
                                CanonicalDatatype::UInt16 => *value <= u64::from(u16::MAX),
                                CanonicalDatatype::UInt32 => *value <= u64::from(u32::MAX),
                                CanonicalDatatype::UInt64 => true,
                                _ => false,
                            } => {}
                    CollectionSumTemplate::SignedConstant(value)
                        if signed
                            && match datatype {
                                CanonicalDatatype::Int8 => i8::try_from(*value).is_ok(),
                                CanonicalDatatype::Int16 => i16::try_from(*value).is_ok(),
                                CanonicalDatatype::Int32 => i32::try_from(*value).is_ok(),
                                CanonicalDatatype::Int64 => true,
                                _ => false,
                            } => {}
                    _ => return Err(invalid("collection-sum template/domain mismatch")),
                }
                Ok((
                    vec![ResultColumnDescriptor {
                        namespace: if signed {
                            ResultIdNamespace::GeneratedDecimalI64
                        } else {
                            ResultIdNamespace::GeneratedDecimalU64
                        },
                        nullable: false,
                    }],
                    false,
                ))
            }
            RelationProofPlanKind::CollectionExtrema {
                input,
                value_column,
                datatype,
                ..
            } => {
                let (descriptors, ground) = input.shape_bounded(depth + 1, nodes)?;
                let value = descriptors
                    .get(*value_column)
                    .copied()
                    .ok_or_else(|| invalid("collection-extrema value column out of range"))?;
                if ground
                    || value.nullable
                    || value.namespace != ResultIdNamespace::NodeValue
                    || !matches!(
                        datatype,
                        CanonicalDatatype::UInt8
                            | CanonicalDatatype::UInt16
                            | CanonicalDatatype::UInt32
                            | CanonicalDatatype::UInt64
                            | CanonicalDatatype::Int8
                            | CanonicalDatatype::Int16
                            | CanonicalDatatype::Int32
                            | CanonicalDatatype::Int64
                    )
                {
                    return Err(invalid(
                        "collection-extrema requires one exact nonnullable integral Store value",
                    ));
                }
                Ok((vec![value], false))
            }
            RelationProofPlanKind::BagUnion { branches, mappings } => {
                if branches.len() < 2
                    || branches.len() != mappings.len()
                    || branches.len() > MAX_RELATION_PROOF_NODES - *nodes
                {
                    return Err(invalid("recursive bag-union plan branch mismatch"));
                }
                let mut expected: Option<(Vec<ResultColumnDescriptor>, bool)> = None;
                for (branch, mapping) in branches.iter().zip(mappings) {
                    let (descriptors, ground) = branch.shape_bounded(depth + 1, nodes)?;
                    let mapped = if ground {
                        if !mapping.is_empty() {
                            return Err(invalid("ground union branch mapping must be empty"));
                        }
                        Vec::new()
                    } else {
                        mapping
                            .iter()
                            .map(|index| {
                                descriptors.get(*index).copied().ok_or_else(|| {
                                    invalid("recursive union mapping out of range")
                                })
                            })
                            .collect::<io::Result<Vec<_>>>()?
                    };
                    if let Some((prior, prior_ground)) = &expected {
                        if *prior_ground != ground || prior != &mapped {
                            return Err(invalid("recursive union branch schema mismatch"));
                        }
                    } else {
                        expected = Some((mapped, ground));
                    }
                }
                Ok(expected.expect("at least two branches"))
            }
            RelationProofPlanKind::BagProjection { input, indices } => {
                let (descriptors, ground) = input.shape_bounded(depth + 1, nodes)?;
                let mut seen = Vec::with_capacity(indices.len());
                if ground {
                    return Err(invalid("bag projection requires a non-ground child"));
                }
                if indices.is_empty() {
                    return Ok((Vec::new(), true));
                }
                let projected = indices
                    .iter()
                    .map(|index| {
                        if seen.contains(index) {
                            return Err(invalid("bag projection indices must be unique"));
                        }
                        seen.push(*index);
                        descriptors
                            .get(*index)
                            .copied()
                            .ok_or_else(|| invalid("bag projection index out of range"))
                    })
                    .collect::<io::Result<Vec<_>>>()?;
                Ok((projected, false))
            }
            RelationProofPlanKind::Distinct { input, key_columns } => {
                let (descriptors, ground) = input.shape_bounded(depth + 1, nodes)?;
                if ground || !(1..=2).contains(&key_columns.len()) {
                    return Err(invalid("distinct requires one or two non-ground key columns"));
                }
                let mut seen = Vec::with_capacity(key_columns.len());
                let projected = key_columns
                    .iter()
                    .map(|index| {
                        if seen.contains(index) {
                            return Err(invalid("distinct key columns must be unique"));
                        }
                        seen.push(*index);
                        let descriptor = descriptors
                            .get(*index)
                            .copied()
                            .ok_or_else(|| invalid("distinct key column out of range"))?;
                        if descriptor.nullable {
                            return Err(invalid("distinct key columns must be nonnullable"));
                        }
                        Ok(descriptor)
                    })
                    .collect::<io::Result<Vec<_>>>()?;
                Ok((projected, false))
            }
            RelationProofPlanKind::LeftAnti { left, right, key_pairs } => {
                let left_shape = left.shape_bounded(depth + 1, nodes)?;
                let (right_descriptors, right_ground) =
                    right.shape_bounded(depth + 1, nodes)?;
                if key_pairs.is_empty() {
                    return Ok(left_shape);
                }
                if left_shape.1 || right_ground || !(1..=2).contains(&key_pairs.len()) {
                    return Err(invalid("invalid correlated recursive anti-join shape"));
                }
                for (left_index, right_index) in key_pairs {
                    let l = left_shape.0.get(*left_index).ok_or_else(|| invalid("anti left key out of range"))?;
                    let r = right_descriptors.get(*right_index).ok_or_else(|| invalid("anti right key out of range"))?;
                    if l.namespace != r.namespace || l.nullable || r.nullable {
                        return Err(invalid("anti correlation key descriptor mismatch"));
                    }
                    // A composite (two-key) anti-join packs both keys as `col0 + 2^64·col1`,
                    // injective only for Store dictionary IDs (`< 2^64`). Namespace equality
                    // does not bound the field width, so require every packed key to be a
                    // Store-ID namespace; scalar keys use injective equality and are exempt.
                    if key_pairs.len() == 2
                        && (!is_store_id_namespace(l.namespace) || !is_store_id_namespace(r.namespace))
                    {
                        return Err(invalid("composite anti-join key must be a Store-ID namespace"));
                    }
                }
                Ok(left_shape)
            }
            RelationProofPlanKind::InnerJoin { left, right, key_pairs } => {
                let (mut descriptors, left_ground) = left.shape_bounded(depth + 1, nodes)?;
                let (right_descriptors, right_ground) = right.shape_bounded(depth + 1, nodes)?;
                if left_ground || right_ground || !(1..=2).contains(&key_pairs.len()) {
                    return Err(invalid("invalid recursive inner-join child shape"));
                }
                let mut shared_left = Vec::new();
                let mut shared_right = Vec::new();
                for (left_index, right_index) in key_pairs {
                    let l = descriptors.get(*left_index).ok_or_else(|| invalid("inner left key out of range"))?;
                    let r = right_descriptors.get(*right_index).ok_or_else(|| invalid("inner right key out of range"))?;
                    if l.namespace != r.namespace || l.nullable || r.nullable
                        || shared_left.contains(left_index) || shared_right.contains(right_index) {
                        return Err(invalid("inner correlation key descriptor mismatch"));
                    }
                    // Composite (two-key) inner joins pack `col0 + 2^64·col1`, injective only
                    // for Store dictionary IDs (`< 2^64`); reject any non-Store-ID packed key.
                    if key_pairs.len() == 2
                        && (!is_store_id_namespace(l.namespace) || !is_store_id_namespace(r.namespace))
                    {
                        return Err(invalid("composite inner-join key must be a Store-ID namespace"));
                    }
                    shared_left.push(*left_index); shared_right.push(*right_index);
                }
                descriptors.extend(right_descriptors.into_iter().enumerate().filter_map(
                    |(index, descriptor)| (!shared_right.contains(&index)).then_some(descriptor)
                ));
                Ok((descriptors, false))
            }
            RelationProofPlanKind::CrossRootHashJoin { left, right, left_key_col, right_key_col } => {
                let (left_descriptors, left_ground) = left.shape_bounded(depth + 1, nodes)?;
                let (right_descriptors, right_ground) = right.shape_bounded(depth + 1, nodes)?;
                if left_ground || right_ground {
                    return Err(invalid("cross-root hash join child is ground"));
                }
                // The node is specialized: `left` is the single-column class set (all key), so it
                // contributes NOTHING to the output — only its membership matters.
                if left_descriptors.len() != 1 || *left_key_col != 0 {
                    return Err(invalid(
                        "cross-root hash join left child must be a single key column (the class set)",
                    ));
                }
                let lk = left_descriptors
                    .first()
                    .ok_or_else(|| invalid("cross-root left key out of range"))?;
                let rk = right_descriptors
                    .get(*right_key_col)
                    .ok_or_else(|| invalid("cross-root right key out of range"))?;
                // Both keys resolve to IRIs whose (dictionary-independent) hashes we join on, so both
                // must be nonnullable node/value id columns. Their raw ids live in SEPARATE dicts and
                // are NEVER compared directly — only their authenticated hashes are joined.
                if lk.nullable
                    || rk.nullable
                    || lk.namespace != ResultIdNamespace::NodeValue
                    || rk.namespace != ResultIdNamespace::NodeValue
                {
                    return Err(invalid(
                        "cross-root hash join keys must be nonnullable node/value (IRI) columns",
                    ));
                }
                // Output = the instance-side (`right`) columns minus the class key column.
                let descriptors: Vec<ResultColumnDescriptor> = right_descriptors
                    .into_iter()
                    .enumerate()
                    .filter_map(|(index, descriptor)| (index != *right_key_col).then_some(descriptor))
                    .collect();
                if descriptors.is_empty() {
                    return Err(invalid("cross-root hash join delivers no output column"));
                }
                Ok((descriptors, false))
            }
            RelationProofPlanKind::LeftOuter { left, right, key_pairs } => {
                let (mut descriptors, left_ground) = left.shape_bounded(depth + 1, nodes)?;
                let (right_descriptors, right_ground) = right.shape_bounded(depth + 1, nodes)?;
                if left_ground || right_ground || key_pairs.len() > 2 {
                    return Err(invalid("invalid recursive left-outer child shape"));
                }
                let mut shared_right = Vec::new();
                for (left_index, right_index) in key_pairs {
                    let l = descriptors.get(*left_index).ok_or_else(|| invalid("outer left key out of range"))?;
                    let r = right_descriptors.get(*right_index).ok_or_else(|| invalid("outer right key out of range"))?;
                    if l.namespace != r.namespace
                        || l.nullable
                        || r.nullable
                        || shared_right.contains(right_index)
                    {
                        return Err(invalid("outer correlation key descriptor mismatch"));
                    }
                    // Composite (two-key) left-outer joins pack `col0 + 2^64·col1`, injective
                    // only for Store dictionary IDs (`< 2^64`); reject any non-Store-ID packed key.
                    if key_pairs.len() == 2
                        && (!is_store_id_namespace(l.namespace) || !is_store_id_namespace(r.namespace))
                    {
                        return Err(invalid("composite left-outer key must be a Store-ID namespace"));
                    }
                    shared_right.push(*right_index);
                }
                descriptors.extend(right_descriptors.into_iter().enumerate().filter_map(
                    |(index, descriptor)| {
                        (!shared_right.contains(&index)).then_some(ResultColumnDescriptor {
                            namespace: descriptor.namespace,
                            nullable: true,
                        })
                    },
                ));
                Ok((descriptors, false))
            }
            RelationProofPlanKind::Equality { input, constraint } => {
                let (descriptors, ground) = input.shape_bounded(depth + 1, nodes)?;
                validate_equality_plan(constraint, &descriptors, ground)?;
                Ok((descriptors, ground))
            }
            RelationProofPlanKind::ExactTupleDisequality { input, pairs } => {
                let (descriptors, ground) = input.shape_bounded(depth + 1, nodes)?;
                validate_tuple_disequality_plan(pairs, &descriptors, ground)?;
                // A pure row filter: schema (and non-ground-ness) pass through unchanged.
                Ok((descriptors, ground))
            }
            RelationProofPlanKind::ExactScalarOrder { input, datatype, .. } => {
                let (descriptors, ground) = input.shape_bounded(depth + 1, nodes)?;
                if ground
                    || descriptors.len() != 1
                    || descriptors[0].namespace != ResultIdNamespace::NodeValue
                    || descriptors[0].nullable
                    || !numeric_datatype(*datatype)
                    || matches!(
                        datatype,
                        CanonicalDatatype::Decimal
                            | CanonicalDatatype::BigInt
                            | CanonicalDatatype::PositiveInteger
                            | CanonicalDatatype::NonNegativeInteger
                            | CanonicalDatatype::NonPositiveInteger
                            | CanonicalDatatype::NegativeInteger
                    )
                {
                    return Err(invalid("invalid exact scalar order child shape"));
                }
                Ok((descriptors, false))
            }
            RelationProofPlanKind::ExactMultiOrder { input, datatypes, key_indices, directions } => {
                let (descriptors, ground) = input.shape_bounded(depth + 1, nodes)?;
                let mut sorted = key_indices.clone(); sorted.sort_unstable();
                if ground || !(2..=3).contains(&descriptors.len()) || datatypes.len() != descriptors.len() || directions.len() != descriptors.len() || sorted != (0..descriptors.len()).collect::<Vec<_>>()
                    || descriptors.iter().any(|d| d.nullable || d.namespace != ResultIdNamespace::NodeValue) {
                    return Err(invalid("invalid exact multikey order child shape"));
                }
                Ok((descriptors, false))
            }
            RelationProofPlanKind::ExactOrderedSlice { input, .. } => {
                if !matches!(
                    input.kind,
                    RelationProofPlanKind::ExactScalarOrder { .. }
                        | RelationProofPlanKind::ExactMultiOrder { .. }
                        | RelationProofPlanKind::ExactOrderedSlice { .. }
                ) {
                    return Err(invalid("ordered slice requires an authenticated ordered child"));
                }
                let (descriptors, ground) = input.shape_bounded(depth + 1, nodes)?;
                if ground || descriptors.is_empty() {
                    return Err(invalid("ordered slice requires a non-ground ordered relation"));
                }
                Ok((descriptors, false))
            }
            RelationProofPlanKind::GroupExtrema { input, key_columns, value_column, datatype, .. } => {
                let (descriptors, ground) = input.shape_bounded(depth + 1, nodes)?;
                if ground
                    || key_columns.len() > 2
                    || key_columns.iter().any(|key| *key == *value_column)
                    || (key_columns.len() == 2 && key_columns[0] == key_columns[1])
                {
                    return Err(invalid("invalid group-extrema child shape"));
                }
                let value = descriptors.get(*value_column).copied().ok_or_else(|| invalid("group-extrema value column out of range"))?;
                if value.nullable || value.namespace != ResultIdNamespace::NodeValue {
                    return Err(invalid("group-extrema requires a nonnullable Store value"));
                }
                if !matches!(datatype,
                    CanonicalDatatype::UInt8 | CanonicalDatatype::UInt16
                    | CanonicalDatatype::UInt32 | CanonicalDatatype::UInt64
                    | CanonicalDatatype::Int8 | CanonicalDatatype::Int16
                    | CanonicalDatatype::Int32 | CanonicalDatatype::Int64)
                {
                    return Err(invalid("group-extrema datatype is outside the exact integral subset"));
                }
                let mut result = key_columns.iter().map(|key_column| {
                    let key = descriptors.get(*key_column).copied()
                        .ok_or_else(|| invalid("group-extrema key column out of range"))?;
                    if key.nullable || key.namespace != ResultIdNamespace::NodeValue {
                        return Err(invalid("group-extrema keys must be nonnullable nodes"));
                    }
                    Ok(key)
                }).collect::<io::Result<Vec<_>>>()?;
                result.push(ResultColumnDescriptor {
                    namespace: ResultIdNamespace::NodeValue,
                    nullable: false,
                });
                Ok((result, false))
            }
            RelationProofPlanKind::TransitiveClosure { edge, seed, .. } => {
                // The seed is a bound NODE endpoint; a literal/value seed can never be a graph
                // subject and would never join, so reject it closed rather than deliver an empty
                // set under a mislabeled identity.
                if !matches!(seed, CanonicalObject::Node(_)) {
                    return Err(invalid("transitive closure seed must be a node IRI"));
                }
                // The edge relation must be exactly a 2-column [subject, object] node/value BGP so
                // that (i) `object == subject` joins are namespace-compatible across rounds and
                // (ii) the delivered reachable set is a single node column. Anything else (a filtered
                // BGP, a generated column, a wrong arity) rejects closed here — the shape is
                // K-INDEPENDENT, so this validates identically for the prover and the verifier.
                let edge_descriptors = super::query::result_column_descriptors(edge)?;
                if super::query::has_ground_result_unit(edge)
                    || edge_descriptors.len() != 2
                    || edge_descriptors.iter().any(|descriptor| {
                        descriptor.nullable
                            || descriptor.namespace != ResultIdNamespace::NodeValue
                    })
                {
                    return Err(invalid(
                        "transitive closure edge must be a two-column node/value triple relation",
                    ));
                }
                // The delivered relation is the reachable SET: one nonnullable NodeValue column.
                Ok((
                    vec![ResultColumnDescriptor {
                        namespace: ResultIdNamespace::NodeValue,
                        nullable: false,
                    }],
                    false,
                ))
            }
        }
    }

    pub fn result_shape(&self) -> io::Result<(Vec<ResultColumnDescriptor>, bool)> {
        let mut nodes = 0;
        self.shape_bounded(1, &mut nodes)
    }
}

#[derive(Clone, Debug)]
pub struct RecursiveRelationEnvelope {
    pub layer_root: ProofHash,
    pub query_digest: ProofHash,
    pub proof: RecursiveRelationProof,
    pub result_commitments: Vec<ArkGT>,
    pub result_columns: Vec<Vec<ArkFr>>,
    pub result_log_domain: usize,
    pub result_len: usize,
    pub ground_result: bool,
    pub result_descriptors: Vec<ResultColumnDescriptor>,
    pub result_values: Vec<Vec<CanonicalResultValue>>,
}

#[derive(Clone, Debug)]
pub enum RecursiveRelationProof {
    Bgp(Box<LayerBgpEnvelope>),
    LayerTripleCount,
    GeneratedU64Filter {
        input: Box<RecursiveRelationEnvelope>,
    },
    GeneratedI64Filter {
        input: Box<RecursiveRelationEnvelope>,
    },
    GroupSum {
        input: Box<RecursiveRelationEnvelope>,
    },
    GroupCount {
        input: Box<RecursiveRelationEnvelope>,
    },
    CollectionCount {
        input: Box<RecursiveRelationEnvelope>,
    },
    CollectionSum {
        input: Box<RecursiveRelationEnvelope>,
    },
    CollectionExtrema {
        input: Box<RecursiveRelationEnvelope>,
    },
    BagUnion {
        branches: Vec<RecursiveRelationEnvelope>,
        proof: Box<BagUnionProof>,
    },
    BagProjection {
        input: Box<RecursiveRelationEnvelope>,
    },
    Distinct {
        input: Box<RecursiveRelationEnvelope>,
        proof: Box<RelationDistinctProof>,
    },
    LeftAnti {
        left: Box<RecursiveRelationEnvelope>,
        right: Box<RecursiveRelationEnvelope>,
        proof: Option<Box<LeftAntiJoinProof>>,
    },
    InnerJoin {
        left: Box<RecursiveRelationEnvelope>,
        right: Box<RecursiveRelationEnvelope>,
        proof: Box<InnerJoinProof>,
    },
    /// The two-root cross-dictionary hash join proof. `left`/`right` are the two single-root child
    /// subtrees (schema-rooted and instance-rooted); `left_hash`/`right_hash` authenticate each
    /// side's class-id column to its true `hash_object(IRI)` column against ITS OWN root; the three
    /// relocations lift the closure hash SET, the `?x` column and the class-hash column to one working
    /// domain; `semijoin` is the full-field LogUp SEMI-JOIN whose kept `?x` column is the output.
    CrossRootHashJoin {
        left: Box<RecursiveRelationEnvelope>,
        right: Box<RecursiveRelationEnvelope>,
        left_hash: Box<DictHashLookupProof>,
        right_hash: Box<DictHashLookupProof>,
        /// The three authenticated columns relocated UP to a shared working domain `2^D`: the schema
        /// closure hash SET, the instance `?x` column, and the instance class-hash column.
        relocate_closure: RelocationProof,
        com_closure_at_d: ArkGT,
        relocate_x: RelocationProof,
        com_x_at_d: ArkGT,
        relocate_h: RelocationProof,
        com_h_at_d: ArkGT,
        /// The full-field LogUp SEMI-JOIN (no magnitude sort): keeps instance rows whose class-hash is
        /// a member of the closure hash set. `semijoin.com_x_kept` is the delivered `?x` output.
        semijoin: Box<HashSemiJoinProof>,
    },
    LeftOuter {
        left: Box<RecursiveRelationEnvelope>,
        right: Box<RecursiveRelationEnvelope>,
        proof: Box<LeftOuterJoinProof>,
    },
    Equality {
        input: Box<RecursiveRelationEnvelope>,
        proof: Box<RelationEqualityProof>,
    },
    TupleDisequality {
        input: Box<RecursiveRelationEnvelope>,
        proof: Box<TupleDisequalityProof>,
    },
    ExactScalarOrder {
        input: Box<RecursiveRelationEnvelope>,
        proof: Box<ExactScalarOrderProof>,
    },
    ExactMultiOrder {
        input: Box<RecursiveRelationEnvelope>,
        proof: Box<ExactMultiOrderProof>,
    },
    ExactOrderedSlice {
        input: Box<RecursiveRelationEnvelope>,
    },
    GroupExtrema {
        input: Box<RecursiveRelationEnvelope>,
    },
    /// Bounded semi-naive transitive closure. `edge` is the ONE authenticated edge relation `E`
    /// (proven once, reused by commitment). `seed` establishes `A_0`. `rounds` are the `K`
    /// accumulate-and-dedupe steps (each a well-formed InnerJoin + BagUnion + Distinct against the
    /// SAME `E`). `fixpoint` is the empty-delta anti-join proving `A_K` is closed. The delivered
    /// result of THIS envelope is `A_K` = the reachable set. See [`RelationProofPlanKind::TransitiveClosure`].
    TransitiveClosure {
        edge: Box<RecursiveRelationEnvelope>,
        seed: TransitiveClosureSeed,
        rounds: Vec<TransitiveClosureRound>,
        fixpoint: TransitiveClosureFixpoint,
    },
}

/// How `A_0` (the semi-naive base) is established.
#[derive(Clone, Debug)]
pub enum TransitiveClosureSeed {
    /// `star`: `A_0 = {a}`, the identity singleton. Carries NO operator proof — the verifier
    /// reconstructs the constant `{a}` directly from the trusted seed node (`path.pl:183`: the
    /// identity branch `X = Y` unconditionally succeeds, so `a` needs no authentication beyond being
    /// a resolvable dictionary node).
    Reflexive,
    /// `plus`: `A_0 = Distinct(object(InnerJoin({a}, E)))` — the deduped one-hop out-neighbourhood
    /// of the constant seed `{a}`.
    Successor {
        /// `InnerJoin({a}, E)` on `{a}.node == E.subject`; output schema `[node, object]`.
        inner: InnerJoinProof,
        /// The InnerJoin output commitments `[node, object]` (checked against the matched projection).
        inner_out: Vec<ArkGT>,
        /// `Distinct(object(inner), [0])` proving the delivered `rows` are exactly the neighbour set.
        distinct: RelationDistinctProof,
        /// `A_0` as resolved node ids; the verifier commits these to obtain `A_0`'s commitment.
        rows: Vec<u64>,
    },
}

/// One accumulate-and-dedupe round `A_{k+1} = Distinct(BagUnion(A_k, object(InnerJoin(A_k, E))))`.
/// Chained by COMMITMENT: the verifier feeds `A_k`'s commitment (from the previous round / seed) and
/// `E`'s commitment into the three audited operator verifiers, then commits `next_rows` to obtain
/// `A_{k+1}` for the following round. NOT nested — this is why the unroll stays `O(K)`, never `2^K`.
#[derive(Clone, Debug)]
pub struct TransitiveClosureRound {
    /// `InnerJoin(A_k, E)` on `A_k.node == E.subject`; output schema `[node, object]`.
    pub inner: InnerJoinProof,
    /// The InnerJoin output commitments `[node, object]`.
    pub inner_out: Vec<ArkGT>,
    /// `BagUnion(A_k, object(inner))`; output is a single-column bag.
    pub union: BagUnionProof,
    /// The BagUnion output commitment `[node]`.
    pub union_out: Vec<ArkGT>,
    /// `Distinct(union, [0])` proving `next_rows` are exactly the deduped union.
    pub distinct: RelationDistinctProof,
    /// `A_{k+1}` as resolved node ids; the verifier commits these to obtain `A_{k+1}`.
    pub next_rows: Vec<u64>,
}

/// The empty-delta fixpoint certificate: `LeftAnti(InnerJoin(A_K, E), A_K)` on `object == node` has
/// ZERO rows, i.e. every one-hop neighbour of `A_K` is already in `A_K` (`A_K` is closed under the
/// step). Together with monotonicity (`A_0 ⊆ A_1 ⊆ … ⊆ A_K`, enforced by union+distinct) and
/// soundness (`A_K ⊆` reachable, enforced by the InnerJoin), this proves `A_K` is exactly the least
/// fixpoint = the reachable set. This is what makes a CYCLE terminate: the delta empties even though a
/// bag/frontier iteration would loop forever.
#[derive(Clone, Debug)]
pub struct TransitiveClosureFixpoint {
    /// `InnerJoin(A_K, E)`; output schema `[node, object]`.
    pub inner: InnerJoinProof,
    /// The InnerJoin output commitments `[node, object]`.
    pub inner_out: Vec<ArkGT>,
    /// `LeftAnti(inner, A_K, [(object=1, node=0)])`; the verifier asserts its output length is 0.
    pub anti: LeftAntiJoinProof,
    /// The (canonically empty) anti-join output commitments; arity matches `inner` (2 columns).
    pub anti_out: Vec<ArkGT>,
}

fn generated_u64_comparison(
    value: u64,
    op: GeneratedU64ComparisonOp,
    constant: u64,
    constant_on_left: bool,
) -> bool {
    let (left, right) = if constant_on_left {
        (constant, value)
    } else {
        (value, constant)
    };
    match op {
        GeneratedU64ComparisonOp::Eq => left == right,
        GeneratedU64ComparisonOp::Lt => left < right,
        GeneratedU64ComparisonOp::Le => left <= right,
        GeneratedU64ComparisonOp::Gt => left > right,
        GeneratedU64ComparisonOp::Ge => left >= right,
    }
}

fn generated_i64_comparison(
    value: i64,
    op: GeneratedU64ComparisonOp,
    constant: i64,
    constant_on_left: bool,
) -> bool {
    let (left, right) = if constant_on_left {
        (constant, value)
    } else {
        (value, constant)
    };
    match op {
        GeneratedU64ComparisonOp::Eq => left == right,
        GeneratedU64ComparisonOp::Lt => left < right,
        GeneratedU64ComparisonOp::Le => left <= right,
        GeneratedU64ComparisonOp::Gt => left > right,
        GeneratedU64ComparisonOp::Ge => left >= right,
    }
}

fn generated_u64_filtered_columns(
    input: &RecursiveRelationEnvelope,
    column: usize,
    op: GeneratedU64ComparisonOp,
    constant: u64,
    constant_on_left: bool,
) -> io::Result<Vec<Vec<ArkFr>>> {
    let values = input
        .result_values
        .get(column)
        .ok_or_else(|| invalid("generated-u64 filter value column out of range"))?;
    if values.len() != input.result_len {
        return Err(invalid("generated-u64 filter input value length mismatch"));
    }
    let mut keep = Vec::new();
    for (row, value) in values.iter().enumerate() {
        let CanonicalResultValue::GeneratedDecimalU64(value) = value else {
            return Err(invalid("generated-u64 filter input is not a generated decimal"));
        };
        if generated_u64_comparison(*value, op, constant, constant_on_left) {
            keep.push(row);
        }
    }
    input
        .result_columns
        .iter()
        .map(|input_column| {
            keep.iter()
                .map(|row| {
                    input_column
                        .get(*row)
                        .copied()
                        .ok_or_else(|| invalid("generated-u64 filter input column is short"))
                })
                .collect()
        })
        .collect()
}

fn generated_i64_filtered_columns(
    input: &RecursiveRelationEnvelope,
    column: usize,
    op: GeneratedU64ComparisonOp,
    constant: i64,
    constant_on_left: bool,
) -> io::Result<Vec<Vec<ArkFr>>> {
    let values = input
        .result_values
        .get(column)
        .ok_or_else(|| invalid("generated-i64 filter value column out of range"))?;
    if values.len() != input.result_len {
        return Err(invalid("generated-i64 filter input value length mismatch"));
    }
    let mut keep = Vec::new();
    for (row, value) in values.iter().enumerate() {
        let CanonicalResultValue::GeneratedDecimalI64(value) = value else {
            return Err(invalid(
                "generated-i64 filter input is not a signed generated decimal",
            ));
        };
        if generated_i64_comparison(*value, op, constant, constant_on_left) {
            keep.push(row);
        }
    }
    input
        .result_columns
        .iter()
        .map(|input_column| {
            keep.iter()
                .map(|row| {
                    input_column
                        .get(*row)
                        .copied()
                        .ok_or_else(|| invalid("generated-i64 filter input column is short"))
                })
                .collect()
        })
        .collect()
}

#[derive(Clone, Copy)]
enum GroupSumValue {
    Unsigned(u64),
    // Signed Prolog arithmetic is unbounded. The public envelope caps rows at four
    // million, so i128 can accumulate every possible i64 input without intermediate
    // overflow; only the final result is required to fit the generated i64 domain.
    Signed(i128),
}

impl GroupSumValue {
    fn zero(datatype: CanonicalDatatype) -> io::Result<Self> {
        match datatype {
            CanonicalDatatype::UInt8
            | CanonicalDatatype::UInt16
            | CanonicalDatatype::UInt32
            | CanonicalDatatype::UInt64 => Ok(Self::Unsigned(0)),
            CanonicalDatatype::Int8
            | CanonicalDatatype::Int16
            | CanonicalDatatype::Int32
            | CanonicalDatatype::Int64 => Ok(Self::Signed(0)),
            _ => Err(invalid("group-sum datatype is outside the integral subset")),
        }
    }

    fn checked_add(self, rhs: Self) -> io::Result<Self> {
        match (self, rhs) {
            (Self::Unsigned(left), Self::Unsigned(right)) => left
                .checked_add(right)
                .map(Self::Unsigned)
                .ok_or_else(|| invalid("group-sum unsigned arithmetic exceeds u64")),
            (Self::Signed(left), Self::Signed(right)) => left
                .checked_add(right)
                .map(Self::Signed)
                .ok_or_else(|| invalid("group-sum signed arithmetic exceeds i64")),
            _ => Err(invalid("group-sum mixes signed and unsigned values")),
        }
    }

    fn field(self) -> io::Result<ArkFr> {
        match self {
            Self::Unsigned(value) => Ok(ArkFr::from_u64(value)),
            Self::Signed(value) => Ok(generated_decimal_i64_field(
                i64::try_from(value)
                    .map_err(|_| invalid("group-sum signed result exceeds i64"))?,
            )),
        }
    }
}

fn integral_catalog_value(
    object: &CanonicalObject,
    datatype: CanonicalDatatype,
) -> io::Result<GroupSumValue> {
    let CanonicalObject::Value {
        datatype: actual,
        lexical,
    } = object
    else {
        return Err(invalid("group-sum value is not a typed literal"));
    };
    if *actual != datatype {
        return Err(invalid("group-sum value has the wrong exact datatype"));
    }
    Ok(match datatype {
        CanonicalDatatype::UInt8 if lexical.len() == 1 => GroupSumValue::Unsigned(u64::from(lexical[0])),
        CanonicalDatatype::UInt16 if lexical.len() == 2 => {
            GroupSumValue::Unsigned(u64::from(u16::from_be_bytes(lexical.as_slice().try_into().unwrap())))
        }
        CanonicalDatatype::UInt32 if lexical.len() == 4 => {
            GroupSumValue::Unsigned(u64::from(u32::from_be_bytes(lexical.as_slice().try_into().unwrap())))
        }
        CanonicalDatatype::UInt64 if lexical.len() == 8 => {
            GroupSumValue::Unsigned(u64::from_be_bytes(lexical.as_slice().try_into().unwrap()))
        }
        CanonicalDatatype::Int8 if lexical.len() == 1 => {
            GroupSumValue::Signed(i128::from((lexical[0] ^ 0x80) as i8))
        }
        CanonicalDatatype::Int16 if lexical.len() == 2 => {
            let encoded = u16::from_be_bytes(lexical.as_slice().try_into().unwrap());
            GroupSumValue::Signed(i128::from((encoded ^ (1 << 15)) as i16))
        }
        CanonicalDatatype::Int32 if lexical.len() == 4 => {
            let encoded = u32::from_be_bytes(lexical.as_slice().try_into().unwrap());
            GroupSumValue::Signed(i128::from((encoded ^ (1 << 31)) as i32))
        }
        CanonicalDatatype::Int64 if lexical.len() == 8 => {
            let encoded = u64::from_be_bytes(lexical.as_slice().try_into().unwrap());
            GroupSumValue::Signed(i128::from((encoded ^ (1 << 63)) as i64))
        }
        CanonicalDatatype::UInt8
        | CanonicalDatatype::UInt16
        | CanonicalDatatype::UInt32
        | CanonicalDatatype::UInt64
        | CanonicalDatatype::Int8
        | CanonicalDatatype::Int16
        | CanonicalDatatype::Int32
        | CanonicalDatatype::Int64 => {
            return Err(invalid("group-sum value has a noncanonical lexical width"))
        }
        _ => return Err(invalid("group-sum datatype is outside the integral subset")),
    })
}

fn grouped_node_rows(
    layer: &impl VerifierLayer,
    input: &RecursiveRelationEnvelope,
    key_columns: &[usize],
) -> io::Result<std::collections::BTreeMap<Vec<Vec<u8>>, (Vec<u64>, Vec<usize>)>> {
    if key_columns.is_empty()
        || key_columns.len() > 2
        || (key_columns.len() == 2 && key_columns[0] == key_columns[1])
    {
        return Err(invalid("group key tuple is invalid"));
    }
    let view = layer.verifier_view();
    view.validate()?;
    let keys = key_columns
        .iter()
        .map(|key_column| {
            let descriptor = input
                .result_descriptors
                .get(*key_column)
                .copied()
                .ok_or_else(|| invalid("group key descriptor out of range"))?;
            if descriptor.nullable || descriptor.namespace != ResultIdNamespace::NodeValue {
                return Err(invalid("group key is not a nonnullable node"));
            }
            let column = input
                .result_columns
                .get(*key_column)
                .ok_or_else(|| invalid("group key column out of range"))?;
            if column.len() != input.result_len {
                return Err(invalid("group key length mismatch"));
            }
            Ok(column)
        })
        .collect::<io::Result<Vec<_>>>()?;
    let mut groups = std::collections::BTreeMap::<
        Vec<Vec<u8>>,
        (Vec<u64>, Vec<usize>),
    >::new();
    for row in 0..input.result_len {
        let resolved = keys
            .iter()
            .map(|column| {
                let field = column[row];
                let record = view
                    .node_value_catalog
                    .iter()
                    .find(|record| ArkFr::from_u64(record.object_id) == field)
                    .ok_or_else(|| invalid("group key is absent from the node dictionary"))?;
                let CanonicalObject::Node(iri) = &record.triple.object else {
                    return Err(invalid("group key is not a node IRI"));
                };
                Ok((iri.clone(), record.object_id))
            })
            .collect::<io::Result<Vec<_>>>()?;
        let canonical_key = resolved.iter().map(|(iri, _)| iri.clone()).collect::<Vec<_>>();
        let key_ids = resolved.iter().map(|(_, id)| *id).collect::<Vec<_>>();
        let entry = groups
            .entry(canonical_key)
            .or_insert((key_ids.clone(), Vec::new()));
        if entry.0 != key_ids {
            return Err(invalid("canonical node tuple maps to multiple Store IDs"));
        }
        entry.1.push(row);
    }
    Ok(groups)
}

fn derived_group_sum_columns(
    layer: &impl VerifierLayer,
    input: &RecursiveRelationEnvelope,
    key_columns: &[usize],
    value_column: usize,
    datatype: CanonicalDatatype,
) -> io::Result<Vec<Vec<ArkFr>>> {
    let values = input
        .result_columns
        .get(value_column)
        .ok_or_else(|| invalid("group-sum value column out of range"))?;
    if values.len() != input.result_len {
        return Err(invalid("group-sum input column length mismatch"));
    }
    let view = layer.verifier_view();
    view.validate()?;
    let resolve_value = |field: ArkFr| -> io::Result<GroupSumValue> {
        let record = view
            .node_value_catalog
            .iter()
            .find(|record| ArkFr::from_u64(record.object_id) == field)
            .ok_or_else(|| invalid("group-sum value id is absent from the authenticated dictionary"))?;
        integral_catalog_value(&record.triple.object, datatype)
    };
    if key_columns.is_empty() {
        if input.result_len == 0 {
            return Ok(vec![Vec::new()]);
        }
        let mut sum = GroupSumValue::zero(datatype)?;
        for field in values {
            sum = sum.checked_add(resolve_value(*field)?)?;
        }
        return Ok(vec![vec![sum.field()?]]);
    }
    if key_columns.iter().any(|key| *key == value_column) {
        return Err(invalid("group-sum key tuple is invalid"));
    }
    // SWI bagof orders free-variable tuples lexicographically in the declared
    // `group_by` variable order. Resolve authenticated expanded IRI bytes for every
    // component; never sort by child-schema position or append-order Store IDs.
    let groups = grouped_node_rows(layer, input, key_columns)?;
    let mut output = vec![Vec::with_capacity(groups.len()); key_columns.len() + 1];
    for (key_ids, rows) in groups.values() {
        let mut sum = GroupSumValue::zero(datatype)?;
        for row in rows {
            sum = sum.checked_add(resolve_value(values[*row])?)?;
        }
        for (column, key_id) in output.iter_mut().zip(key_ids) {
            column.push(ArkFr::from_u64(*key_id));
        }
        output[key_columns.len()].push(sum.field()?);
    }
    Ok(output)
}

fn derived_group_count_columns(
    layer: &impl VerifierLayer,
    input: &RecursiveRelationEnvelope,
    key_columns: &[usize],
) -> io::Result<Vec<Vec<ArkFr>>> {
    if key_columns.is_empty() {
        return derived_cardinality_column(input.result_len, false);
    }
    let groups = grouped_node_rows(layer, input, key_columns)?;
    let mut output = vec![Vec::with_capacity(groups.len()); key_columns.len() + 1];
    for (key_ids, rows) in groups.values() {
        for (column, key_id) in output.iter_mut().zip(key_ids) {
            column.push(ArkFr::from_u64(*key_id));
        }
        output[key_columns.len()].push(ArkFr::from_u64(
            u64::try_from(rows.len())
                .map_err(|_| invalid("group-count cardinality exceeds u64"))?,
        ));
    }
    Ok(output)
}

fn derived_cardinality_column(result_len: usize, emit_zero_for_empty: bool) -> io::Result<Vec<Vec<ArkFr>>> {
    if result_len == 0 && !emit_zero_for_empty {
        return Ok(vec![Vec::new()]);
    }
    Ok(vec![vec![ArkFr::from_u64(
        u64::try_from(result_len).map_err(|_| invalid("collection cardinality exceeds u64"))?,
    )]])
}

fn derived_collection_sum_column(
    layer: &impl VerifierLayer,
    input: &RecursiveRelationEnvelope,
    template: CollectionSumTemplate,
    datatype: CanonicalDatatype,
) -> io::Result<Vec<Vec<ArkFr>>> {
    let mut sum = GroupSumValue::zero(datatype)?;
    match template {
        CollectionSumTemplate::Column(column) => {
            let values = input
                .result_columns
                .get(column)
                .ok_or_else(|| invalid("collection-sum value column out of range"))?;
            if values.len() != input.result_len {
                return Err(invalid("collection-sum input column length mismatch"));
            }
            let view = layer.verifier_view();
            view.validate()?;
            for field in values.iter().take(input.result_len) {
                let record = view
                    .node_value_catalog
                    .iter()
                    .find(|record| ArkFr::from_u64(record.object_id) == *field)
                    .ok_or_else(|| {
                        invalid("collection-sum value id is absent from the authenticated dictionary")
                    })?;
                sum = sum.checked_add(integral_catalog_value(&record.triple.object, datatype)?)?;
            }
        }
        CollectionSumTemplate::UnsignedConstant(value) => {
            for _ in 0..input.result_len {
                sum = sum.checked_add(GroupSumValue::Unsigned(value))?;
            }
        }
        CollectionSumTemplate::SignedConstant(value) => {
            for _ in 0..input.result_len {
                sum = sum.checked_add(GroupSumValue::Signed(i128::from(value)))?;
            }
        }
    }
    Ok(vec![vec![sum.field()?]])
}

fn derived_collection_extrema_column(
    layer: &impl VerifierLayer,
    input: &RecursiveRelationEnvelope,
    value_column: usize,
    datatype: CanonicalDatatype,
    kind: GroupExtremaKind,
) -> io::Result<Vec<Vec<ArkFr>>> {
    // Defense-in-depth: `node_value_op_value` also encodes Float32/64, whose total-order NaN
    // placement diverges from Prolog's IEEE comparison. Assert the integral OP subset here so a
    // future loosening of the `shape_bounded` gate cannot silently admit floating extrema.
    if !integral_op_datatype(datatype) {
        return Err(invalid(
            "collection-extrema datatype must be a fixed-width integral OP type",
        ));
    }
    let values = input
        .result_columns
        .get(value_column)
        .ok_or_else(|| invalid("collection-extrema value column out of range"))?;
    if values.len() != input.result_len {
        return Err(invalid("collection-extrema input column length mismatch"));
    }
    let view = layer.verifier_view();
    view.validate()?;
    let mut selected: Option<(ArkFr, ark_bn254::Fr)> = None;
    for field in values.iter().take(input.result_len) {
        let record = view
            .node_value_catalog
            .iter()
            .find(|record| ArkFr::from_u64(record.object_id) == *field)
            .ok_or_else(|| invalid("collection-extrema value is absent from the catalog"))?;
        let CanonicalObject::Value {
            datatype: actual, ..
        } = &record.triple.object
        else {
            return Err(invalid("collection-extrema value is not a typed literal"));
        };
        if *actual != datatype {
            return Err(invalid("collection-extrema value has the wrong exact datatype"));
        }
        let encoding = super::commitment::node_value_op_value(&record.triple.object)?.0;
        let replace = selected.as_ref().is_none_or(|(_, current)| match kind {
            GroupExtremaKind::Min => encoding.into_bigint() < current.into_bigint(),
            GroupExtremaKind::Max => encoding.into_bigint() > current.into_bigint(),
        });
        if replace {
            selected = Some((*field, encoding));
        }
    }
    Ok(vec![selected.map(|(field, _)| vec![field]).unwrap_or_default()])
}

fn derived_group_extrema_columns(
    layer: &impl VerifierLayer,
    input: &RecursiveRelationEnvelope,
    key_columns: &[usize],
    value_column: usize,
    datatype: CanonicalDatatype,
    kind: GroupExtremaKind,
) -> io::Result<Vec<Vec<ArkFr>>> {
    // Defense-in-depth: reject floating (NaN-divergent) extrema at derivation time, independent
    // of the `shape_bounded` integral gate. `integral_catalog_value` also enforces this per value.
    if !integral_op_datatype(datatype) {
        return Err(invalid(
            "group-extrema datatype must be a fixed-width integral OP type",
        ));
    }
    let values = input
        .result_columns
        .get(value_column)
        .ok_or_else(|| invalid("group-extrema value column out of range"))?;
    if values.len() != input.result_len || key_columns.iter().any(|key| *key == value_column) {
        return Err(invalid("group-extrema input shape mismatch"));
    }
    let view = layer.verifier_view();
    view.validate()?;
    let resolve = |field: ArkFr| -> io::Result<ark_bn254::Fr> {
        let record = view
            .node_value_catalog
            .iter()
            .find(|record| ArkFr::from_u64(record.object_id) == field)
            .ok_or_else(|| invalid("group-extrema value is absent from the catalog"))?;
        // This enforces both the exact integral datatype and its canonical width.
        integral_catalog_value(&record.triple.object, datatype)?;
        Ok(super::commitment::node_value_op_value(&record.triple.object)?.0)
    };
    let choose = |rows: &[usize]| -> io::Result<Option<ArkFr>> {
        let mut selected: Option<(ArkFr, ark_bn254::Fr)> = None;
        for row in rows {
            let field = values[*row];
            let encoding = resolve(field)?;
            let replace = match selected.as_ref() {
                None => true,
                Some((_, current)) => match kind {
                    GroupExtremaKind::Min => encoding.into_bigint() < current.into_bigint(),
                    GroupExtremaKind::Max => encoding.into_bigint() > current.into_bigint(),
                },
            };
            if replace {
                selected = Some((field, encoding));
            }
        }
        Ok(selected.map(|(field, _)| field))
    };
    if key_columns.is_empty() {
        let rows = (0..input.result_len).collect::<Vec<_>>();
        return Ok(vec![choose(&rows)?.into_iter().collect()]);
    }
    let groups = grouped_node_rows(layer, input, key_columns)?;
    let mut output = vec![Vec::with_capacity(groups.len()); key_columns.len() + 1];
    for (key_ids, rows) in groups.values() {
        let selected = choose(rows)?
            .ok_or_else(|| invalid("group-extrema canonical group is unexpectedly empty"))?;
        for (column, key_id) in output.iter_mut().zip(key_ids) {
            column.push(ArkFr::from_u64(*key_id));
        }
        output[key_columns.len()].push(selected);
    }
    Ok(output)
}

// ============================================================================
// Bounded transitive-closure (`star`/`plus`) orchestration helpers.
//
// These reuse the audited InnerJoin/BagUnion/Distinct/LeftAnti operator cores UNCHANGED,
// threading each step by COMMITMENT (never by nesting one round inside the next — that would
// duplicate `A_k` and blow the unroll up to `2^K` nodes). Every operation runs on a fresh
// transcript domain-separated by (root, closure query digest, step index, role), so a proof
// for one step can never be replayed as another.
// ============================================================================

/// The K-independent `edge` BGP child plan `E = Triple(?s, p, ?o)`. The edge child reuses the
/// closure's OWN query digest as its identity tag: it is the closure's single Bgp sub-relation, bound
/// to its concrete plan separately by `plan_digest(edge)` at the leaf, so the shared tag is a harmless
/// identity and — crucially — lets a caller reconstruct the edge envelope from the closure digest with
/// NO per-K or per-closure-internal information (the K-INDEPENDENT reconstruction property).
fn closure_edge_plan(edge: &LayerBgpPlan, closure_digest: ProofHash) -> RelationProofPlan {
    RelationProofPlan {
        query_digest: closure_digest,
        kind: RelationProofPlanKind::Bgp(edge.clone()),
    }
}

/// A fresh transcript for one closure operator call, domain-separated so no sub-proof is fungible
/// with another. Both prover and verifier build this identically.
fn closure_transcript(root: &ProofHash, digest: &ProofHash, step: u64, role: &[u8]) -> Transcript254 {
    let mut transcript = Transcript254::new(b"parture-transitive-closure-envelope");
    transcript.append_bytes(b"pow/tclosure/root", root);
    transcript.append_bytes(b"pow/tclosure/query", digest);
    transcript.append_bytes(b"pow/tclosure/step", &step.to_le_bytes());
    transcript.append_bytes(b"pow/tclosure/role", role);
    transcript
}

/// Resolve the bound seed node `a` to its layer node/value dictionary id, failing CLOSED if the node
/// is absent (a seed that is not a real dictionary node cannot begin an authenticated traversal). Both
/// prover and verifier resolve identically, so the reconstructed `{a}` commitment is fully pinned.
fn resolve_closure_seed_id(
    view: &super::VerifierCommitmentRef<'_>,
    seed: &CanonicalObject,
) -> io::Result<u64> {
    view.node_value_catalog
        .iter()
        .find(|record| &record.triple.object == seed)
        .map(|record| record.object_id)
        .ok_or_else(|| invalid("transitive closure seed node is absent from the layer dictionary"))
}

/// The canonical `{a}` singleton relation commitment: one real row holding the seed id, padded to the
/// domain `2`. Prover and verifier compute this identically (the "constant seed the verifier
/// reconstructs from the AST"); no operator proof authenticates it because the `star` identity branch
/// unconditionally succeeds.
fn closure_seed_commit(seed_id: u64) -> io::Result<RelationCommit> {
    let column = vec![ArkFr::from_u64(seed_id), ArkFr::zero()];
    Ok(RelationCommit {
        commitments: vec![commit_column(&column)?],
        log_domain: 1,
        n: 1,
    })
}

/// The verifier-side commitment of an accumulator SET delivered as public node ids: commit the ids in
/// the delivered order, padded exactly as [`prove_relation_distinct_with_context`] pads its single-key
/// output, so this equals the commitment the prover threaded forward.
fn closure_set_commit(rows: &[u64]) -> io::Result<RelationCommit> {
    let log_domain = rows.len().max(1).next_power_of_two().max(2).trailing_zeros() as usize;
    let mut column: Vec<ArkFr> = rows.iter().map(|id| ArkFr::from_u64(*id)).collect();
    column.resize(1usize << log_domain, ArkFr::zero());
    Ok(RelationCommit {
        commitments: vec![commit_column(&column)?],
        log_domain,
        n: rows.len(),
    })
}

/// Decode an authenticated node/value store id from its canonical field image (the store commits a
/// node id as `ArkFr::from_u64(id)`). Reject a field outside the canonical u64 image rather than
/// truncating.
fn closure_field_node_id(field: ArkFr) -> io::Result<u64> {
    generated_decimal_u64_value(field)
}

/// Wrap flat node ids as single-column distinct rows for [`verify_relation_distinct`].
fn closure_rows(rows: &[u64]) -> Vec<Vec<u64>> {
    rows.iter().map(|id| vec![*id]).collect()
}

/// The deduped node-id SET of a single-column relation witness (prover side), preserving first
/// occurrence order.
fn closure_distinct_ids(witness: &RelationWitness) -> io::Result<Vec<u64>> {
    let column = witness
        .columns
        .first()
        .ok_or_else(|| invalid("transitive closure step relation has no column"))?;
    let mut seen = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    for field in &column[..witness.n] {
        let id = closure_field_node_id(*field)?;
        if seen.insert(id) {
            out.push(id);
        }
    }
    Ok(out)
}

/// The 1-column OBJECT branch of an `InnerJoin(A, E)` result (schema `[node, object]`): column index 1.
fn closure_object_branch(inner: &InnerJoinResult) -> io::Result<RelationWitness> {
    if inner.columns.len() != 2 || inner.commitments.len() != 2 {
        return Err(invalid("transitive closure inner join must yield [node, object]"));
    }
    Ok(RelationWitness {
        columns: vec![inner.columns[1].clone()],
        commitments: vec![inner.commitments[1]],
        log_domain: inner.log_domain,
        n: inner.n,
    })
}

/// The full 2-column `InnerJoin(A, E)` witness (schema `[node, object]`).
fn closure_inner_witness(inner: &InnerJoinResult) -> RelationWitness {
    RelationWitness {
        columns: inner.columns.clone(),
        commitments: inner.commitments.clone(),
        log_domain: inner.log_domain,
        n: inner.n,
    }
}

fn closure_union_witness(union: &BagUnionResult) -> RelationWitness {
    RelationWitness {
        columns: union.columns.clone(),
        commitments: union.commitments.clone(),
        log_domain: union.log_domain,
        n: union.n,
    }
}

fn closure_distinct_witness(distinct: &RelationDistinctResult) -> RelationWitness {
    RelationWitness {
        columns: distinct.columns.clone(),
        commitments: distinct.commitments.clone(),
        log_domain: distinct.log_domain,
        n: distinct.n,
    }
}

impl RecursiveRelationEnvelope {
    fn relation_commit(&self, mapping: Option<&[usize]>) -> io::Result<RelationCommit> {
        let commitments = match mapping {
            Some(mapping) => mapping
                .iter()
                .map(|index| {
                    self.result_commitments
                        .get(*index)
                        .cloned()
                        .ok_or_else(|| invalid("recursive relation mapping out of range"))
                })
                .collect::<io::Result<Vec<_>>>()?,
            None => self.result_commitments.clone(),
        };
        Ok(RelationCommit {
            commitments,
            log_domain: self.result_log_domain,
            n: self.result_len,
        })
    }

    fn relation_witness(&self, mapping: Option<&[usize]>) -> io::Result<RelationWitness> {
        let indexes: Vec<usize> = match mapping {
            Some(mapping) => mapping.to_vec(),
            None => (0..self.result_columns.len()).collect(),
        };
        let columns = indexes
            .iter()
            .map(|index| {
                let mut column = self.result_columns
                    .get(*index)
                    .cloned()
                    .ok_or_else(|| invalid("recursive relation mapping out of range"))?;
                column.resize(1usize << self.result_log_domain, ArkFr::zero());
                Ok(column)
            })
            .collect::<io::Result<Vec<_>>>()?;
        let commitments = indexes
            .iter()
            .map(|index| {
                self.result_commitments
                    .get(*index)
                    .copied()
                    .ok_or_else(|| invalid("recursive relation mapping out of range"))
            })
            .collect::<io::Result<Vec<_>>>()?;
        Ok(RelationWitness {
            columns,
            commitments,
            log_domain: self.result_log_domain,
            n: self.result_len,
        })
    }

    fn finish_proved(
        layer: &LayerCommitment,
        plan: &RelationProofPlan,
        proof: RecursiveRelationProof,
        commitments: Vec<ArkGT>,
        columns: Vec<Vec<ArkFr>>,
        log_domain: usize,
        n: usize,
    ) -> io::Result<Self> {
        let (result_descriptors, ground_result) = plan.result_shape()?;
        if columns.iter().any(|column| column.len() < n) {
            return Err(invalid("recursive prover result column is shorter than its row count"));
        }
        let columns = columns
            .into_iter()
            .map(|column| column[..n].to_vec())
            .collect::<Vec<_>>();
        let result_values = if ground_result {
            Vec::new()
        } else {
            resolve_result_values(layer, layer.state.commitment_root, &columns, &result_descriptors)?
        };
        Ok(Self {
            layer_root: layer.state.commitment_root,
            query_digest: plan.query_digest,
            proof,
            result_commitments: commitments,
            result_columns: columns,
            result_log_domain: log_domain,
            result_len: n,
            ground_result,
            result_descriptors,
            result_values,
        })
    }

    /// Explicitly construct a complete recursive relation proof with caller-owned
    /// prover arithmetic. Child nodes reuse this exact context.
    pub fn prove_with_context(
        context: &StoreProverContext<'_>,
        layer: &LayerCommitment,
        plan: &RelationProofPlan,
    ) -> io::Result<Self> {
        // COMMITMENT DIET (Wave 0): a delta layer's head view / full dictionary is no
        // longer committed, so it is not directly provable (LayerTripleCount, BGP leaves,
        // aggregates all read the materialized head). Reject closed; roll up to a base.
        super::query::require_base_layer_for_query(&layer.state)?;
        plan.result_shape()?;
        let mut nodes = 0;
        Self::prove_bounded(context, layer, plan, 1, &mut nodes)
    }

    /// Explicit Arkworks CPU convenience for recursive proving.
    pub fn prove(layer: &LayerCommitment, plan: &RelationProofPlan) -> io::Result<Self> {
        Self::prove_with_context(&StoreProverContext::cpu(), layer, plan)
    }

    /// TWO-ROOT ENTRY — prove a top-level [`RelationProofPlanKind::CrossRootHashJoin`] over BOTH
    /// committed graphs. `left` is proven against `schema`, `right` against `instance`; each side's
    /// join-key id column is authenticated to its true `hash_object(IRI)` column against ITS OWN root;
    /// the two hash columns are joined by the audited inner join. The delivered relation lives in the
    /// INSTANCE namespace (its output ids are instance node/value ids), so `layer_root` is the instance
    /// root and the result resolves against the instance dictionary.
    ///
    /// The hash reconciliation is a FULL-FIELD LogUp SEMI-JOIN ([`super::semijoin`]) — NOT the audited
    /// magnitude-sort inner join, whose key-universe strict-sort caps keys at 64/128 bits (`hash_object`
    /// keys are ~254-bit and too near the BN254 modulus to soundly bit-reduce). The semi-join is
    /// field-element-native (reciprocal membership; no sort, no bit decomposition). SOUNDNESS: every
    /// delivered `?x` is a real instance row whose class-hash is a MEMBER of the closure set — a
    /// same-integer-id/different-IRI pair (different hashes) never matches. COMPLETENESS CAVEAT: the
    /// membership direction is `kept ⊆ closure`; the honest prover keeps every member, but a
    /// cryptographic no-drop guarantee against a malicious prover is a documented follow-up (see
    /// [`super::semijoin`]).
    pub fn prove_cross_root(
        instance: &LayerCommitment,
        schema: &LayerCommitment,
        plan: &RelationProofPlan,
    ) -> io::Result<Self> {
        Self::prove_cross_root_with_context(&StoreProverContext::cpu(), instance, schema, plan)
    }

    pub fn prove_cross_root_with_context(
        context: &StoreProverContext<'_>,
        instance: &LayerCommitment,
        schema: &LayerCommitment,
        plan: &RelationProofPlan,
    ) -> io::Result<Self> {
        plan.result_shape()?;
        let (left_plan, right_plan, left_key_col, right_key_col) = match &plan.kind {
            RelationProofPlanKind::CrossRootHashJoin { left, right, left_key_col, right_key_col } => {
                (left.as_ref(), right.as_ref(), *left_key_col, *right_key_col)
            }
            _ => return Err(invalid("prove_cross_root requires a CrossRootHashJoin plan")),
        };

        // (1) Prove each child against ITS OWN root (single-root subtrees, standard recursion).
        let mut lnodes = 0usize;
        let left_env = Self::prove_bounded(context, schema, left_plan, 1, &mut lnodes)?;
        let mut rnodes = 0usize;
        let right_env = Self::prove_bounded(context, instance, right_plan, 1, &mut rnodes)?;

        if left_env.result_commitments.len() != 1 || left_key_col != 0 {
            return Err(invalid("cross-root left child must be a single key column (the class set)"));
        }
        // The instance child is exactly `[?x, class]`: one output column plus the class key column.
        if right_env.result_commitments.len() != 2 || right_key_col >= 2 {
            return Err(invalid("cross-root right child must be exactly [output, class-key]"));
        }
        let x_col = 1 - right_key_col;

        let instance_root = instance.state.commitment_root;
        let schema_root = schema.state.commitment_root;
        let db_root = super::commitment::database_root(&instance_root, &schema_root);
        let mut transcript = Transcript254::new(b"parture-cross-root-envelope");
        transcript.append_bytes(b"pow/cross-root/db-root", &db_root);
        transcript.append_bytes(b"pow/cross-root/query", &plan.query_digest);

        // (2) Authenticate each side's key id column -> true IRI-hash column, against its OWN root.
        let leftlog = left_env.result_log_domain;
        let mut left_key = left_env.result_columns[0].clone();
        left_key.resize(1usize << leftlog, ArkFr::zero());
        let (left_hash, v_hash_left) = prove_dict_hash_lookup(
            context, &left_key, &left_env.result_commitments[0], leftlog, left_env.result_len,
            schema, &mut transcript,
        )?;

        let rightlog = right_env.result_log_domain;
        let mut class_key = right_env.result_columns[right_key_col].clone();
        class_key.resize(1usize << rightlog, ArkFr::zero());
        let (right_hash, v_hash_right) = prove_dict_hash_lookup(
            context, &class_key, &right_env.result_commitments[right_key_col], rightlog,
            right_env.result_len, instance, &mut transcript,
        )?;

        // (3) Relocate the three authenticated columns UP to a shared working domain 2^D:
        // the closure hash SET, the instance ?x column, and the instance class-hash column. Each real
        // support lives in [0, real_n) so the UP relocation is lossless; each is bound to its
        // root-authenticated source commitment.
        let d_left = v_hash_left.len().trailing_zeros() as usize;
        let d_right = v_hash_right.len().trailing_zeros() as usize;
        let d_log = d_left.max(d_right);
        let d_size = 1usize << d_log;
        let pad_d = |c: &[ArkFr]| -> Vec<ArkFr> {
            let mut v = c.to_vec();
            v.resize(d_size, ArkFr::zero());
            v
        };
        let closure_at_d = pad_d(&v_hash_left);
        let com_closure_at_d = context.commit_column(&closure_at_d)?;
        let relocate_closure = prove_column_relocation_with_context(
            context, &v_hash_left, &left_hash.lookup.com_v_hash, &closure_at_d, &com_closure_at_d, &mut transcript,
        )?;
        let h_at_d = pad_d(&v_hash_right);
        let com_h_at_d = context.commit_column(&h_at_d)?;
        let relocate_h = prove_column_relocation_with_context(
            context, &v_hash_right, &right_hash.lookup.com_v_hash, &h_at_d, &com_h_at_d, &mut transcript,
        )?;
        let mut x_child = right_env.result_columns[x_col].clone();
        x_child.resize(1usize << rightlog, ArkFr::zero());
        let x_at_d = pad_d(&x_child);
        let com_x_at_d = context.commit_column(&x_at_d)?;
        let relocate_x = prove_column_relocation_with_context(
            context, &x_child, &right_env.result_commitments[x_col], &x_at_d, &com_x_at_d, &mut transcript,
        )?;

        // (4) Full-field LogUp SEMI-JOIN (no magnitude sort): keep instance rows whose class-hash is a
        // member of the closure hash set; deliver the kept ?x column.
        let (semijoin, x_kept, m) = prove_hash_semijoin(
            context, &closure_at_d, &com_closure_at_d, left_env.result_len,
            &x_at_d, &com_x_at_d, &h_at_d, &com_h_at_d, right_env.result_len, &mut transcript,
        )?;

        let com_x_kept = semijoin.com_x_kept;
        Self::finish_proved(
            instance,
            plan,
            RecursiveRelationProof::CrossRootHashJoin {
                left: Box::new(left_env),
                right: Box::new(right_env),
                left_hash: Box::new(left_hash),
                right_hash: Box::new(right_hash),
                relocate_closure,
                com_closure_at_d,
                relocate_x,
                com_x_at_d,
                relocate_h,
                com_h_at_d,
                semijoin: Box::new(semijoin),
            },
            vec![com_x_kept],
            vec![x_kept],
            d_log,
            m,
        )
    }

    fn prove_bounded(
        context: &StoreProverContext<'_>,
        layer: &LayerCommitment,
        plan: &RelationProofPlan,
        depth: usize,
        nodes: &mut usize,
    ) -> io::Result<Self> {
        if depth > MAX_RELATION_PROOF_DEPTH || *nodes >= MAX_RELATION_PROOF_NODES {
            return Err(invalid("recursive relation proving exceeds structural limits"));
        }
        *nodes += 1;
        match &plan.kind {
            RelationProofPlanKind::Bgp(bgp) => {
                let result = prove_layer_bgp_with_result_and_context(context, layer, bgp)?;
                let result_columns = result.real_columns();
                let leaf = LayerBgpEnvelope::new_verified(
                    layer,
                    layer.state.commitment_root,
                    bgp,
                    plan.query_digest,
                    result.proof,
                    result.result_commitments,
                    result_columns,
                    result.result_log_domain,
                    result.result_len,
                    result.predicate_ids,
                )?;
                Self::finish_proved(
                    layer,
                    plan,
                    RecursiveRelationProof::Bgp(Box::new(leaf.clone())),
                    leaf.result_commitments,
                    leaf.result_columns,
                    leaf.result_log_domain,
                    leaf.result_len,
                )
            }
            RelationProofPlanKind::LayerTripleCount => {
                let mut column = vec![ArkFr::zero(); 2];
                column[0] = ArkFr::from_u64(layer.state.materialized_count);
                let commitment = context.commit_column(&column)?;
                Self::finish_proved(
                    layer,
                    plan,
                    RecursiveRelationProof::LayerTripleCount,
                    vec![commitment],
                    vec![column],
                    1,
                    1,
                )
            }
            RelationProofPlanKind::CollectionExtrema {
                input,
                value_column,
                datatype,
                kind,
            } => {
                let input_envelope =
                    Self::prove_bounded(context, layer, input, depth + 1, nodes)?;
                let mut columns = derived_collection_extrema_column(
                    layer,
                    &input_envelope,
                    *value_column,
                    *datatype,
                    *kind,
                )?;
                let n = columns[0].len();
                columns[0].resize(2, ArkFr::zero());
                let commitments = vec![context.commit_column(&columns[0])?];
                Self::finish_proved(
                    layer,
                    plan,
                    RecursiveRelationProof::CollectionExtrema {
                        input: Box::new(input_envelope),
                    },
                    commitments,
                    columns,
                    1,
                    n,
                )
            }
            RelationProofPlanKind::GeneratedU64Filter {
                input,
                column,
                op,
                constant,
                constant_on_left,
            } => {
                let input_envelope =
                    Self::prove_bounded(context, layer, input, depth + 1, nodes)?;
                let mut columns = generated_u64_filtered_columns(
                    &input_envelope,
                    *column,
                    *op,
                    *constant,
                    *constant_on_left,
                )?;
                let n = columns.first().map_or(0, Vec::len);
                let log_domain = n.max(1).next_power_of_two().max(2).trailing_zeros() as usize;
                let domain = 1usize << log_domain;
                for column in &mut columns {
                    column.resize(domain, ArkFr::zero());
                }
                let commitments = columns
                    .iter()
                    .map(|column| context.commit_column(column))
                    .collect::<io::Result<Vec<_>>>()?;
                Self::finish_proved(
                    layer,
                    plan,
                    RecursiveRelationProof::GeneratedU64Filter {
                        input: Box::new(input_envelope),
                    },
                    commitments,
                    columns,
                    log_domain,
                    n,
                )
            }
            RelationProofPlanKind::CollectionSum {
                input,
                template,
                datatype,
            } => {
                let input_envelope =
                    Self::prove_bounded(context, layer, input, depth + 1, nodes)?;
                let mut columns = derived_collection_sum_column(
                    layer,
                    &input_envelope,
                    *template,
                    *datatype,
                )?;
                columns[0].resize(2, ArkFr::zero());
                let commitments = vec![context.commit_column(&columns[0])?];
                Self::finish_proved(
                    layer,
                    plan,
                    RecursiveRelationProof::CollectionSum {
                        input: Box::new(input_envelope),
                    },
                    commitments,
                    columns,
                    1,
                    1,
                )
            }
            RelationProofPlanKind::CollectionCount { input } => {
                let input_envelope =
                    Self::prove_bounded(context, layer, input, depth + 1, nodes)?;
                let mut columns = derived_cardinality_column(input_envelope.result_len, true)?;
                let n = 1;
                let log_domain = 1;
                for column in &mut columns {
                    column.resize(2, ArkFr::zero());
                }
                let commitments = columns
                    .iter()
                    .map(|column| context.commit_column(column))
                    .collect::<io::Result<Vec<_>>>()?;
                Self::finish_proved(
                    layer,
                    plan,
                    RecursiveRelationProof::CollectionCount {
                        input: Box::new(input_envelope),
                    },
                    commitments,
                    columns,
                    log_domain,
                    n,
                )
            }
            RelationProofPlanKind::GeneratedI64Filter {
                input,
                column,
                op,
                constant,
                constant_on_left,
            } => {
                let input_envelope =
                    Self::prove_bounded(context, layer, input, depth + 1, nodes)?;
                let mut columns = generated_i64_filtered_columns(
                    &input_envelope,
                    *column,
                    *op,
                    *constant,
                    *constant_on_left,
                )?;
                let n = columns.first().map_or(0, Vec::len);
                let log_domain = n.max(1).next_power_of_two().max(2).trailing_zeros() as usize;
                let domain = 1usize << log_domain;
                for column in &mut columns {
                    column.resize(domain, ArkFr::zero());
                }
                let commitments = columns
                    .iter()
                    .map(|column| context.commit_column(column))
                    .collect::<io::Result<Vec<_>>>()?;
                Self::finish_proved(
                    layer,
                    plan,
                    RecursiveRelationProof::GeneratedI64Filter {
                        input: Box::new(input_envelope),
                    },
                    commitments,
                    columns,
                    log_domain,
                    n,
                )
            }
            RelationProofPlanKind::GroupSum {
                input,
                key_columns,
                value_column,
                datatype,
            } => {
                let input_envelope =
                    Self::prove_bounded(context, layer, input, depth + 1, nodes)?;
                let mut columns = derived_group_sum_columns(
                    layer,
                    &input_envelope,
                    key_columns,
                    *value_column,
                    *datatype,
                )?;
                let n = columns.first().map_or(0, Vec::len);
                let log_domain = n.max(1).next_power_of_two().max(2).trailing_zeros() as usize;
                let domain = 1usize << log_domain;
                for column in &mut columns {
                    column.resize(domain, ArkFr::zero());
                }
                let commitments = columns
                    .iter()
                    .map(|column| context.commit_column(column))
                    .collect::<io::Result<Vec<_>>>()?;
                Self::finish_proved(
                    layer,
                    plan,
                    RecursiveRelationProof::GroupSum {
                        input: Box::new(input_envelope),
                    },
                    commitments,
                    columns,
                    log_domain,
                    n,
                )
            }
            RelationProofPlanKind::GroupCount { input, key_columns } => {
                let input_envelope =
                    Self::prove_bounded(context, layer, input, depth + 1, nodes)?;
                let mut columns =
                    derived_group_count_columns(layer, &input_envelope, key_columns)?;
                let n = columns.first().map_or(0, Vec::len);
                let log_domain = n.max(1).next_power_of_two().max(2).trailing_zeros() as usize;
                let domain = 1usize << log_domain;
                for column in &mut columns {
                    column.resize(domain, ArkFr::zero());
                }
                let commitments = columns
                    .iter()
                    .map(|column| context.commit_column(column))
                    .collect::<io::Result<Vec<_>>>()?;
                Self::finish_proved(
                    layer,
                    plan,
                    RecursiveRelationProof::GroupCount {
                        input: Box::new(input_envelope),
                    },
                    commitments,
                    columns,
                    log_domain,
                    n,
                )
            }
            RelationProofPlanKind::Distinct { input, key_columns } => {
                let input_envelope =
                    Self::prove_bounded(context, layer, input, depth + 1, nodes)?;
                let witness = input_envelope.relation_witness(None)?;
                let view = layer.verifier_view();
                let resolve_id = |field: ArkFr, namespace: ResultIdNamespace| -> io::Result<u64> {
                    match namespace {
                        ResultIdNamespace::Predicate => view
                            .predicate_catalog
                            .iter()
                            .find(|record| ArkFr::from_u64(record.predicate_id) == field)
                            .map(|record| record.predicate_id),
                        ResultIdNamespace::NodeValue => view
                            .node_value_catalog
                            .iter()
                            .find(|record| ArkFr::from_u64(record.object_id) == field)
                            .map(|record| record.object_id),
                        ResultIdNamespace::GeneratedDecimalU64
                        | ResultIdNamespace::GeneratedDecimalI64 => None,
                    }
                    .ok_or_else(|| invalid("distinct input id absent from planned namespace"))
                };
                let mut seen = std::collections::BTreeSet::new();
                let mut rows = Vec::new();
                for row in 0..witness.n {
                    let tuple = key_columns
                        .iter()
                        .map(|column| {
                            resolve_id(
                                witness.columns[*column][row],
                                input_envelope.result_descriptors[*column].namespace,
                            )
                        })
                        .collect::<io::Result<Vec<_>>>()?;
                    if seen.insert(tuple.clone()) {
                        rows.push(tuple);
                    }
                }
                let mut transcript = Transcript254::new(b"parture-distinct-envelope");
                transcript.append_bytes(b"pow/distinct-envelope/root", &layer.state.commitment_root);
                transcript.append_bytes(b"pow/distinct-envelope/query", &plan.query_digest);
                let result = prove_relation_distinct_with_context(
                    context,
                    &witness,
                    key_columns,
                    &rows,
                    &mut transcript,
                )?;
                Self::finish_proved(
                    layer,
                    plan,
                    RecursiveRelationProof::Distinct {
                        input: Box::new(input_envelope),
                        proof: Box::new(result.proof),
                    },
                    result.commitments,
                    result.columns,
                    result.log_domain,
                    result.n,
                )
            }
            RelationProofPlanKind::BagUnion { branches, mappings } => {
                let mut child_envelopes = Vec::with_capacity(branches.len());
                let mut witnesses = Vec::with_capacity(branches.len());
                let (_, ground) = plan.result_shape()?;
                for ((branch, mapping), _) in branches.iter().zip(mappings).zip(0..) {
                    let child = Self::prove_bounded(context, layer, branch, depth + 1, nodes)?;
                    witnesses.push(child.relation_witness((!ground).then_some(mapping.as_slice()))?);
                    child_envelopes.push(child);
                }
                let mut transcript = Transcript254::new(b"parture-union-envelope");
                transcript.append_bytes(b"pow/union-envelope/root", &layer.state.commitment_root);
                transcript.append_bytes(b"pow/union-envelope/query", &plan.query_digest);
                let result = prove_bag_union_with_context(context, &witnesses, &mut transcript)?;
                Self::finish_proved(layer, plan, RecursiveRelationProof::BagUnion {
                    branches: child_envelopes,
                    proof: Box::new(result.proof),
                }, result.commitments, result.columns, result.log_domain, result.n)
            }
            RelationProofPlanKind::BagProjection { input, indices } => {
                let input_envelope =
                    Self::prove_bounded(context, layer, input, depth + 1, nodes)?;
                let log_domain = input_envelope.result_log_domain;
                let n = input_envelope.result_len;
                let (commitments, columns) = if indices.is_empty() {
                    let unit = ground_unit_column(n, log_domain);
                    (vec![context.commit_column(&unit)?], vec![unit])
                } else {
                    let commitments = indices
                        .iter()
                        .map(|index| {
                            input_envelope
                                .result_commitments
                                .get(*index)
                                .copied()
                                .ok_or_else(|| invalid("bag projection index out of range"))
                        })
                        .collect::<io::Result<Vec<_>>>()?;
                    let columns = indices
                        .iter()
                        .map(|index| {
                            input_envelope
                                .result_columns
                                .get(*index)
                                .cloned()
                                .ok_or_else(|| invalid("bag projection index out of range"))
                        })
                        .collect::<io::Result<Vec<_>>>()?;
                    (commitments, columns)
                };
                Self::finish_proved(
                    layer,
                    plan,
                    RecursiveRelationProof::BagProjection {
                        input: Box::new(input_envelope),
                    },
                    commitments,
                    columns,
                    log_domain,
                    n,
                )
            }
            RelationProofPlanKind::LeftAnti { left, right, key_pairs } => {
                let left_envelope = Self::prove_bounded(context, layer, left, depth + 1, nodes)?;
                let right_envelope = Self::prove_bounded(context, layer, right, depth + 1, nodes)?;
                let left_witness = left_envelope.relation_witness(None)?;
                let right_witness = right_envelope.relation_witness(None)?;
                let (proof, commitments, columns, log_domain, n) = if key_pairs.is_empty() {
                    let (commitments, columns, log_domain, n) =
                        prove_global_not_with_context(context, &left_witness, &right_witness)?;
                    (None, commitments, columns, log_domain, n)
                } else {
                    let mut transcript = Transcript254::new(b"parture-left-anti-envelope");
                    transcript.append_bytes(b"pow/anti-envelope/root", &layer.state.commitment_root);
                    transcript.append_bytes(b"pow/anti-envelope/query", &plan.query_digest);
                    let result = prove_left_anti_join_with_keys_and_context(
                        context, &left_witness, &right_witness, key_pairs, &mut transcript,
                    )?;
                    (Some(Box::new(result.proof)), result.commitments, result.columns, result.log_domain, result.n)
                };
                Self::finish_proved(layer, plan, RecursiveRelationProof::LeftAnti {
                    left: Box::new(left_envelope), right: Box::new(right_envelope), proof,
                }, commitments, columns, log_domain, n)
            }
            RelationProofPlanKind::InnerJoin { left, right, key_pairs } => {
                let left_envelope = Self::prove_bounded(context, layer, left, depth + 1, nodes)?;
                let right_envelope = Self::prove_bounded(context, layer, right, depth + 1, nodes)?;
                let mut transcript = Transcript254::new(b"parture-inner-join-envelope");
                transcript.append_bytes(b"pow/inner-envelope/root", &layer.state.commitment_root);
                transcript.append_bytes(b"pow/inner-envelope/query", &plan.query_digest);
                let result = prove_inner_join_with_keys_and_context(
                    context, &left_envelope.relation_witness(None)?,
                    &right_envelope.relation_witness(None)?, key_pairs, &mut transcript,
                )?;
                Self::finish_proved(layer, plan, RecursiveRelationProof::InnerJoin {
                    left: Box::new(left_envelope), right: Box::new(right_envelope),
                    proof: Box::new(result.proof),
                }, result.commitments, result.columns, result.log_domain, result.n)
            }
            RelationProofPlanKind::CrossRootHashJoin { .. } => {
                // The cross-root node reads TWO graphs; the single-layer recursion has only one
                // layer. It is TOP-LEVEL only, proven via `prove_cross_root` (two-layer). Reject
                // closed here so it can never be nested inside a single-root subtree.
                Err(invalid(
                    "cross-root hash join must be proven via prove_cross_root (two layers), not nested in the single-root recursion",
                ))
            }
            RelationProofPlanKind::LeftOuter { left, right, key_pairs } => {
                let left_envelope = Self::prove_bounded(context, layer, left, depth + 1, nodes)?;
                let right_envelope = Self::prove_bounded(context, layer, right, depth + 1, nodes)?;
                let left_witness = left_envelope.relation_witness(None)?;
                let right_witness = right_envelope.relation_witness(None)?;
                let mut transcript = Transcript254::new(b"parture-left-outer-envelope");
                transcript.append_bytes(b"pow/outer-envelope/root", &layer.state.commitment_root);
                transcript.append_bytes(b"pow/outer-envelope/query", &plan.query_digest);
                let result = if key_pairs.is_empty() {
                    prove_global_left_outer_join_with_context(context, &left_witness, &right_witness, &mut transcript)?
                } else {
                    prove_left_outer_join_with_keys_and_context(context, &left_witness, &right_witness, key_pairs, &mut transcript)?
                };
                Self::finish_proved(layer, plan, RecursiveRelationProof::LeftOuter {
                    left: Box::new(left_envelope), right: Box::new(right_envelope), proof: Box::new(result.proof),
                }, result.commitments, result.columns, result.log_domain, result.n)
            }
            RelationProofPlanKind::Equality { input, constraint } => {
                let input_envelope = Self::prove_bounded(context, layer, input, depth + 1, nodes)?;
                let resolved = resolve_relation_equality_constraint(layer, constraint)?;
                let mut transcript = Transcript254::new(b"parture-equality-envelope");
                transcript.append_bytes(b"pow/equality-envelope/root", &layer.state.commitment_root);
                transcript.append_bytes(b"pow/equality-envelope/query", &plan.query_digest);
                let result = prove_relation_equality_with_context(
                    context, &input_envelope.relation_witness(None)?, &resolved, &mut transcript,
                )?;
                Self::finish_proved(layer, plan, RecursiveRelationProof::Equality {
                    input: Box::new(input_envelope), proof: Box::new(result.proof),
                }, result.commitments, result.columns, result.log_domain, result.n)
            }
            RelationProofPlanKind::ExactTupleDisequality { input, pairs } => {
                let input_envelope = Self::prove_bounded(context, layer, input, depth + 1, nodes)?;
                let resolved = resolve_tuple_disequality_pairs(layer, pairs)?;
                let mut transcript = Transcript254::new(b"parture-tuple-disequality-envelope");
                transcript.append_bytes(b"pow/tuple-disequality-envelope/root", &layer.state.commitment_root);
                transcript.append_bytes(b"pow/tuple-disequality-envelope/query", &plan.query_digest);
                let result = prove_tuple_disequality_with_context(
                    context, &input_envelope.relation_witness(None)?, &resolved, &mut transcript,
                )?;
                Self::finish_proved(layer, plan, RecursiveRelationProof::TupleDisequality {
                    input: Box::new(input_envelope), proof: Box::new(result.proof),
                }, result.commitments, result.columns, result.log_domain, result.n)
            }
            RelationProofPlanKind::ExactScalarOrder { input, datatype, direction } => {
                let input_envelope =
                    Self::prove_bounded(context, layer, input, depth + 1, nodes)?;
                let mut transcript = Transcript254::new(b"parture-order-envelope");
                transcript.append_bytes(b"pow/order-envelope/root", &layer.state.commitment_root);
                transcript.append_bytes(b"pow/order-envelope/query", &plan.query_digest);
                let result = prove_exact_scalar_order_with_context(
                    context,
                    layer,
                    &input_envelope.relation_witness(None)?,
                    *datatype,
                    *direction,
                    &mut transcript,
                )?;
                Self::finish_proved(
                    layer,
                    plan,
                    RecursiveRelationProof::ExactScalarOrder {
                        input: Box::new(input_envelope),
                        proof: Box::new(result.proof),
                    },
                    result.commitments,
                    result.columns,
                    result.log_domain,
                    result.n,
                )
            }
            RelationProofPlanKind::ExactMultiOrder { input, datatypes, key_indices, directions } => {
                let input_envelope = Self::prove_bounded(context, layer, input, depth + 1, nodes)?;
                let mut transcript = Transcript254::new(b"parture-multi-order-envelope");
                transcript.append_bytes(b"pow/multi-order-envelope/root", &layer.state.commitment_root);
                transcript.append_bytes(b"pow/multi-order-envelope/query", &plan.query_digest);
                let result = prove_exact_multi_order_with_context(context, layer, &input_envelope.relation_witness(None)?, datatypes, key_indices, directions, &mut transcript)?;
                Self::finish_proved(layer, plan, RecursiveRelationProof::ExactMultiOrder { input: Box::new(input_envelope), proof: Box::new(result.proof) }, result.commitments, result.columns, result.log_domain, result.n)
            }
            RelationProofPlanKind::ExactOrderedSlice {
                input,
                offset,
                limit,
            } => {
                let input_envelope =
                    Self::prove_bounded(context, layer, input, depth + 1, nodes)?;
                let (start, end) = ordered_slice_bounds(input_envelope.result_len, *offset, *limit);
                let take = end - start;
                let log_domain = take.max(1).next_power_of_two().max(2).trailing_zeros() as usize;
                let domain = 1usize << log_domain;
                let mut columns = input_envelope
                    .result_columns
                    .iter()
                    .map(|column| column[start..end].to_vec())
                    .collect::<Vec<_>>();
                for column in &mut columns {
                    column.resize(domain, ArkFr::zero());
                }
                let commitments = columns
                    .iter()
                    .map(|column| context.commit_column(column))
                    .collect::<io::Result<Vec<_>>>()?;
                Self::finish_proved(
                    layer,
                    plan,
                    RecursiveRelationProof::ExactOrderedSlice {
                        input: Box::new(input_envelope),
                    },
                    commitments,
                    columns,
                    log_domain,
                    take,
                )
            }
            RelationProofPlanKind::GroupExtrema { input, key_columns, value_column, datatype, kind } => {
                let input_envelope = Self::prove_bounded(context, layer, input, depth + 1, nodes)?;
                let mut columns = derived_group_extrema_columns(
                    layer, &input_envelope, key_columns, *value_column, *datatype, *kind,
                )?;
                let n = columns.first().map_or(0, Vec::len);
                let log_domain = n.max(1).next_power_of_two().max(2).trailing_zeros() as usize;
                let domain = 1usize << log_domain;
                for column in &mut columns {
                    column.resize(domain, ArkFr::zero());
                }
                let commitments = columns.iter()
                    .map(|column| context.commit_column(column))
                    .collect::<io::Result<Vec<_>>>()?;
                Self::finish_proved(layer, plan, RecursiveRelationProof::GroupExtrema {
                    input: Box::new(input_envelope),
                }, commitments, columns, log_domain, n)
            }
            RelationProofPlanKind::TransitiveClosure { edge, seed, reflexive, inverted } => {
                Self::prove_transitive_closure(
                    context, layer, plan, edge, seed, *reflexive, *inverted, depth, nodes,
                )
            }
        }
    }

    /// Prove a bounded semi-naive transitive closure. Runs the accumulate-and-dedupe loop until the
    /// one-hop frontier stops growing (the empty-delta fixpoint), building each round from the audited
    /// operator cores and threading `A_k` forward by commitment. See
    /// [`RelationProofPlanKind::TransitiveClosure`].
    #[allow(clippy::too_many_arguments)]
    fn prove_transitive_closure(
        context: &StoreProverContext<'_>,
        layer: &LayerCommitment,
        plan: &RelationProofPlan,
        edge: &LayerBgpPlan,
        seed: &CanonicalObject,
        reflexive: bool,
        inverted: bool,
        depth: usize,
        nodes: &mut usize,
    ) -> io::Result<Self> {
        let root = layer.state.commitment_root;
        let digest = plan.query_digest;
        // The E column joined against `A_k.node`: subject (0) for a forward walk, object (1) for the
        // mirrored inverse walk. In BOTH cases the join output is [node, frontier] with the new frontier
        // at index 1 (the join drops the shared key and appends E's OTHER column), so every downstream
        // operator is identical — only this key index flips.
        let edge_key = usize::from(inverted);

        // (1) Prove the edge relation E ONCE. Reused (by commitment) in every round and the fixpoint.
        let edge_plan = closure_edge_plan(edge, digest);
        let edge_env = Self::prove_bounded(context, layer, &edge_plan, depth + 1, nodes)?;
        let edge_witness = edge_env.relation_witness(None)?;

        // (2) Resolve the constant seed `a` and build the identity singleton {a}.
        let view = layer.verifier_view();
        let seed_id = resolve_closure_seed_id(&view, seed)?;
        let seed_witness =
            RelationWitness::commit_with_context(context, vec![vec![ArkFr::from_u64(seed_id)]], 1)?;

        // (3) A_0: star => {a}; plus => Distinct(object(InnerJoin({a}, E))).
        let (seed_proof, mut accumulator, mut accumulator_ids): (
            TransitiveClosureSeed,
            RelationWitness,
            std::collections::BTreeSet<u64>,
        ) = if reflexive {
            let ids = std::iter::once(seed_id).collect();
            (TransitiveClosureSeed::Reflexive, seed_witness.clone(), ids)
        } else {
            let mut t_inner = closure_transcript(&root, &digest, 0, b"seed-inner");
            let inner = prove_inner_join_with_keys_and_context(
                context, &seed_witness, &edge_witness, &[(0, edge_key)], &mut t_inner,
            )?;
            let object_witness = closure_object_branch(&inner)?;
            let rows = closure_distinct_ids(&object_witness)?;
            let mut t_distinct = closure_transcript(&root, &digest, 0, b"seed-distinct");
            let distinct = prove_relation_distinct_with_context(
                context, &object_witness, &[0], &closure_rows(&rows), &mut t_distinct,
            )?;
            let ids = rows.iter().copied().collect();
            let a0 = closure_distinct_witness(&distinct);
            (
                TransitiveClosureSeed::Successor {
                    inner: inner.proof,
                    inner_out: inner.commitments,
                    distinct: distinct.proof,
                    rows,
                },
                a0,
                ids,
            )
        };

        // (4) Accumulate-and-dedupe rounds until the frontier is closed under the step.
        let mut rounds: Vec<TransitiveClosureRound> = Vec::new();
        let fixpoint = loop {
            if *nodes >= MAX_RELATION_PROOF_NODES {
                return Err(invalid("transitive closure proving exceeds structural node limit"));
            }
            *nodes += 1;
            let step = rounds.len() as u64;

            // step = InnerJoin(A_k, E) on (A_k.node == E.subject) -> [node, object].
            let mut t_inner = closure_transcript(&root, &digest, step, b"step-inner");
            let inner = prove_inner_join_with_keys_and_context(
                context, &accumulator, &edge_witness, &[(0, edge_key)], &mut t_inner,
            )?;
            let object_witness = closure_object_branch(&inner)?;
            let neighbour_ids = closure_distinct_ids(&object_witness)?;

            if neighbour_ids.iter().all(|id| accumulator_ids.contains(id)) {
                // FIXPOINT: every one-hop neighbour of A_k is already present. The anti-join of the
                // join's objects against A_k is empty -> A_k is closed under the step.
                let mut t_anti = closure_transcript(&root, &digest, step, b"fix-anti");
                let anti = prove_left_anti_join_with_keys_and_context(
                    context,
                    &closure_inner_witness(&inner),
                    &accumulator,
                    &[(1, 0)],
                    &mut t_anti,
                )?;
                if anti.n != 0 {
                    return Err(invalid(
                        "transitive closure fixpoint anti-join is unexpectedly nonempty",
                    ));
                }
                break TransitiveClosureFixpoint {
                    inner: inner.proof,
                    inner_out: inner.commitments,
                    anti: anti.proof,
                    anti_out: anti.commitments,
                };
            }

            if rounds.len() >= MAX_TRANSITIVE_CLOSURE_ROUNDS {
                return Err(invalid(
                    "transitive closure frontier is deeper than the round budget (reject closed)",
                ));
            }

            // A_{k+1} = Distinct(BagUnion(A_k, object-bag), [0]).
            let mut t_union = closure_transcript(&root, &digest, step, b"round-union");
            let union = prove_bag_union_with_context(
                context,
                &[accumulator.clone(), object_witness],
                &mut t_union,
            )?;
            let union_witness = closure_union_witness(&union);
            let next_rows = closure_distinct_ids(&union_witness)?;
            let mut t_distinct = closure_transcript(&root, &digest, step, b"round-distinct");
            let distinct = prove_relation_distinct_with_context(
                context, &union_witness, &[0], &closure_rows(&next_rows), &mut t_distinct,
            )?;
            accumulator = closure_distinct_witness(&distinct);
            accumulator_ids = next_rows.iter().copied().collect();
            rounds.push(TransitiveClosureRound {
                inner: inner.proof,
                inner_out: inner.commitments,
                union: union.proof,
                union_out: union.commitments,
                distinct: distinct.proof,
                next_rows,
            });
        };

        // (5) Deliver A_K = accumulator as the reachable SET (one nonnullable NodeValue column).
        let n = accumulator.n;
        let column = accumulator.columns[0][..n].to_vec();
        let commitment = accumulator.commitments[0];
        Self::finish_proved(
            layer,
            plan,
            RecursiveRelationProof::TransitiveClosure {
                edge: Box::new(edge_env),
                seed: seed_proof,
                rounds,
                fixpoint,
            },
            vec![commitment],
            vec![column],
            accumulator.log_domain,
            n,
        )
    }

    fn verify_bounded(
        &self,
        layer: &impl VerifierLayer,
        expected_root: ProofHash,
        plan: &RelationProofPlan,
        depth: usize,
        nodes: &mut usize,
    ) -> io::Result<()> {
        if depth > MAX_RELATION_PROOF_DEPTH || *nodes >= MAX_RELATION_PROOF_NODES {
            return Err(invalid("recursive relation verification exceeds structural limits"));
        }
        *nodes += 1;
        if self.layer_root != expected_root {
            return Err(invalid("recursive relation root does not match trusted root"));
        }
        if self.query_digest != plan.query_digest {
            return Err(invalid("recursive relation query identity mismatch"));
        }
        let (expected_descriptors, expected_ground) = plan.result_shape()?;
        if self.ground_result != expected_ground
            || self.result_descriptors != expected_descriptors
        {
            return Err(invalid("recursive relation result schema mismatch"));
        }

        match (&self.proof, &plan.kind) {
            (
                RecursiveRelationProof::LayerTripleCount,
                RelationProofPlanKind::LayerTripleCount,
            ) => {
                let count = layer.verifier_view().state.materialized_count;
                let mut column = vec![ArkFr::zero(); 2];
                column[0] = ArkFr::from_u64(count);
                if self.result_len != 1
                    || self.result_log_domain != 1
                    || self.result_columns != vec![vec![column[0]]]
                    || self.result_commitments != vec![commit_column(&column)?]
                    || self.result_values
                        != vec![vec![CanonicalResultValue::GeneratedDecimalU64(count)]]
                {
                    return Err(invalid("layer triple-count result does not match trusted root"));
                }
            }
            (
                RecursiveRelationProof::GeneratedU64Filter { input },
                RelationProofPlanKind::GeneratedU64Filter {
                    input: input_plan,
                    column,
                    op,
                    constant,
                    constant_on_left,
                },
            ) => {
                input.verify_bounded(
                    layer,
                    expected_root,
                    input_plan,
                    depth + 1,
                    nodes,
                )?;
                let expected_columns = generated_u64_filtered_columns(
                    input,
                    *column,
                    *op,
                    *constant,
                    *constant_on_left,
                )?;
                let expected_len = expected_columns.first().map_or(0, Vec::len);
                let expected_log = expected_len
                    .max(1)
                    .next_power_of_two()
                    .max(2)
                    .trailing_zeros() as usize;
                if self.result_len != expected_len
                    || self.result_log_domain != expected_log
                    || self.result_columns != expected_columns
                {
                    return Err(invalid(
                        "generated-u64 filter result is not the exact derived selection",
                    ));
                }
                let domain = 1usize << expected_log;
                let expected_commitments = expected_columns
                    .iter()
                    .map(|column| {
                        let mut padded = column.clone();
                        padded.resize(domain, ArkFr::zero());
                        commit_column(&padded)
                    })
                    .collect::<io::Result<Vec<_>>>()?;
                if self.result_commitments != expected_commitments {
                    return Err(invalid("generated-u64 filter commitment mismatch"));
                }
            }
            (
                RecursiveRelationProof::GeneratedI64Filter { input },
                RelationProofPlanKind::GeneratedI64Filter {
                    input: input_plan,
                    column,
                    op,
                    constant,
                    constant_on_left,
                },
            ) => {
                input.verify_bounded(
                    layer,
                    expected_root,
                    input_plan,
                    depth + 1,
                    nodes,
                )?;
                let expected_columns = generated_i64_filtered_columns(
                    input,
                    *column,
                    *op,
                    *constant,
                    *constant_on_left,
                )?;
                let expected_len = expected_columns.first().map_or(0, Vec::len);
                let expected_log = expected_len
                    .max(1)
                    .next_power_of_two()
                    .max(2)
                    .trailing_zeros() as usize;
                if self.result_len != expected_len
                    || self.result_log_domain != expected_log
                    || self.result_columns != expected_columns
                {
                    return Err(invalid(
                        "generated-i64 filter result is not the exact derived selection",
                    ));
                }
                let domain = 1usize << expected_log;
                let expected_commitments = expected_columns
                    .iter()
                    .map(|column| {
                        let mut padded = column.clone();
                        padded.resize(domain, ArkFr::zero());
                        commit_column(&padded)
                    })
                    .collect::<io::Result<Vec<_>>>()?;
                if self.result_commitments != expected_commitments {
                    return Err(invalid("generated-i64 filter commitment mismatch"));
                }
            }
            (
                RecursiveRelationProof::GroupSum { input },
                RelationProofPlanKind::GroupSum {
                    input: input_plan,
                    key_columns,
                    value_column,
                    datatype,
                },
            ) => {
                input.verify_bounded(
                    layer,
                    expected_root,
                    input_plan,
                    depth + 1,
                    nodes,
                )?;
                let expected_columns = derived_group_sum_columns(
                    layer,
                    input,
                    key_columns,
                    *value_column,
                    *datatype,
                )?;
                let expected_len = expected_columns.first().map_or(0, Vec::len);
                let expected_log = expected_len
                    .max(1)
                    .next_power_of_two()
                    .max(2)
                    .trailing_zeros() as usize;
                if self.result_len != expected_len
                    || self.result_log_domain != expected_log
                    || self.result_columns != expected_columns
                {
                    return Err(invalid("group-sum result is not the exact derived aggregate"));
                }
                let domain = 1usize << expected_log;
                let expected_commitments = expected_columns
                    .iter()
                    .map(|column| {
                        let mut padded = column.clone();
                        padded.resize(domain, ArkFr::zero());
                        commit_column(&padded)
                    })
                    .collect::<io::Result<Vec<_>>>()?;
                if self.result_commitments != expected_commitments {
                    return Err(invalid("group-sum commitment mismatch"));
                }
            }
            (
                RecursiveRelationProof::GroupCount { input },
                RelationProofPlanKind::GroupCount {
                    input: input_plan,
                    key_columns,
                },
            ) => {
                input.verify_bounded(
                    layer,
                    expected_root,
                    input_plan,
                    depth + 1,
                    nodes,
                )?;
                let expected_columns =
                    derived_group_count_columns(layer, input, key_columns)?;
                let expected_len = expected_columns.first().map_or(0, Vec::len);
                let expected_log = expected_len
                    .max(1)
                    .next_power_of_two()
                    .max(2)
                    .trailing_zeros() as usize;
                if self.result_len != expected_len
                    || self.result_log_domain != expected_log
                    || self.result_columns != expected_columns
                {
                    return Err(invalid("group-count result is not the exact derived aggregate"));
                }
                let domain = 1usize << expected_log;
                let expected_commitments = expected_columns
                    .iter()
                    .map(|column| {
                        let mut padded = column.clone();
                        padded.resize(domain, ArkFr::zero());
                        commit_column(&padded)
                    })
                    .collect::<io::Result<Vec<_>>>()?;
                if self.result_commitments != expected_commitments {
                    return Err(invalid("group-count commitment mismatch"));
                }
            }
            (
                RecursiveRelationProof::CollectionCount { input },
                RelationProofPlanKind::CollectionCount { input: input_plan },
            ) => {
                input.verify_bounded(
                    layer,
                    expected_root,
                    input_plan,
                    depth + 1,
                    nodes,
                )?;
                let expected_columns = derived_cardinality_column(input.result_len, true)?;
                if self.result_len != 1
                    || self.result_log_domain != 1
                    || self.result_columns != expected_columns
                {
                    return Err(invalid(
                        "collection-count result is not the exact derived cardinality",
                    ));
                }
                let mut padded = expected_columns[0].clone();
                padded.resize(2, ArkFr::zero());
                if self.result_commitments != vec![commit_column(&padded)?] {
                    return Err(invalid("collection-count commitment mismatch"));
                }
            }
            (
                RecursiveRelationProof::CollectionSum { input },
                RelationProofPlanKind::CollectionSum {
                    input: input_plan,
                    template,
                    datatype,
                },
            ) => {
                input.verify_bounded(
                    layer,
                    expected_root,
                    input_plan,
                    depth + 1,
                    nodes,
                )?;
                let expected_columns =
                    derived_collection_sum_column(layer, input, *template, *datatype)?;
                if self.result_len != 1
                    || self.result_log_domain != 1
                    || self.result_columns != expected_columns
                {
                    return Err(invalid(
                        "collection-sum result is not the exact derived aggregate",
                    ));
                }
                let mut padded = expected_columns[0].clone();
                padded.resize(2, ArkFr::zero());
                if self.result_commitments != vec![commit_column(&padded)?] {
                    return Err(invalid("collection-sum commitment mismatch"));
                }
            }
            (
                RecursiveRelationProof::CollectionExtrema { input },
                RelationProofPlanKind::CollectionExtrema {
                    input: input_plan,
                    value_column,
                    datatype,
                    kind,
                },
            ) => {
                input.verify_bounded(
                    layer,
                    expected_root,
                    input_plan,
                    depth + 1,
                    nodes,
                )?;
                let expected_columns = derived_collection_extrema_column(
                    layer,
                    input,
                    *value_column,
                    *datatype,
                    *kind,
                )?;
                let expected_len = expected_columns[0].len();
                if self.result_len != expected_len
                    || self.result_log_domain != 1
                    || self.result_columns != expected_columns
                {
                    return Err(invalid(
                        "collection-extrema result is not the exact derived extremum",
                    ));
                }
                let mut padded = expected_columns[0].clone();
                padded.resize(2, ArkFr::zero());
                if self.result_commitments != vec![commit_column(&padded)?] {
                    return Err(invalid("collection-extrema commitment mismatch"));
                }
            }
            (RecursiveRelationProof::Bgp(envelope), RelationProofPlanKind::Bgp(bgp)) => {
                envelope.verify(layer, expected_root, bgp, plan.query_digest)?;
                if self.result_commitments != envelope.result_commitments
                    || self.result_columns != envelope.result_columns
                    || self.result_log_domain != envelope.result_log_domain
                    || self.result_len != envelope.result_len
                {
                    return Err(invalid("recursive BGP result does not match leaf envelope"));
                }
            }
            (
                RecursiveRelationProof::BagUnion { branches, proof },
                RelationProofPlanKind::BagUnion {
                    branches: branch_plans,
                    mappings,
                },
            ) => {
                if branches.len() < 2
                    || branches.len() != branch_plans.len()
                    || branches.len() != mappings.len()
                {
                    return Err(invalid("recursive bag-union branch count mismatch"));
                }
                let mut inputs = Vec::with_capacity(branches.len());
                for ((branch, branch_plan), mapping) in
                    branches.iter().zip(branch_plans).zip(mappings)
                {
                    branch.verify_bounded(
                        layer,
                        expected_root,
                        branch_plan,
                        depth + 1,
                        nodes,
                    )?;
                    inputs.push(if expected_ground {
                        branch.relation_commit(None)?
                    } else {
                        branch.relation_commit(Some(mapping))?
                    });
                }
                let mut transcript =
                    super::argument::Transcript254::new(b"parture-union-envelope");
                transcript.append_bytes(b"pow/union-envelope/root", &expected_root);
                transcript.append_bytes(b"pow/union-envelope/query", &plan.query_digest);
                verify_bag_union(
                    &inputs,
                    &self.result_commitments,
                    self.result_len,
                    self.result_log_domain,
                    proof,
                    &mut transcript,
                )?;
            }
            (
                RecursiveRelationProof::BagProjection { input },
                RelationProofPlanKind::BagProjection {
                    input: input_plan,
                    indices,
                },
            ) => {
                input.verify_bounded(
                    layer,
                    expected_root,
                    input_plan,
                    depth + 1,
                    nodes,
                )?;
                let (expected_commitments, expected_columns) = if indices.is_empty() {
                    let unit = ground_unit_column(input.result_len, input.result_log_domain);
                    (
                        vec![commit_column(&unit)?],
                        vec![unit[..input.result_len].to_vec()],
                    )
                } else {
                    let commitments = indices
                        .iter()
                        .map(|index| {
                            input
                                .result_commitments
                                .get(*index)
                                .copied()
                                .ok_or_else(|| invalid("bag projection index out of range"))
                        })
                        .collect::<io::Result<Vec<_>>>()?;
                    let columns = indices
                        .iter()
                        .map(|index| {
                            input
                                .result_columns
                                .get(*index)
                                .cloned()
                                .ok_or_else(|| invalid("bag projection index out of range"))
                        })
                        .collect::<io::Result<Vec<_>>>()?;
                    (commitments, columns)
                };
                if self.result_commitments != expected_commitments
                    || self.result_columns != expected_columns
                    || self.result_log_domain != input.result_log_domain
                    || self.result_len != input.result_len
                {
                    return Err(invalid("bag projection changed rows, domain, or multiplicity"));
                }
            }
            (
                RecursiveRelationProof::Distinct { input, proof },
                RelationProofPlanKind::Distinct {
                    input: input_plan,
                    key_columns,
                },
            ) => {
                input.verify_bounded(layer, expected_root, input_plan, depth + 1, nodes)?;
                if self.result_columns.len() != key_columns.len()
                    || self
                        .result_columns
                        .iter()
                        .any(|column| self.result_len > column.len())
                {
                    return Err(invalid("distinct result column shape mismatch"));
                }
                let view = layer.verifier_view();
                let resolve_id = |field: ArkFr, namespace: ResultIdNamespace| -> io::Result<u64> {
                    match namespace {
                        ResultIdNamespace::Predicate => view
                            .predicate_catalog
                            .iter()
                            .find(|record| ArkFr::from_u64(record.predicate_id) == field)
                            .map(|record| record.predicate_id),
                        ResultIdNamespace::NodeValue => view
                            .node_value_catalog
                            .iter()
                            .find(|record| ArkFr::from_u64(record.object_id) == field)
                            .map(|record| record.object_id),
                        ResultIdNamespace::GeneratedDecimalU64
                        | ResultIdNamespace::GeneratedDecimalI64 => None,
                    }
                    .ok_or_else(|| invalid("distinct output id absent from planned namespace"))
                };
                let rows = (0..self.result_len)
                    .map(|row| {
                        self.result_columns
                            .iter()
                            .zip(&expected_descriptors)
                            .map(|(column, descriptor)| resolve_id(column[row], descriptor.namespace))
                            .collect::<io::Result<Vec<_>>>()
                    })
                    .collect::<io::Result<Vec<_>>>()?;
                let mut transcript = Transcript254::new(b"parture-distinct-envelope");
                transcript.append_bytes(b"pow/distinct-envelope/root", &expected_root);
                transcript.append_bytes(b"pow/distinct-envelope/query", &plan.query_digest);
                verify_relation_distinct(
                    &input.relation_commit(None)?,
                    key_columns,
                    &rows,
                    proof,
                    &mut transcript,
                )?;
            }
            (
                RecursiveRelationProof::LeftAnti { left, right, proof },
                RelationProofPlanKind::LeftAnti {
                    left: left_plan,
                    right: right_plan,
                    key_pairs,
                },
            ) => {
                left.verify_bounded(
                    layer,
                    expected_root,
                    left_plan,
                    depth + 1,
                    nodes,
                )?;
                right.verify_bounded(
                    layer,
                    expected_root,
                    right_plan,
                    depth + 1,
                    nodes,
                )?;
                let left_commit = left.relation_commit(None)?;
                let right_commit = right.relation_commit(None)?;
                let mut transcript =
                    super::argument::Transcript254::new(b"parture-left-anti-envelope");
                transcript.append_bytes(b"pow/anti-envelope/root", &expected_root);
                transcript.append_bytes(b"pow/anti-envelope/query", &plan.query_digest);
                match (key_pairs.is_empty(), proof.as_ref()) {
                    (true, None) => verify_global_not(
                        &left_commit,
                        &right_commit,
                        &self.result_commitments,
                        self.result_len,
                        self.result_log_domain,
                    )?,
                    (false, Some(proof)) => verify_left_anti_join_with_keys(
                        &left_commit,
                        &right_commit,
                        key_pairs,
                        &self.result_commitments,
                        self.result_len,
                        self.result_log_domain,
                        proof,
                        &mut transcript,
                    )?,
                    _ => {
                        return Err(invalid(
                            "recursive left-anti key/proof variant mismatch",
                        ))
                    }
                }
            }
            (
                RecursiveRelationProof::LeftOuter { left, right, proof },
                RelationProofPlanKind::LeftOuter {
                    left: left_plan,
                    right: right_plan,
                    key_pairs,
                },
            ) => {
                left.verify_bounded(
                    layer,
                    expected_root,
                    left_plan,
                    depth + 1,
                    nodes,
                )?;
                right.verify_bounded(
                    layer,
                    expected_root,
                    right_plan,
                    depth + 1,
                    nodes,
                )?;
                let left_commit = left.relation_commit(None)?;
                let right_commit = right.relation_commit(None)?;
                let mut transcript =
                    super::argument::Transcript254::new(b"parture-left-outer-envelope");
                transcript.append_bytes(b"pow/outer-envelope/root", &expected_root);
                transcript.append_bytes(b"pow/outer-envelope/query", &plan.query_digest);
                if key_pairs.is_empty() {
                    verify_global_left_outer_join(
                        &left_commit,
                        &right_commit,
                        &self.result_commitments,
                        self.result_len,
                        self.result_log_domain,
                        proof,
                        &mut transcript,
                    )?;
                } else {
                    verify_left_outer_join_with_keys(
                        &left_commit,
                        &right_commit,
                        key_pairs,
                        &self.result_commitments,
                        self.result_len,
                        self.result_log_domain,
                        proof,
                        &mut transcript,
                    )?;
                }
            }
            (
                RecursiveRelationProof::InnerJoin { left, right, proof },
                RelationProofPlanKind::InnerJoin { left: left_plan, right: right_plan, key_pairs },
            ) => {
                left.verify_bounded(layer, expected_root, left_plan, depth + 1, nodes)?;
                right.verify_bounded(layer, expected_root, right_plan, depth + 1, nodes)?;
                let mut transcript = Transcript254::new(b"parture-inner-join-envelope");
                transcript.append_bytes(b"pow/inner-envelope/root", &expected_root);
                transcript.append_bytes(b"pow/inner-envelope/query", &plan.query_digest);
                verify_inner_join_with_keys(
                    &left.relation_commit(None)?, &right.relation_commit(None)?, key_pairs,
                    &self.result_commitments, self.result_len, self.result_log_domain,
                    proof, &mut transcript,
                )?;
            }
            (
                RecursiveRelationProof::Equality { input, proof },
                RelationProofPlanKind::Equality {
                    input: input_plan,
                    constraint,
                },
            ) => {
                input.verify_bounded(
                    layer,
                    expected_root,
                    input_plan,
                    depth + 1,
                    nodes,
                )?;
                let resolved = resolve_relation_equality_constraint(layer, constraint)?;
                let mut transcript =
                    super::argument::Transcript254::new(b"parture-equality-envelope");
                transcript.append_bytes(b"pow/equality-envelope/root", &expected_root);
                transcript.append_bytes(b"pow/equality-envelope/query", &plan.query_digest);
                verify_relation_equality(
                    &input.relation_commit(None)?,
                    &resolved,
                    &self.result_commitments,
                    self.result_len,
                    self.result_log_domain,
                    proof,
                    &mut transcript,
                )?;
            }
            (
                RecursiveRelationProof::TupleDisequality { input, proof },
                RelationProofPlanKind::ExactTupleDisequality {
                    input: input_plan,
                    pairs,
                },
            ) => {
                input.verify_bounded(
                    layer,
                    expected_root,
                    input_plan,
                    depth + 1,
                    nodes,
                )?;
                let resolved = resolve_tuple_disequality_pairs(layer, pairs)?;
                let mut transcript =
                    super::argument::Transcript254::new(b"parture-tuple-disequality-envelope");
                transcript.append_bytes(b"pow/tuple-disequality-envelope/root", &expected_root);
                transcript.append_bytes(b"pow/tuple-disequality-envelope/query", &plan.query_digest);
                verify_tuple_disequality(
                    &input.relation_commit(None)?,
                    &resolved,
                    &self.result_commitments,
                    self.result_len,
                    self.result_log_domain,
                    proof,
                    &mut transcript,
                )?;
            }
            (
                RecursiveRelationProof::ExactScalarOrder { input, proof },
                RelationProofPlanKind::ExactScalarOrder {
                    input: input_plan,
                    datatype,
                    direction,
                },
            ) => {
                input.verify_bounded(
                    layer,
                    expected_root,
                    input_plan,
                    depth + 1,
                    nodes,
                )?;
                let mut transcript = Transcript254::new(b"parture-order-envelope");
                transcript.append_bytes(b"pow/order-envelope/root", &expected_root);
                transcript.append_bytes(b"pow/order-envelope/query", &plan.query_digest);
                verify_exact_scalar_order(
                    layer,
                    &input.relation_commit(None)?,
                    *datatype,
                    *direction,
                    &self.result_commitments,
                    self.result_len,
                    self.result_log_domain,
                    proof,
                    &mut transcript,
                )?;
            }
            (
                RecursiveRelationProof::ExactMultiOrder { input, proof },
                RelationProofPlanKind::ExactMultiOrder { input: input_plan, datatypes, key_indices, directions },
            ) => {
                input.verify_bounded(layer, expected_root, input_plan, depth + 1, nodes)?;
                let mut transcript = Transcript254::new(b"parture-multi-order-envelope");
                transcript.append_bytes(b"pow/multi-order-envelope/root", &expected_root);
                transcript.append_bytes(b"pow/multi-order-envelope/query", &plan.query_digest);
                verify_exact_multi_order(layer, &input.relation_commit(None)?, &self.result_columns, datatypes, key_indices, directions, &self.result_commitments, self.result_len, self.result_log_domain, proof, &mut transcript)?;
            }
            (
                RecursiveRelationProof::ExactOrderedSlice { input },
                RelationProofPlanKind::ExactOrderedSlice {
                    input: input_plan,
                    offset,
                    limit,
                },
            ) => {
                input.verify_bounded(
                    layer,
                    expected_root,
                    input_plan,
                    depth + 1,
                    nodes,
                )?;
                let (start, end) = ordered_slice_bounds(input.result_len, *offset, *limit);
                let take = end - start;
                let expected_log = take.max(1).next_power_of_two().max(2).trailing_zeros() as usize;
                let expected_columns = input
                    .result_columns
                    .iter()
                    .map(|column| column[start..end].to_vec())
                    .collect::<Vec<_>>();
                if self.result_len != take
                    || self.result_log_domain != expected_log
                    || self.result_columns != expected_columns
                {
                    return Err(invalid("ordered slice result is not the exact contiguous child slice"));
                }
                let domain = 1usize << expected_log;
                let expected_commitments = expected_columns
                    .iter()
                    .map(|column| {
                        let mut padded = column.clone();
                        padded.resize(domain, ArkFr::zero());
                        commit_column(&padded)
                    })
                    .collect::<io::Result<Vec<_>>>()?;
                if self.result_commitments != expected_commitments {
                    return Err(invalid("ordered slice commitment mismatch"));
                }
            }
            (
                RecursiveRelationProof::GroupExtrema { input },
                RelationProofPlanKind::GroupExtrema { input: input_plan, key_columns, value_column, datatype, kind },
            ) => {
                input.verify_bounded(layer, expected_root, input_plan, depth + 1, nodes)?;
                let expected_columns = derived_group_extrema_columns(
                    layer, input, key_columns, *value_column, *datatype, *kind,
                )?;
                let expected_len = expected_columns.first().map_or(0, Vec::len);
                let expected_log = expected_len.max(1).next_power_of_two().max(2)
                    .trailing_zeros() as usize;
                if self.result_len != expected_len
                    || self.result_log_domain != expected_log
                    || self.result_columns.len() != expected_columns.len()
                    || self.result_columns.iter().zip(&expected_columns)
                        .any(|(actual, expected)| actual.get(..expected_len) != Some(expected.as_slice()))
                {
                    return Err(invalid("group-extrema result is not the exact canonical aggregate"));
                }
                let domain = 1usize << expected_log;
                let expected_commitments = expected_columns.iter().map(|column| {
                    let mut padded = column.clone();
                    padded.resize(domain, ArkFr::zero());
                    commit_column(&padded)
                }).collect::<io::Result<Vec<_>>>()?;
                if self.result_commitments != expected_commitments {
                    return Err(invalid("group-extrema commitment mismatch"));
                }
            }
            (
                RecursiveRelationProof::TransitiveClosure {
                    edge: edge_env,
                    seed,
                    rounds,
                    fixpoint,
                },
                RelationProofPlanKind::TransitiveClosure {
                    edge,
                    seed: seed_obj,
                    reflexive,
                    inverted,
                },
            ) => {
                let expected_rows = self.verify_transitive_closure(
                    layer,
                    expected_root,
                    plan,
                    edge,
                    seed_obj,
                    *reflexive,
                    *inverted,
                    edge_env,
                    seed,
                    rounds,
                    fixpoint,
                    depth,
                    nodes,
                )?;
                // The delivered relation must be exactly A_K (the reachable set), in the order the
                // proof's final round committed it. Length, columns, and domain are cross-checked here;
                // the shared tail then binds the commitments to these columns and resolves the ids to
                // node IRIs against the trusted dictionary.
                let expected_log = expected_rows
                    .len()
                    .max(1)
                    .next_power_of_two()
                    .max(2)
                    .trailing_zeros() as usize;
                let expected_column: Vec<ArkFr> =
                    expected_rows.iter().map(|id| ArkFr::from_u64(*id)).collect();
                if self.result_len != expected_rows.len()
                    || self.result_log_domain != expected_log
                    || self.result_columns != vec![expected_column]
                {
                    return Err(invalid(
                        "transitive closure delivered relation is not the proven reachable set",
                    ));
                }
            }
            _ => return Err(invalid("recursive relation proof variant mismatch")),
        }

        verify_result_columns(
            &self.result_commitments,
            &self.result_columns,
            self.result_len,
            self.result_log_domain,
        )?;
        if self.ground_result {
            if self.result_commitments.len() != 1
                || self.result_columns.len() != 1
                || !self.result_descriptors.is_empty()
                || !self.result_values.is_empty()
            {
                return Err(invalid("invalid recursive ground-result shape"));
            }
        } else {
            let values = resolve_result_values(
                layer,
                expected_root,
                &self.result_columns,
                &expected_descriptors,
            )?;
            if values != self.result_values {
                return Err(invalid("recursive canonical result mismatch"));
            }
        }
        Ok(())
    }

    /// Verify a bounded semi-naive transitive closure and return the reachable-set node ids the
    /// delivered relation must equal. Reconstructs the closure identity (edge, seed, star/plus) from
    /// the trusted plan, reads `K` = `rounds.len()` from the proof, and replays each audited step
    /// against the SAME `E` and the previous accumulator commitment. Concludes with the empty-delta
    /// fixpoint. No new gadget — this only orchestrates already-audited operator verifiers.
    #[allow(clippy::too_many_arguments)]
    fn verify_transitive_closure(
        &self,
        layer: &impl VerifierLayer,
        expected_root: ProofHash,
        plan: &RelationProofPlan,
        edge: &LayerBgpPlan,
        seed_obj: &CanonicalObject,
        reflexive: bool,
        inverted: bool,
        edge_env: &RecursiveRelationEnvelope,
        seed: &TransitiveClosureSeed,
        rounds: &[TransitiveClosureRound],
        fixpoint: &TransitiveClosureFixpoint,
        depth: usize,
        nodes: &mut usize,
    ) -> io::Result<Vec<u64>> {
        let root = expected_root;
        let digest = plan.query_digest;
        // See `prove_transitive_closure`: forward walks join E.subject (0), inverted walks E.object (1).
        let edge_key = usize::from(inverted);

        // (1) Verify the ONE edge relation E, then extract its commitment for reuse everywhere.
        let edge_plan = closure_edge_plan(edge, digest);
        edge_env.verify_bounded(layer, expected_root, &edge_plan, depth + 1, nodes)?;
        let edge_commit = edge_env.relation_commit(None)?;

        // (2) Reconstruct the constant seed `{a}` from the trusted seed node.
        let seed_id = resolve_closure_seed_id(&layer.verifier_view(), seed_obj)?;

        // Helper: verify one `InnerJoin(A, E) -> [node, object]` step and return (join commit,
        // object-branch commit). The output length / domain are derived from the proof and cross-checked
        // by the audited verifier.
        let verify_inner = |left: &RelationCommit,
                            inner_proof: &InnerJoinProof,
                            inner_out: &[ArkGT],
                            step: u64,
                            role: &[u8]|
         -> io::Result<(RelationCommit, RelationCommit)> {
            if inner_out.len() != 2 {
                return Err(invalid("transitive closure inner join output arity must be 2"));
            }
            let j_len = inner_proof.anti.join_len;
            let j_log = inner_proof.anti.join.num_vars();
            let mut transcript = closure_transcript(&root, &digest, step, role);
            verify_inner_join_with_keys(
                left,
                &edge_commit,
                &[(0, edge_key)],
                inner_out,
                j_len,
                j_log,
                inner_proof,
                &mut transcript,
            )?;
            let join_commit = RelationCommit {
                commitments: inner_out.to_vec(),
                log_domain: j_log,
                n: j_len,
            };
            let object_commit = RelationCommit {
                commitments: vec![inner_out[1]],
                log_domain: j_log,
                n: j_len,
            };
            Ok((join_commit, object_commit))
        };

        // (3) A_0.
        let (mut accumulator_commit, mut accumulator_ids, mut accumulator_rows): (
            RelationCommit,
            std::collections::BTreeSet<u64>,
            Vec<u64>,
        ) = match (reflexive, seed) {
            (true, TransitiveClosureSeed::Reflexive) => {
                // star: A_0 = {a}, reconstructed as a trusted constant (identity always succeeds).
                (
                    closure_seed_commit(seed_id)?,
                    std::iter::once(seed_id).collect(),
                    vec![seed_id],
                )
            }
            (false, TransitiveClosureSeed::Successor { inner, inner_out, distinct, rows }) => {
                // plus: A_0 = Distinct(object(InnerJoin({a}, E))).
                let seed_commit = closure_seed_commit(seed_id)?;
                let (_join, object_commit) =
                    verify_inner(&seed_commit, inner, inner_out, 0, b"seed-inner")?;
                let mut t_distinct = closure_transcript(&root, &digest, 0, b"seed-distinct");
                verify_relation_distinct(
                    &object_commit,
                    &[0],
                    &closure_rows(rows),
                    distinct,
                    &mut t_distinct,
                )?;
                (
                    closure_set_commit(rows)?,
                    rows.iter().copied().collect(),
                    rows.clone(),
                )
            }
            _ => {
                return Err(invalid(
                    "transitive closure seed variant does not match the star/plus plan",
                ))
            }
        };

        // (4) Replay each accumulate-and-dedupe round against the SAME E and the previous accumulator.
        for (step, round) in rounds.iter().enumerate() {
            if *nodes >= MAX_RELATION_PROOF_NODES {
                return Err(invalid(
                    "transitive closure verification exceeds structural node limit",
                ));
            }
            *nodes += 1;
            let step = step as u64;
            let (_join, object_commit) = verify_inner(
                &accumulator_commit,
                &round.inner,
                &round.inner_out,
                step,
                b"step-inner",
            )?;

            // BagUnion(A_k, object-bag): both single-column; output length is the branch-length sum and
            // the domain is bound in the proof, so the audited verifier re-derives and checks them.
            if round.union_out.len() != 1 {
                return Err(invalid("transitive closure union output arity must be 1"));
            }
            let union_len = accumulator_commit
                .n
                .checked_add(object_commit.n)
                .ok_or_else(|| invalid("transitive closure union length overflow"))?;
            let union_log = round.union.sumcheck.round_polys.len();
            let mut t_union = closure_transcript(&root, &digest, step, b"round-union");
            verify_bag_union(
                &[accumulator_commit.clone(), object_commit],
                &round.union_out,
                union_len,
                union_log,
                &round.union,
                &mut t_union,
            )?;
            let union_commit = RelationCommit {
                commitments: round.union_out.clone(),
                log_domain: union_log,
                n: union_len,
            };

            // Distinct(union, [0]) proves next_rows are exactly the deduped union set.
            let mut t_distinct = closure_transcript(&root, &digest, step, b"round-distinct");
            verify_relation_distinct(
                &union_commit,
                &[0],
                &closure_rows(&round.next_rows),
                &round.distinct,
                &mut t_distinct,
            )?;

            // Defensive monotonicity: every element of A_k must survive into A_{k+1}. This is already
            // implied by union-branch-0 + the distinct exact-cover argument, but the redundant public
            // set check costs nothing and pins the completeness invariant for the audit.
            let next_ids: std::collections::BTreeSet<u64> =
                round.next_rows.iter().copied().collect();
            if next_ids.len() != round.next_rows.len() {
                return Err(invalid("transitive closure round produced a duplicate set element"));
            }
            if !accumulator_ids.iter().all(|id| next_ids.contains(id)) {
                return Err(invalid("transitive closure round dropped a prior reachable node"));
            }

            accumulator_commit = closure_set_commit(&round.next_rows)?;
            accumulator_ids = next_ids;
            accumulator_rows = round.next_rows.clone();
        }

        // (5) Empty-delta fixpoint: LeftAnti(InnerJoin(A_K, E), A_K) on (object == node) has 0 rows,
        // i.e. A_K is closed under the step. Passing output_len 0 IS the fixpoint assertion — a
        // non-converged closure makes the audited anti-join verifier reject.
        let step = rounds.len() as u64;
        let (join_commit, _object_commit) = verify_inner(
            &accumulator_commit,
            &fixpoint.inner,
            &fixpoint.inner_out,
            step,
            b"step-inner",
        )?;
        if fixpoint.anti_out.len() != join_commit.commitments.len() {
            return Err(invalid("transitive closure fixpoint anti-join output arity mismatch"));
        }
        let anti_log = fixpoint.anti.join.num_vars();
        let mut t_anti = closure_transcript(&root, &digest, step, b"fix-anti");
        verify_left_anti_join_with_keys(
            &join_commit,
            &accumulator_commit,
            &[(1, 0)],
            &fixpoint.anti_out,
            0,
            anti_log,
            &fixpoint.anti,
            &mut t_anti,
        )?;

        Ok(accumulator_rows)
    }

    pub fn verify(
        &self,
        layer: &impl VerifierLayer,
        expected_root: ProofHash,
        plan: &RelationProofPlan,
    ) -> io::Result<()> {
        // COMMITMENT DIET (Wave 0): delta layers are not directly query-provable.
        super::query::require_base_layer_for_query(layer.verifier_view().state)?;
        // Validate the entire caller-supplied tree once with a global node budget before
        // descending through the equally bounded delivered proof tree.
        plan.result_shape()?;
        let mut nodes = 0;
        self.verify_bounded(layer, expected_root, plan, 1, &mut nodes)
    }

    /// TWO-ROOT ENTRY — verify a top-level [`RelationProofPlanKind::CrossRootHashJoin`]. Validates
    /// BOTH per-graph commitments against the trusted `database_root`, verifies each child against
    /// its OWN root (single-root subtrees — the `verify_bounded` single-root invariant is preserved
    /// WITHIN each child), authenticates each side's key column to its IRI-hash against its own root,
    /// and verifies the inner join over the two hash columns. `database_root` MUST come from a trusted
    /// out-of-band pin, never recomputed from the prover's proof bundle or schema view.
    pub fn verify_cross_root(
        &self,
        instance: &impl VerifierLayer,
        schema: &impl VerifierLayer,
        database_root: ProofHash,
        plan: &RelationProofPlan,
    ) -> io::Result<()> {
        let (expected_descriptors, expected_ground) = plan.result_shape()?;
        let (left_plan, right_plan, left_key_col, right_key_col) = match &plan.kind {
            RelationProofPlanKind::CrossRootHashJoin { left, right, left_key_col, right_key_col } => {
                (left.as_ref(), right.as_ref(), *left_key_col, *right_key_col)
            }
            _ => return Err(invalid("verify_cross_root requires a CrossRootHashJoin plan")),
        };
        let (
            left,
            right,
            left_hash,
            right_hash,
            relocate_closure,
            com_closure_at_d,
            relocate_x,
            com_x_at_d,
            relocate_h,
            com_h_at_d,
            semijoin,
        ) = match &self.proof {
            RecursiveRelationProof::CrossRootHashJoin {
                left,
                right,
                left_hash,
                right_hash,
                relocate_closure,
                com_closure_at_d,
                relocate_x,
                com_x_at_d,
                relocate_h,
                com_h_at_d,
                semijoin,
            } => (
                left.as_ref(),
                right.as_ref(),
                left_hash.as_ref(),
                right_hash.as_ref(),
                relocate_closure,
                com_closure_at_d,
                relocate_x,
                com_x_at_d,
                relocate_h,
                com_h_at_d,
                semijoin.as_ref(),
            ),
            _ => return Err(invalid("cross-root proof/plan mismatch")),
        };

        // (1) Both per-graph commitments validated AND combined into the trusted database anchor.
        super::commitment::validate_database_root(instance, schema, &database_root)?;
        let instance_view = instance.verifier_view();
        let schema_view = schema.verifier_view();
        let instance_root = instance_view.state.commitment_root;
        let schema_root = schema_view.state.commitment_root;

        // Result identity: the output lives in the INSTANCE namespace.
        if self.layer_root != instance_root {
            return Err(invalid("cross-root result is not anchored to the instance root"));
        }
        if self.query_digest != plan.query_digest {
            return Err(invalid("cross-root query identity mismatch"));
        }
        if self.ground_result != expected_ground || self.result_descriptors != expected_descriptors {
            return Err(invalid("cross-root result schema mismatch"));
        }

        // (2) Verify each child against ITS OWN root (single-root subtrees).
        let mut lnodes = 0usize;
        left.verify_bounded(schema, schema_root, left_plan, 1, &mut lnodes)?;
        let mut rnodes = 0usize;
        right.verify_bounded(instance, instance_root, right_plan, 1, &mut rnodes)?;

        if left.result_commitments.len() != 1 || left_key_col != 0 {
            return Err(invalid("cross-root left child must be a single key column"));
        }
        if right.result_commitments.len() != 2 || right_key_col >= 2 {
            return Err(invalid("cross-root right child must be exactly [output, class-key]"));
        }
        let x_col = 1 - right_key_col;

        let mut transcript = Transcript254::new(b"parture-cross-root-envelope");
        transcript.append_bytes(b"pow/cross-root/db-root", &database_root);
        transcript.append_bytes(b"pow/cross-root/query", &plan.query_digest);

        // (3) Authenticate each side's key id column to its IRI-hash against its OWN root.
        verify_dict_hash_lookup(
            &left.result_commitments[0], left.result_log_domain, left.result_len, schema, left_hash, &mut transcript,
        )?;
        verify_dict_hash_lookup(
            &right.result_commitments[right_key_col], right.result_log_domain, right.result_len, instance, right_hash, &mut transcript,
        )?;

        // (4) Verify the three UP-relocations to the shared working domain 2^D, each bound to its
        // root-authenticated source commitment.
        let instance_n_dict = instance_view.state.node_value_count as usize;
        let schema_n_dict = schema_view.state.node_value_count as usize;
        let d_left = super::dicthash::working_domain_log(left.result_log_domain, schema_n_dict);
        let d_right = super::dicthash::working_domain_log(right.result_log_domain, instance_n_dict);
        let d_log = d_left.max(d_right);
        verify_column_relocation(
            &left_hash.lookup.com_v_hash, d_left, com_closure_at_d, d_log, relocate_closure, &mut transcript,
        )?;
        verify_column_relocation(
            &right_hash.lookup.com_v_hash, d_right, com_h_at_d, d_log, relocate_h, &mut transcript,
        )?;
        verify_column_relocation(
            &right.result_commitments[x_col], right.result_log_domain, com_x_at_d, d_log, relocate_x, &mut transcript,
        )?;

        // (5) Full-field LogUp semi-join: kept ?x rows whose class-hash is in the closure hash set.
        verify_hash_semijoin(
            com_closure_at_d, left.result_len, com_x_at_d, com_h_at_d, right.result_len,
            self.result_len, semijoin, &mut transcript,
        )?;
        // The delivered output is the semi-join's kept-?x column at domain 2^D.
        if self.result_log_domain != d_log
            || self.result_commitments != vec![semijoin.com_x_kept]
        {
            return Err(invalid("cross-root result does not match the semi-join output"));
        }
        Ok(())
    }

    fn put_bounded(
        &self,
        out: &mut Vec<u8>,
        depth: usize,
        nodes: &mut usize,
    ) -> io::Result<()> {
        if depth > MAX_RELATION_PROOF_DEPTH || *nodes >= MAX_RELATION_PROOF_NODES {
            return Err(invalid("recursive relation envelope exceeds structural limits"));
        }
        *nodes += 1;
        self.layer_root.put(out)?;
        self.query_digest.put(out)?;
        match &self.proof {
            RecursiveRelationProof::LayerTripleCount => {
                12u8.put(out)?;
            }
            RecursiveRelationProof::GeneratedU64Filter { input } => {
                13u8.put(out)?;
                input.put_bounded(out, depth + 1, nodes)?;
            }
            RecursiveRelationProof::GeneratedI64Filter { input } => {
                15u8.put(out)?;
                input.put_bounded(out, depth + 1, nodes)?;
            }
            RecursiveRelationProof::GroupSum { input } => {
                14u8.put(out)?;
                input.put_bounded(out, depth + 1, nodes)?;
            }
            RecursiveRelationProof::GroupCount { input } => {
                16u8.put(out)?;
                input.put_bounded(out, depth + 1, nodes)?;
            }
            RecursiveRelationProof::CollectionCount { input } => {
                17u8.put(out)?;
                input.put_bounded(out, depth + 1, nodes)?;
            }
            RecursiveRelationProof::CollectionSum { input } => {
                18u8.put(out)?;
                input.put_bounded(out, depth + 1, nodes)?;
            }
            RecursiveRelationProof::CollectionExtrema { input } => {
                19u8.put(out)?;
                input.put_bounded(out, depth + 1, nodes)?;
            }
            RecursiveRelationProof::Bgp(envelope) => {
                0u8.put(out)?;
                envelope.put(out)?;
            }
            RecursiveRelationProof::BagUnion { branches, proof } => {
                1u8.put(out)?;
                if branches.len() > MAX_RELATION_PROOF_NODES {
                    return Err(invalid("too many recursive union branches"));
                }
                branches.len().put(out)?;
                for branch in branches {
                    branch.put_bounded(out, depth + 1, nodes)?;
                }
                proof.put(out)?;
            }
            RecursiveRelationProof::BagProjection { input } => {
                10u8.put(out)?;
                input.put_bounded(out, depth + 1, nodes)?;
            }
            RecursiveRelationProof::Distinct { input, proof } => {
                11u8.put(out)?;
                input.put_bounded(out, depth + 1, nodes)?;
                proof.put(out)?;
            }
            RecursiveRelationProof::LeftAnti { left, right, proof } => {
                2u8.put(out)?;
                left.put_bounded(out, depth + 1, nodes)?;
                right.put_bounded(out, depth + 1, nodes)?;
                proof.put(out)?;
            }
            RecursiveRelationProof::LeftOuter { left, right, proof } => {
                3u8.put(out)?;
                left.put_bounded(out, depth + 1, nodes)?;
                right.put_bounded(out, depth + 1, nodes)?;
                proof.put(out)?;
            }
            RecursiveRelationProof::InnerJoin { left, right, proof } => {
                9u8.put(out)?;
                left.put_bounded(out, depth + 1, nodes)?;
                right.put_bounded(out, depth + 1, nodes)?;
                proof.put(out)?;
            }
            RecursiveRelationProof::CrossRootHashJoin {
                left,
                right,
                left_hash,
                right_hash,
                relocate_closure,
                com_closure_at_d,
                relocate_x,
                com_x_at_d,
                relocate_h,
                com_h_at_d,
                semijoin,
            } => {
                22u8.put(out)?;
                left.put_bounded(out, depth + 1, nodes)?;
                right.put_bounded(out, depth + 1, nodes)?;
                left_hash.put(out)?;
                right_hash.put(out)?;
                relocate_closure.put(out)?;
                com_closure_at_d.put(out)?;
                relocate_x.put(out)?;
                com_x_at_d.put(out)?;
                relocate_h.put(out)?;
                com_h_at_d.put(out)?;
                semijoin.put(out)?;
            }
            RecursiveRelationProof::Equality { input, proof } => {
                4u8.put(out)?;
                input.put_bounded(out, depth + 1, nodes)?;
                proof.put(out)?;
            }
            RecursiveRelationProof::TupleDisequality { input, proof } => {
                20u8.put(out)?;
                input.put_bounded(out, depth + 1, nodes)?;
                proof.put(out)?;
            }
            RecursiveRelationProof::ExactScalarOrder { input, proof } => {
                6u8.put(out)?;
                input.put_bounded(out, depth + 1, nodes)?;
                proof.put(out)?;
            }
            RecursiveRelationProof::ExactOrderedSlice { input } => {
                7u8.put(out)?;
                input.put_bounded(out, depth + 1, nodes)?;
            }
            RecursiveRelationProof::ExactMultiOrder { input, proof } => {
                8u8.put(out)?;
                input.put_bounded(out, depth + 1, nodes)?;
                proof.put(out)?;
            }
            RecursiveRelationProof::GroupExtrema { input } => {
                5u8.put(out)?;
                input.put_bounded(out, depth + 1, nodes)?;
            }
            RecursiveRelationProof::TransitiveClosure {
                edge,
                seed,
                rounds,
                fixpoint,
            } => {
                21u8.put(out)?;
                if rounds.len() > MAX_TRANSITIVE_CLOSURE_ROUNDS {
                    return Err(invalid("too many transitive-closure rounds"));
                }
                edge.put_bounded(out, depth + 1, nodes)?;
                seed.put(out)?;
                rounds.put(out)?;
                fixpoint.put(out)?;
            }
        }
        self.result_commitments.put(out)?;
        self.result_columns.put(out)?;
        self.result_log_domain.put(out)?;
        self.result_len.put(out)?;
        self.ground_result.put(out)?;
        self.result_descriptors.put(out)?;
        self.result_values.put(out)
    }

    fn get_bounded(
        reader: &mut Reader<'_>,
        depth: usize,
        nodes: &mut usize,
    ) -> io::Result<Self> {
        if depth > MAX_RELATION_PROOF_DEPTH || *nodes >= MAX_RELATION_PROOF_NODES {
            return Err(invalid("recursive relation envelope exceeds structural limits"));
        }
        *nodes += 1;
        let layer_root = ProofHash::get(reader)?;
        let query_digest = ProofHash::get(reader)?;
        let proof = match u8::get(reader)? {
            12 => RecursiveRelationProof::LayerTripleCount,
            13 => RecursiveRelationProof::GeneratedU64Filter {
                input: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
            },
            15 => RecursiveRelationProof::GeneratedI64Filter {
                input: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
            },
            14 => RecursiveRelationProof::GroupSum {
                input: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
            },
            16 => RecursiveRelationProof::GroupCount {
                input: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
            },
            17 => RecursiveRelationProof::CollectionCount {
                input: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
            },
            18 => RecursiveRelationProof::CollectionSum {
                input: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
            },
            19 => RecursiveRelationProof::CollectionExtrema {
                input: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
            },
            0 => RecursiveRelationProof::Bgp(Box::<LayerBgpEnvelope>::get(reader)?),
            1 => {
                let len = usize::get(reader)?;
                if len > MAX_RELATION_PROOF_NODES - *nodes {
                    return Err(invalid("too many recursive union branches"));
                }
                let mut branches = Vec::new();
                for _ in 0..len {
                    branches.push(Self::get_bounded(reader, depth + 1, nodes)?);
                }
                RecursiveRelationProof::BagUnion {
                    branches,
                    proof: Box::<BagUnionProof>::get(reader)?,
                }
            }
            2 => RecursiveRelationProof::LeftAnti {
                left: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
                right: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
                proof: Option::<Box<LeftAntiJoinProof>>::get(reader)?,
            },
            3 => RecursiveRelationProof::LeftOuter {
                left: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
                right: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
                proof: Box::<LeftOuterJoinProof>::get(reader)?,
            },
            4 => RecursiveRelationProof::Equality {
                input: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
                proof: Box::<RelationEqualityProof>::get(reader)?,
            },
            20 => RecursiveRelationProof::TupleDisequality {
                input: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
                proof: Box::<TupleDisequalityProof>::get(reader)?,
            },
            5 => RecursiveRelationProof::GroupExtrema {
                input: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
            },
            21 => {
                let edge = Box::new(Self::get_bounded(reader, depth + 1, nodes)?);
                let seed = TransitiveClosureSeed::get(reader)?;
                let rounds = Vec::<TransitiveClosureRound>::get(reader)?;
                if rounds.len() > MAX_TRANSITIVE_CLOSURE_ROUNDS {
                    return Err(invalid("too many transitive-closure rounds"));
                }
                let fixpoint = TransitiveClosureFixpoint::get(reader)?;
                RecursiveRelationProof::TransitiveClosure {
                    edge,
                    seed,
                    rounds,
                    fixpoint,
                }
            }
            6 => RecursiveRelationProof::ExactScalarOrder {
                input: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
                proof: Box::<ExactScalarOrderProof>::get(reader)?,
            },
            7 => RecursiveRelationProof::ExactOrderedSlice {
                input: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
            },
            8 => RecursiveRelationProof::ExactMultiOrder {
                input: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
                proof: Box::<ExactMultiOrderProof>::get(reader)?,
            },
            9 => RecursiveRelationProof::InnerJoin {
                left: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
                right: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
                proof: Box::<InnerJoinProof>::get(reader)?,
            },
            22 => RecursiveRelationProof::CrossRootHashJoin {
                left: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
                right: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
                left_hash: Box::<DictHashLookupProof>::get(reader)?,
                right_hash: Box::<DictHashLookupProof>::get(reader)?,
                relocate_closure: RelocationProof::get(reader)?,
                com_closure_at_d: ArkGT::get(reader)?,
                relocate_x: RelocationProof::get(reader)?,
                com_x_at_d: ArkGT::get(reader)?,
                relocate_h: RelocationProof::get(reader)?,
                com_h_at_d: ArkGT::get(reader)?,
                semijoin: Box::<HashSemiJoinProof>::get(reader)?,
            },
            10 => RecursiveRelationProof::BagProjection {
                input: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
            },
            11 => RecursiveRelationProof::Distinct {
                input: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
                proof: Box::<RelationDistinctProof>::get(reader)?,
            },
            _ => return Err(invalid("invalid recursive relation proof tag")),
        };
        Ok(Self {
            layer_root,
            query_digest,
            proof,
            result_commitments: Vec::<ArkGT>::get(reader)?,
            result_columns: Vec::<Vec<ArkFr>>::get(reader)?,
            result_log_domain: usize::get(reader)?,
            result_len: usize::get(reader)?,
            ground_result: bool::get(reader)?,
            result_descriptors: Vec::<ResultColumnDescriptor>::get(reader)?,
            result_values: Vec::<Vec<CanonicalResultValue>>::get(reader)?,
        })
    }

    pub fn encode(&self) -> io::Result<Vec<u8>> {
        let mut out = Vec::new();
        out.extend_from_slice(RECURSIVE_MAGIC);
        out.extend_from_slice(&RECURSIVE_VERSION.to_le_bytes());
        let mut nodes = 0;
        self.put_bounded(&mut out, 1, &mut nodes)?;
        if out.len().checked_add(CHECKSUM_BYTES).map_or(true, |len| len > MAX_QUERY_ENVELOPE_BYTES) {
            return Err(invalid("recursive relation envelope too large"));
        }
        let checksum = Sha256::digest(&out);
        out.extend_from_slice(&checksum);
        Ok(out)
    }

    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < RECURSIVE_MAGIC.len() + 4 + CHECKSUM_BYTES
            || bytes.len() > MAX_QUERY_ENVELOPE_BYTES
        {
            return Err(invalid("invalid recursive relation envelope size"));
        }
        let body_len = bytes.len() - CHECKSUM_BYTES;
        let (body, checksum) = bytes.split_at(body_len);
        if Sha256::digest(body).as_slice() != checksum {
            return Err(invalid("recursive relation envelope checksum mismatch"));
        }
        let mut reader = Reader { bytes: body, pos: 0, decoded_items: 0 };
        if reader.take(RECURSIVE_MAGIC.len())? != RECURSIVE_MAGIC {
            return Err(invalid("unknown recursive relation envelope magic"));
        }
        if u32::from_le_bytes(reader.take(4)?.try_into().unwrap()) != RECURSIVE_VERSION {
            return Err(invalid("unsupported recursive relation envelope version"));
        }
        let mut nodes = 0;
        let envelope = Self::get_bounded(&mut reader, 1, &mut nodes)?;
        reader.finish()?;
        Ok(envelope)
    }
}

#[cfg(test)]
mod tests {
    use super::super::query::prove_layer_bgp_with_result;
    use super::*;
    use crate::store::sync::open_sync_memory_store;
    use crate::ValueTriple;
    use tdb_succinct::TdbDataType;

    fn iri(value: &str) -> String {
        format!("http://ex/{value}")
    }

    fn fixture() -> (LayerCommitment, LayerBgpPlan) {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for (subject, object) in [("a", "b"), ("b", "c")] {
            builder
                .add_value_triple(ValueTriple::new_node(
                    &iri(subject),
                    &iri("knows"),
                    &iri(object),
                ))
                .unwrap();
        }
        let layer = builder.commit().unwrap();
        let commitment = layer.proof_commitment().unwrap();
        let plan = LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![LayerBgpAtom::all_variables(iri("knows").into_bytes())],
            joins: Vec::new(),
        };
        ((*commitment).clone(), plan)
    }

    fn order_fixture() -> (LayerCommitment, LayerBgpPlan) {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for value in [30u32, 10, 20] {
            builder
                .add_value_triple(ValueTriple::new_value(
                    &iri("metric"),
                    &iri("score"),
                    u32::make_entry(&value),
                ))
                .unwrap();
        }
        let layer = builder.commit().unwrap().proof_commitment().unwrap();
        let plan = LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![LayerBgpAtom {
                subject: AtomTerm::NodeConstant(iri("metric").into_bytes()),
                predicate_term: iri("score").into_bytes(),
                predicate_variable: false,
                object: AtomTerm::Variable,
                object_range: None,
                object_kind: ObjectKindFilter::Any,
                subject_object_equal: false,
                source: ColumnSource::Materialized,
            }],
            joins: Vec::new(),
        };
        ((*layer).clone(), plan)
    }

    fn multi_order_fixture() -> (LayerCommitment, LayerBgpPlan) {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for (subject, predicate, values) in [
            ("left", "x", vec![1u32, 2]),
            ("right", "y", vec![3u32, 4]),
        ] {
            for value in values {
                builder.add_value_triple(ValueTriple::new_value(
                    &iri(subject), &iri(predicate), u32::make_entry(&value),
                )).unwrap();
            }
        }
        let layer = builder.commit().unwrap().proof_commitment().unwrap();
        let atom = |subject: &str, predicate: &str| LayerBgpAtom {
            subject: AtomTerm::NodeConstant(iri(subject).into_bytes()),
            predicate_term: iri(predicate).into_bytes(),
            predicate_variable: false,
            object: AtomTerm::Variable,
            object_range: None,
            object_kind: ObjectKindFilter::Any,
            subject_object_equal: false,
            source: ColumnSource::Materialized,
        };
        ((*layer).clone(), LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![atom("left", "x"), atom("right", "y")],
            joins: vec![JoinStep::cross_product()],
        })
    }

    fn refresh_checksum(bytes: &mut Vec<u8>) {
        let body_len = bytes.len() - CHECKSUM_BYTES;
        let checksum = Sha256::digest(&bytes[..body_len]);
        bytes[body_len..].copy_from_slice(&checksum);
    }

    // ---------------------------------------------------------------------
    // CrossRootHashJoin (two-root cross-dictionary hash join) tests.
    // ---------------------------------------------------------------------

    fn node_layer(triples: &[(&str, &str, &str)]) -> LayerCommitment {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for (s, p, o) in triples {
            builder
                .add_value_triple(ValueTriple::new_node(&iri(s), &iri(p), &iri(o)))
                .unwrap();
        }
        (*builder.commit().unwrap().proof_commitment().unwrap()).clone()
    }

    fn subject_bound_atom(subject: &str, predicate: &str) -> LayerBgpAtom {
        LayerBgpAtom {
            subject: AtomTerm::NodeConstant(iri(subject).into_bytes()),
            predicate_term: iri(predicate).into_bytes(),
            predicate_variable: false,
            object: AtomTerm::Variable,
            object_range: None,
            object_kind: ObjectKindFilter::Any,
            subject_object_equal: false,
            source: ColumnSource::Materialized,
        }
    }

    fn var_subject_atom(predicate: &str) -> LayerBgpAtom {
        LayerBgpAtom {
            subject: AtomTerm::Variable,
            predicate_term: iri(predicate).into_bytes(),
            predicate_variable: false,
            object: AtomTerm::Variable,
            object_range: None,
            object_kind: ObjectKindFilter::Any,
            subject_object_equal: false,
            source: ColumnSource::Materialized,
        }
    }

    fn bgp_leaf(atom: LayerBgpAtom, digest: u8) -> RelationProofPlan {
        RelationProofPlan {
            query_digest: [digest; 32],
            kind: RelationProofPlanKind::Bgp(LayerBgpPlan {
                source: ColumnSource::Materialized,
                atoms: vec![atom],
                joins: Vec::new(),
            }),
        }
    }

    /// `left` = the SCHEMA class set (single object column of `Triple(<subj>, <p>, ?c)`);
    /// `right` = the INSTANCE `Triple(?x, <rt>, ?class)` ([x, class], key col 1).
    fn cross_root_plan(left: RelationProofPlan, right: RelationProofPlan) -> RelationProofPlan {
        RelationProofPlan {
            query_digest: [9u8; 32],
            kind: RelationProofPlanKind::CrossRootHashJoin {
                left: Box::new(left),
                right: Box::new(right),
                left_key_col: 0,
                right_key_col: 1,
            },
        }
    }

    fn class_object_id(layer: &LayerCommitment, class: &str) -> Option<u64> {
        layer
            .node_value_catalog
            .iter()
            .find(|r| matches!(&r.triple.object, crate::proof::CanonicalObject::Node(n) if n == iri(class).as_bytes()))
            .map(|r| r.object_id)
    }

    // The cross-root node reconciles ~254-bit `hash_object` keys via a FULL-FIELD LogUp semi-join
    // (`super::semijoin`) — no magnitude sort, so >=2 distinct full-field hash keys are handled.
    #[test]
    fn cross_root_matches_same_iri_across_independent_dicts() {
        // DICT-INDEPENDENCE: the SAME class IRI "Class" appears in BOTH graphs but at DIFFERENT
        // dict ids (the instance has an extra earlier node that shifts its ids). The hash join
        // still matches it (hashing the IRI, not the id); the unrelated instance class "Other" is
        // excluded. This is the completeness half of the cross-dict property.
        // The class object "Cxxxx" (sorts first, uppercase C) is dict id 1 in the schema. The
        // instance has two earlier-sorting nodes ("Aearly","Bearly") that push "Cxxxx" to id 3.
        let schema = node_layer(&[("aaa", "cls", "Cxxxx"), ("aaa", "zzz", "yyy")]);
        let instance = node_layer(&[
            ("Aearly", "rt", "Bearly"),
            ("px", "rt", "Cxxxx"),
            ("qx", "zzz", "yyy"),
        ]);
        let schema_class = class_object_id(&schema, "Cxxxx").unwrap();
        let inst_class = class_object_id(&instance, "Cxxxx").unwrap();
        assert_ne!(
            schema_class, inst_class,
            "the SAME IRI intentionally has DIFFERENT dict ids across the two graphs"
        );

        let plan = cross_root_plan(
            bgp_leaf(subject_bound_atom("aaa", "cls"), 1),
            bgp_leaf(var_subject_atom("rt"), 2),
        );
        let env = RecursiveRelationEnvelope::prove_cross_root(&instance, &schema, &plan).unwrap();
        let db_root = crate::proof::commitment::database_root(
            &instance.state.commitment_root,
            &schema.state.commitment_root,
        );
        env.verify_cross_root(&instance, &schema, db_root, &plan).unwrap();
        assert_eq!(env.result_len, 1, "only px (class Cxxxx) matches; the Aearly row is excluded");
    }

    #[test]
    fn cross_root_excludes_same_id_different_iri() {
        // THE CRUX cross-dict NEGATIVE: two DIFFERENT class IRIs collide on the SAME integer dict id
        // (identical dictionary structure, only the class IRI differs, so both classes take the same
        // sorted rank). A raw dict-id join would falsely match; the hash join MUST exclude.
        // Identical dictionary STRUCTURE (subject "aaa", filler object "yyy"); only the class IRI
        // differs and both sort FIRST, so schema "Sxxxx" and instance "Ixxxx" take the SAME dict id.
        let schema = node_layer(&[("aaa", "cls", "Sxxxx"), ("aaa", "zzz", "yyy")]);
        let instance = node_layer(&[("aaa", "rt", "Ixxxx"), ("aaa", "zzz", "yyy")]);
        let schema_class = class_object_id(&schema, "Sxxxx").unwrap();
        let inst_class = class_object_id(&instance, "Ixxxx").unwrap();
        assert_eq!(
            schema_class, inst_class,
            "the two DIFFERENT class IRIs must collide on the SAME integer dict id (the whole point)"
        );

        let plan = cross_root_plan(
            bgp_leaf(subject_bound_atom("aaa", "cls"), 1),
            bgp_leaf(var_subject_atom("rt"), 2),
        );
        let env = RecursiveRelationEnvelope::prove_cross_root(&instance, &schema, &plan).unwrap();
        let db_root = crate::proof::commitment::database_root(
            &instance.state.commitment_root,
            &schema.state.commitment_root,
        );
        env.verify_cross_root(&instance, &schema, db_root, &plan).unwrap();
        assert_eq!(
            env.result_len, 0,
            "same-integer-id-DIFFERENT-IRI must NOT match: no raw cross-dict id comparison"
        );
    }

    // The wrong-root and tamper tests validate the NOVEL two-root soundness (database_root binding,
    // per-child root authentication, reject paths) of the cross-root envelope.
    #[test]
    fn cross_root_rejects_wrong_root_and_swapped_graphs() {
        let schema = node_layer(&[("aaa", "cls", "Cxxxx"), ("aaa", "zzz", "yyy")]);
        let instance = node_layer(&[("px", "rt", "Cxxxx"), ("qx", "zzz", "yyy")]);
        let plan = cross_root_plan(
            bgp_leaf(subject_bound_atom("aaa", "cls"), 1),
            bgp_leaf(var_subject_atom("rt"), 2),
        );
        let env = RecursiveRelationEnvelope::prove_cross_root(&instance, &schema, &plan).unwrap();
        let db_root = crate::proof::commitment::database_root(
            &instance.state.commitment_root,
            &schema.state.commitment_root,
        );
        env.verify_cross_root(&instance, &schema, db_root, &plan).unwrap();

        // A tampered trusted database root rejects.
        let mut bad = db_root;
        bad[0] ^= 1;
        assert!(env.verify_cross_root(&instance, &schema, bad, &plan).is_err());

        // Swapping the two graphs (instance<->schema) under the original root rejects: the ordered
        // two-hash `database_root(instance, schema)` no longer matches, and each child is anchored to
        // the wrong root.
        assert!(env.verify_cross_root(&schema, &instance, db_root, &plan).is_err());
    }

    #[test]
    fn cross_root_rejects_tampered_join_key() {
        let schema = node_layer(&[("aaa", "cls", "Cxxxx"), ("aaa", "zzz", "yyy")]);
        let instance = node_layer(&[("px", "rt", "Cxxxx"), ("qx", "zzz", "yyy")]);
        let plan = cross_root_plan(
            bgp_leaf(subject_bound_atom("aaa", "cls"), 1),
            bgp_leaf(var_subject_atom("rt"), 2),
        );
        let mut env = RecursiveRelationEnvelope::prove_cross_root(&instance, &schema, &plan).unwrap();
        let db_root = crate::proof::commitment::database_root(
            &instance.state.commitment_root,
            &schema.state.commitment_root,
        );
        // Corrupt the join's matched key-column commitment: the inner-join projection no longer binds.
        if let RecursiveRelationProof::CrossRootHashJoin { com_closure_at_d, .. } = &mut env.proof {
            *com_closure_at_d = commit_column(&vec![ArkFr::from_u64(7); 2]).unwrap();
        }
        assert!(env.verify_cross_root(&instance, &schema, db_root, &plan).is_err());
    }

    #[test]
    fn composite_join_rejects_non_store_id_key_namespace() {
        // The fixture BGP exposes two nonnullable NodeValue (Store-ID) columns.
        let (_commitment, bgp) = fixture();
        let leaf = || RelationProofPlan {
            query_digest: [0u8; 32],
            kind: RelationProofPlanKind::Bgp(bgp.clone()),
        };
        // GroupCount yields [NodeValue key, GeneratedDecimalU64 count]; the count column is
        // NOT a bounded Store dictionary ID, so it must never reach the composite key packing.
        let group_count = || RelationProofPlan {
            query_digest: [0u8; 32],
            kind: RelationProofPlanKind::GroupCount {
                input: Box::new(leaf()),
                key_columns: vec![0],
            },
        };
        let composite = |kind| RelationProofPlan {
            query_digest: [0u8; 32],
            kind,
        };
        // Each packed join variant must reject a composite key touching the generated column.
        for kind in [
            RelationProofPlanKind::LeftAnti {
                left: Box::new(group_count()),
                right: Box::new(group_count()),
                key_pairs: vec![(0, 0), (1, 1)],
            },
            RelationProofPlanKind::InnerJoin {
                left: Box::new(group_count()),
                right: Box::new(group_count()),
                key_pairs: vec![(0, 0), (1, 1)],
            },
            RelationProofPlanKind::LeftOuter {
                left: Box::new(group_count()),
                right: Box::new(group_count()),
                key_pairs: vec![(0, 0), (1, 1)],
            },
        ] {
            assert!(
                composite(kind).shape_bounded(0, &mut 0).is_err(),
                "composite join over a generated-decimal key must reject"
            );
        }
        // Positive control: a composite join over two Store-ID (NodeValue) keys still passes.
        let ok = RelationProofPlan {
            query_digest: [0u8; 32],
            kind: RelationProofPlanKind::InnerJoin {
                left: Box::new(leaf()),
                right: Box::new(leaf()),
                key_pairs: vec![(0, 0), (1, 1)],
            },
        };
        assert!(ok.shape_bounded(0, &mut 0).is_ok());
    }

    #[test]
    fn integral_op_datatype_excludes_floats_for_extrema() {
        for datatype in [
            CanonicalDatatype::UInt8,
            CanonicalDatatype::UInt16,
            CanonicalDatatype::UInt32,
            CanonicalDatatype::UInt64,
            CanonicalDatatype::Int8,
            CanonicalDatatype::Int16,
            CanonicalDatatype::Int32,
            CanonicalDatatype::Int64,
        ] {
            assert!(integral_op_datatype(datatype));
        }
        for datatype in [
            CanonicalDatatype::Float32,
            CanonicalDatatype::Float64,
            CanonicalDatatype::Decimal,
            CanonicalDatatype::BigInt,
        ] {
            assert!(!integral_op_datatype(datatype));
        }
    }

    /// Byte-identity regression guard for the per-atom `ColumnSource` change. A multi-atom
    /// all-Materialized BGP (every atom defaults to `ColumnSource::Materialized`, plan source
    /// Materialized) must produce the EXACT SAME `LayerBgpProof` — same commitments, same
    /// transcript-derived sumcheck polynomials, same openings — as before per-atom sourcing
    /// existed. The hash below is captured from the pre-change code. If per-atom sourcing ever
    /// perturbs a Materialized proof (extra transcript bytes, re-sourced columns), this fails.
    /// (`plan_digest`/the wire image DO change — a new per-atom `source` byte — but that is the
    /// verifier's trusted plan reconstruction, not the proof's transcript/commitments/root.)
    #[test]
    fn all_materialized_two_atom_join_proof_is_byte_identical() {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for (subject, object) in [("a", "b"), ("b", "c"), ("c", "d")] {
            builder
                .add_value_triple(ValueTriple::new_node(
                    &iri(subject),
                    &iri("knows"),
                    &iri(object),
                ))
                .unwrap();
        }
        let layer = builder.commit().unwrap().proof_commitment().unwrap();
        let plan = LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![
                LayerBgpAtom::all_variables(iri("knows").into_bytes()),
                LayerBgpAtom::all_variables(iri("knows").into_bytes()),
            ],
            joins: vec![JoinStep {
                acc_key: 1,
                rel_key: 0,
                additional_keys: Vec::new(),
            }],
        };
        let result = prove_layer_bgp_with_result(&layer, &plan).unwrap();
        assert_eq!(result.result_len, 2);
        let mut bytes = Vec::new();
        result.proof.put(&mut bytes).unwrap();
        use sha2::Digest as _;
        let hash = sha2::Sha256::digest(&bytes);
        let hash_hex = hash.iter().map(|b| format!("{b:02x}")).collect::<String>();
        assert_eq!(bytes.len(), 142820, "materialized proof byte length changed");
        assert_eq!(
            hash_hex,
            "5130d5e40bd205e4ecc23e0541e3fe38175bc814c525fd7c7ced20e74e16d89d",
            "materialized multi-atom proof transcript/commitments changed (byte-identity regression)"
        );
    }

    fn recursive_leaf(
        layer: &LayerCommitment,
        plan: &LayerBgpPlan,
    ) -> (RelationProofPlan, RecursiveRelationEnvelope) {
        let result = prove_layer_bgp_with_result(layer, plan).unwrap();
        let result_columns = result.real_columns();
        let query_digest = plan_digest(plan).unwrap();
        let leaf = LayerBgpEnvelope::new_verified(
            layer,
            layer.state.commitment_root,
            plan,
            query_digest,
            result.proof,
            result.result_commitments,
            result_columns,
            result.result_log_domain,
            result.result_len,
            result.predicate_ids,
        )
        .unwrap();
        let result_descriptors = super::super::query::result_column_descriptors(plan).unwrap();
        let result_values = leaf
            .result_objects
            .iter()
            .map(|column| {
                column
                    .iter()
                    .cloned()
                    .map(CanonicalResultValue::Object)
                    .collect()
            })
            .collect();
        (
            RelationProofPlan {
                query_digest,
                kind: RelationProofPlanKind::Bgp(plan.clone()),
            },
            RecursiveRelationEnvelope {
                layer_root: layer.state.commitment_root,
                query_digest,
                result_commitments: leaf.result_commitments.clone(),
                result_columns: leaf.result_columns.clone(),
                result_log_domain: leaf.result_log_domain,
                result_len: leaf.result_len,
                ground_result: false,
                result_descriptors,
                result_values,
                proof: RecursiveRelationProof::Bgp(Box::new(leaf)),
            },
        )
    }

    #[test]
    fn dictionary_absence_position_wire_tags_are_strict() {
        for (tag, expected) in [
            (0, DictionaryAbsencePosition::Predicate),
            (1, DictionaryAbsencePosition::Subject),
            (2, DictionaryAbsencePosition::Object),
        ] {
            let mut reader = Reader {
                bytes: &[tag],
                pos: 0,
                decoded_items: 0,
            };
            assert_eq!(DictionaryAbsencePosition::get(&mut reader).unwrap(), expected);
            reader.finish().unwrap();
        }
        let mut reader = Reader {
            bytes: &[3],
            pos: 0,
            decoded_items: 0,
        };
        assert!(DictionaryAbsencePosition::get(&mut reader).is_err());
    }

    #[test]
    fn recursive_exact_scalar_order_roundtrips_and_rejects_plan_tamper() {
        let (layer, bgp) = order_fixture();
        let child = RelationProofPlan {
            query_digest: plan_digest(&bgp).unwrap(),
            kind: RelationProofPlanKind::Bgp(bgp),
        };
        let plan = RelationProofPlan {
            query_digest: [42u8; 32],
            kind: RelationProofPlanKind::ExactScalarOrder {
                input: Box::new(child),
                datatype: CanonicalDatatype::UInt32,
                direction: OrderDirection::Ascending,
            },
        };
        let envelope = RecursiveRelationEnvelope::prove(&layer, &plan).unwrap();
        envelope
            .verify(&layer, layer.state.commitment_root, &plan)
            .unwrap();
        let bytes = envelope.encode().unwrap();
        let decoded = RecursiveRelationEnvelope::decode(&bytes).unwrap();
        decoded
            .verify(&layer, layer.state.commitment_root, &plan)
            .unwrap();

        let mut wrong_plan = plan.clone();
        if let RelationProofPlanKind::ExactScalarOrder { datatype, .. } = &mut wrong_plan.kind {
            *datatype = CanonicalDatatype::Int32;
        }
        assert!(decoded
            .verify(&layer, layer.state.commitment_root, &wrong_plan)
            .is_err());
    }

    fn assert_recursive_inner_join_roundtrips_and_rejects_tamper() {
        let (layer, bgp) = fixture();
        let leaf = || RelationProofPlan {
            query_digest: plan_digest(&bgp).unwrap(),
            kind: RelationProofPlanKind::Bgp(bgp.clone()),
        };
        let plan = RelationProofPlan {
            query_digest: [73u8; 32],
            kind: RelationProofPlanKind::InnerJoin {
                left: Box::new(leaf()), right: Box::new(leaf()),
                key_pairs: vec![(0, 0), (1, 1)],
            },
        };
        let envelope = RecursiveRelationEnvelope::prove(&layer, &plan).unwrap();
        envelope.verify(&layer, layer.state.commitment_root, &plan).unwrap();
        assert_eq!(envelope.result_len, 2);
        assert_eq!(envelope.result_columns.len(), 2);
        let bytes = envelope.encode().unwrap();
        let decoded = RecursiveRelationEnvelope::decode(&bytes).unwrap();
        decoded.verify(&layer, layer.state.commitment_root, &plan).unwrap();

        let mut wrong_keys = plan.clone();
        if let RelationProofPlanKind::InnerJoin { key_pairs, .. } = &mut wrong_keys.kind {
            key_pairs[1] = (0, 1);
        }
        assert!(decoded.verify(&layer, layer.state.commitment_root, &wrong_keys).is_err());
        let mut changed = RecursiveRelationEnvelope::decode(&bytes).unwrap();
        changed.result_columns[0].swap(0, 1);
        for (commitment, column) in changed.result_commitments.iter_mut().zip(&changed.result_columns) {
            *commitment = commit_column(column).unwrap();
        }
        assert!(changed.verify(&layer, layer.state.commitment_root, &plan).is_err());
    }

    #[test]
    fn recursive_inner_join_roundtrips_and_rejects_tamper() {
        std::thread::Builder::new().name("inner-join-proof".into()).stack_size(256 * 1024 * 1024)
            .spawn(assert_recursive_inner_join_roundtrips_and_rejects_tamper).unwrap().join().unwrap();
    }

    #[test]
    fn recursive_bag_projection_preserves_duplicate_rows_and_rejects_tamper() {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for object in ["x", "y"] {
            builder
                .add_value_triple(ValueTriple::new_node(
                    &iri("a"),
                    &iri("p"),
                    &iri(object),
                ))
                .unwrap();
        }
        let layer = builder.commit().unwrap().proof_commitment().unwrap();
        let bgp = LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![LayerBgpAtom::all_variables(iri("p").into_bytes())],
            joins: Vec::new(),
        };
        let leaf = RelationProofPlan {
            query_digest: plan_digest(&bgp).unwrap(),
            kind: RelationProofPlanKind::Bgp(bgp),
        };
        let plan = RelationProofPlan {
            query_digest: [91u8; 32],
            kind: RelationProofPlanKind::BagProjection {
                input: Box::new(leaf),
                indices: vec![0],
            },
        };
        let envelope = RecursiveRelationEnvelope::prove(&layer, &plan).unwrap();
        envelope
            .verify(&layer, layer.state.commitment_root, &plan)
            .unwrap();
        assert_eq!(envelope.result_len, 2);
        assert_eq!(envelope.result_columns.len(), 1);
        assert_eq!(envelope.result_columns[0][0], envelope.result_columns[0][1]);

        let encoded = envelope.encode().unwrap();
        let decoded = RecursiveRelationEnvelope::decode(&encoded).unwrap();
        decoded
            .verify(&layer, layer.state.commitment_root, &plan)
            .unwrap();

        let mut wrong_index = plan.clone();
        if let RelationProofPlanKind::BagProjection { indices, .. } = &mut wrong_index.kind {
            indices[0] = 1;
        }
        assert!(decoded
            .verify(&layer, layer.state.commitment_root, &wrong_index)
            .is_err());

        let mut changed = decoded.clone();
        changed.result_columns[0][0] = match &changed.proof {
            RecursiveRelationProof::BagProjection { input } => input.result_columns[1][0],
            _ => unreachable!(),
        };
        changed.result_commitments[0] = commit_column(&changed.result_columns[0]).unwrap();
        assert!(changed
            .verify(&layer, layer.state.commitment_root, &plan)
            .is_err());

        let duplicate_indices = RelationProofPlan {
            query_digest: [92u8; 32],
            kind: RelationProofPlanKind::BagProjection {
                input: match &plan.kind {
                    RelationProofPlanKind::BagProjection { input, .. } => input.clone(),
                    _ => unreachable!(),
                },
                indices: vec![0, 0],
            },
        };
        assert!(duplicate_indices.result_shape().is_err());
    }

    #[test]
    fn recursive_empty_bag_projection_emits_exact_ground_unit_and_handles_empty_child() {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for object in ["x", "y"] {
            builder
                .add_value_triple(ValueTriple::new_node(
                    &iri("a"),
                    &iri("p"),
                    &iri(object),
                ))
                .unwrap();
        }
        let layer = builder.commit().unwrap().proof_commitment().unwrap();
        let project = |bgp: LayerBgpPlan, digest| RelationProofPlan {
            query_digest: digest,
            kind: RelationProofPlanKind::BagProjection {
                input: Box::new(RelationProofPlan {
                    query_digest: plan_digest(&bgp).unwrap(),
                    kind: RelationProofPlanKind::Bgp(bgp),
                }),
                indices: Vec::new(),
            },
        };

        let present = project(
            LayerBgpPlan {
                source: ColumnSource::Materialized,
                atoms: vec![LayerBgpAtom::all_variables(iri("p").into_bytes())],
                joins: Vec::new(),
            },
            [93u8; 32],
        );
        let envelope = RecursiveRelationEnvelope::prove(&layer, &present).unwrap();
        envelope
            .verify(&layer, layer.state.commitment_root, &present)
            .unwrap();
        assert!(envelope.ground_result);
        assert_eq!(envelope.result_len, 2);
        assert_eq!(envelope.result_columns, vec![vec![ArkFr::one(); 2]]);
        assert!(envelope.result_descriptors.is_empty());
        assert!(envelope.result_values.is_empty());
        let bytes = envelope.encode().unwrap();
        let mut changed = RecursiveRelationEnvelope::decode(&bytes).unwrap();
        changed.result_columns[0][0] = ArkFr::zero();
        let mut padded = changed.result_columns[0].clone();
        padded.resize(1usize << changed.result_log_domain, ArkFr::zero());
        changed.result_commitments[0] = commit_column(&padded).unwrap();
        assert!(changed
            .verify(&layer, layer.state.commitment_root, &present)
            .is_err());

        let absent = project(
            LayerBgpPlan {
                source: ColumnSource::Materialized,
                atoms: vec![LayerBgpAtom {
                    subject: AtomTerm::NodeConstant(iri("missing").into_bytes()),
                    predicate_term: iri("p").into_bytes(),
                    predicate_variable: false,
                    object: AtomTerm::Variable,
                    object_range: None,
                    object_kind: ObjectKindFilter::Any,
                    subject_object_equal: false,
                    source: ColumnSource::Materialized,
                }],
                joins: Vec::new(),
            },
            [94u8; 32],
        );
        let empty = RecursiveRelationEnvelope::prove(&layer, &absent).unwrap();
        empty
            .verify(&layer, layer.state.commitment_root, &absent)
            .unwrap();
        assert!(empty.ground_result);
        assert_eq!(empty.result_len, 0);
        assert_eq!(empty.result_columns, vec![Vec::new()]);
    }

    #[test]
    fn recursive_layer_triple_count_tracks_logical_stack_and_rejects_tamper() {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        let empty_layer = builder.commit().unwrap();
        let plan = RelationProofPlan {
            query_digest: [95u8; 32],
            kind: RelationProofPlanKind::LayerTripleCount,
        };
        let empty_commitment = empty_layer.proof_commitment().unwrap();
        let empty = RecursiveRelationEnvelope::prove(&empty_commitment, &plan).unwrap();
        empty
            .verify(&empty_commitment, empty_commitment.state.commitment_root, &plan)
            .unwrap();
        assert_eq!(empty.result_len, 1);
        assert_eq!(empty.result_columns, vec![vec![ArkFr::zero()]]);
        assert_eq!(
            empty.result_values,
            vec![vec![CanonicalResultValue::GeneratedDecimalU64(0)]]
        );

        let builder = empty_layer.open_write().unwrap();
        for object in ["x", "y", "z"] {
            builder
                .add_value_triple(ValueTriple::new_node(
                    &iri("a"),
                    &iri("p"),
                    &iri(object),
                ))
                .unwrap();
        }
        let parent = builder.commit().unwrap();
        let builder = parent.open_write().unwrap();
        builder
            .remove_value_triple(ValueTriple::new_node(&iri("a"), &iri("p"), &iri("y")))
            .unwrap();
        builder
            .add_value_triple(ValueTriple::new_node(&iri("b"), &iri("q"), &iri("w")))
            .unwrap();
        let head = builder.commit().unwrap().proof_commitment().unwrap();
        // COMMITMENT DIET (Wave 0): a DELTA head no longer commits its materialized head
        // view or its triple count (materialized_count is now 0 and NOT root-bound), so
        // LayerTripleCount is not directly provable over a delta head and reject-closes on
        // BOTH prove and verify. The reconciled delta-stack path (Wave 2/3) will re-enable
        // logical-stack counts by authenticating the delta chain to the head root.
        assert!(head.state.parent_state_root.is_some());
        assert_eq!(head.state.materialized_count, 0);
        assert!(RecursiveRelationEnvelope::prove(&head, &plan).is_err());
        // A well-formed empty-base envelope must also refuse to VERIFY against a delta head.
        assert!(empty
            .verify(&head, head.state.commitment_root, &plan)
            .is_err());
    }

    #[test]
    fn recursive_generated_u64_filter_derives_exact_comparison_and_rejects_tamper() {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for object in ["x", "y", "z"] {
            builder
                .add_value_triple(ValueTriple::new_node(
                    &iri("a"),
                    &iri("p"),
                    &iri(object),
                ))
                .unwrap();
        }
        let layer = builder.commit().unwrap().proof_commitment().unwrap();
        let source = RelationProofPlan {
            query_digest: [97u8; 32],
            kind: RelationProofPlanKind::LayerTripleCount,
        };
        let plan = |op, constant, constant_on_left| RelationProofPlan {
            query_digest: [98u8; 32],
            kind: RelationProofPlanKind::GeneratedU64Filter {
                input: Box::new(source.clone()),
                column: 0,
                op,
                constant,
                constant_on_left,
            },
        };

        for (op, constant, constant_on_left, expected_len) in [
            (GeneratedU64ComparisonOp::Eq, 3, false, 1),
            (GeneratedU64ComparisonOp::Eq, 2, false, 0),
            (GeneratedU64ComparisonOp::Gt, 0, false, 1),
            (GeneratedU64ComparisonOp::Ge, 3, false, 1),
            (GeneratedU64ComparisonOp::Lt, u64::MAX, false, 1),
            (GeneratedU64ComparisonOp::Le, u64::MAX, false, 1),
            (GeneratedU64ComparisonOp::Lt, 3, true, 0),
            (GeneratedU64ComparisonOp::Le, 3, true, 1),
        ] {
            let plan = plan(op, constant, constant_on_left);
            let envelope = RecursiveRelationEnvelope::prove(&layer, &plan).unwrap();
            envelope
                .verify(&layer, layer.state.commitment_root, &plan)
                .unwrap();
            assert_eq!(envelope.result_len, expected_len);
            let decoded = RecursiveRelationEnvelope::decode(&envelope.encode().unwrap()).unwrap();
            decoded
                .verify(&layer, layer.state.commitment_root, &plan)
                .unwrap();
        }

        let true_plan = plan(GeneratedU64ComparisonOp::Ge, 3, false);
        let envelope = RecursiveRelationEnvelope::prove(&layer, &true_plan).unwrap();
        let wrong_operator = plan(GeneratedU64ComparisonOp::Gt, 3, false);
        assert!(envelope
            .verify(&layer, layer.state.commitment_root, &wrong_operator)
            .is_err());
        let wrong_operand_order = plan(GeneratedU64ComparisonOp::Lt, 3, true);
        assert!(envelope
            .verify(&layer, layer.state.commitment_root, &wrong_operand_order)
            .is_err());

        let mut tampered = envelope.clone();
        tampered.result_columns[0][0] = ArkFr::from_u64(4);
        tampered.result_commitments[0] =
            commit_column(&[ArkFr::from_u64(4), ArkFr::zero()]).unwrap();
        tampered.result_values =
            vec![vec![CanonicalResultValue::GeneratedDecimalU64(4)]];
        assert!(tampered
            .verify(&layer, layer.state.commitment_root, &true_plan)
            .is_err());

        let empty_store = open_sync_memory_store();
        let empty_layer = empty_store
            .create_base_layer()
            .unwrap()
            .commit()
            .unwrap()
            .proof_commitment()
            .unwrap();
        let zero = plan(GeneratedU64ComparisonOp::Eq, 0, false);
        let zero_envelope = RecursiveRelationEnvelope::prove(&empty_layer, &zero).unwrap();
        zero_envelope
            .verify(&empty_layer, empty_layer.state.commitment_root, &zero)
            .unwrap();
        assert_eq!(zero_envelope.result_len, 1);
        assert_eq!(
            zero_envelope.result_values,
            vec![vec![CanonicalResultValue::GeneratedDecimalU64(0)]]
        );
    }

    #[test]
    fn recursive_group_sum_preserves_duplicates_zero_and_safe_u32_max_and_rejects_tamper() {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for (predicate, value) in [("p", 0u32), ("q", u32::MAX)] {
            builder
                .add_value_triple(ValueTriple::new_value(
                    &iri("b"),
                    &iri(predicate),
                    u32::make_entry(&value),
                ))
                .unwrap();
        }
        let base = builder.commit().unwrap();
        let builder = base.open_write().unwrap();
        // The lexically earlier key is introduced in a child and therefore has a later
        // Store ID. Output must still follow SWI atom order (`a` before `b`).
        for predicate in ["p", "q"] {
            builder
                .add_value_triple(ValueTriple::new_value(
                    &iri("a"),
                    &iri(predicate),
                    u32::make_entry(&2u32),
                ))
                .unwrap();
        }
        let layer = builder.commit().unwrap().proof_commitment().unwrap();
        let source = RelationProofPlan {
            query_digest: [99u8; 32],
            kind: RelationProofPlanKind::Bgp(LayerBgpPlan {
                source: ColumnSource::Materialized,
                atoms: vec![LayerBgpAtom {
                    subject: AtomTerm::Variable,
                    predicate_term: Vec::new(),
                    predicate_variable: true,
                    object: AtomTerm::Variable,
                    object_range: None,
                    object_kind: ObjectKindFilter::Any,
                    subject_object_equal: false,
                    source: ColumnSource::Materialized,
                }],
                joins: Vec::new(),
            }),
        };
        let plan = RelationProofPlan {
            query_digest: [100u8; 32],
            kind: RelationProofPlanKind::GroupSum {
                input: Box::new(source.clone()),
                key_columns: vec![0],
                value_column: 2,
                datatype: CanonicalDatatype::UInt32,
            },
        };
        // COMMITMENT DIET (Wave 0): this GroupSum reads the DELTA head's materialized view,
        // which is no longer committed — proving reject-closes at the envelope entry. The
        // reconciled delta-stack path (Wave 2/3) will re-enable delta-head aggregates over an
        // authenticated delta chain. (`source`/`plan` build a materialized-source aggregate.)
        assert!(layer.state.parent_state_root.is_some());
        assert!(layer.materialized.is_empty());
        let _ = &source;
        assert!(RecursiveRelationEnvelope::prove(&layer, &plan).is_err());
    }

    #[test]
    fn recursive_two_key_group_sum_uses_declared_lexical_tuple_order() {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        builder
            .add_value_triple(ValueTriple::new_node(&iri("z"), &iri("link"), &iri("z")))
            .unwrap();
        for value in [1u32, 2] {
            builder
                .add_value_triple(ValueTriple::new_value(
                    &iri("z"),
                    &iri("score"),
                    u32::make_entry(&value),
                ))
                .unwrap();
        }
        let base = builder.commit().unwrap();
        let builder = base.open_write().unwrap();
        for (left, right) in [("a", "a"), ("a", "z"), ("z", "a")] {
            builder
                .add_value_triple(ValueTriple::new_node(
                    &iri(left),
                    &iri("link"),
                    &iri(right),
                ))
                .unwrap();
        }
        for value in [3u32, 4, 5] {
            builder
                .add_value_triple(ValueTriple::new_value(
                    &iri("a"),
                    &iri("score"),
                    u32::make_entry(&value),
                ))
                .unwrap();
        }
        let layer = builder.commit().unwrap().proof_commitment().unwrap();
        let source = RelationProofPlan {
            query_digest: [112u8; 32],
            kind: RelationProofPlanKind::Bgp(LayerBgpPlan {
                source: ColumnSource::Materialized,
                atoms: vec![
                    LayerBgpAtom::all_variables(iri("link").into_bytes()),
                    LayerBgpAtom::all_variables(iri("score").into_bytes()),
                ],
                joins: vec![JoinStep {
                    acc_key: 1,
                    rel_key: 0,
                    additional_keys: Vec::new(),
                }],
            }),
        };
        let plan = RelationProofPlan {
            query_digest: [113u8; 32],
            kind: RelationProofPlanKind::GroupSum {
                input: Box::new(source.clone()),
                // Child schema is [left,right,value]; declared group priority is
                // deliberately [right,left].
                key_columns: vec![1, 0],
                value_column: 2,
                datatype: CanonicalDatatype::UInt32,
            },
        };
        // COMMITMENT DIET (Wave 0): this two-key GroupSum reads the DELTA head's materialized
        // view, which is no longer committed — proving reject-closes at the envelope entry.
        // Reconciliation (Wave 2/3) will re-enable delta-head aggregates.
        assert!(layer.state.parent_state_root.is_some());
        assert!(layer.materialized.is_empty());
        let _ = &source;
        assert!(RecursiveRelationEnvelope::prove(&layer, &plan).is_err());
    }

    fn assert_recursive_two_key_group_extrema_uses_declared_node_tuple_order() {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        builder
            .add_value_triple(ValueTriple::new_node(&iri("z"), &iri("link"), &iri("z")))
            .unwrap();
        for value in [1u32, 2] {
            builder
                .add_value_triple(ValueTriple::new_value(
                    &iri("z"), &iri("score"), u32::make_entry(&value),
                ))
                .unwrap();
        }
        let base = builder.commit().unwrap();
        let builder = base.open_write().unwrap();
        for (left, right) in [("a", "a"), ("a", "z"), ("z", "a")] {
            builder
                .add_value_triple(ValueTriple::new_node(
                    &iri(left), &iri("link"), &iri(right),
                ))
                .unwrap();
        }
        for value in [3u32, 4, 5] {
            builder
                .add_value_triple(ValueTriple::new_value(
                    &iri("a"), &iri("score"), u32::make_entry(&value),
                ))
                .unwrap();
        }
        let layer = builder.commit().unwrap().proof_commitment().unwrap();
        let source = RelationProofPlan {
            query_digest: [131u8; 32],
            kind: RelationProofPlanKind::Bgp(LayerBgpPlan {
                source: ColumnSource::Materialized,
                atoms: vec![
                    LayerBgpAtom::all_variables(iri("link").into_bytes()),
                    LayerBgpAtom::all_variables(iri("score").into_bytes()),
                ],
                joins: vec![JoinStep {
                    acc_key: 1,
                    rel_key: 0,
                    additional_keys: Vec::new(),
                }],
            }),
        };
        let plan = RelationProofPlan {
            query_digest: [132u8; 32],
            kind: RelationProofPlanKind::GroupExtrema {
                input: Box::new(source.clone()),
                key_columns: vec![1, 0],
                value_column: 2,
                datatype: CanonicalDatatype::UInt32,
                kind: GroupExtremaKind::Min,
            },
        };
        // COMMITMENT DIET (Wave 0): this two-key GroupExtrema reads the DELTA head's
        // materialized view, which is no longer committed — proving reject-closes at the
        // envelope entry. Reconciliation (Wave 2/3) will re-enable delta-head aggregates.
        assert!(layer.state.parent_state_root.is_some());
        assert!(layer.materialized.is_empty());
        let _ = &source;
        assert!(RecursiveRelationEnvelope::prove(&layer, &plan).is_err());
    }

    #[test]
    fn recursive_two_key_group_extrema_uses_declared_node_tuple_order() {
        std::thread::Builder::new()
            .name("recursive-two-key-group-extrema".into())
            .stack_size(256 * 1024 * 1024)
            .spawn(assert_recursive_two_key_group_extrema_uses_declared_node_tuple_order)
            .unwrap()
            .join()
            .unwrap();
    }

    fn assert_recursive_zero_key_group_extrema_emits_only_for_nonempty_input() {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for (subject, value) in [("a", i64::MIN), ("b", -3), ("c", -3), ("d", i64::MAX)] {
            builder.add_value_triple(ValueTriple::new_value(
                &iri(subject), &iri("score"), i64::make_entry(&value),
            )).unwrap();
        }
        let layer = builder.commit().unwrap().proof_commitment().unwrap();
        let source = |predicate: &str, digest| RelationProofPlan {
            query_digest: digest,
            kind: RelationProofPlanKind::Bgp(LayerBgpPlan {
                source: ColumnSource::Materialized,
                atoms: vec![LayerBgpAtom::all_variables(iri(predicate).into_bytes())],
                joins: Vec::new(),
            }),
        };
        let plan = RelationProofPlan {
            query_digest: [134u8; 32],
            kind: RelationProofPlanKind::GroupExtrema {
                input: Box::new(source("score", [135u8; 32])),
                key_columns: Vec::new(),
                value_column: 1,
                datatype: CanonicalDatatype::Int64,
                kind: GroupExtremaKind::Max,
            },
        };
        let envelope = RecursiveRelationEnvelope::prove(&layer, &plan).unwrap();
        envelope.verify(&layer, layer.state.commitment_root, &plan).unwrap();
        assert_eq!(envelope.result_len, 1);
        let expected = CanonicalResultValue::Object(CanonicalObject::from_store_object(
            &ValueTriple::new_value("", "", i64::make_entry(&i64::MAX)).object,
        ));
        assert_eq!(envelope.result_values, vec![vec![expected]]);

        let empty_plan = RelationProofPlan {
            query_digest: [136u8; 32],
            kind: RelationProofPlanKind::GroupExtrema {
                input: Box::new(source("missing", [137u8; 32])),
                key_columns: Vec::new(),
                value_column: 1,
                datatype: CanonicalDatatype::Int64,
                kind: GroupExtremaKind::Min,
            },
        };
        let empty = RecursiveRelationEnvelope::prove(&layer, &empty_plan).unwrap();
        empty.verify(&layer, layer.state.commitment_root, &empty_plan).unwrap();
        assert_eq!(empty.result_len, 0);
        assert_eq!(empty.result_values, vec![Vec::new()]);
    }


    #[test]
    fn recursive_zero_key_group_extrema_emits_only_for_nonempty_input() {
        std::thread::Builder::new()
            .name("recursive-zero-key-group-extrema".into())
            .stack_size(256 * 1024 * 1024)
            .spawn(assert_recursive_zero_key_group_extrema_emits_only_for_nonempty_input)
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn recursive_two_key_group_count_preserves_bag_cardinality_and_tuple_order() {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        builder
            .add_value_triple(ValueTriple::new_node(&iri("z"), &iri("link"), &iri("z")))
            .unwrap();
        for value in [1u32, 2] {
            builder
                .add_value_triple(ValueTriple::new_value(
                    &iri("z"),
                    &iri("score"),
                    u32::make_entry(&value),
                ))
                .unwrap();
        }
        let base = builder.commit().unwrap();
        let builder = base.open_write().unwrap();
        for (left, right) in [("a", "a"), ("a", "z"), ("z", "a")] {
            builder
                .add_value_triple(ValueTriple::new_node(
                    &iri(left),
                    &iri("link"),
                    &iri(right),
                ))
                .unwrap();
        }
        for value in [3u32, 4, 5] {
            builder
                .add_value_triple(ValueTriple::new_value(
                    &iri("a"),
                    &iri("score"),
                    u32::make_entry(&value),
                ))
                .unwrap();
        }
        let layer = builder.commit().unwrap().proof_commitment().unwrap();
        let source = RelationProofPlan {
            query_digest: [117u8; 32],
            kind: RelationProofPlanKind::Bgp(LayerBgpPlan {
                source: ColumnSource::Materialized,
                atoms: vec![
                    LayerBgpAtom::all_variables(iri("link").into_bytes()),
                    LayerBgpAtom::all_variables(iri("score").into_bytes()),
                ],
                joins: vec![JoinStep {
                    acc_key: 1,
                    rel_key: 0,
                    additional_keys: Vec::new(),
                }],
            }),
        };
        let plan = RelationProofPlan {
            query_digest: [118u8; 32],
            kind: RelationProofPlanKind::GroupCount {
                input: Box::new(source.clone()),
                key_columns: vec![1, 0],
            },
        };
        // COMMITMENT DIET (Wave 0): this two-key GroupCount reads the DELTA head's
        // materialized view, which is no longer committed — proving reject-closes at the
        // envelope entry. Reconciliation (Wave 2/3) will re-enable delta-head aggregates.
        assert!(layer.state.parent_state_root.is_some());
        assert!(layer.materialized.is_empty());
        let _ = &source;
        assert!(RecursiveRelationEnvelope::prove(&layer, &plan).is_err());
    }

    #[test]
    fn recursive_collection_count_emits_zero_for_empty_and_rejects_group_substitution() {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for object in ["x", "y", "z"] {
            builder
                .add_value_triple(ValueTriple::new_node(
                    &iri("a"),
                    &iri("p"),
                    &iri(object),
                ))
                .unwrap();
        }
        let layer = builder.commit().unwrap().proof_commitment().unwrap();
        let source = |predicate: &str, digest| RelationProofPlan {
            query_digest: digest,
            kind: RelationProofPlanKind::Bgp(LayerBgpPlan {
                source: ColumnSource::Materialized,
                atoms: vec![LayerBgpAtom::all_variables(iri(predicate).into_bytes())],
                joins: Vec::new(),
            }),
        };
        let plan = RelationProofPlan {
            query_digest: [120u8; 32],
            kind: RelationProofPlanKind::CollectionCount {
                input: Box::new(source("p", [121u8; 32])),
            },
        };
        let envelope = RecursiveRelationEnvelope::prove(&layer, &plan).unwrap();
        envelope
            .verify(&layer, layer.state.commitment_root, &plan)
            .unwrap();
        assert_eq!(
            envelope.result_values,
            vec![vec![CanonicalResultValue::GeneratedDecimalU64(3)]]
        );
        let decoded = RecursiveRelationEnvelope::decode(&envelope.encode().unwrap()).unwrap();
        decoded
            .verify(&layer, layer.state.commitment_root, &plan)
            .unwrap();
        let group_plan = RelationProofPlan {
            query_digest: [120u8; 32],
            kind: RelationProofPlanKind::GroupCount {
                input: Box::new(source("p", [121u8; 32])),
                key_columns: Vec::new(),
            },
        };
        assert!(decoded
            .verify(&layer, layer.state.commitment_root, &group_plan)
            .is_err());
        let mut tampered = decoded;
        tampered.result_columns[0][0] = ArkFr::from_u64(4);
        tampered.result_values[0][0] = CanonicalResultValue::GeneratedDecimalU64(4);
        tampered.result_commitments[0] =
            commit_column(&[ArkFr::from_u64(4), ArkFr::zero()]).unwrap();
        assert!(tampered
            .verify(&layer, layer.state.commitment_root, &plan)
            .is_err());

        let empty_plan = RelationProofPlan {
            query_digest: [122u8; 32],
            kind: RelationProofPlanKind::CollectionCount {
                input: Box::new(source("missing", [123u8; 32])),
            },
        };
        let empty = RecursiveRelationEnvelope::prove(&layer, &empty_plan).unwrap();
        empty
            .verify(&layer, layer.state.commitment_root, &empty_plan)
            .unwrap();
        assert_eq!(empty.result_len, 1);
        assert_eq!(
            empty.result_values,
            vec![vec![CanonicalResultValue::GeneratedDecimalU64(0)]]
        );
    }

    #[test]
    fn recursive_collection_sum_handles_signed_cancellation_empty_and_domain_bounds() {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for value in [-3i64, 1, 2] {
            builder
                .add_value_triple(ValueTriple::new_value(
                    &iri("a"),
                    &iri("p"),
                    i64::make_entry(&value),
                ))
                .unwrap();
        }
        let layer = builder.commit().unwrap().proof_commitment().unwrap();
        let source = RelationProofPlan {
            query_digest: [124u8; 32],
            kind: RelationProofPlanKind::Bgp(LayerBgpPlan {
                source: ColumnSource::Materialized,
                atoms: vec![LayerBgpAtom::all_variables(iri("p").into_bytes())],
                joins: Vec::new(),
            }),
        };
        let plan = RelationProofPlan {
            query_digest: [125u8; 32],
            kind: RelationProofPlanKind::CollectionSum {
                input: Box::new(source.clone()),
                template: CollectionSumTemplate::Column(1),
                datatype: CanonicalDatatype::Int64,
            },
        };
        let envelope = RecursiveRelationEnvelope::prove(&layer, &plan).unwrap();
        envelope
            .verify(&layer, layer.state.commitment_root, &plan)
            .unwrap();
        assert_eq!(
            envelope.result_values,
            vec![vec![CanonicalResultValue::GeneratedDecimalI64(0)]]
        );
        let decoded = RecursiveRelationEnvelope::decode(&envelope.encode().unwrap()).unwrap();
        decoded
            .verify(&layer, layer.state.commitment_root, &plan)
            .unwrap();
        let group_plan = RelationProofPlan {
            query_digest: [125u8; 32],
            kind: RelationProofPlanKind::GroupSum {
                input: Box::new(source.clone()),
                key_columns: Vec::new(),
                value_column: 1,
                datatype: CanonicalDatatype::Int64,
            },
        };
        assert!(decoded
            .verify(&layer, layer.state.commitment_root, &group_plan)
            .is_err());
        let count_plan = RelationProofPlan {
            query_digest: [125u8; 32],
            kind: RelationProofPlanKind::CollectionCount {
                input: Box::new(source),
            },
        };
        assert!(decoded
            .verify(&layer, layer.state.commitment_root, &count_plan)
            .is_err());
        let mut tampered = decoded;
        tampered.result_columns[0][0] = generated_decimal_i64_field(1);
        tampered.result_values[0][0] = CanonicalResultValue::GeneratedDecimalI64(1);
        tampered.result_commitments[0] =
            commit_column(&[generated_decimal_i64_field(1), ArkFr::zero()]).unwrap();
        assert!(tampered
            .verify(&layer, layer.state.commitment_root, &plan)
            .is_err());

        let RecursiveRelationProof::CollectionSum { input } = &envelope.proof else {
            unreachable!()
        };
        let mut one = (**input).clone();
        one.result_len = 1;
        assert_eq!(
            derived_collection_sum_column(
                &layer,
                &one,
                CollectionSumTemplate::UnsignedConstant(u64::MAX),
                CanonicalDatatype::UInt64,
            )
            .unwrap(),
            vec![vec![ArkFr::from_u64(u64::MAX)]]
        );
        assert!(derived_collection_sum_column(
            &layer,
            input,
            CollectionSumTemplate::UnsignedConstant(u64::MAX),
            CanonicalDatatype::UInt64,
        )
        .is_err());
        assert!(derived_collection_sum_column(
            &layer,
            input,
            CollectionSumTemplate::SignedConstant(i64::MAX),
            CanonicalDatatype::Int64,
        )
        .is_err());
        let mut empty = (**input).clone();
        empty.result_len = 0;
        assert_eq!(
            derived_collection_sum_column(
                &layer,
                &empty,
                CollectionSumTemplate::SignedConstant(-5),
                CanonicalDatatype::Int64,
            )
            .unwrap(),
            vec![vec![ArkFr::zero()]]
        );
    }

    #[test]
    fn recursive_collection_extrema_preserves_original_id_and_empty_has_no_row() {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for (subject, value) in [
            ("a", i64::MIN),
            ("b", -3),
            ("c", -3),
            ("d", 2),
            ("e", i64::MAX),
        ] {
            builder
                .add_value_triple(ValueTriple::new_value(
                    &iri(subject),
                    &iri("p"),
                    i64::make_entry(&value),
                ))
                .unwrap();
        }
        let layer = builder.commit().unwrap().proof_commitment().unwrap();
        let source = |predicate: &str, digest| RelationProofPlan {
            query_digest: digest,
            kind: RelationProofPlanKind::Bgp(LayerBgpPlan {
                source: ColumnSource::Materialized,
                atoms: vec![LayerBgpAtom::all_variables(iri(predicate).into_bytes())],
                joins: Vec::new(),
            }),
        };
        let extrema = |kind, digest| RelationProofPlan {
            query_digest: digest,
            kind: RelationProofPlanKind::CollectionExtrema {
                input: Box::new(source("p", [126u8; 32])),
                value_column: 1,
                datatype: CanonicalDatatype::Int64,
                kind,
            },
        };
        let min_plan = extrema(GroupExtremaKind::Min, [127u8; 32]);
        let min = RecursiveRelationEnvelope::prove(&layer, &min_plan).unwrap();
        min.verify(&layer, layer.state.commitment_root, &min_plan)
            .unwrap();
        assert_eq!(min.result_len, 1);
        let expected_min = CanonicalObject::from_store_object(
            &ValueTriple::new_value("", "", i64::make_entry(&i64::MIN)).object,
        );
        assert_eq!(
            min.result_values,
            vec![vec![CanonicalResultValue::Object(expected_min)]]
        );
        let decoded = RecursiveRelationEnvelope::decode(&min.encode().unwrap()).unwrap();
        decoded
            .verify(&layer, layer.state.commitment_root, &min_plan)
            .unwrap();

        let max_plan = extrema(GroupExtremaKind::Max, [128u8; 32]);
        let max = RecursiveRelationEnvelope::prove(&layer, &max_plan).unwrap();
        max.verify(&layer, layer.state.commitment_root, &max_plan)
            .unwrap();
        assert_eq!(max.result_len, 1);
        assert!(decoded
            .verify(&layer, layer.state.commitment_root, &max_plan)
            .is_err());

        let RecursiveRelationProof::CollectionExtrema { input } = &decoded.proof else {
            unreachable!()
        };
        let non_extreme = input.result_columns[1][1];
        let non_extreme_value = input.result_values[1][1].clone();
        let mut tampered = decoded;
        tampered.result_columns[0][0] = non_extreme;
        tampered.result_values[0][0] = non_extreme_value;
        tampered.result_commitments[0] =
            commit_column(&[non_extreme, ArkFr::zero()]).unwrap();
        assert!(tampered
            .verify(&layer, layer.state.commitment_root, &min_plan)
            .is_err());

        let empty_plan = RelationProofPlan {
            query_digest: [129u8; 32],
            kind: RelationProofPlanKind::CollectionExtrema {
                input: Box::new(source("missing", [130u8; 32])),
                value_column: 1,
                datatype: CanonicalDatatype::Int64,
                kind: GroupExtremaKind::Min,
            },
        };
        let empty = RecursiveRelationEnvelope::prove(&layer, &empty_plan).unwrap();
        empty
            .verify(&layer, layer.state.commitment_root, &empty_plan)
            .unwrap();
        assert_eq!(empty.result_len, 0);
        assert_eq!(empty.result_values, vec![Vec::new()]);
    }

    #[test]
    fn recursive_group_sum_accepts_exact_u64_totals_and_rejects_actual_overflow() {
        let layer_with = |values: &[(&str, &str, u64)]| {
            let store = open_sync_memory_store();
            let builder = store.create_base_layer().unwrap();
            for (subject, predicate, value) in values {
                builder
                    .add_value_triple(ValueTriple::new_value(
                        &iri(subject),
                        &iri(predicate),
                        u64::make_entry(value),
                    ))
                    .unwrap();
            }
            builder.commit().unwrap().proof_commitment().unwrap()
        };
        let source = || RelationProofPlan {
            query_digest: [105u8; 32],
            kind: RelationProofPlanKind::Bgp(LayerBgpPlan {
                source: ColumnSource::Materialized,
                atoms: vec![LayerBgpAtom {
                    subject: AtomTerm::Variable,
                    predicate_term: Vec::new(),
                    predicate_variable: true,
                    object: AtomTerm::Variable,
                    object_range: None,
                    object_kind: ObjectKindFilter::Any,
                    subject_object_equal: false,
                    source: ColumnSource::Materialized,
                }],
                joins: Vec::new(),
            }),
        };
        let sum_plan = |key_columns| RelationProofPlan {
            query_digest: [106u8; 32],
            kind: RelationProofPlanKind::GroupSum {
                input: Box::new(source()),
                key_columns,
                value_column: 2,
                datatype: CanonicalDatatype::UInt64,
            },
        };

        // The maximum is accepted as a checked partitioned sum. This is safe because
        // the BN254 scalar modulus exceeds u64::MAX.
        let partitioned = layer_with(&[("a", "p", u64::MAX - 5), ("a", "q", 5)]);
        let keyed_plan = sum_plan(vec![0]);
        let keyed = RecursiveRelationEnvelope::prove(&partitioned, &keyed_plan).unwrap();
        keyed
            .verify(&partitioned, partitioned.state.commitment_root, &keyed_plan)
            .unwrap();
        assert_eq!(
            keyed.result_values[1],
            vec![CanonicalResultValue::GeneratedDecimalU64(u64::MAX)]
        );

        let near_max = layer_with(&[("a", "p", u64::MAX - 5), ("a", "q", 4)]);
        let near_max_plan = sum_plan(vec![]);
        let near_max_envelope =
            RecursiveRelationEnvelope::prove(&near_max, &near_max_plan).unwrap();
        near_max_envelope
            .verify(&near_max, near_max.state.commitment_root, &near_max_plan)
            .unwrap();
        assert_eq!(
            near_max_envelope.result_values,
            vec![vec![CanonicalResultValue::GeneratedDecimalU64(u64::MAX - 1)]]
        );

        let mut tampered = near_max_envelope;
        tampered.result_columns[0][0] = ArkFr::from_u64(u64::MAX);
        let mut padded = tampered.result_columns[0].clone();
        padded.resize(1usize << tampered.result_log_domain, ArkFr::zero());
        tampered.result_commitments[0] = commit_column(&padded).unwrap();
        assert!(tampered
            .verify(&near_max, near_max.state.commitment_root, &near_max_plan)
            .is_err());

        let wrong_datatype = RelationProofPlan {
            query_digest: [107u8; 32],
            kind: RelationProofPlanKind::GroupSum {
                input: Box::new(source()),
                key_columns: vec![],
                value_column: 2,
                datatype: CanonicalDatatype::UInt32,
            },
        };
        assert!(RecursiveRelationEnvelope::prove(&partitioned, &wrong_datatype).is_err());

        let overflow = layer_with(&[("a", "p", u64::MAX), ("a", "q", 1)]);
        let overflow_plan = sum_plan(vec![]);
        assert!(RecursiveRelationEnvelope::prove(&overflow, &overflow_plan).is_err());
        let child = RecursiveRelationEnvelope::prove(&overflow, &source()).unwrap();
        assert!(derived_group_sum_columns(
            &overflow,
            &child,
            &[],
            2,
            CanonicalDatatype::UInt64,
        )
        .is_err());
    }

    fn assert_recursive_group_sum_authenticates_signed_values_domains_and_overflow() {
        for value in [i64::MIN, -1, 0, 1, i64::MAX] {
            assert_eq!(
                super::super::query::generated_decimal_i64_value(
                    generated_decimal_i64_field(value),
                )
                .unwrap(),
                value
            );
        }
        for (datatype, lexical, expected) in [
            (CanonicalDatatype::Int8, vec![(-3i8 as u8) ^ 0x80], -3i128),
            (
                CanonicalDatatype::Int16,
                ((-300i16 as u16) ^ (1 << 15)).to_be_bytes().to_vec(),
                -300i128,
            ),
            (
                CanonicalDatatype::Int32,
                ((-30_000i32 as u32) ^ (1 << 31)).to_be_bytes().to_vec(),
                -30_000i128,
            ),
            (
                CanonicalDatatype::Int64,
                ((-3_000_000_000i64 as u64) ^ (1 << 63)).to_be_bytes().to_vec(),
                -3_000_000_000i128,
            ),
        ] {
            let GroupSumValue::Signed(actual) = integral_catalog_value(
                &CanonicalObject::Value { datatype, lexical },
                datatype,
            )
            .unwrap()
            else {
                panic!("signed datatype decoded into unsigned sum domain")
            };
            assert_eq!(actual, expected);
        }
        let store = open_sync_memory_store();
        let base_builder = store.create_base_layer().unwrap();
        for (predicate, value) in [("p", i64::MIN), ("q", i64::MAX), ("r", 1)] {
            base_builder
                .add_value_triple(ValueTriple::new_value(
                    &iri("b"),
                    &iri(predicate),
                    i64::make_entry(&value),
                ))
                .unwrap();
        }
        let base = base_builder.commit().unwrap();
        let child_builder = base.open_write().unwrap();
        // Duplicate numeric solutions use distinct predicates because triples are a set.
        // `a` is appended after `b`, but must remain the first canonical node group.
        for (predicate, value) in [("p", -5i64), ("q", 2), ("r", 2)] {
            child_builder
                .add_value_triple(ValueTriple::new_value(
                    &iri("a"),
                    &iri(predicate),
                    i64::make_entry(&value),
                ))
                .unwrap();
        }
        let layer = child_builder.commit().unwrap().proof_commitment().unwrap();
        let source = RelationProofPlan {
            query_digest: [108u8; 32],
            kind: RelationProofPlanKind::Bgp(LayerBgpPlan {
                source: ColumnSource::Materialized,
                atoms: vec![LayerBgpAtom {
                    subject: AtomTerm::Variable,
                    predicate_term: Vec::new(),
                    predicate_variable: true,
                    object: AtomTerm::Variable,
                    object_range: None,
                    object_kind: ObjectKindFilter::Any,
                    subject_object_equal: false,
                    source: ColumnSource::Materialized,
                }],
                joins: Vec::new(),
            }),
        };
        let keyed_plan = RelationProofPlan {
            query_digest: [109u8; 32],
            kind: RelationProofPlanKind::GroupSum {
                input: Box::new(source.clone()),
                key_columns: vec![0],
                value_column: 2,
                datatype: CanonicalDatatype::Int64,
            },
        };
        // COMMITMENT DIET (Wave 0): every GroupSum / generated-filter over this DELTA head
        // reads its materialized view (no longer committed), so proving reject-closes at the
        // envelope entry. Reconciliation (Wave 2/3) will re-enable delta-head aggregates.
        assert!(layer.state.parent_state_root.is_some());
        assert!(layer.materialized.is_empty());
        let _ = &keyed_plan;
        assert!(RecursiveRelationEnvelope::prove(&layer, &keyed_plan).is_err());

        // The signed-value OVERFLOW authentication below runs over BASE layers and is
        // unaffected by the commitment diet, so it is retained verbatim.
        let global_plan = RelationProofPlan {
            query_digest: [110u8; 32],
            kind: RelationProofPlanKind::GroupSum {
                input: Box::new(source.clone()),
                key_columns: vec![],
                value_column: 2,
                datatype: CanonicalDatatype::Int64,
            },
        };
        let overflow_layer = |values: [i64; 2]| {
            let store = open_sync_memory_store();
            let builder = store.create_base_layer().unwrap();
            for (predicate, value) in [("p", values[0]), ("q", values[1])] {
                builder
                    .add_value_triple(ValueTriple::new_value(
                        &iri("a"),
                        &iri(predicate),
                        i64::make_entry(&value),
                    ))
                    .unwrap();
            }
            builder.commit().unwrap().proof_commitment().unwrap()
        };
        for overflow in [
            overflow_layer([i64::MAX, 1]),
            overflow_layer([i64::MIN, -1]),
        ] {
            assert!(RecursiveRelationEnvelope::prove(&overflow, &global_plan).is_err());
            let child = RecursiveRelationEnvelope::prove(&overflow, &source).unwrap();
            assert!(derived_group_sum_columns(
                &overflow,
                &child,
                &[],
                2,
                CanonicalDatatype::Int64,
            )
            .is_err());
        }
    }

    #[test]
    fn recursive_group_sum_authenticates_signed_values_domains_and_overflow() {
        std::thread::Builder::new()
            .name("store-generated-i64-filter-proof".into())
            .stack_size(256 * 1024 * 1024)
            .spawn(assert_recursive_group_sum_authenticates_signed_values_domains_and_overflow)
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn recursive_distinct_proves_exact_classes_roundtrips_and_rejects_tamper() {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for (subject, object) in [("a", "x"), ("a", "y"), ("b", "z")] {
            builder
                .add_value_triple(ValueTriple::new_node(
                    &iri(subject),
                    &iri("p"),
                    &iri(object),
                ))
                .unwrap();
        }
        let layer = builder.commit().unwrap().proof_commitment().unwrap();
        let bgp = LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![LayerBgpAtom::all_variables(iri("p").into_bytes())],
            joins: Vec::new(),
        };
        let leaf = RelationProofPlan {
            query_digest: plan_digest(&bgp).unwrap(),
            kind: RelationProofPlanKind::Bgp(bgp),
        };
        let plan = RelationProofPlan {
            query_digest: [93u8; 32],
            kind: RelationProofPlanKind::Distinct {
                input: Box::new(leaf),
                key_columns: vec![0],
            },
        };
        let envelope = RecursiveRelationEnvelope::prove(&layer, &plan).unwrap();
        envelope.verify(&layer, layer.state.commitment_root, &plan).unwrap();
        assert_eq!(envelope.result_len, 2);
        assert_eq!(envelope.result_columns.len(), 1);

        let bytes = envelope.encode().unwrap();
        let decoded = RecursiveRelationEnvelope::decode(&bytes).unwrap();
        decoded.verify(&layer, layer.state.commitment_root, &plan).unwrap();

        let mut wrong_key = plan.clone();
        if let RelationProofPlanKind::Distinct { key_columns, .. } = &mut wrong_key.kind {
            key_columns[0] = 1;
        }
        assert!(decoded
            .verify(&layer, layer.state.commitment_root, &wrong_key)
            .is_err());

        let mut duplicate = decoded.clone();
        duplicate.result_columns[0][1] = duplicate.result_columns[0][0];
        duplicate.result_commitments[0] = commit_column(&duplicate.result_columns[0]).unwrap();
        assert!(duplicate
            .verify(&layer, layer.state.commitment_root, &plan)
            .is_err());

        let mut changed_proof = decoded.clone();
        if let RecursiveRelationProof::Distinct { proof, .. } = &mut changed_proof.proof {
            proof.entries[0].lengths[0] += 1;
        }
        assert!(changed_proof
            .verify(&layer, layer.state.commitment_root, &plan)
            .is_err());
    }

    #[test]
    fn ordered_slice_bounds_cover_zero_overflow_and_wrapper_order() {
        assert_eq!(ordered_slice_bounds(3, 0, None), (0, 3));
        assert_eq!(ordered_slice_bounds(3, 0, Some(0)), (0, 0));
        assert_eq!(ordered_slice_bounds(3, 99, None), (3, 3));
        assert_eq!(ordered_slice_bounds(3, u64::MAX, Some(u64::MAX)), (3, 3));

        // start(1, limit(2, Goal)): the inner limit first yields [0,2), then start
        // drops one, leaving [1,2). limit(2, start(1, Goal)) instead yields [1,3).
        let inner_limit = ordered_slice_bounds(3, 0, Some(2));
        let outer_start = ordered_slice_bounds(inner_limit.1 - inner_limit.0, 1, None);
        assert_eq!((inner_limit.0 + outer_start.0, inner_limit.0 + outer_start.1), (1, 2));
        let inner_start = ordered_slice_bounds(3, 1, None);
        let outer_limit = ordered_slice_bounds(inner_start.1 - inner_start.0, 0, Some(2));
        assert_eq!((inner_start.0 + outer_limit.0, inner_start.0 + outer_limit.1), (1, 3));
    }

    fn assert_recursive_ordered_slice_roundtrips_and_rejects_content_and_plan_tamper() {
        let (layer, bgp) = order_fixture();
        let unordered_leaf = RelationProofPlan {
            query_digest: plan_digest(&bgp).unwrap(),
            kind: RelationProofPlanKind::Bgp(bgp.clone()),
        };
        let leaf = RelationProofPlan {
            query_digest: plan_digest(&bgp).unwrap(),
            kind: RelationProofPlanKind::Bgp(bgp),
        };
        let order = RelationProofPlan {
            query_digest: [41u8; 32],
            kind: RelationProofPlanKind::ExactScalarOrder {
                input: Box::new(leaf),
                datatype: CanonicalDatatype::UInt32,
                direction: OrderDirection::Ascending,
            },
        };
        let plan = RelationProofPlan {
            query_digest: [43u8; 32],
            kind: RelationProofPlanKind::ExactOrderedSlice {
                input: Box::new(order),
                offset: 1,
                limit: Some(2),
            },
        };
        let envelope = RecursiveRelationEnvelope::prove(&layer, &plan).unwrap();
        envelope.verify(&layer, layer.state.commitment_root, &plan).unwrap();
        assert_eq!(envelope.result_len, 2);
        let bytes = envelope.encode().unwrap();
        let decoded = RecursiveRelationEnvelope::decode(&bytes).unwrap();
        decoded.verify(&layer, layer.state.commitment_root, &plan).unwrap();

        let mut wrong_plan = plan.clone();
        if let RelationProofPlanKind::ExactOrderedSlice { limit, .. } = &mut wrong_plan.kind {
            *limit = Some(1);
        }
        assert!(decoded.verify(&layer, layer.state.commitment_root, &wrong_plan).is_err());

        let mut reordered = RecursiveRelationEnvelope::decode(&bytes).unwrap();
        reordered.result_columns[0].swap(0, 1);
        let mut padded = reordered.result_columns[0].clone();
        padded.resize(1usize << reordered.result_log_domain, ArkFr::zero());
        reordered.result_commitments[0] = commit_column(&padded).unwrap();
        assert!(reordered.verify(&layer, layer.state.commitment_root, &plan).is_err());

        let unordered = RelationProofPlan {
            query_digest: [44u8; 32],
            kind: RelationProofPlanKind::ExactOrderedSlice {
                input: Box::new(unordered_leaf),
                offset: 0,
                limit: Some(1),
            },
        };
        assert!(unordered.result_shape().is_err());
    }

    #[test]
    fn recursive_ordered_slice_roundtrips_and_rejects_content_and_plan_tamper() {
        // Recursive Dory proof values are deliberately boxed in the envelope, but the
        // debug prover itself has large temporary stack frames. Give this arithmetic
        // test the same explicit headroom used by ordinary proving hosts.
        std::thread::Builder::new()
            .name("ordered-slice-proof".into())
            .stack_size(256 * 1024 * 1024)
            .spawn(assert_recursive_ordered_slice_roundtrips_and_rejects_content_and_plan_tamper)
            .unwrap()
            .join()
            .unwrap();
    }

    fn assert_recursive_multikey_order_roundtrip_and_tamper() {
        let (layer, bgp) = multi_order_fixture();
        let child = RelationProofPlan { query_digest: plan_digest(&bgp).unwrap(), kind: RelationProofPlanKind::Bgp(bgp) };
        let plan = RelationProofPlan { query_digest: [61u8; 32], kind: RelationProofPlanKind::ExactMultiOrder {
            input: Box::new(child), datatypes: vec![CanonicalDatatype::UInt32; 2],
            key_indices: vec![1, 0],
            directions: vec![OrderDirection::Descending, OrderDirection::Ascending],
        }};
        let envelope = RecursiveRelationEnvelope::prove(&layer, &plan).unwrap();
        envelope.verify(&layer, layer.state.commitment_root, &plan).unwrap();
        assert_eq!(envelope.result_len, 4);
        let bytes = envelope.encode().unwrap();
        let decoded = RecursiveRelationEnvelope::decode(&bytes).unwrap();
        decoded.verify(&layer, layer.state.commitment_root, &plan).unwrap();

        let mut wrong_direction = plan.clone();
        if let RelationProofPlanKind::ExactMultiOrder { directions, .. } = &mut wrong_direction.kind { directions[1] = OrderDirection::Descending; }
        assert!(decoded.verify(&layer, layer.state.commitment_root, &wrong_direction).is_err());
        let mut wrong_type = plan.clone();
        if let RelationProofPlanKind::ExactMultiOrder { datatypes, .. } = &mut wrong_type.kind { datatypes[0] = CanonicalDatatype::Int32; }
        assert!(decoded.verify(&layer, layer.state.commitment_root, &wrong_type).is_err());
        let mut wrong_key_order = plan.clone();
        if let RelationProofPlanKind::ExactMultiOrder { key_indices, .. } = &mut wrong_key_order.kind { key_indices[1] = 1; }
        assert!(decoded.verify(&layer, layer.state.commitment_root, &wrong_key_order).is_err());

        let mut reordered = RecursiveRelationEnvelope::decode(&bytes).unwrap();
        for column in &mut reordered.result_columns { column.swap(0, 1); }
        for (commitment, column) in reordered.result_commitments.iter_mut().zip(&reordered.result_columns) {
            let mut padded = column.clone(); padded.resize(1usize << reordered.result_log_domain, ArkFr::zero());
            *commitment = commit_column(&padded).unwrap();
        }
        assert!(reordered.verify(&layer, layer.state.commitment_root, &plan).is_err());
    }

    #[test]
    fn recursive_multikey_order_roundtrip_and_tamper() {
        std::thread::Builder::new().name("multi-order-proof".into()).stack_size(256 * 1024 * 1024)
            .spawn(assert_recursive_multikey_order_roundtrip_and_tamper).unwrap().join().unwrap();
    }

    #[test]
    fn decoder_item_budget_is_global_across_nested_collections() {
        let mut reader = Reader {
            bytes: &[],
            pos: 0,
            decoded_items: 0,
        };
        reader.claim_items(MAX_ITEMS - 1).unwrap();
        reader.claim_items(1).unwrap();
        assert!(reader.claim_items(1).is_err());
    }

    #[test]
    fn recursive_plan_rejects_nullable_right_correlation_keys() {
        let (layer, bgp) = fixture();
        let digest = [7u8; 32];
        let leaf = RelationProofPlan {
            query_digest: digest,
            kind: RelationProofPlanKind::Bgp(bgp),
        };
        let nullable_right = RelationProofPlan {
            query_digest: digest,
            kind: RelationProofPlanKind::LeftOuter {
                left: Box::new(leaf.clone()),
                right: Box::new(leaf.clone()),
                key_pairs: Vec::new(),
            },
        };
        let right_shape = nullable_right.result_shape().unwrap().0;
        let nullable_index = right_shape
            .iter()
            .position(|descriptor| descriptor.nullable)
            .unwrap();
        for kind in [
            RelationProofPlanKind::LeftAnti {
                left: Box::new(leaf.clone()),
                right: Box::new(nullable_right.clone()),
                key_pairs: vec![(0, nullable_index)],
            },
            RelationProofPlanKind::LeftOuter {
                left: Box::new(leaf.clone()),
                right: Box::new(nullable_right.clone()),
                key_pairs: vec![(0, nullable_index)],
            },
        ] {
            let plan = RelationProofPlan {
                query_digest: digest,
                kind,
            };
            assert!(plan.result_shape().is_err());
        }
        drop(layer);
    }

    #[test]
    fn ground_result_unit_is_authenticated_but_not_dictionary_resolved() {
        let (layer, _) = fixture();
        let plan = LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![LayerBgpAtom {
                subject: AtomTerm::NodeConstant(iri("a").into_bytes()),
                predicate_term: iri("knows").into_bytes(),
                predicate_variable: false,
                object: AtomTerm::NodeConstant(iri("b").into_bytes()),
                object_range: None,
                object_kind: ObjectKindFilter::Any,
                subject_object_equal: false,
                source: ColumnSource::Materialized,
            }],
            joins: Vec::new(),
        };
        let result = prove_layer_bgp_with_result(&layer, &plan).unwrap();
        assert_eq!(result.result_len, 1);
        let result_columns = result.real_columns();
        let query_digest = plan_digest(&plan).unwrap();
        let mut envelope = LayerBgpEnvelope::new_verified(
            &layer,
            layer.state.commitment_root,
            &plan,
            query_digest,
            result.proof,
            result.result_commitments,
            result_columns,
            result.result_log_domain,
            result.result_len,
            result.predicate_ids,
        )
        .unwrap();
        assert!(envelope.result_objects.is_empty());
        envelope
            .verify(&layer, layer.state.commitment_root, &plan, query_digest)
            .unwrap();
        envelope.result_log_domain += 1;
        assert!(envelope
            .verify(&layer, layer.state.commitment_root, &plan, query_digest)
            .is_err());
    }

    #[test]
    fn envelope_roundtrips_and_rejects_tampering_and_version_confusion() {
        let (layer, plan) = fixture();
        let result = prove_layer_bgp_with_result(&layer, &plan).unwrap();
        let result_columns = result.real_columns();
        let query_digest = plan_digest(&plan).unwrap();
        let envelope = LayerBgpEnvelope::new_verified(
            &layer,
            layer.state.commitment_root,
            &plan,
            query_digest,
            result.proof,
            result.result_commitments,
            result_columns,
            result.result_log_domain,
            result.result_len,
            result.predicate_ids,
        )
        .unwrap();
        let encoded = envelope.encode().unwrap();
        let decoded = LayerBgpEnvelope::decode(&encoded).unwrap();
        decoded
            .verify(&layer, layer.state.commitment_root, &plan, query_digest)
            .unwrap();
        assert_eq!(decoded.result_objects.len(), 2);

        let mut corrupt = encoded.clone();
        corrupt[MAGIC.len() + 8] ^= 1;
        assert!(LayerBgpEnvelope::decode(&corrupt).is_err());

        let mut wrong_version = encoded.clone();
        wrong_version[MAGIC.len()..MAGIC.len() + 4].copy_from_slice(&6u32.to_le_bytes());
        refresh_checksum(&mut wrong_version);
        assert!(LayerBgpEnvelope::decode(&wrong_version).is_err());

        let wrong_plan = LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![LayerBgpAtom::all_variables(iri("likes").into_bytes())],
            joins: Vec::new(),
        };
        assert!(decoded
            .verify(
                &layer,
                layer.state.commitment_root,
                &wrong_plan,
                query_digest
            )
            .is_err());

        let mut wrong_root = layer.state.commitment_root;
        wrong_root[0] ^= 1;
        assert!(decoded
            .verify(&layer, wrong_root, &plan, query_digest)
            .is_err());

        // The first proof field is the atom vector, immediately after the three
        // fixed-size hashes. Reject its claimed allocation before reading elements.
        let mut oversized_count = encoded.clone();
        let atom_count_offset = MAGIC.len() + 4 + 3 * 32;
        oversized_count[atom_count_offset..atom_count_offset + 8]
            .copy_from_slice(&((MAX_ITEMS as u64) + 1).to_le_bytes());
        refresh_checksum(&mut oversized_count);
        assert!(LayerBgpEnvelope::decode(&oversized_count).is_err());

        // Exactly-at-cap counts pass the numeric guard. A truncated vector of the
        // materially large LayerBgpAtomProof type must fail before reserving space for
        // MAX_ITEMS elements (which would dwarf the entire input-size limit).
        let mut truncated_at_cap = encoded[..atom_count_offset + 8].to_vec();
        truncated_at_cap[atom_count_offset..atom_count_offset + 8]
            .copy_from_slice(&(MAX_ITEMS as u64).to_le_bytes());
        let checksum = Sha256::digest(&truncated_at_cap);
        truncated_at_cap.extend_from_slice(&checksum);
        let error = LayerBgpEnvelope::decode(&truncated_at_cap).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);

        let mut forged_objects = decoded.clone();
        forged_objects.result_objects[0][0] = CanonicalObject::Node(iri("forged").into_bytes());
        assert!(forged_objects
            .verify(&layer, layer.state.commitment_root, &plan, query_digest)
            .is_err());
    }

    #[test]
    fn recursive_leaf_roundtrips_and_binds_plan_identity_and_schema() {
        let (layer, bgp) = fixture();
        let (plan, envelope) = recursive_leaf(&layer, &bgp);
        envelope
            .verify(&layer, layer.state.commitment_root, &plan)
            .unwrap();
        let encoded = envelope.encode().unwrap();
        let decoded = RecursiveRelationEnvelope::decode(&encoded).unwrap();
        decoded
            .verify(&layer, layer.state.commitment_root, &plan)
            .unwrap();

        let mut wrong_digest = plan.clone();
        wrong_digest.query_digest[0] ^= 1;
        assert!(decoded
            .verify(&layer, layer.state.commitment_root, &wrong_digest)
            .is_err());

        let mut wrong_schema = decoded.clone();
        wrong_schema.result_descriptors[0].nullable = true;
        assert!(wrong_schema
            .verify(&layer, layer.state.commitment_root, &plan)
            .is_err());

        let mut wrong_variant = plan.clone();
        wrong_variant.kind = RelationProofPlanKind::LeftAnti {
            left: Box::new(plan.clone()),
            right: Box::new(plan.clone()),
            key_pairs: Vec::new(),
        };
        assert!(decoded
            .verify(&layer, layer.state.commitment_root, &wrong_variant)
            .is_err());
    }

    #[cfg(any(feature = "blitzar-cpu", feature = "blitzar-gpu"))]
    #[test]
    fn recursive_proving_context_preserves_the_wire_format() {
        let (layer, bgp) = fixture();
        let plan = RelationProofPlan {
            query_digest: plan_digest(&bgp).unwrap(),
            kind: RelationProofPlanKind::Bgp(bgp),
        };
        let runtime = StoreProverContext::initialize_blitzar(
            dory_pcs::backends::arkworks::BlitzarRuntimeConfig {
                num_precomputed_generators: 0,
            },
        );
        let accelerated = StoreProverContext::blitzar(
            &runtime,
            super::super::argument::ProverSetupTier::Log20,
        )
        .unwrap();
        drop(runtime);

        let cpu = RecursiveRelationEnvelope::prove(&layer, &plan).unwrap();
        let with_context =
            RecursiveRelationEnvelope::prove_with_context(&accelerated, &layer, &plan).unwrap();
        assert_eq!(cpu.encode().unwrap(), with_context.encode().unwrap());
        with_context
            .verify(&layer, layer.state.commitment_root, &plan)
            .unwrap();
    }

    #[test]
    fn recursive_envelope_rejects_depth_bomb_before_proof_verification() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(RECURSIVE_MAGIC);
        bytes.extend_from_slice(&RECURSIVE_VERSION.to_le_bytes());
        // Only the hostile prefix is needed: decoding descends into the left child of
        // each anti node and rejects the depth before touching any missing proof bytes.
        for _ in 0..=MAX_RELATION_PROOF_DEPTH {
            [0u8; 32].put(&mut bytes).unwrap();
            [1u8; 32].put(&mut bytes).unwrap();
            2u8.put(&mut bytes).unwrap();
        }
        let checksum = Sha256::digest(&bytes);
        bytes.extend_from_slice(&checksum);
        let error = RecursiveRelationEnvelope::decode(&bytes).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("structural limits"));
    }

    #[test]
    fn recursive_envelope_rejects_node_bomb_before_allocation() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(RECURSIVE_MAGIC);
        bytes.extend_from_slice(&RECURSIVE_VERSION.to_le_bytes());
        [0u8; 32].put(&mut bytes).unwrap();
        [1u8; 32].put(&mut bytes).unwrap();
        1u8.put(&mut bytes).unwrap();
        MAX_RELATION_PROOF_NODES.put(&mut bytes).unwrap();
        let checksum = Sha256::digest(&bytes);
        bytes.extend_from_slice(&checksum);
        let error = RecursiveRelationEnvelope::decode(&bytes).unwrap_err();
        assert!(error.to_string().contains("too many recursive union branches"));
    }

    // ---- Bounded transitive closure (`star`/`plus`) ----

    fn closure_layer(triples: &[(&str, &str, &str)]) -> LayerCommitment {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for (subject, predicate, object) in triples {
            builder
                .add_value_triple(ValueTriple::new_node(
                    &iri(subject),
                    &iri(predicate),
                    &iri(object),
                ))
                .unwrap();
        }
        (*builder.commit().unwrap().proof_commitment().unwrap()).clone()
    }

    fn closure_plan_dir(
        predicate: &str,
        seed: &str,
        reflexive: bool,
        inverted: bool,
    ) -> RelationProofPlan {
        RelationProofPlan {
            query_digest: [7u8; 32],
            kind: RelationProofPlanKind::TransitiveClosure {
                edge: LayerBgpPlan {
                    source: ColumnSource::Materialized,
                    atoms: vec![LayerBgpAtom::all_variables(iri(predicate).into_bytes())],
                    joins: Vec::new(),
                },
                seed: CanonicalObject::Node(iri(seed).into_bytes()),
                reflexive,
                inverted,
            },
        }
    }

    fn closure_plan(predicate: &str, seed: &str, reflexive: bool) -> RelationProofPlan {
        closure_plan_dir(predicate, seed, reflexive, false)
    }

    fn closure_result_set(envelope: &RecursiveRelationEnvelope) -> std::collections::BTreeSet<String> {
        envelope.result_values[0]
            .iter()
            .map(|value| match value {
                CanonicalResultValue::Object(CanonicalObject::Node(bytes)) => {
                    String::from_utf8(bytes.clone()).unwrap()
                }
                other => panic!("unexpected closure result value {other:?}"),
            })
            .collect()
    }

    fn iris(names: &[&str]) -> std::collections::BTreeSet<String> {
        names.iter().map(|name| iri(name)).collect()
    }

    fn assert_transitive_closure_end_to_end() {
        // Straight chain a -> b -> c -> d.
        let layer = closure_layer(&[("a", "p", "b"), ("b", "p", "c"), ("c", "p", "d")]);
        let root = layer.state.commitment_root;

        // Plus from a = {b, c, d} (one-or-more hops).
        let plus = closure_plan("p", "a", false);
        let env = RecursiveRelationEnvelope::prove(&layer, &plus).unwrap();
        env.verify(&layer, root, &plus).unwrap();
        assert_eq!(closure_result_set(&env), iris(&["b", "c", "d"]));
        // Wire round-trip still verifies.
        let bytes = env.encode().unwrap();
        RecursiveRelationEnvelope::decode(&bytes)
            .unwrap()
            .verify(&layer, root, &plus)
            .unwrap();

        // Star from a = {a, b, c, d} (adds the identity a).
        let star = closure_plan("p", "a", true);
        let env_star = RecursiveRelationEnvelope::prove(&layer, &star).unwrap();
        env_star.verify(&layer, root, &star).unwrap();
        assert_eq!(closure_result_set(&env_star), iris(&["a", "b", "c", "d"]));

        // Substituting the star proof under the plus plan (or vice versa) must reject.
        assert!(env_star.verify(&layer, root, &plus).is_err());
        assert!(env.verify(&layer, root, &star).is_err());

        // CYCLE a -> b -> a: the closure must TERMINATE (empty-delta fixpoint) and give the exact set.
        let cyclic = closure_layer(&[("a", "p", "b"), ("b", "p", "a")]);
        let cyclic_root = cyclic.state.commitment_root;
        let cyclic_plus = closure_plan("p", "a", false);
        let cyclic_env = RecursiveRelationEnvelope::prove(&cyclic, &cyclic_plus).unwrap();
        cyclic_env.verify(&cyclic, cyclic_root, &cyclic_plus).unwrap();
        // plus from a on a<->b = {a, b} (a re-reached via a -> b -> a).
        assert_eq!(closure_result_set(&cyclic_env), iris(&["a", "b"]));

        // Bound TARGET via inversion: over a -> b -> c -> d, the INVERSE plus from d = the set of nodes
        // that reach d = {a, b, c}. Same edge relation; only the traversal direction (join key) flips.
        let inverse_plus = closure_plan_dir("p", "d", false, true);
        let inverse_env = RecursiveRelationEnvelope::prove(&layer, &inverse_plus).unwrap();
        inverse_env.verify(&layer, root, &inverse_plus).unwrap();
        assert_eq!(closure_result_set(&inverse_env), iris(&["a", "b", "c"]));
        // The forward plus and the inverse plus are distinct plans: their proofs are not interchangeable.
        assert!(inverse_env.verify(&layer, root, &plus).is_err());
        assert!(env.verify(&layer, root, &inverse_plus).is_err());

        // ---- Soundness negatives ----

        // (i) A TRUNCATED closure that has NOT converged: drop the last round so the delivered
        // accumulator is A_{K-1} while the fixpoint was proven over A_K. The fixpoint inner join binds
        // A_K's commitment, which no longer matches -> reject.
        let mut truncated = env.clone();
        if let RecursiveRelationProof::TransitiveClosure { rounds, .. } = &mut truncated.proof {
            assert!(rounds.len() >= 2, "chain a->b->c->d should need >= 2 rounds");
            rounds.pop();
        } else {
            panic!("expected a transitive-closure proof");
        }
        assert!(truncated.verify(&layer, root, &plus).is_err());

        // (ii) A SPURIOUS node injected into the final round's set: it never came from an authenticated
        // InnerJoin, so the round's Distinct proof no longer covers the tampered rows -> reject.
        let mut spurious = env.clone();
        if let RecursiveRelationProof::TransitiveClosure { rounds, .. } = &mut spurious.proof {
            let ghost = 999_999u64;
            rounds.last_mut().unwrap().next_rows.push(ghost);
        }
        assert!(spurious.verify(&layer, root, &plus).is_err());

        // (iii) A TAMPERED delivered result set (drop a genuine reachable node, recommit) no longer
        // equals the proven reachable set -> reject.
        let mut tampered = env.clone();
        tampered.result_columns[0].pop();
        tampered.result_len -= 1;
        tampered.result_values[0].pop();
        tampered.result_commitments[0] = {
            let mut column = tampered.result_columns[0].clone();
            column.resize(1usize << tampered.result_log_domain, ArkFr::zero());
            commit_column(&column).unwrap()
        };
        assert!(tampered.verify(&layer, root, &plus).is_err());
    }

    #[test]
    fn transitive_closure_star_plus_cycle_and_soundness() {
        std::thread::Builder::new()
            .name("transitive-closure-proof".into())
            .stack_size(128 * 1024 * 1024)
            .spawn(assert_transitive_closure_end_to_end)
            .unwrap()
            .join()
            .unwrap();
    }
}
