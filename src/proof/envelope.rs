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
    verify_inner_join_with_keys, verify_left_outer_join_with_keys, InnerJoinProof, LeftAntiJoinProof, LeftOuterJoinProof,
};
use super::argument::{commit_column, ColumnOpening, StoreProverContext, SumcheckProof, Transcript254};
use super::bgp::{BgpChainProof, BgpStageProof, JoinStep, RelationCommit, RelationWitness};
use super::commitment::{LayerCommitment, VerifierLayer};
use super::count::CountProof;
use super::dict::DictProof;
use super::distinct::{
    prove_relation_distinct_with_context, verify_relation_distinct,
    RelationDistinctEntryProof, RelationDistinctProof,
};
use super::equality::{
    prove_relation_equality_with_context, verify_relation_equality, RelationEqualityConstraint,
    RelationEqualityProof,
};
use super::filter::{
    AtomFilterProof, BooleanProductProof, CompactionOnlyProof, FilterProof, FullFilterProof,
    SelectionOnlyProof,
};
use super::group::{
    prove_group_extrema_with_context, verify_group_extrema, GroupExtremaEntryProof,
    GroupExtremaKind, GroupExtremaProof,
};
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
    verify_result_columns, generated_decimal_i64_field, AtomTerm, CanonicalResultValue, DictionaryAbsence,
    DictionaryAbsencePosition, LayerBgpAtom,
    LayerBgpAtomProof, LayerBgpPlan, LayerBgpProof, ObjectKindFilter, ObjectRangeBound,
    ObjectRangeSpec, RangeAtomSubproof, ResultColumnDescriptor, ResultIdNamespace,
    SecondRangeSubproof,
};
use super::range::{
    DictLookupProof, ObjectRangeCombineProof, ObjectRangeProof, RangeOp, RangeSelectProof,
};
use super::union::prove_bag_union_with_context;
use super::relocate::RelocationProof;
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
// v20 adds exact private Collect-list integral extrema.
const RECURSIVE_VERSION: u32 = 20;
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
wire_struct!(LayerBgpAtom {
    subject,
    predicate_term,
    predicate_variable,
    object,
    object_range,
    object_kind,
    subject_object_equal
});
wire_struct!(LayerBgpPlan { atoms, joins });

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
    LeftOuter {
        left: Box<RelationProofPlan>,
        right: Box<RelationProofPlan>,
        key_pairs: Vec<(usize, usize)>,
    },
    Equality {
        input: Box<RelationProofPlan>,
        constraint: RelationEqualityPlan,
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
    /// Scalar-key `GroupBy`, with its private bag consumed immediately by
    /// `RangeMin`/`RangeMax`; output is exactly `[key, original-value-id]`.
    GroupExtrema {
        input: Box<RelationProofPlan>,
        key_column: usize,
        value_column: usize,
        datatype: CanonicalDatatype,
        kind: GroupExtremaKind,
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
                    shared_left.push(*left_index); shared_right.push(*right_index);
                }
                descriptors.extend(right_descriptors.into_iter().enumerate().filter_map(
                    |(index, descriptor)| (!shared_right.contains(&index)).then_some(descriptor)
                ));
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
            RelationProofPlanKind::GroupExtrema { input, key_column, value_column, datatype, .. } => {
                let (descriptors, ground) = input.shape_bounded(depth + 1, nodes)?;
                if ground || *key_column == *value_column {
                    return Err(invalid("invalid group-extrema child shape"));
                }
                let key = descriptors.get(*key_column).copied().ok_or_else(|| invalid("group-extrema key column out of range"))?;
                let value = descriptors.get(*value_column).copied().ok_or_else(|| invalid("group-extrema value column out of range"))?;
                if key.nullable || value.nullable || value.namespace != ResultIdNamespace::NodeValue {
                    return Err(invalid("group-extrema requires nonnullable Store-id columns"));
                }
                if !numeric_datatype(*datatype) || matches!(datatype, CanonicalDatatype::Decimal | CanonicalDatatype::BigInt | CanonicalDatatype::PositiveInteger | CanonicalDatatype::NonNegativeInteger | CanonicalDatatype::NonPositiveInteger | CanonicalDatatype::NegativeInteger) {
                    return Err(invalid("group-extrema datatype is not fixed-width order-preserving"));
                }
                Ok((vec![key, ResultColumnDescriptor { namespace: ResultIdNamespace::NodeValue, nullable: false }], false))
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
    LeftOuter {
        left: Box<RecursiveRelationEnvelope>,
        right: Box<RecursiveRelationEnvelope>,
        proof: Box<LeftOuterJoinProof>,
    },
    Equality {
        input: Box<RecursiveRelationEnvelope>,
        proof: Box<RelationEqualityProof>,
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
        proof: Box<GroupExtremaProof>,
    },
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
        plan.result_shape()?;
        let mut nodes = 0;
        Self::prove_bounded(context, layer, plan, 1, &mut nodes)
    }

    /// Explicit Arkworks CPU convenience for recursive proving.
    pub fn prove(layer: &LayerCommitment, plan: &RelationProofPlan) -> io::Result<Self> {
        Self::prove_with_context(&StoreProverContext::cpu(), layer, plan)
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
            RelationProofPlanKind::GroupExtrema { input, key_column, value_column, datatype, kind } => {
                let input_envelope = Self::prove_bounded(context, layer, input, depth + 1, nodes)?;
                let witness = input_envelope.relation_witness(None)?;
                let key_namespace = input_envelope.result_descriptors[*key_column].namespace;
                let view = layer.verifier_view();
                let resolve_id = |field: ArkFr, namespace: ResultIdNamespace| -> io::Result<u64> {
                    let id = match namespace {
                        ResultIdNamespace::Predicate => view.predicate_catalog.iter().find_map(|record| (ArkFr::from_u64(record.predicate_id) == field).then_some(record.predicate_id)),
                        ResultIdNamespace::NodeValue => view.node_value_catalog.iter().find_map(|record| (ArkFr::from_u64(record.object_id) == field).then_some(record.object_id)),
                        ResultIdNamespace::GeneratedDecimalU64
                        | ResultIdNamespace::GeneratedDecimalI64 => None,
                    };
                    id.ok_or_else(|| invalid("group-extrema input id absent from planned namespace"))
                };
                let mut groups = std::collections::BTreeMap::<u64, (u64, ark_bn254::Fr)>::new();
                for row in 0..witness.n {
                    let key = resolve_id(witness.columns[*key_column][row], key_namespace)?;
                    let value = resolve_id(witness.columns[*value_column][row], ResultIdNamespace::NodeValue)?;
                    let encoding = super::commitment::node_value_op_value(
                        &view.node_value_catalog.iter().find(|record| record.object_id == value)
                            .ok_or_else(|| invalid("group-extrema value id absent from catalog"))?.triple.object,
                    )?.0;
                    groups.entry(key).and_modify(|current| {
                        let replace = match kind { GroupExtremaKind::Min => encoding.into_bigint() < current.1.into_bigint(), GroupExtremaKind::Max => encoding.into_bigint() > current.1.into_bigint() };
                        if replace { *current = (value, encoding); }
                    }).or_insert((value, encoding));
                }
                let rows = groups.into_iter().map(|(key, (value, _))| (key, value)).collect::<Vec<_>>();
                let mut transcript = Transcript254::new(b"parture-group-extrema-envelope");
                transcript.append_bytes(b"pow/group-envelope/root", &layer.state.commitment_root);
                transcript.append_bytes(b"pow/group-envelope/query", &plan.query_digest);
                let result = prove_group_extrema_with_context(context, &witness, *key_column, *value_column, *datatype, *kind, &rows, layer, &mut transcript)?;
                Self::finish_proved(layer, plan, RecursiveRelationProof::GroupExtrema {
                    input: Box::new(input_envelope), proof: Box::new(result.proof),
                }, result.commitments, result.columns, result.log_domain, result.n)
            }
        }
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
                RecursiveRelationProof::GroupExtrema { input, proof },
                RelationProofPlanKind::GroupExtrema { input: input_plan, key_column, value_column, datatype, kind },
            ) => {
                input.verify_bounded(layer, expected_root, input_plan, depth + 1, nodes)?;
                if self.result_columns.len() != 2 || self.result_len > self.result_columns[0].len() || self.result_len > self.result_columns[1].len() {
                    return Err(invalid("group-extrema result column shape mismatch"));
                }
                let key_namespace = expected_descriptors[0].namespace;
                let view = layer.verifier_view();
                let resolve_id = |field: ArkFr, namespace: ResultIdNamespace| -> io::Result<u64> {
                    match namespace {
                        ResultIdNamespace::Predicate => view.predicate_catalog.iter()
                            .find(|r| ArkFr::from_u64(r.predicate_id) == field).map(|r| r.predicate_id),
                        ResultIdNamespace::NodeValue => view.node_value_catalog.iter()
                            .find(|r| ArkFr::from_u64(r.object_id) == field).map(|r| r.object_id),
                        ResultIdNamespace::GeneratedDecimalU64
                        | ResultIdNamespace::GeneratedDecimalI64 => None,
                    }.ok_or_else(|| invalid("group-extrema output id absent from planned namespace"))
                };
                let rows = (0..self.result_len).map(|i| Ok((
                    resolve_id(self.result_columns[0][i], key_namespace)?,
                    resolve_id(self.result_columns[1][i], ResultIdNamespace::NodeValue)?,
                ))).collect::<io::Result<Vec<_>>>()?;
                let mut transcript = super::argument::Transcript254::new(b"parture-group-extrema-envelope");
                transcript.append_bytes(b"pow/group-envelope/root", &expected_root);
                transcript.append_bytes(b"pow/group-envelope/query", &plan.query_digest);
                verify_group_extrema(&input.relation_commit(None)?, *key_column, *value_column, *datatype, *kind, &rows, layer, proof, &mut transcript)?;
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

    pub fn verify(
        &self,
        layer: &impl VerifierLayer,
        expected_root: ProofHash,
        plan: &RelationProofPlan,
    ) -> io::Result<()> {
        // Validate the entire caller-supplied tree once with a global node budget before
        // descending through the equally bounded delivered proof tree.
        plan.result_shape()?;
        let mut nodes = 0;
        self.verify_bounded(layer, expected_root, plan, 1, &mut nodes)
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
            RecursiveRelationProof::Equality { input, proof } => {
                4u8.put(out)?;
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
            RecursiveRelationProof::GroupExtrema { input, proof } => {
                5u8.put(out)?;
                input.put_bounded(out, depth + 1, nodes)?;
                proof.put(out)?;
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
            5 => RecursiveRelationProof::GroupExtrema {
                input: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
                proof: Box::<GroupExtremaProof>::get(reader)?,
            },
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
            atoms: vec![LayerBgpAtom {
                subject: AtomTerm::NodeConstant(iri("metric").into_bytes()),
                predicate_term: iri("score").into_bytes(),
                predicate_variable: false,
                object: AtomTerm::Variable,
                object_range: None,
                object_kind: ObjectKindFilter::Any,
                subject_object_equal: false,
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
        };
        ((*layer).clone(), LayerBgpPlan {
            atoms: vec![atom("left", "x"), atom("right", "y")],
            joins: vec![JoinStep::cross_product()],
        })
    }

    fn refresh_checksum(bytes: &mut Vec<u8>) {
        let body_len = bytes.len() - CHECKSUM_BYTES;
        let checksum = Sha256::digest(&bytes[..body_len]);
        bytes[body_len..].copy_from_slice(&checksum);
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
        std::thread::Builder::new().name("inner-join-proof".into()).stack_size(32 * 1024 * 1024)
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
                atoms: vec![LayerBgpAtom {
                    subject: AtomTerm::NodeConstant(iri("missing").into_bytes()),
                    predicate_term: iri("p").into_bytes(),
                    predicate_variable: false,
                    object: AtomTerm::Variable,
                    object_range: None,
                    object_kind: ObjectKindFilter::Any,
                    subject_object_equal: false,
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
        assert_eq!(head.state.materialized_count, 3);

        let envelope = RecursiveRelationEnvelope::prove(&head, &plan).unwrap();
        envelope
            .verify(&head, head.state.commitment_root, &plan)
            .unwrap();
        assert_eq!(envelope.result_columns, vec![vec![ArkFr::from_u64(3)]]);
        assert_eq!(
            envelope.result_values,
            vec![vec![CanonicalResultValue::GeneratedDecimalU64(3)]]
        );
        let bytes = envelope.encode().unwrap();
        let decoded = RecursiveRelationEnvelope::decode(&bytes).unwrap();
        decoded
            .verify(&head, head.state.commitment_root, &plan)
            .unwrap();

        let mut changed = decoded.clone();
        changed.result_columns[0][0] = ArkFr::from_u64(4);
        let padded = vec![ArkFr::from_u64(4), ArkFr::zero()];
        changed.result_commitments[0] = commit_column(&padded).unwrap();
        changed.result_values = vec![vec![CanonicalResultValue::GeneratedDecimalU64(4)]];
        assert!(changed.verify(&head, head.state.commitment_root, &plan).is_err());

        let wrong_plan = RelationProofPlan {
            query_digest: [96u8; 32],
            kind: RelationProofPlanKind::LayerTripleCount,
        };
        assert!(decoded
            .verify(&head, head.state.commitment_root, &wrong_plan)
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
                atoms: vec![LayerBgpAtom {
                    subject: AtomTerm::Variable,
                    predicate_term: Vec::new(),
                    predicate_variable: true,
                    object: AtomTerm::Variable,
                    object_range: None,
                    object_kind: ObjectKindFilter::Any,
                    subject_object_equal: false,
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
        let envelope = RecursiveRelationEnvelope::prove(&layer, &plan).unwrap();
        envelope
            .verify(&layer, layer.state.commitment_root, &plan)
            .unwrap();
        assert_eq!(envelope.result_len, 2);
        let resolved = envelope
            .result_values
            .iter()
            .map(|column| column.clone())
            .collect::<Vec<_>>();
        let mut rows = (0..envelope.result_len)
            .map(|row| (resolved[0][row].clone(), resolved[1][row].clone()))
            .collect::<Vec<_>>();
        rows.sort();
        assert_eq!(
            envelope.result_values[0],
            vec![
                CanonicalResultValue::Object(CanonicalObject::Node(iri("a").into_bytes())),
                CanonicalResultValue::Object(CanonicalObject::Node(iri("b").into_bytes())),
            ]
        );
        assert!(rows.iter().any(|(key, sum)| {
            key == &CanonicalResultValue::Object(CanonicalObject::Node(iri("a").into_bytes()))
                && sum == &CanonicalResultValue::GeneratedDecimalU64(4)
        }));
        assert!(rows.iter().any(|(key, sum)| {
            key == &CanonicalResultValue::Object(CanonicalObject::Node(iri("b").into_bytes()))
                && sum == &CanonicalResultValue::GeneratedDecimalU64(u64::from(u32::MAX))
        }));
        let decoded = RecursiveRelationEnvelope::decode(&envelope.encode().unwrap()).unwrap();
        decoded
            .verify(&layer, layer.state.commitment_root, &plan)
            .unwrap();

        let mut tampered = decoded.clone();
        tampered.result_columns[1][0] =
            tampered.result_columns[1][0] + ArkFr::one();
        let mut padded = tampered.result_columns[1].clone();
        padded.resize(1usize << tampered.result_log_domain, ArkFr::zero());
        tampered.result_commitments[1] = commit_column(&padded).unwrap();
        assert!(tampered
            .verify(&layer, layer.state.commitment_root, &plan)
            .is_err());

        let global_plan = RelationProofPlan {
            query_digest: [103u8; 32],
            kind: RelationProofPlanKind::GroupSum {
                input: Box::new(source.clone()),
                key_columns: vec![],
                value_column: 2,
                datatype: CanonicalDatatype::UInt32,
            },
        };
        let global = RecursiveRelationEnvelope::prove(&layer, &global_plan).unwrap();
        global
            .verify(&layer, layer.state.commitment_root, &global_plan)
            .unwrap();
        assert_eq!(global.result_len, 1);
        assert_eq!(
            global.result_values,
            vec![vec![CanonicalResultValue::GeneratedDecimalU64(
                u64::from(u32::MAX) + 4,
            )]]
        );
        let mut forged_global = global.clone();
        forged_global.result_columns[0][0] = ArkFr::from_u64(u64::from(u32::MAX) + 5);
        forged_global.result_commitments[0] = commit_column(&[
            ArkFr::from_u64(u64::from(u32::MAX) + 5),
            ArkFr::zero(),
        ])
        .unwrap();
        assert!(forged_global
            .verify(&layer, layer.state.commitment_root, &global_plan)
            .is_err());

        let wrong_datatype = RelationProofPlan {
            query_digest: [100u8; 32],
            kind: RelationProofPlanKind::GroupSum {
                input: Box::new(source),
                key_columns: vec![0],
                value_column: 2,
                datatype: CanonicalDatatype::UInt16,
            },
        };
        assert!(decoded
            .verify(&layer, layer.state.commitment_root, &wrong_datatype)
            .is_err());

        let empty_source = RelationProofPlan {
            query_digest: [102u8; 32],
            kind: RelationProofPlanKind::Bgp(LayerBgpPlan {
                atoms: vec![LayerBgpAtom::all_variables(iri("missing").into_bytes())],
                joins: Vec::new(),
            }),
        };
        let empty_plan = RelationProofPlan {
            query_digest: [101u8; 32],
            kind: RelationProofPlanKind::GroupSum {
                input: Box::new(empty_source.clone()),
                key_columns: vec![0],
                value_column: 1,
                datatype: CanonicalDatatype::UInt32,
            },
        };
        let empty = RecursiveRelationEnvelope::prove(&layer, &empty_plan).unwrap();
        empty
            .verify(&layer, layer.state.commitment_root, &empty_plan)
            .unwrap();
        assert_eq!(empty.result_len, 0);
        assert_eq!(empty.result_values, vec![Vec::new(), Vec::new()]);

        let empty_global_plan = RelationProofPlan {
            query_digest: [104u8; 32],
            kind: RelationProofPlanKind::GroupSum {
                input: Box::new(empty_source),
                key_columns: vec![],
                value_column: 1,
                datatype: CanonicalDatatype::UInt32,
            },
        };
        let empty_global = RecursiveRelationEnvelope::prove(&layer, &empty_global_plan).unwrap();
        empty_global
            .verify(&layer, layer.state.commitment_root, &empty_global_plan)
            .unwrap();
        assert_eq!(empty_global.result_len, 0);
        assert_eq!(empty_global.result_values, vec![Vec::new()]);
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
        let envelope = RecursiveRelationEnvelope::prove(&layer, &plan).unwrap();
        envelope
            .verify(&layer, layer.state.commitment_root, &plan)
            .unwrap();
        assert_eq!(envelope.result_len, 4);
        let node = |name: &str| {
            CanonicalResultValue::Object(CanonicalObject::Node(iri(name).into_bytes()))
        };
        assert_eq!(envelope.result_values[0], vec![node("a"), node("a"), node("z"), node("z")]);
        assert_eq!(envelope.result_values[1], vec![node("a"), node("z"), node("a"), node("z")]);
        assert_eq!(
            envelope.result_values[2],
            vec![
                CanonicalResultValue::GeneratedDecimalU64(12),
                CanonicalResultValue::GeneratedDecimalU64(12),
                CanonicalResultValue::GeneratedDecimalU64(3),
                CanonicalResultValue::GeneratedDecimalU64(3),
            ]
        );
        let decoded = RecursiveRelationEnvelope::decode(&envelope.encode().unwrap()).unwrap();
        decoded
            .verify(&layer, layer.state.commitment_root, &plan)
            .unwrap();

        let reordered = RelationProofPlan {
            query_digest: [113u8; 32],
            kind: RelationProofPlanKind::GroupSum {
                input: Box::new(source.clone()),
                key_columns: vec![0, 1],
                value_column: 2,
                datatype: CanonicalDatatype::UInt32,
            },
        };
        assert!(decoded
            .verify(&layer, layer.state.commitment_root, &reordered)
            .is_err());
        for key_columns in [vec![0, 0], vec![0, 3], vec![0, 1, 2]] {
            let invalid_plan = RelationProofPlan {
                query_digest: [114u8; 32],
                kind: RelationProofPlanKind::GroupSum {
                    input: Box::new(source.clone()),
                    key_columns,
                    value_column: 2,
                    datatype: CanonicalDatatype::UInt32,
                },
            };
            assert!(invalid_plan.result_shape().is_err());
        }
        let mut tampered = decoded;
        tampered.result_columns[2][2] = ArkFr::from_u64(8);
        tampered.result_values[2][2] = CanonicalResultValue::GeneratedDecimalU64(8);
        tampered.result_commitments[2] = commit_column(&tampered.result_columns[2]).unwrap();
        assert!(tampered
            .verify(&layer, layer.state.commitment_root, &plan)
            .is_err());
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
        let envelope = RecursiveRelationEnvelope::prove(&layer, &plan).unwrap();
        envelope
            .verify(&layer, layer.state.commitment_root, &plan)
            .unwrap();
        let node = |name: &str| {
            CanonicalResultValue::Object(CanonicalObject::Node(iri(name).into_bytes()))
        };
        assert_eq!(envelope.result_len, 4);
        assert_eq!(envelope.result_values[0], vec![node("a"), node("a"), node("z"), node("z")]);
        assert_eq!(envelope.result_values[1], vec![node("a"), node("z"), node("a"), node("z")]);
        assert_eq!(
            envelope.result_values[2],
            vec![
                CanonicalResultValue::GeneratedDecimalU64(3),
                CanonicalResultValue::GeneratedDecimalU64(3),
                CanonicalResultValue::GeneratedDecimalU64(2),
                CanonicalResultValue::GeneratedDecimalU64(2),
            ]
        );
        let decoded = RecursiveRelationEnvelope::decode(&envelope.encode().unwrap()).unwrap();
        decoded
            .verify(&layer, layer.state.commitment_root, &plan)
            .unwrap();

        let reordered = RelationProofPlan {
            query_digest: [118u8; 32],
            kind: RelationProofPlanKind::GroupCount {
                input: Box::new(source.clone()),
                key_columns: vec![0, 1],
            },
        };
        assert!(decoded
            .verify(&layer, layer.state.commitment_root, &reordered)
            .is_err());
        for key_columns in [vec![0, 0], vec![0, 3], vec![0, 1, 2]] {
            let invalid = RelationProofPlan {
                query_digest: [119u8; 32],
                kind: RelationProofPlanKind::GroupCount {
                    input: Box::new(source.clone()),
                    key_columns,
                },
            };
            assert!(invalid.result_shape().is_err());
        }
        let mut tampered = decoded;
        tampered.result_columns[2][0] = ArkFr::from_u64(4);
        tampered.result_values[2][0] = CanonicalResultValue::GeneratedDecimalU64(4);
        tampered.result_commitments[2] = commit_column(&tampered.result_columns[2]).unwrap();
        assert!(tampered
            .verify(&layer, layer.state.commitment_root, &plan)
            .is_err());
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
                atoms: vec![LayerBgpAtom {
                    subject: AtomTerm::Variable,
                    predicate_term: Vec::new(),
                    predicate_variable: true,
                    object: AtomTerm::Variable,
                    object_range: None,
                    object_kind: ObjectKindFilter::Any,
                    subject_object_equal: false,
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
                atoms: vec![LayerBgpAtom {
                    subject: AtomTerm::Variable,
                    predicate_term: Vec::new(),
                    predicate_variable: true,
                    object: AtomTerm::Variable,
                    object_range: None,
                    object_kind: ObjectKindFilter::Any,
                    subject_object_equal: false,
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
        let keyed = RecursiveRelationEnvelope::prove(&layer, &keyed_plan).unwrap();
        keyed
            .verify(&layer, layer.state.commitment_root, &keyed_plan)
            .unwrap();
        assert_eq!(
            keyed.result_descriptors[1].namespace,
            ResultIdNamespace::GeneratedDecimalI64
        );
        assert_eq!(
            keyed.result_values[0],
            vec![
                CanonicalResultValue::Object(CanonicalObject::Node(iri("a").into_bytes())),
                CanonicalResultValue::Object(CanonicalObject::Node(iri("b").into_bytes())),
            ]
        );
        assert_eq!(
            keyed.result_values[1],
            vec![
                CanonicalResultValue::GeneratedDecimalI64(-1),
                CanonicalResultValue::GeneratedDecimalI64(0),
            ]
        );
        let decoded = RecursiveRelationEnvelope::decode(&keyed.encode().unwrap()).unwrap();
        decoded
            .verify(&layer, layer.state.commitment_root, &keyed_plan)
            .unwrap();

        let filter = |op, constant, constant_on_left| RelationProofPlan {
            query_digest: [111u8; 32],
            kind: RelationProofPlanKind::GeneratedI64Filter {
                input: Box::new(keyed_plan.clone()),
                column: 1,
                op,
                constant,
                constant_on_left,
            },
        };
        for (op, constant, constant_on_left, expected_len) in [
            (GeneratedU64ComparisonOp::Eq, -1, false, 1),
            (GeneratedU64ComparisonOp::Eq, 1, false, 0),
            (GeneratedU64ComparisonOp::Lt, 0, false, 1),
            (GeneratedU64ComparisonOp::Le, 0, false, 2),
            (GeneratedU64ComparisonOp::Gt, -1, false, 1),
            (GeneratedU64ComparisonOp::Ge, -1, false, 2),
            (GeneratedU64ComparisonOp::Lt, -1, true, 1),
            (GeneratedU64ComparisonOp::Le, -1, true, 2),
            (GeneratedU64ComparisonOp::Ge, i64::MIN, false, 2),
            (GeneratedU64ComparisonOp::Le, i64::MAX, false, 2),
        ] {
            let plan = filter(op, constant, constant_on_left);
            let envelope = RecursiveRelationEnvelope::prove(&layer, &plan).unwrap();
            envelope
                .verify(&layer, layer.state.commitment_root, &plan)
                .unwrap();
            assert_eq!(envelope.result_len, expected_len);
            RecursiveRelationEnvelope::decode(&envelope.encode().unwrap())
                .unwrap()
                .verify(&layer, layer.state.commitment_root, &plan)
                .unwrap();
        }

        let retained_plan = filter(GeneratedU64ComparisonOp::Lt, 0, false);
        let retained = RecursiveRelationEnvelope::prove(&layer, &retained_plan).unwrap();
        assert_eq!(
            retained.result_values[1],
            vec![CanonicalResultValue::GeneratedDecimalI64(-1)]
        );
        assert!(retained
            .verify(
                &layer,
                layer.state.commitment_root,
                &filter(GeneratedU64ComparisonOp::Le, 0, false),
            )
            .is_err());
        assert!(retained
            .verify(
                &layer,
                layer.state.commitment_root,
                &filter(GeneratedU64ComparisonOp::Lt, 0, true),
            )
            .is_err());
        let wrong_unsigned_domain = RelationProofPlan {
            query_digest: [111u8; 32],
            kind: RelationProofPlanKind::GeneratedU64Filter {
                input: Box::new(keyed_plan.clone()),
                column: 1,
                op: GeneratedU64ComparisonOp::Lt,
                constant: 0,
                constant_on_left: false,
            },
        };
        assert!(wrong_unsigned_domain.result_shape().is_err());
        let mut filtered_tamper = retained;
        filtered_tamper.result_columns[1][0] = generated_decimal_i64_field(1);
        filtered_tamper.result_values[1][0] = CanonicalResultValue::GeneratedDecimalI64(1);
        filtered_tamper.result_commitments[1] = commit_column(&[
            generated_decimal_i64_field(1),
            ArkFr::zero(),
        ])
        .unwrap();
        assert!(filtered_tamper
            .verify(&layer, layer.state.commitment_root, &retained_plan)
            .is_err());

        let mut sign_tamper = decoded.clone();
        sign_tamper.result_columns[1][0] = generated_decimal_i64_field(1);
        let mut padded = sign_tamper.result_columns[1].clone();
        padded.resize(1usize << sign_tamper.result_log_domain, ArkFr::zero());
        sign_tamper.result_commitments[1] = commit_column(&padded).unwrap();
        assert!(sign_tamper
            .verify(&layer, layer.state.commitment_root, &keyed_plan)
            .is_err());

        let mut domain_tamper = decoded.clone();
        domain_tamper.result_descriptors[1].namespace =
            ResultIdNamespace::GeneratedDecimalU64;
        assert!(domain_tamper
            .verify(&layer, layer.state.commitment_root, &keyed_plan)
            .is_err());

        let global_plan = RelationProofPlan {
            query_digest: [110u8; 32],
            kind: RelationProofPlanKind::GroupSum {
                input: Box::new(source.clone()),
                key_columns: vec![],
                value_column: 2,
                datatype: CanonicalDatatype::Int64,
            },
        };
        let global = RecursiveRelationEnvelope::prove(&layer, &global_plan).unwrap();
        global
            .verify(&layer, layer.state.commitment_root, &global_plan)
            .unwrap();
        assert_eq!(
            global.result_values,
            vec![vec![CanonicalResultValue::GeneratedDecimalI64(-1)]]
        );

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
            .stack_size(32 * 1024 * 1024)
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
            .stack_size(16 * 1024 * 1024)
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
        std::thread::Builder::new().name("multi-order-proof".into()).stack_size(24 * 1024 * 1024)
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
            atoms: vec![LayerBgpAtom {
                subject: AtomTerm::NodeConstant(iri("a").into_bytes()),
                predicate_term: iri("knows").into_bytes(),
                predicate_variable: false,
                object: AtomTerm::NodeConstant(iri("b").into_bytes()),
                object_range: None,
                object_kind: ObjectKindFilter::Any,
                subject_object_equal: false,
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
}
