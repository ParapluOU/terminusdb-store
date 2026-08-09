//! Versioned, bounded wire envelope for an explicitly generated layer BGP proof.

use std::convert::{TryFrom, TryInto};
use std::io::{self, Cursor};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::transcript::Transcript;
use sha2::{Digest, Sha256};

use super::anti::{
    verify_global_left_outer_join, verify_global_not, verify_left_anti_join_with_keys,
    verify_left_outer_join_with_keys, LeftAntiJoinProof, LeftOuterJoinProof,
};
use super::argument::{ColumnOpening, SumcheckProof};
use super::bgp::{BgpChainProof, BgpStageProof, JoinStep, RelationCommit};
use super::commitment::LayerCommitment;
use super::count::CountProof;
use super::dict::DictProof;
use super::filter::{
    AtomFilterProof, BooleanProductProof, CompactionOnlyProof, FilterProof, FullFilterProof,
    SelectionOnlyProof,
};
use super::join2::{BinaryJoinCommitments, BinaryJoinProof, BinaryPairsProof};
use super::membership::MembershipProof;
use super::query::{
    has_ground_result_unit, resolve_typed_result_objects, result_id_namespaces,
    resolve_result_values, verify_layer_bgp, verify_layer_bgp_result_log_domain,
    verify_result_columns, AtomTerm, CanonicalResultValue, GroundAbsence, LayerBgpAtom,
    LayerBgpAtomProof, LayerBgpPlan, LayerBgpProof, ObjectRangeBound, ObjectRangeSpec,
    RangeAtomSubproof, ResultColumnDescriptor, ResultIdNamespace, SecondRangeSubproof,
};
use super::range::{
    DictLookupProof, ObjectRangeCombineProof, ObjectRangeProof, RangeOp, RangeSelectProof,
};
use super::relocate::RelocationProof;
use super::sort::SortedProof;
use super::union::{verify_bag_union, BagUnionProof};
use super::{CanonicalDatatype, CanonicalObject, ProofHash};

const MAGIC: &[u8; 12] = b"TDBWQPROOF\0\0";
const UNION_MAGIC: &[u8; 12] = b"TDBWQUNION\0\0";
const ANTI_MAGIC: &[u8; 12] = b"TDBWQANTI\0\0\0";
const OUTER_MAGIC: &[u8; 12] = b"TDBWQOUTER\0\0";
const RECURSIVE_MAGIC: &[u8; 12] = b"TDBWQTREE\0\0\0";
// v3 adds the root-authenticated ground dictionary-absence branch. Older envelopes fail
// explicitly; v2 had already bound the deterministic Dory parameter-set identity.
const VERSION: u32 = 4;
const UNION_VERSION: u32 = 2;
const ANTI_VERSION: u32 = 4;
const OUTER_VERSION: u32 = 1;
const RECURSIVE_VERSION: u32 = 1;
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
    ground_absence
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
    subject_object_equal
});
wire_struct!(LayerBgpPlan { atoms, joins });

impl Wire for GroundAbsence {
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
            _ => Err(invalid("invalid ground-absence tag")),
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
        }
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        match u8::get(reader)? {
            0 => Ok(Self::NodeValue),
            1 => Ok(Self::Predicate),
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
        }
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        match u8::get(reader)? {
            0 => Ok(Self::Object(Wire::get(reader)?)),
            1 => Ok(Self::Null),
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
        layer: &LayerCommitment,
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
        layer: &LayerCommitment,
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
        layer: &LayerCommitment,
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
        layer: &LayerCommitment,
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
        let mut reader = Reader { bytes: body, pos: 0 };
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
    BagUnion {
        branches: Vec<RelationProofPlan>,
        /// Parent output column -> child output column, once per ordered branch.
        mappings: Vec<Vec<usize>>,
    },
    LeftAnti {
        left: Box<RelationProofPlan>,
        right: Box<RelationProofPlan>,
        key_pairs: Vec<(usize, usize)>,
    },
    LeftOuter {
        left: Box<RelationProofPlan>,
        right: Box<RelationProofPlan>,
        key_pairs: Vec<(usize, usize)>,
    },
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
                    if l.namespace != r.namespace || l.nullable {
                        return Err(invalid("anti correlation key descriptor mismatch"));
                    }
                }
                Ok(left_shape)
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
                    if l.namespace != r.namespace || l.nullable || shared_right.contains(right_index) {
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
    BagUnion {
        branches: Vec<RecursiveRelationEnvelope>,
        proof: Box<BagUnionProof>,
    },
    LeftAnti {
        left: Box<RecursiveRelationEnvelope>,
        right: Box<RecursiveRelationEnvelope>,
        proof: Option<Box<LeftAntiJoinProof>>,
    },
    LeftOuter {
        left: Box<RecursiveRelationEnvelope>,
        right: Box<RecursiveRelationEnvelope>,
        proof: Box<LeftOuterJoinProof>,
    },
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

    fn verify_bounded(
        &self,
        layer: &LayerCommitment,
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
        layer: &LayerCommitment,
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
        let mut reader = Reader { bytes: body, pos: 0 };
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
    fn ground_absence_wire_tags_are_strict() {
        for (tag, expected) in [
            (0, GroundAbsence::Predicate),
            (1, GroundAbsence::Subject),
            (2, GroundAbsence::Object),
        ] {
            let mut reader = Reader {
                bytes: &[tag],
                pos: 0,
            };
            assert_eq!(GroundAbsence::get(&mut reader).unwrap(), expected);
            reader.finish().unwrap();
        }
        let mut reader = Reader {
            bytes: &[3],
            pos: 0,
        };
        assert!(GroundAbsence::get(&mut reader).is_err());
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
        wrong_version[MAGIC.len()..MAGIC.len() + 4].copy_from_slice(&5u32.to_le_bytes());
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
