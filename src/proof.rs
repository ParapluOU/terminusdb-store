//! Reusable dictionary-encoded column/row model for authenticated Store deltas.
//!
//! This module keeps the scheme-independent pieces of the proof pipeline: the
//! canonical datatype/object/triple model, the dictionary-encoded `ProofRecord`
//! column tables, deterministic per-index ordering, and the framing/resource-limit
//! helpers used to (de)serialize commitment files.
//!
//! The per-layer sidecar payload itself lives in [`commitment`], which replaces
//! the abandoned SHA-256 Merkle scheme with a homomorphic column-commitment
//! scheme ("Proof-of-WOQL").
//!
//! This is the crate's proof root: it declares the whole `proof` module tree and
//! owns the on-disk framing, but the actual cryptography lives in the submodules.
//! If a doc here (or in any submodule) uses a term like *commitment*, *sumcheck*,
//! *gadget*, *Fiat–Shamir*, or *layer root* and it is unfamiliar, read
//! [the developer primer](crate::proof::primer) first — every submodule links back
//! to it rather than re-explaining the crypto.

use std::cmp::Ordering;
use std::convert::{TryFrom, TryInto};
use std::io;

use crate::layer::{ObjectType, ValueTriple};
use tdb_succinct::Datatype;

/// 32-byte sidecar root type. Retained as an opaque root/anchor identifier used
/// by the storage plumbing (expected-roots maps, commitment roots).
pub type ProofHash = [u8; 32];

const MAX_RECORDS: usize = 10_000_000;
const PERSISTED_TABLE_COUNT: usize = 5;
// Every one of v4's five persisted sections has the same per-table record cap.
// Aggregate admission is subsequently constrained much more tightly by the
// checked file and decoded-heap budgets; this count guard exists for overflow and
// structural consistency, not as a contradictory lower deployment ceiling.
const MAX_DECODED_RECORDS: usize = MAX_RECORDS * PERSISTED_TABLE_COUNT;
const MAX_FIELD_BYTES: usize = 16 * 1024 * 1024;
/// Hard ceiling on the ENCODED size of one persisted proof-layer sidecar file.
/// Decoding refuses to read (and encoding refuses to produce) anything larger,
/// so a hostile or corrupt file cannot force an unbounded read.
pub const MAX_PROOF_LAYER_BYTES: usize = 256 * 1024 * 1024;
/// Hard ceiling on the estimated RETAINED heap a single decoded sidecar may
/// occupy. Charged incrementally while decoding (before each allocation), so a
/// compact-but-explosive encoding fails closed before it can exhaust memory.
pub const MAX_DECODED_PROOF_BYTES: usize = 512 * 1024 * 1024;

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn validate_field_len(len: usize) -> io::Result<()> {
    if len > MAX_FIELD_BYTES {
        return Err(invalid("proof field too large"));
    }
    Ok(())
}

/// Scheme-independent enumeration of every literal datatype the proof pipeline
/// can carry, decoupled from the Store's own `tdb_succinct::Datatype`. The `u8`
/// discriminant is the stable on-disk tag: it is written into serialized records
/// and folded into the committed DATATYPE column, so variants are only ever
/// APPENDED (never reordered or removed) to keep old commitment roots valid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum CanonicalDatatype {
    String = 0,
    UInt32,
    Int32,
    Float32,
    UInt64,
    Int64,
    Float64,
    Decimal,
    BigInt,
    Boolean,
    LangString,
    AnyUri,
    Language,
    NormalizedString,
    Token,
    NmToken,
    Name,
    NcName,
    Notation,
    QName,
    Id,
    IdRef,
    Entity,
    PositiveInteger,
    NonNegativeInteger,
    NonPositiveInteger,
    NegativeInteger,
    Date,
    DateTime,
    DateTimeStamp,
    Time,
    GYear,
    GMonth,
    GDay,
    GYearMonth,
    GMonthDay,
    Duration,
    YearMonthDuration,
    DayTimeDuration,
    UInt8,
    Int8,
    UInt16,
    Int16,
    Base64Binary,
    HexBinary,
    AnySimpleType,
    // Appended before proof format v4 when TerminusDB's pinned tdb-succinct
    // gained ISO interval values; existing datatype tags remain stable.
    DateTimeInterval,
}

impl CanonicalDatatype {
    /// Map a Store `Datatype` onto its stable canonical tag. Total (every Store
    /// datatype has a counterpart), so no proof-side datatype is ever lost.
    pub fn from_store(value: Datatype) -> Self {
        match value {
            Datatype::String => Self::String,
            Datatype::UInt32 => Self::UInt32,
            Datatype::Int32 => Self::Int32,
            Datatype::Float32 => Self::Float32,
            Datatype::UInt64 => Self::UInt64,
            Datatype::Int64 => Self::Int64,
            Datatype::Float64 => Self::Float64,
            Datatype::Decimal => Self::Decimal,
            Datatype::BigInt => Self::BigInt,
            Datatype::Boolean => Self::Boolean,
            Datatype::LangString => Self::LangString,
            Datatype::AnyURI => Self::AnyUri,
            Datatype::Language => Self::Language,
            Datatype::NormalizedString => Self::NormalizedString,
            Datatype::Token => Self::Token,
            Datatype::NMToken => Self::NmToken,
            Datatype::Name => Self::Name,
            Datatype::NCName => Self::NcName,
            Datatype::Notation => Self::Notation,
            Datatype::QName => Self::QName,
            Datatype::ID => Self::Id,
            Datatype::IDRef => Self::IdRef,
            Datatype::Entity => Self::Entity,
            Datatype::PositiveInteger => Self::PositiveInteger,
            Datatype::NonNegativeInteger => Self::NonNegativeInteger,
            Datatype::NonPositiveInteger => Self::NonPositiveInteger,
            Datatype::NegativeInteger => Self::NegativeInteger,
            Datatype::Date => Self::Date,
            Datatype::DateTime => Self::DateTime,
            Datatype::DateTimeStamp => Self::DateTimeStamp,
            Datatype::Time => Self::Time,
            Datatype::GYear => Self::GYear,
            Datatype::GMonth => Self::GMonth,
            Datatype::GDay => Self::GDay,
            Datatype::GYearMonth => Self::GYearMonth,
            Datatype::GMonthDay => Self::GMonthDay,
            Datatype::Duration => Self::Duration,
            Datatype::YearMonthDuration => Self::YearMonthDuration,
            Datatype::DayTimeDuration => Self::DayTimeDuration,
            Datatype::UInt8 => Self::UInt8,
            Datatype::Int8 => Self::Int8,
            Datatype::UInt16 => Self::UInt16,
            Datatype::Int16 => Self::Int16,
            Datatype::Base64Binary => Self::Base64Binary,
            Datatype::HexBinary => Self::HexBinary,
            Datatype::AnySimpleType => Self::AnySimpleType,
            Datatype::DateTimeInterval => Self::DateTimeInterval,
        }
    }
}

impl TryFrom<u8> for CanonicalDatatype {
    type Error = io::Error;

    fn try_from(value: u8) -> io::Result<Self> {
        const ALL: [CanonicalDatatype; 47] = [
            CanonicalDatatype::String,
            CanonicalDatatype::UInt32,
            CanonicalDatatype::Int32,
            CanonicalDatatype::Float32,
            CanonicalDatatype::UInt64,
            CanonicalDatatype::Int64,
            CanonicalDatatype::Float64,
            CanonicalDatatype::Decimal,
            CanonicalDatatype::BigInt,
            CanonicalDatatype::Boolean,
            CanonicalDatatype::LangString,
            CanonicalDatatype::AnyUri,
            CanonicalDatatype::Language,
            CanonicalDatatype::NormalizedString,
            CanonicalDatatype::Token,
            CanonicalDatatype::NmToken,
            CanonicalDatatype::Name,
            CanonicalDatatype::NcName,
            CanonicalDatatype::Notation,
            CanonicalDatatype::QName,
            CanonicalDatatype::Id,
            CanonicalDatatype::IdRef,
            CanonicalDatatype::Entity,
            CanonicalDatatype::PositiveInteger,
            CanonicalDatatype::NonNegativeInteger,
            CanonicalDatatype::NonPositiveInteger,
            CanonicalDatatype::NegativeInteger,
            CanonicalDatatype::Date,
            CanonicalDatatype::DateTime,
            CanonicalDatatype::DateTimeStamp,
            CanonicalDatatype::Time,
            CanonicalDatatype::GYear,
            CanonicalDatatype::GMonth,
            CanonicalDatatype::GDay,
            CanonicalDatatype::GYearMonth,
            CanonicalDatatype::GMonthDay,
            CanonicalDatatype::Duration,
            CanonicalDatatype::YearMonthDuration,
            CanonicalDatatype::DayTimeDuration,
            CanonicalDatatype::UInt8,
            CanonicalDatatype::Int8,
            CanonicalDatatype::UInt16,
            CanonicalDatatype::Int16,
            CanonicalDatatype::Base64Binary,
            CanonicalDatatype::HexBinary,
            CanonicalDatatype::AnySimpleType,
            CanonicalDatatype::DateTimeInterval,
        ];
        ALL.get(value as usize)
            .copied()
            .ok_or_else(|| invalid("unknown canonical datatype"))
    }
}

/// A triple's object in the scheme-independent proof model: either a graph
/// `Node` (an IRI, carried as raw bytes) or a typed literal `Value` (a
/// `CanonicalDatatype` plus its lexical bytes). This is what the dictionary
/// hashes, the order-preserving VALUE encoding, and the Merkle leaves all
/// decompose; the `Ord` derive gives the canonical ordering the `Value`/dict
/// indices sort by.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CanonicalObject {
    /// A graph node / IRI, stored as its raw UTF-8 bytes.
    Node(Vec<u8>),
    /// A typed literal: its datatype tag plus the lexical form's bytes.
    Value {
        datatype: CanonicalDatatype,
        lexical: Vec<u8>,
    },
}

impl CanonicalObject {
    /// Copy a Store object into the versioned proof representation.
    pub fn from_store_object(object: &ObjectType) -> Self {
        match object {
            ObjectType::Node(node) => Self::Node(node.as_bytes().to_vec()),
            ObjectType::Value(value) => Self::Value {
                datatype: CanonicalDatatype::from_store(value.datatype()),
                lexical: value.to_bytes().to_vec(),
            },
        }
    }
}

/// A whole triple in the canonical proof model: subject and predicate IRIs as
/// raw bytes, plus a [`CanonicalObject`]. These self-contained bytes travel with
/// each [`ProofRecord`] so the verifier never needs a Store dictionary to read
/// the logical term.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CanonicalTriple {
    pub subject: Vec<u8>,
    pub predicate: Vec<u8>,
    pub object: CanonicalObject,
}

/// Native Store ordering keys paired with self-contained logical bytes.
/// The checker never resolves these ids through a Store dictionary; dictionary
/// witnesses bind each query constant to the corresponding id separately.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProofRecord {
    pub subject_id: u64,
    pub predicate_id: u64,
    pub object_id: u64,
    pub triple: CanonicalTriple,
}

impl CanonicalTriple {
    /// Build a triple whose object is a graph `Node` (IRI), from raw byte
    /// operands. Convenience for the common node-object case (mostly tests and
    /// predicate rows); typed-literal objects go through [`CanonicalObject`].
    pub fn node(
        subject: impl Into<Vec<u8>>,
        predicate: impl Into<Vec<u8>>,
        object: impl Into<Vec<u8>>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: CanonicalObject::Node(object.into()),
        }
    }

    fn encode_into(&self, out: &mut Vec<u8>) {
        put_bytes(out, &self.subject);
        put_bytes(out, &self.predicate);
        match &self.object {
            CanonicalObject::Node(value) => {
                out.push(0);
                put_bytes(out, value);
            }
            CanonicalObject::Value { datatype, lexical } => {
                out.push(1);
                out.push(*datatype as u8);
                put_bytes(out, lexical);
            }
        }
    }

    fn encoded_len(&self) -> io::Result<usize> {
        validate_field_len(self.subject.len())?;
        validate_field_len(self.predicate.len())?;
        let (object_len, object_tag_len) = match &self.object {
            CanonicalObject::Node(value) => (value.len(), 1usize),
            CanonicalObject::Value { lexical, .. } => (lexical.len(), 2usize),
        };
        validate_field_len(object_len)?;
        8usize
            .checked_add(self.subject.len())
            .and_then(|value| value.checked_add(8))
            .and_then(|value| value.checked_add(self.predicate.len()))
            .and_then(|value| value.checked_add(object_tag_len))
            .and_then(|value| value.checked_add(8))
            .and_then(|value| value.checked_add(object_len))
            .ok_or_else(|| invalid("proof record encoded length overflow"))
    }

    /// Copy a logical Store triple into the canonical proof representation.
    /// Store ids and later dictionary lookups are intentionally not retained.
    pub fn from_value_triple(triple: &ValueTriple) -> Self {
        Self {
            subject: triple.subject.as_bytes().to_vec(),
            predicate: triple.predicate.as_bytes().to_vec(),
            object: CanonicalObject::from_store_object(&triple.object),
        }
    }
}

/// The deterministic sort orders a record table can be projected into. Each
/// variant names a total order over [`ProofRecord`]s (see `triple_cmp`): the
/// three native triple indices (`Spo`/`Pos`/`Osp`) mirror the Store's own index
/// orders, `Value` orders by the object term (for range/order gadgets), and the
/// three `*Dictionary` variants order by a term + its id (for dictionary
/// commitments). Choosing a canonical order per index is what makes two honest
/// provers commit byte-identical columns.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum ProofIndex {
    Spo = 0,
    Pos = 1,
    Osp = 2,
    Value = 3,
    SubjectDictionary = 4,
    PredicateDictionary = 5,
    ObjectDictionary = 6,
}

fn triple_cmp(a: &ProofRecord, b: &ProofRecord, index: ProofIndex) -> Ordering {
    match index {
        ProofIndex::Spo => (a.subject_id, a.predicate_id, a.object_id).cmp(&(
            b.subject_id,
            b.predicate_id,
            b.object_id,
        )),
        // Native `triples_p` merges IdTriple values, whose order after a fixed
        // predicate is subject then object.
        ProofIndex::Pos => (a.predicate_id, a.subject_id, a.object_id).cmp(&(
            b.predicate_id,
            b.subject_id,
            b.object_id,
        )),
        ProofIndex::Osp => (a.object_id, a.subject_id, a.predicate_id).cmp(&(
            b.object_id,
            b.subject_id,
            b.predicate_id,
        )),
        ProofIndex::Value => (&a.triple.object, a.subject_id, a.predicate_id).cmp(&(
            &b.triple.object,
            b.subject_id,
            b.predicate_id,
        )),
        ProofIndex::SubjectDictionary => {
            (&a.triple.subject, a.subject_id).cmp(&(&b.triple.subject, b.subject_id))
        }
        ProofIndex::PredicateDictionary => {
            (&a.triple.predicate, a.predicate_id).cmp(&(&b.triple.predicate, b.predicate_id))
        }
        ProofIndex::ObjectDictionary => {
            (&a.triple.object, a.object_id).cmp(&(&b.triple.object, b.object_id))
        }
    }
}

// ---------------------------------------------------------------------------
// Serialization / resource-limit framing (scheme-independent)
// ---------------------------------------------------------------------------

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}
fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_be_bytes());
}
fn put_bytes(out: &mut Vec<u8>, value: &[u8]) {
    put_u64(out, value.len() as u64);
    out.extend_from_slice(value);
}
fn put_hash(out: &mut Vec<u8>, value: &ProofHash) {
    out.extend_from_slice(value);
}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
    decoded_records: usize,
    decoded_heap_estimate: usize,
}
impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            position: 0,
            decoded_records: 0,
            decoded_heap_estimate: 0,
        }
    }
    fn take(&mut self, len: usize) -> io::Result<&'a [u8]> {
        let end = self
            .position
            .checked_add(len)
            .ok_or_else(|| invalid("proof length overflow"))?;
        let value = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| invalid("truncated proof file"))?;
        self.position = end;
        Ok(value)
    }
    fn u8(&mut self) -> io::Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_be_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_be_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn hash(&mut self) -> io::Result<ProofHash> {
        Ok(self.take(32)?.try_into().unwrap())
    }
    fn bytes(&mut self) -> io::Result<Vec<u8>> {
        let len = usize::try_from(self.u64()?).map_err(|_| invalid("field length overflow"))?;
        validate_field_len(len)?;
        // Reserve before `to_vec`: hostile lengths must fail before the first
        // dynamic allocation. Charge twice because catalog admission may hold
        // a transient cloned key while the decoded field is still live.
        self.reserve_decoded_heap(
            len.checked_mul(2)
                .ok_or_else(|| invalid("decoded proof field estimate overflow"))?,
        )?;
        Ok(self.take(len)?.to_vec())
    }
    fn done(&self) -> bool {
        self.position == self.bytes.len()
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.position)
    }

    fn reserve_decoded_heap(&mut self, bytes: usize) -> io::Result<()> {
        self.decoded_heap_estimate = self
            .decoded_heap_estimate
            .checked_add(bytes)
            .ok_or_else(|| invalid("decoded proof heap estimate overflow"))?;
        if self.decoded_heap_estimate > MAX_DECODED_PROOF_BYTES {
            return Err(invalid("decoded proof layer exceeds heap budget"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum PersistedTableKind {
    Delta,
    PredicateCatalog,
    ObjectCatalog,
}

fn read_records(r: &mut Reader<'_>, kind: PersistedTableKind) -> io::Result<Vec<ProofRecord>> {
    let count = usize::try_from(r.u64()?).map_err(|_| invalid("record count overflow"))?;
    if count > MAX_RECORDS {
        return Err(invalid("too many proof records"));
    }
    r.decoded_records = r
        .decoded_records
        .checked_add(count)
        .ok_or_else(|| invalid("proof record count overflow"))?;
    if r.decoded_records > MAX_DECODED_RECORDS {
        return Err(invalid("proof layer exceeds aggregate record budget"));
    }
    let record_table = count
        .checked_mul(std::mem::size_of::<ProofRecord>())
        .ok_or_else(|| invalid("decoded proof record heap estimate overflow"))?;
    let projection = index_projection_heap(count)?;
    let minimum_heap = match kind {
        PersistedTableKind::Delta => record_table
            .checked_add(
                projection
                    .checked_mul(4)
                    .ok_or_else(|| invalid("decoded proof projection estimate overflow"))?,
            )
            .ok_or_else(|| invalid("decoded proof table estimate overflow"))?,
        PersistedTableKind::PredicateCatalog => record_table
            .checked_add(projection)
            .and_then(|value| value.checked_add(count.checked_mul(64)?))
            .ok_or_else(|| invalid("decoded predicate catalog estimate overflow"))?,
        PersistedTableKind::ObjectCatalog => record_table
            .checked_mul(2)
            .and_then(|value| value.checked_add(projection.checked_mul(2)?))
            .and_then(|value| value.checked_add(count.checked_mul(64)?))
            .ok_or_else(|| invalid("decoded object catalog estimate overflow"))?,
    };
    r.reserve_decoded_heap(minimum_heap)?;
    let minimum_serialized = count
        .checked_mul(49)
        .ok_or_else(|| invalid("proof record byte count overflow"))?;
    if minimum_serialized > r.remaining() {
        return Err(invalid("truncated proof record table"));
    }
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let subject_id = r.u64()?;
        let predicate_id = r.u64()?;
        let object_id = r.u64()?;
        let subject = r.bytes()?;
        let predicate = r.bytes()?;
        let object = match r.u8()? {
            0 => CanonicalObject::Node(r.bytes()?),
            1 => CanonicalObject::Value {
                datatype: CanonicalDatatype::try_from(r.u8()?)?,
                lexical: r.bytes()?,
            },
            _ => return Err(invalid("bad object tag")),
        };
        out.push(ProofRecord {
            subject_id,
            predicate_id,
            object_id,
            triple: CanonicalTriple {
                subject,
                predicate,
                object,
            },
        });
    }
    Ok(out)
}

fn checked_proof_encoded_growth(total: usize, addition: usize) -> io::Result<usize> {
    let updated = total
        .checked_add(addition)
        .ok_or_else(|| invalid("proof encoded length overflow"))?;
    if updated > MAX_PROOF_LAYER_BYTES {
        return Err(invalid("encoded proof layer exceeds file budget"));
    }
    Ok(updated)
}

fn checked_heap_add(total: &mut usize, value: usize) -> io::Result<()> {
    *total = total
        .checked_add(value)
        .ok_or_else(|| invalid("decoded proof heap estimate overflow"))?;
    if *total > MAX_DECODED_PROOF_BYTES {
        return Err(invalid("decoded proof layer exceeds heap budget"));
    }
    Ok(())
}

/// Incremental admission budget used while rebuilding a proof sidecar from
/// native Store iterators. The final constructor performs a second, complete
/// retained-heap calculation before it builds the sidecar; this guard prevents
/// the input vectors themselves from growing past the same cap first.
pub(crate) struct ProofConstructionBudget {
    bytes: usize,
}

impl ProofConstructionBudget {
    /// Start a fresh admission budget at zero charged bytes.
    pub(crate) fn new() -> Self {
        Self { bytes: 0 }
    }

    fn reserve(&mut self, bytes: usize) -> io::Result<()> {
        checked_heap_add(&mut self.bytes, bytes)
    }

    /// Charge the estimated retained heap of one triple record (element table
    /// growth plus its owned field bytes) against the cap, failing closed if the
    /// running total would exceed [`MAX_DECODED_PROOF_BYTES`].
    pub(crate) fn record(&mut self, record: &ProofRecord) -> io::Result<()> {
        validate_field_len(record.triple.subject.len())?;
        validate_field_len(record.triple.predicate.len())?;
        let object = match &record.triple.object {
            CanonicalObject::Node(value) => value,
            CanonicalObject::Value { lexical, .. } => lexical,
        };
        validate_field_len(object.len())?;
        let dynamic = record
            .triple
            .subject
            .len()
            .checked_add(record.triple.predicate.len())
            .and_then(|value| value.checked_add(object.len()))
            .ok_or_else(|| invalid("proof construction record estimate overflow"))?;
        // Vec capacity growth can retain almost twice the element table while
        // the owned field buffers are exact-sized Store conversions.
        self.reserve(
            std::mem::size_of::<ProofRecord>()
                .checked_mul(2)
                .and_then(|value| value.checked_add(dynamic))
                .ok_or_else(|| invalid("proof construction record estimate overflow"))?,
        )
    }

    /// Charge one node/value dictionary catalog entry (an object term plus its
    /// id) against the cap.
    pub(crate) fn node_value(&mut self, object: &CanonicalObject) -> io::Result<()> {
        let dynamic = match object {
            CanonicalObject::Node(value) => value.len(),
            CanonicalObject::Value { lexical, .. } => lexical.len(),
        };
        validate_field_len(dynamic)?;
        self.reserve(
            std::mem::size_of::<(CanonicalObject, u64)>()
                .checked_mul(2)
                .and_then(|value| value.checked_add(dynamic))
                .ok_or_else(|| invalid("proof construction catalog estimate overflow"))?,
        )
    }

    /// Charge one predicate dictionary catalog entry (a predicate term plus its
    /// id) against the cap.
    pub(crate) fn predicate(&mut self, predicate: &[u8]) -> io::Result<()> {
        validate_field_len(predicate.len())?;
        self.reserve(
            std::mem::size_of::<(Vec<u8>, u64)>()
                .checked_mul(2)
                .and_then(|value| value.checked_add(predicate.len()))
                .ok_or_else(|| invalid("proof construction catalog estimate overflow"))?,
        )
    }
}

fn record_table_heap(records: &[ProofRecord]) -> io::Result<usize> {
    records.iter().try_fold(
        records
            .len()
            .checked_mul(std::mem::size_of::<ProofRecord>())
            .ok_or_else(|| invalid("decoded proof record heap estimate overflow"))?,
        |total, record| {
            let object = match &record.triple.object {
                CanonicalObject::Node(value) => value.len(),
                CanonicalObject::Value { lexical, .. } => lexical.len(),
            };
            total
                .checked_add(record.triple.subject.len())
                .and_then(|value| value.checked_add(record.triple.predicate.len()))
                .and_then(|value| value.checked_add(object))
                .ok_or_else(|| invalid("decoded proof record heap estimate overflow"))
        },
    )
}

fn index_projection_heap(records: usize) -> io::Result<usize> {
    let capacity = records
        .max(1)
        .checked_next_power_of_two()
        .ok_or_else(|| invalid("decoded proof index capacity overflow"))?;
    let hashes = capacity
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or_else(|| invalid("decoded proof hash count overflow"))?;
    records
        .checked_mul(std::mem::size_of::<usize>())
        .and_then(|value| value.checked_add(hashes.checked_mul(32)?))
        .ok_or_else(|| invalid("decoded proof index heap estimate overflow"))
}

fn validate_decoded_heap_budget(
    additions: &[ProofRecord],
    removals: &[ProofRecord],
    materialized: &[ProofRecord],
    predicates: &[ProofRecord],
    objects: &[ProofRecord],
) -> io::Result<()> {
    let mut total = 0usize;
    for records in [additions, removals, materialized, predicates, objects] {
        checked_heap_add(&mut total, record_table_heap(records)?)?;
    }
    for records in [additions, removals, materialized] {
        let projection = index_projection_heap(records.len())?;
        checked_heap_add(
            &mut total,
            projection
                .checked_mul(4)
                .ok_or_else(|| invalid("decoded proof projection estimate overflow"))?,
        )?;
    }
    checked_heap_add(&mut total, index_projection_heap(predicates.len())?)?;
    checked_heap_add(&mut total, index_projection_heap(objects.len())?)?;

    let subject_records = objects
        .iter()
        .filter(|record| matches!(record.triple.object, CanonicalObject::Node(_)))
        .count();
    let subject_dynamic_bytes = objects.iter().try_fold(0usize, |total, record| {
        let bytes = match &record.triple.object {
            CanonicalObject::Node(value) => value.len(),
            CanonicalObject::Value { .. } => 0,
        };
        total
            .checked_add(bytes)
            .ok_or_else(|| invalid("decoded proof subject catalog estimate overflow"))
    })?;
    let subject_table = subject_records
        .checked_mul(std::mem::size_of::<ProofRecord>())
        .and_then(|value| value.checked_add(subject_dynamic_bytes))
        .ok_or_else(|| invalid("decoded proof subject catalog estimate overflow"))?;
    checked_heap_add(&mut total, subject_table)?;
    checked_heap_add(&mut total, index_projection_heap(subject_records)?)?;

    // BTree maps are transient during admission. Charge a conservative node
    // allowance so a compact encoded catalog cannot bypass the retained-heap cap.
    let catalog_entries = predicates
        .len()
        .checked_add(objects.len())
        .ok_or_else(|| invalid("decoded proof catalog count overflow"))?;
    checked_heap_add(
        &mut total,
        catalog_entries
            .checked_mul(64)
            .ok_or_else(|| invalid("decoded proof catalog estimate overflow"))?,
    )?;
    checked_heap_add(
        &mut total,
        commitment::construction_commitment_scratch_bytes(&[
            additions.len(),
            removals.len(),
            materialized.len(),
            predicates.len(),
            objects.len(),
        ])?,
    )
}

fn validate_construction_heap_budget(
    additions: &[ProofRecord],
    removals: &[ProofRecord],
    materialized: &[ProofRecord],
    node_values: &[(CanonicalObject, u64)],
    predicates: &[(Vec<u8>, u64)],
) -> io::Result<()> {
    let mut total = 0usize;
    for records in [additions, removals, materialized] {
        checked_heap_add(&mut total, record_table_heap(records)?)?;
        checked_heap_add(
            &mut total,
            index_projection_heap(records.len())?
                .checked_mul(4)
                .ok_or_else(|| invalid("proof construction projection estimate overflow"))?,
        )?;
    }

    let predicate_dynamic = predicates.iter().try_fold(0usize, |sum, (value, _)| {
        sum.checked_add(value.len())
            .ok_or_else(|| invalid("predicate catalog heap estimate overflow"))
    })?;
    let predicate_table = predicates
        .len()
        .checked_mul(std::mem::size_of::<ProofRecord>())
        .and_then(|value| value.checked_add(predicate_dynamic))
        .ok_or_else(|| invalid("predicate catalog heap estimate overflow"))?;
    checked_heap_add(&mut total, predicate_table)?;
    checked_heap_add(&mut total, index_projection_heap(predicates.len())?)?;

    let (object_dynamic, subject_count, subject_dynamic) = node_values.iter().try_fold(
        (0usize, 0usize, 0usize),
        |(object_sum, subject_count, subject_sum), (object, _)| -> io::Result<_> {
            let bytes = match object {
                CanonicalObject::Node(value) => value.len(),
                CanonicalObject::Value { lexical, .. } => lexical.len(),
            };
            let object_sum = object_sum
                .checked_add(bytes)
                .ok_or_else(|| invalid("object catalog heap estimate overflow"))?;
            match object {
                CanonicalObject::Node(_) => Ok((
                    object_sum,
                    subject_count
                        .checked_add(1)
                        .ok_or_else(|| invalid("subject catalog count overflow"))?,
                    subject_sum
                        .checked_add(bytes)
                        .ok_or_else(|| invalid("subject catalog heap estimate overflow"))?,
                )),
                CanonicalObject::Value { .. } => Ok((object_sum, subject_count, subject_sum)),
            }
        },
    )?;
    let object_table = node_values
        .len()
        .checked_mul(std::mem::size_of::<ProofRecord>())
        .and_then(|value| value.checked_add(object_dynamic))
        .ok_or_else(|| invalid("object catalog heap estimate overflow"))?;
    checked_heap_add(&mut total, object_table)?;
    checked_heap_add(&mut total, index_projection_heap(node_values.len())?)?;
    let subject_table = subject_count
        .checked_mul(std::mem::size_of::<ProofRecord>())
        .and_then(|value| value.checked_add(subject_dynamic))
        .ok_or_else(|| invalid("subject catalog heap estimate overflow"))?;
    checked_heap_add(&mut total, subject_table)?;
    checked_heap_add(&mut total, index_projection_heap(subject_count)?)?;

    let catalog_entries = node_values
        .len()
        .checked_add(predicates.len())
        .ok_or_else(|| invalid("proof construction catalog count overflow"))?;
    checked_heap_add(
        &mut total,
        catalog_entries
            .checked_mul(64)
            .ok_or_else(|| invalid("proof construction catalog estimate overflow"))?,
    )?;
    checked_heap_add(
        &mut total,
        commitment::construction_commitment_scratch_bytes(&[
            additions.len(),
            removals.len(),
            materialized.len(),
            node_values.len(),
            predicates.len(),
        ])?,
    )
}

pub(crate) use commitment::decode_legacy_v3;
pub use commitment::{
    database_root, node_value_op_value, resolution_root, validate_database_root, CommitmentState,
    DictOpening, LayerCommitment, ResolutionOpenings, VerifierCommitment, VerifierCommitmentRef,
    VerifierLayer,
};

/// Conceptual reference for the whole proof subsystem — read this first.
/// Doc-only; other modules intra-doc-link here instead of re-explaining the crypto.
pub mod primer;
pub mod anti;
pub mod argument;
pub mod bgp;
pub mod checkpoint;
pub mod count;
pub mod dict;
pub mod dicthash;
pub mod disequality;
pub mod distinct;
pub mod envelope;
pub mod equality;
pub mod group;
pub mod filter;
pub mod join;
pub mod join2;
pub mod join_link;
pub mod logup;
pub mod logup_gkr;
pub mod membership;
pub mod order;
pub mod lexdistinct;
pub mod lexsort;
pub mod multiorder;
pub mod pairs;
pub mod permutation;
pub mod query;
pub mod range;
pub mod reconcile;
pub mod semijoin;
pub mod relocate;
pub mod sort;
pub mod union;

/// Per-layer sidecar payload for the Proof-of-WOQL commitment scheme.
///
/// This module REPLACES the abandoned SHA-256 Merkle sidecar (`ProofLayer` /
/// `ProofLayerState`). It exposes the same shape the storage plumbing depends
/// on: a validated per-layer payload carrying a small [`CommitmentState`]
/// (parent root + counts + a single root hash) plus the dictionary-encoded
/// column tables and their Dory commitments. `commitment_root` is the
/// domain-separated digest that binds those commitments, dictionaries, counts,
/// and parent root into the portable trust anchor.
pub mod commitment;

/// Independent smoke test that the `dory-pcs` BN254 commit/open/verify API works
/// end-to-end and rejects a tampered evaluation. LayerCommitment integration is
/// covered separately by the commitment and query-proof tests above.
#[cfg(test)]
mod dory_smoke {
    use dory_pcs::backends::arkworks::{
        ArkFr, ArkworksPolynomial, Blake2bTranscript, G1Routines, G2Routines, BN254,
    };
    use dory_pcs::primitives::arithmetic::Field;
    use dory_pcs::primitives::poly::Polynomial;
    use dory_pcs::{prove, setup, verify, Transparent};

    #[test]
    fn dory_commit_open_verify_roundtrip_and_tamper_fails() {
        // Transparent public-parameter setup, in-memory (the `disk-persistence`
        // feature is intentionally off, so no URS file is written).
        let (prover_setup, verifier_setup) = setup::<BN254>(6);

        // Small square layout: nu=2, sigma=2 => 16 dictionary-encoded scalars.
        let nu = 2usize;
        let sigma = 2usize;
        let num_vars = nu + sigma;
        let poly_size = 1usize << num_vars;

        let coefficients: Vec<ArkFr> = (0..poly_size).map(|_| ArkFr::random()).collect();
        let poly = ArkworksPolynomial::new(coefficients);

        // Commit to the column vector: tier-2 (GT) is the succinct commitment,
        // tier-1 are the per-row G1 commitments, commit_blind is 0 in Transparent.
        let (tier_2, tier_1, commit_blind) = poly
            .commit::<BN254, Transparent, G1Routines>(nu, sigma, &prover_setup)
            .expect("dory commit");

        // Open at a random challenge point and produce the evaluation proof.
        let point: Vec<ArkFr> = (0..num_vars).map(|_| ArkFr::random()).collect();
        let evaluation = poly.evaluate(&point);

        let mut prover_transcript = Blake2bTranscript::new(b"parture-dory-smoke");
        let (proof, _blind) = prove::<_, BN254, G1Routines, G2Routines, _, _, Transparent>(
            &poly,
            &point,
            tier_1,
            commit_blind,
            nu,
            sigma,
            &prover_setup,
            &mut prover_transcript,
        )
        .expect("dory prove");

        // Honest opening verifies (tier_2/GT is Copy; verifier_setup is by-value).
        let mut verifier_transcript = Blake2bTranscript::new(b"parture-dory-smoke");
        verify::<_, BN254, G1Routines, G2Routines, _>(
            tier_2,
            evaluation,
            &point,
            &proof,
            verifier_setup.clone(),
            &mut verifier_transcript,
        )
        .expect("honest dory opening must verify");

        // Tampering with the claimed evaluation must be rejected.
        let tampered = evaluation.add(&ArkFr::one());
        let mut tamper_transcript = Blake2bTranscript::new(b"parture-dory-smoke");
        assert!(
            verify::<_, BN254, G1Routines, G2Routines, _>(
                tier_2,
                tampered,
                &point,
                &proof,
                verifier_setup,
                &mut tamper_transcript,
            )
            .is_err(),
            "a tampered evaluation must fail Dory verification"
        );
    }
}
