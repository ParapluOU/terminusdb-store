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
pub const MAX_PROOF_LAYER_BYTES: usize = 256 * 1024 * 1024;
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

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CanonicalObject {
    Node(Vec<u8>),
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
    pub(crate) fn new() -> Self {
        Self { bytes: 0 }
    }

    fn reserve(&mut self, bytes: usize) -> io::Result<()> {
        checked_heap_add(&mut self.bytes, bytes)
    }

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
    database_root, node_value_op_value, validate_database_root, CommitmentState, LayerCommitment,
    VerifierCommitment, VerifierCommitmentRef, VerifierLayer,
};

pub mod anti;
pub mod argument;
pub mod bgp;
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
pub mod logup;
pub mod membership;
pub mod order;
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
pub mod commitment {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::cmp::Ordering;
    use std::collections::{BTreeMap, BTreeSet};
    use std::convert::TryInto;
    use std::io;
    use std::sync::OnceLock;

    use ark_bn254::{G1Projective, G2Projective};
    use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
    use dory_pcs::backends::arkworks::{ArkFr, ArkG1, ArkG2, ArkGT, BN254};
    #[cfg(test)]
    use dory_pcs::backends::arkworks::{ArkworksPolynomial, G1Routines};
    use dory_pcs::primitives::arithmetic::{Field as DoryField, Group, PairingCurve};
    #[cfg(test)]
    use dory_pcs::primitives::poly::Polynomial;
    use dory_pcs::setup::{ProverSetup, VerifierSetup};
    #[cfg(test)]
    use dory_pcs::Transparent;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    const COMMITMENT_MAGIC: &[u8; 8] = b"TSPCW0\0\0";
    // Bumped 4 → 5 when `xsd:date`/`xsd:dateTime` dict entries gained a NON-ZERO
    // order-preserving VALUE (a decoded i64-nanoseconds key; see `node_value_op_value`).
    // A date-containing layer's `commitment_root` therefore differs from a v4 file's, and
    // a stale v4 commitment (which committed dates at the `0` sentinel) must NOT be read as
    // if it used the new semantics — this file-format tag rejects it at decode.
    //
    // The root DOMAIN string (`commitment-root/v4`, `compute_commitment_root_from_parts`)
    // is deliberately NOT bumped: the root's STRUCTURE (which columns/commitments are
    // hashed, in what order) is unchanged — only the CONTENT of the node/value VALUE column
    // changed, and only for entries that are dates. So a layer WITHOUT dates commits the
    // identical `0`-sentinel value column and hashes to a BYTE-IDENTICAL root under v5, and
    // old v4 anchors for date-free layers stay valid.
    pub const COMMITMENT_FORMAT_VERSION: u32 = 5;
    const VERIFIER_COMMITMENT_MAGIC: &[u8; 8] = b"TSPCV0\0\0";
    // Bumped in lockstep (1 → 2): a verifier commitment derived from a v4 prover file
    // likewise committed dates at the `0` sentinel and must be rejected. Its recomputed
    // root uses the same unchanged `commitment-root/v4` domain, so a date-free verifier
    // commitment still recomputes a byte-identical root.
    pub const VERIFIER_COMMITMENT_FORMAT_VERSION: u32 = 2;

    /// Number of committed columns per layer: additions (S,P,O) then removals
    /// (S,P,O), in that fixed order.
    pub const COLUMN_COUNT: usize = 6;
    pub const MATERIALIZED_COLUMN_COUNT: usize = 3;

    /// Largest deterministic Dory domain needed by any structurally admissible
    /// PF4 table. `MAX_RECORDS = 10,000,000` pads to `2^24`; no native layer can
    /// reach a larger proof column without first violating the PF4 record cap.
    pub const SETUP_MAX_LOG_N: usize = 24;
    const LEGACY_SETUP_LOG_N: usize = 20;
    const SETUP_TIER_LOG_N: [usize; 3] = [LEGACY_SETUP_LOG_N, 22, SETUP_MAX_LOG_N];
    const MAX_SETUP_RETAINED_BYTES: usize = 8 * 1024 * 1024;

    /// Domain-separated 32-byte seed for the transparent setup. Derived by hashing
    /// a fixed public label ("nothing up my sleeve"), so it is fully reproducible.
    fn setup_seed() -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"terminus-store/proof-of-woql/dory-transparent-setup/v0");
        hasher.finalize().into()
    }

    /// Deterministically derive the transparent Dory prover setup.
    ///
    /// Determinism is load-bearing for the trustless model: every node must
    /// derive byte-identical public parameters so a commitment is reproducible.
    /// `dory_pcs::setup()` samples its generators from `OsRng` (non-deterministic),
    /// so we bypass it and seed a portable `ChaCha20Rng` from a fixed hashed label,
    /// then sample the G1/G2 generators via arkworks' `UniformRand`.
    ///
    /// In the resolved ark-ec 0.5 backend, `UniformRand<Projective>` samples a
    /// base-field x-coordinate and sign, maps them to the curve, and clears the
    /// cofactor. It does *not* expose a sampled scalar relative to the canonical
    /// generator. Thus the public seed reveals candidate coordinates, but no known
    /// discrete-log relation between accepted points. This remains a nonstandard
    /// seeded sampler rather than an RFC 9380 BN254 G1/G2 suite; ark-bn254 0.5 does
    /// not provide SWU/WB map configurations for either group. Replacing it requires
    /// external cryptographic review and a commitment-format migration, not an
    /// unaudited local G2 map.
    fn deterministic_prover_setup(max_log_n: usize) -> ProverSetup<BN254> {
        use ark_ff::UniformRand;
        let mut rng = ChaCha20Rng::from_seed(setup_seed());
        let n = 1usize << max_log_n.div_ceil(2);
        fn g1(rng: &mut impl rand::Rng) -> ArkG1 {
            loop {
                let point = G1Projective::rand(&mut *rng);
                if point != G1Projective::default() {
                    return ArkG1(point);
                }
            }
        }
        fn g2(rng: &mut impl rand::Rng) -> ArkG2 {
            loop {
                let point = G2Projective::rand(&mut *rng);
                if point != G2Projective::default() {
                    return ArkG2(point);
                }
            }
        }
        let g1_vec: Vec<ArkG1> = (0..n).map(|_| g1(&mut rng)).collect();
        let g2_vec: Vec<ArkG2> = (0..n).map(|_| g2(&mut rng)).collect();
        let h1 = g1(&mut rng);
        let h2 = g2(&mut rng);
        let ht = <BN254 as PairingCurve>::pair(&h1, &h2);
        ProverSetup {
            g1_vec,
            g2_vec,
            h1,
            h2,
            ht,
        }
    }

    fn setup_tier_for_num_vars(num_vars: usize) -> io::Result<usize> {
        SETUP_TIER_LOG_N
            .iter()
            .copied()
            .find(|tier| num_vars <= *tier)
            .ok_or_else(|| {
                invalid(format!(
                    "proof domain 2^{num_vars} exceeds PF4's maximum 2^{SETUP_MAX_LOG_N}"
                ))
            })
    }

    fn setup_generator_count(tier: usize) -> io::Result<usize> {
        1usize
            .checked_shl(tier.div_ceil(2) as u32)
            .ok_or_else(|| invalid("Dory setup generator count overflow"))
    }

    fn setup_retained_bytes(tier: usize) -> io::Result<usize> {
        setup_generator_count(tier)?
            .checked_mul(std::mem::size_of::<ArkG1>() + std::mem::size_of::<ArkG2>())
            .and_then(|bytes| {
                bytes.checked_add(
                    std::mem::size_of::<ArkG1>()
                        + std::mem::size_of::<ArkG2>()
                        + std::mem::size_of::<ArkGT>(),
                )
            })
            .ok_or_else(|| invalid("Dory setup retained-byte estimate overflow"))
    }

    fn preflight_setup_tier(tier: usize) -> io::Result<()> {
        let retained = setup_retained_bytes(tier)?;
        if retained > MAX_SETUP_RETAINED_BYTES {
            return Err(invalid(format!(
                "Dory setup tier 2^{tier} requires {retained} retained bytes, above the PF4 setup budget"
            )));
        }
        Ok(())
    }

    fn padded_domain_for_rows(rows: usize) -> io::Result<(usize, usize)> {
        if rows == 0 {
            return Ok((0, 0));
        }
        if rows > MAX_RECORDS {
            return Err(invalid(format!(
                "proof table has {rows} records, above the PF4 limit of {MAX_RECORDS}"
            )));
        }
        let padded = rows
            .checked_next_power_of_two()
            .ok_or_else(|| invalid("proof column length overflow"))?
            .max(2);
        let num_vars = padded.trailing_zeros() as usize;
        setup_tier_for_num_vars(num_vars)?;
        Ok((padded, num_vars))
    }

    /// Peak transient allocation retained by the optimized storage-side
    /// commitment builder: one padded field column plus Dory's row commitments.
    /// Columns are constructed sequentially, so only the largest table counts.
    pub(super) fn construction_commitment_scratch_bytes(row_counts: &[usize]) -> io::Result<usize> {
        row_counts.iter().try_fold(0usize, |peak, rows| {
            let (padded, num_vars) = padded_domain_for_rows(*rows)?;
            if padded == 0 {
                return Ok(peak);
            }
            let (nu, _) = matrix_dims(num_vars);
            let row_commitments = 1usize
                .checked_shl(nu as u32)
                .and_then(|count| count.checked_mul(std::mem::size_of::<ArkG1>()))
                .ok_or_else(|| invalid("Dory row-commitment scratch estimate overflow"))?;
            let bytes = padded
                .checked_mul(std::mem::size_of::<ArkFr>())
                .and_then(|value| value.checked_add(row_commitments))
                .ok_or_else(|| invalid("Dory commitment scratch estimate overflow"))?;
            Ok(peak.max(bytes))
        })
    }

    /// Stable public identity of the exact parameter tier selected for a domain.
    /// The legacy tier retains its old generator stream; larger tiers are distinct
    /// deterministic parameter sets. Query transcripts bind this identity.
    pub(crate) fn setup_parameter_id(num_vars: usize) -> io::Result<ProofHash> {
        let tier = setup_tier_for_num_vars(num_vars)?;
        let mut hasher = Sha256::new();
        hasher.update(b"terminus-store/proof-of-woql/dory-parameter-set/v0");
        hasher.update(setup_seed());
        hasher.update((tier as u64).to_be_bytes());
        hasher.update(b"matrix:sigma-ceil-nu-floor;coeff:row-major-lsb-first");
        Ok(hasher.finalize().into())
    }

    /// Lazily derive only the deterministic tier required by this domain. Keeping
    /// 2^20 as its own tier preserves all existing v4 commitments byte-for-byte.
    fn dory_prover_setup(num_vars: usize) -> io::Result<&'static ProverSetup<BN254>> {
        static SETUP_20: OnceLock<ProverSetup<BN254>> = OnceLock::new();
        static SETUP_22: OnceLock<ProverSetup<BN254>> = OnceLock::new();
        static SETUP_24: OnceLock<ProverSetup<BN254>> = OnceLock::new();
        let tier = setup_tier_for_num_vars(num_vars)?;
        preflight_setup_tier(tier)?;
        Ok(match tier {
            20 => SETUP_20.get_or_init(|| deterministic_prover_setup(20)),
            22 => SETUP_22.get_or_init(|| deterministic_prover_setup(22)),
            24 => SETUP_24.get_or_init(|| deterministic_prover_setup(24)),
            _ => unreachable!("setup tier selection returned an unknown tier"),
        })
    }

    /// The single source of truth for the row/column matrix layout of a padded
    /// column of `2^num_vars` scalars: `sigma = ceil(num_vars/2)` columns-bits,
    /// `nu = num_vars - sigma` row-bits (so `nu <= sigma`). The flat coefficient
    /// index is `row * 2^sigma + col`, folded LSB-first, so an evaluation point's
    /// first `sigma` variables are the column bits and the last `nu` are the row
    /// bits. The query argument (proof::argument) reuses this so its sumcheck
    /// point convention matches the committed columns exactly.
    pub(crate) fn matrix_dims(num_vars: usize) -> (usize, usize) {
        let sigma = num_vars.div_ceil(2);
        let nu = num_vars - sigma;
        (nu, sigma)
    }

    /// The deterministic prover parameter tier selected for this proof domain.
    pub(crate) fn prover_setup(num_vars: usize) -> io::Result<&'static ProverSetup<BN254>> {
        dory_prover_setup(num_vars)
    }

    /// The verifier setup derived once from the global prover setup.
    pub(crate) fn verifier_setup(num_vars: usize) -> io::Result<&'static VerifierSetup<BN254>> {
        static SETUP_20: OnceLock<VerifierSetup<BN254>> = OnceLock::new();
        static SETUP_22: OnceLock<VerifierSetup<BN254>> = OnceLock::new();
        static SETUP_24: OnceLock<VerifierSetup<BN254>> = OnceLock::new();
        let tier = setup_tier_for_num_vars(num_vars)?;
        let prover = dory_prover_setup(num_vars)?;
        Ok(match tier {
            20 => SETUP_20.get_or_init(|| prover.to_verifier_setup()),
            22 => SETUP_22.get_or_init(|| prover.to_verifier_setup()),
            24 => SETUP_24.get_or_init(|| prover.to_verifier_setup()),
            _ => unreachable!("setup tier selection returned an unknown tier"),
        })
    }

    /// Commit to one dictionary-id column via Dory, returning its succinct GT
    /// (tier-2) commitment. Empty columns get a canonical deterministic empty
    /// commitment (the GT identity).
    ///
    /// The column is padded to a power-of-two length `2^m`; the matrix layout is
    /// `sigma = ceil(m/2)`, `nu = m - sigma` (so `nu <= sigma`, matching Dory's
    /// non-square constraint, and `2^sigma <= 2^ceil(SETUP_MAX_LOG_N/2)` fits the
    /// global generator vectors). In Transparent mode the commitment carries no
    /// blinding, so it is a deterministic function of the ids and the setup.
    fn commit_record_id_column_with_context(
        context: &super::argument::StoreProverContext<'_>,
        records: &[ProofRecord],
        select: fn(&ProofRecord) -> u64,
    ) -> io::Result<ArkGT> {
        if records.is_empty() {
            return Ok(<ArkGT as Group>::identity());
        }
        let padded_len = records
            .len()
            .checked_next_power_of_two()
            .ok_or_else(|| invalid("proof column length overflow"))?;
        let m = padded_len.trailing_zeros() as usize;
        setup_tier_for_num_vars(m)?;
        let mut coefficients = vec![<ArkFr as DoryField>::zero(); padded_len];
        for (slot, record) in coefficients.iter_mut().zip(records) {
            *slot = <ArkFr as DoryField>::from_u64(select(record));
        }
        context.commit_column(&coefficients)
    }

    fn commit_owned_field_column_with_context(
        context: &super::argument::StoreProverContext<'_>,
        coefficients: Vec<ArkFr>,
    ) -> io::Result<ArkGT> {
        if coefficients.is_empty() {
            return Ok(<ArkGT as Group>::identity());
        }
        if !coefficients.len().is_power_of_two() {
            return Err(invalid("committed field column is not power-of-two padded"));
        }
        context.commit_column(&coefficients)
    }

    /// Build the six per-column commitments for a layer delta, in fixed order:
    /// additions (subject, predicate, object) then removals (subject, predicate,
    /// object). Records are expected pre-sorted (SPO) so the columns are canonical.
    // TODO(proof-of-woql): Phase 1c commits each column at its global absolute row
    // offset and composes commitments across the layer chain; here every layer's
    // delta is committed standalone at offset 0.
    fn commit_delta_columns(
        additions: &[ProofRecord],
        removals: &[ProofRecord],
    ) -> io::Result<[ArkGT; COLUMN_COUNT]> {
        commit_delta_columns_with_context(
            &super::argument::StoreProverContext::cpu(), additions, removals,
        )
    }

    fn commit_delta_columns_with_context(
        context: &super::argument::StoreProverContext<'_>,
        additions: &[ProofRecord],
        removals: &[ProofRecord],
    ) -> io::Result<[ArkGT; COLUMN_COUNT]> {
        Ok([
            commit_record_id_column_with_context(context, additions, |record| record.subject_id)?,
            commit_record_id_column_with_context(context, additions, |record| record.predicate_id)?,
            commit_record_id_column_with_context(context, additions, |record| record.object_id)?,
            commit_record_id_column_with_context(context, removals, |record| record.subject_id)?,
            commit_record_id_column_with_context(context, removals, |record| record.predicate_id)?,
            commit_record_id_column_with_context(context, removals, |record| record.object_id)?,
        ])
    }

    fn commit_materialized_columns(
        records: &[ProofRecord],
    ) -> io::Result<[ArkGT; MATERIALIZED_COLUMN_COUNT]> {
        commit_materialized_columns_with_context(
            &super::argument::StoreProverContext::cpu(), records,
        )
    }

    fn commit_materialized_columns_with_context(
        context: &super::argument::StoreProverContext<'_>,
        records: &[ProofRecord],
    ) -> io::Result<[ArkGT; MATERIALIZED_COLUMN_COUNT]> {
        Ok([
            commit_record_id_column_with_context(context, records, |record| record.subject_id)?,
            commit_record_id_column_with_context(context, records, |record| record.predicate_id)?,
            commit_record_id_column_with_context(context, records, |record| record.object_id)?,
        ])
    }

    /// Build the two predicate-dictionary columns from the predicate catalog
    /// (`term -> id`, canonical order): `hash[k] = hash_term(predicate_bytes_k)`,
    /// `id[k] = predicate_id_k`, padded with zeros to a power of two of length
    /// `>= 2` (the Q3 membership sumcheck needs at least one variable). An empty
    /// dictionary yields empty columns. This is the single source of truth used
    /// both to COMMIT the dictionary (here) and to OPEN it (via
    /// `LayerCommitment::predicate_dictionary_columns`), so the two align.
    fn predicate_dict_columns_from_catalog(
        predicate_catalog: &[ProofRecord],
    ) -> io::Result<(Vec<ArkFr>, Vec<ArkFr>)> {
        let d = predicate_catalog.len();
        if d == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let (padded, _) = padded_domain_for_rows(d)?;
        let zero = <ArkFr as DoryField>::zero();
        let mut hashes = vec![zero; padded];
        let mut ids = vec![zero; padded];
        for (k, record) in predicate_catalog.iter().enumerate() {
            hashes[k] = super::dict::hash_term(&record.triple.predicate);
            ids[k] = <ArkFr as DoryField>::from_u64(record.predicate_id);
        }
        Ok((hashes, ids))
    }

    /// Commit the two predicate-dictionary columns via the SAME shared query-side
    /// `commit_column` the Q3 gadget opens, so the committed columns align with
    /// what a membership proof opens. An empty dictionary gets identity commitments.
    fn commit_predicate_dictionary(
        predicate_catalog: &[ProofRecord],
    ) -> io::Result<(ArkGT, ArkGT)> {
        commit_predicate_dictionary_with_context(
            &super::argument::StoreProverContext::cpu(), predicate_catalog,
        )
    }

    fn commit_predicate_dictionary_with_context(
        context: &super::argument::StoreProverContext<'_>,
        predicate_catalog: &[ProofRecord],
    ) -> io::Result<(ArkGT, ArkGT)> {
        if predicate_catalog.is_empty() {
            let identity = <ArkGT as Group>::identity();
            return Ok((identity, identity));
        }
        let (padded, _) = padded_domain_for_rows(predicate_catalog.len())?;
        let mut hashes = vec![<ArkFr as DoryField>::zero(); padded];
        for (slot, record) in hashes.iter_mut().zip(predicate_catalog) {
            *slot = super::dict::hash_term(&record.triple.predicate);
        }
        let hash_commitment = commit_owned_field_column_with_context(context, hashes)?;
        let mut ids = vec![<ArkFr as DoryField>::zero(); padded];
        for (slot, record) in ids.iter_mut().zip(predicate_catalog) {
            *slot = <ArkFr as DoryField>::from_u64(record.predicate_id);
        }
        Ok((hash_commitment, commit_owned_field_column_with_context(context, ids)?))
    }

    /// Build the two node/value-dictionary columns from the node/value catalog
    /// (`object -> id`, canonical order): `hash[k] = hash_object(object_k)`,
    /// `id[k] = object_id_k`, padded with zeros to a power of two of length `>= 2`
    /// (the Q3 membership sumcheck needs at least one variable). Mirrors
    /// `predicate_dict_columns_from_catalog` exactly but hashes the OBJECT (a
    /// `Node` IRI or a typed `Value` literal) via the domain-tagged `hash_object`.
    /// Subjects share this id space (they are the `Node(_)` subset), so a subject
    /// constant resolves through the very same `(hash_object(Node(iri)), id)` row.
    /// Single source of truth for both COMMIT (here) and OPEN (via
    /// `LayerCommitment::node_value_dictionary_columns`), so the two align.
    fn node_value_dict_columns_from_catalog(
        node_value_catalog: &[ProofRecord],
    ) -> io::Result<(Vec<ArkFr>, Vec<ArkFr>)> {
        let padded = node_value_dict_padded_len(node_value_catalog.len())?;
        if padded == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let zero = <ArkFr as DoryField>::zero();
        let mut hashes = vec![zero; padded];
        let mut ids = vec![zero; padded];
        for (k, record) in node_value_catalog.iter().enumerate() {
            hashes[k] = super::dict::hash_object(&record.triple.object);
            ids[k] = <ArkFr as DoryField>::from_u64(record.object_id);
        }
        Ok((hashes, ids))
    }

    /// Commit the two node/value-dictionary columns via the SAME shared query-side
    /// `commit_column` the Q3 gadget opens, so the committed columns align with
    /// what a membership proof opens. An empty dictionary gets identity commitments.
    fn commit_node_value_dictionary(
        node_value_catalog: &[ProofRecord],
    ) -> io::Result<(ArkGT, ArkGT)> {
        commit_node_value_dictionary_with_context(
            &super::argument::StoreProverContext::cpu(), node_value_catalog,
        )
    }

    fn commit_node_value_dictionary_with_context(
        context: &super::argument::StoreProverContext<'_>,
        node_value_catalog: &[ProofRecord],
    ) -> io::Result<(ArkGT, ArkGT)> {
        let padded = node_value_dict_padded_len(node_value_catalog.len())?;
        if padded == 0 {
            let identity = <ArkGT as Group>::identity();
            return Ok((identity, identity));
        }
        let mut hashes = vec![<ArkFr as DoryField>::zero(); padded];
        for (slot, record) in hashes.iter_mut().zip(node_value_catalog) {
            *slot = super::dict::hash_object(&record.triple.object);
        }
        let hash_commitment = commit_owned_field_column_with_context(context, hashes)?;
        let mut ids = vec![<ArkFr as DoryField>::zero(); padded];
        for (slot, record) in ids.iter_mut().zip(node_value_catalog) {
            *slot = <ArkFr as DoryField>::from_u64(record.object_id);
        }
        Ok((hash_commitment, commit_owned_field_column_with_context(context, ids)?))
    }

    /// Reserved datatype tag for a `Node` entry in the node/value datatype column.
    /// `CanonicalDatatype` is `#[repr(u8)]` with 47 real variants (discriminants
    /// `0..=46`, `AnySimpleType = 44`, `DateTimeInterval = 46`), so `255` is safely
    /// distinct from every real discriminant. The later range gadget masks Node
    /// rows out by a datatype-inequality check against this tag.
    /// Reserved discriminator committed in the node/value datatype dictionary column.
    /// Every node has this tag; every literal uses a `CanonicalDatatype` discriminant.
    pub const NODE_DATATYPE_TAG: u64 = 255;

    /// Padded length (a power of two `>= 2`, or `0` for an empty catalog) shared by
    /// EVERY committed node/value dictionary column — `hash`, `id`, `value`,
    /// `datatype`. Single source of truth so the four columns are the exact same
    /// length and stay row-aligned (a range proof opens `value`/`datatype` at the
    /// SAME index a membership proof opens `hash`/`id`).
    fn node_value_dict_padded_len(catalog_len: usize) -> io::Result<usize> {
        if catalog_len == 0 {
            return Ok(0);
        }
        padded_domain_for_rows(catalog_len).map(|(padded, _)| padded)
    }

    /// Fixed byte width of a SUPPORTED fixed-width order-preserving numeric
    /// datatype's `to_lexical` encoding, or `None` for any other datatype
    /// (variable-width like `String`/`Decimal`/`BigInt`, or non-range types).
    ///
    /// CONFIRMED against tdb-succinct `src/tfc/datatypes.rs` (git rev ec15199,
    /// the pinned dep): every listed type's `ToLexical::to_lexical` writes exactly
    /// `width` bytes, and the raw bytes are ALREADY the order-preserving encoding,
    /// so a big-endian unsigned read of the fixed-width bytes is monotone in value:
    ///   * `Boolean` -> 1 byte `vec![0]`/`vec![1]`         (false=0 < true=1).
    ///   * `UInt8/16/32/64` -> 1/2/4/8 bytes, `write_uN::<BigEndian>` (raw BE).
    ///   * `Int8/16/32/64`  -> 1/2/4/8 bytes, offset-binary: `to_lexical` writes
    ///     `(1<<(8*width-1)) ^ (self as uN)` big-endian, so `min` maps to all-zero
    ///     and BE-reading the bytes recovers a monotone unsigned code.
    ///   * `Float32/64` -> 4/8 bytes, IEEE total-order: `to_lexical` writes
    ///     `bits ^ 0x8000..` for non-negative and `bits ^ 0xffff..` for negative,
    ///     big-endian — the standard monotone transform over finite floats.
    fn order_preserving_numeric_width(datatype: CanonicalDatatype) -> Option<usize> {
        use CanonicalDatatype::*;
        Some(match datatype {
            Boolean => 1,
            UInt8 | Int8 => 1,
            UInt16 | Int16 => 2,
            UInt32 | Int32 | Float32 => 4,
            UInt64 | Int64 | Float64 => 8,
            // `Date`/`DateTime` are order-preserving at width 8, but — unlike every
            // entry above — their native `lexical` is NOT the OP encoding (tdb-succinct
            // stores dateTime as variable-width sign+bigint-seconds+variable-nanos, and
            // date as a `year:i64,month:u8,day:u8,offset:i16` struct). Their OP value is
            // DERIVED (decode → chronological key → 8-byte offset-binary) in
            // `node_value_op_value`, which special-cases these two BEFORE the generic
            // fixed-width read below. This function reports the OP *width* (8, the range
            // gadget's `Int64` convention); it does NOT claim the lexical is 8 bytes.
            // `DateTimeStamp`/`DateTimeInterval`/`Time`/`Duration`/g* temporal types are
            // deliberately NOT here (they reject-close to the `0` sentinel): the WOQL AST
            // cannot even name a `DateTimeStamp` constant, and adding them would perturb
            // the committed OP column of any layer holding such values.
            Date | DateTime => 8,
            _ => return None,
        })
    }

    /// The shared 8-byte OFFSET-BINARY encoding of a signed-integer key `n` as an
    /// `ArkFr` — the SAME code the store's `Int64` path emits: `to_lexical` writes
    /// `(1<<63) ^ (n as u64)` big-endian, and a big-endian unsigned read recovers that
    /// value, mapping `i64::MIN → 0` and `i64::MAX → 2^64-1` monotonically. Routing the
    /// date keys through this convention lets the AUDITED range gadget (sign-bit
    /// bit-decomposition, `b_bits = 64`) compare them UNCHANGED. The datatype-tag column
    /// still carries the `Date`/`DateTime` tag, so the gadget masks date rows to a date
    /// query only — there is no cross-type comparison with the numeric `Int64` values
    /// that happen to share this code.
    fn op_offset_binary_i64(n: i64) -> ArkFr {
        <ArkFr as DoryField>::from_u64((n as u64) ^ (1u64 << 63))
    }

    /// Map a decoded UTC `NaiveDateTime` to its i64 nanoseconds-since-epoch key,
    /// SATURATING outside the representable i64-nanoseconds window (≈ 1677-09-21 ..
    /// 2262-04-11). Saturating by chronological side keeps EVERY out-of-window date
    /// correctly ordered against any in-window comparison constant `c`: a far-future date
    /// (`i64::MAX`) is `> c` and a far-past date (`i64::MIN`) is `< c`. Since a range
    /// filter only ever compares a data value to `c` (never two data values to each
    /// other), the fact that two far dates tie is harmless.
    fn datetime_to_saturating_nanos(ndt: chrono::NaiveDateTime) -> i64 {
        let utc = ndt.and_utc();
        // `timestamp()` (seconds) always fits i64 for any `NaiveDateTime`, so its sign is
        // a safe far-past/far-future discriminator when the nanos overflow i64.
        utc.timestamp_nanos_opt()
            .unwrap_or(if utc.timestamp() >= 0 { i64::MAX } else { i64::MIN })
    }

    // --- Defensive native date/dateTime OP decoding ------------------------------------
    //
    // `VerifierCommitmentRef::validate()` recomputes `node_value_op_value` over EVERY
    // committed catalog row of a self-contained, ADVERSARY-SUPPLIED proof. The native
    // tdb-succinct decoders (`storage_to_datetime`, `Date::from_lexical`, and the
    // integer/decimal helpers they call) `.unwrap()`/`.expect()`/index a `Buf`, so a
    // crafted lexical PANICS them — crashing the honest verifier (a trustless-system
    // DoS). We therefore decode the storage bytes ourselves so that ANY byte sequence
    // yields either the correct `i64` OP key (`Ok`) or a clean `io::Error` — never a
    // panic. For every well-formed, store-produced lexical the key is byte-identical to
    // the native decoder; only well-formed-but-out-of-window instants saturate (the
    // audited design) and only MALFORMED/undecodable lexicals become `Err`.
    //
    // Size-prefix bit layout of tdb-succinct's `bigint` storage (see integer.rs).
    const DT_FIRST_SIGN: u8 = 0b1000_0000;
    const DT_FIRST_CONTINUATION: u8 = 0b0100_0000;
    const DT_FIRST_MASK: u8 = 0b0011_1111;
    const DT_CONTINUATION: u8 = 0b1000_0000;
    const DT_BASE_MASK: u8 = 0b0111_1111;

    /// Consume one byte from `cursor`, or fail closed on a truncated lexical.
    fn dt_take_u8(cursor: &mut &[u8]) -> io::Result<u8> {
        match cursor.split_first() {
            Some((byte, rest)) => {
                *cursor = rest;
                Ok(*byte)
            }
            None => Err(invalid("truncated date/dateTime lexical")),
        }
    }

    /// Defensive re-implementation of tdb-succinct's `size_decode`: read the
    /// variable-length size prefix, returning `(sign_is_non_negative, magnitude_byte_count)`.
    /// Uses SATURATING arithmetic so an adversarial run of continuation bytes cannot
    /// overflow-panic (`size <<= 7` panics on `u32` overflow in a debug build); the
    /// (possibly saturated) count is only ever an upper bound, since the magnitude read
    /// below fails closed the moment the buffer runs out.
    fn dt_decode_size_prefix(cursor: &mut &[u8]) -> io::Result<(bool, u64)> {
        let mut size: u64 = 0;
        let mut sign = true;
        let mut first = true;
        loop {
            let raw = dt_take_u8(cursor)?;
            if first {
                sign = raw & DT_FIRST_SIGN != 0;
                let v = if sign { raw } else { !raw };
                let val = u64::from(v & DT_FIRST_MASK);
                if v & DT_FIRST_CONTINUATION == 0 {
                    return Ok((sign, val));
                }
                size = size.saturating_add(val);
                first = false;
            } else {
                let v = if sign { raw } else { !raw };
                let val = u64::from(v & DT_BASE_MASK);
                if v & DT_CONTINUATION == 0 {
                    return Ok((sign, size.saturating_add(val)));
                }
                size = size.saturating_add(val);
            }
            size = size.saturating_mul(128); // `<< 7`, saturating to dodge the debug overflow panic
        }
    }

    /// Defensive re-implementation of `storage_to_bigint_and_sign` specialised to the
    /// dateTime seconds field: decode the signed magnitude, returning the sign and the
    /// `i64` value, or `None` when the magnitude falls OUTSIDE the `i64` range (an
    /// out-of-window instant the caller saturates). Fails closed on a truncated magnitude.
    /// The tdb-succinct original `.expect("…surprisingly large number of seconds")`s here.
    fn dt_decode_seconds(cursor: &mut &[u8]) -> io::Result<(bool, Option<i64>)> {
        let (is_pos, size) = dt_decode_size_prefix(cursor)?;
        if size == 0 {
            return Ok((is_pos, Some(0)));
        }
        // A positive value leaves `i64` once the magnitude exceeds `i64::MAX`; a negative
        // one once it exceeds `i64::MAX + 1` (`-2^63 == i64::MIN` is representable).
        let limit: u128 = if is_pos {
            i64::MAX as u128
        } else {
            (i64::MAX as u128) + 1
        };
        let mut acc: u128 = 0;
        let mut overflow = false;
        for _ in 0..size {
            let byte = dt_take_u8(cursor)?;
            let byte = if is_pos { byte } else { !byte };
            if !overflow {
                acc = (acc << 8) | u128::from(byte);
                if acc > limit {
                    // Stop accumulating (keeping `acc` bounded) but keep consuming the
                    // remaining magnitude bytes so a truncation is still caught below.
                    overflow = true;
                }
            }
        }
        if overflow {
            return Ok((is_pos, None));
        }
        let magnitude = acc as i128;
        let value = if is_pos { magnitude } else { -magnitude };
        match i64::try_from(value) {
            Ok(seconds) => Ok((is_pos, Some(seconds))),
            // Unreachable given `limit`, but fail safe (never panic) rather than trust it.
            Err(_) => Ok((is_pos, None)),
        }
    }

    /// Decode one centary-BCD value (tdb-succinct `centary_decimal_decode`), failing
    /// closed on the `0` byte whose native `i - 1` would underflow-panic.
    fn dt_centary_decode(value: u8) -> io::Result<String> {
        if value == 0 {
            return Err(invalid("malformed dateTime fraction byte"));
        }
        let j = value - 1;
        if j % 11 == 0 {
            Ok(format!("{}", j / 11))
        } else {
            // `j >= 1` in this branch, so `j - d - 1 >= 0` (no u8 underflow).
            let d = j / 11;
            let num = j - d - 1;
            Ok(format!("{num:02}"))
        }
    }

    /// Defensive re-implementation of `decode_fraction`: read the sub-second fraction
    /// digits (bit-inverted when the instant is negative). Consumes the WHOLE fraction
    /// field — its terminal byte clears the continuation bit — so the caller can require
    /// the lexical to be fully consumed. Fails closed on truncation, a malformed digit, or
    /// a fraction longer than the 9 digits a nanosecond count can ever need (which would
    /// underflow the native `9 - fraction.len()`).
    fn dt_decode_fraction(cursor: &mut &[u8], is_pos: bool) -> io::Result<String> {
        // The fraction field is ALWAYS at least one byte (a `0x00` "false zero" when there
        // is no fraction); its absence is a truncated lexical.
        let mut raw = dt_take_u8(cursor)?;
        let first = if is_pos { raw } else { !raw };
        if first == 0x00 {
            return Ok(String::new());
        }
        let mut digits = String::new();
        loop {
            let byte = if is_pos { raw } else { !raw };
            let chunk = dt_centary_decode(byte >> 1)?;
            let single = chunk.len() == 1;
            digits.push_str(&chunk);
            if single || byte & 1 == 0 {
                break;
            }
            if digits.len() > 9 {
                return Err(invalid("dateTime fraction exceeds 9 digits"));
            }
            raw = dt_take_u8(cursor)?;
        }
        Ok(digits)
    }

    /// Convert decoded fraction digits (`<= 9`) to a nanosecond count, mirroring
    /// `storage_to_datetime`'s right-zero-pad-to-9 then parse. Fails closed on anything
    /// longer than 9 digits or non-numeric (never for a store-produced fraction).
    fn dt_fraction_to_nanos(digits: &str) -> io::Result<u32> {
        if digits.len() > 9 {
            return Err(invalid("dateTime fraction exceeds 9 digits"));
        }
        let mut padded = String::with_capacity(9);
        padded.push_str(digits);
        for _ in 0..(9 - digits.len()) {
            padded.push('0');
        }
        padded
            .parse::<u32>()
            .map_err(|_| invalid("dateTime fraction is not a valid nanosecond count"))
    }

    /// Decode a native tdb-succinct `xsd:dateTime` `lexical` to its OP i64 key. WOQL folds
    /// the timezone offset into the instant at parse time (`remove_date_time_offset`) and
    /// stores a bare UTC `NaiveDateTime`, and compares two such values chronologically to
    /// the nanosecond (`date_time/7` standard order) — exactly the epoch-nanoseconds order
    /// this key reproduces. The storage bytes are decoded DEFENSIVELY (see the block
    /// above): a bigint-seconds value outside `i64`, or an instant outside chrono's
    /// representable range, SATURATES to `i64::MIN/MAX` by sign (the audited saturating
    /// design); a truncated/over-long/otherwise-malformed lexical FAILS CLOSED with an
    /// `io::Error` rather than panicking the verifier. For every store-produced lexical the
    /// key is byte-identical to `storage_to_datetime` + `datetime_to_saturating_nanos`.
    fn datetime_lexical_to_op_i64(lexical: &[u8]) -> io::Result<i64> {
        use chrono::{DateTime, Utc};
        let mut cursor: &[u8] = lexical;
        let (is_pos, seconds) = dt_decode_seconds(&mut cursor)?;
        let fraction = dt_decode_fraction(&mut cursor, is_pos)?;
        // A well-formed dateTime lexical is exactly the seconds prefix + fraction field;
        // any trailing bytes are malformed.
        if !cursor.is_empty() {
            return Err(invalid("dateTime lexical has trailing bytes"));
        }
        let seconds = match seconds {
            Some(seconds) => seconds,
            // Bigint seconds beyond i64 ⇒ far outside the i64-nanoseconds window; saturate.
            None => return Ok(if is_pos { i64::MAX } else { i64::MIN }),
        };
        let nanos = if fraction.is_empty() {
            0
        } else {
            dt_fraction_to_nanos(&fraction)?
        };
        // Reconstruct the SAME instant `storage_to_datetime` would (checked constructors),
        // then reuse `datetime_to_saturating_nanos` for a byte-identical key. A negative
        // instant with a fraction is stored as `(seconds - 1, 1e9 - nanos)`.
        let instant = if fraction.is_empty() || is_pos {
            DateTime::<Utc>::from_timestamp(seconds, nanos)
        } else {
            seconds
                .checked_sub(1)
                .and_then(|s| DateTime::<Utc>::from_timestamp(s, 1_000_000_000 - nanos))
        };
        match instant {
            Some(instant) => Ok(datetime_to_saturating_nanos(instant.naive_utc())),
            // Fits i64 seconds but beyond chrono's representable range ⇒ far past/future,
            // outside the i64-nanoseconds window; saturate by sign.
            None => Ok(if is_pos { i64::MAX } else { i64::MIN }),
        }
    }

    /// Decode a native tdb-succinct `xsd:date` `lexical` to its OP i64 key. WOQL compares a
    /// stored `xsd:date` as the Prolog compound `date(Year,Month,Day,Offset)` in standard
    /// order — i.e. `(year, month, day, offset)`-LEXICAL, with the timezone offset as a
    /// LEAST-significant tiebreaker that is never applied to shift the day (a larger offset
    /// sorts later; `"2020-01-01"` and `"2020-01-01Z"` both have offset 0 and compare
    /// equal). We reproduce THAT order — not a pure instant — with a key strictly monotone
    /// in `(year, month, day, offset)`:
    ///
    /// ```text
    ///   key = midnight_utc_nanos(year, month, day)  +  offset
    /// ```
    ///
    /// Consecutive calendar days differ by `86_400 * 1e9 ≈ 8.64e13` nanoseconds, which
    /// dwarfs `|offset|` (a tdb-succinct `Date.offset` is an `i16`, ≤ 32767 in magnitude),
    /// so the offset can only ever break ties among equal `(Y,M,D)` — exactly like
    /// `compare/3`. A comparison constant always has offset 0 (the compiler builds it so),
    /// so the tiebreaker is only exercised by DATA rows, keeping the constant's own key a
    /// pure midnight instant.
    fn date_lexical_to_op_i64(lexical: &[u8]) -> io::Result<i64> {
        use chrono::NaiveDate;
        // A store `xsd:date` lexical is EXACTLY the 12-byte `Date` encoding —
        //   year:i64 (offset-binary BE, 8) | month:u8 (1) | day:u8 (1) | offset:i16 (offset-binary BE, 2).
        // Decode the fixed fields directly instead of `Date::from_lexical`, which reads a
        // `Buf` and PANICS on a short buffer; a wrong length is a truncated or over-long
        // (i.e. malformed) lexical ⇒ fail closed. These fixed-width reads reproduce
        // `Date::from_lexical` byte-for-byte for every valid 12-byte lexical (the same
        // offset-binary transform the store's `i64`/`i16` `to_lexical` applies).
        let [y0, y1, y2, y3, y4, y5, y6, y7, month, day, o0, o1]: [u8; 12] = lexical
            .try_into()
            .map_err(|_| invalid("xsd:date lexical is not exactly 12 bytes"))?;
        let year = (u64::from_be_bytes([y0, y1, y2, y3, y4, y5, y6, y7]) ^ (1u64 << 63)) as i64;
        let offset = (u16::from_be_bytes([o0, o1]) ^ (1u16 << 15)) as i16;
        let base = match i32::try_from(year) {
            Ok(year) => match NaiveDate::from_ymd_opt(year, u32::from(month), u32::from(day)) {
                // Midnight is always a valid time-of-day; still constructed via the checked
                // `and_hms_opt` (never `.expect`) so no input can panic the verifier.
                Some(date) => datetime_to_saturating_nanos(
                    date.and_hms_opt(0, 0, 0)
                        .ok_or_else(|| invalid("xsd:date midnight is not constructible"))?,
                ),
                // A well-formed store `Date` always has a valid calendar day; an
                // out-of-range month/day here means the committed lexical is corrupt —
                // fail closed rather than guess a key.
                None => return Err(invalid("xsd:date lexical has an out-of-range month or day")),
            },
            // A year beyond `i32` is astronomically far past/future and unambiguously
            // outside the i64-nanoseconds window; saturate by its sign (year ≥ 0 ⇒ far
            // future). Here the sign of the YEAR is a sound discriminator precisely because
            // such a year is enormous.
            Err(_) => return Ok(if year >= 0 { i64::MAX } else { i64::MIN }),
        };
        // Apply the offset tiebreaker ONLY to an in-window base. A saturated boundary key
        // (a far date, or the exact representable min/max) stays pinned so the offset can
        // never nudge a far date back across an in-window constant. Real in-window dates
        // sit at midnight, ≥ ~8.5e13 ns from either extreme, so `saturating_add` here never
        // actually saturates for them — it is a guard for the boundary case only.
        Ok(match base {
            i64::MIN | i64::MAX => base,
            _ => base.saturating_add(i64::from(offset)),
        })
    }

    /// The order-preserving unsigned-integer encoding of a catalog entry's object
    /// value as an `ArkFr`, or the `0` sentinel for entries that are never
    /// range-queried (`Node`s, and `String`/variable-width or otherwise
    /// unsupported datatypes). For a supported fixed-width numeric type the store's
    /// `to_lexical` bytes ARE the OP encoding (see `order_preserving_numeric_width`),
    /// so we big-endian read the fixed-width bytes into `[0, 2^{8*width})`. Widths
    /// are `<= 8`, so the value always fits a `u64`. A supported type whose lexical
    /// width does NOT match the expected fixed width is treated as corruption and
    /// fails closed (this cannot happen for store-produced values).
    ///
    /// `Date`/`DateTime` are the sole EXCEPTION: they are order-preserving but their
    /// native lexical is NOT the OP encoding, so they are special-cased below —
    /// decoded to a chronological i64 key and re-encoded via [`op_offset_binary_i64`].
    /// This is `pub` so the woql2 compiler can compute a range constant's OP value
    /// through this EXACT function (single source of truth: the compiler's `c` is then
    /// byte-identical to the committed dict value column by construction).
    pub fn node_value_op_value(object: &CanonicalObject) -> io::Result<ArkFr> {
        let (datatype, lexical) = match object {
            CanonicalObject::Value { datatype, lexical } => (*datatype, lexical),
            CanonicalObject::Node(_) => return Ok(<ArkFr as DoryField>::zero()),
        };
        // Date/DateTime: decode the native storage bytes to a chronological i64 key and
        // re-encode as the shared offset-binary code. Handled BEFORE the generic
        // fixed-width read because their `lexical` is variable-width (so the width check
        // below would reject it), even though `order_preserving_numeric_width` reports 8.
        match datatype {
            CanonicalDatatype::DateTime => {
                return Ok(op_offset_binary_i64(datetime_lexical_to_op_i64(lexical)?));
            }
            CanonicalDatatype::Date => {
                return Ok(op_offset_binary_i64(date_lexical_to_op_i64(lexical)?));
            }
            _ => {}
        }
        let width = match order_preserving_numeric_width(datatype) {
            Some(width) => width,
            None => return Ok(<ArkFr as DoryField>::zero()),
        };
        if lexical.len() != width {
            return Err(invalid(
                "order-preserving numeric literal has unexpected lexical width",
            ));
        }
        let mut buf = [0u8; 8];
        buf[8 - width..].copy_from_slice(lexical);
        Ok(<ArkFr as DoryField>::from_u64(u64::from_be_bytes(buf)))
    }

    /// Reject IEEE NaNs before using the Store's total-order lexical encoding as
    /// WOQL numeric comparison order. Prolog's `X < Y` follows IEEE unordered-NaN
    /// semantics, whereas the storage encoding deliberately assigns every NaN a
    /// total-order position. A range/group proof over a catalog containing a NaN
    /// of the selected float datatype would therefore prove different semantics.
    pub(crate) fn reject_nan_catalog(
        catalog: &[ProofRecord],
        datatype_tag: ArkFr,
    ) -> io::Result<()> {
        let f32_tag = ArkFr::from_u64(CanonicalDatatype::Float32 as u64);
        let f64_tag = ArkFr::from_u64(CanonicalDatatype::Float64 as u64);
        if datatype_tag != f32_tag && datatype_tag != f64_tag {
            return Ok(());
        }
        for record in catalog {
            let is_nan = match &record.triple.object {
                CanonicalObject::Value {
                    datatype: CanonicalDatatype::Float32,
                    lexical,
                } if datatype_tag == f32_tag && lexical.len() == 4 => {
                    let encoded = u32::from_be_bytes(lexical.as_slice().try_into().unwrap());
                    let bits = if encoded & 0x8000_0000 != 0 {
                        encoded ^ 0x8000_0000
                    } else {
                        encoded ^ 0xffff_ffff
                    };
                    f32::from_bits(bits).is_nan()
                }
                CanonicalObject::Value {
                    datatype: CanonicalDatatype::Float64,
                    lexical,
                } if datatype_tag == f64_tag && lexical.len() == 8 => {
                    let encoded = u64::from_be_bytes(lexical.as_slice().try_into().unwrap());
                    let bits = if encoded & 0x8000_0000_0000_0000 != 0 {
                        encoded ^ 0x8000_0000_0000_0000
                    } else {
                        encoded ^ 0xffff_ffff_ffff_ffff
                    };
                    f64::from_bits(bits).is_nan()
                }
                _ => false,
            };
            if is_nan {
                return Err(invalid(
                    "float range/group proof cannot match Prolog semantics in a catalog containing NaN",
                ));
            }
        }
        Ok(())
    }

    /// Build the node/value-dictionary VALUE column (one entry per catalog row, in
    /// the SAME canonical order/padding as `node_value_dict_columns_from_catalog`):
    /// `value[k] = node_value_op_value(object_k)` — the order-preserving numeric
    /// encoding for supported fixed-width numeric literals, `0` otherwise. Padding
    /// rows carry `0` (as `hash`/`id` do; they have `id = 0` so any real-row check
    /// excludes them). Single source of truth for both COMMIT and OPEN.
    fn node_value_dict_value_column(node_value_catalog: &[ProofRecord]) -> io::Result<Vec<ArkFr>> {
        let padded = node_value_dict_padded_len(node_value_catalog.len())?;
        let mut values = vec![<ArkFr as DoryField>::zero(); padded];
        for (k, record) in node_value_catalog.iter().enumerate() {
            values[k] = node_value_op_value(&record.triple.object)?;
        }
        Ok(values)
    }

    /// Build the node/value-dictionary DATATYPE column (one entry per catalog row,
    /// same order/padding): `datatype[k]` is the `CanonicalDatatype` discriminant
    /// for a `Value` object, or the reserved `NODE_DATATYPE_TAG` for a `Node`. Lets
    /// the later range gadget mask rows to the queried datatype by equality. Padding
    /// rows carry `0` (mirrors `hash`/`id`); tag `0` is `String`, which is not a
    /// range-supported type, and padding rows have `id = 0`, so padding never
    /// false-matches a numeric range query. Single source of truth for COMMIT/OPEN.
    fn node_value_dict_datatype_column(
        node_value_catalog: &[ProofRecord],
    ) -> io::Result<Vec<ArkFr>> {
        let padded = node_value_dict_padded_len(node_value_catalog.len())?;
        let mut datatypes = vec![<ArkFr as DoryField>::zero(); padded];
        for (k, record) in node_value_catalog.iter().enumerate() {
            let tag = match &record.triple.object {
                CanonicalObject::Value { datatype, .. } => *datatype as u8 as u64,
                CanonicalObject::Node(_) => NODE_DATATYPE_TAG,
            };
            datatypes[k] = <ArkFr as DoryField>::from_u64(tag);
        }
        Ok(datatypes)
    }

    /// Commit the node/value VALUE and DATATYPE columns via the SAME shared
    /// query-side `commit_column` the range/membership gadgets open, so the
    /// committed columns align with what a proof opens. An empty dictionary gets
    /// identity commitments (mirrors `commit_node_value_dictionary`).
    fn commit_node_value_dict_range(
        node_value_catalog: &[ProofRecord],
    ) -> io::Result<(ArkGT, ArkGT)> {
        commit_node_value_dict_range_with_context(
            &super::argument::StoreProverContext::cpu(), node_value_catalog,
        )
    }

    fn commit_node_value_dict_range_with_context(
        context: &super::argument::StoreProverContext<'_>,
        node_value_catalog: &[ProofRecord],
    ) -> io::Result<(ArkGT, ArkGT)> {
        let values = node_value_dict_value_column(node_value_catalog)?;
        if values.is_empty() {
            let identity = <ArkGT as Group>::identity();
            return Ok((identity, identity));
        }
        let value_commitment = commit_owned_field_column_with_context(context, values)?;
        let datatypes = node_value_dict_datatype_column(node_value_catalog)?;
        Ok((
            value_commitment,
            commit_owned_field_column_with_context(context, datatypes)?,
        ))
    }

    fn serialize_commitment(commitment: &ArkGT) -> io::Result<Vec<u8>> {
        let mut bytes = Vec::new();
        commitment
            .serialize_compressed(&mut bytes)
            .map_err(|error| invalid(format!("commitment serialization failed: {error}")))?;
        Ok(bytes)
    }

    fn deserialize_commitment(bytes: &[u8]) -> io::Result<ArkGT> {
        ArkGT::deserialize_compressed(bytes)
            .map_err(|error| invalid(format!("commitment deserialization failed: {error}")))
    }

    /// Small authenticated summary of a layer's committed delta. Mirrors the
    /// fields the old Merkle `ProofLayerState` exposed, minus the per-index
    /// Merkle roots, plus a single `commitment_root`.
    ///
    /// `PartialEq` is load-bearing: the storage plumbing compares a rebuilt
    /// state against the stored state in `validate_proof_chain_against_store`.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct CommitmentState {
        pub parent_state_root: Option<ProofHash>,
        pub addition_count: u64,
        pub removal_count: u64,
        pub materialized_count: u64,
        pub parent_node_value_count: u64,
        pub parent_predicate_count: u64,
        pub node_value_count: u64,
        pub predicate_count: u64,
        pub commitment_root: ProofHash,
    }

    impl CommitmentState {
        pub fn validate(&self) -> io::Result<()> {
            if self.parent_node_value_count > self.node_value_count
                || self.parent_predicate_count > self.predicate_count
                || (self.parent_state_root.is_none()
                    && (self.parent_node_value_count != 0 || self.parent_predicate_count != 0))
            {
                return Err(invalid("invalid dictionary parent counts"));
            }
            Ok(())
        }
    }

    /// Validated per-layer sidecar payload. Storage treats the encoded form as
    /// opaque bytes.
    #[derive(Clone, Debug)]
    pub struct LayerCommitment {
        pub state: CommitmentState,
        // The dictionary-encoded triple columns, kept sorted (SPO for the delta,
        // dictionary order for catalogs) so commitments are deterministic.
        pub additions: Vec<ProofRecord>,
        pub removals: Vec<ProofRecord>,
        /// Complete logical triple set visible at this immutable layer head.
        pub materialized: Vec<ProofRecord>,
        pub predicate_catalog: Vec<ProofRecord>,
        pub node_value_catalog: Vec<ProofRecord>,
        /// Per-column Dory (tier-2 GT) commitments over the id columns, in fixed
        /// order: additions (S,P,O) then removals (S,P,O). `commitment_root`
        /// binds these. Phase 1b commits each column standalone at offset 0.
        pub column_commitments: [ArkGT; COLUMN_COUNT],
        /// Dory commitments over the complete logical S/P/O columns.
        pub materialized_column_commitments: [ArkGT; MATERIALIZED_COLUMN_COUNT],
        /// Dory commitments of the layer's PREDICATE dictionary as two columns
        /// `(Com(pred_dict_hash), Com(pred_dict_id))`, where `pred_dict_hash[k] =
        /// hash_term(predicate_bytes_k)` and `pred_dict_id[k] = predicate_id_k`
        /// over `predicate_catalog`'s canonical order (padded to a power of two).
        /// `commitment_root` binds these, so a Q3 dictionary-membership proof of an
        /// IRI->id binding is authenticated by the root.
        pub predicate_dict_commitments: (ArkGT, ArkGT),
        /// Dory commitments of the layer's NODE/VALUE dictionary as two columns
        /// `(Com(nv_dict_hash), Com(nv_dict_id))`, where `nv_dict_hash[k] =
        /// hash_object(object_k)` and `nv_dict_id[k] = object_id_k` over
        /// `node_value_catalog`'s canonical order (padded to a power of two).
        /// `commitment_root` binds these, so a Q3 dictionary-membership proof of a
        /// subject/object CONSTANT->id binding is authenticated by the root.
        pub node_value_dict_commitments: (ArkGT, ArkGT),
        /// Dory commitments of TWO extra node/value-dictionary columns that back
        /// the (later) numeric RANGE gadget, over `node_value_catalog`'s canonical
        /// order (same padding/length as `node_value_dict_commitments`):
        ///   `.0` = `Com(nv_dict_value)`: `nv_dict_value[k]` is the ORDER-PRESERVING
        ///          unsigned-integer encoding of object_k's value when object_k is a
        ///          supported fixed-width numeric literal, else the `0` sentinel
        ///          (`Node`s, `String`/variable-width or non-OP datatypes). See
        ///          `node_value_dict_value_column`.
        ///   `.1` = `Com(nv_dict_datatype)`: `nv_dict_datatype[k]` is object_k's
        ///          `CanonicalDatatype` discriminant (a `Value`) or the reserved
        ///          `NODE_DATATYPE_TAG` (a `Node`); lets the later gadget mask rows
        ///          to the queried datatype by equality. See
        ///          `node_value_dict_datatype_column`.
        /// `commitment_root` binds these two, so the value column a range proof
        /// compares against — and the datatype it is masked by — are authenticated
        /// by the root. The `object_id` column is NOT value-ordered on rollup
        /// layers, so the range gadget ranges over this committed OP encoding and
        /// ties each triple's object to its dict row by a per-row membership lookup.
        pub node_value_dict_range_commitments: (ArkGT, ArkGT),
    }

    /// Compact, verifier-only projection of a validated PF4 v4 layer commitment.
    ///
    /// It deliberately omits the native addition/removal tables and the complete
    /// materialized triple table. Those are prover witnesses. The retained state,
    /// column commitments and complete clear dictionary catalogs are exactly the
    /// public inputs needed to recompute the trusted layer root, authenticate
    /// constants and render typed result ids. This type exposes no conversion back
    /// to [`LayerCommitment`], because the omitted witness tables cannot be
    /// reconstructed from commitments.
    #[derive(Clone, Debug)]
    pub struct VerifierCommitment {
        pub state: CommitmentState,
        pub predicate_catalog: Vec<ProofRecord>,
        pub node_value_catalog: Vec<ProofRecord>,
        pub column_commitments: [ArkGT; COLUMN_COUNT],
        pub materialized_column_commitments: [ArkGT; MATERIALIZED_COLUMN_COUNT],
        pub predicate_dict_commitments: (ArkGT, ArkGT),
        pub node_value_dict_commitments: (ArkGT, ArkGT),
        pub node_value_dict_range_commitments: (ArkGT, ArkGT),
    }

    /// Borrowed verifier view shared by full PF4 sidecars and compact artifacts.
    #[derive(Clone, Copy)]
    pub struct VerifierCommitmentRef<'a> {
        pub state: &'a CommitmentState,
        pub predicate_catalog: &'a [ProofRecord],
        pub node_value_catalog: &'a [ProofRecord],
        pub column_commitments: &'a [ArkGT; COLUMN_COUNT],
        pub materialized_column_commitments: &'a [ArkGT; MATERIALIZED_COLUMN_COUNT],
        pub predicate_dict_commitments: &'a (ArkGT, ArkGT),
        pub node_value_dict_commitments: &'a (ArkGT, ArkGT),
        pub node_value_dict_range_commitments: &'a (ArkGT, ArkGT),
    }

    /// Common verifier-facing interface. It intentionally contains no raw-table
    /// accessors, keeping prover witness data outside verification APIs.
    pub trait VerifierLayer {
        fn verifier_view(&self) -> VerifierCommitmentRef<'_>;
    }

    impl VerifierLayer for LayerCommitment {
        fn verifier_view(&self) -> VerifierCommitmentRef<'_> {
            VerifierCommitmentRef {
                state: &self.state,
                predicate_catalog: &self.predicate_catalog,
                node_value_catalog: &self.node_value_catalog,
                column_commitments: &self.column_commitments,
                materialized_column_commitments: &self.materialized_column_commitments,
                predicate_dict_commitments: &self.predicate_dict_commitments,
                node_value_dict_commitments: &self.node_value_dict_commitments,
                node_value_dict_range_commitments: &self.node_value_dict_range_commitments,
            }
        }
    }

    impl VerifierLayer for VerifierCommitment {
        fn verifier_view(&self) -> VerifierCommitmentRef<'_> {
            VerifierCommitmentRef {
                state: &self.state,
                predicate_catalog: &self.predicate_catalog,
                node_value_catalog: &self.node_value_catalog,
                column_commitments: &self.column_commitments,
                materialized_column_commitments: &self.materialized_column_commitments,
                predicate_dict_commitments: &self.predicate_dict_commitments,
                node_value_dict_commitments: &self.node_value_dict_commitments,
                node_value_dict_range_commitments: &self.node_value_dict_range_commitments,
            }
        }
    }

    impl VerifierLayer for VerifierCommitmentRef<'_> {
        fn verifier_view(&self) -> VerifierCommitmentRef<'_> {
            *self
        }
    }

    impl<T: VerifierLayer + ?Sized> VerifierLayer for std::sync::Arc<T> {
        fn verifier_view(&self) -> VerifierCommitmentRef<'_> {
            (**self).verifier_view()
        }
    }

    impl VerifierCommitmentRef<'_> {
        pub fn validate(self) -> io::Result<()> {
            self.state.validate()?;
            let subject_catalog = subject_catalog_from_objects(self.node_value_catalog)?;
            validate_complete_catalog(
                &[],
                &[],
                &[],
                [
                    &subject_catalog,
                    self.predicate_catalog,
                    self.node_value_catalog,
                ],
                self.state.parent_node_value_count,
                self.state.parent_predicate_count,
                self.state.node_value_count,
                self.state.predicate_count,
            )?;
            if commit_predicate_dictionary(self.predicate_catalog)?
                != *self.predicate_dict_commitments
                || commit_node_value_dictionary(self.node_value_catalog)?
                    != *self.node_value_dict_commitments
                || commit_node_value_dict_range(self.node_value_catalog)?
                    != *self.node_value_dict_range_commitments
            {
                return Err(invalid("verifier catalog commitment mismatch"));
            }
            if self.recomputed_commitment_root()? != self.state.commitment_root {
                return Err(invalid("verifier commitment root mismatch"));
            }
            Ok(())
        }

        pub fn recomputed_commitment_root(self) -> io::Result<ProofHash> {
            compute_commitment_root_from_parts(
                self.state,
                self.column_commitments,
                self.materialized_column_commitments,
                self.predicate_dict_commitments,
                self.node_value_dict_commitments,
                self.node_value_dict_range_commitments,
            )
        }

        pub fn node_value_dictionary_columns(self) -> io::Result<(Vec<ArkFr>, Vec<ArkFr>)> {
            node_value_dict_columns_from_catalog(self.node_value_catalog)
        }
    }

    impl PartialEq for LayerCommitment {
        fn eq(&self, other: &Self) -> bool {
            // A validated commitment's state summarizes every column table; two
            // decodings that reconstruct the same authenticated state compare
            // equal regardless of any incidental table-layout choices.
            self.state == other.state
        }
    }

    impl Eq for LayerCommitment {}

    impl LayerCommitment {
        /// Deterministically project this fully validated PF4 v4 sidecar into the
        /// bounded public artifact needed by a remote verifier.
        pub fn verifier_commitment(&self) -> io::Result<VerifierCommitment> {
            self.validate()?;
            let artifact = VerifierCommitment {
                state: self.state.clone(),
                predicate_catalog: self.predicate_catalog.clone(),
                node_value_catalog: self.node_value_catalog.clone(),
                column_commitments: self.column_commitments,
                materialized_column_commitments: self.materialized_column_commitments,
                predicate_dict_commitments: self.predicate_dict_commitments,
                node_value_dict_commitments: self.node_value_dict_commitments,
                node_value_dict_range_commitments: self.node_value_dict_range_commitments,
            };
            artifact.validate()?;
            Ok(artifact)
        }

        /// Build a production sidecar from the complete native dictionary delta.
        /// Replaces `ProofLayer::new_native_with_catalog`. The supplied mappings
        /// contain only ids introduced by this layer; inherited mappings are
        /// authenticated by the parent commitment state.
        #[allow(clippy::too_many_arguments)]
        pub fn from_native_delta(
            parent_state_root: Option<ProofHash>,
            additions: Vec<ProofRecord>,
            removals: Vec<ProofRecord>,
            materialized: Vec<ProofRecord>,
            parent_node_value_count: u64,
            parent_predicate_count: u64,
            node_value_count: u64,
            predicate_count: u64,
            node_value_mappings: Vec<(CanonicalObject, u64)>,
            predicate_mappings: Vec<(Vec<u8>, u64)>,
        ) -> io::Result<Self> {
            Self::from_native_delta_with_context(
                &super::argument::StoreProverContext::cpu(),
                parent_state_root,
                additions,
                removals,
                materialized,
                parent_node_value_count,
                parent_predicate_count,
                node_value_count,
                predicate_count,
                node_value_mappings,
                predicate_mappings,
            )
        }

        /// Build a production sidecar with explicitly selected prover arithmetic.
        /// The context affects computation only; validation and wire format remain
        /// identical to [`Self::from_native_delta`].
        #[allow(clippy::too_many_arguments)]
        pub fn from_native_delta_with_context(
            context: &super::argument::StoreProverContext<'_>,
            parent_state_root: Option<ProofHash>,
            mut additions: Vec<ProofRecord>,
            mut removals: Vec<ProofRecord>,
            mut materialized: Vec<ProofRecord>,
            parent_node_value_count: u64,
            parent_predicate_count: u64,
            node_value_count: u64,
            predicate_count: u64,
            node_value_mappings: Vec<(CanonicalObject, u64)>,
            predicate_mappings: Vec<(Vec<u8>, u64)>,
        ) -> io::Result<Self> {
            // COMMITMENT DIET (Wave 0): a DELTA layer (parent_state_root == Some) commits
            // only its add/remove columns + the INCREMENTAL dictionary it introduced. It
            // must NOT carry the O(state) materialized head view; callers (rebuild) pass an
            // empty table and the incremental (`parent_count+1..=count`) dictionary mappings.
            if parent_state_root.is_some() && !materialized.is_empty() {
                return Err(invalid(
                    "delta layer commitment must not carry a materialized head view",
                ));
            }
            validate_construction_heap_budget(
                &additions,
                &removals,
                &materialized,
                &node_value_mappings,
                &predicate_mappings,
            )?;
            if additions.len() > MAX_RECORDS
                || removals.len() > MAX_RECORDS
                || materialized.len() > MAX_RECORDS
            {
                return Err(invalid("proof delta exceeds storage limit"));
            }
            let [subject_catalog, predicate_catalog, node_value_catalog] =
                build_dictionary_catalogs_from_maps(node_value_mappings, predicate_mappings)?;

            let addition_set: BTreeSet<_> = additions.iter().map(|r| &r.triple).collect();
            let removal_set: BTreeSet<_> = removals.iter().map(|r| &r.triple).collect();
            if addition_set.len() != additions.len()
                || removal_set.len() != removals.len()
                || !addition_set.is_disjoint(&removal_set)
            {
                return Err(invalid("duplicate or conflicting native delta records"));
            }
            if additions.iter().chain(removals.iter()).any(|record| {
                record.subject_id == 0
                    || record.predicate_id == 0
                    || record.object_id == 0
                    || record.subject_id > node_value_count
                    || record.object_id > node_value_count
                    || record.predicate_id > predicate_count
            }) {
                return Err(invalid(
                    "native delta record id lies outside dictionary counts",
                ));
            }
            validate_complete_catalog(
                &additions,
                &removals,
                &materialized,
                [
                    &subject_catalog,
                    &predicate_catalog,
                    &node_value_catalog,
                ],
                parent_node_value_count,
                parent_predicate_count,
                node_value_count,
                predicate_count,
            )?;

            additions.sort_by(|a, b| triple_cmp(a, b, ProofIndex::Spo));
            removals.sort_by(|a, b| triple_cmp(a, b, ProofIndex::Spo));
            materialized.sort_by(|a, b| triple_cmp(a, b, ProofIndex::Spo));

            // Real per-column Dory commitments over the sorted id columns.
            let column_commitments =
                commit_delta_columns_with_context(context, &additions, &removals)?;
            let materialized_column_commitments =
                commit_materialized_columns_with_context(context, &materialized)?;
            // Predicate-dictionary commitments (IRI->id binding, authenticated by
            // the root so Q3 membership proofs can trust it).
            let predicate_dict_commitments =
                commit_predicate_dictionary_with_context(context, &predicate_catalog)?;
            // Node/value-dictionary commitments (subject/object CONSTANT->id
            // binding, authenticated by the root the same way).
            let node_value_dict_commitments =
                commit_node_value_dictionary_with_context(context, &node_value_catalog)?;
            // Node/value order-preserving VALUE + DATATYPE columns backing the
            // (later) numeric range gadget, authenticated by the root the same way.
            let node_value_dict_range_commitments =
                commit_node_value_dict_range_with_context(context, &node_value_catalog)?;

            let state = CommitmentState {
                parent_state_root,
                addition_count: additions.len() as u64,
                removal_count: removals.len() as u64,
                materialized_count: materialized.len() as u64,
                parent_node_value_count,
                parent_predicate_count,
                node_value_count,
                predicate_count,
                commitment_root: [0; 32],
            };
            let mut layer = Self {
                state,
                additions,
                removals,
                materialized,
                predicate_catalog,
                node_value_catalog,
                column_commitments,
                materialized_column_commitments,
                predicate_dict_commitments,
                node_value_dict_commitments,
                node_value_dict_range_commitments,
            };
            layer.state.commitment_root = layer.compute_commitment_root()?;
            layer.validate()?;
            Ok(layer)
        }

        /// Conservative retained-byte estimate for the decoded proof cache. It
        /// charges the column-table capacities plus a fixed allocator allowance
        /// per allocation. It is an accounting weight, not an RSS measurement.
        pub fn estimated_cache_bytes(&self) -> usize {
            const ALLOCATION_OVERHEAD: usize = 32;
            let table_bytes = |records: &[ProofRecord]| {
                records.iter().fold(
                    std::mem::size_of::<Vec<ProofRecord>>()
                        .saturating_add(
                            records
                                .len()
                                .saturating_mul(std::mem::size_of::<ProofRecord>()),
                        )
                        .saturating_add(ALLOCATION_OVERHEAD),
                    |total, record| {
                        let object = match &record.triple.object {
                            CanonicalObject::Node(value) => value.capacity(),
                            CanonicalObject::Value { lexical, .. } => lexical.capacity(),
                        };
                        total
                            .saturating_add(record.triple.subject.capacity())
                            .saturating_add(record.triple.predicate.capacity())
                            .saturating_add(object)
                            .saturating_add(3usize.saturating_mul(ALLOCATION_OVERHEAD))
                    },
                )
            };
            std::mem::size_of::<LayerCommitment>()
                .saturating_add(table_bytes(&self.additions))
                .saturating_add(table_bytes(&self.removals))
                .saturating_add(table_bytes(&self.materialized))
                .saturating_add(table_bytes(&self.predicate_catalog))
                .saturating_add(table_bytes(&self.node_value_catalog))
        }

        pub fn validate(&self) -> io::Result<()> {
            self.state.validate()?;
            // COMMITMENT DIET (Wave 0): a delta layer's materialized columns are NOT bound
            // by its root, so they MUST be empty — otherwise a decoded sidecar could smuggle
            // in unbound (untrusted) head rows. A base layer keeps its materialized view.
            if self.state.parent_state_root.is_some()
                && (!self.materialized.is_empty() || self.state.materialized_count != 0)
            {
                return Err(invalid(
                    "delta layer commitment must not carry a materialized head view",
                ));
            }
            if self.additions.len() > MAX_RECORDS
                || self.removals.len() > MAX_RECORDS
                || self.materialized.len() > MAX_RECORDS
            {
                return Err(invalid("proof delta exceeds storage limit"));
            }
            if self.additions.len() as u64 != self.state.addition_count
                || self.removals.len() as u64 != self.state.removal_count
                || self.materialized.len() as u64 != self.state.materialized_count
            {
                return Err(invalid("commitment delta count mismatch"));
            }
            for table in [&self.additions, &self.removals, &self.materialized] {
                if table
                    .windows(2)
                    .any(|pair| triple_cmp(&pair[0], &pair[1], ProofIndex::Spo) != Ordering::Less)
                {
                    return Err(invalid("commitment delta records not strictly ordered"));
                }
                if table.iter().any(|record| {
                    record.subject_id == 0
                        || record.predicate_id == 0
                        || record.object_id == 0
                        || record.subject_id > self.state.node_value_count
                        || record.object_id > self.state.node_value_count
                        || record.predicate_id > self.state.predicate_count
                }) {
                    return Err(invalid(
                        "commitment delta record id lies outside dictionary counts",
                    ));
                }
            }
            let addition_set: BTreeSet<_> = self.additions.iter().map(|r| &r.triple).collect();
            let removal_set: BTreeSet<_> = self.removals.iter().map(|r| &r.triple).collect();
            if addition_set.len() != self.additions.len()
                || removal_set.len() != self.removals.len()
                || !addition_set.is_disjoint(&removal_set)
            {
                return Err(invalid("duplicate or conflicting native delta records"));
            }
            let materialized_set: BTreeSet<_> = self
                .materialized
                .iter()
                .map(|record| &record.triple)
                .collect();
            if materialized_set.len() != self.materialized.len() {
                return Err(invalid("duplicate materialized logical records"));
            }
            let subject_catalog = subject_catalog_from_objects(&self.node_value_catalog)?;
            validate_complete_catalog(
                &self.additions,
                &self.removals,
                &self.materialized,
                [
                    &subject_catalog,
                    &self.predicate_catalog,
                    &self.node_value_catalog,
                ],
                self.state.parent_node_value_count,
                self.state.parent_predicate_count,
                self.state.node_value_count,
                self.state.predicate_count,
            )?;
            if commit_delta_columns(&self.additions, &self.removals)? != self.column_commitments {
                return Err(invalid("native delta column commitment mismatch"));
            }
            if commit_materialized_columns(&self.materialized)?
                != self.materialized_column_commitments
            {
                return Err(invalid("materialized column commitment mismatch"));
            }
            if commit_predicate_dictionary(&self.predicate_catalog)?
                != self.predicate_dict_commitments
            {
                return Err(invalid("predicate dictionary commitment mismatch"));
            }
            if commit_node_value_dictionary(&self.node_value_catalog)?
                != self.node_value_dict_commitments
            {
                return Err(invalid("node/value dictionary commitment mismatch"));
            }
            if commit_node_value_dict_range(&self.node_value_catalog)?
                != self.node_value_dict_range_commitments
            {
                return Err(invalid("node/value range commitment mismatch"));
            }
            if self.compute_commitment_root()? != self.state.commitment_root {
                return Err(invalid("commitment root mismatch"));
            }
            Ok(())
        }

        pub fn encode(&self) -> io::Result<Vec<u8>> {
            self.validate()?;
            let expected = self.encoded_len()?;
            let mut out = self.encode_body()?;
            put_hash(&mut out, &self.state.commitment_root);
            let checksum = file_checksum(&out);
            out.extend_from_slice(&checksum);
            if out.len() != expected {
                return Err(invalid("commitment encoded length accounting mismatch"));
            }
            Ok(out)
        }

        pub fn decode(bytes: &[u8]) -> io::Result<Self> {
            if bytes.len() < COMMITMENT_MAGIC.len() + 4 + 32 + 32
                || bytes.len() > MAX_PROOF_LAYER_BYTES
            {
                return Err(invalid("truncated proof commitment"));
            }
            let (payload, checksum) = bytes.split_at(bytes.len() - 32);
            if file_checksum(payload).as_slice() != checksum {
                return Err(invalid("commitment file checksum mismatch"));
            }
            let (body, root_bytes) = payload.split_at(payload.len() - 32);
            let stored_root: ProofHash = root_bytes.try_into().unwrap();

            let mut r = Reader::new(body);
            if r.take(COMMITMENT_MAGIC.len())? != COMMITMENT_MAGIC {
                return Err(invalid("bad commitment file magic"));
            }
            if r.u32()? != COMMITMENT_FORMAT_VERSION {
                return Err(invalid("unsupported commitment file version"));
            }
            let parent_state_root = match r.u8()? {
                0 => None,
                1 => Some(r.hash()?),
                _ => return Err(invalid("bad parent root tag")),
            };
            let addition_count = r.u64()?;
            let removal_count = r.u64()?;
            let materialized_count = r.u64()?;
            let parent_node_value_count = r.u64()?;
            let parent_predicate_count = r.u64()?;
            let node_value_count = r.u64()?;
            let predicate_count = r.u64()?;
            let additions = read_records(&mut r, PersistedTableKind::Delta)?;
            let removals = read_records(&mut r, PersistedTableKind::Delta)?;
            let materialized = read_records(&mut r, PersistedTableKind::Delta)?;
            let predicate_records = read_records(&mut r, PersistedTableKind::PredicateCatalog)?;
            let object_records = read_records(&mut r, PersistedTableKind::ObjectCatalog)?;
            let mut column_commitments: [ArkGT; COLUMN_COUNT] =
                [<ArkGT as Group>::identity(); COLUMN_COUNT];
            for slot in column_commitments.iter_mut() {
                let bytes = r.bytes()?;
                *slot = deserialize_commitment(&bytes)?;
            }
            let mut materialized_column_commitments: [ArkGT; MATERIALIZED_COLUMN_COUNT] =
                [<ArkGT as Group>::identity(); MATERIALIZED_COLUMN_COUNT];
            for slot in materialized_column_commitments.iter_mut() {
                *slot = deserialize_commitment(&r.bytes()?)?;
            }
            let predicate_dict_commitments = {
                let hash = deserialize_commitment(&r.bytes()?)?;
                let id = deserialize_commitment(&r.bytes()?)?;
                (hash, id)
            };
            let node_value_dict_commitments = {
                let hash = deserialize_commitment(&r.bytes()?)?;
                let id = deserialize_commitment(&r.bytes()?)?;
                (hash, id)
            };
            // Same fixed order as `dictionary_commitments`/`encode_body`: the
            // node/value VALUE then DATATYPE commitment follow the hash/id pair.
            let node_value_dict_range_commitments = {
                let value = deserialize_commitment(&r.bytes()?)?;
                let datatype = deserialize_commitment(&r.bytes()?)?;
                (value, datatype)
            };
            if !r.done() {
                return Err(invalid("trailing commitment file bytes"));
            }
            validate_decoded_heap_budget(
                &additions,
                &removals,
                &materialized,
                &predicate_records,
                &object_records,
            )?;
            let predicate_mappings = predicate_records
                .into_iter()
                .map(|record| {
                    if record.subject_id != 0
                        || record.predicate_id == 0
                        || record.object_id != 0
                        || !record.triple.subject.is_empty()
                        || record.triple.object != CanonicalObject::Node(Vec::new())
                    {
                        return Err(invalid("malformed persisted predicate catalog"));
                    }
                    Ok((record.triple.predicate, record.predicate_id))
                })
                .collect::<io::Result<Vec<_>>>()?;
            let node_value_mappings = object_records
                .into_iter()
                .map(|record| {
                    if record.subject_id != 0
                        || record.predicate_id != 0
                        || record.object_id == 0
                        || !record.triple.subject.is_empty()
                        || !record.triple.predicate.is_empty()
                    {
                        return Err(invalid("malformed persisted node/value catalog"));
                    }
                    Ok((record.triple.object, record.object_id))
                })
                .collect::<io::Result<Vec<_>>>()?;
            let [_, predicate_catalog, node_value_catalog] =
                build_dictionary_catalogs_from_maps(node_value_mappings, predicate_mappings)?;

            let state = CommitmentState {
                parent_state_root,
                addition_count,
                removal_count,
                materialized_count,
                parent_node_value_count,
                parent_predicate_count,
                node_value_count,
                predicate_count,
                commitment_root: stored_root,
            };
            let layer = Self {
                state,
                additions,
                removals,
                materialized,
                predicate_catalog,
                node_value_catalog,
                column_commitments,
                materialized_column_commitments,
                predicate_dict_commitments,
                node_value_dict_commitments,
                node_value_dict_range_commitments,
            };
            layer.validate()?;
            Ok(layer)
        }

        /// Recompute the `commitment_root` binding purely from the state counts,
        /// parent root, the nine triple-column commitments, and the six dictionary
        /// commitments (independent of the raw tables). A query verifier trusts a
        /// layer's `commitment_root`, and calls this to confirm the commitments it
        /// is about to open are exactly the ones that root authenticates.
        pub fn recomputed_commitment_root(&self) -> io::Result<ProofHash> {
            self.compute_commitment_root()
        }

        /// Reconstruct the root this same native delta would have had in PF4 v3.
        /// This is migration-only: it lets an interrupted v3→v4 chain upgrade
        /// authenticate the next legacy child's parent pointer without retaining
        /// legacy metadata in the v4 format.
        pub(crate) fn legacy_v3_root(
            &self,
            legacy_parent_root: Option<ProofHash>,
        ) -> io::Result<ProofHash> {
            let predicate_catalog: Vec<_> = self
                .predicate_catalog
                .iter()
                .filter(|record| record.predicate_id > self.state.parent_predicate_count)
                .cloned()
                .collect();
            let node_value_catalog: Vec<_> = self
                .node_value_catalog
                .iter()
                .filter(|record| record.object_id > self.state.parent_node_value_count)
                .cloned()
                .collect();
            legacy_v3_root_from_parts(
                legacy_parent_root,
                self.state.addition_count,
                self.state.removal_count,
                self.state.parent_node_value_count,
                self.state.parent_predicate_count,
                self.state.node_value_count,
                self.state.predicate_count,
                &self.column_commitments,
                &commit_predicate_dictionary(&predicate_catalog)?,
                &commit_node_value_dictionary(&node_value_catalog)?,
                &commit_node_value_dict_range(&node_value_catalog)?,
            )
        }

        #[cfg(test)]
        pub(crate) fn encode_legacy_v3_for_test(
            &self,
            legacy_parent_root: Option<ProofHash>,
        ) -> io::Result<Vec<u8>> {
            let predicate_catalog: Vec<_> = self
                .predicate_catalog
                .iter()
                .filter(|record| record.predicate_id > self.state.parent_predicate_count)
                .cloned()
                .collect();
            let node_value_catalog: Vec<_> = self
                .node_value_catalog
                .iter()
                .filter(|record| record.object_id > self.state.parent_node_value_count)
                .cloned()
                .collect();
            let predicate_commitments = commit_predicate_dictionary(&predicate_catalog)?;
            let object_commitments = commit_node_value_dictionary(&node_value_catalog)?;
            let range_commitments = commit_node_value_dict_range(&node_value_catalog)?;
            let root = self.legacy_v3_root(legacy_parent_root)?;
            let mut out = Vec::new();
            out.extend_from_slice(COMMITMENT_MAGIC);
            put_u32(&mut out, 3);
            out.push(legacy_parent_root.is_some() as u8);
            if let Some(parent) = legacy_parent_root {
                put_hash(&mut out, &parent);
            }
            for count in [
                self.state.addition_count,
                self.state.removal_count,
                self.state.parent_node_value_count,
                self.state.parent_predicate_count,
                self.state.node_value_count,
                self.state.predicate_count,
            ] {
                put_u64(&mut out, count);
            }
            for table in [
                &self.additions,
                &self.removals,
                &predicate_catalog,
                &node_value_catalog,
            ] {
                put_records(&mut out, table);
            }
            for commitment in self.column_commitments.iter().chain([
                &predicate_commitments.0,
                &predicate_commitments.1,
                &object_commitments.0,
                &object_commitments.1,
                &range_commitments.0,
                &range_commitments.1,
            ]) {
                put_bytes(&mut out, &serialize_commitment(commitment)?);
            }
            put_hash(&mut out, &root);
            let checksum = file_checksum(&out);
            out.extend_from_slice(&checksum);
            Ok(out)
        }

        /// Rebuild the padded predicate-dictionary columns `(hash, id)` that
        /// `predicate_dict_commitments` commits, so a Q3 membership proof can open
        /// them. Same padding/order as the committed columns.
        pub fn predicate_dictionary_columns(&self) -> io::Result<(Vec<ArkFr>, Vec<ArkFr>)> {
            predicate_dict_columns_from_catalog(&self.predicate_catalog)
        }

        /// Rebuild the padded node/value-dictionary columns `(hash, id)` that
        /// `node_value_dict_commitments` commits, so a Q3 membership proof can open
        /// them for a subject/object constant. Same padding/order as the committed
        /// columns.
        pub fn node_value_dictionary_columns(&self) -> io::Result<(Vec<ArkFr>, Vec<ArkFr>)> {
            node_value_dict_columns_from_catalog(&self.node_value_catalog)
        }

        /// Rebuild the padded node/value-dictionary VALUE column that
        /// `node_value_dict_range_commitments.0` commits, so a range proof can open
        /// it. Same padding/order/length as `node_value_dictionary_columns`.
        pub fn node_value_dictionary_value_column(&self) -> io::Result<Vec<ArkFr>> {
            node_value_dict_value_column(&self.node_value_catalog)
        }

        /// Rebuild the padded node/value-dictionary DATATYPE column that
        /// `node_value_dict_range_commitments.1` commits, so the range gadget can
        /// open it and mask rows to the queried datatype. Same padding/order/length
        /// as `node_value_dictionary_columns`.
        pub fn node_value_dictionary_datatype_column(&self) -> io::Result<Vec<ArkFr>> {
            node_value_dict_datatype_column(&self.node_value_catalog)
        }

        /// The six dictionary commitments, in fixed bound order: predicate hash,
        /// predicate id, node/value hash, node/value id, then the node/value
        /// order-preserving VALUE and DATATYPE columns. `commitment_root` binds
        /// them in this order.
        fn dictionary_commitments(&self) -> [&ArkGT; 6] {
            [
                &self.predicate_dict_commitments.0,
                &self.predicate_dict_commitments.1,
                &self.node_value_dict_commitments.0,
                &self.node_value_dict_commitments.1,
                &self.node_value_dict_range_commitments.0,
                &self.node_value_dict_range_commitments.1,
            ]
        }

        /// The commitment root binds the parent root, the seven counts, the six
        /// native-delta and three materialized-view column commitments, and the
        /// six dictionary commitments (the
        /// predicate hash/id pair, the node/value hash/id pair, then the node/value
        /// order-preserving VALUE + DATATYPE pair) — NOT the raw tables, which are
        /// separately bound by the file checksum and revalidated on decode. The
        /// column commitments in turn bind the sorted id columns; the dictionary
        /// commitments bind the IRI->id and constant->id maps and the value/datatype
        /// encodings.
        fn compute_commitment_root(&self) -> io::Result<ProofHash> {
            compute_commitment_root_from_parts(
                &self.state,
                &self.column_commitments,
                &self.materialized_column_commitments,
                &self.predicate_dict_commitments,
                &self.node_value_dict_commitments,
                &self.node_value_dict_range_commitments,
            )
        }

        fn encode_body(&self) -> io::Result<Vec<u8>> {
            let mut out = Vec::new();
            out.extend_from_slice(COMMITMENT_MAGIC);
            put_u32(&mut out, COMMITMENT_FORMAT_VERSION);
            out.push(self.state.parent_state_root.is_some() as u8);
            if let Some(root) = self.state.parent_state_root {
                put_hash(&mut out, &root);
            }
            put_u64(&mut out, self.state.addition_count);
            put_u64(&mut out, self.state.removal_count);
            put_u64(&mut out, self.state.materialized_count);
            put_u64(&mut out, self.state.parent_node_value_count);
            put_u64(&mut out, self.state.parent_predicate_count);
            put_u64(&mut out, self.state.node_value_count);
            put_u64(&mut out, self.state.predicate_count);
            for table in [
                &self.additions,
                &self.removals,
                &self.materialized,
                &self.predicate_catalog,
                &self.node_value_catalog,
            ] {
                put_records(&mut out, table);
            }
            for commitment in self
                .column_commitments
                .iter()
                .chain(self.materialized_column_commitments.iter())
                .chain(self.dictionary_commitments())
            {
                let bytes = serialize_commitment(commitment)?;
                put_bytes(&mut out, &bytes);
            }
            Ok(out)
        }

        fn encoded_len(&self) -> io::Result<usize> {
            let state_bytes = 1usize
                .checked_add(if self.state.parent_state_root.is_some() {
                    32
                } else {
                    0
                })
                .and_then(|value| value.checked_add(7 * 8))
                .ok_or_else(|| invalid("commitment manifest encoded length overflow"))?;
            let mut total = COMMITMENT_MAGIC
                .len()
                .checked_add(4)
                .and_then(|value| value.checked_add(state_bytes))
                .ok_or_else(|| invalid("commitment encoded length overflow"))?;
            for table in [
                &self.additions,
                &self.removals,
                &self.materialized,
                &self.predicate_catalog,
                &self.node_value_catalog,
            ] {
                total = checked_proof_encoded_growth(total, 8)?;
                for record in table.iter() {
                    let record_len = 24usize
                        .checked_add(record.triple.encoded_len()?)
                        .ok_or_else(|| invalid("commitment encoded length overflow"))?;
                    total = checked_proof_encoded_growth(total, record_len)?;
                }
            }
            for commitment in self
                .column_commitments
                .iter()
                .chain(self.materialized_column_commitments.iter())
                .chain(self.dictionary_commitments())
            {
                let commitment_len = 8usize
                    .checked_add(commitment.compressed_size())
                    .ok_or_else(|| invalid("commitment encoded length overflow"))?;
                total = checked_proof_encoded_growth(total, commitment_len)?;
            }
            // commitment_root + file checksum
            total = checked_proof_encoded_growth(total, 32)?;
            checked_proof_encoded_growth(total, 32)
        }
    }

    fn compute_commitment_root_from_parts(
        state: &CommitmentState,
        column_commitments: &[ArkGT; COLUMN_COUNT],
        materialized_column_commitments: &[ArkGT; MATERIALIZED_COLUMN_COUNT],
        predicate_dict_commitments: &(ArkGT, ArkGT),
        node_value_dict_commitments: &(ArkGT, ArkGT),
        node_value_dict_range_commitments: &(ArkGT, ArkGT),
    ) -> io::Result<ProofHash> {
        // COMMITMENT DIET (Wave 0): a DELTA layer (parent_state_root == Some) no longer
        // commits the materialized head view. Its root therefore binds {parent root,
        // add/remove columns, INCREMENTAL dictionary commitments, counts} and omits the
        // materialized column commitments AND the (now always-zero) materialized_count.
        // Because the bound set changed, the delta root uses a BUMPED, distinct domain
        // (`commitment-root-delta/v6`) so it can never collide with either the byte-
        // identical v4 base root or a stale pre-diet delta root (which committed the full
        // materialized view under the v4 domain and now rejects closed on decode).
        //
        // A BASE/ROLLUP/SQUASH layer (parent_state_root == None) is untouched: same
        // `commitment-root/v4` domain, same counts (including materialized_count), and
        // still binds the materialized column commitments — BYTE-IDENTICAL to before.
        let is_delta = state.parent_state_root.is_some();
        let mut hasher = Sha256::new();
        if is_delta {
            hasher.update(b"terminus-store/proof-of-woql/commitment-root-delta/v6");
        } else {
            hasher.update(b"terminus-store/proof-of-woql/commitment-root/v4");
        }
        match state.parent_state_root {
            Some(root) => {
                hasher.update([1u8]);
                hasher.update(root);
            }
            None => hasher.update([0u8]),
        }
        let counts: &[u64] = if is_delta {
            &[
                state.addition_count,
                state.removal_count,
                state.parent_node_value_count,
                state.parent_predicate_count,
                state.node_value_count,
                state.predicate_count,
            ]
        } else {
            &[
                state.addition_count,
                state.removal_count,
                state.materialized_count,
                state.parent_node_value_count,
                state.parent_predicate_count,
                state.node_value_count,
                state.predicate_count,
            ]
        };
        for count in counts {
            hasher.update(count.to_be_bytes());
        }
        let mut commitments: Vec<&ArkGT> = column_commitments.iter().collect();
        if !is_delta {
            commitments.extend(materialized_column_commitments.iter());
        }
        commitments.extend([
            &predicate_dict_commitments.0,
            &predicate_dict_commitments.1,
            &node_value_dict_commitments.0,
            &node_value_dict_commitments.1,
            &node_value_dict_range_commitments.0,
            &node_value_dict_range_commitments.1,
        ]);
        for commitment in commitments {
            let bytes = serialize_commitment(commitment)?;
            hasher.update((bytes.len() as u64).to_be_bytes());
            hasher.update(&bytes);
        }
        Ok(hasher.finalize().into())
    }

    /// SCHEMA-GRAPH FOUNDATION — the combined `database_root`.
    ///
    /// A query-proof database is TWO separately committed triple graphs: the
    /// instance graph and the schema graph. Each is committed by the SAME per-layer
    /// commitment machinery (`compute_commitment_root_from_parts`) as its own
    /// `LayerCommitment` with its own `commitment_root`; the schema graph is NOT
    /// folded into the instance commitment (that would perturb the v4 instance root
    /// and break byte-identity of every existing instance-only proof). Instead the
    /// two per-graph roots are bound into ONE externally-trusted anchor by a plain
    /// two-hash wrapper — no polynomial machinery, no new committed structure:
    ///
    /// `database_root = SHA256("parture/proof-of-woql/db-root/v1" ‖ instance_root ‖ schema_root)`.
    ///
    /// The ordering (instance first, then schema) is fixed and domain-separated, so
    /// the two roots can never be transposed without changing the anchor. An
    /// instance-only proof never engages the schema root: its trust anchor remains
    /// the instance `commitment_root`, byte-identically to today. The combined root
    /// is engaged only when a query reads the schema graph (Subsumption), and only
    /// to authenticate that BOTH sub-commitments belong to the trusted database.
    pub fn database_root(instance_root: &ProofHash, schema_root: &ProofHash) -> ProofHash {
        let mut hasher = Sha256::new();
        hasher.update(b"parture/proof-of-woql/db-root/v1");
        hasher.update(instance_root);
        hasher.update(schema_root);
        hasher.finalize().into()
    }

    /// Validate BOTH per-graph verifier views independently (each recomputes its own
    /// `commitment_root` from its committed parts via `VerifierCommitmentRef::validate`)
    /// AND confirm the two roots combine to the externally-trusted `database_root`.
    ///
    /// This is the soundness gate a schema-reading verifier runs BEFORE trusting either
    /// per-graph root as a proof anchor: neither sub-root can be swapped, forged, or
    /// mismatched against the trusted database anchor without rejecting closed. Reusing
    /// the audited per-commitment `validate` keeps the trust story a strict superset of
    /// the single-graph case — each graph is authenticated exactly as before, plus the
    /// combination check.
    pub fn validate_database_root(
        instance: &impl VerifierLayer,
        schema: &impl VerifierLayer,
        trusted_database_root: &ProofHash,
    ) -> io::Result<()> {
        let instance_view = instance.verifier_view();
        let schema_view = schema.verifier_view();
        instance_view.validate()?;
        schema_view.validate()?;
        if &database_root(
            &instance_view.state.commitment_root,
            &schema_view.state.commitment_root,
        ) != trusted_database_root
        {
            return Err(invalid(
                "database root does not combine the trusted instance and schema commitment roots",
            ));
        }
        Ok(())
    }

    impl VerifierCommitment {
        /// Validate the complete clear catalogs against their Dory commitments,
        /// then recompute the PF4 v4 root from all retained public commitments.
        pub fn validate(&self) -> io::Result<()> {
            self.state.validate()?;
            if self.state.addition_count > MAX_RECORDS as u64
                || self.state.removal_count > MAX_RECORDS as u64
                || self.state.materialized_count > MAX_RECORDS as u64
            {
                return Err(invalid("verifier commitment count exceeds PF4 limit"));
            }
            let subject_catalog = subject_catalog_from_objects(&self.node_value_catalog)?;
            validate_complete_catalog(
                &[],
                &[],
                &[],
                [
                    &subject_catalog,
                    &self.predicate_catalog,
                    &self.node_value_catalog,
                ],
                self.state.parent_node_value_count,
                self.state.parent_predicate_count,
                self.state.node_value_count,
                self.state.predicate_count,
            )?;
            if commit_predicate_dictionary(&self.predicate_catalog)?
                != self.predicate_dict_commitments
            {
                return Err(invalid("verifier predicate catalog commitment mismatch"));
            }
            if commit_node_value_dictionary(&self.node_value_catalog)?
                != self.node_value_dict_commitments
            {
                return Err(invalid("verifier node/value catalog commitment mismatch"));
            }
            if commit_node_value_dict_range(&self.node_value_catalog)?
                != self.node_value_dict_range_commitments
            {
                return Err(invalid("verifier range catalog commitment mismatch"));
            }
            if self.recomputed_commitment_root()? != self.state.commitment_root {
                return Err(invalid("verifier commitment root mismatch"));
            }
            Ok(())
        }

        pub fn recomputed_commitment_root(&self) -> io::Result<ProofHash> {
            compute_commitment_root_from_parts(
                &self.state,
                &self.column_commitments,
                &self.materialized_column_commitments,
                &self.predicate_dict_commitments,
                &self.node_value_dict_commitments,
                &self.node_value_dict_range_commitments,
            )
        }

        pub fn encode(&self) -> io::Result<Vec<u8>> {
            self.validate()?;
            let mut out = Vec::new();
            out.extend_from_slice(VERIFIER_COMMITMENT_MAGIC);
            put_u32(&mut out, VERIFIER_COMMITMENT_FORMAT_VERSION);
            out.push(self.state.parent_state_root.is_some() as u8);
            if let Some(root) = self.state.parent_state_root {
                put_hash(&mut out, &root);
            }
            for count in [
                self.state.addition_count,
                self.state.removal_count,
                self.state.materialized_count,
                self.state.parent_node_value_count,
                self.state.parent_predicate_count,
                self.state.node_value_count,
                self.state.predicate_count,
            ] {
                put_u64(&mut out, count);
            }
            put_records(&mut out, &self.predicate_catalog);
            put_records(&mut out, &self.node_value_catalog);
            for commitment in self.column_commitments.iter().chain(
                self.materialized_column_commitments.iter(),
            ).chain([
                &self.predicate_dict_commitments.0,
                &self.predicate_dict_commitments.1,
                &self.node_value_dict_commitments.0,
                &self.node_value_dict_commitments.1,
                &self.node_value_dict_range_commitments.0,
                &self.node_value_dict_range_commitments.1,
            ]) {
                put_bytes(&mut out, &serialize_commitment(commitment)?);
            }
            put_hash(&mut out, &self.state.commitment_root);
            let checksum = file_checksum(&out);
            out.extend_from_slice(&checksum);
            if out.len() > MAX_PROOF_LAYER_BYTES {
                return Err(invalid("encoded verifier commitment exceeds file budget"));
            }
            Ok(out)
        }

        pub fn decode(bytes: &[u8]) -> io::Result<Self> {
            if bytes.len() < VERIFIER_COMMITMENT_MAGIC.len() + 4 + 64
                || bytes.len() > MAX_PROOF_LAYER_BYTES
            {
                return Err(invalid("truncated verifier commitment"));
            }
            let (payload, checksum) = bytes.split_at(bytes.len() - 32);
            if file_checksum(payload).as_slice() != checksum {
                return Err(invalid("verifier commitment checksum mismatch"));
            }
            let (body, root_bytes) = payload.split_at(payload.len() - 32);
            let commitment_root: ProofHash = root_bytes.try_into().unwrap();
            let mut r = Reader::new(body);
            if r.take(VERIFIER_COMMITMENT_MAGIC.len())? != VERIFIER_COMMITMENT_MAGIC {
                return Err(invalid("bad verifier commitment magic"));
            }
            if r.u32()? != VERIFIER_COMMITMENT_FORMAT_VERSION {
                return Err(invalid("unsupported verifier commitment version"));
            }
            let parent_state_root = match r.u8()? {
                0 => None,
                1 => Some(r.hash()?),
                _ => return Err(invalid("bad verifier parent root tag")),
            };
            let state = CommitmentState {
                parent_state_root,
                addition_count: r.u64()?,
                removal_count: r.u64()?,
                materialized_count: r.u64()?,
                parent_node_value_count: r.u64()?,
                parent_predicate_count: r.u64()?,
                node_value_count: r.u64()?,
                predicate_count: r.u64()?,
                commitment_root,
            };
            let predicate_catalog = read_records(&mut r, PersistedTableKind::PredicateCatalog)?;
            let node_value_catalog = read_records(&mut r, PersistedTableKind::ObjectCatalog)?;
            validate_decoded_heap_budget(&[], &[], &[], &predicate_catalog, &node_value_catalog)?;
            let mut column_commitments = [<ArkGT as Group>::identity(); COLUMN_COUNT];
            for value in &mut column_commitments {
                *value = deserialize_commitment(&r.bytes()?)?;
            }
            let mut materialized_column_commitments =
                [<ArkGT as Group>::identity(); MATERIALIZED_COLUMN_COUNT];
            for value in &mut materialized_column_commitments {
                *value = deserialize_commitment(&r.bytes()?)?;
            }
            let mut next_pair = || -> io::Result<(ArkGT, ArkGT)> {
                Ok((
                    deserialize_commitment(&r.bytes()?)?,
                    deserialize_commitment(&r.bytes()?)?,
                ))
            };
            let predicate_dict_commitments = next_pair()?;
            let node_value_dict_commitments = next_pair()?;
            let node_value_dict_range_commitments = next_pair()?;
            if !r.done() {
                return Err(invalid("trailing verifier commitment bytes"));
            }
            let artifact = Self {
                state,
                predicate_catalog,
                node_value_catalog,
                column_commitments,
                materialized_column_commitments,
                predicate_dict_commitments,
                node_value_dict_commitments,
                node_value_dict_range_commitments,
            };
            artifact.validate()?;
            Ok(artifact)
        }
    }

    /// Fully validated migration-only representation of the immediately preceding
    /// PF4 v3 sidecar format. Normal reads and pack admission never call this decoder.
    pub(crate) struct LegacyV3Commitment {
        pub parent_state_root: Option<ProofHash>,
        pub commitment_root: ProofHash,
        addition_count: u64,
        removal_count: u64,
        parent_node_value_count: u64,
        parent_predicate_count: u64,
        node_value_count: u64,
        predicate_count: u64,
        additions: Vec<ProofRecord>,
        removals: Vec<ProofRecord>,
        predicate_catalog: Vec<ProofRecord>,
        node_value_catalog: Vec<ProofRecord>,
    }

    impl LegacyV3Commitment {
        pub(crate) fn validate_against_v4(&self, current: &LayerCommitment) -> io::Result<()> {
            let legacy_predicates: Vec<_> = current
                .predicate_catalog
                .iter()
                .filter(|record| record.predicate_id > current.state.parent_predicate_count)
                .cloned()
                .collect();
            let legacy_objects: Vec<_> = current
                .node_value_catalog
                .iter()
                .filter(|record| record.object_id > current.state.parent_node_value_count)
                .cloned()
                .collect();
            if self.addition_count != current.state.addition_count
                || self.removal_count != current.state.removal_count
                || self.parent_node_value_count != current.state.parent_node_value_count
                || self.parent_predicate_count != current.state.parent_predicate_count
                || self.node_value_count != current.state.node_value_count
                || self.predicate_count != current.state.predicate_count
                || self.additions != current.additions
                || self.removals != current.removals
                || self.predicate_catalog != legacy_predicates
                || self.node_value_catalog != legacy_objects
            {
                return Err(invalid(
                    "legacy v3 proof state differs from native layer delta",
                ));
            }
            Ok(())
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn legacy_v3_root_from_parts(
        parent_state_root: Option<ProofHash>,
        addition_count: u64,
        removal_count: u64,
        parent_node_value_count: u64,
        parent_predicate_count: u64,
        node_value_count: u64,
        predicate_count: u64,
        column_commitments: &[ArkGT; COLUMN_COUNT],
        predicate_dict_commitments: &(ArkGT, ArkGT),
        node_value_dict_commitments: &(ArkGT, ArkGT),
        node_value_dict_range_commitments: &(ArkGT, ArkGT),
    ) -> io::Result<ProofHash> {
        let mut hasher = Sha256::new();
        hasher.update(b"terminus-store/proof-of-woql/commitment-root/v3");
        match parent_state_root {
            Some(root) => {
                hasher.update([1u8]);
                hasher.update(root);
            }
            None => hasher.update([0u8]),
        }
        for count in [
            addition_count,
            removal_count,
            parent_node_value_count,
            parent_predicate_count,
            node_value_count,
            predicate_count,
        ] {
            hasher.update(count.to_be_bytes());
        }
        for commitment in column_commitments.iter().chain([
            &predicate_dict_commitments.0,
            &predicate_dict_commitments.1,
            &node_value_dict_commitments.0,
            &node_value_dict_commitments.1,
            &node_value_dict_range_commitments.0,
            &node_value_dict_range_commitments.1,
        ]) {
            let bytes = serialize_commitment(commitment)?;
            hasher.update((bytes.len() as u64).to_be_bytes());
            hasher.update(bytes);
        }
        Ok(hasher.finalize().into())
    }

    pub(crate) fn decode_legacy_v3(bytes: &[u8]) -> io::Result<LegacyV3Commitment> {
        if bytes.len() < COMMITMENT_MAGIC.len() + 4 + 64 || bytes.len() > MAX_PROOF_LAYER_BYTES {
            return Err(invalid("truncated legacy v3 proof commitment"));
        }
        let (payload, checksum) = bytes.split_at(bytes.len() - 32);
        if file_checksum(payload).as_slice() != checksum {
            return Err(invalid("legacy v3 commitment file checksum mismatch"));
        }
        let (body, root_bytes) = payload.split_at(payload.len() - 32);
        let commitment_root: ProofHash = root_bytes.try_into().unwrap();
        let mut r = Reader::new(body);
        if r.take(COMMITMENT_MAGIC.len())? != COMMITMENT_MAGIC || r.u32()? != 3 {
            return Err(invalid("not a PF4 v3 commitment file"));
        }
        let parent_state_root = match r.u8()? {
            0 => None,
            1 => Some(r.hash()?),
            _ => return Err(invalid("bad legacy v3 parent root tag")),
        };
        let addition_count = r.u64()?;
        let removal_count = r.u64()?;
        let parent_node_value_count = r.u64()?;
        let parent_predicate_count = r.u64()?;
        let node_value_count = r.u64()?;
        let predicate_count = r.u64()?;
        let additions = read_records(&mut r, PersistedTableKind::Delta)?;
        let removals = read_records(&mut r, PersistedTableKind::Delta)?;
        let predicate_records = read_records(&mut r, PersistedTableKind::PredicateCatalog)?;
        let object_records = read_records(&mut r, PersistedTableKind::ObjectCatalog)?;
        let mut column_commitments = [<ArkGT as Group>::identity(); COLUMN_COUNT];
        for commitment in &mut column_commitments {
            *commitment = deserialize_commitment(&r.bytes()?)?;
        }
        let predicate_dict_commitments = (
            deserialize_commitment(&r.bytes()?)?,
            deserialize_commitment(&r.bytes()?)?,
        );
        let node_value_dict_commitments = (
            deserialize_commitment(&r.bytes()?)?,
            deserialize_commitment(&r.bytes()?)?,
        );
        let node_value_dict_range_commitments = (
            deserialize_commitment(&r.bytes()?)?,
            deserialize_commitment(&r.bytes()?)?,
        );
        if !r.done() {
            return Err(invalid("trailing legacy v3 commitment bytes"));
        }
        let predicate_mappings = predicate_records
            .into_iter()
            .map(|record| {
                if record.subject_id != 0
                    || record.predicate_id == 0
                    || record.object_id != 0
                    || !record.triple.subject.is_empty()
                    || record.triple.object != CanonicalObject::Node(Vec::new())
                {
                    return Err(invalid("malformed legacy v3 predicate catalog"));
                }
                Ok((record.triple.predicate, record.predicate_id))
            })
            .collect::<io::Result<Vec<_>>>()?;
        let node_value_mappings = object_records
            .into_iter()
            .map(|record| {
                if record.subject_id != 0
                    || record.predicate_id != 0
                    || record.object_id == 0
                    || !record.triple.subject.is_empty()
                    || !record.triple.predicate.is_empty()
                {
                    return Err(invalid("malformed legacy v3 node/value catalog"));
                }
                Ok((record.triple.object, record.object_id))
            })
            .collect::<io::Result<Vec<_>>>()?;
        let [_, predicate_catalog, node_value_catalog] =
            build_dictionary_catalogs_from_maps(node_value_mappings, predicate_mappings)?;
        if additions.len() as u64 != addition_count || removals.len() as u64 != removal_count {
            return Err(invalid("legacy v3 delta count mismatch"));
        }
        if commit_delta_columns(&additions, &removals)? != column_commitments
            || commit_predicate_dictionary(&predicate_catalog)? != predicate_dict_commitments
            || commit_node_value_dictionary(&node_value_catalog)? != node_value_dict_commitments
            || commit_node_value_dict_range(&node_value_catalog)?
                != node_value_dict_range_commitments
        {
            return Err(invalid("legacy v3 raw table commitment mismatch"));
        }
        let recomputed = legacy_v3_root_from_parts(
            parent_state_root,
            addition_count,
            removal_count,
            parent_node_value_count,
            parent_predicate_count,
            node_value_count,
            predicate_count,
            &column_commitments,
            &predicate_dict_commitments,
            &node_value_dict_commitments,
            &node_value_dict_range_commitments,
        )?;
        if recomputed != commitment_root {
            return Err(invalid("legacy v3 commitment root mismatch"));
        }
        Ok(LegacyV3Commitment {
            parent_state_root,
            commitment_root,
            addition_count,
            removal_count,
            parent_node_value_count,
            parent_predicate_count,
            node_value_count,
            predicate_count,
            additions,
            removals,
            predicate_catalog,
            node_value_catalog,
        })
    }

    fn put_records(out: &mut Vec<u8>, records: &[ProofRecord]) {
        put_u64(out, records.len() as u64);
        for record in records {
            put_u64(out, record.subject_id);
            put_u64(out, record.predicate_id);
            put_u64(out, record.object_id);
            record.triple.encode_into(out);
        }
    }

    fn file_checksum(payload: &[u8]) -> ProofHash {
        let mut hasher = Sha256::new();
        hasher.update(b"terminus-store/proof-of-woql/file/v0");
        hasher.update((payload.len() as u64).to_be_bytes());
        hasher.update(payload);
        hasher.finalize().into()
    }

    fn build_dictionary_catalogs_from_maps(
        node_value_mappings: Vec<(CanonicalObject, u64)>,
        predicate_mappings: Vec<(Vec<u8>, u64)>,
    ) -> io::Result<[Vec<ProofRecord>; 3]> {
        let mut subjects = BTreeMap::<Vec<u8>, u64>::new();
        let mut predicates = BTreeMap::<Vec<u8>, u64>::new();
        let mut objects = BTreeMap::<CanonicalObject, u64>::new();
        for (object, id) in node_value_mappings {
            if let CanonicalObject::Node(node) = &object {
                insert_mapping(&mut subjects, node.clone(), id)?;
            }
            insert_mapping(&mut objects, object, id)?;
        }
        for (predicate, id) in predicate_mappings {
            insert_mapping(&mut predicates, predicate, id)?;
        }
        let subject_records = subjects
            .into_iter()
            .map(|(subject, subject_id)| ProofRecord {
                subject_id,
                predicate_id: 0,
                object_id: 0,
                triple: CanonicalTriple::node(subject, Vec::new(), Vec::new()),
            })
            .collect();
        let predicate_records = predicates
            .into_iter()
            .map(|(predicate, predicate_id)| ProofRecord {
                subject_id: 0,
                predicate_id,
                object_id: 0,
                triple: CanonicalTriple::node(Vec::new(), predicate, Vec::new()),
            })
            .collect();
        let object_records = objects
            .into_iter()
            .map(|(object, object_id)| ProofRecord {
                subject_id: 0,
                predicate_id: 0,
                object_id,
                triple: CanonicalTriple {
                    subject: Vec::new(),
                    predicate: Vec::new(),
                    object,
                },
            })
            .collect();
        Ok([subject_records, predicate_records, object_records])
    }

    fn subject_catalog_from_objects(objects: &[ProofRecord]) -> io::Result<Vec<ProofRecord>> {
        let mut subjects = BTreeMap::<Vec<u8>, u64>::new();
        for record in objects {
            if let CanonicalObject::Node(node) = &record.triple.object {
                insert_mapping(&mut subjects, node.clone(), record.object_id)?;
            }
        }
        Ok(subjects
            .into_iter()
            .map(|(subject, subject_id)| ProofRecord {
                subject_id,
                predicate_id: 0,
                object_id: 0,
                triple: CanonicalTriple::node(subject, Vec::new(), Vec::new()),
            })
            .collect())
    }

    fn insert_mapping<K: Ord>(map: &mut BTreeMap<K, u64>, key: K, id: u64) -> io::Result<()> {
        if let Some(previous) = map.insert(key, id) {
            if previous != id {
                return Err(invalid(
                    "canonical dictionary value maps to multiple native ids",
                ));
            }
        }
        Ok(())
    }

    fn insert_bijection<K: Ord + Clone>(
        by_key: &mut BTreeMap<K, u64>,
        by_id: &mut BTreeMap<u64, K>,
        key: K,
        id: u64,
    ) -> io::Result<()> {
        insert_mapping(by_key, key.clone(), id)?;
        if let Some(previous) = by_id.insert(id, key.clone()) {
            if previous != key {
                return Err(invalid(
                    "native dictionary id maps to multiple canonical values",
                ));
            }
        }
        Ok(())
    }

    fn validate_dictionary_catalogs(
        actual: [&[ProofRecord]; 3],
        node_value_count: u64,
        predicate_count: u64,
    ) -> io::Result<()> {
        let mut subjects = BTreeMap::new();
        let mut subject_ids = BTreeMap::new();
        for record in actual[0].iter() {
            if record.subject_id == 0
                || record.subject_id > node_value_count
                || record.predicate_id != 0
                || record.object_id != 0
                || !record.triple.predicate.is_empty()
                || record.triple.object != CanonicalObject::Node(Vec::new())
            {
                return Err(invalid("malformed subject dictionary record"));
            }
            insert_bijection(
                &mut subjects,
                &mut subject_ids,
                record.triple.subject.clone(),
                record.subject_id,
            )?;
        }
        let mut predicates = BTreeMap::new();
        let mut predicate_ids = BTreeMap::new();
        for record in actual[1].iter() {
            if record.predicate_id == 0
                || record.predicate_id > predicate_count
                || record.subject_id != 0
                || record.object_id != 0
                || !record.triple.subject.is_empty()
                || record.triple.object != CanonicalObject::Node(Vec::new())
            {
                return Err(invalid("malformed predicate dictionary record"));
            }
            insert_bijection(
                &mut predicates,
                &mut predicate_ids,
                record.triple.predicate.clone(),
                record.predicate_id,
            )?;
        }
        let mut objects = BTreeMap::new();
        let mut object_ids = BTreeMap::new();
        for record in actual[2].iter() {
            if record.object_id == 0
                || record.object_id > node_value_count
                || record.subject_id != 0
                || record.predicate_id != 0
                || !record.triple.subject.is_empty()
                || !record.triple.predicate.is_empty()
            {
                return Err(invalid("malformed object dictionary record"));
            }
            insert_bijection(
                &mut objects,
                &mut object_ids,
                record.triple.object.clone(),
                record.object_id,
            )?;
        }
        let expected_subjects: BTreeMap<_, _> = objects
            .iter()
            .filter_map(|(object, id)| match object {
                CanonicalObject::Node(node) => Some((node.clone(), *id)),
                CanonicalObject::Value { .. } => None,
            })
            .collect();
        if subjects != expected_subjects {
            return Err(invalid(
                "subject dictionary is not the node dictionary subset",
            ));
        }
        Ok(())
    }

    /// Validate that the (possibly INCREMENTAL) dictionary catalog covers exactly the
    /// id interval this layer INTRODUCED — `(parent_count, count]` — and that every
    /// delta/materialized record id introduced by this layer is bound to canonical
    /// bytes by that catalog.
    ///
    /// For a BASE/ROLLUP/SQUASH layer `parent_*_count == 0`, so the interval is the
    /// full `1..=count` and every record id is bound: this collapses to the exact
    /// pre-diet check and stays BYTE-IDENTICAL. For a DELTA layer the catalog holds
    /// only `parent_count+1..=count` (the commitment-diet incremental dictionary), so
    /// INHERITED ids (`<= parent_count`) are deliberately NOT bound here — they are
    /// authenticated by an ancestor layer's committed dictionary when the delta stack
    /// is reconciled at proof time (Wave 1/2). A delta layer is never directly
    /// query-provable, so no verifier trusts an inherited-id binding from this sidecar.
    #[allow(clippy::too_many_arguments)]
    fn validate_complete_catalog(
        additions: &[ProofRecord],
        removals: &[ProofRecord],
        materialized: &[ProofRecord],
        indexes: [&[ProofRecord]; 3],
        parent_node_value_count: u64,
        parent_predicate_count: u64,
        node_value_count: u64,
        predicate_count: u64,
    ) -> io::Result<()> {
        validate_dictionary_catalogs(indexes, node_value_count, predicate_count)?;
        if indexes[2].len() as u64 != node_value_count - parent_node_value_count
            || indexes[1].len() as u64 != predicate_count - parent_predicate_count
        {
            return Err(invalid(
                "dictionary catalog does not exactly cover the introduced id interval",
            ));
        }
        // Every catalog id must lie strictly above the inherited watermark. For a base
        // layer the watermark is 0, so this is vacuous (ids already validated `>= 1`).
        if indexes[0]
            .iter()
            .any(|record| record.subject_id <= parent_node_value_count)
            || indexes[2]
                .iter()
                .any(|record| record.object_id <= parent_node_value_count)
            || indexes[1]
                .iter()
                .any(|record| record.predicate_id <= parent_predicate_count)
        {
            return Err(invalid(
                "dictionary catalog contains an inherited id below the parent watermark",
            ));
        }

        let mut objects_by_id = BTreeMap::new();
        for record in indexes[2].iter() {
            objects_by_id.insert(record.object_id, record.triple.object.clone());
        }
        let mut predicates_by_id = BTreeMap::new();
        for record in indexes[1].iter() {
            predicates_by_id.insert(record.predicate_id, record.triple.predicate.clone());
        }
        let subjects_by_id: BTreeMap<_, _> = indexes[0]
            .iter()
            .map(|record| (record.subject_id, record.triple.subject.clone()))
            .collect();

        for record in additions
            .iter()
            .chain(removals.iter())
            .chain(materialized.iter())
        {
            // Only ids INTRODUCED by this layer are locally bound to canonical bytes;
            // inherited ids are authenticated by an ancestor at reconciliation time.
            if record.subject_id > parent_node_value_count
                && (subjects_by_id.get(&record.subject_id) != Some(&record.triple.subject)
                    || objects_by_id.get(&record.subject_id)
                        != Some(&CanonicalObject::Node(record.triple.subject.clone())))
            {
                return Err(invalid("subject id is not bound to canonical bytes"));
            }
            if record.predicate_id > parent_predicate_count
                && predicates_by_id.get(&record.predicate_id) != Some(&record.triple.predicate)
            {
                return Err(invalid("predicate id is not bound to canonical bytes"));
            }
            if record.object_id > parent_node_value_count
                && objects_by_id.get(&record.object_id) != Some(&record.triple.object)
            {
                return Err(invalid("object id is not bound to canonical bytes"));
            }
        }
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn record(s: u64, p: u64, o: u64, subj: &str, pred: &str, obj: &str) -> ProofRecord {
            ProofRecord {
                subject_id: s,
                predicate_id: p,
                object_id: o,
                triple: CanonicalTriple::node(subj.as_bytes(), pred.as_bytes(), obj.as_bytes()),
            }
        }

        fn sample() -> LayerCommitment {
            let additions = vec![record(1, 1, 2, "a", "p", "b")];
            LayerCommitment::from_native_delta(
                None,
                additions.clone(),
                vec![],
                additions,
                0,
                0,
                2,
                1,
                vec![
                    (CanonicalObject::Node(b"a".to_vec()), 1),
                    (CanonicalObject::Node(b"b".to_vec()), 2),
                ],
                vec![(b"p".to_vec(), 1)],
            )
            .unwrap()
        }

        #[test]
        fn float_range_catalog_rejects_nan_semantic_mismatch() {
            let encoded_nan = (f64::NAN.to_bits() ^ 0x8000_0000_0000_0000)
                .to_be_bytes()
                .to_vec();
            let catalog = vec![ProofRecord {
                subject_id: 0,
                predicate_id: 0,
                object_id: 1,
                triple: CanonicalTriple {
                    subject: Vec::new(),
                    predicate: Vec::new(),
                    object: CanonicalObject::Value {
                        datatype: CanonicalDatatype::Float64,
                        lexical: encoded_nan,
                    },
                },
            }];
            assert!(reject_nan_catalog(
                &catalog,
                ArkFr::from_u64(CanonicalDatatype::Float64 as u64),
            )
            .is_err());
            assert!(reject_nan_catalog(
                &catalog,
                ArkFr::from_u64(CanonicalDatatype::UInt64 as u64),
            )
            .is_ok());
        }

        #[test]
        fn compact_verifier_commitment_roundtrips_and_omits_witness_tables() {
            let layer = sample();
            let full = layer.encode().unwrap();
            let artifact = layer.verifier_commitment().unwrap();
            let compact = artifact.encode().unwrap();
            let decoded = VerifierCommitment::decode(&compact).unwrap();

            assert_eq!(decoded.state, layer.state);
            assert_eq!(decoded.predicate_catalog, layer.predicate_catalog);
            assert_eq!(decoded.node_value_catalog, layer.node_value_catalog);
            assert!(compact.len() < full.len());
            assert_eq!(decoded.recomputed_commitment_root().unwrap(), layer.state.commitment_root);
        }

        #[test]
        fn compact_verifier_commitment_rejects_root_catalog_commitment_and_count_tampering() {
            let artifact = sample().verifier_commitment().unwrap();

            let mut wrong_root = artifact.clone();
            wrong_root.state.commitment_root[0] ^= 1;
            assert!(wrong_root.validate().is_err());

            let mut wrong_catalog = artifact.clone();
            wrong_catalog.node_value_catalog[0].triple.object =
                CanonicalObject::Node(b"forged".to_vec());
            assert!(wrong_catalog.validate().is_err());

            let mut wrong_commitment = artifact.clone();
            wrong_commitment.materialized_column_commitments[0] =
                <ArkGT as Group>::identity();
            assert!(wrong_commitment.validate().is_err());

            let mut wrong_count = artifact.clone();
            wrong_count.state.node_value_count += 1;
            assert!(wrong_count.validate().is_err());
        }

        #[test]
        fn compact_verifier_commitment_decode_rejects_checksum_and_version_tampering() {
            let mut bytes = sample().verifier_commitment().unwrap().encode().unwrap();
            bytes[VERIFIER_COMMITMENT_MAGIC.len()] ^= 1;
            assert!(VerifierCommitment::decode(&bytes).is_err());

            let mut bytes = sample().verifier_commitment().unwrap().encode().unwrap();
            let last = bytes.len() - 1;
            bytes[last] ^= 1;
            assert!(VerifierCommitment::decode(&bytes).is_err());
        }

        // Same two-node/one-predicate dictionary {a:1, b:2}, {p:1}, but the caller
        // chooses whether the triple lands in additions and/or removals.
        fn delta(additions: Vec<ProofRecord>, removals: Vec<ProofRecord>) -> LayerCommitment {
            let materialized = additions.clone();
            LayerCommitment::from_native_delta(
                None,
                additions,
                removals,
                materialized,
                0,
                0,
                2,
                1,
                vec![
                    (CanonicalObject::Node(b"a".to_vec()), 1),
                    (CanonicalObject::Node(b"b".to_vec()), 2),
                ],
                vec![(b"p".to_vec(), 1)],
            )
            .unwrap()
        }

        #[test]
        fn encode_decode_roundtrip_preserves_state_and_columns() {
            let commitment = sample();
            let bytes = commitment.encode().unwrap();
            assert_eq!(bytes.len(), commitment.encoded_len().unwrap());
            let decoded = LayerCommitment::decode(&bytes).unwrap();
            assert_eq!(decoded.state, commitment.state);
            assert_eq!(decoded.additions, commitment.additions);
            assert_eq!(decoded.node_value_catalog, commitment.node_value_catalog);
            assert_eq!(decoded.column_commitments, commitment.column_commitments);
            assert_eq!(
                decoded.predicate_dict_commitments,
                commitment.predicate_dict_commitments
            );
            assert_eq!(
                decoded.node_value_dict_commitments,
                commitment.node_value_dict_commitments
            );
            assert_eq!(
                decoded.node_value_dict_range_commitments,
                commitment.node_value_dict_range_commitments
            );
            assert_eq!(
                decoded.state.commitment_root,
                commitment.state.commitment_root
            );
        }

        // A base layer with three predicates p1<p2<p3 (dict of size 3, padded to 4)
        // over distinct subjects/objects.
        fn multi_predicate_layer() -> LayerCommitment {
            let additions = vec![
                record(1, 1, 2, "a", "p1", "b"),
                record(3, 2, 4, "c", "p2", "d"),
                record(5, 3, 6, "e", "p3", "f"),
            ];
            LayerCommitment::from_native_delta(
                None,
                additions.clone(),
                vec![],
                additions,
                0,
                0,
                6,
                3,
                vec![
                    (CanonicalObject::Node(b"a".to_vec()), 1),
                    (CanonicalObject::Node(b"b".to_vec()), 2),
                    (CanonicalObject::Node(b"c".to_vec()), 3),
                    (CanonicalObject::Node(b"d".to_vec()), 4),
                    (CanonicalObject::Node(b"e".to_vec()), 5),
                    (CanonicalObject::Node(b"f".to_vec()), 6),
                ],
                vec![
                    (b"p1".to_vec(), 1),
                    (b"p2".to_vec(), 2),
                    (b"p3".to_vec(), 3),
                ],
            )
            .unwrap()
        }

        #[test]
        fn predicate_dictionary_commitments_align_with_shared_commit_column() {
            use crate::proof::argument::commit_column;
            use crate::proof::dict::hash_term;

            let layer = multi_predicate_layer();
            assert_eq!(layer.predicate_catalog.len(), 3);
            // Rebuild the two padded dict columns exactly as the query side would,
            // and confirm they match the stored commitments byte-for-byte.
            let zero = <ArkFr as DoryField>::zero();
            let mut hashes = vec![zero; 4];
            let mut ids = vec![zero; 4];
            for (k, record) in layer.predicate_catalog.iter().enumerate() {
                hashes[k] = hash_term(&record.triple.predicate);
                ids[k] = <ArkFr as DoryField>::from_u64(record.predicate_id);
            }
            assert_eq!(
                commit_column(&hashes).unwrap(),
                layer.predicate_dict_commitments.0
            );
            assert_eq!(
                commit_column(&ids).unwrap(),
                layer.predicate_dict_commitments.1
            );
        }

        #[test]
        fn commitment_root_binds_the_predicate_dictionary() {
            // Determinism: the same layer (incl. the dict commitments) yields the
            // same root across two independent builds.
            assert_eq!(
                multi_predicate_layer().state.commitment_root,
                multi_predicate_layer().state.commitment_root
            );
            // Changing a predicate TERM (same ids) changes the dict-hash commitment,
            // hence the root.
            let changed_additions = vec![
                record(1, 1, 2, "a", "pX", "b"),
                record(3, 2, 4, "c", "p2", "d"),
                record(5, 3, 6, "e", "p3", "f"),
            ];
            let changed = LayerCommitment::from_native_delta(
                None,
                changed_additions.clone(),
                vec![],
                changed_additions,
                0,
                0,
                6,
                3,
                vec![
                    (CanonicalObject::Node(b"a".to_vec()), 1),
                    (CanonicalObject::Node(b"b".to_vec()), 2),
                    (CanonicalObject::Node(b"c".to_vec()), 3),
                    (CanonicalObject::Node(b"d".to_vec()), 4),
                    (CanonicalObject::Node(b"e".to_vec()), 5),
                    (CanonicalObject::Node(b"f".to_vec()), 6),
                ],
                vec![
                    (b"pX".to_vec(), 1),
                    (b"p2".to_vec(), 2),
                    (b"p3".to_vec(), 3),
                ],
            )
            .unwrap();
            assert_ne!(
                multi_predicate_layer().state.commitment_root,
                changed.state.commitment_root
            );
        }

        // A base layer with a NODE object (b, id 2) and a typed VALUE literal
        // object ("lit"^^string, id 3), both under subject a (id 1) / predicate p
        // (id 1). Exercises both `CanonicalObject` variants in the node/value dict.
        fn node_and_value_layer() -> LayerCommitment {
            let value_triple = ProofRecord {
                subject_id: 1,
                predicate_id: 1,
                object_id: 3,
                triple: CanonicalTriple {
                    subject: b"a".to_vec(),
                    predicate: b"p".to_vec(),
                    object: CanonicalObject::Value {
                        datatype: CanonicalDatatype::String,
                        lexical: b"lit".to_vec(),
                    },
                },
            };
            let additions = vec![record(1, 1, 2, "a", "p", "b"), value_triple];
            LayerCommitment::from_native_delta(
                None,
                additions.clone(),
                vec![],
                additions,
                0,
                0,
                3,
                1,
                vec![
                    (CanonicalObject::Node(b"a".to_vec()), 1),
                    (CanonicalObject::Node(b"b".to_vec()), 2),
                    (
                        CanonicalObject::Value {
                            datatype: CanonicalDatatype::String,
                            lexical: b"lit".to_vec(),
                        },
                        3,
                    ),
                ],
                vec![(b"p".to_vec(), 1)],
            )
            .unwrap()
        }

        #[test]
        fn commitment_format_version_is_five() {
            // v5 == date/dateTime dict entries carry a decoded OP value (see
            // `node_value_op_value`); the root domain string stays `commitment-root/v4`.
            assert_eq!(COMMITMENT_FORMAT_VERSION, 5);
            assert_eq!(VERIFIER_COMMITMENT_FORMAT_VERSION, 2);
        }

        #[test]
        fn setup_tiers_cover_the_full_pf4_record_domain() {
            assert_eq!(MAX_DECODED_RECORDS, MAX_RECORDS * PERSISTED_TABLE_COUNT);
            assert_eq!(padded_domain_for_rows(1 << 20).unwrap(), (1 << 20, 20));
            assert_eq!(
                padded_domain_for_rows((1 << 20) + 1).unwrap(),
                (1 << 21, 21)
            );
            assert_eq!(setup_tier_for_num_vars(20).unwrap(), 20);
            assert_eq!(setup_tier_for_num_vars(21).unwrap(), 22);
            assert_eq!(setup_tier_for_num_vars(22).unwrap(), 22);
            assert_eq!(setup_tier_for_num_vars(23).unwrap(), 24);
            assert_eq!(setup_tier_for_num_vars(24).unwrap(), 24);
            assert!(setup_tier_for_num_vars(25)
                .unwrap_err()
                .to_string()
                .contains("exceeds PF4's maximum"));

            let (padded, log) = padded_domain_for_rows(MAX_RECORDS).unwrap();
            assert_eq!(padded, 1 << SETUP_MAX_LOG_N);
            assert_eq!(log, SETUP_MAX_LOG_N);
            assert!(padded_domain_for_rows(MAX_RECORDS + 1)
                .unwrap_err()
                .to_string()
                .contains("above the PF4 limit"));
        }

        #[test]
        fn setup_memory_and_commitment_scratch_are_preflighted() {
            assert_eq!(setup_generator_count(20).unwrap(), 1 << 10);
            assert_eq!(setup_generator_count(22).unwrap(), 1 << 11);
            assert_eq!(setup_generator_count(24).unwrap(), 1 << 12);
            let retained = setup_retained_bytes(SETUP_MAX_LOG_N).unwrap();
            assert!(retained <= MAX_SETUP_RETAINED_BYTES);
            preflight_setup_tier(SETUP_MAX_LOG_N).unwrap();

            let former_boundary = construction_commitment_scratch_bytes(&[(1 << 20) + 1]).unwrap();
            assert!(former_boundary > (1 << 21) * std::mem::size_of::<ArkFr>());
            assert!(former_boundary < MAX_DECODED_PROOF_BYTES);
            let structural_max = construction_commitment_scratch_bytes(&[MAX_RECORDS]).unwrap();
            assert!(structural_max > MAX_DECODED_PROOF_BYTES);
        }

        #[test]
        fn setup_identity_is_deterministic_and_legacy_tier_is_prefix_compatible() {
            use ark_ec::CurveGroup;

            assert_eq!(
                setup_parameter_id(19).unwrap(),
                setup_parameter_id(20).unwrap()
            );
            assert_ne!(
                setup_parameter_id(20).unwrap(),
                setup_parameter_id(21).unwrap()
            );
            assert_ne!(
                setup_parameter_id(22).unwrap(),
                setup_parameter_id(23).unwrap()
            );
            let selected = dory_prover_setup(20).unwrap();
            let independently_rebuilt = deterministic_prover_setup(20);
            assert_eq!(selected.g1_vec, independently_rebuilt.g1_vec);
            assert_eq!(selected.g2_vec, independently_rebuilt.g2_vec);
            assert!(selected
                .g1_vec
                .iter()
                .chain(std::iter::once(&selected.h1))
                .all(|point| point.0 != G1Projective::default()
                    && point.0.into_affine().is_on_curve()
                    && point
                        .0
                        .into_affine()
                        .is_in_correct_subgroup_assuming_on_curve()));
            assert!(selected
                .g2_vec
                .iter()
                .chain(std::iter::once(&selected.h2))
                .all(|point| point.0 != G2Projective::default()
                    && point.0.into_affine().is_on_curve()
                    && point
                        .0
                        .into_affine()
                        .is_in_correct_subgroup_assuming_on_curve()));

            let vector_hash = |g1: &ArkG1, g2: &ArkG2, h1: &ArkG1, h2: &ArkG2| {
                let mut bytes = Vec::new();
                g1.0.serialize_compressed(&mut bytes).unwrap();
                g2.0.serialize_compressed(&mut bytes).unwrap();
                h1.0.serialize_compressed(&mut bytes).unwrap();
                h2.0.serialize_compressed(&mut bytes).unwrap();
                hex::encode(Sha256::digest(bytes))
            };
            assert_eq!(
                vector_hash(
                    &selected.g1_vec[0],
                    &selected.g2_vec[0],
                    &selected.h1,
                    &selected.h2,
                ),
                "5dc71cb471c168203fe617d7e6fd66786cfce1ea50206c80aea5484079608b24"
            );
        }

        #[test]
        fn node_value_dictionary_commitments_align_with_shared_commit_column() {
            use crate::proof::argument::commit_column;
            use crate::proof::dict::hash_object;

            let layer = node_and_value_layer();
            assert_eq!(layer.node_value_catalog.len(), 3);
            // Rebuild the two padded dict columns exactly as the query side would
            // (via the public accessor), and confirm they match the stored
            // commitments byte-for-byte.
            let (hashes, ids) = layer.node_value_dictionary_columns().unwrap();
            assert_eq!(hashes.len(), 4); // 3 entries padded to a power of two
            for (k, record) in layer.node_value_catalog.iter().enumerate() {
                assert_eq!(hashes[k], hash_object(&record.triple.object));
                assert_eq!(ids[k], <ArkFr as DoryField>::from_u64(record.object_id));
            }
            assert_eq!(
                commit_column(&hashes).unwrap(),
                layer.node_value_dict_commitments.0
            );
            assert_eq!(
                commit_column(&ids).unwrap(),
                layer.node_value_dict_commitments.1
            );
        }

        #[test]
        fn commitment_root_binds_the_node_value_dictionary() {
            use crate::proof::argument::commit_column;

            let layer = sample();
            // Baseline: the recomputed root equals the stored root.
            assert_eq!(
                layer.recomputed_commitment_root().unwrap(),
                layer.state.commitment_root
            );
            // Tamper ONE node/value dict commitment component with a distinct but
            // otherwise valid GT: the recomputed root must diverge from the stored
            // root (the node/value dict is now bound into the trust anchor).
            let other = commit_column(&[
                <ArkFr as DoryField>::from_u64(7),
                <ArkFr as DoryField>::from_u64(9),
            ])
            .unwrap();
            assert_ne!(other, layer.node_value_dict_commitments.0);
            let mut tampered = layer.clone();
            tampered.node_value_dict_commitments.0 = other;
            assert_ne!(
                tampered.recomputed_commitment_root().unwrap(),
                tampered.state.commitment_root
            );
        }

        #[test]
        fn encode_decode_preserves_node_value_dictionary_and_root() {
            let commitment = node_and_value_layer();
            let bytes = commitment.encode().unwrap();
            assert_eq!(bytes.len(), commitment.encoded_len().unwrap());
            let decoded = LayerCommitment::decode(&bytes).unwrap();
            assert_eq!(
                decoded.node_value_dict_commitments,
                commitment.node_value_dict_commitments
            );
            assert_eq!(
                decoded.node_value_dict_range_commitments,
                commitment.node_value_dict_range_commitments
            );
            assert_eq!(
                decoded.predicate_dict_commitments,
                commitment.predicate_dict_commitments
            );
            // The recomputed root is stable across encode->decode and equals the
            // stored root on both sides (the new field survives the round trip).
            assert_eq!(
                decoded.recomputed_commitment_root().unwrap(),
                commitment.state.commitment_root
            );
            assert_eq!(
                decoded.recomputed_commitment_root().unwrap(),
                decoded.state.commitment_root
            );
        }

        #[test]
        fn node_value_range_columns_align_with_shared_commit_column() {
            use crate::proof::argument::commit_column;

            let layer = node_and_value_layer();
            // Rebuild the two range columns exactly as the query side would (via the
            // public accessors) and confirm they match the stored commitments
            // byte-for-byte, and share the length of the hash/id columns.
            let (hashes, _ids) = layer.node_value_dictionary_columns().unwrap();
            let values = layer.node_value_dictionary_value_column().unwrap();
            let datatypes = layer.node_value_dictionary_datatype_column().unwrap();
            assert_eq!(values.len(), hashes.len());
            assert_eq!(datatypes.len(), hashes.len());
            assert_eq!(
                commit_column(&values).unwrap(),
                layer.node_value_dict_range_commitments.0
            );
            assert_eq!(
                commit_column(&datatypes).unwrap(),
                layer.node_value_dict_range_commitments.1
            );
        }

        #[test]
        fn commitment_root_binds_the_node_value_value_and_datatype_columns() {
            use crate::proof::argument::commit_column;

            let layer = node_and_value_layer();
            assert_eq!(
                layer.recomputed_commitment_root().unwrap(),
                layer.state.commitment_root
            );
            let other = commit_column(&[
                <ArkFr as DoryField>::from_u64(11),
                <ArkFr as DoryField>::from_u64(13),
            ])
            .unwrap();
            // Tampering the VALUE commitment diverges the recomputed root...
            assert_ne!(other, layer.node_value_dict_range_commitments.0);
            let mut tampered_value = layer.clone();
            tampered_value.node_value_dict_range_commitments.0 = other;
            assert_ne!(
                tampered_value.recomputed_commitment_root().unwrap(),
                tampered_value.state.commitment_root
            );
            // ...and so does tampering the DATATYPE commitment.
            assert_ne!(other, layer.node_value_dict_range_commitments.1);
            let mut tampered_datatype = layer.clone();
            tampered_datatype.node_value_dict_range_commitments.1 = other;
            assert_ne!(
                tampered_datatype.recomputed_commitment_root().unwrap(),
                tampered_datatype.state.commitment_root
            );
        }

        // 255 must not collide with any real CanonicalDatatype discriminant.
        #[test]
        fn reserved_node_datatype_tag_is_not_a_real_discriminant() {
            use std::convert::TryFrom;
            assert_eq!(NODE_DATATYPE_TAG, 255);
            for tag in 0u8..=254 {
                // Every value below the reserved tag that IS a real discriminant is
                // strictly less than 255; TryFrom fails for the gaps. Either way,
                // no real discriminant equals 255.
                if let Ok(dt) = CanonicalDatatype::try_from(tag) {
                    assert_ne!(dt as u64, NODE_DATATYPE_TAG);
                }
            }
        }

        // Build a real base layer through the store carrying numeric value objects
        // and assert the committed VALUE column holds the expected big-endian
        // integers, order-preserving (monotone) within a datatype, with Node/String
        // entries getting the 0 sentinel and the reserved / String datatype tags.
        #[test]
        fn node_value_value_column_encodes_order_preserving_integers() {
            use crate::layer::ValueTriple;
            use crate::store::sync::open_sync_memory_store;
            use std::convert::TryInto;
            use tdb_succinct::TdbDataType;

            let iri = |s: &str| format!("http://ex/{s}");
            let p = iri("p");

            let store = open_sync_memory_store();
            let builder = store.create_base_layer().unwrap();
            // Three UInt32 values (inserted OUT of numeric order) + a Boolean + a
            // Node object + a String value.
            builder
                .add_value_triple(ValueTriple::new_value(
                    &iri("s1"),
                    &p,
                    u32::make_entry(&300u32),
                ))
                .unwrap();
            builder
                .add_value_triple(ValueTriple::new_value(
                    &iri("s2"),
                    &p,
                    u32::make_entry(&5u32),
                ))
                .unwrap();
            builder
                .add_value_triple(ValueTriple::new_value(
                    &iri("s3"),
                    &p,
                    u32::make_entry(&7u32),
                ))
                .unwrap();
            builder
                .add_value_triple(ValueTriple::new_value(
                    &iri("s4"),
                    &p,
                    bool::make_entry(&true),
                ))
                .unwrap();
            builder
                .add_value_triple(ValueTriple::new_node(&iri("s5"), &p, &iri("target")))
                .unwrap();
            builder
                .add_value_triple(ValueTriple::new_string_value(&iri("s6"), &p, "hello"))
                .unwrap();
            let layer = builder.commit().unwrap();
            let layer = layer.proof_commitment().unwrap();

            let values = layer.node_value_dictionary_value_column().unwrap();
            let datatypes = layer.node_value_dictionary_datatype_column().unwrap();
            assert_eq!(values.len(), datatypes.len());
            assert!(values.len() >= layer.node_value_catalog.len());

            let mut uint32_by_index: Vec<(usize, u64)> = Vec::new();
            let mut saw_bool = false;
            let mut saw_string = false;
            let mut saw_node = false;
            for (k, record) in layer.node_value_catalog.iter().enumerate() {
                match &record.triple.object {
                    CanonicalObject::Value {
                        datatype: CanonicalDatatype::UInt32,
                        lexical,
                    } => {
                        assert_eq!(lexical.len(), 4);
                        let expected = u32::from_be_bytes(lexical.as_slice().try_into().unwrap());
                        assert_eq!(values[k], <ArkFr as DoryField>::from_u64(expected as u64));
                        assert_eq!(
                            datatypes[k],
                            <ArkFr as DoryField>::from_u64(CanonicalDatatype::UInt32 as u64)
                        );
                        uint32_by_index.push((k, expected as u64));
                    }
                    CanonicalObject::Value {
                        datatype: CanonicalDatatype::Boolean,
                        lexical,
                    } => {
                        saw_bool = true;
                        // Boolean true -> lexical [1] -> value 1.
                        assert_eq!(lexical.as_slice(), &[1u8]);
                        assert_eq!(values[k], <ArkFr as DoryField>::from_u64(1));
                        assert_eq!(
                            datatypes[k],
                            <ArkFr as DoryField>::from_u64(CanonicalDatatype::Boolean as u64)
                        );
                    }
                    CanonicalObject::Value {
                        datatype: CanonicalDatatype::String,
                        ..
                    } => {
                        saw_string = true;
                        // Variable-width, non-OP: 0 sentinel, String datatype tag (0).
                        assert_eq!(values[k], <ArkFr as DoryField>::zero());
                        assert_eq!(
                            datatypes[k],
                            <ArkFr as DoryField>::from_u64(CanonicalDatatype::String as u64)
                        );
                    }
                    CanonicalObject::Node(_) => {
                        saw_node = true;
                        // Nodes (subjects and the node object): 0 sentinel, reserved tag.
                        assert_eq!(values[k], <ArkFr as DoryField>::zero());
                        assert_eq!(
                            datatypes[k],
                            <ArkFr as DoryField>::from_u64(NODE_DATATYPE_TAG)
                        );
                    }
                    other => panic!("unexpected catalog object {:?}", other),
                }
            }
            assert_eq!(uint32_by_index.len(), 3, "three distinct UInt32 values");
            assert!(saw_bool && saw_string && saw_node);
            // Order-preservation: the committed values 5,7,300 appear in ASCENDING
            // catalog order (the dictionary is sorted by the OP lexical bytes within
            // a datatype, so the committed column is monotone in the numeric value).
            let sorted: Vec<u64> = {
                let mut by_index = uint32_by_index.clone();
                by_index.sort_by_key(|(k, _)| *k);
                by_index.into_iter().map(|(_, v)| v).collect()
            };
            assert_eq!(sorted, vec![5u64, 7, 300]);

            // Padding rows (beyond the catalog) carry the 0 sentinel in both columns.
            for k in layer.node_value_catalog.len()..values.len() {
                assert_eq!(values[k], <ArkFr as DoryField>::zero());
                assert_eq!(datatypes[k], <ArkFr as DoryField>::zero());
            }
        }

        #[test]
        fn every_fixed_width_integer_family_has_exact_monotone_edge_encodings() {
            use crate::layer::ValueTriple;
            use dory_pcs::primitives::arithmetic::Field;
            use tdb_succinct::TdbDataType;

            let encoded = |entry| {
                let object = CanonicalObject::from_store_object(
                    &ValueTriple::new_value("", "", entry).object,
                );
                let datatype = match &object {
                    CanonicalObject::Value { datatype, .. } => *datatype,
                    CanonicalObject::Node(_) => unreachable!(),
                };
                let value = node_value_op_value(&object)
                    .unwrap_or_else(|error| panic!("{:?}: {}; object={:?}", datatype, error, object));
                (datatype, value)
            };

            let unsigned = [
                (encoded(u8::make_entry(&u8::MIN)), CanonicalDatatype::UInt8, 0),
                (encoded(u8::make_entry(&u8::MAX)), CanonicalDatatype::UInt8, u8::MAX as u64),
                (encoded(u16::make_entry(&u16::MIN)), CanonicalDatatype::UInt16, 0),
                (encoded(u16::make_entry(&u16::MAX)), CanonicalDatatype::UInt16, u16::MAX as u64),
                (encoded(u32::make_entry(&u32::MIN)), CanonicalDatatype::UInt32, 0),
                (encoded(u32::make_entry(&u32::MAX)), CanonicalDatatype::UInt32, u32::MAX as u64),
                (encoded(u64::make_entry(&u64::MIN)), CanonicalDatatype::UInt64, 0),
                (encoded(u64::make_entry(&u64::MAX)), CanonicalDatatype::UInt64, u64::MAX),
            ];
            for ((actual_datatype, actual), expected_datatype, expected) in unsigned {
                assert_eq!(actual_datatype, expected_datatype);
                assert_eq!(actual, ArkFr::from_u64(expected));
            }

            macro_rules! signed_edges {
                ($ty:ty, $datatype:expr, $sign:expr) => {{
                    let min = encoded(<$ty>::make_entry(&<$ty>::MIN));
                    let zero_value: $ty = 0;
                    let zero = encoded(<$ty>::make_entry(&zero_value));
                    let max = encoded(<$ty>::make_entry(&<$ty>::MAX));
                    assert_eq!(min.0, $datatype);
                    assert_eq!(zero.0, $datatype);
                    assert_eq!(max.0, $datatype);
                    assert_eq!(min.1, ArkFr::zero());
                    assert_eq!(zero.1, ArkFr::from_u64($sign));
                    assert_eq!(
                        max.1,
                        ArkFr::from_u64($sign.wrapping_mul(2).wrapping_sub(1)),
                    );
                }};
            }
            signed_edges!(i8, CanonicalDatatype::Int8, 1u64 << 7);
            signed_edges!(i16, CanonicalDatatype::Int16, 1u64 << 15);
            signed_edges!(i32, CanonicalDatatype::Int32, 1u64 << 31);
            signed_edges!(i64, CanonicalDatatype::Int64, 1u64 << 63);
        }

        #[test]
        fn date_and_datetime_op_values_are_saturating_i64_nanos_offset_binary() {
            use crate::layer::ValueTriple;
            use chrono::{DateTime, NaiveDate, NaiveDateTime, Utc};
            use dory_pcs::primitives::arithmetic::Field;
            use tdb_succinct::{Date, TdbDataType};

            // The OP value the store commits for a typed entry.
            let op = |entry| {
                let object = CanonicalObject::from_store_object(
                    &ValueTriple::new_value("", "", entry).object,
                );
                node_value_op_value(&object).expect("op value")
            };
            // INDEPENDENT offset-binary code for a known i64 key (mirrors the Int64
            // convention `(n as u64) ^ (1<<63)`), computed WITHOUT the private helper so
            // this test is a genuine oracle for `node_value_op_value`.
            let code = |n: i64| (n as u64) ^ (1u64 << 63);
            let field = |n: i64| ArkFr::from_u64(code(n));
            let utc = |secs: i64, nanos: u32| -> NaiveDateTime {
                DateTime::<Utc>::from_timestamp(secs, nanos).unwrap().naive_utc()
            };

            // --- xsd:dateTime: chronological UTC instant, to the nanosecond ---
            let dt_samples: Vec<(NaiveDateTime, i64)> = vec![
                (utc(0, 0), 0),                                         // the epoch
                (utc(0, 1), 1),                                         // +1 ns
                (utc(-1, 0), -1_000_000_000),                          // 1 s before epoch
                (utc(1_000_000_000, 123), 1_000_000_000 * 1_000_000_000 + 123),
            ];
            for (ndt, expected_nanos) in &dt_samples {
                assert_eq!(
                    op(NaiveDateTime::make_entry(ndt)),
                    field(*expected_nanos),
                    "dateTime OP == offset-binary(i64-nanos) for {ndt}",
                );
            }
            // Order-preserving: sorting by the key strictly increases the offset-binary code
            // (so a < b ⇒ OP(a) < OP(b)) — including ACROSS the epoch (negative keys).
            let mut keys: Vec<i64> = dt_samples.iter().map(|(_, n)| *n).collect();
            keys.sort_unstable();
            for w in keys.windows(2) {
                assert!(code(w[0]) < code(w[1]), "offset-binary is monotone in the key");
            }

            // dateTime SATURATION: outside the i64-nanoseconds window saturates by side.
            let far_future = NaiveDate::from_ymd_opt(5000, 1, 1)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap();
            let far_past = NaiveDate::from_ymd_opt(1000, 1, 1)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap();
            assert_eq!(op(NaiveDateTime::make_entry(&far_future)), field(i64::MAX));
            assert_eq!(op(NaiveDateTime::make_entry(&far_past)), field(i64::MIN));

            // --- xsd:date: (year, month, day, offset)-lexical via midnight_utc_nanos + offset ---
            let date_entry = |y: i64, m: u8, d: u8, off: i16| {
                Date::make_entry(&Date {
                    year: y,
                    month: m,
                    day: d,
                    offset: off,
                })
            };
            let midnight = |y: i32, m: u32, d: u32| -> i64 {
                NaiveDate::from_ymd_opt(y, m, d)
                    .unwrap()
                    .and_hms_opt(0, 0, 0)
                    .unwrap()
                    .and_utc()
                    .timestamp_nanos_opt()
                    .unwrap()
            };
            let n_2020 = midnight(2020, 1, 1);
            // offset 0 (the compiler's constant form) is a pure midnight instant.
            assert_eq!(op(date_entry(2020, 1, 1, 0)), field(n_2020));
            // The timezone offset is a LEAST-significant tiebreaker: a larger offset sorts
            // later, a negative one earlier, matching WOQL's `compare/3` over `date/4`.
            assert_eq!(op(date_entry(2020, 1, 1, 60)), field(n_2020 + 60));
            assert_eq!(op(date_entry(2020, 1, 1, -60)), field(n_2020 - 60));
            // The offset can NEVER cross a day boundary (a day is ~8.64e13 ns, |offset|≤32767):
            // 2019-12-31 < any-offset 2020-01-01 < 2020-01-02.
            let n_dec31 = midnight(2019, 12, 31);
            let n_jan02 = midnight(2020, 1, 2);
            assert!(code(n_dec31) < code(n_2020 - 60), "prior day < offset-shifted date");
            assert!(code(n_2020 + 60) < code(n_jan02), "offset-shifted date < next day");
            // date SATURATION: a year outside the window saturates, and the offset is pinned
            // at the boundary so it can never nudge a far date back toward the constant.
            assert_eq!(op(date_entry(5000, 1, 1, 120)), field(i64::MAX));
            assert_eq!(op(date_entry(1000, 1, 1, -120)), field(i64::MIN));
        }

        #[test]
        fn adversarial_date_and_datetime_lexicals_fail_closed_without_panicking() {
            use chrono::{DateTime, NaiveDate, NaiveDateTime, Utc};
            use dory_pcs::primitives::arithmetic::Field;
            use std::panic::{catch_unwind, AssertUnwindSafe};
            use tdb_succinct::{Date, TdbDataType};

            // `node_value_op_value` runs inside `VerifierCommitmentRef::validate()` over
            // adversary-supplied catalog rows on a trustless proof, so a panic on a crafted
            // date/dateTime lexical is a verifier DoS. Every crafted lexical below must
            // either fail closed (`Err`) or, for a WELL-FORMED-but-out-of-window instant,
            // saturate (`Ok`) — but it must NEVER panic. `catch_unwind` turns any residual
            // panic into a clean test failure. NB the ORIGINAL native decoders panicked on
            // several of these (the exact DoS this test pins shut).
            // Decode UNDER a panic guard: any panic is caught and re-raised as a clear
            // test failure (the DoS we are pinning shut), leaving a plain `io::Result` the
            // assertions can inspect and Debug-format.
            let decode = |datatype: CanonicalDatatype, lexical: Vec<u8>| -> io::Result<ArkFr> {
                catch_unwind(AssertUnwindSafe(|| {
                    node_value_op_value(&CanonicalObject::Value { datatype, lexical })
                }))
                .unwrap_or_else(|_| {
                    panic!("node_value_op_value PANICKED on an adversarial lexical (verifier DoS)")
                })
            };
            // Offset-binary code of an i64 key (the shared saturation boundary encoding).
            let field = |n: i64| ArkFr::from_u64((n as u64) ^ (1u64 << 63));

            // --- (a) xsd:dateTime with bigint-seconds beyond i64 -----------------------
            // Size prefix `0x8C` = positive, 12 magnitude bytes; twelve `0xFF` magnitude
            // bytes (≫ i64::MAX); a `0x00` "false zero" fraction. WELL-FORMED but far
            // outside the i64-nanoseconds window ⇒ SATURATES to i64::MAX (never panics; the
            // native `int.to_i64().expect(...)` would have panicked here).
            let mut dt_huge_pos = vec![0x8Cu8];
            dt_huge_pos.extend(std::iter::repeat(0xFFu8).take(12));
            dt_huge_pos.push(0x00);
            let a = decode(CanonicalDatatype::DateTime, dt_huge_pos);
            assert!(
                matches!(&a, Ok(v) if *v == field(i64::MAX)),
                "over-i64 dateTime saturates to i64::MAX, got {a:?}",
            );
            // Same magnitude, NEGATIVE sign (`0x73` = !0x8C): saturates to i64::MIN.
            let mut dt_huge_neg = vec![0x73u8];
            dt_huge_neg.extend(std::iter::repeat(0x00u8).take(12)); // !0xFF under negative sign
            dt_huge_neg.push(0xFF); // !0x00 fraction "false zero" under negative sign
            let a_neg = decode(CanonicalDatatype::DateTime, dt_huge_neg);
            assert!(
                matches!(&a_neg, Ok(v) if *v == field(i64::MIN)),
                "under-i64 dateTime saturates to i64::MIN, got {a_neg:?}",
            );
            // (a') the SAME huge magnitude but TRUNCATED (missing the fraction byte): an
            // undecodable/malformed lexical ⇒ fails closed with Err (no panic).
            let mut dt_huge_trunc = vec![0x8Cu8];
            dt_huge_trunc.extend(std::iter::repeat(0xFFu8).take(12));
            let a_trunc = decode(CanonicalDatatype::DateTime, dt_huge_trunc);
            assert!(a_trunc.is_err(), "truncated dateTime must Err, got {a_trunc:?}");
            // (a'') size prefix claims 8 magnitude bytes but only 2 are present ⇒ Err.
            let a_short = decode(CanonicalDatatype::DateTime, vec![0x88, 0x01, 0x02]);
            assert!(a_short.is_err(), "short-magnitude dateTime must Err, got {a_short:?}");
            // (a''') empty lexical (no size prefix at all) ⇒ Err.
            let a_empty = decode(CanonicalDatatype::DateTime, Vec::new());
            assert!(a_empty.is_err(), "empty dateTime must Err, got {a_empty:?}");

            // --- (b) truncated xsd:date (fewer than 12 bytes) --------------------------
            let b = decode(CanonicalDatatype::Date, vec![0x00, 0x01, 0x02]);
            assert!(b.is_err(), "truncated date must Err, got {b:?}");

            // --- (c) over-long xsd:date (13 bytes) -------------------------------------
            let c = decode(CanonicalDatatype::Date, vec![0u8; 13]);
            assert!(c.is_err(), "over-long date must Err, got {c:?}");

            // Valid store-produced date/dateTime lexicals still decode to the SAME OP
            // values (ordering + offset tiebreaker unchanged) — proving the defensive
            // decode did not perturb the honest path.
            let op = |entry| {
                node_value_op_value(&CanonicalObject::from_store_object(
                    &crate::layer::ValueTriple::new_value("", "", entry).object,
                ))
                .expect("valid op value")
            };
            let ndt = |secs: i64, nanos: u32| -> NaiveDateTime {
                DateTime::<Utc>::from_timestamp(secs, nanos).unwrap().naive_utc()
            };
            // dateTime: epoch, +1ns, 1s-before-epoch, and a large-nanos instant.
            assert_eq!(op(NaiveDateTime::make_entry(&ndt(0, 0))), field(0));
            assert_eq!(op(NaiveDateTime::make_entry(&ndt(0, 1))), field(1));
            assert_eq!(op(NaiveDateTime::make_entry(&ndt(-1, 0))), field(-1_000_000_000));
            assert_eq!(
                op(NaiveDateTime::make_entry(&ndt(1_000_000_000, 123))),
                field(1_000_000_000 * 1_000_000_000 + 123),
            );
            // date: midnight instant + offset tiebreaker, byte-identical to before.
            let n_2020 = NaiveDate::from_ymd_opt(2020, 1, 1)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap()
                .and_utc()
                .timestamp_nanos_opt()
                .unwrap();
            let dmk = |y: i64, m: u8, d: u8, off: i16| {
                Date::make_entry(&Date { year: y, month: m, day: d, offset: off })
            };
            assert_eq!(op(dmk(2020, 1, 1, 0)), field(n_2020));
            assert_eq!(op(dmk(2020, 1, 1, 60)), field(n_2020 + 60));
            assert_eq!(op(dmk(2020, 1, 1, -60)), field(n_2020 - 60));
        }

        #[test]
        fn non_date_entries_keep_the_zero_sentinel_byte_identically() {
            // The date OP change must NOT perturb any non-date entry: nodes, strings, and
            // every temporal type OTHER than date/dateTime (Time, DateTimeStamp — deliberately
            // reject-closed) must still map to the `0` sentinel, so a layer WITHOUT dates
            // commits a byte-identical node/value VALUE column (and thus an identical
            // `commitment_root`) under the format bump.
            use crate::layer::ValueTriple;
            use chrono::{DateTime, NaiveTime, Utc};
            use dory_pcs::primitives::arithmetic::Field;
            use tdb_succinct::TdbDataType;

            let op = |object: CanonicalObject| node_value_op_value(&object).expect("op value");
            let value_op = |entry| {
                op(CanonicalObject::from_store_object(
                    &ValueTriple::new_value("", "", entry).object,
                ))
            };
            let zero = <ArkFr as Field>::zero();

            // A Node object.
            assert_eq!(op(CanonicalObject::Node(b"http://ex/x".to_vec())), zero);
            // xsd:string.
            assert_eq!(value_op(String::make_entry(&"hello".to_string())), zero);
            // xsd:time — a temporal type that is NOT range-supported: stays sentinel.
            let time = NaiveTime::from_hms_opt(12, 0, 0).unwrap();
            assert_eq!(value_op(NaiveTime::make_entry(&time)), zero);
            // xsd:dateTimeStamp — stores like dateTime but is deliberately reject-closed
            // (the WOQL AST cannot name it); it must remain a sentinel so a layer holding one
            // (but no date/dateTime) stays byte-identical.
            let stamp = tdb_succinct::DateTimeStamp(
                DateTime::<Utc>::from_timestamp(1_600_000_000, 0)
                    .unwrap()
                    .naive_utc(),
            );
            assert_eq!(value_op(tdb_succinct::DateTimeStamp::make_entry(&stamp)), zero);
        }

        #[test]
        fn node_value_dictionary_opens_for_node_and_value_entries() {
            use crate::proof::argument::{commit_column, Transcript254};
            use crate::proof::dict::{hash_object, prove_dict_membership, verify_dict_membership};

            let layer = node_and_value_layer();
            let (dict_hash, dict_id) = layer.node_value_dictionary_columns().unwrap();
            // The committed columns are exactly the ones a membership proof opens.
            assert_eq!(
                commit_column(&dict_hash).unwrap(),
                layer.node_value_dict_commitments.0
            );
            assert_eq!(
                commit_column(&dict_id).unwrap(),
                layer.node_value_dict_commitments.1
            );
            let com_h = layer.node_value_dict_commitments.0;
            let com_i = layer.node_value_dict_commitments.1;

            // A prove+verify round for one `(hash_object(term), id)` pair.
            let open = |q_hash: ArkFr, id: ArkFr| -> io::Result<()> {
                let mut pt = Transcript254::new(b"nv-dict-open");
                let proof = prove_dict_membership(
                    &dict_hash, &dict_id, &com_h, &com_i, q_hash, id, &mut pt,
                )?;
                let mut vt = Transcript254::new(b"nv-dict-open");
                verify_dict_membership(&com_h, &com_i, q_hash, id, &proof, &mut vt)
            };

            // Resolve ids from the catalog for BOTH a Node object and a Value object.
            let node_obj = CanonicalObject::Node(b"b".to_vec());
            let value_obj = CanonicalObject::Value {
                datatype: CanonicalDatatype::String,
                lexical: b"lit".to_vec(),
            };
            let id_of = |obj: &CanonicalObject| -> u64 {
                layer
                    .node_value_catalog
                    .iter()
                    .find(|r| &r.triple.object == obj)
                    .expect("object present in catalog")
                    .object_id
            };

            for obj in [&node_obj, &value_obj] {
                let id = id_of(obj);
                let q_hash = hash_object(obj);
                let id_fr = <ArkFr as DoryField>::from_u64(id);
                // Honest opening verifies.
                assert!(open(q_hash, id_fr).is_ok());
                // A WRONG id for a real term is rejected (the pair is bound).
                assert!(open(q_hash, <ArkFr as DoryField>::from_u64(id + 100)).is_err());
            }

            // An ABSENT term is rejected.
            let absent = hash_object(&CanonicalObject::Node(b"not-in-dictionary".to_vec()));
            assert!(open(absent, <ArkFr as DoryField>::from_u64(2)).is_err());
        }

        #[test]
        fn transparent_setup_is_deterministic_across_fresh_derivations() {
            // Two independently-derived setups must produce byte-identical public
            // parameters, so any node commits a given column to the same value.
            let setup_a = deterministic_prover_setup(4);
            let setup_b = deterministic_prover_setup(4);
            assert_eq!(setup_a.g1_vec, setup_b.g1_vec);
            assert_eq!(setup_a.g2_vec, setup_b.g2_vec);

            let coefficients: Vec<ArkFr> = (1..=4u64).map(<ArkFr as DoryField>::from_u64).collect();
            let poly = ArkworksPolynomial::new(coefficients);
            let (commitment_a, _, _) = poly
                .commit::<BN254, Transparent, G1Routines>(1, 1, &setup_a)
                .unwrap();
            let (commitment_b, _, _) = poly
                .commit::<BN254, Transparent, G1Routines>(1, 1, &setup_b)
                .unwrap();
            assert_eq!(commitment_a, commitment_b);
        }

        #[test]
        fn commitment_root_is_stable_across_independent_calls() {
            // End-to-end determinism: the same delta yields the same root.
            assert_eq!(
                sample().state.commitment_root,
                sample().state.commitment_root
            );
            assert_ne!(sample().state.commitment_root, [0u8; 32]);
        }

        #[test]
        fn changing_one_record_id_changes_the_root() {
            // a-p-b (subject 1, object 2) vs b-p-a (subject 2, object 1): identical
            // dictionary/counts, but the id columns differ, so the commitments and
            // hence the root must differ.
            let a_p_b = delta(vec![record(1, 1, 2, "a", "p", "b")], vec![]);
            let b_p_a = delta(vec![record(2, 1, 1, "b", "p", "a")], vec![]);
            assert_ne!(a_p_b.state.commitment_root, b_p_a.state.commitment_root);
        }

        #[test]
        fn empty_adds_only_removes_only_are_valid_and_distinct() {
            let empty = LayerCommitment::from_native_delta(
                None,
                vec![],
                vec![],
                vec![],
                0,
                0,
                0,
                0,
                vec![],
                vec![],
            )
            .unwrap();
            let adds_only = delta(vec![record(1, 1, 2, "a", "p", "b")], vec![]);
            let removes_only = delta(vec![], vec![record(1, 1, 2, "a", "p", "b")]);

            // Empty columns use the canonical GT-identity commitment.
            let identity = <ArkGT as Group>::identity();
            assert!(empty.column_commitments.iter().all(|c| *c == identity));
            // Adds-only: remove columns are empty; removes-only: add columns empty.
            assert!(adds_only.column_commitments[3..]
                .iter()
                .all(|c| *c == identity));
            assert!(removes_only.column_commitments[..3]
                .iter()
                .all(|c| *c == identity));

            let roots = [
                empty.state.commitment_root,
                adds_only.state.commitment_root,
                removes_only.state.commitment_root,
            ];
            assert_ne!(roots[0], roots[1]);
            assert_ne!(roots[0], roots[2]);
            assert_ne!(roots[1], roots[2]);
        }

        #[test]
        fn tampered_body_is_rejected() {
            let commitment = sample();
            let mut bytes = commitment.encode().unwrap();
            let mid = bytes.len() / 2;
            bytes[mid] ^= 1;
            assert!(LayerCommitment::decode(&bytes).is_err());
        }

        #[test]
        fn tampered_materialized_view_is_rejected() {
            let mut commitment = sample();
            commitment.materialized[0].object_id = 1;
            commitment.materialized[0].triple.object = CanonicalObject::Node(b"a".to_vec());
            assert!(commitment.validate().is_err());
        }

        #[test]
        fn tampered_full_catalog_is_rejected() {
            let mut commitment = sample();
            commitment.node_value_catalog[1].triple.object =
                CanonicalObject::Node(b"tampered".to_vec());
            assert!(commitment.validate().is_err());
        }

        #[test]
        fn tampered_serialized_commitment_is_rejected_even_with_fixed_checksum() {
            let commitment = sample();
            let mut bytes = commitment.encode().unwrap();
            // The serialized commitments sit immediately before the trailing 32-byte
            // commitment_root and 32-byte file checksum. Flip a byte inside the last
            // commitment blob, then repair the file checksum so the rejection is
            // driven by the commitment_root binding, not the outer checksum.
            let commitment_byte = bytes.len() - 64 - 1;
            bytes[commitment_byte] ^= 1;
            let payload_len = bytes.len() - 32;
            let repaired = file_checksum(&bytes[..payload_len]);
            bytes[payload_len..].copy_from_slice(&repaired);
            assert!(LayerCommitment::decode(&bytes).is_err());
        }

        #[test]
        fn catalog_gap_fails_closed() {
            assert!(LayerCommitment::from_native_delta(
                None,
                vec![],
                vec![],
                vec![],
                0,
                0,
                3,
                0,
                vec![
                    (CanonicalObject::Node(b"a".to_vec()), 1),
                    (CanonicalObject::Node(b"c".to_vec()), 3),
                ],
                vec![],
            )
            .is_err());
        }
    }
}

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
