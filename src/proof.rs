//! Reusable dictionary-encoded column/row model for authenticated Store deltas.
//!
//! This module keeps the scheme-independent pieces of the proof pipeline: the
//! canonical datatype/object/triple model, the dictionary-encoded `ProofRecord`
//! column tables, the query pattern model, deterministic per-index ordering, and
//! the framing/resource-limit helpers used to (de)serialize sidecars.
//!
//! The per-layer sidecar payload itself lives in [`commitment`], which replaces
//! the abandoned SHA-256 Merkle scheme with a homomorphic column-commitment
//! scheme ("Proof-of-WOQL"). The commitment logic is presently a STUB; see the
//! `TODO(proof-of-woql)` markers.

use std::cmp::Ordering;
use std::convert::{TryFrom, TryInto};
use std::io;

use crate::layer::{ObjectType, ValueTriple};
use tdb_succinct::Datatype;

/// 32-byte sidecar root type. Retained as an opaque root/anchor identifier used
/// by the storage plumbing (expected-roots maps, commitment roots).
pub type ProofHash = [u8; 32];

const MAX_RECORDS: usize = 10_000_000;
const MAX_DECODED_RECORDS: usize = 4_000_000;
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

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ResolvedPattern {
    pub subject: Option<u64>,
    pub predicate: Option<u64>,
    pub object: Option<u64>,
    pub value_range: Option<(u64, u64)>,
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

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TriplePattern {
    pub subject: Option<Vec<u8>>,
    pub predicate: Option<Vec<u8>>,
    pub object: Option<CanonicalObject>,
    pub value_range: Option<CanonicalValueRange>,
    pub reverse: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanonicalValueRange {
    pub datatype: CanonicalDatatype,
    pub low: Vec<u8>,
    pub high: Vec<u8>,
}

impl CanonicalValueRange {
    pub fn from_store_bounds(
        low: &tdb_succinct::TypedDictEntry,
        high: &tdb_succinct::TypedDictEntry,
    ) -> io::Result<Self> {
        if low.datatype() != high.datatype() {
            return Err(invalid("typed value range bounds have different datatypes"));
        }
        let range = Self {
            datatype: CanonicalDatatype::from_store(low.datatype()),
            low: low.to_bytes().to_vec(),
            high: high.to_bytes().to_vec(),
        };
        range.validate()?;
        Ok(range)
    }

    fn validate(&self) -> io::Result<()> {
        // These are the only Store lexical encodings whose byte ordering is
        // currently covered by differential tests against native range scans.
        // Other datatypes remain explicit in exact-value proofs, but ranges fail
        // closed until their ordering is independently established.
        if !matches!(
            self.datatype,
            CanonicalDatatype::String | CanonicalDatatype::UInt32
        ) {
            return Err(invalid(
                "datatype is not supported for authenticated ranges",
            ));
        }
        if self.low >= self.high {
            return Err(invalid("typed value range must be non-empty and half-open"));
        }
        Ok(())
    }

    fn classify(&self, object: &CanonicalObject) -> Ordering {
        match object {
            CanonicalObject::Node(_) => Ordering::Less,
            CanonicalObject::Value { datatype, lexical } => match datatype.cmp(&self.datatype) {
                Ordering::Equal if lexical < &self.low => Ordering::Less,
                Ordering::Equal if lexical >= &self.high => Ordering::Greater,
                Ordering::Equal => Ordering::Equal,
                other => other,
            },
        }
    }
}

impl TriplePattern {
    // Retained scheme-independent query-pattern logic. `validate`/`carrier_cmp`
    // are currently unused because the Merkle query engine was removed; the
    // Proof-of-WOQL query argument (a sumcheck over committed columns) will
    // reuse them. See TODO(proof-of-woql) in `proof::commitment`.
    #[allow(dead_code)]
    fn validate(&self) -> io::Result<()> {
        if self.reverse && self.value_range.is_none() {
            return Err(invalid(
                "reverse scans are supported only for authenticated value ranges",
            ));
        }
        if self.object.is_some() && self.value_range.is_some() {
            return Err(invalid(
                "exact object and value range are mutually exclusive",
            ));
        }
        if let Some(range) = &self.value_range {
            range.validate()?;
        }
        Ok(())
    }

    pub fn index(&self) -> ProofIndex {
        if self.value_range.is_some() {
            ProofIndex::Value
        } else if self.subject.is_some() {
            ProofIndex::Spo
        } else if self.object.is_some() || self.value_range.is_some() {
            ProofIndex::Osp
        } else if self.predicate.is_some() {
            ProofIndex::Pos
        } else {
            ProofIndex::Spo
        }
    }

    pub fn matches(&self, triple: &CanonicalTriple) -> bool {
        self.subject.as_ref().map_or(true, |v| v == &triple.subject)
            && self
                .predicate
                .as_ref()
                .map_or(true, |v| v == &triple.predicate)
            && self.object.as_ref().map_or(true, |v| v == &triple.object)
            && self.value_range.as_ref().map_or(true, |range| {
                range.classify(&triple.object) == Ordering::Equal
            })
    }

    #[allow(dead_code)]
    fn carrier_cmp(
        &self,
        record: &ProofRecord,
        resolved: &ResolvedPattern,
        index: ProofIndex,
    ) -> Ordering {
        let cmp = |actual: u64, wanted: u64| actual.cmp(&wanted);
        let fields: Vec<Ordering> = match index {
            ProofIndex::Spo => {
                let mut v = Vec::new();
                if let Some(s) = resolved.subject {
                    v.push(cmp(record.subject_id, s));
                    if let Some(p) = resolved.predicate {
                        v.push(cmp(record.predicate_id, p));
                        if let Some(o) = resolved.object {
                            v.push(cmp(record.object_id, o));
                        }
                    }
                }
                v
            }
            ProofIndex::Pos => {
                let mut v = Vec::new();
                if let Some(p) = resolved.predicate {
                    v.push(cmp(record.predicate_id, p));
                    if let Some(o) = resolved.object {
                        v.push(cmp(record.object_id, o));
                    }
                }
                v
            }
            ProofIndex::Osp => {
                let mut v = Vec::new();
                if let Some(object) = resolved.object {
                    v.push(cmp(record.object_id, object));
                }
                v
            }
            ProofIndex::Value => {
                let mut v = Vec::new();
                if let Some(range) = &self.value_range {
                    v.push(range.classify(&record.triple.object));
                }
                v
            }
            ProofIndex::SubjectDictionary
            | ProofIndex::PredicateDictionary
            | ProofIndex::ObjectDictionary => Vec::new(),
        };
        fields
            .into_iter()
            .find(|o| *o != Ordering::Equal)
            .unwrap_or(Ordering::Equal)
    }
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
    predicates: &[ProofRecord],
    objects: &[ProofRecord],
) -> io::Result<()> {
    let mut total = 0usize;
    for records in [additions, removals, predicates, objects] {
        checked_heap_add(&mut total, record_table_heap(records)?)?;
    }
    for records in [additions, removals] {
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
    )
}

fn validate_construction_heap_budget(
    additions: &[ProofRecord],
    removals: &[ProofRecord],
    node_values: &[(CanonicalObject, u64)],
    predicates: &[(Vec<u8>, u64)],
) -> io::Result<()> {
    let mut total = 0usize;
    for records in [additions, removals] {
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
    )
}

pub use commitment::{CommitmentState, LayerCommitment, QueryProof};

pub mod argument;
pub mod filter;

/// Per-layer sidecar payload for the Proof-of-WOQL commitment scheme.
///
/// This module REPLACES the abandoned SHA-256 Merkle sidecar (`ProofLayer` /
/// `ProofLayerState`). It exposes the same shape the storage plumbing depends
/// on: a validated per-layer payload carrying a small [`CommitmentState`]
/// (parent root + counts + a single root hash) plus the dictionary-encoded
/// column tables that Phase 1 will commit over.
///
/// The commitment itself is a STUB: `commitment_root` is currently a
/// domain-separated placeholder digest over the encoded body. See the
/// `TODO(proof-of-woql)` markers.
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
    use dory_pcs::backends::arkworks::{
        ArkFr, ArkG1, ArkG2, ArkGT, ArkworksPolynomial, G1Routines, BN254,
    };
    use dory_pcs::primitives::arithmetic::{Field as DoryField, Group, PairingCurve};
    use dory_pcs::primitives::poly::Polynomial;
    use dory_pcs::setup::{ProverSetup, VerifierSetup};
    use dory_pcs::Transparent;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    const COMMITMENT_MAGIC: &[u8; 8] = b"TSPCW0\0\0";
    pub const COMMITMENT_FORMAT_VERSION: u32 = 1;

    /// Number of committed columns per layer: additions (S,P,O) then removals
    /// (S,P,O), in that fixed order.
    pub const COLUMN_COUNT: usize = 6;

    /// Fixed size of the global transparent Dory setup. `max_log_n = 20` supports
    /// polynomials of up to `2^20` coefficients, i.e. a padded per-column length of
    /// up to ~1,048,576 dictionary ids. A single-layer delta whose additions or
    /// removals exceed that count fails closed until the global setup is enlarged
    /// (Phase 1b cap; cross-layer composition + larger setups are later phases).
    pub const SETUP_MAX_LOG_N: usize = 20;

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
    /// then sample the G1/G2 generators via arkworks' `UniformRand`. The discrete
    /// logs of the resulting points remain unknown (the seed only fixes *which*
    /// random points are chosen, not their DL relations), so this is a valid
    /// nothing-up-my-sleeve transparent setup.
    // TODO(proof-of-woql): a production setup should use a standardized
    // hash-to-curve derivation; a seeded CsPRNG is sufficient for this phase.
    fn deterministic_prover_setup(max_log_n: usize) -> ProverSetup<BN254> {
        use ark_ff::UniformRand;
        let mut rng = ChaCha20Rng::from_seed(setup_seed());
        let n = 1usize << max_log_n.div_ceil(2);
        let g1_vec: Vec<ArkG1> = (0..n).map(|_| ArkG1(G1Projective::rand(&mut rng))).collect();
        let g2_vec: Vec<ArkG2> = (0..n).map(|_| ArkG2(G2Projective::rand(&mut rng))).collect();
        let h1 = ArkG1(G1Projective::rand(&mut rng));
        let h2 = ArkG2(G2Projective::rand(&mut rng));
        let ht = <BN254 as PairingCurve>::pair(&h1, &h2);
        ProverSetup {
            g1_vec,
            g2_vec,
            h1,
            h2,
            ht,
        }
    }

    /// One global, lazily-derived, deterministic transparent prover setup.
    fn dory_prover_setup() -> &'static ProverSetup<BN254> {
        static SETUP: OnceLock<ProverSetup<BN254>> = OnceLock::new();
        SETUP.get_or_init(|| deterministic_prover_setup(SETUP_MAX_LOG_N))
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

    /// The global deterministic prover setup, for the query argument's openings.
    pub(crate) fn prover_setup() -> &'static ProverSetup<BN254> {
        dory_prover_setup()
    }

    /// The verifier setup derived once from the global prover setup.
    pub(crate) fn verifier_setup() -> &'static VerifierSetup<BN254> {
        static VERIFIER_SETUP: OnceLock<VerifierSetup<BN254>> = OnceLock::new();
        VERIFIER_SETUP.get_or_init(|| dory_prover_setup().to_verifier_setup())
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
    fn commit_id_column(ids: &[u64]) -> io::Result<ArkGT> {
        if ids.is_empty() {
            return Ok(<ArkGT as Group>::identity());
        }
        let padded_len = ids
            .len()
            .checked_next_power_of_two()
            .ok_or_else(|| invalid("proof column length overflow"))?;
        let m = padded_len.trailing_zeros() as usize;
        if m > SETUP_MAX_LOG_N {
            return Err(invalid(
                "proof column exceeds the global Dory setup capacity",
            ));
        }
        let (nu, sigma) = matrix_dims(m);
        let mut coefficients: Vec<ArkFr> = Vec::with_capacity(padded_len);
        for id in ids {
            coefficients.push(<ArkFr as DoryField>::from_u64(*id));
        }
        coefficients.resize(padded_len, <ArkFr as DoryField>::zero());
        let poly = ArkworksPolynomial::new(coefficients);
        let (tier_2, _row_commitments, _blind) = poly
            .commit::<BN254, Transparent, G1Routines>(nu, sigma, dory_prover_setup())
            .map_err(|error| invalid(format!("dory column commitment failed: {error}")))?;
        Ok(tier_2)
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
        let subjects = |table: &[ProofRecord]| table.iter().map(|r| r.subject_id).collect::<Vec<_>>();
        let predicates =
            |table: &[ProofRecord]| table.iter().map(|r| r.predicate_id).collect::<Vec<_>>();
        let objects = |table: &[ProofRecord]| table.iter().map(|r| r.object_id).collect::<Vec<_>>();
        Ok([
            commit_id_column(&subjects(additions))?,
            commit_id_column(&predicates(additions))?,
            commit_id_column(&objects(additions))?,
            commit_id_column(&subjects(removals))?,
            commit_id_column(&predicates(removals))?,
            commit_id_column(&objects(removals))?,
        ])
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

    /// Placeholder for the query-time argument produced by the Proof-of-WOQL
    /// scheme once query witnesses are implemented.
    // TODO(proof-of-woql): carry the sumcheck/opening argument over the
    // committed columns instead of an empty marker.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct QueryProof;

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
        pub parent_node_value_count: u64,
        pub parent_predicate_count: u64,
        pub node_value_count: u64,
        pub predicate_count: u64,
        // TODO(proof-of-woql): replace this placeholder digest with the real
        // homomorphic column-commitment root (Dory) over the committed columns.
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
        pub predicate_catalog: Vec<ProofRecord>,
        pub node_value_catalog: Vec<ProofRecord>,
        /// Per-column Dory (tier-2 GT) commitments over the id columns, in fixed
        /// order: additions (S,P,O) then removals (S,P,O). `commitment_root`
        /// binds these. Phase 1b commits each column standalone at offset 0.
        pub column_commitments: [ArkGT; COLUMN_COUNT],
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
        /// Build a production sidecar from the complete native dictionary delta.
        /// Replaces `ProofLayer::new_native_with_catalog`. The supplied mappings
        /// contain only ids introduced by this layer; inherited mappings are
        /// authenticated by the parent commitment state.
        #[allow(clippy::too_many_arguments)]
        pub fn from_native_delta(
            parent_state_root: Option<ProofHash>,
            mut additions: Vec<ProofRecord>,
            mut removals: Vec<ProofRecord>,
            parent_node_value_count: u64,
            parent_predicate_count: u64,
            node_value_count: u64,
            predicate_count: u64,
            node_value_mappings: Vec<(CanonicalObject, u64)>,
            predicate_mappings: Vec<(Vec<u8>, u64)>,
        ) -> io::Result<Self> {
            validate_construction_heap_budget(
                &additions,
                &removals,
                &node_value_mappings,
                &predicate_mappings,
            )?;
            if additions.len() > MAX_RECORDS || removals.len() > MAX_RECORDS {
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
                &[
                    subject_catalog,
                    predicate_catalog.clone(),
                    node_value_catalog.clone(),
                ],
                parent_node_value_count,
                parent_predicate_count,
                node_value_count,
                predicate_count,
            )?;

            additions.sort_by(|a, b| triple_cmp(a, b, ProofIndex::Spo));
            removals.sort_by(|a, b| triple_cmp(a, b, ProofIndex::Spo));

            // Real per-column Dory commitments over the sorted id columns.
            let column_commitments = commit_delta_columns(&additions, &removals)?;

            let state = CommitmentState {
                parent_state_root,
                addition_count: additions.len() as u64,
                removal_count: removals.len() as u64,
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
                predicate_catalog,
                node_value_catalog,
                column_commitments,
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
                .saturating_add(table_bytes(&self.predicate_catalog))
                .saturating_add(table_bytes(&self.node_value_catalog))
        }

        pub fn validate(&self) -> io::Result<()> {
            self.state.validate()?;
            if self.additions.len() > MAX_RECORDS || self.removals.len() > MAX_RECORDS {
                return Err(invalid("proof delta exceeds storage limit"));
            }
            if self.additions.len() as u64 != self.state.addition_count
                || self.removals.len() as u64 != self.state.removal_count
            {
                return Err(invalid("commitment delta count mismatch"));
            }
            for table in [&self.additions, &self.removals] {
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
            let subject_catalog = subject_catalog_from_objects(&self.node_value_catalog)?;
            validate_complete_catalog(
                &self.additions,
                &self.removals,
                &[
                    subject_catalog,
                    self.predicate_catalog.clone(),
                    self.node_value_catalog.clone(),
                ],
                self.state.parent_node_value_count,
                self.state.parent_predicate_count,
                self.state.node_value_count,
                self.state.predicate_count,
            )?;
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
            let parent_node_value_count = r.u64()?;
            let parent_predicate_count = r.u64()?;
            let node_value_count = r.u64()?;
            let predicate_count = r.u64()?;
            let additions = read_records(&mut r, PersistedTableKind::Delta)?;
            let removals = read_records(&mut r, PersistedTableKind::Delta)?;
            let predicate_records = read_records(&mut r, PersistedTableKind::PredicateCatalog)?;
            let object_records = read_records(&mut r, PersistedTableKind::ObjectCatalog)?;
            let mut column_commitments: [ArkGT; COLUMN_COUNT] =
                [<ArkGT as Group>::identity(); COLUMN_COUNT];
            for slot in column_commitments.iter_mut() {
                let bytes = r.bytes()?;
                *slot = deserialize_commitment(&bytes)?;
            }
            if !r.done() {
                return Err(invalid("trailing commitment file bytes"));
            }
            validate_decoded_heap_budget(&additions, &removals, &predicate_records, &object_records)?;
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
                predicate_catalog,
                node_value_catalog,
                column_commitments,
            };
            layer.validate()?;
            Ok(layer)
        }

        /// The commitment root binds the parent root, the six counts, and the
        /// canonical-serialized six per-column Dory commitments (NOT the raw
        /// tables, which are separately bound by the file checksum and revalidated
        /// on decode). The column commitments in turn bind the sorted id columns.
        fn compute_commitment_root(&self) -> io::Result<ProofHash> {
            let mut hasher = Sha256::new();
            hasher.update(b"terminus-store/proof-of-woql/commitment-root/v0");
            match self.state.parent_state_root {
                Some(root) => {
                    hasher.update([1u8]);
                    hasher.update(root);
                }
                None => hasher.update([0u8]),
            }
            hasher.update(self.state.addition_count.to_be_bytes());
            hasher.update(self.state.removal_count.to_be_bytes());
            hasher.update(self.state.parent_node_value_count.to_be_bytes());
            hasher.update(self.state.parent_predicate_count.to_be_bytes());
            hasher.update(self.state.node_value_count.to_be_bytes());
            hasher.update(self.state.predicate_count.to_be_bytes());
            for commitment in &self.column_commitments {
                let bytes = serialize_commitment(commitment)?;
                hasher.update((bytes.len() as u64).to_be_bytes());
                hasher.update(&bytes);
            }
            Ok(hasher.finalize().into())
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
            put_u64(&mut out, self.state.parent_node_value_count);
            put_u64(&mut out, self.state.parent_predicate_count);
            put_u64(&mut out, self.state.node_value_count);
            put_u64(&mut out, self.state.predicate_count);
            for table in [
                &self.additions,
                &self.removals,
                &self.predicate_catalog,
                &self.node_value_catalog,
            ] {
                put_records(&mut out, table);
            }
            for commitment in &self.column_commitments {
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
                .and_then(|value| value.checked_add(6 * 8))
                .ok_or_else(|| invalid("commitment manifest encoded length overflow"))?;
            let mut total = COMMITMENT_MAGIC
                .len()
                .checked_add(4)
                .and_then(|value| value.checked_add(state_bytes))
                .ok_or_else(|| invalid("commitment encoded length overflow"))?;
            for table in [
                &self.additions,
                &self.removals,
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
            for commitment in &self.column_commitments {
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
        actual: &[Vec<ProofRecord>; 3],
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

    #[allow(clippy::too_many_arguments)]
    fn validate_complete_catalog(
        additions: &[ProofRecord],
        removals: &[ProofRecord],
        indexes: &[Vec<ProofRecord>; 3],
        parent_node_value_count: u64,
        parent_predicate_count: u64,
        node_value_count: u64,
        predicate_count: u64,
    ) -> io::Result<()> {
        validate_dictionary_catalogs(indexes, node_value_count, predicate_count)?;
        let expected_node_values = node_value_count
            .checked_sub(parent_node_value_count)
            .ok_or_else(|| invalid("node/value dictionary count decreased"))?;
        let expected_predicates = predicate_count
            .checked_sub(parent_predicate_count)
            .ok_or_else(|| invalid("predicate dictionary count decreased"))?;
        if indexes[2].len() as u64 != expected_node_values
            || indexes[1].len() as u64 != expected_predicates
        {
            return Err(invalid(
                "dictionary catalog does not exactly cover the layer id interval",
            ));
        }

        let mut objects_by_id = BTreeMap::new();
        for record in indexes[2].iter() {
            if record.object_id <= parent_node_value_count {
                return Err(invalid("layer catalog repeats an inherited node/value id"));
            }
            objects_by_id.insert(record.object_id, record.triple.object.clone());
        }
        let mut predicates_by_id = BTreeMap::new();
        for record in indexes[1].iter() {
            if record.predicate_id <= parent_predicate_count {
                return Err(invalid("layer catalog repeats an inherited predicate id"));
            }
            predicates_by_id.insert(record.predicate_id, record.triple.predicate.clone());
        }
        let subjects_by_id: BTreeMap<_, _> = indexes[0]
            .iter()
            .map(|record| (record.subject_id, record.triple.subject.clone()))
            .collect();

        for record in additions.iter().chain(removals.iter()) {
            if record.subject_id > parent_node_value_count
                && (subjects_by_id.get(&record.subject_id) != Some(&record.triple.subject)
                    || objects_by_id.get(&record.subject_id)
                        != Some(&CanonicalObject::Node(record.triple.subject.clone())))
            {
                return Err(invalid(
                    "new subject id is not bound to delta canonical bytes",
                ));
            }
            if record.predicate_id > parent_predicate_count
                && predicates_by_id.get(&record.predicate_id) != Some(&record.triple.predicate)
            {
                return Err(invalid(
                    "new predicate id is not bound to delta canonical bytes",
                ));
            }
            if record.object_id > parent_node_value_count
                && objects_by_id.get(&record.object_id) != Some(&record.triple.object)
            {
                return Err(invalid(
                    "new object id is not bound to delta canonical bytes",
                ));
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
            LayerCommitment::from_native_delta(
                None,
                vec![record(1, 1, 2, "a", "p", "b")],
                vec![],
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

        // Same two-node/one-predicate dictionary {a:1, b:2}, {p:1}, but the caller
        // chooses whether the triple lands in additions and/or removals.
        fn delta(
            additions: Vec<ProofRecord>,
            removals: Vec<ProofRecord>,
        ) -> LayerCommitment {
            LayerCommitment::from_native_delta(
                None,
                additions,
                removals,
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
            assert_eq!(decoded.state.commitment_root, commitment.state.commitment_root);
        }

        #[test]
        fn transparent_setup_is_deterministic_across_fresh_derivations() {
            // Two independently-derived setups must produce byte-identical public
            // parameters, so any node commits a given column to the same value.
            let setup_a = deterministic_prover_setup(4);
            let setup_b = deterministic_prover_setup(4);
            assert_eq!(setup_a.g1_vec, setup_b.g1_vec);
            assert_eq!(setup_a.g2_vec, setup_b.g2_vec);

            let coefficients: Vec<ArkFr> =
                (1..=4u64).map(<ArkFr as DoryField>::from_u64).collect();
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
            assert_eq!(sample().state.commitment_root, sample().state.commitment_root);
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
                None, vec![], vec![], 0, 0, 0, 0, vec![], vec![],
            )
            .unwrap();
            let adds_only = delta(vec![record(1, 1, 2, "a", "p", "b")], vec![]);
            let removes_only = delta(vec![], vec![record(1, 1, 2, "a", "p", "b")]);

            // Empty columns use the canonical GT-identity commitment.
            let identity = <ArkGT as Group>::identity();
            assert!(empty.column_commitments.iter().all(|c| *c == identity));
            // Adds-only: remove columns are empty; removes-only: add columns empty.
            assert!(adds_only.column_commitments[3..].iter().all(|c| *c == identity));
            assert!(removes_only.column_commitments[..3].iter().all(|c| *c == identity));

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

/// Phase 1a smoke test: prove the permissive `dory-pcs` BN254 crypto stack links
/// and that Dory commit/open/verify works end-to-end (and rejects a tampered
/// evaluation). This exercises the REAL crate API only; it does not touch
/// `LayerCommitment`/`CommitmentState` logic, which still carry a placeholder
/// root. Phase 1b will wire the real column commitment through this API.
// TODO(proof-of-woql): replace this smoke test with the real per-column Dory
// commitment inside `proof::commitment` once the wiring lands.
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
