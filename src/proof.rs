//! Authenticated Store layer deltas, complete dictionaries, and compact range proofs.
//!
//! The execution/provenance root binds native ordering ids together with canonical
//! logical bytes and complete id mappings. The physical layer name remains only a
//! locator. Compact verification relies on a
//! root-admission invariant: Store admission rebuilds every ordered index and catalog
//! from the original native layer chain. A Merkle proof alone cannot establish the
//! ordering of hidden leaves.

use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::convert::{TryFrom, TryInto};
use std::io;
use std::sync::Arc;

use crate::layer::{ObjectType, ValueTriple};
use tdb_succinct::Datatype;

pub type ProofHash = [u8; 32];

pub const PROOF_FORMAT_VERSION: u32 = 4;
pub const PROOF_SCHEME: &[u8] = b"terminus-store/layer-delta/sha256-merkle/v4";
pub const CANONICALIZATION: &[u8] = b"terminus-store/logical-triple-bytes/v1";
const MAGIC: &[u8; 8] = b"TSPRF4\0\0";
const MAX_RECORDS: usize = 10_000_000;
const MAX_DECODED_RECORDS: usize = 4_000_000;
const MAX_FIELD_BYTES: usize = 16 * 1024 * 1024;
const MAX_WITNESS_LEAVES: u64 = 1 << 40;
const MAX_WITNESS_RECORDS: usize = 1_000_000;
const MAX_WITNESS_FRONTIER: usize = 1_000_000;
pub const MAX_WITNESS_LAYERS: usize = 1024;
const MAX_WITNESS_BYTES: usize = 64 * 1024 * 1024;
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

fn hash_parts(domain: &[u8], parts: &[&[u8]]) -> ProofHash {
    let mut h = Sha256::new();
    h.update((domain.len() as u64).to_be_bytes());
    h.update(domain);
    for part in parts {
        h.update((part.len() as u64).to_be_bytes());
        h.update(part);
    }
    h.finalize().into()
}

fn synthetic_id(domain: &[u8], value: &[u8]) -> u64 {
    let hash = hash_parts(b"terminus-store/proof/test-id/v4", &[domain, value]);
    u64::from_be_bytes(hash[..8].try_into().unwrap()).max(1)
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

    fn encoded(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.encode_into(&mut out);
        out
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

impl ProofIndex {
    fn tag(self) -> [u8; 1] {
        [self as u8]
    }
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

fn leaf_hash(index: ProofIndex, position: usize, record: &ProofRecord) -> ProofHash {
    hash_parts(
        b"terminus-store/proof/leaf/v4",
        &[
            PROOF_SCHEME,
            CANONICALIZATION,
            &index.tag(),
            &(position as u64).to_be_bytes(),
            &record.subject_id.to_be_bytes(),
            &record.predicate_id.to_be_bytes(),
            &record.object_id.to_be_bytes(),
            &record.triple.encoded(),
        ],
    )
}

fn padding_hash(index: ProofIndex, position: usize) -> ProofHash {
    hash_parts(
        b"terminus-store/proof/padding/v4",
        &[PROOF_SCHEME, &index.tag(), &(position as u64).to_be_bytes()],
    )
}

fn node_hash(index: ProofIndex, level: u32, left: &ProofHash, right: &ProofHash) -> ProofHash {
    hash_parts(
        b"terminus-store/proof/node/v4",
        &[
            PROOF_SCHEME,
            &index.tag(),
            &level.to_be_bytes(),
            left,
            right,
        ],
    )
}

fn described_root(index: ProofIndex, count: usize, tree_root: &ProofHash) -> ProofHash {
    hash_parts(
        b"terminus-store/proof/index-root/v4",
        &[
            PROOF_SCHEME,
            CANONICALIZATION,
            &index.tag(),
            &(count as u64).to_be_bytes(),
            tree_root,
        ],
    )
}

fn tree_levels(records: &[ProofRecord], order: &[usize], index: ProofIndex) -> Vec<Vec<ProofHash>> {
    let capacity = order.len().max(1).next_power_of_two();
    let mut leaves = Vec::with_capacity(capacity);
    for position in 0..capacity {
        leaves.push(if position < order.len() {
            leaf_hash(index, position, &records[order[position]])
        } else {
            padding_hash(index, position)
        });
    }
    let mut levels = vec![leaves];
    let mut level = 0u32;
    while levels.last().unwrap().len() > 1 {
        let previous = levels.last().unwrap();
        let next = previous
            .chunks_exact(2)
            .map(|p| node_hash(index, level, &p[0], &p[1]))
            .collect();
        levels.push(next);
        level += 1;
    }
    levels
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct AuthenticatedIndex {
    index: ProofIndex,
    records: Arc<Vec<ProofRecord>>,
    order: Arc<Vec<usize>>,
    levels: Arc<Vec<Vec<ProofHash>>>,
}

impl AuthenticatedIndex {
    fn build(records: Arc<Vec<ProofRecord>>, index: ProofIndex) -> io::Result<Self> {
        if records.len() > MAX_RECORDS {
            return Err(invalid("proof index exceeds storage limit"));
        }
        let mut order: Vec<_> = (0..records.len()).collect();
        order.sort_by(|a, b| triple_cmp(&records[*a], &records[*b], index));
        if order
            .windows(2)
            .any(|pair| triple_cmp(&records[pair[0]], &records[pair[1]], index) != Ordering::Less)
        {
            return Err(invalid("proof index contains duplicate records"));
        }
        let levels = tree_levels(&records, &order, index);
        Ok(Self {
            index,
            records,
            order: Arc::new(order),
            levels: Arc::new(levels),
        })
    }

    fn len(&self) -> usize {
        self.order.len()
    }

    fn get(&self, position: usize) -> &ProofRecord {
        &self.records[self.order[position]]
    }

    fn iter(&self) -> impl Iterator<Item = &ProofRecord> {
        self.order
            .iter()
            .map(move |position| &self.records[*position])
    }

    fn root(&self) -> ProofHash {
        described_root(self.index, self.len(), &self.levels.last().unwrap()[0])
    }

    fn validate(&self) -> io::Result<()> {
        if self.len() > MAX_RECORDS
            || self.order.windows(2).any(|pair| {
                triple_cmp(&self.records[pair[0]], &self.records[pair[1]], self.index)
                    != Ordering::Less
            })
        {
            return Err(invalid("persisted proof index is not strictly ordered"));
        }
        let expected_capacity = self
            .len()
            .max(1)
            .checked_next_power_of_two()
            .ok_or_else(|| invalid("persisted proof index capacity overflow"))?;
        if self.levels.is_empty() || self.levels[0].len() != expected_capacity {
            return Err(invalid("persisted proof tree has invalid leaf width"));
        }
        for (position, hash) in self.levels[0].iter().enumerate() {
            let expected = if position < self.len() {
                leaf_hash(self.index, position, self.get(position))
            } else {
                padding_hash(self.index, position)
            };
            if hash != &expected {
                return Err(invalid("persisted proof tree leaf mismatch"));
            }
        }
        let mut width = expected_capacity;
        for level in 1..self.levels.len() {
            width /= 2;
            if self.levels[level].len() != width {
                return Err(invalid("persisted proof tree level width mismatch"));
            }
            for (position, hash) in self.levels[level].iter().enumerate() {
                let expected = node_hash(
                    self.index,
                    (level - 1) as u32,
                    &self.levels[level - 1][position * 2],
                    &self.levels[level - 1][position * 2 + 1],
                );
                if hash != &expected {
                    return Err(invalid("persisted proof tree node mismatch"));
                }
            }
        }
        if width != 1 {
            return Err(invalid("persisted proof tree is truncated"));
        }
        Ok(())
    }

    fn prove_dictionary_budgeted(
        &self,
        key: DictionaryKey,
        budget: &mut WitnessBudget,
    ) -> io::Result<DictionaryProof> {
        if self.index != key.index() {
            return Err(invalid("dictionary key uses the wrong persisted index"));
        }
        let start = self
            .order
            .partition_point(|position| key.compare(&self.records[*position]) == Ordering::Less);
        let end = start
            + self.order[start..].partition_point(|position| {
                key.compare(&self.records[*position]) == Ordering::Equal
            });
        if end - start > 1 {
            return Err(invalid("dictionary key maps more than once"));
        }
        let predecessor_position = start.checked_sub(1);
        let successor_position = (end < self.len()).then_some(end);
        let frontier = make_contiguous_frontier(
            &self.levels,
            predecessor_position.unwrap_or(start),
            successor_position.map_or(end, |position| position + 1),
        );
        let key_bytes = match &key {
            DictionaryKey::Subject(value) | DictionaryKey::Predicate(value) => value.len(),
            DictionaryKey::Object(CanonicalObject::Node(value)) => value.len(),
            DictionaryKey::Object(CanonicalObject::Value { lexical, .. }) => lexical.len(),
        };
        budget.reserve_proof(
            (start..end).map(|position| self.get(position)),
            predecessor_position.map(|position| self.get(position)),
            successor_position.map(|position| self.get(position)),
            &frontier,
            key_bytes,
        )?;
        let predecessor = predecessor_position.map(|position| IndexedTriple {
            index: position as u64,
            record: self.get(position).clone(),
        });
        let successor = successor_position.map(|position| IndexedTriple {
            index: position as u64,
            record: self.get(position).clone(),
        });
        Ok(DictionaryProof {
            key,
            leaf_count: self.len() as u64,
            start: start as u64,
            records: (start..end)
                .map(|position| self.get(position).clone())
                .collect(),
            predecessor,
            successor,
            frontier,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProofNode {
    pub level: u32,
    pub index: u64,
    pub hash: ProofHash,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexedTriple {
    pub index: u64,
    pub record: ProofRecord,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RangeProof {
    pub index: ProofIndex,
    pub leaf_count: u64,
    pub start: u64,
    pub end: u64,
    pub resolved: ResolvedPattern,
    pub records: Vec<ProofRecord>,
    pub predecessor: Option<IndexedTriple>,
    pub successor: Option<IndexedTriple>,
    pub frontier: Vec<ProofNode>,
}

impl RangeProof {
    pub fn verify(
        &self,
        expected_root: ProofHash,
        pattern: &TriplePattern,
    ) -> io::Result<Vec<ProofRecord>> {
        pattern.validate()?;
        if self.index != pattern.index() {
            return Err(invalid("range proof uses the wrong index"));
        }
        if self.leaf_count > MAX_WITNESS_LEAVES
            || self.records.len() > MAX_WITNESS_RECORDS
            || self.frontier.len() > MAX_WITNESS_FRONTIER
        {
            return Err(invalid("range proof exceeds verifier limits"));
        }
        let count = usize::try_from(self.leaf_count).map_err(|_| invalid("leaf count overflow"))?;
        let start = usize::try_from(self.start).map_err(|_| invalid("range start overflow"))?;
        let end = usize::try_from(self.end).map_err(|_| invalid("range end overflow"))?;
        if start > end || end > count || self.records.len() != end - start {
            return Err(invalid("invalid range indices"));
        }
        if self.predecessor.as_ref().map(|v| v.index) != start.checked_sub(1).map(|v| v as u64) {
            return Err(invalid("missing or misplaced predecessor"));
        }
        if self.successor.as_ref().map(|v| v.index) != (end < count).then_some(end as u64) {
            return Err(invalid("missing or misplaced successor"));
        }

        let mut known = BTreeMap::<(u32, usize), ProofHash>::new();
        for (offset, triple) in self.records.iter().enumerate() {
            if pattern.carrier_cmp(triple, &self.resolved, self.index) != Ordering::Equal {
                return Err(invalid("record falls outside carrier range"));
            }
            insert_known(
                &mut known,
                (0, start + offset),
                leaf_hash(self.index, start + offset, triple),
            )?;
        }
        for boundary in self.predecessor.iter().chain(self.successor.iter()) {
            let i =
                usize::try_from(boundary.index).map_err(|_| invalid("boundary index overflow"))?;
            insert_known(
                &mut known,
                (0, i),
                leaf_hash(self.index, i, &boundary.record),
            )?;
        }
        if let Some(p) = &self.predecessor {
            if pattern.carrier_cmp(&p.record, &self.resolved, self.index) != Ordering::Less {
                return Err(invalid("predecessor does not bound range"));
            }
        } else if start != 0 {
            return Err(invalid("range omits predecessor"));
        }
        if let Some(s) = &self.successor {
            if pattern.carrier_cmp(&s.record, &self.resolved, self.index) != Ordering::Greater {
                return Err(invalid("successor does not bound range"));
            }
        } else if end != count {
            return Err(invalid("range omits successor"));
        }

        let mut visible: Vec<(usize, &ProofRecord)> = Vec::new();
        if let Some(p) = &self.predecessor {
            visible.push((
                usize::try_from(p.index).map_err(|_| invalid("predecessor index overflow"))?,
                &p.record,
            ));
        }
        visible.extend(self.records.iter().enumerate().map(|(i, t)| (start + i, t)));
        if let Some(s) = &self.successor {
            visible.push((
                usize::try_from(s.index).map_err(|_| invalid("successor index overflow"))?,
                &s.record,
            ));
        }
        visible.sort_by_key(|v| v.0);
        for pair in visible.windows(2) {
            if pair[0].0 == pair[1].0
                || (pair[0].0 + 1 == pair[1].0
                    && triple_cmp(pair[0].1, pair[1].1, self.index) != Ordering::Less)
            {
                return Err(invalid("duplicate or out-of-order visible leaves"));
            }
        }

        verify_frontier(self.index, count, known, &self.frontier, expected_root)?;
        let mut result: Vec<_> = self
            .records
            .iter()
            .filter(|record| pattern.matches(&record.triple))
            .cloned()
            .collect();
        if pattern.reverse {
            result.reverse();
        }
        Ok(result)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum DictionaryKey {
    Subject(Vec<u8>),
    Predicate(Vec<u8>),
    Object(CanonicalObject),
}

impl DictionaryKey {
    fn index(&self) -> ProofIndex {
        match self {
            Self::Subject(_) => ProofIndex::SubjectDictionary,
            Self::Predicate(_) => ProofIndex::PredicateDictionary,
            Self::Object(_) => ProofIndex::ObjectDictionary,
        }
    }

    fn compare(&self, record: &ProofRecord) -> Ordering {
        match self {
            Self::Subject(value) => record.triple.subject.cmp(value),
            Self::Predicate(value) => record.triple.predicate.cmp(value),
            Self::Object(value) => record.triple.object.cmp(value),
        }
    }

    fn id(&self, record: &ProofRecord) -> u64 {
        match self {
            Self::Subject(_) => record.subject_id,
            Self::Predicate(_) => record.predicate_id,
            Self::Object(_) => record.object_id,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DictionaryProof {
    pub key: DictionaryKey,
    pub leaf_count: u64,
    pub start: u64,
    pub records: Vec<ProofRecord>,
    pub predecessor: Option<IndexedTriple>,
    pub successor: Option<IndexedTriple>,
    pub frontier: Vec<ProofNode>,
}

impl DictionaryProof {
    pub fn verify(&self, expected_root: ProofHash) -> io::Result<Option<u64>> {
        if self.leaf_count > MAX_WITNESS_LEAVES
            || self.records.len() > 1
            || self.frontier.len() > MAX_WITNESS_FRONTIER
        {
            return Err(invalid("dictionary proof exceeds verifier limits"));
        }
        let index = self.key.index();
        let count =
            usize::try_from(self.leaf_count).map_err(|_| invalid("dictionary count overflow"))?;
        let start =
            usize::try_from(self.start).map_err(|_| invalid("dictionary start overflow"))?;
        let end = start
            .checked_add(self.records.len())
            .ok_or_else(|| invalid("dictionary range overflow"))?;
        if end > count
            || self.predecessor.as_ref().map(|v| v.index) != start.checked_sub(1).map(|v| v as u64)
            || self.successor.as_ref().map(|v| v.index) != (end < count).then_some(end as u64)
        {
            return Err(invalid("invalid dictionary range indices"));
        }
        if self
            .records
            .iter()
            .any(|record| self.key.compare(record) != Ordering::Equal)
        {
            return Err(invalid("dictionary proof returned a different key"));
        }
        if self.records.len() > 1 {
            return Err(invalid("dictionary key maps more than once"));
        }
        if let Some(predecessor) = &self.predecessor {
            if self.key.compare(&predecessor.record) != Ordering::Less {
                return Err(invalid("dictionary predecessor does not bound key"));
            }
        } else if start != 0 {
            return Err(invalid("dictionary proof omits predecessor"));
        }
        if let Some(successor) = &self.successor {
            if self.key.compare(&successor.record) != Ordering::Greater {
                return Err(invalid("dictionary successor does not bound key"));
            }
        } else if end != count {
            return Err(invalid("dictionary proof omits successor"));
        }

        let mut known = BTreeMap::new();
        for (offset, record) in self.records.iter().enumerate() {
            insert_known(
                &mut known,
                (0, start + offset),
                leaf_hash(index, start + offset, record),
            )?;
        }
        for boundary in self.predecessor.iter().chain(self.successor.iter()) {
            let position =
                usize::try_from(boundary.index).map_err(|_| invalid("boundary overflow"))?;
            insert_known(
                &mut known,
                (0, position),
                leaf_hash(index, position, &boundary.record),
            )?;
        }
        verify_frontier(index, count, known, &self.frontier, expected_root)?;
        Ok(self.records.first().map(|record| self.key.id(record)))
    }
}

fn verify_frontier(
    index: ProofIndex,
    count: usize,
    mut known: BTreeMap<(u32, usize), ProofHash>,
    frontier: &[ProofNode],
    expected_root: ProofHash,
) -> io::Result<()> {
    let capacity = count
        .max(1)
        .checked_next_power_of_two()
        .ok_or_else(|| invalid("Merkle capacity overflow"))?;
    let height = capacity.trailing_zeros();
    if count == 0 {
        insert_known(&mut known, (0, 0), padding_hash(index, 0))?;
    }
    let mut supplied = BTreeSet::new();
    for node in frontier {
        let level = usize::try_from(node.level).map_err(|_| invalid("frontier level overflow"))?;
        let node_index =
            usize::try_from(node.index).map_err(|_| invalid("frontier index overflow"))?;
        if level > height as usize || node_index >= (capacity >> level) {
            return Err(invalid("frontier node out of bounds"));
        }
        let key = (node.level, node_index);
        if !supplied.insert(key) || known.insert(key, node.hash).is_some() {
            return Err(invalid("duplicate or overlapping frontier node"));
        }
    }
    let mut used = BTreeSet::new();
    for level in 0..height {
        // Traverse only disclosed/derived positions. Iterating the padded tree
        // width would let a small witness with a huge declared leaf count force
        // O(capacity) verifier work.
        let positions: Vec<_> = known
            .range((level, 0)..(level + 1, 0))
            .map(|((_, position), _)| *position)
            .collect();
        let mut pairs = BTreeSet::new();
        for position in positions {
            let left = position & !1;
            if !pairs.insert(left) {
                continue;
            }
            let lk = (level, left);
            let rk = (level, left + 1);
            if let (Some(l), Some(r)) = (known.get(&lk).copied(), known.get(&rk).copied()) {
                if supplied.contains(&lk) {
                    used.insert(lk);
                }
                if supplied.contains(&rk) {
                    used.insert(rk);
                }
                let parent = node_hash(index, level, &l, &r);
                let key = (level + 1, left / 2);
                if let Some(existing) = known.get(&key) {
                    if supplied.contains(&key) || existing != &parent {
                        return Err(invalid("redundant or conflicting frontier node"));
                    }
                } else {
                    known.insert(key, parent);
                }
            }
        }
    }
    let actual = known
        .get(&(height, 0))
        .ok_or_else(|| invalid("incomplete Merkle frontier"))?;
    if described_root(index, count, actual) != expected_root || used != supplied {
        return Err(invalid("dictionary Merkle proof mismatch or unused node"));
    }
    Ok(())
}

fn insert_known(
    map: &mut BTreeMap<(u32, usize), ProofHash>,
    key: (u32, usize),
    hash: ProofHash,
) -> io::Result<()> {
    if let Some(old) = map.insert(key, hash) {
        if old != hash {
            return Err(invalid("conflicting Merkle node"));
        }
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProofLayerState {
    pub format_version: u32,
    pub parent_state_root: Option<ProofHash>,
    pub addition_count: u64,
    pub removal_count: u64,
    pub parent_node_value_count: u64,
    pub parent_predicate_count: u64,
    pub node_value_count: u64,
    pub predicate_count: u64,
    pub addition_roots: [ProofHash; 4],
    pub removal_roots: [ProofHash; 4],
    pub dictionary_roots: [ProofHash; 3],
    pub state_root: ProofHash,
}

impl ProofLayerState {
    fn compute_root(&self) -> ProofHash {
        let parent_tag = [self.parent_state_root.is_some() as u8];
        let parent = self.parent_state_root.unwrap_or([0; 32]);
        hash_parts(
            b"terminus-store/proof/layer-manifest/v4",
            &[
                PROOF_SCHEME,
                CANONICALIZATION,
                &self.format_version.to_be_bytes(),
                &parent_tag,
                &parent,
                &self.addition_count.to_be_bytes(),
                &self.removal_count.to_be_bytes(),
                &self.parent_node_value_count.to_be_bytes(),
                &self.parent_predicate_count.to_be_bytes(),
                &self.node_value_count.to_be_bytes(),
                &self.predicate_count.to_be_bytes(),
                &self.addition_roots[0],
                &self.addition_roots[1],
                &self.addition_roots[2],
                &self.addition_roots[3],
                &self.removal_roots[0],
                &self.removal_roots[1],
                &self.removal_roots[2],
                &self.removal_roots[3],
                &self.dictionary_roots[0],
                &self.dictionary_roots[1],
                &self.dictionary_roots[2],
            ],
        )
    }

    pub fn validate(&self) -> io::Result<()> {
        if self.format_version != PROOF_FORMAT_VERSION {
            return Err(invalid("unsupported proof format"));
        }
        if self.parent_node_value_count > self.node_value_count
            || self.parent_predicate_count > self.predicate_count
            || (self.parent_state_root.is_none()
                && (self.parent_node_value_count != 0 || self.parent_predicate_count != 0))
        {
            return Err(invalid("invalid dictionary parent counts"));
        }
        if self.compute_root() != self.state_root {
            return Err(invalid("layer manifest root mismatch"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct ProofLayer {
    pub state: ProofLayerState,
    addition_indexes: [AuthenticatedIndex; 4],
    removal_indexes: [AuthenticatedIndex; 4],
    dictionary_indexes: [AuthenticatedIndex; 3],
}

impl PartialEq for ProofLayer {
    fn eq(&self, other: &Self) -> bool {
        // A validated layer's manifest commits every ordered index and catalog.
        // Compact v4 decoding is free to choose a different canonical-table
        // ordinal layout while reconstructing the same authenticated state.
        self.state == other.state
    }
}

impl Eq for ProofLayer {}

impl ProofLayer {
    /// Approximate retained heap bytes for the decoded v4 proof cache. Shared
    /// canonical delta tables are counted once; ordinal projections and Merkle
    /// levels remain index-specific.
    pub fn estimated_cache_bytes(&self) -> usize {
        let record_bytes = |records: &Arc<Vec<ProofRecord>>| {
            records.iter().fold(
                records
                    .capacity()
                    .saturating_mul(std::mem::size_of::<ProofRecord>()),
                |total, record| {
                    let object = match &record.triple.object {
                        CanonicalObject::Node(value) => value.capacity(),
                        CanonicalObject::Value { lexical, .. } => lexical.capacity(),
                    };
                    total
                        .saturating_add(record.triple.subject.capacity())
                        .saturating_add(record.triple.predicate.capacity())
                        .saturating_add(object)
                },
            )
        };
        let index_bytes = |index: &AuthenticatedIndex| {
            index
                .order
                .capacity()
                .saturating_mul(std::mem::size_of::<usize>())
                .saturating_add(
                    index
                        .levels
                        .iter()
                        .map(|level| level.capacity().saturating_mul(32))
                        .sum::<usize>(),
                )
                .saturating_add(
                    index
                        .levels
                        .capacity()
                        .saturating_mul(std::mem::size_of::<Vec<ProofHash>>()),
                )
        };
        let mut total = record_bytes(&self.addition_indexes[0].records)
            .saturating_add(record_bytes(&self.removal_indexes[0].records));
        for index in self
            .addition_indexes
            .iter()
            .chain(self.removal_indexes.iter())
            .chain(self.dictionary_indexes.iter())
        {
            total = total.saturating_add(index_bytes(index));
        }
        for index in &self.dictionary_indexes {
            total = total.saturating_add(record_bytes(&index.records));
        }
        total
    }

    pub fn new(
        parent_state_root: Option<ProofHash>,
        mut additions: Vec<CanonicalTriple>,
        mut removals: Vec<CanonicalTriple>,
    ) -> io::Result<Self> {
        additions.sort();
        additions.dedup();
        removals.sort();
        removals.dedup();
        if additions.iter().any(|t| removals.binary_search(t).is_ok()) {
            return Err(invalid("same triple occurs in additions and removals"));
        }
        let make = |triples: Vec<CanonicalTriple>| {
            triples
                .into_iter()
                .map(|triple| {
                    let encode_object = |value: &CanonicalObject| {
                        let mut out = Vec::new();
                        match value {
                            CanonicalObject::Node(value) => {
                                out.push(0);
                                put_bytes(&mut out, value);
                            }
                            CanonicalObject::Value { datatype, lexical } => {
                                out.push(1);
                                out.push(*datatype as u8);
                                put_bytes(&mut out, lexical);
                            }
                        }
                        out
                    };
                    let subject = encode_object(&CanonicalObject::Node(triple.subject.clone()));
                    let object = encode_object(&triple.object);
                    ProofRecord {
                        subject_id: synthetic_id(b"node-value", &subject),
                        predicate_id: synthetic_id(b"predicate", &triple.predicate),
                        object_id: synthetic_id(b"node-value", &object),
                        triple,
                    }
                })
                .collect()
        };
        // This constructor is for Store-independent checker tests. Synthetic ids
        // have no cumulative native catalog, so use stable sentinel counts rather
        // than pretending the maximum hash is a Store dictionary length.
        Self::new_native_with_counts(
            parent_state_root,
            make(additions),
            make(removals),
            u64::MAX,
            u64::MAX,
        )
    }

    pub fn new_native(
        parent_state_root: Option<ProofHash>,
        additions: Vec<ProofRecord>,
        removals: Vec<ProofRecord>,
    ) -> io::Result<Self> {
        let node_value_count = additions
            .iter()
            .chain(removals.iter())
            .map(|record| record.subject_id.max(record.object_id))
            .max()
            .unwrap_or(0);
        let predicate_count = additions
            .iter()
            .chain(removals.iter())
            .map(|record| record.predicate_id)
            .max()
            .unwrap_or(0);
        Self::new_native_with_counts(
            parent_state_root,
            additions,
            removals,
            node_value_count,
            predicate_count,
        )
    }

    pub fn new_native_with_counts(
        parent_state_root: Option<ProofHash>,
        additions: Vec<ProofRecord>,
        removals: Vec<ProofRecord>,
        node_value_count: u64,
        predicate_count: u64,
    ) -> io::Result<Self> {
        let dictionary_indexes = build_dictionary_indexes(additions.iter().chain(removals.iter()))?;
        let parent_node_value_count = parent_state_root.map_or(0, |_| node_value_count);
        let parent_predicate_count = parent_state_root.map_or(0, |_| predicate_count);
        Self::new_native_with_dictionary_indexes(
            parent_state_root,
            additions,
            removals,
            parent_node_value_count,
            parent_predicate_count,
            node_value_count,
            predicate_count,
            dictionary_indexes,
        )
    }

    /// Build a production proof layer from the complete native dictionary delta.
    /// The supplied mappings contain only ids introduced by this layer; inherited
    /// mappings are authenticated by the parent proof state.
    pub fn new_native_with_catalog(
        parent_state_root: Option<ProofHash>,
        additions: Vec<ProofRecord>,
        removals: Vec<ProofRecord>,
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
        let dictionary_indexes =
            build_dictionary_indexes_from_maps(node_value_mappings, predicate_mappings)?;
        validate_complete_catalog(
            &additions,
            &removals,
            &dictionary_indexes,
            parent_node_value_count,
            parent_predicate_count,
            node_value_count,
            predicate_count,
        )?;
        Self::new_native_with_dictionary_indexes(
            parent_state_root,
            additions,
            removals,
            parent_node_value_count,
            parent_predicate_count,
            node_value_count,
            predicate_count,
            dictionary_indexes,
        )
    }

    fn new_native_with_dictionary_indexes(
        parent_state_root: Option<ProofHash>,
        additions: Vec<ProofRecord>,
        removals: Vec<ProofRecord>,
        parent_node_value_count: u64,
        parent_predicate_count: u64,
        node_value_count: u64,
        predicate_count: u64,
        dictionary_indexes: [AuthenticatedIndex; 3],
    ) -> io::Result<Self> {
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
        let addition_count = additions.len() as u64;
        let removal_count = removals.len() as u64;
        let addition_indexes = build_indexes(additions)?;
        let removal_indexes = build_indexes(removals)?;
        validate_dictionary_indexes(&dictionary_indexes, node_value_count, predicate_count)?;
        let addition_roots = [
            addition_indexes[0].root(),
            addition_indexes[1].root(),
            addition_indexes[2].root(),
            addition_indexes[3].root(),
        ];
        let removal_roots = [
            removal_indexes[0].root(),
            removal_indexes[1].root(),
            removal_indexes[2].root(),
            removal_indexes[3].root(),
        ];
        let mut state = ProofLayerState {
            format_version: PROOF_FORMAT_VERSION,
            parent_state_root,
            addition_count,
            removal_count,
            parent_node_value_count,
            parent_predicate_count,
            node_value_count,
            predicate_count,
            addition_roots,
            removal_roots,
            dictionary_roots: [
                dictionary_indexes[0].root(),
                dictionary_indexes[1].root(),
                dictionary_indexes[2].root(),
            ],
            state_root: [0; 32],
        };
        state.state_root = state.compute_root();
        Ok(Self {
            state,
            addition_indexes,
            removal_indexes,
            dictionary_indexes,
        })
    }

    pub fn additions(&self) -> Vec<CanonicalTriple> {
        self.addition_indexes[ProofIndex::Spo as usize]
            .iter()
            .map(|r| r.triple.clone())
            .collect()
    }
    pub fn removals(&self) -> Vec<CanonicalTriple> {
        self.removal_indexes[ProofIndex::Spo as usize]
            .iter()
            .map(|r| r.triple.clone())
            .collect()
    }

    pub fn validate(&self) -> io::Result<()> {
        self.state.validate()?;
        for index in self
            .addition_indexes
            .iter()
            .chain(self.removal_indexes.iter())
            .chain(self.dictionary_indexes.iter())
        {
            index.validate()?;
        }
        validate_index_equivalence(&self.addition_indexes)?;
        validate_index_equivalence(&self.removal_indexes)?;
        validate_dictionary_indexes(
            &self.dictionary_indexes,
            self.state.node_value_count,
            self.state.predicate_count,
        )?;
        if self.addition_indexes[0].len() as u64 != self.state.addition_count
            || self.removals().len() as u64 != self.state.removal_count
        {
            return Err(invalid("manifest count mismatch"));
        }
        let addition_roots = [
            self.addition_indexes[0].root(),
            self.addition_indexes[1].root(),
            self.addition_indexes[2].root(),
            self.addition_indexes[3].root(),
        ];
        let removal_roots = [
            self.removal_indexes[0].root(),
            self.removal_indexes[1].root(),
            self.removal_indexes[2].root(),
            self.removal_indexes[3].root(),
        ];
        let dictionary_roots = [
            self.dictionary_indexes[0].root(),
            self.dictionary_indexes[1].root(),
            self.dictionary_indexes[2].root(),
        ];
        if addition_roots != self.state.addition_roots
            || removal_roots != self.state.removal_roots
            || dictionary_roots != self.state.dictionary_roots
        {
            return Err(invalid("manifest index root mismatch"));
        }
        if self.addition_indexes[0]
            .iter()
            .any(|t| self.removal_indexes[0].iter().any(|r| r.triple == t.triple))
        {
            return Err(invalid("conflicting layer delta"));
        }
        Ok(())
    }

    pub fn prove(&self, additions: bool, pattern: &TriplePattern) -> io::Result<RangeProof> {
        let resolved = self.resolve_pattern_for_tests(pattern)?;
        self.prove_resolved(additions, pattern, &resolved)
    }

    pub fn prove_resolved(
        &self,
        additions: bool,
        pattern: &TriplePattern,
        resolved: &ResolvedPattern,
    ) -> io::Result<RangeProof> {
        let mut budget = WitnessBudget::new(1)?;
        self.prove_resolved_with_budget(additions, pattern, resolved, &mut budget)
    }

    pub(crate) fn prove_resolved_with_budget(
        &self,
        additions: bool,
        pattern: &TriplePattern,
        resolved: &ResolvedPattern,
        budget: &mut WitnessBudget,
    ) -> io::Result<RangeProof> {
        pattern.validate()?;
        let indexes = if additions {
            &self.addition_indexes
        } else {
            &self.removal_indexes
        };
        let index = pattern.index();
        make_range_proof(&indexes[index as usize], pattern, resolved, budget)
    }

    fn resolve_pattern_for_tests(&self, pattern: &TriplePattern) -> io::Result<ResolvedPattern> {
        pattern.validate()?;
        let records = self.addition_indexes[0]
            .iter()
            .chain(self.removal_indexes[0].iter());
        let all: Vec<_> = records.collect();
        let subject = pattern.subject.as_ref().and_then(|value| {
            all.iter()
                .find(|r| &r.triple.subject == value)
                .map(|r| r.subject_id)
        });
        let predicate = pattern.predicate.as_ref().and_then(|value| {
            all.iter()
                .find(|r| &r.triple.predicate == value)
                .map(|r| r.predicate_id)
        });
        let object = pattern.object.as_ref().and_then(|value| {
            all.iter()
                .find(|r| &r.triple.object == value)
                .map(|r| r.object_id)
        });
        let value_range = pattern.value_range.as_ref().and_then(|range| {
            let mut ids: Vec<_> = all
                .iter()
                .filter(|r| range.classify(&r.triple.object) == Ordering::Equal)
                .map(|r| r.object_id)
                .collect();
            ids.sort_unstable();
            ids.dedup();
            ids.first()
                .copied()
                .zip(ids.last().and_then(|last| last.checked_add(1)))
        });
        Ok(ResolvedPattern {
            subject,
            predicate,
            object,
            value_range,
        })
    }

    pub fn contribute_resolution(
        &self,
        pattern: &TriplePattern,
        resolved: &mut ResolvedPattern,
    ) -> io::Result<()> {
        pattern.validate()?;
        if let Some(subject) = &pattern.subject {
            if let Some(record) = self.dictionary_indexes[0]
                .iter()
                .find(|record| &record.triple.subject == subject)
            {
                bind_id(&mut resolved.subject, record.subject_id)?;
            }
        }
        if let Some(predicate) = &pattern.predicate {
            if let Some(record) = self.dictionary_indexes[1]
                .iter()
                .find(|record| &record.triple.predicate == predicate)
            {
                bind_id(&mut resolved.predicate, record.predicate_id)?;
            }
        }
        if let Some(object) = &pattern.object {
            if let Some(record) = self.dictionary_indexes[2]
                .iter()
                .find(|record| &record.triple.object == object)
            {
                bind_id(&mut resolved.object, record.object_id)?;
            }
        }
        Ok(())
    }

    pub fn prove_dictionary_bindings(
        &self,
        pattern: &TriplePattern,
    ) -> io::Result<Vec<DictionaryProof>> {
        let mut budget = WitnessBudget::new(1)?;
        self.prove_dictionary_bindings_with_budget(pattern, &mut budget)
    }

    pub(crate) fn prove_dictionary_bindings_with_budget(
        &self,
        pattern: &TriplePattern,
        budget: &mut WitnessBudget,
    ) -> io::Result<Vec<DictionaryProof>> {
        pattern.validate()?;
        let mut proofs = Vec::new();
        if let Some(subject) = &pattern.subject {
            proofs.push(
                self.dictionary_indexes[0]
                    .prove_dictionary_budgeted(DictionaryKey::Subject(subject.clone()), budget)?,
            );
        }
        if let Some(predicate) = &pattern.predicate {
            proofs.push(
                self.dictionary_indexes[1].prove_dictionary_budgeted(
                    DictionaryKey::Predicate(predicate.clone()),
                    budget,
                )?,
            );
        }
        if let Some(object) = &pattern.object {
            proofs.push(
                self.dictionary_indexes[2]
                    .prove_dictionary_budgeted(DictionaryKey::Object(object.clone()), budget)?,
            );
        }
        Ok(proofs)
    }

    pub fn encode(&self) -> io::Result<Vec<u8>> {
        self.validate()?;
        let encoded_len = self.encoded_len()?;
        let mut out = Vec::with_capacity(encoded_len);
        out.extend_from_slice(MAGIC);
        put_u32(&mut out, PROOF_FORMAT_VERSION);
        put_state(&mut out, &self.state);
        // Persist each logical record only once. The remaining authenticated
        // orderings, node-subject catalog, and Merkle levels are deterministic
        // projections rebuilt and root-checked on admission, then Arc-cached.
        put_index_records(&mut out, &self.addition_indexes[ProofIndex::Spo as usize]);
        put_index_records(&mut out, &self.removal_indexes[ProofIndex::Spo as usize]);
        put_index_records(&mut out, &self.dictionary_indexes[1]);
        put_index_records(&mut out, &self.dictionary_indexes[2]);
        let checksum = hash_parts(b"terminus-store/proof/file/v4", &[&out]);
        out.extend_from_slice(&checksum);
        if out.len() != encoded_len {
            return Err(invalid("proof encoded length accounting mismatch"));
        }
        Ok(out)
    }

    fn encoded_len(&self) -> io::Result<usize> {
        let state_bytes = 4usize
            .checked_add(1)
            .and_then(|value| {
                value.checked_add(if self.state.parent_state_root.is_some() {
                    32
                } else {
                    0
                })
            })
            .and_then(|value| value.checked_add(6 * 8))
            .and_then(|value| value.checked_add(11 * 32))
            .and_then(|value| value.checked_add(32))
            .ok_or_else(|| invalid("proof manifest encoded length overflow"))?;
        let mut total = MAGIC
            .len()
            .checked_add(4)
            .and_then(|value| value.checked_add(state_bytes))
            .ok_or_else(|| invalid("proof encoded length overflow"))?;
        for index in [
            &self.addition_indexes[ProofIndex::Spo as usize],
            &self.removal_indexes[ProofIndex::Spo as usize],
            &self.dictionary_indexes[1],
            &self.dictionary_indexes[2],
        ] {
            total = checked_proof_encoded_growth(total, 8)?;
            for record in index.iter() {
                let record_len = 24usize
                    .checked_add(record.triple.encoded_len()?)
                    .ok_or_else(|| invalid("proof encoded length overflow"))?;
                total = checked_proof_encoded_growth(total, record_len)?;
            }
        }
        checked_proof_encoded_growth(total, 32)
    }

    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < MAGIC.len() + 4 + 32 || bytes.len() > MAX_PROOF_LAYER_BYTES {
            return Err(invalid("truncated proof layer"));
        }
        let (payload, checksum) = bytes.split_at(bytes.len() - 32);
        if hash_parts(b"terminus-store/proof/file/v4", &[payload]).as_slice() != checksum {
            return Err(invalid("proof file checksum mismatch"));
        }
        let mut r = Reader::new(payload);
        if r.take(MAGIC.len())? != MAGIC {
            return Err(invalid("bad proof file magic"));
        }
        if r.u32()? != PROOF_FORMAT_VERSION {
            return Err(invalid("unsupported proof file version"));
        }
        let state = read_state(&mut r)?;
        let additions = read_records(&mut r, PersistedTableKind::Delta)?;
        let removals = read_records(&mut r, PersistedTableKind::Delta)?;
        let predicate_records = read_records(&mut r, PersistedTableKind::PredicateCatalog)?;
        let object_records = read_records(&mut r, PersistedTableKind::ObjectCatalog)?;
        if !r.done() {
            return Err(invalid("trailing proof file bytes"));
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
        let addition_indexes = build_indexes(additions)?;
        let removal_indexes = build_indexes(removals)?;
        let dictionary_indexes =
            build_dictionary_indexes_from_maps(node_value_mappings, predicate_mappings)?;
        let layer = Self {
            state,
            addition_indexes,
            removal_indexes,
            dictionary_indexes,
        };
        layer.validate()?;
        Ok(layer)
    }
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

fn bind_id(slot: &mut Option<u64>, id: u64) -> io::Result<()> {
    match slot {
        Some(existing) if *existing != id => Err(invalid(
            "canonical dictionary value maps to multiple native ids",
        )),
        Some(_) => Ok(()),
        None => {
            *slot = Some(id);
            Ok(())
        }
    }
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
/// retained-heap calculation before it builds indexes; this guard prevents the
/// input vectors themselves from growing past the same cap first.
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

fn build_indexes(records: Vec<ProofRecord>) -> io::Result<[AuthenticatedIndex; 4]> {
    let records = Arc::new(records);
    Ok([
        AuthenticatedIndex::build(records.clone(), ProofIndex::Spo)?,
        AuthenticatedIndex::build(records.clone(), ProofIndex::Pos)?,
        AuthenticatedIndex::build(records.clone(), ProofIndex::Osp)?,
        AuthenticatedIndex::build(records, ProofIndex::Value)?,
    ])
}

fn build_dictionary_indexes<'a>(
    records: impl Iterator<Item = &'a ProofRecord>,
) -> io::Result<[AuthenticatedIndex; 3]> {
    let mut node_values = Vec::new();
    let mut predicates = Vec::new();
    for record in records {
        node_values.push((
            CanonicalObject::Node(record.triple.subject.clone()),
            record.subject_id,
        ));
        node_values.push((record.triple.object.clone(), record.object_id));
        predicates.push((record.triple.predicate.clone(), record.predicate_id));
    }
    build_dictionary_indexes_from_maps(node_values, predicates)
}

fn build_dictionary_indexes_from_maps(
    node_value_mappings: Vec<(CanonicalObject, u64)>,
    predicate_mappings: Vec<(Vec<u8>, u64)>,
) -> io::Result<[AuthenticatedIndex; 3]> {
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
    Ok([
        AuthenticatedIndex::build(Arc::new(subject_records), ProofIndex::SubjectDictionary)?,
        AuthenticatedIndex::build(Arc::new(predicate_records), ProofIndex::PredicateDictionary)?,
        AuthenticatedIndex::build(Arc::new(object_records), ProofIndex::ObjectDictionary)?,
    ])
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

fn validate_dictionary_indexes(
    actual: &[AuthenticatedIndex; 3],
    node_value_count: u64,
    predicate_count: u64,
) -> io::Result<()> {
    if actual[0].index != ProofIndex::SubjectDictionary
        || actual[1].index != ProofIndex::PredicateDictionary
        || actual[2].index != ProofIndex::ObjectDictionary
    {
        return Err(invalid("persisted dictionary indexes have incorrect kinds"));
    }
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

fn validate_complete_catalog(
    additions: &[ProofRecord],
    removals: &[ProofRecord],
    indexes: &[AuthenticatedIndex; 3],
    parent_node_value_count: u64,
    parent_predicate_count: u64,
    node_value_count: u64,
    predicate_count: u64,
) -> io::Result<()> {
    validate_dictionary_indexes(indexes, node_value_count, predicate_count)?;
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

fn validate_index_equivalence(indexes: &[AuthenticatedIndex; 4]) -> io::Result<()> {
    if indexes[0].index != ProofIndex::Spo
        || indexes[1].index != ProofIndex::Pos
        || indexes[2].index != ProofIndex::Osp
        || indexes[3].index != ProofIndex::Value
    {
        return Err(invalid("persisted proof indexes have incorrect kinds"));
    }
    for index in &indexes[1..] {
        if !Arc::ptr_eq(&index.records, &indexes[0].records) {
            return Err(invalid(
                "authenticated orderings do not share one canonical record table",
            ));
        }
    }
    Ok(())
}

fn make_range_proof(
    authenticated: &AuthenticatedIndex,
    pattern: &TriplePattern,
    resolved: &ResolvedPattern,
    budget: &mut WitnessBudget,
) -> io::Result<RangeProof> {
    let index = authenticated.index;
    if authenticated.order.windows(2).any(|pair| {
        triple_cmp(
            &authenticated.records[pair[0]],
            &authenticated.records[pair[1]],
            index,
        ) != Ordering::Less
    }) {
        return Err(invalid("index records not strictly ordered"));
    }
    let start = authenticated.order.partition_point(|position| {
        pattern.carrier_cmp(&authenticated.records[*position], resolved, index) == Ordering::Less
    });
    let end = start
        + authenticated.order[start..].partition_point(|position| {
            pattern.carrier_cmp(&authenticated.records[*position], resolved, index)
                == Ordering::Equal
        });
    let predecessor_position = start.checked_sub(1);
    let successor_position = (end < authenticated.len()).then_some(end);
    let frontier = make_contiguous_frontier(
        &authenticated.levels,
        predecessor_position.unwrap_or(start),
        successor_position.map_or(end, |position| position + 1),
    );
    budget.reserve_proof(
        (start..end).map(|position| authenticated.get(position)),
        predecessor_position.map(|position| authenticated.get(position)),
        successor_position.map(|position| authenticated.get(position)),
        &frontier,
        0,
    )?;
    let predecessor = predecessor_position.map(|i| IndexedTriple {
        index: i as u64,
        record: authenticated.get(i).clone(),
    });
    let successor = successor_position.map(|i| IndexedTriple {
        index: i as u64,
        record: authenticated.get(i).clone(),
    });
    Ok(RangeProof {
        index,
        leaf_count: authenticated.len() as u64,
        start: start as u64,
        end: end as u64,
        resolved: resolved.clone(),
        records: (start..end)
            .map(|position| authenticated.get(position).clone())
            .collect(),
        predecessor,
        successor,
        frontier,
    })
}

fn make_contiguous_frontier(
    levels: &[Vec<ProofHash>],
    mut known_start: usize,
    mut known_end: usize,
) -> Vec<ProofNode> {
    if known_start == known_end {
        known_end = known_end.saturating_add(1);
    }
    let mut frontier = Vec::new();
    for level in 0..levels.len() - 1 {
        if known_start % 2 == 1 {
            let sibling = known_start - 1;
            frontier.push(ProofNode {
                level: level as u32,
                index: sibling as u64,
                hash: levels[level][sibling],
            });
        }
        if known_end % 2 == 1 && known_end < levels[level].len() {
            frontier.push(ProofNode {
                level: level as u32,
                index: known_end as u64,
                hash: levels[level][known_end],
            });
        }
        known_start /= 2;
        known_end = known_end.saturating_add(1) / 2;
    }
    frontier.sort_by_key(|n| (n.level, n.index));
    frontier.dedup_by_key(|n| (n.level, n.index));
    frontier
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerRangeWitness {
    pub state: ProofLayerState,
    pub dictionary: Vec<DictionaryProof>,
    pub additions: Option<RangeProof>,
    pub removals: Option<RangeProof>,
}

pub(crate) struct WitnessBudget {
    records: usize,
    frontier: usize,
    bytes: usize,
}

impl WitnessBudget {
    pub(crate) fn new(layer_count: usize) -> io::Result<Self> {
        if layer_count > MAX_WITNESS_LAYERS {
            return Err(invalid("proof layer chain exceeds verifier limit"));
        }
        let bytes = layer_count
            .checked_mul(512)
            .ok_or_else(|| invalid("witness byte count overflow"))?;
        if bytes > MAX_WITNESS_BYTES {
            return Err(invalid("proof chain exceeds aggregate verifier limits"));
        }
        Ok(Self {
            records: 0,
            frontier: 0,
            bytes,
        })
    }

    fn reserve_proof<'a>(
        &mut self,
        records: impl Iterator<Item = &'a ProofRecord>,
        predecessor: Option<&'a ProofRecord>,
        successor: Option<&'a ProofRecord>,
        frontier: &[ProofNode],
        extra_bytes: usize,
    ) -> io::Result<()> {
        let mut record_count = self.records;
        let mut byte_count = self
            .bytes
            .checked_add(extra_bytes)
            .ok_or_else(|| invalid("witness byte count overflow"))?;
        for record in records.chain(predecessor).chain(successor) {
            record_count = record_count
                .checked_add(1)
                .ok_or_else(|| invalid("witness record count overflow"))?;
            if record_count > MAX_WITNESS_RECORDS {
                return Err(invalid("proof chain exceeds aggregate verifier limits"));
            }
            byte_count = byte_count
                .checked_add(proof_record_bytes(record)?)
                .ok_or_else(|| invalid("witness byte count overflow"))?;
            if byte_count > MAX_WITNESS_BYTES {
                return Err(invalid("proof chain exceeds aggregate verifier limits"));
            }
        }
        let frontier_count = self
            .frontier
            .checked_add(frontier.len())
            .ok_or_else(|| invalid("witness frontier count overflow"))?;
        if frontier_count > MAX_WITNESS_FRONTIER {
            return Err(invalid("proof chain exceeds aggregate verifier limits"));
        }
        byte_count = byte_count
            .checked_add(
                frontier
                    .len()
                    .checked_mul(44)
                    .ok_or_else(|| invalid("witness frontier byte count overflow"))?,
            )
            .ok_or_else(|| invalid("witness byte count overflow"))?;
        if byte_count > MAX_WITNESS_BYTES {
            return Err(invalid("proof chain exceeds aggregate verifier limits"));
        }
        self.records = record_count;
        self.frontier = frontier_count;
        self.bytes = byte_count;
        Ok(())
    }
}

fn proof_record_bytes(record: &ProofRecord) -> io::Result<usize> {
    let object_bytes = match &record.triple.object {
        CanonicalObject::Node(value) => value.len(),
        CanonicalObject::Value { lexical, .. } => lexical.len(),
    };
    32usize
        .checked_add(record.triple.subject.len())
        .and_then(|value| value.checked_add(record.triple.predicate.len()))
        .and_then(|value| value.checked_add(object_bytes))
        .ok_or_else(|| invalid("witness record byte count overflow"))
}

fn add_proof_budget(
    records: &[ProofRecord],
    predecessor: Option<&IndexedTriple>,
    successor: Option<&IndexedTriple>,
    frontier: &[ProofNode],
    record_count: &mut usize,
    frontier_count: &mut usize,
    byte_count: &mut usize,
) -> io::Result<()> {
    let boundary_count = usize::from(predecessor.is_some()) + usize::from(successor.is_some());
    *record_count = record_count
        .checked_add(records.len())
        .and_then(|value| value.checked_add(boundary_count))
        .ok_or_else(|| invalid("witness record count overflow"))?;
    *frontier_count = frontier_count
        .checked_add(frontier.len())
        .ok_or_else(|| invalid("witness frontier count overflow"))?;
    for record in records
        .iter()
        .chain(predecessor.map(|value| &value.record))
        .chain(successor.map(|value| &value.record))
    {
        *byte_count = byte_count
            .checked_add(proof_record_bytes(record)?)
            .ok_or_else(|| invalid("witness byte count overflow"))?;
    }
    *byte_count = byte_count
        .checked_add(
            frontier
                .len()
                .checked_mul(44)
                .ok_or_else(|| invalid("witness frontier byte count overflow"))?,
        )
        .ok_or_else(|| invalid("witness byte count overflow"))?;
    Ok(())
}

pub fn validate_witness_limits(layers: &[LayerRangeWitness]) -> io::Result<()> {
    if layers.len() > MAX_WITNESS_LAYERS {
        return Err(invalid("proof layer chain exceeds verifier limit"));
    }
    let mut records = 0usize;
    let mut frontier = 0usize;
    let mut bytes = layers
        .len()
        .checked_mul(512)
        .ok_or_else(|| invalid("witness byte count overflow"))?;
    for layer in layers {
        for proof in &layer.dictionary {
            add_proof_budget(
                &proof.records,
                proof.predecessor.as_ref(),
                proof.successor.as_ref(),
                &proof.frontier,
                &mut records,
                &mut frontier,
                &mut bytes,
            )?;
            let key_bytes = match &proof.key {
                DictionaryKey::Subject(value) | DictionaryKey::Predicate(value) => value.len(),
                DictionaryKey::Object(CanonicalObject::Node(value)) => value.len(),
                DictionaryKey::Object(CanonicalObject::Value { lexical, .. }) => lexical.len(),
            };
            bytes = bytes
                .checked_add(key_bytes)
                .ok_or_else(|| invalid("witness byte count overflow"))?;
        }
        for proof in layer.additions.iter().chain(layer.removals.iter()) {
            add_proof_budget(
                &proof.records,
                proof.predecessor.as_ref(),
                proof.successor.as_ref(),
                &proof.frontier,
                &mut records,
                &mut frontier,
                &mut bytes,
            )?;
        }
    }
    if records > MAX_WITNESS_RECORDS || frontier > MAX_WITNESS_FRONTIER || bytes > MAX_WITNESS_BYTES
    {
        return Err(invalid("proof chain exceeds aggregate verifier limits"));
    }
    Ok(())
}

pub fn verify_layer_chain(
    expected_top_root: ProofHash,
    pattern: &TriplePattern,
    layers: &[LayerRangeWitness],
) -> io::Result<Vec<CanonicalTriple>> {
    if layers.is_empty() {
        return Err(invalid("empty proof layer chain"));
    }
    validate_witness_limits(layers)?;
    pattern.validate()?;
    let mut expected_keys = Vec::new();
    if let Some(subject) = &pattern.subject {
        expected_keys.push(DictionaryKey::Subject(subject.clone()));
    }
    if let Some(predicate) = &pattern.predicate {
        expected_keys.push(DictionaryKey::Predicate(predicate.clone()));
    }
    if let Some(object) = &pattern.object {
        expected_keys.push(DictionaryKey::Object(object.clone()));
    }
    let mut expected_parent = None;
    let mut resolved = ResolvedPattern::default();
    let mut previous_counts = (0, 0);
    for (position, layer) in layers.iter().enumerate() {
        layer.state.validate()?;
        if layer.state.parent_state_root != expected_parent {
            return Err(invalid("broken proof layer parent chain"));
        }
        if layer.state.parent_node_value_count != previous_counts.0
            || layer.state.parent_predicate_count != previous_counts.1
        {
            return Err(invalid("dictionary parent counts do not match proof chain"));
        }
        previous_counts = (layer.state.node_value_count, layer.state.predicate_count);
        if layer.dictionary.len() != expected_keys.len() {
            return Err(invalid("missing dictionary binding proof"));
        }
        for (proof, key) in layer.dictionary.iter().zip(expected_keys.iter()) {
            if &proof.key != key {
                return Err(invalid("dictionary proof binds a different query term"));
            }
            let root_index = (key.index() as usize) - (ProofIndex::SubjectDictionary as usize);
            if let Some(id) = proof.verify(layer.state.dictionary_roots[root_index])? {
                match key {
                    DictionaryKey::Subject(_) => bind_id(&mut resolved.subject, id)?,
                    DictionaryKey::Predicate(_) => bind_id(&mut resolved.predicate, id)?,
                    DictionaryKey::Object(_) => bind_id(&mut resolved.object, id)?,
                }
            }
        }
        expected_parent = Some(layer.state.state_root);
        if position + 1 == layers.len() && layer.state.state_root != expected_top_root {
            return Err(invalid("top proof state root mismatch"));
        }
    }

    let missing_binding = (pattern.subject.is_some() && resolved.subject.is_none())
        || (pattern.predicate.is_some() && resolved.predicate.is_none())
        || (pattern.object.is_some() && resolved.object.is_none());
    if missing_binding {
        if layers
            .iter()
            .any(|layer| layer.additions.is_some() || layer.removals.is_some())
        {
            return Err(invalid(
                "dictionary-empty query contains unnecessary range evidence",
            ));
        }
        return Ok(Vec::new());
    }

    let mut visible = BTreeMap::<(u64, u64, u64), ProofRecord>::new();
    for layer in layers {
        let addition_proof = layer
            .additions
            .as_ref()
            .ok_or_else(|| invalid("missing addition range proof"))?;
        let removal_proof = layer
            .removals
            .as_ref()
            .ok_or_else(|| invalid("missing removal range proof"))?;
        let expected_index = pattern.index();
        if addition_proof.index != expected_index || removal_proof.index != expected_index {
            return Err(invalid("range proof uses the wrong index"));
        }
        let ai = addition_proof.index as usize;
        let ri = removal_proof.index as usize;
        let additions = addition_proof.verify(layer.state.addition_roots[ai], pattern)?;
        let removals = removal_proof.verify(layer.state.removal_roots[ri], pattern)?;
        if addition_proof.leaf_count != layer.state.addition_count
            || removal_proof.leaf_count != layer.state.removal_count
        {
            return Err(invalid("range proof count differs from manifest"));
        }
        if addition_proof.resolved != resolved || removal_proof.resolved != resolved {
            return Err(invalid(
                "query ids are not bound by authenticated dictionaries",
            ));
        }
        for record in removals {
            let key = (record.subject_id, record.predicate_id, record.object_id);
            if visible.remove(&key).map(|old| old.triple) != Some(record.triple) {
                return Err(invalid("non-canonical removal of an absent triple"));
            }
        }
        for record in additions {
            let key = (record.subject_id, record.predicate_id, record.object_id);
            if visible.insert(key, record).is_some() {
                return Err(invalid("non-canonical addition of an existing triple"));
            }
        }
    }
    let mut records: Vec<_> = visible.into_values().collect();
    if pattern.value_range.is_some() {
        let introduction = |id: u64| {
            layers
                .iter()
                .position(|layer| id <= layer.state.node_value_count)
                .unwrap_or(layers.len())
        };
        records.sort_by(|a, b| {
            let layer_order = introduction(a.object_id).cmp(&introduction(b.object_id));
            if layer_order != Ordering::Equal {
                return layer_order;
            }
            let object_order = if pattern.reverse {
                b.object_id.cmp(&a.object_id)
            } else {
                a.object_id.cmp(&b.object_id)
            };
            object_order
                .then_with(|| a.subject_id.cmp(&b.subject_id))
                .then_with(|| a.predicate_id.cmp(&b.predicate_id))
        });
    } else {
        records.sort_by(|a, b| triple_cmp(a, b, pattern.index()));
        if pattern.reverse {
            records.reverse();
        }
    }
    Ok(records.into_iter().map(|record| record.triple).collect())
}

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
fn put_state(out: &mut Vec<u8>, state: &ProofLayerState) {
    put_u32(out, state.format_version);
    out.push(state.parent_state_root.is_some() as u8);
    if let Some(root) = state.parent_state_root {
        put_hash(out, &root);
    }
    put_u64(out, state.addition_count);
    put_u64(out, state.removal_count);
    put_u64(out, state.parent_node_value_count);
    put_u64(out, state.parent_predicate_count);
    put_u64(out, state.node_value_count);
    put_u64(out, state.predicate_count);
    for root in state
        .addition_roots
        .iter()
        .chain(state.removal_roots.iter())
        .chain(state.dictionary_roots.iter())
    {
        put_hash(out, root);
    }
    put_hash(out, &state.state_root);
}
fn put_index_records(out: &mut Vec<u8>, index: &AuthenticatedIndex) {
    put_u64(out, index.len() as u64);
    for record in index.iter() {
        put_u64(out, record.subject_id);
        put_u64(out, record.predicate_id);
        put_u64(out, record.object_id);
        record.triple.encode_into(out);
    }
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
fn read_state(r: &mut Reader<'_>) -> io::Result<ProofLayerState> {
    let format_version = r.u32()?;
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
    let mut roots = [[0; 32]; 11];
    for root in &mut roots {
        *root = r.hash()?;
    }
    Ok(ProofLayerState {
        format_version,
        parent_state_root,
        addition_count,
        removal_count,
        parent_node_value_count,
        parent_predicate_count,
        node_value_count,
        predicate_count,
        addition_roots: [roots[0], roots[1], roots[2], roots[3]],
        removal_roots: [roots[4], roots[5], roots[6], roots[7]],
        dictionary_roots: [roots[8], roots[9], roots[10]],
        state_root: r.hash()?,
    })
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

#[cfg(test)]
mod tests {
    use super::*;
    fn t(s: &str, p: &str, o: &str) -> CanonicalTriple {
        CanonicalTriple::node(s.as_bytes(), p.as_bytes(), o.as_bytes())
    }
    fn layer() -> ProofLayer {
        ProofLayer::new(
            None,
            vec![
                t("a", "p", "1"),
                t("a", "p", "2"),
                t("a", "q", "3"),
                t("b", "p", "4"),
            ],
            vec![],
        )
        .unwrap()
    }

    #[test]
    fn deterministic_roundtrip_and_all_indexes() {
        let a = layer();
        let b =
            ProofLayer::new(None, a.additions().iter().rev().cloned().collect(), vec![]).unwrap();
        assert_eq!(a.state, b.state);
        let encoded = a.encode().unwrap();
        assert_eq!(encoded.len(), a.encoded_len().unwrap());
        assert_eq!(ProofLayer::decode(&encoded).unwrap(), a);
        assert_ne!(a.state.addition_roots[0], a.state.addition_roots[1]);
    }
    #[test]
    fn membership_prefix_filter_and_nonmembership() {
        let l = layer();
        let prefix = TriplePattern {
            subject: Some(b"a".to_vec()),
            predicate: Some(b"p".to_vec()),
            object: None,
            ..TriplePattern::default()
        };
        let proof = l.prove(true, &prefix).unwrap();
        assert_eq!(
            proof
                .verify(l.state.addition_roots[0], &prefix)
                .unwrap()
                .len(),
            2
        );
        let filter = TriplePattern {
            subject: Some(b"a".to_vec()),
            predicate: None,
            object: Some(CanonicalObject::Node(b"3".to_vec())),
            ..TriplePattern::default()
        };
        assert_eq!(
            l.prove(true, &filter)
                .unwrap()
                .verify(l.state.addition_roots[0], &filter)
                .unwrap()
                .into_iter()
                .map(|record| record.triple)
                .collect::<Vec<_>>(),
            vec![t("a", "q", "3")]
        );
        let absent = TriplePattern {
            subject: Some(b"aa".to_vec()),
            predicate: None,
            object: None,
            ..TriplePattern::default()
        };
        assert!(l
            .prove(true, &absent)
            .unwrap()
            .verify(l.state.addition_roots[0], &absent)
            .unwrap()
            .is_empty());
    }
    #[test]
    fn adversarial_mutations_fail() {
        let l = layer();
        let p = TriplePattern {
            subject: Some(b"a".to_vec()),
            predicate: Some(b"p".to_vec()),
            object: None,
            ..TriplePattern::default()
        };
        let root = l.state.addition_roots[0];
        let mut changed = l.prove(true, &p).unwrap();
        changed.records[0].triple.object = CanonicalObject::Node(b"evil".to_vec());
        assert!(changed.verify(root, &p).is_err());
        let mut changed = l.prove(true, &p).unwrap();
        if let Some(node) = changed.frontier.first_mut() {
            node.hash[0] ^= 1;
        } else {
            changed.frontier.push(ProofNode {
                level: 63,
                index: u64::MAX,
                hash: [0; 32],
            });
        }
        assert!(changed.verify(root, &p).is_err());
        let mut changed = l.prove(true, &p).unwrap();
        changed.successor = None;
        assert!(changed.verify(root, &p).is_err());
        let mut changed = l.prove(true, &p).unwrap();
        changed.start += 1;
        assert!(changed.verify(root, &p).is_err());
        let mut changed = l.prove(true, &p).unwrap();
        changed.frontier.push(ProofNode {
            level: 2,
            index: 0,
            hash: l.addition_indexes[0].levels[2][0],
        });
        assert!(changed.verify(root, &p).is_err());
        let mut changed = l.prove(true, &p).unwrap();
        changed.leaf_count = MAX_WITNESS_LEAVES + 1;
        assert!(changed.verify(root, &p).is_err());
    }
    #[test]
    fn overlay_tombstones_and_unsupported_reverse_fail_closed() {
        let base = ProofLayer::new(None, vec![t("a", "p", "1"), t("b", "p", "2")], vec![]).unwrap();
        let child = ProofLayer::new(
            Some(base.state.state_root),
            vec![t("c", "p", "3")],
            vec![t("a", "p", "1")],
        )
        .unwrap();
        let p = TriplePattern::default();
        let witnesses = vec![
            LayerRangeWitness {
                state: base.state.clone(),
                dictionary: vec![],
                additions: Some(base.prove(true, &p).unwrap()),
                removals: Some(base.prove(false, &p).unwrap()),
            },
            LayerRangeWitness {
                state: child.state.clone(),
                dictionary: vec![],
                additions: Some(child.prove(true, &p).unwrap()),
                removals: Some(child.prove(false, &p).unwrap()),
            },
        ];
        let forward = verify_layer_chain(child.state.state_root, &p, &witnesses).unwrap();
        assert_eq!(
            forward.iter().cloned().collect::<BTreeSet<_>>(),
            [t("b", "p", "2"), t("c", "p", "3")]
                .iter()
                .cloned()
                .collect()
        );
        let mut wrong_index = witnesses.clone();
        wrong_index[0].additions.as_mut().unwrap().index = ProofIndex::SubjectDictionary;
        assert!(verify_layer_chain(child.state.state_root, &p, &wrong_index).is_err());
        let reverse = TriplePattern {
            reverse: true,
            ..TriplePattern::default()
        };
        assert!(base.prove(true, &reverse).is_err());
    }
    #[test]
    fn overlay_rejects_noop_transitions() {
        let base = ProofLayer::new(None, vec![t("a", "p", "1")], vec![]).unwrap();
        let child =
            ProofLayer::new(Some(base.state.state_root), vec![t("a", "p", "1")], vec![]).unwrap();
        let pattern = TriplePattern::default();
        let witnesses = vec![
            LayerRangeWitness {
                state: base.state.clone(),
                dictionary: vec![],
                additions: Some(base.prove(true, &pattern).unwrap()),
                removals: Some(base.prove(false, &pattern).unwrap()),
            },
            LayerRangeWitness {
                state: child.state.clone(),
                dictionary: vec![],
                additions: Some(child.prove(true, &pattern).unwrap()),
                removals: Some(child.prove(false, &pattern).unwrap()),
            },
        ];
        assert!(verify_layer_chain(child.state.state_root, &pattern, &witnesses).is_err());
    }
    #[test]
    fn persisted_index_mutation_fails_validation() {
        let mut l = layer();
        Arc::make_mut(&mut l.addition_indexes[1].order).swap(0, 1);
        assert!(l.validate().is_err());
    }

    #[test]
    fn complete_dictionary_catalog_proves_unused_bindings_and_absence() {
        let l = ProofLayer::new_native_with_catalog(
            None,
            vec![],
            vec![],
            0,
            0,
            3,
            1,
            vec![
                (CanonicalObject::Node(b"a".to_vec()), 1),
                (CanonicalObject::Node(b"b".to_vec()), 2),
                (CanonicalObject::Node(b"c".to_vec()), 3),
            ],
            vec![(b"p".to_vec(), 1)],
        )
        .unwrap();
        let pattern = TriplePattern {
            subject: Some(b"b".to_vec()),
            ..TriplePattern::default()
        };
        let mut resolved = ResolvedPattern::default();
        l.contribute_resolution(&pattern, &mut resolved).unwrap();
        assert_eq!(resolved.subject, Some(2));
        let witness = LayerRangeWitness {
            state: l.state.clone(),
            dictionary: l.prove_dictionary_bindings(&pattern).unwrap(),
            additions: Some(l.prove_resolved(true, &pattern, &resolved).unwrap()),
            removals: Some(l.prove_resolved(false, &pattern, &resolved).unwrap()),
        };
        assert!(
            verify_layer_chain(l.state.state_root, &pattern, &[witness.clone()])
                .unwrap()
                .is_empty()
        );

        let absent = TriplePattern {
            subject: Some(b"bb".to_vec()),
            ..TriplePattern::default()
        };
        let absent_witness = LayerRangeWitness {
            state: l.state.clone(),
            dictionary: l.prove_dictionary_bindings(&absent).unwrap(),
            additions: None,
            removals: None,
        };
        assert!(
            verify_layer_chain(l.state.state_root, &absent, &[absent_witness])
                .unwrap()
                .is_empty()
        );

        let mut bad_id = witness.clone();
        bad_id.dictionary[0].records[0].subject_id = 3;
        assert!(verify_layer_chain(l.state.state_root, &pattern, &[bad_id]).is_err());
        let mut bad_key = witness.clone();
        bad_key.dictionary[0].key = DictionaryKey::Subject(b"c".to_vec());
        assert!(verify_layer_chain(l.state.state_root, &pattern, &[bad_key]).is_err());
        let mut bad_boundary = witness.clone();
        bad_boundary.dictionary[0]
            .predecessor
            .as_mut()
            .unwrap()
            .record
            .triple
            .subject = b"z".to_vec();
        assert!(verify_layer_chain(l.state.state_root, &pattern, &[bad_boundary]).is_err());
        let mut bad_frontier = witness.clone();
        bad_frontier.dictionary[0].frontier[0].hash[0] ^= 1;
        assert!(verify_layer_chain(l.state.state_root, &pattern, &[bad_frontier]).is_err());
        let mut bad_resolution = witness;
        bad_resolution.additions.as_mut().unwrap().resolved.subject = Some(3);
        assert!(verify_layer_chain(l.state.state_root, &pattern, &[bad_resolution]).is_err());

        assert!(ProofLayer::new_native_with_catalog(
            None,
            vec![],
            vec![],
            0,
            0,
            2,
            0,
            vec![
                (CanonicalObject::Node(b"a".to_vec()), 1),
                (CanonicalObject::Node(b"b".to_vec()), 1),
            ],
            vec![],
        )
        .is_err());

        // Production catalogs are exactly the new contiguous id intervals:
        // gaps, inherited ids, and canonical-value mismatches all fail closed.
        assert!(ProofLayer::new_native_with_catalog(
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
        assert!(ProofLayer::new_native_with_catalog(
            Some([7; 32]),
            vec![],
            vec![],
            1,
            0,
            2,
            0,
            vec![
                (CanonicalObject::Node(b"inherited".to_vec()), 1),
                (CanonicalObject::Node(b"new".to_vec()), 2),
            ],
            vec![],
        )
        .is_err());
        let record = ProofRecord {
            subject_id: 1,
            predicate_id: 1,
            object_id: 2,
            triple: t("a", "p", "actual"),
        };
        assert!(ProofLayer::new_native_with_catalog(
            None,
            vec![record],
            vec![],
            0,
            0,
            2,
            1,
            vec![
                (CanonicalObject::Node(b"a".to_vec()), 1),
                (CanonicalObject::Node(b"different".to_vec()), 2),
            ],
            vec![(b"p".to_vec(), 1)],
        )
        .is_err());
    }

    #[test]
    fn datetime_interval_exact_values_roundtrip_but_ranges_fail_closed() {
        let triple = CanonicalTriple {
            subject: b"event".to_vec(),
            predicate: b"during".to_vec(),
            object: CanonicalObject::Value {
                datatype: CanonicalDatatype::DateTimeInterval,
                lexical: b"2020-01-01/2020-02-01".to_vec(),
            },
        };
        let layer = ProofLayer::new(None, vec![triple.clone()], vec![]).unwrap();
        assert_eq!(ProofLayer::decode(&layer.encode().unwrap()).unwrap(), layer);
        let exact = TriplePattern {
            object: Some(triple.object.clone()),
            ..TriplePattern::default()
        };
        assert_eq!(layer.prove(true, &exact).unwrap().records.len(), 1);

        let unsupported_range = TriplePattern {
            value_range: Some(CanonicalValueRange {
                datatype: CanonicalDatatype::DateTimeInterval,
                low: b"2020-01-01/2020-02-01".to_vec(),
                high: b"2021-01-01/2021-02-01".to_vec(),
            }),
            ..TriplePattern::default()
        };
        assert!(layer.prove(true, &unsupported_range).is_err());
    }

    #[test]
    fn huge_declared_trees_are_rejected_with_sparse_verifier_work() {
        let boundary = ProofRecord {
            subject_id: 1,
            predicate_id: 1,
            object_id: 1,
            triple: t("a", "p", "o"),
        };
        let pattern = TriplePattern {
            subject: Some(b"z".to_vec()),
            ..TriplePattern::default()
        };
        let range = RangeProof {
            index: ProofIndex::Spo,
            leaf_count: MAX_WITNESS_LEAVES,
            start: MAX_WITNESS_LEAVES,
            end: MAX_WITNESS_LEAVES,
            resolved: ResolvedPattern {
                subject: Some(2),
                ..ResolvedPattern::default()
            },
            records: vec![],
            predecessor: Some(IndexedTriple {
                index: MAX_WITNESS_LEAVES - 1,
                record: boundary.clone(),
            }),
            successor: None,
            frontier: vec![],
        };
        assert!(range.verify([0; 32], &pattern).is_err());

        let dictionary = DictionaryProof {
            key: DictionaryKey::Subject(b"z".to_vec()),
            leaf_count: MAX_WITNESS_LEAVES,
            start: MAX_WITNESS_LEAVES,
            records: vec![],
            predecessor: Some(IndexedTriple {
                index: MAX_WITNESS_LEAVES - 1,
                record: boundary,
            }),
            successor: None,
            frontier: vec![],
        };
        assert!(dictionary.verify([0; 32]).is_err());
    }

    #[test]
    fn decode_rejects_projected_heap_and_truncation_before_record_allocation() {
        let layer = layer();
        let mut bytes = layer.encode().unwrap();
        let checksum_position = bytes.len() - 32;
        let first_table_count = MAGIC.len() + 4 + 4 + 1 + (6 * 8) + (11 * 32) + 32;

        bytes[first_table_count..first_table_count + 8]
            .copy_from_slice(&2_000_000u64.to_be_bytes());
        let checksum = hash_parts(
            b"terminus-store/proof/file/v4",
            &[&bytes[..checksum_position]],
        );
        bytes[checksum_position..].copy_from_slice(&checksum);
        let error = ProofLayer::decode(&bytes).unwrap_err();
        assert!(error.to_string().contains("heap budget"));

        bytes[first_table_count..first_table_count + 8]
            .copy_from_slice(&1_000_000u64.to_be_bytes());
        let checksum = hash_parts(
            b"terminus-store/proof/file/v4",
            &[&bytes[..checksum_position]],
        );
        bytes[checksum_position..].copy_from_slice(&checksum);
        let error = ProofLayer::decode(&bytes).unwrap_err();
        assert!(error.to_string().contains("truncated proof record table"));
    }

    #[test]
    fn encode_and_incremental_witness_budgets_fail_before_growth() {
        assert!(checked_proof_encoded_growth(MAX_PROOF_LAYER_BYTES, 1).is_err());
        assert!(checked_proof_encoded_growth(usize::MAX, 1).is_err());

        let record = ProofRecord {
            subject_id: 1,
            predicate_id: 1,
            object_id: 1,
            triple: t("a", "p", "o"),
        };
        let mut records = WitnessBudget::new(1).unwrap();
        records.records = MAX_WITNESS_RECORDS;
        assert!(records
            .reserve_proof(std::iter::once(&record), None, None, &[], 0)
            .is_err());
        assert_eq!(records.records, MAX_WITNESS_RECORDS);

        let mut frontier = WitnessBudget::new(1).unwrap();
        frontier.frontier = MAX_WITNESS_FRONTIER;
        assert!(frontier
            .reserve_proof(
                std::iter::empty(),
                None,
                None,
                &[ProofNode {
                    level: 0,
                    index: 0,
                    hash: [0; 32],
                }],
                0,
            )
            .is_err());
        assert_eq!(frontier.frontier, MAX_WITNESS_FRONTIER);

        let mut bytes = WitnessBudget::new(1).unwrap();
        bytes.bytes = MAX_WITNESS_BYTES;
        assert!(bytes
            .reserve_proof(std::iter::empty(), None, None, &[], 1)
            .is_err());
        assert_eq!(bytes.bytes, MAX_WITNESS_BYTES);
    }

    #[test]
    fn encoder_and_construction_budget_reject_oversized_canonical_fields() {
        let oversized = vec![b'x'; MAX_FIELD_BYTES + 1];
        let proof = ProofLayer::new(
            None,
            vec![CanonicalTriple::node(
                oversized.clone(),
                b"p".to_vec(),
                b"o".to_vec(),
            )],
            vec![],
        )
        .unwrap();
        assert!(proof
            .encode()
            .unwrap_err()
            .to_string()
            .contains("field too large"));

        let record = ProofRecord {
            subject_id: 1,
            predicate_id: 1,
            object_id: 1,
            triple: CanonicalTriple::node(b"s".to_vec(), b"p".to_vec(), oversized),
        };
        let mut budget = ProofConstructionBudget::new();
        assert!(budget
            .record(&record)
            .unwrap_err()
            .to_string()
            .contains("field too large"));
    }

    #[test]
    fn reader_charges_dynamic_heap_before_copying_field() {
        let mut encoded = Vec::new();
        put_u64(&mut encoded, 1);
        encoded.push(b'x');
        let mut reader = Reader::new(&encoded);
        reader.decoded_heap_estimate = MAX_DECODED_PROOF_BYTES - 1;
        assert!(reader
            .bytes()
            .unwrap_err()
            .to_string()
            .contains("heap budget"));
        assert_eq!(reader.position, 8);
    }

    #[test]
    fn corrupted_file_and_manifest_fail() {
        let l = layer();
        let mut bytes = l.encode().unwrap();
        let middle = bytes.len() / 2;
        bytes[middle] ^= 1;
        assert!(ProofLayer::decode(&bytes).is_err());
        let mut state = l.state.clone();
        state.addition_count += 1;
        assert!(state.validate().is_err());
    }
}
