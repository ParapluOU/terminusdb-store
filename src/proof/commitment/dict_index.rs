//! Whole-dictionary FACTS and the TERM-SORTED dictionary index of a base layer — the two root
//! additions of `doc/proof-of-woql/result-sized-proofs.md` §3.2–§3.4.
//!
//! # Facts
//!
//! Some verifier guards read a property of the WHOLE dictionary, not of the rows a query touches:
//! a numeric range/order/aggregate over a float datatype is unfaithful to WOQL if ANY entry of that
//! datatype is a NaN (`reject_nan_catalog`). A term-less verifier has no catalog to scan, so a base
//! layer binds the answer as a small [`DictFacts`] record computed once from its complete
//! dictionary. A delta layer has no facts (its catalog is an increment); a guard over a delta fails
//! closed exactly as before.
//!
//! # Term-sorted index
//!
//! Per namespace (node/value, predicate) a base layer also binds the root of a Merkle tree whose
//! leaves are `(key(term), id)` sorted by `key`, where `key` is the store's existing term hash
//! (`hash_object` for node/value, `hash_term` for predicates) as a 32-byte big-endian integer.
//! Position 0 holds a LOW sentinel, positions `1..=n` the `n` real entries in strictly increasing
//! key order, position `n + 1` a HIGH sentinel; the row is padded to a power of two.
//!
//! The index was the `term -> id` witness of the retired attested squashed head
//! (`result-sized-proofs.md` §3.3). No verifier opens it any more: term lookups over a term-less
//! view go through a proven checkpoint's keyed term index (`checkpoint::smt`), which the
//! checkpoint view places in these two root slots. A base layer still binds the sorted roots (a
//! term-bearing verifier recomputes them from the clear catalogs in `validate`), so dropping them
//! is a `resolution_root` format change.

use super::*;
use ark_ff::{BigInteger, PrimeField};

/// Domain of a sorted-index LEAF preimage (kind byte follows).
const SORTED_LEAF_DOMAIN: &[u8] = b"terminus-store/pow/dict-sorted-leaf/v1";
/// Domain of a sorted-index internal node (distinct from the id-indexed tree's node domain).
const SORTED_NODE_DOMAIN: &[u8] = b"terminus-store/pow/dict-sorted-node/v1";
/// Domain of the [`DictFacts`] hash folded into `resolution_root` v2.
const FACTS_DOMAIN: &[u8] = b"terminus-store/pow/dict-facts/v1";

const KIND_LOW: u8 = 0;
const KIND_REAL: u8 = 1;
const KIND_HIGH: u8 = 2;
const KIND_PAD: u8 = 3;

/// Whole-dictionary facts a verifier guard reads instead of scanning the catalog.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct DictFacts {
    /// Some `xsd:float` entry of the node/value dictionary is a NaN.
    pub float32_nan: bool,
    /// Some `xsd:double` entry of the node/value dictionary is a NaN.
    pub float64_nan: bool,
}

impl DictFacts {
    /// Compute the facts of a COMPLETE node/value dictionary.
    pub(crate) fn from_catalog(node_value_catalog: &[ProofRecord]) -> Self {
        let f32_tag = ArkFr::from_u64(CanonicalDatatype::Float32 as u64);
        let f64_tag = ArkFr::from_u64(CanonicalDatatype::Float64 as u64);
        let mut facts = Self::default();
        for record in node_value_catalog {
            facts.float32_nan |= object_is_selected_nan(&record.triple.object, f32_tag);
            facts.float64_nan |= object_is_selected_nan(&record.triple.object, f64_tag);
        }
        facts
    }

    /// The facts of a set of node/value terms (the checkpoint's append: the facts of the new
    /// entries, OR-ed into the facts of the dictionary they extend).
    pub(crate) fn from_objects<'a>(objects: impl IntoIterator<Item = &'a CanonicalObject>) -> Self {
        let f32_tag = ArkFr::from_u64(CanonicalDatatype::Float32 as u64);
        let f64_tag = ArkFr::from_u64(CanonicalDatatype::Float64 as u64);
        let mut facts = Self::default();
        for object in objects {
            facts.float32_nan |= object_is_selected_nan(object, f32_tag);
            facts.float64_nan |= object_is_selected_nan(object, f64_tag);
        }
        facts
    }

    /// The facts of the union of two dictionaries.
    pub(crate) fn union(self, other: Self) -> Self {
        Self {
            float32_nan: self.float32_nan | other.float32_nan,
            float64_nan: self.float64_nan | other.float64_nan,
        }
    }

    /// The one-byte wire/hash form: bit 0 = float32 NaN, bit 1 = float64 NaN.
    pub(crate) fn to_byte(self) -> u8 {
        (self.float32_nan as u8) | ((self.float64_nan as u8) << 1)
    }

    /// Inverse of [`Self::to_byte`]; any other bit set is malformed.
    pub(crate) fn from_byte(byte: u8) -> io::Result<Self> {
        if byte & !0b11 != 0 {
            return Err(invalid("malformed dictionary facts byte"));
        }
        Ok(Self { float32_nan: byte & 1 != 0, float64_nan: byte & 2 != 0 })
    }

    /// `SHA256(FACTS_DOMAIN ‖ byte)` — what `resolution_root` v2 binds.
    pub fn hash(self) -> ProofHash {
        let mut hasher = Sha256::new();
        hasher.update(FACTS_DOMAIN);
        hasher.update([self.to_byte()]);
        hasher.finalize().into()
    }

    /// The NaN guard over the facts: the replacement for [`reject_nan_catalog`] on an attested
    /// term-less view. A non-float `datatype_tag` is a no-op, exactly as there.
    pub(crate) fn reject_nan(self, datatype_tag: ArkFr) -> io::Result<()> {
        let nan = if datatype_tag == ArkFr::from_u64(CanonicalDatatype::Float32 as u64) {
            self.float32_nan
        } else if datatype_tag == ArkFr::from_u64(CanonicalDatatype::Float64 as u64) {
            self.float64_nan
        } else {
            false
        };
        if nan {
            return Err(invalid(
                "float range/group proof cannot match Prolog semantics in a catalog containing NaN",
            ));
        }
        Ok(())
    }
}

/// What a base/rollup/squash layer binds beyond its id-indexed trees: the facts and the two
/// term-sorted index roots. `None` on a delta layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DictIndex {
    pub facts: DictFacts,
    pub nv_sorted_root: ProofHash,
    pub pred_sorted_root: ProofHash,
}

impl DictIndex {
    /// Compute the index of a BASE layer from its complete catalogs (ids dense from 1).
    pub(crate) fn from_catalogs(
        node_value_catalog: &[ProofRecord],
        predicate_catalog: &[ProofRecord],
    ) -> io::Result<Self> {
        Ok(Self {
            facts: DictFacts::from_catalog(node_value_catalog),
            nv_sorted_root: SortedIndexTree::node_value(node_value_catalog)?.root(),
            pred_sorted_root: SortedIndexTree::predicate(predicate_catalog)?.root(),
        })
    }

    /// The index a layer with this `state` must carry: `Some` for a base layer (computed from its
    /// catalogs), `None` for a delta.
    pub(crate) fn for_layer(
        state_parent: Option<ProofHash>,
        node_value_catalog: &[ProofRecord],
        predicate_catalog: &[ProofRecord],
    ) -> io::Result<Option<Self>> {
        match state_parent {
            None => Ok(Some(Self::from_catalogs(node_value_catalog, predicate_catalog)?)),
            Some(_) => Ok(None),
        }
    }
}

/// Which sorted index a key is looked up in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SortedNamespace {
    NodeValue,
    Predicate,
}

/// The sorted-index key of a node/value term: `hash_object` as a 32-byte big-endian integer.
pub fn node_value_sort_key(object: &CanonicalObject) -> ProofHash {
    field_key(crate::proof::dict::hash_object(object))
}

/// The sorted-index key of a predicate IRI: `hash_term` as a 32-byte big-endian integer.
pub fn predicate_sort_key(iri: &[u8]) -> ProofHash {
    field_key(crate::proof::dict::hash_term(iri))
}

fn field_key(value: ArkFr) -> ProofHash {
    let bytes = value.0.into_bigint().to_bytes_be();
    let mut key = [0u8; 32];
    key[32 - bytes.len()..].copy_from_slice(&bytes);
    key
}

/// One sorted-index leaf.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SortedLeaf {
    /// Below every key; only at position 0.
    Low,
    /// A real `(key, id)` entry; only at positions `1..=n`.
    Real { key: ProofHash, id: u64 },
    /// Above every key; only at position `n + 1`.
    High,
}

impl SortedLeaf {
    fn hash(&self) -> ProofHash {
        let mut hasher = Sha256::new();
        hasher.update(SORTED_LEAF_DOMAIN);
        match self {
            SortedLeaf::Low => hasher.update([KIND_LOW]),
            SortedLeaf::Real { key, id } => {
                hasher.update([KIND_REAL]);
                hasher.update(key);
                hasher.update(id.to_be_bytes());
            }
            SortedLeaf::High => hasher.update([KIND_HIGH]),
        }
        hasher.finalize().into()
    }
}

fn pad_leaf() -> ProofHash {
    let mut hasher = Sha256::new();
    hasher.update(SORTED_LEAF_DOMAIN);
    hasher.update([KIND_PAD]);
    hasher.finalize().into()
}

fn sorted_node_hash(left: &ProofHash, right: &ProofHash) -> ProofHash {
    let mut hasher = Sha256::new();
    hasher.update(SORTED_NODE_DOMAIN);
    hasher.update(left);
    hasher.update(right);
    hasher.finalize().into()
}

/// The padded leaf count and tree height for `n` real entries (`n + 2` leaves with sentinels).
fn sorted_shape(n: u64) -> io::Result<(usize, usize)> {
    let leaves = usize::try_from(n)
        .ok()
        .and_then(|n| n.checked_add(2))
        .and_then(usize::checked_next_power_of_two)
        .ok_or_else(|| invalid("sorted dictionary index size overflow"))?;
    Ok((leaves, leaves.trailing_zeros() as usize))
}

/// A term-sorted index tree (prover side, and the term-bearing recompute).
pub(crate) struct SortedIndexTree {
    levels: Vec<Vec<ProofHash>>,
    /// The `n + 2` meaningful leaves in position order (sentinels included).
    #[cfg_attr(not(test), allow(dead_code))]
    leaves: Vec<SortedLeaf>,
}

impl SortedIndexTree {
    fn build(mut entries: Vec<(ProofHash, u64)>) -> io::Result<Self> {
        entries.sort();
        if entries.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(invalid("two dictionary terms share a sorted-index key"));
        }
        let n = entries.len() as u64;
        let (padded, _) = sorted_shape(n)?;
        let mut leaves = Vec::with_capacity(entries.len() + 2);
        leaves.push(SortedLeaf::Low);
        leaves.extend(entries.into_iter().map(|(key, id)| SortedLeaf::Real { key, id }));
        leaves.push(SortedLeaf::High);
        let mut row: Vec<ProofHash> = leaves.iter().map(SortedLeaf::hash).collect();
        row.resize(padded, pad_leaf());
        let mut levels = vec![row];
        while levels.last().map(Vec::len).unwrap_or(0) > 1 {
            let next = levels
                .last()
                .unwrap()
                .chunks(2)
                .map(|pair| sorted_node_hash(&pair[0], &pair[1]))
                .collect();
            levels.push(next);
        }
        Ok(Self { levels, leaves })
    }

    /// The node/value index of a BASE catalog.
    pub(crate) fn node_value(node_value_catalog: &[ProofRecord]) -> io::Result<Self> {
        Self::build(
            node_value_catalog
                .iter()
                .map(|record| (node_value_sort_key(&record.triple.object), record.object_id))
                .collect(),
        )
    }

    /// The predicate index of a BASE catalog.
    pub(crate) fn predicate(predicate_catalog: &[ProofRecord]) -> io::Result<Self> {
        Self::build(
            predicate_catalog
                .iter()
                .map(|record| (predicate_sort_key(&record.triple.predicate), record.predicate_id))
                .collect(),
        )
    }

    pub(crate) fn root(&self) -> ProofHash {
        self.levels.last().and_then(|row| row.first()).copied().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &ProofHash) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    fn node(id: u64, iri: &str) -> ProofRecord {
        ProofRecord {
            subject_id: 0,
            predicate_id: 0,
            object_id: id,
            triple: CanonicalTriple {
                subject: Vec::new(),
                predicate: Vec::new(),
                object: CanonicalObject::Node(iri.as_bytes().to_vec()),
            },
        }
    }

    fn value(id: u64, datatype: CanonicalDatatype, lexical: Vec<u8>) -> ProofRecord {
        ProofRecord {
            subject_id: 0,
            predicate_id: 0,
            object_id: id,
            triple: CanonicalTriple {
                subject: Vec::new(),
                predicate: Vec::new(),
                object: CanonicalObject::Value { datatype, lexical },
            },
        }
    }

    fn real_keys(tree: &SortedIndexTree) -> Vec<(ProofHash, u64)> {
        tree.leaves
            .iter()
            .filter_map(|leaf| match leaf {
                SortedLeaf::Real { key, id } => Some((*key, *id)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn duplicate_keys_do_not_build() {
        assert!(SortedIndexTree::node_value(&[node(1, "ex:a"), node(2, "ex:a")]).is_err());
    }

    /// Golden facts + sorted roots for a fixed catalog (re-derive by hand if the leaf layout
    /// changes; any drift changes every base layer's `resolution_root`).
    #[test]
    fn facts_and_sorted_roots_golden() {
        let nan32 = f32::NAN.to_bits();
        let nv = vec![
            node(1, "ex:a"),
            value(2, CanonicalDatatype::UInt32, 7u32.to_be_bytes().to_vec()),
            value(3, CanonicalDatatype::Float32, (nan32 ^ 0x8000_0000).to_be_bytes().to_vec()),
        ];
        let facts = DictFacts::from_catalog(&nv);
        assert!(facts.float32_nan && !facts.float64_nan);
        assert_eq!(DictFacts::from_byte(facts.to_byte()).unwrap(), facts);
        assert!(DictFacts::from_byte(4).is_err());
        assert!(facts.reject_nan(ArkFr::from_u64(CanonicalDatatype::Float32 as u64)).is_err());
        assert!(facts.reject_nan(ArkFr::from_u64(CanonicalDatatype::Float64 as u64)).is_ok());
        assert!(facts.reject_nan(ArkFr::from_u64(CanonicalDatatype::UInt32 as u64)).is_ok());
        // Independent `sha256sum` oracles over the exact preimages.
        assert_eq!(hex(&DictFacts::default().hash()), GOLDEN_EMPTY_FACTS);
        assert_eq!(hex(&facts.hash()), GOLDEN_F32_NAN_FACTS);
        let empty = DictIndex::from_catalogs(&[], &[]).unwrap();
        assert_eq!(hex(&empty.nv_sorted_root), GOLDEN_EMPTY_SORTED_ROOT);
        assert_eq!(empty.pred_sorted_root, empty.nv_sorted_root);
        // A populated index: real leaves sit between the sentinels in key order.
        let tree = SortedIndexTree::node_value(&nv).unwrap();
        let keys = real_keys(&tree);
        assert_eq!(keys.len(), 3);
        assert!(keys.windows(2).all(|pair| pair[0].0 < pair[1].0));
        assert_eq!(DictIndex::from_catalogs(&nv, &[]).unwrap().nv_sorted_root, tree.root());
    }

    const GOLDEN_EMPTY_FACTS: &str = "9688acdf25cebcfe8f6a0a87e769c4fc4f87cbd3b39e2196732754c2fc01d97b";
    const GOLDEN_F32_NAN_FACTS: &str = "b359636dd86b1161c6f867f1bd718eaa06f6e889f79c70522752c51006e587ac";
    const GOLDEN_EMPTY_SORTED_ROOT: &str = "10c5f4168348044fd0a34e5f083bd6048d15194c51d49cb146e77a5d6bf86f21";
}
