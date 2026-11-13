//! Whole-dictionary FACTS and the TERM-SORTED dictionary index of a base layer — the two root
//! additions that make every single-layer read result-sized over an ATTESTED root
//! (`doc/proof-of-woql/result-sized-proofs.md` §3.2–§3.4).
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
//! The id-indexed Merkle trees (`dict_merkle`) prove `id -> term`, not `term -> id`, and cannot
//! prove a term ABSENT. Per namespace (node/value, predicate) a base layer also binds a Merkle tree
//! whose leaves are `(key(term), id)` sorted by `key`, where `key` is the store's existing term
//! hash (`hash_object` for node/value, `hash_term` for predicates) as a 32-byte big-endian integer.
//! Position 0 holds a LOW sentinel, positions `1..=n` the `n` real entries in strictly increasing
//! key order, position `n + 1` a HIGH sentinel; the row is padded to a power of two.
//!
//! * MEMBERSHIP of `c`: the real leaf `(key(c), id)` and its path give `c`'s id.
//! * NON-MEMBERSHIP of `c`: two leaves at consecutive positions `i, i + 1` with
//!   `left < key(c) < right` (a sentinel is below/above every key) and their paths.
//!
//! The verifier derives the key from the PUBLIC constant itself and the id from the opened leaf —
//! never a prover-claimed id — before any transcript is drawn. Leaf kinds are pinned to their
//! positions (LOW only at 0, HIGH only at `n + 1`, real only at `1..=n`), so a sentinel cannot be
//! moved inward to fake an absence.
//!
//! Sortedness and agreement of this index with the id-indexed trees and the Dory dictionary
//! columns are WELL-FORMEDNESS properties of the root: a verifier relies on them only for a root
//! whose well-formedness the caller vouches for ([`TrustedRoot::Attested`](crate::proof::envelope::TrustedRoot)),
//! i.e. a root the validators recomputed from the data. A term-bearing verifier recomputes the
//! index from the clear catalogs in `validate`.

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

/// One positional opening of a sorted-index leaf.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SortedLeafOpening {
    pub leaf: SortedLeaf,
    pub position: u64,
    pub siblings: Vec<ProofHash>,
}

/// The witness for one `term -> id` lookup in a sorted index: the term's leaf, or the adjacent
/// pair that brackets its key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SortedLookup {
    Member(SortedLeafOpening),
    Absent(SortedLeafOpening, SortedLeafOpening),
}

/// A term-sorted index tree (prover side, and the term-bearing recompute).
pub(crate) struct SortedIndexTree {
    levels: Vec<Vec<ProofHash>>,
    /// The `n + 2` meaningful leaves in position order (sentinels included).
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

    fn open_position(&self, position: usize) -> SortedLeafOpening {
        let height = self.levels.len() - 1;
        let mut siblings = Vec::with_capacity(height);
        let mut at = position;
        for level in &self.levels[..height] {
            siblings.push(level[at ^ 1]);
            at >>= 1;
        }
        SortedLeafOpening { leaf: self.leaves[position], position: position as u64, siblings }
    }

    /// The lookup witness for `key`: its leaf if present, else the bracketing adjacent pair.
    pub(crate) fn lookup(&self, key: &ProofHash) -> SortedLookup {
        // Real leaves occupy positions 1..=n, sorted.
        let real = &self.leaves[1..self.leaves.len() - 1];
        match real.binary_search_by(|leaf| match leaf {
            SortedLeaf::Real { key: k, .. } => k.cmp(key),
            _ => unreachable!("only real leaves lie between the sentinels"),
        }) {
            Ok(index) => SortedLookup::Member(self.open_position(index + 1)),
            // `index` real keys are below `key`: the bracket is positions (index, index + 1).
            Err(index) => {
                SortedLookup::Absent(self.open_position(index), self.open_position(index + 1))
            }
        }
    }
}

/// Verify one leaf opening against `root` for an index of `n` real entries, pinning the leaf kind
/// to its position. Returns nothing; the caller reads the (now authenticated) leaf.
fn verify_leaf_opening(root: &ProofHash, n: u64, opening: &SortedLeafOpening) -> io::Result<()> {
    let (_, height) = sorted_shape(n)?;
    let last = n
        .checked_add(1)
        .ok_or_else(|| invalid("sorted dictionary index size overflow"))?;
    let kind_ok = match opening.leaf {
        SortedLeaf::Low => opening.position == 0,
        SortedLeaf::High => opening.position == last,
        SortedLeaf::Real { id, .. } => {
            opening.position >= 1 && opening.position <= n && id >= 1 && id <= n
        }
    };
    if !kind_ok {
        return Err(invalid("sorted-index leaf kind does not match its position"));
    }
    if opening.siblings.len() != height {
        return Err(invalid("sorted-index opening path has the wrong height"));
    }
    let mut hash = opening.leaf.hash();
    let mut position = opening.position;
    for sibling in &opening.siblings {
        hash = if position & 1 == 0 {
            sorted_node_hash(&hash, sibling)
        } else {
            sorted_node_hash(sibling, &hash)
        };
        position >>= 1;
    }
    if position != 0 || &hash != root {
        return Err(invalid("sorted-index opening does not authenticate against the bound root"));
    }
    Ok(())
}

/// Verify a sorted-index lookup for the verifier-computed `key` against the root-bound `root` of an
/// index over `n` real entries (a base layer's dictionary count). `Ok(Some(id))` = present with that
/// id, `Ok(None)` = absent; any malformed, mis-positioned, non-adjacent, out-of-order or foreign
/// opening is an `Err`. Pure — never touches a transcript.
pub(crate) fn verify_sorted_lookup(
    root: &ProofHash,
    n: u64,
    key: &ProofHash,
    lookup: &SortedLookup,
) -> io::Result<Option<u64>> {
    match lookup {
        SortedLookup::Member(opening) => {
            let SortedLeaf::Real { key: leaf_key, id } = opening.leaf else {
                return Err(invalid("sorted-index membership must open a real leaf"));
            };
            if &leaf_key != key {
                return Err(invalid("sorted-index membership leaf is for another key"));
            }
            verify_leaf_opening(root, n, opening)?;
            Ok(Some(id))
        }
        SortedLookup::Absent(left, right) => {
            if left.position.checked_add(1) != Some(right.position) {
                return Err(invalid("sorted-index absence pair is not adjacent"));
            }
            let below = match left.leaf {
                SortedLeaf::Low => true,
                SortedLeaf::Real { key: leaf_key, .. } => &leaf_key < key,
                SortedLeaf::High => false,
            };
            let above = match right.leaf {
                SortedLeaf::High => true,
                SortedLeaf::Real { key: leaf_key, .. } => key < &leaf_key,
                SortedLeaf::Low => false,
            };
            if !below || !above {
                return Err(invalid("sorted-index absence pair does not bracket the key"));
            }
            verify_leaf_opening(root, n, left)?;
            verify_leaf_opening(root, n, right)?;
            Ok(None)
        }
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

    fn catalog(n: u64) -> Vec<ProofRecord> {
        (1..=n).map(|id| node(id, &format!("ex:n{id}"))).collect()
    }

    fn key_of(iri: &str) -> ProofHash {
        node_value_sort_key(&CanonicalObject::Node(iri.as_bytes().to_vec()))
    }

    #[test]
    fn membership_and_absence_round_trip() {
        for n in [0u64, 1, 2, 3, 6, 7] {
            let tree = SortedIndexTree::node_value(&catalog(n)).unwrap();
            for id in 1..=n {
                let key = key_of(&format!("ex:n{id}"));
                assert_eq!(verify_sorted_lookup(&tree.root(), n, &key, &tree.lookup(&key)).unwrap(), Some(id));
            }
            for absent in ["ex:missing", "ex:other", "", "ex:n0"] {
                let key = key_of(absent);
                let lookup = tree.lookup(&key);
                assert!(matches!(lookup, SortedLookup::Absent(..)));
                assert_eq!(verify_sorted_lookup(&tree.root(), n, &key, &lookup).unwrap(), None);
            }
        }
    }

    /// Absence below the smallest and above the largest key is provable (the sentinel brackets).
    #[test]
    fn absence_at_both_ends_uses_the_sentinels() {
        let tree = SortedIndexTree::node_value(&catalog(5)).unwrap();
        let below = [0u8; 32];
        let above = [0xffu8; 32];
        match tree.lookup(&below) {
            SortedLookup::Absent(left, _) => assert_eq!(left.leaf, SortedLeaf::Low),
            other => panic!("{other:?}"),
        }
        match tree.lookup(&above) {
            SortedLookup::Absent(_, right) => assert_eq!(right.leaf, SortedLeaf::High),
            other => panic!("{other:?}"),
        }
        assert_eq!(verify_sorted_lookup(&tree.root(), 5, &below, &tree.lookup(&below)).unwrap(), None);
        assert_eq!(verify_sorted_lookup(&tree.root(), 5, &above, &tree.lookup(&above)).unwrap(), None);
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
    fn wrong_id_is_rejected() {
        let tree = SortedIndexTree::node_value(&catalog(6)).unwrap();
        let key = key_of("ex:n3");
        let SortedLookup::Member(mut opening) = tree.lookup(&key) else { panic!() };
        opening.leaf = SortedLeaf::Real { key, id: 4 };
        assert!(verify_sorted_lookup(&tree.root(), 6, &key, &SortedLookup::Member(opening)).is_err());
    }

    #[test]
    fn non_adjacent_pair_is_rejected() {
        let tree = SortedIndexTree::node_value(&catalog(6)).unwrap();
        let key = key_of("ex:missing");
        let SortedLookup::Absent(left, right) = tree.lookup(&key) else { panic!() };
        // Widen the honest bracket by one position on one side: it still brackets the key in
        // order, only adjacency fails.
        let (wide_left, wide_right) = if left.position > 0 {
            (tree.open_position(left.position as usize - 1), right)
        } else {
            (left, tree.open_position(right.position as usize + 1))
        };
        assert!(verify_sorted_lookup(&tree.root(), 6, &key, &SortedLookup::Absent(wide_left, wide_right)).is_err());
    }

    #[test]
    fn out_of_order_pair_is_rejected() {
        let tree = SortedIndexTree::node_value(&catalog(6)).unwrap();
        let key = key_of("ex:missing");
        let SortedLookup::Absent(left, right) = tree.lookup(&key) else { panic!() };
        // Swapped: right first, left second (not adjacent ascending, and not bracketing).
        assert!(verify_sorted_lookup(&tree.root(), 6, &key, &SortedLookup::Absent(right.clone(), left.clone())).is_err());
        // An adjacent pair elsewhere does not bracket this key.
        let elsewhere = (1..=5u64)
            .map(|p| (tree.open_position(p as usize), tree.open_position(p as usize + 1)))
            .find(|(l, _)| l.position != left.position)
            .unwrap();
        assert!(verify_sorted_lookup(&tree.root(), 6, &key, &SortedLookup::Absent(elsewhere.0, elsewhere.1)).is_err());
    }

    #[test]
    fn pair_from_another_root_is_rejected() {
        let tree = SortedIndexTree::node_value(&catalog(6)).unwrap();
        let other = SortedIndexTree::node_value(&catalog(5)).unwrap();
        let key = key_of("ex:missing");
        assert!(verify_sorted_lookup(&tree.root(), 6, &key, &other.lookup(&key)).is_err());
        let member = key_of("ex:n2");
        assert!(verify_sorted_lookup(&tree.root(), 6, &member, &other.lookup(&member)).is_err());
    }

    #[test]
    fn sentinel_abuse_is_rejected() {
        let tree = SortedIndexTree::node_value(&catalog(6)).unwrap();
        let keys = real_keys(&tree);
        // Claim a present key (the 3rd smallest) absent by presenting a LOW sentinel moved inward:
        // the authentic path of position 3 with the leaf relabeled LOW.
        let present = keys[2].0;
        let mut fake_low = tree.open_position(3);
        fake_low.leaf = SortedLeaf::Low;
        let right = tree.open_position(4);
        assert!(verify_sorted_lookup(&tree.root(), 6, &present, &SortedLookup::Absent(fake_low, right)).is_err());
        // A HIGH sentinel claimed at an inner position.
        let left = tree.open_position(2);
        let mut fake_high = tree.open_position(3);
        fake_high.leaf = SortedLeaf::High;
        assert!(verify_sorted_lookup(&tree.root(), 6, &present, &SortedLookup::Absent(left, fake_high)).is_err());
        // A membership claim on a sentinel.
        assert!(verify_sorted_lookup(&tree.root(), 6, &present, &SortedLookup::Member(tree.open_position(0))).is_err());
        // A padding slot (position n+2..) is never openable: n = 6 → 8 leaves, positions 0..=7.
        assert_eq!(tree.leaves.len(), 8);
    }

    #[test]
    fn a_present_term_claimed_absent_is_rejected() {
        let tree = SortedIndexTree::node_value(&catalog(6)).unwrap();
        let keys = real_keys(&tree);
        for (i, (key, _)) in keys.iter().enumerate() {
            // Every adjacent pair around the present key's position fails to bracket it strictly.
            for left_pos in [i, i + 1] {
                let pair = SortedLookup::Absent(tree.open_position(left_pos), tree.open_position(left_pos + 1));
                assert!(verify_sorted_lookup(&tree.root(), 6, key, &pair).is_err());
            }
        }
    }

    #[test]
    fn wrong_height_or_count_is_rejected() {
        let tree = SortedIndexTree::node_value(&catalog(6)).unwrap();
        let key = key_of("ex:n1");
        let lookup = tree.lookup(&key);
        // n = 6 vs claimed 2 (a 4-leaf tree): height mismatch.
        assert!(verify_sorted_lookup(&tree.root(), 2, &key, &lookup).is_err());
        let SortedLookup::Member(mut short) = lookup else { panic!() };
        short.siblings.pop();
        assert!(verify_sorted_lookup(&tree.root(), 6, &key, &SortedLookup::Member(short)).is_err());
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
