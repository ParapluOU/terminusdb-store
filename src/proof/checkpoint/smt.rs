//! Keyed sparse Merkle trees of a checkpoint: the term index (`hash(term) -> chain id`, one per
//! namespace) and the triple index (the present triples).
//!
//! # Structure (compact sparse Merkle tree)
//!
//! Keys are 256-bit; bit `d` of a key (`0 <= d < 256`) selects the branch at depth `d`, read from
//! the LOW end of the big-endian key (the hash bits, not the always-zero top bits of a field
//! element). The hash of the subtree holding a key set `S` at depth `d` is
//!
//! * `EMPTY` when `S` is empty,
//! * `leaf(k, v)` when `S = {k}` — at ANY depth (a single key is not pushed to depth 256),
//! * `node(h(S_0), h(S_1))` otherwise, `S_b` the keys whose bit `d` is `b`.
//!
//! The three preimage kinds are domain-separated (and every hash carries a namespace byte, so a
//! path of one index never verifies in another). The tree is canonical: a key set has exactly one
//! root.
//!
//! # Paths
//!
//! The path of a key descends from the root while the subtree holds two or more keys and stops
//! at the first subtree holding at most one; it lists the siblings top-down and names the
//! terminal (`Empty` or `Leaf(k', v')`). Recomputing the root from a path and checking a leaf
//! terminal's key agrees with the path's bits is a membership proof (terminal = the key's leaf) or
//! a non-membership proof (terminal `Empty`, or another key's leaf on that prefix). Paths have
//! `O(log n)` siblings for hash-distributed keys.
//!
//! # Updates
//!
//! An INSERT verifies a non-membership path and rebuilds the path with the new leaf (splitting an
//! occupied terminal at the first differing bit). A DELETE verifies a membership path and removes
//! the leaf; to keep the tree canonical, a sibling that is itself a single leaf moves up past every
//! empty sibling above it, so the delete proof reveals the deepest sibling's preimage (a leaf or a
//! node) — a hash cannot be both, so the verifier learns which.

use std::collections::BTreeMap;
use std::io;

use sha2::{Digest, Sha256};

use crate::proof::{invalid, ProofHash};

/// Key width in bits (and the maximum path depth).
pub const SMT_KEY_BITS: usize = 256;

/// Node/value term index namespace.
pub const NS_NV_INDEX: u8 = 1;
/// Predicate term index namespace.
pub const NS_PRED_INDEX: u8 = 2;
/// Triple index namespace.
pub const NS_TRIPLE_INDEX: u8 = 3;

const EMPTY_DOMAIN: &[u8] = b"terminus-store/pow/checkpoint/smt-empty/v1";
const LEAF_DOMAIN: &[u8] = b"terminus-store/pow/checkpoint/smt-leaf/v1";
const NODE_DOMAIN: &[u8] = b"terminus-store/pow/checkpoint/smt-node/v1";

pub type SmtKey = [u8; 32];

pub fn smt_empty(namespace: u8) -> ProofHash {
    let mut hasher = Sha256::new();
    hasher.update(EMPTY_DOMAIN);
    hasher.update([namespace]);
    hasher.finalize().into()
}

pub fn smt_leaf(namespace: u8, key: &SmtKey, value: u64) -> ProofHash {
    let mut hasher = Sha256::new();
    hasher.update(LEAF_DOMAIN);
    hasher.update([namespace]);
    hasher.update(key);
    hasher.update(value.to_be_bytes());
    hasher.finalize().into()
}

pub fn smt_node(namespace: u8, left: &ProofHash, right: &ProofHash) -> ProofHash {
    let mut hasher = Sha256::new();
    hasher.update(NODE_DOMAIN);
    hasher.update([namespace]);
    hasher.update(left);
    hasher.update(right);
    hasher.finalize().into()
}

/// Bit `depth` of `key` (from the low end of the big-endian integer).
pub fn key_bit(key: &SmtKey, depth: usize) -> bool {
    (key[31 - depth / 8] >> (depth % 8)) & 1 == 1
}

/// The end of a path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SmtTerminal {
    Empty,
    Leaf { key: SmtKey, value: u64 },
}

/// The deepest sibling's preimage in a delete proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SmtSibling {
    Leaf { key: SmtKey, value: u64 },
    Node { left: ProofHash, right: ProofHash },
}

/// A path: siblings top-down (`siblings[d]` is the sibling at depth `d + 1`, i.e. the other child
/// of the depth-`d` node) and the terminal at depth `siblings.len()`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SmtPath {
    pub siblings: Vec<ProofHash>,
    pub terminal: SmtTerminal,
}

/// A delete proof: the key's membership path plus the deepest sibling's preimage (absent only
/// when the key is the tree's sole entry).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SmtDeleteProof {
    pub path: SmtPath,
    pub sibling: Option<SmtSibling>,
}

fn terminal_hash(namespace: u8, terminal: &SmtTerminal) -> ProofHash {
    match terminal {
        SmtTerminal::Empty => smt_empty(namespace),
        SmtTerminal::Leaf { key, value } => smt_leaf(namespace, key, *value),
    }
}

/// Fold `current` (the subtree at depth `from`) up to the root through `siblings[..from]`.
fn fold_up(namespace: u8, key: &SmtKey, siblings: &[ProofHash], from: usize, mut current: ProofHash) -> ProofHash {
    for depth in (0..from).rev() {
        current = if key_bit(key, depth) {
            smt_node(namespace, &siblings[depth], &current)
        } else {
            smt_node(namespace, &current, &siblings[depth])
        };
    }
    current
}

/// Check `path` is a path for `key` in the tree with `root`: a leaf terminal must sit on `key`'s
/// prefix, and the path must hash to `root`.
fn check_path(namespace: u8, root: &ProofHash, key: &SmtKey, path: &SmtPath) -> io::Result<()> {
    let depth = path.siblings.len();
    if depth > SMT_KEY_BITS {
        return Err(invalid("checkpoint index path is deeper than the key"));
    }
    if let SmtTerminal::Leaf { key: other, .. } = &path.terminal {
        if (0..depth).any(|d| key_bit(other, d) != key_bit(key, d)) {
            return Err(invalid("checkpoint index path: terminal leaf is off the key's prefix"));
        }
    }
    let computed = fold_up(namespace, key, &path.siblings, depth, terminal_hash(namespace, &path.terminal));
    if &computed != root {
        return Err(invalid("checkpoint index path does not open the index root"));
    }
    Ok(())
}

/// Verify `key -> value` is in the tree with `root`; returns the value.
pub fn verify_membership(namespace: u8, root: &ProofHash, key: &SmtKey, path: &SmtPath) -> io::Result<u64> {
    check_path(namespace, root, key, path)?;
    match &path.terminal {
        SmtTerminal::Leaf { key: found, value } if found == key => Ok(*value),
        _ => Err(invalid("checkpoint index: key is not a member")),
    }
}

/// Verify `key` is absent from the tree with `root`.
pub fn verify_non_membership(namespace: u8, root: &ProofHash, key: &SmtKey, path: &SmtPath) -> io::Result<()> {
    check_path(namespace, root, key, path)?;
    match &path.terminal {
        SmtTerminal::Leaf { key: found, .. } if found == key => {
            Err(invalid("checkpoint index: key is already a member"))
        }
        _ => Ok(()),
    }
}

/// Verify the non-membership of `key` in `root` and return the root after inserting `key -> value`.
pub fn verify_insert(
    namespace: u8,
    root: &ProofHash,
    key: &SmtKey,
    value: u64,
    path: &SmtPath,
) -> io::Result<ProofHash> {
    verify_non_membership(namespace, root, key, path)?;
    let depth = path.siblings.len();
    let new_leaf = smt_leaf(namespace, key, value);
    let subtree = match &path.terminal {
        SmtTerminal::Empty => new_leaf,
        SmtTerminal::Leaf { key: other, value: other_value } => {
            let split = (depth..SMT_KEY_BITS)
                .find(|&d| key_bit(other, d) != key_bit(key, d))
                .ok_or_else(|| invalid("checkpoint index: distinct keys share every bit"))?;
            let other_leaf = smt_leaf(namespace, other, *other_value);
            let mut current = if key_bit(key, split) {
                smt_node(namespace, &other_leaf, &new_leaf)
            } else {
                smt_node(namespace, &new_leaf, &other_leaf)
            };
            let empty = smt_empty(namespace);
            for d in (depth..split).rev() {
                current = if key_bit(key, d) {
                    smt_node(namespace, &empty, &current)
                } else {
                    smt_node(namespace, &current, &empty)
                };
            }
            current
        }
    };
    Ok(fold_up(namespace, key, &path.siblings, depth, subtree))
}

/// Verify the membership of `key -> value` in `root` and return the root after deleting it.
pub fn verify_delete(
    namespace: u8,
    root: &ProofHash,
    key: &SmtKey,
    value: u64,
    proof: &SmtDeleteProof,
) -> io::Result<ProofHash> {
    if verify_membership(namespace, root, key, &proof.path)? != value {
        return Err(invalid("checkpoint index: deleted entry has another value"));
    }
    let siblings = &proof.path.siblings;
    let depth = siblings.len();
    let empty = smt_empty(namespace);
    if depth == 0 {
        if proof.sibling.is_some() {
            return Err(invalid("checkpoint index: sole-entry delete carries a sibling"));
        }
        return Ok(empty);
    }
    let sibling = proof
        .sibling
        .as_ref()
        .ok_or_else(|| invalid("checkpoint index: delete proof lacks the sibling preimage"))?;
    let last = siblings[depth - 1];
    match sibling {
        SmtSibling::Node { left, right } => {
            if smt_node(namespace, left, right) != last {
                return Err(invalid("checkpoint index: sibling node preimage mismatch"));
            }
            // The sibling holds two or more keys: the parent keeps them under a node.
            Ok(fold_up(namespace, key, siblings, depth, empty))
        }
        SmtSibling::Leaf { key: other, value: other_value } => {
            if smt_leaf(namespace, other, *other_value) != last {
                return Err(invalid("checkpoint index: sibling leaf preimage mismatch"));
            }
            if (0..depth - 1).any(|d| key_bit(other, d) != key_bit(key, d))
                || key_bit(other, depth - 1) == key_bit(key, depth - 1)
            {
                return Err(invalid("checkpoint index: sibling leaf is not the key's sibling"));
            }
            // The parent now holds one key: the leaf moves up past every empty sibling.
            let mut at = depth - 1;
            while at > 0 && siblings[at - 1] == empty {
                at -= 1;
            }
            Ok(fold_up(namespace, key, siblings, at, last))
        }
    }
}

// ---------------------------------------------------------------------------
// Prover-side tree
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Node {
    Empty,
    Leaf { key: SmtKey, value: u64, hash: ProofHash },
    Internal { hash: ProofHash, left: Box<Node>, right: Box<Node> },
}

impl Node {
    fn hash(&self, namespace: u8) -> ProofHash {
        match self {
            Node::Empty => smt_empty(namespace),
            Node::Leaf { hash, .. } | Node::Internal { hash, .. } => *hash,
        }
    }
}

/// The prover's canonical compact sparse Merkle tree.
#[derive(Clone, Debug)]
pub struct Smt {
    namespace: u8,
    root: Node,
    len: usize,
}

impl Smt {
    pub fn new(namespace: u8) -> Self {
        Self { namespace, root: Node::Empty, len: 0 }
    }

    /// Build from entries (keys must be distinct).
    pub fn from_entries(namespace: u8, entries: impl IntoIterator<Item = (SmtKey, u64)>) -> io::Result<Self> {
        let map: BTreeMap<SmtKey, u64> = {
            let mut map = BTreeMap::new();
            for (key, value) in entries {
                if map.insert(key, value).is_some() {
                    return Err(invalid("checkpoint index: duplicate key"));
                }
            }
            map
        };
        let entries: Vec<(SmtKey, u64)> = map.into_iter().collect();
        let len = entries.len();
        let root = Self::build(namespace, entries, 0);
        Ok(Self { namespace, root, len })
    }

    fn build(namespace: u8, entries: Vec<(SmtKey, u64)>, depth: usize) -> Node {
        match entries.len() {
            0 => Node::Empty,
            1 => {
                let (key, value) = entries[0];
                Node::Leaf { key, value, hash: smt_leaf(namespace, &key, value) }
            }
            _ => {
                let (right, left): (Vec<_>, Vec<_>) =
                    entries.into_iter().partition(|(key, _)| key_bit(key, depth));
                Self::internal(namespace, Self::build(namespace, left, depth + 1), Self::build(namespace, right, depth + 1))
            }
        }
    }

    fn internal(namespace: u8, left: Node, right: Node) -> Node {
        let hash = smt_node(namespace, &left.hash(namespace), &right.hash(namespace));
        Node::Internal { hash, left: Box::new(left), right: Box::new(right) }
    }

    pub fn root(&self) -> ProofHash {
        self.root.hash(self.namespace)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    /// The path of `key` (a membership or non-membership witness).
    pub fn path(&self, key: &SmtKey) -> SmtPath {
        let mut siblings = Vec::new();
        let mut node = &self.root;
        let mut depth = 0;
        loop {
            match node {
                Node::Empty => return SmtPath { siblings, terminal: SmtTerminal::Empty },
                Node::Leaf { key, value, .. } => {
                    return SmtPath { siblings, terminal: SmtTerminal::Leaf { key: *key, value: *value } }
                }
                Node::Internal { left, right, .. } => {
                    if key_bit(key, depth) {
                        siblings.push(left.hash(self.namespace));
                        node = right;
                    } else {
                        siblings.push(right.hash(self.namespace));
                        node = left;
                    }
                    depth += 1;
                }
            }
        }
    }

    /// The delete witness of `key` (its path plus the deepest sibling's preimage).
    pub fn delete_proof(&self, key: &SmtKey) -> SmtDeleteProof {
        let path = self.path(key);
        let depth = path.siblings.len();
        if depth == 0 {
            return SmtDeleteProof { path, sibling: None };
        }
        // Walk to the parent of the terminal and read the other child.
        let mut node = &self.root;
        for d in 0..depth - 1 {
            if let Node::Internal { left, right, .. } = node {
                node = if key_bit(key, d) { right } else { left };
            }
        }
        let sibling = match node {
            Node::Internal { left, right, .. } => {
                let other = if key_bit(key, depth - 1) { left } else { right };
                match other.as_ref() {
                    Node::Leaf { key, value, .. } => Some(SmtSibling::Leaf { key: *key, value: *value }),
                    Node::Internal { left, right, .. } => Some(SmtSibling::Node {
                        left: left.hash(self.namespace),
                        right: right.hash(self.namespace),
                    }),
                    Node::Empty => None,
                }
            }
            _ => None,
        };
        SmtDeleteProof { path, sibling }
    }

    /// Insert `key -> value` (the key must be absent).
    pub fn insert(&mut self, key: SmtKey, value: u64) -> io::Result<()> {
        let namespace = self.namespace;
        let root = std::mem::replace(&mut self.root, Node::Empty);
        self.root = Self::insert_at(namespace, root, key, value, 0)?;
        self.len += 1;
        Ok(())
    }

    fn insert_at(namespace: u8, node: Node, key: SmtKey, value: u64, depth: usize) -> io::Result<Node> {
        match node {
            Node::Empty => Ok(Node::Leaf { key, value, hash: smt_leaf(namespace, &key, value) }),
            Node::Leaf { key: other, value: other_value, hash } => {
                if other == key {
                    return Err(invalid("checkpoint index: key is already a member"));
                }
                let existing = Node::Leaf { key: other, value: other_value, hash };
                let fresh = Node::Leaf { key, value, hash: smt_leaf(namespace, &key, value) };
                Ok(Self::split(namespace, existing, other, fresh, key, depth))
            }
            Node::Internal { left, right, .. } => {
                if key_bit(&key, depth) {
                    let right = Self::insert_at(namespace, *right, key, value, depth + 1)?;
                    Ok(Self::internal(namespace, *left, right))
                } else {
                    let left = Self::insert_at(namespace, *left, key, value, depth + 1)?;
                    Ok(Self::internal(namespace, left, *right))
                }
            }
        }
    }

    fn split(namespace: u8, a: Node, a_key: SmtKey, b: Node, b_key: SmtKey, depth: usize) -> Node {
        match (key_bit(&a_key, depth), key_bit(&b_key, depth)) {
            (false, true) => Self::internal(namespace, a, b),
            (true, false) => Self::internal(namespace, b, a),
            (false, false) => Self::internal(namespace, Self::split(namespace, a, a_key, b, b_key, depth + 1), Node::Empty),
            (true, true) => Self::internal(namespace, Node::Empty, Self::split(namespace, a, a_key, b, b_key, depth + 1)),
        }
    }

    /// Delete `key` (it must be present).
    pub fn delete(&mut self, key: &SmtKey) -> io::Result<()> {
        let namespace = self.namespace;
        let root = std::mem::replace(&mut self.root, Node::Empty);
        self.root = Self::delete_at(namespace, root, key, 0)?;
        self.len -= 1;
        Ok(())
    }

    fn delete_at(namespace: u8, node: Node, key: &SmtKey, depth: usize) -> io::Result<Node> {
        match node {
            Node::Empty => Err(invalid("checkpoint index: key is not a member")),
            Node::Leaf { key: other, .. } if &other == key => Ok(Node::Empty),
            Node::Leaf { .. } => Err(invalid("checkpoint index: key is not a member")),
            Node::Internal { left, right, .. } => {
                let (left, right) = if key_bit(key, depth) {
                    (*left, Self::delete_at(namespace, *right, key, depth + 1)?)
                } else {
                    (Self::delete_at(namespace, *left, key, depth + 1)?, *right)
                };
                // Canonical collapse: a node left with one leaf and an empty side becomes the leaf.
                Ok(match (&left, &right) {
                    (Node::Empty, Node::Leaf { .. }) => right,
                    (Node::Leaf { .. }, Node::Empty) => left,
                    (Node::Empty, Node::Empty) => Node::Empty,
                    _ => Self::internal(namespace, left, right),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(i: u64) -> SmtKey {
        let mut hasher = Sha256::new();
        hasher.update(i.to_be_bytes());
        hasher.finalize().into()
    }

    /// Keys sharing a long low-bit prefix (forces deep splits and chains).
    fn clustered(i: u64) -> SmtKey {
        let mut k = [0u8; 32];
        k[0] = i as u8;
        k
    }

    #[test]
    fn incremental_updates_match_from_scratch_and_verify() {
        for make in [key as fn(u64) -> SmtKey, clustered] {
            let ns = NS_TRIPLE_INDEX;
            let mut tree = Smt::new(ns);
            let mut present = BTreeMap::new();
            for i in 0..40u64 {
                let k = make(i);
                let root = tree.root();
                let path = tree.path(&k);
                let new_root = verify_insert(ns, &root, &k, i, &path).unwrap();
                tree.insert(k, i).unwrap();
                present.insert(k, i);
                assert_eq!(tree.root(), new_root);
                assert_eq!(Smt::from_entries(ns, present.clone()).unwrap().root(), new_root);
            }
            for i in (0..40u64).step_by(3).chain([1, 39, 38]) {
                let k = make(i);
                if !present.contains_key(&k) {
                    continue;
                }
                let root = tree.root();
                let proof = tree.delete_proof(&k);
                let new_root = verify_delete(ns, &root, &k, i, &proof).unwrap();
                tree.delete(&k).unwrap();
                present.remove(&k);
                assert_eq!(tree.root(), new_root, "delete {i}");
                assert_eq!(Smt::from_entries(ns, present.clone()).unwrap().root(), new_root);
            }
            // Delete everything down to empty.
            for (k, v) in present.clone() {
                let proof = tree.delete_proof(&k);
                let new_root = verify_delete(ns, &tree.root(), &k, v, &proof).unwrap();
                tree.delete(&k).unwrap();
                assert_eq!(tree.root(), new_root);
            }
            assert_eq!(tree.root(), smt_empty(ns));
        }
    }

    #[test]
    fn membership_and_non_membership_forgeries_reject() {
        let ns = NS_NV_INDEX;
        let tree = Smt::from_entries(ns, (0..20u64).map(|i| (key(i), i + 1))).unwrap();
        let root = tree.root();
        // Honest.
        assert_eq!(verify_membership(ns, &root, &key(5), &tree.path(&key(5))).unwrap(), 6);
        verify_non_membership(ns, &root, &key(99), &tree.path(&key(99))).unwrap();
        // A member's path cannot prove absence; an absent key's path cannot prove membership.
        assert!(verify_non_membership(ns, &root, &key(5), &tree.path(&key(5))).is_err());
        assert!(verify_membership(ns, &root, &key(99), &tree.path(&key(99))).is_err());
        // Claiming the member's terminal is empty does not open the root.
        let mut forged = tree.path(&key(5));
        forged.terminal = SmtTerminal::Empty;
        assert!(verify_non_membership(ns, &root, &key(5), &forged).is_err());
        // A path cut short above the real terminal does not open the root.
        let mut short = tree.path(&key(5));
        short.siblings.pop();
        assert!(verify_non_membership(ns, &root, &key(5), &short).is_err());
        // Another namespace.
        assert!(verify_membership(NS_PRED_INDEX, &root, &key(5), &tree.path(&key(5))).is_err());
        // Wrong value.
        let mut wrong = tree.path(&key(5));
        wrong.terminal = SmtTerminal::Leaf { key: key(5), value: 7 };
        assert!(verify_membership(ns, &root, &key(5), &wrong).is_err());
        // A delete whose sibling preimage lies about its kind.
        let mut del = tree.delete_proof(&key(5));
        del.sibling = match del.sibling {
            Some(SmtSibling::Node { .. }) => Some(SmtSibling::Leaf { key: key(6), value: 7 }),
            Some(SmtSibling::Leaf { .. }) => Some(SmtSibling::Node { left: [0; 32], right: [0; 32] }),
            None => None,
        };
        assert!(verify_delete(ns, &root, &key(5), 6, &del).is_err());
    }
}
