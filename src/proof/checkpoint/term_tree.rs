//! The id-indexed, append-only term tree of a checkpoint (`id -> term`).
//!
//! A fixed-depth ([`TERM_TREE_DEPTH`]) SHA-256 Merkle tree whose leaf at position `id - 1` commits
//! the term with chain id `id`. Chain ids are stable and dense, so the tree only ever gains leaves
//! at its end. An append of `m` terms to a tree of `n` leaves is proven by the tree's FRONTIER —
//! the roots of the complete left subtrees along the path to position `n`, one per set bit of `n`
//! — which the verifier checks against the old root (`O(depth)` hashes) and then extends with the
//! new leaves (`O(m + depth)` hashes), exactly as an incremental Merkle accumulator does. Fixed
//! depth means the root never re-shapes as the tree grows, so no append ever rehashes old leaves.

use std::io;

use sha2::{Digest, Sha256};

use crate::proof::commitment::NODE_DATATYPE_TAG;
use crate::proof::{invalid, CanonicalObject, ProofHash};

/// Depth of the term tree: room for `2^32` ids per namespace.
pub const TERM_TREE_DEPTH: usize = 32;

const LEAF_DOMAIN: &[u8] = b"terminus-store/pow/checkpoint/term-tree-leaf/v1";
const EMPTY_DOMAIN: &[u8] = b"terminus-store/pow/checkpoint/term-tree-empty/v1";
const NODE_DOMAIN: &[u8] = b"terminus-store/pow/checkpoint/term-tree-node/v1";

/// Node/value namespace tag.
pub const NS_NODE_VALUE: u8 = 1;
/// Predicate namespace tag.
pub const NS_PREDICATE: u8 = 2;

/// The `(datatype_tag, bytes)` decomposition of a node/value term (a predicate IRI uses the node
/// tag), the same split `hash_object` and the layer dictionary trees use.
pub fn object_leaf_parts(object: &CanonicalObject) -> (u8, &[u8]) {
    match object {
        CanonicalObject::Node(iri) => (NODE_DATATYPE_TAG as u8, iri.as_slice()),
        CanonicalObject::Value { datatype, lexical } => (*datatype as u8, lexical.as_slice()),
    }
}

/// `SHA256(LEAF ‖ ns ‖ id ‖ tag ‖ len ‖ bytes)`.
pub fn term_leaf(namespace: u8, id: u64, tag: u8, bytes: &[u8]) -> ProofHash {
    let mut hasher = Sha256::new();
    hasher.update(LEAF_DOMAIN);
    hasher.update([namespace]);
    hasher.update(id.to_be_bytes());
    hasher.update([tag]);
    hasher.update((bytes.len() as u64).to_be_bytes());
    hasher.update(bytes);
    hasher.finalize().into()
}

fn node(namespace: u8, left: &ProofHash, right: &ProofHash) -> ProofHash {
    let mut hasher = Sha256::new();
    hasher.update(NODE_DOMAIN);
    hasher.update([namespace]);
    hasher.update(left);
    hasher.update(right);
    hasher.finalize().into()
}

/// `empty[l]` = the root of an all-empty subtree of height `l`.
fn empty_hashes(namespace: u8) -> [ProofHash; TERM_TREE_DEPTH + 1] {
    let mut out = [[0u8; 32]; TERM_TREE_DEPTH + 1];
    let mut hasher = Sha256::new();
    hasher.update(EMPTY_DOMAIN);
    hasher.update([namespace]);
    out[0] = hasher.finalize().into();
    for level in 1..=TERM_TREE_DEPTH {
        out[level] = node(namespace, &out[level - 1], &out[level - 1]);
    }
    out
}

/// The verifier's incremental accumulator: the tree size plus the frontier (`branch[l]` is
/// meaningful exactly where bit `l` of `size` is set).
#[derive(Clone, Debug)]
pub struct Accumulator {
    namespace: u8,
    size: u64,
    branch: [ProofHash; TERM_TREE_DEPTH],
    empty: [ProofHash; TERM_TREE_DEPTH + 1],
}

impl Accumulator {
    /// Rebuild the accumulator of a tree of `size` leaves from its frontier (one hash per set bit
    /// of `size`, lowest level first) and check it reproduces `root`.
    pub fn from_frontier(
        namespace: u8,
        size: u64,
        frontier: &[ProofHash],
        root: &ProofHash,
    ) -> io::Result<Self> {
        if size >= 1u64 << TERM_TREE_DEPTH {
            return Err(invalid("checkpoint term tree is full"));
        }
        if frontier.len() != size.count_ones() as usize {
            return Err(invalid("checkpoint term-tree frontier has the wrong length"));
        }
        let mut branch = [[0u8; 32]; TERM_TREE_DEPTH];
        let mut next = frontier.iter();
        for (level, slot) in branch.iter_mut().enumerate() {
            if size >> level & 1 == 1 {
                *slot = *next.next().unwrap();
            }
        }
        let acc = Self { namespace, size, branch, empty: empty_hashes(namespace) };
        if &acc.root() != root {
            return Err(invalid("checkpoint term-tree frontier does not open the previous root"));
        }
        Ok(acc)
    }

    /// Append one leaf.
    pub fn push(&mut self, leaf: ProofHash) -> io::Result<()> {
        if self.size + 1 >= 1u64 << TERM_TREE_DEPTH {
            return Err(invalid("checkpoint term tree is full"));
        }
        let mut current = leaf;
        for level in 0..TERM_TREE_DEPTH {
            if self.size >> level & 1 == 0 {
                self.branch[level] = current;
                break;
            }
            current = node(self.namespace, &self.branch[level], &current);
        }
        self.size += 1;
        Ok(())
    }

    /// The root of the current tree.
    pub fn root(&self) -> ProofHash {
        let mut current = self.empty[0];
        for level in 0..TERM_TREE_DEPTH {
            current = if self.size >> level & 1 == 1 {
                node(self.namespace, &self.branch[level], &current)
            } else {
                node(self.namespace, &current, &self.empty[level])
            };
        }
        current
    }

    pub fn size(&self) -> u64 {
        self.size
    }
}

/// The prover's term tree: every leaf plus the complete internal nodes, so frontiers and roots
/// are read off directly.
#[derive(Clone, Debug)]
pub struct TermTree {
    namespace: u8,
    /// `levels[l][i]` = the root of the complete subtree of height `l` covering leaves
    /// `i·2^l .. (i+1)·2^l`.
    levels: Vec<Vec<ProofHash>>,
}

impl TermTree {
    pub fn new(namespace: u8) -> Self {
        Self { namespace, levels: vec![Vec::new(); TERM_TREE_DEPTH + 1] }
    }

    pub fn len(&self) -> u64 {
        self.levels[0].len() as u64
    }

    pub fn push(&mut self, leaf: ProofHash) -> io::Result<()> {
        if self.len() + 1 >= 1u64 << TERM_TREE_DEPTH {
            return Err(invalid("checkpoint term tree is full"));
        }
        self.levels[0].push(leaf);
        let mut index = self.levels[0].len() - 1;
        for level in 0..TERM_TREE_DEPTH {
            if index & 1 == 0 {
                break;
            }
            let parent = node(self.namespace, &self.levels[level][index - 1], &self.levels[level][index]);
            self.levels[level + 1].push(parent);
            index >>= 1;
        }
        Ok(())
    }

    /// The frontier of the current tree (one hash per set bit of the size, lowest level first).
    pub fn frontier(&self) -> Vec<ProofHash> {
        let size = self.len();
        (0..TERM_TREE_DEPTH)
            .filter(|level| size >> level & 1 == 1)
            .map(|level| self.levels[level][(size >> level) as usize - 1])
            .collect()
    }

    pub fn root(&self) -> ProofHash {
        let size = self.len();
        let frontier = self.frontier();
        let mut branch = [[0u8; 32]; TERM_TREE_DEPTH];
        let mut next = frontier.iter();
        for (level, slot) in branch.iter_mut().enumerate() {
            if size >> level & 1 == 1 {
                *slot = *next.next().unwrap();
            }
        }
        Accumulator { namespace: self.namespace, size, branch, empty: empty_hashes(self.namespace) }
            .root()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaf(i: u64) -> ProofHash {
        term_leaf(NS_NODE_VALUE, i + 1, 0, &i.to_be_bytes())
    }

    /// Naive full-depth root over `leaves`.
    fn naive_root(leaves: &[ProofHash]) -> ProofHash {
        let empty = empty_hashes(NS_NODE_VALUE);
        let mut level: Vec<ProofHash> = leaves.to_vec();
        for l in 0..TERM_TREE_DEPTH {
            let mut next = Vec::new();
            for pair in level.chunks(2) {
                let right = if pair.len() == 2 { pair[1] } else { empty[l] };
                next.push(node(NS_NODE_VALUE, &pair[0], &right));
            }
            if next.is_empty() {
                next.push(empty[l + 1]);
            }
            level = next;
        }
        level[0]
    }

    #[test]
    fn frontier_append_matches_naive_root() {
        let leaves: Vec<ProofHash> = (0..37).map(leaf).collect();
        let mut tree = TermTree::new(NS_NODE_VALUE);
        assert_eq!(tree.root(), naive_root(&[]));
        for (i, l) in leaves.iter().enumerate() {
            tree.push(*l).unwrap();
            assert_eq!(tree.root(), naive_root(&leaves[..=i]), "size {}", i + 1);
        }
        for split in 0..=leaves.len() {
            let mut prefix = TermTree::new(NS_NODE_VALUE);
            for l in &leaves[..split] {
                prefix.push(*l).unwrap();
            }
            let mut acc = Accumulator::from_frontier(
                NS_NODE_VALUE, split as u64, &prefix.frontier(), &prefix.root(),
            )
            .unwrap();
            for l in &leaves[split..] {
                acc.push(*l).unwrap();
            }
            assert_eq!(acc.root(), naive_root(&leaves));
        }
    }

    #[test]
    fn forged_frontier_rejects() {
        let mut tree = TermTree::new(NS_NODE_VALUE);
        for i in 0..11 {
            tree.push(leaf(i)).unwrap();
        }
        let mut frontier = tree.frontier();
        frontier[1][0] ^= 1;
        assert!(Accumulator::from_frontier(NS_NODE_VALUE, 11, &frontier, &tree.root()).is_err());
        assert!(Accumulator::from_frontier(NS_NODE_VALUE, 10, &tree.frontier(), &tree.root()).is_err());
        assert!(Accumulator::from_frontier(NS_PREDICATE, 11, &tree.frontier(), &tree.root()).is_err());
    }
}
