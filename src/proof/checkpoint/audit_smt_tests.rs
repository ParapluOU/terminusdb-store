//! Adversarial audit tests (smt). See doc/proof-of-woql/checkpoint-proofs.md step 6.
//!
//! Targets the keyed compact sparse Merkle tree (`smt.rs`) and the append-only term tree
//! (`term_tree.rs`) as the checkpoint verifiers consume them. Passing tests pin soundness
//! properties that were checked; `#[ignore = "audit finding: ..."]` tests assert a rejection the
//! code does not currently perform.

use std::collections::BTreeSet;

use sha2::{Digest, Sha256};

use super::smt::*;
use super::state::{nv_term_leaf, CheckpointState};
use super::term_tree::{term_leaf, verify_term_opening, Accumulator, TermTree, NS_NODE_VALUE, NS_PREDICATE};
use super::tests::{genesis, rich_chain, step};
use crate::proof::commitment::{node_value_sort_key, predicate_sort_key, DictOpening, NODE_DATATYPE_TAG};
use crate::proof::{CanonicalObject, ProofHash};

// ---------------------------------------------------------------------------
// Key pools
// ---------------------------------------------------------------------------

fn hashed(i: u64) -> SmtKey {
    let mut hasher = Sha256::new();
    hasher.update(b"audit");
    hasher.update(i.to_be_bytes());
    hasher.finalize().into()
}

/// Keys that agree on their low 248 bits (differ only in byte 0, the TOP bits): deep chains, and
/// pairs (0x00.., 0x80..) that split only at bit 255, the deepest level.
fn clustered(i: u8) -> SmtKey {
    let mut k = [0u8; 32];
    k[0] = i;
    k
}

/// A mixed pool: hashed keys, deep clusters, and keys sharing a 9-bit low prefix.
fn pool() -> Vec<SmtKey> {
    let mut keys: Vec<SmtKey> = (0..10).map(hashed).collect();
    keys.extend([0x00u8, 0x80, 0x40, 0xc0, 0x01, 0x81].iter().map(|b| clustered(*b)));
    for i in 0..4u8 {
        let mut k = hashed(100 + i as u64);
        k[31] = 0x5a;
        k[30] = (k[30] & !1) | 1;
        keys.push(k);
    }
    keys
}

/// A tiny deterministic PRNG (xorshift64*).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
}

fn canonical(ns: u8, set: &BTreeSet<SmtKey>) -> ProofHash {
    Smt::from_entries(ns, set.iter().map(|k| (*k, 1))).unwrap().root()
}

// ---------------------------------------------------------------------------
// SMT: canonicality of verifier-side updates
// ---------------------------------------------------------------------------

/// Random insert/delete sequences tracked ONLY through the verifier's update functions always
/// land on the canonical root of the current key set (delete collapse, deep splits, bit-255
/// splits, emptying the tree).
#[test]
fn verifier_updates_stay_canonical_under_random_sequences() {
    let ns = NS_TRIPLE_INDEX;
    let keys = pool();
    for seed in 1..=12u64 {
        let mut rng = Rng(seed * 0x9e37_79b9_7f4a_7c15);
        let mut tree = Smt::new(ns);
        let mut set = BTreeSet::new();
        let mut root = tree.root();
        for _ in 0..200 {
            let k = keys[(rng.next() % keys.len() as u64) as usize];
            if set.contains(&k) {
                let proof = tree.delete_proof(&k);
                root = verify_delete(ns, &root, &k, 1, &proof).unwrap();
                tree.delete(&k).unwrap();
                set.remove(&k);
                // The deleted key is provably absent afterwards.
                verify_non_membership(ns, &root, &k, &tree.path(&k)).unwrap();
            } else {
                let path = tree.path(&k);
                root = verify_insert(ns, &root, &k, 1, &path).unwrap();
                tree.insert(k, 1).unwrap();
                set.insert(k);
                assert_eq!(verify_membership(ns, &root, &k, &tree.path(&k)).unwrap(), 1);
            }
            assert_eq!(root, canonical(ns, &set), "seed {}", seed);
            assert_eq!(root, tree.root());
        }
    }
}

/// Every subset of an 8-key pool (including bit-255 and deep-cluster pairs) has its own root:
/// no two key sets share a root, and the empty set's root is the namespace's EMPTY.
#[test]
fn distinct_key_sets_have_distinct_roots() {
    let ns = NS_NV_INDEX;
    let keys: Vec<SmtKey> = [hashed(1), hashed(2), hashed(3), clustered(0x00), clustered(0x80), clustered(0x40), clustered(0x01), hashed(4)].to_vec();
    let mut seen = BTreeSet::new();
    for mask in 0u32..(1 << keys.len()) {
        let set: BTreeSet<SmtKey> = (0..keys.len()).filter(|i| mask >> i & 1 == 1).map(|i| keys[i]).collect();
        assert!(seen.insert(canonical(ns, &set)), "root collision at mask {:b}", mask);
    }
    assert_eq!(canonical(ns, &BTreeSet::new()), smt_empty(ns));
    // Values are bound: the same key set under different values gives different roots.
    let a = Smt::from_entries(ns, [(hashed(1), 1), (hashed(2), 2)]).unwrap().root();
    let b = Smt::from_entries(ns, [(hashed(1), 2), (hashed(2), 1)]).unwrap().root();
    assert_ne!(a, b);
}

// ---------------------------------------------------------------------------
// SMT: path-shape forgeries
// ---------------------------------------------------------------------------

/// For every member and every depth, a path cut short (terminal at an internal node, claimed
/// Empty or claimed to be any member's leaf) or extended (extra EMPTY siblings below the real
/// terminal, or the real terminal pushed down) never verifies as a non-membership or an insert,
/// and never as a membership of a different value.
#[test]
fn truncated_or_extended_paths_never_prove_absence() {
    let ns = NS_TRIPLE_INDEX;
    let keys = pool();
    let tree = Smt::from_entries(ns, keys.iter().map(|k| (*k, 1))).unwrap();
    let root = tree.root();
    let empty = smt_empty(ns);
    for k in &keys {
        let honest = tree.path(k);
        let d = honest.siblings.len();
        let mut terminals = vec![SmtTerminal::Empty];
        terminals.extend(keys.iter().map(|o| SmtTerminal::Leaf { key: *o, value: 1 }));
        terminals.extend(keys.iter().filter(|o| *o != k).map(|o| SmtTerminal::Leaf { key: *o, value: 0 }));
        for cut in 0..=d {
            for terminal in &terminals {
                let forged = SmtPath { siblings: honest.siblings[..cut].to_vec(), terminal: terminal.clone() };
                if forged == honest {
                    continue;
                }
                assert!(verify_non_membership(ns, &root, k, &forged).is_err(), "cut {} {:?}", cut, terminal);
                assert!(verify_insert(ns, &root, k, 7, &forged).is_err());
                if let Ok(v) = verify_membership(ns, &root, k, &forged) {
                    assert_eq!(v, 1);
                }
            }
        }
        // Extended: extra empty siblings below, with any terminal.
        for extra in 1..4 {
            let mut siblings = honest.siblings.clone();
            siblings.extend(std::iter::repeat(empty).take(extra));
            for terminal in &terminals {
                let forged = SmtPath { siblings: siblings.clone(), terminal: terminal.clone() };
                assert!(verify_non_membership(ns, &root, k, &forged).is_err());
                assert!(verify_membership(ns, &root, k, &forged).is_err());
            }
        }
    }
    // Absent keys: the honest absence path is the only one; the absent key cannot borrow a
    // member's path (a shorter/longer path or another branch).
    for i in 500..540u64 {
        let absent = hashed(i);
        let honest = tree.path(&absent);
        verify_non_membership(ns, &root, &absent, &honest).unwrap();
        assert!(verify_membership(ns, &root, &absent, &honest).is_err());
        for k in &keys {
            let other = tree.path(k);
            if other != honest {
                assert!(verify_non_membership(ns, &root, &absent, &other).is_err());
            }
        }
    }
}

/// A path whose depth exceeds the key width rejects instead of indexing out of bounds.
#[test]
fn over_deep_paths_reject() {
    let ns = NS_NV_INDEX;
    let path = SmtPath { siblings: vec![smt_empty(ns); SMT_KEY_BITS + 1], terminal: SmtTerminal::Empty };
    assert!(verify_non_membership(ns, &smt_empty(ns), &hashed(1), &path).is_err());
    assert!(verify_insert(ns, &smt_empty(ns), &hashed(1), 1, &path).is_err());
}

/// The bit-255 split (keys equal except the top bit) round-trips through the verifier: a 255-deep
/// chain of empty siblings, the deletes collapse it back to a single root leaf.
#[test]
fn deepest_split_round_trips() {
    let ns = NS_TRIPLE_INDEX;
    let (a, b) = (clustered(0x00), clustered(0x80));
    let mut tree = Smt::new(ns);
    let r1 = verify_insert(ns, &tree.root(), &a, 1, &tree.path(&a)).unwrap();
    tree.insert(a, 1).unwrap();
    let r2 = verify_insert(ns, &r1, &b, 1, &tree.path(&b)).unwrap();
    tree.insert(b, 1).unwrap();
    assert_eq!(r2, tree.root());
    assert_eq!(tree.path(&a).siblings.len(), 256);
    let r3 = verify_delete(ns, &r2, &a, 1, &tree.delete_proof(&a)).unwrap();
    assert_eq!(r3, smt_leaf(ns, &b, 1));
}

// ---------------------------------------------------------------------------
// SMT: delete witnesses
// ---------------------------------------------------------------------------

/// Every alternative sibling preimage a prover could ship (every leaf of the tree, every internal
/// node preimage reachable on any path, the empty case) either rejects or yields the canonical
/// root of the remaining set — a delete witness cannot steer the post-delete root.
#[test]
fn every_alternative_delete_sibling_rejects_or_is_canonical() {
    let ns = NS_TRIPLE_INDEX;
    let keys = pool();
    let tree = Smt::from_entries(ns, keys.iter().map(|k| (*k, 1))).unwrap();
    let root = tree.root();
    let all: BTreeSet<SmtKey> = keys.iter().copied().collect();
    // Candidate preimages: all leaves (both values) and every internal node's (left, right).
    let mut candidates: Vec<Option<SmtSibling>> = vec![None];
    for k in &keys {
        candidates.push(Some(SmtSibling::Leaf { key: *k, value: 1 }));
        candidates.push(Some(SmtSibling::Leaf { key: *k, value: 0 }));
        let path = tree.path(k);
        // Rebuild each internal node on the path from its two children.
        let mut current = match &path.terminal {
            SmtTerminal::Empty => smt_empty(ns),
            SmtTerminal::Leaf { key, value } => smt_leaf(ns, key, *value),
        };
        for depth in (0..path.siblings.len()).rev() {
            let (l, r) = if key_bit(k, depth) { (path.siblings[depth], current) } else { (current, path.siblings[depth]) };
            candidates.push(Some(SmtSibling::Node { left: l, right: r }));
            current = smt_node(ns, &l, &r);
        }
        assert_eq!(current, root);
    }
    for k in &keys {
        let honest = tree.delete_proof(k);
        let mut rest = all.clone();
        rest.remove(k);
        let expected = canonical(ns, &rest);
        assert_eq!(verify_delete(ns, &root, k, 1, &honest).unwrap(), expected);
        for sibling in &candidates {
            let forged = SmtDeleteProof { path: honest.path.clone(), sibling: sibling.clone() };
            if let Ok(r) = verify_delete(ns, &root, k, 1, &forged) {
                assert_eq!(r, expected, "a forged delete sibling steered the root");
            }
        }
        // A wrong value, or a non-member, cannot be deleted.
        assert!(verify_delete(ns, &root, k, 2, &honest).is_err());
    }
    for i in 600..620u64 {
        let absent = hashed(i);
        let path = tree.path(&absent);
        for sibling in &candidates {
            let forged = SmtDeleteProof { path: path.clone(), sibling: sibling.clone() };
            assert!(verify_delete(ns, &root, &absent, 1, &forged).is_err());
        }
    }
}

// ---------------------------------------------------------------------------
// SMT: domain separation
// ---------------------------------------------------------------------------

/// A path of one index never verifies in another (nv, predicate, triple), and leaf, node and
/// empty preimages never coincide across or within namespaces.
#[test]
fn indexes_are_domain_separated() {
    let keys = pool();
    let nss = [NS_NV_INDEX, NS_PRED_INDEX, NS_TRIPLE_INDEX];
    for ns in nss {
        let tree = Smt::from_entries(ns, keys.iter().map(|k| (*k, 1))).unwrap();
        for other in nss.iter().filter(|o| **o != ns) {
            for k in &keys {
                assert!(verify_membership(*other, &tree.root(), k, &tree.path(k)).is_err());
            }
            let absent = hashed(999);
            assert!(verify_non_membership(*other, &tree.root(), &absent, &tree.path(&absent)).is_err());
        }
    }
    let mut hashes = BTreeSet::new();
    for ns in nss {
        assert!(hashes.insert(smt_empty(ns)));
        assert!(hashes.insert(smt_leaf(ns, &hashed(1), 1)));
        assert!(hashes.insert(smt_node(ns, &smt_empty(ns), &smt_empty(ns))));
    }
    // A term-tree leaf/empty is not an SMT preimage either.
    assert!(hashes.insert(term_leaf(NS_NODE_VALUE, 1, 0, b"x")));
}

// ---------------------------------------------------------------------------
// Term tree
// ---------------------------------------------------------------------------

/// The term-tree empty leaf, replicated from the module's private domain.
fn term_tree_empty_leaf(ns: u8) -> ProofHash {
    let mut hasher = Sha256::new();
    hasher.update(b"terminus-store/pow/checkpoint/term-tree-empty/v1");
    hasher.update([ns]);
    hasher.finalize().into()
}

fn opening(tree: &TermTree, id: u64, tag: u8, bytes: &[u8]) -> DictOpening {
    DictOpening { id, datatype_tag: tag, term_bytes: bytes.to_vec(), position: (id - 1) as usize, siblings: tree.siblings(id - 1).unwrap() }
}

/// `verify_term_opening` bounds: id 0, id past the count, a position other than `id - 1`, a
/// sibling list of the wrong depth, another id at the right position, a non-IRI predicate, and
/// opening the slot just past the count (whose leaf is the empty leaf, not a term) all reject.
#[test]
fn term_opening_bounds_and_empty_slots_reject() {
    let ns = NS_NODE_VALUE;
    let mut tree = TermTree::new(ns);
    for id in 1..=5u64 {
        tree.push(term_leaf(ns, id, 0, &id.to_be_bytes())).unwrap();
    }
    let root = tree.root();
    let good = opening(&tree, 3, 0, &3u64.to_be_bytes());
    verify_term_opening(ns, &root, 5, 3, &good).unwrap();
    // id 0.
    let mut zero = good.clone();
    zero.id = 0;
    assert!(verify_term_opening(ns, &root, 5, 0, &zero).is_err());
    // Position mismatch.
    let mut pos = good.clone();
    pos.position = 3;
    assert!(verify_term_opening(ns, &root, 5, 3, &pos).is_err());
    // Depth 33 (an extra sibling) and depth 31.
    let mut deep = good.clone();
    deep.siblings.push([0; 32]);
    assert!(verify_term_opening(ns, &root, 5, 3, &deep).is_err());
    // A leaf moved to another id (ids are inside the leaf preimage).
    let moved = DictOpening { id: 4, position: 3, siblings: tree.siblings(3).unwrap(), ..good.clone() };
    assert!(verify_term_opening(ns, &root, 5, 4, &moved).is_err());
    // The empty slot at id 6: claim the count is 6 (a caller passing an inflated count) — the
    // empty leaf is not any term's preimage, so nothing opens there.
    let mut grown = tree.clone();
    grown.push(term_tree_empty_leaf(ns)).unwrap();
    assert_eq!(grown.root(), root, "an empty leaf appended leaves the root unchanged");
    for tag in [0u8, 1, NODE_DATATYPE_TAG as u8] {
        for bytes in [&b""[..], &6u64.to_be_bytes()[..], &[0u8; 32][..]] {
            let o = opening(&grown, 6, tag, bytes);
            assert!(verify_term_opening(ns, &root, 6, 6, &o).is_err());
        }
    }
    // A predicate opening with a value tag.
    let mut ptree = TermTree::new(NS_PREDICATE);
    ptree.push(term_leaf(NS_PREDICATE, 1, 0, b"p")).unwrap();
    let o = opening(&ptree, 1, 0, b"p");
    assert!(verify_term_opening(NS_PREDICATE, &ptree.root(), 1, 1, &o).is_err());
}

/// The frontier is unique for a given (size, root): tampering any frontier hash, or offering the
/// frontier of a neighbouring size, rejects; appends from it match the prover's tree.
#[test]
fn frontier_is_bound_for_the_true_size() {
    let ns = NS_NODE_VALUE;
    let mut tree = TermTree::new(ns);
    for n in 0..40u64 {
        let root = tree.root();
        let frontier = tree.frontier();
        let mut acc = Accumulator::from_frontier(ns, n, &frontier, &root).unwrap();
        for i in 0..frontier.len() {
            let mut bad = frontier.clone();
            bad[i][5] ^= 0x10;
            assert!(Accumulator::from_frontier(ns, n, &bad, &root).is_err());
        }
        if n > 0 {
            // Size n-1 with the true frontier truncated/reshaped never opens (the real leaf at
            // n-1 is not the empty leaf).
            let mut smaller = TermTree::new(ns);
            for i in 0..n - 1 {
                smaller.push(term_leaf(ns, i + 1, 0, &i.to_be_bytes())).unwrap();
            }
            assert!(Accumulator::from_frontier(ns, n - 1, &smaller.frontier(), &root).is_err());
        }
        let leaf = term_leaf(ns, n + 1, 0, &n.to_be_bytes());
        acc.push(leaf).unwrap();
        tree.push(leaf).unwrap();
        assert_eq!(acc.root(), tree.root());
    }
}

/// LOW (defense in depth): the term-tree root does not bind its leaf count. A frontier claiming
/// `n + 1` leaves whose last leaf is the EMPTY leaf opens the root of the `n`-leaf tree. Not
/// exploitable today — every caller takes the size from a root-bound artifact count — but an
/// `Accumulator` taken alone authenticates a wrong size.
#[test]
#[ignore = "audit finding: term-tree root does not bind its size (empty-leaf padding opens as n+1)"]
fn term_tree_frontier_for_an_inflated_size_rejects() {
    let ns = NS_NODE_VALUE;
    let mut tree = TermTree::new(ns);
    for id in 1..=5u64 {
        tree.push(term_leaf(ns, id, 0, &id.to_be_bytes())).unwrap();
    }
    let root = tree.root();
    let mut inflated = tree.clone();
    inflated.push(term_tree_empty_leaf(ns)).unwrap();
    assert!(
        Accumulator::from_frontier(ns, 6, &inflated.frontier(), &root).is_err(),
        "a frontier for 6 leaves opened the 5-leaf root"
    );
}

// ---------------------------------------------------------------------------
// Real checkpoints: index <-> term tree consistency and lookup soundness
// ---------------------------------------------------------------------------

fn check_state(state: &CheckpointState) {
    let a = &state.artifact;
    // Every node/value term: membership in the keyed index gives exactly its id, in range, and the
    // term tree opens that id to the same term.
    for id in 1..=a.node_value_count {
        let term = state.node_value(id).unwrap().clone();
        let key = node_value_sort_key(&term);
        let path = state.index_path(false, &key);
        assert_eq!(verify_membership(NS_NV_INDEX, &a.nv_term_index_root, &key, &path).unwrap(), id);
        assert!(verify_non_membership(NS_NV_INDEX, &a.nv_term_index_root, &key, &path).is_err());
        let opening = state.term_opening(false, id).unwrap();
        assert_eq!(verify_term_opening(NS_NODE_VALUE, &a.nv_term_tree_root, a.node_value_count, id, &opening).unwrap(), term);
        assert_eq!(state.nv_tree.siblings(id - 1).unwrap(), opening.siblings);
        assert_eq!(nv_term_leaf(id, &term), term_leaf(NS_NODE_VALUE, id, opening.datatype_tag, &opening.term_bytes));
        // The index path of one term never proves another term's membership or absence.
        if id > 1 {
            let other = state.node_value(id - 1).unwrap();
            let other_key = node_value_sort_key(other);
            assert!(verify_membership(NS_NV_INDEX, &a.nv_term_index_root, &other_key, &path).is_err());
            assert!(verify_non_membership(NS_NV_INDEX, &a.nv_term_index_root, &other_key, &path).is_err());
        }
        // The node/value index root is not the predicate index root.
        assert!(verify_membership(NS_NV_INDEX, &a.pred_term_index_root, &key, &path).is_err());
    }
    for id in 1..=a.predicate_count {
        let iri = state.predicate(id).unwrap().to_vec();
        let key = predicate_sort_key(&iri);
        let path = state.index_path(true, &key);
        assert_eq!(verify_membership(NS_PRED_INDEX, &a.pred_term_index_root, &key, &path).unwrap(), id);
        let opening = state.term_opening(true, id).unwrap();
        assert_eq!(
            verify_term_opening(NS_PREDICATE, &a.pred_term_tree_root, a.predicate_count, id, &opening).unwrap(),
            CanonicalObject::Node(iri.clone())
        );
        // A predicate IRI's nv key (hash_object of Node) is a different key from its pred key.
        assert_ne!(node_value_sort_key(&CanonicalObject::Node(iri.clone())), key);
    }
    // Absent terms prove absence only.
    for i in 0..10u64 {
        let absent = CanonicalObject::Node(format!("http://absent/{i}").into_bytes());
        let key = node_value_sort_key(&absent);
        let path = state.index_path(false, &key);
        verify_non_membership(NS_NV_INDEX, &a.nv_term_index_root, &key, &path).unwrap();
        assert!(verify_membership(NS_NV_INDEX, &a.nv_term_index_root, &key, &path).is_err());
    }
    // The triple index holds exactly the present triples.
    assert_eq!(state.triple_index.len() as u64, a.present_count);
    for t in state.present() {
        let key = super::state::triple_key(t);
        let path = state.triple_index.path(&key);
        assert_eq!(verify_membership(NS_TRIPLE_INDEX, &a.triple_index_root, &key, &path).unwrap(), 1);
    }
    assert_eq!(state.nv_index.len() as u64, a.node_value_count);
    assert_eq!(state.pred_index.len() as u64, a.predicate_count);
    // Every index value lies in 1..=count (the range the consumers re-check).
    for id in 1..=a.node_value_count {
        let key = node_value_sort_key(state.node_value(id).unwrap());
        if let SmtTerminal::Leaf { value, .. } = state.index_path(false, &key).terminal {
            assert!(value >= 1 && value <= a.node_value_count);
        }
    }
}

/// On a real chain (genesis plus an inductive checkpoint over seven deltas with removals,
/// re-adds, new predicates and values) the keyed indexes, the term trees and the triple index
/// agree with each other and with the counts, and lookups are sound in both directions.
#[test]
fn real_checkpoint_indexes_and_term_trees_agree() {
    let chain = rich_chain();
    let c0 = genesis(&chain);
    check_state(&c0);
    let (c1, _) = step(&chain, &c0, 0, 3);
    check_state(&c1);
    let (c2, _) = step(&chain, &c1, 3, chain.commitments.len() - 1);
    check_state(&c2);
    // The verifier-side inductive index roots equal the from-scratch canonical roots of the same
    // key sets (inductive inserts did not drift from canonical form).
    let nv: Vec<_> = (1..=c2.artifact.node_value_count)
        .map(|id| (node_value_sort_key(c2.node_value(id).unwrap()), id))
        .collect();
    assert_eq!(Smt::from_entries(NS_NV_INDEX, nv).unwrap().root(), c2.artifact.nv_term_index_root);
    let triples: Vec<_> = c2.present().iter().map(|t| (super::state::triple_key(t), 1)).collect();
    assert_eq!(Smt::from_entries(NS_TRIPLE_INDEX, triples).unwrap().root(), c2.artifact.triple_index_root);
}
