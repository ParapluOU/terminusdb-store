//! Write-transition tests: the recomputed candidate equals the real store's child layer for writes
//! with new terms, removals, right after a checkpoint and after several deltas since; and every
//! forgery the design names rejects.

use std::io;

use tdb_succinct::TdbDataType;

use super::smt::{SmtPath, SmtTerminal};
use super::tests::{genesis, iri, node, rich_chain, step, Chain, Op};
use super::transition::prove_write_transition_inner;
use super::*;
use crate::layer::ValueTriple;
use crate::proof::commitment::{LayerCommitment, VerifierLayer};
use crate::proof::{CanonicalObject, CanonicalTriple, ProofHash, ProofRecord};

fn diff_of(ops: &[Op]) -> WriteDiff {
    let mut additions = Vec::new();
    let mut removals = Vec::new();
    for op in ops {
        match op {
            Op::Add(t) => additions.push(CanonicalTriple::from_value_triple(t)),
            Op::Remove(t) => removals.push(CanonicalTriple::from_value_triple(t)),
        }
    }
    WriteDiff::new(additions, removals).unwrap()
}

/// The real store's candidate: a child of layer `base` carrying `ops` (not pushed onto the chain).
fn store_candidate(chain: &Chain, base: usize, ops: &[Op]) -> std::sync::Arc<LayerCommitment> {
    let builder = chain.layers[base].open_write().unwrap();
    for op in ops {
        match op {
            Op::Add(t) => builder.add_value_triple(t.clone()).unwrap(),
            Op::Remove(t) => builder.remove_value_triple(t.clone()).unwrap(),
        }
    }
    builder.commit().unwrap().proof_commitment().unwrap()
}

/// Prove the write `ops` on layer `base` from checkpoint `ck` (the checkpoint of layer `ck_at`),
/// check the recomputed candidate equals the store's, and round-trip the wire.
fn honest(chain: &Chain, ck: &CheckpointState, ck_at: usize, base: usize, ops: &[Op]) -> WriteTransition {
    let deltas = chain.deltas(ck_at, base);
    let transition = WriteTransition::prove(ck, &deltas, diff_of(ops)).unwrap();
    let real = store_candidate(chain, base, ops);
    let recomputed = transition
        .verify(ck.artifact.checkpoint_root, chain.root(base), real.resolution_root())
        .unwrap();
    assert_eq!(recomputed.state, real.state, "the recomputed candidate commits what the store's does");
    assert_eq!(recomputed.resolution_root(), real.resolution_root());
    let bytes = transition.encode().unwrap();
    let diff = decode_and_verify_write_transition(ck.artifact.checkpoint_root, chain.root(base), real.resolution_root(), &bytes)
        .unwrap();
    assert_eq!(diff, transition.diff);
    transition
}

fn verify(t: &WriteTransition, ck: ProofHash, base: ProofHash, candidate: ProofHash) -> io::Result<LayerCommitment> {
    t.verify(ck, base, candidate)
}

fn assert_rejects(result: io::Result<LayerCommitment>, expect: &str) {
    let error = result.err().unwrap_or_else(|| panic!("expected a rejection containing {:?}", expect));
    assert!(error.to_string().contains(expect), "rejected for {:?}, expected {:?}", error, expect);
}

// ---------------------------------------------------------------------------
// Honest writes
// ---------------------------------------------------------------------------

#[test]
fn write_right_after_the_genesis_checkpoint_with_new_terms_values_and_removals() {
    let chain = rich_chain();
    let c0 = genesis(&chain);
    let ops = vec![
        Op::Add(node("a1", "p1", "zz-new")),
        Op::Add(node("brand-new", "p-new", "b1")),
        Op::Add(node("a2", "p2", "b3")),
        Op::Add(ValueTriple::new_string_value(&iri("a2"), &iri("name"), "Bea")),
        Op::Add(ValueTriple::new_value(&iri("a3"), &iri("age"), u32::make_entry(&7))),
        Op::Add(ValueTriple::new_value(&iri("a3"), &iri("score"), f64::make_entry(&-0.5))),
        Op::Add(ValueTriple::new_value(&iri("a3"), &iri("delta"), i64::make_entry(&-3))),
        Op::Add(ValueTriple::new_value(&iri("a3"), &iri("flag"), bool::make_entry(&true))),
        Op::Remove(node("a1", "p1", "b1")),
        Op::Remove(ValueTriple::new_string_value(&iri("a2"), &iri("name"), "Ann")),
    ];
    let t = honest(&chain, &c0, 0, 0, &ops);
    assert!(t.deltas.is_empty());
}

#[test]
fn write_after_several_deltas_since_the_checkpoint() {
    let chain = rich_chain();
    let c0 = genesis(&chain);
    // Terms minted since the checkpoint (a5, b8, p3, a9, p4, b10), a triple added since (a5 p1 b5)
    // removed, a triple removed since (a3 p1 b3, removed in d7) re-added, a triple removed and
    // re-added since (a4 p2 b4) removed again, an untouched checkpoint triple removed.
    let ops = vec![
        Op::Remove(node("a5", "p1", "b5")),
        Op::Add(node("a3", "p1", "b3")),
        Op::Remove(node("a4", "p2", "b4")),
        Op::Remove(node("a3", "p1", "b6")),
        Op::Add(node("a9", "p3", "b10")),
        Op::Add(node("a7", "p4", "fresh")),
        Op::Add(node("a7", "p3", "b8")),
    ];
    let t = honest(&chain, &c0, 0, 7, &ops);
    assert_eq!(t.deltas.len(), 7);
    // Every intermediate base works too.
    for base in 1..7 {
        honest(&chain, &c0, 0, base, &[Op::Add(node("x", "p1", &format!("y{base}")))]);
    }
}

#[test]
fn write_right_after_an_inductive_checkpoint_and_after_deltas_since_it() {
    let chain = rich_chain();
    let c0 = genesis(&chain);
    let (c3, _) = step(&chain, &c0, 0, 3);
    let ops = vec![Op::Add(node("a1", "p3", "b8")), Op::Remove(node("a1", "p1", "b1")), Op::Add(node("n", "p9", "m"))];
    assert!(honest(&chain, &c3, 3, 3, &ops).deltas.is_empty());
    let ops = vec![Op::Remove(node("a10", "p4", "b10")), Op::Add(node("a10", "p4", "b11")), Op::Add(node("a9", "p1", "b1"))];
    assert_eq!(honest(&chain, &c3, 3, 7, &ops).deltas.len(), 4);
}

#[test]
fn removal_only_and_empty_writes() {
    let chain = rich_chain();
    let c0 = genesis(&chain);
    honest(&chain, &c0, 0, 2, &[Op::Remove(node("a4", "p2", "b4")), Op::Remove(node("a5", "p1", "b5"))]);
    honest(&chain, &c0, 0, 7, &[]);
}

// ---------------------------------------------------------------------------
// Forgeries
// ---------------------------------------------------------------------------

fn base_ops() -> Vec<Op> {
    vec![
        Op::Add(node("a1", "p1", "new-o")),
        Op::Add(node("a2", "p2", "b1")),
        Op::Remove(node("a2", "p1", "b2")),
    ]
}

#[test]
fn a_wrong_candidate_root_on_an_honest_diff_rejects() {
    // Audit F1: the diff is honest, the claimed candidate is not what it produces.
    let chain = rich_chain();
    let c0 = genesis(&chain);
    let ops = base_ops();
    let t = WriteTransition::prove(&c0, &[], diff_of(&ops)).unwrap();
    let other = store_candidate(&chain, 0, &[Op::Add(node("a1", "p1", "other-o"))]).resolution_root();
    assert_rejects(verify(&t, c0.artifact.checkpoint_root, chain.root(0), other), "does not recompute");
    assert_rejects(verify(&t, c0.artifact.checkpoint_root, chain.root(0), chain.root(1)), "does not recompute");
    let mut flipped = store_candidate(&chain, 0, &ops).resolution_root();
    flipped[0] ^= 1;
    assert_rejects(verify(&t, c0.artifact.checkpoint_root, chain.root(0), flipped), "does not recompute");
    // The same honest candidate named against another base rejects too.
    let real = store_candidate(&chain, 0, &ops).resolution_root();
    assert_rejects(verify(&t, c0.artifact.checkpoint_root, chain.root(1), real), "do not end at the write's base");
}

#[test]
fn a_candidate_numbering_new_terms_in_another_order_rejects() {
    let chain = rich_chain();
    let c0 = genesis(&chain);
    let ops = vec![Op::Add(node("a1", "p1", "new-x")), Op::Add(node("a1", "p1", "new-y"))];
    let t = WriteTransition::prove(&c0, &[], diff_of(&ops)).unwrap();
    let real = store_candidate(&chain, 0, &ops);
    // The same delta with its two new ids swapped.
    let base = &chain.commitments[0].state;
    let (x, y) = (base.node_value_count + 1, base.node_value_count + 2);
    let swap = |id: u64| if id == x { y } else if id == y { x } else { id };
    let additions: Vec<ProofRecord> = real
        .additions
        .iter()
        .map(|r| ProofRecord { subject_id: swap(r.subject_id), predicate_id: r.predicate_id, object_id: swap(r.object_id), triple: r.triple.clone() })
        .collect();
    let mappings = real.node_value_catalog.iter().map(|r| (r.triple.object.clone(), swap(r.object_id))).collect();
    let forged = LayerCommitment::from_native_delta(
        Some(base.commitment_root),
        additions,
        Vec::new(),
        Vec::new(),
        base.node_value_count,
        base.predicate_count,
        real.state.node_value_count,
        real.state.predicate_count,
        mappings,
        Vec::new(),
    )
    .unwrap();
    assert_ne!(forged.resolution_root(), real.resolution_root());
    assert_rejects(verify(&t, c0.artifact.checkpoint_root, chain.root(0), forged.resolution_root()), "does not recompute");
}

fn member_index(paths: &[SmtPath]) -> usize {
    paths.iter().position(|p| matches!(p.terminal, SmtTerminal::Leaf { .. })).unwrap()
}

#[test]
fn forged_term_paths_reject() {
    let chain = rich_chain();
    let c0 = genesis(&chain);
    let ops = base_ops();
    let honest_t = WriteTransition::prove(&c0, &[], diff_of(&ops)).unwrap();
    let real = store_candidate(&chain, 0, &ops).resolution_root();
    let ck = c0.artifact.checkpoint_root;
    let base = chain.root(0);

    // A membership path claiming another id.
    let mut t = honest_t.clone();
    let i = member_index(&t.witness.nv_term_paths);
    if let SmtTerminal::Leaf { value, .. } = &mut t.witness.nv_term_paths[i].terminal {
        *value += 1;
    }
    assert_rejects(verify(&t, ck, base, real), "does not open the index root");
    // A membership path with a tampered sibling.
    let mut t = honest_t.clone();
    let i = member_index(&t.witness.pred_term_paths);
    t.witness.pred_term_paths[i].siblings[0][0] ^= 1;
    assert_rejects(verify(&t, ck, base, real), "does not open the index root");
    // A non-membership path for a term that exists: its leaf replaced by an empty terminal.
    let mut t = honest_t.clone();
    let i = member_index(&t.witness.nv_term_paths);
    t.witness.nv_term_paths[i].terminal = SmtTerminal::Empty;
    assert_rejects(verify(&t, ck, base, real), "does not open the index root");
    // ... or by another key's leaf from elsewhere in the index.
    let mut t = honest_t.clone();
    let i = member_index(&t.witness.nv_term_paths);
    let j = (0..t.witness.nv_term_paths.len()).rev().find(|&j| j != i && matches!(t.witness.nv_term_paths[j].terminal, SmtTerminal::Leaf { .. })).unwrap();
    t.witness.nv_term_paths[i] = t.witness.nv_term_paths[j].clone();
    assert!(verify(&t, ck, base, real).is_err());
    // Paths reordered, dropped or added.
    let mut t = honest_t.clone();
    t.witness.nv_term_paths.swap(0, 1);
    assert!(verify(&t, ck, base, real).is_err());
    let mut t = honest_t.clone();
    t.witness.nv_term_paths.pop();
    assert_rejects(verify(&t, ck, base, real), "term paths disagree");
    let mut t = honest_t.clone();
    t.witness.pred_term_paths.push(t.witness.pred_term_paths[0].clone());
    assert_rejects(verify(&t, ck, base, real), "term paths disagree");
    // The honest transition still verifies.
    verify(&honest_t, ck, base, real).unwrap();
}

#[test]
fn a_new_term_claimed_absent_while_a_delta_since_minted_it_is_not_renumbered() {
    // a5 was minted by d1; a write using it after d1 must reuse d1's id, which the verifier takes
    // from the delta's clear catalog after the checkpoint index proves it absent at C_0.
    let chain = rich_chain();
    let c0 = genesis(&chain);
    honest(&chain, &c0, 0, 1, &[Op::Add(node("a5", "p2", "b5"))]);
}

/// A witness minted for a diff that breaks the delta invariant (the prover's own check off).
fn unchecked(chain: &Chain, ck: &CheckpointState, ck_at: usize, base: usize, ops: &[Op]) -> WriteTransition {
    let deltas = chain.deltas(ck_at, base);
    let diff = diff_of(ops);
    let witness = prove_write_transition_inner(ck, &deltas, &diff, false).unwrap();
    WriteTransition {
        checkpoint: ck.artifact.clone(),
        deltas: deltas.iter().map(|d| d.verifier_commitment().unwrap()).collect(),
        diff,
        witness,
    }
}

/// What a lenient store would produce for these records (the claimed candidate of a cheating
/// proposer): the diff's records with the verifier's own ids.
fn any_candidate() -> ProofHash {
    [7; 32]
}

#[test]
fn removing_an_absent_triple_rejects() {
    let chain = rich_chain();
    let c0 = genesis(&chain);
    let ck = c0.artifact.checkpoint_root;
    let expect = "removes a triple that is not present";
    // Existing terms, never a triple.
    let t = unchecked(&chain, &c0, 0, 0, &[Op::Remove(node("a1", "p2", "b3"))]);
    assert_rejects(verify(&t, ck, chain.root(0), any_candidate()), expect);
    // Present at the checkpoint, removed by a delta since (d1 removed a2 p1 b2).
    let t = unchecked(&chain, &c0, 0, 3, &[Op::Remove(node("a2", "p1", "b2"))]);
    assert_rejects(verify(&t, ck, chain.root(3), any_candidate()), expect);
    // Added and removed since (d2 added, d3 removed a7 p3 b8).
    let t = unchecked(&chain, &c0, 0, 3, &[Op::Remove(node("a7", "p3", "b8"))]);
    assert_rejects(verify(&t, ck, chain.root(3), any_candidate()), expect);
    // A triple with a term nobody ever minted.
    let t = unchecked(&chain, &c0, 0, 0, &[Op::Remove(node("a1", "p1", "never"))]);
    assert_rejects(verify(&t, ck, chain.root(0), any_candidate()), expect);
    // The honest path of the triple's absence swapped for a fake presence.
    let mut t = unchecked(&chain, &c0, 0, 0, &[Op::Remove(node("a1", "p2", "b3"))]);
    let ids = {
        let s = &chain.commitments[0];
        let id = |o: &CanonicalObject| s.node_value_catalog.iter().find(|r| &r.triple.object == o).unwrap().object_id;
        let pid = |p: &str| s.predicate_catalog.iter().find(|r| r.triple.predicate == iri(p).as_bytes()).unwrap().predicate_id;
        [id(&CanonicalObject::Node(iri("a1").into_bytes())), pid("p2"), id(&CanonicalObject::Node(iri("b3").into_bytes()))]
    };
    t.witness.triple_paths[0].terminal = SmtTerminal::Leaf { key: triple_key(&ids), value: TRIPLE_PRESENT };
    assert_rejects(verify(&t, ck, chain.root(0), any_candidate()), "does not open the index root");
}

#[test]
fn adding_a_present_triple_rejects() {
    let chain = rich_chain();
    let c0 = genesis(&chain);
    let ck = c0.artifact.checkpoint_root;
    let expect = "adds a triple that is already present";
    // Present at the checkpoint, untouched since.
    let t = unchecked(&chain, &c0, 0, 7, &[Op::Add(node("a2", "p2", "b9")), Op::Add(node("a1", "p1", "b1"))]);
    assert_rejects(verify(&t, ck, chain.root(7), any_candidate()), expect);
    let t = unchecked(&chain, &c0, 0, 0, &[Op::Add(node("a3", "p1", "b3"))]);
    assert_rejects(verify(&t, ck, chain.root(0), any_candidate()), expect);
    // Absent at the checkpoint, added by a delta since (d1 added a5 p1 b5).
    let t = unchecked(&chain, &c0, 0, 2, &[Op::Add(node("a5", "p1", "b5"))]);
    assert_rejects(verify(&t, ck, chain.root(2), any_candidate()), expect);
    // Removed and re-added since (d6 removed, d7 re-added a4 p2 b4).
    let t = unchecked(&chain, &c0, 0, 7, &[Op::Add(node("a4", "p2", "b4"))]);
    assert_rejects(verify(&t, ck, chain.root(7), any_candidate()), expect);
    // The presence path turned into a fake absence.
    let mut t = unchecked(&chain, &c0, 0, 0, &[Op::Add(node("a3", "p1", "b3"))]);
    t.witness.triple_paths[0].terminal = SmtTerminal::Empty;
    assert_rejects(verify(&t, ck, chain.root(0), any_candidate()), "does not open the index root");
    // A triple path dropped or added.
    let mut t = WriteTransition::prove(&c0, &[], diff_of(&base_ops())).unwrap();
    let real = store_candidate(&chain, 0, &base_ops()).resolution_root();
    let extra = t.witness.triple_paths[0].clone();
    t.witness.triple_paths.push(extra);
    assert_rejects(verify(&t, ck, chain.root(0), real), "surplus triple-index paths");
    t.witness.triple_paths.truncate(1);
    assert_rejects(verify(&t, ck, chain.root(0), real), "lacks a triple-index path");
}

#[test]
fn a_witness_for_another_checkpoint_or_chain_rejects() {
    let chain = rich_chain();
    let c0 = genesis(&chain);
    let (c3, _) = step(&chain, &c0, 0, 3);
    let ops = vec![Op::Add(node("a1", "p1", "b9")), Op::Remove(node("a1", "p1", "b1"))];
    let real = store_candidate(&chain, 3, &ops).resolution_root();
    let from_c0 = honest(&chain, &c0, 0, 3, &ops);
    let from_c3 = honest(&chain, &c3, 3, 3, &ops);
    // The ledger trusts C_3; a transition shipping C_0 rejects, and vice versa.
    assert_rejects(verify(&from_c0, c3.artifact.checkpoint_root, chain.root(3), real), "another checkpoint");
    assert_rejects(verify(&from_c3, c0.artifact.checkpoint_root, chain.root(3), real), "another checkpoint");
    // C_3's artifact with C_0's witness (and C_0's deltas since).
    let mut t = from_c0.clone();
    t.checkpoint = c3.artifact.clone();
    assert!(verify(&t, c3.artifact.checkpoint_root, chain.root(3), real).is_err());
    // C_3's artifact and no deltas, but C_0's paths.
    let mut t = from_c3.clone();
    t.witness.nv_term_paths = from_c0.witness.nv_term_paths.clone();
    t.witness.pred_term_paths = from_c0.witness.pred_term_paths.clone();
    t.witness.triple_paths = from_c0.witness.triple_paths.clone();
    assert!(verify(&t, c3.artifact.checkpoint_root, chain.root(3), real).is_err());

    // Another chain with the same base content but one extra triple: its checkpoint's paths do not
    // open this chain's checkpoint.
    let twin = Chain::new(vec![
        node("a1", "p1", "b1"),
        node("a2", "p1", "b2"),
        node("a3", "p1", "b3"),
        node("a4", "p2", "b4"),
        ValueTriple::new_value(&iri("a1"), &iri("age"), u32::make_entry(&30)),
        ValueTriple::new_string_value(&iri("a2"), &iri("name"), "Ann"),
        node("zz", "p1", "zz"),
    ]);
    let twin_c0 = genesis(&twin);
    let foreign = WriteTransition::prove(&twin_c0, &[], diff_of(&ops)).unwrap();
    let c0_real = store_candidate(&chain, 0, &ops).resolution_root();
    let mut t = WriteTransition::prove(&c0, &[], diff_of(&ops)).unwrap();
    verify(&t, c0.artifact.checkpoint_root, chain.root(0), c0_real).unwrap();
    t.witness = foreign.witness.clone();
    assert!(verify(&t, c0.artifact.checkpoint_root, chain.root(0), c0_real).is_err());
    // The other chain's transition, verified against this chain's trusted checkpoint.
    assert_rejects(verify(&foreign, c0.artifact.checkpoint_root, chain.root(0), c0_real), "another checkpoint");
}

#[test]
fn deltas_since_swapped_missing_truncated_or_tampered_reject() {
    let chain = rich_chain();
    let c0 = genesis(&chain);
    let ops = vec![Op::Add(node("a1", "p1", "b9"))];
    let base = chain.root(7);
    let real = store_candidate(&chain, 7, &ops).resolution_root();
    let ck = c0.artifact.checkpoint_root;
    let honest_t = honest(&chain, &c0, 0, 7, &ops);

    let mut t = honest_t.clone();
    t.deltas.swap(1, 2);
    t.witness.deltas.swap(1, 2);
    assert_rejects(verify(&t, ck, base, real), "do not chain from the checkpoint's head");
    let mut t = honest_t.clone();
    t.deltas.remove(3);
    t.witness.deltas.remove(3);
    assert_rejects(verify(&t, ck, base, real), "do not chain from the checkpoint's head");
    let mut t = honest_t.clone();
    t.deltas.remove(0);
    t.witness.deltas.remove(0);
    assert_rejects(verify(&t, ck, base, real), "do not chain from the checkpoint's head");
    let mut t = honest_t.clone();
    t.deltas.pop();
    t.witness.deltas.pop();
    assert_rejects(verify(&t, ck, base, real), "do not end at the write's base");
    let mut t = honest_t.clone();
    t.deltas.clear();
    t.witness.deltas.clear();
    assert_rejects(verify(&t, ck, base, real), "do not end at the write's base");
    // A delta view without its records, or records for another delta.
    let mut t = honest_t.clone();
    t.witness.deltas.pop();
    assert_rejects(verify(&t, ck, base, real), "another number of deltas");
    let mut t = honest_t.clone();
    t.witness.deltas.swap(0, 1);
    assert!(verify(&t, ck, base, real).is_err());
    // A delta's clear records tampered (a removal dropped, an addition invented).
    let mut t = honest_t.clone();
    t.witness.deltas[0].removals.pop();
    assert_rejects(verify(&t, ck, base, real), "disagree with the delta's counts");
    let mut t = honest_t.clone();
    let first = t.witness.deltas[0].additions[0];
    t.witness.deltas[0].additions[0] = [first[0], first[1], first[2] + 1];
    assert!(verify(&t, ck, base, real).is_err());
    // A term-less delta view.
    let mut t = honest_t.clone();
    t.deltas[2] = chain.commitments[3].verifier_commitment_termless().unwrap();
    assert!(verify(&t, ck, base, real).is_err());
    // A foreign delta spliced in (a sibling of d1 off the base).
    let sibling = store_candidate(&chain, 0, &[Op::Add(node("q", "p1", "r"))]);
    let mut t = honest_t.clone();
    t.deltas[0] = sibling.verifier_commitment().unwrap();
    assert!(verify(&t, ck, base, real).is_err());
    verify(&honest_t, ck, base, real).unwrap();
}

#[test]
fn a_diff_changed_after_proving_rejects() {
    let chain = rich_chain();
    let c0 = genesis(&chain);
    let ck = c0.artifact.checkpoint_root;
    let ops = base_ops();
    let real = store_candidate(&chain, 0, &ops).resolution_root();
    let honest_t = WriteTransition::prove(&c0, &[], diff_of(&ops)).unwrap();
    // Unsorted / duplicate / conflicting diffs.
    let mut t = honest_t.clone();
    t.diff.additions.reverse();
    assert_rejects(verify(&t, ck, chain.root(0), real), "not strictly sorted");
    let mut t = honest_t.clone();
    let dup = t.diff.additions[0].clone();
    t.diff.additions.insert(0, dup);
    assert_rejects(verify(&t, ck, chain.root(0), real), "not strictly sorted");
    let mut t = honest_t.clone();
    t.diff.removals.push(t.diff.additions[0].clone());
    t.diff.removals.sort();
    assert_rejects(verify(&t, ck, chain.root(0), real), "adds and removes the same triple");
    // A triple dropped: its candidate no longer recomputes (or the paths no longer line up).
    let mut t = honest_t.clone();
    t.diff.removals.clear();
    assert!(verify(&t, ck, chain.root(0), real).is_err());
    // One object swapped for another existing term.
    let mut t = honest_t.clone();
    t.diff.additions[0].object = CanonicalObject::Node(iri("b2").into_bytes());
    t.diff.additions.sort();
    assert!(verify(&t, ck, chain.root(0), real).is_err());
}

#[test]
fn wire_rejects_malformed_bytes() {
    let chain = rich_chain();
    let c0 = genesis(&chain);
    let ops = vec![Op::Add(node("a1", "p1", "new-o")), Op::Remove(node("a5", "p1", "b5"))];
    let t = honest(&chain, &c0, 0, 4, &ops);
    let bytes = t.encode().unwrap();
    let decoded = WriteTransition::decode(&bytes).unwrap();
    assert_eq!(decoded.encode().unwrap(), bytes);
    let mut flipped = bytes.clone();
    flipped[40] ^= 1;
    assert!(WriteTransition::decode(&flipped).is_err());
    assert!(WriteTransition::decode(&bytes[..bytes.len() - 1]).is_err());
    assert!(WriteTransition::decode(&bytes[..20]).is_err());
    let oversized = vec![0u8; MAX_WRITE_TRANSITION_BYTES + 1];
    assert!(WriteTransition::decode(&oversized).is_err());
    let real = store_candidate(&chain, 4, &ops).resolution_root();
    assert!(decode_and_verify_write_transition(c0.artifact.checkpoint_root, chain.root(4), real, &flipped).is_err());
}

/// Measurement (release): verify time and sizes for a 10-triple diff over a fresh checkpoint.
#[test]
#[ignore]
fn measure_write_transition_costs() {
    use std::time::Instant;
    for n in [1_000usize, 10_000, 100_000] {
        let base: Vec<ValueTriple> = (0..n)
            .map(|i| node(&format!("s{}", i / 4), &format!("p{}", i % 7), &format!("o{i}")))
            .collect();
        let chain = Chain::new(base);
        let (c0, _) = genesis_checkpoint(&chain.commitments[0]).unwrap();
        // 10 triples: 4 additions with new terms, 2 additions over existing terms, 4 removals.
        let mut ops = Vec::new();
        for i in 0..4 {
            ops.push(Op::Add(node(&format!("new{i}"), "p1", &format!("newo{i}"))));
        }
        for i in 0..2 {
            ops.push(Op::Add(node(&format!("s{i}"), "p3", &format!("o{}", n - 1 - i))));
        }
        for i in 0..4 {
            let j = i * (n / 4) + 1;
            ops.push(Op::Remove(node(&format!("s{}", j / 4), &format!("p{}", j % 7), &format!("o{j}"))));
        }
        let real = store_candidate(&chain, 0, &ops).resolution_root();
        let t = Instant::now();
        let transition = WriteTransition::prove(&c0, &[], diff_of(&ops)).unwrap();
        let prove_time = t.elapsed();
        let bytes = transition.encode().unwrap();
        let mut witness_only = transition.clone();
        witness_only.diff = WriteDiff::default();
        witness_only.witness = WriteTransitionWitness::default();
        let shell = witness_only.encode().unwrap().len();
        let root = c0.artifact.checkpoint_root;
        decode_and_verify_write_transition(root, chain.root(0), real, &bytes).unwrap();
        let runs = 20;
        let t = Instant::now();
        for _ in 0..runs {
            decode_and_verify_write_transition(root, chain.root(0), real, &bytes).unwrap();
        }
        let verify_time = t.elapsed() / runs;
        let paths = transition.witness.nv_term_paths.len() + transition.witness.pred_term_paths.len() + transition.witness.triple_paths.len();
        eprintln!(
            "n={n}: prove {prove_time:?}, verify {verify_time:?}, transition {} B (checkpoint artifact + framing {} B, diff + witness {} B, {paths} paths)",
            bytes.len(),
            shell,
            bytes.len() - shell,
        );
    }
}
