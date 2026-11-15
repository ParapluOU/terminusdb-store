//! Proven-checkpoint tests: honest round trips over real store chains, the from-scratch equality
//! of chained checkpoints, the sorted-id assumption, every adversarial forgery the design names,
//! the wire formats, and (ignored) the measurement run.

use std::io;
use std::sync::Arc;

use tdb_succinct::TdbDataType;

use super::inductive::{new_terms, prove_checkpoint_inner, Forgery};
use super::*;
use crate::layer::{Layer, ValueTriple};
use crate::proof::argument::StoreProverContext;
use crate::proof::commitment::{LayerCommitment, VerifierCommitment, VerifierLayer};
use crate::proof::{CanonicalObject, CanonicalTriple, ProofRecord};
use crate::store::sync::{open_sync_memory_store, SyncStoreLayer};

pub(super) fn iri(s: &str) -> String {
    format!("http://ex/{s}")
}

pub(super) fn node(s: &str, p: &str, o: &str) -> ValueTriple {
    ValueTriple::new_node(&iri(s), &iri(p), &iri(o))
}

pub(super) enum Op {
    Add(ValueTriple),
    Remove(ValueTriple),
}

/// A real store chain: a base layer plus deltas, with each layer's store handle and sidecar.
pub(super) struct Chain {
    pub(super) layers: Vec<SyncStoreLayer>,
    pub(super) commitments: Vec<Arc<LayerCommitment>>,
}

impl Chain {
    pub(super) fn new(base: Vec<ValueTriple>) -> Self {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for t in base {
            builder.add_value_triple(t).unwrap();
        }
        let layer = builder.commit().unwrap();
        let commitment = layer.proof_commitment().unwrap();
        Chain { layers: vec![layer], commitments: vec![commitment] }
    }

    pub(super) fn push(&mut self, ops: Vec<Op>) {
        let builder = self.layers.last().unwrap().open_write().unwrap();
        for op in ops {
            match op {
                Op::Add(t) => builder.add_value_triple(t).unwrap(),
                Op::Remove(t) => builder.remove_value_triple(t).unwrap(),
            }
        }
        let layer = builder.commit().unwrap();
        self.commitments.push(layer.proof_commitment().unwrap());
        self.layers.push(layer);
    }

    pub(super) fn deltas(&self, from: usize, to: usize) -> Vec<&LayerCommitment> {
        self.commitments[from + 1..=to].iter().map(|c| c.as_ref()).collect()
    }

    pub(super) fn root(&self, i: usize) -> ProofHash {
        self.commitments[i].resolution_root()
    }
}

pub(super) fn genesis(chain: &Chain) -> CheckpointState {
    let (state, proof) = genesis_checkpoint(&chain.commitments[0]).unwrap();
    let view = chain.commitments[0].verifier_commitment().unwrap();
    verify_genesis_checkpoint(view.verifier_view(), chain.root(0), &state.artifact, &proof).unwrap();
    state
}

pub(super) fn views(deltas: &[&LayerCommitment]) -> Vec<VerifierCommitment> {
    deltas.iter().map(|d| d.verifier_commitment().unwrap()).collect()
}

fn verify(
    prev: &CheckpointArtifact,
    deltas: &[&LayerCommitment],
    head_root: ProofHash,
    next: &CheckpointArtifact,
    proof: &CheckpointProof,
) -> io::Result<()> {
    let owned = views(deltas);
    let refs: Vec<_> = owned.iter().map(|v| v.verifier_view()).collect();
    verify_checkpoint(prev, &refs, head_root, next, proof)
}

/// Prove and verify `from -> to`.
pub(super) fn step(chain: &Chain, prev: &CheckpointState, from: usize, to: usize) -> (CheckpointState, CheckpointProof) {
    let deltas = chain.deltas(from, to);
    let (next, proof) = prove_checkpoint(prev, &deltas).unwrap();
    verify(&prev.artifact, &deltas, chain.root(to), &next.artifact, &proof).unwrap();
    (next, proof)
}

/// The checkpoint of layer `i` built from scratch out of the store's own materialized view.
fn scratch(chain: &Chain, i: usize) -> CheckpointArtifact {
    let layer = &chain.layers[i];
    let present: Vec<IdTriple> = layer.triples().map(|t| [t.subject, t.predicate, t.object]).collect();
    let nv = (1..=layer.node_and_value_count() as u64)
        .map(|id| CanonicalObject::from_store_object(&layer.id_object(id).unwrap()))
        .collect();
    let pred = (1..=layer.predicate_count() as u64)
        .map(|id| layer.id_predicate(id).unwrap().into_bytes())
        .collect();
    CheckpointState::from_parts(
        &StoreProverContext::cpu(),
        chain.root(i),
        chain.commitments[i].state.commitment_root,
        None,
        present,
        nv,
        pred,
    )
    .unwrap()
    .artifact
}

fn assert_same_state(a: &CheckpointArtifact, b: &CheckpointArtifact) {
    assert_eq!(a.state_digest().unwrap(), b.state_digest().unwrap());
}

/// base (with values) and seven deltas: removals, re-adds, new predicates, values, an empty delta.
pub(super) fn rich_chain() -> Chain {
    let mut chain = Chain::new(vec![
        node("a1", "p1", "b1"),
        node("a2", "p1", "b2"),
        node("a3", "p1", "b3"),
        node("a4", "p2", "b4"),
        ValueTriple::new_value(&iri("a1"), &iri("age"), u32::make_entry(&30u32)),
        ValueTriple::new_string_value(&iri("a2"), &iri("name"), "Ann"),
    ]);
    // d1: remove inherited, add with new terms.
    chain.push(vec![
        Op::Remove(node("a1", "p1", "b1")),
        Op::Remove(node("a2", "p1", "b2")),
        Op::Add(node("a5", "p1", "b5")),
        Op::Add(node("a3", "p1", "b6")),
    ]);
    // d2: a brand-new predicate and new values.
    chain.push(vec![
        Op::Add(node("a7", "p3", "b8")),
        Op::Add(ValueTriple::new_value(&iri("a7"), &iri("age"), u32::make_entry(&41u32))),
        Op::Add(ValueTriple::new_value(&iri("a7"), &iri("score"), f64::make_entry(&2.5f64))),
    ]);
    // d3: remove what d2 added, re-add what d1 removed.
    chain.push(vec![
        Op::Remove(node("a7", "p3", "b8")),
        Op::Add(node("a1", "p1", "b1")),
    ]);
    // d4: empty.
    chain.push(vec![]);
    // d5: a single addition.
    chain.push(vec![Op::Add(node("a9", "p1", "b9"))]);
    // d6: a single removal.
    chain.push(vec![Op::Remove(node("a4", "p2", "b4"))]);
    // d7: mixed.
    chain.push(vec![
        Op::Add(node("a4", "p2", "b4")),
        Op::Add(node("a10", "p4", "b10")),
        Op::Remove(node("a3", "p1", "b3")),
    ]);
    chain
}

// ---------------------------------------------------------------------------
// The sorted-id assumption
// ---------------------------------------------------------------------------

#[test]
fn delta_new_ids_are_assigned_in_sorted_term_order() {
    let mut chain = Chain::new(vec![node("m", "p", "n")]);
    // New terms of every kind, inserted out of order.
    chain.push(vec![
        Op::Add(node("zeta", "q2", "alpha")),
        Op::Add(ValueTriple::new_string_value(&iri("m"), &iri("q1"), "zz")),
        Op::Add(ValueTriple::new_string_value(&iri("m"), &iri("q1"), "aa")),
        Op::Add(ValueTriple::new_value(&iri("m"), &iri("q0"), u32::make_entry(&9u32))),
        Op::Add(ValueTriple::new_value(&iri("m"), &iri("q0"), u32::make_entry(&2u32))),
        Op::Add(ValueTriple::new_value(&iri("m"), &iri("q3"), i64::make_entry(&-7))),
        Op::Add(ValueTriple::new_value(&iri("m"), &iri("q3"), i64::make_entry(&5i64))),
        Op::Add(ValueTriple::new_value(&iri("m"), &iri("q4"), f64::make_entry(&-1.5))),
        Op::Add(ValueTriple::new_value(&iri("m"), &iri("q4"), f64::make_entry(&0.25f64))),
        Op::Add(ValueTriple::new_value(&iri("m"), &iri("q5"), bool::make_entry(&true))),
        Op::Add(node("beta", "q2", "m")),
    ]);
    let delta = &chain.commitments[1];
    let parent = &chain.commitments[0];
    assert!(delta.state.node_value_count > parent.state.node_value_count + 8);
    // The store numbers a child's new terms parent_count + 1 .. in canonical sorted order.
    let (nv, pred) = new_terms(&delta.state, &delta.node_value_catalog, &delta.predicate_catalog).unwrap();
    assert!(nv.windows(2).all(|w| w[0] < w[1]));
    assert!(pred.windows(2).all(|w| w[0] < w[1]));
    for (i, record) in delta.node_value_catalog.iter().enumerate() {
        let _ = i;
        let by_id = &nv[(record.object_id - delta.state.parent_node_value_count - 1) as usize];
        assert_eq!(by_id, &record.triple.object);
    }
    // A catalog whose ids run against the term order is refused.
    let mut swapped = delta.node_value_catalog.clone();
    let (a, b) = (swapped[0].object_id, swapped[1].object_id);
    swapped[0].object_id = b;
    swapped[1].object_id = a;
    assert!(new_terms(&delta.state, &swapped, &delta.predicate_catalog).is_err());
}

// ---------------------------------------------------------------------------
// Honest round trips
// ---------------------------------------------------------------------------

#[test]
fn genesis_round_trip_and_matches_scratch() {
    let chain = rich_chain();
    let c0 = genesis(&chain);
    assert_eq!(c0.artifact.present_count, 6);
    assert_same_state(&c0.artifact, &scratch(&chain, 0));
    assert_eq!(c0.artifact.present_commitments, chain.commitments[0].materialized_column_commitments);
}

#[test]
fn genesis_of_tiny_bases() {
    for base in [vec![], vec![node("x", "p", "y")]] {
        let chain = Chain::new(base);
        let c0 = genesis(&chain);
        assert_same_state(&c0.artifact, &scratch(&chain, 0));
    }
}

#[test]
fn genesis_forgeries_reject() {
    let chain = rich_chain();
    let (state, proof) = genesis_checkpoint(&chain.commitments[0]).unwrap();
    let view = chain.commitments[0].verifier_commitment().unwrap();
    let check = |artifact: &CheckpointArtifact, proof: &GenesisProof, root: ProofHash| {
        verify_genesis_checkpoint(view.verifier_view(), root, artifact, proof)
    };
    // Wrong root.
    assert!(check(&state.artifact, &proof, chain.root(1)).is_err());
    // Dropped / altered present triple.
    let mut dropped = proof.clone();
    dropped.present.pop();
    assert!(check(&state.artifact, &dropped, chain.root(0)).is_err());
    let mut altered = proof.clone();
    altered.present[0][2] = altered.present[1][2];
    assert!(check(&state.artifact, &altered, chain.root(0)).is_err());
    // Forged artifact (facts flipped, root recomputed).
    let mut forged = state.artifact.clone();
    forged.facts.float32_nan = true;
    forged.checkpoint_root = forged.compute_root().unwrap();
    let err = check(&forged, &proof, chain.root(0)).unwrap_err();
    assert!(err.to_string().contains("does not commit the base layer"), "{}", err);
    // A term-less base view cannot anchor genesis.
    let termless = chain.commitments[0].verifier_commitment_termless().unwrap();
    assert!(verify_genesis_checkpoint(termless.verifier_view(), chain.root(0), &state.artifact, &proof).is_err());
}

#[test]
fn one_delta_round_trip() {
    let chain = rich_chain();
    let c0 = genesis(&chain);
    let (c1, _) = step(&chain, &c0, 0, 1);
    assert_same_state(&c1.artifact, &scratch(&chain, 1));
    assert_eq!(c1.artifact.prev_checkpoint_root, Some(c0.artifact.checkpoint_root));
}

#[test]
fn many_deltas_with_removals_re_adds_new_predicates_and_an_empty_delta() {
    let chain = rich_chain();
    let c0 = genesis(&chain);
    let (c7, _) = step(&chain, &c0, 0, 7);
    assert_same_state(&c7.artifact, &scratch(&chain, 7));
    // Each delta on its own, including the empty one (d4) and single-row ones (d5, d6).
    let mut prev = c0;
    for i in 1..=7 {
        let (next, _) = step(&chain, &prev, i - 1, i);
        assert_same_state(&next.artifact, &scratch(&chain, i));
        prev = next;
    }
}

#[test]
fn chained_checkpoints_equal_a_from_scratch_checkpoint() {
    let chain = rich_chain();
    let c0 = genesis(&chain);
    let (c3, _) = step(&chain, &c0, 0, 3);
    let (c7, _) = step(&chain, &c3, 3, 7);
    let direct = scratch(&chain, 7);
    assert_same_state(&c7.artifact, &direct);
    assert_eq!(c7.artifact.present_commitments, direct.present_commitments);
    assert_eq!(c7.artifact.nv_dict_segments, direct.nv_dict_segments);
    assert_eq!(c7.artifact.triple_index_root, direct.triple_index_root);
    assert_eq!(c7.artifact.prev_checkpoint_root, Some(c3.artifact.checkpoint_root));
    // The verifier of C_3 -> C_7 never sees C_0 or the deltas before 3.
    assert_eq!(c7.present(), scratch_present(&chain, 7).as_slice());
}

fn scratch_present(chain: &Chain, i: usize) -> Vec<IdTriple> {
    let mut v: Vec<IdTriple> = chain.layers[i].triples().map(|t| [t.subject, t.predicate, t.object]).collect();
    v.sort();
    v
}

#[test]
fn a_chain_emptied_and_refilled() {
    let mut chain = Chain::new(vec![node("a", "p", "b"), node("c", "p", "d")]);
    chain.push(vec![Op::Remove(node("a", "p", "b")), Op::Remove(node("c", "p", "d"))]);
    chain.push(vec![Op::Add(node("a", "p", "b"))]);
    let c0 = genesis(&chain);
    let (c1, _) = step(&chain, &c0, 0, 1);
    assert_eq!(c1.artifact.present_count, 0);
    let (c2, _) = step(&chain, &c1, 1, 2);
    assert_same_state(&c2.artifact, &scratch(&chain, 2));
}

#[test]
fn nan_facts_follow_the_new_terms() {
    let mut chain = Chain::new(vec![node("a", "p", "b")]);
    chain.push(vec![Op::Add(ValueTriple::new_value(&iri("a"), &iri("x"), f64::make_entry(&f64::NAN)))]);
    let c0 = genesis(&chain);
    assert!(!c0.artifact.facts.float64_nan);
    let (c1, _) = step(&chain, &c0, 0, 1);
    assert!(c1.artifact.facts.float64_nan);
    assert!(!c1.artifact.facts.float32_nan);
    // The forged fact (NaN hidden) rejects.
    let deltas = chain.deltas(0, 1);
    let (forged, proof) = prove_checkpoint_inner(&StoreProverContext::cpu(), &c0, &deltas, &Forgery::FlipNanFact).unwrap();
    let err = verify(&c0.artifact, &deltas, chain.root(1), &forged.artifact, &proof).unwrap_err();
    assert!(err.to_string().contains("facts"), "{}", err);
}

// ---------------------------------------------------------------------------
// Adversarial
// ---------------------------------------------------------------------------

/// Prove with `forgery` and require the verifier to reject with a message containing one of the
/// `|`-separated alternatives in `expect`.
fn reject(chain: &Chain, prev: &CheckpointState, from: usize, to: usize, forgery: Forgery, expect: &str) {
    let deltas = chain.deltas(from, to);
    let (forged, proof) = prove_checkpoint_inner(&StoreProverContext::cpu(), prev, &deltas, &forgery).unwrap();
    let err = verify(&prev.artifact, &deltas, chain.root(to), &forged.artifact, &proof)
        .expect_err(&format!("{forgery:?} must reject"));
    let message = err.to_string();
    assert!(expect.split('|').any(|e| message.contains(e)), "{:?}: {}", forgery, err);
}

#[test]
fn forged_present_relations_reject() {
    let chain = rich_chain();
    let c0 = genesis(&chain);
    // Dropped / extra triple: the count no longer follows from the deltas.
    reject(&chain, &c0, 0, 3, Forgery::DropPresent(2), "present count");
    reject(&chain, &c0, 0, 3, Forgery::ExtraPresent([1, 1, 1]), "present count");
    // Count-preserving forgeries reach the link.
    reject(&chain, &c0, 0, 3, Forgery::ReplacePresent(1, [1, 1, 1]), "checkpoint present link");
    reject(&chain, &c0, 0, 3, Forgery::DuplicatePresent(1), "checkpoint present link");
    // Same multiset, out of order: only sortedness catches it.
    let deltas = chain.deltas(0, 3);
    let (forged, proof) =
        prove_checkpoint_inner(&StoreProverContext::cpu(), &c0, &deltas, &Forgery::SwapPresent(0)).unwrap();
    let err = verify(&c0.artifact, &deltas, chain.root(3), &forged.artifact, &proof).unwrap_err();
    assert!(err.to_string().contains("checkpoint sortedness"), "{}", err);
    assert!(proof.sorted.is_some());
    // A nonzero padding row: only the zero-check catches it.
    let honest = prove_checkpoint(&c0, &deltas).unwrap().0;
    assert!(!(honest.artifact.present_count as usize).is_power_of_two());
    let (forged, proof) =
        prove_checkpoint_inner(&StoreProverContext::cpu(), &c0, &deltas, &Forgery::NonZeroPadding).unwrap();
    let err = verify(&c0.artifact, &deltas, chain.root(3), &forged.artifact, &proof).unwrap_err();
    assert!(err.to_string().contains("checkpoint padding zero-check"), "{}", err);
}

#[test]
fn forged_dictionary_appends_reject() {
    let chain = rich_chain();
    let c0 = genesis(&chain);
    for forgery in [Forgery::NvDictValue, Forgery::NvDictOffset, Forgery::NvDictExtra] {
        reject(&chain, &c0, 0, 2, forgery, "dictionary append");
    }
}

#[test]
fn forged_term_index_updates_reject() {
    let chain = rich_chain();
    let c0 = genesis(&chain);
    reject(&chain, &c0, 0, 1, Forgery::NvIndexWrongKey, "term index does not insert|does not open the index root");
}

/// A malicious delta (built directly, bypassing the store) on top of `parent`.
fn malicious_delta(
    parent: &LayerCommitment,
    additions: Vec<ProofRecord>,
    new_nv: Vec<(CanonicalObject, u64)>,
) -> LayerCommitment {
    let nv_count = parent.state.node_value_count + new_nv.len() as u64;
    LayerCommitment::from_native_delta(
        Some(parent.state.commitment_root),
        additions,
        Vec::new(),
        Vec::new(),
        parent.state.node_value_count,
        parent.state.predicate_count,
        nv_count,
        parent.state.predicate_count,
        new_nv,
        Vec::new(),
    )
    .unwrap()
}

fn record(chain: &Chain, t: IdTriple) -> ProofRecord {
    let layer = &chain.layers[0];
    let subject = match CanonicalObject::from_store_object(&layer.id_object(t[0]).unwrap()) {
        CanonicalObject::Node(s) => s,
        _ => panic!("subject is a node"),
    };
    let predicate = layer.id_predicate(t[1]).unwrap().into_bytes();
    let object = CanonicalObject::from_store_object(&layer.id_object(t[2]).unwrap());
    ProofRecord {
        subject_id: t[0],
        predicate_id: t[1],
        object_id: t[2],
        triple: CanonicalTriple { subject, predicate, object },
    }
}

#[test]
fn re_adding_a_present_triple_rejects() {
    let chain = Chain::new(vec![node("a", "p", "b"), node("c", "p", "d")]);
    let c0 = genesis(&chain);
    let t = c0.present()[0];
    let delta = malicious_delta(&chain.commitments[0], vec![record(&chain, t)], Vec::new());
    let deltas = [&delta];
    // The honest prover refuses.
    assert!(prove_checkpoint(&c0, &deltas).is_err());
    // A prover that keeps the triple once ("both added and still present") is caught by the
    // triple index: the addition has no non-membership witness.
    let (forged, proof) =
        prove_checkpoint_inner(&StoreProverContext::cpu(), &c0, &deltas, &Forgery::AllowReAdd).unwrap();
    let err = verify(&c0.artifact, &deltas, delta.resolution_root(), &forged.artifact, &proof).unwrap_err();
    assert!(err.to_string().contains("already a member"), "{}", err);
}

#[test]
fn re_minting_a_term_rejects_and_forged_absence_rejects() {
    let chain = Chain::new(vec![node("a", "p", "b")]);
    let c0 = genesis(&chain);
    // A delta minting the existing node `a` again at a new id.
    let existing = c0.node_value(1).unwrap().clone();
    let delta = malicious_delta(&chain.commitments[0], Vec::new(), vec![(existing, c0.artifact.node_value_count + 1)]);
    let deltas = [&delta];
    assert!(prove_checkpoint(&c0, &deltas).is_err());
    let (forged, proof) =
        prove_checkpoint_inner(&StoreProverContext::cpu(), &c0, &deltas, &Forgery::ForgeAbsence).unwrap();
    let err = verify(&c0.artifact, &deltas, delta.resolution_root(), &forged.artifact, &proof).unwrap_err();
    assert!(err.to_string().contains("does not open the index root"), "{}", err);
}

#[test]
fn triple_index_forgery_rejects() {
    let chain = rich_chain();
    let c0 = genesis(&chain);
    reject(&chain, &c0, 0, 1, Forgery::TripleIndexSkipRemoval, "triple index does not follow|does not open the index root");
}

#[test]
fn checkpoint_bound_elsewhere_rejects() {
    let chain = rich_chain();
    let c0 = genesis(&chain);
    let deltas = chain.deltas(0, 3);
    let (c3, proof) = prove_checkpoint(&c0, &deltas).unwrap();
    // Another trusted head.
    let err = verify(&c0.artifact, &deltas, chain.root(2), &c3.artifact, &proof).unwrap_err();
    assert!(err.to_string().contains("another head"), "{}", err);
    // Another previous checkpoint (the C_1 of the same chain, and a foreign genesis).
    let (c1, _) = step(&chain, &c0, 0, 1);
    let err = verify(&c1.artifact, &deltas, chain.root(3), &c3.artifact, &proof).unwrap_err();
    assert!(err.to_string().contains("another previous checkpoint"), "{}", err);
    let foreign = genesis(&Chain::new(vec![node("q", "r", "s")]));
    assert!(verify(&foreign.artifact, &deltas, chain.root(3), &c3.artifact, &proof).is_err());
    // A next artifact re-pointed at another previous checkpoint (root recomputed) fails the
    // transcript-bound link.
    let mut repointed = c3.artifact.clone();
    repointed.prev_checkpoint_root = Some(c1.artifact.checkpoint_root);
    repointed.checkpoint_root = repointed.compute_root().unwrap();
    assert!(verify(&c1.artifact, &chain.deltas(1, 3), chain.root(3), &repointed, &proof).is_err());
}

#[test]
fn deltas_out_of_order_or_skipped_reject() {
    let chain = rich_chain();
    let c0 = genesis(&chain);
    let deltas = chain.deltas(0, 3);
    let (c3, proof) = prove_checkpoint(&c0, &deltas).unwrap();
    let reordered = vec![deltas[1], deltas[0], deltas[2]];
    let err = verify(&c0.artifact, &reordered, chain.root(3), &c3.artifact, &proof).unwrap_err();
    assert!(err.to_string().contains("do not chain"), "{}", err);
    let skipped = vec![deltas[0], deltas[2]];
    let err = verify(&c0.artifact, &skipped, chain.root(3), &c3.artifact, &proof).unwrap_err();
    assert!(err.to_string().contains("do not chain"), "{}", err);
    assert!(prove_checkpoint(&c0, &reordered).is_err());
    assert!(prove_checkpoint(&c0, &skipped).is_err());
    // Starting past C_j's head.
    assert!(verify(&c0.artifact, &chain.deltas(1, 3), chain.root(3), &c3.artifact, &proof).is_err());
}

#[test]
fn proof_replayed_for_another_delta_set_rejects() {
    // Two branches off the same base.
    let base = vec![node("a", "p", "b"), node("c", "p", "d"), node("e", "p", "f")];
    let mut left = Chain::new(base.clone());
    left.push(vec![Op::Add(node("g", "p", "h")), Op::Remove(node("a", "p", "b"))]);
    let mut right = Chain { layers: vec![left.layers[0].clone()], commitments: vec![left.commitments[0].clone()] };
    right.push(vec![Op::Add(node("g", "p", "i")), Op::Remove(node("c", "p", "d"))]);
    let c0 = genesis(&left);
    let (cl, pl) = step(&left, &c0, 0, 1);
    let (cr, pr) = step(&right, &c0, 0, 1);
    let ld = left.deltas(0, 1);
    let rd = right.deltas(0, 1);
    // Left proof + left artifact, right deltas.
    assert!(verify(&c0.artifact, &rd, left.root(1), &cl.artifact, &pl).is_err());
    assert!(verify(&c0.artifact, &rd, right.root(1), &cl.artifact, &pl).is_err());
    // Left proof with the right artifact and deltas.
    assert!(verify(&c0.artifact, &rd, right.root(1), &cr.artifact, &pl).is_err());
    // Right proof for the left statement.
    assert!(verify(&c0.artifact, &ld, left.root(1), &cl.artifact, &pr).is_err());
    // Each verifies for its own statement.
    verify(&c0.artifact, &rd, right.root(1), &cr.artifact, &pr).unwrap();
}

#[test]
fn tampered_proof_parts_reject() {
    let chain = rich_chain();
    let c0 = genesis(&chain);
    let deltas = chain.deltas(0, 2);
    let (c2, proof) = prove_checkpoint(&c0, &deltas).unwrap();
    let check = |proof: &CheckpointProof| verify(&c0.artifact, &deltas, chain.root(2), &c2.artifact, proof);
    // A shipped delta record that is not the committed one.
    let mut p = proof.clone();
    p.deltas[0].additions[0][2] = p.deltas[0].removals[0][2];
    assert!(check(&p).is_err());
    // A forged term-tree frontier.
    let mut p = proof.clone();
    p.nv_tree_frontier[0][0] ^= 1;
    assert!(check(&p).is_err());
    // Evaluation claims.
    let mut p = proof.clone();
    p.next_present_evals[0] = p.next_present_evals[1];
    assert!(check(&p).is_err());
    // Missing sortedness / padding proofs.
    let mut p = proof.clone();
    p.sorted = None;
    assert!(check(&p).is_err());
    let mut p = proof;
    p.padding = None;
    assert!(check(&p).is_err());
}

// ---------------------------------------------------------------------------
// Wire
// ---------------------------------------------------------------------------

#[test]
fn wire_round_trip_and_byte_in_verification() {
    let chain = rich_chain();
    let (c0, genesis_proof) = genesis_checkpoint(&chain.commitments[0]).unwrap();
    let base_bytes = chain.commitments[0].verifier_commitment().unwrap().encode().unwrap();
    let c0_bytes = c0.artifact.encode().unwrap();
    let g_bytes = genesis_proof.encode().unwrap();
    let decoded = decode_and_verify_genesis_checkpoint(&base_bytes, chain.root(0), &c0_bytes, &g_bytes).unwrap();
    assert_eq!(decoded, c0.artifact);

    let deltas = chain.deltas(0, 7);
    let (c7, proof) = prove_checkpoint(&c0, &deltas).unwrap();
    let delta_bytes: Vec<Vec<u8>> =
        deltas.iter().map(|d| d.verifier_commitment().unwrap().encode().unwrap()).collect();
    let delta_refs: Vec<&[u8]> = delta_bytes.iter().map(|b| b.as_slice()).collect();
    let c7_bytes = c7.artifact.encode().unwrap();
    let proof_bytes = proof.encode().unwrap();
    assert_eq!(CheckpointArtifact::decode(&c7_bytes).unwrap(), c7.artifact);
    assert_eq!(CheckpointProof::decode(&proof_bytes).unwrap().encode().unwrap(), proof_bytes);
    let verified = decode_and_verify_checkpoint(
        c0.artifact.checkpoint_root, &c0_bytes, &delta_refs, chain.root(7), &c7_bytes, &proof_bytes,
    )
    .unwrap();
    assert_eq!(verified, c7.artifact);

    // An untrusted previous checkpoint, corrupted bytes, a foreign magic.
    assert!(decode_and_verify_checkpoint(
        c7.artifact.checkpoint_root, &c0_bytes, &delta_refs, chain.root(7), &c7_bytes, &proof_bytes,
    )
    .is_err());
    let mut corrupt = proof_bytes.clone();
    corrupt[40] ^= 1;
    assert!(CheckpointProof::decode(&corrupt).is_err());
    assert!(CheckpointProof::decode(&c7_bytes).is_err());
    assert!(CheckpointArtifact::decode(&proof_bytes).is_err());
    assert!(GenesisProof::decode(&c0_bytes).is_err());
    let mut truncated = g_bytes.clone();
    truncated.truncate(g_bytes.len() - 1);
    assert!(GenesisProof::decode(&truncated).is_err());
}

// ---------------------------------------------------------------------------
// Measurement (run with `--ignored --nocapture`)
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn measure_checkpoint_costs() {
    use std::time::Instant;
    for n in [1_000usize, 10_000, 100_000] {
        let base: Vec<ValueTriple> = (0..n)
            .map(|i| node(&format!("s{}", i / 4), &format!("p{}", i % 7), &format!("o{i}")))
            .collect();
        let mut chain = Chain::new(base);
        // A 100-triple delta: 60 additions (new terms) and 40 removals.
        let mut ops = Vec::new();
        for i in 0..60 {
            ops.push(Op::Add(node(&format!("new{i}"), "p1", &format!("newo{i}"))));
        }
        for i in 0..40 {
            let j = i * (n / 40);
            ops.push(Op::Remove(node(&format!("s{}", j / 4), &format!("p{}", j % 7), &format!("o{j}"))));
        }
        chain.push(ops);
        let t = Instant::now();
        let (c0, _) = genesis_checkpoint(&chain.commitments[0]).unwrap();
        let genesis_time = t.elapsed();
        let deltas = chain.deltas(0, 1);
        let t = Instant::now();
        let (c1, proof) = prove_checkpoint(&c0, &deltas).unwrap();
        let prove_time = t.elapsed();
        let bytes = proof.encode().unwrap();
        let owned = views(&deltas);
        let refs: Vec<_> = owned.iter().map(|v| v.verifier_view()).collect();
        // Warm the verifier's setups once, then time.
        verify_checkpoint(&c0.artifact, &refs, chain.root(1), &c1.artifact, &proof).unwrap();
        let t = Instant::now();
        verify_checkpoint(&c0.artifact, &refs, chain.root(1), &c1.artifact, &proof).unwrap();
        let verify_time = t.elapsed();
        println!(
            "n={n}: genesis {genesis_time:?}, prove {prove_time:?}, proof {} bytes ({:.1} KiB), artifact {} bytes, verify {verify_time:?}",
            bytes.len(),
            bytes.len() as f64 / 1024.0,
            c1.artifact.encode().unwrap().len(),
        );
    }
}
