//! Step 4: checkpoint records (what the chain verifies to record a checkpoint), the from-scratch
//! rebuild of a recorded checkpoint, and the diff a child layer spells.

use super::tests::{genesis, node, rich_chain, Chain, Op};
use super::*;
use crate::proof::argument::StoreProverContext;
use crate::proof::commitment::LayerCommitment;
use crate::proof::CanonicalTriple;

fn record(chain: &Chain, prev: &CheckpointState, from: usize, to: usize) -> (CheckpointState, Vec<u8>) {
    let deltas: Vec<&LayerCommitment> = chain.deltas(from, to);
    let (next, record) = CheckpointRecord::prove(prev, &deltas).unwrap();
    (next, record.encode().unwrap())
}

fn err(r: std::io::Result<VerifiedCheckpointRecord>) -> String {
    r.err().expect("the record must reject").to_string()
}

#[test]
fn a_record_verifies_byte_in_and_names_its_deltas() {
    let chain = rich_chain();
    let c0 = genesis(&chain);
    let (c3, bytes) = record(&chain, &c0, 0, 3);
    let verified = decode_and_verify_checkpoint_record(c0.artifact.checkpoint_root, chain.root(3), &bytes).unwrap();
    assert_eq!(verified.artifact.checkpoint_root, c3.artifact.checkpoint_root);
    assert_eq!(verified.delta_roots, (1..=3).map(|i| chain.root(i)).collect::<Vec<_>>());
    // and inductively on from the recorded one
    let (c7, bytes) = record(&chain, &c3, 3, 7);
    let verified = decode_and_verify_checkpoint_record(c3.artifact.checkpoint_root, chain.root(7), &bytes).unwrap();
    assert_eq!(verified.artifact.checkpoint_root, c7.artifact.checkpoint_root);
    assert_eq!(verified.artifact.prev_checkpoint_root, Some(c3.artifact.checkpoint_root));
}

#[test]
fn forged_foreign_and_stale_records_reject() {
    let chain = rich_chain();
    let c0 = genesis(&chain);
    let (c3, bytes) = record(&chain, &c0, 0, 3);

    // a forged artifact: the next checkpoint claims another present count (root recomputed so it
    // is self-consistent — the proof must catch it)
    let mut forged = CheckpointRecord::decode(&bytes).unwrap();
    forged.next.present_count += 1;
    forged.next.checkpoint_root = forged.next.compute_root().unwrap();
    let e = err(forged.verify(c0.artifact.checkpoint_root, chain.root(3)));
    assert!(!e.is_empty());
    // a tampered root without recompute
    let mut forged = CheckpointRecord::decode(&bytes).unwrap();
    forged.next.checkpoint_root[0] ^= 1;
    assert!(err(forged.verify(c0.artifact.checkpoint_root, chain.root(3))).contains("recompute"));
    // a delta dropped
    let mut forged = CheckpointRecord::decode(&bytes).unwrap();
    forged.deltas.pop();
    let _ = err(forged.verify(c0.artifact.checkpoint_root, chain.root(3)));
    // bytes flipped
    let mut flipped = bytes.clone();
    let mid = flipped.len() / 2;
    flipped[mid] ^= 0xff;
    let _ = err(decode_and_verify_checkpoint_record(c0.artifact.checkpoint_root, chain.root(3), &flipped));

    // the record names another head than the one the chain accepted
    assert!(err(decode_and_verify_checkpoint_record(c0.artifact.checkpoint_root, chain.root(2), &bytes)).contains("another head"));

    // stale base: the chain already recorded C_3, a record from C_0 no longer applies
    assert!(err(decode_and_verify_checkpoint_record(c3.artifact.checkpoint_root, chain.root(3), &bytes))
        .contains("another previous checkpoint"));

    // another chain: a record proven over a different catalog does not verify against this one's
    let other = Chain::new(vec![node("x1", "q1", "y1")]);
    let mut other = other;
    other.push(vec![Op::Add(node("x2", "q1", "y2"))]);
    let o0 = genesis(&other);
    let (_, other_bytes) = record(&other, &o0, 0, 1);
    assert!(err(decode_and_verify_checkpoint_record(c0.artifact.checkpoint_root, chain.root(1), &other_bytes))
        .contains("another previous checkpoint"));
    // even claiming the other chain's own roots, it cannot stand in for this chain's checkpoint
    let foreign = CheckpointRecord::decode(&other_bytes).unwrap();
    let mut spliced = CheckpointRecord::decode(&bytes).unwrap();
    spliced.deltas = foreign.deltas;
    let _ = err(spliced.verify(c0.artifact.checkpoint_root, chain.root(3)));
}

#[test]
fn a_recorded_checkpoint_rebuilds_from_its_head_layer() {
    let chain = rich_chain();
    let c0 = genesis(&chain);
    let (c3, _) = record(&chain, &c0, 0, 3);
    let context = StoreProverContext::cpu();
    let rebuilt = CheckpointState::of_layer(&context, &chain.layers[3], Some(c0.artifact.checkpoint_root)).unwrap();
    assert_eq!(rebuilt.artifact.checkpoint_root, c3.artifact.checkpoint_root);
    let genesis_rebuilt = CheckpointState::of_layer(&context, &chain.layers[0], None).unwrap();
    assert_eq!(genesis_rebuilt.artifact.checkpoint_root, c0.artifact.checkpoint_root);
    // the rebuilt state proves on like the proven one
    let (c7, _) = record(&chain, &rebuilt, 3, 7);
    let (c7b, _) = record(&chain, &c3, 3, 7);
    assert_eq!(c7.artifact.checkpoint_root, c7b.artifact.checkpoint_root);
}

#[test]
fn a_delta_spells_its_diff() {
    let chain = rich_chain();
    let d1 = &chain.commitments[1];
    let diff = WriteDiff::of_delta(d1).unwrap();
    let iri = super::tests::iri;
    let canonical = |s: &str, p: &str, o: &str| CanonicalTriple::node(iri(s).as_bytes(), iri(p).as_bytes(), iri(o).as_bytes());
    let mut additions = vec![canonical("a5", "p1", "b5"), canonical("a3", "p1", "b6")];
    let mut removals = vec![canonical("a1", "p1", "b1"), canonical("a2", "p1", "b2")];
    additions.sort();
    removals.sort();
    assert_eq!(diff, WriteDiff { additions, removals });
    assert!(WriteDiff::of_delta(&chain.commitments[0]).is_err(), "a base layer is not a delta");
    // the empty delta spells the empty diff
    assert_eq!(WriteDiff::of_delta(&chain.commitments[4]).unwrap(), WriteDiff::default());
}
