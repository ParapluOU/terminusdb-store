//! Adversarial audit tests (transition). See doc/proof-of-woql/checkpoint-proofs.md step 6.
//!
//! The ledger accepts the candidate root the verifier recomputes. If that recompute ever differs
//! from what the real store builds for the same diff (new-term numbering, value canonicalization),
//! a proposer could get a head accepted that no store layer has. These tests use the store's own
//! child layer as the oracle over datatypes and values the step-3 tests do not cover.

use std::io;

use tdb_succinct::tfc::decimal::Decimal;
use tdb_succinct::TdbDataType;

use super::tests::{genesis, iri, node, rich_chain, Chain, Op};
use super::*;
use crate::layer::ValueTriple;
use crate::proof::commitment::LayerCommitment;
use crate::proof::{CanonicalTriple, ProofHash};

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

/// The verifier's recompute must equal the store's child, and the store's root must be the only
/// candidate it accepts.
fn assert_matches_store(chain: &Chain, ck: &CheckpointState, ck_at: usize, base: usize, ops: &[Op]) {
    let deltas = chain.deltas(ck_at, base);
    let transition = WriteTransition::prove(ck, &deltas, diff_of(ops)).unwrap();
    let real = store_candidate(chain, base, ops);
    let recomputed: io::Result<LayerCommitment> =
        transition.verify(ck.artifact.checkpoint_root, chain.root(base), real.resolution_root());
    let recomputed = recomputed.unwrap_or_else(|e| panic!("the store's own candidate was rejected: {e}"));
    assert_eq!(recomputed.state, real.state);
    let other: ProofHash = [0xAB; 32];
    assert!(transition.verify(ck.artifact.checkpoint_root, chain.root(base), other).is_err());
}

fn exotic_values() -> Vec<Op> {
    let s = iri("ex");
    let v = |p: &str, e| Op::Add(ValueTriple::new_value(&s, &iri(p), e));
    vec![
        v("u8", u8::make_entry(&200u8)),
        v("u8", u8::make_entry(&3u8)),
        v("i8", i8::make_entry(&(-100i8))),
        v("i8", i8::make_entry(&(5i8))),
        v("u16", u16::make_entry(&65535u16)),
        v("i16", i16::make_entry(&(-2i16))),
        v("i32", i32::make_entry(&i32::MIN)),
        v("i32", i32::make_entry(&(-1i32))),
        v("i32", i32::make_entry(&(1i32))),
        v("u64", u64::make_entry(&u64::MAX)),
        v("u64", u64::make_entry(&0u64)),
        v("i64", i64::make_entry(&i64::MIN)),
        v("i64", i64::make_entry(&i64::MAX)),
        v("f32", f32::make_entry(&(-0.0f32))),
        v("f32", f32::make_entry(&(0.0f32))),
        v("f32", f32::make_entry(&f32::NEG_INFINITY)),
        v("f64", f64::make_entry(&(-0.0f64))),
        v("f64", f64::make_entry(&(0.0f64))),
        v("f64", f64::make_entry(&f64::NAN)),
        v("f64", f64::make_entry(&f64::INFINITY)),
        v("f64", f64::make_entry(&(-1e300f64))),
        v("dec", Decimal::make_entry(&Decimal::new("-12.50".to_string()).unwrap())),
        v("dec", Decimal::make_entry(&Decimal::new("3.1".to_string()).unwrap())),
        v("dec", Decimal::make_entry(&Decimal::new("0".to_string()).unwrap())),
        v("big", rug::Integer::make_entry(&rug::Integer::from(-7))),
        v("big", rug::Integer::make_entry(&(rug::Integer::from(1) << 200))),
        v("bool", bool::make_entry(&false)),
        // A string spelling a node IRI, and the node itself: distinct terms.
        Op::Add(ValueTriple::new_string_value(&s, &iri("str"), &iri("b1"))),
        Op::Add(ValueTriple::new_string_value(&s, &iri("str"), "")),
        Op::Add(ValueTriple::new_string_value(&s, &iri("str"), "\u{0}")),
        Op::Add(ValueTriple::new_string_value(&s, &iri("str"), "é")),
        // A predicate IRI that is also used as a subject and an object.
        Op::Add(node("p1", "p1", "p1")),
    ]
}

#[test]
fn exotic_new_values_number_as_the_store_does_right_after_genesis() {
    let chain = rich_chain();
    let c0 = genesis(&chain);
    assert_matches_store(&chain, &c0, 0, 0, &exotic_values());
}

#[test]
fn exotic_new_values_number_as_the_store_does_after_deltas_since() {
    let chain = rich_chain();
    let c0 = genesis(&chain);
    assert_matches_store(&chain, &c0, 0, 7, &exotic_values());
}

#[test]
fn values_minted_since_then_reused_and_removed_match_the_store() {
    // A delta since the checkpoint mints the exotic values; the write then removes some and
    // re-uses others with new subjects, so ids come from the deltas' clear new terms.
    let mut chain = rich_chain();
    let c0 = genesis(&chain);
    let at = chain.layers.len() - 1;
    chain.push(exotic_values());
    let base = chain.layers.len() - 1;
    let s = iri("ex");
    let ops = vec![
        Op::Remove(ValueTriple::new_value(&s, &iri("f64"), f64::make_entry(&(-0.0f64)))),
        Op::Remove(ValueTriple::new_value(&s, &iri("f64"), f64::make_entry(&f64::NAN))),
        Op::Add(ValueTriple::new_value(&iri("other"), &iri("f64"), f64::make_entry(&(0.0f64)))),
        Op::Add(ValueTriple::new_value(&iri("other"), &iri("f32"), f32::make_entry(&(0.5f32)))),
    ];
    assert_eq!(at, 7);
    assert_matches_store(&chain, &c0, 0, base, &ops);
}

#[test]
fn every_exotic_value_builds_a_proof_sidecar() {
    let mut bad = Vec::new();
    for (i, op) in exotic_values().into_iter().enumerate() {
        let Op::Add(t) = op else { unreachable!() };
        let chain = Chain::new(vec![node("a", "b", "c")]);
        let builder = chain.layers[0].open_write().unwrap();
        builder.add_value_triple(t.clone()).unwrap();
        let layer = match builder.commit() {
            Ok(layer) => layer,
            Err(e) => {
                bad.push(format!("#{i} {:?}: {e}", t.object));
                continue;
            }
        };
        if let Err(e) = layer.proof_commitment() {
            bad.push(format!("#{i} {:?}: {e}", t.object));
        }
    }
    assert!(bad.is_empty(), "{:#?}", bad);
}
