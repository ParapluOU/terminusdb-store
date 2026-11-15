//! Adversarial audit tests (inductive). See doc/proof-of-woql/checkpoint-proofs.md step 6.
//!
//! The inductive verifier requires each delta's new ids to follow canonical term order
//! (`new_terms`). If the store numbers some datatype differently, an honest delta could never be
//! checkpointed (liveness), and the transition verifier, which numbers new terms the same way,
//! would disagree with the store. These tests checkpoint real store deltas over datatypes and
//! values the step-1 tests do not cover, and a triple added, removed and re-added across deltas
//! (multiset bookkeeping across the four-block link and the triple index).

use tdb_succinct::tfc::decimal::Decimal;
use tdb_succinct::TdbDataType;

use super::tests::{genesis, iri, node, rich_chain, step, Op};
use crate::layer::ValueTriple;

#[test]
fn deltas_with_exotic_values_and_a_readded_triple_checkpoint() {
    let mut chain = rich_chain();
    let c0 = genesis(&chain);
    let s = iri("ex");
    let v = |p: &str, e| Op::Add(ValueTriple::new_value(&s, &iri(p), e));
    chain.push(vec![
        v("i8", i8::make_entry(&(-100i8))),
        v("i8", i8::make_entry(&(5i8))),
        v("i32", i32::make_entry(&i32::MIN)),
        v("u64", u64::make_entry(&u64::MAX)),
        v("f32", f32::make_entry(&(-0.0f32))),
        v("f32", f32::make_entry(&(0.0f32))),
        v("f64", f64::make_entry(&f64::NAN)),
        v("f64", f64::make_entry(&(-0.0f64))),
        v("dec", Decimal::make_entry(&Decimal::new("-12.50".to_string()).unwrap())),
        v("dec", Decimal::make_entry(&Decimal::new("3.1".to_string()).unwrap())),
        v("big", rug::Integer::make_entry(&rug::Integer::from(-7))),
        v("big", rug::Integer::make_entry(&(rug::Integer::from(1) << 200))),
        Op::Add(ValueTriple::new_string_value(&s, &iri("str"), "")),
        Op::Add(node("p1", "p1", "p1")),
    ]);
    // The same triple added, removed and re-added in three consecutive deltas.
    let t = || node("dup-s", "dup-p", "dup-o");
    chain.push(vec![Op::Add(t())]);
    chain.push(vec![Op::Remove(t())]);
    chain.push(vec![Op::Add(t()), Op::Remove(ValueTriple::new_value(&s, &iri("f64"), f64::make_entry(&f64::NAN)))]);
    let head = chain.layers.len() - 1;
    // `step` proves and verifies the inductive checkpoint over all four deltas.
    let (ck, _) = step(&chain, &c0, 0, head);
    assert_eq!(ck.artifact.head_root, chain.root(head));
}
