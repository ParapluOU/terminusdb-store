//! Queries over proven checkpoints (`doc/proof-of-woql/checkpoint-proofs.md` §6 step 2): honest
//! round trips over a checkpoint with no deltas since and over a checkpoint plus several deltas
//! (removals and new terms), equal to the term-bearing reference over a base layer of the same
//! logical state; result-sized witnesses; the adversarial cases the design names; and (ignored) the
//! measurement run.

use std::io;
use std::sync::Arc;
use std::time::Instant;

use dory_pcs::backends::arkworks::ArkFr;
use dory_pcs::primitives::arithmetic::Field;
use tdb_succinct::TdbDataType;

use super::smt::SmtTerminal;
use super::view::{prove_checkpoint_view_forged, ViewForgery};
use super::*;
use crate::layer::{Layer, ValueTriple};
use crate::proof::argument::StoreProverContext;
use crate::proof::commitment::{LayerCommitment, SortedNamespace, VerifierLayer};
use crate::proof::envelope::{
    decode_and_verify_checkpoint_delta_read, decode_and_verify_checkpoint_read, plan_digest,
    CheckpointRelationEnvelope, EqualityValueDomain, RecursiveRelationEnvelope,
    RelationEqualityPlan, RelationProofPlan, RelationProofPlanKind,
};
use crate::proof::order::OrderDirection;
use crate::proof::query::{AtomTerm, CanonicalResultValue, ColumnSource, LayerBgpAtom, LayerBgpPlan, ObjectRangeSpec};
use crate::proof::range::RangeOp;
use crate::proof::{CanonicalDatatype, CanonicalObject};
use crate::store::sync::{open_sync_memory_store, SyncStoreLayer};

fn iri(s: &str) -> String {
    format!("http://ex/{s}")
}

fn big_stack(name: &str, f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new().name(name.into()).stack_size(256 * 1024 * 1024).spawn(f).unwrap().join().unwrap();
}

fn knows(s: &str, o: &str) -> ValueTriple {
    ValueTriple::new_node(&iri(s), &iri("knows"), &iri(o))
}
fn score(s: &str, v: u32) -> ValueTriple {
    ValueTriple::new_value(&iri(s), &iri("score"), u32::make_entry(&v))
}
fn reading(s: &str, v: f64) -> ValueTriple {
    ValueTriple::new_value(&iri(s), &iri("reading"), f64::make_entry(&v))
}

enum Op {
    Add(ValueTriple),
    Remove(ValueTriple),
}

/// A real store chain with each layer's sidecar.
struct Chain {
    layers: Vec<SyncStoreLayer>,
    commitments: Vec<Arc<LayerCommitment>>,
}

impl Chain {
    fn new(base: Vec<ValueTriple>) -> Self {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for t in base {
            builder.add_value_triple(t).unwrap();
        }
        let layer = builder.commit().unwrap();
        let commitment = layer.proof_commitment().unwrap();
        Chain { layers: vec![layer], commitments: vec![commitment] }
    }
    fn push(&mut self, ops: Vec<Op>) {
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
    fn deltas(&self, from: usize, to: usize) -> Vec<&LayerCommitment> {
        self.commitments[from + 1..=to].iter().map(|c| c.as_ref()).collect()
    }
    fn root(&self, i: usize) -> ProofHash {
        self.commitments[i].resolution_root()
    }
    fn genesis(&self) -> CheckpointState {
        genesis_checkpoint(&self.commitments[0]).unwrap().0
    }
    /// A fresh base layer holding layer `i`'s logical triples: the reference's id space.
    fn reference(&self, i: usize) -> LayerCommitment {
        let layer = &self.layers[i];
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for t in layer.triples() {
            builder.add_value_triple(layer.id_triple_to_string(&t).unwrap()).unwrap();
        }
        (*builder.commit().unwrap().proof_commitment().unwrap()).clone()
    }
    fn delta_view_bytes(&self, from: usize, to: usize) -> Vec<Vec<u8>> {
        self.deltas(from, to).iter().map(|d| d.verifier_commitment().unwrap().encode().unwrap()).collect()
    }
}

/// Base + three deltas: removals of inherited triples, re-adds, new terms (subjects, objects, a new
/// predicate and new values). The head state: `a knows b knows c knows d`, `likes a c`, scores
/// `c1: 30, 10`, `c2: 5, 20`.
fn chain() -> Chain {
    let mut chain = Chain::new(vec![
        knows("a", "b"),
        knows("b", "c"),
        knows("x", "y"),
        knows("y", "z"),
        score("c1", 30),
        score("c1", 99),
        score("junk", 7),
    ]);
    chain.push(vec![Op::Remove(knows("x", "y")), Op::Add(knows("c", "d")), Op::Remove(score("c1", 99)), Op::Add(score("c1", 10))]);
    chain.push(vec![Op::Add(score("c2", 5)), Op::Remove(score("junk", 7)), Op::Add(ValueTriple::new_node(&iri("a"), &iri("likes"), &iri("c")))]);
    chain.push(vec![Op::Remove(knows("y", "z")), Op::Add(score("c2", 20))]);
    chain
}

fn atom(predicate: &str) -> LayerBgpAtom {
    LayerBgpAtom::all_variables(iri(predicate).into_bytes())
}
fn leaf(atoms: Vec<LayerBgpAtom>) -> RelationProofPlan {
    let bgp = LayerBgpPlan { source: ColumnSource::Materialized, atoms, joins: Vec::new() };
    RelationProofPlan { query_digest: plan_digest(&bgp).unwrap(), kind: RelationProofPlanKind::Bgp(bgp) }
}
fn bgp_plan() -> RelationProofPlan {
    leaf(vec![atom("knows")])
}
fn range_plan(datatype: CanonicalDatatype, predicate: &str, c: u64) -> RelationProofPlan {
    let mut a = atom(predicate);
    a.object_range = Some(ObjectRangeSpec {
        op: RangeOp::Gt,
        datatype: ArkFr::from_u64(datatype as u64),
        c: ArkFr::from_u64(c),
        b_bits: 64,
        second: None,
    });
    leaf(vec![a])
}
fn order_plan() -> RelationProofPlan {
    let mut a = atom("score");
    a.subject = AtomTerm::NodeConstant(iri("c1").into_bytes());
    RelationProofPlan {
        query_digest: [41u8; 32],
        kind: RelationProofPlanKind::ExactScalarOrder {
            input: Box::new(leaf(vec![a])),
            datatype: CanonicalDatatype::UInt32,
            direction: OrderDirection::Ascending,
        },
    }
}
fn group_sum_plan() -> RelationProofPlan {
    RelationProofPlan {
        query_digest: [42u8; 32],
        kind: RelationProofPlanKind::GroupSum {
            input: Box::new(leaf(vec![atom("score")])),
            key_columns: vec![0],
            value_column: 1,
            datatype: CanonicalDatatype::UInt32,
        },
    }
}
fn equality_plan(value: &str) -> RelationProofPlan {
    RelationProofPlan {
        query_digest: [43u8; 32],
        kind: RelationProofPlanKind::Equality {
            input: Box::new(leaf(vec![atom("knows")])),
            constraint: RelationEqualityPlan::ColumnConstant {
                column: 0,
                domain: EqualityValueDomain::Node,
                value: CanonicalObject::Node(iri(value).into_bytes()),
                constant_on_left: false,
            },
        },
    }
}
fn object_constant_plan(value: &str) -> RelationProofPlan {
    let mut a = atom("knows");
    a.object = AtomTerm::NodeConstant(iri(value).into_bytes());
    leaf(vec![a])
}
fn closure_plan(seed: &str) -> RelationProofPlan {
    RelationProofPlan {
        query_digest: [44u8; 32],
        kind: RelationProofPlanKind::TransitiveClosure {
            edge: Box::new(RelationProofPlan {
                query_digest: [44u8; 32],
                kind: RelationProofPlanKind::Bgp(LayerBgpPlan {
                    source: ColumnSource::Materialized,
                    atoms: vec![atom("knows")],
                    joins: Vec::new(),
                }),
            }),
            seed: CanonicalObject::Node(iri(seed).into_bytes()),
            reflexive: false,
            inverted: false,
        },
    }
}

fn plans() -> Vec<(&'static str, RelationProofPlan)> {
    vec![
        ("bgp", bgp_plan()),
        ("range", range_plan(CanonicalDatatype::UInt32, "score", 15)),
        ("order", order_plan()),
        ("group sum", group_sum_plan()),
        ("equality constant", equality_plan("a")),
        ("absent equality constant", equality_plan("missing")),
        ("removed-term constant", object_constant_plan("z")),
        ("absent object constant", object_constant_plan("missing")),
        ("closure from a constant", closure_plan("a")),
    ]
}

fn sorted_rows(values: &[Vec<CanonicalResultValue>], len: usize) -> Vec<String> {
    let mut rows: Vec<String> =
        (0..len).map(|row| values.iter().map(|c| format!("{:?}", c.get(row))).collect::<Vec<_>>().join("|")).collect();
    rows.sort();
    rows
}

/// A proven checkpoint query, prover side.
struct Read {
    query: CheckpointQueryLayer,
    envelope: CheckpointRelationEnvelope,
    plan: RelationProofPlan,
}

impl Read {
    fn prove(query: CheckpointQueryLayer, plan: RelationProofPlan) -> io::Result<Self> {
        let envelope = RecursiveRelationEnvelope::prove(query.layer(), &plan)?;
        let envelope = CheckpointRelationEnvelope::from_envelope(envelope, &query, &plan)?;
        Ok(Self { query, envelope, plan })
    }
    fn verify(&self, envelope: &CheckpointRelationEnvelope) -> io::Result<()> {
        envelope.verify(self.query.view(), &self.plan)
    }
    fn verify_bytes(&self) -> io::Result<CheckpointRelationEnvelope> {
        let artifact = self.query.artifact();
        decode_and_verify_checkpoint_read(
            &self.envelope.encode()?,
            &artifact.encode()?,
            &self.query.view_proof().encode()?,
            artifact.checkpoint_root,
            artifact.head_root,
            &self.plan,
        )
    }
}

fn query_layer(state: CheckpointState) -> CheckpointQueryLayer {
    CheckpointQueryLayer::new(&StoreProverContext::cpu(), Arc::new(state)).unwrap()
}

/// Case (a): the checkpoint of the head, no deltas since. Case (b): the genesis checkpoint plus
/// all three deltas, through the byte-in delta read. Both equal the term-bearing reference.
#[test]
fn checkpoint_reads_equal_the_term_bearing_reference() {
    big_stack("checkpoint-reads", || {
        let chain = chain();
        let c0 = chain.genesis();
        let (c3, c3_proof) = prove_checkpoint(&c0, &chain.deltas(0, 3)).unwrap();
        let reference = chain.reference(3);
        let views = chain.delta_view_bytes(0, 3);
        let view_refs: Vec<&[u8]> = views.iter().map(Vec::as_slice).collect();
        for (name, plan) in plans() {
            let expected = RecursiveRelationEnvelope::prove(&reference, &plan).unwrap();
            expected.verify(&reference, reference.state.commitment_root, &plan).unwrap();
            let want = sorted_rows(&expected.result_values, expected.result_len);

            // (a) the head checkpoint itself.
            let read = Read::prove(query_layer(c3.clone()), plan.clone()).unwrap_or_else(|e| panic!("{}: {}", name, e));
            let got = read.verify_bytes().unwrap_or_else(|e| panic!("{} (a): {}", name, e));
            assert_eq!(sorted_rows(&got.envelope.result_values, got.envelope.result_len), want, "{name} (a)");
            if name == "order" {
                assert_eq!(got.envelope.result_values, expected.result_values, "order is positional");
            }
            // The no-deltas read binds the trusted head: another head root rejects.
            let artifact = read.query.artifact();
            assert!(
                decode_and_verify_checkpoint_read(
                    &read.envelope.encode().unwrap(),
                    &artifact.encode().unwrap(),
                    &read.query.view_proof().encode().unwrap(),
                    artifact.checkpoint_root,
                    chain.root(2),
                    &plan,
                )
                .is_err(),
                "{name} (a): a checkpoint read verified against another head"
            );

            // (b) C_0 + the deltas since: the head checkpoint is the ephemeral C'.
            let got = decode_and_verify_checkpoint_delta_read(
                &read.envelope.encode().unwrap(),
                &c0.artifact.encode().unwrap(),
                c0.artifact.checkpoint_root,
                &view_refs,
                chain.root(3),
                &c3.artifact.encode().unwrap(),
                &c3_proof.encode().unwrap(),
                &read.query.view_proof().encode().unwrap(),
                &plan,
            )
            .unwrap_or_else(|e| panic!("{} (b): {}", name, e));
            assert_eq!(sorted_rows(&got.envelope.result_values, got.envelope.result_len), want, "{name} (b)");
        }
    });
}

/// The witness is result-sized: 300 unrelated filler triples (with their own terms) added in a
/// delta change no opening count.
#[test]
fn checkpoint_witness_does_not_grow_with_the_catalog() {
    big_stack("checkpoint-witness-size", || {
        let small = chain();
        let mut big = chain();
        big.push((0..300).map(|i| Op::Add(ValueTriple::new_node(&iri(&format!("f{i}")), &iri("other"), &iri(&format!("g{i}"))))).collect());
        let count = |e: &CheckpointRelationEnvelope| {
            e.openings.node_value_openings().count()
                + e.openings.predicate_openings().count()
                + e.openings.keyed_lookups().count()
        };
        for (name, plan) in plans() {
            let head = |chain: &Chain, to: usize| prove_checkpoint(&chain.genesis(), &chain.deltas(0, to)).unwrap().0;
            let a = Read::prove(query_layer(head(&small, 3)), plan.clone()).unwrap();
            let b = Read::prove(query_layer(head(&big, 4)), plan).unwrap();
            b.verify(&b.envelope).unwrap();
            assert_eq!(count(&a.envelope), count(&b.envelope), "{name}: witness grew with the catalog");
        }
    });
}

/// The dictionary commitments append across a segment boundary exactly as a from-scratch
/// segmented commitment (a short segment length stands in for 2^24).
#[test]
fn dictionary_append_crosses_segment_boundaries() {
    let chain = chain();
    let c0 = chain.genesis();
    let (c3, _) = prove_checkpoint(&c0, &chain.deltas(0, 3)).unwrap();
    let new_nv = &c3.nv_terms[c0.nv_terms.len()..];
    let new_pred = &c3.pred_terms[c0.pred_terms.len()..];
    let seg = 3;
    let empty = CheckpointArtifact::empty_dictionary();
    let (base_nv, base_pred) = super::state::append_dictionary_segmented(&empty, &c0.nv_terms, &c0.pred_terms, seg).unwrap();
    let mut prev = c0.artifact.clone();
    prev.nv_dict_segments = base_nv;
    prev.pred_dict_segments = base_pred;
    let (nv, pred) = super::state::append_dictionary_segmented(&prev, new_nv, new_pred, seg).unwrap();
    let (scratch_nv, scratch_pred) =
        super::state::append_dictionary_segmented(&empty, &c3.nv_terms, &c3.pred_terms, seg).unwrap();
    assert!(scratch_nv.len() > prev.nv_dict_segments.len(), "the append opens new segments");
    assert_eq!(nv, scratch_nv);
    assert_eq!(pred, scratch_pred);
    // With the production segment length everything sits in segment 0.
    assert_eq!(c3.artifact.nv_dict_segments.len(), 1);
}

/// A result id opened against a different checkpoint (another chain whose term tree has another
/// term at the same id) rejects; so does verifying against the other checkpoint's view.
#[test]
fn result_id_opened_against_another_checkpoint_rejects() {
    big_stack("checkpoint-foreign-opening", || {
        let chain = chain();
        let read = Read::prove(query_layer(prove_checkpoint(&chain.genesis(), &chain.deltas(0, 3)).unwrap().0), bgp_plan()).unwrap();
        read.verify(&read.envelope).unwrap();
        let mut other = Chain::new(vec![knows("p", "q"), knows("q", "r"), knows("r", "s"), knows("s", "t")]);
        other.push(vec![Op::Add(knows("t", "u")), Op::Add(score("u", 1))]);
        other.push(vec![Op::Add(knows("u", "v"))]);
        other.push(vec![Op::Add(knows("v", "w"))]);
        let foreign = query_layer(prove_checkpoint(&other.genesis(), &other.deltas(0, 3)).unwrap().0);
        let mut forged = read.envelope.clone();
        let ids: Vec<u64> = forged.openings.node_value_openings().map(|o| o.id).collect();
        let id = ids[0];
        let substitute = foreign.state().term_opening(false, id).unwrap();
        assert_ne!(substitute.term_bytes, forged.openings.node_value_openings().next().unwrap().term_bytes);
        forged.openings.node_value_map_for_test().insert(id, substitute);
        assert!(read.verify(&forged).is_err(), "a foreign checkpoint's opening must not resolve");
        assert!(read.envelope.verify(foreign.view(), &read.plan).is_err(), "another checkpoint's view rejects");
    });
}

/// A constant the term index holds cannot be "proven absent": a forged empty terminal on its path
/// and another checkpoint's genuine non-membership path both reject.
#[test]
fn present_constant_claimed_absent_rejects() {
    big_stack("checkpoint-forged-absence", || {
        let chain = chain();
        let read = Read::prove(query_layer(prove_checkpoint(&chain.genesis(), &chain.deltas(0, 3)).unwrap().0), equality_plan("a")).unwrap();
        read.verify(&read.envelope).unwrap();
        let key = crate::proof::commitment::node_value_sort_key(&CanonicalObject::Node(iri("a").into_bytes()));
        let mut forged = read.envelope.clone();
        let path = forged.openings.keyed_map_for_test().get_mut(&(SortedNamespace::NodeValue, key)).unwrap();
        assert!(matches!(path.terminal, SmtTerminal::Leaf { .. }));
        path.terminal = SmtTerminal::Empty;
        assert!(read.verify(&forged).is_err());
        // A genuine absence path from a checkpoint that lacks `a`.
        let other = Chain::new(vec![knows("p", "q")]);
        let absent = other.genesis().index_path(false, &key);
        let mut forged = read.envelope.clone();
        forged.openings.keyed_map_for_test().insert((SortedNamespace::NodeValue, key), absent);
        assert!(read.verify(&forged).is_err());
        // Dropping the lookup, or adding an unconsumed one, also rejects.
        let mut dropped = read.envelope.clone();
        dropped.openings.keyed_map_for_test().clear();
        assert!(read.verify(&dropped).is_err());
        let mut padded = read.envelope.clone();
        padded.openings.keyed_map_for_test().insert((SortedNamespace::Predicate, [7u8; 32]), other.genesis().index_path(true, &[7u8; 32]));
        assert!(read.verify(&padded).is_err());
    });
}

/// Forged dictionary VALUE / DATATYPE / hash / id entries in the view's standard columns are
/// caught by the relocation against the checkpoint's own commitments.
#[test]
fn forged_view_dictionary_entries_reject() {
    let chain = chain();
    let (c3, _) = prove_checkpoint(&chain.genesis(), &chain.deltas(0, 3)).unwrap();
    let honest = prove_checkpoint_view(&StoreProverContext::cpu(), &c3).unwrap().1;
    verify_checkpoint_view(&c3.artifact, c3.artifact.checkpoint_root, &honest).unwrap();
    for forgery in [ViewForgery::NvValue, ViewForgery::NvDatatype, ViewForgery::NvHash, ViewForgery::NvId, ViewForgery::PredHash, ViewForgery::NvPadding] {
        let forged = prove_checkpoint_view_forged(&StoreProverContext::cpu(), &c3, forgery).unwrap();
        assert!(
            verify_checkpoint_view(&c3.artifact, c3.artifact.checkpoint_root, &forged).is_err(),
            "{forgery:?} verified"
        );
    }
    // A swapped standard commitment (honest proof otherwise) also rejects.
    let mut swapped = honest.clone();
    swapped.nv_commitments.swap(2, 3);
    assert!(verify_checkpoint_view(&c3.artifact, c3.artifact.checkpoint_root, &swapped).is_err());
}

/// A NaN-bearing checkpoint: the float range neither proves honestly nor verifies through any
/// bypass — flipped facts break the trusted root, and a proof over a NaN-free twin does not verify
/// against the NaN checkpoint's view.
#[test]
fn nan_checkpoint_guard_cannot_be_bypassed() {
    big_stack("checkpoint-nan", || {
        let plan = range_plan(CanonicalDatatype::Float64, "reading", 0);
        let mut nan = Chain::new(vec![reading("s1", 1.5), reading("s2", 2.5)]);
        nan.push(vec![Op::Add(reading("s3", f64::NAN))]);
        let mut twin = Chain::new(vec![reading("s1", 1.5), reading("s2", 2.5)]);
        twin.push(vec![Op::Add(reading("s3", 3.5))]);
        let nan_c = query_layer(prove_checkpoint(&nan.genesis(), &nan.deltas(0, 1)).unwrap().0);
        assert!(nan_c.artifact().facts.float64_nan);
        assert!(Read::prove(nan_c, plan.clone()).is_err(), "an honest prover cannot prove a range over a NaN dictionary");
        let nan_c = query_layer(prove_checkpoint(&nan.genesis(), &nan.deltas(0, 1)).unwrap().0);
        // Flip the fact: the artifact no longer carries the recorded root.
        let mut flipped = nan_c.artifact().clone();
        flipped.facts.float64_nan = false;
        assert!(verify_checkpoint_view(&flipped, nan_c.artifact().checkpoint_root, nan_c.view_proof()).is_err());
        flipped.checkpoint_root = flipped.compute_root().unwrap();
        assert!(verify_checkpoint_view(&flipped, nan_c.artifact().checkpoint_root, nan_c.view_proof()).is_err());
        // A proof over the NaN-free twin does not verify against the NaN checkpoint's view.
        let twin_read = Read::prove(query_layer(prove_checkpoint(&twin.genesis(), &twin.deltas(0, 1)).unwrap().0), plan.clone()).unwrap();
        twin_read.verify(&twin_read.envelope).unwrap();
        assert!(twin_read.envelope.verify(nan_c.view(), &plan).is_err());
    });
}

/// A view proof of one checkpoint never verifies for another (another head, or another chain).
#[test]
fn checkpoint_view_from_another_head_rejects() {
    let chain = chain();
    let c0 = chain.genesis();
    let (c2, _) = prove_checkpoint(&c0, &chain.deltas(0, 2)).unwrap();
    let (c3, _) = prove_checkpoint(&c0, &chain.deltas(0, 3)).unwrap();
    let context = StoreProverContext::cpu();
    let view2 = prove_checkpoint_view(&context, &c2).unwrap().1;
    let view3 = prove_checkpoint_view(&context, &c3).unwrap().1;
    verify_checkpoint_view(&c3.artifact, c3.artifact.checkpoint_root, &view3).unwrap();
    assert!(verify_checkpoint_view(&c3.artifact, c3.artifact.checkpoint_root, &view2).is_err());
    assert!(verify_checkpoint_view(&c2.artifact, c3.artifact.checkpoint_root, &view2).is_err());
    // The same state at another head (an empty delta re-checkpointed) is another checkpoint.
    let mut longer = chain;
    longer.push(vec![Op::Add(knows("q1", "q2"))]);
    longer.push(vec![Op::Remove(knows("q1", "q2"))]);
    let (c5, _) = prove_checkpoint(&c3, &longer.deltas(3, 5)).unwrap();
    assert!(verify_checkpoint_view(&c5.artifact, c5.artifact.checkpoint_root, &view3).is_err());
}

/// The delta read rejects deltas out of order, a missing delta, and a base checkpoint that was not
/// proven for the deltas' base head.
#[test]
fn delta_read_rejects_bad_deltas_and_foreign_base_checkpoints() {
    big_stack("checkpoint-delta-read-adversarial", || {
        let chain = chain();
        let c0 = chain.genesis();
        let (c3, c3_proof) = prove_checkpoint(&c0, &chain.deltas(0, 3)).unwrap();
        let read = Read::prove(query_layer(c3.clone()), bgp_plan()).unwrap();
        let views = chain.delta_view_bytes(0, 3);
        let attempt = |base: &CheckpointArtifact, base_root: ProofHash, views: &[&[u8]], proof: &CheckpointProof| {
            decode_and_verify_checkpoint_delta_read(
                &read.envelope.encode().unwrap(),
                &base.encode().unwrap(),
                base_root,
                views,
                chain.root(3),
                &c3.artifact.encode().unwrap(),
                &proof.encode().unwrap(),
                &read.query.view_proof().encode().unwrap(),
                &read.plan,
            )
        };
        let honest: Vec<&[u8]> = views.iter().map(Vec::as_slice).collect();
        attempt(&c0.artifact, c0.artifact.checkpoint_root, &honest, &c3_proof).unwrap();
        let swapped: Vec<&[u8]> = vec![&views[1], &views[0], &views[2]];
        assert!(attempt(&c0.artifact, c0.artifact.checkpoint_root, &swapped, &c3_proof).is_err());
        let missing: Vec<&[u8]> = vec![&views[0], &views[2]];
        assert!(attempt(&c0.artifact, c0.artifact.checkpoint_root, &missing, &c3_proof).is_err());
        let truncated: Vec<&[u8]> = vec![&views[0], &views[1]];
        assert!(attempt(&c0.artifact, c0.artifact.checkpoint_root, &truncated, &c3_proof).is_err());
        // A checkpoint proven for another head (C_1) used as the base of all three deltas.
        let (c1, _) = prove_checkpoint(&c0, &chain.deltas(0, 1)).unwrap();
        assert!(attempt(&c1.artifact, c1.artifact.checkpoint_root, &honest, &c3_proof).is_err());
        // Another chain's genesis with the same root claimed.
        let other = Chain::new(vec![knows("a", "b"), knows("b", "c")]).genesis();
        assert!(attempt(&other.artifact, other.artifact.checkpoint_root, &honest, &c3_proof).is_err());
        // The base artifact must carry the trusted root.
        assert!(attempt(&c0.artifact, c1.artifact.checkpoint_root, &honest, &c3_proof).is_err());
    });
}

/// Measurement (ignored): proof size and prove/verify time of a plain BGP, a range and a group sum
/// over a checkpoint with no deltas since and over a checkpoint plus a delta, at `n` filler
/// triples. Run: `cargo test --release --lib checkpoint_query_measurements -- --ignored --nocapture`.
#[test]
#[ignore]
fn checkpoint_query_measurements() {
    big_stack("checkpoint-query-measure", || {
        for n in [1_000usize, 10_000, 100_000] {
            let mut base: Vec<ValueTriple> = vec![knows("a", "b"), knows("b", "c"), score("c1", 30), score("c1", 10), score("c2", 5), score("c2", 20)];
            base.extend((0..n).map(|i| ValueTriple::new_node(&iri(&format!("s{i}")), &iri("other"), &iri(&format!("o{i}")))));
            let mut chain = Chain::new(base);
            // One delta of 100 triples: 60 additions with new terms, 40 removals of filler.
            let mut ops: Vec<Op> = (0..60).map(|i| Op::Add(ValueTriple::new_node(&iri(&format!("n{i}")), &iri("other"), &iri(&format!("m{i}"))))).collect();
            ops.extend((0..40).map(|i| Op::Remove(ValueTriple::new_node(&iri(&format!("s{i}")), &iri("other"), &iri(&format!("o{i}"))))));
            chain.push(ops);
            let context = StoreProverContext::cpu();
            let c0 = chain.genesis();
            let t = Instant::now();
            let (c1, c1_proof) = prove_checkpoint(&c0, &chain.deltas(0, 1)).unwrap();
            let head_prove = t.elapsed();
            let c1_proof_bytes = c1_proof.encode().unwrap();
            for (label, state) in [("a: checkpoint, no deltas since", &c0), ("b: checkpoint + 1 delta", &c1)] {
                let t = Instant::now();
                let query = CheckpointQueryLayer::new(&context, Arc::new(state.clone())).unwrap();
                let view_prove = t.elapsed();
                let view_bytes = query.view_proof().encode().unwrap();
                let t = Instant::now();
                verify_checkpoint_view(query.artifact(), query.artifact().checkpoint_root, query.view_proof()).unwrap();
                let view_verify = t.elapsed();
                eprintln!(
                    "n={n} {label}: view proof {} B, prove {:.2?}, verify {:.2?}{}",
                    view_bytes.len(), view_prove, view_verify,
                    if label.starts_with('b') { format!("; head checkpoint proof {} B, prove {:.2?}", c1_proof_bytes.len(), head_prove) } else { String::new() },
                );
                let query = Arc::new(query);
                for (name, plan) in [("plain BGP", bgp_plan()), ("range", range_plan(CanonicalDatatype::UInt32, "score", 15)), ("group sum", group_sum_plan())] {
                    let t = Instant::now();
                    let envelope = RecursiveRelationEnvelope::prove(query.layer(), &plan).unwrap();
                    let envelope = CheckpointRelationEnvelope::from_envelope(envelope, &query, &plan).unwrap();
                    let prove = t.elapsed();
                    let bytes = envelope.encode().unwrap();
                    let artifact = query.artifact().encode().unwrap();
                    let t = Instant::now();
                    if label.starts_with('a') {
                        decode_and_verify_checkpoint_read(&bytes, &artifact, &view_bytes, query.artifact().checkpoint_root, query.artifact().head_root, &plan).unwrap();
                    } else {
                        let views = chain.delta_view_bytes(0, 1);
                        let refs: Vec<&[u8]> = views.iter().map(Vec::as_slice).collect();
                        decode_and_verify_checkpoint_delta_read(
                            &bytes, &c0.artifact.encode().unwrap(), c0.artifact.checkpoint_root, &refs,
                            chain.root(1), &artifact, &c1_proof_bytes, &view_bytes, &plan,
                        )
                        .unwrap();
                    }
                    let verify = t.elapsed();
                    let shipped = bytes.len() + view_bytes.len() + artifact.len()
                        + if label.starts_with('b') {
                            c1_proof_bytes.len() + chain.delta_view_bytes(0, 1).iter().map(Vec::len).sum::<usize>()
                                + c0.artifact.encode().unwrap().len()
                        } else { 0 };
                    eprintln!(
                        "  {name:10} envelope {} B, shipped total {} B, prove {:.2?}, verify (all) {:.2?}",
                        bytes.len(), shipped, prove, verify,
                    );
                }
            }
        }
    });
}
