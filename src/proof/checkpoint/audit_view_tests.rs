//! Adversarial audit tests (view). See doc/proof-of-woql/checkpoint-proofs.md step 6.
//!
//! Step 2 (checkpoint view) audit: the relocation's edge domains (count 1, `m_nv < m_pred` so the
//! zero-extension factor is nontrivial), a standard commitment over a longer domain, keyed paths
//! from the wrong namespace, the flag routing of the view's bare commitment, and the per-head view
//! binding of `verify_checkpoint_read_view`.

use std::io;
use std::sync::Arc;

use dory_pcs::backends::arkworks::ArkFr;
use dory_pcs::primitives::arithmetic::Field;
use tdb_succinct::TdbDataType;

use super::view::{prove_checkpoint_view_forged, ViewForgery};
use super::*;
use crate::layer::ValueTriple;
use crate::proof::CanonicalObject;
use crate::proof::argument::{commit_column, StoreProverContext};
use crate::proof::commitment::{LayerCommitment, SortedNamespace, VerifierLayer};
use crate::proof::envelope::{
    plan_digest, verify_checkpoint_read_view, EqualityValueDomain, RelationEqualityPlan, CheckpointRelationEnvelope, RecursiveRelationEnvelope,
    RelationProofPlan, RelationProofPlanKind, TermlessRelationEnvelope,
};
use crate::proof::query::{ColumnSource, LayerBgpAtom, LayerBgpPlan};
use crate::store::sync::{open_sync_memory_store, SyncStoreLayer};

fn iri(s: &str) -> String {
    format!("http://ex/{s}")
}

fn big_stack(name: &str, f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new().name(name.into()).stack_size(256 * 1024 * 1024).spawn(f).unwrap().join().unwrap();
}

fn node(s: &str, p: &str, o: &str) -> ValueTriple {
    ValueTriple::new_node(&iri(s), &iri(p), &iri(o))
}

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
    fn push(&mut self, adds: Vec<ValueTriple>) {
        let builder = self.layers.last().unwrap().open_write().unwrap();
        for t in adds {
            builder.add_value_triple(t).unwrap();
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
    fn delta_view_bytes(&self, from: usize, to: usize) -> Vec<Vec<u8>> {
        self.deltas(from, to).iter().map(|d| d.verifier_commitment().unwrap().encode().unwrap()).collect()
    }
}

fn leaf(atoms: Vec<LayerBgpAtom>) -> RelationProofPlan {
    let bgp = LayerBgpPlan { source: ColumnSource::Materialized, atoms, joins: Vec::new() };
    RelationProofPlan { query_digest: plan_digest(&bgp).unwrap(), kind: RelationProofPlanKind::Bgp(bgp) }
}

fn bgp(predicate: &str) -> RelationProofPlan {
    leaf(vec![LayerBgpAtom::all_variables(iri(predicate).into_bytes())])
}

fn query_layer(state: CheckpointState) -> CheckpointQueryLayer {
    CheckpointQueryLayer::new(&StoreProverContext::cpu(), Arc::new(state)).unwrap()
}

fn prove_read(query: &CheckpointQueryLayer, plan: &RelationProofPlan) -> io::Result<CheckpointRelationEnvelope> {
    let envelope = RecursiveRelationEnvelope::prove(query.layer(), plan)?;
    CheckpointRelationEnvelope::from_envelope(envelope, query, plan)
}

fn assert_forgeries_reject(state: &CheckpointState, forgeries: &[ViewForgery]) {
    let context = StoreProverContext::cpu();
    let honest = prove_checkpoint_view(&context, state).unwrap().1;
    verify_checkpoint_view(&state.artifact, state.artifact.checkpoint_root, &honest).unwrap();
    for forgery in forgeries {
        let forged = prove_checkpoint_view_forged(&context, state, *forgery).unwrap();
        assert!(
            verify_checkpoint_view(&state.artifact, state.artifact.checkpoint_root, &forged).is_err(),
            "{forgery:?} verified"
        );
    }
}

/// Q1 edge: a node/value dictionary of ONE entry (`a knows a`, `a likes a`: `m_nv = 1`, row 1 is
/// padding). The padding row forgery and every entry forgery must reject, and an honest read over
/// the one-entry view must verify.
#[test]
fn one_entry_dictionaries_relocate_and_padding_forgery_rejects() {
    big_stack("audit-view-one-entry", || {
        let chain = Chain::new(vec![node("a", "knows", "a"), node("a", "likes", "a")]);
        let c0 = chain.genesis();
        assert_eq!(c0.artifact.node_value_count, 1);
        assert_eq!(c0.artifact.predicate_count, 2);
        assert_forgeries_reject(&c0, &[ViewForgery::NvPadding, ViewForgery::NvValue, ViewForgery::NvHash, ViewForgery::NvId, ViewForgery::PredHash]);
        let query = query_layer(c0);
        let plan = bgp("knows");
        let envelope = prove_read(&query, &plan).unwrap();
        envelope.verify(query.view(), &plan).unwrap();
        assert_eq!(envelope.envelope.result_len, 1);
    });
}

/// Q1 edge: `m_nv < m_pred` (3 node/values -> m = 2, 5 predicates -> m = 3), so the node/value
/// claims carry a nontrivial zero-extension factor `(1 - r_2)` into the fixed opening. Honest
/// verifies; forging the node/value padding row (row 3, below 2^m_nv but above the count) or any
/// entry rejects.
#[test]
fn extension_factor_path_binds_node_value_columns() {
    let mut triples: Vec<ValueTriple> = (1..=4).map(|i| node("s", &format!("p{i}"), "o")).collect();
    triples.push(ValueTriple::new_value(&iri("s"), &iri("p5"), u32::make_entry(&7u32)));
    let chain = Chain::new(triples);
    let c0 = chain.genesis();
    assert_eq!(c0.artifact.node_value_count, 3);
    assert_eq!(c0.artifact.predicate_count, 5);
    assert_forgeries_reject(
        &c0,
        &[ViewForgery::NvPadding, ViewForgery::NvValue, ViewForgery::NvDatatype, ViewForgery::NvHash, ViewForgery::NvId, ViewForgery::PredHash],
    );
}

/// Q1: a standard commitment of a LONGER column (the honest value column extended to `2^(m+1)`
/// with a nonzero row at `2^m`) cannot be opened at the `m`-variable point — Dory binds the layout,
/// so the honest opening no longer verifies against it.
#[test]
fn standard_commitment_over_a_longer_domain_rejects() {
    let chain = Chain::new(vec![node("a", "knows", "b"), node("b", "knows", "c")]);
    let c0 = chain.genesis();
    let n = c0.nv_terms.len();
    let m = n.next_power_of_two().max(2);
    let [_, value, _] = super::state::nv_dict_segment(&c0.nv_terms).unwrap();
    let mut longer = value.clone();
    longer.resize(2 * m, ArkFr::zero());
    longer[m] = ArkFr::one();
    let honest = prove_checkpoint_view(&StoreProverContext::cpu(), &c0).unwrap().1;
    let mut forged = honest.clone();
    forged.nv_commitments[2] = commit_column(&longer).unwrap();
    assert!(verify_checkpoint_view(&c0.artifact, c0.artifact.checkpoint_root, &forged).is_err());
    // The zero-extended column is the same polynomial; whether or not its commitment coincides
    // with the honest one, it must never yield a different accepted view commitment.
    let mut zero_ext = value;
    zero_ext.resize(2 * m, ArkFr::zero());
    let mut padded = honest.clone();
    padded.nv_commitments[2] = commit_column(&zero_ext).unwrap();
    if padded.nv_commitments[2] != honest.nv_commitments[2] {
        assert!(verify_checkpoint_view(&c0.artifact, c0.artifact.checkpoint_root, &padded).is_err());
    }
}

/// Q3: a constant's keyed lookup answered with the OTHER index's genuine path for the same key
/// rejects: separate roots and SMT namespaces. The IRI `knows` is both a predicate and a node here
/// (looked up as an object constant), so both indexes hold terms with that spelling.
#[test]
fn keyed_path_from_the_other_index_rejects() {
    big_stack("audit-view-cross-index", || {
        let chain = Chain::new(vec![node("a", "knows", "b"), node("b", "likes", "knows")]);
        let c0 = chain.genesis();
        let state = c0.clone();
        let query = query_layer(c0);
        // Equality against a constant resolves through `node_value_term_id` (the keyed index); a
        // layer-BGP constant is carried as an id instead (hash-column membership + term-tree belt).
        let plan = RelationProofPlan {
            query_digest: [77u8; 32],
            kind: RelationProofPlanKind::Equality {
                input: Box::new(bgp("likes")),
                constraint: RelationEqualityPlan::ColumnConstant {
                    column: 1,
                    domain: EqualityValueDomain::Node,
                    value: CanonicalObject::Node(iri("knows").into_bytes()),
                    constant_on_left: false,
                },
            },
        };
        let envelope = prove_read(&query, &plan).unwrap();
        envelope.verify(query.view(), &plan).unwrap();
        assert_eq!(envelope.envelope.result_len, 1);
        let keys: Vec<(SortedNamespace, ProofHash)> = envelope.openings.keyed_lookups().map(|(k, _)| *k).collect();
        assert!(keys.iter().any(|(ns, _)| *ns == SortedNamespace::NodeValue), "the object constant is looked up");
        for (namespace, key) in keys {
            let other = state.index_path(namespace != SortedNamespace::Predicate, &key);
            let mut forged = envelope.clone();
            forged.openings.keyed_map_for_test().insert((namespace, key), other);
            assert!(envelope.verify(query.view(), &plan).is_ok());
            assert!(forged.verify(query.view(), &plan).is_err(), "{namespace:?} lookup answered from the other index");
        }
    });
}

/// Q2: the view's bare commitment (`VerifiedCheckpointView::commitment`, public) carries the
/// term-tree / term-index roots in the Merkle / sorted slots but its own `verifier_view()` is
/// neither `checkpoint` nor `attested`. Verifying the checkpoint envelope's witness through the
/// ordinary term-less envelope against it must fail closed (no constant lookup, no term-tree
/// opening read as a dictionary Merkle opening).
#[test]
fn view_commitment_without_flags_fails_closed() {
    big_stack("audit-view-flags", || {
        let chain = Chain::new(vec![node("a", "knows", "b"), node("b", "knows", "c")]);
        let query = query_layer(chain.genesis());
        let plan = bgp("knows");
        let envelope = prove_read(&query, &plan).unwrap();
        envelope.verify(query.view(), &plan).unwrap();
        let commitment = query.view().commitment();
        let view = commitment.verifier_view();
        assert!(!view.attested && !view.checkpoint && !view.terms_present);
        let termless = TermlessRelationEnvelope { envelope: envelope.envelope.clone(), openings: envelope.openings.clone() };
        assert!(termless.verify(commitment, commitment.resolution_root(), &plan).is_err());
    });
}

/// Q5: the per-head view. Without deltas the checkpoint must be OF the trusted head and carry the
/// trusted root; a nonempty delta list without a head proof rejects; with deltas, the view proof
/// must be the head checkpoint's (the base checkpoint's view proof rejects), and the trusted head
/// root must be the deltas' head.
#[test]
fn checkpoint_read_view_binds_checkpoint_head_and_view() {
    let mut chain = Chain::new(vec![node("a", "knows", "b"), node("b", "knows", "c")]);
    chain.push(vec![node("c", "knows", "d")]);
    let c0 = chain.genesis();
    let (c1, c1_proof) = prove_checkpoint(&c0, &chain.deltas(0, 1)).unwrap();
    let context = StoreProverContext::cpu();
    let v0 = prove_checkpoint_view(&context, &c0).unwrap().1.encode().unwrap();
    let v1 = prove_checkpoint_view(&context, &c1).unwrap().1.encode().unwrap();
    let a0 = c0.artifact.encode().unwrap();
    let a1 = c1.artifact.encode().unwrap();
    let p1 = c1_proof.encode().unwrap();
    let views = chain.delta_view_bytes(0, 1);
    let refs: Vec<&[u8]> = views.iter().map(Vec::as_slice).collect();
    let r0 = c0.artifact.checkpoint_root;

    // Honest both ways.
    let got = verify_checkpoint_read_view(&a0, r0, &[], chain.root(0), None, &v0).unwrap();
    assert_eq!(got.checkpoint_root(), r0);
    let got = verify_checkpoint_read_view(&a0, r0, &refs, chain.root(1), Some((&a1, &p1)), &v1).unwrap();
    assert_eq!(got.checkpoint_root(), c1.artifact.checkpoint_root);

    // No deltas: the checkpoint is not of the trusted head (the chain moved on).
    assert!(verify_checkpoint_read_view(&a0, r0, &[], chain.root(1), None, &v0).is_err());
    // No head proof but deltas named.
    assert!(verify_checkpoint_read_view(&a0, r0, &refs, chain.root(1), None, &v0).is_err());
    // The base checkpoint's view proof with the head checkpoint.
    assert!(verify_checkpoint_read_view(&a0, r0, &refs, chain.root(1), Some((&a1, &p1)), &v0).is_err());
    // A head checkpoint for another head root.
    assert!(verify_checkpoint_read_view(&a0, r0, &refs, chain.root(0), Some((&a1, &p1)), &v1).is_err());
    // The head checkpoint's own artifact posing as the trusted checkpoint without deltas.
    assert!(verify_checkpoint_read_view(&a1, r0, &[], chain.root(1), None, &v1).is_err());
}
