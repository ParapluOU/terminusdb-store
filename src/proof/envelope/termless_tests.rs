//! Tests for the single-layer TERM-LESS recursive envelope ([`TermlessRelationEnvelope`]): honest
//! round trips over the admitted operator shapes, the O(result) witness size, and the adversarial
//! rejections (forged / foreign / dropped / extra / duplicated openings, swapped rows, stripped
//! witness, substituted term-bearing commitment, double carrier, out-of-scope plans).

use super::plan::{RelationEqualityPlan, RelationProofPlan, RelationProofPlanKind};
use super::recursive::RecursiveRelationEnvelope;
use super::termless::{
    require_termless_plan, TermlessRelationEnvelope, TERMLESS_TREE_MAGIC, TERMLESS_TREE_VERSION,
};
use super::wire::{plan_digest, Wire, CHECKSUM_BYTES};

use sha2::{Digest, Sha256};

use crate::proof::argument::commit_column;
use crate::proof::commitment::{DictOpening, LayerCommitment, ResolutionOpenings, VerifierLayer};
use crate::proof::query::{
    AtomTerm, CanonicalResultValue, ColumnSource, LayerBgpAtom, LayerBgpPlan, ObjectKindFilter,
};
use crate::proof::CanonicalObject;
use crate::store::sync::open_sync_memory_store;
use crate::ValueTriple;

fn iri(value: &str) -> String {
    format!("http://ex/{value}")
}

/// Run a proof test on a big stack (the proof code nests large values).
fn big_stack(name: &str, f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name(name.into())
        .stack_size(256 * 1024 * 1024)
        .spawn(f)
        .unwrap()
        .join()
        .unwrap();
}

/// A base layer: `a knows b`, `b knows c`, plus `filler` unrelated `other` edges that grow the
/// dictionaries without touching any `knows` result.
fn layer_with(tag: &str, filler: usize) -> LayerCommitment {
    let store = open_sync_memory_store();
    let builder = store.create_base_layer().unwrap();
    let t = |n: &str| iri(&format!("{tag}{n}"));
    for (subject, object) in [("a", "b"), ("b", "c")] {
        builder
            .add_value_triple(ValueTriple::new_node(&t(subject), &iri("knows"), &t(object)))
            .unwrap();
    }
    for i in 0..filler {
        builder
            .add_value_triple(ValueTriple::new_node(
                &t(&format!("s{i}")),
                &iri("other"),
                &t(&format!("o{i}")),
            ))
            .unwrap();
    }
    (*builder.commit().unwrap().proof_commitment().unwrap()).clone()
}

fn knows_bgp() -> LayerBgpPlan {
    LayerBgpPlan {
        source: ColumnSource::Materialized,
        atoms: vec![LayerBgpAtom::all_variables(iri("knows").into_bytes())],
        joins: Vec::new(),
    }
}

fn leaf(bgp: LayerBgpPlan) -> RelationProofPlan {
    RelationProofPlan { query_digest: plan_digest(&bgp).unwrap(), kind: RelationProofPlanKind::Bgp(bgp) }
}

fn single_plan() -> RelationProofPlan {
    leaf(knows_bgp())
}

/// `?x knows ?y, ?y knows ?z` as an InnerJoin of two leaves on `left.object == right.subject`.
fn join_plan() -> RelationProofPlan {
    RelationProofPlan {
        query_digest: [7u8; 32],
        kind: RelationProofPlanKind::InnerJoin {
            left: Box::new(single_plan()),
            right: Box::new(single_plan()),
            key_pairs: vec![(1, 0)],
        },
    }
}

/// `Distinct(?x)` over `?x knows ?y`.
fn distinct_plan() -> RelationProofPlan {
    RelationProofPlan {
        query_digest: [8u8; 32],
        kind: RelationProofPlanKind::Distinct {
            input: Box::new(RelationProofPlan {
                query_digest: [9u8; 32],
                kind: RelationProofPlanKind::BagProjection {
                    input: Box::new(single_plan()),
                    indices: vec![0],
                },
            }),
            key_columns: vec![0],
        },
    }
}

/// `<A_a> knows ?o` — a SUBJECT constant (exercises the matched-constant belt through the carrier).
fn constant_plan(tag: &str) -> RelationProofPlan {
    let mut atom = LayerBgpAtom::all_variables(iri("knows").into_bytes());
    atom.subject = AtomTerm::NodeConstant(iri(&format!("{tag}a")).into_bytes());
    leaf(LayerBgpPlan { source: ColumnSource::Materialized, atoms: vec![atom], joins: Vec::new() })
}

struct Fixture {
    layer: LayerCommitment,
    plan: RelationProofPlan,
    term_bearing: RecursiveRelationEnvelope,
    termless: TermlessRelationEnvelope,
    commitment: crate::proof::commitment::VerifierCommitment,
}

fn fixture(layer: LayerCommitment, plan: RelationProofPlan) -> Fixture {
    let term_bearing = RecursiveRelationEnvelope::prove(&layer, &plan).unwrap();
    term_bearing.verify(&layer, layer.state.commitment_root, &plan).unwrap();
    let termless =
        TermlessRelationEnvelope::from_envelope(term_bearing.clone(), &layer, &plan).unwrap();
    let commitment = layer.verifier_commitment_termless().unwrap();
    Fixture { layer, plan, term_bearing, termless, commitment }
}

impl Fixture {
    fn root(&self) -> crate::proof::ProofHash {
        self.layer.resolution_root()
    }
    /// Decode a byte image and verify it against this fixture's term-less commitment + root.
    fn verify_bytes(&self, bytes: &[u8]) -> std::io::Result<()> {
        let commitment = crate::proof::commitment::VerifierCommitment::decode(
            &self.commitment.encode().unwrap(),
        )?;
        TermlessRelationEnvelope::decode(bytes)?.verify(&commitment, self.root(), &self.plan)
    }
    fn with_openings(&self, nv: Vec<DictOpening>, pred: Vec<DictOpening>) -> TermlessRelationEnvelope {
        TermlessRelationEnvelope {
            envelope: self.termless.envelope.clone(),
            openings: ResolutionOpenings::from_wire(nv, pred).unwrap(),
        }
    }
    fn nv(&self) -> Vec<DictOpening> {
        self.termless.openings.node_value_openings().cloned().collect()
    }
    fn pred(&self) -> Vec<DictOpening> {
        self.termless.openings.predicate_openings().cloned().collect()
    }
}

/// Re-frame a hand-built body (envelope + raw opening lists) with a valid checksum, so the decoder's
/// own checks (not the checksum) are what a malformed witness meets.
fn frame(envelope: &RecursiveRelationEnvelope, nv: &[DictOpening], pred: &[DictOpening]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(TERMLESS_TREE_MAGIC);
    out.extend_from_slice(&TERMLESS_TREE_VERSION.to_le_bytes());
    let mut nodes = 0;
    envelope.put_bounded(&mut out, 1, &mut nodes).unwrap();
    nv.to_vec().put(&mut out).unwrap();
    pred.to_vec().put(&mut out).unwrap();
    let checksum = Sha256::digest(&out);
    out.extend_from_slice(&checksum);
    out
}

#[test]
fn termless_envelope_roundtrips_and_matches_term_bearing_results() {
    big_stack("termless-roundtrip", || {
        for plan in [single_plan(), join_plan(), distinct_plan(), constant_plan("A_")] {
            let f = fixture(layer_with("A_", 3), plan);
            let bytes = f.termless.encode().unwrap();
            f.verify_bytes(&bytes).unwrap();
            let decoded = TermlessRelationEnvelope::decode(&bytes).unwrap();
            // The delivered (verified) result terms are exactly the term-bearing envelope's.
            assert_eq!(decoded.envelope.result_values, f.term_bearing.result_values);
            assert!(!decoded.envelope.result_values.is_empty());
            // A verified envelope verifies again (the consumption record is per verification).
            decoded.verify(&f.commitment, f.root(), &f.plan).unwrap();
            decoded.verify(&f.commitment, f.root(), &f.plan).unwrap();
        }
    });
}

#[test]
fn termless_witness_is_sized_by_the_result_not_the_catalog() {
    big_stack("termless-size", || {
        let small = fixture(layer_with("A_", 2), join_plan());
        let large = fixture(layer_with("A_", 200), join_plan());
        let small_commitment = small.commitment.encode().unwrap().len();
        let large_commitment = large.commitment.encode().unwrap().len();
        assert_eq!(small_commitment, large_commitment, "term-less commitment is O(1)");
        // Same result, same number of openings; only the (log-depth) sibling paths differ.
        assert_eq!(small.nv().len(), large.nv().len());
        assert_eq!(small.pred().len(), large.pred().len());
        let small_bytes = small.termless.encode().unwrap().len();
        let large_bytes = large.termless.encode().unwrap().len();
        let term_bearing = large.layer.verifier_commitment().unwrap().encode().unwrap().len();
        let tb_small = small.term_bearing.encode().unwrap().len();
        let tb_large = large.term_bearing.encode().unwrap().len();
        eprintln!(
            "term-less envelope {} -> {} bytes; recursive envelope {} -> {}; witness openings nv {} pred {}; commitment {} (term-bearing {})",
            small_bytes, large_bytes, tb_small, tb_large, small.nv().len(), small.pred().len(), large_commitment, term_bearing
        );
        // The witness adds O(result * log N) over the (already log-N) operator proof.
        assert!(large_bytes - tb_large < small_bytes - tb_small + 4 * 1024);
        assert!(term_bearing > large_commitment * 2, "the term-bearing commitment grows");
        // Exactly the ids the verifier needs: the 3 distinct result nodes + the 1 predicate.
        assert_eq!(small.nv().len(), 3);
        assert_eq!(small.pred().len(), 1);
    });
}

#[test]
fn termless_rejects_forged_and_foreign_openings() {
    big_stack("termless-forged", || {
        let f = fixture(layer_with("A_", 3), join_plan());
        // (a) Forged term: an opening for id X carrying another id's term (same position/path).
        let mut nv = f.nv();
        let other = nv[1].term_bytes.clone();
        nv[0].term_bytes = other;
        assert!(f.with_openings(nv, f.pred()).verify(&f.commitment, f.root(), &f.plan).is_err());
        // (b) Wrong id for a correct term: relabel opening 0 with an unused id.
        let mut nv = f.nv();
        let free = (1..=f.layer.state.node_value_count)
            .find(|id| nv.iter().all(|o| o.id != *id))
            .unwrap();
        nv[0].id = free;
        assert!(f.with_openings(nv, f.pred()).verify(&f.commitment, f.root(), &f.plan).is_err());
        // (c) Openings against another root: the same query over a different catalog (same ids,
        //     different terms) — every opening is honest for ITS tree, none for this root.
        let foreign = fixture(layer_with("B_", 3), join_plan());
        assert_eq!(foreign.nv().iter().map(|o| o.id).collect::<Vec<_>>(), f.nv().iter().map(|o| o.id).collect::<Vec<_>>());
        assert!(f.with_openings(foreign.nv(), foreign.pred()).verify(&f.commitment, f.root(), &f.plan).is_err());
        // Predicate opening forged (the predicate IRI is shared, so swap in a node opening's term).
        let mut pred = f.pred();
        pred[0].term_bytes = f.nv()[0].term_bytes.clone();
        assert!(f.with_openings(f.nv(), pred).verify(&f.commitment, f.root(), &f.plan).is_err());
        // The honest witness still verifies (the mutations above were the only difference).
        f.with_openings(f.nv(), f.pred()).verify(&f.commitment, f.root(), &f.plan).unwrap();
    });
}

#[test]
fn termless_rejects_dropped_extra_and_duplicated_openings() {
    big_stack("termless-exact", || {
        let f = fixture(layer_with("A_", 3), join_plan());
        // Dropped: each single node/value opening missing, and the predicate opening missing.
        for i in 0..f.nv().len() {
            let mut nv = f.nv();
            nv.remove(i);
            assert!(f.with_openings(nv, f.pred()).verify(&f.commitment, f.root(), &f.plan).is_err());
        }
        assert!(f.with_openings(f.nv(), Vec::new()).verify(&f.commitment, f.root(), &f.plan).is_err());
        // Stripped entirely.
        assert!(f.with_openings(Vec::new(), Vec::new()).verify(&f.commitment, f.root(), &f.plan).is_err());
        // Extra: an HONEST opening for an id the statement never resolves.
        let oracle = ResolutionOpenings::oracle(f.layer.verifier_view()).unwrap();
        let unused = (1..=f.layer.state.node_value_count)
            .find(|id| f.nv().iter().all(|o| o.id != *id))
            .unwrap();
        let extra = oracle.node_value_opening(unused).unwrap().into_owned();
        let mut nv = f.nv();
        nv.push(extra);
        let padded = f.with_openings(nv.clone(), f.pred());
        assert!(padded.verify(&f.commitment, f.root(), &f.plan).is_err());
        assert!(f.verify_bytes(&padded.encode().unwrap()).is_err());
        // Duplicated: the same id twice in the wire list is malformed at decode.
        let mut dup = f.nv();
        dup.push(dup[0].clone());
        let bytes = frame(&f.termless.envelope, &dup, &f.pred());
        assert!(TermlessRelationEnvelope::decode(&bytes).is_err());
        // The framing helper itself is faithful: the honest lists re-frame to a verifying image.
        f.verify_bytes(&frame(&f.termless.envelope, &f.nv(), &f.pred())).unwrap();
        // Trailing bytes / truncation / checksum.
        let honest = f.termless.encode().unwrap();
        let mut flipped = honest.clone();
        let mid = flipped.len() / 2;
        flipped[mid] ^= 1;
        assert!(f.verify_bytes(&flipped).is_err());
        assert!(f.verify_bytes(&honest[..honest.len() - 1]).is_err());
        assert!(honest.len() > CHECKSUM_BYTES);
    });
}

#[test]
fn termless_rejects_swapped_rows_and_tampered_terms() {
    big_stack("termless-rows", || {
        let f = fixture(layer_with("A_", 3), single_plan());
        // Swap two delivered rows (ids) and re-commit, so only the proof can catch it.
        let mut swapped = f.termless.clone();
        for column in swapped.envelope.result_columns.iter_mut() {
            column.swap(0, 1);
        }
        let domain = 1usize << swapped.envelope.result_log_domain;
        swapped.envelope.result_commitments = swapped
            .envelope
            .result_columns
            .iter()
            .map(|column| {
                let mut padded = column.clone();
                padded.resize(domain, <dory_pcs::backends::arkworks::ArkFr as dory_pcs::primitives::arithmetic::Field>::zero());
                commit_column(&padded).unwrap()
            })
            .collect();
        for column in swapped.envelope.result_values.iter_mut() {
            column.swap(0, 1);
        }
        assert!(swapped.verify(&f.commitment, f.root(), &f.plan).is_err());
        // Swap only the delivered TERMS (ids intact): the opened terms disagree.
        let mut terms = f.termless.clone();
        terms.envelope.result_values[1].swap(0, 1);
        assert!(terms.verify(&f.commitment, f.root(), &f.plan).is_err());
        // Replace a delivered term with an attacker term.
        let mut evil = f.termless.clone();
        evil.envelope.result_values[0][0] =
            CanonicalResultValue::Object(CanonicalObject::Node(iri("Evil").into_bytes()));
        assert!(evil.verify(&f.commitment, f.root(), &f.plan).is_err());
    });
}

#[test]
fn termless_rejects_wrong_commitment_root_and_carrier() {
    big_stack("termless-anchor", || {
        let f = fixture(layer_with("A_", 3), single_plan());
        // A term-BEARING commitment substituted for the term-less one.
        let term_bearing = f.layer.verifier_commitment().unwrap();
        assert!(term_bearing.terms_present);
        assert!(f.termless.verify(&term_bearing, f.root(), &f.plan).is_err());
        // Another layer's term-less commitment (and its own root).
        let other = layer_with("B_", 3);
        let other_commitment = other.verifier_commitment_termless().unwrap();
        assert!(f.termless.verify(&other_commitment, other.resolution_root(), &f.plan).is_err());
        assert!(f.termless.verify(&other_commitment, f.root(), &f.plan).is_err());
        // The right commitment, the wrong trusted root.
        assert!(f.termless.verify(&f.commitment, other.resolution_root(), &f.plan).is_err());
        // A different plan (anti-substitution).
        assert!(f.termless.verify(&f.commitment, f.root(), &constant_plan("A_")).is_err());
        // Two carriers: a leaf that ALSO embeds the openings.
        let mut double = f.termless.clone();
        if let super::recursive::RecursiveRelationProof::Bgp(leaf) = &mut double.envelope.proof {
            leaf.proof.openings = Some(f.termless.openings.clone());
        } else {
            panic!("single plan proves a Bgp leaf");
        }
        assert!(double.verify(&f.commitment, f.root(), &f.plan).is_err());
        // A term-less commitment fed to the TERM-BEARING envelope path still fails closed.
        assert!(f.term_bearing.verify(&f.commitment, f.layer.state.commitment_root, &f.plan).is_err());
        // An oracle carrier is never encodable or acceptable as a witness.
        let oracle = ResolutionOpenings::oracle(f.layer.verifier_view()).unwrap();
        let mut out = Vec::new();
        assert!(oracle.put(&mut out).is_err());
        let with_oracle = TermlessRelationEnvelope { envelope: f.termless.envelope.clone(), openings: oracle };
        assert!(with_oracle.verify(&f.commitment, f.root(), &f.plan).is_err());
    });
}

#[test]
fn termless_plan_gate_rejects_value_reading_shapes() {
    let mut ranged = LayerBgpAtom::all_variables(iri("knows").into_bytes());
    ranged.object_kind = ObjectKindFilter::Node;
    assert!(require_termless_plan(&leaf(LayerBgpPlan {
        source: ColumnSource::Materialized,
        atoms: vec![ranged],
        joins: Vec::new(),
    }))
    .is_err());
    let group = RelationProofPlan {
        query_digest: [1u8; 32],
        kind: RelationProofPlanKind::GroupCount { input: Box::new(single_plan()), key_columns: vec![0] },
    };
    assert!(require_termless_plan(&group).is_err());
    let equality = RelationProofPlan {
        query_digest: [2u8; 32],
        kind: RelationProofPlanKind::Equality {
            input: Box::new(single_plan()),
            constraint: RelationEqualityPlan::ColumnConstant {
                column: 0,
                domain: super::plan::EqualityValueDomain::Node,
                value: CanonicalObject::Node(iri("A_a").into_bytes()),
                constant_on_left: false,
            },
        },
    };
    assert!(require_termless_plan(&equality).is_err());
    assert!(require_termless_plan(&join_plan()).is_ok());
    assert!(require_termless_plan(&distinct_plan()).is_ok());
}

