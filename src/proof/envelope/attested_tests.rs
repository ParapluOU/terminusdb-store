//! Tests for the ATTESTED single-layer term-less envelope (`TrustedRoot::Attested`): value-reading,
//! fact-reading and term-lookup-reading statements verify term-less with result-sized witnesses;
//! the same envelopes reject under `TrustedRoot::Bare`; and the new root content (facts, term-sorted
//! index) cannot be forged, substituted or abused.

use super::plan::{EqualityValueDomain, RelationEqualityPlan, RelationProofPlan, RelationProofPlanKind};
use super::recursive::RecursiveRelationEnvelope;
use super::termless::{TermlessRelationEnvelope, TrustedRoot};
use super::wire::plan_digest;

use dory_pcs::backends::arkworks::ArkFr;
use dory_pcs::primitives::arithmetic::Field;
use tdb_succinct::TdbDataType;

use crate::proof::commitment::{
    node_value_sort_key, LayerCommitment, SortedLookup, SortedNamespace, VerifierCommitment,
    VerifierLayer,
};
use crate::proof::order::OrderDirection;
use crate::proof::query::{
    AtomTerm, ColumnSource, LayerBgpAtom, LayerBgpPlan, ObjectRangeSpec,
};
use crate::proof::range::RangeOp;
use crate::proof::{CanonicalDatatype, CanonicalObject};
use crate::store::sync::open_sync_memory_store;
use crate::ValueTriple;

fn iri(value: &str) -> String {
    format!("http://ex/{value}")
}

fn big_stack(name: &str, f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name(name.into())
        .stack_size(256 * 1024 * 1024)
        .spawn(f)
        .unwrap()
        .join()
        .unwrap();
}

/// A base layer: `a knows b knows c`, u32 `score`s on `c1`/`c2`, and `filler` unrelated edges that
/// grow the dictionaries without touching any result.
fn layer(tag: &str, filler: usize) -> LayerCommitment {
    let store = open_sync_memory_store();
    let builder = store.create_base_layer().unwrap();
    let t = |n: &str| iri(&format!("{tag}{n}"));
    for (s, o) in [("a", "b"), ("b", "c")] {
        builder.add_value_triple(ValueTriple::new_node(&t(s), &iri("knows"), &t(o))).unwrap();
    }
    for (s, v) in [("c1", 30u32), ("c1", 10), ("c2", 5), ("c2", 20)] {
        builder
            .add_value_triple(ValueTriple::new_value(&t(s), &iri("score"), u32::make_entry(&v)))
            .unwrap();
    }
    for i in 0..filler {
        builder
            .add_value_triple(ValueTriple::new_node(&t(&format!("s{i}")), &iri("other"), &t(&format!("o{i}"))))
            .unwrap();
    }
    (*builder.commit().unwrap().proof_commitment().unwrap()).clone()
}

fn leaf(atoms: Vec<LayerBgpAtom>) -> RelationProofPlan {
    let bgp = LayerBgpPlan { source: ColumnSource::Materialized, atoms, joins: Vec::new() };
    RelationProofPlan { query_digest: plan_digest(&bgp).unwrap(), kind: RelationProofPlanKind::Bgp(bgp) }
}

fn atom(predicate: &str) -> LayerBgpAtom {
    LayerBgpAtom::all_variables(iri(predicate).into_bytes())
}

/// `?c score ?v, ?v > 15`.
fn range_plan() -> RelationProofPlan {
    let mut a = atom("score");
    a.object_range = Some(ObjectRangeSpec {
        op: RangeOp::Gt,
        datatype: ArkFr::from_u64(CanonicalDatatype::UInt32 as u64),
        c: ArkFr::from_u64(15),
        b_bits: 64,
        second: None,
    });
    leaf(vec![a])
}

/// `<tag c1> score ?v` ordered ascending by `?v`.
fn order_plan(tag: &str) -> RelationProofPlan {
    let mut a = atom("score");
    a.subject = AtomTerm::NodeConstant(iri(&format!("{tag}c1")).into_bytes());
    RelationProofPlan {
        query_digest: [41u8; 32],
        kind: RelationProofPlanKind::ExactScalarOrder {
            input: Box::new(leaf(vec![a])),
            datatype: CanonicalDatatype::UInt32,
            direction: OrderDirection::Ascending,
        },
    }
}

/// GroupBy `?c` Sum `?v` over `?c score ?v`.
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

/// `?x knows ?y, ?x == <value>`.
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

/// `?s knows <value>` — a dictionary-absence proof when `value` is absent.
fn object_constant_plan(value: &str) -> RelationProofPlan {
    let mut a = atom("knows");
    a.object = AtomTerm::NodeConstant(iri(value).into_bytes());
    leaf(vec![a])
}

/// `<seed> knows+ ?y`.
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

struct Attested {
    layer: LayerCommitment,
    plan: RelationProofPlan,
    term_bearing: RecursiveRelationEnvelope,
    termless: TermlessRelationEnvelope,
    commitment: VerifierCommitment,
}

impl Attested {
    fn new(layer: LayerCommitment, plan: RelationProofPlan) -> Self {
        let term_bearing = RecursiveRelationEnvelope::prove(&layer, &plan).unwrap();
        term_bearing.verify(&layer, layer.state.commitment_root, &plan).unwrap();
        let root = layer.resolution_root();
        let termless =
            TermlessRelationEnvelope::from_envelope(term_bearing.clone(), &layer, &plan, TrustedRoot::Attested(root))
                .unwrap();
        let commitment = layer.verifier_commitment_termless().unwrap();
        Self { layer, plan, term_bearing, termless, commitment }
    }
    fn root(&self) -> crate::proof::ProofHash {
        self.layer.resolution_root()
    }
    fn verify(&self, envelope: &TermlessRelationEnvelope) -> std::io::Result<()> {
        envelope.verify(&self.commitment, TrustedRoot::Attested(self.root()), &self.plan)
    }
    /// The byte path: encode, decode both artifacts, verify attested.
    fn verify_bytes(&self) -> std::io::Result<TermlessRelationEnvelope> {
        let commitment = VerifierCommitment::decode(&self.commitment.encode()?)?;
        let decoded = TermlessRelationEnvelope::decode(&self.termless.encode()?)?;
        decoded.verify(&commitment, TrustedRoot::Attested(self.root()), &self.plan)?;
        Ok(decoded)
    }
    fn with_sorted(&self, edit: impl FnOnce(&mut std::collections::BTreeMap<(SortedNamespace, crate::proof::ProofHash), SortedLookup>)) -> TermlessRelationEnvelope {
        let mut openings = self.termless.openings.clone();
        edit(openings.sorted_map_for_test());
        TermlessRelationEnvelope { envelope: self.termless.envelope.clone(), openings }
    }
}

/// Every value-reading / lookup-reading statement verifies term-less over an attested root, from
/// bytes, with the term-bearing reference's exact result, and only result-sized openings.
#[test]
fn attested_termless_verifies_value_and_lookup_reading_shapes() {
    big_stack("attested-shapes", || {
        for (name, plan) in [
            ("range", range_plan()),
            ("order", order_plan("")),
            ("group sum", group_sum_plan()),
            ("equality constant", equality_plan("a")),
            ("absent equality constant", equality_plan("missing")),
            ("absent object constant", object_constant_plan("missing")),
            ("closure from a constant", closure_plan("a")),
        ] {
            let small = Attested::new(layer("", 0), plan.clone());
            let decoded = small.verify_bytes().unwrap_or_else(|e| panic!("{}: {}", name, e));
            assert_eq!(decoded.envelope.result_values, small.term_bearing.result_values, "{name}");
            assert_eq!(decoded.envelope.result_len, small.term_bearing.result_len, "{name}");
            // The witness does not grow with the catalog: 200 unrelated filler edges add no opening.
            let big = Attested::new(layer("", 200), plan);
            big.verify(&big.termless).unwrap();
            let count = |a: &Attested| {
                a.termless.openings.node_value_openings().count()
                    + a.termless.openings.predicate_openings().count()
                    + a.termless.openings.sorted_lookups().count()
            };
            assert_eq!(count(&small), count(&big), "{name}: witness grew with the catalog");
        }
    });
}

/// An attested-style envelope (no per-row value openings, sorted-index lookups) never verifies
/// under `Bare`: the value-reading shapes are outside the Bare scope, and the id-only ones would
/// need no sorted lookups.
#[test]
fn bare_rejects_attested_style_proofs() {
    big_stack("bare-rejects-attested", || {
        for plan in [range_plan(), order_plan(""), group_sum_plan(), equality_plan("a"), closure_plan("a"), object_constant_plan("missing")] {
            let a = Attested::new(layer("", 0), plan);
            assert!(a.termless.envelope.result_len > 0 || matches!(a.plan.kind, RelationProofPlanKind::Bgp(_)));
            assert!(a.termless.verify(&a.commitment, TrustedRoot::Bare(a.root()), &a.plan).is_err());
        }
    });
}

/// Range: the per-row value openings are not part of an attested proof, and smuggling them in is
/// rejected (one canonical witness).
#[test]
fn attested_range_carries_no_value_openings() {
    big_stack("attested-range-oid", || {
        let a = Attested::new(layer("", 0), range_plan());
        let super::recursive::RecursiveRelationProof::Bgp(leaf) = &a.termless.envelope.proof else { panic!() };
        let range = leaf.proof.atoms[0].range.as_ref().unwrap();
        assert!(range.oid.is_empty() && range.value_openings.is_none());
        let mut smuggled = a.termless.clone();
        if let super::recursive::RecursiveRelationProof::Bgp(leaf) = &mut smuggled.envelope.proof {
            leaf.proof.atoms[0].range.as_mut().unwrap().oid = vec![1];
        }
        assert!(a.verify(&smuggled).is_err());
    });
}

/// An attested range proof against a commitment whose VALUE column commitment was swapped for
/// another layer's rejects by root mismatch — whether or not the forger also recomputes the
/// commitment root.
#[test]
fn attested_range_rejects_a_swapped_value_column_commitment() {
    big_stack("attested-range-swap", || {
        let a = Attested::new(layer("", 0), range_plan());
        let other = layer("X_", 3).verifier_commitment_termless().unwrap();
        let mut swapped = a.commitment.clone();
        swapped.node_value_dict_range_commitments.0 = other.node_value_dict_range_commitments.0;
        assert!(a.termless.verify(&swapped, TrustedRoot::Attested(a.root()), &a.plan).is_err());
        // Recompute the commitment root so the artifact is internally consistent: its resolution
        // root no longer equals the attested one.
        swapped.state.commitment_root = swapped.recomputed_commitment_root().unwrap();
        swapped.validate().unwrap();
        assert_ne!(swapped.resolution_root(), a.root());
        assert!(a.termless.verify(&swapped, TrustedRoot::Attested(a.root()), &a.plan).is_err());
        // And it cannot be "re-attested" to its own root: the envelope's transcript is bound to the
        // honest commitment root.
        assert!(a.termless.verify(&swapped, TrustedRoot::Attested(swapped.resolution_root()), &a.plan).is_err());
        a.verify(&a.termless).unwrap();
    });
}

/// The facts and sorted roots are bound by `resolution_root` v2: a term-less artifact carrying
/// different facts or sorted roots does not anchor to the attested root.
#[test]
fn facts_and_sorted_roots_are_bound_by_the_resolution_root() {
    big_stack("attested-index-binding", || {
        let a = Attested::new(layer("", 0), equality_plan("a"));
        let index = a.commitment.dict_index.unwrap();
        let mut forged = [index; 3];
        forged[0].facts.float64_nan = !forged[0].facts.float64_nan;
        forged[1].nv_sorted_root[0] ^= 1;
        forged[2].pred_sorted_root[0] ^= 1;
        for index in forged {
            let mut c = a.commitment.clone();
            c.dict_index = Some(index);
            assert_ne!(c.resolution_root(), a.root());
            assert!(a.termless.verify(&c, TrustedRoot::Attested(a.root()), &a.plan).is_err());
        }
        // A base commitment without an index, or a delta one with, does not even validate.
        let mut none = a.commitment.clone();
        none.dict_index = None;
        assert!(none.validate().is_err());
        // A term-bearing artifact recomputes the index from its catalogs.
        let mut bearing = a.layer.verifier_commitment().unwrap();
        bearing.dict_index.as_mut().unwrap().nv_sorted_root[1] ^= 1;
        assert!(bearing.validate().is_err());
    });
}

/// Sorted-index lookups in a shipped witness: a missing, extra, foreign, or mis-keyed lookup, a
/// forged id, and a present constant claimed absent all reject.
#[test]
fn attested_sorted_lookups_reject_forgery() {
    big_stack("attested-sorted-forgery", || {
        let present = Attested::new(layer("", 6), equality_plan("a"));
        let absent = Attested::new(layer("", 6), equality_plan("missing"));
        let foreign = layer("X_", 6);
        let key_a = node_value_sort_key(&CanonicalObject::Node(iri("a").into_bytes()));
        let key_missing = node_value_sort_key(&CanonicalObject::Node(iri("missing").into_bytes()));
        present.verify(&present.termless).unwrap();
        absent.verify(&absent.termless).unwrap();
        assert_eq!(present.termless.openings.sorted_lookups().count(), 1);

        // Missing lookup.
        assert!(present.verify(&present.with_sorted(|m| { m.clear(); })).is_err());
        // Extra, unconsumed lookup.
        let extra = absent.termless.openings.sorted_lookups().next().unwrap().1.clone();
        assert!(present
            .verify(&present.with_sorted(|m| { m.insert((SortedNamespace::NodeValue, key_missing), extra); }))
            .is_err());
        // Wrong id in a membership leaf.
        assert!(present
            .verify(&present.with_sorted(|m| {
                if let Some(SortedLookup::Member(o)) = m.get_mut(&(SortedNamespace::NodeValue, key_a)) {
                    if let crate::proof::commitment::SortedLeaf::Real { id, .. } = &mut o.leaf { *id += 1; }
                }
            }))
            .is_err());
        // Wrong namespace key.
        assert!(present
            .verify(&present.with_sorted(|m| {
                let v = m.remove(&(SortedNamespace::NodeValue, key_a)).unwrap();
                m.insert((SortedNamespace::Predicate, key_a), v);
            }))
            .is_err());
        // A lookup from another layer's index (another root).
        let foreign_tree = crate::proof::commitment::SortedIndexTree::node_value(&foreign.node_value_catalog).unwrap();
        assert!(absent
            .verify(&absent.with_sorted(|m| { m.insert((SortedNamespace::NodeValue, key_missing), foreign_tree.lookup(&key_missing)); }))
            .is_err());
        // A present constant claimed ABSENT: substitute a bracketing-looking adjacent pair.
        let honest = crate::proof::commitment::SortedIndexTree::node_value(&present.layer.node_value_catalog).unwrap();
        let SortedLookup::Absent(left, right) = honest.lookup(&key_missing) else { panic!() };
        assert!(present
            .verify(&present.with_sorted(|m| { m.insert((SortedNamespace::NodeValue, key_a), SortedLookup::Absent(left, right)); }))
            .is_err());
        // An absent constant claimed PRESENT with some other term's leaf.
        let member_a = honest.lookup(&key_a);
        assert!(absent
            .verify(&absent.with_sorted(|m| { m.insert((SortedNamespace::NodeValue, key_missing), member_a); }))
            .is_err());
    });
}

/// The NaN guard is a root fact under attestation: a float range over a catalog holding a NaN of
/// that datatype rejects exactly as the term-bearing catalog scan does.
#[test]
fn attested_nan_guard_reads_the_root_facts() {
    big_stack("attested-nan", || {
        let build = |with_nan: bool| {
            let store = open_sync_memory_store();
            let builder = store.create_base_layer().unwrap();
            let mut values = vec![1.5f64, 2.5];
            if with_nan {
                values.push(f64::NAN);
            }
            for v in values {
                builder
                    .add_value_triple(ValueTriple::new_value(&iri("m"), &iri("f"), f64::make_entry(&v)))
                    .unwrap();
            }
            (*builder.commit().unwrap().proof_commitment().unwrap()).clone()
        };
        let clean = build(false);
        let c = clean
            .node_value_catalog
            .iter()
            .find(|r| matches!(&r.triple.object, CanonicalObject::Value { datatype: CanonicalDatatype::Float64, .. }))
            .map(|r| crate::proof::commitment::node_value_op_value(&r.triple.object).unwrap())
            .unwrap();
        let plan = || {
            let mut a = atom("f");
            a.object_range = Some(ObjectRangeSpec {
                op: RangeOp::Ge,
                datatype: ArkFr::from_u64(CanonicalDatatype::Float64 as u64),
                c,
                b_bits: 64,
                second: None,
            });
            leaf(vec![a])
        };
        // Clean catalog: proves and verifies attested.
        let a = Attested::new(clean, plan());
        a.verify_bytes().unwrap();
        // NaN catalog: the term-bearing prover's catalog scan rejects, and the attested verifier
        // rejects the clean-catalog proof's envelope against the NaN layer's facts (it never
        // reaches a gadget it could fool).
        let dirty = build(true);
        assert!(dirty.dict_index.unwrap().facts.float64_nan);
        assert!(RecursiveRelationEnvelope::prove(&dirty, &plan()).is_err());
        let dirty_commitment = dirty.verifier_commitment_termless().unwrap();
        assert!(a.termless.verify(&dirty_commitment, TrustedRoot::Attested(dirty.resolution_root()), &a.plan).is_err());
        // Directly: the facts guard of the dirty view rejects the float datatype.
        let mut view = dirty_commitment.verifier_view();
        view.attested = true;
        assert!(view.reject_nan_whole_catalog(ArkFr::from_u64(CanonicalDatatype::Float64 as u64)).is_err());
        assert!(view.reject_nan_whole_catalog(ArkFr::from_u64(CanonicalDatatype::UInt32 as u64)).is_ok());
        let _ = ArkFr::zero();
    });
}

/// The TRUST BOUNDARY of `Attested`: it relies on the root's well-formedness completely and
/// checks none of it. A malformed root — here a term-sorted index that maps key(a) to id(b) —
/// makes `?x knows ?y, ?x == a` verify with the rows of `b` under `Attested`, only because the
/// caller vouches for that root; the same artifact rejects under `Bare`. This is why the only
/// roots verified as attested are the ledger's (`LedgerAttestedRoot`), recorded only after a
/// k-of-n quorum of staked supernodes recomputed them (catalog-write / admission acks).
#[test]
fn attested_trusts_the_root_it_is_given_and_bare_does_not() {
    big_stack("attested-trust-boundary", || {
        let base = layer("", 0);
        let id_of = |name: &str| {
            base.node_value_catalog
                .iter()
                .find(|r| r.triple.object == CanonicalObject::Node(iri(name).into_bytes()))
                .unwrap()
                .object_id
        };
        let (id_a, id_b) = (id_of("a"), id_of("b"));
        // Forged sorted index: a and b swap ids.
        let mut forged_catalog = base.node_value_catalog.clone();
        for r in forged_catalog.iter_mut() {
            if r.object_id == id_a { r.object_id = id_b } else if r.object_id == id_b { r.object_id = id_a }
        }
        let forged_tree = crate::proof::commitment::SortedIndexTree::node_value(&forged_catalog).unwrap();
        let mut commitment = base.verifier_commitment_termless().unwrap();
        commitment.dict_index.as_mut().unwrap().nv_sorted_root = forged_tree.root();
        commitment.validate().unwrap();
        let forged_root = commitment.resolution_root();
        // Honest proof of `?x == b`, re-keyed as a lookup of `a`.
        let honest_b = Attested::new(base.clone(), equality_plan("b"));
        let key_a = node_value_sort_key(&CanonicalObject::Node(iri("a").into_bytes()));
        let key_b = node_value_sort_key(&CanonicalObject::Node(iri("b").into_bytes()));
        let forged = honest_b.with_sorted(|m| {
            m.remove(&(SortedNamespace::NodeValue, key_b)).unwrap();
            m.insert((SortedNamespace::NodeValue, key_a), forged_tree.lookup(&key_a));
        });
        // Round-trip through bytes like a remote verifier.
        let forged = TermlessRelationEnvelope::decode(&forged.encode().unwrap()).unwrap();
        let commitment = VerifierCommitment::decode(&commitment.encode().unwrap()).unwrap();
        let plan_a = equality_plan("a");
        // Honest answer for ?x == a is the single row (a, b); the accepted one is (b, c): a
        // malformed root the caller vouches for is believed.
        forged
            .verify(&commitment, TrustedRoot::Attested(forged_root), &plan_a)
            .expect("Attested believes the root it is given, malformed or not");
        // Under Bare the same artifact rejects.
        assert!(forged.verify(&commitment, TrustedRoot::Bare(forged_root), &plan_a).is_err());
    });
}
