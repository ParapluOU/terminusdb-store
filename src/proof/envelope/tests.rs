//! Unit tests for the relation-proof envelope.
//!
//! Moved verbatim from the original single-file `envelope` module. The imports below reproduce
//! what the old in-file `mod tests { use super::*; }` used to see: the `pub(super)` internals it
//! exercises directly, plus the sibling-proof-module and external types it references.
#![allow(unused_imports)]

// Envelope-internal helpers the tests drive directly (kept `pub(super)` for exactly this).
use super::columns::{
    derived_collection_sum_column, derived_group_sum_columns, integral_catalog_value, GroupSumValue,
};
use super::plan::{integral_op_datatype, ordered_slice_bounds};
use super::wire::{Reader, Wire, CHECKSUM_BYTES, MAGIC, MAX_ITEMS, RECURSIVE_MAGIC, RECURSIVE_VERSION};

use std::convert::{TryFrom, TryInto};
use std::io::{self, Cursor};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_ff::PrimeField;
use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::transcript::Transcript;
use sha2::{Digest, Sha256};

use crate::proof::anti::{
    prove_global_left_outer_join_with_context, prove_global_not_with_context,
    prove_inner_join_with_keys_and_context, prove_left_anti_join_with_keys_and_context, prove_left_outer_join_with_keys_and_context,
    verify_global_left_outer_join, verify_global_not, verify_left_anti_join_with_keys,
    verify_inner_join_with_keys, verify_left_outer_join_with_keys, InnerJoinProof, InnerJoinResult,
    LeftAntiJoinProof, LeftOuterJoinProof,
};
use crate::proof::argument::{commit_column, ColumnOpening, StoreProverContext, SumcheckProof, Transcript254};
use crate::proof::bgp::{BgpChainProof, BgpStageProof, JoinStep, RelationCommit, RelationWitness};
use crate::proof::commitment::{LayerCommitment, VerifierLayer};
use crate::proof::count::CountProof;
use crate::proof::dict::DictProof;
use crate::proof::dicthash::{
    prove_dict_hash_lookup, verify_dict_hash_lookup, DictHashLookupProof, HashLookupInnerProof,
};
use crate::proof::semijoin::{
    prove_hash_semijoin, verify_hash_semijoin, HashSemiJoinProof, SelMaskProof,
};
use crate::proof::distinct::{
    prove_relation_distinct_with_context, verify_relation_distinct,
    RelationDistinctEntryProof, RelationDistinctProof, RelationDistinctResult,
};
use crate::proof::disequality::{
    prove_tuple_disequality_with_context, verify_tuple_disequality, DisequalityNode,
    DisequalityPair, TupleDisequalityProof, TupleDisequalityStageProof,
};
use crate::proof::equality::{
    prove_relation_equality_with_context, verify_relation_equality, RelationEqualityConstraint,
    RelationEqualityProof,
};
use crate::proof::filter::{
    AtomFilterProof, BooleanProductProof, CompactionOnlyProof, FilterProof, FullFilterProof,
    SelectionOnlyProof,
};
use crate::proof::group::{GroupExtremaEntryProof, GroupExtremaKind, GroupExtremaProof};
use crate::proof::join2::{BinaryJoinCommitments, BinaryJoinProof, BinaryPairsProof};
use crate::proof::membership::MembershipProof;
use crate::proof::multiorder::{prove_exact_multi_order_with_context, verify_exact_multi_order, ExactMultiOrderProof};
use crate::proof::lexsort::LexicographicSortedProof;
use crate::proof::order::{
    prove_exact_scalar_order_with_context, verify_exact_scalar_order, ExactScalarOrderProof,
    OrderDirection,
};
use crate::proof::permutation::RelationPermutationProof;
use crate::proof::query::{
    has_ground_result_unit, prove_layer_bgp_with_result_and_context,
    resolve_typed_result_objects, result_id_namespaces,
    resolve_result_values, verify_layer_bgp, verify_layer_bgp_result_log_domain,
    verify_result_columns, generated_decimal_i64_field, generated_decimal_u64_value,
    AtomTerm, CanonicalResultValue, ColumnSource,
    DictionaryAbsence,
    DictionaryAbsencePosition, LayerBgpAtom,
    LayerBgpAtomProof, LayerBgpPlan, LayerBgpProof, ObjectKindFilter, ObjectRangeBound,
    ObjectRangeSpec, RangeAtomSubproof, ResultColumnDescriptor, ResultIdNamespace,
    SecondRangeSubproof,
};
use crate::proof::range::{
    DictLookupProof, ObjectRangeCombineProof, ObjectRangeProof, RangeOp, RangeSelectProof,
};
use crate::proof::union::{prove_bag_union_with_context, BagUnionResult};
use crate::proof::relocate::{prove_column_relocation_with_context, verify_column_relocation, RelocationProof};
use crate::proof::sort::SortedProof;
use crate::proof::union::{verify_bag_union, BagUnionProof};
use crate::proof::{CanonicalDatatype, CanonicalObject, ProofHash};
    use super::super::query::prove_layer_bgp_with_result;
    use super::*;
    use crate::store::sync::open_sync_memory_store;
    use crate::ValueTriple;
    use tdb_succinct::TdbDataType;

    fn iri(value: &str) -> String {
        format!("http://ex/{value}")
    }

    fn fixture() -> (LayerCommitment, LayerBgpPlan) {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for (subject, object) in [("a", "b"), ("b", "c")] {
            builder
                .add_value_triple(ValueTriple::new_node(
                    &iri(subject),
                    &iri("knows"),
                    &iri(object),
                ))
                .unwrap();
        }
        let layer = builder.commit().unwrap();
        let commitment = layer.proof_commitment().unwrap();
        let plan = LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![LayerBgpAtom::all_variables(iri("knows").into_bytes())],
            joins: Vec::new(),
        };
        ((*commitment).clone(), plan)
    }

    fn order_fixture() -> (LayerCommitment, LayerBgpPlan) {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for value in [30u32, 10, 20] {
            builder
                .add_value_triple(ValueTriple::new_value(
                    &iri("metric"),
                    &iri("score"),
                    u32::make_entry(&value),
                ))
                .unwrap();
        }
        let layer = builder.commit().unwrap().proof_commitment().unwrap();
        let plan = LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![LayerBgpAtom {
                subject: AtomTerm::NodeConstant(iri("metric").into_bytes()),
                predicate_term: iri("score").into_bytes(),
                predicate_variable: false,
                object: AtomTerm::Variable,
                object_range: None,
                object_kind: ObjectKindFilter::Any,
                subject_object_equal: false,
                source: ColumnSource::Materialized,
            }],
            joins: Vec::new(),
        };
        ((*layer).clone(), plan)
    }

    fn multi_order_fixture() -> (LayerCommitment, LayerBgpPlan) {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for (subject, predicate, values) in [
            ("left", "x", vec![1u32, 2]),
            ("right", "y", vec![3u32, 4]),
        ] {
            for value in values {
                builder.add_value_triple(ValueTriple::new_value(
                    &iri(subject), &iri(predicate), u32::make_entry(&value),
                )).unwrap();
            }
        }
        let layer = builder.commit().unwrap().proof_commitment().unwrap();
        let atom = |subject: &str, predicate: &str| LayerBgpAtom {
            subject: AtomTerm::NodeConstant(iri(subject).into_bytes()),
            predicate_term: iri(predicate).into_bytes(),
            predicate_variable: false,
            object: AtomTerm::Variable,
            object_range: None,
            object_kind: ObjectKindFilter::Any,
            subject_object_equal: false,
            source: ColumnSource::Materialized,
        };
        ((*layer).clone(), LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![atom("left", "x"), atom("right", "y")],
            joins: vec![JoinStep::cross_product()],
        })
    }

    fn refresh_checksum(bytes: &mut Vec<u8>) {
        let body_len = bytes.len() - CHECKSUM_BYTES;
        let checksum = Sha256::digest(&bytes[..body_len]);
        bytes[body_len..].copy_from_slice(&checksum);
    }

    // ---------------------------------------------------------------------
    // CrossRootHashJoin (two-root cross-dictionary hash join) tests.
    // ---------------------------------------------------------------------

    fn node_layer(triples: &[(&str, &str, &str)]) -> LayerCommitment {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for (s, p, o) in triples {
            builder
                .add_value_triple(ValueTriple::new_node(&iri(s), &iri(p), &iri(o)))
                .unwrap();
        }
        (*builder.commit().unwrap().proof_commitment().unwrap()).clone()
    }

    fn subject_bound_atom(subject: &str, predicate: &str) -> LayerBgpAtom {
        LayerBgpAtom {
            subject: AtomTerm::NodeConstant(iri(subject).into_bytes()),
            predicate_term: iri(predicate).into_bytes(),
            predicate_variable: false,
            object: AtomTerm::Variable,
            object_range: None,
            object_kind: ObjectKindFilter::Any,
            subject_object_equal: false,
            source: ColumnSource::Materialized,
        }
    }

    fn var_subject_atom(predicate: &str) -> LayerBgpAtom {
        LayerBgpAtom {
            subject: AtomTerm::Variable,
            predicate_term: iri(predicate).into_bytes(),
            predicate_variable: false,
            object: AtomTerm::Variable,
            object_range: None,
            object_kind: ObjectKindFilter::Any,
            subject_object_equal: false,
            source: ColumnSource::Materialized,
        }
    }

    fn bgp_leaf(atom: LayerBgpAtom, digest: u8) -> RelationProofPlan {
        RelationProofPlan {
            query_digest: [digest; 32],
            kind: RelationProofPlanKind::Bgp(LayerBgpPlan {
                source: ColumnSource::Materialized,
                atoms: vec![atom],
                joins: Vec::new(),
            }),
        }
    }

    /// `left` = the SCHEMA class set (single object column of `Triple(<subj>, <p>, ?c)`);
    /// `right` = the INSTANCE `Triple(?x, <rt>, ?class)` ([x, class], key col 1).
    fn cross_root_plan(left: RelationProofPlan, right: RelationProofPlan) -> RelationProofPlan {
        RelationProofPlan {
            query_digest: [9u8; 32],
            kind: RelationProofPlanKind::CrossRootHashJoin {
                left: Box::new(left),
                right: Box::new(right),
                left_key_col: 0,
                right_key_col: 1,
            },
        }
    }

    fn class_object_id(layer: &LayerCommitment, class: &str) -> Option<u64> {
        layer
            .node_value_catalog
            .iter()
            .find(|r| matches!(&r.triple.object, crate::proof::CanonicalObject::Node(n) if n == iri(class).as_bytes()))
            .map(|r| r.object_id)
    }

    // The cross-root node reconciles ~254-bit `hash_object` keys via a FULL-FIELD LogUp semi-join
    // (`super::semijoin`) — no magnitude sort, so >=2 distinct full-field hash keys are handled.
    #[test]
    fn cross_root_matches_same_iri_across_independent_dicts() {
        // DICT-INDEPENDENCE: the SAME class IRI "Class" appears in BOTH graphs but at DIFFERENT
        // dict ids (the instance has an extra earlier node that shifts its ids). The hash join
        // still matches it (hashing the IRI, not the id); the unrelated instance class "Other" is
        // excluded. This is the completeness half of the cross-dict property.
        // The class object "Cxxxx" (sorts first, uppercase C) is dict id 1 in the schema. The
        // instance has two earlier-sorting nodes ("Aearly","Bearly") that push "Cxxxx" to id 3.
        let schema = node_layer(&[("aaa", "cls", "Cxxxx"), ("aaa", "zzz", "yyy")]);
        let instance = node_layer(&[
            ("Aearly", "rt", "Bearly"),
            ("px", "rt", "Cxxxx"),
            ("qx", "zzz", "yyy"),
        ]);
        let schema_class = class_object_id(&schema, "Cxxxx").unwrap();
        let inst_class = class_object_id(&instance, "Cxxxx").unwrap();
        assert_ne!(
            schema_class, inst_class,
            "the SAME IRI intentionally has DIFFERENT dict ids across the two graphs"
        );

        let plan = cross_root_plan(
            bgp_leaf(subject_bound_atom("aaa", "cls"), 1),
            bgp_leaf(var_subject_atom("rt"), 2),
        );
        let env = RecursiveRelationEnvelope::prove_cross_root(&instance, &schema, &plan).unwrap();
        let db_root = crate::proof::commitment::database_root(
            &instance.state.commitment_root,
            &schema.state.commitment_root,
        );
        env.verify_cross_root(&instance, &schema, db_root, &plan).unwrap();
        assert_eq!(env.result_len, 1, "only px (class Cxxxx) matches; the Aearly row is excluded");
    }

    #[test]
    fn cross_root_excludes_same_id_different_iri() {
        // THE CRUX cross-dict NEGATIVE: two DIFFERENT class IRIs collide on the SAME integer dict id
        // (identical dictionary structure, only the class IRI differs, so both classes take the same
        // sorted rank). A raw dict-id join would falsely match; the hash join MUST exclude.
        // Identical dictionary STRUCTURE (subject "aaa", filler object "yyy"); only the class IRI
        // differs and both sort FIRST, so schema "Sxxxx" and instance "Ixxxx" take the SAME dict id.
        let schema = node_layer(&[("aaa", "cls", "Sxxxx"), ("aaa", "zzz", "yyy")]);
        let instance = node_layer(&[("aaa", "rt", "Ixxxx"), ("aaa", "zzz", "yyy")]);
        let schema_class = class_object_id(&schema, "Sxxxx").unwrap();
        let inst_class = class_object_id(&instance, "Ixxxx").unwrap();
        assert_eq!(
            schema_class, inst_class,
            "the two DIFFERENT class IRIs must collide on the SAME integer dict id (the whole point)"
        );

        let plan = cross_root_plan(
            bgp_leaf(subject_bound_atom("aaa", "cls"), 1),
            bgp_leaf(var_subject_atom("rt"), 2),
        );
        let env = RecursiveRelationEnvelope::prove_cross_root(&instance, &schema, &plan).unwrap();
        let db_root = crate::proof::commitment::database_root(
            &instance.state.commitment_root,
            &schema.state.commitment_root,
        );
        env.verify_cross_root(&instance, &schema, db_root, &plan).unwrap();
        assert_eq!(
            env.result_len, 0,
            "same-integer-id-DIFFERENT-IRI must NOT match: no raw cross-dict id comparison"
        );
    }

    // The wrong-root and tamper tests validate the NOVEL two-root soundness (database_root binding,
    // per-child root authentication, reject paths) of the cross-root envelope.
    #[test]
    fn cross_root_rejects_wrong_root_and_swapped_graphs() {
        let schema = node_layer(&[("aaa", "cls", "Cxxxx"), ("aaa", "zzz", "yyy")]);
        let instance = node_layer(&[("px", "rt", "Cxxxx"), ("qx", "zzz", "yyy")]);
        let plan = cross_root_plan(
            bgp_leaf(subject_bound_atom("aaa", "cls"), 1),
            bgp_leaf(var_subject_atom("rt"), 2),
        );
        let env = RecursiveRelationEnvelope::prove_cross_root(&instance, &schema, &plan).unwrap();
        let db_root = crate::proof::commitment::database_root(
            &instance.state.commitment_root,
            &schema.state.commitment_root,
        );
        env.verify_cross_root(&instance, &schema, db_root, &plan).unwrap();

        // A tampered trusted database root rejects.
        let mut bad = db_root;
        bad[0] ^= 1;
        assert!(env.verify_cross_root(&instance, &schema, bad, &plan).is_err());

        // Swapping the two graphs (instance<->schema) under the original root rejects: the ordered
        // two-hash `database_root(instance, schema)` no longer matches, and each child is anchored to
        // the wrong root.
        assert!(env.verify_cross_root(&schema, &instance, db_root, &plan).is_err());
    }

    #[test]
    fn cross_root_rejects_tampered_join_key() {
        let schema = node_layer(&[("aaa", "cls", "Cxxxx"), ("aaa", "zzz", "yyy")]);
        let instance = node_layer(&[("px", "rt", "Cxxxx"), ("qx", "zzz", "yyy")]);
        let plan = cross_root_plan(
            bgp_leaf(subject_bound_atom("aaa", "cls"), 1),
            bgp_leaf(var_subject_atom("rt"), 2),
        );
        let mut env = RecursiveRelationEnvelope::prove_cross_root(&instance, &schema, &plan).unwrap();
        let db_root = crate::proof::commitment::database_root(
            &instance.state.commitment_root,
            &schema.state.commitment_root,
        );
        // Corrupt the join's matched key-column commitment: the inner-join projection no longer binds.
        if let RecursiveRelationProof::CrossRootHashJoin { com_closure_at_d, .. } = &mut env.proof {
            *com_closure_at_d = commit_column(&vec![ArkFr::from_u64(7); 2]).unwrap();
        }
        assert!(env.verify_cross_root(&instance, &schema, db_root, &plan).is_err());
    }

    #[test]
    fn composite_join_rejects_non_store_id_key_namespace() {
        // The fixture BGP exposes two nonnullable NodeValue (Store-ID) columns.
        let (_commitment, bgp) = fixture();
        let leaf = || RelationProofPlan {
            query_digest: [0u8; 32],
            kind: RelationProofPlanKind::Bgp(bgp.clone()),
        };
        // GroupCount yields [NodeValue key, GeneratedDecimalU64 count]; the count column is
        // NOT a bounded Store dictionary ID, so it must never reach the composite key packing.
        let group_count = || RelationProofPlan {
            query_digest: [0u8; 32],
            kind: RelationProofPlanKind::GroupCount {
                input: Box::new(leaf()),
                key_columns: vec![0],
            },
        };
        let composite = |kind| RelationProofPlan {
            query_digest: [0u8; 32],
            kind,
        };
        // Each packed join variant must reject a composite key touching the generated column.
        for kind in [
            RelationProofPlanKind::LeftAnti {
                left: Box::new(group_count()),
                right: Box::new(group_count()),
                key_pairs: vec![(0, 0), (1, 1)],
            },
            RelationProofPlanKind::InnerJoin {
                left: Box::new(group_count()),
                right: Box::new(group_count()),
                key_pairs: vec![(0, 0), (1, 1)],
            },
            RelationProofPlanKind::LeftOuter {
                left: Box::new(group_count()),
                right: Box::new(group_count()),
                key_pairs: vec![(0, 0), (1, 1)],
            },
        ] {
            assert!(
                composite(kind).shape_bounded(0, &mut 0).is_err(),
                "composite join over a generated-decimal key must reject"
            );
        }
        // Positive control: a composite join over two Store-ID (NodeValue) keys still passes.
        let ok = RelationProofPlan {
            query_digest: [0u8; 32],
            kind: RelationProofPlanKind::InnerJoin {
                left: Box::new(leaf()),
                right: Box::new(leaf()),
                key_pairs: vec![(0, 0), (1, 1)],
            },
        };
        assert!(ok.shape_bounded(0, &mut 0).is_ok());
    }

    #[test]
    fn integral_op_datatype_excludes_floats_for_extrema() {
        for datatype in [
            CanonicalDatatype::UInt8,
            CanonicalDatatype::UInt16,
            CanonicalDatatype::UInt32,
            CanonicalDatatype::UInt64,
            CanonicalDatatype::Int8,
            CanonicalDatatype::Int16,
            CanonicalDatatype::Int32,
            CanonicalDatatype::Int64,
        ] {
            assert!(integral_op_datatype(datatype));
        }
        for datatype in [
            CanonicalDatatype::Float32,
            CanonicalDatatype::Float64,
            CanonicalDatatype::Decimal,
            CanonicalDatatype::BigInt,
        ] {
            assert!(!integral_op_datatype(datatype));
        }
    }

    /// Byte-identity regression guard for the per-atom `ColumnSource` change. A multi-atom
    /// all-Materialized BGP (every atom defaults to `ColumnSource::Materialized`, plan source
    /// Materialized) must produce the EXACT SAME `LayerBgpProof` — same commitments, same
    /// transcript-derived sumcheck polynomials, same openings — as before per-atom sourcing
    /// existed. The hash below is captured from the pre-change code. If per-atom sourcing ever
    /// perturbs a Materialized proof (extra transcript bytes, re-sourced columns), this fails.
    /// (`plan_digest`/the wire image DO change — a new per-atom `source` byte — but that is the
    /// verifier's trusted plan reconstruction, not the proof's transcript/commitments/root.)
    #[test]
    fn all_materialized_two_atom_join_proof_is_byte_identical() {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for (subject, object) in [("a", "b"), ("b", "c"), ("c", "d")] {
            builder
                .add_value_triple(ValueTriple::new_node(
                    &iri(subject),
                    &iri("knows"),
                    &iri(object),
                ))
                .unwrap();
        }
        let layer = builder.commit().unwrap().proof_commitment().unwrap();
        let plan = LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![
                LayerBgpAtom::all_variables(iri("knows").into_bytes()),
                LayerBgpAtom::all_variables(iri("knows").into_bytes()),
            ],
            joins: vec![JoinStep {
                acc_key: 1,
                rel_key: 0,
                additional_keys: Vec::new(),
            }],
        };
        let result = prove_layer_bgp_with_result(&layer, &plan).unwrap();
        assert_eq!(result.result_len, 2);
        let mut bytes = Vec::new();
        result.proof.put(&mut bytes).unwrap();
        use sha2::Digest as _;
        let hash = sha2::Sha256::digest(&bytes);
        let hash_hex = hash.iter().map(|b| format!("{b:02x}")).collect::<String>();
        assert_eq!(bytes.len(), 142820, "materialized proof byte length changed");
        assert_eq!(
            hash_hex,
            "5130d5e40bd205e4ecc23e0541e3fe38175bc814c525fd7c7ced20e74e16d89d",
            "materialized multi-atom proof transcript/commitments changed (byte-identity regression)"
        );
    }

    fn recursive_leaf(
        layer: &LayerCommitment,
        plan: &LayerBgpPlan,
    ) -> (RelationProofPlan, RecursiveRelationEnvelope) {
        let result = prove_layer_bgp_with_result(layer, plan).unwrap();
        let result_columns = result.real_columns();
        let query_digest = plan_digest(plan).unwrap();
        let leaf = LayerBgpEnvelope::new_verified(
            layer,
            layer.state.commitment_root,
            plan,
            query_digest,
            result.proof,
            result.result_commitments,
            result_columns,
            result.result_log_domain,
            result.result_len,
            result.predicate_ids,
        )
        .unwrap();
        let result_descriptors = super::super::query::result_column_descriptors(plan).unwrap();
        let result_values = leaf
            .result_objects
            .iter()
            .map(|column| {
                column
                    .iter()
                    .cloned()
                    .map(CanonicalResultValue::Object)
                    .collect()
            })
            .collect();
        (
            RelationProofPlan {
                query_digest,
                kind: RelationProofPlanKind::Bgp(plan.clone()),
            },
            RecursiveRelationEnvelope {
                layer_root: layer.state.commitment_root,
                query_digest,
                result_commitments: leaf.result_commitments.clone(),
                result_columns: leaf.result_columns.clone(),
                result_log_domain: leaf.result_log_domain,
                result_len: leaf.result_len,
                ground_result: false,
                result_descriptors,
                result_values,
                proof: RecursiveRelationProof::Bgp(Box::new(leaf)),
            },
        )
    }

    #[test]
    fn dictionary_absence_position_wire_tags_are_strict() {
        for (tag, expected) in [
            (0, DictionaryAbsencePosition::Predicate),
            (1, DictionaryAbsencePosition::Subject),
            (2, DictionaryAbsencePosition::Object),
        ] {
            let mut reader = Reader {
                bytes: &[tag],
                pos: 0,
                decoded_items: 0,
            };
            assert_eq!(DictionaryAbsencePosition::get(&mut reader).unwrap(), expected);
            reader.finish().unwrap();
        }
        let mut reader = Reader {
            bytes: &[3],
            pos: 0,
            decoded_items: 0,
        };
        assert!(DictionaryAbsencePosition::get(&mut reader).is_err());
    }

    #[test]
    fn recursive_exact_scalar_order_roundtrips_and_rejects_plan_tamper() {
        let (layer, bgp) = order_fixture();
        let child = RelationProofPlan {
            query_digest: plan_digest(&bgp).unwrap(),
            kind: RelationProofPlanKind::Bgp(bgp),
        };
        let plan = RelationProofPlan {
            query_digest: [42u8; 32],
            kind: RelationProofPlanKind::ExactScalarOrder {
                input: Box::new(child),
                datatype: CanonicalDatatype::UInt32,
                direction: OrderDirection::Ascending,
            },
        };
        let envelope = RecursiveRelationEnvelope::prove(&layer, &plan).unwrap();
        envelope
            .verify(&layer, layer.state.commitment_root, &plan)
            .unwrap();
        let bytes = envelope.encode().unwrap();
        let decoded = RecursiveRelationEnvelope::decode(&bytes).unwrap();
        decoded
            .verify(&layer, layer.state.commitment_root, &plan)
            .unwrap();

        let mut wrong_plan = plan.clone();
        if let RelationProofPlanKind::ExactScalarOrder { datatype, .. } = &mut wrong_plan.kind {
            *datatype = CanonicalDatatype::Int32;
        }
        assert!(decoded
            .verify(&layer, layer.state.commitment_root, &wrong_plan)
            .is_err());
    }

    fn assert_recursive_inner_join_roundtrips_and_rejects_tamper() {
        let (layer, bgp) = fixture();
        let leaf = || RelationProofPlan {
            query_digest: plan_digest(&bgp).unwrap(),
            kind: RelationProofPlanKind::Bgp(bgp.clone()),
        };
        let plan = RelationProofPlan {
            query_digest: [73u8; 32],
            kind: RelationProofPlanKind::InnerJoin {
                left: Box::new(leaf()), right: Box::new(leaf()),
                key_pairs: vec![(0, 0), (1, 1)],
            },
        };
        let envelope = RecursiveRelationEnvelope::prove(&layer, &plan).unwrap();
        envelope.verify(&layer, layer.state.commitment_root, &plan).unwrap();
        assert_eq!(envelope.result_len, 2);
        assert_eq!(envelope.result_columns.len(), 2);
        let bytes = envelope.encode().unwrap();
        let decoded = RecursiveRelationEnvelope::decode(&bytes).unwrap();
        decoded.verify(&layer, layer.state.commitment_root, &plan).unwrap();

        let mut wrong_keys = plan.clone();
        if let RelationProofPlanKind::InnerJoin { key_pairs, .. } = &mut wrong_keys.kind {
            key_pairs[1] = (0, 1);
        }
        assert!(decoded.verify(&layer, layer.state.commitment_root, &wrong_keys).is_err());
        let mut changed = RecursiveRelationEnvelope::decode(&bytes).unwrap();
        changed.result_columns[0].swap(0, 1);
        for (commitment, column) in changed.result_commitments.iter_mut().zip(&changed.result_columns) {
            *commitment = commit_column(column).unwrap();
        }
        assert!(changed.verify(&layer, layer.state.commitment_root, &plan).is_err());
    }

    #[test]
    fn recursive_inner_join_roundtrips_and_rejects_tamper() {
        std::thread::Builder::new().name("inner-join-proof".into()).stack_size(256 * 1024 * 1024)
            .spawn(assert_recursive_inner_join_roundtrips_and_rejects_tamper).unwrap().join().unwrap();
    }

    #[test]
    fn recursive_bag_projection_preserves_duplicate_rows_and_rejects_tamper() {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for object in ["x", "y"] {
            builder
                .add_value_triple(ValueTriple::new_node(
                    &iri("a"),
                    &iri("p"),
                    &iri(object),
                ))
                .unwrap();
        }
        let layer = builder.commit().unwrap().proof_commitment().unwrap();
        let bgp = LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![LayerBgpAtom::all_variables(iri("p").into_bytes())],
            joins: Vec::new(),
        };
        let leaf = RelationProofPlan {
            query_digest: plan_digest(&bgp).unwrap(),
            kind: RelationProofPlanKind::Bgp(bgp),
        };
        let plan = RelationProofPlan {
            query_digest: [91u8; 32],
            kind: RelationProofPlanKind::BagProjection {
                input: Box::new(leaf),
                indices: vec![0],
            },
        };
        let envelope = RecursiveRelationEnvelope::prove(&layer, &plan).unwrap();
        envelope
            .verify(&layer, layer.state.commitment_root, &plan)
            .unwrap();
        assert_eq!(envelope.result_len, 2);
        assert_eq!(envelope.result_columns.len(), 1);
        assert_eq!(envelope.result_columns[0][0], envelope.result_columns[0][1]);

        let encoded = envelope.encode().unwrap();
        let decoded = RecursiveRelationEnvelope::decode(&encoded).unwrap();
        decoded
            .verify(&layer, layer.state.commitment_root, &plan)
            .unwrap();

        let mut wrong_index = plan.clone();
        if let RelationProofPlanKind::BagProjection { indices, .. } = &mut wrong_index.kind {
            indices[0] = 1;
        }
        assert!(decoded
            .verify(&layer, layer.state.commitment_root, &wrong_index)
            .is_err());

        let mut changed = decoded.clone();
        changed.result_columns[0][0] = match &changed.proof {
            RecursiveRelationProof::BagProjection { input } => input.result_columns[1][0],
            _ => unreachable!(),
        };
        changed.result_commitments[0] = commit_column(&changed.result_columns[0]).unwrap();
        assert!(changed
            .verify(&layer, layer.state.commitment_root, &plan)
            .is_err());

        let duplicate_indices = RelationProofPlan {
            query_digest: [92u8; 32],
            kind: RelationProofPlanKind::BagProjection {
                input: match &plan.kind {
                    RelationProofPlanKind::BagProjection { input, .. } => input.clone(),
                    _ => unreachable!(),
                },
                indices: vec![0, 0],
            },
        };
        assert!(duplicate_indices.result_shape().is_err());
    }

    #[test]
    fn recursive_empty_bag_projection_emits_exact_ground_unit_and_handles_empty_child() {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for object in ["x", "y"] {
            builder
                .add_value_triple(ValueTriple::new_node(
                    &iri("a"),
                    &iri("p"),
                    &iri(object),
                ))
                .unwrap();
        }
        let layer = builder.commit().unwrap().proof_commitment().unwrap();
        let project = |bgp: LayerBgpPlan, digest| RelationProofPlan {
            query_digest: digest,
            kind: RelationProofPlanKind::BagProjection {
                input: Box::new(RelationProofPlan {
                    query_digest: plan_digest(&bgp).unwrap(),
                    kind: RelationProofPlanKind::Bgp(bgp),
                }),
                indices: Vec::new(),
            },
        };

        let present = project(
            LayerBgpPlan {
                source: ColumnSource::Materialized,
                atoms: vec![LayerBgpAtom::all_variables(iri("p").into_bytes())],
                joins: Vec::new(),
            },
            [93u8; 32],
        );
        let envelope = RecursiveRelationEnvelope::prove(&layer, &present).unwrap();
        envelope
            .verify(&layer, layer.state.commitment_root, &present)
            .unwrap();
        assert!(envelope.ground_result);
        assert_eq!(envelope.result_len, 2);
        assert_eq!(envelope.result_columns, vec![vec![ArkFr::one(); 2]]);
        assert!(envelope.result_descriptors.is_empty());
        assert!(envelope.result_values.is_empty());
        let bytes = envelope.encode().unwrap();
        let mut changed = RecursiveRelationEnvelope::decode(&bytes).unwrap();
        changed.result_columns[0][0] = ArkFr::zero();
        let mut padded = changed.result_columns[0].clone();
        padded.resize(1usize << changed.result_log_domain, ArkFr::zero());
        changed.result_commitments[0] = commit_column(&padded).unwrap();
        assert!(changed
            .verify(&layer, layer.state.commitment_root, &present)
            .is_err());

        let absent = project(
            LayerBgpPlan {
                source: ColumnSource::Materialized,
                atoms: vec![LayerBgpAtom {
                    subject: AtomTerm::NodeConstant(iri("missing").into_bytes()),
                    predicate_term: iri("p").into_bytes(),
                    predicate_variable: false,
                    object: AtomTerm::Variable,
                    object_range: None,
                    object_kind: ObjectKindFilter::Any,
                    subject_object_equal: false,
                    source: ColumnSource::Materialized,
                }],
                joins: Vec::new(),
            },
            [94u8; 32],
        );
        let empty = RecursiveRelationEnvelope::prove(&layer, &absent).unwrap();
        empty
            .verify(&layer, layer.state.commitment_root, &absent)
            .unwrap();
        assert!(empty.ground_result);
        assert_eq!(empty.result_len, 0);
        assert_eq!(empty.result_columns, vec![Vec::new()]);
    }

    #[test]
    fn recursive_layer_triple_count_tracks_logical_stack_and_rejects_tamper() {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        let empty_layer = builder.commit().unwrap();
        let plan = RelationProofPlan {
            query_digest: [95u8; 32],
            kind: RelationProofPlanKind::LayerTripleCount,
        };
        let empty_commitment = empty_layer.proof_commitment().unwrap();
        let empty = RecursiveRelationEnvelope::prove(&empty_commitment, &plan).unwrap();
        empty
            .verify(&empty_commitment, empty_commitment.state.commitment_root, &plan)
            .unwrap();
        assert_eq!(empty.result_len, 1);
        assert_eq!(empty.result_columns, vec![vec![ArkFr::zero()]]);
        assert_eq!(
            empty.result_values,
            vec![vec![CanonicalResultValue::GeneratedDecimalU64(0)]]
        );

        let builder = empty_layer.open_write().unwrap();
        for object in ["x", "y", "z"] {
            builder
                .add_value_triple(ValueTriple::new_node(
                    &iri("a"),
                    &iri("p"),
                    &iri(object),
                ))
                .unwrap();
        }
        let parent = builder.commit().unwrap();
        let builder = parent.open_write().unwrap();
        builder
            .remove_value_triple(ValueTriple::new_node(&iri("a"), &iri("p"), &iri("y")))
            .unwrap();
        builder
            .add_value_triple(ValueTriple::new_node(&iri("b"), &iri("q"), &iri("w")))
            .unwrap();
        let head = builder.commit().unwrap().proof_commitment().unwrap();
        // COMMITMENT DIET (Wave 0): a DELTA head no longer commits its materialized head
        // view or its triple count (materialized_count is now 0 and NOT root-bound), so
        // LayerTripleCount is not directly provable over a delta head and reject-closes on
        // BOTH prove and verify. The reconciled delta-stack path (Wave 2/3) will re-enable
        // logical-stack counts by authenticating the delta chain to the head root.
        assert!(head.state.parent_state_root.is_some());
        assert_eq!(head.state.materialized_count, 0);
        assert!(RecursiveRelationEnvelope::prove(&head, &plan).is_err());
        // A well-formed empty-base envelope must also refuse to VERIFY against a delta head.
        assert!(empty
            .verify(&head, head.state.commitment_root, &plan)
            .is_err());
    }

    #[test]
    fn recursive_generated_u64_filter_derives_exact_comparison_and_rejects_tamper() {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for object in ["x", "y", "z"] {
            builder
                .add_value_triple(ValueTriple::new_node(
                    &iri("a"),
                    &iri("p"),
                    &iri(object),
                ))
                .unwrap();
        }
        let layer = builder.commit().unwrap().proof_commitment().unwrap();
        let source = RelationProofPlan {
            query_digest: [97u8; 32],
            kind: RelationProofPlanKind::LayerTripleCount,
        };
        let plan = |op, constant, constant_on_left| RelationProofPlan {
            query_digest: [98u8; 32],
            kind: RelationProofPlanKind::GeneratedU64Filter {
                input: Box::new(source.clone()),
                column: 0,
                op,
                constant,
                constant_on_left,
            },
        };

        for (op, constant, constant_on_left, expected_len) in [
            (GeneratedU64ComparisonOp::Eq, 3, false, 1),
            (GeneratedU64ComparisonOp::Eq, 2, false, 0),
            (GeneratedU64ComparisonOp::Gt, 0, false, 1),
            (GeneratedU64ComparisonOp::Ge, 3, false, 1),
            (GeneratedU64ComparisonOp::Lt, u64::MAX, false, 1),
            (GeneratedU64ComparisonOp::Le, u64::MAX, false, 1),
            (GeneratedU64ComparisonOp::Lt, 3, true, 0),
            (GeneratedU64ComparisonOp::Le, 3, true, 1),
        ] {
            let plan = plan(op, constant, constant_on_left);
            let envelope = RecursiveRelationEnvelope::prove(&layer, &plan).unwrap();
            envelope
                .verify(&layer, layer.state.commitment_root, &plan)
                .unwrap();
            assert_eq!(envelope.result_len, expected_len);
            let decoded = RecursiveRelationEnvelope::decode(&envelope.encode().unwrap()).unwrap();
            decoded
                .verify(&layer, layer.state.commitment_root, &plan)
                .unwrap();
        }

        let true_plan = plan(GeneratedU64ComparisonOp::Ge, 3, false);
        let envelope = RecursiveRelationEnvelope::prove(&layer, &true_plan).unwrap();
        let wrong_operator = plan(GeneratedU64ComparisonOp::Gt, 3, false);
        assert!(envelope
            .verify(&layer, layer.state.commitment_root, &wrong_operator)
            .is_err());
        let wrong_operand_order = plan(GeneratedU64ComparisonOp::Lt, 3, true);
        assert!(envelope
            .verify(&layer, layer.state.commitment_root, &wrong_operand_order)
            .is_err());

        let mut tampered = envelope.clone();
        tampered.result_columns[0][0] = ArkFr::from_u64(4);
        tampered.result_commitments[0] =
            commit_column(&[ArkFr::from_u64(4), ArkFr::zero()]).unwrap();
        tampered.result_values =
            vec![vec![CanonicalResultValue::GeneratedDecimalU64(4)]];
        assert!(tampered
            .verify(&layer, layer.state.commitment_root, &true_plan)
            .is_err());

        let empty_store = open_sync_memory_store();
        let empty_layer = empty_store
            .create_base_layer()
            .unwrap()
            .commit()
            .unwrap()
            .proof_commitment()
            .unwrap();
        let zero = plan(GeneratedU64ComparisonOp::Eq, 0, false);
        let zero_envelope = RecursiveRelationEnvelope::prove(&empty_layer, &zero).unwrap();
        zero_envelope
            .verify(&empty_layer, empty_layer.state.commitment_root, &zero)
            .unwrap();
        assert_eq!(zero_envelope.result_len, 1);
        assert_eq!(
            zero_envelope.result_values,
            vec![vec![CanonicalResultValue::GeneratedDecimalU64(0)]]
        );
    }

    #[test]
    fn recursive_group_sum_preserves_duplicates_zero_and_safe_u32_max_and_rejects_tamper() {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for (predicate, value) in [("p", 0u32), ("q", u32::MAX)] {
            builder
                .add_value_triple(ValueTriple::new_value(
                    &iri("b"),
                    &iri(predicate),
                    u32::make_entry(&value),
                ))
                .unwrap();
        }
        let base = builder.commit().unwrap();
        let builder = base.open_write().unwrap();
        // The lexically earlier key is introduced in a child and therefore has a later
        // Store ID. Output must still follow SWI atom order (`a` before `b`).
        for predicate in ["p", "q"] {
            builder
                .add_value_triple(ValueTriple::new_value(
                    &iri("a"),
                    &iri(predicate),
                    u32::make_entry(&2u32),
                ))
                .unwrap();
        }
        let layer = builder.commit().unwrap().proof_commitment().unwrap();
        let source = RelationProofPlan {
            query_digest: [99u8; 32],
            kind: RelationProofPlanKind::Bgp(LayerBgpPlan {
                source: ColumnSource::Materialized,
                atoms: vec![LayerBgpAtom {
                    subject: AtomTerm::Variable,
                    predicate_term: Vec::new(),
                    predicate_variable: true,
                    object: AtomTerm::Variable,
                    object_range: None,
                    object_kind: ObjectKindFilter::Any,
                    subject_object_equal: false,
                    source: ColumnSource::Materialized,
                }],
                joins: Vec::new(),
            }),
        };
        let plan = RelationProofPlan {
            query_digest: [100u8; 32],
            kind: RelationProofPlanKind::GroupSum {
                input: Box::new(source.clone()),
                key_columns: vec![0],
                value_column: 2,
                datatype: CanonicalDatatype::UInt32,
            },
        };
        // COMMITMENT DIET (Wave 0): this GroupSum reads the DELTA head's materialized view,
        // which is no longer committed — proving reject-closes at the envelope entry. The
        // reconciled delta-stack path (Wave 2/3) will re-enable delta-head aggregates over an
        // authenticated delta chain. (`source`/`plan` build a materialized-source aggregate.)
        assert!(layer.state.parent_state_root.is_some());
        assert!(layer.materialized.is_empty());
        let _ = &source;
        assert!(RecursiveRelationEnvelope::prove(&layer, &plan).is_err());
    }

    #[test]
    fn recursive_two_key_group_sum_uses_declared_lexical_tuple_order() {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        builder
            .add_value_triple(ValueTriple::new_node(&iri("z"), &iri("link"), &iri("z")))
            .unwrap();
        for value in [1u32, 2] {
            builder
                .add_value_triple(ValueTriple::new_value(
                    &iri("z"),
                    &iri("score"),
                    u32::make_entry(&value),
                ))
                .unwrap();
        }
        let base = builder.commit().unwrap();
        let builder = base.open_write().unwrap();
        for (left, right) in [("a", "a"), ("a", "z"), ("z", "a")] {
            builder
                .add_value_triple(ValueTriple::new_node(
                    &iri(left),
                    &iri("link"),
                    &iri(right),
                ))
                .unwrap();
        }
        for value in [3u32, 4, 5] {
            builder
                .add_value_triple(ValueTriple::new_value(
                    &iri("a"),
                    &iri("score"),
                    u32::make_entry(&value),
                ))
                .unwrap();
        }
        let layer = builder.commit().unwrap().proof_commitment().unwrap();
        let source = RelationProofPlan {
            query_digest: [112u8; 32],
            kind: RelationProofPlanKind::Bgp(LayerBgpPlan {
                source: ColumnSource::Materialized,
                atoms: vec![
                    LayerBgpAtom::all_variables(iri("link").into_bytes()),
                    LayerBgpAtom::all_variables(iri("score").into_bytes()),
                ],
                joins: vec![JoinStep {
                    acc_key: 1,
                    rel_key: 0,
                    additional_keys: Vec::new(),
                }],
            }),
        };
        let plan = RelationProofPlan {
            query_digest: [113u8; 32],
            kind: RelationProofPlanKind::GroupSum {
                input: Box::new(source.clone()),
                // Child schema is [left,right,value]; declared group priority is
                // deliberately [right,left].
                key_columns: vec![1, 0],
                value_column: 2,
                datatype: CanonicalDatatype::UInt32,
            },
        };
        // COMMITMENT DIET (Wave 0): this two-key GroupSum reads the DELTA head's materialized
        // view, which is no longer committed — proving reject-closes at the envelope entry.
        // Reconciliation (Wave 2/3) will re-enable delta-head aggregates.
        assert!(layer.state.parent_state_root.is_some());
        assert!(layer.materialized.is_empty());
        let _ = &source;
        assert!(RecursiveRelationEnvelope::prove(&layer, &plan).is_err());
    }

    fn assert_recursive_two_key_group_extrema_uses_declared_node_tuple_order() {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        builder
            .add_value_triple(ValueTriple::new_node(&iri("z"), &iri("link"), &iri("z")))
            .unwrap();
        for value in [1u32, 2] {
            builder
                .add_value_triple(ValueTriple::new_value(
                    &iri("z"), &iri("score"), u32::make_entry(&value),
                ))
                .unwrap();
        }
        let base = builder.commit().unwrap();
        let builder = base.open_write().unwrap();
        for (left, right) in [("a", "a"), ("a", "z"), ("z", "a")] {
            builder
                .add_value_triple(ValueTriple::new_node(
                    &iri(left), &iri("link"), &iri(right),
                ))
                .unwrap();
        }
        for value in [3u32, 4, 5] {
            builder
                .add_value_triple(ValueTriple::new_value(
                    &iri("a"), &iri("score"), u32::make_entry(&value),
                ))
                .unwrap();
        }
        let layer = builder.commit().unwrap().proof_commitment().unwrap();
        let source = RelationProofPlan {
            query_digest: [131u8; 32],
            kind: RelationProofPlanKind::Bgp(LayerBgpPlan {
                source: ColumnSource::Materialized,
                atoms: vec![
                    LayerBgpAtom::all_variables(iri("link").into_bytes()),
                    LayerBgpAtom::all_variables(iri("score").into_bytes()),
                ],
                joins: vec![JoinStep {
                    acc_key: 1,
                    rel_key: 0,
                    additional_keys: Vec::new(),
                }],
            }),
        };
        let plan = RelationProofPlan {
            query_digest: [132u8; 32],
            kind: RelationProofPlanKind::GroupExtrema {
                input: Box::new(source.clone()),
                key_columns: vec![1, 0],
                value_column: 2,
                datatype: CanonicalDatatype::UInt32,
                kind: GroupExtremaKind::Min,
            },
        };
        // COMMITMENT DIET (Wave 0): this two-key GroupExtrema reads the DELTA head's
        // materialized view, which is no longer committed — proving reject-closes at the
        // envelope entry. Reconciliation (Wave 2/3) will re-enable delta-head aggregates.
        assert!(layer.state.parent_state_root.is_some());
        assert!(layer.materialized.is_empty());
        let _ = &source;
        assert!(RecursiveRelationEnvelope::prove(&layer, &plan).is_err());
    }

    #[test]
    fn recursive_two_key_group_extrema_uses_declared_node_tuple_order() {
        std::thread::Builder::new()
            .name("recursive-two-key-group-extrema".into())
            .stack_size(256 * 1024 * 1024)
            .spawn(assert_recursive_two_key_group_extrema_uses_declared_node_tuple_order)
            .unwrap()
            .join()
            .unwrap();
    }

    fn assert_recursive_zero_key_group_extrema_emits_only_for_nonempty_input() {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for (subject, value) in [("a", i64::MIN), ("b", -3), ("c", -3), ("d", i64::MAX)] {
            builder.add_value_triple(ValueTriple::new_value(
                &iri(subject), &iri("score"), i64::make_entry(&value),
            )).unwrap();
        }
        let layer = builder.commit().unwrap().proof_commitment().unwrap();
        let source = |predicate: &str, digest| RelationProofPlan {
            query_digest: digest,
            kind: RelationProofPlanKind::Bgp(LayerBgpPlan {
                source: ColumnSource::Materialized,
                atoms: vec![LayerBgpAtom::all_variables(iri(predicate).into_bytes())],
                joins: Vec::new(),
            }),
        };
        let plan = RelationProofPlan {
            query_digest: [134u8; 32],
            kind: RelationProofPlanKind::GroupExtrema {
                input: Box::new(source("score", [135u8; 32])),
                key_columns: Vec::new(),
                value_column: 1,
                datatype: CanonicalDatatype::Int64,
                kind: GroupExtremaKind::Max,
            },
        };
        let envelope = RecursiveRelationEnvelope::prove(&layer, &plan).unwrap();
        envelope.verify(&layer, layer.state.commitment_root, &plan).unwrap();
        assert_eq!(envelope.result_len, 1);
        let expected = CanonicalResultValue::Object(CanonicalObject::from_store_object(
            &ValueTriple::new_value("", "", i64::make_entry(&i64::MAX)).object,
        ));
        assert_eq!(envelope.result_values, vec![vec![expected]]);

        let empty_plan = RelationProofPlan {
            query_digest: [136u8; 32],
            kind: RelationProofPlanKind::GroupExtrema {
                input: Box::new(source("missing", [137u8; 32])),
                key_columns: Vec::new(),
                value_column: 1,
                datatype: CanonicalDatatype::Int64,
                kind: GroupExtremaKind::Min,
            },
        };
        let empty = RecursiveRelationEnvelope::prove(&layer, &empty_plan).unwrap();
        empty.verify(&layer, layer.state.commitment_root, &empty_plan).unwrap();
        assert_eq!(empty.result_len, 0);
        assert_eq!(empty.result_values, vec![Vec::new()]);
    }


    #[test]
    fn recursive_zero_key_group_extrema_emits_only_for_nonempty_input() {
        std::thread::Builder::new()
            .name("recursive-zero-key-group-extrema".into())
            .stack_size(256 * 1024 * 1024)
            .spawn(assert_recursive_zero_key_group_extrema_emits_only_for_nonempty_input)
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn recursive_two_key_group_count_preserves_bag_cardinality_and_tuple_order() {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        builder
            .add_value_triple(ValueTriple::new_node(&iri("z"), &iri("link"), &iri("z")))
            .unwrap();
        for value in [1u32, 2] {
            builder
                .add_value_triple(ValueTriple::new_value(
                    &iri("z"),
                    &iri("score"),
                    u32::make_entry(&value),
                ))
                .unwrap();
        }
        let base = builder.commit().unwrap();
        let builder = base.open_write().unwrap();
        for (left, right) in [("a", "a"), ("a", "z"), ("z", "a")] {
            builder
                .add_value_triple(ValueTriple::new_node(
                    &iri(left),
                    &iri("link"),
                    &iri(right),
                ))
                .unwrap();
        }
        for value in [3u32, 4, 5] {
            builder
                .add_value_triple(ValueTriple::new_value(
                    &iri("a"),
                    &iri("score"),
                    u32::make_entry(&value),
                ))
                .unwrap();
        }
        let layer = builder.commit().unwrap().proof_commitment().unwrap();
        let source = RelationProofPlan {
            query_digest: [117u8; 32],
            kind: RelationProofPlanKind::Bgp(LayerBgpPlan {
                source: ColumnSource::Materialized,
                atoms: vec![
                    LayerBgpAtom::all_variables(iri("link").into_bytes()),
                    LayerBgpAtom::all_variables(iri("score").into_bytes()),
                ],
                joins: vec![JoinStep {
                    acc_key: 1,
                    rel_key: 0,
                    additional_keys: Vec::new(),
                }],
            }),
        };
        let plan = RelationProofPlan {
            query_digest: [118u8; 32],
            kind: RelationProofPlanKind::GroupCount {
                input: Box::new(source.clone()),
                key_columns: vec![1, 0],
            },
        };
        // COMMITMENT DIET (Wave 0): this two-key GroupCount reads the DELTA head's
        // materialized view, which is no longer committed — proving reject-closes at the
        // envelope entry. Reconciliation (Wave 2/3) will re-enable delta-head aggregates.
        assert!(layer.state.parent_state_root.is_some());
        assert!(layer.materialized.is_empty());
        let _ = &source;
        assert!(RecursiveRelationEnvelope::prove(&layer, &plan).is_err());
    }

    #[test]
    fn recursive_collection_count_emits_zero_for_empty_and_rejects_group_substitution() {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for object in ["x", "y", "z"] {
            builder
                .add_value_triple(ValueTriple::new_node(
                    &iri("a"),
                    &iri("p"),
                    &iri(object),
                ))
                .unwrap();
        }
        let layer = builder.commit().unwrap().proof_commitment().unwrap();
        let source = |predicate: &str, digest| RelationProofPlan {
            query_digest: digest,
            kind: RelationProofPlanKind::Bgp(LayerBgpPlan {
                source: ColumnSource::Materialized,
                atoms: vec![LayerBgpAtom::all_variables(iri(predicate).into_bytes())],
                joins: Vec::new(),
            }),
        };
        let plan = RelationProofPlan {
            query_digest: [120u8; 32],
            kind: RelationProofPlanKind::CollectionCount {
                input: Box::new(source("p", [121u8; 32])),
            },
        };
        let envelope = RecursiveRelationEnvelope::prove(&layer, &plan).unwrap();
        envelope
            .verify(&layer, layer.state.commitment_root, &plan)
            .unwrap();
        assert_eq!(
            envelope.result_values,
            vec![vec![CanonicalResultValue::GeneratedDecimalU64(3)]]
        );
        let decoded = RecursiveRelationEnvelope::decode(&envelope.encode().unwrap()).unwrap();
        decoded
            .verify(&layer, layer.state.commitment_root, &plan)
            .unwrap();
        let group_plan = RelationProofPlan {
            query_digest: [120u8; 32],
            kind: RelationProofPlanKind::GroupCount {
                input: Box::new(source("p", [121u8; 32])),
                key_columns: Vec::new(),
            },
        };
        assert!(decoded
            .verify(&layer, layer.state.commitment_root, &group_plan)
            .is_err());
        let mut tampered = decoded;
        tampered.result_columns[0][0] = ArkFr::from_u64(4);
        tampered.result_values[0][0] = CanonicalResultValue::GeneratedDecimalU64(4);
        tampered.result_commitments[0] =
            commit_column(&[ArkFr::from_u64(4), ArkFr::zero()]).unwrap();
        assert!(tampered
            .verify(&layer, layer.state.commitment_root, &plan)
            .is_err());

        let empty_plan = RelationProofPlan {
            query_digest: [122u8; 32],
            kind: RelationProofPlanKind::CollectionCount {
                input: Box::new(source("missing", [123u8; 32])),
            },
        };
        let empty = RecursiveRelationEnvelope::prove(&layer, &empty_plan).unwrap();
        empty
            .verify(&layer, layer.state.commitment_root, &empty_plan)
            .unwrap();
        assert_eq!(empty.result_len, 1);
        assert_eq!(
            empty.result_values,
            vec![vec![CanonicalResultValue::GeneratedDecimalU64(0)]]
        );
    }

    #[test]
    fn recursive_collection_sum_handles_signed_cancellation_empty_and_domain_bounds() {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for value in [-3i64, 1, 2] {
            builder
                .add_value_triple(ValueTriple::new_value(
                    &iri("a"),
                    &iri("p"),
                    i64::make_entry(&value),
                ))
                .unwrap();
        }
        let layer = builder.commit().unwrap().proof_commitment().unwrap();
        let source = RelationProofPlan {
            query_digest: [124u8; 32],
            kind: RelationProofPlanKind::Bgp(LayerBgpPlan {
                source: ColumnSource::Materialized,
                atoms: vec![LayerBgpAtom::all_variables(iri("p").into_bytes())],
                joins: Vec::new(),
            }),
        };
        let plan = RelationProofPlan {
            query_digest: [125u8; 32],
            kind: RelationProofPlanKind::CollectionSum {
                input: Box::new(source.clone()),
                template: CollectionSumTemplate::Column(1),
                datatype: CanonicalDatatype::Int64,
            },
        };
        let envelope = RecursiveRelationEnvelope::prove(&layer, &plan).unwrap();
        envelope
            .verify(&layer, layer.state.commitment_root, &plan)
            .unwrap();
        assert_eq!(
            envelope.result_values,
            vec![vec![CanonicalResultValue::GeneratedDecimalI64(0)]]
        );
        let decoded = RecursiveRelationEnvelope::decode(&envelope.encode().unwrap()).unwrap();
        decoded
            .verify(&layer, layer.state.commitment_root, &plan)
            .unwrap();
        let group_plan = RelationProofPlan {
            query_digest: [125u8; 32],
            kind: RelationProofPlanKind::GroupSum {
                input: Box::new(source.clone()),
                key_columns: Vec::new(),
                value_column: 1,
                datatype: CanonicalDatatype::Int64,
            },
        };
        assert!(decoded
            .verify(&layer, layer.state.commitment_root, &group_plan)
            .is_err());
        let count_plan = RelationProofPlan {
            query_digest: [125u8; 32],
            kind: RelationProofPlanKind::CollectionCount {
                input: Box::new(source),
            },
        };
        assert!(decoded
            .verify(&layer, layer.state.commitment_root, &count_plan)
            .is_err());
        let mut tampered = decoded;
        tampered.result_columns[0][0] = generated_decimal_i64_field(1);
        tampered.result_values[0][0] = CanonicalResultValue::GeneratedDecimalI64(1);
        tampered.result_commitments[0] =
            commit_column(&[generated_decimal_i64_field(1), ArkFr::zero()]).unwrap();
        assert!(tampered
            .verify(&layer, layer.state.commitment_root, &plan)
            .is_err());

        let RecursiveRelationProof::CollectionSum { input } = &envelope.proof else {
            unreachable!()
        };
        let mut one = (**input).clone();
        one.result_len = 1;
        assert_eq!(
            derived_collection_sum_column(
                &layer,
                &one,
                CollectionSumTemplate::UnsignedConstant(u64::MAX),
                CanonicalDatatype::UInt64,
            )
            .unwrap(),
            vec![vec![ArkFr::from_u64(u64::MAX)]]
        );
        assert!(derived_collection_sum_column(
            &layer,
            input,
            CollectionSumTemplate::UnsignedConstant(u64::MAX),
            CanonicalDatatype::UInt64,
        )
        .is_err());
        assert!(derived_collection_sum_column(
            &layer,
            input,
            CollectionSumTemplate::SignedConstant(i64::MAX),
            CanonicalDatatype::Int64,
        )
        .is_err());
        let mut empty = (**input).clone();
        empty.result_len = 0;
        assert_eq!(
            derived_collection_sum_column(
                &layer,
                &empty,
                CollectionSumTemplate::SignedConstant(-5),
                CanonicalDatatype::Int64,
            )
            .unwrap(),
            vec![vec![ArkFr::zero()]]
        );
    }

    #[test]
    fn recursive_collection_extrema_preserves_original_id_and_empty_has_no_row() {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for (subject, value) in [
            ("a", i64::MIN),
            ("b", -3),
            ("c", -3),
            ("d", 2),
            ("e", i64::MAX),
        ] {
            builder
                .add_value_triple(ValueTriple::new_value(
                    &iri(subject),
                    &iri("p"),
                    i64::make_entry(&value),
                ))
                .unwrap();
        }
        let layer = builder.commit().unwrap().proof_commitment().unwrap();
        let source = |predicate: &str, digest| RelationProofPlan {
            query_digest: digest,
            kind: RelationProofPlanKind::Bgp(LayerBgpPlan {
                source: ColumnSource::Materialized,
                atoms: vec![LayerBgpAtom::all_variables(iri(predicate).into_bytes())],
                joins: Vec::new(),
            }),
        };
        let extrema = |kind, digest| RelationProofPlan {
            query_digest: digest,
            kind: RelationProofPlanKind::CollectionExtrema {
                input: Box::new(source("p", [126u8; 32])),
                value_column: 1,
                datatype: CanonicalDatatype::Int64,
                kind,
            },
        };
        let min_plan = extrema(GroupExtremaKind::Min, [127u8; 32]);
        let min = RecursiveRelationEnvelope::prove(&layer, &min_plan).unwrap();
        min.verify(&layer, layer.state.commitment_root, &min_plan)
            .unwrap();
        assert_eq!(min.result_len, 1);
        let expected_min = CanonicalObject::from_store_object(
            &ValueTriple::new_value("", "", i64::make_entry(&i64::MIN)).object,
        );
        assert_eq!(
            min.result_values,
            vec![vec![CanonicalResultValue::Object(expected_min)]]
        );
        let decoded = RecursiveRelationEnvelope::decode(&min.encode().unwrap()).unwrap();
        decoded
            .verify(&layer, layer.state.commitment_root, &min_plan)
            .unwrap();

        let max_plan = extrema(GroupExtremaKind::Max, [128u8; 32]);
        let max = RecursiveRelationEnvelope::prove(&layer, &max_plan).unwrap();
        max.verify(&layer, layer.state.commitment_root, &max_plan)
            .unwrap();
        assert_eq!(max.result_len, 1);
        assert!(decoded
            .verify(&layer, layer.state.commitment_root, &max_plan)
            .is_err());

        let RecursiveRelationProof::CollectionExtrema { input } = &decoded.proof else {
            unreachable!()
        };
        let non_extreme = input.result_columns[1][1];
        let non_extreme_value = input.result_values[1][1].clone();
        let mut tampered = decoded;
        tampered.result_columns[0][0] = non_extreme;
        tampered.result_values[0][0] = non_extreme_value;
        tampered.result_commitments[0] =
            commit_column(&[non_extreme, ArkFr::zero()]).unwrap();
        assert!(tampered
            .verify(&layer, layer.state.commitment_root, &min_plan)
            .is_err());

        let empty_plan = RelationProofPlan {
            query_digest: [129u8; 32],
            kind: RelationProofPlanKind::CollectionExtrema {
                input: Box::new(source("missing", [130u8; 32])),
                value_column: 1,
                datatype: CanonicalDatatype::Int64,
                kind: GroupExtremaKind::Min,
            },
        };
        let empty = RecursiveRelationEnvelope::prove(&layer, &empty_plan).unwrap();
        empty
            .verify(&layer, layer.state.commitment_root, &empty_plan)
            .unwrap();
        assert_eq!(empty.result_len, 0);
        assert_eq!(empty.result_values, vec![Vec::new()]);
    }

    #[test]
    fn recursive_group_sum_accepts_exact_u64_totals_and_rejects_actual_overflow() {
        let layer_with = |values: &[(&str, &str, u64)]| {
            let store = open_sync_memory_store();
            let builder = store.create_base_layer().unwrap();
            for (subject, predicate, value) in values {
                builder
                    .add_value_triple(ValueTriple::new_value(
                        &iri(subject),
                        &iri(predicate),
                        u64::make_entry(value),
                    ))
                    .unwrap();
            }
            builder.commit().unwrap().proof_commitment().unwrap()
        };
        let source = || RelationProofPlan {
            query_digest: [105u8; 32],
            kind: RelationProofPlanKind::Bgp(LayerBgpPlan {
                source: ColumnSource::Materialized,
                atoms: vec![LayerBgpAtom {
                    subject: AtomTerm::Variable,
                    predicate_term: Vec::new(),
                    predicate_variable: true,
                    object: AtomTerm::Variable,
                    object_range: None,
                    object_kind: ObjectKindFilter::Any,
                    subject_object_equal: false,
                    source: ColumnSource::Materialized,
                }],
                joins: Vec::new(),
            }),
        };
        let sum_plan = |key_columns| RelationProofPlan {
            query_digest: [106u8; 32],
            kind: RelationProofPlanKind::GroupSum {
                input: Box::new(source()),
                key_columns,
                value_column: 2,
                datatype: CanonicalDatatype::UInt64,
            },
        };

        // The maximum is accepted as a checked partitioned sum. This is safe because
        // the BN254 scalar modulus exceeds u64::MAX.
        let partitioned = layer_with(&[("a", "p", u64::MAX - 5), ("a", "q", 5)]);
        let keyed_plan = sum_plan(vec![0]);
        let keyed = RecursiveRelationEnvelope::prove(&partitioned, &keyed_plan).unwrap();
        keyed
            .verify(&partitioned, partitioned.state.commitment_root, &keyed_plan)
            .unwrap();
        assert_eq!(
            keyed.result_values[1],
            vec![CanonicalResultValue::GeneratedDecimalU64(u64::MAX)]
        );

        let near_max = layer_with(&[("a", "p", u64::MAX - 5), ("a", "q", 4)]);
        let near_max_plan = sum_plan(vec![]);
        let near_max_envelope =
            RecursiveRelationEnvelope::prove(&near_max, &near_max_plan).unwrap();
        near_max_envelope
            .verify(&near_max, near_max.state.commitment_root, &near_max_plan)
            .unwrap();
        assert_eq!(
            near_max_envelope.result_values,
            vec![vec![CanonicalResultValue::GeneratedDecimalU64(u64::MAX - 1)]]
        );

        let mut tampered = near_max_envelope;
        tampered.result_columns[0][0] = ArkFr::from_u64(u64::MAX);
        let mut padded = tampered.result_columns[0].clone();
        padded.resize(1usize << tampered.result_log_domain, ArkFr::zero());
        tampered.result_commitments[0] = commit_column(&padded).unwrap();
        assert!(tampered
            .verify(&near_max, near_max.state.commitment_root, &near_max_plan)
            .is_err());

        let wrong_datatype = RelationProofPlan {
            query_digest: [107u8; 32],
            kind: RelationProofPlanKind::GroupSum {
                input: Box::new(source()),
                key_columns: vec![],
                value_column: 2,
                datatype: CanonicalDatatype::UInt32,
            },
        };
        assert!(RecursiveRelationEnvelope::prove(&partitioned, &wrong_datatype).is_err());

        let overflow = layer_with(&[("a", "p", u64::MAX), ("a", "q", 1)]);
        let overflow_plan = sum_plan(vec![]);
        assert!(RecursiveRelationEnvelope::prove(&overflow, &overflow_plan).is_err());
        let child = RecursiveRelationEnvelope::prove(&overflow, &source()).unwrap();
        assert!(derived_group_sum_columns(
            &overflow,
            &child,
            &[],
            2,
            CanonicalDatatype::UInt64,
        )
        .is_err());
    }

    fn assert_recursive_group_sum_authenticates_signed_values_domains_and_overflow() {
        for value in [i64::MIN, -1, 0, 1, i64::MAX] {
            assert_eq!(
                super::super::query::generated_decimal_i64_value(
                    generated_decimal_i64_field(value),
                )
                .unwrap(),
                value
            );
        }
        for (datatype, lexical, expected) in [
            (CanonicalDatatype::Int8, vec![(-3i8 as u8) ^ 0x80], -3i128),
            (
                CanonicalDatatype::Int16,
                ((-300i16 as u16) ^ (1 << 15)).to_be_bytes().to_vec(),
                -300i128,
            ),
            (
                CanonicalDatatype::Int32,
                ((-30_000i32 as u32) ^ (1 << 31)).to_be_bytes().to_vec(),
                -30_000i128,
            ),
            (
                CanonicalDatatype::Int64,
                ((-3_000_000_000i64 as u64) ^ (1 << 63)).to_be_bytes().to_vec(),
                -3_000_000_000i128,
            ),
        ] {
            let GroupSumValue::Signed(actual) = integral_catalog_value(
                &CanonicalObject::Value { datatype, lexical },
                datatype,
            )
            .unwrap()
            else {
                panic!("signed datatype decoded into unsigned sum domain")
            };
            assert_eq!(actual, expected);
        }
        let store = open_sync_memory_store();
        let base_builder = store.create_base_layer().unwrap();
        for (predicate, value) in [("p", i64::MIN), ("q", i64::MAX), ("r", 1)] {
            base_builder
                .add_value_triple(ValueTriple::new_value(
                    &iri("b"),
                    &iri(predicate),
                    i64::make_entry(&value),
                ))
                .unwrap();
        }
        let base = base_builder.commit().unwrap();
        let child_builder = base.open_write().unwrap();
        // Duplicate numeric solutions use distinct predicates because triples are a set.
        // `a` is appended after `b`, but must remain the first canonical node group.
        for (predicate, value) in [("p", -5i64), ("q", 2), ("r", 2)] {
            child_builder
                .add_value_triple(ValueTriple::new_value(
                    &iri("a"),
                    &iri(predicate),
                    i64::make_entry(&value),
                ))
                .unwrap();
        }
        let layer = child_builder.commit().unwrap().proof_commitment().unwrap();
        let source = RelationProofPlan {
            query_digest: [108u8; 32],
            kind: RelationProofPlanKind::Bgp(LayerBgpPlan {
                source: ColumnSource::Materialized,
                atoms: vec![LayerBgpAtom {
                    subject: AtomTerm::Variable,
                    predicate_term: Vec::new(),
                    predicate_variable: true,
                    object: AtomTerm::Variable,
                    object_range: None,
                    object_kind: ObjectKindFilter::Any,
                    subject_object_equal: false,
                    source: ColumnSource::Materialized,
                }],
                joins: Vec::new(),
            }),
        };
        let keyed_plan = RelationProofPlan {
            query_digest: [109u8; 32],
            kind: RelationProofPlanKind::GroupSum {
                input: Box::new(source.clone()),
                key_columns: vec![0],
                value_column: 2,
                datatype: CanonicalDatatype::Int64,
            },
        };
        // COMMITMENT DIET (Wave 0): every GroupSum / generated-filter over this DELTA head
        // reads its materialized view (no longer committed), so proving reject-closes at the
        // envelope entry. Reconciliation (Wave 2/3) will re-enable delta-head aggregates.
        assert!(layer.state.parent_state_root.is_some());
        assert!(layer.materialized.is_empty());
        let _ = &keyed_plan;
        assert!(RecursiveRelationEnvelope::prove(&layer, &keyed_plan).is_err());

        // The signed-value OVERFLOW authentication below runs over BASE layers and is
        // unaffected by the commitment diet, so it is retained verbatim.
        let global_plan = RelationProofPlan {
            query_digest: [110u8; 32],
            kind: RelationProofPlanKind::GroupSum {
                input: Box::new(source.clone()),
                key_columns: vec![],
                value_column: 2,
                datatype: CanonicalDatatype::Int64,
            },
        };
        let overflow_layer = |values: [i64; 2]| {
            let store = open_sync_memory_store();
            let builder = store.create_base_layer().unwrap();
            for (predicate, value) in [("p", values[0]), ("q", values[1])] {
                builder
                    .add_value_triple(ValueTriple::new_value(
                        &iri("a"),
                        &iri(predicate),
                        i64::make_entry(&value),
                    ))
                    .unwrap();
            }
            builder.commit().unwrap().proof_commitment().unwrap()
        };
        for overflow in [
            overflow_layer([i64::MAX, 1]),
            overflow_layer([i64::MIN, -1]),
        ] {
            assert!(RecursiveRelationEnvelope::prove(&overflow, &global_plan).is_err());
            let child = RecursiveRelationEnvelope::prove(&overflow, &source).unwrap();
            assert!(derived_group_sum_columns(
                &overflow,
                &child,
                &[],
                2,
                CanonicalDatatype::Int64,
            )
            .is_err());
        }
    }

    #[test]
    fn recursive_group_sum_authenticates_signed_values_domains_and_overflow() {
        std::thread::Builder::new()
            .name("store-generated-i64-filter-proof".into())
            .stack_size(256 * 1024 * 1024)
            .spawn(assert_recursive_group_sum_authenticates_signed_values_domains_and_overflow)
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn recursive_distinct_proves_exact_classes_roundtrips_and_rejects_tamper() {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for (subject, object) in [("a", "x"), ("a", "y"), ("b", "z")] {
            builder
                .add_value_triple(ValueTriple::new_node(
                    &iri(subject),
                    &iri("p"),
                    &iri(object),
                ))
                .unwrap();
        }
        let layer = builder.commit().unwrap().proof_commitment().unwrap();
        let bgp = LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![LayerBgpAtom::all_variables(iri("p").into_bytes())],
            joins: Vec::new(),
        };
        let leaf = RelationProofPlan {
            query_digest: plan_digest(&bgp).unwrap(),
            kind: RelationProofPlanKind::Bgp(bgp),
        };
        let plan = RelationProofPlan {
            query_digest: [93u8; 32],
            kind: RelationProofPlanKind::Distinct {
                input: Box::new(leaf),
                key_columns: vec![0],
            },
        };
        let envelope = RecursiveRelationEnvelope::prove(&layer, &plan).unwrap();
        envelope.verify(&layer, layer.state.commitment_root, &plan).unwrap();
        assert_eq!(envelope.result_len, 2);
        assert_eq!(envelope.result_columns.len(), 1);

        let bytes = envelope.encode().unwrap();
        let decoded = RecursiveRelationEnvelope::decode(&bytes).unwrap();
        decoded.verify(&layer, layer.state.commitment_root, &plan).unwrap();

        let mut wrong_key = plan.clone();
        if let RelationProofPlanKind::Distinct { key_columns, .. } = &mut wrong_key.kind {
            key_columns[0] = 1;
        }
        assert!(decoded
            .verify(&layer, layer.state.commitment_root, &wrong_key)
            .is_err());

        let mut duplicate = decoded.clone();
        duplicate.result_columns[0][1] = duplicate.result_columns[0][0];
        duplicate.result_commitments[0] = commit_column(&duplicate.result_columns[0]).unwrap();
        assert!(duplicate
            .verify(&layer, layer.state.commitment_root, &plan)
            .is_err());

        let mut changed_proof = decoded.clone();
        if let RecursiveRelationProof::Distinct { proof, .. } = &mut changed_proof.proof {
            proof.entries[0].lengths[0] += 1;
        }
        assert!(changed_proof
            .verify(&layer, layer.state.commitment_root, &plan)
            .is_err());
    }

    #[test]
    fn ordered_slice_bounds_cover_zero_overflow_and_wrapper_order() {
        assert_eq!(ordered_slice_bounds(3, 0, None), (0, 3));
        assert_eq!(ordered_slice_bounds(3, 0, Some(0)), (0, 0));
        assert_eq!(ordered_slice_bounds(3, 99, None), (3, 3));
        assert_eq!(ordered_slice_bounds(3, u64::MAX, Some(u64::MAX)), (3, 3));

        // start(1, limit(2, Goal)): the inner limit first yields [0,2), then start
        // drops one, leaving [1,2). limit(2, start(1, Goal)) instead yields [1,3).
        let inner_limit = ordered_slice_bounds(3, 0, Some(2));
        let outer_start = ordered_slice_bounds(inner_limit.1 - inner_limit.0, 1, None);
        assert_eq!((inner_limit.0 + outer_start.0, inner_limit.0 + outer_start.1), (1, 2));
        let inner_start = ordered_slice_bounds(3, 1, None);
        let outer_limit = ordered_slice_bounds(inner_start.1 - inner_start.0, 0, Some(2));
        assert_eq!((inner_start.0 + outer_limit.0, inner_start.0 + outer_limit.1), (1, 3));
    }

    fn assert_recursive_ordered_slice_roundtrips_and_rejects_content_and_plan_tamper() {
        let (layer, bgp) = order_fixture();
        let unordered_leaf = RelationProofPlan {
            query_digest: plan_digest(&bgp).unwrap(),
            kind: RelationProofPlanKind::Bgp(bgp.clone()),
        };
        let leaf = RelationProofPlan {
            query_digest: plan_digest(&bgp).unwrap(),
            kind: RelationProofPlanKind::Bgp(bgp),
        };
        let order = RelationProofPlan {
            query_digest: [41u8; 32],
            kind: RelationProofPlanKind::ExactScalarOrder {
                input: Box::new(leaf),
                datatype: CanonicalDatatype::UInt32,
                direction: OrderDirection::Ascending,
            },
        };
        let plan = RelationProofPlan {
            query_digest: [43u8; 32],
            kind: RelationProofPlanKind::ExactOrderedSlice {
                input: Box::new(order),
                offset: 1,
                limit: Some(2),
            },
        };
        let envelope = RecursiveRelationEnvelope::prove(&layer, &plan).unwrap();
        envelope.verify(&layer, layer.state.commitment_root, &plan).unwrap();
        assert_eq!(envelope.result_len, 2);
        let bytes = envelope.encode().unwrap();
        let decoded = RecursiveRelationEnvelope::decode(&bytes).unwrap();
        decoded.verify(&layer, layer.state.commitment_root, &plan).unwrap();

        let mut wrong_plan = plan.clone();
        if let RelationProofPlanKind::ExactOrderedSlice { limit, .. } = &mut wrong_plan.kind {
            *limit = Some(1);
        }
        assert!(decoded.verify(&layer, layer.state.commitment_root, &wrong_plan).is_err());

        let mut reordered = RecursiveRelationEnvelope::decode(&bytes).unwrap();
        reordered.result_columns[0].swap(0, 1);
        let mut padded = reordered.result_columns[0].clone();
        padded.resize(1usize << reordered.result_log_domain, ArkFr::zero());
        reordered.result_commitments[0] = commit_column(&padded).unwrap();
        assert!(reordered.verify(&layer, layer.state.commitment_root, &plan).is_err());

        let unordered = RelationProofPlan {
            query_digest: [44u8; 32],
            kind: RelationProofPlanKind::ExactOrderedSlice {
                input: Box::new(unordered_leaf),
                offset: 0,
                limit: Some(1),
            },
        };
        assert!(unordered.result_shape().is_err());
    }

    #[test]
    fn recursive_ordered_slice_roundtrips_and_rejects_content_and_plan_tamper() {
        // Recursive Dory proof values are deliberately boxed in the envelope, but the
        // debug prover itself has large temporary stack frames. Give this arithmetic
        // test the same explicit headroom used by ordinary proving hosts.
        std::thread::Builder::new()
            .name("ordered-slice-proof".into())
            .stack_size(256 * 1024 * 1024)
            .spawn(assert_recursive_ordered_slice_roundtrips_and_rejects_content_and_plan_tamper)
            .unwrap()
            .join()
            .unwrap();
    }

    fn assert_recursive_multikey_order_roundtrip_and_tamper() {
        let (layer, bgp) = multi_order_fixture();
        let child = RelationProofPlan { query_digest: plan_digest(&bgp).unwrap(), kind: RelationProofPlanKind::Bgp(bgp) };
        let plan = RelationProofPlan { query_digest: [61u8; 32], kind: RelationProofPlanKind::ExactMultiOrder {
            input: Box::new(child), datatypes: vec![CanonicalDatatype::UInt32; 2],
            key_indices: vec![1, 0],
            directions: vec![OrderDirection::Descending, OrderDirection::Ascending],
        }};
        let envelope = RecursiveRelationEnvelope::prove(&layer, &plan).unwrap();
        envelope.verify(&layer, layer.state.commitment_root, &plan).unwrap();
        assert_eq!(envelope.result_len, 4);
        let bytes = envelope.encode().unwrap();
        let decoded = RecursiveRelationEnvelope::decode(&bytes).unwrap();
        decoded.verify(&layer, layer.state.commitment_root, &plan).unwrap();

        let mut wrong_direction = plan.clone();
        if let RelationProofPlanKind::ExactMultiOrder { directions, .. } = &mut wrong_direction.kind { directions[1] = OrderDirection::Descending; }
        assert!(decoded.verify(&layer, layer.state.commitment_root, &wrong_direction).is_err());
        let mut wrong_type = plan.clone();
        if let RelationProofPlanKind::ExactMultiOrder { datatypes, .. } = &mut wrong_type.kind { datatypes[0] = CanonicalDatatype::Int32; }
        assert!(decoded.verify(&layer, layer.state.commitment_root, &wrong_type).is_err());
        let mut wrong_key_order = plan.clone();
        if let RelationProofPlanKind::ExactMultiOrder { key_indices, .. } = &mut wrong_key_order.kind { key_indices[1] = 1; }
        assert!(decoded.verify(&layer, layer.state.commitment_root, &wrong_key_order).is_err());

        let mut reordered = RecursiveRelationEnvelope::decode(&bytes).unwrap();
        for column in &mut reordered.result_columns { column.swap(0, 1); }
        for (commitment, column) in reordered.result_commitments.iter_mut().zip(&reordered.result_columns) {
            let mut padded = column.clone(); padded.resize(1usize << reordered.result_log_domain, ArkFr::zero());
            *commitment = commit_column(&padded).unwrap();
        }
        assert!(reordered.verify(&layer, layer.state.commitment_root, &plan).is_err());
    }

    #[test]
    fn recursive_multikey_order_roundtrip_and_tamper() {
        std::thread::Builder::new().name("multi-order-proof".into()).stack_size(256 * 1024 * 1024)
            .spawn(assert_recursive_multikey_order_roundtrip_and_tamper).unwrap().join().unwrap();
    }

    #[test]
    fn decoder_item_budget_is_global_across_nested_collections() {
        let mut reader = Reader {
            bytes: &[],
            pos: 0,
            decoded_items: 0,
        };
        reader.claim_items(MAX_ITEMS - 1).unwrap();
        reader.claim_items(1).unwrap();
        assert!(reader.claim_items(1).is_err());
    }

    #[test]
    fn recursive_plan_rejects_nullable_right_correlation_keys() {
        let (layer, bgp) = fixture();
        let digest = [7u8; 32];
        let leaf = RelationProofPlan {
            query_digest: digest,
            kind: RelationProofPlanKind::Bgp(bgp),
        };
        let nullable_right = RelationProofPlan {
            query_digest: digest,
            kind: RelationProofPlanKind::LeftOuter {
                left: Box::new(leaf.clone()),
                right: Box::new(leaf.clone()),
                key_pairs: Vec::new(),
            },
        };
        let right_shape = nullable_right.result_shape().unwrap().0;
        let nullable_index = right_shape
            .iter()
            .position(|descriptor| descriptor.nullable)
            .unwrap();
        for kind in [
            RelationProofPlanKind::LeftAnti {
                left: Box::new(leaf.clone()),
                right: Box::new(nullable_right.clone()),
                key_pairs: vec![(0, nullable_index)],
            },
            RelationProofPlanKind::LeftOuter {
                left: Box::new(leaf.clone()),
                right: Box::new(nullable_right.clone()),
                key_pairs: vec![(0, nullable_index)],
            },
        ] {
            let plan = RelationProofPlan {
                query_digest: digest,
                kind,
            };
            assert!(plan.result_shape().is_err());
        }
        drop(layer);
    }

    #[test]
    fn ground_result_unit_is_authenticated_but_not_dictionary_resolved() {
        let (layer, _) = fixture();
        let plan = LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![LayerBgpAtom {
                subject: AtomTerm::NodeConstant(iri("a").into_bytes()),
                predicate_term: iri("knows").into_bytes(),
                predicate_variable: false,
                object: AtomTerm::NodeConstant(iri("b").into_bytes()),
                object_range: None,
                object_kind: ObjectKindFilter::Any,
                subject_object_equal: false,
                source: ColumnSource::Materialized,
            }],
            joins: Vec::new(),
        };
        let result = prove_layer_bgp_with_result(&layer, &plan).unwrap();
        assert_eq!(result.result_len, 1);
        let result_columns = result.real_columns();
        let query_digest = plan_digest(&plan).unwrap();
        let mut envelope = LayerBgpEnvelope::new_verified(
            &layer,
            layer.state.commitment_root,
            &plan,
            query_digest,
            result.proof,
            result.result_commitments,
            result_columns,
            result.result_log_domain,
            result.result_len,
            result.predicate_ids,
        )
        .unwrap();
        assert!(envelope.result_objects.is_empty());
        envelope
            .verify(&layer, layer.state.commitment_root, &plan, query_digest)
            .unwrap();
        envelope.result_log_domain += 1;
        assert!(envelope
            .verify(&layer, layer.state.commitment_root, &plan, query_digest)
            .is_err());
    }

    #[test]
    fn envelope_roundtrips_and_rejects_tampering_and_version_confusion() {
        let (layer, plan) = fixture();
        let result = prove_layer_bgp_with_result(&layer, &plan).unwrap();
        let result_columns = result.real_columns();
        let query_digest = plan_digest(&plan).unwrap();
        let envelope = LayerBgpEnvelope::new_verified(
            &layer,
            layer.state.commitment_root,
            &plan,
            query_digest,
            result.proof,
            result.result_commitments,
            result_columns,
            result.result_log_domain,
            result.result_len,
            result.predicate_ids,
        )
        .unwrap();
        let encoded = envelope.encode().unwrap();
        let decoded = LayerBgpEnvelope::decode(&encoded).unwrap();
        decoded
            .verify(&layer, layer.state.commitment_root, &plan, query_digest)
            .unwrap();
        assert_eq!(decoded.result_objects.len(), 2);

        let mut corrupt = encoded.clone();
        corrupt[MAGIC.len() + 8] ^= 1;
        assert!(LayerBgpEnvelope::decode(&corrupt).is_err());

        let mut wrong_version = encoded.clone();
        wrong_version[MAGIC.len()..MAGIC.len() + 4].copy_from_slice(&6u32.to_le_bytes());
        refresh_checksum(&mut wrong_version);
        assert!(LayerBgpEnvelope::decode(&wrong_version).is_err());

        let wrong_plan = LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![LayerBgpAtom::all_variables(iri("likes").into_bytes())],
            joins: Vec::new(),
        };
        assert!(decoded
            .verify(
                &layer,
                layer.state.commitment_root,
                &wrong_plan,
                query_digest
            )
            .is_err());

        let mut wrong_root = layer.state.commitment_root;
        wrong_root[0] ^= 1;
        assert!(decoded
            .verify(&layer, wrong_root, &plan, query_digest)
            .is_err());

        // The first proof field is the atom vector, immediately after the three
        // fixed-size hashes. Reject its claimed allocation before reading elements.
        let mut oversized_count = encoded.clone();
        let atom_count_offset = MAGIC.len() + 4 + 3 * 32;
        oversized_count[atom_count_offset..atom_count_offset + 8]
            .copy_from_slice(&((MAX_ITEMS as u64) + 1).to_le_bytes());
        refresh_checksum(&mut oversized_count);
        assert!(LayerBgpEnvelope::decode(&oversized_count).is_err());

        // Exactly-at-cap counts pass the numeric guard. A truncated vector of the
        // materially large LayerBgpAtomProof type must fail before reserving space for
        // MAX_ITEMS elements (which would dwarf the entire input-size limit).
        let mut truncated_at_cap = encoded[..atom_count_offset + 8].to_vec();
        truncated_at_cap[atom_count_offset..atom_count_offset + 8]
            .copy_from_slice(&(MAX_ITEMS as u64).to_le_bytes());
        let checksum = Sha256::digest(&truncated_at_cap);
        truncated_at_cap.extend_from_slice(&checksum);
        let error = LayerBgpEnvelope::decode(&truncated_at_cap).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);

        let mut forged_objects = decoded.clone();
        forged_objects.result_objects[0][0] = CanonicalObject::Node(iri("forged").into_bytes());
        assert!(forged_objects
            .verify(&layer, layer.state.commitment_root, &plan, query_digest)
            .is_err());
    }

    #[test]
    fn recursive_leaf_roundtrips_and_binds_plan_identity_and_schema() {
        let (layer, bgp) = fixture();
        let (plan, envelope) = recursive_leaf(&layer, &bgp);
        envelope
            .verify(&layer, layer.state.commitment_root, &plan)
            .unwrap();
        let encoded = envelope.encode().unwrap();
        let decoded = RecursiveRelationEnvelope::decode(&encoded).unwrap();
        decoded
            .verify(&layer, layer.state.commitment_root, &plan)
            .unwrap();

        let mut wrong_digest = plan.clone();
        wrong_digest.query_digest[0] ^= 1;
        assert!(decoded
            .verify(&layer, layer.state.commitment_root, &wrong_digest)
            .is_err());

        let mut wrong_schema = decoded.clone();
        wrong_schema.result_descriptors[0].nullable = true;
        assert!(wrong_schema
            .verify(&layer, layer.state.commitment_root, &plan)
            .is_err());

        let mut wrong_variant = plan.clone();
        wrong_variant.kind = RelationProofPlanKind::LeftAnti {
            left: Box::new(plan.clone()),
            right: Box::new(plan.clone()),
            key_pairs: Vec::new(),
        };
        assert!(decoded
            .verify(&layer, layer.state.commitment_root, &wrong_variant)
            .is_err());
    }

    #[cfg(any(feature = "blitzar-cpu", feature = "blitzar-gpu"))]
    #[test]
    fn recursive_proving_context_preserves_the_wire_format() {
        let (layer, bgp) = fixture();
        let plan = RelationProofPlan {
            query_digest: plan_digest(&bgp).unwrap(),
            kind: RelationProofPlanKind::Bgp(bgp),
        };
        let runtime = StoreProverContext::initialize_blitzar(
            dory_pcs::backends::arkworks::BlitzarRuntimeConfig {
                num_precomputed_generators: 0,
            },
        );
        let accelerated = StoreProverContext::blitzar(
            &runtime,
            super::super::argument::ProverSetupTier::Log20,
        )
        .unwrap();
        drop(runtime);

        let cpu = RecursiveRelationEnvelope::prove(&layer, &plan).unwrap();
        let with_context =
            RecursiveRelationEnvelope::prove_with_context(&accelerated, &layer, &plan).unwrap();
        assert_eq!(cpu.encode().unwrap(), with_context.encode().unwrap());
        with_context
            .verify(&layer, layer.state.commitment_root, &plan)
            .unwrap();
    }

    #[test]
    fn recursive_envelope_rejects_depth_bomb_before_proof_verification() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(RECURSIVE_MAGIC);
        bytes.extend_from_slice(&RECURSIVE_VERSION.to_le_bytes());
        // Only the hostile prefix is needed: decoding descends into the left child of
        // each anti node and rejects the depth before touching any missing proof bytes.
        for _ in 0..=MAX_RELATION_PROOF_DEPTH {
            [0u8; 32].put(&mut bytes).unwrap();
            [1u8; 32].put(&mut bytes).unwrap();
            2u8.put(&mut bytes).unwrap();
        }
        let checksum = Sha256::digest(&bytes);
        bytes.extend_from_slice(&checksum);
        let error = RecursiveRelationEnvelope::decode(&bytes).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("structural limits"));
    }

    #[test]
    fn recursive_envelope_rejects_node_bomb_before_allocation() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(RECURSIVE_MAGIC);
        bytes.extend_from_slice(&RECURSIVE_VERSION.to_le_bytes());
        [0u8; 32].put(&mut bytes).unwrap();
        [1u8; 32].put(&mut bytes).unwrap();
        1u8.put(&mut bytes).unwrap();
        MAX_RELATION_PROOF_NODES.put(&mut bytes).unwrap();
        let checksum = Sha256::digest(&bytes);
        bytes.extend_from_slice(&checksum);
        let error = RecursiveRelationEnvelope::decode(&bytes).unwrap_err();
        assert!(error.to_string().contains("too many recursive union branches"));
    }

    // ---- Bounded transitive closure (`star`/`plus`) ----

    fn closure_layer(triples: &[(&str, &str, &str)]) -> LayerCommitment {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for (subject, predicate, object) in triples {
            builder
                .add_value_triple(ValueTriple::new_node(
                    &iri(subject),
                    &iri(predicate),
                    &iri(object),
                ))
                .unwrap();
        }
        (*builder.commit().unwrap().proof_commitment().unwrap()).clone()
    }

    fn closure_plan_dir(
        predicate: &str,
        seed: &str,
        reflexive: bool,
        inverted: bool,
    ) -> RelationProofPlan {
        RelationProofPlan {
            query_digest: [7u8; 32],
            kind: RelationProofPlanKind::TransitiveClosure {
                // The single-predicate edge is a `Bgp` leaf relation (the closure edge is now a general
                // `RelationProofPlan`); it carries the closure's own digest as its identity tag.
                edge: Box::new(RelationProofPlan {
                    query_digest: [7u8; 32],
                    kind: RelationProofPlanKind::Bgp(LayerBgpPlan {
                        source: ColumnSource::Materialized,
                        atoms: vec![LayerBgpAtom::all_variables(iri(predicate).into_bytes())],
                        joins: Vec::new(),
                    }),
                }),
                seed: CanonicalObject::Node(iri(seed).into_bytes()),
                reflexive,
                inverted,
            },
        }
    }

    fn closure_plan(predicate: &str, seed: &str, reflexive: bool) -> RelationProofPlan {
        closure_plan_dir(predicate, seed, reflexive, false)
    }

    fn closure_result_set(envelope: &RecursiveRelationEnvelope) -> std::collections::BTreeSet<String> {
        envelope.result_values[0]
            .iter()
            .map(|value| match value {
                CanonicalResultValue::Object(CanonicalObject::Node(bytes)) => {
                    String::from_utf8(bytes.clone()).unwrap()
                }
                other => panic!("unexpected closure result value {other:?}"),
            })
            .collect()
    }

    fn iris(names: &[&str]) -> std::collections::BTreeSet<String> {
        names.iter().map(|name| iri(name)).collect()
    }

    fn assert_transitive_closure_end_to_end() {
        // Straight chain a -> b -> c -> d.
        let layer = closure_layer(&[("a", "p", "b"), ("b", "p", "c"), ("c", "p", "d")]);
        let root = layer.state.commitment_root;

        // Plus from a = {b, c, d} (one-or-more hops).
        let plus = closure_plan("p", "a", false);
        let env = RecursiveRelationEnvelope::prove(&layer, &plus).unwrap();
        env.verify(&layer, root, &plus).unwrap();
        assert_eq!(closure_result_set(&env), iris(&["b", "c", "d"]));
        // Wire round-trip still verifies.
        let bytes = env.encode().unwrap();
        RecursiveRelationEnvelope::decode(&bytes)
            .unwrap()
            .verify(&layer, root, &plus)
            .unwrap();

        // Star from a = {a, b, c, d} (adds the identity a).
        let star = closure_plan("p", "a", true);
        let env_star = RecursiveRelationEnvelope::prove(&layer, &star).unwrap();
        env_star.verify(&layer, root, &star).unwrap();
        assert_eq!(closure_result_set(&env_star), iris(&["a", "b", "c", "d"]));

        // Substituting the star proof under the plus plan (or vice versa) must reject.
        assert!(env_star.verify(&layer, root, &plus).is_err());
        assert!(env.verify(&layer, root, &star).is_err());

        // CYCLE a -> b -> a: the closure must TERMINATE (empty-delta fixpoint) and give the exact set.
        let cyclic = closure_layer(&[("a", "p", "b"), ("b", "p", "a")]);
        let cyclic_root = cyclic.state.commitment_root;
        let cyclic_plus = closure_plan("p", "a", false);
        let cyclic_env = RecursiveRelationEnvelope::prove(&cyclic, &cyclic_plus).unwrap();
        cyclic_env.verify(&cyclic, cyclic_root, &cyclic_plus).unwrap();
        // plus from a on a<->b = {a, b} (a re-reached via a -> b -> a).
        assert_eq!(closure_result_set(&cyclic_env), iris(&["a", "b"]));

        // Bound TARGET via inversion: over a -> b -> c -> d, the INVERSE plus from d = the set of nodes
        // that reach d = {a, b, c}. Same edge relation; only the traversal direction (join key) flips.
        let inverse_plus = closure_plan_dir("p", "d", false, true);
        let inverse_env = RecursiveRelationEnvelope::prove(&layer, &inverse_plus).unwrap();
        inverse_env.verify(&layer, root, &inverse_plus).unwrap();
        assert_eq!(closure_result_set(&inverse_env), iris(&["a", "b", "c"]));
        // The forward plus and the inverse plus are distinct plans: their proofs are not interchangeable.
        assert!(inverse_env.verify(&layer, root, &plus).is_err());
        assert!(env.verify(&layer, root, &inverse_plus).is_err());

        // ---- Soundness negatives ----

        // (i) A TRUNCATED closure that has NOT converged: drop the last round so the delivered
        // accumulator is A_{K-1} while the fixpoint was proven over A_K. The fixpoint inner join binds
        // A_K's commitment, which no longer matches -> reject.
        let mut truncated = env.clone();
        if let RecursiveRelationProof::TransitiveClosure { rounds, .. } = &mut truncated.proof {
            assert!(rounds.len() >= 2, "chain a->b->c->d should need >= 2 rounds");
            rounds.pop();
        } else {
            panic!("expected a transitive-closure proof");
        }
        assert!(truncated.verify(&layer, root, &plus).is_err());

        // (ii) A SPURIOUS node injected into the final round's set: it never came from an authenticated
        // InnerJoin, so the round's Distinct proof no longer covers the tampered rows -> reject.
        let mut spurious = env.clone();
        if let RecursiveRelationProof::TransitiveClosure { rounds, .. } = &mut spurious.proof {
            let ghost = 999_999u64;
            rounds.last_mut().unwrap().next_rows.push(ghost);
        }
        assert!(spurious.verify(&layer, root, &plus).is_err());

        // (iii) A TAMPERED delivered result set (drop a genuine reachable node, recommit) no longer
        // equals the proven reachable set -> reject.
        let mut tampered = env.clone();
        tampered.result_columns[0].pop();
        tampered.result_len -= 1;
        tampered.result_values[0].pop();
        tampered.result_commitments[0] = {
            let mut column = tampered.result_columns[0].clone();
            column.resize(1usize << tampered.result_log_domain, ArkFr::zero());
            commit_column(&column).unwrap()
        };
        assert!(tampered.verify(&layer, root, &plus).is_err());
    }

    #[test]
    fn transitive_closure_star_plus_cycle_and_soundness() {
        std::thread::Builder::new()
            .name("transitive-closure-proof".into())
            .stack_size(128 * 1024 * 1024)
            .spawn(assert_transitive_closure_end_to_end)
            .unwrap()
            .join()
            .unwrap();
    }

    // ---- Bounded ALL-PAIRS transitive closure (2-column pairs reachability) ----

    fn closure_pairs_plan_dir(predicate: &str, inverted: bool, digest: [u8; 32]) -> RelationProofPlan {
        RelationProofPlan {
            query_digest: digest,
            kind: RelationProofPlanKind::TransitiveClosurePairs {
                // The single-predicate pairs edge is a `Bgp` leaf carrying the closure's own digest as its
                // identity tag (the pairs closure edge is now a general `RelationProofPlan`) — byte-identical
                // to the previous `closure_edge_plan(edge, digest)` wrapping.
                edge: Box::new(RelationProofPlan {
                    query_digest: digest,
                    kind: RelationProofPlanKind::Bgp(LayerBgpPlan {
                        source: ColumnSource::Materialized,
                        atoms: vec![LayerBgpAtom::all_variables(iri(predicate).into_bytes())],
                        joins: Vec::new(),
                    }),
                }),
                inverted,
            },
        }
    }

    fn closure_pairs_result_set(
        env: &RecursiveRelationEnvelope,
    ) -> std::collections::BTreeSet<(String, String)> {
        let node = |value: &CanonicalResultValue| match value {
            CanonicalResultValue::Object(CanonicalObject::Node(bytes)) => {
                String::from_utf8(bytes.clone()).unwrap()
            }
            other => panic!("unexpected closure result value {other:?}"),
        };
        env.result_values[0]
            .iter()
            .zip(&env.result_values[1])
            .map(|(x, y)| (node(x), node(y)))
            .collect()
    }

    fn iri_pairs(pairs: &[(&str, &str)]) -> std::collections::BTreeSet<(String, String)> {
        pairs.iter().map(|(x, y)| (iri(x), iri(y))).collect()
    }

    fn assert_all_pairs_closure_end_to_end() {
        // 3-cycle a -> b -> c -> a plus a feeder d -> a (d lies on no cycle). The closure must
        // TERMINATE (empty-delta fixpoint) despite the cycle.
        let layer = closure_layer(&[
            ("a", "p", "b"),
            ("b", "p", "c"),
            ("c", "p", "a"),
            ("d", "p", "a"),
        ]);
        let root = layer.state.commitment_root;

        let plus = closure_pairs_plan_dir("p", false, [9u8; 32]);
        let env = RecursiveRelationEnvelope::prove(&layer, &plus).unwrap();
        env.verify(&layer, root, &plus).unwrap();
        // a, b, c each reach every node of the cycle INCLUDING themselves (via the 3-cycle); d reaches
        // {a, b, c} but not itself (no path returns to d). 12 pairs.
        let forward = iri_pairs(&[
            ("a", "a"), ("a", "b"), ("a", "c"),
            ("b", "a"), ("b", "b"), ("b", "c"),
            ("c", "a"), ("c", "b"), ("c", "c"),
            ("d", "a"), ("d", "b"), ("d", "c"),
        ]);
        assert_eq!(closure_pairs_result_set(&env), forward);

        // Wire round-trip still verifies (exercises the tag-25 put/get).
        let bytes = env.encode().unwrap();
        RecursiveRelationEnvelope::decode(&bytes)
            .unwrap()
            .verify(&layer, root, &plus)
            .unwrap();

        // INVERSE all-pairs = the TRANSPOSE of the forward relation: (x, y) iff y reaches x forward.
        let inverse_plan = closure_pairs_plan_dir("p", true, [10u8; 32]);
        let inverse_env = RecursiveRelationEnvelope::prove(&layer, &inverse_plan).unwrap();
        inverse_env.verify(&layer, root, &inverse_plan).unwrap();
        let inverse = iri_pairs(&[
            ("a", "a"), ("a", "b"), ("a", "c"),
            ("b", "a"), ("b", "b"), ("b", "c"),
            ("c", "a"), ("c", "b"), ("c", "c"),
            ("a", "d"), ("b", "d"), ("c", "d"),
        ]);
        assert_eq!(closure_pairs_result_set(&inverse_env), inverse);
        // Forward and inverse are distinct plans (distinct digests): proofs are not interchangeable.
        assert!(inverse_env.verify(&layer, root, &plus).is_err());
        assert!(env.verify(&layer, root, &inverse_plan).is_err());

        // ---- Soundness negatives ----

        // (i) A TRUNCATED (non-converged) closure: drop the last round so the delivered accumulator is
        // A_{K-1} while the fixpoint was proven over A_K. The fixpoint inner join binds A_K's commitment
        // at a step index that no longer matches -> reject.
        let mut truncated = env.clone();
        if let RecursiveRelationProof::TransitiveClosurePairs { rounds, .. } = &mut truncated.proof {
            assert!(!rounds.is_empty(), "the cyclic graph needs >= 1 round");
            rounds.pop();
        } else {
            panic!("expected an all-pairs closure proof");
        }
        assert!(truncated.verify(&layer, root, &plus).is_err());

        // (ii) A SPURIOUS pair injected into the final round's set: it never came from an authenticated
        // InnerJoin, so the round's 2-column Distinct proof no longer covers the tampered rows -> reject.
        let mut spurious = env.clone();
        if let RecursiveRelationProof::TransitiveClosurePairs { rounds, .. } = &mut spurious.proof {
            rounds.last_mut().unwrap().next_pairs.push((999_998, 999_999));
        }
        assert!(spurious.verify(&layer, root, &plus).is_err());

        // (iii) A DROPPED genuine reachable pair from the delivered result (recommit both columns) no
        // longer equals the proven reachable-pairs set -> reject.
        let mut tampered = env.clone();
        tampered.result_columns[0].pop();
        tampered.result_columns[1].pop();
        tampered.result_len -= 1;
        tampered.result_values[0].pop();
        tampered.result_values[1].pop();
        let redomain = 1usize << tampered.result_log_domain;
        for col in 0..2 {
            let mut column = tampered.result_columns[col].clone();
            column.resize(redomain, ArkFr::zero());
            tampered.result_commitments[col] = commit_column(&column).unwrap();
        }
        assert!(tampered.verify(&layer, root, &plus).is_err());
    }

    #[test]
    fn all_pairs_transitive_closure_cycle_and_soundness() {
        std::thread::Builder::new()
            .name("all-pairs-closure-proof".into())
            .stack_size(128 * 1024 * 1024)
            .spawn(assert_all_pairs_closure_end_to_end)
            .unwrap()
            .join()
            .unwrap();
    }

    // -----------------------------------------------------------------------
    // Ordered rdf:List walk (ReadDocument increment 1).
    // -----------------------------------------------------------------------

    fn rdf_first() -> String {
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#first".to_string()
    }
    fn rdf_rest() -> String {
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#rest".to_string()
    }
    fn rdf_nil() -> String {
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#nil".to_string()
    }
    fn cell(i: usize) -> String {
        format!("http://ex/cell{i}")
    }

    /// Build a base layer from raw node triples (IRIs used verbatim — NOT `iri()`-wrapped — so the RDF
    /// Collection constants keep their exact well-known IRIs).
    fn list_layer(triples: Vec<(String, String, String)>) -> LayerCommitment {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for (s, p, o) in &triples {
            builder
                .add_value_triple(ValueTriple::new_node(s.as_str(), p.as_str(), o.as_str()))
                .unwrap();
        }
        (*builder.commit().unwrap().proof_commitment().unwrap()).clone()
    }

    /// The standard cons chain `S -P-> [elements...]` using cells `cell0, cell1, …` and the RDF
    /// `first`/`rest`/`nil` constants.
    fn chain_triples(subject: &str, predicate: &str, elements: &[&str]) -> Vec<(String, String, String)> {
        let mut t = Vec::new();
        if elements.is_empty() {
            t.push((subject.to_string(), predicate.to_string(), rdf_nil()));
            return t;
        }
        t.push((subject.to_string(), predicate.to_string(), cell(0)));
        for (i, e) in elements.iter().enumerate() {
            t.push((cell(i), rdf_first(), iri(e)));
            let rest = if i + 1 < elements.len() { cell(i + 1) } else { rdf_nil() };
            t.push((cell(i), rdf_rest(), rest));
        }
        t
    }

    fn list_walk_plan(subject: &str, predicate: &str, digest: u8) -> RelationProofPlan {
        RelationProofPlan {
            query_digest: [digest; 32],
            kind: RelationProofPlanKind::OrderedListWalk {
                subject: CanonicalObject::Node(subject.as_bytes().to_vec()),
                predicate: predicate.as_bytes().to_vec(),
            },
        }
    }

    fn node_object_id(layer: &LayerCommitment, iri_str: &str) -> u64 {
        layer
            .node_value_catalog
            .iter()
            .find(|r| matches!(&r.triple.object, CanonicalObject::Node(n) if n.as_slice() == iri_str.as_bytes()))
            .map(|r| r.object_id)
            .expect("node id present in layer dictionary")
    }

    fn walk_elements(env: &RecursiveRelationEnvelope) -> Vec<String> {
        env.result_values[0]
            .iter()
            .map(|v| match v {
                CanonicalResultValue::Object(CanonicalObject::Node(b)) => {
                    String::from_utf8(b.clone()).unwrap()
                }
                other => panic!("unexpected list element value {:?}", other),
            })
            .collect()
    }

    fn big_stack(f: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .name("ordered-list-walk-proof".into())
            .stack_size(512 * 1024 * 1024)
            .spawn(f)
            .unwrap()
            .join()
            .unwrap();
    }

    fn assert_ordered_list_walk_happy() {
        let s = iri("doc");
        let p = iri("tracks");
        // Duplicate element `a` at two positions: order + multiplicity are both non-trivial.
        let layer = list_layer(chain_triples(&s, &p, &["a", "b", "a"]));
        let root = layer.state.commitment_root;
        let plan = list_walk_plan(&s, &p, 42);

        let env = RecursiveRelationEnvelope::prove(&layer, &plan).unwrap();
        env.verify(&layer, root, &plan).unwrap();
        assert_eq!(walk_elements(&env), vec![iri("a"), iri("b"), iri("a")]);

        // Wire round-trip still verifies.
        let bytes = env.encode().unwrap();
        RecursiveRelationEnvelope::decode(&bytes)
            .unwrap()
            .verify(&layer, root, &plan)
            .unwrap();

        // Single-element list.
        let s1 = iri("solo");
        let one = list_layer(chain_triples(&s1, &p, &["z"]));
        let one_plan = list_walk_plan(&s1, &p, 42);
        let one_env = RecursiveRelationEnvelope::prove(&one, &one_plan).unwrap();
        one_env.verify(&one, one.state.commitment_root, &one_plan).unwrap();
        assert_eq!(walk_elements(&one_env), vec![iri("z")]);
    }

    fn assert_ordered_list_walk_soundness() {
        let s = iri("doc");
        let p = iri("tracks");
        let ctx = StoreProverContext::cpu();
        let subj_obj = CanonicalObject::Node(s.as_bytes().to_vec());

        // Non-palindrome baseline so a reordered order is a genuinely DIFFERENT answer.
        let layer = list_layer(chain_triples(&s, &p, &["a", "b", "c"]));
        let root = layer.state.commitment_root;
        let plan = list_walk_plan(&s, &p, 42);
        let env = RecursiveRelationEnvelope::prove(&layer, &plan).unwrap();
        env.verify(&layer, root, &plan).unwrap();
        assert_eq!(walk_elements(&env), vec![iri("a"), iri("b"), iri("c")]);

        let c0 = node_object_id(&layer, &cell(0));
        let c1 = node_object_id(&layer, &cell(1));
        let c2 = node_object_id(&layer, &cell(2));
        let a_id = node_object_id(&layer, &iri("a"));

        // (1) TAIL REORDER -> rejected by the SUCCESSOR check. C[0] is held at the true head, so the new
        // explicit C[0]==head opening PASSES; the mis-ordered tail (C[1..] is no longer the rest-chain)
        // is what the successor step rejects. This deliberately isolates the successor guard from the
        // head guard (a valid permutation-order, wrong chain order, correct head).
        let reorder = super::list_walk::prove_ordered_list_walk_forged(
            &ctx, &layer, &plan, &subj_obj, p.as_bytes(), vec![c0, c2, c1],
        );
        assert!(
            reorder.and_then(|e| e.verify(&layer, root, &plan)).is_err(),
            "tail-reordered elements must be rejected (successor)"
        );

        // (1b) WRONG C[0] (position-0 forgery) -> rejected by the NEW explicit C[0]==head opening. The
        // ordered column `[c2, c1, c0]` is a genuine permutation of the cons set (permutation PASSES),
        // has exactly one nil successor (terminator would PASS) and the right cons count (cons_len
        // PASSES) -- it only starts at a NON-head cons cell. The successor gadget constrains C[1..K] but
        // NOT C[0], so with the structural argument set aside this reversal is caught SOLELY by the
        // position-0 opening, which the verifier checks right after the permutation and BEFORE the
        // successor step. We assert the C[0]==head error is the one that fires, proving the new check
        // (not the successor) is the load-bearing guard for the first cell.
        let wrong_head = super::list_walk::prove_ordered_list_walk_forged(
            &ctx, &layer, &plan, &subj_obj, p.as_bytes(), vec![c2, c1, c0],
        );
        let wrong_head_err = wrong_head
            .and_then(|e| e.verify(&layer, root, &plan))
            .expect_err("a non-head first cons cell must be rejected");
        assert!(
            wrong_head_err.to_string().contains("C[0] != head"),
            "the explicit C[0]==head opening must be the rejecting guard, got: {wrong_head_err}"
        );

        // (2) DROPPED middle cell -> rejected by the PERMUTATION (cell set is incomplete).
        assert!(
            super::list_walk::prove_ordered_list_walk_forged(
                &ctx, &layer, &plan, &subj_obj, p.as_bytes(), vec![c0, c2],
            )
            .and_then(|e| e.verify(&layer, root, &plan))
            .is_err(),
            "dropped middle cell must be rejected (permutation)"
        );

        // (3) INJECTED extra node (not a cons cell) -> rejected by the PERMUTATION membership.
        assert!(
            super::list_walk::prove_ordered_list_walk_forged(
                &ctx, &layer, &plan, &subj_obj, p.as_bytes(), vec![c0, c1, a_id, c2],
            )
            .and_then(|e| e.verify(&layer, root, &plan))
            .is_err(),
            "injected non-cons node must be rejected (permutation)"
        );

        // (4) APPENDED tail past nil (extra/duplicate cell) -> rejected by the PERMUTATION.
        assert!(
            super::list_walk::prove_ordered_list_walk_forged(
                &ctx, &layer, &plan, &subj_obj, p.as_bytes(), vec![c0, c1, c2, c0],
            )
            .and_then(|e| e.verify(&layer, root, &plan))
            .is_err(),
            "appended tail must be rejected (permutation)"
        );

        // (5) EARLY-nil TRUNCATION (drop the last cell) -> rejected by the PERMUTATION completeness.
        assert!(
            super::list_walk::prove_ordered_list_walk_forged(
                &ctx, &layer, &plan, &subj_obj, p.as_bytes(), vec![c0, c1],
            )
            .and_then(|e| e.verify(&layer, root, &plan))
            .is_err(),
            "early truncation must be rejected (permutation)"
        );

        // (6) BRANCHING rest (a cell with two rests) -> honest prove rejects (permutation/count).
        let mut branch_rest = chain_triples(&s, &p, &["a", "b", "c"]);
        branch_rest.push((cell(1), rdf_rest(), cell(9)));
        branch_rest.push((cell(9), rdf_first(), iri("x")));
        branch_rest.push((cell(9), rdf_rest(), rdf_nil()));
        let bl = list_layer(branch_rest);
        assert!(
            RecursiveRelationEnvelope::prove(&bl, &list_walk_plan(&s, &p, 42)).is_err(),
            "branching rest must be rejected"
        );

        // (7) BRANCHING first (a cell with two firsts) -> honest prove rejects (first-join count).
        let mut branch_first = chain_triples(&s, &p, &["a", "b", "c"]);
        branch_first.push((cell(1), rdf_first(), iri("bb")));
        let bf = list_layer(branch_first);
        assert!(
            RecursiveRelationEnvelope::prove(&bf, &list_walk_plan(&s, &p, 42)).is_err(),
            "branching first must be rejected"
        );

        // (8) REST CYCLE (no nil) -> honest prove rejects (terminator count is 0; the closure also
        // never reaches nil).
        let cyc = list_layer(vec![
            (s.clone(), p.clone(), cell(0)),
            (cell(0), rdf_first(), iri("a")),
            (cell(0), rdf_rest(), cell(1)),
            (cell(1), rdf_first(), iri("b")),
            (cell(1), rdf_rest(), cell(0)),
        ]);
        assert!(
            RecursiveRelationEnvelope::prove(&cyc, &list_walk_plan(&s, &p, 42)).is_err(),
            "rest cycle must be rejected"
        );

        // (9) NESTED list element (an element that is itself a cons head) -> honest prove rejects.
        let nested = list_layer(vec![
            (s.clone(), p.clone(), cell(0)),
            (cell(0), rdf_first(), cell(5)), // element 0 IS a cons head (inner list)
            (cell(0), rdf_rest(), cell(1)),
            (cell(1), rdf_first(), iri("b")),
            (cell(1), rdf_rest(), rdf_nil()),
            (cell(5), rdf_first(), iri("q")),
            (cell(5), rdf_rest(), rdf_nil()),
        ]);
        assert!(
            RecursiveRelationEnvelope::prove(&nested, &list_walk_plan(&s, &p, 42)).is_err(),
            "nested-list element must be rejected"
        );

        // (10) WRONG-HEAD substitution: a proof for `doc` verified against a plan for a different
        // subject -> the anchor BGP plan digest no longer matches.
        let plan_wrong_subject = list_walk_plan(&iri("other"), &p, 42);
        assert!(
            env.verify(&layer, root, &plan_wrong_subject).is_err(),
            "wrong-head substitution must be rejected"
        );

        // (11) STATEMENT/TRANSCRIPT substitution: a different predicate, and a different query digest.
        let plan_wrong_pred = list_walk_plan(&s, &iri("authors"), 42);
        assert!(
            env.verify(&layer, root, &plan_wrong_pred).is_err(),
            "wrong predicate must be rejected"
        );
        let plan_wrong_digest = list_walk_plan(&s, &p, 7);
        assert!(
            env.verify(&layer, root, &plan_wrong_digest).is_err(),
            "wrong query digest must be rejected"
        );
    }

    #[test]
    fn ordered_list_walk_end_to_end() {
        big_stack(assert_ordered_list_walk_happy);
    }

    #[test]
    fn ordered_list_walk_soundness_negatives() {
        big_stack(assert_ordered_list_walk_soundness);
    }

    // -----------------------------------------------------------------------
    // Ordered sys:Array walk (ReadDocument increment 4).
    // -----------------------------------------------------------------------

    fn rdf_type() -> String {
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string()
    }
    fn sys_index() -> String {
        "http://terminusdb.com/schema/sys#index".to_string()
    }
    fn sys_value() -> String {
        "http://terminusdb.com/schema/sys#value".to_string()
    }
    fn sys_array() -> String {
        "http://terminusdb.com/schema/sys#Array".to_string()
    }

    /// One raw triple of an array seed: a plain node triple, or a `sys:index` literal (a `UInt64`, the
    /// order-preserving unsigned datatype the composite authenticates against).
    enum ArrTriple {
        Node(String, String, String),
        Index(String, u64),
    }

    /// Build a base layer from raw array triples (IRIs used verbatim so the RDF/`sys:` constants keep
    /// their exact well-known IRIs; `sys:index` literals are typed `UInt64`).
    fn array_layer(triples: Vec<ArrTriple>) -> LayerCommitment {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for triple in &triples {
            match triple {
                ArrTriple::Node(s, p, o) => {
                    builder
                        .add_value_triple(ValueTriple::new_node(s.as_str(), p.as_str(), o.as_str()))
                        .unwrap();
                }
                ArrTriple::Index(cell, index) => {
                    builder
                        .add_value_triple(ValueTriple::new_value(
                            cell.as_str(),
                            &sys_index(),
                            u64::make_entry(index),
                        ))
                        .unwrap();
                }
            }
        }
        (*builder.commit().unwrap().proof_commitment().unwrap()).clone()
    }

    /// A well-formed dense array `S -P-> {cells}`. `cells` are given in PHYSICAL storage order; each is
    /// `(index, value-suffix)`. Cell nodes are `cell0, cell1, …` (physical order), NOT index order — so
    /// a layout whose physical order differs from index order exercises the ordering.
    fn dense_array_triples(
        subject: &str,
        predicate: &str,
        cells: &[(u64, &str)],
    ) -> Vec<ArrTriple> {
        let mut t = Vec::new();
        for (k, (index, value)) in cells.iter().enumerate() {
            let c = cell(k);
            t.push(ArrTriple::Node(subject.to_string(), predicate.to_string(), c.clone()));
            t.push(ArrTriple::Node(c.clone(), rdf_type(), sys_array()));
            t.push(ArrTriple::Index(c.clone(), *index));
            t.push(ArrTriple::Node(c.clone(), sys_value(), iri(value)));
        }
        t
    }

    fn array_walk_plan(subject: &str, predicate: &str, digest: u8) -> RelationProofPlan {
        RelationProofPlan {
            query_digest: [digest; 32],
            kind: RelationProofPlanKind::OrderedArrayWalk {
                subject: CanonicalObject::Node(subject.as_bytes().to_vec()),
                predicate: predicate.as_bytes().to_vec(),
            },
        }
    }

    fn walk_values(env: &RecursiveRelationEnvelope) -> Vec<String> {
        env.result_values[0]
            .iter()
            .map(|v| match v {
                CanonicalResultValue::Object(CanonicalObject::Node(b)) => {
                    String::from_utf8(b.clone()).unwrap()
                }
                other => panic!("unexpected array element value {:?}", other),
            })
            .collect()
    }

    fn assert_ordered_array_walk_happy() {
        let s = iri("doc");
        let p = iri("items");

        // Physical order == index order: [0->x, 1->y, 2->z].
        let layer = array_layer(dense_array_triples(&s, &p, &[(0, "x"), (1, "y"), (2, "z")]));
        let root = layer.state.commitment_root;
        let plan = array_walk_plan(&s, &p, 77);
        let env = RecursiveRelationEnvelope::prove(&layer, &plan).unwrap();
        env.verify(&layer, root, &plan).unwrap();
        assert_eq!(walk_values(&env), vec![iri("x"), iri("y"), iri("z")]);

        // Wire round-trip still verifies.
        let bytes = env.encode().unwrap();
        RecursiveRelationEnvelope::decode(&bytes)
            .unwrap()
            .verify(&layer, root, &plan)
            .unwrap();

        // Physical order != index order: stored [idx2->a, idx0->b, idx1->c] => ordered [b, c, a].
        let s2 = iri("doc2");
        let layer2 = array_layer(dense_array_triples(&s2, &p, &[(2, "a"), (0, "b"), (1, "c")]));
        let plan2 = array_walk_plan(&s2, &p, 77);
        let env2 = RecursiveRelationEnvelope::prove(&layer2, &plan2).unwrap();
        env2.verify(&layer2, layer2.state.commitment_root, &plan2).unwrap();
        assert_eq!(walk_values(&env2), vec![iri("b"), iri("c"), iri("a")]);

        // Duplicate VALUE at two positions (`[a, b, a]`): values are carried positionally, never
        // deduped.
        let s3 = iri("doc3");
        let layer3 = array_layer(dense_array_triples(&s3, &p, &[(0, "a"), (1, "b"), (2, "a")]));
        let plan3 = array_walk_plan(&s3, &p, 77);
        let env3 = RecursiveRelationEnvelope::prove(&layer3, &plan3).unwrap();
        env3.verify(&layer3, layer3.state.commitment_root, &plan3).unwrap();
        assert_eq!(walk_values(&env3), vec![iri("a"), iri("b"), iri("a")]);

        // Single element.
        let s4 = iri("solo");
        let layer4 = array_layer(dense_array_triples(&s4, &p, &[(0, "z")]));
        let plan4 = array_walk_plan(&s4, &p, 77);
        let env4 = RecursiveRelationEnvelope::prove(&layer4, &plan4).unwrap();
        env4.verify(&layer4, layer4.state.commitment_root, &plan4).unwrap();
        assert_eq!(walk_values(&env4), vec![iri("z")]);
    }

    fn assert_ordered_array_walk_soundness() {
        let s = iri("doc");
        let p = iri("items");
        let ctx = StoreProverContext::cpu();
        let subj = CanonicalObject::Node(s.as_bytes().to_vec());

        // Non-sorted physical layout so the honest order is non-trivial: physical positions [2, 0, 1].
        let layer = array_layer(dense_array_triples(&s, &p, &[(2, "a"), (0, "b"), (1, "c")]));
        let root = layer.state.commitment_root;
        let plan = array_walk_plan(&s, &p, 77);
        let env = RecursiveRelationEnvelope::prove(&layer, &plan).unwrap();
        env.verify(&layer, root, &plan).unwrap();
        assert_eq!(walk_values(&env), vec![iri("b"), iri("c"), iri("a")]);

        // (1) REORDERED value column -> rejected by the DENSITY certificate. Forcing the delivery to
        // physical/cell order [0,1,2] (values [a,b,c]) makes the routed index column [2,0,1] instead of
        // the identity [0,1,2], so the permutation-to-identity fails at verify.
        let reorder = super::array_walk::prove_ordered_array_walk_forged(
            &ctx, &layer, &plan, &subj, p.as_bytes(), vec![0, 1, 2],
        );
        assert!(
            reorder.and_then(|e| e.verify(&layer, root, &plan)).is_err(),
            "reordered value column must be rejected (density permutation-to-identity)"
        );

        // (2) DROPPED cell -> the density permutation is a TOTAL bijection over all `n` cells, so a
        // routing that omits a cell (length n-1) cannot even be constructed (prove rejects).
        assert!(
            super::array_walk::prove_ordered_array_walk_forged(
                &ctx, &layer, &plan, &subj, p.as_bytes(), vec![1, 2],
            )
            .is_err(),
            "a dropped cell cannot be routed by the density bijection"
        );

        // (3) DUPLICATE index (two cells index 0, none index 1) -> honest prove rejects (density: not a
        // dense 0..n-1 permutation).
        let dup = array_layer(dense_array_triples(&s, &p, &[(0, "a"), (0, "b"), (2, "c")]));
        assert!(
            RecursiveRelationEnvelope::prove(&dup, &array_walk_plan(&s, &p, 77)).is_err(),
            "duplicate sys:index must be rejected"
        );

        // (4) INJECTED extra cell that is NOT a well-formed array cell (no sys:index) -> honest prove
        // rejects (the index functional bind's count no longer equals the cell count).
        let mut injected = dense_array_triples(&s, &p, &[(0, "a"), (1, "b"), (2, "c")]);
        injected.push(ArrTriple::Node(s.clone(), p.clone(), cell(99)));
        injected.push(ArrTriple::Node(cell(99), rdf_type(), sys_array()));
        injected.push(ArrTriple::Node(cell(99), sys_value(), iri("w")));
        let inj = array_layer(injected);
        assert!(
            RecursiveRelationEnvelope::prove(&inj, &array_walk_plan(&s, &p, 77)).is_err(),
            "an injected non-cell P-object must be rejected (index functional bind)"
        );

        // (5) GAP / sparse array `[0, 2]` (n=2, index 2 >= n) -> honest prove rejects (out-of-range
        // index; the reference null-fills a gap, so a sparse array is not a dense one).
        let sparse = array_layer(dense_array_triples(&s, &p, &[(0, "a"), (2, "b")]));
        assert!(
            RecursiveRelationEnvelope::prove(&sparse, &array_walk_plan(&s, &p, 77)).is_err(),
            "a sparse/gapped array must be rejected"
        );

        // (6) A cell with TWO sys:value -> honest prove rejects (the value functional-bind count).
        let mut two_values = dense_array_triples(&s, &p, &[(0, "a"), (1, "b"), (2, "c")]);
        two_values.push(ArrTriple::Node(cell(1), sys_value(), iri("bb")));
        let tv = array_layer(two_values);
        assert!(
            RecursiveRelationEnvelope::prove(&tv, &array_walk_plan(&s, &p, 77)).is_err(),
            "a cell with two sys:value must be rejected"
        );

        // (7) INDEX >= n (`[0, 1, 5]` with n=3) -> honest prove rejects (out-of-range index).
        let big = array_layer(dense_array_triples(&s, &p, &[(0, "a"), (1, "b"), (5, "c")]));
        assert!(
            RecursiveRelationEnvelope::prove(&big, &array_walk_plan(&s, &p, 77)).is_err(),
            "an index >= n must be rejected"
        );

        // (8) STATEMENT substitution: a proof aimed at a different subject / predicate / query digest is
        // rejected (the child plan digests / query identity no longer match).
        assert!(
            env.verify(&layer, root, &array_walk_plan(&iri("other"), &p, 77)).is_err(),
            "wrong subject must be rejected"
        );
        assert!(
            env.verify(&layer, root, &array_walk_plan(&s, &iri("authors"), 77)).is_err(),
            "wrong predicate must be rejected"
        );
        assert!(
            env.verify(&layer, root, &array_walk_plan(&s, &p, 9)).is_err(),
            "wrong query digest must be rejected"
        );
    }

    #[test]
    fn ordered_array_walk_end_to_end() {
        big_stack(assert_ordered_array_walk_happy);
    }

    #[test]
    fn ordered_array_walk_soundness_negatives() {
        big_stack(assert_ordered_array_walk_soundness);
    }

    // -----------------------------------------------------------------------
    // FOUNDATION-A / FOUNDATION-B over the reconciled `present` (ReadDocument 6c-i).
    //
    // Parity discipline: build a 2-layer delta stack (base + 1 delta) whose reconciled `present`
    // equals a ROLLED-UP materialized single layer's triple set, then assert the reconciled
    // edge-relation / closure over `present` EQUALS the materialized edge-relation / closure. The
    // reconcile >=2-records-per-group floor is respected (base = 6 adds; delta = 2 removes + 3 adds).
    // -----------------------------------------------------------------------

    fn rdf_first_iri() -> String {
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#first".to_string()
    }
    fn rdf_rest_iri() -> String {
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#rest".to_string()
    }
    fn rdf_nil_iri() -> String {
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#nil".to_string()
    }

    /// The final present triple set the delta stack and the rolled-up base agree on: a cons chain
    /// `cell0 -rest-> cell1 -rest-> cell2 -rest-> cell3 -rest-> nil` plus per-cell `first` edges.
    fn closure_final_triples() -> Vec<(String, String, String)> {
        let rest = rdf_rest_iri();
        let first = rdf_first_iri();
        let nil = rdf_nil_iri();
        vec![
            (iri("cell0"), rest.clone(), iri("cell1")),
            (iri("cell1"), rest.clone(), iri("cell2")),
            (iri("cell2"), rest.clone(), iri("cell3")),
            (iri("cell3"), rest.clone(), nil.clone()),
            (iri("cell0"), first.clone(), iri("e0")),
            (iri("cell1"), first.clone(), iri("e1")),
            (iri("cell3"), first.clone(), iri("e3")),
        ]
    }

    /// A base + 1 delta stack whose reconciled `present` == [`closure_final_triples`]. The base
    /// stores `cell0..cell2 -rest-> .. -rest-> nil` (+ first edges); the delta REMOVES the old
    /// `cell2 -rest-> nil` / `cell2 -first-> e2` and ADDS `cell2 -rest-> cell3`, `cell3 -rest-> nil`,
    /// `cell3 -first-> e3` — 2 removals + 3 additions (both >= 2, satisfying the reconcile floor).
    fn closure_delta_stack() -> Vec<std::sync::Arc<LayerCommitment>> {
        let rest = rdf_rest_iri();
        let first = rdf_first_iri();
        let nil = rdf_nil_iri();
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for (s, p, o) in [
            (iri("cell0"), rest.clone(), iri("cell1")),
            (iri("cell1"), rest.clone(), iri("cell2")),
            (iri("cell2"), rest.clone(), nil.clone()),
            (iri("cell0"), first.clone(), iri("e0")),
            (iri("cell1"), first.clone(), iri("e1")),
            (iri("cell2"), first.clone(), iri("e2")),
        ] {
            builder.add_value_triple(ValueTriple::new_node(&s, &p, &o)).unwrap();
        }
        let base = builder.commit().unwrap();
        let base_lc = base.proof_commitment().unwrap();

        let child = base.open_write().unwrap();
        child.remove_value_triple(ValueTriple::new_node(&iri("cell2"), &rest, &nil)).unwrap();
        child.remove_value_triple(ValueTriple::new_node(&iri("cell2"), &first, &iri("e2"))).unwrap();
        child.add_value_triple(ValueTriple::new_node(&iri("cell2"), &rest, &iri("cell3"))).unwrap();
        child.add_value_triple(ValueTriple::new_node(&iri("cell3"), &rest, &nil)).unwrap();
        child.add_value_triple(ValueTriple::new_node(&iri("cell3"), &first, &iri("e3"))).unwrap();
        let head = child.commit().unwrap();
        vec![base_lc, head.proof_commitment().unwrap()]
    }

    /// A rolled-up single materialized base layer holding exactly [`closure_final_triples`].
    fn closure_rolled_up() -> LayerCommitment {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for (s, p, o) in closure_final_triples() {
            builder.add_value_triple(ValueTriple::new_node(&s, &p, &o)).unwrap();
        }
        (*builder.commit().unwrap().proof_commitment().unwrap()).clone()
    }

    fn arc_refs(layers: &[std::sync::Arc<LayerCommitment>]) -> Vec<&LayerCommitment> {
        layers.iter().map(|l| l.as_ref()).collect()
    }

    /// Resolve a combined-dict node id to its IRI over the chain views (test-only reverse lookup).
    fn id_to_iri(views: &[crate::proof::VerifierCommitmentRef<'_>], id: u64) -> String {
        for view in views {
            for record in view.node_value_catalog {
                if record.object_id == id {
                    if let CanonicalObject::Node(bytes) = &record.triple.object {
                        return String::from_utf8(bytes.clone()).unwrap();
                    }
                }
            }
        }
        panic!("combined-dict id {id} absent from the chain catalogs");
    }

    fn node_iri(value: &CanonicalResultValue) -> String {
        match value {
            CanonicalResultValue::Object(CanonicalObject::Node(bytes)) => {
                String::from_utf8(bytes.clone()).unwrap()
            }
            other => panic!("unexpected result value: {:?}", other),
        }
    }

    /// Prove+verify a materialized 2-column edge relation `t(?s, predicate, ?o)` over `layer`, and
    /// return its rows as an IRI-pair set.
    fn materialized_edge_pairs(
        layer: &LayerCommitment,
        predicate: &str,
    ) -> std::collections::BTreeSet<(String, String)> {
        let plan = RelationProofPlan {
            query_digest: [41u8; 32],
            kind: RelationProofPlanKind::Bgp(LayerBgpPlan {
                source: ColumnSource::Materialized,
                atoms: vec![LayerBgpAtom::all_variables(predicate.to_string().into_bytes())],
                joins: Vec::new(),
            }),
        };
        let env = RecursiveRelationEnvelope::prove(layer, &plan).unwrap();
        env.verify(layer, layer.state.commitment_root, &plan).unwrap();
        env.result_values[0]
            .iter()
            .zip(env.result_values[1].iter())
            .map(|(s, o)| (node_iri(s), node_iri(o)))
            .collect()
    }

    /// A materialized reflexive `predicate` transitive closure seeded at `seed`, over `layer`.
    fn materialized_closure_set(
        layer: &LayerCommitment,
        predicate: &str,
        seed: &str,
    ) -> std::collections::BTreeSet<String> {
        let plan = RelationProofPlan {
            query_digest: [43u8; 32],
            kind: RelationProofPlanKind::TransitiveClosure {
                edge: Box::new(RelationProofPlan {
                    query_digest: [43u8; 32],
                    kind: RelationProofPlanKind::Bgp(LayerBgpPlan {
                        source: ColumnSource::Materialized,
                        atoms: vec![LayerBgpAtom::all_variables(predicate.to_string().into_bytes())],
                        joins: Vec::new(),
                    }),
                }),
                seed: CanonicalObject::Node(seed.to_string().into_bytes()),
                reflexive: true,
                inverted: false,
            },
        };
        let env = RecursiveRelationEnvelope::prove(layer, &plan).unwrap();
        env.verify(layer, layer.state.commitment_root, &plan).unwrap();
        closure_result_set(&env)
    }

    /// FOUNDATION-A edge-relation parity: the reconciled `t(?s, rdf:rest, ?o)` over `present` EQUALS
    /// the materialized edge relation on the rolled-up base (same rows, same IRIs). Also exercises the
    /// bound-subject anchor leaf `t(cell0, rdf:rest, ?o)` = {cell1}.
    fn assert_reconciled_edge_parity() {
        let stack = closure_delta_stack();
        let lref = arc_refs(&stack);
        let head_root = lref.last().unwrap().state.commitment_root;
        let views: Vec<crate::proof::VerifierCommitmentRef<'_>> =
            lref.iter().map(|l| l.verifier_view()).collect();

        let rest = rdf_rest_iri();
        let materialized = materialized_edge_pairs(&closure_rolled_up(), &rest);
        assert_eq!(materialized.len(), 4, "expected 4 rdf:rest edges");

        // Reconcile ONCE, then prove the edge leaf on the SHARED transcript (FOUNDATION-A).
        let context = StoreProverContext::cpu();
        let mut pt = Transcript254::new(b"parture-delta-reconcile-envelope");
        let (reconcile, artifacts) =
            crate::proof::reconcile::prove_reconcile_core(&context, &lref, &mut pt).unwrap();
        let (edge_proof, edge_witness) = super::reconciled::prove_reconciled_leaf(
            &context,
            &lref,
            &artifacts,
            super::reconciled::predicate_edge_leaf_atom(rest.as_bytes()),
            &mut pt,
        )
        .unwrap();

        // Verify reconcile + the edge leaf on a fresh mirrored transcript; get the authenticated commit.
        let mut vt = Transcript254::new(b"parture-delta-reconcile-envelope");
        let verified =
            crate::proof::reconcile::verify_reconcile_core(&lref, head_root, &reconcile, &mut vt)
                .unwrap();
        let head_nv = lref.last().unwrap().verifier_view().state.node_value_count as usize;
        let edge_commit = super::reconciled::verify_reconciled_leaf(
            &lref,
            &reconcile,
            &verified,
            head_nv,
            super::reconciled::predicate_edge_leaf_atom(rest.as_bytes()),
            &edge_proof,
            &mut vt,
        )
        .unwrap();

        // Bind the prover's delivered columns to the verified commitment (the resolve-binding step),
        // then resolve the [S, O] id rows to IRIs.
        assert_eq!(edge_witness.commitments, edge_commit.commitments);
        for (col, com) in edge_witness.columns.iter().zip(&edge_commit.commitments) {
            assert_eq!(commit_column(col).unwrap(), *com, "prover column must open the verified commitment");
        }
        let n = edge_witness.n;
        let reconciled: std::collections::BTreeSet<(String, String)> = (0..n)
            .map(|i| {
                let s = crate::proof::query::generated_decimal_u64_value(edge_witness.columns[0][i]).unwrap();
                let o = crate::proof::query::generated_decimal_u64_value(edge_witness.columns[1][i]).unwrap();
                (id_to_iri(&views, s), id_to_iri(&views, o))
            })
            .collect();
        assert_eq!(reconciled, materialized, "reconciled edge relation must equal the materialized one");

        // Bound-subject anchor leaf `t(cell0, rdf:rest, ?o)` (FOUNDATION-A, kept=[O]) = {cell1}.
        let (anchor_proof, anchor_witness) = super::reconciled::prove_reconciled_leaf(
            &context,
            &lref,
            &artifacts,
            super::reconciled::bound_subject_leaf_atom(
                &CanonicalObject::Node(iri("cell0").into_bytes()),
                rest.as_bytes(),
            ),
            &mut pt,
        )
        .unwrap();
        let anchor_commit = super::reconciled::verify_reconciled_leaf(
            &lref,
            &reconcile,
            &verified,
            head_nv,
            super::reconciled::bound_subject_leaf_atom(
                &CanonicalObject::Node(iri("cell0").into_bytes()),
                rest.as_bytes(),
            ),
            &anchor_proof,
            &mut vt,
        )
        .unwrap();
        assert_eq!(anchor_witness.n, 1, "cell0 has exactly one rdf:rest successor");
        assert_eq!(anchor_commit.n, 1);
        let head_id =
            crate::proof::query::generated_decimal_u64_value(anchor_witness.columns[0][0]).unwrap();
        assert_eq!(id_to_iri(&views, head_id), iri("cell1"));
    }

    /// FOUNDATION-B closure parity: a reconciled reflexive `rdf:rest` closure seeded at `cell0` over
    /// `present` EQUALS the materialized closure on the rolled-up base ({cell0,cell1,cell2,cell3,nil}).
    fn assert_reconciled_closure_parity() {
        let stack = closure_delta_stack();
        let lref = arc_refs(&stack);
        let head_root = lref.last().unwrap().state.commitment_root;
        let rest = rdf_rest_iri();

        let materialized = materialized_closure_set(&closure_rolled_up(), &rest, &iri("cell0"));
        assert_eq!(
            materialized,
            vec![iri("cell0"), iri("cell1"), iri("cell2"), iri("cell3"), rdf_nil_iri()]
                .into_iter()
                .collect::<std::collections::BTreeSet<String>>()
        );

        let (proof, prover_ids) = super::reconciled::prove_reconciled_transitive_closure(
            &lref,
            rest.as_bytes(),
            iri("cell0").as_bytes(),
            true,
        )
        .unwrap();
        let verifier_ids = super::reconciled::verify_reconciled_transitive_closure(
            &lref,
            head_root,
            rest.as_bytes(),
            iri("cell0").as_bytes(),
            true,
            &proof,
        )
        .unwrap();
        assert_eq!(
            prover_ids.iter().copied().collect::<std::collections::BTreeSet<_>>(),
            verifier_ids.iter().copied().collect::<std::collections::BTreeSet<_>>(),
            "prover and verifier must agree on the reachable id set"
        );

        let views: Vec<crate::proof::VerifierCommitmentRef<'_>> =
            lref.iter().map(|l| l.verifier_view()).collect();
        let reconciled: std::collections::BTreeSet<String> =
            verifier_ids.iter().map(|id| id_to_iri(&views, *id)).collect();
        assert_eq!(reconciled, materialized, "reconciled closure must equal the materialized closure");
    }

    /// FOUNDATION-B soundness: a wrong head root, a tampered `present` edge column, a forged closure
    /// round, and a truncated (non-converged) closure each fail verify closed.
    fn assert_reconciled_closure_soundness() {
        let stack = closure_delta_stack();
        let lref = arc_refs(&stack);
        let head_root = lref.last().unwrap().state.commitment_root;
        let rest = rdf_rest_iri();
        let seed = iri("cell0");

        let (proof, _) = super::reconciled::prove_reconciled_transitive_closure(
            &lref,
            rest.as_bytes(),
            seed.as_bytes(),
            true,
        )
        .unwrap();
        // Sanity: the honest proof verifies.
        super::reconciled::verify_reconciled_transitive_closure(
            &lref, head_root, rest.as_bytes(), seed.as_bytes(), true, &proof,
        )
        .unwrap();

        // (i) Wrong / swapped head root — the reconciliation anchor rejects.
        let mut bogus = head_root;
        bogus[0] ^= 0xff;
        assert!(super::reconciled::verify_reconciled_transitive_closure(
            &lref, bogus, rest.as_bytes(), seed.as_bytes(), true, &proof,
        )
        .is_err());
        assert!(super::reconciled::verify_reconciled_transitive_closure(
            &lref, lref[0].state.commitment_root, rest.as_bytes(), seed.as_bytes(), true, &proof,
        )
        .is_err());

        // (ii) Tampered `present` edge column: swap the kept subject/object commitments so the edge
        //      leaf's compaction no longer opens them — the FOUNDATION-A `present_bgp` verify rejects.
        let mut tampered_edge = proof.clone();
        tampered_edge.edge_leaf.atoms[0].com_kept.swap(0, 1);
        assert!(super::reconciled::verify_reconciled_transitive_closure(
            &lref, head_root, rest.as_bytes(), seed.as_bytes(), true, &tampered_edge,
        )
        .is_err());

        // (iii) A spurious node forged into the final closure round — the round's Distinct exact-cover
        //       + monotonicity check rejects (the ghost never came from an authenticated InnerJoin).
        let mut forged_round = proof.clone();
        assert!(!forged_round.rounds.is_empty(), "chain should need >= 2 rounds");
        forged_round.rounds.last_mut().unwrap().next_rows.push(999_999);
        assert!(super::reconciled::verify_reconciled_transitive_closure(
            &lref, head_root, rest.as_bytes(), seed.as_bytes(), true, &forged_round,
        )
        .is_err());

        // (iv) A truncated (non-converged) closure: drop the last round so the delivered accumulator
        //      is A_{K-1} while the fixpoint was proven over A_K — the fixpoint anti-join rejects.
        let mut truncated = proof.clone();
        truncated.rounds.pop();
        assert!(super::reconciled::verify_reconciled_transitive_closure(
            &lref, head_root, rest.as_bytes(), seed.as_bytes(), true, &truncated,
        )
        .is_err());
    }

    #[test]
    fn reconciled_edge_relation_parity() {
        big_stack(assert_reconciled_edge_parity);
    }

    #[test]
    fn reconciled_transitive_closure_parity() {
        big_stack(assert_reconciled_closure_parity);
    }

    #[test]
    fn reconciled_transitive_closure_soundness() {
        big_stack(assert_reconciled_closure_soundness);
    }

    // -----------------------------------------------------------------------
    // Reconciled ordered rdf:List walk over a delta head's `present` (ReadDocument 6c-ii).
    // -----------------------------------------------------------------------

    /// A base + 1 delta stack whose reconciled `present` is the list `doc -tracks-> [a, b, a]`
    /// (cons chain cell0 -> cell1 -> cell2 -> nil; first edges a/b/a). The base stores a 4-element
    /// list `[a, b, x, y]` (cell2/cell3 tail); the delta REMOVES cell2/cell3's old first/rest (4
    /// removals) and ADDS `cell2 -first-> a`, `cell2 -rest-> nil` (2 additions) — both committed
    /// groups have >= 2 records, satisfying the reconcile floor.
    fn list_delta_stack() -> Vec<std::sync::Arc<LayerCommitment>> {
        let rest = rdf_rest();
        let first = rdf_first();
        let nil = rdf_nil();
        let doc = iri("doc");
        let tracks = iri("tracks");
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for (s, p, o) in [
            (doc.clone(), tracks.clone(), cell(0)),
            (cell(0), first.clone(), iri("a")),
            (cell(0), rest.clone(), cell(1)),
            (cell(1), first.clone(), iri("b")),
            (cell(1), rest.clone(), cell(2)),
            (cell(2), first.clone(), iri("x")),
            (cell(2), rest.clone(), cell(3)),
            (cell(3), first.clone(), iri("y")),
            (cell(3), rest.clone(), nil.clone()),
        ] {
            builder.add_value_triple(ValueTriple::new_node(&s, &p, &o)).unwrap();
        }
        let base = builder.commit().unwrap();
        let base_lc = base.proof_commitment().unwrap();

        let child = base.open_write().unwrap();
        child.remove_value_triple(ValueTriple::new_node(&cell(2), &first, &iri("x"))).unwrap();
        child.remove_value_triple(ValueTriple::new_node(&cell(2), &rest, &cell(3))).unwrap();
        child.remove_value_triple(ValueTriple::new_node(&cell(3), &first, &iri("y"))).unwrap();
        child.remove_value_triple(ValueTriple::new_node(&cell(3), &rest, &nil)).unwrap();
        child.add_value_triple(ValueTriple::new_node(&cell(2), &first, &iri("a"))).unwrap();
        child.add_value_triple(ValueTriple::new_node(&cell(2), &rest, &nil)).unwrap();
        let head = child.commit().unwrap();
        vec![base_lc, head.proof_commitment().unwrap()]
    }

    fn object_iris(objs: &[CanonicalObject]) -> Vec<String> {
        objs.iter()
            .map(|o| match o {
                CanonicalObject::Node(b) => String::from_utf8(b.clone()).unwrap(),
                other => panic!("unexpected list element object {:?}", other),
            })
            .collect()
    }

    /// 6c-ii PARITY: the reconciled ordered-list walk over the 2-layer delta stack delivers the EXACT
    /// ordered elements `[a, b, a]` (order + duplicate multiplicity) AND equals the materialized
    /// `OrderedListWalk` on a rolled-up base.
    fn assert_reconciled_list_walk_parity() {
        let stack = list_delta_stack();
        let lref = arc_refs(&stack);
        let head_root = lref.last().unwrap().state.commitment_root;
        let subj = CanonicalObject::Node(iri("doc").into_bytes());
        let pred = iri("tracks");

        let (proof, prover_elements) =
            super::list_walk::prove_ordered_list_walk_reconciled(&lref, &subj, pred.as_bytes())
                .unwrap();
        let verifier_elements = super::list_walk::verify_ordered_list_walk_reconciled(
            &lref, head_root, &subj, pred.as_bytes(), &proof,
        )
        .unwrap();

        assert_eq!(object_iris(&prover_elements), object_iris(&verifier_elements));
        assert_eq!(
            object_iris(&verifier_elements),
            vec![iri("a"), iri("b"), iri("a")],
            "reconciled list walk must deliver the exact ordered [a, b, a]"
        );

        // PARITY with the materialized OrderedListWalk on a rolled-up single base layer.
        let rolled = list_layer(chain_triples(&iri("doc"), &iri("tracks"), &["a", "b", "a"]));
        let mat_plan = list_walk_plan(&iri("doc"), &iri("tracks"), 42);
        let mat_env = RecursiveRelationEnvelope::prove(&rolled, &mat_plan).unwrap();
        mat_env.verify(&rolled, rolled.state.commitment_root, &mat_plan).unwrap();
        assert_eq!(
            object_iris(&verifier_elements),
            walk_elements(&mat_env),
            "reconciled element column must equal the materialized OrderedListWalk"
        );
    }

    /// 6c-ii SOUNDNESS over `present`: each list forgery (reorder / wrong head / dropped / injected /
    /// gap) fails verify at the SAME gadget as the materialized walk, plus a statement substitution.
    fn assert_reconciled_list_walk_soundness() {
        let stack = list_delta_stack();
        let lref = arc_refs(&stack);
        let head_root = lref.last().unwrap().state.commitment_root;
        let subj = CanonicalObject::Node(iri("doc").into_bytes());
        let pred = iri("tracks");
        let pbytes = pred.as_bytes();

        // Sanity: the honest reconciled walk verifies.
        let (proof, _) =
            super::list_walk::prove_ordered_list_walk_reconciled(&lref, &subj, pbytes).unwrap();
        super::list_walk::verify_ordered_list_walk_reconciled(
            &lref, head_root, &subj, pbytes, &proof,
        )
        .unwrap();

        let views: Vec<crate::proof::VerifierCommitmentRef<'_>> =
            lref.iter().map(|l| l.verifier_view()).collect();
        let c0 = super::reconciled::resolve_combined_node_id(&views, cell(0).as_bytes());
        let c1 = super::reconciled::resolve_combined_node_id(&views, cell(1).as_bytes());
        let c2 = super::reconciled::resolve_combined_node_id(&views, cell(2).as_bytes());
        let a_id = super::reconciled::resolve_combined_node_id(&views, iri("a").as_bytes());

        // Prove with a caller-chosen chain order, then verify — mirrors the materialized forging hook.
        let forge = |order: Vec<u64>| {
            super::list_walk::prove_ordered_list_walk_reconciled_forged(&lref, &subj, pbytes, order)
                .and_then(|(proof, _)| {
                    super::list_walk::verify_ordered_list_walk_reconciled(
                        &lref, head_root, &subj, pbytes, &proof,
                    )
                })
        };

        // (1) TAIL REORDER [c0, c2, c1] -> rejected by the SUCCESSOR (C[0]==head still passes).
        assert!(forge(vec![c0, c2, c1]).is_err(), "reordered chain must be rejected (successor)");

        // (2) WRONG C[0] [c2, c1, c0] -> rejected by the explicit C[0]==head opening (a genuine
        //     permutation of the cons set that starts at a non-head cell).
        let wrong_head_err = forge(vec![c2, c1, c0]).expect_err("non-head first cons cell must reject");
        assert!(
            wrong_head_err.to_string().contains("C[0] != head"),
            "the C[0]==head opening must be the rejecting guard, got: {wrong_head_err}"
        );

        // (3) DROPPED middle cell [c0, c2] -> rejected by the PERMUTATION completeness.
        assert!(forge(vec![c0, c2]).is_err(), "dropped cell must be rejected (permutation)");

        // (4) INJECTED non-cons node [c0, c1, a_id, c2] -> rejected by the PERMUTATION membership.
        assert!(
            forge(vec![c0, c1, a_id, c2]).is_err(),
            "injected non-cons node must be rejected (permutation)"
        );

        // (5) GAP / APPENDED duplicate [c0, c1, c2, c0] -> rejected by the PERMUTATION.
        assert!(forge(vec![c0, c1, c2, c0]).is_err(), "appended/gap cell must be rejected (permutation)");

        // (6) STATEMENT substitution: verify the honest proof against a DIFFERENT predicate — the
        //     anchor leaf reads a different `present` relation and the gluing digest changes, so the
        //     re-aimed proof fails closed.
        assert!(
            super::list_walk::verify_ordered_list_walk_reconciled(
                &lref, head_root, &subj, iri("authors").as_bytes(), &proof,
            )
            .is_err(),
            "wrong predicate must be rejected (statement binding)"
        );
    }

    #[test]
    fn reconciled_list_walk_parity() {
        big_stack(assert_reconciled_list_walk_parity);
    }

    #[test]
    fn reconciled_list_walk_soundness_negatives() {
        big_stack(assert_reconciled_list_walk_soundness);
    }

    // -----------------------------------------------------------------------
    // Reconciled ordered sys:Array walk over a delta head's `present` (ReadDocument 6d).
    // -----------------------------------------------------------------------

    /// Add one raw array triple (a plain node edge, or a `sys:index` `UInt64` literal) to a builder.
    fn add_arr(builder: &crate::store::sync::SyncStoreLayerBuilder, t: &ArrTriple) {
        match t {
            ArrTriple::Node(s, p, o) => {
                builder.add_value_triple(ValueTriple::new_node(s.as_str(), p.as_str(), o.as_str())).unwrap();
            }
            ArrTriple::Index(cell, index) => {
                builder
                    .add_value_triple(ValueTriple::new_value(cell.as_str(), &sys_index(), u64::make_entry(index)))
                    .unwrap();
            }
        }
    }

    /// Remove one raw array triple from a write child (the inverse of [`add_arr`]).
    fn remove_arr(builder: &crate::store::sync::SyncStoreLayerBuilder, t: &ArrTriple) {
        match t {
            ArrTriple::Node(s, p, o) => {
                builder.remove_value_triple(ValueTriple::new_node(s.as_str(), p.as_str(), o.as_str())).unwrap();
            }
            ArrTriple::Index(cell, index) => {
                builder
                    .remove_value_triple(ValueTriple::new_value(cell.as_str(), &sys_index(), u64::make_entry(index)))
                    .unwrap();
            }
        }
    }

    /// Build a base + 1 delta stack: `base` triples committed as the base layer, then `removals`
    /// removed + `additions` added in one delta layer (both committed groups must have >= 2 records to
    /// satisfy the reconcile floor). The reconciled `present` == `(base - removals) + additions`.
    fn build_delta_stack(
        base: Vec<ArrTriple>,
        removals: Vec<ArrTriple>,
        additions: Vec<ArrTriple>,
    ) -> Vec<std::sync::Arc<LayerCommitment>> {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for t in &base {
            add_arr(&builder, t);
        }
        let base_layer = builder.commit().unwrap();
        let base_lc = base_layer.proof_commitment().unwrap();

        let child = base_layer.open_write().unwrap();
        for t in &removals {
            remove_arr(&child, t);
        }
        for t in &additions {
            add_arr(&child, t);
        }
        let head = child.commit().unwrap();
        vec![base_lc, head.proof_commitment().unwrap()]
    }

    /// The base triples of a dense/malformed array `subject -predicate-> cells`, with the LAST cell's
    /// `sys:index`/`sys:value` deliberately WRONG (index 999 / value `__wrong__`) so a 2-removal +
    /// 2-addition delta can correct exactly it — a valid delta stack whose `present` is the target array.
    fn array_base_wrong_last(subject: &str, predicate: &str, cells: &[(u64, &str)]) -> Vec<ArrTriple> {
        let last = cells.len() - 1;
        let mut base = Vec::new();
        for (k, (index, value)) in cells.iter().enumerate() {
            let c = cell(k);
            base.push(ArrTriple::Node(subject.to_string(), predicate.to_string(), c.clone()));
            base.push(ArrTriple::Node(c.clone(), rdf_type(), sys_array()));
            if k == last {
                base.push(ArrTriple::Index(c.clone(), 999));
                base.push(ArrTriple::Node(c.clone(), sys_value(), iri("__wrong__")));
            } else {
                base.push(ArrTriple::Index(c.clone(), *index));
                base.push(ArrTriple::Node(c.clone(), sys_value(), iri(value)));
            }
        }
        base
    }

    /// The 2-removal / 2-addition delta that corrects the last cell of [`array_base_wrong_last`].
    fn array_last_cell_delta(cells: &[(u64, &str)]) -> (Vec<ArrTriple>, Vec<ArrTriple>) {
        let last = cells.len() - 1;
        let c = cell(last);
        let (li, lv) = cells[last];
        let removals = vec![
            ArrTriple::Index(c.clone(), 999),
            ArrTriple::Node(c.clone(), sys_value(), iri("__wrong__")),
        ];
        let additions = vec![
            ArrTriple::Index(c.clone(), li),
            ArrTriple::Node(c.clone(), sys_value(), iri(lv)),
        ];
        (removals, additions)
    }

    /// A base + 1 delta stack whose reconciled `present` is the dense/malformed array
    /// `subject -predicate-> cells`, respecting the reconcile >= 2-records-per-group floor.
    fn array_delta_stack(
        subject: &str,
        predicate: &str,
        cells: &[(u64, &str)],
    ) -> Vec<std::sync::Arc<LayerCommitment>> {
        let base = array_base_wrong_last(subject, predicate, cells);
        let (removals, additions) = array_last_cell_delta(cells);
        build_delta_stack(base, removals, additions)
    }

    /// 6d PARITY: the reconciled ordered-array walk over the 2-layer delta stack delivers the EXACT
    /// `sys:index`-ordered `sys:value` sequence AND equals the materialized `OrderedArrayWalk` on a
    /// rolled-up base — for a plain array `[x, y, z]` and a duplicate-value array `[a, b, a]`.
    fn assert_reconciled_array_walk_parity() {
        let p = iri("items");
        for (doc, cells, expected) in [
            ("adoc1", vec![(0u64, "x"), (1, "y"), (2, "z")], vec!["x", "y", "z"]),
            ("adoc2", vec![(0, "a"), (1, "b"), (2, "a")], vec!["a", "b", "a"]),
        ] {
            let s = iri(doc);
            let stack = array_delta_stack(&s, &p, &cells);
            let lref = arc_refs(&stack);
            let head_root = lref.last().unwrap().state.commitment_root;
            let subj = CanonicalObject::Node(s.clone().into_bytes());

            let (proof, prover_elements) =
                super::array_walk::prove_ordered_array_walk_reconciled(&lref, &subj, p.as_bytes())
                    .unwrap();
            let verifier_elements = super::array_walk::verify_ordered_array_walk_reconciled(
                &lref, head_root, &subj, p.as_bytes(), &proof,
            )
            .unwrap();

            assert_eq!(object_iris(&prover_elements), object_iris(&verifier_elements));
            let expected: Vec<String> = expected.iter().map(|v| iri(v)).collect();
            assert_eq!(
                object_iris(&verifier_elements),
                expected,
                "reconciled array walk must deliver the exact sys:index-ordered values"
            );

            // PARITY with the materialized OrderedArrayWalk on a rolled-up single base layer.
            let rolled = array_layer(dense_array_triples(&s, &p, &cells));
            let mat_plan = array_walk_plan(&s, &p, 88);
            let mat_env = RecursiveRelationEnvelope::prove(&rolled, &mat_plan).unwrap();
            mat_env.verify(&rolled, rolled.state.commitment_root, &mat_plan).unwrap();
            assert_eq!(
                object_iris(&verifier_elements),
                walk_values(&mat_env),
                "reconciled value column must equal the materialized OrderedArrayWalk"
            );
        }
    }

    /// 6d SOUNDNESS over `present`: each array forgery (reorder / drop / dup-index / inject / gap+
    /// index>=n / two-values) fails at the SAME gadget as the materialized walk, plus a statement
    /// substitution. The reorder/drop cases use the caller-chosen density-permutation forging hook; the
    /// malformed-array cases reject at HONEST prove.
    fn assert_reconciled_array_walk_soundness() {
        let p = iri("items");

        // Sanity: the honest reconciled walk over a dense delta-stack array verifies.
        let s = iri("asnd");
        let subj = CanonicalObject::Node(s.clone().into_bytes());
        let stack = array_delta_stack(&s, &p, &[(0, "a"), (1, "b"), (2, "c")]);
        let lref = arc_refs(&stack);
        let head_root = lref.last().unwrap().state.commitment_root;
        let (proof, _) =
            super::array_walk::prove_ordered_array_walk_reconciled(&lref, &subj, p.as_bytes()).unwrap();
        super::array_walk::verify_ordered_array_walk_reconciled(
            &lref, head_root, &subj, p.as_bytes(), &proof,
        )
        .unwrap();

        // (1) REORDER (density permutation-to-identity). A 2-element array has exactly two density-
        // permutation orders; EXACTLY ONE is the honest one (routed index == identity [0,1]) and
        // verifies, the OTHER reorders the delivery so the routed index != identity and rejects. This
        // holds regardless of which physical order `present` delivers the cells in.
        let s2 = iri("aorder");
        let subj2 = CanonicalObject::Node(s2.clone().into_bytes());
        let stack2 = array_delta_stack(&s2, &p, &[(0, "m"), (1, "n")]);
        let lref2 = arc_refs(&stack2);
        let hr2 = lref2.last().unwrap().state.commitment_root;
        let verifies = |order: Vec<usize>| {
            super::array_walk::prove_ordered_array_walk_reconciled_forged(&lref2, &subj2, p.as_bytes(), order)
                .and_then(|(proof, _)| {
                    super::array_walk::verify_ordered_array_walk_reconciled(
                        &lref2, hr2, &subj2, p.as_bytes(), &proof,
                    )
                })
                .is_ok()
        };
        let ok_count = [vec![0usize, 1], vec![1usize, 0]].into_iter().filter(|o| verifies(o.to_vec())).count();
        assert_eq!(
            ok_count, 1,
            "exactly one density order verifies; a reordered delivery must be rejected (density permutation-to-identity)"
        );

        // (2) DROPPED cell -> a length-(n-1) density routing is not a full bijection over [0,n) (prove
        // rejects at the permutation gadget).
        assert!(
            super::array_walk::prove_ordered_array_walk_reconciled_forged(
                &lref2, &subj2, p.as_bytes(), vec![0],
            )
            .is_err(),
            "a dropped cell cannot be routed by the density bijection"
        );

        // (3) DUPLICATE index (two cells index 0, none index 1) -> honest prove rejects (density: not a
        // dense 0..n-1 permutation).
        let sdup = iri("adup");
        let dsub = CanonicalObject::Node(sdup.clone().into_bytes());
        let dstack = array_delta_stack(&sdup, &p, &[(0, "a"), (1, "b"), (0, "c")]);
        assert!(
            super::array_walk::prove_ordered_array_walk_reconciled(&arc_refs(&dstack), &dsub, p.as_bytes())
                .is_err(),
            "a duplicate sys:index over present must be rejected"
        );

        // (4) INJECTED extra P-object that is NOT a well-formed array cell (no sys:index) -> honest
        // prove rejects (the index functional bind's count no longer equals the cell count).
        let sinj = iri("ainj");
        let isub = CanonicalObject::Node(sinj.clone().into_bytes());
        let cells = [(0u64, "a"), (1, "b"), (2, "c")];
        let mut base = array_base_wrong_last(&sinj, &p, &cells);
        let inj = cell(99);
        base.push(ArrTriple::Node(sinj.clone(), p.clone(), inj.clone()));
        base.push(ArrTriple::Node(inj.clone(), rdf_type(), sys_array()));
        base.push(ArrTriple::Node(inj.clone(), sys_value(), iri("w")));
        let (rem, add) = array_last_cell_delta(&cells);
        let istack = build_delta_stack(base, rem, add);
        assert!(
            super::array_walk::prove_ordered_array_walk_reconciled(&arc_refs(&istack), &isub, p.as_bytes())
                .is_err(),
            "an injected non-cell P-object over present must be rejected (index functional bind)"
        );

        // (5) GAP / index >= n (`[0, 2]`, n=2 with index 2 >= n) -> honest prove rejects (out-of-range
        // index; the reference null-fills a gap, so a sparse array is not a dense one).
        let sgap = iri("agap");
        let gsub = CanonicalObject::Node(sgap.clone().into_bytes());
        let gstack = array_delta_stack(&sgap, &p, &[(0, "a"), (2, "b")]);
        assert!(
            super::array_walk::prove_ordered_array_walk_reconciled(&arc_refs(&gstack), &gsub, p.as_bytes())
                .is_err(),
            "a sparse/gapped array (index >= n) over present must be rejected"
        );

        // (6) A cell with TWO sys:value -> honest prove rejects (the value functional-bind count).
        let stv = iri("atwo");
        let tsub = CanonicalObject::Node(stv.clone().into_bytes());
        let mut base = array_base_wrong_last(&stv, &p, &cells);
        base.push(ArrTriple::Node(cell(1), sys_value(), iri("bb")));
        let (rem, add) = array_last_cell_delta(&cells);
        let tstack = build_delta_stack(base, rem, add);
        assert!(
            super::array_walk::prove_ordered_array_walk_reconciled(&arc_refs(&tstack), &tsub, p.as_bytes())
                .is_err(),
            "a cell with two sys:value over present must be rejected (value functional bind)"
        );

        // (7) STATEMENT substitution: verify the honest proof against a DIFFERENT predicate — the leaves
        // read a different `present` relation and the gluing digest changes, so the re-aimed proof fails.
        assert!(
            super::array_walk::verify_ordered_array_walk_reconciled(
                &lref, head_root, &subj, iri("authors").as_bytes(), &proof,
            )
            .is_err(),
            "wrong predicate must be rejected (statement binding)"
        );
    }

    #[test]
    fn reconciled_array_walk_parity() {
        big_stack(assert_reconciled_array_walk_parity);
    }

    #[test]
    fn reconciled_array_walk_soundness_negatives() {
        big_stack(assert_reconciled_array_walk_soundness);
    }
