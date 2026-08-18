//! Unit tests for the commitment machinery (round-trips, root binding, catalog validation,
//! and the order-preserving value/date encodings). Exercised throughout the `proof::` suites.

use super::*;

fn record(s: u64, p: u64, o: u64, subj: &str, pred: &str, obj: &str) -> ProofRecord {
    ProofRecord {
        subject_id: s,
        predicate_id: p,
        object_id: o,
        triple: CanonicalTriple::node(subj.as_bytes(), pred.as_bytes(), obj.as_bytes()),
    }
}

fn sample() -> LayerCommitment {
    let additions = vec![record(1, 1, 2, "a", "p", "b")];
    LayerCommitment::from_native_delta(
        None,
        additions.clone(),
        vec![],
        additions,
        0,
        0,
        2,
        1,
        vec![
            (CanonicalObject::Node(b"a".to_vec()), 1),
            (CanonicalObject::Node(b"b".to_vec()), 2),
        ],
        vec![(b"p".to_vec(), 1)],
    )
    .unwrap()
}

#[test]
fn float_range_catalog_rejects_nan_semantic_mismatch() {
    let encoded_nan = (f64::NAN.to_bits() ^ 0x8000_0000_0000_0000)
        .to_be_bytes()
        .to_vec();
    let catalog = vec![ProofRecord {
        subject_id: 0,
        predicate_id: 0,
        object_id: 1,
        triple: CanonicalTriple {
            subject: Vec::new(),
            predicate: Vec::new(),
            object: CanonicalObject::Value {
                datatype: CanonicalDatatype::Float64,
                lexical: encoded_nan,
            },
        },
    }];
    assert!(reject_nan_catalog(
        &catalog,
        ArkFr::from_u64(CanonicalDatatype::Float64 as u64),
    )
    .is_err());
    assert!(reject_nan_catalog(
        &catalog,
        ArkFr::from_u64(CanonicalDatatype::UInt64 as u64),
    )
    .is_ok());
}

#[test]
fn compact_verifier_commitment_roundtrips_and_omits_witness_tables() {
    let layer = sample();
    let full = layer.encode().unwrap();
    let artifact = layer.verifier_commitment().unwrap();
    let compact = artifact.encode().unwrap();
    let decoded = VerifierCommitment::decode(&compact).unwrap();

    assert_eq!(decoded.state, layer.state);
    assert_eq!(decoded.predicate_catalog, layer.predicate_catalog);
    assert_eq!(decoded.node_value_catalog, layer.node_value_catalog);
    assert!(compact.len() < full.len());
    assert_eq!(decoded.recomputed_commitment_root().unwrap(), layer.state.commitment_root);
}

#[test]
fn compact_verifier_commitment_rejects_root_catalog_commitment_and_count_tampering() {
    let artifact = sample().verifier_commitment().unwrap();

    let mut wrong_root = artifact.clone();
    wrong_root.state.commitment_root[0] ^= 1;
    assert!(wrong_root.validate().is_err());

    let mut wrong_catalog = artifact.clone();
    wrong_catalog.node_value_catalog[0].triple.object =
        CanonicalObject::Node(b"forged".to_vec());
    assert!(wrong_catalog.validate().is_err());

    let mut wrong_commitment = artifact.clone();
    wrong_commitment.materialized_column_commitments[0] =
        <ArkGT as Group>::identity();
    assert!(wrong_commitment.validate().is_err());

    let mut wrong_count = artifact.clone();
    wrong_count.state.node_value_count += 1;
    assert!(wrong_count.validate().is_err());
}

#[test]
fn compact_verifier_commitment_decode_rejects_checksum_and_version_tampering() {
    let mut bytes = sample().verifier_commitment().unwrap().encode().unwrap();
    bytes[VERIFIER_COMMITMENT_MAGIC.len()] ^= 1;
    assert!(VerifierCommitment::decode(&bytes).is_err());

    let mut bytes = sample().verifier_commitment().unwrap().encode().unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    assert!(VerifierCommitment::decode(&bytes).is_err());
}

// Same two-node/one-predicate dictionary {a:1, b:2}, {p:1}, but the caller
// chooses whether the triple lands in additions and/or removals.
fn delta(additions: Vec<ProofRecord>, removals: Vec<ProofRecord>) -> LayerCommitment {
    let materialized = additions.clone();
    LayerCommitment::from_native_delta(
        None,
        additions,
        removals,
        materialized,
        0,
        0,
        2,
        1,
        vec![
            (CanonicalObject::Node(b"a".to_vec()), 1),
            (CanonicalObject::Node(b"b".to_vec()), 2),
        ],
        vec![(b"p".to_vec(), 1)],
    )
    .unwrap()
}

#[test]
fn encode_decode_roundtrip_preserves_state_and_columns() {
    let commitment = sample();
    let bytes = commitment.encode().unwrap();
    assert_eq!(bytes.len(), commitment.encoded_len().unwrap());
    let decoded = LayerCommitment::decode(&bytes).unwrap();
    assert_eq!(decoded.state, commitment.state);
    assert_eq!(decoded.additions, commitment.additions);
    assert_eq!(decoded.node_value_catalog, commitment.node_value_catalog);
    assert_eq!(decoded.column_commitments, commitment.column_commitments);
    assert_eq!(
        decoded.predicate_dict_commitments,
        commitment.predicate_dict_commitments
    );
    assert_eq!(
        decoded.node_value_dict_commitments,
        commitment.node_value_dict_commitments
    );
    assert_eq!(
        decoded.node_value_dict_range_commitments,
        commitment.node_value_dict_range_commitments
    );
    assert_eq!(
        decoded.state.commitment_root,
        commitment.state.commitment_root
    );
}

// A base layer with three predicates p1<p2<p3 (dict of size 3, padded to 4)
// over distinct subjects/objects.
fn multi_predicate_layer() -> LayerCommitment {
    let additions = vec![
        record(1, 1, 2, "a", "p1", "b"),
        record(3, 2, 4, "c", "p2", "d"),
        record(5, 3, 6, "e", "p3", "f"),
    ];
    LayerCommitment::from_native_delta(
        None,
        additions.clone(),
        vec![],
        additions,
        0,
        0,
        6,
        3,
        vec![
            (CanonicalObject::Node(b"a".to_vec()), 1),
            (CanonicalObject::Node(b"b".to_vec()), 2),
            (CanonicalObject::Node(b"c".to_vec()), 3),
            (CanonicalObject::Node(b"d".to_vec()), 4),
            (CanonicalObject::Node(b"e".to_vec()), 5),
            (CanonicalObject::Node(b"f".to_vec()), 6),
        ],
        vec![
            (b"p1".to_vec(), 1),
            (b"p2".to_vec(), 2),
            (b"p3".to_vec(), 3),
        ],
    )
    .unwrap()
}

#[test]
fn predicate_dictionary_commitments_align_with_shared_commit_column() {
    use crate::proof::argument::commit_column;
    use crate::proof::dict::hash_term;

    let layer = multi_predicate_layer();
    assert_eq!(layer.predicate_catalog.len(), 3);
    // Rebuild the two padded dict columns exactly as the query side would,
    // and confirm they match the stored commitments byte-for-byte.
    let zero = <ArkFr as DoryField>::zero();
    let mut hashes = vec![zero; 4];
    let mut ids = vec![zero; 4];
    for (k, record) in layer.predicate_catalog.iter().enumerate() {
        hashes[k] = hash_term(&record.triple.predicate);
        ids[k] = <ArkFr as DoryField>::from_u64(record.predicate_id);
    }
    assert_eq!(
        commit_column(&hashes).unwrap(),
        layer.predicate_dict_commitments.0
    );
    assert_eq!(
        commit_column(&ids).unwrap(),
        layer.predicate_dict_commitments.1
    );
}

#[test]
fn commitment_root_binds_the_predicate_dictionary() {
    // Determinism: the same layer (incl. the dict commitments) yields the
    // same root across two independent builds.
    assert_eq!(
        multi_predicate_layer().state.commitment_root,
        multi_predicate_layer().state.commitment_root
    );
    // Changing a predicate TERM (same ids) changes the dict-hash commitment,
    // hence the root.
    let changed_additions = vec![
        record(1, 1, 2, "a", "pX", "b"),
        record(3, 2, 4, "c", "p2", "d"),
        record(5, 3, 6, "e", "p3", "f"),
    ];
    let changed = LayerCommitment::from_native_delta(
        None,
        changed_additions.clone(),
        vec![],
        changed_additions,
        0,
        0,
        6,
        3,
        vec![
            (CanonicalObject::Node(b"a".to_vec()), 1),
            (CanonicalObject::Node(b"b".to_vec()), 2),
            (CanonicalObject::Node(b"c".to_vec()), 3),
            (CanonicalObject::Node(b"d".to_vec()), 4),
            (CanonicalObject::Node(b"e".to_vec()), 5),
            (CanonicalObject::Node(b"f".to_vec()), 6),
        ],
        vec![
            (b"pX".to_vec(), 1),
            (b"p2".to_vec(), 2),
            (b"p3".to_vec(), 3),
        ],
    )
    .unwrap();
    assert_ne!(
        multi_predicate_layer().state.commitment_root,
        changed.state.commitment_root
    );
}

// A base layer with a NODE object (b, id 2) and a typed VALUE literal
// object ("lit"^^string, id 3), both under subject a (id 1) / predicate p
// (id 1). Exercises both `CanonicalObject` variants in the node/value dict.
fn node_and_value_layer() -> LayerCommitment {
    let value_triple = ProofRecord {
        subject_id: 1,
        predicate_id: 1,
        object_id: 3,
        triple: CanonicalTriple {
            subject: b"a".to_vec(),
            predicate: b"p".to_vec(),
            object: CanonicalObject::Value {
                datatype: CanonicalDatatype::String,
                lexical: b"lit".to_vec(),
            },
        },
    };
    let additions = vec![record(1, 1, 2, "a", "p", "b"), value_triple];
    LayerCommitment::from_native_delta(
        None,
        additions.clone(),
        vec![],
        additions,
        0,
        0,
        3,
        1,
        vec![
            (CanonicalObject::Node(b"a".to_vec()), 1),
            (CanonicalObject::Node(b"b".to_vec()), 2),
            (
                CanonicalObject::Value {
                    datatype: CanonicalDatatype::String,
                    lexical: b"lit".to_vec(),
                },
                3,
            ),
        ],
        vec![(b"p".to_vec(), 1)],
    )
    .unwrap()
}

#[test]
fn commitment_format_version_is_six() {
    // v6 == the sidecar carries the two id-indexed dictionary Merkle roots (I1); the
    // `commitment_root` domain/structure is unchanged (`commitment-root/v4`), so a
    // date-free layer still hashes to a BYTE-IDENTICAL root.
    assert_eq!(COMMITMENT_FORMAT_VERSION, 6);
    // v4 == the compact artifact gained the I2b S1 `terms_present` flag byte and made
    // the two clear catalogs optional (term-less variant). `commitment_root` unchanged.
    assert_eq!(VERIFIER_COMMITMENT_FORMAT_VERSION, 4);
}

#[test]
fn setup_tiers_cover_the_full_pf4_record_domain() {
    assert_eq!(MAX_DECODED_RECORDS, MAX_RECORDS * PERSISTED_TABLE_COUNT);
    assert_eq!(padded_domain_for_rows(1 << 20).unwrap(), (1 << 20, 20));
    assert_eq!(
        padded_domain_for_rows((1 << 20) + 1).unwrap(),
        (1 << 21, 21)
    );
    assert_eq!(setup_tier_for_num_vars(20).unwrap(), 20);
    assert_eq!(setup_tier_for_num_vars(21).unwrap(), 22);
    assert_eq!(setup_tier_for_num_vars(22).unwrap(), 22);
    assert_eq!(setup_tier_for_num_vars(23).unwrap(), 24);
    assert_eq!(setup_tier_for_num_vars(24).unwrap(), 24);
    assert!(setup_tier_for_num_vars(25)
        .unwrap_err()
        .to_string()
        .contains("exceeds PF4's maximum"));

    let (padded, log) = padded_domain_for_rows(MAX_RECORDS).unwrap();
    assert_eq!(padded, 1 << SETUP_MAX_LOG_N);
    assert_eq!(log, SETUP_MAX_LOG_N);
    assert!(padded_domain_for_rows(MAX_RECORDS + 1)
        .unwrap_err()
        .to_string()
        .contains("above the PF4 limit"));
}

#[test]
fn setup_memory_and_commitment_scratch_are_preflighted() {
    assert_eq!(setup_generator_count(20).unwrap(), 1 << 10);
    assert_eq!(setup_generator_count(22).unwrap(), 1 << 11);
    assert_eq!(setup_generator_count(24).unwrap(), 1 << 12);
    let retained = setup_retained_bytes(SETUP_MAX_LOG_N).unwrap();
    assert!(retained <= MAX_SETUP_RETAINED_BYTES);
    preflight_setup_tier(SETUP_MAX_LOG_N).unwrap();

    let former_boundary = construction_commitment_scratch_bytes(&[(1 << 20) + 1]).unwrap();
    assert!(former_boundary > (1 << 21) * std::mem::size_of::<ArkFr>());
    assert!(former_boundary < MAX_DECODED_PROOF_BYTES);
    let structural_max = construction_commitment_scratch_bytes(&[MAX_RECORDS]).unwrap();
    assert!(structural_max > MAX_DECODED_PROOF_BYTES);
}

#[test]
fn setup_identity_is_deterministic_and_legacy_tier_is_prefix_compatible() {
    use ark_ec::CurveGroup;

    assert_eq!(
        setup_parameter_id(19).unwrap(),
        setup_parameter_id(20).unwrap()
    );
    assert_ne!(
        setup_parameter_id(20).unwrap(),
        setup_parameter_id(21).unwrap()
    );
    assert_ne!(
        setup_parameter_id(22).unwrap(),
        setup_parameter_id(23).unwrap()
    );
    let selected = dory_prover_setup(20).unwrap();
    let independently_rebuilt = deterministic_prover_setup(20);
    assert_eq!(selected.g1_vec, independently_rebuilt.g1_vec);
    assert_eq!(selected.g2_vec, independently_rebuilt.g2_vec);
    assert!(selected
        .g1_vec
        .iter()
        .chain(std::iter::once(&selected.h1))
        .all(|point| point.0 != G1Projective::default()
            && point.0.into_affine().is_on_curve()
            && point
                .0
                .into_affine()
                .is_in_correct_subgroup_assuming_on_curve()));
    assert!(selected
        .g2_vec
        .iter()
        .chain(std::iter::once(&selected.h2))
        .all(|point| point.0 != G2Projective::default()
            && point.0.into_affine().is_on_curve()
            && point
                .0
                .into_affine()
                .is_in_correct_subgroup_assuming_on_curve()));

    let vector_hash = |g1: &ArkG1, g2: &ArkG2, h1: &ArkG1, h2: &ArkG2| {
        let mut bytes = Vec::new();
        g1.0.serialize_compressed(&mut bytes).unwrap();
        g2.0.serialize_compressed(&mut bytes).unwrap();
        h1.0.serialize_compressed(&mut bytes).unwrap();
        h2.0.serialize_compressed(&mut bytes).unwrap();
        hex::encode(Sha256::digest(bytes))
    };
    assert_eq!(
        vector_hash(
            &selected.g1_vec[0],
            &selected.g2_vec[0],
            &selected.h1,
            &selected.h2,
        ),
        "5dc71cb471c168203fe617d7e6fd66786cfce1ea50206c80aea5484079608b24"
    );
}

/// GOLDEN VECTOR: pins the FULL default-tier generator set (every `g1_vec`/`g2_vec` entry plus the
/// blinding `h1`/`h2`/`ht`) and the transcript-bound `setup_parameter_id(20)` that now folds a hash
/// of those generators. A silent sampler change — a dependency/`UniformRand` bump that remaps the
/// fixed seed to different curve points — flips both pins and fails CI, so two builds can never
/// agree on the transcript-bound parameter id while holding different generators.
#[test]
fn setup_parameter_id_binds_full_generator_set_golden() {
    let selected = dory_prover_setup(20).unwrap();
    let mut bytes = Vec::new();
    for g in &selected.g1_vec {
        g.0.serialize_compressed(&mut bytes).unwrap();
    }
    for g in &selected.g2_vec {
        g.0.serialize_compressed(&mut bytes).unwrap();
    }
    selected.h1.0.serialize_compressed(&mut bytes).unwrap();
    selected.h2.0.serialize_compressed(&mut bytes).unwrap();
    selected.ht.serialize_compressed(&mut bytes).unwrap();
    assert_eq!(
        hex::encode(Sha256::digest(&bytes)),
        "87825ccad5f975d727efc348149692f422c9c0092c80d397d9a26c2307f41ee0"
    );
    assert_eq!(
        hex::encode(setup_parameter_id(20).unwrap()),
        "acd27a9745a11e71701faac18fbde8a34e37af8ed7f63f6cc24efa24c6086613"
    );
}

#[test]
fn node_value_dictionary_commitments_align_with_shared_commit_column() {
    use crate::proof::argument::commit_column;
    use crate::proof::dict::hash_object;

    let layer = node_and_value_layer();
    assert_eq!(layer.node_value_catalog.len(), 3);
    // Rebuild the two padded dict columns exactly as the query side would
    // (via the public accessor), and confirm they match the stored
    // commitments byte-for-byte.
    let (hashes, ids) = layer.node_value_dictionary_columns().unwrap();
    assert_eq!(hashes.len(), 4); // 3 entries padded to a power of two
    for (k, record) in layer.node_value_catalog.iter().enumerate() {
        assert_eq!(hashes[k], hash_object(&record.triple.object));
        assert_eq!(ids[k], <ArkFr as DoryField>::from_u64(record.object_id));
    }
    assert_eq!(
        commit_column(&hashes).unwrap(),
        layer.node_value_dict_commitments.0
    );
    assert_eq!(
        commit_column(&ids).unwrap(),
        layer.node_value_dict_commitments.1
    );
}

#[test]
fn commitment_root_binds_the_node_value_dictionary() {
    use crate::proof::argument::commit_column;

    let layer = sample();
    // Baseline: the recomputed root equals the stored root.
    assert_eq!(
        layer.recomputed_commitment_root().unwrap(),
        layer.state.commitment_root
    );
    // Tamper ONE node/value dict commitment component with a distinct but
    // otherwise valid GT: the recomputed root must diverge from the stored
    // root (the node/value dict is now bound into the trust anchor).
    let other = commit_column(&[
        <ArkFr as DoryField>::from_u64(7),
        <ArkFr as DoryField>::from_u64(9),
    ])
    .unwrap();
    assert_ne!(other, layer.node_value_dict_commitments.0);
    let mut tampered = layer.clone();
    tampered.node_value_dict_commitments.0 = other;
    assert_ne!(
        tampered.recomputed_commitment_root().unwrap(),
        tampered.state.commitment_root
    );
}

#[test]
fn encode_decode_preserves_node_value_dictionary_and_root() {
    let commitment = node_and_value_layer();
    let bytes = commitment.encode().unwrap();
    assert_eq!(bytes.len(), commitment.encoded_len().unwrap());
    let decoded = LayerCommitment::decode(&bytes).unwrap();
    assert_eq!(
        decoded.node_value_dict_commitments,
        commitment.node_value_dict_commitments
    );
    assert_eq!(
        decoded.node_value_dict_range_commitments,
        commitment.node_value_dict_range_commitments
    );
    assert_eq!(
        decoded.predicate_dict_commitments,
        commitment.predicate_dict_commitments
    );
    // The recomputed root is stable across encode->decode and equals the
    // stored root on both sides (the new field survives the round trip).
    assert_eq!(
        decoded.recomputed_commitment_root().unwrap(),
        commitment.state.commitment_root
    );
    assert_eq!(
        decoded.recomputed_commitment_root().unwrap(),
        decoded.state.commitment_root
    );
}

#[test]
fn node_value_range_columns_align_with_shared_commit_column() {
    use crate::proof::argument::commit_column;

    let layer = node_and_value_layer();
    // Rebuild the two range columns exactly as the query side would (via the
    // public accessors) and confirm they match the stored commitments
    // byte-for-byte, and share the length of the hash/id columns.
    let (hashes, _ids) = layer.node_value_dictionary_columns().unwrap();
    let values = layer.node_value_dictionary_value_column().unwrap();
    let datatypes = layer.node_value_dictionary_datatype_column().unwrap();
    assert_eq!(values.len(), hashes.len());
    assert_eq!(datatypes.len(), hashes.len());
    assert_eq!(
        commit_column(&values).unwrap(),
        layer.node_value_dict_range_commitments.0
    );
    assert_eq!(
        commit_column(&datatypes).unwrap(),
        layer.node_value_dict_range_commitments.1
    );
}

#[test]
fn commitment_root_binds_the_node_value_value_and_datatype_columns() {
    use crate::proof::argument::commit_column;

    let layer = node_and_value_layer();
    assert_eq!(
        layer.recomputed_commitment_root().unwrap(),
        layer.state.commitment_root
    );
    let other = commit_column(&[
        <ArkFr as DoryField>::from_u64(11),
        <ArkFr as DoryField>::from_u64(13),
    ])
    .unwrap();
    // Tampering the VALUE commitment diverges the recomputed root...
    assert_ne!(other, layer.node_value_dict_range_commitments.0);
    let mut tampered_value = layer.clone();
    tampered_value.node_value_dict_range_commitments.0 = other;
    assert_ne!(
        tampered_value.recomputed_commitment_root().unwrap(),
        tampered_value.state.commitment_root
    );
    // ...and so does tampering the DATATYPE commitment.
    assert_ne!(other, layer.node_value_dict_range_commitments.1);
    let mut tampered_datatype = layer.clone();
    tampered_datatype.node_value_dict_range_commitments.1 = other;
    assert_ne!(
        tampered_datatype.recomputed_commitment_root().unwrap(),
        tampered_datatype.state.commitment_root
    );
}

// 255 must not collide with any real CanonicalDatatype discriminant.
#[test]
fn reserved_node_datatype_tag_is_not_a_real_discriminant() {
    use std::convert::TryFrom;
    assert_eq!(NODE_DATATYPE_TAG, 255);
    for tag in 0u8..=254 {
        // Every value below the reserved tag that IS a real discriminant is
        // strictly less than 255; TryFrom fails for the gaps. Either way,
        // no real discriminant equals 255.
        if let Ok(dt) = CanonicalDatatype::try_from(tag) {
            assert_ne!(dt as u64, NODE_DATATYPE_TAG);
        }
    }
}

// Build a real base layer through the store carrying numeric value objects
// and assert the committed VALUE column holds the expected big-endian
// integers, order-preserving (monotone) within a datatype, with Node/String
// entries getting the 0 sentinel and the reserved / String datatype tags.
#[test]
fn node_value_value_column_encodes_order_preserving_integers() {
    use crate::layer::ValueTriple;
    use crate::store::sync::open_sync_memory_store;
    use std::convert::TryInto;
    use tdb_succinct::TdbDataType;

    let iri = |s: &str| format!("http://ex/{s}");
    let p = iri("p");

    let store = open_sync_memory_store();
    let builder = store.create_base_layer().unwrap();
    // Three UInt32 values (inserted OUT of numeric order) + a Boolean + a
    // Node object + a String value.
    builder
        .add_value_triple(ValueTriple::new_value(
            &iri("s1"),
            &p,
            u32::make_entry(&300u32),
        ))
        .unwrap();
    builder
        .add_value_triple(ValueTriple::new_value(
            &iri("s2"),
            &p,
            u32::make_entry(&5u32),
        ))
        .unwrap();
    builder
        .add_value_triple(ValueTriple::new_value(
            &iri("s3"),
            &p,
            u32::make_entry(&7u32),
        ))
        .unwrap();
    builder
        .add_value_triple(ValueTriple::new_value(
            &iri("s4"),
            &p,
            bool::make_entry(&true),
        ))
        .unwrap();
    builder
        .add_value_triple(ValueTriple::new_node(&iri("s5"), &p, &iri("target")))
        .unwrap();
    builder
        .add_value_triple(ValueTriple::new_string_value(&iri("s6"), &p, "hello"))
        .unwrap();
    let layer = builder.commit().unwrap();
    let layer = layer.proof_commitment().unwrap();

    let values = layer.node_value_dictionary_value_column().unwrap();
    let datatypes = layer.node_value_dictionary_datatype_column().unwrap();
    assert_eq!(values.len(), datatypes.len());
    assert!(values.len() >= layer.node_value_catalog.len());

    let mut uint32_by_index: Vec<(usize, u64)> = Vec::new();
    let mut saw_bool = false;
    let mut saw_string = false;
    let mut saw_node = false;
    for (k, record) in layer.node_value_catalog.iter().enumerate() {
        match &record.triple.object {
            CanonicalObject::Value {
                datatype: CanonicalDatatype::UInt32,
                lexical,
            } => {
                assert_eq!(lexical.len(), 4);
                let expected = u32::from_be_bytes(lexical.as_slice().try_into().unwrap());
                assert_eq!(values[k], <ArkFr as DoryField>::from_u64(expected as u64));
                assert_eq!(
                    datatypes[k],
                    <ArkFr as DoryField>::from_u64(CanonicalDatatype::UInt32 as u64)
                );
                uint32_by_index.push((k, expected as u64));
            }
            CanonicalObject::Value {
                datatype: CanonicalDatatype::Boolean,
                lexical,
            } => {
                saw_bool = true;
                // Boolean true -> lexical [1] -> value 1.
                assert_eq!(lexical.as_slice(), &[1u8]);
                assert_eq!(values[k], <ArkFr as DoryField>::from_u64(1));
                assert_eq!(
                    datatypes[k],
                    <ArkFr as DoryField>::from_u64(CanonicalDatatype::Boolean as u64)
                );
            }
            CanonicalObject::Value {
                datatype: CanonicalDatatype::String,
                ..
            } => {
                saw_string = true;
                // Variable-width, non-OP: 0 sentinel, String datatype tag (0).
                assert_eq!(values[k], <ArkFr as DoryField>::zero());
                assert_eq!(
                    datatypes[k],
                    <ArkFr as DoryField>::from_u64(CanonicalDatatype::String as u64)
                );
            }
            CanonicalObject::Node(_) => {
                saw_node = true;
                // Nodes (subjects and the node object): 0 sentinel, reserved tag.
                assert_eq!(values[k], <ArkFr as DoryField>::zero());
                assert_eq!(
                    datatypes[k],
                    <ArkFr as DoryField>::from_u64(NODE_DATATYPE_TAG)
                );
            }
            other => panic!("unexpected catalog object {:?}", other),
        }
    }
    assert_eq!(uint32_by_index.len(), 3, "three distinct UInt32 values");
    assert!(saw_bool && saw_string && saw_node);
    // Order-preservation: the committed values 5,7,300 appear in ASCENDING
    // catalog order (the dictionary is sorted by the OP lexical bytes within
    // a datatype, so the committed column is monotone in the numeric value).
    let sorted: Vec<u64> = {
        let mut by_index = uint32_by_index.clone();
        by_index.sort_by_key(|(k, _)| *k);
        by_index.into_iter().map(|(_, v)| v).collect()
    };
    assert_eq!(sorted, vec![5u64, 7, 300]);

    // Padding rows (beyond the catalog) carry the 0 sentinel in both columns.
    for k in layer.node_value_catalog.len()..values.len() {
        assert_eq!(values[k], <ArkFr as DoryField>::zero());
        assert_eq!(datatypes[k], <ArkFr as DoryField>::zero());
    }
}

#[test]
fn every_fixed_width_integer_family_has_exact_monotone_edge_encodings() {
    use crate::layer::ValueTriple;
    use dory_pcs::primitives::arithmetic::Field;
    use tdb_succinct::TdbDataType;

    let encoded = |entry| {
        let object = CanonicalObject::from_store_object(
            &ValueTriple::new_value("", "", entry).object,
        );
        let datatype = match &object {
            CanonicalObject::Value { datatype, .. } => *datatype,
            CanonicalObject::Node(_) => unreachable!(),
        };
        let value = node_value_op_value(&object)
            .unwrap_or_else(|error| panic!("{:?}: {}; object={:?}", datatype, error, object));
        (datatype, value)
    };

    let unsigned = [
        (encoded(u8::make_entry(&u8::MIN)), CanonicalDatatype::UInt8, 0),
        (encoded(u8::make_entry(&u8::MAX)), CanonicalDatatype::UInt8, u8::MAX as u64),
        (encoded(u16::make_entry(&u16::MIN)), CanonicalDatatype::UInt16, 0),
        (encoded(u16::make_entry(&u16::MAX)), CanonicalDatatype::UInt16, u16::MAX as u64),
        (encoded(u32::make_entry(&u32::MIN)), CanonicalDatatype::UInt32, 0),
        (encoded(u32::make_entry(&u32::MAX)), CanonicalDatatype::UInt32, u32::MAX as u64),
        (encoded(u64::make_entry(&u64::MIN)), CanonicalDatatype::UInt64, 0),
        (encoded(u64::make_entry(&u64::MAX)), CanonicalDatatype::UInt64, u64::MAX),
    ];
    for ((actual_datatype, actual), expected_datatype, expected) in unsigned {
        assert_eq!(actual_datatype, expected_datatype);
        assert_eq!(actual, ArkFr::from_u64(expected));
    }

    macro_rules! signed_edges {
        ($ty:ty, $datatype:expr, $sign:expr) => {{
            let min = encoded(<$ty>::make_entry(&<$ty>::MIN));
            let zero_value: $ty = 0;
            let zero = encoded(<$ty>::make_entry(&zero_value));
            let max = encoded(<$ty>::make_entry(&<$ty>::MAX));
            assert_eq!(min.0, $datatype);
            assert_eq!(zero.0, $datatype);
            assert_eq!(max.0, $datatype);
            assert_eq!(min.1, ArkFr::zero());
            assert_eq!(zero.1, ArkFr::from_u64($sign));
            assert_eq!(
                max.1,
                ArkFr::from_u64($sign.wrapping_mul(2).wrapping_sub(1)),
            );
        }};
    }
    signed_edges!(i8, CanonicalDatatype::Int8, 1u64 << 7);
    signed_edges!(i16, CanonicalDatatype::Int16, 1u64 << 15);
    signed_edges!(i32, CanonicalDatatype::Int32, 1u64 << 31);
    signed_edges!(i64, CanonicalDatatype::Int64, 1u64 << 63);
}

#[test]
fn date_and_datetime_op_values_are_saturating_i64_nanos_offset_binary() {
    use crate::layer::ValueTriple;
    use chrono::{DateTime, NaiveDate, NaiveDateTime, Utc};
    use dory_pcs::primitives::arithmetic::Field;
    use tdb_succinct::{Date, TdbDataType};

    // The OP value the store commits for a typed entry.
    let op = |entry| {
        let object = CanonicalObject::from_store_object(
            &ValueTriple::new_value("", "", entry).object,
        );
        node_value_op_value(&object).expect("op value")
    };
    // INDEPENDENT offset-binary code for a known i64 key (mirrors the Int64
    // convention `(n as u64) ^ (1<<63)`), computed WITHOUT the private helper so
    // this test is a genuine oracle for `node_value_op_value`.
    let code = |n: i64| (n as u64) ^ (1u64 << 63);
    let field = |n: i64| ArkFr::from_u64(code(n));
    let utc = |secs: i64, nanos: u32| -> NaiveDateTime {
        DateTime::<Utc>::from_timestamp(secs, nanos).unwrap().naive_utc()
    };

    // --- xsd:dateTime: chronological UTC instant, to the nanosecond ---
    let dt_samples: Vec<(NaiveDateTime, i64)> = vec![
        (utc(0, 0), 0),                                         // the epoch
        (utc(0, 1), 1),                                         // +1 ns
        (utc(-1, 0), -1_000_000_000),                          // 1 s before epoch
        (utc(1_000_000_000, 123), 1_000_000_000 * 1_000_000_000 + 123),
    ];
    for (ndt, expected_nanos) in &dt_samples {
        assert_eq!(
            op(NaiveDateTime::make_entry(ndt)),
            field(*expected_nanos),
            "dateTime OP == offset-binary(i64-nanos) for {ndt}",
        );
    }
    // Order-preserving: sorting by the key strictly increases the offset-binary code
    // (so a < b ⇒ OP(a) < OP(b)) — including ACROSS the epoch (negative keys).
    let mut keys: Vec<i64> = dt_samples.iter().map(|(_, n)| *n).collect();
    keys.sort_unstable();
    for w in keys.windows(2) {
        assert!(code(w[0]) < code(w[1]), "offset-binary is monotone in the key");
    }

    // dateTime SATURATION: outside the i64-nanoseconds window saturates by side.
    let far_future = NaiveDate::from_ymd_opt(5000, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    let far_past = NaiveDate::from_ymd_opt(1000, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    assert_eq!(op(NaiveDateTime::make_entry(&far_future)), field(i64::MAX));
    assert_eq!(op(NaiveDateTime::make_entry(&far_past)), field(i64::MIN));

    // --- xsd:date: (year, month, day, offset)-lexical via midnight_utc_nanos + offset ---
    let date_entry = |y: i64, m: u8, d: u8, off: i16| {
        Date::make_entry(&Date {
            year: y,
            month: m,
            day: d,
            offset: off,
        })
    };
    let midnight = |y: i32, m: u32, d: u32| -> i64 {
        NaiveDate::from_ymd_opt(y, m, d)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp_nanos_opt()
            .unwrap()
    };
    let n_2020 = midnight(2020, 1, 1);
    // offset 0 (the compiler's constant form) is a pure midnight instant.
    assert_eq!(op(date_entry(2020, 1, 1, 0)), field(n_2020));
    // The timezone offset is a LEAST-significant tiebreaker: a larger offset sorts
    // later, a negative one earlier, matching WOQL's `compare/3` over `date/4`.
    assert_eq!(op(date_entry(2020, 1, 1, 60)), field(n_2020 + 60));
    assert_eq!(op(date_entry(2020, 1, 1, -60)), field(n_2020 - 60));
    // The offset can NEVER cross a day boundary (a day is ~8.64e13 ns, |offset|≤32767):
    // 2019-12-31 < any-offset 2020-01-01 < 2020-01-02.
    let n_dec31 = midnight(2019, 12, 31);
    let n_jan02 = midnight(2020, 1, 2);
    assert!(code(n_dec31) < code(n_2020 - 60), "prior day < offset-shifted date");
    assert!(code(n_2020 + 60) < code(n_jan02), "offset-shifted date < next day");
    // date SATURATION: a year outside the window saturates, and the offset is pinned
    // at the boundary so it can never nudge a far date back toward the constant.
    assert_eq!(op(date_entry(5000, 1, 1, 120)), field(i64::MAX));
    assert_eq!(op(date_entry(1000, 1, 1, -120)), field(i64::MIN));
}

#[test]
fn adversarial_date_and_datetime_lexicals_fail_closed_without_panicking() {
    use chrono::{DateTime, NaiveDate, NaiveDateTime, Utc};
    use dory_pcs::primitives::arithmetic::Field;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use tdb_succinct::{Date, TdbDataType};

    // `node_value_op_value` runs inside `VerifierCommitmentRef::validate()` over
    // adversary-supplied catalog rows on a trustless proof, so a panic on a crafted
    // date/dateTime lexical is a verifier DoS. Every crafted lexical below must
    // either fail closed (`Err`) or, for a WELL-FORMED-but-out-of-window instant,
    // saturate (`Ok`) — but it must NEVER panic. `catch_unwind` turns any residual
    // panic into a clean test failure. NB the ORIGINAL native decoders panicked on
    // several of these (the exact DoS this test pins shut).
    // Decode UNDER a panic guard: any panic is caught and re-raised as a clear
    // test failure (the DoS we are pinning shut), leaving a plain `io::Result` the
    // assertions can inspect and Debug-format.
    let decode = |datatype: CanonicalDatatype, lexical: Vec<u8>| -> io::Result<ArkFr> {
        catch_unwind(AssertUnwindSafe(|| {
            node_value_op_value(&CanonicalObject::Value { datatype, lexical })
        }))
        .unwrap_or_else(|_| {
            panic!("node_value_op_value PANICKED on an adversarial lexical (verifier DoS)")
        })
    };
    // Offset-binary code of an i64 key (the shared saturation boundary encoding).
    let field = |n: i64| ArkFr::from_u64((n as u64) ^ (1u64 << 63));

    // --- (a) xsd:dateTime with bigint-seconds beyond i64 -----------------------
    // Size prefix `0x8C` = positive, 12 magnitude bytes; twelve `0xFF` magnitude
    // bytes (≫ i64::MAX); a `0x00` "false zero" fraction. WELL-FORMED but far
    // outside the i64-nanoseconds window ⇒ SATURATES to i64::MAX (never panics; the
    // native `int.to_i64().expect(...)` would have panicked here).
    let mut dt_huge_pos = vec![0x8Cu8];
    dt_huge_pos.extend(std::iter::repeat(0xFFu8).take(12));
    dt_huge_pos.push(0x00);
    let a = decode(CanonicalDatatype::DateTime, dt_huge_pos);
    assert!(
        matches!(&a, Ok(v) if *v == field(i64::MAX)),
        "over-i64 dateTime saturates to i64::MAX, got {a:?}",
    );
    // Same magnitude, NEGATIVE sign (`0x73` = !0x8C): saturates to i64::MIN.
    let mut dt_huge_neg = vec![0x73u8];
    dt_huge_neg.extend(std::iter::repeat(0x00u8).take(12)); // !0xFF under negative sign
    dt_huge_neg.push(0xFF); // !0x00 fraction "false zero" under negative sign
    let a_neg = decode(CanonicalDatatype::DateTime, dt_huge_neg);
    assert!(
        matches!(&a_neg, Ok(v) if *v == field(i64::MIN)),
        "under-i64 dateTime saturates to i64::MIN, got {a_neg:?}",
    );
    // (a') the SAME huge magnitude but TRUNCATED (missing the fraction byte): an
    // undecodable/malformed lexical ⇒ fails closed with Err (no panic).
    let mut dt_huge_trunc = vec![0x8Cu8];
    dt_huge_trunc.extend(std::iter::repeat(0xFFu8).take(12));
    let a_trunc = decode(CanonicalDatatype::DateTime, dt_huge_trunc);
    assert!(a_trunc.is_err(), "truncated dateTime must Err, got {a_trunc:?}");
    // (a'') size prefix claims 8 magnitude bytes but only 2 are present ⇒ Err.
    let a_short = decode(CanonicalDatatype::DateTime, vec![0x88, 0x01, 0x02]);
    assert!(a_short.is_err(), "short-magnitude dateTime must Err, got {a_short:?}");
    // (a''') empty lexical (no size prefix at all) ⇒ Err.
    let a_empty = decode(CanonicalDatatype::DateTime, Vec::new());
    assert!(a_empty.is_err(), "empty dateTime must Err, got {a_empty:?}");

    // --- (b) truncated xsd:date (fewer than 12 bytes) --------------------------
    let b = decode(CanonicalDatatype::Date, vec![0x00, 0x01, 0x02]);
    assert!(b.is_err(), "truncated date must Err, got {b:?}");

    // --- (c) over-long xsd:date (13 bytes) -------------------------------------
    let c = decode(CanonicalDatatype::Date, vec![0u8; 13]);
    assert!(c.is_err(), "over-long date must Err, got {c:?}");

    // Valid store-produced date/dateTime lexicals still decode to the SAME OP
    // values (ordering + offset tiebreaker unchanged) — proving the defensive
    // decode did not perturb the honest path.
    let op = |entry| {
        node_value_op_value(&CanonicalObject::from_store_object(
            &crate::layer::ValueTriple::new_value("", "", entry).object,
        ))
        .expect("valid op value")
    };
    let ndt = |secs: i64, nanos: u32| -> NaiveDateTime {
        DateTime::<Utc>::from_timestamp(secs, nanos).unwrap().naive_utc()
    };
    // dateTime: epoch, +1ns, 1s-before-epoch, and a large-nanos instant.
    assert_eq!(op(NaiveDateTime::make_entry(&ndt(0, 0))), field(0));
    assert_eq!(op(NaiveDateTime::make_entry(&ndt(0, 1))), field(1));
    assert_eq!(op(NaiveDateTime::make_entry(&ndt(-1, 0))), field(-1_000_000_000));
    assert_eq!(
        op(NaiveDateTime::make_entry(&ndt(1_000_000_000, 123))),
        field(1_000_000_000 * 1_000_000_000 + 123),
    );
    // date: midnight instant + offset tiebreaker, byte-identical to before.
    let n_2020 = NaiveDate::from_ymd_opt(2020, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap()
        .and_utc()
        .timestamp_nanos_opt()
        .unwrap();
    let dmk = |y: i64, m: u8, d: u8, off: i16| {
        Date::make_entry(&Date { year: y, month: m, day: d, offset: off })
    };
    assert_eq!(op(dmk(2020, 1, 1, 0)), field(n_2020));
    assert_eq!(op(dmk(2020, 1, 1, 60)), field(n_2020 + 60));
    assert_eq!(op(dmk(2020, 1, 1, -60)), field(n_2020 - 60));
}

#[test]
fn non_date_entries_keep_the_zero_sentinel_byte_identically() {
    // The date OP change must NOT perturb any non-date entry: nodes, strings, and
    // every temporal type OTHER than date/dateTime (Time, DateTimeStamp — deliberately
    // reject-closed) must still map to the `0` sentinel, so a layer WITHOUT dates
    // commits a byte-identical node/value VALUE column (and thus an identical
    // `commitment_root`) under the format bump.
    use crate::layer::ValueTriple;
    use chrono::{DateTime, NaiveTime, Utc};
    use dory_pcs::primitives::arithmetic::Field;
    use tdb_succinct::TdbDataType;

    let op = |object: CanonicalObject| node_value_op_value(&object).expect("op value");
    let value_op = |entry| {
        op(CanonicalObject::from_store_object(
            &ValueTriple::new_value("", "", entry).object,
        ))
    };
    let zero = <ArkFr as Field>::zero();

    // A Node object.
    assert_eq!(op(CanonicalObject::Node(b"http://ex/x".to_vec())), zero);
    // xsd:string.
    assert_eq!(value_op(String::make_entry(&"hello".to_string())), zero);
    // xsd:time — a temporal type that is NOT range-supported: stays sentinel.
    let time = NaiveTime::from_hms_opt(12, 0, 0).unwrap();
    assert_eq!(value_op(NaiveTime::make_entry(&time)), zero);
    // xsd:dateTimeStamp — stores like dateTime but is deliberately reject-closed
    // (the WOQL AST cannot name it); it must remain a sentinel so a layer holding one
    // (but no date/dateTime) stays byte-identical.
    let stamp = tdb_succinct::DateTimeStamp(
        DateTime::<Utc>::from_timestamp(1_600_000_000, 0)
            .unwrap()
            .naive_utc(),
    );
    assert_eq!(value_op(tdb_succinct::DateTimeStamp::make_entry(&stamp)), zero);
}

#[test]
fn node_value_dictionary_opens_for_node_and_value_entries() {
    use crate::proof::argument::{commit_column, Transcript254};
    use crate::proof::dict::{hash_object, prove_dict_membership, verify_dict_membership};

    let layer = node_and_value_layer();
    let (dict_hash, dict_id) = layer.node_value_dictionary_columns().unwrap();
    // The committed columns are exactly the ones a membership proof opens.
    assert_eq!(
        commit_column(&dict_hash).unwrap(),
        layer.node_value_dict_commitments.0
    );
    assert_eq!(
        commit_column(&dict_id).unwrap(),
        layer.node_value_dict_commitments.1
    );
    let com_h = layer.node_value_dict_commitments.0;
    let com_i = layer.node_value_dict_commitments.1;

    // A prove+verify round for one `(hash_object(term), id)` pair.
    let open = |q_hash: ArkFr, id: ArkFr| -> io::Result<()> {
        let mut pt = Transcript254::new(b"nv-dict-open");
        let proof = prove_dict_membership(
            &dict_hash, &dict_id, &com_h, &com_i, q_hash, id, &mut pt,
        )?;
        let mut vt = Transcript254::new(b"nv-dict-open");
        verify_dict_membership(&com_h, &com_i, q_hash, id, &proof, &mut vt)
    };

    // Resolve ids from the catalog for BOTH a Node object and a Value object.
    let node_obj = CanonicalObject::Node(b"b".to_vec());
    let value_obj = CanonicalObject::Value {
        datatype: CanonicalDatatype::String,
        lexical: b"lit".to_vec(),
    };
    let id_of = |obj: &CanonicalObject| -> u64 {
        layer
            .node_value_catalog
            .iter()
            .find(|r| &r.triple.object == obj)
            .expect("object present in catalog")
            .object_id
    };

    for obj in [&node_obj, &value_obj] {
        let id = id_of(obj);
        let q_hash = hash_object(obj);
        let id_fr = <ArkFr as DoryField>::from_u64(id);
        // Honest opening verifies.
        assert!(open(q_hash, id_fr).is_ok());
        // A WRONG id for a real term is rejected (the pair is bound).
        assert!(open(q_hash, <ArkFr as DoryField>::from_u64(id + 100)).is_err());
    }

    // An ABSENT term is rejected.
    let absent = hash_object(&CanonicalObject::Node(b"not-in-dictionary".to_vec()));
    assert!(open(absent, <ArkFr as DoryField>::from_u64(2)).is_err());
}

#[test]
fn transparent_setup_is_deterministic_across_fresh_derivations() {
    // Two independently-derived setups must produce byte-identical public
    // parameters, so any node commits a given column to the same value.
    let setup_a = deterministic_prover_setup(4);
    let setup_b = deterministic_prover_setup(4);
    assert_eq!(setup_a.g1_vec, setup_b.g1_vec);
    assert_eq!(setup_a.g2_vec, setup_b.g2_vec);

    let coefficients: Vec<ArkFr> = (1..=4u64).map(<ArkFr as DoryField>::from_u64).collect();
    let poly = ArkworksPolynomial::new(coefficients);
    let (commitment_a, _, _) = poly
        .commit::<BN254, Transparent, G1Routines>(1, 1, &setup_a)
        .unwrap();
    let (commitment_b, _, _) = poly
        .commit::<BN254, Transparent, G1Routines>(1, 1, &setup_b)
        .unwrap();
    assert_eq!(commitment_a, commitment_b);
}

#[test]
fn commitment_root_is_stable_across_independent_calls() {
    // End-to-end determinism: the same delta yields the same root.
    assert_eq!(
        sample().state.commitment_root,
        sample().state.commitment_root
    );
    assert_ne!(sample().state.commitment_root, [0u8; 32]);
}

#[test]
fn changing_one_record_id_changes_the_root() {
    // a-p-b (subject 1, object 2) vs b-p-a (subject 2, object 1): identical
    // dictionary/counts, but the id columns differ, so the commitments and
    // hence the root must differ.
    let a_p_b = delta(vec![record(1, 1, 2, "a", "p", "b")], vec![]);
    let b_p_a = delta(vec![record(2, 1, 1, "b", "p", "a")], vec![]);
    assert_ne!(a_p_b.state.commitment_root, b_p_a.state.commitment_root);
}

#[test]
fn empty_adds_only_removes_only_are_valid_and_distinct() {
    let empty = LayerCommitment::from_native_delta(
        None,
        vec![],
        vec![],
        vec![],
        0,
        0,
        0,
        0,
        vec![],
        vec![],
    )
    .unwrap();
    let adds_only = delta(vec![record(1, 1, 2, "a", "p", "b")], vec![]);
    let removes_only = delta(vec![], vec![record(1, 1, 2, "a", "p", "b")]);

    // Empty columns use the canonical GT-identity commitment.
    let identity = <ArkGT as Group>::identity();
    assert!(empty.column_commitments.iter().all(|c| *c == identity));
    // Adds-only: remove columns are empty; removes-only: add columns empty.
    assert!(adds_only.column_commitments[3..]
        .iter()
        .all(|c| *c == identity));
    assert!(removes_only.column_commitments[..3]
        .iter()
        .all(|c| *c == identity));

    let roots = [
        empty.state.commitment_root,
        adds_only.state.commitment_root,
        removes_only.state.commitment_root,
    ];
    assert_ne!(roots[0], roots[1]);
    assert_ne!(roots[0], roots[2]);
    assert_ne!(roots[1], roots[2]);
}

#[test]
fn tampered_body_is_rejected() {
    let commitment = sample();
    let mut bytes = commitment.encode().unwrap();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 1;
    assert!(LayerCommitment::decode(&bytes).is_err());
}

#[test]
fn tampered_materialized_view_is_rejected() {
    let mut commitment = sample();
    commitment.materialized[0].object_id = 1;
    commitment.materialized[0].triple.object = CanonicalObject::Node(b"a".to_vec());
    assert!(commitment.validate().is_err());
}

#[test]
fn tampered_full_catalog_is_rejected() {
    let mut commitment = sample();
    commitment.node_value_catalog[1].triple.object =
        CanonicalObject::Node(b"tampered".to_vec());
    assert!(commitment.validate().is_err());
}

#[test]
fn tampered_serialized_commitment_is_rejected_even_with_fixed_checksum() {
    let commitment = sample();
    let mut bytes = commitment.encode().unwrap();
    // The serialized commitments sit immediately before the trailing 32-byte
    // commitment_root and 32-byte file checksum. Flip a byte inside the last
    // commitment blob, then repair the file checksum so the rejection is
    // driven by the commitment_root binding, not the outer checksum.
    let commitment_byte = bytes.len() - 64 - 1;
    bytes[commitment_byte] ^= 1;
    let payload_len = bytes.len() - 32;
    let repaired = file_checksum(&bytes[..payload_len]);
    bytes[payload_len..].copy_from_slice(&repaired);
    assert!(LayerCommitment::decode(&bytes).is_err());
}

#[test]
fn catalog_gap_fails_closed() {
    assert!(LayerCommitment::from_native_delta(
        None,
        vec![],
        vec![],
        vec![],
        0,
        0,
        3,
        0,
        vec![
            (CanonicalObject::Node(b"a".to_vec()), 1),
            (CanonicalObject::Node(b"c".to_vec()), 3),
        ],
        vec![],
    )
    .is_err());
}

// A store-backed base + one delta child, returned as owned LayerCommitments.
// The head is a genuine delta (parent_state_root == Some) — the only reliable way
// to get a valid delta sidecar for the term-less producer's base/rollup-only gate.
fn store_base_and_delta() -> (std::sync::Arc<LayerCommitment>, std::sync::Arc<LayerCommitment>) {
    use crate::layer::ValueTriple;
    use crate::store::sync::open_sync_memory_store;
    let iri = |s: &str| format!("http://ex/{s}");
    let store = open_sync_memory_store();
    let builder = store.create_base_layer().unwrap();
    for (s, o) in [("a1", "b1"), ("a2", "b2"), ("a3", "b3")] {
        builder
            .add_value_triple(ValueTriple::new_node(&iri(s), &iri("p1"), &iri(o)))
            .unwrap();
    }
    let base = builder.commit().unwrap();
    let base_lc = base.proof_commitment().unwrap();
    let child = base.open_write().unwrap();
    child
        .add_value_triple(ValueTriple::new_node(&iri("a4"), &iri("p1"), &iri("b4")))
        .unwrap();
    let head = child.commit().unwrap();
    let head_lc = head.proof_commitment().unwrap();
    (base_lc, head_lc)
}

#[test]
fn termless_base_artifact_roundtrips_and_drops_catalogs() {
    let layer = sample();

    // Term-bearing (default) baseline.
    let bearing = layer.verifier_commitment().unwrap();
    assert!(bearing.terms_present);
    let bearing_bytes = bearing.encode().unwrap();

    // Term-less producer emits a catalog-less artifact.
    let termless = layer.verifier_commitment_termless().unwrap();
    assert!(!termless.terms_present);
    assert!(termless.predicate_catalog.is_empty());
    assert!(termless.node_value_catalog.is_empty());

    // It round-trips through encode -> decode -> validate.
    let termless_bytes = termless.encode().unwrap();
    let decoded = VerifierCommitment::decode(&termless_bytes).unwrap();
    assert!(!decoded.terms_present);
    assert!(decoded.predicate_catalog.is_empty());
    assert!(decoded.node_value_catalog.is_empty());
    decoded.validate().unwrap();

    // The catalog-less encoding is strictly smaller (the O(dict) bytes are gone).
    assert!(termless_bytes.len() < bearing_bytes.len());

    // recomputed_commitment_root() holds from the STORED commitments (no catalog),
    // and commitment_root is BYTE-IDENTICAL to the term-bearing artifact of the same
    // layer — the flag lives in the framing, not in commitment_root.
    assert_eq!(
        decoded.recomputed_commitment_root().unwrap(),
        decoded.state.commitment_root
    );
    assert_eq!(termless.state.commitment_root, bearing.state.commitment_root);
    assert_eq!(termless.state.commitment_root, layer.state.commitment_root);

    // resolution_root is likewise identical (same commitment_root + Merkle roots).
    assert_eq!(termless.resolution_root(), bearing.resolution_root());
}

#[test]
fn termless_producer_rejects_a_delta_layer() {
    let (base_lc, head_lc) = store_base_and_delta();
    // The base (parent_state_root == None) produces a term-less artifact.
    assert!(base_lc.state.parent_state_root.is_none());
    let termless = base_lc.verifier_commitment_termless().unwrap();
    assert!(!termless.terms_present);
    termless.validate().unwrap();

    // The delta head (parent_state_root == Some) is rejected: term-less delta
    // querying needs cross-layer re-mint over faithful hashes (blocked on I4).
    assert!(head_lc.state.parent_state_root.is_some());
    assert!(head_lc.verifier_commitment_termless().is_err());
    // The term-BEARING producer still accepts the delta (unchanged behavior).
    assert!(head_lc.verifier_commitment().is_ok());
}

#[test]
fn termless_validate_rejects_smuggled_catalog() {
    let layer = sample();
    let mut smuggled = layer.verifier_commitment_termless().unwrap();
    // Splice a catalog row into a term-less artifact: validate must fail closed
    // (a term-less artifact carries no catalog; the flag stays false).
    let bearing = layer.verifier_commitment().unwrap();
    smuggled.node_value_catalog = vec![bearing.node_value_catalog[0].clone()];
    assert!(smuggled.validate().is_err());
}

#[test]
fn termless_decode_rejects_bad_flag_byte() {
    let termless = sample().verifier_commitment_termless().unwrap();
    let mut bytes = termless.encode().unwrap();
    // The flag byte sits immediately after the 8-byte magic + 4-byte version.
    let flag_offset = VERIFIER_COMMITMENT_MAGIC.len() + 4;
    bytes[flag_offset] = 2; // neither 0 nor 1
                            // Repair the trailing file checksum so the rejection is driven by the
                            // flag parse, not the outer checksum.
    let payload_len = bytes.len() - 32;
    let repaired = file_checksum(&bytes[..payload_len]);
    bytes[payload_len..].copy_from_slice(&repaired);
    assert!(VerifierCommitment::decode(&bytes).is_err());
}

#[test]
fn term_bearing_artifact_is_v4_and_reports_terms_present() {
    // The term-bearing path is behaviorally unchanged except for the v3->v4 bump +
    // flag byte: it still ships + round-trips its catalogs, now under v4.
    let layer = sample();
    let artifact = layer.verifier_commitment().unwrap();
    assert!(artifact.terms_present);
    let bytes = artifact.encode().unwrap();
    // version u32 (big-endian) lives right after the magic; must read 4.
    let v = u32::from_be_bytes(
        bytes[VERIFIER_COMMITMENT_MAGIC.len()..VERIFIER_COMMITMENT_MAGIC.len() + 4]
            .try_into()
            .unwrap(),
    );
    assert_eq!(v, VERIFIER_COMMITMENT_FORMAT_VERSION);
    assert_eq!(v, 4);
    let decoded = VerifierCommitment::decode(&bytes).unwrap();
    assert!(decoded.terms_present);
    assert_eq!(decoded.predicate_catalog, layer.predicate_catalog);
    assert_eq!(decoded.node_value_catalog, layer.node_value_catalog);
}
