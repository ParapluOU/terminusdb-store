//! Dictionary catalog building and integrity validation.
//!
//! A layer's dictionaries map opaque ids to canonical bytes (predicate IRIs, and node/value
//! terms). Before those catalogs are committed — and again when a verifier re-derives them —
//! they must be well-formed: ids and bytes in bijection, canonically sorted, parent counts
//! consistent, and no id bound to two different terms. This module builds the catalogs from
//! the id maps and holds the validation that enforces those invariants, so a committed
//! dictionary can be trusted as an authenticated id<->term table.

use super::*;

pub(super) fn put_records(out: &mut Vec<u8>, records: &[ProofRecord]) {
    put_u64(out, records.len() as u64);
    for record in records {
        put_u64(out, record.subject_id);
        put_u64(out, record.predicate_id);
        put_u64(out, record.object_id);
        record.triple.encode_into(out);
    }
}

pub(super) fn file_checksum(payload: &[u8]) -> ProofHash {
    let mut hasher = Sha256::new();
    hasher.update(b"terminus-store/proof-of-woql/file/v0");
    hasher.update((payload.len() as u64).to_be_bytes());
    hasher.update(payload);
    hasher.finalize().into()
}

pub(super) fn build_dictionary_catalogs_from_maps(
    node_value_mappings: Vec<(CanonicalObject, u64)>,
    predicate_mappings: Vec<(Vec<u8>, u64)>,
) -> io::Result<[Vec<ProofRecord>; 3]> {
    let mut subjects = BTreeMap::<Vec<u8>, u64>::new();
    let mut predicates = BTreeMap::<Vec<u8>, u64>::new();
    let mut objects = BTreeMap::<CanonicalObject, u64>::new();
    for (object, id) in node_value_mappings {
        if let CanonicalObject::Node(node) = &object {
            insert_mapping(&mut subjects, node.clone(), id)?;
        }
        insert_mapping(&mut objects, object, id)?;
    }
    for (predicate, id) in predicate_mappings {
        insert_mapping(&mut predicates, predicate, id)?;
    }
    let subject_records = subjects
        .into_iter()
        .map(|(subject, subject_id)| ProofRecord {
            subject_id,
            predicate_id: 0,
            object_id: 0,
            triple: CanonicalTriple::node(subject, Vec::new(), Vec::new()),
        })
        .collect();
    let predicate_records = predicates
        .into_iter()
        .map(|(predicate, predicate_id)| ProofRecord {
            subject_id: 0,
            predicate_id,
            object_id: 0,
            triple: CanonicalTriple::node(Vec::new(), predicate, Vec::new()),
        })
        .collect();
    let object_records = objects
        .into_iter()
        .map(|(object, object_id)| ProofRecord {
            subject_id: 0,
            predicate_id: 0,
            object_id,
            triple: CanonicalTriple {
                subject: Vec::new(),
                predicate: Vec::new(),
                object,
            },
        })
        .collect();
    Ok([subject_records, predicate_records, object_records])
}

pub(super) fn subject_catalog_from_objects(objects: &[ProofRecord]) -> io::Result<Vec<ProofRecord>> {
    let mut subjects = BTreeMap::<Vec<u8>, u64>::new();
    for record in objects {
        if let CanonicalObject::Node(node) = &record.triple.object {
            insert_mapping(&mut subjects, node.clone(), record.object_id)?;
        }
    }
    Ok(subjects
        .into_iter()
        .map(|(subject, subject_id)| ProofRecord {
            subject_id,
            predicate_id: 0,
            object_id: 0,
            triple: CanonicalTriple::node(subject, Vec::new(), Vec::new()),
        })
        .collect())
}

pub(super) fn insert_mapping<K: Ord>(map: &mut BTreeMap<K, u64>, key: K, id: u64) -> io::Result<()> {
    if let Some(previous) = map.insert(key, id) {
        if previous != id {
            return Err(invalid(
                "canonical dictionary value maps to multiple native ids",
            ));
        }
    }
    Ok(())
}

pub(super) fn insert_bijection<K: Ord + Clone>(
    by_key: &mut BTreeMap<K, u64>,
    by_id: &mut BTreeMap<u64, K>,
    key: K,
    id: u64,
) -> io::Result<()> {
    insert_mapping(by_key, key.clone(), id)?;
    if let Some(previous) = by_id.insert(id, key.clone()) {
        if previous != key {
            return Err(invalid(
                "native dictionary id maps to multiple canonical values",
            ));
        }
    }
    Ok(())
}

pub(super) fn validate_dictionary_catalogs(
    actual: [&[ProofRecord]; 3],
    node_value_count: u64,
    predicate_count: u64,
) -> io::Result<()> {
    let mut subjects = BTreeMap::new();
    let mut subject_ids = BTreeMap::new();
    for record in actual[0].iter() {
        if record.subject_id == 0
            || record.subject_id > node_value_count
            || record.predicate_id != 0
            || record.object_id != 0
            || !record.triple.predicate.is_empty()
            || record.triple.object != CanonicalObject::Node(Vec::new())
        {
            return Err(invalid("malformed subject dictionary record"));
        }
        insert_bijection(
            &mut subjects,
            &mut subject_ids,
            record.triple.subject.clone(),
            record.subject_id,
        )?;
    }
    let mut predicates = BTreeMap::new();
    let mut predicate_ids = BTreeMap::new();
    for record in actual[1].iter() {
        if record.predicate_id == 0
            || record.predicate_id > predicate_count
            || record.subject_id != 0
            || record.object_id != 0
            || !record.triple.subject.is_empty()
            || record.triple.object != CanonicalObject::Node(Vec::new())
        {
            return Err(invalid("malformed predicate dictionary record"));
        }
        insert_bijection(
            &mut predicates,
            &mut predicate_ids,
            record.triple.predicate.clone(),
            record.predicate_id,
        )?;
    }
    let mut objects = BTreeMap::new();
    let mut object_ids = BTreeMap::new();
    for record in actual[2].iter() {
        if record.object_id == 0
            || record.object_id > node_value_count
            || record.subject_id != 0
            || record.predicate_id != 0
            || !record.triple.subject.is_empty()
            || !record.triple.predicate.is_empty()
        {
            return Err(invalid("malformed object dictionary record"));
        }
        insert_bijection(
            &mut objects,
            &mut object_ids,
            record.triple.object.clone(),
            record.object_id,
        )?;
    }
    let expected_subjects: BTreeMap<_, _> = objects
        .iter()
        .filter_map(|(object, id)| match object {
            CanonicalObject::Node(node) => Some((node.clone(), *id)),
            CanonicalObject::Value { .. } => None,
        })
        .collect();
    if subjects != expected_subjects {
        return Err(invalid(
            "subject dictionary is not the node dictionary subset",
        ));
    }
    Ok(())
}

/// Validate that the (possibly INCREMENTAL) dictionary catalog covers exactly the
/// id interval this layer INTRODUCED — `(parent_count, count]` — and that every
/// delta/materialized record id introduced by this layer is bound to canonical
/// bytes by that catalog.
///
/// For a BASE/ROLLUP/SQUASH layer `parent_*_count == 0`, so the interval is the
/// full `1..=count` and every record id is bound: this collapses to the exact
/// pre-diet check and stays BYTE-IDENTICAL. For a DELTA layer the catalog holds
/// only `parent_count+1..=count` (the commitment-diet incremental dictionary), so
/// INHERITED ids (`<= parent_count`) are deliberately NOT bound here — they are
/// authenticated by an ancestor layer's committed dictionary when the delta stack
/// is reconciled at proof time (Wave 1/2). A delta layer is never directly
/// query-provable, so no verifier trusts an inherited-id binding from this sidecar.
#[allow(clippy::too_many_arguments)]
pub(super) fn validate_complete_catalog(
    additions: &[ProofRecord],
    removals: &[ProofRecord],
    materialized: &[ProofRecord],
    indexes: [&[ProofRecord]; 3],
    parent_node_value_count: u64,
    parent_predicate_count: u64,
    node_value_count: u64,
    predicate_count: u64,
) -> io::Result<()> {
    validate_dictionary_catalogs(indexes, node_value_count, predicate_count)?;
    if indexes[2].len() as u64 != node_value_count - parent_node_value_count
        || indexes[1].len() as u64 != predicate_count - parent_predicate_count
    {
        return Err(invalid(
            "dictionary catalog does not exactly cover the introduced id interval",
        ));
    }
    // Every catalog id must lie strictly above the inherited watermark. For a base
    // layer the watermark is 0, so this is vacuous (ids already validated `>= 1`).
    if indexes[0]
        .iter()
        .any(|record| record.subject_id <= parent_node_value_count)
        || indexes[2]
            .iter()
            .any(|record| record.object_id <= parent_node_value_count)
        || indexes[1]
            .iter()
            .any(|record| record.predicate_id <= parent_predicate_count)
    {
        return Err(invalid(
            "dictionary catalog contains an inherited id below the parent watermark",
        ));
    }

    let mut objects_by_id = BTreeMap::new();
    for record in indexes[2].iter() {
        objects_by_id.insert(record.object_id, record.triple.object.clone());
    }
    let mut predicates_by_id = BTreeMap::new();
    for record in indexes[1].iter() {
        predicates_by_id.insert(record.predicate_id, record.triple.predicate.clone());
    }
    let subjects_by_id: BTreeMap<_, _> = indexes[0]
        .iter()
        .map(|record| (record.subject_id, record.triple.subject.clone()))
        .collect();

    for record in additions
        .iter()
        .chain(removals.iter())
        .chain(materialized.iter())
    {
        // Only ids INTRODUCED by this layer are locally bound to canonical bytes;
        // inherited ids are authenticated by an ancestor at reconciliation time.
        if record.subject_id > parent_node_value_count
            && (subjects_by_id.get(&record.subject_id) != Some(&record.triple.subject)
                || objects_by_id.get(&record.subject_id)
                    != Some(&CanonicalObject::Node(record.triple.subject.clone())))
        {
            return Err(invalid("subject id is not bound to canonical bytes"));
        }
        if record.predicate_id > parent_predicate_count
            && predicates_by_id.get(&record.predicate_id) != Some(&record.triple.predicate)
        {
            return Err(invalid("predicate id is not bound to canonical bytes"));
        }
        if record.object_id > parent_node_value_count
            && objects_by_id.get(&record.object_id) != Some(&record.triple.object)
        {
            return Err(invalid("object id is not bound to canonical bytes"));
        }
    }
    Ok(())
}

