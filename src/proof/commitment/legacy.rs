//! Migration-only decoder for the previous (PF4 v3) sidecar format.
//!
//! Older stored commitments used a v3 layout. This module decodes just enough of a v3
//! sidecar ([`decode_legacy_v3`] into [`LegacyV3Commitment`]) to migrate it forward. Normal
//! reads and pack admission never touch this path; it exists so historical data stays
//! readable across the format bump.

use super::*;

/// Fully validated migration-only representation of the immediately preceding
/// PF4 v3 sidecar format. Normal reads and pack admission never call this decoder.
pub(crate) struct LegacyV3Commitment {
    pub parent_state_root: Option<ProofHash>,
    pub commitment_root: ProofHash,
    addition_count: u64,
    removal_count: u64,
    parent_node_value_count: u64,
    parent_predicate_count: u64,
    node_value_count: u64,
    predicate_count: u64,
    additions: Vec<ProofRecord>,
    removals: Vec<ProofRecord>,
    predicate_catalog: Vec<ProofRecord>,
    node_value_catalog: Vec<ProofRecord>,
}

impl LegacyV3Commitment {
    pub(crate) fn validate_against_v4(&self, current: &LayerCommitment) -> io::Result<()> {
        let legacy_predicates: Vec<_> = current
            .predicate_catalog
            .iter()
            .filter(|record| record.predicate_id > current.state.parent_predicate_count)
            .cloned()
            .collect();
        let legacy_objects: Vec<_> = current
            .node_value_catalog
            .iter()
            .filter(|record| record.object_id > current.state.parent_node_value_count)
            .cloned()
            .collect();
        if self.addition_count != current.state.addition_count
            || self.removal_count != current.state.removal_count
            || self.parent_node_value_count != current.state.parent_node_value_count
            || self.parent_predicate_count != current.state.parent_predicate_count
            || self.node_value_count != current.state.node_value_count
            || self.predicate_count != current.state.predicate_count
            || self.additions != current.additions
            || self.removals != current.removals
            || self.predicate_catalog != legacy_predicates
            || self.node_value_catalog != legacy_objects
        {
            return Err(invalid(
                "legacy v3 proof state differs from native layer delta",
            ));
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn legacy_v3_root_from_parts(
    parent_state_root: Option<ProofHash>,
    addition_count: u64,
    removal_count: u64,
    parent_node_value_count: u64,
    parent_predicate_count: u64,
    node_value_count: u64,
    predicate_count: u64,
    column_commitments: &[ArkGT; COLUMN_COUNT],
    predicate_dict_commitments: &(ArkGT, ArkGT),
    node_value_dict_commitments: &(ArkGT, ArkGT),
    node_value_dict_range_commitments: &(ArkGT, ArkGT),
) -> io::Result<ProofHash> {
    let mut hasher = Sha256::new();
    hasher.update(b"terminus-store/proof-of-woql/commitment-root/v3");
    match parent_state_root {
        Some(root) => {
            hasher.update([1u8]);
            hasher.update(root);
        }
        None => hasher.update([0u8]),
    }
    for count in [
        addition_count,
        removal_count,
        parent_node_value_count,
        parent_predicate_count,
        node_value_count,
        predicate_count,
    ] {
        hasher.update(count.to_be_bytes());
    }
    for commitment in column_commitments.iter().chain([
        &predicate_dict_commitments.0,
        &predicate_dict_commitments.1,
        &node_value_dict_commitments.0,
        &node_value_dict_commitments.1,
        &node_value_dict_range_commitments.0,
        &node_value_dict_range_commitments.1,
    ]) {
        let bytes = serialize_commitment(commitment)?;
        hasher.update((bytes.len() as u64).to_be_bytes());
        hasher.update(bytes);
    }
    Ok(hasher.finalize().into())
}

pub(crate) fn decode_legacy_v3(bytes: &[u8]) -> io::Result<LegacyV3Commitment> {
    if bytes.len() < COMMITMENT_MAGIC.len() + 4 + 64 || bytes.len() > MAX_PROOF_LAYER_BYTES {
        return Err(invalid("truncated legacy v3 proof commitment"));
    }
    let (payload, checksum) = bytes.split_at(bytes.len() - 32);
    if file_checksum(payload).as_slice() != checksum {
        return Err(invalid("legacy v3 commitment file checksum mismatch"));
    }
    let (body, root_bytes) = payload.split_at(payload.len() - 32);
    let commitment_root: ProofHash = root_bytes.try_into().unwrap();
    let mut r = Reader::new(body);
    if r.take(COMMITMENT_MAGIC.len())? != COMMITMENT_MAGIC || r.u32()? != 3 {
        return Err(invalid("not a PF4 v3 commitment file"));
    }
    let parent_state_root = match r.u8()? {
        0 => None,
        1 => Some(r.hash()?),
        _ => return Err(invalid("bad legacy v3 parent root tag")),
    };
    let addition_count = r.u64()?;
    let removal_count = r.u64()?;
    let parent_node_value_count = r.u64()?;
    let parent_predicate_count = r.u64()?;
    let node_value_count = r.u64()?;
    let predicate_count = r.u64()?;
    let additions = read_records(&mut r, PersistedTableKind::Delta)?;
    let removals = read_records(&mut r, PersistedTableKind::Delta)?;
    let predicate_records = read_records(&mut r, PersistedTableKind::PredicateCatalog)?;
    let object_records = read_records(&mut r, PersistedTableKind::ObjectCatalog)?;
    let mut column_commitments = [<ArkGT as Group>::identity(); COLUMN_COUNT];
    for commitment in &mut column_commitments {
        *commitment = deserialize_commitment(&r.bytes()?)?;
    }
    let predicate_dict_commitments = (
        deserialize_commitment(&r.bytes()?)?,
        deserialize_commitment(&r.bytes()?)?,
    );
    let node_value_dict_commitments = (
        deserialize_commitment(&r.bytes()?)?,
        deserialize_commitment(&r.bytes()?)?,
    );
    let node_value_dict_range_commitments = (
        deserialize_commitment(&r.bytes()?)?,
        deserialize_commitment(&r.bytes()?)?,
    );
    if !r.done() {
        return Err(invalid("trailing legacy v3 commitment bytes"));
    }
    let predicate_mappings = predicate_records
        .into_iter()
        .map(|record| {
            if record.subject_id != 0
                || record.predicate_id == 0
                || record.object_id != 0
                || !record.triple.subject.is_empty()
                || record.triple.object != CanonicalObject::Node(Vec::new())
            {
                return Err(invalid("malformed legacy v3 predicate catalog"));
            }
            Ok((record.triple.predicate, record.predicate_id))
        })
        .collect::<io::Result<Vec<_>>>()?;
    let node_value_mappings = object_records
        .into_iter()
        .map(|record| {
            if record.subject_id != 0
                || record.predicate_id != 0
                || record.object_id == 0
                || !record.triple.subject.is_empty()
                || !record.triple.predicate.is_empty()
            {
                return Err(invalid("malformed legacy v3 node/value catalog"));
            }
            Ok((record.triple.object, record.object_id))
        })
        .collect::<io::Result<Vec<_>>>()?;
    let [_, predicate_catalog, node_value_catalog] =
        build_dictionary_catalogs_from_maps(node_value_mappings, predicate_mappings)?;
    if additions.len() as u64 != addition_count || removals.len() as u64 != removal_count {
        return Err(invalid("legacy v3 delta count mismatch"));
    }
    if commit_delta_columns(&additions, &removals)? != column_commitments
        || commit_predicate_dictionary(&predicate_catalog)? != predicate_dict_commitments
        || commit_node_value_dictionary(&node_value_catalog)? != node_value_dict_commitments
        || commit_node_value_dict_range(&node_value_catalog)?
            != node_value_dict_range_commitments
    {
        return Err(invalid("legacy v3 raw table commitment mismatch"));
    }
    let recomputed = legacy_v3_root_from_parts(
        parent_state_root,
        addition_count,
        removal_count,
        parent_node_value_count,
        parent_predicate_count,
        node_value_count,
        predicate_count,
        &column_commitments,
        &predicate_dict_commitments,
        &node_value_dict_commitments,
        &node_value_dict_range_commitments,
    )?;
    if recomputed != commitment_root {
        return Err(invalid("legacy v3 commitment root mismatch"));
    }
    Ok(LegacyV3Commitment {
        parent_state_root,
        commitment_root,
        addition_count,
        removal_count,
        parent_node_value_count,
        parent_predicate_count,
        node_value_count,
        predicate_count,
        additions,
        removals,
        predicate_catalog,
        node_value_catalog,
    })
}

