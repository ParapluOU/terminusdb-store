//! Building a layer's committed columns.
//!
//! A committed layer is, underneath, a handful of columns of numbers: the dictionary ids of
//! its added and removed triples (subject, predicate, object), the ids of its full
//! materialized head, and its two dictionaries (predicate, and node/value) expressed as
//! parallel hash/id columns plus the extra value/datatype columns the range gadget needs.
//! This module is the machinery that turns those clear tables into Dory column commitments —
//! the exact succinct values the layer's `commitment_root` binds. Everything here is
//! deterministic: the same tables always produce the same commitments, so any node can
//! recompute and check them. It also holds [`reject_nan_catalog`], the guard that keeps IEEE
//! NaNs out of a catalog before its total-order encoding is (mis)used as comparison order.
//!
//! A [Dory column commitment](crate::proof::primer) (primer §3–§4) is the short,
//! binding fingerprint each function here produces; the same tables always yield
//! the same commitment.

use super::*;

/// Commit to one dictionary-id column via Dory, returning its succinct GT
/// (tier-2) commitment. Empty columns get a canonical deterministic empty
/// commitment (the GT identity).
///
/// The column is padded to a power-of-two length `2^m`; the matrix layout is
/// `sigma = ceil(m/2)`, `nu = m - sigma` (so `nu <= sigma`, matching Dory's
/// non-square constraint, and `2^sigma <= 2^ceil(SETUP_MAX_LOG_N/2)` fits the
/// global generator vectors). In Transparent mode the commitment carries no
/// blinding, so it is a deterministic function of the ids and the setup.
pub(super) fn commit_record_id_column_with_context(
    context: &crate::proof::argument::StoreProverContext<'_>,
    records: &[ProofRecord],
    select: fn(&ProofRecord) -> u64,
) -> io::Result<ArkGT> {
    if records.is_empty() {
        return Ok(<ArkGT as Group>::identity());
    }
    let padded_len = records
        .len()
        .checked_next_power_of_two()
        .ok_or_else(|| invalid("proof column length overflow"))?;
    let m = padded_len.trailing_zeros() as usize;
    setup_tier_for_num_vars(m)?;
    let mut coefficients = vec![<ArkFr as DoryField>::zero(); padded_len];
    for (slot, record) in coefficients.iter_mut().zip(records) {
        *slot = <ArkFr as DoryField>::from_u64(select(record));
    }
    context.commit_column(&coefficients)
}

/// Commit an already-built, power-of-two-padded field column via Dory. An empty
/// column gets the canonical empty commitment (GT identity). Used for the
/// dictionary hash/id/value/datatype columns, which are assembled as field
/// vectors rather than selected from record ids.
pub(super) fn commit_owned_field_column_with_context(
    context: &crate::proof::argument::StoreProverContext<'_>,
    coefficients: Vec<ArkFr>,
) -> io::Result<ArkGT> {
    if coefficients.is_empty() {
        return Ok(<ArkGT as Group>::identity());
    }
    if !coefficients.len().is_power_of_two() {
        return Err(invalid("committed field column is not power-of-two padded"));
    }
    context.commit_column(&coefficients)
}

/// Build the six per-column commitments for a layer delta, in fixed order:
/// additions (subject, predicate, object) then removals (subject, predicate,
/// object). Records are expected pre-sorted (SPO) so the columns are canonical.
// TODO(proof-of-woql): Phase 1c commits each column at its global absolute row
// offset and composes commitments across the layer chain; here every layer's
// delta is committed standalone at offset 0.
pub(super) fn commit_delta_columns(
    additions: &[ProofRecord],
    removals: &[ProofRecord],
) -> io::Result<[ArkGT; COLUMN_COUNT]> {
    commit_delta_columns_with_context(
        &crate::proof::argument::StoreProverContext::cpu(), additions, removals,
    )
}

/// [`commit_delta_columns`] with explicitly selected prover arithmetic (e.g. a
/// Blitzar-accelerated context). Arithmetic-only: the resulting commitments are
/// identical regardless of context.
pub(super) fn commit_delta_columns_with_context(
    context: &crate::proof::argument::StoreProverContext<'_>,
    additions: &[ProofRecord],
    removals: &[ProofRecord],
) -> io::Result<[ArkGT; COLUMN_COUNT]> {
    Ok([
        commit_record_id_column_with_context(context, additions, |record| record.subject_id)?,
        commit_record_id_column_with_context(context, additions, |record| record.predicate_id)?,
        commit_record_id_column_with_context(context, additions, |record| record.object_id)?,
        commit_record_id_column_with_context(context, removals, |record| record.subject_id)?,
        commit_record_id_column_with_context(context, removals, |record| record.predicate_id)?,
        commit_record_id_column_with_context(context, removals, |record| record.object_id)?,
    ])
}

/// Build the three materialized-head column commitments (S,P,O) for a
/// base/rollup layer, from its complete logical triple set (pre-sorted SPO).
pub(super) fn commit_materialized_columns(
    records: &[ProofRecord],
) -> io::Result<[ArkGT; MATERIALIZED_COLUMN_COUNT]> {
    commit_materialized_columns_with_context(
        &crate::proof::argument::StoreProverContext::cpu(), records,
    )
}

/// [`commit_materialized_columns`] with explicitly selected prover arithmetic
/// (arithmetic-only; identical commitments).
pub(super) fn commit_materialized_columns_with_context(
    context: &crate::proof::argument::StoreProverContext<'_>,
    records: &[ProofRecord],
) -> io::Result<[ArkGT; MATERIALIZED_COLUMN_COUNT]> {
    Ok([
        commit_record_id_column_with_context(context, records, |record| record.subject_id)?,
        commit_record_id_column_with_context(context, records, |record| record.predicate_id)?,
        commit_record_id_column_with_context(context, records, |record| record.object_id)?,
    ])
}

/// Build the two predicate-dictionary columns from the predicate catalog
/// (`term -> id`, canonical order): `hash[k] = hash_term(predicate_bytes_k)`,
/// `id[k] = predicate_id_k`, padded with zeros to a power of two of length
/// `>= 2` (the Q3 membership sumcheck needs at least one variable). An empty
/// dictionary yields empty columns. This is the single source of truth used
/// both to COMMIT the dictionary (here) and to OPEN it (via
/// `LayerCommitment::predicate_dictionary_columns`), so the two align.
pub(super) fn predicate_dict_columns_from_catalog(
    predicate_catalog: &[ProofRecord],
) -> io::Result<(Vec<ArkFr>, Vec<ArkFr>)> {
    let d = predicate_catalog.len();
    if d == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let (padded, _) = padded_domain_for_rows(d)?;
    let zero = <ArkFr as DoryField>::zero();
    let mut hashes = vec![zero; padded];
    let mut ids = vec![zero; padded];
    for (k, record) in predicate_catalog.iter().enumerate() {
        hashes[k] = crate::proof::dict::hash_term(&record.triple.predicate);
        ids[k] = <ArkFr as DoryField>::from_u64(record.predicate_id);
    }
    Ok((hashes, ids))
}

/// Commit the two predicate-dictionary columns via the SAME shared query-side
/// `commit_column` the Q3 gadget opens, so the committed columns align with
/// what a membership proof opens. An empty dictionary gets identity commitments.
pub(super) fn commit_predicate_dictionary(
    predicate_catalog: &[ProofRecord],
) -> io::Result<(ArkGT, ArkGT)> {
    commit_predicate_dictionary_with_context(
        &crate::proof::argument::StoreProverContext::cpu(), predicate_catalog,
    )
}

/// [`commit_predicate_dictionary`] with explicitly selected prover arithmetic
/// (arithmetic-only; identical commitments).
pub(super) fn commit_predicate_dictionary_with_context(
    context: &crate::proof::argument::StoreProverContext<'_>,
    predicate_catalog: &[ProofRecord],
) -> io::Result<(ArkGT, ArkGT)> {
    if predicate_catalog.is_empty() {
        let identity = <ArkGT as Group>::identity();
        return Ok((identity, identity));
    }
    let (padded, _) = padded_domain_for_rows(predicate_catalog.len())?;
    let mut hashes = vec![<ArkFr as DoryField>::zero(); padded];
    for (slot, record) in hashes.iter_mut().zip(predicate_catalog) {
        *slot = crate::proof::dict::hash_term(&record.triple.predicate);
    }
    let hash_commitment = commit_owned_field_column_with_context(context, hashes)?;
    let mut ids = vec![<ArkFr as DoryField>::zero(); padded];
    for (slot, record) in ids.iter_mut().zip(predicate_catalog) {
        *slot = <ArkFr as DoryField>::from_u64(record.predicate_id);
    }
    Ok((hash_commitment, commit_owned_field_column_with_context(context, ids)?))
}

/// Build the two node/value-dictionary columns from the node/value catalog
/// (`object -> id`, canonical order): `hash[k] = hash_object(object_k)`,
/// `id[k] = object_id_k`, padded with zeros to a power of two of length `>= 2`
/// (the Q3 membership sumcheck needs at least one variable). Mirrors
/// `predicate_dict_columns_from_catalog` exactly but hashes the OBJECT (a
/// `Node` IRI or a typed `Value` literal) via the domain-tagged `hash_object`.
/// Subjects share this id space (they are the `Node(_)` subset), so a subject
/// constant resolves through the very same `(hash_object(Node(iri)), id)` row.
/// Single source of truth for both COMMIT (here) and OPEN (via
/// `LayerCommitment::node_value_dictionary_columns`), so the two align.
pub(super) fn node_value_dict_columns_from_catalog(
    node_value_catalog: &[ProofRecord],
) -> io::Result<(Vec<ArkFr>, Vec<ArkFr>)> {
    let padded = node_value_dict_padded_len(node_value_catalog.len())?;
    if padded == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let zero = <ArkFr as DoryField>::zero();
    let mut hashes = vec![zero; padded];
    let mut ids = vec![zero; padded];
    for (k, record) in node_value_catalog.iter().enumerate() {
        hashes[k] = crate::proof::dict::hash_object(&record.triple.object);
        ids[k] = <ArkFr as DoryField>::from_u64(record.object_id);
    }
    Ok((hashes, ids))
}

/// Commit the two node/value-dictionary columns via the SAME shared query-side
/// `commit_column` the Q3 gadget opens, so the committed columns align with
/// what a membership proof opens. An empty dictionary gets identity commitments.
pub(super) fn commit_node_value_dictionary(
    node_value_catalog: &[ProofRecord],
) -> io::Result<(ArkGT, ArkGT)> {
    commit_node_value_dictionary_with_context(
        &crate::proof::argument::StoreProverContext::cpu(), node_value_catalog,
    )
}

/// [`commit_node_value_dictionary`] with explicitly selected prover arithmetic
/// (arithmetic-only; identical commitments).
pub(super) fn commit_node_value_dictionary_with_context(
    context: &crate::proof::argument::StoreProverContext<'_>,
    node_value_catalog: &[ProofRecord],
) -> io::Result<(ArkGT, ArkGT)> {
    let padded = node_value_dict_padded_len(node_value_catalog.len())?;
    if padded == 0 {
        let identity = <ArkGT as Group>::identity();
        return Ok((identity, identity));
    }
    let mut hashes = vec![<ArkFr as DoryField>::zero(); padded];
    for (slot, record) in hashes.iter_mut().zip(node_value_catalog) {
        *slot = crate::proof::dict::hash_object(&record.triple.object);
    }
    let hash_commitment = commit_owned_field_column_with_context(context, hashes)?;
    let mut ids = vec![<ArkFr as DoryField>::zero(); padded];
    for (slot, record) in ids.iter_mut().zip(node_value_catalog) {
        *slot = <ArkFr as DoryField>::from_u64(record.object_id);
    }
    Ok((hash_commitment, commit_owned_field_column_with_context(context, ids)?))
}

/// Whether `object` is an IEEE NaN of the queried float datatype (`datatype_tag` must be
/// the `Float32`/`Float64` tag; any other tag returns `false`). Shared by
/// [`reject_nan_catalog`] (term-bearing, whole catalog) and [`reject_nan_objects`]
/// (Merkle-dict I2b step-3 TERM-LESS, the opened touched terms).
fn object_is_selected_nan(object: &CanonicalObject, datatype_tag: ArkFr) -> bool {
    let f32_tag = ArkFr::from_u64(CanonicalDatatype::Float32 as u64);
    let f64_tag = ArkFr::from_u64(CanonicalDatatype::Float64 as u64);
    match object {
        CanonicalObject::Value {
            datatype: CanonicalDatatype::Float32,
            lexical,
        } if datatype_tag == f32_tag && lexical.len() == 4 => {
            let encoded = u32::from_be_bytes(lexical.as_slice().try_into().unwrap());
            let bits = if encoded & 0x8000_0000 != 0 {
                encoded ^ 0x8000_0000
            } else {
                encoded ^ 0xffff_ffff
            };
            f32::from_bits(bits).is_nan()
        }
        CanonicalObject::Value {
            datatype: CanonicalDatatype::Float64,
            lexical,
        } if datatype_tag == f64_tag && lexical.len() == 8 => {
            let encoded = u64::from_be_bytes(lexical.as_slice().try_into().unwrap());
            let bits = if encoded & 0x8000_0000_0000_0000 != 0 {
                encoded ^ 0x8000_0000_0000_0000
            } else {
                encoded ^ 0xffff_ffff_ffff_ffff
            };
            f64::from_bits(bits).is_nan()
        }
        _ => false,
    }
}

/// Reject IEEE NaNs before using the Store's total-order lexical encoding as
/// WOQL numeric comparison order. Prolog's `X < Y` follows IEEE unordered-NaN
/// semantics, whereas the storage encoding deliberately assigns every NaN a
/// total-order position. A range/group proof over a catalog containing a NaN
/// of the selected float datatype would therefore prove different semantics.
pub(crate) fn reject_nan_catalog(
    catalog: &[ProofRecord],
    datatype_tag: ArkFr,
) -> io::Result<()> {
    let f32_tag = ArkFr::from_u64(CanonicalDatatype::Float32 as u64);
    let f64_tag = ArkFr::from_u64(CanonicalDatatype::Float64 as u64);
    if datatype_tag != f32_tag && datatype_tag != f64_tag {
        return Ok(());
    }
    for record in catalog {
        if object_is_selected_nan(&record.triple.object, datatype_tag) {
            return Err(invalid(
                "float range/group proof cannot match Prolog semantics in a catalog containing NaN",
            ));
        }
    }
    Ok(())
}

/// Merkle-dict I2b step 3 (TERM-LESS): the NaN guard over the OPENED touched terms rather
/// than the (dropped) clear catalog. Only touched rows affect a range/group answer, so
/// rejecting a queried-float NaN among the authenticated touched terms is sufficient to
/// keep the total-order encoding faithful to Prolog's unordered-NaN semantics. A non-float
/// `datatype_tag` is a no-op (mirrors [`reject_nan_catalog`]).
pub(crate) fn reject_nan_objects(
    objects: &[CanonicalObject],
    datatype_tag: ArkFr,
) -> io::Result<()> {
    let f32_tag = ArkFr::from_u64(CanonicalDatatype::Float32 as u64);
    let f64_tag = ArkFr::from_u64(CanonicalDatatype::Float64 as u64);
    if datatype_tag != f32_tag && datatype_tag != f64_tag {
        return Ok(());
    }
    for object in objects {
        if object_is_selected_nan(object, datatype_tag) {
            return Err(invalid(
                "float range proof cannot match Prolog semantics: a touched term is NaN",
            ));
        }
    }
    Ok(())
}

/// Build the node/value-dictionary VALUE column (one entry per catalog row, in
/// the SAME canonical order/padding as `node_value_dict_columns_from_catalog`):
/// `value[k] = node_value_op_value(object_k)` — the order-preserving numeric
/// encoding for supported fixed-width numeric literals, `0` otherwise. Padding
/// rows carry `0` (as `hash`/`id` do; they have `id = 0` so any real-row check
/// excludes them). Single source of truth for both COMMIT and OPEN.
pub(super) fn node_value_dict_value_column(node_value_catalog: &[ProofRecord]) -> io::Result<Vec<ArkFr>> {
    let padded = node_value_dict_padded_len(node_value_catalog.len())?;
    let mut values = vec![<ArkFr as DoryField>::zero(); padded];
    for (k, record) in node_value_catalog.iter().enumerate() {
        values[k] = node_value_op_value(&record.triple.object)?;
    }
    Ok(values)
}

/// Build the node/value-dictionary DATATYPE column (one entry per catalog row,
/// same order/padding): `datatype[k]` is the `CanonicalDatatype` discriminant
/// for a `Value` object, or the reserved `NODE_DATATYPE_TAG` for a `Node`. Lets
/// the later range gadget mask rows to the queried datatype by equality. Padding
/// rows carry `0` (mirrors `hash`/`id`); tag `0` is `String`, which is not a
/// range-supported type, and padding rows have `id = 0`, so padding never
/// false-matches a numeric range query. Single source of truth for COMMIT/OPEN.
pub(super) fn node_value_dict_datatype_column(
    node_value_catalog: &[ProofRecord],
) -> io::Result<Vec<ArkFr>> {
    let padded = node_value_dict_padded_len(node_value_catalog.len())?;
    let mut datatypes = vec![<ArkFr as DoryField>::zero(); padded];
    for (k, record) in node_value_catalog.iter().enumerate() {
        let tag = match &record.triple.object {
            CanonicalObject::Value { datatype, .. } => *datatype as u8 as u64,
            CanonicalObject::Node(_) => NODE_DATATYPE_TAG,
        };
        datatypes[k] = <ArkFr as DoryField>::from_u64(tag);
    }
    Ok(datatypes)
}

/// Commit the node/value VALUE and DATATYPE columns via the SAME shared
/// query-side `commit_column` the range/membership gadgets open, so the
/// committed columns align with what a proof opens. An empty dictionary gets
/// identity commitments (mirrors `commit_node_value_dictionary`).
pub(super) fn commit_node_value_dict_range(
    node_value_catalog: &[ProofRecord],
) -> io::Result<(ArkGT, ArkGT)> {
    commit_node_value_dict_range_with_context(
        &crate::proof::argument::StoreProverContext::cpu(), node_value_catalog,
    )
}

/// [`commit_node_value_dict_range`] with explicitly selected prover arithmetic
/// (arithmetic-only; identical commitments).
pub(super) fn commit_node_value_dict_range_with_context(
    context: &crate::proof::argument::StoreProverContext<'_>,
    node_value_catalog: &[ProofRecord],
) -> io::Result<(ArkGT, ArkGT)> {
    let values = node_value_dict_value_column(node_value_catalog)?;
    if values.is_empty() {
        let identity = <ArkGT as Group>::identity();
        return Ok((identity, identity));
    }
    let value_commitment = commit_owned_field_column_with_context(context, values)?;
    let datatypes = node_value_dict_datatype_column(node_value_catalog)?;
    Ok((
        value_commitment,
        commit_owned_field_column_with_context(context, datatypes)?,
    ))
}

