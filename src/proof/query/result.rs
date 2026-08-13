//! Turning proven Store-id result columns back into human-meaningful values.
//!
//! A BGP proof authenticates result columns as columns of dictionary IDs (plus a Dory
//! commitment per column). This module is the trusted bridge from those IDs back to the
//! actual IRIs / typed literals / generated scalars a caller wants to read. Every function
//! here re-opens the layer's dictionaries against the externally trusted `commitment_root`
//! before trusting the clear catalog, so resolution is itself an authenticated step and
//! never a blind table lookup. It also decides each column's ID namespace from the plan
//! (predicate vs node/value vs generated scalar) so two IDs that happen to share a number
//! are never confused.

use super::*;

/// Bind caller-delivered result ID columns to the commitments authenticated by a query
/// proof. This is intentionally deterministic: pad each real column to the proven domain
/// and recompute its Dory commitment. It adds work linear in the delivered result size,
/// but no new proof-generation lifecycle or trusted component.
pub fn verify_result_columns(
    commitments: &[ArkGT],
    columns: &[Vec<ArkFr>],
    result_len: usize,
    result_log_domain: usize,
) -> io::Result<()> {
    if result_log_domain == 0 || result_log_domain >= usize::BITS as usize {
        return Err(invalid("result column log-domain out of range"));
    }
    let domain = 1usize << result_log_domain;
    if result_len > domain || commitments.len() != columns.len() {
        return Err(invalid(
            "result column shape does not match its commitments",
        ));
    }
    for (commitment, column) in commitments.iter().zip(columns) {
        if column.len() != result_len {
            return Err(invalid(
                "result column length does not match the proven length",
            ));
        }
        let mut padded = column.clone();
        padded.resize(domain, ArkFr::zero());
        if commit_column(&padded)? != *commitment {
            return Err(invalid(
                "delivered result column does not match its commitment",
            ));
        }
    }
    Ok(())
}

/// Resolve proven Store ID columns through the layer's authenticated node/value
/// dictionary. The clear catalog in [`LayerCommitment`] is a verifier-side opening of
/// the root-bound dictionary commitments, so it is not trusted until both its `(hash,
/// id)` columns have been rebuilt and recommitted here.
pub fn resolve_node_value_result_objects(
    source: &impl VerifierLayer,
    expected_root: ProofHash,
    columns: &[Vec<ArkFr>],
) -> io::Result<Vec<Vec<CanonicalObject>>> {
    let layer = source.verifier_view();
    if layer.state.commitment_root != expected_root {
        return Err(invalid(
            "layer commitment_root does not match the externally-trusted expected root",
        ));
    }
    layer.validate()?;

    let (hashes, ids) = layer.node_value_dictionary_columns()?;
    if commit_column(&hashes)? != layer.node_value_dict_commitments.0
        || commit_column(&ids)? != layer.node_value_dict_commitments.1
    {
        return Err(invalid(
            "node/value catalog does not open the root-bound dictionary commitments",
        ));
    }

    columns
        .iter()
        .map(|column| {
            column
                .iter()
                .map(|id| {
                    layer
                        .node_value_catalog
                        .iter()
                        .find(|record| ArkFr::from_u64(record.object_id) == *id)
                        .map(|record| record.triple.object.clone())
                        .ok_or_else(|| {
                            invalid("result id is absent from the node/value dictionary")
                        })
                })
                .collect()
        })
        .collect()
}

/// Resolve result columns through their explicitly declared, plan-derived Store ID
/// namespaces. Predicate IDs are opened only against the root-bound predicate catalog;
/// subject/object IDs only against the node/value catalog.
pub fn resolve_typed_result_objects(
    layer: &impl VerifierLayer,
    expected_root: ProofHash,
    columns: &[Vec<ArkFr>],
    namespaces: &[ResultIdNamespace],
) -> io::Result<Vec<Vec<CanonicalObject>>> {
    let descriptors: Vec<_> = namespaces
        .iter()
        .map(|namespace| ResultColumnDescriptor {
            namespace: *namespace,
            nullable: false,
        })
        .collect();
    resolve_result_values(layer, expected_root, columns, &descriptors)?
        .into_iter()
        .map(|column| {
            column
                .into_iter()
                .map(|value| match value {
                    CanonicalResultValue::Object(object) => Ok(object),
                    CanonicalResultValue::Null => {
                        Err(invalid("null in a nonnullable result column"))
                    }
                    CanonicalResultValue::GeneratedDecimalU64(_) => Err(invalid(
                        "generated scalar cannot resolve as a Store dictionary object",
                    )),
                    CanonicalResultValue::GeneratedDecimalI64(_) => Err(invalid(
                        "generated scalar cannot resolve as a Store dictionary object",
                    )),
                })
                .collect()
        })
        .collect()
}

/// Resolve nullable typed result columns. Every real Store dictionary ID must be
/// nonzero. Zero resolves to `Null` exactly for a descriptor marked nullable; it is an
/// error in a nonnullable column and is never looked up in either dictionary catalog.
pub fn resolve_result_values(
    source: &impl VerifierLayer,
    expected_root: ProofHash,
    columns: &[Vec<ArkFr>],
    descriptors: &[ResultColumnDescriptor],
) -> io::Result<Vec<Vec<CanonicalResultValue>>> {
    let layer = source.verifier_view();
    if columns.len() != descriptors.len() {
        return Err(invalid("result column descriptor shape mismatch"));
    }
    if layer.state.commitment_root != expected_root {
        return Err(invalid(
            "layer commitment_root does not match the externally-trusted expected root",
        ));
    }
    layer.validate()?;
    if layer
        .predicate_catalog
        .iter()
        .any(|record| record.predicate_id == 0)
        || layer
            .node_value_catalog
            .iter()
            .any(|record| record.object_id == 0)
    {
        return Err(invalid(
            "Store dictionary id zero is reserved for nullable query results",
        ));
    }

    columns
        .iter()
        .zip(descriptors)
        .map(|(column, descriptor)| {
            column
                .iter()
                .map(|id| {
                    if descriptor.namespace == ResultIdNamespace::GeneratedDecimalU64 {
                        let value = id.0.into_bigint().as_ref()[0];
                        if ArkFr::from_u64(value) != *id {
                            return Err(invalid(
                                "generated decimal result is outside canonical u64 range",
                            ));
                        }
                        return Ok(CanonicalResultValue::GeneratedDecimalU64(value));
                    }
                    if descriptor.namespace == ResultIdNamespace::GeneratedDecimalI64 {
                        return Ok(CanonicalResultValue::GeneratedDecimalI64(
                            generated_decimal_i64_value(*id)?,
                        ));
                    }
                    if id.is_zero() {
                        return if descriptor.nullable {
                            Ok(CanonicalResultValue::Null)
                        } else {
                            Err(invalid("null id in a nonnullable result column"))
                        };
                    }
                    let object = match descriptor.namespace {
                        ResultIdNamespace::NodeValue => layer
                            .node_value_catalog
                            .iter()
                            .find(|record| ArkFr::from_u64(record.object_id) == *id)
                            .map(|record| record.triple.object.clone())
                            .ok_or_else(|| {
                                invalid("result id is absent from the node/value dictionary")
                            })?,
                        ResultIdNamespace::Predicate => layer
                            .predicate_catalog
                            .iter()
                            .find(|record| ArkFr::from_u64(record.predicate_id) == *id)
                            .map(|record| CanonicalObject::Node(record.triple.predicate.clone()))
                            .ok_or_else(|| {
                                invalid("result id is absent from the predicate dictionary")
                            })?,
                        ResultIdNamespace::GeneratedDecimalU64 => unreachable!(),
                        ResultIdNamespace::GeneratedDecimalI64 => unreachable!(),
                    };
                    Ok(CanonicalResultValue::Object(object))
                })
                .collect()
        })
        .collect()
}

pub fn result_column_descriptors(
    plan: &LayerBgpPlan,
) -> io::Result<Vec<ResultColumnDescriptor>> {
    Ok(result_id_namespaces(plan)?
        .into_iter()
        .map(|namespace| ResultColumnDescriptor {
            namespace,
            nullable: false,
        })
        .collect())
}

/// Derive the namespace of every final result column while validating that each natural
/// join compares IDs from the same Store dictionary namespace.
pub fn result_id_namespaces(plan: &LayerBgpPlan) -> io::Result<Vec<ResultIdNamespace>> {
    fn atom_namespaces(atom: &LayerBgpAtom) -> Vec<ResultIdNamespace> {
        let mut out = Vec::new();
        if matches!(atom.subject, AtomTerm::Variable) {
            out.push(ResultIdNamespace::NodeValue);
        }
        if atom.predicate_variable {
            out.push(ResultIdNamespace::Predicate);
        }
        if matches!(atom.object, AtomTerm::Variable) && !atom.subject_object_equal {
            out.push(ResultIdNamespace::NodeValue);
        }
        out
    }

    let relational = relational_atom_count(plan);
    if relational == 0 {
        return Ok(Vec::new());
    }
    let mut acc = atom_namespaces(&plan.atoms[0]);
    for (step, atom) in plan.joins.iter().zip(&plan.atoms[1..relational]) {
        let rel = atom_namespaces(atom);
        if step.is_cross_product() {
            acc.extend(rel);
            continue;
        }
        let mut pairs = vec![(step.acc_key, step.rel_key)];
        pairs.extend(step.additional_keys.iter().copied());
        for (left, right) in &pairs {
            if *left >= acc.len() || *right >= rel.len() {
                return Err(invalid("BGP join namespace column index out of range"));
            }
            if acc[*left] != rel[*right] {
                return Err(invalid(
                    "BGP join compares predicate and node/value ID namespaces",
                ));
            }
        }
        if step.additional_keys.is_empty() {
            let key = acc.remove(step.acc_key);
            let mut rel_other = rel;
            rel_other.remove(step.rel_key);
            acc.push(key);
            acc.extend(rel_other);
        } else {
            for (index, namespace) in rel.into_iter().enumerate() {
                if !pairs.iter().any(|(_, right)| *right == index) {
                    acc.push(namespace);
                }
            }
        }
    }
    Ok(acc)
}

/// Verify caller-delivered canonical IRIs/typed literals against proven Store ID
/// columns and the root-authenticated node/value dictionary.
pub fn verify_node_value_result_objects(
    layer: &impl VerifierLayer,
    expected_root: ProofHash,
    columns: &[Vec<ArkFr>],
    objects: &[Vec<CanonicalObject>],
) -> io::Result<()> {
    let resolved = resolve_node_value_result_objects(layer, expected_root, columns)?;
    if resolved != objects {
        return Err(invalid(
            "delivered result objects do not match the authenticated Store IDs",
        ));
    }
    Ok(())
}
