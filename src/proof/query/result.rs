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
/// dictionary, via id-indexed Merkle openings (I1). β re-anchor: the EXTERNAL trust anchor
/// is now the `resolution_root` (which binds `commitment_root` AND `nv_merkle_root` AND
/// `pred_merkle_root`), so the Merkle root this resolution opens against is bound DIRECTLY
/// by the trusted anchor, not merely transitively via the catalog. `validate()` then
/// authenticates the clear catalog AND recomputes the root-bound `nv_merkle_root` (a
/// tampered catalog or root fails closed there); each delivered id is then opened
/// positionally against that root, so resolution is an authenticated `O(k·log N)` step
/// rather than a linear catalog scan.
pub fn resolve_node_value_result_objects(
    source: &impl VerifierLayer,
    trusted_resolution_root: ProofHash,
    columns: &[Vec<ArkFr>],
) -> io::Result<Vec<Vec<CanonicalObject>>> {
    // No openings: term-bearing resolves via the rebuilt tree; a term-less view fails closed.
    resolve_node_value_result_objects_impl(source, trusted_resolution_root, columns, None)
}

/// Merkle-dict I2b S3: resolve node/value result ids over a TERM-LESS artifact via
/// prover-shipped Merkle `openings` (the dropped catalog can no longer rebuild the tree). The
/// term-BEARING path is unchanged (it ignores `openings` and rebuilds the tree). `#[cfg(test)]`:
/// in S3 the term-less resolve ENTRY POINT exists only in test builds — it must NOT be an
/// external end-to-end query path until S4's per-constant belt-hash gates term-less BGP
/// verification (else the I2b object-constant forge). S5 un-gates the shipping wiring.
#[cfg(test)]
pub(crate) fn resolve_node_value_result_objects_with_openings(
    source: &impl VerifierLayer,
    trusted_resolution_root: ProofHash,
    columns: &[Vec<ArkFr>],
    openings: &ResolutionOpenings,
) -> io::Result<Vec<Vec<CanonicalObject>>> {
    resolve_node_value_result_objects_impl(source, trusted_resolution_root, columns, Some(openings))
}

fn resolve_node_value_result_objects_impl(
    source: &impl VerifierLayer,
    trusted_resolution_root: ProofHash,
    columns: &[Vec<ArkFr>],
    openings: Option<&ResolutionOpenings>,
) -> io::Result<Vec<Vec<CanonicalObject>>> {
    let layer = source.verifier_view();
    if layer.resolution_root() != trusted_resolution_root {
        return Err(invalid(
            "layer resolution_root does not match the externally-trusted resolution root",
        ));
    }
    layer.validate()?;
    // TERM-LESS (S3): DELIBERATELY route through the prover openings — never the empty-catalog
    // rebuild path (which would fail-close only incidentally). A missing openings carrier here
    // is an explicit fail-close; a missing per-id opening fails inside the resolver.
    if !layer.terms_present && openings.is_none() {
        return Err(invalid(
            "term-less node/value resolution requires prover-shipped Merkle openings",
        ));
    }
    // Term-bearing rebuilds+root-checks the tree once; term-less leaves it `None` and opens
    // the prover-shipped openings instead.
    let nv_tree = if layer.terms_present {
        Some(node_value_merkle_tree(layer)?)
    } else {
        None
    };

    columns
        .iter()
        .map(|column| {
            column
                .iter()
                .map(|id| resolve_node_value_id_via_merkle(id, layer, nv_tree.as_ref(), openings))
                .collect()
        })
        .collect()
}

/// The canonical u64 an id field element must equal to index a dictionary. Rejects a
/// field element that is not a small (`< 2^64`) integer, exactly as the generated-decimal
/// path does before trusting a scalar.
pub(super) fn canonical_dictionary_id(id: &ArkFr) -> io::Result<u64> {
    let raw = id.0.into_bigint().as_ref()[0];
    if ArkFr::from_u64(raw) != *id {
        return Err(invalid("dictionary result id is outside canonical u64 range"));
    }
    Ok(raw)
}

/// Build the node/value Merkle tree and confirm it reproduces the root-bound
/// `nv_merkle_root` (already authenticated against `commitment_root` by `validate`).
fn node_value_merkle_tree(layer: VerifierCommitmentRef<'_>) -> io::Result<DictMerkleTree> {
    let tree = build_node_value_tree(
        layer.node_value_catalog,
        layer.state.parent_node_value_count,
        layer.state.node_value_count,
    )?;
    if tree.root() != layer.nv_merkle_root {
        return Err(invalid(
            "rebuilt node/value merkle root does not match the root-bound root",
        ));
    }
    Ok(tree)
}

/// Build the predicate Merkle tree and confirm it reproduces the root-bound
/// `pred_merkle_root`.
fn predicate_merkle_tree(layer: VerifierCommitmentRef<'_>) -> io::Result<DictMerkleTree> {
    let tree = build_predicate_tree(
        layer.predicate_catalog,
        layer.state.parent_predicate_count,
        layer.state.predicate_count,
    )?;
    if tree.root() != layer.pred_merkle_root {
        return Err(invalid(
            "rebuilt predicate merkle root does not match the root-bound root",
        ));
    }
    Ok(tree)
}

/// Open one node/value id against the root-bound `nv_merkle_root`, returning the
/// authenticated term. TERM-BEARING (`layer.terms_present`): open the rebuilt+root-checked
/// `tree` (I1). TERM-LESS (Merkle-dict I2b S3): the catalog is dropped, so `tree` is `None`;
/// consume the prover-shipped `openings` for this id (fail-closed `Err` if absent) and verify
/// it against the SAME root-bound root. Either way `verify_opening` re-forces the interval +
/// structural position against `nv_merkle_root`, so a wrong/tampered opening is rejected.
fn resolve_node_value_id_via_merkle(
    id: &ArkFr,
    layer: VerifierCommitmentRef<'_>,
    tree: Option<&DictMerkleTree>,
    openings: Option<&ResolutionOpenings>,
) -> io::Result<CanonicalObject> {
    let raw = canonical_dictionary_id(id)?;
    let rebuilt;
    let opening: &DictOpening = if layer.terms_present {
        let tree = tree
            .ok_or_else(|| invalid("node/value merkle tree unavailable for resolution"))?;
        rebuilt = tree.open(raw)?;
        &rebuilt
    } else {
        openings
            .ok_or_else(|| {
                invalid("term-less node/value resolution requires prover-shipped Merkle openings")
            })?
            .node_value_opening(raw)?
    };
    verify_opening(
        &layer.nv_merkle_root,
        layer.state.parent_node_value_count,
        layer.state.node_value_count,
        raw,
        opening,
    )
}

/// Open one predicate id against the root-bound `pred_merkle_root`, returning the
/// authenticated predicate IRI as a `Node` (mirroring the legacy predicate resolver).
/// Term-bearing rebuilds+opens `tree`; term-less (S3) consumes the prover-shipped `openings`
/// (fail-closed if absent) — both verified against `pred_merkle_root`.
fn resolve_predicate_id_via_merkle(
    id: &ArkFr,
    layer: VerifierCommitmentRef<'_>,
    tree: Option<&DictMerkleTree>,
    openings: Option<&ResolutionOpenings>,
) -> io::Result<CanonicalObject> {
    let raw = canonical_dictionary_id(id)?;
    let rebuilt;
    let opening: &DictOpening = if layer.terms_present {
        let tree = tree
            .ok_or_else(|| invalid("predicate merkle tree unavailable for resolution"))?;
        rebuilt = tree.open(raw)?;
        &rebuilt
    } else {
        openings
            .ok_or_else(|| {
                invalid("term-less predicate resolution requires prover-shipped Merkle openings")
            })?
            .predicate_opening(raw)?
    };
    verify_opening(
        &layer.pred_merkle_root,
        layer.state.parent_predicate_count,
        layer.state.predicate_count,
        raw,
        opening,
    )
}

/// Resolve result columns through their explicitly declared, plan-derived Store ID
/// namespaces. Predicate IDs are opened only against the root-bound predicate catalog;
/// subject/object IDs only against the node/value catalog.
pub fn resolve_typed_result_objects(
    layer: &impl VerifierLayer,
    trusted_resolution_root: ProofHash,
    columns: &[Vec<ArkFr>],
    namespaces: &[ResultIdNamespace],
) -> io::Result<Vec<Vec<CanonicalObject>>> {
    // No openings: term-bearing resolves via the rebuilt trees; a term-less view fails closed.
    resolve_typed_result_objects_impl(layer, trusted_resolution_root, columns, namespaces, None)
}

/// Merkle-dict I2b S5: resolve typed result columns over a TERM-LESS artifact via the
/// prover-shipped Merkle `openings` (carried in [`LayerBgpProof::openings`]). The term-BEARING
/// path is unchanged (`openings = None`, rebuilds the trees). PRODUCTION (drops the S3
/// `#[cfg(test)]` gate): term-less BGP verification is now gated by S4's MANDATORY per-constant
/// belt-hash, so a term-less proof that reaches this resolver already passed the belt; a missing
/// opening still fails closed here.
pub fn resolve_typed_result_objects_with_openings(
    layer: &impl VerifierLayer,
    trusted_resolution_root: ProofHash,
    columns: &[Vec<ArkFr>],
    namespaces: &[ResultIdNamespace],
    openings: Option<&ResolutionOpenings>,
) -> io::Result<Vec<Vec<CanonicalObject>>> {
    resolve_typed_result_objects_impl(layer, trusted_resolution_root, columns, namespaces, openings)
}

fn resolve_typed_result_objects_impl(
    layer: &impl VerifierLayer,
    trusted_resolution_root: ProofHash,
    columns: &[Vec<ArkFr>],
    namespaces: &[ResultIdNamespace],
    openings: Option<&ResolutionOpenings>,
) -> io::Result<Vec<Vec<CanonicalObject>>> {
    let descriptors: Vec<_> = namespaces
        .iter()
        .map(|namespace| ResultColumnDescriptor {
            namespace: *namespace,
            nullable: false,
        })
        .collect();
    resolve_result_values_impl(layer, trusted_resolution_root, columns, &descriptors, openings)?
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
    trusted_resolution_root: ProofHash,
    columns: &[Vec<ArkFr>],
    descriptors: &[ResultColumnDescriptor],
) -> io::Result<Vec<Vec<CanonicalResultValue>>> {
    // No openings: term-bearing resolves via the rebuilt trees; a term-less view fails closed.
    resolve_result_values_impl(source, trusted_resolution_root, columns, descriptors, None)
}

/// Merkle-dict I2b S5: resolve typed result columns over a TERM-LESS artifact via prover-shipped
/// Merkle `openings` (carried in [`LayerBgpProof::openings`]). The term-BEARING path is unchanged
/// (ignores `openings`, rebuilds the trees). PRODUCTION (drops the S3 `#[cfg(test)]` gate): the
/// term-less resolve path is now gated end-to-end by S4's mandatory per-constant belt-hash in
/// `verify_layer_bgp`; a missing opening fails closed here.
pub fn resolve_result_values_with_openings(
    source: &impl VerifierLayer,
    trusted_resolution_root: ProofHash,
    columns: &[Vec<ArkFr>],
    descriptors: &[ResultColumnDescriptor],
    openings: Option<&ResolutionOpenings>,
) -> io::Result<Vec<Vec<CanonicalResultValue>>> {
    resolve_result_values_impl(source, trusted_resolution_root, columns, descriptors, openings)
}

fn resolve_result_values_impl(
    source: &impl VerifierLayer,
    trusted_resolution_root: ProofHash,
    columns: &[Vec<ArkFr>],
    descriptors: &[ResultColumnDescriptor],
    openings: Option<&ResolutionOpenings>,
) -> io::Result<Vec<Vec<CanonicalResultValue>>> {
    let layer = source.verifier_view();
    if columns.len() != descriptors.len() {
        return Err(invalid("result column descriptor shape mismatch"));
    }
    if layer.resolution_root() != trusted_resolution_root {
        return Err(invalid(
            "layer resolution_root does not match the externally-trusted resolution root",
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

    // Build each dictionary Merkle tree once (only when a column actually reads it), and
    // resolve every delivered id by a positional opening against its root-bound root.
    let needs_node_value = descriptors
        .iter()
        .any(|d| d.namespace == ResultIdNamespace::NodeValue);
    let needs_predicate = descriptors
        .iter()
        .any(|d| d.namespace == ResultIdNamespace::Predicate);
    // TERM-LESS (S3): the two clear catalogs are dropped, so the trees cannot be rebuilt —
    // DELIBERATELY route through the prover openings (a missing carrier is an explicit
    // fail-close, not an incidental empty-catalog match-nothing). Term-bearing rebuilds +
    // root-checks each tree exactly as before.
    if !layer.terms_present && (needs_node_value || needs_predicate) && openings.is_none() {
        return Err(invalid(
            "term-less resolution requires prover-shipped Merkle openings",
        ));
    }
    let nv_tree = if needs_node_value && layer.terms_present {
        Some(node_value_merkle_tree(layer)?)
    } else {
        None
    };
    let pred_tree = if needs_predicate && layer.terms_present {
        Some(predicate_merkle_tree(layer)?)
    } else {
        None
    };

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
                        ResultIdNamespace::NodeValue => {
                            resolve_node_value_id_via_merkle(id, layer, nv_tree.as_ref(), openings)?
                        }
                        ResultIdNamespace::Predicate => {
                            resolve_predicate_id_via_merkle(id, layer, pred_tree.as_ref(), openings)?
                        }
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
    trusted_resolution_root: ProofHash,
    columns: &[Vec<ArkFr>],
    objects: &[Vec<CanonicalObject>],
) -> io::Result<()> {
    let resolved = resolve_node_value_result_objects(layer, trusted_resolution_root, columns)?;
    if resolved != objects {
        return Err(invalid(
            "delivered result objects do not match the authenticated Store IDs",
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// LEGACY rebuild-the-catalog resolution (I1 differential baseline).
//
// The pre-I1 resolution scanned the clear catalog linearly. It is retained ONLY for the
// differential test, which resolves every result id via BOTH this path and the Merkle path
// and asserts equal terms. When I2 drops the clear catalog these disappear entirely.
// ---------------------------------------------------------------------------

/// Pre-I1 node/value resolution: linear catalog scan (differential baseline).
#[cfg(test)]
pub(crate) fn resolve_node_value_result_objects_via_catalog(
    source: &impl VerifierLayer,
    trusted_resolution_root: ProofHash,
    columns: &[Vec<ArkFr>],
) -> io::Result<Vec<Vec<CanonicalObject>>> {
    let layer = source.verifier_view();
    if layer.resolution_root() != trusted_resolution_root {
        return Err(invalid(
            "layer resolution_root does not match the externally-trusted resolution root",
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

/// Pre-I1 typed resolution: linear catalog scan (differential baseline).
#[cfg(test)]
pub(crate) fn resolve_result_values_via_catalog(
    source: &impl VerifierLayer,
    trusted_resolution_root: ProofHash,
    columns: &[Vec<ArkFr>],
    descriptors: &[ResultColumnDescriptor],
) -> io::Result<Vec<Vec<CanonicalResultValue>>> {
    let layer = source.verifier_view();
    if columns.len() != descriptors.len() {
        return Err(invalid("result column descriptor shape mismatch"));
    }
    if layer.resolution_root() != trusted_resolution_root {
        return Err(invalid(
            "layer resolution_root does not match the externally-trusted resolution root",
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
