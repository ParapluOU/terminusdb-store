//! Result resolution: authenticated result bindings → `CanonicalObject`s.
//!
//! The present read/join/BGP proofs deliver result columns as authenticated (hash or id)
//! bindings. This module turns each such binding back into a concrete `CanonicalObject` (a
//! node IRI or a typed literal) by looking it up against the CHAIN dictionary catalogs — a
//! term's canonical bytes live in the catalog of the layer whose incremental dict range
//! defines it, and those catalogs are root-authenticated by the reconciliation verifier. A
//! forged or altered binding fails the bind. Mirrors the base `resolve_result_values`
//! discipline, generalized across the N-layer chain.

use super::*;

// ===========================================================================
// Wave 4 — RESULT RESOLUTION: authenticated (hash|id) result bindings → CanonicalObjects.
//
// The BGP-over-present proofs authenticate result COMMITMENTS whose columns are binding
// HASHES (single-atom filter, `hash_object`/`hash_term`) or COMBINED-DICT IDS (join). This
// turns them into readable answers — real IRIs / typed literals — WITHOUT re-running the query:
//   1. The prover's claimed result VALUES (real prefix) are tied to the authenticated result
//      commitment by re-committing (pad to the result domain, `commit_column`, require equality
//      with the proof's committed output). A forged/altered binding fails this bind.
//   2. Each value resolves to a `CanonicalObject` against the CHAIN catalogs — a term's canonical
//      bytes live in the catalog of the layer whose incremental dict range defines it, so the
//      resolver searches every layer's root-bound `node_value_catalog` / `predicate_catalog`
//      (each id/hash is unique across the disjoint ranges). The catalogs are authenticated by the
//      per-layer `validate()` the reconciliation verifier already ran (changed IRI/datatype/lexical
//      bytes re-hash differently and fail the root-bound dictionary commitment).
// Mirrors the base `resolve_result_values` discipline, generalized across the N-layer chain.
// ===========================================================================

/// How to resolve a result-column value to a `CanonicalObject`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolveKind {
    /// value = `hash_object(object)`; resolve via the chain node/value catalogs.
    NodeValueHash,
    /// value = combined-dict `object_id`; resolve via the chain node/value catalogs.
    NodeValueId,
    /// value = `hash_term(predicate)`; resolve via the chain predicate catalogs.
    PredicateHash,
}

pub(super) fn resolve_one(
    views: &[VerifierCommitmentRef<'_>],
    value: ArkFr,
    kind: ResolveKind,
) -> io::Result<CanonicalObject> {
    match kind {
        ResolveKind::NodeValueHash => {
            for view in views {
                for rec in view.node_value_catalog {
                    if hash_object(&rec.triple.object) == value {
                        return Ok(rec.triple.object.clone());
                    }
                }
            }
            Err(invalid("resolve: result hash is absent from the chain node/value dictionaries"))
        }
        ResolveKind::NodeValueId => {
            for view in views {
                for rec in view.node_value_catalog {
                    if ark(rec.object_id) == value {
                        return Ok(rec.triple.object.clone());
                    }
                }
            }
            Err(invalid("resolve: result id is absent from the chain node/value dictionaries"))
        }
        ResolveKind::PredicateHash => {
            for view in views {
                for rec in view.predicate_catalog {
                    if hash_term(&rec.triple.predicate) == value {
                        return Ok(CanonicalObject::Node(rec.triple.predicate.clone()));
                    }
                }
            }
            Err(invalid("resolve: result hash is absent from the chain predicate dictionaries"))
        }
    }
}

/// Resolve authenticated result columns to `CanonicalObject`s. `claimed_columns` are the
/// prover's real-prefix result values; they are bound to `com_out` (the proof's committed
/// result, at `2^domain_log`) by re-commitment before resolution.
pub(super) fn resolve_result_columns(
    views: &[VerifierCommitmentRef<'_>],
    claimed_columns: &[Vec<ArkFr>],
    com_out: &[ArkGT],
    result_len: usize,
    domain_log: usize,
    kinds: &[ResolveKind],
) -> io::Result<Vec<Vec<CanonicalObject>>> {
    if claimed_columns.len() != com_out.len() || claimed_columns.len() != kinds.len() {
        return Err(invalid("resolve: result column shape mismatch"));
    }
    // (1) Bind the claimed values to the authenticated result commitments.
    for (col, com) in claimed_columns.iter().zip(com_out.iter()) {
        if col.len() != result_len {
            return Err(invalid("resolve: claimed result column length mismatch"));
        }
        let mut padded = col.clone();
        padded.resize(1usize << domain_log, f0());
        if commit_column(&padded)? != *com {
            return Err(invalid(
                "resolve: claimed result values do not open the authenticated result commitment",
            ));
        }
    }
    // (2) Resolve each value against the root-bound chain catalogs.
    claimed_columns
        .iter()
        .zip(kinds)
        .map(|(col, kind)| col.iter().map(|v| resolve_one(views, *v, *kind)).collect())
        .collect()
}

/// Bind an authenticated combined-dict node/value-id column to its commitment `com` and resolve it
/// to `CanonicalObject`s over the CHAIN catalogs — the value-resolution surface a reconciled walk
/// composite needs for its delivered element column (ReadDocument 6c-ii). `ids` are the prover's
/// PUBLIC delivered ids; step (1) re-commits them at `log_domain` and requires equality with the
/// authenticated `com` (a forged/altered id fails the bind), step (2) resolves each surviving id via
/// the root-bound combined node/value catalogs (fail-closed on an absent id). This is exactly the
/// `resolve_result_columns` discipline for one `NodeValueId` column — the walk's `com_elements` is
/// authenticated by the walk's own gluing gadgets, so this ties the readable answer to it.
pub(crate) fn resolve_node_value_id_column(
    views: &[VerifierCommitmentRef<'_>],
    ids: &[u64],
    com: ArkGT,
    log_domain: usize,
) -> io::Result<Vec<CanonicalObject>> {
    let column: Vec<ArkFr> = ids.iter().map(|id| ark(*id)).collect();
    let mut resolved = resolve_result_columns(
        views,
        &[column],
        &[com],
        ids.len(),
        log_domain,
        &[ResolveKind::NodeValueId],
    )?;
    resolved
        .pop()
        .ok_or_else(|| invalid("resolve: empty node/value-id column"))
}

/// The output-column resolution kinds for a single-atom filter (the two non-constant columns).
pub(super) fn filter_output_kinds(position: FilterPosition) -> [ResolveKind; 2] {
    match position {
        // output [S, O]
        FilterPosition::Predicate => [ResolveKind::NodeValueHash, ResolveKind::NodeValueHash],
        // output [P, O]
        FilterPosition::Subject => [ResolveKind::PredicateHash, ResolveKind::NodeValueHash],
        // output [S, P]
        FilterPosition::Object => [ResolveKind::NodeValueHash, ResolveKind::PredicateHash],
    }
}

/// Verify a single-atom read AND resolve its result to `CanonicalObject`s (the readable answer,
/// authenticated against the chain — no query re-run). `claimed_columns` are the prover's
/// published result values (from [`PresentReadResult`]).
#[allow(clippy::too_many_arguments)]
pub fn verify_and_resolve_single_atom(
    layers: &[&LayerCommitment],
    trusted_head_root: ProofHash,
    position: FilterPosition,
    constant_hash: ArkFr,
    reconcile: &ReconcileProof,
    filter: &PresentFilterProof,
    claimed_columns: &[Vec<ArkFr>],
) -> io::Result<Vec<Vec<CanonicalObject>>> {
    let mut transcript = Transcript254::new(b"parture-delta-reconcile-envelope");
    let verified = verify_reconcile_core(layers, trusted_head_root, reconcile, &mut transcript)?;
    let out_len = verify_present_filter(&verified, position, constant_hash, filter, &mut transcript)?;
    let views: Vec<VerifierCommitmentRef<'_>> = layers.iter().map(|l| l.verifier_view()).collect();
    let kinds = filter_output_kinds(position);
    resolve_result_columns(&views, claimed_columns, &filter.com_out, out_len, verified.working_log, &kinds)
}

/// Verify a two-atom join read AND resolve its `[?x, ?y, ?z]` bindings to `CanonicalObject`s.
#[allow(clippy::too_many_arguments)]
pub fn verify_and_resolve_two_atom_join(
    layers: &[&LayerCommitment],
    trusted_head_root: ProofHash,
    const_p: ArkFr,
    const_q: ArkFr,
    reconcile: &ReconcileProof,
    join: &PresentJoinProof,
    claimed_columns: &[Vec<ArkFr>],
) -> io::Result<Vec<Vec<CanonicalObject>>> {
    let mut transcript = Transcript254::new(b"parture-delta-reconcile-envelope");
    let verified = verify_reconcile_core(layers, trusted_head_root, reconcile, &mut transcript)?;
    let head_nv = layers.last().unwrap().verifier_view().state.node_value_count as usize;
    let out_len = verify_present_join(reconcile, &verified, head_nv, const_p, const_q, join, &mut transcript)?;
    let views: Vec<VerifierCommitmentRef<'_>> = layers.iter().map(|l| l.verifier_view()).collect();
    // Join output [id_x, id_y, id_z] are all node/value combined-dict ids.
    let kinds = [ResolveKind::NodeValueId, ResolveKind::NodeValueId, ResolveKind::NodeValueId];
    resolve_result_columns(&views, claimed_columns, &join.com_join_out, out_len, join.join_log, &kinds)
}

