//! `GroupBy` aggregates over the reconciled `present` relation via VERIFIER-RECOMPUTE.
//!
//! The delta-stack analogue of the base path's group aggregation
//! (`crates/woql2/.../columns.rs::expected_group_count_columns` etc.): the group aggregate is a
//! DETERMINISTIC function of the root-authenticated child relation, so it needs NO new crypto —
//! the verifier RE-COMPUTES it over the authenticated present rows.
//!
//! The child BGP result relation is authenticated + resolved by [`verify_and_resolve_bgp`] (the
//! SAME reconcile + present-BGP + resolve path a plain read uses: each kept id column is bound to
//! the head root and resolved to a `CanonicalObject` via the root-bound chain dictionaries, incl.
//! the C2a value↔term tie for value columns). The aggregate then groups the AUTHENTICATED resolved
//! rows and recomputes per group; a fabricated / mis-grouped result is impossible because the
//! verifier never trusts a prover-claimed group — it derives every group from the authenticated
//! rows itself (a tampered child row fails the resolution re-commitment inside
//! [`verify_and_resolve_bgp`] and rejects before any grouping).
//!
//! GROUPING is bound to the AUTHENTICATED key column: groups are keyed by each row's resolved
//! key `CanonicalObject` (a Node IRI). The combined chain dictionary is injective (an IRI is minted
//! in exactly one layer, so it has ONE id across the chain), so grouping by the resolved IRI equals
//! grouping by the authenticated key id — and matches TerminusDB's `group_by/4` (which groups
//! solutions by the group-key binding). This slice covers `GroupCount` (cardinality per group, no
//! value-authentication); `GroupSum` / group `Min`/`Max` extend it with a numeric value column.

use std::collections::BTreeMap;

use super::*;

/// One recomputed group: the group's resolved key objects (one Node IRI per key column, in
/// `key_columns` order) and the number of authenticated child rows in the group.
pub type GroupCountEntry = (Vec<CanonicalObject>, u64);

/// Prove `GroupCount(BGP)` over an N-layer delta stack: reconcile + present-BGP over the child
/// relation (IDENTICAL to [`prove_read_bgp`] — the child rows are the delivered/authenticated
/// data; the groups are recomputed at verify). Returns the reconcile proof, the BGP proof, and the
/// child result relation (its kept id columns).
pub fn prove_read_group_count_bgp(
    layers: &[&LayerCommitment],
    atoms: &[PresentBgpAtom],
    joins: &[JoinStep],
) -> io::Result<(ReconcileProof, PresentBgpProof, PresentReadResult)> {
    prove_read_bgp(layers, atoms, joins)
}

/// Recompute the `GroupCount` output from the root-authenticated resolved child rows: group the
/// rows by their key `CanonicalObject` tuple (each must be a Node IRI) and count per group. The
/// groups are returned sorted by the canonical key-IRI-bytes tuple (deterministic, prover-blind).
pub(super) fn recompute_group_count(
    resolved: &[Vec<CanonicalObject>],
    key_columns: &[usize],
) -> io::Result<Vec<GroupCountEntry>> {
    if key_columns.is_empty() {
        return Err(invalid("reconciled group-count requires at least one key column"));
    }
    let n = resolved.first().map_or(0, Vec::len);
    for &kc in key_columns {
        let column = resolved
            .get(kc)
            .ok_or_else(|| invalid("reconciled group-count key column out of range"))?;
        if column.len() != n {
            return Err(invalid("reconciled group-count key column length mismatch"));
        }
    }
    // Key each group by the canonical IRI-bytes tuple (sorted), carrying the resolved key objects
    // for the readable output. A non-node key (a value literal) reject-closes — TerminusDB groups
    // node-valued keys here, and a value key has no authenticated node identity in this path.
    let mut groups: BTreeMap<Vec<Vec<u8>>, (Vec<CanonicalObject>, u64)> = BTreeMap::new();
    for row in 0..n {
        let mut key_bytes = Vec::with_capacity(key_columns.len());
        let mut key_objects = Vec::with_capacity(key_columns.len());
        for &kc in key_columns {
            match &resolved[kc][row] {
                object @ CanonicalObject::Node(iri) => {
                    key_bytes.push(iri.clone());
                    key_objects.push(object.clone());
                }
                _ => {
                    return Err(invalid(
                        "reconciled group-count key is not a node IRI (value-keyed grouping is unsupported)",
                    ))
                }
            }
        }
        let entry = groups.entry(key_bytes).or_insert((key_objects, 0));
        entry.1 = entry
            .1
            .checked_add(1)
            .ok_or_else(|| invalid("reconciled group-count exceeds u64"))?;
    }
    Ok(groups.into_values().collect())
}

/// Verify + resolve a reconciled `GroupCount(BGP)` read against the trusted head root, then RECOMPUTE
/// the per-group counts over the authenticated resolved child rows. `child_columns` are the prover's
/// delivered child id columns (bound to the head root by [`verify_and_resolve_bgp`] — a tampered /
/// fabricated child row rejects there). `key_columns` index the child schema (== the store result
/// column order). Returns the groups, sorted by key.
#[allow(clippy::too_many_arguments)]
pub fn verify_and_resolve_group_count_bgp(
    layers: &[&LayerCommitment],
    trusted_head_root: ProofHash,
    atoms: &[PresentBgpAtom],
    joins: &[JoinStep],
    key_columns: &[usize],
    reconcile: &ReconcileProof,
    bgp: &PresentBgpProof,
    child_columns: &[Vec<ArkFr>],
) -> io::Result<Vec<GroupCountEntry>> {
    let resolved = verify_and_resolve_bgp(
        layers,
        trusted_head_root,
        atoms,
        joins,
        reconcile,
        bgp,
        child_columns,
    )?;
    recompute_group_count(&resolved, key_columns)
}
