//! How the commitment root binds a layer — the trust anchor.
//!
//! A verifier should have to trust exactly ONE 32-byte value per layer. That value is the
//! `commitment_root`, and this module computes it: [`compute_commitment_root_from_parts`]
//! hashes, in a fixed domain-separated order, the committed state (parent root + counts) and
//! every column/dictionary/range Dory commitment into that single digest. Because the digest
//! covers all of them, tampering with any committed column, dictionary entry, or count
//! changes the root — so binding a proof to the root binds it to the entire committed layer.
//! [`database_root`] combines an instance root with a schema root into one database-level
//! anchor, and [`validate_database_root`] checks that combination.
//!
//! This is the concrete `commitment_root` of the [layer-commitment
//! model](crate::proof::primer) (primer §9) — the single 32-byte value a
//! verifier trusts per layer.

use super::*;

/// Fold a layer's committed parts into its single `commitment_root` digest.
///
/// Hashes, in a FIXED domain-separated order, the authenticated state (parent
/// root + counts) followed by every column / dictionary / range Dory commitment.
/// A delta layer (`parent_state_root == Some`) omits the materialized head view
/// and uses a distinct domain (see the inline note), so its bound set can never
/// collide with a base layer's. The order and domain are load-bearing: they are
/// what makes the root bind the whole layer, and changing either would invalidate
/// every existing anchor — hence any edit here is a wire-format break.
pub(super) fn compute_commitment_root_from_parts(
    state: &CommitmentState,
    column_commitments: &[ArkGT; COLUMN_COUNT],
    materialized_column_commitments: &[ArkGT; MATERIALIZED_COLUMN_COUNT],
    predicate_dict_commitments: &(ArkGT, ArkGT),
    node_value_dict_commitments: &(ArkGT, ArkGT),
    node_value_dict_range_commitments: &(ArkGT, ArkGT),
) -> io::Result<ProofHash> {
    // COMMITMENT DIET (Wave 0): a DELTA layer (parent_state_root == Some) no longer
    // commits the materialized head view. Its root therefore binds {parent root,
    // add/remove columns, INCREMENTAL dictionary commitments, counts} and omits the
    // materialized column commitments AND the (now always-zero) materialized_count.
    // Because the bound set changed, the delta root uses a BUMPED, distinct domain
    // (`commitment-root-delta/v6`) so it can never collide with either the byte-
    // identical v4 base root or a stale pre-diet delta root (which committed the full
    // materialized view under the v4 domain and now rejects closed on decode).
    //
    // A BASE/ROLLUP/SQUASH layer (parent_state_root == None) is untouched: same
    // `commitment-root/v4` domain, same counts (including materialized_count), and
    // still binds the materialized column commitments — BYTE-IDENTICAL to before.
    let is_delta = state.parent_state_root.is_some();
    let mut hasher = Sha256::new();
    if is_delta {
        hasher.update(b"terminus-store/proof-of-woql/commitment-root-delta/v6");
    } else {
        hasher.update(b"terminus-store/proof-of-woql/commitment-root/v4");
    }
    match state.parent_state_root {
        Some(root) => {
            hasher.update([1u8]);
            hasher.update(root);
        }
        None => hasher.update([0u8]),
    }
    let counts: &[u64] = if is_delta {
        &[
            state.addition_count,
            state.removal_count,
            state.parent_node_value_count,
            state.parent_predicate_count,
            state.node_value_count,
            state.predicate_count,
        ]
    } else {
        &[
            state.addition_count,
            state.removal_count,
            state.materialized_count,
            state.parent_node_value_count,
            state.parent_predicate_count,
            state.node_value_count,
            state.predicate_count,
        ]
    };
    for count in counts {
        hasher.update(count.to_be_bytes());
    }
    let mut commitments: Vec<&ArkGT> = column_commitments.iter().collect();
    if !is_delta {
        commitments.extend(materialized_column_commitments.iter());
    }
    commitments.extend([
        &predicate_dict_commitments.0,
        &predicate_dict_commitments.1,
        &node_value_dict_commitments.0,
        &node_value_dict_commitments.1,
        &node_value_dict_range_commitments.0,
        &node_value_dict_range_commitments.1,
    ]);
    for commitment in commitments {
        let bytes = serialize_commitment(commitment)?;
        hasher.update((bytes.len() as u64).to_be_bytes());
        hasher.update(&bytes);
    }
    Ok(hasher.finalize().into())
}

/// SCHEMA-GRAPH FOUNDATION — the combined `database_root`.
///
/// A query-proof database is TWO separately committed triple graphs: the
/// instance graph and the schema graph. Each is committed by the SAME per-layer
/// commitment machinery (`compute_commitment_root_from_parts`) as its own
/// `LayerCommitment` with its own `commitment_root`; the schema graph is NOT
/// folded into the instance commitment (that would perturb the v4 instance root
/// and break byte-identity of every existing instance-only proof). Instead the
/// two per-graph roots are bound into ONE externally-trusted anchor by a plain
/// two-hash wrapper — no polynomial machinery, no new committed structure:
///
/// `database_root = SHA256("parture/proof-of-woql/db-root/v1" ‖ instance_root ‖ schema_root)`.
///
/// The ordering (instance first, then schema) is fixed and domain-separated, so
/// the two roots can never be transposed without changing the anchor. An
/// instance-only proof never engages the schema root: its trust anchor remains
/// the instance-graph anchor. The combined root is engaged only when a query
/// reads the schema graph (Subsumption), and only to authenticate that BOTH
/// sub-commitments belong to the trusted database.
///
/// β RE-ANCHOR: this is a pure two-hash combiner over two 32-byte per-graph
/// anchors. [`validate_database_root`] now feeds it each graph's
/// `resolution_root` (not its `commitment_root`), so the combined anchor binds
/// all four dictionary Merkle roots (instance + schema) directly. Callers that
/// build the expected `database_root` out of band must combine the two per-graph
/// `resolution_root`s accordingly.
pub fn database_root(instance_root: &ProofHash, schema_root: &ProofHash) -> ProofHash {
    let mut hasher = Sha256::new();
    hasher.update(b"parture/proof-of-woql/db-root/v1");
    hasher.update(instance_root);
    hasher.update(schema_root);
    hasher.finalize().into()
}

/// Validate BOTH per-graph verifier views independently (each recomputes its own
/// `commitment_root` from its committed parts via `VerifierCommitmentRef::validate`)
/// AND confirm the two roots combine to the externally-trusted `database_root`.
///
/// This is the soundness gate a schema-reading verifier runs BEFORE trusting either
/// per-graph root as a proof anchor: neither sub-root can be swapped, forged, or
/// mismatched against the trusted database anchor without rejecting closed. Reusing
/// the audited per-commitment `validate` keeps the trust story a strict superset of
/// the single-graph case — each graph is authenticated exactly as before, plus the
/// combination check.
pub fn validate_database_root(
    instance: &impl VerifierLayer,
    schema: &impl VerifierLayer,
    trusted_database_root: &ProofHash,
) -> io::Result<()> {
    let instance_view = instance.verifier_view();
    let schema_view = schema.verifier_view();
    instance_view.validate()?;
    schema_view.validate()?;
    // β re-anchor: the combined database anchor now binds each graph's `resolution_root`
    // (which binds that graph's `commitment_root` AND its two dictionary Merkle roots) rather
    // than only its `commitment_root`. Both views were `validate()`d just above (each graph's
    // Merkle roots recomputed from its root-bound catalog), so both per-graph resolution_roots
    // are fully authenticated; the trusted `database_root` therefore binds all four Merkle
    // roots (instance + schema) directly, not merely transitively via the catalogs.
    if &database_root(
        &instance_view.resolution_root(),
        &schema_view.resolution_root(),
    ) != trusted_database_root
    {
        return Err(invalid(
            "database root does not combine the trusted instance and schema resolution roots",
        ));
    }
    Ok(())
}

