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
use ark_ff::PrimeField;

// ===========================================================================
// Wave 4 — RESULT RESOLUTION: authenticated (hash|id) result bindings → CanonicalObjects.
//
// The BGP-over-present proofs authenticate result COMMITMENTS whose columns are binding
// HASHES (single-atom filter, `hash_object`/`hash_term`) or COMBINED-DICT IDS (join). This
// turns them into readable answers — real IRIs / typed literals — WITHOUT re-running the query:
//   1. The prover's claimed result VALUES (real prefix) are tied to the authenticated result
//      commitment by re-committing (pad to the result domain, `commit_column`, require equality
//      with the proof's committed output). A forged/altered binding fails this bind.
//   2. Each value resolves to a `CanonicalObject`:
//      * an ID column (`NodeValueId`, from the join/BGP/delta-read/walk paths) is resolved by an
//        id-indexed positional Merkle OPENING (increment I3): every dictionary id lies in exactly
//        ONE layer's introduced interval `(parent_count, count]`, and which layer is a PUBLIC
//        function of the id and the chain-validated (abutting, root-bound) counts — so the id is
//        ROUTED to its introducing layer and opened at the STRUCTURAL position
//        `id - parent_count - 1` against that layer's root-bound `nv_merkle_root`. This is the
//        delta-stack analogue of the base-path I1 opening (`query::result`), across the chain;
//        it costs `O(log N)` per id rather than a whole-dictionary scan and never trusts a
//        prover-chosen layer (the interval check in `verify_opening` re-forces the routing).
//      * a HASH column (`NodeValueHash`/`PredicateHash`, from the single-atom filter, whose result
//        bindings are proof-authenticated hashes with NO delivered id) is ALSO resolved by an
//        id-routed positional Merkle opening (increment I2a): the prover ships an UNTRUSTED id-hint
//        per result row, which we route + open exactly like an id column, THEN require a MANDATORY
//        belt-hash — `hash_object`/`hash_term` of the opened term must equal the authenticated
//        present hash (`value`). A wrong id-hint (a different real entry, or a padding slot) opens
//        to a term whose hash ≠ `value` and is rejected, so the hint needs no authentication; the
//        catalog scan is gone (the clear catalog is now read ONLY by `validate()`'s re-MSM until I2b
//        drops it). The Merkle roots are root-bound exactly as on the id path.
// Mirrors the base `resolve_result_values` / I1 opening discipline, generalized across the chain.
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

/// The canonical `u64` an id field element must equal to index a dictionary. Rejects a field
/// element that is not a small (`< 2^64`) integer, exactly as the base-path resolver does before
/// trusting a scalar as a dictionary id (`query::result::canonical_dictionary_id`). `pub(super)`
/// so the filter prover (`present_filter`) can recover the `u64` id-hints it ships (I2a) from the
/// artifacts' `present_id_*` field columns.
pub(super) fn canonical_dictionary_id(id: &ArkFr) -> io::Result<u64> {
    let raw = id.0.into_bigint().as_ref()[0];
    if ArkFr::from_u64(raw) != *id {
        return Err(invalid("resolve: result id is outside canonical u64 range"));
    }
    Ok(raw)
}

/// Chain-routed, id-indexed Merkle resolver (increment I3).
///
/// For each dictionary id it ROUTES to the unique chain layer whose introduced interval
/// `(parent_count, count]` contains the id — a PUBLIC function of the id and the chain-validated,
/// root-bound per-layer counts (`validate_chain_states` forbids overlaps/gaps; each layer's counts
/// are hashed into its `commitment_root`, chained up to the trusted head) — then opens that layer's
/// tree at the structural position `id - parent_count - 1` against that layer's root-bound Merkle
/// root. Each layer's tree is (re)built from its root-bound catalog at most once and cached; the
/// rebuilt root is re-checked against the carried `nv_merkle_root`/`pred_merkle_root` (already tied
/// to `commitment_root` by the reconcile verifier's per-layer `validate()`), and `verify_opening`
/// re-forces the interval + position, so a prover cannot point an id at the wrong layer's tree.
struct ChainDictResolver<'a> {
    views: &'a [VerifierCommitmentRef<'a>],
    nv_trees: Vec<Option<DictMerkleTree>>,
    pred_trees: Vec<Option<DictMerkleTree>>,
    /// Merkle-dict I2b S3: prover-shipped, UNCOMMITTED openings for a TERM-LESS view in the
    /// chain (the dropped catalog can no longer rebuild its tree). `None` on the ordinary
    /// term-bearing delta path (every `LayerCommitment` view carries its catalog). When a
    /// routed view is term-less, the resolver opens from these instead of rebuilding, still
    /// verifying against that view's root-bound Merkle root (β re-anchor).
    openings: Option<&'a ResolutionOpenings>,
}

impl<'a> ChainDictResolver<'a> {
    fn new(views: &'a [VerifierCommitmentRef<'a>], openings: Option<&'a ResolutionOpenings>) -> Self {
        let n = views.len();
        Self {
            views,
            nv_trees: (0..n).map(|_| None).collect(),
            pred_trees: (0..n).map(|_| None).collect(),
            openings,
        }
    }

    /// The unique layer index whose node/value interval `(parent, count]` contains `id`. The
    /// intervals abut disjointly along a chain-validated stack, so exactly one matches a real id;
    /// none (or, defensively, more than one) fails closed.
    fn nv_layer(&self, id: u64) -> io::Result<usize> {
        let mut found: Option<usize> = None;
        for (i, view) in self.views.iter().enumerate() {
            if id > view.state.parent_node_value_count && id <= view.state.node_value_count {
                if found.is_some() {
                    return Err(invalid(
                        "resolve: node/value id lies in more than one layer's introduced interval",
                    ));
                }
                found = Some(i);
            }
        }
        found.ok_or_else(|| {
            invalid("resolve: node/value id lies outside every layer's introduced interval")
        })
    }

    /// The unique layer index whose predicate interval `(parent, count]` contains `id`.
    fn pred_layer(&self, id: u64) -> io::Result<usize> {
        let mut found: Option<usize> = None;
        for (i, view) in self.views.iter().enumerate() {
            if id > view.state.parent_predicate_count && id <= view.state.predicate_count {
                if found.is_some() {
                    return Err(invalid(
                        "resolve: predicate id lies in more than one layer's introduced interval",
                    ));
                }
                found = Some(i);
            }
        }
        found.ok_or_else(|| {
            invalid("resolve: predicate id lies outside every layer's introduced interval")
        })
    }

    /// Build (once) and cache layer `i`'s node/value tree, confirming the rebuilt root equals the
    /// carried, root-bound `nv_merkle_root`.
    fn nv_tree(&mut self, i: usize) -> io::Result<&DictMerkleTree> {
        if self.nv_trees[i].is_none() {
            let view = self.views[i];
            let tree = build_node_value_tree(
                view.node_value_catalog,
                view.state.parent_node_value_count,
                view.state.node_value_count,
            )?;
            if tree.root() != view.nv_merkle_root {
                return Err(invalid(
                    "resolve: rebuilt node/value merkle root does not match the bound root",
                ));
            }
            self.nv_trees[i] = Some(tree);
        }
        Ok(self.nv_trees[i].as_ref().unwrap())
    }

    /// Build (once) and cache layer `i`'s predicate tree, confirming the rebuilt root equals the
    /// carried, root-bound `pred_merkle_root`.
    fn pred_tree(&mut self, i: usize) -> io::Result<&DictMerkleTree> {
        if self.pred_trees[i].is_none() {
            let view = self.views[i];
            let tree = build_predicate_tree(
                view.predicate_catalog,
                view.state.parent_predicate_count,
                view.state.predicate_count,
            )?;
            if tree.root() != view.pred_merkle_root {
                return Err(invalid(
                    "resolve: rebuilt predicate merkle root does not match the bound root",
                ));
            }
            self.pred_trees[i] = Some(tree);
        }
        Ok(self.pred_trees[i].as_ref().unwrap())
    }

    /// Route + open a node/value id to its authenticated term. `expected_hash`, when supplied
    /// (belt-and-suspenders on the delta path, per the design's "Opening soundness" decision),
    /// additionally requires `hash_object(term)` to equal the proof-authenticated hash — a
    /// mismatch rejects.
    fn resolve_nv_id(
        &mut self,
        id: u64,
        expected_hash: Option<ArkFr>,
    ) -> io::Result<CanonicalObject> {
        let i = self.nv_layer(id)?;
        let view = self.views[i];
        // TERM-BEARING (the ordinary delta path): rebuild + open the routed layer's tree.
        // TERM-LESS (Merkle-dict I2b S3): the catalog is dropped, so open the prover-shipped
        // opening instead (fail-closed if absent). Both verify against the SAME root-bound
        // `nv_merkle_root`, and `verify_opening` re-forces the interval + structural position,
        // so a wrong opening cannot authenticate at the routed layer.
        let object = if view.terms_present {
            let tree = self.nv_tree(i)?;
            let opening = tree.open(id)?;
            verify_opening(
                &view.nv_merkle_root,
                view.state.parent_node_value_count,
                view.state.node_value_count,
                id,
                &opening,
            )?
        } else {
            let opening = self
                .openings
                .ok_or_else(|| {
                    invalid("term-less node/value resolution requires prover-shipped Merkle openings")
                })?
                .node_value_opening(id)?;
            verify_opening(
                &view.nv_merkle_root,
                view.state.parent_node_value_count,
                view.state.node_value_count,
                id,
                opening,
            )?
        };
        if let Some(hash) = expected_hash {
            if hash_object(&object) != hash {
                return Err(invalid(
                    "resolve: opened node/value term hash does not match the proof-authenticated hash",
                ));
            }
        }
        Ok(object)
    }

    /// Route + open a predicate id to its authenticated IRI (as a `Node`, mirroring the legacy
    /// predicate resolver). `expected_hash`, when supplied, requires `hash_term(iri)` to equal the
    /// proof-authenticated hash.
    fn resolve_pred_id(
        &mut self,
        id: u64,
        expected_hash: Option<ArkFr>,
    ) -> io::Result<CanonicalObject> {
        let i = self.pred_layer(id)?;
        let view = self.views[i];
        // Term-bearing rebuilds+opens the routed layer's tree; term-less (S3) opens the
        // prover-shipped opening (fail-closed if absent) — both against `pred_merkle_root`.
        let object = if view.terms_present {
            let tree = self.pred_tree(i)?;
            let opening = tree.open(id)?;
            verify_opening(
                &view.pred_merkle_root,
                view.state.parent_predicate_count,
                view.state.predicate_count,
                id,
                &opening,
            )?
        } else {
            let opening = self
                .openings
                .ok_or_else(|| {
                    invalid("term-less predicate resolution requires prover-shipped Merkle openings")
                })?
                .predicate_opening(id)?;
            verify_opening(
                &view.pred_merkle_root,
                view.state.parent_predicate_count,
                view.state.predicate_count,
                id,
                opening,
            )?
        };
        if let Some(hash) = expected_hash {
            let iri = match &object {
                CanonicalObject::Node(bytes) => bytes.as_slice(),
                _ => {
                    return Err(invalid(
                        "resolve: predicate leaf did not reconstruct as a node IRI",
                    ))
                }
            };
            if hash_term(iri) != hash {
                return Err(invalid(
                    "resolve: opened predicate term hash does not match the proof-authenticated hash",
                ));
            }
        }
        Ok(object)
    }

    /// Resolve one authenticated result value by its column kind and (for the hash kinds) an
    /// UNTRUSTED prover `id_hint` (I2a).
    ///
    /// * `NodeValueId` — the value IS the combined-dict id; route + open it (I3). No separate
    ///   authenticated term-hash exists on this path, so no belt-hash (`None`).
    /// * `NodeValueHash` / `PredicateHash` — the single-atom filter output binding, delivered as
    ///   the proof-authenticated present hash (`present_o`/`present_s`/`present_p`) with NO id. The
    ///   prover ships an id-hint; we ROUTE it to its layer's tree and open it under a MANDATORY
    ///   belt-hash `expected_hash = value`. Soundness rests entirely on that belt-hash: a wrong
    ///   id-hint (a different real entry, or a padding slot) opens to a term whose
    ///   `hash_object`/`hash_term` (dict.rs — the SAME function that built `value`) ≠ `value` and
    ///   is REJECTED (Blake2b collision-resistance), so the untrusted hint needs no authentication.
    ///   The catalog scan is gone; every hash-kind resolve is now an id-routed Merkle opening.
    fn resolve(
        &mut self,
        value: ArkFr,
        kind: ResolveKind,
        id_hint: Option<u64>,
    ) -> io::Result<CanonicalObject> {
        match kind {
            ResolveKind::NodeValueId => self.resolve_nv_id(canonical_dictionary_id(&value)?, None),
            ResolveKind::NodeValueHash => {
                let id = id_hint.ok_or_else(|| {
                    invalid("resolve: node/value hash result requires a prover id-hint (I2a)")
                })?;
                // Belt-hash MANDATORY (`Some(value)`): `value` is the proof-authenticated present hash.
                self.resolve_nv_id(id, Some(value))
            }
            ResolveKind::PredicateHash => {
                let id = id_hint.ok_or_else(|| {
                    invalid("resolve: predicate hash result requires a prover id-hint (I2a)")
                })?;
                self.resolve_pred_id(id, Some(value))
            }
        }
    }
}

/// Resolve authenticated result columns to `CanonicalObject`s. `claimed_columns` are the
/// prover's real-prefix result values; they are bound to `com_out` (the proof's committed
/// result, at `2^domain_log`) by re-commitment before resolution.
///
/// `id_hints`, when supplied, is one UNTRUSTED prover id-hint column per result column (aligned
/// row-for-row). It is MANDATORY for the hash kinds (`NodeValueHash`/`PredicateHash`, the
/// single-atom filter path) and ignored for `NodeValueId` (the value IS the id). The resolution
/// belt-hash binds each hint to the authenticated result hash, so the hint is not committed and
/// not part of `commitment_root` (I2a).
///
/// `openings`, when supplied (Merkle-dict I2b S3), is the prover-shipped, UNCOMMITTED
/// [`ResolutionOpenings`] used to resolve ids routed to a TERM-LESS view (whose dropped catalog
/// cannot rebuild its tree). It is `None` on the ordinary delta path (all `LayerCommitment`
/// views are term-bearing); every routed view is opened against its own root-bound Merkle root
/// either way, so `openings` — like `id_hints` — is a proof-side witness, never committed.
pub(super) fn resolve_result_columns(
    views: &[VerifierCommitmentRef<'_>],
    claimed_columns: &[Vec<ArkFr>],
    com_out: &[ArkGT],
    result_len: usize,
    domain_log: usize,
    kinds: &[ResolveKind],
    id_hints: Option<&[Vec<u64>]>,
    openings: Option<&ResolutionOpenings>,
) -> io::Result<Vec<Vec<CanonicalObject>>> {
    if claimed_columns.len() != com_out.len() || claimed_columns.len() != kinds.len() {
        return Err(invalid("resolve: result column shape mismatch"));
    }
    if let Some(hints) = id_hints {
        if hints.len() != claimed_columns.len() {
            return Err(invalid("resolve: id-hint column count mismatch"));
        }
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
    // (2) Resolve each value against the root-bound chain dictionaries by an id-routed positional
    // Merkle opening — id columns route the value-as-id (I3); hash columns route the prover's
    // (belt-hash-bound) id-hint (I2a). No catalog scan.
    let mut resolver = ChainDictResolver::new(views, openings);
    let mut resolved = Vec::with_capacity(claimed_columns.len());
    for (col_idx, (col, kind)) in claimed_columns.iter().zip(kinds).enumerate() {
        // Per-column id-hints: MANDATORY for the hash kinds (the belt-hash binds them), unused
        // for id kinds (`NodeValueId`, where the value already IS the id).
        let col_hints: Option<&Vec<u64>> = match kind {
            ResolveKind::NodeValueHash | ResolveKind::PredicateHash => {
                let hints = id_hints.ok_or_else(|| {
                    invalid("resolve: hash-kind result column requires prover id-hints (I2a)")
                })?;
                let h = &hints[col_idx];
                if h.len() != col.len() {
                    return Err(invalid("resolve: id-hint column length mismatch"));
                }
                Some(h)
            }
            ResolveKind::NodeValueId => None,
        };
        let mut out = Vec::with_capacity(col.len());
        for (row, value) in col.iter().enumerate() {
            out.push(resolver.resolve(*value, *kind, col_hints.map(|h| h[row]))?);
        }
        resolved.push(out);
    }
    Ok(resolved)
}

// ---------------------------------------------------------------------------
// I3 test-support: direct chain-routed Merkle resolution + the pre-I3 catalog-scan baseline,
// so the differential test can assert Merkle-routed resolution == the legacy scan for every id.
// ---------------------------------------------------------------------------

/// Resolve a node/value id directly by chain-routed Merkle opening (optional belt-and-suspenders
/// hash cross-check). Test-only surface over [`ChainDictResolver`]. Term-bearing views rebuild.
#[cfg(test)]
pub(crate) fn resolve_node_value_id_via_chain_merkle(
    views: &[VerifierCommitmentRef<'_>],
    id: u64,
    expected_hash: Option<ArkFr>,
) -> io::Result<CanonicalObject> {
    ChainDictResolver::new(views, None).resolve_nv_id(id, expected_hash)
}

/// Resolve a predicate id directly by chain-routed Merkle opening (optional hash cross-check).
#[cfg(test)]
pub(crate) fn resolve_predicate_id_via_chain_merkle(
    views: &[VerifierCommitmentRef<'_>],
    id: u64,
    expected_hash: Option<ArkFr>,
) -> io::Result<CanonicalObject> {
    ChainDictResolver::new(views, None).resolve_pred_id(id, expected_hash)
}

/// Merkle-dict I2b S3 differential/negative test surface: resolve a node/value id over the
/// chain when a routed view is TERM-LESS, consuming prover-shipped `openings`.
#[cfg(test)]
pub(crate) fn resolve_node_value_id_via_chain_merkle_with_openings(
    views: &[VerifierCommitmentRef<'_>],
    id: u64,
    expected_hash: Option<ArkFr>,
    openings: &ResolutionOpenings,
) -> io::Result<CanonicalObject> {
    ChainDictResolver::new(views, Some(openings)).resolve_nv_id(id, expected_hash)
}

/// Merkle-dict I2b S3 differential/negative test surface: resolve a predicate id over the chain
/// when a routed view is TERM-LESS, consuming prover-shipped `openings`.
#[cfg(test)]
pub(crate) fn resolve_predicate_id_via_chain_merkle_with_openings(
    views: &[VerifierCommitmentRef<'_>],
    id: u64,
    expected_hash: Option<ArkFr>,
    openings: &ResolutionOpenings,
) -> io::Result<CanonicalObject> {
    ChainDictResolver::new(views, Some(openings)).resolve_pred_id(id, expected_hash)
}

/// Pre-I3 node/value id resolution: linear scan of the chain catalogs (differential baseline).
#[cfg(test)]
pub(crate) fn resolve_node_value_id_via_catalog_scan(
    views: &[VerifierCommitmentRef<'_>],
    id: u64,
) -> io::Result<CanonicalObject> {
    for view in views {
        for rec in view.node_value_catalog {
            if rec.object_id == id {
                return Ok(rec.triple.object.clone());
            }
        }
    }
    Err(invalid("resolve: result id is absent from the chain node/value dictionaries"))
}

/// Pre-I3 predicate id resolution: linear scan of the chain catalogs (differential baseline).
#[cfg(test)]
pub(crate) fn resolve_predicate_id_via_catalog_scan(
    views: &[VerifierCommitmentRef<'_>],
    id: u64,
) -> io::Result<CanonicalObject> {
    for view in views {
        for rec in view.predicate_catalog {
            if rec.predicate_id == id {
                return Ok(CanonicalObject::Node(rec.triple.predicate.clone()));
            }
        }
    }
    Err(invalid("resolve: result id is absent from the chain predicate dictionaries"))
}

/// Pre-I2a node/value HASH resolution: linear scan of the chain catalogs for the matching
/// `hash_object` — the legacy filter-path oracle the I2a differential test asserts the new
/// Merkle+belt-hash resolution reproduces exactly.
#[cfg(test)]
pub(crate) fn resolve_node_value_hash_via_catalog_scan(
    views: &[VerifierCommitmentRef<'_>],
    value: ArkFr,
) -> io::Result<CanonicalObject> {
    for view in views {
        for rec in view.node_value_catalog {
            if hash_object(&rec.triple.object) == value {
                return Ok(rec.triple.object.clone());
            }
        }
    }
    Err(invalid("resolve: result hash is absent from the chain node/value dictionaries"))
}

/// Pre-I2a predicate HASH resolution: linear scan of the chain catalogs for the matching
/// `hash_term` (the predicate-column differential baseline).
#[cfg(test)]
pub(crate) fn resolve_predicate_hash_via_catalog_scan(
    views: &[VerifierCommitmentRef<'_>],
    value: ArkFr,
) -> io::Result<CanonicalObject> {
    for view in views {
        for rec in view.predicate_catalog {
            if hash_term(&rec.triple.predicate) == value {
                return Ok(CanonicalObject::Node(rec.triple.predicate.clone()));
            }
        }
    }
    Err(invalid("resolve: result hash is absent from the chain predicate dictionaries"))
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
        None,
        // Term-bearing `LayerCommitment` views: no term-less openings.
        None,
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
/// published result values (from [`PresentReadResult`]); `id_hints` are the prover's UNTRUSTED
/// per-column id-hints (also from [`PresentReadResult`]), which the MANDATORY resolution belt-hash
/// binds to the authenticated result hashes (I2a).
#[allow(clippy::too_many_arguments)]
pub fn verify_and_resolve_single_atom(
    layers: &[&LayerCommitment],
    trusted_head_root: ProofHash,
    position: FilterPosition,
    constant_hash: ArkFr,
    reconcile: &ReconcileProof,
    filter: &PresentFilterProof,
    claimed_columns: &[Vec<ArkFr>],
    id_hints: &[Vec<u64>],
) -> io::Result<Vec<Vec<CanonicalObject>>> {
    let mut transcript = Transcript254::new(b"parture-delta-reconcile-envelope");
    let verified = verify_reconcile_core(layers, trusted_head_root, reconcile, &mut transcript)?;
    let out_len = verify_present_filter(&verified, position, constant_hash, filter, &mut transcript)?;
    let views: Vec<VerifierCommitmentRef<'_>> = layers.iter().map(|l| l.verifier_view()).collect();
    let kinds = filter_output_kinds(position);
    resolve_result_columns(
        &views,
        claimed_columns,
        &filter.com_out,
        out_len,
        verified.working_log,
        &kinds,
        Some(id_hints),
        // Delta path resolves over full term-bearing `LayerCommitment` views (term-less is
        // base/rollup-only and is fail-closed upstream at `present_bgp.rs` require_terms_present).
        None,
    )
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
    // Join output [id_x, id_y, id_z] are all node/value combined-dict ids (no id-hints needed).
    let kinds = [ResolveKind::NodeValueId, ResolveKind::NodeValueId, ResolveKind::NodeValueId];
    resolve_result_columns(&views, claimed_columns, &join.com_join_out, out_len, join.join_log, &kinds, None, None)
}

