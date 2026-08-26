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

    /// READABLE-ANSWER `NodeValueId` resolution (join/BGP/delta/walk output → `CanonicalObject`).
    ///
    /// SOUNDNESS BELT (delta-reconcile TERM-LESS fix). Unlike [`resolve_nv_id`]'s value-terms
    /// caller ([`verify_dict_value_terms_chain`], which belts every opened term by RE-COMMITTING its
    /// order-preserving `(value, datatype)` to the root-bound range-dict columns), this readable
    /// path has NO external belt: the delivered value IS the combined-dict id and the opened term is
    /// returned VERBATIM as the answer. So it must never open an UNBELTED id against a view whose
    /// `nv_merkle_root` is not bound to the trusted head anchor. Exactly two view kinds have a
    /// root-bound `nv_merkle_root`:
    ///   * a TERM-BEARING view — [`nv_tree`](Self::nv_tree) rebuilds the tree from the root-bound
    ///     catalog and re-checks it equals the carried root (`validate()`'s catalog re-MSM binds the
    ///     catalog to `commitment_root`), and
    ///   * the TERM-LESS HEAD — `verify_reconcile_core` checks `head.resolution_root() ==
    ///     trusted_head_root`, and `resolution_root = H(commitment_root, nv_merkle_root,
    ///     pred_merkle_root)` binds the head's Merkle root directly.
    /// A NON-HEAD TERM-LESS view has NEITHER: its Merkle root is not folded into `commitment_root`
    /// (the chain only carries the root forward), its `resolution_root` is never checked, and
    /// `validate()` skips the catalog re-MSM — so the carried `nv_merkle_root` is prover-controlled.
    /// An unbelted open there would authenticate a FORGED term: the confirmed audit forgery ships a
    /// non-head term-less ancestor whose replaced `nv_merkle_root` holds `ex:Evil` at `id(ex:A)` plus
    /// a matching fake opening, resolving `triple(?s, ex:p, ?o)`'s `?o` to `ex:Evil`. FAIL-CLOSED.
    ///
    /// Honest term-less remote verification uses a TERM-LESS HEAD with TERM-BEARING ancestors (see
    /// `termless_head_views`), so this never rejects an honest readable resolution. A fully
    /// term-less chain would need a root-bound `id -> hash` belt (a prover combined-dict lookup over
    /// the result column, mirroring `PresentIdAuth`/`verify_dict_value_terms_chain`) to resolve
    /// ancestor-minted result ids; that is a cross-cutting proof-format change, out of scope for this
    /// localized soundness fix, so until then such a resolution fails closed rather than trust an
    /// unbound ancestor root.
    ///
    /// C4 fully-term-less RENDER belt (ADDITIVE, never a bypass): when `belt_hash = Some(H)`, `H`
    /// is a ROOT-BOUND `hash_object(term(id))` produced by [`verify_value_render_belt`] (an
    /// authenticated `id → hash` lookup against the reconciliation's combined dict). It relaxes the
    /// D5-FIX fail-close ONLY for a routed NON-HEAD term-less view, and is then handed to
    /// [`resolve_nv_id`](Self::resolve_nv_id) as the MANDATORY belt-hash — the opened term must
    /// satisfy `hash_object(term) == H`, so a forged non-head `nv_merkle_root` cannot launder a
    /// wrong term through. With `belt_hash = None` this is BYTE-IDENTICAL to the pre-belt fail-close.
    fn resolve_nv_id_readable(
        &mut self,
        id: u64,
        belt_hash: Option<ArkFr>,
    ) -> io::Result<CanonicalObject> {
        let i = self.nv_layer(id)?;
        let view = self.views[i];
        let is_head = i + 1 == self.views.len();
        if !view.terms_present && !is_head && belt_hash.is_none() {
            return Err(invalid(
                "resolve: unbelted NodeValueId routed to a NON-HEAD term-less view whose \
                 nv_merkle_root is not root-bound; refusing to resolve an unauthenticated term \
                 (delta-reconcile term-less fix — fail-closed)",
            ));
        }
        self.resolve_nv_id(id, belt_hash)
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
    /// * `NodeValueId` — the value IS the combined-dict id; route + open it (I3) via the
    ///   readable-answer guard [`resolve_nv_id_readable`](Self::resolve_nv_id_readable), which
    ///   FAIL-CLOSES an unbelted open routed to a NON-HEAD term-less view (whose `nv_merkle_root` is
    ///   not root-bound — the confirmed forgery). No per-row authenticated term-hash exists on this
    ///   path, so there is no belt-hash to pass.
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
        belt_hash: Option<ArkFr>,
    ) -> io::Result<CanonicalObject> {
        match kind {
            ResolveKind::NodeValueId => {
                // `belt_hash` (the fully-term-less RENDER belt) applies ONLY here — the value IS the
                // id, so there is no per-row authenticated present hash to belt against otherwise.
                self.resolve_nv_id_readable(canonical_dictionary_id(&value)?, belt_hash)
            }
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

/// Merkle-dict I2b step 3 (CROSS-LAYER, C2a): the value↔term consistency verifier for a numeric
/// object-value RANGE over a TERM-LESS delta / reconciled `present`. The chain analogue of
/// `verify_dict_value_terms`: the object
/// ids of a delta / reconciled range span LAYERS, so step (3)'s per-id resolution drives the
/// audited [`ChainDictResolver`] (which ROUTES each id to its introducing view and opens it
/// term-less against that view's root-bound `nv_merkle_root`, fail-closed on a missing/forged
/// opening) instead of the single-layer `resolve_node_value_id_via_merkle`. Everything else is
/// VERBATIM from the base helper:
///   * step (2) recommits `oid` (real ids, `0`-padded to `domain`) and requires
///     `Com(oid) == com_obj_id` — pinning the touched set to the EXACT authenticated object-id
///     column (`auth_o.com_id`); a substituted/subset id shifts the commitment and rejects;
///   * step (4) re-derives `ov = node_value_op_value(term)` (the single-source order-preserving
///     encoder the committed value column is built with) and `dt` = the datatype discriminant /
///     `NODE_DATATYPE_TAG`, byte-identically to the committed datatype column;
///   * step (5) recommits `ov`/`dt` and requires `Com(ov) == com_v_val`, `Com(dt) == com_v_dt` —
///     a forged `dict_value`/`dict_datatype` rejects.
/// Returns the DISTINCT opened touched terms (for the residual NaN gate). TRANSCRIPT-FREE (pure
/// commitment recompute, like the base helper). The chain re-anchor is the caller's
/// `verify_reconcile_core` / `verify_delta_read_core` (every view `validate()`d + chained to the
/// trusted head root before this runs), exactly as the sibling chain result-resolver relies on.
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_dict_value_terms_chain(
    views: &[VerifierCommitmentRef<'_>],
    com_obj_id: &ArkGT,
    com_v_val: &ArkGT,
    com_v_dt: &ArkGT,
    real_n: usize,
    domain: usize,
    oid: &[u64],
    openings: &ResolutionOpenings,
) -> io::Result<Vec<CanonicalObject>> {
    if domain == 0 || !domain.is_power_of_two() {
        return Err(invalid("value↔term domain must be a non-zero power of two"));
    }
    if real_n > domain || oid.len() != real_n {
        return Err(invalid(
            "value↔term oid vector length does not match the real row count",
        ));
    }

    // (2) Recommit the oid column (real rows from the vector, 0-pad beyond real_n — the SAME
    // layout the honest lookup gathers over, and the SAME 0-padding the object column commits
    // into `com_obj_id`) and bind it to the object-id commitment.
    let mut oid_col = vec![f0(); domain];
    for (slot, &raw) in oid_col.iter_mut().zip(oid.iter()) {
        *slot = ark(raw);
    }
    if commit_column(&oid_col)? != *com_obj_id {
        return Err(invalid(
            "value↔term: recommitted oid column does not match the object-id commitment",
        ));
    }

    // (3)-(4) Per-row re-derive (ov, dt) from the CHAIN-routed authenticated Merkle term. Distinct
    // ids are opened once; padding/null rows (`id == 0`) never index a dict row and are skipped.
    let mut resolver = ChainDictResolver::new(views, Some(openings));
    let mut ov_col = vec![f0(); domain];
    let mut dt_col = vec![f0(); domain];
    let mut cache: std::collections::BTreeMap<u64, (ArkFr, ArkFr)> = std::collections::BTreeMap::new();
    let mut touched: Vec<CanonicalObject> = Vec::new();
    for (i, &raw) in oid.iter().enumerate() {
        if raw == 0 {
            continue;
        }
        let (ov, dt) = if let Some(entry) = cache.get(&raw) {
            *entry
        } else {
            // No belt-hash on the value-terms path (no proof-authenticated per-row hash exists);
            // the recommit ties (2)/(5) are what bind the opened terms to the committed columns.
            let opened = resolver.resolve_nv_id(raw, None)?;
            let ov = node_value_op_value(&opened)?;
            // Byte-identical to `node_value_dict_datatype_column`: a `Value`'s discriminant,
            // a `Node`'s `NODE_DATATYPE_TAG`.
            let dt = ark(match &opened {
                CanonicalObject::Node(_) => NODE_DATATYPE_TAG,
                CanonicalObject::Value { datatype, .. } => *datatype as u8 as u64,
            });
            cache.insert(raw, (ov, dt));
            touched.push(opened);
            (ov, dt)
        };
        ov_col[i] = ov;
        dt_col[i] = dt;
    }

    // (5) Bind the re-derived value/datatype columns to the committed lookup columns.
    if commit_column(&ov_col)? != *com_v_val {
        return Err(invalid(
            "value↔term: recommitted value column does not match com_v_val (forged dict_value)",
        ));
    }
    if commit_column(&dt_col)? != *com_v_dt {
        return Err(invalid(
            "value↔term: recommitted datatype column does not match com_v_dt (forged dict_datatype)",
        ));
    }
    Ok(touched)
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
#[allow(clippy::too_many_arguments)]
pub(super) fn resolve_result_columns(
    views: &[VerifierCommitmentRef<'_>],
    claimed_columns: &[Vec<ArkFr>],
    com_out: &[ArkGT],
    result_len: usize,
    domain_log: usize,
    kinds: &[ResolveKind],
    id_hints: Option<&[Vec<u64>]>,
    openings: Option<&ResolutionOpenings>,
    belt_hashes: Option<&[Vec<ArkFr>]>,
) -> io::Result<Vec<Vec<CanonicalObject>>> {
    if claimed_columns.len() != com_out.len() || claimed_columns.len() != kinds.len() {
        return Err(invalid("resolve: result column shape mismatch"));
    }
    if let Some(hints) = id_hints {
        if hints.len() != claimed_columns.len() {
            return Err(invalid("resolve: id-hint column count mismatch"));
        }
    }
    // The C4 fully-term-less RENDER belt (one root-bound `hash_object(term(id))` per real row of a
    // `NodeValueId` column). Aligned one column per result column; `None` on every non-walk caller
    // (byte-identical to the pre-belt resolver).
    if let Some(belt) = belt_hashes {
        if belt.len() != claimed_columns.len() {
            return Err(invalid("resolve: belt-hash column count mismatch"));
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
        // Per-column RENDER belt (used only by the fully-term-less walk render path; every other
        // caller passes `None`). One root-bound hash per real row, aligned to `col`.
        let col_belt: Option<&Vec<ArkFr>> = match belt_hashes {
            Some(belt) => {
                let b = &belt[col_idx];
                if b.len() != col.len() {
                    return Err(invalid("resolve: belt-hash column length mismatch"));
                }
                Some(b)
            }
            None => None,
        };
        let mut out = Vec::with_capacity(col.len());
        for (row, value) in col.iter().enumerate() {
            out.push(resolver.resolve(
                *value,
                *kind,
                col_hints.map(|h| h[row]),
                col_belt.map(|b| b[row]),
            )?);
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
///
/// `openings`, when supplied (C3), is the prover-shipped, UNCOMMITTED [`ResolutionOpenings`] used to
/// resolve ids routed to a TERM-LESS view (whose dropped catalog cannot rebuild its tree) — the
/// term-less reconciled walk/closure output-resolution path. `None` on the ordinary term-bearing
/// path (the resolver rebuilds each routed tree from the clear catalog and ignores it); either way
/// each routed view is opened against its own root-bound Merkle root, so `openings` is a proof-side
/// witness, never committed.
///
/// `belt_hashes`, when supplied (the C4 fully-term-less RENDER belt), is one root-bound
/// `hash_object(term(ids[i]))` per delivered id (from [`verify_value_render_belt`]); it is threaded
/// as the MANDATORY belt-hash so each id routed to a NON-HEAD term-less view opens to a term whose
/// hash equals the authenticated belt hash. `None` on the term-bearing / term-less-HEAD path (a
/// non-head term-less id stays fail-closed, byte-identically to before the belt).
pub(crate) fn resolve_node_value_id_column(
    views: &[VerifierCommitmentRef<'_>],
    ids: &[u64],
    com: ArkGT,
    log_domain: usize,
    openings: Option<&ResolutionOpenings>,
    belt_hashes: Option<&[ArkFr]>,
) -> io::Result<Vec<CanonicalObject>> {
    let column: Vec<ArkFr> = ids.iter().map(|id| ark(*id)).collect();
    // Forward the single-column belt (if any) as a one-element per-column belt aligned to `ids`.
    let belt_cols: Option<Vec<Vec<ArkFr>>> = belt_hashes.map(|b| vec![b.to_vec()]);
    let mut resolved = resolve_result_columns(
        views,
        &[column],
        &[com],
        ids.len(),
        log_domain,
        &[ResolveKind::NodeValueId],
        None,
        // Term-less routed views open from the prover-shipped `openings`; a term-bearing chain
        // passes `None` and the resolver rebuilds from the clear catalogs.
        openings,
        belt_cols.as_deref(),
    )?;
    resolved
        .pop()
        .ok_or_else(|| invalid("resolve: empty node/value-id column"))
}

// ===========================================================================
// C4 — the fully-term-less RENDER belt: a root-bound `id → hash` certificate that lets an ordered
// array/list walk resolve its ANCESTOR-minted value ids even when EVERY layer view is term-less.
//
// The ordered-walk render (`resolve_node_value_id_column`) delivers value combined-dict ids that may
// route to a NON-HEAD term-less view, whose `nv_merkle_root` is prover-controlled (never folded into
// `commitment_root`, `resolution_root` unchecked) — so an unbelted open there is fail-closed (the D5
// forgery). The belt closes the gap WITHOUT trusting that root: it proves a per-row hash column `H`
// where `(v_i, H_i)` is a row of the ROOT-BOUND combined `(id, hash)` node/value dict
// (`reconcile.combined_nv`), i.e. `H_i = hash_object(term(v_i))`. `H_i` is then the MANDATORY
// belt-hash the render's `resolve_nv_id` checks against the Merkle-opened term — a forged term hashes
// to something ≠ `H_i` and rejects (Blake2b 2nd-preimage resistance + no-re-mint distinctness). The
// belt is STRICTLY ADDITIVE: absent it (or if any of its relocations/lookup/recommit fail) the walk
// stays fail-closed. It reuses ONLY audited gadgets (`PresentIdAuth`'s `prove/verify_hash_lookup_inner`,
// column relocation, the reconciliation's combined dict) — no new crypto.
// ===========================================================================

/// The fully-term-less value-render belt (§C4): binds the walk's delivered value-id column to a
/// root-bound `id → hash` lookup against the reconciliation's combined node/value dict, yielding one
/// authenticated `hash_object(term(id))` per real row. Every field is `Wire` (the walk proof carries
/// it as an `Option`). See [`prove_value_render_belt`] / [`verify_value_render_belt`].
#[derive(Clone, Debug)]
pub struct ValueRenderBelt {
    /// The belt working domain `2^belt_log = 2^max(L, working_log)` (`L` = the walk's value-id column
    /// domain, `working_log` = the reconciliation domain). Verifier-recomputed; a mismatch rejects.
    pub belt_log: usize,
    /// Relocation binding the walk's `com_out` value-id column (at `L`) up to the belt domain.
    pub reloc_ids: RelocationProof,
    pub com_ids_at_belt: ArkGT,
    /// Relocation binding the root-bound combined-dict id column (`reconcile.combined_nv.com_id_at_k`,
    /// at `working_log`) up to the belt domain.
    pub reloc_dict_id: RelocationProof,
    pub com_dict_id_at_belt: ArkGT,
    /// Relocation binding the root-bound combined-dict hash column (`reconcile.combined_nv.com_hash_at_k`)
    /// up to the belt domain.
    pub reloc_dict_hash: RelocationProof,
    pub com_dict_hash_at_belt: ArkGT,
    /// The audited `id → hash` inner lookup: its `com_v_hash` commits the TRUE per-row dict hash of
    /// each delivered value id.
    pub lookup: HashLookupInnerProof,
    /// The real-prefix authenticated hashes `H_i = hash_object(term(v_i))`; recommitted (padded to
    /// `2^belt_log`) it MUST equal `lookup.com_v_hash`. Returned by the verifier as the render belt.
    pub value_hashes: Vec<ArkFr>,
}

/// PRODUCER (C4): build a [`ValueRenderBelt`] for a fully-term-less ordered walk, on the caller's
/// fresh walk-bound transcript (`aw_transcript`/`lw_transcript` with tag `b"value-render-belt"`).
/// `com_out` is the walk's delivered value-id commitment (at `domain_log = L`); `value_ids` its real
/// delivered ids (`real_n = value_ids.len()`); `working_log` the reconciliation working domain. The
/// combined `(hash, id)` dict columns are REBUILT here (deterministic bag union, identical to
/// `prove_reconcile_core`) so the belt relocations bind against the SAME root-bound
/// `reconcile.combined_nv` commitments the verifier reads — a THROWAWAY transcript carries the rebuild
/// (its proof material is discarded; only the transcript-independent columns/commitments are used).
pub(crate) fn prove_value_render_belt(
    context: &StoreProverContext<'_>,
    layers: &[&LayerCommitment],
    working_log: usize,
    com_out: ArkGT,
    domain_log: usize,
    value_ids: &[u64],
    transcript: &mut Transcript254,
) -> io::Result<ValueRenderBelt> {
    let n = value_ids.len();
    let belt_log = domain_log.max(working_log);
    let head_nv = layers
        .last()
        .ok_or_else(|| invalid("value-render belt: empty layer chain"))?
        .state
        .node_value_count as usize;

    // Rebuild the combined `(hash, id)` node/value dict at `working_log` — byte-for-byte the bag union
    // `prove_reconcile_core` builds (deterministic concatenation of each layer's incremental dict), so
    // `combined.com_{id,hash}_at_k` equal `reconcile.combined_nv.com_{id,hash}_at_k`. On a throwaway
    // transcript: the verifier never replays this rebuild (it reuses the reconcile commitments).
    let mut nv_branches = Vec::new();
    for layer in layers {
        if !layer.node_value_catalog.is_empty() {
            let (h, id) = layer.node_value_dictionary_columns()?;
            let dn = layer.node_value_catalog.len();
            nv_branches.push(RelationWitness {
                columns: vec![h, id],
                commitments: vec![
                    layer.node_value_dict_commitments.0,
                    layer.node_value_dict_commitments.1,
                ],
                log_domain: dict_log(dn)?,
                n: dn,
            });
        }
    }
    let mut throwaway = Transcript254::new(b"pow/value-render-belt/dict-rebuild");
    let combined = prove_combined_dict(context, &nv_branches, working_log, &mut throwaway)?;

    // (2) The delivered value-id column at `L` (== `com_out`), relocated up to the belt domain.
    let ids_data: Vec<ArkFr> = value_ids.iter().map(|v| ark(*v)).collect();
    let ids_at_l = pad_to(&ids_data, domain_log);
    let ids_at_belt = pad_to(&ids_at_l, belt_log);
    let com_ids_at_belt = context.commit_column(&ids_at_belt)?;
    let reloc_ids = prove_column_relocation_with_context(
        context, &ids_at_l, &com_out, &ids_at_belt, &com_ids_at_belt, transcript,
    )?;

    // (3) The root-bound combined-dict `(id, hash)` columns, relocated up to the belt domain.
    let dict_id_at_belt = pad_to(&combined.id_at_k, belt_log);
    let dict_hash_at_belt = pad_to(&combined.hash_at_k, belt_log);
    let com_dict_id_at_belt = context.commit_column(&dict_id_at_belt)?;
    let com_dict_hash_at_belt = context.commit_column(&dict_hash_at_belt)?;
    let reloc_dict_id = prove_column_relocation_with_context(
        context, &combined.id_at_k, &combined.com_id_at_k, &dict_id_at_belt, &com_dict_id_at_belt, transcript,
    )?;
    let reloc_dict_hash = prove_column_relocation_with_context(
        context, &combined.hash_at_k, &combined.com_hash_at_k, &dict_hash_at_belt, &com_dict_hash_at_belt, transcript,
    )?;

    // (4) The audited `id → hash` inner lookup over the delivered ids (real_n = n, n_dict = head count).
    let lookup = prove_hash_lookup_inner(
        context, &ids_at_belt, &com_ids_at_belt, &dict_id_at_belt, &dict_hash_at_belt,
        &com_dict_id_at_belt, &com_dict_hash_at_belt, n, head_nv, transcript,
    )?;
    // (5) The real-prefix hash column (honest witness; padding rows are zero and recommit as zero).
    let full_hashes = honest_hash_lookup(&ids_at_belt, &dict_id_at_belt, &dict_hash_at_belt, n, head_nv).0;
    let value_hashes = full_hashes[..n].to_vec();

    Ok(ValueRenderBelt {
        belt_log,
        reloc_ids,
        com_ids_at_belt,
        reloc_dict_id,
        com_dict_id_at_belt,
        reloc_dict_hash,
        com_dict_hash_at_belt,
        lookup,
        value_hashes,
    })
}

/// VERIFIER (C4): re-verify a [`ValueRenderBelt`] on the caller's fresh walk-bound transcript and
/// return the `n` root-bound render hashes `H_i = hash_object(term(v_i))`. `com_out` is the walk's
/// value-id commitment (at `domain_log = L`), `n` its real row count, `working_log` the
/// reconciliation domain. Steps: (1) `belt_log == max(L, working_log)`; (2) bind the belt id column to
/// `com_out`; (3) bind the belt dict columns to the ROOT-BOUND `reconcile.combined_nv`; (4) the
/// authenticated inner lookup with `n_dict` VERIFIER-DERIVED from the head `node_value_count` (never
/// prover-chosen); (5) recommit `value_hashes` (padded to `2^belt_log`) equals `lookup.com_v_hash`.
/// Any failure rejects, leaving the walk render fail-closed. The caller (`verify_reconcile_core`) has
/// already re-verified `combined_nv` against the root-bound per-layer branches before this runs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_value_render_belt(
    views: &[VerifierCommitmentRef<'_>],
    reconcile: &ReconcileProof,
    com_out: ArkGT,
    n: usize,
    domain_log: usize,
    working_log: usize,
    belt: &ValueRenderBelt,
    transcript: &mut Transcript254,
) -> io::Result<Vec<ArkFr>> {
    let belt_log = domain_log.max(working_log);
    if belt.belt_log != belt_log {
        return Err(invalid(
            "value-render belt: belt_log does not equal max(value-column log, reconcile working log)",
        ));
    }
    // (2) The belt id column IS the walk's delivered value-id column (bound to `com_out`).
    verify_column_relocation(
        &com_out, domain_log, &belt.com_ids_at_belt, belt_log, &belt.reloc_ids, transcript,
    )?;
    // (3) The belt dict columns ARE the reconciliation's root-bound combined `(id, hash)` dict.
    verify_column_relocation(
        &reconcile.combined_nv.com_id_at_k, working_log, &belt.com_dict_id_at_belt, belt_log,
        &belt.reloc_dict_id, transcript,
    )?;
    verify_column_relocation(
        &reconcile.combined_nv.com_hash_at_k, working_log, &belt.com_dict_hash_at_belt, belt_log,
        &belt.reloc_dict_hash, transcript,
    )?;
    // (4) Authenticated `id → hash`: `n_dict` is the head node/value count (VERIFIER-DERIVED — the
    // combined dict has exactly `head_nv` real rows; a prover-chosen `n_dict` cannot mask a pole).
    let head_nv = views
        .last()
        .ok_or_else(|| invalid("value-render belt: empty view chain"))?
        .state
        .node_value_count as usize;
    verify_hash_lookup_inner(
        &belt.com_ids_at_belt, &belt.com_dict_id_at_belt, &belt.com_dict_hash_at_belt, n, head_nv,
        &belt.lookup, transcript,
    )?;
    // (5) The delivered `value_hashes` (real prefix) recommit to the authenticated lookup output.
    if belt.value_hashes.len() != n {
        return Err(invalid("value-render belt: value_hashes length does not match the row count"));
    }
    let mut padded = belt.value_hashes.clone();
    padded.resize(1usize << belt_log, f0());
    if commit_column(&padded)? != belt.lookup.com_v_hash {
        return Err(invalid(
            "value-render belt: recommitted value hashes do not open the authenticated id→hash lookup",
        ));
    }
    Ok(belt.value_hashes.clone())
}

/// C3 (term-less reconciled CONSTANT resolution): route+open a prover-CLAIMED node/value `id`
/// against its introducing view's root-bound `nv_merkle_root` (via the single prover-shipped
/// `opening`) and BELT-HASH the opened term against `expected` (`hash_object` of the constant). The
/// reconciled analogue of the S4 `belt_hash_node_value_constant` (`query::layer_bgp`): on a
/// term-less view the combined-dict hash columns are prover-chosen, so a catalog SCAN reads empty
/// and would fail-OPEN to a garbage id a prover could steer (the closure seed → a wrong-but-real
/// id). Instead this drives the audited [`ChainDictResolver`], which routes `id` to its layer and
/// re-forces the interval + structural position under `verify_opening`, then requires
/// `hash_object(opened) == hash_object(expected)`. A wrong id, a substituted term, or a missing
/// opening fails closed. If a routed view is itself term-bearing the resolver rebuilds+opens its
/// tree and the shipped opening is redundant (still belt-checked). Returns unit on success.
pub(crate) fn resolve_termless_node_constant(
    views: &[VerifierCommitmentRef<'_>],
    id: u64,
    expected: &CanonicalObject,
    opening: &DictOpening,
) -> io::Result<()> {
    if opening.id != id {
        return Err(invalid(
            "term-less reconciled constant: opening id does not match the claimed id",
        ));
    }
    let mut openings = ResolutionOpenings::default();
    openings.insert_node_value(opening.clone());
    // `resolve_nv_id` with `Some(expected_hash)` performs the mandatory belt-hash (opened term must
    // hash to `hash_object(expected)`), rejecting a wrong id / substituted term.
    ChainDictResolver::new(views, Some(&openings)).resolve_nv_id(id, Some(hash_object(expected)))?;
    Ok(())
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
    let views: Vec<VerifierCommitmentRef<'_>> = layers.iter().map(|l| l.verifier_view()).collect();
    let verified = verify_reconcile_core(&views, trusted_head_root, reconcile, &mut transcript)?;
    let out_len = verify_present_filter(&verified, position, constant_hash, filter, &mut transcript)?;
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
        // No RENDER belt on the single-atom path (hash-kind columns already belt each id-hint).
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
    let views: Vec<VerifierCommitmentRef<'_>> = layers.iter().map(|l| l.verifier_view()).collect();
    let verified = verify_reconcile_core(&views, trusted_head_root, reconcile, &mut transcript)?;
    let head_nv = layers.last().unwrap().verifier_view().state.node_value_count as usize;
    let out_len = verify_present_join(reconcile, &verified, head_nv, const_p, const_q, join, &mut transcript)?;
    // Join output [id_x, id_y, id_z] are all node/value combined-dict ids (no id-hints needed).
    let kinds = [ResolveKind::NodeValueId, ResolveKind::NodeValueId, ResolveKind::NodeValueId];
    resolve_result_columns(&views, claimed_columns, &join.com_join_out, out_len, join.join_log, &kinds, None, None, None)
}

