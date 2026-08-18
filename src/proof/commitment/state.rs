//! What a layer commits: the committed-state types and their serialization.
//!
//! These are the data structures that make up a layer's trust anchor. [`CommitmentState`] is
//! the tiny authenticated summary (parent root, the dictionary/triple counts, and the single
//! 32-byte `commitment_root`). [`LayerCommitment`] is the full prover-side sidecar: the state
//! plus the clear column tables and every Dory commitment over them. [`VerifierCommitment`]
//! (and the borrowed [`VerifierCommitmentRef`]) are the trimmed verifier-only projection —
//! the public commitments and clear dictionary catalogs a remote checker needs, WITHOUT the
//! prover-only witness tables — reached through the [`VerifierLayer`] trait. This module also
//! carries the `impl LayerCommitment` build/validate/encode/decode logic and the
//! GT-commitment (de)serialization helpers. The actual root arithmetic that ties all these
//! commitments together lives next door in [`super::root`].

use super::*;

pub(super) fn serialize_commitment(commitment: &ArkGT) -> io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    commitment
        .serialize_compressed(&mut bytes)
        .map_err(|error| invalid(format!("commitment serialization failed: {error}")))?;
    Ok(bytes)
}

pub(super) fn deserialize_commitment(bytes: &[u8]) -> io::Result<ArkGT> {
    ArkGT::deserialize_compressed(bytes)
        .map_err(|error| invalid(format!("commitment deserialization failed: {error}")))
}

/// Small authenticated summary of a layer's committed delta. Mirrors the
/// fields the old Merkle `ProofLayerState` exposed, minus the per-index
/// Merkle roots, plus a single `commitment_root`.
///
/// `PartialEq` is load-bearing: the storage plumbing compares a rebuilt
/// state against the stored state in `validate_proof_chain_against_store`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommitmentState {
    pub parent_state_root: Option<ProofHash>,
    pub addition_count: u64,
    pub removal_count: u64,
    pub materialized_count: u64,
    pub parent_node_value_count: u64,
    pub parent_predicate_count: u64,
    pub node_value_count: u64,
    pub predicate_count: u64,
    pub commitment_root: ProofHash,
}

impl CommitmentState {
    pub fn validate(&self) -> io::Result<()> {
        if self.parent_node_value_count > self.node_value_count
            || self.parent_predicate_count > self.predicate_count
            || (self.parent_state_root.is_none()
                && (self.parent_node_value_count != 0 || self.parent_predicate_count != 0))
        {
            return Err(invalid("invalid dictionary parent counts"));
        }
        Ok(())
    }
}

/// Validated per-layer sidecar payload. Storage treats the encoded form as
/// opaque bytes.
#[derive(Clone, Debug)]
pub struct LayerCommitment {
    pub state: CommitmentState,
    // The dictionary-encoded triple columns, kept sorted (SPO for the delta,
    // dictionary order for catalogs) so commitments are deterministic.
    pub additions: Vec<ProofRecord>,
    pub removals: Vec<ProofRecord>,
    /// Complete logical triple set visible at this immutable layer head.
    pub materialized: Vec<ProofRecord>,
    pub predicate_catalog: Vec<ProofRecord>,
    pub node_value_catalog: Vec<ProofRecord>,
    /// Per-column Dory (tier-2 GT) commitments over the id columns, in fixed
    /// order: additions (S,P,O) then removals (S,P,O). `commitment_root`
    /// binds these. Phase 1b commits each column standalone at offset 0.
    pub column_commitments: [ArkGT; COLUMN_COUNT],
    /// Dory commitments over the complete logical S/P/O columns.
    pub materialized_column_commitments: [ArkGT; MATERIALIZED_COLUMN_COUNT],
    /// Dory commitments of the layer's PREDICATE dictionary as two columns
    /// `(Com(pred_dict_hash), Com(pred_dict_id))`, where `pred_dict_hash[k] =
    /// hash_term(predicate_bytes_k)` and `pred_dict_id[k] = predicate_id_k`
    /// over `predicate_catalog`'s canonical order (padded to a power of two).
    /// `commitment_root` binds these, so a Q3 dictionary-membership proof of an
    /// IRI->id binding is authenticated by the root.
    pub predicate_dict_commitments: (ArkGT, ArkGT),
    /// Dory commitments of the layer's NODE/VALUE dictionary as two columns
    /// `(Com(nv_dict_hash), Com(nv_dict_id))`, where `nv_dict_hash[k] =
    /// hash_object(object_k)` and `nv_dict_id[k] = object_id_k` over
    /// `node_value_catalog`'s canonical order (padded to a power of two).
    /// `commitment_root` binds these, so a Q3 dictionary-membership proof of a
    /// subject/object CONSTANT->id binding is authenticated by the root.
    pub node_value_dict_commitments: (ArkGT, ArkGT),
    /// Dory commitments of TWO extra node/value-dictionary columns that back
    /// the (later) numeric RANGE gadget, over `node_value_catalog`'s canonical
    /// order (same padding/length as `node_value_dict_commitments`):
    ///   `.0` = `Com(nv_dict_value)`: `nv_dict_value[k]` is the ORDER-PRESERVING
    ///          unsigned-integer encoding of object_k's value when object_k is a
    ///          supported fixed-width numeric literal, else the `0` sentinel
    ///          (`Node`s, `String`/variable-width or non-OP datatypes). See
    ///          `node_value_dict_value_column`.
    ///   `.1` = `Com(nv_dict_datatype)`: `nv_dict_datatype[k]` is object_k's
    ///          `CanonicalDatatype` discriminant (a `Value`) or the reserved
    ///          `NODE_DATATYPE_TAG` (a `Node`); lets the later gadget mask rows
    ///          to the queried datatype by equality. See
    ///          `node_value_dict_datatype_column`.
    /// `commitment_root` binds these two, so the value column a range proof
    /// compares against — and the datatype it is masked by — are authenticated
    /// by the root. The `object_id` column is NOT value-ordered on rollup
    /// layers, so the range gadget ranges over this committed OP encoding and
    /// ties each triple's object to its dict row by a per-row membership lookup.
    pub node_value_dict_range_commitments: (ArkGT, ArkGT),
    /// Root of the id-indexed SHA-256 Merkle tree over `node_value_catalog`
    /// (leaf `p` = the id `parent_node_value_count + p + 1`), for succinct
    /// id->term resolution (I1). ADDITIVE: NOT bound by `commitment_root`; it is
    /// a deterministic function of the catalog and is bound instead by the
    /// `resolution_root` wrapper (and re-checked against the root-bound catalog
    /// in `validate`). See `super::dict_merkle`.
    pub nv_merkle_root: ProofHash,
    /// Root of the id-indexed SHA-256 Merkle tree over `predicate_catalog`
    /// (separate id namespace). Same additive discipline as `nv_merkle_root`.
    pub pred_merkle_root: ProofHash,
}

/// Compact, verifier-only projection of a validated PF4 v4 layer commitment.
///
/// It deliberately omits the native addition/removal tables and the complete
/// materialized triple table. Those are prover witnesses. The retained state,
/// column commitments and complete clear dictionary catalogs are exactly the
/// public inputs needed to recompute the trusted layer root, authenticate
/// constants and render typed result ids. This type exposes no conversion back
/// to [`LayerCommitment`], because the omitted witness tables cannot be
/// reconstructed from commitments.
#[derive(Clone, Debug)]
pub struct VerifierCommitment {
    pub state: CommitmentState,
    /// TERM-BEARING (`true`) vs TERM-LESS (`false`) framing (Merkle-dict I2b S1).
    /// When `true`, the two clear catalogs below are present and `validate()`
    /// re-MSMs them + recomputes the Merkle roots, exactly as before (v3
    /// behavior). When `false`, the catalogs are ABSENT (empty) — the O(dict)
    /// term bytes are dropped; every fixed-size commitment + both id-indexed
    /// Merkle roots are retained, so the artifact is O(1) in dictionary size and
    /// still recomputes `commitment_root`/`resolution_root`. A term-less artifact
    /// is BASE/ROLLUP ONLY (a delta needs cross-layer re-mint over faithful
    /// hashes, blocked on I4) and is NOT yet queryable — S2–S4 add the
    /// fail-close guards, opening-based resolution, and per-constant belt-hash.
    /// This flag lives in the verifier-artifact FRAMING only; it is NOT folded
    /// into `commitment_root` (which stays byte-identical to the term-bearing
    /// artifact of the same layer).
    pub terms_present: bool,
    pub predicate_catalog: Vec<ProofRecord>,
    pub node_value_catalog: Vec<ProofRecord>,
    pub column_commitments: [ArkGT; COLUMN_COUNT],
    pub materialized_column_commitments: [ArkGT; MATERIALIZED_COLUMN_COUNT],
    pub predicate_dict_commitments: (ArkGT, ArkGT),
    pub node_value_dict_commitments: (ArkGT, ArkGT),
    pub node_value_dict_range_commitments: (ArkGT, ArkGT),
    /// Id-indexed node/value and predicate dictionary Merkle roots (I1); see
    /// [`LayerCommitment::nv_merkle_root`].
    pub nv_merkle_root: ProofHash,
    pub pred_merkle_root: ProofHash,
}

/// Borrowed verifier view shared by full PF4 sidecars and compact artifacts.
#[derive(Clone, Copy)]
pub struct VerifierCommitmentRef<'a> {
    pub state: &'a CommitmentState,
    /// Mirrors [`VerifierCommitment::terms_present`]: `false` ⇒ the two catalog
    /// slices below are empty and `validate()` skips the catalog re-MSM + Merkle
    /// recompute. A [`LayerCommitment`] view is always `true` (it carries the
    /// full catalogs); only a term-less artifact's view is `false`.
    pub terms_present: bool,
    pub predicate_catalog: &'a [ProofRecord],
    pub node_value_catalog: &'a [ProofRecord],
    pub column_commitments: &'a [ArkGT; COLUMN_COUNT],
    pub materialized_column_commitments: &'a [ArkGT; MATERIALIZED_COLUMN_COUNT],
    pub predicate_dict_commitments: &'a (ArkGT, ArkGT),
    pub node_value_dict_commitments: &'a (ArkGT, ArkGT),
    pub node_value_dict_range_commitments: &'a (ArkGT, ArkGT),
    /// Id-indexed dictionary Merkle roots (I1). `ProofHash` is `Copy`, so these
    /// are carried by value, keeping the view `Copy`.
    pub nv_merkle_root: ProofHash,
    pub pred_merkle_root: ProofHash,
}

/// Common verifier-facing interface. It intentionally contains no raw-table
/// accessors, keeping prover witness data outside verification APIs.
pub trait VerifierLayer {
    fn verifier_view(&self) -> VerifierCommitmentRef<'_>;

    /// The two-hash `resolution_root` anchor for this layer (β re-anchor): it
    /// binds `commitment_root` (and thereby every id-authentication gadget) to
    /// the two id-indexed dictionary Merkle roots. This is the EXTERNAL trust
    /// anchor a verifier is handed and checks against — binding the Merkle roots
    /// DIRECTLY, not merely transitively through the catalog. A convenience
    /// wrapper over [`VerifierCommitmentRef::resolution_root`] so any layer type
    /// (owned sidecar, compact artifact, borrowed view, `Arc`) exposes it.
    fn resolution_root(&self) -> ProofHash {
        self.verifier_view().resolution_root()
    }

    /// Merkle-dict I2b S2: is this layer's verifier artifact TERM-BEARING (`true`) or
    /// TERM-LESS (`false`)? A term-less artifact dropped its two clear catalogs, so the
    /// `validate()` catalog re-MSM — the ONLY check tying the 6 committed Dory dict columns
    /// (esp. the order-preserving VALUE + DATATYPE columns) to the REAL terms — is skipped.
    /// Every value/datatype/catalog-reading gadget (range, order, group, `dictionary_absence`,
    /// delta re-mint) MUST fail-close when this is `false` (see `require_terms_present`); on a
    /// term-less view those columns are prover-chosen and a gadget would produce a
    /// wrong-but-verifying result. A full prover sidecar ([`LayerCommitment`]) is always `true`.
    fn terms_present(&self) -> bool {
        self.verifier_view().terms_present
    }
}

impl VerifierLayer for LayerCommitment {
    fn verifier_view(&self) -> VerifierCommitmentRef<'_> {
        VerifierCommitmentRef {
            state: &self.state,
            // A full prover sidecar always carries its clear catalogs.
            terms_present: true,
            predicate_catalog: &self.predicate_catalog,
            node_value_catalog: &self.node_value_catalog,
            column_commitments: &self.column_commitments,
            materialized_column_commitments: &self.materialized_column_commitments,
            predicate_dict_commitments: &self.predicate_dict_commitments,
            node_value_dict_commitments: &self.node_value_dict_commitments,
            node_value_dict_range_commitments: &self.node_value_dict_range_commitments,
            nv_merkle_root: self.nv_merkle_root,
            pred_merkle_root: self.pred_merkle_root,
        }
    }
}

impl VerifierLayer for VerifierCommitment {
    fn verifier_view(&self) -> VerifierCommitmentRef<'_> {
        VerifierCommitmentRef {
            state: &self.state,
            terms_present: self.terms_present,
            predicate_catalog: &self.predicate_catalog,
            node_value_catalog: &self.node_value_catalog,
            column_commitments: &self.column_commitments,
            materialized_column_commitments: &self.materialized_column_commitments,
            predicate_dict_commitments: &self.predicate_dict_commitments,
            node_value_dict_commitments: &self.node_value_dict_commitments,
            node_value_dict_range_commitments: &self.node_value_dict_range_commitments,
            nv_merkle_root: self.nv_merkle_root,
            pred_merkle_root: self.pred_merkle_root,
        }
    }
}

impl VerifierLayer for VerifierCommitmentRef<'_> {
    fn verifier_view(&self) -> VerifierCommitmentRef<'_> {
        *self
    }
}

impl<T: VerifierLayer + ?Sized> VerifierLayer for std::sync::Arc<T> {
    fn verifier_view(&self) -> VerifierCommitmentRef<'_> {
        (**self).verifier_view()
    }
}

impl VerifierCommitmentRef<'_> {
    pub fn validate(self) -> io::Result<()> {
        self.state.validate()?;
        if self.terms_present {
            let subject_catalog = subject_catalog_from_objects(self.node_value_catalog)?;
            validate_complete_catalog(
                &[],
                &[],
                &[],
                [
                    &subject_catalog,
                    self.predicate_catalog,
                    self.node_value_catalog,
                ],
                self.state.parent_node_value_count,
                self.state.parent_predicate_count,
                self.state.node_value_count,
                self.state.predicate_count,
            )?;
            if commit_predicate_dictionary(self.predicate_catalog)?
                != *self.predicate_dict_commitments
                || commit_node_value_dictionary(self.node_value_catalog)?
                    != *self.node_value_dict_commitments
                || commit_node_value_dict_range(self.node_value_catalog)?
                    != *self.node_value_dict_range_commitments
            {
                return Err(invalid("verifier catalog commitment mismatch"));
            }
            self.validate_merkle_roots()?;
        } else {
            // TERM-LESS (Merkle-dict I2b S1): the clear catalogs are dropped, so the
            // catalog re-MSM (the only check tying the 6 committed Dory dict columns to
            // real terms) and the catalog-based Merkle-root RECOMPUTE are BOTH skipped.
            // The two Merkle roots are instead bound by the external `resolution_root`
            // anchor a verifier checks (β re-anchor). We still fail closed on two
            // structural invariants: the catalogs must genuinely be absent, and a
            // term-less artifact is BASE/ROLLUP only (a delta needs cross-layer re-mint
            // over faithful hashes = blocked on I4). Everything catalog-free below is
            // KEPT — critically `recomputed_commitment_root()` (which recomputes from the
            // STORED commitments + counts, NOT the catalog).
            if !self.predicate_catalog.is_empty() || !self.node_value_catalog.is_empty() {
                return Err(invalid("term-less verifier commitment must not carry catalogs"));
            }
            if self.state.parent_state_root.is_some() {
                return Err(invalid(
                    "term-less verifier commitment must be a base/rollup layer",
                ));
            }
        }
        if self.recomputed_commitment_root()? != self.state.commitment_root {
            return Err(invalid("verifier commitment root mismatch"));
        }
        Ok(())
    }

    /// Recompute the two id-indexed dictionary Merkle roots (I1) from the
    /// now-authenticated catalog and confirm they match the carried roots. Because
    /// the catalog is bound to `commitment_root` by the Dory dictionary commitments
    /// checked just above, this transitively authenticates the Merkle roots — a
    /// tampered root (or a catalog that no longer hashes to it) rejects here.
    fn validate_merkle_roots(self) -> io::Result<()> {
        if node_value_merkle_root(
            self.node_value_catalog,
            self.state.parent_node_value_count,
            self.state.node_value_count,
        )? != self.nv_merkle_root
            || predicate_merkle_root(
                self.predicate_catalog,
                self.state.parent_predicate_count,
                self.state.predicate_count,
            )? != self.pred_merkle_root
        {
            return Err(invalid("verifier dictionary merkle root mismatch"));
        }
        Ok(())
    }

    /// The two-hash resolution anchor for this layer: it binds `commitment_root`
    /// (and thereby every id-authentication gadget) to the two dictionary Merkle
    /// roots that succinct id->term resolution opens against.
    pub fn resolution_root(self) -> ProofHash {
        resolution_root(
            &self.state.commitment_root,
            &self.nv_merkle_root,
            &self.pred_merkle_root,
        )
    }

    pub fn recomputed_commitment_root(self) -> io::Result<ProofHash> {
        compute_commitment_root_from_parts(
            self.state,
            self.column_commitments,
            self.materialized_column_commitments,
            self.predicate_dict_commitments,
            self.node_value_dict_commitments,
            self.node_value_dict_range_commitments,
        )
    }

    /// Merkle-dict I2b S2 fail-close BELT. Reject any read that would consume the (now
    /// prover-untied) value/datatype/catalog data on a TERM-LESS artifact. This is the
    /// exhaustiveness backstop behind the per-gadget entry guards: on a term-less view the
    /// clear catalogs are empty and the catalog re-MSM is skipped, so `validate()` no longer
    /// ties the committed VALUE/DATATYPE columns to real terms — any range/order/group/absence/
    /// re-mint reader here would silently read an empty catalog (or prover-chosen columns) and
    /// return a wrong-but-verifying result. Fail closed instead. `what` names the reader for the
    /// error. (Term-bearing is a no-op; a full [`LayerCommitment`] view is always term-bearing.)
    pub fn require_terms_present(self, what: &str) -> io::Result<()> {
        if !self.terms_present {
            return Err(invalid(format!(
                "{what} requires a terms-bearing artifact \
                 (term-less Merkle-dict I2b artifact fail-closes value/datatype/catalog reads)"
            )));
        }
        Ok(())
    }

    /// Rebuild the padded node/value-dictionary `(hash, id)` columns. Merkle-dict I2b S2:
    /// `Err` on a TERM-LESS view (belt) — these columns feed range/order/group, never the
    /// commitment-based constant-membership path, so gating them term-less is precise.
    pub fn node_value_dictionary_columns(self) -> io::Result<(Vec<ArkFr>, Vec<ArkFr>)> {
        self.require_terms_present("node/value dictionary (hash,id) column reconstruction")?;
        node_value_dict_columns_from_catalog(self.node_value_catalog)
    }

    /// Rebuild the padded predicate-dictionary `(hash, id)` columns. Merkle-dict I2b S2:
    /// `Err` on a TERM-LESS view (belt).
    pub fn predicate_dictionary_columns(self) -> io::Result<(Vec<ArkFr>, Vec<ArkFr>)> {
        self.require_terms_present("predicate dictionary (hash,id) column reconstruction")?;
        predicate_dict_columns_from_catalog(self.predicate_catalog)
    }

    /// Rebuild the padded order-preserving node/value VALUE column that
    /// `node_value_dict_range_commitments.0` commits. Merkle-dict I2b S2: `Err` on a TERM-LESS
    /// view — this OP-value column is read ONLY by numeric range/order/group, whose soundness
    /// rests on the catalog re-MSM a term-less artifact drops. THE core belt accessor.
    pub fn node_value_dictionary_value_column(self) -> io::Result<Vec<ArkFr>> {
        self.require_terms_present("order-preserving VALUE column (numeric range/order/group)")?;
        node_value_dict_value_column(self.node_value_catalog)
    }

    /// Rebuild the padded node/value DATATYPE column that `node_value_dict_range_commitments.1`
    /// commits. Merkle-dict I2b S2: `Err` on a TERM-LESS view — same reasoning as the VALUE column.
    pub fn node_value_dictionary_datatype_column(self) -> io::Result<Vec<ArkFr>> {
        self.require_terms_present("DATATYPE column (numeric range/order/group)")?;
        node_value_dict_datatype_column(self.node_value_catalog)
    }
}

impl PartialEq for LayerCommitment {
    fn eq(&self, other: &Self) -> bool {
        // A validated commitment's state summarizes every column table; two
        // decodings that reconstruct the same authenticated state compare
        // equal regardless of any incidental table-layout choices.
        self.state == other.state
    }
}

impl Eq for LayerCommitment {}

impl LayerCommitment {
    /// β re-anchor: the layer's two-hash `resolution_root` anchor (binds `commitment_root`
    /// AND both id-indexed dictionary Merkle roots). Inherent convenience so callers/tests
    /// reach it on an owned `LayerCommitment` (or an `Arc<LayerCommitment>` via `Deref`)
    /// without importing the [`VerifierLayer`] trait.
    pub fn resolution_root(&self) -> ProofHash {
        self.verifier_view().resolution_root()
    }

    /// Deterministically project this fully validated PF4 v4 sidecar into the
    /// bounded public artifact needed by a remote verifier.
    pub fn verifier_commitment(&self) -> io::Result<VerifierCommitment> {
        self.validate()?;
        let artifact = VerifierCommitment {
            state: self.state.clone(),
            terms_present: true,
            predicate_catalog: self.predicate_catalog.clone(),
            node_value_catalog: self.node_value_catalog.clone(),
            column_commitments: self.column_commitments,
            materialized_column_commitments: self.materialized_column_commitments,
            predicate_dict_commitments: self.predicate_dict_commitments,
            node_value_dict_commitments: self.node_value_dict_commitments,
            node_value_dict_range_commitments: self.node_value_dict_range_commitments,
            nv_merkle_root: self.nv_merkle_root,
            pred_merkle_root: self.pred_merkle_root,
        };
        artifact.validate()?;
        Ok(artifact)
    }

    /// Merkle-dict I2b S1: project this sidecar into the TERM-LESS public
    /// artifact — every fixed-size commitment + both id-indexed dictionary Merkle
    /// roots + the state, but WITHOUT the two O(dict) clear catalogs. The result
    /// is O(1) in dictionary size (killing the 256 MiB catalog shipping) and
    /// round-trips through `encode`/`decode`/`validate`, but it is NOT yet
    /// queryable: resolution/range/order/group/re-mint all still read the (now
    /// absent) catalog and are wired to fail-close only in S2–S4.
    ///
    /// BASE/ROLLUP ONLY. A delta layer (`parent_state_root == Some`) is rejected:
    /// term-less delta querying needs cross-layer re-mint over faithful hashes,
    /// which is blocked on I4; the base is the only directly-query-provable layer.
    ///
    /// `commitment_root` is byte-identical to the term-bearing artifact of the
    /// same layer — the `terms_present` flag lives in the artifact FRAMING, not in
    /// `commitment_root`.
    pub fn verifier_commitment_termless(&self) -> io::Result<VerifierCommitment> {
        self.validate()?;
        if self.state.parent_state_root.is_some() {
            return Err(invalid(
                "term-less verifier commitment requires a base/rollup layer \
                 (delta needs cross-layer re-mint over faithful hashes, blocked on I4)",
            ));
        }
        let artifact = VerifierCommitment {
            state: self.state.clone(),
            terms_present: false,
            predicate_catalog: Vec::new(),
            node_value_catalog: Vec::new(),
            column_commitments: self.column_commitments,
            materialized_column_commitments: self.materialized_column_commitments,
            predicate_dict_commitments: self.predicate_dict_commitments,
            node_value_dict_commitments: self.node_value_dict_commitments,
            node_value_dict_range_commitments: self.node_value_dict_range_commitments,
            nv_merkle_root: self.nv_merkle_root,
            pred_merkle_root: self.pred_merkle_root,
        };
        artifact.validate()?;
        Ok(artifact)
    }

    /// Build a production sidecar from the complete native dictionary delta.
    /// Replaces `ProofLayer::new_native_with_catalog`. The supplied mappings
    /// contain only ids introduced by this layer; inherited mappings are
    /// authenticated by the parent commitment state.
    #[allow(clippy::too_many_arguments)]
    pub fn from_native_delta(
        parent_state_root: Option<ProofHash>,
        additions: Vec<ProofRecord>,
        removals: Vec<ProofRecord>,
        materialized: Vec<ProofRecord>,
        parent_node_value_count: u64,
        parent_predicate_count: u64,
        node_value_count: u64,
        predicate_count: u64,
        node_value_mappings: Vec<(CanonicalObject, u64)>,
        predicate_mappings: Vec<(Vec<u8>, u64)>,
    ) -> io::Result<Self> {
        Self::from_native_delta_with_context(
            &crate::proof::argument::StoreProverContext::cpu(),
            parent_state_root,
            additions,
            removals,
            materialized,
            parent_node_value_count,
            parent_predicate_count,
            node_value_count,
            predicate_count,
            node_value_mappings,
            predicate_mappings,
        )
    }

    /// Build a production sidecar with explicitly selected prover arithmetic.
    /// The context affects computation only; validation and wire format remain
    /// identical to [`Self::from_native_delta`].
    #[allow(clippy::too_many_arguments)]
    pub fn from_native_delta_with_context(
        context: &crate::proof::argument::StoreProverContext<'_>,
        parent_state_root: Option<ProofHash>,
        mut additions: Vec<ProofRecord>,
        mut removals: Vec<ProofRecord>,
        mut materialized: Vec<ProofRecord>,
        parent_node_value_count: u64,
        parent_predicate_count: u64,
        node_value_count: u64,
        predicate_count: u64,
        node_value_mappings: Vec<(CanonicalObject, u64)>,
        predicate_mappings: Vec<(Vec<u8>, u64)>,
    ) -> io::Result<Self> {
        // COMMITMENT DIET (Wave 0): a DELTA layer (parent_state_root == Some) commits
        // only its add/remove columns + the INCREMENTAL dictionary it introduced. It
        // must NOT carry the O(state) materialized head view; callers (rebuild) pass an
        // empty table and the incremental (`parent_count+1..=count`) dictionary mappings.
        if parent_state_root.is_some() && !materialized.is_empty() {
            return Err(invalid(
                "delta layer commitment must not carry a materialized head view",
            ));
        }
        validate_construction_heap_budget(
            &additions,
            &removals,
            &materialized,
            &node_value_mappings,
            &predicate_mappings,
        )?;
        if additions.len() > MAX_RECORDS
            || removals.len() > MAX_RECORDS
            || materialized.len() > MAX_RECORDS
        {
            return Err(invalid("proof delta exceeds storage limit"));
        }
        let [subject_catalog, predicate_catalog, node_value_catalog] =
            build_dictionary_catalogs_from_maps(node_value_mappings, predicate_mappings)?;

        let addition_set: BTreeSet<_> = additions.iter().map(|r| &r.triple).collect();
        let removal_set: BTreeSet<_> = removals.iter().map(|r| &r.triple).collect();
        if addition_set.len() != additions.len()
            || removal_set.len() != removals.len()
            || !addition_set.is_disjoint(&removal_set)
        {
            return Err(invalid("duplicate or conflicting native delta records"));
        }
        if additions.iter().chain(removals.iter()).any(|record| {
            record.subject_id == 0
                || record.predicate_id == 0
                || record.object_id == 0
                || record.subject_id > node_value_count
                || record.object_id > node_value_count
                || record.predicate_id > predicate_count
        }) {
            return Err(invalid(
                "native delta record id lies outside dictionary counts",
            ));
        }
        validate_complete_catalog(
            &additions,
            &removals,
            &materialized,
            [
                &subject_catalog,
                &predicate_catalog,
                &node_value_catalog,
            ],
            parent_node_value_count,
            parent_predicate_count,
            node_value_count,
            predicate_count,
        )?;

        additions.sort_by(|a, b| triple_cmp(a, b, ProofIndex::Spo));
        removals.sort_by(|a, b| triple_cmp(a, b, ProofIndex::Spo));
        materialized.sort_by(|a, b| triple_cmp(a, b, ProofIndex::Spo));

        // Real per-column Dory commitments over the sorted id columns.
        let column_commitments =
            commit_delta_columns_with_context(context, &additions, &removals)?;
        let materialized_column_commitments =
            commit_materialized_columns_with_context(context, &materialized)?;
        // Predicate-dictionary commitments (IRI->id binding, authenticated by
        // the root so Q3 membership proofs can trust it).
        let predicate_dict_commitments =
            commit_predicate_dictionary_with_context(context, &predicate_catalog)?;
        // Node/value-dictionary commitments (subject/object CONSTANT->id
        // binding, authenticated by the root the same way).
        let node_value_dict_commitments =
            commit_node_value_dictionary_with_context(context, &node_value_catalog)?;
        // Node/value order-preserving VALUE + DATATYPE columns backing the
        // (later) numeric range gadget, authenticated by the root the same way.
        let node_value_dict_range_commitments =
            commit_node_value_dict_range_with_context(context, &node_value_catalog)?;
        // Id-indexed dictionary Merkle roots (I1). O(delta): built over exactly the
        // catalog this layer introduced, keyed by id position. NOT folded into
        // `commitment_root` (kept byte-identical); bound by `resolution_root`.
        let nv_merkle_root = node_value_merkle_root(
            &node_value_catalog,
            parent_node_value_count,
            node_value_count,
        )?;
        let pred_merkle_root =
            predicate_merkle_root(&predicate_catalog, parent_predicate_count, predicate_count)?;

        let state = CommitmentState {
            parent_state_root,
            addition_count: additions.len() as u64,
            removal_count: removals.len() as u64,
            materialized_count: materialized.len() as u64,
            parent_node_value_count,
            parent_predicate_count,
            node_value_count,
            predicate_count,
            commitment_root: [0; 32],
        };
        let mut layer = Self {
            state,
            additions,
            removals,
            materialized,
            predicate_catalog,
            node_value_catalog,
            column_commitments,
            materialized_column_commitments,
            predicate_dict_commitments,
            node_value_dict_commitments,
            node_value_dict_range_commitments,
            nv_merkle_root,
            pred_merkle_root,
        };
        layer.state.commitment_root = layer.compute_commitment_root()?;
        layer.validate()?;
        Ok(layer)
    }

    /// Conservative retained-byte estimate for the decoded proof cache. It
    /// charges the column-table capacities plus a fixed allocator allowance
    /// per allocation. It is an accounting weight, not an RSS measurement.
    pub fn estimated_cache_bytes(&self) -> usize {
        const ALLOCATION_OVERHEAD: usize = 32;
        let table_bytes = |records: &[ProofRecord]| {
            records.iter().fold(
                std::mem::size_of::<Vec<ProofRecord>>()
                    .saturating_add(
                        records
                            .len()
                            .saturating_mul(std::mem::size_of::<ProofRecord>()),
                    )
                    .saturating_add(ALLOCATION_OVERHEAD),
                |total, record| {
                    let object = match &record.triple.object {
                        CanonicalObject::Node(value) => value.capacity(),
                        CanonicalObject::Value { lexical, .. } => lexical.capacity(),
                    };
                    total
                        .saturating_add(record.triple.subject.capacity())
                        .saturating_add(record.triple.predicate.capacity())
                        .saturating_add(object)
                        .saturating_add(3usize.saturating_mul(ALLOCATION_OVERHEAD))
                },
            )
        };
        std::mem::size_of::<LayerCommitment>()
            .saturating_add(table_bytes(&self.additions))
            .saturating_add(table_bytes(&self.removals))
            .saturating_add(table_bytes(&self.materialized))
            .saturating_add(table_bytes(&self.predicate_catalog))
            .saturating_add(table_bytes(&self.node_value_catalog))
    }

    pub fn validate(&self) -> io::Result<()> {
        self.state.validate()?;
        // COMMITMENT DIET (Wave 0): a delta layer's materialized columns are NOT bound
        // by its root, so they MUST be empty — otherwise a decoded sidecar could smuggle
        // in unbound (untrusted) head rows. A base layer keeps its materialized view.
        if self.state.parent_state_root.is_some()
            && (!self.materialized.is_empty() || self.state.materialized_count != 0)
        {
            return Err(invalid(
                "delta layer commitment must not carry a materialized head view",
            ));
        }
        if self.additions.len() > MAX_RECORDS
            || self.removals.len() > MAX_RECORDS
            || self.materialized.len() > MAX_RECORDS
        {
            return Err(invalid("proof delta exceeds storage limit"));
        }
        if self.additions.len() as u64 != self.state.addition_count
            || self.removals.len() as u64 != self.state.removal_count
            || self.materialized.len() as u64 != self.state.materialized_count
        {
            return Err(invalid("commitment delta count mismatch"));
        }
        for table in [&self.additions, &self.removals, &self.materialized] {
            if table
                .windows(2)
                .any(|pair| triple_cmp(&pair[0], &pair[1], ProofIndex::Spo) != Ordering::Less)
            {
                return Err(invalid("commitment delta records not strictly ordered"));
            }
            if table.iter().any(|record| {
                record.subject_id == 0
                    || record.predicate_id == 0
                    || record.object_id == 0
                    || record.subject_id > self.state.node_value_count
                    || record.object_id > self.state.node_value_count
                    || record.predicate_id > self.state.predicate_count
            }) {
                return Err(invalid(
                    "commitment delta record id lies outside dictionary counts",
                ));
            }
        }
        let addition_set: BTreeSet<_> = self.additions.iter().map(|r| &r.triple).collect();
        let removal_set: BTreeSet<_> = self.removals.iter().map(|r| &r.triple).collect();
        if addition_set.len() != self.additions.len()
            || removal_set.len() != self.removals.len()
            || !addition_set.is_disjoint(&removal_set)
        {
            return Err(invalid("duplicate or conflicting native delta records"));
        }
        let materialized_set: BTreeSet<_> = self
            .materialized
            .iter()
            .map(|record| &record.triple)
            .collect();
        if materialized_set.len() != self.materialized.len() {
            return Err(invalid("duplicate materialized logical records"));
        }
        let subject_catalog = subject_catalog_from_objects(&self.node_value_catalog)?;
        validate_complete_catalog(
            &self.additions,
            &self.removals,
            &self.materialized,
            [
                &subject_catalog,
                &self.predicate_catalog,
                &self.node_value_catalog,
            ],
            self.state.parent_node_value_count,
            self.state.parent_predicate_count,
            self.state.node_value_count,
            self.state.predicate_count,
        )?;
        if commit_delta_columns(&self.additions, &self.removals)? != self.column_commitments {
            return Err(invalid("native delta column commitment mismatch"));
        }
        if commit_materialized_columns(&self.materialized)?
            != self.materialized_column_commitments
        {
            return Err(invalid("materialized column commitment mismatch"));
        }
        if commit_predicate_dictionary(&self.predicate_catalog)?
            != self.predicate_dict_commitments
        {
            return Err(invalid("predicate dictionary commitment mismatch"));
        }
        if commit_node_value_dictionary(&self.node_value_catalog)?
            != self.node_value_dict_commitments
        {
            return Err(invalid("node/value dictionary commitment mismatch"));
        }
        if commit_node_value_dict_range(&self.node_value_catalog)?
            != self.node_value_dict_range_commitments
        {
            return Err(invalid("node/value range commitment mismatch"));
        }
        // I1: the id-indexed dictionary Merkle roots are a deterministic function of
        // the (root-bound) catalogs; recompute and confirm the carried roots match.
        if node_value_merkle_root(
            &self.node_value_catalog,
            self.state.parent_node_value_count,
            self.state.node_value_count,
        )? != self.nv_merkle_root
            || predicate_merkle_root(
                &self.predicate_catalog,
                self.state.parent_predicate_count,
                self.state.predicate_count,
            )? != self.pred_merkle_root
        {
            return Err(invalid("dictionary merkle root mismatch"));
        }
        if self.compute_commitment_root()? != self.state.commitment_root {
            return Err(invalid("commitment root mismatch"));
        }
        Ok(())
    }

    pub fn encode(&self) -> io::Result<Vec<u8>> {
        self.validate()?;
        let expected = self.encoded_len()?;
        let mut out = self.encode_body()?;
        put_hash(&mut out, &self.state.commitment_root);
        let checksum = file_checksum(&out);
        out.extend_from_slice(&checksum);
        if out.len() != expected {
            return Err(invalid("commitment encoded length accounting mismatch"));
        }
        Ok(out)
    }

    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < COMMITMENT_MAGIC.len() + 4 + 32 + 32
            || bytes.len() > MAX_PROOF_LAYER_BYTES
        {
            return Err(invalid("truncated proof commitment"));
        }
        let (payload, checksum) = bytes.split_at(bytes.len() - 32);
        if file_checksum(payload).as_slice() != checksum {
            return Err(invalid("commitment file checksum mismatch"));
        }
        let (body, root_bytes) = payload.split_at(payload.len() - 32);
        let stored_root: ProofHash = root_bytes.try_into().unwrap();

        let mut r = Reader::new(body);
        if r.take(COMMITMENT_MAGIC.len())? != COMMITMENT_MAGIC {
            return Err(invalid("bad commitment file magic"));
        }
        if r.u32()? != COMMITMENT_FORMAT_VERSION {
            return Err(invalid("unsupported commitment file version"));
        }
        let parent_state_root = match r.u8()? {
            0 => None,
            1 => Some(r.hash()?),
            _ => return Err(invalid("bad parent root tag")),
        };
        let addition_count = r.u64()?;
        let removal_count = r.u64()?;
        let materialized_count = r.u64()?;
        let parent_node_value_count = r.u64()?;
        let parent_predicate_count = r.u64()?;
        let node_value_count = r.u64()?;
        let predicate_count = r.u64()?;
        let additions = read_records(&mut r, PersistedTableKind::Delta)?;
        let removals = read_records(&mut r, PersistedTableKind::Delta)?;
        let materialized = read_records(&mut r, PersistedTableKind::Delta)?;
        let predicate_records = read_records(&mut r, PersistedTableKind::PredicateCatalog)?;
        let object_records = read_records(&mut r, PersistedTableKind::ObjectCatalog)?;
        let mut column_commitments: [ArkGT; COLUMN_COUNT] =
            [<ArkGT as Group>::identity(); COLUMN_COUNT];
        for slot in column_commitments.iter_mut() {
            let bytes = r.bytes()?;
            *slot = deserialize_commitment(&bytes)?;
        }
        let mut materialized_column_commitments: [ArkGT; MATERIALIZED_COLUMN_COUNT] =
            [<ArkGT as Group>::identity(); MATERIALIZED_COLUMN_COUNT];
        for slot in materialized_column_commitments.iter_mut() {
            *slot = deserialize_commitment(&r.bytes()?)?;
        }
        let predicate_dict_commitments = {
            let hash = deserialize_commitment(&r.bytes()?)?;
            let id = deserialize_commitment(&r.bytes()?)?;
            (hash, id)
        };
        let node_value_dict_commitments = {
            let hash = deserialize_commitment(&r.bytes()?)?;
            let id = deserialize_commitment(&r.bytes()?)?;
            (hash, id)
        };
        // Same fixed order as `dictionary_commitments`/`encode_body`: the
        // node/value VALUE then DATATYPE commitment follow the hash/id pair.
        let node_value_dict_range_commitments = {
            let value = deserialize_commitment(&r.bytes()?)?;
            let datatype = deserialize_commitment(&r.bytes()?)?;
            (value, datatype)
        };
        // I1: the two id-indexed dictionary Merkle roots follow the commitments.
        // `validate()` below recomputes them from the decoded catalog and rejects a
        // mismatch, so a tampered stored root fails closed.
        let nv_merkle_root = r.hash()?;
        let pred_merkle_root = r.hash()?;
        if !r.done() {
            return Err(invalid("trailing commitment file bytes"));
        }
        validate_decoded_heap_budget(
            &additions,
            &removals,
            &materialized,
            &predicate_records,
            &object_records,
        )?;
        let predicate_mappings = predicate_records
            .into_iter()
            .map(|record| {
                if record.subject_id != 0
                    || record.predicate_id == 0
                    || record.object_id != 0
                    || !record.triple.subject.is_empty()
                    || record.triple.object != CanonicalObject::Node(Vec::new())
                {
                    return Err(invalid("malformed persisted predicate catalog"));
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
                    return Err(invalid("malformed persisted node/value catalog"));
                }
                Ok((record.triple.object, record.object_id))
            })
            .collect::<io::Result<Vec<_>>>()?;
        let [_, predicate_catalog, node_value_catalog] =
            build_dictionary_catalogs_from_maps(node_value_mappings, predicate_mappings)?;

        let state = CommitmentState {
            parent_state_root,
            addition_count,
            removal_count,
            materialized_count,
            parent_node_value_count,
            parent_predicate_count,
            node_value_count,
            predicate_count,
            commitment_root: stored_root,
        };
        let layer = Self {
            state,
            additions,
            removals,
            materialized,
            predicate_catalog,
            node_value_catalog,
            column_commitments,
            materialized_column_commitments,
            predicate_dict_commitments,
            node_value_dict_commitments,
            node_value_dict_range_commitments,
            nv_merkle_root,
            pred_merkle_root,
        };
        layer.validate()?;
        Ok(layer)
    }

    /// Recompute the `commitment_root` binding purely from the state counts,
    /// parent root, the nine triple-column commitments, and the six dictionary
    /// commitments (independent of the raw tables). A query verifier trusts a
    /// layer's `commitment_root`, and calls this to confirm the commitments it
    /// is about to open are exactly the ones that root authenticates.
    pub fn recomputed_commitment_root(&self) -> io::Result<ProofHash> {
        self.compute_commitment_root()
    }

    /// Reconstruct the root this same native delta would have had in PF4 v3.
    /// This is migration-only: it lets an interrupted v3→v4 chain upgrade
    /// authenticate the next legacy child's parent pointer without retaining
    /// legacy metadata in the v4 format.
    pub(crate) fn legacy_v3_root(
        &self,
        legacy_parent_root: Option<ProofHash>,
    ) -> io::Result<ProofHash> {
        let predicate_catalog: Vec<_> = self
            .predicate_catalog
            .iter()
            .filter(|record| record.predicate_id > self.state.parent_predicate_count)
            .cloned()
            .collect();
        let node_value_catalog: Vec<_> = self
            .node_value_catalog
            .iter()
            .filter(|record| record.object_id > self.state.parent_node_value_count)
            .cloned()
            .collect();
        legacy_v3_root_from_parts(
            legacy_parent_root,
            self.state.addition_count,
            self.state.removal_count,
            self.state.parent_node_value_count,
            self.state.parent_predicate_count,
            self.state.node_value_count,
            self.state.predicate_count,
            &self.column_commitments,
            &commit_predicate_dictionary(&predicate_catalog)?,
            &commit_node_value_dictionary(&node_value_catalog)?,
            &commit_node_value_dict_range(&node_value_catalog)?,
        )
    }

    #[cfg(test)]
    pub(crate) fn encode_legacy_v3_for_test(
        &self,
        legacy_parent_root: Option<ProofHash>,
    ) -> io::Result<Vec<u8>> {
        let predicate_catalog: Vec<_> = self
            .predicate_catalog
            .iter()
            .filter(|record| record.predicate_id > self.state.parent_predicate_count)
            .cloned()
            .collect();
        let node_value_catalog: Vec<_> = self
            .node_value_catalog
            .iter()
            .filter(|record| record.object_id > self.state.parent_node_value_count)
            .cloned()
            .collect();
        let predicate_commitments = commit_predicate_dictionary(&predicate_catalog)?;
        let object_commitments = commit_node_value_dictionary(&node_value_catalog)?;
        let range_commitments = commit_node_value_dict_range(&node_value_catalog)?;
        let root = self.legacy_v3_root(legacy_parent_root)?;
        let mut out = Vec::new();
        out.extend_from_slice(COMMITMENT_MAGIC);
        put_u32(&mut out, 3);
        out.push(legacy_parent_root.is_some() as u8);
        if let Some(parent) = legacy_parent_root {
            put_hash(&mut out, &parent);
        }
        for count in [
            self.state.addition_count,
            self.state.removal_count,
            self.state.parent_node_value_count,
            self.state.parent_predicate_count,
            self.state.node_value_count,
            self.state.predicate_count,
        ] {
            put_u64(&mut out, count);
        }
        for table in [
            &self.additions,
            &self.removals,
            &predicate_catalog,
            &node_value_catalog,
        ] {
            put_records(&mut out, table);
        }
        for commitment in self.column_commitments.iter().chain([
            &predicate_commitments.0,
            &predicate_commitments.1,
            &object_commitments.0,
            &object_commitments.1,
            &range_commitments.0,
            &range_commitments.1,
        ]) {
            put_bytes(&mut out, &serialize_commitment(commitment)?);
        }
        put_hash(&mut out, &root);
        let checksum = file_checksum(&out);
        out.extend_from_slice(&checksum);
        Ok(out)
    }

    /// Rebuild the padded predicate-dictionary columns `(hash, id)` that
    /// `predicate_dict_commitments` commits, so a Q3 membership proof can open
    /// them. Same padding/order as the committed columns. Merkle-dict I2b S2:
    /// delegates to the single gated [`VerifierCommitmentRef`] belt accessor (a
    /// prover sidecar is always term-bearing, so the gate is a no-op here).
    pub fn predicate_dictionary_columns(&self) -> io::Result<(Vec<ArkFr>, Vec<ArkFr>)> {
        self.verifier_view().predicate_dictionary_columns()
    }

    /// Rebuild the padded node/value-dictionary columns `(hash, id)` that
    /// `node_value_dict_commitments` commits, so a Q3 membership proof can open
    /// them for a subject/object constant. Same padding/order as the committed
    /// columns. Merkle-dict I2b S2: delegates to the gated belt accessor.
    pub fn node_value_dictionary_columns(&self) -> io::Result<(Vec<ArkFr>, Vec<ArkFr>)> {
        self.verifier_view().node_value_dictionary_columns()
    }

    /// Rebuild the padded node/value-dictionary VALUE column that
    /// `node_value_dict_range_commitments.0` commits, so a range proof can open
    /// it. Same padding/order/length as `node_value_dictionary_columns`.
    /// Merkle-dict I2b S2: delegates to the gated belt accessor.
    pub fn node_value_dictionary_value_column(&self) -> io::Result<Vec<ArkFr>> {
        self.verifier_view().node_value_dictionary_value_column()
    }

    /// Rebuild the padded node/value-dictionary DATATYPE column that
    /// `node_value_dict_range_commitments.1` commits, so the range gadget can
    /// open it and mask rows to the queried datatype. Same padding/order/length
    /// as `node_value_dictionary_columns`. Merkle-dict I2b S2: delegates to the
    /// gated belt accessor.
    pub fn node_value_dictionary_datatype_column(&self) -> io::Result<Vec<ArkFr>> {
        self.verifier_view().node_value_dictionary_datatype_column()
    }

    /// The six dictionary commitments, in fixed bound order: predicate hash,
    /// predicate id, node/value hash, node/value id, then the node/value
    /// order-preserving VALUE and DATATYPE columns. `commitment_root` binds
    /// them in this order.
    fn dictionary_commitments(&self) -> [&ArkGT; 6] {
        [
            &self.predicate_dict_commitments.0,
            &self.predicate_dict_commitments.1,
            &self.node_value_dict_commitments.0,
            &self.node_value_dict_commitments.1,
            &self.node_value_dict_range_commitments.0,
            &self.node_value_dict_range_commitments.1,
        ]
    }

    /// The commitment root binds the parent root, the seven counts, the six
    /// native-delta and three materialized-view column commitments, and the
    /// six dictionary commitments (the
    /// predicate hash/id pair, the node/value hash/id pair, then the node/value
    /// order-preserving VALUE + DATATYPE pair) — NOT the raw tables, which are
    /// separately bound by the file checksum and revalidated on decode. The
    /// column commitments in turn bind the sorted id columns; the dictionary
    /// commitments bind the IRI->id and constant->id maps and the value/datatype
    /// encodings.
    fn compute_commitment_root(&self) -> io::Result<ProofHash> {
        compute_commitment_root_from_parts(
            &self.state,
            &self.column_commitments,
            &self.materialized_column_commitments,
            &self.predicate_dict_commitments,
            &self.node_value_dict_commitments,
            &self.node_value_dict_range_commitments,
        )
    }

    fn encode_body(&self) -> io::Result<Vec<u8>> {
        let mut out = Vec::new();
        out.extend_from_slice(COMMITMENT_MAGIC);
        put_u32(&mut out, COMMITMENT_FORMAT_VERSION);
        out.push(self.state.parent_state_root.is_some() as u8);
        if let Some(root) = self.state.parent_state_root {
            put_hash(&mut out, &root);
        }
        put_u64(&mut out, self.state.addition_count);
        put_u64(&mut out, self.state.removal_count);
        put_u64(&mut out, self.state.materialized_count);
        put_u64(&mut out, self.state.parent_node_value_count);
        put_u64(&mut out, self.state.parent_predicate_count);
        put_u64(&mut out, self.state.node_value_count);
        put_u64(&mut out, self.state.predicate_count);
        for table in [
            &self.additions,
            &self.removals,
            &self.materialized,
            &self.predicate_catalog,
            &self.node_value_catalog,
        ] {
            put_records(&mut out, table);
        }
        for commitment in self
            .column_commitments
            .iter()
            .chain(self.materialized_column_commitments.iter())
            .chain(self.dictionary_commitments())
        {
            let bytes = serialize_commitment(commitment)?;
            put_bytes(&mut out, &bytes);
        }
        // I1: the two id-indexed dictionary Merkle roots, at the tail of the body
        // (just before the trailing `commitment_root` + checksum the caller appends).
        put_hash(&mut out, &self.nv_merkle_root);
        put_hash(&mut out, &self.pred_merkle_root);
        Ok(out)
    }

    pub(super) fn encoded_len(&self) -> io::Result<usize> {
        let state_bytes = 1usize
            .checked_add(if self.state.parent_state_root.is_some() {
                32
            } else {
                0
            })
            .and_then(|value| value.checked_add(7 * 8))
            .ok_or_else(|| invalid("commitment manifest encoded length overflow"))?;
        let mut total = COMMITMENT_MAGIC
            .len()
            .checked_add(4)
            .and_then(|value| value.checked_add(state_bytes))
            .ok_or_else(|| invalid("commitment encoded length overflow"))?;
        for table in [
            &self.additions,
            &self.removals,
            &self.materialized,
            &self.predicate_catalog,
            &self.node_value_catalog,
        ] {
            total = checked_proof_encoded_growth(total, 8)?;
            for record in table.iter() {
                let record_len = 24usize
                    .checked_add(record.triple.encoded_len()?)
                    .ok_or_else(|| invalid("commitment encoded length overflow"))?;
                total = checked_proof_encoded_growth(total, record_len)?;
            }
        }
        for commitment in self
            .column_commitments
            .iter()
            .chain(self.materialized_column_commitments.iter())
            .chain(self.dictionary_commitments())
        {
            let commitment_len = 8usize
                .checked_add(commitment.compressed_size())
                .ok_or_else(|| invalid("commitment encoded length overflow"))?;
            total = checked_proof_encoded_growth(total, commitment_len)?;
        }
        // I1: two 32-byte dictionary Merkle roots in the body.
        total = checked_proof_encoded_growth(total, 64)?;
        // commitment_root + file checksum
        total = checked_proof_encoded_growth(total, 32)?;
        checked_proof_encoded_growth(total, 32)
    }
}

impl VerifierCommitment {
    /// β re-anchor: the artifact's two-hash `resolution_root` anchor (binds `commitment_root`
    /// AND both id-indexed dictionary Merkle roots). Inherent convenience mirroring
    /// [`LayerCommitment::resolution_root`], so remote verifiers reach it without importing the
    /// [`VerifierLayer`] trait.
    pub fn resolution_root(&self) -> ProofHash {
        self.verifier_view().resolution_root()
    }

    /// Validate the complete clear catalogs against their Dory commitments,
    /// then recompute the PF4 v4 root from all retained public commitments.
    pub fn validate(&self) -> io::Result<()> {
        self.state.validate()?;
        if self.state.addition_count > MAX_RECORDS as u64
            || self.state.removal_count > MAX_RECORDS as u64
            || self.state.materialized_count > MAX_RECORDS as u64
        {
            return Err(invalid("verifier commitment count exceeds PF4 limit"));
        }
        if self.terms_present {
            let subject_catalog = subject_catalog_from_objects(&self.node_value_catalog)?;
            validate_complete_catalog(
                &[],
                &[],
                &[],
                [
                    &subject_catalog,
                    &self.predicate_catalog,
                    &self.node_value_catalog,
                ],
                self.state.parent_node_value_count,
                self.state.parent_predicate_count,
                self.state.node_value_count,
                self.state.predicate_count,
            )?;
            if commit_predicate_dictionary(&self.predicate_catalog)?
                != self.predicate_dict_commitments
            {
                return Err(invalid("verifier predicate catalog commitment mismatch"));
            }
            if commit_node_value_dictionary(&self.node_value_catalog)?
                != self.node_value_dict_commitments
            {
                return Err(invalid("verifier node/value catalog commitment mismatch"));
            }
            if commit_node_value_dict_range(&self.node_value_catalog)?
                != self.node_value_dict_range_commitments
            {
                return Err(invalid("verifier range catalog commitment mismatch"));
            }
            // I1: recompute the id-indexed dictionary Merkle roots from the
            // now-authenticated catalogs and confirm the carried roots match.
            if node_value_merkle_root(
                &self.node_value_catalog,
                self.state.parent_node_value_count,
                self.state.node_value_count,
            )? != self.nv_merkle_root
                || predicate_merkle_root(
                    &self.predicate_catalog,
                    self.state.parent_predicate_count,
                    self.state.predicate_count,
                )? != self.pred_merkle_root
            {
                return Err(invalid("verifier dictionary merkle root mismatch"));
            }
        } else {
            // TERM-LESS (Merkle-dict I2b S1): catalogs dropped ⇒ SKIP the catalog
            // re-MSM (the 3 `commit_*` checks above — the only verifier check tying the
            // committed Dory dict columns to real terms) and the catalog-based Merkle
            // recompute. The two Merkle roots are instead bound by `resolution_root`
            // (the external anchor). Fail closed on the two structural invariants: no
            // catalog may be smuggled in, and a term-less artifact is BASE/ROLLUP only
            // (a delta needs cross-layer re-mint over faithful hashes = blocked on I4).
            // Every catalog-free check is KEPT — the count-limit above and the
            // `recomputed_commitment_root()` below (recomputed from the STORED
            // commitments + counts, NOT the catalog).
            if !self.predicate_catalog.is_empty() || !self.node_value_catalog.is_empty() {
                return Err(invalid("term-less verifier commitment must not carry catalogs"));
            }
            if self.state.parent_state_root.is_some() {
                return Err(invalid(
                    "term-less verifier commitment must be a base/rollup layer",
                ));
            }
        }
        if self.recomputed_commitment_root()? != self.state.commitment_root {
            return Err(invalid("verifier commitment root mismatch"));
        }
        Ok(())
    }

    pub fn recomputed_commitment_root(&self) -> io::Result<ProofHash> {
        compute_commitment_root_from_parts(
            &self.state,
            &self.column_commitments,
            &self.materialized_column_commitments,
            &self.predicate_dict_commitments,
            &self.node_value_dict_commitments,
            &self.node_value_dict_range_commitments,
        )
    }

    pub fn encode(&self) -> io::Result<Vec<u8>> {
        self.validate()?;
        let mut out = Vec::new();
        out.extend_from_slice(VERIFIER_COMMITMENT_MAGIC);
        put_u32(&mut out, VERIFIER_COMMITMENT_FORMAT_VERSION);
        // I2b S1: the term-bearing/term-less framing flag (byte just after the
        // version). Term-bearing (`1`) writes the two catalog blocks below;
        // term-less (`0`) skips them (the O(dict) term bytes are dropped).
        out.push(self.terms_present as u8);
        out.push(self.state.parent_state_root.is_some() as u8);
        if let Some(root) = self.state.parent_state_root {
            put_hash(&mut out, &root);
        }
        for count in [
            self.state.addition_count,
            self.state.removal_count,
            self.state.materialized_count,
            self.state.parent_node_value_count,
            self.state.parent_predicate_count,
            self.state.node_value_count,
            self.state.predicate_count,
        ] {
            put_u64(&mut out, count);
        }
        if self.terms_present {
            put_records(&mut out, &self.predicate_catalog);
            put_records(&mut out, &self.node_value_catalog);
        }
        for commitment in self.column_commitments.iter().chain(
            self.materialized_column_commitments.iter(),
        ).chain([
            &self.predicate_dict_commitments.0,
            &self.predicate_dict_commitments.1,
            &self.node_value_dict_commitments.0,
            &self.node_value_dict_commitments.1,
            &self.node_value_dict_range_commitments.0,
            &self.node_value_dict_range_commitments.1,
        ]) {
            put_bytes(&mut out, &serialize_commitment(commitment)?);
        }
        // I1: two id-indexed dictionary Merkle roots, before the trailing root.
        put_hash(&mut out, &self.nv_merkle_root);
        put_hash(&mut out, &self.pred_merkle_root);
        put_hash(&mut out, &self.state.commitment_root);
        let checksum = file_checksum(&out);
        out.extend_from_slice(&checksum);
        if out.len() > MAX_PROOF_LAYER_BYTES {
            return Err(invalid("encoded verifier commitment exceeds file budget"));
        }
        Ok(out)
    }

    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        // + 1 for the I2b S1 `terms_present` flag byte after the version.
        if bytes.len() < VERIFIER_COMMITMENT_MAGIC.len() + 4 + 1 + 64
            || bytes.len() > MAX_PROOF_LAYER_BYTES
        {
            return Err(invalid("truncated verifier commitment"));
        }
        let (payload, checksum) = bytes.split_at(bytes.len() - 32);
        if file_checksum(payload).as_slice() != checksum {
            return Err(invalid("verifier commitment checksum mismatch"));
        }
        let (body, root_bytes) = payload.split_at(payload.len() - 32);
        let commitment_root: ProofHash = root_bytes.try_into().unwrap();
        let mut r = Reader::new(body);
        if r.take(VERIFIER_COMMITMENT_MAGIC.len())? != VERIFIER_COMMITMENT_MAGIC {
            return Err(invalid("bad verifier commitment magic"));
        }
        if r.u32()? != VERIFIER_COMMITMENT_FORMAT_VERSION {
            return Err(invalid("unsupported verifier commitment version"));
        }
        // I2b S1: the term-bearing/term-less framing flag.
        let terms_present = match r.u8()? {
            0 => false,
            1 => true,
            _ => return Err(invalid("bad verifier terms-present flag")),
        };
        let parent_state_root = match r.u8()? {
            0 => None,
            1 => Some(r.hash()?),
            _ => return Err(invalid("bad verifier parent root tag")),
        };
        let state = CommitmentState {
            parent_state_root,
            addition_count: r.u64()?,
            removal_count: r.u64()?,
            materialized_count: r.u64()?,
            parent_node_value_count: r.u64()?,
            parent_predicate_count: r.u64()?,
            node_value_count: r.u64()?,
            predicate_count: r.u64()?,
            commitment_root,
        };
        // Term-bearing carries the two catalog blocks; term-less omits them (the
        // decoded catalogs stay empty and `validate()` skips the catalog checks).
        let (predicate_catalog, node_value_catalog) = if terms_present {
            let predicate_catalog =
                read_records(&mut r, PersistedTableKind::PredicateCatalog)?;
            let node_value_catalog = read_records(&mut r, PersistedTableKind::ObjectCatalog)?;
            validate_decoded_heap_budget(
                &[],
                &[],
                &[],
                &predicate_catalog,
                &node_value_catalog,
            )?;
            (predicate_catalog, node_value_catalog)
        } else {
            (Vec::new(), Vec::new())
        };
        let mut column_commitments = [<ArkGT as Group>::identity(); COLUMN_COUNT];
        for value in &mut column_commitments {
            *value = deserialize_commitment(&r.bytes()?)?;
        }
        let mut materialized_column_commitments =
            [<ArkGT as Group>::identity(); MATERIALIZED_COLUMN_COUNT];
        for value in &mut materialized_column_commitments {
            *value = deserialize_commitment(&r.bytes()?)?;
        }
        let mut next_pair = || -> io::Result<(ArkGT, ArkGT)> {
            Ok((
                deserialize_commitment(&r.bytes()?)?,
                deserialize_commitment(&r.bytes()?)?,
            ))
        };
        let predicate_dict_commitments = next_pair()?;
        let node_value_dict_commitments = next_pair()?;
        let node_value_dict_range_commitments = next_pair()?;
        // I1: the two id-indexed dictionary Merkle roots (recomputed + checked by
        // `validate` against the decoded catalogs below).
        let nv_merkle_root = r.hash()?;
        let pred_merkle_root = r.hash()?;
        if !r.done() {
            return Err(invalid("trailing verifier commitment bytes"));
        }
        let artifact = Self {
            state,
            terms_present,
            predicate_catalog,
            node_value_catalog,
            column_commitments,
            materialized_column_commitments,
            predicate_dict_commitments,
            node_value_dict_commitments,
            node_value_dict_range_commitments,
            nv_merkle_root,
            pred_merkle_root,
        };
        artifact.validate()?;
        Ok(artifact)
    }
}

