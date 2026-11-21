//! What a layer commitment is when this build cannot make one.
//!
//! The storage plumbing names [`LayerCommitment`] in signatures it keeps
//! whether or not the `proof` feature is on: `proof_layer`,
//! `write_proof_layer`, `rebuild_proof_layer`, the proof cache, the pack
//! importer's admission step. Rather than `#[cfg]` every one of those (and
//! their callers, and their callers' callers), this module supplies the type
//! with no commitments in it and no way to build one.
//!
//! Every constructor fails with [`io::ErrorKind::Unsupported`] and a sentence
//! naming the feature. That is the only honest answer: a commitment binds Dory
//! commitments over the layer's dictionary-encoded columns, so a build without
//! the Dory stack cannot produce one, cannot read one, and must not pretend a
//! layer is proven. Reaching any of these is a configuration error, never data
//! loss — the sidecar is a separate file, so a layer written here is exactly a
//! layer an unforked terminus-store would have written.

use std::io;

use super::CommitmentState;

fn unsupported<T>(what: &str) -> io::Result<T> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        format!(
            "{what} requires the terminus-store `proof` feature, which this build does not have"
        ),
    ))
}

/// The per-layer sidecar payload, as far as a build without the `proof`
/// feature is concerned: a shape with no inhabitants reachable through its own
/// API.
///
/// The field is here because the storage plumbing reads `proof.state` when it
/// has a commitment in hand; it never gets one.
#[derive(Clone, Debug)]
pub struct LayerCommitment {
    pub state: CommitmentState,
}

impl LayerCommitment {
    /// Build proof state from a layer's native delta. Always fails here.
    #[allow(clippy::too_many_arguments)]
    pub fn from_native_delta(
        _parent_state_root: Option<super::ProofHash>,
        _additions: Vec<super::ProofRecord>,
        _removals: Vec<super::ProofRecord>,
        _materialized: Vec<super::ProofRecord>,
        _parent_node_value_count: u64,
        _parent_predicate_count: u64,
        _node_value_count: u64,
        _predicate_count: u64,
        _node_value_mappings: Vec<(super::CanonicalObject, u64)>,
        _predicate_mappings: Vec<(Vec<u8>, u64)>,
    ) -> io::Result<Self> {
        unsupported("computing a layer commitment")
    }

    /// Parse a stored sidecar. Always fails here, which is what keeps a
    /// proof-off build from reporting a layer as proven on the strength of a
    /// file it cannot check.
    pub fn decode(_bytes: &[u8]) -> io::Result<Self> {
        unsupported("reading a layer commitment")
    }

    /// Serialize this sidecar. Unreachable: nothing constructs the value.
    pub fn encode(&self) -> io::Result<Vec<u8>> {
        unsupported("writing a layer commitment")
    }

    /// Conservative retained-heap estimate, for the proof cache's weighting.
    pub fn estimated_cache_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

/// Read a v3 (Merkle-era) sidecar during migration. Always fails here.
pub(crate) fn decode_legacy_v3(_bytes: &[u8]) -> io::Result<CommitmentState> {
    unsupported("migrating a legacy layer commitment")
}

impl CommitmentState {
    /// Check a migrated v3 state against the v4 sidecar rebuilt beside it.
    /// Unreachable: neither one can be produced here.
    pub(crate) fn validate_against_v4(&self, _current: &LayerCommitment) -> io::Result<()> {
        unsupported("validating a migrated layer commitment")
    }
}

impl LayerCommitment {
    /// Recompute the v3 (Merkle-era) root this layer would have had. Always
    /// fails here.
    pub(crate) fn legacy_v3_root(
        &self,
        _legacy_parent_root: Option<super::ProofHash>,
    ) -> io::Result<super::ProofHash> {
        unsupported("recomputing a legacy layer root")
    }
}

/// Peak scratch heap a commitment construction would need. Nothing is ever
/// constructed here, so nothing is reserved.
pub(crate) fn construction_commitment_scratch_bytes(_row_counts: &[usize]) -> io::Result<usize> {
    Ok(0)
}
