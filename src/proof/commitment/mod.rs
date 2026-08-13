//! Per-layer commitment machinery for the Proof-of-WOQL scheme — a layer's TRUST ANCHOR.
//!
//! # What a layer commits, in plain terms
//!
//! A store layer is an immutable set of triples plus the dictionaries that map opaque ids to
//! their canonical bytes. This module turns that content into a small, tamper-evident package
//! so that anyone can later prove facts about the layer to a verifier who trusts only ONE
//! 32-byte value.
//!
//! The package (a [`LayerCommitment`]) has three parts:
//!  * a tiny authenticated summary ([`CommitmentState`]): the parent layer's root, the triple
//!    and dictionary counts, and the layer's own 32-byte `commitment_root`;
//!  * the clear column tables (added / removed / materialized triples as dictionary-id
//!    columns, and the predicate + node/value dictionaries as hash/id columns);
//!  * a Dory *column commitment* over each of those columns — a short value that pins down the
//!    whole column, so no single entry can change without changing the commitment.
//!
//! # How the root binds it
//!
//! The `root` submodule hashes the summary and every column commitment together, in a fixed
//! order, into the single `commitment_root`. Because the root covers all of them, trusting
//! that one root is equivalent to trusting the entire committed layer: a proof anchored to the
//! root cannot lie about any committed triple, dictionary binding, or count. A remote verifier
//! needs none of the prover's witness tables — the trimmed [`VerifierCommitment`] carries just
//! the public commitments and clear dictionaries needed to recompute the root and check
//! openings against it.
//!
//! # Submodule map
//!
//! This file (the module facade) only declares the submodules and re-exports the public
//! surface. The work is split by concern:
//!  * `setup` — the deterministic, trapdoor-free Dory public parameters.
//!  * `columns` — building each committed column (plus the NaN guard).
//!  * `op_value` — the order-preserving encoding of numeric/date VALUES for range proofs.
//!  * `state` — the committed-layer types, their serialization, build & validate.
//!  * `root` — computing the `commitment_root` (and the database root) that binds it all.
//!  * `catalog` — dictionary catalog construction and integrity validation.
//!  * `legacy` — a migration-only decoder for the previous on-disk format.
//!
//! Public paths are unchanged: everything re-exported below is still reachable at
//! `terminus_store::proof::commitment::*`, exactly as when this was one inline module.

use super::*;
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::convert::TryInto;
use std::io;
use std::sync::OnceLock;

use ark_bn254::{G1Projective, G2Projective};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use dory_pcs::backends::arkworks::{ArkFr, ArkG1, ArkG2, ArkGT, BN254};
#[cfg(test)]
use dory_pcs::backends::arkworks::{ArkworksPolynomial, G1Routines};
use dory_pcs::primitives::arithmetic::{Field as DoryField, Group, PairingCurve};
#[cfg(test)]
use dory_pcs::primitives::poly::Polynomial;
use dory_pcs::setup::{ProverSetup, VerifierSetup};
#[cfg(test)]
use dory_pcs::Transparent;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

const COMMITMENT_MAGIC: &[u8; 8] = b"TSPCW0\0\0";
// Bumped 4 → 5 when `xsd:date`/`xsd:dateTime` dict entries gained a NON-ZERO
// order-preserving VALUE (a decoded i64-nanoseconds key; see `node_value_op_value`).
// A date-containing layer's `commitment_root` therefore differs from a v4 file's, and
// a stale v4 commitment (which committed dates at the `0` sentinel) must NOT be read as
// if it used the new semantics — this file-format tag rejects it at decode.
//
// The root DOMAIN string (`commitment-root/v4`, `compute_commitment_root_from_parts`)
// is deliberately NOT bumped: the root's STRUCTURE (which columns/commitments are
// hashed, in what order) is unchanged — only the CONTENT of the node/value VALUE column
// changed, and only for entries that are dates. So a layer WITHOUT dates commits the
// identical `0`-sentinel value column and hashes to a BYTE-IDENTICAL root under v5, and
// old v4 anchors for date-free layers stay valid.
pub const COMMITMENT_FORMAT_VERSION: u32 = 5;
const VERIFIER_COMMITMENT_MAGIC: &[u8; 8] = b"TSPCV0\0\0";
// Bumped in lockstep (1 → 2): a verifier commitment derived from a v4 prover file
// likewise committed dates at the `0` sentinel and must be rejected. Its recomputed
// root uses the same unchanged `commitment-root/v4` domain, so a date-free verifier
// commitment still recomputes a byte-identical root.
pub const VERIFIER_COMMITMENT_FORMAT_VERSION: u32 = 2;

/// Number of committed columns per layer: additions (S,P,O) then removals
/// (S,P,O), in that fixed order.
pub const COLUMN_COUNT: usize = 6;
pub const MATERIALIZED_COLUMN_COUNT: usize = 3;

/// Largest deterministic Dory domain needed by any structurally admissible
/// PF4 table. `MAX_RECORDS = 10,000,000` pads to `2^24`; no native layer can
/// reach a larger proof column without first violating the PF4 record cap.
pub const SETUP_MAX_LOG_N: usize = 24;
const LEGACY_SETUP_LOG_N: usize = 20;
const SETUP_TIER_LOG_N: [usize; 3] = [LEGACY_SETUP_LOG_N, 22, SETUP_MAX_LOG_N];
const MAX_SETUP_RETAINED_BYTES: usize = 8 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Submodules — see each file's module doc for what it commits/builds/validates.
// ---------------------------------------------------------------------------
mod catalog;
mod columns;
mod legacy;
mod op_value;
mod root;
mod setup;
mod state;
#[cfg(test)]
mod tests;

// ---------------------------------------------------------------------------
// Public surface. This is the EXACT set of items the single-file `commitment`
// module exposed; only their definition site moved into the submodules above.
// ---------------------------------------------------------------------------
pub use state::{
    CommitmentState, LayerCommitment, VerifierCommitment, VerifierCommitmentRef, VerifierLayer,
};
pub use root::{database_root, validate_database_root};
pub use op_value::{node_value_op_value, NODE_DATATYPE_TAG};

// Crate-internal surface (sibling proof modules + storage plumbing, and the
// framing/budget code in `proof.rs` itself). `construction_commitment_scratch_bytes`
// was `pub(super)` (proof-visible) as an inline item; one level deeper it reaches
// proof as `pub(crate)` — internal only, never part of the external `pub` surface.
pub(crate) use columns::reject_nan_catalog;
pub(crate) use legacy::decode_legacy_v3;
pub(crate) use setup::{
    construction_commitment_scratch_bytes, matrix_dims, prover_setup, setup_parameter_id,
    verifier_setup,
};

// ---------------------------------------------------------------------------
// Internal flat-namespace hoist: re-expose every submodule's cross-referenced
// helpers under this module so each submodule's `use super::*` sees the same
// flat helper set the single inline module had. These were all parent-private
// helpers widened only to `pub(super)`; this does NOT widen the external
// surface (the `pub`/`pub(crate)` items are re-exported explicitly above).
// ---------------------------------------------------------------------------
use catalog::*;
use columns::*;
use legacy::*;
use op_value::*;
use root::*;
use setup::*;
use state::*;
