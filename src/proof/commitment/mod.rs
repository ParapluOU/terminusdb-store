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
//!
//! This module is the [layer-commitment model](crate::proof::primer) (primer §9)
//! made concrete. For what a [commitment](crate::proof::primer) and the
//! [`commitment_root`/`resolution_root`](crate::proof::primer) anchors are, read
//! the developer primer first.

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
// Bumped 5 → 6 when the sidecar gained the two id-indexed dictionary Merkle roots
// (`nv_merkle_root`/`pred_merkle_root`) for succinct id->term resolution (increment I1). The
// roots are ADDITIVE: they are a deterministic function of the already-committed catalog and
// are NOT folded into `commitment_root` (the root DOMAIN and STRUCTURE are unchanged, so a
// layer's `commitment_root` stays BYTE-IDENTICAL to v5). The bump exists only so a pre-I1 v5
// file — which lacks the two roots in its body — cannot be read under the new layout.
/// On-disk format tag for a full prover-side [`LayerCommitment`] sidecar. Decoding
/// rejects any other value, so an older-layout file cannot be misread. See the
/// version-history comment above for what each bump changed and why bumps do NOT
/// perturb `commitment_root`.
pub const COMMITMENT_FORMAT_VERSION: u32 = 6;
const VERIFIER_COMMITMENT_MAGIC: &[u8; 8] = b"TSPCV0\0\0";
// Bumped in lockstep (1 → 2): a verifier commitment derived from a v4 prover file
// likewise committed dates at the `0` sentinel and must be rejected. Its recomputed
// root uses the same unchanged `commitment-root/v4` domain, so a date-free verifier
// commitment still recomputes a byte-identical root.
// Bumped in lockstep (2 → 3): the compact artifact likewise now carries the two dictionary
// Merkle roots so a remote verifier can compute `resolution_root` and open ids positionally.
// Bumped 3 → 4 (Merkle-dict I2b S1): the compact artifact gained a `terms_present` flag byte
// (right after this version) and made the two clear catalogs OPTIONAL. Term-bearing (flag=1)
// is byte-for-byte the v3 body with the flag prepended; term-less (flag=0) DROPS the two
// catalog blocks entirely (O(1) in dictionary size). The `commitment_root` DOMAIN/STRUCTURE
// is unchanged and the flag is NOT folded into it, so a layer's `commitment_root` stays
// BYTE-IDENTICAL across term-bearing and term-less; the bump exists so a pre-S1 v3 file
// (which lacks the flag byte) cannot be read under the new layout.
/// On-disk format tag for the compact verifier-only [`VerifierCommitment`]
/// artifact. Bumped in lockstep with the prover format; decoding rejects any
/// other value. See the version-history comment above for each bump.
pub const VERIFIER_COMMITMENT_FORMAT_VERSION: u32 = 4;

/// Number of committed columns per layer: additions (S,P,O) then removals
/// (S,P,O), in that fixed order.
pub const COLUMN_COUNT: usize = 6;
/// Number of committed columns for a base/rollup layer's materialized head view:
/// the complete subject/predicate/object id columns (S,P,O).
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
mod dict_merkle;
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
pub use dict_merkle::resolution_root;
pub use op_value::{node_value_op_value, NODE_DATATYPE_TAG};

// Crate-internal surface (sibling proof modules + storage plumbing, and the
// framing/budget code in `proof.rs` itself). `construction_commitment_scratch_bytes`
// was `pub(super)` (proof-visible) as an inline item; one level deeper it reaches
// proof as `pub(crate)` — internal only, never part of the external `pub` surface.
pub(crate) use columns::reject_nan_catalog;
pub(crate) use columns::reject_nan_objects;
pub(crate) use dict_merkle::{
    build_node_value_tree, build_predicate_tree, verify_opening, DictMerkleTree,
};
// Merkle-dict I2b S5: the term-less opening witness (`DictOpening`/`ResolutionOpenings`) is a
// PUBLIC (uncommitted) proof-wire type — a term-less `LayerBgpProof` carries it. Re-exported `pub`
// so the proof module can surface it and woql2 can name it.
pub use dict_merkle::{DictOpening, ResolutionOpenings};
// Merkle-dict I2b S5: the prover-side term-less-opening BUILDER is now PRODUCTION (the shipping
// term-less producer mints openings; the anti-forge coupling is S4's mandatory belt-hash, not a
// compile-time gate).
pub(crate) use dict_merkle::build_resolution_openings;
// The single-layer term-less envelope's carried witness: one-carrier selection + the layer wrapper.
pub(crate) use state::{effective_openings, TermlessWithOpenings};
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
use dict_merkle::*;
use legacy::*;
use op_value::*;
use root::*;
use setup::*;
use state::*;
