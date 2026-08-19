//! Delta-stack RECONCILIATION over an N-layer parent chain (Waves 1-3 core).
//!
//! Proves a read query over a **base + k deltas** layer stack by RECONCILING the
//! current triple set at proof time and authenticating it against the committed
//! per-layer deltas — WITHOUT any materialized head view (the Wave-0 commitment
//! diet removed it). The read delivered here is the reconciled triple COUNT (the
//! exact single-atom count answer over `present`); the full BGP/filter/range/join
//! over `present` is the next pass.
//!
//! # The math (reconciliation = a multiset equality)
//! terminus-store maintains the delta invariant (add only absent triples, remove
//! only present ones), so the current set `present` satisfies a CLEAN multiset
//! equality over canonical (dictionary-independent) terms, chain-wide:
//! ```text
//! present  ⊎  (⊎_layers removals)   ≡   (⊎_layers additions)
//! ```
//! (a removal in layer `j` cancels exactly one earlier addition). The prover supplies
//! `present`; this module proves the equality via a two-sided masked LogUp link (the
//! discipline that closed the self-join padding bug), which authenticates `present`
//! against the committed delta columns. Under the invariant `present` is duplicate-free.
//!
//! # Cross-layer dictionary authentication (Wave 1, N-layer)
//! Each layer commits only the dictionary entries IT introduced
//! (`[parent_*_count+1 ..= *_count]`); a term's id is defined in exactly the layer
//! whose incremental range contains it, and the chain's ranges abut disjointly. We
//! reconcile in canonical-term-HASH space so a term hashes identically regardless of
//! which layer minted its id:
//!   * a COMBINED node/value dictionary is the exact bag union of EVERY layer's
//!     incremental node/value dict (each bound to its own `commitment_root`), so it
//!     resolves any id from anywhere in the chain in one authenticated table;
//!   * a COMBINED predicate dictionary is the same over every layer's incremental
//!     predicate dict, hashing predicates with `hash_term`;
//!   * a triple's canonical identity is `(hash_object(S), hash_term(P), hash_object(O))`,
//!     every component root-authenticated via the audited [`super::dicthash`] lookup
//!     inner against the combined dict.
//!
//! # Root binding (Wave 2 envelope, N-layer)
//! The verifier validates each layer against ITS OWN `commitment_root`, checks the
//! whole parent chain (`layer_i.parent_state_root == layer_{i-1}.commitment_root`, the
//! base a root, the head pinned to the externally-trusted `trusted_head_root`) and the
//! dictionary-count abutment. No layer may be spliced, reordered, dropped, or foreign.
//!
//! # Scope of THIS pass (deferred — see the module report)
//!   * the query is the reconciled triple COUNT; BGP/filter/range over `present` is next;
//!   * each non-empty committed group (a layer's additions or removals) must have ≥ 2
//!     records (so its column commitment lives at a relocatable ≥ 2 domain); single-row
//!     deltas need a relocation-gadget extension to the `2^0` domain.
//!
//! # Submodule map
//!
//! This file is the module facade: it holds the chain-wide constants, the shared field
//! helpers, and the imports, and re-exports the public surface unchanged. The work is split
//! by concern:
//!  * `reconcile_core` — proving `present` IS the reconciled set (the multiset-equality link)
//!    plus chain-state validation and the reconcile proof structs.
//!  * `combined_dict` — cross-layer combined id<->term dictionary authentication.
//!  * `present_filter` / `present_join` / `present_bgp` — single-atom, two-atom-join, and
//!    general n-atom BGP reads over the authenticated `present` relation.
//!  * `present_result` — resolving authenticated result bindings back to `CanonicalObject`s.
//!  * `delta_read` — `AddedTriple` / `DeletedTriple` reads over the head layer's delta groups.
//!
//! Public paths are unchanged: every item re-exported below stays reachable at
//! `terminus_store::proof::reconcile::*`, exactly as when this was a single file.

use std::io;

use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::transcript::Transcript;

use super::argument::{
    commit_column, mle_full_eval, powers, verify_column_openings, ColumnOpening,
    StoreProverContext, Transcript254,
};
use super::anti::{
    prove_inner_join_with_keys_and_context, verify_inner_join_with_keys, InnerJoinProof,
};
use super::bgp::{
    prove_bgp_chain_with_result_and_context, verify_bgp_chain, BgpChainProof, JoinStep,
    RelationCommit, RelationWitness,
};
use super::commitment::{
    build_node_value_tree, build_predicate_tree, node_value_op_value, reject_nan_objects,
    verify_opening, DictMerkleTree, DictOpening, LayerCommitment, ResolutionOpenings,
    VerifierCommitmentRef, VerifierLayer, NODE_DATATYPE_TAG,
};
// Merkle-dict I2b S3 differential/negative tests: the prover-side opening builder is test-only
// in the reconcile module (the term-less chain-resolver path is exercised via the `_with_openings`
// test helpers over a term-less base view).
#[cfg(test)]
use super::commitment::build_resolution_openings;
use super::dict::{hash_object, hash_term};
use super::filter::{
    prove_boolean_product_with_context, prove_compaction_only_with_context,
    prove_filter_indicator_with_context, prove_selection_only_with_context, verify_boolean_product,
    verify_compaction_only, verify_filter_indicator, verify_selection_only, AtomFilterSpec,
    BooleanProductProof, CompactionOnlyProof, FilterProof, Position, SelectionOnlyProof,
};
use super::dicthash::{
    honest_hash_lookup, prove_hash_lookup_inner, verify_hash_lookup_inner, HashLookupInnerProof,
};
use super::logup_gkr::{
    bind_and_verify_fraction_tree, check_leaf_tie, eq_at_index, prove_fraction_sum_zero,
    FractionInstance, GkrFractionProof, LinearForm, LinearTerm, Src,
};
use super::permutation::{
    prove_relation_permutation_with_context, verify_relation_permutation, RelationPermutationProof,
};
use super::query::ObjectRangeSpec;
use super::sort::{
    prove_strictly_sorted_bits_with_context, verify_strictly_sorted_bits, SortedProof,
};
use super::range::{
    object_range_sel_column, prove_object_range_select_with_context, verify_object_range_select,
    ObjectRangeProof, OBJECT_RANGE_B_BITS,
};
use super::relocate::{
    prove_column_relocation_with_context, verify_column_relocation, RelocationProof,
};
use super::union::{prove_bag_union_with_context, verify_bag_union, BagUnionProof};
use super::{invalid, CanonicalObject, CanonicalTriple, ProofHash, ProofRecord};

const RECONCILE_DOMAIN: &[u8] = b"parture/proof-of-woql/delta-reconcile/v1-nlayer";

#[inline]
fn f0() -> ArkFr {
    ArkFr::zero()
}
#[inline]
fn f1() -> ArkFr {
    ArkFr::one()
}
#[inline]
fn ark(v: u64) -> ArkFr {
    ArkFr::from_u64(v)
}

fn pad_to(values: &[ArkFr], log: usize) -> Vec<ArkFr> {
    let mut out = values.to_vec();
    out.resize(1usize << log, f0());
    out
}

/// The relocatable power-of-two domain a group of `n` records is committed at
/// (`commit_record_id_column` pads to `next_pow2(n)`; this pass requires the result
/// to be a relocatable `>= 2` domain, i.e. `n >= 2`).
fn group_col_log(n: usize) -> io::Result<usize> {
    if n < 2 {
        return Err(invalid(
            "reconcile: each non-empty committed group must have >= 2 records (single-row deltas unsupported this pass)",
        ));
    }
    Ok(n.next_power_of_two().trailing_zeros() as usize)
}

/// Dictionary domain log for a catalog of `len` entries (matches
/// `node_value_dict_padded_len`/`padded_domain_for_rows`: `next_pow2(len).max(2)`).
fn dict_log(len: usize) -> io::Result<usize> {
    if len == 0 {
        return Err(invalid("reconcile: empty dictionary catalog"));
    }
    Ok(len.next_power_of_two().max(2).trailing_zeros() as usize)
}


// ---------------------------------------------------------------------------
// Submodules — see each file's module doc for what it proves/verifies.
// ---------------------------------------------------------------------------
mod combined_dict;
mod delta_read;
mod present_bgp;
mod present_filter;
mod present_join;
mod present_result;
mod reconcile_core;
#[cfg(test)]
mod tests;

// ---------------------------------------------------------------------------
// Public surface. This is the EXACT set of `pub` items the single-file
// `reconcile` module exposed; only their definition site moved above.
// ---------------------------------------------------------------------------
pub use reconcile_core::{
    prove_reconcile, prove_reconcile_with_context, verify_reconcile, CombinedDictProof,
    CommittedGroupProof, GroupHashProof, ReconcileArtifacts, ReconcileProof, ReconcileVerified,
    ReconciledPresent,
};
pub use combined_dict::CombinedRangeDictProof;
pub use present_filter::{
    prove_read_single_atom, verify_read_single_atom, FilterPosition, PresentFilterProof,
    PresentReadResult,
};
pub use present_join::{
    prove_read_two_atom_join, verify_read_two_atom_join, PresentIdAuth, PresentJoinProof,
};
pub use present_result::{
    verify_and_resolve_single_atom, verify_and_resolve_two_atom_join, ResolveKind,
};
pub use present_bgp::{
    attach_termless_present_range_openings, attach_termless_reconcile_openings,
    build_termless_present_range_openings, build_termless_reconcile_openings, prove_read_bgp,
    verify_and_resolve_bgp, verify_and_resolve_bgp_from_views, verify_read_bgp,
    verify_read_bgp_from_views, ColumnDistinctProof, PresentAtomFilter, PresentBgpAtom,
    PresentBgpAtomProof, PresentBgpProof, PresentRangeSubproof, PresentSecondRange, PresentTerm,
};
pub use delta_read::{
    attach_termless_delta_openings, build_termless_delta_openings, prove_delta_read,
    verify_and_resolve_delta_read, verify_and_resolve_delta_read_from_views, verify_delta_read,
    verify_delta_read_from_views, DeltaReadProof, DeltaSource,
};

// ---------------------------------------------------------------------------
// Internal flat-namespace hoist: re-expose every submodule's cross-referenced
// helpers under this module so each submodule's `use super::*` sees the same
// flat helper set the single file had. These were parent-private helpers
// widened only to `pub(super)`; the `pub` surface is re-exported explicitly
// above, so this does NOT widen the external surface.
//
// FOUNDATION-A (ReadDocument 6c-i): the two hoists that carry a `pub(crate)` item —
// `reconcile_core` (`prove/verify_reconcile_core`) and `present_bgp`
// (`prove/verify_present_bgp`) — are `pub(crate) use` globs, so those cores (and ONLY
// those, since a glob re-export caps each name at its own visibility — the `pub(super)`
// helpers stay reconcile-local) are reachable from the sibling `envelope` module. This is
// what lets a walk/closure composite reconcile ONCE and answer each leaf read against the
// SHARED `ReconcileArtifacts`/transcript, never re-reconciling per leaf. Their logic is
// untouched — only the visibility path widened.
// ---------------------------------------------------------------------------
// `pub(crate)` glob so the reconciled `OrderedArrayWalk` (envelope 6d) can reach the `pub(crate)`
// combined `(id, value, datatype)` RANGE-dict builders (`prove/verify_combined_range_dict`,
// `range_dict_branches`/`range_dict_branch_commits`, `CombinedRangeDictBuilt`) it authenticates each
// `sys:index` against; the `pub(super)` helpers (`prove/verify_combined_dict`, `CombinedDictBuilt`)
// stay reconcile-local (a glob caps each name at its own visibility), so the external surface is
// unchanged and the explicit `pub use combined_dict::CombinedRangeDictProof;` above still wins.
pub(crate) use combined_dict::*;
pub(crate) use present_bgp::*;
use present_filter::*;
use present_join::*;
// `pub(crate)` glob so the reconciled walk composites (envelope 6c-ii) can reach the
// `pub(crate)` chain value-resolver `resolve_node_value_id_column`; the `pub(super)` helpers
// (`resolve_result_columns`, `filter_output_kinds`, …) stay reconcile-local (a glob caps each
// name at its own visibility), so the external surface is unchanged.
pub(crate) use present_result::*;
pub(crate) use reconcile_core::*;
