//! Layer-level query proofs — the entry points that prove/verify a whole query against
//! ONE committed store layer, anchored to that layer's `commitment_root`.
//!
//! A caller asks a question in terms of IRIs and variables ("all `(?s, ?o)` with predicate
//! `<P>`", or a multi-atom conjunction with joins and value ranges). These entry points
//! return the complete answer PLUS a succinct proof a verifier can check without re-running
//! the query. Every proof is anchored to the layer's `commitment_root`, so the verifier only
//! has to trust that one 32-byte root.
//!
//! This file is only the facade: it re-exports the public API and wires the pieces together.
//! The work is split by concern across submodules, from smallest building block to full query:
//!
//! * `column_source` — which committed columns a query reads (materialized head vs the
//!   layer's own additions/removals delta), plus the shared selection/anchoring gates.
//! * `filter_query` — Q1: a single triple pattern by predicate IRI, `t(?s, <P>, ?o)`.
//! * `graph_pattern` — Q2: a 2-atom "friend-of-a-friend" self-join, `t(?x,<P>,?y),t(?y,<P>,?z)`.
//! * `plan` — the public plan/atom/proof/result TYPES, plan validation, and the
//!   Fiat–Shamir statement binding that pins a plan into the proof's challenges.
//! * `result` — turning proven Store-id result columns back into IRIs / typed literals.
//! * `range_atom` — adding a numeric value-range test (`?a < c`) to one atom.
//! * `layer_bgp` — Q_k: the general k-atom conjunctive BGP prove/verify entry points.
//!
//! Public paths are unchanged: everything re-exported below is still reachable at
//! `terminus_store::proof::query::*`, exactly as when this was a single file.

use std::io;

use ark_ff::PrimeField;
use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::transcript::Transcript;

use super::argument::{commit_column, StoreProverContext, Transcript254};
use super::bgp::{
    prove_bgp_chain_with_result_and_context, verify_bgp_chain, BgpChainProof, JoinStep,
    RelationCommit, RelationWitness,
};
use super::commitment::{
    build_node_value_tree, build_predicate_tree, node_value_op_value, reject_nan_catalog,
    reject_nan_objects, CommitmentState, DictMerkleTree, DictOpening,
    LayerCommitment, ResolutionOpenings, VerifierCommitmentRef, VerifierLayer, NODE_DATATYPE_TAG,
};
use super::dict::{
    hash_object, hash_term, prove_dict_membership_with_context, verify_dict_membership, DictProof,
};
use super::filter::{
    atom_kept_columns, compact_selected, honest_witness, prove_atom_filter_with_context,
    prove_boolean_product_with_context, prove_compaction_only_with_context,
    prove_filter_with_context, prove_selection_only_with_context, verify_atom_filter,
    verify_boolean_product, verify_compaction_only, verify_filter, verify_selection_only,
    AtomFilterProof, AtomFilterSpec, BooleanProductProof, CompactionOnlyProof, FullFilterProof,
    Position, SelectionOnlyProof,
};
use super::invalid;
use super::join::{
    compute_self_join, prove_self_join_complete_with_context, self_join_commitments_with_context,
    self_join_domain, verify_self_join_complete, SelfJoinCommitments, SelfJoinCompleteProof,
};
use super::range::{
    object_range_dtmatch_column, object_range_sel_column, prove_object_range_select_with_context,
    verify_object_range_select, ObjectRangeProof, RangeOp,
};
use super::relocate::{
    prove_column_relocation_with_context, verify_column_relocation, RelocationProof,
};
use super::ProofHash;
use super::ProofRecord;
use super::{CanonicalDatatype, CanonicalObject};

// --- Shared column helpers used by every submodule (parent-private; each submodule picks
// --- them up via its own `use super::*`). Field zero, the power-of-two padding convention
// --- committed columns use, and zero-padding a column up to a target length.

#[inline]
fn f0() -> ArkFr {
    ArkFr::zero()
}

/// The padded hypercube size for a materialized head view's columns, matching
/// `commit_id_column`'s `next_power_of_two(materialized.len())`. Requires `>= 2` (a
/// one-row layer has `k = 0`, below the sumcheck's minimum of one variable).
fn padded_len(real_rows: usize) -> io::Result<usize> {
    let padded = real_rows
        .checked_next_power_of_two()
        .ok_or_else(|| invalid("layer row count overflow"))?;
    if padded < 2 {
        return Err(invalid(
            "layer has too few rows for a query proof (need >= 2)",
        ));
    }
    Ok(padded)
}

fn pad(v: &[ArkFr], n: usize) -> Vec<ArkFr> {
    let mut out = v.to_vec();
    out.resize(n, f0());
    out
}


// ---------------------------------------------------------------------------
// Submodules (see each file's module doc for what it proves/verifies).
// ---------------------------------------------------------------------------
mod column_source;
mod filter_query;
mod graph_pattern;
mod plan;
mod result;
mod range_atom;
mod layer_bgp;
#[cfg(test)]
mod tests;

// ---------------------------------------------------------------------------
// Public API facade. This is the EXACT set of items previously exposed at
// `terminus_store::proof::query::*`; only their definition site moved.
// ---------------------------------------------------------------------------
pub use column_source::ColumnSource;
pub use filter_query::{
    prove_layer_filter, prove_layer_filter_with_context, prove_layer_filter_with_source,
    verify_layer_filter, verify_layer_filter_with_source, CombinedFilterProof,
};
pub use graph_pattern::{
    prove_layer_graph_pattern2, prove_layer_graph_pattern2_with_context,
    prove_layer_graph_pattern2_with_source, verify_layer_graph_pattern2,
    verify_layer_graph_pattern2_with_source, GraphPattern2Proof,
};
pub use plan::{
    generated_decimal_i64_field, generated_decimal_i64_value, generated_decimal_u64_value,
    AtomTerm, CanonicalResultValue, DictionaryAbsence, DictionaryAbsencePosition, LayerBgpAtom,
    LayerBgpAtomProof, LayerBgpPlan, LayerBgpProof, LayerBgpResult, ObjectKindFilter,
    ObjectRangeBound, ObjectRangeSpec, RangeAtomSubproof, RangeDatatypeGuardProof,
    ResultColumnDescriptor,
    ResultIdNamespace, SecondRangeSubproof,
};
pub use result::{
    resolve_node_value_result_objects, resolve_result_values, resolve_result_values_with_openings,
    resolve_typed_result_objects, resolve_typed_result_objects_with_openings,
    result_column_descriptors, result_id_namespaces, verify_node_value_result_objects,
    verify_result_columns,
};
// Merkle-dict I2b S5: term-less resolution over prover-shipped Merkle openings is now PRODUCTION
// (`resolve_result_values_with_openings` / `resolve_typed_result_objects_with_openings` above) —
// the term-less BGP path is gated end-to-end by S4's mandatory per-constant belt-hash in
// `verify_layer_bgp`. This single-column node/value variant stays test-only (differential probes).
#[cfg(test)]
pub(crate) use result::resolve_node_value_result_objects_with_openings;
pub use layer_bgp::{
    attach_termless_bgp_openings, build_termless_bgp_openings, build_termless_range_openings,
    prove_layer_bgp, prove_layer_bgp_with_result, prove_layer_bgp_with_result_and_context,
    verify_layer_bgp, verify_layer_bgp_result_log_domain,
};
// Merkle-dict I2b S4: the per-matched-constant belt-hash entry that consumes the prover-shipped
// constant openings for a TERM-LESS artifact. `#[cfg(test)]` + `pub(crate)` — the openings-
// consuming verify path exists only in test builds until S5 wires the shipping FFI, so a term-less
// artifact through the PUBLIC `verify_layer_bgp` still fails closed (no belt-less constant-match).
#[cfg(test)]
pub(crate) use layer_bgp::verify_layer_bgp_with_constant_openings;

// Crate-internal API (consumed by sibling proof modules such as `envelope`).
pub(crate) use column_source::require_base_layer_for_query;
pub(crate) use plan::has_ground_result_unit;
// Merkle-dict I2b step 3 (value↔term consistency): the shared audited helper + its Merkle
// resolver / id decoder, re-exposed at `pub(crate)` so the sibling `order`/`multiorder`/`group`
// gadgets (in `crate::proof`, one level up) can tie their term-less value/datatype columns to the
// Merkle-authenticated terms exactly as the range gadget does.
pub(crate) use result::{
    canonical_dictionary_id, resolve_node_value_id_via_merkle, verify_dict_value_terms,
};

// I1 differential-test baseline: the pre-Merkle catalog-scan resolvers, reachable from the
// query test module to assert the Merkle path resolves every id identically.
#[cfg(test)]
pub(crate) use result::{
    resolve_node_value_result_objects_via_catalog, resolve_result_values_via_catalog,
};
// Merkle-dict I2b S5: the prover-side opening builder is now PRODUCTION — the term-less BGP
// producer (`attach_termless_bgp_openings`) mints the shipped opening witness with it.
pub(crate) use super::commitment::build_resolution_openings;

// ---------------------------------------------------------------------------
// Internal hoist: re-expose the cross-module `pub(super)` helpers under this
// module's namespace so every submodule's `use super::*` (and the test module)
// sees the same flat helper set the single-file version had. Not part of the
// public surface (all `pub(super)`/`pub(crate)`; capped to this module tree).
// ---------------------------------------------------------------------------
#[allow(unused_imports)]
use column_source::{
    resolve_predicate_id, source_column_commitments, source_count, source_layer_columns,
};
#[allow(unused_imports)]
use filter_query::bind_statement;
#[allow(unused_imports)]
use plan::{
    atom_filter_spec, atom_term_object, bind_layer_bgp_statement, check_ids_consistent,
    dictionary_absence, is_all_ground, relational_atom_count, resolve_node_value_id, validate_atom,
    validate_bgp_plan, AtomIds,
};
#[allow(unused_imports)]
use range_atom::{dict_domain_len, prove_ranged_atom, verify_ranged_atom};
#[cfg(test)]
#[allow(unused_imports)]
use column_source::materialized_layer_columns;
