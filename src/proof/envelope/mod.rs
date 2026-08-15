//! The recursive relation-proof **envelope** — a self-describing, root-bound proof of a
//! relational query result.
//!
//! # What this is, in plain terms
//!
//! A store layer is an immutable set of triples identified by a 32-byte cryptographic *root*.
//! This module packages a *proof* that some relational query, run over the triples committed
//! under that exact root, produced exactly the delivered rows — packaged so that a verifier who
//! trusts only the root (never the prover) can re-check it offline.
//!
//! The proof is shaped like the query's OPERATOR TREE. A leaf is a single basic-graph-pattern
//! read of one layer; interior nodes are relational operators (join, distinct, group/aggregate,
//! order, anti/`NOT`, union, a cross-graph hash join, a bounded transitive closure). Each node
//! carries a complete, independently checkable proof that its output relation follows from its
//! children's output relations. Verification walks the tree bottom-up; if every node checks and
//! the leaves are bound to the trusted root, the delivered result is sound.
//!
//! # The two proof front-ends
//!
//! * [`RecursiveRelationEnvelope`] is the general recursive tree: a [`RelationProofPlan`] (the
//!   PUBLIC, variable-blind operator tree the verifier reconstructs from its own trusted query)
//!   paired with a [`RecursiveRelationProof`] (the matching per-node proof payloads).
//! * [`LayerBgpEnvelope`], [`BagUnionEnvelope`], [`LeftAntiJoinEnvelope`] and
//!   [`LeftOuterJoinEnvelope`] are the older single-shape portable envelopes for the common
//!   flat cases; each is a struct with `new_verified` / `verify` / `encode` / `decode`.
//!
//! # Self-describing wire format + anti-substitution
//!
//! Every envelope `encode`s to bytes framed by a per-shape MAGIC tag + version + a
//! length/item-budgeted body + a SHA-256 checksum, and `decode`s back under strict size limits.
//! Decoding is ONLY parsing — it performs no trust checks. The trust is in `verify`, which:
//!  1. checks the envelope's `layer_root` equals the caller's trusted root;
//!  2. checks the envelope's plan/query digests equal the digests of the plan the caller
//!     INDEPENDENTLY compiled — the *anti-substitution* guard. A prover cannot swap in a
//!     different (e.g. cheaper or unrelated) plan, because the verifier reconstructs the plan
//!     identity from its own query and rejects any mismatch before opening the proof. The digest
//!     is taken over the plan's exact wire image, including which committed columns each atom
//!     read (materialized vs. additions vs. removals), via [`plan_digest`];
//!  3. runs the per-node cryptographic checks against columns opened relative to the root.
//!
//! # What each plan kind proves (and its key soundness anchor)
//!
//! * **Bgp** — one basic-graph-pattern read of the layer; leaf case bound directly to the root.
//! * **binary joins** (`InnerJoin` / `LeftAnti` / `LeftOuter`) — the output is exactly the
//!   matched / anti-matched / left-outer-padded combination of two authenticated children on the
//!   declared key columns; composite two-key joins require Store-id keys so key packing stays
//!   injective.
//! * **Distinct** — the output is the deduplicated key set of the child (equivalence classes +
//!   multiplicity removal).
//! * **group / aggregate** (`GroupSum`, `GroupCount`, `GroupExtrema`, `Collection*`) — the
//!   aggregate is RE-DERIVED by the verifier over the root-authenticated child rows and
//!   dictionary (see the `columns` module), so no bespoke gadget is needed; grouping is by
//!   authenticated expanded-IRI order, never by raw id.
//! * **order / slice** (`ExactScalarOrder`, `ExactMultiOrder`, `ExactOrderedSlice`) — the output
//!   is a proven permutation of the child into sorted order, then a contiguous slice.
//! * **BagUnion** — the output is the exact multiset union of compatible-schema branches.
//! * **CrossRootHashJoin** — the ONE two-root node: it joins a schema graph and an instance
//!   graph by their dictionary-independent IRI hashes, so equal integers in different
//!   dictionaries never collide; both roots are bound to one database root.
//! * **TransitiveClosure** — the reachable set of a `star`/`plus` path, proven as a bounded
//!   semi-naive unroll whose soundness rests on an empty-delta fixpoint certificate.
//!
//! # Submodule map
//!
//! * `wire` — framing/serialization core: the put/get trait, the bounded reader, size/item
//!   budgets, MAGIC/version constants, [`plan_digest`], and every leaf serialization.
//! * `plan` — the [`RelationProofPlan`] operator tree, the per-operator configuration enums,
//!   structural validation, and the depth/shape-bounded schema derivation.
//! * `layer_bgp` / `bag_union` / `left_anti` / `left_outer` — the four flat portable envelopes.
//! * `columns` — verifier-recomputed derived columns for the generated/aggregate operators.
//! * `recursive` — the general recursive prover/verifier driver and the bounded
//!   transitive-closure orchestration.

mod array_walk;
mod bag_union;
mod columns;
mod layer_bgp;
mod left_anti;
mod left_outer;
mod list_walk;
mod plan;
mod reconciled;
mod recursive;
mod wire;

// ---------------------------------------------------------------------------
// Public surface. These re-exports reproduce, item-for-item, the set that was
// `pub` in the original single-file module, so every external path
// (`terminus_store::proof::envelope::X`) resolves unchanged.
// ---------------------------------------------------------------------------
pub use wire::{
    plan_digest, MAX_QUERY_ENVELOPE_BYTES, MAX_RELATION_PROOF_DEPTH, MAX_RELATION_PROOF_NODES,
    MAX_TRANSITIVE_CLOSURE_ROUNDS,
};

pub use plan::{
    resolve_relation_equality_constraint, resolve_tuple_disequality_pairs, CollectionSumTemplate,
    EqualityValueDomain, GeneratedU64ComparisonOp, RelationEqualityPlan, RelationProofPlan,
    RelationProofPlanKind, TupleDisequalityNode, TupleDisequalityPairPlan,
};

pub use layer_bgp::LayerBgpEnvelope;
pub use bag_union::BagUnionEnvelope;
pub use left_anti::LeftAntiJoinEnvelope;
pub use left_outer::LeftOuterJoinEnvelope;

pub use recursive::{
    RecursiveRelationEnvelope, RecursiveRelationProof, TransitiveClosureFixpoint,
    TransitiveClosurePairsFixpoint, TransitiveClosurePairsRound, TransitiveClosurePairsSeed,
    TransitiveClosureRound, TransitiveClosureSeed,
};

// FOUNDATION for reading walk/closure composites over the reconciled `present` (ReadDocument
// increment 6c-i) — the reconciled `TransitiveClosure` entry (FOUNDATION-B). The FOUNDATION-A leaf
// provider (`prove/verify_reconciled_leaf`, `resolve_combined_node_id`, the leaf-atom builders)
// stays `pub(crate)` in the submodule — it is the shared enabler consumed by the reconciled list
// walk (6c-ii) and schema closure (6e), not a standalone public entry.
pub use reconciled::{
    prove_reconciled_transitive_closure, resolve_reconciled_node_ids,
    verify_reconciled_transitive_closure, ReconciledClosureProof,
};

// The reconciled ordered `rdf:List` walk over a delta head's `present` (ReadDocument 6c-ii) — the
// standalone prove/verify pair the compiler lift (6c-iii) routes a delta-path `List` to. Built on the
// 6c-i foundation; the materialized `OrderedListWalk` composite stays byte-identical (a PARALLEL entry).
pub use list_walk::{
    prove_ordered_list_walk_reconciled, verify_ordered_list_walk_reconciled,
    OrderedListWalkReconciledProof,
};

// The reconciled ordered `sys:Array` walk over a delta head's `present` (ReadDocument 6d) — the
// standalone prove/verify pair the compiler lift routes a delta-path `Array` to. Built on the 6c-i
// foundation + the audited array gluing gadgets, swapping the single-layer index dict lookup for the
// combined `(id, value, datatype)` range-dict authentication; the materialized `OrderedArrayWalk`
// composite stays byte-identical (a PARALLEL entry).
pub use array_walk::{
    prove_ordered_array_walk_reconciled, verify_ordered_array_walk_reconciled,
    OrderedArrayWalkReconciledProof,
};

#[cfg(test)]
mod tests;
