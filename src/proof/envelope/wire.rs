//! Wire format and framing for the relation-proof envelope.
//!
//! This is the low-level serialization layer shared by every envelope shape. It is pure
//! encoding/decoding and resource-bounding — it performs NO trust checks (those live in each
//! envelope's `verify`).
//!
//! Key pieces:
//! * `Wire` — a tiny put/get trait implemented for every leaf type that appears in an
//!   envelope (primitives, the arkworks field/group elements, and each operator sub-proof).
//! * `Reader` — a cursor over the input with two hard budgets: a byte length it can never
//!   read past, and a GLOBAL decoded-item counter (`claim_items`) shared across all nested
//!   vectors, so an attacker cannot force tens of millions of tiny allocations from a small,
//!   individually valid input.
//! * The `*_MAGIC` tags + `*_VERSION` numbers + `CHECKSUM_BYTES` that frame each encoded
//!   envelope, and the `MAX_*` admission limits.
//! * [`plan_digest`] — the stable hash of the public (variable-blind) layer BGP plan. The
//!   verifier recomputes it from its own compiled query, which is what pins WHICH committed
//!   columns a proof was allowed to read (the anti-substitution guard).
//!
//! The `Wire` implementations for the envelope-internal transitive-closure records live here
//! too, so all serialization sits in one place; those types are defined in the `recursive`
//! module and imported below.
#![allow(unused_imports)]

use super::recursive::{
    TransitiveClosureFixpoint, TransitiveClosurePairsFixpoint, TransitiveClosurePairsRound,
    TransitiveClosurePairsSeed, TransitiveClosureRound, TransitiveClosureSeed,
};
// C4-slice-3: the three reconciled WALK/CLOSURE proof CONTAINERS (ReadDocument composites). Every
// walk sub-proof LEAF already implements `Wire` above (shared with the base materialized
// `RecursiveRelationProof` walk family — TransitiveClosureSeed/Round/Fixpoint, InnerJoinProof,
// LeftAntiJoinProof, RelationEqualityProof, RelationDistinctProof, DictLookupProof, FilterProof,
// SuccessorProof, RelationPermutationProof, RelocationProof, BagUnionProof, CombinedRangeDictProof,
// PresentBgpProof, ReconcileProof, ColumnOpening/DictOpening/ResolutionOpenings, ArkGT/ArkFr), so
// ONLY these three CONTAINERS get new `wire_struct!` + framing impls near the bottom of this file.
use super::array_walk::OrderedArrayWalkReconciledProof;
use super::list_walk::OrderedListWalkReconciledProof;
use super::reconciled::ReconciledClosureProof;
use std::convert::{TryFrom, TryInto};
use std::io::{self, Cursor};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_ff::PrimeField;
use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::transcript::Transcript;
use sha2::{Digest, Sha256};

use crate::proof::anti::{
    prove_global_left_outer_join_with_context, prove_global_not_with_context,
    prove_inner_join_with_keys_and_context, prove_left_anti_join_with_keys_and_context, prove_left_outer_join_with_keys_and_context,
    verify_global_left_outer_join, verify_global_not, verify_left_anti_join_with_keys,
    verify_inner_join_with_keys, verify_left_outer_join_with_keys, InnerJoinProof, InnerJoinResult,
    LeftAntiJoinProof, LeftOuterJoinProof,
};
use crate::proof::argument::{commit_column, ColumnOpening, StoreProverContext, SumcheckProof, Transcript254};
use crate::proof::bgp::{BgpChainProof, BgpStageProof, JoinStep, RelationCommit, RelationWitness};
use crate::proof::commitment::{
    DictOpening, LayerCommitment, ResolutionOpenings, 
    SortedNamespace, VerifierLayer,
};
use crate::proof::count::CountProof;
use crate::proof::dict::DictProof;
use crate::proof::logup_gkr::{GkrFractionProof, GkrLayerProof};
use crate::proof::dicthash::{
    prove_dict_hash_lookup, verify_dict_hash_lookup, DictHashLookupProof, HashLookupInnerProof,
};
use crate::proof::semijoin::{
    prove_hash_semijoin, verify_hash_semijoin, HashSemiJoinProof, SelMaskProof,
};
use crate::proof::distinct::{
    prove_relation_distinct_with_context, verify_relation_distinct,
    RelationDistinctEntryProof, RelationDistinctProof, RelationDistinctResult,
};
use crate::proof::disequality::{
    prove_tuple_disequality_with_context, verify_tuple_disequality, DisequalityNode,
    DisequalityPair, TupleDisequalityProof, TupleDisequalityStageProof,
};
use crate::proof::equality::{
    prove_relation_equality_with_context, verify_relation_equality, RelationEqualityConstraint,
    RelationEqualityProof,
};
use crate::proof::filter::{
    AtomFilterProof, BooleanProductProof, CompactionOnlyProof, FilterProof, FullFilterProof,
    SelectionOnlyProof,
};
use crate::proof::group::{GroupExtremaEntryProof, GroupExtremaKind, GroupExtremaProof};
use crate::proof::join2::{BinaryJoinCommitments, BinaryJoinProof, BinaryPairsProof};
use crate::proof::join_link::JoinLinkProof;
use crate::proof::membership::MembershipProof;
use crate::proof::multiorder::{prove_exact_multi_order_with_context, verify_exact_multi_order, ExactMultiOrderProof};
use crate::proof::lexsort::LexicographicSortedProof;
use crate::proof::order::{
    prove_exact_scalar_order_with_context, verify_exact_scalar_order, ExactScalarOrderProof,
    OrderDirection,
};
use crate::proof::permutation::RelationPermutationProof;
use crate::proof::query::{
    has_ground_result_unit, prove_layer_bgp_with_result_and_context,
    resolve_typed_result_objects, result_id_namespaces,
    resolve_result_values, verify_layer_bgp, verify_layer_bgp_result_log_domain,
    verify_result_columns, generated_decimal_i64_field, generated_decimal_u64_value,
    AtomTerm, CanonicalResultValue, ColumnSource,
    DictionaryAbsence,
    DictionaryAbsencePosition, LayerBgpAtom,
    LayerBgpAtomProof, LayerBgpPlan, LayerBgpProof, ObjectKindFilter, ObjectRangeBound,
    ObjectRangeSpec, RangeAtomSubproof, RangeDatatypeGuardProof, ResultColumnDescriptor,
    ResultIdNamespace, SecondRangeSubproof,
};
use crate::proof::range::{
    DictLookupProof, ObjectRangeCombineProof, ObjectRangeProof, RangeOp, RangeSelectProof,
};
use crate::proof::union::{prove_bag_union_with_context, BagUnionResult};
use crate::proof::relocate::{prove_column_relocation_with_context, verify_column_relocation, RelocationProof};
use crate::proof::sort::SortedProof;
use crate::proof::union::{verify_bag_union, BagUnionProof};
use crate::proof::{CanonicalDatatype, CanonicalObject, ProofHash};
// C4: the delta-read plain-BGP proof family (all leaves already `Wire`; the container structs
// below get `wire_struct!`/hand-written enum impls in this file — see the block near the bottom).
use crate::proof::lexdistinct::LexDistinctProof;
use crate::proof::reconcile::{
    ColumnDistinctFullFieldProof, ColumnDistinctProof, CombinedDictProof, CombinedRangeDictProof,
    CommittedGroupProof, DeltaReadProof, GroupHashProof, PresentAtomFilter, PresentBgpAtomProof,
    PresentBgpProof, PresentIdAuth, PresentRangeDatatypeGuardProof, PresentRangeSubproof,
    GroupReadAgg, GroupReadSpec, OrderReadSpec, PresentScalarOrderProof, PresentSecondRange,
    ReconcileProof, ReconciledGroupReadProof, ReconciledOrderReadProof, ReconciledReadBgpProof,
    RemintProof, ValueRenderBelt,
};
/// 12-byte magic tag opening the framed byte image of the base query-proof envelope
/// (`LayerBgpEnvelope`). Each envelope/proof shape below has its OWN distinct `*_MAGIC` tag so a
/// decoder can tell the formats apart (and reject a foreign one) before parsing any body.
///
/// WIRE-FORMAT INVARIANT — these bytes are permanent. Never change the value of an existing
/// `*_MAGIC` (below): doing so rejects every previously emitted proof of that shape. A NEW proof
/// shape gets a NEW magic; an existing shape whose byte layout changes bumps its `*_VERSION`
/// instead (see [`VERSION`]).
pub(super) const MAGIC: &[u8; 12] = b"TDBWQPROOF\0\0";
/// Framing magic for this proof shape — a permanent wire-format constant; never change these bytes (see [`MAGIC`]).
pub(super) const UNION_MAGIC: &[u8; 12] = b"TDBWQUNION\0\0";
/// Framing magic for this proof shape — a permanent wire-format constant; never change these bytes (see [`MAGIC`]).
pub(super) const ANTI_MAGIC: &[u8; 12] = b"TDBWQANTI\0\0\0";
/// Framing magic for this proof shape — a permanent wire-format constant; never change these bytes (see [`MAGIC`]).
pub(super) const OUTER_MAGIC: &[u8; 12] = b"TDBWQOUTER\0\0";
/// Framing magic for this proof shape — a permanent wire-format constant; never change these bytes (see [`MAGIC`]).
pub(super) const RECURSIVE_MAGIC: &[u8; 12] = b"TDBWQTREE\0\0\0";
// C4: the delta-read plain-BGP proof family gets its OWN independent wire format (its bytes are
// `DeltaReadProof`, NOT a `LayerBgpEnvelope`), so it carries a fresh MAGIC/VERSION and never
// touches the envelope `VERSION`/`UNION`/`ANTI`/`OUTER`/`RECURSIVE` constants above.
/// Framing magic for this proof shape — a permanent wire-format constant; never change these bytes (see [`MAGIC`]).
pub(super) const DELTA_READ_MAGIC: &[u8; 12] = b"TDBWQDLTA\0\0\0";
// C4-slice-2: the reconciled MATERIALIZED-present plain-BGP READ (a `ReconciledReadBgpProof` =
// `ReconcileProof` + `PresentBgpProof`) gets its OWN independent wire format, distinct from the
// delta-read (`DELTA_READ_*`) format and every envelope format above. Pure addition.
/// Framing magic for this proof shape — a permanent wire-format constant; never change these bytes (see [`MAGIC`]).
pub(super) const RECONCILE_READ_MAGIC: &[u8; 12] = b"TDBWQRCON\0\0\0";
// C4 GROUP byte-in: the reconciled `GroupBy`-aggregate read (a `ReconciledGroupReadProof` =
// `ReconcileProof` + `PresentBgpProof` + the delivered child result columns) gets its OWN
// independent framed wire format, distinct from the plain reconcile-read (`RECONCILE_READ_*`),
// delta-read (`DELTA_READ_*`), and every envelope format. Pure addition — no existing format is
// touched. The child columns cross the wire (they are root-authenticated at verify); the GroupSpec
// does NOT (it is the verifier's own re-lowered query — anti-substitution).
/// Framing magic for this proof shape — a permanent wire-format constant; never change these bytes (see [`MAGIC`]).
pub(super) const GROUP_READ_MAGIC: &[u8; 12] = b"TDBWQGRUP\0\0\0";
// C4 ORDER byte-in: the reconciled single-column numeric `OrderBy` read (a `ReconciledOrderReadProof`
// = `ReconcileProof` + `PresentBgpProof` + `PresentScalarOrderProof` + the delivered ordered column +
// the `OrderReadSpec` witness) gets its OWN independent framed wire format, disjoint from every other.
// Pure addition — no existing format is touched. The ordered column + order proof cross the wire
// (root-authenticated at verify); the order direction/datatype are the verifier's re-lowered query.
/// Framing magic for this proof shape — a permanent wire-format constant; never change these bytes (see [`MAGIC`]).
pub(super) const ORDER_READ_MAGIC: &[u8; 12] = b"TDBWQORDR\0\0\0";
// C4-slice-3: each of the three reconciled WALK/CLOSURE proofs is a SELF-CONTAINED shippable
// container (it carries its own `reconcile` + leaves + walk-structure sub-proofs), so each gets its
// OWN independent wire format — a fresh DISTINCT 12-byte magic, disjoint from every format above and
// from each other. Pure additions: no existing MAGIC/VERSION/golden is touched.
/// Framing magic for this proof shape — a permanent wire-format constant; never change these bytes (see [`MAGIC`]).
pub(super) const CLOSURE_READ_MAGIC: &[u8; 12] = b"TDBWQCLOS\0\0\0";
/// Framing magic for this proof shape — a permanent wire-format constant; never change these bytes (see [`MAGIC`]).
pub(super) const LIST_WALK_READ_MAGIC: &[u8; 12] = b"TDBWQLIST\0\0\0";
/// Framing magic for this proof shape — a permanent wire-format constant; never change these bytes (see [`MAGIC`]).
pub(super) const ARRAY_WALK_READ_MAGIC: &[u8; 12] = b"TDBWQARRY\0\0\0";

/// Canonical hidden column for a zero-column relation: one on every real row and zero
/// on commitment padding. This is the same representation used by all-ground BGPs.
pub(super) fn ground_unit_column(n: usize, log_domain: usize) -> Vec<ArkFr> {
    let mut column = vec![ArkFr::zero(); 1usize << log_domain];
    column[..n].fill(ArkFr::one());
    column
}
// v5 adds the explicit transcript/wire-bound object-kind filter. Older envelopes fail
// explicitly rather than guessing whether a variable object meant any value, a node,
// or a literal.
// v6: LogUp-GKR migration of `dict` membership — `DictProof` drops the committed
// `dstar` reciprocal and the batched sumcheck for a GKR fraction-tree proof + the
// three data-column leaf evaluations, so every envelope carrying a constant-atom
// (predicate/subject/object) membership has different proof bytes.
// v7: LogUp-GKR migration of `membership` (Phase 2c-2b) — `MembershipProof` dropped its
// two committed reciprocal GT commitments (`rec_u`/`rec_of`) and batched sumcheck for a
// block-disjoint GKR fraction-tree proof + three data-column leaf evaluations + a
// co-running `(1 - chi_{n_U})*w = 0` GAP-1 Identity sumcheck. Every envelope carrying a
// multi-atom BGP join (join2 embeds two memberships) or a hash semi-join has different
// proof bytes. The UNION/ANTI/OUTER envelopes embed the same BGP-chain / semi-join, so
// they bump in lockstep.
// v8: LogUp-GKR migration of the atom-filter compaction LINK — `FullFilterProof` dropped
// its two committed reciprocal GT commitments (`cstar`/`dstar`) and the fused C/D/E link
// sumcheck for a block-disjoint GKR fraction-tree proof (kept outputs == selected inputs)
// + the link data-column evaluations + a second batched opening. The IsZero selection
// sub-proof (A, B) is unchanged. Every constant-position atom (predicate/subject/object
// filter) embeds a `FullFilterProof`, so any BGP with a constant atom has different proof
// bytes; UNION/ANTI/OUTER embed the same BGP chain and bump in lockstep.
// v9: LogUp-GKR migration of the sort SHIFT LogUp — `SortedProof` dropped its two
// committed reciprocal GT commitments (`rec_c`/`rec_d`) and the multiset-link sumcheck for
// a block-disjoint GKR fraction-tree proof (`U_shift` is the index-augmented one-step shift
// of `U`) + the three link data-column leaf evaluations + a second batched opening; the
// RANGE sub-proof (booleanity + recomposition) is unchanged. Every multi-atom BGP join
// (join2 embeds a `SortedProof` on the sorted key column) has different proof bytes;
// UNION/ANTI/OUTER embed the same BGP chain / semi-join and bump in lockstep.
// v10: LogUp-GKR migration of the RANGE reciprocal-bearing pieces — (A) the compaction-only
// LINK (`CompactionOnlyProof`, the range/equality/anti/semijoin compaction path) dropped its
// two committed reciprocal GT commitments (`cstar`/`dstar`) and the C/D/E link sumcheck for a
// block-disjoint GKR fraction tree + link data-column evaluations + one batched opening; (B)
// the authenticated dict-value lookup (`DictLookupProof`) dropped its two committed reciprocal
// GT commitments (`rec_q`/`rec_d`) and the batched multiset-link sumcheck for a block-disjoint
// GKR fraction tree + seven data-column leaf evaluations + one batched opening, plus a
// co-running `(1 − χ_{n_dict})·w = 0` GAP-1 Identity sumcheck and its `w` opening. Every range
// atom embeds BOTH (its `RangeAtomSubproof` carries a `CompactionOnlyProof` and an
// `ObjectRangeProof`→`DictLookupProof`); the ANTI/OUTER envelopes embed a `CompactionOnlyProof`
// directly (`LeftAntiJoinProof.compaction`); UNION wraps `LayerBgpEnvelope` branches. So all
// four bump in lockstep (established convention).
// v11: LogUp-GKR migration of the keyed-join LINKS (Inc 3) — `BinaryPairsProof` (embedded in
// every `BinaryJoinProof`, i.e. every multi-atom BGP join stage) dropped its four committed
// reciprocal GT commitments (`rec_src_l`/`rec_cand_l`/`rec_src_r`/`rec_cand_r`) and the batched
// membership sumcheck for two block-disjoint GKR fraction-tree links (`JoinLinkProof` × LEFT/
// RIGHT: each a GKR tree + committed-column leaf evaluations + one batched opening + a co-running
// `(1 − χ_{n_R|n_S})·mult = 0` GAP-1 Identity sumcheck and its `mult` opening). The embedded
// `SortedProof`/`membership`/`count` are unchanged (already migrated). Every multi-atom BGP join
// has different proof bytes; the UNION/ANTI/OUTER envelopes embed the same BGP chain / semi-join,
// so all four bump in lockstep.
// v12: Merkle-dict I2b S5 — `LayerBgpProof` gained the optional term-less resolution witness
// `openings` (`Option<ResolutionOpenings>`). Term-bearing proofs carry `None` (one added option
// tag byte); a term-less proof ships the matched-constant + resolved-result openings. Every
// query-BGP envelope's proof bytes shift by the added tag.
// v13: Merkle-dict I2b step 3 — `RangeAtomSubproof` gained the term-less value↔term witness
// (`oid: Vec<u64>` + `value_openings: Option<ResolutionOpenings>`). A term-bearing range atom
// carries an empty vector + `None` (a length prefix + one option tag); a term-less range atom
// ships the object-id vector + per-distinct value openings. Any BGP proof containing a ranged
// atom shifts, so the format version bumps.
/// Wire-format version of the base query-proof envelope body (framed after [`MAGIC`]). A decoder
/// accepts ONLY this exact value and rejects anything else.
///
/// WIRE-FORMAT INVARIANT — bump this by one whenever the encoded byte layout of a
/// `LayerBgpEnvelope` changes (any field added / removed / reordered anywhere in the embedded
/// proof tree). The running log above records why each past bump happened; never re-use or rewind
/// a version number. Each sibling `*_VERSION` below follows the same rule for its own shape.
pub(super) const VERSION: u32 = 13;
// v10: LogUp-GKR bag-union (`Or`) migration (Inc 4) — `BagUnionProof` dropped its `N+1`
// committed reciprocal GT commitments (`reciprocal_commitments`) and the masked multiset-link
// sumcheck for ONE multi-block GKR fraction tree (`gkr`: `N+1` blocks tiled into a
// `k+high_bits`-var tree, block selected by the top `high_bits` via `eq_at_index`) + the
// `(N+1)·arity` committed-column leaf evaluations (`col_evals`) + one batched opening. All
// numerators are PUBLIC additive prefix block indicators (no committed multiplicity => no
// GAP-1, no auxiliary Identity). `commitment_root` is byte-identical (reciprocals never touched
// it). The `LeftOuterJoinProof` (OUTER) embeds a `BagUnionProof` directly and the recursive
// transitive-closure union nodes embed it too, so OUTER/RECURSIVE bump in lockstep; ANTI does
// NOT embed a union (unchanged).
// v11: Merkle-dict I2b S5 — the union branches are `LayerBgpEnvelope`s embedding a `LayerBgpProof`,
// which gained the optional `openings` field, so every branch's proof bytes shift (lockstep).
// v12: Merkle-dict I2b step 3 — the union branches embed a `LayerBgpProof` whose `RangeAtomSubproof`
// gained the term-less value↔term witness; lockstep bump.
/// Wire-format version of this shape's framed body (accepted exactly). Bump on any byte-layout change; never rewind or re-use — see [`VERSION`].
pub(super) const UNION_VERSION: u32 = 12;
// v12: Merkle-dict I2b S5 — the anti sides are `LayerBgpEnvelope`s embedding a `LayerBgpProof`
// (added `openings`); lockstep bump.
// v13: Merkle-dict I2b step 3 — the anti sides embed a `LayerBgpProof` whose `RangeAtomSubproof`
// gained the term-less value↔term witness; lockstep bump.
/// Wire-format version of this shape's framed body (accepted exactly). Bump on any byte-layout change; never rewind or re-use — see [`VERSION`].
pub(super) const ANTI_VERSION: u32 = 13;
// v10: Merkle-dict I2b S5 — the outer sides are `LayerBgpEnvelope`s embedding a `LayerBgpProof`
// (added `openings`); lockstep bump.
// v11: Merkle-dict I2b step 3 — the outer sides embed a `LayerBgpProof` whose `RangeAtomSubproof`
// gained the term-less value↔term witness; lockstep bump.
/// Wire-format version of this shape's framed body (accepted exactly). Bump on any byte-layout change; never rewind or re-use — see [`VERSION`].
pub(super) const OUTER_VERSION: u32 = 11;
// v21 generalizes grouped extrema to zero through two canonical node keys.
// v22 adds the bounded transitive-closure (`star`/`plus`) composite node.
// v23 adds the ordered `sys:Array` walk composite node (tag 24).
// v24: LogUp-GKR `DictProof` migration (see the `VERSION` note) — recursive
// envelopes embed layer-BGP leaves whose membership proof bytes changed.
// v25: LogUp-GKR `RelationPermutationProof` migration (Inc 2) — the recursive
// `ExactScalarOrder`/`ExactMultiOrder`/`OrderedListWalk`/`OrderedArrayWalk` nodes each
// embed a full-row permutation proof, which dropped its two committed reciprocal GT
// commitments and batched sumcheck for a GKR fraction-tree proof + `2*arity`
// data-column leaf evaluations, so those envelopes' proof bytes changed.
// v26: LogUp-GKR `HashLookupInnerProof` migration (Inc 2/5 corollary) — the recursive
// `CrossRootHashJoin` node embeds two `DictHashLookupProof`s, whose inner `id → hash`
// lookup dropped the two committed reciprocal GT commitments (`rec_q`/`rec_d`) and its
// batched sumcheck for a block-disjoint GKR fraction-tree proof + five data-column leaf
// evaluations, so those envelopes' proof bytes changed.
// v27: LogUp-GKR `MembershipProof` migration (Phase 2c-2b, see the `VERSION` note) —
// recursive envelopes embed BGP-chain leaves (join2's two memberships) and hash
// semi-joins whose membership proof bytes changed.
// v28: LogUp-GKR atom-filter compaction LINK migration (see the `VERSION` v8 note) —
// recursive envelopes embed layer-BGP leaves whose per-atom `FullFilterProof` bytes
// changed (committed `cstar`/`dstar` + C/D/E link sumcheck -> GKR fraction tree).
// v29: LogUp-GKR sort SHIFT LogUp migration (see the `VERSION` v9 note) — recursive
// envelopes embed BGP-chain leaves (join2/pairs sort the key column) and the ordered
// `ExactScalarOrder`/`ExactMultiOrder`/`OrderedListWalk`/`OrderedArrayWalk` composites
// (each embeds a `SortedProof`), whose bytes changed (committed `rec_c`/`rec_d` +
// multiset-link sumcheck -> GKR fraction tree).
// v30: LogUp-GKR migration of the RANGE reciprocal-bearing pieces (see the `VERSION` v10 note)
// — recursive envelopes embed layer-BGP leaves whose range-atom `CompactionOnlyProof` +
// `ObjectRangeProof`→`DictLookupProof` bytes changed, and the ordered `OrderedArrayWalk` /
// list-walk composites embed a `DictLookupProof` directly, whose committed reciprocals
// (`rec_q`/`rec_d`) + link sumcheck became a GKR fraction tree + GAP-1 Identity.
// v31: LogUp-GKR keyed-join LINK migration (see the `VERSION` v11 note) — recursive envelopes
// embed BGP-chain leaves (join2's two keyed-join links), whose four committed reciprocals +
// batched membership sumcheck became two `JoinLinkProof` GKR fraction trees + GAP-1 Identities.
// v32: LogUp-GKR bag-union (`Or`) migration (Inc 4, see the `UNION_VERSION` v10 note) — the
// bounded transitive-closure (`star`/`plus`) round nodes each embed a `BagUnionProof`, which
// dropped its `N+1` committed reciprocal GT commitments and multiset-link sumcheck for one
// multi-block GKR fraction tree + `(N+1)·arity` column leaf evaluations + a batched opening, so
// those recursive envelopes' proof bytes changed.
// v33: LogUp-GKR migration of the positional-successor primitive (`SuccessorProof`) — the ordered
// `rdf:List` walk composite (`OrderedListWalk`, tag) embeds a `SuccessorProof`, which dropped its
// two committed reciprocal GT commitments (`rec_c`/`rec_d`) and the whole shift-link sumcheck (+
// its batched opening) for a GKR fraction-tree proof (the SAME `sort_link_instance` as the already-
// migrated `SortedProof`) + the three link data-column leaf evaluations + one batched opening, so
// those recursive envelopes' proof bytes changed. (No `VERSION`/UNION/ANTI/OUTER bump: the list
// walk is a recursive-only composite; the top-level BGP/join/semi-join envelopes never embed it.)
// v34: Merkle-dict I2b S5 — recursive envelopes embed a `RecursiveRelationProof::Bgp` leaf
// (`LayerBgpEnvelope` -> `LayerBgpProof`), which gained the optional `openings` field, so every
// recursive envelope carrying a BGP leaf shifts by the added option tag.
// v35: Merkle-dict I2b step 3 — recursive envelopes embed a `RecursiveRelationProof::Bgp` leaf
// (`LayerBgpProof`) whose `RangeAtomSubproof` gained the term-less value↔term witness (`oid` +
// `value_openings`), so every recursive envelope carrying a BGP leaf with a ranged atom shifts.
// v36: Merkle-dict I2b step 3 (ORDER + GROUP) — the recursive `ExactScalarOrder`/`ExactMultiOrder`/
// `GroupExtrema` variants gained the term-less value↔term witness: `ExactScalarOrderProof` +
// `{oid, value_openings}`, `ExactMultiOrderProof` + `value_openings`, `GroupExtremaEntryProof` +
// `oid`, `GroupExtremaProof` + `value_openings`. A term-bearing order/group proof carries empty
// vectors + `None` (length prefixes + one option tag), so every recursive envelope with an
// order/group node shifts. Order/group appear ONLY under `RecursiveRelationProof` (the standalone
// UNION/ANTI/OUTER envelopes embed only `LayerBgpEnvelope` branches), so ONLY this version bumps.
/// Wire-format version of this shape's framed body (accepted exactly). Bump on any byte-layout change; never rewind or re-use — see [`VERSION`].
pub(super) const RECURSIVE_VERSION: u32 = 36;
// C4: initial version of the independent delta-read plain-BGP wire format (see `DELTA_READ_MAGIC`).
/// Wire-format version of this shape's framed body (accepted exactly). Bump on any byte-layout change; never rewind or re-use — see [`VERSION`].
pub(super) const DELTA_READ_VERSION: u32 = 1;
// C4-slice-2: initial version of the independent reconciled materialized-present read wire format
// (see `RECONCILE_READ_MAGIC`).
/// Wire-format version of this shape's framed body (accepted exactly). Bump on any byte-layout change; never rewind or re-use — see [`VERSION`].
pub(super) const RECONCILE_READ_VERSION: u32 = 1;
// C4 GROUP byte-in: initial version of the independent reconciled group-aggregate read wire format
// (see `GROUP_READ_MAGIC`).
/// Wire-format version of this shape's framed body (accepted exactly). Bump on any byte-layout change; never rewind or re-use — see [`VERSION`].
pub(super) const GROUP_READ_VERSION: u32 = 1;
// C4 ORDER byte-in: initial version of the independent reconciled order read wire format (see
// `ORDER_READ_MAGIC`).
/// Wire-format version of this shape's framed body (accepted exactly). Bump on any byte-layout change; never rewind or re-use — see [`VERSION`].
pub(super) const ORDER_READ_VERSION: u32 = 1;
// C4-slice-3: initial version of each independent reconciled WALK/CLOSURE read wire format (see the
// `CLOSURE_READ_MAGIC` / `LIST_WALK_READ_MAGIC` / `ARRAY_WALK_READ_MAGIC` constants).
/// Wire-format version of this shape's framed body (accepted exactly). Bump on any byte-layout change; never rewind or re-use — see [`VERSION`].
pub(super) const CLOSURE_READ_VERSION: u32 = 1;
/// Wire-format version of this shape's framed body (accepted exactly). Bump on any byte-layout change; never rewind or re-use — see [`VERSION`].
pub(super) const LIST_WALK_READ_VERSION: u32 = 1;
/// Wire-format version of this shape's framed body (accepted exactly). Bump on any byte-layout change; never rewind or re-use — see [`VERSION`].
pub(super) const ARRAY_WALK_READ_VERSION: u32 = 1;
pub(super) const CHECKSUM_BYTES: usize = 32;
pub(super) const MAX_ITEMS: usize = 4_000_000;
pub(super) const MAX_BLOB_BYTES: usize = 64 * 1024 * 1024;
/// Hard cap for an untrusted portable query-proof envelope.
pub const MAX_QUERY_ENVELOPE_BYTES: usize = 256 * 1024 * 1024;
/// Hard cap on the DEPTH (operator-tree nesting) of a decoded relation proof.
///
/// Keep this conservative: decoding a node includes large cryptographic proof values on
/// the stack, so a much larger syntactic limit could exhaust ordinary verifier threads
/// before the guard is reached.
pub const MAX_RELATION_PROOF_DEPTH: usize = 64;
/// Hard cap on the total NUMBER of nodes in a decoded relation-proof operator tree.
/// A decoded tree exceeding either this or [`MAX_RELATION_PROOF_DEPTH`] is rejected while
/// parsing (a resource guard on untrusted input, independent of the crypto checks).
pub const MAX_RELATION_PROOF_NODES: usize = 4096;
/// The maximum number of bounded semi-naive closure ROUNDS a `RelationProofPlanKind::TransitiveClosure`
/// proof may carry. `K` is prover-chosen (the depth at which the frontier stops growing), NOT declared
/// in the plan; soundness rests entirely on the empty-delta fixpoint, so this bound is purely a COST
/// guard against an unbounded (or pathologically deep) unroll. A closure whose reachable frontier from
/// the seed is deeper than this budget REJECTS CLOSED (there is no committed graph-diameter bound to
/// certify a shorter proof). Each round threads three audited operator proofs, so this also keeps the
/// flat round list well inside [`MAX_RELATION_PROOF_NODES`].
pub const MAX_TRANSITIVE_CLOSURE_ROUNDS: usize = 64;

pub(super) fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

pub(super) struct Reader<'a> {
    pub(super) bytes: &'a [u8],
    pub(super) pos: usize,
    pub(super) decoded_items: usize,
}

impl<'a> Reader<'a> {
    pub(super) fn take(&mut self, len: usize) -> io::Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| invalid("envelope length overflow"))?;
        if end > self.bytes.len() {
            return Err(invalid("truncated query-proof envelope"));
        }
        let value = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(value)
    }

    pub(super) fn finish(self) -> io::Result<()> {
        if self.pos != self.bytes.len() {
            return Err(invalid("trailing bytes in query-proof envelope"));
        }
        Ok(())
    }

    pub(super) fn claim_items(&mut self, len: usize) -> io::Result<()> {
        self.decoded_items = self
            .decoded_items
            .checked_add(len)
            .ok_or_else(|| invalid("decoded envelope item count overflow"))?;
        if self.decoded_items > MAX_ITEMS {
            return Err(invalid("decoded envelope exceeds global item budget"));
        }
        Ok(())
    }
}

pub(super) trait Wire: Sized {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()>;
    fn get(reader: &mut Reader<'_>) -> io::Result<Self>;
}

impl<T: Wire> Wire for Box<T> {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        (**self).put(out)
    }

    #[inline(never)]
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        Ok(Box::new(T::get(reader)?))
    }
}

impl Wire for u8 {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        out.push(*self);
        Ok(())
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        Ok(reader.take(1)?[0])
    }
}
impl Wire for bool {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        (*self as u8).put(out)
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        match u8::get(reader)? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(invalid("invalid boolean")),
        }
    }
}
impl Wire for u64 {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        out.extend_from_slice(&self.to_le_bytes());
        Ok(())
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        Ok(u64::from_le_bytes(reader.take(8)?.try_into().unwrap()))
    }
}
impl Wire for i64 {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        out.extend_from_slice(&self.to_le_bytes());
        Ok(())
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        Ok(i64::from_le_bytes(reader.take(8)?.try_into().unwrap()))
    }
}
impl Wire for usize {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        u64::try_from(*self)
            .map_err(|_| invalid("usize does not fit envelope u64"))?
            .put(out)
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        usize::try_from(u64::get(reader)?).map_err(|_| invalid("envelope u64 does not fit usize"))
    }
}
impl Wire for ProofHash {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        out.extend_from_slice(self);
        Ok(())
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        Ok(reader.take(32)?.try_into().unwrap())
    }
}
impl<T: Wire> Wire for Option<T> {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        match self {
            None => 0u8.put(out),
            Some(value) => {
                1u8.put(out)?;
                value.put(out)
            }
        }
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        match u8::get(reader)? {
            0 => Ok(None),
            1 => Ok(Some(T::get(reader)?)),
            _ => Err(invalid("invalid option tag")),
        }
    }
}
impl<T: Wire> Wire for Vec<T> {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        if self.len() > MAX_ITEMS {
            return Err(invalid("too many envelope items"));
        }
        self.len().put(out)?;
        for value in self {
            value.put(out)?;
        }
        Ok(())
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        let len = usize::get(reader)?;
        if len > MAX_ITEMS {
            return Err(invalid("too many envelope items"));
        }
        // Nested vectors share one budget. Without this, a bounded-size recursive
        // envelope can force tens of millions of decode iterations by distributing
        // individually-valid tiny vectors across many relation nodes.
        reader.claim_items(len)?;
        // `len` is attacker-controlled. Even though it is capped, reserving it up
        // front can allocate far more than the bounded input for a large `T` before
        // the first truncated element is noticed. Grow only after elements decode.
        let mut values = Vec::new();
        for _ in 0..len {
            values.push(T::get(reader)?);
        }
        Ok(values)
    }
}
impl<A: Wire, B: Wire> Wire for (A, B) {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        self.0.put(out)?;
        self.1.put(out)
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        Ok((A::get(reader)?, B::get(reader)?))
    }
}

pub(super) fn put_blob(bytes: &[u8], out: &mut Vec<u8>) -> io::Result<()> {
    if bytes.len() > MAX_BLOB_BYTES {
        return Err(invalid("envelope blob too large"));
    }
    bytes.len().put(out)?;
    out.extend_from_slice(bytes);
    Ok(())
}
pub(super) fn get_blob<'a>(reader: &mut Reader<'a>) -> io::Result<&'a [u8]> {
    let len = usize::get(reader)?;
    if len > MAX_BLOB_BYTES {
        return Err(invalid("envelope blob too large"));
    }
    reader.take(len)
}

macro_rules! canonical_wire {
    ($ty:ty) => {
        impl Wire for $ty {
            fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
                let mut bytes = Vec::new();
                self.serialize_compressed(&mut bytes)
                    .map_err(|_| invalid("canonical proof serialization failed"))?;
                put_blob(&bytes, out)
            }
            fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
                let bytes = get_blob(reader)?;
                let mut cursor = Cursor::new(bytes);
                let value = Self::deserialize_compressed(&mut cursor)
                    .map_err(|_| invalid("invalid canonical proof encoding"))?;
                if cursor.position() as usize != bytes.len() {
                    return Err(invalid("trailing canonical proof bytes"));
                }
                Ok(value)
            }
        }
    };
}
canonical_wire!(ArkFr);
canonical_wire!(ArkGT);
canonical_wire!(ColumnOpening);

macro_rules! wire_struct {
    ($ty:ty { $($field:ident),+ $(,)? }) => {
        impl Wire for $ty {
            fn put(&self, out: &mut Vec<u8>) -> io::Result<()> { $(self.$field.put(out)?;)+ Ok(()) }
            fn get(reader: &mut Reader<'_>) -> io::Result<Self> { Ok(Self { $($field: Wire::get(reader)?,)+ }) }
        }
    };
}
pub(super) use wire_struct;

wire_struct!(SumcheckProof {
    round_polys,
    claimed_evals
});
// LogUp-GKR migration: the reciprocal `dstar` commitment and the batched sumcheck
// are gone; a GKR fraction-tree proof + the three data-column evaluations at the
// leaf-opening point carry the membership statement.
wire_struct!(GkrLayerProof {
    round_polys,
    child_p_left,
    child_p_right,
    child_q_left,
    child_q_right
});
wire_struct!(GkrFractionProof {
    root_p,
    root_q,
    layers
});
wire_struct!(DictProof {
    gkr,
    col_evals,
    opening,
    mult_commitment
});
wire_struct!(FilterProof { sumcheck, opening });
// LogUp-GKR migration of the atom-filter compaction LINK: the two committed reciprocal
// commitments (`cstar`/`dstar`) and the fused C/D/E link sumcheck are gone; a GKR
// fraction-tree proof + the link data-column evaluations at the leaf-opening point + a
// second batched opening carry the "kept outputs == selected inputs" multiset link. The
// IsZero selection sub-proof (A, B) is unchanged.
wire_struct!(FullFilterProof {
    sumcheck,
    opening,
    sel_commitment,
    w_commitment,
    link,
    link_col_evals,
    link_opening
});
wire_struct!(SelectionOnlyProof {
    sumcheck,
    opening,
    w_commitment
});
wire_struct!(BooleanProductProof { sumcheck, opening });
// LogUp-GKR migration of the compaction-only LINK (the range/equality/anti/semijoin
// compaction path): the two committed reciprocal commitments (`cstar`/`dstar`) and the
// C/D/E link sumcheck are gone; a GKR fraction-tree proof + the link data-column
// evaluations at the leaf-opening point + one batched opening carry the "kept outputs ==
// selected inputs" multiset link (the SAME `filter_link_instance` the full filter emits).
wire_struct!(CompactionOnlyProof {
    link,
    link_col_evals,
    link_opening
});
wire_struct!(RelationEqualityProof {
    indicator,
    selection_commitment,
    inverse_commitment,
    compaction
});
wire_struct!(TupleDisequalityStageProof {
    indicator,
    selection_commitment,
    inverse_commitment,
    compaction,
    output_commitments,
    output_len
});
wire_struct!(TupleDisequalityProof { stages });
wire_struct!(RelationDistinctEntryProof {
    filters,
    commitments,
    lengths
});
wire_struct!(RelationDistinctProof { entries });
// LogUp-GKR migration (Inc 2): the two committed reciprocal commitments and the
// batched sumcheck are gone; a GKR fraction-tree proof + the `2*arity` data-column
// evaluations at the leaf-opening point carry the two-multiset equality.
wire_struct!(RelationPermutationProof {
    gkr,
    col_evals,
    opening
});
wire_struct!(ExactScalarOrderProof {
    permutation,
    relocate_output_id,
    relocate_dict_id,
    relocate_dict_value,
    relocate_dict_datatype,
    com_output_id_at_d,
    com_dict_id_at_d,
    com_dict_value_at_d,
    com_dict_datatype_at_d,
    lookup,
    datatype_filter,
    datatype_selection_commitment,
    datatype_inverse_commitment,
    sorted,
    // Merkle-dict I2b step 3: the term-less value↔term witness (empty vec + `None` on a
    // term-bearing artifact). Uncommitted — re-checked by `verify_dict_value_terms`.
    oid,
    value_openings
});
wire_struct!(LexicographicSortedProof { sorted });
// Merkle-dict I2b step 3: `value_openings` carries the term-less value↔term witness; the per-key
// `oid` is the shipped `result_columns[k]`, so no `oid` field here. `None` on a term-bearing artifact.
wire_struct!(ExactMultiOrderProof { permutation, lexicographic, value_openings });

impl Wire for OrderDirection {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        match self { Self::Ascending => 0u8, Self::Descending => 1u8 }.put(out)
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        match u8::get(reader)? {
            0 => Ok(Self::Ascending),
            1 => Ok(Self::Descending),
            _ => Err(invalid("invalid order direction")),
        }
    }
}
wire_struct!(GroupExtremaEntryProof {
    group,
    group_commitments,
    group_len,
    membership,
    membership_commitments,
    membership_len,
    relocate_value,
    relocate_dict_id,
    relocate_dict_value,
    relocate_dict_datatype,
    com_value_at_d,
    com_dict_id_at_d,
    com_dict_value_at_d,
    com_dict_datatype_at_d,
    range,
    com_range_selection,
    // Merkle-dict I2b step 3: the term-less per-group value-column object ids (empty on a
    // term-bearing artifact). Uncommitted — recommitted to `com_value_at_d` at verify.
    oid
});
wire_struct!(GroupExtremaProof { entries, value_openings });

impl Wire for GroupExtremaKind {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        match self { Self::Min => 0u8, Self::Max => 1u8 }.put(out)
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        match u8::get(reader)? {
            0 => Ok(Self::Min),
            1 => Ok(Self::Max),
            _ => Err(invalid("invalid group-extrema kind")),
        }
    }
}
wire_struct!(RelocationProof {
    source_eval,
    target_eval,
    source_opening,
    target_opening
});
wire_struct!(BagUnionProof {
    relocations,
    relocated_commitments,
    gkr,
    col_evals,
    opening
});
wire_struct!(LeftAntiJoinProof {
    left_relocations,
    right_relocations,
    join_commitments,
    join,
    join_len,
    zero,
    zero_selection_commitment,
    zero_inverse_commitment,
    compaction
});
wire_struct!(InnerJoinProof { anti, unmatched_left_commitments, unmatched_len });
// The positional-successor primitive (`proof::sort`) consumed by the ordered `rdf:List` walk.
// LogUp-GKR migration of the shift LogUp: the two committed reciprocal commitments
// (`rec_c`/`rec_d`) and the multiset-link sumcheck (the whole `sumcheck`/`opening`) are gone;
// a GKR fraction-tree proof (the SAME `sort_link_instance` as `SortedProof`) + the three link
// data-column evaluations at the leaf point + one batched opening carry the range-free shift
// link (`next` is the index-augmented one-step shift of `base` with committed `mult`).
wire_struct!(crate::proof::sort::SuccessorProof {
    mult_commitment,
    link,
    link_col_evals,
    link_opening
});
wire_struct!(TransitiveClosureRound {
    inner,
    inner_out,
    union,
    union_out,
    distinct,
    next_rows
});
wire_struct!(TransitiveClosureFixpoint { inner, inner_out, anti, anti_out });
wire_struct!(TransitiveClosurePairsRound {
    inner,
    inner_out,
    union,
    union_out,
    distinct,
    next_pairs
});
wire_struct!(TransitiveClosurePairsFixpoint { inner, inner_out, anti, anti_out });
wire_struct!(TransitiveClosurePairsSeed { distinct, pairs });
impl Wire for TransitiveClosureSeed {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        match self {
            TransitiveClosureSeed::Reflexive => 0u8.put(out),
            TransitiveClosureSeed::Successor {
                inner,
                inner_out,
                distinct,
                rows,
            } => {
                1u8.put(out)?;
                inner.put(out)?;
                inner_out.put(out)?;
                distinct.put(out)?;
                rows.put(out)
            }
        }
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        match u8::get(reader)? {
            0 => Ok(TransitiveClosureSeed::Reflexive),
            1 => Ok(TransitiveClosureSeed::Successor {
                inner: Wire::get(reader)?,
                inner_out: Wire::get(reader)?,
                distinct: Wire::get(reader)?,
                rows: Wire::get(reader)?,
            }),
            _ => Err(invalid("invalid transitive-closure seed tag")),
        }
    }
}
wire_struct!(LeftOuterJoinProof {
    anti,
    unmatched_left_commitments,
    unmatched_len,
    union
});
// LogUp-GKR migration of the sort SHIFT LogUp: the two committed reciprocal
// commitments (`rec_c`/`rec_d`) and the multiset-link sumcheck are gone; a GKR
// fraction-tree proof + the link data-column evaluations at the leaf point + a second
// batched opening carry the index-augmented shift link (`U_shift` is the one-step shift
// of `U` with committed multiplicity `mult_c`). The RANGE sub-proof (booleanity +
// recomposition) is unchanged.
wire_struct!(SortedProof {
    sumcheck,
    opening,
    u_shift_commitment,
    mult_c_commitment,
    link,
    link_col_evals,
    link_opening,
    bit_commitments
});
// LogUp-GKR migration (GAP-1 auxiliary-Identity closure): `MembershipProof` dropped the
// two committed reciprocal GT commitments (`rec_u`/`rec_of`) and the batched sumcheck for
// a block-disjoint GKR fraction-tree proof + the three data-column leaf evaluations, plus
// a co-running `(1 - chi_{n_U})*w = 0` Identity sumcheck and its `w` opening.
wire_struct!(MembershipProof {
    gkr,
    col_evals,
    opening,
    id_sumcheck,
    id_opening
});
wire_struct!(CountProof { sumcheck, opening });
wire_struct!(BinaryJoinCommitments {
    r_key,
    r_other,
    s_key,
    s_other,
    r_other_out,
    key,
    s_other_out,
    rho_l,
    rho_r,
    u,
    w_l,
    w_r
});
// LogUp-GKR migration of the keyed-join links (Inc 3): `BinaryPairsProof` dropped its four
// committed reciprocal GT commitments (`rec_src_l`/`rec_cand_l`/`rec_src_r`/`rec_cand_r`) and
// the batched membership sumcheck for two block-disjoint GKR fraction-tree links (each a
// `JoinLinkProof`: GKR tree + committed-column leaf evaluations + one batched opening + a
// co-running `(1 - chi_{n_R|n_S})*mult = 0` GAP-1 Identity sumcheck and its `mult` opening).
wire_struct!(JoinLinkProof {
    gkr,
    col_evals,
    opening,
    id_sumcheck,
    id_opening
});
wire_struct!(BinaryPairsProof {
    mult_l_commitment,
    mult_r_commitment,
    link_l,
    link_r,
    sorted
});
wire_struct!(BinaryJoinProof {
    pairs,
    sorted_u,
    membership_l,
    membership_r,
    count,
    n_u
});
wire_struct!(BgpStageProof {
    acc_relocations,
    rel_relocations,
    commitments,
    join,
    m_j
});
wire_struct!(BgpChainProof { stages });
wire_struct!(RangeSelectProof {
    sumcheck,
    opening,
    bit_commitments
});
// LogUp-GKR migration of the authenticated dict-value lookup: the two committed reciprocal
// commitments (`rec_q`/`rec_d`) and the batched multiset-link sumcheck are gone; a
// block-disjoint GKR fraction-tree proof + the seven data-column leaf evaluations + one
// batched opening carry the `Σ w/(α−d) − Σ 1/(α−q) = 0` link, plus a co-running
// `(1 − χ_{n_dict})·w = 0` GAP-1 Identity sumcheck and its `w` opening.
wire_struct!(DictLookupProof {
    gkr,
    col_evals,
    opening,
    id_sumcheck,
    id_opening,
    com_v_val,
    com_v_dt,
    com_w
});
wire_struct!(HashLookupInnerProof {
    gkr,
    col_evals,
    opening,
    com_v_hash,
    com_w
});
// C4 fully-term-less value-render belt: the three relocations + their targets, the audited
// `id → hash` inner lookup, and the real-prefix authenticated hash column. Every field is already
// `Wire` (`RelocationProof`, `ArkGT`, `HashLookupInnerProof`, `Vec<ArkFr>`). It rides the walk
// proofs as an `Option<ValueRenderBelt>` (length-tagged ⇒ byte-identity fixed point holds).
wire_struct!(ValueRenderBelt {
    belt_log,
    reloc_ids,
    com_ids_at_belt,
    reloc_dict_id,
    com_dict_id_at_belt,
    reloc_dict_hash,
    com_dict_hash_at_belt,
    lookup,
    value_hashes
});
wire_struct!(DictHashLookupProof {
    relocate_dict_hash,
    relocate_dict_id,
    com_dict_hash_at_d,
    com_dict_id_at_d,
    relocate_obj,
    com_obj_at_d,
    lookup
});
wire_struct!(SelMaskProof { sumcheck, opening });
wire_struct!(HashSemiJoinProof {
    sel_mask,
    com_sel,
    compaction,
    com_x_kept,
    com_h_kept,
    com_w,
    membership
});
wire_struct!(ObjectRangeCombineProof { sumcheck, opening });
wire_struct!(ObjectRangeProof {
    lookup,
    datatype,
    com_dt_match,
    com_dt_w,
    range,
    com_range_sel,
    combine
});
wire_struct!(SecondRangeSubproof {
    object_range,
    com_range_sel_at_d,
    relocate_range_sel,
    com_range_sel_at_table,
    intersect,
    com_intersection_sel
});
wire_struct!(RangeAtomSubproof {
    relocate_object,
    relocate_dict_id,
    relocate_dict_value,
    relocate_dict_datatype,
    com_object_at_d,
    com_dict_id_at_d,
    com_dict_value_at_d,
    com_dict_datatype_at_d,
    object_range,
    com_range_sel_at_d,
    relocate_range_sel,
    com_range_sel_at_table,
    second,
    equality,
    com_equality_sel,
    // NUMERIC-FAMILY datatype guard (fail-closed fidelity): `Some` only for a genuine numeric
    // range; `None` for datatype-agnostic object-KIND filters.
    datatype_guard,
    combine,
    com_combined_sel,
    compaction,
    // Merkle-dict I2b step 3: the term-less value↔term witness (empty vec + `None` on a
    // term-bearing artifact). Uncommitted — re-checked per id by `verify_dict_value_terms`.
    oid,
    value_openings
});
// The DATATYPE-HOMOGENEITY guard sub-proof (present only for a genuine numeric range atom).
wire_struct!(RangeDatatypeGuardProof {
    relocate_dt_match,
    com_dt_match_at_table,
    bad,
    com_bad
});
wire_struct!(LayerBgpAtomProof {
    predicate_membership,
    subject_membership,
    object_membership,
    filter,
    range,
    kept_commitments,
    m,
    subject_id,
    object_id
});
// Merkle-dict I2b S5: one positional Merkle opening (leaf preimage + sibling path). The
// UNCOMMITTED term-less resolution witness — never in `commitment_root`/`resolution_root`,
// re-verified per id by `verify_opening` at consume time.
wire_struct!(DictOpening {
    id,
    datatype_tag,
    term_bytes,
    position,
    siblings
});
// Merkle-dict I2b S5: the two-namespace opening witness. Encoded as node/value then predicate
// opening lists (deterministic id order); decoded through `from_wire`, which keys each by its own
// id and rejects duplicates. Pure parsing — verification re-checks each opening against the
// trusted root.
impl Wire for ResolutionOpenings {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        // A prover-side oracle opens lazily; only its minted `consumed_witness` is shippable.
        if self.is_oracle() {
            return Err(invalid("a resolution-openings oracle is not a shippable witness"));
        }
        if self.keyed_lookups().next().is_some() {
            return Err(invalid("keyed term-index lookups are carried only by the checkpoint envelope"));
        }
        put_id_openings(self, out)
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        let node_value: Vec<DictOpening> = Wire::get(reader)?;
        let predicate: Vec<DictOpening> = Wire::get(reader)?;
        ResolutionOpenings::from_wire(node_value, predicate)
    }
}

/// The id-indexed openings of a witness (node/value then predicate lists), in the format every
/// carrier uses; the term-less envelope appends its sorted-index lookups after them.
pub(super) fn put_id_openings(openings: &ResolutionOpenings, out: &mut Vec<u8>) -> io::Result<()> {
    {
        let this = openings;
        if this.is_oracle() {
            return Err(invalid("a resolution-openings oracle is not a shippable witness"));
        }
        let nv: Vec<&DictOpening> = this.node_value_openings().collect();
        let pred: Vec<&DictOpening> = this.predicate_openings().collect();
        if nv.len() > MAX_ITEMS || pred.len() > MAX_ITEMS {
            return Err(invalid("too many resolution openings"));
        }
        nv.len().put(out)?;
        for opening in nv {
            opening.put(out)?;
        }
        pred.len().put(out)?;
        for opening in pred {
            opening.put(out)?;
        }
        Ok(())
    }
}
// Merkle-dict I2b S5: `openings` carries the term-less resolution witness (`None` on term-bearing
// proofs — a single option tag). This is a WIRE-FORMAT change: every envelope embedding a
// `LayerBgpProof` (the query BGP leaf, the bag-union branches, the anti/outer sides, and the
// recursive Bgp leaf) shifts, so `VERSION`/`UNION`/`ANTI`/`OUTER`/`RECURSIVE` all bump below.
wire_struct!(LayerBgpProof {
    atoms,
    chain,
    dictionary_absence,
    openings
});
wire_struct!(JoinStep {
    acc_key,
    rel_key,
    additional_keys
});
wire_struct!(ObjectRangeBound { op, c });
wire_struct!(ObjectRangeSpec {
    op,
    datatype,
    c,
    b_bits,
    second
});
// `source` is part of each atom's wire image, so `plan_digest` (and therefore the envelope
// anti-substitution check) fixes which committed columns THIS atom read. This is what stops a
// prover relabelling one atom of a cross-source join from additions to materialized (or
// swapping additions↔removals): the verifier reconstructs the same per-atom source and opens
// against the columns it selects.
wire_struct!(LayerBgpAtom {
    subject,
    predicate_term,
    predicate_variable,
    object,
    object_range,
    object_kind,
    subject_object_equal,
    source
});
impl Wire for ColumnSource {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        match self {
            Self::Materialized => 0u8,
            Self::Additions => 1,
            Self::Removals => 2,
        }
        .put(out)
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        match u8::get(reader)? {
            0 => Ok(Self::Materialized),
            1 => Ok(Self::Additions),
            2 => Ok(Self::Removals),
            _ => Err(invalid("invalid column-source tag")),
        }
    }
}
// `source` is part of the plan's wire image, so `plan_digest` (and therefore the envelope
// anti-substitution check) binds which columns the proof read.
wire_struct!(LayerBgpPlan { atoms, joins, source });

wire_struct!(DictionaryAbsence {
    atom_index,
    position
});

impl Wire for DictionaryAbsencePosition {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        match self {
            Self::Predicate => 0u8,
            Self::Subject => 1u8,
            Self::Object => 2u8,
        }
        .put(out)
    }

    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        match u8::get(reader)? {
            0 => Ok(Self::Predicate),
            1 => Ok(Self::Subject),
            2 => Ok(Self::Object),
            _ => Err(invalid("invalid dictionary-absence position tag")),
        }
    }
}

impl Wire for AtomFilterProof {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        match self {
            Self::Identity => 0u8.put(out),
            Self::Compacted(proof) => {
                1u8.put(out)?;
                proof.put(out)
            }
        }
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        match u8::get(reader)? {
            0 => Ok(Self::Identity),
            1 => Ok(Self::Compacted(Wire::get(reader)?)),
            _ => Err(invalid("invalid atom-filter tag")),
        }
    }
}
impl Wire for RangeOp {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        match self {
            Self::Lt => 0u8,
            Self::Le => 1,
            Self::Gt => 2,
            Self::Ge => 3,
        }
        .put(out)
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        match u8::get(reader)? {
            0 => Ok(Self::Lt),
            1 => Ok(Self::Le),
            2 => Ok(Self::Gt),
            3 => Ok(Self::Ge),
            _ => Err(invalid("invalid range-op tag")),
        }
    }
}
impl Wire for ObjectKindFilter {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        match self {
            Self::Any => 0u8,
            Self::Node => 1,
            Self::Literal => 2,
        }
        .put(out)
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        match u8::get(reader)? {
            0 => Ok(Self::Any),
            1 => Ok(Self::Node),
            2 => Ok(Self::Literal),
            _ => Err(invalid("invalid object-kind filter tag")),
        }
    }
}
impl Wire for AtomTerm {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        match self {
            Self::Variable => 0u8.put(out),
            Self::NodeConstant(bytes) => {
                1u8.put(out)?;
                bytes.put(out)
            }
            Self::ValueConstant(datatype, bytes) => {
                2u8.put(out)?;
                (*datatype as u8).put(out)?;
                bytes.put(out)
            }
        }
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        match u8::get(reader)? {
            0 => Ok(Self::Variable),
            1 => Ok(Self::NodeConstant(Wire::get(reader)?)),
            2 => Ok(Self::ValueConstant(
                CanonicalDatatype::try_from(u8::get(reader)?)?,
                Wire::get(reader)?,
            )),
            _ => Err(invalid("invalid atom-term tag")),
        }
    }
}
impl Wire for CanonicalObject {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        match self {
            Self::Node(bytes) => {
                0u8.put(out)?;
                bytes.put(out)
            }
            Self::Value { datatype, lexical } => {
                1u8.put(out)?;
                (*datatype as u8).put(out)?;
                lexical.put(out)
            }
        }
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        match u8::get(reader)? {
            0 => Ok(Self::Node(Wire::get(reader)?)),
            1 => Ok(Self::Value {
                datatype: CanonicalDatatype::try_from(u8::get(reader)?)?,
                lexical: Wire::get(reader)?,
            }),
            _ => Err(invalid("invalid canonical-object tag")),
        }
    }
}
impl Wire for ResultIdNamespace {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        match self {
            Self::NodeValue => 0u8.put(out),
            Self::Predicate => 1u8.put(out),
            Self::GeneratedDecimalU64 => 2u8.put(out),
            Self::GeneratedDecimalI64 => 3u8.put(out),
        }
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        match u8::get(reader)? {
            0 => Ok(Self::NodeValue),
            1 => Ok(Self::Predicate),
            2 => Ok(Self::GeneratedDecimalU64),
            3 => Ok(Self::GeneratedDecimalI64),
            _ => Err(invalid("invalid result namespace tag")),
        }
    }
}
wire_struct!(ResultColumnDescriptor {
    namespace,
    nullable
});
impl Wire for CanonicalResultValue {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        match self {
            Self::Object(object) => {
                0u8.put(out)?;
                object.put(out)
            }
            Self::Null => 1u8.put(out),
            Self::GeneratedDecimalU64(value) => {
                2u8.put(out)?;
                value.put(out)
            }
            Self::GeneratedDecimalI64(value) => {
                3u8.put(out)?;
                value.put(out)
            }
        }
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        match u8::get(reader)? {
            0 => Ok(Self::Object(Wire::get(reader)?)),
            1 => Ok(Self::Null),
            2 => Ok(Self::GeneratedDecimalU64(u64::get(reader)?)),
            3 => Ok(Self::GeneratedDecimalI64(i64::get(reader)?)),
            _ => Err(invalid("invalid canonical result value tag")),
        }
    }
}

// ===========================================================================
// C4: the delta-read plain-BGP proof family. Every cryptographic LEAF already implements
// `Wire` above (shared with the base `LayerBgpProof` family); the container structs below just
// forward to those leaves in field order via `wire_struct!` (plus one hand-written enum tag for
// `PresentAtomFilter`). Bottom-up so each impl reads against already-`Wire` fields.
// ===========================================================================
// 1. The limb-lexicographic strict-distinctness proof (I4-a no-re-mint carrier).
wire_struct!(LexDistinctProof {
    com_hi,
    com_lo,
    com_lo_shift,
    hi_nondec,
    lo_succ,
    aux_commitments,
    bit_commitments,
    sumcheck,
    opening
});
// 2. `present`'s per-position id-column authentication.
wire_struct!(PresentIdAuth { com_id, lookup });
// 3. One committed group's `id -> hash` inner lookup + the binding relocation.
wire_struct!(GroupHashProof {
    obj_reloc,
    com_obj_at_k,
    lookup
});
// 4. Bounded-key distinctness of one committed column (present-is-a-set guard).
wire_struct!(ColumnDistinctProof { perm, com_sorted, sorted });
// 5. An atom's SECOND range bound (half-open interval) authenticated selection + intersection.
wire_struct!(PresentSecondRange {
    object_range,
    com_range_sel,
    intersect,
    com_intersection_sel
});
// 6. The combined `(id, value, datatype)` range dictionary relocated to the working domain.
wire_struct!(CombinedRangeDictProof {
    union,
    com_id,
    com_value,
    com_datatype,
    log,
    n,
    reloc_id,
    reloc_value,
    reloc_datatype,
    com_id_at_k,
    com_value_at_k,
    com_datatype_at_k
});
// 7. The combined `(hash, id)` node/value | predicate dictionary relocated to the working domain.
wire_struct!(CombinedDictProof {
    union,
    com_hash,
    com_id,
    log,
    n,
    reloc_hash,
    reloc_id,
    com_hash_at_k,
    com_id_at_k
});
// 8. Full-field (whole-`Fr`) distinctness of one committed column (I4-a via `LexDistinctProof`).
wire_struct!(ColumnDistinctFullFieldProof { perm, com_sorted, lex });
// 9. The no-re-mint / no-re-use carrier: full-field distinctness of the combined nv + predicate
// HASH columns (`nv`, `pred`) AND their ID columns (`nv_id`, `pred_id`). The id fields are appended
// after the hash fields (matching `prove_remint`'s field/transcript order); this grows the wire
// format additively — the byte-identity round-trip goldens re-derive their bytes, so no golden
// byte array changes.
wire_struct!(RemintProof { nv, pred, nv_id, pred_id });
// 10. A committed group's three authenticated canonical hash columns (S/O vs nv, P vs predicate).
wire_struct!(CommittedGroupProof { s, o, p });
// 11. A ranged atom's sub-proofs over `present`.
wire_struct!(PresentRangeSubproof {
    object_range,
    com_range_sel,
    second,
    equality,
    com_equality_sel,
    // NUMERIC-FAMILY datatype guard (fail-closed fidelity): `Com(equality_sel ⊙
    // numeric_mismatch) == Com(0)` over the authenticated v_dt/dt_match (same working domain).
    datatype_guard,
    combine,
    com_combined_sel,
    compaction
});
wire_struct!(PresentRangeDatatypeGuardProof {
    bad,
    com_bad
});
// 12. The per-atom filter: a PLAIN equality atom or a RANGED object atom (hand-written enum tag,
// mirroring `AtomFilterProof` above). Which arm is used is fixed by the PUBLIC atom, not the proof.
impl Wire for PresentAtomFilter {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        match self {
            Self::Plain {
                selection,
                com_sel,
                compaction,
            } => {
                0u8.put(out)?;
                selection.put(out)?;
                com_sel.put(out)?;
                compaction.put(out)
            }
            Self::Range(proof) => {
                1u8.put(out)?;
                proof.put(out)
            }
        }
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        match u8::get(reader)? {
            0 => Ok(Self::Plain {
                selection: Wire::get(reader)?,
                com_sel: Wire::get(reader)?,
                compaction: Wire::get(reader)?,
            }),
            1 => Ok(Self::Range(Wire::get(reader)?)),
            _ => Err(invalid("invalid present atom-filter tag")),
        }
    }
}
// 13. The per-atom half of a `present` BGP proof: the atom filter + compacted kept-column commits.
wire_struct!(PresentBgpAtomProof { filter, com_kept, m });
// 14. A general n-atom BGP proof over `present`. `openings`/`range_object_oid`/`range_value_openings`
// are the UNCOMMITTED term-less resolution witnesses — now serialized (they ride the existing
// Option/Vec/ResolutionOpenings `Wire` impls), still re-verified per id at resolve time.
wire_struct!(PresentBgpProof {
    auth_s,
    auth_o,
    auth_p,
    present_distinct,
    range_dict,
    atoms,
    chain,
    result_len,
    result_log_domain,
    result_arity,
    openings,
    range_object_oid,
    range_value_openings
});
// 15. The top-level delta-read proof. `remint` is `pub(crate)` — accessible here (same crate).
wire_struct!(DeltaReadProof {
    combined_nv,
    combined_pred,
    remint,
    working_log,
    group,
    group_len,
    bgp,
    openings
});

impl DeltaReadProof {
    /// Frame this delta-read proof for shipping as BYTES: MAGIC ‖ VERSION ‖ body ‖ SHA-256
    /// checksum, exactly like the base `LayerBgpProof::encode` but with the independent
    /// `DELTA_READ_*` constants. Serialization does NOT change verification: the decoded proof is
    /// re-verified by the SAME `verify_delta_read{,_from_views}` function.
    pub fn encode(&self) -> io::Result<Vec<u8>> {
        let mut out = Vec::new();
        out.extend_from_slice(DELTA_READ_MAGIC);
        out.extend_from_slice(&DELTA_READ_VERSION.to_le_bytes());
        self.put(&mut out)?;
        if out
            .len()
            .checked_add(CHECKSUM_BYTES)
            .map_or(true, |len| len > MAX_QUERY_ENVELOPE_BYTES)
        {
            return Err(invalid("delta-read proof too large"));
        }
        let checksum = Sha256::digest(&out);
        out.extend_from_slice(&checksum);
        Ok(out)
    }

    /// Decode a framed delta-read proof. This is ONLY parsing under the strict size/item budgets —
    /// it performs NO trust checks; the decoded proof is handed back to `verify_delta_read*`.
    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < DELTA_READ_MAGIC.len() + 4 + CHECKSUM_BYTES
            || bytes.len() > MAX_QUERY_ENVELOPE_BYTES
        {
            return Err(invalid("invalid delta-read proof size"));
        }
        let body_len = bytes.len() - CHECKSUM_BYTES;
        let (body, checksum) = bytes.split_at(body_len);
        if Sha256::digest(body).as_slice() != checksum {
            return Err(invalid("delta-read proof checksum mismatch"));
        }
        let mut reader = Reader {
            bytes: body,
            pos: 0,
            decoded_items: 0,
        };
        if reader.take(DELTA_READ_MAGIC.len())? != DELTA_READ_MAGIC {
            return Err(invalid("unknown delta-read proof magic"));
        }
        let version = u32::from_le_bytes(reader.take(4)?.try_into().unwrap());
        if version != DELTA_READ_VERSION {
            return Err(invalid("unsupported delta-read proof version"));
        }
        let proof = Self::get(&mut reader)?;
        reader.finish()?;
        Ok(proof)
    }
}

// ===========================================================================
// C4-slice-2: WIRE SERIALIZATION of the reconciled MATERIALIZED-present plain-BGP READ. Unlike a
// delta-read (one `DeltaReadProof`), a reconciled present read carries TWO proof structs — the
// `ReconcileProof` (proving `present` IS the head triple set) and the already-`Wire`
// `PresentBgpProof` (the n-atom BGP over that `present`). Every field of `ReconcileProof` is
// already `Wire` (all leaves shared with the base/delta families); the single new container just
// forwards in field order, and the `ReconciledReadBgpProof` carrier ships both in one framed blob.
// ===========================================================================
// The full N-layer reconciliation proof. `remint` is `pub(crate)` — accessible here (same crate).
wire_struct!(ReconcileProof {
    combined_nv,
    combined_pred,
    remint,
    working_log,
    groups,
    com_present_s,
    com_present_p,
    com_present_o,
    link_gkr,
    link_col_evals,
    link_opening,
    present_len
});
// The shippable reconciled materialized-present read: the reconciliation proof + the present BGP.
wire_struct!(ReconciledReadBgpProof { reconcile, bgp });

impl ReconciledReadBgpProof {
    /// Frame this reconciled materialized-present read for shipping as BYTES:
    /// MAGIC ‖ VERSION ‖ body ‖ SHA-256 checksum, byte-identically to
    /// [`DeltaReadProof::encode`] but with the independent `RECONCILE_READ_*` constants.
    /// Serialization does NOT change verification: the decoded proof is re-verified by the SAME
    /// `verify_read_bgp{,_from_views}` function over the verifier's OWN query spec.
    pub fn encode(&self) -> io::Result<Vec<u8>> {
        let mut out = Vec::new();
        out.extend_from_slice(RECONCILE_READ_MAGIC);
        out.extend_from_slice(&RECONCILE_READ_VERSION.to_le_bytes());
        self.put(&mut out)?;
        if out
            .len()
            .checked_add(CHECKSUM_BYTES)
            .map_or(true, |len| len > MAX_QUERY_ENVELOPE_BYTES)
        {
            return Err(invalid("reconciled read proof too large"));
        }
        let checksum = Sha256::digest(&out);
        out.extend_from_slice(&checksum);
        Ok(out)
    }

    /// Decode a framed reconciled materialized-present read proof. This is ONLY parsing under the
    /// strict size/item budgets — it performs NO trust checks; the decoded proof is handed back to
    /// `verify_read_bgp*`.
    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < RECONCILE_READ_MAGIC.len() + 4 + CHECKSUM_BYTES
            || bytes.len() > MAX_QUERY_ENVELOPE_BYTES
        {
            return Err(invalid("invalid reconciled read proof size"));
        }
        let body_len = bytes.len() - CHECKSUM_BYTES;
        let (body, checksum) = bytes.split_at(body_len);
        if Sha256::digest(body).as_slice() != checksum {
            return Err(invalid("reconciled read proof checksum mismatch"));
        }
        let mut reader = Reader {
            bytes: body,
            pos: 0,
            decoded_items: 0,
        };
        if reader.take(RECONCILE_READ_MAGIC.len())? != RECONCILE_READ_MAGIC {
            return Err(invalid("unknown reconciled read proof magic"));
        }
        let version = u32::from_le_bytes(reader.take(4)?.try_into().unwrap());
        if version != RECONCILE_READ_VERSION {
            return Err(invalid("unsupported reconciled read proof version"));
        }
        let proof = Self::get(&mut reader)?;
        reader.finish()?;
        Ok(proof)
    }
}

// ===========================================================================
// C4 GROUP byte-in: WIRE SERIALIZATION of the reconciled GroupBy-aggregate read. Its bytes are the
// reconcile + present-BGP proof (both already `Wire`) PLUS the delivered child result columns
// (`Vec<Vec<ArkFr>>`, via the existing Vec/ArkFr `Wire` impls). The container forwards in field
// order via `wire_struct!` and gets its OWN independent framed format (`GROUP_READ_*`), disjoint
// from every format above. The GroupSpec is NOT on the wire (anti-substitution). Serialization
// changes NO verification: the decoded proof re-verifies through the SAME
// `verify_and_resolve_group_*_bgp_from_views`.
// ===========================================================================
// The wire-carried GroupSpec (anti-substitution witness). `key_columns` rides the `Vec<usize>` Wire
// impl; the aggregate is a hand-written tag (Count / Sum / Extrema) carrying the value column, the
// datatype (as a `u8` tag via `CanonicalDatatype`), and the min/max kind (already `Wire`).
impl Wire for GroupReadAgg {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        match self {
            Self::Count => 0u8.put(out),
            Self::Sum { value_column, datatype } => {
                1u8.put(out)?;
                value_column.put(out)?;
                (*datatype as u8).put(out)
            }
            Self::Extrema { value_column, datatype, kind } => {
                2u8.put(out)?;
                value_column.put(out)?;
                (*datatype as u8).put(out)?;
                kind.put(out)
            }
        }
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        match u8::get(reader)? {
            0 => Ok(Self::Count),
            1 => Ok(Self::Sum {
                value_column: Wire::get(reader)?,
                datatype: CanonicalDatatype::try_from(u8::get(reader)?)?,
            }),
            2 => Ok(Self::Extrema {
                value_column: Wire::get(reader)?,
                datatype: CanonicalDatatype::try_from(u8::get(reader)?)?,
                kind: Wire::get(reader)?,
            }),
            _ => Err(invalid("invalid group-read aggregate tag")),
        }
    }
}
wire_struct!(GroupReadSpec { key_columns, agg });
wire_struct!(ReconciledGroupReadProof { reconcile, bgp, child_columns, spec });

impl ReconciledGroupReadProof {
    /// Frame this reconciled group-aggregate read for shipping as BYTES:
    /// MAGIC ‖ VERSION ‖ body ‖ SHA-256 checksum, byte-identically to
    /// [`ReconciledReadBgpProof::encode`] but with the independent `GROUP_READ_*` constants.
    pub fn encode(&self) -> io::Result<Vec<u8>> {
        let mut out = Vec::new();
        out.extend_from_slice(GROUP_READ_MAGIC);
        out.extend_from_slice(&GROUP_READ_VERSION.to_le_bytes());
        self.put(&mut out)?;
        if out
            .len()
            .checked_add(CHECKSUM_BYTES)
            .map_or(true, |len| len > MAX_QUERY_ENVELOPE_BYTES)
        {
            return Err(invalid("reconciled group read proof too large"));
        }
        let checksum = Sha256::digest(&out);
        out.extend_from_slice(&checksum);
        Ok(out)
    }

    /// Decode a framed reconciled group-aggregate read proof. ONLY parsing under the strict
    /// size/item budgets — NO trust checks; the decoded proof is handed back to
    /// `verify_and_resolve_group_*_bgp_from_views`.
    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < GROUP_READ_MAGIC.len() + 4 + CHECKSUM_BYTES
            || bytes.len() > MAX_QUERY_ENVELOPE_BYTES
        {
            return Err(invalid("invalid reconciled group read proof size"));
        }
        let body_len = bytes.len() - CHECKSUM_BYTES;
        let (body, checksum) = bytes.split_at(body_len);
        if Sha256::digest(body).as_slice() != checksum {
            return Err(invalid("reconciled group read proof checksum mismatch"));
        }
        let mut reader = Reader {
            bytes: body,
            pos: 0,
            decoded_items: 0,
        };
        if reader.take(GROUP_READ_MAGIC.len())? != GROUP_READ_MAGIC {
            return Err(invalid("unknown reconciled group read proof magic"));
        }
        let version = u32::from_le_bytes(reader.take(4)?.try_into().unwrap());
        if version != GROUP_READ_VERSION {
            return Err(invalid("unsupported reconciled group read proof version"));
        }
        let proof = Self::get(&mut reader)?;
        reader.finish()?;
        Ok(proof)
    }
}

// ===========================================================================
// C4 ORDER byte-in: WIRE SERIALIZATION of the reconciled single-column numeric OrderBy read. The
// order proof `PresentScalarOrderProof` carries real crypto (the permutation + combined-range-dict +
// authenticated-value lookup + datatype filter + nondecreasing-sort subproofs); every field is
// ALREADY `Wire` (all leaves shared with the base/BGP families), so it just gets a `wire_struct!`.
// The container adds the reconcile + BGP proofs, the delivered ordered column, and the `OrderReadSpec`
// witness, and gets its OWN independent framed format (`ORDER_READ_*`). Serialization changes NO
// verification: the decoded proof re-verifies through the SAME `verify_and_resolve_order_bgp_from_views`.
// ===========================================================================
// C2a value↔term witness (`oid` + `value_openings`) rides the existing Vec/Option/ResolutionOpenings
// `Wire` impls; empty/`None` on a term-bearing artifact.
wire_struct!(PresentScalarOrderProof {
    permutation,
    com_output,
    range_dict,
    lookup,
    datatype_filter,
    datatype_selection_commitment,
    datatype_inverse_commitment,
    sorted,
    oid,
    value_openings
});
// The wire-carried order spec (anti-substitution witness): datatype as a `u8` tag via
// `CanonicalDatatype`, direction via its existing `Wire` impl.
impl Wire for OrderReadSpec {
    fn put(&self, out: &mut Vec<u8>) -> io::Result<()> {
        (self.datatype as u8).put(out)?;
        self.direction.put(out)
    }
    fn get(reader: &mut Reader<'_>) -> io::Result<Self> {
        Ok(Self {
            datatype: CanonicalDatatype::try_from(u8::get(reader)?)?,
            direction: Wire::get(reader)?,
        })
    }
}
wire_struct!(ReconciledOrderReadProof { reconcile, bgp, order, ordered_columns, spec });

impl ReconciledOrderReadProof {
    /// Frame this reconciled order read for shipping as BYTES: MAGIC ‖ VERSION ‖ body ‖ SHA-256
    /// checksum, byte-identically to [`ReconciledGroupReadProof::encode`] but with the independent
    /// `ORDER_READ_*` constants.
    pub fn encode(&self) -> io::Result<Vec<u8>> {
        let mut out = Vec::new();
        out.extend_from_slice(ORDER_READ_MAGIC);
        out.extend_from_slice(&ORDER_READ_VERSION.to_le_bytes());
        self.put(&mut out)?;
        if out
            .len()
            .checked_add(CHECKSUM_BYTES)
            .map_or(true, |len| len > MAX_QUERY_ENVELOPE_BYTES)
        {
            return Err(invalid("reconciled order read proof too large"));
        }
        let checksum = Sha256::digest(&out);
        out.extend_from_slice(&checksum);
        Ok(out)
    }

    /// Decode a framed reconciled order read proof. ONLY parsing under the strict size/item budgets —
    /// NO trust checks; the decoded proof is handed back to `verify_and_resolve_order_bgp_from_views`.
    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < ORDER_READ_MAGIC.len() + 4 + CHECKSUM_BYTES
            || bytes.len() > MAX_QUERY_ENVELOPE_BYTES
        {
            return Err(invalid("invalid reconciled order read proof size"));
        }
        let body_len = bytes.len() - CHECKSUM_BYTES;
        let (body, checksum) = bytes.split_at(body_len);
        if Sha256::digest(body).as_slice() != checksum {
            return Err(invalid("reconciled order read proof checksum mismatch"));
        }
        let mut reader = Reader {
            bytes: body,
            pos: 0,
            decoded_items: 0,
        };
        if reader.take(ORDER_READ_MAGIC.len())? != ORDER_READ_MAGIC {
            return Err(invalid("unknown reconciled order read proof magic"));
        }
        let version = u32::from_le_bytes(reader.take(4)?.try_into().unwrap());
        if version != ORDER_READ_VERSION {
            return Err(invalid("unsupported reconciled order read proof version"));
        }
        let proof = Self::get(&mut reader)?;
        reader.finish()?;
        Ok(proof)
    }
}

// ===========================================================================
// C4-slice-3: WIRE SERIALIZATION of the three reconciled WALK/CLOSURE proofs (ReadDocument
// composites). Unlike the plain/present reads, each walk is SELF-CONTAINED — it carries its own
// `reconcile` + per-leaf `PresentBgpProof`s + the C3 term-less openings + the walk-structure
// sub-proofs — so the container IS the shippable proof (no extra carrier wrapper). Every field type
// is ALREADY `Wire` (the walk sub-proof leaves are shared with the base materialized
// `RecursiveRelationProof` walk family), so the three containers just forward in field order via
// `wire_struct!`; each then gets its OWN independent framed format, byte-identically to
// `DeltaReadProof::encode`/`ReconciledReadBgpProof::encode` but with distinct `*_READ_*` constants.
// Serialization changes NO verification: each decoded proof is re-verified by the SAME audited
// `verify_*_reconciled_from_views` / `verify_reconciled_transitive_closure_from_views`.
// ===========================================================================
// 1. The reconciled bounded transitive-closure proof (FOUNDATION-B).
wire_struct!(ReconciledClosureProof {
    reconcile,
    edge_leaf,
    seed,
    rounds,
    fixpoint,
    seed_opening,
    reachable_openings
});

impl ReconciledClosureProof {
    /// Frame this reconciled transitive-closure proof for shipping as BYTES: MAGIC ‖ VERSION ‖
    /// body ‖ SHA-256 checksum, byte-identically to [`DeltaReadProof::encode`] but with the
    /// independent `CLOSURE_READ_*` constants. Serialization does NOT change verification: the
    /// decoded proof is re-verified by the SAME
    /// `verify_reconciled_transitive_closure{,_from_views}` over the verifier's OWN query spec.
    pub fn encode(&self) -> io::Result<Vec<u8>> {
        let mut out = Vec::new();
        out.extend_from_slice(CLOSURE_READ_MAGIC);
        out.extend_from_slice(&CLOSURE_READ_VERSION.to_le_bytes());
        self.put(&mut out)?;
        if out
            .len()
            .checked_add(CHECKSUM_BYTES)
            .map_or(true, |len| len > MAX_QUERY_ENVELOPE_BYTES)
        {
            return Err(invalid("reconciled closure proof too large"));
        }
        let checksum = Sha256::digest(&out);
        out.extend_from_slice(&checksum);
        Ok(out)
    }

    /// Decode a framed reconciled transitive-closure proof. This is ONLY parsing under the strict
    /// size/item budgets — it performs NO trust checks; the decoded proof is handed back to
    /// `verify_reconciled_transitive_closure*`.
    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < CLOSURE_READ_MAGIC.len() + 4 + CHECKSUM_BYTES
            || bytes.len() > MAX_QUERY_ENVELOPE_BYTES
        {
            return Err(invalid("invalid reconciled closure proof size"));
        }
        let body_len = bytes.len() - CHECKSUM_BYTES;
        let (body, checksum) = bytes.split_at(body_len);
        if Sha256::digest(body).as_slice() != checksum {
            return Err(invalid("reconciled closure proof checksum mismatch"));
        }
        let mut reader = Reader {
            bytes: body,
            pos: 0,
            decoded_items: 0,
        };
        if reader.take(CLOSURE_READ_MAGIC.len())? != CLOSURE_READ_MAGIC {
            return Err(invalid("unknown reconciled closure proof magic"));
        }
        let version = u32::from_le_bytes(reader.take(4)?.try_into().unwrap());
        if version != CLOSURE_READ_VERSION {
            return Err(invalid("unsupported reconciled closure proof version"));
        }
        let proof = Self::get(&mut reader)?;
        reader.finish()?;
        Ok(proof)
    }
}

// 2. The reconciled ordered `rdf:List` walk proof. Every field already `Wire`; forwarded in def
// order (the C3 term-less `element_openings`/`nil_opening` ride the `Option<…>` `Wire` impls and
// stay uncommitted witnesses re-verified at resolve time).
wire_struct!(OrderedListWalkReconciledProof {
    reconcile,
    anchor_leaf,
    anchor_opening,
    head_id,
    rest_edge_leaf,
    first_edge_leaf,
    seed,
    rounds,
    fixpoint,
    strip_nil,
    cons_commitments,
    cons_len,
    cons_log_domain,
    walk_perm,
    com_walk,
    head_opening,
    rest_join,
    rest_join_out,
    rest_reloc_subj,
    rest_reloc_obj,
    com_next,
    successor,
    terminator,
    terminator_out,
    first_join,
    first_join_out,
    first_reloc_subj,
    first_reloc_obj,
    com_elements,
    nested_join,
    nested_join_out,
    element_ids,
    element_openings,
    nil_opening,
    value_belt
});

impl OrderedListWalkReconciledProof {
    /// Frame this reconciled ordered `rdf:List` walk proof for shipping as BYTES: MAGIC ‖ VERSION ‖
    /// body ‖ SHA-256 checksum, byte-identically to [`DeltaReadProof::encode`] but with the
    /// independent `LIST_WALK_READ_*` constants. Serialization does NOT change verification: the
    /// decoded proof is re-verified by the SAME `verify_ordered_list_walk_reconciled{,_from_views}`.
    pub fn encode(&self) -> io::Result<Vec<u8>> {
        let mut out = Vec::new();
        out.extend_from_slice(LIST_WALK_READ_MAGIC);
        out.extend_from_slice(&LIST_WALK_READ_VERSION.to_le_bytes());
        self.put(&mut out)?;
        if out
            .len()
            .checked_add(CHECKSUM_BYTES)
            .map_or(true, |len| len > MAX_QUERY_ENVELOPE_BYTES)
        {
            return Err(invalid("reconciled list-walk proof too large"));
        }
        let checksum = Sha256::digest(&out);
        out.extend_from_slice(&checksum);
        Ok(out)
    }

    /// Decode a framed reconciled ordered `rdf:List` walk proof. ONLY parsing under the strict
    /// size/item budgets — NO trust checks; the decoded proof is handed back to
    /// `verify_ordered_list_walk_reconciled*`.
    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < LIST_WALK_READ_MAGIC.len() + 4 + CHECKSUM_BYTES
            || bytes.len() > MAX_QUERY_ENVELOPE_BYTES
        {
            return Err(invalid("invalid reconciled list-walk proof size"));
        }
        let body_len = bytes.len() - CHECKSUM_BYTES;
        let (body, checksum) = bytes.split_at(body_len);
        if Sha256::digest(body).as_slice() != checksum {
            return Err(invalid("reconciled list-walk proof checksum mismatch"));
        }
        let mut reader = Reader {
            bytes: body,
            pos: 0,
            decoded_items: 0,
        };
        if reader.take(LIST_WALK_READ_MAGIC.len())? != LIST_WALK_READ_MAGIC {
            return Err(invalid("unknown reconciled list-walk proof magic"));
        }
        let version = u32::from_le_bytes(reader.take(4)?.try_into().unwrap());
        if version != LIST_WALK_READ_VERSION {
            return Err(invalid("unsupported reconciled list-walk proof version"));
        }
        let proof = Self::get(&mut reader)?;
        reader.finish()?;
        Ok(proof)
    }
}

// 3. The reconciled ordered `sys:Array` walk proof. Every field already `Wire` (incl. the
// `CombinedRangeDictProof` density-dict, the `DictLookupProof`/`FilterProof` datatype gadgets, and
// the C3 term-less `value_openings`/`sys_array_opening`/`index_value_openings` witnesses).
wire_struct!(OrderedArrayWalkReconciledProof {
    reconcile,
    cells_leaf,
    index_edge_leaf,
    value_edge_leaf,
    type_edge_leaf,
    n,
    cell_log_domain,
    index_join,
    index_join_out,
    index_reloc_subj,
    index_reloc_obj,
    com_index,
    value_join,
    value_join_out,
    value_reloc_subj,
    value_reloc_obj,
    com_value,
    type_join,
    type_join_out,
    type_reloc_subj,
    type_reloc_obj,
    com_type,
    type_equality,
    type_equality_out,
    range_dict,
    common_log_domain,
    reloc_index_at_d,
    com_index_at_d,
    reloc_value_at_d,
    com_value_at_d,
    reloc_dict_id,
    com_dict_id_at_d,
    reloc_dict_value,
    com_dict_value_at_d,
    reloc_dict_datatype,
    com_dict_datatype_at_d,
    lookup,
    datatype_filter,
    datatype_selection_commitment,
    datatype_inverse_commitment,
    order_perm,
    com_out,
    value_ids,
    value_openings,
    sys_array_opening,
    index_oid,
    index_value_openings,
    value_belt
});

impl OrderedArrayWalkReconciledProof {
    /// Frame this reconciled ordered `sys:Array` walk proof for shipping as BYTES: MAGIC ‖ VERSION ‖
    /// body ‖ SHA-256 checksum, byte-identically to [`DeltaReadProof::encode`] but with the
    /// independent `ARRAY_WALK_READ_*` constants. Serialization does NOT change verification: the
    /// decoded proof is re-verified by the SAME `verify_ordered_array_walk_reconciled{,_from_views}`.
    pub fn encode(&self) -> io::Result<Vec<u8>> {
        let mut out = Vec::new();
        out.extend_from_slice(ARRAY_WALK_READ_MAGIC);
        out.extend_from_slice(&ARRAY_WALK_READ_VERSION.to_le_bytes());
        self.put(&mut out)?;
        if out
            .len()
            .checked_add(CHECKSUM_BYTES)
            .map_or(true, |len| len > MAX_QUERY_ENVELOPE_BYTES)
        {
            return Err(invalid("reconciled array-walk proof too large"));
        }
        let checksum = Sha256::digest(&out);
        out.extend_from_slice(&checksum);
        Ok(out)
    }

    /// Decode a framed reconciled ordered `sys:Array` walk proof. ONLY parsing under the strict
    /// size/item budgets — NO trust checks; the decoded proof is handed back to
    /// `verify_ordered_array_walk_reconciled*`.
    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < ARRAY_WALK_READ_MAGIC.len() + 4 + CHECKSUM_BYTES
            || bytes.len() > MAX_QUERY_ENVELOPE_BYTES
        {
            return Err(invalid("invalid reconciled array-walk proof size"));
        }
        let body_len = bytes.len() - CHECKSUM_BYTES;
        let (body, checksum) = bytes.split_at(body_len);
        if Sha256::digest(body).as_slice() != checksum {
            return Err(invalid("reconciled array-walk proof checksum mismatch"));
        }
        let mut reader = Reader {
            bytes: body,
            pos: 0,
            decoded_items: 0,
        };
        if reader.take(ARRAY_WALK_READ_MAGIC.len())? != ARRAY_WALK_READ_MAGIC {
            return Err(invalid("unknown reconciled array-walk proof magic"));
        }
        let version = u32::from_le_bytes(reader.take(4)?.try_into().unwrap());
        if version != ARRAY_WALK_READ_VERSION {
            return Err(invalid("unsupported reconciled array-walk proof version"));
        }
        let proof = Self::get(&mut reader)?;
        reader.finish()?;
        Ok(proof)
    }
}

/// Encode then decode a [`DictProof`] through the envelope `Wire` framing, so the
/// migrated GKR proof shape can be asserted to survive serialization and re-verify.
#[cfg(test)]
pub(crate) fn wire_roundtrip_dict_proof(proof: &DictProof) -> io::Result<DictProof> {
    let mut bytes = Vec::new();
    proof.put(&mut bytes)?;
    let mut reader = Reader {
        bytes: &bytes,
        pos: 0,
        decoded_items: 0,
    };
    let decoded = DictProof::get(&mut reader)?;
    reader.finish()?;
    Ok(decoded)
}

/// Stable identity of the variable-blind public plan. Callers independently compile
/// their query and verification rejects a different plan before opening the proof.
pub fn plan_digest(plan: &LayerBgpPlan) -> io::Result<ProofHash> {
    let mut bytes = Vec::new();
    plan.put(&mut bytes)?;
    let mut hasher = Sha256::new();
    hasher.update(b"terminus-store/query-proof-plan/v2");
    hasher.update(&bytes);
    Ok(hasher.finalize().into())
}

// ---------------------------------------------------------------------------------------------
// Term-sorted index lookups (single-layer term-less envelope v2 only).
// ---------------------------------------------------------------------------------------------

