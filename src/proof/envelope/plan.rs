//! The relational operator tree ([`RelationProofPlan`]) and its structural validation.
//!
//! A plan is the PUBLIC, variable-blind description of the query being proven: an operator
//! tree whose leaves are single-layer basic-graph-pattern reads and whose interior nodes are
//! relational operators ([`RelationProofPlanKind`]). It carries no private data and no witness
//! — the verifier reconstructs the identical plan from its own trusted query, then checks the
//! proof against it. That is why the plan is what the anti-substitution digest is taken over.
//!
//! This module owns:
//! * the plan tree and the per-operator configuration enums (comparison ops, aggregate
//!   templates, equality/disequality shapes);
//! * `shape_bounded` / [`RelationProofPlan::result_shape`] — a depth- and node-bounded walk
//!   that derives each node's output column schema (namespaces + nullability) and rejects any
//!   structurally invalid tree BEFORE any cryptography runs. Getting the shape right here is
//!   what lets every operator downstream trust its children's column types;
//! * the resolve helpers ([`resolve_relation_equality_constraint`],
//!   [`resolve_tuple_disequality_pairs`]) that turn public constants into the dictionary ids
//!   the gadgets consume, resolved against the trusted layer (an absent constant maps to an
//!   impossible id, i.e. fails closed).
#![allow(unused_imports)]

use super::wire::{invalid, MAX_RELATION_PROOF_DEPTH, MAX_RELATION_PROOF_NODES};
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
use crate::proof::commitment::{LayerCommitment, VerifierLayer};
use crate::proof::count::CountProof;
use crate::proof::dict::DictProof;
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
    ObjectRangeSpec, RangeAtomSubproof, ResultColumnDescriptor, ResultIdNamespace,
    SecondRangeSubproof,
};
use crate::proof::range::{
    DictLookupProof, ObjectRangeCombineProof, ObjectRangeProof, RangeOp, RangeSelectProof,
};
use crate::proof::union::{prove_bag_union_with_context, BagUnionResult};
use crate::proof::relocate::{prove_column_relocation_with_context, verify_column_relocation, RelocationProof};
use crate::proof::sort::SortedProof;
use crate::proof::union::{verify_bag_union, BagUnionProof};
use crate::proof::{CanonicalDatatype, CanonicalObject, ProofHash};
/// One authoritative recursive read-statement plan. Operator children are complete
/// authenticated relations, so composition never needs to reinterpret a child proof as
/// a leaf BGP. `query_digest` is the caller-domain identity for this exact node.
#[derive(Clone, Debug)]
pub struct RelationProofPlan {
    pub query_digest: ProofHash,
    pub kind: RelationProofPlanKind,
}

#[derive(Clone, Debug)]
pub enum RelationProofPlanKind {
    Bgp(LayerBgpPlan),
    /// One generated nonnegative xsd:decimal equal to the complete logical triple set
    /// visible at this immutable layer head. The trusted root binds materialized_count.
    LayerTripleCount,
    /// Exact row filter for a generated nonnegative integral decimal column. The child
    /// values are already authenticated and public, so the verifier derives the complete
    /// filtered relation without a separate polynomial proof.
    GeneratedU64Filter {
        input: Box<RelationProofPlan>,
        column: usize,
        op: GeneratedU64ComparisonOp,
        constant: u64,
        constant_on_left: bool,
    },
    /// Exact row filter for a generated signed integral decimal column. Signedness is
    /// selected by the authenticated result descriptor, never inferred from the field.
    GeneratedI64Filter {
        input: Box<RelationProofPlan>,
        column: usize,
        op: GeneratedU64ComparisonOp,
        constant: i64,
        constant_on_left: bool,
    },
    /// Exact zero-/one-key grouping and checked u64/i64 sum over fixed-width integral
    /// Store values.
    /// The child rows and root-bound dictionary are public to the verifier, so the
    /// aggregate is derived directly without exposing the private intermediate list.
    GroupSum {
        input: Box<RelationProofPlan>,
        key_columns: Vec<usize>,
        value_column: usize,
        datatype: CanonicalDatatype,
    },
    /// Exact zero-/one-/two-node-key grouping with the private bag consumed by
    /// Prolog `length/2`. The generated nonnegative decimal is the number of
    /// authenticated child solutions in each group, including duplicates.
    GroupCount {
        input: Box<RelationProofPlan>,
        key_columns: Vec<usize>,
    },
    /// Exact cardinality of the private list produced by Prolog `findall/3` and
    /// consumed immediately by `length/2`. Unlike zero-key GroupBy, empty input
    /// emits one generated decimal row containing zero.
    CollectionCount {
        input: Box<RelationProofPlan>,
    },
    /// Exact sum of the private list produced by Prolog `findall/3`, consumed
    /// immediately by `sum_list/2`. Empty input emits one typed generated zero.
    CollectionSum {
        input: Box<RelationProofPlan>,
        template: CollectionSumTemplate,
        datatype: CanonicalDatatype,
    },
    /// Exact min/max of a private list produced by `findall/3`. Empty input
    /// fails like leading Prolog RangeMin/RangeMax and therefore emits no row.
    CollectionExtrema {
        input: Box<RelationProofPlan>,
        value_column: usize,
        datatype: CanonicalDatatype,
        kind: GroupExtremaKind,
    },
    BagUnion {
        branches: Vec<RelationProofPlan>,
        /// Parent output column -> child output column, once per ordered branch.
        mappings: Vec<Vec<usize>>,
    },
    /// Exact bag projection/reorder. Rows, domain, order, and multiplicity are inherited
    /// unchanged from the authenticated child; only a unique subset of columns is kept.
    /// An empty subset converts a non-ground relation to the canonical hidden UNIT
    /// representation of a zero-column relation.
    BagProjection {
        input: Box<RelationProofPlan>,
        indices: Vec<usize>,
    },
    /// Exact unordered set of the selected one or two child columns. This statement
    /// authenticates equivalence classes and multiplicity removal, but not SWI's
    /// first-occurrence output order.
    Distinct {
        input: Box<RelationProofPlan>,
        key_columns: Vec<usize>,
    },
    LeftAnti {
        left: Box<RelationProofPlan>,
        right: Box<RelationProofPlan>,
        key_pairs: Vec<(usize, usize)>,
    },
    InnerJoin {
        left: Box<RelationProofPlan>,
        right: Box<RelationProofPlan>,
        key_pairs: Vec<(usize, usize)>,
    },
    /// CROSS-DICTIONARY two-root hash join — the ONE plan node that reads TWO committed
    /// graphs (INSTANCE + SCHEMA) at once. `left` is proven against the SCHEMA root, `right`
    /// against the INSTANCE root; each side's join-key id column is resolved to an
    /// authenticated `hash_object(IRI)` column (via [`crate::proof::dicthash::prove_dict_hash_lookup`])
    /// against ITS OWN root, and the two hash columns are joined by the UNCHANGED audited
    /// [`crate::proof::anti::prove_inner_join_with_keys`]. Because `hash_object` is deterministic AND
    /// dictionary-INDEPENDENT, the SAME IRI hashes identically in both graphs, so the hash join
    /// reconciles the two separate dictionaries correctly; a schema-dict id and an instance-dict
    /// id that are equal INTEGERS but different IRIs have DIFFERENT hashes and never match (no
    /// raw cross-dict id comparison anywhere). Both per-graph roots are bound to the trusted
    /// [`crate::proof::commitment::database_root`] before either is trusted as an anchor.
    ///
    /// This node is TOP-LEVEL ONLY: it is proven/verified through the dedicated two-layer entries
    /// `RecursiveRelationEnvelope::prove_cross_root` / `RecursiveRelationEnvelope::verify_cross_root`;
    /// the single-layer `RecursiveRelationEnvelope::prove_bounded` recursion rejects it closed
    /// (it has only one layer), so every EXISTING single-root plan is byte-identical. `left` is
    /// specialized to a SINGLE key column (the class set, `left_key_col == 0`, delivering nothing to
    /// the output); the output schema is `right`'s columns MINUS `right_key_col` (instance namespace).
    CrossRootHashJoin {
        /// The SCHEMA-graph child (e.g. the downward `sys:inherits` subsumption closure of a class):
        /// a single node/value id column that becomes the join membership set.
        left: Box<RelationProofPlan>,
        /// The INSTANCE-graph child (e.g. `Triple(?x, rdf:type, ?class)`): its `right_key_col` is the
        /// class id column; the OTHER columns (e.g. `?x`) are delivered as the output.
        right: Box<RelationProofPlan>,
        /// The single join-key column of `left` (must be `0`; `left` is single-column).
        left_key_col: usize,
        /// The join-key (class) column of `right`; dropped from the output.
        right_key_col: usize,
    },
    LeftOuter {
        left: Box<RelationProofPlan>,
        right: Box<RelationProofPlan>,
        key_pairs: Vec<(usize, usize)>,
    },
    Equality {
        input: Box<RelationProofPlan>,
        constraint: RelationEqualityPlan,
    },
    /// Exact EDGE-SIMPLE row filter for a finite single-predicate repeated-predicate
    /// property path. Drops the rows in which some hop-pair reuses a stored edge (the
    /// only over-count of a conjunctive bag-join versus Prolog's edge-simple path
    /// semantics); keeps every node-repeating solution. One committed filter runs per
    /// `pairs` entry (a hop-pair), each carrying all columns through, so the schema is
    /// inherited unchanged (the endpoint projection is a separate `BagProjection`).
    ExactTupleDisequality {
        input: Box<RelationProofPlan>,
        pairs: Vec<TupleDisequalityPairPlan>,
    },
    ExactScalarOrder {
        input: Box<RelationProofPlan>,
        datatype: CanonicalDatatype,
        direction: OrderDirection,
    },
    ExactMultiOrder {
        input: Box<RelationProofPlan>,
        datatypes: Vec<CanonicalDatatype>,
        key_indices: Vec<usize>,
        directions: Vec<OrderDirection>,
    },
    /// Exact contiguous slice of an authenticated ordered relation. `None` returns the
    /// complete suffix after `offset`; `Some(limit)` returns at most that many rows.
    ExactOrderedSlice {
        input: Box<RelationProofPlan>,
        offset: u64,
        limit: Option<u64>,
    },
    /// Exact zero-/one-/two-node-key `GroupBy`, with its private bag consumed
    /// immediately by `RangeMin`/`RangeMax`. Declared key priority determines
    /// canonical expanded-IRI tuple order; the value remains an original Store ID.
    GroupExtrema {
        input: Box<RelationProofPlan>,
        key_columns: Vec<usize>,
        value_column: usize,
        datatype: CanonicalDatatype,
        kind: GroupExtremaKind,
    },
    /// Bounded transitive closure (`star`/`plus`) of a SINGLE predicate from a BOUND source,
    /// delivering the exact reachable SET of endpoint nodes as one nonnullable NodeValue column.
    ///
    /// This is a COMPOSITE ORCHESTRATION over the audited `InnerJoin`/`BagUnion`/`Distinct`/`LeftAnti`
    /// operators — NO new cryptographic gadget. The plan node is `K`-INDEPENDENT: it carries only the
    /// closure PATTERN (edge relation, seed node, star-vs-plus). The number of semi-naive rounds `K`
    /// lives in the PROOF, not the plan, so the verifier reconstructs this identity from its own trusted
    /// AST without knowing `K`, then reads `K` from the delivered proof and validates each round.
    ///
    /// Faithful to `path.pl` (`star(P)` = zero-or-more, `plus(P)` = one-or-more) ONLY as a SET closure,
    /// which equals the reference's edge-simple BAG of endpoints exactly when the surrounding query
    /// collapses the bag to a set (an endpoint `Distinct`/`Count`). The compiler therefore emits this
    /// node ONLY under such a gate; bare `star`/`plus` rejects closed.
    TransitiveClosure {
        /// The edge relation `E`: any 2-column [start, end] (both NodeValue) authenticated relation,
        /// NOT just a single-predicate `Triple(?s, p, ?o)` BGP. For a single predicate it is a
        /// `Bgp(Triple(?s, p, ?o))`; for a COMPOUND closure body (`(p|q)`, `(p/q)`) it is the derived
        /// edge relation — a deduped bag-union of the branch edges (`(p|q)`), or the two-atom join
        /// projected to its (start, end) endpoints (`(p/q)`). The store closure driver is
        /// source-agnostic over `E`: it takes it BY COMMITMENT (proven ONCE via `prove_bounded`, then
        /// reused across every round and the fixpoint), so ONLY what `E` *is* changes for a compound
        /// body — the semi-naive round/fixpoint machinery is byte-identical. The compiler-side digest
        /// binds `E`'s full structure, so `(p/q)+` / `(p|q)+` / single-predicate are non-interchangeable.
        edge: Box<RelationProofPlan>,
        /// The bound seed node (a node IRI); resolved to its dictionary id at prove/verify.
        seed: CanonicalObject,
        /// `true` = `star` (the identity branch `X = Y` unconditionally succeeds, so the seed itself is
        /// in the set); `false` = `plus` (one-or-more hops, so the seed is present only via a cycle).
        reflexive: bool,
        /// Traversal direction over the [subject, object] edge. `false` walks `subject -> object`
        /// (join `A_k.node == E.subject`, take `E.object`); `true` walks the INVERSE edge
        /// `object -> subject` (join `A_k.node == E.object`, take `E.subject`) — this is how a bound
        /// TARGET (or an inverse-predicate closure) is mirrored, `path.pl:114-134`. The ONLY thing it
        /// changes is the InnerJoin key column on `E`; every other operator is direction-agnostic.
        inverted: bool,
    },
    /// Bounded ALL-PAIRS transitive closure (`path(?x, plus(p), ?y)`, both endpoints free), delivering
    /// the exact 2-column reachability relation `[src, dst]` (every pair connected by ≥1 `p`-edge).
    ///
    /// Same COMPOSITE discipline as [`Self::TransitiveClosure`] — a `K`-INDEPENDENT plan node over the
    /// audited `InnerJoin`/`BagUnion`/`Distinct`/`LeftAnti` cores, NO new cryptographic gadget — but the
    /// accumulator is a 2-column pairs relation rather than a 1-column reachable set. PLUS-ONLY: there is
    /// no reflexive/star variant, because `path.pl`'s both-free `star` identity branch (`X = Y`,
    /// path.pl:183) yields a single UNBOUND row, NOT a domain diagonal — so the compiler reject-closes
    /// all-pairs `star` rather than fabricate a diagonal the reference never returns.
    ///
    /// Faithful to `path.pl` as a SET closure: the reference's both-free `plus` enumerates the
    /// edge-simple BAG of endpoint pairs, which equals this pairs SET exactly once the surrounding
    /// `Distinct([x,y])` collapses the bag (a shortest walk repeats no edge, so edge-simple ≥1-hop
    /// reachability = transitive reachability). The compiler emits this node ONLY under that gate.
    TransitiveClosurePairs {
        /// The edge relation `E`: any 2-column [start, end] (both NodeValue) authenticated relation, NOT
        /// just a single-predicate `Triple(?s, p, ?o)` BGP. For a single predicate it is a
        /// `Bgp(Triple(?s, p, ?o))`; for a DERIVED all-pairs edge — an all-pairs WILDCARD `.` closure or
        /// a COMPOUND `(p|q)+` / `(p/q)+` body (W1) — it is the projected/deduped derived relation. The
        /// pairs closure driver is source-agnostic over `E`: it takes it BY COMMITMENT (proven ONCE via
        /// `prove_bounded`, then reused across the seed, every round, and the fixpoint), so ONLY what `E`
        /// *is* changes for a derived edge — the semi-naive round/fixpoint machinery is byte-identical.
        /// The compiler-side digest binds `E`'s full structure, so all-pairs-`p` / all-pairs-`.` /
        /// all-pairs-`(p|q)` are non-interchangeable.
        edge: Box<RelationProofPlan>,
        /// Traversal direction over `E`. `false` walks `subject -> object` (`A_0 = (subject, object)`,
        /// round join `A_k.dst == E.subject`, extend by `E.object`); `true` walks the INVERSE edge
        /// (`A_0 = (object, subject)`, round join `A_k.dst == E.object`, extend by `E.subject`) — this
        /// is how an inverse-predicate closure is mirrored. (A DERIVED edge bakes orientation into its own
        /// structure, so the compiler emits `inverted = false` for it.)
        inverted: bool,
    },
    /// Authenticated ORDERED walk of one `rdf:List` value: given a BOUND subject node `S` and a
    /// bound list-valued predicate `P`, deliver the list's elements `E[0..K]` IN ORDER as one
    /// nonnullable NodeValue column (ReadDocument increment 1). This is a COMPOSITE ORCHESTRATION
    /// over audited operators — a bound BGP anchor, a `rdf:rest` `TransitiveClosure`, a `LeftAnti`
    /// stripping `rdf:nil`, a `permutation`, the `sort.rs` positional-successor primitive, and
    /// `rdf:rest`/`rdf:first` `InnerJoin`s reconciled by `relocation` — NO new cryptographic gadget.
    ///
    /// A stored `rdf:List` is the cons chain `t(S,P,head)`, each cell `t(Ci,rdf:first,ei)` +
    /// `t(Ci,rdf:rest,C{i+1})`, terminating `t(Ck,rdf:rest,rdf:nil)`. Cons-cell ids are FRESH random
    /// dictionary nodes, so order is the chain TOPOLOGY, not a stored key — hence the successor walk.
    /// The `rdf:first`/`rdf:rest`/`rdf:nil` IRIs are well-known constants resolved against the trusted
    /// layer dictionary at prove/verify. Runs over the single INSTANCE layer (like `TransitiveClosure`,
    /// there is no separate graph field). Fails CLOSED (never emits a bogus order) on: not exactly one
    /// `t(S,P,head)`; a non-functional `rest`/`first` (a cell with two of either); a `rest` cycle (via
    /// the closure round budget); an over-budget chain; or a nested-list element (an element that is
    /// itself a cons head — deferred to a later increment).
    OrderedListWalk {
        /// The bound list-owning subject; must be a node IRI (a literal can never own a list edge).
        subject: CanonicalObject,
        /// The bound list-valued predicate IRI bytes (the `P` of `t(S,P,head)`).
        predicate: Vec<u8>,
    },
    /// Authenticated ORDERED walk of one 1-D DENSE `sys:Array` value: given a BOUND subject node `S`
    /// and a bound array-valued predicate `P`, deliver the array's elements `Out[0..n]` in ascending
    /// `sys:index` order as one nonnullable NodeValue column (ReadDocument increment 4). This is a
    /// COMPOSITE ORCHESTRATION over audited operators — a complete bound-subject BGP for the cell set,
    /// three `sys:index`/`sys:value`/`rdf:type` functional edge-binds (`InnerJoin` + `relocation`), an
    /// `Equality` filter forcing `rdf:type == sys:Array`, an authenticated dictionary value/datatype
    /// lookup, a datatype-completeness filter, and a single `permutation` to the public identity — NO
    /// new cryptographic gadget.
    ///
    /// A stored array is a FLAT bag of cells (`t(S,P,cell)`, each `t(cell,rdf:type,sys:Array)`,
    /// `t(cell,sys:index,i)`, `t(cell,sys:value,v)`); the parent points at EVERY cell, so the cell set
    /// needs no closure/successor walk (unlike a list). Order is the authenticated numeric `sys:index`,
    /// and density (`{indices} == {0,…,n-1}`) is certified by routing the decoded positions through a
    /// permutation whose output index column the verifier reconstructs as the identity `[0,…,n-1]`. The
    /// `rdf:type`/`sys:index`/`sys:value`/`sys:Array` IRIs are well-known constants resolved against the
    /// trusted layer dictionary at prove/verify. Runs over the single INSTANCE layer (like
    /// `OrderedListWalk`, no separate graph field). Fails CLOSED (never emits a bogus order) on: a cell
    /// with 0-or->1 `sys:index`/`sys:value`/`rdf:type`; a non-`sys:Array` cell; an index of the wrong
    /// datatype; a sparse/oversized index (`≥ n`); or a gap/duplicate (the density permutation).
    OrderedArrayWalk {
        /// The bound array-owning subject; must be a node IRI (a literal can never own an array edge).
        subject: CanonicalObject,
        /// The bound array-valued predicate IRI bytes (the `P` of `t(S,P,cell)`).
        predicate: Vec<u8>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeneratedU64ComparisonOp {
    Eq,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollectionSumTemplate {
    Column(usize),
    UnsignedConstant(u64),
    SignedConstant(i64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EqualityValueDomain {
    Predicate,
    Node,
    Datatype(CanonicalDatatype),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RelationEqualityPlan {
    ColumnConstant {
        column: usize,
        domain: EqualityValueDomain,
        value: CanonicalObject,
        constant_on_left: bool,
    },
    Columns {
        left: usize,
        right: usize,
        domain: EqualityValueDomain,
    },
    Constants {
        left: CanonicalObject,
        right: CanonicalObject,
    },
}

/// One node position in a tuple-disequality edge fingerprint: a committed input column
/// of the child relation, or a public constant node IRI (a bound path endpoint),
/// resolved through the trusted layer dictionary at prove/verify time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TupleDisequalityNode {
    Column(usize),
    Constant(CanonicalObject),
}

/// One hop-pair disequality: drop the rows where stored edge `A` equals stored edge `B`. `a0`/`b0`
/// are the two hops' SUBJECT positions and `a1`/`b1` their OBJECT positions (both in the WALK
/// frame — an inverse hop swaps subject/object, the predicate is unchanged).
///
/// `predicate` selects the fingerprint arity, mirroring
/// [`crate::proof::disequality::DisequalityPair`]:
///   * `None` — the 2-column `(subject, object)` fold, for CONCRETE same-predicate finite repeats
///     (the predicate is a compile-time constant identical across the hops and cancels out).
///   * `Some((pa, pb))` — the 3-column FULL-TRIPLE `(subject, predicate, object)` fold, for hops
///     with a FREE (wildcard `.`) predicate. Each predicate position is a `Predicate`-namespace
///     COLUMN (a materialized wildcard predicate id) or a predicate-IRI CONSTANT (a concrete hop
///     paired with a wildcard one). Faithful to `path.pl`'s `make_edge(subject, predicate, object)`
///     edge identity; GENERALIZES the 2-column fold.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TupleDisequalityPairPlan {
    pub a0: TupleDisequalityNode,
    pub a1: TupleDisequalityNode,
    pub b0: TupleDisequalityNode,
    pub b1: TupleDisequalityNode,
    pub predicate: Option<(TupleDisequalityNode, TupleDisequalityNode)>,
}

pub(super) fn numeric_datatype(datatype: CanonicalDatatype) -> bool {
    matches!(
        datatype,
        CanonicalDatatype::UInt8
            | CanonicalDatatype::Int8
            | CanonicalDatatype::UInt16
            | CanonicalDatatype::Int16
            | CanonicalDatatype::UInt32
            | CanonicalDatatype::Int32
            | CanonicalDatatype::Float32
            | CanonicalDatatype::UInt64
            | CanonicalDatatype::Int64
            | CanonicalDatatype::Float64
            | CanonicalDatatype::Decimal
            | CanonicalDatatype::BigInt
            | CanonicalDatatype::PositiveInteger
            | CanonicalDatatype::NonNegativeInteger
            | CanonicalDatatype::NonPositiveInteger
            | CanonicalDatatype::NegativeInteger
    )
}

/// The fixed-width integral OP datatypes whose canonical field encoding is a total order
/// matching Prolog's integer comparison. Floats are deliberately excluded: `node_value_op_value`
/// accepts `Float32`/`Float64`, but their total-order NaN placement diverges from Prolog's IEEE
/// semantics, so any extrema derivation must reject them.
pub(super) fn integral_op_datatype(datatype: CanonicalDatatype) -> bool {
    matches!(
        datatype,
        CanonicalDatatype::UInt8
            | CanonicalDatatype::UInt16
            | CanonicalDatatype::UInt32
            | CanonicalDatatype::UInt64
            | CanonicalDatatype::Int8
            | CanonicalDatatype::Int16
            | CanonicalDatatype::Int32
            | CanonicalDatatype::Int64
    )
}

/// A Store dictionary ID namespace whose field values are guaranteed `< 2^64`. Composite
/// two-key joins pack the keys as `col0 + 2^64·col1`, which is injective only for such bounded
/// keys; a generated-decimal namespace can reach ~2^254 and would collide, so it must never be
/// packed.
pub(super) fn is_store_id_namespace(namespace: ResultIdNamespace) -> bool {
    matches!(
        namespace,
        ResultIdNamespace::NodeValue | ResultIdNamespace::Predicate
    )
}

pub(super) fn ordered_slice_bounds(source_len: usize, offset: u64, limit: Option<u64>) -> (usize, usize) {
    let start = usize::try_from(offset)
        .unwrap_or(usize::MAX)
        .min(source_len);
    let available = source_len - start;
    let take = limit
        .map(|count| usize::try_from(count).unwrap_or(usize::MAX).min(available))
        .unwrap_or(available);
    (start, start + take)
}

pub(super) fn validate_equality_plan(
    constraint: &RelationEqualityPlan,
    descriptors: &[ResultColumnDescriptor],
    ground: bool,
) -> io::Result<()> {
    let validate_column = |column: usize, domain: EqualityValueDomain| {
        if ground {
            return Err(invalid("equality column constraint cannot target a ground relation"));
        }
        let descriptor = descriptors
            .get(column)
            .ok_or_else(|| invalid("equality column out of range"))?;
        if descriptor.nullable {
            return Err(invalid("equality cannot target a nullable column"));
        }
        let namespace = match domain {
            EqualityValueDomain::Predicate => ResultIdNamespace::Predicate,
            EqualityValueDomain::Node | EqualityValueDomain::Datatype(_) => {
                ResultIdNamespace::NodeValue
            }
        };
        if descriptor.namespace != namespace {
            return Err(invalid("equality column namespace mismatch"));
        }
        Ok(())
    };
    match constraint {
        RelationEqualityPlan::ColumnConstant {
            column,
            domain,
            value,
            ..
        } => {
            validate_column(*column, *domain)?;
            match (domain, value) {
                (EqualityValueDomain::Node, CanonicalObject::Node(_)) => Ok(()),
                (
                    EqualityValueDomain::Datatype(expected),
                    CanonicalObject::Value { datatype, .. },
                ) if expected == datatype => Ok(()),
                _ => Err(invalid("equality constant does not match its exact value domain")),
            }
        }
        RelationEqualityPlan::Columns {
            left,
            right,
            domain,
        } => {
            validate_column(*left, *domain)?;
            validate_column(*right, *domain)
        }
        RelationEqualityPlan::Constants { left, right } => {
            if let (
                CanonicalObject::Value { datatype: left, .. },
                CanonicalObject::Value {
                    datatype: right, ..
                },
            ) = (left, right)
            {
                if left != right && (numeric_datatype(*left) || numeric_datatype(*right)) {
                    return Err(invalid(
                        "cross-datatype numeric Equals has coercive executor semantics",
                    ));
                }
            }
            Ok(())
        }
    }
}

/// Resolve a public (variable-blind) [`RelationEqualityPlan`] into the concrete
/// [`RelationEqualityConstraint`] the equality gadget checks, using `source`'s OWN authenticated
/// dictionary view (never the proof).
///
/// A `ColumnConstant` constant term is looked up in the layer's node/value catalog to its
/// dictionary id — an ABSENT constant resolves to a fresh out-of-range id, so it can never match a
/// stored row. `Columns` passes the two column indices through unchanged; `Constants` collapses to
/// a zero/one flag by comparing the two public constants directly. This is the verifier-side bridge
/// from the query's public constants to field-element ids, resolved against the trusted root.
pub fn resolve_relation_equality_constraint(
    source: &impl VerifierLayer,
    constraint: &RelationEqualityPlan,
) -> io::Result<RelationEqualityConstraint> {
    let layer = source.verifier_view();
    match constraint {
        RelationEqualityPlan::ColumnConstant { column, value, .. } => {
            // A constant is mapped to its id by a catalog SCAN, and an absent constant maps to an
            // impossible id. A TERM-LESS view has an empty catalog, so every constant would look
            // absent (equality under-reports, a disequality over-reports) — fail closed.
            layer.require_terms_present("equality against a constant (catalog term->id lookup)")?;
            layer.validate()?;
            let id = layer
                .node_value_catalog
                .iter()
                .find(|record| &record.triple.object == value)
                .map(|record| record.object_id)
                .unwrap_or_else(|| layer.state.node_value_count.saturating_add(1));
            Ok(RelationEqualityConstraint::ColumnConstant {
                column: *column,
                value: ArkFr::from_u64(id),
            })
        }
        RelationEqualityPlan::Columns { left, right, .. } => {
            Ok(RelationEqualityConstraint::Columns {
                left: *left,
                right: *right,
            })
        }
        RelationEqualityPlan::Constants { left, right } => {
            let equal = left == right;
            Ok(RelationEqualityConstraint::Constants {
                left: ArkFr::zero(),
                right: if equal { ArkFr::zero() } else { ArkFr::one() },
            })
        }
    }
}

/// Structural validation for a tuple-disequality plan. The child must be non-ground and
/// every referenced column must be a nonnullable node/value column (edges are compared
/// on their node/value ids); a constant node reference must be a node IRI. Rejects
/// closed everything else, mirroring `validate_equality_plan`'s discipline.
pub(super) fn validate_tuple_disequality_plan(
    pairs: &[TupleDisequalityPairPlan],
    descriptors: &[ResultColumnDescriptor],
    ground: bool,
) -> io::Result<()> {
    if ground {
        return Err(invalid("tuple disequality cannot target a ground relation"));
    }
    if pairs.is_empty() {
        return Err(invalid("tuple disequality needs at least one hop-pair"));
    }
    // Subject/object positions live in the node/value namespace; a constant must be a node IRI.
    let check_node = |node: &TupleDisequalityNode| -> io::Result<()> {
        match node {
            TupleDisequalityNode::Column(column) => {
                let descriptor = descriptors
                    .get(*column)
                    .ok_or_else(|| invalid("tuple disequality column out of range"))?;
                if descriptor.nullable {
                    return Err(invalid("tuple disequality cannot target a nullable column"));
                }
                if descriptor.namespace != ResultIdNamespace::NodeValue {
                    return Err(invalid(
                        "tuple disequality column must be a node/value namespace",
                    ));
                }
                Ok(())
            }
            TupleDisequalityNode::Constant(CanonicalObject::Node(_)) => Ok(()),
            TupleDisequalityNode::Constant(_) => {
                Err(invalid("tuple disequality constant must be a node IRI"))
            }
        }
    };
    // The full-triple PREDICATE position lives in the predicate namespace; a constant predicate is a
    // node-IRI (resolved through the predicate dictionary at prove/verify time).
    let check_predicate = |node: &TupleDisequalityNode| -> io::Result<()> {
        match node {
            TupleDisequalityNode::Column(column) => {
                let descriptor = descriptors
                    .get(*column)
                    .ok_or_else(|| invalid("tuple disequality predicate column out of range"))?;
                if descriptor.nullable {
                    return Err(invalid(
                        "tuple disequality cannot target a nullable predicate column",
                    ));
                }
                if descriptor.namespace != ResultIdNamespace::Predicate {
                    return Err(invalid(
                        "tuple disequality predicate column must be a predicate namespace",
                    ));
                }
                Ok(())
            }
            TupleDisequalityNode::Constant(CanonicalObject::Node(_)) => Ok(()),
            TupleDisequalityNode::Constant(_) => {
                Err(invalid("tuple disequality predicate constant must be a predicate IRI"))
            }
        }
    };
    for pair in pairs {
        check_node(&pair.a0)?;
        check_node(&pair.a1)?;
        check_node(&pair.b0)?;
        check_node(&pair.b1)?;
        if let Some((pa, pb)) = &pair.predicate {
            check_predicate(pa)?;
            check_predicate(pb)?;
        }
    }
    Ok(())
}

/// Resolve the plan-level hop-pairs (canonical objects) into the field-level pairs the
/// gadget consumes: each column reference is carried verbatim; each constant node IRI is
/// resolved to its `object_id` in the trusted layer dictionary (a not-present constant
/// deterministically maps to `node_value_count + 1` — an impossible id → fail-closed),
/// exactly as [`resolve_relation_equality_constraint`] resolves an equality constant.
pub fn resolve_tuple_disequality_pairs(
    source: &impl VerifierLayer,
    pairs: &[TupleDisequalityPairPlan],
) -> io::Result<Vec<DisequalityPair>> {
    let layer = source.verifier_view();
    // Constant node IRIs resolve by a catalog SCAN (absent => impossible id). A TERM-LESS view has
    // an empty catalog, so a present constant would read as absent — fail closed when any is used.
    let has_constant = pairs.iter().any(|pair| {
        let mut nodes = vec![&pair.a0, &pair.a1, &pair.b0, &pair.b1];
        if let Some((left, right)) = &pair.predicate {
            nodes.push(left);
            nodes.push(right);
        }
        nodes
            .into_iter()
            .any(|node| matches!(node, TupleDisequalityNode::Constant(_)))
    });
    if has_constant {
        layer.require_terms_present("tuple disequality against a constant (catalog term->id lookup)")?;
    }
    layer.validate()?;
    let resolve = |node: &TupleDisequalityNode| -> DisequalityNode {
        match node {
            TupleDisequalityNode::Column(column) => DisequalityNode::Column(*column),
            TupleDisequalityNode::Constant(value) => {
                let id = layer
                    .node_value_catalog
                    .iter()
                    .find(|record| &record.triple.object == value)
                    .map(|record| record.object_id)
                    .unwrap_or_else(|| layer.state.node_value_count.saturating_add(1));
                DisequalityNode::Constant(ArkFr::from_u64(id))
            }
        }
    };
    // A full-triple predicate position: a column carries verbatim; a predicate-IRI constant resolves
    // through the PREDICATE dictionary (a not-present predicate maps to `predicate_count + 1` — an
    // impossible id → fail-closed), mirroring `resolve` over the node/value dictionary.
    let resolve_predicate = |node: &TupleDisequalityNode| -> DisequalityNode {
        match node {
            TupleDisequalityNode::Column(column) => DisequalityNode::Column(*column),
            TupleDisequalityNode::Constant(CanonicalObject::Node(iri)) => {
                let id = layer
                    .predicate_catalog
                    .iter()
                    .find(|record| &record.triple.predicate == iri)
                    .map(|record| record.predicate_id)
                    .unwrap_or_else(|| layer.state.predicate_count.saturating_add(1));
                DisequalityNode::Constant(ArkFr::from_u64(id))
            }
            // `validate_tuple_disequality_plan` already rejects a non-node predicate constant; this
            // maps a would-be-invalid one to an impossible id rather than panicking (fail-closed).
            TupleDisequalityNode::Constant(_) => {
                DisequalityNode::Constant(ArkFr::from_u64(layer.state.predicate_count.saturating_add(1)))
            }
        }
    };
    Ok(pairs
        .iter()
        .map(|pair| DisequalityPair {
            a0: resolve(&pair.a0),
            a1: resolve(&pair.a1),
            b0: resolve(&pair.b0),
            b1: resolve(&pair.b1),
            predicate: pair
                .predicate
                .as_ref()
                .map(|(pa, pb)| (resolve_predicate(pa), resolve_predicate(pb))),
        })
        .collect())
}

impl RelationProofPlan {
    pub(super) fn shape_bounded(
        &self,
        depth: usize,
        nodes: &mut usize,
    ) -> io::Result<(Vec<ResultColumnDescriptor>, bool)> {
        if depth > MAX_RELATION_PROOF_DEPTH || *nodes >= MAX_RELATION_PROOF_NODES {
            return Err(invalid("recursive relation plan exceeds structural limits"));
        }
        *nodes += 1;
        match &self.kind {
            RelationProofPlanKind::Bgp(plan) => Ok((
                crate::proof::query::result_column_descriptors(plan)?,
                has_ground_result_unit(plan),
            )),
            RelationProofPlanKind::LayerTripleCount => Ok((
                vec![ResultColumnDescriptor {
                    namespace: ResultIdNamespace::GeneratedDecimalU64,
                    nullable: false,
                }],
                false,
            )),
            RelationProofPlanKind::GeneratedU64Filter { input, column, .. } => {
                let (descriptors, ground) = input.shape_bounded(depth + 1, nodes)?;
                let descriptor = descriptors
                    .get(*column)
                    .ok_or_else(|| invalid("generated-u64 filter column out of range"))?;
                if ground
                    || descriptor.nullable
                    || descriptor.namespace != ResultIdNamespace::GeneratedDecimalU64
                {
                    return Err(invalid(
                        "generated-u64 filter requires a nonnullable generated decimal column",
                    ));
                }
                Ok((descriptors, false))
            }
            RelationProofPlanKind::GeneratedI64Filter { input, column, .. } => {
                let (descriptors, ground) = input.shape_bounded(depth + 1, nodes)?;
                let descriptor = descriptors
                    .get(*column)
                    .ok_or_else(|| invalid("generated-i64 filter column out of range"))?;
                if ground
                    || descriptor.nullable
                    || descriptor.namespace != ResultIdNamespace::GeneratedDecimalI64
                {
                    return Err(invalid(
                        "generated-i64 filter requires a nonnullable signed generated decimal column",
                    ));
                }
                Ok((descriptors, false))
            }
            RelationProofPlanKind::GroupSum {
                input,
                key_columns,
                value_column,
                datatype,
            } => {
                let (descriptors, ground) = input.shape_bounded(depth + 1, nodes)?;
                if ground
                    || key_columns.len() > 2
                    || key_columns.iter().any(|key| *key == *value_column)
                    || (key_columns.len() == 2 && key_columns[0] == key_columns[1])
                {
                    return Err(invalid("group-sum requires zero to two unique key columns distinct from its value column"));
                }
                let value = descriptors
                    .get(*value_column)
                    .copied()
                    .ok_or_else(|| invalid("group-sum value column out of range"))?;
                if value.nullable
                    || value.namespace != ResultIdNamespace::NodeValue
                    || !matches!(
                        datatype,
                        CanonicalDatatype::UInt8
                            | CanonicalDatatype::UInt16
                            | CanonicalDatatype::UInt32
                            | CanonicalDatatype::UInt64
                            | CanonicalDatatype::Int8
                            | CanonicalDatatype::Int16
                            | CanonicalDatatype::Int32
                            | CanonicalDatatype::Int64
                    )
                {
                    return Err(invalid(
                        "group-sum requires an optional nonnullable node key and a fixed-width integral value",
                    ));
                }
                let generated = ResultColumnDescriptor {
                    namespace: if matches!(
                        datatype,
                        CanonicalDatatype::Int8
                            | CanonicalDatatype::Int16
                            | CanonicalDatatype::Int32
                            | CanonicalDatatype::Int64
                    ) {
                        ResultIdNamespace::GeneratedDecimalI64
                    } else {
                        ResultIdNamespace::GeneratedDecimalU64
                    },
                    nullable: false,
                };
                let mut result = key_columns
                    .iter()
                    .map(|key_column| {
                        let key = descriptors
                            .get(*key_column)
                            .copied()
                            .ok_or_else(|| invalid("group-sum key column out of range"))?;
                        if key.nullable || key.namespace != ResultIdNamespace::NodeValue {
                            return Err(invalid("group-sum keys must be nonnullable nodes"));
                        }
                        Ok(key)
                    })
                    .collect::<io::Result<Vec<_>>>()?;
                result.push(generated);
                Ok((result, false))
            }
            RelationProofPlanKind::GroupCount { input, key_columns } => {
                let (descriptors, ground) = input.shape_bounded(depth + 1, nodes)?;
                if ground
                    || key_columns.len() > 2
                    || (key_columns.len() == 2 && key_columns[0] == key_columns[1])
                {
                    return Err(invalid("group-count requires zero to two unique key columns"));
                }
                let mut result = key_columns
                    .iter()
                    .map(|key_column| {
                        let key = descriptors
                            .get(*key_column)
                            .copied()
                            .ok_or_else(|| invalid("group-count key column out of range"))?;
                        if key.nullable || key.namespace != ResultIdNamespace::NodeValue {
                            return Err(invalid("group-count keys must be nonnullable nodes"));
                        }
                        Ok(key)
                    })
                    .collect::<io::Result<Vec<_>>>()?;
                result.push(ResultColumnDescriptor {
                    namespace: ResultIdNamespace::GeneratedDecimalU64,
                    nullable: false,
                });
                Ok((result, false))
            }
            RelationProofPlanKind::CollectionCount { input } => {
                input.shape_bounded(depth + 1, nodes)?;
                Ok((
                    vec![ResultColumnDescriptor {
                        namespace: ResultIdNamespace::GeneratedDecimalU64,
                        nullable: false,
                    }],
                    false,
                ))
            }
            RelationProofPlanKind::CollectionSum {
                input,
                template,
                datatype,
            } => {
                let (descriptors, _) = input.shape_bounded(depth + 1, nodes)?;
                let signed = matches!(
                    datatype,
                    CanonicalDatatype::Int8
                        | CanonicalDatatype::Int16
                        | CanonicalDatatype::Int32
                        | CanonicalDatatype::Int64
                );
                let unsigned = matches!(
                    datatype,
                    CanonicalDatatype::UInt8
                        | CanonicalDatatype::UInt16
                        | CanonicalDatatype::UInt32
                        | CanonicalDatatype::UInt64
                );
                if !signed && !unsigned {
                    return Err(invalid("collection-sum datatype is outside the integral subset"));
                }
                match template {
                    CollectionSumTemplate::Column(column) => {
                        let value = descriptors
                            .get(*column)
                            .copied()
                            .ok_or_else(|| invalid("collection-sum value column out of range"))?;
                        if value.nullable || value.namespace != ResultIdNamespace::NodeValue {
                            return Err(invalid(
                                "collection-sum column must be a nonnullable Store value",
                            ));
                        }
                    }
                    CollectionSumTemplate::UnsignedConstant(value)
                        if unsigned
                            && match datatype {
                                CanonicalDatatype::UInt8 => *value <= u64::from(u8::MAX),
                                CanonicalDatatype::UInt16 => *value <= u64::from(u16::MAX),
                                CanonicalDatatype::UInt32 => *value <= u64::from(u32::MAX),
                                CanonicalDatatype::UInt64 => true,
                                _ => false,
                            } => {}
                    CollectionSumTemplate::SignedConstant(value)
                        if signed
                            && match datatype {
                                CanonicalDatatype::Int8 => i8::try_from(*value).is_ok(),
                                CanonicalDatatype::Int16 => i16::try_from(*value).is_ok(),
                                CanonicalDatatype::Int32 => i32::try_from(*value).is_ok(),
                                CanonicalDatatype::Int64 => true,
                                _ => false,
                            } => {}
                    _ => return Err(invalid("collection-sum template/domain mismatch")),
                }
                Ok((
                    vec![ResultColumnDescriptor {
                        namespace: if signed {
                            ResultIdNamespace::GeneratedDecimalI64
                        } else {
                            ResultIdNamespace::GeneratedDecimalU64
                        },
                        nullable: false,
                    }],
                    false,
                ))
            }
            RelationProofPlanKind::CollectionExtrema {
                input,
                value_column,
                datatype,
                ..
            } => {
                let (descriptors, ground) = input.shape_bounded(depth + 1, nodes)?;
                let value = descriptors
                    .get(*value_column)
                    .copied()
                    .ok_or_else(|| invalid("collection-extrema value column out of range"))?;
                if ground
                    || value.nullable
                    || value.namespace != ResultIdNamespace::NodeValue
                    || !matches!(
                        datatype,
                        CanonicalDatatype::UInt8
                            | CanonicalDatatype::UInt16
                            | CanonicalDatatype::UInt32
                            | CanonicalDatatype::UInt64
                            | CanonicalDatatype::Int8
                            | CanonicalDatatype::Int16
                            | CanonicalDatatype::Int32
                            | CanonicalDatatype::Int64
                    )
                {
                    return Err(invalid(
                        "collection-extrema requires one exact nonnullable integral Store value",
                    ));
                }
                Ok((vec![value], false))
            }
            RelationProofPlanKind::BagUnion { branches, mappings } => {
                if branches.len() < 2
                    || branches.len() != mappings.len()
                    || branches.len() > MAX_RELATION_PROOF_NODES - *nodes
                {
                    return Err(invalid("recursive bag-union plan branch mismatch"));
                }
                let mut expected: Option<(Vec<ResultColumnDescriptor>, bool)> = None;
                for (branch, mapping) in branches.iter().zip(mappings) {
                    let (descriptors, ground) = branch.shape_bounded(depth + 1, nodes)?;
                    let mapped = if ground {
                        if !mapping.is_empty() {
                            return Err(invalid("ground union branch mapping must be empty"));
                        }
                        Vec::new()
                    } else {
                        mapping
                            .iter()
                            .map(|index| {
                                descriptors.get(*index).copied().ok_or_else(|| {
                                    invalid("recursive union mapping out of range")
                                })
                            })
                            .collect::<io::Result<Vec<_>>>()?
                    };
                    if let Some((prior, prior_ground)) = &expected {
                        if *prior_ground != ground || prior != &mapped {
                            return Err(invalid("recursive union branch schema mismatch"));
                        }
                    } else {
                        expected = Some((mapped, ground));
                    }
                }
                Ok(expected.expect("at least two branches"))
            }
            RelationProofPlanKind::BagProjection { input, indices } => {
                let (descriptors, ground) = input.shape_bounded(depth + 1, nodes)?;
                let mut seen = Vec::with_capacity(indices.len());
                if ground {
                    return Err(invalid("bag projection requires a non-ground child"));
                }
                if indices.is_empty() {
                    return Ok((Vec::new(), true));
                }
                let projected = indices
                    .iter()
                    .map(|index| {
                        if seen.contains(index) {
                            return Err(invalid("bag projection indices must be unique"));
                        }
                        seen.push(*index);
                        descriptors
                            .get(*index)
                            .copied()
                            .ok_or_else(|| invalid("bag projection index out of range"))
                    })
                    .collect::<io::Result<Vec<_>>>()?;
                Ok((projected, false))
            }
            RelationProofPlanKind::Distinct { input, key_columns } => {
                let (descriptors, ground) = input.shape_bounded(depth + 1, nodes)?;
                if ground || !(1..=2).contains(&key_columns.len()) {
                    return Err(invalid("distinct requires one or two non-ground key columns"));
                }
                let mut seen = Vec::with_capacity(key_columns.len());
                let projected = key_columns
                    .iter()
                    .map(|index| {
                        if seen.contains(index) {
                            return Err(invalid("distinct key columns must be unique"));
                        }
                        seen.push(*index);
                        let descriptor = descriptors
                            .get(*index)
                            .copied()
                            .ok_or_else(|| invalid("distinct key column out of range"))?;
                        if descriptor.nullable {
                            return Err(invalid("distinct key columns must be nonnullable"));
                        }
                        Ok(descriptor)
                    })
                    .collect::<io::Result<Vec<_>>>()?;
                Ok((projected, false))
            }
            RelationProofPlanKind::LeftAnti { left, right, key_pairs } => {
                let left_shape = left.shape_bounded(depth + 1, nodes)?;
                let (right_descriptors, right_ground) =
                    right.shape_bounded(depth + 1, nodes)?;
                if key_pairs.is_empty() {
                    return Ok(left_shape);
                }
                if left_shape.1 || right_ground || !(1..=2).contains(&key_pairs.len()) {
                    return Err(invalid("invalid correlated recursive anti-join shape"));
                }
                for (left_index, right_index) in key_pairs {
                    let l = left_shape.0.get(*left_index).ok_or_else(|| invalid("anti left key out of range"))?;
                    let r = right_descriptors.get(*right_index).ok_or_else(|| invalid("anti right key out of range"))?;
                    if l.namespace != r.namespace || l.nullable || r.nullable {
                        return Err(invalid("anti correlation key descriptor mismatch"));
                    }
                    // A composite (two-key) anti-join packs both keys as `col0 + 2^64·col1`,
                    // injective only for Store dictionary IDs (`< 2^64`). Namespace equality
                    // does not bound the field width, so require every packed key to be a
                    // Store-ID namespace; scalar keys use injective equality and are exempt.
                    if key_pairs.len() == 2
                        && (!is_store_id_namespace(l.namespace) || !is_store_id_namespace(r.namespace))
                    {
                        return Err(invalid("composite anti-join key must be a Store-ID namespace"));
                    }
                }
                Ok(left_shape)
            }
            RelationProofPlanKind::InnerJoin { left, right, key_pairs } => {
                let (mut descriptors, left_ground) = left.shape_bounded(depth + 1, nodes)?;
                let (right_descriptors, right_ground) = right.shape_bounded(depth + 1, nodes)?;
                if left_ground || right_ground || !(1..=2).contains(&key_pairs.len()) {
                    return Err(invalid("invalid recursive inner-join child shape"));
                }
                let mut shared_left = Vec::new();
                let mut shared_right = Vec::new();
                for (left_index, right_index) in key_pairs {
                    let l = descriptors.get(*left_index).ok_or_else(|| invalid("inner left key out of range"))?;
                    let r = right_descriptors.get(*right_index).ok_or_else(|| invalid("inner right key out of range"))?;
                    if l.namespace != r.namespace || l.nullable || r.nullable
                        || shared_left.contains(left_index) || shared_right.contains(right_index) {
                        return Err(invalid("inner correlation key descriptor mismatch"));
                    }
                    // Composite (two-key) inner joins pack `col0 + 2^64·col1`, injective only
                    // for Store dictionary IDs (`< 2^64`); reject any non-Store-ID packed key.
                    if key_pairs.len() == 2
                        && (!is_store_id_namespace(l.namespace) || !is_store_id_namespace(r.namespace))
                    {
                        return Err(invalid("composite inner-join key must be a Store-ID namespace"));
                    }
                    shared_left.push(*left_index); shared_right.push(*right_index);
                }
                descriptors.extend(right_descriptors.into_iter().enumerate().filter_map(
                    |(index, descriptor)| (!shared_right.contains(&index)).then_some(descriptor)
                ));
                Ok((descriptors, false))
            }
            RelationProofPlanKind::CrossRootHashJoin { left, right, left_key_col, right_key_col } => {
                let (left_descriptors, left_ground) = left.shape_bounded(depth + 1, nodes)?;
                let (right_descriptors, right_ground) = right.shape_bounded(depth + 1, nodes)?;
                if left_ground || right_ground {
                    return Err(invalid("cross-root hash join child is ground"));
                }
                // The node is specialized: `left` is the single-column class set (all key), so it
                // contributes NOTHING to the output — only its membership matters.
                if left_descriptors.len() != 1 || *left_key_col != 0 {
                    return Err(invalid(
                        "cross-root hash join left child must be a single key column (the class set)",
                    ));
                }
                let lk = left_descriptors
                    .first()
                    .ok_or_else(|| invalid("cross-root left key out of range"))?;
                let rk = right_descriptors
                    .get(*right_key_col)
                    .ok_or_else(|| invalid("cross-root right key out of range"))?;
                // Both keys resolve to IRIs whose (dictionary-independent) hashes we join on, so both
                // must be nonnullable node/value id columns. Their raw ids live in SEPARATE dicts and
                // are NEVER compared directly — only their authenticated hashes are joined.
                if lk.nullable
                    || rk.nullable
                    || lk.namespace != ResultIdNamespace::NodeValue
                    || rk.namespace != ResultIdNamespace::NodeValue
                {
                    return Err(invalid(
                        "cross-root hash join keys must be nonnullable node/value (IRI) columns",
                    ));
                }
                // Output = the instance-side (`right`) columns minus the class key column.
                let descriptors: Vec<ResultColumnDescriptor> = right_descriptors
                    .into_iter()
                    .enumerate()
                    .filter_map(|(index, descriptor)| (index != *right_key_col).then_some(descriptor))
                    .collect();
                if descriptors.is_empty() {
                    return Err(invalid("cross-root hash join delivers no output column"));
                }
                Ok((descriptors, false))
            }
            RelationProofPlanKind::LeftOuter { left, right, key_pairs } => {
                let (mut descriptors, left_ground) = left.shape_bounded(depth + 1, nodes)?;
                let (right_descriptors, right_ground) = right.shape_bounded(depth + 1, nodes)?;
                if left_ground || right_ground || key_pairs.len() > 2 {
                    return Err(invalid("invalid recursive left-outer child shape"));
                }
                let mut shared_right = Vec::new();
                for (left_index, right_index) in key_pairs {
                    let l = descriptors.get(*left_index).ok_or_else(|| invalid("outer left key out of range"))?;
                    let r = right_descriptors.get(*right_index).ok_or_else(|| invalid("outer right key out of range"))?;
                    if l.namespace != r.namespace
                        || l.nullable
                        || r.nullable
                        || shared_right.contains(right_index)
                    {
                        return Err(invalid("outer correlation key descriptor mismatch"));
                    }
                    // Composite (two-key) left-outer joins pack `col0 + 2^64·col1`, injective
                    // only for Store dictionary IDs (`< 2^64`); reject any non-Store-ID packed key.
                    if key_pairs.len() == 2
                        && (!is_store_id_namespace(l.namespace) || !is_store_id_namespace(r.namespace))
                    {
                        return Err(invalid("composite left-outer key must be a Store-ID namespace"));
                    }
                    shared_right.push(*right_index);
                }
                descriptors.extend(right_descriptors.into_iter().enumerate().filter_map(
                    |(index, descriptor)| {
                        (!shared_right.contains(&index)).then_some(ResultColumnDescriptor {
                            namespace: descriptor.namespace,
                            nullable: true,
                        })
                    },
                ));
                Ok((descriptors, false))
            }
            RelationProofPlanKind::Equality { input, constraint } => {
                let (descriptors, ground) = input.shape_bounded(depth + 1, nodes)?;
                validate_equality_plan(constraint, &descriptors, ground)?;
                Ok((descriptors, ground))
            }
            RelationProofPlanKind::ExactTupleDisequality { input, pairs } => {
                let (descriptors, ground) = input.shape_bounded(depth + 1, nodes)?;
                validate_tuple_disequality_plan(pairs, &descriptors, ground)?;
                // A pure row filter: schema (and non-ground-ness) pass through unchanged.
                Ok((descriptors, ground))
            }
            RelationProofPlanKind::ExactScalarOrder { input, datatype, .. } => {
                let (descriptors, ground) = input.shape_bounded(depth + 1, nodes)?;
                if ground
                    || descriptors.len() != 1
                    || descriptors[0].namespace != ResultIdNamespace::NodeValue
                    || descriptors[0].nullable
                    || !numeric_datatype(*datatype)
                    || matches!(
                        datatype,
                        CanonicalDatatype::Decimal
                            | CanonicalDatatype::BigInt
                            | CanonicalDatatype::PositiveInteger
                            | CanonicalDatatype::NonNegativeInteger
                            | CanonicalDatatype::NonPositiveInteger
                            | CanonicalDatatype::NegativeInteger
                            // Float32/64: IEEE total-order OP key diverges from WOQL numeric order
                            // (−0.0/NaN); float sort keys reject-close (matches `supported_datatype`).
                            | CanonicalDatatype::Float32
                            | CanonicalDatatype::Float64
                    )
                {
                    return Err(invalid("invalid exact scalar order child shape"));
                }
                Ok((descriptors, false))
            }
            RelationProofPlanKind::ExactMultiOrder { input, datatypes, key_indices, directions } => {
                let (descriptors, ground) = input.shape_bounded(depth + 1, nodes)?;
                let mut sorted = key_indices.clone(); sorted.sort_unstable();
                if ground || !(2..=3).contains(&descriptors.len()) || datatypes.len() != descriptors.len() || directions.len() != descriptors.len() || sorted != (0..descriptors.len()).collect::<Vec<_>>()
                    || descriptors.iter().any(|d| d.nullable || d.namespace != ResultIdNamespace::NodeValue) {
                    return Err(invalid("invalid exact multikey order child shape"));
                }
                Ok((descriptors, false))
            }
            RelationProofPlanKind::ExactOrderedSlice { input, .. } => {
                if !matches!(
                    input.kind,
                    RelationProofPlanKind::ExactScalarOrder { .. }
                        | RelationProofPlanKind::ExactMultiOrder { .. }
                        | RelationProofPlanKind::ExactOrderedSlice { .. }
                ) {
                    return Err(invalid("ordered slice requires an authenticated ordered child"));
                }
                let (descriptors, ground) = input.shape_bounded(depth + 1, nodes)?;
                if ground || descriptors.is_empty() {
                    return Err(invalid("ordered slice requires a non-ground ordered relation"));
                }
                Ok((descriptors, false))
            }
            RelationProofPlanKind::GroupExtrema { input, key_columns, value_column, datatype, .. } => {
                let (descriptors, ground) = input.shape_bounded(depth + 1, nodes)?;
                if ground
                    || key_columns.len() > 2
                    || key_columns.iter().any(|key| *key == *value_column)
                    || (key_columns.len() == 2 && key_columns[0] == key_columns[1])
                {
                    return Err(invalid("invalid group-extrema child shape"));
                }
                let value = descriptors.get(*value_column).copied().ok_or_else(|| invalid("group-extrema value column out of range"))?;
                if value.nullable || value.namespace != ResultIdNamespace::NodeValue {
                    return Err(invalid("group-extrema requires a nonnullable Store value"));
                }
                if !matches!(datatype,
                    CanonicalDatatype::UInt8 | CanonicalDatatype::UInt16
                    | CanonicalDatatype::UInt32 | CanonicalDatatype::UInt64
                    | CanonicalDatatype::Int8 | CanonicalDatatype::Int16
                    | CanonicalDatatype::Int32 | CanonicalDatatype::Int64)
                {
                    return Err(invalid("group-extrema datatype is outside the exact integral subset"));
                }
                let mut result = key_columns.iter().map(|key_column| {
                    let key = descriptors.get(*key_column).copied()
                        .ok_or_else(|| invalid("group-extrema key column out of range"))?;
                    if key.nullable || key.namespace != ResultIdNamespace::NodeValue {
                        return Err(invalid("group-extrema keys must be nonnullable nodes"));
                    }
                    Ok(key)
                }).collect::<io::Result<Vec<_>>>()?;
                result.push(ResultColumnDescriptor {
                    namespace: ResultIdNamespace::NodeValue,
                    nullable: false,
                });
                Ok((result, false))
            }
            RelationProofPlanKind::TransitiveClosure { edge, seed, .. } => {
                // The seed is a bound NODE endpoint; a literal/value seed can never be a graph
                // subject and would never join, so reject it closed rather than deliver an empty
                // set under a mislabeled identity.
                if !matches!(seed, CanonicalObject::Node(_)) {
                    return Err(invalid("transitive closure seed must be a node IRI"));
                }
                // The edge relation must deliver exactly a 2-column [start, end] node/value relation so
                // that (i) `end == start` joins are namespace-compatible across rounds and (ii) the
                // delivered reachable set is a single node column. A single-predicate edge is a
                // `Bgp(Triple(?s, p, ?o))`; a compound edge is a deduped bag-union / projected join of
                // node triples — either way its `result_shape` must be two nonnullable NodeValue
                // columns. Anything else (a ground unit, a generated column, a wrong arity) rejects
                // closed here — the shape is K-INDEPENDENT, so this validates identically for the
                // prover and the verifier, and it recursively validates the whole edge subtree.
                let (edge_descriptors, edge_ground) = edge.result_shape()?;
                if edge_ground
                    || edge_descriptors.len() != 2
                    || edge_descriptors.iter().any(|descriptor| {
                        descriptor.nullable
                            || descriptor.namespace != ResultIdNamespace::NodeValue
                    })
                {
                    return Err(invalid(
                        "transitive closure edge must be a two-column node/value triple relation",
                    ));
                }
                // The delivered relation is the reachable SET: one nonnullable NodeValue column.
                Ok((
                    vec![ResultColumnDescriptor {
                        namespace: ResultIdNamespace::NodeValue,
                        nullable: false,
                    }],
                    false,
                ))
            }
            RelationProofPlanKind::TransitiveClosurePairs { edge, .. } => {
                // The edge relation must deliver exactly a 2-column [start, end] node/value relation (same
                // K-INDEPENDENT shape guard as the single-seed closure), so `A_k.dst == E.<start>` joins
                // are namespace-compatible across rounds and the delivered pairs are two node columns. A
                // single-predicate edge is a `Bgp(Triple(?s, p, ?o))`; a DERIVED all-pairs edge (wildcard
                // `.` / compound) is a projected/deduped node relation — either way its `result_shape` must
                // be two nonnullable NodeValue columns. Anything else (a ground unit, a generated column, a
                // wrong arity) rejects closed here — the shape is K-INDEPENDENT, so this validates
                // identically for prover and verifier, and it recursively validates the whole edge subtree.
                let (edge_descriptors, edge_ground) = edge.result_shape()?;
                if edge_ground
                    || edge_descriptors.len() != 2
                    || edge_descriptors.iter().any(|descriptor| {
                        descriptor.nullable
                            || descriptor.namespace != ResultIdNamespace::NodeValue
                    })
                {
                    return Err(invalid(
                        "all-pairs closure edge must be a two-column node/value triple relation",
                    ));
                }
                // The delivered relation is the 2-column reachability relation: two nonnullable
                // NodeValue columns [src, dst].
                Ok((
                    vec![
                        ResultColumnDescriptor {
                            namespace: ResultIdNamespace::NodeValue,
                            nullable: false,
                        },
                        ResultColumnDescriptor {
                            namespace: ResultIdNamespace::NodeValue,
                            nullable: false,
                        },
                    ],
                    false,
                ))
            }
            RelationProofPlanKind::OrderedListWalk { subject, predicate } => {
                // Structural, K-INDEPENDENT validation (identical for prover and verifier): the
                // subject must be a node (a literal can never own a list edge) and the predicate
                // IRI must be nonempty. The number of cons cells `K` lives in the PROOF, never here.
                if !matches!(subject, CanonicalObject::Node(_)) {
                    return Err(invalid("ordered-list-walk subject must be a node IRI"));
                }
                if predicate.is_empty() {
                    return Err(invalid("ordered-list-walk predicate IRI must be nonempty"));
                }
                // The delivered relation is the ordered element sequence: one nonnullable NodeValue
                // column (`domains: [None]`).
                Ok((
                    vec![ResultColumnDescriptor {
                        namespace: ResultIdNamespace::NodeValue,
                        nullable: false,
                    }],
                    false,
                ))
            }
            RelationProofPlanKind::OrderedArrayWalk { subject, predicate } => {
                // Structural, N-INDEPENDENT validation (identical for prover and verifier): the subject
                // must be a node (a literal can never own an array edge) and the predicate IRI must be
                // nonempty. The number of cells `n` lives in the PROOF, never here.
                if !matches!(subject, CanonicalObject::Node(_)) {
                    return Err(invalid("ordered-array-walk subject must be a node IRI"));
                }
                if predicate.is_empty() {
                    return Err(invalid("ordered-array-walk predicate IRI must be nonempty"));
                }
                // The delivered relation is the ordered value sequence: one nonnullable NodeValue column.
                Ok((
                    vec![ResultColumnDescriptor {
                        namespace: ResultIdNamespace::NodeValue,
                        nullable: false,
                    }],
                    false,
                ))
            }
        }
    }

    /// Derive this plan's output schema — the per-column [`ResultColumnDescriptor`]s and whether the
    /// result is ground (a single unit row, `true`) — by a depth- and node-bounded walk of the
    /// operator tree. The verifier calls this to learn the shape the delivered rows MUST have before
    /// checking any proof node; a plan deeper than [`MAX_RELATION_PROOF_DEPTH`] or with more than
    /// [`MAX_RELATION_PROOF_NODES`] nodes rejects closed.
    pub fn result_shape(&self) -> io::Result<(Vec<ResultColumnDescriptor>, bool)> {
        let mut nodes = 0;
        self.shape_bounded(1, &mut nodes)
    }
}
