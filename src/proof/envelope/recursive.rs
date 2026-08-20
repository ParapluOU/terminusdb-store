//! The general recursive prover/verifier driver and its transitive-closure orchestration.
//!
//! [`RecursiveRelationEnvelope`] is the full operator-tree proof. It pairs a public
//! [`RelationProofPlan`](super::plan::RelationProofPlan) (the tree the verifier reconstructs
//! from its own trusted query) with a [`RecursiveRelationProof`] whose variants mirror the plan
//! kinds one-for-one and hold the per-node proof payloads. A node's children are themselves
//! complete authenticated relations, so composition never has to re-interpret a child proof as
//! a raw leaf read.
//!
//! * `prove_bounded` / `verify_bounded` are the two mirrored recursive walks. They dispatch on
//!   the plan kind, verify each child, then run that operator's audited gadget (from the sibling
//!   `join2` / `distinct` / `group` / `order` / `anti` / `union` / `semijoin` proof modules)
//!   against the children's authenticated commitments. Both are depth- and node-bounded so an
//!   untrusted tree cannot exhaust the verifier.
//! * `put_bounded` / `get_bounded` are the matching manual (de)serialization of the plan tree
//!   and proof tree; `encode` / `decode` add the framing.
//! * `prove_cross_root` / `verify_cross_root` are the ONLY two-layer entries — the cross-root
//!   hash join reads an instance graph and a schema graph at once, binding both roots to one
//!   database root.
//!
//! # Bounded transitive closure (`star` / `plus`)
//!
//! The transitive-closure node is a COMPOSITE orchestration over the audited
//! InnerJoin / BagUnion / Distinct / LeftAnti operators — it introduces no new cryptographic
//! gadget. It walks a single edge relation semi-naively: round `k+1` joins the current
//! accumulator against the edge relation, unions the new endpoints in, and dedupes. The number
//! of rounds `K` lives in the PROOF (not the plan), and soundness rests on the empty-delta
//! FIXPOINT certificate — a left-anti-join proving the accumulator has no un-seen one-hop
//! neighbour left — which is exactly what lets a cyclic graph terminate. The `closure_*`
//! helpers below thread each round by commitment (never by nesting), which keeps the unroll
//! `O(K)` rather than `O(2^K)`, each step on its own domain-separated transcript.
#![allow(unused_imports)]

use super::columns::*;
use super::layer_bgp::LayerBgpEnvelope;
use super::plan::*;
use super::wire::*;
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
use crate::proof::union::{output_log_domain, verify_bag_union, BagUnionProof};
use crate::proof::{CanonicalDatatype, CanonicalObject, ProofHash};
#[derive(Clone, Debug)]
pub struct RecursiveRelationEnvelope {
    pub layer_root: ProofHash,
    pub query_digest: ProofHash,
    pub proof: RecursiveRelationProof,
    pub result_commitments: Vec<ArkGT>,
    pub result_columns: Vec<Vec<ArkFr>>,
    pub result_log_domain: usize,
    pub result_len: usize,
    pub ground_result: bool,
    pub result_descriptors: Vec<ResultColumnDescriptor>,
    pub result_values: Vec<Vec<CanonicalResultValue>>,
}

#[derive(Clone, Debug)]
pub enum RecursiveRelationProof {
    Bgp(Box<LayerBgpEnvelope>),
    LayerTripleCount,
    GeneratedU64Filter {
        input: Box<RecursiveRelationEnvelope>,
    },
    GeneratedI64Filter {
        input: Box<RecursiveRelationEnvelope>,
    },
    GroupSum {
        input: Box<RecursiveRelationEnvelope>,
    },
    GroupCount {
        input: Box<RecursiveRelationEnvelope>,
    },
    CollectionCount {
        input: Box<RecursiveRelationEnvelope>,
    },
    CollectionSum {
        input: Box<RecursiveRelationEnvelope>,
    },
    CollectionExtrema {
        input: Box<RecursiveRelationEnvelope>,
    },
    BagUnion {
        branches: Vec<RecursiveRelationEnvelope>,
        proof: Box<BagUnionProof>,
    },
    BagProjection {
        input: Box<RecursiveRelationEnvelope>,
    },
    Distinct {
        input: Box<RecursiveRelationEnvelope>,
        proof: Box<RelationDistinctProof>,
    },
    LeftAnti {
        left: Box<RecursiveRelationEnvelope>,
        right: Box<RecursiveRelationEnvelope>,
        proof: Option<Box<LeftAntiJoinProof>>,
    },
    InnerJoin {
        left: Box<RecursiveRelationEnvelope>,
        right: Box<RecursiveRelationEnvelope>,
        proof: Box<InnerJoinProof>,
    },
    /// The two-root cross-dictionary hash join proof. `left`/`right` are the two single-root child
    /// subtrees (schema-rooted and instance-rooted); `left_hash`/`right_hash` authenticate each
    /// side's class-id column to its true `hash_object(IRI)` column against ITS OWN root; the three
    /// relocations lift the closure hash SET, the `?x` column and the class-hash column to one working
    /// domain; `semijoin` is the full-field LogUp SEMI-JOIN whose kept `?x` column is the output.
    CrossRootHashJoin {
        left: Box<RecursiveRelationEnvelope>,
        right: Box<RecursiveRelationEnvelope>,
        left_hash: Box<DictHashLookupProof>,
        right_hash: Box<DictHashLookupProof>,
        /// The three authenticated columns relocated UP to a shared working domain `2^D`: the schema
        /// closure hash SET, the instance `?x` column, and the instance class-hash column.
        relocate_closure: RelocationProof,
        com_closure_at_d: ArkGT,
        relocate_x: RelocationProof,
        com_x_at_d: ArkGT,
        relocate_h: RelocationProof,
        com_h_at_d: ArkGT,
        /// The full-field LogUp SEMI-JOIN (no magnitude sort): keeps instance rows whose class-hash is
        /// a member of the closure hash set. `semijoin.com_x_kept` is the delivered `?x` output.
        semijoin: Box<HashSemiJoinProof>,
    },
    LeftOuter {
        left: Box<RecursiveRelationEnvelope>,
        right: Box<RecursiveRelationEnvelope>,
        proof: Box<LeftOuterJoinProof>,
    },
    Equality {
        input: Box<RecursiveRelationEnvelope>,
        proof: Box<RelationEqualityProof>,
    },
    TupleDisequality {
        input: Box<RecursiveRelationEnvelope>,
        proof: Box<TupleDisequalityProof>,
    },
    ExactScalarOrder {
        input: Box<RecursiveRelationEnvelope>,
        proof: Box<ExactScalarOrderProof>,
    },
    ExactMultiOrder {
        input: Box<RecursiveRelationEnvelope>,
        proof: Box<ExactMultiOrderProof>,
    },
    ExactOrderedSlice {
        input: Box<RecursiveRelationEnvelope>,
    },
    GroupExtrema {
        input: Box<RecursiveRelationEnvelope>,
    },
    /// Bounded semi-naive transitive closure. `edge` is the ONE authenticated edge relation `E`
    /// (proven once, reused by commitment). `seed` establishes `A_0`. `rounds` are the `K`
    /// accumulate-and-dedupe steps (each a well-formed InnerJoin + BagUnion + Distinct against the
    /// SAME `E`). `fixpoint` is the empty-delta anti-join proving `A_K` is closed. The delivered
    /// result of THIS envelope is `A_K` = the reachable set. See [`RelationProofPlanKind::TransitiveClosure`].
    TransitiveClosure {
        edge: Box<RecursiveRelationEnvelope>,
        seed: TransitiveClosureSeed,
        rounds: Vec<TransitiveClosureRound>,
        fixpoint: TransitiveClosureFixpoint,
    },
    /// Bounded semi-naive ALL-PAIRS transitive closure (`path(?x, plus(p), ?y)`, both endpoints
    /// free). Same composite discipline as [`Self::TransitiveClosure`], but the accumulator is a
    /// 2-column pairs relation `A ⊆ (src, dst)` rather than a 1-column reachable set: `edge` is the
    /// ONE authenticated edge relation `E` (proven once, reused by commitment), `seed` establishes
    /// `A_0 = Distinct([src,dst], π(E))` (the direction-projected one-hop pairs), `rounds` extend each
    /// known pair by one edge and dedup on the FULL 2-tuple, and `fixpoint` is the empty-delta
    /// anti-join over BOTH columns proving `A_K` is closed under right-extension. The delivered result
    /// of THIS envelope is the 2-column pairs relation `A_K`. PLUS-ONLY — there is no reflexive/star
    /// variant here (see `plan_transitive_closure_path`: all-pairs `star` is reject-closed because
    /// `path.pl`'s both-free identity branch `X = Y` is a single UNBOUND row, not a domain diagonal).
    /// See [`RelationProofPlanKind::TransitiveClosurePairs`].
    TransitiveClosurePairs {
        edge: Box<RecursiveRelationEnvelope>,
        seed: TransitiveClosurePairsSeed,
        rounds: Vec<TransitiveClosurePairsRound>,
        fixpoint: TransitiveClosurePairsFixpoint,
    },
    /// Authenticated ordered `rdf:List` walk (ReadDocument increment 1). All sub-proofs — the four
    /// authenticated child relations (anchor, closure, `rdf:rest` edges, `rdf:first` edges) and the
    /// gluing gadgets (strip-nil anti-join, walk permutation, rest/first inner-joins + relocations,
    /// the positional successor) — live in [`super::list_walk::OrderedListWalkProof`]. The delivered
    /// result of THIS envelope is the ordered element column. See
    /// [`RelationProofPlanKind::OrderedListWalk`].
    OrderedListWalk(Box<super::list_walk::OrderedListWalkProof>),
    /// Authenticated ordered 1-D dense `sys:Array` walk (ReadDocument increment 4). All sub-proofs —
    /// the four authenticated child relations (cell set, `sys:index`/`sys:value`/`rdf:type` edges) and
    /// the gluing gadgets (three functional edge-binds, the `sys:Array` equality filter, the dictionary
    /// value/datatype lookup, the datatype-completeness filter, and the density+order permutation) —
    /// live in [`super::array_walk::OrderedArrayWalkProof`]. The delivered result of THIS envelope is
    /// the value-in-index-order column. See [`RelationProofPlanKind::OrderedArrayWalk`].
    OrderedArrayWalk(Box<super::array_walk::OrderedArrayWalkProof>),
}

/// How `A_0` (the semi-naive base) is established.
#[derive(Clone, Debug)]
pub enum TransitiveClosureSeed {
    /// `star`: `A_0 = {a}`, the identity singleton. Carries NO operator proof — the verifier
    /// reconstructs the constant `{a}` directly from the trusted seed node (`path.pl:183`: the
    /// identity branch `X = Y` unconditionally succeeds, so `a` needs no authentication beyond being
    /// a resolvable dictionary node).
    Reflexive,
    /// `plus`: `A_0 = Distinct(object(InnerJoin({a}, E)))` — the deduped one-hop out-neighbourhood
    /// of the constant seed `{a}`.
    Successor {
        /// `InnerJoin({a}, E)` on `{a}.node == E.subject`; output schema `[node, object]`.
        inner: InnerJoinProof,
        /// The InnerJoin output commitments `[node, object]` (checked against the matched projection).
        inner_out: Vec<ArkGT>,
        /// `Distinct(object(inner), [0])` proving the delivered `rows` are exactly the neighbour set.
        distinct: RelationDistinctProof,
        /// `A_0` as resolved node ids; the verifier commits these to obtain `A_0`'s commitment.
        rows: Vec<u64>,
    },
}

/// One accumulate-and-dedupe round `A_{k+1} = Distinct(BagUnion(A_k, object(InnerJoin(A_k, E))))`.
/// Chained by COMMITMENT: the verifier feeds `A_k`'s commitment (from the previous round / seed) and
/// `E`'s commitment into the three audited operator verifiers, then commits `next_rows` to obtain
/// `A_{k+1}` for the following round. NOT nested — this is why the unroll stays `O(K)`, never `2^K`.
#[derive(Clone, Debug)]
pub struct TransitiveClosureRound {
    /// `InnerJoin(A_k, E)` on `A_k.node == E.subject`; output schema `[node, object]`.
    pub inner: InnerJoinProof,
    /// The InnerJoin output commitments `[node, object]`.
    pub inner_out: Vec<ArkGT>,
    /// `BagUnion(A_k, object(inner))`; output is a single-column bag.
    pub union: BagUnionProof,
    /// The BagUnion output commitment `[node]`.
    pub union_out: Vec<ArkGT>,
    /// `Distinct(union, [0])` proving `next_rows` are exactly the deduped union.
    pub distinct: RelationDistinctProof,
    /// `A_{k+1}` as resolved node ids; the verifier commits these to obtain `A_{k+1}`.
    pub next_rows: Vec<u64>,
}

/// The empty-delta fixpoint certificate: `LeftAnti(InnerJoin(A_K, E), A_K)` on `object == node` has
/// ZERO rows, i.e. every one-hop neighbour of `A_K` is already in `A_K` (`A_K` is closed under the
/// step). Together with monotonicity (`A_0 ⊆ A_1 ⊆ … ⊆ A_K`, enforced by union+distinct) and
/// soundness (`A_K ⊆` reachable, enforced by the InnerJoin), this proves `A_K` is exactly the least
/// fixpoint = the reachable set. This is what makes a CYCLE terminate: the delta empties even though a
/// bag/frontier iteration would loop forever.
#[derive(Clone, Debug)]
pub struct TransitiveClosureFixpoint {
    /// `InnerJoin(A_K, E)`; output schema `[node, object]`.
    pub inner: InnerJoinProof,
    /// The InnerJoin output commitments `[node, object]`.
    pub inner_out: Vec<ArkGT>,
    /// `LeftAnti(inner, A_K, [(object=1, node=0)])`; the verifier asserts its output length is 0.
    pub anti: LeftAntiJoinProof,
    /// The (canonically empty) anti-join output commitments; arity matches `inner` (2 columns).
    pub anti_out: Vec<ArkGT>,
}

/// How the ALL-PAIRS base `A_0 ⊆ (src, dst)` is established: `A_0 = Distinct([src,dst], π(E))` — the
/// direction-projected one-hop pairs of the committed edge relation `E`, deduped on the full 2-tuple.
/// PLUS-only, so there is NO reflexive variant (unlike [`TransitiveClosureSeed`]).
#[derive(Clone, Debug)]
pub struct TransitiveClosurePairsSeed {
    /// `Distinct(π_direction(E), [0, 1])` proving `pairs` are exactly the deduped edge pairs. Its
    /// INPUT is the direction-projected 2-column `E` (a reordering of `E`'s authenticated columns —
    /// forward `[subject, object]`, inverted `[object, subject]`), so no fresh commitment is needed.
    pub distinct: RelationDistinctProof,
    /// `A_0` as resolved `(src, dst)` node ids; the verifier commits these to obtain `A_0`'s commitment.
    pub pairs: Vec<(u64, u64)>,
}

/// One ALL-PAIRS accumulate-and-dedupe round
/// `A_{k+1} = Distinct([src,dst], BagUnion(A_k, π_{src,ext}(InnerJoin(A_k, E))))`. Chained by
/// COMMITMENT exactly like [`TransitiveClosureRound`], but every relation is 2-column.
#[derive(Clone, Debug)]
pub struct TransitiveClosurePairsRound {
    /// `InnerJoin(A_k, E)` on `A_k.dst == E.<edge-start>`; output schema `[src, dst, ext]` (3 columns).
    pub inner: InnerJoinProof,
    /// The InnerJoin output commitments `[src, dst, ext]`.
    pub inner_out: Vec<ArkGT>,
    /// `BagUnion(A_k, π_{src,ext}(inner))`; output is a 2-column bag `[src, dst]`.
    pub union: BagUnionProof,
    /// The BagUnion output commitments `[src, dst]`.
    pub union_out: Vec<ArkGT>,
    /// `Distinct(union, [0, 1])` proving `next_pairs` are exactly the deduped 2-tuple union.
    pub distinct: RelationDistinctProof,
    /// `A_{k+1}` as resolved `(src, dst)` node ids; the verifier commits these to obtain `A_{k+1}`.
    pub next_pairs: Vec<(u64, u64)>,
}

/// The ALL-PAIRS empty-delta fixpoint certificate: `LeftAnti(π_{src,ext}(InnerJoin(A_K, E)), A_K)` on
/// `[(src==src), (ext==dst)]` has ZERO rows — every one-edge right-extension of a pair in `A_K` is
/// already in `A_K`. Together with monotonicity (`A_0 ⊆ … ⊆ A_K`) and soundness (`A_K ⊆` the ≥1-hop
/// closure) this proves `A_K` is exactly the transitive closure over 2-tuples. This is what makes a
/// CYCLE terminate.
#[derive(Clone, Debug)]
pub struct TransitiveClosurePairsFixpoint {
    /// `InnerJoin(A_K, E)`; output schema `[src, dst, ext]` (3 columns).
    pub inner: InnerJoinProof,
    /// The InnerJoin output commitments `[src, dst, ext]`.
    pub inner_out: Vec<ArkGT>,
    /// `LeftAnti(π_{src,ext}(inner), A_K, [(0,0), (1,1)])`; the verifier asserts its output length is 0.
    pub anti: LeftAntiJoinProof,
    /// The (canonically empty) anti-join output commitments; arity matches the pairs relation (2 columns).
    pub anti_out: Vec<ArkGT>,
}
// ============================================================================
// Bounded transitive-closure (`star`/`plus`) orchestration helpers.
//
// These reuse the audited InnerJoin/BagUnion/Distinct/LeftAnti operator cores UNCHANGED,
// threading each step by COMMITMENT (never by nesting one round inside the next — that would
// duplicate `A_k` and blow the unroll up to `2^K` nodes). Every operation runs on a fresh
// transcript domain-separated by (root, closure query digest, step index, role), so a proof
// for one step can never be replayed as another.
// ============================================================================

/// A fresh transcript for one closure operator call, domain-separated so no sub-proof is fungible
/// with another. Both prover and verifier build this identically.
pub(super) fn closure_transcript(root: &ProofHash, digest: &ProofHash, step: u64, role: &[u8]) -> Transcript254 {
    let mut transcript = Transcript254::new(b"parture-transitive-closure-envelope");
    transcript.append_bytes(b"pow/tclosure/root", root);
    transcript.append_bytes(b"pow/tclosure/query", digest);
    transcript.append_bytes(b"pow/tclosure/step", &step.to_le_bytes());
    transcript.append_bytes(b"pow/tclosure/role", role);
    transcript
}

/// Resolve the bound seed node `a` to its layer node/value dictionary id, failing CLOSED if the node
/// is absent (a seed that is not a real dictionary node cannot begin an authenticated traversal). Both
/// prover and verifier resolve identically, so the reconstructed `{a}` commitment is fully pinned.
pub(super) fn resolve_closure_seed_id(
    view: &crate::proof::VerifierCommitmentRef<'_>,
    seed: &CanonicalObject,
) -> io::Result<u64> {
    view.node_value_catalog
        .iter()
        .find(|record| &record.triple.object == seed)
        .map(|record| record.object_id)
        .ok_or_else(|| invalid("transitive closure seed node is absent from the layer dictionary"))
}

/// The canonical `{a}` singleton relation commitment: one real row holding the seed id, padded to the
/// domain `2`. Prover and verifier compute this identically (the "constant seed the verifier
/// reconstructs from the AST"); no operator proof authenticates it because the `star` identity branch
/// unconditionally succeeds.
pub(super) fn closure_seed_commit(seed_id: u64) -> io::Result<RelationCommit> {
    let column = vec![ArkFr::from_u64(seed_id), ArkFr::zero()];
    Ok(RelationCommit {
        commitments: vec![commit_column(&column)?],
        log_domain: 1,
        n: 1,
    })
}

/// The verifier-side commitment of an accumulator SET delivered as public node ids: commit the ids in
/// the delivered order, padded exactly as [`prove_relation_distinct_with_context`] pads its single-key
/// output, so this equals the commitment the prover threaded forward.
pub(super) fn closure_set_commit(rows: &[u64]) -> io::Result<RelationCommit> {
    let log_domain = rows.len().max(1).next_power_of_two().max(2).trailing_zeros() as usize;
    let mut column: Vec<ArkFr> = rows.iter().map(|id| ArkFr::from_u64(*id)).collect();
    column.resize(1usize << log_domain, ArkFr::zero());
    Ok(RelationCommit {
        commitments: vec![commit_column(&column)?],
        log_domain,
        n: rows.len(),
    })
}

/// Decode an authenticated node/value store id from its canonical field image (the store commits a
/// node id as `ArkFr::from_u64(id)`). Reject a field outside the canonical u64 image rather than
/// truncating.
pub(super) fn closure_field_node_id(field: ArkFr) -> io::Result<u64> {
    generated_decimal_u64_value(field)
}

/// Wrap flat node ids as single-column distinct rows for [`verify_relation_distinct`].
pub(super) fn closure_rows(rows: &[u64]) -> Vec<Vec<u64>> {
    rows.iter().map(|id| vec![*id]).collect()
}

/// The deduped node-id SET of a single-column relation witness (prover side), preserving first
/// occurrence order.
pub(super) fn closure_distinct_ids(witness: &RelationWitness) -> io::Result<Vec<u64>> {
    let column = witness
        .columns
        .first()
        .ok_or_else(|| invalid("transitive closure step relation has no column"))?;
    let mut seen = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    for field in &column[..witness.n] {
        let id = closure_field_node_id(*field)?;
        if seen.insert(id) {
            out.push(id);
        }
    }
    Ok(out)
}

/// The 1-column OBJECT branch of an `InnerJoin(A, E)` result (schema `[node, object]`): column index 1.
pub(super) fn closure_object_branch(inner: &InnerJoinResult) -> io::Result<RelationWitness> {
    if inner.columns.len() != 2 || inner.commitments.len() != 2 {
        return Err(invalid("transitive closure inner join must yield [node, object]"));
    }
    Ok(RelationWitness {
        columns: vec![inner.columns[1].clone()],
        commitments: vec![inner.commitments[1]],
        log_domain: inner.log_domain,
        n: inner.n,
    })
}

/// The full 2-column `InnerJoin(A, E)` witness (schema `[node, object]`).
pub(super) fn closure_inner_witness(inner: &InnerJoinResult) -> RelationWitness {
    RelationWitness {
        columns: inner.columns.clone(),
        commitments: inner.commitments.clone(),
        log_domain: inner.log_domain,
        n: inner.n,
    }
}

pub(super) fn closure_union_witness(union: &BagUnionResult) -> RelationWitness {
    RelationWitness {
        columns: union.columns.clone(),
        commitments: union.commitments.clone(),
        log_domain: union.log_domain,
        n: union.n,
    }
}

pub(super) fn closure_distinct_witness(distinct: &RelationDistinctResult) -> RelationWitness {
    RelationWitness {
        columns: distinct.columns.clone(),
        commitments: distinct.commitments.clone(),
        log_domain: distinct.log_domain,
        n: distinct.n,
    }
}

/// The seed + rounds + fixpoint of a bounded semi-naive transitive closure — the SOURCE-AGNOSTIC
/// heart of [`RecursiveRelationEnvelope::prove_transitive_closure`], factored out so a reconciled
/// (delta-stack) closure can reuse the EXACT round/fixpoint machinery over a `present`-sourced edge.
///
/// It consumes ONLY the already-committed `edge_witness` (the E relation columns/commitments) and
/// the resolved `seed_id`, then drives the audited InnerJoin / BagUnion / Distinct / LeftAnti
/// operator cores UNCHANGED, threading each accumulator forward by commitment on fresh
/// `(root, digest, step, role)`-separated transcripts. Whether E was proven by a materialized
/// single-layer BGP leaf or a reconciled FOUNDATION-A `present_bgp` leaf makes no difference here —
/// only the edge PROVIDER and the seed RESOLUTION are source-specific. The materialized path calls
/// this verbatim, so its proofs are byte-identical.
pub(super) struct ClosureRoundsWitness {
    pub seed: TransitiveClosureSeed,
    pub accumulator: RelationWitness,
    pub rounds: Vec<TransitiveClosureRound>,
    pub fixpoint: TransitiveClosureFixpoint,
    /// The reachable node-id set in delivered order (the accumulator's real prefix).
    pub reachable_ids: Vec<u64>,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prove_closure_rounds(
    context: &StoreProverContext<'_>,
    edge_witness: &RelationWitness,
    seed_id: u64,
    root: ProofHash,
    digest: ProofHash,
    reflexive: bool,
    inverted: bool,
    nodes: &mut usize,
) -> io::Result<ClosureRoundsWitness> {
    // See `prove_transitive_closure`: forward walks join E.subject (0), inverted walks E.object (1).
    let edge_key = usize::from(inverted);
    let seed_witness =
        RelationWitness::commit_with_context(context, vec![vec![ArkFr::from_u64(seed_id)]], 1)?;

    // (3) A_0: star => {a}; plus => Distinct(object(InnerJoin({a}, E))).
    let (seed_proof, mut accumulator, mut accumulator_ids): (
        TransitiveClosureSeed,
        RelationWitness,
        std::collections::BTreeSet<u64>,
    ) = if reflexive {
        let ids = std::iter::once(seed_id).collect();
        (TransitiveClosureSeed::Reflexive, seed_witness.clone(), ids)
    } else {
        let mut t_inner = closure_transcript(&root, &digest, 0, b"seed-inner");
        let inner = prove_inner_join_with_keys_and_context(
            context, &seed_witness, edge_witness, &[(0, edge_key)], &mut t_inner,
        )?;
        let object_witness = closure_object_branch(&inner)?;
        let rows = closure_distinct_ids(&object_witness)?;
        let mut t_distinct = closure_transcript(&root, &digest, 0, b"seed-distinct");
        let distinct = prove_relation_distinct_with_context(
            context, &object_witness, &[0], &closure_rows(&rows), &mut t_distinct,
        )?;
        let ids = rows.iter().copied().collect();
        let a0 = closure_distinct_witness(&distinct);
        (
            TransitiveClosureSeed::Successor {
                inner: inner.proof,
                inner_out: inner.commitments,
                distinct: distinct.proof,
                rows,
            },
            a0,
            ids,
        )
    };

    // (4) Accumulate-and-dedupe rounds until the frontier is closed under the step.
    let mut rounds: Vec<TransitiveClosureRound> = Vec::new();
    let fixpoint = loop {
        if *nodes >= MAX_RELATION_PROOF_NODES {
            return Err(invalid("transitive closure proving exceeds structural node limit"));
        }
        *nodes += 1;
        let step = rounds.len() as u64;

        // step = InnerJoin(A_k, E) on (A_k.node == E.subject) -> [node, object].
        let mut t_inner = closure_transcript(&root, &digest, step, b"step-inner");
        let inner = prove_inner_join_with_keys_and_context(
            context, &accumulator, edge_witness, &[(0, edge_key)], &mut t_inner,
        )?;
        let object_witness = closure_object_branch(&inner)?;
        let neighbour_ids = closure_distinct_ids(&object_witness)?;

        if neighbour_ids.iter().all(|id| accumulator_ids.contains(id)) {
            // FIXPOINT: every one-hop neighbour of A_k is already present. The anti-join of the
            // join's objects against A_k is empty -> A_k is closed under the step.
            let mut t_anti = closure_transcript(&root, &digest, step, b"fix-anti");
            let anti = prove_left_anti_join_with_keys_and_context(
                context,
                &closure_inner_witness(&inner),
                &accumulator,
                &[(1, 0)],
                &mut t_anti,
            )?;
            if anti.n != 0 {
                return Err(invalid(
                    "transitive closure fixpoint anti-join is unexpectedly nonempty",
                ));
            }
            break TransitiveClosureFixpoint {
                inner: inner.proof,
                inner_out: inner.commitments,
                anti: anti.proof,
                anti_out: anti.commitments,
            };
        }

        if rounds.len() >= MAX_TRANSITIVE_CLOSURE_ROUNDS {
            return Err(invalid(
                "transitive closure frontier is deeper than the round budget (reject closed)",
            ));
        }

        // A_{k+1} = Distinct(BagUnion(A_k, object-bag), [0]).
        let mut t_union = closure_transcript(&root, &digest, step, b"round-union");
        let union = prove_bag_union_with_context(
            context,
            &[accumulator.clone(), object_witness],
            &mut t_union,
        )?;
        let union_witness = closure_union_witness(&union);
        let next_rows = closure_distinct_ids(&union_witness)?;
        let mut t_distinct = closure_transcript(&root, &digest, step, b"round-distinct");
        let distinct = prove_relation_distinct_with_context(
            context, &union_witness, &[0], &closure_rows(&next_rows), &mut t_distinct,
        )?;
        accumulator = closure_distinct_witness(&distinct);
        accumulator_ids = next_rows.iter().copied().collect();
        rounds.push(TransitiveClosureRound {
            inner: inner.proof,
            inner_out: inner.commitments,
            union: union.proof,
            union_out: union.commitments,
            distinct: distinct.proof,
            next_rows,
        });
    };

    let n = accumulator.n;
    let reachable_ids = accumulator.columns[0][..n]
        .iter()
        .map(|field| closure_field_node_id(*field))
        .collect::<io::Result<Vec<u64>>>()?;
    Ok(ClosureRoundsWitness {
        seed: seed_proof,
        accumulator,
        rounds,
        fixpoint,
        reachable_ids,
    })
}

/// The verifier mirror of [`prove_closure_rounds`]: replay the seed + rounds + fixpoint over an
/// already-authenticated `edge_commit` + resolved `seed_id`, reconstructing each accumulator
/// commitment from the PUBLIC row sets and re-running the audited operator verifiers on the same
/// `(root, digest, step, role)` transcripts. Returns the reachable node-id set. Source-agnostic —
/// the materialized `verify_transitive_closure` calls it verbatim, so its checks are unchanged.
#[allow(clippy::too_many_arguments)]
pub(super) fn verify_closure_rounds(
    edge_commit: &RelationCommit,
    seed_id: u64,
    root: ProofHash,
    digest: ProofHash,
    reflexive: bool,
    inverted: bool,
    seed: &TransitiveClosureSeed,
    rounds: &[TransitiveClosureRound],
    fixpoint: &TransitiveClosureFixpoint,
    nodes: &mut usize,
) -> io::Result<Vec<u64>> {
    // See `prove_transitive_closure`: forward walks join E.subject (0), inverted walks E.object (1).
    let edge_key = usize::from(inverted);

    // Helper: verify one `InnerJoin(A, E) -> [node, object]` step and return (join commit,
    // object-branch commit). The output length / domain are derived from the proof and cross-checked
    // by the audited verifier.
    let verify_inner = |left: &RelationCommit,
                        inner_proof: &InnerJoinProof,
                        inner_out: &[ArkGT],
                        step: u64,
                        role: &[u8]|
     -> io::Result<(RelationCommit, RelationCommit)> {
        if inner_out.len() != 2 {
            return Err(invalid("transitive closure inner join output arity must be 2"));
        }
        let j_len = inner_proof.anti.join_len;
        let j_log = inner_proof.anti.join.num_vars();
        let mut transcript = closure_transcript(&root, &digest, step, role);
        verify_inner_join_with_keys(
            left,
            edge_commit,
            &[(0, edge_key)],
            inner_out,
            j_len,
            j_log,
            inner_proof,
            &mut transcript,
        )?;
        let join_commit = RelationCommit {
            commitments: inner_out.to_vec(),
            log_domain: j_log,
            n: j_len,
        };
        let object_commit = RelationCommit {
            commitments: vec![inner_out[1]],
            log_domain: j_log,
            n: j_len,
        };
        Ok((join_commit, object_commit))
    };

    // (3) A_0.
    let (mut accumulator_commit, mut accumulator_ids, mut accumulator_rows): (
        RelationCommit,
        std::collections::BTreeSet<u64>,
        Vec<u64>,
    ) = match (reflexive, seed) {
        (true, TransitiveClosureSeed::Reflexive) => {
            // star: A_0 = {a}, reconstructed as a trusted constant (identity always succeeds).
            (
                closure_seed_commit(seed_id)?,
                std::iter::once(seed_id).collect(),
                vec![seed_id],
            )
        }
        (false, TransitiveClosureSeed::Successor { inner, inner_out, distinct, rows }) => {
            // plus: A_0 = Distinct(object(InnerJoin({a}, E))).
            let seed_commit = closure_seed_commit(seed_id)?;
            let (_join, object_commit) =
                verify_inner(&seed_commit, inner, inner_out, 0, b"seed-inner")?;
            let mut t_distinct = closure_transcript(&root, &digest, 0, b"seed-distinct");
            verify_relation_distinct(
                &object_commit,
                &[0],
                &closure_rows(rows),
                distinct,
                &mut t_distinct,
            )?;
            (
                closure_set_commit(rows)?,
                rows.iter().copied().collect(),
                rows.clone(),
            )
        }
        _ => {
            return Err(invalid(
                "transitive closure seed variant does not match the star/plus plan",
            ))
        }
    };

    // (4) Replay each accumulate-and-dedupe round against the SAME E and the previous accumulator.
    for (step, round) in rounds.iter().enumerate() {
        if *nodes >= MAX_RELATION_PROOF_NODES {
            return Err(invalid(
                "transitive closure verification exceeds structural node limit",
            ));
        }
        *nodes += 1;
        let step = step as u64;
        let (_join, object_commit) = verify_inner(
            &accumulator_commit,
            &round.inner,
            &round.inner_out,
            step,
            b"step-inner",
        )?;

        // BagUnion(A_k, object-bag): both single-column; output length is the branch-length sum and
        // the domain is bound in the proof, so the audited verifier re-derives and checks them.
        if round.union_out.len() != 1 {
            return Err(invalid("transitive closure union output arity must be 1"));
        }
        let union_len = accumulator_commit
            .n
            .checked_add(object_commit.n)
            .ok_or_else(|| invalid("transitive closure union length overflow"))?;
        let union_log = output_log_domain(union_len)?;
        let mut t_union = closure_transcript(&root, &digest, step, b"round-union");
        verify_bag_union(
            &[accumulator_commit.clone(), object_commit],
            &round.union_out,
            union_len,
            union_log,
            &round.union,
            &mut t_union,
        )?;
        let union_commit = RelationCommit {
            commitments: round.union_out.clone(),
            log_domain: union_log,
            n: union_len,
        };

        // Distinct(union, [0]) proves next_rows are exactly the deduped union set.
        let mut t_distinct = closure_transcript(&root, &digest, step, b"round-distinct");
        verify_relation_distinct(
            &union_commit,
            &[0],
            &closure_rows(&round.next_rows),
            &round.distinct,
            &mut t_distinct,
        )?;

        // Defensive monotonicity: every element of A_k must survive into A_{k+1}. This is already
        // implied by union-branch-0 + the distinct exact-cover argument, but the redundant public
        // set check costs nothing and pins the completeness invariant for the audit.
        let next_ids: std::collections::BTreeSet<u64> =
            round.next_rows.iter().copied().collect();
        if next_ids.len() != round.next_rows.len() {
            return Err(invalid("transitive closure round produced a duplicate set element"));
        }
        if !accumulator_ids.iter().all(|id| next_ids.contains(id)) {
            return Err(invalid("transitive closure round dropped a prior reachable node"));
        }

        accumulator_commit = closure_set_commit(&round.next_rows)?;
        accumulator_ids = next_ids;
        accumulator_rows = round.next_rows.clone();
    }

    // (5) Empty-delta fixpoint: LeftAnti(InnerJoin(A_K, E), A_K) on (object == node) has 0 rows,
    // i.e. A_K is closed under the step. Passing output_len 0 IS the fixpoint assertion — a
    // non-converged closure makes the audited anti-join verifier reject.
    let step = rounds.len() as u64;
    let (join_commit, _object_commit) = verify_inner(
        &accumulator_commit,
        &fixpoint.inner,
        &fixpoint.inner_out,
        step,
        b"step-inner",
    )?;
    if fixpoint.anti_out.len() != join_commit.commitments.len() {
        return Err(invalid("transitive closure fixpoint anti-join output arity mismatch"));
    }
    let anti_log = fixpoint.anti.join.num_vars();
    let mut t_anti = closure_transcript(&root, &digest, step, b"fix-anti");
    verify_left_anti_join_with_keys(
        &join_commit,
        &accumulator_commit,
        &[(1, 0)],
        &fixpoint.anti_out,
        0,
        anti_log,
        &fixpoint.anti,
        &mut t_anti,
    )?;

    Ok(accumulator_rows)
}

// ============================================================================
// Bounded ALL-PAIRS transitive-closure orchestration helpers (2-column accumulator).
//
// Same discipline as the single-seed helpers above — reuse the audited InnerJoin/BagUnion/Distinct/
// LeftAnti cores UNCHANGED, thread each round by COMMITMENT, one fresh domain-separated transcript per
// operator call — but the accumulator is a 2-column pairs relation `A ⊆ (src, dst)`.
// ============================================================================

/// The verifier-side commitment of a pairs accumulator delivered as public `(src, dst)` node ids,
/// padded EXACTLY as [`prove_relation_distinct_with_context`] pads its two-key output, so this equals
/// the commitment the prover threaded forward.
pub(super) fn closure_pairs_commit(pairs: &[(u64, u64)]) -> io::Result<RelationCommit> {
    let log_domain = pairs.len().max(1).next_power_of_two().max(2).trailing_zeros() as usize;
    let size = 1usize << log_domain;
    let mut src: Vec<ArkFr> = pairs.iter().map(|(s, _)| ArkFr::from_u64(*s)).collect();
    let mut dst: Vec<ArkFr> = pairs.iter().map(|(_, d)| ArkFr::from_u64(*d)).collect();
    src.resize(size, ArkFr::zero());
    dst.resize(size, ArkFr::zero());
    Ok(RelationCommit {
        commitments: vec![commit_column(&src)?, commit_column(&dst)?],
        log_domain,
        n: pairs.len(),
    })
}

/// The direction-projected 2-column base `π(E)` for the ALL-PAIRS seed (prover side). Forward
/// `A_0 = (subject, object)`; inverted `A_0 = (object, subject)` — a REORDERING of `E`'s already
/// authenticated columns/commitments, so no fresh commitment is engaged.
pub(super) fn closure_pairs_seed_input(edge: &RelationWitness, inverted: bool) -> io::Result<RelationWitness> {
    if edge.columns.len() != 2 || edge.commitments.len() != 2 {
        return Err(invalid("all-pairs closure edge must be a [subject, object] relation"));
    }
    let (a, b) = if inverted { (1, 0) } else { (0, 1) };
    Ok(RelationWitness {
        columns: vec![edge.columns[a].clone(), edge.columns[b].clone()],
        commitments: vec![edge.commitments[a], edge.commitments[b]],
        log_domain: edge.log_domain,
        n: edge.n,
    })
}

/// The direction-projected 2-column base commitment for the ALL-PAIRS seed (verifier side); mirrors
/// [`closure_pairs_seed_input`] over the authenticated edge commitment.
pub(super) fn closure_pairs_seed_input_commit(edge: &RelationCommit, inverted: bool) -> io::Result<RelationCommit> {
    if edge.commitments.len() != 2 {
        return Err(invalid("all-pairs closure edge must be a [subject, object] relation"));
    }
    let (a, b) = if inverted { (1, 0) } else { (0, 1) };
    Ok(RelationCommit {
        commitments: vec![edge.commitments[a], edge.commitments[b]],
        log_domain: edge.log_domain,
        n: edge.n,
    })
}

/// The `[src, ext]` extended-pair branch of an `InnerJoin(A, E)` result (schema `[src, dst, ext]`):
/// columns 0 and 2 (the accumulator's own `src` and the edge's far endpoint), the pair extended by one hop.
pub(super) fn closure_pairs_extended_branch(inner: &InnerJoinResult) -> io::Result<RelationWitness> {
    if inner.columns.len() != 3 || inner.commitments.len() != 3 {
        return Err(invalid("all-pairs closure inner join must yield [src, dst, ext]"));
    }
    Ok(RelationWitness {
        columns: vec![inner.columns[0].clone(), inner.columns[2].clone()],
        commitments: vec![inner.commitments[0], inner.commitments[2]],
        log_domain: inner.log_domain,
        n: inner.n,
    })
}

/// The deduped `(src, dst)` pair SET of a 2-column relation witness (prover side), preserving first
/// occurrence order.
pub(super) fn closure_pairs_distinct(witness: &RelationWitness) -> io::Result<Vec<(u64, u64)>> {
    if witness.columns.len() != 2 {
        return Err(invalid("all-pairs closure pairs relation must have two columns"));
    }
    let mut seen = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    for i in 0..witness.n {
        let s = closure_field_node_id(witness.columns[0][i])?;
        let d = closure_field_node_id(witness.columns[1][i])?;
        if seen.insert((s, d)) {
            out.push((s, d));
        }
    }
    Ok(out)
}

/// Wrap flat `(src, dst)` pairs as two-column distinct rows for [`prove_relation_distinct_with_context`]
/// / [`verify_relation_distinct`].
pub(super) fn closure_pairs_rows(pairs: &[(u64, u64)]) -> Vec<Vec<u64>> {
    pairs.iter().map(|(s, d)| vec![*s, *d]).collect()
}

/// The seed + rounds + fixpoint of a bounded ALL-PAIRS semi-naive transitive closure. Mirrors
/// [`prove_closure_rounds`] but every relation is 2-column and PLUS-only (no reflexive seed).
pub(super) struct ClosurePairsRoundsWitness {
    pub seed: TransitiveClosurePairsSeed,
    pub accumulator: RelationWitness,
    pub rounds: Vec<TransitiveClosurePairsRound>,
    pub fixpoint: TransitiveClosurePairsFixpoint,
}

pub(super) fn prove_closure_pairs_rounds(
    context: &StoreProverContext<'_>,
    edge_witness: &RelationWitness,
    root: ProofHash,
    digest: ProofHash,
    inverted: bool,
    nodes: &mut usize,
) -> io::Result<ClosurePairsRoundsWitness> {
    // Forward joins `A_k.dst == E.subject` (edge col 0); inverted joins `A_k.dst == E.object` (col 1).
    let edge_key = usize::from(inverted);

    // A_0 = Distinct([src,dst], π_direction(E)): the deduped one-hop pairs.
    let seed_input = closure_pairs_seed_input(edge_witness, inverted)?;
    let seed_pairs = closure_pairs_distinct(&seed_input)?;
    let mut t_seed = closure_transcript(&root, &digest, 0, b"pairs-seed-distinct");
    let seed_distinct = prove_relation_distinct_with_context(
        context, &seed_input, &[0, 1], &closure_pairs_rows(&seed_pairs), &mut t_seed,
    )?;
    let mut accumulator = closure_distinct_witness(&seed_distinct);
    let mut accumulator_pairs: std::collections::BTreeSet<(u64, u64)> =
        seed_pairs.iter().copied().collect();
    let seed = TransitiveClosurePairsSeed {
        distinct: seed_distinct.proof,
        pairs: seed_pairs,
    };

    let mut rounds: Vec<TransitiveClosurePairsRound> = Vec::new();
    let fixpoint = loop {
        if *nodes >= MAX_RELATION_PROOF_NODES {
            return Err(invalid("all-pairs closure proving exceeds structural node limit"));
        }
        *nodes += 1;
        let step = rounds.len() as u64;

        // step = InnerJoin(A_k, E) on (A_k.dst == E.<start>) -> [src, dst, ext].
        let mut t_inner = closure_transcript(&root, &digest, step, b"pairs-step-inner");
        let inner = prove_inner_join_with_keys_and_context(
            context, &accumulator, edge_witness, &[(1, edge_key)], &mut t_inner,
        )?;
        let extended = closure_pairs_extended_branch(&inner)?;
        let extended_pairs = closure_pairs_distinct(&extended)?;

        if extended_pairs.iter().all(|pair| accumulator_pairs.contains(pair)) {
            // FIXPOINT: every one-edge right-extension of A_k is already present. The anti-join of the
            // extended pairs against A_k (on the FULL 2-tuple) is empty -> A_k is closed.
            let mut t_anti = closure_transcript(&root, &digest, step, b"pairs-fix-anti");
            let anti = prove_left_anti_join_with_keys_and_context(
                context, &extended, &accumulator, &[(0, 0), (1, 1)], &mut t_anti,
            )?;
            if anti.n != 0 {
                return Err(invalid(
                    "all-pairs closure fixpoint anti-join is unexpectedly nonempty",
                ));
            }
            break TransitiveClosurePairsFixpoint {
                inner: inner.proof,
                inner_out: inner.commitments,
                anti: anti.proof,
                anti_out: anti.commitments,
            };
        }

        if rounds.len() >= MAX_TRANSITIVE_CLOSURE_ROUNDS {
            return Err(invalid(
                "all-pairs closure frontier is deeper than the round budget (reject closed)",
            ));
        }

        // A_{k+1} = Distinct(BagUnion(A_k, extended-pairs), [0, 1]).
        let mut t_union = closure_transcript(&root, &digest, step, b"pairs-round-union");
        let union = prove_bag_union_with_context(
            context,
            &[accumulator.clone(), extended],
            &mut t_union,
        )?;
        let union_witness = closure_union_witness(&union);
        let next_pairs = closure_pairs_distinct(&union_witness)?;
        let mut t_distinct = closure_transcript(&root, &digest, step, b"pairs-round-distinct");
        let distinct = prove_relation_distinct_with_context(
            context, &union_witness, &[0, 1], &closure_pairs_rows(&next_pairs), &mut t_distinct,
        )?;
        accumulator = closure_distinct_witness(&distinct);
        accumulator_pairs = next_pairs.iter().copied().collect();
        rounds.push(TransitiveClosurePairsRound {
            inner: inner.proof,
            inner_out: inner.commitments,
            union: union.proof,
            union_out: union.commitments,
            distinct: distinct.proof,
            next_pairs,
        });
    };

    Ok(ClosurePairsRoundsWitness {
        seed,
        accumulator,
        rounds,
        fixpoint,
    })
}

/// The verifier mirror of [`prove_closure_pairs_rounds`].
#[allow(clippy::too_many_arguments)]
pub(super) fn verify_closure_pairs_rounds(
    edge_commit: &RelationCommit,
    root: ProofHash,
    digest: ProofHash,
    inverted: bool,
    seed: &TransitiveClosurePairsSeed,
    rounds: &[TransitiveClosurePairsRound],
    fixpoint: &TransitiveClosurePairsFixpoint,
    nodes: &mut usize,
) -> io::Result<Vec<(u64, u64)>> {
    let edge_key = usize::from(inverted);

    // Verify one `InnerJoin(A, E) -> [src, dst, ext]` and return (join commit, extended-branch commit).
    let verify_inner = |left: &RelationCommit,
                        inner_proof: &InnerJoinProof,
                        inner_out: &[ArkGT],
                        step: u64,
                        role: &[u8]|
     -> io::Result<(RelationCommit, RelationCommit)> {
        if inner_out.len() != 3 {
            return Err(invalid("all-pairs closure inner join output arity must be 3"));
        }
        let j_len = inner_proof.anti.join_len;
        let j_log = inner_proof.anti.join.num_vars();
        let mut transcript = closure_transcript(&root, &digest, step, role);
        verify_inner_join_with_keys(
            left,
            edge_commit,
            &[(1, edge_key)],
            inner_out,
            j_len,
            j_log,
            inner_proof,
            &mut transcript,
        )?;
        let join_commit = RelationCommit {
            commitments: inner_out.to_vec(),
            log_domain: j_log,
            n: j_len,
        };
        let extended_commit = RelationCommit {
            commitments: vec![inner_out[0], inner_out[2]],
            log_domain: j_log,
            n: j_len,
        };
        Ok((join_commit, extended_commit))
    };

    // A_0 = Distinct([src,dst], π_direction(E)).
    let seed_input_commit = closure_pairs_seed_input_commit(edge_commit, inverted)?;
    let mut t_seed = closure_transcript(&root, &digest, 0, b"pairs-seed-distinct");
    verify_relation_distinct(
        &seed_input_commit,
        &[0, 1],
        &closure_pairs_rows(&seed.pairs),
        &seed.distinct,
        &mut t_seed,
    )?;
    let mut accumulator_pairs: std::collections::BTreeSet<(u64, u64)> =
        std::collections::BTreeSet::new();
    for pair in &seed.pairs {
        if !accumulator_pairs.insert(*pair) {
            return Err(invalid("all-pairs closure seed contains a duplicate pair"));
        }
    }
    let mut accumulator_commit = closure_pairs_commit(&seed.pairs)?;
    let mut accumulator_rows = seed.pairs.clone();

    // Replay each accumulate-and-dedupe round against the SAME E and the previous accumulator.
    for (step, round) in rounds.iter().enumerate() {
        if *nodes >= MAX_RELATION_PROOF_NODES {
            return Err(invalid(
                "all-pairs closure verification exceeds structural node limit",
            ));
        }
        *nodes += 1;
        let step = step as u64;
        let (_join, extended_commit) = verify_inner(
            &accumulator_commit,
            &round.inner,
            &round.inner_out,
            step,
            b"pairs-step-inner",
        )?;

        if round.union_out.len() != 2 {
            return Err(invalid("all-pairs closure union output arity must be 2"));
        }
        let union_len = accumulator_commit
            .n
            .checked_add(extended_commit.n)
            .ok_or_else(|| invalid("all-pairs closure union length overflow"))?;
        let union_log = output_log_domain(union_len)?;
        let mut t_union = closure_transcript(&root, &digest, step, b"pairs-round-union");
        verify_bag_union(
            &[accumulator_commit.clone(), extended_commit],
            &round.union_out,
            union_len,
            union_log,
            &round.union,
            &mut t_union,
        )?;
        let union_commit = RelationCommit {
            commitments: round.union_out.clone(),
            log_domain: union_log,
            n: union_len,
        };

        let mut t_distinct = closure_transcript(&root, &digest, step, b"pairs-round-distinct");
        verify_relation_distinct(
            &union_commit,
            &[0, 1],
            &closure_pairs_rows(&round.next_pairs),
            &round.distinct,
            &mut t_distinct,
        )?;

        // Defensive monotonicity over the FULL 2-tuple: no duplicate pair, every prior pair survives.
        let next_ids: std::collections::BTreeSet<(u64, u64)> =
            round.next_pairs.iter().copied().collect();
        if next_ids.len() != round.next_pairs.len() {
            return Err(invalid("all-pairs closure round produced a duplicate pair"));
        }
        if !accumulator_pairs.iter().all(|pair| next_ids.contains(pair)) {
            return Err(invalid("all-pairs closure round dropped a prior reachable pair"));
        }

        accumulator_commit = closure_pairs_commit(&round.next_pairs)?;
        accumulator_pairs = next_ids;
        accumulator_rows = round.next_pairs.clone();
    }

    // Empty-delta fixpoint: LeftAnti(π(InnerJoin(A_K, E)), A_K) on [(0,0),(1,1)] has 0 rows.
    let step = rounds.len() as u64;
    let (_join, extended_commit) = verify_inner(
        &accumulator_commit,
        &fixpoint.inner,
        &fixpoint.inner_out,
        step,
        b"pairs-step-inner",
    )?;
    if fixpoint.anti_out.len() != extended_commit.commitments.len() {
        return Err(invalid("all-pairs closure fixpoint anti-join output arity mismatch"));
    }
    let anti_log = fixpoint.anti.join.num_vars();
    let mut t_anti = closure_transcript(&root, &digest, step, b"pairs-fix-anti");
    verify_left_anti_join_with_keys(
        &extended_commit,
        &accumulator_commit,
        &[(0, 0), (1, 1)],
        &fixpoint.anti_out,
        0,
        anti_log,
        &fixpoint.anti,
        &mut t_anti,
    )?;

    Ok(accumulator_rows)
}

impl RecursiveRelationEnvelope {
    pub(super) fn relation_commit(&self, mapping: Option<&[usize]>) -> io::Result<RelationCommit> {
        let commitments = match mapping {
            Some(mapping) => mapping
                .iter()
                .map(|index| {
                    self.result_commitments
                        .get(*index)
                        .cloned()
                        .ok_or_else(|| invalid("recursive relation mapping out of range"))
                })
                .collect::<io::Result<Vec<_>>>()?,
            None => self.result_commitments.clone(),
        };
        Ok(RelationCommit {
            commitments,
            log_domain: self.result_log_domain,
            n: self.result_len,
        })
    }

    pub(super) fn relation_witness(&self, mapping: Option<&[usize]>) -> io::Result<RelationWitness> {
        let indexes: Vec<usize> = match mapping {
            Some(mapping) => mapping.to_vec(),
            None => (0..self.result_columns.len()).collect(),
        };
        let columns = indexes
            .iter()
            .map(|index| {
                let mut column = self.result_columns
                    .get(*index)
                    .cloned()
                    .ok_or_else(|| invalid("recursive relation mapping out of range"))?;
                column.resize(1usize << self.result_log_domain, ArkFr::zero());
                Ok(column)
            })
            .collect::<io::Result<Vec<_>>>()?;
        let commitments = indexes
            .iter()
            .map(|index| {
                self.result_commitments
                    .get(*index)
                    .copied()
                    .ok_or_else(|| invalid("recursive relation mapping out of range"))
            })
            .collect::<io::Result<Vec<_>>>()?;
        Ok(RelationWitness {
            columns,
            commitments,
            log_domain: self.result_log_domain,
            n: self.result_len,
        })
    }

    pub(super) fn finish_proved(
        layer: &LayerCommitment,
        plan: &RelationProofPlan,
        proof: RecursiveRelationProof,
        commitments: Vec<ArkGT>,
        columns: Vec<Vec<ArkFr>>,
        log_domain: usize,
        n: usize,
    ) -> io::Result<Self> {
        let (result_descriptors, ground_result) = plan.result_shape()?;
        if columns.iter().any(|column| column.len() < n) {
            return Err(invalid("recursive prover result column is shorter than its row count"));
        }
        let columns = columns
            .into_iter()
            .map(|column| column[..n].to_vec())
            .collect::<Vec<_>>();
        let result_values = if ground_result {
            Vec::new()
        } else {
            resolve_result_values(layer, layer.resolution_root(), &columns, &result_descriptors)?
        };
        Ok(Self {
            layer_root: layer.state.commitment_root,
            query_digest: plan.query_digest,
            proof,
            result_commitments: commitments,
            result_columns: columns,
            result_log_domain: log_domain,
            result_len: n,
            ground_result,
            result_descriptors,
            result_values,
        })
    }

    /// Explicitly construct a complete recursive relation proof with caller-owned
    /// prover arithmetic. Child nodes reuse this exact context.
    pub fn prove_with_context(
        context: &StoreProverContext<'_>,
        layer: &LayerCommitment,
        plan: &RelationProofPlan,
    ) -> io::Result<Self> {
        // COMMITMENT DIET (Wave 0): a delta layer's head view / full dictionary is no
        // longer committed, so it is not directly provable (LayerTripleCount, BGP leaves,
        // aggregates all read the materialized head). Reject closed; roll up to a base.
        crate::proof::query::require_base_layer_for_query(&layer.state)?;
        plan.result_shape()?;
        let mut nodes = 0;
        Self::prove_bounded(context, layer, plan, 1, &mut nodes)
    }

    /// Explicit Arkworks CPU convenience for recursive proving.
    pub fn prove(layer: &LayerCommitment, plan: &RelationProofPlan) -> io::Result<Self> {
        Self::prove_with_context(&StoreProverContext::cpu(), layer, plan)
    }

    /// TWO-ROOT ENTRY — prove a top-level [`RelationProofPlanKind::CrossRootHashJoin`] over BOTH
    /// committed graphs. `left` is proven against `schema`, `right` against `instance`; each side's
    /// join-key id column is authenticated to its true `hash_object(IRI)` column against ITS OWN root;
    /// the two hash columns are joined by the audited inner join. The delivered relation lives in the
    /// INSTANCE namespace (its output ids are instance node/value ids), so `layer_root` is the instance
    /// root and the result resolves against the instance dictionary.
    ///
    /// The hash reconciliation is a FULL-FIELD LogUp SEMI-JOIN ([`crate::proof::semijoin`]) — NOT the audited
    /// magnitude-sort inner join, whose key-universe strict-sort caps keys at 64/128 bits (`hash_object`
    /// keys are ~254-bit and too near the BN254 modulus to soundly bit-reduce). The semi-join is
    /// field-element-native (reciprocal membership; no sort, no bit decomposition). SOUNDNESS: every
    /// delivered `?x` is a real instance row whose class-hash is a MEMBER of the closure set — a
    /// same-integer-id/different-IRI pair (different hashes) never matches. COMPLETENESS CAVEAT: the
    /// membership direction is `kept ⊆ closure`; the honest prover keeps every member, but a
    /// cryptographic no-drop guarantee against a malicious prover is a documented follow-up (see
    /// [`crate::proof::semijoin`]).
    pub fn prove_cross_root(
        instance: &LayerCommitment,
        schema: &LayerCommitment,
        plan: &RelationProofPlan,
    ) -> io::Result<Self> {
        Self::prove_cross_root_with_context(&StoreProverContext::cpu(), instance, schema, plan)
    }

    pub fn prove_cross_root_with_context(
        context: &StoreProverContext<'_>,
        instance: &LayerCommitment,
        schema: &LayerCommitment,
        plan: &RelationProofPlan,
    ) -> io::Result<Self> {
        plan.result_shape()?;
        let (left_plan, right_plan, left_key_col, right_key_col) = match &plan.kind {
            RelationProofPlanKind::CrossRootHashJoin { left, right, left_key_col, right_key_col } => {
                (left.as_ref(), right.as_ref(), *left_key_col, *right_key_col)
            }
            _ => return Err(invalid("prove_cross_root requires a CrossRootHashJoin plan")),
        };

        // (1) Prove each child against ITS OWN root (single-root subtrees, standard recursion).
        let mut lnodes = 0usize;
        let left_env = Self::prove_bounded(context, schema, left_plan, 1, &mut lnodes)?;
        let mut rnodes = 0usize;
        let right_env = Self::prove_bounded(context, instance, right_plan, 1, &mut rnodes)?;

        if left_env.result_commitments.len() != 1 || left_key_col != 0 {
            return Err(invalid("cross-root left child must be a single key column (the class set)"));
        }
        // The instance child is exactly `[?x, class]`: one output column plus the class key column.
        if right_env.result_commitments.len() != 2 || right_key_col >= 2 {
            return Err(invalid("cross-root right child must be exactly [output, class-key]"));
        }
        let x_col = 1 - right_key_col;

        let instance_root = instance.state.commitment_root;
        let schema_root = schema.state.commitment_root;
        // β re-anchor: the cross-root anchor bound into the transcript (and checked by the verifier's
        // `validate_database_root`) now combines each graph's `resolution_root`, matching the verifier.
        // `instance_root`/`schema_root` (commitment roots) are still used below for the per-child
        // identity/relocation binding (Option P), so keep them.
        let db_root = crate::proof::commitment::database_root(
            &instance.resolution_root(),
            &schema.resolution_root(),
        );
        let mut transcript = Transcript254::new(b"parture-cross-root-envelope");
        transcript.append_bytes(b"pow/cross-root/db-root", &db_root);
        transcript.append_bytes(b"pow/cross-root/query", &plan.query_digest);

        // (2) Authenticate each side's key id column -> true IRI-hash column, against its OWN root.
        let leftlog = left_env.result_log_domain;
        let mut left_key = left_env.result_columns[0].clone();
        left_key.resize(1usize << leftlog, ArkFr::zero());
        let (left_hash, v_hash_left) = prove_dict_hash_lookup(
            context, &left_key, &left_env.result_commitments[0], leftlog, left_env.result_len,
            schema, &mut transcript,
        )?;

        let rightlog = right_env.result_log_domain;
        let mut class_key = right_env.result_columns[right_key_col].clone();
        class_key.resize(1usize << rightlog, ArkFr::zero());
        let (right_hash, v_hash_right) = prove_dict_hash_lookup(
            context, &class_key, &right_env.result_commitments[right_key_col], rightlog,
            right_env.result_len, instance, &mut transcript,
        )?;

        // (3) Relocate the three authenticated columns UP to a shared working domain 2^D:
        // the closure hash SET, the instance ?x column, and the instance class-hash column. Each real
        // support lives in [0, real_n) so the UP relocation is lossless; each is bound to its
        // root-authenticated source commitment.
        let d_left = v_hash_left.len().trailing_zeros() as usize;
        let d_right = v_hash_right.len().trailing_zeros() as usize;
        let d_log = d_left.max(d_right);
        let d_size = 1usize << d_log;
        let pad_d = |c: &[ArkFr]| -> Vec<ArkFr> {
            let mut v = c.to_vec();
            v.resize(d_size, ArkFr::zero());
            v
        };
        let closure_at_d = pad_d(&v_hash_left);
        let com_closure_at_d = context.commit_column(&closure_at_d)?;
        let relocate_closure = prove_column_relocation_with_context(
            context, &v_hash_left, &left_hash.lookup.com_v_hash, &closure_at_d, &com_closure_at_d, &mut transcript,
        )?;
        let h_at_d = pad_d(&v_hash_right);
        let com_h_at_d = context.commit_column(&h_at_d)?;
        let relocate_h = prove_column_relocation_with_context(
            context, &v_hash_right, &right_hash.lookup.com_v_hash, &h_at_d, &com_h_at_d, &mut transcript,
        )?;
        let mut x_child = right_env.result_columns[x_col].clone();
        x_child.resize(1usize << rightlog, ArkFr::zero());
        let x_at_d = pad_d(&x_child);
        let com_x_at_d = context.commit_column(&x_at_d)?;
        let relocate_x = prove_column_relocation_with_context(
            context, &x_child, &right_env.result_commitments[x_col], &x_at_d, &com_x_at_d, &mut transcript,
        )?;

        // (4) Full-field LogUp SEMI-JOIN (no magnitude sort): keep instance rows whose class-hash is a
        // member of the closure hash set; deliver the kept ?x column.
        let (semijoin, x_kept, m) = prove_hash_semijoin(
            context, &closure_at_d, &com_closure_at_d, left_env.result_len,
            &x_at_d, &com_x_at_d, &h_at_d, &com_h_at_d, right_env.result_len, &mut transcript,
        )?;

        let com_x_kept = semijoin.com_x_kept;
        Self::finish_proved(
            instance,
            plan,
            RecursiveRelationProof::CrossRootHashJoin {
                left: Box::new(left_env),
                right: Box::new(right_env),
                left_hash: Box::new(left_hash),
                right_hash: Box::new(right_hash),
                relocate_closure,
                com_closure_at_d,
                relocate_x,
                com_x_at_d,
                relocate_h,
                com_h_at_d,
                semijoin: Box::new(semijoin),
            },
            vec![com_x_kept],
            vec![x_kept],
            d_log,
            m,
        )
    }

    pub(super) fn prove_bounded(
        context: &StoreProverContext<'_>,
        layer: &LayerCommitment,
        plan: &RelationProofPlan,
        depth: usize,
        nodes: &mut usize,
    ) -> io::Result<Self> {
        if depth > MAX_RELATION_PROOF_DEPTH || *nodes >= MAX_RELATION_PROOF_NODES {
            return Err(invalid("recursive relation proving exceeds structural limits"));
        }
        *nodes += 1;
        match &plan.kind {
            RelationProofPlanKind::Bgp(bgp) => {
                let result = prove_layer_bgp_with_result_and_context(context, layer, bgp)?;
                let result_columns = result.real_columns();
                let leaf = LayerBgpEnvelope::new_verified(
                    layer,
                    layer.state.commitment_root,
                    bgp,
                    plan.query_digest,
                    result.proof,
                    result.result_commitments,
                    result_columns,
                    result.result_log_domain,
                    result.result_len,
                    result.predicate_ids,
                )?;
                Self::finish_proved(
                    layer,
                    plan,
                    RecursiveRelationProof::Bgp(Box::new(leaf.clone())),
                    leaf.result_commitments,
                    leaf.result_columns,
                    leaf.result_log_domain,
                    leaf.result_len,
                )
            }
            RelationProofPlanKind::LayerTripleCount => {
                let mut column = vec![ArkFr::zero(); 2];
                column[0] = ArkFr::from_u64(layer.state.materialized_count);
                let commitment = context.commit_column(&column)?;
                Self::finish_proved(
                    layer,
                    plan,
                    RecursiveRelationProof::LayerTripleCount,
                    vec![commitment],
                    vec![column],
                    1,
                    1,
                )
            }
            RelationProofPlanKind::CollectionExtrema {
                input,
                value_column,
                datatype,
                kind,
            } => {
                let input_envelope =
                    Self::prove_bounded(context, layer, input, depth + 1, nodes)?;
                let mut columns = derived_collection_extrema_column(
                    layer,
                    &input_envelope,
                    *value_column,
                    *datatype,
                    *kind,
                )?;
                let n = columns[0].len();
                columns[0].resize(2, ArkFr::zero());
                let commitments = vec![context.commit_column(&columns[0])?];
                Self::finish_proved(
                    layer,
                    plan,
                    RecursiveRelationProof::CollectionExtrema {
                        input: Box::new(input_envelope),
                    },
                    commitments,
                    columns,
                    1,
                    n,
                )
            }
            RelationProofPlanKind::GeneratedU64Filter {
                input,
                column,
                op,
                constant,
                constant_on_left,
            } => {
                let input_envelope =
                    Self::prove_bounded(context, layer, input, depth + 1, nodes)?;
                let mut columns = generated_u64_filtered_columns(
                    &input_envelope,
                    *column,
                    *op,
                    *constant,
                    *constant_on_left,
                )?;
                let n = columns.first().map_or(0, Vec::len);
                let log_domain = n.max(1).next_power_of_two().max(2).trailing_zeros() as usize;
                let domain = 1usize << log_domain;
                for column in &mut columns {
                    column.resize(domain, ArkFr::zero());
                }
                let commitments = columns
                    .iter()
                    .map(|column| context.commit_column(column))
                    .collect::<io::Result<Vec<_>>>()?;
                Self::finish_proved(
                    layer,
                    plan,
                    RecursiveRelationProof::GeneratedU64Filter {
                        input: Box::new(input_envelope),
                    },
                    commitments,
                    columns,
                    log_domain,
                    n,
                )
            }
            RelationProofPlanKind::CollectionSum {
                input,
                template,
                datatype,
            } => {
                let input_envelope =
                    Self::prove_bounded(context, layer, input, depth + 1, nodes)?;
                let mut columns = derived_collection_sum_column(
                    layer,
                    &input_envelope,
                    *template,
                    *datatype,
                )?;
                columns[0].resize(2, ArkFr::zero());
                let commitments = vec![context.commit_column(&columns[0])?];
                Self::finish_proved(
                    layer,
                    plan,
                    RecursiveRelationProof::CollectionSum {
                        input: Box::new(input_envelope),
                    },
                    commitments,
                    columns,
                    1,
                    1,
                )
            }
            RelationProofPlanKind::CollectionCount { input } => {
                let input_envelope =
                    Self::prove_bounded(context, layer, input, depth + 1, nodes)?;
                let mut columns = derived_cardinality_column(input_envelope.result_len, true)?;
                let n = 1;
                let log_domain = 1;
                for column in &mut columns {
                    column.resize(2, ArkFr::zero());
                }
                let commitments = columns
                    .iter()
                    .map(|column| context.commit_column(column))
                    .collect::<io::Result<Vec<_>>>()?;
                Self::finish_proved(
                    layer,
                    plan,
                    RecursiveRelationProof::CollectionCount {
                        input: Box::new(input_envelope),
                    },
                    commitments,
                    columns,
                    log_domain,
                    n,
                )
            }
            RelationProofPlanKind::GeneratedI64Filter {
                input,
                column,
                op,
                constant,
                constant_on_left,
            } => {
                let input_envelope =
                    Self::prove_bounded(context, layer, input, depth + 1, nodes)?;
                let mut columns = generated_i64_filtered_columns(
                    &input_envelope,
                    *column,
                    *op,
                    *constant,
                    *constant_on_left,
                )?;
                let n = columns.first().map_or(0, Vec::len);
                let log_domain = n.max(1).next_power_of_two().max(2).trailing_zeros() as usize;
                let domain = 1usize << log_domain;
                for column in &mut columns {
                    column.resize(domain, ArkFr::zero());
                }
                let commitments = columns
                    .iter()
                    .map(|column| context.commit_column(column))
                    .collect::<io::Result<Vec<_>>>()?;
                Self::finish_proved(
                    layer,
                    plan,
                    RecursiveRelationProof::GeneratedI64Filter {
                        input: Box::new(input_envelope),
                    },
                    commitments,
                    columns,
                    log_domain,
                    n,
                )
            }
            RelationProofPlanKind::GroupSum {
                input,
                key_columns,
                value_column,
                datatype,
            } => {
                let input_envelope =
                    Self::prove_bounded(context, layer, input, depth + 1, nodes)?;
                let mut columns = derived_group_sum_columns(
                    layer,
                    &input_envelope,
                    key_columns,
                    *value_column,
                    *datatype,
                )?;
                let n = columns.first().map_or(0, Vec::len);
                let log_domain = n.max(1).next_power_of_two().max(2).trailing_zeros() as usize;
                let domain = 1usize << log_domain;
                for column in &mut columns {
                    column.resize(domain, ArkFr::zero());
                }
                let commitments = columns
                    .iter()
                    .map(|column| context.commit_column(column))
                    .collect::<io::Result<Vec<_>>>()?;
                Self::finish_proved(
                    layer,
                    plan,
                    RecursiveRelationProof::GroupSum {
                        input: Box::new(input_envelope),
                    },
                    commitments,
                    columns,
                    log_domain,
                    n,
                )
            }
            RelationProofPlanKind::GroupCount { input, key_columns } => {
                let input_envelope =
                    Self::prove_bounded(context, layer, input, depth + 1, nodes)?;
                let mut columns =
                    derived_group_count_columns(layer, &input_envelope, key_columns)?;
                let n = columns.first().map_or(0, Vec::len);
                let log_domain = n.max(1).next_power_of_two().max(2).trailing_zeros() as usize;
                let domain = 1usize << log_domain;
                for column in &mut columns {
                    column.resize(domain, ArkFr::zero());
                }
                let commitments = columns
                    .iter()
                    .map(|column| context.commit_column(column))
                    .collect::<io::Result<Vec<_>>>()?;
                Self::finish_proved(
                    layer,
                    plan,
                    RecursiveRelationProof::GroupCount {
                        input: Box::new(input_envelope),
                    },
                    commitments,
                    columns,
                    log_domain,
                    n,
                )
            }
            RelationProofPlanKind::Distinct { input, key_columns } => {
                let input_envelope =
                    Self::prove_bounded(context, layer, input, depth + 1, nodes)?;
                let witness = input_envelope.relation_witness(None)?;
                let view = layer.verifier_view();
                let resolve_id = |field: ArkFr, namespace: ResultIdNamespace| -> io::Result<u64> {
                    match namespace {
                        ResultIdNamespace::Predicate => view
                            .predicate_catalog
                            .iter()
                            .find(|record| ArkFr::from_u64(record.predicate_id) == field)
                            .map(|record| record.predicate_id),
                        ResultIdNamespace::NodeValue => view
                            .node_value_catalog
                            .iter()
                            .find(|record| ArkFr::from_u64(record.object_id) == field)
                            .map(|record| record.object_id),
                        ResultIdNamespace::GeneratedDecimalU64
                        | ResultIdNamespace::GeneratedDecimalI64 => None,
                    }
                    .ok_or_else(|| invalid("distinct input id absent from planned namespace"))
                };
                let mut seen = std::collections::BTreeSet::new();
                let mut rows = Vec::new();
                for row in 0..witness.n {
                    let tuple = key_columns
                        .iter()
                        .map(|column| {
                            resolve_id(
                                witness.columns[*column][row],
                                input_envelope.result_descriptors[*column].namespace,
                            )
                        })
                        .collect::<io::Result<Vec<_>>>()?;
                    if seen.insert(tuple.clone()) {
                        rows.push(tuple);
                    }
                }
                let mut transcript = Transcript254::new(b"parture-distinct-envelope");
                transcript.append_bytes(b"pow/distinct-envelope/root", &layer.state.commitment_root);
                transcript.append_bytes(b"pow/distinct-envelope/query", &plan.query_digest);
                let result = prove_relation_distinct_with_context(
                    context,
                    &witness,
                    key_columns,
                    &rows,
                    &mut transcript,
                )?;
                Self::finish_proved(
                    layer,
                    plan,
                    RecursiveRelationProof::Distinct {
                        input: Box::new(input_envelope),
                        proof: Box::new(result.proof),
                    },
                    result.commitments,
                    result.columns,
                    result.log_domain,
                    result.n,
                )
            }
            RelationProofPlanKind::BagUnion { branches, mappings } => {
                let mut child_envelopes = Vec::with_capacity(branches.len());
                let mut witnesses = Vec::with_capacity(branches.len());
                let (_, ground) = plan.result_shape()?;
                for ((branch, mapping), _) in branches.iter().zip(mappings).zip(0..) {
                    let child = Self::prove_bounded(context, layer, branch, depth + 1, nodes)?;
                    witnesses.push(child.relation_witness((!ground).then_some(mapping.as_slice()))?);
                    child_envelopes.push(child);
                }
                let mut transcript = Transcript254::new(b"parture-union-envelope");
                transcript.append_bytes(b"pow/union-envelope/root", &layer.state.commitment_root);
                transcript.append_bytes(b"pow/union-envelope/query", &plan.query_digest);
                let result = prove_bag_union_with_context(context, &witnesses, &mut transcript)?;
                Self::finish_proved(layer, plan, RecursiveRelationProof::BagUnion {
                    branches: child_envelopes,
                    proof: Box::new(result.proof),
                }, result.commitments, result.columns, result.log_domain, result.n)
            }
            RelationProofPlanKind::BagProjection { input, indices } => {
                let input_envelope =
                    Self::prove_bounded(context, layer, input, depth + 1, nodes)?;
                let log_domain = input_envelope.result_log_domain;
                let n = input_envelope.result_len;
                let (commitments, columns) = if indices.is_empty() {
                    let unit = ground_unit_column(n, log_domain);
                    (vec![context.commit_column(&unit)?], vec![unit])
                } else {
                    let commitments = indices
                        .iter()
                        .map(|index| {
                            input_envelope
                                .result_commitments
                                .get(*index)
                                .copied()
                                .ok_or_else(|| invalid("bag projection index out of range"))
                        })
                        .collect::<io::Result<Vec<_>>>()?;
                    let columns = indices
                        .iter()
                        .map(|index| {
                            input_envelope
                                .result_columns
                                .get(*index)
                                .cloned()
                                .ok_or_else(|| invalid("bag projection index out of range"))
                        })
                        .collect::<io::Result<Vec<_>>>()?;
                    (commitments, columns)
                };
                Self::finish_proved(
                    layer,
                    plan,
                    RecursiveRelationProof::BagProjection {
                        input: Box::new(input_envelope),
                    },
                    commitments,
                    columns,
                    log_domain,
                    n,
                )
            }
            RelationProofPlanKind::LeftAnti { left, right, key_pairs } => {
                let left_envelope = Self::prove_bounded(context, layer, left, depth + 1, nodes)?;
                let right_envelope = Self::prove_bounded(context, layer, right, depth + 1, nodes)?;
                let left_witness = left_envelope.relation_witness(None)?;
                let right_witness = right_envelope.relation_witness(None)?;
                let (proof, commitments, columns, log_domain, n) = if key_pairs.is_empty() {
                    let (commitments, columns, log_domain, n) =
                        prove_global_not_with_context(context, &left_witness, &right_witness)?;
                    (None, commitments, columns, log_domain, n)
                } else {
                    let mut transcript = Transcript254::new(b"parture-left-anti-envelope");
                    transcript.append_bytes(b"pow/anti-envelope/root", &layer.state.commitment_root);
                    transcript.append_bytes(b"pow/anti-envelope/query", &plan.query_digest);
                    let result = prove_left_anti_join_with_keys_and_context(
                        context, &left_witness, &right_witness, key_pairs, &mut transcript,
                    )?;
                    (Some(Box::new(result.proof)), result.commitments, result.columns, result.log_domain, result.n)
                };
                Self::finish_proved(layer, plan, RecursiveRelationProof::LeftAnti {
                    left: Box::new(left_envelope), right: Box::new(right_envelope), proof,
                }, commitments, columns, log_domain, n)
            }
            RelationProofPlanKind::InnerJoin { left, right, key_pairs } => {
                let left_envelope = Self::prove_bounded(context, layer, left, depth + 1, nodes)?;
                let right_envelope = Self::prove_bounded(context, layer, right, depth + 1, nodes)?;
                let mut transcript = Transcript254::new(b"parture-inner-join-envelope");
                transcript.append_bytes(b"pow/inner-envelope/root", &layer.state.commitment_root);
                transcript.append_bytes(b"pow/inner-envelope/query", &plan.query_digest);
                let result = prove_inner_join_with_keys_and_context(
                    context, &left_envelope.relation_witness(None)?,
                    &right_envelope.relation_witness(None)?, key_pairs, &mut transcript,
                )?;
                Self::finish_proved(layer, plan, RecursiveRelationProof::InnerJoin {
                    left: Box::new(left_envelope), right: Box::new(right_envelope),
                    proof: Box::new(result.proof),
                }, result.commitments, result.columns, result.log_domain, result.n)
            }
            RelationProofPlanKind::CrossRootHashJoin { .. } => {
                // The cross-root node reads TWO graphs; the single-layer recursion has only one
                // layer. It is TOP-LEVEL only, proven via `prove_cross_root` (two-layer). Reject
                // closed here so it can never be nested inside a single-root subtree.
                Err(invalid(
                    "cross-root hash join must be proven via prove_cross_root (two layers), not nested in the single-root recursion",
                ))
            }
            RelationProofPlanKind::LeftOuter { left, right, key_pairs } => {
                let left_envelope = Self::prove_bounded(context, layer, left, depth + 1, nodes)?;
                let right_envelope = Self::prove_bounded(context, layer, right, depth + 1, nodes)?;
                let left_witness = left_envelope.relation_witness(None)?;
                let right_witness = right_envelope.relation_witness(None)?;
                let mut transcript = Transcript254::new(b"parture-left-outer-envelope");
                transcript.append_bytes(b"pow/outer-envelope/root", &layer.state.commitment_root);
                transcript.append_bytes(b"pow/outer-envelope/query", &plan.query_digest);
                let result = if key_pairs.is_empty() {
                    prove_global_left_outer_join_with_context(context, &left_witness, &right_witness, &mut transcript)?
                } else {
                    prove_left_outer_join_with_keys_and_context(context, &left_witness, &right_witness, key_pairs, &mut transcript)?
                };
                Self::finish_proved(layer, plan, RecursiveRelationProof::LeftOuter {
                    left: Box::new(left_envelope), right: Box::new(right_envelope), proof: Box::new(result.proof),
                }, result.commitments, result.columns, result.log_domain, result.n)
            }
            RelationProofPlanKind::Equality { input, constraint } => {
                let input_envelope = Self::prove_bounded(context, layer, input, depth + 1, nodes)?;
                let resolved = resolve_relation_equality_constraint(layer, constraint)?;
                let mut transcript = Transcript254::new(b"parture-equality-envelope");
                transcript.append_bytes(b"pow/equality-envelope/root", &layer.state.commitment_root);
                transcript.append_bytes(b"pow/equality-envelope/query", &plan.query_digest);
                let result = prove_relation_equality_with_context(
                    context, &input_envelope.relation_witness(None)?, &resolved, &mut transcript,
                )?;
                Self::finish_proved(layer, plan, RecursiveRelationProof::Equality {
                    input: Box::new(input_envelope), proof: Box::new(result.proof),
                }, result.commitments, result.columns, result.log_domain, result.n)
            }
            RelationProofPlanKind::ExactTupleDisequality { input, pairs } => {
                let input_envelope = Self::prove_bounded(context, layer, input, depth + 1, nodes)?;
                let resolved = resolve_tuple_disequality_pairs(layer, pairs)?;
                let mut transcript = Transcript254::new(b"parture-tuple-disequality-envelope");
                transcript.append_bytes(b"pow/tuple-disequality-envelope/root", &layer.state.commitment_root);
                transcript.append_bytes(b"pow/tuple-disequality-envelope/query", &plan.query_digest);
                let result = prove_tuple_disequality_with_context(
                    context, &input_envelope.relation_witness(None)?, &resolved, &mut transcript,
                )?;
                Self::finish_proved(layer, plan, RecursiveRelationProof::TupleDisequality {
                    input: Box::new(input_envelope), proof: Box::new(result.proof),
                }, result.commitments, result.columns, result.log_domain, result.n)
            }
            RelationProofPlanKind::ExactScalarOrder { input, datatype, direction } => {
                let input_envelope =
                    Self::prove_bounded(context, layer, input, depth + 1, nodes)?;
                let mut transcript = Transcript254::new(b"parture-order-envelope");
                transcript.append_bytes(b"pow/order-envelope/root", &layer.state.commitment_root);
                transcript.append_bytes(b"pow/order-envelope/query", &plan.query_digest);
                let result = prove_exact_scalar_order_with_context(
                    context,
                    layer,
                    &input_envelope.relation_witness(None)?,
                    *datatype,
                    *direction,
                    &mut transcript,
                )?;
                Self::finish_proved(
                    layer,
                    plan,
                    RecursiveRelationProof::ExactScalarOrder {
                        input: Box::new(input_envelope),
                        proof: Box::new(result.proof),
                    },
                    result.commitments,
                    result.columns,
                    result.log_domain,
                    result.n,
                )
            }
            RelationProofPlanKind::ExactMultiOrder { input, datatypes, key_indices, directions } => {
                let input_envelope = Self::prove_bounded(context, layer, input, depth + 1, nodes)?;
                let mut transcript = Transcript254::new(b"parture-multi-order-envelope");
                transcript.append_bytes(b"pow/multi-order-envelope/root", &layer.state.commitment_root);
                transcript.append_bytes(b"pow/multi-order-envelope/query", &plan.query_digest);
                let result = prove_exact_multi_order_with_context(context, layer, &input_envelope.relation_witness(None)?, datatypes, key_indices, directions, &mut transcript)?;
                Self::finish_proved(layer, plan, RecursiveRelationProof::ExactMultiOrder { input: Box::new(input_envelope), proof: Box::new(result.proof) }, result.commitments, result.columns, result.log_domain, result.n)
            }
            RelationProofPlanKind::ExactOrderedSlice {
                input,
                offset,
                limit,
            } => {
                let input_envelope =
                    Self::prove_bounded(context, layer, input, depth + 1, nodes)?;
                let (start, end) = ordered_slice_bounds(input_envelope.result_len, *offset, *limit);
                let take = end - start;
                let log_domain = take.max(1).next_power_of_two().max(2).trailing_zeros() as usize;
                let domain = 1usize << log_domain;
                let mut columns = input_envelope
                    .result_columns
                    .iter()
                    .map(|column| column[start..end].to_vec())
                    .collect::<Vec<_>>();
                for column in &mut columns {
                    column.resize(domain, ArkFr::zero());
                }
                let commitments = columns
                    .iter()
                    .map(|column| context.commit_column(column))
                    .collect::<io::Result<Vec<_>>>()?;
                Self::finish_proved(
                    layer,
                    plan,
                    RecursiveRelationProof::ExactOrderedSlice {
                        input: Box::new(input_envelope),
                    },
                    commitments,
                    columns,
                    log_domain,
                    take,
                )
            }
            RelationProofPlanKind::GroupExtrema { input, key_columns, value_column, datatype, kind } => {
                let input_envelope = Self::prove_bounded(context, layer, input, depth + 1, nodes)?;
                let mut columns = derived_group_extrema_columns(
                    layer, &input_envelope, key_columns, *value_column, *datatype, *kind,
                )?;
                let n = columns.first().map_or(0, Vec::len);
                let log_domain = n.max(1).next_power_of_two().max(2).trailing_zeros() as usize;
                let domain = 1usize << log_domain;
                for column in &mut columns {
                    column.resize(domain, ArkFr::zero());
                }
                let commitments = columns.iter()
                    .map(|column| context.commit_column(column))
                    .collect::<io::Result<Vec<_>>>()?;
                Self::finish_proved(layer, plan, RecursiveRelationProof::GroupExtrema {
                    input: Box::new(input_envelope),
                }, commitments, columns, log_domain, n)
            }
            RelationProofPlanKind::TransitiveClosure { edge, seed, reflexive, inverted } => {
                Self::prove_transitive_closure(
                    context, layer, plan, edge, seed, *reflexive, *inverted, depth, nodes,
                )
            }
            RelationProofPlanKind::TransitiveClosurePairs { edge, inverted } => {
                Self::prove_transitive_closure_pairs(
                    context, layer, plan, edge, *inverted, depth, nodes,
                )
            }
            RelationProofPlanKind::OrderedListWalk { subject, predicate } => {
                super::list_walk::prove_ordered_list_walk(
                    context, layer, plan, subject, predicate, depth, nodes,
                )
            }
            RelationProofPlanKind::OrderedArrayWalk { subject, predicate } => {
                super::array_walk::prove_ordered_array_walk(
                    context, layer, plan, subject, predicate, depth, nodes,
                )
            }
        }
    }

    /// Prove a bounded semi-naive transitive closure. Runs the accumulate-and-dedupe loop until the
    /// one-hop frontier stops growing (the empty-delta fixpoint), building each round from the audited
    /// operator cores and threading `A_k` forward by commitment. See
    /// [`RelationProofPlanKind::TransitiveClosure`].
    #[allow(clippy::too_many_arguments)]
    fn prove_transitive_closure(
        context: &StoreProverContext<'_>,
        layer: &LayerCommitment,
        plan: &RelationProofPlan,
        edge: &RelationProofPlan,
        seed: &CanonicalObject,
        reflexive: bool,
        inverted: bool,
        depth: usize,
        nodes: &mut usize,
    ) -> io::Result<Self> {
        let root = layer.state.commitment_root;
        let digest = plan.query_digest;

        // (1) Prove the edge relation E ONCE. `E` is a general authenticated 2-column relation (a
        //     single-predicate `Bgp`, or a compound derived edge — a deduped bag-union / projected
        //     join); the closure is SOURCE-AGNOSTIC over it, taking it BY COMMITMENT and reusing it in
        //     every round and the fixpoint. The edge plan carries its own K-INDEPENDENT query digest
        //     (the compiler binds `E`'s structure into the closure digest), so the verifier reconstructs
        //     it from its own trusted query.
        let edge_env = Self::prove_bounded(context, layer, edge, depth + 1, nodes)?;
        let edge_witness = edge_env.relation_witness(None)?;

        // (2) Resolve the constant seed `a` (source-specific: single-layer dictionary).
        let view = layer.verifier_view();
        let seed_id = resolve_closure_seed_id(&view, seed)?;

        // (3)-(4) The SOURCE-AGNOSTIC seed + rounds + fixpoint (shared with the reconciled closure).
        let ClosureRoundsWitness { seed: seed_proof, accumulator, rounds, fixpoint, reachable_ids: _ } =
            prove_closure_rounds(context, &edge_witness, seed_id, root, digest, reflexive, inverted, nodes)?;

        // (5) Deliver A_K = accumulator as the reachable SET (one nonnullable NodeValue column).
        let n = accumulator.n;
        let column = accumulator.columns[0][..n].to_vec();
        let commitment = accumulator.commitments[0];
        Self::finish_proved(
            layer,
            plan,
            RecursiveRelationProof::TransitiveClosure {
                edge: Box::new(edge_env),
                seed: seed_proof,
                rounds,
                fixpoint,
            },
            vec![commitment],
            vec![column],
            accumulator.log_domain,
            n,
        )
    }

    /// Prove a bounded ALL-PAIRS semi-naive transitive closure (`path(?x, plus(p), ?y)`, both free).
    /// Delivers the 2-column reachable-pairs relation `A_K`. See
    /// [`RelationProofPlanKind::TransitiveClosurePairs`].
    #[allow(clippy::too_many_arguments)]
    fn prove_transitive_closure_pairs(
        context: &StoreProverContext<'_>,
        layer: &LayerCommitment,
        plan: &RelationProofPlan,
        edge: &RelationProofPlan,
        inverted: bool,
        depth: usize,
        nodes: &mut usize,
    ) -> io::Result<Self> {
        let root = layer.state.commitment_root;
        let digest = plan.query_digest;

        // (1) Prove the edge relation E ONCE. `E` is a general authenticated 2-column relation (a
        //     single-predicate `Bgp`, or a DERIVED all-pairs edge — a wildcard/compound projected join);
        //     the pairs closure is SOURCE-AGNOSTIC over it, taking it BY COMMITMENT and reusing it in the
        //     seed, every round, and the fixpoint. The edge plan carries its own K-INDEPENDENT query digest
        //     (the compiler binds `E`'s structure into the closure digest), so the verifier reconstructs it
        //     from its own trusted query.
        let edge_env = Self::prove_bounded(context, layer, edge, depth + 1, nodes)?;
        let edge_witness = edge_env.relation_witness(None)?;

        // (2)-(4) seed (A_0 = deduped one-hop pairs) + rounds + empty-delta fixpoint.
        let ClosurePairsRoundsWitness {
            seed,
            accumulator,
            rounds,
            fixpoint,
        } = prove_closure_pairs_rounds(context, &edge_witness, root, digest, inverted, nodes)?;

        // (5) Deliver A_K = the 2-column reachable-pairs relation [src, dst].
        let n = accumulator.n;
        let columns = vec![
            accumulator.columns[0][..n].to_vec(),
            accumulator.columns[1][..n].to_vec(),
        ];
        let commitments = accumulator.commitments.clone();
        Self::finish_proved(
            layer,
            plan,
            RecursiveRelationProof::TransitiveClosurePairs {
                edge: Box::new(edge_env),
                seed,
                rounds,
                fixpoint,
            },
            commitments,
            columns,
            accumulator.log_domain,
            n,
        )
    }

    pub(super) fn verify_bounded(
        &self,
        layer: &impl VerifierLayer,
        expected_root: ProofHash,
        plan: &RelationProofPlan,
        depth: usize,
        nodes: &mut usize,
    ) -> io::Result<()> {
        if depth > MAX_RELATION_PROOF_DEPTH || *nodes >= MAX_RELATION_PROOF_NODES {
            return Err(invalid("recursive relation verification exceeds structural limits"));
        }
        *nodes += 1;
        // CROSS-PATH GUARD (E1): reject a TERM-LESS DELTA view on this single-layer envelope path.
        // The top-level `verify` fail-closes ALL deltas earlier via `require_base_layer_for_query`,
        // but the two-root `verify_cross_root` reaches `verify_bounded` on each per-graph side
        // WITHOUT that gate. A term-less delta is now constructible/decodable (E1), so guard the
        // single-layer choke itself: a delta is an increment, not the full resolved state, and must
        // not verify against one layer's own root. Inert for every term-bearing layer and for a
        // term-less BASE view (`parent_state_root == None`); only a term-less delta trips it. The
        // reconciled/delta-read path (E2) is where delta querying is opened.
        let view = layer.verifier_view();
        if view.state.parent_state_root.is_some() && !view.terms_present {
            return Err(invalid(
                "term-less delta layer cannot be verified on the single-layer base/rollup envelope path \
                 (a delta is an increment, not the full state); verify it through the reconciled/delta-read path",
            ));
        }
        if self.layer_root != expected_root {
            return Err(invalid("recursive relation root does not match trusted root"));
        }
        if self.query_digest != plan.query_digest {
            return Err(invalid("recursive relation query identity mismatch"));
        }
        let (expected_descriptors, expected_ground) = plan.result_shape()?;
        if self.ground_result != expected_ground
            || self.result_descriptors != expected_descriptors
        {
            return Err(invalid("recursive relation result schema mismatch"));
        }

        match (&self.proof, &plan.kind) {
            (
                RecursiveRelationProof::LayerTripleCount,
                RelationProofPlanKind::LayerTripleCount,
            ) => {
                let count = layer.verifier_view().state.materialized_count;
                let mut column = vec![ArkFr::zero(); 2];
                column[0] = ArkFr::from_u64(count);
                if self.result_len != 1
                    || self.result_log_domain != 1
                    || self.result_columns != vec![vec![column[0]]]
                    || self.result_commitments != vec![commit_column(&column)?]
                    || self.result_values
                        != vec![vec![CanonicalResultValue::GeneratedDecimalU64(count)]]
                {
                    return Err(invalid("layer triple-count result does not match trusted root"));
                }
            }
            (
                RecursiveRelationProof::GeneratedU64Filter { input },
                RelationProofPlanKind::GeneratedU64Filter {
                    input: input_plan,
                    column,
                    op,
                    constant,
                    constant_on_left,
                },
            ) => {
                input.verify_bounded(
                    layer,
                    expected_root,
                    input_plan,
                    depth + 1,
                    nodes,
                )?;
                let expected_columns = generated_u64_filtered_columns(
                    input,
                    *column,
                    *op,
                    *constant,
                    *constant_on_left,
                )?;
                let expected_len = expected_columns.first().map_or(0, Vec::len);
                let expected_log = expected_len
                    .max(1)
                    .next_power_of_two()
                    .max(2)
                    .trailing_zeros() as usize;
                if self.result_len != expected_len
                    || self.result_log_domain != expected_log
                    || self.result_columns != expected_columns
                {
                    return Err(invalid(
                        "generated-u64 filter result is not the exact derived selection",
                    ));
                }
                let domain = 1usize << expected_log;
                let expected_commitments = expected_columns
                    .iter()
                    .map(|column| {
                        let mut padded = column.clone();
                        padded.resize(domain, ArkFr::zero());
                        commit_column(&padded)
                    })
                    .collect::<io::Result<Vec<_>>>()?;
                if self.result_commitments != expected_commitments {
                    return Err(invalid("generated-u64 filter commitment mismatch"));
                }
            }
            (
                RecursiveRelationProof::GeneratedI64Filter { input },
                RelationProofPlanKind::GeneratedI64Filter {
                    input: input_plan,
                    column,
                    op,
                    constant,
                    constant_on_left,
                },
            ) => {
                input.verify_bounded(
                    layer,
                    expected_root,
                    input_plan,
                    depth + 1,
                    nodes,
                )?;
                let expected_columns = generated_i64_filtered_columns(
                    input,
                    *column,
                    *op,
                    *constant,
                    *constant_on_left,
                )?;
                let expected_len = expected_columns.first().map_or(0, Vec::len);
                let expected_log = expected_len
                    .max(1)
                    .next_power_of_two()
                    .max(2)
                    .trailing_zeros() as usize;
                if self.result_len != expected_len
                    || self.result_log_domain != expected_log
                    || self.result_columns != expected_columns
                {
                    return Err(invalid(
                        "generated-i64 filter result is not the exact derived selection",
                    ));
                }
                let domain = 1usize << expected_log;
                let expected_commitments = expected_columns
                    .iter()
                    .map(|column| {
                        let mut padded = column.clone();
                        padded.resize(domain, ArkFr::zero());
                        commit_column(&padded)
                    })
                    .collect::<io::Result<Vec<_>>>()?;
                if self.result_commitments != expected_commitments {
                    return Err(invalid("generated-i64 filter commitment mismatch"));
                }
            }
            (
                RecursiveRelationProof::GroupSum { input },
                RelationProofPlanKind::GroupSum {
                    input: input_plan,
                    key_columns,
                    value_column,
                    datatype,
                },
            ) => {
                input.verify_bounded(
                    layer,
                    expected_root,
                    input_plan,
                    depth + 1,
                    nodes,
                )?;
                let expected_columns = derived_group_sum_columns(
                    layer,
                    input,
                    key_columns,
                    *value_column,
                    *datatype,
                )?;
                let expected_len = expected_columns.first().map_or(0, Vec::len);
                let expected_log = expected_len
                    .max(1)
                    .next_power_of_two()
                    .max(2)
                    .trailing_zeros() as usize;
                if self.result_len != expected_len
                    || self.result_log_domain != expected_log
                    || self.result_columns != expected_columns
                {
                    return Err(invalid("group-sum result is not the exact derived aggregate"));
                }
                let domain = 1usize << expected_log;
                let expected_commitments = expected_columns
                    .iter()
                    .map(|column| {
                        let mut padded = column.clone();
                        padded.resize(domain, ArkFr::zero());
                        commit_column(&padded)
                    })
                    .collect::<io::Result<Vec<_>>>()?;
                if self.result_commitments != expected_commitments {
                    return Err(invalid("group-sum commitment mismatch"));
                }
            }
            (
                RecursiveRelationProof::GroupCount { input },
                RelationProofPlanKind::GroupCount {
                    input: input_plan,
                    key_columns,
                },
            ) => {
                input.verify_bounded(
                    layer,
                    expected_root,
                    input_plan,
                    depth + 1,
                    nodes,
                )?;
                let expected_columns =
                    derived_group_count_columns(layer, input, key_columns)?;
                let expected_len = expected_columns.first().map_or(0, Vec::len);
                let expected_log = expected_len
                    .max(1)
                    .next_power_of_two()
                    .max(2)
                    .trailing_zeros() as usize;
                if self.result_len != expected_len
                    || self.result_log_domain != expected_log
                    || self.result_columns != expected_columns
                {
                    return Err(invalid("group-count result is not the exact derived aggregate"));
                }
                let domain = 1usize << expected_log;
                let expected_commitments = expected_columns
                    .iter()
                    .map(|column| {
                        let mut padded = column.clone();
                        padded.resize(domain, ArkFr::zero());
                        commit_column(&padded)
                    })
                    .collect::<io::Result<Vec<_>>>()?;
                if self.result_commitments != expected_commitments {
                    return Err(invalid("group-count commitment mismatch"));
                }
            }
            (
                RecursiveRelationProof::CollectionCount { input },
                RelationProofPlanKind::CollectionCount { input: input_plan },
            ) => {
                input.verify_bounded(
                    layer,
                    expected_root,
                    input_plan,
                    depth + 1,
                    nodes,
                )?;
                let expected_columns = derived_cardinality_column(input.result_len, true)?;
                if self.result_len != 1
                    || self.result_log_domain != 1
                    || self.result_columns != expected_columns
                {
                    return Err(invalid(
                        "collection-count result is not the exact derived cardinality",
                    ));
                }
                let mut padded = expected_columns[0].clone();
                padded.resize(2, ArkFr::zero());
                if self.result_commitments != vec![commit_column(&padded)?] {
                    return Err(invalid("collection-count commitment mismatch"));
                }
            }
            (
                RecursiveRelationProof::CollectionSum { input },
                RelationProofPlanKind::CollectionSum {
                    input: input_plan,
                    template,
                    datatype,
                },
            ) => {
                input.verify_bounded(
                    layer,
                    expected_root,
                    input_plan,
                    depth + 1,
                    nodes,
                )?;
                let expected_columns =
                    derived_collection_sum_column(layer, input, *template, *datatype)?;
                if self.result_len != 1
                    || self.result_log_domain != 1
                    || self.result_columns != expected_columns
                {
                    return Err(invalid(
                        "collection-sum result is not the exact derived aggregate",
                    ));
                }
                let mut padded = expected_columns[0].clone();
                padded.resize(2, ArkFr::zero());
                if self.result_commitments != vec![commit_column(&padded)?] {
                    return Err(invalid("collection-sum commitment mismatch"));
                }
            }
            (
                RecursiveRelationProof::CollectionExtrema { input },
                RelationProofPlanKind::CollectionExtrema {
                    input: input_plan,
                    value_column,
                    datatype,
                    kind,
                },
            ) => {
                input.verify_bounded(
                    layer,
                    expected_root,
                    input_plan,
                    depth + 1,
                    nodes,
                )?;
                let expected_columns = derived_collection_extrema_column(
                    layer,
                    input,
                    *value_column,
                    *datatype,
                    *kind,
                )?;
                let expected_len = expected_columns[0].len();
                if self.result_len != expected_len
                    || self.result_log_domain != 1
                    || self.result_columns != expected_columns
                {
                    return Err(invalid(
                        "collection-extrema result is not the exact derived extremum",
                    ));
                }
                let mut padded = expected_columns[0].clone();
                padded.resize(2, ArkFr::zero());
                if self.result_commitments != vec![commit_column(&padded)?] {
                    return Err(invalid("collection-extrema commitment mismatch"));
                }
            }
            (RecursiveRelationProof::Bgp(envelope), RelationProofPlanKind::Bgp(bgp)) => {
                envelope.verify(layer, expected_root, bgp, plan.query_digest)?;
                if self.result_commitments != envelope.result_commitments
                    || self.result_columns != envelope.result_columns
                    || self.result_log_domain != envelope.result_log_domain
                    || self.result_len != envelope.result_len
                {
                    return Err(invalid("recursive BGP result does not match leaf envelope"));
                }
            }
            (
                RecursiveRelationProof::BagUnion { branches, proof },
                RelationProofPlanKind::BagUnion {
                    branches: branch_plans,
                    mappings,
                },
            ) => {
                if branches.len() < 2
                    || branches.len() != branch_plans.len()
                    || branches.len() != mappings.len()
                {
                    return Err(invalid("recursive bag-union branch count mismatch"));
                }
                let mut inputs = Vec::with_capacity(branches.len());
                for ((branch, branch_plan), mapping) in
                    branches.iter().zip(branch_plans).zip(mappings)
                {
                    branch.verify_bounded(
                        layer,
                        expected_root,
                        branch_plan,
                        depth + 1,
                        nodes,
                    )?;
                    inputs.push(if expected_ground {
                        branch.relation_commit(None)?
                    } else {
                        branch.relation_commit(Some(mapping))?
                    });
                }
                let mut transcript =
                    crate::proof::argument::Transcript254::new(b"parture-union-envelope");
                transcript.append_bytes(b"pow/union-envelope/root", &expected_root);
                transcript.append_bytes(b"pow/union-envelope/query", &plan.query_digest);
                verify_bag_union(
                    &inputs,
                    &self.result_commitments,
                    self.result_len,
                    self.result_log_domain,
                    proof,
                    &mut transcript,
                )?;
            }
            (
                RecursiveRelationProof::BagProjection { input },
                RelationProofPlanKind::BagProjection {
                    input: input_plan,
                    indices,
                },
            ) => {
                input.verify_bounded(
                    layer,
                    expected_root,
                    input_plan,
                    depth + 1,
                    nodes,
                )?;
                let (expected_commitments, expected_columns) = if indices.is_empty() {
                    let unit = ground_unit_column(input.result_len, input.result_log_domain);
                    (
                        vec![commit_column(&unit)?],
                        vec![unit[..input.result_len].to_vec()],
                    )
                } else {
                    let commitments = indices
                        .iter()
                        .map(|index| {
                            input
                                .result_commitments
                                .get(*index)
                                .copied()
                                .ok_or_else(|| invalid("bag projection index out of range"))
                        })
                        .collect::<io::Result<Vec<_>>>()?;
                    let columns = indices
                        .iter()
                        .map(|index| {
                            input
                                .result_columns
                                .get(*index)
                                .cloned()
                                .ok_or_else(|| invalid("bag projection index out of range"))
                        })
                        .collect::<io::Result<Vec<_>>>()?;
                    (commitments, columns)
                };
                if self.result_commitments != expected_commitments
                    || self.result_columns != expected_columns
                    || self.result_log_domain != input.result_log_domain
                    || self.result_len != input.result_len
                {
                    return Err(invalid("bag projection changed rows, domain, or multiplicity"));
                }
            }
            (
                RecursiveRelationProof::Distinct { input, proof },
                RelationProofPlanKind::Distinct {
                    input: input_plan,
                    key_columns,
                },
            ) => {
                input.verify_bounded(layer, expected_root, input_plan, depth + 1, nodes)?;
                if self.result_columns.len() != key_columns.len()
                    || self
                        .result_columns
                        .iter()
                        .any(|column| self.result_len > column.len())
                {
                    return Err(invalid("distinct result column shape mismatch"));
                }
                let view = layer.verifier_view();
                let resolve_id = |field: ArkFr, namespace: ResultIdNamespace| -> io::Result<u64> {
                    match namespace {
                        ResultIdNamespace::Predicate => view
                            .predicate_catalog
                            .iter()
                            .find(|record| ArkFr::from_u64(record.predicate_id) == field)
                            .map(|record| record.predicate_id),
                        ResultIdNamespace::NodeValue => view
                            .node_value_catalog
                            .iter()
                            .find(|record| ArkFr::from_u64(record.object_id) == field)
                            .map(|record| record.object_id),
                        ResultIdNamespace::GeneratedDecimalU64
                        | ResultIdNamespace::GeneratedDecimalI64 => None,
                    }
                    .ok_or_else(|| invalid("distinct output id absent from planned namespace"))
                };
                let rows = (0..self.result_len)
                    .map(|row| {
                        self.result_columns
                            .iter()
                            .zip(&expected_descriptors)
                            .map(|(column, descriptor)| resolve_id(column[row], descriptor.namespace))
                            .collect::<io::Result<Vec<_>>>()
                    })
                    .collect::<io::Result<Vec<_>>>()?;
                let mut transcript = Transcript254::new(b"parture-distinct-envelope");
                transcript.append_bytes(b"pow/distinct-envelope/root", &expected_root);
                transcript.append_bytes(b"pow/distinct-envelope/query", &plan.query_digest);
                verify_relation_distinct(
                    &input.relation_commit(None)?,
                    key_columns,
                    &rows,
                    proof,
                    &mut transcript,
                )?;
            }
            (
                RecursiveRelationProof::LeftAnti { left, right, proof },
                RelationProofPlanKind::LeftAnti {
                    left: left_plan,
                    right: right_plan,
                    key_pairs,
                },
            ) => {
                left.verify_bounded(
                    layer,
                    expected_root,
                    left_plan,
                    depth + 1,
                    nodes,
                )?;
                right.verify_bounded(
                    layer,
                    expected_root,
                    right_plan,
                    depth + 1,
                    nodes,
                )?;
                let left_commit = left.relation_commit(None)?;
                let right_commit = right.relation_commit(None)?;
                let mut transcript =
                    crate::proof::argument::Transcript254::new(b"parture-left-anti-envelope");
                transcript.append_bytes(b"pow/anti-envelope/root", &expected_root);
                transcript.append_bytes(b"pow/anti-envelope/query", &plan.query_digest);
                match (key_pairs.is_empty(), proof.as_ref()) {
                    (true, None) => verify_global_not(
                        &left_commit,
                        &right_commit,
                        &self.result_commitments,
                        self.result_len,
                        self.result_log_domain,
                    )?,
                    (false, Some(proof)) => verify_left_anti_join_with_keys(
                        &left_commit,
                        &right_commit,
                        key_pairs,
                        &self.result_commitments,
                        self.result_len,
                        self.result_log_domain,
                        proof,
                        &mut transcript,
                    )?,
                    _ => {
                        return Err(invalid(
                            "recursive left-anti key/proof variant mismatch",
                        ))
                    }
                }
            }
            (
                RecursiveRelationProof::LeftOuter { left, right, proof },
                RelationProofPlanKind::LeftOuter {
                    left: left_plan,
                    right: right_plan,
                    key_pairs,
                },
            ) => {
                left.verify_bounded(
                    layer,
                    expected_root,
                    left_plan,
                    depth + 1,
                    nodes,
                )?;
                right.verify_bounded(
                    layer,
                    expected_root,
                    right_plan,
                    depth + 1,
                    nodes,
                )?;
                let left_commit = left.relation_commit(None)?;
                let right_commit = right.relation_commit(None)?;
                let mut transcript =
                    crate::proof::argument::Transcript254::new(b"parture-left-outer-envelope");
                transcript.append_bytes(b"pow/outer-envelope/root", &expected_root);
                transcript.append_bytes(b"pow/outer-envelope/query", &plan.query_digest);
                if key_pairs.is_empty() {
                    verify_global_left_outer_join(
                        &left_commit,
                        &right_commit,
                        &self.result_commitments,
                        self.result_len,
                        self.result_log_domain,
                        proof,
                        &mut transcript,
                    )?;
                } else {
                    verify_left_outer_join_with_keys(
                        &left_commit,
                        &right_commit,
                        key_pairs,
                        &self.result_commitments,
                        self.result_len,
                        self.result_log_domain,
                        proof,
                        &mut transcript,
                    )?;
                }
            }
            (
                RecursiveRelationProof::InnerJoin { left, right, proof },
                RelationProofPlanKind::InnerJoin { left: left_plan, right: right_plan, key_pairs },
            ) => {
                left.verify_bounded(layer, expected_root, left_plan, depth + 1, nodes)?;
                right.verify_bounded(layer, expected_root, right_plan, depth + 1, nodes)?;
                let mut transcript = Transcript254::new(b"parture-inner-join-envelope");
                transcript.append_bytes(b"pow/inner-envelope/root", &expected_root);
                transcript.append_bytes(b"pow/inner-envelope/query", &plan.query_digest);
                verify_inner_join_with_keys(
                    &left.relation_commit(None)?, &right.relation_commit(None)?, key_pairs,
                    &self.result_commitments, self.result_len, self.result_log_domain,
                    proof, &mut transcript,
                )?;
            }
            (
                RecursiveRelationProof::Equality { input, proof },
                RelationProofPlanKind::Equality {
                    input: input_plan,
                    constraint,
                },
            ) => {
                input.verify_bounded(
                    layer,
                    expected_root,
                    input_plan,
                    depth + 1,
                    nodes,
                )?;
                let resolved = resolve_relation_equality_constraint(layer, constraint)?;
                let mut transcript =
                    crate::proof::argument::Transcript254::new(b"parture-equality-envelope");
                transcript.append_bytes(b"pow/equality-envelope/root", &expected_root);
                transcript.append_bytes(b"pow/equality-envelope/query", &plan.query_digest);
                verify_relation_equality(
                    &input.relation_commit(None)?,
                    &resolved,
                    &self.result_commitments,
                    self.result_len,
                    self.result_log_domain,
                    proof,
                    &mut transcript,
                )?;
            }
            (
                RecursiveRelationProof::TupleDisequality { input, proof },
                RelationProofPlanKind::ExactTupleDisequality {
                    input: input_plan,
                    pairs,
                },
            ) => {
                input.verify_bounded(
                    layer,
                    expected_root,
                    input_plan,
                    depth + 1,
                    nodes,
                )?;
                let resolved = resolve_tuple_disequality_pairs(layer, pairs)?;
                let mut transcript =
                    crate::proof::argument::Transcript254::new(b"parture-tuple-disequality-envelope");
                transcript.append_bytes(b"pow/tuple-disequality-envelope/root", &expected_root);
                transcript.append_bytes(b"pow/tuple-disequality-envelope/query", &plan.query_digest);
                verify_tuple_disequality(
                    &input.relation_commit(None)?,
                    &resolved,
                    &self.result_commitments,
                    self.result_len,
                    self.result_log_domain,
                    proof,
                    &mut transcript,
                )?;
            }
            (
                RecursiveRelationProof::ExactScalarOrder { input, proof },
                RelationProofPlanKind::ExactScalarOrder {
                    input: input_plan,
                    datatype,
                    direction,
                },
            ) => {
                input.verify_bounded(
                    layer,
                    expected_root,
                    input_plan,
                    depth + 1,
                    nodes,
                )?;
                let mut transcript = Transcript254::new(b"parture-order-envelope");
                transcript.append_bytes(b"pow/order-envelope/root", &expected_root);
                transcript.append_bytes(b"pow/order-envelope/query", &plan.query_digest);
                verify_exact_scalar_order(
                    layer,
                    &input.relation_commit(None)?,
                    *datatype,
                    *direction,
                    &self.result_commitments,
                    self.result_len,
                    self.result_log_domain,
                    proof,
                    &mut transcript,
                )?;
            }
            (
                RecursiveRelationProof::ExactMultiOrder { input, proof },
                RelationProofPlanKind::ExactMultiOrder { input: input_plan, datatypes, key_indices, directions },
            ) => {
                input.verify_bounded(layer, expected_root, input_plan, depth + 1, nodes)?;
                let mut transcript = Transcript254::new(b"parture-multi-order-envelope");
                transcript.append_bytes(b"pow/multi-order-envelope/root", &expected_root);
                transcript.append_bytes(b"pow/multi-order-envelope/query", &plan.query_digest);
                verify_exact_multi_order(layer, &input.relation_commit(None)?, &self.result_columns, datatypes, key_indices, directions, &self.result_commitments, self.result_len, self.result_log_domain, proof, &mut transcript)?;
            }
            (
                RecursiveRelationProof::ExactOrderedSlice { input },
                RelationProofPlanKind::ExactOrderedSlice {
                    input: input_plan,
                    offset,
                    limit,
                },
            ) => {
                input.verify_bounded(
                    layer,
                    expected_root,
                    input_plan,
                    depth + 1,
                    nodes,
                )?;
                let (start, end) = ordered_slice_bounds(input.result_len, *offset, *limit);
                let take = end - start;
                let expected_log = take.max(1).next_power_of_two().max(2).trailing_zeros() as usize;
                let expected_columns = input
                    .result_columns
                    .iter()
                    .map(|column| column[start..end].to_vec())
                    .collect::<Vec<_>>();
                if self.result_len != take
                    || self.result_log_domain != expected_log
                    || self.result_columns != expected_columns
                {
                    return Err(invalid("ordered slice result is not the exact contiguous child slice"));
                }
                let domain = 1usize << expected_log;
                let expected_commitments = expected_columns
                    .iter()
                    .map(|column| {
                        let mut padded = column.clone();
                        padded.resize(domain, ArkFr::zero());
                        commit_column(&padded)
                    })
                    .collect::<io::Result<Vec<_>>>()?;
                if self.result_commitments != expected_commitments {
                    return Err(invalid("ordered slice commitment mismatch"));
                }
            }
            (
                RecursiveRelationProof::GroupExtrema { input },
                RelationProofPlanKind::GroupExtrema { input: input_plan, key_columns, value_column, datatype, kind },
            ) => {
                input.verify_bounded(layer, expected_root, input_plan, depth + 1, nodes)?;
                let expected_columns = derived_group_extrema_columns(
                    layer, input, key_columns, *value_column, *datatype, *kind,
                )?;
                let expected_len = expected_columns.first().map_or(0, Vec::len);
                let expected_log = expected_len.max(1).next_power_of_two().max(2)
                    .trailing_zeros() as usize;
                if self.result_len != expected_len
                    || self.result_log_domain != expected_log
                    || self.result_columns.len() != expected_columns.len()
                    || self.result_columns.iter().zip(&expected_columns)
                        .any(|(actual, expected)| actual.get(..expected_len) != Some(expected.as_slice()))
                {
                    return Err(invalid("group-extrema result is not the exact canonical aggregate"));
                }
                let domain = 1usize << expected_log;
                let expected_commitments = expected_columns.iter().map(|column| {
                    let mut padded = column.clone();
                    padded.resize(domain, ArkFr::zero());
                    commit_column(&padded)
                }).collect::<io::Result<Vec<_>>>()?;
                if self.result_commitments != expected_commitments {
                    return Err(invalid("group-extrema commitment mismatch"));
                }
            }
            (
                RecursiveRelationProof::TransitiveClosure {
                    edge: edge_env,
                    seed,
                    rounds,
                    fixpoint,
                },
                RelationProofPlanKind::TransitiveClosure {
                    edge,
                    seed: seed_obj,
                    reflexive,
                    inverted,
                },
            ) => {
                let expected_rows = self.verify_transitive_closure(
                    layer,
                    expected_root,
                    plan,
                    edge,
                    seed_obj,
                    *reflexive,
                    *inverted,
                    edge_env,
                    seed,
                    rounds,
                    fixpoint,
                    depth,
                    nodes,
                )?;
                // The delivered relation must be exactly A_K (the reachable set), in the order the
                // proof's final round committed it. Length, columns, and domain are cross-checked here;
                // the shared tail then binds the commitments to these columns and resolves the ids to
                // node IRIs against the trusted dictionary.
                let expected_log = expected_rows
                    .len()
                    .max(1)
                    .next_power_of_two()
                    .max(2)
                    .trailing_zeros() as usize;
                let expected_column: Vec<ArkFr> =
                    expected_rows.iter().map(|id| ArkFr::from_u64(*id)).collect();
                if self.result_len != expected_rows.len()
                    || self.result_log_domain != expected_log
                    || self.result_columns != vec![expected_column]
                {
                    return Err(invalid(
                        "transitive closure delivered relation is not the proven reachable set",
                    ));
                }
            }
            (
                RecursiveRelationProof::TransitiveClosurePairs {
                    edge: edge_env,
                    seed,
                    rounds,
                    fixpoint,
                },
                RelationProofPlanKind::TransitiveClosurePairs { edge, inverted },
            ) => {
                let expected_pairs = self.verify_transitive_closure_pairs(
                    layer,
                    expected_root,
                    plan,
                    edge,
                    *inverted,
                    edge_env,
                    seed,
                    rounds,
                    fixpoint,
                    depth,
                    nodes,
                )?;
                // The delivered relation must be exactly A_K (the reachable pairs), in the order the
                // proof's final round committed it, over BOTH columns. Length, columns, and domain are
                // cross-checked here; the shared tail then binds the commitments to these columns and
                // resolves the ids to node IRIs against the trusted dictionary.
                let expected_log = expected_pairs
                    .len()
                    .max(1)
                    .next_power_of_two()
                    .max(2)
                    .trailing_zeros() as usize;
                let expected_src: Vec<ArkFr> =
                    expected_pairs.iter().map(|(s, _)| ArkFr::from_u64(*s)).collect();
                let expected_dst: Vec<ArkFr> =
                    expected_pairs.iter().map(|(_, d)| ArkFr::from_u64(*d)).collect();
                if self.result_len != expected_pairs.len()
                    || self.result_log_domain != expected_log
                    || self.result_columns != vec![expected_src, expected_dst]
                {
                    return Err(invalid(
                        "all-pairs closure delivered relation is not the proven reachable pairs set",
                    ));
                }
            }
            (
                RecursiveRelationProof::OrderedListWalk(walk),
                RelationProofPlanKind::OrderedListWalk { subject, predicate },
            ) => {
                self.verify_ordered_list_walk(
                    layer, expected_root, plan, subject, predicate, walk, depth, nodes,
                )?;
            }
            (
                RecursiveRelationProof::OrderedArrayWalk(walk),
                RelationProofPlanKind::OrderedArrayWalk { subject, predicate },
            ) => {
                self.verify_ordered_array_walk(
                    layer, expected_root, plan, subject, predicate, walk, depth, nodes,
                )?;
            }
            _ => return Err(invalid("recursive relation proof variant mismatch")),
        }

        verify_result_columns(
            &self.result_commitments,
            &self.result_columns,
            self.result_len,
            self.result_log_domain,
        )?;
        if self.ground_result {
            if self.result_commitments.len() != 1
                || self.result_columns.len() != 1
                || !self.result_descriptors.is_empty()
                || !self.result_values.is_empty()
            {
                return Err(invalid("invalid recursive ground-result shape"));
            }
        } else {
            let values = resolve_result_values(
                layer,
                layer.resolution_root(),
                &self.result_columns,
                &expected_descriptors,
            )?;
            if values != self.result_values {
                return Err(invalid("recursive canonical result mismatch"));
            }
        }
        Ok(())
    }

    /// Verify a bounded semi-naive transitive closure and return the reachable-set node ids the
    /// delivered relation must equal. Reconstructs the closure identity (edge, seed, star/plus) from
    /// the trusted plan, reads `K` = `rounds.len()` from the proof, and replays each audited step
    /// against the SAME `E` and the previous accumulator commitment. Concludes with the empty-delta
    /// fixpoint. No new gadget — this only orchestrates already-audited operator verifiers.
    #[allow(clippy::too_many_arguments)]
    fn verify_transitive_closure(
        &self,
        layer: &impl VerifierLayer,
        expected_root: ProofHash,
        plan: &RelationProofPlan,
        edge: &RelationProofPlan,
        seed_obj: &CanonicalObject,
        reflexive: bool,
        inverted: bool,
        edge_env: &RecursiveRelationEnvelope,
        seed: &TransitiveClosureSeed,
        rounds: &[TransitiveClosureRound],
        fixpoint: &TransitiveClosureFixpoint,
        depth: usize,
        nodes: &mut usize,
    ) -> io::Result<Vec<u64>> {
        let root = expected_root;
        let digest = plan.query_digest;

        // (1) Verify the ONE edge relation E (a general 2-column authenticated relation — single
        //     predicate or a compound derived edge), then extract its commitment for reuse everywhere.
        //     The edge plan carries its own K-INDEPENDENT digest, reconstructed from the trusted query.
        edge_env.verify_bounded(layer, expected_root, edge, depth + 1, nodes)?;
        let edge_commit = edge_env.relation_commit(None)?;

        // (2) Reconstruct the constant seed `{a}` from the trusted seed node (single-layer dictionary).
        let seed_id = resolve_closure_seed_id(&layer.verifier_view(), seed_obj)?;

        // (3)-(5) The SOURCE-AGNOSTIC seed + rounds + fixpoint replay (shared with the reconciled closure).
        verify_closure_rounds(
            &edge_commit, seed_id, root, digest, reflexive, inverted, seed, rounds, fixpoint, nodes,
        )
    }

    /// Verify a bounded ALL-PAIRS semi-naive transitive closure and return the reachable `(src, dst)`
    /// set the delivered relation must equal. Mirrors [`Self::verify_transitive_closure`] over 2 columns.
    #[allow(clippy::too_many_arguments)]
    fn verify_transitive_closure_pairs(
        &self,
        layer: &impl VerifierLayer,
        expected_root: ProofHash,
        plan: &RelationProofPlan,
        edge: &RelationProofPlan,
        inverted: bool,
        edge_env: &RecursiveRelationEnvelope,
        seed: &TransitiveClosurePairsSeed,
        rounds: &[TransitiveClosurePairsRound],
        fixpoint: &TransitiveClosurePairsFixpoint,
        depth: usize,
        nodes: &mut usize,
    ) -> io::Result<Vec<(u64, u64)>> {
        let root = expected_root;
        let digest = plan.query_digest;

        // (1) Verify the ONE edge relation E (a general 2-column authenticated relation — single predicate
        //     or a DERIVED all-pairs edge), then extract its commitment for reuse everywhere. The edge plan
        //     carries its own K-INDEPENDENT digest, reconstructed from the trusted query.
        edge_env.verify_bounded(layer, expected_root, edge, depth + 1, nodes)?;
        let edge_commit = edge_env.relation_commit(None)?;

        // (2)-(5) The seed + rounds + fixpoint replay over the 2-column accumulator.
        verify_closure_pairs_rounds(
            &edge_commit, root, digest, inverted, seed, rounds, fixpoint, nodes,
        )
    }

    pub fn verify(
        &self,
        layer: &impl VerifierLayer,
        expected_root: ProofHash,
        plan: &RelationProofPlan,
    ) -> io::Result<()> {
        // COMMITMENT DIET (Wave 0): delta layers are not directly query-provable.
        crate::proof::query::require_base_layer_for_query(layer.verifier_view().state)?;
        // Validate the entire caller-supplied tree once with a global node budget before
        // descending through the equally bounded delivered proof tree.
        plan.result_shape()?;
        let mut nodes = 0;
        self.verify_bounded(layer, expected_root, plan, 1, &mut nodes)
    }

    /// TWO-ROOT ENTRY — verify a top-level [`RelationProofPlanKind::CrossRootHashJoin`]. Validates
    /// BOTH per-graph commitments against the trusted `database_root`, verifies each child against
    /// its OWN root (single-root subtrees — the `verify_bounded` single-root invariant is preserved
    /// WITHIN each child), authenticates each side's key column to its IRI-hash against its own root,
    /// and verifies the inner join over the two hash columns. `database_root` MUST come from a trusted
    /// out-of-band pin, never recomputed from the prover's proof bundle or schema view.
    pub fn verify_cross_root(
        &self,
        instance: &impl VerifierLayer,
        schema: &impl VerifierLayer,
        database_root: ProofHash,
        plan: &RelationProofPlan,
    ) -> io::Result<()> {
        let (expected_descriptors, expected_ground) = plan.result_shape()?;
        let (left_plan, right_plan, left_key_col, right_key_col) = match &plan.kind {
            RelationProofPlanKind::CrossRootHashJoin { left, right, left_key_col, right_key_col } => {
                (left.as_ref(), right.as_ref(), *left_key_col, *right_key_col)
            }
            _ => return Err(invalid("verify_cross_root requires a CrossRootHashJoin plan")),
        };
        let (
            left,
            right,
            left_hash,
            right_hash,
            relocate_closure,
            com_closure_at_d,
            relocate_x,
            com_x_at_d,
            relocate_h,
            com_h_at_d,
            semijoin,
        ) = match &self.proof {
            RecursiveRelationProof::CrossRootHashJoin {
                left,
                right,
                left_hash,
                right_hash,
                relocate_closure,
                com_closure_at_d,
                relocate_x,
                com_x_at_d,
                relocate_h,
                com_h_at_d,
                semijoin,
            } => (
                left.as_ref(),
                right.as_ref(),
                left_hash.as_ref(),
                right_hash.as_ref(),
                relocate_closure,
                com_closure_at_d,
                relocate_x,
                com_x_at_d,
                relocate_h,
                com_h_at_d,
                semijoin.as_ref(),
            ),
            _ => return Err(invalid("cross-root proof/plan mismatch")),
        };

        // (1) Both per-graph commitments validated AND combined into the trusted database anchor.
        crate::proof::commitment::validate_database_root(instance, schema, &database_root)?;
        let instance_view = instance.verifier_view();
        let schema_view = schema.verifier_view();
        let instance_root = instance_view.state.commitment_root;
        let schema_root = schema_view.state.commitment_root;

        // Result identity: the output lives in the INSTANCE namespace.
        if self.layer_root != instance_root {
            return Err(invalid("cross-root result is not anchored to the instance root"));
        }
        if self.query_digest != plan.query_digest {
            return Err(invalid("cross-root query identity mismatch"));
        }
        if self.ground_result != expected_ground || self.result_descriptors != expected_descriptors {
            return Err(invalid("cross-root result schema mismatch"));
        }

        // (2) Verify each child against ITS OWN root (single-root subtrees).
        let mut lnodes = 0usize;
        left.verify_bounded(schema, schema_root, left_plan, 1, &mut lnodes)?;
        let mut rnodes = 0usize;
        right.verify_bounded(instance, instance_root, right_plan, 1, &mut rnodes)?;

        if left.result_commitments.len() != 1 || left_key_col != 0 {
            return Err(invalid("cross-root left child must be a single key column"));
        }
        if right.result_commitments.len() != 2 || right_key_col >= 2 {
            return Err(invalid("cross-root right child must be exactly [output, class-key]"));
        }
        let x_col = 1 - right_key_col;

        let mut transcript = Transcript254::new(b"parture-cross-root-envelope");
        transcript.append_bytes(b"pow/cross-root/db-root", &database_root);
        transcript.append_bytes(b"pow/cross-root/query", &plan.query_digest);

        // (3) Authenticate each side's key id column to its IRI-hash against its OWN root.
        verify_dict_hash_lookup(
            &left.result_commitments[0], left.result_log_domain, left.result_len, schema, left_hash, &mut transcript,
        )?;
        verify_dict_hash_lookup(
            &right.result_commitments[right_key_col], right.result_log_domain, right.result_len, instance, right_hash, &mut transcript,
        )?;

        // (4) Verify the three UP-relocations to the shared working domain 2^D, each bound to its
        // root-authenticated source commitment.
        let instance_n_dict = instance_view.state.node_value_count as usize;
        let schema_n_dict = schema_view.state.node_value_count as usize;
        let d_left = crate::proof::dicthash::working_domain_log(left.result_log_domain, schema_n_dict);
        let d_right = crate::proof::dicthash::working_domain_log(right.result_log_domain, instance_n_dict);
        let d_log = d_left.max(d_right);
        verify_column_relocation(
            &left_hash.lookup.com_v_hash, d_left, com_closure_at_d, d_log, relocate_closure, &mut transcript,
        )?;
        verify_column_relocation(
            &right_hash.lookup.com_v_hash, d_right, com_h_at_d, d_log, relocate_h, &mut transcript,
        )?;
        verify_column_relocation(
            &right.result_commitments[x_col], right.result_log_domain, com_x_at_d, d_log, relocate_x, &mut transcript,
        )?;

        // (5) Full-field LogUp semi-join: kept ?x rows whose class-hash is in the closure hash set.
        verify_hash_semijoin(
            com_closure_at_d, left.result_len, com_x_at_d, com_h_at_d, right.result_len,
            self.result_len, semijoin, &mut transcript,
        )?;
        // The delivered output is the semi-join's kept-?x column at domain 2^D.
        if self.result_log_domain != d_log
            || self.result_commitments != vec![semijoin.com_x_kept]
        {
            return Err(invalid("cross-root result does not match the semi-join output"));
        }

        // (6) Bind the CLEAR delivered result columns to the now-authenticated semi-join
        // commitments (mirrors the single-root `verify_bounded` postcondition at recursive.rs
        // ~3091). Without this a prover could keep a genuine envelope yet substitute arbitrary
        // authenticated dictionary ids into `self.result_columns`. The output is instance-anchored
        // (established by the `self.layer_root != instance_root` check above), so result values
        // resolve against the INSTANCE namespace.
        verify_result_columns(
            &self.result_commitments,
            &self.result_columns,
            self.result_len,
            self.result_log_domain,
        )?;
        if self.ground_result {
            if self.result_commitments.len() != 1
                || self.result_columns.len() != 1
                || !self.result_descriptors.is_empty()
                || !self.result_values.is_empty()
            {
                return Err(invalid("invalid cross-root ground-result shape"));
            }
        } else {
            let values = resolve_result_values(
                instance,
                instance.resolution_root(),
                &self.result_columns,
                &expected_descriptors,
            )?;
            if values != self.result_values {
                return Err(invalid("cross-root canonical result mismatch"));
            }
        }
        Ok(())
    }

    pub(super) fn put_bounded(
        &self,
        out: &mut Vec<u8>,
        depth: usize,
        nodes: &mut usize,
    ) -> io::Result<()> {
        if depth > MAX_RELATION_PROOF_DEPTH || *nodes >= MAX_RELATION_PROOF_NODES {
            return Err(invalid("recursive relation envelope exceeds structural limits"));
        }
        *nodes += 1;
        self.layer_root.put(out)?;
        self.query_digest.put(out)?;
        match &self.proof {
            RecursiveRelationProof::LayerTripleCount => {
                12u8.put(out)?;
            }
            RecursiveRelationProof::GeneratedU64Filter { input } => {
                13u8.put(out)?;
                input.put_bounded(out, depth + 1, nodes)?;
            }
            RecursiveRelationProof::GeneratedI64Filter { input } => {
                15u8.put(out)?;
                input.put_bounded(out, depth + 1, nodes)?;
            }
            RecursiveRelationProof::GroupSum { input } => {
                14u8.put(out)?;
                input.put_bounded(out, depth + 1, nodes)?;
            }
            RecursiveRelationProof::GroupCount { input } => {
                16u8.put(out)?;
                input.put_bounded(out, depth + 1, nodes)?;
            }
            RecursiveRelationProof::CollectionCount { input } => {
                17u8.put(out)?;
                input.put_bounded(out, depth + 1, nodes)?;
            }
            RecursiveRelationProof::CollectionSum { input } => {
                18u8.put(out)?;
                input.put_bounded(out, depth + 1, nodes)?;
            }
            RecursiveRelationProof::CollectionExtrema { input } => {
                19u8.put(out)?;
                input.put_bounded(out, depth + 1, nodes)?;
            }
            RecursiveRelationProof::Bgp(envelope) => {
                0u8.put(out)?;
                envelope.put(out)?;
            }
            RecursiveRelationProof::BagUnion { branches, proof } => {
                1u8.put(out)?;
                if branches.len() > MAX_RELATION_PROOF_NODES {
                    return Err(invalid("too many recursive union branches"));
                }
                branches.len().put(out)?;
                for branch in branches {
                    branch.put_bounded(out, depth + 1, nodes)?;
                }
                proof.put(out)?;
            }
            RecursiveRelationProof::BagProjection { input } => {
                10u8.put(out)?;
                input.put_bounded(out, depth + 1, nodes)?;
            }
            RecursiveRelationProof::Distinct { input, proof } => {
                11u8.put(out)?;
                input.put_bounded(out, depth + 1, nodes)?;
                proof.put(out)?;
            }
            RecursiveRelationProof::LeftAnti { left, right, proof } => {
                2u8.put(out)?;
                left.put_bounded(out, depth + 1, nodes)?;
                right.put_bounded(out, depth + 1, nodes)?;
                proof.put(out)?;
            }
            RecursiveRelationProof::LeftOuter { left, right, proof } => {
                3u8.put(out)?;
                left.put_bounded(out, depth + 1, nodes)?;
                right.put_bounded(out, depth + 1, nodes)?;
                proof.put(out)?;
            }
            RecursiveRelationProof::InnerJoin { left, right, proof } => {
                9u8.put(out)?;
                left.put_bounded(out, depth + 1, nodes)?;
                right.put_bounded(out, depth + 1, nodes)?;
                proof.put(out)?;
            }
            RecursiveRelationProof::CrossRootHashJoin {
                left,
                right,
                left_hash,
                right_hash,
                relocate_closure,
                com_closure_at_d,
                relocate_x,
                com_x_at_d,
                relocate_h,
                com_h_at_d,
                semijoin,
            } => {
                22u8.put(out)?;
                left.put_bounded(out, depth + 1, nodes)?;
                right.put_bounded(out, depth + 1, nodes)?;
                left_hash.put(out)?;
                right_hash.put(out)?;
                relocate_closure.put(out)?;
                com_closure_at_d.put(out)?;
                relocate_x.put(out)?;
                com_x_at_d.put(out)?;
                relocate_h.put(out)?;
                com_h_at_d.put(out)?;
                semijoin.put(out)?;
            }
            RecursiveRelationProof::Equality { input, proof } => {
                4u8.put(out)?;
                input.put_bounded(out, depth + 1, nodes)?;
                proof.put(out)?;
            }
            RecursiveRelationProof::TupleDisequality { input, proof } => {
                20u8.put(out)?;
                input.put_bounded(out, depth + 1, nodes)?;
                proof.put(out)?;
            }
            RecursiveRelationProof::ExactScalarOrder { input, proof } => {
                6u8.put(out)?;
                input.put_bounded(out, depth + 1, nodes)?;
                proof.put(out)?;
            }
            RecursiveRelationProof::ExactOrderedSlice { input } => {
                7u8.put(out)?;
                input.put_bounded(out, depth + 1, nodes)?;
            }
            RecursiveRelationProof::ExactMultiOrder { input, proof } => {
                8u8.put(out)?;
                input.put_bounded(out, depth + 1, nodes)?;
                proof.put(out)?;
            }
            RecursiveRelationProof::GroupExtrema { input } => {
                5u8.put(out)?;
                input.put_bounded(out, depth + 1, nodes)?;
            }
            RecursiveRelationProof::TransitiveClosure {
                edge,
                seed,
                rounds,
                fixpoint,
            } => {
                21u8.put(out)?;
                if rounds.len() > MAX_TRANSITIVE_CLOSURE_ROUNDS {
                    return Err(invalid("too many transitive-closure rounds"));
                }
                edge.put_bounded(out, depth + 1, nodes)?;
                seed.put(out)?;
                rounds.put(out)?;
                fixpoint.put(out)?;
            }
            RecursiveRelationProof::TransitiveClosurePairs {
                edge,
                seed,
                rounds,
                fixpoint,
            } => {
                25u8.put(out)?;
                if rounds.len() > MAX_TRANSITIVE_CLOSURE_ROUNDS {
                    return Err(invalid("too many all-pairs transitive-closure rounds"));
                }
                edge.put_bounded(out, depth + 1, nodes)?;
                seed.put(out)?;
                rounds.put(out)?;
                fixpoint.put(out)?;
            }
            RecursiveRelationProof::OrderedListWalk(walk) => {
                23u8.put(out)?;
                walk.put_bounded(out, depth, nodes)?;
            }
            RecursiveRelationProof::OrderedArrayWalk(walk) => {
                24u8.put(out)?;
                walk.put_bounded(out, depth, nodes)?;
            }
        }
        self.result_commitments.put(out)?;
        self.result_columns.put(out)?;
        self.result_log_domain.put(out)?;
        self.result_len.put(out)?;
        self.ground_result.put(out)?;
        self.result_descriptors.put(out)?;
        self.result_values.put(out)
    }

    pub(super) fn get_bounded(
        reader: &mut Reader<'_>,
        depth: usize,
        nodes: &mut usize,
    ) -> io::Result<Self> {
        if depth > MAX_RELATION_PROOF_DEPTH || *nodes >= MAX_RELATION_PROOF_NODES {
            return Err(invalid("recursive relation envelope exceeds structural limits"));
        }
        *nodes += 1;
        let layer_root = ProofHash::get(reader)?;
        let query_digest = ProofHash::get(reader)?;
        let proof = match u8::get(reader)? {
            12 => RecursiveRelationProof::LayerTripleCount,
            13 => RecursiveRelationProof::GeneratedU64Filter {
                input: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
            },
            15 => RecursiveRelationProof::GeneratedI64Filter {
                input: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
            },
            14 => RecursiveRelationProof::GroupSum {
                input: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
            },
            16 => RecursiveRelationProof::GroupCount {
                input: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
            },
            17 => RecursiveRelationProof::CollectionCount {
                input: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
            },
            18 => RecursiveRelationProof::CollectionSum {
                input: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
            },
            19 => RecursiveRelationProof::CollectionExtrema {
                input: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
            },
            0 => RecursiveRelationProof::Bgp(Box::<LayerBgpEnvelope>::get(reader)?),
            1 => {
                let len = usize::get(reader)?;
                if len > MAX_RELATION_PROOF_NODES - *nodes {
                    return Err(invalid("too many recursive union branches"));
                }
                let mut branches = Vec::new();
                for _ in 0..len {
                    branches.push(Self::get_bounded(reader, depth + 1, nodes)?);
                }
                RecursiveRelationProof::BagUnion {
                    branches,
                    proof: Box::<BagUnionProof>::get(reader)?,
                }
            }
            2 => RecursiveRelationProof::LeftAnti {
                left: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
                right: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
                proof: Option::<Box<LeftAntiJoinProof>>::get(reader)?,
            },
            3 => RecursiveRelationProof::LeftOuter {
                left: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
                right: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
                proof: Box::<LeftOuterJoinProof>::get(reader)?,
            },
            4 => RecursiveRelationProof::Equality {
                input: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
                proof: Box::<RelationEqualityProof>::get(reader)?,
            },
            20 => RecursiveRelationProof::TupleDisequality {
                input: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
                proof: Box::<TupleDisequalityProof>::get(reader)?,
            },
            5 => RecursiveRelationProof::GroupExtrema {
                input: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
            },
            21 => {
                let edge = Box::new(Self::get_bounded(reader, depth + 1, nodes)?);
                let seed = TransitiveClosureSeed::get(reader)?;
                let rounds = Vec::<TransitiveClosureRound>::get(reader)?;
                if rounds.len() > MAX_TRANSITIVE_CLOSURE_ROUNDS {
                    return Err(invalid("too many transitive-closure rounds"));
                }
                let fixpoint = TransitiveClosureFixpoint::get(reader)?;
                RecursiveRelationProof::TransitiveClosure {
                    edge,
                    seed,
                    rounds,
                    fixpoint,
                }
            }
            25 => {
                let edge = Box::new(Self::get_bounded(reader, depth + 1, nodes)?);
                let seed = TransitiveClosurePairsSeed::get(reader)?;
                let rounds = Vec::<TransitiveClosurePairsRound>::get(reader)?;
                if rounds.len() > MAX_TRANSITIVE_CLOSURE_ROUNDS {
                    return Err(invalid("too many all-pairs transitive-closure rounds"));
                }
                let fixpoint = TransitiveClosurePairsFixpoint::get(reader)?;
                RecursiveRelationProof::TransitiveClosurePairs {
                    edge,
                    seed,
                    rounds,
                    fixpoint,
                }
            }
            6 => RecursiveRelationProof::ExactScalarOrder {
                input: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
                proof: Box::<ExactScalarOrderProof>::get(reader)?,
            },
            7 => RecursiveRelationProof::ExactOrderedSlice {
                input: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
            },
            8 => RecursiveRelationProof::ExactMultiOrder {
                input: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
                proof: Box::<ExactMultiOrderProof>::get(reader)?,
            },
            9 => RecursiveRelationProof::InnerJoin {
                left: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
                right: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
                proof: Box::<InnerJoinProof>::get(reader)?,
            },
            22 => RecursiveRelationProof::CrossRootHashJoin {
                left: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
                right: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
                left_hash: Box::<DictHashLookupProof>::get(reader)?,
                right_hash: Box::<DictHashLookupProof>::get(reader)?,
                relocate_closure: RelocationProof::get(reader)?,
                com_closure_at_d: ArkGT::get(reader)?,
                relocate_x: RelocationProof::get(reader)?,
                com_x_at_d: ArkGT::get(reader)?,
                relocate_h: RelocationProof::get(reader)?,
                com_h_at_d: ArkGT::get(reader)?,
                semijoin: Box::<HashSemiJoinProof>::get(reader)?,
            },
            10 => RecursiveRelationProof::BagProjection {
                input: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
            },
            11 => RecursiveRelationProof::Distinct {
                input: Box::new(Self::get_bounded(reader, depth + 1, nodes)?),
                proof: Box::<RelationDistinctProof>::get(reader)?,
            },
            23 => RecursiveRelationProof::OrderedListWalk(Box::new(
                super::list_walk::OrderedListWalkProof::get_bounded(reader, depth, nodes)?,
            )),
            24 => RecursiveRelationProof::OrderedArrayWalk(Box::new(
                super::array_walk::OrderedArrayWalkProof::get_bounded(reader, depth, nodes)?,
            )),
            _ => return Err(invalid("invalid recursive relation proof tag")),
        };
        Ok(Self {
            layer_root,
            query_digest,
            proof,
            result_commitments: Vec::<ArkGT>::get(reader)?,
            result_columns: Vec::<Vec<ArkFr>>::get(reader)?,
            result_log_domain: usize::get(reader)?,
            result_len: usize::get(reader)?,
            ground_result: bool::get(reader)?,
            result_descriptors: Vec::<ResultColumnDescriptor>::get(reader)?,
            result_values: Vec::<Vec<CanonicalResultValue>>::get(reader)?,
        })
    }

    pub fn encode(&self) -> io::Result<Vec<u8>> {
        let mut out = Vec::new();
        out.extend_from_slice(RECURSIVE_MAGIC);
        out.extend_from_slice(&RECURSIVE_VERSION.to_le_bytes());
        let mut nodes = 0;
        self.put_bounded(&mut out, 1, &mut nodes)?;
        if out.len().checked_add(CHECKSUM_BYTES).map_or(true, |len| len > MAX_QUERY_ENVELOPE_BYTES) {
            return Err(invalid("recursive relation envelope too large"));
        }
        let checksum = Sha256::digest(&out);
        out.extend_from_slice(&checksum);
        Ok(out)
    }

    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < RECURSIVE_MAGIC.len() + 4 + CHECKSUM_BYTES
            || bytes.len() > MAX_QUERY_ENVELOPE_BYTES
        {
            return Err(invalid("invalid recursive relation envelope size"));
        }
        let body_len = bytes.len() - CHECKSUM_BYTES;
        let (body, checksum) = bytes.split_at(body_len);
        if Sha256::digest(body).as_slice() != checksum {
            return Err(invalid("recursive relation envelope checksum mismatch"));
        }
        let mut reader = Reader { bytes: body, pos: 0, decoded_items: 0 };
        if reader.take(RECURSIVE_MAGIC.len())? != RECURSIVE_MAGIC {
            return Err(invalid("unknown recursive relation envelope magic"));
        }
        if u32::from_le_bytes(reader.take(4)?.try_into().unwrap()) != RECURSIVE_VERSION {
            return Err(invalid("unsupported recursive relation envelope version"));
        }
        let mut nodes = 0;
        let envelope = Self::get_bounded(&mut reader, 1, &mut nodes)?;
        reader.finish()?;
        Ok(envelope)
    }
}
