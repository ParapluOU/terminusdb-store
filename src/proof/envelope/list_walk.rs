//! The ordered `rdf:List` walk composite ([`RelationProofPlanKind::OrderedListWalk`]) — ReadDocument
//! increment 1. Given a BOUND subject node `S` and a bound list-valued predicate `P`, it proves the
//! EXACT ORDERED element sequence of the stored list, delivered as one node/value column.
//!
//! # Why an ordered walk (in plain terms)
//!
//! TerminusDB stores an ordered list as a singly-linked "cons chain": `t(S,P,head)` points at the
//! first cell; each cell `Ci` carries `t(Ci,rdf:first,ei)` (its element) and `t(Ci,rdf:rest,C{i+1})`
//! (the link to the next cell); the chain ends with `t(Ck,rdf:rest,rdf:nil)`. The cons-cell ids are
//! FRESH RANDOM dictionary nodes — so the ORDER is the chain TOPOLOGY, not any stored numeric key. A
//! plain set-closure over `rdf:rest` proves WHICH cells are reachable, but a set has no order and no
//! positions: it would accept `[b,a]` for a stored `[a,b]`, or collapse a duplicate `[a,a]`. So the
//! order must be proven by an authenticated SUCCESSOR WALK.
//!
//! # The composite (compose audited pieces — no new cryptography)
//!
//! 1. **Anchor** — a bound-subject/bound-predicate BGP `t(S,P,?head)` proves the single head cons cell
//!    (exactly one head triple; 0 or >1 rejects closed).
//! 2. **Membership + termination** — a reflexive `TransitiveClosure` over the `rdf:rest` edge from
//!    `head` yields the complete reachable SET; its empty-delta fixpoint proves the chain terminates
//!    (a cycle can never reach the `rdf:nil` terminator and blows the round budget). The reachable set
//!    also contains `rdf:nil` (reached as the last cell's `rest`); a `LeftAnti` against the singleton
//!    `{rdf:nil}` STRIPS it, leaving exactly the cons-cell set.
//! 3. **Order binding** — a `permutation` proves the ordered cons column `C[0..K]` is exactly that
//!    cons-cell set (same cells, each once — distinctness + completeness). Then, for each cell, an
//!    `InnerJoin` against the authenticated `rdf:rest` edge relation reads its real successor: the join
//!    output row count pinned to `K+1` forces `rest` to be FUNCTIONAL (a cell with two `rest`s over- or
//!    under-shoots the count); a `relocation` pins the join's SUBJECT column positionally back to `C`
//!    (so each read successor belongs to the RIGHT cell); and the audited positional-SUCCESSOR primitive
//!    (`proof::sort`) proves the read successor column is the one-step shift of `C` (`rest(C[i])=C[i+1]`).
//!    Together these pin THE exact order; the anchored head and the `rdf:nil` terminator then fall out
//!    structurally (the chain's unique source is the head; the one successor that is not another cell is
//!    `rdf:nil`).
//! 4. **Element resolution** — the SAME functional-edge machinery on the `rdf:first` edge reads each
//!    cell's element positionally, delivering the ordered element column `E`. A final `InnerJoin` proves
//!    no element is itself a cons head (nested lists are rejected closed — deferred to a later increment).
//!
//! Every reject boundary FAILS CLOSED (never emits a bogus order): not exactly one head; a non-functional
//! `rest`/`first` (the join count / positional-subject relocation fails); a `rest` cycle (the closure
//! budget); an over-budget chain; a nested-list element (the final join is non-empty).
//!
//! # Anti-substitution
//!
//! Each sub-proof runs on a fresh transcript domain-separated by `(root, this node's query digest, role)`
//! so no sub-proof is fungible with another, and the four authenticated child relations carry their own
//! plan digests (derived deterministically from this node's digest), so the verifier reconstructs the
//! whole shape from its own trusted query.
#![allow(clippy::too_many_arguments)]

use std::collections::{BTreeMap, BTreeSet};
use std::io;

use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::transcript::Transcript;
use sha2::{Digest, Sha256};

use super::plan::{RelationProofPlan, RelationProofPlanKind};
use super::recursive::{RecursiveRelationEnvelope, RecursiveRelationProof};
use super::wire::{invalid, Reader, Wire};

use crate::proof::anti::{
    prove_inner_join_with_keys_and_context, prove_left_anti_join_with_keys_and_context,
    verify_inner_join_with_keys, verify_left_anti_join_with_keys, InnerJoinProof, LeftAntiJoinProof,
};
use crate::proof::argument::{StoreProverContext, Transcript254};
use crate::proof::bgp::{RelationCommit, RelationWitness};
use crate::proof::commitment::{LayerCommitment, VerifierLayer};
use crate::proof::equality::{
    prove_relation_equality_with_context, verify_relation_equality, RelationEqualityConstraint,
};
use crate::proof::permutation::{
    prove_relation_permutation_with_context, verify_relation_permutation, RelationPermutationProof,
};
use crate::proof::query::{
    generated_decimal_u64_value, AtomTerm, CanonicalResultValue, ColumnSource, LayerBgpAtom,
    LayerBgpPlan, ObjectKindFilter,
};
use crate::proof::relocate::{
    prove_column_relocation_with_context, verify_column_relocation, RelocationProof,
};
use crate::proof::sort::{
    prove_successor_lookup_with_context, verify_successor_lookup, SuccessorProof,
};
use crate::proof::{CanonicalObject, ProofHash};

// The three well-known RDF Collection IRIs, resolved against the trusted layer dictionary. These are
// data CONSTANTS (not plan fields): every stored `rdf:List` uses exactly these predicates/terminator,
// so both prover and verifier hardcode and resolve them identically.
const RDF_FIRST: &[u8] = b"http://www.w3.org/1999/02/22-rdf-syntax-ns#first";
const RDF_REST: &[u8] = b"http://www.w3.org/1999/02/22-rdf-syntax-ns#rest";
const RDF_NIL: &[u8] = b"http://www.w3.org/1999/02/22-rdf-syntax-ns#nil";

// ---------------------------------------------------------------------------
// Deterministic child identities + per-sub-proof transcripts (anti-substitution).
// ---------------------------------------------------------------------------

/// A child plan's query digest, derived deterministically from THIS node's digest + a role tag, so the
/// verifier reconstructs the child identity from its own trusted query with no per-`K` information.
fn child_digest(parent: &ProofHash, role: &[u8]) -> ProofHash {
    let mut hasher = Sha256::new();
    hasher.update(b"pow/list-walk/child-digest/v0");
    hasher.update(parent);
    hasher.update((role.len() as u64).to_le_bytes());
    hasher.update(role);
    let mut out = [0u8; 32];
    out.copy_from_slice(&hasher.finalize());
    out
}

/// A fresh transcript for one gluing gadget, domain-separated by `(root, this node's query digest,
/// role)` so no sub-proof can be replayed as another. Both prover and verifier build it identically.
fn lw_transcript(root: &ProofHash, digest: &ProofHash, role: &[u8]) -> Transcript254 {
    let mut transcript = Transcript254::new(b"parture-ordered-list-walk-envelope");
    transcript.append_bytes(b"pow/list-walk/root", root);
    transcript.append_bytes(b"pow/list-walk/query", digest);
    transcript.append_bytes(b"pow/list-walk/role", role);
    transcript
}

// ---------------------------------------------------------------------------
// Child plan builders (reconstructed identically by prover and verifier).
// ---------------------------------------------------------------------------

/// `t(S, P, ?head)` — a bound-subject, bound-predicate, variable-object single-atom BGP.
fn anchor_plan(digest: ProofHash, subject: &CanonicalObject, predicate: &[u8]) -> io::Result<RelationProofPlan> {
    let subject_bytes = match subject {
        CanonicalObject::Node(bytes) => bytes.clone(),
        _ => return Err(invalid("ordered-list-walk subject must be a node IRI")),
    };
    Ok(RelationProofPlan {
        query_digest: child_digest(&digest, b"anchor"),
        kind: RelationProofPlanKind::Bgp(LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![LayerBgpAtom {
                subject: AtomTerm::NodeConstant(subject_bytes),
                predicate_term: predicate.to_vec(),
                predicate_variable: false,
                object: AtomTerm::Variable,
                object_range: None,
                object_kind: ObjectKindFilter::Any,
                subject_object_equal: false,
                source: ColumnSource::Materialized,
            }],
            joins: Vec::new(),
        }),
    })
}

/// The two-column edge relation `t(?s, predicate, ?o)` (subject, object), reused for the closure edge
/// and for the rest/first functional-edge joins.
fn edge_bgp(predicate: &[u8]) -> LayerBgpPlan {
    LayerBgpPlan {
        source: ColumnSource::Materialized,
        atoms: vec![LayerBgpAtom::all_variables(predicate.to_vec())],
        joins: Vec::new(),
    }
}

/// A top-level 2-column edge relation as its own plan (used for the rest/first inner joins).
fn edge_relation_plan(digest: ProofHash, predicate: &[u8], role: &[u8]) -> RelationProofPlan {
    RelationProofPlan {
        query_digest: child_digest(&digest, role),
        kind: RelationProofPlanKind::Bgp(edge_bgp(predicate)),
    }
}

/// The reflexive `rdf:rest` transitive closure seeded at the anchored `head`.
fn closure_plan(digest: ProofHash, head: &CanonicalObject) -> RelationProofPlan {
    RelationProofPlan {
        query_digest: child_digest(&digest, b"closure"),
        kind: RelationProofPlanKind::TransitiveClosure {
            edge: edge_bgp(RDF_REST),
            seed: head.clone(),
            reflexive: true,
            inverted: false,
        },
    }
}

/// Resolve a node IRI to its dictionary id, or a deterministic IMPOSSIBLE id (`node_value_count + 1`)
/// when absent — so an absent constant fails CLOSED downstream rather than matching anything. Both
/// prover and verifier resolve identically.
fn resolve_node_id_or_absent(view: &crate::proof::VerifierCommitmentRef<'_>, iri: &[u8]) -> u64 {
    let target = CanonicalObject::Node(iri.to_vec());
    view.node_value_catalog
        .iter()
        .find(|record| record.triple.object == target)
        .map(|record| record.object_id)
        .unwrap_or_else(|| view.state.node_value_count.saturating_add(1))
}

/// Decode an authenticated node/value store id from its canonical field image.
fn field_id(field: ArkFr) -> io::Result<u64> {
    generated_decimal_u64_value(field)
}

// ---------------------------------------------------------------------------
// The proof payload.
// ---------------------------------------------------------------------------

/// Every sub-proof of one ordered `rdf:List` walk. The four child ENVELOPES are complete authenticated
/// relations (each bound to the trusted root); the remaining fields are the gluing gadgets and the
/// committed intermediate columns they tie together.
#[derive(Clone, Debug)]
pub struct OrderedListWalkProof {
    /// `t(S,P,?head)` — proves the single head cons cell.
    pub(crate) anchor: Box<RecursiveRelationEnvelope>,
    /// Reflexive `rdf:rest` `TransitiveClosure` from `head` — the reachable cons-cell set (incl. `nil`).
    pub(crate) closure: Box<RecursiveRelationEnvelope>,
    /// `t(?s,rdf:rest,?o)` — the authenticated rest-edge relation joined per cell.
    pub(crate) rest_edges: Box<RecursiveRelationEnvelope>,
    /// `t(?s,rdf:first,?o)` — the authenticated first-edge relation joined per cell.
    pub(crate) first_edges: Box<RecursiveRelationEnvelope>,

    /// `LeftAnti(closure, {rdf:nil})` stripping the terminator; its output is the cons-cell SET.
    pub(crate) strip_nil: LeftAntiJoinProof,
    /// The cons-cell set commitment(s) / length / domain (the anti-join's unmatched-left output).
    pub(crate) cons_commitments: Vec<ArkGT>,
    pub(crate) cons_len: usize,
    pub(crate) cons_log_domain: usize,

    /// `permutation(C ≡ cons-cell set)` — the ordered cons column is exactly the cell set.
    pub(crate) walk_perm: RelationPermutationProof,
    /// The committed ordered cons column `C` (the permutation output).
    pub(crate) com_walk: ArkGT,

    /// `InnerJoin(C, rest_edges)` reading each cell's successor; its two output commitments.
    pub(crate) rest_join: InnerJoinProof,
    pub(crate) rest_join_out: Vec<ArkGT>,
    /// `relocation`: the join's subject column equals `C` positionally (successor belongs to the cell).
    pub(crate) rest_reloc_subj: RelocationProof,
    /// `relocation`: the join's object column, lifted to `C`'s domain, is the committed successor column.
    pub(crate) rest_reloc_obj: RelocationProof,
    /// The committed successor column `C_next` (at `C`'s domain).
    pub(crate) com_next: ArkGT,
    /// The positional-successor proof: `C_next[i] = C[i+1]` (the exact order).
    pub(crate) successor: SuccessorProof,
    /// TERMINATOR: an equality-filter proving EXACTLY ONE successor equals `rdf:nil`. Combined with the
    /// successor step (which forces every `C_next[i]` for `i<K` to be another cons cell, never `nil`),
    /// this pins the single `nil` to the LAST position — so the chain genuinely TERMINATES at `nil`.
    /// A `rest` CYCLE (no cell rests to `nil`) makes the count `0` and rejects here.
    pub(crate) terminator: crate::proof::equality::RelationEqualityProof,
    /// The (single-row) terminator filter output commitment(s).
    pub(crate) terminator_out: Vec<ArkGT>,

    /// `InnerJoin(C, first_edges)` reading each cell's element; its two output commitments.
    pub(crate) first_join: InnerJoinProof,
    pub(crate) first_join_out: Vec<ArkGT>,
    pub(crate) first_reloc_subj: RelocationProof,
    pub(crate) first_reloc_obj: RelocationProof,
    /// The committed ordered element column `E` (the delivered result).
    pub(crate) com_elements: ArkGT,

    /// `InnerJoin(E, first_edges)` — must be EMPTY: no element is itself a cons head (nested-list reject).
    pub(crate) nested_join: InnerJoinProof,
    pub(crate) nested_join_out: Vec<ArkGT>,
}

impl OrderedListWalkProof {
    pub(super) fn put_bounded(
        &self,
        out: &mut Vec<u8>,
        depth: usize,
        nodes: &mut usize,
    ) -> io::Result<()> {
        self.anchor.put_bounded(out, depth + 1, nodes)?;
        self.closure.put_bounded(out, depth + 1, nodes)?;
        self.rest_edges.put_bounded(out, depth + 1, nodes)?;
        self.first_edges.put_bounded(out, depth + 1, nodes)?;
        self.strip_nil.put(out)?;
        self.cons_commitments.put(out)?;
        self.cons_len.put(out)?;
        self.cons_log_domain.put(out)?;
        self.walk_perm.put(out)?;
        self.com_walk.put(out)?;
        self.rest_join.put(out)?;
        self.rest_join_out.put(out)?;
        self.rest_reloc_subj.put(out)?;
        self.rest_reloc_obj.put(out)?;
        self.com_next.put(out)?;
        self.successor.put(out)?;
        self.terminator.put(out)?;
        self.terminator_out.put(out)?;
        self.first_join.put(out)?;
        self.first_join_out.put(out)?;
        self.first_reloc_subj.put(out)?;
        self.first_reloc_obj.put(out)?;
        self.com_elements.put(out)?;
        self.nested_join.put(out)?;
        self.nested_join_out.put(out)
    }

    pub(super) fn get_bounded(
        reader: &mut Reader<'_>,
        depth: usize,
        nodes: &mut usize,
    ) -> io::Result<Self> {
        Ok(Self {
            anchor: Box::new(RecursiveRelationEnvelope::get_bounded(reader, depth + 1, nodes)?),
            closure: Box::new(RecursiveRelationEnvelope::get_bounded(reader, depth + 1, nodes)?),
            rest_edges: Box::new(RecursiveRelationEnvelope::get_bounded(reader, depth + 1, nodes)?),
            first_edges: Box::new(RecursiveRelationEnvelope::get_bounded(reader, depth + 1, nodes)?),
            strip_nil: LeftAntiJoinProof::get(reader)?,
            cons_commitments: Vec::<ArkGT>::get(reader)?,
            cons_len: usize::get(reader)?,
            cons_log_domain: usize::get(reader)?,
            walk_perm: RelationPermutationProof::get(reader)?,
            com_walk: ArkGT::get(reader)?,
            rest_join: InnerJoinProof::get(reader)?,
            rest_join_out: Vec::<ArkGT>::get(reader)?,
            rest_reloc_subj: RelocationProof::get(reader)?,
            rest_reloc_obj: RelocationProof::get(reader)?,
            com_next: ArkGT::get(reader)?,
            successor: SuccessorProof::get(reader)?,
            terminator: crate::proof::equality::RelationEqualityProof::get(reader)?,
            terminator_out: Vec::<ArkGT>::get(reader)?,
            first_join: InnerJoinProof::get(reader)?,
            first_join_out: Vec::<ArkGT>::get(reader)?,
            first_reloc_subj: RelocationProof::get(reader)?,
            first_reloc_obj: RelocationProof::get(reader)?,
            com_elements: ArkGT::get(reader)?,
            nested_join: InnerJoinProof::get(reader)?,
            nested_join_out: Vec::<ArkGT>::get(reader)?,
        })
    }
}

// ---------------------------------------------------------------------------
// PROVE.
// ---------------------------------------------------------------------------

/// Prove the ordered element sequence of the `rdf:List` at `(subject, predicate)`. Returns the complete
/// envelope whose delivered relation is the ordered element column. See the module doc for the composite;
/// every failure path rejects CLOSED (returns an error), never emitting a bogus order.
pub(super) fn prove_ordered_list_walk(
    context: &StoreProverContext<'_>,
    layer: &LayerCommitment,
    plan: &RelationProofPlan,
    subject: &CanonicalObject,
    predicate: &[u8],
    depth: usize,
    nodes: &mut usize,
) -> io::Result<RecursiveRelationEnvelope> {
    prove_ordered_list_walk_impl(context, layer, plan, subject, predicate, depth, nodes, None)
}

/// Test-only forging entry: prove with a CALLER-CHOSEN chain order (a permutation-order, a truncation,
/// an injection, …) to confirm each forgery is rejected at the right gadget (a reordered but valid
/// permutation-order rejects at the SUCCESSOR; a wrong-cells order rejects at the PERMUTATION).
#[cfg(test)]
pub(super) fn prove_ordered_list_walk_forged(
    context: &StoreProverContext<'_>,
    layer: &LayerCommitment,
    plan: &RelationProofPlan,
    subject: &CanonicalObject,
    predicate: &[u8],
    walk_override: Vec<u64>,
) -> io::Result<RecursiveRelationEnvelope> {
    let mut nodes = 0usize;
    prove_ordered_list_walk_impl(
        context, layer, plan, subject, predicate, 1, &mut nodes, Some(walk_override),
    )
}

fn prove_ordered_list_walk_impl(
    context: &StoreProverContext<'_>,
    layer: &LayerCommitment,
    plan: &RelationProofPlan,
    subject: &CanonicalObject,
    predicate: &[u8],
    depth: usize,
    nodes: &mut usize,
    walk_override: Option<Vec<u64>>,
) -> io::Result<RecursiveRelationEnvelope> {
    let root = layer.state.commitment_root;
    let digest = plan.query_digest;

    // (1) ANCHOR: prove t(S,P,?head); require exactly one head triple.
    let anchor_p = anchor_plan(digest, subject, predicate)?;
    let anchor = RecursiveRelationEnvelope::prove_bounded(context, layer, &anchor_p, depth + 1, nodes)?;
    if anchor.result_len != 1 || anchor.result_columns.len() != 1 {
        return Err(invalid("ordered-list-walk anchor must be exactly one head triple"));
    }
    let head_iri = match anchor.result_values.first().and_then(|row| row.first()) {
        Some(CanonicalResultValue::Object(CanonicalObject::Node(bytes))) => bytes.clone(),
        _ => return Err(invalid("ordered-list-walk head must be a node")),
    };
    let head = CanonicalObject::Node(head_iri);
    let head_id = field_id(anchor.result_columns[0][0])?;

    // (2) CLOSURE (reflexive rest closure from head) + strip the rdf:nil terminator.
    let closure_p = closure_plan(digest, &head);
    let closure = RecursiveRelationEnvelope::prove_bounded(context, layer, &closure_p, depth + 1, nodes)?;

    let nil_id = resolve_node_id_or_absent(&layer.verifier_view(), RDF_NIL);
    let nil_witness =
        RelationWitness::commit_with_context(context, vec![vec![ArkFr::from_u64(nil_id)]], 1)?;
    let closure_witness = closure.relation_witness(None)?;
    let mut t_strip = lw_transcript(&root, &digest, b"strip-nil");
    let cons = prove_left_anti_join_with_keys_and_context(
        context, &closure_witness, &nil_witness, &[(0, 0)], &mut t_strip,
    )?;
    let cons_len = cons.n;
    let cons_log_domain = cons.log_domain;
    let cons_commitments = cons.commitments.clone();

    // The cons-cell set as ids (any order — the permutation is order-blind).
    let cons_ids: Vec<u64> = cons.columns[0][..cons_len]
        .iter()
        .map(|f| field_id(*f))
        .collect::<io::Result<Vec<_>>>()?;

    // (3) ORDER: reconstruct the true chain order by following rdf:rest from head to nil, then prove
    // C is a permutation of the cons-cell set.
    let rest_map = build_edge_map(&edge_columns(&closure_edge_env(&closure))?)?;
    // The rest edges live inside the closure's proven edge child; re-prove them as a top-level relation
    // below for the join. Here we only need the map to order the walk.
    let walk_ids = match walk_override {
        Some(ids) => ids,
        None => chain_order(head_id, nil_id, &rest_map, cons_len)?,
    };
    let cons_index: BTreeMap<u64, usize> = cons_ids.iter().enumerate().map(|(i, id)| (*id, i)).collect();
    let row_order: Vec<usize> = walk_ids
        .iter()
        .map(|id| cons_index.get(id).copied())
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| invalid("ordered-list-walk chain visited a non-cons node"))?;

    let cons_witness = RelationWitness {
        columns: cons.columns.clone(),
        commitments: cons.commitments.clone(),
        log_domain: cons_log_domain,
        n: cons_len,
    };
    let mut t_perm = lw_transcript(&root, &digest, b"walk-perm");
    let walk = prove_relation_permutation_with_context(context, &cons_witness, &row_order, &mut t_perm)?;
    let com_walk = walk.commitments[0];
    let walk_col = walk.columns[0].clone(); // padded to 2^cons_log_domain
    let c_witness = RelationWitness {
        columns: vec![walk_col.clone()],
        commitments: vec![com_walk],
        log_domain: cons_log_domain,
        n: cons_len,
    };

    // (4) rest edge relation (top-level, for the functional-edge join) + first edge relation.
    let rest_p = edge_relation_plan(digest, RDF_REST, b"rest-edges");
    let rest_edges = RecursiveRelationEnvelope::prove_bounded(context, layer, &rest_p, depth + 1, nodes)?;
    let first_p = edge_relation_plan(digest, RDF_FIRST, b"first-edges");
    let first_edges = RecursiveRelationEnvelope::prove_bounded(context, layer, &first_p, depth + 1, nodes)?;

    // (5) rest binding: each cell's real successor, functional, positionally tied to C, shift of C.
    let (com_next, _next_col, rest_bind) = prove_edge_bind(
        context, &root, &digest, &c_witness, &walk_col, cons_len, cons_log_domain,
        &rest_edges, b"rest",
    )?;
    let mut t_succ = lw_transcript(&root, &digest, b"successor");
    let base = walk_col.clone();
    let next = pad_to(&_next_col, cons_log_domain);
    let successor = prove_successor_lookup_with_context(
        context, &base, &next, cons_len, &com_walk, &com_next, &mut t_succ,
    )?;

    // TERMINATOR: prove EXACTLY ONE successor equals rdf:nil. With the successor step forcing every
    // non-last successor to be another cons cell, this pins the single nil to the last position, so the
    // chain genuinely terminates. A rest CYCLE has no cell resting to nil -> count 0 -> reject.
    let next_witness = RelationWitness {
        columns: vec![next.clone()],
        commitments: vec![com_next],
        log_domain: cons_log_domain,
        n: cons_len,
    };
    let nil_constraint = RelationEqualityConstraint::ColumnConstant {
        column: 0,
        value: ArkFr::from_u64(nil_id),
    };
    let mut t_term = lw_transcript(&root, &digest, b"terminator");
    let terminator =
        prove_relation_equality_with_context(context, &next_witness, &nil_constraint, &mut t_term)?;
    if terminator.n != 1 {
        return Err(invalid(
            "ordered-list-walk chain does not terminate at exactly one rdf:nil (cycle or missing terminator)",
        ));
    }

    // (6) first binding: each cell's element, delivered as E.
    let (com_elements, elem_col, first_bind) = prove_edge_bind(
        context, &root, &digest, &c_witness, &walk_col, cons_len, cons_log_domain,
        &first_edges, b"first",
    )?;

    // (7) nested-list reject: no delivered element is itself a cons head (a first-edge subject).
    let e_witness = RelationWitness {
        columns: vec![pad_to(&elem_col, cons_log_domain)],
        commitments: vec![com_elements],
        log_domain: cons_log_domain,
        n: cons_len,
    };
    let first_witness = first_edges.relation_witness(None)?;
    let mut t_nested = lw_transcript(&root, &digest, b"nested");
    let nested = prove_inner_join_with_keys_and_context(
        context, &e_witness, &first_witness, &[(0, 0)], &mut t_nested,
    )?;
    if nested.n != 0 {
        return Err(invalid(
            "ordered-list-walk element is itself a cons head (nested list not supported)",
        ));
    }

    let proof = OrderedListWalkProof {
        anchor: Box::new(anchor),
        closure: Box::new(closure),
        rest_edges: Box::new(rest_edges),
        first_edges: Box::new(first_edges),
        strip_nil: cons.proof,
        cons_commitments,
        cons_len,
        cons_log_domain,
        walk_perm: walk.proof,
        com_walk,
        rest_join: rest_bind.join,
        rest_join_out: rest_bind.join_out,
        rest_reloc_subj: rest_bind.reloc_subj,
        rest_reloc_obj: rest_bind.reloc_obj,
        com_next,
        successor,
        terminator: terminator.proof,
        terminator_out: terminator.commitments,
        first_join: first_bind.join,
        first_join_out: first_bind.join_out,
        first_reloc_subj: first_bind.reloc_subj,
        first_reloc_obj: first_bind.reloc_obj,
        com_elements,
        nested_join: nested.proof,
        nested_join_out: nested.commitments,
    };

    let result_column = elem_col[..cons_len].to_vec();
    RecursiveRelationEnvelope::finish_proved(
        layer,
        plan,
        RecursiveRelationProof::OrderedListWalk(Box::new(proof)),
        vec![com_elements],
        vec![result_column],
        cons_log_domain,
        cons_len,
    )
}

/// The intermediate sub-proofs of ONE functional-edge binding (rest or first).
struct EdgeBind {
    join: InnerJoinProof,
    join_out: Vec<ArkGT>,
    reloc_subj: RelocationProof,
    reloc_obj: RelocationProof,
}

/// Prove that `edge_env` (a 2-column `t(?s,pred,?o)` relation) assigns to each cell of `C` EXACTLY ONE
/// object, positionally: `InnerJoin(C, edge)` with output count pinned to `L` (functionality), a
/// relocation forcing the join's subject column to equal `C` (the object belongs to the right cell), and
/// a relocation lifting the join's object column onto `C`'s domain (the returned committed column). No
/// ordering is claimed here (that is the caller's successor step for `rest`); the object column is
/// carried positionally, so any element datatype is fine.
fn prove_edge_bind(
    context: &StoreProverContext<'_>,
    root: &ProofHash,
    digest: &ProofHash,
    c_witness: &RelationWitness,
    walk_col: &[ArkFr],
    l: usize,
    cons_log_domain: usize,
    edge_env: &RecursiveRelationEnvelope,
    role: &[u8],
) -> io::Result<(ArkGT, Vec<ArkFr>, EdgeBind)> {
    let edge_witness = edge_env.relation_witness(None)?;
    let mut t_join = lw_transcript(root, digest, &role_tag(role, b"join"));
    let join = prove_inner_join_with_keys_and_context(
        context, c_witness, &edge_witness, &[(0, 0)], &mut t_join,
    )?;
    // Functionality + reachability: exactly one edge per cell => output count == L.
    if join.n != l || join.columns.len() != 2 || join.commitments.len() != 2 {
        return Err(invalid("ordered-list-walk edge join is not functional (count != cell count)"));
    }
    let subj_out = join.columns[0].clone();
    let obj_out = join.columns[1].clone();
    let com_subj = join.commitments[0];
    let com_obj = join.commitments[1];

    // Positional subject binding: the join's subject column IS C (so the object belongs to the RIGHT
    // cell). Relocation forces positional agreement, catching a mis-multiplicity (a cell matched twice
    // and another zero times would make the subject column differ from C).
    let mut t_subj = lw_transcript(root, digest, &role_tag(role, b"subj-reloc"));
    let reloc_subj = prove_column_relocation_with_context(
        context, &subj_out, &com_subj, walk_col, &c_witness.commitments[0], &mut t_subj,
    )?;

    // Lift the object column onto C's domain and re-commit it as the returned column.
    let obj_at_cons = pad_to(&obj_out[..l], cons_log_domain);
    let com_obj_at_cons = context.commit_column(&obj_at_cons)?;
    let mut t_obj = lw_transcript(root, digest, &role_tag(role, b"obj-reloc"));
    let reloc_obj = prove_column_relocation_with_context(
        context, &obj_out, &com_obj, &obj_at_cons, &com_obj_at_cons, &mut t_obj,
    )?;

    Ok((
        com_obj_at_cons,
        obj_at_cons,
        EdgeBind {
            join: join.proof,
            join_out: join.commitments,
            reloc_subj,
            reloc_obj,
        },
    ))
}

// ---------------------------------------------------------------------------
// Prover-only helpers to reconstruct the chain order.
// ---------------------------------------------------------------------------

/// The closure envelope's proven edge child (`t(?s,rdf:rest,?o)`), used only to read the rest edges for
/// ordering the walk on the prover side.
fn closure_edge_env(closure: &RecursiveRelationEnvelope) -> &RecursiveRelationEnvelope {
    match &closure.proof {
        RecursiveRelationProof::TransitiveClosure { edge, .. } => edge,
        _ => closure, // unreachable for a well-formed closure child; the caller re-checks the shape.
    }
}

/// The `[subject, object]` id columns of a proven 2-column edge relation (prover side).
fn edge_columns(edge_env: &RecursiveRelationEnvelope) -> io::Result<(Vec<u64>, Vec<u64>)> {
    if edge_env.result_columns.len() != 2 {
        return Err(invalid("ordered-list-walk edge relation must have two columns"));
    }
    let n = edge_env.result_len;
    let subjects = edge_env.result_columns[0][..n]
        .iter()
        .map(|f| field_id(*f))
        .collect::<io::Result<Vec<_>>>()?;
    let objects = edge_env.result_columns[1][..n]
        .iter()
        .map(|f| field_id(*f))
        .collect::<io::Result<Vec<_>>>()?;
    Ok((subjects, objects))
}

/// Map each subject to its list of objects (prover side; used only to order the walk).
fn build_edge_map((subjects, objects): &(Vec<u64>, Vec<u64>)) -> io::Result<BTreeMap<u64, Vec<u64>>> {
    let mut map: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    for (s, o) in subjects.iter().zip(objects.iter()) {
        map.entry(*s).or_default().push(*o);
    }
    Ok(map)
}

/// Follow `rest` from `head` to `nil`, returning the ordered cons cells (excluding `nil`). Bounded by
/// the cons-cell count so a cycle / non-terminating chain cannot loop (it is rejected downstream by the
/// permutation / count checks). Returns whatever prefix it can build; soundness is enforced by the
/// gadgets, not by this prover convenience.
fn chain_order(
    head: u64,
    nil: u64,
    rest_map: &BTreeMap<u64, Vec<u64>>,
    budget: usize,
) -> io::Result<Vec<u64>> {
    let mut walk = Vec::new();
    let mut seen = BTreeSet::new();
    let mut cur = head;
    while cur != nil {
        if walk.len() > budget || !seen.insert(cur) {
            break;
        }
        walk.push(cur);
        match rest_map.get(&cur) {
            Some(objs) if objs.len() == 1 => cur = objs[0],
            _ => break, // dead-end or non-functional: the count/permutation checks reject downstream.
        }
    }
    Ok(walk)
}

fn role_tag(role: &[u8], suffix: &[u8]) -> Vec<u8> {
    let mut tag = role.to_vec();
    tag.push(b'/');
    tag.extend_from_slice(suffix);
    tag
}

fn pad_to(column: &[ArkFr], log_domain: usize) -> Vec<ArkFr> {
    let mut out = column.to_vec();
    out.resize(1usize << log_domain, ArkFr::zero());
    out
}

// ---------------------------------------------------------------------------
// VERIFY.
// ---------------------------------------------------------------------------

impl RecursiveRelationEnvelope {
    /// Verify an ordered `rdf:List` walk against the trusted root: re-verify each child relation, then
    /// replay every gluing gadget against the children's authenticated commitments, binding the
    /// delivered element column to the root. Mirrors `prove_ordered_list_walk` step for step.
    pub(super) fn verify_ordered_list_walk(
        &self,
        layer: &impl VerifierLayer,
        expected_root: ProofHash,
        plan: &RelationProofPlan,
        subject: &CanonicalObject,
        predicate: &[u8],
        walk: &OrderedListWalkProof,
        depth: usize,
        nodes: &mut usize,
    ) -> io::Result<()> {
        let root = expected_root;
        let digest = plan.query_digest;

        // (1) ANCHOR: re-verify and read the single head cons cell.
        let anchor_p = anchor_plan(digest, subject, predicate)?;
        walk.anchor.verify_bounded(layer, root, &anchor_p, depth + 1, nodes)?;
        if walk.anchor.result_len != 1 || walk.anchor.result_columns.len() != 1 {
            return Err(invalid("ordered-list-walk anchor must be exactly one head triple"));
        }
        let head_iri = match walk.anchor.result_values.first().and_then(|row| row.first()) {
            Some(CanonicalResultValue::Object(CanonicalObject::Node(bytes))) => bytes.clone(),
            _ => return Err(invalid("ordered-list-walk head must be a node")),
        };
        let head = CanonicalObject::Node(head_iri);

        // (2) CLOSURE + strip nil.
        let closure_p = closure_plan(digest, &head);
        walk.closure.verify_bounded(layer, root, &closure_p, depth + 1, nodes)?;
        let closure_commit = walk.closure.relation_commit(None)?;

        let nil_id = resolve_node_id_or_absent(&layer.verifier_view(), RDF_NIL);
        let nil_commit = singleton_commit(nil_id)?;
        let cons_commit = RelationCommit {
            commitments: walk.cons_commitments.clone(),
            log_domain: walk.cons_log_domain,
            n: walk.cons_len,
        };
        let mut t_strip = lw_transcript(&root, &digest, b"strip-nil");
        verify_left_anti_join_with_keys(
            &closure_commit,
            &nil_commit,
            &[(0, 0)],
            &walk.cons_commitments,
            walk.cons_len,
            walk.cons_log_domain,
            &walk.strip_nil,
            &mut t_strip,
        )?;

        // (3) ORDER: C is a permutation of the cons-cell set.
        let mut t_perm = lw_transcript(&root, &digest, b"walk-perm");
        verify_relation_permutation(
            &cons_commit,
            std::slice::from_ref(&walk.com_walk),
            walk.cons_len,
            walk.cons_log_domain,
            &walk.walk_perm,
            &mut t_perm,
        )?;
        let c_commit = RelationCommit {
            commitments: vec![walk.com_walk],
            log_domain: walk.cons_log_domain,
            n: walk.cons_len,
        };

        // (4) rest / first edge relations.
        let rest_p = edge_relation_plan(digest, RDF_REST, b"rest-edges");
        walk.rest_edges.verify_bounded(layer, root, &rest_p, depth + 1, nodes)?;
        let rest_commit = walk.rest_edges.relation_commit(None)?;
        let first_p = edge_relation_plan(digest, RDF_FIRST, b"first-edges");
        walk.first_edges.verify_bounded(layer, root, &first_p, depth + 1, nodes)?;
        let first_commit = walk.first_edges.relation_commit(None)?;

        // (5) rest binding + successor (order).
        verify_edge_bind(
            &root, &digest, &c_commit, &rest_commit, walk.cons_len, walk.cons_log_domain,
            &walk.rest_join, &walk.rest_join_out, &walk.rest_reloc_subj, &walk.rest_reloc_obj,
            walk.com_next, b"rest",
        )?;
        let mut t_succ = lw_transcript(&root, &digest, b"successor");
        verify_successor_lookup(
            &walk.com_walk, &walk.com_next, walk.cons_len, &walk.successor, &mut t_succ,
        )?;

        // TERMINATOR: exactly one successor equals rdf:nil (the chain terminates; a cycle rejects here).
        let next_commit = RelationCommit {
            commitments: vec![walk.com_next],
            log_domain: walk.cons_log_domain,
            n: walk.cons_len,
        };
        let nil_constraint = RelationEqualityConstraint::ColumnConstant {
            column: 0,
            value: ArkFr::from_u64(nil_id),
        };
        let mut t_term = lw_transcript(&root, &digest, b"terminator");
        verify_relation_equality(
            &next_commit,
            &nil_constraint,
            &walk.terminator_out,
            1,
            walk.cons_log_domain,
            &walk.terminator,
            &mut t_term,
        )?;

        // (6) first binding (elements).
        verify_edge_bind(
            &root, &digest, &c_commit, &first_commit, walk.cons_len, walk.cons_log_domain,
            &walk.first_join, &walk.first_join_out, &walk.first_reloc_subj, &walk.first_reloc_obj,
            walk.com_elements, b"first",
        )?;

        // (7) nested-list reject: InnerJoin(E, first_edges) is empty.
        let e_commit = RelationCommit {
            commitments: vec![walk.com_elements],
            log_domain: walk.cons_log_domain,
            n: walk.cons_len,
        };
        let nested_log = walk.nested_join.anti.join.num_vars();
        let mut t_nested = lw_transcript(&root, &digest, b"nested");
        verify_inner_join_with_keys(
            &e_commit,
            &first_commit,
            &[(0, 0)],
            &walk.nested_join_out,
            0,
            nested_log,
            &walk.nested_join,
            &mut t_nested,
        )?;

        // (8) Bind the delivered relation to the proven ordered element column.
        if self.result_len != walk.cons_len
            || self.result_log_domain != walk.cons_log_domain
            || self.result_commitments != vec![walk.com_elements]
        {
            return Err(invalid(
                "ordered-list-walk delivered relation is not the proven ordered element column",
            ));
        }
        Ok(())
    }
}

/// Verify one functional-edge binding: the inner join is a well-formed count-`L` match, its subject
/// column equals `C` positionally, and its object column (lifted to `C`'s domain) is `com_obj`.
fn verify_edge_bind(
    root: &ProofHash,
    digest: &ProofHash,
    c_commit: &RelationCommit,
    edge_commit: &RelationCommit,
    l: usize,
    cons_log_domain: usize,
    join: &InnerJoinProof,
    join_out: &[ArkGT],
    reloc_subj: &RelocationProof,
    reloc_obj: &RelocationProof,
    com_obj: ArkGT,
    role: &[u8],
) -> io::Result<()> {
    if join_out.len() != 2 {
        return Err(invalid("ordered-list-walk edge join output arity must be 2"));
    }
    let join_log = join.anti.join.num_vars();
    let mut t_join = lw_transcript(root, digest, &role_tag(role, b"join"));
    verify_inner_join_with_keys(
        c_commit, edge_commit, &[(0, 0)], join_out, l, join_log, join, &mut t_join,
    )?;
    // Subject column == C positionally (relocation forces positional agreement across domains).
    let mut t_subj = lw_transcript(root, digest, &role_tag(role, b"subj-reloc"));
    verify_column_relocation(
        &join_out[0], join_log, &c_commit.commitments[0], cons_log_domain, reloc_subj, &mut t_subj,
    )?;
    // Object column, lifted to C's domain, is com_obj.
    let mut t_obj = lw_transcript(root, digest, &role_tag(role, b"obj-reloc"));
    verify_column_relocation(
        &join_out[1], join_log, &com_obj, cons_log_domain, reloc_obj, &mut t_obj,
    )
}

fn singleton_commit(id: u64) -> io::Result<RelationCommit> {
    let column = vec![ArkFr::from_u64(id), ArkFr::zero()];
    Ok(RelationCommit {
        commitments: vec![crate::proof::argument::commit_column(&column)?],
        log_domain: 1,
        n: 1,
    })
}
