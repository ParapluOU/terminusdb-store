//! The ordered `sys:Array` walk composite ([`RelationProofPlanKind::OrderedArrayWalk`]) — ReadDocument
//! increment 4. Given a BOUND subject node `S` and a bound array-valued predicate `P`, it proves the
//! EXACT ORDERED `sys:value` sequence of a stored 1-D DENSE array, delivered as one node/value column.
//!
//! # How TerminusDB stores an array (in plain terms)
//!
//! Unlike an `rdf:List` (a linked cons chain), an array is a FLAT bag of numbered cells. For property
//! `P` of document `S`, each element is its own cell node with three facts:
//!   * `t(S, P, cell)`         — the parent points at EVERY cell (so the cell set is complete and needs
//!                               no closure/successor walk, unlike a list);
//!   * `t(cell, rdf:type, sys:Array)`
//!   * `t(cell, sys:index, i)` — a numeric index, base 0 (the order key);
//!   * `t(cell, sys:value, v)` — the element itself (a node or literal).
//! The ORDER is the authenticated numeric `sys:index`, and a DENSE 1-D array uses exactly the indices
//! `0,1,…,n-1` with no gaps and no duplicates.
//!
//! # What "faithful" means and why a set is not enough
//!
//! Just proving the SET of `(index, value)` pairs is insufficient: the delivered answer must be the
//! values IN ASCENDING INDEX ORDER, and the array must be DENSE (a stored `[0,2]` is NOT the same as a
//! two-element array — the reference `null`-fills the gap, so we must reject sparse here and only accept
//! a genuine dense array). So order AND density must be cryptographically certified.
//!
//! # The composite (compose audited pieces — the ONE new glue is the density certificate)
//!
//! 1. **(A) Cell set** — a bound-subject/bound-predicate BGP `t(S,P,?cell)` proves the COMPLETE set of
//!    `n` cells (every row kept). Because the parent points at every cell, this is the whole membership
//!    certificate: no cell can be dropped (the BGP is complete + root-authenticated) or injected.
//! 2. **(B/C) Edge relations** — the authenticated 2-column relations `t(?s,sys:index,?o)`,
//!    `t(?s,sys:value,?o)`, and `t(?s,rdf:type,?o)`.
//! 3. **(D/E/F) Functional binds** — [`super::list_walk::prove_edge_bind`] joins each cell to its ONE
//!    `sys:index`, its ONE `sys:value`, and its ONE `rdf:type`, pinning the join count to `n` (so a cell
//!    with 0 or ≥2 of any of them breaks the count and rejects) and positionally relocating each read
//!    object back onto the cell column. This delivers, aligned cell-for-cell: the index id, the value
//!    id, and the type id. A following `Equality` filter forces EVERY cell's type to equal `sys:Array`
//!    (a `P`-object that is not a well-formed array cell — a literal, a mistyped node — fails closed).
//! 4. **(G) DENSITY + ORDER certificate (the one new glue)** — authenticate each cell's index id to its
//!    dictionary VALUE and DATATYPE ([`crate::proof::range::prove_dict_value_lookup`]), force the
//!    datatype to the order-preserving unsigned integer type (a completeness filter, so every index is a
//!    genuine numeric position), decode each authenticated value to a `u64` position `p_i`, and reject
//!    any `p_i ≥ n`. Then a SINGLE `permutation` routes the pair `(position, value)` of each cell onto
//!    the row `(j, out_j)`, and the verifier RECONSTRUCTS the output index column as the public identity
//!    vector `[0,1,…,n-1]`. Because the identity entries are DISTINCT and there are `n` of them, the
//!    permutation's multiset equality forces `{p_i}` to be a bijection onto `{0,…,n-1}` — DENSE,
//!    distinct, 0-based, no gap, no duplicate (pigeonhole) — AND, because the SAME permutation carries
//!    the value column, the delivered `Out[j]` is exactly the value of the cell whose index is `j`
//!    (value-in-index-order). The value column is carried POSITIONALLY (never compared as a magnitude),
//!    so any element datatype is fine.
//!
//! Every reject boundary FAILS CLOSED (never emits a bogus order): a cell with 0/≥2 `sys:index`,
//! `sys:value`, or `rdf:type` (functional-bind count); a non-`sys:Array` cell (the type filter); an
//! index of the wrong datatype (the datatype-completeness filter); an index `≥ n` (the decode bound);
//! and a gap or duplicate (the density permutation-to-identity multiset check).
//!
//! # A note on the index datatype (soundness-relevant)
//!
//! The design doc writes the index as `xsd:nonNegativeInteger`. This STORE does NOT order-preserving-
//! encode `nonNegativeInteger` (it maps to the `0` sentinel in [`crate::proof::commitment`]'s
//! `node_value_op_value`), so a `nonNegativeInteger` value cannot be recovered from committed data at
//! all. The composite therefore authenticates the index under the order-preserving fixed-width unsigned
//! datatype [`INDEX_DATATYPE`] (`UInt64`), whose committed order-preserving value IS the raw integer.
//! The compiler mirror / frame integration (a separate follow-on) maps the schema's declared index type
//! onto this store-level convention.
//!
//! # Multi-dimensional arrays
//!
//! This store composite reads ONLY the level-1 `sys:index` column; multi-dimensional arrays (`sys:index2`
//! …) are primarily a COMPILER/frame reject (design §9). If a multi-dimensional array nevertheless
//! reaches this composite, its single `sys:index` column is not a dense `0..n-1` permutation, so the
//! density certificate rejects it closed.
//!
//! # Anti-substitution
//!
//! Each sub-proof runs on a fresh transcript domain-separated by `(root, this node's query digest,
//! role)`, and the four child relations carry their own plan digests derived deterministically from this
//! node's digest, so the verifier reconstructs the whole shape from its own trusted query and no
//! sub-proof is fungible with another.
#![allow(clippy::too_many_arguments)]

use std::convert::TryFrom;
use std::io;

use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::transcript::Transcript;
use sha2::{Digest, Sha256};

use super::list_walk::{
    field_id, pad_to, prove_edge_bind, prove_edge_bind_witness, resolve_node_id_or_absent,
    verify_edge_bind,
};
use super::plan::{RelationProofPlan, RelationProofPlanKind};
use super::reconciled::{
    bound_subject_leaf_atom, predicate_edge_leaf_atom, prove_reconciled_leaf,
    resolve_combined_node_id, resolve_combined_node_id_termless, verify_reconciled_leaf_from_views,
    RECONCILE_TRANSCRIPT,
};
use super::recursive::{RecursiveRelationEnvelope, RecursiveRelationProof};
use super::wire::{invalid, Reader, Wire};

use crate::proof::reconcile::{
    build_termless_present_range_openings, prove_combined_range_dict, prove_reconcile_core,
    range_dict_branch_commits, range_dict_branches, resolve_node_value_id_column,
    verify_combined_range_dict, verify_dict_value_terms_chain, verify_reconcile_core,
    CombinedRangeDictProof, PresentBgpProof, ReconcileProof,
};

use crate::proof::argument::{commit_column, StoreProverContext, Transcript254};
use crate::proof::bgp::{RelationCommit, RelationWitness};
use crate::proof::commitment::{
    build_resolution_openings, DictOpening, LayerCommitment, ResolutionOpenings, VerifierLayer,
};
use crate::proof::equality::{
    prove_relation_equality_with_context, verify_relation_equality, RelationEqualityConstraint,
    RelationEqualityProof,
};
use crate::proof::filter::{
    prove_filter_indicator_with_context, verify_filter_indicator, FilterProof,
};
use crate::proof::permutation::{
    prove_relation_permutation_with_context, verify_relation_permutation, RelationPermutationProof,
};
use crate::proof::query::{
    generated_decimal_u64_value, AtomTerm, ColumnSource, LayerBgpAtom, LayerBgpPlan,
    ObjectKindFilter,
};
use crate::proof::range::{
    honest_dict_lookup, prove_dict_value_lookup_with_context, verify_dict_value_lookup,
    DictLookupProof,
};
use crate::proof::relocate::{
    prove_column_relocation_with_context, verify_column_relocation, RelocationProof,
};
use crate::proof::anti::InnerJoinProof;
use crate::proof::{CanonicalDatatype, CanonicalObject, ProofHash};

// The well-known predicate / class IRIs, resolved against the trusted layer dictionary. These are data
// CONSTANTS (not plan fields): every stored `sys:Array` uses exactly these, so prover and verifier
// hardcode and resolve them identically.
const RDF_TYPE: &[u8] = b"http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
const SYS_INDEX: &[u8] = b"http://terminusdb.com/schema/sys#index";
const SYS_VALUE: &[u8] = b"http://terminusdb.com/schema/sys#value";
const SYS_ARRAY: &[u8] = b"http://terminusdb.com/schema/sys#Array";

/// The datatype the composite requires for a `sys:index` literal. The store's order-preserving value
/// encoding of a fixed-width UNSIGNED integer IS the raw integer (a big-endian read of its bytes), so a
/// `UInt64` index authenticates directly to its numeric position. `nonNegativeInteger` (the doc's
/// nominal type) is NOT order-preserving-encoded by this store, so it could not be recovered from
/// committed data — see the module docs.
const INDEX_DATATYPE: CanonicalDatatype = CanonicalDatatype::UInt64;

// ---------------------------------------------------------------------------
// Deterministic child identities + per-sub-proof transcripts (anti-substitution).
// ---------------------------------------------------------------------------

/// A child plan's query digest, derived deterministically from THIS node's digest + a role tag. The
/// distinct domain label keeps array-walk child digests disjoint from list-walk's, even for the same
/// parent digest.
fn aw_child_digest(parent: &ProofHash, role: &[u8]) -> ProofHash {
    let mut hasher = Sha256::new();
    hasher.update(b"pow/array-walk/child-digest/v0");
    hasher.update(parent);
    hasher.update((role.len() as u64).to_le_bytes());
    hasher.update(role);
    let mut out = [0u8; 32];
    out.copy_from_slice(&hasher.finalize());
    out
}

/// A fresh transcript for one gluing gadget, domain-separated by `(root, this node's query digest,
/// role)`. Both prover and verifier build it identically.
fn aw_transcript(root: &ProofHash, digest: &ProofHash, role: &[u8]) -> Transcript254 {
    let mut transcript = Transcript254::new(b"parture-ordered-array-walk-envelope");
    transcript.append_bytes(b"pow/array-walk/root", root);
    transcript.append_bytes(b"pow/array-walk/query", digest);
    transcript.append_bytes(b"pow/array-walk/role", role);
    transcript
}

// ---------------------------------------------------------------------------
// Child plan builders (reconstructed identically by prover and verifier).
// ---------------------------------------------------------------------------

/// `t(S, P, ?cell)` — a bound-subject, bound-predicate, variable-object single-atom BGP. Unlike the
/// list walk's anchor (exactly one head), EVERY row is kept: this is the complete cell set.
fn cells_plan(
    digest: ProofHash,
    subject: &CanonicalObject,
    predicate: &[u8],
) -> io::Result<RelationProofPlan> {
    let subject_bytes = match subject {
        CanonicalObject::Node(bytes) => bytes.clone(),
        _ => return Err(invalid("ordered-array-walk subject must be a node IRI")),
    };
    Ok(RelationProofPlan {
        query_digest: aw_child_digest(&digest, b"cells"),
        kind: RelationProofPlanKind::Bgp(LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![LayerBgpAtom {
                subject: AtomTerm::NodeConstant(subject_bytes),
                predicate_term: predicate.to_vec(),
                predicate_variable: false,
                // `Any`, not `Node`: a malformed literal `P`-object must be KEPT (so it is counted) and
                // then rejected downstream by the functional binds — never silently dropped.
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

/// A top-level 2-column edge relation `t(?s, predicate, ?o)` as its own plan (used for the
/// index/value/type functional binds).
fn edge_relation_plan(digest: ProofHash, predicate: &[u8], role: &[u8]) -> RelationProofPlan {
    RelationProofPlan {
        query_digest: aw_child_digest(&digest, role),
        kind: RelationProofPlanKind::Bgp(LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![LayerBgpAtom::all_variables(predicate.to_vec())],
            joins: Vec::new(),
        }),
    }
}

// ---------------------------------------------------------------------------
// The proof payload.
// ---------------------------------------------------------------------------

/// Every sub-proof of one ordered `sys:Array` walk. The four child ENVELOPES are complete authenticated
/// relations (each bound to the trusted root); the remaining fields are the gluing gadgets and the
/// committed intermediate columns they tie together.
#[derive(Clone, Debug)]
pub struct OrderedArrayWalkProof {
    /// `t(S,P,?cell)` — the COMPLETE cell set (the whole membership certificate).
    pub(crate) cells: Box<RecursiveRelationEnvelope>,
    /// `t(?s,sys:index,?o)` — the authenticated index-edge relation.
    pub(crate) index_edges: Box<RecursiveRelationEnvelope>,
    /// `t(?s,sys:value,?o)` — the authenticated value-edge relation.
    pub(crate) value_edges: Box<RecursiveRelationEnvelope>,
    /// `t(?s,rdf:type,?o)` — the authenticated type-edge relation.
    pub(crate) type_edges: Box<RecursiveRelationEnvelope>,

    pub(crate) cons_len: usize,
    pub(crate) cell_log_domain: usize,

    /// (D) functional index bind: one `sys:index` per cell, positionally aligned.
    pub(crate) index_join: InnerJoinProof,
    pub(crate) index_join_out: Vec<ArkGT>,
    pub(crate) index_reloc_subj: RelocationProof,
    pub(crate) index_reloc_obj: RelocationProof,
    pub(crate) com_index: ArkGT,

    /// (E) functional value bind: one `sys:value` per cell, positionally aligned.
    pub(crate) value_join: InnerJoinProof,
    pub(crate) value_join_out: Vec<ArkGT>,
    pub(crate) value_reloc_subj: RelocationProof,
    pub(crate) value_reloc_obj: RelocationProof,
    pub(crate) com_value: ArkGT,

    /// (F) functional type bind: one `rdf:type` per cell + an equality filter forcing `== sys:Array`.
    pub(crate) type_join: InnerJoinProof,
    pub(crate) type_join_out: Vec<ArkGT>,
    pub(crate) type_reloc_subj: RelocationProof,
    pub(crate) type_reloc_obj: RelocationProof,
    pub(crate) com_type: ArkGT,
    pub(crate) type_equality: RelationEqualityProof,
    pub(crate) type_equality_out: Vec<ArkGT>,

    // (G) density + order certificate ----------------------------------------------------------
    /// The shared working domain `2^common_log_domain` for the dictionary lookup + permutation.
    pub(crate) common_log_domain: usize,
    /// The cell-aligned index-id column lifted to the working domain.
    pub(crate) reloc_index_at_d: RelocationProof,
    pub(crate) com_index_at_d: ArkGT,
    /// The cell-aligned value-id column lifted to the working domain (the permutation's value input).
    pub(crate) reloc_value_at_d: RelocationProof,
    pub(crate) com_value_at_d: ArkGT,
    /// The three authenticated dictionary columns lifted to the working domain.
    pub(crate) reloc_dict_id: RelocationProof,
    pub(crate) com_dict_id_at_d: ArkGT,
    pub(crate) reloc_dict_value: RelocationProof,
    pub(crate) com_dict_value_at_d: ArkGT,
    pub(crate) reloc_dict_datatype: RelocationProof,
    pub(crate) com_dict_datatype_at_d: ArkGT,
    /// Authenticates each cell's index id to its true dictionary VALUE (`com_v_val`) and DATATYPE
    /// (`com_v_dt`). `com_v_val` is the permutation's position input.
    pub(crate) lookup: DictLookupProof,
    /// Forces every index's datatype to `INDEX_DATATYPE` (completeness: `sel == χ_n`).
    pub(crate) datatype_filter: FilterProof,
    pub(crate) datatype_selection_commitment: ArkGT,
    pub(crate) datatype_inverse_commitment: ArkGT,
    /// The density+order permutation: `[position, value] -> [identity(0..n-1), Out]`.
    pub(crate) order_perm: RelationPermutationProof,
    /// The delivered ordered value column `Out` (the permutation's value output).
    pub(crate) com_out: ArkGT,
}

impl OrderedArrayWalkProof {
    pub(super) fn put_bounded(
        &self,
        out: &mut Vec<u8>,
        depth: usize,
        nodes: &mut usize,
    ) -> io::Result<()> {
        self.cells.put_bounded(out, depth + 1, nodes)?;
        self.index_edges.put_bounded(out, depth + 1, nodes)?;
        self.value_edges.put_bounded(out, depth + 1, nodes)?;
        self.type_edges.put_bounded(out, depth + 1, nodes)?;
        self.cons_len.put(out)?;
        self.cell_log_domain.put(out)?;
        self.index_join.put(out)?;
        self.index_join_out.put(out)?;
        self.index_reloc_subj.put(out)?;
        self.index_reloc_obj.put(out)?;
        self.com_index.put(out)?;
        self.value_join.put(out)?;
        self.value_join_out.put(out)?;
        self.value_reloc_subj.put(out)?;
        self.value_reloc_obj.put(out)?;
        self.com_value.put(out)?;
        self.type_join.put(out)?;
        self.type_join_out.put(out)?;
        self.type_reloc_subj.put(out)?;
        self.type_reloc_obj.put(out)?;
        self.com_type.put(out)?;
        self.type_equality.put(out)?;
        self.type_equality_out.put(out)?;
        self.common_log_domain.put(out)?;
        self.reloc_index_at_d.put(out)?;
        self.com_index_at_d.put(out)?;
        self.reloc_value_at_d.put(out)?;
        self.com_value_at_d.put(out)?;
        self.reloc_dict_id.put(out)?;
        self.com_dict_id_at_d.put(out)?;
        self.reloc_dict_value.put(out)?;
        self.com_dict_value_at_d.put(out)?;
        self.reloc_dict_datatype.put(out)?;
        self.com_dict_datatype_at_d.put(out)?;
        self.lookup.put(out)?;
        self.datatype_filter.put(out)?;
        self.datatype_selection_commitment.put(out)?;
        self.datatype_inverse_commitment.put(out)?;
        self.order_perm.put(out)?;
        self.com_out.put(out)
    }

    pub(super) fn get_bounded(
        reader: &mut Reader<'_>,
        depth: usize,
        nodes: &mut usize,
    ) -> io::Result<Self> {
        Ok(Self {
            cells: Box::new(RecursiveRelationEnvelope::get_bounded(reader, depth + 1, nodes)?),
            index_edges: Box::new(RecursiveRelationEnvelope::get_bounded(reader, depth + 1, nodes)?),
            value_edges: Box::new(RecursiveRelationEnvelope::get_bounded(reader, depth + 1, nodes)?),
            type_edges: Box::new(RecursiveRelationEnvelope::get_bounded(reader, depth + 1, nodes)?),
            cons_len: usize::get(reader)?,
            cell_log_domain: usize::get(reader)?,
            index_join: InnerJoinProof::get(reader)?,
            index_join_out: Vec::<ArkGT>::get(reader)?,
            index_reloc_subj: RelocationProof::get(reader)?,
            index_reloc_obj: RelocationProof::get(reader)?,
            com_index: ArkGT::get(reader)?,
            value_join: InnerJoinProof::get(reader)?,
            value_join_out: Vec::<ArkGT>::get(reader)?,
            value_reloc_subj: RelocationProof::get(reader)?,
            value_reloc_obj: RelocationProof::get(reader)?,
            com_value: ArkGT::get(reader)?,
            type_join: InnerJoinProof::get(reader)?,
            type_join_out: Vec::<ArkGT>::get(reader)?,
            type_reloc_subj: RelocationProof::get(reader)?,
            type_reloc_obj: RelocationProof::get(reader)?,
            com_type: ArkGT::get(reader)?,
            type_equality: RelationEqualityProof::get(reader)?,
            type_equality_out: Vec::<ArkGT>::get(reader)?,
            common_log_domain: usize::get(reader)?,
            reloc_index_at_d: RelocationProof::get(reader)?,
            com_index_at_d: ArkGT::get(reader)?,
            reloc_value_at_d: RelocationProof::get(reader)?,
            com_value_at_d: ArkGT::get(reader)?,
            reloc_dict_id: RelocationProof::get(reader)?,
            com_dict_id_at_d: ArkGT::get(reader)?,
            reloc_dict_value: RelocationProof::get(reader)?,
            com_dict_value_at_d: ArkGT::get(reader)?,
            reloc_dict_datatype: RelocationProof::get(reader)?,
            com_dict_datatype_at_d: ArkGT::get(reader)?,
            lookup: DictLookupProof::get(reader)?,
            datatype_filter: FilterProof::get(reader)?,
            datatype_selection_commitment: ArkGT::get(reader)?,
            datatype_inverse_commitment: ArkGT::get(reader)?,
            order_perm: RelationPermutationProof::get(reader)?,
            com_out: ArkGT::get(reader)?,
        })
    }
}

// ---------------------------------------------------------------------------
// PROVE.
// ---------------------------------------------------------------------------

/// Prove the ordered `sys:value` sequence of the `sys:Array` at `(subject, predicate)`. Returns the
/// complete envelope whose delivered relation is the value-in-index-order column. Every failure path
/// rejects CLOSED (returns an error), never emitting a bogus order.
pub(super) fn prove_ordered_array_walk(
    context: &StoreProverContext<'_>,
    layer: &LayerCommitment,
    plan: &RelationProofPlan,
    subject: &CanonicalObject,
    predicate: &[u8],
    depth: usize,
    nodes: &mut usize,
) -> io::Result<RecursiveRelationEnvelope> {
    prove_ordered_array_walk_impl(context, layer, plan, subject, predicate, depth, nodes, None)
}

/// Test-only forging entry: prove with a CALLER-CHOSEN density permutation row order (which output
/// position each cell lands at). A wrong order makes the delivered value column NOT value-in-index-order
/// and makes the routed index column differ from the public identity — so it is rejected by the density
/// certificate at verify. A row order that is not a full permutation of `[0,n)` (a drop/injection) is
/// rejected at prove by the permutation gadget's own validation.
#[cfg(test)]
pub(super) fn prove_ordered_array_walk_forged(
    context: &StoreProverContext<'_>,
    layer: &LayerCommitment,
    plan: &RelationProofPlan,
    subject: &CanonicalObject,
    predicate: &[u8],
    row_order_override: Vec<usize>,
) -> io::Result<RecursiveRelationEnvelope> {
    let mut nodes = 0usize;
    prove_ordered_array_walk_impl(
        context,
        layer,
        plan,
        subject,
        predicate,
        1,
        &mut nodes,
        Some(row_order_override),
    )
}

fn prove_ordered_array_walk_impl(
    context: &StoreProverContext<'_>,
    layer: &LayerCommitment,
    plan: &RelationProofPlan,
    subject: &CanonicalObject,
    predicate: &[u8],
    depth: usize,
    nodes: &mut usize,
    row_order_override: Option<Vec<usize>>,
) -> io::Result<RecursiveRelationEnvelope> {
    let root = layer.state.commitment_root;
    let digest = plan.query_digest;

    // (A) CELL SET: t(S,P,?cell), every row kept = the complete cell set.
    let cells_p = cells_plan(digest, subject, predicate)?;
    let cells = RecursiveRelationEnvelope::prove_bounded(context, layer, &cells_p, depth + 1, nodes)?;
    if cells.result_columns.len() != 1 {
        return Err(invalid("ordered-array-walk cell set must be a single node column"));
    }
    let n = cells.result_len;
    if n == 0 {
        return Err(invalid("ordered-array-walk array has no cells"));
    }
    let cell_log_domain = cells.result_log_domain;
    let com_cells = cells.result_commitments[0];
    let cells_col = pad_to(&cells.result_columns[0], cell_log_domain);
    let c_witness = RelationWitness {
        columns: vec![cells_col.clone()],
        commitments: vec![com_cells],
        log_domain: cell_log_domain,
        n,
    };

    // (B/C/type) edge relations.
    let index_p = edge_relation_plan(digest, SYS_INDEX, b"index-edges");
    let index_edges = RecursiveRelationEnvelope::prove_bounded(context, layer, &index_p, depth + 1, nodes)?;
    let value_p = edge_relation_plan(digest, SYS_VALUE, b"value-edges");
    let value_edges = RecursiveRelationEnvelope::prove_bounded(context, layer, &value_p, depth + 1, nodes)?;
    let type_p = edge_relation_plan(digest, RDF_TYPE, b"type-edges");
    let type_edges = RecursiveRelationEnvelope::prove_bounded(context, layer, &type_p, depth + 1, nodes)?;

    // (D) index functional bind: exactly one sys:index per cell, positionally aligned.
    let (com_index, index_col, index_bind) = prove_edge_bind(
        context, &root, &digest, &c_witness, &cells_col, n, cell_log_domain, &index_edges, b"index",
    )?;
    // (E) value functional bind: exactly one sys:value per cell, positionally aligned.
    let (com_value, value_col, value_bind) = prove_edge_bind(
        context, &root, &digest, &c_witness, &cells_col, n, cell_log_domain, &value_edges, b"value",
    )?;
    // (F) type functional bind + equality filter forcing rdf:type == sys:Array for EVERY cell.
    let (com_type, type_col, type_bind) = prove_edge_bind(
        context, &root, &digest, &c_witness, &cells_col, n, cell_log_domain, &type_edges, b"type",
    )?;
    let sys_array_id = resolve_node_id_or_absent(&layer.verifier_view(), SYS_ARRAY);
    let type_witness = RelationWitness {
        columns: vec![type_col.clone()],
        commitments: vec![com_type],
        log_domain: cell_log_domain,
        n,
    };
    let array_constraint = RelationEqualityConstraint::ColumnConstant {
        column: 0,
        value: ArkFr::from_u64(sys_array_id),
    };
    let mut t_type = aw_transcript(&root, &digest, b"type-membership");
    let type_eq =
        prove_relation_equality_with_context(context, &type_witness, &array_constraint, &mut t_type)?;
    if type_eq.n != n {
        return Err(invalid(
            "ordered-array-walk has a cell whose rdf:type is not sys:Array",
        ));
    }

    // (G) DENSITY + ORDER certificate.
    let density = prove_density_order(
        context,
        layer,
        &root,
        &digest,
        com_index,
        &index_col,
        com_value,
        &value_col,
        n,
        cell_log_domain,
        row_order_override,
    )?;

    let proof = OrderedArrayWalkProof {
        cells: Box::new(cells),
        index_edges: Box::new(index_edges),
        value_edges: Box::new(value_edges),
        type_edges: Box::new(type_edges),
        cons_len: n,
        cell_log_domain,
        index_join: index_bind.join,
        index_join_out: index_bind.join_out,
        index_reloc_subj: index_bind.reloc_subj,
        index_reloc_obj: index_bind.reloc_obj,
        com_index,
        value_join: value_bind.join,
        value_join_out: value_bind.join_out,
        value_reloc_subj: value_bind.reloc_subj,
        value_reloc_obj: value_bind.reloc_obj,
        com_value,
        type_join: type_bind.join,
        type_join_out: type_bind.join_out,
        type_reloc_subj: type_bind.reloc_subj,
        type_reloc_obj: type_bind.reloc_obj,
        com_type,
        type_equality: type_eq.proof,
        type_equality_out: type_eq.commitments,
        common_log_domain: density.common_log_domain,
        reloc_index_at_d: density.reloc_index_at_d,
        com_index_at_d: density.com_index_at_d,
        reloc_value_at_d: density.reloc_value_at_d,
        com_value_at_d: density.com_value_at_d,
        reloc_dict_id: density.reloc_dict_id,
        com_dict_id_at_d: density.com_dict_id_at_d,
        reloc_dict_value: density.reloc_dict_value,
        com_dict_value_at_d: density.com_dict_value_at_d,
        reloc_dict_datatype: density.reloc_dict_datatype,
        com_dict_datatype_at_d: density.com_dict_datatype_at_d,
        lookup: density.lookup,
        datatype_filter: density.datatype_filter,
        datatype_selection_commitment: density.datatype_selection_commitment,
        datatype_inverse_commitment: density.datatype_inverse_commitment,
        order_perm: density.order_perm,
        com_out: density.com_out,
    };

    let result_column = density.out_col[..n].to_vec();
    RecursiveRelationEnvelope::finish_proved(
        layer,
        plan,
        RecursiveRelationProof::OrderedArrayWalk(Box::new(proof)),
        vec![density.com_out],
        vec![result_column],
        density.common_log_domain,
        n,
    )
}

/// The prover outputs of the density+order certificate (the one new glue).
struct DensityProve {
    common_log_domain: usize,
    reloc_index_at_d: RelocationProof,
    com_index_at_d: ArkGT,
    reloc_value_at_d: RelocationProof,
    com_value_at_d: ArkGT,
    reloc_dict_id: RelocationProof,
    com_dict_id_at_d: ArkGT,
    reloc_dict_value: RelocationProof,
    com_dict_value_at_d: ArkGT,
    reloc_dict_datatype: RelocationProof,
    com_dict_datatype_at_d: ArkGT,
    lookup: DictLookupProof,
    datatype_filter: FilterProof,
    datatype_selection_commitment: ArkGT,
    datatype_inverse_commitment: ArkGT,
    order_perm: RelationPermutationProof,
    com_out: ArkGT,
    out_col: Vec<ArkFr>,
}

/// Prove the density+order certificate. Authenticates each cell's `sys:index` id to its dictionary
/// value/datatype, forces the datatype, decodes to a `u64` position, rejects `p_i ≥ n`, and routes the
/// `(position, value)` pairs through ONE permutation onto `(identity, Out)`. See the module docs.
fn prove_density_order(
    context: &StoreProverContext<'_>,
    layer: &LayerCommitment,
    root: &ProofHash,
    digest: &ProofHash,
    com_index: ArkGT,
    index_col: &[ArkFr],
    com_value: ArkGT,
    value_col: &[ArkFr],
    n: usize,
    cell_log_domain: usize,
    row_order_override: Option<Vec<usize>>,
) -> io::Result<DensityProve> {
    // Authenticated dictionary columns (id / order-preserving value / datatype), row-aligned.
    let n_dict = layer.node_value_catalog.len();
    let (_, dict_id) = layer.node_value_dictionary_columns()?;
    let dict_value = layer.node_value_dictionary_value_column()?;
    let dict_datatype = layer.node_value_dictionary_datatype_column()?;
    if dict_id.is_empty() || !dict_id.len().is_power_of_two() {
        return Err(invalid("ordered-array-walk requires a nonempty authenticated dictionary"));
    }
    let dict_log_domain = dict_id.len().trailing_zeros() as usize;
    let common_log_domain = cell_log_domain.max(dict_log_domain);
    let common_domain = 1usize << common_log_domain;

    let pad = |column: &[ArkFr]| -> Vec<ArkFr> {
        let mut result = column.to_vec();
        result.resize(common_domain, ArkFr::zero());
        result
    };

    let mut t = aw_transcript(root, digest, b"density");
    t.append_bytes(b"pow/array-walk/rows", &(n as u64).to_le_bytes());
    t.append_bytes(b"pow/array-walk/common-log-domain", &(common_log_domain as u64).to_le_bytes());

    // Lift the cell-aligned index/value columns and the dictionary columns UP to the shared domain. The
    // relocation binds each lifted column to the same authenticated source commitment, so the cell
    // alignment is preserved across the domain change.
    let index_at_d = pad(&index_col[..n.min(index_col.len())]);
    let com_index_at_d = context.commit_column(&index_at_d)?;
    let reloc_index_at_d = prove_column_relocation_with_context(
        context, index_col, &com_index, &index_at_d, &com_index_at_d, &mut t,
    )?;

    let value_at_d = pad(&value_col[..n.min(value_col.len())]);
    let com_value_at_d = context.commit_column(&value_at_d)?;
    let reloc_value_at_d = prove_column_relocation_with_context(
        context, value_col, &com_value, &value_at_d, &com_value_at_d, &mut t,
    )?;

    let dict_id_at_d = pad(&dict_id);
    let dict_value_at_d = pad(&dict_value);
    let dict_datatype_at_d = pad(&dict_datatype);
    let com_dict_id_at_d = context.commit_column(&dict_id_at_d)?;
    let reloc_dict_id = prove_column_relocation_with_context(
        context, &dict_id, &layer.node_value_dict_commitments.1, &dict_id_at_d, &com_dict_id_at_d, &mut t,
    )?;
    let com_dict_value_at_d = context.commit_column(&dict_value_at_d)?;
    let reloc_dict_value = prove_column_relocation_with_context(
        context, &dict_value, &layer.node_value_dict_range_commitments.0, &dict_value_at_d,
        &com_dict_value_at_d, &mut t,
    )?;
    let com_dict_datatype_at_d = context.commit_column(&dict_datatype_at_d)?;
    let reloc_dict_datatype = prove_column_relocation_with_context(
        context, &dict_datatype, &layer.node_value_dict_range_commitments.1, &dict_datatype_at_d,
        &com_dict_datatype_at_d, &mut t,
    )?;

    // Authenticate each index id -> (value, datatype). `com_v_val` becomes the permutation's position
    // column; `com_v_dt` feeds the datatype filter.
    let lookup = prove_dict_value_lookup_with_context(
        context, &index_at_d, &com_index_at_d, &dict_id_at_d, &dict_value_at_d, &dict_datatype_at_d,
        &com_dict_id_at_d, &com_dict_value_at_d, &com_dict_datatype_at_d, n, n_dict, &mut t,
    )?;
    let (v_val, v_dt, _w) =
        honest_dict_lookup(&index_at_d, &dict_id_at_d, &dict_value_at_d, &dict_datatype_at_d, n, n_dict);

    // Datatype completeness: EVERY index must be the required order-preserving unsigned type, so its
    // decoded value is a genuine numeric position. `sel == χ_n` proves no index has a foreign datatype.
    let expected_datatype = ArkFr::from_u64(INDEX_DATATYPE as u64);
    let (datatype_filter, datatype_selection_commitment, datatype_inverse_commitment, _) =
        prove_filter_indicator_with_context(context, &v_dt, expected_datatype, n, &lookup.com_v_dt, &mut t)?;
    let mut chi = vec![ArkFr::zero(); common_domain];
    chi[..n].fill(ArkFr::one());
    if datatype_selection_commitment != context.commit_column(&chi)? {
        return Err(invalid(
            "ordered-array-walk has a sys:index of the wrong datatype (not a decodable position)",
        ));
    }

    // Decode each authenticated value to its integer position and reject any out-of-range index. The
    // permutation-to-identity below is the ACTUAL density soundness; this is a fail-fast bound.
    let mut positions = Vec::with_capacity(n);
    for value in v_val.iter().take(n) {
        let p = generated_decimal_u64_value(*value)?;
        let p = usize::try_from(p).map_err(|_| invalid("ordered-array-walk index does not fit usize"))?;
        if p >= n {
            return Err(invalid(
                "ordered-array-walk sys:index is >= the cell count (sparse or oversized array)",
            ));
        }
        positions.push(p);
    }

    // The honest row order: output position `j` is filled by the cell whose index decodes to `j`. A
    // duplicate index leaves a slot doubly-claimed (rejected here); with distinct indices in `[0,n)`
    // this is a full bijection.
    let mut honest_row_order = vec![usize::MAX; n];
    for (i, &p) in positions.iter().enumerate() {
        if honest_row_order[p] != usize::MAX {
            return Err(invalid(
                "ordered-array-walk has a duplicate sys:index (not a dense 0..n-1 permutation)",
            ));
        }
        honest_row_order[p] = i;
    }

    let forging = row_order_override.is_some();
    let row_order = row_order_override.unwrap_or(honest_row_order);

    // Route [position, value] -> [output-position, Out] by `row_order`. When honest, output-position
    // is exactly the identity `[0,1,…,n-1]`.
    let input_witness = RelationWitness {
        columns: vec![v_val.clone(), value_at_d.clone()],
        commitments: vec![lookup.com_v_val, com_value_at_d],
        log_domain: common_log_domain,
        n,
    };
    let perm = prove_relation_permutation_with_context(context, &input_witness, &row_order, &mut t)?;
    if perm.commitments.len() != 2 || perm.columns.len() != 2 {
        return Err(invalid("ordered-array-walk density permutation has the wrong arity"));
    }

    // The identity column the verifier will reconstruct from `n`. When honest, the routed position
    // column equals it; assert that here so an honest proof is self-consistent (a forged row order is
    // deliberately allowed through so the test can confirm VERIFY rejects it).
    let com_identity = identity_commitment(n, common_log_domain)?;
    if !forging && perm.commitments[0] != com_identity {
        return Err(invalid(
            "ordered-array-walk routed index column is not the dense identity 0..n-1",
        ));
    }

    Ok(DensityProve {
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
        order_perm: perm.proof,
        com_out: perm.commitments[1],
        out_col: perm.columns[1].clone(),
    })
}

/// The verifier-reconstructed identity relation commitment `Com([0,1,…,n-1])` padded to `2^log_domain`.
/// Both prover and verifier build it from `n` alone — this is what forces the routed index column to be
/// exactly the dense identity (and hence `{p_i}` to be a bijection onto `{0..n-1}`).
fn identity_commitment(n: usize, log_domain: usize) -> io::Result<ArkGT> {
    let mut column = vec![ArkFr::zero(); 1usize << log_domain];
    for (j, slot) in column.iter_mut().enumerate().take(n) {
        *slot = ArkFr::from_u64(j as u64);
    }
    commit_column(&column)
}

// ---------------------------------------------------------------------------
// VERIFY.
// ---------------------------------------------------------------------------

impl RecursiveRelationEnvelope {
    /// Verify an ordered `sys:Array` walk against the trusted root: re-verify each child relation, then
    /// replay every gluing gadget against the children's authenticated commitments, binding the
    /// delivered value column to the root. Mirrors `prove_ordered_array_walk` step for step.
    pub(super) fn verify_ordered_array_walk(
        &self,
        layer: &impl VerifierLayer,
        expected_root: ProofHash,
        plan: &RelationProofPlan,
        subject: &CanonicalObject,
        predicate: &[u8],
        walk: &OrderedArrayWalkProof,
        depth: usize,
        nodes: &mut usize,
    ) -> io::Result<()> {
        let root = expected_root;
        let digest = plan.query_digest;

        // (A) CELL SET: re-verify and read the complete cell set.
        let cells_p = cells_plan(digest, subject, predicate)?;
        walk.cells.verify_bounded(layer, root, &cells_p, depth + 1, nodes)?;
        if walk.cells.result_columns.len() != 1 {
            return Err(invalid("ordered-array-walk cell set must be a single node column"));
        }
        let n = walk.cells.result_len;
        if n == 0 || n != walk.cons_len {
            return Err(invalid("ordered-array-walk cell count mismatch"));
        }
        let cell_log_domain = walk.cells.result_log_domain;
        if cell_log_domain != walk.cell_log_domain {
            return Err(invalid("ordered-array-walk cell domain mismatch"));
        }
        let c_commit = walk.cells.relation_commit(None)?;

        // (B/C/type) edge relations.
        let index_p = edge_relation_plan(digest, SYS_INDEX, b"index-edges");
        walk.index_edges.verify_bounded(layer, root, &index_p, depth + 1, nodes)?;
        let index_commit = walk.index_edges.relation_commit(None)?;
        let value_p = edge_relation_plan(digest, SYS_VALUE, b"value-edges");
        walk.value_edges.verify_bounded(layer, root, &value_p, depth + 1, nodes)?;
        let value_commit = walk.value_edges.relation_commit(None)?;
        let type_p = edge_relation_plan(digest, RDF_TYPE, b"type-edges");
        walk.type_edges.verify_bounded(layer, root, &type_p, depth + 1, nodes)?;
        let type_commit = walk.type_edges.relation_commit(None)?;

        // (D) index functional bind.
        verify_edge_bind(
            &root, &digest, &c_commit, &index_commit, n, cell_log_domain,
            &walk.index_join, &walk.index_join_out, &walk.index_reloc_subj, &walk.index_reloc_obj,
            walk.com_index, b"index",
        )?;
        // (E) value functional bind.
        verify_edge_bind(
            &root, &digest, &c_commit, &value_commit, n, cell_log_domain,
            &walk.value_join, &walk.value_join_out, &walk.value_reloc_subj, &walk.value_reloc_obj,
            walk.com_value, b"value",
        )?;
        // (F) type functional bind + equality filter forcing rdf:type == sys:Array for EVERY cell.
        verify_edge_bind(
            &root, &digest, &c_commit, &type_commit, n, cell_log_domain,
            &walk.type_join, &walk.type_join_out, &walk.type_reloc_subj, &walk.type_reloc_obj,
            walk.com_type, b"type",
        )?;
        let sys_array_id = resolve_node_id_or_absent(&layer.verifier_view(), SYS_ARRAY);
        let type_commit_c = RelationCommit {
            commitments: vec![walk.com_type],
            log_domain: cell_log_domain,
            n,
        };
        let array_constraint = RelationEqualityConstraint::ColumnConstant {
            column: 0,
            value: ArkFr::from_u64(sys_array_id),
        };
        let mut t_type = aw_transcript(&root, &digest, b"type-membership");
        verify_relation_equality(
            &type_commit_c, &array_constraint, &walk.type_equality_out, n, cell_log_domain,
            &walk.type_equality, &mut t_type,
        )?;

        // (G) DENSITY + ORDER certificate.
        self.verify_density_order(layer, &root, &digest, walk, n, cell_log_domain)?;

        // (8) Bind the delivered relation to the proven ordered value column.
        if self.result_len != n
            || self.result_log_domain != walk.common_log_domain
            || self.result_commitments != vec![walk.com_out]
        {
            return Err(invalid(
                "ordered-array-walk delivered relation is not the proven ordered value column",
            ));
        }
        Ok(())
    }

    /// Verify the density+order certificate. Re-verifies each relocation, the authenticated dictionary
    /// lookup, the datatype-completeness filter, and the permutation whose OUTPUT index column the
    /// verifier reconstructs as the public identity `[0..n-1]` — which is what certifies density.
    fn verify_density_order(
        &self,
        layer: &impl VerifierLayer,
        root: &ProofHash,
        digest: &ProofHash,
        walk: &OrderedArrayWalkProof,
        n: usize,
        cell_log_domain: usize,
    ) -> io::Result<()> {
        let view = layer.verifier_view();
        // Merkle-dict I2b S2 fail-close: the ordered-array-walk density certificate authenticates
        // each cell's `sys:index` id to its order-preserving dictionary VALUE/DATATYPE and decodes
        // it to a position — the exact value/datatype read a TERM-LESS artifact severs from real
        // terms. A pure-resolve ReadDocument works term-less, but this value-decoding walk must not.
        view.require_terms_present("ordered-array-walk density (index value/datatype decode)")?;
        let n_dict = view.node_value_catalog.len();
        let (_, dict_id) = view.node_value_dictionary_columns()?;
        if dict_id.is_empty() || !dict_id.len().is_power_of_two() {
            return Err(invalid("ordered-array-walk requires a nonempty authenticated dictionary"));
        }
        let dict_log_domain = dict_id.len().trailing_zeros() as usize;
        let common_log_domain = cell_log_domain.max(dict_log_domain);
        if common_log_domain != walk.common_log_domain {
            return Err(invalid("ordered-array-walk working domain mismatch"));
        }

        let mut t = aw_transcript(root, digest, b"density");
        t.append_bytes(b"pow/array-walk/rows", &(n as u64).to_le_bytes());
        t.append_bytes(b"pow/array-walk/common-log-domain", &(common_log_domain as u64).to_le_bytes());

        // Relocations lifting the cell-aligned index/value columns and the dictionary columns UP.
        verify_column_relocation(
            &walk.com_index, cell_log_domain, &walk.com_index_at_d, common_log_domain,
            &walk.reloc_index_at_d, &mut t,
        )?;
        verify_column_relocation(
            &walk.com_value, cell_log_domain, &walk.com_value_at_d, common_log_domain,
            &walk.reloc_value_at_d, &mut t,
        )?;
        verify_column_relocation(
            &view.node_value_dict_commitments.1, dict_log_domain, &walk.com_dict_id_at_d,
            common_log_domain, &walk.reloc_dict_id, &mut t,
        )?;
        verify_column_relocation(
            &view.node_value_dict_range_commitments.0, dict_log_domain, &walk.com_dict_value_at_d,
            common_log_domain, &walk.reloc_dict_value, &mut t,
        )?;
        verify_column_relocation(
            &view.node_value_dict_range_commitments.1, dict_log_domain, &walk.com_dict_datatype_at_d,
            common_log_domain, &walk.reloc_dict_datatype, &mut t,
        )?;

        // Authenticate each index id -> (value, datatype).
        verify_dict_value_lookup(
            &walk.com_index_at_d, &walk.com_dict_id_at_d, &walk.com_dict_value_at_d,
            &walk.com_dict_datatype_at_d, n, n_dict, &walk.lookup, &mut t,
        )?;

        // Datatype completeness: every index is the required order-preserving unsigned type.
        let expected_datatype = ArkFr::from_u64(INDEX_DATATYPE as u64);
        verify_filter_indicator(
            &walk.lookup.com_v_dt, expected_datatype, n, &walk.datatype_selection_commitment,
            &walk.datatype_inverse_commitment, &walk.datatype_filter, &mut t,
        )?;
        let mut chi = vec![ArkFr::zero(); 1usize << common_log_domain];
        chi[..n].fill(ArkFr::one());
        if walk.datatype_selection_commitment != commit_column(&chi)? {
            return Err(invalid(
                "ordered-array-walk has a sys:index of the wrong datatype (not a decodable position)",
            ));
        }

        // The density permutation: input [position(=v_val), value], output [identity, Out]. The
        // verifier SUPPLIES the reconstructed identity commitment as the output index column, so a
        // routed index column that is not exactly `[0..n-1]` (a gap, a duplicate, an out-of-range index,
        // or a reordered delivery) makes the permutation fail — this is the density certificate.
        let input_commit = RelationCommit {
            commitments: vec![walk.lookup.com_v_val, walk.com_value_at_d],
            log_domain: common_log_domain,
            n,
        };
        let com_identity = identity_commitment(n, common_log_domain)?;
        verify_relation_permutation(
            &input_commit,
            &[com_identity, walk.com_out],
            n,
            common_log_domain,
            &walk.order_perm,
            &mut t,
        )
    }
}

// ===========================================================================
// RECONCILED ordered `sys:Array` walk over a DELTA HEAD's `present` (ReadDocument increment 6d).
//
// This is the PARALLEL sibling of the materialized [`prove_ordered_array_walk`] above: it proves the
// EXACT ordered `sys:value` sequence of a stored 1-D DENSE array at a bound `(subject, predicate)` over
// a NON-materialized delta stack (base..head, head last), reconciling the current triple set at proof
// time. It closely MIRRORS the reconciled `rdf:List` walk (6c-ii, `list_walk.rs`) — reconcile ONCE,
// read every leaf over `present` via the 6c-i FOUNDATION, and run the SAME source-agnostic gluing
// gadgets — but an array has NO closure (the parent points at every cell, so the bound-subject cell
// leaf IS the whole membership certificate; FOUNDATION-A only). The materialized path is UNTOUCHED.
//
// # The three source-specific differences from the materialized array walk
//
// 1. **Reconcile ONCE.** `prove_reconcile_core` runs a single O(chain) reconciliation; the cell leaf
//    `t(S,P,?cell)`, the `sys:index`/`sys:value`/`rdf:type` edge leaves are read against the SHARED
//    `ReconcileArtifacts` on the SHARED transcript (FOUNDATION-A `prove_reconciled_leaf`), each
//    O(present). Every leaf's committed columns are bound to the trusted head root via the audited
//    reconcile invariant + `present_bgp` completeness. The gluing gadgets run on FRESH transcripts.
// 2. **THE WRINKLE — the index decode authenticates against the COMBINED range dict.** The materialized
//    density certificate decodes each `sys:index` id to a `u64` position by authenticating it against a
//    SINGLE layer's `(id, value, datatype)` dictionary. Over `present` a cell's index id is a
//    combined-dict id whose canonical value lives in whichever layer minted it, so it MUST be
//    authenticated against the COMBINED `(id, value, datatype)` range dictionary — the exact bag-union
//    dict `present_bgp`'s ranged atom builds ([`prove_combined_range_dict`], every branch root-bound).
//    This is the one genuinely-new soundness seam: a forged / mis-decoded index would corrupt the
//    order/density, and the range-dict authentication + the density permutation-to-identity together
//    pin the true order over the chain.
// 3. **Values resolved over the CHAIN catalogs.** The delivered ordered value column is resolved via
//    `resolve_node_value_id_column` (the combined-dict `NodeValueId` chain resolver), NOT the
//    single-layer resolver — its ids' canonical bytes live in whichever layer minted them.
//
// Every gluing gadget — the three functional `prove_edge_bind_witness`es (index/value/type), the
// `sys:Array` type-equality, and the DENSITY permutation-to-identity — is BYTE-IDENTICAL to the
// materialized path; only the leaf source and the dict-lookup source change.

/// The domain-separation digest for a reconciled ordered-array walk's gluing-gadget transcripts,
/// deterministic from `(subject, predicate)` so prover and verifier reconstruct the IDENTICAL
/// `aw_transcript(root, digest, role)` / `lw_transcript` (functional-bind) sequences. A DISTINCT domain
/// tag keeps it disjoint from BOTH the materialized array walk's `plan.query_digest` AND the reconciled
/// LIST walk's digest, so no sub-proof is fungible across walks.
fn reconciled_array_walk_digest(subject: &CanonicalObject, predicate: &[u8]) -> io::Result<ProofHash> {
    let subject_bytes = match subject {
        CanonicalObject::Node(bytes) => bytes,
        _ => return Err(invalid("reconciled ordered-array-walk subject must be a node IRI")),
    };
    let mut hasher = Sha256::new();
    hasher.update(b"pow/reconciled-ordered-array-walk/digest/v0");
    hasher.update((subject_bytes.len() as u64).to_le_bytes());
    hasher.update(subject_bytes);
    hasher.update((predicate.len() as u64).to_le_bytes());
    hasher.update(predicate);
    let mut out = [0u8; 32];
    out.copy_from_slice(&hasher.finalize());
    Ok(out)
}

/// Every sub-proof of one reconciled ordered `sys:Array` walk. Compared to the materialized
/// [`OrderedArrayWalkProof`], the four child ENVELOPES are replaced by ONE [`ReconcileProof`] + the
/// per-leaf [`PresentBgpProof`]s; the density certificate carries the [`CombinedRangeDictProof`]
/// authenticating each `sys:index` id over the chain (in place of the materialized single-layer dict);
/// and EVERY gluing-gadget field is byte-identical to the materialized proof.
#[derive(Clone, Debug)]
pub struct OrderedArrayWalkReconciledProof {
    /// Proves `present` IS the current triple set of the whole delta stack (bound to the head root).
    pub reconcile: ReconcileProof,
    /// FOUNDATION-A cell leaf `t(S,P,?cell)` over `present` — the COMPLETE cell set (n cells).
    pub cells_leaf: PresentBgpProof,
    /// FOUNDATION-A `t(?s,sys:index,?o)` leaf over `present` — the index functional-bind edge.
    pub index_edge_leaf: PresentBgpProof,
    /// FOUNDATION-A `t(?s,sys:value,?o)` leaf over `present` — the value functional-bind edge.
    pub value_edge_leaf: PresentBgpProof,
    /// FOUNDATION-A `t(?s,rdf:type,?o)` leaf over `present` — the type functional-bind edge.
    pub type_edge_leaf: PresentBgpProof,

    pub n: usize,
    pub cell_log_domain: usize,

    // ---- Gluing gadgets — BYTE-IDENTICAL to `OrderedArrayWalkProof` ----
    /// (D) functional index bind: one `sys:index` per cell, positionally aligned.
    pub index_join: InnerJoinProof,
    pub index_join_out: Vec<ArkGT>,
    pub index_reloc_subj: RelocationProof,
    pub index_reloc_obj: RelocationProof,
    pub com_index: ArkGT,
    /// (E) functional value bind: one `sys:value` per cell, positionally aligned.
    pub value_join: InnerJoinProof,
    pub value_join_out: Vec<ArkGT>,
    pub value_reloc_subj: RelocationProof,
    pub value_reloc_obj: RelocationProof,
    pub com_value: ArkGT,
    /// (F) functional type bind + equality filter forcing `rdf:type == sys:Array`.
    pub type_join: InnerJoinProof,
    pub type_join_out: Vec<ArkGT>,
    pub type_reloc_subj: RelocationProof,
    pub type_reloc_obj: RelocationProof,
    pub com_type: ArkGT,
    pub type_equality: RelationEqualityProof,
    pub type_equality_out: Vec<ArkGT>,

    // (G) density + order certificate over the COMBINED range dict -----------------------------------
    /// The combined `(id, value, datatype)` range dictionary authenticating every `sys:index` id over
    /// the chain (the 6d wrinkle; in place of the materialized single-layer dict). Its `com_*_at_k`
    /// commitments live at the reconciliation working domain and are re-verified against the root-bound
    /// per-layer branch commitments.
    pub range_dict: CombinedRangeDictProof,
    pub common_log_domain: usize,
    pub reloc_index_at_d: RelocationProof,
    pub com_index_at_d: ArkGT,
    pub reloc_value_at_d: RelocationProof,
    pub com_value_at_d: ArkGT,
    pub reloc_dict_id: RelocationProof,
    pub com_dict_id_at_d: ArkGT,
    pub reloc_dict_value: RelocationProof,
    pub com_dict_value_at_d: ArkGT,
    pub reloc_dict_datatype: RelocationProof,
    pub com_dict_datatype_at_d: ArkGT,
    pub lookup: DictLookupProof,
    pub datatype_filter: FilterProof,
    pub datatype_selection_commitment: ArkGT,
    pub datatype_inverse_commitment: ArkGT,
    pub order_perm: RelationPermutationProof,
    pub com_out: ArkGT,

    /// The ordered value combined-dict ids (PUBLIC): bound to `com_out` by re-commit at verify, then
    /// resolved to `CanonicalObject`s over the chain catalogs.
    pub value_ids: Vec<u64>,

    /// C3b: the prover-shipped, UNCOMMITTED Merkle openings for the delivered [`Self::value_ids`],
    /// used to render them to IRIs / literals over a TERM-LESS head (`resolve_node_value_id_column`).
    /// `None` on the ordinary term-bearing path (the resolver rebuilds each id-indexed tree from the
    /// clear catalog and ignores them); minted by [`attach_termless_array_walk_openings`]. IN-MEMORY
    /// ONLY — like [`super::list_walk::OrderedListWalkReconciledProof::element_openings`], NEVER folded
    /// into `commitment_root`/`resolution_root` and carrying NO wire/golden/version change (reconciled
    /// walk proofs are not in `wire.rs`).
    pub value_openings: Option<ResolutionOpenings>,
    /// C3b: the prover-shipped, UNCOMMITTED Merkle opening for the `sys:Array` class constant, used to
    /// resolve+belt-hash its combined-dict id over a TERM-LESS head
    /// (`resolve_combined_node_id_termless`) — that id keys the `rdf:type == sys:Array` equality, so a
    /// wrong id would bypass the type check. `None` on the term-bearing path (resolved by the audited
    /// catalog scan). Same in-memory-only discipline as `value_openings`.
    pub sys_array_opening: Option<DictOpening>,
    /// C3b (the C2a value re-pin witness): the per-cell `sys:index` id column — the SAME column the
    /// density lookup's `com_index_at_d` / `com_obj_id` commits. [`verify_dict_value_terms_chain`]
    /// recommits it to `com_index_at_d` and opens each DISTINCT id's TERM cross-layer to re-derive the
    /// order-preserving VALUE/DATATYPE, pinning the density order key to the real terms (the array
    /// REORDER forge a term-less head opens). Populated at prove from the density index column (the
    /// prover already holds it); the opening set is minted separately.
    pub index_oid: Vec<u64>,
    /// C3b: the prover-shipped, UNCOMMITTED Merkle openings over the DISTINCT [`Self::index_oid`] ids,
    /// used by [`verify_dict_value_terms_chain`] to open each index literal's term over a TERM-LESS
    /// head. `None` on the term-bearing path (its committed value/datatype are catalog-tied by each
    /// view's `validate()` re-MSM); leaving it unset keeps a term-less density fail-closed. Same
    /// in-memory-only discipline as `value_openings`.
    pub index_value_openings: Option<ResolutionOpenings>,
}

/// Prove the ordered `sys:value` sequence of the `sys:Array` at `(subject, predicate)` over the
/// reconciled `present` of the delta stack `layers` (base..head, `len >= 2`). Returns the proof and the
/// ordered value column resolved against the chain catalogs. Every failure path rejects CLOSED, never
/// emitting a bogus order — exactly like the materialized entry.
pub fn prove_ordered_array_walk_reconciled(
    layers: &[&LayerCommitment],
    subject: &CanonicalObject,
    predicate: &[u8],
) -> io::Result<(OrderedArrayWalkReconciledProof, Vec<CanonicalObject>)> {
    prove_ordered_array_walk_reconciled_impl(layers, subject, predicate, None)
}

/// Test-only forging entry mirroring [`prove_ordered_array_walk_forged`]: prove with a CALLER-CHOSEN
/// density permutation row order, to confirm a broken order/density is rejected by the density
/// permutation-to-identity at verify (a wrong order makes the routed index column differ from the
/// public identity `[0,n-1]`; a non-permutation of `[0,n)` is rejected at prove by the gadget).
#[cfg(test)]
pub fn prove_ordered_array_walk_reconciled_forged(
    layers: &[&LayerCommitment],
    subject: &CanonicalObject,
    predicate: &[u8],
    row_order_override: Vec<usize>,
) -> io::Result<(OrderedArrayWalkReconciledProof, Vec<CanonicalObject>)> {
    prove_ordered_array_walk_reconciled_impl(layers, subject, predicate, Some(row_order_override))
}

fn prove_ordered_array_walk_reconciled_impl(
    layers: &[&LayerCommitment],
    subject: &CanonicalObject,
    predicate: &[u8],
    row_order_override: Option<Vec<usize>>,
) -> io::Result<(OrderedArrayWalkReconciledProof, Vec<CanonicalObject>)> {
    let context = StoreProverContext::cpu();
    let root = layers
        .last()
        .ok_or_else(|| invalid("reconciled ordered-array-walk needs a head layer"))?
        .state
        .commitment_root;
    let digest = reconciled_array_walk_digest(subject, predicate)?;
    let views: Vec<crate::proof::VerifierCommitmentRef<'_>> =
        layers.iter().map(|l| l.verifier_view()).collect();

    // Reconcile ONCE, then read every leaf against the SHARED artifacts on the SHARED transcript.
    let mut transcript = Transcript254::new(RECONCILE_TRANSCRIPT);
    let (reconcile, artifacts) = prove_reconcile_core(&context, layers, &mut transcript)?;

    // (A) CELL SET leaf t(S,P,?cell): the COMPLETE cell set (every row kept), NOT a single head.
    let cells_atom = bound_subject_leaf_atom(subject, predicate);
    let (cells_leaf, c_witness) =
        prove_reconciled_leaf(&context, layers, &artifacts, cells_atom, &mut transcript)?;
    if c_witness.columns.len() != 1 {
        return Err(invalid("reconciled ordered-array-walk cell set must be a single node column"));
    }
    let n = c_witness.n;
    if n == 0 {
        return Err(invalid("reconciled ordered-array-walk array has no cells"));
    }
    let cell_log_domain = c_witness.log_domain;
    let cells_col = c_witness.columns[0].clone();

    // (B/C/type) EDGE leaves (shared transcript, in this order: index, value, type).
    let index_atom = predicate_edge_leaf_atom(SYS_INDEX);
    let (index_edge_leaf, index_edge_witness) =
        prove_reconciled_leaf(&context, layers, &artifacts, index_atom, &mut transcript)?;
    let value_atom = predicate_edge_leaf_atom(SYS_VALUE);
    let (value_edge_leaf, value_edge_witness) =
        prove_reconciled_leaf(&context, layers, &artifacts, value_atom, &mut transcript)?;
    let type_atom = predicate_edge_leaf_atom(RDF_TYPE);
    let (type_edge_leaf, type_edge_witness) =
        prove_reconciled_leaf(&context, layers, &artifacts, type_atom, &mut transcript)?;

    // (D/E/F) functional binds — the SAME source-agnostic core the materialized array walk + the
    // reconciled list walk use (`prove_edge_bind_witness`), consuming the reconciled leaf witnesses.
    let (com_index, index_col, index_bind) = prove_edge_bind_witness(
        &context, &root, &digest, &c_witness, &cells_col, n, cell_log_domain, &index_edge_witness,
        b"index",
    )?;
    let (com_value, value_col, value_bind) = prove_edge_bind_witness(
        &context, &root, &digest, &c_witness, &cells_col, n, cell_log_domain, &value_edge_witness,
        b"value",
    )?;
    let (com_type, type_col, type_bind) = prove_edge_bind_witness(
        &context, &root, &digest, &c_witness, &cells_col, n, cell_log_domain, &type_edge_witness,
        b"type",
    )?;

    // Type equality: force every cell's rdf:type == sys:Array (the sys:Array id is resolved over the
    // CHAIN catalogs, fail-closed on absence).
    let sys_array_id = resolve_combined_node_id(&views, SYS_ARRAY);
    let type_witness = RelationWitness {
        columns: vec![type_col.clone()],
        commitments: vec![com_type],
        log_domain: cell_log_domain,
        n,
    };
    let array_constraint = RelationEqualityConstraint::ColumnConstant {
        column: 0,
        value: ArkFr::from_u64(sys_array_id),
    };
    let mut t_type = aw_transcript(&root, &digest, b"type-membership");
    let type_eq =
        prove_relation_equality_with_context(&context, &type_witness, &array_constraint, &mut t_type)?;
    if type_eq.n != n {
        return Err(invalid(
            "reconciled ordered-array-walk has a cell whose rdf:type is not sys:Array",
        ));
    }

    // (G) DENSITY + ORDER certificate over the COMBINED range dict (the 6d wrinkle).
    let density = prove_density_order_reconciled(
        &context, layers, artifacts.working_log, &root, &digest, com_index, &index_col, com_value,
        &value_col, n, cell_log_domain, row_order_override,
    )?;

    // (9) Resolve the delivered ordered value column over the chain catalogs (binds value_ids to
    // com_out first).
    let value_ids: Vec<u64> = density.out_col[..n]
        .iter()
        .map(|f| field_id(*f))
        .collect::<io::Result<Vec<_>>>()?;
    let elements =
        resolve_node_value_id_column(&views, &value_ids, density.com_out, density.common_log_domain, None)?;

    let proof = OrderedArrayWalkReconciledProof {
        reconcile,
        cells_leaf,
        index_edge_leaf,
        value_edge_leaf,
        type_edge_leaf,
        n,
        cell_log_domain,
        index_join: index_bind.join,
        index_join_out: index_bind.join_out,
        index_reloc_subj: index_bind.reloc_subj,
        index_reloc_obj: index_bind.reloc_obj,
        com_index,
        value_join: value_bind.join,
        value_join_out: value_bind.join_out,
        value_reloc_subj: value_bind.reloc_subj,
        value_reloc_obj: value_bind.reloc_obj,
        com_value,
        type_join: type_bind.join,
        type_join_out: type_bind.join_out,
        type_reloc_subj: type_bind.reloc_subj,
        type_reloc_obj: type_bind.reloc_obj,
        com_type,
        type_equality: type_eq.proof,
        type_equality_out: type_eq.commitments,
        range_dict: density.range_dict,
        common_log_domain: density.common_log_domain,
        reloc_index_at_d: density.reloc_index_at_d,
        com_index_at_d: density.com_index_at_d,
        reloc_value_at_d: density.reloc_value_at_d,
        com_value_at_d: density.com_value_at_d,
        reloc_dict_id: density.reloc_dict_id,
        com_dict_id_at_d: density.com_dict_id_at_d,
        reloc_dict_value: density.reloc_dict_value,
        com_dict_value_at_d: density.com_dict_value_at_d,
        reloc_dict_datatype: density.reloc_dict_datatype,
        com_dict_datatype_at_d: density.com_dict_datatype_at_d,
        lookup: density.lookup,
        datatype_filter: density.datatype_filter,
        datatype_selection_commitment: density.datatype_selection_commitment,
        datatype_inverse_commitment: density.datatype_inverse_commitment,
        order_perm: density.order_perm,
        com_out: density.com_out,
        value_ids,
        // Term-bearing default: values are rendered by the catalog resolver, sys:Array by the catalog
        // scan, and each index literal's value/datatype is catalog-tied by every view's `validate()`
        // re-MSM. The term-less producer attaches the openings via `attach_termless_array_walk_openings`;
        // a term-bearing verifier ignores them. `index_oid` IS populated at prove (the prover already
        // holds the density index column) so the producer can mint openings over it.
        value_openings: None,
        sys_array_opening: None,
        index_oid: density.index_oid,
        index_value_openings: None,
    };
    Ok((proof, elements))
}

/// The prover outputs of the reconciled density+order certificate.
struct DensityReconciledProve {
    range_dict: CombinedRangeDictProof,
    common_log_domain: usize,
    reloc_index_at_d: RelocationProof,
    com_index_at_d: ArkGT,
    reloc_value_at_d: RelocationProof,
    com_value_at_d: ArkGT,
    reloc_dict_id: RelocationProof,
    com_dict_id_at_d: ArkGT,
    reloc_dict_value: RelocationProof,
    com_dict_value_at_d: ArkGT,
    reloc_dict_datatype: RelocationProof,
    com_dict_datatype_at_d: ArkGT,
    lookup: DictLookupProof,
    datatype_filter: FilterProof,
    datatype_selection_commitment: ArkGT,
    datatype_inverse_commitment: ArkGT,
    order_perm: RelationPermutationProof,
    com_out: ArkGT,
    out_col: Vec<ArkFr>,
    /// C3b: the per-cell `sys:index` ids the lookup authenticated (the SAME real prefix
    /// `com_index_at_d` commits) — the value re-pin witness carried into the proof for the term-less
    /// density tie ([`verify_dict_value_terms_chain`]).
    index_oid: Vec<u64>,
}

/// Prove the reconciled density+order certificate. IDENTICAL to the materialized [`prove_density_order`]
/// except the authenticated dictionary is the COMBINED `(id, value, datatype)` range dict over the
/// whole chain (in place of a single layer's dict) — so a cell's `sys:index` id, wherever it was minted,
/// authenticates to its true order-preserving value + datatype. Authenticates each index id, forces the
/// datatype, decodes to a `u64` position, rejects `p_i >= n`, and routes `(position, value)` through ONE
/// permutation onto `(identity, Out)`.
#[allow(clippy::too_many_arguments)]
fn prove_density_order_reconciled(
    context: &StoreProverContext<'_>,
    layers: &[&LayerCommitment],
    working_log: usize,
    root: &ProofHash,
    digest: &ProofHash,
    com_index: ArkGT,
    index_col: &[ArkFr],
    com_value: ArkGT,
    value_col: &[ArkFr],
    n: usize,
    cell_log_domain: usize,
    row_order_override: Option<Vec<usize>>,
) -> io::Result<DensityReconciledProve> {
    let mut t = aw_transcript(root, digest, b"density");

    // (0) The COMBINED `(id, value, datatype)` range dictionary over the whole chain — the exact
    // bag-union dict the present-BGP ranged atom builds, every branch bound to its own layer root, all
    // relocated to the reconciliation working domain. This is what replaces the materialized array
    // walk's single-layer dict lookup.
    let branches = range_dict_branches(layers)?;
    let n_dict: usize = branches.iter().map(|branch| branch.n).sum();
    let dict = prove_combined_range_dict(context, &branches, working_log, &mut t)?;
    let dict_log_domain = working_log;
    let common_log_domain = cell_log_domain.max(dict_log_domain);
    let common_domain = 1usize << common_log_domain;

    let pad = |column: &[ArkFr]| -> Vec<ArkFr> {
        let mut result = column.to_vec();
        result.resize(common_domain, ArkFr::zero());
        result
    };

    t.append_bytes(b"pow/array-walk/rows", &(n as u64).to_le_bytes());
    t.append_bytes(b"pow/array-walk/common-log-domain", &(common_log_domain as u64).to_le_bytes());

    // Lift the cell-aligned index/value columns and the combined dict columns UP to the shared domain
    // (an identity relocation when the domains already agree; the source commitments are preserved).
    let index_at_d = pad(&index_col[..n.min(index_col.len())]);
    let com_index_at_d = context.commit_column(&index_at_d)?;
    let reloc_index_at_d = prove_column_relocation_with_context(
        context, index_col, &com_index, &index_at_d, &com_index_at_d, &mut t,
    )?;

    let value_at_d = pad(&value_col[..n.min(value_col.len())]);
    let com_value_at_d = context.commit_column(&value_at_d)?;
    let reloc_value_at_d = prove_column_relocation_with_context(
        context, value_col, &com_value, &value_at_d, &com_value_at_d, &mut t,
    )?;

    let dict_id_at_d = pad(&dict.id_at_k);
    let dict_value_at_d = pad(&dict.value_at_k);
    let dict_datatype_at_d = pad(&dict.datatype_at_k);
    let com_dict_id_at_d = context.commit_column(&dict_id_at_d)?;
    let reloc_dict_id = prove_column_relocation_with_context(
        context, &dict.id_at_k, &dict.com_id_at_k, &dict_id_at_d, &com_dict_id_at_d, &mut t,
    )?;
    let com_dict_value_at_d = context.commit_column(&dict_value_at_d)?;
    let reloc_dict_value = prove_column_relocation_with_context(
        context, &dict.value_at_k, &dict.com_value_at_k, &dict_value_at_d, &com_dict_value_at_d, &mut t,
    )?;
    let com_dict_datatype_at_d = context.commit_column(&dict_datatype_at_d)?;
    let reloc_dict_datatype = prove_column_relocation_with_context(
        context, &dict.datatype_at_k, &dict.com_datatype_at_k, &dict_datatype_at_d,
        &com_dict_datatype_at_d, &mut t,
    )?;

    // Authenticate each index id -> (value, datatype) against the combined range dict.
    let lookup = prove_dict_value_lookup_with_context(
        context, &index_at_d, &com_index_at_d, &dict_id_at_d, &dict_value_at_d, &dict_datatype_at_d,
        &com_dict_id_at_d, &com_dict_value_at_d, &com_dict_datatype_at_d, n, n_dict, &mut t,
    )?;
    let (v_val, v_dt, _w) =
        honest_dict_lookup(&index_at_d, &dict_id_at_d, &dict_value_at_d, &dict_datatype_at_d, n, n_dict);

    // Datatype completeness: EVERY index is the required order-preserving unsigned type (`sel == χ_n`).
    let expected_datatype = ArkFr::from_u64(INDEX_DATATYPE as u64);
    let (datatype_filter, datatype_selection_commitment, datatype_inverse_commitment, _) =
        prove_filter_indicator_with_context(context, &v_dt, expected_datatype, n, &lookup.com_v_dt, &mut t)?;
    let mut chi = vec![ArkFr::zero(); common_domain];
    chi[..n].fill(ArkFr::one());
    if datatype_selection_commitment != context.commit_column(&chi)? {
        return Err(invalid(
            "reconciled ordered-array-walk has a sys:index of the wrong datatype (not a decodable position)",
        ));
    }

    // Decode each authenticated value to its integer position and reject any out-of-range index.
    let mut positions = Vec::with_capacity(n);
    for value in v_val.iter().take(n) {
        let p = generated_decimal_u64_value(*value)?;
        let p = usize::try_from(p).map_err(|_| invalid("reconciled ordered-array-walk index does not fit usize"))?;
        if p >= n {
            return Err(invalid(
                "reconciled ordered-array-walk sys:index is >= the cell count (sparse or oversized array)",
            ));
        }
        positions.push(p);
    }

    // The honest row order: output position `j` is filled by the cell whose index decodes to `j`.
    let mut honest_row_order = vec![usize::MAX; n];
    for (i, &p) in positions.iter().enumerate() {
        if honest_row_order[p] != usize::MAX {
            return Err(invalid(
                "reconciled ordered-array-walk has a duplicate sys:index (not a dense 0..n-1 permutation)",
            ));
        }
        honest_row_order[p] = i;
    }

    let forging = row_order_override.is_some();
    let row_order = row_order_override.unwrap_or(honest_row_order);

    // Route [position, value] -> [output-position, Out]. When honest, output-position == identity.
    let input_witness = RelationWitness {
        columns: vec![v_val.clone(), value_at_d.clone()],
        commitments: vec![lookup.com_v_val, com_value_at_d],
        log_domain: common_log_domain,
        n,
    };
    let perm = prove_relation_permutation_with_context(context, &input_witness, &row_order, &mut t)?;
    if perm.commitments.len() != 2 || perm.columns.len() != 2 {
        return Err(invalid("reconciled ordered-array-walk density permutation has the wrong arity"));
    }

    let com_identity = identity_commitment(n, common_log_domain)?;
    if !forging && perm.commitments[0] != com_identity {
        return Err(invalid(
            "reconciled ordered-array-walk routed index column is not the dense identity 0..n-1",
        ));
    }

    // C3b: the real per-cell `sys:index` id column (the SAME prefix `com_index_at_d` commits), carried
    // into the proof as the value re-pin witness so a term-less density can tie each authenticated
    // index id to its Merkle-opened order-preserving value/datatype.
    let index_oid: Vec<u64> = index_at_d[..n]
        .iter()
        .map(|f| field_id(*f))
        .collect::<io::Result<Vec<_>>>()?;

    Ok(DensityReconciledProve {
        range_dict: dict.proof,
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
        order_perm: perm.proof,
        com_out: perm.commitments[1],
        out_col: perm.columns[1].clone(),
        index_oid,
    })
}

/// Verify a reconciled ordered `sys:Array` walk against the trusted head root, returning the ordered
/// value column resolved over the chain catalogs. Mirrors [`prove_ordered_array_walk_reconciled`] step
/// for step: re-verify the single reconciliation + the four `present` leaves on the SHARED transcript,
/// replay every functional bind + the `sys:Array` equality, then the density certificate whose index
/// decode authenticates against the combined range dict. Any tampered leaf / forged index / broken
/// order fails CLOSED.
pub fn verify_ordered_array_walk_reconciled(
    layers: &[&LayerCommitment],
    trusted_head_root: ProofHash,
    subject: &CanonicalObject,
    predicate: &[u8],
    proof: &OrderedArrayWalkReconciledProof,
) -> io::Result<Vec<CanonicalObject>> {
    let views: Vec<crate::proof::VerifierCommitmentRef<'_>> =
        layers.iter().map(|l| l.verifier_view()).collect();
    verify_ordered_array_walk_reconciled_from_views(&views, trusted_head_root, subject, predicate, proof)
}

/// Verify a reconciled ordered `sys:Array` walk FROM verifier VIEWS (C3b) — the term-less-capable
/// core. Views-based analogue of [`verify_ordered_array_walk_reconciled`] (split exactly like
/// [`super::list_walk::verify_ordered_list_walk_reconciled_from_views`]): a TERM-LESS head view
/// (`LayerCommitment::verifier_commitment_termless`, term-bearing elsewhere) can be passed for the
/// term-less remote verifier. The walk STRUCTURE is id-space (the cell/index/value/type leaves
/// deliver combined-dict id columns; every functional bind / equality / density permutation consumes
/// committed id columns), so it is term-less-safe unchanged; the term-sensitive reads go term-less:
/// [`verify_reconciled_leaf_from_views`] for the four `present` leaves (plain conjunctive atoms, no
/// range ⇒ C1/E3-sound), the `sys:Array` class resolution via [`resolve_combined_node_id_termless`]
/// (belt-hashed against `proof.sys_array_opening`, fail-closed on a missing/forged opening — the id
/// keys the `rdf:type == sys:Array` equality), the delivered-value render via
/// [`resolve_node_value_id_column`] threading `proof.value_openings`, AND — THE C3b CRUX — the density
/// value re-pin (`verify_dict_value_terms_chain`, inside [`verify_density_order_reconciled`]) that
/// ties each authenticated `sys:index` id to its Merkle-opened order-preserving value/datatype (the
/// committed range-dict value column is UNAUTHENTICATED on a term-less view, so without the re-pin a
/// forged value would certify a verifying array REORDER).
pub fn verify_ordered_array_walk_reconciled_from_views(
    views: &[crate::proof::VerifierCommitmentRef<'_>],
    trusted_head_root: ProofHash,
    subject: &CanonicalObject,
    predicate: &[u8],
    proof: &OrderedArrayWalkReconciledProof,
) -> io::Result<Vec<CanonicalObject>> {
    // β re-anchor: the transcript/gadget-binding `root` is the head's COMMITMENT root (matching
    // `prove_ordered_array_walk_reconciled`); `trusted_head_root` is now a `resolution_root` used
    // ONLY for the head-pin equality inside `verify_reconcile_core` below.
    let head = *views
        .last()
        .ok_or_else(|| invalid("reconciled ordered-array-walk needs a head layer"))?;
    let root = head.state.commitment_root;
    let digest = reconciled_array_walk_digest(subject, predicate)?;

    // Reconcile + the four leaves, replayed on the SHARED transcript in the SAME order as prove.
    let mut transcript = Transcript254::new(RECONCILE_TRANSCRIPT);
    let verified = verify_reconcile_core(views, trusted_head_root, &proof.reconcile, &mut transcript)?;
    let head_nv = head.state.node_value_count as usize;

    // (A) CELL SET leaf.
    let cells_atom = bound_subject_leaf_atom(subject, predicate);
    let c_commit = verify_reconciled_leaf_from_views(
        views, &proof.reconcile, &verified, head_nv, cells_atom, &proof.cells_leaf, &mut transcript,
    )?;
    if c_commit.commitments.len() != 1 {
        return Err(invalid("reconciled ordered-array-walk cell set must be a single node column"));
    }
    let n = c_commit.n;
    if n == 0 || n != proof.n {
        return Err(invalid("reconciled ordered-array-walk cell count mismatch"));
    }
    let cell_log_domain = c_commit.log_domain;
    if cell_log_domain != proof.cell_log_domain {
        return Err(invalid("reconciled ordered-array-walk cell domain mismatch"));
    }

    // (B/C/type) EDGE leaves (same shared-transcript order as prove: index, value, type).
    let index_atom = predicate_edge_leaf_atom(SYS_INDEX);
    let index_commit = verify_reconciled_leaf_from_views(
        views, &proof.reconcile, &verified, head_nv, index_atom, &proof.index_edge_leaf, &mut transcript,
    )?;
    let value_atom = predicate_edge_leaf_atom(SYS_VALUE);
    let value_commit = verify_reconciled_leaf_from_views(
        views, &proof.reconcile, &verified, head_nv, value_atom, &proof.value_edge_leaf, &mut transcript,
    )?;
    let type_atom = predicate_edge_leaf_atom(RDF_TYPE);
    let type_commit = verify_reconciled_leaf_from_views(
        views, &proof.reconcile, &verified, head_nv, type_atom, &proof.type_edge_leaf, &mut transcript,
    )?;

    // (D/E/F) functional binds — byte-identical to the materialized path.
    verify_edge_bind(
        &root, &digest, &c_commit, &index_commit, n, cell_log_domain, &proof.index_join,
        &proof.index_join_out, &proof.index_reloc_subj, &proof.index_reloc_obj, proof.com_index, b"index",
    )?;
    verify_edge_bind(
        &root, &digest, &c_commit, &value_commit, n, cell_log_domain, &proof.value_join,
        &proof.value_join_out, &proof.value_reloc_subj, &proof.value_reloc_obj, proof.com_value, b"value",
    )?;
    verify_edge_bind(
        &root, &digest, &c_commit, &type_commit, n, cell_log_domain, &proof.type_join,
        &proof.type_join_out, &proof.type_reloc_subj, &proof.type_reloc_obj, proof.com_type, b"type",
    )?;

    // Type equality: rdf:type == sys:Array for EVERY cell. C3b: term-less-safe class resolution —
    // belt-hashed against the prover-shipped opening on a term-less chain, the audited catalog scan on
    // a fully term-bearing one. The resolved id keys the equality constraint, so a wrong sys:Array id
    // (an unguarded scan on a term-less head returns the impossible-id sentinel) would let a non-array
    // cell pass the type check.
    let sys_array_id =
        resolve_combined_node_id_termless(views, SYS_ARRAY, proof.sys_array_opening.as_ref())?;
    let type_commit_c = RelationCommit {
        commitments: vec![proof.com_type],
        log_domain: cell_log_domain,
        n,
    };
    let array_constraint = RelationEqualityConstraint::ColumnConstant {
        column: 0,
        value: ArkFr::from_u64(sys_array_id),
    };
    let mut t_type = aw_transcript(&root, &digest, b"type-membership");
    verify_relation_equality(
        &type_commit_c, &array_constraint, &proof.type_equality_out, n, cell_log_domain,
        &proof.type_equality, &mut t_type,
    )?;

    // (G) DENSITY + ORDER certificate over the combined range dict (incl. the C3b term-less value
    // re-pin tying each authenticated sys:index id to its Merkle-opened order-preserving value).
    verify_density_order_reconciled(views, verified.working_log, &root, &digest, proof, n, cell_log_domain)?;

    // (9) Resolve the delivered ordered value column over the chain (binds value_ids to com_out).
    // C3b: a TERM-LESS head opens each value id from the prover-shipped `value_openings` (fail-closed
    // on a missing/forged one); a term-bearing chain passes `None` and rebuilds from catalogs.
    if proof.value_ids.len() != n {
        return Err(invalid("reconciled ordered-array-walk value id count mismatch"));
    }
    resolve_node_value_id_column(
        views,
        &proof.value_ids,
        proof.com_out,
        proof.common_log_domain,
        proof.value_openings.as_ref(),
    )
}

/// C4-slice-3 byte-in remote-verifier entry: verify a reconciled ordered `sys:Array` walk proof
/// shipped as BYTES against a trusted head root, holding ONLY the encoded proof + the encoded
/// term-less [`VerifierCommitment`](crate::proof::commitment::VerifierCommitment) chain (no
/// `LayerCommitment` sidecars, no DB). Decodes each layer commitment and the walk proof, then
/// re-verifies through the audited [`verify_ordered_array_walk_reconciled_from_views`] —
/// serialization does NOT change verification. `subject`/`predicate`/`trusted_head_root` are the
/// VERIFIER's OWN query parameters (anti-substitution): they are NEVER parsed from the proof bytes.
pub fn decode_and_verify_ordered_array_walk_reconciled(
    array_walk_proof_bytes: &[u8],
    verifier_commitment_bytes: &[&[u8]],
    trusted_head_root: ProofHash,
    subject: &CanonicalObject,
    predicate: &[u8],
) -> io::Result<Vec<CanonicalObject>> {
    // Own the decoded commitments FIRST, so the borrowed views below cannot dangle.
    let owned: Vec<crate::proof::commitment::VerifierCommitment> = verifier_commitment_bytes
        .iter()
        .map(|b| crate::proof::commitment::VerifierCommitment::decode(b))
        .collect::<io::Result<_>>()?;
    let views: Vec<crate::proof::VerifierCommitmentRef<'_>> =
        owned.iter().map(|c| c.verifier_view()).collect();
    let proof = OrderedArrayWalkReconciledProof::decode(array_walk_proof_bytes)?;
    verify_ordered_array_walk_reconciled_from_views(
        &views,
        trusted_head_root,
        subject,
        predicate,
        &proof,
    )
}

/// Verify the reconciled density+order certificate. Re-verifies the combined range dict against the
/// root-bound per-layer branches, each relocation, the authenticated dict lookup, the datatype filter,
/// and the permutation whose OUTPUT index column the verifier reconstructs as the public identity
/// `[0..n-1]` (the density certificate). Mirrors [`prove_density_order_reconciled`] step for step.
#[allow(clippy::too_many_arguments)]
fn verify_density_order_reconciled(
    views: &[crate::proof::VerifierCommitmentRef<'_>],
    working_log: usize,
    root: &ProofHash,
    digest: &ProofHash,
    proof: &OrderedArrayWalkReconciledProof,
    n: usize,
    cell_log_domain: usize,
) -> io::Result<()> {
    let mut t = aw_transcript(root, digest, b"density");

    // (0) Re-verify the combined range dict against the root-bound per-layer branch commitments
    // (the prover cannot substitute foreign branches).
    let branches = range_dict_branch_commits(views)?;
    let n_dict: usize = branches.iter().map(|branch| branch.n).sum();
    verify_combined_range_dict(&branches, n_dict, working_log, &proof.range_dict, &mut t)?;
    let dict_log_domain = working_log;
    let common_log_domain = cell_log_domain.max(dict_log_domain);
    if common_log_domain != proof.common_log_domain {
        return Err(invalid("reconciled ordered-array-walk working domain mismatch"));
    }

    t.append_bytes(b"pow/array-walk/rows", &(n as u64).to_le_bytes());
    t.append_bytes(b"pow/array-walk/common-log-domain", &(common_log_domain as u64).to_le_bytes());

    // Relocations lifting the cell-aligned index/value columns and the combined dict columns UP.
    verify_column_relocation(
        &proof.com_index, cell_log_domain, &proof.com_index_at_d, common_log_domain,
        &proof.reloc_index_at_d, &mut t,
    )?;
    verify_column_relocation(
        &proof.com_value, cell_log_domain, &proof.com_value_at_d, common_log_domain,
        &proof.reloc_value_at_d, &mut t,
    )?;
    verify_column_relocation(
        &proof.range_dict.com_id_at_k, dict_log_domain, &proof.com_dict_id_at_d, common_log_domain,
        &proof.reloc_dict_id, &mut t,
    )?;
    verify_column_relocation(
        &proof.range_dict.com_value_at_k, dict_log_domain, &proof.com_dict_value_at_d, common_log_domain,
        &proof.reloc_dict_value, &mut t,
    )?;
    verify_column_relocation(
        &proof.range_dict.com_datatype_at_k, dict_log_domain, &proof.com_dict_datatype_at_d,
        common_log_domain, &proof.reloc_dict_datatype, &mut t,
    )?;

    // Authenticate each index id -> (value, datatype) against the combined dict.
    verify_dict_value_lookup(
        &proof.com_index_at_d, &proof.com_dict_id_at_d, &proof.com_dict_value_at_d,
        &proof.com_dict_datatype_at_d, n, n_dict, &proof.lookup, &mut t,
    )?;

    // C3b THE CRUX — value re-pin over a TERM-LESS view. The committed range-dict value/datatype
    // columns the lookup just decoded against are NOT tied to the real terms on a term-less view (the
    // per-catalog re-MSM `validate()` restores is dropped). WITHOUT the tie a malicious prover forges
    // the range-dict value column so a real `sys:index` id decodes to a DIFFERENT position — a
    // verifying array REORDER. Restore it exactly like the C2a present-range arm: recommit the per-cell
    // `sys:index` id column to `com_index_at_d` (the SAME column the lookup queried against) and
    // re-derive each id's order-preserving value/datatype from its CHAIN-opened Merkle term, requiring
    // `Com(ov) == lookup.com_v_val` / `Com(dt) == lookup.com_v_dt` — the EXACT lookup columns the
    // density decision (position decode + permutation input) reads. No NaN gate here: `sys:index` is an
    // unsigned integer (`INDEX_DATATYPE`), never a Float64. No openings ⇒ fail CLOSED. Term-bearing:
    // unchanged (each `validate()`'s re-MSM already ties them).
    if !views.iter().all(|view| view.terms_present) {
        let openings = proof.index_value_openings.as_ref().ok_or_else(|| {
            invalid("term-less array density requires prover-shipped index value openings")
        })?;
        verify_dict_value_terms_chain(
            views,
            &proof.com_index_at_d,
            &proof.lookup.com_v_val,
            &proof.lookup.com_v_dt,
            n,
            1usize << common_log_domain,
            &proof.index_oid,
            openings,
        )?;
    }

    // Datatype completeness.
    let expected_datatype = ArkFr::from_u64(INDEX_DATATYPE as u64);
    verify_filter_indicator(
        &proof.lookup.com_v_dt, expected_datatype, n, &proof.datatype_selection_commitment,
        &proof.datatype_inverse_commitment, &proof.datatype_filter, &mut t,
    )?;
    let mut chi = vec![ArkFr::zero(); 1usize << common_log_domain];
    chi[..n].fill(ArkFr::one());
    if proof.datatype_selection_commitment != commit_column(&chi)? {
        return Err(invalid(
            "reconciled ordered-array-walk has a sys:index of the wrong datatype (not a decodable position)",
        ));
    }

    // The density permutation: input [position(=v_val), value], output [identity, Out].
    let input_commit = RelationCommit {
        commitments: vec![proof.lookup.com_v_val, proof.com_value_at_d],
        log_domain: common_log_domain,
        n,
    };
    let com_identity = identity_commitment(n, common_log_domain)?;
    verify_relation_permutation(
        &input_commit,
        &[com_identity, proof.com_out],
        n,
        common_log_domain,
        &proof.order_perm,
        &mut t,
    )
}

/// C3b (producer): attach the UNCOMMITTED term-less opening witnesses to a proven
/// [`OrderedArrayWalkReconciledProof`], so it can ship alongside a TERM-LESS reconciled artifact.
/// Mints THREE witness sets:
///   * `value_openings` — the delivered-value [`ResolutionOpenings`] (for the ordered value render,
///     mirroring [`super::list_walk::attach_termless_list_walk_openings`]'s element openings);
///   * `sys_array_opening` — the `sys:Array` class [`DictOpening`] (routed to its introducing layer
///     via [`build_resolution_openings`] for the single resolved id; belt-hashed at verify, its id
///     keys the `rdf:type == sys:Array` equality);
///   * `index_oid` + `index_value_openings` — the C2a value re-pin witness (the per-cell `sys:index`
///     id column + Merkle openings over its DISTINCT ids), built by
///     [`build_termless_present_range_openings`] exactly as the present-range arm does. This is what
///     ties each authenticated index id to its real order-preserving value/datatype so a term-less
///     head cannot forge the range-dict value column into an array REORDER.
/// `value_ids` are the walk's delivered value column ([`OrderedArrayWalkReconciledProof::value_ids`]);
/// `sys_array_id` is the resolved combined-dict id of `sys:Array` (`resolve_combined_node_id(&views,
/// SYS_ARRAY)` on the term-bearing chain); `index_ids` is the per-cell `sys:index` id column
/// ([`OrderedArrayWalkReconciledProof::index_oid`], populated at prove). The prover still holds the
/// term-bearing catalogs, so it rebuilds each id-indexed tree; the witnesses are re-verified per id at
/// the verifier by `verify_opening` against the root-bound `nv_merkle_root` (forged ⇒ rejected,
/// missing ⇒ fail-closed) and are NEVER folded into `commitment_root`/`resolution_root`. Over-inclusion
/// is safe; every opened constant/literal must be present in the dict (a positive opening cannot be
/// minted for an absent term).
pub fn attach_termless_array_walk_openings(
    proof: &mut OrderedArrayWalkReconciledProof,
    layers: &[&LayerCommitment],
    value_ids: &[u64],
    sys_array_id: u64,
    index_ids: &[u64],
) -> io::Result<()> {
    let views: Vec<crate::proof::VerifierCommitmentRef<'_>> =
        layers.iter().map(|l| l.verifier_view()).collect();
    // Delivered ordered values: rebuild each id-indexed tree, then open it.
    proof.value_openings = Some(build_resolution_openings(&views, value_ids, &[])?);
    // The single sys:Array class opening (routed to its introducing layer, opened positionally).
    let sys_array_openings = build_resolution_openings(&views, &[sys_array_id], &[])?;
    proof.sys_array_opening = Some(sys_array_openings.node_value_opening(sys_array_id)?.clone());
    // The C2a index value re-pin witness: the per-cell index id column + openings over its distinct
    // non-zero ids (same builder the present-range arm uses for its object column).
    let index_column: Vec<ArkFr> = index_ids.iter().map(|id| ArkFr::from_u64(*id)).collect();
    let (oid, index_openings) = build_termless_present_range_openings(layers, &index_column)?;
    proof.index_oid = oid;
    proof.index_value_openings = Some(index_openings);
    Ok(())
}
