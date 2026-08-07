//! End-to-end single-pattern query BY PREDICATE IRI against a REAL committed
//! store layer, anchored to the layer's `commitment_root`.
//!
//! A query names a predicate IRI (term bytes), not a raw dictionary id. The proof
//! combines two arguments under ONE shared transcript:
//!   1. Q3 dictionary membership ([`super::dict`]): the revealed `predicate_id` is
//!      the genuine dictionary id for `hash_term(predicate_term)`, authenticated by
//!      the layer's committed predicate dictionary (`predicate_dict_commitments`).
//!   2. Q1 filter compaction ([`super::filter`]): `(S_out, O_out)[0..m]` is exactly
//!      the compaction of the rows whose predicate id equals that id.
//!
//! Both the column commitments and the predicate-dictionary commitments are bound
//! by `commitment_root` (1b + 3b-1). The verifier gets only the public IRI bytes
//! plus a revealed id (checked via membership — never trusted out of band), and
//! concludes `(S_out, O_out)` is exactly the complete result of `t(?s, <IRI>, ?o)`
//! without rerunning the query. Node/value (subject/object) dictionary binding is
//! deferred; absent-predicate (non-membership) proofs are a later feature — an IRI
//! not in the layer fails closed.

use std::io;

use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::transcript::Transcript;

use super::argument::Transcript254;
use super::bgp::{
    prove_bgp_chain, verify_bgp_chain, BgpChainProof, JoinStep, RelationCommit, RelationWitness,
};
use super::commitment::LayerCommitment;
use super::ProofHash;
use super::dict::{
    hash_object, hash_term, prove_dict_membership, verify_dict_membership, DictProof,
};
use super::filter::{
    atom_kept_columns, compact_selected, honest_witness, prove_atom_filter, prove_filter,
    verify_atom_filter, verify_filter, AtomFilterProof, AtomFilterSpec, FullFilterProof, Position,
};
use super::invalid;
use super::{CanonicalDatatype, CanonicalObject};
use super::join::{
    compute_self_join, prove_self_join_complete, self_join_commitments, self_join_domain,
    verify_self_join_complete, SelfJoinCommitments, SelfJoinCompleteProof,
};
use super::relocate::{prove_column_relocation, verify_column_relocation, RelocationProof};

const QUERY_DOMAIN: &[u8] = b"parture/proof-of-woql/q1-query-by-iri/v0";

#[inline]
fn f0() -> ArkFr {
    ArkFr::zero()
}

/// A complete query-by-IRI proof: the predicate-dictionary membership proof
/// (IRI -> id) and the filter/compaction proof (id -> result), both produced
/// under one shared transcript.
#[derive(Clone, Debug)]
pub struct CombinedFilterProof {
    pub membership: DictProof,
    pub filter: FullFilterProof,
}

/// The padded hypercube size for a materialized base layer's columns, matching
/// `commit_id_column`'s `next_power_of_two(additions.len())`. Requires `>= 2` (a
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

/// Extract the `(S, P, O)` `ArkFr` columns and real-row count `n` from a
/// materialized base layer, in the exact stored (SPO-sorted) order and padding
/// that produced `column_commitments[0..3]`.
fn base_layer_columns(
    layer: &LayerCommitment,
) -> io::Result<(Vec<ArkFr>, Vec<ArkFr>, Vec<ArkFr>, usize)> {
    if !layer.removals.is_empty() {
        return Err(invalid(
            "layer filter requires a materialized base layer (no removals)",
        ));
    }
    let n = layer.additions.len();
    let padded = padded_len(n)?;
    let mut s = vec![f0(); padded];
    let mut p = vec![f0(); padded];
    let mut o = vec![f0(); padded];
    for (i, record) in layer.additions.iter().enumerate() {
        s[i] = ArkFr::from_u64(record.subject_id);
        p[i] = ArkFr::from_u64(record.predicate_id);
        o[i] = ArkFr::from_u64(record.object_id);
    }
    Ok((s, p, o, n))
}

/// Resolve a predicate IRI (term bytes) to its dictionary id in this layer, or
/// fail closed if the term is not present (non-membership proofs are deferred).
fn resolve_predicate_id(layer: &LayerCommitment, predicate_term: &[u8]) -> io::Result<u64> {
    layer
        .predicate_catalog
        .iter()
        .find(|record| record.triple.predicate == predicate_term)
        .map(|record| record.predicate_id)
        .ok_or_else(|| invalid("predicate not in layer dictionary"))
}

/// Bind the public statement into the shared transcript before either sub-proof:
/// the trust anchor (`commitment_root`, which binds the column + dictionary
/// commitments), the public predicate IRI bytes, and the revealed id.
fn bind_statement(
    transcript: &mut Transcript254,
    layer: &LayerCommitment,
    predicate_term: &[u8],
    predicate_id: u64,
) {
    transcript.append_bytes(b"pow/q1-iri/commitment-root", &layer.state.commitment_root);
    transcript.append_bytes(b"pow/q1-iri/predicate-term", predicate_term);
    transcript.append_bytes(b"pow/q1-iri/predicate-id", &predicate_id.to_le_bytes());
}

/// Prove `t(?s, <predicate_term>, ?o)` against a materialized base layer. Resolves
/// the predicate IRI to its committed dictionary id, proves that binding (Q3
/// membership), then proves the filtered result for that id (Q1). Returns the
/// combined proof, the output commitments, the result length `m`, and the REVEALED
/// resolved id (authenticated by the membership proof — not secret).
pub fn prove_layer_filter(
    layer: &LayerCommitment,
    predicate_term: &[u8],
) -> io::Result<(CombinedFilterProof, ArkGT, ArkGT, usize, u64)> {
    let (s, p, o, n) = base_layer_columns(layer)?;
    let predicate_id = resolve_predicate_id(layer, predicate_term)?;

    let (dict_hash, dict_id) = layer.predicate_dictionary_columns()?;
    let com_dict_hash = layer.predicate_dict_commitments.0;
    let com_dict_id = layer.predicate_dict_commitments.1;

    let mut transcript = Transcript254::new(QUERY_DOMAIN);
    bind_statement(&mut transcript, layer, predicate_term, predicate_id);

    // Q3 membership: (hash_term(IRI), id) is a genuine dictionary row.
    let q_hash = hash_term(predicate_term);
    let id = ArkFr::from_u64(predicate_id);
    let membership = prove_dict_membership(
        &dict_hash,
        &dict_id,
        &com_dict_hash,
        &com_dict_id,
        q_hash,
        id,
        &mut transcript,
    )?;

    // Q1 filter: the result for that id.
    let com_s = layer.column_commitments[0];
    let com_p = layer.column_commitments[1];
    let com_o = layer.column_commitments[2];
    let pred = ArkFr::from_u64(predicate_id);
    let (filter, com_sout, com_oout, m) =
        prove_filter(&s, &p, &o, pred, n, &com_s, &com_p, &com_o, &mut transcript)?;

    Ok((
        CombinedFilterProof { membership, filter },
        com_sout,
        com_oout,
        m,
        predicate_id,
    ))
}

/// Verify a query-by-IRI proof. Anchors to `commitment_root` (which binds both the
/// column and predicate-dictionary commitments), independently computes
/// `q_hash = hash_term(predicate_term)`, checks the revealed `predicate_id` is the
/// genuine dictionary id for that IRI (membership), then checks the filtered
/// result for that id (filter) — all under the shared transcript. The verifier
/// never trusts the id out of band and never rescans the table.
pub fn verify_layer_filter(
    layer: &LayerCommitment,
    expected_root: ProofHash,
    predicate_term: &[u8],
    predicate_id: u64,
    com_sout: &ArkGT,
    com_oout: &ArkGT,
    m: usize,
    proof: &CombinedFilterProof,
) -> io::Result<()> {
    // FIX 4: pin to an EXTERNALLY-trusted root FIRST (mirroring import_pack's
    // external-root check), so the proof is bound to a root the verifier already
    // trusts rather than whatever LayerCommitment the prover supplied.
    if layer.state.commitment_root != expected_root {
        return Err(invalid(
            "layer commitment_root does not match the externally-trusted expected root",
        ));
    }
    if layer.recomputed_commitment_root()? != layer.state.commitment_root {
        return Err(invalid(
            "layer commitment_root does not bind its committed columns/dictionary",
        ));
    }
    if layer.state.removal_count != 0 {
        return Err(invalid(
            "layer filter requires a materialized base layer (no removals)",
        ));
    }
    let n = layer.state.addition_count as usize;
    let padded = padded_len(n)?;
    let expected_k = padded.trailing_zeros() as usize;
    if proof.filter.sumcheck.round_polys.len() != expected_k {
        return Err(invalid(
            "proof dimension does not match the layer's committed columns",
        ));
    }

    let mut transcript = Transcript254::new(QUERY_DOMAIN);
    bind_statement(&mut transcript, layer, predicate_term, predicate_id);

    // Membership: the revealed id is authentic for this IRI.
    let q_hash = hash_term(predicate_term);
    let id = ArkFr::from_u64(predicate_id);
    verify_dict_membership(
        &layer.predicate_dict_commitments.0,
        &layer.predicate_dict_commitments.1,
        q_hash,
        id,
        &proof.membership,
        &mut transcript,
    )?;

    // Filter: the result for that id.
    let com_s = layer.column_commitments[0];
    let com_p = layer.column_commitments[1];
    let com_o = layer.column_commitments[2];
    let pred = ArkFr::from_u64(predicate_id);
    verify_filter(
        &com_s,
        &com_p,
        &com_o,
        pred,
        n,
        com_sout,
        com_oout,
        m,
        &proof.filter,
        &mut transcript,
    )
}

// ===========================================================================
// Q2: 2-atom graph-pattern query t(?x, <P>, ?y), t(?y, <P>, ?z) end-to-end.
// ===========================================================================

const GRAPH2_DOMAIN: &[u8] = b"parture/proof-of-woql/q2-graph-pattern2-by-iri/v0";

fn pad(v: &[ArkFr], n: usize) -> Vec<ArkFr> {
    let mut out = v.to_vec();
    out.resize(n, f0());
    out
}

/// A complete 2-atom graph-pattern proof, composed under one shared transcript and
/// anchored to `commitment_root`:
///   * `membership` — Q3 dictionary binding of the predicate IRI to its id.
///   * `filter` (+ `com_sout`/`com_oout`) — Q1 complete filter of the layer by that
///     id into `F = (Sf, Of)` = the predicate-P `(subject, object)` pairs, committed
///     at the TABLE domain and root-anchored via the filter argument.
///   * `relocate_s`/`relocate_o` — cross-domain equality tying the filter output
///     (table domain) to the self-join input `Sf`/`Of` (join domain), positionally.
///   * `join` (+ `join_commitments`) — the complete self-join of `F` on `Of == Sf`
///     into `(X, Y, Z)`.
/// `n_f`/`n_u` are the (sub-proof-pinned) filter-result count and distinct-key count,
/// carried so the verifier can re-derive the join domain.
#[derive(Clone, Debug)]
pub struct GraphPattern2Proof {
    pub membership: DictProof,
    pub filter: FullFilterProof,
    pub com_sout: ArkGT,
    pub com_oout: ArkGT,
    pub relocate_s: RelocationProof,
    pub relocate_o: RelocationProof,
    pub join: SelfJoinCompleteProof,
    pub join_commitments: SelfJoinCommitments,
    pub n_f: usize,
    pub n_u: usize,
}

/// Prove the COMPLETE result of the 2-atom graph pattern
/// `t(?x, <predicate_term>, ?y), t(?y, <predicate_term>, ?z)` against a materialized
/// base layer: every `(x, y, z)` such that `(x, y)` and `(y, z)` are both triples
/// with predicate `<predicate_term>`. Composes, under one shared transcript anchored
/// to `commitment_root`: Q3 dict binding (IRI -> id) + Q1 complete filter (id -> F) +
/// cross-domain relocation (F -> join input) + the complete self-join (F -> result).
/// Returns the combined proof, the result commitments `(Com X, Com Y, Com Z)`, the
/// result length `m_J`, and the REVEALED resolved id (authenticated by membership).
pub fn prove_layer_graph_pattern2(
    layer: &LayerCommitment,
    predicate_term: &[u8],
) -> io::Result<(GraphPattern2Proof, ArkGT, ArkGT, ArkGT, usize, u64)> {
    let (s, p, o, n) = base_layer_columns(layer)?;
    let predicate_id = resolve_predicate_id(layer, predicate_term)?;

    let (dict_hash, dict_id) = layer.predicate_dictionary_columns()?;
    let com_dict_hash = layer.predicate_dict_commitments.0;
    let com_dict_id = layer.predicate_dict_commitments.1;

    let mut transcript = Transcript254::new(GRAPH2_DOMAIN);
    bind_statement(&mut transcript, layer, predicate_term, predicate_id);

    // (1) Q3 membership: (hash_term(IRI), id) is a genuine dictionary row.
    let q_hash = hash_term(predicate_term);
    let id = ArkFr::from_u64(predicate_id);
    let membership = prove_dict_membership(
        &dict_hash,
        &dict_id,
        &com_dict_hash,
        &com_dict_id,
        q_hash,
        id,
        &mut transcript,
    )?;

    // (2) Q1 complete filter: F = (Sf, Of) = the predicate-P (subject, object) pairs.
    let com_s = layer.column_commitments[0];
    let com_p = layer.column_commitments[1];
    let com_o = layer.column_commitments[2];
    let pred = ArkFr::from_u64(predicate_id);
    let (filter, com_sout, com_oout, m) =
        prove_filter(&s, &p, &o, pred, n, &com_s, &com_p, &com_o, &mut transcript)?;
    // Recompute the dense filter output (same witness prove_filter used) to obtain the
    // F columns; `s_out`/`o_out` are length `2^a` (table domain), dense in `[0, m)`.
    let (sel, _w) = honest_witness(&p, pred, n);
    let (s_out, o_out, _m) = compact_selected(&s, &o, &sel);
    let sf = &s_out[..m];
    let of = &o_out[..m];

    // (3) Complete self-join of F on Of == Sf, committed at the join domain.
    let (x, y, z, rho_l, rho_r, u, w_l, w_r, m_j) = compute_self_join(sf, of);
    let n_u = u.len();
    let n_join = self_join_domain(m, m_j, n_u)?;
    let jcom =
        self_join_commitments(sf, of, &x, &y, &z, &rho_l, &rho_r, &u, &w_l, &w_r)?;

    // (4) Relocate F from the table domain to the join domain (positional equality).
    let target_sf = pad(sf, n_join);
    let target_of = pad(of, n_join);
    let relocate_s =
        prove_column_relocation(&s_out, &com_sout, &target_sf, &jcom.sf, &mut transcript)?;
    let relocate_o =
        prove_column_relocation(&o_out, &com_oout, &target_of, &jcom.of, &mut transcript)?;

    // (5) The complete self-join argument over the relocated F.
    let join = prove_self_join_complete(
        sf, of, &x, &y, &z, &rho_l, &rho_r, &u, &w_l, &w_r, m_j, &jcom, &mut transcript,
    )?;

    let (com_x, com_y, com_z) = (jcom.x, jcom.y, jcom.z);
    let proof = GraphPattern2Proof {
        membership,
        filter,
        com_sout,
        com_oout,
        relocate_s,
        relocate_o,
        join,
        join_commitments: jcom,
        n_f: m,
        n_u,
    };
    Ok((proof, com_x, com_y, com_z, m_j, predicate_id))
}

/// Verify a 2-atom graph-pattern proof. Anchors to `commitment_root`; re-derives the
/// join domain from the proof's pinned `n_F`/`n_U` and the public `m_J`; checks the
/// caller's result commitments are the proof's join output; then re-runs, under the
/// shared transcript in prover order: dict membership (IRI -> revealed id), the
/// complete filter (id -> F), the two cross-domain relocations (F -> join input), and
/// the complete self-join (F -> result). Concludes `(Com X, Com Y, Com Z)` is EXACTLY
/// the complete result of `t(?x, <IRI>, ?y), t(?y, <IRI>, ?z)`, without rerunning it.
#[allow(clippy::too_many_arguments)]
pub fn verify_layer_graph_pattern2(
    layer: &LayerCommitment,
    expected_root: ProofHash,
    predicate_term: &[u8],
    predicate_id: u64,
    com_x: &ArkGT,
    com_y: &ArkGT,
    com_z: &ArkGT,
    m_j: usize,
    proof: &GraphPattern2Proof,
) -> io::Result<()> {
    // FIX 4: pin to an EXTERNALLY-trusted root FIRST.
    if layer.state.commitment_root != expected_root {
        return Err(invalid(
            "layer commitment_root does not match the externally-trusted expected root",
        ));
    }
    if layer.recomputed_commitment_root()? != layer.state.commitment_root {
        return Err(invalid(
            "layer commitment_root does not bind its committed columns/dictionary",
        ));
    }
    if layer.state.removal_count != 0 {
        return Err(invalid(
            "layer graph-pattern query requires a materialized base layer (no removals)",
        ));
    }
    let n = layer.state.addition_count as usize;
    let padded = padded_len(n)?;
    let a = padded.trailing_zeros() as usize;
    if proof.filter.sumcheck.round_polys.len() != a {
        return Err(invalid(
            "proof dimension does not match the layer's committed columns",
        ));
    }

    // The caller's result commitments must be the proof's committed join output.
    if proof.join_commitments.x != *com_x
        || proof.join_commitments.y != *com_y
        || proof.join_commitments.z != *com_z
    {
        return Err(invalid(
            "result commitments do not match the proof's join output",
        ));
    }

    // Re-derive the join domain from the pinned filter-result / distinct-key counts.
    let n_f = proof.n_f;
    let n_u = proof.n_u;
    if n_f > padded {
        return Err(invalid("filter result length exceeds the table domain"));
    }
    let n_join = self_join_domain(n_f, m_j, n_u)?;
    let b = n_join.trailing_zeros() as usize;

    let mut transcript = Transcript254::new(GRAPH2_DOMAIN);
    bind_statement(&mut transcript, layer, predicate_term, predicate_id);

    // (1) Membership: the revealed id is authentic for this IRI.
    let q_hash = hash_term(predicate_term);
    let id = ArkFr::from_u64(predicate_id);
    verify_dict_membership(
        &layer.predicate_dict_commitments.0,
        &layer.predicate_dict_commitments.1,
        q_hash,
        id,
        &proof.membership,
        &mut transcript,
    )?;

    // (2) Filter: F = (S_out, O_out) is the complete predicate-P result (length n_F).
    let com_s = layer.column_commitments[0];
    let com_p = layer.column_commitments[1];
    let com_o = layer.column_commitments[2];
    let pred = ArkFr::from_u64(predicate_id);
    verify_filter(
        &com_s,
        &com_p,
        &com_o,
        pred,
        n,
        &proof.com_sout,
        &proof.com_oout,
        n_f,
        &proof.filter,
        &mut transcript,
    )?;

    // (3) Relocate: the filter output equals the join input Sf/Of, positionally.
    verify_column_relocation(
        &proof.com_sout,
        a,
        &proof.join_commitments.sf,
        b,
        &proof.relocate_s,
        &mut transcript,
    )?;
    verify_column_relocation(
        &proof.com_oout,
        a,
        &proof.join_commitments.of,
        b,
        &proof.relocate_o,
        &mut transcript,
    )?;

    // (4) Complete self-join of F -> (X, Y, Z).
    verify_self_join_complete(
        &proof.join_commitments,
        n_f,
        n_u,
        m_j,
        &proof.join,
        &mut transcript,
    )
}

// ===========================================================================
// Q_k: general k-atom conjunctive BGP over ONE committed layer, left-deep.
// ===========================================================================

const LAYER_BGP_DOMAIN: &[u8] = b"parture/proof-of-woql/qk-layer-bgp-by-iri/v0";

/// One term (subject or object) of a layer BGP atom: a query VARIABLE (kept as an
/// output column) or a CONSTANT authenticated against the layer's node/value dictionary.
/// `NodeConstant` is a node IRI (valid in the subject or object position); `ValueConstant`
/// is a typed literal (valid in the object position only). A node IRI resolves through the
/// SAME node/value id whether it appears as a subject or an object constant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AtomTerm {
    Variable,
    NodeConstant(Vec<u8>),
    ValueConstant(CanonicalDatatype, Vec<u8>),
}

/// One atom of a layer BGP. v1 SCOPE: the PREDICATE is always a constant IRI
/// (authenticated by the layer's predicate dictionary). The subject/object may each be a
/// VARIABLE (a kept output column) or a CONSTANT (authenticated against the layer's
/// node/value dictionary and folded into the atom filter). Constraints (enforced at
/// prove AND verify): the subject may not be a `ValueConstant` (literals never occupy the
/// subject position), and at least one of subject/object must be a `Variable` (a
/// fully-ground atom is an existence check, out of scope for v1).
#[derive(Clone, Debug)]
pub struct LayerBgpAtom {
    pub subject: AtomTerm,
    pub predicate_term: Vec<u8>,
    pub object: AtomTerm,
}

impl LayerBgpAtom {
    /// The classic all-variable atom `t(?s, <predicate_term>, ?o)`: both subject and
    /// object are query variables (the two kept output columns).
    pub fn all_variables(predicate_term: Vec<u8>) -> Self {
        Self {
            subject: AtomTerm::Variable,
            predicate_term,
            object: AtomTerm::Variable,
        }
    }
}

/// A layer BGP plan: `k` atoms and the `k - 1` left-deep join steps. `joins[j]` names
/// the single shared variable's column on the evolving accumulator (`acc_key`) and on
/// atom `j + 1`'s relation (`rel_key`). Each atom relation has ONE column per VARIABLE
/// (in S,O order): two for an all-variable atom, one for a subject/object-constant atom.
/// The accumulator schema evolves as `[r_other_out.., key, s_other_out..]` per step.
#[derive(Clone, Debug)]
pub struct LayerBgpPlan {
    pub atoms: Vec<LayerBgpAtom>,
    pub joins: Vec<JoinStep>,
}

/// The per-atom half of a layer BGP proof: the dictionary binding of the predicate IRI to
/// its id, the optional subject/object CONSTANT bindings (node/value dictionary), the
/// complete per-atom filter into the dense KEPT (variable) columns, one commitment per
/// kept column (in S,O order), the filter output length, and the REVEALED subject/object
/// constant ids. Those ids are public inputs to verification: the verifier feeds the SAME
/// id to both the membership check (constant -> id) and the filter's constant, so a prover
/// cannot authenticate one id and filter on another.
#[derive(Clone, Debug)]
pub struct LayerBgpAtomProof {
    pub predicate_membership: DictProof,
    pub subject_membership: Option<DictProof>,
    pub object_membership: Option<DictProof>,
    pub filter: AtomFilterProof,
    pub kept_commitments: Vec<ArkGT>,
    pub m: usize,
    pub subject_id: Option<u64>,
    pub object_id: Option<u64>,
}

/// A complete k-atom layer BGP proof: one per-atom (membership + filter) proof, plus the
/// left-deep binary-join chain over the filtered relations, all under one transcript.
#[derive(Clone, Debug)]
pub struct LayerBgpProof {
    pub atoms: Vec<LayerBgpAtomProof>,
    pub chain: BgpChainProof,
}

/// The resolved dictionary ids for one atom's constants: the predicate is always a
/// constant in v1 (so `predicate_id` is unconditional); `subject_id`/`object_id` are
/// `Some` iff that position is a constant.
#[derive(Clone, Copy, Debug)]
struct AtomIds {
    predicate_id: u64,
    subject_id: Option<u64>,
    object_id: Option<u64>,
}

/// The canonical dictionary object a constant term resolves through, or `None` for a
/// variable. A node IRI (subject or object) resolves through `Node(iri)`; a typed literal
/// through `Value{..}` — both share the node/value id space via `hash_object`.
fn atom_term_object(term: &AtomTerm) -> Option<CanonicalObject> {
    match term {
        AtomTerm::Variable => None,
        AtomTerm::NodeConstant(iri) => Some(CanonicalObject::Node(iri.clone())),
        AtomTerm::ValueConstant(datatype, lexical) => Some(CanonicalObject::Value {
            datatype: *datatype,
            lexical: lexical.clone(),
        }),
    }
}

/// Resolve a subject/object CONSTANT to its node/value dictionary id in this layer, or
/// fail closed if the constant is absent (non-membership proofs are deferred, mirroring
/// [`resolve_predicate_id`]). Subjects share the node/value id space with node objects, so
/// the same `(hash_object(term), id)` row authenticates the constant in either position.
fn resolve_node_value_id(layer: &LayerCommitment, object: &CanonicalObject) -> io::Result<u64> {
    layer
        .node_value_catalog
        .iter()
        .find(|record| &record.triple.object == object)
        .map(|record| record.object_id)
        .ok_or_else(|| invalid("constant not in layer node/value dictionary"))
}

/// Enforce the v1 atom constraints: the subject is a variable or a node constant (never a
/// value literal), and at least one of subject/object is a variable (fully-ground atoms
/// are a later existence check).
fn validate_atom(atom: &LayerBgpAtom) -> io::Result<()> {
    if matches!(atom.subject, AtomTerm::ValueConstant(..)) {
        return Err(invalid(
            "layer BGP subject may not be a value-literal constant",
        ));
    }
    if !matches!(atom.subject, AtomTerm::Variable) && !matches!(atom.object, AtomTerm::Variable) {
        return Err(invalid(
            "layer BGP atom must have at least one variable (ground atoms unsupported in v1)",
        ));
    }
    Ok(())
}

/// The `(constant/id)` presence carried in a proof MUST match the atom's constant
/// structure: a variable position has no id, a constant position has one. Rejects a
/// malformed proof that, e.g., claims a subject id for a variable subject (which would
/// otherwise silently reshape the filter spec).
fn check_ids_consistent(atom: &LayerBgpAtom, ids: &AtomIds) -> io::Result<()> {
    let subject_is_constant = !matches!(atom.subject, AtomTerm::Variable);
    let object_is_constant = !matches!(atom.object, AtomTerm::Variable);
    if subject_is_constant != ids.subject_id.is_some()
        || object_is_constant != ids.object_id.is_some()
    {
        return Err(invalid(
            "layer BGP atom constant/id presence does not match the plan",
        ));
    }
    Ok(())
}

/// The atom's [`AtomFilterSpec`] from its resolved ids: `P` is always a constant; `S`/`O`
/// are constants iff their id is `Some`; the kept (variable) positions are the constants'
/// complement in S,P,O order (a subset of `{S, O}` since `P` is always constant). The
/// constant list order (P, then S, then O) is fixed and identical on both sides.
fn atom_filter_spec(atom: &LayerBgpAtom, ids: &AtomIds) -> AtomFilterSpec {
    let mut constants = vec![(Position::P, ArkFr::from_u64(ids.predicate_id))];
    if let Some(subject_id) = ids.subject_id {
        constants.push((Position::S, ArkFr::from_u64(subject_id)));
    }
    if let Some(object_id) = ids.object_id {
        constants.push((Position::O, ArkFr::from_u64(object_id)));
    }
    let mut kept = Vec::new();
    if matches!(atom.subject, AtomTerm::Variable) {
        kept.push(Position::S);
    }
    if matches!(atom.object, AtomTerm::Variable) {
        kept.push(Position::O);
    }
    AtomFilterSpec { constants, kept }
}

/// Bind one atom term (subject or object) plus its optional resolved id into the shared
/// transcript: a domain-separated tag per variant (variable / node / value) and, for a
/// constant, the term bytes and revealed id. Identical on both sides.
fn bind_atom_term(transcript: &mut Transcript254, term: &AtomTerm, id: Option<u64>) {
    match term {
        AtomTerm::Variable => transcript.append_bytes(b"pow/bgp-layer/term-variable", &[]),
        AtomTerm::NodeConstant(iri) => transcript.append_bytes(b"pow/bgp-layer/term-node", iri),
        AtomTerm::ValueConstant(datatype, lexical) => {
            transcript.append_bytes(b"pow/bgp-layer/term-value-dt", &[*datatype as u8]);
            transcript.append_bytes(b"pow/bgp-layer/term-value-lex", lexical);
        }
    }
    if let Some(id) = id {
        transcript.append_bytes(b"pow/bgp-layer/term-id", &id.to_le_bytes());
    }
}

/// Bind the public statement of a layer BGP: the trust anchor (`commitment_root`, which
/// binds the column, predicate-dictionary AND node/value-dictionary commitments), `k`, and
/// every atom's subject term, predicate IRI + id, and object term (each constant carrying
/// its revealed id), then the join plan. Identical on both sides.
fn bind_layer_bgp_statement(
    transcript: &mut Transcript254,
    layer: &LayerCommitment,
    plan: &LayerBgpPlan,
    atom_ids: &[AtomIds],
) {
    transcript.append_bytes(b"pow/bgp-layer/commitment-root", &layer.state.commitment_root);
    transcript.append_bytes(b"pow/bgp-layer/k", &(plan.atoms.len() as u64).to_le_bytes());
    for (atom, ids) in plan.atoms.iter().zip(atom_ids.iter()) {
        bind_atom_term(transcript, &atom.subject, ids.subject_id);
        transcript.append_bytes(b"pow/bgp-layer/pred-term", &atom.predicate_term);
        transcript.append_bytes(b"pow/bgp-layer/pred-id", &ids.predicate_id.to_le_bytes());
        bind_atom_term(transcript, &atom.object, ids.object_id);
    }
    for step in &plan.joins {
        transcript.append_bytes(b"pow/bgp-layer/acc-key", &(step.acc_key as u64).to_le_bytes());
        transcript.append_bytes(b"pow/bgp-layer/rel-key", &(step.rel_key as u64).to_le_bytes());
    }
}

fn validate_bgp_plan(plan: &LayerBgpPlan) -> io::Result<usize> {
    let k = plan.atoms.len();
    if k < 2 {
        return Err(invalid("layer BGP needs at least two atoms"));
    }
    if plan.joins.len() != k - 1 {
        return Err(invalid("layer BGP plan must have exactly k-1 join steps"));
    }
    Ok(k)
}

/// Prove the COMPLETE result of a k-atom conjunctive BGP against a materialized base
/// layer. Each atom is filtered by its (dictionary-authenticated) constant PREDICATE and
/// any (dictionary-authenticated) subject/object CONSTANTS into its dense KEPT (variable)
/// columns; the resulting `k` relations (one column per variable, in S,O order) are joined
/// by the left-deep binary-join chain ([`super::bgp`]) on the plan's shared variables — all
/// under ONE shared transcript anchored to `commitment_root`. Returns the combined proof
/// (whose per-atom parts carry the revealed subject/object constant ids), the final result
/// column commitments (schema `[r_other_out.., key, s_other_out..]`), the result length `m`,
/// and the REVEALED resolved predicate ids (authenticated by membership).
pub fn prove_layer_bgp(
    layer: &LayerCommitment,
    plan: &LayerBgpPlan,
) -> io::Result<(LayerBgpProof, Vec<ArkGT>, usize, Vec<u64>)> {
    let k = validate_bgp_plan(plan)?;
    let (s, p, o, n) = base_layer_columns(layer)?;

    // Resolve every atom's constant ids up front (fail closed if any is absent): the
    // predicate IRI via the predicate dictionary, each subject/object constant via the
    // node/value dictionary.
    let mut atom_ids = Vec::with_capacity(k);
    for atom in &plan.atoms {
        validate_atom(atom)?;
        let predicate_id = resolve_predicate_id(layer, &atom.predicate_term)?;
        let subject_id = match atom_term_object(&atom.subject) {
            Some(object) => Some(resolve_node_value_id(layer, &object)?),
            None => None,
        };
        let object_id = match atom_term_object(&atom.object) {
            Some(object) => Some(resolve_node_value_id(layer, &object)?),
            None => None,
        };
        atom_ids.push(AtomIds {
            predicate_id,
            subject_id,
            object_id,
        });
    }
    let predicate_ids: Vec<u64> = atom_ids.iter().map(|ids| ids.predicate_id).collect();

    let (pred_dict_hash, pred_dict_id) = layer.predicate_dictionary_columns()?;
    let com_pred_hash = layer.predicate_dict_commitments.0;
    let com_pred_id = layer.predicate_dict_commitments.1;
    let (nv_dict_hash, nv_dict_id) = layer.node_value_dictionary_columns()?;
    let com_nv_hash = layer.node_value_dict_commitments.0;
    let com_nv_id = layer.node_value_dict_commitments.1;
    let com_s = layer.column_commitments[0];
    let com_p = layer.column_commitments[1];
    let com_o = layer.column_commitments[2];
    let a = padded_len(n)?.trailing_zeros() as usize;

    let mut transcript = Transcript254::new(LAYER_BGP_DOMAIN);
    bind_layer_bgp_statement(&mut transcript, layer, plan, &atom_ids);

    let mut atom_proofs = Vec::with_capacity(k);
    let mut relations: Vec<RelationWitness> = Vec::with_capacity(k);
    for (atom, ids) in plan.atoms.iter().zip(atom_ids.iter()) {
        // (1) Predicate IRI -> id, authenticated by the predicate dictionary.
        let predicate_membership = prove_dict_membership(
            &pred_dict_hash,
            &pred_dict_id,
            &com_pred_hash,
            &com_pred_id,
            hash_term(&atom.predicate_term),
            ArkFr::from_u64(ids.predicate_id),
            &mut transcript,
        )?;
        // (2) Subject / object CONSTANT -> id, authenticated by the node/value dictionary.
        // The SAME id is folded into the atom filter below, so a prover cannot authenticate
        // one id here and filter on a different one.
        let subject_membership = match (atom_term_object(&atom.subject), ids.subject_id) {
            (Some(object), Some(subject_id)) => Some(prove_dict_membership(
                &nv_dict_hash,
                &nv_dict_id,
                &com_nv_hash,
                &com_nv_id,
                hash_object(&object),
                ArkFr::from_u64(subject_id),
                &mut transcript,
            )?),
            _ => None,
        };
        let object_membership = match (atom_term_object(&atom.object), ids.object_id) {
            (Some(object), Some(object_id)) => Some(prove_dict_membership(
                &nv_dict_hash,
                &nv_dict_id,
                &com_nv_hash,
                &com_nv_id,
                hash_object(&object),
                ArkFr::from_u64(object_id),
                &mut transcript,
            )?),
            _ => None,
        };
        // (3) The complete per-atom filter (predicate + any subject/object constants) into
        // the dense KEPT (variable) columns, root-anchored via the filter argument.
        let spec = atom_filter_spec(atom, ids);
        let (filter, kept_commitments, m) =
            prove_atom_filter(&spec, &s, &p, &o, n, &com_s, &com_p, &com_o, &mut transcript)?;
        // (4) The dense kept columns (== what prove_atom_filter committed) as the chain
        // relation; its arity equals the number of variables in the atom.
        let (kept_columns, m_columns) = atom_kept_columns(&spec, &s, &p, &o, n)?;
        debug_assert_eq!(m_columns, m);
        relations.push(RelationWitness {
            columns: kept_columns,
            commitments: kept_commitments.clone(),
            log_domain: a,
            n: m,
        });
        atom_proofs.push(LayerBgpAtomProof {
            predicate_membership,
            subject_membership,
            object_membership,
            filter,
            kept_commitments,
            m,
            subject_id: ids.subject_id,
            object_id: ids.object_id,
        });
    }

    // (5) The left-deep chain over the k filtered relations, same shared transcript.
    let (chain, result_coms, result_m) = prove_bgp_chain(&relations, &plan.joins, &mut transcript)?;

    Ok((
        LayerBgpProof {
            atoms: atom_proofs,
            chain,
        },
        result_coms,
        result_m,
        predicate_ids,
    ))
}

/// Verify a k-atom layer BGP proof. Anchors to `commitment_root` FIRST (exactly like
/// [`verify_layer_graph_pattern2`]); then, under the shared transcript in prover order,
/// re-checks every atom's dictionary binding + complete filter, and the left-deep join
/// chain over the filtered relations (which enforces the two continuity conditions).
/// Concludes `result_coms[0..m]` is EXACTLY the complete BGP result, without rerunning it.
#[allow(clippy::too_many_arguments)]
pub fn verify_layer_bgp(
    layer: &LayerCommitment,
    expected_root: ProofHash,
    plan: &LayerBgpPlan,
    predicate_ids: &[u64],
    result_coms: &[ArkGT],
    m: usize,
    proof: &LayerBgpProof,
) -> io::Result<()> {
    // Anchor to an EXTERNALLY-trusted root FIRST (FIX 4), before any sub-proof.
    if layer.state.commitment_root != expected_root {
        return Err(invalid(
            "layer commitment_root does not match the externally-trusted expected root",
        ));
    }
    if layer.recomputed_commitment_root()? != layer.state.commitment_root {
        return Err(invalid(
            "layer commitment_root does not bind its committed columns/dictionary",
        ));
    }
    if layer.state.removal_count != 0 {
        return Err(invalid(
            "layer BGP query requires a materialized base layer (no removals)",
        ));
    }
    let k = validate_bgp_plan(plan)?;
    if proof.atoms.len() != k || predicate_ids.len() != k {
        return Err(invalid("layer BGP proof/atom/id count mismatch"));
    }
    let n = layer.state.addition_count as usize;
    let a = padded_len(n)?.trailing_zeros() as usize;

    // Reconstruct each atom's resolved ids from the public predicate ids + the
    // proof-carried subject/object constant ids, checking the id presence matches the
    // atom's constant structure (and the atom itself satisfies the v1 constraints).
    let mut atom_ids = Vec::with_capacity(k);
    for (i, (atom, ap)) in plan.atoms.iter().zip(proof.atoms.iter()).enumerate() {
        validate_atom(atom)?;
        let ids = AtomIds {
            predicate_id: predicate_ids[i],
            subject_id: ap.subject_id,
            object_id: ap.object_id,
        };
        check_ids_consistent(atom, &ids)?;
        // The membership presence must match the constant structure too.
        if ids.subject_id.is_some() != ap.subject_membership.is_some()
            || ids.object_id.is_some() != ap.object_membership.is_some()
        {
            return Err(invalid(
                "layer BGP atom membership presence does not match the plan",
            ));
        }
        atom_ids.push(ids);
    }

    let mut transcript = Transcript254::new(LAYER_BGP_DOMAIN);
    bind_layer_bgp_statement(&mut transcript, layer, plan, &atom_ids);

    let com_pred_hash = layer.predicate_dict_commitments.0;
    let com_pred_id = layer.predicate_dict_commitments.1;
    let com_nv_hash = layer.node_value_dict_commitments.0;
    let com_nv_id = layer.node_value_dict_commitments.1;
    let com_s = layer.column_commitments[0];
    let com_p = layer.column_commitments[1];
    let com_o = layer.column_commitments[2];

    let mut relations: Vec<RelationCommit> = Vec::with_capacity(k);
    for ((atom, ap), ids) in plan
        .atoms
        .iter()
        .zip(proof.atoms.iter())
        .zip(atom_ids.iter())
    {
        // A compacted atom filter's dimension must match the layer's committed table
        // columns (defense-in-depth; the Dory openings would also catch a mismatch).
        if let AtomFilterProof::Compacted(inner) = &ap.filter {
            if inner.sumcheck.round_polys.len() != a {
                return Err(invalid(
                    "atom filter dimension does not match the layer's committed columns",
                ));
            }
        }
        // (1) Predicate membership: the revealed id is authentic for this IRI.
        verify_dict_membership(
            &com_pred_hash,
            &com_pred_id,
            hash_term(&atom.predicate_term),
            ArkFr::from_u64(ids.predicate_id),
            &ap.predicate_membership,
            &mut transcript,
        )?;
        // (2) Subject / object constant memberships, in the SAME order as the prover. The
        // authenticated id is reused verbatim in the filter spec below (the security tie).
        if let Some(subject_id) = ids.subject_id {
            let object = atom_term_object(&atom.subject)
                .ok_or_else(|| invalid("subject id present for a variable subject"))?;
            let membership = ap
                .subject_membership
                .as_ref()
                .ok_or_else(|| invalid("missing subject constant membership"))?;
            verify_dict_membership(
                &com_nv_hash,
                &com_nv_id,
                hash_object(&object),
                ArkFr::from_u64(subject_id),
                membership,
                &mut transcript,
            )?;
        }
        if let Some(object_id) = ids.object_id {
            let object = atom_term_object(&atom.object)
                .ok_or_else(|| invalid("object id present for a variable object"))?;
            let membership = ap
                .object_membership
                .as_ref()
                .ok_or_else(|| invalid("missing object constant membership"))?;
            verify_dict_membership(
                &com_nv_hash,
                &com_nv_id,
                hash_object(&object),
                ArkFr::from_u64(object_id),
                membership,
                &mut transcript,
            )?;
        }
        // (3) Filter: `kept_commitments[0..m]` is the complete per-atom result. The spec
        // reuses the SAME ids the memberships authenticated.
        let spec = atom_filter_spec(atom, ids);
        verify_atom_filter(
            &spec,
            n,
            &com_s,
            &com_p,
            &com_o,
            &ap.kept_commitments,
            ap.m,
            &ap.filter,
            &mut transcript,
        )?;
        relations.push(RelationCommit {
            commitments: ap.kept_commitments.clone(),
            log_domain: a,
            n: ap.m,
        });
    }

    // (4) The left-deep chain over the k filtered relations.
    let (chain_coms, chain_m) =
        verify_bgp_chain(&relations, &plan.joins, &proof.chain, &mut transcript)?;
    if chain_m != m || chain_coms.as_slice() != result_coms {
        return Err(invalid(
            "layer BGP result does not match the proof's chain output",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::ValueTriple;
    use crate::proof::argument::commit_column;
    use crate::store::sync::open_sync_memory_store;

    const P1: &[u8] = b"http://ex/p1";
    const P2: &[u8] = b"http://ex/p2";

    fn iri(suffix: &str) -> String {
        format!("http://ex/{suffix}")
    }

    // Build a materialized base layer from (subject, predicate, object) IRI triples.
    fn build_layer(triples: &[(&str, &str, &str)]) -> std::sync::Arc<LayerCommitment> {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for (s, p, o) in triples {
            builder
                .add_value_triple(ValueTriple::new_node(&iri(s), &iri(p), &iri(o)))
                .unwrap();
        }
        let (layer, _state) = builder.commit_with_proof().unwrap();
        layer.proof_commitment().unwrap().unwrap()
    }

    // Independently recompute the expected dense result for a predicate id.
    fn expected_result(layer: &LayerCommitment, pred_id: u64) -> (Vec<ArkFr>, Vec<ArkFr>, usize) {
        let padded = padded_len(layer.additions.len()).unwrap();
        let mut s_out = vec![f0(); padded];
        let mut o_out = vec![f0(); padded];
        let mut m = 0usize;
        for record in &layer.additions {
            if record.predicate_id == pred_id {
                s_out[m] = ArkFr::from_u64(record.subject_id);
                o_out[m] = ArkFr::from_u64(record.object_id);
                m += 1;
            }
        }
        (s_out, o_out, m)
    }

    fn sample_layer() -> std::sync::Arc<LayerCommitment> {
        build_layer(&[
            ("a1", "p1", "b1"),
            ("a2", "p1", "b2"),
            ("a3", "p1", "b3"),
            ("a4", "p1", "b4"),
            ("c1", "p2", "d1"),
            ("c2", "p2", "d2"),
        ])
    }

    #[test]
    fn query_commitment_convention_matches_layer_commitment_columns() {
        let lc = build_layer(&[("a1", "p1", "b1"), ("a2", "p1", "b2"), ("c1", "p2", "d1")]);
        let (s, p, o, _n) = base_layer_columns(&lc).unwrap();
        assert_eq!(commit_column(&s).unwrap(), lc.column_commitments[0]);
        assert_eq!(commit_column(&p).unwrap(), lc.column_commitments[1]);
        assert_eq!(commit_column(&o).unwrap(), lc.column_commitments[2]);
    }

    #[test]
    fn prove_and_verify_by_iri_with_exact_result() {
        let lc = sample_layer();
        let (proof, com_sout, com_oout, m, resolved_id) = prove_layer_filter(&lc, P1).unwrap();
        // The verifier is given ONLY the IRI bytes + the revealed id (authenticated
        // by membership) -- never an out-of-band id.
        verify_layer_filter(&lc, lc.state.commitment_root, P1,resolved_id, &com_sout, &com_oout, m, &proof).unwrap();

        // Result is EXACTLY the p1 rows (count + contents).
        assert_eq!(m, 4);
        let (exp_s, exp_o, exp_m) = expected_result(&lc, resolved_id);
        assert_eq!(exp_m, m);
        assert_eq!(commit_column(&exp_s).unwrap(), com_sout);
        assert_eq!(commit_column(&exp_o).unwrap(), com_oout);
    }

    #[test]
    fn wrong_revealed_id_for_the_iri_is_rejected() {
        let lc = sample_layer();
        let (proof, com_sout, com_oout, m, resolved_id) = prove_layer_filter(&lc, P1).unwrap();
        // Claim a different id for the same IRI -> membership fails.
        let wrong = resolved_id.wrapping_add(1);
        assert!(
            verify_layer_filter(&lc, lc.state.commitment_root, P1, wrong, &com_sout, &com_oout, m, &proof)
                .is_err()
        );
    }

    #[test]
    fn substituted_id_from_a_different_predicate_is_rejected() {
        let lc = sample_layer();
        let (proof, com_sout, com_oout, m, _p1_id) = prove_layer_filter(&lc, P1).unwrap();
        // p2's genuine id, but paired with p1's IRI: (hash_term(p1), id_p2) is not a
        // dictionary row -> membership fails.
        let p2_id = resolve_predicate_id(&lc, P2).unwrap();
        assert!(
            verify_layer_filter(&lc, lc.state.commitment_root, P1, p2_id, &com_sout, &com_oout, m, &proof)
                .is_err()
        );
    }

    #[test]
    fn absent_predicate_iri_fails_closed() {
        let lc = sample_layer();
        let err = prove_layer_filter(&lc, b"http://ex/absent").unwrap_err();
        assert!(err.to_string().contains("predicate not in layer dictionary"));
    }

    #[test]
    fn tampered_result_is_rejected_under_the_combined_transcript() {
        let lc = sample_layer();
        let (proof, com_sout, com_oout, m, resolved_id) = prove_layer_filter(&lc, P1).unwrap();
        // Wrong S_out commitment (use the O_out one) -> filter opening fails.
        assert!(
            verify_layer_filter(&lc, lc.state.commitment_root, P1,resolved_id, &com_oout, &com_oout, m, &proof).is_err()
        );
        // Honest still verifies.
        verify_layer_filter(&lc, lc.state.commitment_root, P1,resolved_id, &com_sout, &com_oout, m, &proof).unwrap();
    }

    #[test]
    fn non_materialized_layer_fails_closed() {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for (s, p, o) in [("a1", "p1", "b1"), ("a2", "p1", "b2")] {
            builder
                .add_value_triple(ValueTriple::new_node(&iri(s), &iri(p), &iri(o)))
                .unwrap();
        }
        let (base, _) = builder.commit_with_proof().unwrap();
        let child = base.open_write().unwrap();
        child
            .remove_value_triple(ValueTriple::new_node(&iri("a1"), &iri("p1"), &iri("b1")))
            .unwrap();
        let (head, _) = child.commit_with_proof().unwrap();
        let lc = head.proof_commitment().unwrap().unwrap();
        assert!(prove_layer_filter(&lc, P1).is_err());
    }

    // ---- Q2: 2-atom graph-pattern (self-join) end-to-end ----

    const KNOWS: &[u8] = b"http://ex/knows";

    // Independently recompute the complete 2-hop result for a predicate id, from the
    // layer's additions in stored order (matching the filter output's F ordering).
    fn expected_join(
        lc: &LayerCommitment,
        pred_id: u64,
    ) -> (Vec<ArkFr>, Vec<ArkFr>, Vec<ArkFr>, usize, usize) {
        let mut sf = Vec::new();
        let mut of = Vec::new();
        for record in &lc.additions {
            if record.predicate_id == pred_id {
                sf.push(ArkFr::from_u64(record.subject_id));
                of.push(ArkFr::from_u64(record.object_id));
            }
        }
        let (x, y, z, _rl, _rr, u, _wl, _wr, m_j) = compute_self_join(&sf, &of);
        let n_join = self_join_domain(sf.len(), m_j, u.len()).unwrap();
        (x, y, z, n_join, m_j)
    }

    #[test]
    fn graph_pattern2_two_hop_knows_paths_end_to_end() {
        // knows: a->b, b->c, c->d, a->e (e a sink). 2-hop paths (x->y->z): (a,b,c) and
        // (b,c,d). One unrelated "likes" edge exercises the filter's predicate isolation.
        let lc = build_layer(&[
            ("a", "knows", "b"),
            ("b", "knows", "c"),
            ("c", "knows", "d"),
            ("a", "knows", "e"),
            ("a", "likes", "zz"),
        ]);
        let (proof, com_x, com_y, com_z, m_j, id) =
            prove_layer_graph_pattern2(&lc, KNOWS).unwrap();
        // Verifier sees ONLY the IRI + revealed id + result commitments.
        verify_layer_graph_pattern2(&lc, lc.state.commitment_root, KNOWS,id, &com_x, &com_y, &com_z, m_j, &proof).unwrap();

        // Result is EXACTLY the two 2-hop paths (count + contents).
        assert_eq!(m_j, 2);
        let (x, y, z, n_join, exp_m) = expected_join(&lc, id);
        assert_eq!(exp_m, m_j);
        assert_eq!(commit_column(&pad(&x, n_join)).unwrap(), com_x);
        assert_eq!(commit_column(&pad(&y, n_join)).unwrap(), com_y);
        assert_eq!(commit_column(&pad(&z, n_join)).unwrap(), com_z);
    }

    #[test]
    fn graph_pattern2_mismatched_result_or_count_is_rejected() {
        let lc = build_layer(&[("a", "knows", "b"), ("b", "knows", "c"), ("c", "knows", "d")]);
        let (proof, com_x, com_y, com_z, m_j, id) =
            prove_layer_graph_pattern2(&lc, KNOWS).unwrap();
        // Wrong result commitment (pass Com Z for Com X) -> mismatch with proof output.
        assert!(
            verify_layer_graph_pattern2(&lc, lc.state.commitment_root, KNOWS,id, &com_z, &com_y, &com_z, m_j, &proof)
                .is_err()
        );
        // Wrong claimed count.
        assert!(
            verify_layer_graph_pattern2(&lc, lc.state.commitment_root, KNOWS,id, &com_x, &com_y, &com_z, m_j + 1, &proof)
                .is_err()
        );
        // Honest verifies.
        verify_layer_graph_pattern2(&lc, lc.state.commitment_root, KNOWS,id, &com_x, &com_y, &com_z, m_j, &proof).unwrap();
    }

    #[test]
    fn graph_pattern2_fabricated_join_output_is_rejected() {
        // Fabricate a bogus X-commitment and pass it consistently (defeats the equality
        // gate); the self-join column opening still binds X to the sumcheck -> REJECT.
        let lc = build_layer(&[("a", "knows", "b"), ("b", "knows", "c"), ("c", "knows", "d")]);
        let (proof, _com_x, com_y, com_z, m_j, id) =
            prove_layer_graph_pattern2(&lc, KNOWS).unwrap();
        let (_, _, _, n_join, _) = expected_join(&lc, id);
        let fake_x = commit_column(&pad(&[ArkFr::from_u64(424242)], n_join)).unwrap();
        let mut bad = proof.clone();
        bad.join_commitments.x = fake_x;
        assert!(
            verify_layer_graph_pattern2(&lc, lc.state.commitment_root, KNOWS,id, &fake_x, &com_y, &com_z, m_j, &bad)
                .is_err()
        );
    }

    #[test]
    fn graph_pattern2_no_two_hop_paths_empty_verifies() {
        // Disjoint edges a->b and c->d: no y with x->y and y->z, so the result is empty.
        let lc = build_layer(&[("a", "knows", "b"), ("c", "knows", "d")]);
        let (proof, com_x, com_y, com_z, m_j, id) =
            prove_layer_graph_pattern2(&lc, KNOWS).unwrap();
        assert_eq!(m_j, 0);
        verify_layer_graph_pattern2(&lc, lc.state.commitment_root, KNOWS,id, &com_x, &com_y, &com_z, m_j, &proof).unwrap();
    }

    #[test]
    fn graph_pattern2_tampered_layer_root_is_rejected() {
        let lc = build_layer(&[("a", "knows", "b"), ("b", "knows", "c"), ("c", "knows", "d")]);
        let good_root = lc.state.commitment_root;
        let (proof, com_x, com_y, com_z, m_j, id) =
            prove_layer_graph_pattern2(&lc, KNOWS).unwrap();
        // Corrupt the anchor: a layer whose stored root does not bind its columns.
        let mut tampered = (*lc).clone();
        tampered.state.commitment_root[0] ^= 0xff;
        // Even pinned to the (correct) external root, the tampered layer's stored root
        // no longer matches it -> the FIX 4 anchor rejects.
        assert!(verify_layer_graph_pattern2(
            &tampered, good_root, KNOWS, id, &com_x, &com_y, &com_z, m_j, &proof
        )
        .is_err());
    }

    #[test]
    fn graph_pattern2_wrong_expected_root_is_rejected() {
        // FIX 4 (external-root anchor): an honest proof against an honest layer must
        // still REJECT when the verifier pins a DIFFERENT externally-trusted root.
        let lc = build_layer(&[("a", "knows", "b"), ("b", "knows", "c"), ("c", "knows", "d")]);
        let (proof, com_x, com_y, com_z, m_j, id) =
            prove_layer_graph_pattern2(&lc, KNOWS).unwrap();
        let mut wrong_root = lc.state.commitment_root;
        wrong_root[0] ^= 0xff;
        assert!(verify_layer_graph_pattern2(
            &lc, wrong_root, KNOWS, id, &com_x, &com_y, &com_z, m_j, &proof
        )
        .is_err());
        // The correct external root still verifies.
        verify_layer_graph_pattern2(
            &lc,
            lc.state.commitment_root,
            KNOWS,
            id,
            &com_x,
            &com_y,
            &com_z,
            m_j,
            &proof,
        )
        .unwrap();
    }

    // ---- Q_k: general k-atom layer BGP (left-deep chain) end-to-end ----

    const LIKES: &[u8] = b"http://ex/likes";
    const OWNS: &[u8] = b"http://ex/owns";

    fn atom(term: &[u8]) -> LayerBgpAtom {
        LayerBgpAtom::all_variables(term.to_vec())
    }

    // The two (subject, object) columns of one atom relation, filtered from the layer's
    // additions in stored order (matching the filter output ordering).
    fn atom_relation(lc: &LayerCommitment, pred_id: u64) -> Vec<Vec<ArkFr>> {
        let mut s = Vec::new();
        let mut o = Vec::new();
        for rec in &lc.additions {
            if rec.predicate_id == pred_id {
                s.push(ArkFr::from_u64(rec.subject_id));
                o.push(ArkFr::from_u64(rec.object_id));
            }
        }
        vec![s, o]
    }

    // Independently recompute the complete BGP result columns (real length) and the final
    // join domain, folding the audited compute_binary_join over the plan's join steps.
    fn expected_layer_bgp(
        lc: &LayerCommitment,
        ids: &[u64],
        joins: &[JoinStep],
    ) -> (Vec<Vec<ArkFr>>, usize) {
        use crate::proof::join2::{binary_join_domain, compute_binary_join};
        let split = |cols: &[Vec<ArkFr>], key: usize| -> (Vec<ArkFr>, Vec<Vec<ArkFr>>) {
            let k = cols[key].clone();
            let other = cols
                .iter()
                .enumerate()
                .filter(|(c, _)| *c != key)
                .map(|(_, c)| c.clone())
                .collect();
            (k, other)
        };
        let mut acc = atom_relation(lc, ids[0]);
        let mut acc_len = acc[0].len();
        let mut domain = padded_len(lc.additions.len()).unwrap();
        for (j, step) in joins.iter().enumerate() {
            let rel = atom_relation(lc, ids[j + 1]);
            let rel_len = rel[0].len();
            let (r_key, r_other) = split(&acc, step.acc_key);
            let (s_key, s_other) = split(&rel, step.rel_key);
            let r_refs: Vec<&[ArkFr]> = r_other.iter().map(|c| c.as_slice()).collect();
            let s_refs: Vec<&[ArkFr]> = s_other.iter().map(|c| c.as_slice()).collect();
            let w = compute_binary_join(&r_key, &r_refs, &s_key, &s_refs);
            domain = binary_join_domain(acc_len, rel_len, w.m_j, w.n_u).unwrap();
            let mut new_acc = w.r_other_out.clone();
            new_acc.push(w.key.clone());
            new_acc.extend(w.s_other_out.clone());
            acc = new_acc;
            acc_len = w.m_j;
        }
        (acc, domain)
    }

    #[test]
    fn layer_bgp_three_atom_path_end_to_end() {
        // knows: a->b, c->b, e->f;  likes: b->g, f->h;  owns: g->k, h->m.
        // BGP t(?x,knows,?y), t(?y,likes,?z), t(?z,owns,?w). Paths (x,y,z,w):
        //   (a,b,g,k), (c,b,g,k), (e,f,h,m).
        let lc = build_layer(&[
            ("a", "knows", "b"),
            ("c", "knows", "b"),
            ("e", "knows", "f"),
            ("b", "likes", "g"),
            ("f", "likes", "h"),
            ("g", "owns", "k"),
            ("h", "owns", "m"),
        ]);
        let plan = LayerBgpPlan {
            atoms: vec![atom(KNOWS), atom(LIKES), atom(OWNS)],
            joins: vec![
                JoinStep { acc_key: 1, rel_key: 0 }, // y = O(knows) == S(likes)
                JoinStep { acc_key: 2, rel_key: 0 }, // z = O(likes) == S(owns)
            ],
        };
        let (proof, result_coms, m, ids) = prove_layer_bgp(&lc, &plan).unwrap();
        // Verifier sees ONLY the IRIs + revealed ids + result commitments.
        verify_layer_bgp(&lc, lc.state.commitment_root, &plan, &ids, &result_coms, m, &proof).unwrap();

        // Result is EXACTLY the three 3-hop paths (count + committed columns).
        assert_eq!(m, 3);
        assert_eq!(proof.atoms.len(), 3);
        let (exp_cols, domain) = expected_layer_bgp(&lc, &ids, &plan.joins);
        assert_eq!(result_coms.len(), exp_cols.len());
        assert_eq!(exp_cols.len(), 4); // [x, y, z, w]
        for (c, col) in exp_cols.iter().enumerate() {
            assert_eq!(commit_column(&pad(col, domain)).unwrap(), result_coms[c]);
        }
    }

    #[test]
    fn layer_bgp_k2_matches_self_join_capstone() {
        // The k=2 layer BGP (two knows atoms joined on the shared y) reduces to the
        // existing 2-atom self-join capstone: SAME result count AND committed columns.
        let lc = build_layer(&[
            ("a", "knows", "b"),
            ("b", "knows", "c"),
            ("c", "knows", "d"),
            ("a", "knows", "e"),
            ("a", "likes", "zz"),
        ]);
        let (gp, com_x, com_y, com_z, gp_m, gp_id) =
            prove_layer_graph_pattern2(&lc, KNOWS).unwrap();
        verify_layer_graph_pattern2(
            &lc, lc.state.commitment_root, KNOWS, gp_id, &com_x, &com_y, &com_z, gp_m, &gp,
        )
        .unwrap();

        let plan = LayerBgpPlan {
            atoms: vec![atom(KNOWS), atom(KNOWS)],
            joins: vec![JoinStep { acc_key: 1, rel_key: 0 }],
        };
        let (proof, coms, m, ids) = prove_layer_bgp(&lc, &plan).unwrap();
        verify_layer_bgp(&lc, lc.state.commitment_root, &plan, &ids, &coms, m, &proof).unwrap();

        assert_eq!(m, gp_m);
        assert_eq!(coms, vec![com_x, com_y, com_z]);
    }

    #[test]
    fn layer_bgp_wrong_expected_root_is_rejected() {
        let lc = build_layer(&[
            ("a", "knows", "b"),
            ("b", "likes", "c"),
            ("c", "owns", "d"),
        ]);
        let plan = LayerBgpPlan {
            atoms: vec![atom(KNOWS), atom(LIKES), atom(OWNS)],
            joins: vec![
                JoinStep { acc_key: 1, rel_key: 0 },
                JoinStep { acc_key: 2, rel_key: 0 },
            ],
        };
        let (proof, result_coms, m, ids) = prove_layer_bgp(&lc, &plan).unwrap();
        // A DIFFERENT externally-trusted root must reject even an honest proof.
        let mut wrong_root = lc.state.commitment_root;
        wrong_root[0] ^= 0xff;
        assert!(
            verify_layer_bgp(&lc, wrong_root, &plan, &ids, &result_coms, m, &proof).is_err()
        );
        // The correct external root still verifies.
        verify_layer_bgp(&lc, lc.state.commitment_root, &plan, &ids, &result_coms, m, &proof)
            .unwrap();
    }

    #[test]
    fn layer_bgp_absent_predicate_fails_closed() {
        let lc = build_layer(&[("a", "knows", "b"), ("b", "knows", "c")]);
        let plan = LayerBgpPlan {
            atoms: vec![atom(KNOWS), atom(b"http://ex/absent")],
            joins: vec![JoinStep { acc_key: 1, rel_key: 0 }],
        };
        assert!(prove_layer_bgp(&lc, &plan).is_err());
    }

    // ---- Q_k with subject/object CONSTANTS (node/value dictionary) end-to-end ----

    const NAME: &[u8] = b"http://ex/name";

    // Build a materialized base layer from arbitrary ValueTriples (node OR value objects).
    fn build_layer_triples(triples: Vec<ValueTriple>) -> std::sync::Arc<LayerCommitment> {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for t in triples {
            builder.add_value_triple(t).unwrap();
        }
        let (layer, _state) = builder.commit_with_proof().unwrap();
        layer.proof_commitment().unwrap().unwrap()
    }

    fn node_const(suffix: &str) -> AtomTerm {
        AtomTerm::NodeConstant(iri(suffix).into_bytes())
    }

    // One atom's dense KEPT (variable) columns from the layer additions in stored order,
    // applying the predicate + any subject/object constant ids. Mirrors EXACTLY what
    // prove_atom_filter compacts (kept in S,O order); one column per variable.
    fn kept_relation(
        lc: &LayerCommitment,
        atom: &LayerBgpAtom,
        predicate_id: u64,
        subject_id: Option<u64>,
        object_id: Option<u64>,
    ) -> Vec<Vec<ArkFr>> {
        let keep_s = matches!(atom.subject, AtomTerm::Variable);
        let keep_o = matches!(atom.object, AtomTerm::Variable);
        let mut cols: Vec<Vec<ArkFr>> = Vec::new();
        if keep_s {
            cols.push(Vec::new());
        }
        if keep_o {
            cols.push(Vec::new());
        }
        for rec in &lc.additions {
            if rec.predicate_id == predicate_id
                && subject_id.map_or(true, |s| rec.subject_id == s)
                && object_id.map_or(true, |o| rec.object_id == o)
            {
                let mut idx = 0;
                if keep_s {
                    cols[idx].push(ArkFr::from_u64(rec.subject_id));
                    idx += 1;
                }
                if keep_o {
                    cols[idx].push(ArkFr::from_u64(rec.object_id));
                }
            }
        }
        cols
    }

    // Independently recompute the complete BGP result columns (real length) and final join
    // domain over per-atom kept relations, folding the audited compute_binary_join over the
    // plan's join steps (mirrors expected_layer_bgp but for variable-arity relations).
    fn expected_const_bgp(
        lc: &LayerCommitment,
        atoms: &[LayerBgpAtom],
        predicate_ids: &[u64],
        subject_ids: &[Option<u64>],
        object_ids: &[Option<u64>],
        joins: &[JoinStep],
    ) -> (Vec<Vec<ArkFr>>, usize) {
        use crate::proof::join2::{binary_join_domain, compute_binary_join};
        let split = |cols: &[Vec<ArkFr>], key: usize| -> (Vec<ArkFr>, Vec<Vec<ArkFr>>) {
            let k = cols[key].clone();
            let other = cols
                .iter()
                .enumerate()
                .filter(|(c, _)| *c != key)
                .map(|(_, c)| c.clone())
                .collect();
            (k, other)
        };
        let rel = |i: usize| {
            kept_relation(lc, &atoms[i], predicate_ids[i], subject_ids[i], object_ids[i])
        };
        let mut acc = rel(0);
        let mut acc_len = acc[0].len();
        let mut domain = padded_len(lc.additions.len()).unwrap();
        for (j, step) in joins.iter().enumerate() {
            let r = rel(j + 1);
            let rel_len = r[0].len();
            let (r_key, r_other) = split(&acc, step.acc_key);
            let (s_key, s_other) = split(&r, step.rel_key);
            let r_refs: Vec<&[ArkFr]> = r_other.iter().map(|c| c.as_slice()).collect();
            let s_refs: Vec<&[ArkFr]> = s_other.iter().map(|c| c.as_slice()).collect();
            let w = compute_binary_join(&r_key, &r_refs, &s_key, &s_refs);
            domain = binary_join_domain(acc_len, rel_len, w.m_j, w.n_u).unwrap();
            let mut new_acc = w.r_other_out.clone();
            new_acc.push(w.key.clone());
            new_acc.extend(w.s_other_out.clone());
            acc = new_acc;
            acc_len = w.m_j;
        }
        (acc, domain)
    }

    // Prove+verify a 2-atom const-bearing plan, then assert the result equals the
    // INDEPENDENTLY-recomputed filtered+joined result (count + committed columns).
    fn check_const_bgp(lc: &LayerCommitment, plan: &LayerBgpPlan, expected_m: usize) {
        let (proof, coms, m, ids) = prove_layer_bgp(lc, plan).unwrap();
        verify_layer_bgp(lc, lc.state.commitment_root, plan, &ids, &coms, m, &proof).unwrap();
        assert_eq!(m, expected_m);
        let subject_ids: Vec<Option<u64>> = proof.atoms.iter().map(|a| a.subject_id).collect();
        let object_ids: Vec<Option<u64>> = proof.atoms.iter().map(|a| a.object_id).collect();
        let (exp_cols, domain) =
            expected_const_bgp(lc, &plan.atoms, &ids, &subject_ids, &object_ids, &plan.joins);
        assert_eq!(coms.len(), exp_cols.len());
        for (c, col) in exp_cols.iter().enumerate() {
            assert_eq!(commit_column(&pad(col, domain)).unwrap(), coms[c], "column {c}");
        }
    }

    #[test]
    fn layer_bgp_object_node_constant_end_to_end() {
        // t(?x, likes, ?z), t(?x, knows, <b>): x likes z AND x knows <b>.
        // likes: a->m,a->o,c->n,d->p ; knows: a->b,c->b,d->q. x knows b => {a,c}.
        // Join on x (subject of both): (a,m),(a,o),(c,n). The const atom is the RIGHT
        // relation (arity 1), exercising an empty s_other in the join.
        let lc = build_layer(&[
            ("a", "likes", "m"),
            ("a", "likes", "o"),
            ("c", "likes", "n"),
            ("d", "likes", "p"),
            ("a", "knows", "b"),
            ("c", "knows", "b"),
            ("d", "knows", "q"),
        ]);
        let plan = LayerBgpPlan {
            atoms: vec![
                LayerBgpAtom::all_variables(LIKES.to_vec()),
                LayerBgpAtom {
                    subject: AtomTerm::Variable,
                    predicate_term: KNOWS.to_vec(),
                    object: node_const("b"),
                },
            ],
            joins: vec![JoinStep { acc_key: 0, rel_key: 0 }], // x = S(likes) == S(knows)
        };
        let (proof, _coms, _m, _ids) = prove_layer_bgp(&lc, &plan).unwrap();
        // The revealed object id equals <b>'s node/value dictionary id; the variable
        // atom reveals no constant ids.
        let b_id = resolve_node_value_id(&lc, &CanonicalObject::Node(iri("b").into_bytes())).unwrap();
        assert_eq!(proof.atoms[1].object_id, Some(b_id));
        assert_eq!(proof.atoms[1].subject_id, None);
        assert_eq!(proof.atoms[0].subject_id, None);
        assert_eq!(proof.atoms[0].object_id, None);
        // Result [z, x] equals the independently-recomputed filtered+joined result.
        check_const_bgp(&lc, &plan, 3);
    }

    #[test]
    fn layer_bgp_object_value_constant_end_to_end() {
        // t(?x, name, "hi"), t(?x, knows, ?z): x named "hi" AND x knows z. The value-literal
        // atom is the LEFT relation (arity 1), exercising an empty r_other in the join.
        let name_iri = iri("name");
        let lc = build_layer_triples(vec![
            ValueTriple::new_string_value(&iri("a"), &name_iri, "hi"),
            ValueTriple::new_string_value(&iri("c"), &name_iri, "hi"),
            ValueTriple::new_string_value(&iri("d"), &name_iri, "bye"),
            ValueTriple::new_node(&iri("a"), &iri("knows"), &iri("m")),
            ValueTriple::new_node(&iri("a"), &iri("knows"), &iri("o")),
            ValueTriple::new_node(&iri("c"), &iri("knows"), &iri("n")),
        ]);
        // Recover the EXACT canonical "hi" value object from the catalog (its lexical bytes
        // are the store's typed encoding, not literally b"hi"): find a's name-triple object.
        let name_id = resolve_predicate_id(&lc, NAME).unwrap();
        let a_id = resolve_node_value_id(&lc, &CanonicalObject::Node(iri("a").into_bytes())).unwrap();
        let hi_id = lc
            .additions
            .iter()
            .find(|r| r.subject_id == a_id && r.predicate_id == name_id)
            .unwrap()
            .object_id;
        let (dt, lex) = match &lc
            .node_value_catalog
            .iter()
            .find(|r| r.object_id == hi_id)
            .unwrap()
            .triple
            .object
        {
            CanonicalObject::Value { datatype, lexical } => (*datatype, lexical.clone()),
            _ => panic!("expected a value literal"),
        };
        let plan = LayerBgpPlan {
            atoms: vec![
                LayerBgpAtom {
                    subject: AtomTerm::Variable,
                    predicate_term: NAME.to_vec(),
                    object: AtomTerm::ValueConstant(dt, lex),
                },
                LayerBgpAtom::all_variables(KNOWS.to_vec()),
            ],
            joins: vec![JoinStep { acc_key: 0, rel_key: 0 }], // x = S(name) == S(knows)
        };
        let (proof, _coms, _m, _ids) = prove_layer_bgp(&lc, &plan).unwrap();
        assert_eq!(proof.atoms[0].object_id, Some(hi_id));
        assert_eq!(proof.atoms[0].subject_id, None);
        // Result [x, z] = {(a,m),(a,o),(c,n)}.
        check_const_bgp(&lc, &plan, 3);
    }

    #[test]
    fn layer_bgp_subject_node_constant_end_to_end() {
        // t(<a>, knows, ?y), t(?y, likes, ?z): a knows y AND y likes z. The subject-const
        // atom is the LEFT relation (arity 1, keeps O = y).
        // knows: a->b,a->c,d->e ; likes: b->m,b->o,c->n. a knows y => {b,c}.
        let lc = build_layer(&[
            ("a", "knows", "b"),
            ("a", "knows", "c"),
            ("d", "knows", "e"),
            ("b", "likes", "m"),
            ("b", "likes", "o"),
            ("c", "likes", "n"),
        ]);
        let plan = LayerBgpPlan {
            atoms: vec![
                LayerBgpAtom {
                    subject: node_const("a"),
                    predicate_term: KNOWS.to_vec(),
                    object: AtomTerm::Variable,
                },
                LayerBgpAtom::all_variables(LIKES.to_vec()),
            ],
            joins: vec![JoinStep { acc_key: 0, rel_key: 0 }], // y = O(knows) == S(likes)
        };
        let (proof, _coms, _m, _ids) = prove_layer_bgp(&lc, &plan).unwrap();
        let a_id = resolve_node_value_id(&lc, &CanonicalObject::Node(iri("a").into_bytes())).unwrap();
        assert_eq!(proof.atoms[0].subject_id, Some(a_id));
        assert_eq!(proof.atoms[0].object_id, None);
        // Result [y, z] = {(b,m),(b,o),(c,n)}.
        check_const_bgp(&lc, &plan, 3);
    }

    #[test]
    fn layer_bgp_mixed_constant_and_variable_atoms_and_rejections() {
        // t(?x, knows, ?y), t(?y, likes, <g>): x knows y AND y likes <g>. Joins on the
        // OBJECT variable y (acc_key = 1) of the all-variable atom against the object-const
        // atom (arity 1). knows: a->b,c->b,a->e ; likes: b->g,e->h,f->g. y likes g => {b,f}.
        let lc = build_layer(&[
            ("a", "knows", "b"),
            ("c", "knows", "b"),
            ("a", "knows", "e"),
            ("b", "likes", "g"),
            ("e", "likes", "h"),
            ("f", "likes", "g"),
        ]);
        let plan = LayerBgpPlan {
            atoms: vec![
                LayerBgpAtom::all_variables(KNOWS.to_vec()),
                LayerBgpAtom {
                    subject: AtomTerm::Variable,
                    predicate_term: LIKES.to_vec(),
                    object: node_const("g"),
                },
            ],
            joins: vec![JoinStep { acc_key: 1, rel_key: 0 }], // y = O(knows) == S(likes)
        };
        // Honest: result [x, y] = {(a,b),(c,b)}.
        check_const_bgp(&lc, &plan, 2);

        let (proof, coms, m, ids) = prove_layer_bgp(&lc, &plan).unwrap();
        verify_layer_bgp(&lc, lc.state.commitment_root, &plan, &ids, &coms, m, &proof).unwrap();

        // Wrong revealed constant id: the verifier feeds the SAME id to both the membership
        // and the filter, and binds it into the statement transcript -> a substituted id
        // breaks the node/value membership (the pair is no longer a dictionary row).
        let mut bad_id = proof.clone();
        bad_id.atoms[1].object_id = Some(proof.atoms[1].object_id.unwrap().wrapping_add(1));
        assert!(
            verify_layer_bgp(&lc, lc.state.commitment_root, &plan, &ids, &coms, m, &bad_id).is_err()
        );

        // Substituting a DIFFERENT genuine node id (here <b>'s id) for <g> is also rejected:
        // membership binds the PAIR (hash_object(<g>), id), so <b>'s id fails for <g>.
        let b_id = resolve_node_value_id(&lc, &CanonicalObject::Node(iri("b").into_bytes())).unwrap();
        let mut sub_id = proof.clone();
        sub_id.atoms[1].object_id = Some(b_id);
        assert!(
            verify_layer_bgp(&lc, lc.state.commitment_root, &plan, &ids, &coms, m, &sub_id).is_err()
        );

        // Wrong externally-trusted root rejects even an honest proof.
        let mut wrong_root = lc.state.commitment_root;
        wrong_root[0] ^= 0xff;
        assert!(
            verify_layer_bgp(&lc, wrong_root, &plan, &ids, &coms, m, &proof).is_err()
        );

        // Honest still verifies.
        verify_layer_bgp(&lc, lc.state.commitment_root, &plan, &ids, &coms, m, &proof).unwrap();
    }

    #[test]
    fn layer_bgp_absent_constant_fails_closed() {
        let lc = build_layer(&[("a", "knows", "b"), ("b", "likes", "c")]);
        // The object constant <absent> is not in the layer's node/value dictionary.
        let plan = LayerBgpPlan {
            atoms: vec![
                LayerBgpAtom::all_variables(KNOWS.to_vec()),
                LayerBgpAtom {
                    subject: AtomTerm::Variable,
                    predicate_term: LIKES.to_vec(),
                    object: node_const("absent"),
                },
            ],
            joins: vec![JoinStep { acc_key: 1, rel_key: 0 }],
        };
        let err = prove_layer_bgp(&lc, &plan).unwrap_err();
        assert!(err
            .to_string()
            .contains("constant not in layer node/value dictionary"));
    }

    #[test]
    fn layer_bgp_ground_atom_and_value_subject_are_rejected() {
        let lc = build_layer(&[("a", "knows", "b"), ("b", "likes", "c")]);
        // A fully-ground atom (subject AND object constant) is out of v1 scope.
        let ground = LayerBgpPlan {
            atoms: vec![
                LayerBgpAtom {
                    subject: node_const("a"),
                    predicate_term: KNOWS.to_vec(),
                    object: node_const("b"),
                },
                LayerBgpAtom::all_variables(LIKES.to_vec()),
            ],
            joins: vec![JoinStep { acc_key: 0, rel_key: 0 }],
        };
        assert!(prove_layer_bgp(&lc, &ground).is_err());
        // A value literal in the SUBJECT position is rejected.
        let value_subject = LayerBgpPlan {
            atoms: vec![
                LayerBgpAtom {
                    subject: AtomTerm::ValueConstant(CanonicalDatatype::String, b"x".to_vec()),
                    predicate_term: KNOWS.to_vec(),
                    object: AtomTerm::Variable,
                },
                LayerBgpAtom::all_variables(LIKES.to_vec()),
            ],
            joins: vec![JoinStep { acc_key: 0, rel_key: 0 }],
        };
        assert!(prove_layer_bgp(&lc, &value_subject).is_err());
    }
}
