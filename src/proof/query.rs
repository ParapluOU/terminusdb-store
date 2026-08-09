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

use super::argument::{commit_column, Transcript254};
use super::bgp::{
    prove_bgp_chain_with_result, verify_bgp_chain, BgpChainProof, JoinStep, RelationCommit,
    RelationWitness,
};
use super::commitment::LayerCommitment;
use super::dict::{
    hash_object, hash_term, prove_dict_membership, verify_dict_membership, DictProof,
};
use super::filter::{
    atom_kept_columns, compact_selected, honest_witness, prove_atom_filter, prove_boolean_product,
    prove_compaction_only, prove_filter, prove_selection_only, verify_atom_filter,
    verify_boolean_product, verify_compaction_only, verify_filter, verify_selection_only,
    AtomFilterProof, AtomFilterSpec, BooleanProductProof, CompactionOnlyProof, FullFilterProof,
    Position, SelectionOnlyProof,
};
use super::invalid;
use super::join::{
    compute_self_join, prove_self_join_complete, self_join_commitments, self_join_domain,
    verify_self_join_complete, SelfJoinCommitments, SelfJoinCompleteProof,
};
use super::range::{
    object_range_sel_column, prove_object_range_select, verify_object_range_select,
    ObjectRangeProof, RangeOp,
};
use super::relocate::{prove_column_relocation, verify_column_relocation, RelocationProof};
use super::ProofHash;
use super::{CanonicalDatatype, CanonicalObject};

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

/// Extract the `(S, P, O)` `ArkFr` columns and real-row count `n` from a
/// complete logical view, in the exact deterministic order and padding that
/// produced `materialized_column_commitments`.
fn materialized_layer_columns(
    layer: &LayerCommitment,
) -> io::Result<(Vec<ArkFr>, Vec<ArkFr>, Vec<ArkFr>, usize)> {
    let n = layer.materialized.len();
    let padded = padded_len(n)?;
    let mut s = vec![f0(); padded];
    let mut p = vec![f0(); padded];
    let mut o = vec![f0(); padded];
    for (i, record) in layer.materialized.iter().enumerate() {
        s[i] = ArkFr::from_u64(record.subject_id);
        p[i] = ArkFr::from_u64(record.predicate_id);
        o[i] = ArkFr::from_u64(record.object_id);
    }
    Ok((s, p, o, n))
}

/// Resolve a predicate IRI (term bytes) to its dictionary id in this layer. A missing
/// term is authenticated only for a standalone ground query; relational atoms fail closed.
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

/// Prove `t(?s, <predicate_term>, ?o)` against a complete materialized head view. Resolves
/// the predicate IRI to its committed dictionary id, proves that binding (Q3
/// membership), then proves the filtered result for that id (Q1). Returns the
/// combined proof, the output commitments, the result length `m`, and the REVEALED
/// resolved id (authenticated by the membership proof — not secret).
pub fn prove_layer_filter(
    layer: &LayerCommitment,
    predicate_term: &[u8],
) -> io::Result<(CombinedFilterProof, ArkGT, ArkGT, usize, u64)> {
    let (s, p, o, n) = materialized_layer_columns(layer)?;
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
    let [com_s, com_p, com_o] = layer.materialized_column_commitments;
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
    let n = layer.state.materialized_count as usize;
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
    let [com_s, com_p, com_o] = layer.materialized_column_commitments;
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
    let (s, p, o, n) = materialized_layer_columns(layer)?;
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
    let [com_s, com_p, com_o] = layer.materialized_column_commitments;
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
    let jcom = self_join_commitments(sf, of, &x, &y, &z, &rho_l, &rho_r, &u, &w_l, &w_r)?;

    // (4) Relocate F from the table domain to the join domain (positional equality).
    let target_sf = pad(sf, n_join);
    let target_of = pad(of, n_join);
    let relocate_s =
        prove_column_relocation(&s_out, &com_sout, &target_sf, &jcom.sf, &mut transcript)?;
    let relocate_o =
        prove_column_relocation(&o_out, &com_oout, &target_of, &jcom.of, &mut transcript)?;

    // (5) The complete self-join argument over the relocated F.
    let join = prove_self_join_complete(
        sf,
        of,
        &x,
        &y,
        &z,
        &rho_l,
        &rho_r,
        &u,
        &w_l,
        &w_r,
        m_j,
        &jcom,
        &mut transcript,
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
    let n = layer.state.materialized_count as usize;
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
    let [com_s, com_p, com_o] = layer.materialized_column_commitments;
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

const LAYER_BGP_DOMAIN: &[u8] = b"parture/proof-of-woql/qk-layer-bgp-by-iri/v1";

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

/// Store ID namespace for an authenticated result column. Predicate dictionary IDs and
/// node/value dictionary IDs are independent even when their numeric field values match.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResultIdNamespace {
    NodeValue,
    Predicate,
}

/// Authoritative interpretation of one public result column. Nullability is a query-plan
/// property, never inferred from a delivered field value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResultColumnDescriptor {
    pub namespace: ResultIdNamespace,
    pub nullable: bool,
}

/// Canonical public result value. `Null` is distinct from every Store dictionary object
/// and is represented by field zero only in a nullable result column.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum CanonicalResultValue {
    Object(CanonicalObject),
    Null,
}

/// A numeric RANGE filter on an atom's VARIABLE object: `?o op c`, where the object's
/// dict-authenticated typed value (of datatype discriminant `T`) is compared to the
/// PUBLIC order-preserving-encoded constant `c`. `b_bits` is the range gadget's
/// bit-width (`64` for the store's `<= 8`-byte numerics — see [`super::range`]). Bound
/// into the layer BGP statement so the query is fully pinned.
#[derive(Clone, Debug)]
pub struct ObjectRangeBound {
    pub op: RangeOp,
    pub c: ArkFr,
}

#[derive(Clone, Debug)]
pub struct ObjectRangeSpec {
    pub op: RangeOp,
    /// The datatype discriminant `T` the object must match (a `CanonicalDatatype as u64`
    /// tag, or the Node tag), as a field element.
    pub datatype: ArkFr,
    /// The order-preserving-encoded comparison constant `c`, as a field element.
    pub c: ArkFr,
    pub b_bits: usize,
    /// Optional second bound over the same datatype and bit-width. This represents an
    /// intersection, e.g. `V >= low AND V < high` for WOQL's half-open ranges.
    pub second: Option<ObjectRangeBound>,
}

/// One atom of a layer BGP. A constant predicate IRI is authenticated by the layer's
/// predicate dictionary; a predicate variable retains the root-bound materialized P
/// column directly. The subject/object may each be a
/// VARIABLE (a kept output column) or a CONSTANT (authenticated against the layer's
/// node/value dictionary and folded into the atom filter). A fully-ground atom is
/// represented internally by the filter gadget's authenticated UNIT column and is
/// accepted alone or in an all-ground conjunction. An optional `object_range`
/// adds a numeric RANGE filter on the object — valid ONLY when the object is a
/// `Variable` (a ranged object is a kept variable, never a constant). Constraints
/// (enforced at prove AND verify): the subject may not be a `ValueConstant` (literals
/// never occupy the subject position), and an `object_range` requires a `Variable`
/// object with a valid bit-width.
#[derive(Clone, Debug)]
pub struct LayerBgpAtom {
    pub subject: AtomTerm,
    /// Empty for a predicate variable; otherwise the constant predicate IRI.
    pub predicate_term: Vec<u8>,
    pub predicate_variable: bool,
    pub object: AtomTerm,
    pub object_range: Option<ObjectRangeSpec>,
    /// Require the subject and object variables to be equal within each selected row.
    pub subject_object_equal: bool,
}

impl LayerBgpAtom {
    /// The classic all-variable atom `t(?s, <predicate_term>, ?o)`: both subject and
    /// object are query variables (the two kept output columns), no object range.
    pub fn all_variables(predicate_term: Vec<u8>) -> Self {
        Self {
            subject: AtomTerm::Variable,
            predicate_term,
            predicate_variable: false,
            object: AtomTerm::Variable,
            object_range: None,
            subject_object_equal: false,
        }
    }

    /// Attach a numeric object-value range filter to this atom (the object must be a
    /// `Variable`; enforced at prove/verify by [`validate_atom`]).
    pub fn with_object_range(mut self, range: ObjectRangeSpec) -> Self {
        self.object_range = Some(range);
        self
    }
}

/// The additional authenticated selection and table-domain intersection used when an
/// object range has a second bound.
#[derive(Clone, Debug)]
pub struct SecondRangeSubproof {
    pub object_range: ObjectRangeProof,
    pub com_range_sel_at_d: ArkGT,
    pub relocate_range_sel: RelocationProof,
    pub com_range_sel_at_table: ArkGT,
    pub intersect: BooleanProductProof,
    pub com_intersection_sel: ArkGT,
}

/// A layer BGP plan: a relational plan has `k` atoms and `k - 1` left-deep join steps;
/// an all-ground zero-arity conjunction has no keyed joins. `joins[j]` names
/// the first shared variable's column on the evolving accumulator (`acc_key`) and on
/// atom `j + 1`'s relation (`rel_key`), plus any further shared key pairs. Each atom
/// relation has ONE column per distinct VARIABLE (in S,P,O order).
#[derive(Clone, Debug)]
pub struct LayerBgpPlan {
    pub atoms: Vec<LayerBgpAtom>,
    pub joins: Vec<JoinStep>,
}

/// The composed sub-proofs for a RANGED atom `t(?s, <p>, ?a) [?a op c]`: the object
/// column and the three node/value dictionary columns relocated to a common
/// power-of-two domain `D = 2^max(a, dict_log)` (each tied to its root-bound original),
/// the authenticated object-range selection at `D`, the range selection relocated back
/// to the table domain, the constant-equality selection, their product (the combined
/// selection), and the compaction of the kept columns by that combined selection.
/// `d_log`/`n_dict` are re-derived by the verifier from the (root-verified) layer, so
/// they are NOT carried here; every commitment below is authenticated against a
/// root-bound source (the relocations) or the shared transcript.
#[derive(Clone, Debug)]
pub struct RangeAtomSubproof {
    /// Object column (table -> D) + the three dict columns (dict -> D) relocations.
    pub relocate_object: RelocationProof,
    pub relocate_dict_id: RelocationProof,
    pub relocate_dict_value: RelocationProof,
    pub relocate_dict_datatype: RelocationProof,
    /// The relocated copies at domain `D` (targets; sources are root-bound).
    pub com_object_at_d: ArkGT,
    pub com_dict_id_at_d: ArkGT,
    pub com_dict_value_at_d: ArkGT,
    pub com_dict_datatype_at_d: ArkGT,
    /// The authenticated object-range selection at `D` and its `Com(range_sel)`.
    pub object_range: ObjectRangeProof,
    pub com_range_sel_at_d: ArkGT,
    /// `range_sel` relocated from `D` back to the table domain, and its commitment.
    pub relocate_range_sel: RelocationProof,
    pub com_range_sel_at_table: ArkGT,
    /// Present exactly when the public range spec carries a second bound.
    pub second: Option<SecondRangeSubproof>,
    /// The constant-equality selection (predicate + any subject constant) and `Com(sel)`.
    pub equality: SelectionOnlyProof,
    pub com_equality_sel: ArkGT,
    /// The product `combined_sel = equality_sel · range_sel` and its commitment.
    pub combine: BooleanProductProof,
    pub com_combined_sel: ArkGT,
    /// The compaction of the kept columns by `combined_sel` (kept commitments live on the
    /// enclosing [`LayerBgpAtomProof`]).
    pub compaction: CompactionOnlyProof,
}

/// The per-atom half of a layer BGP proof: an optional dictionary binding for a constant
/// predicate IRI, optional subject/object CONSTANT bindings (node/value dictionary), the
/// per-atom selection (EITHER a plain constant `filter` OR a `range` subproof, matching
/// the plan atom), one commitment per kept column (in S,P,O order), the result length, and
/// the REVEALED subject/object constant ids. Those ids are public inputs to verification:
/// the verifier feeds the SAME id to both the membership check (constant -> id) and the
/// filter's constant, so a prover cannot authenticate one id and filter on another.
/// Exactly one of `filter`/`range` is `Some`, determined by the PLAN atom's
/// `object_range` (never by the prover-supplied proof).
#[derive(Clone, Debug)]
pub struct LayerBgpAtomProof {
    pub predicate_membership: Option<DictProof>,
    pub subject_membership: Option<DictProof>,
    pub object_membership: Option<DictProof>,
    pub filter: Option<AtomFilterProof>,
    pub range: Option<RangeAtomSubproof>,
    pub kept_commitments: Vec<ArkGT>,
    pub m: usize,
    pub subject_id: Option<u64>,
    pub object_id: Option<u64>,
}

/// A complete k-atom layer BGP proof: one per-atom (membership + filter) proof, plus the
/// left-deep binary-join chain over the filtered relations when `k >= 2`, all under one
/// transcript. A single-atom query needs no join proof: its complete filtered relation is
/// already the complete query result.
#[derive(Clone, Debug)]
pub struct LayerBgpProof {
    pub atoms: Vec<LayerBgpAtomProof>,
    pub chain: Option<BgpChainProof>,
    /// A single fully-ground query whose result is empty because this term is absent
    /// from a root-authenticated dictionary. No row-filter proof is needed: a triple
    /// containing a dictionary-absent term cannot exist.
    pub ground_absence: Option<GroundAbsence>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroundAbsence {
    Predicate,
    Subject,
    Object,
}

/// Prover-side BGP output including the values bound by the result commitments. Columns
/// are padded to `2^result_log_domain`; query rows occupy `[0, result_len)`.
pub struct LayerBgpResult {
    pub proof: LayerBgpProof,
    pub result_commitments: Vec<ArkGT>,
    pub result_columns: Vec<Vec<ArkFr>>,
    pub result_log_domain: usize,
    pub result_len: usize,
    pub predicate_ids: Vec<u64>,
}

impl LayerBgpResult {
    /// Copy the real result prefix, omitting commitment-domain padding.
    pub fn real_columns(&self) -> Vec<Vec<ArkFr>> {
        self.result_columns
            .iter()
            .map(|column| column[..self.result_len].to_vec())
            .collect()
    }
}

/// Bind caller-delivered result ID columns to the commitments authenticated by a query
/// proof. This is intentionally deterministic: pad each real column to the proven domain
/// and recompute its Dory commitment. It adds work linear in the delivered result size,
/// but no new proof-generation lifecycle or trusted component.
pub fn verify_result_columns(
    commitments: &[ArkGT],
    columns: &[Vec<ArkFr>],
    result_len: usize,
    result_log_domain: usize,
) -> io::Result<()> {
    if result_log_domain == 0 || result_log_domain >= usize::BITS as usize {
        return Err(invalid("result column log-domain out of range"));
    }
    let domain = 1usize << result_log_domain;
    if result_len > domain || commitments.len() != columns.len() {
        return Err(invalid(
            "result column shape does not match its commitments",
        ));
    }
    for (commitment, column) in commitments.iter().zip(columns) {
        if column.len() != result_len {
            return Err(invalid(
                "result column length does not match the proven length",
            ));
        }
        let mut padded = column.clone();
        padded.resize(domain, ArkFr::zero());
        if commit_column(&padded)? != *commitment {
            return Err(invalid(
                "delivered result column does not match its commitment",
            ));
        }
    }
    Ok(())
}

/// Resolve proven Store ID columns through the layer's authenticated node/value
/// dictionary. The clear catalog in [`LayerCommitment`] is a verifier-side opening of
/// the root-bound dictionary commitments, so it is not trusted until both its `(hash,
/// id)` columns have been rebuilt and recommitted here.
pub fn resolve_node_value_result_objects(
    layer: &LayerCommitment,
    expected_root: ProofHash,
    columns: &[Vec<ArkFr>],
) -> io::Result<Vec<Vec<CanonicalObject>>> {
    if layer.state.commitment_root != expected_root {
        return Err(invalid(
            "layer commitment_root does not match the externally-trusted expected root",
        ));
    }
    layer.validate()?;

    let (hashes, ids) = layer.node_value_dictionary_columns()?;
    if commit_column(&hashes)? != layer.node_value_dict_commitments.0
        || commit_column(&ids)? != layer.node_value_dict_commitments.1
    {
        return Err(invalid(
            "node/value catalog does not open the root-bound dictionary commitments",
        ));
    }

    columns
        .iter()
        .map(|column| {
            column
                .iter()
                .map(|id| {
                    layer
                        .node_value_catalog
                        .iter()
                        .find(|record| ArkFr::from_u64(record.object_id) == *id)
                        .map(|record| record.triple.object.clone())
                        .ok_or_else(|| {
                            invalid("result id is absent from the node/value dictionary")
                        })
                })
                .collect()
        })
        .collect()
}

/// Resolve result columns through their explicitly declared, plan-derived Store ID
/// namespaces. Predicate IDs are opened only against the root-bound predicate catalog;
/// subject/object IDs only against the node/value catalog.
pub fn resolve_typed_result_objects(
    layer: &LayerCommitment,
    expected_root: ProofHash,
    columns: &[Vec<ArkFr>],
    namespaces: &[ResultIdNamespace],
) -> io::Result<Vec<Vec<CanonicalObject>>> {
    if columns.len() != namespaces.len() {
        return Err(invalid("result column namespace shape mismatch"));
    }
    if layer.state.commitment_root != expected_root {
        return Err(invalid(
            "layer commitment_root does not match the externally-trusted expected root",
        ));
    }
    layer.validate()?;

    columns
        .iter()
        .zip(namespaces)
        .map(|(column, namespace)| {
            column
                .iter()
                .map(|id| match namespace {
                    ResultIdNamespace::NodeValue => layer
                        .node_value_catalog
                        .iter()
                        .find(|record| ArkFr::from_u64(record.object_id) == *id)
                        .map(|record| record.triple.object.clone())
                        .ok_or_else(|| invalid("result id is absent from the node/value dictionary")),
                    ResultIdNamespace::Predicate => layer
                        .predicate_catalog
                        .iter()
                        .find(|record| ArkFr::from_u64(record.predicate_id) == *id)
                        .map(|record| CanonicalObject::Node(record.triple.predicate.clone()))
                        .ok_or_else(|| invalid("result id is absent from the predicate dictionary")),
                })
                .collect()
        })
        .collect()
}

/// Resolve nullable typed result columns. Every real Store dictionary ID must be
/// nonzero. Zero resolves to `Null` exactly for a descriptor marked nullable; it is an
/// error in a nonnullable column and is never looked up in either dictionary catalog.
pub fn resolve_result_values(
    layer: &LayerCommitment,
    expected_root: ProofHash,
    columns: &[Vec<ArkFr>],
    descriptors: &[ResultColumnDescriptor],
) -> io::Result<Vec<Vec<CanonicalResultValue>>> {
    if columns.len() != descriptors.len() {
        return Err(invalid("result column descriptor shape mismatch"));
    }
    if layer.state.commitment_root != expected_root {
        return Err(invalid(
            "layer commitment_root does not match the externally-trusted expected root",
        ));
    }
    layer.validate()?;
    if layer
        .predicate_catalog
        .iter()
        .any(|record| record.predicate_id == 0)
        || layer
            .node_value_catalog
            .iter()
            .any(|record| record.object_id == 0)
    {
        return Err(invalid(
            "Store dictionary id zero is reserved for nullable query results",
        ));
    }

    columns
        .iter()
        .zip(descriptors)
        .map(|(column, descriptor)| {
            column
                .iter()
                .map(|id| {
                    if id.is_zero() {
                        return if descriptor.nullable {
                            Ok(CanonicalResultValue::Null)
                        } else {
                            Err(invalid("null id in a nonnullable result column"))
                        };
                    }
                    let object = match descriptor.namespace {
                        ResultIdNamespace::NodeValue => layer
                            .node_value_catalog
                            .iter()
                            .find(|record| ArkFr::from_u64(record.object_id) == *id)
                            .map(|record| record.triple.object.clone())
                            .ok_or_else(|| {
                                invalid("result id is absent from the node/value dictionary")
                            })?,
                        ResultIdNamespace::Predicate => layer
                            .predicate_catalog
                            .iter()
                            .find(|record| ArkFr::from_u64(record.predicate_id) == *id)
                            .map(|record| CanonicalObject::Node(record.triple.predicate.clone()))
                            .ok_or_else(|| {
                                invalid("result id is absent from the predicate dictionary")
                            })?,
                    };
                    Ok(CanonicalResultValue::Object(object))
                })
                .collect()
        })
        .collect()
}

pub fn result_column_descriptors(
    plan: &LayerBgpPlan,
) -> io::Result<Vec<ResultColumnDescriptor>> {
    Ok(result_id_namespaces(plan)?
        .into_iter()
        .map(|namespace| ResultColumnDescriptor {
            namespace,
            nullable: false,
        })
        .collect())
}

/// Derive the namespace of every final result column while validating that each natural
/// join compares IDs from the same Store dictionary namespace.
pub fn result_id_namespaces(plan: &LayerBgpPlan) -> io::Result<Vec<ResultIdNamespace>> {
    fn atom_namespaces(atom: &LayerBgpAtom) -> Vec<ResultIdNamespace> {
        let mut out = Vec::new();
        if matches!(atom.subject, AtomTerm::Variable) {
            out.push(ResultIdNamespace::NodeValue);
        }
        if atom.predicate_variable {
            out.push(ResultIdNamespace::Predicate);
        }
        if matches!(atom.object, AtomTerm::Variable) && !atom.subject_object_equal {
            out.push(ResultIdNamespace::NodeValue);
        }
        out
    }

    let relational = relational_atom_count(plan);
    if relational == 0 {
        return Ok(Vec::new());
    }
    let mut acc = atom_namespaces(&plan.atoms[0]);
    for (step, atom) in plan.joins.iter().zip(&plan.atoms[1..relational]) {
        let rel = atom_namespaces(atom);
        if step.is_cross_product() {
            acc.extend(rel);
            continue;
        }
        let mut pairs = vec![(step.acc_key, step.rel_key)];
        pairs.extend(step.additional_keys.iter().copied());
        for (left, right) in &pairs {
            if *left >= acc.len() || *right >= rel.len() {
                return Err(invalid("BGP join namespace column index out of range"));
            }
            if acc[*left] != rel[*right] {
                return Err(invalid(
                    "BGP join compares predicate and node/value ID namespaces",
                ));
            }
        }
        if step.additional_keys.is_empty() {
            let key = acc.remove(step.acc_key);
            let mut rel_other = rel;
            rel_other.remove(step.rel_key);
            acc.push(key);
            acc.extend(rel_other);
        } else {
            for (index, namespace) in rel.into_iter().enumerate() {
                if !pairs.iter().any(|(_, right)| *right == index) {
                    acc.push(namespace);
                }
            }
        }
    }
    Ok(acc)
}

/// Verify caller-delivered canonical IRIs/typed literals against proven Store ID
/// columns and the root-authenticated node/value dictionary.
pub fn verify_node_value_result_objects(
    layer: &LayerCommitment,
    expected_root: ProofHash,
    columns: &[Vec<ArkFr>],
    objects: &[Vec<CanonicalObject>],
) -> io::Result<()> {
    let resolved = resolve_node_value_result_objects(layer, expected_root, columns)?;
    if resolved != objects {
        return Err(invalid(
            "delivered result objects do not match the authenticated Store IDs",
        ));
    }
    Ok(())
}

/// The resolved dictionary ids for one atom's constants. Each id is `Some` exactly when
/// that position is a constant.
#[derive(Clone, Copy, Debug)]
struct AtomIds {
    predicate_id: Option<u64>,
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
/// fail closed if the constant is absent (except for the standalone-ground shortcut).
/// Subjects share the node/value id space with node objects, so
/// the same `(hash_object(term), id)` row authenticates the constant in either position.
fn resolve_node_value_id(layer: &LayerCommitment, object: &CanonicalObject) -> io::Result<u64> {
    layer
        .node_value_catalog
        .iter()
        .find(|record| &record.triple.object == object)
        .map(|record| record.object_id)
        .ok_or_else(|| invalid("constant not in layer node/value dictionary"))
}

/// Enforce the atom constraints: the subject is a variable or a node constant (never a
/// value literal). Fully-ground atoms are valid here; [`validate_bgp_plan`] allows them
/// only when every atom is ground, because mixed boolean/relation composition is separate.
fn validate_atom(atom: &LayerBgpAtom) -> io::Result<()> {
    if atom.predicate_variable != atom.predicate_term.is_empty() {
        return Err(invalid(
            "predicate variable must have an empty term and predicate constant must not",
        ));
    }
    if matches!(atom.subject, AtomTerm::ValueConstant(..)) {
        return Err(invalid(
            "layer BGP subject may not be a value-literal constant",
        ));
    }
    if let Some(range) = &atom.object_range {
        // A ranged object must be a KEPT variable — never a constant (a range on a
        // fixed object is a compile-time-true/false, out of v1 scope).
        if !matches!(atom.object, AtomTerm::Variable) {
            return Err(invalid("layer BGP object range requires a variable object"));
        }
        if !(1..=252).contains(&range.b_bits) {
            return Err(invalid(
                "layer BGP object range bit-width out of range (need 1 <= B <= 252)",
            ));
        }
    }
    if atom.subject_object_equal
        && (!matches!(atom.subject, AtomTerm::Variable)
            || !matches!(atom.object, AtomTerm::Variable))
    {
        return Err(invalid(
            "subject/object equality requires two variable positions",
        ));
    }
    Ok(())
}

/// The `(constant/id)` presence carried in a proof MUST match the atom's constant
/// structure: a variable position has no id, a constant position has one. Rejects a
/// malformed proof that, e.g., claims a subject id for a variable subject (which would
/// otherwise silently reshape the filter spec).
fn check_ids_consistent(atom: &LayerBgpAtom, ids: &AtomIds) -> io::Result<()> {
    if atom.predicate_variable != ids.predicate_id.is_none() {
        return Err(invalid(
            "layer BGP predicate constant/id presence does not match the plan",
        ));
    }
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

/// The atom's [`AtomFilterSpec`] from its resolved ids. Constant positions carry their
/// authenticated IDs; variable positions are retained in S,P,O order. The constant list
/// order (P, then S, then O) is fixed and identical on both sides.
fn atom_filter_spec(atom: &LayerBgpAtom, ids: &AtomIds) -> AtomFilterSpec {
    let mut constants = Vec::new();
    if let Some(predicate_id) = ids.predicate_id {
        constants.push((Position::P, ArkFr::from_u64(predicate_id)));
    }
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
    if atom.predicate_variable {
        kept.push(Position::P);
    }
    if matches!(atom.object, AtomTerm::Variable) {
        kept.push(Position::O);
    }
    let equalities = if atom.subject_object_equal {
        kept.retain(|position| *position != Position::O);
        vec![(Position::S, Position::O)]
    } else {
        Vec::new()
    };
    AtomFilterSpec {
        constants,
        kept,
        equalities,
    }
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
/// every atom's subject term, predicate variant/term/optional id, and object term (each
/// constant carrying its revealed id), then the join plan. Identical on both sides.
fn bind_layer_bgp_statement(
    transcript: &mut Transcript254,
    layer: &LayerCommitment,
    plan: &LayerBgpPlan,
    atom_ids: &[AtomIds],
) {
    transcript.append_bytes(
        b"pow/bgp-layer/commitment-root",
        &layer.state.commitment_root,
    );
    transcript.append_bytes(b"pow/bgp-layer/k", &(plan.atoms.len() as u64).to_le_bytes());
    for (atom, ids) in plan.atoms.iter().zip(atom_ids.iter()) {
        bind_atom_term(transcript, &atom.subject, ids.subject_id);
        transcript.append_bytes(
            b"pow/bgp-layer/pred-variable",
            &[u8::from(atom.predicate_variable)],
        );
        transcript.append_bytes(b"pow/bgp-layer/pred-term", &atom.predicate_term);
        if let Some(predicate_id) = ids.predicate_id {
            transcript.append_bytes(b"pow/bgp-layer/pred-id", &predicate_id.to_le_bytes());
        }
        bind_atom_term(transcript, &atom.object, ids.object_id);
        transcript.append_bytes(
            b"pow/bgp-layer/subject-object-equal",
            &[u8::from(atom.subject_object_equal)],
        );
        // Pin the optional object range (op, datatype T, constant c, bit-width) so the
        // whole comparison is part of the statement the challenges depend on.
        match &atom.object_range {
            None => transcript.append_bytes(b"pow/bgp-layer/obj-range-none", &[]),
            Some(range) => {
                let op_tag: u8 = match range.op {
                    RangeOp::Lt => 0,
                    RangeOp::Le => 1,
                    RangeOp::Gt => 2,
                    RangeOp::Ge => 3,
                };
                transcript.append_bytes(b"pow/bgp-layer/obj-range-op", &[op_tag]);
                transcript.append_field(b"pow/bgp-layer/obj-range-dt", &range.datatype);
                transcript.append_field(b"pow/bgp-layer/obj-range-c", &range.c);
                transcript.append_bytes(
                    b"pow/bgp-layer/obj-range-bbits",
                    &(range.b_bits as u64).to_le_bytes(),
                );
                match &range.second {
                    None => transcript.append_bytes(b"pow/bgp-layer/obj-range-second-none", &[]),
                    Some(second) => {
                        let op_tag: u8 = match second.op {
                            RangeOp::Lt => 0,
                            RangeOp::Le => 1,
                            RangeOp::Gt => 2,
                            RangeOp::Ge => 3,
                        };
                        transcript.append_bytes(b"pow/bgp-layer/obj-range-second-op", &[op_tag]);
                        transcript.append_field(b"pow/bgp-layer/obj-range-second-c", &second.c);
                    }
                }
            }
        }
    }
    for step in &plan.joins {
        transcript.append_bytes(
            b"pow/bgp-layer/acc-key",
            &(step.acc_key as u64).to_le_bytes(),
        );
        transcript.append_bytes(
            b"pow/bgp-layer/rel-key",
            &(step.rel_key as u64).to_le_bytes(),
        );
        transcript.append_bytes(
            b"pow/bgp-layer/additional-key-count",
            &(step.additional_keys.len() as u64).to_le_bytes(),
        );
        for (acc_key, rel_key) in &step.additional_keys {
            transcript.append_bytes(
                b"pow/bgp-layer/additional-acc-key",
                &(*acc_key as u64).to_le_bytes(),
            );
            transcript.append_bytes(
                b"pow/bgp-layer/additional-rel-key",
                &(*rel_key as u64).to_le_bytes(),
            );
        }
    }
}

fn validate_bgp_plan(plan: &LayerBgpPlan) -> io::Result<usize> {
    let k = plan.atoms.len();
    if k == 0 {
        return Err(invalid("layer BGP needs at least one atom"));
    }
    let ground_count = plan
        .atoms
        .iter()
        .filter(|atom| {
            !atom.predicate_variable
                && !matches!(atom.subject, AtomTerm::Variable)
                && !matches!(atom.object, AtomTerm::Variable)
        })
        .count();
    if ground_count == k {
        if !plan.joins.is_empty() {
            return Err(invalid("all-ground BGP must not carry keyed join steps"));
        }
    } else {
        let relational_count = k - ground_count;
        if plan.atoms[..relational_count].iter().any(atom_is_ground)
            || plan.atoms[relational_count..]
                .iter()
                .any(|atom| !atom_is_ground(atom))
        {
            return Err(invalid(
                "mixed BGP must place every ground atom after relational atoms",
            ));
        }
        if plan.joins.len() != relational_count - 1 {
            return Err(invalid(
                "relational BGP prefix must have exactly k-1 join steps",
            ));
        }
    }
    result_id_namespaces(plan)?;
    Ok(k)
}

fn atom_is_ground(atom: &LayerBgpAtom) -> bool {
    !atom.predicate_variable
        && !matches!(atom.subject, AtomTerm::Variable)
        && !matches!(atom.object, AtomTerm::Variable)
}

fn relational_atom_count(plan: &LayerBgpPlan) -> usize {
    plan.atoms.iter().take_while(|atom| !atom_is_ground(atom)).count()
}

fn is_all_ground(plan: &LayerBgpPlan) -> bool {
    !plan.atoms.is_empty() && relational_atom_count(plan) == 0
}

/// Whether the plan's sole result column is the filter gadget's internal UNIT column.
/// It authenticates a zero-arity relation's cardinality but is not a Store dictionary ID.
pub(crate) fn has_ground_result_unit(plan: &LayerBgpPlan) -> bool {
    !plan.atoms.is_empty() && is_all_ground(plan)
}

fn ground_dictionary_absence(
    layer: &LayerCommitment,
    plan: &LayerBgpPlan,
) -> Option<GroundAbsence> {
    let [atom] = plan.atoms.as_slice() else {
        return None;
    };
    if matches!(atom.subject, AtomTerm::Variable) || matches!(atom.object, AtomTerm::Variable) {
        return None;
    }
    if !atom.predicate_variable
        && !layer
        .predicate_catalog
        .iter()
        .any(|record| record.triple.predicate == atom.predicate_term)
    {
        return Some(GroundAbsence::Predicate);
    }
    let subject = atom_term_object(&atom.subject).expect("ground subject must be constant");
    if !layer
        .node_value_catalog
        .iter()
        .any(|record| record.triple.object == subject)
    {
        return Some(GroundAbsence::Subject);
    }
    let object = atom_term_object(&atom.object).expect("ground object must be constant");
    if !layer
        .node_value_catalog
        .iter()
        .any(|record| record.triple.object == object)
    {
        return Some(GroundAbsence::Object);
    }
    None
}

/// The padded power-of-two length of every committed node/value dictionary column,
/// derived from the (root-verified) real dict-row count `n_dict` — the SAME
/// `next_pow2(n_dict).max(2)` convention `commit_node_value_dict*` uses. A ranged atom
/// needs a non-empty dictionary (an empty dict cannot authenticate any object value).
fn dict_domain_len(n_dict: usize) -> io::Result<usize> {
    if n_dict == 0 {
        return Err(invalid(
            "layer BGP object range requires a non-empty node/value dictionary",
        ));
    }
    let padded = n_dict
        .checked_next_power_of_two()
        .ok_or_else(|| invalid("node/value dictionary length overflow"))?
        .max(2);
    Ok(padded)
}

/// The full (padded) base column for a kept position — a subset of `{S, O}` for a ranged
/// atom (a ranged object is a variable).
fn kept_base_column(pos: Position, s: &[ArkFr], p: &[ArkFr], o: &[ArkFr]) -> Vec<ArkFr> {
    match pos {
        Position::S => s.to_vec(),
        Position::P => p.to_vec(),
        Position::O => o.to_vec(),
    }
}

/// The root-bound commitment for a kept position (mirrors [`kept_base_column`]).
fn kept_base_commitment(pos: Position, com_s: &ArkGT, com_p: &ArkGT, com_o: &ArkGT) -> ArkGT {
    match pos {
        Position::S => *com_s,
        Position::P => *com_p,
        Position::O => *com_o,
    }
}

/// The common relocation domain for a ranged atom: `D = 2^max(a, dict_log)`, large
/// enough to hold BOTH the table object column (`2^a`) and the dict columns
/// (`2^dict_log`), so both can be relocated in and the range gadget run over one domain.
fn range_common_domain(a: usize, n_dict: usize) -> io::Result<(usize, usize)> {
    let dict_len = dict_domain_len(n_dict)?;
    let dict_log = dict_len.trailing_zeros() as usize;
    Ok((a.max(dict_log), dict_log))
}

/// The root-bound count of rows in the complete node/value dictionary committed at
/// this head. The count and the full catalog are both authenticated by the head root.
fn root_bound_n_dict(layer: &LayerCommitment) -> io::Result<usize> {
    let n_dict = layer.state.node_value_count as usize;
    if n_dict != layer.node_value_catalog.len() {
        return Err(invalid(
            "root-bound node/value dictionary count does not match the committed catalog length",
        ));
    }
    Ok(n_dict)
}

/// Prove one RANGED atom `t(?s, <p>, ?a) [?a op c]`: relocate the object column + the
/// three dict columns to the common domain `D`, prove the authenticated object-range
/// selection at `D`, relocate that `range_sel` back to the table domain, prove the
/// constant-equality selection, multiply them into the combined selection, and compact
/// the kept columns by it. Returns the range subproof, the kept-column commitments (S,O
/// order), the dense kept columns (for the chain witness), and the result length `m`.
/// All sub-proofs run on the SHARED `transcript` in this exact order.
#[allow(clippy::too_many_arguments)]
fn prove_ranged_atom(
    spec: &AtomFilterSpec,
    range: &ObjectRangeSpec,
    s: &[ArkFr],
    p: &[ArkFr],
    o: &[ArkFr],
    n: usize,
    a: usize,
    com_s: &ArkGT,
    com_p: &ArkGT,
    com_o: &ArkGT,
    layer: &LayerCommitment,
    transcript: &mut Transcript254,
) -> io::Result<(RangeAtomSubproof, Vec<ArkGT>, Vec<Vec<ArkFr>>, usize)> {
    let n_dict = root_bound_n_dict(layer)?;
    let (d_log, dict_log) = range_common_domain(a, n_dict)?;
    let d = 1usize << d_log;

    // Root-bound dict columns/commitments (format v3).
    let (_nv_hash, dict_id) = layer.node_value_dictionary_columns()?;
    let dict_value = layer.node_value_dictionary_value_column()?;
    let dict_datatype = layer.node_value_dictionary_datatype_column()?;
    let com_dict_id = layer.node_value_dict_commitments.1;
    let com_dict_value = layer.node_value_dict_range_commitments.0;
    let com_dict_datatype = layer.node_value_dict_range_commitments.1;
    if dict_id.len() != (1usize << dict_log) {
        return Err(invalid("node/value dictionary column length inconsistent"));
    }

    // (a) Relocate the object column (table -> D) and the three dict columns (dict -> D),
    // each tied to its root-bound original. When a domain already equals D the relocation
    // is same-domain (still authenticates the fresh copy).
    let o_at_d = pad(o, d);
    let com_object_at_d = commit_column(&o_at_d)?;
    let relocate_object = prove_column_relocation(o, com_o, &o_at_d, &com_object_at_d, transcript)?;

    let dict_id_at_d = pad(&dict_id, d);
    let com_dict_id_at_d = commit_column(&dict_id_at_d)?;
    let relocate_dict_id = prove_column_relocation(
        &dict_id,
        &com_dict_id,
        &dict_id_at_d,
        &com_dict_id_at_d,
        transcript,
    )?;

    let dict_value_at_d = pad(&dict_value, d);
    let com_dict_value_at_d = commit_column(&dict_value_at_d)?;
    let relocate_dict_value = prove_column_relocation(
        &dict_value,
        &com_dict_value,
        &dict_value_at_d,
        &com_dict_value_at_d,
        transcript,
    )?;

    let dict_datatype_at_d = pad(&dict_datatype, d);
    let com_dict_datatype_at_d = commit_column(&dict_datatype_at_d)?;
    let relocate_dict_datatype = prove_column_relocation(
        &dict_datatype,
        &com_dict_datatype,
        &dict_datatype_at_d,
        &com_dict_datatype_at_d,
        transcript,
    )?;

    // (b) Authenticated object-range selection over the relocated columns at D.
    let (object_range, com_range_sel_at_d) = prove_object_range_select(
        &o_at_d,
        &com_object_at_d,
        &dict_id_at_d,
        &dict_value_at_d,
        &dict_datatype_at_d,
        &com_dict_id_at_d,
        &com_dict_value_at_d,
        &com_dict_datatype_at_d,
        n_dict,
        range.datatype,
        range.op,
        range.c,
        range.b_bits,
        n,
        transcript,
    )?;
    // The SAME sel column that Com(range_sel) commits (χ_n-masked, so zero beyond n).
    let range_sel_at_d = object_range_sel_column(
        &o_at_d,
        &dict_id_at_d,
        &dict_value_at_d,
        &dict_datatype_at_d,
        n_dict,
        range.datatype,
        range.op,
        range.c,
        range.b_bits,
        n,
    );

    // (c) Relocate range_sel from D back to the table domain (its support is [0, n) ⊆
    // [0, 2^a), so truncating to the table domain is lossless; the relocation ties the
    // table copy to the authoritative Com(range_sel) at D).
    let table = 1usize << a;
    let range_sel_at_table = range_sel_at_d[..table].to_vec();
    let com_range_sel_at_table = commit_column(&range_sel_at_table)?;
    let relocate_range_sel = prove_column_relocation(
        &range_sel_at_d,
        &com_range_sel_at_d,
        &range_sel_at_table,
        &com_range_sel_at_table,
        transcript,
    )?;

    // (c2) A second public bound reuses the already authenticated/relocated object and
    // dictionary columns. Prove its selection independently, relocate it to the table
    // domain, then authenticate the intersection of both masks.
    let (effective_range_sel, com_effective_range_sel, second) =
        if let Some(second_bound) = &range.second {
            let (second_object_range, second_com_at_d) = prove_object_range_select(
                &o_at_d,
                &com_object_at_d,
                &dict_id_at_d,
                &dict_value_at_d,
                &dict_datatype_at_d,
                &com_dict_id_at_d,
                &com_dict_value_at_d,
                &com_dict_datatype_at_d,
                n_dict,
                range.datatype,
                second_bound.op,
                second_bound.c,
                range.b_bits,
                n,
                transcript,
            )?;
            let second_sel_at_d = object_range_sel_column(
                &o_at_d,
                &dict_id_at_d,
                &dict_value_at_d,
                &dict_datatype_at_d,
                n_dict,
                range.datatype,
                second_bound.op,
                second_bound.c,
                range.b_bits,
                n,
            );
            let second_sel_at_table = second_sel_at_d[..table].to_vec();
            let second_com_at_table = commit_column(&second_sel_at_table)?;
            let second_relocate = prove_column_relocation(
                &second_sel_at_d,
                &second_com_at_d,
                &second_sel_at_table,
                &second_com_at_table,
                transcript,
            )?;
            let (intersect, com_intersection_sel) = prove_boolean_product(
                &range_sel_at_table,
                &second_sel_at_table,
                &com_range_sel_at_table,
                &second_com_at_table,
                transcript,
            )?;
            let intersection: Vec<ArkFr> = range_sel_at_table
                .iter()
                .zip(second_sel_at_table.iter())
                .map(|(x, y)| *x * *y)
                .collect();
            (
                intersection,
                com_intersection_sel,
                Some(SecondRangeSubproof {
                    object_range: second_object_range,
                    com_range_sel_at_d: second_com_at_d,
                    relocate_range_sel: second_relocate,
                    com_range_sel_at_table: second_com_at_table,
                    intersect,
                    com_intersection_sel,
                }),
            )
        } else {
            (range_sel_at_table.clone(), com_range_sel_at_table, None)
        };

    // (d) Constant-equality selection (predicate + any subject constant) at the table
    // domain — pins com_equality_sel to the equality indicator (never touches the object).
    let (equality, com_equality_sel, eq_sel) =
        prove_selection_only(spec, s, p, o, n, com_s, com_p, com_o, transcript)?;

    // (e) Combined selection = equality_sel ⊙ range_sel (both boolean + χ_n-masked).
    let (combine, com_combined_sel) = prove_boolean_product(
        &eq_sel,
        &effective_range_sel,
        &com_equality_sel,
        &com_effective_range_sel,
        transcript,
    )?;
    let combined_sel: Vec<ArkFr> = eq_sel
        .iter()
        .zip(effective_range_sel.iter())
        .map(|(x, y)| *x * *y)
        .collect();

    // (f) Compact the kept variable columns by the combined selection.
    let sources: Vec<Vec<ArkFr>> = spec
        .kept
        .iter()
        .map(|pos| kept_base_column(*pos, s, p, o))
        .collect();
    let source_coms: Vec<ArkGT> = spec
        .kept
        .iter()
        .map(|pos| kept_base_commitment(*pos, com_s, com_p, com_o))
        .collect();
    let (compaction, kept_commitments, kept_columns, m) = prove_compaction_only(
        &sources,
        &source_coms,
        &combined_sel,
        &com_combined_sel,
        transcript,
    )?;

    Ok((
        RangeAtomSubproof {
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
            combine,
            com_combined_sel,
            compaction,
        },
        kept_commitments,
        kept_columns,
        m,
    ))
}

/// Verify one RANGED atom's sub-proofs, mirroring [`prove_ranged_atom`] on the SHARED
/// `transcript` in prover order: the four relocations to `D`, the authenticated
/// object-range selection at `D`, the range selection relocated back to the table
/// domain, the constant-equality selection, the combined-selection product, and the
/// compaction of the kept columns into `kept_commitments[0..m]`. Anchoring/root checks
/// happen once in the caller BEFORE this runs.
#[allow(clippy::too_many_arguments)]
fn verify_ranged_atom(
    spec: &AtomFilterSpec,
    range: &ObjectRangeSpec,
    n: usize,
    a: usize,
    com_s: &ArkGT,
    com_p: &ArkGT,
    com_o: &ArkGT,
    layer: &LayerCommitment,
    sub: &RangeAtomSubproof,
    kept_commitments: &[ArkGT],
    m: usize,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    let n_dict = root_bound_n_dict(layer)?;
    let (d_log, dict_log) = range_common_domain(a, n_dict)?;
    let com_dict_id = layer.node_value_dict_commitments.1;
    let com_dict_value = layer.node_value_dict_range_commitments.0;
    let com_dict_datatype = layer.node_value_dict_range_commitments.1;

    // Explicit sumcheck-dimension checks on the ranged sub-proofs, mirroring the
    // plain-atom path's `filter.sumcheck.round_polys.len() == a` check (defense-in-depth;
    // the Dory openings already pin these transitively). The object-range gadget runs
    // entirely at the common domain `D` (its lookup/datatype-mask/range/combine sumchecks
    // each have `d_log` variables); the constant-equality selection, the combined-selection
    // product, and the compaction all run at the table domain (`a` variables).
    for (rounds, expected) in [
        (sub.object_range.lookup.sumcheck.round_polys.len(), d_log),
        (sub.object_range.datatype.sumcheck.round_polys.len(), d_log),
        (sub.object_range.range.sumcheck.round_polys.len(), d_log),
        (sub.object_range.combine.sumcheck.round_polys.len(), d_log),
        (sub.equality.sumcheck.round_polys.len(), a),
        (sub.combine.sumcheck.round_polys.len(), a),
        (sub.compaction.sumcheck.round_polys.len(), a),
    ] {
        if rounds != expected {
            return Err(invalid(
                "ranged atom sub-proof dimension does not match the layer's committed domains",
            ));
        }
    }
    if let Some(second) = &sub.second {
        for rounds in [
            second.object_range.lookup.sumcheck.round_polys.len(),
            second.object_range.datatype.sumcheck.round_polys.len(),
            second.object_range.range.sumcheck.round_polys.len(),
            second.object_range.combine.sumcheck.round_polys.len(),
        ] {
            if rounds != d_log {
                return Err(invalid(
                    "second range sub-proof dimension does not match the dictionary domain",
                ));
            }
        }
        if second.intersect.sumcheck.round_polys.len() != a {
            return Err(invalid(
                "range-intersection dimension does not match the table domain",
            ));
        }
    }

    // (a) Verify the four relocations to the common domain D against the root-bound
    // originals (com_o + the three dict commitments).
    verify_column_relocation(
        com_o,
        a,
        &sub.com_object_at_d,
        d_log,
        &sub.relocate_object,
        transcript,
    )?;
    verify_column_relocation(
        &com_dict_id,
        dict_log,
        &sub.com_dict_id_at_d,
        d_log,
        &sub.relocate_dict_id,
        transcript,
    )?;
    verify_column_relocation(
        &com_dict_value,
        dict_log,
        &sub.com_dict_value_at_d,
        d_log,
        &sub.relocate_dict_value,
        transcript,
    )?;
    verify_column_relocation(
        &com_dict_datatype,
        dict_log,
        &sub.com_dict_datatype_at_d,
        d_log,
        &sub.relocate_dict_datatype,
        transcript,
    )?;

    // (b) Verify the authenticated object-range selection at D.
    verify_object_range_select(
        &sub.com_object_at_d,
        &sub.com_dict_id_at_d,
        &sub.com_dict_value_at_d,
        &sub.com_dict_datatype_at_d,
        n_dict,
        range.datatype,
        range.op,
        range.c,
        range.b_bits,
        n,
        &sub.com_range_sel_at_d,
        &sub.object_range,
        transcript,
    )?;

    // (c) Verify range_sel relocated from D back to the table domain.
    verify_column_relocation(
        &sub.com_range_sel_at_d,
        d_log,
        &sub.com_range_sel_at_table,
        a,
        &sub.relocate_range_sel,
        transcript,
    )?;

    // (c2) A public second bound requires exactly one matching subproof. Its selection
    // is authenticated against the same relocated columns and intersected with the first.
    let com_effective_range_sel = match (&range.second, &sub.second) {
        (None, None) => sub.com_range_sel_at_table,
        (Some(bound), Some(second)) => {
            verify_object_range_select(
                &sub.com_object_at_d,
                &sub.com_dict_id_at_d,
                &sub.com_dict_value_at_d,
                &sub.com_dict_datatype_at_d,
                n_dict,
                range.datatype,
                bound.op,
                bound.c,
                range.b_bits,
                n,
                &second.com_range_sel_at_d,
                &second.object_range,
                transcript,
            )?;
            verify_column_relocation(
                &second.com_range_sel_at_d,
                d_log,
                &second.com_range_sel_at_table,
                a,
                &second.relocate_range_sel,
                transcript,
            )?;
            verify_boolean_product(
                &sub.com_range_sel_at_table,
                &second.com_range_sel_at_table,
                &second.com_intersection_sel,
                &second.intersect,
                transcript,
            )?;
            second.com_intersection_sel
        }
        _ => {
            return Err(invalid(
                "second range subproof presence does not match the public plan",
            ))
        }
    };

    // (d) Verify the constant-equality selection.
    verify_selection_only(
        spec,
        n,
        com_s,
        com_p,
        com_o,
        &sub.com_equality_sel,
        &sub.equality,
        transcript,
    )?;

    // (e) Verify the combined selection = equality_sel · range_sel.
    verify_boolean_product(
        &sub.com_equality_sel,
        &com_effective_range_sel,
        &sub.com_combined_sel,
        &sub.combine,
        transcript,
    )?;

    // (f) Verify the compaction of the kept columns by the combined selection.
    let source_coms: Vec<ArkGT> = spec
        .kept
        .iter()
        .map(|pos| kept_base_commitment(*pos, com_s, com_p, com_o))
        .collect();
    verify_compaction_only(
        &source_coms,
        &sub.com_combined_sel,
        kept_commitments,
        m,
        &sub.compaction,
        transcript,
    )
}

/// Prove the COMPLETE result of a k-atom conjunctive BGP against a materialized head
/// layer. Each atom is filtered by its dictionary-authenticated constants into its dense
/// KEPT columns; predicate variables retain P directly. The resulting `k` relations (one
/// column per variable, in S,P,O order) are joined
/// by the left-deep binary-join chain ([`super::bgp`]) on the plan's shared variables — all
/// under ONE shared transcript anchored to `commitment_root`. Returns the combined proof
/// (whose per-atom parts carry the revealed subject/object constant ids), the final result
/// column commitments (schema `[r_other_out.., key, s_other_out..]`), the result length `m`,
/// and the REVEALED resolved predicate ids (authenticated by membership).
pub fn prove_layer_bgp(
    layer: &LayerCommitment,
    plan: &LayerBgpPlan,
) -> io::Result<(LayerBgpProof, Vec<ArkGT>, usize, Vec<u64>)> {
    let result = prove_layer_bgp_with_result(layer, plan)?;
    Ok((
        result.proof,
        result.result_commitments,
        result.result_len,
        result.predicate_ids,
    ))
}

/// Prove the complete BGP and retain the authenticated result values for delivery by an
/// explicitly invoking caller. The commitment-only [`prove_layer_bgp`] API remains
/// available for compatibility.
pub fn prove_layer_bgp_with_result(
    layer: &LayerCommitment,
    plan: &LayerBgpPlan,
) -> io::Result<LayerBgpResult> {
    let k = validate_bgp_plan(plan)?;
    let (s, p, o, n) = materialized_layer_columns(layer)?;

    if let Some(absence) = ground_dictionary_absence(layer, plan) {
        // The authenticated dictionary does not contain one public ground term, so the
        // triple cannot occur. Preserve the filter gadget's one hidden UNIT column shape,
        // but commit its empty (all-zero) result directly.
        let domain = padded_len(n)?;
        let zero_column = vec![f0(); domain];
        return Ok(LayerBgpResult {
            proof: LayerBgpProof {
                atoms: Vec::new(),
                chain: None,
                ground_absence: Some(absence),
            },
            result_commitments: vec![commit_column(&zero_column)?],
            result_columns: vec![zero_column],
            result_log_domain: domain.trailing_zeros() as usize,
            result_len: 0,
            predicate_ids: Vec::new(),
        });
    }

    // Resolve every atom's constant ids up front (fail closed if any is absent): the
    // predicate IRI via the predicate dictionary, each subject/object constant via the
    // node/value dictionary.
    let mut atom_ids = Vec::with_capacity(k);
    for atom in &plan.atoms {
        validate_atom(atom)?;
        let predicate_id = if atom.predicate_variable {
            None
        } else {
            Some(resolve_predicate_id(layer, &atom.predicate_term)?)
        };
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
    // Keep the legacy one-entry-per-atom public vector. Predicate variables have no
    // dictionary membership opening and use the plan-bound canonical placeholder zero.
    let predicate_ids: Vec<u64> = atom_ids
        .iter()
        .map(|ids| ids.predicate_id.unwrap_or(0))
        .collect();

    let (pred_dict_hash, pred_dict_id) = layer.predicate_dictionary_columns()?;
    let com_pred_hash = layer.predicate_dict_commitments.0;
    let com_pred_id = layer.predicate_dict_commitments.1;
    let (nv_dict_hash, nv_dict_id) = layer.node_value_dictionary_columns()?;
    let com_nv_hash = layer.node_value_dict_commitments.0;
    let com_nv_id = layer.node_value_dict_commitments.1;
    let [com_s, com_p, com_o] = layer.materialized_column_commitments;
    let a = padded_len(n)?.trailing_zeros() as usize;

    let mut transcript = Transcript254::new(LAYER_BGP_DOMAIN);
    bind_layer_bgp_statement(&mut transcript, layer, plan, &atom_ids);

    let mut atom_proofs = Vec::with_capacity(k);
    let mut relations: Vec<RelationWitness> = Vec::with_capacity(k);
    for (atom, ids) in plan.atoms.iter().zip(atom_ids.iter()) {
        // (1) Predicate IRI -> id, authenticated by the predicate dictionary.
        let predicate_membership = match ids.predicate_id {
            Some(predicate_id) => Some(prove_dict_membership(
                &pred_dict_hash,
                &pred_dict_id,
                &com_pred_hash,
                &com_pred_id,
                hash_term(&atom.predicate_term),
                ArkFr::from_u64(predicate_id),
                &mut transcript,
            )?),
            None => None,
        };
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
        // (3) The complete per-atom selection into the dense KEPT (variable) columns,
        // root-anchored. A PLAIN atom folds predicate + any subject/object constants via
        // the fused atom filter; a RANGED atom ANDs the equality selection with an
        // authenticated object-value range (relocations + object-range + combine +
        // compaction). Both yield the same shape of result: kept commitments + columns + m.
        let spec = atom_filter_spec(atom, ids);
        let (filter, range, kept_commitments, kept_columns, m) = match &atom.object_range {
            None => {
                let (filter, kept_commitments, m) = prove_atom_filter(
                    &spec,
                    &s,
                    &p,
                    &o,
                    n,
                    &com_s,
                    &com_p,
                    &com_o,
                    &mut transcript,
                )?;
                // The dense kept columns (== what prove_atom_filter committed) as the
                // chain relation; its arity equals the number of variables in the atom.
                let (kept_columns, m_columns) = atom_kept_columns(&spec, &s, &p, &o, n)?;
                debug_assert_eq!(m_columns, m);
                (Some(filter), None, kept_commitments, kept_columns, m)
            }
            Some(object_range) => {
                let (sub, kept_commitments, kept_columns, m) = prove_ranged_atom(
                    &spec,
                    object_range,
                    &s,
                    &p,
                    &o,
                    n,
                    a,
                    &com_s,
                    &com_p,
                    &com_o,
                    layer,
                    &mut transcript,
                )?;
                (None, Some(sub), kept_commitments, kept_columns, m)
            }
        };
        // (4) Register the filtered relation for the left-deep chain (same layout for a
        // plain or a ranged atom: kept columns at the table domain, real length m).
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
            range,
            kept_commitments,
            m,
            subject_id: ids.subject_id,
            object_id: ids.object_id,
        });
    }

    // (5) The left-deep chain over the filtered relations, when there is something to
    // join. For k=1 the complete atom filter result is already the query result.
    let relational_count = relational_atom_count(plan);
    let (chain, mut result_coms, mut result_columns, result_log_domain, mut result_m) =
        if is_all_ground(plan) {
            let result_m = usize::from(relations.iter().all(|relation| relation.n == 1));
            let mut unit = vec![f0(); 1usize << a];
            if result_m == 1 {
                unit[0] = ArkFr::one();
            }
            (None, vec![commit_column(&unit)?], vec![unit], a, result_m)
        } else if relational_count == 1 {
            let relation = &relations[0];
            (
                None,
                relation.commitments.clone(),
                relation.columns.clone(),
                relation.log_domain,
                relation.n,
            )
        } else {
            let result = prove_bgp_chain_with_result(
                &relations[..relational_count],
                &plan.joins,
                &mut transcript,
            )?;
            (
                Some(result.proof),
                result.commitments,
                result.columns,
                result.log_domain,
                result.n,
            )
        };

    if relational_count != 0
        && relations[relational_count..]
            .iter()
            .any(|relation| relation.n == 0)
    {
        result_m = 0;
        result_columns = result_columns
            .iter()
            .map(|_| vec![f0(); 1usize << result_log_domain])
            .collect();
        result_coms = result_columns
            .iter()
            .map(|column| commit_column(column))
            .collect::<io::Result<_>>()?;
    }

    Ok(LayerBgpResult {
        proof: LayerBgpProof {
            atoms: atom_proofs,
            chain,
            ground_absence: None,
        },
        result_commitments: result_coms,
        result_columns,
        result_log_domain,
        result_len: result_m,
        predicate_ids,
    })
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
    let k = validate_bgp_plan(plan)?;
    if let Some(claimed_absence) = proof.ground_absence {
        if !has_ground_result_unit(plan) {
            return Err(invalid(
                "dictionary-absence shortcut requires a single ground atom",
            ));
        }
        // Recommit every clear catalog/table before using catalog nonmembership. This
        // turns the scan below into an opening of the externally trusted layer root,
        // rather than trusting an unauthenticated convenience catalog.
        layer.validate()?;
        if ground_dictionary_absence(layer, plan) != Some(claimed_absence) {
            return Err(invalid(
                "claimed ground dictionary absence is not authentic",
            ));
        }
        if !proof.atoms.is_empty()
            || proof.chain.is_some()
            || !predicate_ids.is_empty()
            || m != 0
            || result_coms.len() != 1
        {
            return Err(invalid("invalid ground dictionary-absence proof shape"));
        }
        let zero_column = vec![f0(); padded_len(layer.state.materialized_count as usize)?];
        if result_coms[0] != commit_column(&zero_column)? {
            return Err(invalid(
                "ground dictionary-absence result is not the empty UNIT relation",
            ));
        }
        return Ok(());
    }
    if proof.atoms.len() != k || predicate_ids.len() != k {
        return Err(invalid("layer BGP proof/atom/id count mismatch"));
    }
    let n = layer.state.materialized_count as usize;
    let a = padded_len(n)?.trailing_zeros() as usize;

    // Reconstruct each atom's resolved ids from the public predicate ids + the
    // proof-carried subject/object constant ids, checking the id presence matches the
    // atom's constant structure (and the atom itself satisfies the v1 constraints).
    let mut atom_ids = Vec::with_capacity(k);
    for (i, (atom, ap)) in plan.atoms.iter().zip(proof.atoms.iter()).enumerate() {
        validate_atom(atom)?;
        let ids = AtomIds {
            predicate_id: if atom.predicate_variable {
                if predicate_ids[i] != 0 {
                    return Err(invalid(
                        "predicate variable must carry the canonical zero id placeholder",
                    ));
                }
                None
            } else {
                Some(predicate_ids[i])
            },
            subject_id: ap.subject_id,
            object_id: ap.object_id,
        };
        check_ids_consistent(atom, &ids)?;
        // The membership presence must match the constant structure too.
        if ids.predicate_id.is_some() != ap.predicate_membership.is_some()
            || ids.subject_id.is_some() != ap.subject_membership.is_some()
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
    let [com_s, com_p, com_o] = layer.materialized_column_commitments;

    let mut relations: Vec<RelationCommit> = Vec::with_capacity(k);
    for ((atom, ap), ids) in plan
        .atoms
        .iter()
        .zip(proof.atoms.iter())
        .zip(atom_ids.iter())
    {
        // A compacted PLAIN atom filter's dimension must match the layer's committed
        // table columns (defense-in-depth; the Dory openings would also catch a mismatch).
        if let Some(AtomFilterProof::Compacted(inner)) = &ap.filter {
            if inner.sumcheck.round_polys.len() != a {
                return Err(invalid(
                    "atom filter dimension does not match the layer's committed columns",
                ));
            }
        }
        // (1) Predicate membership: the revealed id is authentic for this IRI.
        if let Some(predicate_id) = ids.predicate_id {
            verify_dict_membership(
                &com_pred_hash,
                &com_pred_id,
                hash_term(&atom.predicate_term),
                ArkFr::from_u64(predicate_id),
                ap.predicate_membership
                    .as_ref()
                    .ok_or_else(|| invalid("constant predicate membership is absent"))?,
                &mut transcript,
            )?;
        }
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
        // (3) Selection: `kept_commitments[0..m]` is the complete per-atom result. Which
        // path runs is decided by the PLAN atom's `object_range` (public), NOT by the
        // prover-supplied proof — a plain atom must carry a `filter`, a ranged atom a
        // `range`. The spec reuses the SAME ids the memberships authenticated.
        let spec = atom_filter_spec(atom, ids);
        match &atom.object_range {
            None => {
                if ap.range.is_some() {
                    return Err(invalid("plain layer BGP atom carries a range subproof"));
                }
                let filter = ap
                    .filter
                    .as_ref()
                    .ok_or_else(|| invalid("plain layer BGP atom missing its filter proof"))?;
                verify_atom_filter(
                    &spec,
                    n,
                    &com_s,
                    &com_p,
                    &com_o,
                    &ap.kept_commitments,
                    ap.m,
                    filter,
                    &mut transcript,
                )?;
            }
            Some(object_range) => {
                if ap.filter.is_some() {
                    return Err(invalid(
                        "ranged layer BGP atom carries a plain filter proof",
                    ));
                }
                let sub = ap
                    .range
                    .as_ref()
                    .ok_or_else(|| invalid("ranged layer BGP atom missing its range subproof"))?;
                verify_ranged_atom(
                    &spec,
                    object_range,
                    n,
                    a,
                    &com_s,
                    &com_p,
                    &com_o,
                    layer,
                    sub,
                    &ap.kept_commitments,
                    ap.m,
                    &mut transcript,
                )?;
            }
        }
        relations.push(RelationCommit {
            commitments: ap.kept_commitments.clone(),
            log_domain: a,
            n: ap.m,
        });
    }

    // (4) The left-deep chain over the filtered relations, when k >= 2. A k=1 proof
    // must not smuggle in an unrelated chain: its public result is exactly the verified
    // atom-filter output.
    let relational_count = relational_atom_count(plan);
    let ground_relations = &relations[relational_count..];
    if ground_relations
        .iter()
        .any(|relation| relation.commitments.len() != 1 || relation.n > 1)
    {
        return Err(invalid("invalid ground BGP relation shape"));
    }
    let (mut chain_coms, mut chain_m) = if is_all_ground(plan) {
        if proof.chain.is_some() {
            return Err(invalid("invalid all-ground BGP relation shape"));
        }
        let result_m = usize::from(relations.iter().all(|relation| relation.n == 1));
        let mut unit = vec![f0(); 1usize << a];
        if result_m == 1 {
            unit[0] = ArkFr::one();
        }
        (vec![commit_column(&unit)?], result_m)
    } else if relational_count == 1 {
        if proof.chain.is_some() {
            return Err(invalid("single relational atom carries a join-chain proof"));
        }
        (relations[0].commitments.clone(), relations[0].n)
    } else {
        let chain = proof
            .chain
            .as_ref()
            .ok_or_else(|| invalid("multi-atom layer BGP is missing its join-chain proof"))?;
        verify_bgp_chain(
            &relations[..relational_count],
            &plan.joins,
            chain,
            &mut transcript,
        )?
    };
    if relational_count != 0 && ground_relations.iter().any(|relation| relation.n == 0) {
        let result_log_domain = if relational_count == 1 {
            relations[0].log_domain
        } else {
            proof
                .chain
                .as_ref()
                .and_then(|chain| chain.stages.last())
                .ok_or_else(|| invalid("mixed BGP is missing its final join stage"))?
                .join
                .num_vars()
        };
        let zero = vec![f0(); 1usize << result_log_domain];
        let zero_commitment = commit_column(&zero)?;
        chain_coms.fill(zero_commitment);
        chain_m = 0;
    }
    if chain_m != m || chain_coms.as_slice() != result_coms {
        return Err(invalid(
            "layer BGP result does not match the proof's chain output",
        ));
    }
    Ok(())
}

/// Re-derive the only valid public result commitment domain from the authenticated plan
/// and proof shape. This is especially important for a false mixed-arity gate, whose
/// all-zero commitments are group identities and therefore do not encode a domain size.
pub fn verify_layer_bgp_result_log_domain(
    layer: &LayerCommitment,
    plan: &LayerBgpPlan,
    proof: &LayerBgpProof,
    actual: usize,
) -> io::Result<()> {
    validate_bgp_plan(plan)?;
    let relational_count = relational_atom_count(plan);
    let expected = if relational_count <= 1 {
        padded_len(layer.state.materialized_count as usize)?.trailing_zeros() as usize
    } else {
        proof
            .chain
            .as_ref()
            .and_then(|chain| chain.stages.last())
            .ok_or_else(|| invalid("BGP result domain is missing its final join stage"))?
            .join
            .num_vars()
    };
    if actual != expected {
        return Err(invalid("BGP result commitment domain mismatch"));
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
        let layer = builder.commit().unwrap();
        layer.proof_commitment().unwrap()
    }

    // Independently recompute the expected dense result for a predicate id.
    fn expected_result(layer: &LayerCommitment, pred_id: u64) -> (Vec<ArkFr>, Vec<ArkFr>, usize) {
        let padded = padded_len(layer.materialized.len()).unwrap();
        let mut s_out = vec![f0(); padded];
        let mut o_out = vec![f0(); padded];
        let mut m = 0usize;
        for record in &layer.materialized {
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
        let (s, p, o, _n) = materialized_layer_columns(&lc).unwrap();
        assert_eq!(
            commit_column(&s).unwrap(),
            lc.materialized_column_commitments[0]
        );
        assert_eq!(
            commit_column(&p).unwrap(),
            lc.materialized_column_commitments[1]
        );
        assert_eq!(
            commit_column(&o).unwrap(),
            lc.materialized_column_commitments[2]
        );
    }

    #[test]
    fn prove_and_verify_by_iri_with_exact_result() {
        let lc = sample_layer();
        let (proof, com_sout, com_oout, m, resolved_id) = prove_layer_filter(&lc, P1).unwrap();
        // The verifier is given ONLY the IRI bytes + the revealed id (authenticated
        // by membership) -- never an out-of-band id.
        verify_layer_filter(
            &lc,
            lc.state.commitment_root,
            P1,
            resolved_id,
            &com_sout,
            &com_oout,
            m,
            &proof,
        )
        .unwrap();

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
        assert!(verify_layer_filter(
            &lc,
            lc.state.commitment_root,
            P1,
            wrong,
            &com_sout,
            &com_oout,
            m,
            &proof
        )
        .is_err());
    }

    #[test]
    fn substituted_id_from_a_different_predicate_is_rejected() {
        let lc = sample_layer();
        let (proof, com_sout, com_oout, m, _p1_id) = prove_layer_filter(&lc, P1).unwrap();
        // p2's genuine id, but paired with p1's IRI: (hash_term(p1), id_p2) is not a
        // dictionary row -> membership fails.
        let p2_id = resolve_predicate_id(&lc, P2).unwrap();
        assert!(verify_layer_filter(
            &lc,
            lc.state.commitment_root,
            P1,
            p2_id,
            &com_sout,
            &com_oout,
            m,
            &proof
        )
        .is_err());
    }

    #[test]
    fn absent_predicate_iri_fails_closed() {
        let lc = sample_layer();
        let err = prove_layer_filter(&lc, b"http://ex/absent").unwrap_err();
        assert!(err
            .to_string()
            .contains("predicate not in layer dictionary"));
    }

    #[test]
    fn tampered_result_is_rejected_under_the_combined_transcript() {
        let lc = sample_layer();
        let (proof, com_sout, com_oout, m, resolved_id) = prove_layer_filter(&lc, P1).unwrap();
        // Wrong S_out commitment (use the O_out one) -> filter opening fails.
        assert!(verify_layer_filter(
            &lc,
            lc.state.commitment_root,
            P1,
            resolved_id,
            &com_oout,
            &com_oout,
            m,
            &proof
        )
        .is_err());
        // Honest still verifies.
        verify_layer_filter(
            &lc,
            lc.state.commitment_root,
            P1,
            resolved_id,
            &com_sout,
            &com_oout,
            m,
            &proof,
        )
        .unwrap();
    }

    #[test]
    fn child_add_and_remove_proves_against_complete_head_view() {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for (s, p, o) in [("a1", "p1", "b1"), ("a2", "p1", "b2"), ("a3", "p1", "b3")] {
            builder
                .add_value_triple(ValueTriple::new_node(&iri(s), &iri(p), &iri(o)))
                .unwrap();
        }
        let base = builder.commit().unwrap();
        let child = base.open_write().unwrap();
        child
            .remove_value_triple(ValueTriple::new_node(&iri("a1"), &iri("p1"), &iri("b1")))
            .unwrap();
        child
            .add_value_triple(ValueTriple::new_node(&iri("a4"), &iri("p1"), &iri("b4")))
            .unwrap();
        let head = child.commit().unwrap();
        let lc = head.proof_commitment().unwrap();
        assert_eq!(lc.state.addition_count, 1);
        assert_eq!(lc.state.removal_count, 1);
        assert_eq!(lc.state.materialized_count, 3);
        let subjects: Vec<_> = lc
            .materialized
            .iter()
            .map(|record| record.triple.subject.as_slice())
            .collect();
        assert!(!subjects.contains(&iri("a1").as_bytes()));
        assert!(subjects.contains(&iri("a2").as_bytes()));
        assert!(subjects.contains(&iri("a3").as_bytes()));
        assert!(subjects.contains(&iri("a4").as_bytes()));

        let (proof, com_s, com_o, m, id) = prove_layer_filter(&lc, P1).unwrap();
        verify_layer_filter(
            &lc,
            lc.state.commitment_root,
            P1,
            id,
            &com_s,
            &com_o,
            m,
            &proof,
        )
        .unwrap();
        assert_eq!(m, 3);
    }

    #[test]
    fn child_additions_prove_with_inherited_parent_rows() {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for (s, p, o) in [("a1", "p1", "b1"), ("a2", "p1", "b2")] {
            builder
                .add_value_triple(ValueTriple::new_node(&iri(s), &iri(p), &iri(o)))
                .unwrap();
        }
        let base = builder.commit().unwrap();
        let child = base.open_write().unwrap();
        for (s, o) in [("a3", "b3"), ("a4", "b4")] {
            child
                .add_value_triple(ValueTriple::new_node(&iri(s), &iri("pn"), &iri(o)))
                .unwrap();
        }
        let head = child.commit().unwrap();
        let lc = head.proof_commitment().unwrap();
        assert_eq!(lc.state.removal_count, 0);
        assert!(lc.state.parent_state_root.is_some());
        assert_eq!(lc.state.addition_count, 2);
        assert_eq!(lc.state.materialized_count, 4);

        let pn: &[u8] = b"http://ex/pn";
        let (proof, com_sout, com_oout, m, id) = prove_layer_filter(&lc, pn).unwrap();
        verify_layer_filter(
            &lc,
            lc.state.commitment_root,
            pn,
            id,
            &com_sout,
            &com_oout,
            m,
            &proof,
        )
        .unwrap();
        assert_eq!(m, 2);
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
        for record in &lc.materialized {
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
        let (proof, com_x, com_y, com_z, m_j, id) = prove_layer_graph_pattern2(&lc, KNOWS).unwrap();
        // Verifier sees ONLY the IRI + revealed id + result commitments.
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
        let lc = build_layer(&[
            ("a", "knows", "b"),
            ("b", "knows", "c"),
            ("c", "knows", "d"),
        ]);
        let (proof, com_x, com_y, com_z, m_j, id) = prove_layer_graph_pattern2(&lc, KNOWS).unwrap();
        // Wrong result commitment (pass Com Z for Com X) -> mismatch with proof output.
        assert!(verify_layer_graph_pattern2(
            &lc,
            lc.state.commitment_root,
            KNOWS,
            id,
            &com_z,
            &com_y,
            &com_z,
            m_j,
            &proof
        )
        .is_err());
        // Wrong claimed count.
        assert!(verify_layer_graph_pattern2(
            &lc,
            lc.state.commitment_root,
            KNOWS,
            id,
            &com_x,
            &com_y,
            &com_z,
            m_j + 1,
            &proof
        )
        .is_err());
        // Honest verifies.
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

    #[test]
    fn graph_pattern2_fabricated_join_output_is_rejected() {
        // Fabricate a bogus X-commitment and pass it consistently (defeats the equality
        // gate); the self-join column opening still binds X to the sumcheck -> REJECT.
        let lc = build_layer(&[
            ("a", "knows", "b"),
            ("b", "knows", "c"),
            ("c", "knows", "d"),
        ]);
        let (proof, _com_x, com_y, com_z, m_j, id) =
            prove_layer_graph_pattern2(&lc, KNOWS).unwrap();
        let (_, _, _, n_join, _) = expected_join(&lc, id);
        let fake_x = commit_column(&pad(&[ArkFr::from_u64(424242)], n_join)).unwrap();
        let mut bad = proof.clone();
        bad.join_commitments.x = fake_x;
        assert!(verify_layer_graph_pattern2(
            &lc,
            lc.state.commitment_root,
            KNOWS,
            id,
            &fake_x,
            &com_y,
            &com_z,
            m_j,
            &bad
        )
        .is_err());
    }

    #[test]
    fn graph_pattern2_no_two_hop_paths_empty_verifies() {
        // Disjoint edges a->b and c->d: no y with x->y and y->z, so the result is empty.
        let lc = build_layer(&[("a", "knows", "b"), ("c", "knows", "d")]);
        let (proof, com_x, com_y, com_z, m_j, id) = prove_layer_graph_pattern2(&lc, KNOWS).unwrap();
        assert_eq!(m_j, 0);
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

    #[test]
    fn graph_pattern2_tampered_layer_root_is_rejected() {
        let lc = build_layer(&[
            ("a", "knows", "b"),
            ("b", "knows", "c"),
            ("c", "knows", "d"),
        ]);
        let good_root = lc.state.commitment_root;
        let (proof, com_x, com_y, com_z, m_j, id) = prove_layer_graph_pattern2(&lc, KNOWS).unwrap();
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
        let lc = build_layer(&[
            ("a", "knows", "b"),
            ("b", "knows", "c"),
            ("c", "knows", "d"),
        ]);
        let (proof, com_x, com_y, com_z, m_j, id) = prove_layer_graph_pattern2(&lc, KNOWS).unwrap();
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
        for rec in &lc.materialized {
            if rec.predicate_id == pred_id {
                s.push(ArkFr::from_u64(rec.subject_id));
                o.push(ArkFr::from_u64(rec.object_id));
            }
        }
        vec![s, o]
    }

    #[test]
    fn delivered_result_columns_are_bound_to_the_proven_commitments() {
        let lc = build_layer(&[
            ("a", "knows", "b"),
            ("b", "knows", "c"),
            ("x", "likes", "y"),
        ]);
        let plan = LayerBgpPlan {
            atoms: vec![atom(KNOWS)],
            joins: Vec::new(),
        };
        let result = prove_layer_bgp_with_result(&lc, &plan).unwrap();
        verify_layer_bgp(
            &lc,
            lc.state.commitment_root,
            &plan,
            &result.predicate_ids,
            &result.result_commitments,
            result.result_len,
            &result.proof,
        )
        .unwrap();

        let columns = result.real_columns();
        verify_result_columns(
            &result.result_commitments,
            &columns,
            result.result_len,
            result.result_log_domain,
        )
        .unwrap();

        let mut tampered = columns;
        tampered[0][0] = tampered[0][0] + ArkFr::one();
        assert!(verify_result_columns(
            &result.result_commitments,
            &tampered,
            result.result_len,
            result.result_log_domain,
        )
        .is_err());
    }

    #[test]
    fn delivered_lexical_objects_are_bound_to_authenticated_result_ids() {
        let lc = build_layer(&[("a", "knows", "b"), ("b", "knows", "c")]);
        let plan = LayerBgpPlan {
            atoms: vec![atom(KNOWS)],
            joins: Vec::new(),
        };
        let result = prove_layer_bgp_with_result(&lc, &plan).unwrap();
        let columns = result.real_columns();
        let objects =
            resolve_node_value_result_objects(&lc, lc.state.commitment_root, &columns).unwrap();

        assert_eq!(objects.len(), 2);
        assert_eq!(objects[0][0], CanonicalObject::Node(iri("a").into_bytes()));
        assert_eq!(objects[1][0], CanonicalObject::Node(iri("b").into_bytes()));
        verify_node_value_result_objects(&lc, lc.state.commitment_root, &columns, &objects).unwrap();

        let mut tampered = objects.clone();
        tampered[1][0] = CanonicalObject::Node(iri("c").into_bytes());
        assert!(
            verify_node_value_result_objects(
                &lc,
                lc.state.commitment_root,
                &columns,
                &tampered,
            )
            .is_err()
        );

        let mut wrong_root = lc.state.commitment_root;
        wrong_root[0] ^= 0xff;
        assert!(resolve_node_value_result_objects(&lc, wrong_root, &columns).is_err());

        // The serialized catalog is only a claimed opening. Changing it without
        // changing the root-bound commitment must not change the resolved value.
        let mut tampered_catalog = (*lc).clone();
        tampered_catalog.node_value_catalog[0].triple.object =
            CanonicalObject::Node(iri("forged").into_bytes());
        assert!(
            resolve_node_value_result_objects(
                &tampered_catalog,
                lc.state.commitment_root,
                &columns,
            )
            .is_err()
        );
    }

    #[test]
    fn nullable_result_descriptors_are_authoritative() {
        let lc = build_layer(&[("a", "knows", "b")]);
        let node_id = lc
            .node_value_catalog
            .iter()
            .find(|record| record.triple.object == CanonicalObject::Node(iri("a").into_bytes()))
            .unwrap()
            .object_id;
        let predicate_id = resolve_predicate_id(&lc, KNOWS).unwrap();
        let columns = vec![
            vec![ArkFr::from_u64(node_id)],
            vec![ArkFr::from_u64(predicate_id)],
            vec![ArkFr::zero()],
        ];
        let descriptors = vec![
            ResultColumnDescriptor {
                namespace: ResultIdNamespace::NodeValue,
                nullable: false,
            },
            ResultColumnDescriptor {
                namespace: ResultIdNamespace::Predicate,
                nullable: false,
            },
            ResultColumnDescriptor {
                namespace: ResultIdNamespace::NodeValue,
                nullable: true,
            },
        ];
        let values = resolve_result_values(
            &lc,
            lc.state.commitment_root,
            &columns,
            &descriptors,
        )
        .unwrap();
        assert_eq!(
            values[0][0],
            CanonicalResultValue::Object(CanonicalObject::Node(iri("a").into_bytes()))
        );
        assert_eq!(
            values[1][0],
            CanonicalResultValue::Object(CanonicalObject::Node(iri("knows").into_bytes()))
        );
        assert_eq!(values[2][0], CanonicalResultValue::Null);

        let mut nonnullable = descriptors.clone();
        nonnullable[2].nullable = false;
        assert!(resolve_result_values(
            &lc,
            lc.state.commitment_root,
            &columns,
            &nonnullable,
        )
        .is_err());

        let mut forged = columns;
        forged[2][0] = ArkFr::from_u64(u64::MAX);
        assert!(resolve_result_values(
            &lc,
            lc.state.commitment_root,
            &forged,
            &descriptors,
        )
        .is_err());
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
        let mut domain = padded_len(lc.materialized.len()).unwrap();
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
                JoinStep {
                    acc_key: 1,
                    rel_key: 0,
                    additional_keys: Vec::new(),
                }, // y = O(knows) == S(likes)
                JoinStep {
                    acc_key: 2,
                    rel_key: 0,
                    additional_keys: Vec::new(),
                }, // z = O(likes) == S(owns)
            ],
        };
        let (proof, result_coms, m, ids) = prove_layer_bgp(&lc, &plan).unwrap();
        // Verifier sees ONLY the IRIs + revealed ids + result commitments.
        verify_layer_bgp(
            &lc,
            lc.state.commitment_root,
            &plan,
            &ids,
            &result_coms,
            m,
            &proof,
        )
        .unwrap();

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
            &lc,
            lc.state.commitment_root,
            KNOWS,
            gp_id,
            &com_x,
            &com_y,
            &com_z,
            gp_m,
            &gp,
        )
        .unwrap();

        let plan = LayerBgpPlan {
            atoms: vec![atom(KNOWS), atom(KNOWS)],
            joins: vec![JoinStep {
                acc_key: 1,
                rel_key: 0,
                additional_keys: Vec::new(),
            }],
        };
        let (proof, coms, m, ids) = prove_layer_bgp(&lc, &plan).unwrap();
        verify_layer_bgp(&lc, lc.state.commitment_root, &plan, &ids, &coms, m, &proof).unwrap();

        assert_eq!(m, gp_m);
        assert_eq!(coms, vec![com_x, com_y, com_z]);
    }

    #[test]
    fn layer_bgp_k1_is_exactly_the_complete_atom_filter() {
        let lc = build_layer(&[
            ("a", "knows", "b"),
            ("b", "knows", "c"),
            ("x", "likes", "y"),
        ]);
        let plan = LayerBgpPlan {
            atoms: vec![atom(KNOWS)],
            joins: vec![],
        };

        let (proof, coms, m, ids) = prove_layer_bgp(&lc, &plan).unwrap();
        verify_layer_bgp(&lc, lc.state.commitment_root, &plan, &ids, &coms, m, &proof).unwrap();

        assert_eq!(m, 2);
        assert!(proof.chain.is_none());
        let expected = atom_relation(&lc, ids[0]);
        let domain = padded_len(lc.materialized.len()).unwrap();
        assert_eq!(coms.len(), 2);
        for (commitment, column) in coms.iter().zip(expected.iter()) {
            assert_eq!(*commitment, commit_column(&pad(column, domain)).unwrap());
        }
    }

    #[test]
    fn predicate_variable_retains_root_bound_materialized_p_column() {
        let lc = build_layer(&[
            ("a", "knows", "b"),
            ("b", "knows", "c"),
            ("x", "likes", "y"),
        ]);
        let plan = LayerBgpPlan {
            atoms: vec![LayerBgpAtom {
                subject: AtomTerm::Variable,
                predicate_term: Vec::new(),
                predicate_variable: true,
                object: AtomTerm::Variable,
                object_range: None,
                subject_object_equal: false,
            }],
            joins: vec![],
        };

        let result = prove_layer_bgp_with_result(&lc, &plan).unwrap();
        verify_layer_bgp(
            &lc,
            lc.state.commitment_root,
            &plan,
            &result.predicate_ids,
            &result.result_commitments,
            result.result_len,
            &result.proof,
        )
        .unwrap();

        assert_eq!(result.predicate_ids, vec![0]);
        assert_eq!(result.result_len, lc.materialized.len());
        assert_eq!(result.result_commitments, lc.materialized_column_commitments);
        assert_eq!(
            result_id_namespaces(&plan).unwrap(),
            vec![
                ResultIdNamespace::NodeValue,
                ResultIdNamespace::Predicate,
                ResultIdNamespace::NodeValue,
            ]
        );
        let resolved = resolve_typed_result_objects(
            &lc,
            lc.state.commitment_root,
            &result.real_columns(),
            &result_id_namespaces(&plan).unwrap(),
        )
        .unwrap();
        assert_eq!(resolved[1].len(), lc.materialized.len());
        assert!(resolved[1]
            .iter()
            .all(|value| matches!(value, CanonicalObject::Node(_))));
    }

    #[test]
    fn predicate_to_node_value_join_is_rejected() {
        let variable_predicate = LayerBgpAtom {
            subject: node_const("a"),
            predicate_term: Vec::new(),
            predicate_variable: true,
            object: node_const("b"),
            object_range: None,
            subject_object_equal: false,
        };
        let plan = LayerBgpPlan {
            atoms: vec![variable_predicate, atom(KNOWS)],
            joins: vec![JoinStep {
                acc_key: 0,
                rel_key: 0,
                additional_keys: Vec::new(),
            }],
        };
        assert!(result_id_namespaces(&plan).is_err());
    }

    #[test]
    fn layer_bgp_k2_rejects_a_missing_join_chain() {
        let lc = build_layer(&[("a", "knows", "b"), ("b", "knows", "c")]);
        let plan = LayerBgpPlan {
            atoms: vec![atom(KNOWS), atom(KNOWS)],
            joins: vec![JoinStep {
                acc_key: 1,
                rel_key: 0,
                additional_keys: Vec::new(),
            }],
        };
        let (mut proof, coms, m, ids) = prove_layer_bgp(&lc, &plan).unwrap();
        proof.chain = None;
        assert!(
            verify_layer_bgp(&lc, lc.state.commitment_root, &plan, &ids, &coms, m, &proof,)
                .is_err()
        );
    }

    #[test]
    fn layer_bgp_wrong_expected_root_is_rejected() {
        let lc = build_layer(&[("a", "knows", "b"), ("b", "likes", "c"), ("c", "owns", "d")]);
        let plan = LayerBgpPlan {
            atoms: vec![atom(KNOWS), atom(LIKES), atom(OWNS)],
            joins: vec![
                JoinStep {
                    acc_key: 1,
                    rel_key: 0,
                    additional_keys: Vec::new(),
                },
                JoinStep {
                    acc_key: 2,
                    rel_key: 0,
                    additional_keys: Vec::new(),
                },
            ],
        };
        let (proof, result_coms, m, ids) = prove_layer_bgp(&lc, &plan).unwrap();
        // A DIFFERENT externally-trusted root must reject even an honest proof.
        let mut wrong_root = lc.state.commitment_root;
        wrong_root[0] ^= 0xff;
        assert!(verify_layer_bgp(&lc, wrong_root, &plan, &ids, &result_coms, m, &proof).is_err());
        // The correct external root still verifies.
        verify_layer_bgp(
            &lc,
            lc.state.commitment_root,
            &plan,
            &ids,
            &result_coms,
            m,
            &proof,
        )
        .unwrap();
    }

    #[test]
    fn layer_bgp_absent_predicate_fails_closed() {
        let lc = build_layer(&[("a", "knows", "b"), ("b", "knows", "c")]);
        let plan = LayerBgpPlan {
            atoms: vec![atom(KNOWS), atom(b"http://ex/absent")],
            joins: vec![JoinStep {
                acc_key: 1,
                rel_key: 0,
                additional_keys: Vec::new(),
            }],
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
        let layer = builder.commit().unwrap();
        layer.proof_commitment().unwrap()
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
        for rec in &lc.materialized {
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
            kept_relation(
                lc,
                &atoms[i],
                predicate_ids[i],
                subject_ids[i],
                object_ids[i],
            )
        };
        let mut acc = rel(0);
        let mut acc_len = acc[0].len();
        let mut domain = padded_len(lc.materialized.len()).unwrap();
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
        let (exp_cols, domain) = expected_const_bgp(
            lc,
            &plan.atoms,
            &ids,
            &subject_ids,
            &object_ids,
            &plan.joins,
        );
        assert_eq!(coms.len(), exp_cols.len());
        for (c, col) in exp_cols.iter().enumerate() {
            assert_eq!(
                commit_column(&pad(col, domain)).unwrap(),
                coms[c],
                "column {c}"
            );
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
                    predicate_variable: false,
                    object: node_const("b"),
                    object_range: None,
                    subject_object_equal: false,
                },
            ],
            joins: vec![JoinStep {
                acc_key: 0,
                rel_key: 0,
                additional_keys: Vec::new(),
            }], // x = S(likes) == S(knows)
        };
        let (proof, _coms, _m, _ids) = prove_layer_bgp(&lc, &plan).unwrap();
        // The revealed object id equals <b>'s node/value dictionary id; the variable
        // atom reveals no constant ids.
        let b_id =
            resolve_node_value_id(&lc, &CanonicalObject::Node(iri("b").into_bytes())).unwrap();
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
        let a_id =
            resolve_node_value_id(&lc, &CanonicalObject::Node(iri("a").into_bytes())).unwrap();
        let hi_id = lc
            .materialized
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
                    predicate_variable: false,
                    object: AtomTerm::ValueConstant(dt, lex),
                    object_range: None,
                    subject_object_equal: false,
                },
                LayerBgpAtom::all_variables(KNOWS.to_vec()),
            ],
            joins: vec![JoinStep {
                acc_key: 0,
                rel_key: 0,
                additional_keys: Vec::new(),
            }], // x = S(name) == S(knows)
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
                    predicate_variable: false,
                    object: AtomTerm::Variable,
                    object_range: None,
                    subject_object_equal: false,
                },
                LayerBgpAtom::all_variables(LIKES.to_vec()),
            ],
            joins: vec![JoinStep {
                acc_key: 0,
                rel_key: 0,
                additional_keys: Vec::new(),
            }], // y = O(knows) == S(likes)
        };
        let (proof, _coms, _m, _ids) = prove_layer_bgp(&lc, &plan).unwrap();
        let a_id =
            resolve_node_value_id(&lc, &CanonicalObject::Node(iri("a").into_bytes())).unwrap();
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
                    predicate_variable: false,
                    object: node_const("g"),
                    object_range: None,
                    subject_object_equal: false,
                },
            ],
            joins: vec![JoinStep {
                acc_key: 1,
                rel_key: 0,
                additional_keys: Vec::new(),
            }], // y = O(knows) == S(likes)
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
        assert!(verify_layer_bgp(
            &lc,
            lc.state.commitment_root,
            &plan,
            &ids,
            &coms,
            m,
            &bad_id
        )
        .is_err());

        // Substituting a DIFFERENT genuine node id (here <b>'s id) for <g> is also rejected:
        // membership binds the PAIR (hash_object(<g>), id), so <b>'s id fails for <g>.
        let b_id =
            resolve_node_value_id(&lc, &CanonicalObject::Node(iri("b").into_bytes())).unwrap();
        let mut sub_id = proof.clone();
        sub_id.atoms[1].object_id = Some(b_id);
        assert!(verify_layer_bgp(
            &lc,
            lc.state.commitment_root,
            &plan,
            &ids,
            &coms,
            m,
            &sub_id
        )
        .is_err());

        // Wrong externally-trusted root rejects even an honest proof.
        let mut wrong_root = lc.state.commitment_root;
        wrong_root[0] ^= 0xff;
        assert!(verify_layer_bgp(&lc, wrong_root, &plan, &ids, &coms, m, &proof).is_err());

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
                    predicate_variable: false,
                    object: node_const("absent"),
                    object_range: None,
                    subject_object_equal: false,
                },
            ],
            joins: vec![JoinStep {
                acc_key: 1,
                rel_key: 0,
                additional_keys: Vec::new(),
            }],
        };
        let err = prove_layer_bgp(&lc, &plan).unwrap_err();
        assert!(err
            .to_string()
            .contains("constant not in layer node/value dictionary"));
    }

    #[test]
    fn layer_bgp_single_ground_atom_proves_presence_and_absence() {
        let lc = build_layer(&[("a", "knows", "b"), ("b", "likes", "c")]);
        let atom = |subject: &str, predicate: &[u8], object: &str| LayerBgpPlan {
            atoms: vec![LayerBgpAtom {
                subject: node_const(subject),
                predicate_term: predicate.to_vec(),
                predicate_variable: false,
                object: node_const(object),
                object_range: None,
                subject_object_equal: false,
            }],
            joins: Vec::new(),
        };

        for (plan, expected_len, expected_absence) in [
            (atom("a", KNOWS, "b"), 1, None),
            (atom("a", LIKES, "c"), 0, None),
            (
                atom("a", KNOWS, "missing-object"),
                0,
                Some(GroundAbsence::Object),
            ),
            (
                atom("missing-subject", KNOWS, "b"),
                0,
                Some(GroundAbsence::Subject),
            ),
            (
                atom("a", b"http://ex/missing-predicate", "b"),
                0,
                Some(GroundAbsence::Predicate),
            ),
        ] {
            let result = prove_layer_bgp_with_result(&lc, &plan).unwrap();
            assert_eq!(result.result_len, expected_len);
            assert_eq!(result.proof.ground_absence, expected_absence);
            assert_eq!(result.result_commitments.len(), 1);
            assert_eq!(result.result_columns.len(), 1);
            verify_layer_bgp(
                &lc,
                lc.state.commitment_root,
                &plan,
                &result.predicate_ids,
                &result.result_commitments,
                result.result_len,
                &result.proof,
            )
            .unwrap();
            if let Some(absence) = expected_absence {
                let mut forged = result.proof.clone();
                forged.ground_absence = Some(match absence {
                    GroundAbsence::Predicate => GroundAbsence::Subject,
                    GroundAbsence::Subject | GroundAbsence::Object => GroundAbsence::Predicate,
                });
                assert!(verify_layer_bgp(
                    &lc,
                    lc.state.commitment_root,
                    &plan,
                    &result.predicate_ids,
                    &result.result_commitments,
                    result.result_len,
                    &forged,
                )
                .is_err());

                let mut forged_present = result.proof.clone();
                forged_present.ground_absence = None;
                assert!(verify_layer_bgp(
                    &lc,
                    lc.state.commitment_root,
                    &plan,
                    &result.predicate_ids,
                    &result.result_commitments,
                    1,
                    &forged_present,
                )
                .is_err());
            }
        }
    }

    #[test]
    fn layer_bgp_all_ground_conjunction_ands_authenticated_existence_bits() {
        let lc = build_layer(&[("a", "knows", "b"), ("b", "likes", "c")]);
        let atom = |subject: &str, predicate: &[u8], object: &str| LayerBgpAtom {
            subject: node_const(subject),
            predicate_term: predicate.to_vec(),
            predicate_variable: false,
            object: node_const(object),
            object_range: None,
            subject_object_equal: false,
        };
        for (plan, expected_len) in [
            (
                LayerBgpPlan {
                    atoms: vec![atom("a", KNOWS, "b"), atom("b", LIKES, "c")],
                    joins: Vec::new(),
                },
                1,
            ),
            (
                LayerBgpPlan {
                    atoms: vec![atom("a", KNOWS, "b"), atom("a", LIKES, "c")],
                    joins: Vec::new(),
                },
                0,
            ),
            (
                LayerBgpPlan {
                    atoms: vec![atom("a", KNOWS, "b"), atom("a", KNOWS, "b")],
                    joins: Vec::new(),
                },
                1,
            ),
        ] {
            let result = prove_layer_bgp_with_result(&lc, &plan).unwrap();
            assert_eq!(result.result_len, expected_len);
            assert_eq!(result.result_commitments.len(), 1);
            verify_layer_bgp(
                &lc,
                lc.state.commitment_root,
                &plan,
                &result.predicate_ids,
                &result.result_commitments,
                result.result_len,
                &result.proof,
            )
            .unwrap();

            let mut forged_atom = result.proof.clone();
            forged_atom.atoms[0].m = 1 - forged_atom.atoms[0].m;
            assert!(verify_layer_bgp(
                &lc,
                lc.state.commitment_root,
                &plan,
                &result.predicate_ids,
                &result.result_commitments,
                result.result_len,
                &forged_atom,
            )
            .is_err());
            assert!(verify_layer_bgp(
                &lc,
                lc.state.commitment_root,
                &plan,
                &result.predicate_ids,
                &result.result_commitments,
                1 - expected_len,
                &result.proof,
            )
            .is_err());
        }
    }

    #[test]
    fn layer_bgp_ground_atoms_gate_relational_results() {
        let lc = build_layer(&[("a", "knows", "b"), ("b", "likes", "c")]);
        for (ground, expected_len) in [
            (
                LayerBgpAtom {
                    subject: node_const("a"),
                    predicate_term: KNOWS.to_vec(),
                    predicate_variable: false,
                    object: node_const("b"),
                    object_range: None,
                    subject_object_equal: false,
                },
                1,
            ),
            (
                LayerBgpAtom {
                    subject: node_const("a"),
                    predicate_term: LIKES.to_vec(),
                    predicate_variable: false,
                    object: node_const("c"),
                    object_range: None,
                    subject_object_equal: false,
                },
                0,
            ),
        ] {
            let plan = LayerBgpPlan {
                atoms: vec![LayerBgpAtom::all_variables(LIKES.to_vec()), ground],
                joins: Vec::new(),
            };
            let result = prove_layer_bgp_with_result(&lc, &plan).unwrap();
            assert_eq!(result.result_len, expected_len);
            verify_layer_bgp(
                &lc,
                lc.state.commitment_root,
                &plan,
                &result.predicate_ids,
                &result.result_commitments,
                result.result_len,
                &result.proof,
            )
            .unwrap();
            let mut forged = result.proof.clone();
            forged.atoms[1].m = 1 - forged.atoms[1].m;
            assert!(verify_layer_bgp(
                &lc,
                lc.state.commitment_root,
                &plan,
                &result.predicate_ids,
                &result.result_commitments,
                result.result_len,
                &forged,
            )
            .is_err());
        }
    }

    #[test]
    fn layer_bgp_value_subject_is_rejected() {
        let lc = build_layer(&[("a", "knows", "b"), ("b", "likes", "c")]);
        // A value literal in the SUBJECT position is rejected.
        let value_subject = LayerBgpPlan {
            atoms: vec![
                LayerBgpAtom {
                    subject: AtomTerm::ValueConstant(CanonicalDatatype::String, b"x".to_vec()),
                    predicate_term: KNOWS.to_vec(),
                    predicate_variable: false,
                    object: AtomTerm::Variable,
                    object_range: None,
                    subject_object_equal: false,
                },
                LayerBgpAtom::all_variables(LIKES.to_vec()),
            ],
            joins: vec![JoinStep {
                acc_key: 0,
                rel_key: 0,
                additional_keys: Vec::new(),
            }],
        };
        assert!(prove_layer_bgp(&lc, &value_subject).is_err());
    }

    // ---- Q_k with a numeric object-value RANGE filter (CP-C) end-to-end ----

    const AGE: &[u8] = b"http://ex/age";
    const UINT32_TAG: u64 = CanonicalDatatype::UInt32 as u64;
    const BOOLEAN_TAG: u64 = CanonicalDatatype::Boolean as u64;

    fn fr(v: u64) -> ArkFr {
        ArkFr::from_u64(v)
    }

    fn cmp_op(op: RangeOp, a: u64, b: u64) -> bool {
        match op {
            RangeOp::Lt => a < b,
            RangeOp::Le => a <= b,
            RangeOp::Gt => a > b,
            RangeOp::Ge => a >= b,
        }
    }

    // A base layer of `t(person, age, <UInt32>)` and `t(person, name, <node>)` triples.
    // Ages are inserted OUT of numeric order; every person has exactly one name.
    fn build_age_name_layer() -> std::sync::Arc<LayerCommitment> {
        use tdb_succinct::TdbDataType;
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        let age = iri("age");
        let name = iri("name");
        for (person, a) in [
            ("alice", 30u32),
            ("bob", 15),
            ("carol", 18),
            ("dave", 40),
            ("erin", 18),
        ] {
            builder
                .add_value_triple(ValueTriple::new_value(
                    &iri(person),
                    &age,
                    u32::make_entry(&a),
                ))
                .unwrap();
            builder
                .add_value_triple(ValueTriple::new_node(
                    &iri(person),
                    &name,
                    &iri(&format!("{person}_n")),
                ))
                .unwrap();
        }
        let layer = builder.commit().unwrap();
        layer.proof_commitment().unwrap()
    }

    // A layer with FEW triples but MANY distinct node/values: 3 people, each with a
    // distinct age and a distinct name node. The node/value dictionary (9 entries -> 16
    // rows) is LARGER than the table (6 triples -> 8 rows), so the common relocation
    // domain D = 16 strictly EXCEEDS the table domain 2^a = 8 — exercising the up
    // (object/dict -> D) and down (range_sel -> table) relocation branches end-to-end.
    fn build_distinct_age_name_layer() -> std::sync::Arc<LayerCommitment> {
        use tdb_succinct::TdbDataType;
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        let age = iri("age");
        let name = iri("name");
        for (person, a) in [("p1", 10u32), ("p2", 25), ("p3", 5)] {
            builder
                .add_value_triple(ValueTriple::new_value(
                    &iri(person),
                    &age,
                    u32::make_entry(&a),
                ))
                .unwrap();
            builder
                .add_value_triple(ValueTriple::new_node(
                    &iri(person),
                    &name,
                    &iri(&format!("{person}_n")),
                ))
                .unwrap();
        }
        let layer = builder.commit().unwrap();
        layer.proof_commitment().unwrap()
    }

    // The UInt32 value behind a node/value dict id (its order-preserving encoding is the
    // big-endian u32, so a BE read recovers the value), or None for a non-UInt32 object.
    fn uint32_value_of(lc: &LayerCommitment, object_id: u64) -> Option<u64> {
        use std::convert::TryInto;
        lc.node_value_catalog
            .iter()
            .find(|r| r.object_id == object_id)
            .and_then(|r| match &r.triple.object {
                CanonicalObject::Value {
                    datatype: CanonicalDatatype::UInt32,
                    lexical,
                } => Some(u32::from_be_bytes(lexical.as_slice().try_into().unwrap()) as u64),
                _ => None,
            })
    }

    fn age_range_plan(op: RangeOp, c: u64) -> LayerBgpPlan {
        LayerBgpPlan {
            atoms: vec![
                LayerBgpAtom::all_variables(AGE.to_vec()).with_object_range(ObjectRangeSpec {
                    op,
                    datatype: fr(UINT32_TAG),
                    c: fr(c),
                    b_bits: 64,
                    second: None,
                }),
                LayerBgpAtom::all_variables(NAME.to_vec()),
            ],
            // x = S(age) == S(name).
            joins: vec![JoinStep {
                acc_key: 0,
                rel_key: 0,
                additional_keys: Vec::new(),
            }],
        }
    }

    // Independently recompute the complete result of
    // `t(?x, age, ?a) [?a op c], t(?x, name, ?n)`: the age relation is the range-filtered
    // [S, O] pairs (in stored order), joined with the name [S, O] pairs on the shared x.
    // Returns the result columns [a, x, n], the final join domain, and the row count.
    fn expected_age_name_join(
        lc: &LayerCommitment,
        op: RangeOp,
        c: u64,
    ) -> (Vec<Vec<ArkFr>>, usize, usize) {
        use crate::proof::join2::{binary_join_domain, compute_binary_join};
        let age_id = resolve_predicate_id(lc, AGE).unwrap();
        let name_id = resolve_predicate_id(lc, NAME).unwrap();
        let (mut age_s, mut age_o) = (Vec::new(), Vec::new());
        for rec in &lc.materialized {
            if rec.predicate_id == age_id {
                if let Some(v) = uint32_value_of(lc, rec.object_id) {
                    if cmp_op(op, v, c) {
                        age_s.push(fr(rec.subject_id));
                        age_o.push(fr(rec.object_id));
                    }
                }
            }
        }
        let (mut name_s, mut name_o) = (Vec::new(), Vec::new());
        for rec in &lc.materialized {
            if rec.predicate_id == name_id {
                name_s.push(fr(rec.subject_id));
                name_o.push(fr(rec.object_id));
            }
        }
        let w = compute_binary_join(&age_s, &[age_o.as_slice()], &name_s, &[name_o.as_slice()]);
        let domain = binary_join_domain(age_s.len(), name_s.len(), w.m_j, w.n_u).unwrap();
        let mut cols = w.r_other_out.clone(); // [a]
        cols.push(w.key.clone()); // x
        cols.extend(w.s_other_out.clone()); // [n]
        (cols, domain, w.m_j)
    }

    // Prove+verify an age-range BGP and assert the committed result equals the
    // independently recomputed filtered+joined answer (count + committed columns).
    fn check_age_range(lc: &LayerCommitment, op: RangeOp, c: u64, expected_m: usize) {
        let plan = age_range_plan(op, c);
        let (proof, coms, m, ids) = prove_layer_bgp(lc, &plan).unwrap();
        verify_layer_bgp(lc, lc.state.commitment_root, &plan, &ids, &coms, m, &proof).unwrap();

        let (exp_cols, domain, exp_m) = expected_age_name_join(lc, op, c);
        assert_eq!(m, exp_m, "op {:?} c {}", op, c);
        assert_eq!(m, expected_m, "op {:?} c {}", op, c);
        assert_eq!(coms.len(), exp_cols.len());
        for (i, col) in exp_cols.iter().enumerate() {
            assert_eq!(
                commit_column(&pad(col, domain)).unwrap(),
                coms[i],
                "op {:?} c {} col {}",
                op,
                c,
                i
            );
        }
    }

    #[test]
    fn layer_bgp_object_range_all_ops_and_boundary_end_to_end() {
        // ages: alice=30, bob=15, carol=18, dave=40, erin=18. c = 18 straddles them and
        // hits the boundary (carol, erin == 18).
        let lc = build_age_name_layer();
        // Gt 18 -> {alice, dave}; Ge 18 -> {alice, carol, dave, erin}.
        check_age_range(&lc, RangeOp::Gt, 18, 2);
        check_age_range(&lc, RangeOp::Ge, 18, 4);
        // Lt 18 -> {bob}; Le 18 -> {bob, carol, erin}.
        check_age_range(&lc, RangeOp::Lt, 18, 1);
        check_age_range(&lc, RangeOp::Le, 18, 3);
    }

    #[test]
    fn layer_bgp_object_range_dict_domain_exceeds_table_end_to_end() {
        // D = 16 (dict) strictly exceeds the table domain 8: the object column relocates
        // UP to D, the dict columns stay at D, and range_sel relocates DOWN to the table.
        let lc = build_distinct_age_name_layer();
        // Sanity: the dict domain is larger than the table domain here.
        let n = lc.materialized.len();
        assert!(dict_domain_len(lc.node_value_catalog.len()).unwrap() > padded_len(n).unwrap());
        // ages p1=10, p2=25, p3=5. age >= 10 -> {p1, p2}.
        check_age_range(&lc, RangeOp::Ge, 10, 2);
    }

    #[test]
    fn layer_bgp_object_range_excludes_everything_m_zero() {
        // Nothing satisfies age > 1000: the ranged relation is empty, so the join is too.
        let lc = build_age_name_layer();
        let plan = age_range_plan(RangeOp::Gt, 1000);
        let (proof, coms, m, ids) = prove_layer_bgp(&lc, &plan).unwrap();
        assert_eq!(m, 0);
        verify_layer_bgp(&lc, lc.state.commitment_root, &plan, &ids, &coms, m, &proof).unwrap();
        let (_cols, _domain, exp_m) = expected_age_name_join(&lc, RangeOp::Gt, 1000);
        assert_eq!(exp_m, 0);
    }

    #[test]
    fn layer_bgp_two_sided_object_range_intersects_both_bounds() {
        // Half-open [18, 40): carol=18, erin=18, alice=30 are included; bob=15 and
        // dave=40 are excluded. The second mask must be present and multiplied with the
        // first before the age relation is compacted.
        let lc = build_age_name_layer();
        let mut plan = age_range_plan(RangeOp::Ge, 18);
        plan.atoms[0].object_range.as_mut().unwrap().second = Some(ObjectRangeBound {
            op: RangeOp::Lt,
            c: fr(40),
        });
        let (proof, coms, m, ids) = prove_layer_bgp(&lc, &plan).unwrap();
        assert_eq!(m, 3);
        verify_layer_bgp(&lc, lc.state.commitment_root, &plan, &ids, &coms, m, &proof).unwrap();

        // The upper bound is part of the public statement and its own authenticated
        // selection. Changing it cannot verify against the original proof.
        let mut wrong_bound = plan.clone();
        wrong_bound.atoms[0]
            .object_range
            .as_mut()
            .unwrap()
            .second
            .as_mut()
            .unwrap()
            .c = fr(41);
        assert!(verify_layer_bgp(
            &lc,
            lc.state.commitment_root,
            &wrong_bound,
            &ids,
            &coms,
            m,
            &proof,
        )
        .is_err());

        let mut missing_second = proof.clone();
        missing_second.atoms[0].range.as_mut().unwrap().second = None;
        assert!(verify_layer_bgp(
            &lc,
            lc.state.commitment_root,
            &plan,
            &ids,
            &coms,
            m,
            &missing_second,
        )
        .is_err());
    }

    #[test]
    fn layer_bgp_subject_object_self_loop_is_same_row_equality() {
        let lc = build_layer(&[
            ("a", "knows", "a"),
            ("a", "knows", "b"),
            ("c", "knows", "c"),
            ("x", "likes", "x"),
        ]);
        let mut atom = LayerBgpAtom::all_variables(KNOWS.to_vec());
        atom.subject_object_equal = true;
        let plan = LayerBgpPlan {
            atoms: vec![atom],
            joins: vec![],
        };
        let (proof, commitments, m, ids) = prove_layer_bgp(&lc, &plan).unwrap();
        assert_eq!(m, 2);
        assert_eq!(commitments.len(), 1);
        verify_layer_bgp(
            &lc,
            lc.state.commitment_root,
            &plan,
            &ids,
            &commitments,
            m,
            &proof,
        )
        .unwrap();

        let mut equality_removed = plan.clone();
        equality_removed.atoms[0].subject_object_equal = false;
        assert!(verify_layer_bgp(
            &lc,
            lc.state.commitment_root,
            &equality_removed,
            &ids,
            &commitments,
            m,
            &proof,
        )
        .is_err());

        let predicate_id = ids[0];
        let mut expected: Vec<ArkFr> = lc
            .materialized
            .iter()
            .filter(|row| row.predicate_id == predicate_id && row.subject_id == row.object_id)
            .map(|row| fr(row.subject_id))
            .collect();
        expected.resize(padded_len(lc.materialized.len()).unwrap(), f0());
        assert_eq!(commitments[0], commit_column(&expected).unwrap());
    }

    #[test]
    fn layer_bgp_object_range_wrong_expected_root_is_rejected() {
        let lc = build_age_name_layer();
        let plan = age_range_plan(RangeOp::Gt, 18);
        let (proof, coms, m, ids) = prove_layer_bgp(&lc, &plan).unwrap();
        let mut wrong_root = lc.state.commitment_root;
        wrong_root[0] ^= 0xff;
        assert!(verify_layer_bgp(&lc, wrong_root, &plan, &ids, &coms, m, &proof).is_err());
        // The correct external root still verifies.
        verify_layer_bgp(&lc, lc.state.commitment_root, &plan, &ids, &coms, m, &proof).unwrap();
    }

    #[test]
    fn layer_bgp_object_range_tampered_constant_or_datatype_is_rejected() {
        let lc = build_age_name_layer();
        let plan = age_range_plan(RangeOp::Gt, 18);
        let (proof, coms, m, ids) = prove_layer_bgp(&lc, &plan).unwrap();

        // Verifying against a DIFFERENT range constant: the constant is bound into the
        // statement AND the object-range sub-proof, so the transcript diverges -> REJECT.
        let plan_wrong_c = age_range_plan(RangeOp::Gt, 19);
        assert!(verify_layer_bgp(
            &lc,
            lc.state.commitment_root,
            &plan_wrong_c,
            &ids,
            &coms,
            m,
            &proof
        )
        .is_err());

        // Verifying against a DIFFERENT datatype tag (Boolean instead of UInt32) -> REJECT.
        let plan_wrong_dt = LayerBgpPlan {
            atoms: vec![
                LayerBgpAtom::all_variables(AGE.to_vec()).with_object_range(ObjectRangeSpec {
                    op: RangeOp::Gt,
                    datatype: fr(BOOLEAN_TAG),
                    c: fr(18),
                    b_bits: 64,
                    second: None,
                }),
                LayerBgpAtom::all_variables(NAME.to_vec()),
            ],
            joins: vec![JoinStep {
                acc_key: 0,
                rel_key: 0,
                additional_keys: Vec::new(),
            }],
        };
        assert!(verify_layer_bgp(
            &lc,
            lc.state.commitment_root,
            &plan_wrong_dt,
            &ids,
            &coms,
            m,
            &proof
        )
        .is_err());

        // A DIFFERENT operator (Ge instead of Gt) -> REJECT (and would also change m).
        let plan_wrong_op = age_range_plan(RangeOp::Ge, 18);
        assert!(verify_layer_bgp(
            &lc,
            lc.state.commitment_root,
            &plan_wrong_op,
            &ids,
            &coms,
            m,
            &proof
        )
        .is_err());

        // Honest still verifies.
        verify_layer_bgp(&lc, lc.state.commitment_root, &plan, &ids, &coms, m, &proof).unwrap();
    }

    #[test]
    fn layer_bgp_object_range_rejects_range_on_a_constant_object() {
        // A range on a CONSTANT object is out of v1 scope (validate_atom rejects it).
        let lc = build_age_name_layer();
        let plan = LayerBgpPlan {
            atoms: vec![
                LayerBgpAtom {
                    subject: AtomTerm::Variable,
                    predicate_term: AGE.to_vec(),
                    predicate_variable: false,
                    object: node_const("alice_n"),
                    object_range: Some(ObjectRangeSpec {
                        op: RangeOp::Gt,
                        datatype: fr(UINT32_TAG),
                        c: fr(18),
                        b_bits: 64,
                        second: None,
                    }),
                    subject_object_equal: false,
                },
                LayerBgpAtom::all_variables(NAME.to_vec()),
            ],
            joins: vec![JoinStep {
                acc_key: 0,
                rel_key: 0,
                additional_keys: Vec::new(),
            }],
        };
        assert!(prove_layer_bgp(&lc, &plan).is_err());
    }
}
