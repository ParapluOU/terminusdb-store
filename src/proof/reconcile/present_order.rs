//! Single-column, full-key, numeric `OrderBy` over the reconciled `present` relation.
//!
//! The first ORDER slice on the delta-stack path: prove `OrderBy([v ASC|DESC], BGP)` where the
//! reconciled `present` BGP binds ONE numeric object column `v` (a fixed-width order-preserving
//! `UInt`/`Int`/`Boolean`). It is built by REUSE, composing three already-audited layers:
//!
//!  1. RECONCILE + PRESENT-BGP ([`prove_present_bgp`]) authenticates `present` against the trusted
//!     head root and delivers the single kept column of combined-dict node/value ids for `v` — the
//!     order gadget's INPUT relation. Its commitment is the same the BGP proved (the verifier feeds
//!     the BGP-authenticated result commitment as the permutation input, never a fresh one).
//!  2. The order gadget proves the OUTPUT is a full-row PERMUTATION of that input
//!     ([`crate::proof::permutation`]) — same rows, reordered, no fabricated/dropped row.
//!  3. The SORT KEY is each output id's ORDER-PRESERVING dictionary VALUE, authenticated across the
//!     chain against the COMBINED `(id, value, datatype)` range dictionary ([`prove_combined_range_dict`]),
//!     exactly as a numeric RANGE over `present` authenticates its object values (C2a). The gadget
//!     proves the authenticated value column is NONDECREASING ([`crate::proof::sort`]); descending
//!     sorts the affine `u64::MAX − value` ascending. This is the FIDELITY crux: the committed
//!     object-id is NOT value-ordered on rollup/materialized layers, so sorting the raw id would give
//!     a wrong order vs TerminusDB — the nondecreasing chain MUST run over the authenticated VALUE
//!     column (with its DATATYPE authenticated too), never the raw id.
//!
//! F4-MAX discipline (mirrors the base path [`crate::proof::order`]): ONLY a full-key numeric order
//! compiles. Float/NaN keys are reject-closed (the store IEEE total-order float key diverges from
//! WOQL numeric order on `-0.0`/`NaN`), and the woql2 compiler only ships a single-column full-key
//! numeric `ExactScalarOrder` here; a subset-key/multi-key/non-numeric order never reaches this
//! gadget. Term-less value↔term authentication reuses the C2a chain opening
//! ([`verify_dict_value_terms_chain`]); a term-less order without openings fails CLOSED.

use super::*;
use crate::proof::commitment::build_resolution_openings;
use crate::proof::order::OrderDirection;
use crate::proof::range::{
    honest_dict_lookup, prove_dict_value_lookup_with_context, verify_dict_value_lookup,
    DictLookupProof,
};
use crate::proof::sort::{prove_nondecreasing_bits_with_context, verify_nondecreasing_bits};
use crate::proof::CanonicalDatatype;

const PRESENT_ORDER_DOMAIN: &[u8] = b"parture/proof-of-woql/present-scalar-order/v0";
/// The order-preserving dictionary VALUE encodings of the supported numeric datatypes are all
/// `< 2^64` (same bound the RANGE gadget's `OBJECT_RANGE_B_BITS` uses), so the nondecreasing sort
/// key is 64-bit.
const PRESENT_ORDER_BITS: usize = 64;

/// The fixed-width order-preserving numeric datatypes a scalar order may sort by. Float32/Float64
/// are DELIBERATELY excluded (the store IEEE total-order float key sorts `-0.0` below `+0.0` and
/// `NaN` above `+∞`, diverging from WOQL numeric order); mirrors [`crate::proof::order`]'s
/// `supported_datatype` and the woql2 compiler's reject-close, so this gadget never receives a float
/// key in production and stays that way as defense-in-depth.
pub(super) fn supported_order_datatype(datatype: CanonicalDatatype) -> bool {
    matches!(
        datatype,
        CanonicalDatatype::Boolean
            | CanonicalDatatype::UInt8
            | CanonicalDatatype::UInt16
            | CanonicalDatatype::UInt32
            | CanonicalDatatype::UInt64
            | CanonicalDatatype::Int8
            | CanonicalDatatype::Int16
            | CanonicalDatatype::Int32
            | CanonicalDatatype::Int64
    )
}

/// A proof that the ordered output is a full-row permutation of the reconciled BGP result relation
/// which is NONDECREASING by each row's chain-authenticated order-preserving VALUE: the permutation
/// subproof, the output id-column commitment (the permutation output, at the working domain), the
/// combined chain `(id, value, datatype)` range dictionary, the authenticated dict lookup binding
/// `com_v_val`/`com_v_dt`, the datatype-completeness filter, and the nondecreasing-order proof over
/// the (possibly affine-complemented) sort key.
#[derive(Clone, Debug)]
pub struct PresentScalarOrderProof {
    pub permutation: RelationPermutationProof,
    pub com_output: ArkGT,
    pub range_dict: CombinedRangeDictProof,
    pub lookup: DictLookupProof,
    pub datatype_filter: FilterProof,
    pub datatype_selection_commitment: ArkGT,
    pub datatype_inverse_commitment: ArkGT,
    pub sorted: SortedProof,
    /// C2a value↔term witness for a TERM-LESS order: the SORTED output id column's real ids
    /// (`oid[i]` = the id at sorted row `i`) plus one Merkle opening per DISTINCT id, re-verified
    /// against the root-bound `nv_merkle_root` by `verify_dict_value_terms_chain`. EMPTY / `None`
    /// on a TERM-BEARING artifact (the catalog re-MSM ties the sort key to real terms). UNCOMMITTED —
    /// never folded into `commitment_root`/`resolution_root`; a missing opening (or recommit
    /// mismatch) fails closed. Also serves the ordered result column's RESOLUTION (same id set).
    pub oid: Vec<u64>,
    pub value_openings: Option<ResolutionOpenings>,
}

fn bind_present_order_statement(
    transcript: &mut Transcript254,
    datatype: CanonicalDatatype,
    direction: OrderDirection,
    n: usize,
    working_log: usize,
) {
    transcript.append_bytes(b"pow/present-order/domain", PRESENT_ORDER_DOMAIN);
    transcript.append_bytes(b"pow/present-order/datatype", &[datatype as u8]);
    transcript.append_bytes(
        b"pow/present-order/direction",
        &[match direction {
            OrderDirection::Ascending => 0,
            OrderDirection::Descending => 1,
        }],
    );
    transcript.append_bytes(b"pow/present-order/rows", &(n as u64).to_le_bytes());
    transcript.append_bytes(
        b"pow/present-order/working-log",
        &(working_log as u64).to_le_bytes(),
    );
}

/// Prove a single-column full-key numeric order of `input` (one combined-dict node/value id column
/// at the working domain, the reconciled BGP's authenticated result relation). Returns the proof
/// and the ordered output id column (real prefix). The combined range dict and `input` already live
/// at the working domain, so no relocation is needed (unlike the base single-layer order gadget).
#[allow(clippy::too_many_arguments)]
pub(super) fn prove_present_scalar_order(
    context: &StoreProverContext<'_>,
    layers: &[&LayerCommitment],
    artifacts: &ReconcileArtifacts,
    input: &RelationWitness,
    head_nv: usize,
    datatype: CanonicalDatatype,
    direction: OrderDirection,
    transcript: &mut Transcript254,
) -> io::Result<(PresentScalarOrderProof, Vec<ArkFr>)> {
    use ark_ff::PrimeField;
    if !supported_order_datatype(datatype) {
        return Err(invalid(
            "present order requires a fixed-width order-preserving numeric datatype (no float/string)",
        ));
    }
    if input.columns.len() != 1 || input.commitments.len() != 1 {
        return Err(invalid(
            "present order requires a single-column (full-key) BGP result relation",
        ));
    }
    if input.log_domain != artifacts.working_log {
        return Err(invalid(
            "present order: the BGP result relation must be at the reconcile working domain",
        ));
    }
    let working_log = artifacts.working_log;
    let domain = 1usize << working_log;
    let n = input.n;
    let expected_dt = ArkFr::from_u64(datatype as u64);

    // (1) The combined `(id, value, datatype)` range dictionary across the chain (root-bound,
    // relocated to the working domain) — the SAME authenticated value/datatype columns a numeric
    // RANGE over `present` uses. This is what makes the sort key the ORDER-PRESERVING VALUE, not the
    // raw committed id.
    let branches = range_dict_branches(layers)?;
    let dict = prove_combined_range_dict(context, &branches, working_log, transcript)?;

    // (2) Look up each input id's (value, datatype) to derive the sort order and to reject a mixed
    // datatype (a non-`datatype` value must not sit in a single-datatype order).
    let input_col = &input.columns[0];
    let (in_val, in_dt, _w) = honest_dict_lookup(
        input_col,
        &dict.id_at_k,
        &dict.value_at_k,
        &dict.datatype_at_k,
        n,
        head_nv,
    );
    if in_dt[..n].iter().any(|d| *d != expected_dt) {
        return Err(invalid("present order input contains another datatype"));
    }
    let mut row_order: Vec<usize> = (0..n).collect();
    row_order.sort_by(|a, b| in_val[*a].0.into_bigint().cmp(&in_val[*b].0.into_bigint()));
    if direction == OrderDirection::Descending {
        row_order.reverse();
    }

    // (3) The full-row permutation input -> sorted output (rows preserved, none fabricated/dropped).
    bind_present_order_statement(transcript, datatype, direction, n, working_log);
    let permutation = prove_relation_permutation_with_context(context, input, &row_order, transcript)?;
    let com_output = permutation.commitments[0];
    let output_col = permutation.columns[0].clone(); // padded to the working domain

    // (4) Authenticate each OUTPUT id's (value, datatype) against the combined range dict — binds
    // `lookup.com_v_val`/`lookup.com_v_dt` to the TRUE dict value/datatype of each sorted row.
    let lookup = prove_dict_value_lookup_with_context(
        context,
        &output_col,
        &com_output,
        &dict.id_at_k,
        &dict.value_at_k,
        &dict.datatype_at_k,
        &dict.com_id_at_k,
        &dict.com_value_at_k,
        &dict.com_datatype_at_k,
        n,
        head_nv,
        transcript,
    )?;

    // (5) Every output row is EXACTLY `datatype` (a complete datatype selection over the real rows),
    // so the sort key is single-datatype and the affine descending transform is well-defined.
    let (_out_val0, out_dt, _w2) = honest_dict_lookup(
        &output_col,
        &dict.id_at_k,
        &dict.value_at_k,
        &dict.datatype_at_k,
        n,
        head_nv,
    );
    let (datatype_filter, datatype_selection_commitment, datatype_inverse_commitment, _) =
        prove_filter_indicator_with_context(context, &out_dt, expected_dt, n, &lookup.com_v_dt, transcript)?;
    let mut chi = vec![f0(); domain];
    chi[..n].fill(f1());
    if datatype_selection_commitment != context.commit_column(&chi)? {
        return Err(invalid("present order datatype selection is not complete"));
    }

    // (6) The authenticated value column is NONDECREASING (descending: sort `u64::MAX - value`).
    let (out_val, _out_dt2, _w3) = honest_dict_lookup(
        &output_col,
        &dict.id_at_k,
        &dict.value_at_k,
        &dict.datatype_at_k,
        n,
        head_nv,
    );
    let (sort_values, sort_commitment) = match direction {
        OrderDirection::Ascending => (out_val.clone(), lookup.com_v_val),
        OrderDirection::Descending => {
            let max = ArkFr::from_u64(u64::MAX);
            let sv: Vec<ArkFr> = (0..domain)
                .map(|i| if i < n { max - out_val[i] } else { f0() })
                .collect();
            (sv, max * datatype_selection_commitment - lookup.com_v_val)
        }
    };
    if context.commit_column(&sort_values)? != sort_commitment {
        return Err(invalid("present order affine sort-key commitment mismatch"));
    }
    let sorted = prove_nondecreasing_bits_with_context(
        context,
        &sort_values,
        n,
        PRESENT_ORDER_BITS,
        &sort_commitment,
        transcript,
    )?;

    let ordered_col = output_col[..n].to_vec();
    Ok((
        PresentScalarOrderProof {
            permutation: permutation.proof,
            com_output,
            range_dict: dict.proof,
            lookup,
            datatype_filter,
            datatype_selection_commitment,
            datatype_inverse_commitment,
            sorted,
            // Prover default: the term-less producer attaches these separately via
            // `attach_termless_present_order_openings`; a term-bearing artifact leaves them empty.
            oid: Vec::new(),
            value_openings: None,
        },
        ordered_col,
    ))
}

/// Verify a single-column full-key numeric order proof over the reconciled BGP result, mirroring
/// [`prove_present_scalar_order`] on the shared transcript in prover order. `input_com` is the
/// BGP-AUTHENTICATED result commitment (from [`verify_present_bgp`]) — the permutation input, never
/// a fresh one. On success returns `(ordered output commitments, n, working_log)`.
#[allow(clippy::too_many_arguments)]
pub(super) fn verify_present_scalar_order(
    views: &[VerifierCommitmentRef<'_>],
    verified: &ReconcileVerified,
    input_com: &ArkGT,
    n: usize,
    head_nv: usize,
    datatype: CanonicalDatatype,
    direction: OrderDirection,
    proof: &PresentScalarOrderProof,
    transcript: &mut Transcript254,
) -> io::Result<(Vec<ArkGT>, usize, usize)> {
    if !supported_order_datatype(datatype) {
        return Err(invalid(
            "present order requires a fixed-width order-preserving numeric datatype (no float/string)",
        ));
    }
    let working_log = verified.working_log;
    let domain = 1usize << working_log;
    let expected_dt = ArkFr::from_u64(datatype as u64);

    // (1) The combined range dictionary across the chain.
    let branches = range_dict_branch_commits(views)?;
    verify_combined_range_dict(&branches, head_nv, working_log, &proof.range_dict, transcript)?;

    // (2) Output is the input's rows reordered — the input commitment is the BGP-authenticated one.
    bind_present_order_statement(transcript, datatype, direction, n, working_log);
    let input = RelationCommit {
        commitments: vec![*input_com],
        log_domain: working_log,
        n,
    };
    let output_commitments = [proof.com_output];
    verify_relation_permutation(&input, &output_commitments, n, working_log, &proof.permutation, transcript)?;

    // (2t) C2a value↔term tie for a TERM-LESS view: recommit the shipped `oid` to the output id
    // commitment and rederive value/datatype from the chain-opened terms, forcing
    // `lookup.com_v_val`/`lookup.com_v_dt` == OP-encode(real term) at every sorted row. No openings
    // ⇒ fail CLOSED. Transcript-free; runs before the (transcript-consuming) lookup. Term-bearing:
    // the catalog re-MSM ties them (a `reject_nan_catalog` no-op for these non-float datatypes).
    let all_terms_present = views.iter().all(|v| v.terms_present);
    if !all_terms_present {
        let openings = proof.value_openings.as_ref().ok_or_else(|| {
            invalid("term-less present order requires the prover-shipped value<->term Merkle openings")
        })?;
        let touched = verify_dict_value_terms_chain(
            views,
            &proof.com_output,
            &proof.lookup.com_v_val,
            &proof.lookup.com_v_dt,
            n,
            domain,
            &proof.oid,
            openings,
        )?;
        reject_nan_objects(&touched, expected_dt)?;
    } else {
        for view in views {
            reject_nan_catalog(view.node_value_catalog, expected_dt)?;
        }
    }

    // (3) The authenticated (value, datatype) lookup over the output ids.
    verify_dict_value_lookup(
        &proof.com_output,
        &proof.range_dict.com_id_at_k,
        &proof.range_dict.com_value_at_k,
        &proof.range_dict.com_datatype_at_k,
        n,
        head_nv,
        &proof.lookup,
        transcript,
    )?;

    // (4) Every output row is `datatype` (complete datatype selection).
    verify_filter_indicator(
        &proof.lookup.com_v_dt,
        expected_dt,
        n,
        &proof.datatype_selection_commitment,
        &proof.datatype_inverse_commitment,
        &proof.datatype_filter,
        transcript,
    )?;
    let mut chi = vec![f0(); domain];
    chi[..n].fill(f1());
    if proof.datatype_selection_commitment != commit_column(&chi)? {
        return Err(invalid("present order datatype selection is not complete"));
    }

    // (5) The authenticated value column is nondecreasing (descending: affine `u64::MAX - value`).
    let sort_commitment = match direction {
        OrderDirection::Ascending => proof.lookup.com_v_val,
        OrderDirection::Descending => {
            ArkFr::from_u64(u64::MAX) * proof.datatype_selection_commitment - proof.lookup.com_v_val
        }
    };
    verify_nondecreasing_bits(&sort_commitment, n, PRESENT_ORDER_BITS, &proof.sorted, transcript)?;

    Ok((vec![proof.com_output], n, working_log))
}

// ===========================================================================
// Top-level entry points: prove / verify+resolve a reconciled single-column full-key numeric
// OrderBy over the materialized `present`. Mirrors `prove_read_bgp` / `verify_and_resolve_bgp`.
// ===========================================================================

/// Prove `OrderBy([v ASC|DESC], BGP)` over an N-layer delta stack, where the reconciled `present`
/// BGP binds ONE numeric object column `v`. Reconcile + present-BGP + order share ONE transcript, so
/// the ordered answer is bound to the reconciliation-authenticated relation. Returns the reconcile
/// proof, the BGP proof, the order proof, and the ORDERED result column (combined-dict ids;
/// resolution deferred to [`verify_and_resolve_order_bgp`]).
pub fn prove_read_order_bgp(
    layers: &[&LayerCommitment],
    atoms: &[PresentBgpAtom],
    joins: &[JoinStep],
    datatype: CanonicalDatatype,
    direction: OrderDirection,
) -> io::Result<(ReconcileProof, PresentBgpProof, PresentScalarOrderProof, PresentReadResult)> {
    let context = StoreProverContext::cpu();
    let mut transcript = Transcript254::new(b"parture-delta-reconcile-envelope");
    let (reconcile, artifacts) = prove_reconcile_core(&context, layers, &mut transcript)?;
    let (bgp, bgp_result) =
        prove_present_bgp(&context, layers, &artifacts, atoms, joins, &mut transcript)?;
    // Full-key single-column numeric order: the BGP must expose exactly one column at the working
    // domain (a subset/multi-key or wider-arity order is reject-closed upstream and here).
    if bgp.result_arity != 1 {
        return Err(invalid(
            "present order: OrderBy requires a single-column (full-key) BGP result",
        ));
    }
    if bgp.result_log_domain != artifacts.working_log {
        return Err(invalid(
            "present order: the BGP result relation must be at the reconcile working domain",
        ));
    }
    let working_log = artifacts.working_log;
    let domain = 1usize << working_log;
    let n = bgp.result_len;
    let head_nv = layers.last().unwrap().state.node_value_count as usize;
    let mut col = bgp_result.columns[0].clone();
    col.resize(domain, f0());
    let com_input = context.commit_column(&col)?;
    let input = RelationWitness {
        columns: vec![col],
        commitments: vec![com_input],
        log_domain: working_log,
        n,
    };
    let (order, ordered_col) = prove_present_scalar_order(
        &context, layers, &artifacts, &input, head_nv, datatype, direction, &mut transcript,
    )?;
    let result = PresentReadResult { columns: vec![ordered_col], len: n, id_hints: Vec::new() };
    Ok((reconcile, bgp, order, result))
}

/// C2a (producer): attach the TERM-LESS value↔term witness to a proven order proof, so the ordered
/// result resolves+verifies over a TERM-LESS artifact. `ordered_ids` is the prover's ORDERED output
/// id column (`PresentReadResult::columns[0]`). Records them as `oid` and opens each DISTINCT id
/// against the node/value tree. A term-bearing verifier ignores both fields; leaving them unset
/// keeps a term-less order fail-closed. Mirrors `build_termless_order_openings`.
pub fn attach_termless_present_order_openings(
    proof: &mut PresentScalarOrderProof,
    layers: &[&LayerCommitment],
    ordered_ids: &[ArkFr],
) -> io::Result<()> {
    let mut oid: Vec<u64> = Vec::with_capacity(ordered_ids.len());
    let mut nv_ids: Vec<u64> = Vec::new();
    for value in ordered_ids {
        let raw = canonical_dictionary_id(value)?;
        oid.push(raw);
        if raw != 0 {
            nv_ids.push(raw);
        }
    }
    let views: Vec<VerifierCommitmentRef<'_>> = layers.iter().map(|l| l.verifier_view()).collect();
    let openings = build_resolution_openings(&views, &nv_ids, &[])?;
    proof.oid = oid;
    proof.value_openings = Some(openings);
    Ok(())
}

/// Verify a reconciled single-column full-key numeric order read (count only). Thin wrapper over
/// [`verify_read_order_bgp_from_views`] that builds term-bearing views from the full sidecars.
#[allow(clippy::too_many_arguments)]
pub fn verify_read_order_bgp(
    layers: &[&LayerCommitment],
    trusted_head_root: ProofHash,
    atoms: &[PresentBgpAtom],
    joins: &[JoinStep],
    datatype: CanonicalDatatype,
    direction: OrderDirection,
    reconcile: &ReconcileProof,
    bgp: &PresentBgpProof,
    order: &PresentScalarOrderProof,
) -> io::Result<usize> {
    let views: Vec<VerifierCommitmentRef<'_>> = layers.iter().map(|l| l.verifier_view()).collect();
    let (_coms, n, _log) = verify_read_order_bgp_from_views(
        &views, trusted_head_root, atoms, joins, datatype, direction, reconcile, bgp, order,
    )?;
    Ok(n)
}

/// Verify a reconciled single-column full-key numeric order read FROM verifier VIEWS. Reconcile +
/// present-BGP + order share ONE transcript; the order's permutation input is the BGP-authenticated
/// result commitment. Returns `(ordered output commitments, n, working_log)`.
#[allow(clippy::too_many_arguments)]
pub fn verify_read_order_bgp_from_views(
    views: &[VerifierCommitmentRef<'_>],
    trusted_head_root: ProofHash,
    atoms: &[PresentBgpAtom],
    joins: &[JoinStep],
    datatype: CanonicalDatatype,
    direction: OrderDirection,
    reconcile: &ReconcileProof,
    bgp: &PresentBgpProof,
    order: &PresentScalarOrderProof,
) -> io::Result<(Vec<ArkGT>, usize, usize)> {
    let mut transcript = Transcript254::new(b"parture-delta-reconcile-envelope");
    let verified = verify_reconcile_core(views, trusted_head_root, reconcile, &mut transcript)?;
    let head_nv = views.last().unwrap().state.node_value_count as usize;
    let (result_coms, result_m, result_log_domain) = verify_present_bgp(
        views, &reconcile.combined_nv, &reconcile.combined_pred, &verified, head_nv, atoms, joins,
        bgp, &mut transcript,
    )?;
    if result_coms.len() != 1 {
        return Err(invalid(
            "present order: OrderBy requires a single-column (full-key) BGP result",
        ));
    }
    if result_log_domain != verified.working_log {
        return Err(invalid(
            "present order: the BGP result relation must be at the reconcile working domain",
        ));
    }
    verify_present_scalar_order(
        views, &verified, &result_coms[0], result_m, head_nv, datatype, direction, order,
        &mut transcript,
    )
}

/// Verify a reconciled single-column full-key numeric order read AND resolve the ordered result
/// bindings to `CanonicalObject`s (the readable, ordered answer). Thin wrapper over
/// [`verify_and_resolve_order_bgp_from_views`] that builds term-bearing views.
#[allow(clippy::too_many_arguments)]
pub fn verify_and_resolve_order_bgp(
    layers: &[&LayerCommitment],
    trusted_head_root: ProofHash,
    atoms: &[PresentBgpAtom],
    joins: &[JoinStep],
    datatype: CanonicalDatatype,
    direction: OrderDirection,
    reconcile: &ReconcileProof,
    bgp: &PresentBgpProof,
    order: &PresentScalarOrderProof,
    claimed_ordered_columns: &[Vec<ArkFr>],
) -> io::Result<Vec<Vec<CanonicalObject>>> {
    let views: Vec<VerifierCommitmentRef<'_>> = layers.iter().map(|l| l.verifier_view()).collect();
    verify_and_resolve_order_bgp_from_views(
        &views, trusted_head_root, atoms, joins, datatype, direction, reconcile, bgp, order,
        claimed_ordered_columns,
    )
}

/// Views-based analogue of [`verify_and_resolve_order_bgp`]. The single ordered result column is a
/// combined-dict node/value id column; resolution over a TERM-LESS view consumes the prover-shipped
/// `order.value_openings` (the SAME opening set as the C2a value tie — the ordered ids are a
/// permutation of the input ids), each re-verified against the root-bound `nv_merkle_root`. A
/// term-bearing artifact leaves `value_openings` `None` and the resolver rebuilds from catalogs.
#[allow(clippy::too_many_arguments)]
pub fn verify_and_resolve_order_bgp_from_views(
    views: &[VerifierCommitmentRef<'_>],
    trusted_head_root: ProofHash,
    atoms: &[PresentBgpAtom],
    joins: &[JoinStep],
    datatype: CanonicalDatatype,
    direction: OrderDirection,
    reconcile: &ReconcileProof,
    bgp: &PresentBgpProof,
    order: &PresentScalarOrderProof,
    claimed_ordered_columns: &[Vec<ArkFr>],
) -> io::Result<Vec<Vec<CanonicalObject>>> {
    let (ordered_coms, ordered_n, ordered_log) = verify_read_order_bgp_from_views(
        views, trusted_head_root, atoms, joins, datatype, direction, reconcile, bgp, order,
    )?;
    let kinds = [ResolveKind::NodeValueId];
    resolve_result_columns(
        views,
        claimed_ordered_columns,
        &ordered_coms,
        ordered_n,
        ordered_log,
        &kinds,
        None,
        order.value_openings.as_ref(),
    )
}

// ============================================================================================
// C4 ORDER byte-in: DB-free reconciled single-column numeric OrderBy over the TERM-LESS chain.
//
// Unlike a plain/count read (which delivers only a row count), an OrderBy RESOLVES the ordered rows,
// so — like the group path — the byte-in bundle ships the reconcile + present-BGP proof PLUS the
// order proof PLUS the delivered ORDERED result column, and the prover mints the term-less openings:
//   * the child BGP's numeric range atom (the `?v >= 0` datatype pin) needs the value<->term RANGE
//     openings over the reconciled present object column for term-less BGP verification;
//   * the order gadget's value<->term tie AND the ordered-result RESOLUTION both consume the ORDERED
//     output ids' openings (`attach_termless_present_order_openings`).
// The order DIRECTION/DATATYPE come from the VERIFIER's re-lowered query — never the proof bytes: the
// crypto transcript binds them (a wrong direction/datatype diverges Fiat-Shamir and rejects), and the
// wire-carried `OrderReadSpec` is COMPARED to the verifier's spec as an early anti-substitution belt
// (the byte analogue of the in-memory `expected.order == proof.order` check).
// ============================================================================================

/// The `OrderBy` spec an order read was PROVED for (sort datatype + direction). Carried ONLY as an
/// anti-substitution witness: the byte-in verifier compares it to its OWN re-lowered query spec and
/// reject-closes on mismatch. The recompute/verify always uses the VERIFIER's params (which the
/// transcript also binds). Mirrors the compiler's `OrderSpec`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrderReadSpec {
    pub datatype: CanonicalDatatype,
    pub direction: OrderDirection,
}

/// The shippable reconciled single-column numeric `OrderBy` read: the [`ReconcileProof`] + the child
/// [`PresentBgpProof`] + the [`PresentScalarOrderProof`] + the delivered ORDERED result column + the
/// [`OrderReadSpec`] witness. Its `encode`/`decode` (framed MAGIC ‖ VERSION ‖ body ‖ SHA-256 with the
/// independent `ORDER_READ_*` constants) is transparent to verification: the decoded proof re-verifies
/// through the SAME `verify_and_resolve_order_bgp_from_views` over the verifier's OWN spec.
#[derive(Clone, Debug)]
pub struct ReconciledOrderReadProof {
    pub reconcile: ReconcileProof,
    pub bgp: PresentBgpProof,
    pub order: PresentScalarOrderProof,
    pub ordered_columns: Vec<Vec<ArkFr>>,
    pub spec: OrderReadSpec,
}

/// Prover-side: mint ALL term-less resolution witnesses a byte-in ORDER read needs into the proven
/// child BGP + order proofs, so the SAME proof also verifies + RESOLVES the ordered rows DB-free over
/// a TERM-LESS `VerifierCommitment` chain. Mirrors [`attach_termless_group_openings`]:
///   * for a ranged child (the `?v >= 0` datatype pin), the value↔term RANGE openings over the
///     reconciled present object column (on the BGP proof);
///   * the ORDERED output ids' openings (on the order proof) — they serve BOTH the order gadget's
///     value tie and the ordered-result resolution.
/// A term-bearing verifier ignores every witness; leaving them unattached keeps a term-less order
/// fail-closed. Call once after proving.
pub fn attach_termless_order_openings(
    bgp: &mut PresentBgpProof,
    order: &mut PresentScalarOrderProof,
    layers: &[&LayerCommitment],
    atoms: &[PresentBgpAtom],
    ordered_ids: &[ArkFr],
) -> io::Result<()> {
    if atoms.iter().any(|a| a.object_range.is_some()) {
        let poids: Vec<ArkFr> =
            reconcile_present(layers)?.iter().map(|r| ark(r.object_id)).collect();
        attach_termless_present_range_openings(bgp, layers, &poids)?;
    }
    attach_termless_present_order_openings(order, layers, ordered_ids)?;
    Ok(())
}

/// C4 byte-in remote-verifier entry: verify + resolve a reconciled single-column numeric `OrderBy`
/// read shipped as BYTES against a trusted head root, holding ONLY the encoded proof + the encoded
/// term-less `VerifierCommitment` chain. `atoms`/`joins`/`datatype`/`direction`/`trusted_head_root`
/// are the VERIFIER's OWN parameters (anti-substitution): NEVER parsed from the proof bytes. The
/// wire-carried `OrderReadSpec` is compared to `(datatype, direction)` and reject-closes on mismatch;
/// the audited [`verify_and_resolve_order_bgp_from_views`] then re-verifies + resolves the ordered
/// rows (the transcript also binds direction/datatype, so a wrong direction independently rejects).
#[allow(clippy::too_many_arguments)]
pub fn decode_and_verify_order(
    order_proof_bytes: &[u8],
    verifier_commitment_bytes: &[&[u8]],
    trusted_head_root: ProofHash,
    atoms: &[PresentBgpAtom],
    joins: &[JoinStep],
    datatype: CanonicalDatatype,
    direction: OrderDirection,
) -> io::Result<Vec<Vec<CanonicalObject>>> {
    let owned: Vec<crate::proof::commitment::VerifierCommitment> = verifier_commitment_bytes
        .iter()
        .map(|b| crate::proof::commitment::VerifierCommitment::decode(b))
        .collect::<io::Result<_>>()?;
    let proof = ReconciledOrderReadProof::decode(order_proof_bytes)?;
    // Anti-substitution belt: the wire-carried order spec must equal the VERIFIER's re-lowered spec.
    let expected = OrderReadSpec { datatype, direction };
    if proof.spec != expected {
        return Err(invalid(
            "reconciled order byte-in: the proof's order spec (datatype/direction) does not match the verifier's query (anti-substitution)",
        ));
    }
    let views: Vec<VerifierCommitmentRef<'_>> = owned.iter().map(|c| c.verifier_view()).collect();
    verify_and_resolve_order_bgp_from_views(
        &views,
        trusted_head_root,
        atoms,
        joins,
        datatype,
        direction,
        &proof.reconcile,
        &proof.bgp,
        &proof.order,
        &proof.ordered_columns,
    )
}
