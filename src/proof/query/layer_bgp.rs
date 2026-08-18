//! Q_k — the general k-atom conjunctive BGP entry points (prove/verify a whole query).
//!
//! This is the top of the layer-query stack: given a validated [`LayerBgpPlan`], it proves
//! (or verifies) the COMPLETE result of a k-atom conjunctive graph pattern over one committed
//! layer. It drives everything the other modules provide — per-atom dictionary membership and
//! filtering (plain or ranged), then the left-deep binary-join chain over the filtered
//! relations — all under one shared transcript anchored to `commitment_root`. It also handles
//! the special shapes: a single atom (no join needed), an all-ground conjunction (an
//! authenticated yes/no cardinality), and dictionary-absence (a constant missing from the layer
//! means the
//! whole conjunction is provably empty).

use super::*;

const LAYER_BGP_DOMAIN: &[u8] = b"parture/proof-of-woql/qk-layer-bgp-by-iri/v1";
const NODE_DATATYPE_TAG: u64 = 255;

/// Prove the COMPLETE result of a k-atom conjunctive BGP against a materialized head
/// layer. Each atom is filtered by its dictionary-authenticated constants into its dense
/// KEPT columns; predicate variables retain P directly. The resulting `k` relations (one
/// column per variable, in S,P,O order) are joined
/// by the left-deep binary-join chain ([`crate::proof::bgp`]) on the plan's shared variables — all
/// under ONE shared transcript anchored to `commitment_root`. Returns the combined proof
/// (whose per-atom parts carry the revealed subject/object constant ids), the final result
/// column commitments (schema `[r_other_out.., key, s_other_out..]`), the result length `m`,
/// and the REVEALED resolved predicate ids (authenticated by membership).
pub fn prove_layer_bgp(
    layer: &LayerCommitment,
    plan: &LayerBgpPlan,
) -> io::Result<(LayerBgpProof, Vec<ArkGT>, usize, Vec<u64>)> {
    let result = prove_layer_bgp_with_result_and_context(&StoreProverContext::cpu(), layer, plan)?;
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
    prove_layer_bgp_with_result_and_context(&StoreProverContext::cpu(), layer, plan)
}

pub fn prove_layer_bgp_with_result_and_context(
    context: &StoreProverContext<'_>,
    layer: &LayerCommitment,
    plan: &LayerBgpPlan,
) -> io::Result<LayerBgpResult> {
    require_base_layer_for_query(&layer.state)?;
    let k = validate_bgp_plan(plan)?;
    // Plan-DEFAULT columns/count. Each RELATIONAL atom RE-sources its own `(s, p, o, n)` from
    // `atom.source` inside the loop below (that is what admits a cross-source join); the
    // plan default is used only where no single atom relation is read: the all-ground UNIT
    // column domain and the dictionary-absence zero column.
    let (_s0, _p0, _o0, n0) = source_layer_columns(layer, plan.source)?;

    if let Some(absence) = dictionary_absence(layer, plan) {
        // A root-authenticated dictionary omits one public constant, so the atom and the
        // whole conjunction are empty. Emit canonical zero commitments at the layer table
        // domain, preserving either the public relational arity or the hidden ground UNIT.
        let domain = padded_len(n0)?;
        let zero_column = vec![f0(); domain];
        let result_arity = result_id_namespaces(plan)?.len().max(1);
        let zero_commitment = context.commit_column(&zero_column)?;
        return Ok(LayerBgpResult {
            proof: LayerBgpProof {
                atoms: Vec::new(),
                chain: None,
                dictionary_absence: Some(absence),
                // Term-bearing by default; the S5 term-less producer attaches openings after the
                // fact. (Dictionary-absence anyway fail-closes term-less — `require_terms_present`.)
                openings: None,
            },
            result_commitments: vec![zero_commitment; result_arity],
            result_columns: vec![zero_column; result_arity],
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
    // Plan-default table-domain log, used ONLY for the all-ground UNIT column (no relational
    // atom is joined there). Relational atoms use their OWN `a` (from their own source) below.
    let a0 = padded_len(n0)?.trailing_zeros() as usize;

    let mut transcript = Transcript254::new(LAYER_BGP_DOMAIN);
    bind_layer_bgp_statement(&mut transcript, layer, plan, &atom_ids);

    let mut atom_proofs = Vec::with_capacity(k);
    let mut relations: Vec<RelationWitness> = Vec::with_capacity(k);
    for (atom, ids) in plan.atoms.iter().zip(atom_ids.iter()) {
        // (0) PER-ATOM column sourcing: this atom's `(s, p, o, n)`, its three root-bound
        // `(S, P, O)` column commitments, and its table-domain log `a` all come from THIS
        // atom's `source` — materialized head, or the layer's own additions / removals delta.
        // Every set is bound by the SAME `commitment_root`, and all sources share the predicate
        // and node/value dictionaries, so a cross-source atom's filter/join is the audited
        // gadget unchanged (only which committed columns feed it differs). The real length `n`
        // is PINNED to this atom's committed `source_count`.
        let (s, p, o, n) = source_layer_columns(layer, atom.source)?;
        let [com_s, com_p, com_o] = source_column_commitments(layer.verifier_view(), atom.source);
        let a = padded_len(n)?.trailing_zeros() as usize;
        // (1) Predicate IRI -> id, authenticated by the predicate dictionary.
        let predicate_membership = match ids.predicate_id {
            Some(predicate_id) => Some(prove_dict_membership_with_context(
                context, &pred_dict_hash,
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
            (Some(object), Some(subject_id)) => Some(prove_dict_membership_with_context(
                context, &nv_dict_hash,
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
            (Some(object), Some(object_id)) => Some(prove_dict_membership_with_context(
                context, &nv_dict_hash,
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
        let node_kind_range = ObjectRangeSpec {
            op: RangeOp::Ge, datatype: ArkFr::from_u64(NODE_DATATYPE_TAG),
            c: ArkFr::zero(), b_bits: 64, second: None,
        };
        let (effective_range, complement_range) = match atom.object_kind {
            ObjectKindFilter::Any => (atom.object_range.as_ref(), false),
            ObjectKindFilter::Node => (Some(&node_kind_range), false),
            ObjectKindFilter::Literal => (Some(&node_kind_range), true),
        };
        let (filter, range, kept_commitments, kept_columns, m) = match effective_range {
            None => {
                let (filter, kept_commitments, m) = prove_atom_filter_with_context(
                    context, &spec,
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
                    context, &spec,
                    object_range,
                    complement_range,
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
            // The zero-arity UNIT lives at the PLAN-DEFAULT domain `a0` (no atom relation is
            // read for the result); the verifier recomputes it from the same `plan.source`.
            let mut unit = vec![f0(); 1usize << a0];
            if result_m == 1 {
                unit[0] = ArkFr::one();
            }
            (None, vec![context.commit_column(&unit)?], vec![unit], a0, result_m)
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
            let result = prove_bgp_chain_with_result_and_context(
                context, &relations[..relational_count],
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
            .map(|column| context.commit_column(column))
            .collect::<io::Result<_>>()?;
    }

    Ok(LayerBgpResult {
        proof: LayerBgpProof {
            atoms: atom_proofs,
            chain,
            dictionary_absence: None,
            // Term-bearing by default; `attach_termless_bgp_openings` fills this for the
            // term-less shipping path (Merkle-dict I2b S5).
            openings: None,
        },
        result_commitments: result_coms,
        result_columns,
        result_log_domain,
        result_len: result_m,
        predicate_ids,
    })
}

/// Merkle-dict I2b S5: attach the UNCOMMITTED term-less resolution witness to a freshly proven
/// base/rollup BGP result, so it can ship alongside a TERM-LESS artifact
/// ([`VerifierCommitment::verifier_commitment_termless`]). The prover still holds the full
/// term-bearing catalog, so it rebuilds the id-indexed dictionary trees and opens:
///   * every DISTINCT matched-constant id (predicate IRI + subject/object constants) — the S4
///     belt-hash witness tying each authenticated constant to its real `resolution_root`-bound
///     term; and
///   * every DISTINCT resolved-result id (per the plan's result namespaces) — the S3 resolution
///     witness.
///
/// The openings are stored on `result.proof.openings`; they are NEVER folded into
/// `commitment_root`/`resolution_root` and are re-verified per id by `verify_opening` at the
/// verifier (a forged/absent opening fails closed). BASE/ROLLUP single-layer only.
///
/// Over-inclusion is safe (the verifier only looks up the ids it needs, each independently
/// checked); under-inclusion fails closed. A term-bearing verifier ignores the witness entirely.
pub fn attach_termless_bgp_openings(
    layer: &LayerCommitment,
    plan: &LayerBgpPlan,
    result: &mut LayerBgpResult,
) -> io::Result<()> {
    require_base_layer_for_query(&layer.state)?;
    let mut nv_ids: Vec<u64> = Vec::new();
    let mut pred_ids: Vec<u64> = Vec::new();

    // (a) Matched-constant ids (S4 belt-hash). A variable predicate carries the canonical zero
    // placeholder (skip); subject/object constants come from the per-atom proof.
    for &pid in &result.predicate_ids {
        if pid != 0 {
            pred_ids.push(pid);
        }
    }
    for atom in &result.proof.atoms {
        if let Some(id) = atom.subject_id {
            nv_ids.push(id);
        }
        if let Some(id) = atom.object_id {
            nv_ids.push(id);
        }
    }

    // (b) Resolved-result ids (S3 resolution). A ground-result UNIT / dictionary-absence carries
    // no dictionary result ids.
    if !has_ground_result_unit(plan) && result.proof.dictionary_absence.is_none() {
        let namespaces = result_id_namespaces(plan)?;
        if namespaces.len() != result.result_columns.len() {
            return Err(invalid(
                "term-less openings: result column count does not match the plan namespaces",
            ));
        }
        for (column, namespace) in result.real_columns().iter().zip(&namespaces) {
            for id in column {
                match namespace {
                    ResultIdNamespace::NodeValue => {
                        nv_ids.push(super::result::canonical_dictionary_id(id)?)
                    }
                    ResultIdNamespace::Predicate => {
                        pred_ids.push(super::result::canonical_dictionary_id(id)?)
                    }
                    // Generated scalars are not dictionary ids — no opening needed.
                    ResultIdNamespace::GeneratedDecimalU64
                    | ResultIdNamespace::GeneratedDecimalI64 => {}
                }
            }
        }
    }

    // The reserved null id 0 is never a real dictionary id (never a constant, never resolved).
    nv_ids.retain(|&id| id != 0);
    pred_ids.retain(|&id| id != 0);

    // The builder routes each id to the unique interval that contains it and dedups internally.
    let view = layer.verifier_view();
    let openings = build_resolution_openings(&[view], &nv_ids, &pred_ids)?;
    result.proof.openings = Some(openings);
    Ok(())
}

/// Verify a k-atom layer BGP proof. Anchors to `commitment_root` FIRST (exactly like
/// [`verify_layer_graph_pattern2`]); then, under the shared transcript in prover order,
/// re-checks every atom's dictionary binding + complete filter, and the left-deep join
/// chain over the filtered relations (which enforces the two continuity conditions).
/// Concludes `result_coms[0..m]` is EXACTLY the complete BGP result, without rerunning it.
pub fn verify_layer_bgp(
    source: &impl VerifierLayer,
    trusted_resolution_root: ProofHash,
    plan: &LayerBgpPlan,
    predicate_ids: &[u64],
    result_coms: &[ArkGT],
    m: usize,
    proof: &LayerBgpProof,
) -> io::Result<()> {
    // Merkle-dict I2b S5: the per-matched-CONSTANT belt-hash openings now travel INSIDE the proof
    // (`proof.openings`). A term-bearing proof carries `None` and verifies unchanged (the belt is a
    // no-op there — the catalog re-MSM already supplied term fidelity). A TERM-LESS proof with its
    // openings verifies the belt; a term-less proof reaching here WITHOUT openings fails closed
    // (the residual guard in `verify_layer_bgp_impl`). No separate FFI plumbing is needed: the wire
    // carries the witness.
    verify_layer_bgp_impl(
        source,
        trusted_resolution_root,
        plan,
        predicate_ids,
        result_coms,
        m,
        proof,
        proof.openings.as_ref(),
    )
}

/// Merkle-dict I2b S5 test shim: openings now live INSIDE the proof (`proof.openings`), so this
/// injects the separately-supplied `openings` into a proof clone and routes through the PUBLIC
/// [`verify_layer_bgp`] — keeping the S3/S4 belt-hash tests (which pass a bare proof + openings)
/// exercising the exact production path.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_layer_bgp_with_constant_openings(
    source: &impl VerifierLayer,
    trusted_resolution_root: ProofHash,
    plan: &LayerBgpPlan,
    predicate_ids: &[u64],
    result_coms: &[ArkGT],
    m: usize,
    proof: &LayerBgpProof,
    openings: &ResolutionOpenings,
) -> io::Result<()> {
    let mut proof = proof.clone();
    proof.openings = Some(openings.clone());
    verify_layer_bgp(
        source,
        trusted_resolution_root,
        plan,
        predicate_ids,
        result_coms,
        m,
        &proof,
    )
}

#[allow(clippy::too_many_arguments)]
fn verify_layer_bgp_impl(
    source: &impl VerifierLayer,
    trusted_resolution_root: ProofHash,
    plan: &LayerBgpPlan,
    predicate_ids: &[u64],
    result_coms: &[ArkGT],
    m: usize,
    proof: &LayerBgpProof,
    openings: Option<&ResolutionOpenings>,
) -> io::Result<()> {
    let layer = source.verifier_view();
    require_base_layer_for_query(layer.state)?;
    // β re-anchor (FIX 4): pin to an EXTERNALLY-trusted `resolution_root` FIRST, before any
    // sub-proof — it binds `commitment_root` AND the two dictionary Merkle roots directly.
    if layer.resolution_root() != trusted_resolution_root {
        return Err(invalid(
            "layer resolution_root does not match the externally-trusted resolution root",
        ));
    }
    if layer.recomputed_commitment_root()? != layer.state.commitment_root {
        return Err(invalid(
            "layer commitment_root does not bind its committed columns/dictionary",
        ));
    }
    // Merkle-dict I2b S4: the blanket interim term-less fail-close is RELAXED here. Constant
    // matches are now tied to real terms by a per-matched-CONSTANT belt-hash (below), so a
    // term-less artifact is verifiable GIVEN the prover-shipped constant `openings`. The
    // `dictionary_absence` branch keeps its OWN blanket guard (absence cannot be proven from a
    // dropped catalog), and the residual fail-close (term-less WITHOUT openings ⇒ Err) is enforced
    // just after it, before the atom loop.
    let k = validate_bgp_plan(plan)?;
    if let Some(claimed_absence) = proof.dictionary_absence {
        // Merkle-dict I2b S2 fail-close: `dictionary_absence` proves a constant is ABSENT by
        // scanning the clear catalog. A TERM-LESS artifact carries EMPTY catalogs, so every
        // constant would appear absent (a fail-OPEN forgery: certify a present term as absent).
        // Reject before scanning. (Term-less is base/rollup-only, but the base BGP path IS
        // term-less-queryable for point lookups/joins, so this branch is reachable and must guard.)
        layer.require_terms_present("dictionary-absence proof (catalog nonmembership scan)")?;
        // Recommit every clear catalog/table before using catalog nonmembership. This
        // turns the scan below into an opening of the externally trusted layer root,
        // rather than trusting an unauthenticated convenience catalog.
        layer.validate()?;
        if dictionary_absence(&layer, plan) != Some(claimed_absence) {
            return Err(invalid(
                "claimed dictionary absence is not authentic",
            ));
        }
        let result_arity = result_id_namespaces(plan)?.len().max(1);
        if !proof.atoms.is_empty()
            || proof.chain.is_some()
            || !predicate_ids.is_empty()
            || m != 0
            || result_coms.len() != result_arity
        {
            return Err(invalid("invalid dictionary-absence proof shape"));
        }
        let zero_column =
            vec![f0(); padded_len(source_count(layer.state, plan.source) as usize)?];
        let zero_commitment = commit_column(&zero_column)?;
        if result_coms.iter().any(|commitment| *commitment != zero_commitment) {
            return Err(invalid(
                "dictionary-absence result is not the canonical empty relation",
            ));
        }
        return Ok(());
    }
    // Merkle-dict I2b S4 residual fail-close: past the `dictionary_absence` branch we are on the
    // atom / all-ground (ASK) constant-match path. A TERM-LESS artifact here has prover-chosen dict
    // HASH columns (no catalog re-MSM), so every matched constant MUST be belt-hashed against its
    // real `resolution_root`-bound Merkle term — which needs the prover-shipped `openings`. Without
    // them (the PUBLIC `verify_layer_bgp` entry, i.e. every FFI caller until S5) fail closed, even
    // for an all-variable BGP: no term-less constant-match verifies without the belt. (Term-bearing
    // is unaffected; the belt below is a no-op there.)
    if !layer.terms_present && openings.is_none() {
        return Err(invalid(
            "term-less layer BGP constant-match requires prover-shipped constant Merkle openings",
        ));
    }
    if proof.atoms.len() != k || predicate_ids.len() != k {
        return Err(invalid("layer BGP proof/atom/id count mismatch"));
    }
    // Plan-default domain, used ONLY for the all-ground UNIT (relational atoms use their own
    // per-atom `a` sourced inside the loop). Mirrors the prover's `a0`.
    let a0 = padded_len(source_count(layer.state, plan.source) as usize)?
        .trailing_zeros() as usize;

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
    bind_layer_bgp_statement(&mut transcript, &layer, plan, &atom_ids);

    let com_pred_hash = layer.predicate_dict_commitments.0;
    let com_pred_id = layer.predicate_dict_commitments.1;
    let com_nv_hash = layer.node_value_dict_commitments.0;
    let com_nv_id = layer.node_value_dict_commitments.1;

    let mut relations: Vec<RelationCommit> = Vec::with_capacity(k);
    for ((atom, ap), ids) in plan
        .atoms
        .iter()
        .zip(proof.atoms.iter())
        .zip(atom_ids.iter())
    {
        // (0) PER-ATOM sourcing, mirroring the prover: this atom's committed row count `n`,
        // table-domain log `a`, and its three `(S, P, O)` column commitments come from THIS
        // atom's `source` (all bound by the same trusted root). The source is fixed by the
        // verifier's trusted plan (part of the plan wire image / `plan_digest`), so a prover
        // cannot relabel an atom's source: the columns/count it must open against are the ones
        // this reconstruction selects.
        let n = source_count(layer.state, atom.source) as usize;
        let a = padded_len(n)?.trailing_zeros() as usize;
        let [com_s, com_p, com_o] = source_column_commitments(layer, atom.source);
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
            // Merkle-dict I2b S4 belt-hash: on a TERM-LESS artifact `com_pred_hash` is prover-chosen,
            // so the membership alone could authenticate a fabricated IRI->id mapping. Tie this
            // authenticated id to its real `pred_merkle_root`-bound IRI. (No-op term-bearing.)
            belt_hash_predicate_constant(layer, openings, predicate_id, &atom.predicate_term)?;
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
            // Merkle-dict I2b S4 belt-hash: tie this authenticated subject id to its real
            // `nv_merkle_root`-bound term (no-op term-bearing).
            belt_hash_node_value_constant(layer, openings, subject_id, &object)?;
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
            // Merkle-dict I2b S4 belt-hash: tie this authenticated object id to its real
            // `nv_merkle_root`-bound term — THE deferred I2b object-constant forge closer.
            belt_hash_node_value_constant(layer, openings, object_id, &object)?;
        }
        // (3) Selection: `kept_commitments[0..m]` is the complete per-atom result. Which
        // path runs is decided by the PLAN atom's `object_range` (public), NOT by the
        // prover-supplied proof — a plain atom must carry a `filter`, a ranged atom a
        // `range`. The spec reuses the SAME ids the memberships authenticated.
        let spec = atom_filter_spec(atom, ids);
        let node_kind_range = ObjectRangeSpec {
            op: RangeOp::Ge, datatype: ArkFr::from_u64(NODE_DATATYPE_TAG),
            c: ArkFr::zero(), b_bits: 64, second: None,
        };
        let (effective_range, complement_range) = match atom.object_kind {
            ObjectKindFilter::Any => (atom.object_range.as_ref(), false),
            ObjectKindFilter::Node => (Some(&node_kind_range), false),
            ObjectKindFilter::Literal => (Some(&node_kind_range), true),
        };
        match effective_range {
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
                    complement_range,
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
        // Plan-default domain `a0` for the zero-arity UNIT (matches the prover).
        let mut unit = vec![f0(); 1usize << a0];
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

/// Merkle-dict I2b S4 per-matched-CONSTANT belt-hash (node/value position). TERM-BEARING: no-op —
/// `validate()`'s catalog re-MSM already tied `com_nv_hash` to the real terms. TERM-LESS:
/// `com_nv_hash` is prover-chosen, so `verify_dict_membership` alone could authenticate a
/// fabricated term->id mapping; open the authenticated constant `id` against the
/// `resolution_root`-bound `nv_merkle_root` (via the prover-shipped `openings`, fail-closed if
/// absent) and require the opened term to hash to the SAME value the membership matched
/// (`hash_object`, the Blake2b used by `verify_dict_membership` + the I2a belt). A wrong id or a
/// substituted term ⇒ hash mismatch ⇒ Err. `verify_opening` independently re-forces the interval +
/// structural position against the trusted root, so no id/term substitution survives.
fn belt_hash_node_value_constant(
    layer: VerifierCommitmentRef<'_>,
    openings: Option<&ResolutionOpenings>,
    id: u64,
    constant: &CanonicalObject,
) -> io::Result<()> {
    if layer.terms_present {
        return Ok(());
    }
    let opening = openings
        .ok_or_else(|| {
            invalid(
                "term-less layer BGP constant-match requires prover-shipped constant Merkle openings",
            )
        })?
        .node_value_opening(id)?;
    let opened = verify_opening(
        &layer.nv_merkle_root,
        layer.state.parent_node_value_count,
        layer.state.node_value_count,
        id,
        opening,
    )?;
    if hash_object(&opened) != hash_object(constant) {
        return Err(invalid(
            "term-less BGP belt-hash: opened node/value term does not match the matched constant",
        ));
    }
    Ok(())
}

/// Merkle-dict I2b S4 per-matched-CONSTANT belt-hash (predicate position). As
/// [`belt_hash_node_value_constant`], but opens `pred_merkle_root` and compares `hash_term` of the
/// reconstructed IRI (predicate leaves are node IRIs) against the matched predicate constant.
fn belt_hash_predicate_constant(
    layer: VerifierCommitmentRef<'_>,
    openings: Option<&ResolutionOpenings>,
    id: u64,
    predicate_term: &[u8],
) -> io::Result<()> {
    if layer.terms_present {
        return Ok(());
    }
    let opening = openings
        .ok_or_else(|| {
            invalid(
                "term-less layer BGP constant-match requires prover-shipped constant Merkle openings",
            )
        })?
        .predicate_opening(id)?;
    let opened = verify_opening(
        &layer.pred_merkle_root,
        layer.state.parent_predicate_count,
        layer.state.predicate_count,
        id,
        opening,
    )?;
    let iri = match &opened {
        CanonicalObject::Node(bytes) => bytes.as_slice(),
        _ => {
            return Err(invalid(
                "term-less BGP belt-hash: predicate leaf did not reconstruct as a node IRI",
            ))
        }
    };
    if hash_term(iri) != hash_term(predicate_term) {
        return Err(invalid(
            "term-less BGP belt-hash: opened predicate IRI does not match the matched constant",
        ));
    }
    Ok(())
}

/// Re-derive the only valid public result commitment domain from the authenticated plan
/// and proof shape. This is especially important for a false mixed-arity gate, whose
/// all-zero commitments are group identities and therefore do not encode a domain size.
pub fn verify_layer_bgp_result_log_domain(
    source: &impl VerifierLayer,
    plan: &LayerBgpPlan,
    proof: &LayerBgpProof,
    actual: usize,
) -> io::Result<()> {
    let layer = source.verifier_view();
    validate_bgp_plan(plan)?;
    let relational_count = relational_atom_count(plan);
    let expected = if proof.dictionary_absence.is_some() || relational_count == 0 {
        // Dictionary-absence zero column and the all-ground UNIT both live at the PLAN-DEFAULT
        // source domain (no single atom relation is read for the result).
        padded_len(source_count(layer.state, plan.source) as usize)?.trailing_zeros() as usize
    } else if relational_count == 1 {
        // A single relational atom's result IS its own filtered relation, so its domain is that
        // atom's OWN source domain (== `plan.source` for a uniform plan; `validate_bgp_plan`
        // places the relational atom first, so `atoms[0]` is it).
        padded_len(source_count(layer.state, plan.atoms[0].source) as usize)?
            .trailing_zeros() as usize
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
