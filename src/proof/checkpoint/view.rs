//! Checkpoint views for queries (`doc/proof-of-woql/checkpoint-proofs.md` §5, §6 step 2).
//!
//! The single-layer query gadgets read a BASE layer's verifier view: the materialized S, P, O
//! commitments, the dictionary columns `(hash, id)`, `(value, datatype)` (node/value) and
//! `(hash, id)` (predicate) in the standard Dory layout of their own padded length, the id-indexed
//! `id -> term` trees, the `term -> id` index and the dictionary facts. A proven checkpoint holds
//! all of it — but its dictionary columns are committed in the FIXED segment layout
//! (`fixed_commit`), which a gadget cannot open at its own domain. This module RELOCATES them
//! once per checkpoint:
//!
//! * the prover commits the view's columns in the standard layout (rows in chain-id order
//!   `id - 1`, zero-padded to `next_pow2(n).max(2)`, exactly as a base layer commits its catalog
//!   columns) and ships those six commitments;
//! * after they are absorbed, the verifier draws one point `r` over `M = max(m_nv, m_pred)`
//!   variables; the prover opens the standard columns at `r[..m]` (batched per namespace, the id
//!   columns evaluated by the verifier in closed form — `id(i) = i + 1` below the count, zero
//!   above) and the checkpoint's segment-0 commitments at `(r, 0^(24 - M))` (one batched opening
//!   of the four fixed columns, with the zero-extension factor `Π_{i >= m}(1 - r_i)` applied);
//! * agreement at a random point of two multilinear polynomials over `M` variables means the
//!   standard columns ARE the checkpoint's columns restricted to their prefix (Schwartz–Zippel,
//!   error `M / |F|`), and the checkpoint's columns are zero beyond their counts by construction
//!   (the verifier appended every entry itself, inductively).
//!
//! The rest of the view is read off the checkpoint artifact: the present commitments are exactly
//! the materialized ones a base layer of these triples commits (`present_domain_log`); the facts,
//! term-tree roots and term-index roots are the checkpoint's. The result is a TERM-LESS base
//! [`VerifierCommitment`] flagged `checkpoint` (resolution opens the term tree, constant lookups
//! the keyed index) and `attested` (every well-formedness fact the attested path assumes is
//! PROVEN here: the commitments are the canonical function of the terms — appended by the verifier
//! or relocated above —, the facts are recomputed from the terms, the index and the tree are built
//! from the same terms, and the present relation is the proven SPO-sorted set).
//!
//! # Cost
//!
//! The view proof is per checkpoint, not per query. Prover: the six standard commitments
//! (`O(n)` MSM), two dense openings at their own domains, and one prefix-sparse 24-variable opening
//! of the fixed columns — `O(n)` field work plus the fixed `2^12`-wide Dory rounds
//! (`argument::prove_prefix_column_openings_with_context`). Verifier: three Dory verifications.
//! Proof: six GT commitments, four evaluations, three openings.
//!
//! # Scope
//!
//! A view needs its dictionary within the PF4 record cap (`MAX_RECORDS`, the largest standard
//! domain any gadget opens — the same cap every base layer has), i.e. within the first fixed
//! segment. The checkpoint itself commits up to `2^32 - 1` entries per namespace.

use std::io;
use std::sync::Arc;

use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::{Field, Group};
use dory_pcs::primitives::transcript::Transcript;

use super::state::{nv_dict_segment, pred_dict_segment, CheckpointArtifact, CheckpointState};
use crate::proof::argument::{
    mle_full_eval, prove_prefix_column_openings_with_context, verify_column_openings, ColumnOpening,
    StoreProverContext, Transcript254,
};
use crate::proof::commitment::{
    CheckpointViewCommitments, LayerCommitment, ResolutionOpenings, VerifierCommitment,
    VerifierCommitmentRef, VerifierLayer, SETUP_MAX_LOG_N,
};
use crate::proof::{invalid, CanonicalObject, CanonicalTriple, ProofHash, ProofRecord};

const VIEW_TRANSCRIPT_LABEL: &[u8] = b"parture-checkpoint-view";
const VIEW_DOMAIN: &[u8] = b"parture/proof-of-woql/checkpoint-view/v1";

/// The relocation of a checkpoint's dictionary columns into the standard layout.
#[derive(Clone, Debug)]
pub struct CheckpointViewProof {
    /// Node/value `hash`, `id`, `value`, `datatype` in the standard layout (identities when the
    /// dictionary is empty).
    pub nv_commitments: Vec<ArkGT>,
    /// Predicate `hash`, `id` in the standard layout (identities when empty).
    pub pred_commitments: Vec<ArkGT>,
    /// Node/value `hash`, `value`, `datatype` at `r[..m_nv]` (empty when the dictionary is).
    pub nv_evals: Vec<ArkFr>,
    /// Predicate `hash` at `r[..m_pred]` (empty when the dictionary is).
    pub pred_evals: Vec<ArkFr>,
    pub nv_opening: Option<ColumnOpening>,
    pub pred_opening: Option<ColumnOpening>,
    /// The checkpoint's segment-0 columns at `(r, 0^(24 - M))`.
    pub fixed_opening: Option<ColumnOpening>,
}

/// `log2` of the standard padded domain of a dictionary of `n` entries (`0` when empty).
fn standard_log(n: u64) -> io::Result<usize> {
    if n == 0 {
        return Ok(0);
    }
    if n > crate::proof::MAX_RECORDS as u64 {
        return Err(invalid(
            "a checkpoint query view needs its dictionary within the PF4 record cap (the largest \
             domain a query gadget opens)",
        ));
    }
    Ok((n as usize).next_power_of_two().max(2).trailing_zeros() as usize)
}

/// `MLE(f)(point)` for `f(i) = i + 1` when `i < n`, `0` otherwise, over `2^point.len()` entries,
/// in `O(point.len()^2)`.
///
/// Enumerating `i < n` by the highest bit `k` where `n` has a one and `i` a zero (above `k` the two
/// agree, below it `i` is free): the free low bits sum `eq` to one and `i·eq` to
/// `Σ_{j<k} 2^j r_j`, and the fixed high part contributes `Π_{j>k} eq(n_j, r_j) · (1 - r_k)`.
pub(crate) fn prefix_id_eval(point: &[ArkFr], n: u64) -> io::Result<ArkFr> {
    let bits = point.len();
    if bits < 64 && n > 1u64 << bits {
        return Err(invalid("prefix id column exceeds its domain"));
    }
    let two = ArkFr::from_u64(2);
    let pow = |j: usize| (0..j).fold(ArkFr::one(), |acc, _| acc * two);
    let low_index = |k: usize| (0..k).fold(ArkFr::zero(), |acc, j| acc + pow(j) * point[j]);
    if bits < 64 && n == 1u64 << bits {
        // Every row: Σ eq = 1 and Σ i·eq = idx(r).
        return Ok(low_index(bits) + ArkFr::one());
    }
    let mut sum_eq = ArkFr::zero();
    let mut sum_index = ArkFr::zero();
    for k in 0..bits {
        if n >> k & 1 == 0 {
            continue;
        }
        let mut weight = ArkFr::one() - point[k];
        let mut high = ArkFr::zero();
        for j in k + 1..bits {
            let bit = n >> j & 1 == 1;
            weight = weight * if bit { point[j] } else { ArkFr::one() - point[j] };
            if bit {
                high = high + pow(j);
            }
        }
        sum_eq = sum_eq + weight;
        sum_index = sum_index + weight * (high + low_index(k));
    }
    Ok(sum_index + sum_eq)
}

/// The standard-layout columns of the view, rows in chain-id order.
struct ViewColumns {
    nv: [Vec<ArkFr>; 4],
    pred: [Vec<ArkFr>; 2],
    /// The unpadded checkpoint columns (`hash`, `value`, `datatype`, predicate `hash`) — the
    /// prefixes of the fixed-layout segment-0 columns.
    fixed: [Vec<ArkFr>; 4],
}

fn view_columns(state: &CheckpointState) -> io::Result<ViewColumns> {
    let nv_len = state.nv_terms.len() as u64;
    let pred_len = state.pred_terms.len() as u64;
    let pad = |column: &[ArkFr], log: usize, n: u64| -> Vec<ArkFr> {
        if n == 0 {
            return Vec::new();
        }
        let mut out = column.to_vec();
        out.resize(1 << log, ArkFr::zero());
        out
    };
    let ids = |n: u64, log: usize| -> Vec<ArkFr> {
        if n == 0 {
            return Vec::new();
        }
        let mut out: Vec<ArkFr> = (1..=n).map(ArkFr::from_u64).collect();
        out.resize(1 << log, ArkFr::zero());
        out
    };
    let m_nv = standard_log(nv_len)?;
    let m_pred = standard_log(pred_len)?;
    let [hash, value, datatype] = nv_dict_segment(&state.nv_terms)?;
    let pred_hash = pred_dict_segment(&state.pred_terms);
    Ok(ViewColumns {
        nv: [pad(&hash, m_nv, nv_len), ids(nv_len, m_nv), pad(&value, m_nv, nv_len), pad(&datatype, m_nv, nv_len)],
        pred: [pad(&pred_hash, m_pred, pred_len), ids(pred_len, m_pred)],
        fixed: [hash, value, datatype, pred_hash],
    })
}

fn bind_view(transcript: &mut Transcript254, artifact: &CheckpointArtifact, proof_coms: &[&ArkGT]) {
    transcript.append_bytes(b"pow/checkpoint-view/domain", VIEW_DOMAIN);
    transcript.append_bytes(b"pow/checkpoint-view/checkpoint-root", &artifact.checkpoint_root);
    for count in [artifact.present_count, artifact.node_value_count, artifact.predicate_count] {
        transcript.append_bytes(b"pow/checkpoint-view/count", &count.to_le_bytes());
    }
    for commitment in proof_coms {
        transcript.append_group(b"pow/checkpoint-view/standard-com", *commitment);
    }
}

fn draw_point(transcript: &mut Transcript254, vars: usize) -> Vec<ArkFr> {
    (0..vars).map(|_| transcript.challenge_scalar(b"pow/checkpoint-view/point")).collect()
}

/// `Π_{i >= low} (1 - r_i)`.
fn extension_factor(point: &[ArkFr], low: usize) -> ArkFr {
    point[low..].iter().fold(ArkFr::one(), |acc, r| acc * (ArkFr::one() - *r))
}

/// The fixed-layout point `(r, 0^(24 - r.len()))`.
fn fixed_point(r: &[ArkFr]) -> Vec<ArkFr> {
    let mut point = r.to_vec();
    point.resize(SETUP_MAX_LOG_N, ArkFr::zero());
    point
}

/// Test-only forgeries of the view's STANDARD columns (the checkpoint's own columns stay honest);
/// each must be rejected by the relocation.
#[cfg(test)]
#[derive(Clone, Copy, Debug)]
pub(crate) enum ViewForgery {
    NvValue,
    NvDatatype,
    NvHash,
    NvId,
    PredHash,
    NvPadding,
}

#[cfg(test)]
pub(crate) fn prove_checkpoint_view_forged(
    context: &StoreProverContext<'_>,
    state: &CheckpointState,
    forgery: ViewForgery,
) -> io::Result<CheckpointViewProof> {
    let mut columns = view_columns(state)?;
    let one = ArkFr::one();
    match forgery {
        ViewForgery::NvValue => columns.nv[2][0] = columns.nv[2][0] + one,
        ViewForgery::NvDatatype => columns.nv[3][0] = columns.nv[3][0] + one,
        ViewForgery::NvHash => columns.nv[0][0] = columns.nv[0][0] + one,
        ViewForgery::NvId => columns.nv[1][0] = columns.nv[1][0] + one,
        ViewForgery::PredHash => columns.pred[0][0] = columns.pred[0][0] + one,
        ViewForgery::NvPadding => {
            let last = columns.nv[2].len() - 1;
            columns.nv[2][last] = columns.nv[2][last] + one;
        }
    }
    Ok(prove_view_columns(context, state, columns)?.1)
}

/// PROVER: the view proof of `state`, with the view's commitments.
pub fn prove_checkpoint_view(
    context: &StoreProverContext<'_>,
    state: &CheckpointState,
) -> io::Result<(CheckpointViewCommitments, CheckpointViewProof)> {
    prove_view_columns(context, state, view_columns(state)?)
}

fn prove_view_columns(
    context: &StoreProverContext<'_>,
    state: &CheckpointState,
    columns: ViewColumns,
) -> io::Result<(CheckpointViewCommitments, CheckpointViewProof)> {
    let artifact = &state.artifact;
    let commit = |column: &Vec<ArkFr>| -> io::Result<ArkGT> {
        if column.is_empty() { Ok(<ArkGT as Group>::identity()) } else { context.commit_column(column) }
    };
    let nv_commitments = columns.nv.iter().map(|c| commit(c)).collect::<io::Result<Vec<_>>>()?;
    let pred_commitments = columns.pred.iter().map(|c| commit(c)).collect::<io::Result<Vec<_>>>()?;
    let m_nv = standard_log(artifact.node_value_count)?;
    let m_pred = standard_log(artifact.predicate_count)?;
    let mut transcript = Transcript254::new(VIEW_TRANSCRIPT_LABEL);
    let all: Vec<&ArkGT> = nv_commitments.iter().chain(pred_commitments.iter()).collect();
    bind_view(&mut transcript, artifact, &all);
    let vars = m_nv.max(m_pred);
    let r = draw_point(&mut transcript, vars);

    let (nv_evals, nv_opening) = if m_nv > 0 {
        let point = &r[..m_nv];
        let evals = [0, 2, 3].iter().map(|&c| mle_full_eval(&columns.nv[c], point)).collect();
        (evals, Some(context.prove_column_openings(&columns.nv, point, &mut transcript)?))
    } else {
        (Vec::new(), None)
    };
    let (pred_evals, pred_opening) = if m_pred > 0 {
        let point = &r[..m_pred];
        let evals = vec![mle_full_eval(&columns.pred[0], point)];
        (evals, Some(context.prove_column_openings(&columns.pred, point, &mut transcript)?))
    } else {
        (Vec::new(), None)
    };
    let fixed_opening = if vars > 0 {
        let (fixed, coms) = fixed_columns(artifact, &columns.fixed)?;
        Some(prove_prefix_column_openings_with_context(context, &fixed, &coms, &fixed_point(&r), &mut transcript)?)
    } else {
        None
    };
    let view = CheckpointViewCommitments {
        node_value_count: artifact.node_value_count,
        predicate_count: artifact.predicate_count,
        materialized_count: artifact.present_count,
        materialized_column_commitments: artifact.present_commitments,
        predicate_dict_commitments: (pred_commitments[0], pred_commitments[1]),
        node_value_dict_commitments: (nv_commitments[0], nv_commitments[1]),
        node_value_dict_range_commitments: (nv_commitments[2], nv_commitments[3]),
    };
    let proof = CheckpointViewProof {
        nv_commitments,
        pred_commitments,
        nv_evals,
        pred_evals,
        nv_opening,
        pred_opening,
        fixed_opening,
    };
    Ok((view, proof))
}

/// The fixed-layout columns opened by the relocation (node/value hash, value, datatype, then
/// predicate hash — each present only when its dictionary is nonempty) with their segment-0
/// commitments.
fn fixed_columns<'a>(
    artifact: &CheckpointArtifact,
    columns: &'a [Vec<ArkFr>; 4],
) -> io::Result<(Vec<&'a [ArkFr]>, Vec<ArkGT>)> {
    let mut out = Vec::new();
    let mut coms = Vec::new();
    if artifact.node_value_count > 0 {
        let segment = artifact.nv_dict_segments.first().ok_or_else(|| invalid("checkpoint lacks its dictionary segment"))?;
        for c in 0..3 {
            out.push(columns[c].as_slice());
            coms.push(segment[c]);
        }
    }
    if artifact.predicate_count > 0 {
        out.push(columns[3].as_slice());
        coms.push(*artifact.pred_dict_segments.first().ok_or_else(|| invalid("checkpoint lacks its dictionary segment"))?);
    }
    Ok((out, coms))
}

/// A checkpoint view the verifier has checked: the term-less base commitment the single-layer
/// gadgets verify against, built by [`verify_checkpoint_view`] from a trusted checkpoint artifact
/// and its view proof. Only that function constructs one.
#[derive(Clone, Debug)]
pub struct VerifiedCheckpointView {
    commitment: VerifierCommitment,
    checkpoint_root: ProofHash,
}

impl VerifiedCheckpointView {
    /// The view's base commitment (term-less; its roots are the checkpoint's term structures).
    pub fn commitment(&self) -> &VerifierCommitment {
        &self.commitment
    }
    /// The checkpoint the view was verified against.
    pub fn checkpoint_root(&self) -> ProofHash {
        self.checkpoint_root
    }
    /// The root every query transcript over this view is bound to.
    pub fn commitment_root(&self) -> ProofHash {
        self.commitment.state.commitment_root
    }
}

/// VERIFIER: check a view proof against the checkpoint whose root the caller trusts and build the
/// view's base commitment.
pub fn verify_checkpoint_view(
    artifact: &CheckpointArtifact,
    trusted_checkpoint_root: ProofHash,
    proof: &CheckpointViewProof,
) -> io::Result<VerifiedCheckpointView> {
    artifact.validate()?;
    if artifact.checkpoint_root != trusted_checkpoint_root {
        return Err(invalid("checkpoint view: the artifact is not the trusted checkpoint"));
    }
    let m_nv = standard_log(artifact.node_value_count)?;
    let m_pred = standard_log(artifact.predicate_count)?;
    if proof.nv_commitments.len() != 4 || proof.pred_commitments.len() != 2 {
        return Err(invalid("checkpoint view: wrong number of standard commitments"));
    }
    let identity = <ArkGT as Group>::identity();
    let check_empty = |m: usize, coms: &[ArkGT], evals: &[ArkFr], opening: &Option<ColumnOpening>, n: usize| {
        if m == 0 {
            if coms.iter().any(|c| *c != identity) || !evals.is_empty() || opening.is_some() {
                return Err(invalid("checkpoint view: an empty dictionary carries columns"));
            }
        } else if evals.len() != n || opening.is_none() {
            return Err(invalid("checkpoint view: missing evaluations or opening"));
        }
        Ok(())
    };
    check_empty(m_nv, &proof.nv_commitments, &proof.nv_evals, &proof.nv_opening, 3)?;
    check_empty(m_pred, &proof.pred_commitments, &proof.pred_evals, &proof.pred_opening, 1)?;

    let mut transcript = Transcript254::new(VIEW_TRANSCRIPT_LABEL);
    let all: Vec<&ArkGT> = proof.nv_commitments.iter().chain(proof.pred_commitments.iter()).collect();
    bind_view(&mut transcript, artifact, &all);
    let vars = m_nv.max(m_pred);
    let r = draw_point(&mut transcript, vars);
    let stage = |what: &'static str| move |error: io::Error| invalid(format!("checkpoint view {what}: {error}"));

    let mut fixed_claims = Vec::new();
    if m_nv > 0 {
        let point = &r[..m_nv];
        let id = prefix_id_eval(point, artifact.node_value_count)?;
        let e = &proof.nv_evals;
        verify_column_openings(
            &proof.nv_commitments,
            point,
            &[e[0], id, e[1], e[2]],
            proof.nv_opening.as_ref().unwrap(),
            &mut transcript,
        )
        .map_err(stage("node/value columns"))?;
        let factor = extension_factor(&r, m_nv);
        fixed_claims.extend(e.iter().map(|v| *v * factor));
    }
    if m_pred > 0 {
        let point = &r[..m_pred];
        let id = prefix_id_eval(point, artifact.predicate_count)?;
        verify_column_openings(
            &proof.pred_commitments,
            point,
            &[proof.pred_evals[0], id],
            proof.pred_opening.as_ref().unwrap(),
            &mut transcript,
        )
        .map_err(stage("predicate columns"))?;
        fixed_claims.push(proof.pred_evals[0] * extension_factor(&r, m_pred));
    }
    match (&proof.fixed_opening, vars) {
        (None, 0) => {}
        (Some(opening), v) if v > 0 => {
            let empty: [Vec<ArkFr>; 4] = Default::default();
            let (_, coms) = fixed_columns(artifact, &empty)?;
            verify_column_openings(&coms, &fixed_point(&r), &fixed_claims, opening, &mut transcript)
                .map_err(stage("relocation"))?;
        }
        _ => return Err(invalid("checkpoint view: relocation opening presence mismatch")),
    }

    let commitments = CheckpointViewCommitments {
        node_value_count: artifact.node_value_count,
        predicate_count: artifact.predicate_count,
        materialized_count: artifact.present_count,
        materialized_column_commitments: artifact.present_commitments,
        predicate_dict_commitments: (proof.pred_commitments[0], proof.pred_commitments[1]),
        node_value_dict_commitments: (proof.nv_commitments[0], proof.nv_commitments[1]),
        node_value_dict_range_commitments: (proof.nv_commitments[2], proof.nv_commitments[3]),
    };
    let commitment = VerifierCommitment::checkpoint_view(
        &commitments,
        (artifact.nv_term_tree_root, artifact.pred_term_tree_root),
        (artifact.nv_term_index_root, artifact.pred_term_index_root),
        artifact.facts,
    )?;
    Ok(VerifiedCheckpointView { commitment, checkpoint_root: artifact.checkpoint_root })
}

/// The layer a checkpoint view verifies through: the view's commitment, flagged `checkpoint` and
/// `attested`, plus the resolution witness shipped with one verification.
pub(crate) struct CheckpointQueryView<'a> {
    pub(crate) commitment: &'a VerifierCommitment,
    pub(crate) openings: &'a ResolutionOpenings,
}

impl VerifierLayer for CheckpointQueryView<'_> {
    fn verifier_view(&self) -> VerifierCommitmentRef<'_> {
        VerifierCommitmentRef { attested: true, checkpoint: true, ..self.commitment.verifier_view() }
    }
    fn shipped_openings(&self) -> Option<&ResolutionOpenings> {
        Some(self.openings)
    }
}

/// PROVER: everything needed to prove queries over one checkpoint — the term-bearing base sidecar
/// of its view (catalogs in chain-id order) that the gadgets prove with, the view proof, the
/// verifier's view (built by running the view verifier) and the checkpoint state the resolution
/// witness is opened from.
pub struct CheckpointQueryLayer {
    layer: LayerCommitment,
    view: VerifiedCheckpointView,
    proof: CheckpointViewProof,
    state: Arc<CheckpointState>,
}

impl CheckpointQueryLayer {
    /// Prove the view of `state` and build its query sidecar.
    pub fn new(context: &StoreProverContext<'_>, state: Arc<CheckpointState>) -> io::Result<Self> {
        let (commitments, proof) = prove_checkpoint_view(context, &state)?;
        let view = verify_checkpoint_view(&state.artifact, state.artifact.checkpoint_root, &proof)?;
        let node = |id: u64| -> io::Result<&CanonicalObject> {
            state.node_value(id).ok_or_else(|| invalid("checkpoint present triple names an unknown id"))
        };
        let mut materialized = Vec::with_capacity(state.present.len());
        for t in &state.present {
            let subject = match node(t[0])? {
                CanonicalObject::Node(iri) => iri.clone(),
                CanonicalObject::Value { .. } => return Err(invalid("checkpoint subject is not a node")),
            };
            let predicate = state
                .predicate(t[1])
                .ok_or_else(|| invalid("checkpoint present triple names an unknown predicate"))?
                .to_vec();
            materialized.push(ProofRecord {
                subject_id: t[0],
                predicate_id: t[1],
                object_id: t[2],
                triple: CanonicalTriple { subject, predicate, object: node(t[2])?.clone() },
            });
        }
        let node_value_catalog = state
            .nv_terms
            .iter()
            .enumerate()
            .map(|(i, object)| ProofRecord {
                subject_id: 0,
                predicate_id: 0,
                object_id: i as u64 + 1,
                triple: CanonicalTriple { subject: Vec::new(), predicate: Vec::new(), object: object.clone() },
            })
            .collect();
        let predicate_catalog = state
            .pred_terms
            .iter()
            .enumerate()
            .map(|(i, iri)| ProofRecord {
                subject_id: 0,
                predicate_id: i as u64 + 1,
                object_id: 0,
                triple: CanonicalTriple::node(Vec::new(), iri.clone(), Vec::new()),
            })
            .collect();
        let layer = LayerCommitment::checkpoint_view_layer(&commitments, materialized, node_value_catalog, predicate_catalog)?;
        if layer.state.commitment_root != view.commitment_root() {
            return Err(invalid("checkpoint query sidecar does not reproduce the view's commitment root"));
        }
        Ok(Self { layer, view, proof, state })
    }

    /// The term-bearing base sidecar the gadgets prove with.
    pub fn layer(&self) -> &LayerCommitment {
        &self.layer
    }
    /// The verifier's view (what [`verify_checkpoint_view`] builds from the artifact and proof).
    pub fn view(&self) -> &VerifiedCheckpointView {
        &self.view
    }
    /// The view proof to ship with the checkpoint artifact.
    pub fn view_proof(&self) -> &CheckpointViewProof {
        &self.proof
    }
    /// The checkpoint.
    pub fn state(&self) -> &Arc<CheckpointState> {
        &self.state
    }
    /// The checkpoint artifact.
    pub fn artifact(&self) -> &CheckpointArtifact {
        &self.state.artifact
    }
    /// The root every query transcript over this view is bound to.
    pub fn commitment_root(&self) -> ProofHash {
        self.view.commitment_root()
    }
    /// PROVER-ONLY: a lazy resolution witness over the checkpoint's term structures.
    pub(crate) fn oracle(&self) -> ResolutionOpenings {
        ResolutionOpenings::checkpoint_oracle(Arc::clone(&self.state))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_id_eval_matches_the_table() {
        for bits in 1..=6usize {
            let point: Vec<ArkFr> = (0..bits).map(|j| ArkFr::from_u64(17 * j as u64 + 3)).collect();
            for n in 0..=(1u64 << bits) {
                let mut table = vec![ArkFr::zero(); 1 << bits];
                for i in 0..n {
                    table[i as usize] = ArkFr::from_u64(i + 1);
                }
                assert_eq!(prefix_id_eval(&point, n).unwrap(), mle_full_eval(&table, &point), "bits {bits} n {n}");
            }
            assert!(prefix_id_eval(&point, (1 << bits) + 1).is_err());
        }
    }
}
