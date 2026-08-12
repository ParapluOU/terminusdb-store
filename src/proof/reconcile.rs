//! Delta-stack RECONCILIATION — a walking skeleton (Waves 1-3 core).
//!
//! Proves a read query over a **base + 1 delta** layer stack by RECONCILING the
//! current triple set at proof time and authenticating it against the committed
//! per-layer deltas — WITHOUT any materialized head view (the Wave-0 commitment
//! diet removed it). This is the miniature of the whole delta-stack path; the
//! N-layer chain and the full BGP/filter/range/join generalize it later.
//!
//! # The math (reconciliation = a multiset equality)
//! terminus-store maintains the delta invariant (add only absent triples, remove
//! only present ones), so the current set `present` satisfies a CLEAN multiset
//! equality over canonical (dictionary-independent) terms:
//! ```text
//! present  ⊎  D.removals   ≡   B.additions  ⊎  D.additions
//! ```
//! The prover supplies `present`; this module proves the equality via a two-sided
//! masked LogUp link (the discipline that closed the self-join padding bug), which
//! authenticates `present` against the committed delta columns. Under the invariant
//! `present` is duplicate-free (multiplicities ∈ {0,1}).
//!
//! # Cross-layer dictionary authentication (Wave 1)
//! The additions/removals columns are dictionary-IDS local to each layer. A
//! parent-child chain SHARES the dictionary (the child extends the parent: inherited
//! ids ≤ `parent_node_value_count` are byte-identical to the base's, new ids are
//! `> parent_count`). `D.additions` therefore mixes inherited and D-local ids. We
//! reconcile in canonical-term-HASH space (`hash_object`, dictionary-independent) so
//! the same term hashes identically regardless of which layer minted its id:
//!   * a COMBINED node/value dictionary is authenticated as the exact bag union of
//!     the base dict (bound to `base.commitment_root`) and the delta's INCREMENTAL
//!     dict (bound to `delta.commitment_root`) — this resolves every id, inherited or
//!     D-local, in one authenticated table (`super::union` + `super::dicthash` inner);
//!   * each committed group's subject/object id column is authenticated `id → hash`
//!     against that combined dict (the audited [`super::dicthash`] lookup inner).
//!
//! # Root binding (Wave 2 envelope)
//! The verifier validates each layer against ITS OWN `commitment_root`, checks the
//! chain (`delta.parent_state_root == base.commitment_root`) and pins the head to the
//! externally-trusted `delta.commitment_root`. No layer may be spliced or substituted.
//!
//! # Scope of THIS skeleton (deferred to later passes — see the module report)
//!   * exactly base + 1 delta (a 2-way inherited/local dict split), not an N-layer chain;
//!   * the PREDICATE is carried as its root-bound raw id under a *no-new-predicates*
//!     guard (`delta.predicate_count == parent`), so ids are layer-consistent; full
//!     predicate hashing (a combined predicate dict) is the trivial extension;
//!   * the query is the reconciled triple COUNT (the exact answer of the single-atom
//!     read over `present`); BGP/filter/range over `present` is Wave 3;
//!   * each committed group must have ≥ 2 records (so its column commitment lives at a
//!     relocatable ≥ 2 domain).

use std::io;

use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::transcript::Transcript;

use super::argument::{
    prove_sumcheck, verify_column_openings, verify_sumcheck, ClaimShape, ColumnOpening, Expr,
    Operand, StoreProverContext, SumcheckClaim, SumcheckProof, SumcheckShape, Term, Transcript254,
};
use super::bgp::{RelationCommit, RelationWitness};
use super::commitment::{LayerCommitment, VerifierCommitmentRef, VerifierLayer};
use super::dict::hash_object;
use super::dicthash::{
    honest_hash_lookup, prove_hash_lookup_inner, verify_hash_lookup_inner, HashLookupInnerProof,
};
use super::logup::{reciprocal_claim, reciprocal_witness, rlc_fold};
use super::relocate::{
    prove_column_relocation_with_context, verify_column_relocation, RelocationProof,
};
use super::union::{prove_bag_union_with_context, verify_bag_union, BagUnionProof};
use super::{invalid, CanonicalObject, ProofHash};

const RECONCILE_DOMAIN: &[u8] = b"parture/proof-of-woql/delta-reconcile/v0";

// Column commitment order inside `LayerCommitment::column_commitments`:
// additions (S,P,O) then removals (S,P,O).
const CC_ADD_S: usize = 0;
const CC_ADD_P: usize = 1;
const CC_ADD_O: usize = 2;
const CC_REM_S: usize = 3;
const CC_REM_P: usize = 4;
const CC_REM_O: usize = 5;

// Link column groups (order fixed for the 4-way multiset link):
//   0 = B.additions, 1 = D.additions, 2 = present, 3 = D.removals.
// The equality  present ⊎ D.rem ≡ B.add ⊎ D.add  becomes the zero-sum
//   Σ_x [ χ0·rec_Badd + χ1·rec_Dadd − χ2·rec_present − χ3·rec_Drem ] = 0.
const NUM_LINK_COLUMNS: usize = 16;

#[inline]
fn f0() -> ArkFr {
    ArkFr::zero()
}
#[inline]
fn f1() -> ArkFr {
    ArkFr::one()
}
#[inline]
fn ark(v: u64) -> ArkFr {
    ArkFr::from_u64(v)
}

fn pad_to(values: &[ArkFr], log: usize) -> Vec<ArkFr> {
    let mut out = values.to_vec();
    out.resize(1usize << log, f0());
    out
}

/// The relocatable power-of-two domain a group of `n` records is committed at
/// (`commit_record_id_column` pads to `next_pow2(n)`; this skeleton requires the
/// result to be a relocatable `>= 2` domain, i.e. `n >= 2`).
fn group_col_log(n: usize) -> io::Result<usize> {
    if n < 2 {
        return Err(invalid(
            "reconcile skeleton requires each committed group (B.add / D.add / D.rem) to have >= 2 records",
        ));
    }
    let padded = n.next_power_of_two();
    Ok(padded.trailing_zeros() as usize)
}

/// Node/value dictionary domain log for a catalog of `len` entries (matches
/// `node_value_dict_padded_len`: `next_pow2(len).max(2)`).
fn dict_log(len: usize) -> io::Result<usize> {
    if len == 0 {
        return Err(invalid("reconcile: empty node/value catalog"));
    }
    Ok(len.next_power_of_two().max(2).trailing_zeros() as usize)
}

/// One authenticated `id → hash` inner lookup against the combined dict, plus the
/// relocation that binds the obj id column into the shared working domain `2^K`.
#[derive(Clone, Debug)]
pub struct GroupHashProof {
    /// Relocate the root-bound obj id column (at its column domain) up to `2^K`.
    pub obj_reloc: RelocationProof,
    pub com_obj_at_k: ArkGT,
    /// The batched hash lookup over `[obj_at_k, combined dict at 2^K]`. Its
    /// `com_v_hash` is the authenticated hash column consumed by the link.
    pub lookup: HashLookupInnerProof,
}

/// A full base+delta reconciliation proof.
#[derive(Clone, Debug)]
pub struct ReconcileProof {
    // (1) Combined node/value dictionary = bag union of the two root-bound dicts.
    pub combined_dict: BagUnionProof,
    pub com_combined_hash: ArkGT,
    pub com_combined_id: ArkGT,
    pub combined_log: usize,
    pub combined_n: usize,
    // Combined dict relocated up to the shared working domain 2^K.
    pub reloc_dict_hash: RelocationProof,
    pub reloc_dict_id: RelocationProof,
    pub com_dict_hash_at_k: ArkGT,
    pub com_dict_id_at_k: ArkGT,
    // (2) Six authenticated subject/object hash lookups, in the fixed order
    //     [Badd_S, Badd_O, Dadd_S, Dadd_O, Drem_S, Drem_O].
    pub group_hashes: Vec<GroupHashProof>,
    // (3) Three predicate-id relocations, in the order [Badd_P, Dadd_P, Drem_P].
    pub pid_relocs: Vec<RelocationProof>,
    pub com_pid_at_k: Vec<ArkGT>,
    // (4) present's freshly committed S/P/O columns at 2^K.
    pub com_present_s: ArkGT,
    pub com_present_p: ArkGT,
    pub com_present_o: ArkGT,
    // (5) The 4-way masked LogUp link.
    pub com_rec: [ArkGT; 4],
    pub link_sumcheck: SumcheckProof,
    pub link_opening: ColumnOpening,
    // Bookkeeping.
    pub working_log: usize,
    pub present_len: usize,
}

/// The prover's reconciled `present` relation in canonical HASH space (returned so a
/// caller/test can assert it against the materialized ground truth).
pub struct ReconciledPresent {
    /// Per present triple, `(hash_object(subject), predicate_id, hash_object(object))`.
    pub rows: Vec<(ArkFr, u64, ArkFr)>,
}

fn subject_hash(record: &super::ProofRecord) -> ArkFr {
    hash_object(&CanonicalObject::Node(record.triple.subject.clone()))
}
fn object_hash(record: &super::ProofRecord) -> ArkFr {
    hash_object(&record.triple.object)
}

/// Reconcile `present = B.additions ⊎ D.additions − D.removals` by canonical triple
/// (the clean §3 multiset difference; rejects a removal absent from the union, i.e. a
/// violated invariant).
fn reconcile_present(base: &LayerCommitment, delta: &LayerCommitment) -> io::Result<Vec<super::ProofRecord>> {
    let mut present: Vec<super::ProofRecord> = base.additions.clone();
    present.extend(delta.additions.iter().cloned());
    for rem in &delta.removals {
        match present.iter().position(|r| r.triple == rem.triple) {
            Some(pos) => {
                present.remove(pos);
            }
            None => {
                return Err(invalid(
                    "reconcile: removal is absent from B.additions ⊎ D.additions (delta invariant violated)",
                ));
            }
        }
    }
    Ok(present)
}

/// Build a `RelationWitness` view of a layer's committed node/value dictionary as a
/// two-column `[hash, id]` relation, bound to the layer's root-committed
/// `node_value_dict_commitments`.
fn dict_branch_witness(layer: &LayerCommitment) -> io::Result<RelationWitness> {
    let (hash_col, id_col) = layer.node_value_dictionary_columns()?;
    let n = layer.node_value_catalog.len();
    let log = dict_log(n)?;
    if hash_col.len() != 1usize << log || id_col.len() != 1usize << log {
        return Err(invalid("reconcile: dict column length does not match its domain"));
    }
    Ok(RelationWitness {
        columns: vec![hash_col, id_col],
        commitments: vec![
            layer.node_value_dict_commitments.0,
            layer.node_value_dict_commitments.1,
        ],
        log_domain: log,
        n,
    })
}

fn dict_branch_commit(layer_view: &VerifierCommitmentRef<'_>) -> io::Result<RelationCommit> {
    let n = layer_view.node_value_catalog.len();
    let log = dict_log(n)?;
    Ok(RelationCommit {
        commitments: vec![
            layer_view.node_value_dict_commitments.0,
            layer_view.node_value_dict_commitments.1,
        ],
        log_domain: log,
        n,
    })
}

/// Prove the base+delta reconciliation and the reconciled triple count.
///
/// `trusted_head_root` MUST be the externally-trusted head anchor; it is required to
/// equal `delta.state.commitment_root`.
pub fn prove_reconcile(
    base: &LayerCommitment,
    delta: &LayerCommitment,
) -> io::Result<(ReconcileProof, ReconciledPresent)> {
    prove_reconcile_with_context(&StoreProverContext::cpu(), base, delta)
}

pub fn prove_reconcile_with_context(
    context: &StoreProverContext<'_>,
    base: &LayerCommitment,
    delta: &LayerCommitment,
) -> io::Result<(ReconcileProof, ReconciledPresent)> {
    // ---- Structural / chain preconditions (mirrored by the verifier) ----
    if base.state.parent_state_root.is_some() {
        return Err(invalid("reconcile: base layer must be a root (no parent)"));
    }
    if delta.state.parent_state_root != Some(base.state.commitment_root) {
        return Err(invalid("reconcile: delta does not chain the base commitment root"));
    }
    if delta.state.parent_node_value_count != base.state.node_value_count {
        return Err(invalid("reconcile: delta parent node/value count must equal base node/value count"));
    }
    if delta.state.predicate_count != delta.state.parent_predicate_count
        || delta.state.parent_predicate_count != base.state.predicate_count
    {
        return Err(invalid(
            "reconcile skeleton: delta must introduce no new predicates (predicate ids stay layer-consistent)",
        ));
    }

    let n_badd = base.additions.len();
    let n_dadd = delta.additions.len();
    let n_drem = delta.removals.len();
    let badd_log = group_col_log(n_badd)?;
    let dadd_log = group_col_log(n_dadd)?;
    let drem_log = group_col_log(n_drem)?;

    // ---- Reconcile present (ground truth for the prover) ----
    let present = reconcile_present(base, delta)?;
    let present_len = present.len();
    let present_log = present_len.next_power_of_two().max(2).trailing_zeros() as usize;

    let mut transcript = Transcript254::new(b"parture-delta-reconcile-envelope");
    transcript.append_bytes(b"pow/reconcile/domain", RECONCILE_DOMAIN);
    transcript.append_bytes(b"pow/reconcile/base-root", &base.state.commitment_root);
    transcript.append_bytes(b"pow/reconcile/delta-root", &delta.state.commitment_root);

    // ---- (1) Combined dictionary = bag union of base dict and delta incremental dict ----
    let base_branch = dict_branch_witness(base)?;
    let delta_branch = dict_branch_witness(delta)?;
    let union = prove_bag_union_with_context(context, &[base_branch, delta_branch], &mut transcript)?;
    let combined_hash = union.columns[0].clone();
    let combined_id = union.columns[1].clone();
    let com_combined_hash = union.commitments[0];
    let com_combined_id = union.commitments[1];
    let combined_log = union.log_domain;
    let combined_n = union.n; // = base_nv + delta_nv (all reachable ids)

    // ---- Working domain 2^K: big enough for the combined dict and every group ----
    let working_log = combined_log
        .max(present_log)
        .max(badd_log)
        .max(dadd_log)
        .max(drem_log);
    let k_size = 1usize << working_log;

    // Relocate the combined dict up to 2^K (same-domain when combined_log == K).
    let dict_hash_at_k = pad_to(&combined_hash, working_log);
    let dict_id_at_k = pad_to(&combined_id, working_log);
    let com_dict_hash_at_k = context.commit_column(&dict_hash_at_k)?;
    let com_dict_id_at_k = context.commit_column(&dict_id_at_k)?;
    let reloc_dict_hash = prove_column_relocation_with_context(
        context, &combined_hash, &com_combined_hash, &dict_hash_at_k, &com_dict_hash_at_k, &mut transcript,
    )?;
    let reloc_dict_id = prove_column_relocation_with_context(
        context, &combined_id, &com_combined_id, &dict_id_at_k, &com_dict_id_at_k, &mut transcript,
    )?;

    // ---- (2) Six authenticated subject/object hash lookups ----
    // (records, id-selector, source column commitment, source column log, real length)
    let sub = (|r: &super::ProofRecord| r.subject_id) as fn(&super::ProofRecord) -> u64;
    let obj = (|r: &super::ProofRecord| r.object_id) as fn(&super::ProofRecord) -> u64;
    let obj_specs = [
        (&base.additions, sub, base.column_commitments[CC_ADD_S], badd_log, n_badd),
        (&base.additions, obj, base.column_commitments[CC_ADD_O], badd_log, n_badd),
        (&delta.additions, sub, delta.column_commitments[CC_ADD_S], dadd_log, n_dadd),
        (&delta.additions, obj, delta.column_commitments[CC_ADD_O], dadd_log, n_dadd),
        (&delta.removals, sub, delta.column_commitments[CC_REM_S], drem_log, n_drem),
        (&delta.removals, obj, delta.column_commitments[CC_REM_O], drem_log, n_drem),
    ];

    let mut group_hashes = Vec::with_capacity(6);
    let mut group_vhash_data: Vec<Vec<ArkFr>> = Vec::with_capacity(6);
    for (records, select, com_source, source_log, real_n) in obj_specs {
        // Reconstruct the root-bound obj id column at its own commitment domain.
        let mut obj_source = vec![f0(); 1usize << source_log];
        for (slot, record) in obj_source.iter_mut().zip(records.iter()) {
            *slot = ark(select(record));
        }
        let obj_at_k = pad_to(&obj_source, working_log);
        let com_obj_at_k = context.commit_column(&obj_at_k)?;
        let obj_reloc = prove_column_relocation_with_context(
            context, &obj_source, &com_source, &obj_at_k, &com_obj_at_k, &mut transcript,
        )?;
        let lookup = prove_hash_lookup_inner(
            context, &obj_at_k, &com_obj_at_k, &dict_id_at_k, &dict_hash_at_k,
            &com_dict_id_at_k, &com_dict_hash_at_k, real_n, combined_n, &mut transcript,
        )?;
        // The authenticated v_hash column data for the link (fully determined by the proof).
        let v_hash = honest_hash_lookup(&obj_at_k, &dict_id_at_k, &dict_hash_at_k, real_n, combined_n).0;
        group_hashes.push(GroupHashProof { obj_reloc, com_obj_at_k, lookup });
        group_vhash_data.push(v_hash);
    }

    // ---- (3) Three predicate-id relocations ----
    let pid_specs: [(&Vec<super::ProofRecord>, ArkGT, usize); 3] = [
        (&base.additions, base.column_commitments[CC_ADD_P], badd_log),
        (&delta.additions, delta.column_commitments[CC_ADD_P], dadd_log),
        (&delta.removals, delta.column_commitments[CC_REM_P], drem_log),
    ];
    let mut pid_relocs = Vec::with_capacity(3);
    let mut com_pid_at_k = Vec::with_capacity(3);
    let mut pid_data: Vec<Vec<ArkFr>> = Vec::with_capacity(3);
    for (records, com_source, source_log) in pid_specs {
        let mut pid_source = vec![f0(); 1usize << source_log];
        for (slot, record) in pid_source.iter_mut().zip(records.iter()) {
            *slot = ark(record.predicate_id);
        }
        let pid_at_k = pad_to(&pid_source, working_log);
        let com_pid = context.commit_column(&pid_at_k)?;
        let reloc = prove_column_relocation_with_context(
            context, &pid_source, &com_source, &pid_at_k, &com_pid, &mut transcript,
        )?;
        pid_relocs.push(reloc);
        com_pid_at_k.push(com_pid);
        pid_data.push(pid_at_k);
    }

    // ---- (4) present's canonical S/P/O columns at 2^K ----
    let mut present_s = vec![f0(); k_size];
    let mut present_p = vec![f0(); k_size];
    let mut present_o = vec![f0(); k_size];
    let mut present_rows = Vec::with_capacity(present_len);
    for (i, record) in present.iter().enumerate() {
        let sh = subject_hash(record);
        let oh = object_hash(record);
        present_s[i] = sh;
        present_p[i] = ark(record.predicate_id);
        present_o[i] = oh;
        present_rows.push((sh, record.predicate_id, oh));
    }
    let com_present_s = context.commit_column(&present_s)?;
    let com_present_p = context.commit_column(&present_p)?;
    let com_present_o = context.commit_column(&present_o)?;

    // ---- (5) 4-way masked LogUp link ----
    // Group value columns (S,P,O) in the fixed link order.
    let group_value_data: [[&Vec<ArkFr>; 3]; 4] = [
        [&group_vhash_data[0], &pid_data[0], &group_vhash_data[1]], // B.add
        [&group_vhash_data[2], &pid_data[1], &group_vhash_data[3]], // D.add
        [&present_s, &present_p, &present_o],                       // present
        [&group_vhash_data[4], &pid_data[2], &group_vhash_data[5]], // D.rem
    ];
    let group_value_com: [[ArkGT; 3]; 4] = [
        [group_hashes[0].lookup.com_v_hash, com_pid_at_k[0], group_hashes[1].lookup.com_v_hash],
        [group_hashes[2].lookup.com_v_hash, com_pid_at_k[1], group_hashes[3].lookup.com_v_hash],
        [com_present_s, com_present_p, com_present_o],
        [group_hashes[4].lookup.com_v_hash, com_pid_at_k[2], group_hashes[5].lookup.com_v_hash],
    ];
    let group_len = [n_badd, n_dadd, present_len, n_drem];

    for coms in &group_value_com {
        for com in coms {
            transcript.append_group(b"pow/reconcile/link-val", com);
        }
    }
    let beta = transcript.challenge_scalar(b"pow/reconcile/link-beta");
    let alpha = transcript.challenge_scalar(b"pow/reconcile/link-alpha");

    let mut rec_data: Vec<Vec<ArkFr>> = Vec::with_capacity(4);
    let mut com_rec_vec: Vec<ArkGT> = Vec::with_capacity(4);
    for values in &group_value_data {
        let fold = rlc_fold(beta, &[values[0].as_slice(), values[1].as_slice(), values[2].as_slice()]);
        let rec = reciprocal_witness(alpha, &fold);
        com_rec_vec.push(context.commit_column(&rec)?);
        rec_data.push(rec);
    }
    for com in &com_rec_vec {
        transcript.append_group(b"pow/reconcile/link-rec", com);
    }

    // Assemble the 16 link columns [s,p,o,rec] × 4 groups.
    let mut columns: Vec<Vec<ArkFr>> = Vec::with_capacity(NUM_LINK_COLUMNS);
    for g in 0..4 {
        columns.push(group_value_data[g][0].clone());
        columns.push(group_value_data[g][1].clone());
        columns.push(group_value_data[g][2].clone());
        columns.push(rec_data[g].clone());
    }
    let shape = link_shape(working_log, group_len, alpha, beta);
    let (link_sumcheck, r, _evals) = prove_sumcheck(&shape, &columns, &mut transcript)?;
    let link_opening = context.prove_column_openings(&columns, &r, &mut transcript)?;

    let com_rec = [com_rec_vec[0], com_rec_vec[1], com_rec_vec[2], com_rec_vec[3]];

    Ok((
        ReconcileProof {
            combined_dict: union.proof,
            com_combined_hash,
            com_combined_id,
            combined_log,
            combined_n,
            reloc_dict_hash,
            reloc_dict_id,
            com_dict_hash_at_k,
            com_dict_id_at_k,
            group_hashes,
            pid_relocs,
            com_pid_at_k,
            com_present_s,
            com_present_p,
            com_present_o,
            com_rec,
            link_sumcheck,
            link_opening,
            working_log,
            present_len,
        },
        ReconciledPresent { rows: present_rows },
    ))
}

/// The 4-way link sumcheck shape: four reciprocal (Identity) claims pinning each
/// group's `rec = 1/(alpha − fold[S,P,O])`, plus the masked multiset-equality link.
fn link_shape(k: usize, lens: [usize; 4], alpha: ArkFr, beta: ArkFr) -> SumcheckShape {
    let mut claims = Vec::with_capacity(5);
    for g in 0..4 {
        claims.push(reciprocal_claim(
            g * 4 + 3,
            &[g * 4, g * 4 + 1, g * 4 + 2],
            alpha,
            beta,
        ));
    }
    // Σ_x [ χ0·rec0 + χ1·rec1 − χ2·rec2 − χ3·rec3 ] = 0.
    let link = SumcheckClaim {
        shape: ClaimShape::ZeroSum,
        expr: Expr {
            terms: vec![
                Term { coeff: f1(), operands: vec![Operand::Chi(0), Operand::Column(3)] },
                Term { coeff: f1(), operands: vec![Operand::Chi(1), Operand::Column(7)] },
                Term { coeff: f0() - f1(), operands: vec![Operand::Chi(2), Operand::Column(11)] },
                Term { coeff: f0() - f1(), operands: vec![Operand::Chi(3), Operand::Column(15)] },
            ],
        },
        target: f0(),
    };
    claims.push(link);
    SumcheckShape {
        num_vars: k,
        num_columns: NUM_LINK_COLUMNS,
        chi_bounds: vec![lens[0], lens[1], lens[2], lens[3]],
        claims,
    }
}

/// Verify a base+delta reconciliation. On success returns the authenticated
/// reconciled triple COUNT `M` (the single-atom count query answer over `present`).
pub fn verify_reconcile(
    base: &impl VerifierLayer,
    delta: &impl VerifierLayer,
    trusted_head_root: ProofHash,
    proof: &ReconcileProof,
) -> io::Result<usize> {
    let base_view = base.verifier_view();
    let delta_view = delta.verifier_view();
    // (1) Each layer authenticated against ITS OWN commitment root.
    base_view.validate()?;
    delta_view.validate()?;

    // (2) Chain + trusted head anchor.
    if base_view.state.parent_state_root.is_some() {
        return Err(invalid("reconcile: base layer must be a root (no parent)"));
    }
    if delta_view.state.parent_state_root != Some(base_view.state.commitment_root) {
        return Err(invalid("reconcile: delta does not chain the base commitment root"));
    }
    if delta_view.state.commitment_root != trusted_head_root {
        return Err(invalid("reconcile: delta head root does not match the trusted anchor"));
    }
    // (3) Skeleton guards: dict extends base; no new predicates.
    if delta_view.state.parent_node_value_count != base_view.state.node_value_count {
        return Err(invalid("reconcile: delta parent node/value count must equal base node/value count"));
    }
    if delta_view.state.predicate_count != delta_view.state.parent_predicate_count
        || delta_view.state.parent_predicate_count != base_view.state.predicate_count
    {
        return Err(invalid(
            "reconcile skeleton: delta must introduce no new predicates",
        ));
    }

    let n_badd = base_view.state.addition_count as usize;
    let n_dadd = delta_view.state.addition_count as usize;
    let n_drem = delta_view.state.removal_count as usize;
    let badd_log = group_col_log(n_badd)?;
    let dadd_log = group_col_log(n_dadd)?;
    let drem_log = group_col_log(n_drem)?;

    // The reconciled count is fixed by the public delta counts (verifier-computed,
    // never read from the proof). This is the count query's answer AND the length
    // mask that makes the multiset link pass only for the true reconciled present.
    let present_len = n_badd
        .checked_add(n_dadd)
        .and_then(|s| s.checked_sub(n_drem))
        .ok_or_else(|| invalid("reconcile: present length underflow"))?;
    if present_len != proof.present_len {
        return Err(invalid("reconcile: claimed present length disagrees with the delta counts"));
    }
    let present_log = present_len.next_power_of_two().max(2).trailing_zeros() as usize;

    let mut transcript = Transcript254::new(b"parture-delta-reconcile-envelope");
    transcript.append_bytes(b"pow/reconcile/domain", RECONCILE_DOMAIN);
    transcript.append_bytes(b"pow/reconcile/base-root", &base_view.state.commitment_root);
    transcript.append_bytes(b"pow/reconcile/delta-root", &delta_view.state.commitment_root);

    // (1) Verify the combined dict = bag union of the two root-bound dicts.
    let base_branch = dict_branch_commit(&base_view)?;
    let delta_branch = dict_branch_commit(&delta_view)?;
    let combined_n = base_branch.n + delta_branch.n;
    if proof.combined_n != combined_n {
        return Err(invalid("reconcile: combined dict length mismatch"));
    }
    verify_bag_union(
        &[base_branch, delta_branch],
        &[proof.com_combined_hash, proof.com_combined_id],
        combined_n,
        proof.combined_log,
        &proof.combined_dict,
        &mut transcript,
    )?;

    let working_log = proof.combined_log
        .max(present_log)
        .max(badd_log)
        .max(dadd_log)
        .max(drem_log);
    if working_log != proof.working_log {
        return Err(invalid("reconcile: working domain mismatch"));
    }

    verify_column_relocation(
        &proof.com_combined_hash, proof.combined_log, &proof.com_dict_hash_at_k, working_log,
        &proof.reloc_dict_hash, &mut transcript,
    )?;
    verify_column_relocation(
        &proof.com_combined_id, proof.combined_log, &proof.com_dict_id_at_k, working_log,
        &proof.reloc_dict_id, &mut transcript,
    )?;

    // (2) Verify the six subject/object hash lookups against the combined dict.
    if proof.group_hashes.len() != 6 {
        return Err(invalid("reconcile: expected six group hash proofs"));
    }
    let obj_specs: [(ArkGT, usize, usize); 6] = [
        (base_view.column_commitments[CC_ADD_S], badd_log, n_badd),
        (base_view.column_commitments[CC_ADD_O], badd_log, n_badd),
        (delta_view.column_commitments[CC_ADD_S], dadd_log, n_dadd),
        (delta_view.column_commitments[CC_ADD_O], dadd_log, n_dadd),
        (delta_view.column_commitments[CC_REM_S], drem_log, n_drem),
        (delta_view.column_commitments[CC_REM_O], drem_log, n_drem),
    ];
    for group in 0..6 {
        let (com_source, source_log, real_n) = obj_specs[group];
        let gh = &proof.group_hashes[group];
        verify_column_relocation(
            &com_source, source_log, &gh.com_obj_at_k, working_log, &gh.obj_reloc, &mut transcript,
        )?;
        verify_hash_lookup_inner(
            &gh.com_obj_at_k, &proof.com_dict_id_at_k, &proof.com_dict_hash_at_k, real_n, combined_n,
            &gh.lookup, &mut transcript,
        )?;
    }

    // (3) Verify the three predicate-id relocations.
    if proof.pid_relocs.len() != 3 || proof.com_pid_at_k.len() != 3 {
        return Err(invalid("reconcile: expected three predicate relocations"));
    }
    let pid_specs: [(ArkGT, usize); 3] = [
        (base_view.column_commitments[CC_ADD_P], badd_log),
        (delta_view.column_commitments[CC_ADD_P], dadd_log),
        (delta_view.column_commitments[CC_REM_P], drem_log),
    ];
    for i in 0..3 {
        let (com_source, source_log) = pid_specs[i];
        verify_column_relocation(
            &com_source, source_log, &proof.com_pid_at_k[i], working_log, &proof.pid_relocs[i], &mut transcript,
        )?;
    }

    // (5) Verify the 4-way masked LogUp link.
    let group_value_com: [[ArkGT; 3]; 4] = [
        [proof.group_hashes[0].lookup.com_v_hash, proof.com_pid_at_k[0], proof.group_hashes[1].lookup.com_v_hash],
        [proof.group_hashes[2].lookup.com_v_hash, proof.com_pid_at_k[1], proof.group_hashes[3].lookup.com_v_hash],
        [proof.com_present_s, proof.com_present_p, proof.com_present_o],
        [proof.group_hashes[4].lookup.com_v_hash, proof.com_pid_at_k[2], proof.group_hashes[5].lookup.com_v_hash],
    ];
    let group_len = [n_badd, n_dadd, present_len, n_drem];

    for coms in &group_value_com {
        for com in coms {
            transcript.append_group(b"pow/reconcile/link-val", com);
        }
    }
    let beta = transcript.challenge_scalar(b"pow/reconcile/link-beta");
    let alpha = transcript.challenge_scalar(b"pow/reconcile/link-alpha");
    for com in &proof.com_rec {
        transcript.append_group(b"pow/reconcile/link-rec", com);
    }

    if proof.link_sumcheck.round_polys.len() != working_log {
        return Err(invalid("reconcile: link sumcheck domain mismatch"));
    }
    let shape = link_shape(working_log, group_len, alpha, beta);
    let (r, evals) = verify_sumcheck(&shape, &proof.link_sumcheck, &mut transcript)?;
    let mut commitments = Vec::with_capacity(NUM_LINK_COLUMNS);
    for g in 0..4 {
        commitments.push(group_value_com[g][0]);
        commitments.push(group_value_com[g][1]);
        commitments.push(group_value_com[g][2]);
        commitments.push(proof.com_rec[g]);
    }
    verify_column_openings(&commitments, &r, &evals, &proof.link_opening, &mut transcript)?;

    Ok(present_len)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::ValueTriple;
    use crate::store::sync::open_sync_memory_store;
    use std::sync::Arc;

    fn iri(s: &str) -> String {
        format!("http://ex/{s}")
    }

    /// Base { a1..a4 -> p1 -> b1..b4 }; delta REMOVES {a1->b1, a2->b2} and ADDS
    /// {a3->b5 (b5 new), a5->b6 (a5,b6 new)}. So present = {a3->b3, a4->b4, a3->b5,
    /// a5->b6}, materialized count 4. Exercises inherited AND D-local ids in D.add.
    fn build_base_delta() -> (Arc<LayerCommitment>, Arc<LayerCommitment>) {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for (s, o) in [("a1", "b1"), ("a2", "b2"), ("a3", "b3"), ("a4", "b4")] {
            builder
                .add_value_triple(ValueTriple::new_node(&iri(s), &iri("p1"), &iri(o)))
                .unwrap();
        }
        let base = builder.commit().unwrap();
        let base_lc = base.proof_commitment().unwrap();

        let child = base.open_write().unwrap();
        for (s, o) in [("a1", "b1"), ("a2", "b2")] {
            child
                .remove_value_triple(ValueTriple::new_node(&iri(s), &iri("p1"), &iri(o)))
                .unwrap();
        }
        for (s, o) in [("a3", "b5"), ("a5", "b6")] {
            child
                .add_value_triple(ValueTriple::new_node(&iri(s), &iri("p1"), &iri(o)))
                .unwrap();
        }
        let head = child.commit().unwrap();
        let delta_lc = head.proof_commitment().unwrap();
        (base_lc, delta_lc)
    }

    /// The materialized ground-truth present set as canonical (S,P,O) hash rows.
    fn materialized_rows(base: &LayerCommitment, delta: &LayerCommitment) -> Vec<(ArkFr, u64, ArkFr)> {
        let present = reconcile_present(base, delta).unwrap();
        present
            .iter()
            .map(|r| (subject_hash(r), r.predicate_id, object_hash(r)))
            .collect()
    }

    fn bag_eq(mut a: Vec<(ArkFr, u64, ArkFr)>, mut b: Vec<(ArkFr, u64, ArkFr)>) -> bool {
        if a.len() != b.len() {
            return false;
        }
        let key = |t: &(ArkFr, u64, ArkFr)| (format!("{:?}", t.0), t.1, format!("{:?}", t.2));
        a.sort_by_key(&key);
        b.sort_by_key(&key);
        a == b
    }

    #[test]
    fn reconciled_present_count_matches_materialized() {
        let (base, delta) = build_base_delta();
        let (proof, present) = prove_reconcile(&base, &delta).unwrap();
        let m = verify_reconcile(&*base, &*delta, delta.state.commitment_root, &proof).unwrap();

        // The authenticated count equals the materialized triple count (ground truth).
        let ground = materialized_rows(&base, &delta);
        assert_eq!(m, ground.len(), "reconciled count must equal materialized count");
        assert_eq!(m, 4);
        // present (the prover's committed relation) equals the reconciled ground truth.
        assert!(
            bag_eq(present.rows.clone(), ground),
            "reconciled present bag must equal the materialized bag"
        );
    }

    #[test]
    fn wrong_trusted_head_root_rejects() {
        let (base, delta) = build_base_delta();
        let (proof, _) = prove_reconcile(&base, &delta).unwrap();
        // Head anchored to the BASE root (or any non-delta root) must reject.
        assert!(verify_reconcile(&*base, &*delta, base.state.commitment_root, &proof).is_err());
        let mut bogus = delta.state.commitment_root;
        bogus[0] ^= 0xff;
        assert!(verify_reconcile(&*base, &*delta, bogus, &proof).is_err());
    }

    #[test]
    fn swapped_layers_reject() {
        let (base, delta) = build_base_delta();
        let (proof, _) = prove_reconcile(&base, &delta).unwrap();
        // Swapping base/delta breaks the chain (base has no parent; delta is not a root).
        assert!(verify_reconcile(&*delta, &*base, base.state.commitment_root, &proof).is_err());
        assert!(verify_reconcile(&*delta, &*base, delta.state.commitment_root, &proof).is_err());
    }

    #[test]
    fn tampered_present_row_rejects() {
        let (base, delta) = build_base_delta();
        let (mut proof, _) = prove_reconcile(&base, &delta).unwrap();
        // Replace present's committed subject column with a fabricated one: the link
        // (with the verifier-computed present length) no longer balances.
        let bogus = crate::proof::argument::commit_column(&vec![ark(999); 1usize << proof.working_log]).unwrap();
        proof.com_present_s = bogus;
        assert!(verify_reconcile(&*base, &*delta, delta.state.commitment_root, &proof).is_err());
    }

    #[test]
    fn hidden_removal_rejects() {
        // A prover who tries to HIDE a removal (to fabricate presence of a removed
        // triple) must misrepresent the root-bound D.removals column set. Tampering the
        // authenticated removal-subject hash column breaks its lookup binding, and even
        // if it passed, the 4-way link would no longer balance.
        let (base, delta) = build_base_delta();
        let (mut proof, _) = prove_reconcile(&base, &delta).unwrap();
        // group_hashes[4] = D.removals subject hash lookup.
        let bogus = crate::proof::argument::commit_column(&vec![ark(13); 1usize << proof.working_log]).unwrap();
        proof.group_hashes[4].lookup.com_v_hash = bogus;
        assert!(verify_reconcile(&*base, &*delta, delta.state.commitment_root, &proof).is_err());
    }

    #[test]
    fn injected_removal_rejects() {
        // Claiming an extra removal (to drop a live row) changes the reconciled count.
        // The verifier computes present length from the root-bound delta counts, so a
        // forged present length is rejected (count binding).
        let (base, delta) = build_base_delta();
        let (proof, _) = prove_reconcile(&base, &delta).unwrap();
        let mut forged = proof.clone();
        forged.present_len -= 1;
        assert!(verify_reconcile(&*base, &*delta, delta.state.commitment_root, &forged).is_err());
    }

    #[test]
    fn tampered_combined_dict_rejects() {
        let (base, delta) = build_base_delta();
        let (mut proof, _) = prove_reconcile(&base, &delta).unwrap();
        // Corrupt the combined dict hash commitment: the relocation / lookup binding
        // to the root-bound branch dicts fails.
        let bogus = crate::proof::argument::commit_column(&vec![ark(7); 1usize << proof.combined_log]).unwrap();
        proof.com_combined_hash = bogus;
        assert!(verify_reconcile(&*base, &*delta, delta.state.commitment_root, &proof).is_err());
    }
}
