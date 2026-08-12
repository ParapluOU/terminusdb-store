//! Delta-stack RECONCILIATION over an N-layer parent chain (Waves 1-3 core).
//!
//! Proves a read query over a **base + k deltas** layer stack by RECONCILING the
//! current triple set at proof time and authenticating it against the committed
//! per-layer deltas — WITHOUT any materialized head view (the Wave-0 commitment
//! diet removed it). The read delivered here is the reconciled triple COUNT (the
//! exact single-atom count answer over `present`); the full BGP/filter/range/join
//! over `present` is the next pass.
//!
//! # The math (reconciliation = a multiset equality)
//! terminus-store maintains the delta invariant (add only absent triples, remove
//! only present ones), so the current set `present` satisfies a CLEAN multiset
//! equality over canonical (dictionary-independent) terms, chain-wide:
//! ```text
//! present  ⊎  (⊎_layers removals)   ≡   (⊎_layers additions)
//! ```
//! (a removal in layer `j` cancels exactly one earlier addition). The prover supplies
//! `present`; this module proves the equality via a two-sided masked LogUp link (the
//! discipline that closed the self-join padding bug), which authenticates `present`
//! against the committed delta columns. Under the invariant `present` is duplicate-free.
//!
//! # Cross-layer dictionary authentication (Wave 1, N-layer)
//! Each layer commits only the dictionary entries IT introduced
//! (`[parent_*_count+1 ..= *_count]`); a term's id is defined in exactly the layer
//! whose incremental range contains it, and the chain's ranges abut disjointly. We
//! reconcile in canonical-term-HASH space so a term hashes identically regardless of
//! which layer minted its id:
//!   * a COMBINED node/value dictionary is the exact bag union of EVERY layer's
//!     incremental node/value dict (each bound to its own `commitment_root`), so it
//!     resolves any id from anywhere in the chain in one authenticated table;
//!   * a COMBINED predicate dictionary is the same over every layer's incremental
//!     predicate dict, hashing predicates with `hash_term`;
//!   * a triple's canonical identity is `(hash_object(S), hash_term(P), hash_object(O))`,
//!     every component root-authenticated via the audited [`super::dicthash`] lookup
//!     inner against the combined dict.
//!
//! # Root binding (Wave 2 envelope, N-layer)
//! The verifier validates each layer against ITS OWN `commitment_root`, checks the
//! whole parent chain (`layer_i.parent_state_root == layer_{i-1}.commitment_root`, the
//! base a root, the head pinned to the externally-trusted `trusted_head_root`) and the
//! dictionary-count abutment. No layer may be spliced, reordered, dropped, or foreign.
//!
//! # Scope of THIS pass (deferred — see the module report)
//!   * the query is the reconciled triple COUNT; BGP/filter/range over `present` is next;
//!   * each non-empty committed group (a layer's additions or removals) must have ≥ 2
//!     records (so its column commitment lives at a relocatable ≥ 2 domain); single-row
//!     deltas need a relocation-gadget extension to the `2^0` domain.

use std::io;

use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::transcript::Transcript;

use super::argument::{
    commit_column, prove_sumcheck, verify_column_openings, verify_sumcheck, ClaimShape,
    ColumnOpening, Expr, Operand, StoreProverContext, SumcheckClaim, SumcheckProof, SumcheckShape,
    Term, Transcript254,
};
use super::anti::{
    prove_inner_join_with_keys_and_context, verify_inner_join_with_keys, InnerJoinProof,
};
use super::bgp::{
    prove_bgp_chain_with_result_and_context, verify_bgp_chain, BgpChainProof, JoinStep,
    RelationCommit, RelationWitness,
};
use super::commitment::{LayerCommitment, VerifierCommitmentRef, VerifierLayer};
use super::dict::{hash_object, hash_term};
use super::filter::{
    prove_compaction_only_with_context, prove_filter_indicator_with_context,
    prove_selection_only_with_context, verify_compaction_only, verify_filter_indicator,
    verify_selection_only, AtomFilterSpec, CompactionOnlyProof, FilterProof, Position,
    SelectionOnlyProof,
};
use super::dicthash::{
    honest_hash_lookup, prove_hash_lookup_inner, verify_hash_lookup_inner, HashLookupInnerProof,
};
use super::logup::{reciprocal_claim, reciprocal_witness, rlc_fold};
use super::relocate::{
    prove_column_relocation_with_context, verify_column_relocation, RelocationProof,
};
use super::union::{prove_bag_union_with_context, verify_bag_union, BagUnionProof};
use super::{invalid, CanonicalObject, ProofHash, ProofRecord};

const RECONCILE_DOMAIN: &[u8] = b"parture/proof-of-woql/delta-reconcile/v1-nlayer";

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
/// (`commit_record_id_column` pads to `next_pow2(n)`; this pass requires the result
/// to be a relocatable `>= 2` domain, i.e. `n >= 2`).
fn group_col_log(n: usize) -> io::Result<usize> {
    if n < 2 {
        return Err(invalid(
            "reconcile: each non-empty committed group must have >= 2 records (single-row deltas unsupported this pass)",
        ));
    }
    Ok(n.next_power_of_two().trailing_zeros() as usize)
}

/// Dictionary domain log for a catalog of `len` entries (matches
/// `node_value_dict_padded_len`/`padded_domain_for_rows`: `next_pow2(len).max(2)`).
fn dict_log(len: usize) -> io::Result<usize> {
    if len == 0 {
        return Err(invalid("reconcile: empty dictionary catalog"));
    }
    Ok(len.next_power_of_two().max(2).trailing_zeros() as usize)
}

/// One authenticated `id → hash` inner lookup against a combined dict, plus the
/// relocation that binds the root-committed obj id column into the working domain.
#[derive(Clone, Debug)]
pub struct GroupHashProof {
    pub obj_reloc: RelocationProof,
    pub com_obj_at_k: ArkGT,
    pub lookup: HashLookupInnerProof,
}

/// A committed group's three authenticated canonical hash columns: subject and object
/// against the combined node/value dict, predicate against the combined predicate dict.
#[derive(Clone, Debug)]
pub struct CommittedGroupProof {
    pub s: GroupHashProof,
    pub o: GroupHashProof,
    pub p: GroupHashProof,
}

/// A combined dictionary (bag union of every layer's incremental dict, or a single
/// root-bound dict when only one layer contributes) relocated to the working domain.
#[derive(Clone, Debug)]
pub struct CombinedDictProof {
    /// Present iff two or more layers contribute a dict (else a single-layer direct dict).
    pub union: Option<BagUnionProof>,
    pub com_hash: ArkGT,
    pub com_id: ArkGT,
    pub log: usize,
    pub n: usize,
    pub reloc_hash: RelocationProof,
    pub reloc_id: RelocationProof,
    pub com_hash_at_k: ArkGT,
    pub com_id_at_k: ArkGT,
}

/// A full N-layer reconciliation proof.
#[derive(Clone, Debug)]
pub struct ReconcileProof {
    pub combined_nv: CombinedDictProof,
    pub combined_pred: CombinedDictProof,
    pub working_log: usize,
    /// Committed groups (each layer's additions then removals, in chain order).
    pub groups: Vec<CommittedGroupProof>,
    /// present's freshly committed canonical S/P/O hash columns at 2^K.
    pub com_present_s: ArkGT,
    pub com_present_p: ArkGT,
    pub com_present_o: ArkGT,
    /// One reciprocal commitment per group, then present's, in link order.
    pub com_rec: Vec<ArkGT>,
    pub link_sumcheck: SumcheckProof,
    pub link_opening: ColumnOpening,
    pub present_len: usize,
}

/// The prover's reconciled `present` relation in canonical HASH space (returned so a
/// caller/test can assert it against the materialized ground truth).
pub struct ReconciledPresent {
    /// Per present triple, `(hash_object(subject), hash_term(predicate), hash_object(object))`.
    pub rows: Vec<(ArkFr, ArkFr, ArkFr)>,
}

fn subject_hash(record: &ProofRecord) -> ArkFr {
    hash_object(&CanonicalObject::Node(record.triple.subject.clone()))
}
fn predicate_hash(record: &ProofRecord) -> ArkFr {
    hash_term(&record.triple.predicate)
}
fn object_hash(record: &ProofRecord) -> ArkFr {
    hash_object(&record.triple.object)
}

/// Reconcile `present = (⊎ additions) − (⊎ removals)` by canonical triple, chain-wide
/// (pools every layer's additions, then cancels one addition per removal; rejects a
/// removal absent from the pool, i.e. a violated §3 invariant).
fn reconcile_present(layers: &[&LayerCommitment]) -> io::Result<Vec<ProofRecord>> {
    let mut present: Vec<ProofRecord> = Vec::new();
    for layer in layers {
        present.extend(layer.additions.iter().cloned());
    }
    for layer in layers {
        for rem in &layer.removals {
            match present.iter().position(|r| r.triple == rem.triple) {
                Some(pos) => {
                    present.remove(pos);
                }
                None => {
                    return Err(invalid(
                        "reconcile: a removal is absent from the pooled additions (delta invariant violated)",
                    ));
                }
            }
        }
    }
    Ok(present)
}

// -- Structural chain validation (mirrored by prover and verifier) --

fn validate_chain_states(states: &[&super::commitment::CommitmentState]) -> io::Result<()> {
    if states.len() < 2 {
        return Err(invalid("reconcile: need a base plus at least one delta layer"));
    }
    if states[0].parent_state_root.is_some() {
        return Err(invalid("reconcile: base layer must be a root (no parent)"));
    }
    for i in 1..states.len() {
        if states[i].parent_state_root != Some(states[i - 1].commitment_root) {
            return Err(invalid("reconcile: broken parent-state-root chain (spliced/reordered/foreign layer)"));
        }
        if states[i].parent_node_value_count != states[i - 1].node_value_count {
            return Err(invalid("reconcile: node/value dict counts do not abut along the chain"));
        }
        if states[i].parent_predicate_count != states[i - 1].predicate_count {
            return Err(invalid("reconcile: predicate dict counts do not abut along the chain"));
        }
    }
    Ok(())
}

// -- Combined dictionary (bag union of every contributing layer's dict, or direct) --

struct CombinedDictBuilt {
    proof: CombinedDictProof,
    hash_at_k: Vec<ArkFr>,
    id_at_k: Vec<ArkFr>,
    com_hash_at_k: ArkGT,
    com_id_at_k: ArkGT,
    n: usize,
}

fn prove_combined_dict(
    context: &StoreProverContext<'_>,
    branches: &[RelationWitness],
    working_log: usize,
    transcript: &mut Transcript254,
) -> io::Result<CombinedDictBuilt> {
    if branches.is_empty() {
        return Err(invalid("reconcile: combined dict needs at least one contributing layer"));
    }
    let (union, hash_data, id_data, com_hash, com_id, log, n) = if branches.len() >= 2 {
        let res = prove_bag_union_with_context(context, branches, transcript)?;
        (
            Some(res.proof),
            res.columns[0].clone(),
            res.columns[1].clone(),
            res.commitments[0],
            res.commitments[1],
            res.log_domain,
            res.n,
        )
    } else {
        let b = &branches[0];
        (None, b.columns[0].clone(), b.columns[1].clone(), b.commitments[0], b.commitments[1], b.log_domain, b.n)
    };
    let hash_at_k = pad_to(&hash_data, working_log);
    let id_at_k = pad_to(&id_data, working_log);
    let com_hash_at_k = context.commit_column(&hash_at_k)?;
    let com_id_at_k = context.commit_column(&id_at_k)?;
    let reloc_hash = prove_column_relocation_with_context(
        context, &hash_data, &com_hash, &hash_at_k, &com_hash_at_k, transcript,
    )?;
    let reloc_id = prove_column_relocation_with_context(
        context, &id_data, &com_id, &id_at_k, &com_id_at_k, transcript,
    )?;
    Ok(CombinedDictBuilt {
        proof: CombinedDictProof {
            union,
            com_hash,
            com_id,
            log,
            n,
            reloc_hash,
            reloc_id,
            com_hash_at_k,
            com_id_at_k,
        },
        hash_at_k,
        id_at_k,
        com_hash_at_k,
        com_id_at_k,
        n,
    })
}

fn verify_combined_dict(
    branches: &[RelationCommit],
    expected_n: usize,
    working_log: usize,
    proof: &CombinedDictProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    if branches.is_empty() {
        return Err(invalid("reconcile: combined dict needs at least one contributing layer"));
    }
    if proof.n != expected_n {
        return Err(invalid("reconcile: combined dict length mismatch"));
    }
    if branches.len() >= 2 {
        let union = proof
            .union
            .as_ref()
            .ok_or_else(|| invalid("reconcile: missing bag-union proof for a multi-layer dict"))?;
        verify_bag_union(
            branches,
            &[proof.com_hash, proof.com_id],
            expected_n,
            proof.log,
            union,
            transcript,
        )?;
    } else {
        if proof.union.is_some() {
            return Err(invalid("reconcile: unexpected bag-union proof for a single-layer dict"));
        }
        let b = &branches[0];
        if proof.com_hash != b.commitments[0]
            || proof.com_id != b.commitments[1]
            || proof.log != b.log_domain
            || proof.n != b.n
        {
            return Err(invalid("reconcile: single-layer combined dict is not the layer's root-bound dict"));
        }
    }
    verify_column_relocation(
        &proof.com_hash, proof.log, &proof.com_hash_at_k, working_log, &proof.reloc_hash, transcript,
    )?;
    verify_column_relocation(
        &proof.com_id, proof.log, &proof.com_id_at_k, working_log, &proof.reloc_id, transcript,
    )?;
    Ok(())
}

// -- Group description shared by prover and verifier (canonical order) --

/// One committed group: its per-column commitments (subject, predicate, object), its
/// column domain log, real length, and its link sign (`+1` additions, `−1` removals).
struct GroupDesc {
    s_com: ArkGT,
    p_com: ArkGT,
    o_com: ArkGT,
    col_log: usize,
    real_len: usize,
    sign: ArkFr,
}

/// Build the committed group descriptors in canonical order: for each layer, its
/// additions group (if non-empty) then its removals group (if non-empty).
fn group_descs(
    column_commitments: &[&[ArkGT; 6]],
    addition_counts: &[usize],
    removal_counts: &[usize],
) -> io::Result<Vec<GroupDesc>> {
    let mut descs = Vec::new();
    for i in 0..column_commitments.len() {
        let cc = column_commitments[i];
        if addition_counts[i] > 0 {
            descs.push(GroupDesc {
                s_com: cc[0],
                p_com: cc[1],
                o_com: cc[2],
                col_log: group_col_log(addition_counts[i])?,
                real_len: addition_counts[i],
                sign: f1(),
            });
        }
        if removal_counts[i] > 0 {
            descs.push(GroupDesc {
                s_com: cc[3],
                p_com: cc[4],
                o_com: cc[5],
                col_log: group_col_log(removal_counts[i])?,
                real_len: removal_counts[i],
                sign: f0() - f1(),
            });
        }
    }
    Ok(descs)
}

/// Prove the N-layer reconciliation and the reconciled triple count.
///
/// `layers` is ordered base..head (`len >= 2`). `trusted_head_root` MUST be the
/// externally-trusted head anchor; it is required to equal the head `commitment_root`.
pub fn prove_reconcile(layers: &[&LayerCommitment]) -> io::Result<(ReconcileProof, ReconciledPresent)> {
    prove_reconcile_with_context(&StoreProverContext::cpu(), layers)
}

pub fn prove_reconcile_with_context(
    context: &StoreProverContext<'_>,
    layers: &[&LayerCommitment],
) -> io::Result<(ReconcileProof, ReconciledPresent)> {
    let mut transcript = Transcript254::new(b"parture-delta-reconcile-envelope");
    let (proof, artifacts) = prove_reconcile_core(context, layers, &mut transcript)?;
    Ok((proof, ReconciledPresent { rows: artifacts.present_rows }))
}

/// Prover-side authenticated `present` relation (hash columns at 2^K) plus bookkeeping.
/// Returned by [`prove_reconcile_core`] so a read query can be proved over `present` on the
/// SAME transcript — that shared transcript is what binds the BGP to the
/// reconciliation-authenticated relation.
pub struct ReconcileArtifacts {
    pub present_s: Vec<ArkFr>,
    pub present_p: Vec<ArkFr>,
    pub present_o: Vec<ArkFr>,
    pub com_present_s: ArkGT,
    pub com_present_p: ArkGT,
    pub com_present_o: ArkGT,
    /// present's subject/object COMBINED-DICT ids at 2^K (the small canonical id space,
    /// authenticated in the join path against the reconciliation hashes). Predicate ids are
    /// not needed for joins (predicate filtering stays in hash space).
    pub present_id_s: Vec<ArkFr>,
    pub present_id_o: Vec<ArkFr>,
    pub working_log: usize,
    pub present_len: usize,
    /// The combined node/value dict at 2^K (for authenticating present's ids in the join).
    nv_dict: CombinedDictBuilt,
    present_rows: Vec<(ArkFr, ArkFr, ArkFr)>,
}

fn prove_reconcile_core(
    context: &StoreProverContext<'_>,
    layers: &[&LayerCommitment],
    transcript: &mut Transcript254,
) -> io::Result<(ReconcileProof, ReconcileArtifacts)> {
    let states: Vec<&super::commitment::CommitmentState> = layers.iter().map(|l| &l.state).collect();
    validate_chain_states(&states)?;
    let head = layers.last().unwrap();
    let head_nv = head.state.node_value_count as usize;
    let head_pred = head.state.predicate_count as usize;

    // Reconcile present (ground truth for the prover).
    let present = reconcile_present(layers)?;
    let present_len = present.len();
    let present_log = present_len.next_power_of_two().max(2).trailing_zeros() as usize;

    // Working domain 2^K (verifier-recomputable from public counts alone).
    let mut working_log = dict_log(head_nv)?.max(dict_log(head_pred)?).max(present_log);
    for layer in layers {
        if layer.additions.len() >= 2 {
            working_log = working_log.max(group_col_log(layer.additions.len())?);
        }
        if layer.removals.len() >= 2 {
            working_log = working_log.max(group_col_log(layer.removals.len())?);
        }
    }
    let k_size = 1usize << working_log;

    transcript.append_bytes(b"pow/reconcile/domain", RECONCILE_DOMAIN);
    for layer in layers {
        transcript.append_bytes(b"pow/reconcile/layer-root", &layer.state.commitment_root);
    }

    // Combined node/value + predicate dictionaries (bag union across the whole chain).
    let mut nv_branches = Vec::new();
    let mut pred_branches = Vec::new();
    for layer in layers {
        if !layer.node_value_catalog.is_empty() {
            let (h, id) = layer.node_value_dictionary_columns()?;
            let n = layer.node_value_catalog.len();
            nv_branches.push(RelationWitness {
                columns: vec![h, id],
                commitments: vec![layer.node_value_dict_commitments.0, layer.node_value_dict_commitments.1],
                log_domain: dict_log(n)?,
                n,
            });
        }
        if !layer.predicate_catalog.is_empty() {
            let (h, id) = layer.predicate_dictionary_columns()?;
            let n = layer.predicate_catalog.len();
            pred_branches.push(RelationWitness {
                columns: vec![h, id],
                commitments: vec![layer.predicate_dict_commitments.0, layer.predicate_dict_commitments.1],
                log_domain: dict_log(n)?,
                n,
            });
        }
    }
    let combined_nv = prove_combined_dict(context, &nv_branches, working_log, transcript)?;
    let combined_pred = prove_combined_dict(context, &pred_branches, working_log, transcript)?;

    // Committed group descriptors (canonical order).
    let column_commitments: Vec<&[ArkGT; 6]> = layers.iter().map(|l| &l.column_commitments).collect();
    let addition_counts: Vec<usize> = layers.iter().map(|l| l.additions.len()).collect();
    let removal_counts: Vec<usize> = layers.iter().map(|l| l.removals.len()).collect();
    let descs = group_descs(&column_commitments, &addition_counts, &removal_counts)?;

    // The matching records per group, in the same canonical order.
    let mut record_groups: Vec<(&[ProofRecord], bool)> = Vec::new(); // (records, is_addition)
    for layer in layers {
        if !layer.additions.is_empty() {
            record_groups.push((layer.additions.as_slice(), true));
        }
        if !layer.removals.is_empty() {
            record_groups.push((layer.removals.as_slice(), false));
        }
    }

    // Authenticate each group's canonical S/P/O hash columns.
    let mut groups = Vec::with_capacity(descs.len());
    // group value column data at 2^K, in link column order [S, P, O] per group.
    let mut group_spo_data: Vec<[Vec<ArkFr>; 3]> = Vec::with_capacity(descs.len());
    let mut group_spo_com: Vec<[ArkGT; 3]> = Vec::with_capacity(descs.len());
    for (desc, &(records, _)) in descs.iter().zip(record_groups.iter()) {
        let s = prove_group_hash(
            context, records, |r| r.subject_id, &desc.s_com, desc.col_log, desc.real_len,
            working_log, &combined_nv, transcript,
        )?;
        let o = prove_group_hash(
            context, records, |r| r.object_id, &desc.o_com, desc.col_log, desc.real_len,
            working_log, &combined_nv, transcript,
        )?;
        let p = prove_group_hash(
            context, records, |r| r.predicate_id, &desc.p_com, desc.col_log, desc.real_len,
            working_log, &combined_pred, transcript,
        )?;
        group_spo_data.push([s.1, p.1, o.1]);
        group_spo_com.push([s.0.lookup.com_v_hash, p.0.lookup.com_v_hash, o.0.lookup.com_v_hash]);
        groups.push(CommittedGroupProof { s: s.0, o: o.0, p: p.0 });
    }

    // present's canonical S/P/O columns at 2^K.
    let mut present_s = vec![f0(); k_size];
    let mut present_p = vec![f0(); k_size];
    let mut present_o = vec![f0(); k_size];
    let mut present_id_s = vec![f0(); k_size];
    let mut present_id_o = vec![f0(); k_size];
    let mut present_rows = Vec::with_capacity(present_len);
    for (i, record) in present.iter().enumerate() {
        let sh = subject_hash(record);
        let ph = predicate_hash(record);
        let oh = object_hash(record);
        present_s[i] = sh;
        present_p[i] = ph;
        present_o[i] = oh;
        present_id_s[i] = ark(record.subject_id);
        present_id_o[i] = ark(record.object_id);
        present_rows.push((sh, ph, oh));
    }
    let com_present_s = context.commit_column(&present_s)?;
    let com_present_p = context.commit_column(&present_p)?;
    let com_present_o = context.commit_column(&present_o)?;

    // The full link group list = committed groups then present (sign −1).
    let g = descs.len() + 1;
    let mut lens: Vec<usize> = descs.iter().map(|d| d.real_len).collect();
    lens.push(present_len);
    let mut signs: Vec<ArkFr> = descs.iter().map(|d| d.sign).collect();
    signs.push(f0() - f1());

    let mut value_data: Vec<[&Vec<ArkFr>; 3]> = Vec::with_capacity(g);
    let mut value_com: Vec<[ArkGT; 3]> = Vec::with_capacity(g);
    for i in 0..descs.len() {
        value_data.push([&group_spo_data[i][0], &group_spo_data[i][1], &group_spo_data[i][2]]);
        value_com.push(group_spo_com[i]);
    }
    value_data.push([&present_s, &present_p, &present_o]);
    value_com.push([com_present_s, com_present_p, com_present_o]);

    for coms in &value_com {
        for com in coms {
            transcript.append_group(b"pow/reconcile/link-val", com);
        }
    }
    let beta = transcript.challenge_scalar(b"pow/reconcile/link-beta");
    let alpha = transcript.challenge_scalar(b"pow/reconcile/link-alpha");

    let mut rec_data: Vec<Vec<ArkFr>> = Vec::with_capacity(g);
    let mut com_rec: Vec<ArkGT> = Vec::with_capacity(g);
    for values in &value_data {
        let fold = rlc_fold(beta, &[values[0].as_slice(), values[1].as_slice(), values[2].as_slice()]);
        let rec = reciprocal_witness(alpha, &fold);
        com_rec.push(context.commit_column(&rec)?);
        rec_data.push(rec);
    }
    for com in &com_rec {
        transcript.append_group(b"pow/reconcile/link-rec", com);
    }

    // Assemble the 4G link columns [S, P, O, rec] per group.
    let mut columns: Vec<Vec<ArkFr>> = Vec::with_capacity(4 * g);
    for i in 0..g {
        columns.push(value_data[i][0].clone());
        columns.push(value_data[i][1].clone());
        columns.push(value_data[i][2].clone());
        columns.push(rec_data[i].clone());
    }
    let shape = link_shape(working_log, &lens, &signs, alpha, beta);
    let (link_sumcheck, r, _evals) = prove_sumcheck(&shape, &columns, transcript)?;
    let link_opening = context.prove_column_openings(&columns, &r, transcript)?;

    Ok((
        ReconcileProof {
            combined_nv: combined_nv.proof.clone(),
            combined_pred: combined_pred.proof,
            working_log,
            groups,
            com_present_s,
            com_present_p,
            com_present_o,
            com_rec,
            link_sumcheck,
            link_opening,
            present_len,
        },
        ReconcileArtifacts {
            present_s,
            present_p,
            present_o,
            com_present_s,
            com_present_p,
            com_present_o,
            present_id_s,
            present_id_o,
            working_log,
            present_len,
            nv_dict: combined_nv,
            present_rows,
        },
    ))
}

/// Authenticate one id column (`select`) of `records` against a combined dict, at 2^K.
/// Returns the group hash proof and the authenticated hash column data.
#[allow(clippy::too_many_arguments)]
fn prove_group_hash(
    context: &StoreProverContext<'_>,
    records: &[ProofRecord],
    select: impl Fn(&ProofRecord) -> u64,
    com_source: &ArkGT,
    source_log: usize,
    real_n: usize,
    working_log: usize,
    dict: &CombinedDictBuilt,
    transcript: &mut Transcript254,
) -> io::Result<(GroupHashProof, Vec<ArkFr>)> {
    let mut obj_source = vec![f0(); 1usize << source_log];
    for (slot, record) in obj_source.iter_mut().zip(records.iter()) {
        *slot = ark(select(record));
    }
    let obj_at_k = pad_to(&obj_source, working_log);
    let com_obj_at_k = context.commit_column(&obj_at_k)?;
    let obj_reloc = prove_column_relocation_with_context(
        context, &obj_source, com_source, &obj_at_k, &com_obj_at_k, transcript,
    )?;
    let lookup = prove_hash_lookup_inner(
        context, &obj_at_k, &com_obj_at_k, &dict.id_at_k, &dict.hash_at_k,
        &dict.com_id_at_k, &dict.com_hash_at_k, real_n, dict.n, transcript,
    )?;
    let v_hash = honest_hash_lookup(&obj_at_k, &dict.id_at_k, &dict.hash_at_k, real_n, dict.n).0;
    Ok((GroupHashProof { obj_reloc, com_obj_at_k, lookup }, v_hash))
}

/// The G-way link sumcheck shape: one reciprocal (Identity) claim per group pinning
/// `rec = 1/(alpha − fold[S,P,O])`, plus the signed masked multiset-equality link.
fn link_shape(k: usize, lens: &[usize], signs: &[ArkFr], alpha: ArkFr, beta: ArkFr) -> SumcheckShape {
    let g = lens.len();
    let mut claims = Vec::with_capacity(g + 1);
    for i in 0..g {
        claims.push(reciprocal_claim(i * 4 + 3, &[i * 4, i * 4 + 1, i * 4 + 2], alpha, beta));
    }
    let terms = (0..g)
        .map(|i| Term {
            coeff: signs[i],
            operands: vec![Operand::Chi(i), Operand::Column(i * 4 + 3)],
        })
        .collect();
    claims.push(SumcheckClaim { shape: ClaimShape::ZeroSum, expr: Expr { terms }, target: f0() });
    SumcheckShape {
        num_vars: k,
        num_columns: 4 * g,
        chi_bounds: lens.to_vec(),
        claims,
    }
}

/// Verify an N-layer reconciliation. On success returns the authenticated reconciled
/// triple COUNT `M` (the single-atom count query answer over `present`).
pub fn verify_reconcile(
    layers: &[&LayerCommitment],
    trusted_head_root: ProofHash,
    proof: &ReconcileProof,
) -> io::Result<usize> {
    let mut transcript = Transcript254::new(b"parture-delta-reconcile-envelope");
    let verified = verify_reconcile_core(layers, trusted_head_root, proof, &mut transcript)?;
    Ok(verified.present_len)
}

/// Verifier-side authenticated `present` handle (its column commitments at 2^K + counts),
/// returned by [`verify_reconcile_core`] so a read query can be verified over it on the SAME
/// transcript.
pub struct ReconcileVerified {
    pub com_present_s: ArkGT,
    pub com_present_p: ArkGT,
    pub com_present_o: ArkGT,
    pub working_log: usize,
    pub present_len: usize,
}

fn verify_reconcile_core(
    layers: &[&LayerCommitment],
    trusted_head_root: ProofHash,
    proof: &ReconcileProof,
    transcript: &mut Transcript254,
) -> io::Result<ReconcileVerified> {
    let views: Vec<VerifierCommitmentRef<'_>> = layers.iter().map(|l| l.verifier_view()).collect();
    // (1) Each layer authenticated against ITS OWN commitment root.
    for view in &views {
        view.validate()?;
    }
    // (2) Whole parent chain + trusted head anchor.
    let states: Vec<&super::commitment::CommitmentState> = views.iter().map(|v| v.state).collect();
    validate_chain_states(&states)?;
    if states.last().unwrap().commitment_root != trusted_head_root {
        return Err(invalid("reconcile: head root does not match the trusted anchor"));
    }
    let head = states.last().unwrap();
    let head_nv = head.node_value_count as usize;
    let head_pred = head.predicate_count as usize;

    let addition_counts: Vec<usize> = states.iter().map(|s| s.addition_count as usize).collect();
    let removal_counts: Vec<usize> = states.iter().map(|s| s.removal_count as usize).collect();
    let total_add: usize = addition_counts.iter().sum();
    let total_rem: usize = removal_counts.iter().sum();
    let present_len = total_add
        .checked_sub(total_rem)
        .ok_or_else(|| invalid("reconcile: total removals exceed total additions"))?;
    if present_len != proof.present_len {
        return Err(invalid("reconcile: claimed present length disagrees with the delta counts"));
    }
    let present_log = present_len.next_power_of_two().max(2).trailing_zeros() as usize;

    let mut working_log = dict_log(head_nv)?.max(dict_log(head_pred)?).max(present_log);
    for &n in addition_counts.iter().chain(removal_counts.iter()) {
        if n >= 2 {
            working_log = working_log.max(group_col_log(n)?);
        }
    }
    if working_log != proof.working_log {
        return Err(invalid("reconcile: working domain mismatch"));
    }

    transcript.append_bytes(b"pow/reconcile/domain", RECONCILE_DOMAIN);
    for view in &views {
        transcript.append_bytes(b"pow/reconcile/layer-root", &view.state.commitment_root);
    }

    // Combined dict branches (verifier side) — one per layer with a non-empty incremental dict.
    let mut nv_branches = Vec::new();
    let mut pred_branches = Vec::new();
    for view in &views {
        let nv_incr = (view.state.node_value_count - view.state.parent_node_value_count) as usize;
        if nv_incr > 0 {
            nv_branches.push(RelationCommit {
                commitments: vec![view.node_value_dict_commitments.0, view.node_value_dict_commitments.1],
                log_domain: dict_log(nv_incr)?,
                n: nv_incr,
            });
        }
        let pred_incr = (view.state.predicate_count - view.state.parent_predicate_count) as usize;
        if pred_incr > 0 {
            pred_branches.push(RelationCommit {
                commitments: vec![view.predicate_dict_commitments.0, view.predicate_dict_commitments.1],
                log_domain: dict_log(pred_incr)?,
                n: pred_incr,
            });
        }
    }
    verify_combined_dict(&nv_branches, head_nv, working_log, &proof.combined_nv, transcript)?;
    verify_combined_dict(&pred_branches, head_pred, working_log, &proof.combined_pred, transcript)?;

    // Committed group descriptors (same canonical order the prover used).
    let column_commitments: Vec<&[ArkGT; 6]> = views.iter().map(|v| v.column_commitments).collect();
    let descs = group_descs(&column_commitments, &addition_counts, &removal_counts)?;
    if descs.len() != proof.groups.len() {
        return Err(invalid("reconcile: committed group count mismatch"));
    }

    // Verify each group's three canonical hash lookups.
    for (desc, gp) in descs.iter().zip(proof.groups.iter()) {
        verify_group_hash(&desc.s_com, desc.col_log, desc.real_len, working_log, &proof.combined_nv, &gp.s, transcript)?;
        verify_group_hash(&desc.o_com, desc.col_log, desc.real_len, working_log, &proof.combined_nv, &gp.o, transcript)?;
        verify_group_hash(&desc.p_com, desc.col_log, desc.real_len, working_log, &proof.combined_pred, &gp.p, transcript)?;
    }

    // Verify the G-way masked LogUp link.
    let g = descs.len() + 1;
    if proof.com_rec.len() != g {
        return Err(invalid("reconcile: reciprocal commitment count mismatch"));
    }
    let mut lens: Vec<usize> = descs.iter().map(|d| d.real_len).collect();
    lens.push(present_len);
    let mut signs: Vec<ArkFr> = descs.iter().map(|d| d.sign).collect();
    signs.push(f0() - f1());

    let mut value_com: Vec<[ArkGT; 3]> = Vec::with_capacity(g);
    for gp in &proof.groups {
        value_com.push([gp.s.lookup.com_v_hash, gp.p.lookup.com_v_hash, gp.o.lookup.com_v_hash]);
    }
    value_com.push([proof.com_present_s, proof.com_present_p, proof.com_present_o]);

    for coms in &value_com {
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
    let shape = link_shape(working_log, &lens, &signs, alpha, beta);
    let (r, evals) = verify_sumcheck(&shape, &proof.link_sumcheck, transcript)?;
    let mut commitments = Vec::with_capacity(4 * g);
    for i in 0..g {
        commitments.push(value_com[i][0]);
        commitments.push(value_com[i][1]);
        commitments.push(value_com[i][2]);
        commitments.push(proof.com_rec[i]);
    }
    verify_column_openings(&commitments, &r, &evals, &proof.link_opening, transcript)?;

    Ok(ReconcileVerified {
        com_present_s: proof.com_present_s,
        com_present_p: proof.com_present_p,
        com_present_o: proof.com_present_o,
        working_log,
        present_len,
    })
}

fn verify_group_hash(
    com_source: &ArkGT,
    source_log: usize,
    real_n: usize,
    working_log: usize,
    dict: &CombinedDictProof,
    gp: &GroupHashProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    verify_column_relocation(
        com_source, source_log, &gp.com_obj_at_k, working_log, &gp.obj_reloc, transcript,
    )?;
    verify_hash_lookup_inner(
        &gp.com_obj_at_k, &dict.com_id_at_k, &dict.com_hash_at_k, real_n, dict.n, &gp.lookup, transcript,
    )
}

// ===========================================================================
// Wave 3 — single-atom BGP over the authenticated `present` relation.
//
// `present` is authenticated (by the reconciliation above) as three hash columns
// [hash_object(S), hash_term(P), hash_object(O)] committed at 2^K with real length M.
// A single-atom read with ONE constant position is an EXACT filter+compaction over
// those committed columns, proved on the SAME transcript as the reconciliation (so it
// is bound to the authenticated relation). The audited Q1 filter indicator pins the
// selection to EXACTLY the equality-with-constant indicator (complete AND sound: every
// matching row is selected, no non-matching row is), and the audited compaction emits
// EXACTLY the selected rows of the two output columns. The delivered result columns are
// authenticated hashes of the output bindings; resolving them to canonical IRIs/values
// is deferred (see the module report).
// ===========================================================================

/// Which `present` position carries the query CONSTANT (the other two are output vars).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterPosition {
    Subject,
    Predicate,
    Object,
}

/// A single-atom filter proof over `present`: the equality-indicator selection and the
/// compaction of the two output columns to the matching rows.
#[derive(Clone, Debug)]
pub struct PresentFilterProof {
    pub filter: FilterProof,
    pub com_sel: ArkGT,
    pub com_w: ArkGT,
    pub compaction: CompactionOnlyProof,
    /// The two authenticated output hash columns (the non-constant variables), matching rows.
    pub com_out: Vec<ArkGT>,
    pub out_len: usize,
}

/// The delivered read result: the output hash columns (real prefix) and the row count.
pub struct PresentReadResult {
    /// Two columns of authenticated binding HASHES (output variables), length `len`.
    pub columns: Vec<Vec<ArkFr>>,
    pub len: usize,
}

/// Map a filter position to `(filter column, filter commitment, [output columns], [output commitments])`.
fn filter_selection<'a>(
    artifacts: &'a ReconcileArtifacts,
    position: FilterPosition,
) -> (&'a [ArkFr], ArkGT, [&'a [ArkFr]; 2], [ArkGT; 2]) {
    match position {
        FilterPosition::Predicate => (
            artifacts.present_p.as_slice(),
            artifacts.com_present_p,
            [artifacts.present_s.as_slice(), artifacts.present_o.as_slice()],
            [artifacts.com_present_s, artifacts.com_present_o],
        ),
        FilterPosition::Subject => (
            artifacts.present_s.as_slice(),
            artifacts.com_present_s,
            [artifacts.present_p.as_slice(), artifacts.present_o.as_slice()],
            [artifacts.com_present_p, artifacts.com_present_o],
        ),
        FilterPosition::Object => (
            artifacts.present_o.as_slice(),
            artifacts.com_present_o,
            [artifacts.present_s.as_slice(), artifacts.present_p.as_slice()],
            [artifacts.com_present_s, artifacts.com_present_p],
        ),
    }
}

/// The verifier-side output commitments for a filter position (from the reconciled handle).
fn filter_output_commitments(verified: &ReconcileVerified, position: FilterPosition) -> (ArkGT, [ArkGT; 2]) {
    match position {
        FilterPosition::Predicate => (verified.com_present_p, [verified.com_present_s, verified.com_present_o]),
        FilterPosition::Subject => (verified.com_present_s, [verified.com_present_p, verified.com_present_o]),
        FilterPosition::Object => (verified.com_present_o, [verified.com_present_s, verified.com_present_p]),
    }
}

/// Prove an exact equality-with-constant filter on `filter_col` (committed as `com_filter`,
/// real length `real_n`) and compact `sources` to the matching rows. Reused by the
/// single-atom read (hash output columns) AND each join atom (id output columns).
#[allow(clippy::too_many_arguments)]
fn prove_filter_part(
    context: &StoreProverContext<'_>,
    filter_col: &[ArkFr],
    com_filter: &ArkGT,
    constant: ArkFr,
    real_n: usize,
    sources: &[Vec<ArkFr>],
    source_coms: &[ArkGT],
    transcript: &mut Transcript254,
) -> io::Result<(PresentFilterProof, Vec<Vec<ArkFr>>)> {
    let size = filter_col.len();
    let sel: Vec<ArkFr> = (0..size)
        .map(|i| if i < real_n && filter_col[i] == constant { f1() } else { f0() })
        .collect();
    let (filter, com_sel, com_w, _sel_eval) = prove_filter_indicator_with_context(
        context, filter_col, constant, real_n, com_filter, transcript,
    )?;
    let (compaction, com_out, out_cols, out_len) = prove_compaction_only_with_context(
        context, sources, source_coms, &sel, &com_sel, transcript,
    )?;
    Ok((PresentFilterProof { filter, com_sel, com_w, compaction, com_out, out_len }, out_cols))
}

fn verify_filter_part(
    com_filter: &ArkGT,
    constant: ArkFr,
    real_n: usize,
    source_coms: &[ArkGT],
    proof: &PresentFilterProof,
    transcript: &mut Transcript254,
) -> io::Result<usize> {
    verify_filter_indicator(
        com_filter, constant, real_n, &proof.com_sel, &proof.com_w, &proof.filter, transcript,
    )?;
    verify_compaction_only(
        source_coms, &proof.com_sel, &proof.com_out, proof.out_len, &proof.compaction, transcript,
    )?;
    Ok(proof.out_len)
}

fn prove_present_filter(
    context: &StoreProverContext<'_>,
    artifacts: &ReconcileArtifacts,
    position: FilterPosition,
    constant_hash: ArkFr,
    transcript: &mut Transcript254,
) -> io::Result<(PresentFilterProof, PresentReadResult)> {
    let (filter_col, com_filter, sources, source_coms) = filter_selection(artifacts, position);
    let source_vecs: Vec<Vec<ArkFr>> = sources.iter().map(|c| c.to_vec()).collect();
    let (part, out_cols) = prove_filter_part(
        context, filter_col, &com_filter, constant_hash, artifacts.present_len, &source_vecs, &source_coms, transcript,
    )?;
    let out_len = part.out_len;
    let columns: Vec<Vec<ArkFr>> = out_cols.iter().map(|c| c[..out_len].to_vec()).collect();
    Ok((part, PresentReadResult { columns, len: out_len }))
}

fn verify_present_filter(
    verified: &ReconcileVerified,
    position: FilterPosition,
    constant_hash: ArkFr,
    proof: &PresentFilterProof,
    transcript: &mut Transcript254,
) -> io::Result<usize> {
    let (com_filter, source_coms) = filter_output_commitments(verified, position);
    verify_filter_part(&com_filter, constant_hash, verified.present_len, &source_coms, proof, transcript)
}

/// Prove a single-atom read `Triple(...)` with ONE constant position over the reconciled
/// N-layer `present`: reconcile + authenticated filter, one transcript. The delivered
/// result columns are the two output variables as authenticated binding HASHES.
pub fn prove_read_single_atom(
    layers: &[&LayerCommitment],
    position: FilterPosition,
    constant_hash: ArkFr,
) -> io::Result<(ReconcileProof, PresentFilterProof, PresentReadResult)> {
    let context = StoreProverContext::cpu();
    let mut transcript = Transcript254::new(b"parture-delta-reconcile-envelope");
    let (reconcile, artifacts) = prove_reconcile_core(&context, layers, &mut transcript)?;
    let (filter, result) = prove_present_filter(&context, &artifacts, position, constant_hash, &mut transcript)?;
    Ok((reconcile, filter, result))
}

/// Verify a single-atom read. On success returns the authenticated result row count.
pub fn verify_read_single_atom(
    layers: &[&LayerCommitment],
    trusted_head_root: ProofHash,
    position: FilterPosition,
    constant_hash: ArkFr,
    reconcile: &ReconcileProof,
    filter: &PresentFilterProof,
) -> io::Result<usize> {
    let mut transcript = Transcript254::new(b"parture-delta-reconcile-envelope");
    let verified = verify_reconcile_core(layers, trusted_head_root, reconcile, &mut transcript)?;
    verify_present_filter(&verified, position, constant_hash, filter, &mut transcript)
}

// ===========================================================================
// Wave 3 — two-atom JOIN over present:  Triple(?x, p, ?y) & Triple(?y, q, ?z).
//
// The join key ?y is a term shared between atom-1's OBJECT and atom-2's SUBJECT. In hash
// space the key is a ~254-bit field element (no magnitude-sort join). We therefore bridge
// to the COMBINED-DICT id space — a small, dense, chain-wide canonical id per term — and
// reuse the audited magnitude-sort inner join:
//   * present's subject/object COMBINED ids are authenticated by an `id → hash` lookup
//     against the combined node/value dict whose result column is required to EQUAL the
//     reconciliation-authenticated hash column (`com_present_s`/`_o`). This pins each id to
//     the correct term (unique id ↔ hash; `hash_object` collision-resistant), so the ids
//     faithfully represent `present` without leaving the authenticated relation.
//   * each atom is the audited equality filter on the hash predicate column, compacting the
//     authenticated id columns to `R1 = [id_x, id_y]` (predicate p) and `R2 = [id_y, id_z]`
//     (predicate q).
//   * the audited complete-and-sound inner join joins `R1.col1 == R2.col0` on the small ids
//     (no 254-bit cap), delivering exactly the `[id_x, id_y, id_z]` solutions.
// Result ids resolve to canonical IRIs/values against the chain catalogs — deferred (report).
// ===========================================================================

/// `present`'s id-column authentication: the committed id column and the lookup pinning it to
/// the reconciliation-authenticated hash column.
#[derive(Clone, Debug)]
pub struct PresentIdAuth {
    pub com_id: ArkGT,
    pub lookup: HashLookupInnerProof,
}

/// A two-atom join proof over `present`.
#[derive(Clone, Debug)]
pub struct PresentJoinProof {
    pub auth_s: PresentIdAuth,
    pub auth_o: PresentIdAuth,
    pub filter1: PresentFilterProof,
    pub filter2: PresentFilterProof,
    pub inner: InnerJoinProof,
    pub com_join_out: Vec<ArkGT>,
    pub join_out_len: usize,
    pub join_log: usize,
}

/// Authenticate a `present` id column (already at 2^K) against the combined node/value dict:
/// the lookup's `com_v_hash` is later required to equal the reconciliation-authenticated hash.
fn prove_id_auth(
    context: &StoreProverContext<'_>,
    id_col: &[ArkFr],
    real_n: usize,
    dict: &CombinedDictBuilt,
    transcript: &mut Transcript254,
) -> io::Result<PresentIdAuth> {
    let com_id = context.commit_column(id_col)?;
    let lookup = prove_hash_lookup_inner(
        context, id_col, &com_id, &dict.id_at_k, &dict.hash_at_k, &dict.com_id_at_k, &dict.com_hash_at_k,
        real_n, dict.n, transcript,
    )?;
    Ok(PresentIdAuth { com_id, lookup })
}

fn verify_id_auth(
    reconcile_nv: &CombinedDictProof,
    real_n: usize,
    n_dict: usize,
    expected_hash_com: ArkGT,
    auth: &PresentIdAuth,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    verify_hash_lookup_inner(
        &auth.com_id, &reconcile_nv.com_id_at_k, &reconcile_nv.com_hash_at_k, real_n, n_dict, &auth.lookup, transcript,
    )?;
    if auth.lookup.com_v_hash != expected_hash_com {
        return Err(invalid("join: present id column is not authenticated to the reconciled hashes"));
    }
    Ok(())
}

fn prove_present_join(
    context: &StoreProverContext<'_>,
    artifacts: &ReconcileArtifacts,
    const_p: ArkFr,
    const_q: ArkFr,
    transcript: &mut Transcript254,
) -> io::Result<(PresentJoinProof, PresentReadResult)> {
    let m = artifacts.present_len;
    // (1) Authenticate present's subject/object combined-dict ids.
    let auth_s = prove_id_auth(context, &artifacts.present_id_s, m, &artifacts.nv_dict, transcript)?;
    let auth_o = prove_id_auth(context, &artifacts.present_id_o, m, &artifacts.nv_dict, transcript)?;

    // (2) Two predicate atoms → id relations. atom1 Triple(?x,p,?y): x=subject, y=object.
    //     atom2 Triple(?y,q,?z): y=subject, z=object. Both compact [id_s, id_o].
    let id_sources = vec![artifacts.present_id_s.clone(), artifacts.present_id_o.clone()];
    let id_source_coms = vec![auth_s.com_id, auth_o.com_id];
    let (filter1, r1_cols) = prove_filter_part(
        context, &artifacts.present_p, &artifacts.com_present_p, const_p, m, &id_sources, &id_source_coms, transcript,
    )?;
    let (filter2, r2_cols) = prove_filter_part(
        context, &artifacts.present_p, &artifacts.com_present_p, const_q, m, &id_sources, &id_source_coms, transcript,
    )?;

    // (3) Inner join on the shared ?y: R1.col1 (id_y = object of atom1) == R2.col0 (id_y = subject of atom2).
    let r1 = RelationWitness {
        columns: r1_cols,
        commitments: filter1.com_out.clone(),
        log_domain: artifacts.working_log,
        n: filter1.out_len,
    };
    let r2 = RelationWitness {
        columns: r2_cols,
        commitments: filter2.com_out.clone(),
        log_domain: artifacts.working_log,
        n: filter2.out_len,
    };
    let join = prove_inner_join_with_keys_and_context(context, &r1, &r2, &[(1, 0)], transcript)?;
    let out_cols: Vec<Vec<ArkFr>> = join.columns.iter().map(|c| c[..join.n].to_vec()).collect();

    Ok((
        PresentJoinProof {
            auth_s,
            auth_o,
            filter1,
            filter2,
            inner: join.proof,
            com_join_out: join.commitments,
            join_out_len: join.n,
            join_log: join.log_domain,
        },
        PresentReadResult { columns: out_cols, len: join.n },
    ))
}

#[allow(clippy::too_many_arguments)]
fn verify_present_join(
    reconcile: &ReconcileProof,
    verified: &ReconcileVerified,
    head_nv: usize,
    const_p: ArkFr,
    const_q: ArkFr,
    proof: &PresentJoinProof,
    transcript: &mut Transcript254,
) -> io::Result<usize> {
    let m = verified.present_len;
    // (1) present id authentication against the reconciled hashes.
    verify_id_auth(&reconcile.combined_nv, m, head_nv, verified.com_present_s, &proof.auth_s, transcript)?;
    verify_id_auth(&reconcile.combined_nv, m, head_nv, verified.com_present_o, &proof.auth_o, transcript)?;

    // (2) The two predicate atoms.
    let id_source_coms = [proof.auth_s.com_id, proof.auth_o.com_id];
    let m1 = verify_filter_part(&verified.com_present_p, const_p, m, &id_source_coms, &proof.filter1, transcript)?;
    let m2 = verify_filter_part(&verified.com_present_p, const_q, m, &id_source_coms, &proof.filter2, transcript)?;

    // (3) The inner join.
    let r1 = RelationCommit { commitments: proof.filter1.com_out.clone(), log_domain: verified.working_log, n: m1 };
    let r2 = RelationCommit { commitments: proof.filter2.com_out.clone(), log_domain: verified.working_log, n: m2 };
    verify_inner_join_with_keys(
        &r1, &r2, &[(1, 0)], &proof.com_join_out, proof.join_out_len, proof.join_log, &proof.inner, transcript,
    )?;
    Ok(proof.join_out_len)
}

/// Prove a two-atom join read `Triple(?x, p, ?y) & Triple(?y, q, ?z)` over the reconciled
/// N-layer `present`. The delivered result is the `[id_x, id_y, id_z]` solutions as combined-dict
/// ids (resolution to canonical IRIs/values deferred).
pub fn prove_read_two_atom_join(
    layers: &[&LayerCommitment],
    const_p: ArkFr,
    const_q: ArkFr,
) -> io::Result<(ReconcileProof, PresentJoinProof, PresentReadResult)> {
    let context = StoreProverContext::cpu();
    let mut transcript = Transcript254::new(b"parture-delta-reconcile-envelope");
    let (reconcile, artifacts) = prove_reconcile_core(&context, layers, &mut transcript)?;
    let (join, result) = prove_present_join(&context, &artifacts, const_p, const_q, &mut transcript)?;
    Ok((reconcile, join, result))
}

/// Verify a two-atom join read. On success returns the authenticated solution count.
pub fn verify_read_two_atom_join(
    layers: &[&LayerCommitment],
    trusted_head_root: ProofHash,
    const_p: ArkFr,
    const_q: ArkFr,
    reconcile: &ReconcileProof,
    join: &PresentJoinProof,
) -> io::Result<usize> {
    let mut transcript = Transcript254::new(b"parture-delta-reconcile-envelope");
    let verified = verify_reconcile_core(layers, trusted_head_root, reconcile, &mut transcript)?;
    let head_nv = layers.last().unwrap().verifier_view().state.node_value_count as usize;
    verify_present_join(reconcile, &verified, head_nv, const_p, const_q, join, &mut transcript)
}

// ===========================================================================
// Wave 4 — RESULT RESOLUTION: authenticated (hash|id) result bindings → CanonicalObjects.
//
// The BGP-over-present proofs authenticate result COMMITMENTS whose columns are binding
// HASHES (single-atom filter, `hash_object`/`hash_term`) or COMBINED-DICT IDS (join). This
// turns them into readable answers — real IRIs / typed literals — WITHOUT re-running the query:
//   1. The prover's claimed result VALUES (real prefix) are tied to the authenticated result
//      commitment by re-committing (pad to the result domain, `commit_column`, require equality
//      with the proof's committed output). A forged/altered binding fails this bind.
//   2. Each value resolves to a `CanonicalObject` against the CHAIN catalogs — a term's canonical
//      bytes live in the catalog of the layer whose incremental dict range defines it, so the
//      resolver searches every layer's root-bound `node_value_catalog` / `predicate_catalog`
//      (each id/hash is unique across the disjoint ranges). The catalogs are authenticated by the
//      per-layer `validate()` the reconciliation verifier already ran (changed IRI/datatype/lexical
//      bytes re-hash differently and fail the root-bound dictionary commitment).
// Mirrors the base `resolve_result_values` discipline, generalized across the N-layer chain.
// ===========================================================================

/// How to resolve a result-column value to a `CanonicalObject`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolveKind {
    /// value = `hash_object(object)`; resolve via the chain node/value catalogs.
    NodeValueHash,
    /// value = combined-dict `object_id`; resolve via the chain node/value catalogs.
    NodeValueId,
    /// value = `hash_term(predicate)`; resolve via the chain predicate catalogs.
    PredicateHash,
}

fn resolve_one(
    views: &[VerifierCommitmentRef<'_>],
    value: ArkFr,
    kind: ResolveKind,
) -> io::Result<CanonicalObject> {
    match kind {
        ResolveKind::NodeValueHash => {
            for view in views {
                for rec in view.node_value_catalog {
                    if hash_object(&rec.triple.object) == value {
                        return Ok(rec.triple.object.clone());
                    }
                }
            }
            Err(invalid("resolve: result hash is absent from the chain node/value dictionaries"))
        }
        ResolveKind::NodeValueId => {
            for view in views {
                for rec in view.node_value_catalog {
                    if ark(rec.object_id) == value {
                        return Ok(rec.triple.object.clone());
                    }
                }
            }
            Err(invalid("resolve: result id is absent from the chain node/value dictionaries"))
        }
        ResolveKind::PredicateHash => {
            for view in views {
                for rec in view.predicate_catalog {
                    if hash_term(&rec.triple.predicate) == value {
                        return Ok(CanonicalObject::Node(rec.triple.predicate.clone()));
                    }
                }
            }
            Err(invalid("resolve: result hash is absent from the chain predicate dictionaries"))
        }
    }
}

/// Resolve authenticated result columns to `CanonicalObject`s. `claimed_columns` are the
/// prover's real-prefix result values; they are bound to `com_out` (the proof's committed
/// result, at `2^domain_log`) by re-commitment before resolution.
fn resolve_result_columns(
    views: &[VerifierCommitmentRef<'_>],
    claimed_columns: &[Vec<ArkFr>],
    com_out: &[ArkGT],
    result_len: usize,
    domain_log: usize,
    kinds: &[ResolveKind],
) -> io::Result<Vec<Vec<CanonicalObject>>> {
    if claimed_columns.len() != com_out.len() || claimed_columns.len() != kinds.len() {
        return Err(invalid("resolve: result column shape mismatch"));
    }
    // (1) Bind the claimed values to the authenticated result commitments.
    for (col, com) in claimed_columns.iter().zip(com_out.iter()) {
        if col.len() != result_len {
            return Err(invalid("resolve: claimed result column length mismatch"));
        }
        let mut padded = col.clone();
        padded.resize(1usize << domain_log, f0());
        if commit_column(&padded)? != *com {
            return Err(invalid(
                "resolve: claimed result values do not open the authenticated result commitment",
            ));
        }
    }
    // (2) Resolve each value against the root-bound chain catalogs.
    claimed_columns
        .iter()
        .zip(kinds)
        .map(|(col, kind)| col.iter().map(|v| resolve_one(views, *v, *kind)).collect())
        .collect()
}

/// The output-column resolution kinds for a single-atom filter (the two non-constant columns).
fn filter_output_kinds(position: FilterPosition) -> [ResolveKind; 2] {
    match position {
        // output [S, O]
        FilterPosition::Predicate => [ResolveKind::NodeValueHash, ResolveKind::NodeValueHash],
        // output [P, O]
        FilterPosition::Subject => [ResolveKind::PredicateHash, ResolveKind::NodeValueHash],
        // output [S, P]
        FilterPosition::Object => [ResolveKind::NodeValueHash, ResolveKind::PredicateHash],
    }
}

/// Verify a single-atom read AND resolve its result to `CanonicalObject`s (the readable answer,
/// authenticated against the chain — no query re-run). `claimed_columns` are the prover's
/// published result values (from [`PresentReadResult`]).
#[allow(clippy::too_many_arguments)]
pub fn verify_and_resolve_single_atom(
    layers: &[&LayerCommitment],
    trusted_head_root: ProofHash,
    position: FilterPosition,
    constant_hash: ArkFr,
    reconcile: &ReconcileProof,
    filter: &PresentFilterProof,
    claimed_columns: &[Vec<ArkFr>],
) -> io::Result<Vec<Vec<CanonicalObject>>> {
    let mut transcript = Transcript254::new(b"parture-delta-reconcile-envelope");
    let verified = verify_reconcile_core(layers, trusted_head_root, reconcile, &mut transcript)?;
    let out_len = verify_present_filter(&verified, position, constant_hash, filter, &mut transcript)?;
    let views: Vec<VerifierCommitmentRef<'_>> = layers.iter().map(|l| l.verifier_view()).collect();
    let kinds = filter_output_kinds(position);
    resolve_result_columns(&views, claimed_columns, &filter.com_out, out_len, verified.working_log, &kinds)
}

/// Verify a two-atom join read AND resolve its `[?x, ?y, ?z]` bindings to `CanonicalObject`s.
#[allow(clippy::too_many_arguments)]
pub fn verify_and_resolve_two_atom_join(
    layers: &[&LayerCommitment],
    trusted_head_root: ProofHash,
    const_p: ArkFr,
    const_q: ArkFr,
    reconcile: &ReconcileProof,
    join: &PresentJoinProof,
    claimed_columns: &[Vec<ArkFr>],
) -> io::Result<Vec<Vec<CanonicalObject>>> {
    let mut transcript = Transcript254::new(b"parture-delta-reconcile-envelope");
    let verified = verify_reconcile_core(layers, trusted_head_root, reconcile, &mut transcript)?;
    let head_nv = layers.last().unwrap().verifier_view().state.node_value_count as usize;
    let out_len = verify_present_join(reconcile, &verified, head_nv, const_p, const_q, join, &mut transcript)?;
    let views: Vec<VerifierCommitmentRef<'_>> = layers.iter().map(|l| l.verifier_view()).collect();
    // Join output [id_x, id_y, id_z] are all node/value combined-dict ids.
    let kinds = [ResolveKind::NodeValueId, ResolveKind::NodeValueId, ResolveKind::NodeValueId];
    resolve_result_columns(&views, claimed_columns, &join.com_join_out, out_len, join.join_log, &kinds)
}

// ===========================================================================
// Wave 6 — GENERAL n-atom conjunctive BGP over the reconciled `present` relation.
//
// This generalizes the single-atom filter and 2-atom join above to the SAME read-only
// BGP fragment the base-layer path (`super::query::prove_layer_bgp`) supports over a
// materialized layer: an arbitrary left-deep conjunctive chain of k atoms, constants in
// any position, and repeated-variable same-row equality — now over a DELTA STACK, with
// every input column authenticated to the trusted head root by the reconciliation above.
//
// It is built by REUSE, not new crypto. `present` is authenticated (by the reconciliation)
// as three canonical HASH columns `[hash_object(S), hash_term(P), hash_object(O)]` at 2^K,
// plus (via [`prove_id_auth`]) its subject/object COMBINED-DICT id columns pinned to those
// hashes. Each atom is then the SAME audited composition the base RANGED atom uses:
//   1. an authenticated EQUALITY SELECTION over `present`'s root-bound HASH columns
//      ([`super::filter::prove_selection_only`]) — predicate/subject/object CONSTANTS are
//      public canonical hashes compared against the authenticated hash columns, and a
//      repeated variable is a same-row hash equality; this pins `sel` to EXACTLY the
//      matching rows (complete AND sound), and an absent constant simply selects nothing;
//   2. a COMPACTION ([`super::filter::prove_compaction_only`]) of the atom's KEPT variable
//      columns — the reconciliation-authenticated combined-dict id columns (`present_id_s`
//      / `present_id_o`, each pinned row-for-row to the selected term by [`verify_id_auth`])
//      — by that same `sel`, yielding the atom's filtered relation as small joinable ids.
// The k filtered relations feed the audited left-deep join chain ([`super::bgp`]) verbatim,
// so >= 3-atom joins on shared node variables work by reuse. Every result column is a
// combined-dict node/value id, resolved to canonical IRIs/typed literals by the chain
// catalog resolver ([`resolve_result_columns`]) exactly as the 2-atom join is.
//
// SCOPE (this pass): the predicate of each atom is a CONSTANT IRI (its hash); every kept
// output/join column is therefore a node/value id (uniform, joinable, resolvable). Each
// atom keeps >= 1 variable. Predicate VARIABLES and fully-ground atoms reject-closed
// (a later pass authenticates `present`'s predicate ids the same way as its node ids).
// NUMERIC RANGE over `present` is deferred (it needs an authenticated order-preserving
// value/datatype column for `present`, derived from the chain dict range commitments —
// see the module report), so a ranged atom is not representable here.
// ===========================================================================

/// One subject/object position of a `present` BGP atom: a kept query VARIABLE (an output
/// combined-dict id) or a CONSTANT given as its public canonical `hash_object(...)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresentTerm {
    Variable,
    /// The public `hash_object(canonical object)` of the constant (a node IRI in the
    /// subject or object position, or a typed literal in the object position).
    Constant(ArkFr),
}

/// One atom of a `present` BGP: a CONSTANT predicate (its public `hash_term` IRI hash),
/// subject and object each a [`PresentTerm`], and an optional same-row subject==object
/// equality (both positions must then be variables). Mirrors the base
/// [`super::query::LayerBgpAtom`] restricted to this pass's supported shape.
#[derive(Clone, Copy, Debug)]
pub struct PresentBgpAtom {
    pub subject: PresentTerm,
    /// `hash_term(predicate IRI)` — the predicate is a constant this pass.
    pub predicate_hash: ArkFr,
    pub object: PresentTerm,
    pub subject_object_equal: bool,
}

/// The per-atom half of a `present` BGP proof: the authenticated equality selection over
/// `present`'s hash columns, plus the compaction of the atom's kept id columns to the
/// matching rows.
#[derive(Clone, Debug)]
pub struct PresentBgpAtomProof {
    pub selection: SelectionOnlyProof,
    pub com_sel: ArkGT,
    pub compaction: CompactionOnlyProof,
    /// One commitment per kept (variable) column, in S,O order (predicate is constant).
    pub com_kept: Vec<ArkGT>,
    pub m: usize,
}

/// A general n-atom BGP proof over `present`: the two present-id authentications shared by
/// every atom, one per-atom (selection + compaction) proof, and the left-deep join chain
/// over the filtered relations when `k >= 2` (a single atom's filtered relation is already
/// the whole result). All under the reconciliation transcript.
#[derive(Clone, Debug)]
pub struct PresentBgpProof {
    pub auth_s: PresentIdAuth,
    pub auth_o: PresentIdAuth,
    pub atoms: Vec<PresentBgpAtomProof>,
    pub chain: Option<BgpChainProof>,
    pub result_len: usize,
    pub result_log_domain: usize,
    pub result_arity: usize,
}

/// Build the hash-space [`AtomFilterSpec`] (predicate/subject/object constants +
/// subject==object equality) and the KEPT variable positions (S then O; predicate is a
/// constant so it is never kept) for a `present` atom. Rejects the unsupported shapes.
fn present_atom_spec(atom: &PresentBgpAtom) -> io::Result<(AtomFilterSpec, Vec<Position>)> {
    let mut constants = vec![(Position::P, atom.predicate_hash)];
    if let PresentTerm::Constant(h) = atom.subject {
        constants.push((Position::S, h));
    }
    if let PresentTerm::Constant(h) = atom.object {
        constants.push((Position::O, h));
    }
    let mut kept = Vec::new();
    if matches!(atom.subject, PresentTerm::Variable) {
        kept.push(Position::S);
    }
    if matches!(atom.object, PresentTerm::Variable) {
        kept.push(Position::O);
    }
    let equalities = if atom.subject_object_equal {
        if !matches!(atom.subject, PresentTerm::Variable) || !matches!(atom.object, PresentTerm::Variable) {
            return Err(invalid("present BGP: subject==object equality requires two variable positions"));
        }
        // The object is aliased to the subject: keep only the subject column.
        kept.retain(|p| *p != Position::O);
        vec![(Position::S, Position::O)]
    } else {
        Vec::new()
    };
    if kept.is_empty() {
        return Err(invalid(
            "present BGP: each atom must keep at least one variable this pass (fully-ground atoms unsupported)",
        ));
    }
    // The spec is a valid S/P/O cover by construction; `prove/verify_selection_only`
    // re-check it internally (`AtomFilterSpec::validate`).
    let spec = AtomFilterSpec { constants, kept: kept.clone(), equalities };
    Ok((spec, kept))
}

/// The kept-column combined-dict id commitments for `kept`, in order (S -> `present_id_s`,
/// O -> `present_id_o`). Both are authenticated to the reconciled hashes by [`verify_id_auth`].
fn present_kept_source_coms(
    kept: &[Position],
    auth_s: &PresentIdAuth,
    auth_o: &PresentIdAuth,
) -> io::Result<Vec<ArkGT>> {
    kept.iter()
        .map(|pos| match pos {
            Position::S => Ok(auth_s.com_id),
            Position::O => Ok(auth_o.com_id),
            Position::P => Err(invalid("present BGP: predicate variable is unsupported this pass")),
        })
        .collect()
}

fn prove_present_bgp(
    context: &StoreProverContext<'_>,
    artifacts: &ReconcileArtifacts,
    atoms: &[PresentBgpAtom],
    joins: &[JoinStep],
    transcript: &mut Transcript254,
) -> io::Result<(PresentBgpProof, PresentReadResult)> {
    if atoms.is_empty() {
        return Err(invalid("present BGP needs at least one atom"));
    }
    let m = artifacts.present_len;
    // (1) Authenticate present's subject/object combined-dict id columns (shared by every atom).
    let auth_s = prove_id_auth(context, &artifacts.present_id_s, m, &artifacts.nv_dict, transcript)?;
    let auth_o = prove_id_auth(context, &artifacts.present_id_o, m, &artifacts.nv_dict, transcript)?;

    // (2) Per-atom: authenticated equality selection over present's HASH columns, then
    //     compaction of the kept combined-dict id columns to the matching rows.
    let mut atom_proofs = Vec::with_capacity(atoms.len());
    let mut relations: Vec<RelationWitness> = Vec::with_capacity(atoms.len());
    for atom in atoms {
        let (spec, kept) = present_atom_spec(atom)?;
        let (selection, com_sel, sel) = prove_selection_only_with_context(
            context, &spec,
            &artifacts.present_s,
            &artifacts.present_p,
            &artifacts.present_o,
            m,
            &artifacts.com_present_s,
            &artifacts.com_present_p,
            &artifacts.com_present_o,
            transcript,
        )?;
        let source_coms = present_kept_source_coms(&kept, &auth_s, &auth_o)?;
        let sources: Vec<Vec<ArkFr>> = kept
            .iter()
            .map(|pos| match pos {
                Position::S => artifacts.present_id_s.clone(),
                Position::O => artifacts.present_id_o.clone(),
                Position::P => unreachable!("predicate is a constant this pass"),
            })
            .collect();
        let (compaction, com_kept, kept_out, mm) = prove_compaction_only_with_context(
            context, &sources, &source_coms, &sel, &com_sel, transcript,
        )?;
        relations.push(RelationWitness {
            columns: kept_out,
            commitments: com_kept.clone(),
            log_domain: artifacts.working_log,
            n: mm,
        });
        atom_proofs.push(PresentBgpAtomProof { selection, com_sel, compaction, com_kept, m: mm });
    }

    // (3) The left-deep chain over the filtered relations (k = 1 is already the result).
    let (chain, result_columns, result_com_count, result_log_domain, result_m) = if relations.len() == 1 {
        let r = &relations[0];
        (None, r.columns.clone(), r.commitments.len(), r.log_domain, r.n)
    } else {
        let res = prove_bgp_chain_with_result_and_context(context, &relations, joins, transcript)?;
        let arity = res.commitments.len();
        (Some(res.proof), res.columns, arity, res.log_domain, res.n)
    };

    let columns: Vec<Vec<ArkFr>> = result_columns.iter().map(|c| c[..result_m].to_vec()).collect();
    Ok((
        PresentBgpProof {
            auth_s,
            auth_o,
            atoms: atom_proofs,
            chain,
            result_len: result_m,
            result_log_domain,
            result_arity: result_com_count,
        },
        PresentReadResult { columns, len: result_m },
    ))
}

fn verify_present_bgp(
    reconcile: &ReconcileProof,
    verified: &ReconcileVerified,
    head_nv: usize,
    atoms: &[PresentBgpAtom],
    joins: &[JoinStep],
    proof: &PresentBgpProof,
    transcript: &mut Transcript254,
) -> io::Result<(Vec<ArkGT>, usize, usize)> {
    if atoms.is_empty() {
        return Err(invalid("present BGP needs at least one atom"));
    }
    if proof.atoms.len() != atoms.len() {
        return Err(invalid("present BGP atom count mismatch"));
    }
    let m = verified.present_len;
    // (1) present id authentication against the reconciled hashes.
    verify_id_auth(&reconcile.combined_nv, m, head_nv, verified.com_present_s, &proof.auth_s, transcript)?;
    verify_id_auth(&reconcile.combined_nv, m, head_nv, verified.com_present_o, &proof.auth_o, transcript)?;

    // (2) Per-atom selection + compaction, rebuilding each filtered relation from the
    //     authenticated kept commitments (never trusting a fresh relation commitment).
    let mut relations: Vec<RelationCommit> = Vec::with_capacity(atoms.len());
    for (atom, ap) in atoms.iter().zip(proof.atoms.iter()) {
        let (spec, kept) = present_atom_spec(atom)?;
        verify_selection_only(
            &spec, m, &verified.com_present_s, &verified.com_present_p, &verified.com_present_o,
            &ap.com_sel, &ap.selection, transcript,
        )?;
        let source_coms = present_kept_source_coms(&kept, &proof.auth_s, &proof.auth_o)?;
        if ap.com_kept.len() != source_coms.len() {
            return Err(invalid("present BGP: kept commitment arity does not match the atom's variables"));
        }
        verify_compaction_only(&source_coms, &ap.com_sel, &ap.com_kept, ap.m, &ap.compaction, transcript)?;
        relations.push(RelationCommit {
            commitments: ap.com_kept.clone(),
            log_domain: verified.working_log,
            n: ap.m,
        });
    }

    // (3) The left-deep chain over the filtered relations.
    let (result_coms, result_m, result_log_domain) = if relations.len() == 1 {
        if proof.chain.is_some() {
            return Err(invalid("single-atom present BGP carries a spurious join chain"));
        }
        (relations[0].commitments.clone(), relations[0].n, verified.working_log)
    } else {
        let chain = proof
            .chain
            .as_ref()
            .ok_or_else(|| invalid("multi-atom present BGP is missing its join chain"))?;
        let (coms, mm) = verify_bgp_chain(&relations, joins, chain, transcript)?;
        let log = chain
            .stages
            .last()
            .ok_or_else(|| invalid("present BGP join chain has no stages"))?
            .join
            .num_vars();
        (coms, mm, log)
    };

    if proof.result_len != result_m
        || proof.result_arity != result_coms.len()
        || proof.result_log_domain != result_log_domain
    {
        return Err(invalid("present BGP proof result shape disagrees with the verified chain"));
    }
    Ok((result_coms, result_m, result_log_domain))
}

/// Prove a general n-atom conjunctive BGP over the reconciled N-layer `present`. `atoms`
/// are the conjuncts (constant predicate, variable/constant subject+object, optional
/// subject==object); `joins` is the left-deep plan over the kept id columns (empty for a
/// single atom). Reconciliation + BGP share ONE transcript, so the result is bound to the
/// reconciliation-authenticated relation. The delivered result columns are combined-dict
/// node/value ids (resolution deferred to [`verify_and_resolve_bgp`]).
pub fn prove_read_bgp(
    layers: &[&LayerCommitment],
    atoms: &[PresentBgpAtom],
    joins: &[JoinStep],
) -> io::Result<(ReconcileProof, PresentBgpProof, PresentReadResult)> {
    let context = StoreProverContext::cpu();
    let mut transcript = Transcript254::new(b"parture-delta-reconcile-envelope");
    let (reconcile, artifacts) = prove_reconcile_core(&context, layers, &mut transcript)?;
    let (bgp, result) = prove_present_bgp(&context, &artifacts, atoms, joins, &mut transcript)?;
    Ok((reconcile, bgp, result))
}

/// Verify a general n-atom BGP read over `present`. On success returns the authenticated
/// result row count.
pub fn verify_read_bgp(
    layers: &[&LayerCommitment],
    trusted_head_root: ProofHash,
    atoms: &[PresentBgpAtom],
    joins: &[JoinStep],
    reconcile: &ReconcileProof,
    proof: &PresentBgpProof,
) -> io::Result<usize> {
    let mut transcript = Transcript254::new(b"parture-delta-reconcile-envelope");
    let verified = verify_reconcile_core(layers, trusted_head_root, reconcile, &mut transcript)?;
    let head_nv = layers.last().unwrap().verifier_view().state.node_value_count as usize;
    let (_coms, result_m, _log) =
        verify_present_bgp(reconcile, &verified, head_nv, atoms, joins, proof, &mut transcript)?;
    Ok(result_m)
}

/// Verify a general n-atom BGP read AND resolve its result bindings to `CanonicalObject`s.
/// Every result column is a combined-dict node/value id (predicates are constants), so
/// each resolves against the chain node/value catalogs.
pub fn verify_and_resolve_bgp(
    layers: &[&LayerCommitment],
    trusted_head_root: ProofHash,
    atoms: &[PresentBgpAtom],
    joins: &[JoinStep],
    reconcile: &ReconcileProof,
    proof: &PresentBgpProof,
    claimed_columns: &[Vec<ArkFr>],
) -> io::Result<Vec<Vec<CanonicalObject>>> {
    let mut transcript = Transcript254::new(b"parture-delta-reconcile-envelope");
    let verified = verify_reconcile_core(layers, trusted_head_root, reconcile, &mut transcript)?;
    let head_nv = layers.last().unwrap().verifier_view().state.node_value_count as usize;
    let (result_coms, result_m, result_log_domain) =
        verify_present_bgp(reconcile, &verified, head_nv, atoms, joins, proof, &mut transcript)?;
    let views: Vec<VerifierCommitmentRef<'_>> = layers.iter().map(|l| l.verifier_view()).collect();
    let kinds = vec![ResolveKind::NodeValueId; result_coms.len()];
    resolve_result_columns(&views, claimed_columns, &result_coms, result_m, result_log_domain, &kinds)
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

    fn commit_col(v: &[ArkFr]) -> ArkGT {
        crate::proof::argument::commit_column(v).unwrap()
    }

    /// The materialized ground-truth present set as canonical (S,P,O) hash rows.
    fn materialized_rows(layers: &[&LayerCommitment]) -> Vec<(ArkFr, ArkFr, ArkFr)> {
        reconcile_present(layers)
            .unwrap()
            .iter()
            .map(|r| (subject_hash(r), predicate_hash(r), object_hash(r)))
            .collect()
    }

    fn bag_eq(mut a: Vec<(ArkFr, ArkFr, ArkFr)>, mut b: Vec<(ArkFr, ArkFr, ArkFr)>) -> bool {
        if a.len() != b.len() {
            return false;
        }
        let key = |t: &(ArkFr, ArkFr, ArkFr)| (format!("{:?}", t.0), format!("{:?}", t.1), format!("{:?}", t.2));
        a.sort_by_key(&key);
        b.sort_by_key(&key);
        a == b
    }

    fn refs(layers: &[Arc<LayerCommitment>]) -> Vec<&LayerCommitment> {
        layers.iter().map(|l| l.as_ref()).collect()
    }

    // --- Base + 1 delta (skeleton case must not regress) ---
    fn base_plus_one() -> Vec<Arc<LayerCommitment>> {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for (s, o) in [("a1", "b1"), ("a2", "b2"), ("a3", "b3"), ("a4", "b4")] {
            builder.add_value_triple(ValueTriple::new_node(&iri(s), &iri("p1"), &iri(o))).unwrap();
        }
        let base = builder.commit().unwrap();
        let base_lc = base.proof_commitment().unwrap();
        let child = base.open_write().unwrap();
        for (s, o) in [("a1", "b1"), ("a2", "b2")] {
            child.remove_value_triple(ValueTriple::new_node(&iri(s), &iri("p1"), &iri(o))).unwrap();
        }
        for (s, o) in [("a3", "b5"), ("a5", "b6")] {
            child.add_value_triple(ValueTriple::new_node(&iri(s), &iri("p1"), &iri(o))).unwrap();
        }
        let head = child.commit().unwrap();
        vec![base_lc, head.proof_commitment().unwrap()]
    }

    #[test]
    fn base_plus_one_reconciles_and_counts() {
        let layers = base_plus_one();
        let lref = refs(&layers);
        let (proof, present) = prove_reconcile(&lref).unwrap();
        let head_root = lref.last().unwrap().state.commitment_root;
        let m = verify_reconcile(&lref, head_root, &proof).unwrap();
        let ground = materialized_rows(&lref);
        assert_eq!(m, ground.len());
        assert_eq!(m, 4);
        assert!(bag_eq(present.rows.clone(), ground));
    }

    // --- Base + 3 deltas: adds AND removes across layers, add-then-remove,
    //     remove-of-inherited, and a delta with a BRAND-NEW predicate. ---
    fn base_plus_three() -> Vec<Arc<LayerCommitment>> {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        // Base: a1..a5 -> p1 -> b1..b5
        for i in 1..=5 {
            builder
                .add_value_triple(ValueTriple::new_node(&iri(&format!("a{i}")), &iri("p1"), &iri(&format!("b{i}"))))
                .unwrap();
        }
        let base = builder.commit().unwrap();
        let base_lc = base.proof_commitment().unwrap();

        // D1: remove {a1->b1, a2->b2} (inherited); add {a6->b6 (new terms), a3->b7 (b7 new)}.
        let d1 = base.open_write().unwrap();
        d1.remove_value_triple(ValueTriple::new_node(&iri("a1"), &iri("p1"), &iri("b1"))).unwrap();
        d1.remove_value_triple(ValueTriple::new_node(&iri("a2"), &iri("p1"), &iri("b2"))).unwrap();
        d1.add_value_triple(ValueTriple::new_node(&iri("a6"), &iri("p1"), &iri("b6"))).unwrap();
        d1.add_value_triple(ValueTriple::new_node(&iri("a3"), &iri("p1"), &iri("b7"))).unwrap();
        let d1c = d1.commit().unwrap();
        let d1_lc = d1c.proof_commitment().unwrap();

        // D2: add-then-remove — add {a7->b8, a8->b9}, then D3 removes a7->b8.
        //     Also uses a BRAND-NEW predicate p2 for {a7->b8}.
        let d2 = d1c.open_write().unwrap();
        d2.add_value_triple(ValueTriple::new_node(&iri("a7"), &iri("p2"), &iri("b8"))).unwrap();
        d2.add_value_triple(ValueTriple::new_node(&iri("a8"), &iri("p1"), &iri("b9"))).unwrap();
        let d2c = d2.commit().unwrap();
        let d2_lc = d2c.proof_commitment().unwrap();

        // D3: remove {a7->b8 (added in D2, p2), a3->b3 (inherited from base)}; add {a9->b10, a1->b1 (re-add removed base terms)}.
        let d3 = d2c.open_write().unwrap();
        d3.remove_value_triple(ValueTriple::new_node(&iri("a7"), &iri("p2"), &iri("b8"))).unwrap();
        d3.remove_value_triple(ValueTriple::new_node(&iri("a3"), &iri("p1"), &iri("b3"))).unwrap();
        d3.add_value_triple(ValueTriple::new_node(&iri("a9"), &iri("p1"), &iri("b10"))).unwrap();
        d3.add_value_triple(ValueTriple::new_node(&iri("a1"), &iri("p1"), &iri("b1"))).unwrap();
        let d3c = d3.commit().unwrap();
        let d3_lc = d3c.proof_commitment().unwrap();

        vec![base_lc, d1_lc, d2_lc, d3_lc]
    }

    #[test]
    fn n_layer_reconciles_and_counts() {
        let layers = base_plus_three();
        let lref = refs(&layers);
        let (proof, present) = prove_reconcile(&lref).unwrap();
        let head_root = lref.last().unwrap().state.commitment_root;
        let m = verify_reconcile(&lref, head_root, &proof).unwrap();
        let ground = materialized_rows(&lref);
        assert_eq!(m, ground.len(), "reconciled count must equal materialized count");
        assert!(bag_eq(present.rows.clone(), ground), "present bag must equal materialized bag");
        // Sanity: the new predicate p2's triple a7->b8 was added (D2) and removed (D3),
        // so it is NOT present; a1->b1 was removed (D1) then re-added (D3) → present.
    }

    #[test]
    fn new_predicate_delta_reconciles() {
        // A base with one predicate, then a delta introducing a brand-new predicate.
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for (s, o) in [("a1", "b1"), ("a2", "b2"), ("a3", "b3")] {
            builder.add_value_triple(ValueTriple::new_node(&iri(s), &iri("p1"), &iri(o))).unwrap();
        }
        let base = builder.commit().unwrap();
        let base_lc = base.proof_commitment().unwrap();
        let child = base.open_write().unwrap();
        // New predicate p2 on two triples (>= 2 for the group domain), reusing inherited subjects.
        child.add_value_triple(ValueTriple::new_node(&iri("a1"), &iri("p2"), &iri("b1"))).unwrap();
        child.add_value_triple(ValueTriple::new_node(&iri("a2"), &iri("p2"), &iri("b2"))).unwrap();
        let head = child.commit().unwrap();
        let head_lc = head.proof_commitment().unwrap();
        assert!(head_lc.state.predicate_count > head_lc.state.parent_predicate_count, "delta must add a predicate");

        let layers = vec![base_lc, head_lc];
        let lref = refs(&layers);
        let (proof, present) = prove_reconcile(&lref).unwrap();
        let head_root = lref.last().unwrap().state.commitment_root;
        let m = verify_reconcile(&lref, head_root, &proof).unwrap();
        let ground = materialized_rows(&lref);
        assert_eq!(m, ground.len());
        assert_eq!(m, 5); // 3 base + 2 new-predicate additions, no removals
        assert!(bag_eq(present.rows.clone(), ground));
    }

    // --- Soundness negatives (generalized to the chain) ---

    #[test]
    fn wrong_or_swapped_head_root_rejects() {
        let layers = base_plus_three();
        let lref = refs(&layers);
        let (proof, _) = prove_reconcile(&lref).unwrap();
        // Anchor to a non-head layer root.
        assert!(verify_reconcile(&lref, lref[0].state.commitment_root, &proof).is_err());
        let mut bogus = lref.last().unwrap().state.commitment_root;
        bogus[0] ^= 0xff;
        assert!(verify_reconcile(&lref, bogus, &proof).is_err());
    }

    #[test]
    fn dropped_middle_layer_rejects() {
        let layers = base_plus_three();
        let lref = refs(&layers);
        let (proof, _) = prove_reconcile(&lref).unwrap();
        let head_root = lref.last().unwrap().state.commitment_root;
        // Drop D2 (index 2): the parent-state-root chain no longer links D3 to D1.
        let dropped = vec![lref[0], lref[1], lref[3]];
        assert!(verify_reconcile(&dropped, head_root, &proof).is_err());
    }

    #[test]
    fn reordered_middle_layers_reject() {
        let layers = base_plus_three();
        let lref = refs(&layers);
        let (proof, _) = prove_reconcile(&lref).unwrap();
        let head_root = lref.last().unwrap().state.commitment_root;
        // Swap D1 and D2: the chain breaks.
        let reordered = vec![lref[0], lref[2], lref[1], lref[3]];
        assert!(verify_reconcile(&reordered, head_root, &proof).is_err());
    }

    #[test]
    fn spliced_foreign_layer_rejects() {
        let layers = base_plus_three();
        let other = base_plus_one();
        let lref = refs(&layers);
        let oref = refs(&other);
        let (proof, _) = prove_reconcile(&lref).unwrap();
        let head_root = lref.last().unwrap().state.commitment_root;
        // Replace D2 with a foreign delta from an unrelated stack.
        let spliced = vec![lref[0], lref[1], oref[1], lref[3]];
        assert!(verify_reconcile(&spliced, head_root, &proof).is_err());
    }

    #[test]
    fn tampered_present_row_rejects() {
        let layers = base_plus_three();
        let lref = refs(&layers);
        let (mut proof, _) = prove_reconcile(&lref).unwrap();
        let head_root = lref.last().unwrap().state.commitment_root;
        proof.com_present_s = commit_col(&vec![ark(999); 1usize << proof.working_log]);
        assert!(verify_reconcile(&lref, head_root, &proof).is_err());
    }

    #[test]
    fn hidden_removal_in_any_layer_rejects() {
        // Tamper a removal group's authenticated hash column (misrepresenting the
        // root-bound removal set of some layer): its lookup binding breaks.
        let layers = base_plus_three();
        let lref = refs(&layers);
        let (mut proof, _) = prove_reconcile(&lref).unwrap();
        let head_root = lref.last().unwrap().state.commitment_root;
        // Find a removal group (sign −1) and corrupt its subject hash lookup.
        let column_commitments: Vec<&[ArkGT; 6]> = lref.iter().map(|l| &l.column_commitments).collect();
        let adds: Vec<usize> = lref.iter().map(|l| l.additions.len()).collect();
        let rems: Vec<usize> = lref.iter().map(|l| l.removals.len()).collect();
        let descs = group_descs(&column_commitments, &adds, &rems).unwrap();
        let rem_idx = descs.iter().position(|d| d.sign == f0() - f1()).expect("a removal group exists");
        proof.groups[rem_idx].s.lookup.com_v_hash = commit_col(&vec![ark(13); 1usize << proof.working_log]);
        assert!(verify_reconcile(&lref, head_root, &proof).is_err());
    }

    #[test]
    fn injected_removal_count_rejects() {
        // Claiming one fewer live row (present length) than the delta counts imply.
        let layers = base_plus_three();
        let lref = refs(&layers);
        let (proof, _) = prove_reconcile(&lref).unwrap();
        let head_root = lref.last().unwrap().state.commitment_root;
        let mut forged = proof.clone();
        forged.present_len -= 1;
        assert!(verify_reconcile(&lref, head_root, &forged).is_err());
    }

    #[test]
    fn tampered_combined_dict_rejects() {
        let layers = base_plus_three();
        let lref = refs(&layers);
        let (mut proof, _) = prove_reconcile(&lref).unwrap();
        let head_root = lref.last().unwrap().state.commitment_root;
        proof.combined_nv.com_hash = commit_col(&vec![ark(7); 1usize << proof.combined_nv.log]);
        assert!(verify_reconcile(&lref, head_root, &proof).is_err());
    }

    // ============================================================
    // Wave 3 — single-atom BGP (filter) over present.
    // ============================================================

    fn hash_pred(iri: &str) -> ArkFr {
        hash_term(iri.as_bytes())
    }
    fn hash_node(iri: &str) -> ArkFr {
        hash_object(&CanonicalObject::Node(iri.as_bytes().to_vec()))
    }

    /// Base+delta with MIXED predicates so a predicate filter selects a proper subset.
    /// Base: {a1 p1 b1, a2 p1 b2, a3 p2 c3, a4 p2 c4, a7 p1 b7, a8 p2 c8}.
    /// Delta: remove {a1 p1 b1, a7 p1 b7}; add {a5 p1 b5, a6 p2 c6}.
    /// present = {a2 p1 b2, a3 p2 c3, a4 p2 c4, a8 p2 c8, a5 p1 b5, a6 p2 c6}.
    fn filter_stack() -> Vec<Arc<LayerCommitment>> {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for (s, p, o) in [
            ("a1", "p1", "b1"), ("a2", "p1", "b2"), ("a3", "p2", "c3"),
            ("a4", "p2", "c4"), ("a7", "p1", "b7"), ("a8", "p2", "c8"),
        ] {
            builder.add_value_triple(ValueTriple::new_node(&iri(s), &iri(p), &iri(o))).unwrap();
        }
        let base = builder.commit().unwrap();
        let base_lc = base.proof_commitment().unwrap();
        let child = base.open_write().unwrap();
        child.remove_value_triple(ValueTriple::new_node(&iri("a1"), &iri("p1"), &iri("b1"))).unwrap();
        child.remove_value_triple(ValueTriple::new_node(&iri("a7"), &iri("p1"), &iri("b7"))).unwrap();
        child.add_value_triple(ValueTriple::new_node(&iri("a5"), &iri("p1"), &iri("b5"))).unwrap();
        child.add_value_triple(ValueTriple::new_node(&iri("a6"), &iri("p2"), &iri("c6"))).unwrap();
        let head = child.commit().unwrap();
        vec![base_lc, head.proof_commitment().unwrap()]
    }

    /// Materialized single-atom filter: present rows whose `position` hashes to `constant`,
    /// output the other two columns (matching `filter_selection`'s output order).
    fn materialized_filter(layers: &[&LayerCommitment], position: FilterPosition, constant: ArkFr) -> Vec<(ArkFr, ArkFr)> {
        reconcile_present(layers)
            .unwrap()
            .iter()
            .filter_map(|r| {
                let (sh, ph, oh) = (subject_hash(r), predicate_hash(r), object_hash(r));
                let (key, out) = match position {
                    FilterPosition::Predicate => (ph, (sh, oh)),
                    FilterPosition::Subject => (sh, (ph, oh)),
                    FilterPosition::Object => (oh, (sh, ph)),
                };
                if key == constant { Some(out) } else { None }
            })
            .collect()
    }

    fn out_pairs(result: &PresentReadResult) -> Vec<(ArkFr, ArkFr)> {
        (0..result.len).map(|i| (result.columns[0][i], result.columns[1][i])).collect()
    }

    fn bag_eq_pairs(mut a: Vec<(ArkFr, ArkFr)>, mut b: Vec<(ArkFr, ArkFr)>) -> bool {
        if a.len() != b.len() {
            return false;
        }
        let key = |t: &(ArkFr, ArkFr)| (format!("{:?}", t.0), format!("{:?}", t.1));
        a.sort_by_key(&key);
        b.sort_by_key(&key);
        a == b
    }

    fn run_filter(position: FilterPosition, constant: ArkFr, expect: usize) {
        let layers = filter_stack();
        let lref = refs(&layers);
        let (rec, filt, result) = prove_read_single_atom(&lref, position, constant).unwrap();
        let head_root = lref.last().unwrap().state.commitment_root;
        let m = verify_read_single_atom(&lref, head_root, position, constant, &rec, &filt).unwrap();
        let ground = materialized_filter(&lref, position, constant);
        assert_eq!(m, ground.len(), "count must equal materialized filter");
        assert_eq!(m, expect);
        assert!(bag_eq_pairs(out_pairs(&result), ground), "result bag must equal materialized filter");
    }

    #[test]
    fn single_atom_predicate_constant() {
        // Triple(?s, p1, ?o): present has 2 p1 rows (a2->b2, a5->b5) and 4 p2 rows.
        run_filter(FilterPosition::Predicate, hash_pred("http://ex/p1"), 2);
        run_filter(FilterPosition::Predicate, hash_pred("http://ex/p2"), 4);
        // A predicate absent from present yields the empty answer.
        run_filter(FilterPosition::Predicate, hash_pred("http://ex/pX"), 0);
    }

    #[test]
    fn single_atom_subject_constant() {
        // Triple(a2, ?p, ?o): exactly {a2 p1 b2}.
        run_filter(FilterPosition::Subject, hash_node("http://ex/a2"), 1);
        // A removed subject (a1) is absent.
        run_filter(FilterPosition::Subject, hash_node("http://ex/a1"), 0);
    }

    #[test]
    fn single_atom_object_constant() {
        // Triple(?s, ?p, c3): exactly {a3 p2 c3}.
        run_filter(FilterPosition::Object, hash_node("http://ex/c3"), 1);
    }

    #[test]
    fn filter_fabricated_result_row_rejects() {
        let layers = filter_stack();
        let lref = refs(&layers);
        let c = hash_pred("http://ex/p1");
        let (rec, mut filt, _) = prove_read_single_atom(&lref, FilterPosition::Predicate, c).unwrap();
        let head_root = lref.last().unwrap().state.commitment_root;
        // Replace an output column commitment with a fabricated one: the compaction binding fails.
        filt.com_out[0] = commit_col(&vec![ark(4242); 1usize << rec.working_log]);
        assert!(verify_read_single_atom(&lref, head_root, FilterPosition::Predicate, c, &rec, &filt).is_err());
    }

    #[test]
    fn filter_dropped_row_rejects() {
        let layers = filter_stack();
        let lref = refs(&layers);
        let c = hash_pred("http://ex/p2"); // 4 matching rows
        let (rec, filt, _) = prove_read_single_atom(&lref, FilterPosition::Predicate, c).unwrap();
        let head_root = lref.last().unwrap().state.commitment_root;
        let mut forged = filt.clone();
        forged.out_len -= 1; // claim one fewer matching row
        assert!(verify_read_single_atom(&lref, head_root, FilterPosition::Predicate, c, &rec, &forged).is_err());
    }

    #[test]
    fn filter_wrong_constant_rejects() {
        let layers = filter_stack();
        let lref = refs(&layers);
        let c = hash_pred("http://ex/p1");
        let (rec, filt, _) = prove_read_single_atom(&lref, FilterPosition::Predicate, c).unwrap();
        let head_root = lref.last().unwrap().state.commitment_root;
        // Verifying the p1 proof against the p2 constant must reject (selection is p1-bound).
        let wrong = hash_pred("http://ex/p2");
        assert!(verify_read_single_atom(&lref, head_root, FilterPosition::Predicate, wrong, &rec, &filt).is_err());
    }

    // ============================================================
    // Wave 3 — two-atom JOIN over present.
    // ============================================================

    /// A stack whose present supports Triple(?x, p1, ?y) & Triple(?y, p2, ?z):
    /// present p1 = {x1->y1, x2->y1, x5->y1}, p2 = {y1->z1, y1->z2, y1->z5}
    /// (base adds two non-joining rows x3->p1->y3, y4->p2->z4 which the delta removes).
    /// Join on ?y = y1 => 3 x-bindings × 3 z-bindings = 9 solutions.
    fn join_stack() -> Vec<Arc<LayerCommitment>> {
        let store = open_sync_memory_store();
        let b = store.create_base_layer().unwrap();
        for (s, p, o) in [
            ("x1", "p1", "y1"), ("x2", "p1", "y1"), ("y1", "p2", "z1"),
            ("y1", "p2", "z2"), ("x3", "p1", "y3"), ("y4", "p2", "z4"),
        ] {
            b.add_value_triple(ValueTriple::new_node(&iri(s), &iri(p), &iri(o))).unwrap();
        }
        let base = b.commit().unwrap();
        let base_lc = base.proof_commitment().unwrap();
        let child = base.open_write().unwrap();
        child.remove_value_triple(ValueTriple::new_node(&iri("x3"), &iri("p1"), &iri("y3"))).unwrap();
        child.remove_value_triple(ValueTriple::new_node(&iri("y4"), &iri("p2"), &iri("z4"))).unwrap();
        child.add_value_triple(ValueTriple::new_node(&iri("x5"), &iri("p1"), &iri("y1"))).unwrap();
        child.add_value_triple(ValueTriple::new_node(&iri("y1"), &iri("p2"), &iri("z5"))).unwrap();
        let head = child.commit().unwrap();
        vec![base_lc, head.proof_commitment().unwrap()]
    }

    /// Materialized 2-atom join over present, in COMBINED-DICT id space (matching the proof's
    /// `[id_x, id_y, id_z]` output).
    fn materialized_join(layers: &[&LayerCommitment], p: &str, q: &str) -> Vec<(ArkFr, ArkFr, ArkFr)> {
        let present = reconcile_present(layers).unwrap();
        let mut out = Vec::new();
        for a in present.iter().filter(|r| r.triple.predicate == p.as_bytes()) {
            for c in present.iter().filter(|r| r.triple.predicate == q.as_bytes()) {
                if a.object_id == c.subject_id {
                    out.push((ark(a.subject_id), ark(a.object_id), ark(c.object_id)));
                }
            }
        }
        out
    }

    fn join_out(result: &PresentReadResult) -> Vec<(ArkFr, ArkFr, ArkFr)> {
        (0..result.len)
            .map(|i| (result.columns[0][i], result.columns[1][i], result.columns[2][i]))
            .collect()
    }

    #[test]
    fn two_atom_join_over_present() {
        let layers = join_stack();
        let lref = refs(&layers);
        let (cp, cq) = (hash_pred("http://ex/p1"), hash_pred("http://ex/p2"));
        let (rec, join, result) = prove_read_two_atom_join(&lref, cp, cq).unwrap();
        let head_root = lref.last().unwrap().state.commitment_root;
        let n = verify_read_two_atom_join(&lref, head_root, cp, cq, &rec, &join).unwrap();
        let ground = materialized_join(&lref, "http://ex/p1", "http://ex/p2");
        assert_eq!(n, ground.len(), "join count must equal materialized join");
        assert_eq!(n, 9);
        assert!(bag_eq(join_out(&result), ground), "join result bag must equal materialized join");
    }

    #[test]
    fn join_fabricated_row_rejects() {
        let layers = join_stack();
        let lref = refs(&layers);
        let (cp, cq) = (hash_pred("http://ex/p1"), hash_pred("http://ex/p2"));
        let (rec, mut join, _) = prove_read_two_atom_join(&lref, cp, cq).unwrap();
        let head_root = lref.last().unwrap().state.commitment_root;
        // Fabricate an output column: the inner-join matched-projection binding fails.
        join.com_join_out[0] = commit_col(&vec![ark(777); 1usize << join.join_log]);
        assert!(verify_read_two_atom_join(&lref, head_root, cp, cq, &rec, &join).is_err());
    }

    #[test]
    fn join_dropped_row_rejects() {
        let layers = join_stack();
        let lref = refs(&layers);
        let (cp, cq) = (hash_pred("http://ex/p1"), hash_pred("http://ex/p2"));
        let (rec, join, _) = prove_read_two_atom_join(&lref, cp, cq).unwrap();
        let head_root = lref.last().unwrap().state.commitment_root;
        let mut forged = join.clone();
        forged.join_out_len -= 1; // claim one fewer solution
        assert!(verify_read_two_atom_join(&lref, head_root, cp, cq, &rec, &forged).is_err());
    }

    #[test]
    fn join_wrong_predicate_rejects() {
        let layers = join_stack();
        let lref = refs(&layers);
        let (cp, cq) = (hash_pred("http://ex/p1"), hash_pred("http://ex/p2"));
        let (rec, join, _) = prove_read_two_atom_join(&lref, cp, cq).unwrap();
        let head_root = lref.last().unwrap().state.commitment_root;
        // Verify against a wrong second predicate: the second atom's selection is p2-bound.
        let wrong = hash_pred("http://ex/pX");
        assert!(verify_read_two_atom_join(&lref, head_root, cp, wrong, &rec, &join).is_err());
    }

    #[test]
    fn join_tampered_present_id_rejects() {
        // A prover who feeds a wrong combined-dict id for present (to fabricate a joining
        // pair) fails the id → hash authentication against the reconciled hashes.
        let layers = join_stack();
        let lref = refs(&layers);
        let (cp, cq) = (hash_pred("http://ex/p1"), hash_pred("http://ex/p2"));
        let (rec, mut join, _) = prove_read_two_atom_join(&lref, cp, cq).unwrap();
        let head_root = lref.last().unwrap().state.commitment_root;
        join.auth_s.com_id = commit_col(&vec![ark(3); 1usize << rec.working_log]);
        assert!(verify_read_two_atom_join(&lref, head_root, cp, cq, &rec, &join).is_err());
    }

    // ============================================================
    // Wave 4 — result resolution to CanonicalObjects.
    // ============================================================

    fn rows_of(cols: &[Vec<CanonicalObject>]) -> Vec<Vec<CanonicalObject>> {
        if cols.is_empty() {
            return vec![];
        }
        let n = cols[0].len();
        (0..n).map(|i| cols.iter().map(|c| c[i].clone()).collect()).collect()
    }

    fn bag_eq_rows(mut a: Vec<Vec<CanonicalObject>>, mut b: Vec<Vec<CanonicalObject>>) -> bool {
        if a.len() != b.len() {
            return false;
        }
        let key = |r: &Vec<CanonicalObject>| format!("{r:?}");
        a.sort_by_key(&key);
        b.sort_by_key(&key);
        a == b
    }

    /// Materialized single-atom filter, RESOLVED to CanonicalObjects (matching `filter_output_kinds`).
    fn materialized_filter_resolved(
        layers: &[&LayerCommitment],
        position: FilterPosition,
        constant: ArkFr,
    ) -> Vec<Vec<CanonicalObject>> {
        reconcile_present(layers)
            .unwrap()
            .iter()
            .filter_map(|r| {
                let (sh, ph, oh) = (subject_hash(r), predicate_hash(r), object_hash(r));
                let node_s = CanonicalObject::Node(r.triple.subject.clone());
                let node_p = CanonicalObject::Node(r.triple.predicate.clone());
                let obj = r.triple.object.clone();
                let (key, out) = match position {
                    FilterPosition::Predicate => (ph, vec![node_s, obj]),
                    FilterPosition::Subject => (sh, vec![node_p, obj]),
                    FilterPosition::Object => (oh, vec![node_s, node_p]),
                };
                if key == constant { Some(out) } else { None }
            })
            .collect()
    }

    fn materialized_join_resolved(layers: &[&LayerCommitment], p: &str, q: &str) -> Vec<Vec<CanonicalObject>> {
        let present = reconcile_present(layers).unwrap();
        let mut out = Vec::new();
        for a in present.iter().filter(|r| r.triple.predicate == p.as_bytes()) {
            for c in present.iter().filter(|r| r.triple.predicate == q.as_bytes()) {
                if a.object_id == c.subject_id {
                    out.push(vec![
                        CanonicalObject::Node(a.triple.subject.clone()),
                        a.triple.object.clone(),
                        c.triple.object.clone(),
                    ]);
                }
            }
        }
        out
    }

    #[test]
    fn single_atom_predicate_resolved() {
        let layers = filter_stack();
        let lref = refs(&layers);
        let c = hash_pred("http://ex/p2"); // 4 rows
        let (rec, filt, result) = prove_read_single_atom(&lref, FilterPosition::Predicate, c).unwrap();
        let head_root = lref.last().unwrap().state.commitment_root;
        let resolved = verify_and_resolve_single_atom(
            &lref, head_root, FilterPosition::Predicate, c, &rec, &filt, &result.columns,
        )
        .unwrap();
        let ground = materialized_filter_resolved(&lref, FilterPosition::Predicate, c);
        assert!(bag_eq_rows(rows_of(&resolved), ground), "resolved filter must equal materialized");
        // Real IRIs, not hashes.
        assert!(resolved[0].iter().all(|o| matches!(o, CanonicalObject::Node(_))));
    }

    #[test]
    fn two_atom_join_resolved() {
        let layers = join_stack();
        let lref = refs(&layers);
        let (cp, cq) = (hash_pred("http://ex/p1"), hash_pred("http://ex/p2"));
        let (rec, join, result) = prove_read_two_atom_join(&lref, cp, cq).unwrap();
        let head_root = lref.last().unwrap().state.commitment_root;
        let resolved = verify_and_resolve_two_atom_join(
            &lref, head_root, cp, cq, &rec, &join, &result.columns,
        )
        .unwrap();
        let ground = materialized_join_resolved(&lref, "http://ex/p1", "http://ex/p2");
        assert_eq!(resolved.len(), 3);
        assert!(bag_eq_rows(rows_of(&resolved), ground), "resolved join must equal materialized");
    }

    /// A stack with typed string-VALUE objects, to show resolution yields `CanonicalObject::Value`.
    fn value_stack() -> Vec<Arc<LayerCommitment>> {
        let store = open_sync_memory_store();
        let b = store.create_base_layer().unwrap();
        b.add_value_triple(ValueTriple::new_string_value(&iri("a1"), &iri("p1"), "hello")).unwrap();
        b.add_value_triple(ValueTriple::new_string_value(&iri("a2"), &iri("p1"), "world")).unwrap();
        b.add_value_triple(ValueTriple::new_node(&iri("a3"), &iri("p1"), &iri("b3"))).unwrap();
        b.add_value_triple(ValueTriple::new_node(&iri("a4"), &iri("p1"), &iri("b4"))).unwrap();
        let base = b.commit().unwrap();
        let base_lc = base.proof_commitment().unwrap();
        let child = base.open_write().unwrap();
        child.remove_value_triple(ValueTriple::new_node(&iri("a3"), &iri("p1"), &iri("b3"))).unwrap();
        child.remove_value_triple(ValueTriple::new_node(&iri("a4"), &iri("p1"), &iri("b4"))).unwrap();
        child.add_value_triple(ValueTriple::new_string_value(&iri("a5"), &iri("p1"), "foo")).unwrap();
        child.add_value_triple(ValueTriple::new_node(&iri("a6"), &iri("p1"), &iri("b6"))).unwrap();
        let head = child.commit().unwrap();
        vec![base_lc, head.proof_commitment().unwrap()]
    }

    #[test]
    fn resolved_typed_value_objects() {
        let layers = value_stack();
        let lref = refs(&layers);
        let c = hash_pred("http://ex/p1");
        let (rec, filt, result) = prove_read_single_atom(&lref, FilterPosition::Predicate, c).unwrap();
        let head_root = lref.last().unwrap().state.commitment_root;
        let resolved = verify_and_resolve_single_atom(
            &lref, head_root, FilterPosition::Predicate, c, &rec, &filt, &result.columns,
        )
        .unwrap();
        // present = {a1 "hello", a2 "world", a5 "foo", a6 b6}: the O column resolves 3 Values + 1 Node.
        let ground = materialized_filter_resolved(&lref, FilterPosition::Predicate, c);
        assert!(bag_eq_rows(rows_of(&resolved), ground));
        let objects = &resolved[1]; // the O column
        let values = objects.iter().filter(|o| matches!(o, CanonicalObject::Value { .. })).count();
        let nodes = objects.iter().filter(|o| matches!(o, CanonicalObject::Node(_))).count();
        assert_eq!((values, nodes), (3, 1), "three typed literals and one node IRI resolved");
    }

    #[test]
    fn resolve_forged_binding_rejects() {
        // A prover who publishes an altered result value (a fabricated binding) fails the
        // re-commitment bind to the authenticated result commitment.
        let layers = filter_stack();
        let lref = refs(&layers);
        let c = hash_pred("http://ex/p2");
        let (rec, filt, result) = prove_read_single_atom(&lref, FilterPosition::Predicate, c).unwrap();
        let head_root = lref.last().unwrap().state.commitment_root;
        let mut forged = result.columns.clone();
        forged[0][0] = ark(123_456_789); // a fabricated subject binding
        assert!(verify_and_resolve_single_atom(
            &lref, head_root, FilterPosition::Predicate, c, &rec, &filt, &forged,
        )
        .is_err());
    }

    #[test]
    fn resolve_forged_join_binding_rejects() {
        let layers = join_stack();
        let lref = refs(&layers);
        let (cp, cq) = (hash_pred("http://ex/p1"), hash_pred("http://ex/p2"));
        let (rec, join, result) = prove_read_two_atom_join(&lref, cp, cq).unwrap();
        let head_root = lref.last().unwrap().state.commitment_root;
        let mut forged = result.columns.clone();
        forged[2][0] = ark(424_242); // a fabricated ?z binding
        assert!(verify_and_resolve_two_atom_join(
            &lref, head_root, cp, cq, &rec, &join, &forged,
        )
        .is_err());
    }

    // ============================================================
    // Wave 6 — general n-atom BGP over present.
    // ============================================================

    fn var() -> PresentTerm {
        PresentTerm::Variable
    }
    fn node_const(iri: &str) -> PresentTerm {
        PresentTerm::Constant(hash_node(iri))
    }
    fn atom(subject: PresentTerm, predicate: &str, object: PresentTerm) -> PresentBgpAtom {
        PresentBgpAtom {
            subject,
            predicate_hash: hash_pred(predicate),
            object,
            subject_object_equal: false,
        }
    }
    fn scalar_join(acc_key: usize, rel_key: usize) -> JoinStep {
        JoinStep { acc_key, rel_key, additional_keys: Vec::new() }
    }

    /// A base+delta stack whose present supports a 3-atom chain
    /// Triple(?x, p1, ?y) & Triple(?y, p2, ?z) & Triple(?z, p3, ?w):
    ///   present p1 = {x1->y1, x2->y1, x4->y1}   (base x3->y3 removed; delta adds x4->y1)
    ///   present p2 = {y1->z1}                    (base y1->z9 removed)
    ///   present p3 = {z1->w1, z1->w2, z1->w3}    (delta adds z1->w3)
    /// Join on ?y=y1 (3 x) and ?z=z1 (3 w) => 3 x-bindings x 3 w-bindings = 9 solutions.
    fn bgp_stack() -> Vec<Arc<LayerCommitment>> {
        let store = open_sync_memory_store();
        let b = store.create_base_layer().unwrap();
        for (s, p, o) in [
            ("x1", "p1", "y1"), ("x2", "p1", "y1"), ("y1", "p2", "z1"),
            ("z1", "p3", "w1"), ("z1", "p3", "w2"),
            ("x3", "p1", "y3"), ("y1", "p2", "z9"),
        ] {
            b.add_value_triple(ValueTriple::new_node(&iri(s), &iri(p), &iri(o))).unwrap();
        }
        let base = b.commit().unwrap();
        let base_lc = base.proof_commitment().unwrap();
        let child = base.open_write().unwrap();
        child.remove_value_triple(ValueTriple::new_node(&iri("x3"), &iri("p1"), &iri("y3"))).unwrap();
        child.remove_value_triple(ValueTriple::new_node(&iri("y1"), &iri("p2"), &iri("z9"))).unwrap();
        child.add_value_triple(ValueTriple::new_node(&iri("x4"), &iri("p1"), &iri("y1"))).unwrap();
        child.add_value_triple(ValueTriple::new_node(&iri("z1"), &iri("p3"), &iri("w3"))).unwrap();
        let head = child.commit().unwrap();
        vec![base_lc, head.proof_commitment().unwrap()]
    }

    /// Materialized 3-atom chain over present, resolved to CanonicalObjects [x, y, z, w].
    fn materialized_3chain_resolved(
        layers: &[&LayerCommitment], p1: &str, p2: &str, p3: &str,
    ) -> Vec<Vec<CanonicalObject>> {
        let present = reconcile_present(layers).unwrap();
        let mut out = Vec::new();
        for a in present.iter().filter(|r| r.triple.predicate == p1.as_bytes()) {
            for bb in present.iter().filter(|r| r.triple.predicate == p2.as_bytes()) {
                if a.object_id != bb.subject_id { continue; }
                for c in present.iter().filter(|r| r.triple.predicate == p3.as_bytes()) {
                    if bb.object_id != c.subject_id { continue; }
                    out.push(vec![
                        CanonicalObject::Node(a.triple.subject.clone()),
                        a.triple.object.clone(),
                        bb.triple.object.clone(),
                        c.triple.object.clone(),
                    ]);
                }
            }
        }
        out
    }

    #[test]
    fn three_atom_bgp_over_present() {
        let layers = bgp_stack();
        let lref = refs(&layers);
        let atoms = [
            atom(var(), "http://ex/p1", var()), // Triple(?x, p1, ?y)
            atom(var(), "http://ex/p2", var()), // Triple(?y, p2, ?z)
            atom(var(), "http://ex/p3", var()), // Triple(?z, p3, ?w)
        ];
        let joins = [scalar_join(1, 0), scalar_join(2, 0)]; // acc.y==r2.y ; acc.z==r3.z
        let (rec, bgp, result) = prove_read_bgp(&lref, &atoms, &joins).unwrap();
        let head_root = lref.last().unwrap().state.commitment_root;
        let n = verify_read_bgp(&lref, head_root, &atoms, &joins, &rec, &bgp).unwrap();
        assert_eq!(n, 9, "3-atom chain over the delta stack must have 9 solutions");
        assert_eq!(result.len, 9);
        let resolved = verify_and_resolve_bgp(
            &lref, head_root, &atoms, &joins, &rec, &bgp, &result.columns,
        )
        .unwrap();
        let ground =
            materialized_3chain_resolved(&lref, "http://ex/p1", "http://ex/p2", "http://ex/p3");
        assert_eq!(ground.len(), 9);
        assert!(bag_eq_rows(rows_of(&resolved), ground), "resolved 3-chain must equal materialized");
    }

    #[test]
    fn multi_constant_bgp_over_present() {
        // Triple(x1, p1, ?y) & Triple(?y, p2, ?z): subject constant on atom 0 pins ?y=y1,
        // then join p2 -> exactly (y1, z1). Result schema [y, z].
        let layers = bgp_stack();
        let lref = refs(&layers);
        let atoms = [
            atom(node_const("http://ex/x1"), "http://ex/p1", var()),
            atom(var(), "http://ex/p2", var()),
        ];
        let joins = [scalar_join(0, 0)]; // acc.y == r2.y
        let (rec, bgp, result) = prove_read_bgp(&lref, &atoms, &joins).unwrap();
        let head_root = lref.last().unwrap().state.commitment_root;
        let n = verify_read_bgp(&lref, head_root, &atoms, &joins, &rec, &bgp).unwrap();
        assert_eq!(n, 1, "subject-constant 2-atom join has exactly one solution");
        let resolved = verify_and_resolve_bgp(
            &lref, head_root, &atoms, &joins, &rec, &bgp, &result.columns,
        )
        .unwrap();
        assert_eq!(
            rows_of(&resolved),
            vec![vec![
                CanonicalObject::Node(iri("y1").into_bytes()),
                CanonicalObject::Node(iri("z1").into_bytes()),
            ]],
        );
    }

    #[test]
    fn single_atom_object_constant_bgp() {
        // Triple(?x, p1, y1): object constant, keep subject. present p1 objects=y1: x1,x2,x4.
        let layers = bgp_stack();
        let lref = refs(&layers);
        let atoms = [atom(var(), "http://ex/p1", node_const("http://ex/y1"))];
        let (rec, bgp, result) = prove_read_bgp(&lref, &atoms, &[]).unwrap();
        let head_root = lref.last().unwrap().state.commitment_root;
        let n = verify_read_bgp(&lref, head_root, &atoms, &[], &rec, &bgp).unwrap();
        assert_eq!(n, 3);
        let resolved = verify_and_resolve_bgp(
            &lref, head_root, &atoms, &[], &rec, &bgp, &result.columns,
        )
        .unwrap();
        let subjects: Vec<CanonicalObject> = resolved.into_iter().flatten().collect();
        let ground: Vec<CanonicalObject> = ["x1", "x2", "x4"]
            .iter()
            .map(|s| CanonicalObject::Node(iri(s).into_bytes()))
            .collect();
        assert!(bag_eq_rows(
            subjects.into_iter().map(|o| vec![o]).collect(),
            ground.into_iter().map(|o| vec![o]).collect(),
        ));
    }

    /// A stack whose present has self-loops (subject==object) and non-loops:
    /// present = {a1->a1, a2->a2, a3->b3, a5->a5, a6->b6}. Triple(?s, pself, ?s) => {a1,a2,a5}.
    fn self_stack() -> Vec<Arc<LayerCommitment>> {
        let store = open_sync_memory_store();
        let b = store.create_base_layer().unwrap();
        for (s, o) in [("a1", "a1"), ("a2", "a2"), ("a3", "b3"), ("a4", "a4"), ("a7", "a7")] {
            b.add_value_triple(ValueTriple::new_node(&iri(s), &iri("pself"), &iri(o))).unwrap();
        }
        let base = b.commit().unwrap();
        let base_lc = base.proof_commitment().unwrap();
        let child = base.open_write().unwrap();
        child.remove_value_triple(ValueTriple::new_node(&iri("a4"), &iri("pself"), &iri("a4"))).unwrap();
        child.remove_value_triple(ValueTriple::new_node(&iri("a7"), &iri("pself"), &iri("a7"))).unwrap();
        child.add_value_triple(ValueTriple::new_node(&iri("a5"), &iri("pself"), &iri("a5"))).unwrap();
        child.add_value_triple(ValueTriple::new_node(&iri("a6"), &iri("pself"), &iri("b6"))).unwrap();
        let head = child.commit().unwrap();
        vec![base_lc, head.proof_commitment().unwrap()]
    }

    #[test]
    fn repeated_variable_bgp_over_present() {
        let layers = self_stack();
        let lref = refs(&layers);
        let self_atom = PresentBgpAtom {
            subject: var(),
            predicate_hash: hash_pred("http://ex/pself"),
            object: var(),
            subject_object_equal: true,
        };
        let atoms = [self_atom];
        let (rec, bgp, result) = prove_read_bgp(&lref, &atoms, &[]).unwrap();
        let head_root = lref.last().unwrap().state.commitment_root;
        let n = verify_read_bgp(&lref, head_root, &atoms, &[], &rec, &bgp).unwrap();
        assert_eq!(n, 3, "three present self-loops match Triple(?s, pself, ?s)");
        let resolved = verify_and_resolve_bgp(
            &lref, head_root, &atoms, &[], &rec, &bgp, &result.columns,
        )
        .unwrap();
        let got: Vec<Vec<CanonicalObject>> = rows_of(&resolved);
        let ground: Vec<Vec<CanonicalObject>> = ["a1", "a2", "a5"]
            .iter()
            .map(|s| vec![CanonicalObject::Node(iri(s).into_bytes())])
            .collect();
        assert!(bag_eq_rows(got, ground), "self-join must equal materialized a1,a2,a5");
    }

    // --- Soundness negatives over the general BGP path ---

    #[test]
    fn bgp_fabricated_relation_commitment_rejects() {
        let layers = bgp_stack();
        let lref = refs(&layers);
        let atoms = [
            atom(var(), "http://ex/p1", var()),
            atom(var(), "http://ex/p2", var()),
            atom(var(), "http://ex/p3", var()),
        ];
        let joins = [scalar_join(1, 0), scalar_join(2, 0)];
        let (rec, mut bgp, _) = prove_read_bgp(&lref, &atoms, &joins).unwrap();
        let head_root = lref.last().unwrap().state.commitment_root;
        // Fabricate atom 0's first kept column: the compaction binding fails.
        bgp.atoms[0].com_kept[0] = commit_col(&vec![ark(4242); 1usize << rec.working_log]);
        assert!(verify_read_bgp(&lref, head_root, &atoms, &joins, &rec, &bgp).is_err());
    }

    #[test]
    fn bgp_dropped_row_rejects() {
        let layers = bgp_stack();
        let lref = refs(&layers);
        let atoms = [
            atom(var(), "http://ex/p1", var()),
            atom(var(), "http://ex/p2", var()),
            atom(var(), "http://ex/p3", var()),
        ];
        let joins = [scalar_join(1, 0), scalar_join(2, 0)];
        let (rec, bgp, _) = prove_read_bgp(&lref, &atoms, &joins).unwrap();
        let head_root = lref.last().unwrap().state.commitment_root;
        // Claim one fewer matching row in atom 0's compaction.
        let mut forged = bgp.clone();
        forged.atoms[0].m -= 1;
        assert!(verify_read_bgp(&lref, head_root, &atoms, &joins, &rec, &forged).is_err());
    }

    #[test]
    fn bgp_wrong_constant_rejects() {
        let layers = bgp_stack();
        let lref = refs(&layers);
        let atoms = [
            atom(var(), "http://ex/p1", var()),
            atom(var(), "http://ex/p2", var()),
        ];
        let joins = [scalar_join(1, 0)];
        let (rec, bgp, _) = prove_read_bgp(&lref, &atoms, &joins).unwrap();
        let head_root = lref.last().unwrap().state.commitment_root;
        // Verify against a different first predicate: the selection is p1-bound.
        let wrong = [
            atom(var(), "http://ex/pX", var()),
            atom(var(), "http://ex/p2", var()),
        ];
        assert!(verify_read_bgp(&lref, head_root, &wrong, &joins, &rec, &bgp).is_err());
    }

    #[test]
    fn bgp_tampered_present_id_rejects() {
        // Feeding a wrong combined-dict id for present (to fabricate a joining tuple)
        // fails the id -> hash authentication against the reconciled hashes.
        let layers = bgp_stack();
        let lref = refs(&layers);
        let atoms = [
            atom(var(), "http://ex/p1", var()),
            atom(var(), "http://ex/p2", var()),
        ];
        let joins = [scalar_join(1, 0)];
        let (rec, mut bgp, _) = prove_read_bgp(&lref, &atoms, &joins).unwrap();
        let head_root = lref.last().unwrap().state.commitment_root;
        bgp.auth_s.com_id = commit_col(&vec![ark(3); 1usize << rec.working_log]);
        assert!(verify_read_bgp(&lref, head_root, &atoms, &joins, &rec, &bgp).is_err());
    }

    #[test]
    fn bgp_forged_result_binding_rejects() {
        // A prover who publishes an altered result value fails the re-commitment bind.
        let layers = bgp_stack();
        let lref = refs(&layers);
        let atoms = [
            atom(var(), "http://ex/p1", var()),
            atom(var(), "http://ex/p2", var()),
            atom(var(), "http://ex/p3", var()),
        ];
        let joins = [scalar_join(1, 0), scalar_join(2, 0)];
        let (rec, bgp, result) = prove_read_bgp(&lref, &atoms, &joins).unwrap();
        let head_root = lref.last().unwrap().state.commitment_root;
        let mut forged = result.columns.clone();
        forged[3][0] = ark(987_654_321); // fabricated ?w binding
        assert!(verify_and_resolve_bgp(
            &lref, head_root, &atoms, &joins, &rec, &bgp, &forged,
        )
        .is_err());
    }

    #[test]
    fn bgp_non_satisfying_join_tuple_is_excluded() {
        // Ground-truth completeness+soundness: the reconciled answer equals the join over
        // the MATERIALIZED present exactly, so a Cartesian (non-satisfying) tuple such as
        // (x1, y1, z1, w) with a w whose z != z1 never appears. present p3 only has z1->w*,
        // and the base's non-joining rows (x3->y3, y1->z9) were removed, so any fabricated
        // tuple would inflate the count beyond the materialized 9.
        let layers = bgp_stack();
        let lref = refs(&layers);
        let atoms = [
            atom(var(), "http://ex/p1", var()),
            atom(var(), "http://ex/p2", var()),
            atom(var(), "http://ex/p3", var()),
        ];
        let joins = [scalar_join(1, 0), scalar_join(2, 0)];
        let (rec, bgp, result) = prove_read_bgp(&lref, &atoms, &joins).unwrap();
        let head_root = lref.last().unwrap().state.commitment_root;
        let n = verify_read_bgp(&lref, head_root, &atoms, &joins, &rec, &bgp).unwrap();
        let ground =
            materialized_3chain_resolved(&lref, "http://ex/p1", "http://ex/p2", "http://ex/p3");
        assert_eq!(n, ground.len());
        // Every delivered tuple satisfies both join predicates (y-link and z-link).
        let resolved = verify_and_resolve_bgp(
            &lref, head_root, &atoms, &joins, &rec, &bgp, &result.columns,
        )
        .unwrap();
        for row in rows_of(&resolved) {
            // y == z-atom's subject was enforced by the join; assert the tuple is one the
            // materialized set contains (no fabricated Cartesian tuple slipped through).
            assert!(ground.contains(&row));
        }
    }
}
