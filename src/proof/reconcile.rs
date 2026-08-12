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
    prove_sumcheck, verify_column_openings, verify_sumcheck, ClaimShape, ColumnOpening, Expr,
    Operand, StoreProverContext, SumcheckClaim, SumcheckProof, SumcheckShape, Term, Transcript254,
};
use super::bgp::{RelationCommit, RelationWitness};
use super::commitment::{LayerCommitment, VerifierCommitmentRef, VerifierLayer};
use super::dict::{hash_object, hash_term};
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

    let mut transcript = Transcript254::new(b"parture-delta-reconcile-envelope");
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
    let combined_nv = prove_combined_dict(context, &nv_branches, working_log, &mut transcript)?;
    let combined_pred = prove_combined_dict(context, &pred_branches, working_log, &mut transcript)?;

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
            working_log, &combined_nv, &mut transcript,
        )?;
        let o = prove_group_hash(
            context, records, |r| r.object_id, &desc.o_com, desc.col_log, desc.real_len,
            working_log, &combined_nv, &mut transcript,
        )?;
        let p = prove_group_hash(
            context, records, |r| r.predicate_id, &desc.p_com, desc.col_log, desc.real_len,
            working_log, &combined_pred, &mut transcript,
        )?;
        group_spo_data.push([s.1, p.1, o.1]);
        group_spo_com.push([s.0.lookup.com_v_hash, p.0.lookup.com_v_hash, o.0.lookup.com_v_hash]);
        groups.push(CommittedGroupProof { s: s.0, o: o.0, p: p.0 });
    }

    // present's canonical S/P/O columns at 2^K.
    let mut present_s = vec![f0(); k_size];
    let mut present_p = vec![f0(); k_size];
    let mut present_o = vec![f0(); k_size];
    let mut present_rows = Vec::with_capacity(present_len);
    for (i, record) in present.iter().enumerate() {
        let sh = subject_hash(record);
        let ph = predicate_hash(record);
        let oh = object_hash(record);
        present_s[i] = sh;
        present_p[i] = ph;
        present_o[i] = oh;
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
    let (link_sumcheck, r, _evals) = prove_sumcheck(&shape, &columns, &mut transcript)?;
    let link_opening = context.prove_column_openings(&columns, &r, &mut transcript)?;

    Ok((
        ReconcileProof {
            combined_nv: combined_nv.proof,
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
        ReconciledPresent { rows: present_rows },
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

    let mut transcript = Transcript254::new(b"parture-delta-reconcile-envelope");
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
    verify_combined_dict(&nv_branches, head_nv, working_log, &proof.combined_nv, &mut transcript)?;
    verify_combined_dict(&pred_branches, head_pred, working_log, &proof.combined_pred, &mut transcript)?;

    // Committed group descriptors (same canonical order the prover used).
    let column_commitments: Vec<&[ArkGT; 6]> = views.iter().map(|v| v.column_commitments).collect();
    let descs = group_descs(&column_commitments, &addition_counts, &removal_counts)?;
    if descs.len() != proof.groups.len() {
        return Err(invalid("reconcile: committed group count mismatch"));
    }

    // Verify each group's three canonical hash lookups.
    for (desc, gp) in descs.iter().zip(proof.groups.iter()) {
        verify_group_hash(&desc.s_com, desc.col_log, desc.real_len, working_log, &proof.combined_nv, &gp.s, &mut transcript)?;
        verify_group_hash(&desc.o_com, desc.col_log, desc.real_len, working_log, &proof.combined_nv, &gp.o, &mut transcript)?;
        verify_group_hash(&desc.p_com, desc.col_log, desc.real_len, working_log, &proof.combined_pred, &gp.p, &mut transcript)?;
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
    let (r, evals) = verify_sumcheck(&shape, &proof.link_sumcheck, &mut transcript)?;
    let mut commitments = Vec::with_capacity(4 * g);
    for i in 0..g {
        commitments.push(value_com[i][0]);
        commitments.push(value_com[i][1]);
        commitments.push(value_com[i][2]);
        commitments.push(proof.com_rec[i]);
    }
    verify_column_openings(&commitments, &r, &evals, &proof.link_opening, &mut transcript)?;

    Ok(present_len)
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
}
