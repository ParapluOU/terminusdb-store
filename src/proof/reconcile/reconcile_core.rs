//! Reconcile core: proving `present` IS the current triple set of the whole layer stack.
//!
//! This is the heart of delta-stack proving. The prover hands over a candidate `present`
//! set; this module proves the multiset equality `present ⊎ (⊎ removals) ≡ (⊎ additions)`
//! over the whole parent chain, in canonical hash space, via a two-sided masked LogUp link
//! against the committed per-layer delta columns. If it verifies, `present` really is the
//! reconciled set (and, under the store's add-absent/remove-present invariant, is
//! duplicate-free) — WITHOUT ever materializing a head view. It also carries the chain-state
//! validation (parent-root chaining + dictionary-count abutment), the reconcile proof
//! structs, and the per-group hash sub-proofs the link is built from.

use super::*;

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
    /// I4 no-re-mint: full-field distinctness of the two combined dict-hash columns (no canonical
    /// term re-minted at two ids across the chain), proved over the root-bound `Com(dict_hash)`
    /// columns — the committed-data replacement for the old clear-catalog guard.
    pub(crate) remint: RemintProof,
    pub working_log: usize,
    /// Committed groups (each layer's additions then removals, in chain order).
    pub groups: Vec<CommittedGroupProof>,
    /// present's freshly committed canonical S/P/O hash columns at 2^K.
    pub com_present_s: ArkGT,
    pub com_present_p: ArkGT,
    pub com_present_o: ArkGT,
    /// The multi-block LogUp-GKR fraction-sum-zero proof for the signed reconcile link
    /// `Σ_g sign_g · Σ_{row < len_g} 1/(alpha − rlc_β(S_g,P_g,O_g)) = 0` (committed reciprocals
    /// dropped: `G` blocks tiled into a `working_log + high_bits`-var tree, block selected by
    /// `eq_at_index`, all numerators PUBLIC signed prefixes).
    pub link_gkr: GkrFractionProof,
    /// Evaluations of the `3·G` committed link columns (`[S, P, O]` per group in link order,
    /// committed groups then present) at `r_low` (the low `working_log` bits of the GKR leaf point).
    pub link_col_evals: Vec<ArkFr>,
    pub link_opening: ColumnOpening,
    pub present_len: usize,
}

/// The prover's reconciled `present` relation in canonical HASH space (returned so a
/// caller/test can assert it against the materialized ground truth).
pub struct ReconciledPresent {
    /// Per present triple, `(hash_object(subject), hash_term(predicate), hash_object(object))`.
    pub rows: Vec<(ArkFr, ArkFr, ArkFr)>,
}

pub(super) fn subject_hash(record: &ProofRecord) -> ArkFr {
    hash_object(&CanonicalObject::Node(record.triple.subject.clone()))
}
pub(super) fn predicate_hash(record: &ProofRecord) -> ArkFr {
    hash_term(&record.triple.predicate)
}
pub(super) fn object_hash(record: &ProofRecord) -> ArkFr {
    hash_object(&record.triple.object)
}

/// Reconcile `present = (⊎ additions) − (⊎ removals)` by canonical triple, chain-wide
/// (pools every layer's additions, then cancels one addition per removal; rejects a
/// removal absent from the pool, i.e. a violated §3 invariant).
pub(super) fn reconcile_present(layers: &[&LayerCommitment]) -> io::Result<Vec<ProofRecord>> {
    use std::collections::HashMap;
    // Tally how many times each canonical triple is removed across the whole chain. The
    // previous O(additions × removals) linear-scan-and-`Vec::remove` cancelled one addition per
    // removal by taking the FIRST occurrence and shifting the tail; cancelling exactly
    // `count[key]` earliest occurrences of each key (walking additions in the SAME pooled order)
    // yields the byte-identical surviving set in the identical order, in O(additions + removals).
    let mut to_cancel: HashMap<&CanonicalTriple, usize> = HashMap::new();
    for layer in layers {
        for rem in &layer.removals {
            *to_cancel.entry(&rem.triple).or_insert(0) += 1;
        }
    }
    let mut present: Vec<ProofRecord> = Vec::new();
    for layer in layers {
        for add in &layer.additions {
            if let Some(remaining) = to_cancel.get_mut(&add.triple) {
                if *remaining > 0 {
                    *remaining -= 1;
                    continue;
                }
            }
            present.push(add.clone());
        }
    }
    // Any removal left uncancelled had no matching pooled addition (violated §3 invariant).
    if to_cancel.values().any(|&remaining| remaining > 0) {
        return Err(invalid(
            "reconcile: a removal is absent from the pooled additions (delta invariant violated)",
        ));
    }
    Ok(present)
}

// -- Structural chain validation (mirrored by prover and verifier) --

pub(super) fn validate_chain_states(states: &[&crate::proof::commitment::CommitmentState]) -> io::Result<()> {
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

// -- Group description shared by prover and verifier (canonical order) --

/// One committed group: its per-column commitments (subject, predicate, object), its
/// column domain log, real length, and its link sign (`+1` additions, `−1` removals).
pub(super) struct GroupDesc {
    s_com: ArkGT,
    p_com: ArkGT,
    o_com: ArkGT,
    col_log: usize,
    real_len: usize,
    pub(super) sign: ArkFr,
}

/// Build the committed group descriptors in canonical order: for each layer, its
/// additions group (if non-empty) then its removals group (if non-empty).
pub(super) fn group_descs(
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
/// Returned by `prove_reconcile_core` so a read query can be proved over `present` on the
/// SAME transcript — that shared transcript is what binds the BGP to the
/// reconciliation-authenticated relation.
pub struct ReconcileArtifacts {
    pub present_s: Vec<ArkFr>,
    pub present_p: Vec<ArkFr>,
    pub present_o: Vec<ArkFr>,
    pub com_present_s: ArkGT,
    pub com_present_p: ArkGT,
    pub com_present_o: ArkGT,
    /// present's subject/object/predicate COMBINED-DICT ids at 2^K (the small canonical id
    /// space, authenticated against the reconciliation hashes). Subject/object are joined on;
    /// the predicate id is used to form the bounded canonical-triple key for the
    /// duplicate-freeness assertion (Wave 7 hardening).
    pub present_id_s: Vec<ArkFr>,
    pub present_id_o: Vec<ArkFr>,
    pub present_id_p: Vec<ArkFr>,
    pub working_log: usize,
    pub present_len: usize,
    /// The combined node/value + predicate dicts at 2^K (for authenticating present's ids).
    pub(super) nv_dict: CombinedDictBuilt,
    pub(super) pred_dict: CombinedDictBuilt,
    pub(super) present_rows: Vec<(ArkFr, ArkFr, ArkFr)>,
}

pub(crate) fn prove_reconcile_core(
    context: &StoreProverContext<'_>,
    layers: &[&LayerCommitment],
    transcript: &mut Transcript254,
) -> io::Result<(ReconcileProof, ReconcileArtifacts)> {
    let states: Vec<&crate::proof::commitment::CommitmentState> = layers.iter().map(|l| &l.state).collect();
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

    // (1u) NO-RE-MINT (I4): prove each combined dict-hash column is all-distinct over COMMITTED,
    // root-bound data (the bag union of every layer's `Com(dict_hash)`), replacing the old
    // clear-catalog recompute guard. Read `combined_nv`/`combined_pred` by reference here — they are
    // moved into `ReconcileArtifacts` below. Runs before any group-hash step trusts term↔id.
    let remint = prove_remint(context, &combined_nv, &combined_pred, working_log, transcript)?;

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
    let mut present_id_p = vec![f0(); k_size];
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
        present_id_p[i] = ark(record.predicate_id);
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
    // beta/alpha drawn AFTER every link data-column (the S/P/O hash columns bound at `link-val`
    // above) commitment is bound — the logup_gkr caller contract (leaves fixed before any per-layer
    // GKR challenge). There is NO reciprocal commitment anymore, so this is the last binding before
    // the folds.
    let beta = transcript.challenge_scalar(b"pow/reconcile/link-beta");
    let alpha = transcript.challenge_scalar(b"pow/reconcile/link-alpha");

    // The signed multi-block fraction-sum-zero instance. Its 3G committed link columns
    // ([S, P, O] per group, already at 2^working_log) are zero-extended block-disjointly into the
    // 2^num_vars leaf vector: group `g`'s columns fill block `g`; padding blocks stay zero.
    let inst = reconcile_link_instance(alpha, beta, &lens, &signs, working_log)?;
    let leaf_count = inst.size();
    let mut columns: Vec<Vec<ArkFr>> = Vec::with_capacity(3 * g);
    for i in 0..g {
        columns.push(value_data[i][0].clone());
        columns.push(value_data[i][1].clone());
        columns.push(value_data[i][2].clone());
    }
    let mut extended: Vec<Vec<ArkFr>> = Vec::with_capacity(inst.num_cols);
    for (block, cols3) in columns.chunks(3).enumerate() {
        let base = block * k_size;
        for column in cols3 {
            let mut ext = vec![f0(); leaf_count];
            ext[base..base + k_size].copy_from_slice(column);
            extended.push(ext);
        }
    }
    let (link_gkr, r_leaf) = prove_fraction_sum_zero(&inst, &extended, transcript)?;

    // Single batched Dory opening of the 3G length-2^working_log link columns at r_low (the low
    // `working_log` bits of the GKR leaf point), bound to their commitments on this transcript.
    let r_low = &r_leaf[..working_log];
    let link_col_evals: Vec<ArkFr> = columns.iter().map(|c| mle_full_eval(c, r_low)).collect();
    let link_opening = context.prove_column_openings(&columns, r_low, transcript)?;

    Ok((
        ReconcileProof {
            combined_nv: combined_nv.proof.clone(),
            combined_pred: combined_pred.proof.clone(),
            remint,
            working_log,
            groups,
            com_present_s,
            com_present_p,
            com_present_o,
            link_gkr,
            link_col_evals,
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
            present_id_p,
            working_log,
            present_len,
            nv_dict: combined_nv,
            pred_dict: combined_pred,
            present_rows,
        },
    ))
}

/// Authenticate one id column (`select`) of `records` against a combined dict, at 2^K.
/// Returns the group hash proof and the authenticated hash column data.
#[allow(clippy::too_many_arguments)]
pub(super) fn prove_group_hash(
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

/// The PUBLIC multi-block fraction-sum-zero instance for the signed reconcile link, built
/// IDENTICALLY by prover and verifier from public counts/signs. `G = lens.len()` groups
/// (each layer's additions group `sign +1`, removals group `sign −1`, then present `sign −1`),
/// each with 3 committed canonical hash columns `[S, P, O]` at `2^working_log` per block. The
/// `G` blocks are padded to `G_pad = next_pow2(G)` and tiled into a `working_log + high_bits`-var
/// tree; the low `working_log` bits index within a block, the top `high_bits` select the block.
///
/// Column layout (`num_cols = G·3`): group `g`'s fold columns `[S_g, P_g, O_g]` at instance
/// indices `[g·3, g·3+3)`. Each column is block-placed (nonzero only in its own leaf block), so on
/// the leaves of block `g` the denominator `q_leaf = alpha − Σ_{block} Σ_{d<3} beta^d·col`
/// reduces to `alpha − (S_g + beta·P_g + beta²·O_g)` (= `alpha − rlc_β(S,P,O)`, the exact fold the
/// retired `reciprocal_claim` pinned) and to the constant `alpha` on padding blocks. The numerator
/// `p_leaf` is the SIGNED PUBLIC block indicator `sign_g·([i < base_g + len_g] − [i < base_g])`
/// (`base_g = g·2^working_log`) — no committed multiplicity (the reciprocal column is gone), so
/// there is NO `committed × public-Chi` mask (no GAP-1, no auxiliary Identity); within-block and
/// padding-block rows self-annihilate (`p = 0`).
pub(super) fn reconcile_link_instance(
    alpha: ArkFr,
    beta: ArkFr,
    lens: &[usize],
    signs: &[ArkFr],
    working_log: usize,
) -> io::Result<FractionInstance> {
    let groups = lens.len();
    let group_pad = groups
        .checked_next_power_of_two()
        .ok_or_else(|| invalid("reconcile: link group count overflow"))?;
    let high_bits = group_pad.trailing_zeros() as usize;
    let num_vars = working_log
        .checked_add(high_bits)
        .filter(|nv| *nv < usize::BITS as usize)
        .ok_or_else(|| invalid("reconcile: link leaf domain overflow"))?;
    let domain = 1usize << working_log;
    let arity = 3usize;
    let num_cols = groups * arity;
    let beta_powers = powers(beta, arity);

    // q_leaf = alpha - Σ_{block g} Σ_{d<3} beta^d·col_{g,d} (blocks disjoint).
    let mut q_terms = Vec::with_capacity(num_cols);
    for block in 0..groups {
        for (d, power) in beta_powers.iter().enumerate() {
            q_terms.push(LinearTerm {
                coeff: f0() - *power,
                src: Src::Column(block * arity + d),
            });
        }
    }

    // p_leaf: each group's first len_g rows count with sign_g; everything else (within-block
    // padding + padding blocks) 0. All prefix bounds are PUBLIC (root-authenticated counts).
    let mut p_terms = Vec::with_capacity(2 * groups);
    for (g, (&len_g, &sign_g)) in lens.iter().zip(signs.iter()).enumerate() {
        let base = g * domain;
        let hi = base + len_g.min(domain);
        p_terms.push(LinearTerm {
            coeff: sign_g,
            src: Src::Prefix(hi),
        });
        p_terms.push(LinearTerm {
            coeff: f0() - sign_g,
            src: Src::Prefix(base),
        });
    }

    Ok(FractionInstance {
        num_vars,
        num_cols,
        p_form: LinearForm {
            constant: f0(),
            terms: p_terms,
        },
        q_form: LinearForm {
            constant: alpha,
            terms: q_terms,
        },
    })
}

/// Verify an N-layer reconciliation. On success returns the authenticated reconciled
/// triple COUNT `M` (the single-atom count query answer over `present`).
pub fn verify_reconcile(
    layers: &[&LayerCommitment],
    trusted_head_root: ProofHash,
    proof: &ReconcileProof,
) -> io::Result<usize> {
    let mut transcript = Transcript254::new(b"parture-delta-reconcile-envelope");
    let views: Vec<VerifierCommitmentRef<'_>> = layers.iter().map(|l| l.verifier_view()).collect();
    let verified = verify_reconcile_core(&views, trusted_head_root, proof, &mut transcript)?;
    Ok(verified.present_len)
}

/// Verifier-side authenticated `present` handle (its column commitments at 2^K + counts),
/// returned by `verify_reconcile_core` so a read query can be verified over it on the SAME
/// transcript.
pub struct ReconcileVerified {
    pub com_present_s: ArkGT,
    pub com_present_p: ArkGT,
    pub com_present_o: ArkGT,
    pub working_log: usize,
    pub present_len: usize,
}

pub(crate) fn verify_reconcile_core(
    views: &[VerifierCommitmentRef<'_>],
    trusted_head_root: ProofHash,
    proof: &ReconcileProof,
    transcript: &mut Transcript254,
) -> io::Result<ReconcileVerified> {
    // (1) Each layer authenticated against ITS OWN commitment root.
    for view in views {
        view.validate()?;
    }
    // (2) Whole parent chain + trusted head anchor.
    let states: Vec<&crate::proof::commitment::CommitmentState> = views.iter().map(|v| v.state).collect();
    validate_chain_states(&states)?;
    // β re-anchor: the trusted head anchor is now the HEAD layer's `resolution_root` (binds
    // `commitment_root` AND its two dictionary Merkle roots directly). Every layer view was
    // `validate()`d above (Merkle roots recomputed from each root-bound catalog), so the head
    // view's resolution_root is fully authenticated. A fabricated head Merkle root cannot hash
    // to the trusted anchor.
    if views.last().unwrap().resolution_root() != trusted_head_root {
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
    for view in views {
        transcript.append_bytes(b"pow/reconcile/layer-root", &view.state.commitment_root);
    }

    // Combined dict branches (verifier side) — one per layer with a non-empty incremental dict.
    let mut nv_branches = Vec::new();
    let mut pred_branches = Vec::new();
    for view in views {
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

    // (1u) NO-RE-MINT (I4): verify each combined dict-hash column is all-distinct over the COMMITTED,
    // root-bound `Com(dict_hash)` columns just authenticated above (the bag union of every layer's
    // dict-hash rows) — no canonical term re-minted at two ids across the chain. This is the
    // committed-data replacement for the old clear-catalog `assert_combined_catalog_hash_unique`, and
    // fail-closes the entire delta/reconciled family (present BGP/filter/join/result, list/array walks)
    // before any id-space step trusts the term↔id bijection. Transcript position mirrors `prove_remint`.
    verify_remint(&proof.combined_nv, &proof.combined_pred, working_log, &proof.remint, transcript)?;

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

    // Verify the G-way signed multi-block LogUp-GKR link.
    let g = descs.len() + 1;
    if proof.link_col_evals.len() != 3 * g {
        return Err(invalid("reconcile: link column-eval count mismatch"));
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
    // Same binding order as the prover: all link data-column commitments, THEN beta/alpha.
    let beta = transcript.challenge_scalar(b"pow/reconcile/link-beta");
    let alpha = transcript.challenge_scalar(b"pow/reconcile/link-alpha");

    // Bind the public instance and verify the multi-block GKR tree -> leaf point + reduced claims.
    let inst = reconcile_link_instance(alpha, beta, &lens, &signs, working_log)?;
    if proof.link_gkr.num_vars() != inst.num_vars {
        return Err(invalid("reconcile: link fraction layer count mismatch"));
    }
    let (r_leaf, p_claim, q_claim) =
        bind_and_verify_fraction_tree(&inst, &proof.link_gkr, transcript)?;

    // Open the 3G length-2^working_log committed link columns at r_low, bound to their commitments
    // on this transcript.
    let r_low = &r_leaf[..working_log];
    let r_high = &r_leaf[working_log..];
    let mut commitments = Vec::with_capacity(3 * g);
    for i in 0..g {
        commitments.push(value_com[i][0]);
        commitments.push(value_com[i][1]);
        commitments.push(value_com[i][2]);
    }
    verify_column_openings(&commitments, r_low, &proof.link_col_evals, &proof.link_opening, transcript)?;

    // Reconstruct each block's leaf-MLE from its length-2^working_log opening via the PUBLIC block
    // selector eq_at_index(block, r_high) (the multilinear [block == g] indicator over the high
    // bits). Group g's columns are tensor-scaled into block g only, STRUCTURALLY forcing them to
    // zero in every other block (no cross-group leak), then the leaf tie checks the tree's claims.
    let mut leaf_evals = Vec::with_capacity(inst.num_cols);
    for block in 0..g {
        let selector = eq_at_index(block, r_high);
        for column in 0..3 {
            leaf_evals.push(selector * proof.link_col_evals[block * 3 + column]);
        }
    }
    check_leaf_tie(&inst, &leaf_evals, p_claim, q_claim, &r_leaf)?;

    Ok(ReconcileVerified {
        com_present_s: proof.com_present_s,
        com_present_p: proof.com_present_p,
        com_present_o: proof.com_present_o,
        working_log,
        present_len,
    })
}

pub(super) fn verify_group_hash(
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

