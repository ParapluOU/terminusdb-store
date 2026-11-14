//! The inductive checkpoint proof `C_j -> C_k` over the deltas `j+1..k`.
//!
//! # Transcript order
//!
//! Everything the link reads is absorbed before `beta`/`alpha` are drawn: the previous and the
//! next checkpoint roots (which bind both present commitments and every other component), the
//! trusted head root, every delta's `commitment_root` (which binds its addition/removal column
//! commitments, and the clear triples are re-committed against them before the transcript starts),
//! the four group lengths and the working domain, and the four present commitments explicitly.
//! The GKR tree then fixes the leaf point; the claimed evaluations of the committed present columns
//! are absorbed by the batched opener before its own challenge; the padding zero-check and the
//! sortedness gadget follow, each binding its commitments before its challenges.

use std::collections::BTreeSet;
use std::io;

use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::{Field, Group};
use dory_pcs::primitives::transcript::Transcript;

use super::fixed_commit::{append_segmented, DICT_SEGMENT_LEN};
use super::genesis::{node_values_by_id, predicates_by_id};
use super::smt::{verify_delete, verify_insert, SmtTerminal, NS_NV_INDEX, NS_PRED_INDEX, NS_TRIPLE_INDEX};
use super::state::{
    append_dictionary, nv_dict_segment, nv_term_leaf, pred_term_leaf, present_columns,
    present_domain_log, triple_key, CheckpointArtifact, CheckpointState, TRIPLE_PRESENT,
};
use super::term_tree::{Accumulator, NS_NODE_VALUE, NS_PREDICATE};
use super::{commit_layer_id_column, CheckpointProof, DeltaUpdateProof, IdTriple, PaddingProof};
use crate::proof::argument::{
    mle_full_eval, prove_sumcheck, verify_column_openings, verify_sumcheck, ClaimShape, Expr,
    Operand, StoreProverContext, SumcheckClaim, SumcheckShape, Term, Transcript254,
};
use crate::proof::commitment::{
    node_value_sort_key, predicate_sort_key, CommitmentState, DictFacts, LayerCommitment,
    VerifierCommitmentRef, VerifierLayer,
};
use crate::proof::logup_gkr::{
    bind_and_verify_fraction_tree, check_leaf_tie, eq_at_index, prove_fraction_sum_zero,
};
use crate::proof::reconcile::reconcile_link_instance;
use crate::proof::sort::{prove_strictly_sorted_bits_with_context, verify_strictly_sorted_bits};
use crate::proof::{invalid, CanonicalObject, ProofHash, ProofRecord};

const TRANSCRIPT_LABEL: &[u8] = b"parture-checkpoint-proof";
const CHECKPOINT_PROOF_DOMAIN: &[u8] = b"parture/proof-of-woql/checkpoint/v1";

/// Test-only forgeries a malicious prover could attempt; each must be rejected by the verifier.
/// Production callers always pass [`Forgery::None`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) enum Forgery {
    None,
    /// Drop present row `i` of `C_k`.
    DropPresent(usize),
    /// Add a triple to `C_k`.
    ExtraPresent(IdTriple),
    /// Replace present row `i` with another triple (count unchanged).
    ReplacePresent(usize, IdTriple),
    /// Overwrite row `i + 1` with row `i` (a duplicate, count unchanged).
    DuplicatePresent(usize),
    /// Swap rows `i` and `i + 1` (same multiset, unsorted).
    SwapPresent(usize),
    /// A nonzero padding row in `C_k`'s columns.
    NonZeroPadding,
    /// The first new node/value term's hash entry is off by one.
    NvDictValue,
    /// The new node/value segment is appended one row too far.
    NvDictOffset,
    /// An extra node/value entry is appended after the real ones.
    NvDictExtra,
    /// The first new node/value term is inserted into the term index at a wrong key.
    NvIndexWrongKey,
    /// A new term that already has an id is "proven absent" (a re-mint): the prover turns its
    /// membership path into a fake non-membership one.
    ForgeAbsence,
    /// The first removal's triple stays in the triple index.
    TripleIndexSkipRemoval,
    /// An addition of an already-present triple is accepted (a delta-invariant violation).
    AllowReAdd,
    /// The float64 NaN fact is flipped.
    FlipNanFact,
}

/// Prove `C_k` from `prev = C_j` and the deltas `j+1..k` (ordered, `deltas[0]` the child of
/// `C_j`'s head).
pub fn prove_checkpoint(
    prev: &CheckpointState,
    deltas: &[&LayerCommitment],
) -> io::Result<(CheckpointState, CheckpointProof)> {
    prove_checkpoint_with_context(&StoreProverContext::cpu(), prev, deltas)
}

/// [`prove_checkpoint`] with explicit prover arithmetic.
pub fn prove_checkpoint_with_context(
    context: &StoreProverContext<'_>,
    prev: &CheckpointState,
    deltas: &[&LayerCommitment],
) -> io::Result<(CheckpointState, CheckpointProof)> {
    prove_checkpoint_inner(context, prev, deltas, &Forgery::None)
}

/// The chain from `C_j`'s head through every delta: parent roots chain and dictionary counts abut.
fn check_delta_chain(prev: &CheckpointArtifact, states: &[&CommitmentState]) -> io::Result<()> {
    if states.is_empty() {
        return Err(invalid("a checkpoint proof covers at least one delta"));
    }
    let mut parent_root = prev.head_commitment_root;
    let mut nv = prev.node_value_count;
    let mut pred = prev.predicate_count;
    for state in states {
        if state.parent_state_root != Some(parent_root) {
            return Err(invalid(
                "checkpoint deltas do not chain from the previous head (spliced, reordered, skipped or foreign layer)",
            ));
        }
        if state.parent_node_value_count != nv || state.parent_predicate_count != pred {
            return Err(invalid("checkpoint delta dictionary counts do not abut"));
        }
        parent_root = state.commitment_root;
        nv = state.node_value_count;
        pred = state.predicate_count;
    }
    Ok(())
}

/// A delta's new terms in id order — and the check that the store numbered them in sorted order
/// (`parent_count + 1 ..` ascending in canonical term order: nodes before values, each sorted),
/// which the write-transition proof relies on to recompute a candidate head's ids.
pub(super) fn new_terms(state: &CommitmentState, nv_catalog: &[ProofRecord], pred_catalog: &[ProofRecord])
    -> io::Result<(Vec<CanonicalObject>, Vec<Vec<u8>>)> {
    let nv = node_values_by_id(nv_catalog, state.parent_node_value_count + 1, state.node_value_count)?;
    let pred = predicates_by_id(pred_catalog, state.parent_predicate_count + 1, state.predicate_count)?;
    if nv.windows(2).any(|w| w[0] >= w[1]) || pred.windows(2).any(|w| w[0] >= w[1]) {
        return Err(invalid("a delta's new dictionary ids are not assigned in sorted term order"));
    }
    Ok((nv, pred))
}

fn id_triples(records: &[ProofRecord]) -> Vec<IdTriple> {
    records.iter().map(|r| [r.subject_id, r.predicate_id, r.object_id]).collect()
}

/// `log2` of the padded domain of a group of `len` rows (`>= 1`).
fn group_log(len: usize) -> usize {
    len.next_power_of_two().max(2).trailing_zeros() as usize
}

fn working_log(n_prev: u64, additions: usize, removals: usize, n_next: u64) -> io::Result<usize> {
    Ok(present_domain_log(n_prev)?
        .max(present_domain_log(n_next)?)
        .max(group_log(additions))
        .max(group_log(removals)))
}

fn signs() -> [ArkFr; 4] {
    let minus = ArkFr::zero() - ArkFr::one();
    [ArkFr::one(), ArkFr::one(), minus, minus]
}

#[allow(clippy::too_many_arguments)]
fn bind_statement(
    transcript: &mut Transcript254,
    prev: &CheckpointArtifact,
    next: &CheckpointArtifact,
    head_root: &ProofHash,
    delta_roots: &[ProofHash],
    lens: &[usize; 4],
    working_log: usize,
) {
    transcript.append_bytes(b"pow/checkpoint/domain", CHECKPOINT_PROOF_DOMAIN);
    transcript.append_bytes(b"pow/checkpoint/prev-root", &prev.checkpoint_root);
    transcript.append_bytes(b"pow/checkpoint/next-root", &next.checkpoint_root);
    transcript.append_bytes(b"pow/checkpoint/head-root", head_root);
    transcript.append_bytes(b"pow/checkpoint/delta-count", &(delta_roots.len() as u64).to_le_bytes());
    for root in delta_roots {
        transcript.append_bytes(b"pow/checkpoint/delta-root", root);
    }
    for len in lens {
        transcript.append_bytes(b"pow/checkpoint/group-len", &(*len as u64).to_le_bytes());
    }
    transcript.append_bytes(b"pow/checkpoint/working-log", &(working_log as u64).to_le_bytes());
    for commitment in prev.present_commitments.iter().chain(next.present_commitments.iter()) {
        transcript.append_group(b"pow/checkpoint/present-com", commitment);
    }
}

/// `2^k` as a field element.
fn pow2(k: usize) -> ArkFr {
    (0..k).fold(ArkFr::one(), |acc, _| acc + acc)
}

/// Bit widths of the packed sort key: subject/object ids are `<= nv_count`, predicates
/// `<= pred_count`.
fn sort_widths(nv_count: u64, pred_count: u64) -> (usize, usize) {
    ((64 - nv_count.leading_zeros()) as usize, (64 - pred_count.leading_zeros()) as usize)
}

/// `Com(S·2^(bp+bn) + P·2^bn + O)` from the three column commitments.
fn packed_commitment(commitments: &[ArkGT; 3], bn: usize, bp: usize) -> ArkGT {
    commitments[0]
        .scale(&pow2(bp + bn))
        .add(&commitments[1].scale(&pow2(bn)))
        .add(&commitments[2])
}

fn packed_column(columns: &[Vec<ArkFr>; 3], bn: usize, bp: usize) -> Vec<ArkFr> {
    let ws = pow2(bp + bn);
    let wp = pow2(bn);
    (0..columns[0].len()).map(|i| columns[0][i] * ws + columns[1][i] * wp + columns[2][i]).collect()
}

/// The padding zero-check shape: `(1 - chi_n)·col = 0` for each of the three columns.
fn padding_shape(num_vars: usize, n: usize) -> SumcheckShape {
    let minus = ArkFr::zero() - ArkFr::one();
    SumcheckShape {
        num_vars,
        num_columns: 3,
        chi_bounds: vec![n],
        claims: (0..3)
            .map(|c| SumcheckClaim {
                shape: ClaimShape::Identity,
                expr: Expr {
                    terms: vec![
                        Term { coeff: ArkFr::one(), operands: vec![Operand::Column(c)] },
                        Term { coeff: minus, operands: vec![Operand::Chi(0), Operand::Column(c)] },
                    ],
                },
                target: ArkFr::zero(),
            })
            .collect(),
    }
}

/// `MLE(column zero-padded to 2^point.len())(point)` for a clear id column.
fn clear_column_eval(ids: impl Iterator<Item = u64>, point: &[ArkFr]) -> ArkFr {
    let mut acc = ArkFr::zero();
    for (i, id) in ids.enumerate() {
        if id == 0 {
            continue;
        }
        let mut weight = ArkFr::one();
        for (bit, r) in point.iter().enumerate() {
            weight = weight * if (i >> bit) & 1 == 1 { *r } else { ArkFr::one() - *r };
        }
        acc = acc + weight * ArkFr::from_u64(id);
    }
    acc
}

/// `Π_{b >= low} (1 - r_b)`: the zero-extension factor of a column committed at `2^low`.
fn extension_factor(point: &[ArkFr], low: usize) -> ArkFr {
    point[low..].iter().fold(ArkFr::one(), |acc, r| acc * (ArkFr::one() - *r))
}

pub(crate) fn prove_checkpoint_inner(
    context: &StoreProverContext<'_>,
    prev: &CheckpointState,
    deltas: &[&LayerCommitment],
    forgery: &Forgery,
) -> io::Result<(CheckpointState, CheckpointProof)> {
    let states: Vec<&CommitmentState> = deltas.iter().map(|d| &d.state).collect();
    check_delta_chain(&prev.artifact, &states)?;
    let head = deltas.last().unwrap();

    // Dictionary: the deltas' new terms in id order.
    let mut new_nv = Vec::new();
    let mut new_pred = Vec::new();
    for delta in deltas {
        let (nv, pred) = new_terms(&delta.state, &delta.node_value_catalog, &delta.predicate_catalog)?;
        new_nv.extend(nv);
        new_pred.extend(pred);
    }

    // Present relation + triple index, delta by delta (removals, then additions).
    let mut present: BTreeSet<IdTriple> = prev.present.iter().copied().collect();
    let mut triple_index = prev.triple_index.clone();
    let mut delta_proofs = Vec::with_capacity(deltas.len());
    let mut all_additions = Vec::new();
    let mut all_removals = Vec::new();
    let mut skipped_removal = false;
    for delta in deltas {
        let additions = id_triples(&delta.additions);
        let removals = id_triples(&delta.removals);
        let mut removal_paths = Vec::with_capacity(removals.len());
        for triple in &removals {
            let key = triple_key(triple);
            removal_paths.push(triple_index.delete_proof(&key));
            if *forgery == Forgery::TripleIndexSkipRemoval && !skipped_removal {
                skipped_removal = true;
            } else {
                triple_index.delete(&key)?;
            }
            if !present.remove(triple) {
                return Err(invalid("checkpoint delta removes an absent triple"));
            }
        }
        let mut addition_paths = Vec::with_capacity(additions.len());
        for triple in &additions {
            let key = triple_key(triple);
            addition_paths.push(triple_index.path(&key));
            if present.contains(triple) {
                if *forgery == Forgery::AllowReAdd {
                    continue;
                }
                return Err(invalid("checkpoint delta adds a present triple"));
            }
            triple_index.insert(key, TRIPLE_PRESENT)?;
            present.insert(*triple);
        }
        all_additions.extend_from_slice(&additions);
        all_removals.extend_from_slice(&removals);
        delta_proofs.push(DeltaUpdateProof { additions, removals, removal_paths, addition_paths });
    }
    let mut present: Vec<IdTriple> = present.into_iter().collect();
    match forgery {
        Forgery::DropPresent(i) => {
            present.remove(*i);
        }
        Forgery::ExtraPresent(t) => {
            present.push(*t);
            present.sort();
        }
        Forgery::ReplacePresent(i, t) => {
            present[*i] = *t;
            present.sort();
        }
        Forgery::DuplicatePresent(i) => present[*i + 1] = present[*i],
        Forgery::SwapPresent(i) => present.swap(*i, *i + 1),
        _ => {}
    }

    // Term trees.
    let nv_tree_frontier = prev.nv_tree.frontier();
    let pred_tree_frontier = prev.pred_tree.frontier();
    let mut nv_tree = prev.nv_tree.clone();
    let mut pred_tree = prev.pred_tree.clone();
    let mut nv_terms = prev.nv_terms.clone();
    let mut pred_terms = prev.pred_terms.clone();
    for object in &new_nv {
        nv_terms.push(object.clone());
        nv_tree.push(nv_term_leaf(nv_terms.len() as u64, object))?;
    }
    for predicate in &new_pred {
        pred_terms.push(predicate.clone());
        pred_tree.push(pred_term_leaf(pred_terms.len() as u64, predicate))?;
    }

    // Term indexes.
    let mut nv_index = prev.nv_index.clone();
    let mut nv_index_paths = Vec::with_capacity(new_nv.len());
    for (i, object) in new_nv.iter().enumerate() {
        let id = prev.artifact.node_value_count + i as u64 + 1;
        let mut key = node_value_sort_key(object);
        if *forgery == Forgery::NvIndexWrongKey && i == 0 {
            key[0] ^= 0x80;
        }
        let mut path = nv_index.path(&key);
        if matches!(&path.terminal, SmtTerminal::Leaf { key: found, .. } if *found == key) {
            if *forgery != Forgery::ForgeAbsence {
                return Err(invalid("checkpoint delta re-mints a node/value term"));
            }
            path.terminal = SmtTerminal::Empty;
            nv_index_paths.push(path);
            continue;
        }
        nv_index_paths.push(path);
        nv_index.insert(key, id)?;
    }
    let mut pred_index = prev.pred_index.clone();
    let mut pred_index_paths = Vec::with_capacity(new_pred.len());
    for (i, predicate) in new_pred.iter().enumerate() {
        let id = prev.artifact.predicate_count + i as u64 + 1;
        let key = predicate_sort_key(predicate);
        pred_index_paths.push(pred_index.path(&key));
        pred_index.insert(key, id)?;
    }

    // Dictionary commitments (homomorphic append) and facts.
    let (mut nv_dict_segments, pred_dict_segments) =
        append_dictionary(&prev.artifact, &new_nv, &new_pred)?;
    if matches!(forgery, Forgery::NvDictValue | Forgery::NvDictOffset | Forgery::NvDictExtra) {
        let mut segment = nv_dict_segment(&new_nv)?;
        let mut offset = prev.artifact.node_value_count as usize;
        match forgery {
            Forgery::NvDictValue => segment[0][0] = segment[0][0] + ArkFr::one(),
            Forgery::NvDictOffset => offset += 1,
            _ => {
                for column in segment.iter_mut() {
                    column.push(column[0]);
                }
            }
        }
        let mut columns: [Vec<ArkGT>; 3] = Default::default();
        for c in 0..3 {
            let old: Vec<ArkGT> = prev.artifact.nv_dict_segments.iter().map(|s| s[c]).collect();
            // The forged rows may overrun the honest length by one: commit them past it.
            let mut old = old;
            if old.len() < super::fixed_commit::segment_count(offset as u64, DICT_SEGMENT_LEN) {
                old.push(<ArkGT as Group>::identity());
            }
            columns[c] = append_segmented(&old, offset, &segment[c], DICT_SEGMENT_LEN)?;
        }
        nv_dict_segments = (0..columns[0].len()).map(|k| [columns[0][k], columns[1][k], columns[2][k]]).collect();
    }
    let mut facts = prev.artifact.facts.union(DictFacts::from_objects(new_nv.iter()));
    if *forgery == Forgery::FlipNanFact {
        facts.float64_nan = !facts.float64_nan;
    }

    // Present columns and their commitments.
    let mut next_cols = present_columns(&present)?;
    if *forgery == Forgery::NonZeroPadding {
        let last = next_cols[0].len() - 1;
        if last < present.len() {
            return Err(invalid("forgery needs a padding row"));
        }
        next_cols[0][last] = ArkFr::one();
    }
    let present_commitments = if present.is_empty() {
        [<ArkGT as Group>::identity(); 3]
    } else {
        [
            context.commit_column(&next_cols[0])?,
            context.commit_column(&next_cols[1])?,
            context.commit_column(&next_cols[2])?,
        ]
    };

    let mut artifact = CheckpointArtifact {
        head_root: head.resolution_root(),
        head_commitment_root: head.state.commitment_root,
        prev_checkpoint_root: Some(prev.artifact.checkpoint_root),
        present_count: present.len() as u64,
        node_value_count: head.state.node_value_count,
        predicate_count: head.state.predicate_count,
        present_commitments,
        nv_dict_segments,
        pred_dict_segments,
        nv_term_tree_root: nv_tree.root(),
        pred_term_tree_root: pred_tree.root(),
        nv_term_index_root: nv_index.root(),
        pred_term_index_root: pred_index.root(),
        triple_index_root: triple_index.root(),
        facts,
        checkpoint_root: [0; 32],
    };
    artifact.checkpoint_root = artifact.compute_root()?;

    // ---- The present-relation argument ----
    let n_prev = prev.present.len();
    let n_next = present.len();
    let lens = [n_prev, all_additions.len(), all_removals.len(), n_next];
    let log = working_log(n_prev as u64, lens[1], lens[2], n_next as u64)?;
    let size = 1usize << log;
    let delta_roots: Vec<ProofHash> = deltas.iter().map(|d| d.state.commitment_root).collect();
    let mut transcript = Transcript254::new(TRANSCRIPT_LABEL);
    bind_statement(&mut transcript, &prev.artifact, &artifact, &artifact.head_root, &delta_roots, &lens, log);
    let beta = transcript.challenge_scalar(b"pow/checkpoint/link-beta");
    let alpha = transcript.challenge_scalar(b"pow/checkpoint/link-alpha");
    let inst = reconcile_link_instance(alpha, beta, &lens, &signs(), log)?;

    let prev_cols = present_columns(&prev.present)?;
    let clear_cols = |triples: &[IdTriple]| -> [Vec<ArkFr>; 3] {
        let mut cols = [vec![ArkFr::zero(); size], vec![ArkFr::zero(); size], vec![ArkFr::zero(); size]];
        for (row, t) in triples.iter().enumerate() {
            for c in 0..3 {
                cols[c][row] = ArkFr::from_u64(t[c]);
            }
        }
        cols
    };
    let blocks: [[Vec<ArkFr>; 3]; 4] = [
        prev_cols.clone().map(|c| { let mut c = c; c.resize(size, ArkFr::zero()); c }),
        clear_cols(&all_additions),
        clear_cols(&all_removals),
        next_cols.clone().map(|c| { let mut c = c; c.resize(size, ArkFr::zero()); c }),
    ];
    let leaf_count = inst.size();
    let mut extended = Vec::with_capacity(12);
    for (block, cols) in blocks.iter().enumerate() {
        for column in cols {
            let mut ext = vec![ArkFr::zero(); leaf_count];
            ext[block * size..(block + 1) * size].copy_from_slice(column);
            extended.push(ext);
        }
    }
    let (link, r_leaf) = prove_fraction_sum_zero(&inst, &extended, &mut transcript)?;
    let r_low = &r_leaf[..log];

    let open = |cols: &[Vec<ArkFr>; 3], n: usize, transcript: &mut Transcript254| -> io::Result<(Vec<ArkFr>, Option<_>)> {
        if n == 0 {
            return Ok((Vec::new(), None));
        }
        let own = &r_low[..cols[0].len().trailing_zeros() as usize];
        let evals = cols.iter().map(|c| mle_full_eval(c, own)).collect();
        let opening = context.prove_column_openings(cols, own, transcript)?;
        Ok((evals, Some(opening)))
    };
    let (prev_present_evals, prev_present_opening) = open(&prev_cols, n_prev, &mut transcript)?;
    let (next_present_evals, next_present_opening) = open(&next_cols, n_next, &mut transcript)?;

    let next_log = next_cols[0].len().trailing_zeros() as usize;
    let padding = if n_next > 0 {
        let shape = padding_shape(next_log, n_next);
        let columns = next_cols.to_vec();
        let (sumcheck, r, _) = prove_sumcheck(&shape, &columns, &mut transcript)?;
        let opening = context.prove_column_openings(&columns, &r, &mut transcript)?;
        Some(PaddingProof { sumcheck, opening })
    } else {
        None
    };
    let sorted = if n_next >= 2 {
        let (bn, bp) = sort_widths(artifact.node_value_count, artifact.predicate_count);
        let packed = packed_column(&next_cols, bn, bp);
        let com = packed_commitment(&artifact.present_commitments, bn, bp);
        Some(prove_strictly_sorted_bits_with_context(context, &packed, n_next, 2 * bn + bp, &com, &mut transcript)?)
    } else {
        None
    };

    let proof = CheckpointProof {
        deltas: delta_proofs,
        nv_tree_frontier,
        pred_tree_frontier,
        nv_index_paths,
        pred_index_paths,
        working_log: log,
        link,
        prev_present_evals,
        prev_present_opening,
        next_present_evals,
        next_present_opening,
        padding,
        sorted,
    };
    let state = CheckpointState {
        artifact,
        present,
        nv_terms,
        pred_terms,
        nv_tree,
        pred_tree,
        nv_index,
        pred_index,
        triple_index,
    };
    Ok((state, proof))
}

/// Verify `next = C_k` against `prev = C_j`, the deltas' TERM-BEARING verifier views `j+1..k`
/// (in chain order) and the trusted head root `X_k`.
///
/// The caller vouches for `prev` (its `checkpoint_root` is the one the chain recorded); everything
/// else is checked here. The work is `O(delta)` MSM and `O(delta · log n)` hashing plus the
/// succinct GKR / sumcheck / Dory verification — never the history before `C_j`.
pub fn verify_checkpoint(
    prev: &CheckpointArtifact,
    deltas: &[VerifierCommitmentRef<'_>],
    trusted_head_root: ProofHash,
    next: &CheckpointArtifact,
    proof: &CheckpointProof,
) -> io::Result<()> {
    prev.validate()?;
    next.validate()?;
    if next.prev_checkpoint_root != Some(prev.checkpoint_root) {
        return Err(invalid("checkpoint is bound to another previous checkpoint"));
    }
    if next.head_root != trusted_head_root {
        return Err(invalid("checkpoint is bound to another head"));
    }

    // The deltas: authentic, term-bearing, chained from C_j's head to the trusted head.
    for view in deltas {
        view.require_terms_present("checkpoint delta")?;
        view.validate()?;
    }
    let states: Vec<&CommitmentState> = deltas.iter().map(|v| v.state).collect();
    check_delta_chain(prev, &states)?;
    let head = deltas.last().unwrap();
    if head.resolution_root() != trusted_head_root {
        return Err(invalid("checkpoint deltas do not end at the trusted head"));
    }
    if next.head_commitment_root != head.state.commitment_root
        || next.node_value_count != head.state.node_value_count
        || next.predicate_count != head.state.predicate_count
    {
        return Err(invalid("checkpoint head summary disagrees with the head layer"));
    }

    // The deltas' clear triples, re-committed against their root-bound column commitments.
    if proof.deltas.len() != deltas.len() {
        return Err(invalid("checkpoint proof covers another number of deltas"));
    }
    let mut all_additions = Vec::new();
    let mut all_removals = Vec::new();
    for (view, update) in deltas.iter().zip(&proof.deltas) {
        let state = view.state;
        if update.additions.len() as u64 != state.addition_count
            || update.removals.len() as u64 != state.removal_count
            || update.removal_paths.len() != update.removals.len()
            || update.addition_paths.len() != update.additions.len()
        {
            return Err(invalid("checkpoint delta records disagree with the delta's counts"));
        }
        for table in [&update.additions, &update.removals] {
            if table.windows(2).any(|w| w[0] >= w[1]) {
                return Err(invalid("checkpoint delta records are not strictly SPO-sorted"));
            }
            if table.iter().any(|t| {
                t[0] == 0 || t[1] == 0 || t[2] == 0
                    || t[0] > state.node_value_count
                    || t[2] > state.node_value_count
                    || t[1] > state.predicate_count
            }) {
                return Err(invalid("checkpoint delta record id lies outside the dictionary"));
            }
        }
        let removal_set: BTreeSet<&IdTriple> = update.removals.iter().collect();
        if update.additions.iter().any(|t| removal_set.contains(t)) {
            return Err(invalid("checkpoint delta adds and removes the same triple"));
        }
        let column = |table: &[IdTriple], c: usize| -> Vec<u64> { table.iter().map(|t| t[c]).collect() };
        for c in 0..3 {
            if commit_layer_id_column(&column(&update.additions, c))? != view.column_commitments[c]
                || commit_layer_id_column(&column(&update.removals, c))? != view.column_commitments[3 + c]
            {
                return Err(invalid("checkpoint delta records are not the delta's committed records"));
            }
        }
        all_additions.extend_from_slice(&update.additions);
        all_removals.extend_from_slice(&update.removals);
    }

    // Triple index: removals delete members, additions insert non-members, delta by delta.
    let mut triple_root = prev.triple_index_root;
    for update in &proof.deltas {
        for (triple, path) in update.removals.iter().zip(&update.removal_paths) {
            triple_root = verify_delete(NS_TRIPLE_INDEX, &triple_root, &triple_key(triple), TRIPLE_PRESENT, path)?;
        }
        for (triple, path) in update.additions.iter().zip(&update.addition_paths) {
            triple_root = verify_insert(NS_TRIPLE_INDEX, &triple_root, &triple_key(triple), TRIPLE_PRESENT, path)?;
        }
    }
    if triple_root != next.triple_index_root {
        return Err(invalid("checkpoint triple index does not follow from the deltas"));
    }

    // Dictionary append.
    let mut new_nv = Vec::new();
    let mut new_pred = Vec::new();
    for view in deltas {
        let (nv, pred) = new_terms(view.state, view.node_value_catalog, view.predicate_catalog)?;
        new_nv.extend(nv);
        new_pred.extend(pred);
    }
    let (nv_dict, pred_dict) = append_dictionary(prev, &new_nv, &new_pred)?;
    if nv_dict != next.nv_dict_segments || pred_dict != next.pred_dict_segments {
        return Err(invalid("checkpoint dictionary append does not commit the deltas' new terms"));
    }
    let mut nv_tree = Accumulator::from_frontier(
        NS_NODE_VALUE, prev.node_value_count, &proof.nv_tree_frontier, &prev.nv_term_tree_root,
    )?;
    let mut pred_tree = Accumulator::from_frontier(
        NS_PREDICATE, prev.predicate_count, &proof.pred_tree_frontier, &prev.pred_term_tree_root,
    )?;
    if proof.nv_index_paths.len() != new_nv.len() || proof.pred_index_paths.len() != new_pred.len() {
        return Err(invalid("checkpoint term index witnesses disagree with the new terms"));
    }
    let mut nv_index = prev.nv_term_index_root;
    for (i, (object, path)) in new_nv.iter().zip(&proof.nv_index_paths).enumerate() {
        let id = prev.node_value_count + i as u64 + 1;
        nv_tree.push(nv_term_leaf(id, object))?;
        nv_index = verify_insert(NS_NV_INDEX, &nv_index, &node_value_sort_key(object), id, path)?;
    }
    let mut pred_index = prev.pred_term_index_root;
    for (i, (predicate, path)) in new_pred.iter().zip(&proof.pred_index_paths).enumerate() {
        let id = prev.predicate_count + i as u64 + 1;
        pred_tree.push(pred_term_leaf(id, predicate))?;
        pred_index = verify_insert(NS_PRED_INDEX, &pred_index, &predicate_sort_key(predicate), id, path)?;
    }
    if nv_tree.root() != next.nv_term_tree_root || pred_tree.root() != next.pred_term_tree_root {
        return Err(invalid("checkpoint term tree does not append the deltas' new terms"));
    }
    if nv_index != next.nv_term_index_root || pred_index != next.pred_term_index_root {
        return Err(invalid("checkpoint term index does not insert the deltas' new terms"));
    }
    if next.facts != prev.facts.union(DictFacts::from_objects(new_nv.iter())) {
        return Err(invalid("checkpoint dictionary facts do not follow from the new terms"));
    }

    // Present count.
    let n_prev = prev.present_count as usize;
    let n_next = (prev.present_count + all_additions.len() as u64)
        .checked_sub(all_removals.len() as u64)
        .ok_or_else(|| invalid("checkpoint deltas remove more triples than are present"))?;
    if n_next != next.present_count {
        return Err(invalid("checkpoint present count does not follow from the deltas"));
    }
    let n_next = n_next as usize;

    // The signed four-block link.
    let lens = [n_prev, all_additions.len(), all_removals.len(), n_next];
    let log = working_log(n_prev as u64, lens[1], lens[2], n_next as u64)?;
    if proof.working_log != log {
        return Err(invalid("checkpoint link working domain mismatch"));
    }
    let delta_roots: Vec<ProofHash> = deltas.iter().map(|d| d.state.commitment_root).collect();
    let mut transcript = Transcript254::new(TRANSCRIPT_LABEL);
    bind_statement(&mut transcript, prev, next, &trusted_head_root, &delta_roots, &lens, log);
    let beta = transcript.challenge_scalar(b"pow/checkpoint/link-beta");
    let alpha = transcript.challenge_scalar(b"pow/checkpoint/link-alpha");
    let inst = reconcile_link_instance(alpha, beta, &lens, &signs(), log)?;
    if proof.link.num_vars() != inst.num_vars {
        return Err(invalid("checkpoint link layer count mismatch"));
    }
    let stage = |what: &'static str| move |error: io::Error| invalid(format!("checkpoint {what}: {error}"));
    let (r_leaf, p_claim, q_claim) =
        bind_and_verify_fraction_tree(&inst, &proof.link, &mut transcript).map_err(stage("present link"))?;
    let r_low = &r_leaf[..log];
    let r_high = &r_leaf[log..];

    let opened = |commitments: &[ArkGT; 3],
                      n: usize,
                      evals: &[ArkFr],
                      opening: &Option<crate::proof::argument::ColumnOpening>,
                      transcript: &mut Transcript254|
     -> io::Result<[ArkFr; 3]> {
        if n == 0 {
            if !evals.is_empty() || opening.is_some() {
                return Err(invalid("checkpoint link opens an empty relation"));
            }
            return Ok([ArkFr::zero(); 3]);
        }
        let own = present_domain_log(n as u64)?;
        let opening = opening.as_ref().ok_or_else(|| invalid("checkpoint link lacks a present opening"))?;
        if evals.len() != 3 {
            return Err(invalid("checkpoint link present evaluation count mismatch"));
        }
        verify_column_openings(commitments, &r_low[..own], evals, opening, transcript)?;
        let factor = extension_factor(r_low, own);
        Ok([evals[0] * factor, evals[1] * factor, evals[2] * factor])
    };
    let prev_evals = opened(&prev.present_commitments, n_prev, &proof.prev_present_evals, &proof.prev_present_opening, &mut transcript)
        .map_err(stage("present link"))?;
    let next_evals = opened(&next.present_commitments, n_next, &proof.next_present_evals, &proof.next_present_opening, &mut transcript)
        .map_err(stage("present link"))?;
    let clear = |triples: &[IdTriple]| -> [ArkFr; 3] {
        [0, 1, 2].map(|c| clear_column_eval(triples.iter().map(|t| t[c]), r_low))
    };
    let block_evals = [prev_evals, clear(&all_additions), clear(&all_removals), next_evals];
    let mut leaf_evals = Vec::with_capacity(12);
    for (block, evals) in block_evals.iter().enumerate() {
        let selector = eq_at_index(block, r_high);
        for eval in evals {
            leaf_evals.push(selector * *eval);
        }
    }
    check_leaf_tie(&inst, &leaf_evals, p_claim, q_claim, &r_leaf).map_err(stage("present link"))?;

    // C_k's padding rows are zero.
    let next_log = present_domain_log(n_next as u64)?;
    match (&proof.padding, n_next) {
        (None, 0) => {}
        (Some(padding), n) if n > 0 => {
            let shape = padding_shape(next_log, n);
            let (r, evals) =
                verify_sumcheck(&shape, &padding.sumcheck, &mut transcript).map_err(stage("padding zero-check"))?;
            verify_column_openings(&next.present_commitments, &r, &evals, &padding.opening, &mut transcript)
                .map_err(stage("padding zero-check"))?;
        }
        _ => return Err(invalid("checkpoint padding proof presence mismatch")),
    }

    // C_k is strictly SPO-sorted.
    match (&proof.sorted, n_next) {
        (None, n) if n < 2 => {}
        (Some(sorted), n) if n >= 2 => {
            if sorted.sumcheck.round_polys.len() != next_log {
                return Err(invalid("checkpoint sortedness proof has the wrong domain"));
            }
            let (bn, bp) = sort_widths(next.node_value_count, next.predicate_count);
            let com = packed_commitment(&next.present_commitments, bn, bp);
            verify_strictly_sorted_bits(&com, n, 2 * bn + bp, sorted, &mut transcript)
                .map_err(stage("sortedness"))?;
        }
        _ => return Err(invalid("checkpoint sortedness proof presence mismatch")),
    }
    Ok(())
}
