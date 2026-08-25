//! Batched authenticated `id → hash(IRI)` dictionary lookup (CROSS-DICT join gadget 1).
//!
//! PLAIN-ENGLISH: given a committed column of dictionary ids, this proves a committed
//! output column holds, for each row, the collision-resistant HASH of the IRI/value that
//! id names in the layer's own root-bound dictionary. Because the hash depends only on the
//! IRI (not on which integer id a particular graph happened to assign it), two graphs with
//! SEPARATE dictionaries can be joined by joining on these authenticated hashes — see
//! [`super::semijoin`] and the cross-root join node. The soundness engine is the same LogUp
//! multiset argument as [`super::dict`]; this file adds the root-relocation plumbing.
//!
//! This is the single-output-column analogue of [`super::range::prove_dict_value_lookup`]
//! (the batched `id → (value, datatype)` object-range lookup): given a committed id
//! column `obj_id` (a relation's join-key column, the ids of node/value dictionary
//! entries) and the layer's already-committed, root-bound node/value dictionary columns
//! `(nv_dict_id, nv_dict_hash)` (committed as `node_value_dict_commitments =
//! (Com(nv_dict_hash), Com(nv_dict_id))`), it produces a committed AUTHENTICATED column
//! `v_hash` such that `v_hash[i] = hash_object(IRI-of(obj_id[i]))` for every real row.
//!
//! # Why this exists — the cross-dictionary reconciliation
//! The two committed triple graphs (INSTANCE and SCHEMA) have SEPARATE dictionaries under
//! SEPARATE `commitment_root`s: the SAME class IRI generally has DIFFERENT integer ids in
//! the two dicts, and two unrelated IRIs can share an id. So a raw dict-id equality across
//! the two graphs is BOTH unsound (colliding ids) AND incomplete (same IRI, differing
//! ids). [`super::dict::hash_object`] is deterministic AND dictionary-INDEPENDENT — the
//! same IRI hashes identically in both graphs — so joining on the AUTHENTICATED `v_hash`
//! reconciles the two dicts correctly. This gadget is what authenticates each side's
//! `v_hash` against ITS OWN root; the [`super::envelope::RelationProofPlanKind::CrossRootHashJoin`]
//! node then binds both roots to the trusted `database_root` and joins on the hashes.
//!
//! # The gadget (LogUp-GKR, block-disjoint two-side layout — Inc 2/5 migration)
//! Over the shared `2^k` domain `size` (dict columns relocated up to it, `obj_id` relocated
//! up to it), the committed DATA columns `[obj_id, v_hash, dict_id, dict_hash, w]`:
//!   * fold each side's `(id, hash)` with β powers, SAME power per logical column so the
//!     fold fingerprints the PAIR identically on both sides: `q_fold = obj_id + β·v_hash`,
//!     `d_fold = dict_id + β·dict_hash`;
//!   * multiplicity `w[j]` counts the real query rows whose id is dict row `j`;
//!   * prove `Σ_{i<real_n} 1/(α − q_fold[i]) − Σ_j w[j]/(α − d_fold[j]) = 0` with the GKR
//!     fraction tree ([`super::logup_gkr`]) over a `2·size` leaf vector laid out
//!     BLOCK-DISJOINT: the query rows in the LOW half `[0, size)` (denominator `α − q_fold`,
//!     numerator the ADDITIVE public prefix indicator `+[i < real_n]`), the dict rows in the
//!     HIGH half `[size, 2·size)` (denominator `α − d_fold`, numerator the committed
//!     multiplicity `−w`). NO reciprocal column is committed anywhere (`rec_q`/`rec_d` are
//!     gone). `β` fingerprints the `(id, hash)` PAIR; `dict_id` uniqueness pins the matched
//!     dict row, so `v_hash[i]` is FORCED to that row's true hash (a wrong `v_hash[i]` leaves
//!     an uncancelled pole and the fraction sum's root `P ≠ 0`).
//!
//! Block-disjoint structural enforcement (Inc 2). The committed columns stay length `size`;
//! the prover feeds the tree their zero-extensions — query columns zero on the high half,
//! dict columns + `w` zero on the low half. The verifier opens the length-`size` committed
//! columns at the low `k` bits `r_low` of the GKR leaf point and reconstructs each `2·size`
//! leaf-MLE via the top-bit factor: `(1 − r_high)·col(r_low)` for a query column (low half)
//! and `r_high·col(r_low)` for a dict column (high half). This STRUCTURALLY forces each
//! side's fold columns to zero in the OTHER block (the same enforcement `permutation` uses),
//! so neither side can leak into the other.
//!
//! GAP-1 masking, structural. The reciprocal-era `Σ χ_{n_dict}·w·rec_d` masked out the dict
//! padding tail `[n_dict, size)` (rows whose relocated `dict_id`/`dict_hash` are ROOT-BOUND
//! to zero, so `d_fold = 0`) to stop a prover using a padding pole `−w/α` to forge. Here that
//! mask needs NO co-running `(1 − χ_{n_dict})·w = 0` Identity: the honest `w` is zero on the
//! tail, and a FORGED nonzero `w` there produces a leaf `−w/α` whose pole at `α − 0` is
//! UNMATCHED — every real query fold is `obj_id[i] + β·v_hash[i]` with a genuine dict id
//! `obj_id[i] ≥ 1`, hence nonzero (whp over the post-commit `β`), so no query leaf has
//! denominator `α`. An unmatched padding pole makes the fraction sum's root `P ≠ 0` and the
//! proof REJECTS. This is the SAME self-annihilating-padding mechanism [`super::dict`] relies
//! on for its own dict padding (whose `q_fold ≠ 0` is likewise required), extended here to a
//! committed query column. (Caller contract: the root-bound `obj_id` column carries genuine
//! dictionary ids `≥ 1` on its real rows; every real caller supplies exactly that.)
//!
//! # Root authentication
//! The transcript is seeded with the OWNING layer's `commitment_root` (as
//! [`super::query`]'s `bind_statement` does), and the dict columns fed to the LogUp are
//! bound to the ROOT-COMMITTED `node_value_dict_commitments` via
//! [`super::relocate::verify_column_relocation`]. So `v_hash` is authenticated against that
//! one root — never a prover-chosen dictionary.
//!
//! See the developer primer for the shared concepts this builds on:
//! [LogUp](crate::proof::primer), [commitment](crate::proof::primer), and
//! [the layer-commitment / resolution-root model](crate::proof::primer).

use std::io;

use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::transcript::Transcript;

use super::argument::{
    mle_full_eval, powers, verify_column_openings, ColumnOpening, StoreProverContext, Transcript254,
};
use super::logup_gkr::{
    bind_and_verify_fraction_tree, check_leaf_tie, prove_fraction_sum_zero, FractionInstance,
    GkrFractionProof, LinearForm, LinearTerm, Src,
};
use super::relocate::{
    prove_column_relocation_with_context, verify_column_relocation, RelocationProof,
};
use super::commitment::{LayerCommitment, VerifierLayer};
use super::{invalid, ProofHash};

// Column ordering for the batched hash-lookup DATA columns (each committed at the shared
// `2^k` domain `size`). The reciprocal witnesses `rec_q`/`rec_d` are GONE: the LogUp-GKR
// migration proves the same log-derivative sum via a block-disjoint fraction tree over
// uncommitted intermediates, so no reciprocal column is committed anywhere.
//   * query side (LOW half of the `2·size` leaf vector): the join-key id + its hash;
//   * dict side (HIGH half): the dictionary id, its hash, and the multiplicity.
const HL_COL_OBJ_ID: usize = 0;
const HL_COL_V_HASH: usize = 1;
const HL_COL_DICT_ID: usize = 2;
const HL_COL_DICT_HASH: usize = 3;
const HL_COL_W: usize = 4;
const HL_NUM_COLUMNS: usize = 5;

const HASH_LOOKUP_DOMAIN: &[u8] = b"pow/dict-hash-lookup/domain-v1";
const HASH_LOOKUP_ROOT: &[u8] = b"pow/dict-hash-lookup/root";
const HASH_LOOKUP_STMT: &[u8] = b"pow/dict-hash-lookup/statement";
const HASH_LOOKUP_BETA: &[u8] = b"pow/dict-hash-lookup/fingerprint-beta";
const HASH_LOOKUP_ALPHA: &[u8] = b"pow/dict-hash-lookup/reciprocal-alpha";

#[inline]
fn f0() -> ArkFr {
    ArkFr::zero()
}
#[inline]
fn f1() -> ArkFr {
    ArkFr::one()
}

fn padded(values: &[ArkFr], log: usize) -> Vec<ArkFr> {
    let mut out = values.to_vec();
    out.resize(1usize << log, ArkFr::zero());
    out
}

/// Zero-extend a length-`size` column into the LOW half of a `2·size` leaf vector (the
/// query block); the high half stays zero. MLE at `(r_low, r_high)` is
/// `(1 − r_high)·MLE_size(col, r_low)`.
fn low_half(col: &[ArkFr], leaf_count: usize) -> Vec<ArkFr> {
    let mut ext = vec![f0(); leaf_count];
    ext[..col.len()].copy_from_slice(col);
    ext
}

/// Zero-extend a length-`size` column into the HIGH half of a `2·size` leaf vector (the
/// dict block); the low half stays zero. MLE at `(r_low, r_high)` is
/// `r_high·MLE_size(col, r_low)`.
fn high_half(col: &[ArkFr], leaf_count: usize) -> Vec<ArkFr> {
    let mut ext = vec![f0(); leaf_count];
    ext[leaf_count / 2..].copy_from_slice(col);
    ext
}

/// The batched hash-lookup GKR fraction proof: the fraction-sum-zero proof for
/// `Σ_{i<real_n} 1/(α − q_fold) − Σ_j w/(α − d_fold) = 0`, the claimed evaluations of the
/// five DATA columns at `r_low` (the low `k` bits of the GKR leaf point), one batched Dory
/// opening of those columns at `r_low`, and the two freshly committed columns the verifier
/// binds. NO reciprocal (`rec_q`/`rec_d`) column exists anymore.
#[derive(Clone, Debug)]
pub struct HashLookupInnerProof {
    pub gkr: GkrFractionProof,
    pub col_evals: Vec<ArkFr>,
    pub opening: ColumnOpening,
    pub com_v_hash: ArkGT,
    pub com_w: ArkGT,
}

/// The PUBLIC block-disjoint fraction-sum-zero instance over a length-`2·size` leaf vector
/// (`size = 2^k`): query rows in the low half `[0, size)`, dict rows in the high half
/// `[size, 2·size)`. Built IDENTICALLY by prover and verifier from public inputs only.
///   * denominator `q_leaf = α − Σ_d β^d·(query_col_d + dict_col_d)` — since the blocks are
///     disjoint, this is `α − q_fold` on the low half and `α − d_fold` on the high half,
///     with the SAME `β^d` fingerprinting a given logical column (id → β^0, hash → β^1) on
///     both sides;
///   * numerator `p_leaf = +[i < real_n] − w`: each real query row counts `+1` (public
///     prefix indicator, low half), each dict row `−w[j]` (committed multiplicity, high
///     half); every padding query row gets `0` and self-annihilates.
fn hash_lookup_fraction_instance(alpha: ArkFr, beta: ArkFr, real_n: usize, k: usize) -> FractionInstance {
    let beta_powers = powers(beta, 2);
    let q_terms = vec![
        // Query side (low half): obj_id + β·v_hash.
        LinearTerm { coeff: f0() - beta_powers[0], src: Src::Column(HL_COL_OBJ_ID) },
        LinearTerm { coeff: f0() - beta_powers[1], src: Src::Column(HL_COL_V_HASH) },
        // Dict side (high half): dict_id + β·dict_hash, SAME β powers, so the fold
        // fingerprints the (id, hash) PAIR identically on both sides.
        LinearTerm { coeff: f0() - beta_powers[0], src: Src::Column(HL_COL_DICT_ID) },
        LinearTerm { coeff: f0() - beta_powers[1], src: Src::Column(HL_COL_DICT_HASH) },
    ];
    let p_terms = vec![
        LinearTerm { coeff: f1(), src: Src::Prefix(real_n) },
        LinearTerm { coeff: f0() - f1(), src: Src::Column(HL_COL_W) },
    ];
    FractionInstance {
        num_vars: k + 1,
        num_cols: HL_NUM_COLUMNS,
        p_form: LinearForm { constant: f0(), terms: p_terms },
        q_form: LinearForm { constant: alpha, terms: q_terms },
    }
}

fn bind_hash_lookup_statement(transcript: &mut Transcript254, real_n: usize, n_dict: usize) {
    transcript.append_bytes(HASH_LOOKUP_DOMAIN, b"parture/proof-of-woql/dict-hash-lookup/v0");
    transcript.append_bytes(HASH_LOOKUP_STMT, &(real_n as u64).to_le_bytes());
    transcript.append_bytes(HASH_LOOKUP_STMT, &(n_dict as u64).to_le_bytes());
}

/// The honest witness: `v_hash[i]` is the dict hash of the real dict row whose id equals
/// `obj_id[i]` (else the `0` sentinel, only on padding rows, which are masked out); `w[j]`
/// counts the real query rows (`i < real_n`) mapping to dict row `j`.
pub(crate) fn honest_hash_lookup(
    obj_id: &[ArkFr],
    dict_id: &[ArkFr],
    dict_hash: &[ArkFr],
    real_n: usize,
    n_dict: usize,
) -> (Vec<ArkFr>, Vec<ArkFr>) {
    let size = obj_id.len();
    let real_dict = n_dict.min(dict_id.len());
    let mut v_hash = vec![f0(); size];
    for i in 0..size {
        for j in 0..real_dict {
            if dict_id[j] == obj_id[i] {
                v_hash[i] = dict_hash[j];
                break;
            }
        }
    }
    let mut w = vec![f0(); size];
    for j in 0..real_dict {
        let mut count = 0u64;
        for id in obj_id.iter().take(real_n.min(size)) {
            if *id == dict_id[j] {
                count += 1;
            }
        }
        w[j] = ArkFr::from_u64(count);
    }
    (v_hash, w)
}

/// Prove the batched hash lookup over columns that already share ONE `2^k` domain.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_hash_lookup_inner(
    context: &StoreProverContext<'_>,
    obj_id: &[ArkFr],
    com_obj_id: &ArkGT,
    dict_id: &[ArkFr],
    dict_hash: &[ArkFr],
    com_dict_id: &ArkGT,
    com_dict_hash: &ArkGT,
    real_n: usize,
    n_dict: usize,
    transcript: &mut Transcript254,
) -> io::Result<HashLookupInnerProof> {
    let size = obj_id.len();
    if !size.is_power_of_two() || size < 2 {
        return Err(invalid("hash-lookup object-id column length must be a power of two >= 2"));
    }
    for column in [dict_id, dict_hash] {
        if column.len() != size {
            return Err(invalid("hash-lookup column length mismatch (all columns share one domain)"));
        }
    }
    if real_n > size || n_dict > size {
        return Err(invalid("hash-lookup real length exceeds the column domain"));
    }
    let k = size.trailing_zeros() as usize;
    let (v_hash, w) = honest_hash_lookup(obj_id, dict_id, dict_hash, real_n, n_dict);

    // Commit the two fresh DATA columns, THEN draw beta/alpha. Caller contract (Inc 0
    // audit): alpha MUST be drawn after the column commitments are bound so the fraction
    // leaves are fixed before any per-layer GKR challenge. There is no reciprocal
    // commitment anymore, so this is the last binding before the fold challenges.
    let com_v_hash = context.commit_column(&v_hash)?;
    let com_w = context.commit_column(&w)?;
    bind_hash_lookup_statement(transcript, real_n, n_dict);
    transcript.append_group(b"pow/dict-hash/com-obj-id", com_obj_id);
    transcript.append_group(b"pow/dict-hash/com-dict-id", com_dict_id);
    transcript.append_group(b"pow/dict-hash/com-dict-hash", com_dict_hash);
    transcript.append_group(b"pow/dict-hash/com-v-hash", &com_v_hash);
    transcript.append_group(b"pow/dict-hash/com-w", &com_w);
    let beta = transcript.challenge_scalar(HASH_LOOKUP_BETA);
    let alpha = transcript.challenge_scalar(HASH_LOOKUP_ALPHA);

    // Prove the fraction sum over the block-disjoint `2·size` zero-extensions: query
    // columns fill the low half, dict columns + w the high half. The tree lands at
    // `r_leaf` (length `k + 1`), whose low `k` bits are the opening point.
    let inst = hash_lookup_fraction_instance(alpha, beta, real_n, k);
    let leaf_count = inst.size();
    let extended = vec![
        low_half(obj_id, leaf_count),
        low_half(&v_hash, leaf_count),
        high_half(dict_id, leaf_count),
        high_half(dict_hash, leaf_count),
        high_half(&w, leaf_count),
    ];
    let (gkr, r_leaf) = prove_fraction_sum_zero(&inst, &extended, transcript)?;

    // Single batched Dory opening of the length-`size` DATA columns at `r_low`, bound to
    // their (root-anchored / freshly committed) commitments on this transcript.
    let r_low = &r_leaf[..k];
    let columns = vec![obj_id.to_vec(), v_hash, dict_id.to_vec(), dict_hash.to_vec(), w];
    let col_evals: Vec<ArkFr> = columns.iter().map(|c| mle_full_eval(c, r_low)).collect();
    let opening = context.prove_column_openings(&columns, r_low, transcript)?;
    Ok(HashLookupInnerProof { gkr, col_evals, opening, com_v_hash, com_w })
}

/// Verify the batched hash lookup over columns sharing ONE `2^k` domain. On success the
/// proof's `com_v_hash` commits the TRUE dict hash of each real `obj_id` row.
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_hash_lookup_inner(
    com_obj_id: &ArkGT,
    com_dict_id: &ArkGT,
    com_dict_hash: &ArkGT,
    real_n: usize,
    n_dict: usize,
    proof: &HashLookupInnerProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    // The GKR tree has `k + 1` layers (`2·size` leaves for a length-`size = 2^k` domain),
    // so `k = num_vars - 1` and the committed columns have length `size = 2^k`.
    let num_vars = proof.gkr.num_vars();
    if num_vars < 2 || num_vars >= usize::BITS as usize {
        return Err(invalid("hash-lookup proof has an implausible variable count"));
    }
    let k = num_vars - 1;
    if real_n > (1usize << k) || n_dict > (1usize << k) {
        return Err(invalid("hash-lookup real length exceeds the column domain"));
    }
    if proof.col_evals.len() != HL_NUM_COLUMNS {
        return Err(invalid("hash-lookup proof has the wrong number of column evaluations"));
    }
    bind_hash_lookup_statement(transcript, real_n, n_dict);
    transcript.append_group(b"pow/dict-hash/com-obj-id", com_obj_id);
    transcript.append_group(b"pow/dict-hash/com-dict-id", com_dict_id);
    transcript.append_group(b"pow/dict-hash/com-dict-hash", com_dict_hash);
    transcript.append_group(b"pow/dict-hash/com-v-hash", &proof.com_v_hash);
    transcript.append_group(b"pow/dict-hash/com-w", &proof.com_w);
    let beta = transcript.challenge_scalar(HASH_LOOKUP_BETA);
    let alpha = transcript.challenge_scalar(HASH_LOOKUP_ALPHA);

    // Bind the public instance and verify the GKR tree -> the leaf point + claims.
    let inst = hash_lookup_fraction_instance(alpha, beta, real_n, k);
    let (r_leaf, p_claim, q_claim) = bind_and_verify_fraction_tree(&inst, &proof.gkr, transcript)?;

    // Open the length-`size` committed columns at `r_low`, bound to their commitments.
    let r_low = &r_leaf[..k];
    let r_high = r_leaf[k];
    let commitments = [
        *com_obj_id,
        proof.com_v_hash,
        *com_dict_id,
        *com_dict_hash,
        proof.com_w,
    ];
    verify_column_openings(&commitments, r_low, &proof.col_evals, &proof.opening, transcript)?;

    // Reconstruct each `2·size` leaf-MLE from the length-`size` opening via the top-bit
    // zero-extension factor: query columns live in the LOW half (factor `1 − r_high`), dict
    // columns + w in the HIGH half (factor `r_high`). This is the block-disjoint
    // enforcement — each side's fold columns reconstruct to zero in the OTHER block, so
    // neither side can leak into the other — then tie to the GKR-reduced leaf claims.
    let one_minus_high = f1() - r_high;
    let leaf_evals = vec![
        one_minus_high * proof.col_evals[HL_COL_OBJ_ID],
        one_minus_high * proof.col_evals[HL_COL_V_HASH],
        r_high * proof.col_evals[HL_COL_DICT_ID],
        r_high * proof.col_evals[HL_COL_DICT_HASH],
        r_high * proof.col_evals[HL_COL_W],
    ];
    check_leaf_tie(&inst, &leaf_evals, p_claim, q_claim, &r_leaf)
}

/// The full authenticated `id → hash(IRI)` lookup proof: the root-bound dict columns
/// relocated up to the working domain `D = max(obj_log, dict_log)`, the query id column
/// relocated up to `D`, and the batched inner lookup. `lookup.com_v_hash` (at `D`) is the
/// authenticated hash column downstream gadgets (the cross-root inner join) consume.
#[derive(Clone, Debug)]
pub struct DictHashLookupProof {
    pub relocate_dict_hash: RelocationProof,
    pub relocate_dict_id: RelocationProof,
    pub com_dict_hash_at_d: ArkGT,
    pub com_dict_id_at_d: ArkGT,
    pub relocate_obj: RelocationProof,
    pub com_obj_at_d: ArkGT,
    pub lookup: HashLookupInnerProof,
}

/// The working-domain log for a lookup whose query column is at `obj_log` and whose layer
/// dictionary has `n_dict` entries (padded to `dict_len = n_dict.next_power_of_two().max(2)`).
/// `D = max(obj_log, dict_log)`: the dict must fit (so `D >= dict_log`), and relocating the
/// smaller query column UP is always valid.
pub fn working_domain_log(obj_log: usize, n_dict: usize) -> usize {
    let dict_log = n_dict.next_power_of_two().max(2).trailing_zeros() as usize;
    obj_log.max(dict_log)
}

/// Prove that `com_v_hash` (in the returned proof, at domain `2^D`) commits the true
/// `hash_object(IRI-of(obj_id[i]))` of each real row of the committed `obj_id` column,
/// authenticated against `layer`'s `commitment_root`. The transcript is seeded with that
/// root, so `com_v_hash` is bound to the owning layer.
///
/// Returns the proof AND the authenticated `v_hash` column (length `2^D`), which the caller
/// (the cross-root node) feeds as a join key. The column is fully determined by the proof, so
/// returning it exposes nothing the verifier does not re-derive from `com_v_hash`.
#[allow(clippy::too_many_arguments)]
pub fn prove_dict_hash_lookup(
    context: &StoreProverContext<'_>,
    obj_id: &[ArkFr],
    com_obj_id: &ArkGT,
    obj_log: usize,
    real_n: usize,
    layer: &LayerCommitment,
    transcript: &mut Transcript254,
) -> io::Result<(DictHashLookupProof, Vec<ArkFr>)> {
    if obj_id.len() != 1usize << obj_log {
        return Err(invalid("hash-lookup obj column length does not match obj_log"));
    }
    let root: ProofHash = layer.state.commitment_root;
    transcript.append_bytes(HASH_LOOKUP_ROOT, &root);

    // Root-bound node/value dictionary columns `(hash, id)`.
    let (dict_hash, dict_id) = layer.node_value_dictionary_columns()?;
    let n_dict = layer.state.node_value_count as usize;
    if n_dict != layer.node_value_catalog.len() || n_dict == 0 {
        return Err(invalid("hash-lookup dictionary count mismatch or empty"));
    }
    let dict_log = dict_id.len().trailing_zeros() as usize;
    let d_log = obj_log.max(dict_log);

    // Relocate the root-committed dict columns up to the working domain D.
    let dict_hash_at_d = padded(&dict_hash, d_log);
    let dict_id_at_d = padded(&dict_id, d_log);
    let com_dict_hash_at_d = context.commit_column(&dict_hash_at_d)?;
    let com_dict_id_at_d = context.commit_column(&dict_id_at_d)?;
    let relocate_dict_hash = prove_column_relocation_with_context(
        context, &dict_hash, &layer.node_value_dict_commitments.0, &dict_hash_at_d, &com_dict_hash_at_d, transcript,
    )?;
    let relocate_dict_id = prove_column_relocation_with_context(
        context, &dict_id, &layer.node_value_dict_commitments.1, &dict_id_at_d, &com_dict_id_at_d, transcript,
    )?;

    // Relocate the committed query id column up to D.
    let obj_at_d = padded(obj_id, d_log);
    let com_obj_at_d = context.commit_column(&obj_at_d)?;
    let relocate_obj = prove_column_relocation_with_context(
        context, obj_id, com_obj_id, &obj_at_d, &com_obj_at_d, transcript,
    )?;

    let lookup = prove_hash_lookup_inner(
        context, &obj_at_d, &com_obj_at_d, &dict_id_at_d, &dict_hash_at_d,
        &com_dict_id_at_d, &com_dict_hash_at_d, real_n, n_dict, transcript,
    )?;
    // The authenticated hash column (length 2^D), determined entirely by `lookup.com_v_hash`.
    let v_hash = honest_hash_lookup(&obj_at_d, &dict_id_at_d, &dict_hash_at_d, real_n, n_dict).0;

    Ok((
        DictHashLookupProof {
            relocate_dict_hash,
            relocate_dict_id,
            com_dict_hash_at_d,
            com_dict_id_at_d,
            relocate_obj,
            com_obj_at_d,
            lookup,
        },
        v_hash,
    ))
}

/// Verify a full `id → hash(IRI)` lookup against `layer`'s root. On success,
/// `proof.lookup.com_v_hash` (at domain `2^D`, `D = working_domain_log(obj_log, n_dict)`)
/// commits the true dict hash of each real `obj_id` row. The caller (the cross-root node)
/// must have already validated `layer`'s view (via `validate_database_root`).
pub fn verify_dict_hash_lookup(
    com_obj_id: &ArkGT,
    obj_log: usize,
    real_n: usize,
    layer: &impl VerifierLayer,
    proof: &DictHashLookupProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    let view = layer.verifier_view();
    let root: ProofHash = view.state.commitment_root;
    transcript.append_bytes(HASH_LOOKUP_ROOT, &root);

    let n_dict = view.state.node_value_count as usize;
    if n_dict != view.node_value_catalog.len() || n_dict == 0 {
        return Err(invalid("hash-lookup dictionary count mismatch or empty"));
    }
    let dict_len = n_dict.next_power_of_two().max(2);
    let dict_log = dict_len.trailing_zeros() as usize;
    let d_log = obj_log.max(dict_log);

    verify_column_relocation(
        &view.node_value_dict_commitments.0, dict_log, &proof.com_dict_hash_at_d, d_log, &proof.relocate_dict_hash, transcript,
    )?;
    verify_column_relocation(
        &view.node_value_dict_commitments.1, dict_log, &proof.com_dict_id_at_d, d_log, &proof.relocate_dict_id, transcript,
    )?;
    verify_column_relocation(
        com_obj_id, obj_log, &proof.com_obj_at_d, d_log, &proof.relocate_obj, transcript,
    )?;

    verify_hash_lookup_inner(
        &proof.com_obj_at_d, &proof.com_dict_id_at_d, &proof.com_dict_hash_at_d, real_n, n_dict, &proof.lookup, transcript,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::ValueTriple;
    use crate::proof::argument::commit_column;
    use crate::proof::dict::hash_object;
    use crate::proof::CanonicalObject;
    use crate::store::sync::open_sync_memory_store;
    use std::collections::HashMap;

    fn transcript() -> Transcript254 {
        Transcript254::new(b"parture-dict-hash-lookup-test")
    }

    fn build_layer() -> std::sync::Arc<LayerCommitment> {
        let iri = |s: &str| format!("http://ex/{s}");
        let p = iri("rel");
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        // A mix of node objects (class-like IRIs) and a value, so the node/value dict has
        // several node entries whose ids we look up to their IRI hashes.
        for s in ["A", "B", "C", "D", "Other"] {
            builder
                .add_value_triple(ValueTriple::new_node(&iri(s), &p, &iri("Class")))
                .unwrap();
        }
        builder
            .add_value_triple(ValueTriple::new_string_value(&iri("s"), &p, "hello"))
            .unwrap();
        builder.commit().unwrap().proof_commitment().unwrap()
    }

    /// object_id -> CanonicalObject over the whole node/value dictionary.
    fn object_map(layer: &LayerCommitment) -> HashMap<u64, CanonicalObject> {
        layer
            .node_value_catalog
            .iter()
            .map(|r| (r.object_id, r.triple.object.clone()))
            .collect()
    }

    /// Build a committed id column of length `2^obj_log` from `ids` (real prefix), the rest
    /// padded with the `0` sentinel.
    fn obj_column(ids: &[u64], obj_log: usize) -> (Vec<ArkFr>, ArkGT) {
        let mut col = vec![f0(); 1usize << obj_log];
        for (slot, id) in col.iter_mut().zip(ids) {
            *slot = ArkFr::from_u64(*id);
        }
        let com = commit_column(&col).unwrap();
        (col, com)
    }

    #[test]
    fn hash_lookup_authenticates_id_to_iri_hash() {
        let layer = build_layer();
        let map = object_map(&layer);
        // Pick three node ids from the dictionary.
        let node_ids: Vec<u64> = layer
            .node_value_catalog
            .iter()
            .filter(|r| matches!(r.triple.object, CanonicalObject::Node(_)))
            .map(|r| r.object_id)
            .take(3)
            .collect();
        assert!(node_ids.len() >= 3, "need >=3 node objects");
        let obj_log = node_ids.len().next_power_of_two().max(2).trailing_zeros() as usize;
        let (obj, com_obj) = obj_column(&node_ids, obj_log);

        let mut t = transcript();
        let (proof, _vh) = prove_dict_hash_lookup(
            &StoreProverContext::cpu(), &obj, &com_obj, obj_log, node_ids.len(), &layer, &mut t,
        )
        .unwrap();

        let mut tv = transcript();
        verify_dict_hash_lookup(&com_obj, obj_log, node_ids.len(), &*layer, &proof, &mut tv).unwrap();

        // The authenticated v_hash column must be exactly hash_object(IRI-of(id)) per row.
        let d_log = working_domain_log(obj_log, layer.node_value_catalog.len());
        let mut expected = vec![f0(); 1usize << d_log];
        for (i, id) in node_ids.iter().enumerate() {
            expected[i] = hash_object(&map[id]);
        }
        assert_eq!(
            proof.lookup.com_v_hash,
            commit_column(&expected).unwrap(),
            "authenticated v_hash column commits the true per-row IRI hashes"
        );
    }

    #[test]
    fn hash_lookup_rejects_wrong_obj_commitment() {
        let layer = build_layer();
        let node_ids: Vec<u64> = layer
            .node_value_catalog
            .iter()
            .filter(|r| matches!(r.triple.object, CanonicalObject::Node(_)))
            .map(|r| r.object_id)
            .take(3)
            .collect();
        let obj_log = node_ids.len().next_power_of_two().max(2).trailing_zeros() as usize;
        let (obj, com_obj) = obj_column(&node_ids, obj_log);
        let mut t = transcript();
        let (proof, _vh) = prove_dict_hash_lookup(
            &StoreProverContext::cpu(), &obj, &com_obj, obj_log, node_ids.len(), &layer, &mut t,
        )
        .unwrap();
        // A different committed id column (swap two ids) must fail the obj relocation bind.
        let mut swapped = node_ids.clone();
        swapped.swap(0, 1);
        let (_obj2, com_obj2) = obj_column(&swapped, obj_log);
        let mut tv = transcript();
        assert!(
            verify_dict_hash_lookup(&com_obj2, obj_log, node_ids.len(), &*layer, &proof, &mut tv).is_err(),
            "verifying against a different committed id column must reject"
        );
    }

    #[test]
    fn hash_lookup_rejects_tampered_hash_column() {
        let layer = build_layer();
        let node_ids: Vec<u64> = layer
            .node_value_catalog
            .iter()
            .filter(|r| matches!(r.triple.object, CanonicalObject::Node(_)))
            .map(|r| r.object_id)
            .take(3)
            .collect();
        let obj_log = node_ids.len().next_power_of_two().max(2).trailing_zeros() as usize;
        let (obj, com_obj) = obj_column(&node_ids, obj_log);
        let mut t = transcript();
        let (mut proof, _vh) = prove_dict_hash_lookup(
            &StoreProverContext::cpu(), &obj, &com_obj, obj_log, node_ids.len(), &layer, &mut t,
        )
        .unwrap();
        // Corrupt the authenticated hash-column commitment: the sumcheck opening no longer
        // binds, so verification rejects.
        let bogus = commit_column(&vec![ArkFr::from_u64(12345); 1usize << working_domain_log(obj_log, layer.node_value_catalog.len())]).unwrap();
        proof.lookup.com_v_hash = bogus;
        let mut tv = transcript();
        assert!(
            verify_dict_hash_lookup(&com_obj, obj_log, node_ids.len(), &*layer, &proof, &mut tv).is_err(),
            "a tampered v_hash commitment must reject"
        );
    }

    // GAP-1 forge negative. The reciprocal-era `χ_{n_dict}` mask zeroed the dict padding
    // tail so a prover could not use a padding pole `−w/α` (dict padding rows have
    // root-bound `dict_id = dict_hash = 0`, so `d_fold = 0`, a pole at `α`) to fake a
    // lookup. The migrated gadget drops that mask and closes the same forge STRUCTURALLY:
    // the honest `w` is zero on the tail, and a FORGED nonzero `w` there is a leaf `−w/α`
    // whose pole is UNMATCHED (every real query fold is `obj_id + β·v_hash` with a genuine
    // dict id `≥ 1`, hence nonzero whp), so the fraction sum's root `P ≠ 0` and the proof
    // rejects. This drives the inner prover flow with an explicit padding-tail forgery.
    #[test]
    fn hash_lookup_rejects_padding_tail_forged_pole() {
        let fr = ArkFr::from_u64;
        // Dict of 3 real entries (ids 1,2,3) padded to size 4: row 3 is padding
        // (dict_id = dict_hash = 0 -> d_fold = 0 -> a pole at α).
        let size = 4usize;
        let k = 2usize;
        let dict_id = vec![fr(1), fr(2), fr(3), f0()];
        let dict_hash = vec![fr(10), fr(20), fr(30), f0()];
        let n_dict = 3usize;
        // Query two genuine member ids (both ≥ 1) — honestly this verifies.
        let obj_id = vec![fr(1), fr(2), f0(), f0()];
        let real_n = 2usize;
        assert_eq!(obj_id.len(), size);

        let context = StoreProverContext::cpu();
        let com_obj = commit_column(&obj_id).unwrap();
        let com_dict_id = commit_column(&dict_id).unwrap();
        let com_dict_hash = commit_column(&dict_hash).unwrap();

        // Honest witness, then FORGE a nonzero multiplicity on the padding dict row 3.
        let (v_hash, mut w) = honest_hash_lookup(&obj_id, &dict_id, &dict_hash, real_n, n_dict);
        assert!(w[3].is_zero(), "honest padding multiplicity must be zero");
        w[3] = f1(); // the padding-tail pole forgery

        let com_v_hash = context.commit_column(&v_hash).unwrap();
        let com_w = context.commit_column(&w).unwrap();
        let mut pt = transcript();
        bind_hash_lookup_statement(&mut pt, real_n, n_dict);
        pt.append_group(b"pow/dict-hash/com-obj-id", &com_obj);
        pt.append_group(b"pow/dict-hash/com-dict-id", &com_dict_id);
        pt.append_group(b"pow/dict-hash/com-dict-hash", &com_dict_hash);
        pt.append_group(b"pow/dict-hash/com-v-hash", &com_v_hash);
        pt.append_group(b"pow/dict-hash/com-w", &com_w);
        let beta = pt.challenge_scalar(HASH_LOOKUP_BETA);
        let alpha = pt.challenge_scalar(HASH_LOOKUP_ALPHA);

        let inst = hash_lookup_fraction_instance(alpha, beta, real_n, k);
        let leaf_count = inst.size();
        let extended = vec![
            low_half(&obj_id, leaf_count),
            low_half(&v_hash, leaf_count),
            high_half(&dict_id, leaf_count),
            high_half(&dict_hash, leaf_count),
            high_half(&w, leaf_count),
        ];
        let (gkr, r_leaf) = prove_fraction_sum_zero(&inst, &extended, &mut pt).unwrap();
        // The forged padding pole is UNMATCHED, so the zero-sum breaks at the root.
        assert!(!gkr.root_p.is_zero(), "forged padding-tail pole must break the zero-sum");

        let r_low = &r_leaf[..k];
        let columns = vec![obj_id.clone(), v_hash, dict_id.clone(), dict_hash.clone(), w];
        let col_evals: Vec<ArkFr> = columns.iter().map(|c| mle_full_eval(c, r_low)).collect();
        let opening = context.prove_column_openings(&columns, r_low, &mut pt).unwrap();
        let proof = HashLookupInnerProof { gkr, col_evals, opening, com_v_hash, com_w };

        let mut vt = transcript();
        assert!(
            verify_hash_lookup_inner(
                &com_obj, &com_dict_id, &com_dict_hash, real_n, n_dict, &proof, &mut vt,
            )
            .is_err(),
            "a forged padding-tail multiplicity pole must reject"
        );
    }
}
