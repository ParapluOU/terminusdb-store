//! Batched authenticated `id → hash(IRI)` dictionary lookup (CROSS-DICT join gadget 1).
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
//! # The gadget (cloned LogUp shape)
//! Over the shared `2^k` domain (dict columns relocated up to it, `obj_id` relocated up to
//! it), columns `[obj_id, v_hash, rec_q, dict_id, dict_hash, w, rec_d]`:
//!   * fold each side's `(id, hash)` with β powers: `q_fold = obj_id + β·v_hash`,
//!     `d_fold = dict_id + β·dict_hash`;
//!   * reciprocal witnesses `rec_q = 1/(α − q_fold)`, `rec_d = 1/(α − d_fold)` pinned by
//!     Identity claims;
//!   * multiplicity `w[j]` counts the real query rows whose id is dict row `j`;
//!   * masked LogUp link `Σ χ_n·rec_q − Σ χ_{n_dict}·w·rec_d = 0`.
//! `β` fingerprints the `(id, hash)` PAIR; `dict_id` uniqueness pins the matched dict row,
//! so `v_hash[i]` is FORCED to that row's true hash (a wrong `v_hash[i]` leaves an
//! uncancelled pole and the identity fails). Same soundness as the value lookup, minus the
//! datatype column.
//!
//! # Root authentication
//! The transcript is seeded with the OWNING layer's `commitment_root` (as
//! [`super::query`]'s `bind_statement` does), and the dict columns fed to the LogUp are
//! bound to the ROOT-COMMITTED `node_value_dict_commitments` via
//! [`super::relocate::verify_column_relocation`]. So `v_hash` is authenticated against that
//! one root — never a prover-chosen dictionary.

use std::io;

use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::transcript::Transcript;

use super::argument::{
    prove_sumcheck, verify_column_openings, verify_sumcheck, ColumnOpening, StoreProverContext,
    SumcheckProof, SumcheckShape, Transcript254,
};
use super::logup::{masked_multiset_link_claim, reciprocal_claim, reciprocal_witness, rlc_fold};
use super::relocate::{
    prove_column_relocation_with_context, verify_column_relocation, RelocationProof,
};
use super::commitment::{LayerCommitment, VerifierLayer};
use super::{invalid, ProofHash};

// Column ordering for the batched hash-lookup sumcheck (one shared `2^k` domain).
const HL_COL_OBJ_ID: usize = 0;
const HL_COL_V_HASH: usize = 1;
const HL_COL_REC_Q: usize = 2;
const HL_COL_DICT_ID: usize = 3;
const HL_COL_DICT_HASH: usize = 4;
const HL_COL_W: usize = 5;
const HL_COL_REC_D: usize = 6;
const HL_NUM_COLUMNS: usize = 7;
// chi_bounds indices: χ_n (real query rows) and χ_{n_dict} (real dict rows).
const HL_CHI_N: usize = 0;
const HL_CHI_NDICT: usize = 1;

const HASH_LOOKUP_DOMAIN: &[u8] = b"pow/dict-hash-lookup/domain-v1";
const HASH_LOOKUP_ROOT: &[u8] = b"pow/dict-hash-lookup/root";
const HASH_LOOKUP_STMT: &[u8] = b"pow/dict-hash-lookup/statement";
const HASH_LOOKUP_BETA: &[u8] = b"pow/dict-hash-lookup/fingerprint-beta";
const HASH_LOOKUP_ALPHA: &[u8] = b"pow/dict-hash-lookup/reciprocal-alpha";

#[inline]
fn f0() -> ArkFr {
    ArkFr::zero()
}

fn padded(values: &[ArkFr], log: usize) -> Vec<ArkFr> {
    let mut out = values.to_vec();
    out.resize(1usize << log, ArkFr::zero());
    out
}

/// The batched hash-lookup sumcheck: the two reciprocal (Identity) claims + the masked
/// LogUp link (ZeroSum) over `[obj_id, v_hash, rec_q, dict_id, dict_hash, w, rec_d]`.
#[derive(Clone, Debug)]
pub struct HashLookupInnerProof {
    pub sumcheck: SumcheckProof,
    pub opening: ColumnOpening,
    pub com_v_hash: ArkGT,
    pub com_w: ArkGT,
    pub com_rec_q: ArkGT,
    pub com_rec_d: ArkGT,
}

fn hash_lookup_shape(k: usize, real_n: usize, n_dict: usize, alpha: ArkFr, beta: ArkFr) -> SumcheckShape {
    // Fold order [id, hash] -> id + β·hash on both sides.
    let rec_q = reciprocal_claim(HL_COL_REC_Q, &[HL_COL_OBJ_ID, HL_COL_V_HASH], alpha, beta);
    let rec_d = reciprocal_claim(HL_COL_REC_D, &[HL_COL_DICT_ID, HL_COL_DICT_HASH], alpha, beta);
    let link = masked_multiset_link_claim(
        &[super::argument::Operand::Chi(HL_CHI_N)],
        HL_COL_REC_Q,
        &[
            super::argument::Operand::Chi(HL_CHI_NDICT),
            super::argument::Operand::Column(HL_COL_W),
        ],
        HL_COL_REC_D,
    );
    SumcheckShape {
        num_vars: k,
        num_columns: HL_NUM_COLUMNS,
        chi_bounds: vec![real_n, n_dict],
        claims: vec![rec_q, rec_d, link],
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
fn honest_hash_lookup(
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
fn prove_hash_lookup_inner(
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

    let fold_q = rlc_fold(beta, &[obj_id, &v_hash]);
    let fold_d = rlc_fold(beta, &[dict_id, dict_hash]);
    let rec_q = reciprocal_witness(alpha, &fold_q);
    let rec_d = reciprocal_witness(alpha, &fold_d);
    let com_rec_q = context.commit_column(&rec_q)?;
    let com_rec_d = context.commit_column(&rec_d)?;
    transcript.append_group(b"pow/dict-hash/com-rec-q", &com_rec_q);
    transcript.append_group(b"pow/dict-hash/com-rec-d", &com_rec_d);

    let shape = hash_lookup_shape(k, real_n, n_dict, alpha, beta);
    let columns = vec![
        obj_id.to_vec(),
        v_hash,
        rec_q,
        dict_id.to_vec(),
        dict_hash.to_vec(),
        w,
        rec_d,
    ];
    let (sumcheck, r, _evals) = prove_sumcheck(&shape, &columns, transcript)?;
    let opening = context.prove_column_openings(&columns, &r, transcript)?;
    Ok(HashLookupInnerProof { sumcheck, opening, com_v_hash, com_w, com_rec_q, com_rec_d })
}

/// Verify the batched hash lookup over columns sharing ONE `2^k` domain. On success the
/// proof's `com_v_hash` commits the TRUE dict hash of each real `obj_id` row.
#[allow(clippy::too_many_arguments)]
fn verify_hash_lookup_inner(
    com_obj_id: &ArkGT,
    com_dict_id: &ArkGT,
    com_dict_hash: &ArkGT,
    real_n: usize,
    n_dict: usize,
    proof: &HashLookupInnerProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    let k = proof.sumcheck.round_polys.len();
    if k == 0 || k >= usize::BITS as usize {
        return Err(invalid("hash-lookup proof has an implausible variable count"));
    }
    if real_n > (1usize << k) || n_dict > (1usize << k) {
        return Err(invalid("hash-lookup real length exceeds the column domain"));
    }
    bind_hash_lookup_statement(transcript, real_n, n_dict);
    transcript.append_group(b"pow/dict-hash/com-obj-id", com_obj_id);
    transcript.append_group(b"pow/dict-hash/com-dict-id", com_dict_id);
    transcript.append_group(b"pow/dict-hash/com-dict-hash", com_dict_hash);
    transcript.append_group(b"pow/dict-hash/com-v-hash", &proof.com_v_hash);
    transcript.append_group(b"pow/dict-hash/com-w", &proof.com_w);
    let beta = transcript.challenge_scalar(HASH_LOOKUP_BETA);
    let alpha = transcript.challenge_scalar(HASH_LOOKUP_ALPHA);
    transcript.append_group(b"pow/dict-hash/com-rec-q", &proof.com_rec_q);
    transcript.append_group(b"pow/dict-hash/com-rec-d", &proof.com_rec_d);

    let shape = hash_lookup_shape(k, real_n, n_dict, alpha, beta);
    let (r, evals) = verify_sumcheck(&shape, &proof.sumcheck, transcript)?;
    let commitments = [
        *com_obj_id,
        proof.com_v_hash,
        proof.com_rec_q,
        *com_dict_id,
        *com_dict_hash,
        proof.com_w,
        proof.com_rec_d,
    ];
    verify_column_openings(&commitments, &r, &evals, &proof.opening, transcript)
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
}
