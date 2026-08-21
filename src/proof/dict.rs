//! Q3 dictionary-binding LogUp membership gadget (blueprint §4), in isolation.
//!
//! Given committed dictionary columns `dict_hash[k]`, `dict_id[k]` (`k in [0, D)`,
//! padded to a power of two) and a PUBLIC query pair `(q_hash, id)` — where the
//! verifier computes `q_hash = hash_term(constant)` itself — this proves
//! `(q_hash, id)` is a genuine row of the dictionary, via single-element LogUp
//! membership. Q3b will commit the dictionary inside `LayerCommitment` and wire
//! this into `prove_layer_filter`; here it stands alone.
//!
//! Gadget (LogUp-GKR, Inc 1 of the reciprocal-elimination migration):
//!   * fold each dict row `dict_fold[k] = beta*dict_hash[k] + dict_id[k]`; the
//!     query folds to the PUBLIC scalar `q_fold = beta*q_hash + id`.
//!   * commit ONLY the DATA columns `[dict_hash, dict_id, mult]`, where `mult[k]`
//!     is a multiplicity indicator (1 at the matching row, else 0). NO reciprocal
//!     column is committed (the old `dstar` is gone).
//!   * prove `Sum_k mult[k]/(alpha - dict_fold[k]) - 1/(alpha - q_fold) = 0` with
//!     the GKR fraction tree ([`super::logup_gkr`]): the `D` dictionary rows form
//!     one leaf group (numerator `mult[k]`, denominator `alpha - dict_fold[k]`) and
//!     the public target folds in as ONE extra signed leaf `(-1, alpha - q_fold)`,
//!     so the tree's root claim is always `P = 0`. The prover reveals only the root
//!     `(P, Q)` and a short per-layer sumcheck; NO per-row reciprocal MSM.
//!
//! Column length vs. leaf count. The target's extra leaf pushes the leaf count to
//! `next_pow2(D + 1) = 2D` (`D` is a committed power of two), but the dictionary
//! columns stay committed at length `D` (unchanged; root-anchored in the BGP
//! consumer). The prover feeds the tree the `2D` ZERO-EXTENSIONS of the data
//! columns; the verifier opens the length-`D` committed columns at the low `k` bits
//! `r_low` of the GKR leaf point and reconstructs each `2D` leaf-MLE as
//! `(1 - r_high)*open(r_low)`. A single batched Dory opening of `[dict_hash,
//! dict_id, mult]` at `r_low` binds all three to their root-anchored commitments.
//!
//! GAP-3 (target-pole soundness), enforced. The target leaf sits at position `D`,
//! which lives in the zero-extended HIGH half `[D, 2D)`. The verifier reconstructs
//! the fold columns there as `(1 - r_high)*open(r_low)`, i.e. the MLE of the column
//! that is IDENTICALLY ZERO past the `D` committed rows. So `dict_fold[D] = 0` is
//! forced by the opening, and the target denominator is EXACTLY `alpha - q_fold` —
//! a prover CANNOT commit a nonzero `dict_hash[D]`/`dict_id[D]` to shift the pole
//! and cancel a genuine dictionary pole (the classic forge). Because position `D`
//! is outside the committed length-`D` domain, this is enforced STRUCTURALLY by the
//! leaf tie + Dory binding rather than by a co-running Identity claim over the
//! committed columns (there is nothing there for such a claim to constrain); the
//! structural enforcement is strictly stronger — the forge is not even expressible.
//!
//! Soundness: at random `alpha` (drawn AFTER `dict`/`mult`/`beta` are committed),
//! `Sum_k mult[k]/(alpha - dict_fold[k]) = 1/(alpha - q_fold)` holds as a rational
//! identity iff `q_fold` is in `{dict_fold[k]}` with total multiplicity exactly 1
//! (RHS is a single simple pole with residue 1, so the residue-at-infinity forces
//! `Sum mult = 1`). `beta` fingerprints `(hash, id)`, so it binds the PAIR, not
//! hash-or-id alone. Clean-room from the log-derivative identity; no COSL code.

use std::io;

use ark_bn254::Fr;
use ark_ff::PrimeField;
use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::transcript::Transcript;

use super::argument::{
    mle_full_eval, verify_column_openings, ColumnOpening, StoreProverContext, Transcript254,
};
#[cfg(test)]
use super::argument::{commit_column, prove_column_openings};
use super::invalid;
use super::logup_gkr::{
    bind_and_verify_fraction_tree, check_leaf_tie, prove_fraction_sum_zero, FractionBuilder,
    FractionInstance, GkrFractionProof,
};
use super::CanonicalObject;

// Column ordering for the dictionary-membership data columns `[hash, id, mult]`.
// The reciprocal witness `dstar` is GONE: the LogUp-GKR migration proves the same
// log-derivative sum via a fraction tree over uncommitted intermediates, so no
// reciprocal column is committed anywhere.
const COL_HASH: usize = 0;
const COL_ID: usize = 1;
const COL_MULT: usize = 2;

const LABEL_BETA: &[u8] = b"pow/dict/fingerprint-beta";
const LABEL_ALPHA: &[u8] = b"pow/dict/reciprocal-alpha";
// Fiat-Shamir binding of the PUBLIC query pair `(q_hash, id)` BEFORE beta/alpha.
// Without this the folded target `q_fold = beta*q_hash + id` is not pinned to the
// challenge, enabling a prob-1 standalone forge (`id' = id_i + beta*(h_i - q)`
// yields the same `q_fold`). Both prover and verifier absorb the pair here,
// symmetrically, right after the data-column commitments and before `beta`.
const LABEL_QUERY_HASH: &[u8] = b"pow/dict/query-hash";
const LABEL_QUERY_ID: &[u8] = b"pow/dict/query-id";

#[inline]
fn f0() -> ArkFr {
    ArkFr::zero()
}
#[inline]
fn f1() -> ArkFr {
    ArkFr::one()
}

/// Deterministic hash-to-field: `Blake2b-512(term)` reduced modulo the BN254
/// scalar field. The verifier applies the SAME function to the public query
/// constant, so `q_hash = hash_term(constant)`. (Blake2b-512's 64-byte digest
/// reduced mod the ~254-bit field has negligible bias; the exact digest width is
/// not soundness-relevant as long as it is collision-resistant and applied
/// identically on both sides.)
pub fn hash_term(term: &[u8]) -> ArkFr {
    use blake2::{Blake2b512, Digest};
    let mut hasher = Blake2b512::new();
    hasher.update(term);
    let digest = hasher.finalize();
    ArkFr(Fr::from_le_bytes_mod_order(digest.as_slice()))
}

/// Deterministic hash-to-field for a canonical OBJECT (which shares its id space
/// with subjects: a subject constant is a `Node(iri)` too). Same Blake2b-512 ->
/// `Fr::from_le_bytes_mod_order` primitive as `hash_term`, but over DOMAIN-TAGGED
/// bytes so it is collision-separated from `hash_term` (predicate hashing) and so
/// a `Node` can never collide with a `Value`:
///   * `Node(iri)`            -> `Blake2b512(0x00 ++ iri)`
///   * `Value{datatype, lex}` -> `Blake2b512(0x01 ++ (datatype as u8)
///                                 ++ (lex.len() as u64, big-endian) ++ lex)`
/// The fixed one-byte tag separates the two variants; the fixed one-byte datatype
/// discriminant (`CanonicalDatatype` is `#[repr(u8)]`) followed by an 8-byte
/// length prefix makes `(datatype, lexical)` unambiguously parseable, so no two
/// distinct values share a preimage. CRITICAL invariant: the SAME node IRI hashes
/// identically whether it appears as a subject constant or an object constant
/// (both are `Node(iri)`), because they resolve through the one shared id. The
/// verifier applies this SAME function to the PUBLIC query constant, so it must be
/// deterministic and total.
pub fn hash_object(obj: &CanonicalObject) -> ArkFr {
    use blake2::{Blake2b512, Digest};
    let mut hasher = Blake2b512::new();
    match obj {
        CanonicalObject::Node(iri) => {
            hasher.update([0x00u8]);
            hasher.update(iri);
        }
        CanonicalObject::Value { datatype, lexical } => {
            hasher.update([0x01u8]);
            hasher.update([*datatype as u8]);
            hasher.update((lexical.len() as u64).to_be_bytes());
            hasher.update(lexical);
        }
    }
    let digest = hasher.finalize();
    ArkFr(Fr::from_le_bytes_mod_order(digest.as_slice()))
}

/// A dictionary-membership proof: the GKR fraction-sum-zero proof for
/// `Sum mult/(alpha - dict_fold) - 1/(alpha - q_fold) = 0`, the claimed evaluations
/// of the three DATA columns at `r_low` (the low `k` bits of the GKR leaf point),
/// one batched Dory opening of those columns at `r_low`, and the multiplicity
/// commitment the verifier binds. NO reciprocal (`dstar`) column exists anymore.
#[derive(Clone, Debug)]
pub struct DictProof {
    pub gkr: GkrFractionProof,
    pub col_evals: Vec<ArkFr>,
    pub opening: ColumnOpening,
    pub mult_commitment: ArkGT,
}

/// The PUBLIC fraction-sum-zero instance for a length-`dict_len` dictionary:
///   * one leaf group over the `dict_len` committed dictionary rows, numerator
///     `mult[k]`, denominator `alpha - (dict_id[k] + beta*dict_hash[k])`;
///   * one public target leaf `(-1, alpha - q_fold)`.
/// The leaf count pads to `next_pow2(dict_len + 1) = 2*dict_len`; the target lands
/// at position `dict_len` (the first, structurally-zero, high-half leaf). Built
/// IDENTICALLY by prover and verifier from public inputs only.
fn dict_fraction_instance(
    alpha: ArkFr,
    beta: ArkFr,
    q_fold: ArkFr,
    dict_len: usize,
) -> FractionInstance {
    // Fold order [dict_id, dict_hash] gives `dict_id + beta*dict_hash`, matching the
    // public query fold `q_fold = beta*q_hash + id` assembled in `query_fold`.
    let mut builder = FractionBuilder::new(alpha, beta, vec![COL_ID, COL_HASH], 3);
    builder.add_mult_group(f1(), COL_MULT, dict_len);
    builder.add_target(f0() - f1(), q_fold);
    builder.finish()
}

/// Zero-extend a length-`D` column to the fraction leaf count `to_len` (`= 2D`).
/// The high half `[D, 2D)` is all zero, so its multilinear extension at a point
/// `(r_low, r_high)` is `(1 - r_high) * MLE_D(col, r_low)` — the exact factor the
/// verifier applies when reconstructing the leaf-MLEs from a length-`D` opening.
fn zero_extend(col: &[ArkFr], to_len: usize) -> Vec<ArkFr> {
    let mut extended = vec![f0(); to_len];
    extended[..col.len()].copy_from_slice(col);
    extended
}

/// The PUBLIC dictionary/query fold `q_fold = beta*q_hash + id`.
fn query_fold(beta: ArkFr, q_hash: ArkFr, id: ArkFr) -> ArkFr {
    beta * q_hash + id
}

fn check_dimensions(dict_hash: &[ArkFr], dict_id: &[ArkFr]) -> io::Result<usize> {
    let size = dict_hash.len();
    if !size.is_power_of_two() {
        return Err(invalid("dictionary column length must be a power of two"));
    }
    if dict_id.len() != size {
        return Err(invalid("dictionary hash/id column length mismatch"));
    }
    let k = size.trailing_zeros() as usize;
    if k == 0 {
        return Err(invalid("dictionary has too few rows for a membership proof"));
    }
    Ok(k)
}

/// Prove that `(q_hash, id)` is a genuine row of the committed dictionary.
pub fn prove_dict_membership(
    dict_hash: &[ArkFr],
    dict_id: &[ArkFr],
    com_dict_hash: &ArkGT,
    com_dict_id: &ArkGT,
    q_hash: ArkFr,
    id: ArkFr,
    transcript: &mut Transcript254,
) -> io::Result<DictProof> {
    prove_dict_membership_with_context(
        &StoreProverContext::cpu(),
        dict_hash,
        dict_id,
        com_dict_hash,
        com_dict_id,
        q_hash,
        id,
        transcript,
    )
}

/// As [`prove_dict_membership`], with an explicit prover-arithmetic context.
#[allow(clippy::too_many_arguments)]
pub fn prove_dict_membership_with_context(
    context: &StoreProverContext<'_>,
    dict_hash: &[ArkFr],
    dict_id: &[ArkFr],
    com_dict_hash: &ArkGT,
    com_dict_id: &ArkGT,
    q_hash: ArkFr,
    id: ArkFr,
    transcript: &mut Transcript254,
) -> io::Result<DictProof> {
    let k = check_dimensions(dict_hash, dict_id)?;
    let size = dict_hash.len();

    // Multiplicity indicator: 1 at the (unique) row matching the PAIR, else 0.
    let mult: Vec<ArkFr> = (0..size)
        .map(|i| {
            if dict_hash[i] == q_hash && dict_id[i] == id {
                f1()
            } else {
                f0()
            }
        })
        .collect();
    let mult_commitment = context.commit_column(&mult)?;

    // Bind the DATA columns, THEN draw beta/alpha. Caller contract (Inc 0 audit):
    // `alpha` MUST be drawn after the column commitments are bound so the fraction
    // leaves are fixed before any per-layer GKR challenge. There is no reciprocal
    // commitment anymore, so this is the last binding before the fold challenges.
    transcript.append_group(b"pow/dict/com-hash", com_dict_hash);
    transcript.append_group(b"pow/dict/com-id", com_dict_id);
    transcript.append_group(b"pow/dict/com-mult", &mult_commitment);
    // Bind the PUBLIC query pair `(q_hash, id)` BEFORE drawing beta/alpha so the
    // folded target `q_fold = beta*q_hash + id` is fixed against the challenge.
    // MUST append byte-identically to the verifier (see `verify_dict_membership`).
    transcript.append_field(LABEL_QUERY_HASH, &q_hash);
    transcript.append_field(LABEL_QUERY_ID, &id);
    let beta = transcript.challenge_scalar(LABEL_BETA);
    let alpha = transcript.challenge_scalar(LABEL_ALPHA);
    let q_fold = query_fold(beta, q_hash, id);

    // Prove the fraction sum over the `2D` zero-extended data columns. The fraction
    // instance folds the dictionary rows and the public target leaf; the tree lands
    // at `r_leaf` (length `k + 1`), whose low `k` bits are the opening point.
    let inst = dict_fraction_instance(alpha, beta, q_fold, size);
    let leaf_count = inst.size();
    let extended = vec![
        zero_extend(dict_hash, leaf_count),
        zero_extend(dict_id, leaf_count),
        zero_extend(&mult, leaf_count),
    ];
    let (gkr, r_leaf) = prove_fraction_sum_zero(&inst, &extended, transcript)?;

    // Single batched Dory opening of the length-`D` data columns at `r_low`, bound
    // to their (root-anchored, in the BGP consumer) commitments on this transcript.
    let r_low = &r_leaf[..k];
    let columns = vec![dict_hash.to_vec(), dict_id.to_vec(), mult];
    let col_evals: Vec<ArkFr> = columns.iter().map(|c| mle_full_eval(c, r_low)).collect();
    let opening = context.prove_column_openings(&columns, r_low, transcript)?;

    Ok(DictProof {
        gkr,
        col_evals,
        opening,
        mult_commitment,
    })
}

/// Verify a dictionary-membership proof against the committed dictionary columns
/// and the PUBLIC query pair `(q_hash, id)`.
pub fn verify_dict_membership(
    com_dict_hash: &ArkGT,
    com_dict_id: &ArkGT,
    q_hash: ArkFr,
    id: ArkFr,
    proof: &DictProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    // The GKR tree has `k + 1` layers (`2D` leaves for a length-`D = 2^k` dict), so
    // `k = num_vars - 1` and the committed columns have length `D = 2^k`.
    let num_vars = proof.gkr.num_vars();
    if num_vars < 2 || num_vars >= usize::BITS as usize {
        return Err(invalid("dictionary proof has an implausible variable count"));
    }
    let k = num_vars - 1;
    let dict_len = 1usize << k;
    if proof.col_evals.len() != 3 {
        return Err(invalid("dictionary proof has the wrong number of column evaluations"));
    }

    // Same binding order as the prover: DATA columns, THEN beta/alpha.
    transcript.append_group(b"pow/dict/com-hash", com_dict_hash);
    transcript.append_group(b"pow/dict/com-id", com_dict_id);
    transcript.append_group(b"pow/dict/com-mult", &proof.mult_commitment);
    // Bind the PUBLIC query pair `(q_hash, id)` BEFORE drawing beta/alpha. MUST
    // append byte-identically to the prover (see `prove_dict_membership_with_context`).
    transcript.append_field(LABEL_QUERY_HASH, &q_hash);
    transcript.append_field(LABEL_QUERY_ID, &id);
    let beta = transcript.challenge_scalar(LABEL_BETA);
    let alpha = transcript.challenge_scalar(LABEL_ALPHA);
    let q_fold = query_fold(beta, q_hash, id);

    // Bind the public instance and verify the GKR tree -> the leaf point + claims.
    let inst = dict_fraction_instance(alpha, beta, q_fold, dict_len);
    let (r_leaf, p_claim, q_claim) =
        bind_and_verify_fraction_tree(&inst, &proof.gkr, transcript)?;

    // Open the length-`D` committed columns at `r_low`, bound to their commitments.
    let r_low = &r_leaf[..k];
    let r_high = r_leaf[k];
    verify_column_openings(
        &[*com_dict_hash, *com_dict_id, proof.mult_commitment],
        r_low,
        &proof.col_evals,
        &proof.opening,
        transcript,
    )?;

    // Reconstruct each `2D` leaf-MLE from the length-`D` opening via the
    // `(1 - r_high)` zero-extension factor, then tie to the GKR-reduced leaf claims.
    // This is the GAP-3 enforcement: the target leaf at position `D` lives in the
    // high half, so its fold columns reconstruct to ZERO (a prover cannot commit a
    // nonzero value there); its pole is therefore exactly `alpha - q_fold` and
    // cannot be shifted to forge membership of a non-member.
    let one_minus_high = f1() - r_high;
    let leaf_evals: Vec<ArkFr> = proof
        .col_evals
        .iter()
        .map(|eval| one_minus_high * *eval)
        .collect();
    check_leaf_tie(&inst, &leaf_evals, p_claim, q_claim, &r_leaf)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transcript() -> Transcript254 {
        Transcript254::new(b"parture-dict-test")
    }

    // A dictionary of `terms`: dict_hash[k] = hash_term(term), dict_id[k] = k+1
    // (ids start at 1). Padded to a power of two with zeros.
    fn dictionary(terms: &[&str]) -> (Vec<ArkFr>, Vec<ArkFr>, Vec<(ArkFr, u64)>) {
        let d = terms.len();
        let padded = d.next_power_of_two().max(2);
        let mut dict_hash = vec![f0(); padded];
        let mut dict_id = vec![f0(); padded];
        let mut entries = Vec::new();
        for (k, term) in terms.iter().enumerate() {
            let h = hash_term(term.as_bytes());
            let id = (k + 1) as u64;
            dict_hash[k] = h;
            dict_id[k] = ArkFr::from_u64(id);
            entries.push((h, id));
        }
        (dict_hash, dict_id, entries)
    }

    fn commit(column: &[ArkFr]) -> ArkGT {
        commit_column(column).unwrap()
    }

    fn run(
        dict_hash: &[ArkFr],
        dict_id: &[ArkFr],
        q_hash: ArkFr,
        id: ArkFr,
        tamper: impl FnOnce(&mut DictProof),
    ) -> io::Result<()> {
        let com_h = commit(dict_hash);
        let com_i = commit(dict_id);
        let mut pt = transcript();
        let mut proof =
            prove_dict_membership(dict_hash, dict_id, &com_h, &com_i, q_hash, id, &mut pt)?;
        tamper(&mut proof);
        let mut vt = transcript();
        verify_dict_membership(&com_h, &com_i, q_hash, id, &proof, &mut vt)
    }

    #[test]
    fn correct_membership_verifies() {
        let (dict_hash, dict_id, entries) = dictionary(&["alice", "bob", "carol", "dave", "erin"]);
        // Prove the "carol" entry.
        let (q_hash, id) = entries[2];
        assert!(run(&dict_hash, &dict_id, q_hash, ArkFr::from_u64(id), |_| {}).is_ok());
    }

    #[test]
    fn wrong_id_for_a_real_term_is_rejected() {
        let (dict_hash, dict_id, entries) = dictionary(&["alice", "bob", "carol"]);
        let (q_hash, _id) = entries[1]; // bob's hash
        let wrong_id = ArkFr::from_u64(99); // but a wrong id
        assert!(run(&dict_hash, &dict_id, q_hash, wrong_id, |_| {}).is_err());
    }

    #[test]
    fn absent_term_is_rejected() {
        let (dict_hash, dict_id, _entries) = dictionary(&["alice", "bob", "carol"]);
        let q_hash = hash_term(b"mallory"); // not in the dictionary
        let id = ArkFr::from_u64(2);
        assert!(run(&dict_hash, &dict_id, q_hash, id, |_| {}).is_err());
    }

    #[test]
    fn mismatched_pair_is_rejected() {
        // q_hash from term A, id from term B: beta binds the pair, so reject.
        let (dict_hash, dict_id, entries) = dictionary(&["alice", "bob", "carol", "dave"]);
        let (hash_a, _id_a) = entries[0]; // alice's hash
        let (_hash_b, id_b) = entries[1]; // bob's id
        assert!(run(&dict_hash, &dict_id, hash_a, ArkFr::from_u64(id_b), |_| {}).is_err());
    }

    #[test]
    fn query_fold_collision_forge_is_rejected() {
        // Regression for the Fiat-Shamir soundness bug: `(q_hash, id)` was NOT
        // absorbed before `beta`/`alpha`, so the folded target
        //   q_fold = beta*q_hash + id
        // was not pinned to the challenge. That enabled a prob-1 standalone
        // forge: take an HONEST proof for a real row `(h_i, id_i)` (which fixes
        //   q_fold = beta*h_i + id_i),
        // then present a DIFFERENT query hash `q` with
        //   id' = id_i + beta*(h_i - q)
        // so that beta*q + id' == beta*h_i + id_i == the honest q_fold. Pre-fix
        // the verifier drew the SAME `beta` (query not bound), reconstructed the
        // IDENTICAL fraction instance, and ACCEPTED the forged pair `(q, id')`.
        //
        // Form used: the CONCRETE forge. We reconstruct the prover's honestly
        // derived `beta` by replaying the fixed transcript prefix (data-column
        // commitments, then the bound query pair), build `id'`, and assert the
        // fixed verifier REJECTS `(q, id')` -- because it now binds `(q, id')`
        // before `beta`, derives a DIFFERENT `beta'`, and no longer reconstructs
        // the instance the honest proof was produced against.
        let (dict_hash, dict_id, entries) = dictionary(&["alice", "bob", "carol", "dave"]);
        let (h_i, id_i_u64) = entries[1]; // the real "bob" row
        let id_i = ArkFr::from_u64(id_i_u64);

        let com_h = commit(&dict_hash);
        let com_i = commit(&dict_id);

        // Honest proof for the real row `(h_i, id_i)`.
        let mut pt = transcript();
        let proof =
            prove_dict_membership(&dict_hash, &dict_id, &com_h, &com_i, h_i, id_i, &mut pt).unwrap();

        // Positive control: the honest pair verifies.
        let mut vt_ok = transcript();
        assert!(verify_dict_membership(&com_h, &com_i, h_i, id_i, &proof, &mut vt_ok).is_ok());

        // Reconstruct the prover's `beta` EXACTLY as the fixed opener derives it:
        // com-hash, com-id, com-mult, then the bound query pair (h_i, id_i),
        // then the beta challenge.
        let mut bt = transcript();
        bt.append_group(b"pow/dict/com-hash", &com_h);
        bt.append_group(b"pow/dict/com-id", &com_i);
        bt.append_group(b"pow/dict/com-mult", &proof.mult_commitment);
        bt.append_field(LABEL_QUERY_HASH, &h_i);
        bt.append_field(LABEL_QUERY_ID, &id_i);
        let beta = bt.challenge_scalar(LABEL_BETA);

        // Forge: a DIFFERENT query hash `q` with a matching `q_fold`.
        let q = hash_term(b"mallory"); // not the bob hash
        assert_ne!(q, h_i, "the forged query hash must differ from the real row");
        let id_prime = id_i + beta * (h_i - q);
        assert_ne!(id_prime, id_i, "the forged id must differ (beta*(h_i-q) != 0)");
        // Sanity: under the prover's `beta` the forge preserves the fold value.
        assert_eq!(beta * q + id_prime, beta * h_i + id_i);

        // The fixed verifier binds `(q, id')` before `beta`, so it re-derives a
        // different challenge and REJECTS the forged pair against the honest proof.
        let mut vt_bad = transcript();
        assert!(
            verify_dict_membership(&com_h, &com_i, q, id_prime, &proof, &mut vt_bad).is_err(),
            "query-fold collision forge must be rejected"
        );
    }

    #[test]
    fn hash_term_is_deterministic_and_separates() {
        assert_eq!(hash_term(b"predicate:knows"), hash_term(b"predicate:knows"));
        assert_ne!(hash_term(b"predicate:knows"), hash_term(b"predicate:likes"));
        assert_ne!(hash_term(b""), hash_term(b"x"));
    }

    #[test]
    fn hash_object_is_deterministic_and_separates() {
        use crate::proof::CanonicalDatatype;

        let node = |iri: &[u8]| CanonicalObject::Node(iri.to_vec());
        let value = |dt: CanonicalDatatype, lex: &[u8]| CanonicalObject::Value {
            datatype: dt,
            lexical: lex.to_vec(),
        };

        // Determinism: the same object hashes identically every time.
        assert_eq!(hash_object(&node(b"iri:Person")), hash_object(&node(b"iri:Person")));
        assert_eq!(
            hash_object(&value(CanonicalDatatype::String, b"hi")),
            hash_object(&value(CanonicalDatatype::String, b"hi"))
        );

        // The SAME node IRI as a subject constant and as an object constant hashes
        // identically (both are `Node(iri)`) — the shared-id invariant.
        assert_eq!(hash_object(&node(b"iri:x")), hash_object(&node(b"iri:x")));

        // Distinct nodes differ.
        assert_ne!(hash_object(&node(b"iri:x")), hash_object(&node(b"iri:y")));

        // Node vs Value are domain-separated even when the raw bytes coincide.
        assert_ne!(
            hash_object(&node(b"hi")),
            hash_object(&value(CanonicalDatatype::String, b"hi"))
        );

        // Same lexical, different datatype -> distinct.
        assert_ne!(
            hash_object(&value(CanonicalDatatype::String, b"hi")),
            hash_object(&value(CanonicalDatatype::AnyUri, b"hi"))
        );

        // Different lexical, same datatype -> distinct.
        assert_ne!(
            hash_object(&value(CanonicalDatatype::String, b"hi")),
            hash_object(&value(CanonicalDatatype::String, b"ho"))
        );

        // Length-prefix guards against a datatype/lexical boundary ambiguity: a
        // one-byte-longer lexical can never masquerade as a different value.
        assert_ne!(
            hash_object(&value(CanonicalDatatype::String, b"ab")),
            hash_object(&value(CanonicalDatatype::String, b"a"))
        );

        // Separated from the predicate hasher (`hash_term`) as well.
        assert_ne!(hash_object(&node(b"iri:x")), hash_term(b"iri:x"));
    }

    #[test]
    fn tampered_multiplicity_is_rejected() {
        let (dict_hash, dict_id, entries) = dictionary(&["alice", "bob", "carol", "dave"]);
        let (q_hash, id) = entries[2];
        let id = ArkFr::from_u64(id);

        // Honest multiplicity commitment for a DIFFERENT (empty) selection, so the
        // committed mult no longer matches the proof's claimed mult evaluation ->
        // the Dory opening (which binds mult) fails.
        let bad_mult = commit(&vec![f0(); dict_hash.len()]);
        assert!(run(&dict_hash, &dict_id, q_hash, id, |proof| {
            proof.mult_commitment = bad_mult;
        })
        .is_err());

        // A witness with TWO ones (wrong multiplicity, Sum mult = 2) -> reject:
        // build it explicitly and drive prove/verify with that mult column.
        let size = dict_hash.len();
        let mut two_ones = vec![f0(); size];
        two_ones[2] = f1();
        two_ones[0] = f1(); // a spurious second 1
        assert!(prove_with_explicit_mult(&dict_hash, &dict_id, q_hash, id, &two_ones).is_err());

        // Right count but WRONG position (points at a non-matching row) -> reject.
        let mut wrong_pos = vec![f0(); size];
        wrong_pos[0] = f1();
        assert!(prove_with_explicit_mult(&dict_hash, &dict_id, q_hash, id, &wrong_pos).is_err());
    }

    // Prove/verify driving an EXPLICIT (possibly malicious) multiplicity column
    // through the honest LogUp-GKR prover flow.
    fn prove_with_explicit_mult(
        dict_hash: &[ArkFr],
        dict_id: &[ArkFr],
        q_hash: ArkFr,
        id: ArkFr,
        mult: &[ArkFr],
    ) -> io::Result<()> {
        let k = check_dimensions(dict_hash, dict_id)?;
        let com_h = commit(dict_hash);
        let com_i = commit(dict_id);
        let mult_commitment = commit(mult);

        let mut pt = transcript();
        pt.append_group(b"pow/dict/com-hash", &com_h);
        pt.append_group(b"pow/dict/com-id", &com_i);
        pt.append_group(b"pow/dict/com-mult", &mult_commitment);
        let beta = pt.challenge_scalar(LABEL_BETA);
        let alpha = pt.challenge_scalar(LABEL_ALPHA);
        let q_fold = query_fold(beta, q_hash, id);

        let inst = dict_fraction_instance(alpha, beta, q_fold, dict_hash.len());
        let leaf_count = inst.size();
        let extended = vec![
            zero_extend(dict_hash, leaf_count),
            zero_extend(dict_id, leaf_count),
            zero_extend(mult, leaf_count),
        ];
        let (gkr, r_leaf) = prove_fraction_sum_zero(&inst, &extended, &mut pt)?;
        let r_low = &r_leaf[..k];
        let columns = vec![dict_hash.to_vec(), dict_id.to_vec(), mult.to_vec()];
        let col_evals: Vec<ArkFr> = columns.iter().map(|c| mle_full_eval(c, r_low)).collect();
        let opening = prove_column_openings(&columns, r_low, &mut pt)?;
        let proof = DictProof {
            gkr,
            col_evals,
            opening,
            mult_commitment,
        };
        let mut vt = transcript();
        verify_dict_membership(&com_h, &com_i, q_hash, id, &proof, &mut vt)
    }

    // GAP-3 forge negative. A prover tries to prove membership of a NON-member
    // `q` by SHIFTING the public target pole: instead of the honest target leaf
    // `(-1, alpha - q_fold)` it forges a nonzero fold value at the target leaf
    // position `D` so the pole becomes `alpha - q_m` for a genuine member `m`, then
    // points `mult` at `m` so the two poles cancel and the fraction sum is zero.
    // The forge produces a VALID zero-sum GKR proof (root P = 0), but the verifier
    // reconstructs the fold at position `D` from the HONEST length-`D` opening via
    // the `(1 - r_high)` zero-extension — which is identically zero there — so the
    // reconstructed target pole is `alpha - q_fold != alpha - q_m` and the leaf tie
    // rejects. This is exactly the pole-shift attack the module comment warns about.
    #[test]
    fn gap3_forged_target_pole_is_rejected() {
        let (dict_hash, dict_id, entries) = dictionary(&["alice", "bob", "carol", "dave"]);
        // A full power-of-two dictionary (D == 4), so the target leaf is at the very
        // first high-half position D = 4 and there is no spare low-half padding row.
        let k = check_dimensions(&dict_hash, &dict_id).unwrap();
        let dict_len = dict_hash.len();
        assert_eq!(dict_len, 4);

        // The (non-member) query claims carol's id for a hash that is NOT in the dict.
        let (_carol_hash, carol_id) = entries[2];
        let q_hash = hash_term(b"mallory");
        let id = ArkFr::from_u64(carol_id);

        let com_h = commit(&dict_hash);
        let com_i = commit(&dict_id);
        // Honest multiplicity indicator points at the genuine `carol` row (index 2),
        // whose fold the forged target pole will be shifted to cancel.
        let mut mult = vec![f0(); dict_len];
        mult[2] = f1();
        let mult_commitment = commit(&mult);

        let mut pt = transcript();
        pt.append_group(b"pow/dict/com-hash", &com_h);
        pt.append_group(b"pow/dict/com-id", &com_i);
        pt.append_group(b"pow/dict/com-mult", &mult_commitment);
        let beta = pt.challenge_scalar(LABEL_BETA);
        let alpha = pt.challenge_scalar(LABEL_ALPHA);
        let q_fold = query_fold(beta, q_hash, id);
        let carol_fold = beta * dict_hash[2] + dict_id[2];

        // Build the honest instance, but FORGE the extended fold columns: put
        // `carol_fold - q_fold` into `dict_id` at the target leaf position D so the
        // target denominator `alpha - fold[D] - q_fold` becomes `alpha - carol_fold`.
        let inst = dict_fraction_instance(alpha, beta, q_fold, dict_len);
        let leaf_count = inst.size();
        assert_eq!(leaf_count, 2 * dict_len);
        let ext_hash = zero_extend(&dict_hash, leaf_count);
        let mut ext_id = zero_extend(&dict_id, leaf_count);
        let ext_mult = zero_extend(&mult, leaf_count);
        ext_id[dict_len] = carol_fold - q_fold; // the pole-shifting forgery
        let (gkr, r_leaf) =
            prove_fraction_sum_zero(&inst, &[ext_hash, ext_id, ext_mult], &mut pt).unwrap();
        // The forged leaves genuinely sum to zero (the forge "worked" at tree level).
        assert!(gkr.root_p.is_zero(), "forged fraction sum should be zero");

        // The Dory opening is over the HONEST length-D committed columns.
        let r_low = &r_leaf[..k];
        let columns = vec![dict_hash.to_vec(), dict_id.to_vec(), mult.clone()];
        let col_evals: Vec<ArkFr> = columns.iter().map(|c| mle_full_eval(c, r_low)).collect();
        let opening = prove_column_openings(&columns, r_low, &mut pt).unwrap();
        let proof = DictProof { gkr, col_evals, opening, mult_commitment };

        // The leaf tie reconstructs fold[D] = 0 (zero-extension) -> target pole is
        // alpha - q_fold, contradicting the forged tree -> REJECT.
        let mut vt = transcript();
        assert!(verify_dict_membership(&com_h, &com_i, q_hash, id, &proof, &mut vt).is_err());
    }

    // The migrated `DictProof` survives a wire round-trip and re-verifies.
    #[test]
    fn dict_proof_round_trips_through_wire() {
        use crate::proof::envelope::wire_roundtrip_dict_proof;
        let (dict_hash, dict_id, entries) = dictionary(&["alice", "bob", "carol", "dave", "erin"]);
        let (q_hash, id) = entries[3];
        let id = ArkFr::from_u64(id);
        let com_h = commit(&dict_hash);
        let com_i = commit(&dict_id);
        let mut pt = transcript();
        let proof =
            prove_dict_membership(&dict_hash, &dict_id, &com_h, &com_i, q_hash, id, &mut pt).unwrap();
        let decoded = wire_roundtrip_dict_proof(&proof).unwrap();
        let mut vt = transcript();
        assert!(verify_dict_membership(&com_h, &com_i, q_hash, id, &decoded, &mut vt).is_ok());
    }
}
