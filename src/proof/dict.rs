//! Q3 dictionary-binding LogUp membership gadget (blueprint §4), in isolation.
//!
//! Given committed dictionary columns `dict_hash[k]`, `dict_id[k]` (`k in [0, D)`,
//! padded to a power of two) and a PUBLIC query pair `(q_hash, id)` — where the
//! verifier computes `q_hash = hash_term(constant)` itself — this proves
//! `(q_hash, id)` is a genuine row of the dictionary, via single-element LogUp
//! membership. Q3b will commit the dictionary inside `LayerCommitment` and wire
//! this into `prove_layer_filter`; here it stands alone.
//!
//! Gadget:
//!   * fold each dict row `dict_fold[k] = beta*dict_hash[k] + dict_id[k]`; the
//!     query folds to the PUBLIC scalar `q_fold = beta*q_hash + id`.
//!   * commit a multiplicity indicator `mult[k]` (1 at the matching row, else 0)
//!     and the reciprocal witness `dstar[k] = 1/(alpha - dict_fold[k])`.
//!   * Identity claim (logup):  `dstar[k]*(alpha - dict_fold[k]) - 1 = 0`.
//!   * ZeroSum claim with a PUBLIC NONZERO target:
//!       `Sum_k mult[k]*dstar[k] = 1/(alpha - q_fold)`.
//!     The target `1/(alpha - q_fold)` is a public scalar the verifier computes;
//!     the batched sumcheck's initial claim is the rho-combination of the
//!     per-claim `target`s (see `SumcheckClaim::target`).
//!
//! Soundness: at random `alpha` (drawn after `mult`/`dict`/`beta` are committed,
//! and after `dstar`), `Sum_k mult[k]/(alpha - dict_fold[k]) = 1/(alpha - q_fold)`
//! holds as a rational identity iff `q_fold` is in `{dict_fold[k]}` with total
//! multiplicity exactly 1 (RHS is a single simple pole with residue 1, so the
//! residue-at-infinity forces `Sum mult = 1`). `beta` fingerprints `(hash, id)`,
//! so it binds the PAIR, not hash-or-id alone. Clean-room from the log-derivative
//! identity; no COSL code.

use std::io;

use ark_bn254::Fr;
use ark_ff::PrimeField;
use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::transcript::Transcript;

use super::argument::{
    commit_column, prove_column_openings, prove_sumcheck, verify_column_openings, verify_sumcheck,
    ClaimShape, ColumnOpening, Expr, Operand, SumcheckClaim, SumcheckProof, SumcheckShape, Term,
    Transcript254,
};
use super::invalid;
use super::logup::{reciprocal_claim, reciprocal_witness, rlc_fold};

// Column ordering for the dictionary membership sumcheck.
const COL_HASH: usize = 0;
const COL_ID: usize = 1;
const COL_MULT: usize = 2;
const COL_DSTAR: usize = 3;

const LABEL_BETA: &[u8] = b"pow/dict/fingerprint-beta";
const LABEL_ALPHA: &[u8] = b"pow/dict/reciprocal-alpha";

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

/// A dictionary-membership proof: the batched sumcheck over the reciprocal (C)
/// and public-target LogUp (E) claims, one batched Dory opening of the four
/// columns at `r`, and the internal witness commitments the verifier binds.
#[derive(Clone, Debug)]
pub struct DictProof {
    pub sumcheck: SumcheckProof,
    pub opening: ColumnOpening,
    pub mult_commitment: ArkGT,
    pub dstar_commitment: ArkGT,
}

/// The two claims over columns `[dict_hash, dict_id, mult, dstar]`:
///   C (Identity): `dstar*(alpha - (beta*dict_hash + dict_id)) - 1 = 0`
///   E (ZeroSum, target `1/(alpha - q_fold)`): `Sum mult*dstar = target`
fn dict_shape(k: usize, alpha: ArkFr, beta: ArkFr, target: ArkFr) -> SumcheckShape {
    // reciprocal fold order [dict_id, dict_hash] gives dict_id + beta*dict_hash.
    let claim_c = reciprocal_claim(COL_DSTAR, &[COL_ID, COL_HASH], alpha, beta);
    let claim_e = SumcheckClaim {
        shape: ClaimShape::ZeroSum,
        expr: Expr {
            terms: vec![Term {
                coeff: f1(),
                operands: vec![Operand::Column(COL_MULT), Operand::Column(COL_DSTAR)],
            }],
        },
        target,
    };
    SumcheckShape {
        num_vars: k,
        num_columns: 4,
        chi_bounds: Vec::new(),
        claims: vec![claim_c, claim_e],
    }
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
    let mult_commitment = commit_column(&mult)?;

    // Round 1: bind the dictionary + multiplicity, then draw beta/alpha.
    transcript.append_group(b"pow/dict/com-hash", com_dict_hash);
    transcript.append_group(b"pow/dict/com-id", com_dict_id);
    transcript.append_group(b"pow/dict/com-mult", &mult_commitment);
    let beta = transcript.challenge_scalar(LABEL_BETA);
    let alpha = transcript.challenge_scalar(LABEL_ALPHA);

    // Reciprocal witness over the beta-fingerprinted dictionary rows.
    let dict_fold = rlc_fold(beta, &[dict_id, dict_hash]);
    let dstar = reciprocal_witness(alpha, &dict_fold);
    let dstar_commitment = commit_column(&dstar)?;

    // Round 2: bind the reciprocal commitment, form the public target, run the
    // batched sumcheck, and open all four columns at r.
    transcript.append_group(b"pow/dict/com-dstar", &dstar_commitment);
    let target = query_target(alpha, beta, q_hash, id)?;
    let shape = dict_shape(k, alpha, beta, target);
    let columns = vec![dict_hash.to_vec(), dict_id.to_vec(), mult, dstar];
    let (sumcheck, r, _evals) = prove_sumcheck(&shape, &columns, transcript)?;
    let opening = prove_column_openings(&columns, &r, transcript)?;

    Ok(DictProof {
        sumcheck,
        opening,
        mult_commitment,
        dstar_commitment,
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
    let k = proof.sumcheck.round_polys.len();
    if k == 0 || k >= usize::BITS as usize {
        return Err(invalid("dictionary proof has an implausible variable count"));
    }

    transcript.append_group(b"pow/dict/com-hash", com_dict_hash);
    transcript.append_group(b"pow/dict/com-id", com_dict_id);
    transcript.append_group(b"pow/dict/com-mult", &proof.mult_commitment);
    let beta = transcript.challenge_scalar(LABEL_BETA);
    let alpha = transcript.challenge_scalar(LABEL_ALPHA);
    transcript.append_group(b"pow/dict/com-dstar", &proof.dstar_commitment);

    let target = query_target(alpha, beta, q_hash, id)?;
    let shape = dict_shape(k, alpha, beta, target);
    let (r, evals) = verify_sumcheck(&shape, &proof.sumcheck, transcript)?;
    verify_column_openings(
        &[
            *com_dict_hash,
            *com_dict_id,
            proof.mult_commitment,
            proof.dstar_commitment,
        ],
        &r,
        &evals,
        &proof.opening,
        transcript,
    )
}

/// The public ZeroSum target `1/(alpha - (beta*q_hash + id))`.
fn query_target(alpha: ArkFr, beta: ArkFr, q_hash: ArkFr, id: ArkFr) -> io::Result<ArkFr> {
    let q_fold = beta * q_hash + id;
    (alpha - q_fold)
        .inv()
        .ok_or_else(|| invalid("query fold collides with the reciprocal challenge"))
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
    fn hash_term_is_deterministic_and_separates() {
        assert_eq!(hash_term(b"predicate:knows"), hash_term(b"predicate:knows"));
        assert_ne!(hash_term(b"predicate:knows"), hash_term(b"predicate:likes"));
        assert_ne!(hash_term(b""), hash_term(b"x"));
    }

    #[test]
    fn tampered_multiplicity_is_rejected() {
        let (dict_hash, dict_id, entries) = dictionary(&["alice", "bob", "carol", "dave"]);
        let (q_hash, id) = entries[2];
        let id = ArkFr::from_u64(id);

        // Honest multiplicity commitment for a DIFFERENT (empty) selection, so the
        // committed mult no longer matches the sumcheck's claimed values -> the
        // Dory opening (which binds mult) fails.
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

    // Prove/verify driving an EXPLICIT (possibly malicious) multiplicity column.
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
        let dict_fold = rlc_fold(beta, &[dict_id, dict_hash]);
        let dstar = reciprocal_witness(alpha, &dict_fold);
        let dstar_commitment = commit(&dstar);
        pt.append_group(b"pow/dict/com-dstar", &dstar_commitment);
        let target = query_target(alpha, beta, q_hash, id)?;
        let shape = dict_shape(k, alpha, beta, target);
        let columns = vec![
            dict_hash.to_vec(),
            dict_id.to_vec(),
            mult.to_vec(),
            dstar,
        ];
        let (sumcheck, r, _evals) = prove_sumcheck(&shape, &columns, &mut pt)?;
        let opening = prove_column_openings(&columns, &r, &mut pt)?;
        let proof = DictProof {
            sumcheck,
            opening,
            mult_commitment,
            dstar_commitment,
        };
        let mut vt = transcript();
        verify_dict_membership(&com_h, &com_i, q_hash, id, &proof, &mut vt)
    }
}
