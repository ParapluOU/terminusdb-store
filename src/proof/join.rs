//! Q2 self-join SOUNDNESS gadget (Phase 2c-1), in isolation over a committed
//! relation. Reuses the LogUp membership pattern (proof::logup + the Q3
//! single-element pattern generalized to multiplicities).
//!
//! Model: a 2-atom self-join `t(?x, P, ?y), t(?y, P, ?z)` reduces to filtering by
//! `P` once into a candidate relation `F = (Sf, Of)` (Q1, done), then self-joining
//! `F` on `Of == Sf` (the shared variable `y`) to output `(x, y, z)`. This gadget
//! proves the SOUNDNESS of a claimed output: every output row is a genuine join of
//! two `F` rows on `y`. Concretely, for every `k < m_J`:
//!   * `(X_out_k, Y_out_k)` is a row of `F` (left atom), AND
//!   * `(Y_out_k, Z_out_k)` is a row of `F` (right atom).
//! The shared committed `Y_out` column links the two memberships, so each output
//! row is a valid `(x, y, z)`. This forbids FABRICATED rows; it does NOT yet
//! forbid OMITTED rows (completeness is Phase 2c-2 — see the known-gap test).
//!
//! Both memberships are LogUp lookups against the SAME `F` fingerprint
//! `f_fold_a = beta*Sf_a + Of_a`: the left folds output rows as
//! `l_fold_k = beta*X_k + Y_k` and the right as `r_fold_k = beta*Y_k + Z_k`, and
//! each is proved to lie in `{f_fold_a}` with prover-committed multiplicities.
//! One committed `F`, two membership instances. Clean-room from the log-derivative
//! identity; no COSL code.

use std::io;

use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::transcript::Transcript;

use super::argument::{
    commit_column, prove_column_openings, prove_sumcheck, verify_column_openings, verify_sumcheck,
    ColumnOpening, Operand, SumcheckProof, SumcheckShape, Transcript254,
};
use super::invalid;
use super::logup::{multiset_link_claim, reciprocal_claim, reciprocal_witness, rlc_fold};

// Column ordering for the batched self-join sumcheck.
const COL_SF: usize = 0;
const COL_OF: usize = 1;
const COL_X: usize = 2;
const COL_Y: usize = 3;
const COL_Z: usize = 4;
const COL_MULT_L: usize = 5;
const COL_MULT_R: usize = 6;
const COL_DSTAR_F: usize = 7;
const COL_CSTAR_L: usize = 8;
const COL_CSTAR_R: usize = 9;
const NUM_COLUMNS: usize = 10;
// chi_bounds index for chi_{m_J} (the output length).
const CHI_MJ: usize = 0;

const LABEL_BETA: &[u8] = b"pow/join/fingerprint-beta";
const LABEL_ALPHA: &[u8] = b"pow/join/reciprocal-alpha";

#[inline]
fn f0() -> ArkFr {
    ArkFr::zero()
}

/// A self-join soundness proof: the batched sumcheck over the three reciprocal
/// (Identity) claims and the two LogUp-link (ZeroSum) claims, one batched Dory
/// opening over all ten columns, and the internal witness commitments the
/// verifier binds (`Com(X/Y/Z)` are returned separately as the query output).
#[derive(Clone, Debug)]
pub struct JoinSoundnessProof {
    pub sumcheck: SumcheckProof,
    pub opening: ColumnOpening,
    pub mult_l_commitment: ArkGT,
    pub mult_r_commitment: ArkGT,
    pub dstar_f_commitment: ArkGT,
    pub cstar_l_commitment: ArkGT,
    pub cstar_r_commitment: ArkGT,
}

/// The common power-of-two domain for a relation of `mu` rows and an output of
/// `m_j` rows (both zero-padded onto it); `>= 2` so the sumcheck has a variable.
pub(crate) fn join_padded_len(mu: usize, m_j: usize) -> io::Result<usize> {
    mu.max(m_j)
        .checked_next_power_of_two()
        .ok_or_else(|| invalid("join domain size overflow"))
        .map(|n| n.max(2))
}

fn padded(column: &[ArkFr], n: usize) -> Vec<ArkFr> {
    let mut out = column.to_vec();
    out.resize(n, f0());
    out
}

/// The multiplicity column over `F`: `mult[a] = |{ k : (out_a[k], out_b[k]) ==
/// (Sf_a, Of_a) }|` (raw row equality), as a length-`n` field vector (0 on padding
/// `F` rows). Assumes `F` rows are distinct (true for a filtered relation: a fixed
/// predicate makes `(subject, object)` unique).
pub(crate) fn multiplicity_over_f(
    sf: &[ArkFr],
    of: &[ArkFr],
    out_a: &[ArkFr],
    out_b: &[ArkFr],
    n: usize,
) -> Vec<ArkFr> {
    let mut mult = vec![f0(); n];
    for a in 0..sf.len() {
        let mut count = 0u64;
        for k in 0..out_a.len() {
            if out_a[k] == sf[a] && out_b[k] == of[a] {
                count += 1;
            }
        }
        mult[a] = ArkFr::from_u64(count);
    }
    mult
}

/// The five claims over columns
/// `[Sf, Of, X, Y, Z, mult_L, mult_R, dstar_F, cstar_L, cstar_R]`:
///   * Identity: `dstar_F*(alpha - (beta*Sf + Of)) - 1 = 0`
///   * Identity: `cstar_L*(alpha - (beta*X  + Y )) - 1 = 0`
///   * Identity: `cstar_R*(alpha - (beta*Y  + Z )) - 1 = 0`
///   * ZeroSum (left):  `Sum chi_{m_J}*cstar_L - Sum mult_L*dstar_F = 0`
///   * ZeroSum (right): `Sum chi_{m_J}*cstar_R - Sum mult_R*dstar_F = 0`
fn join_shape(k: usize, m_j: usize, alpha: ArkFr, beta: ArkFr) -> SumcheckShape {
    let dstar = reciprocal_claim(COL_DSTAR_F, &[COL_OF, COL_SF], alpha, beta);
    let cstar_l = reciprocal_claim(COL_CSTAR_L, &[COL_Y, COL_X], alpha, beta);
    let cstar_r = reciprocal_claim(COL_CSTAR_R, &[COL_Z, COL_Y], alpha, beta);
    let link_l = multiset_link_claim(
        Operand::Chi(CHI_MJ),
        COL_CSTAR_L,
        Operand::Column(COL_MULT_L),
        COL_DSTAR_F,
    );
    let link_r = multiset_link_claim(
        Operand::Chi(CHI_MJ),
        COL_CSTAR_R,
        Operand::Column(COL_MULT_R),
        COL_DSTAR_F,
    );
    SumcheckShape {
        num_vars: k,
        num_columns: NUM_COLUMNS,
        chi_bounds: vec![m_j],
        claims: vec![dstar, cstar_l, cstar_r, link_l, link_r],
    }
}

/// Prove self-join soundness with EXPLICIT multiplicity witnesses (length `n`).
/// The honest entry point [`prove_self_join_soundness`] derives them; tests use
/// this to inject wrong multiplicities.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_self_join_soundness_with_witness(
    sf: &[ArkFr],
    of: &[ArkFr],
    com_sf: &ArkGT,
    com_of: &ArkGT,
    x_out: &[ArkFr],
    y_out: &[ArkFr],
    z_out: &[ArkFr],
    mult_l: &[ArkFr],
    mult_r: &[ArkFr],
    transcript: &mut Transcript254,
) -> io::Result<(JoinSoundnessProof, ArkGT, ArkGT, ArkGT, usize)> {
    let mu = sf.len();
    let m_j = x_out.len();
    if of.len() != mu {
        return Err(invalid("relation Sf/Of length mismatch"));
    }
    if y_out.len() != m_j || z_out.len() != m_j {
        return Err(invalid("output X/Y/Z length mismatch"));
    }
    let n = join_padded_len(mu, m_j)?;
    if mult_l.len() != n || mult_r.len() != n {
        return Err(invalid("multiplicity column length mismatch"));
    }
    let k = n.trailing_zeros() as usize;

    let sf_p = padded(sf, n);
    let of_p = padded(of, n);
    let x_p = padded(x_out, n);
    let y_p = padded(y_out, n);
    let z_p = padded(z_out, n);

    // Round 1: commit the output + multiplicity columns (raw-data witnesses, so
    // committed before the fingerprint challenge), bind all data commitments.
    let com_x = commit_column(&x_p)?;
    let com_y = commit_column(&y_p)?;
    let com_z = commit_column(&z_p)?;
    let mult_l_commitment = commit_column(mult_l)?;
    let mult_r_commitment = commit_column(mult_r)?;
    bind_data_commitments(
        transcript,
        com_sf,
        com_of,
        &com_x,
        &com_y,
        &com_z,
        &mult_l_commitment,
        &mult_r_commitment,
    );
    let beta = transcript.challenge_scalar(LABEL_BETA);
    let alpha = transcript.challenge_scalar(LABEL_ALPHA);

    // Round 2: reciprocal witnesses over the beta-fingerprinted rows.
    let dstar_f = reciprocal_witness(alpha, &rlc_fold(beta, &[&of_p, &sf_p]));
    let cstar_l = reciprocal_witness(alpha, &rlc_fold(beta, &[&y_p, &x_p]));
    let cstar_r = reciprocal_witness(alpha, &rlc_fold(beta, &[&z_p, &y_p]));
    let dstar_f_commitment = commit_column(&dstar_f)?;
    let cstar_l_commitment = commit_column(&cstar_l)?;
    let cstar_r_commitment = commit_column(&cstar_r)?;
    bind_reciprocal_commitments(
        transcript,
        &dstar_f_commitment,
        &cstar_l_commitment,
        &cstar_r_commitment,
    );

    let shape = join_shape(k, m_j, alpha, beta);
    let columns = vec![
        sf_p,
        of_p,
        x_p,
        y_p,
        z_p,
        mult_l.to_vec(),
        mult_r.to_vec(),
        dstar_f,
        cstar_l,
        cstar_r,
    ];
    let (sumcheck, r, _evals) = prove_sumcheck(&shape, &columns, transcript)?;
    let opening = prove_column_openings(&columns, &r, transcript)?;

    Ok((
        JoinSoundnessProof {
            sumcheck,
            opening,
            mult_l_commitment,
            mult_r_commitment,
            dstar_f_commitment,
            cstar_l_commitment,
            cstar_r_commitment,
        },
        com_x,
        com_y,
        com_z,
        m_j,
    ))
}

/// Prove that a claimed self-join output `(X_out, Y_out, Z_out)` (length `m_J`) is
/// SOUND against the committed relation `F = (Sf, Of)`: every output row's
/// `(x, y)` and `(y, z)` is a genuine `F` row. `com_sf`/`com_of` must commit `Sf`/
/// `Of` padded to `join_padded_len(mu, m_J)`. Returns the proof, `Com(X/Y/Z)`, and
/// `m_J`.
pub fn prove_self_join_soundness(
    sf: &[ArkFr],
    of: &[ArkFr],
    com_sf: &ArkGT,
    com_of: &ArkGT,
    x_out: &[ArkFr],
    y_out: &[ArkFr],
    z_out: &[ArkFr],
    transcript: &mut Transcript254,
) -> io::Result<(JoinSoundnessProof, ArkGT, ArkGT, ArkGT, usize)> {
    let n = join_padded_len(sf.len(), x_out.len())?;
    let mult_l = multiplicity_over_f(sf, of, x_out, y_out, n);
    let mult_r = multiplicity_over_f(sf, of, y_out, z_out, n);
    prove_self_join_soundness_with_witness(
        sf, of, com_sf, com_of, x_out, y_out, z_out, &mult_l, &mult_r, transcript,
    )
}

/// Verify a self-join soundness proof. The verifier concludes every output row is
/// a genuine `(x, y, z)` join of two `F` rows (no fabricated rows). Omitted rows
/// are NOT detected here (completeness is Phase 2c-2).
#[allow(clippy::too_many_arguments)]
pub fn verify_self_join_soundness(
    com_sf: &ArkGT,
    com_of: &ArkGT,
    com_x: &ArkGT,
    com_y: &ArkGT,
    com_z: &ArkGT,
    m_j: usize,
    proof: &JoinSoundnessProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    let k = proof.sumcheck.round_polys.len();
    if k == 0 || k >= usize::BITS as usize {
        return Err(invalid("join proof has an implausible variable count"));
    }
    if m_j > (1usize << k) {
        return Err(invalid("output length exceeds the join domain"));
    }

    bind_data_commitments(
        transcript,
        com_sf,
        com_of,
        com_x,
        com_y,
        com_z,
        &proof.mult_l_commitment,
        &proof.mult_r_commitment,
    );
    let beta = transcript.challenge_scalar(LABEL_BETA);
    let alpha = transcript.challenge_scalar(LABEL_ALPHA);
    bind_reciprocal_commitments(
        transcript,
        &proof.dstar_f_commitment,
        &proof.cstar_l_commitment,
        &proof.cstar_r_commitment,
    );

    let shape = join_shape(k, m_j, alpha, beta);
    let (r, evals) = verify_sumcheck(&shape, &proof.sumcheck, transcript)?;
    verify_column_openings(
        &[
            *com_sf,
            *com_of,
            *com_x,
            *com_y,
            *com_z,
            proof.mult_l_commitment,
            proof.mult_r_commitment,
            proof.dstar_f_commitment,
            proof.cstar_l_commitment,
            proof.cstar_r_commitment,
        ],
        &r,
        &evals,
        &proof.opening,
        transcript,
    )
}

#[allow(clippy::too_many_arguments)]
fn bind_data_commitments(
    transcript: &mut Transcript254,
    com_sf: &ArkGT,
    com_of: &ArkGT,
    com_x: &ArkGT,
    com_y: &ArkGT,
    com_z: &ArkGT,
    com_mult_l: &ArkGT,
    com_mult_r: &ArkGT,
) {
    transcript.append_group(b"pow/join/com-sf", com_sf);
    transcript.append_group(b"pow/join/com-of", com_of);
    transcript.append_group(b"pow/join/com-x", com_x);
    transcript.append_group(b"pow/join/com-y", com_y);
    transcript.append_group(b"pow/join/com-z", com_z);
    transcript.append_group(b"pow/join/com-mult-l", com_mult_l);
    transcript.append_group(b"pow/join/com-mult-r", com_mult_r);
}

fn bind_reciprocal_commitments(
    transcript: &mut Transcript254,
    com_dstar_f: &ArkGT,
    com_cstar_l: &ArkGT,
    com_cstar_r: &ArkGT,
) {
    transcript.append_group(b"pow/join/com-dstar-f", com_dstar_f);
    transcript.append_group(b"pow/join/com-cstar-l", com_cstar_l);
    transcript.append_group(b"pow/join/com-cstar-r", com_cstar_r);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fr(v: u64) -> ArkFr {
        ArkFr::from_u64(v)
    }

    fn frs(vals: &[u64]) -> Vec<ArkFr> {
        vals.iter().map(|v| fr(*v)).collect()
    }

    fn transcript() -> Transcript254 {
        Transcript254::new(b"parture-join-test")
    }

    fn commit_padded(column: &[ArkFr], n: usize) -> ArkGT {
        commit_column(&padded(column, n)).unwrap()
    }

    // F = {(1,2),(2,3),(3,1),(2,4)}; self-join on Of==Sf gives the valid output
    // {(1,2,3),(1,2,4),(2,3,1),(3,1,2)}.
    fn relation() -> (Vec<ArkFr>, Vec<ArkFr>) {
        (frs(&[1, 2, 3, 2]), frs(&[2, 3, 1, 4]))
    }

    fn valid_output() -> (Vec<ArkFr>, Vec<ArkFr>, Vec<ArkFr>) {
        (frs(&[1, 1, 2, 3]), frs(&[2, 2, 3, 1]), frs(&[3, 4, 1, 2]))
    }

    // Honest run: commit F at the right domain, prove, verify.
    fn run(
        sf: &[ArkFr],
        of: &[ArkFr],
        x: &[ArkFr],
        y: &[ArkFr],
        z: &[ArkFr],
    ) -> io::Result<()> {
        let n = join_padded_len(sf.len(), x.len()).unwrap();
        let com_sf = commit_padded(sf, n);
        let com_of = commit_padded(of, n);
        let mut pt = transcript();
        let (proof, com_x, com_y, com_z, m) =
            prove_self_join_soundness(sf, of, &com_sf, &com_of, x, y, z, &mut pt)?;
        let mut vt = transcript();
        verify_self_join_soundness(&com_sf, &com_of, &com_x, &com_y, &com_z, m, &proof, &mut vt)
    }

    #[test]
    fn valid_join_output_verifies() {
        let (sf, of) = relation();
        let (x, y, z) = valid_output();
        assert!(run(&sf, &of, &x, &y, &z).is_ok());
    }

    #[test]
    fn fabricated_left_row_is_rejected() {
        // Append (x=9, y=2, z=3): (9,2) is NOT an F row, though (2,3) is.
        let (sf, of) = relation();
        let x = frs(&[1, 1, 2, 3, 9]);
        let y = frs(&[2, 2, 3, 1, 2]);
        let z = frs(&[3, 4, 1, 2, 3]);
        assert!(run(&sf, &of, &x, &y, &z).is_err());
    }

    #[test]
    fn fabricated_right_row_is_rejected() {
        // Append (x=1, y=2, z=9): (1,2) is an F row, but (2,9) is NOT.
        let (sf, of) = relation();
        let x = frs(&[1, 1, 2, 3, 1]);
        let y = frs(&[2, 2, 3, 1, 2]);
        let z = frs(&[3, 4, 1, 2, 9]);
        assert!(run(&sf, &of, &x, &y, &z).is_err());
    }

    #[test]
    fn wrong_multiplicities_are_rejected() {
        let (sf, of) = relation();
        let (x, y, z) = valid_output();
        let n = join_padded_len(sf.len(), x.len()).unwrap();
        let com_sf = commit_padded(&sf, n);
        let com_of = commit_padded(&of, n);

        // Honest multiplicities, then corrupt one entry of mult_L.
        let mut mult_l = multiplicity_over_f(&sf, &of, &x, &y, n);
        let mult_r = multiplicity_over_f(&sf, &of, &y, &z, n);
        mult_l[0] = mult_l[0] + fr(1);

        let mut pt = transcript();
        let (proof, com_x, com_y, com_z, m) = prove_self_join_soundness_with_witness(
            &sf, &of, &com_sf, &com_of, &x, &y, &z, &mult_l, &mult_r, &mut pt,
        )
        .unwrap();
        let mut vt = transcript();
        assert!(verify_self_join_soundness(
            &com_sf, &com_of, &com_x, &com_y, &com_z, m, &proof, &mut vt
        )
        .is_err());
    }

    #[test]
    fn known_gap_omitting_a_valid_pair_still_verifies() {
        // SOUNDNESS-ONLY: dropping a genuine join row (1,2,4) leaves a valid subset
        // -- every remaining row is still a real join -- so it VERIFIES here.
        // Phase 2c-2 (completeness) will make omission FAIL.
        let (sf, of) = relation();
        let x = frs(&[1, 2, 3]); // (1,2,4) omitted
        let y = frs(&[2, 3, 1]);
        let z = frs(&[3, 1, 2]);
        assert!(run(&sf, &of, &x, &y, &z).is_ok());
    }

    #[test]
    fn empty_output_verifies() {
        let (sf, of) = relation();
        assert!(run(&sf, &of, &[], &[], &[]).is_ok());
    }
}
