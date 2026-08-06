//! GATHER gadget (Phase 2c-2d): the self-join completeness-closer that ties
//! 2c-1's per-F-row match multiplicities (`mult_L`, `mult_R`) to 2c-2's group
//! counts (`w_L`, `w_R`). This forbids "rob Peter to pay Paul" (omit one valid
//! join row + duplicate another, which soundness (2c-1) and the count identity
//! (2c-2c) miss). Clean-room from the math; no COSL code.
//!
//! For the self-join `t(?x,P,?y), t(?y,P,?z)` on `F = (Sf, Of)` (join on `Of==Sf`,
//! the shared `y`), define over the distinct key union `U`:
//!   * `w_L[i] = #{ a : Of_a == U[i] }`  (left matches for key `U[i]`),
//!   * `w_R[i] = #{ b : Sf_b == U[i] }`  (right matches for key `U[i]`).
//! Then for every F-row `a`, the number of output rows whose LEFT atom is `a` must
//! be the number of right partners for its key `Of_a`, i.e.
//!   `mult_L[a] = w_R[ group(Of_a) ]`,  and symmetrically  `mult_R[b] = w_L[ group(Sf_b) ]`.
//!
//! Gadget (two 2-column-fingerprint multiset lookups, one per side, fresh
//! challenges each):
//!   LEFT:  `in_fold[a]  = Of[a] + beta_l*mult_L[a]`  (F side, weight 1),
//!          `out_fold[i] = U[i] + beta_l*w_R[i]`      (U side, weighted by w_L[i]),
//!          pin reciprocals, then `Sum chi_{n_F}*rec_in - Sum w_L*rec_out = 0`.
//!   Because exactly `w_L[i]` F-rows have `Of==U[i]`, the out side repeats
//!   `(U[i], w_R[i])` `w_L[i]` times, so the pair-multiset equality forces every
//!   such F-row's `mult_L == w_R[i]`. RIGHT is symmetric with `Sf, mult_R, w_L`.
//!
//! COMPLETENESS (why this closes the gap): omitting a valid output row `(a,b)`
//! decrements BOTH `mult_L[a]` and `mult_R[b]`; duplicating `(a',b')` increments
//! BOTH `mult_L[a']` and `mult_R[b']`. For all mults to stay equal to the gathered
//! `w`, one needs `a'=a` and `b'=b` (a no-op). So any real omit+duplicate changes
//! some `mult`, caught by the left or right gather. Distinctness of `U` (2c-2a)
//! and the correctness of `w_L`/`w_R` as group counts (2c-2b) are composed upstream.
//!
//! NOTE vs proof-of-sql: their production sort-merge join closes completeness a
//! DIFFERENT way (output rows carry input-row index pointers; an index-augmented
//! membership + strict monotonicity of the combined `2^64*left_idx + right_idx`
//! forbids duplicate output rows). It has no `mult<->w` gather. Both close the gap;
//! this module implements our gather architecture. See the report.

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

const COL_OF: usize = 0;
const COL_SF: usize = 1;
const COL_MULT_L: usize = 2;
const COL_MULT_R: usize = 3;
const COL_U: usize = 4;
const COL_WL: usize = 5;
const COL_WR: usize = 6;
const COL_REC_IN_L: usize = 7;
const COL_REC_OUT_L: usize = 8;
const COL_REC_IN_R: usize = 9;
const COL_REC_OUT_R: usize = 10;
const NUM_COLUMNS: usize = 11;
// chi_bounds index for chi_{n_F} (the F side real length).
const CHI_NF: usize = 0;

const LABEL_BETA_L: &[u8] = b"pow/gather/left-beta";
const LABEL_ALPHA_L: &[u8] = b"pow/gather/left-alpha";
const LABEL_BETA_R: &[u8] = b"pow/gather/right-beta";
const LABEL_ALPHA_R: &[u8] = b"pow/gather/right-alpha";

#[inline]
fn f0() -> ArkFr {
    ArkFr::zero()
}

/// A gather proof: the batched sumcheck over the two 2-column-fingerprint
/// membership lookups (left + right), one batched Dory opening over all eleven
/// columns, and the four reciprocal commitments.
#[derive(Clone, Debug)]
pub struct GatherProof {
    pub sumcheck: SumcheckProof,
    pub opening: ColumnOpening,
    pub rec_in_l_commitment: ArkGT,
    pub rec_out_l_commitment: ArkGT,
    pub rec_in_r_commitment: ArkGT,
    pub rec_out_r_commitment: ArkGT,
}

/// The common power-of-two domain (`>= 2`) for `n_f` F-rows and `n_u` distinct keys.
pub(crate) fn gather_domain(n_f: usize, n_u: usize) -> io::Result<usize> {
    n_f.max(n_u)
        .checked_next_power_of_two()
        .ok_or_else(|| invalid("gather domain size overflow"))
        .map(|d| d.max(2))
}

fn padded(column: &[ArkFr], n: usize) -> Vec<ArkFr> {
    let mut out = column.to_vec();
    out.resize(n, f0());
    out
}

#[allow(clippy::too_many_arguments)]
fn gather_shape(
    k: usize,
    n_f: usize,
    alpha_l: ArkFr,
    beta_l: ArkFr,
    alpha_r: ArkFr,
    beta_r: ArkFr,
) -> SumcheckShape {
    // LEFT: in = (Of, mult_L); out = (U, w_R) weighted by w_L.
    let rec_in_l = reciprocal_claim(COL_REC_IN_L, &[COL_OF, COL_MULT_L], alpha_l, beta_l);
    let rec_out_l = reciprocal_claim(COL_REC_OUT_L, &[COL_U, COL_WR], alpha_l, beta_l);
    let link_l = multiset_link_claim(
        Operand::Chi(CHI_NF),
        COL_REC_IN_L,
        Operand::Column(COL_WL),
        COL_REC_OUT_L,
    );
    // RIGHT: in = (Sf, mult_R); out = (U, w_L) weighted by w_R.
    let rec_in_r = reciprocal_claim(COL_REC_IN_R, &[COL_SF, COL_MULT_R], alpha_r, beta_r);
    let rec_out_r = reciprocal_claim(COL_REC_OUT_R, &[COL_U, COL_WL], alpha_r, beta_r);
    let link_r = multiset_link_claim(
        Operand::Chi(CHI_NF),
        COL_REC_IN_R,
        Operand::Column(COL_WR),
        COL_REC_OUT_R,
    );
    SumcheckShape {
        num_vars: k,
        num_columns: NUM_COLUMNS,
        chi_bounds: vec![n_f],
        claims: vec![rec_in_l, rec_out_l, link_l, rec_in_r, rec_out_r, link_r],
    }
}

/// Prove the gather relation: `mult_L[a] = w_R[group(Of_a)]` for every F-row `a`,
/// and symmetrically `mult_R[b] = w_L[group(Sf_b)]`. All input columns must be
/// committed padded to `gather_domain(n_F, n_U)`.
#[allow(clippy::too_many_arguments)]
pub fn prove_gather(
    of: &[ArkFr],
    sf: &[ArkFr],
    mult_l: &[ArkFr],
    mult_r: &[ArkFr],
    u: &[ArkFr],
    w_l: &[ArkFr],
    w_r: &[ArkFr],
    com_of: &ArkGT,
    com_sf: &ArkGT,
    com_mult_l: &ArkGT,
    com_mult_r: &ArkGT,
    com_u: &ArkGT,
    com_wl: &ArkGT,
    com_wr: &ArkGT,
    transcript: &mut Transcript254,
) -> io::Result<GatherProof> {
    let n_f = of.len();
    let n_u = u.len();
    if sf.len() != n_f || mult_l.len() != n_f || mult_r.len() != n_f {
        return Err(invalid("F columns (Of/Sf/mult_L/mult_R) length mismatch"));
    }
    if w_l.len() != n_u || w_r.len() != n_u {
        return Err(invalid("group-count columns (w_L/w_R) length mismatch"));
    }
    let n = gather_domain(n_f, n_u)?;
    let k = n.trailing_zeros() as usize;

    let of_p = padded(of, n);
    let sf_p = padded(sf, n);
    let mult_l_p = padded(mult_l, n);
    let mult_r_p = padded(mult_r, n);
    let u_p = padded(u, n);
    let w_l_p = padded(w_l, n);
    let w_r_p = padded(w_r, n);

    // Round 1: bind the seven given commitments, draw fresh challenges per side.
    transcript.append_group(b"pow/gather/com-of", com_of);
    transcript.append_group(b"pow/gather/com-sf", com_sf);
    transcript.append_group(b"pow/gather/com-mult-l", com_mult_l);
    transcript.append_group(b"pow/gather/com-mult-r", com_mult_r);
    transcript.append_group(b"pow/gather/com-u", com_u);
    transcript.append_group(b"pow/gather/com-wl", com_wl);
    transcript.append_group(b"pow/gather/com-wr", com_wr);
    let beta_l = transcript.challenge_scalar(LABEL_BETA_L);
    let alpha_l = transcript.challenge_scalar(LABEL_ALPHA_L);
    let beta_r = transcript.challenge_scalar(LABEL_BETA_R);
    let alpha_r = transcript.challenge_scalar(LABEL_ALPHA_R);

    // Round 2: reciprocal witnesses over the (value, mult)/(value, w) fingerprints.
    let rec_in_l = reciprocal_witness(alpha_l, &rlc_fold(beta_l, &[&of_p, &mult_l_p]));
    let rec_out_l = reciprocal_witness(alpha_l, &rlc_fold(beta_l, &[&u_p, &w_r_p]));
    let rec_in_r = reciprocal_witness(alpha_r, &rlc_fold(beta_r, &[&sf_p, &mult_r_p]));
    let rec_out_r = reciprocal_witness(alpha_r, &rlc_fold(beta_r, &[&u_p, &w_l_p]));
    let rec_in_l_commitment = commit_column(&rec_in_l)?;
    let rec_out_l_commitment = commit_column(&rec_out_l)?;
    let rec_in_r_commitment = commit_column(&rec_in_r)?;
    let rec_out_r_commitment = commit_column(&rec_out_r)?;
    transcript.append_group(b"pow/gather/com-rec-in-l", &rec_in_l_commitment);
    transcript.append_group(b"pow/gather/com-rec-out-l", &rec_out_l_commitment);
    transcript.append_group(b"pow/gather/com-rec-in-r", &rec_in_r_commitment);
    transcript.append_group(b"pow/gather/com-rec-out-r", &rec_out_r_commitment);

    let shape = gather_shape(k, n_f, alpha_l, beta_l, alpha_r, beta_r);
    let columns = vec![
        of_p, sf_p, mult_l_p, mult_r_p, u_p, w_l_p, w_r_p, rec_in_l, rec_out_l, rec_in_r, rec_out_r,
    ];
    let (sumcheck, r, _evals) = prove_sumcheck(&shape, &columns, transcript)?;
    let opening = prove_column_openings(&columns, &r, transcript)?;

    Ok(GatherProof {
        sumcheck,
        opening,
        rec_in_l_commitment,
        rec_out_l_commitment,
        rec_in_r_commitment,
        rec_out_r_commitment,
    })
}

/// Verify a gather proof.
#[allow(clippy::too_many_arguments)]
pub fn verify_gather(
    com_of: &ArkGT,
    com_sf: &ArkGT,
    com_mult_l: &ArkGT,
    com_mult_r: &ArkGT,
    com_u: &ArkGT,
    com_wl: &ArkGT,
    com_wr: &ArkGT,
    n_f: usize,
    n_u: usize,
    proof: &GatherProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    let k = proof.sumcheck.round_polys.len();
    if k == 0 || k >= usize::BITS as usize {
        return Err(invalid("gather proof has an implausible variable count"));
    }
    if n_f > (1usize << k) || n_u > (1usize << k) {
        return Err(invalid("real length exceeds the gather domain"));
    }

    transcript.append_group(b"pow/gather/com-of", com_of);
    transcript.append_group(b"pow/gather/com-sf", com_sf);
    transcript.append_group(b"pow/gather/com-mult-l", com_mult_l);
    transcript.append_group(b"pow/gather/com-mult-r", com_mult_r);
    transcript.append_group(b"pow/gather/com-u", com_u);
    transcript.append_group(b"pow/gather/com-wl", com_wl);
    transcript.append_group(b"pow/gather/com-wr", com_wr);
    let beta_l = transcript.challenge_scalar(LABEL_BETA_L);
    let alpha_l = transcript.challenge_scalar(LABEL_ALPHA_L);
    let beta_r = transcript.challenge_scalar(LABEL_BETA_R);
    let alpha_r = transcript.challenge_scalar(LABEL_ALPHA_R);
    transcript.append_group(b"pow/gather/com-rec-in-l", &proof.rec_in_l_commitment);
    transcript.append_group(b"pow/gather/com-rec-out-l", &proof.rec_out_l_commitment);
    transcript.append_group(b"pow/gather/com-rec-in-r", &proof.rec_in_r_commitment);
    transcript.append_group(b"pow/gather/com-rec-out-r", &proof.rec_out_r_commitment);

    let shape = gather_shape(k, n_f, alpha_l, beta_l, alpha_r, beta_r);
    let (r, evals) = verify_sumcheck(&shape, &proof.sumcheck, transcript)?;
    verify_column_openings(
        &[
            *com_of,
            *com_sf,
            *com_mult_l,
            *com_mult_r,
            *com_u,
            *com_wl,
            *com_wr,
            proof.rec_in_l_commitment,
            proof.rec_out_l_commitment,
            proof.rec_in_r_commitment,
            proof.rec_out_r_commitment,
        ],
        &r,
        &evals,
        &proof.opening,
        transcript,
    )
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
        Transcript254::new(b"parture-gather-test")
    }

    fn commit_padded(column: &[ArkFr], n: usize) -> ArkGT {
        commit_column(&padded(column, n)).unwrap()
    }

    // gather[a] = w[ group(values[a]) ], where group(v) is the index in U.
    fn gather_of(u: &[ArkFr], w: &[ArkFr], values: &[ArkFr]) -> Vec<ArkFr> {
        values
            .iter()
            .map(|v| {
                let i = u.iter().position(|uv| uv == v).expect("value must be in U");
                w[i]
            })
            .collect()
    }

    // A consistent self-join scenario over F = {(1,2),(3,2),(2,4),(2,5)}.
    // U = {1,2,3,4,5}; w_L (Of-group counts) and w_R (Sf-group counts) as below.
    fn scenario() -> (Vec<ArkFr>, Vec<ArkFr>, Vec<ArkFr>, Vec<ArkFr>, Vec<ArkFr>) {
        let sf = frs(&[1, 3, 2, 2]);
        let of = frs(&[2, 2, 4, 5]);
        let u = frs(&[1, 2, 3, 4, 5]);
        let w_l = frs(&[0, 2, 0, 1, 1]); // #{Of == U[i]}
        let w_r = frs(&[1, 2, 1, 0, 0]); // #{Sf == U[i]}
        (sf, of, u, w_l, w_r)
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        of: &[ArkFr],
        sf: &[ArkFr],
        mult_l: &[ArkFr],
        mult_r: &[ArkFr],
        u: &[ArkFr],
        w_l: &[ArkFr],
        w_r: &[ArkFr],
    ) -> io::Result<()> {
        let n = gather_domain(of.len(), u.len()).unwrap();
        let com_of = commit_padded(of, n);
        let com_sf = commit_padded(sf, n);
        let com_mult_l = commit_padded(mult_l, n);
        let com_mult_r = commit_padded(mult_r, n);
        let com_u = commit_padded(u, n);
        let com_wl = commit_padded(w_l, n);
        let com_wr = commit_padded(w_r, n);
        let mut pt = transcript();
        let proof = prove_gather(
            of, sf, mult_l, mult_r, u, w_l, w_r, &com_of, &com_sf, &com_mult_l, &com_mult_r, &com_u,
            &com_wl, &com_wr, &mut pt,
        )?;
        let mut vt = transcript();
        verify_gather(
            &com_of, &com_sf, &com_mult_l, &com_mult_r, &com_u, &com_wl, &com_wr, of.len(), u.len(),
            &proof, &mut vt,
        )
    }

    #[test]
    fn correct_gather_verifies() {
        let (sf, of, u, w_l, w_r) = scenario();
        let mult_l = gather_of(&u, &w_r, &of); // [2,2,0,0]
        let mult_r = gather_of(&u, &w_l, &sf); // [0,0,2,2]
        assert_eq!(mult_l, frs(&[2, 2, 0, 0]));
        assert_eq!(mult_r, frs(&[0, 0, 2, 2]));
        assert!(run(&of, &sf, &mult_l, &mult_r, &u, &w_l, &w_r).is_ok());
    }

    #[test]
    fn tampered_mult_l_is_rejected() {
        let (sf, of, u, w_l, w_r) = scenario();
        let mut mult_l = gather_of(&u, &w_r, &of);
        let mult_r = gather_of(&u, &w_l, &sf);
        mult_l[0] = mult_l[0] + fr(1); // F-row 0's left match-count no longer = w_R
        assert!(run(&of, &sf, &mult_l, &mult_r, &u, &w_l, &w_r).is_err());
    }

    #[test]
    fn tampered_mult_r_is_rejected() {
        let (sf, of, u, w_l, w_r) = scenario();
        let mult_l = gather_of(&u, &w_r, &of);
        let mut mult_r = gather_of(&u, &w_l, &sf);
        mult_r[2] = mult_r[2] - fr(1); // F-row 2's right match-count no longer = w_L
        assert!(run(&of, &sf, &mult_l, &mult_r, &u, &w_l, &w_r).is_err());
    }

    #[test]
    fn rob_peter_to_pay_paul_shape_is_caught_by_one_side() {
        // Documents the closer: the FULL attack (omit one output pair + duplicate
        // another) is exercised end-to-end at 2c-2e. Here, at the gather layer, we
        // show the mult perturbation it induces is caught. Omitting output (a=0,b=2)
        // and duplicating (a=1,b=3) leaves mult_L unchanged ([2,2,..]) but shifts
        // mult_R: row 2 loses a partner (2 -> 1), row 3 gains one (2 -> 3). The right
        // gather then rejects (mult_R[2] != w_L[group(Sf_2)]).
        let (sf, of, u, w_l, w_r) = scenario();
        let mult_l = gather_of(&u, &w_r, &of); // unchanged by same-left-atom shuffle
        let mut mult_r = gather_of(&u, &w_l, &sf);
        mult_r[2] = mult_r[2] - fr(1);
        mult_r[3] = mult_r[3] + fr(1);
        assert!(run(&of, &sf, &mult_l, &mult_r, &u, &w_l, &w_r).is_err());
    }

    #[test]
    fn empty_relation_verifies() {
        // n_F = 0: no F rows; U also empty. All mult/w are empty -> both links are
        // 0 = 0.
        assert!(run(&[], &[], &[], &[], &[], &[], &[]).is_ok());
    }
}
