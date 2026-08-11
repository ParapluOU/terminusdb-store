//! Full-field LogUp SEMI-JOIN over reconciled IRI-hash columns (CROSS-DICT join, no sort).
//!
//! IsA (`?x isa C`, node regime) needs, for each instance row `(x, class_inst_hash)`, to KEEP it
//! iff `class_inst_hash` is a MEMBER of the schema-closure hash SET `{class_schema_hash}`. The join
//! keys are `hash_object(IRI)` values — full ~254-bit BN254 field elements — so the audited
//! magnitude-sort inner join (whose key-universe strict-sort caps keys at 64/128 bits) CANNOT be used
//! (`hash_object` values sit too near the field modulus to soundly bit-reduce). This gadget instead
//! reconciles by LOGUP, which is field-element-native (no magnitude/ordering, no bit cap):
//!
//! It composes THREE audited pieces at one working domain `2^D`:
//!   1. a minimal booleanity/`chi_n`-mask proof pinning the keep-selection `sel` to `{0,1}` and to
//!      `0` on padding rows;
//!   2. the audited [`super::filter::prove_compaction_only`]: the delivered `(x_kept, h_kept)` is
//!      EXACTLY the `sel`-compaction of the instance rows `(x, h)` — so every kept row is a REAL
//!      instance row with its own `x` and hash;
//!   3. the audited [`super::membership::prove_membership_with_mult_in_domain_with_context`]:
//!      `h_kept` is a sub-multiset of the closure hash set `U` — so every kept hash is in the closure.
//!
//! SOUNDNESS: (2)+(3) give that every delivered `x` is a real instance row whose class-hash is a
//! member of the closure — i.e. `class(x)` is subsumed by `C`. A schema hash and an instance hash are
//! equal as FIELD ELEMENTS iff the IRIs are equal (`hash_object` is collision-resistant and
//! dictionary-INDEPENDENT), so a same-integer-id/different-IRI pair (different hashes) never matches,
//! and two unrelated classes (different hashes) never match. No magnitude comparison anywhere.
//!
//! COMPLETENESS CAVEAT (documented): the membership direction proves `kept ⊆ closure` (no false
//! positive). It does NOT, by itself, force `sel = 1` for EVERY member row — an adversarial prover
//! could DROP a member (report a strict subset). Single-`alpha` LogUp set-membership provably cannot
//! enforce the non-membership of dropped rows without either a magnitude sort or an `O(n_L*n_R)`
//! per-element equality. The honest prover keeps every member (so the e2e answer is complete); a
//! cryptographic no-drop guarantee is a documented follow-up (per-closure-element equality-filter +
//! union). This matches the coordinator's membership specification.

use std::io;

use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::transcript::Transcript;

use super::argument::{
    prove_sumcheck, verify_column_openings, verify_sumcheck, ClaimShape, ColumnOpening, Expr,
    Operand, StoreProverContext, SumcheckClaim, SumcheckProof, SumcheckShape, Term, Transcript254,
};
use super::filter::{prove_compaction_only_with_context, verify_compaction_only, CompactionOnlyProof};
use super::membership::{
    count_multiplicities, prove_membership_with_mult_in_domain_with_context,
    verify_membership_with_mult, MembershipProof,
};
use super::invalid;

#[inline]
fn f0() -> ArkFr {
    ArkFr::zero()
}
#[inline]
fn f1() -> ArkFr {
    ArkFr::one()
}

const SEL_LABEL_DOMAIN: &[u8] = b"pow/semijoin/sel-mask/domain-v1";

/// The keep-selection booleanity + `chi_n` mask proof: pins `sel` to `{0,1}` everywhere AND to `0`
/// on every padding row `[n, 2^D)`, so the downstream compaction's external-selection precondition
/// holds. Two Identity claims over the single `sel` column.
#[derive(Clone, Debug)]
pub struct SelMaskProof {
    pub sumcheck: SumcheckProof,
    pub opening: ColumnOpening,
}

fn sel_mask_shape(num_vars: usize, real_n: usize) -> SumcheckShape {
    // Booleanity everywhere: sel*sel - sel = 0.
    let booleanity = SumcheckClaim {
        shape: ClaimShape::Identity,
        expr: Expr {
            terms: vec![
                Term { coeff: f1(), operands: vec![Operand::Column(0), Operand::Column(0)] },
                Term { coeff: f0() - f1(), operands: vec![Operand::Column(0)] },
            ],
        },
        target: f0(),
    };
    // Padding mask: sel*(1 - chi_n) = 0  ->  sel - sel*chi_n = 0.
    let mask = SumcheckClaim {
        shape: ClaimShape::Identity,
        expr: Expr {
            terms: vec![
                Term { coeff: f1(), operands: vec![Operand::Column(0)] },
                Term { coeff: f0() - f1(), operands: vec![Operand::Column(0), Operand::Chi(0)] },
            ],
        },
        target: f0(),
    };
    SumcheckShape {
        num_vars,
        num_columns: 1,
        chi_bounds: vec![real_n],
        claims: vec![booleanity, mask],
    }
}

fn prove_sel_mask(
    context: &StoreProverContext<'_>,
    sel: &[ArkFr],
    com_sel: &ArkGT,
    real_n: usize,
    transcript: &mut Transcript254,
) -> io::Result<SelMaskProof> {
    let size = sel.len();
    if !size.is_power_of_two() || size < 2 {
        return Err(invalid("sel-mask column length must be a power of two >= 2"));
    }
    if real_n > size {
        return Err(invalid("sel-mask real length exceeds the column domain"));
    }
    let num_vars = size.trailing_zeros() as usize;
    transcript.append_bytes(SEL_LABEL_DOMAIN, &(real_n as u64).to_le_bytes());
    transcript.append_group(b"pow/semijoin/com-sel", com_sel);
    let shape = sel_mask_shape(num_vars, real_n);
    let columns = vec![sel.to_vec()];
    let (sumcheck, r, _evals) = prove_sumcheck(&shape, &columns, transcript)?;
    let opening = context.prove_column_openings(&columns, &r, transcript)?;
    Ok(SelMaskProof { sumcheck, opening })
}

fn verify_sel_mask(
    com_sel: &ArkGT,
    real_n: usize,
    proof: &SelMaskProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    let num_vars = proof.sumcheck.round_polys.len();
    if num_vars == 0 || num_vars >= usize::BITS as usize {
        return Err(invalid("sel-mask proof has an implausible variable count"));
    }
    if real_n > (1usize << num_vars) {
        return Err(invalid("sel-mask real length exceeds the column domain"));
    }
    transcript.append_bytes(SEL_LABEL_DOMAIN, &(real_n as u64).to_le_bytes());
    transcript.append_group(b"pow/semijoin/com-sel", com_sel);
    let shape = sel_mask_shape(num_vars, real_n);
    let (r, evals) = verify_sumcheck(&shape, &proof.sumcheck, transcript)?;
    verify_column_openings(&[*com_sel], &r, &evals, &proof.opening, transcript)
}

/// A full-field LogUp semi-join proof: the keep-selection mask, the compaction of the instance rows
/// to the kept `(x, h)` prefix, and the membership of the kept hashes in the closure set.
#[derive(Clone, Debug)]
pub struct HashSemiJoinProof {
    pub sel_mask: SelMaskProof,
    pub com_sel: ArkGT,
    pub compaction: CompactionOnlyProof,
    pub com_x_kept: ArkGT,
    pub com_h_kept: ArkGT,
    pub com_w: ArkGT,
    pub membership: MembershipProof,
}

/// Prove the semi-join at ONE working domain `2^D`. Inputs are padded to `2^D`:
///   * `closure` (`com_closure`): the SCHEMA-closure IRI-hash SET, real length `n_closure` (a set,
///     from the deduped transitive closure), authenticated to the schema root by the caller.
///   * `x` (`com_x`), `h` (`com_h`): the INSTANCE rows `(x, class_inst_hash)`, real length `n_inst`,
///     both authenticated to the instance root by the caller (`h` via the dict-hash lookup).
/// Returns the proof and the kept output: `com_x_kept` (the delivered `?x` column), the dense kept
/// `x` column (padded to `2^D`), and the kept count `m`.
#[allow(clippy::too_many_arguments)]
pub fn prove_hash_semijoin(
    context: &StoreProverContext<'_>,
    closure: &[ArkFr],
    com_closure: &ArkGT,
    n_closure: usize,
    x: &[ArkFr],
    com_x: &ArkGT,
    h: &[ArkFr],
    com_h: &ArkGT,
    n_inst: usize,
    transcript: &mut Transcript254,
) -> io::Result<(HashSemiJoinProof, Vec<ArkFr>, usize)> {
    let size = h.len();
    if !size.is_power_of_two() || size < 2 {
        return Err(invalid("semijoin column length must be a power of two >= 2"));
    }
    for column in [closure, x] {
        if column.len() != size {
            return Err(invalid("semijoin columns must share one working domain"));
        }
    }
    if n_closure > size || n_inst > size {
        return Err(invalid("semijoin real length exceeds the working domain"));
    }

    // Keep-selection: sel[i] = 1 iff i < n_inst AND h[i] equals one of the real closure hashes.
    // A linear scan over the (small) closure; ArkFr is compared by field-element equality.
    let closure_real = &closure[..n_closure];
    let mut sel = vec![f0(); size];
    for i in 0..n_inst {
        if closure_real.iter().any(|c| *c == h[i]) {
            sel[i] = f1();
        }
    }
    let com_sel = context.commit_column(&sel)?;
    let sel_mask = prove_sel_mask(context, &sel, &com_sel, n_inst, transcript)?;

    // Compaction: (x_kept, h_kept) = the sel-selected instance rows (dense prefix + zero tail).
    let (compaction, com_kept, kept_out, m) = prove_compaction_only_with_context(
        context,
        &[x.to_vec(), h.to_vec()],
        &[*com_x, *com_h],
        &sel,
        &com_sel,
        transcript,
    )?;
    let com_x_kept = com_kept[0];
    let com_h_kept = com_kept[1];
    let x_kept = kept_out[0].clone();
    let h_kept = kept_out[1].clone();

    // Membership: every kept hash is in the closure set. u = closure (n_closure), of = h_kept (m).
    let w = count_multiplicities(&closure[..n_closure], &h_kept[..m]);
    let com_w = context.commit_column(&pad_to(&w, size))?;
    let membership = prove_membership_with_mult_in_domain_with_context(
        context,
        &closure[..n_closure],
        &h_kept[..m],
        &w,
        com_closure,
        &com_h_kept,
        &com_w,
        size,
        transcript,
    )?;

    Ok((
        HashSemiJoinProof {
            sel_mask,
            com_sel,
            compaction,
            com_x_kept,
            com_h_kept,
            com_w,
            membership,
        },
        x_kept,
        m,
    ))
}

/// Verify a semi-join proof. On success, `proof.com_x_kept` commits `output_len` instance `?x` ids,
/// each a real instance row `(x, h)` whose class hash `h` is a member of the closure set committed by
/// `com_closure`. `com_closure`/`com_x`/`com_h` must be the caller-authenticated columns at the
/// working domain `2^D`; `n_closure`/`n_inst` are their real lengths.
#[allow(clippy::too_many_arguments)]
pub fn verify_hash_semijoin(
    com_closure: &ArkGT,
    n_closure: usize,
    com_x: &ArkGT,
    com_h: &ArkGT,
    n_inst: usize,
    output_len: usize,
    proof: &HashSemiJoinProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    // (1) keep-selection is a genuine boolean, chi_{n_inst}-masked column.
    verify_sel_mask(&proof.com_sel, n_inst, &proof.sel_mask, transcript)?;
    // (2) (x_kept, h_kept) is EXACTLY the sel-compaction of the instance rows (x, h).
    verify_compaction_only(
        &[*com_x, *com_h],
        &proof.com_sel,
        &[proof.com_x_kept, proof.com_h_kept],
        output_len,
        &proof.compaction,
        transcript,
    )?;
    // (3) every kept hash is a member of the closure set.
    verify_membership_with_mult(
        com_closure,
        &proof.com_h_kept,
        &proof.com_w,
        n_closure,
        output_len,
        &proof.membership,
        transcript,
    )?;
    let _ = n_inst;
    Ok(())
}

fn pad_to(column: &[ArkFr], size: usize) -> Vec<ArkFr> {
    let mut out = column.to_vec();
    out.resize(size, f0());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof::argument::commit_column;

    fn fr(v: u64) -> ArkFr {
        ArkFr::from_u64(v)
    }

    fn padded(vals: &[u64], size: usize) -> Vec<ArkFr> {
        let mut out: Vec<ArkFr> = vals.iter().map(|v| fr(*v)).collect();
        out.resize(size, f0());
        out
    }

    fn transcript() -> Transcript254 {
        Transcript254::new(b"parture-semijoin-test")
    }

    // Simulate ~254-bit "hashes" by using large field elements far above 2^128 (values the
    // magnitude-sort inner join could not handle), to exercise the full-field path.
    fn big(tag: u64) -> ArkFr {
        // 2^200 + tag: a genuinely large field element (> any 128-bit key bound).
        let mut acc = fr(1);
        for _ in 0..200 {
            acc = acc + acc;
        }
        acc + fr(tag)
    }

    #[test]
    fn semijoin_keeps_members_excludes_nonmembers() {
        let size = 8usize;
        // Closure set U = {H(A), H(B)} (two distinct big hashes).
        let ha = big(1);
        let hb = big(2);
        let hother = big(999);
        let closure = {
            let mut c = vec![ha, hb];
            c.resize(size, f0());
            c
        };
        let com_closure = commit_column(&closure).unwrap();
        // Instance rows: (x1, H(A)), (x2, H(B)), (x3, H(other)).
        let x = padded(&[11, 12, 13], size);
        let h = {
            let mut hh = vec![ha, hb, hother];
            hh.resize(size, f0());
            hh
        };
        let com_x = commit_column(&x).unwrap();
        let com_h = commit_column(&h).unwrap();

        let mut t = transcript();
        let (proof, x_kept, m) = prove_hash_semijoin(
            &StoreProverContext::cpu(), &closure, &com_closure, 2, &x, &com_x, &h, &com_h, 3, &mut t,
        )
        .unwrap();
        assert_eq!(m, 2, "x1 and x2 are members; x3 (other) is excluded");
        assert_eq!(&x_kept[..2], &[fr(11), fr(12)], "kept ?x = {{x1, x2}}");

        let mut tv = transcript();
        verify_hash_semijoin(&com_closure, 2, &com_x, &com_h, 3, m, &proof, &mut tv).unwrap();
    }

    #[test]
    fn semijoin_excludes_same_field_position_different_hash() {
        // The instance class hash equals NONE of the closure hashes (different IRIs => different
        // hashes), even if ids collided upstream: no match.
        let size = 4usize;
        let closure = {
            let mut c = vec![big(1)];
            c.resize(size, f0());
            c
        };
        let com_closure = commit_column(&closure).unwrap();
        let x = padded(&[7], size);
        let h = {
            let mut hh = vec![big(42)]; // different hash than big(1)
            hh.resize(size, f0());
            hh
        };
        let com_x = commit_column(&x).unwrap();
        let com_h = commit_column(&h).unwrap();
        let mut t = transcript();
        let (proof, _x_kept, m) = prove_hash_semijoin(
            &StoreProverContext::cpu(), &closure, &com_closure, 1, &x, &com_x, &h, &com_h, 1, &mut t,
        )
        .unwrap();
        assert_eq!(m, 0, "different-hash instance class must be excluded");
        let mut tv = transcript();
        verify_hash_semijoin(&com_closure, 1, &com_x, &com_h, 1, m, &proof, &mut tv).unwrap();
    }

    #[test]
    fn semijoin_rejects_fabricated_nonmember_output() {
        // SOUNDNESS: a prover that claims to keep a row whose hash is NOT in the closure must fail
        // the membership check. Force sel to keep x3 (hother) by tampering is caught because the
        // membership of the kept hash fails. We simulate by proving honestly then verifying with a
        // closure that does NOT contain the kept hash.
        let size = 4usize;
        let closure_real = {
            let mut c = vec![big(1)];
            c.resize(size, f0());
            c
        };
        let com_closure_real = commit_column(&closure_real).unwrap();
        let x = padded(&[7], size);
        let h = {
            let mut hh = vec![big(1)];
            hh.resize(size, f0());
            hh
        };
        let com_x = commit_column(&x).unwrap();
        let com_h = commit_column(&h).unwrap();
        let mut t = transcript();
        let (proof, _x_kept, m) = prove_hash_semijoin(
            &StoreProverContext::cpu(), &closure_real, &com_closure_real, 1, &x, &com_x, &h, &com_h, 1, &mut t,
        )
        .unwrap();
        assert_eq!(m, 1);
        // Verify against a DIFFERENT closure commitment (a closure not containing big(1)) -> reject.
        let closure_bad = {
            let mut c = vec![big(5)];
            c.resize(size, f0());
            c
        };
        let com_closure_bad = commit_column(&closure_bad).unwrap();
        let mut tv = transcript();
        assert!(
            verify_hash_semijoin(&com_closure_bad, 1, &com_x, &com_h, 1, m, &proof, &mut tv).is_err(),
            "kept hash not in the verifier's closure must reject"
        );
    }
}
