//! General k-atom conjunctive BGP as a LEFT-DEEP binary-join chain (Phase 7.6).
//!
//! A conjunctive basic graph pattern is a set of atoms `A_1..A_k`; the answer is the
//! natural join over shared variables. Planned left-deep: `T_1 = R_1`; for `i = 2..k`,
//! `T_i = T_{i-1} ⋈_v R_i` on the SINGLE variable `v` shared between the accumulator's
//! evolving schema and `R_i`. The 2-atom self-join capstone (`proof::query`) is the
//! `k = 2`, `R_1 == R_2` special case; this module generalizes the *composition* to an
//! arbitrary chain over already-committed input relations.
//!
//! This layer does NOT filter: it takes the input relations as GIVEN (each a set of
//! columns committed at some power-of-two domain, plus a real length) and proves the
//! join tree. That isolates the two soundness-critical continuity conditions as the
//! testable novel core; the layer-level wiring (filter each atom, then relocate into
//! the chain) lives in `proof::query`.
//!
//! v1 SCOPE: exactly ONE shared variable per join step — the plan names the single key
//! column on each side. Composite / multi-variable keys are deferred (7.3); the crypto
//! here proves the join on the single named key, and mapping columns to variables is
//! the planner's job.
//!
//! Every stage reuses the audited [`super::join2`] binary join verbatim and the
//! [`super::relocate`] cross-domain equality gadget; nothing about the join/sumcheck
//! math is reimplemented. The chain contributes exactly the two continuity checks:
//!
//!   1. **Commitment continuity.** Stage `i`'s join INPUT commitments are tied to stage
//!      `i-1`'s actual OUTPUT commitments via [`verify_column_relocation`] (the whole
//!      accumulator). The verifier never trusts a fresh input commitment that is not
//!      relocate-linked to the prior output — the relocation TARGET *is* the commitment
//!      the join consumes, so the two cannot diverge.
//!   2. **Real-length continuity.** Each intermediate relation's real length is NOT a
//!      free variable: stage `i`'s `n_r` is PINNED to stage `i-1`'s proven `m_j` (a
//!      public `usize` threaded by the verifier, never read from the proof). This is
//!      the FIX-2/3 discipline from the self-join audit — a zero-padding row must not
//!      be smuggle-able as a real join row.

use std::io;

use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::transcript::Transcript;

use super::argument::{commit_column, Transcript254};
use super::invalid;
use super::join2::{
    binary_join_commitments, binary_join_domain, compute_binary_join, prove_binary_join,
    verify_binary_join, BinaryJoinCommitments, BinaryJoinProof,
};
use super::relocate::{prove_column_relocation, verify_column_relocation, RelocationProof};

const CHAIN_DOMAIN: &[u8] = b"parture/proof-of-woql/bgp-left-deep-chain/v0";

#[inline]
fn f0() -> ArkFr {
    ArkFr::zero()
}

/// Zero-pad `v` up to a power-of-two domain `2^log_domain` (the layout every column
/// commitment uses). `v.len()` must not exceed the domain.
fn zero_pad(v: &[ArkFr], log_domain: usize) -> Vec<ArkFr> {
    let mut out = v.to_vec();
    out.resize(1usize << log_domain, f0());
    out
}

/// The join OUTPUT commitments in the canonical order `[r_other_out.., key,
/// s_other_out..]` — exactly what [`prove_binary_join`] returns and what becomes the
/// next stage's accumulator commitments.
fn output_commitments(com: &BinaryJoinCommitments) -> Vec<ArkGT> {
    let mut out = com.r_other_out.clone();
    out.push(com.key);
    out.extend(com.s_other_out.iter().copied());
    out
}

/// The join INPUT commitments for one side, listed in the ORIGINAL relation-column
/// order: the key column at `key_idx` maps to `key_com`, every other column maps to
/// `other_coms` in order. Used to line up each relocation target with its source
/// column positionally.
fn input_targets(key_com: ArkGT, other_coms: &[ArkGT], key_idx: usize, arity: usize) -> Vec<ArkGT> {
    let mut targets = Vec::with_capacity(arity);
    let mut d = 0usize;
    for c in 0..arity {
        if c == key_idx {
            targets.push(key_com);
        } else {
            targets.push(other_coms[d]);
            d += 1;
        }
    }
    targets
}

/// Split real-length columns into `(key_col, other_cols)` in column order.
fn split_key(cols: &[Vec<ArkFr>], key: usize) -> (&[ArkFr], Vec<&[ArkFr]>) {
    let key_col = cols[key].as_slice();
    let other: Vec<&[ArkFr]> = cols
        .iter()
        .enumerate()
        .filter(|(c, _)| *c != key)
        .map(|(_, col)| col.as_slice())
        .collect();
    (key_col, other)
}

/// A committed input relation on the VERIFIER side: per-column Dory commitments (all at
/// `2^log_domain`), the log-domain, and the real row count.
#[derive(Clone, Debug)]
pub struct RelationCommit {
    pub commitments: Vec<ArkGT>,
    pub log_domain: usize,
    pub n: usize,
}

/// A committed input relation on the PROVER side: the padded column data (each length
/// `2^log_domain`, zero-padded beyond `n`) plus the matching per-column commitments.
#[derive(Clone)]
pub struct RelationWitness {
    pub columns: Vec<Vec<ArkFr>>,
    pub commitments: Vec<ArkGT>,
    pub log_domain: usize,
    pub n: usize,
}

impl RelationWitness {
    /// Commit a relation from its REAL-length columns (each length `n`): pad to the
    /// canonical power-of-two domain (`>= 2`) and Dory-commit every column.
    pub fn commit(real_columns: Vec<Vec<ArkFr>>, n: usize) -> io::Result<Self> {
        if real_columns.is_empty() {
            return Err(invalid("relation must have at least one column"));
        }
        if real_columns.iter().any(|c| c.len() != n) {
            return Err(invalid("relation column length must equal the real row count"));
        }
        let padded_len = n
            .checked_next_power_of_two()
            .ok_or_else(|| invalid("relation domain overflow"))?
            .max(2);
        let log_domain = padded_len.trailing_zeros() as usize;
        let mut columns = Vec::with_capacity(real_columns.len());
        let mut commitments = Vec::with_capacity(real_columns.len());
        for col in &real_columns {
            let padded = zero_pad(col, log_domain);
            commitments.push(commit_column(&padded)?);
            columns.push(padded);
        }
        Ok(Self {
            columns,
            commitments,
            log_domain,
            n,
        })
    }

    /// The verifier-facing view (drops the column data).
    pub fn view(&self) -> RelationCommit {
        RelationCommit {
            commitments: self.commitments.clone(),
            log_domain: self.log_domain,
            n: self.n,
        }
    }
}

/// One left-deep join step: the key column index on the accumulator (LEFT) and on the
/// fresh relation (RIGHT). The accumulator's schema evolves as `[r_other_out.., key,
/// s_other_out..]`, so `acc_key` indexes into that evolving list at each step.
#[derive(Clone, Copy, Debug)]
pub struct JoinStep {
    pub acc_key: usize,
    pub rel_key: usize,
}

/// One stage's proof: the relocations that carry the accumulator's committed output
/// columns AND the fresh relation's committed columns into this stage's binary-join
/// input domain, plus the committed join instance, the join proof, and the proven
/// output length. `n_u` rides inside `join.n_u`.
#[derive(Clone, Debug)]
pub struct BgpStageProof {
    /// Relocation per accumulator column (prior output domain -> this r-input), in
    /// accumulator-column order.
    pub acc_relocations: Vec<RelocationProof>,
    /// Relocation per fresh-relation column (its own domain -> this s-input), in
    /// relation-column order.
    pub rel_relocations: Vec<RelocationProof>,
    /// Every input+output column committed at `binary_join_domain(n_r,n_s,m_j,n_u)`.
    pub commitments: BinaryJoinCommitments,
    pub join: BinaryJoinProof,
    pub m_j: usize,
}

/// A full left-deep chain proof: one stage per join step (`k - 1` of them).
#[derive(Clone, Debug)]
pub struct BgpChainProof {
    pub stages: Vec<BgpStageProof>,
}

/// Shared shape validation over the verifier-facing view.
fn validate_shape(relations: &[RelationCommit], plan: &[JoinStep]) -> io::Result<()> {
    if relations.len() < 2 {
        return Err(invalid("bgp chain needs at least two relations"));
    }
    if plan.len() != relations.len() - 1 {
        return Err(invalid("bgp plan must have exactly k-1 join steps"));
    }
    for rel in relations {
        if rel.commitments.is_empty() {
            return Err(invalid("bgp relation must have at least one column"));
        }
        if rel.log_domain == 0 || rel.log_domain >= usize::BITS as usize {
            return Err(invalid("bgp relation log-domain out of range"));
        }
        if rel.n > (1usize << rel.log_domain) {
            return Err(invalid("bgp relation real length exceeds its domain"));
        }
    }
    Ok(())
}

/// Bind the whole public statement into the shared transcript BEFORE any stage: the
/// domain separator, `k`, and every input relation's arity/domain/length/commitments,
/// then the plan's key indices. Identical on both sides.
fn bind_chain_statement(
    transcript: &mut Transcript254,
    relations: &[RelationCommit],
    plan: &[JoinStep],
) {
    transcript.append_bytes(b"pow/bgp/domain", CHAIN_DOMAIN);
    transcript.append_bytes(b"pow/bgp/k", &(relations.len() as u64).to_le_bytes());
    for rel in relations {
        transcript.append_bytes(b"pow/bgp/rel-arity", &(rel.commitments.len() as u64).to_le_bytes());
        transcript.append_bytes(b"pow/bgp/rel-log-domain", &(rel.log_domain as u64).to_le_bytes());
        transcript.append_bytes(b"pow/bgp/rel-n", &(rel.n as u64).to_le_bytes());
        for com in &rel.commitments {
            transcript.append_group(b"pow/bgp/rel-com", com);
        }
    }
    for step in plan {
        transcript.append_bytes(b"pow/bgp/acc-key", &(step.acc_key as u64).to_le_bytes());
        transcript.append_bytes(b"pow/bgp/rel-key", &(step.rel_key as u64).to_le_bytes());
    }
}

/// Prove the complete left-deep join chain over the given committed relations. Returns
/// the chain proof, the FINAL result column commitments (in `[r_other_out.., key,
/// s_other_out..]` order), and the final result length `m_J`.
pub fn prove_bgp_chain(
    relations: &[RelationWitness],
    plan: &[JoinStep],
    transcript: &mut Transcript254,
) -> io::Result<(BgpChainProof, Vec<ArkGT>, usize)> {
    let overrides = vec![None; plan.len()];
    prove_bgp_chain_inner(relations, plan, &overrides, transcript)
}

/// Prove the chain from EXPLICIT per-stage accumulator lengths (the honest entry passes
/// all `None`, so each stage uses the prior stage's proven `m_j`). Tests use the
/// override to inject a real-length-continuity attack (an inflated intermediate length).
pub(crate) fn prove_bgp_chain_inner(
    relations: &[RelationWitness],
    plan: &[JoinStep],
    r_len_overrides: &[Option<usize>],
    transcript: &mut Transcript254,
) -> io::Result<(BgpChainProof, Vec<ArkGT>, usize)> {
    let views: Vec<RelationCommit> = relations.iter().map(|r| r.view()).collect();
    validate_shape(&views, plan)?;
    if r_len_overrides.len() != plan.len() {
        return Err(invalid("bgp length-override count mismatch"));
    }
    for rel in relations {
        if rel.columns.len() != rel.commitments.len()
            || rel.columns.iter().any(|c| c.len() != 1usize << rel.log_domain)
        {
            return Err(invalid("bgp relation column layout does not match its commitments"));
        }
    }

    bind_chain_statement(transcript, &views, plan);

    // Accumulator starts as R_1 (relations[0]).
    let mut acc_cols: Vec<Vec<ArkFr>> = relations[0].columns.clone();
    let mut acc_coms: Vec<ArkGT> = relations[0].commitments.clone();
    let mut acc_domain_log = relations[0].log_domain;
    let mut acc_len = relations[0].n;

    let mut stages = Vec::with_capacity(plan.len());
    for (j, step) in plan.iter().enumerate() {
        let rel = &relations[j + 1];
        let acc_arity = acc_cols.len();
        let rel_arity = rel.columns.len();
        if step.acc_key >= acc_arity || step.rel_key >= rel_arity {
            return Err(invalid("bgp join key index out of range"));
        }

        // Real-length continuity: honest length is the prior stage's proven m_j.
        let r_len = r_len_overrides[j].unwrap_or(acc_len);
        if r_len > (1usize << acc_domain_log) {
            return Err(invalid("bgp accumulator length exceeds its domain"));
        }

        // Real-length input columns for the join gadget (it derives the domain itself).
        let acc_real: Vec<Vec<ArkFr>> = acc_cols.iter().map(|c| c[..r_len].to_vec()).collect();
        let rel_real: Vec<Vec<ArkFr>> = rel.columns.iter().map(|c| c[..rel.n].to_vec()).collect();
        let (r_key, r_other) = split_key(&acc_real, step.acc_key);
        let (s_key, s_other) = split_key(&rel_real, step.rel_key);

        let witness = compute_binary_join(r_key, &r_other, s_key, &s_other);
        let m_j = witness.m_j;
        let com = binary_join_commitments(r_key, &r_other, s_key, &s_other, &witness)?;
        let d_j = binary_join_domain(r_len, rel.n, m_j, witness.n_u)?;
        let d_j_log = d_j.trailing_zeros() as usize;

        // (1) Relocate every accumulator column: prior output commitment (com_source at
        // the prior domain) -> this stage's r-input commitment (com_target at d_j). This
        // is what ties the join input to the actual prior output (continuity #1).
        let acc_targets = input_targets(com.r_key, &com.r_other, step.acc_key, acc_arity);
        let mut acc_relocations = Vec::with_capacity(acc_arity);
        for c in 0..acc_arity {
            let target = zero_pad(&acc_real[c], d_j_log);
            acc_relocations.push(prove_column_relocation(
                &acc_cols[c],
                &acc_coms[c],
                &target,
                &acc_targets[c],
                transcript,
            )?);
        }

        // (2) Relocate every fresh-relation column: its own commitment -> this s-input.
        let rel_targets = input_targets(com.s_key, &com.s_other, step.rel_key, rel_arity);
        let mut rel_relocations = Vec::with_capacity(rel_arity);
        for c in 0..rel_arity {
            let target = zero_pad(&rel_real[c], d_j_log);
            rel_relocations.push(prove_column_relocation(
                &rel.columns[c],
                &rel.commitments[c],
                &target,
                &rel_targets[c],
                transcript,
            )?);
        }

        // (3) The audited binary join over the relocated inputs.
        let (join, out_coms, out_m) =
            prove_binary_join(r_key, &r_other, s_key, &s_other, &com, transcript)?;
        debug_assert_eq!(out_m, m_j);

        // Advance the accumulator to this stage's output (padded at d_j).
        let mut new_cols: Vec<Vec<ArkFr>> =
            witness.r_other_out.iter().map(|c| zero_pad(c, d_j_log)).collect();
        new_cols.push(zero_pad(&witness.key, d_j_log));
        new_cols.extend(witness.s_other_out.iter().map(|c| zero_pad(c, d_j_log)));
        acc_cols = new_cols;
        acc_coms = out_coms;
        acc_domain_log = d_j_log;
        acc_len = m_j;

        stages.push(BgpStageProof {
            acc_relocations,
            rel_relocations,
            commitments: com,
            join,
            m_j,
        });
    }

    Ok((BgpChainProof { stages }, acc_coms, acc_len))
}

/// Verify a left-deep chain proof against the committed input relations. Returns the
/// FINAL result column commitments and length. The verifier re-derives each stage's
/// domain from PINNED lengths and never rescans a relation.
pub fn verify_bgp_chain(
    relations: &[RelationCommit],
    plan: &[JoinStep],
    proof: &BgpChainProof,
    transcript: &mut Transcript254,
) -> io::Result<(Vec<ArkGT>, usize)> {
    validate_shape(relations, plan)?;
    if proof.stages.len() != plan.len() {
        return Err(invalid("bgp chain stage count does not match the plan"));
    }

    bind_chain_statement(transcript, relations, plan);

    let mut acc_coms = relations[0].commitments.clone();
    let mut acc_domain_log = relations[0].log_domain;
    let mut acc_len = relations[0].n;

    for (j, step) in plan.iter().enumerate() {
        let rel = &relations[j + 1];
        let stage = &proof.stages[j];
        let acc_arity = acc_coms.len();
        let rel_arity = rel.commitments.len();
        if step.acc_key >= acc_arity || step.rel_key >= rel_arity {
            return Err(invalid("bgp join key index out of range"));
        }
        // The committed stage schema must match the accumulator / relation arities.
        if stage.commitments.r_other.len() != acc_arity - 1
            || stage.commitments.r_other_out.len() != acc_arity - 1
            || stage.commitments.s_other.len() != rel_arity - 1
            || stage.commitments.s_other_out.len() != rel_arity - 1
        {
            return Err(invalid("bgp stage commitment arity mismatch"));
        }
        if stage.acc_relocations.len() != acc_arity || stage.rel_relocations.len() != rel_arity {
            return Err(invalid("bgp stage relocation count mismatch"));
        }

        // Real-length continuity (#2): n_r is PINNED to the prior stage's proven m_j
        // (or R_1's real length for the first stage) — NOT read from the proof. It
        // feeds both the domain derivation and the join's chi_{n_r} source mask, so an
        // inflated intermediate length cannot promote a padding row to a real one.
        let n_r = acc_len;
        let n_s = rel.n;
        let m_j = stage.m_j;
        let n_u = stage.join.n_u;
        let d_j = binary_join_domain(n_r, n_s, m_j, n_u)?;
        let d_j_log = d_j.trailing_zeros() as usize;

        // Commitment continuity (#1): each accumulator column's join input commitment
        // MUST be relocate-linked to the prior stage's actual output commitment. The
        // relocation TARGET is the very commitment the join consumes, so a fresh/tampered
        // input that is not the prior output is rejected here, before the join runs.
        let acc_targets = input_targets(
            stage.commitments.r_key,
            &stage.commitments.r_other,
            step.acc_key,
            acc_arity,
        );
        for c in 0..acc_arity {
            verify_column_relocation(
                &acc_coms[c],
                acc_domain_log,
                &acc_targets[c],
                d_j_log,
                &stage.acc_relocations[c],
                transcript,
            )?;
        }

        // Fresh-relation columns are likewise relocate-linked to their committed inputs.
        let rel_targets = input_targets(
            stage.commitments.s_key,
            &stage.commitments.s_other,
            step.rel_key,
            rel_arity,
        );
        for c in 0..rel_arity {
            verify_column_relocation(
                &rel.commitments[c],
                rel.log_domain,
                &rel_targets[c],
                d_j_log,
                &stage.rel_relocations[c],
                transcript,
            )?;
        }

        // The audited binary join, with n_r pinned to the prior length.
        verify_binary_join(&stage.commitments, n_r, n_s, m_j, &stage.join, transcript)?;

        // Advance the accumulator.
        acc_coms = output_commitments(&stage.commitments);
        acc_domain_log = d_j_log;
        acc_len = m_j;
    }

    Ok((acc_coms, acc_len))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frs(vals: &[u64]) -> Vec<ArkFr> {
        vals.iter().map(|v| ArkFr::from_u64(*v)).collect()
    }

    fn transcript() -> Transcript254 {
        Transcript254::new(b"parture-bgp-test")
    }

    // Commit a relation from its real-length columns.
    fn relation(cols: &[&[u64]], n: usize) -> RelationWitness {
        let real: Vec<Vec<ArkFr>> = cols.iter().map(|c| frs(c)).collect();
        RelationWitness::commit(real, n).unwrap()
    }

    // A 3-relation left-deep path chain:
    //   R1(x, y): (1,10),(2,10),(3,20)            key y (col 1)
    //   R2(y, z): (10,100),(20,200),(20,201)      key y (col 0)  -> acc [x, y, z]
    //   R3(z, w): (100,1000),(201,2010)           key z (col 0)  -> acc [x, y, z, w]
    // Stage 0 (⋈ y) -> {(1,10,100),(2,10,100),(3,20,200),(3,20,201)}  m0 = 4.
    // Stage 1 (⋈ z) -> {(1,10,100,1000),(2,10,100,1000),(3,20,201,2010)}  m1 = 3.
    fn path_chain() -> (Vec<RelationWitness>, Vec<JoinStep>) {
        let r1 = relation(&[&[1, 2, 3], &[10, 10, 20]], 3);
        let r2 = relation(&[&[10, 20, 20], &[100, 200, 201]], 3);
        let r3 = relation(&[&[100, 201], &[1000, 2010]], 2);
        let plan = vec![
            JoinStep { acc_key: 1, rel_key: 0 }, // acc.y == R2.y
            JoinStep { acc_key: 2, rel_key: 0 }, // acc.z == R3.z
        ];
        (vec![r1, r2, r3], plan)
    }

    #[test]
    fn honest_three_atom_chain_verifies_with_exact_result() {
        let (rels, plan) = path_chain();
        let views: Vec<RelationCommit> = rels.iter().map(|r| r.view()).collect();

        let mut pt = transcript();
        let (proof, final_coms, m) = prove_bgp_chain(&rels, &plan, &mut pt).unwrap();
        assert_eq!(m, 3);
        assert_eq!(proof.stages.len(), 2);
        assert_eq!(proof.stages[0].m_j, 4);

        let mut vt = transcript();
        let (v_coms, v_m) = verify_bgp_chain(&views, &plan, &proof, &mut vt).unwrap();
        assert_eq!(v_m, 3);
        assert_eq!(v_coms, final_coms);

        // Independent recomputation of the final [x, y, z, w] result, padded to the
        // final domain. n_u at stage 1 = |{100,200,201} ∪ {100,201}| = 3, m1 = 3,
        // n_r = 4, n_s = 2 -> domain = next_pow2(4) = 4.
        assert_eq!(v_coms.len(), 4);
        let final_log = 2usize; // domain 4
        let exp_x = zero_pad(&frs(&[1, 2, 3]), final_log);
        let exp_y = zero_pad(&frs(&[10, 10, 20]), final_log);
        let exp_z = zero_pad(&frs(&[100, 100, 201]), final_log);
        let exp_w = zero_pad(&frs(&[1000, 1000, 2010]), final_log);
        assert_eq!(v_coms[0], commit_column(&exp_x).unwrap());
        assert_eq!(v_coms[1], commit_column(&exp_y).unwrap());
        assert_eq!(v_coms[2], commit_column(&exp_z).unwrap());
        assert_eq!(v_coms[3], commit_column(&exp_w).unwrap());
    }

    #[test]
    fn empty_intermediate_result_chain_verifies() {
        // Stage 0 produces nothing (no shared y), so the whole chain is empty.
        let r1 = relation(&[&[1, 2], &[10, 20]], 2);
        let r2 = relation(&[&[30, 40], &[100, 200]], 2); // no y in {10,20}
        let r3 = relation(&[&[100], &[1000]], 1);
        let plan = vec![
            JoinStep { acc_key: 1, rel_key: 0 },
            JoinStep { acc_key: 2, rel_key: 0 },
        ];
        let rels = vec![r1, r2, r3];
        let views: Vec<RelationCommit> = rels.iter().map(|r| r.view()).collect();
        let mut pt = transcript();
        let (proof, _c, m) = prove_bgp_chain(&rels, &plan, &mut pt).unwrap();
        assert_eq!(m, 0);
        assert_eq!(proof.stages[0].m_j, 0);
        let mut vt = transcript();
        let (_v, v_m) = verify_bgp_chain(&views, &plan, &proof, &mut vt).unwrap();
        assert_eq!(v_m, 0);
    }

    // ---- ATTACK 1: commitment continuity ----

    #[test]
    fn tampered_stage_input_commitment_breaks_relocation_link() {
        // Honest chain, then swap stage 1's accumulator KEY input commitment (which must
        // equal stage 0's output z-commitment) for a commitment to a DIFFERENT column.
        // The stage-1 accumulator relocation ties stage 0's real output z-column to this
        // tampered target -> its Dory opening fails -> reject. This is exactly "feed
        // stage i a different input commitment than stage i-1 output".
        let (rels, plan) = path_chain();
        let views: Vec<RelationCommit> = rels.iter().map(|r| r.view()).collect();
        let mut pt = transcript();
        let (proof, _c, _m) = prove_bgp_chain(&rels, &plan, &mut pt).unwrap();

        // Sanity: the honest proof verifies.
        let mut vt = transcript();
        assert!(verify_bgp_chain(&views, &plan, &proof, &mut vt).is_ok());

        // stage 1 (index 1) joins on acc_key = 2 (z), so r_key is the z input. The
        // stage-1 join domain is 4 (log 2); commit a DIFFERENT column at that domain so
        // the rejection is the relocation link failing, not a size mismatch.
        let bogus = commit_column(&zero_pad(&frs(&[424242]), 2)).unwrap();
        let mut tampered = proof.clone();
        tampered.stages[1].commitments.r_key = bogus;
        let mut vt2 = transcript();
        assert!(verify_bgp_chain(&views, &plan, &tampered, &mut vt2).is_err());
    }

    // ---- ATTACK 2: real-length continuity ----

    // A domain-stable 3-relation chain: R3 is wide enough (n = 6, n_u = 6/7) that the
    // final join domain is next_pow2(6/7) = 8 whether the accumulator length is the
    // honest m0 = 2 or the inflated 3, so the ONLY difference an inflated intermediate
    // length makes is the pinned n_r (chi_{n_r} + bind_instance) -> the join rejects.
    // R1 carries a third, NON-joining row (9,99) purely so stage 0's output domain is 4
    // (not 2), leaving a genuine padding row at index 2 for the attack to try to promote.
    //   R1(x, y): (1,10),(2,10),(9,99)  key y (col 1)  -> stage-0 domain 4, m0 = 2
    //   R2(y, z): (10,100)              key y (col 0)  -> acc [x, y, z]
    //   R3(z, w): (100,1000) + 5 non-matching rows     key z (col 0)
    fn wide_chain() -> (Vec<RelationWitness>, Vec<JoinStep>) {
        let r1 = relation(&[&[1, 2, 9], &[10, 10, 99]], 3);
        let r2 = relation(&[&[10], &[100]], 1);
        let r3 = relation(
            &[&[100, 50, 51, 52, 53, 54], &[1000, 1, 2, 3, 4, 5]],
            6,
        );
        let plan = vec![
            JoinStep { acc_key: 1, rel_key: 0 },
            JoinStep { acc_key: 2, rel_key: 0 },
        ];
        (vec![r1, r2, r3], plan)
    }

    #[test]
    fn honest_wide_chain_verifies() {
        let (rels, plan) = wide_chain();
        let views: Vec<RelationCommit> = rels.iter().map(|r| r.view()).collect();
        let mut pt = transcript();
        let (proof, _c, m) = prove_bgp_chain(&rels, &plan, &mut pt).unwrap();
        assert_eq!(proof.stages[0].m_j, 2); // m0
        assert_eq!(m, 2); // (1,10,100,1000),(2,10,100,1000)
        let mut vt = transcript();
        assert!(verify_bgp_chain(&views, &plan, &proof, &mut vt).is_ok());
    }

    #[test]
    fn inflated_intermediate_length_is_rejected() {
        // Prove stage 1 with n_r = m0 + 1 = 3 (promoting one accumulator padding row to
        // "real"), keeping stage 0 honest (it still proves m0 = 2). The verifier threads
        // n_r = stage0.m_j = 2 and rejects the stage-1 join whose transcript/shape were
        // built with n_r = 3.
        let (rels, plan) = wide_chain();
        let views: Vec<RelationCommit> = rels.iter().map(|r| r.view()).collect();

        let overrides = vec![None, Some(3usize)]; // inflate stage 1's accumulator length
        let mut pt = transcript();
        let (proof, _c, _m) = prove_bgp_chain_inner(&rels, &plan, &overrides, &mut pt).unwrap();
        // The dishonest stage still reports the honest prior m_j (stage 0 is untouched).
        assert_eq!(proof.stages[0].m_j, 2);

        let mut vt = transcript();
        assert!(verify_bgp_chain(&views, &plan, &proof, &mut vt).is_err());
    }

    #[test]
    fn wrong_key_index_is_rejected() {
        let (rels, plan) = path_chain();
        let views: Vec<RelationCommit> = rels.iter().map(|r| r.view()).collect();
        let mut pt = transcript();
        let (proof, _c, _m) = prove_bgp_chain(&rels, &plan, &mut pt).unwrap();
        // A verifier plan that names a key column out of range fails closed.
        let mut bad_plan = plan.clone();
        bad_plan[1].acc_key = 9;
        let mut vt = transcript();
        assert!(verify_bgp_chain(&views, &bad_plan, &proof, &mut vt).is_err());
    }
}
