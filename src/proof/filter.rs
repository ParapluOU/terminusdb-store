//! Q1 filter selection-indicator (IsZero) gadget — the first real query-correctness
//! proof of the Proof-of-WOQL argument.
//!
//! Given a committed predicate column `Com(P)` over `N = 2^k` rows (`n` of them
//! real, the rest padding) and a PUBLIC constant predicate id `pred`, this gadget
//! proves that a committed selection vector `sel` equals EXACTLY
//! `[ P_i == pred AND i < n ]` for every row — filter correct AND complete (no
//! matching row omitted, no non-matching/padding row selected).
//!
//! Blueprint (IsZero): the prover commits `sel in {0,1}^N` and a pseudo-inverse
//! witness `w`, with `lhs_i = P_i - pred`, and proves two Identity (zero-at-every-
//! row) claims via the batched sumcheck ([`super::argument`]):
//!   * A:  `sel_i * lhs_i = 0`
//!   * B:  `(chi_n_i - sel_i) - lhs_i * w_i = 0`
//! These pin `sel_i = [P_i == pred AND i < n]` uniquely (booleanity subsumed).
//! `chi_n` is PUBLIC (the real-row count `n` lives in the layer root), so it is
//! never committed: the verifier derives `chi_n(r)` closed-form
//! ([`super::argument::chi_n_eval`]), which also masks the padding-collision
//! hazard (a padding row whose `P` equals `pred` must still have `sel = 0`).
//!
//! Honest witnesses:
//!   * `sel_i = 1` iff `i < n && P_i == pred`.
//!   * `w_i = lhs_i^{-1}` iff `i < n && lhs_i != 0`, else `0` (the `i < n` mask is
//!     required: on padding rows B forces `lhs_i * w_i = 0`, so `w` must be 0 on
//!     padding non-matches). On matching rows (`lhs_i = 0`) `w` is unconstrained.
//!
//! Phase 2b-1 only: NO LogUp compaction to a dense output (that is 2b-2).

use std::io;

use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;
use dory_pcs::primitives::transcript::Transcript;

use super::argument::{
    commit_column, prove_column_openings, prove_sumcheck, verify_column_openings, verify_sumcheck,
    ClaimShape, ColumnOpening, Expr, Operand, SumcheckClaim, SumcheckProof, SumcheckShape, Term,
    Transcript254,
};
use super::invalid;
use super::logup::{
    multiset_link_claim, pseudo_inverse, reciprocal_claim, reciprocal_witness, rlc_fold,
};

// Column ordering. The 2b-1 indicator gadget uses [P, sel, w].
const COL_P: usize = 0;
const COL_SEL: usize = 1;
const COL_W: usize = 2;
// chi_bounds indices: chi_n (real-row count n) and chi_m (output length m).
const CHI_N: usize = 0;
const CHI_M: usize = 1;

// Generalized per-atom filter column layout (filter on ANY subset of {S,P,O}). The
// base columns come first so a constant position folds directly into claims A/B; the
// kept (variable) output columns follow, then the two reciprocal columns, then the
// optional unit INPUT column used only for a ground atom (|C| = 3).
const A_COL_S: usize = 0;
const A_COL_P: usize = 1;
const A_COL_O: usize = 2;
const A_COL_SEL: usize = 3;
const A_COL_W: usize = 4;
const A_COL_KEPT_BASE: usize = 5;

#[inline]
fn f0() -> ArkFr {
    ArkFr::zero()
}
#[inline]
fn f1() -> ArkFr {
    ArkFr::one()
}

/// A filter-indicator proof: the batched sumcheck over the two Identity claims,
/// plus one batched Dory opening of `[P, sel, w]` at the sumcheck point `r`.
#[derive(Clone, Debug)]
pub struct FilterProof {
    pub sumcheck: SumcheckProof,
    pub opening: ColumnOpening,
}

/// The two Identity claims A and B over columns `[P, sel, w]`, with `chi_n(n)`
/// and the public constant `pred` folded into the expression coefficients.
fn filter_shape(k: usize, real_rows: usize, pred: ArkFr) -> SumcheckShape {
    let neg_one = f0() - f1();
    let neg_pred = f0() - pred;

    // A: sel * (P - pred) = sel*P + (-pred)*sel
    let claim_a = SumcheckClaim {
        shape: ClaimShape::Identity,
        expr: Expr {
            terms: vec![
                Term {
                    coeff: f1(),
                    operands: vec![Operand::Column(COL_SEL), Operand::Column(COL_P)],
                },
                Term {
                    coeff: neg_pred,
                    operands: vec![Operand::Column(COL_SEL)],
                },
            ],
        },
        target: f0(),
    };

    // B: (chi_n - sel) - (P - pred)*w = chi_n - sel - P*w + pred*w
    let claim_b = SumcheckClaim {
        shape: ClaimShape::Identity,
        expr: Expr {
            terms: vec![
                Term {
                    coeff: f1(),
                    operands: vec![Operand::Chi(CHI_N)],
                },
                Term {
                    coeff: neg_one,
                    operands: vec![Operand::Column(COL_SEL)],
                },
                Term {
                    coeff: neg_one,
                    operands: vec![Operand::Column(COL_P), Operand::Column(COL_W)],
                },
                Term {
                    coeff: pred,
                    operands: vec![Operand::Column(COL_W)],
                },
            ],
        },
        target: f0(),
    };

    SumcheckShape {
        num_vars: k,
        num_columns: 3,
        chi_bounds: vec![real_rows],
        claims: vec![claim_a, claim_b],
    }
}

/// Compute the honest `(sel, w)` witnesses for a predicate column.
pub(crate) fn honest_witness(
    predicate_column: &[ArkFr],
    pred: ArkFr,
    real_rows: usize,
) -> (Vec<ArkFr>, Vec<ArkFr>) {
    let size = predicate_column.len();
    let lhs: Vec<ArkFr> = predicate_column.iter().map(|p| *p - pred).collect();
    let sel: Vec<ArkFr> = (0..size)
        .map(|i| {
            if i < real_rows && lhs[i].is_zero() {
                f1()
            } else {
                f0()
            }
        })
        .collect();
    // w is the pseudo-inverse of lhs masked to the real rows [0, n): 0 on padding
    // rows (B forces lhs*w = 0 there) and on matching rows (lhs = 0).
    let masked_lhs: Vec<ArkFr> = (0..size)
        .map(|i| if i < real_rows { lhs[i] } else { f0() })
        .collect();
    let w = pseudo_inverse(&masked_lhs);
    (sel, w)
}

/// Prove the filter indicator for an EXPLICIT witness `(sel, w)`. The honest
/// entry point [`prove_filter_indicator`] derives the witness and calls this;
/// tests use it to inject malicious witnesses.
pub(crate) fn prove_filter_with_witness(
    predicate_column: &[ArkFr],
    pred: ArkFr,
    real_rows: usize,
    sel: &[ArkFr],
    w: &[ArkFr],
    predicate_commitment: &ArkGT,
    transcript: &mut Transcript254,
) -> io::Result<(FilterProof, ArkGT, ArkGT, ArkFr)> {
    let size = predicate_column.len();
    if !size.is_power_of_two() {
        return Err(invalid("predicate column length must be a power of two"));
    }
    if sel.len() != size || w.len() != size {
        return Err(invalid("witness length does not match the predicate column"));
    }
    if real_rows > size {
        return Err(invalid("real-row count exceeds column length"));
    }
    let k = size.trailing_zeros() as usize;

    let sel_commitment = commit_column(sel)?;
    let w_commitment = commit_column(w)?;

    // Bind every column commitment BEFORE the zero-check point is drawn.
    transcript.append_group(b"pow/filter/com-p", predicate_commitment);
    transcript.append_group(b"pow/filter/com-sel", &sel_commitment);
    transcript.append_group(b"pow/filter/com-w", &w_commitment);

    let shape = filter_shape(k, real_rows, pred);
    let columns = vec![predicate_column.to_vec(), sel.to_vec(), w.to_vec()];
    let (sumcheck, r, evals) = prove_sumcheck(&shape, &columns, transcript)?;
    let opening = prove_column_openings(&columns, &r, transcript)?;

    Ok((
        FilterProof { sumcheck, opening },
        sel_commitment,
        w_commitment,
        evals[COL_SEL],
    ))
}

/// Prove that a filter selection vector is correct and complete for the given
/// predicate column and PUBLIC constant. Commits `sel` and `w` internally.
/// Returns the proof, `Com(sel)`, `Com(w)`, and `sel(r)` (for chaining into the
/// 2b-2 compaction).
pub fn prove_filter_indicator(
    predicate_column: &[ArkFr],
    pred: ArkFr,
    real_rows: usize,
    predicate_commitment: &ArkGT,
    transcript: &mut Transcript254,
) -> io::Result<(FilterProof, ArkGT, ArkGT, ArkFr)> {
    let (sel, w) = honest_witness(predicate_column, pred, real_rows);
    prove_filter_with_witness(
        predicate_column,
        pred,
        real_rows,
        &sel,
        &w,
        predicate_commitment,
        transcript,
    )
}

/// Verify a filter-indicator proof. Re-derives the zero-check point, checks the
/// batched sumcheck against the claimed evals + `chi_n(r)` + `(eval(P) - pred)`,
/// and verifies the batched Dory opening at `r`. Returns `sel(r)` for chaining.
pub fn verify_filter_indicator(
    predicate_commitment: &ArkGT,
    pred: ArkFr,
    real_rows: usize,
    sel_commitment: &ArkGT,
    w_commitment: &ArkGT,
    proof: &FilterProof,
    transcript: &mut Transcript254,
) -> io::Result<ArkFr> {
    // k is the number of sumcheck rounds; the columns' size (2^k) is bound by the
    // Dory opening, which verifies the commitments against a length-k point.
    let k = proof.sumcheck.round_polys.len();
    if k == 0 || k >= usize::BITS as usize {
        return Err(invalid("filter proof has an implausible variable count"));
    }
    if real_rows > (1usize << k) {
        return Err(invalid("real-row count exceeds column length"));
    }

    transcript.append_group(b"pow/filter/com-p", predicate_commitment);
    transcript.append_group(b"pow/filter/com-sel", sel_commitment);
    transcript.append_group(b"pow/filter/com-w", w_commitment);

    let shape = filter_shape(k, real_rows, pred);
    let (r, evals) = verify_sumcheck(&shape, &proof.sumcheck, transcript)?;
    verify_column_openings(
        &[*predicate_commitment, *sel_commitment, *w_commitment],
        &r,
        &evals,
        &proof.opening,
        transcript,
    )?;
    Ok(evals[COL_SEL])
}

// ===========================================================================
// Full Q1: filter indicator + LogUp compaction to the dense (S_out, O_out).
// ===========================================================================

/// A complete Q1 filter query proof: the batched sumcheck over the two indicator
/// claims (A, B) AND the three LogUp compaction claims (C, D, E), one batched
/// Dory opening of all nine columns at `r`, and the internal witness commitments
/// the verifier binds into the transcript.
#[derive(Clone, Debug)]
pub struct FullFilterProof {
    pub sumcheck: SumcheckProof,
    pub opening: ColumnOpening,
    pub sel_commitment: ArkGT,
    pub w_commitment: ArkGT,
    pub cstar_commitment: ArkGT,
    pub dstar_commitment: ArkGT,
}

/// The variable/constant positions of a triple atom.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Position {
    S,
    P,
    O,
}

impl Position {
    fn base_col(self) -> usize {
        match self {
            Position::S => A_COL_S,
            Position::P => A_COL_P,
            Position::O => A_COL_O,
        }
    }
    fn tag(self) -> u8 {
        match self {
            Position::S => 0,
            Position::P => 1,
            Position::O => 2,
        }
    }
}

/// A per-atom filter: the constant positions (with their constant ids) and the KEPT
/// (variable) positions in atom order. `kept` must be exactly `{S,P,O}` minus the
/// constant positions.
#[derive(Clone, Debug)]
pub struct AtomFilterSpec {
    pub constants: Vec<(Position, ArkFr)>,
    pub kept: Vec<Position>,
}

/// A kept (output) column source: a variable base position, or the public all-ones
/// UNIT column used for a ground atom (|C| = 3) so C/D/E runs uniformly.
#[derive(Clone, Copy, Debug)]
enum KeptSource {
    Base(Position),
    Unit,
}

impl AtomFilterSpec {
    /// The classic predicate-only filter: `P` constant, keep `S` and `O`.
    pub fn predicate(pred: ArkFr) -> Self {
        Self {
            constants: vec![(Position::P, pred)],
            kept: vec![Position::S, Position::O],
        }
    }

    fn validate(&self) -> io::Result<()> {
        if self.constants.len() + self.kept.len() != 3 {
            return Err(invalid("atom filter spec must cover S, P, O exactly once"));
        }
        let mut is_constant = [false; 3];
        for (pos, _) in &self.constants {
            if is_constant[pos.tag() as usize] {
                return Err(invalid("duplicate constant position in atom filter"));
            }
            is_constant[pos.tag() as usize] = true;
        }
        let expected: Vec<Position> = [Position::S, Position::P, Position::O]
            .iter()
            .copied()
            .filter(|p| !is_constant[p.tag() as usize])
            .collect();
        if self.kept != expected {
            return Err(invalid(
                "atom filter kept positions must be the constants' complement in atom order",
            ));
        }
        Ok(())
    }

    /// Kept-column sources: the variable positions, or a single UNIT column for a
    /// ground atom (no variable positions).
    fn kept_sources(&self) -> Vec<KeptSource> {
        if self.kept.is_empty() {
            vec![KeptSource::Unit]
        } else {
            self.kept.iter().map(|p| KeptSource::Base(*p)).collect()
        }
    }
}

/// A per-atom filter proof. `Identity` is the |C| = 0 case (`t(?s,?p,?o)`): the result
/// is every real row, so the kept columns ARE the base columns and no sumcheck is
/// produced. `Compacted` carries the selection + compaction argument (|C| >= 1).
#[derive(Clone, Debug)]
pub enum AtomFilterProof {
    Identity,
    Compacted(FullFilterProof),
}

fn a_col_cstar(num_kept: usize) -> usize {
    A_COL_KEPT_BASE + num_kept
}
fn a_col_dstar(num_kept: usize) -> usize {
    A_COL_KEPT_BASE + num_kept + 1
}
fn a_col_unit_in(num_kept: usize) -> usize {
    A_COL_KEPT_BASE + num_kept + 2
}

/// Selection claims A and B generalized over the constant positions, folded with a
/// PUBLIC Fiat-Shamir coefficient `beta_c` baked into the term coefficients (no new
/// committed column):
///   lhs_i = Σ_r beta_c^r * (Col_{pos_r}[i] - c_r)
///   A: sel * lhs = 0        B: (chi_n - sel) - lhs * w = 0
/// For |C| = 1 this is exactly the classic Q1 (a single term, `beta_c^0 = 1`).
fn selection_claims(spec: &AtomFilterSpec, beta_c: ArkFr) -> (SumcheckClaim, SumcheckClaim) {
    let mut a_terms = Vec::with_capacity(2 * spec.constants.len());
    let mut b_terms = vec![
        Term {
            coeff: f1(),
            operands: vec![Operand::Chi(CHI_N)],
        },
        Term {
            coeff: f0() - f1(),
            operands: vec![Operand::Column(A_COL_SEL)],
        },
    ];
    let mut power = f1();
    for (pos, c) in &spec.constants {
        let col = pos.base_col();
        // A: beta_c^r * sel*Col - (beta_c^r * c) * sel
        a_terms.push(Term {
            coeff: power,
            operands: vec![Operand::Column(A_COL_SEL), Operand::Column(col)],
        });
        a_terms.push(Term {
            coeff: f0() - power * *c,
            operands: vec![Operand::Column(A_COL_SEL)],
        });
        // B: -(beta_c^r) * Col*w + (beta_c^r * c) * w
        b_terms.push(Term {
            coeff: f0() - power,
            operands: vec![Operand::Column(col), Operand::Column(A_COL_W)],
        });
        b_terms.push(Term {
            coeff: power * *c,
            operands: vec![Operand::Column(A_COL_W)],
        });
        power = power * beta_c;
    }
    (
        SumcheckClaim {
            shape: ClaimShape::Identity,
            expr: Expr { terms: a_terms },
            target: f0(),
        },
        SumcheckClaim {
            shape: ClaimShape::Identity,
            expr: Expr { terms: b_terms },
            target: f0(),
        },
    )
}

/// The full atom-filter shape: selection (A, B) + compaction (C, D, E) over the kept
/// columns. The compaction half is arity-generic (folds the kept INPUT columns for
/// `cstar` and the kept OUTPUT columns for `dstar`).
#[allow(clippy::too_many_arguments)]
fn atom_shape(
    spec: &AtomFilterSpec,
    num_vars: usize,
    real_rows: usize,
    output_len: usize,
    num_kept: usize,
    has_unit: bool,
    beta_c: ArkFr,
    alpha: ArkFr,
    beta: ArkFr,
) -> SumcheckShape {
    let (claim_a, claim_b) = selection_claims(spec, beta_c);

    let kept_in: Vec<usize> = if has_unit {
        vec![a_col_unit_in(num_kept)]
    } else {
        spec.kept.iter().map(|p| p.base_col()).collect()
    };
    let kept_out: Vec<usize> = (0..num_kept).map(|d| A_COL_KEPT_BASE + d).collect();
    let cstar = a_col_cstar(num_kept);
    let dstar = a_col_dstar(num_kept);

    // C: cstar reciprocal over the kept INPUT fold; D: dstar over the kept OUTPUT fold.
    let claim_c = reciprocal_claim(cstar, &kept_in, alpha, beta);
    let claim_d = reciprocal_claim(dstar, &kept_out, alpha, beta);
    // E (LogUp link): Σ sel*cstar - Σ chi_m*dstar = 0.
    let claim_e = multiset_link_claim(Operand::Column(A_COL_SEL), cstar, Operand::Chi(CHI_M), dstar);

    let num_columns = a_col_unit_in(num_kept) + usize::from(has_unit);
    SumcheckShape {
        num_vars,
        num_columns,
        chi_bounds: vec![real_rows, output_len],
        claims: vec![claim_a, claim_b, claim_c, claim_d, claim_e],
    }
}

const A_LABEL_DOMAIN: &[u8] = b"pow/atom-filter/domain-v1";
const A_LABEL_SPEC: &[u8] = b"pow/atom-filter/spec";
const A_LABEL_BETA_C: &[u8] = b"pow/atom-filter/const-beta";
const A_LABEL_BETA: &[u8] = b"pow/atom-filter/logup-beta";
const A_LABEL_ALPHA: &[u8] = b"pow/atom-filter/logup-alpha";

fn base_column<'a>(pos: Position, s: &'a [ArkFr], p: &'a [ArkFr], o: &'a [ArkFr]) -> &'a [ArkFr] {
    match pos {
        Position::S => s,
        Position::P => p,
        Position::O => o,
    }
}

fn base_commitment(pos: Position, com_s: &ArkGT, com_p: &ArkGT, com_o: &ArkGT) -> ArkGT {
    match pos {
        Position::S => *com_s,
        Position::P => *com_p,
        Position::O => *com_o,
    }
}

/// Bind the PUBLIC atom spec (constant positions + values, kept positions) so the
/// challenges depend on the whole statement.
fn bind_spec(transcript: &mut Transcript254, spec: &AtomFilterSpec) {
    transcript.append_bytes(A_LABEL_DOMAIN, b"parture/proof-of-woql/atom-filter/v0");
    transcript.append_bytes(A_LABEL_SPEC, &(spec.constants.len() as u64).to_le_bytes());
    for (pos, c) in &spec.constants {
        transcript.append_bytes(A_LABEL_SPEC, &[pos.tag()]);
        transcript.append_field(A_LABEL_SPEC, c);
    }
    for pos in &spec.kept {
        transcript.append_bytes(A_LABEL_SPEC, &[pos.tag()]);
    }
}

/// Honest selection: `sel_i = 1` iff `i < real_rows` AND every constant position matches.
fn atom_sel(
    spec: &AtomFilterSpec,
    s: &[ArkFr],
    p: &[ArkFr],
    o: &[ArkFr],
    real_rows: usize,
) -> Vec<ArkFr> {
    (0..s.len())
        .map(|i| {
            let matches = i < real_rows
                && spec
                    .constants
                    .iter()
                    .all(|(pos, c)| base_column(*pos, s, p, o)[i] == *c);
            if matches {
                f1()
            } else {
                f0()
            }
        })
        .collect()
}

/// Honest pseudo-inverse witness `w` for the constant RLC residual (needs `beta_c`):
/// `w_i = 1/lhs_i` on in-range non-matching rows, `0` on padding and matching rows.
fn atom_w(
    spec: &AtomFilterSpec,
    s: &[ArkFr],
    p: &[ArkFr],
    o: &[ArkFr],
    real_rows: usize,
    beta_c: ArkFr,
) -> Vec<ArkFr> {
    let masked_lhs: Vec<ArkFr> = (0..s.len())
        .map(|i| {
            if i >= real_rows {
                return f0();
            }
            let mut acc = f0();
            let mut power = f1();
            for (pos, c) in &spec.constants {
                acc = acc + power * (base_column(*pos, s, p, o)[i] - *c);
                power = power * beta_c;
            }
            acc
        })
        .collect();
    pseudo_inverse(&masked_lhs)
}

/// The kept-source INPUT columns (full length), in kept order.
fn kept_source_columns(
    spec: &AtomFilterSpec,
    s: &[ArkFr],
    p: &[ArkFr],
    o: &[ArkFr],
) -> Vec<Vec<ArkFr>> {
    spec.kept_sources()
        .into_iter()
        .map(|src| match src {
            KeptSource::Base(pos) => base_column(pos, s, p, o).to_vec(),
            KeptSource::Unit => vec![f1(); s.len()],
        })
        .collect()
}

/// Compact several columns in lockstep by `sel` (dense prefix + zero tail); returns
/// one output column per input and the match count `m`.
fn compact_columns(sources: &[Vec<ArkFr>], sel: &[ArkFr]) -> (Vec<Vec<ArkFr>>, usize) {
    let size = sel.len();
    let mut outs: Vec<Vec<ArkFr>> = sources.iter().map(|_| vec![f0(); size]).collect();
    let mut m = 0usize;
    for i in 0..size {
        if sel[i] == f1() {
            for (d, src) in sources.iter().enumerate() {
                outs[d][m] = src[i];
            }
            m += 1;
        }
    }
    (outs, m)
}

/// Compact the selected rows of `(S, O)` into a dense output. Retained for the query
/// layer's honest recomputation of the predicate-only result.
pub(crate) fn compact_selected(
    s: &[ArkFr],
    o: &[ArkFr],
    sel: &[ArkFr],
) -> (Vec<ArkFr>, Vec<ArkFr>, usize) {
    let (mut outs, m) = compact_columns(&[s.to_vec(), o.to_vec()], sel);
    let o_out = outs.pop().unwrap();
    let s_out = outs.pop().unwrap();
    (s_out, o_out, m)
}

fn slices(cols: &[Vec<ArkFr>]) -> Vec<&[ArkFr]> {
    cols.iter().map(|c| c.as_slice()).collect()
}

/// The DENSE kept OUTPUT columns that [`prove_atom_filter`] commits (`com_kept`),
/// plus the match count `m`, WITHOUT rerunning the sumcheck. For |C| = 0 the kept
/// columns ARE the base columns (full padded length, `m = real_rows`); for |C| >= 1
/// they are the `sel`-compaction of the kept-source columns (dense prefix in
/// `[0, m)`, zero tail, each length `s.len()`). The returned columns equal what
/// `prove_atom_filter` committed bit-for-bit — it derives them via the very same
/// `kept_source_columns`/`atom_sel`/`compact_columns` pipeline — so the query layer
/// can build a chain relation whose per-column commitments are exactly `com_kept`.
pub(crate) fn atom_kept_columns(
    spec: &AtomFilterSpec,
    s: &[ArkFr],
    p: &[ArkFr],
    o: &[ArkFr],
    real_rows: usize,
) -> io::Result<(Vec<Vec<ArkFr>>, usize)> {
    spec.validate()?;
    if spec.constants.is_empty() {
        let cols = spec
            .kept
            .iter()
            .map(|pos| base_column(*pos, s, p, o).to_vec())
            .collect();
        return Ok((cols, real_rows));
    }
    let sources = kept_source_columns(spec, s, p, o);
    let sel = atom_sel(spec, s, p, o, real_rows);
    Ok(compact_columns(&sources, &sel))
}

/// Prove the selection + compaction argument for an EXPLICIT `sel`/`kept_out` witness
/// (the honest entry derives them; tests inject malicious ones). `sources` are the
/// kept-source INPUT columns; `w`/`cstar`/`dstar` are always derived honestly (`w`
/// from the Fiat-Shamir `beta_c`).
#[allow(clippy::too_many_arguments)]
fn prove_compacted_with_witness(
    spec: &AtomFilterSpec,
    s: &[ArkFr],
    p: &[ArkFr],
    o: &[ArkFr],
    real_rows: usize,
    sel: &[ArkFr],
    sources: &[Vec<ArkFr>],
    kept_out: &[Vec<ArkFr>],
    output_len: usize,
    com_s: &ArkGT,
    com_p: &ArkGT,
    com_o: &ArkGT,
    transcript: &mut Transcript254,
) -> io::Result<(FullFilterProof, Vec<ArkGT>, usize)> {
    let size = s.len();
    if !size.is_power_of_two() || size < 2 {
        return Err(invalid("atom filter column length must be a power of two >= 2"));
    }
    for column in [p, o, sel] {
        if column.len() != size {
            return Err(invalid("atom filter column length mismatch"));
        }
    }
    if real_rows > size || output_len > size {
        return Err(invalid("atom filter row count exceeds column length"));
    }
    let num_kept = sources.len();
    if kept_out.len() != num_kept || sources.iter().chain(kept_out).any(|c| c.len() != size) {
        return Err(invalid("atom filter kept column layout mismatch"));
    }
    let has_unit = spec.kept.is_empty();
    let num_vars = size.trailing_zeros() as usize;

    // Round 1: commit the DATA columns (sel + kept outputs + the unit input if a ground
    // atom), bind the spec + all data commitments, then draw the challenges.
    let com_sel = commit_column(sel)?;
    let mut com_kept = Vec::with_capacity(num_kept);
    for column in kept_out {
        com_kept.push(commit_column(column)?);
    }
    let com_unit_in = if has_unit {
        Some(commit_column(&sources[0])?)
    } else {
        None
    };

    bind_spec(transcript, spec);
    transcript.append_group(b"pow/atom/com-s", com_s);
    transcript.append_group(b"pow/atom/com-p", com_p);
    transcript.append_group(b"pow/atom/com-o", com_o);
    transcript.append_group(b"pow/atom/com-sel", &com_sel);
    for c in &com_kept {
        transcript.append_group(b"pow/atom/com-kept", c);
    }
    if let Some(c) = &com_unit_in {
        transcript.append_group(b"pow/atom/com-unit-in", c);
    }
    let beta_c = transcript.challenge_scalar(A_LABEL_BETA_C);
    let beta = transcript.challenge_scalar(A_LABEL_BETA);
    let alpha = transcript.challenge_scalar(A_LABEL_ALPHA);

    // Round 2: derived witnesses (w from beta_c; reciprocals from alpha/beta), bound
    // AFTER the challenges are drawn.
    let w = atom_w(spec, s, p, o, real_rows, beta_c);
    let cstar = reciprocal_witness(alpha, &rlc_fold(beta, &slices(sources)));
    let dstar = reciprocal_witness(alpha, &rlc_fold(beta, &slices(kept_out)));
    let com_w = commit_column(&w)?;
    let com_cstar = commit_column(&cstar)?;
    let com_dstar = commit_column(&dstar)?;
    transcript.append_group(b"pow/atom/com-w", &com_w);
    transcript.append_group(b"pow/atom/com-cstar", &com_cstar);
    transcript.append_group(b"pow/atom/com-dstar", &com_dstar);

    let shape = atom_shape(
        spec, num_vars, real_rows, output_len, num_kept, has_unit, beta_c, alpha, beta,
    );
    let mut columns = vec![s.to_vec(), p.to_vec(), o.to_vec(), sel.to_vec(), w];
    columns.extend(kept_out.iter().cloned());
    columns.push(cstar);
    columns.push(dstar);
    if has_unit {
        columns.push(sources[0].clone());
    }
    let (sumcheck, r, _evals) = prove_sumcheck(&shape, &columns, transcript)?;
    let opening = prove_column_openings(&columns, &r, transcript)?;

    Ok((
        FullFilterProof {
            sumcheck,
            opening,
            sel_commitment: com_sel,
            w_commitment: com_w,
            cstar_commitment: com_cstar,
            dstar_commitment: com_dstar,
        },
        com_kept,
        output_len,
    ))
}

/// Honest selection + compaction for a spec with at least one constant.
#[allow(clippy::too_many_arguments)]
fn prove_compacted(
    spec: &AtomFilterSpec,
    s: &[ArkFr],
    p: &[ArkFr],
    o: &[ArkFr],
    real_rows: usize,
    com_s: &ArkGT,
    com_p: &ArkGT,
    com_o: &ArkGT,
    transcript: &mut Transcript254,
) -> io::Result<(FullFilterProof, Vec<ArkGT>, usize)> {
    let sources = kept_source_columns(spec, s, p, o);
    let sel = atom_sel(spec, s, p, o, real_rows);
    let (kept_out, m) = compact_columns(&sources, &sel);
    prove_compacted_with_witness(
        spec, s, p, o, real_rows, &sel, &sources, &kept_out, m, com_s, com_p, com_o, transcript,
    )
}

/// Prove a generalized per-atom filter: filter the table by the constant positions and
/// compact the KEPT (variable) positions into dense output columns (one per kept
/// position). Returns the proof, one commitment per kept column, and the result count
/// `m`. Cases: |C| = 0 -> the whole real prefix (kept columns ARE the base columns, no
/// argument); |C| = 1/2 -> selection + compaction; |C| = 3 (ground atom) -> a UNIT kept
/// column so C/D/E run uniformly and `m` is the present/absent count (0 or 1).
#[allow(clippy::too_many_arguments)]
pub fn prove_atom_filter(
    spec: &AtomFilterSpec,
    s: &[ArkFr],
    p: &[ArkFr],
    o: &[ArkFr],
    real_rows: usize,
    com_s: &ArkGT,
    com_p: &ArkGT,
    com_o: &ArkGT,
    transcript: &mut Transcript254,
) -> io::Result<(AtomFilterProof, Vec<ArkGT>, usize)> {
    spec.validate()?;
    if spec.constants.is_empty() {
        let com_kept = spec
            .kept
            .iter()
            .map(|pos| base_commitment(*pos, com_s, com_p, com_o))
            .collect();
        return Ok((AtomFilterProof::Identity, com_kept, real_rows));
    }
    let (inner, com_kept, m) =
        prove_compacted(spec, s, p, o, real_rows, com_s, com_p, com_o, transcript)?;
    Ok((AtomFilterProof::Compacted(inner), com_kept, m))
}

#[allow(clippy::too_many_arguments)]
fn verify_compacted(
    spec: &AtomFilterSpec,
    real_rows: usize,
    com_s: &ArkGT,
    com_p: &ArkGT,
    com_o: &ArkGT,
    com_kept: &[ArkGT],
    output_len: usize,
    proof: &FullFilterProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    let num_vars = proof.sumcheck.round_polys.len();
    if num_vars == 0 || num_vars >= usize::BITS as usize {
        return Err(invalid("atom filter proof has an implausible variable count"));
    }
    let size = 1usize << num_vars;
    if real_rows > size || output_len > size {
        return Err(invalid("atom filter row count exceeds column length"));
    }
    let num_kept = spec.kept_sources().len();
    if com_kept.len() != num_kept {
        return Err(invalid("atom filter kept commitment count mismatch"));
    }
    let has_unit = spec.kept.is_empty();
    // The unit INPUT column is public; the verifier recomputes its commitment.
    let com_unit_in = if has_unit {
        Some(commit_column(&vec![f1(); size])?)
    } else {
        None
    };

    bind_spec(transcript, spec);
    transcript.append_group(b"pow/atom/com-s", com_s);
    transcript.append_group(b"pow/atom/com-p", com_p);
    transcript.append_group(b"pow/atom/com-o", com_o);
    transcript.append_group(b"pow/atom/com-sel", &proof.sel_commitment);
    for c in com_kept {
        transcript.append_group(b"pow/atom/com-kept", c);
    }
    if let Some(c) = &com_unit_in {
        transcript.append_group(b"pow/atom/com-unit-in", c);
    }
    let beta_c = transcript.challenge_scalar(A_LABEL_BETA_C);
    let beta = transcript.challenge_scalar(A_LABEL_BETA);
    let alpha = transcript.challenge_scalar(A_LABEL_ALPHA);
    transcript.append_group(b"pow/atom/com-w", &proof.w_commitment);
    transcript.append_group(b"pow/atom/com-cstar", &proof.cstar_commitment);
    transcript.append_group(b"pow/atom/com-dstar", &proof.dstar_commitment);

    let shape = atom_shape(
        spec, num_vars, real_rows, output_len, num_kept, has_unit, beta_c, alpha, beta,
    );
    let (r, evals) = verify_sumcheck(&shape, &proof.sumcheck, transcript)?;

    let mut coms = vec![
        *com_s,
        *com_p,
        *com_o,
        proof.sel_commitment,
        proof.w_commitment,
    ];
    coms.extend_from_slice(com_kept);
    coms.push(proof.cstar_commitment);
    coms.push(proof.dstar_commitment);
    if let Some(c) = com_unit_in {
        coms.push(c);
    }
    verify_column_openings(&coms, &r, &evals, &proof.opening, transcript)
}

/// Verify a generalized per-atom filter proof. The verifier concludes `com_kept[0..m]`
/// is EXACTLY the complete filtered result for `spec` over the first `real_rows` rows.
#[allow(clippy::too_many_arguments)]
pub fn verify_atom_filter(
    spec: &AtomFilterSpec,
    real_rows: usize,
    com_s: &ArkGT,
    com_p: &ArkGT,
    com_o: &ArkGT,
    com_kept: &[ArkGT],
    output_len: usize,
    proof: &AtomFilterProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    spec.validate()?;
    match proof {
        AtomFilterProof::Identity => {
            if !spec.constants.is_empty() {
                return Err(invalid("identity atom proof for a non-empty constant set"));
            }
            let expected: Vec<ArkGT> = spec
                .kept
                .iter()
                .map(|pos| base_commitment(*pos, com_s, com_p, com_o))
                .collect();
            if com_kept != expected.as_slice() || output_len != real_rows {
                return Err(invalid(
                    "identity atom result does not equal the base columns",
                ));
            }
            Ok(())
        }
        AtomFilterProof::Compacted(inner) => {
            if spec.constants.is_empty() {
                return Err(invalid("compacted atom proof for an empty constant set"));
            }
            verify_compacted(
                spec, real_rows, com_s, com_p, com_o, com_kept, output_len, inner, transcript,
            )
        }
    }
}

/// Prove the classic predicate-only Q1 (`P` constant, keep `S`, `O`) as the |C| = 1
/// special case. Returns the proof, `Com(S_out)`, `Com(O_out)`, and `m`. Retained so
/// the dictionary-bound query layer (`proof::query`) is unchanged.
#[allow(clippy::too_many_arguments)]
pub fn prove_filter(
    s: &[ArkFr],
    p: &[ArkFr],
    o: &[ArkFr],
    pred: ArkFr,
    real_rows: usize,
    com_s: &ArkGT,
    com_p: &ArkGT,
    com_o: &ArkGT,
    transcript: &mut Transcript254,
) -> io::Result<(FullFilterProof, ArkGT, ArkGT, usize)> {
    let spec = AtomFilterSpec::predicate(pred);
    let (inner, com_kept, m) =
        prove_compacted(&spec, s, p, o, real_rows, com_s, com_p, com_o, transcript)?;
    Ok((inner, com_kept[0], com_kept[1], m))
}

/// Verify the classic predicate-only Q1 (|C| = 1, keep `S`, `O`).
#[allow(clippy::too_many_arguments)]
pub fn verify_filter(
    com_s: &ArkGT,
    com_p: &ArkGT,
    com_o: &ArkGT,
    pred: ArkFr,
    real_rows: usize,
    com_sout: &ArkGT,
    com_oout: &ArkGT,
    output_len: usize,
    proof: &FullFilterProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    let spec = AtomFilterSpec::predicate(pred);
    verify_compacted(
        &spec,
        real_rows,
        com_s,
        com_p,
        com_o,
        &[*com_sout, *com_oout],
        output_len,
        proof,
        transcript,
    )
}

// ===========================================================================
// Selection-only entry: prove a committed `sel` equals the constant-equality
// indicator, WITHOUT compacting (claims A, B only).
// ===========================================================================
//
// The counterpart to compaction-only: a range atom needs the EQUALITY part of its
// selection (predicate + any subject constant) as a separately committed boolean
// column so it can be multiplied by the (independently proven) range selection. This
// runs ONLY the selection claims A, B — bit-for-bit the full filter's A, B (the same
// generalized `selection_claims` with the PUBLIC `beta_c` constant fold) — over
// `[S, P, O, sel, w]`, pinning `sel[i] = χ_n[i]·Π_c[base_c[i] == const_c]`. It does
// NOT compact, so no kept columns or LogUp reciprocals appear.

const SO_LABEL_DOMAIN: &[u8] = b"pow/selection-only/domain-v1";
const SO_LABEL_BETA_C: &[u8] = b"pow/selection-only/const-beta";

/// A selection-only proof: the batched sumcheck over claims A, B and one batched Dory
/// opening of `[S, P, O, sel, w]` at `r`, plus the `w` commitment the verifier binds.
/// `Com(sel)` is returned separately (the authoritative equality-selection column).
#[derive(Clone, Debug)]
pub struct SelectionOnlyProof {
    pub sumcheck: SumcheckProof,
    pub opening: ColumnOpening,
    pub w_commitment: ArkGT,
}

/// Claims A, B for `spec` over `[S, P, O, sel, w]` (indices S=0,P=1,O=2,sel=3,w=4),
/// with `χ_n` the single chi bound.
fn selection_only_shape(
    spec: &AtomFilterSpec,
    num_vars: usize,
    real_rows: usize,
    beta_c: ArkFr,
) -> SumcheckShape {
    let (claim_a, claim_b) = selection_claims(spec, beta_c);
    SumcheckShape {
        num_vars,
        num_columns: 5,
        chi_bounds: vec![real_rows],
        claims: vec![claim_a, claim_b],
    }
}

/// Prove that `Com(sel)` (returned) commits EXACTLY the constant-equality indicator of
/// `spec` over the first `real_rows` rows: `sel[i] = 1` iff `i < real_rows` and every
/// constant position of `spec` matches, else `0`. Requires at least one constant. Also
/// returns the `sel` column itself so a composing layer can reuse the very column this
/// proof committed (e.g. as a factor of a combined selection).
#[allow(clippy::too_many_arguments)]
pub fn prove_selection_only(
    spec: &AtomFilterSpec,
    s: &[ArkFr],
    p: &[ArkFr],
    o: &[ArkFr],
    real_rows: usize,
    com_s: &ArkGT,
    com_p: &ArkGT,
    com_o: &ArkGT,
    transcript: &mut Transcript254,
) -> io::Result<(SelectionOnlyProof, ArkGT, Vec<ArkFr>)> {
    spec.validate()?;
    if spec.constants.is_empty() {
        return Err(invalid("selection-only requires at least one constant position"));
    }
    let size = s.len();
    if !size.is_power_of_two() || size < 2 {
        return Err(invalid("selection-only column length must be a power of two >= 2"));
    }
    if p.len() != size || o.len() != size {
        return Err(invalid("selection-only column length mismatch"));
    }
    if real_rows > size {
        return Err(invalid("selection-only row count exceeds column length"));
    }
    let num_vars = size.trailing_zeros() as usize;

    let sel = atom_sel(spec, s, p, o, real_rows);
    let com_sel = commit_column(&sel)?;

    bind_spec(transcript, spec);
    transcript.append_bytes(SO_LABEL_DOMAIN, b"parture/proof-of-woql/selection-only/v0");
    transcript.append_group(b"pow/so/com-s", com_s);
    transcript.append_group(b"pow/so/com-p", com_p);
    transcript.append_group(b"pow/so/com-o", com_o);
    transcript.append_group(b"pow/so/com-sel", &com_sel);
    let beta_c = transcript.challenge_scalar(SO_LABEL_BETA_C);

    let w = atom_w(spec, s, p, o, real_rows, beta_c);
    let com_w = commit_column(&w)?;
    transcript.append_group(b"pow/so/com-w", &com_w);

    let shape = selection_only_shape(spec, num_vars, real_rows, beta_c);
    let columns = vec![s.to_vec(), p.to_vec(), o.to_vec(), sel.clone(), w];
    let (sumcheck, r, _evals) = prove_sumcheck(&shape, &columns, transcript)?;
    let opening = prove_column_openings(&columns, &r, transcript)?;

    Ok((
        SelectionOnlyProof {
            sumcheck,
            opening,
            w_commitment: com_w,
        },
        com_sel,
        sel,
    ))
}

/// Verify a selection-only proof: `com_sel` commits EXACTLY the constant-equality
/// indicator of `spec` over the first `real_rows` rows.
#[allow(clippy::too_many_arguments)]
pub fn verify_selection_only(
    spec: &AtomFilterSpec,
    real_rows: usize,
    com_s: &ArkGT,
    com_p: &ArkGT,
    com_o: &ArkGT,
    com_sel: &ArkGT,
    proof: &SelectionOnlyProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    spec.validate()?;
    if spec.constants.is_empty() {
        return Err(invalid("selection-only requires at least one constant position"));
    }
    let num_vars = proof.sumcheck.round_polys.len();
    if num_vars == 0 || num_vars >= usize::BITS as usize {
        return Err(invalid("selection-only proof has an implausible variable count"));
    }
    if real_rows > (1usize << num_vars) {
        return Err(invalid("selection-only row count exceeds column length"));
    }

    bind_spec(transcript, spec);
    transcript.append_bytes(SO_LABEL_DOMAIN, b"parture/proof-of-woql/selection-only/v0");
    transcript.append_group(b"pow/so/com-s", com_s);
    transcript.append_group(b"pow/so/com-p", com_p);
    transcript.append_group(b"pow/so/com-o", com_o);
    transcript.append_group(b"pow/so/com-sel", com_sel);
    let beta_c = transcript.challenge_scalar(SO_LABEL_BETA_C);
    transcript.append_group(b"pow/so/com-w", &proof.w_commitment);

    let shape = selection_only_shape(spec, num_vars, real_rows, beta_c);
    let (r, evals) = verify_sumcheck(&shape, &proof.sumcheck, transcript)?;
    verify_column_openings(
        &[*com_s, *com_p, *com_o, *com_sel, proof.w_commitment],
        &r,
        &evals,
        &proof.opening,
        transcript,
    )
}

// ===========================================================================
// Boolean-product entry: commit `sel = a ⊙ b` and prove the Identity `sel − a·b = 0`.
// ===========================================================================
//
// Combines two committed boolean, χ_n-masked columns into their elementwise product,
// the way a range atom forms `combined_sel = equality_sel · range_sel`. The Identity
// `sel[i] − a[i]·b[i] = 0` at EVERY row pins `sel` to the product (and, when `a`/`b`
// are boolean and zero on padding, makes `sel` boolean and zero on padding too).

const PR_LABEL_DOMAIN: &[u8] = b"pow/bool-product/domain-v1";

/// A boolean-product proof: the Identity sumcheck over `[a, b, sel]` and its opening.
#[derive(Clone, Debug)]
pub struct BooleanProductProof {
    pub sumcheck: SumcheckProof,
    pub opening: ColumnOpening,
}

fn boolean_product_shape(num_vars: usize) -> SumcheckShape {
    // sel − a·b = 0  (a=col0, b=col1, sel=col2).
    let claim = SumcheckClaim {
        shape: ClaimShape::Identity,
        expr: Expr {
            terms: vec![
                Term {
                    coeff: f1(),
                    operands: vec![Operand::Column(2)],
                },
                Term {
                    coeff: f0() - f1(),
                    operands: vec![Operand::Column(0), Operand::Column(1)],
                },
            ],
        },
        target: f0(),
    };
    SumcheckShape {
        num_vars,
        num_columns: 3,
        chi_bounds: Vec::new(),
        claims: vec![claim],
    }
}

/// Commit `sel = a ⊙ b` and prove `sel − a·b = 0`. Returns the proof and `Com(sel)`.
pub fn prove_boolean_product(
    a: &[ArkFr],
    b: &[ArkFr],
    com_a: &ArkGT,
    com_b: &ArkGT,
    transcript: &mut Transcript254,
) -> io::Result<(BooleanProductProof, ArkGT)> {
    let size = a.len();
    if b.len() != size {
        return Err(invalid("boolean-product column length mismatch"));
    }
    if !size.is_power_of_two() || size < 2 {
        return Err(invalid("boolean-product column length must be a power of two >= 2"));
    }
    let num_vars = size.trailing_zeros() as usize;
    let sel: Vec<ArkFr> = a.iter().zip(b.iter()).map(|(x, y)| *x * *y).collect();
    let com_sel = commit_column(&sel)?;

    transcript.append_bytes(PR_LABEL_DOMAIN, b"parture/proof-of-woql/bool-product/v0");
    transcript.append_group(b"pow/prod/com-a", com_a);
    transcript.append_group(b"pow/prod/com-b", com_b);
    transcript.append_group(b"pow/prod/com-sel", &com_sel);

    let shape = boolean_product_shape(num_vars);
    let columns = vec![a.to_vec(), b.to_vec(), sel];
    let (sumcheck, r, _evals) = prove_sumcheck(&shape, &columns, transcript)?;
    let opening = prove_column_openings(&columns, &r, transcript)?;
    Ok((BooleanProductProof { sumcheck, opening }, com_sel))
}

/// Verify a boolean-product proof: `com_sel` commits the elementwise product of the
/// columns committed by `com_a` and `com_b`.
pub fn verify_boolean_product(
    com_a: &ArkGT,
    com_b: &ArkGT,
    com_sel: &ArkGT,
    proof: &BooleanProductProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    let num_vars = proof.sumcheck.round_polys.len();
    if num_vars == 0 || num_vars >= usize::BITS as usize {
        return Err(invalid("boolean-product proof has an implausible variable count"));
    }
    transcript.append_bytes(PR_LABEL_DOMAIN, b"parture/proof-of-woql/bool-product/v0");
    transcript.append_group(b"pow/prod/com-a", com_a);
    transcript.append_group(b"pow/prod/com-b", com_b);
    transcript.append_group(b"pow/prod/com-sel", com_sel);

    let shape = boolean_product_shape(num_vars);
    let (r, evals) = verify_sumcheck(&shape, &proof.sumcheck, transcript)?;
    verify_column_openings(
        &[*com_a, *com_b, *com_sel],
        &r,
        &evals,
        &proof.opening,
        transcript,
    )
}

// ===========================================================================
// Compaction-only entry: compact kept columns by an EXTERNAL boolean selection.
// ===========================================================================
//
// The full atom filter ([`prove_atom_filter`]) fuses the SELECTION claims (A, B,
// which derive `sel` from the constant-equality indicator) with the COMPACTION
// claims (C, D, E, the LogUp dense-compaction) into one sumcheck welded to a single
// internally-derived `sel`. Range filters need to compact by a COMBINED selection
// `sel = equality_sel · range_sel` that is NOT the plain equality indicator, so this
// entry runs ONLY the compaction claims (C, D, E) over an EXTERNAL, already-committed
// boolean `sel` column plus the kept SOURCE columns:
//   * C: `cstar[i] = 1/(alpha − fold_beta(sources[i]))`         (reciprocal, Identity)
//   * D: `dstar[j] = 1/(alpha − fold_beta(kept_out[j]))`         (reciprocal, Identity)
//   * E: `Σ_i sel[i]·cstar[i] − Σ_j χ_m[j]·dstar[j] = 0`         (LogUp link, ZeroSum)
// These are BIT-FOR-BIT the C/D/E claims of the full filter (same fold order, same
// masks), so the audited compaction soundness carries over unchanged.
//
// SECURITY TIE (the caller's responsibility): `sel` MUST be boolean and χ_n-masked.
// This entry does NOT re-prove those properties — they are inherited from whatever
// pins `com_sel` upstream (for a range atom: `sel = equality_sel · range_sel`, a
// product of two independently-proven boolean, χ_n-masked columns, tied to THIS
// `com_sel` by a product Identity). Given that, E proves the multiset
// `{sources[i] : sel[i]=1}` equals `{kept_out[j] : j < m}` AND pins `m` to the
// number of selected rows — exactly the audited dense-compaction guarantee.

const CO_LABEL_DOMAIN: &[u8] = b"pow/compaction-only/domain-v1";
const CO_LABEL_BETA: &[u8] = b"pow/compaction-only/logup-beta";
const CO_LABEL_ALPHA: &[u8] = b"pow/compaction-only/logup-alpha";

/// A compaction-only proof: the batched sumcheck over the two reciprocal claims
/// (C, D) and the LogUp link (E), one batched Dory opening of
/// `[sources.., sel, kept_out.., cstar, dstar]` at `r`, and the two reciprocal
/// commitments the verifier binds. The kept OUTPUT commitments are returned
/// separately (the authoritative dense-result columns).
#[derive(Clone, Debug)]
pub struct CompactionOnlyProof {
    pub sumcheck: SumcheckProof,
    pub opening: ColumnOpening,
    pub cstar_commitment: ArkGT,
    pub dstar_commitment: ArkGT,
}

/// The C/D/E compaction shape over `[sources.., sel, kept_out.., cstar, dstar]`, with
/// `χ_m` (the output length) the single chi bound. Column indices: sources
/// `0..num_kept`, `sel` at `num_kept`, kept outputs `num_kept+1..2·num_kept+1`,
/// `cstar` at `2·num_kept+1`, `dstar` at `2·num_kept+2`.
fn compaction_only_shape(
    num_vars: usize,
    output_len: usize,
    num_kept: usize,
    alpha: ArkFr,
    beta: ArkFr,
) -> SumcheckShape {
    let sel_col = num_kept;
    let kept_out_base = num_kept + 1;
    let cstar = 2 * num_kept + 1;
    let dstar = 2 * num_kept + 2;
    let kept_in: Vec<usize> = (0..num_kept).collect();
    let kept_out: Vec<usize> = (0..num_kept).map(|d| kept_out_base + d).collect();
    let claim_c = reciprocal_claim(cstar, &kept_in, alpha, beta);
    let claim_d = reciprocal_claim(dstar, &kept_out, alpha, beta);
    // chi index 0 is χ_m (the sole chi bound); sel is the (external) input mask.
    let claim_e = multiset_link_claim(Operand::Column(sel_col), cstar, Operand::Chi(0), dstar);
    SumcheckShape {
        num_vars,
        num_columns: 2 * num_kept + 3,
        chi_bounds: vec![output_len],
        claims: vec![claim_c, claim_d, claim_e],
    }
}

/// Bind the compaction-only statement: the domain tag, the kept arity, and the source
/// / selection / kept-output commitments. Identical on both sides.
fn bind_compaction_only(
    transcript: &mut Transcript254,
    source_commitments: &[ArkGT],
    com_sel: &ArkGT,
    com_kept: &[ArkGT],
) {
    transcript.append_bytes(CO_LABEL_DOMAIN, b"parture/proof-of-woql/compaction-only/v0");
    transcript.append_bytes(CO_LABEL_DOMAIN, &(source_commitments.len() as u64).to_le_bytes());
    for c in source_commitments {
        transcript.append_group(b"pow/co/com-src", c);
    }
    transcript.append_group(b"pow/co/com-sel", com_sel);
    for c in com_kept {
        transcript.append_group(b"pow/co/com-kept", c);
    }
}

/// Prove that the dense columns `kept_out[0..m]` are EXACTLY the `sel`-compaction of
/// `sources` (one output column per source, dense prefix + zero tail), for an EXTERNAL
/// boolean, χ_n-masked `sel` committed as `com_sel`. `source_commitments` must commit
/// `sources` (root-anchored by the caller). Returns the proof, one commitment per kept
/// output column, the dense kept output columns themselves (full padded length, for a
/// composing layer's witness), and the match count `m`.
pub fn prove_compaction_only(
    sources: &[Vec<ArkFr>],
    source_commitments: &[ArkGT],
    sel: &[ArkFr],
    com_sel: &ArkGT,
    transcript: &mut Transcript254,
) -> io::Result<(CompactionOnlyProof, Vec<ArkGT>, Vec<Vec<ArkFr>>, usize)> {
    let num_kept = sources.len();
    if num_kept == 0 {
        return Err(invalid("compaction-only needs at least one kept column"));
    }
    if source_commitments.len() != num_kept {
        return Err(invalid("compaction-only source/commitment count mismatch"));
    }
    let size = sel.len();
    if !size.is_power_of_two() || size < 2 {
        return Err(invalid("compaction-only column length must be a power of two >= 2"));
    }
    if sources.iter().any(|c| c.len() != size) {
        return Err(invalid("compaction-only source column length mismatch"));
    }
    let num_vars = size.trailing_zeros() as usize;

    // Dense compaction of the sources by the external sel.
    let (kept_out, m) = compact_columns(sources, sel);
    let mut com_kept = Vec::with_capacity(num_kept);
    for col in &kept_out {
        com_kept.push(commit_column(col)?);
    }

    // Round 1: bind the data commitments, then draw the LogUp challenges.
    bind_compaction_only(transcript, source_commitments, com_sel, &com_kept);
    let beta = transcript.challenge_scalar(CO_LABEL_BETA);
    let alpha = transcript.challenge_scalar(CO_LABEL_ALPHA);

    // Round 2: derived reciprocals, bound AFTER the challenges.
    let cstar = reciprocal_witness(alpha, &rlc_fold(beta, &slices(sources)));
    let dstar = reciprocal_witness(alpha, &rlc_fold(beta, &slices(&kept_out)));
    let com_cstar = commit_column(&cstar)?;
    let com_dstar = commit_column(&dstar)?;
    transcript.append_group(b"pow/co/com-cstar", &com_cstar);
    transcript.append_group(b"pow/co/com-dstar", &com_dstar);

    let shape = compaction_only_shape(num_vars, m, num_kept, alpha, beta);
    let mut columns: Vec<Vec<ArkFr>> = sources.to_vec();
    columns.push(sel.to_vec());
    columns.extend(kept_out.iter().cloned());
    columns.push(cstar);
    columns.push(dstar);
    let (sumcheck, r, _evals) = prove_sumcheck(&shape, &columns, transcript)?;
    let opening = prove_column_openings(&columns, &r, transcript)?;

    Ok((
        CompactionOnlyProof {
            sumcheck,
            opening,
            cstar_commitment: com_cstar,
            dstar_commitment: com_dstar,
        },
        com_kept,
        kept_out,
        m,
    ))
}

/// Verify a compaction-only proof: `com_kept[0..output_len]` is EXACTLY the
/// `sel`-compaction of the columns committed by `source_commitments`, for the external
/// boolean selection committed as `com_sel`. The caller is responsible for the
/// upstream binding that makes `com_sel` a genuine boolean, χ_n-masked selection.
pub fn verify_compaction_only(
    source_commitments: &[ArkGT],
    com_sel: &ArkGT,
    com_kept: &[ArkGT],
    output_len: usize,
    proof: &CompactionOnlyProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    let num_kept = source_commitments.len();
    if num_kept == 0 {
        return Err(invalid("compaction-only needs at least one kept column"));
    }
    if com_kept.len() != num_kept {
        return Err(invalid("compaction-only kept commitment count mismatch"));
    }
    let num_vars = proof.sumcheck.round_polys.len();
    if num_vars == 0 || num_vars >= usize::BITS as usize {
        return Err(invalid("compaction-only proof has an implausible variable count"));
    }
    let size = 1usize << num_vars;
    if output_len > size {
        return Err(invalid("compaction-only output length exceeds column length"));
    }

    bind_compaction_only(transcript, source_commitments, com_sel, com_kept);
    let beta = transcript.challenge_scalar(CO_LABEL_BETA);
    let alpha = transcript.challenge_scalar(CO_LABEL_ALPHA);
    transcript.append_group(b"pow/co/com-cstar", &proof.cstar_commitment);
    transcript.append_group(b"pow/co/com-dstar", &proof.dstar_commitment);

    let shape = compaction_only_shape(num_vars, output_len, num_kept, alpha, beta);
    let (r, evals) = verify_sumcheck(&shape, &proof.sumcheck, transcript)?;

    let mut coms: Vec<ArkGT> = source_commitments.to_vec();
    coms.push(*com_sel);
    coms.extend_from_slice(com_kept);
    coms.push(proof.cstar_commitment);
    coms.push(proof.dstar_commitment);
    verify_column_openings(&coms, &r, &evals, &proof.opening, transcript)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fr(v: u64) -> ArkFr {
        ArkFr::from_u64(v)
    }

    fn transcript() -> Transcript254 {
        Transcript254::new(b"parture-filter-test")
    }

    // Build a padded predicate column of length 16 from `n` real ids (the rest
    // padded with `pad`).
    fn predicate_column(real_ids: &[u64], pad: u64) -> Vec<ArkFr> {
        assert!(real_ids.len() <= 16);
        (0..16usize)
            .map(|i| fr(if i < real_ids.len() { real_ids[i] } else { pad }))
            .collect()
    }

    fn commit(column: &[ArkFr]) -> ArkGT {
        commit_column(column).unwrap()
    }

    fn run(
        p_col: &[ArkFr],
        pred: ArkFr,
        n: usize,
        sel: &[ArkFr],
        w: &[ArkFr],
    ) -> io::Result<ArkFr> {
        let com_p = commit(p_col);
        let mut pt = transcript();
        let (proof, com_sel, com_w, _sel_eval) =
            prove_filter_with_witness(p_col, pred, n, sel, w, &com_p, &mut pt)?;
        let mut vt = transcript();
        verify_filter_indicator(&com_p, pred, n, &com_sel, &com_w, &proof, &mut vt)
    }

    #[test]
    fn compaction_only_matches_independent_result_and_rejects_tampering() {
        // Two "kept" source columns (S/O style) over a 2^3 domain and an EXTERNAL
        // boolean sel selecting rows 1, 2, 5 (an arbitrary boolean column, NOT a
        // predicate indicator — exactly the combined-sel case the range layer needs).
        let size = 8usize;
        let s: Vec<ArkFr> = [10u64, 11, 12, 13, 14, 15, 16, 17].iter().map(|v| fr(*v)).collect();
        let o: Vec<ArkFr> = [20u64, 21, 22, 23, 24, 25, 26, 27].iter().map(|v| fr(*v)).collect();
        let mut sel = vec![f0(); size];
        for i in [1usize, 2, 5] {
            sel[i] = f1();
        }
        let sources = vec![s.clone(), o.clone()];
        let com_sources = vec![commit(&s), commit(&o)];
        let com_sel = commit(&sel);

        let mut pt = transcript();
        let (proof, com_kept, kept_cols, m) =
            prove_compaction_only(&sources, &com_sources, &sel, &com_sel, &mut pt).unwrap();
        let mut vt = transcript();
        verify_compaction_only(&com_sources, &com_sel, &com_kept, m, &proof, &mut vt).unwrap();

        // m and the kept output commitments equal an INDEPENDENT dense compaction.
        assert_eq!(m, 3);
        let (exp, exp_m) = compact_columns(&sources, &sel);
        assert_eq!(exp_m, m);
        assert_eq!(com_kept.len(), 2);
        assert_eq!(com_kept[0], commit(&exp[0]));
        assert_eq!(com_kept[1], commit(&exp[1]));
        // The returned dense kept columns match the independent compaction too.
        assert_eq!(kept_cols, exp);

        // Tampered kept commitments (columns swapped) are rejected.
        let mut vt2 = transcript();
        assert!(verify_compaction_only(
            &com_sources,
            &com_sel,
            &[com_kept[1], com_kept[0]],
            m,
            &proof,
            &mut vt2
        )
        .is_err());

        // A wrong output length is rejected.
        let mut vt3 = transcript();
        assert!(verify_compaction_only(&com_sources, &com_sel, &com_kept, m + 1, &proof, &mut vt3)
            .is_err());

        // A different sel commitment than the proof was built over is rejected.
        let mut other_sel = vec![f0(); size];
        other_sel[0] = f1();
        let com_other_sel = commit(&other_sel);
        let mut vt4 = transcript();
        assert!(verify_compaction_only(&com_sources, &com_other_sel, &com_kept, m, &proof, &mut vt4)
            .is_err());
    }

    #[test]
    fn correct_selection_verifies() {
        // ids: 5,1,5,2,1,9,1,5 real (n=8); const=1 matches rows 1,4,6.
        let p = predicate_column(&[5, 1, 5, 2, 1, 9, 1, 5], 0);
        let n = 8;
        let pred = fr(1);
        let (sel, w) = honest_witness(&p, pred, n);
        // sanity: sel is exactly the in-range matches.
        for i in 0..16 {
            let expected = if i < n && p[i] == pred { f1() } else { f0() };
            assert_eq!(sel[i], expected);
        }
        assert!(run(&p, pred, n, &sel, &w).is_ok());
    }

    #[test]
    fn omission_of_a_true_match_is_rejected() {
        let p = predicate_column(&[5, 1, 5, 2, 1, 9, 1, 5], 0);
        let n = 8;
        let pred = fr(1);
        let (mut sel, w) = honest_witness(&p, pred, n);
        // Flip a genuine match (row 4) from 1 -> 0.
        assert_eq!(sel[4], f1());
        sel[4] = f0();
        assert!(run(&p, pred, n, &sel, &w).is_err());
    }

    #[test]
    fn false_positive_on_a_nonmatch_is_rejected() {
        let p = predicate_column(&[5, 1, 5, 2, 1, 9, 1, 5], 0);
        let n = 8;
        let pred = fr(1);
        let (mut sel, w) = honest_witness(&p, pred, n);
        // Select a non-matching in-range row (row 0, P=5 != 1).
        assert_eq!(sel[0], f0());
        sel[0] = f1();
        assert!(run(&p, pred, n, &sel, &w).is_err());
    }

    #[test]
    fn padding_collision_is_masked_and_a_selected_padding_row_is_rejected() {
        // Row 10 is PADDING (n=8) but its P value equals the queried constant.
        let mut ids = vec![5u64, 1, 5, 2, 1, 9, 1, 5];
        let p = {
            let mut col = predicate_column(&ids, 0);
            col[10] = fr(1); // padding row collides with pred
            col
        };
        let _ = &mut ids;
        let n = 8;
        let pred = fr(1);
        let (mut sel, w) = honest_witness(&p, pred, n);
        // Honest sel MUST be 0 on the colliding padding row (chi_n masks it).
        assert_eq!(sel[10], f0());
        assert!(run(&p, pred, n, &sel, &w).is_ok());
        // A prover that selects the padding row is rejected.
        sel[10] = f1();
        assert!(run(&p, pred, n, &sel, &w).is_err());
    }

    #[test]
    fn wrong_pseudo_inverse_on_a_nonmatch_row_is_rejected() {
        // w is pinned on NON-matching rows (lhs != 0): B forces w = chi_n/lhs.
        // Corrupt w on an in-range non-match row (row 0, P=5): honest w = lhs^{-1},
        // set it to 0 -> B: 1 - 0 - lhs*0 = 1 != 0 -> reject.
        // (On MATCHING rows lhs = 0, so w there is unconstrained and a nonzero
        // value would NOT be a soundness violation.)
        let p = predicate_column(&[5, 1, 5, 2, 1, 9, 1, 5], 0);
        let n = 8;
        let pred = fr(1);
        let (sel, mut w) = honest_witness(&p, pred, n);
        assert_ne!(w[0], f0()); // row 0 is a non-match, so w[0] = (5-1)^{-1}
        w[0] = f0();
        assert!(run(&p, pred, n, &sel, &w).is_err());
    }

    #[test]
    fn nonzero_w_on_a_matching_row_is_harmless() {
        // Documents the gadget's exact guarantee: w is unconstrained on matching
        // rows, so a nonzero value there is NOT a violation and still verifies.
        let p = predicate_column(&[5, 1, 5, 2, 1, 9, 1, 5], 0);
        let n = 8;
        let pred = fr(1);
        let (sel, mut w) = honest_witness(&p, pred, n);
        assert_eq!(sel[1], f1()); // row 1 matches, honest w[1] = 0
        assert_eq!(w[1], f0());
        w[1] = fr(12345);
        assert!(run(&p, pred, n, &sel, &w).is_ok());
    }

    #[test]
    fn empty_result_and_all_match_verify() {
        // Empty: pred absent from the real rows.
        let p = predicate_column(&[5, 1, 5, 2, 1, 9, 1, 5], 0);
        let n = 8;
        let pred = fr(7); // no real row equals 7
        let (sel, w) = honest_witness(&p, pred, n);
        assert!(sel.iter().all(|s| *s == f0()));
        assert!(run(&p, pred, n, &sel, &w).is_ok());

        // All-match: every real row equals pred.
        let p_all = predicate_column(&[3, 3, 3, 3, 3, 3, 3, 3], 0);
        let n2 = 8;
        let pred2 = fr(3);
        let (sel2, w2) = honest_witness(&p_all, pred2, n2);
        for i in 0..16 {
            assert_eq!(sel2[i], if i < n2 { f1() } else { f0() });
        }
        assert!(run(&p_all, pred2, n2, &sel2, &w2).is_ok());
    }

    // ---- Full Q1 (filter + LogUp compaction) ----

    // S, P, O over 8 real rows (padded to 16). pred=1 matches rows 0,2,4,6.
    fn q1_tables() -> (Vec<ArkFr>, Vec<ArkFr>, Vec<ArkFr>, usize) {
        let s = predicate_column(&[10, 20, 30, 40, 50, 60, 70, 80], 0);
        let p = predicate_column(&[1, 2, 1, 3, 1, 2, 1, 9], 0);
        let o = predicate_column(&[100, 200, 300, 400, 500, 600, 700, 800], 0);
        (s, p, o, 8)
    }

    fn q1_run_honest(
        s: &[ArkFr],
        p: &[ArkFr],
        o: &[ArkFr],
        pred: ArkFr,
        n: usize,
    ) -> io::Result<usize> {
        let (com_s, com_p, com_o) = (commit(s), commit(p), commit(o));
        let mut pt = transcript();
        let (proof, com_sout, com_oout, m) =
            prove_filter(s, p, o, pred, n, &com_s, &com_p, &com_o, &mut pt)?;
        let mut vt = transcript();
        verify_filter(
            &com_s, &com_p, &com_o, pred, n, &com_sout, &com_oout, m, &proof, &mut vt,
        )?;
        Ok(m)
    }

    // ---- Generalized per-atom filter (any subset of {S,P,O} constant) ----

    fn col16(real: &[u64]) -> Vec<ArkFr> {
        (0..16usize)
            .map(|i| fr(if i < real.len() { real[i] } else { 0 }))
            .collect()
    }

    fn mkspec(constants: &[(Position, u64)], kept: &[Position]) -> AtomFilterSpec {
        AtomFilterSpec {
            constants: constants.iter().map(|(p, c)| (*p, fr(*c))).collect(),
            kept: kept.to_vec(),
        }
    }

    // 8 real rows (padded to 16). Rows: (S,P,O).
    fn atom_table() -> (Vec<ArkFr>, Vec<ArkFr>, Vec<ArkFr>, usize) {
        let s = col16(&[1, 2, 3, 1, 5, 6, 1, 5]);
        let p = col16(&[10, 10, 20, 10, 30, 10, 20, 30]);
        let o = col16(&[100, 200, 300, 400, 500, 600, 700, 800]);
        (s, p, o, 8)
    }

    // Honest prove+verify of the atom filter. Returns (m, kept commitments).
    fn atom_honest(
        spec: &AtomFilterSpec,
        s: &[ArkFr],
        p: &[ArkFr],
        o: &[ArkFr],
        n: usize,
    ) -> io::Result<(usize, Vec<ArkGT>)> {
        let (com_s, com_p, com_o) = (commit(s), commit(p), commit(o));
        let mut pt = transcript();
        let (proof, com_kept, m) =
            prove_atom_filter(spec, s, p, o, n, &com_s, &com_p, &com_o, &mut pt)?;
        let mut vt = transcript();
        verify_atom_filter(spec, n, &com_s, &com_p, &com_o, &com_kept, m, &proof, &mut vt)?;
        Ok((m, com_kept))
    }

    // Prove+verify a compacted atom filter with an EXPLICIT sel + kept_out (malicious
    // injection). Requires |C| >= 1.
    #[allow(clippy::too_many_arguments)]
    fn atom_with_witness(
        spec: &AtomFilterSpec,
        s: &[ArkFr],
        p: &[ArkFr],
        o: &[ArkFr],
        n: usize,
        sel: &[ArkFr],
        kept_out: &[Vec<ArkFr>],
        m: usize,
    ) -> io::Result<()> {
        let (com_s, com_p, com_o) = (commit(s), commit(p), commit(o));
        let sources = kept_source_columns(spec, s, p, o);
        let mut pt = transcript();
        let (proof, com_kept, _) = prove_compacted_with_witness(
            spec, s, p, o, n, sel, &sources, kept_out, m, &com_s, &com_p, &com_o, &mut pt,
        )?;
        let mut vt = transcript();
        verify_compacted(spec, n, &com_s, &com_p, &com_o, &com_kept, m, &proof, &mut vt)
    }

    // Honest sel + compaction for a spec (helper for the injection tests).
    fn atom_honest_witness(
        spec: &AtomFilterSpec,
        s: &[ArkFr],
        p: &[ArkFr],
        o: &[ArkFr],
        n: usize,
    ) -> (Vec<ArkFr>, Vec<Vec<ArkFr>>, usize) {
        let sel = atom_sel(spec, s, p, o, n);
        let sources = kept_source_columns(spec, s, p, o);
        let (kept_out, m) = compact_columns(&sources, &sel);
        (sel, kept_out, m)
    }

    #[test]
    fn q1_known_table_dense_output_matches_and_verifies() {
        let (s, p, o, n) = q1_tables();
        let pred = fr(1);
        let (sel, _) = honest_witness(&p, pred, n);
        let (s_out, o_out, m) = compact_selected(&s, &o, &sel);
        assert_eq!(m, 4);
        assert_eq!(&s_out[..4], &[fr(10), fr(30), fr(50), fr(70)]);
        assert_eq!(&o_out[..4], &[fr(100), fr(300), fr(500), fr(700)]);
        assert_eq!(q1_run_honest(&s, &p, &o, pred, n).unwrap(), 4);
    }

    #[test]
    fn atom_filter_honest_all_arities() {
        let (s, p, o, n) = atom_table();
        // |C| = 0: t(?s,?p,?o) -> all real rows; kept columns ARE the base columns.
        let (m0, kept0) = atom_honest(&mkspec(&[], &[Position::S, Position::P, Position::O]), &s, &p, &o, n).unwrap();
        assert_eq!(m0, n);
        assert_eq!(kept0, vec![commit(&s), commit(&p), commit(&o)]);
        // |C| = 1, P constant (the classic Q1): P=10 -> rows 0,1,3,5.
        let (m1, _) = atom_honest(&mkspec(&[(Position::P, 10)], &[Position::S, Position::O]), &s, &p, &o, n).unwrap();
        assert_eq!(m1, 4);
        // |C| = 1, S constant: S=1 -> rows 0,3,6.
        let (m1b, _) = atom_honest(&mkspec(&[(Position::S, 1)], &[Position::P, Position::O]), &s, &p, &o, n).unwrap();
        assert_eq!(m1b, 3);
        // |C| = 2: S=1 AND P=10 -> rows 0,3; kept O = [100, 400].
        let (m2, kept2) = atom_honest(&mkspec(&[(Position::S, 1), (Position::P, 10)], &[Position::O]), &s, &p, &o, n).unwrap();
        assert_eq!(m2, 2);
        assert_eq!(kept2, vec![commit(&col16(&[100, 400]))]);
        // |C| = 3 (ground atom) present: (1,10,100) is row 0 -> m = 1.
        let (m3, _) = atom_honest(&mkspec(&[(Position::S, 1), (Position::P, 10), (Position::O, 100)], &[]), &s, &p, &o, n).unwrap();
        assert_eq!(m3, 1);
        // |C| = 3 absent: (1,10,999) is not present -> m = 0.
        let (m3a, _) = atom_honest(&mkspec(&[(Position::S, 1), (Position::P, 10), (Position::O, 999)], &[]), &s, &p, &o, n).unwrap();
        assert_eq!(m3a, 0);
    }

    #[test]
    fn atom_filter_omission_is_rejected() {
        let (s, p, o, n) = atom_table();
        for spec in [
            mkspec(&[(Position::P, 10)], &[Position::S, Position::O]),
            mkspec(&[(Position::S, 1), (Position::P, 10)], &[Position::O]),
        ] {
            let (mut sel, _, _) = atom_honest_witness(&spec, &s, &p, &o, n);
            // Drop the first genuine match (flip a 1 -> 0).
            let hit = (0..n).find(|&i| sel[i] == f1()).unwrap();
            sel[hit] = f0();
            let sources = kept_source_columns(&spec, &s, &p, &o);
            let (kept_out, m) = compact_columns(&sources, &sel);
            assert!(atom_with_witness(&spec, &s, &p, &o, n, &sel, &kept_out, m).is_err());
        }
    }

    #[test]
    fn atom_filter_false_positive_is_rejected() {
        let (s, p, o, n) = atom_table();
        for spec in [
            mkspec(&[(Position::P, 10)], &[Position::S, Position::O]),
            mkspec(&[(Position::S, 1), (Position::P, 10)], &[Position::O]),
        ] {
            let (mut sel, _, _) = atom_honest_witness(&spec, &s, &p, &o, n);
            // Select an in-range NON-matching row.
            let miss = (0..n).find(|&i| sel[i] == f0()).unwrap();
            sel[miss] = f1();
            let sources = kept_source_columns(&spec, &s, &p, &o);
            let (kept_out, m) = compact_columns(&sources, &sel);
            assert!(atom_with_witness(&spec, &s, &p, &o, n, &sel, &kept_out, m).is_err());
        }
    }

    #[test]
    fn atom_filter_padding_collision_is_masked_and_selected_padding_rejects() {
        let (s, mut p, o, n) = atom_table();
        p[10] = fr(10); // a PADDING row (>= n) collides with the queried predicate.
        let spec = mkspec(&[(Position::P, 10)], &[Position::S, Position::O]);
        let (mut sel, kept_out, m) = atom_honest_witness(&spec, &s, &p, &o, n);
        assert_eq!(sel[10], f0()); // chi_n masks it out of the honest selection.
        assert!(atom_with_witness(&spec, &s, &p, &o, n, &sel, &kept_out, m).is_ok());
        // A prover that selects the colliding padding row is rejected.
        sel[10] = f1();
        let (kept2, m2) = compact_columns(&kept_source_columns(&spec, &s, &p, &o), &sel);
        assert!(atom_with_witness(&spec, &s, &p, &o, n, &sel, &kept2, m2).is_err());
    }

    #[test]
    fn atom_filter_reordered_output_verifies_but_duplicate_is_rejected() {
        let (s, p, o, n) = atom_table();
        let spec = mkspec(&[(Position::P, 10)], &[Position::S, Position::O]);
        let (sel, kept_out, m) = atom_honest_witness(&spec, &s, &p, &o, n);
        // Reordering the true output rows is a valid multiset -> verifies.
        let mut reordered = kept_out.clone();
        for col in reordered.iter_mut() {
            col[..m].reverse();
        }
        assert!(atom_with_witness(&spec, &s, &p, &o, n, &sel, &reordered, m).is_ok());
        // Duplicating a row (wrong multiplicity) -> rejected.
        let mut dup = kept_out.clone();
        for col in dup.iter_mut() {
            col[m - 1] = col[0];
        }
        assert!(atom_with_witness(&spec, &s, &p, &o, n, &sel, &dup, m).is_err());
    }

    #[test]
    fn atom_filter_wrong_m_is_rejected() {
        let (s, p, o, n) = atom_table();
        let spec = mkspec(&[(Position::P, 10)], &[Position::S, Position::O]);
        let (sel, kept_out, m) = atom_honest_witness(&spec, &s, &p, &o, n); // m = 4
        assert!(atom_with_witness(&spec, &s, &p, &o, n, &sel, &kept_out, m + 1).is_err());
        assert!(atom_with_witness(&spec, &s, &p, &o, n, &sel, &kept_out, m - 1).is_err());
    }

    #[test]
    fn atom_filter_rlc_cancellation_is_rejected() {
        // |C| = 2 (S=1, P=10). A malicious in-range row is built so that
        // (S - 1) + beta_guess*(P - 10) = 0 for a GUESSED beta_guess = 2: pick P = 5,
        // then S = 11 makes 10 + 2*(-5) = 0. Marking it selected passes only if the
        // real Fiat-Shamir beta_c equals the guess; since beta_c is drawn AFTER S,P are
        // committed, beta_c != 2 -> lhs != 0 -> claim A rejects.
        let s = col16(&[1, 11, 3, 1]);
        let p = col16(&[10, 5, 20, 10]);
        let o = col16(&[100, 200, 300, 400]);
        let n = 4;
        let spec = mkspec(&[(Position::S, 1), (Position::P, 10)], &[Position::O]);
        let mut sel = atom_sel(&spec, &s, &p, &o, n);
        assert_eq!(sel[1], f0()); // row 1 is (S=11, P=5): not a genuine match.
        sel[1] = f1();
        let (kept_out, m) = compact_columns(&kept_source_columns(&spec, &s, &p, &o), &sel);
        assert!(atom_with_witness(&spec, &s, &p, &o, n, &sel, &kept_out, m).is_err());
    }

    #[test]
    fn q1_empty_result_and_all_match_verify() {
        let (s, p, o, n) = q1_tables();
        // Empty: pred absent from the real rows -> m = 0.
        assert_eq!(q1_run_honest(&s, &p, &o, fr(42), n).unwrap(), 0);

        // All-match: every real row matches -> m = n.
        let p_all = predicate_column(&[7, 7, 7, 7, 7, 7, 7, 7], 0);
        assert_eq!(q1_run_honest(&s, &p_all, &o, fr(7), n).unwrap(), n);
    }

    #[test]
    fn q1_end_to_end_over_several_predicates() {
        let (s, p, o, n) = q1_tables();
        for (pred_id, expected_m) in [(1u64, 4usize), (2, 2), (3, 1), (9, 1), (42, 0)] {
            let m = q1_run_honest(&s, &p, &o, fr(pred_id), n).unwrap();
            assert_eq!(m, expected_m, "predicate {pred_id}");
        }
    }
}
