//! Tuple (edge-pair) DISEQUALITY selection over an already-authenticated relation —
//! the store gadget that makes finite REPEATED-PREDICATE property paths (`p^n`,
//! `Times{n,n}`, `Times{a,b}`) provably exact under the leading-Prolog EDGE-SIMPLE
//! path semantics (`src/core/query/path.pl`).
//!
//! # Why this gadget exists (the exact over-count)
//!
//! A finite single-predicate path `x0 -p-> x1 -p-> ... -p-> xn` is lowered to a
//! conjunctive bag-join over `n` triples (`Triple(x_{i}, p, x_{i+1})`). Plain
//! bag-join reproduces one row per distinct intermediate assignment — one per
//! distinct *walk*. But Prolog's `run_pattern_` keeps an `Open_Set` and FAILS a hop
//! whose resolved edge `(subject, predicate, object)` already appears in the set:
//! paths are edge-SIMPLE (no stored triple traversed twice within one solution).
//! NODES may repeat (`a->b->a` over two DISTINCT edges is a VALID solution and MUST
//! be kept); only an EDGE may not repeat. With a FIXED predicate `p`, two hops reuse
//! the same stored edge iff their `(subject, object)` pairs coincide. So the bag-join
//! over-counts EXACTLY the rows in which some hop-pair reuses an edge; dropping those
//! rows (and only those) recovers the edge-simple bag.
//!
//! # The gadget (reuse — NO new sumcheck)
//!
//! For each unordered hop-pair `(e, f)` the caller supplies the two edges' node
//! references `edge_e = (a0, a1)`, `edge_f = (b0, b1)` (each a committed column index
//! or a public constant node id). We form a multi-column random-linear-combination
//! (RLC) fingerprint difference over the ALREADY-committed relation columns:
//!
//! ```text
//!   Δ_i = (a0_i - b0_i) + γ·(a1_i - b1_i)
//! ```
//!
//! where `γ` is a Fiat–Shamir challenge drawn AFTER the relation's column commitments
//! are bound to the transcript (Schwartz–Zippel: `Δ_i == 0` iff `a0_i == b0_i` AND
//! `a1_i == b1_i`, except with probability ≤ 1/|F|). We then REUSE the audited IsZero
//! indicator ([`super::filter::prove_filter_indicator`]) to pin, for a predicate
//! column `P_i = Σ_c coeff_c·col_c[i]` and public constant `pred = -offset`,
//!
//! ```text
//!   sel_i = [ Δ_i == 0  AND  i < n ]      (edge reused, on a real row)
//! ```
//!
//! and KEEP the COMPLEMENT via the audited compaction argument
//! ([`super::filter::prove_compaction_only`]): the selection handed to the compaction
//! is `keep = χ_n − sel` (keep-where-Δ-nonzero), committed as the HOMOMORPHIC
//! difference `com_keep = com(χ_n) − com(sel)`. Because the IsZero gadget forces
//! `sel = 0` on padding (`i ≥ n`, the χ_n mask of claim B), `keep = χ_n − sel` is
//! boolean and χ_n-masked, so padding never survives (even a padding row whose Δ
//! collides to 0). One such filter runs per hop-pair, sequentially — each stage
//! carries ALL relation columns through (only rows shrink) so later pairs reference
//! the same column indices; the endpoint projection is applied by the caller on top.
//!
//! # What is (deliberately) NOT here
//!
//! * NO new polynomial argument: `Δ` is a linear fold of committed columns, the
//!   indicator is the exact `filter.rs` IsZero (claims A/B), the keep-complement is a
//!   homomorphic commitment difference, and the compaction is the exact
//!   `filter.rs` LogUp `prove_compaction_only`.
//! * Predicate-ARITY per pair. A [`DisequalityPair`] with `predicate == None` is the 2-column
//!   `(subject, object)` fold: the predicate id is identical across the hops sharing it and cancels
//!   out of the fingerprint, so the COMPILER partitions a CONCRETE mixed-predicate repeat by
//!   predicate and emits this 2-column fold ONLY for same-predicate hop-pairs (different-predicate
//!   hops are always distinct stored edges). A pair with `predicate == Some((pa, pb))` folds the
//!   predicate id column in with a γ² coefficient — the FULL-TRIPLE `(subject, predicate, object)`
//!   fold — for hops with a FREE (wildcard `.`) predicate, which cannot be statically partitioned.
//!   The 3-column fold GENERALIZES the 2-column one (equal concrete predicate ⇒ γ² term vanishes;
//!   differing concrete predicate ⇒ γ² term forces `Δ != 0`), so both reduce to the same
//!   IsZero+compaction machinery with only the fold arity changing.
//! * Its OWN transcript domain tag ([`TDQ_DOMAIN`]) so a disequality proof can never
//!   be replayed as an equality (or any other) selection.

use std::collections::BTreeMap;
use std::io;

use ark_serialize::CanonicalSerialize;
use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::{Field, Group};
use dory_pcs::primitives::transcript::Transcript;

use super::argument::{commit_column, StoreProverContext, Transcript254};
use super::bgp::{RelationCommit, RelationWitness};
use super::filter::{
    honest_witness, prove_compaction_only_with_context, prove_filter_indicator_with_context,
    verify_compaction_only, verify_filter_indicator, CompactionOnlyProof, FilterProof,
};
use super::invalid;

/// Domain separation tag bound at the head of every tuple-disequality transcript —
/// distinct from the equality gadget's tag so the two selections can never collide.
const TDQ_DOMAIN: &[u8] = b"parture/proof-of-woql/tuple-disequality/v0";

/// One node position in an edge fingerprint: a committed input column, or a public
/// constant node id (an endpoint IRI resolved through the trusted layer dictionary).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisequalityNode {
    Column(usize),
    Constant(ArkFr),
}

/// One hop-pair disequality: drop the rows where stored edge `A` equals stored edge `B`. `a0`/`b0`
/// are the two hops' SUBJECT positions and `a1`/`b1` their OBJECT positions (both in the walk
/// frame — see the compiler's `build_disequality_pairs`).
///
/// `predicate` selects the fingerprint ARITY:
///   * `None` (the 2-column `(subject, object)` fold) — two hops share a stored edge iff their
///     subject AND object coincide. Used for CONCRETE same-predicate finite repeats (P5a), where
///     the predicate is a compile-time constant identical across the hops and cancels out of the
///     fingerprint, so folding it in is unnecessary.
///   * `Some((pa, pb))` (the 3-column `(subject, predicate, object)` FULL-TRIPLE fold) — two hops
///     share a stored edge iff their subject AND predicate AND object coincide. Used whenever a hop
///     has a FREE (wildcard `.`) predicate: two wildcard hops cannot be partitioned by a
///     compile-time predicate, so the predicate id column (`pa`/`pb`) must be folded into the
///     fingerprint with a γ² coefficient. This GENERALIZES the 2-column fold: for two concrete
///     same-predicate hops `pa == pb` is an equal constant (γ² term vanishes) and for concrete
///     different-predicate hops `pa != pb` forces `Δ != 0` unconditionally (always distinct), so a
///     3-column pair reproduces P5a's per-predicate partition exactly. Faithful to `path.pl`'s
///     edge identity `make_edge(subject, predicate, object)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DisequalityPair {
    pub a0: DisequalityNode,
    pub a1: DisequalityNode,
    pub b0: DisequalityNode,
    pub b1: DisequalityNode,
    pub predicate: Option<(DisequalityNode, DisequalityNode)>,
}

/// One sequential filter stage's proof (one hop-pair): the IsZero indicator over the
/// RLC-fold predicate, its selection + pseudo-inverse commitments, the keep-complement
/// compaction, and this stage's compacted output (carried because it is an
/// INTERMEDIATE relation the parent envelope never sees — the compaction argument
/// authenticates it against `com_keep`).
#[derive(Clone, Debug)]
pub struct TupleDisequalityStageProof {
    pub indicator: FilterProof,
    pub selection_commitment: ArkGT,
    pub inverse_commitment: ArkGT,
    pub compaction: CompactionOnlyProof,
    pub output_commitments: Vec<ArkGT>,
    pub output_len: usize,
}

/// The complete tuple-disequality proof: one stage per hop-pair, applied in order.
#[derive(Clone, Debug)]
pub struct TupleDisequalityProof {
    pub stages: Vec<TupleDisequalityStageProof>,
}

/// Prover result: the final compacted relation (all columns retained, only real rows
/// dropped) plus the proof.
pub struct TupleDisequalityResult {
    pub proof: TupleDisequalityProof,
    pub commitments: Vec<ArkGT>,
    pub columns: Vec<Vec<ArkFr>>,
    pub log_domain: usize,
    pub n: usize,
}

fn validate_relation(relation: &RelationCommit) -> io::Result<()> {
    if relation.commitments.is_empty() {
        return Err(invalid("disequality relation has no columns"));
    }
    if relation.log_domain == 0 || relation.log_domain >= usize::BITS as usize {
        return Err(invalid("disequality relation domain out of range"));
    }
    if relation.n > (1usize << relation.log_domain) {
        return Err(invalid("disequality relation length exceeds its domain"));
    }
    Ok(())
}

fn validate_pairs(pairs: &[DisequalityPair], arity: usize) -> io::Result<()> {
    if pairs.is_empty() {
        return Err(invalid("tuple disequality needs at least one hop-pair"));
    }
    let check = |node: &DisequalityNode| match node {
        DisequalityNode::Column(column) if *column >= arity => {
            Err(invalid("disequality column out of range"))
        }
        _ => Ok(()),
    };
    for pair in pairs {
        check(&pair.a0)?;
        check(&pair.a1)?;
        check(&pair.b0)?;
        check(&pair.b1)?;
        if let Some((pa, pb)) = &pair.predicate {
            check(pa)?;
            check(pb)?;
        }
    }
    Ok(())
}

/// Bind one stage's statement into the transcript: the domain tag, the current
/// relation meta + column commitments, and the pair descriptor. `γ` is drawn ONLY
/// after this, so no prover can adapt a committed column to forge a reuse. Identical
/// on both sides.
fn bind_stage(
    transcript: &mut Transcript254,
    commitments: &[ArkGT],
    log_domain: usize,
    n: usize,
    pair: &DisequalityPair,
) {
    let scalar = |value: &ArkFr| {
        let mut bytes = Vec::new();
        value
            .serialize_compressed(&mut bytes)
            .expect("serializing a field element to memory cannot fail");
        bytes
    };
    let node = |transcript: &mut Transcript254, node: &DisequalityNode| match node {
        DisequalityNode::Column(column) => {
            transcript.append_bytes(b"pow/tuple-diseq/node-kind", &[0]);
            transcript.append_bytes(b"pow/tuple-diseq/node-col", &(*column as u64).to_le_bytes());
        }
        DisequalityNode::Constant(value) => {
            transcript.append_bytes(b"pow/tuple-diseq/node-kind", &[1]);
            transcript.append_bytes(b"pow/tuple-diseq/node-val", &scalar(value));
        }
    };
    transcript.append_bytes(b"pow/tuple-diseq/domain", TDQ_DOMAIN);
    for value in [log_domain, n, commitments.len()] {
        transcript.append_bytes(b"pow/tuple-diseq/meta", &(value as u64).to_le_bytes());
    }
    for commitment in commitments {
        transcript.append_group(b"pow/tuple-diseq/input", commitment);
    }
    for reference in [&pair.a0, &pair.a1, &pair.b0, &pair.b1] {
        node(transcript, reference);
    }
    // Bind the full-triple predicate positions ONLY when present, so a 2-column pair's transcript is
    // byte-identical to before this generalization (P5a unbroken) while a 3-column pair draws γ over a
    // distinct statement (a 2-column proof can never be replayed as a 3-column one, or vice versa).
    if let Some((pa, pb)) = &pair.predicate {
        transcript.append_bytes(b"pow/tuple-diseq/pred", &[1]);
        node(transcript, pa);
        node(transcript, pb);
    }
}

/// Accumulate the RLC-fold predicate coefficients + constant offset for one pair at a
/// drawn `γ`. Returns `(coeffs_by_column, offset)` where the predicate column is
/// `P_i = Σ coeffs[c]·col_c[i]` and the IsZero constant is `pred = -offset` (so
/// `P_i == pred  ⇔  Δ_i == 0`).
///
/// The fingerprint difference is `Δ = (a0−b0) + γ·(a1−b1) [+ γ²·(pa−pb)]`, where the γ² predicate
/// term is present iff `pair.predicate` is `Some` (the full-triple fold). Schwartz–Zippel: `Δ == 0`
/// iff every coordinate difference is zero (subject, object, and — for a 3-column pair — predicate),
/// except with probability ≤ deg(Δ)/|F| ≤ 2/|F|.
fn fold_pair(pair: &DisequalityPair, gamma: ArkFr) -> (BTreeMap<usize, ArkFr>, ArkFr) {
    let mut coeffs: BTreeMap<usize, ArkFr> = BTreeMap::new();
    let mut offset = ArkFr::zero();
    let add = |node: &DisequalityNode, weight: ArkFr, coeffs: &mut BTreeMap<usize, ArkFr>, offset: &mut ArkFr| {
        match node {
            DisequalityNode::Column(column) => {
                let entry = coeffs.entry(*column).or_insert_with(ArkFr::zero);
                *entry = *entry + weight;
            }
            DisequalityNode::Constant(value) => {
                *offset = *offset + weight * *value;
            }
        }
    };
    let one = ArkFr::one();
    add(&pair.a0, one, &mut coeffs, &mut offset);
    add(&pair.b0, ArkFr::zero() - one, &mut coeffs, &mut offset);
    add(&pair.a1, gamma, &mut coeffs, &mut offset);
    add(&pair.b1, ArkFr::zero() - gamma, &mut coeffs, &mut offset);
    if let Some((pa, pb)) = &pair.predicate {
        // γ² weight for the predicate id column (walk-frame predicate, unchanged by orientation).
        let gamma2 = gamma * gamma;
        add(pa, gamma2, &mut coeffs, &mut offset);
        add(pb, ArkFr::zero() - gamma2, &mut coeffs, &mut offset);
    }
    (coeffs, offset)
}

/// Build the predicate column and its homomorphic commitment from a folded pair. A
/// degenerate pair (every column coefficient cancels to zero) would make the predicate
/// a pure constant, which cannot correspond to two DISTINCT edges over fresh
/// intermediates; reject it closed rather than silently drop/keep every row.
fn predicate_from_fold(
    coeffs: &BTreeMap<usize, ArkFr>,
    offset: ArkFr,
    columns: Option<&[Vec<ArkFr>]>,
    commitments: &[ArkGT],
    size: usize,
) -> io::Result<(Option<Vec<ArkFr>>, ArkGT, ArkFr)> {
    let nonzero: Vec<(usize, ArkFr)> = coeffs
        .iter()
        .filter(|(_, coeff)| !coeff.is_zero())
        .map(|(column, coeff)| (*column, *coeff))
        .collect();
    if nonzero.is_empty() {
        return Err(invalid(
            "degenerate tuple-disequality pair: fingerprint has no committed column term",
        ));
    }
    // pred = -offset  (so P_i == pred  ⇔  P_i + offset == Δ_i == 0).
    let pred = ArkFr::zero() - offset;
    let mut commitment = <ArkGT as Group>::identity();
    for (column, coeff) in &nonzero {
        commitment = commitment.add(&commitments[*column].scale(coeff));
    }
    let column = columns.map(|columns| {
        let mut predicate = vec![ArkFr::zero(); size];
        for (col, coeff) in &nonzero {
            for (row, value) in columns[*col].iter().enumerate() {
                predicate[row] = predicate[row] + *coeff * *value;
            }
        }
        predicate
    });
    Ok((column, commitment, pred))
}

/// The χ_n real-prefix indicator column (1 on `[0, n)`, 0 on padding), committed by
/// both sides so the keep-complement `com_keep = com(χ_n) − com(sel)` is reconstructed
/// identically. `n` is PUBLIC (the layer/relation row count), so this leaks nothing.
fn chi_commitment(n: usize, size: usize) -> io::Result<(Vec<ArkFr>, ArkGT)> {
    let mut chi = vec![ArkFr::zero(); size];
    for value in chi.iter_mut().take(n) {
        *value = ArkFr::one();
    }
    let commitment = commit_column(&chi)?;
    Ok((chi, commitment))
}

/// Prove the tuple-disequality filter: starting from `relation`, drop exactly the rows in
/// which any listed hop-`pairs` edge coincides (same subject AND object under the fixed
/// predicate), applied as a sequence of stages. Returns the surviving relation and the
/// proof. Used to enforce edge-simple semantics for repeated-predicate paths.
pub fn prove_tuple_disequality(
    relation: &RelationWitness,
    pairs: &[DisequalityPair],
    transcript: &mut Transcript254,
) -> io::Result<TupleDisequalityResult> {
    prove_tuple_disequality_with_context(&StoreProverContext::cpu(), relation, pairs, transcript)
}

/// As [`prove_tuple_disequality`], with an explicit prover-arithmetic context.
pub fn prove_tuple_disequality_with_context(
    context: &StoreProverContext<'_>,
    relation: &RelationWitness,
    pairs: &[DisequalityPair],
    transcript: &mut Transcript254,
) -> io::Result<TupleDisequalityResult> {
    validate_relation(&relation.view())?;
    validate_pairs(pairs, relation.commitments.len())?;
    let log_domain = relation.log_domain;
    let size = 1usize << log_domain;

    // The evolving (compacted) relation. Every stage keeps ALL columns; only the real
    // row count shrinks, so pair column indices stay valid across stages.
    let mut columns = relation.columns.clone();
    let mut commitments = relation.commitments.clone();
    let mut n = relation.n;

    let mut stages = Vec::with_capacity(pairs.len());
    for pair in pairs {
        // (1) Bind this stage's committed columns, THEN draw the fold challenge γ.
        bind_stage(transcript, &commitments, log_domain, n, pair);
        let gamma = transcript.challenge_scalar(b"pow/tuple-diseq/gamma");

        // (2) RLC fingerprint difference as an IsZero predicate over committed columns.
        let (coeffs, offset) = fold_pair(pair, gamma);
        let (predicate, predicate_commitment, pred) =
            predicate_from_fold(&coeffs, offset, Some(&columns), &commitments, size)?;
        let predicate = predicate.expect("prover supplies column data");

        // (3) Pin sel = [Δ == 0 ∧ i < n] via the audited IsZero indicator.
        let (indicator, selection_commitment, inverse_commitment, _) =
            prove_filter_indicator_with_context(
                context,
                &predicate,
                pred,
                n,
                &predicate_commitment,
                transcript,
            )?;
        // Recompute the honest sel (deterministic — matches the committed sel) to form
        // the keep-complement. w is unused here (only sel drives the compaction).
        let (sel, _w) = honest_witness(&predicate, pred, n);

        // (4) keep = χ_n − sel, com_keep = com(χ_n) − com(sel) (homomorphic).
        let (chi, chi_commitment) = chi_commitment(n, size)?;
        let keep: Vec<ArkFr> = chi
            .iter()
            .zip(&sel)
            .map(|(chi, sel)| *chi - *sel)
            .collect();
        let keep_commitment = chi_commitment.add(&selection_commitment.scale(&(ArkFr::zero() - ArkFr::one())));

        // (5) Compact keeping the Δ-nonzero real rows (the audited LogUp compaction).
        let (compaction, output_commitments, output_columns, m) =
            prove_compaction_only_with_context(
                context,
                &columns,
                &commitments,
                &keep,
                &keep_commitment,
                transcript,
            )?;

        stages.push(TupleDisequalityStageProof {
            indicator,
            selection_commitment,
            inverse_commitment,
            compaction,
            output_commitments: output_commitments.clone(),
            output_len: m,
        });

        columns = output_columns;
        commitments = output_commitments;
        n = m;
    }

    Ok(TupleDisequalityResult {
        proof: TupleDisequalityProof { stages },
        commitments,
        columns,
        log_domain,
        n,
    })
}

/// Verify a tuple-disequality proof: the relation committed by `output_commitments` is
/// exactly `relation` with every edge-reuse row (per the hop-`pairs`) removed and nothing
/// else changed.
#[allow(clippy::too_many_arguments)]
pub fn verify_tuple_disequality(
    relation: &RelationCommit,
    pairs: &[DisequalityPair],
    output_commitments: &[ArkGT],
    output_len: usize,
    output_log_domain: usize,
    proof: &TupleDisequalityProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    validate_relation(relation)?;
    validate_pairs(pairs, relation.commitments.len())?;
    if output_log_domain != relation.log_domain {
        return Err(invalid("tuple disequality output domain mismatch"));
    }
    if proof.stages.len() != pairs.len() {
        return Err(invalid("tuple disequality stage/pair count mismatch"));
    }
    let log_domain = relation.log_domain;
    let size = 1usize << log_domain;

    let mut commitments = relation.commitments.clone();
    let mut n = relation.n;

    for (pair, stage) in pairs.iter().zip(&proof.stages) {
        // (1) Rebind + redraw γ exactly as the prover did.
        bind_stage(transcript, &commitments, log_domain, n, pair);
        let gamma = transcript.challenge_scalar(b"pow/tuple-diseq/gamma");

        // (2) Reconstruct the predicate commitment + constant (verifier has no columns).
        let (coeffs, offset) = fold_pair(pair, gamma);
        let (_, predicate_commitment, pred) =
            predicate_from_fold(&coeffs, offset, None, &commitments, size)?;

        // (3) Verify the IsZero indicator pins the reuse selection.
        verify_filter_indicator(
            &predicate_commitment,
            pred,
            n,
            &stage.selection_commitment,
            &stage.inverse_commitment,
            &stage.indicator,
            transcript,
        )?;

        // (4) Reconstruct com_keep = com(χ_n) − com(sel).
        let (_, chi_commitment) = chi_commitment(n, size)?;
        let keep_commitment =
            chi_commitment.add(&stage.selection_commitment.scale(&(ArkFr::zero() - ArkFr::one())));

        // (5) Verify the compaction produced this stage's authenticated output.
        verify_compaction_only(
            &commitments,
            &keep_commitment,
            &stage.output_commitments,
            stage.output_len,
            &stage.compaction,
            transcript,
        )?;

        if stage.output_commitments.len() != commitments.len() {
            return Err(invalid("tuple disequality stage changed the column count"));
        }
        commitments = stage.output_commitments.clone();
        n = stage.output_len;
    }

    // The final stage's output must be exactly the relation the parent envelope claims.
    if n != output_len {
        return Err(invalid("tuple disequality output length mismatch"));
    }
    if commitments.len() != output_commitments.len()
        || commitments
            .iter()
            .zip(output_commitments)
            .any(|(computed, claimed)| computed != claimed)
    {
        return Err(invalid("tuple disequality output commitment mismatch"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fr(value: u64) -> ArkFr {
        ArkFr::from_u64(value)
    }

    fn transcript() -> Transcript254 {
        Transcript254::new(b"tuple-disequality-test")
    }

    // A two-hop relation over columns [x0, x1, x2] (subject, middle, object). One
    // hop-pair (edge0 = (x0,x1), edge1 = (x1,x2)) drops rows where x0==x1 AND x1==x2
    // — i.e. a single self-loop traversed twice.
    fn pair_two_hop() -> DisequalityPair {
        DisequalityPair {
            a0: DisequalityNode::Column(0),
            a1: DisequalityNode::Column(1),
            b0: DisequalityNode::Column(1),
            b1: DisequalityNode::Column(2),
            predicate: None,
        }
    }

    #[test]
    fn drops_only_edge_reuse_rows_and_keeps_node_repeats() {
        // Rows (x0,x1,x2):
        //   (1,1,1)  self-loop twice  -> edge0==edge1 -> DROP
        //   (1,2,1)  a->b->a          -> edges (1,2),(2,1) distinct -> KEEP
        //   (1,2,3)  a->b->c          -> KEEP
        //   (1,1,2)  (1,1),(1,2)      -> distinct -> KEEP  (node repeat, edge distinct)
        let x0 = vec![fr(1), fr(1), fr(1), fr(1)];
        let x1 = vec![fr(1), fr(2), fr(2), fr(1)];
        let x2 = vec![fr(1), fr(1), fr(3), fr(2)];
        let relation = RelationWitness::commit(vec![x0, x1, x2], 4).unwrap();
        let pairs = [pair_two_hop()];

        let result =
            prove_tuple_disequality(&relation, &pairs, &mut transcript()).unwrap();
        verify_tuple_disequality(
            &relation.view(),
            &pairs,
            &result.commitments,
            result.n,
            result.log_domain,
            &result.proof,
            &mut transcript(),
        )
        .unwrap();

        // Exactly the three non-reuse rows survive.
        assert_eq!(result.n, 3);
        let size = 1usize << result.log_domain;
        assert_eq!(result.columns[0][..3], [fr(1), fr(1), fr(1)]);
        assert_eq!(result.columns[1][..3], [fr(2), fr(2), fr(1)]);
        assert_eq!(result.columns[2][..3], [fr(1), fr(3), fr(2)]);
        // Padding is zeroed.
        for column in &result.columns {
            assert_eq!(column.len(), size);
            assert!(column[3..].iter().all(|value| value.is_zero()));
        }
    }

    #[test]
    fn padding_row_whose_delta_is_zero_never_survives() {
        // n = 3 real rows, all KEEP (no reuse). The domain pads to 4, so row 3 is
        // genuine padding: all-zero, its Δ = (0-0) + γ(0-0) = 0 collides with the reuse
        // predicate. The χ_n mask must force sel = 0 there so keep = χ_n − sel = 0 and
        // it never compacts in — otherwise the surviving count would be 4, not 3.
        let x0 = vec![fr(5), fr(7), fr(2)];
        let x1 = vec![fr(6), fr(8), fr(3)];
        let x2 = vec![fr(9), fr(4), fr(1)];
        let relation = RelationWitness::commit(vec![x0, x1, x2], 3).unwrap();
        let pairs = [pair_two_hop()];
        // Sanity: domain is padded beyond n, so a genuine all-zero padding row exists.
        assert!((1usize << relation.log_domain) > relation.n);

        let result = prove_tuple_disequality(&relation, &pairs, &mut transcript()).unwrap();
        verify_tuple_disequality(
            &relation.view(),
            &pairs,
            &result.commitments,
            result.n,
            result.log_domain,
            &result.proof,
            &mut transcript(),
        )
        .unwrap();
        assert_eq!(result.n, 3);
    }

    #[test]
    fn constant_subject_endpoint_folds_into_the_predicate() {
        // edge0 = (const a=1, x0), edge1 = (x0, x1): reuse iff 1==x0 AND x0==x1, i.e.
        // the self-loop (a,p,a) traversed twice from a fixed start.
        //   (x0,x1): (1,1) reuse -> DROP;  (2,1) keep;  (2,3) keep
        let x0 = vec![fr(1), fr(2), fr(2)];
        let x1 = vec![fr(1), fr(1), fr(3)];
        let relation = RelationWitness::commit(vec![x0, x1], 3).unwrap();
        let pairs = [DisequalityPair {
            a0: DisequalityNode::Constant(fr(1)),
            a1: DisequalityNode::Column(0),
            b0: DisequalityNode::Column(0),
            b1: DisequalityNode::Column(1),
            predicate: None,
        }];
        let result = prove_tuple_disequality(&relation, &pairs, &mut transcript()).unwrap();
        verify_tuple_disequality(
            &relation.view(),
            &pairs,
            &result.commitments,
            result.n,
            result.log_domain,
            &result.proof,
            &mut transcript(),
        )
        .unwrap();
        assert_eq!(result.n, 2);
        assert_eq!(result.columns[0][..2], [fr(2), fr(2)]);
        assert_eq!(result.columns[1][..2], [fr(1), fr(3)]);
    }

    #[test]
    fn multi_pair_three_hop_intersects_all_disequalities() {
        // Three hops -> nodes [x0,x1,x2,x3], edges e0=(x0,x1) e1=(x1,x2) e2=(x2,x3).
        // C(3,2) = 3 pairs. Row keeps iff NO two edges coincide.
        //   (1,1,1,1): e0=e1=e2 self-loop -> DROP
        //   (1,2,1,2): e0=(1,2) e2=(1,2) coincide -> DROP
        //   (1,2,3,4): all distinct -> KEEP
        let x0 = vec![fr(1), fr(1), fr(1)];
        let x1 = vec![fr(1), fr(2), fr(2)];
        let x2 = vec![fr(1), fr(1), fr(3)];
        let x3 = vec![fr(1), fr(2), fr(4)];
        let relation = RelationWitness::commit(vec![x0, x1, x2, x3], 3).unwrap();
        let col = DisequalityNode::Column;
        let pairs = [
            DisequalityPair { a0: col(0), a1: col(1), b0: col(1), b1: col(2), predicate: None }, // e0,e1
            DisequalityPair { a0: col(0), a1: col(1), b0: col(2), b1: col(3), predicate: None }, // e0,e2
            DisequalityPair { a0: col(1), a1: col(2), b0: col(2), b1: col(3), predicate: None }, // e1,e2
        ];
        let result = prove_tuple_disequality(&relation, &pairs, &mut transcript()).unwrap();
        verify_tuple_disequality(
            &relation.view(),
            &pairs,
            &result.commitments,
            result.n,
            result.log_domain,
            &result.proof,
            &mut transcript(),
        )
        .unwrap();
        assert_eq!(result.n, 1);
        assert_eq!(result.columns[0][0], fr(1));
        assert_eq!(result.columns[3][0], fr(4));
    }

    // A 2-hop WILDCARD relation over columns [s0, p0, mid, p1, o1]:
    //   edge0 walk = (s0=col0, o0=mid=col2), predicate p0=col1
    //   edge1 walk = (mid=col2, o1=col4),    predicate p1=col3
    // The full-triple pair drops rows where s0==mid AND mid==o1 AND p0==p1 — the SAME stored edge
    // (a,p,a) traversed twice — but KEEPS a row whose (s,o) coincide yet whose PREDICATE differs.
    fn pair_full_triple() -> DisequalityPair {
        DisequalityPair {
            a0: DisequalityNode::Column(0),
            a1: DisequalityNode::Column(2),
            b0: DisequalityNode::Column(2),
            b1: DisequalityNode::Column(4),
            predicate: Some((DisequalityNode::Column(1), DisequalityNode::Column(3))),
        }
    }

    #[test]
    fn full_triple_disequality_drops_full_edge_reuse_and_keeps_predicate_difference() {
        // Rows (s0, p0, mid, p1, o1):
        //   A (1,5,1,5,1)  s0==mid==o1, p0==p1  -> SAME stored edge (a,5,a) twice     -> DROP
        //   B (1,5,1,6,1)  s0==mid==o1 but p0!=p1 -> (a,5,a) then (a,6,a): distinct    -> KEEP
        //   C (1,5,2,5,3)  mid differs -> two distinct edges                          -> KEEP
        //   D (2,5,1,5,1)  s0 != mid -> not the same edge                             -> KEEP
        let s0 = vec![fr(1), fr(1), fr(1), fr(2)];
        let p0 = vec![fr(5), fr(5), fr(5), fr(5)];
        let mid = vec![fr(1), fr(1), fr(2), fr(1)];
        let p1 = vec![fr(5), fr(6), fr(5), fr(5)];
        let o1 = vec![fr(1), fr(1), fr(3), fr(1)];
        let relation = RelationWitness::commit(vec![s0, p0, mid, p1, o1], 4).unwrap();
        let pairs = [pair_full_triple()];

        let result = prove_tuple_disequality(&relation, &pairs, &mut transcript()).unwrap();
        verify_tuple_disequality(
            &relation.view(),
            &pairs,
            &result.commitments,
            result.n,
            result.log_domain,
            &result.proof,
            &mut transcript(),
        )
        .unwrap();

        // Exactly the three non-full-reuse rows (B, C, D) survive; the full-triple reuse row A drops.
        assert_eq!(result.n, 3, "only the same-subject-predicate-object reuse row is dropped");
        // Row B survived (the (s,o)-coincident but predicate-differing row): its p1 is 6, mid is 1.
        assert_eq!(result.columns[0][..3], [fr(1), fr(1), fr(2)]);
        assert_eq!(result.columns[3][..3], [fr(6), fr(5), fr(5)], "predicate-difference row kept");
        assert_eq!(result.columns[2][..3], [fr(1), fr(2), fr(1)]);
    }

    #[test]
    fn two_column_pair_ignores_predicate_and_would_drop_the_predicate_difference_row() {
        // The SAME rows as above, but a 2-column (predicate-BLIND) pair drops on (s,o) alone: rows A
        // AND B both collapse (s0==mid==o1), so ONLY C and D survive. This contrasts the 3-column fold
        // (which keeps B) with the 2-column fold (which cannot distinguish B's differing predicate) —
        // exactly why a wildcard multi-hop needs the full-triple arity.
        let s0 = vec![fr(1), fr(1), fr(1), fr(2)];
        let p0 = vec![fr(5), fr(5), fr(5), fr(5)];
        let mid = vec![fr(1), fr(1), fr(2), fr(1)];
        let p1 = vec![fr(5), fr(6), fr(5), fr(5)];
        let o1 = vec![fr(1), fr(1), fr(3), fr(1)];
        let relation = RelationWitness::commit(vec![s0, p0, mid, p1, o1], 4).unwrap();
        let pairs = [DisequalityPair {
            a0: DisequalityNode::Column(0),
            a1: DisequalityNode::Column(2),
            b0: DisequalityNode::Column(2),
            b1: DisequalityNode::Column(4),
            predicate: None,
        }];
        let result = prove_tuple_disequality(&relation, &pairs, &mut transcript()).unwrap();
        verify_tuple_disequality(
            &relation.view(),
            &pairs,
            &result.commitments,
            result.n,
            result.log_domain,
            &result.proof,
            &mut transcript(),
        )
        .unwrap();
        assert_eq!(result.n, 2, "2-column fold drops BOTH (s,o)-coincident rows A and B");
    }

    #[test]
    fn full_triple_pair_with_predicate_constant_matches_a_mixed_concrete_hop() {
        // The MIXED `Sequence([., p])` shape: hop0's predicate is a wildcard COLUMN, hop1's is the
        // concrete CONSTANT id 5. Drop iff s0==mid AND mid==o1 AND p0(col1)==5. Same rows as above:
        //   A (p0=5) -> 5==5 & full self-loop -> DROP;  B (p0=5, but p1=6 unused) -> still full reuse
        //   on (s,o,p0==5)? Row B has p0=5 too, so with predicate = (col1=p0) vs const 5, B's p0==5
        //   -> B DROPS here (its hop0 predicate matches the constant). Only C, D survive.
        let s0 = vec![fr(1), fr(1), fr(1), fr(2)];
        let p0 = vec![fr(5), fr(5), fr(5), fr(5)];
        let mid = vec![fr(1), fr(1), fr(2), fr(1)];
        let p1 = vec![fr(5), fr(6), fr(5), fr(5)];
        let o1 = vec![fr(1), fr(1), fr(3), fr(1)];
        let relation = RelationWitness::commit(vec![s0, p0, mid, p1, o1], 4).unwrap();
        // Predicate positions: hop0 = wildcard column p0 (col1); hop1 = concrete constant id 5.
        let pairs = [DisequalityPair {
            a0: DisequalityNode::Column(0),
            a1: DisequalityNode::Column(2),
            b0: DisequalityNode::Column(2),
            b1: DisequalityNode::Column(4),
            predicate: Some((DisequalityNode::Column(1), DisequalityNode::Constant(fr(5)))),
        }];
        let result = prove_tuple_disequality(&relation, &pairs, &mut transcript()).unwrap();
        verify_tuple_disequality(
            &relation.view(),
            &pairs,
            &result.commitments,
            result.n,
            result.log_domain,
            &result.proof,
            &mut transcript(),
        )
        .unwrap();
        // A and B (both s0==mid==o1 with p0==5==const) drop; C (mid differs) and D (s0 differs) keep.
        assert_eq!(result.n, 2);
        assert_eq!(result.columns[0][..2], [fr(1), fr(2)]);
    }

    #[test]
    fn tampered_output_length_is_rejected() {
        let x0 = vec![fr(1), fr(1), fr(1)];
        let x1 = vec![fr(1), fr(2), fr(2)];
        let x2 = vec![fr(1), fr(1), fr(3)];
        let relation = RelationWitness::commit(vec![x0, x1, x2], 3).unwrap();
        let pairs = [pair_two_hop()];
        let result = prove_tuple_disequality(&relation, &pairs, &mut transcript()).unwrap();
        // Claim one more surviving row than the proof authenticates.
        assert!(verify_tuple_disequality(
            &relation.view(),
            &pairs,
            &result.commitments,
            result.n + 1,
            result.log_domain,
            &result.proof,
            &mut transcript(),
        )
        .is_err());
    }

    #[test]
    fn tampered_output_commitment_is_rejected() {
        let x0 = vec![fr(1), fr(1)];
        let x1 = vec![fr(1), fr(2)];
        let x2 = vec![fr(1), fr(1)];
        let relation = RelationWitness::commit(vec![x0, x1, x2], 2).unwrap();
        let pairs = [pair_two_hop()];
        let result = prove_tuple_disequality(&relation, &pairs, &mut transcript()).unwrap();
        let mut tampered = result.commitments.clone();
        tampered[0] = commit_column(&vec![fr(99); 1usize << result.log_domain]).unwrap();
        assert!(verify_tuple_disequality(
            &relation.view(),
            &pairs,
            &tampered,
            result.n,
            result.log_domain,
            &result.proof,
            &mut transcript(),
        )
        .is_err());
    }
}
