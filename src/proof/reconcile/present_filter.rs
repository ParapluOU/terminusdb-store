//! Single-atom BGP read over the authenticated `present` relation.
//!
//! Given a reconciled+authenticated `present`, a one-atom pattern with ONE constant position
//! (e.g. `Triple(?s, <P>, ?o)`) is an EXACT filter-plus-compaction over `present`'s committed
//! columns, proved on the SAME transcript as the reconciliation so the answer is bound to the
//! authenticated relation and, through it, to the trusted head root. The result columns are
//! the authenticated hash bindings of the matching rows (resolution to IRIs/literals is the
//! `present_result` submodule).

use super::*;

// ===========================================================================
// Wave 3 — single-atom BGP over the authenticated `present` relation.
//
// `present` is authenticated (by the reconciliation above) as three hash columns
// [hash_object(S), hash_term(P), hash_object(O)] committed at 2^K with real length M.
// A single-atom read with ONE constant position is an EXACT filter+compaction over
// those committed columns, proved on the SAME transcript as the reconciliation (so it
// is bound to the authenticated relation). The audited Q1 filter indicator pins the
// selection to EXACTLY the equality-with-constant indicator (complete AND sound: every
// matching row is selected, no non-matching row is), and the audited compaction emits
// EXACTLY the selected rows of the two output columns. The delivered result columns are
// authenticated hashes of the output bindings; resolving them to canonical IRIs/values
// is deferred (see the module report).
// ===========================================================================

/// Which `present` position carries the query CONSTANT (the other two are output vars).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterPosition {
    Subject,
    Predicate,
    Object,
}

/// A single-atom filter proof over `present`: the equality-indicator selection and the
/// compaction of the two output columns to the matching rows.
#[derive(Clone, Debug)]
pub struct PresentFilterProof {
    pub filter: FilterProof,
    pub com_sel: ArkGT,
    pub com_w: ArkGT,
    pub compaction: CompactionOnlyProof,
    /// The two authenticated output hash columns (the non-constant variables), matching rows.
    pub com_out: Vec<ArkGT>,
    pub out_len: usize,
}

/// The delivered read result: the output hash columns (real prefix) and the row count.
pub struct PresentReadResult {
    /// Two columns of authenticated binding HASHES (output variables), length `len`.
    pub columns: Vec<Vec<ArkFr>>,
    pub len: usize,
    /// UNTRUSTED per-column dictionary id-hints (I2a) — one hint column per output `columns`
    /// entry, aligned row-for-row, letting the verifier resolve each hash binding by an id-routed
    /// Merkle opening under a MANDATORY belt-hash instead of a catalog scan. Empty on the
    /// id-carrying result paths (join / BGP), which resolve the value directly as an id. NOT
    /// committed and NOT part of `commitment_root` — the belt-hash is the only thing that binds it.
    pub id_hints: Vec<Vec<u64>>,
}

/// Map a filter position to `(filter column, filter commitment, [output columns], [output commitments])`.
pub(super) fn filter_selection<'a>(
    artifacts: &'a ReconcileArtifacts,
    position: FilterPosition,
) -> (&'a [ArkFr], ArkGT, [&'a [ArkFr]; 2], [ArkGT; 2]) {
    match position {
        FilterPosition::Predicate => (
            artifacts.present_p.as_slice(),
            artifacts.com_present_p,
            [artifacts.present_s.as_slice(), artifacts.present_o.as_slice()],
            [artifacts.com_present_s, artifacts.com_present_o],
        ),
        FilterPosition::Subject => (
            artifacts.present_s.as_slice(),
            artifacts.com_present_s,
            [artifacts.present_p.as_slice(), artifacts.present_o.as_slice()],
            [artifacts.com_present_p, artifacts.com_present_o],
        ),
        FilterPosition::Object => (
            artifacts.present_o.as_slice(),
            artifacts.com_present_o,
            [artifacts.present_s.as_slice(), artifacts.present_p.as_slice()],
            [artifacts.com_present_s, artifacts.com_present_p],
        ),
    }
}

/// The two OUTPUT-column dictionary id columns (as field elements) parallel to
/// [`filter_selection`]'s output hash columns, for sourcing the I2a resolution id-hints. Uses the
/// artifacts' `present_id_*` columns (`present_id_s`/`present_id_o` are node/value combined-dict
/// ids, `present_id_p` a predicate id), which are indexed identically to `present_s`/`present_o`/
/// `present_p`, so compacting them by the SAME filter selection yields row-aligned hints whose
/// opened term hashes back to the output hash (the belt-hash the verifier re-checks).
pub(super) fn filter_id_sources(artifacts: &ReconcileArtifacts, position: FilterPosition) -> [&[ArkFr]; 2] {
    match position {
        // output [S, O]
        FilterPosition::Predicate => [artifacts.present_id_s.as_slice(), artifacts.present_id_o.as_slice()],
        // output [P, O]
        FilterPosition::Subject => [artifacts.present_id_p.as_slice(), artifacts.present_id_o.as_slice()],
        // output [S, P]
        FilterPosition::Object => [artifacts.present_id_s.as_slice(), artifacts.present_id_p.as_slice()],
    }
}

/// The verifier-side output commitments for a filter position (from the reconciled handle).
pub(super) fn filter_output_commitments(verified: &ReconcileVerified, position: FilterPosition) -> (ArkGT, [ArkGT; 2]) {
    match position {
        FilterPosition::Predicate => (verified.com_present_p, [verified.com_present_s, verified.com_present_o]),
        FilterPosition::Subject => (verified.com_present_s, [verified.com_present_p, verified.com_present_o]),
        FilterPosition::Object => (verified.com_present_o, [verified.com_present_s, verified.com_present_p]),
    }
}

/// Prove an exact equality-with-constant filter on `filter_col` (committed as `com_filter`,
/// real length `real_n`) and compact `sources` to the matching rows. Reused by the
/// single-atom read (hash output columns) AND each join atom (id output columns).
#[allow(clippy::too_many_arguments)]
pub(super) fn prove_filter_part(
    context: &StoreProverContext<'_>,
    filter_col: &[ArkFr],
    com_filter: &ArkGT,
    constant: ArkFr,
    real_n: usize,
    sources: &[Vec<ArkFr>],
    source_coms: &[ArkGT],
    transcript: &mut Transcript254,
) -> io::Result<(PresentFilterProof, Vec<Vec<ArkFr>>)> {
    let size = filter_col.len();
    let sel: Vec<ArkFr> = (0..size)
        .map(|i| if i < real_n && filter_col[i] == constant { f1() } else { f0() })
        .collect();
    let (filter, com_sel, com_w, _sel_eval) = prove_filter_indicator_with_context(
        context, filter_col, constant, real_n, com_filter, transcript,
    )?;
    let (compaction, com_out, out_cols, out_len) = prove_compaction_only_with_context(
        context, sources, source_coms, &sel, &com_sel, transcript,
    )?;
    Ok((PresentFilterProof { filter, com_sel, com_w, compaction, com_out, out_len }, out_cols))
}

pub(super) fn verify_filter_part(
    com_filter: &ArkGT,
    constant: ArkFr,
    real_n: usize,
    source_coms: &[ArkGT],
    proof: &PresentFilterProof,
    transcript: &mut Transcript254,
) -> io::Result<usize> {
    verify_filter_indicator(
        com_filter, constant, real_n, &proof.com_sel, &proof.com_w, &proof.filter, transcript,
    )?;
    verify_compaction_only(
        source_coms, &proof.com_sel, &proof.com_out, proof.out_len, &proof.compaction, transcript,
    )?;
    Ok(proof.out_len)
}

pub(super) fn prove_present_filter(
    context: &StoreProverContext<'_>,
    artifacts: &ReconcileArtifacts,
    position: FilterPosition,
    constant_hash: ArkFr,
    transcript: &mut Transcript254,
) -> io::Result<(PresentFilterProof, PresentReadResult)> {
    let (filter_col, com_filter, sources, source_coms) = filter_selection(artifacts, position);
    let id_sources = filter_id_sources(artifacts, position);
    let source_vecs: Vec<Vec<ArkFr>> = sources.iter().map(|c| c.to_vec()).collect();
    let (part, out_cols) = prove_filter_part(
        context, filter_col, &com_filter, constant_hash, artifacts.present_len, &source_vecs, &source_coms, transcript,
    )?;
    let out_len = part.out_len;
    let columns: Vec<Vec<ArkFr>> = out_cols.iter().map(|c| c[..out_len].to_vec()).collect();
    // I2a: prover id-hints for the (hash-typed) output columns. The selection is exactly the
    // `prove_filter_part` indicator `sel[i] = (i < present_len) && filter_col[i] == constant_hash`,
    // and `compact_columns` emits selected rows in ASCENDING index order — so gathering the id
    // sources at the same ascending selected indices aligns each hint row-for-row with `columns`.
    // UNTRUSTED: the resolution belt-hash (`hash_object`/`hash_term`(opened term) == output hash)
    // is what binds them; a wrong hint is rejected at resolve time.
    let selected: Vec<usize> = (0..artifacts.present_len)
        .filter(|&i| filter_col[i] == constant_hash)
        .collect();
    let id_hints: Vec<Vec<u64>> = id_sources
        .iter()
        .map(|src| {
            selected
                .iter()
                .map(|&i| canonical_dictionary_id(&src[i]))
                .collect::<io::Result<Vec<u64>>>()
        })
        .collect::<io::Result<Vec<_>>>()?;
    debug_assert!(id_hints.iter().all(|h| h.len() == out_len));
    Ok((part, PresentReadResult { columns, len: out_len, id_hints }))
}

pub(super) fn verify_present_filter(
    verified: &ReconcileVerified,
    position: FilterPosition,
    constant_hash: ArkFr,
    proof: &PresentFilterProof,
    transcript: &mut Transcript254,
) -> io::Result<usize> {
    let (com_filter, source_coms) = filter_output_commitments(verified, position);
    verify_filter_part(&com_filter, constant_hash, verified.present_len, &source_coms, proof, transcript)
}

/// Prove a single-atom read `Triple(...)` with ONE constant position over the reconciled
/// N-layer `present`: reconcile + authenticated filter, one transcript. The delivered
/// result columns are the two output variables as authenticated binding HASHES.
pub fn prove_read_single_atom(
    layers: &[&LayerCommitment],
    position: FilterPosition,
    constant_hash: ArkFr,
) -> io::Result<(ReconcileProof, PresentFilterProof, PresentReadResult)> {
    prove_read_single_atom_with_context(
        &StoreProverContext::cpu(),
        layers,
        position,
        constant_hash,
    )
}

/// [`prove_read_single_atom`] with a caller-supplied prover context (for GPU/Blitzar MSM offload).
/// Identical bytes and proof to the CPU convenience.
pub fn prove_read_single_atom_with_context(
    context: &StoreProverContext<'_>,
    layers: &[&LayerCommitment],
    position: FilterPosition,
    constant_hash: ArkFr,
) -> io::Result<(ReconcileProof, PresentFilterProof, PresentReadResult)> {
    let mut transcript = Transcript254::new(b"parture-delta-reconcile-envelope");
    let (reconcile, artifacts) = prove_reconcile_core(context, layers, &mut transcript)?;
    let (filter, result) = prove_present_filter(context, &artifacts, position, constant_hash, &mut transcript)?;
    Ok((reconcile, filter, result))
}

/// Verify a single-atom read. On success returns the authenticated result row count.
pub fn verify_read_single_atom(
    layers: &[&LayerCommitment],
    trusted_head_root: ProofHash,
    position: FilterPosition,
    constant_hash: ArkFr,
    reconcile: &ReconcileProof,
    filter: &PresentFilterProof,
) -> io::Result<usize> {
    let mut transcript = Transcript254::new(b"parture-delta-reconcile-envelope");
    let views: Vec<VerifierCommitmentRef<'_>> = layers.iter().map(|l| l.verifier_view()).collect();
    let verified = verify_reconcile_core(&views, trusted_head_root, reconcile, &mut transcript)?;
    verify_present_filter(&verified, position, constant_hash, filter, &mut transcript)
}

