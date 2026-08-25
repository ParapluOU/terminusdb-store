//! Which committed triple columns a layer query reads, and how they are pulled out.
//!
//! A committed layer stores three overlapping "views" of its triples: the full
//! MATERIALIZED head (the layer's complete logical contents) and the two per-commit
//! DELTA files it introduced — its own ADDITIONS and its own REMOVALS. [`ColumnSource`]
//! names which of those three a given query reads. Everything else in this module is the
//! plumbing that, given a source, hands back the right `(subject, predicate, object)`
//! columns, the right root-bound column commitments, and the right real-row count — always
//! the exact bytes/ordering that were committed, so a proof built over them stays anchored
//! to the layer's `commitment_root`.
//!
//! It also holds two gates every query path goes through first: [`resolve_predicate_id`]
//! (turn a predicate IRI into its committed dictionary id) and
//! [`require_base_layer_for_query`] (delta-only layers are not directly query-provable).

use super::*;

/// Extract the `(S, P, O)` `ArkFr` columns and real-row count `n` from the complete
/// materialized head view, in the exact deterministic order and padding that produced
/// `materialized_column_commitments`. Equivalent to `source_layer_columns(_, Materialized)`;
/// retained for the commitment-convention test.
#[cfg(test)]
pub(super) fn materialized_layer_columns(
    layer: &LayerCommitment,
) -> io::Result<(Vec<ArkFr>, Vec<ArkFr>, Vec<ArkFr>, usize)> {
    source_layer_columns(layer, ColumnSource::Materialized)
}

/// Which committed triple-column set a layer query reads. `Materialized` is the complete
/// logical head view — the ONLY behavior before delta reads existed, and the default. The
/// two delta sources are the layer's OWN per-commit files: `Additions` backs WOQL
/// `AddedTriple` (the layer ADDITION file — "this commit, no deeper") and `Removals` backs
/// WOQL `DeletedTriple` (the layer REMOVAL file). Because these are the layer's own delta,
/// the proof is FAITHFUL with no parent-chain reasoning: the additions/removals `ProofRecord`
/// tables and their Dory column commitments (`column_commitments[0..3]` / `[3..6]`) are ALL
/// bound by the SAME `commitment_root` as the materialized columns.
///
/// The source is bound TWO ways, so a prover can neither prove additions and claim
/// materialized nor swap additions↔removals: (a) it is carried in [`LayerBgpPlan`], which the
/// verifier reconstructs from its OWN trusted compiled query (the envelope anti-substitution
/// property) and therefore fixes independently; and (b) it is domain-separated in the
/// Fiat–Shamir transcript (see `bind_layer_bgp_statement`). The `Materialized` tag emits NO
/// extra transcript bytes, so pre-existing materialized proofs are byte-identical.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ColumnSource {
    /// The complete logical head view (the default; the only behavior before delta reads).
    #[default]
    Materialized,
    /// The layer's own ADDITIONS delta file (backs WOQL `AddedTriple` — "this commit, no deeper").
    Additions,
    /// The layer's own REMOVALS delta file (backs WOQL `DeletedTriple`).
    Removals,
}

/// The prover-side `ProofRecord` rows for `source` (full witness tables, prover-only).
fn source_records(layer: &LayerCommitment, source: ColumnSource) -> &[ProofRecord] {
    match source {
        ColumnSource::Materialized => &layer.materialized,
        ColumnSource::Additions => &layer.additions,
        ColumnSource::Removals => &layer.removals,
    }
}

/// The three root-bound Dory column commitments `(S, P, O)` for `source`. `column_commitments`
/// is `additions (S,P,O)` then `removals (S,P,O)`; `materialized_column_commitments` is the
/// resolved head columns. Verifier-facing (no witness tables), used on both prover and
/// verifier sides for the exact same selection.
pub(super) fn source_column_commitments(
    layer: VerifierCommitmentRef<'_>,
    source: ColumnSource,
) -> [ArkGT; 3] {
    match source {
        ColumnSource::Materialized => *layer.materialized_column_commitments,
        ColumnSource::Additions => [
            layer.column_commitments[0],
            layer.column_commitments[1],
            layer.column_commitments[2],
        ],
        ColumnSource::Removals => [
            layer.column_commitments[3],
            layer.column_commitments[4],
            layer.column_commitments[5],
        ],
    }
}

/// COMMITMENT DIET (Wave 0): a DELTA layer (`parent_state_root == Some`) no longer
/// commits its materialized head view (it is empty and NOT root-bound) and commits only
/// the INCREMENTAL dictionary it introduced, so an inherited constant (e.g. a predicate
/// defined in an ancestor) cannot be resolved locally. A delta layer is therefore NOT
/// directly query-provable at the store level — matching the compiler's base-only gate.
/// Direct delta-head queries reject closed here; the reconciled delta-stack path
/// (Wave 2/3) will re-enable them by authenticating terms across the chain. Roll up or
/// squash a delta stack to a base layer to query it directly today.
pub(crate) fn require_base_layer_for_query(state: &CommitmentState) -> io::Result<()> {
    if state.parent_state_root.is_some() {
        return Err(invalid(
            "delta layer is not directly query-provable after the commitment diet; \
             roll up/squash to a base layer or use the reconciled delta-stack path",
        ));
    }
    Ok(())
}

/// The real (unpadded) row count committed for `source`, used as the real-length pin
/// everywhere the materialized count was. All three counts are bound by `commitment_root`.
pub(super) fn source_count(state: &CommitmentState, source: ColumnSource) -> u64 {
    match source {
        ColumnSource::Materialized => state.materialized_count,
        ColumnSource::Additions => state.addition_count,
        ColumnSource::Removals => state.removal_count,
    }
}

/// Extract the `(S, P, O)` `ArkFr` columns and real-row count `n` for `source`, in the exact
/// deterministic order and power-of-two padding that produced the source's column
/// commitments (identical convention to `materialized_layer_columns`, which is the
/// `Materialized` case).
pub(super) fn source_layer_columns(
    layer: &LayerCommitment,
    source: ColumnSource,
) -> io::Result<(Vec<ArkFr>, Vec<ArkFr>, Vec<ArkFr>, usize)> {
    let records = source_records(layer, source);
    let n = records.len();
    let padded = padded_len(n)?;
    let mut s = vec![f0(); padded];
    let mut p = vec![f0(); padded];
    let mut o = vec![f0(); padded];
    for (i, record) in records.iter().enumerate() {
        s[i] = ArkFr::from_u64(record.subject_id);
        p[i] = ArkFr::from_u64(record.predicate_id);
        o[i] = ArkFr::from_u64(record.object_id);
    }
    Ok((s, p, o, n))
}

/// Resolve a predicate IRI (term bytes) to its dictionary id in this layer. A missing
/// term is authenticated only for a standalone ground query; relational atoms fail closed.
pub(super) fn resolve_predicate_id(layer: &LayerCommitment, predicate_term: &[u8]) -> io::Result<u64> {
    layer
        .predicate_catalog
        .iter()
        .find(|record| record.triple.predicate == predicate_term)
        .map(|record| record.predicate_id)
        .ok_or_else(|| invalid("predicate not in layer dictionary"))
}
