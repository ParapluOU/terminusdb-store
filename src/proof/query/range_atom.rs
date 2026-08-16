//! Adding a numeric value-range test to one BGP atom (e.g. `?a < 100`).
//!
//! A plain atom keeps rows whose predicate/constants match. A RANGED atom additionally
//! keeps only rows whose object VALUE satisfies a numeric comparison. Because the committed
//! object column stores dictionary IDs (not the values themselves), proving a range means:
//! line the object column up with the dictionary's authenticated value columns on one shared
//! power-of-two domain, run the range gadget there, carry the resulting keep/drop mask back,
//! AND it with the ordinary constant-equality mask, and compact the kept columns by the
//! combined mask. [`prove_ranged_atom`] / [`verify_ranged_atom`] are the two halves; the rest
//! are small helpers that pick the shared domain and the kept columns. Everything runs on the
//! caller's shared Fiat–Shamir transcript so the whole atom is one bound statement.

use super::*;

/// The padded power-of-two length of every committed node/value dictionary column,
/// derived from the (root-verified) real dict-row count `n_dict` — the SAME
/// `next_pow2(n_dict).max(2)` convention `commit_node_value_dict*` uses. A ranged atom
/// needs a non-empty dictionary (an empty dict cannot authenticate any object value).
pub(super) fn dict_domain_len(n_dict: usize) -> io::Result<usize> {
    if n_dict == 0 {
        return Err(invalid(
            "layer BGP object range requires a non-empty node/value dictionary",
        ));
    }
    let padded = n_dict
        .checked_next_power_of_two()
        .ok_or_else(|| invalid("node/value dictionary length overflow"))?
        .max(2);
    Ok(padded)
}

/// The full (padded) base column for a kept position — a subset of `{S, O}` for a ranged
/// atom (a ranged object is a variable).
fn kept_base_column(pos: Position, s: &[ArkFr], p: &[ArkFr], o: &[ArkFr]) -> Vec<ArkFr> {
    match pos {
        Position::S => s.to_vec(),
        Position::P => p.to_vec(),
        Position::O => o.to_vec(),
    }
}

/// The root-bound commitment for a kept position (mirrors [`kept_base_column`]).
fn kept_base_commitment(pos: Position, com_s: &ArkGT, com_p: &ArkGT, com_o: &ArkGT) -> ArkGT {
    match pos {
        Position::S => *com_s,
        Position::P => *com_p,
        Position::O => *com_o,
    }
}

/// The common relocation domain for a ranged atom: `D = 2^max(a, dict_log)`, large
/// enough to hold BOTH the table object column (`2^a`) and the dict columns
/// (`2^dict_log`), so both can be relocated in and the range gadget run over one domain.
fn range_common_domain(a: usize, n_dict: usize) -> io::Result<(usize, usize)> {
    let dict_len = dict_domain_len(n_dict)?;
    let dict_log = dict_len.trailing_zeros() as usize;
    Ok((a.max(dict_log), dict_log))
}

/// The root-bound count of rows in the complete node/value dictionary committed at
/// this head. The count and the full catalog are both authenticated by the head root.
fn root_bound_n_dict(source: &impl VerifierLayer) -> io::Result<usize> {
    let layer = source.verifier_view();
    let n_dict = layer.state.node_value_count as usize;
    if n_dict != layer.node_value_catalog.len() {
        return Err(invalid(
            "root-bound node/value dictionary count does not match the committed catalog length",
        ));
    }
    Ok(n_dict)
}

/// Prove one RANGED atom `t(?s, <p>, ?a) [?a op c]`: relocate the object column + the
/// three dict columns to the common domain `D`, prove the authenticated object-range
/// selection at `D`, relocate that `range_sel` back to the table domain, prove the
/// constant-equality selection, multiply them into the combined selection, and compact
/// the kept columns by it. Returns the range subproof, the kept-column commitments (S,O
/// order), the dense kept columns (for the chain witness), and the result length `m`.
/// All sub-proofs run on the SHARED `transcript` in this exact order.
#[allow(clippy::too_many_arguments)]
pub(super) fn prove_ranged_atom(
    context: &StoreProverContext<'_>,
    spec: &AtomFilterSpec,
    range: &ObjectRangeSpec,
    complement_range: bool,
    s: &[ArkFr],
    p: &[ArkFr],
    o: &[ArkFr],
    n: usize,
    a: usize,
    com_s: &ArkGT,
    com_p: &ArkGT,
    com_o: &ArkGT,
    layer: &LayerCommitment,
    transcript: &mut Transcript254,
) -> io::Result<(RangeAtomSubproof, Vec<ArkGT>, Vec<Vec<ArkFr>>, usize)> {
    let n_dict = root_bound_n_dict(layer)?;
    reject_nan_catalog(&layer.node_value_catalog, range.datatype)?;
    let (d_log, dict_log) = range_common_domain(a, n_dict)?;
    let d = 1usize << d_log;

    // Root-bound dict columns/commitments (format v3).
    let (_nv_hash, dict_id) = layer.node_value_dictionary_columns()?;
    let dict_value = layer.node_value_dictionary_value_column()?;
    let dict_datatype = layer.node_value_dictionary_datatype_column()?;
    let com_dict_id = layer.node_value_dict_commitments.1;
    let com_dict_value = layer.node_value_dict_range_commitments.0;
    let com_dict_datatype = layer.node_value_dict_range_commitments.1;
    if dict_id.len() != (1usize << dict_log) {
        return Err(invalid("node/value dictionary column length inconsistent"));
    }

    // (a) Relocate the object column (table -> D) and the three dict columns (dict -> D),
    // each tied to its root-bound original. When a domain already equals D the relocation
    // is same-domain (still authenticates the fresh copy).
    let o_at_d = pad(o, d);
    let com_object_at_d = context.commit_column(&o_at_d)?;
    let relocate_object = prove_column_relocation_with_context(
        context, o, com_o, &o_at_d, &com_object_at_d, transcript,
    )?;

    let dict_id_at_d = pad(&dict_id, d);
    let com_dict_id_at_d = context.commit_column(&dict_id_at_d)?;
    let relocate_dict_id = prove_column_relocation_with_context(
        context, &dict_id,
        &com_dict_id,
        &dict_id_at_d,
        &com_dict_id_at_d,
        transcript,
    )?;

    let dict_value_at_d = pad(&dict_value, d);
    let com_dict_value_at_d = context.commit_column(&dict_value_at_d)?;
    let relocate_dict_value = prove_column_relocation_with_context(
        context, &dict_value,
        &com_dict_value,
        &dict_value_at_d,
        &com_dict_value_at_d,
        transcript,
    )?;

    let dict_datatype_at_d = pad(&dict_datatype, d);
    let com_dict_datatype_at_d = context.commit_column(&dict_datatype_at_d)?;
    let relocate_dict_datatype = prove_column_relocation_with_context(
        context, &dict_datatype,
        &com_dict_datatype,
        &dict_datatype_at_d,
        &com_dict_datatype_at_d,
        transcript,
    )?;

    // (b) Authenticated object-range selection over the relocated columns at D.
    let (object_range, com_range_sel_at_d) = prove_object_range_select_with_context(
        context, &o_at_d,
        &com_object_at_d,
        &dict_id_at_d,
        &dict_value_at_d,
        &dict_datatype_at_d,
        &com_dict_id_at_d,
        &com_dict_value_at_d,
        &com_dict_datatype_at_d,
        n_dict,
        range.datatype,
        range.op,
        range.c,
        range.b_bits,
        n,
        transcript,
    )?;
    // The SAME sel column that Com(range_sel) commits (χ_n-masked, so zero beyond n).
    let range_sel_at_d = object_range_sel_column(
        &o_at_d,
        &dict_id_at_d,
        &dict_value_at_d,
        &dict_datatype_at_d,
        n_dict,
        range.datatype,
        range.op,
        range.c,
        range.b_bits,
        n,
    );
    let (selected_sel_at_d, com_selected_sel_at_d) = if complement_range {
        if range.second.is_some() { return Err(invalid("complemented object range cannot have a second bound")); }
        let selected = range_sel_at_d.iter().enumerate().map(|(index, value)| {
            if index < n { ArkFr::one() - *value } else { ArkFr::zero() }
        }).collect::<Vec<_>>();
        let mut chi = vec![ArkFr::zero(); d]; chi[..n].fill(ArkFr::one());
        let derived = context.commit_column(&chi)? - com_range_sel_at_d;
        if context.commit_column(&selected)? != derived {
            return Err(invalid("literal complement commitment mismatch"));
        }
        (selected, derived)
    } else {
        (range_sel_at_d.clone(), com_range_sel_at_d)
    };

    // (c) Relocate range_sel from D back to the table domain (its support is [0, n) ⊆
    // [0, 2^a), so truncating to the table domain is lossless; the relocation ties the
    // table copy to the authoritative Com(range_sel) at D).
    let table = 1usize << a;
    let range_sel_at_table = selected_sel_at_d[..table].to_vec();
    let com_range_sel_at_table = context.commit_column(&range_sel_at_table)?;
    let relocate_range_sel = prove_column_relocation_with_context(
        context, &selected_sel_at_d,
        &com_selected_sel_at_d,
        &range_sel_at_table,
        &com_range_sel_at_table,
        transcript,
    )?;

    // (c2) A second public bound reuses the already authenticated/relocated object and
    // dictionary columns. Prove its selection independently, relocate it to the table
    // domain, then authenticate the intersection of both masks.
    let (effective_range_sel, com_effective_range_sel, second) =
        if let Some(second_bound) = &range.second {
            let (second_object_range, second_com_at_d) = prove_object_range_select_with_context(
                context, &o_at_d,
                &com_object_at_d,
                &dict_id_at_d,
                &dict_value_at_d,
                &dict_datatype_at_d,
                &com_dict_id_at_d,
                &com_dict_value_at_d,
                &com_dict_datatype_at_d,
                n_dict,
                range.datatype,
                second_bound.op,
                second_bound.c,
                range.b_bits,
                n,
                transcript,
            )?;
            let second_sel_at_d = object_range_sel_column(
                &o_at_d,
                &dict_id_at_d,
                &dict_value_at_d,
                &dict_datatype_at_d,
                n_dict,
                range.datatype,
                second_bound.op,
                second_bound.c,
                range.b_bits,
                n,
            );
            let second_sel_at_table = second_sel_at_d[..table].to_vec();
            let second_com_at_table = context.commit_column(&second_sel_at_table)?;
            let second_relocate = prove_column_relocation_with_context(
                context, &second_sel_at_d,
                &second_com_at_d,
                &second_sel_at_table,
                &second_com_at_table,
                transcript,
            )?;
            let (intersect, com_intersection_sel) = prove_boolean_product_with_context(
                context, &range_sel_at_table,
                &second_sel_at_table,
                &com_range_sel_at_table,
                &second_com_at_table,
                transcript,
            )?;
            let intersection: Vec<ArkFr> = range_sel_at_table
                .iter()
                .zip(second_sel_at_table.iter())
                .map(|(x, y)| *x * *y)
                .collect();
            (
                intersection,
                com_intersection_sel,
                Some(SecondRangeSubproof {
                    object_range: second_object_range,
                    com_range_sel_at_d: second_com_at_d,
                    relocate_range_sel: second_relocate,
                    com_range_sel_at_table: second_com_at_table,
                    intersect,
                    com_intersection_sel,
                }),
            )
        } else {
            (range_sel_at_table.clone(), com_range_sel_at_table, None)
        };

    // (d) Constant-equality selection (predicate + any subject constant) at the table
    // domain — pins com_equality_sel to the equality indicator (never touches the object).
    let (equality, com_equality_sel, eq_sel) =
        prove_selection_only_with_context(
            context, spec, s, p, o, n, com_s, com_p, com_o, transcript,
        )?;

    // (e) Combined selection = equality_sel ⊙ range_sel (both boolean + χ_n-masked).
    let (combine, com_combined_sel) = prove_boolean_product_with_context(
        context, &eq_sel,
        &effective_range_sel,
        &com_equality_sel,
        &com_effective_range_sel,
        transcript,
    )?;
    let combined_sel: Vec<ArkFr> = eq_sel
        .iter()
        .zip(effective_range_sel.iter())
        .map(|(x, y)| *x * *y)
        .collect();

    // (f) Compact the kept variable columns by the combined selection.
    let sources: Vec<Vec<ArkFr>> = spec
        .kept
        .iter()
        .map(|pos| kept_base_column(*pos, s, p, o))
        .collect();
    let source_coms: Vec<ArkGT> = spec
        .kept
        .iter()
        .map(|pos| kept_base_commitment(*pos, com_s, com_p, com_o))
        .collect();
    let (compaction, kept_commitments, kept_columns, m) = prove_compaction_only_with_context(
        context, &sources,
        &source_coms,
        &combined_sel,
        &com_combined_sel,
        transcript,
    )?;

    Ok((
        RangeAtomSubproof {
            relocate_object,
            relocate_dict_id,
            relocate_dict_value,
            relocate_dict_datatype,
            com_object_at_d,
            com_dict_id_at_d,
            com_dict_value_at_d,
            com_dict_datatype_at_d,
            object_range,
            com_range_sel_at_d,
            relocate_range_sel,
            com_range_sel_at_table,
            second,
            equality,
            com_equality_sel,
            combine,
            com_combined_sel,
            compaction,
        },
        kept_commitments,
        kept_columns,
        m,
    ))
}

/// Verify one RANGED atom's sub-proofs, mirroring [`prove_ranged_atom`] on the SHARED
/// `transcript` in prover order: the four relocations to `D`, the authenticated
/// object-range selection at `D`, the range selection relocated back to the table
/// domain, the constant-equality selection, the combined-selection product, and the
/// compaction of the kept columns into `kept_commitments[0..m]`. Anchoring/root checks
/// happen once in the caller BEFORE this runs.
#[allow(clippy::too_many_arguments)]
pub(super) fn verify_ranged_atom(
    spec: &AtomFilterSpec,
    range: &ObjectRangeSpec,
    complement_range: bool,
    n: usize,
    a: usize,
    com_s: &ArkGT,
    com_p: &ArkGT,
    com_o: &ArkGT,
    layer: VerifierCommitmentRef<'_>,
    sub: &RangeAtomSubproof,
    kept_commitments: &[ArkGT],
    m: usize,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    let n_dict = root_bound_n_dict(&layer)?;
    reject_nan_catalog(layer.node_value_catalog, range.datatype)?;
    let (d_log, dict_log) = range_common_domain(a, n_dict)?;
    let com_dict_id = layer.node_value_dict_commitments.1;
    let com_dict_value = layer.node_value_dict_range_commitments.0;
    let com_dict_datatype = layer.node_value_dict_range_commitments.1;

    // Explicit sumcheck-dimension checks on the ranged sub-proofs, mirroring the
    // plain-atom path's `filter.sumcheck.round_polys.len() == a` check (defense-in-depth;
    // the Dory openings already pin these transitively). The object-range gadget runs
    // entirely at the common domain `D` (its lookup/datatype-mask/range/combine sumchecks
    // each have `d_log` variables); the constant-equality selection, the combined-selection
    // product, and the compaction all run at the table domain (`a` variables).
    // The LogUp-GKR lookup / compaction carry `2·domain` leaves, so their fraction tree has
    // `domain_log + 1` layers (`num_vars()`); the other sub-proofs remain plain sumchecks.
    for (rounds, expected) in [
        (sub.object_range.lookup.gkr.num_vars(), d_log + 1),
        (sub.object_range.datatype.sumcheck.round_polys.len(), d_log),
        (sub.object_range.range.sumcheck.round_polys.len(), d_log),
        (sub.object_range.combine.sumcheck.round_polys.len(), d_log),
        (sub.equality.sumcheck.round_polys.len(), a),
        (sub.combine.sumcheck.round_polys.len(), a),
        (sub.compaction.link.num_vars(), a + 1),
    ] {
        if rounds != expected {
            return Err(invalid(
                "ranged atom sub-proof dimension does not match the layer's committed domains",
            ));
        }
    }
    if let Some(second) = &sub.second {
        for (rounds, expected) in [
            (second.object_range.lookup.gkr.num_vars(), d_log + 1),
            (second.object_range.datatype.sumcheck.round_polys.len(), d_log),
            (second.object_range.range.sumcheck.round_polys.len(), d_log),
            (second.object_range.combine.sumcheck.round_polys.len(), d_log),
        ] {
            if rounds != expected {
                return Err(invalid(
                    "second range sub-proof dimension does not match the dictionary domain",
                ));
            }
        }
        if second.intersect.sumcheck.round_polys.len() != a {
            return Err(invalid(
                "range-intersection dimension does not match the table domain",
            ));
        }
    }

    // (a) Verify the four relocations to the common domain D against the root-bound
    // originals (com_o + the three dict commitments).
    verify_column_relocation(
        com_o,
        a,
        &sub.com_object_at_d,
        d_log,
        &sub.relocate_object,
        transcript,
    )?;
    verify_column_relocation(
        &com_dict_id,
        dict_log,
        &sub.com_dict_id_at_d,
        d_log,
        &sub.relocate_dict_id,
        transcript,
    )?;
    verify_column_relocation(
        &com_dict_value,
        dict_log,
        &sub.com_dict_value_at_d,
        d_log,
        &sub.relocate_dict_value,
        transcript,
    )?;
    verify_column_relocation(
        &com_dict_datatype,
        dict_log,
        &sub.com_dict_datatype_at_d,
        d_log,
        &sub.relocate_dict_datatype,
        transcript,
    )?;

    // (b) Verify the authenticated object-range selection at D.
    verify_object_range_select(
        &sub.com_object_at_d,
        &sub.com_dict_id_at_d,
        &sub.com_dict_value_at_d,
        &sub.com_dict_datatype_at_d,
        n_dict,
        range.datatype,
        range.op,
        range.c,
        range.b_bits,
        n,
        &sub.com_range_sel_at_d,
        &sub.object_range,
        transcript,
    )?;

    // (c) Verify range_sel relocated from D back to the table domain.
    let com_selected_sel_at_d = if complement_range {
        if range.second.is_some() { return Err(invalid("complemented object range cannot have a second bound")); }
        let mut chi = vec![ArkFr::zero(); 1usize << d_log]; chi[..n].fill(ArkFr::one());
        commit_column(&chi)? - sub.com_range_sel_at_d
    } else {
        sub.com_range_sel_at_d
    };
    verify_column_relocation(
        &com_selected_sel_at_d,
        d_log,
        &sub.com_range_sel_at_table,
        a,
        &sub.relocate_range_sel,
        transcript,
    )?;

    // (c2) A public second bound requires exactly one matching subproof. Its selection
    // is authenticated against the same relocated columns and intersected with the first.
    let com_effective_range_sel = match (&range.second, &sub.second) {
        (None, None) => sub.com_range_sel_at_table,
        (Some(bound), Some(second)) => {
            verify_object_range_select(
                &sub.com_object_at_d,
                &sub.com_dict_id_at_d,
                &sub.com_dict_value_at_d,
                &sub.com_dict_datatype_at_d,
                n_dict,
                range.datatype,
                bound.op,
                bound.c,
                range.b_bits,
                n,
                &second.com_range_sel_at_d,
                &second.object_range,
                transcript,
            )?;
            verify_column_relocation(
                &second.com_range_sel_at_d,
                d_log,
                &second.com_range_sel_at_table,
                a,
                &second.relocate_range_sel,
                transcript,
            )?;
            verify_boolean_product(
                &sub.com_range_sel_at_table,
                &second.com_range_sel_at_table,
                &second.com_intersection_sel,
                &second.intersect,
                transcript,
            )?;
            second.com_intersection_sel
        }
        _ => {
            return Err(invalid(
                "second range subproof presence does not match the public plan",
            ))
        }
    };

    // (d) Verify the constant-equality selection.
    verify_selection_only(
        spec,
        n,
        com_s,
        com_p,
        com_o,
        &sub.com_equality_sel,
        &sub.equality,
        transcript,
    )?;

    // (e) Verify the combined selection = equality_sel · range_sel.
    verify_boolean_product(
        &sub.com_equality_sel,
        &com_effective_range_sel,
        &sub.com_combined_sel,
        &sub.combine,
        transcript,
    )?;

    // (f) Verify the compaction of the kept columns by the combined selection.
    let source_coms: Vec<ArkGT> = spec
        .kept
        .iter()
        .map(|pos| kept_base_commitment(*pos, com_s, com_p, com_o))
        .collect();
    verify_compaction_only(
        &source_coms,
        &sub.com_combined_sel,
        kept_commitments,
        m,
        &sub.compaction,
        transcript,
    )
}
