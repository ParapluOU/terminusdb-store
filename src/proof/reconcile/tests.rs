//! Unit tests for delta-stack reconciliation and the `present`/delta read proofs (multiset
//! equality, chain binding, single/two-atom/BGP reads, delta reads, and their soundness
//! negatives). These are the `proof::reconcile` suite that exercises the reconcile split.

use super::*;
use crate::layer::ValueTriple;
use crate::proof::range::RangeOp;
use crate::store::sync::open_sync_memory_store;
use std::sync::Arc;

fn iri(s: &str) -> String {
    format!("http://ex/{s}")
}

fn commit_col(v: &[ArkFr]) -> ArkGT {
    crate::proof::argument::commit_column(v).unwrap()
}

/// The materialized ground-truth present set as canonical (S,P,O) hash rows.
fn materialized_rows(layers: &[&LayerCommitment]) -> Vec<(ArkFr, ArkFr, ArkFr)> {
    reconcile_present(layers)
        .unwrap()
        .iter()
        .map(|r| (subject_hash(r), predicate_hash(r), object_hash(r)))
        .collect()
}

fn bag_eq(mut a: Vec<(ArkFr, ArkFr, ArkFr)>, mut b: Vec<(ArkFr, ArkFr, ArkFr)>) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let key = |t: &(ArkFr, ArkFr, ArkFr)| (format!("{:?}", t.0), format!("{:?}", t.1), format!("{:?}", t.2));
    a.sort_by_key(&key);
    b.sort_by_key(&key);
    a == b
}

fn refs(layers: &[Arc<LayerCommitment>]) -> Vec<&LayerCommitment> {
    layers.iter().map(|l| l.as_ref()).collect()
}

// --- Base + 1 delta (skeleton case must not regress) ---
fn base_plus_one() -> Vec<Arc<LayerCommitment>> {
    let store = open_sync_memory_store();
    let builder = store.create_base_layer().unwrap();
    for (s, o) in [("a1", "b1"), ("a2", "b2"), ("a3", "b3"), ("a4", "b4")] {
        builder.add_value_triple(ValueTriple::new_node(&iri(s), &iri("p1"), &iri(o))).unwrap();
    }
    let base = builder.commit().unwrap();
    let base_lc = base.proof_commitment().unwrap();
    let child = base.open_write().unwrap();
    for (s, o) in [("a1", "b1"), ("a2", "b2")] {
        child.remove_value_triple(ValueTriple::new_node(&iri(s), &iri("p1"), &iri(o))).unwrap();
    }
    for (s, o) in [("a3", "b5"), ("a5", "b6")] {
        child.add_value_triple(ValueTriple::new_node(&iri(s), &iri("p1"), &iri(o))).unwrap();
    }
    let head = child.commit().unwrap();
    vec![base_lc, head.proof_commitment().unwrap()]
}

#[test]
fn base_plus_one_reconciles_and_counts() {
    let layers = base_plus_one();
    let lref = refs(&layers);
    let (proof, present) = prove_reconcile(&lref).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    let m = verify_reconcile(&lref, head_root, &proof).unwrap();
    let ground = materialized_rows(&lref);
    assert_eq!(m, ground.len());
    assert_eq!(m, 4);
    assert!(bag_eq(present.rows.clone(), ground));
}

// --- Base + 3 deltas: adds AND removes across layers, add-then-remove,
//     remove-of-inherited, and a delta with a BRAND-NEW predicate. ---
fn base_plus_three() -> Vec<Arc<LayerCommitment>> {
    let store = open_sync_memory_store();
    let builder = store.create_base_layer().unwrap();
    // Base: a1..a5 -> p1 -> b1..b5
    for i in 1..=5 {
        builder
            .add_value_triple(ValueTriple::new_node(&iri(&format!("a{i}")), &iri("p1"), &iri(&format!("b{i}"))))
            .unwrap();
    }
    let base = builder.commit().unwrap();
    let base_lc = base.proof_commitment().unwrap();

    // D1: remove {a1->b1, a2->b2} (inherited); add {a6->b6 (new terms), a3->b7 (b7 new)}.
    let d1 = base.open_write().unwrap();
    d1.remove_value_triple(ValueTriple::new_node(&iri("a1"), &iri("p1"), &iri("b1"))).unwrap();
    d1.remove_value_triple(ValueTriple::new_node(&iri("a2"), &iri("p1"), &iri("b2"))).unwrap();
    d1.add_value_triple(ValueTriple::new_node(&iri("a6"), &iri("p1"), &iri("b6"))).unwrap();
    d1.add_value_triple(ValueTriple::new_node(&iri("a3"), &iri("p1"), &iri("b7"))).unwrap();
    let d1c = d1.commit().unwrap();
    let d1_lc = d1c.proof_commitment().unwrap();

    // D2: add-then-remove — add {a7->b8, a8->b9}, then D3 removes a7->b8.
    //     Also uses a BRAND-NEW predicate p2 for {a7->b8}.
    let d2 = d1c.open_write().unwrap();
    d2.add_value_triple(ValueTriple::new_node(&iri("a7"), &iri("p2"), &iri("b8"))).unwrap();
    d2.add_value_triple(ValueTriple::new_node(&iri("a8"), &iri("p1"), &iri("b9"))).unwrap();
    let d2c = d2.commit().unwrap();
    let d2_lc = d2c.proof_commitment().unwrap();

    // D3: remove {a7->b8 (added in D2, p2), a3->b3 (inherited from base)}; add {a9->b10, a1->b1 (re-add removed base terms)}.
    let d3 = d2c.open_write().unwrap();
    d3.remove_value_triple(ValueTriple::new_node(&iri("a7"), &iri("p2"), &iri("b8"))).unwrap();
    d3.remove_value_triple(ValueTriple::new_node(&iri("a3"), &iri("p1"), &iri("b3"))).unwrap();
    d3.add_value_triple(ValueTriple::new_node(&iri("a9"), &iri("p1"), &iri("b10"))).unwrap();
    d3.add_value_triple(ValueTriple::new_node(&iri("a1"), &iri("p1"), &iri("b1"))).unwrap();
    let d3c = d3.commit().unwrap();
    let d3_lc = d3c.proof_commitment().unwrap();

    vec![base_lc, d1_lc, d2_lc, d3_lc]
}

#[test]
fn n_layer_reconciles_and_counts() {
    let layers = base_plus_three();
    let lref = refs(&layers);
    let (proof, present) = prove_reconcile(&lref).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    let m = verify_reconcile(&lref, head_root, &proof).unwrap();
    let ground = materialized_rows(&lref);
    assert_eq!(m, ground.len(), "reconciled count must equal materialized count");
    assert!(bag_eq(present.rows.clone(), ground), "present bag must equal materialized bag");
    // Sanity: the new predicate p2's triple a7->b8 was added (D2) and removed (D3),
    // so it is NOT present; a1->b1 was removed (D1) then re-added (D3) → present.
}

#[test]
fn new_predicate_delta_reconciles() {
    // A base with one predicate, then a delta introducing a brand-new predicate.
    let store = open_sync_memory_store();
    let builder = store.create_base_layer().unwrap();
    for (s, o) in [("a1", "b1"), ("a2", "b2"), ("a3", "b3")] {
        builder.add_value_triple(ValueTriple::new_node(&iri(s), &iri("p1"), &iri(o))).unwrap();
    }
    let base = builder.commit().unwrap();
    let base_lc = base.proof_commitment().unwrap();
    let child = base.open_write().unwrap();
    // New predicate p2 on two triples (>= 2 for the group domain), reusing inherited subjects.
    child.add_value_triple(ValueTriple::new_node(&iri("a1"), &iri("p2"), &iri("b1"))).unwrap();
    child.add_value_triple(ValueTriple::new_node(&iri("a2"), &iri("p2"), &iri("b2"))).unwrap();
    let head = child.commit().unwrap();
    let head_lc = head.proof_commitment().unwrap();
    assert!(head_lc.state.predicate_count > head_lc.state.parent_predicate_count, "delta must add a predicate");

    let layers = vec![base_lc, head_lc];
    let lref = refs(&layers);
    let (proof, present) = prove_reconcile(&lref).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    let m = verify_reconcile(&lref, head_root, &proof).unwrap();
    let ground = materialized_rows(&lref);
    assert_eq!(m, ground.len());
    assert_eq!(m, 5); // 3 base + 2 new-predicate additions, no removals
    assert!(bag_eq(present.rows.clone(), ground));
}

// --- Soundness negatives (generalized to the chain) ---

#[test]
fn wrong_or_swapped_head_root_rejects() {
    let layers = base_plus_three();
    let lref = refs(&layers);
    let (proof, _) = prove_reconcile(&lref).unwrap();
    // Anchor to a non-head layer root.
    assert!(verify_reconcile(&lref, lref[0].resolution_root(), &proof).is_err());
    let mut bogus = lref.last().unwrap().resolution_root();
    bogus[0] ^= 0xff;
    assert!(verify_reconcile(&lref, bogus, &proof).is_err());
}

#[test]
fn dropped_middle_layer_rejects() {
    let layers = base_plus_three();
    let lref = refs(&layers);
    let (proof, _) = prove_reconcile(&lref).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    // Drop D2 (index 2): the parent-state-root chain no longer links D3 to D1.
    let dropped = vec![lref[0], lref[1], lref[3]];
    assert!(verify_reconcile(&dropped, head_root, &proof).is_err());
}

#[test]
fn reordered_middle_layers_reject() {
    let layers = base_plus_three();
    let lref = refs(&layers);
    let (proof, _) = prove_reconcile(&lref).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    // Swap D1 and D2: the chain breaks.
    let reordered = vec![lref[0], lref[2], lref[1], lref[3]];
    assert!(verify_reconcile(&reordered, head_root, &proof).is_err());
}

#[test]
fn spliced_foreign_layer_rejects() {
    let layers = base_plus_three();
    let other = base_plus_one();
    let lref = refs(&layers);
    let oref = refs(&other);
    let (proof, _) = prove_reconcile(&lref).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    // Replace D2 with a foreign delta from an unrelated stack.
    let spliced = vec![lref[0], lref[1], oref[1], lref[3]];
    assert!(verify_reconcile(&spliced, head_root, &proof).is_err());
}

#[test]
fn tampered_present_row_rejects() {
    let layers = base_plus_three();
    let lref = refs(&layers);
    let (mut proof, _) = prove_reconcile(&lref).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    proof.com_present_s = commit_col(&vec![ark(999); 1usize << proof.working_log]);
    assert!(verify_reconcile(&lref, head_root, &proof).is_err());
}

#[test]
fn hidden_removal_in_any_layer_rejects() {
    // Tamper a removal group's authenticated hash column (misrepresenting the
    // root-bound removal set of some layer): its lookup binding breaks.
    let layers = base_plus_three();
    let lref = refs(&layers);
    let (mut proof, _) = prove_reconcile(&lref).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    // Find a removal group (sign −1) and corrupt its subject hash lookup.
    let column_commitments: Vec<&[ArkGT; 6]> = lref.iter().map(|l| &l.column_commitments).collect();
    let adds: Vec<usize> = lref.iter().map(|l| l.additions.len()).collect();
    let rems: Vec<usize> = lref.iter().map(|l| l.removals.len()).collect();
    let descs = group_descs(&column_commitments, &adds, &rems).unwrap();
    let rem_idx = descs.iter().position(|d| d.sign == f0() - f1()).expect("a removal group exists");
    proof.groups[rem_idx].s.lookup.com_v_hash = commit_col(&vec![ark(13); 1usize << proof.working_log]);
    assert!(verify_reconcile(&lref, head_root, &proof).is_err());
}

#[test]
fn injected_removal_count_rejects() {
    // Claiming one fewer live row (present length) than the delta counts imply.
    let layers = base_plus_three();
    let lref = refs(&layers);
    let (proof, _) = prove_reconcile(&lref).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    let mut forged = proof.clone();
    forged.present_len -= 1;
    assert!(verify_reconcile(&lref, head_root, &forged).is_err());
}

#[test]
fn tampered_combined_dict_rejects() {
    let layers = base_plus_three();
    let lref = refs(&layers);
    let (mut proof, _) = prove_reconcile(&lref).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    proof.combined_nv.com_hash = commit_col(&vec![ark(7); 1usize << proof.combined_nv.log]);
    assert!(verify_reconcile(&lref, head_root, &proof).is_err());
}

// ============================================================
// Wave 3 — single-atom BGP (filter) over present.
// ============================================================

fn hash_pred(iri: &str) -> ArkFr {
    hash_term(iri.as_bytes())
}
fn hash_node(iri: &str) -> ArkFr {
    hash_object(&CanonicalObject::Node(iri.as_bytes().to_vec()))
}

/// Base+delta with MIXED predicates so a predicate filter selects a proper subset.
/// Base: {a1 p1 b1, a2 p1 b2, a3 p2 c3, a4 p2 c4, a7 p1 b7, a8 p2 c8}.
/// Delta: remove {a1 p1 b1, a7 p1 b7}; add {a5 p1 b5, a6 p2 c6}.
/// present = {a2 p1 b2, a3 p2 c3, a4 p2 c4, a8 p2 c8, a5 p1 b5, a6 p2 c6}.
fn filter_stack() -> Vec<Arc<LayerCommitment>> {
    let store = open_sync_memory_store();
    let builder = store.create_base_layer().unwrap();
    for (s, p, o) in [
        ("a1", "p1", "b1"), ("a2", "p1", "b2"), ("a3", "p2", "c3"),
        ("a4", "p2", "c4"), ("a7", "p1", "b7"), ("a8", "p2", "c8"),
    ] {
        builder.add_value_triple(ValueTriple::new_node(&iri(s), &iri(p), &iri(o))).unwrap();
    }
    let base = builder.commit().unwrap();
    let base_lc = base.proof_commitment().unwrap();
    let child = base.open_write().unwrap();
    child.remove_value_triple(ValueTriple::new_node(&iri("a1"), &iri("p1"), &iri("b1"))).unwrap();
    child.remove_value_triple(ValueTriple::new_node(&iri("a7"), &iri("p1"), &iri("b7"))).unwrap();
    child.add_value_triple(ValueTriple::new_node(&iri("a5"), &iri("p1"), &iri("b5"))).unwrap();
    child.add_value_triple(ValueTriple::new_node(&iri("a6"), &iri("p2"), &iri("c6"))).unwrap();
    let head = child.commit().unwrap();
    vec![base_lc, head.proof_commitment().unwrap()]
}

/// Materialized single-atom filter: present rows whose `position` hashes to `constant`,
/// output the other two columns (matching `filter_selection`'s output order).
fn materialized_filter(layers: &[&LayerCommitment], position: FilterPosition, constant: ArkFr) -> Vec<(ArkFr, ArkFr)> {
    reconcile_present(layers)
        .unwrap()
        .iter()
        .filter_map(|r| {
            let (sh, ph, oh) = (subject_hash(r), predicate_hash(r), object_hash(r));
            let (key, out) = match position {
                FilterPosition::Predicate => (ph, (sh, oh)),
                FilterPosition::Subject => (sh, (ph, oh)),
                FilterPosition::Object => (oh, (sh, ph)),
            };
            if key == constant { Some(out) } else { None }
        })
        .collect()
}

fn out_pairs(result: &PresentReadResult) -> Vec<(ArkFr, ArkFr)> {
    (0..result.len).map(|i| (result.columns[0][i], result.columns[1][i])).collect()
}

fn bag_eq_pairs(mut a: Vec<(ArkFr, ArkFr)>, mut b: Vec<(ArkFr, ArkFr)>) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let key = |t: &(ArkFr, ArkFr)| (format!("{:?}", t.0), format!("{:?}", t.1));
    a.sort_by_key(&key);
    b.sort_by_key(&key);
    a == b
}

fn run_filter(position: FilterPosition, constant: ArkFr, expect: usize) {
    let layers = filter_stack();
    let lref = refs(&layers);
    let (rec, filt, result) = prove_read_single_atom(&lref, position, constant).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    let m = verify_read_single_atom(&lref, head_root, position, constant, &rec, &filt).unwrap();
    let ground = materialized_filter(&lref, position, constant);
    assert_eq!(m, ground.len(), "count must equal materialized filter");
    assert_eq!(m, expect);
    assert!(bag_eq_pairs(out_pairs(&result), ground), "result bag must equal materialized filter");
}

#[test]
fn single_atom_predicate_constant() {
    // Triple(?s, p1, ?o): present has 2 p1 rows (a2->b2, a5->b5) and 4 p2 rows.
    run_filter(FilterPosition::Predicate, hash_pred("http://ex/p1"), 2);
    run_filter(FilterPosition::Predicate, hash_pred("http://ex/p2"), 4);
    // A predicate absent from present yields the empty answer.
    run_filter(FilterPosition::Predicate, hash_pred("http://ex/pX"), 0);
}

#[test]
fn single_atom_subject_constant() {
    // Triple(a2, ?p, ?o): exactly {a2 p1 b2}.
    run_filter(FilterPosition::Subject, hash_node("http://ex/a2"), 1);
    // A removed subject (a1) is absent.
    run_filter(FilterPosition::Subject, hash_node("http://ex/a1"), 0);
}

#[test]
fn single_atom_object_constant() {
    // Triple(?s, ?p, c3): exactly {a3 p2 c3}.
    run_filter(FilterPosition::Object, hash_node("http://ex/c3"), 1);
}

#[test]
fn filter_fabricated_result_row_rejects() {
    let layers = filter_stack();
    let lref = refs(&layers);
    let c = hash_pred("http://ex/p1");
    let (rec, mut filt, _) = prove_read_single_atom(&lref, FilterPosition::Predicate, c).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    // Replace an output column commitment with a fabricated one: the compaction binding fails.
    filt.com_out[0] = commit_col(&vec![ark(4242); 1usize << rec.working_log]);
    assert!(verify_read_single_atom(&lref, head_root, FilterPosition::Predicate, c, &rec, &filt).is_err());
}

#[test]
fn filter_dropped_row_rejects() {
    let layers = filter_stack();
    let lref = refs(&layers);
    let c = hash_pred("http://ex/p2"); // 4 matching rows
    let (rec, filt, _) = prove_read_single_atom(&lref, FilterPosition::Predicate, c).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    let mut forged = filt.clone();
    forged.out_len -= 1; // claim one fewer matching row
    assert!(verify_read_single_atom(&lref, head_root, FilterPosition::Predicate, c, &rec, &forged).is_err());
}

#[test]
fn filter_wrong_constant_rejects() {
    let layers = filter_stack();
    let lref = refs(&layers);
    let c = hash_pred("http://ex/p1");
    let (rec, filt, _) = prove_read_single_atom(&lref, FilterPosition::Predicate, c).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    // Verifying the p1 proof against the p2 constant must reject (selection is p1-bound).
    let wrong = hash_pred("http://ex/p2");
    assert!(verify_read_single_atom(&lref, head_root, FilterPosition::Predicate, wrong, &rec, &filt).is_err());
}

// ============================================================
// Wave 3 — two-atom JOIN over present.
// ============================================================

/// A stack whose present supports Triple(?x, p1, ?y) & Triple(?y, p2, ?z):
/// present p1 = {x1->y1, x2->y1, x5->y1}, p2 = {y1->z1, y1->z2, y1->z5}
/// (base adds two non-joining rows x3->p1->y3, y4->p2->z4 which the delta removes).
/// Join on ?y = y1 => 3 x-bindings × 3 z-bindings = 9 solutions.
fn join_stack() -> Vec<Arc<LayerCommitment>> {
    let store = open_sync_memory_store();
    let b = store.create_base_layer().unwrap();
    for (s, p, o) in [
        ("x1", "p1", "y1"), ("x2", "p1", "y1"), ("y1", "p2", "z1"),
        ("y1", "p2", "z2"), ("x3", "p1", "y3"), ("y4", "p2", "z4"),
    ] {
        b.add_value_triple(ValueTriple::new_node(&iri(s), &iri(p), &iri(o))).unwrap();
    }
    let base = b.commit().unwrap();
    let base_lc = base.proof_commitment().unwrap();
    let child = base.open_write().unwrap();
    child.remove_value_triple(ValueTriple::new_node(&iri("x3"), &iri("p1"), &iri("y3"))).unwrap();
    child.remove_value_triple(ValueTriple::new_node(&iri("y4"), &iri("p2"), &iri("z4"))).unwrap();
    child.add_value_triple(ValueTriple::new_node(&iri("x5"), &iri("p1"), &iri("y1"))).unwrap();
    child.add_value_triple(ValueTriple::new_node(&iri("y1"), &iri("p2"), &iri("z5"))).unwrap();
    let head = child.commit().unwrap();
    vec![base_lc, head.proof_commitment().unwrap()]
}

/// Materialized 2-atom join over present, in COMBINED-DICT id space (matching the proof's
/// `[id_x, id_y, id_z]` output).
fn materialized_join(layers: &[&LayerCommitment], p: &str, q: &str) -> Vec<(ArkFr, ArkFr, ArkFr)> {
    let present = reconcile_present(layers).unwrap();
    let mut out = Vec::new();
    for a in present.iter().filter(|r| r.triple.predicate == p.as_bytes()) {
        for c in present.iter().filter(|r| r.triple.predicate == q.as_bytes()) {
            if a.object_id == c.subject_id {
                out.push((ark(a.subject_id), ark(a.object_id), ark(c.object_id)));
            }
        }
    }
    out
}

fn join_out(result: &PresentReadResult) -> Vec<(ArkFr, ArkFr, ArkFr)> {
    (0..result.len)
        .map(|i| (result.columns[0][i], result.columns[1][i], result.columns[2][i]))
        .collect()
}

#[test]
fn two_atom_join_over_present() {
    let layers = join_stack();
    let lref = refs(&layers);
    let (cp, cq) = (hash_pred("http://ex/p1"), hash_pred("http://ex/p2"));
    let (rec, join, result) = prove_read_two_atom_join(&lref, cp, cq).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    let n = verify_read_two_atom_join(&lref, head_root, cp, cq, &rec, &join).unwrap();
    let ground = materialized_join(&lref, "http://ex/p1", "http://ex/p2");
    assert_eq!(n, ground.len(), "join count must equal materialized join");
    assert_eq!(n, 9);
    assert!(bag_eq(join_out(&result), ground), "join result bag must equal materialized join");
}

#[test]
fn join_fabricated_row_rejects() {
    let layers = join_stack();
    let lref = refs(&layers);
    let (cp, cq) = (hash_pred("http://ex/p1"), hash_pred("http://ex/p2"));
    let (rec, mut join, _) = prove_read_two_atom_join(&lref, cp, cq).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    // Fabricate an output column: the inner-join matched-projection binding fails.
    join.com_join_out[0] = commit_col(&vec![ark(777); 1usize << join.join_log]);
    assert!(verify_read_two_atom_join(&lref, head_root, cp, cq, &rec, &join).is_err());
}

#[test]
fn join_dropped_row_rejects() {
    let layers = join_stack();
    let lref = refs(&layers);
    let (cp, cq) = (hash_pred("http://ex/p1"), hash_pred("http://ex/p2"));
    let (rec, join, _) = prove_read_two_atom_join(&lref, cp, cq).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    let mut forged = join.clone();
    forged.join_out_len -= 1; // claim one fewer solution
    assert!(verify_read_two_atom_join(&lref, head_root, cp, cq, &rec, &forged).is_err());
}

#[test]
fn join_wrong_predicate_rejects() {
    let layers = join_stack();
    let lref = refs(&layers);
    let (cp, cq) = (hash_pred("http://ex/p1"), hash_pred("http://ex/p2"));
    let (rec, join, _) = prove_read_two_atom_join(&lref, cp, cq).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    // Verify against a wrong second predicate: the second atom's selection is p2-bound.
    let wrong = hash_pred("http://ex/pX");
    assert!(verify_read_two_atom_join(&lref, head_root, cp, wrong, &rec, &join).is_err());
}

#[test]
fn join_tampered_present_id_rejects() {
    // A prover who feeds a wrong combined-dict id for present (to fabricate a joining
    // pair) fails the id → hash authentication against the reconciled hashes.
    let layers = join_stack();
    let lref = refs(&layers);
    let (cp, cq) = (hash_pred("http://ex/p1"), hash_pred("http://ex/p2"));
    let (rec, mut join, _) = prove_read_two_atom_join(&lref, cp, cq).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    join.auth_s.com_id = commit_col(&vec![ark(3); 1usize << rec.working_log]);
    assert!(verify_read_two_atom_join(&lref, head_root, cp, cq, &rec, &join).is_err());
}

// ============================================================
// Wave 4 — result resolution to CanonicalObjects.
// ============================================================

fn rows_of(cols: &[Vec<CanonicalObject>]) -> Vec<Vec<CanonicalObject>> {
    if cols.is_empty() {
        return vec![];
    }
    let n = cols[0].len();
    (0..n).map(|i| cols.iter().map(|c| c[i].clone()).collect()).collect()
}

fn bag_eq_rows(mut a: Vec<Vec<CanonicalObject>>, mut b: Vec<Vec<CanonicalObject>>) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let key = |r: &Vec<CanonicalObject>| format!("{r:?}");
    a.sort_by_key(&key);
    b.sort_by_key(&key);
    a == b
}

/// Materialized single-atom filter, RESOLVED to CanonicalObjects (matching `filter_output_kinds`).
fn materialized_filter_resolved(
    layers: &[&LayerCommitment],
    position: FilterPosition,
    constant: ArkFr,
) -> Vec<Vec<CanonicalObject>> {
    reconcile_present(layers)
        .unwrap()
        .iter()
        .filter_map(|r| {
            let (sh, ph, oh) = (subject_hash(r), predicate_hash(r), object_hash(r));
            let node_s = CanonicalObject::Node(r.triple.subject.clone());
            let node_p = CanonicalObject::Node(r.triple.predicate.clone());
            let obj = r.triple.object.clone();
            let (key, out) = match position {
                FilterPosition::Predicate => (ph, vec![node_s, obj]),
                FilterPosition::Subject => (sh, vec![node_p, obj]),
                FilterPosition::Object => (oh, vec![node_s, node_p]),
            };
            if key == constant { Some(out) } else { None }
        })
        .collect()
}

fn materialized_join_resolved(layers: &[&LayerCommitment], p: &str, q: &str) -> Vec<Vec<CanonicalObject>> {
    let present = reconcile_present(layers).unwrap();
    let mut out = Vec::new();
    for a in present.iter().filter(|r| r.triple.predicate == p.as_bytes()) {
        for c in present.iter().filter(|r| r.triple.predicate == q.as_bytes()) {
            if a.object_id == c.subject_id {
                out.push(vec![
                    CanonicalObject::Node(a.triple.subject.clone()),
                    a.triple.object.clone(),
                    c.triple.object.clone(),
                ]);
            }
        }
    }
    out
}

#[test]
fn single_atom_predicate_resolved() {
    let layers = filter_stack();
    let lref = refs(&layers);
    let c = hash_pred("http://ex/p2"); // 4 rows
    let (rec, filt, result) = prove_read_single_atom(&lref, FilterPosition::Predicate, c).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    let resolved = verify_and_resolve_single_atom(
        &lref, head_root, FilterPosition::Predicate, c, &rec, &filt, &result.columns, &result.id_hints,
    )
    .unwrap();
    let ground = materialized_filter_resolved(&lref, FilterPosition::Predicate, c);
    assert!(bag_eq_rows(rows_of(&resolved), ground), "resolved filter must equal materialized");
    // Real IRIs, not hashes.
    assert!(resolved[0].iter().all(|o| matches!(o, CanonicalObject::Node(_))));
}

#[test]
fn two_atom_join_resolved() {
    let layers = join_stack();
    let lref = refs(&layers);
    let (cp, cq) = (hash_pred("http://ex/p1"), hash_pred("http://ex/p2"));
    let (rec, join, result) = prove_read_two_atom_join(&lref, cp, cq).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    let resolved = verify_and_resolve_two_atom_join(
        &lref, head_root, cp, cq, &rec, &join, &result.columns,
    )
    .unwrap();
    let ground = materialized_join_resolved(&lref, "http://ex/p1", "http://ex/p2");
    assert_eq!(resolved.len(), 3);
    assert!(bag_eq_rows(rows_of(&resolved), ground), "resolved join must equal materialized");
}

/// A stack with typed string-VALUE objects, to show resolution yields `CanonicalObject::Value`.
fn value_stack() -> Vec<Arc<LayerCommitment>> {
    let store = open_sync_memory_store();
    let b = store.create_base_layer().unwrap();
    b.add_value_triple(ValueTriple::new_string_value(&iri("a1"), &iri("p1"), "hello")).unwrap();
    b.add_value_triple(ValueTriple::new_string_value(&iri("a2"), &iri("p1"), "world")).unwrap();
    b.add_value_triple(ValueTriple::new_node(&iri("a3"), &iri("p1"), &iri("b3"))).unwrap();
    b.add_value_triple(ValueTriple::new_node(&iri("a4"), &iri("p1"), &iri("b4"))).unwrap();
    let base = b.commit().unwrap();
    let base_lc = base.proof_commitment().unwrap();
    let child = base.open_write().unwrap();
    child.remove_value_triple(ValueTriple::new_node(&iri("a3"), &iri("p1"), &iri("b3"))).unwrap();
    child.remove_value_triple(ValueTriple::new_node(&iri("a4"), &iri("p1"), &iri("b4"))).unwrap();
    child.add_value_triple(ValueTriple::new_string_value(&iri("a5"), &iri("p1"), "foo")).unwrap();
    child.add_value_triple(ValueTriple::new_node(&iri("a6"), &iri("p1"), &iri("b6"))).unwrap();
    let head = child.commit().unwrap();
    vec![base_lc, head.proof_commitment().unwrap()]
}

#[test]
fn resolved_typed_value_objects() {
    let layers = value_stack();
    let lref = refs(&layers);
    let c = hash_pred("http://ex/p1");
    let (rec, filt, result) = prove_read_single_atom(&lref, FilterPosition::Predicate, c).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    let resolved = verify_and_resolve_single_atom(
        &lref, head_root, FilterPosition::Predicate, c, &rec, &filt, &result.columns, &result.id_hints,
    )
    .unwrap();
    // present = {a1 "hello", a2 "world", a5 "foo", a6 b6}: the O column resolves 3 Values + 1 Node.
    let ground = materialized_filter_resolved(&lref, FilterPosition::Predicate, c);
    assert!(bag_eq_rows(rows_of(&resolved), ground));
    let objects = &resolved[1]; // the O column
    let values = objects.iter().filter(|o| matches!(o, CanonicalObject::Value { .. })).count();
    let nodes = objects.iter().filter(|o| matches!(o, CanonicalObject::Node(_))).count();
    assert_eq!((values, nodes), (3, 1), "three typed literals and one node IRI resolved");
}

#[test]
fn resolve_forged_binding_rejects() {
    // A prover who publishes an altered result value (a fabricated binding) fails the
    // re-commitment bind to the authenticated result commitment.
    let layers = filter_stack();
    let lref = refs(&layers);
    let c = hash_pred("http://ex/p2");
    let (rec, filt, result) = prove_read_single_atom(&lref, FilterPosition::Predicate, c).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    let mut forged = result.columns.clone();
    forged[0][0] = ark(123_456_789); // a fabricated subject binding
    assert!(verify_and_resolve_single_atom(
        &lref, head_root, FilterPosition::Predicate, c, &rec, &filt, &forged, &result.id_hints,
    )
    .is_err());
}

#[test]
fn resolve_forged_join_binding_rejects() {
    let layers = join_stack();
    let lref = refs(&layers);
    let (cp, cq) = (hash_pred("http://ex/p1"), hash_pred("http://ex/p2"));
    let (rec, join, result) = prove_read_two_atom_join(&lref, cp, cq).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    let mut forged = result.columns.clone();
    forged[2][0] = ark(424_242); // a fabricated ?z binding
    assert!(verify_and_resolve_two_atom_join(
        &lref, head_root, cp, cq, &rec, &join, &forged,
    )
    .is_err());
}

// ============================================================
// Merkle-dict I2a — single-atom filter hash->term resolution via id-routed Merkle openings
// under a MANDATORY belt-hash (prover id-hints, catalog-scan retired on this path).
// ============================================================

/// The catalog-scan oracle resolution of a single-atom filter's published output columns — the
/// pre-I2a filter path, kept as the differential baseline the Merkle+belt-hash path must match.
fn i2a_catalog_scan_resolved(
    lref: &[&LayerCommitment],
    position: FilterPosition,
    result: &PresentReadResult,
) -> Vec<Vec<CanonicalObject>> {
    let kinds = filter_output_kinds(position);
    let views: Vec<VerifierCommitmentRef<'_>> = lref.iter().map(|l| l.verifier_view()).collect();
    result
        .columns
        .iter()
        .enumerate()
        .map(|(col_idx, col)| {
            col.iter()
                .map(|&value| match kinds[col_idx] {
                    ResolveKind::NodeValueHash => {
                        resolve_node_value_hash_via_catalog_scan(&views, value).unwrap()
                    }
                    ResolveKind::PredicateHash => {
                        resolve_predicate_hash_via_catalog_scan(&views, value).unwrap()
                    }
                    ResolveKind::NodeValueId => unreachable!("filter output is never an id kind"),
                })
                .collect()
        })
        .collect()
}

fn i2a_run_differential(position: FilterPosition, constant: ArkFr, expect_rows: usize) {
    let layers = filter_stack();
    let lref = refs(&layers);
    let (rec, filt, result) = prove_read_single_atom(&lref, position, constant).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    // Every hint column aligns with its output column (real prefix).
    assert_eq!(result.id_hints.len(), result.columns.len());
    assert!(result.id_hints.iter().all(|h| h.len() == result.len));
    assert_eq!(result.len, expect_rows);
    // Production path: id-routed Merkle openings under the mandatory belt-hash.
    let merkle = verify_and_resolve_single_atom(
        &lref, head_root, position, constant, &rec, &filt, &result.columns, &result.id_hints,
    )
    .unwrap();
    // Legacy catalog-scan oracle over the SAME published columns.
    let legacy = i2a_catalog_scan_resolved(&lref, position, &result);
    // EXACT (per column & per row) equality — a far stronger check than bag equality.
    assert_eq!(merkle, legacy, "I2a Merkle+belt-hash must reproduce the catalog scan exactly");
    // Sanity: real terms, not hashes.
    assert!(merkle.iter().flatten().all(|o| matches!(
        o,
        CanonicalObject::Node(_) | CanonicalObject::Value { .. }
    )));
}

#[test]
fn i2a_filter_merkle_belt_hash_equals_catalog_scan() {
    // Predicate=p2: 4 rows; subjects a3,a4,a8 (base) + a6 (delta) and objects c3,c4,c8 (base) +
    // c6 (delta) exercise cross-layer id routing on BOTH node/value output columns.
    i2a_run_differential(FilterPosition::Predicate, hash_pred("http://ex/p1"), 2);
    i2a_run_differential(FilterPosition::Predicate, hash_pred("http://ex/p2"), 4);
    // Subject=a2: output [P, O] exercises the PredicateHash column (id-routed predicate tree).
    i2a_run_differential(FilterPosition::Subject, hash_node("http://ex/a2"), 1);
    // Object=c3: output [S, P] exercises node/value + predicate output columns.
    i2a_run_differential(FilterPosition::Object, hash_node("http://ex/c3"), 1);
}

#[test]
fn i2a_wrong_id_hint_rejected_by_belt_hash() {
    // A wrong id-hint pointing at a DIFFERENT real dict entry in the SAME layer: routing +
    // opening SUCCEED, but the opened term's hash != the authenticated output hash, so the
    // MANDATORY belt-hash rejects. This is the sole soundness guard on the untrusted hint.
    let layers = filter_stack();
    let lref = refs(&layers);
    let c = hash_pred("http://ex/p2"); // subjects a3,a4,a8,a6
    let (rec, filt, mut result) = prove_read_single_atom(&lref, FilterPosition::Predicate, c).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    // Point row 0's subject hint (a3) at row 1's real subject id (a4) — both base-layer entries.
    assert_ne!(result.id_hints[0][0], result.id_hints[0][1]);
    result.id_hints[0][0] = result.id_hints[0][1];
    assert!(verify_and_resolve_single_atom(
        &lref, head_root, FilterPosition::Predicate, c, &rec, &filt, &result.columns, &result.id_hints,
    )
    .is_err());
}

#[test]
fn i2a_cross_layer_id_hint_rejected_by_belt_hash() {
    // A base-layer subject's hint pointed at a DELTA-layer real id: routing sends it to the head
    // layer's tree and opens a valid (but different) term, and the belt-hash rejects. Confirms
    // the guard holds even when the wrong id lives in another layer's interval.
    let layers = filter_stack();
    let lref = refs(&layers);
    let c = hash_pred("http://ex/p2");
    let (rec, filt, mut result) = prove_read_single_atom(&lref, FilterPosition::Predicate, c).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    // Subject hints in ascending present order = [a3(base), a4(base), a8(base), a6(delta)].
    let base_subject = result.id_hints[0][0];
    let delta_subject = result.id_hints[0][3];
    assert_ne!(base_subject, delta_subject);
    result.id_hints[0][0] = delta_subject; // base row now claims the delta id
    assert!(verify_and_resolve_single_atom(
        &lref, head_root, FilterPosition::Predicate, c, &rec, &filt, &result.columns, &result.id_hints,
    )
    .is_err());
}

#[test]
fn i2a_out_of_range_id_hint_rejected_by_routing() {
    // An id-hint outside EVERY layer's introduced interval fails closed at routing (`nv_layer`),
    // before any opening — the interval partition is a public function of the chain-validated
    // counts, never a prover choice.
    let layers = filter_stack();
    let lref = refs(&layers);
    let c = hash_pred("http://ex/p2");
    let (rec, filt, mut result) = prove_read_single_atom(&lref, FilterPosition::Predicate, c).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    result.id_hints[0][0] = 10_000_000; // far beyond head_node_value_count
    assert!(verify_and_resolve_single_atom(
        &lref, head_root, FilterPosition::Predicate, c, &rec, &filt, &result.columns, &result.id_hints,
    )
    .is_err());
}

#[test]
fn i2a_tampered_value_rejected_by_recommitment_bind() {
    // The belt-hash is only meaningful because `value` is the proof-authenticated present hash:
    // a tampered published output value fails the step-1 re-commitment bind to `com_out` BEFORE
    // resolution, so the prover cannot swap in a `value` that matches a chosen id-hint's term.
    let layers = filter_stack();
    let lref = refs(&layers);
    let c = hash_pred("http://ex/p2");
    let (rec, filt, result) = prove_read_single_atom(&lref, FilterPosition::Predicate, c).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    let mut tampered = result.columns.clone();
    tampered[1][0] = ark(987_654_321); // a fabricated object-column hash
    assert!(verify_and_resolve_single_atom(
        &lref, head_root, FilterPosition::Predicate, c, &rec, &filt, &tampered, &result.id_hints,
    )
    .is_err());
}

#[test]
fn i2a_missing_id_hints_fail_closed() {
    // The hash-kind resolve path REQUIRES id-hints — an empty hint set fails closed (never a
    // silent catalog scan).
    let layers = filter_stack();
    let lref = refs(&layers);
    let c = hash_pred("http://ex/p2");
    let (rec, filt, result) = prove_read_single_atom(&lref, FilterPosition::Predicate, c).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    assert!(verify_and_resolve_single_atom(
        &lref, head_root, FilterPosition::Predicate, c, &rec, &filt, &result.columns, &[],
    )
    .is_err());
}

// ============================================================
// Wave 6 — general n-atom BGP over present.
// ============================================================

fn var() -> PresentTerm {
    PresentTerm::Variable
}
fn node_const(iri: &str) -> PresentTerm {
    PresentTerm::Constant(hash_node(iri))
}
fn atom(subject: PresentTerm, predicate: &str, object: PresentTerm) -> PresentBgpAtom {
    PresentBgpAtom {
        subject,
        predicate_hash: hash_pred(predicate),
        object,
        subject_object_equal: false,
        object_range: None,
    }
}
fn scalar_join(acc_key: usize, rel_key: usize) -> JoinStep {
    JoinStep { acc_key, rel_key, additional_keys: Vec::new() }
}

/// A base+delta stack whose present supports a 3-atom chain
/// Triple(?x, p1, ?y) & Triple(?y, p2, ?z) & Triple(?z, p3, ?w):
///   present p1 = {x1->y1, x2->y1, x4->y1}   (base x3->y3 removed; delta adds x4->y1)
///   present p2 = {y1->z1}                    (base y1->z9 removed)
///   present p3 = {z1->w1, z1->w2, z1->w3}    (delta adds z1->w3)
/// Join on ?y=y1 (3 x) and ?z=z1 (3 w) => 3 x-bindings x 3 w-bindings = 9 solutions.
fn bgp_stack() -> Vec<Arc<LayerCommitment>> {
    let store = open_sync_memory_store();
    let b = store.create_base_layer().unwrap();
    for (s, p, o) in [
        ("x1", "p1", "y1"), ("x2", "p1", "y1"), ("y1", "p2", "z1"),
        ("z1", "p3", "w1"), ("z1", "p3", "w2"),
        ("x3", "p1", "y3"), ("y1", "p2", "z9"),
    ] {
        b.add_value_triple(ValueTriple::new_node(&iri(s), &iri(p), &iri(o))).unwrap();
    }
    let base = b.commit().unwrap();
    let base_lc = base.proof_commitment().unwrap();
    let child = base.open_write().unwrap();
    child.remove_value_triple(ValueTriple::new_node(&iri("x3"), &iri("p1"), &iri("y3"))).unwrap();
    child.remove_value_triple(ValueTriple::new_node(&iri("y1"), &iri("p2"), &iri("z9"))).unwrap();
    child.add_value_triple(ValueTriple::new_node(&iri("x4"), &iri("p1"), &iri("y1"))).unwrap();
    child.add_value_triple(ValueTriple::new_node(&iri("z1"), &iri("p3"), &iri("w3"))).unwrap();
    let head = child.commit().unwrap();
    vec![base_lc, head.proof_commitment().unwrap()]
}

/// Materialized 3-atom chain over present, resolved to CanonicalObjects [x, y, z, w].
fn materialized_3chain_resolved(
    layers: &[&LayerCommitment], p1: &str, p2: &str, p3: &str,
) -> Vec<Vec<CanonicalObject>> {
    let present = reconcile_present(layers).unwrap();
    let mut out = Vec::new();
    for a in present.iter().filter(|r| r.triple.predicate == p1.as_bytes()) {
        for bb in present.iter().filter(|r| r.triple.predicate == p2.as_bytes()) {
            if a.object_id != bb.subject_id { continue; }
            for c in present.iter().filter(|r| r.triple.predicate == p3.as_bytes()) {
                if bb.object_id != c.subject_id { continue; }
                out.push(vec![
                    CanonicalObject::Node(a.triple.subject.clone()),
                    a.triple.object.clone(),
                    bb.triple.object.clone(),
                    c.triple.object.clone(),
                ]);
            }
        }
    }
    out
}

#[test]
fn three_atom_bgp_over_present() {
    let layers = bgp_stack();
    let lref = refs(&layers);
    let atoms = [
        atom(var(), "http://ex/p1", var()), // Triple(?x, p1, ?y)
        atom(var(), "http://ex/p2", var()), // Triple(?y, p2, ?z)
        atom(var(), "http://ex/p3", var()), // Triple(?z, p3, ?w)
    ];
    let joins = [scalar_join(1, 0), scalar_join(2, 0)]; // acc.y==r2.y ; acc.z==r3.z
    let (rec, bgp, result) = prove_read_bgp(&lref, &atoms, &joins).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    let n = verify_read_bgp(&lref, head_root, &atoms, &joins, &rec, &bgp).unwrap();
    assert_eq!(n, 9, "3-atom chain over the delta stack must have 9 solutions");
    assert_eq!(result.len, 9);
    let resolved = verify_and_resolve_bgp(
        &lref, head_root, &atoms, &joins, &rec, &bgp, &result.columns,
    )
    .unwrap();
    let ground =
        materialized_3chain_resolved(&lref, "http://ex/p1", "http://ex/p2", "http://ex/p3");
    assert_eq!(ground.len(), 9);
    assert!(bag_eq_rows(rows_of(&resolved), ground), "resolved 3-chain must equal materialized");
}

#[test]
fn multi_constant_bgp_over_present() {
    // Triple(x1, p1, ?y) & Triple(?y, p2, ?z): subject constant on atom 0 pins ?y=y1,
    // then join p2 -> exactly (y1, z1). Result schema [y, z].
    let layers = bgp_stack();
    let lref = refs(&layers);
    let atoms = [
        atom(node_const("http://ex/x1"), "http://ex/p1", var()),
        atom(var(), "http://ex/p2", var()),
    ];
    let joins = [scalar_join(0, 0)]; // acc.y == r2.y
    let (rec, bgp, result) = prove_read_bgp(&lref, &atoms, &joins).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    let n = verify_read_bgp(&lref, head_root, &atoms, &joins, &rec, &bgp).unwrap();
    assert_eq!(n, 1, "subject-constant 2-atom join has exactly one solution");
    let resolved = verify_and_resolve_bgp(
        &lref, head_root, &atoms, &joins, &rec, &bgp, &result.columns,
    )
    .unwrap();
    assert_eq!(
        rows_of(&resolved),
        vec![vec![
            CanonicalObject::Node(iri("y1").into_bytes()),
            CanonicalObject::Node(iri("z1").into_bytes()),
        ]],
    );
}

#[test]
fn single_atom_object_constant_bgp() {
    // Triple(?x, p1, y1): object constant, keep subject. present p1 objects=y1: x1,x2,x4.
    let layers = bgp_stack();
    let lref = refs(&layers);
    let atoms = [atom(var(), "http://ex/p1", node_const("http://ex/y1"))];
    let (rec, bgp, result) = prove_read_bgp(&lref, &atoms, &[]).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    let n = verify_read_bgp(&lref, head_root, &atoms, &[], &rec, &bgp).unwrap();
    assert_eq!(n, 3);
    let resolved = verify_and_resolve_bgp(
        &lref, head_root, &atoms, &[], &rec, &bgp, &result.columns,
    )
    .unwrap();
    let subjects: Vec<CanonicalObject> = resolved.into_iter().flatten().collect();
    let ground: Vec<CanonicalObject> = ["x1", "x2", "x4"]
        .iter()
        .map(|s| CanonicalObject::Node(iri(s).into_bytes()))
        .collect();
    assert!(bag_eq_rows(
        subjects.into_iter().map(|o| vec![o]).collect(),
        ground.into_iter().map(|o| vec![o]).collect(),
    ));
}

/// A stack whose present has self-loops (subject==object) and non-loops:
/// present = {a1->a1, a2->a2, a3->b3, a5->a5, a6->b6}. Triple(?s, pself, ?s) => {a1,a2,a5}.
fn self_stack() -> Vec<Arc<LayerCommitment>> {
    let store = open_sync_memory_store();
    let b = store.create_base_layer().unwrap();
    for (s, o) in [("a1", "a1"), ("a2", "a2"), ("a3", "b3"), ("a4", "a4"), ("a7", "a7")] {
        b.add_value_triple(ValueTriple::new_node(&iri(s), &iri("pself"), &iri(o))).unwrap();
    }
    let base = b.commit().unwrap();
    let base_lc = base.proof_commitment().unwrap();
    let child = base.open_write().unwrap();
    child.remove_value_triple(ValueTriple::new_node(&iri("a4"), &iri("pself"), &iri("a4"))).unwrap();
    child.remove_value_triple(ValueTriple::new_node(&iri("a7"), &iri("pself"), &iri("a7"))).unwrap();
    child.add_value_triple(ValueTriple::new_node(&iri("a5"), &iri("pself"), &iri("a5"))).unwrap();
    child.add_value_triple(ValueTriple::new_node(&iri("a6"), &iri("pself"), &iri("b6"))).unwrap();
    let head = child.commit().unwrap();
    vec![base_lc, head.proof_commitment().unwrap()]
}

#[test]
fn repeated_variable_bgp_over_present() {
    let layers = self_stack();
    let lref = refs(&layers);
    let self_atom = PresentBgpAtom {
        subject: var(),
        predicate_hash: hash_pred("http://ex/pself"),
        object: var(),
        subject_object_equal: true,
        object_range: None,
    };
    let atoms = [self_atom];
    let (rec, bgp, result) = prove_read_bgp(&lref, &atoms, &[]).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    let n = verify_read_bgp(&lref, head_root, &atoms, &[], &rec, &bgp).unwrap();
    assert_eq!(n, 3, "three present self-loops match Triple(?s, pself, ?s)");
    let resolved = verify_and_resolve_bgp(
        &lref, head_root, &atoms, &[], &rec, &bgp, &result.columns,
    )
    .unwrap();
    let got: Vec<Vec<CanonicalObject>> = rows_of(&resolved);
    let ground: Vec<Vec<CanonicalObject>> = ["a1", "a2", "a5"]
        .iter()
        .map(|s| vec![CanonicalObject::Node(iri(s).into_bytes())])
        .collect();
    assert!(bag_eq_rows(got, ground), "self-join must equal materialized a1,a2,a5");
}

// --- Soundness negatives over the general BGP path ---

#[test]
fn bgp_fabricated_relation_commitment_rejects() {
    let layers = bgp_stack();
    let lref = refs(&layers);
    let atoms = [
        atom(var(), "http://ex/p1", var()),
        atom(var(), "http://ex/p2", var()),
        atom(var(), "http://ex/p3", var()),
    ];
    let joins = [scalar_join(1, 0), scalar_join(2, 0)];
    let (rec, mut bgp, _) = prove_read_bgp(&lref, &atoms, &joins).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    // Fabricate atom 0's first kept column: the compaction binding fails.
    bgp.atoms[0].com_kept[0] = commit_col(&vec![ark(4242); 1usize << rec.working_log]);
    assert!(verify_read_bgp(&lref, head_root, &atoms, &joins, &rec, &bgp).is_err());
}

#[test]
fn bgp_dropped_row_rejects() {
    let layers = bgp_stack();
    let lref = refs(&layers);
    let atoms = [
        atom(var(), "http://ex/p1", var()),
        atom(var(), "http://ex/p2", var()),
        atom(var(), "http://ex/p3", var()),
    ];
    let joins = [scalar_join(1, 0), scalar_join(2, 0)];
    let (rec, bgp, _) = prove_read_bgp(&lref, &atoms, &joins).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    // Claim one fewer matching row in atom 0's compaction.
    let mut forged = bgp.clone();
    forged.atoms[0].m -= 1;
    assert!(verify_read_bgp(&lref, head_root, &atoms, &joins, &rec, &forged).is_err());
}

#[test]
fn bgp_wrong_constant_rejects() {
    let layers = bgp_stack();
    let lref = refs(&layers);
    let atoms = [
        atom(var(), "http://ex/p1", var()),
        atom(var(), "http://ex/p2", var()),
    ];
    let joins = [scalar_join(1, 0)];
    let (rec, bgp, _) = prove_read_bgp(&lref, &atoms, &joins).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    // Verify against a different first predicate: the selection is p1-bound.
    let wrong = [
        atom(var(), "http://ex/pX", var()),
        atom(var(), "http://ex/p2", var()),
    ];
    assert!(verify_read_bgp(&lref, head_root, &wrong, &joins, &rec, &bgp).is_err());
}

#[test]
fn bgp_tampered_present_id_rejects() {
    // Feeding a wrong combined-dict id for present (to fabricate a joining tuple)
    // fails the id -> hash authentication against the reconciled hashes.
    let layers = bgp_stack();
    let lref = refs(&layers);
    let atoms = [
        atom(var(), "http://ex/p1", var()),
        atom(var(), "http://ex/p2", var()),
    ];
    let joins = [scalar_join(1, 0)];
    let (rec, mut bgp, _) = prove_read_bgp(&lref, &atoms, &joins).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    bgp.auth_s.com_id = commit_col(&vec![ark(3); 1usize << rec.working_log]);
    assert!(verify_read_bgp(&lref, head_root, &atoms, &joins, &rec, &bgp).is_err());
}

#[test]
fn bgp_forged_result_binding_rejects() {
    // A prover who publishes an altered result value fails the re-commitment bind.
    let layers = bgp_stack();
    let lref = refs(&layers);
    let atoms = [
        atom(var(), "http://ex/p1", var()),
        atom(var(), "http://ex/p2", var()),
        atom(var(), "http://ex/p3", var()),
    ];
    let joins = [scalar_join(1, 0), scalar_join(2, 0)];
    let (rec, bgp, result) = prove_read_bgp(&lref, &atoms, &joins).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    let mut forged = result.columns.clone();
    forged[3][0] = ark(987_654_321); // fabricated ?w binding
    assert!(verify_and_resolve_bgp(
        &lref, head_root, &atoms, &joins, &rec, &bgp, &forged,
    )
    .is_err());
}

#[test]
fn bgp_non_satisfying_join_tuple_is_excluded() {
    // Ground-truth completeness+soundness: the reconciled answer equals the join over
    // the MATERIALIZED present exactly, so a Cartesian (non-satisfying) tuple such as
    // (x1, y1, z1, w) with a w whose z != z1 never appears. present p3 only has z1->w*,
    // and the base's non-joining rows (x3->y3, y1->z9) were removed, so any fabricated
    // tuple would inflate the count beyond the materialized 9.
    let layers = bgp_stack();
    let lref = refs(&layers);
    let atoms = [
        atom(var(), "http://ex/p1", var()),
        atom(var(), "http://ex/p2", var()),
        atom(var(), "http://ex/p3", var()),
    ];
    let joins = [scalar_join(1, 0), scalar_join(2, 0)];
    let (rec, bgp, result) = prove_read_bgp(&lref, &atoms, &joins).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    let n = verify_read_bgp(&lref, head_root, &atoms, &joins, &rec, &bgp).unwrap();
    let ground =
        materialized_3chain_resolved(&lref, "http://ex/p1", "http://ex/p2", "http://ex/p3");
    assert_eq!(n, ground.len());
    // Every delivered tuple satisfies both join predicates (y-link and z-link).
    let resolved = verify_and_resolve_bgp(
        &lref, head_root, &atoms, &joins, &rec, &bgp, &result.columns,
    )
    .unwrap();
    for row in rows_of(&resolved) {
        // y == z-atom's subject was enforced by the join; assert the tuple is one the
        // materialized set contains (no fabricated Cartesian tuple slipped through).
        assert!(ground.contains(&row));
    }
}

// ============================================================
// Wave 6 — NUMERIC RANGE over present.
// ============================================================

fn uint32_tag() -> ArkFr {
    ArkFr::from_u64(crate::proof::CanonicalDatatype::UInt32 as u64)
}

fn u32_of(object: &CanonicalObject) -> Option<u64> {
    match object {
        CanonicalObject::Value {
            datatype: crate::proof::CanonicalDatatype::UInt32,
            lexical,
        } if lexical.len() == 4 => {
            Some(u32::from_be_bytes([lexical[0], lexical[1], lexical[2], lexical[3]]) as u64)
        }
        _ => None,
    }
}

fn cmp_op(op: RangeOp, a: u64, b: u64) -> bool {
    match op {
        RangeOp::Lt => a < b,
        RangeOp::Le => a <= b,
        RangeOp::Gt => a > b,
        RangeOp::Ge => a >= b,
    }
}

/// A base+delta stack of `t(person, age, <UInt32>)` + `t(person, name, <node>)`, with a
/// removed person (frank) and an added person (grace) in the delta.
/// present ages: alice=30, bob=15, carol=18, dave=40, erin=18, grace=25.
fn age_stack() -> Vec<Arc<LayerCommitment>> {
    use tdb_succinct::TdbDataType;
    let store = open_sync_memory_store();
    let b = store.create_base_layer().unwrap();
    for (person, age) in [
        ("alice", 30u32), ("bob", 15), ("carol", 18), ("dave", 40), ("erin", 18),
        ("frank", 99),
    ] {
        b.add_value_triple(ValueTriple::new_value(&iri(person), &iri("age"), u32::make_entry(&age))).unwrap();
        b.add_value_triple(ValueTriple::new_node(&iri(person), &iri("name"), &iri(&format!("{person}_n")))).unwrap();
    }
    let base = b.commit().unwrap();
    let base_lc = base.proof_commitment().unwrap();
    let child = base.open_write().unwrap();
    child.remove_value_triple(ValueTriple::new_value(&iri("frank"), &iri("age"), u32::make_entry(&99u32))).unwrap();
    child.remove_value_triple(ValueTriple::new_node(&iri("frank"), &iri("name"), &iri("frank_n"))).unwrap();
    child.add_value_triple(ValueTriple::new_value(&iri("grace"), &iri("age"), u32::make_entry(&25u32))).unwrap();
    child.add_value_triple(ValueTriple::new_node(&iri("grace"), &iri("name"), &iri("grace_n"))).unwrap();
    let head = child.commit().unwrap();
    vec![base_lc, head.proof_commitment().unwrap()]
}

fn range_atom(op: RangeOp, c: u64, second: Option<(RangeOp, u64)>) -> PresentBgpAtom {
    PresentBgpAtom {
        subject: var(),
        predicate_hash: hash_pred("http://ex/age"),
        object: var(),
        subject_object_equal: false,
        object_range: Some(ObjectRangeSpec {
            op,
            datatype: uint32_tag(),
            c: ArkFr::from_u64(c),
            b_bits: OBJECT_RANGE_B_BITS,
            second: second.map(|(op, c)| crate::proof::query::ObjectRangeBound {
                op,
                c: ArkFr::from_u64(c),
            }),
        }),
    }
}

/// Materialized single-atom age range, resolved to [x, v] rows (matching the kept S,O order).
fn materialized_age_range(
    layers: &[&LayerCommitment], predicate: impl Fn(u64) -> bool,
) -> Vec<Vec<CanonicalObject>> {
    reconcile_present(layers)
        .unwrap()
        .iter()
        .filter(|r| r.triple.predicate == b"http://ex/age")
        .filter_map(|r| {
            let v = u32_of(&r.triple.object)?;
            if predicate(v) {
                Some(vec![CanonicalObject::Node(r.triple.subject.clone()), r.triple.object.clone()])
            } else {
                None
            }
        })
        .collect()
}

fn run_age_range(op: RangeOp, c: u64, expect: usize) {
    let layers = age_stack();
    let lref = refs(&layers);
    let atoms = [range_atom(op, c, None)];
    let (rec, bgp, result) = prove_read_bgp(&lref, &atoms, &[]).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    let n = verify_read_bgp(&lref, head_root, &atoms, &[], &rec, &bgp).unwrap();
    let ground = materialized_age_range(&lref, |v| cmp_op(op, v, c));
    assert_eq!(n, ground.len(), "op {:?} c {}: count must equal materialized", op, c);
    assert_eq!(n, expect, "op {:?} c {}", op, c);
    let resolved = verify_and_resolve_bgp(&lref, head_root, &atoms, &[], &rec, &bgp, &result.columns).unwrap();
    assert!(
        bag_eq_rows(rows_of(&resolved), ground),
        "op {:?} c {}: resolved must equal materialized",
        op,
        c
    );
}

#[test]
fn range_single_bound_ge_boundary_over_present() {
    // present ages: 30,15,18,40,18(erin),25. c=18 hits the boundary (carol,erin == 18):
    // Ge INCLUDES them, Gt EXCLUDES them — the key completeness/soundness distinction.
    run_age_range(RangeOp::Ge, 18, 5); // 30,18,40,18,25
}

#[test]
fn range_single_bound_gt_boundary_over_present() {
    run_age_range(RangeOp::Gt, 18, 3); // 30,40,25
}

#[test]
fn range_single_bound_le_over_present() {
    run_age_range(RangeOp::Le, 18, 3); // 15,18,18
}

#[test]
fn range_excludes_everything_over_present() {
    run_age_range(RangeOp::Gt, 1000, 0); // empty range -> empty result
}

#[test]
fn range_in_range_over_present() {
    // InRange [18, 30): >= 18 AND < 30 -> {carol 18, erin 18, grace 25} = 3.
    let layers = age_stack();
    let lref = refs(&layers);
    let atoms = [range_atom(RangeOp::Ge, 18, Some((RangeOp::Lt, 30)))];
    let (rec, bgp, result) = prove_read_bgp(&lref, &atoms, &[]).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    let n = verify_read_bgp(&lref, head_root, &atoms, &[], &rec, &bgp).unwrap();
    let ground = materialized_age_range(&lref, |v| v >= 18 && v < 30);
    assert_eq!(n, 3);
    assert_eq!(n, ground.len());
    let resolved = verify_and_resolve_bgp(&lref, head_root, &atoms, &[], &rec, &bgp, &result.columns).unwrap();
    assert!(bag_eq_rows(rows_of(&resolved), ground));
}

/// Materialized age-range join `Triple(?x, age, ?v)[?v op c] & Triple(?x, name, ?n)`,
/// result schema [v, x, n] (matching the left-deep chain output for acc_key=rel_key=0).
fn materialized_age_name_join(
    layers: &[&LayerCommitment], predicate: impl Fn(u64) -> bool,
) -> Vec<Vec<CanonicalObject>> {
    let present = reconcile_present(layers).unwrap();
    let mut out = Vec::new();
    for a in present.iter().filter(|r| r.triple.predicate == b"http://ex/age") {
        let Some(v) = u32_of(&a.triple.object) else { continue };
        if !predicate(v) { continue; }
        for nrec in present.iter().filter(|r| r.triple.predicate == b"http://ex/name") {
            if nrec.subject_id == a.subject_id {
                out.push(vec![
                    a.triple.object.clone(),
                    CanonicalObject::Node(a.triple.subject.clone()),
                    nrec.triple.object.clone(),
                ]);
            }
        }
    }
    out
}

#[test]
fn range_join_over_present() {
    // Triple(?x, age, ?v)[?v >= 18] & Triple(?x, name, ?n): 5 people qualify, each with
    // exactly one name -> 5 solutions [v, x, n].
    let layers = age_stack();
    let lref = refs(&layers);
    let atoms = [
        range_atom(RangeOp::Ge, 18, None),
        atom(var(), "http://ex/name", var()),
    ];
    let joins = [scalar_join(0, 0)]; // acc.x == name.x
    let (rec, bgp, result) = prove_read_bgp(&lref, &atoms, &joins).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    let n = verify_read_bgp(&lref, head_root, &atoms, &joins, &rec, &bgp).unwrap();
    let ground = materialized_age_name_join(&lref, |v| v >= 18);
    assert_eq!(n, 5);
    assert_eq!(n, ground.len());
    let resolved = verify_and_resolve_bgp(&lref, head_root, &atoms, &joins, &rec, &bgp, &result.columns).unwrap();
    assert!(bag_eq_rows(rows_of(&resolved), ground), "range join must equal materialized");
}

// --- Range soundness negatives ---

#[test]
fn range_wrong_bound_rejects() {
    let layers = age_stack();
    let lref = refs(&layers);
    let atoms = [range_atom(RangeOp::Ge, 18, None)];
    let (rec, bgp, _) = prove_read_bgp(&lref, &atoms, &[]).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    // Verify the >=18 proof against a >=19 bound: the range selection is 18-bound.
    let wrong = [range_atom(RangeOp::Ge, 19, None)];
    assert!(verify_read_bgp(&lref, head_root, &wrong, &[], &rec, &bgp).is_err());
}

#[test]
fn range_wrong_op_rejects() {
    let layers = age_stack();
    let lref = refs(&layers);
    let atoms = [range_atom(RangeOp::Ge, 18, None)];
    let (rec, bgp, _) = prove_read_bgp(&lref, &atoms, &[]).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    let wrong = [range_atom(RangeOp::Gt, 18, None)];
    assert!(verify_read_bgp(&lref, head_root, &wrong, &[], &rec, &bgp).is_err());
}

#[test]
fn range_tampered_value_column_rejects() {
    // Tampering the combined range dict's authenticated VALUE column breaks its
    // relocation binding to the root-bound per-layer range commitments.
    let layers = age_stack();
    let lref = refs(&layers);
    let atoms = [range_atom(RangeOp::Ge, 18, None)];
    let (rec, mut bgp, _) = prove_read_bgp(&lref, &atoms, &[]).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    let rd = bgp.range_dict.as_mut().expect("a range dict was built");
    rd.com_value_at_k = commit_col(&vec![ark(7); 1usize << rd.log.max(rec.working_log)]);
    assert!(verify_read_bgp(&lref, head_root, &atoms, &[], &rec, &bgp).is_err());
}

#[test]
fn range_fabricated_row_rejects() {
    let layers = age_stack();
    let lref = refs(&layers);
    let atoms = [range_atom(RangeOp::Ge, 18, None)];
    let (rec, mut bgp, _) = prove_read_bgp(&lref, &atoms, &[]).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    // Fabricate a kept output column: the compaction binding fails.
    bgp.atoms[0].com_kept[0] = commit_col(&vec![ark(4242); 1usize << rec.working_log]);
    assert!(verify_read_bgp(&lref, head_root, &atoms, &[], &rec, &bgp).is_err());
}

#[test]
fn range_dropped_row_rejects() {
    let layers = age_stack();
    let lref = refs(&layers);
    let atoms = [range_atom(RangeOp::Ge, 18, None)];
    let (rec, bgp, _) = prove_read_bgp(&lref, &atoms, &[]).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    let mut forged = bgp.clone();
    forged.atoms[0].m -= 1; // claim one fewer in-range row
    assert!(verify_read_bgp(&lref, head_root, &atoms, &[], &rec, &forged).is_err());
}

#[test]
fn range_plan_mismatch_rejects() {
    // A proof built WITH a range cannot be reinterpreted as a plain atom (the verifier's
    // trusted plan drives which gadgets run; a plain plan has no range dict).
    let layers = age_stack();
    let lref = refs(&layers);
    let atoms = [range_atom(RangeOp::Ge, 18, None)];
    let (rec, bgp, _) = prove_read_bgp(&lref, &atoms, &[]).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    let plain = [atom(var(), "http://ex/age", var())];
    assert!(verify_read_bgp(&lref, head_root, &plain, &[], &rec, &bgp).is_err());
}

// ============================================================
// Wave 7 — trust-boundary hardening (present set-ness + combined-dict id uniqueness).
// ============================================================

#[test]
fn distinct_checker_rejects_a_duplicate_column() {
    // An honest store cannot emit a duplicated present row or a duplicated dict id, so drive
    // the assert's checker DIRECTLY on a crafted column: a distinct column proves + verifies;
    // a column with a duplicate has a zero adjacent gap after sorting, which the strict-sort
    // range check ([1, 2^bits)) rejects — so NO proof exists (the prover cannot lie).
    let ctx = StoreProverContext::cpu();
    let log = 3usize; // domain 8
    let bits = 16usize;
    let mut column = pad_to(&[ark(3), ark(5), ark(8), ark(2)], log);
    let com = commit_col(&column);
    let mut pt = Transcript254::new(b"wave7-distinct");
    let proof = prove_column_distinct(&ctx, &column, com, 4, log, bits, &mut pt).unwrap();
    let mut vt = Transcript254::new(b"wave7-distinct");
    verify_column_distinct(com, 4, log, bits, &proof, &mut vt).unwrap();
    // Now inject a duplicate (value 3 appears twice): after sorting, the adjacent gap of the
    // two 3s is 0, which the strict-sort range check rejects — either the prover cannot build
    // a proof, or the verifier rejects it. No valid distinctness proof passes.
    column[2] = ark(3);
    let com_dup = commit_col(&column);
    let mut dt = Transcript254::new(b"wave7-dup");
    match prove_column_distinct(&ctx, &column, com_dup, 4, log, bits, &mut dt) {
        Err(_) => {}
        Ok(dup_proof) => {
            let mut dvt = Transcript254::new(b"wave7-dup");
            assert!(
                verify_column_distinct(com_dup, 4, log, bits, &dup_proof, &mut dvt).is_err(),
                "the strict-sort assert must reject a duplicated column"
            );
        }
    }
}

#[test]
fn full_field_column_distinct_accepts_distinct_hashes() {
    // Full-field (~254-bit) analogue of `distinct_checker_rejects_a_duplicate_column`: a column
    // of DISTINCT canonical field elements — a mix of lower-⅓ values (`hi = 0`) and near-`p`
    // values (`hi = p_hi`, differing only in the low limb), exactly the shape a combined hash
    // catalog takes — proves + verifies via the permutation + I4-a lex-distinct composition, for
    // which the bounded strict-sort's 253-bit gap cannot cover.
    let ctx = StoreProverContext::cpu();
    let log = 3usize; // domain 8
    let column = pad_to(
        &[
            ark(3),
            ark(100_000),
            ark(u64::MAX),
            ark(1) - ark(9),  // p - 8: near p, hi = p_hi
            ark(1) - ark(2),  // p - 1: the max canonical value (equal hi, larger lo)
            ark(1) - ark(50), // p - 49: another near-p value, equal hi
        ],
        log,
    );
    let com = commit_col(&column);
    let mut pt = Transcript254::new(b"i4b-fullfield-distinct");
    let proof = prove_column_distinct_fullfield(&ctx, &column, com, 6, log, &mut pt).unwrap();
    let mut vt = Transcript254::new(b"i4b-fullfield-distinct");
    verify_column_distinct_fullfield(com, 6, log, &proof, &mut vt).unwrap();
}

#[test]
fn full_field_column_distinct_rejects_duplicate() {
    // A repeated full-field value: after the permutation sorts the duplicate adjacent, I4-a's
    // strict-lex increase sees an equal pair (`lo_shift - lo - 1 = -1`) and range-fails — so
    // either the prover cannot build a proof or the verifier rejects it. No valid distinctness
    // proof passes (the prover cannot lie).
    let ctx = StoreProverContext::cpu();
    let log = 3usize;
    let dup = ark(1) - ark(9); // p - 8
    let column = pad_to(&[ark(3), dup, ark(100_000), dup], log);
    let com = commit_col(&column);
    let mut dt = Transcript254::new(b"i4b-fullfield-dup");
    match prove_column_distinct_fullfield(&ctx, &column, com, 4, log, &mut dt) {
        Err(_) => {}
        Ok(dup_proof) => {
            let mut dvt = Transcript254::new(b"i4b-fullfield-dup");
            assert!(
                verify_column_distinct_fullfield(com, 4, log, &dup_proof, &mut dvt).is_err(),
                "the lex-distinct gadget must reject a duplicated full-field column"
            );
        }
    }
}

#[test]
fn full_field_column_distinct_rejects_tampered_com() {
    // Prove a valid full-field column, then verify against a commitment to a DIFFERENT column:
    // the permutation binding to the wrong `com` fails.
    let ctx = StoreProverContext::cpu();
    let log = 3usize;
    let column = pad_to(&[ark(3), ark(100_000), ark(1) - ark(9), ark(1) - ark(2)], log);
    let com = commit_col(&column);
    let mut pt = Transcript254::new(b"i4b-fullfield-tamper");
    let proof = prove_column_distinct_fullfield(&ctx, &column, com, 4, log, &mut pt).unwrap();
    let wrong = commit_col(&pad_to(&[ark(11), ark(22), ark(33), ark(44)], log));
    let mut vt = Transcript254::new(b"i4b-fullfield-tamper");
    assert!(verify_column_distinct_fullfield(wrong, 4, log, &proof, &mut vt).is_err());
}

#[test]
fn full_field_column_distinct_single_row_trivial() {
    // A single-row column is a trivially-distinct set: `n == 1` proves + verifies (the bounded
    // helper requires `n >= 2`; the full-field helper's floor is `n >= 1`).
    let ctx = StoreProverContext::cpu();
    let log = 2usize; // domain 4 (the permutation needs a relocatable domain, `working_log >= 1`)
    let column = pad_to(&[ark(1) - ark(2)], log); // a single near-p value
    let com = commit_col(&column);
    let mut pt = Transcript254::new(b"i4b-fullfield-single");
    let proof = prove_column_distinct_fullfield(&ctx, &column, com, 1, log, &mut pt).unwrap();
    let mut vt = Transcript254::new(b"i4b-fullfield-single");
    verify_column_distinct_fullfield(com, 1, log, &proof, &mut vt).unwrap();
}

#[test]
fn tampered_present_distinct_rejects() {
    let layers = bgp_stack();
    let lref = refs(&layers);
    let atoms = [atom(var(), "http://ex/p1", var())];
    let (rec, mut bgp, _) = prove_read_bgp(&lref, &atoms, &[]).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    bgp.present_distinct.as_mut().unwrap().com_sorted =
        commit_col(&vec![ark(1); 1usize << rec.working_log]);
    assert!(verify_read_bgp(&lref, head_root, &atoms, &[], &rec, &bgp).is_err());
}

#[test]
fn dropped_present_distinct_rejects() {
    // A prover who omits the present-set assertion (as if present had < 2 rows) is rejected:
    // the verifier fixes the presence from the public present length.
    let layers = bgp_stack();
    let lref = refs(&layers);
    let atoms = [atom(var(), "http://ex/p1", var())];
    let (rec, mut bgp, _) = prove_read_bgp(&lref, &atoms, &[]).unwrap();
    let head_root = lref.last().unwrap().resolution_root();
    bgp.present_distinct = None;
    assert!(verify_read_bgp(&lref, head_root, &atoms, &[], &rec, &bgp).is_err());
}

// ============================================================
// DELTA-READS — AddedTriple / DeletedTriple over the HEAD delta layer.
//
// Base: {a1 p1 b1, a2 p1 b2, a3 p1 b3, a4 p2 c4}.
// Head delta: remove {a1 p1 b1, a2 p1 b2} (inherited, base terms); add {a5 p1 b5, a6 p1 b6}.
//   head.additions == {a5 p1 b5, a6 p1 b6}  (new terms; ancestor predicate p1)
//   head.removals  == {a1 p1 b1, a2 p1 b2}  (ANCESTOR/base terms — resolved via chain dicts)
// ============================================================

fn delta_read_stack() -> Vec<Arc<LayerCommitment>> {
    let store = open_sync_memory_store();
    let b = store.create_base_layer().unwrap();
    for (s, p, o) in [("a1", "p1", "b1"), ("a2", "p1", "b2"), ("a3", "p1", "b3"), ("a4", "p2", "c4")] {
        b.add_value_triple(ValueTriple::new_node(&iri(s), &iri(p), &iri(o))).unwrap();
    }
    let base = b.commit().unwrap();
    let base_lc = base.proof_commitment().unwrap();
    let child = base.open_write().unwrap();
    child.remove_value_triple(ValueTriple::new_node(&iri("a1"), &iri("p1"), &iri("b1"))).unwrap();
    child.remove_value_triple(ValueTriple::new_node(&iri("a2"), &iri("p1"), &iri("b2"))).unwrap();
    child.add_value_triple(ValueTriple::new_node(&iri("a5"), &iri("p1"), &iri("b5"))).unwrap();
    child.add_value_triple(ValueTriple::new_node(&iri("a6"), &iri("p1"), &iri("b6"))).unwrap();
    let head = child.commit().unwrap();
    let head_lc = head.proof_commitment().unwrap();
    assert_eq!(head_lc.additions.len(), 2, "head must have exactly 2 additions");
    assert_eq!(head_lc.removals.len(), 2, "head must have exactly 2 removals");
    vec![base_lc, head_lc]
}

/// The HEAD layer's committed additions/removals matching `pred`, output [S, O] resolved.
fn delta_group_rows(records: &[ProofRecord], pred: &str) -> Vec<Vec<CanonicalObject>> {
    records
        .iter()
        .filter(|r| r.triple.predicate == pred.as_bytes())
        .map(|r| vec![CanonicalObject::Node(r.triple.subject.clone()), r.triple.object.clone()])
        .collect()
}

#[test]
fn added_triple_predicate_constant() {
    // AddedTriple(?s, p1, ?o): EXACTLY this commit's p1 additions {a5->b5, a6->b6}. Base rows
    // (a1/a2 removed; a3 still present but NOT added) are excluded — an inherited/materialized
    // row is not provable as an added row.
    let layers = delta_read_stack();
    let lref = refs(&layers);
    let head_root = lref.last().unwrap().resolution_root();
    let atoms = [atom(var(), "http://ex/p1", var())];
    let (proof, result) = prove_delta_read(&lref, DeltaSource::Additions, &atoms, &[]).unwrap();
    let n = verify_delta_read(&lref, head_root, DeltaSource::Additions, &atoms, &[], &proof).unwrap();
    assert_eq!(n, 2);
    let resolved = verify_and_resolve_delta_read(
        &lref, head_root, DeltaSource::Additions, &atoms, &[], &proof, &result.columns,
    )
    .unwrap();
    let ground = delta_group_rows(&lref.last().unwrap().additions, "http://ex/p1");
    assert_eq!(ground.len(), 2);
    assert!(bag_eq_rows(rows_of(&resolved), ground), "added rows must equal the head additions");
}

#[test]
fn added_triple_inherited_row_excluded() {
    // p2 appears ONLY on the inherited base row a4->c4 (no addition this commit), so
    // AddedTriple(?s, p2, ?o) is empty: an inherited/materialized row is NOT an AddedTriple.
    let layers = delta_read_stack();
    let lref = refs(&layers);
    let head_root = lref.last().unwrap().resolution_root();
    let atoms = [atom(var(), "http://ex/p2", var())];
    let (proof, result) = prove_delta_read(&lref, DeltaSource::Additions, &atoms, &[]).unwrap();
    let n = verify_delta_read(&lref, head_root, DeltaSource::Additions, &atoms, &[], &proof).unwrap();
    assert_eq!(n, 0, "p2 has no additions this commit (only an inherited base row)");
    let resolved = verify_and_resolve_delta_read(
        &lref, head_root, DeltaSource::Additions, &atoms, &[], &proof, &result.columns,
    )
    .unwrap();
    assert!(rows_of(&resolved).is_empty());
}

#[test]
fn added_triple_subject_constant() {
    // AddedTriple(a5, p1, ?o): keep the object; exactly {b5}.
    let layers = delta_read_stack();
    let lref = refs(&layers);
    let head_root = lref.last().unwrap().resolution_root();
    let atoms = [atom(node_const("http://ex/a5"), "http://ex/p1", var())];
    let (proof, result) = prove_delta_read(&lref, DeltaSource::Additions, &atoms, &[]).unwrap();
    let n = verify_delta_read(&lref, head_root, DeltaSource::Additions, &atoms, &[], &proof).unwrap();
    assert_eq!(n, 1);
    let resolved = verify_and_resolve_delta_read(
        &lref, head_root, DeltaSource::Additions, &atoms, &[], &proof, &result.columns,
    )
    .unwrap();
    assert_eq!(rows_of(&resolved), vec![vec![CanonicalObject::Node(iri("b5").into_bytes())]]);
}

#[test]
fn deleted_triple_predicate_constant() {
    // DeletedTriple(?s, p1, ?o): EXACTLY this commit's p1 removals {a1->b1, a2->b2}. Both
    // reference ANCESTOR (base-defined) terms — resolved via the combined chain dictionaries.
    let layers = delta_read_stack();
    let lref = refs(&layers);
    let head_root = lref.last().unwrap().resolution_root();
    let atoms = [atom(var(), "http://ex/p1", var())];
    let (proof, result) = prove_delta_read(&lref, DeltaSource::Removals, &atoms, &[]).unwrap();
    let n = verify_delta_read(&lref, head_root, DeltaSource::Removals, &atoms, &[], &proof).unwrap();
    assert_eq!(n, 2);
    let resolved = verify_and_resolve_delta_read(
        &lref, head_root, DeltaSource::Removals, &atoms, &[], &proof, &result.columns,
    )
    .unwrap();
    let ground = delta_group_rows(&lref.last().unwrap().removals, "http://ex/p1");
    assert_eq!(ground.len(), 2);
    assert!(bag_eq_rows(rows_of(&resolved), ground), "deleted rows must equal the head removals");
}

// --- Soundness negatives ---

#[test]
fn delta_read_wrong_head_root_rejects() {
    let layers = delta_read_stack();
    let lref = refs(&layers);
    let atoms = [atom(var(), "http://ex/p1", var())];
    let (proof, _) = prove_delta_read(&lref, DeltaSource::Additions, &atoms, &[]).unwrap();
    // Anchor to a non-head (base) root.
    assert!(verify_delta_read(&lref, lref[0].resolution_root(), DeltaSource::Additions, &atoms, &[], &proof).is_err());
    // Corrupt the head root.
    let mut bogus = lref.last().unwrap().resolution_root();
    bogus[0] ^= 0xff;
    assert!(verify_delta_read(&lref, bogus, DeltaSource::Additions, &atoms, &[], &proof).is_err());
}

#[test]
fn delta_read_fabricated_group_hash_rejects() {
    // Substituting a fabricated subject-hash column for the additions group breaks its
    // relocation/lookup binding to the ROOT-committed additions column: a fabricated added
    // row cannot be proved.
    let layers = delta_read_stack();
    let lref = refs(&layers);
    let head_root = lref.last().unwrap().resolution_root();
    let atoms = [atom(var(), "http://ex/p1", var())];
    let (mut proof, _) = prove_delta_read(&lref, DeltaSource::Additions, &atoms, &[]).unwrap();
    proof.group.s.lookup.com_v_hash = commit_col(&vec![ark(4242); 1usize << proof.working_log]);
    assert!(verify_delta_read(&lref, head_root, DeltaSource::Additions, &atoms, &[], &proof).is_err());
}

#[test]
fn delta_read_dropped_row_rejects() {
    let layers = delta_read_stack();
    let lref = refs(&layers);
    let head_root = lref.last().unwrap().resolution_root();
    let atoms = [atom(var(), "http://ex/p1", var())];
    let (proof, _) = prove_delta_read(&lref, DeltaSource::Additions, &atoms, &[]).unwrap();
    let mut forged = proof.clone();
    forged.bgp.atoms[0].m -= 1; // claim one fewer added row
    assert!(verify_delta_read(&lref, head_root, DeltaSource::Additions, &atoms, &[], &forged).is_err());
}

#[test]
fn delta_read_wrong_constant_rejects() {
    // A p1 proof must not verify against a p2 query (the selection is p1-bound).
    let layers = delta_read_stack();
    let lref = refs(&layers);
    let head_root = lref.last().unwrap().resolution_root();
    let atoms = [atom(var(), "http://ex/p1", var())];
    let (proof, _) = prove_delta_read(&lref, DeltaSource::Additions, &atoms, &[]).unwrap();
    let wrong = [atom(var(), "http://ex/p2", var())];
    assert!(verify_delta_read(&lref, head_root, DeltaSource::Additions, &wrong, &[], &proof).is_err());
}

#[test]
fn delta_read_source_mismatch_rejects() {
    // An ADDITIONS proof must not verify as a REMOVALS (DeletedTriple) read: the source is
    // transcript-domain-separated AND binds a different set of root columns.
    let layers = delta_read_stack();
    let lref = refs(&layers);
    let head_root = lref.last().unwrap().resolution_root();
    let atoms = [atom(var(), "http://ex/p1", var())];
    let (proof, _) = prove_delta_read(&lref, DeltaSource::Additions, &atoms, &[]).unwrap();
    assert!(verify_delta_read(&lref, head_root, DeltaSource::Removals, &atoms, &[], &proof).is_err());
}

#[test]
fn delta_read_forged_result_binding_rejects() {
    // A prover who publishes an altered result value fails the re-commitment bind.
    let layers = delta_read_stack();
    let lref = refs(&layers);
    let head_root = lref.last().unwrap().resolution_root();
    let atoms = [atom(var(), "http://ex/p1", var())];
    let (proof, result) = prove_delta_read(&lref, DeltaSource::Additions, &atoms, &[]).unwrap();
    let mut forged = result.columns.clone();
    forged[0][0] = ark(987_654_321); // fabricated subject binding
    assert!(verify_and_resolve_delta_read(
        &lref, head_root, DeltaSource::Additions, &atoms, &[], &proof, &forged,
    )
    .is_err());
}

// --- E2a: term-less delta-read VERIFY path (views-based entries) ---

/// POSITIVE differential (COUNT): a plain conjunctive `AddedTriple(?s, <const p>, ?o)` counts
/// IDENTICALLY through the term-bearing `verify_delta_read(&layers, ..)` and the views entry
/// `verify_delta_read_from_views(&views, ..)` built with a TERM-LESS head view (term-bearing
/// ancestors). Plain BGP (no value/datatype read) is sound term-less, so the count must agree.
#[test]
fn delta_read_from_views_termless_head_matches_term_bearing_count() {
    use crate::proof::commitment::VerifierLayer;
    let layers = delta_read_stack();
    let lref = refs(&layers);
    let head_root = lref.last().unwrap().resolution_root();
    let atoms = [atom(var(), "http://ex/p1", var())];
    let (proof, _result) = prove_delta_read(&lref, DeltaSource::Additions, &atoms, &[]).unwrap();

    // Term-bearing sidecar entry.
    let n_bearing =
        verify_delta_read(&lref, head_root, DeltaSource::Additions, &atoms, &[], &proof).unwrap();

    // Views entry: TERM-LESS head + term-bearing ancestors. The owned term-less commitment must
    // outlive the ref, so bind it in a `let` before building the views vec.
    let head_termless = lref.last().unwrap().verifier_commitment_termless().unwrap();
    assert!(!head_termless.verifier_view().terms_present);
    let mut views: Vec<VerifierCommitmentRef<'_>> =
        lref[..lref.len() - 1].iter().map(|l| l.verifier_view()).collect();
    views.push(head_termless.verifier_view());
    let n_views =
        verify_delta_read_from_views(&views, head_root, DeltaSource::Additions, &atoms, &[], &proof)
            .unwrap();

    assert_eq!(n_bearing, 2);
    assert_eq!(n_bearing, n_views, "term-less views must give the SAME count as term-bearing");
}

/// D2 NEGATIVE: a delta-read with an object-value RANGE atom must FAIL CLOSED when verified over a
/// TERM-LESS view (the order-preserving value/datatype is not authenticated term-less). The SAME
/// range query over all-term-bearing views still verifies (regression).
#[test]
fn delta_read_from_views_range_termless_rejects() {
    use crate::proof::commitment::VerifierLayer;
    // age_stack head additions == {grace age 25, grace name grace_n} (group_len 2).
    let layers = age_stack();
    let lref = refs(&layers);
    let head_root = lref.last().unwrap().resolution_root();
    // AddedTriple(?x, age, ?v)[?v >= 18] over the additions -> exactly {grace 25}.
    let atoms = [range_atom(RangeOp::Ge, 18, None)];
    let (proof, _result) = prove_delta_read(&lref, DeltaSource::Additions, &atoms, &[]).unwrap();

    // Regression: all-term-bearing views verify the range delta-read (D2 guard is a no-op).
    let bearing_views: Vec<VerifierCommitmentRef<'_>> =
        lref.iter().map(|l| l.verifier_view()).collect();
    let n = verify_delta_read_from_views(
        &bearing_views, head_root, DeltaSource::Additions, &atoms, &[], &proof,
    )
    .unwrap();
    assert_eq!(n, 1, "grace age 25 >= 18");

    // D2: a TERM-LESS head view must fail closed on the object-value range.
    let head_termless = lref.last().unwrap().verifier_commitment_termless().unwrap();
    let mut termless_views: Vec<VerifierCommitmentRef<'_>> =
        lref[..lref.len() - 1].iter().map(|l| l.verifier_view()).collect();
    termless_views.push(head_termless.verifier_view());
    assert!(
        verify_delta_read_from_views(
            &termless_views, head_root, DeltaSource::Additions, &atoms, &[], &proof,
        )
        .is_err(),
        "object-value range over a term-less view must fail closed (D2)"
    );
}

/// RESOLUTION fail-close: `verify_and_resolve_delta_read_from_views` on a TERM-LESS head view
/// Errs — term-less id->term resolution needs prover-shipped Merkle openings, and E2a still passes
/// `None` (E2b adds the openings carrier). The added terms (a5/a6/b5/b6) are minted by the head, so
/// they route to the term-less head view and fail closed.
#[test]
fn resolve_delta_read_from_views_termless_head_fail_closes() {
    use crate::proof::commitment::VerifierLayer;
    let layers = delta_read_stack();
    let lref = refs(&layers);
    let head_root = lref.last().unwrap().resolution_root();
    let atoms = [atom(var(), "http://ex/p1", var())];
    let (proof, result) = prove_delta_read(&lref, DeltaSource::Additions, &atoms, &[]).unwrap();

    let head_termless = lref.last().unwrap().verifier_commitment_termless().unwrap();
    let mut views: Vec<VerifierCommitmentRef<'_>> =
        lref[..lref.len() - 1].iter().map(|l| l.verifier_view()).collect();
    views.push(head_termless.verifier_view());
    assert!(
        verify_and_resolve_delta_read_from_views(
            &views, head_root, DeltaSource::Additions, &atoms, &[], &proof, &result.columns,
        )
        .is_err(),
        "term-less resolve without openings must Err (E2a)"
    );
}

// ============================================================
// Increment I3 — delta-stack (reconciled-present) id->term resolution via per-layer,
// id-routed Merkle openings. The differential gate + the routing/root/hash negatives.
// ============================================================

/// PRIMARY correctness gate: over a base+3-delta stack whose dictionary ids are introduced across
/// several layers, chain-routed Merkle resolution must equal the legacy catalog-scan resolution for
/// EVERY node/value AND predicate id — and the happy-path hash cross-check must accept.
#[test]
fn i3_chain_merkle_resolution_equals_catalog_scan_for_every_id() {
    let layers = base_plus_three();
    let lref = refs(&layers);
    let views: Vec<VerifierCommitmentRef<'_>> = lref.iter().map(|l| l.verifier_view()).collect();
    let head = views.last().unwrap();

    // The stack really spreads ids across layers (else the test is vacuous).
    assert!(views.len() == 4, "base + 3 deltas");
    let nv_contributors = views
        .iter()
        .filter(|v| v.state.node_value_count > v.state.parent_node_value_count)
        .count();
    let pred_contributors = views
        .iter()
        .filter(|v| v.state.predicate_count > v.state.parent_predicate_count)
        .count();
    assert!(nv_contributors >= 2, "node/value ids must be introduced across >=2 layers");
    assert!(pred_contributors >= 2, "predicate ids must be introduced across >=2 layers");

    // Every node/value id: Merkle == scan; and Merkle with the correct hash accepts.
    for id in 1..=head.state.node_value_count {
        let merkle = resolve_node_value_id_via_chain_merkle(&views, id, None).unwrap();
        let scan = resolve_node_value_id_via_catalog_scan(&views, id).unwrap();
        assert_eq!(merkle, scan, "node/value id {id}: merkle != scan");
        // Belt-and-suspenders hash cross-check accepts the authentic term hash.
        let checked =
            resolve_node_value_id_via_chain_merkle(&views, id, Some(hash_object(&merkle))).unwrap();
        assert_eq!(checked, merkle);
    }

    // Every predicate id: Merkle == scan; and Merkle with the correct predicate hash accepts.
    for id in 1..=head.state.predicate_count {
        let merkle = resolve_predicate_id_via_chain_merkle(&views, id, None).unwrap();
        let scan = resolve_predicate_id_via_catalog_scan(&views, id).unwrap();
        assert_eq!(merkle, scan, "predicate id {id}: merkle != scan");
        let iri = match &merkle {
            CanonicalObject::Node(bytes) => bytes.clone(),
            _ => panic!("predicate resolves to a node IRI"),
        };
        let checked =
            resolve_predicate_id_via_chain_merkle(&views, id, Some(hash_term(&iri))).unwrap();
        assert_eq!(checked, merkle);
    }
}

/// An id introduced by a delta layer must NOT authenticate against a different layer's root+interval
/// (the routing is structural): `verify_opening`'s interval check re-forces `id ∈ (parent, count]`
/// for the layer it is opened against, so a base id opened against a delta layer is rejected, and a
/// delta id is unroutable when only the base layer is presented.
#[test]
fn i3_wrong_layer_routing_is_rejected() {
    let layers = base_plus_three();
    let lref = refs(&layers);
    let views: Vec<VerifierCommitmentRef<'_>> = lref.iter().map(|l| l.verifier_view()).collect();
    let base = views[0];
    let d1 = views[1];
    assert!(d1.state.node_value_count > d1.state.parent_node_value_count, "d1 introduces ids");

    // A base id (2) opens correctly against the base layer...
    let base_tree = build_node_value_tree(
        base.node_value_catalog,
        base.state.parent_node_value_count,
        base.state.node_value_count,
    )
    .unwrap();
    let opening = base_tree.open(2).unwrap();
    assert!(verify_opening(
        &base.nv_merkle_root,
        base.state.parent_node_value_count,
        base.state.node_value_count,
        2,
        &opening,
    )
    .is_ok());
    // ...but the SAME opening claimed against d1's root+interval is rejected (id 2 <= d1.parent).
    assert!(verify_opening(
        &d1.nv_merkle_root,
        d1.state.parent_node_value_count,
        d1.state.node_value_count,
        2,
        &opening,
    )
    .is_err());

    // A delta id is unroutable to (absent from) the base layer alone, exactly as the scan baseline.
    let delta_id = base.state.node_value_count + 1;
    assert!(resolve_node_value_id_via_catalog_scan(&[base], delta_id).is_err());
    assert!(resolve_node_value_id_via_chain_merkle(&[base], delta_id, None).is_err());
    // With the full chain it resolves against its introducing layer.
    assert!(resolve_node_value_id_via_chain_merkle(&views, delta_id, None).is_ok());
}

/// A tampered per-layer Merkle root is rejected: the resolver rebuilds each layer's tree from its
/// (root-bound) catalog and requires the rebuilt root to equal the carried `nv_merkle_root`.
#[test]
fn i3_tampered_layer_root_is_rejected() {
    let layers = base_plus_three();
    let lref = refs(&layers);
    let base = lref[0].verifier_view();

    // Honest: id 1 resolves.
    assert!(resolve_node_value_id_via_chain_merkle(&[base], 1, None).is_ok());

    // Flip a bit in the carried node/value Merkle root: the rebuilt-root check fails closed.
    let mut bad_root = base.nv_merkle_root;
    bad_root[0] ^= 0x01;
    let tampered = VerifierCommitmentRef { nv_merkle_root: bad_root, ..base };
    assert!(resolve_node_value_id_via_chain_merkle(&[tampered], 1, None).is_err());
}

/// Belt-and-suspenders: the opened leaf's term-hash is cross-checked against a proof-authenticated
/// hash; a mismatch is rejected (node/value AND predicate).
#[test]
fn i3_leaf_term_hash_mismatch_is_rejected() {
    let layers = base_plus_three();
    let lref = refs(&layers);
    let views: Vec<VerifierCommitmentRef<'_>> = lref.iter().map(|l| l.verifier_view()).collect();

    // Node/value: id 1's term hashed against id 2's (distinct) hash is rejected.
    let term1 = resolve_node_value_id_via_chain_merkle(&views, 1, None).unwrap();
    let term2 = resolve_node_value_id_via_chain_merkle(&views, 2, None).unwrap();
    assert_ne!(hash_object(&term1), hash_object(&term2));
    assert!(resolve_node_value_id_via_chain_merkle(&views, 1, Some(hash_object(&term1))).is_ok());
    assert!(resolve_node_value_id_via_chain_merkle(&views, 1, Some(hash_object(&term2))).is_err());

    // Predicate: an authentic-hash check accepts; a wrong-hash check rejects.
    let pred1 = match resolve_predicate_id_via_chain_merkle(&views, 1, None).unwrap() {
        CanonicalObject::Node(bytes) => bytes,
        _ => panic!("predicate resolves to a node IRI"),
    };
    assert!(resolve_predicate_id_via_chain_merkle(&views, 1, Some(hash_term(&pred1))).is_ok());
    let wrong = hash_pred("http://ex/not-a-real-predicate");
    assert_ne!(hash_term(&pred1), wrong);
    assert!(resolve_predicate_id_via_chain_merkle(&views, 1, Some(wrong)).is_err());
}

/// Merkle-dict I2b S3: over the CHAIN resolver, resolving ids on a TERM-LESS base view via
/// prover-shipped Merkle openings yields terms IDENTICAL to the term-bearing rebuild path (node/
/// value AND predicate, both reconstruction branches), the belt-hash still binds, and the path
/// fails CLOSED on a missing carrier / missing per-id opening / belt-hash mismatch. (Term-less is
/// base/rollup-only, so this exercises a single-view chain — the delta stack stays term-bearing
/// and is fail-closed upstream at the re-mint choke.)
#[test]
fn s3_termless_chain_resolution_matches_term_bearing() {
    use crate::proof::commitment::VerifierLayer;
    // A base layer carrying Node + Value objects and a predicate.
    let store = open_sync_memory_store();
    let builder = store.create_base_layer().unwrap();
    for (s, p, o) in [("a", "knows", "b"), ("b", "knows", "c")] {
        builder
            .add_value_triple(ValueTriple::new_node(&iri(s), &iri(p), &iri(o)))
            .unwrap();
    }
    builder
        .add_value_triple(ValueTriple::new_string_value(&iri("a"), &iri("label"), "hello"))
        .unwrap();
    let base = builder.commit().unwrap();
    let base_lc = base.proof_commitment().unwrap();

    let bearing_view = base_lc.verifier_view();
    let termless = base_lc.verifier_commitment_termless().unwrap();
    let tl_view = termless.verifier_view();
    assert!(!tl_view.terms_present);

    let nv_ids: Vec<u64> = (1..=bearing_view.state.node_value_count).collect();
    let pred_ids: Vec<u64> = (1..=bearing_view.state.predicate_count).collect();
    assert!(nv_ids.len() >= 3 && !pred_ids.is_empty());
    // Prover ships openings from the term-bearing view (UNCOMMITTED witness).
    let openings = build_resolution_openings(&[bearing_view], &nv_ids, &pred_ids).unwrap();

    // Node/value: term-less opening resolution == term-bearing rebuild; belt-hash accepts.
    let mut saw_node = false;
    let mut saw_value = false;
    for &id in &nv_ids {
        let bearing = resolve_node_value_id_via_chain_merkle(&[bearing_view], id, None).unwrap();
        let via_openings =
            resolve_node_value_id_via_chain_merkle_with_openings(&[tl_view], id, None, &openings)
                .unwrap();
        assert_eq!(bearing, via_openings, "node/value id {id}: term-less != term-bearing");
        match &bearing {
            CanonicalObject::Node(_) => saw_node = true,
            CanonicalObject::Value { .. } => saw_value = true,
        }
        assert!(resolve_node_value_id_via_chain_merkle_with_openings(
            &[tl_view],
            id,
            Some(hash_object(&via_openings)),
            &openings,
        )
        .is_ok());
    }
    assert!(saw_node && saw_value, "covers Node + Value reconstruction");

    // Predicate: term-less == term-bearing; belt-hash accepts.
    for &id in &pred_ids {
        let bearing = resolve_predicate_id_via_chain_merkle(&[bearing_view], id, None).unwrap();
        let via_openings =
            resolve_predicate_id_via_chain_merkle_with_openings(&[tl_view], id, None, &openings)
                .unwrap();
        assert_eq!(bearing, via_openings, "predicate id {id}: term-less != term-bearing");
        let iri_bytes = match &via_openings {
            CanonicalObject::Node(b) => b.clone(),
            _ => panic!("predicate resolves to a node IRI"),
        };
        assert!(resolve_predicate_id_via_chain_merkle_with_openings(
            &[tl_view],
            id,
            Some(hash_term(&iri_bytes)),
            &openings,
        )
        .is_ok());
    }

    // NEGATIVES on the term-less chain path.
    // (a) No openings carrier ⇒ Err (fail-closed, not empty).
    assert!(resolve_node_value_id_via_chain_merkle(&[tl_view], nv_ids[0], None).is_err());
    // (b) Missing per-id opening ⇒ Err.
    let others: Vec<u64> = nv_ids.iter().copied().filter(|&x| x != nv_ids[0]).collect();
    let missing = build_resolution_openings(&[bearing_view], &others, &[]).unwrap();
    assert!(resolve_node_value_id_via_chain_merkle_with_openings(
        &[tl_view],
        nv_ids[0],
        None,
        &missing,
    )
    .is_err());
    // (c) Belt-hash mismatch on the term-less path ⇒ Err.
    let t0 =
        resolve_node_value_id_via_chain_merkle_with_openings(&[tl_view], nv_ids[0], None, &openings)
            .unwrap();
    let t1 =
        resolve_node_value_id_via_chain_merkle_with_openings(&[tl_view], nv_ids[1], None, &openings)
            .unwrap();
    assert_ne!(hash_object(&t0), hash_object(&t1));
    assert!(resolve_node_value_id_via_chain_merkle_with_openings(
        &[tl_view],
        nv_ids[0],
        Some(hash_object(&t1)),
        &openings,
    )
    .is_err());
}
