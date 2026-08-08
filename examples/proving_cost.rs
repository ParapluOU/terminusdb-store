//! RELEASE-MODE proving-cost benchmark for the Proof-of-WOQL layer BGP argument.
//!
//! Run with:  `cargo run --release --example proving_cost`
//! (optionally pass triple counts:  `cargo run --release --example proving_cost 1000 10000`)
//!
//! A DEBUG build of this is meaningless (the arkworks/Dory field & pairing
//! arithmetic is ~1-2 orders of magnitude slower without optimizations). If you
//! did not pass `--release`, the numbers below are worthless -- the harness
//! prints a loud warning in that case.
//!
//! It measures, per dataset size N (number of triples), for the representative
//! query
//!
//!     t(?x, knows, ?y), t(?x, age, ?a) [?a > 18]   (join on ?x)
//!
//! the following phases:
//!   * setup          -- the ONE-TIME global deterministic Dory transparent setup
//!                        (isolated via a cold-minus-warm commit_with_proof; this
//!                        is the OnceLock `deterministic_prover_setup` cost).
//!   * commit-native  -- building the native store base layer (no proof).
//!   * commit-Dory    -- the EXTRA cost of `commit_with_proof` over `commit`,
//!                        i.e. the commit-time Dory column-commitment MSMs
//!                        (6 triple columns + the dict/value columns).
//!   * prove          -- `prove_layer_bgp` (sumcheck + Dory openings + the range
//!                        bit-decomposition/lookup + the binary join).
//!   * prove(plain)   -- the same 2-atom join WITHOUT the range filter, so
//!                        (prove - prove_plain) isolates the range-gadget cost.
//!   * verify         -- `verify_layer_bgp` (warm; the first verify additionally
//!                        pays a one-time verifier-setup, reported separately).
//!   * proof-size     -- exact serialized size of the query proof, computed by
//!                        walking the (fully public) proof tree and serializing
//!                        every Dory opening / sumcheck / commitment element with
//!                        arkworks `CanonicalSerialize` (the query proof itself
//!                        has no in-tree serde/CanonicalSerialize wrapper yet).

use std::time::{Duration, Instant};

use ark_serialize::CanonicalSerialize;
use dory_pcs::backends::arkworks::{ArkFr, ArkGT};
use dory_pcs::primitives::arithmetic::Field;

use terminus_store::layer::ValueTriple;
use terminus_store::proof::argument::{ColumnOpening, SumcheckProof};
use terminus_store::proof::bgp::{BgpChainProof, BgpStageProof, JoinStep};
use terminus_store::proof::commitment::LayerCommitment;
use terminus_store::proof::count::CountProof;
use terminus_store::proof::dict::DictProof;
use terminus_store::proof::filter::{
    AtomFilterProof, BooleanProductProof, CompactionOnlyProof, FilterProof, FullFilterProof,
    SelectionOnlyProof,
};
use terminus_store::proof::join2::{BinaryJoinCommitments, BinaryJoinProof, BinaryPairsProof};
use terminus_store::proof::membership::MembershipProof;
use terminus_store::proof::query::{
    prove_layer_bgp, verify_layer_bgp, LayerBgpAtom, LayerBgpAtomProof, LayerBgpPlan, LayerBgpProof,
    ObjectRangeSpec, RangeAtomSubproof,
};
use terminus_store::proof::range::{
    DictLookupProof, ObjectRangeCombineProof, ObjectRangeProof, RangeOp, RangeSelectProof,
};
use terminus_store::proof::relocate::RelocationProof;
use terminus_store::proof::sort::SortedProof;
use terminus_store::proof::CanonicalDatatype;
use terminus_store::store::sync::open_sync_memory_store;

const AGE: &[u8] = b"http://ex/age";
const KNOWS: &[u8] = b"http://ex/knows";

// ---------------------------------------------------------------------------
// Proof-size walk. Every proof struct field in the tree is `pub`, and every
// leaf (ArkGT / ArkFr / ColumnOpening=ArkDoryProof) implements CanonicalSerialize.
// ---------------------------------------------------------------------------

fn clen<T: CanonicalSerialize>(x: &T) -> usize {
    let mut buf = Vec::new();
    x.serialize_compressed(&mut buf).expect("serialize");
    buf.len()
}

fn opening_bytes(op: &ColumnOpening) -> usize {
    clen(op)
}

fn sumcheck_bytes(sc: &SumcheckProof) -> usize {
    let mut n = 0;
    for round in &sc.round_polys {
        for f in round {
            n += clen(f);
        }
    }
    for f in &sc.claimed_evals {
        n += clen(f);
    }
    n
}

fn gts(v: &[ArkGT]) -> usize {
    v.iter().map(clen).sum()
}

fn dict_bytes(d: &DictProof) -> usize {
    sumcheck_bytes(&d.sumcheck)
        + opening_bytes(&d.opening)
        + clen(&d.mult_commitment)
        + clen(&d.dstar_commitment)
}

fn reloc_bytes(r: &RelocationProof) -> usize {
    clen(&r.source_eval)
        + clen(&r.target_eval)
        + opening_bytes(&r.source_opening)
        + opening_bytes(&r.target_opening)
}

fn filter_proof_bytes(f: &FilterProof) -> usize {
    sumcheck_bytes(&f.sumcheck) + opening_bytes(&f.opening)
}

fn full_filter_bytes(f: &FullFilterProof) -> usize {
    sumcheck_bytes(&f.sumcheck)
        + opening_bytes(&f.opening)
        + clen(&f.sel_commitment)
        + clen(&f.w_commitment)
        + clen(&f.cstar_commitment)
        + clen(&f.dstar_commitment)
}

fn atom_filter_bytes(f: &AtomFilterProof) -> usize {
    match f {
        AtomFilterProof::Identity => 0,
        AtomFilterProof::Compacted(inner) => full_filter_bytes(inner),
    }
}

fn selection_only_bytes(s: &SelectionOnlyProof) -> usize {
    sumcheck_bytes(&s.sumcheck) + opening_bytes(&s.opening) + clen(&s.w_commitment)
}

fn boolean_product_bytes(b: &BooleanProductProof) -> usize {
    sumcheck_bytes(&b.sumcheck) + opening_bytes(&b.opening)
}

fn compaction_bytes(c: &CompactionOnlyProof) -> usize {
    sumcheck_bytes(&c.sumcheck)
        + opening_bytes(&c.opening)
        + clen(&c.cstar_commitment)
        + clen(&c.dstar_commitment)
}

fn range_select_bytes(r: &RangeSelectProof) -> usize {
    sumcheck_bytes(&r.sumcheck) + opening_bytes(&r.opening) + gts(&r.bit_commitments)
}

fn dict_lookup_bytes(d: &DictLookupProof) -> usize {
    sumcheck_bytes(&d.sumcheck)
        + opening_bytes(&d.opening)
        + clen(&d.com_v_val)
        + clen(&d.com_v_dt)
        + clen(&d.com_w)
        + clen(&d.com_rec_q)
        + clen(&d.com_rec_d)
}

fn object_range_combine_bytes(o: &ObjectRangeCombineProof) -> usize {
    sumcheck_bytes(&o.sumcheck) + opening_bytes(&o.opening)
}

fn object_range_bytes(o: &ObjectRangeProof) -> usize {
    dict_lookup_bytes(&o.lookup)
        + filter_proof_bytes(&o.datatype)
        + clen(&o.com_dt_match)
        + clen(&o.com_dt_w)
        + range_select_bytes(&o.range)
        + clen(&o.com_range_sel)
        + object_range_combine_bytes(&o.combine)
}

fn range_atom_bytes(r: &RangeAtomSubproof) -> usize {
    reloc_bytes(&r.relocate_object)
        + reloc_bytes(&r.relocate_dict_id)
        + reloc_bytes(&r.relocate_dict_value)
        + reloc_bytes(&r.relocate_dict_datatype)
        + clen(&r.com_object_at_d)
        + clen(&r.com_dict_id_at_d)
        + clen(&r.com_dict_value_at_d)
        + clen(&r.com_dict_datatype_at_d)
        + object_range_bytes(&r.object_range)
        + clen(&r.com_range_sel_at_d)
        + reloc_bytes(&r.relocate_range_sel)
        + clen(&r.com_range_sel_at_table)
        + selection_only_bytes(&r.equality)
        + clen(&r.com_equality_sel)
        + boolean_product_bytes(&r.combine)
        + clen(&r.com_combined_sel)
        + compaction_bytes(&r.compaction)
}

fn sorted_bytes(s: &SortedProof) -> usize {
    sumcheck_bytes(&s.sumcheck)
        + opening_bytes(&s.opening)
        + clen(&s.u_shift_commitment)
        + clen(&s.mult_c_commitment)
        + clen(&s.rec_c_commitment)
        + clen(&s.rec_d_commitment)
        + gts(&s.bit_commitments)
}

fn membership_bytes(m: &MembershipProof) -> usize {
    sumcheck_bytes(&m.sumcheck)
        + opening_bytes(&m.opening)
        + clen(&m.rec_u_commitment)
        + clen(&m.rec_of_commitment)
}

fn count_bytes(c: &CountProof) -> usize {
    sumcheck_bytes(&c.sumcheck) + opening_bytes(&c.opening)
}

fn binary_pairs_bytes(p: &BinaryPairsProof) -> usize {
    sumcheck_bytes(&p.membership_sumcheck)
        + opening_bytes(&p.membership_opening)
        + clen(&p.mult_l_commitment)
        + clen(&p.mult_r_commitment)
        + clen(&p.rec_src_l_commitment)
        + clen(&p.rec_cand_l_commitment)
        + clen(&p.rec_src_r_commitment)
        + clen(&p.rec_cand_r_commitment)
        + sorted_bytes(&p.sorted)
}

fn join_commitments_bytes(c: &BinaryJoinCommitments) -> usize {
    clen(&c.r_key)
        + gts(&c.r_other)
        + clen(&c.s_key)
        + gts(&c.s_other)
        + gts(&c.r_other_out)
        + clen(&c.key)
        + gts(&c.s_other_out)
        + clen(&c.rho_l)
        + clen(&c.rho_r)
        + clen(&c.u)
        + clen(&c.w_l)
        + clen(&c.w_r)
}

fn binary_join_bytes(j: &BinaryJoinProof) -> usize {
    binary_pairs_bytes(&j.pairs)
        + sorted_bytes(&j.sorted_u)
        + membership_bytes(&j.membership_l)
        + membership_bytes(&j.membership_r)
        + count_bytes(&j.count)
}

fn bgp_stage_bytes(s: &BgpStageProof) -> usize {
    s.acc_relocations.iter().map(reloc_bytes).sum::<usize>()
        + s.rel_relocations.iter().map(reloc_bytes).sum::<usize>()
        + join_commitments_bytes(&s.commitments)
        + binary_join_bytes(&s.join)
}

fn bgp_chain_bytes(c: &BgpChainProof) -> usize {
    c.stages.iter().map(bgp_stage_bytes).sum()
}

fn atom_proof_bytes(a: &LayerBgpAtomProof) -> usize {
    let mut n = dict_bytes(&a.predicate_membership);
    if let Some(s) = &a.subject_membership {
        n += dict_bytes(s);
    }
    if let Some(o) = &a.object_membership {
        n += dict_bytes(o);
    }
    if let Some(f) = &a.filter {
        n += atom_filter_bytes(f);
    }
    if let Some(r) = &a.range {
        n += range_atom_bytes(r);
    }
    n += gts(&a.kept_commitments);
    n
}

fn proof_bytes(proof: &LayerBgpProof, result_coms: &[ArkGT]) -> usize {
    proof.atoms.iter().map(atom_proof_bytes).sum::<usize>()
        + proof.chain.as_ref().map(bgp_chain_bytes).unwrap_or(0)
        + gts(result_coms)
}

// ---------------------------------------------------------------------------
// Dataset construction.
// ---------------------------------------------------------------------------

/// Add `persons` person subjects, each with a `knows -> person` node edge and an
/// `age -> UInt32` value object (ages cycle 0..79, so ~76% satisfy `age > 18`).
fn populate(builder: &terminus_store::store::sync::SyncStoreLayerBuilder, persons: usize) {
    use tdb_succinct::TdbDataType;
    for i in 0..persons {
        let subj = format!("http://ex/person{i:08}");
        let target = format!("http://ex/person{:08}", (i + 1) % persons);
        builder
            .add_value_triple(ValueTriple::new_node(
                &subj,
                "http://ex/knows",
                &target,
            ))
            .unwrap();
        let age = (i % 80) as u32;
        builder
            .add_value_triple(ValueTriple::new_value(
                &subj,
                "http://ex/age",
                u32::make_entry(&age),
            ))
            .unwrap();
    }
}

/// Build a native-only base layer (no proof), returning the wall time.
fn build_native(persons: usize) -> Duration {
    let store = open_sync_memory_store();
    let builder = store.create_base_layer().unwrap();
    populate(&builder, persons);
    let t = Instant::now();
    let _layer = builder.commit().unwrap();
    t.elapsed()
}

/// Build a base layer WITH proof commitments, returning (wall time, LayerCommitment).
fn build_proof(persons: usize) -> (Duration, std::sync::Arc<LayerCommitment>) {
    let store = open_sync_memory_store();
    let builder = store.create_base_layer().unwrap();
    populate(&builder, persons);
    let t = Instant::now();
    let (layer, _state) = builder.commit_with_proof().unwrap();
    let lc = layer.proof_commitment().unwrap().unwrap();
    (t.elapsed(), lc)
}

fn ranged_plan() -> LayerBgpPlan {
    LayerBgpPlan {
        atoms: vec![
            LayerBgpAtom::all_variables(AGE.to_vec()).with_object_range(ObjectRangeSpec {
                op: RangeOp::Gt,
                datatype: ArkFr::from_u64(CanonicalDatatype::UInt32 as u64),
                c: ArkFr::from_u64(18),
                b_bits: 64,
                second: None,
            }),
            LayerBgpAtom::all_variables(KNOWS.to_vec()),
        ],
        joins: vec![JoinStep { acc_key: 0, rel_key: 0, additional_keys: Vec::new() }],
    }
}

fn plain_plan() -> LayerBgpPlan {
    LayerBgpPlan {
        atoms: vec![
            LayerBgpAtom::all_variables(AGE.to_vec()),
            LayerBgpAtom::all_variables(KNOWS.to_vec()),
        ],
        joins: vec![JoinStep { acc_key: 0, rel_key: 0, additional_keys: Vec::new() }],
    }
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

struct Row {
    n_triples: usize,
    persons: usize,
    log_domain: u32,
    commit_native: Duration,
    commit_dory: Duration,
    prove: Duration,
    prove_plain: Duration,
    verify: Duration,
    proof_size: usize,
    result_rows: usize,
}

fn bench_size(n_triples: usize, prove_runs: usize) -> Row {
    let persons = (n_triples / 2).max(1);
    let real_triples = persons * 2;

    // (1) commit-native (no proof) and commit-with-proof, on separate identical stores.
    let commit_native = build_native(persons);
    let (commit_proof, lc) = build_proof(persons);
    let commit_dory = commit_proof.saturating_sub(commit_native);
    let log_domain = (real_triples.next_power_of_two()).trailing_zeros();

    // (2) representative ranged BGP: prove (median) + verify (warm) + proof size.
    let plan = ranged_plan();
    let mut prove_times = Vec::new();
    let mut last = None;
    for _ in 0..prove_runs {
        let t = Instant::now();
        let out = prove_layer_bgp(&lc, &plan).unwrap();
        prove_times.push(t.elapsed());
        last = Some(out);
    }
    let (proof, coms, m, ids) = last.unwrap();
    let prove = median(prove_times);

    // Warm one verify (first verify pays the one-time verifier setup), then measure.
    verify_layer_bgp(&lc, lc.state.commitment_root, &plan, &ids, &coms, m, &proof).unwrap();
    let mut verify_times = Vec::new();
    for _ in 0..prove_runs {
        let t = Instant::now();
        verify_layer_bgp(&lc, lc.state.commitment_root, &plan, &ids, &coms, m, &proof).unwrap();
        verify_times.push(t.elapsed());
    }
    let verify = median(verify_times);
    let proof_size = proof_bytes(&proof, &coms);

    // (3) plain (no-range) 2-atom join: isolates the range-gadget prove cost.
    let pplan = plain_plan();
    let mut plain_times = Vec::new();
    for _ in 0..prove_runs {
        let t = Instant::now();
        let _ = prove_layer_bgp(&lc, &pplan).unwrap();
        plain_times.push(t.elapsed());
    }
    let prove_plain = median(plain_times);

    Row {
        n_triples: real_triples,
        persons,
        log_domain,
        commit_native,
        commit_dory,
        prove,
        prove_plain,
        verify,
        proof_size,
        result_rows: m,
    }
}

fn main() {
    // Loudly flag a debug build -- the numbers are otherwise meaningless.
    if cfg!(debug_assertions) {
        eprintln!("############################################################");
        eprintln!("## WARNING: DEBUG build. Re-run with --release.            ##");
        eprintln!("## These timings are NOT representative (10-100x slower).  ##");
        eprintln!("############################################################");
    } else {
        println!("[release build confirmed: debug_assertions off]");
    }

    // --- One-time setup isolation --------------------------------------------
    // Warm the store/tokio/allocator with a NATIVE-only tiny build (does NOT
    // trigger the Dory setup OnceLock). Then the first commit_with_proof pays the
    // one-time deterministic Dory prover setup; the second does not. The
    // difference isolates the global setup cost. Likewise the first verify pays
    // the one-time verifier setup.
    let _warm = build_native(1);
    let (cold_commit, _lc0) = build_proof(4);
    let (warm_commit, lc_tiny) = build_proof(4);
    let setup_prover = cold_commit.saturating_sub(warm_commit);

    let tiny_plan = ranged_plan();
    let (tp, tc, tm, tid) = prove_layer_bgp(&lc_tiny, &tiny_plan).unwrap();
    let t = Instant::now();
    verify_layer_bgp(&lc_tiny, lc_tiny.state.commitment_root, &tiny_plan, &tid, &tc, tm, &tp)
        .unwrap();
    let cold_verify = t.elapsed();
    let t = Instant::now();
    verify_layer_bgp(&lc_tiny, lc_tiny.state.commitment_root, &tiny_plan, &tid, &tc, tm, &tp)
        .unwrap();
    let warm_verify = t.elapsed();
    let setup_verifier = cold_verify.saturating_sub(warm_verify);

    println!();
    println!("=== ONE-TIME SETUP (process-global, amortized over all proofs) ===");
    println!(
        "  Dory prover setup (deterministic_prover_setup, 2^{} gens): ~{:.1} ms",
        20usize.div_ceil(2),
        ms(setup_prover)
    );
    println!(
        "  Dory verifier setup (to_verifier_setup, first verify):     ~{:.1} ms",
        ms(setup_verifier)
    );

    // --- Per-size benchmark ---------------------------------------------------
    let sizes: Vec<usize> = {
        let args: Vec<String> = std::env::args().skip(1).collect();
        if args.is_empty() {
            vec![1_000, 10_000, 100_000]
        } else {
            args.iter().map(|a| a.parse().expect("size int")).collect()
        }
    };

    let mut rows = Vec::new();
    for &n in &sizes {
        // Adaptive run count: cheap sizes get 3 runs (median), expensive get 1.
        let runs = if n <= 20_000 { 3 } else { 1 };
        println!("\n[running N~{n} triples, {runs} prove run(s)...]");
        let row = bench_size(n, runs);
        println!(
            "  done: domain=2^{}, result_rows={}, prove={:.1}ms",
            row.log_domain,
            row.result_rows,
            ms(row.prove)
        );
        rows.push(row);
    }

    // --- Table ----------------------------------------------------------------
    println!("\n================================ RESULTS (RELEASE) ================================");
    println!(
        "{:>9} {:>7} {:>6} | {:>10} {:>10} | {:>11} {:>11} {:>9} | {:>10} {:>8}",
        "triples",
        "persons",
        "domain",
        "cmt-native",
        "cmt-Dory",
        "prove",
        "prove-plain",
        "verify",
        "proofsize",
        "rows",
    );
    println!(
        "{:>9} {:>7} {:>6} | {:>10} {:>10} | {:>11} {:>11} {:>9} | {:>10} {:>8}",
        "", "", "", "(ms)", "(ms)", "(ms)", "(ms)", "(ms)", "(bytes)", "",
    );
    println!("{}", "-".repeat(98));
    for r in &rows {
        println!(
            "{:>9} {:>7} {:>6} | {:>10.1} {:>10.1} | {:>11.1} {:>11.1} {:>9.1} | {:>10} {:>8}",
            r.n_triples,
            r.persons,
            format!("2^{}", r.log_domain),
            ms(r.commit_native),
            ms(r.commit_dory),
            ms(r.prove),
            ms(r.prove_plain),
            ms(r.verify),
            r.proof_size,
            r.result_rows,
        );
    }
    println!("{}", "=".repeat(98));
    println!(
        "\nNote: 'cmt-Dory' = commit_with_proof MINUS commit (native) = commit-time Dory\n\
         column-commitment MSMs. 'prove-plain' = same 2-atom join without the range\n\
         filter, so (prove - prove-plain) isolates the range gadget's prove cost.\n\
         'setup' above is process-global and paid ONCE, not per proof."
    );
}
