//! Unit/integration tests for the layer-query proof entry points (moved verbatim).
    use super::*;
    use crate::layer::ValueTriple;
    use crate::proof::argument::commit_column;
    use crate::store::sync::open_sync_memory_store;

    const P1: &[u8] = b"http://ex/p1";
    const P2: &[u8] = b"http://ex/p2";

    fn iri(suffix: &str) -> String {
        format!("http://ex/{suffix}")
    }

    // Build a materialized base layer from (subject, predicate, object) IRI triples.
    fn build_layer(triples: &[(&str, &str, &str)]) -> std::sync::Arc<LayerCommitment> {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for (s, p, o) in triples {
            builder
                .add_value_triple(ValueTriple::new_node(&iri(s), &iri(p), &iri(o)))
                .unwrap();
        }
        let layer = builder.commit().unwrap();
        layer.proof_commitment().unwrap()
    }

    // Independently recompute the expected dense result for a predicate id.
    fn expected_result(layer: &LayerCommitment, pred_id: u64) -> (Vec<ArkFr>, Vec<ArkFr>, usize) {
        let padded = padded_len(layer.materialized.len()).unwrap();
        let mut s_out = vec![f0(); padded];
        let mut o_out = vec![f0(); padded];
        let mut m = 0usize;
        for record in &layer.materialized {
            if record.predicate_id == pred_id {
                s_out[m] = ArkFr::from_u64(record.subject_id);
                o_out[m] = ArkFr::from_u64(record.object_id);
                m += 1;
            }
        }
        (s_out, o_out, m)
    }

    fn sample_layer() -> std::sync::Arc<LayerCommitment> {
        build_layer(&[
            ("a1", "p1", "b1"),
            ("a2", "p1", "b2"),
            ("a3", "p1", "b3"),
            ("a4", "p1", "b4"),
            ("c1", "p2", "d1"),
            ("c2", "p2", "d2"),
        ])
    }

    #[test]
    fn query_commitment_convention_matches_layer_commitment_columns() {
        let lc = build_layer(&[("a1", "p1", "b1"), ("a2", "p1", "b2"), ("c1", "p2", "d1")]);
        let (s, p, o, _n) = materialized_layer_columns(&lc).unwrap();
        assert_eq!(
            commit_column(&s).unwrap(),
            lc.materialized_column_commitments[0]
        );
        assert_eq!(
            commit_column(&p).unwrap(),
            lc.materialized_column_commitments[1]
        );
        assert_eq!(
            commit_column(&o).unwrap(),
            lc.materialized_column_commitments[2]
        );
    }

    #[test]
    fn prove_and_verify_by_iri_with_exact_result() {
        let lc = sample_layer();
        let (proof, com_sout, com_oout, m, resolved_id) = prove_layer_filter(&lc, P1).unwrap();
        // The verifier is given ONLY the IRI bytes + the revealed id (authenticated
        // by membership) -- never an out-of-band id.
        verify_layer_filter(
            &lc,
            lc.state.commitment_root,
            P1,
            resolved_id,
            &com_sout,
            &com_oout,
            m,
            &proof,
        )
        .unwrap();

        // Result is EXACTLY the p1 rows (count + contents).
        assert_eq!(m, 4);
        let (exp_s, exp_o, exp_m) = expected_result(&lc, resolved_id);
        assert_eq!(exp_m, m);
        assert_eq!(commit_column(&exp_s).unwrap(), com_sout);
        assert_eq!(commit_column(&exp_o).unwrap(), com_oout);
    }

    #[test]
    fn wrong_revealed_id_for_the_iri_is_rejected() {
        let lc = sample_layer();
        let (proof, com_sout, com_oout, m, resolved_id) = prove_layer_filter(&lc, P1).unwrap();
        // Claim a different id for the same IRI -> membership fails.
        let wrong = resolved_id.wrapping_add(1);
        assert!(verify_layer_filter(
            &lc,
            lc.state.commitment_root,
            P1,
            wrong,
            &com_sout,
            &com_oout,
            m,
            &proof
        )
        .is_err());
    }

    #[test]
    fn substituted_id_from_a_different_predicate_is_rejected() {
        let lc = sample_layer();
        let (proof, com_sout, com_oout, m, _p1_id) = prove_layer_filter(&lc, P1).unwrap();
        // p2's genuine id, but paired with p1's IRI: (hash_term(p1), id_p2) is not a
        // dictionary row -> membership fails.
        let p2_id = resolve_predicate_id(&lc, P2).unwrap();
        assert!(verify_layer_filter(
            &lc,
            lc.state.commitment_root,
            P1,
            p2_id,
            &com_sout,
            &com_oout,
            m,
            &proof
        )
        .is_err());
    }

    #[test]
    fn absent_predicate_iri_fails_closed() {
        let lc = sample_layer();
        let err = prove_layer_filter(&lc, b"http://ex/absent").unwrap_err();
        assert!(err
            .to_string()
            .contains("predicate not in layer dictionary"));
    }

    #[test]
    fn tampered_result_is_rejected_under_the_combined_transcript() {
        let lc = sample_layer();
        let (proof, com_sout, com_oout, m, resolved_id) = prove_layer_filter(&lc, P1).unwrap();
        // Wrong S_out commitment (use the O_out one) -> filter opening fails.
        assert!(verify_layer_filter(
            &lc,
            lc.state.commitment_root,
            P1,
            resolved_id,
            &com_oout,
            &com_oout,
            m,
            &proof
        )
        .is_err());
        // Honest still verifies.
        verify_layer_filter(
            &lc,
            lc.state.commitment_root,
            P1,
            resolved_id,
            &com_sout,
            &com_oout,
            m,
            &proof,
        )
        .unwrap();
    }

    #[test]
    fn child_add_and_remove_proves_against_complete_head_view() {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for (s, p, o) in [("a1", "p1", "b1"), ("a2", "p1", "b2"), ("a3", "p1", "b3")] {
            builder
                .add_value_triple(ValueTriple::new_node(&iri(s), &iri(p), &iri(o)))
                .unwrap();
        }
        let base = builder.commit().unwrap();
        let child = base.open_write().unwrap();
        child
            .remove_value_triple(ValueTriple::new_node(&iri("a1"), &iri("p1"), &iri("b1")))
            .unwrap();
        child
            .add_value_triple(ValueTriple::new_node(&iri("a4"), &iri("p1"), &iri("b4")))
            .unwrap();
        let head = child.commit().unwrap();
        let lc = head.proof_commitment().unwrap();
        assert_eq!(lc.state.addition_count, 1);
        assert_eq!(lc.state.removal_count, 1);
        // COMMITMENT DIET (Wave 0): a delta layer no longer carries a materialized head
        // view (empty, count 0) and is not directly query-provable — its head view and
        // inherited constants are not locally committed. Direct proving reject-closes;
        // the reconciled delta-stack path (Wave 2/3) will re-enable delta-head queries.
        assert!(lc.state.parent_state_root.is_some());
        assert!(lc.materialized.is_empty());
        assert_eq!(lc.state.materialized_count, 0);
        assert!(prove_layer_filter(&lc, P1).is_err());
    }

    #[test]
    fn child_additions_prove_with_inherited_parent_rows() {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for (s, p, o) in [("a1", "p1", "b1"), ("a2", "p1", "b2")] {
            builder
                .add_value_triple(ValueTriple::new_node(&iri(s), &iri(p), &iri(o)))
                .unwrap();
        }
        let base = builder.commit().unwrap();
        let child = base.open_write().unwrap();
        for (s, o) in [("a3", "b3"), ("a4", "b4")] {
            child
                .add_value_triple(ValueTriple::new_node(&iri(s), &iri("pn"), &iri(o)))
                .unwrap();
        }
        let head = child.commit().unwrap();
        let lc = head.proof_commitment().unwrap();
        assert_eq!(lc.state.removal_count, 0);
        assert!(lc.state.parent_state_root.is_some());
        assert_eq!(lc.state.addition_count, 2);
        // COMMITMENT DIET (Wave 0): no materialized head view is committed for a delta
        // layer, and it is not directly query-provable (reject-close).
        assert!(lc.materialized.is_empty());
        assert_eq!(lc.state.materialized_count, 0);
        let pn: &[u8] = b"http://ex/pn";
        assert!(prove_layer_filter(&lc, pn).is_err());
    }

    // ---- Delta reads: AddedTriple / DeletedTriple over the layer's own delta files ----

    // Base p1 { a1..a4 -> b1..b4 }; child ADDS { a5,a6 } and REMOVES { a1,a2 }, all p1.
    // So additions == {a5,a6}, removals == {a1,a2}, materialized == {a3,a4,a5,a6}.
    fn delta_layer() -> std::sync::Arc<LayerCommitment> {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for s in ["a1", "a2", "a3", "a4"] {
            let o = format!("b{}", &s[1..]);
            builder
                .add_value_triple(ValueTriple::new_node(&iri(s), &iri("p1"), &iri(&o)))
                .unwrap();
        }
        let base = builder.commit().unwrap();
        let child = base.open_write().unwrap();
        for s in ["a1", "a2"] {
            let o = format!("b{}", &s[1..]);
            child
                .remove_value_triple(ValueTriple::new_node(&iri(s), &iri("p1"), &iri(&o)))
                .unwrap();
        }
        for s in ["a5", "a6"] {
            let o = format!("b{}", &s[1..]);
            child
                .add_value_triple(ValueTriple::new_node(&iri(s), &iri("p1"), &iri(&o)))
                .unwrap();
        }
        let head = child.commit().unwrap();
        head.proof_commitment().unwrap()
    }

    // COMMITMENT DIET (Wave 0): a direct query `t(?s, p1, ?o)` over a DELTA layer
    // reject-closes for EVERY column source. Additions/removals columns are still
    // root-bound, but a delta layer no longer commits its materialized head view and
    // commits only the INCREMENTAL dictionary it introduced, so an inherited constant
    // (here p1, defined in the base) cannot be resolved locally. The reconciled
    // delta-stack path (Wave 2/3) will re-enable delta-head queries by authenticating
    // terms across the chain; today, roll up/squash to a base layer to query directly.
    fn assert_delta_query_rejects(lc: &LayerCommitment, source: ColumnSource) {
        assert!(lc.state.parent_state_root.is_some());
        let plan = LayerBgpPlan::with_source(
            vec![LayerBgpAtom::all_variables(P1.to_vec())],
            Vec::new(),
            source,
        );
        assert!(prove_layer_bgp_with_result(lc, &plan).is_err());
    }

    #[test]
    fn delta_added_and_removed_triples_reject_direct_query_after_diet() {
        let lc = delta_layer();
        assert!(lc.state.parent_state_root.is_some());
        assert_eq!(lc.state.addition_count, 2);
        assert_eq!(lc.state.removal_count, 2);
        // No materialized head-view dump on a delta layer any more.
        assert!(lc.materialized.is_empty());
        assert_eq!(lc.state.materialized_count, 0);
        // Every source reject-closes (Additions/Removals/Materialized alike).
        assert_delta_query_rejects(&lc, ColumnSource::Additions);
        assert_delta_query_rejects(&lc, ColumnSource::Removals);
        assert_delta_query_rejects(&lc, ColumnSource::Materialized);
    }

    #[test]
    fn inherited_row_is_not_an_added_or_deleted_triple() {
        // (Wave 0) A delta layer is not directly query-provable; every source reject-closes.
        let lc = delta_layer();
        assert_delta_query_rejects(&lc, ColumnSource::Additions);
        assert_delta_query_rejects(&lc, ColumnSource::Removals);
    }

    #[test]
    fn delta_source_substitution_fails_verification() {
        // (Wave 0) Direct delta proving reject-closes at the BGP entry before any
        // source-substitution attack can even be set up.
        let lc = delta_layer();
        for source in [
            ColumnSource::Additions,
            ColumnSource::Removals,
            ColumnSource::Materialized,
        ] {
            let plan = LayerBgpPlan::with_source(
                vec![LayerBgpAtom::all_variables(P1.to_vec())],
                Vec::new(),
                source,
            );
            assert!(prove_layer_bgp_with_result(&lc, &plan).is_err());
        }
    }

    #[test]
    fn delta_filter_entry_rejects_after_diet() {
        // The legacy Q1 filter entry also reject-closes over a delta layer for every source.
        let lc = delta_layer();
        for source in [
            ColumnSource::Additions,
            ColumnSource::Removals,
            ColumnSource::Materialized,
        ] {
            assert!(prove_layer_filter_with_source(
                &StoreProverContext::cpu(),
                &lc,
                source,
                P1
            )
            .is_err());
        }
    }

    // ---- Cross-source delta joins: one BGP whose atoms read DIFFERENT column sets ----

    const PADD: &[u8] = b"http://ex/padd";
    const PMAT: &[u8] = b"http://ex/pmat";
    const PREM: &[u8] = b"http://ex/prem";

    // Base pmat edges y{i}->z{i} (inherited into materialized); child ADDS padd edges
    // x{i}->y{i} (into BOTH the additions delta AND the materialized head).
    fn cross_source_add_mat_layer() -> std::sync::Arc<LayerCommitment> {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for i in [1, 2] {
            builder
                .add_value_triple(ValueTriple::new_node(
                    &iri(&format!("y{i}")),
                    &iri("pmat"),
                    &iri(&format!("z{i}")),
                ))
                .unwrap();
        }
        let base = builder.commit().unwrap();
        let child = base.open_write().unwrap();
        for i in [1, 2] {
            child
                .add_value_triple(ValueTriple::new_node(
                    &iri(&format!("x{i}")),
                    &iri("padd"),
                    &iri(&format!("y{i}")),
                ))
                .unwrap();
        }
        let head = child.commit().unwrap();
        head.proof_commitment().unwrap()
    }

    // Base prem edges y{i}->z{i}; child ADDS padd edges x{i}->y{i} and REMOVES the prem edges
    // (so additions == {x_i->y_i}, removals == {y_i->z_i}).
    fn cross_source_add_rem_layer() -> std::sync::Arc<LayerCommitment> {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for i in [1, 2] {
            builder
                .add_value_triple(ValueTriple::new_node(
                    &iri(&format!("y{i}")),
                    &iri("prem"),
                    &iri(&format!("z{i}")),
                ))
                .unwrap();
        }
        let base = builder.commit().unwrap();
        let child = base.open_write().unwrap();
        for i in [1, 2] {
            child
                .add_value_triple(ValueTriple::new_node(
                    &iri(&format!("x{i}")),
                    &iri("padd"),
                    &iri(&format!("y{i}")),
                ))
                .unwrap();
            child
                .remove_value_triple(ValueTriple::new_node(
                    &iri(&format!("y{i}")),
                    &iri("prem"),
                    &iri(&format!("z{i}")),
                ))
                .unwrap();
        }
        let head = child.commit().unwrap();
        head.proof_commitment().unwrap()
    }

    // AddedTriple(?x, padd, ?y) [Additions] & Triple(?y, pmat, ?z) [Materialized], joined on ?y
    // (atom0 object column == atom1 subject column). Note the two atoms have DIFFERENT committed
    // row counts (additions=2 -> domain 2, materialized=4 -> domain 4), so this also exercises
    // the join gadget's cross-domain relocation.
    fn additions_join_materialized_plan() -> LayerBgpPlan {
        LayerBgpPlan {
            // Plan default is Materialized (mixed plan); atom0 overrides to Additions.
            source: ColumnSource::Materialized,
            atoms: vec![
                LayerBgpAtom::all_variables(PADD.to_vec()).with_source(ColumnSource::Additions),
                LayerBgpAtom::all_variables(PMAT.to_vec()).with_source(ColumnSource::Materialized),
            ],
            joins: vec![JoinStep {
                acc_key: 1,
                rel_key: 0,
                additional_keys: Vec::new(),
            }],
        }
    }

    #[test]
    fn cross_source_additions_join_materialized_rejects_after_diet() {
        // (Wave 0) A cross-source join reading a delta layer's Materialized head view is no
        // longer provable — the head view is not committed and a delta layer is not directly
        // query-provable. It reject-closes at the BGP entry.
        let lc = cross_source_add_mat_layer();
        assert!(lc.state.parent_state_root.is_some());
        assert_eq!(lc.state.addition_count, 2);
        assert!(lc.materialized.is_empty());
        assert_eq!(lc.state.materialized_count, 0);
        assert!(prove_layer_bgp_with_result(&lc, &additions_join_materialized_plan()).is_err());
    }

    #[test]
    fn cross_source_additions_join_removals_rejects_after_diet() {
        // (Wave 0) Even an additions⋈removals join (both delta sources ARE root-bound) is
        // rejected as a DIRECT delta-layer query. The reconciled delta-stack path (Wave 2/3)
        // will carry cross-source delta joins over an authenticated delta stack instead.
        let lc = cross_source_add_rem_layer();
        assert!(lc.state.parent_state_root.is_some());
        assert_eq!(lc.state.addition_count, 2);
        assert_eq!(lc.state.removal_count, 2);
        let plan = LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![
                LayerBgpAtom::all_variables(PADD.to_vec()).with_source(ColumnSource::Additions),
                LayerBgpAtom::all_variables(PREM.to_vec()).with_source(ColumnSource::Removals),
            ],
            joins: vec![JoinStep {
                acc_key: 1,
                rel_key: 0,
                additional_keys: Vec::new(),
            }],
        };
        assert!(prove_layer_bgp_with_result(&lc, &plan).is_err());
    }

    // ---- Q2: 2-atom graph-pattern (self-join) end-to-end ----

    const KNOWS: &[u8] = b"http://ex/knows";

    // Independently recompute the complete 2-hop result for a predicate id, from the
    // layer's additions in stored order (matching the filter output's F ordering).
    fn expected_join(
        lc: &LayerCommitment,
        pred_id: u64,
    ) -> (Vec<ArkFr>, Vec<ArkFr>, Vec<ArkFr>, usize, usize) {
        let mut sf = Vec::new();
        let mut of = Vec::new();
        for record in &lc.materialized {
            if record.predicate_id == pred_id {
                sf.push(ArkFr::from_u64(record.subject_id));
                of.push(ArkFr::from_u64(record.object_id));
            }
        }
        let (x, y, z, _rl, _rr, u, _wl, _wr, m_j) = compute_self_join(&sf, &of);
        let n_join = self_join_domain(sf.len(), m_j, u.len()).unwrap();
        (x, y, z, n_join, m_j)
    }

    #[test]
    fn graph_pattern2_two_hop_knows_paths_end_to_end() {
        // knows: a->b, b->c, c->d, a->e (e a sink). 2-hop paths (x->y->z): (a,b,c) and
        // (b,c,d). One unrelated "likes" edge exercises the filter's predicate isolation.
        let lc = build_layer(&[
            ("a", "knows", "b"),
            ("b", "knows", "c"),
            ("c", "knows", "d"),
            ("a", "knows", "e"),
            ("a", "likes", "zz"),
        ]);
        let (proof, com_x, com_y, com_z, m_j, id) = prove_layer_graph_pattern2(&lc, KNOWS).unwrap();
        // Verifier sees ONLY the IRI + revealed id + result commitments.
        verify_layer_graph_pattern2(
            &lc,
            lc.state.commitment_root,
            KNOWS,
            id,
            &com_x,
            &com_y,
            &com_z,
            m_j,
            &proof,
        )
        .unwrap();

        // Result is EXACTLY the two 2-hop paths (count + contents).
        assert_eq!(m_j, 2);
        let (x, y, z, n_join, exp_m) = expected_join(&lc, id);
        assert_eq!(exp_m, m_j);
        assert_eq!(commit_column(&pad(&x, n_join)).unwrap(), com_x);
        assert_eq!(commit_column(&pad(&y, n_join)).unwrap(), com_y);
        assert_eq!(commit_column(&pad(&z, n_join)).unwrap(), com_z);
    }

    #[test]
    fn graph_pattern2_mismatched_result_or_count_is_rejected() {
        let lc = build_layer(&[
            ("a", "knows", "b"),
            ("b", "knows", "c"),
            ("c", "knows", "d"),
        ]);
        let (proof, com_x, com_y, com_z, m_j, id) = prove_layer_graph_pattern2(&lc, KNOWS).unwrap();
        // Wrong result commitment (pass Com Z for Com X) -> mismatch with proof output.
        assert!(verify_layer_graph_pattern2(
            &lc,
            lc.state.commitment_root,
            KNOWS,
            id,
            &com_z,
            &com_y,
            &com_z,
            m_j,
            &proof
        )
        .is_err());
        // Wrong claimed count.
        assert!(verify_layer_graph_pattern2(
            &lc,
            lc.state.commitment_root,
            KNOWS,
            id,
            &com_x,
            &com_y,
            &com_z,
            m_j + 1,
            &proof
        )
        .is_err());
        // Honest verifies.
        verify_layer_graph_pattern2(
            &lc,
            lc.state.commitment_root,
            KNOWS,
            id,
            &com_x,
            &com_y,
            &com_z,
            m_j,
            &proof,
        )
        .unwrap();
    }

    #[test]
    fn graph_pattern2_fabricated_join_output_is_rejected() {
        // Fabricate a bogus X-commitment and pass it consistently (defeats the equality
        // gate); the self-join column opening still binds X to the sumcheck -> REJECT.
        let lc = build_layer(&[
            ("a", "knows", "b"),
            ("b", "knows", "c"),
            ("c", "knows", "d"),
        ]);
        let (proof, _com_x, com_y, com_z, m_j, id) =
            prove_layer_graph_pattern2(&lc, KNOWS).unwrap();
        let (_, _, _, n_join, _) = expected_join(&lc, id);
        let fake_x = commit_column(&pad(&[ArkFr::from_u64(424242)], n_join)).unwrap();
        let mut bad = proof.clone();
        bad.join_commitments.x = fake_x;
        assert!(verify_layer_graph_pattern2(
            &lc,
            lc.state.commitment_root,
            KNOWS,
            id,
            &fake_x,
            &com_y,
            &com_z,
            m_j,
            &bad
        )
        .is_err());
    }

    #[test]
    fn graph_pattern2_no_two_hop_paths_empty_verifies() {
        // Disjoint edges a->b and c->d: no y with x->y and y->z, so the result is empty.
        let lc = build_layer(&[("a", "knows", "b"), ("c", "knows", "d")]);
        let (proof, com_x, com_y, com_z, m_j, id) = prove_layer_graph_pattern2(&lc, KNOWS).unwrap();
        assert_eq!(m_j, 0);
        verify_layer_graph_pattern2(
            &lc,
            lc.state.commitment_root,
            KNOWS,
            id,
            &com_x,
            &com_y,
            &com_z,
            m_j,
            &proof,
        )
        .unwrap();
    }

    #[test]
    fn graph_pattern2_tampered_layer_root_is_rejected() {
        let lc = build_layer(&[
            ("a", "knows", "b"),
            ("b", "knows", "c"),
            ("c", "knows", "d"),
        ]);
        let good_root = lc.state.commitment_root;
        let (proof, com_x, com_y, com_z, m_j, id) = prove_layer_graph_pattern2(&lc, KNOWS).unwrap();
        // Corrupt the anchor: a layer whose stored root does not bind its columns.
        let mut tampered = (*lc).clone();
        tampered.state.commitment_root[0] ^= 0xff;
        // Even pinned to the (correct) external root, the tampered layer's stored root
        // no longer matches it -> the FIX 4 anchor rejects.
        assert!(verify_layer_graph_pattern2(
            &tampered, good_root, KNOWS, id, &com_x, &com_y, &com_z, m_j, &proof
        )
        .is_err());
    }

    #[test]
    fn graph_pattern2_wrong_expected_root_is_rejected() {
        // FIX 4 (external-root anchor): an honest proof against an honest layer must
        // still REJECT when the verifier pins a DIFFERENT externally-trusted root.
        let lc = build_layer(&[
            ("a", "knows", "b"),
            ("b", "knows", "c"),
            ("c", "knows", "d"),
        ]);
        let (proof, com_x, com_y, com_z, m_j, id) = prove_layer_graph_pattern2(&lc, KNOWS).unwrap();
        let mut wrong_root = lc.state.commitment_root;
        wrong_root[0] ^= 0xff;
        assert!(verify_layer_graph_pattern2(
            &lc, wrong_root, KNOWS, id, &com_x, &com_y, &com_z, m_j, &proof
        )
        .is_err());
        // The correct external root still verifies.
        verify_layer_graph_pattern2(
            &lc,
            lc.state.commitment_root,
            KNOWS,
            id,
            &com_x,
            &com_y,
            &com_z,
            m_j,
            &proof,
        )
        .unwrap();
    }

    // ---- Q_k: general k-atom layer BGP (left-deep chain) end-to-end ----

    const LIKES: &[u8] = b"http://ex/likes";
    const OWNS: &[u8] = b"http://ex/owns";

    fn atom(term: &[u8]) -> LayerBgpAtom {
        LayerBgpAtom::all_variables(term.to_vec())
    }

    // The two (subject, object) columns of one atom relation, filtered from the layer's
    // additions in stored order (matching the filter output ordering).
    fn atom_relation(lc: &LayerCommitment, pred_id: u64) -> Vec<Vec<ArkFr>> {
        let mut s = Vec::new();
        let mut o = Vec::new();
        for rec in &lc.materialized {
            if rec.predicate_id == pred_id {
                s.push(ArkFr::from_u64(rec.subject_id));
                o.push(ArkFr::from_u64(rec.object_id));
            }
        }
        vec![s, o]
    }

    #[test]
    fn delivered_result_columns_are_bound_to_the_proven_commitments() {
        let lc = build_layer(&[
            ("a", "knows", "b"),
            ("b", "knows", "c"),
            ("x", "likes", "y"),
        ]);
        let plan = LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![atom(KNOWS)],
            joins: Vec::new(),
        };
        let result = prove_layer_bgp_with_result(&lc, &plan).unwrap();
        verify_layer_bgp(
            &lc,
            lc.state.commitment_root,
            &plan,
            &result.predicate_ids,
            &result.result_commitments,
            result.result_len,
            &result.proof,
        )
        .unwrap();

        let columns = result.real_columns();
        verify_result_columns(
            &result.result_commitments,
            &columns,
            result.result_len,
            result.result_log_domain,
        )
        .unwrap();

        let mut tampered = columns;
        tampered[0][0] = tampered[0][0] + ArkFr::one();
        assert!(verify_result_columns(
            &result.result_commitments,
            &tampered,
            result.result_len,
            result.result_log_domain,
        )
        .is_err());
    }

    #[test]
    fn delivered_lexical_objects_are_bound_to_authenticated_result_ids() {
        let lc = build_layer(&[("a", "knows", "b"), ("b", "knows", "c")]);
        let plan = LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![atom(KNOWS)],
            joins: Vec::new(),
        };
        let result = prove_layer_bgp_with_result(&lc, &plan).unwrap();
        let columns = result.real_columns();
        let objects =
            resolve_node_value_result_objects(&lc, lc.state.commitment_root, &columns).unwrap();

        assert_eq!(objects.len(), 2);
        assert_eq!(objects[0][0], CanonicalObject::Node(iri("a").into_bytes()));
        assert_eq!(objects[1][0], CanonicalObject::Node(iri("b").into_bytes()));
        verify_node_value_result_objects(&lc, lc.state.commitment_root, &columns, &objects).unwrap();

        let mut tampered = objects.clone();
        tampered[1][0] = CanonicalObject::Node(iri("c").into_bytes());
        assert!(
            verify_node_value_result_objects(
                &lc,
                lc.state.commitment_root,
                &columns,
                &tampered,
            )
            .is_err()
        );

        let mut wrong_root = lc.state.commitment_root;
        wrong_root[0] ^= 0xff;
        assert!(resolve_node_value_result_objects(&lc, wrong_root, &columns).is_err());

        // The serialized catalog is only a claimed opening. Changing it without
        // changing the root-bound commitment must not change the resolved value.
        let mut tampered_catalog = (*lc).clone();
        tampered_catalog.node_value_catalog[0].triple.object =
            CanonicalObject::Node(iri("forged").into_bytes());
        assert!(
            resolve_node_value_result_objects(
                &tampered_catalog,
                lc.state.commitment_root,
                &columns,
            )
            .is_err()
        );
    }

    #[test]
    fn nullable_result_descriptors_are_authoritative() {
        let lc = build_layer(&[("a", "knows", "b")]);
        let node_id = lc
            .node_value_catalog
            .iter()
            .find(|record| record.triple.object == CanonicalObject::Node(iri("a").into_bytes()))
            .unwrap()
            .object_id;
        let predicate_id = resolve_predicate_id(&lc, KNOWS).unwrap();
        let columns = vec![
            vec![ArkFr::from_u64(node_id)],
            vec![ArkFr::from_u64(predicate_id)],
            vec![ArkFr::zero()],
        ];
        let descriptors = vec![
            ResultColumnDescriptor {
                namespace: ResultIdNamespace::NodeValue,
                nullable: false,
            },
            ResultColumnDescriptor {
                namespace: ResultIdNamespace::Predicate,
                nullable: false,
            },
            ResultColumnDescriptor {
                namespace: ResultIdNamespace::NodeValue,
                nullable: true,
            },
        ];
        let values = resolve_result_values(
            &lc,
            lc.state.commitment_root,
            &columns,
            &descriptors,
        )
        .unwrap();
        assert_eq!(
            values[0][0],
            CanonicalResultValue::Object(CanonicalObject::Node(iri("a").into_bytes()))
        );
        assert_eq!(
            values[1][0],
            CanonicalResultValue::Object(CanonicalObject::Node(iri("knows").into_bytes()))
        );
        assert_eq!(values[2][0], CanonicalResultValue::Null);

        let mut nonnullable = descriptors.clone();
        nonnullable[2].nullable = false;
        assert!(resolve_result_values(
            &lc,
            lc.state.commitment_root,
            &columns,
            &nonnullable,
        )
        .is_err());

        let mut forged = columns;
        forged[2][0] = ArkFr::from_u64(u64::MAX);
        assert!(resolve_result_values(
            &lc,
            lc.state.commitment_root,
            &forged,
            &descriptors,
        )
        .is_err());
    }

    // Independently recompute the complete BGP result columns (real length) and the final
    // join domain, folding the audited compute_binary_join over the plan's join steps.
    fn expected_layer_bgp(
        lc: &LayerCommitment,
        ids: &[u64],
        joins: &[JoinStep],
    ) -> (Vec<Vec<ArkFr>>, usize) {
        use crate::proof::join2::{binary_join_domain, compute_binary_join};
        let split = |cols: &[Vec<ArkFr>], key: usize| -> (Vec<ArkFr>, Vec<Vec<ArkFr>>) {
            let k = cols[key].clone();
            let other = cols
                .iter()
                .enumerate()
                .filter(|(c, _)| *c != key)
                .map(|(_, c)| c.clone())
                .collect();
            (k, other)
        };
        let mut acc = atom_relation(lc, ids[0]);
        let mut acc_len = acc[0].len();
        let mut domain = padded_len(lc.materialized.len()).unwrap();
        for (j, step) in joins.iter().enumerate() {
            let rel = atom_relation(lc, ids[j + 1]);
            let rel_len = rel[0].len();
            let (r_key, r_other) = split(&acc, step.acc_key);
            let (s_key, s_other) = split(&rel, step.rel_key);
            let r_refs: Vec<&[ArkFr]> = r_other.iter().map(|c| c.as_slice()).collect();
            let s_refs: Vec<&[ArkFr]> = s_other.iter().map(|c| c.as_slice()).collect();
            let w = compute_binary_join(&r_key, &r_refs, &s_key, &s_refs);
            domain = binary_join_domain(acc_len, rel_len, w.m_j, w.n_u).unwrap();
            let mut new_acc = w.r_other_out.clone();
            new_acc.push(w.key.clone());
            new_acc.extend(w.s_other_out.clone());
            acc = new_acc;
            acc_len = w.m_j;
        }
        (acc, domain)
    }

    #[test]
    fn layer_bgp_three_atom_path_end_to_end() {
        // knows: a->b, c->b, e->f;  likes: b->g, f->h;  owns: g->k, h->m.
        // BGP t(?x,knows,?y), t(?y,likes,?z), t(?z,owns,?w). Paths (x,y,z,w):
        //   (a,b,g,k), (c,b,g,k), (e,f,h,m).
        let lc = build_layer(&[
            ("a", "knows", "b"),
            ("c", "knows", "b"),
            ("e", "knows", "f"),
            ("b", "likes", "g"),
            ("f", "likes", "h"),
            ("g", "owns", "k"),
            ("h", "owns", "m"),
        ]);
        let plan = LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![atom(KNOWS), atom(LIKES), atom(OWNS)],
            joins: vec![
                JoinStep {
                    acc_key: 1,
                    rel_key: 0,
                    additional_keys: Vec::new(),
                }, // y = O(knows) == S(likes)
                JoinStep {
                    acc_key: 2,
                    rel_key: 0,
                    additional_keys: Vec::new(),
                }, // z = O(likes) == S(owns)
            ],
        };
        let (proof, result_coms, m, ids) = prove_layer_bgp(&lc, &plan).unwrap();
        // Verifier sees ONLY the IRIs + revealed ids + result commitments.
        verify_layer_bgp(
            &lc,
            lc.state.commitment_root,
            &plan,
            &ids,
            &result_coms,
            m,
            &proof,
        )
        .unwrap();

        // Result is EXACTLY the three 3-hop paths (count + committed columns).
        assert_eq!(m, 3);
        assert_eq!(proof.atoms.len(), 3);
        let (exp_cols, domain) = expected_layer_bgp(&lc, &ids, &plan.joins);
        assert_eq!(result_coms.len(), exp_cols.len());
        assert_eq!(exp_cols.len(), 4); // [x, y, z, w]
        for (c, col) in exp_cols.iter().enumerate() {
            assert_eq!(commit_column(&pad(col, domain)).unwrap(), result_coms[c]);
        }
    }

    #[test]
    fn layer_bgp_k2_matches_self_join_capstone() {
        // The k=2 layer BGP (two knows atoms joined on the shared y) reduces to the
        // existing 2-atom self-join capstone: SAME result count AND committed columns.
        let lc = build_layer(&[
            ("a", "knows", "b"),
            ("b", "knows", "c"),
            ("c", "knows", "d"),
            ("a", "knows", "e"),
            ("a", "likes", "zz"),
        ]);
        let (gp, com_x, com_y, com_z, gp_m, gp_id) =
            prove_layer_graph_pattern2(&lc, KNOWS).unwrap();
        verify_layer_graph_pattern2(
            &lc,
            lc.state.commitment_root,
            KNOWS,
            gp_id,
            &com_x,
            &com_y,
            &com_z,
            gp_m,
            &gp,
        )
        .unwrap();

        let plan = LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![atom(KNOWS), atom(KNOWS)],
            joins: vec![JoinStep {
                acc_key: 1,
                rel_key: 0,
                additional_keys: Vec::new(),
            }],
        };
        let (proof, coms, m, ids) = prove_layer_bgp(&lc, &plan).unwrap();
        verify_layer_bgp(&lc, lc.state.commitment_root, &plan, &ids, &coms, m, &proof).unwrap();

        assert_eq!(m, gp_m);
        assert_eq!(coms, vec![com_x, com_y, com_z]);
    }

    #[test]
    fn layer_bgp_k1_is_exactly_the_complete_atom_filter() {
        let lc = build_layer(&[
            ("a", "knows", "b"),
            ("b", "knows", "c"),
            ("x", "likes", "y"),
        ]);
        let plan = LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![atom(KNOWS)],
            joins: vec![],
        };

        let (proof, coms, m, ids) = prove_layer_bgp(&lc, &plan).unwrap();
        verify_layer_bgp(&lc, lc.state.commitment_root, &plan, &ids, &coms, m, &proof).unwrap();

        assert_eq!(m, 2);
        assert!(proof.chain.is_none());
        let expected = atom_relation(&lc, ids[0]);
        let domain = padded_len(lc.materialized.len()).unwrap();
        assert_eq!(coms.len(), 2);
        for (commitment, column) in coms.iter().zip(expected.iter()) {
            assert_eq!(*commitment, commit_column(&pad(column, domain)).unwrap());
        }
    }

    #[test]
    fn predicate_variable_retains_root_bound_materialized_p_column() {
        let lc = build_layer(&[
            ("a", "knows", "b"),
            ("b", "knows", "c"),
            ("x", "likes", "y"),
        ]);
        let plan = LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![LayerBgpAtom {
                subject: AtomTerm::Variable,
                predicate_term: Vec::new(),
                predicate_variable: true,
                object: AtomTerm::Variable,
                object_range: None,
                object_kind: ObjectKindFilter::Any,
                subject_object_equal: false,
                source: ColumnSource::Materialized,
            }],
            joins: vec![],
        };

        let result = prove_layer_bgp_with_result(&lc, &plan).unwrap();
        verify_layer_bgp(
            &lc,
            lc.state.commitment_root,
            &plan,
            &result.predicate_ids,
            &result.result_commitments,
            result.result_len,
            &result.proof,
        )
        .unwrap();

        assert_eq!(result.predicate_ids, vec![0]);
        assert_eq!(result.result_len, lc.materialized.len());
        assert_eq!(result.result_commitments, lc.materialized_column_commitments);
        assert_eq!(
            result_id_namespaces(&plan).unwrap(),
            vec![
                ResultIdNamespace::NodeValue,
                ResultIdNamespace::Predicate,
                ResultIdNamespace::NodeValue,
            ]
        );
        let resolved = resolve_typed_result_objects(
            &lc,
            lc.state.commitment_root,
            &result.real_columns(),
            &result_id_namespaces(&plan).unwrap(),
        )
        .unwrap();
        assert_eq!(resolved[1].len(), lc.materialized.len());
        assert!(resolved[1]
            .iter()
            .all(|value| matches!(value, CanonicalObject::Node(_))));
    }

    #[test]
    fn predicate_to_node_value_join_is_rejected() {
        let variable_predicate = LayerBgpAtom {
            subject: node_const("a"),
            predicate_term: Vec::new(),
            predicate_variable: true,
            object: node_const("b"),
            object_range: None,
            object_kind: ObjectKindFilter::Any,
            subject_object_equal: false,
            source: ColumnSource::Materialized,
        };
        let plan = LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![variable_predicate, atom(KNOWS)],
            joins: vec![JoinStep {
                acc_key: 0,
                rel_key: 0,
                additional_keys: Vec::new(),
            }],
        };
        assert!(result_id_namespaces(&plan).is_err());
    }

    #[test]
    fn layer_bgp_k2_rejects_a_missing_join_chain() {
        let lc = build_layer(&[("a", "knows", "b"), ("b", "knows", "c")]);
        let plan = LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![atom(KNOWS), atom(KNOWS)],
            joins: vec![JoinStep {
                acc_key: 1,
                rel_key: 0,
                additional_keys: Vec::new(),
            }],
        };
        let (mut proof, coms, m, ids) = prove_layer_bgp(&lc, &plan).unwrap();
        proof.chain = None;
        assert!(
            verify_layer_bgp(&lc, lc.state.commitment_root, &plan, &ids, &coms, m, &proof,)
                .is_err()
        );
    }

    #[test]
    fn layer_bgp_wrong_expected_root_is_rejected() {
        let lc = build_layer(&[("a", "knows", "b"), ("b", "likes", "c"), ("c", "owns", "d")]);
        let plan = LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![atom(KNOWS), atom(LIKES), atom(OWNS)],
            joins: vec![
                JoinStep {
                    acc_key: 1,
                    rel_key: 0,
                    additional_keys: Vec::new(),
                },
                JoinStep {
                    acc_key: 2,
                    rel_key: 0,
                    additional_keys: Vec::new(),
                },
            ],
        };
        let (proof, result_coms, m, ids) = prove_layer_bgp(&lc, &plan).unwrap();
        // A DIFFERENT externally-trusted root must reject even an honest proof.
        let mut wrong_root = lc.state.commitment_root;
        wrong_root[0] ^= 0xff;
        assert!(verify_layer_bgp(&lc, wrong_root, &plan, &ids, &result_coms, m, &proof).is_err());
        // The correct external root still verifies.
        verify_layer_bgp(
            &lc,
            lc.state.commitment_root,
            &plan,
            &ids,
            &result_coms,
            m,
            &proof,
        )
        .unwrap();
    }

    #[test]
    fn layer_bgp_absent_predicate_authenticates_empty_relation() {
        let lc = build_layer(&[("a", "knows", "b"), ("b", "knows", "c")]);
        let plan = LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![atom(KNOWS), atom(b"http://ex/absent")],
            joins: vec![JoinStep {
                acc_key: 1,
                rel_key: 0,
                additional_keys: Vec::new(),
            }],
        };
        let result = prove_layer_bgp_with_result(&lc, &plan).unwrap();
        assert_eq!(result.result_len, 0);
        assert_eq!(result.result_commitments.len(), 3);
        assert_eq!(
            result.proof.dictionary_absence,
            Some(DictionaryAbsence {
                atom_index: 1,
                position: DictionaryAbsencePosition::Predicate,
            })
        );
        verify_layer_bgp(
            &lc,
            lc.state.commitment_root,
            &plan,
            &result.predicate_ids,
            &result.result_commitments,
            result.result_len,
            &result.proof,
        )
        .unwrap();
    }

    // ---- Q_k with subject/object CONSTANTS (node/value dictionary) end-to-end ----

    const NAME: &[u8] = b"http://ex/name";

    // Build a materialized base layer from arbitrary ValueTriples (node OR value objects).
    fn build_layer_triples(triples: Vec<ValueTriple>) -> std::sync::Arc<LayerCommitment> {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for t in triples {
            builder.add_value_triple(t).unwrap();
        }
        let layer = builder.commit().unwrap();
        layer.proof_commitment().unwrap()
    }

    fn node_const(suffix: &str) -> AtomTerm {
        AtomTerm::NodeConstant(iri(suffix).into_bytes())
    }

    // One atom's dense KEPT (variable) columns from the layer additions in stored order,
    // applying the predicate + any subject/object constant ids. Mirrors EXACTLY what
    // prove_atom_filter compacts (kept in S,O order); one column per variable.
    fn kept_relation(
        lc: &LayerCommitment,
        atom: &LayerBgpAtom,
        predicate_id: u64,
        subject_id: Option<u64>,
        object_id: Option<u64>,
    ) -> Vec<Vec<ArkFr>> {
        let keep_s = matches!(atom.subject, AtomTerm::Variable);
        let keep_o = matches!(atom.object, AtomTerm::Variable);
        let mut cols: Vec<Vec<ArkFr>> = Vec::new();
        if keep_s {
            cols.push(Vec::new());
        }
        if keep_o {
            cols.push(Vec::new());
        }
        for rec in &lc.materialized {
            if rec.predicate_id == predicate_id
                && subject_id.map_or(true, |s| rec.subject_id == s)
                && object_id.map_or(true, |o| rec.object_id == o)
            {
                let mut idx = 0;
                if keep_s {
                    cols[idx].push(ArkFr::from_u64(rec.subject_id));
                    idx += 1;
                }
                if keep_o {
                    cols[idx].push(ArkFr::from_u64(rec.object_id));
                }
            }
        }
        cols
    }

    // Independently recompute the complete BGP result columns (real length) and final join
    // domain over per-atom kept relations, folding the audited compute_binary_join over the
    // plan's join steps (mirrors expected_layer_bgp but for variable-arity relations).
    fn expected_const_bgp(
        lc: &LayerCommitment,
        atoms: &[LayerBgpAtom],
        predicate_ids: &[u64],
        subject_ids: &[Option<u64>],
        object_ids: &[Option<u64>],
        joins: &[JoinStep],
    ) -> (Vec<Vec<ArkFr>>, usize) {
        use crate::proof::join2::{binary_join_domain, compute_binary_join};
        let split = |cols: &[Vec<ArkFr>], key: usize| -> (Vec<ArkFr>, Vec<Vec<ArkFr>>) {
            let k = cols[key].clone();
            let other = cols
                .iter()
                .enumerate()
                .filter(|(c, _)| *c != key)
                .map(|(_, c)| c.clone())
                .collect();
            (k, other)
        };
        let rel = |i: usize| {
            kept_relation(
                lc,
                &atoms[i],
                predicate_ids[i],
                subject_ids[i],
                object_ids[i],
            )
        };
        let mut acc = rel(0);
        let mut acc_len = acc[0].len();
        let mut domain = padded_len(lc.materialized.len()).unwrap();
        for (j, step) in joins.iter().enumerate() {
            let r = rel(j + 1);
            let rel_len = r[0].len();
            let (r_key, r_other) = split(&acc, step.acc_key);
            let (s_key, s_other) = split(&r, step.rel_key);
            let r_refs: Vec<&[ArkFr]> = r_other.iter().map(|c| c.as_slice()).collect();
            let s_refs: Vec<&[ArkFr]> = s_other.iter().map(|c| c.as_slice()).collect();
            let w = compute_binary_join(&r_key, &r_refs, &s_key, &s_refs);
            domain = binary_join_domain(acc_len, rel_len, w.m_j, w.n_u).unwrap();
            let mut new_acc = w.r_other_out.clone();
            new_acc.push(w.key.clone());
            new_acc.extend(w.s_other_out.clone());
            acc = new_acc;
            acc_len = w.m_j;
        }
        (acc, domain)
    }

    // Prove+verify a 2-atom const-bearing plan, then assert the result equals the
    // INDEPENDENTLY-recomputed filtered+joined result (count + committed columns).
    fn check_const_bgp(lc: &LayerCommitment, plan: &LayerBgpPlan, expected_m: usize) {
        let (proof, coms, m, ids) = prove_layer_bgp(lc, plan).unwrap();
        verify_layer_bgp(lc, lc.state.commitment_root, plan, &ids, &coms, m, &proof).unwrap();
        assert_eq!(m, expected_m);
        let subject_ids: Vec<Option<u64>> = proof.atoms.iter().map(|a| a.subject_id).collect();
        let object_ids: Vec<Option<u64>> = proof.atoms.iter().map(|a| a.object_id).collect();
        let (exp_cols, domain) = expected_const_bgp(
            lc,
            &plan.atoms,
            &ids,
            &subject_ids,
            &object_ids,
            &plan.joins,
        );
        assert_eq!(coms.len(), exp_cols.len());
        for (c, col) in exp_cols.iter().enumerate() {
            assert_eq!(
                commit_column(&pad(col, domain)).unwrap(),
                coms[c],
                "column {c}"
            );
        }
    }

    #[test]
    fn layer_bgp_object_node_constant_end_to_end() {
        // t(?x, likes, ?z), t(?x, knows, <b>): x likes z AND x knows <b>.
        // likes: a->m,a->o,c->n,d->p ; knows: a->b,c->b,d->q. x knows b => {a,c}.
        // Join on x (subject of both): (a,m),(a,o),(c,n). The const atom is the RIGHT
        // relation (arity 1), exercising an empty s_other in the join.
        let lc = build_layer(&[
            ("a", "likes", "m"),
            ("a", "likes", "o"),
            ("c", "likes", "n"),
            ("d", "likes", "p"),
            ("a", "knows", "b"),
            ("c", "knows", "b"),
            ("d", "knows", "q"),
        ]);
        let plan = LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![
                LayerBgpAtom::all_variables(LIKES.to_vec()),
                LayerBgpAtom {
                    subject: AtomTerm::Variable,
                    predicate_term: KNOWS.to_vec(),
                    predicate_variable: false,
                    object: node_const("b"),
                    object_range: None,
                    object_kind: ObjectKindFilter::Any,
                    subject_object_equal: false,
                    source: ColumnSource::Materialized,
                },
            ],
            joins: vec![JoinStep {
                acc_key: 0,
                rel_key: 0,
                additional_keys: Vec::new(),
            }], // x = S(likes) == S(knows)
        };
        let (proof, _coms, _m, _ids) = prove_layer_bgp(&lc, &plan).unwrap();
        // The revealed object id equals <b>'s node/value dictionary id; the variable
        // atom reveals no constant ids.
        let b_id =
            resolve_node_value_id(&lc, &CanonicalObject::Node(iri("b").into_bytes())).unwrap();
        assert_eq!(proof.atoms[1].object_id, Some(b_id));
        assert_eq!(proof.atoms[1].subject_id, None);
        assert_eq!(proof.atoms[0].subject_id, None);
        assert_eq!(proof.atoms[0].object_id, None);
        // Result [z, x] equals the independently-recomputed filtered+joined result.
        check_const_bgp(&lc, &plan, 3);
    }

    #[test]
    fn layer_bgp_object_value_constant_end_to_end() {
        // t(?x, name, "hi"), t(?x, knows, ?z): x named "hi" AND x knows z. The value-literal
        // atom is the LEFT relation (arity 1), exercising an empty r_other in the join.
        let name_iri = iri("name");
        let lc = build_layer_triples(vec![
            ValueTriple::new_string_value(&iri("a"), &name_iri, "hi"),
            ValueTriple::new_string_value(&iri("c"), &name_iri, "hi"),
            ValueTriple::new_string_value(&iri("d"), &name_iri, "bye"),
            ValueTriple::new_node(&iri("a"), &iri("knows"), &iri("m")),
            ValueTriple::new_node(&iri("a"), &iri("knows"), &iri("o")),
            ValueTriple::new_node(&iri("c"), &iri("knows"), &iri("n")),
        ]);
        // Recover the EXACT canonical "hi" value object from the catalog (its lexical bytes
        // are the store's typed encoding, not literally b"hi"): find a's name-triple object.
        let name_id = resolve_predicate_id(&lc, NAME).unwrap();
        let a_id =
            resolve_node_value_id(&lc, &CanonicalObject::Node(iri("a").into_bytes())).unwrap();
        let hi_id = lc
            .materialized
            .iter()
            .find(|r| r.subject_id == a_id && r.predicate_id == name_id)
            .unwrap()
            .object_id;
        let (dt, lex) = match &lc
            .node_value_catalog
            .iter()
            .find(|r| r.object_id == hi_id)
            .unwrap()
            .triple
            .object
        {
            CanonicalObject::Value { datatype, lexical } => (*datatype, lexical.clone()),
            _ => panic!("expected a value literal"),
        };
        let plan = LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![
                LayerBgpAtom {
                    subject: AtomTerm::Variable,
                    predicate_term: NAME.to_vec(),
                    predicate_variable: false,
                    object: AtomTerm::ValueConstant(dt, lex),
                    object_range: None,
                    object_kind: ObjectKindFilter::Any,
                    subject_object_equal: false,
                    source: ColumnSource::Materialized,
                },
                LayerBgpAtom::all_variables(KNOWS.to_vec()),
            ],
            joins: vec![JoinStep {
                acc_key: 0,
                rel_key: 0,
                additional_keys: Vec::new(),
            }], // x = S(name) == S(knows)
        };
        let (proof, _coms, _m, _ids) = prove_layer_bgp(&lc, &plan).unwrap();
        assert_eq!(proof.atoms[0].object_id, Some(hi_id));
        assert_eq!(proof.atoms[0].subject_id, None);
        // Result [x, z] = {(a,m),(a,o),(c,n)}.
        check_const_bgp(&lc, &plan, 3);
    }

    #[test]
    fn layer_bgp_subject_node_constant_end_to_end() {
        // t(<a>, knows, ?y), t(?y, likes, ?z): a knows y AND y likes z. The subject-const
        // atom is the LEFT relation (arity 1, keeps O = y).
        // knows: a->b,a->c,d->e ; likes: b->m,b->o,c->n. a knows y => {b,c}.
        let lc = build_layer(&[
            ("a", "knows", "b"),
            ("a", "knows", "c"),
            ("d", "knows", "e"),
            ("b", "likes", "m"),
            ("b", "likes", "o"),
            ("c", "likes", "n"),
        ]);
        let plan = LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![
                LayerBgpAtom {
                    subject: node_const("a"),
                    predicate_term: KNOWS.to_vec(),
                    predicate_variable: false,
                    object: AtomTerm::Variable,
                    object_range: None,
                    object_kind: ObjectKindFilter::Any,
                    subject_object_equal: false,
                    source: ColumnSource::Materialized,
                },
                LayerBgpAtom::all_variables(LIKES.to_vec()),
            ],
            joins: vec![JoinStep {
                acc_key: 0,
                rel_key: 0,
                additional_keys: Vec::new(),
            }], // y = O(knows) == S(likes)
        };
        let (proof, _coms, _m, _ids) = prove_layer_bgp(&lc, &plan).unwrap();
        let a_id =
            resolve_node_value_id(&lc, &CanonicalObject::Node(iri("a").into_bytes())).unwrap();
        assert_eq!(proof.atoms[0].subject_id, Some(a_id));
        assert_eq!(proof.atoms[0].object_id, None);
        // Result [y, z] = {(b,m),(b,o),(c,n)}.
        check_const_bgp(&lc, &plan, 3);
    }

    #[test]
    fn layer_bgp_mixed_constant_and_variable_atoms_and_rejections() {
        // t(?x, knows, ?y), t(?y, likes, <g>): x knows y AND y likes <g>. Joins on the
        // OBJECT variable y (acc_key = 1) of the all-variable atom against the object-const
        // atom (arity 1). knows: a->b,c->b,a->e ; likes: b->g,e->h,f->g. y likes g => {b,f}.
        let lc = build_layer(&[
            ("a", "knows", "b"),
            ("c", "knows", "b"),
            ("a", "knows", "e"),
            ("b", "likes", "g"),
            ("e", "likes", "h"),
            ("f", "likes", "g"),
        ]);
        let plan = LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![
                LayerBgpAtom::all_variables(KNOWS.to_vec()),
                LayerBgpAtom {
                    subject: AtomTerm::Variable,
                    predicate_term: LIKES.to_vec(),
                    predicate_variable: false,
                    object: node_const("g"),
                    object_range: None,
                    object_kind: ObjectKindFilter::Any,
                    subject_object_equal: false,
                    source: ColumnSource::Materialized,
                },
            ],
            joins: vec![JoinStep {
                acc_key: 1,
                rel_key: 0,
                additional_keys: Vec::new(),
            }], // y = O(knows) == S(likes)
        };
        // Honest: result [x, y] = {(a,b),(c,b)}.
        check_const_bgp(&lc, &plan, 2);

        let (proof, coms, m, ids) = prove_layer_bgp(&lc, &plan).unwrap();
        verify_layer_bgp(&lc, lc.state.commitment_root, &plan, &ids, &coms, m, &proof).unwrap();

        // Wrong revealed constant id: the verifier feeds the SAME id to both the membership
        // and the filter, and binds it into the statement transcript -> a substituted id
        // breaks the node/value membership (the pair is no longer a dictionary row).
        let mut bad_id = proof.clone();
        bad_id.atoms[1].object_id = Some(proof.atoms[1].object_id.unwrap().wrapping_add(1));
        assert!(verify_layer_bgp(
            &lc,
            lc.state.commitment_root,
            &plan,
            &ids,
            &coms,
            m,
            &bad_id
        )
        .is_err());

        // Substituting a DIFFERENT genuine node id (here <b>'s id) for <g> is also rejected:
        // membership binds the PAIR (hash_object(<g>), id), so <b>'s id fails for <g>.
        let b_id =
            resolve_node_value_id(&lc, &CanonicalObject::Node(iri("b").into_bytes())).unwrap();
        let mut sub_id = proof.clone();
        sub_id.atoms[1].object_id = Some(b_id);
        assert!(verify_layer_bgp(
            &lc,
            lc.state.commitment_root,
            &plan,
            &ids,
            &coms,
            m,
            &sub_id
        )
        .is_err());

        // Wrong externally-trusted root rejects even an honest proof.
        let mut wrong_root = lc.state.commitment_root;
        wrong_root[0] ^= 0xff;
        assert!(verify_layer_bgp(&lc, wrong_root, &plan, &ids, &coms, m, &proof).is_err());

        // Honest still verifies.
        verify_layer_bgp(&lc, lc.state.commitment_root, &plan, &ids, &coms, m, &proof).unwrap();
    }

    #[test]
    fn layer_bgp_absent_constant_authenticates_empty_relational_result() {
        let lc = build_layer(&[("a", "knows", "b"), ("b", "likes", "c")]);
        // The object constant <absent> is not in the layer's node/value dictionary.
        let plan = LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![
                LayerBgpAtom::all_variables(KNOWS.to_vec()),
                LayerBgpAtom {
                    subject: AtomTerm::Variable,
                    predicate_term: LIKES.to_vec(),
                    predicate_variable: false,
                    object: node_const("absent"),
                    object_range: None,
                    object_kind: ObjectKindFilter::Any,
                    subject_object_equal: false,
                    source: ColumnSource::Materialized,
                },
            ],
            joins: vec![JoinStep {
                acc_key: 1,
                rel_key: 0,
                additional_keys: Vec::new(),
            }],
        };
        let result = prove_layer_bgp_with_result(&lc, &plan).unwrap();
        assert_eq!(result.result_len, 0);
        assert_eq!(result.result_commitments.len(), 2);
        assert_eq!(
            result.proof.dictionary_absence,
            Some(DictionaryAbsence {
                atom_index: 1,
                position: DictionaryAbsencePosition::Object,
            })
        );
        verify_layer_bgp(
            &lc,
            lc.state.commitment_root,
            &plan,
            &result.predicate_ids,
            &result.result_commitments,
            result.result_len,
            &result.proof,
        )
        .unwrap();
        verify_layer_bgp_result_log_domain(
            &lc,
            &plan,
            &result.proof,
            result.result_log_domain,
        )
        .unwrap();

        let mut wrong_atom = result.proof.clone();
        wrong_atom.dictionary_absence.as_mut().unwrap().atom_index = 0;
        assert!(verify_layer_bgp(
            &lc,
            lc.state.commitment_root,
            &plan,
            &result.predicate_ids,
            &result.result_commitments,
            result.result_len,
            &wrong_atom,
        )
        .is_err());

        let mut wrong_commitments = result.result_commitments.clone();
        wrong_commitments.pop();
        assert!(verify_layer_bgp(
            &lc,
            lc.state.commitment_root,
            &plan,
            &result.predicate_ids,
            &wrong_commitments,
            result.result_len,
            &result.proof,
        )
        .is_err());
    }

    #[test]
    fn layer_bgp_single_ground_atom_proves_presence_and_absence() {
        let lc = build_layer(&[("a", "knows", "b"), ("b", "likes", "c")]);
        let atom = |subject: &str, predicate: &[u8], object: &str| LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![LayerBgpAtom {
                subject: node_const(subject),
                predicate_term: predicate.to_vec(),
                predicate_variable: false,
                object: node_const(object),
                object_range: None,
                object_kind: ObjectKindFilter::Any,
                subject_object_equal: false,
                source: ColumnSource::Materialized,
            }],
            joins: Vec::new(),
        };

        for (plan, expected_len, expected_absence) in [
            (atom("a", KNOWS, "b"), 1, None),
            (atom("a", LIKES, "c"), 0, None),
            (
                atom("a", KNOWS, "missing-object"),
                0,
                Some(DictionaryAbsence { atom_index: 0, position: DictionaryAbsencePosition::Object }),
            ),
            (
                atom("missing-subject", KNOWS, "b"),
                0,
                Some(DictionaryAbsence { atom_index: 0, position: DictionaryAbsencePosition::Subject }),
            ),
            (
                atom("a", b"http://ex/missing-predicate", "b"),
                0,
                Some(DictionaryAbsence { atom_index: 0, position: DictionaryAbsencePosition::Predicate }),
            ),
        ] {
            let result = prove_layer_bgp_with_result(&lc, &plan).unwrap();
            assert_eq!(result.result_len, expected_len);
            assert_eq!(result.proof.dictionary_absence, expected_absence);
            assert_eq!(result.result_commitments.len(), 1);
            assert_eq!(result.result_columns.len(), 1);
            verify_layer_bgp(
                &lc,
                lc.state.commitment_root,
                &plan,
                &result.predicate_ids,
                &result.result_commitments,
                result.result_len,
                &result.proof,
            )
            .unwrap();
            if let Some(absence) = expected_absence {
                let mut forged = result.proof.clone();
                forged.dictionary_absence = Some(DictionaryAbsence {
                    atom_index: absence.atom_index,
                    position: match absence.position {
                        DictionaryAbsencePosition::Predicate => DictionaryAbsencePosition::Subject,
                        DictionaryAbsencePosition::Subject | DictionaryAbsencePosition::Object => DictionaryAbsencePosition::Predicate,
                    },
                });
                assert!(verify_layer_bgp(
                    &lc,
                    lc.state.commitment_root,
                    &plan,
                    &result.predicate_ids,
                    &result.result_commitments,
                    result.result_len,
                    &forged,
                )
                .is_err());

                let mut forged_present = result.proof.clone();
                forged_present.dictionary_absence = None;
                assert!(verify_layer_bgp(
                    &lc,
                    lc.state.commitment_root,
                    &plan,
                    &result.predicate_ids,
                    &result.result_commitments,
                    1,
                    &forged_present,
                )
                .is_err());
            }
        }
    }

    #[test]
    fn layer_bgp_all_ground_conjunction_ands_authenticated_existence_bits() {
        let lc = build_layer(&[("a", "knows", "b"), ("b", "likes", "c")]);
        let atom = |subject: &str, predicate: &[u8], object: &str| LayerBgpAtom {
            subject: node_const(subject),
            predicate_term: predicate.to_vec(),
            predicate_variable: false,
            object: node_const(object),
            object_range: None,
            object_kind: ObjectKindFilter::Any,
            subject_object_equal: false,
            source: ColumnSource::Materialized,
        };
        for (plan, expected_len) in [
            (
                LayerBgpPlan {
                    source: ColumnSource::Materialized,
                    atoms: vec![atom("a", KNOWS, "b"), atom("b", LIKES, "c")],
                    joins: Vec::new(),
                },
                1,
            ),
            (
                LayerBgpPlan {
                    source: ColumnSource::Materialized,
                    atoms: vec![atom("a", KNOWS, "b"), atom("a", LIKES, "c")],
                    joins: Vec::new(),
                },
                0,
            ),
            (
                LayerBgpPlan {
                    source: ColumnSource::Materialized,
                    atoms: vec![atom("a", KNOWS, "b"), atom("a", KNOWS, "b")],
                    joins: Vec::new(),
                },
                1,
            ),
        ] {
            let result = prove_layer_bgp_with_result(&lc, &plan).unwrap();
            assert_eq!(result.result_len, expected_len);
            assert_eq!(result.result_commitments.len(), 1);
            verify_layer_bgp(
                &lc,
                lc.state.commitment_root,
                &plan,
                &result.predicate_ids,
                &result.result_commitments,
                result.result_len,
                &result.proof,
            )
            .unwrap();

            let mut forged_atom = result.proof.clone();
            forged_atom.atoms[0].m = 1 - forged_atom.atoms[0].m;
            assert!(verify_layer_bgp(
                &lc,
                lc.state.commitment_root,
                &plan,
                &result.predicate_ids,
                &result.result_commitments,
                result.result_len,
                &forged_atom,
            )
            .is_err());
            assert!(verify_layer_bgp(
                &lc,
                lc.state.commitment_root,
                &plan,
                &result.predicate_ids,
                &result.result_commitments,
                1 - expected_len,
                &result.proof,
            )
            .is_err());
        }
    }

    #[test]
    fn layer_bgp_ground_atoms_gate_relational_results() {
        let lc = build_layer(&[("a", "knows", "b"), ("b", "likes", "c")]);
        for (ground, expected_len) in [
            (
                LayerBgpAtom {
                    subject: node_const("a"),
                    predicate_term: KNOWS.to_vec(),
                    predicate_variable: false,
                    object: node_const("b"),
                    object_range: None,
                    object_kind: ObjectKindFilter::Any,
                    subject_object_equal: false,
                    source: ColumnSource::Materialized,
                },
                1,
            ),
            (
                LayerBgpAtom {
                    subject: node_const("a"),
                    predicate_term: LIKES.to_vec(),
                    predicate_variable: false,
                    object: node_const("c"),
                    object_range: None,
                    object_kind: ObjectKindFilter::Any,
                    subject_object_equal: false,
                    source: ColumnSource::Materialized,
                },
                0,
            ),
        ] {
            let plan = LayerBgpPlan {
                source: ColumnSource::Materialized,
                atoms: vec![LayerBgpAtom::all_variables(LIKES.to_vec()), ground],
                joins: Vec::new(),
            };
            let result = prove_layer_bgp_with_result(&lc, &plan).unwrap();
            assert_eq!(result.result_len, expected_len);
            verify_layer_bgp(
                &lc,
                lc.state.commitment_root,
                &plan,
                &result.predicate_ids,
                &result.result_commitments,
                result.result_len,
                &result.proof,
            )
            .unwrap();
            let mut forged = result.proof.clone();
            forged.atoms[1].m = 1 - forged.atoms[1].m;
            assert!(verify_layer_bgp(
                &lc,
                lc.state.commitment_root,
                &plan,
                &result.predicate_ids,
                &result.result_commitments,
                result.result_len,
                &forged,
            )
            .is_err());
        }
    }

    #[test]
    fn layer_bgp_value_subject_is_rejected() {
        let lc = build_layer(&[("a", "knows", "b"), ("b", "likes", "c")]);
        // A value literal in the SUBJECT position is rejected.
        let value_subject = LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![
                LayerBgpAtom {
                    subject: AtomTerm::ValueConstant(CanonicalDatatype::String, b"x".to_vec()),
                    predicate_term: KNOWS.to_vec(),
                    predicate_variable: false,
                    object: AtomTerm::Variable,
                    object_range: None,
                    object_kind: ObjectKindFilter::Any,
                    subject_object_equal: false,
                    source: ColumnSource::Materialized,
                },
                LayerBgpAtom::all_variables(LIKES.to_vec()),
            ],
            joins: vec![JoinStep {
                acc_key: 0,
                rel_key: 0,
                additional_keys: Vec::new(),
            }],
        };
        assert!(prove_layer_bgp(&lc, &value_subject).is_err());
    }

    // ---- Q_k with a numeric object-value RANGE filter (CP-C) end-to-end ----

    const AGE: &[u8] = b"http://ex/age";
    const UINT32_TAG: u64 = CanonicalDatatype::UInt32 as u64;
    const BOOLEAN_TAG: u64 = CanonicalDatatype::Boolean as u64;

    fn fr(v: u64) -> ArkFr {
        ArkFr::from_u64(v)
    }

    fn cmp_op(op: RangeOp, a: u64, b: u64) -> bool {
        match op {
            RangeOp::Lt => a < b,
            RangeOp::Le => a <= b,
            RangeOp::Gt => a > b,
            RangeOp::Ge => a >= b,
        }
    }

    // A base layer of `t(person, age, <UInt32>)` and `t(person, name, <node>)` triples.
    // Ages are inserted OUT of numeric order; every person has exactly one name.
    fn build_age_name_layer() -> std::sync::Arc<LayerCommitment> {
        use tdb_succinct::TdbDataType;
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        let age = iri("age");
        let name = iri("name");
        for (person, a) in [
            ("alice", 30u32),
            ("bob", 15),
            ("carol", 18),
            ("dave", 40),
            ("erin", 18),
        ] {
            builder
                .add_value_triple(ValueTriple::new_value(
                    &iri(person),
                    &age,
                    u32::make_entry(&a),
                ))
                .unwrap();
            builder
                .add_value_triple(ValueTriple::new_node(
                    &iri(person),
                    &name,
                    &iri(&format!("{person}_n")),
                ))
                .unwrap();
        }
        let layer = builder.commit().unwrap();
        layer.proof_commitment().unwrap()
    }

    fn build_mixed_object_kind_layer() -> std::sync::Arc<LayerCommitment> {
        use tdb_succinct::TdbDataType;
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        let data = iri("data");
        let label = iri("label");
        let flag = iri("flag");
        builder
            .add_value_triple(ValueTriple::new_value(
                &iri("alice"),
                &data,
                u32::make_entry(&42),
            ))
            .unwrap();
        builder
            .add_value_triple(ValueTriple::new_value(
                &iri("bob"),
                &data,
                bool::make_entry(&true),
            ))
            .unwrap();
        builder
            .add_value_triple(ValueTriple::new_node(
                &iri("carol"),
                &data,
                &iri("node-object"),
            ))
            .unwrap();
        builder
            .add_value_triple(ValueTriple::new_value(
                &iri("alice"),
                &flag,
                bool::make_entry(&false),
            ))
            .unwrap();
        for person in ["alice", "bob", "carol"] {
            builder
                .add_value_triple(ValueTriple::new_node(
                    &iri(person),
                    &label,
                    &iri(&format!("{person}-label")),
                ))
                .unwrap();
        }
        builder.commit().unwrap().proof_commitment().unwrap()
    }

    #[test]
    fn layer_bgp_object_kind_filters_are_exact_and_transcript_bound() {
        let lc = build_mixed_object_kind_layer();
        let literal_plan = LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![LayerBgpAtom::all_variables(iri("data").into_bytes())
                .with_object_kind(ObjectKindFilter::Literal)],
            joins: vec![],
        };
        let result = prove_layer_bgp_with_result(&lc, &literal_plan).unwrap();
        assert_eq!(result.result_len, 2);
        verify_layer_bgp(
            &lc,
            lc.state.commitment_root,
            &literal_plan,
            &result.predicate_ids,
            &result.result_commitments,
            result.result_len,
            &result.proof,
        )
        .unwrap();
        let real = result.real_columns();
        let resolved = resolve_typed_result_objects(
            &lc,
            lc.state.commitment_root,
            &real,
            &result_id_namespaces(&literal_plan).unwrap(),
        )
        .unwrap();
        assert!(resolved[1]
            .iter()
            .all(|object| matches!(object, CanonicalObject::Value { .. })));
        assert!(resolved[1].iter().any(|object| matches!(
            object,
            CanonicalObject::Value { datatype: CanonicalDatatype::UInt32, .. }
        )));
        assert!(resolved[1].iter().any(|object| matches!(
            object,
            CanonicalObject::Value { datatype: CanonicalDatatype::Boolean, .. }
        )));

        let node_plan = LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![LayerBgpAtom::all_variables(iri("data").into_bytes())
                .with_object_kind(ObjectKindFilter::Node)],
            joins: vec![],
        };
        let node_result = prove_layer_bgp_with_result(&lc, &node_plan).unwrap();
        assert_eq!(node_result.result_len, 1);
        verify_layer_bgp(
            &lc,
            lc.state.commitment_root,
            &node_plan,
            &node_result.predicate_ids,
            &node_result.result_commitments,
            node_result.result_len,
            &node_result.proof,
        )
        .unwrap();

        // The public kind is part of the statement; a literal proof cannot be replayed
        // as a node or unconstrained proof even though all three use the same Store IDs.
        assert!(verify_layer_bgp(
            &lc,
            lc.state.commitment_root,
            &node_plan,
            &result.predicate_ids,
            &result.result_commitments,
            result.result_len,
            &result.proof,
        )
        .is_err());
        let any_plan = LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![LayerBgpAtom::all_variables(iri("data").into_bytes())],
            joins: vec![],
        };
        assert!(verify_layer_bgp(
            &lc,
            lc.state.commitment_root,
            &any_plan,
            &result.predicate_ids,
            &result.result_commitments,
            result.result_len,
            &result.proof,
        )
        .is_err());
    }

    #[test]
    fn layer_bgp_literal_kind_supports_empty_predicate_variables_and_joins() {
        let lc = build_mixed_object_kind_layer();

        let empty_plan = LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![LayerBgpAtom::all_variables(iri("label").into_bytes())
                .with_object_kind(ObjectKindFilter::Literal)],
            joins: vec![],
        };
        let empty = prove_layer_bgp_with_result(&lc, &empty_plan).unwrap();
        assert_eq!(empty.result_len, 0);
        verify_layer_bgp(
            &lc,
            lc.state.commitment_root,
            &empty_plan,
            &empty.predicate_ids,
            &empty.result_commitments,
            0,
            &empty.proof,
        )
        .unwrap();

        let predicate_variable = LayerBgpAtom {
            subject: AtomTerm::Variable,
            predicate_term: Vec::new(),
            predicate_variable: true,
            object: AtomTerm::Variable,
            object_range: None,
            object_kind: ObjectKindFilter::Literal,
            subject_object_equal: false,
            source: ColumnSource::Materialized,
        };
        let predicate_plan = LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![predicate_variable],
            joins: vec![],
        };
        let all_literals = prove_layer_bgp_with_result(&lc, &predicate_plan).unwrap();
        assert_eq!(all_literals.result_len, 3);
        assert_eq!(all_literals.result_commitments.len(), 3);
        verify_layer_bgp(
            &lc,
            lc.state.commitment_root,
            &predicate_plan,
            &all_literals.predicate_ids,
            &all_literals.result_commitments,
            all_literals.result_len,
            &all_literals.proof,
        )
        .unwrap();

        let join_plan = LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![
                LayerBgpAtom::all_variables(iri("data").into_bytes())
                    .with_object_kind(ObjectKindFilter::Literal),
                LayerBgpAtom::all_variables(iri("label").into_bytes()),
            ],
            joins: vec![JoinStep {
                acc_key: 0,
                rel_key: 0,
                additional_keys: vec![],
            }],
        };
        let joined = prove_layer_bgp_with_result(&lc, &join_plan).unwrap();
        assert_eq!(joined.result_len, 2);
        assert_eq!(joined.result_commitments.len(), 3);
        verify_layer_bgp(
            &lc,
            lc.state.commitment_root,
            &join_plan,
            &joined.predicate_ids,
            &joined.result_commitments,
            joined.result_len,
            &joined.proof,
        )
        .unwrap();
    }

    // A layer with FEW triples but MANY distinct node/values: 3 people, each with a
    // distinct age and a distinct name node. The node/value dictionary (9 entries -> 16
    // rows) is LARGER than the table (6 triples -> 8 rows), so the common relocation
    // domain D = 16 strictly EXCEEDS the table domain 2^a = 8 — exercising the up
    // (object/dict -> D) and down (range_sel -> table) relocation branches end-to-end.
    fn build_distinct_age_name_layer() -> std::sync::Arc<LayerCommitment> {
        use tdb_succinct::TdbDataType;
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        let age = iri("age");
        let name = iri("name");
        for (person, a) in [("p1", 10u32), ("p2", 25), ("p3", 5)] {
            builder
                .add_value_triple(ValueTriple::new_value(
                    &iri(person),
                    &age,
                    u32::make_entry(&a),
                ))
                .unwrap();
            builder
                .add_value_triple(ValueTriple::new_node(
                    &iri(person),
                    &name,
                    &iri(&format!("{person}_n")),
                ))
                .unwrap();
        }
        let layer = builder.commit().unwrap();
        layer.proof_commitment().unwrap()
    }

    // The UInt32 value behind a node/value dict id (its order-preserving encoding is the
    // big-endian u32, so a BE read recovers the value), or None for a non-UInt32 object.
    fn uint32_value_of(lc: &LayerCommitment, object_id: u64) -> Option<u64> {
        use std::convert::TryInto;
        lc.node_value_catalog
            .iter()
            .find(|r| r.object_id == object_id)
            .and_then(|r| match &r.triple.object {
                CanonicalObject::Value {
                    datatype: CanonicalDatatype::UInt32,
                    lexical,
                } => Some(u32::from_be_bytes(lexical.as_slice().try_into().unwrap()) as u64),
                _ => None,
            })
    }

    fn age_range_plan(op: RangeOp, c: u64) -> LayerBgpPlan {
        LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![
                LayerBgpAtom::all_variables(AGE.to_vec()).with_object_range(ObjectRangeSpec {
                    op,
                    datatype: fr(UINT32_TAG),
                    c: fr(c),
                    b_bits: 64,
                    second: None,
                }),
                LayerBgpAtom::all_variables(NAME.to_vec()),
            ],
            // x = S(age) == S(name).
            joins: vec![JoinStep {
                acc_key: 0,
                rel_key: 0,
                additional_keys: Vec::new(),
            }],
        }
    }

    // Independently recompute the complete result of
    // `t(?x, age, ?a) [?a op c], t(?x, name, ?n)`: the age relation is the range-filtered
    // [S, O] pairs (in stored order), joined with the name [S, O] pairs on the shared x.
    // Returns the result columns [a, x, n], the final join domain, and the row count.
    fn expected_age_name_join(
        lc: &LayerCommitment,
        op: RangeOp,
        c: u64,
    ) -> (Vec<Vec<ArkFr>>, usize, usize) {
        use crate::proof::join2::{binary_join_domain, compute_binary_join};
        let age_id = resolve_predicate_id(lc, AGE).unwrap();
        let name_id = resolve_predicate_id(lc, NAME).unwrap();
        let (mut age_s, mut age_o) = (Vec::new(), Vec::new());
        for rec in &lc.materialized {
            if rec.predicate_id == age_id {
                if let Some(v) = uint32_value_of(lc, rec.object_id) {
                    if cmp_op(op, v, c) {
                        age_s.push(fr(rec.subject_id));
                        age_o.push(fr(rec.object_id));
                    }
                }
            }
        }
        let (mut name_s, mut name_o) = (Vec::new(), Vec::new());
        for rec in &lc.materialized {
            if rec.predicate_id == name_id {
                name_s.push(fr(rec.subject_id));
                name_o.push(fr(rec.object_id));
            }
        }
        let w = compute_binary_join(&age_s, &[age_o.as_slice()], &name_s, &[name_o.as_slice()]);
        let domain = binary_join_domain(age_s.len(), name_s.len(), w.m_j, w.n_u).unwrap();
        let mut cols = w.r_other_out.clone(); // [a]
        cols.push(w.key.clone()); // x
        cols.extend(w.s_other_out.clone()); // [n]
        (cols, domain, w.m_j)
    }

    // Prove+verify an age-range BGP and assert the committed result equals the
    // independently recomputed filtered+joined answer (count + committed columns).
    fn check_age_range(lc: &LayerCommitment, op: RangeOp, c: u64, expected_m: usize) {
        let plan = age_range_plan(op, c);
        let (proof, coms, m, ids) = prove_layer_bgp(lc, &plan).unwrap();
        verify_layer_bgp(lc, lc.state.commitment_root, &plan, &ids, &coms, m, &proof).unwrap();

        let (exp_cols, domain, exp_m) = expected_age_name_join(lc, op, c);
        assert_eq!(m, exp_m, "op {:?} c {}", op, c);
        assert_eq!(m, expected_m, "op {:?} c {}", op, c);
        assert_eq!(coms.len(), exp_cols.len());
        for (i, col) in exp_cols.iter().enumerate() {
            assert_eq!(
                commit_column(&pad(col, domain)).unwrap(),
                coms[i],
                "op {:?} c {} col {}",
                op,
                c,
                i
            );
        }
    }

    #[test]
    fn layer_bgp_object_range_all_ops_and_boundary_end_to_end() {
        // ages: alice=30, bob=15, carol=18, dave=40, erin=18. c = 18 straddles them and
        // hits the boundary (carol, erin == 18).
        let lc = build_age_name_layer();
        // Gt 18 -> {alice, dave}; Ge 18 -> {alice, carol, dave, erin}.
        check_age_range(&lc, RangeOp::Gt, 18, 2);
        check_age_range(&lc, RangeOp::Ge, 18, 4);
        // Lt 18 -> {bob}; Le 18 -> {bob, carol, erin}.
        check_age_range(&lc, RangeOp::Lt, 18, 1);
        check_age_range(&lc, RangeOp::Le, 18, 3);
    }

    #[test]
    fn layer_bgp_object_range_dict_domain_exceeds_table_end_to_end() {
        // D = 16 (dict) strictly exceeds the table domain 8: the object column relocates
        // UP to D, the dict columns stay at D, and range_sel relocates DOWN to the table.
        let lc = build_distinct_age_name_layer();
        // Sanity: the dict domain is larger than the table domain here.
        let n = lc.materialized.len();
        assert!(dict_domain_len(lc.node_value_catalog.len()).unwrap() > padded_len(n).unwrap());
        // ages p1=10, p2=25, p3=5. age >= 10 -> {p1, p2}.
        check_age_range(&lc, RangeOp::Ge, 10, 2);
    }

    #[test]
    fn layer_bgp_object_range_excludes_everything_m_zero() {
        // Nothing satisfies age > 1000: the ranged relation is empty, so the join is too.
        let lc = build_age_name_layer();
        let plan = age_range_plan(RangeOp::Gt, 1000);
        let (proof, coms, m, ids) = prove_layer_bgp(&lc, &plan).unwrap();
        assert_eq!(m, 0);
        verify_layer_bgp(&lc, lc.state.commitment_root, &plan, &ids, &coms, m, &proof).unwrap();
        let (_cols, _domain, exp_m) = expected_age_name_join(&lc, RangeOp::Gt, 1000);
        assert_eq!(exp_m, 0);
    }

    #[test]
    fn layer_bgp_two_sided_object_range_intersects_both_bounds() {
        // Half-open [18, 40): carol=18, erin=18, alice=30 are included; bob=15 and
        // dave=40 are excluded. The second mask must be present and multiplied with the
        // first before the age relation is compacted.
        let lc = build_age_name_layer();
        let mut plan = age_range_plan(RangeOp::Ge, 18);
        plan.atoms[0].object_range.as_mut().unwrap().second = Some(ObjectRangeBound {
            op: RangeOp::Lt,
            c: fr(40),
        });
        let (proof, coms, m, ids) = prove_layer_bgp(&lc, &plan).unwrap();
        assert_eq!(m, 3);
        verify_layer_bgp(&lc, lc.state.commitment_root, &plan, &ids, &coms, m, &proof).unwrap();

        // The upper bound is part of the public statement and its own authenticated
        // selection. Changing it cannot verify against the original proof.
        let mut wrong_bound = plan.clone();
        wrong_bound.atoms[0]
            .object_range
            .as_mut()
            .unwrap()
            .second
            .as_mut()
            .unwrap()
            .c = fr(41);
        assert!(verify_layer_bgp(
            &lc,
            lc.state.commitment_root,
            &wrong_bound,
            &ids,
            &coms,
            m,
            &proof,
        )
        .is_err());

        let mut missing_second = proof.clone();
        missing_second.atoms[0].range.as_mut().unwrap().second = None;
        assert!(verify_layer_bgp(
            &lc,
            lc.state.commitment_root,
            &plan,
            &ids,
            &coms,
            m,
            &missing_second,
        )
        .is_err());
    }

    #[test]
    fn layer_bgp_subject_object_self_loop_is_same_row_equality() {
        let lc = build_layer(&[
            ("a", "knows", "a"),
            ("a", "knows", "b"),
            ("c", "knows", "c"),
            ("x", "likes", "x"),
        ]);
        let mut atom = LayerBgpAtom::all_variables(KNOWS.to_vec());
        atom.subject_object_equal = true;
        let plan = LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![atom],
            joins: vec![],
        };
        let (proof, commitments, m, ids) = prove_layer_bgp(&lc, &plan).unwrap();
        assert_eq!(m, 2);
        assert_eq!(commitments.len(), 1);
        verify_layer_bgp(
            &lc,
            lc.state.commitment_root,
            &plan,
            &ids,
            &commitments,
            m,
            &proof,
        )
        .unwrap();

        let mut equality_removed = plan.clone();
        equality_removed.atoms[0].subject_object_equal = false;
        assert!(verify_layer_bgp(
            &lc,
            lc.state.commitment_root,
            &equality_removed,
            &ids,
            &commitments,
            m,
            &proof,
        )
        .is_err());

        let predicate_id = ids[0];
        let mut expected: Vec<ArkFr> = lc
            .materialized
            .iter()
            .filter(|row| row.predicate_id == predicate_id && row.subject_id == row.object_id)
            .map(|row| fr(row.subject_id))
            .collect();
        expected.resize(padded_len(lc.materialized.len()).unwrap(), f0());
        assert_eq!(commitments[0], commit_column(&expected).unwrap());
    }

    #[test]
    fn layer_bgp_object_range_wrong_expected_root_is_rejected() {
        let lc = build_age_name_layer();
        let plan = age_range_plan(RangeOp::Gt, 18);
        let (proof, coms, m, ids) = prove_layer_bgp(&lc, &plan).unwrap();
        let mut wrong_root = lc.state.commitment_root;
        wrong_root[0] ^= 0xff;
        assert!(verify_layer_bgp(&lc, wrong_root, &plan, &ids, &coms, m, &proof).is_err());
        // The correct external root still verifies.
        verify_layer_bgp(&lc, lc.state.commitment_root, &plan, &ids, &coms, m, &proof).unwrap();
    }

    #[test]
    fn layer_bgp_object_range_tampered_constant_or_datatype_is_rejected() {
        let lc = build_age_name_layer();
        let plan = age_range_plan(RangeOp::Gt, 18);
        let (proof, coms, m, ids) = prove_layer_bgp(&lc, &plan).unwrap();

        // Verifying against a DIFFERENT range constant: the constant is bound into the
        // statement AND the object-range sub-proof, so the transcript diverges -> REJECT.
        let plan_wrong_c = age_range_plan(RangeOp::Gt, 19);
        assert!(verify_layer_bgp(
            &lc,
            lc.state.commitment_root,
            &plan_wrong_c,
            &ids,
            &coms,
            m,
            &proof
        )
        .is_err());

        // Verifying against a DIFFERENT datatype tag (Boolean instead of UInt32) -> REJECT.
        let plan_wrong_dt = LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![
                LayerBgpAtom::all_variables(AGE.to_vec()).with_object_range(ObjectRangeSpec {
                    op: RangeOp::Gt,
                    datatype: fr(BOOLEAN_TAG),
                    c: fr(18),
                    b_bits: 64,
                    second: None,
                }),
                LayerBgpAtom::all_variables(NAME.to_vec()),
            ],
            joins: vec![JoinStep {
                acc_key: 0,
                rel_key: 0,
                additional_keys: Vec::new(),
            }],
        };
        assert!(verify_layer_bgp(
            &lc,
            lc.state.commitment_root,
            &plan_wrong_dt,
            &ids,
            &coms,
            m,
            &proof
        )
        .is_err());

        // A DIFFERENT operator (Ge instead of Gt) -> REJECT (and would also change m).
        let plan_wrong_op = age_range_plan(RangeOp::Ge, 18);
        assert!(verify_layer_bgp(
            &lc,
            lc.state.commitment_root,
            &plan_wrong_op,
            &ids,
            &coms,
            m,
            &proof
        )
        .is_err());

        // Honest still verifies.
        verify_layer_bgp(&lc, lc.state.commitment_root, &plan, &ids, &coms, m, &proof).unwrap();
    }

    #[test]
    fn layer_bgp_object_range_rejects_range_on_a_constant_object() {
        // A range on a CONSTANT object is out of v1 scope (validate_atom rejects it).
        let lc = build_age_name_layer();
        let plan = LayerBgpPlan {
            source: ColumnSource::Materialized,
            atoms: vec![
                LayerBgpAtom {
                    subject: AtomTerm::Variable,
                    predicate_term: AGE.to_vec(),
                    predicate_variable: false,
                    object: node_const("alice_n"),
                    object_range: Some(ObjectRangeSpec {
                        op: RangeOp::Gt,
                        datatype: fr(UINT32_TAG),
                        c: fr(18),
                        b_bits: 64,
                        second: None,
                    }),
                    object_kind: ObjectKindFilter::Any,
                    subject_object_equal: false,
                    source: ColumnSource::Materialized,
                },
                LayerBgpAtom::all_variables(NAME.to_vec()),
            ],
            joins: vec![JoinStep {
                acc_key: 0,
                rel_key: 0,
                additional_keys: Vec::new(),
            }],
        };
        assert!(prove_layer_bgp(&lc, &plan).is_err());
    }

    // A base layer carrying both `Node` objects and a string `Value` object, plus two
    // predicates — enough to exercise every namespace of the resolvers.
    fn mixed_layer() -> std::sync::Arc<LayerCommitment> {
        let store = open_sync_memory_store();
        let builder = store.create_base_layer().unwrap();
        for (s, p, o) in [
            ("a", "knows", "b"),
            ("b", "knows", "c"),
            ("a", "likes", "c"),
        ] {
            builder
                .add_value_triple(ValueTriple::new_node(&iri(s), &iri(p), &iri(o)))
                .unwrap();
        }
        // String-valued objects land in the shared node/value dictionary as `Value`s.
        builder
            .add_value_triple(ValueTriple::new_string_value(&iri("a"), &iri("label"), "hello"))
            .unwrap();
        builder
            .add_value_triple(ValueTriple::new_string_value(&iri("b"), &iri("label"), "world"))
            .unwrap();
        let layer = builder.commit().unwrap();
        layer.proof_commitment().unwrap()
    }

    // I1 DIFFERENTIAL TEST: resolving EVERY dictionary id through the new Merkle-opening
    // path must produce byte-identical terms to the old rebuild-the-catalog scan.
    #[test]
    fn merkle_resolution_matches_legacy_catalog_scan_for_every_id() {
        let lc = mixed_layer();
        let root = lc.state.commitment_root;

        // Columns holding EVERY node/value id and EVERY predicate id in the catalogs.
        let mut nv_ids: Vec<u64> = lc.node_value_catalog.iter().map(|r| r.object_id).collect();
        nv_ids.sort_unstable();
        let mut pred_ids: Vec<u64> = lc.predicate_catalog.iter().map(|r| r.predicate_id).collect();
        pred_ids.sort_unstable();
        assert!(nv_ids.len() >= 5 && !pred_ids.is_empty());
        let nv_col: Vec<ArkFr> = nv_ids.iter().map(|&id| ArkFr::from_u64(id)).collect();
        let pred_col: Vec<ArkFr> = pred_ids.iter().map(|&id| ArkFr::from_u64(id)).collect();

        // Node/value-only resolver: Merkle == legacy for every id.
        let merkle_nv =
            resolve_node_value_result_objects(&lc, root, std::slice::from_ref(&nv_col)).unwrap();
        let legacy_nv =
            resolve_node_value_result_objects_via_catalog(&lc, root, std::slice::from_ref(&nv_col))
                .unwrap();
        assert_eq!(merkle_nv, legacy_nv);
        // The layer really does carry both a Node and a Value term (so the differential
        // covers both reconstruction branches).
        assert!(merkle_nv[0]
            .iter()
            .any(|o| matches!(o, CanonicalObject::Node(_))));
        assert!(merkle_nv[0]
            .iter()
            .any(|o| matches!(o, CanonicalObject::Value { .. })));

        // Typed resolver over node/value + predicate + a nullable-zero column: Merkle == legacy.
        let columns = vec![nv_col.clone(), pred_col.clone(), vec![f0(), f0()]];
        let descriptors = vec![
            ResultColumnDescriptor {
                namespace: ResultIdNamespace::NodeValue,
                nullable: false,
            },
            ResultColumnDescriptor {
                namespace: ResultIdNamespace::Predicate,
                nullable: false,
            },
            ResultColumnDescriptor {
                namespace: ResultIdNamespace::NodeValue,
                nullable: true,
            },
        ];
        let merkle = resolve_result_values(&lc, root, &columns, &descriptors).unwrap();
        let legacy = resolve_result_values_via_catalog(&lc, root, &columns, &descriptors).unwrap();
        assert_eq!(merkle, legacy);
        // Predicate ids reconstruct as `Node(iri)` on both paths.
        assert!(matches!(
            merkle[1][0],
            CanonicalResultValue::Object(CanonicalObject::Node(_))
        ));
        assert_eq!(merkle[2][0], CanonicalResultValue::Null);
    }

    // The `resolution_root` wrapper binds `commitment_root` to the two Merkle roots WITHOUT
    // perturbing `commitment_root` itself (the additive I1 anchor).
    #[test]
    fn resolution_root_wraps_without_touching_commitment_root() {
        let lc = mixed_layer();
        let view = lc.verifier_view();
        assert_eq!(
            view.resolution_root(),
            crate::proof::resolution_root(
                &lc.state.commitment_root,
                &lc.nv_merkle_root,
                &lc.pred_merkle_root,
            )
        );
        let anchor = view.resolution_root();
        // Perturbing a Merkle root moves the anchor...
        assert_ne!(
            crate::proof::resolution_root(&lc.state.commitment_root, &[9u8; 32], &lc.pred_merkle_root),
            anchor
        );
        // ...but `commitment_root` is computed independently of the Merkle roots and is
        // unchanged (byte-identity of the base anchor is preserved under I1).
        assert_eq!(view.recomputed_commitment_root().unwrap(), lc.state.commitment_root);
    }
