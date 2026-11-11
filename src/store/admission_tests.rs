//! Proof admission over rollup-substituted ancestry, and deep proof-on chains.

use super::*;
use std::time::Instant;

/// Commit `depth` single-triple children on a fresh base, holding the head
/// between commits (as a writer does), and return per-commit timings sampled
/// around the given depths.
async fn deep_chain_timings(
    store: &Store,
    depth: usize,
    samples: &[usize],
) -> Vec<(usize, f64, f64)> {
    let base = store.create_base_layer().await.unwrap();
    base.add_value_triple(ValueTriple::new_node("s0", "p", "o0"))
        .unwrap();
    let mut head = base.commit().await.unwrap();
    let mut out = Vec::new();
    let mut window = Vec::new();
    for i in 1..=depth {
        let builder = head.open_write().await.unwrap();
        builder
            .add_value_triple(ValueTriple::new_node(&format!("s{i}"), "p", &format!("o{i}")))
            .unwrap();
        let start = Instant::now();
        head = builder.commit().await.unwrap();
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        window.push(elapsed);
        if window.len() > 20 {
            window.remove(0);
        }
        if samples.contains(&i) {
            let mean = window.iter().sum::<f64>() / window.len() as f64;
            eprintln!("depth {i}: mean commit {mean:.2} ms over the last {} commits", window.len());
            // Admission alone: rebuilding the head's proof state from its
            // delta, as commit does, separated from the native layer write.
            let start = Instant::now();
            for _ in 0..5 {
                store
                    .layer_store
                    .rebuild_proof_layer(head.name())
                    .await
                    .unwrap();
            }
            let admission = start.elapsed().as_secs_f64() * 1000.0 / 5.0;
            eprintln!("depth {i}: proof admission (rebuild_proof_layer) {admission:.2} ms");
            out.push((i, mean, admission));
        }
    }
    out
}

#[tokio::test]
#[ignore = "timing probe; run explicitly with --ignored --nocapture"]
async fn deep_proof_chain_commit_timings() {
    let depth: usize = std::env::var("DEEP_CHAIN_DEPTH")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(5000);
    let dir = tempfile::tempdir().unwrap();
    let store = open_directory_store(dir.path());
    let samples: Vec<usize> = [100usize, 500, 1000, 2000, 3000, 4000, 5000, 10000]
        .iter()
        .copied()
        .filter(|sample| *sample <= depth)
        .collect();
    deep_chain_timings(&store, depth, &samples).await;
}

use crate::storage::directory::DirectoryLayerStore;
use crate::storage::{PersistentLayerStore, VALIDATED_ROLLUP_VERSION};

/// The sidecar the unrolled path (the original physical chain, no rollups)
/// produces for `name`, encoded.
async fn unrolled_sidecar(store: &Store, name: [u32; 5]) -> Vec<u8> {
    let layer = store
        .layer_store
        .get_unrolled_layer(name)
        .await
        .unwrap()
        .unwrap();
    store
        .layer_store
        .rebuild_proof_layer_from(name, layer)
        .await
        .unwrap()
        .encode()
        .unwrap()
}

async fn assert_sidecars_match_unrolled(store: &Store, names: &[[u32; 5]]) {
    for name in names {
        let stored = store.layer_store.proof_state_bytes(*name).await.unwrap();
        assert_eq!(
            stored.as_ref(),
            unrolled_sidecar(store, *name).await.as_slice(),
            "sidecar of {name:?} differs from the unrolled path"
        );
        // Admission (rollup-substituted) rebuild agrees byte for byte too.
        let admitted = store
            .layer_store
            .rebuild_proof_layer(*name)
            .await
            .unwrap()
            .encode()
            .unwrap();
        assert_eq!(stored.as_ref(), admitted.as_slice());
    }
}

async fn child_with(
    parent: &StoreLayer,
    add: &[(&str, &str, &str)],
    remove: &[(&str, &str, &str)],
) -> StoreLayer {
    let builder = parent.open_write().await.unwrap();
    for (s, p, o) in add {
        builder
            .add_value_triple(ValueTriple::new_node(s, p, o))
            .unwrap();
    }
    for (s, p, o) in remove {
        builder
            .remove_value_triple(ValueTriple::new_node(s, p, o))
            .unwrap();
    }
    builder.commit().await.unwrap()
}

async fn check_sidecars_with_rollups(store: Store) {
    let base = store.create_base_layer().await.unwrap();
    base.add_value_triple(ValueTriple::new_node("a", "p", "b"))
        .unwrap();
    base.add_value_triple(ValueTriple::new_string_value("a", "label", "alpha"))
        .unwrap();
    let base = base.commit().await.unwrap();
    let mut names = vec![base.name()];
    let mut head = base.clone();
    for i in 0..4 {
        head = child_with(
            &head,
            &[(&format!("n{i}"), "q", "b"), ("a", &format!("p{i}"), "z")],
            if i == 2 { &[("a", "p", "b")] } else { &[] },
        )
        .await;
        names.push(head.name());
    }
    let upto = store.get_layer_from_id(names[1]).await.unwrap().unwrap();
    // A precise rollup up to the first child, then a full rollup.
    head.rollup_upto(&upto).await.unwrap();
    let rolled = store.get_layer_from_id(head.name()).await.unwrap().unwrap();
    for i in 4..7 {
        head = child_with(
            &if i == 4 { rolled.clone() } else { head.clone() },
            &[(&format!("m{i}"), "q", &format!("o{i}"))],
            if i == 5 { &[("n0", "q", "b")] } else { &[] },
        )
        .await;
        names.push(head.name());
    }
    head.rollup().await.unwrap();
    let rolled = store.get_layer_from_id(head.name()).await.unwrap().unwrap();
    for i in 7..10 {
        head = child_with(
            &if i == 7 { rolled.clone() } else { head.clone() },
            &[(&format!("k{i}"), &format!("r{i}"), "a")],
            if i == 8 { &[("m4", "q", "o4")] } else { &[] },
        )
        .await;
        names.push(head.name());
    }
    assert_sidecars_match_unrolled(&store, &names).await;
    // The whole-chain validation still accepts every layer.
    store
        .layer_store
        .validate_proof_chain_against_store(head.name())
        .await
        .unwrap();
}

#[tokio::test]
async fn admission_sidecars_match_unrolled_on_rolled_up_chains_memory() {
    check_sidecars_with_rollups(open_memory_store()).await;
}

#[tokio::test]
async fn admission_sidecars_match_unrolled_on_rolled_up_chains_directory() {
    let dir = tempfile::tempdir().unwrap();
    check_sidecars_with_rollups(open_directory_store(dir.path())).await;
}

#[tokio::test]
async fn admission_sidecars_match_unrolled_on_rolled_up_chains_archive() {
    let dir = tempfile::tempdir().unwrap();
    check_sidecars_with_rollups(open_archive_store(dir.path(), 64)).await;
}

#[tokio::test]
async fn admission_sidecars_match_unrolled_without_rollups() {
    let store = open_memory_store();
    let base = store.create_base_layer().await.unwrap();
    base.add_value_triple(ValueTriple::new_node("a", "p", "b"))
        .unwrap();
    let mut head = base.commit().await.unwrap();
    let mut names = vec![head.name()];
    for i in 0..6 {
        head = child_with(&head, &[(&format!("x{i}"), "p", "b")], &[]).await;
        names.push(head.name());
    }
    assert_sidecars_match_unrolled(&store, &names).await;
}

/// A chain, its head's name and a directory layer store over the same files.
async fn directory_chain() -> (tempfile::TempDir, Store, DirectoryLayerStore, Vec<StoreLayer>) {
    let dir = tempfile::tempdir().unwrap();
    let store = open_directory_store(dir.path());
    let raw = DirectoryLayerStore::new(dir.path());
    let base = store.create_base_layer().await.unwrap();
    base.add_value_triple(ValueTriple::new_node("a", "p", "b"))
        .unwrap();
    let mut layers = vec![base.commit().await.unwrap()];
    for i in 0..3 {
        let next = child_with(layers.last().unwrap(), &[(&format!("c{i}"), "p", "b")], &[]).await;
        layers.push(next);
    }
    (dir, store, raw, layers)
}

/// A base layer in `raw` with the same counts as the chain but other terms:
/// a forged rollup whose id map does not match the original's.
async fn forged_rollup(raw: &DirectoryLayerStore) -> [u32; 5] {
    let mut builder = raw.create_base_layer().await.unwrap();
    let name = builder.name();
    builder.add_value_triple(ValueTriple::new_node("A", "p", "B"));
    for i in 0..3 {
        builder.add_value_triple(ValueTriple::new_node(&format!("C{i}"), "p", "B"));
    }
    builder.commit_boxed().await.unwrap();
    raw.finalize_layer(name).await.unwrap();
    name
}

#[tokio::test]
async fn forged_rollup_is_rejected_at_registration() {
    let (_dir, _store, raw, layers) = directory_chain().await;
    let head = layers.last().unwrap().name();
    let forged = forged_rollup(&raw).await;
    let error = raw.register_rollup(head, forged).await.unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    assert!(!raw.layer_has_rollup(head).await.unwrap());
}

#[tokio::test]
async fn forged_rollup_registered_behind_the_stores_back_is_rejected_for_admission() {
    let (dir, _store, raw, layers) = directory_chain().await;
    let head = layers.last().unwrap().name();
    let forged = forged_rollup(&raw).await;
    // As older code or a hand edit would: a plain record, never validated.
    raw.write_rollup_file(head, forged).await.unwrap();
    drop(layers);
    // A fresh store (empty cache) commits a child of the head: admission meets
    // the unvalidated rollup, validates it, and fails closed.
    let store = open_directory_store(dir.path());
    let head_layer = store.get_layer_from_id(head).await.unwrap().unwrap();
    let builder = head_layer.open_write().await.unwrap();
    builder
        .add_value_triple(ValueTriple::new_node("d", "p", "b"))
        .unwrap();
    let error = builder.commit().await.err().unwrap();
    assert_eq!(error.kind(), io::ErrorKind::InvalidData, "{error}");
    assert_eq!(
        raw.read_rollup_record(head).await.unwrap(),
        Some((1, forged)),
        "a forged rollup is never marked validated"
    );
}

#[tokio::test]
async fn plain_rollup_record_is_validated_lazily_and_marked() {
    let (dir, store, raw, layers) = directory_chain().await;
    let head_name = layers.last().unwrap().name();
    let head_layer = store.layer_store.get_layer(head_name).await.unwrap().unwrap();
    let rollup = store.layer_store.perform_rollup(head_layer).await.unwrap();
    // Registered the way older code did: a plain record.
    raw.write_rollup_file(head_name, rollup).await.unwrap();
    drop(layers);
    drop(store);
    // A fresh store meets the plain record on its first admission.
    let store = open_directory_store(dir.path());
    let head = store.get_layer_from_id(head_name).await.unwrap().unwrap();
    let child = child_with(&head, &[("d", "p", "b")], &[]).await;
    assert_eq!(
        raw.read_rollup_record(head_name).await.unwrap(),
        Some((VALIDATED_ROLLUP_VERSION, rollup))
    );
    assert_sidecars_match_unrolled(&store, &[child.name()]).await;
}

#[tokio::test]
async fn registered_rollups_of_proof_layers_are_marked_validated() {
    let (_dir, store, raw, layers) = directory_chain().await;
    let head = layers.last().unwrap();
    head.rollup().await.unwrap();
    let (version, _) = raw.read_rollup_record(head.name()).await.unwrap().unwrap();
    assert_eq!(version, VALIDATED_ROLLUP_VERSION);
    let _ = store;
    let mid = &layers[2];
    mid.rollup_upto(&layers[0]).await.unwrap();
    let (version, _) = raw.read_rollup_record(mid.name()).await.unwrap().unwrap();
    assert_eq!(version, VALIDATED_ROLLUP_VERSION);
}

/// A proof-on chain deeper than the old 1024-layer admission cap keeps
/// committing, and per-commit time does not grow with depth.
#[tokio::test]
async fn proof_chain_commits_past_the_old_depth_cap() {
    let store = open_memory_store();
    let timings = deep_chain_timings(&store, 1100, &[100, 1100]).await;
    let (early, late) = (timings[0].2, timings[1].2);
    eprintln!("admission at depth 100: {early:.2} ms, at depth 1100: {late:.2} ms");
    // The unrolled admission this replaced grew ~linearly (11x here).
    assert!(late < early * 3.0 + 5.0, "proof admission time grows with depth");
}
