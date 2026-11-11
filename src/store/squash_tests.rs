//! Determinism of a squash: the same logical state reached through different
//! physical histories squashes to the same base layer and proof roots.

use super::*;
use crate::proof::ProofHash;
use crate::storage::directory::DirectoryLayerStore;
use crate::storage::{FileLoad, PersistentLayerStore};
use crate::storage::consts::FILENAME_ENUM_MAP;
use tdb_succinct::TdbDataType;

fn final_state() -> Vec<ValueTriple> {
    vec![
        ValueTriple::new_node("alice", "knows", "bob"),
        ValueTriple::new_node("bob", "knows", "carol"),
        ValueTriple::new_node("carol", "type", "Person"),
        ValueTriple::new_string_value("alice", "name", "Alice"),
        ValueTriple::new_string_value("bob", "name", "Bob"),
        ValueTriple::new_value("alice", "age", u32::make_entry(&31)),
        ValueTriple::new_value("bob", "age", i64::make_entry(&-7)),
        ValueTriple::new_value("carol", "score", u32::make_entry(&5)),
    ]
}

async fn commit_all(builder: StoreLayerBuilder, triples: &[ValueTriple]) -> StoreLayer {
    for triple in triples {
        builder.add_value_triple(triple.clone()).unwrap();
    }
    builder.commit().await.unwrap()
}

/// History 1: everything in one base layer.
async fn history_single_base(store: &Store) -> StoreLayer {
    commit_all(store.create_base_layer().await.unwrap(), &final_state()).await
}

/// History 2: one triple per commit, in reverse order, with detours whose
/// terms stay behind in the dictionaries: triples added and removed again,
/// a triple removed and re-added.
async fn history_with_detours(store: &Store) -> StoreLayer {
    let state = final_state();
    let mut head = commit_all(
        store.create_base_layer().await.unwrap(),
        &[ValueTriple::new_node("zed", "likes", "ghost")],
    )
    .await;
    for triple in state.iter().rev() {
        head = commit_all(head.open_write().await.unwrap(), &[triple.clone()]).await;
    }
    let detour = head.open_write().await.unwrap();
    detour
        .remove_value_triple(ValueTriple::new_node("zed", "likes", "ghost"))
        .unwrap();
    detour
        .remove_value_triple(state[0].clone())
        .unwrap();
    detour
        .add_value_triple(ValueTriple::new_value("dave", "age", u32::make_entry(&99)))
        .unwrap();
    head = detour.commit().await.unwrap();
    let back = head.open_write().await.unwrap();
    back.add_value_triple(state[0].clone()).unwrap();
    back.remove_value_triple(ValueTriple::new_value("dave", "age", u32::make_entry(&99)))
        .unwrap();
    back.commit().await.unwrap()
}

/// History 3: two halves in different splits, with a precise rollup and a
/// full rollup in between, so the head loads through rollups.
async fn history_with_rollups(store: &Store) -> StoreLayer {
    let state = final_state();
    let base = commit_all(store.create_base_layer().await.unwrap(), &state[..2]).await;
    let mid = commit_all(base.open_write().await.unwrap(), &state[2..4]).await;
    let more = commit_all(mid.open_write().await.unwrap(), &state[4..6]).await;
    more.rollup_upto(&base).await.unwrap();
    let more = store.get_layer_from_id(more.name()).await.unwrap().unwrap();
    let last = commit_all(more.open_write().await.unwrap(), &state[6..]).await;
    last.rollup().await.unwrap();
    let last = store.get_layer_from_id(last.name()).await.unwrap().unwrap();
    // One empty-effect commit on top of the rolled-up head.
    let extra = last.open_write().await.unwrap();
    extra
        .add_value_triple(ValueTriple::new_node("tmp", "p", "tmp"))
        .unwrap();
    let extra = extra.commit().await.unwrap();
    let undo = extra.open_write().await.unwrap();
    undo.remove_value_triple(ValueTriple::new_node("tmp", "p", "tmp"))
        .unwrap();
    undo.commit().await.unwrap()
}

/// Every file of a layer in a directory store, by name.
async fn layer_files(dir: &std::path::Path, name: [u32; 5]) -> Vec<(String, Vec<u8>)> {
    let raw = DirectoryLayerStore::new(dir);
    let mut files = Vec::new();
    let mut names: Vec<_> = FILENAME_ENUM_MAP.keys().copied().collect();
    names.sort();
    for file in names {
        if raw.file_exists(name, file).await.unwrap() {
            let handle = raw.get_file(name, file).await.unwrap();
            files.push((file.to_string(), handle.map().await.unwrap().to_vec()));
        }
    }
    files
}

async fn squash_in_fresh_directory(
    build: impl for<'a> FnOnce(&'a Store) -> std::pin::Pin<Box<dyn std::future::Future<Output = StoreLayer> + 'a>>,
) -> (
    DeterministicSquash,
    Vec<(String, Vec<u8>)>,
    (ProofHash, ProofHash),
    tempfile::TempDir,
) {
    let dir = tempfile::tempdir().unwrap();
    let store = open_directory_store(dir.path());
    let head = build(&store).await;
    let expected = store.squashed_roots(head.name()).await.unwrap();
    let squash = head.squash_deterministic().await.unwrap();
    let files = layer_files(dir.path(), squash.layer.name()).await;
    (squash, files, expected, dir)
}

#[tokio::test]
async fn same_logical_state_squashes_to_the_same_roots_and_files() {
    let results = vec![
        squash_in_fresh_directory(|s| Box::pin(history_single_base(s))).await,
        squash_in_fresh_directory(|s| Box::pin(history_with_detours(s))).await,
        squash_in_fresh_directory(|s| Box::pin(history_with_rollups(s))).await,
    ];
    let (first, first_files, _, _) = &results[0];
    assert!(!first_files.is_empty());
    for (squash, files, expected, _) in &results {
        assert_eq!(
            (squash.commitment_root, squash.resolution_root),
            *expected,
            "squashed_roots predicts squash_deterministic"
        );
        assert_eq!(squash.commitment_root, first.commitment_root);
        assert_eq!(squash.resolution_root, first.resolution_root);
        assert_eq!(
            files.iter().map(|(name, _)| name).collect::<Vec<_>>(),
            first_files.iter().map(|(name, _)| name).collect::<Vec<_>>()
        );
        for ((name, bytes), (_, first_bytes)) in files.iter().zip(first_files.iter()) {
            assert_eq!(bytes, first_bytes, "squashed file {name} differs");
        }
        assert!(squash.layer.parent_name().is_none());
        let mut triples: Vec<_> = squash
            .layer
            .triples()
            .map(|t| squash.layer.id_triple_to_string(&t).unwrap())
            .collect();
        triples.sort();
        let mut state = final_state();
        state.sort();
        assert_eq!(triples, state);
    }
    // Squashing a squash is a fixed point.
    let again = first.layer.squash_deterministic().await.unwrap();
    assert_eq!(again.commitment_root, first.commitment_root);
}

#[tokio::test]
async fn imported_history_squashes_to_the_same_root_as_the_exporter() {
    let exporter = open_memory_store();
    let head = history_with_detours(&exporter).await;
    let names = head.retrieve_layer_stack_names().await.unwrap();
    let pack = exporter
        .export_layers(Box::new(names.clone().into_iter()))
        .await
        .unwrap();
    let importer = open_memory_store();
    importer
        .import_layers(&pack, Box::new(names.into_iter()))
        .await
        .unwrap();
    let imported = importer.get_layer_from_id(head.name()).await.unwrap().unwrap();
    let local = head.squash_deterministic().await.unwrap();
    let remote = imported.squash_deterministic().await.unwrap();
    assert_eq!(local.commitment_root, remote.commitment_root);
    assert_eq!(local.resolution_root, remote.resolution_root);
    // And both match a store that only ever held the final state.
    let fresh = open_memory_store();
    let single = history_single_base(&fresh).await;
    assert_eq!(
        single.squash_deterministic().await.unwrap().commitment_root,
        local.commitment_root
    );
}

#[tokio::test]
async fn different_states_squash_to_different_roots() {
    let store = open_memory_store();
    let head = history_single_base(&store).await;
    let changed = head.open_write().await.unwrap();
    changed
        .add_value_triple(ValueTriple::new_node("x", "y", "z"))
        .unwrap();
    let changed = changed.commit().await.unwrap();
    assert_ne!(
        head.squash_deterministic().await.unwrap().commitment_root,
        changed.squash_deterministic().await.unwrap().commitment_root
    );
}

#[tokio::test]
async fn squash_deterministic_requires_proof_state() {
    let store = open_memory_store();
    let builder = store.create_base_layer().await.unwrap();
    builder
        .add_value_triple(ValueTriple::new_node("a", "b", "c"))
        .unwrap();
    let layer = builder.commit_without_proof().await.unwrap();
    assert!(layer.squash_deterministic().await.is_err());
}
