//! Stable-toolchain authenticated Store mutation benchmark.
//!
//! This measures Store index construction only. It never generates a query proof.

use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use tempfile::tempdir;
use terminus_store::layer::ValueTriple;
use terminus_store::storage::name_to_string;
use terminus_store::store::sync::SyncStoreLayer;
use terminus_store::Layer;

fn add_rows(
    builder: &terminus_store::store::sync::SyncStoreLayerBuilder,
    start: usize,
    count: usize,
) {
    for n in start..start + count {
        builder
            .add_value_triple(ValueTriple::new_node(
                &format!("subject-{n:010}"),
                "predicate",
                &format!("object-{n:010}"),
            ))
            .unwrap();
    }
}

fn layer_path(root: &Path, id: [u32; 5]) -> PathBuf {
    let name = name_to_string(id);
    root.join(&name[..3]).join(name)
}

fn directory_bytes(path: &Path, exclude_proof: bool) -> io::Result<u64> {
    let mut total = 0u64;
    for item in fs::read_dir(path)? {
        let item = item?;
        if item.file_type()?.is_file() && (!exclude_proof || item.file_name() != "proof_state.pf4")
        {
            total = total
                .checked_add(item.metadata()?.len())
                .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "layer byte count overflow"))?;
        }
    }
    Ok(total)
}

fn proof_bytes(root: &Path, layer: &SyncStoreLayer) -> io::Result<(u64, u64)> {
    let path = layer_path(root, layer.name());
    let native = directory_bytes(&path, true)?;
    let proof = fs::metadata(path.join("proof_state.pf4"))?.len();
    Ok((native, proof))
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn print_measurement(name: &str, elapsed: Duration, native: u64, proof: u64) {
    let amplification = if native == 0 {
        0.0
    } else {
        proof as f64 / native as f64
    };
    println!(
        "{name}: elapsed_ms={:.3} native_bytes={native} proof_bytes={proof} proof_to_native={amplification:.3}",
        milliseconds(elapsed)
    );
}

fn resident_bytes() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmRSS:"))?;
    let kib = line.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    kib.checked_mul(1024)
}

fn main() -> io::Result<()> {
    let mut args = env::args().skip(1);
    let base_rows = args
        .next()
        .map(|value| value.parse().expect("base_rows must be an integer"))
        .unwrap_or(10_000usize);
    let child_rows = args
        .next()
        .map(|value| value.parse().expect("child_rows must be an integer"))
        .unwrap_or(250usize);
    let deep_layers = args
        .next()
        .map(|value| value.parse().expect("deep_layers must be an integer"))
        .unwrap_or(16usize);

    let root = tempdir()?;
    let store = terminus_store::open_sync_directory_store(root.path());

    let legacy_builder = store.create_base_layer()?;
    add_rows(&legacy_builder, 0, base_rows);
    let started = Instant::now();
    let legacy = legacy_builder.commit()?;
    let legacy_commit = started.elapsed();
    println!(
        "legacy_base_commit: elapsed_ms={:.3}",
        milliseconds(legacy_commit)
    );

    let started = Instant::now();
    legacy.migrate_proof_chain()?;
    let migration = started.elapsed();
    let (native, proof) = proof_bytes(root.path(), &legacy)?;
    print_measurement("large_base_migration", migration, native, proof);

    let proof_parent_builder = store.create_base_layer()?;
    add_rows(&proof_parent_builder, base_rows * 2, base_rows);
    let (proof_parent, _) = proof_parent_builder.commit_with_proof()?;
    let proof_child_builder = proof_parent.open_write()?;
    add_rows(&proof_child_builder, base_rows * 4, child_rows);
    let started = Instant::now();
    let (proof_child, _) = proof_child_builder.commit_with_proof()?;
    let proof_child_commit = started.elapsed();
    let (native, proof) = proof_bytes(root.path(), &proof_child)?;
    print_measurement(
        "small_proof_child_commit",
        proof_child_commit,
        native,
        proof,
    );

    let legacy_parent_builder = store.create_base_layer()?;
    add_rows(&legacy_parent_builder, base_rows * 6, base_rows);
    let legacy_parent = legacy_parent_builder.commit()?;
    let legacy_child_builder = legacy_parent.open_write()?;
    add_rows(&legacy_child_builder, base_rows * 8, child_rows);
    let started = Instant::now();
    let _legacy_child = legacy_child_builder.commit()?;
    println!(
        "small_legacy_child_commit: elapsed_ms={:.3}",
        milliseconds(started.elapsed())
    );

    let started = Instant::now();
    let squashed = proof_child.squash()?;
    let squash = started.elapsed();
    let (native, proof) = proof_bytes(root.path(), &squashed)?;
    print_measurement("proof_squash", squash, native, proof);

    let second_builder = store.create_base_layer()?;
    add_rows(&second_builder, base_rows * 10, base_rows);
    let (second, _) = second_builder.commit_with_proof()?;
    let merge_temp = tempdir()?;
    let started = Instant::now();
    let merged_id =
        store.merge_base_layers(&[proof_parent.name(), second.name()], merge_temp.path())?;
    let merge = started.elapsed();
    let merged = store
        .get_layer_from_id(merged_id)?
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "merged layer not found"))?;
    let (native, proof) = proof_bytes(root.path(), &merged)?;
    print_measurement("proof_merge", merge, native, proof);

    let cold_root = tempdir()?;
    let deep_head = {
        let deep_store = terminus_store::open_sync_directory_store(cold_root.path());
        let builder = deep_store.create_base_layer()?;
        add_rows(&builder, base_rows * 20, base_rows);
        let (mut head, _) = builder.commit_with_proof()?;
        for layer_number in 0..deep_layers {
            let builder = head.open_write()?;
            add_rows(&builder, base_rows * 30 + layer_number, 1);
            head = builder.commit_with_proof()?.0;
        }
        head.name()
    };
    let cold_store = terminus_store::open_sync_directory_store(cold_root.path());
    let cold_head = cold_store
        .get_layer_from_id(deep_head)?
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "deep proof head not found"))?;
    let rss_before = resident_bytes();
    let started = Instant::now();
    let commitment = cold_head
        .proof_commitment()?
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "deep proof commitment missing"))?;
    let cold_load = started.elapsed();
    let root_hash = commitment.state.commitment_root;
    drop(commitment);
    let rss_after = resident_bytes();
    let decoded_cache = cold_store.proof_cache_memory_bytes();
    let started = Instant::now();
    let steady_commitment = cold_head
        .proof_commitment()?
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "cached proof commitment missing"))?;
    let steady_load = started.elapsed();
    let steady_cache = cold_store.proof_cache_memory_bytes();
    assert_eq!(steady_commitment.state.commitment_root, root_hash);
    assert_eq!(
        steady_cache, decoded_cache,
        "steady load rebuilt cached commitments"
    );
    println!(
        "cold_deep_commitment_load: layers={} elapsed_ms={:.3} rss_delta_bytes={} decoded_cache_bytes={}",
        deep_layers + 1,
        milliseconds(cold_load),
        rss_after.unwrap_or(0).saturating_sub(rss_before.unwrap_or(0)),
        decoded_cache,
    );
    println!(
        "steady_deep_commitment_load: layers={} elapsed_ms={:.3} decoded_cache_bytes={}",
        deep_layers + 1,
        milliseconds(steady_load),
        steady_cache,
    );

    Ok(())
}
