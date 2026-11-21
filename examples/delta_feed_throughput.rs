//! What a diff log costs to materialise, measured.
//!
//! A reactive cache wants to poll an append-only feed of per-commit diffs the
//! way it would poll a Kafka topic: often, cheaply, and with a latency that
//! does not depend on how deep the commit history is. TerminusDB already
//! writes that feed — every commit is an immutable delta layer whose additions
//! and removals are exactly the diff — but nothing serves it, and the question
//! is whether a background task can turn layers into feed records fast enough
//! to stay out of the write path's way.
//!
//! This is the measurement that decides it. It builds a layer stack with the
//! low-level store API (no Prolog, no server) and times the three stages a
//! materialiser performs once per commit:
//!
//!   1. **ids** — iterate the layer's own additions and removals.
//!      `internal_triple_additions`/`_removals` walk the layer's adjacency
//!      lists directly, so this is the floor: no dictionary, no allocation.
//!   2. **terms** — resolve each id to its logical term through
//!      `id_subject`/`id_predicate`/`id_object`. Random access into the
//!      mapped dictionaries, and the stage most likely to dominate.
//!   3. **record** — render one line of newline-delimited JSON per changed
//!      triple, which is what a feed would actually hand a client.
//!
//! Stage 2 is measured twice: once resolving every id, and once with a
//! per-commit memo, since a commit that touches many triples of one document
//! repeats the same subject id over and over. The difference is the answer to
//! "is a resolution cache worth it".
//!
//! Usage:
//!
//! ```text
//! cargo run --release --example delta_feed_throughput [base] [delta] [commits]
//! ```
//!
//! Defaults: a 200k-triple base, 10k-triple commits, 8 of them. `--release`
//! matters: a debug build measures rustc, not the store.

use std::collections::HashMap;
use std::env;
use std::fmt::Write as _;
use std::sync::Arc;
use std::time::{Duration, Instant};

use terminus_store::layer::{InternalLayer, LayerBuilder, ObjectType, ValueTriple};
use terminus_store::storage::archive::{
    ArchiveLayerStore, DirectoryArchiveBackend, LruArchiveBackend,
};
use terminus_store::storage::{CachedLayerStore, LayerStore, LockingHashMapLayerCache};
use terminus_store::Layer;

/// One changed triple, as a feed would carry it: logical terms, not store ids,
/// because an id means nothing to a client and nothing across databases.
struct Change {
    added: bool,
    subject: String,
    predicate: String,
    object: ObjectType,
}

fn render(change: &Change, out: &mut String) {
    out.push_str(if change.added {
        "{\"op\":\"add\",\"s\":\""
    } else {
        "{\"op\":\"del\",\"s\":\""
    });
    out.push_str(&change.subject);
    out.push_str("\",\"p\":\"");
    out.push_str(&change.predicate);
    match &change.object {
        ObjectType::Node(node) => {
            out.push_str("\",\"o\":\"");
            out.push_str(node);
            out.push_str("\"}\n");
        }
        ObjectType::Value(value) => {
            let _ = write!(
                out,
                "\",\"v\":\"{:?}\"}}\n",
                value.as_val::<String, String>()
            );
        }
    }
}

fn per_second(count: usize, elapsed: Duration) -> f64 {
    count as f64 / elapsed.as_secs_f64()
}

fn report(stage: &str, count: usize, elapsed: Duration) {
    println!(
        "  {stage:<22} {count:>9} in {:>8.1} ms = {:>10.0}/s   ({:.3} us each)",
        elapsed.as_secs_f64() * 1000.0,
        per_second(count, elapsed),
        elapsed.as_secs_f64() * 1e6 / count.max(1) as f64,
    );
}

/// A triple whose subject repeats across a run of predicates, the shape a
/// document write actually has: one subject, a dozen fields.
fn triple(entity: usize, field: usize) -> ValueTriple {
    if field % 3 == 0 {
        ValueTriple::new_node(
            &format!("http://example.org/e/{entity}"),
            &format!("http://example.org/p/{}", field % 12),
            &format!("http://example.org/e/{}", entity.wrapping_mul(7) % 100_000),
        )
    } else {
        ValueTriple::new_string_value(
            &format!("http://example.org/e/{entity}"),
            &format!("http://example.org/p/{}", field % 12),
            &format!("value for {entity}/{field}"),
        )
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let args: Vec<String> = env::args().collect();
    let parse = |i: usize, default: usize| {
        args.get(i)
            .map(|a| a.parse().expect("arguments must be numbers"))
            .unwrap_or(default)
    };
    let base_triples = parse(1, 200_000);
    let delta_triples = parse(2, 10_000);
    let commits = parse(3, 8);

    let dir = tempfile::tempdir().unwrap();
    let backend = DirectoryArchiveBackend::new(dir.path().to_path_buf());
    let lru = LruArchiveBackend::new(backend.clone(), backend, 512);
    let store = Arc::new(CachedLayerStore::new(
        ArchiveLayerStore::new(lru.clone(), lru),
        LockingHashMapLayerCache::new(),
    ));

    println!(
        "building: base {base_triples} triples, {commits} commits of {delta_triples} \
         (additions and removals), archive store at {}",
        dir.path().display()
    );

    let started = Instant::now();
    let mut builder: Box<dyn LayerBuilder> = store.create_base_layer().await.unwrap();
    let mut head = builder.name();
    for i in 0..base_triples {
        builder.add_value_triple(triple(i / 12, i % 12));
    }
    builder.commit_boxed().await.unwrap();
    println!(
        "  base committed in {:.1} s",
        started.elapsed().as_secs_f64()
    );

    // Each commit adds a fresh run of entities and removes an older one, so
    // every layer has both a positive and a negative delta.
    let mut layers = Vec::new();
    for commit in 0..commits {
        let mut child = store.create_child_layer(head).await.unwrap();
        head = child.name();
        let first_new = base_triples + commit * delta_triples;
        for i in first_new..first_new + delta_triples {
            child.add_value_triple(triple(i / 12, i % 12));
        }
        let first_old = commit * delta_triples;
        for i in first_old..first_old + delta_triples / 10 {
            child.remove_value_triple(triple(i / 12, i % 12));
        }
        child.commit_boxed().await.unwrap();
        store.finalize_layer(head).await.unwrap();
        layers.push(head);
    }

    println!("\nmaterialising each commit's diff, one layer at a time:");
    let mut totals = (
        (0usize, Duration::ZERO),
        (0usize, Duration::ZERO),
        (0usize, Duration::ZERO),
        (0usize, Duration::ZERO),
        (0usize, Duration::ZERO),
    );
    for (commit, name) in layers.iter().enumerate() {
        let layer: Arc<InternalLayer> = store.get_layer(*name).await.unwrap().unwrap();

        // 1. ids only.
        let start = Instant::now();
        let mut ids = Vec::with_capacity(delta_triples * 2);
        for id in layer.internal_triple_additions() {
            ids.push((true, id));
        }
        for id in layer.internal_triple_removals() {
            ids.push((false, id));
        }
        let ids_elapsed = start.elapsed();

        // 2. resolve every id.
        let start = Instant::now();
        let mut changes = Vec::with_capacity(ids.len());
        for (added, id) in &ids {
            changes.push(Change {
                added: *added,
                subject: layer.id_subject(id.subject).unwrap(),
                predicate: layer.id_predicate(id.predicate).unwrap(),
                object: layer.id_object(id.object).unwrap(),
            });
        }
        let terms_elapsed = start.elapsed();

        // 2b. resolve with a per-commit memo: a document write repeats its
        // subject across every field it touches, and predicates repeat across
        // every document of a class.
        let start = Instant::now();
        let mut subjects: HashMap<u64, String> = HashMap::new();
        let mut predicates: HashMap<u64, String> = HashMap::new();
        let mut memoed = 0usize;
        for (_, id) in &ids {
            let s = subjects
                .entry(id.subject)
                .or_insert_with(|| layer.id_subject(id.subject).unwrap());
            let p = predicates
                .entry(id.predicate)
                .or_insert_with(|| layer.id_predicate(id.predicate).unwrap());
            memoed += s.len() + p.len();
            let _ = layer.id_object(id.object).unwrap();
        }
        let memo_elapsed = start.elapsed();
        assert!(memoed > 0);

        // 3. render the feed records.
        let start = Instant::now();
        let mut out = String::with_capacity(ids.len() * 128);
        for change in &changes {
            render(change, &mut out);
        }
        let render_elapsed = start.elapsed();
        let bytes = out.len();

        println!(
            "\ncommit {commit} — {} changed triples ({} subjects, {} predicates distinct)",
            ids.len(),
            subjects.len(),
            predicates.len()
        );
        report("1 ids", ids.len(), ids_elapsed);
        report("2 ids+terms", ids.len(), terms_elapsed);
        report("2b ids+terms (memo)", ids.len(), memo_elapsed);
        report("3 render", ids.len(), render_elapsed);
        println!(
            "  {:<22} {:>9} bytes = {:.1} MB/s end to end",
            "record bytes",
            bytes,
            bytes as f64 / 1e6 / (terms_elapsed + render_elapsed).as_secs_f64()
        );

        totals.0 .0 += ids.len();
        totals.0 .1 += ids_elapsed;
        totals.1 .0 += ids.len();
        totals.1 .1 += terms_elapsed;
        totals.2 .0 += ids.len();
        totals.2 .1 += memo_elapsed;
        totals.3 .0 += ids.len();
        totals.3 .1 += render_elapsed;
        totals.4 .0 += bytes;
        totals.4 .1 += terms_elapsed + render_elapsed;
    }

    println!("\nacross {commits} commits:");
    report("1 ids", totals.0 .0, totals.0 .1);
    report("2 ids+terms", totals.1 .0, totals.1 .1);
    report("2b ids+terms (memo)", totals.2 .0, totals.2 .1);
    report("3 render", totals.3 .0, totals.3 .1);
    println!(
        "  {:<22} {:>9} bytes = {:.1} MB/s",
        "record bytes",
        totals.4 .0,
        totals.4 .0 as f64 / 1e6 / totals.4 .1.as_secs_f64()
    );
    let per_commit = (totals.1 .1 + totals.3 .1).as_secs_f64() * 1000.0 / commits as f64;
    let per_commit_memo = (totals.2 .1 + totals.3 .1).as_secs_f64() * 1000.0 / commits as f64;
    println!(
        "\nper commit: {per_commit:.1} ms to materialise {} changed triples \
         ({per_commit_memo:.1} ms memoed), so one core sustains {:.0} commits/s of this size \
         ({:.0} memoed) — or {:.1} M changed triples/s either way.",
        totals.1 .0 / commits,
        1000.0 / per_commit,
        1000.0 / per_commit_memo,
        totals.1 .0 as f64 / 1e6 / (totals.2 .1 + totals.3 .1).as_secs_f64(),
    );
}
