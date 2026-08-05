# Authenticated Store proof handoff

This is the durable resume checklist for the Store portion of
[`PARTURE_ASYNC_QUERY_PROOF_PLAN.md`](../../terminusdb/docs/PARTURE_ASYNC_QUERY_PROOF_PLAN.md).
It records the verified Store implementation and the remaining cross-repository
work. Work in dependency order; do not resume TerminusDB protocol integration
until an independent Store audit confirms the gate below.

## Repository state at handoff

- Store checkout: `/home/paseo/projects/parture/modules/terminusdb-store-paraplu`
- TerminusDB checkout: `/home/paseo/projects/parture/modules/terminusdb`
- Store base: detached `097aa5d32de037142f26935a98ba6604e8a91fb4`
- Store origin: `git@github.com:ParapluOU/terminusdb-store.git`
- TerminusDB branch: `parture-async-proof-trace`
- No commit or push has been made or authorized.
- Only the designated implementation agent may edit. Preserve all unrelated
  changes and do not delegate code writing.

The current Store tree is dirty in:

```text
Cargo.toml
README.md
src/lib.rs
src/storage/archive.rs
src/storage/cache.rs
src/storage/consts.rs
src/storage/directory.rs
src/storage/layer.rs
src/storage/memory.rs
src/storage/pack.rs
src/store/mod.rs
src/store/sync.rs
docs/AUTHENTICATED_PROOF_FORMAT.md       (new)
docs/AUTHENTICATED_PROOF_TODO.md         (new)
examples/proof_store_bench.rs            (new)
src/proof.rs                             (new)
```

`src/layer/internal/object_iterator.rs` matches `HEAD`. Do not run a repository-wide
formatter: this checkout's historical formatting would rewrite that unrelated
file. Format only files intentionally changed in this work.

On 2026-08-05, the post-audit `cargo test` passed all 330 tests plus doc tests,
and the focused pack gate passed 15 tests. `cargo check --lib --example
proof_store_bench` passed, and `git diff --check` was clean. These tests include
exact subprocess crash
points, two-process same-ID publication races, malformed pack containment,
proof/cache/resource-limit mutations, and differential normal/proof reads.
The independent Store audit is still required before protocol integration.

## Last verified capability

- [x] Optional proof mode; ordinary Store commits and reads remain untraced.
- [x] V4 canonical native per-layer addition/removal and complete dictionary
  catalog commitments, with parent-linked state roots.
- [x] SPO, predicate-subject-object, object-subject-predicate, and canonical
  typed-value authenticated orders.
- [x] Compact complete range evidence with boundaries/frontier and independent
  dictionary membership/non-membership verification.
- [x] Exact canonical scans plus differential `String` and `UInt32` half-open
  ranges; unsupported typed/reverse modes fail closed.
- [x] Independent checker reconstruction of roots, rows, additions, removals,
  non-canonical transitions, and chain ancestry.
- [x] Explicit legacy migration; proof child of legacy parent fails closed.
- [x] Rollup proof identity uses original unrolled files; squash/squash-up-to
  create new roots; all-proof base merge regenerates proof state.
- [x] Bounded pack v2 parsing, zero-copy private validation staging,
  out-of-band expected roots, adversarial mutation rejection, explicit legacy
  export, and an aggregate compressed/decoded/ancestry/proof/publication peak
  budget.
- [x] Directory/archive proof sidecars survive reopen; archive LRU proof
  migration is covered by the current green suite.
- [x] CPU-heavy proof rebuild, encode, decode, and witness construction run in
  blocking workers where ownership permits; cold proof loads are bounded
  single-flight and unresolved loads cannot be evicted and duplicated.
- [x] Directory and archive pack publication use durable transaction markers,
  cross-process/per-layer locks, atomic no-replace publication, hidden rollback,
  reopen recovery, and all-or-none logical visibility.
- [x] Proof sidecar publication is single-assignment for memory, directory, and
  archive stores; archive construction handles admit exactly one writer.

This is not yet an end-to-end query proof. The Store root/evidence is not wired
to the TerminusDB foreground iterator or protocol v3.

## Store verification completed in this cycle

- [x] Read the complete current diffs before editing, including directory and
  archive transaction recovery and proof-cache behavior.
- [x] Run `git diff --check`, then format only the changed Rust files listed
  below. Confirm `git diff -- src/layer/internal/object_iterator.rs` remains
  empty.
- [x] Compile immediately. Fix incomplete signatures/borrow errors without
  weakening fail-closed behavior.
- [x] Add regression tests for every reported audit blocker before moving to
  TerminusDB.
- [ ] Ask the coordinator for the pending Store audit result and feed every
  deficiency back through the same implementation agent.

Use this exact toolchain:

```sh
cd /home/paseo/projects/parture/modules/terminusdb-store-paraplu
export PATH=/nix/store/hiknqamwkn01i55z0jwmlvprdd2191ml-gnum4-1.4.21/bin:/home/paseo/.cargo/bin:/usr/local/bin:/usr/bin:/bin
/home/paseo/.cargo/bin/cargo check --lib --example proof_store_bench
```

When code is ready to format, target only the files changed by this work:

```sh
rustfmt --edition 2018 \
  src/proof.rs src/lib.rs \
  src/storage/archive.rs src/storage/cache.rs src/storage/consts.rs \
  src/storage/directory.rs src/storage/layer.rs src/storage/memory.rs \
  src/storage/pack.rs src/store/mod.rs src/store/sync.rs \
  examples/proof_store_bench.rs
git diff --check
```

## 1. Freeze source, format, and API identities

- [x] Finish the Store `0.22.0` breaking API boundary. Required transaction and
  conditional-publication methods cannot be default methods that silently
  overwrite or return `Unsupported` only after partial publication.
- [x] Confirm Store declares `tdb-succinct = "0.1.2"` and uses the exact
  standalone patch revision
  `ec1519923e44e73cad17264f21534065855be6e0`.
- [ ] Before path-pinning Store into TerminusDB, replace TerminusDB's branch
  patch with that exact revision and use a path dependency for this Store.
  Inspect `cargo tree -d` and `cargo tree -i tdb-succinct`; there must be one
  Cargo source identity and one `TypedDictEntry` type.
- [x] Document that proof file v4, pack v2, query protocol v3, and crate API
  0.22 are independent version axes. Add compatibility tests for each accepted
  combination.
- [x] Update all import docs to require every requested layer and reject the
  whole import when any requested layer is missing.
- [x] Make `pf3` wording truthful everywhere. V3 was development-only and never
  shipped. Directory native files can be explicitly rebuilt into v4. Old
  archive `ProofState` bytes are unsupported; no v3 decoder/tool currently
  exists and none should be claimed.

## 2. Bound all adversarial traversal and memory before allocation

- [x] Add cycle detection and depth checks inside parent traversal, before
  building an unbounded layer-name vector. Cover self-parent, two-node cycle,
  missing parent, and depth `limit + 1`.
- [x] Enforce the same early cycle/depth rules in pack ancestry collection and
  proof-chain validation.
- [x] Complete a conservative decoded-heap estimator before reconstructing
  ordinal arrays, dictionary maps, and Merkle levels. The encoded 256 MiB cap
  must not permit multi-gigabyte decoded state.
- [x] Reject hostile record counts and field lengths before `Vec::with_capacity`
  or equivalent allocation. Use checked `u64 -> usize`, addition,
  multiplication, and next-power-of-two operations.
- [x] Add an encode/output cap before growing a proof `Vec`; production proof
  construction must reject an object that cannot be encoded within the stated
  file budget.
- [x] Make witness budgeting incremental. Reserve aggregate layer, record,
  boundary, frontier, and canonical-byte budgets before cloning records or
  allocating frontier sets; do not build all layer witnesses and validate only
  afterward.
- [x] Apply equivalent bounded decoding to checker inputs and all pack-derived
  proof payloads. Add mutation tests for maximum counts, overflow counts,
  oversized fields, power-of-two expansion, and deep empty chains.

## 3. Make mutation and proof publication single-assignment

- [x] Finish rollup abort guards and add injected failures for file acquisition,
  delta generation, and finalize in `perform_rollup`, `perform_rollup_upto`,
  and imprecise rollup. Assert no construction directory/map entry survives.
- [x] Verify the existing proof commit, squash, squash-up-to, and merge guards
  still abort on every post-create error.
- [x] Replace Directory proof-state check-then-rename with an atomic no-replace
  primitive. On platforms without a safe no-replace operation, return
  `Unsupported`; never use metadata-check plus rename.
- [x] Add an archive compare-and-swap/conditional-replace primitive. Migration
  must atomically replace the exact archive bytes it read, or fail if another
  writer changed them. Do not reconstruct and unconditionally overwrite.
- [x] Test two simultaneous migrations/writes of the same proof sidecar. Exactly
  one publication wins; the final native archive and root remain valid; neither
  path panics or clobbers newer bytes.
- [x] Fsync proof bytes and every parent directory required to make the
  publication durable before reporting success.

## 4. Finish crash-atomic pack visibility

- [x] Complete the transaction-marker protocol for `DirectoryLayerStore`.
  Every staged layer must carry the same transaction token before its physical
  target can appear. All `directories`, `directory_exists`, `get_file`, and
  `file_exists` paths must ignore uncommitted transaction layers. One durable,
  atomically created global commit marker makes the whole batch visible.
- [x] Hold a cross-process import lock through stale-transaction recovery,
  collision checks, staging, target publication, parent-directory fsync, and
  commit-marker fsync. Recovery must remove or safely complete every
  uncommitted batch without deleting a live transaction.
- [x] Implement the same semantics for `DirectoryArchiveBackend`, including a
  marker that is present before an archive target appears so there is no
  marker-free visibility window.
- [x] Require archive backend capabilities for atomic batch publish,
  write-if-absent, compare-and-swap replacement, removal/recovery, and
  visibility checks. Custom backends must implement them or fail before any
  state change; unsafe default overwrite behavior is forbidden.
- [x] Make non-Linux no-replace implementations fail `Unsupported` unless a
  genuinely atomic primitive is provided. The previous TOCTOU metadata check
  is unacceptable.
- [x] Add subprocess crash/fault injection at the durable staging, visibility
  marker, target publication, and global commit boundaries.
  Real multi-layer writer processes and independent readers assert no requested
  layer before commit and the complete batch after commit. Recovery is also
  killed after quarantine rename and cleans residual hidden staging on restart.
  A post-commit directory-fsync failure returns the typed
  `ImportCommitDurabilityIndeterminate` outcome: the batch is logically visible,
  durability is unconfirmed, and callers must inspect/recover rather than
  blindly retry. `Ok` is returned only after the final fsync succeeds.
- [x] Test same-ID and overlapping multi-layer imports across two store handles
  and, for filesystem backends, two processes. Exactly one transaction wins.
- [x] Test reopen recovery before and after the commit point for both directory
  and archive stores.

## 5. Make pack limits symmetric and staging memory explicit

- [x] Finish `export_layers_with_limits` and enforce layer, entry, per-entry,
  uncompressed, and compressed budgets while exporting, before returning a
  pack. A successfully exported pack must be importable with the same limits.
- [x] Keep default `max_entry_bytes >= MAX_PROOF_LAYER_BYTES` or reduce the
  proof cap consistently. Do not advertise a valid proof file that the default
  pack path cannot transport.
- [x] Document the 512-layer default as a negotiated resource limit, not an
  undocumented export restriction. Test caller-raised and caller-lowered
  limits.
- [x] Count copied pre-existing ancestry bytes as well as ancestry layers before
  staging. Bound compressed input copies, decoded tar maps, native staging,
  proof decode/reconstruction, and publication buffers as one peak-memory
  model.
- [x] Stream gzip export into a bounded temporary file so compressed output
  never grows in a heap `Vec` beside a mapped native entry. Charge compressed
  output plus the mapped entry against `max_peak_bytes`, and read the final
  bounded result only after mappings and encoder buffers are gone. Run gzip/tar
  import decoding in a blocking worker, charging its owned compressed copy
  before allocation.
- [x] Eliminate avoidable clones: consume imported maps into publication, drop
  staging before publish, and use shared immutable bytes or streaming where the
  backend contract permits it.
- [ ] Add an OS-level peak-RSS benchmark near the default limit. Unit tests now
  cover aggregate peak rejection, bounded caller iterators, compressed-output
  rejection during encoding, malformed/missing/duplicate structures, copied
  ancestry, and skipped unrequested entries; process RSS sampling remains.

## 6. Bound and verify proof caching

- [x] Complete the proof cache's configurable entry and byte capacities.
  Preserve single-flight loading for resident keys and never hold a synchronous
  mutex across `.await`. Entry and in-flight limits are hard caps; the byte
  limit is a conservative logical accounting weight (capacity, containers,
  Arc/fixed allocator overhead), not an OS RSS cap.
- [x] Decide and test behavior when one decoded proof exceeds the byte limit
  (serve it to the caller but do not retain it).
- [x] Fix and test archive LRU named-entry replacement: removing a non-LRU entry
  must subtract the removed resolved value's bytes. A later eviction must not
  panic with positive accounting and an empty cache.
- [x] Add proof-cache tests for LRU eviction, byte eviction, invalidation,
  concurrent cold load, failed load retry, and exact/upper-bound retained byte
  reporting.
- [x] Ensure cache invalidation follows successful proof/archive replacement
  and never exposes stale proof state.

## 7. Store verification gate

- [x] Run focused proof tests (`proof_`: 27 passed; `proof::tests`: 14 passed):

  ```sh
  cargo test proof_
  cargo test proof::tests
  ```

- [x] Run pack/publication/cache/failure-injection tests for memory, directory,
  archive, and cached archive backends.
- [x] Run the final post-audit complete Store suite (330 passed plus doc tests):

  ```sh
  cargo test
  git diff --check
  ```

- [x] Confirm ordinary commits/queries never create, decode, or reconstruct
  proof state and remain result/order compatible.
- [x] Re-run every adversarial mutation: leaf, boundary, frontier, index,
  dictionary, parent root/count, sidecar transplant, expected pack root,
  manifest, archive proof replacement, early exhaustion, and resource-limit
  attack.
- [ ] Obtain another independent Store audit. Do not begin protocol integration
  while any soundness, atomicity, or bounded-resource finding remains open.

## 8. Integrate the native root into TerminusDB protocol v3

- [ ] Path-pin this verified Store and the exact ParapluOU dependency revisions
  in `/home/paseo/projects/parture/modules/terminusdb/src/rust/Cargo.toml`.
- [ ] At native commit finalization, publish a single-assignment binding from
  TerminusDB commit/catalog identity to the exact Store v4 `state_root`, parent
  root, and catalog counts. A producer must not be able to replace both a
  sidecar and a mutable binding undetected.
- [ ] Define a proof-aware foreground Store iterator API in
  `terminusdb-store-prolog` that uses the same native search traversal once and
  exposes stable bounds, dictionary IDs, candidate order, exhaustion, pruning,
  and compact range evidence. Background code may assemble retained page/hash
  material but must not rerun Store search.
- [ ] Remove protocol v2 full `SnapshotRef` export and `_proof_exports` from
  proof protocol v3. Do not carry the export forward as a fallback.
- [ ] Bind canonical query -> canonical proof plan deterministically and have
  the checker validate that lowering, rather than trusting a producer plan
  hash.
- [ ] Bind protocol version, executor semantics ID, checker contract, backend
  image/program/VK ID, request ID/nonce, native Store root, query hash, proof
  plan hash, ordered result root, deterministic cost, and trace root in the
  public statement.
- [ ] Preserve one foreground SWI execution. The response returns rows and a
  pending sealed-job ID; no background worker may call WOQL, Prolog search, or
  Store query APIs.
- [ ] Keep proof tracing optional/off by default. Unsupported operators,
  optimizer states, types, binding states, updates, retries, and control flow
  must fail before execution, never fall back to an untraced operator.
- [ ] Extend the initial supported subset only after the checker validates
  binding creation/restoration, success/failure, backtracking, pruning, and
  exhaustion. Use the existing ParapluOU SWI/swipl-rs forks; do not create
  replacement forks.

## 9. Checker and prover verification gate

- [ ] Update `terminusdb-query-proof-core` to consume v3 native Store evidence
  and independently derive complete scan matches, transitions, results, and
  cost without TerminusDB, Prolog, or Store access.
- [ ] Add golden vectors and differential traced/untraced tests for every
  supported WOQL operator and result order.
- [ ] Add adversarial omissions, insertions, reorderings, failed-branch skips,
  fake pruning/exhaustion, wrong binding restoration, wrong plan/query/root,
  cross-snapshot evidence, result/cost mutation, and request replay.
- [ ] Run native checker and durable sealed/aborted/recovery lifecycle tests.
- [ ] Run the RISC Zero guest with `RISC0_DEV_MODE=0`, verify its receipt against
  an independently derived expected statement/image ID, and mutate statement,
  journal, receipt, and trailing bytes.
- [ ] Run OpenVM application and aggregated-STARK verifier-only tests with
  pinned verifier material and exact inner/outer canonical decoding.
- [ ] Run Jolt/Dory tests while labelling that backend computational-integrity
  only and non-hiding; verify expected statement/program ID/proof/trailing-byte
  mutations fail.
- [ ] Confirm all backends accept the same sealed v3 witness and public
  statement without a second query execution.
- [ ] Keep generic trace/checker/prover/receipt code in the TerminusDB Rust
  workspace. Keep Parture market/payment authorization and settlement outside.

Useful TerminusDB commands, after the Store gate and path pin:

```sh
cd /home/paseo/projects/parture/modules/terminusdb/src/rust
cargo test -p terminusdb-query-trace
cargo test -p terminusdb-query-proof-core
cargo test -p terminusdb-query-prover
cargo test -p terminusdb-store-prolog

cargo run -p terminusdb-query-prover --bin terminusdb-query-native-check -- \
  check /path/to/job
RISC0_DEV_MODE=0 cargo test --release -p terminusdb-query-proof-risc0 --features prove
cargo test --release -p terminusdb-query-proof-openvm
cargo test --release -p terminusdb-query-proof-jolt
```

Confirm exact feature/test names against each `Cargo.toml`; do not silently
substitute development-mode or mocked proofs if a prover toolchain is missing.
Record the exact blocker and continue independent tests.

The foreground Prolog diagnostic is documented in
[`QUERY_TRACE_FORMAT.md`](../../terminusdb/docs/QUERY_TRACE_FORMAT.md). After
building `libterminusdb_dylib.so`, its current command is:

```sh
cd /home/paseo/projects/parture/modules/terminusdb
swipl -q -g "consult('src/load_paths.pl'), \
  use_foreign_library('src/rust/target/debug/libterminusdb_dylib.so'), \
  use_module(core(util/syntax)), \
  use_module(core(query/trace/benchmark),[]), \
  query_trace_benchmark:run(3), halt"
```

## 10. Performance acceptance and future format

- [ ] Replace the single-sample Store diagnostic with warmup plus repeated
  samples, median/p95/min/max, wall and CPU time, peak RSS/allocation where
  available, exact bytes, and checked row/root equality.
- [ ] Record rustc/Cargo version, profile/features, CPU, RAM, kernel, filesystem,
  Store commit, dataset seed/shape, layer count, and cache/reopen policy.
  “Cold” must state whether it means decoded-cache cold, process cold, or OS
  page-cache cold.
- [ ] Measure disabled proof-mode overhead separately for commit and query;
  measure proof commit/migration, first authenticated read, steady read,
  witness bytes/time, cache bytes/eviction, pack memory/time, and all mutation
  paths.
- [ ] Set explicit accepted regression budgets before claiming production
  readiness. The current 10.5-13.6x disk amplification and cold reconstruction
  are diagnostics, not budgets.
- [ ] Keep the current v4 limitation explicit: it reconstructs all ordinal
  projections and Merkle levels and is not page-addressable.
- [ ] Design the next shared-page format only after the vertical slice is
  correct: persist one canonical ID/catalog table plus paged ordinal/hash
  projections so a proof-aware iterator reads boundary pages and logarithmic
  authentication paths without rebuilding the whole layer.
- [ ] Benchmark the complete one-query foreground path and asynchronous proving
  path independently. Assert exactly one foreground query/Prolog/Store search,
  identical normal results, bounded trace/evidence, and no query work in the
  worker.

Latest recorded v4 Store diagnostic command and numbers (2026-08-05):

```sh
cargo run --release --example proof_store_bench -- 2000 100 16
```

- legacy base commit: 5.663 ms;
- v4 migration: 41.169 ms, 13.564x proof/native bytes;
- proof child: 18.754 ms versus 6.894 ms legacy, 10.544x bytes;
- squash: 158.071 ms, 13.257x bytes;
- merge: 253.048 ms, 13.282x bytes;
- 17-layer first authenticated scan: 31.615 ms, 49,152-byte sampled RSS
  delta, 2,479,433 estimated cache bytes, 2,016 rows;
- repeated scan: 2.970 ms with unchanged estimated cache bytes and row count.

These are single-run measurements on the current machine only.
