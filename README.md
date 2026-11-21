# terminusdb-store, a tokio-enabled data store for triple data

## Proof-of-WOQL store support

### The `proof` feature

Proving is **opt-in at compile time**: `proof` is not a default feature. With it
off this crate is upstream terminus-store — the same triples, the same layer
files, none of the Dory/Arkworks stack (some 50 crates of field and curve
arithmetic) in the dependency graph. That matters because most consumers reach
this crate transitively, through TerminusDB's `terminusdb-store-prolog` or the
grpc label-store client, and a transitive consumer cannot pass
`default-features = false` on someone else's dependency declaration. Opting in,
it can do:

```toml
terminus-store = { git = "...", branch = "parture-proof-of-woql", features = ["proof"] }
```

The storage API keeps its shape either way: `proof_layer`, `write_proof_layer`,
`migrate_proof_chain` and `import_layers_with_expected_roots` are all still
there with the feature off, and each fails with `io::ErrorKind::Unsupported` and
a sentence naming the feature, rather than silently reporting a layer as
unproven. `TERMINUSDB_PROOF_STATE=on` is likewise a startup error, not a quiet
"not proving". The scheme-independent part — the canonical
datatype/object/triple model, `ProofRecord`, the sidecar framing and its limits
— is always compiled, since the storage plumbing names those types in
signatures it keeps either way.

A store that never proves is byte-identical either way: proof state is a
separate file (`proof_state.pf4`), so a layer written by a feature-off build is
exactly a layer an unforked terminus-store would have written, and a build that
later turns the feature on can produce the sidecar from the native delta with
`migrate_proof_chain`. A sidecar arriving in a pack on a feature-off build is
dropped rather than kept unvalidated, so a later proving build never trusts
bytes an earlier one admitted blind.

This crate's own test suite is mostly the proof system; develop it with
`cargo test --features proof` (or `make test`), and check the other
configuration with `make test-bare`.

### What this fork is

This ParapluOU fork is the Store half of TerminusDB's Proof-of-WOQL capability.
The current cross-repository design and status live in
[`doc/proof-of-woql/overview.md`](../../doc/proof-of-woql/overview.md). The older
authenticated-sidecar and guest-VM plans have been retired; this implementation
uses native Dory column commitments, multilinear sumcheck, and LogUp arguments.

Every committed layer includes its intrinsic authenticated columns. Query proving
is still explicitly invoked by the caller and does not belong in Store mutation or
the normal query path. Scheduling, persistence, and proof-generation lifecycle are
deliberately left to the application or `terminusdb-rs` domain layer.
Parture payment policy is intentionally outside this generic Store and prover
framework.

`StoreProverContext` selects prover arithmetic without changing storage or proof
formats. Its stock and Blitzar-G1 constructors retain Arkworks BN254 G2, while
`cpu_with_g2_backend` and `blitzar_with_g2_backend` accept a caller-owned exact
`ArkG2` provider for profiling or a future accelerator. Blitzar 4.3.0 does not
expose the BN254 G2 twist/Fq2 ABI, so Store never coerces G2 through its G1
handle.

The active implementation commits materialized layer columns with Dory, proves
constant filtering and one- or two-sided object ranges, and composes connected single-key BGPs with
sumcheck and LogUp arguments. It supports one or more atoms. Verification is anchored
to an independently trusted `ProofHash`; a root supplied by the prover is not a trust
anchor.

### Retired sidecar design (historical context)

The remainder of this Parture section describes the earlier Merkle-sidecar/protocol
prototype. It is retained as historical context only and does not describe the active
Dory query-proof implementation or its API.

### Why protocol v3 must remove full logical export

The current TerminusDB protocol v2 vertical slice authenticates a canonical
full-snapshot export. It was useful to prove the checker and real prover paths,
but it is not the final architecture: materializing a full snapshot duplicates
Store work, is not derived from the same native index ranges used by the
foreground search, and leaves a privileged producer able to replace both the
catalog binding and export. Protocol v3 must bind the query directly to a
native authenticated Store root and complete range/dictionary evidence. It
must not create, transfer, or trust a full logical export.

### Store v4 design: verified state

The verified Store cycle implements a mandatory `proof_state.pf4` commitment file.
It authenticates each native layer's complete additions, removals, and newly
allocated dictionary interval. Each canonical leaf binds the native S/P/O IDs,
canonical logical bytes, and its ordered position. The manifest commits exact
parent/current catalog counts and roots for:

- native SPO, predicate-subject-object, and object-subject-predicate orders;
- a canonical typed-value order; and
- complete subject, predicate, and object dictionary catalogs.

The layer `state_root` domain-separates and commits those roots, counts, the
parent `state_root`, and the format contract. Chaining those manifests gives a
parent-linked provenance root for the exact layer history; random 160-bit layer
names remain physical locators, never trust anchors. Squash and merge can
legitimately produce different roots for the same visible triples because they
change authenticated history and catalogs.

Range evidence includes all returned records, predecessor/successor boundaries,
and a Merkle frontier. Dictionary evidence proves membership or
non-membership, so a verifier can establish complete range results rather than
membership of selected rows. Exact S/P/O scans cover canonical Store
datatypes, including `DateTimeInterval`. Typed half-open ranges are currently
supported only for `String` and `UInt32`; other typed ranges and unsupported
reverse modes fail closed before evidence construction.

The independent Rust verifier recomputes leaf, index, manifest, and parent
roots and applies additions/removals itself. It does not trust a
producer-recorded scan outcome. This Store checker is necessary but not yet
wired into the foreground WOQL iterator or protocol v3.

Version 4 persists each canonical delta/catalog record once and reconstructs
the ordinal projections and Merkle levels on first authenticated load. The
decoded projections share canonical record tables and are cached. This reduced
measured proof/native disk amplification substantially, but the format is not
page-addressable and still reconstructs every projection on a cold proof read.

### Mutation, migration, and history semantics

- Every new base or child commit builds authentication data from native layer
  files before the layer becomes visible; this does not generate a query proof.
- Children require an intrinsically committed parent and fail closed otherwise.
- Rollups are physical accelerators and retain the original unrolled proof
  identity. Each physical rollup also carries a commitment for its own native
  representation.
- Admission of a new layer resolves its delta's ids against its own files over
  the cached or rolled-up ancestry, so it costs O(delta) at any chain depth and
  writes sidecars byte-identical to the unrolled chain's. A rollup stands in
  for a layer during admission only once it is validated to resolve every id of
  that layer to the same term with the same counts: rollups of proof-carrying
  layers are validated when registered (record version 2); a rollup registered
  otherwise (older code, by hand) is validated on first use, and admission
  fails closed when it does not hold. Rollup records are local and never travel
  in packs.
- Whole-chain operations (chain validation, migration, pack import validation)
  walk the original chain once, base first, without a depth cap.
- Squash and squash-up-to create new authenticated history and new roots.
  `StoreLayer::squash_deterministic` gives a base layer whose files and roots
  are a pure function of the logical state, whatever the history;
  `Store::squashed_roots` computes those roots without writing.
- Base merge always regenerates authenticated state; legacy inputs must first be
  migrated explicitly.
- `pf3` was a development-only format and was never shipped. There is no v3
  archive decoder or migration tool. Directory data can be rebuilt explicitly
  from native files into v4; old archive `ProofState` payloads are unsupported
  and fail closed.

### Packs, roots, and publication visibility

Pack v2 bounds parsing and aggregate peak memory, requires intrinsic PF4 data for
every requested layer, validates native layers and proof chains in shared-byte
private staging, and rejects collisions. Independently supplied expected roots
are optional authenticity anchors; intrinsic consistency is always checked. A
legacy pack must be migrated and re-exported. The pack is transport, not a trust
source: a root received from the same untrusted pack does not authenticate
anything.

Export streams compressed bytes through a bounded temporary file, and import
gzip/tar parsing runs in a peak-charged blocking worker. Directory and archive
publication is an atomic visibility transaction.
Imported physical targets carry a transaction marker and remain invisible
until one durable global commit marker exists. Cross-process and per-layer
locks serialize recovery and conflicting publication; rollback first moves
uncommitted targets to hidden quarantine. `Ok` is returned only after the
commit-marker directory fsync; a failure after logical visibility is a typed
indeterminate, no-blind-retry result. Exact multi-layer subprocess crash-point
tests, crash-during-quarantine restart, reader all-or-none checks,
overlapping-batch races, and two-process same-ID races cover both filesystem
backends. Custom archive backends must implement the required
atomic batch, write-if-absent, compare-and-swap, recovery, and visibility
operations rather than inheriting an unsafe fallback.

### Storage/API version and exact TerminusDB integration

The in-progress breaking Store API is labelled `0.22.0`; proof file format v4
is a separate version. Before compiling TerminusDB against this checkout:

1. finish and verify the Store audit TODO;
2. make both workspaces resolve exactly one `tdb-succinct` source/revision;
3. path-pin TerminusDB's `terminus-store` dependency to this ParapluOU checkout;
4. expose a proof-aware native iterator that searches once and returns canonical
   bounds, dictionary bindings, exhaustion, and compact evidence from that same
   index traversal;
5. replace protocol v2 `SnapshotRef`/full export with a protocol v3 Store root
   and evidence reference, binding the commit catalog record to the exact v4
   root and catalog counts;
6. bind the canonical query, proof plan, options, executor/checker contract,
   request nonce, state root, ordered result root, deterministic cost, and
   backend image/program identity in the public statement;
7. expose the same native verifier through the Rust and TerminusDB integration
   boundaries without rerunning Prolog or Store search;
8. preserve the existing atomic sealed/aborted job lifecycle and keep all
   unsupported WOQL/control states fail-closed before execution.

The trust boundary is the independently anchored Store `state_root` plus the
versioned query/checker/program identities. A proof only establishes complete
execution relative to those public inputs; it does not establish that a random
layer locator, a root supplied inside its own pack, or a producer-controlled
sidecar is authoritative.

### Current status and honest limits

On 2026-08-05 the Store suite passed all 330 tests plus doc tests, the focused
proof-name filter passed 27 tests, the `proof::tests` module passed 14 tests,
and the pack filter passed 15 tests. This includes adversarial proof/pack
mutations, bounded-resource cases, subprocess crash recovery, and
cross-process publication races. A release
diagnostic using 2,000 base triples, 100 child triples, and 16 child layers
reported 10.5-13.6x proof/native disk amplification. A fresh 2026-08-05 release
run measured 31.615 ms for the first 17-layer authenticated scan, 2.970 ms for
the repeated scan, 2,479,433 estimated decoded-cache bytes, and a 49,152-byte
sampled RSS delta. These are single-run diagnostics, not accepted performance
budgets.

Crash-atomic pack visibility, exclusive proof-state publication, early
cycle/depth and heap guards, bounded-cache behavior, and pack-budget symmetry
are implemented and covered by the Store suite. An independent Store audit and
an OS-level peak-RSS acceptance benchmark remain. The v4 Store is not yet
integrated into TerminusDB's foreground Prolog path. Do not describe this
repository as an end-to-end query proof, a page-addressable proof store, or a
production-ready migration.

[![Build Status](https://github.com/terminusdb/terminusdb-store/workflows/Build/badge.svg)](https://github.com/terminusdb/terminusdb-store/actions)
[![Crate](https://img.shields.io/crates/v/terminus-store.svg)](https://crates.io/crates/terminus-store)
[![Documentation](https://docs.rs/terminus-store/badge.svg)](https://docs.rs/terminus-store/)
[![codecov](https://codecov.io/gh/terminusdb/terminusdb-store/branch/main/graph/badge.svg)](https://codecov.io/gh/terminusdb/terminusdb-store)

## Overview
This library implements a way to store triple data - data that
consists of a subject, predicate and an object, where object can
either be some value, or a node (a string that can appear both in
subject and object position).

An example of triple data is:
````
cow says value(moo).
duck says value(quack).
cow likes node(duck).
duck hates node(cow).
````
In `cow says value(moo)`, `cow` is the subject, `says` is the
predicate, and `value(moo)` is the object.

In `cow likes node(duck)`, `cow` is the subject, `likes` is the
predicate, and `node(duck)` is the object.

terminusdb-store allows you to store a lot of such facts, and search
through them efficiently.

This library is intended as a common base for anyone who wishes to
build a database containing triple data. It makes very few assumptions
on what valid data is, only focusing on the actual storage aspect.

This library is tokio-enabled. Any i/o and locking happens through
futures, and as a result, many of the functions in this library return
futures. These futures are intended to run on a tokio runtime, and
many of them will fail outside of one. If you do not wish to use
tokio, there's a small sync wrapper in `store::sync` which embeds its
own tokio runtime, exposing a purely synchronous API.

## Usage
Add this to your `Cargo.toml`:

```toml
[dependencies]
terminus-store = "0.19.2"
```

create a directory where you want the store to be, then open that store with
```rust
let store = terminus_store::open_directory_store("/path/to/store").await.unwrap();
```

Or use the sync wrapper:
```rust
let store = terminus_store::open_sync_directory_store("/path/to/store").unwrap();
```

For more information, [visit the documentation on docs.rs](https://docs.rs/terminus-store/).

See also the `examples/` directory for some basic examples.

## Upgrading from 0.19 or earlier
Starting with version 0.20.0, terminus-store uses a new storage format, which bundles all files into a single archive, and also supports value types. Stores created using 0.19 or earlier will not work with 0.20 or later. However, there is a conversion tool to convert existing pre-v20 stores: [terminusdb-10-to-11](https://github.com/terminusdb/terminusdb-10-to-11/).

## Roadmap

We are constantly developing terminusdb-store to make it a high quality succinct graph representation versioned datastorage layer. To help facilitate understanding of our aims for this project we have laid out a [Roadmap](./docs/ROADMAP.md). If you would like to assist in the development of terminusdb-store, or you think something should be added to the roadmap please contact us.

## License
terminus-store is licensed under Apache 2.0.

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md)

## See also
- The Terminus database, for which this library was written: [Website](https://terminusdb.com) - [GitHub](https://github.com/terminusdb/)
- Our prolog bindings for this library: [terminus_store_prolog](https://github.com/terminusdb/terminus_store_prolog/)
- The HDT format, which the terminusdb-store layer format is based on: [Website](http://www.rdfhdt.org/)
