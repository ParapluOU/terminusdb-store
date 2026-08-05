# Authenticated layer proof format

This fork optionally writes an authenticated logical-delta sidecar for a native
Terminus Store layer. Normal commits and queries do not write or read the
sidecar. `commit_with_proof` and the explicit `migrate_proof_chain` operation
build it; neither operation invokes RISC Zero or any other ZK prover.

## Version 4 contract

The directory filename is `proof_state.pf4`. Archive stores use the appended
`ProofState` archive-presence bit. The file begins with `TSPRF4`, carries format
version 4, and ends with a domain-separated SHA-256 file checksum. The checksum
detects corruption; the externally retained `state_root` is the security
anchor.

The manifest commits:

- the parent state root and exact parent/current dictionary counts;
- addition and removal counts;
- four roots for each delta: native SPO, native predicate-subject-object,
  native object-subject-predicate, and canonical typed-value order;
- subject, predicate, and object dictionary roots; and
- all of the preceding fields in the layer `state_root`.

Every leaf binds its ordered position, native subject/predicate/object IDs, and
canonical logical bytes. Production construction enumerates the complete new
contiguous node/value and predicate ID intervals, including unused dictionary
entries. It rejects gaps, inherited-ID repetition, non-bijective mappings, and
delta records whose new IDs do not match catalog bytes. The child state commits
the parent root and parent counts, making the snapshot root a chain commitment.
Layer names remain random immutable locators and are not security identities.

Version 4 stores one canonical addition table, one canonical removal table, one
predicate catalog, and one node/value catalog. Subject catalog entries are
derived from the node subset. The other orderings and Merkle levels are
deterministic ordinal projections rebuilt, fully root-checked, and Arc-cached on
the first authenticated read. All four delta projections share one decoded
record table. A normal untraced Store read never performs this rebuild.

Version 3 was a development-only sidecar and was never shipped. It is
deliberately not accepted as version 4. Directory layers that contain only
`proof_state.pf3` are treated as legacy and can be explicitly rebuilt from
their native files into v4. An archive containing an old `ProofState` payload
fails closed because archive presence bits do not encode a filename version.
No v3 archive decoder or migration tool exists; such development archives are
unsupported. There is no implicit query-time migration.

## Query evidence

Authenticated range witnesses contain predecessor/successor boundaries and a
Merkle frontier, proving the complete carrier interval rather than membership
of returned rows alone. Dictionary membership or non-membership binds query
terms to native IDs. The independent verifier recomputes leaf, tree, index,
manifest, and parent-chain roots; applies removals/additions itself; rejects
non-canonical no-op transitions; and derives the final rows.

Exact-value scans support every canonical datatype represented by the pinned
ParapluOU `tdb-succinct`, including `DateTimeInterval`. Typed half-open ranges
currently support only `String` and `UInt32`, the two orderings covered by
differential native-Store tests. Other typed ranges and non-range reverse scans
fail closed before evidence construction.

Verifier defaults reject proof files larger than 256 MiB, chains deeper than
1,024 layers, and aggregate witnesses above one million records, one million
frontier nodes, or 64 MiB of estimated payload. Integer conversions and size
arithmetic are checked.

## Mutation, rollup, squash, and merge

Authenticated index construction happens after the native builder has emitted
its delta and before backend publication. It is delta/catalog bounded and does
not enumerate the visible snapshot. Any failure after layer-directory creation
aborts the new layer for normal proof commits, squash, squash-up-to, and base
merge.

Rollups are physical accelerators only. Proof migration, validation, and
witness generation load the original unrolled layer files, so a rollup neither
changes nor substitutes the proof identity. Squash and squash-up-to create new
history and therefore new proof roots. A base merge regenerates proof state only
when every input base is proof-enabled and validated; a mixed merge remains a
legacy layer.

## Packs

Proof-capable exports carry a root manifest declaring pack version 2 and
`proof-state=pf4`. Import parses bounded tar data into zero-copy private
staging, validates required and succinct native structures, and validates the
entire proof chain before attempting publication. Every requested
proof-enabled layer requires a caller-supplied, out-of-band expected state
root. Same-ID collisions fail rather than overwrite.

Pack limits cover caller-provided layer iterators, compressed output while it
is produced, individual and aggregate decoded entries, copied target ancestry,
one bounded proof reconstruction, and backend publication conversion. The
gzip exporter streams to a bounded temporary file rather than retaining a
growing compressed `Vec` beside mapped native input; gzip/tar import parsing
runs in a charged blocking worker. The directory and directory-archive backends
use durable transaction and visibility markers, cross-process locks, atomic
no-replace publication, and recovery. A batch is logically invisible until the
global commit-marker rename. `Ok` requires the following commit-directory
fsync; failure returns an explicit indeterminate outcome because the marker is
already visible and the operation is not retry-safe. Recovery quarantines
uncommitted targets before deleting them and removes interrupted quarantine and
staging artifacts. Multi-layer subprocess crash tests exercise partial target
publication, marker durability, commit, and crash-during-recovery boundaries;
overlapping two-process batches admit exactly one complete publisher.

`export_layers_legacy` omits both the pack manifest and proof sidecars. This is
the explicit compatibility route for old consumers. Regular `import_layers`
accepts legacy/no-proof packs but rejects proof state without external roots.
`PackImportLimits` makes compressed, entry, aggregate, layer, ancestry, and
aggregate peak-memory budgets caller-configurable. The current implementation
has deterministic budget tests but not an OS-level peak-RSS acceptance result.

The proof cache has hard resident-entry and in-flight decode caps. Its byte cap
uses a conservative logical weight covering capacities, container bodies, Arc
control blocks, and fixed allocator allowances; it is not an OS RSS guarantee.

## Performance status

Run the stable benchmark with:

```text
cargo run --release --example proof_store_bench -- 2000 100 16
```

On the development machine, the compact v4 file reduced proof/native byte
amplification from 59–96x in the earlier duplicated v3 layout to 10.5–13.6x.
The 2026-08-05 release run measured a 17-layer first authenticated scan over
2,016 rows at 31.615 ms with a 49,152-byte sampled RSS delta and 2,479,433
estimated decoded-cache bytes; the repeated scan took 2.970 ms. These figures
are single-run diagnostics, not portability guarantees or accepted service
budgets.

The remaining performance limitation is first-read reconstruction of all
ordinal projections and Merkle levels. The current format is not page-addressed
and therefore is not yet suitable for very large proof layers under a strict
latency/RSS budget. The next storage-format revision should persist paged
ordinal/hash projections referencing one canonical ID/catalog table, so a range
witness reads boundary pages and logarithmic hash paths without restoring the
duplicated v3 payload layout.
