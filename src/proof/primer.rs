//! # Proof-of-WOQL: a primer for developers (not cryptographers)
//!
//! This module contains **no code** — it is the shared conceptual reference for
//! the whole `proof` subsystem. Other modules link here (e.g.
//! `[crate::proof::argument]`, `[crate::proof::commitment]`) instead of
//! re-explaining the cryptography. If a doc comment uses a word like *commitment*,
//! *sumcheck*, *gadget*, or *Fiat–Shamir* and you are not sure what it means,
//! this is the page to read first.
//!
//! You do **not** need a cryptography background to work in this codebase. You do
//! need the mental model below.
//!
//! ---
//!
//! ## 1. What problem this solves
//!
//! TerminusDB answers a read query (a WOQL `Triple`/BGP/aggregate) by scanning its
//! stored graph. Normally you have to **trust the server** that ran the query, or
//! **re-run it yourself** against a full copy of the database.
//!
//! Proof-of-WOQL lets a *prover* (whoever holds the data) produce a short
//! **proof** that a given answer is the correct answer, and lets a *verifier*
//! check it against a single ~32-byte **root** — with **no database, no re-scan,
//! and no trust in the prover**. If the prover lies about even one row, the proof
//! fails to verify (that is *soundness*).
//!
//! Concretely: the store commits to each layer once at write time (producing the
//! root). Later, anyone can prove "running query Q over this committed state yields
//! rows R" and anyone can verify it from `(root, proof_bytes)` alone.
//!
//! This is modelled on **Proof-of-SQL** (commitments + sumcheck over the data
//! columns). It is **not** a zero-knowledge system: the proof reveals the answer
//! rows (that is the point — you want to learn the answer *and* trust it). It also
//! does not dump or re-execute a VM trace; it argues directly about the committed
//! columns.
//!
//! ---
//!
//! ## 2. Everything is field arithmetic
//!
//! All values in a proof live in a single big **prime field**: the BN254 scalar
//! field, represented by [`crate::proof`]'s `ArkFr` (from the `dory-pcs` crate).
//! Think of it as integers modulo a fixed ~254-bit prime `p`: you can add,
//! subtract, and multiply, and everything wraps around mod `p`.
//!
//! Data is turned into field elements first: a triple's subject/predicate/object
//! *dictionary ids* become field elements; a numeric literal becomes a field
//! element via an order-preserving encoding; a string/IRI is hashed into the
//! field. A whole **column** of a table (e.g. every row's object id) becomes a
//! *vector* of field elements.
//!
//! Why a field? Because the checks we want ("this column is a permutation of that
//! one", "every selected row satisfies the filter") can be written as **polynomial
//! identities**, and polynomial identities over a large field can be checked
//! cheaply and soundly by evaluating at a random point (see §5).
//!
//! ---
//!
//! ## 3. Commitments and homomorphism
//!
//! A **commitment** is a short, fixed-size "fingerprint" of a big vector of field
//! elements. Two properties matter:
//!
//! * **Binding** — you cannot find two different vectors with the same commitment,
//!   so once you publish `Com(v)` you are locked into `v`.
//! * **Homomorphic** — commitments respect linear algebra:
//!   `Com(a) + Com(b) = Com(a + b)` and `x · Com(a) = Com(x · a)`.
//!
//! Homomorphism is the superpower: the **verifier can do arithmetic on the
//! committed data without ever seeing it**. It can form `Com(a) − Com(b)`, scale
//! commitments by public challenges, and check that the result is a commitment to
//! zero — all in constant size.
//!
//! In this codebase a commitment is an `ArkGT` (an element of BN254's pairing
//! *target group* `GT`). A column of any length commits to one `ArkGT`.
//!
//! ---
//!
//! ## 4. Dory — the polynomial commitment scheme (PCS)
//!
//! [Dory](https://eprint.iacr.org/2020/1274) is the specific commitment scheme we
//! use, implemented in the `dory-pcs` crate and driven from
//! [`crate::proof::argument`]. What you need to know:
//!
//! * It commits a vector (viewed as a small matrix) of field elements to a single
//!   `ArkGT` using fixed **public generators** — it is *transparent* (no secret
//!   trusted-setup toxic waste; the setup is just public group elements).
//! * It supports **opening / evaluation proofs**: given `Com(v)` and a point `r`,
//!   the prover can convince the verifier that the multilinear polynomial of `v`
//!   evaluates to a claimed value `y` at `r` — in `O(log n)` proof size and
//!   verifier work, not `O(n)`.
//! * The MSMs (multi-scalar multiplications) that build commitments run through a
//!   [`crate::proof::argument::StoreProverContext`], so the same code path can
//!   later be GPU-accelerated (Blitzar) without changing the bytes.
//!
//! We commit **each column** of a layer's tables (subject ids, predicate ids,
//! object ids, and the dictionary's value/datatype columns). Those per-column
//! commitments, hashed together, are the layer's `commitment_root` (§8).
//!
//! ---
//!
//! ## 5. Multilinear extensions and sumcheck — the workhorse
//!
//! Take a column of length `n = 2^k`. Index its entries by the `k`-bit binary
//! strings, i.e. the corners of the boolean hypercube `{0,1}^k`. There is a unique
//! **multilinear polynomial** (degree ≤ 1 in each of `k` variables) that agrees
//! with the column on those corners — its *multilinear extension* (MLE).
//!
//! **Sumcheck** is a protocol that proves a statement of the form
//! "`Σ_{x ∈ {0,1}^k} P(x) = S`" (a sum of some polynomial `P` built from one or
//! more committed columns) in `k` rounds. Each round the prover sends a tiny
//! one-variable polynomial and the verifier replies with a random challenge; after
//! `k` rounds the whole sum claim is reduced to checking **one evaluation**
//! `P(r)` at a random point `r ∈ field^k`. That single evaluation is then
//! discharged by Dory openings of the columns `P` was built from.
//!
//! Almost every relational operation becomes a sumcheck:
//!
//! * *filter* — prove `selector · (predicate − target) = 0` on every row;
//! * *equality / range* — prove a boolean indicator column really is 0/1 and
//!   matches the comparison;
//! * *permutation / join* — prove two multisets of (packed) keys are equal.
//!
//! The `< 2`/power-of-two requirements you see (e.g. "layer has too few rows for a
//! query proof (need >= 2)") come from this: a column must have at least one
//! hypercube variable, and columns are zero-padded up to the next power of two.
//!
//! ---
//!
//! ## 6. Gadgets
//!
//! A **gadget** is a small, reusable proof sub-protocol for exactly one relational
//! or logical operation, expressed as a polynomial identity discharged by sumcheck
//! + Dory openings on a shared transcript. The store's gadget modules include:
//!
//! * [`crate::proof::filter`] / [`crate::proof::range`] — selection predicates;
//! * [`crate::proof::bgp`] / [`crate::proof::join`] / [`crate::proof::join2`] —
//!   basic graph patterns and left-deep join chains;
//! * [`crate::proof::membership`] / [`crate::proof::dict`] — dictionary lookups;
//! * [`crate::proof::pairs`] / [`crate::proof::sort`] / [`crate::proof::order`] —
//!   ordering and all-pairs closures;
//! * [`crate::proof::union`] / [`crate::proof::anti`] /
//!   [`crate::proof::distinct`] — set operations, anti-/outer-joins, dedup;
//! * [`crate::proof::disequality`] — inequality of packed tuples.
//!
//! The compiler (the woql2 `proof` module, in the downstream `terminusdb-woql2`
//! crate) chooses *which* gadgets to run for a query; the gadgets do the
//! cryptography. A gadget never decides query semantics — it proves the mechanical
//! identity it was handed.
//!
//! ---
//!
//! ## 7. LogUp and LogUp-GKR — multiset / lookup arguments
//!
//! Some claims are about **multisets**, not row-by-row arithmetic: "these keys are
//! a sub-multiset of those", "`present` equals `additions` minus `removals`", "this
//! id appears in the committed dictionary". The tool is a **lookup argument**
//! (LogUp): it proves a multiset relation by checking a single **rational-sum
//! identity** `Σ 1/(x + a_i) = Σ m_j/(x + b_j)` at a random challenge `x`. Signed
//! multiplicities let it express set difference (used by delta reconciliation).
//!
//! [`crate::proof::logup_gkr`] batches these sums efficiently through a small **GKR
//! circuit** (a layered arithmetic-circuit argument). You will see LogUp behind
//! joins, dictionary membership, and the `present = ⊎additions − ⊎removals`
//! reconciliation.
//!
//! ---
//!
//! ## 8. Fiat–Shamir and the transcript — and the one invariant that matters
//!
//! Sumcheck and the openings are *interactive* (verifier sends random challenges).
//! To make a non-interactive proof, we use the **Fiat–Shamir transform**: instead
//! of a live verifier, challenges are derived by hashing a running **transcript**
//! of everything public so far (parameters, commitments, claimed evaluations,
//! queried values). See [`crate::proof::argument::Transcript254`].
//!
//! **THE INVARIANT (memorise this):** *every prover-controlled value that a
//! challenge-gated equation depends on must be absorbed into the transcript
//! **before** that challenge is drawn.* If a challenge is drawn before a value it
//! is supposed to bind, a malicious prover can pick that value *after* seeing the
//! challenge and forge a passing proof for a false statement. This class of bug is
//! real and was found here (a batching challenge `γ` drawn before the per-column
//! evaluations it batched). When you add or move any `challenge_scalar(...)` call,
//! check that every value its equation uses is already in the transcript.
//!
//! ---
//!
//! ## 9. The layer-commitment model (the trust anchor)
//!
//! The store is an append-only stack of **layers**, each a *delta*: the triples it
//! adds and removes, plus the dictionary entries it introduces. At write time each
//! layer is committed once ([`crate::proof::commitment`]):
//!
//! * Dory commitments of its columns (subject/predicate/object id columns, plus the
//!   node/value and predicate dictionary columns and the order-preserving
//!   value/datatype columns);
//! * a hash binding all of those into the `commitment_root` (an O(1)
//!   [`crate::proof::ProofHash`]);
//! * two dictionary **Merkle** roots (`nv_merkle_root`, `pred_merkle_root`) that
//!   authenticate id → term resolution.
//!
//! The externally-trusted anchor a verifier is handed is the
//! **`resolution_root = H(commitment_root ‖ nv_merkle_root ‖ pred_merkle_root)`** —
//! it binds *both* the committed columns and the dictionaries used to turn ids back
//! into readable IRIs/literals. (Verifying against the bare `commitment_root` is a
//! mistake — it omits the dictionary binding.) These commitments are persisted with
//! the layer (the `proof_state` sidecar) and travel with pack replication, so a
//! prover never has to recompute them.
//!
//! ---
//!
//! ## 10. Prove → verify, end to end
//!
//! 1. **Compile.** The woql2 `proof` compiler lowers a WOQL query into a
//!    *variable-blind plan* of gadgets. The plan is the safety line: it only
//!    accepts the query fragment the gadgets can prove faithfully and rejects
//!    everything else *closed*.
//! 2. **Prove.** The prover runs the plan over the committed columns (reading the
//!    raw layer data it holds), producing sumcheck transcripts, Dory openings, and
//!    the resolved answer rows.
//! 3. **Verify.** Given only `(resolution_root, proof_bytes)`, the verifier
//!    **re-derives the same plan from its own copy of the query** (never from the
//!    proof — this *anti-substitution* stops a proof for one query masquerading as
//!    another), checks every gadget identity, and confirms the answer resolves
//!    against the root. No database is touched.
//!
//! For a **delta stack**, step 2/3 first **reconcile** the present relation from
//! the layer chain (`present = ⊎additions − ⊎removals`, bound to the trusted head)
//! before running the BGP/aggregate gadgets on top — see
//! [`crate::proof::reconcile`].
//!
//! ---
//!
//! ## 11. Two words that are NOT the same: soundness vs fidelity
//!
//! * **Cryptographic soundness** — you cannot produce a passing proof of a *false*
//!   statement (the polynomial identities force it). This is what the gadgets and
//!   Fiat–Shamir guarantee.
//! * **Semantic fidelity** — the statement the proof proves actually matches what
//!   **TerminusDB's own query engine** would compute (right rows, right ordering,
//!   right datatype comparison semantics). A gadget can be perfectly *sound* about
//!   an identity that is the *wrong* identity for the query.
//!
//! Both are required. When in doubt, a construct is **rejected closed** rather than
//! proved approximately — fail-closed beats wrong-but-verifying. Much of the
//! compiler's caution (and this codebase's audit history) is about fidelity, not
//! soundness.
//!
//! ---
//!
//! ## Glossary
//!
//! * **PCS** — polynomial commitment scheme (here: Dory).
//! * **MLE** — multilinear extension: the unique multilinear polynomial matching a
//!   `2^k`-vector on the hypercube.
//! * **MSM** — multi-scalar multiplication: the group operation that builds a
//!   commitment; the expensive part, GPU-offloadable.
//! * **Transcript** — the running hash of public values that Fiat–Shamir draws
//!   challenges from.
//! * **Gadget** — a reusable per-operation proof sub-protocol.
//! * **Reconcile** — reconstruct the current (`present`) relation from a stack of
//!   add/remove layer deltas without materializing full state.
//! * **Term-less vs term-bearing** — whether a commitment view carries the clear
//!   dictionary terms (`term-bearing`) or only the O(1) commitments and Merkle
//!   roots (`term-less`), resolving terms via prover-supplied, root-checked
//!   openings.
