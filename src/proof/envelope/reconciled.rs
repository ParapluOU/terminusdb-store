//! FOUNDATION for reading walk/closure composites over the reconciled `present` (ReadDocument
//! increment 6c-i) — the shared enabler for 6c (list walk), 6d (array walk), and 6e (schema
//! closure) over a DELTA HEAD. No new cryptography: this is orchestration + plumbing over the
//! EXISTING reconcile + `present_bgp` machinery.
//!
//! # Why a parallel surface
//!
//! The materialized walk composites (`list_walk` / `array_walk` / `TransitiveClosure`) read their
//! leaves (the anchor, the `rdf:rest`/`rdf:first`/`sys:index` edge relations, the closure edge)
//! through `ColumnSource::Materialized`, which hard-rejects a non-materialized delta head. A delta
//! head has no single materialized layer — its current triple set is RECONCILED at proof time
//! (`super::super::reconcile`), delivered as authenticated hash columns + combined-dict id columns
//! at `2^K`. So the leaf reads must go through the reconcile + `present_bgp` path, not
//! `ColumnSource`. This module is that PARALLEL leaf provider; `column_source.rs` is untouched, so
//! every materialized proof stays byte-identical.
//!
//! # FOUNDATION-A — the reconciled leaf provider (reconcile-once, many-reads)
//!
//! A walk triggers exactly ONE reconciliation, threaded to every leaf. [`prove_reconciled_leaf`]
//! expresses ONE walk-leaf read — a bound-subject atom `t(S, P, ?o)` (kept `[O]`) or a
//! predicate-CONSTANT edge atom `t(?s, P, ?o)` (kept `[S, O]`) — as a single-atom
//! [`PresentBgpAtom`], proves it via `prove_present_bgp` against the SHARED [`ReconcileArtifacts`]
//! on the SHARED transcript, and returns its result as a combined-dict-id `RelationWitness`
//! (prover) / `RelationCommit` (verifier) — the exact shape the source-agnostic gluing gadgets
//! consume. Because `prove_present_bgp` already takes the artifacts + transcript as parameters
//! (it does NOT reconcile internally), calling it once per leaf on one transcript IS the
//! reconcile-once, many-reads surface — the expensive `O(chain)` reconciliation runs once; each
//! leaf is `O(present)`.
//!
//! `resolve_combined_node_id` resolves a constant IRI (`rdf:nil`, a closure seed) to its
//! combined-dict id over the chain catalogs, failing CLOSED on absence — the chain analogue of the
//! materialized `resolve_node_id_or_absent`.
//!
//! # FOUNDATION-B — the reconciled `TransitiveClosure`
//!
//! [`prove_reconciled_transitive_closure`] mirrors the materialized `prove_transitive_closure`, but
//! the edge relation is a FOUNDATION-A predicate-constant edge read over `present` and the seed is
//! resolved via `resolve_combined_node_id`. The round/fixpoint machinery is the SHARED
//! `prove_closure_rounds` / `verify_closure_rounds` (source-agnostic: it consumes only
//! `edge_witness` + accumulator commitments), so nothing about the semi-naive unroll, the
//! monotonicity checks, or the empty-delta fixpoint changes. This is needed by 6c (`rdf:rest`) AND
//! 6e (`sys:inherits`).
//!
//! # Soundness composition (audited later; the load-bearing seams here)
//!
//! The edge relation E is proven on the RECONCILE transcript (`prove_present_bgp` shares it with
//! `prove_reconcile_core`), so E's committed columns are bound to the trusted head root via the
//! audited reconcile invariant + `present_bgp` completeness. The closure round/fixpoint operators
//! run on FRESH `(root, digest, step, role)`-separated transcripts and consume E's commitment —
//! EXACTLY as the materialized closure consumes a materialized-BGP edge child's commitment. So the
//! binding chain is: reconcile ⇒ present bound to head root; `present_bgp` ⇒ E bound to present;
//! round operators ⇒ reachable set derived from E's commitment. The genuinely-new seams this
//! module must get right (and that get their own adversarial audit): edge-completeness (present is
//! the complete authenticated current set, so `t(?s, P, ?o)` misses no edge), id-space consistency
//! (the edge ids and the resolved seed id live in the SAME combined-dict id space), and transcript
//! composition (one reconcile transcript for the leaves, fresh closure transcripts for the rounds).

use std::io;

use dory_pcs::backends::arkworks::ArkFr;
use dory_pcs::primitives::arithmetic::Field;
use sha2::{Digest, Sha256};

use crate::proof::argument::{StoreProverContext, Transcript254};
use crate::proof::bgp::{RelationCommit, RelationWitness};
use crate::proof::commitment::{
    build_resolution_openings, DictOpening, LayerCommitment, ResolutionOpenings, VerifierLayer,
};
use crate::proof::dict::{hash_object, hash_term};
use crate::proof::reconcile::{
    prove_present_bgp, prove_reconcile_core, resolve_node_value_id_column,
    resolve_termless_node_constant, verify_present_bgp, verify_reconcile_core, PresentBgpAtom,
    PresentBgpProof, PresentReadResult, PresentTerm, ReconcileArtifacts, ReconcileProof,
    ReconcileVerified,
};
use crate::proof::{invalid, CanonicalObject, ProofHash, VerifierCommitmentRef};

use super::recursive::{
    prove_closure_rounds, verify_closure_rounds, ClosureRoundsWitness, TransitiveClosureFixpoint,
    TransitiveClosureRound, TransitiveClosureSeed,
};

/// The reconciliation envelope transcript label — MUST match `prove_reconcile_core` /
/// `verify_reconcile_core` (all the one-shot `prove_read_*` entries use this label too), so the
/// leaf reads share the reconciliation's Fiat-Shamir transcript. `pub(super)` so the sibling
/// reconciled composites (the 6c-ii list walk) open their shared transcript with the IDENTICAL
/// label — the leaves MUST reconcile-once on one transcript for the binding to root to hold.
pub(super) const RECONCILE_TRANSCRIPT: &[u8] = b"parture-delta-reconcile-envelope";

// ===========================================================================
// FOUNDATION-A — the reconciled leaf provider + combined-dict constant resolution.
// ===========================================================================

/// A walk-leaf atom `t(S, P, ?o)`: a bound subject + constant predicate, variable object
/// (kept `[O]`). Delivers the object combined-dict id column — a list/array anchor's head cell.
pub(crate) fn bound_subject_leaf_atom(subject: &CanonicalObject, predicate: &[u8]) -> PresentBgpAtom {
    PresentBgpAtom {
        subject: PresentTerm::Constant(hash_object(subject)),
        predicate_hash: hash_term(predicate),
        object: PresentTerm::Variable,
        subject_object_equal: false,
        object_range: None,
    }
}

/// A walk-EDGE atom `t(?s, P, ?o)`: a constant predicate, variable subject+object (kept `[S, O]`).
/// Delivers the `[subject_id, object_id]` combined-dict id relation — the edge relation a
/// `TransitiveClosure` / functional-edge join consumes. The `present_bgp` predicate-VARIABLE reject
/// does NOT apply here: the predicate is a CONSTANT, only the subject/object are variables.
pub(crate) fn predicate_edge_leaf_atom(predicate: &[u8]) -> PresentBgpAtom {
    PresentBgpAtom {
        subject: PresentTerm::Variable,
        predicate_hash: hash_term(predicate),
        object: PresentTerm::Variable,
        subject_object_equal: false,
        object_range: None,
    }
}

/// Prove ONE walk-leaf read over the SHARED reconciled `present` (FOUNDATION-A). Express `atom` as
/// a single-atom `present_bgp`, prove it against `artifacts` on the shared `transcript` (so every
/// leaf reconciles ONCE), and return its result as a combined-dict-id `RelationWitness` (the exact
/// shape `prove_closure_rounds` / the gluing gadgets consume). Callable N times on one
/// `(artifacts, transcript)` — that IS the reconcile-once, many-reads surface (the O(chain)
/// reconciliation ran once in `prove_reconcile_core`; each leaf here is O(present)).
pub(crate) fn prove_reconciled_leaf(
    context: &StoreProverContext<'_>,
    layers: &[&LayerCommitment],
    artifacts: &ReconcileArtifacts,
    atom: PresentBgpAtom,
    transcript: &mut Transcript254,
) -> io::Result<(PresentBgpProof, RelationWitness)> {
    let (proof, result) = prove_present_bgp(context, layers, artifacts, &[atom], &[], transcript)?;
    let witness = leaf_witness(&proof, &result)?;
    Ok((proof, witness))
}

/// Rebuild the prover-side `RelationWitness` for a single-atom present read: the kept combined-dict
/// id columns re-padded to the working domain the proof's `com_kept` commitments were taken at. The
/// compaction output is dense over `[0, m)` then zero, so re-padding the delivered real prefix
/// reproduces the committed column EXACTLY (the same discipline `RecursiveRelationEnvelope::
/// relation_witness` uses). The commitments come straight from the proof — never re-committed here.
fn leaf_witness(proof: &PresentBgpProof, result: &PresentReadResult) -> io::Result<RelationWitness> {
    let atom = proof
        .atoms
        .first()
        .ok_or_else(|| invalid("reconciled leaf: present BGP has no atom"))?;
    if proof.atoms.len() != 1 {
        return Err(invalid("reconciled leaf expects exactly one present BGP atom"));
    }
    if result.columns.len() != atom.com_kept.len() {
        return Err(invalid("reconciled leaf kept-column arity mismatch"));
    }
    let log_domain = proof.result_log_domain;
    let domain = 1usize << log_domain;
    let columns = result
        .columns
        .iter()
        .map(|col| {
            let mut padded = col.clone();
            padded.resize(domain, ArkFr::zero());
            padded
        })
        .collect::<Vec<_>>();
    Ok(RelationWitness {
        columns,
        commitments: atom.com_kept.clone(),
        log_domain,
        n: result.len,
    })
}

/// Verify ONE walk-leaf read (FOUNDATION-A) FROM verifier VIEWS (C3), returning the authenticated
/// combined-dict-id `RelationCommit` the gluing gadgets consume. Mirror of [`prove_reconciled_leaf`]
/// — the `verify_present_bgp` call re-derives the kept commitments from the atom's own sub-proofs
/// (never trusting a fresh relation commitment), so a tampered leaf relation is rejected here. Split
/// exactly like [`verify_read_bgp_from_views`](crate::proof::reconcile::verify_read_bgp_from_views):
/// it accepts the borrowed [`VerifierCommitmentRef`] slice directly, so a TERM-LESS reconcile view
/// (built via [`LayerCommitment::verifier_commitment_termless`] on the head, term-bearing elsewhere)
/// can be passed for the term-less remote verifier. The leaf atom carries no range
/// (`object_range: None`), so the plain conjunctive `present_bgp` path is term-less-sound (C1/E3);
/// a tampered leaf relation is rejected by the atom's own re-derived kept commitments.
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_reconciled_leaf_from_views(
    views: &[VerifierCommitmentRef<'_>],
    reconcile: &ReconcileProof,
    verified: &ReconcileVerified,
    head_nv: usize,
    atom: PresentBgpAtom,
    proof: &PresentBgpProof,
    transcript: &mut Transcript254,
) -> io::Result<RelationCommit> {
    let (commitments, n, log_domain) = verify_present_bgp(
        views,
        &reconcile.combined_nv,
        &reconcile.combined_pred,
        verified,
        head_nv,
        &[atom],
        &[],
        proof,
        transcript,
    )?;
    Ok(RelationCommit { commitments, log_domain, n })
}

/// Resolve a node IRI to its COMBINED-dict id over the whole chain's root-authenticated catalogs
/// (a term's id is defined in the catalog of the layer whose disjoint watermark range minted it;
/// each id is globally unique, and every catalog is bound to the trusted head root by the
/// reconciliation's per-layer `validate()`). FAILS CLOSED: an absent node maps to a deterministic
/// IMPOSSIBLE id (head `node_value_count + 1`, above every real id) so an unknown constant/seed
/// matches NOTHING downstream — the chain analogue of the materialized `resolve_node_id_or_absent`.
/// Prover and verifier resolve identically over the same authenticated views.
pub(crate) fn resolve_combined_node_id(views: &[VerifierCommitmentRef<'_>], iri: &[u8]) -> u64 {
    let target = CanonicalObject::Node(iri.to_vec());
    for view in views {
        if let Some(record) = view
            .node_value_catalog
            .iter()
            .find(|record| record.triple.object == target)
        {
            return record.object_id;
        }
    }
    views
        .iter()
        .map(|view| view.state.node_value_count)
        .max()
        .unwrap_or(0)
        .saturating_add(1)
}

/// Resolve a node IRI to its COMBINED-dict id (C3), safe over a TERM-LESS reconcile view.
///
/// THE SOUNDNESS RISK term-less: `resolve_combined_node_id` SCANS the clear catalog, which a
/// term-less view drops — the scan reads EMPTY and fail-OPENS to the impossible-id sentinel a
/// prover could steer (a wrong-but-real closure SEED certifies the WRONG closure). So this NEVER
/// scans a term-less chain; it demands a prover-shipped [`DictOpening`] carrying the claimed id and
/// BELT-HASHES it (`resolve_termless_node_constant`, the reconciled analogue of the S4
/// `belt_hash_node_value_constant`): the id is routed to its introducing view and opened against
/// that view's root-bound `nv_merkle_root`, then the opened term MUST hash to
/// `hash_object(Node(iri))`. A wrong id / substituted term / MISSING opening fails closed.
///
/// * ALL views term-bearing → delegate to the audited `resolve_combined_node_id` scan (each
///   catalog is root-bound by the reconciliation's per-layer `validate()`) — backward compatible.
/// * ANY view term-less → require `opening` (fail-closed on `None`), belt-hash it, return its
///   claimed id. A constant genuinely ABSENT from the dict cannot produce an authenticating opening
///   (no such term exists), so the term-less path fails closed on an absent constant — a stricter
///   (sound) restriction than the scan's impossible-id sentinel; term-less absence is out of scope
///   (it would need `dictionary_absence`, not a positive opening).
pub(crate) fn resolve_combined_node_id_termless(
    views: &[VerifierCommitmentRef<'_>],
    iri: &[u8],
    opening: Option<&DictOpening>,
) -> io::Result<u64> {
    if views.iter().all(|view| view.terms_present) {
        return Ok(resolve_combined_node_id(views, iri));
    }
    let opening = opening.ok_or_else(|| {
        invalid("term-less reconciled constant resolution requires a prover-shipped Merkle opening")
    })?;
    let claimed_id = opening.id;
    resolve_termless_node_constant(
        views,
        claimed_id,
        &CanonicalObject::Node(iri.to_vec()),
        opening,
    )?;
    Ok(claimed_id)
}

/// Resolve reconciled transitive-closure output node-ids back to their IRIs
/// ([`CanonicalObject`]s) over the whole chain's root-authenticated catalogs — the exact INVERSE of
/// `resolve_combined_node_id`, used by ReadDocument 6e (a reconciled `sys:inherits` ancestor
/// closure over a DELTA schema head delivers combined-dict ids, which the compiler must render as
/// class IRIs to drive the per-ancestor frame reads).
///
/// SOUNDNESS: each combined-dict id is globally unique and minted in exactly one layer's catalog;
/// every catalog is bound to the trusted head root by the reconciliation's per-layer `validate()`
/// (which the closure verify that produced `ids` has already run against the same `layers`), so the
/// `id -> object` lookup is root-authenticated. FAILS CLOSED on an id absent from every catalog (an
/// unknown / forged id resolves to nothing → error), so a bogus closure output cannot silently map to
/// a real IRI. Prover and verifier resolve identically over the same authenticated views.
pub fn resolve_reconciled_node_ids(
    layers: &[&LayerCommitment],
    ids: &[u64],
) -> io::Result<Vec<CanonicalObject>> {
    let views: Vec<VerifierCommitmentRef<'_>> = layers.iter().map(|l| l.verifier_view()).collect();
    ids.iter()
        .map(|&id| {
            views
                .iter()
                .find_map(|view| {
                    view.node_value_catalog
                        .iter()
                        .find(|record| record.object_id == id)
                        .map(|record| record.triple.object.clone())
                })
                .ok_or_else(|| invalid("reconciled closure output id has no chain catalog entry"))
        })
        .collect()
}

/// Resolve reconciled closure output node-ids to their `CanonicalObject`s FROM verifier VIEWS (C3),
/// safe over a TERM-LESS head. Views-based, term-less-capable analogue of
/// [`resolve_reconciled_node_ids`] (whose catalog scan fail-closes on a term-less head). The
/// reachable `ids` are ALREADY authenticated by the closure verify, so this is a pure id→term
/// render: it binds them to their own accumulator-shaped commitment (`closure_set_commit`, exactly
/// the padding the closure threaded) and routes each id through the audited chain Merkle resolver
/// (`resolve_node_value_id_column`). A TERM-LESS routed view opens from the prover-shipped
/// `openings` (fail-closed on a missing/forged one, each re-verified by `verify_opening` against the
/// root-bound `nv_merkle_root`); a term-bearing chain passes `None` and rebuilds from catalogs.
pub fn resolve_reconciled_node_ids_from_views(
    views: &[VerifierCommitmentRef<'_>],
    ids: &[u64],
    openings: Option<&ResolutionOpenings>,
) -> io::Result<Vec<CanonicalObject>> {
    let commit = super::recursive::closure_set_commit(ids)?;
    // No fully-term-less RENDER belt here (closure ids are already authenticated by the closure
    // verify; a non-head term-less id stays fail-closed).
    resolve_node_value_id_column(views, ids, commit.commitments[0], commit.log_domain, openings, None)
}

// ===========================================================================
// FOUNDATION-B — the reconciled `TransitiveClosure`.
// ===========================================================================

/// A reconciled bounded transitive-closure proof: the single reconciliation, the FOUNDATION-A
/// predicate-constant edge leaf over `present`, and the SHARED seed + rounds + fixpoint (the exact
/// materialized round machinery, driven over the reconciled edge). Reachable-set delivery is the
/// resolved node-id set the verifier reconstructs from the round row sets.
#[derive(Clone, Debug)]
pub struct ReconciledClosureProof {
    /// Proves `present` IS the current triple set of the whole layer stack (bound to the head root).
    pub reconcile: ReconcileProof,
    /// FOUNDATION-A: the `t(?s, edge_predicate, ?o)` edge relation over `present` (combined-dict ids).
    pub edge_leaf: PresentBgpProof,
    /// `A_0`: `Reflexive` (star `{a}`) or the `Successor` one-hop neighbourhood (plus).
    pub seed: TransitiveClosureSeed,
    /// The accumulate-and-dedupe rounds threaded by commitment over the reconciled edge.
    pub rounds: Vec<TransitiveClosureRound>,
    /// The empty-delta fixpoint certificate (`A_K` is closed under the reconciled edge step).
    pub fixpoint: TransitiveClosureFixpoint,
    /// C3c: the prover-shipped, UNCOMMITTED Merkle opening for the closure SEED constant, used to
    /// resolve+belt-hash the seed id over a TERM-LESS head (`resolve_combined_node_id_termless`).
    /// `None` on the ordinary term-bearing path (the seed is resolved by the audited catalog scan);
    /// minted by [`attach_termless_closure_openings`]. IN-MEMORY ONLY — like
    /// [`PresentBgpProof::openings`], it is NEVER folded into `commitment_root`/`resolution_root`
    /// and carries NO wire/golden/version change (reconciled walk proofs are not in `wire.rs`).
    pub seed_opening: Option<DictOpening>,
    /// C3c: the prover-shipped, UNCOMMITTED Merkle openings for the REACHABLE ids, used to render
    /// them to IRIs over a TERM-LESS head (`resolve_reconciled_node_ids_from_views`). `None` on the
    /// term-bearing path. Same in-memory-only discipline as `seed_opening`.
    pub reachable_openings: Option<ResolutionOpenings>,
}

/// The domain-separation digest for a reconciled closure's round/fixpoint operator transcripts.
/// Deterministic from `(edge predicate, resolved seed id, reflexive)` so prover and verifier build
/// the IDENTICAL `closure_transcript(root, digest, step, role)` sequence. (The materialized closure
/// uses `plan.query_digest`; the reconciled closure has no plan node yet — 6c-ii/6e add one — so it
/// derives an equivalent tag here.)
fn reconciled_closure_digest(predicate: &[u8], seed_id: u64, reflexive: bool) -> ProofHash {
    let mut hasher = Sha256::new();
    hasher.update(b"pow/reconciled-transitive-closure/digest/v0");
    hasher.update((predicate.len() as u64).to_le_bytes());
    hasher.update(predicate);
    hasher.update(seed_id.to_le_bytes());
    hasher.update([u8::from(reflexive)]);
    let mut out = [0u8; 32];
    out.copy_from_slice(&hasher.finalize());
    out
}

/// Prove a bounded (reflexive-or-plus) transitive closure of `edge_predicate` seeded at `seed_iri`
/// over the reconciled `present` of the layer stack `layers` (ordered base..head, `len >= 2`).
/// Returns the proof and the reachable combined-dict node-id set (in delivered order). The walk
/// reconciles ONCE (the edge leaf shares the reconciliation transcript); the round/fixpoint
/// machinery is the SHARED source-agnostic `prove_closure_rounds`.
pub fn prove_reconciled_transitive_closure(
    layers: &[&LayerCommitment],
    edge_predicate: &[u8],
    seed_iri: &[u8],
    reflexive: bool,
) -> io::Result<(ReconciledClosureProof, Vec<u64>)> {
    let context = StoreProverContext::cpu();
    let mut transcript = Transcript254::new(RECONCILE_TRANSCRIPT);

    // Reconcile ONCE — the O(chain) work; threaded (by `artifacts`) to every leaf below.
    let (reconcile, artifacts) = prove_reconcile_core(&context, layers, &mut transcript)?;

    // FOUNDATION-A: the predicate-constant edge relation E = `t(?s, P, ?o)` over `present`, proven
    // on the SHARED reconciliation transcript (so E's committed columns are bound to the head root).
    let atom = predicate_edge_leaf_atom(edge_predicate);
    let (edge_leaf, edge_witness) =
        prove_reconciled_leaf(&context, layers, &artifacts, atom, &mut transcript)?;

    // Seed resolution over the chain catalogs (fail-closed on absence).
    let views: Vec<VerifierCommitmentRef<'_>> = layers.iter().map(|l| l.verifier_view()).collect();
    let seed_id = resolve_combined_node_id(&views, seed_iri);

    // Rounds over FRESH `(root, digest)`-separated transcripts (unchanged machinery).
    let root = layers
        .last()
        .ok_or_else(|| invalid("reconciled closure needs a head layer"))?
        .state
        .commitment_root;
    let digest = reconciled_closure_digest(edge_predicate, seed_id, reflexive);
    let mut nodes = 0usize;
    let ClosureRoundsWitness { seed, rounds, fixpoint, reachable_ids, .. } = prove_closure_rounds(
        &context,
        &edge_witness,
        seed_id,
        root,
        digest,
        reflexive,
        false,
        &mut nodes,
    )?;

    Ok((
        ReconciledClosureProof {
            reconcile,
            edge_leaf,
            seed,
            rounds,
            fixpoint,
            // Term-bearing default: the seed is resolved by the catalog scan and reachable ids by a
            // catalog render. The term-less producer attaches these via
            // `attach_termless_closure_openings`; a term-bearing verifier ignores them.
            seed_opening: None,
            reachable_openings: None,
        },
        reachable_ids,
    ))
}

/// C3c (producer): attach the UNCOMMITTED term-less opening witnesses to a proven
/// [`ReconciledClosureProof`], so it can ship alongside a TERM-LESS reconciled artifact. Mints the
/// SEED [`DictOpening`] (routed to its introducing layer via `build_resolution_openings` for the
/// single resolved seed id) plus the REACHABLE-id [`ResolutionOpenings`] (for the output render).
/// `reachable_ids` are the closure's proven reachable set (the second return of
/// [`prove_reconciled_transitive_closure`]). Mirrors
/// [`attach_termless_reconcile_openings`](crate::proof::reconcile::attach_termless_reconcile_openings)
/// / [`build_termless_delta_openings`](crate::proof::reconcile::build_termless_delta_openings). The
/// prover still holds the term-bearing catalogs, so it resolves the seed by the scan and rebuilds
/// each id-indexed tree; the witnesses are re-verified per id at the verifier by `verify_opening`
/// against the root-bound `nv_merkle_root` (forged ⇒ rejected, missing ⇒ fail-closed) and are NEVER
/// folded into `commitment_root`/`resolution_root`. Over-inclusion is safe; the seed constant must
/// be present in the dict (term-less absence is out of scope — a positive opening cannot be minted).
pub fn attach_termless_closure_openings(
    proof: &mut ReconciledClosureProof,
    layers: &[&LayerCommitment],
    seed_iri: &[u8],
    reachable_ids: &[u64],
) -> io::Result<()> {
    let views: Vec<VerifierCommitmentRef<'_>> = layers.iter().map(|l| l.verifier_view()).collect();
    // The prover holds term-bearing catalogs: resolve the seed by the scan, then open it.
    let seed_id = resolve_combined_node_id(&views, seed_iri);
    let seed_openings = build_resolution_openings(&views, &[seed_id], &[])?;
    proof.seed_opening = Some(seed_openings.node_value_opening(seed_id)?.clone());
    proof.reachable_openings = Some(build_resolution_openings(&views, reachable_ids, &[])?);
    Ok(())
}

/// Verify a reconciled transitive closure and return the reachable combined-dict node-id set. Binds
/// `present` to `trusted_head_root` (the reconciliation anchor), re-verifies the FOUNDATION-A edge
/// leaf, re-resolves the seed, then replays the SHARED seed + rounds + fixpoint over the
/// authenticated edge commitment. A tampered edge leaf, a forged round, or a truncated (non-
/// converged) closure each fails closed — exactly as in the materialized closure.
pub fn verify_reconciled_transitive_closure(
    layers: &[&LayerCommitment],
    trusted_head_root: ProofHash,
    edge_predicate: &[u8],
    seed_iri: &[u8],
    reflexive: bool,
    proof: &ReconciledClosureProof,
) -> io::Result<Vec<u64>> {
    let views: Vec<VerifierCommitmentRef<'_>> = layers.iter().map(|l| l.verifier_view()).collect();
    verify_reconciled_transitive_closure_from_views(
        &views,
        trusted_head_root,
        edge_predicate,
        seed_iri,
        reflexive,
        proof,
    )
}

/// Verify a reconciled transitive closure FROM verifier VIEWS (C3c) — the term-less-capable core.
/// Views-based analogue of [`verify_reconciled_transitive_closure`] (split like
/// [`verify_read_bgp_from_views`](crate::proof::reconcile::verify_read_bgp_from_views)): a TERM-LESS
/// head view (`LayerCommitment::verifier_commitment_termless`, term-bearing elsewhere) can be
/// passed. The round/fixpoint machinery (`verify_closure_rounds`) is commitment-only, so it is
/// term-less-safe unchanged; the two term-sensitive reads go term-less:
/// `verify_reconciled_leaf_from_views` (plain conjunctive edge atom, no range ⇒ C1/E3-sound) and
/// the SEED resolution via `resolve_combined_node_id_termless` (belt-hashed against
/// `proof.seed_opening`, fail-closed on a missing/forged opening — the seed constant must be present
/// and CORRECT, or the wrong closure would be certified).
pub fn verify_reconciled_transitive_closure_from_views(
    views: &[VerifierCommitmentRef<'_>],
    trusted_head_root: ProofHash,
    edge_predicate: &[u8],
    seed_iri: &[u8],
    reflexive: bool,
    proof: &ReconciledClosureProof,
) -> io::Result<Vec<u64>> {
    let mut transcript = Transcript254::new(RECONCILE_TRANSCRIPT);

    let verified =
        verify_reconcile_core(views, trusted_head_root, &proof.reconcile, &mut transcript)?;
    let head = *views
        .last()
        .ok_or_else(|| invalid("reconciled closure needs a head layer"))?;
    let head_nv = head.state.node_value_count as usize;

    let atom = predicate_edge_leaf_atom(edge_predicate);
    let edge_commit = verify_reconciled_leaf_from_views(
        views,
        &proof.reconcile,
        &verified,
        head_nv,
        atom,
        &proof.edge_leaf,
        &mut transcript,
    )?;

    // C3c: term-less-safe seed resolution — belt-hashed against the prover-shipped opening on a
    // term-less chain, the audited catalog scan on a fully term-bearing one.
    let seed_id = resolve_combined_node_id_termless(views, seed_iri, proof.seed_opening.as_ref())?;
    let digest = reconciled_closure_digest(edge_predicate, seed_id, reflexive);
    let mut nodes = 0usize;
    // β re-anchor: the closure gadget binds the head's COMMITMENT root (matching
    // `prove_reconciled_transitive_closure`); `trusted_head_root` is now a `resolution_root`
    // used ONLY for the head-pin equality inside `verify_reconcile_core` above.
    let head_commitment_root = head.state.commitment_root;
    verify_closure_rounds(
        &edge_commit,
        seed_id,
        head_commitment_root,
        digest,
        reflexive,
        false,
        &proof.seed,
        &proof.rounds,
        &proof.fixpoint,
        &mut nodes,
    )
}

/// C4-slice-3 byte-in remote-verifier entry: verify a reconciled transitive-closure proof shipped
/// as BYTES against a trusted head root, holding ONLY the encoded proof + the encoded term-less
/// [`VerifierCommitment`](crate::proof::commitment::VerifierCommitment) chain (no `LayerCommitment`
/// sidecars, no DB). Decodes each layer commitment and the closure proof, then re-verifies through
/// the audited [`verify_reconciled_transitive_closure_from_views`] — serialization does NOT change
/// verification. `edge_predicate`/`seed_iri`/`reflexive`/`trusted_head_root` are the VERIFIER's OWN
/// query parameters (anti-substitution): they are NEVER parsed from the proof bytes.
pub fn decode_and_verify_reconciled_transitive_closure(
    closure_proof_bytes: &[u8],
    verifier_commitment_bytes: &[&[u8]],
    trusted_head_root: ProofHash,
    edge_predicate: &[u8],
    seed_iri: &[u8],
    reflexive: bool,
) -> io::Result<Vec<u64>> {
    // Own the decoded commitments FIRST, so the borrowed views below cannot dangle.
    let owned: Vec<crate::proof::commitment::VerifierCommitment> = verifier_commitment_bytes
        .iter()
        .map(|b| crate::proof::commitment::VerifierCommitment::decode(b))
        .collect::<io::Result<_>>()?;
    let views: Vec<VerifierCommitmentRef<'_>> = owned.iter().map(|c| c.verifier_view()).collect();
    let proof = ReconciledClosureProof::decode(closure_proof_bytes)?;
    verify_reconciled_transitive_closure_from_views(
        &views,
        trusted_head_root,
        edge_predicate,
        seed_iri,
        reflexive,
        &proof,
    )
}
