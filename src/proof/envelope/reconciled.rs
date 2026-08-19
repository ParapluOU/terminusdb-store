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
//! [`resolve_combined_node_id`] resolves a constant IRI (`rdf:nil`, a closure seed) to its
//! combined-dict id over the chain catalogs, failing CLOSED on absence — the chain analogue of the
//! materialized `resolve_node_id_or_absent`.
//!
//! # FOUNDATION-B — the reconciled `TransitiveClosure`
//!
//! [`prove_reconciled_transitive_closure`] mirrors the materialized `prove_transitive_closure`, but
//! the edge relation is a FOUNDATION-A predicate-constant edge read over `present` and the seed is
//! resolved via [`resolve_combined_node_id`]. The round/fixpoint machinery is the SHARED
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
use crate::proof::commitment::{LayerCommitment, VerifierLayer};
use crate::proof::dict::{hash_object, hash_term};
use crate::proof::reconcile::{
    prove_present_bgp, prove_reconcile_core, verify_present_bgp, verify_reconcile_core,
    PresentBgpAtom, PresentBgpProof, PresentReadResult, PresentTerm, ReconcileArtifacts,
    ReconcileProof, ReconcileVerified,
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

/// Verify ONE walk-leaf read (FOUNDATION-A) on the shared `transcript`, returning the authenticated
/// combined-dict-id `RelationCommit` the gluing gadgets consume. Mirror of [`prove_reconciled_leaf`]
/// — the `verify_present_bgp` call re-derives the kept commitments from the atom's own sub-proofs
/// (never trusting a fresh relation commitment), so a tampered leaf relation is rejected here.
pub(crate) fn verify_reconciled_leaf(
    layers: &[&LayerCommitment],
    reconcile: &ReconcileProof,
    verified: &ReconcileVerified,
    head_nv: usize,
    atom: PresentBgpAtom,
    proof: &PresentBgpProof,
    transcript: &mut Transcript254,
) -> io::Result<RelationCommit> {
    let views: Vec<VerifierCommitmentRef<'_>> = layers.iter().map(|l| l.verifier_view()).collect();
    let (commitments, n, log_domain) = verify_present_bgp(
        &views,
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

/// Resolve reconciled transitive-closure output node-ids back to their IRIs
/// ([`CanonicalObject`]s) over the whole chain's root-authenticated catalogs — the exact INVERSE of
/// [`resolve_combined_node_id`], used by ReadDocument 6e (a reconciled `sys:inherits` ancestor
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
        ReconciledClosureProof { reconcile, edge_leaf, seed, rounds, fixpoint },
        reachable_ids,
    ))
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
    let mut transcript = Transcript254::new(RECONCILE_TRANSCRIPT);

    let verified = verify_reconcile_core(layers, trusted_head_root, &proof.reconcile, &mut transcript)?;
    let head_nv = layers
        .last()
        .ok_or_else(|| invalid("reconciled closure needs a head layer"))?
        .verifier_view()
        .state
        .node_value_count as usize;

    let atom = predicate_edge_leaf_atom(edge_predicate);
    let edge_commit = verify_reconciled_leaf(
        layers,
        &proof.reconcile,
        &verified,
        head_nv,
        atom,
        &proof.edge_leaf,
        &mut transcript,
    )?;

    let views: Vec<VerifierCommitmentRef<'_>> = layers.iter().map(|l| l.verifier_view()).collect();
    let seed_id = resolve_combined_node_id(&views, seed_iri);
    let digest = reconciled_closure_digest(edge_predicate, seed_id, reflexive);
    let mut nodes = 0usize;
    // β re-anchor: the closure gadget binds the head's COMMITMENT root (matching
    // `prove_reconciled_transitive_closure`); `trusted_head_root` is now a `resolution_root`
    // used ONLY for the head-pin equality inside `verify_reconcile_core` above.
    let head_commitment_root = layers
        .last()
        .ok_or_else(|| invalid("reconciled closure needs a head layer"))?
        .state
        .commitment_root;
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
