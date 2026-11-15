//! The single-layer TERM-LESS recursive envelope: a [`RecursiveRelationEnvelope`] over one base
//! layer plus the Merkle resolution witness for exactly the ids its verification resolves.
//!
//! # Why
//!
//! The term-BEARING [`VerifierCommitment`] carries the layer's two clear dictionaries, so a proof
//! verified against it ships bytes that grow with the catalog. The TERM-LESS commitment is O(1)
//! (roots + commitments + counts), but dropping the dictionaries means the verifier can no longer
//! rebuild the id-indexed Merkle trees to turn a proven id into its term. This envelope ships the
//! missing piece: one positional [`DictOpening`](crate::proof::commitment::DictOpening) per
//! DISTINCT id the verifier resolves (result ids at every operator node + every matched constant
//! of every BGP leaf), so the proof is O(result), not O(catalog).
//!
//! # Trust
//!
//! The openings are an UNCOMMITTED witness: every one is re-checked by `verify_opening` against the
//! `nv_merkle_root`/`pred_merkle_root` that the caller's TRUSTED `resolution_root` binds (the head
//! is the only layer, so its Merkle roots are bound directly — no ancestor-root gap). A missing
//! opening fails closed at the lookup, a forged one at `verify_opening`, an extra one at the
//! exact-consumption check after the walk, and a duplicate id at decode. The verifier's plan is its
//! own (anti-substitution), and the envelope is accepted only against a term-less, base, validated
//! commitment whose `resolution_root` equals the trusted root.
//!
//! # Scope
//!
//! The trusted root is authentic, but nothing vouches for its WELL-FORMEDNESS, so only operators
//! whose verification reads ids (never dictionary VALUES or clear catalogs) are admitted — see
//! [`require_termless_plan`]. Everything that needs the value/datatype columns or a catalog scan
//! (range/kind filters, order, group/aggregate over values, closures from a constant seed, walks,
//! dictionary absence, constant equality) fails closed here; those statements prove term-less over
//! a proven checkpoint instead (`checkpoint_read`), whose view establishes every well-formedness
//! fact they need ([`require_attested_termless_plan`]).

use super::plan::*;
use super::recursive::RecursiveRelationEnvelope;
use super::wire::*;
use std::convert::TryInto;
use std::io;

use sha2::{Digest, Sha256};

use crate::proof::commitment::{
    LayerCommitment, ResolutionOpenings, TermlessWithOpenings, VerifierCommitment, VerifierLayer,
};
use crate::proof::query::{ObjectKindFilter, require_base_layer_for_query};
use crate::proof::ProofHash;

/// Framing magic for the single-layer term-less recursive envelope — a permanent wire-format
/// constant; never change these bytes (see `MAGIC` in `wire`).
pub(super) const TERMLESS_TREE_MAGIC: &[u8; 12] = b"TDBWQTLTREE\0";
/// Wire-format version of the term-less recursive envelope body (accepted exactly). v2 appends the
/// term-sorted index lookups after the id-indexed openings; v3 drops them again (they served only
/// the retired attested squashed head).
pub(super) const TERMLESS_TREE_VERSION: u32 = 3;

/// The term-less scope over a view whose well-formedness is PROVEN (a checkpoint view, the only
/// view that sets `attested`): every single-layer statement except a two-root
/// cross-root join (not a single-layer statement) and the document list/array walks (their
/// verifiers still read the clear catalogs; they are proved by `ReadDocument`, never by a proven
/// read, and keep the term-bearing form). Depth-bounded like every plan walk.
pub fn require_attested_termless_plan(plan: &RelationProofPlan) -> io::Result<()> {
    fn walk(plan: &RelationProofPlan, depth: usize) -> io::Result<()> {
        if depth > MAX_RELATION_PROOF_DEPTH {
            return Err(invalid("term-less plan exceeds structural limits"));
        }
        match &plan.kind {
            RelationProofPlanKind::Bgp(_) | RelationProofPlanKind::LayerTripleCount => Ok(()),
            RelationProofPlanKind::GeneratedU64Filter { input, .. }
            | RelationProofPlanKind::GeneratedI64Filter { input, .. }
            | RelationProofPlanKind::CollectionCount { input }
            | RelationProofPlanKind::BagProjection { input, .. }
            | RelationProofPlanKind::Distinct { input, .. }
            | RelationProofPlanKind::Equality { input, .. }
            | RelationProofPlanKind::ExactTupleDisequality { input, .. }
            | RelationProofPlanKind::GroupSum { input, .. }
            | RelationProofPlanKind::GroupCount { input, .. }
            | RelationProofPlanKind::CollectionSum { input, .. }
            | RelationProofPlanKind::CollectionExtrema { input, .. }
            | RelationProofPlanKind::GroupExtrema { input, .. }
            | RelationProofPlanKind::ExactScalarOrder { input, .. }
            | RelationProofPlanKind::ExactMultiOrder { input, .. }
            | RelationProofPlanKind::ExactOrderedSlice { input, .. } => walk(input, depth + 1),
            RelationProofPlanKind::BagUnion { branches, .. } => {
                branches.iter().try_for_each(|branch| walk(branch, depth + 1))
            }
            RelationProofPlanKind::LeftAnti { left, right, .. }
            | RelationProofPlanKind::InnerJoin { left, right, .. }
            | RelationProofPlanKind::LeftOuter { left, right, .. } => {
                walk(left, depth + 1)?;
                walk(right, depth + 1)
            }
            RelationProofPlanKind::TransitiveClosure { edge, .. }
            | RelationProofPlanKind::TransitiveClosurePairs { edge, .. } => walk(edge, depth + 1),
            RelationProofPlanKind::CrossRootHashJoin { .. } => Err(invalid(
                "a cross-root join is not a single-layer statement (no term-less form)",
            )),
            RelationProofPlanKind::OrderedListWalk { .. }
            | RelationProofPlanKind::OrderedArrayWalk { .. } => Err(invalid(
                "an ordered list/array walk is not provable over a term-less commitment",
            )),
        }
    }
    walk(plan, 1)
}

/// A single-layer recursive envelope plus its term-less resolution witness.
#[derive(Clone, Debug)]
pub struct TermlessRelationEnvelope {
    /// The operator-tree proof, byte-for-byte the term-bearing one (no leaf carries openings).
    pub envelope: RecursiveRelationEnvelope,
    /// One Merkle opening per distinct id the verifier resolves — no more, no fewer.
    pub openings: ResolutionOpenings,
}

/// Admit only plan kinds whose verification over a TERM-LESS view uses ids alone. The plan is the
/// VERIFIER's own (compiled from its own query), so this is a statement-shape gate, not a parse of
/// prover data. Depth-bounded like every plan walk.
pub fn require_termless_plan(plan: &RelationProofPlan) -> io::Result<()> {
    fn walk(plan: &RelationProofPlan, depth: usize) -> io::Result<()> {
        if depth > MAX_RELATION_PROOF_DEPTH {
            return Err(invalid("term-less plan exceeds structural limits"));
        }
        let reject = |what: &str| -> io::Result<()> {
            Err(invalid(format!(
                "{what} is not provable over a term-less commitment (it reads dictionary values or \
                 catalogs)"
            )))
        };
        match &plan.kind {
            RelationProofPlanKind::Bgp(bgp) => {
                for atom in &bgp.atoms {
                    if atom.object_range.is_some() || atom.object_kind != ObjectKindFilter::Any {
                        return reject("an object range/kind filter");
                    }
                }
                Ok(())
            }
            RelationProofPlanKind::LayerTripleCount => Ok(()),
            RelationProofPlanKind::GeneratedU64Filter { input, .. }
            | RelationProofPlanKind::GeneratedI64Filter { input, .. }
            | RelationProofPlanKind::CollectionCount { input }
            | RelationProofPlanKind::BagProjection { input, .. }
            | RelationProofPlanKind::Distinct { input, .. } => walk(input, depth + 1),
            RelationProofPlanKind::BagUnion { branches, .. } => {
                branches.iter().try_for_each(|branch| walk(branch, depth + 1))
            }
            RelationProofPlanKind::LeftAnti { left, right, .. }
            | RelationProofPlanKind::InnerJoin { left, right, .. }
            | RelationProofPlanKind::LeftOuter { left, right, .. } => {
                walk(left, depth + 1)?;
                walk(right, depth + 1)
            }
            RelationProofPlanKind::Equality { input, constraint } => match constraint {
                RelationEqualityPlan::ColumnConstant { .. } => reject("equality against a constant"),
                RelationEqualityPlan::Columns { .. } | RelationEqualityPlan::Constants { .. } => {
                    walk(input, depth + 1)
                }
            },
            RelationProofPlanKind::ExactTupleDisequality { input, pairs } => {
                let constant = pairs.iter().any(|pair| {
                    let mut nodes = vec![&pair.a0, &pair.a1, &pair.b0, &pair.b1];
                    if let Some((left, right)) = &pair.predicate {
                        nodes.push(left);
                        nodes.push(right);
                    }
                    nodes
                        .into_iter()
                        .any(|node| matches!(node, TupleDisequalityNode::Constant(_)))
                });
                if constant {
                    return reject("tuple disequality against a constant");
                }
                walk(input, depth + 1)
            }
            RelationProofPlanKind::GroupSum { .. } => reject("GroupBy Sum"),
            RelationProofPlanKind::GroupCount { .. } => reject("GroupBy Count"),
            RelationProofPlanKind::CollectionSum { .. } => reject("collection Sum"),
            RelationProofPlanKind::CollectionExtrema { .. } => reject("collection Min/Max"),
            RelationProofPlanKind::GroupExtrema { .. } => reject("GroupBy Min/Max"),
            RelationProofPlanKind::ExactScalarOrder { .. }
            | RelationProofPlanKind::ExactMultiOrder { .. }
            | RelationProofPlanKind::ExactOrderedSlice { .. } => reject("an ordered result"),
            RelationProofPlanKind::CrossRootHashJoin { .. } => reject("a cross-root join"),
            RelationProofPlanKind::TransitiveClosure { .. }
            | RelationProofPlanKind::TransitiveClosurePairs { .. } => reject("a transitive closure"),
            RelationProofPlanKind::OrderedListWalk { .. }
            | RelationProofPlanKind::OrderedArrayWalk { .. } => reject("an ordered list/array walk"),
        }
    }
    walk(plan, 1)
}

/// The commitment gate shared by prover dry run and verifier: TERM-LESS, a BASE layer, internally
/// consistent (`validate`), and anchored to the caller's trusted `resolution_root`. Returns the
/// authenticated `commitment_root` the envelope's transcript is bound to.
fn termless_anchor(
    commitment: &VerifierCommitment,
    trusted_resolution_root: ProofHash,
) -> io::Result<ProofHash> {
    let view = commitment.verifier_view();
    if view.terms_present {
        return Err(invalid(
            "the term-less envelope verifies only against a TERM-LESS verifier commitment",
        ));
    }
    require_base_layer_for_query(view.state)?;
    view.validate()?;
    if view.resolution_root() != trusted_resolution_root {
        return Err(invalid(
            "layer resolution_root does not match the externally-trusted resolution root",
        ));
    }
    Ok(view.state.commitment_root)
}

impl TermlessRelationEnvelope {
    /// PROVER: wrap an envelope (already proven over `layer`, a term-bearing base sidecar) with the
    /// exact term-less witness. Runs the real verifier over `layer`'s TERM-LESS commitment with a
    /// lazy opener over `layer`'s trees, records every id it opens, and ships exactly those
    /// openings — so the shipped witness is complete and minimal by construction (the dry run is
    /// the verifier). `plan` must be the plan the envelope was proven for.
    ///
    pub fn from_envelope(
        envelope: RecursiveRelationEnvelope,
        layer: &LayerCommitment,
        plan: &RelationProofPlan,
    ) -> io::Result<Self> {
        require_termless_plan(plan)?;
        let commitment = layer.verifier_commitment_termless()?;
        let commitment_root = termless_anchor(&commitment, layer.resolution_root())?;
        let oracle = ResolutionOpenings::oracle(layer.verifier_view())?;
        envelope.verify(
            &TermlessWithOpenings { commitment: &commitment, openings: &oracle, attested: false },
            commitment_root,
            plan,
        )?;
        let openings = oracle.consumed_witness()?;
        Ok(Self { envelope, openings })
    }

    /// VERIFIER: check this envelope against a TERM-LESS `commitment`, the caller's trusted
    /// `resolution_root`, and the caller's OWN `plan`. Every result term is resolved through the
    /// shipped openings (each re-verified against the root-bound Merkle roots), and the verifier
    /// must consume every shipped opening. `Ok(())` means the envelope's delivered result is sound.
    ///
    pub fn verify(
        &self,
        commitment: &VerifierCommitment,
        trusted_resolution_root: ProofHash,
        plan: &RelationProofPlan,
    ) -> io::Result<()> {
        require_termless_plan(plan)?;
        let commitment_root = termless_anchor(commitment, trusted_resolution_root)?;
        if self.openings.is_oracle() {
            return Err(invalid("an oracle carrier is never a shipped witness"));
        }
        // A fresh consumption record per verification (a shared decoded value never carries one
        // verification's bookkeeping into another).
        let openings = self.openings.fresh_tracked();
        self.envelope.verify(
            &TermlessWithOpenings { commitment, openings: &openings, attested: false },
            commitment_root,
            plan,
        )?;
        openings.require_exactly_consumed()
    }

    /// Frame as BYTES: `TERMLESS_TREE_MAGIC` ‖ version ‖ (node-bounded envelope body ‖ openings) ‖
    /// SHA-256 checksum. Serialization never changes verification.
    pub fn encode(&self) -> io::Result<Vec<u8>> {
        let mut out = Vec::new();
        out.extend_from_slice(TERMLESS_TREE_MAGIC);
        out.extend_from_slice(&TERMLESS_TREE_VERSION.to_le_bytes());
        let mut nodes = 0;
        self.envelope.put_bounded(&mut out, 1, &mut nodes)?;
        if self.openings.keyed_lookups().next().is_some() {
            return Err(invalid("a term-less envelope carries no checkpoint term-index lookups"));
        }
        put_id_openings(&self.openings, &mut out)?;
        if out.len().checked_add(CHECKSUM_BYTES).map_or(true, |len| len > MAX_QUERY_ENVELOPE_BYTES) {
            return Err(invalid("term-less relation envelope too large"));
        }
        let checksum = Sha256::digest(&out);
        out.extend_from_slice(&checksum);
        Ok(out)
    }

    /// Decode a framed term-less envelope. ONLY parsing (size/item/depth/node budgets, duplicate-id
    /// rejection) — no trust; hand the result to [`Self::verify`].
    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < TERMLESS_TREE_MAGIC.len() + 4 + CHECKSUM_BYTES
            || bytes.len() > MAX_QUERY_ENVELOPE_BYTES
        {
            return Err(invalid("invalid term-less relation envelope size"));
        }
        let (body, checksum) = bytes.split_at(bytes.len() - CHECKSUM_BYTES);
        if Sha256::digest(body).as_slice() != checksum {
            return Err(invalid("term-less relation envelope checksum mismatch"));
        }
        let mut reader = Reader { bytes: body, pos: 0, decoded_items: 0 };
        if reader.take(TERMLESS_TREE_MAGIC.len())? != TERMLESS_TREE_MAGIC {
            return Err(invalid("unknown term-less relation envelope magic"));
        }
        if u32::from_le_bytes(reader.take(4)?.try_into().unwrap()) != TERMLESS_TREE_VERSION {
            return Err(invalid("unsupported term-less relation envelope version"));
        }
        let mut nodes = 0;
        let envelope = RecursiveRelationEnvelope::get_bounded(&mut reader, 1, &mut nodes)?;
        let openings = ResolutionOpenings::get(&mut reader)?;
        reader.finish()?;
        Ok(Self { envelope, openings })
    }
}
