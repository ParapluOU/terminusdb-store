//! General n-atom conjunctive BGP over the reconciled `present` relation.
//!
//! The full basic-graph-pattern read: a left-deep chain of joins over several `present`
//! atoms, each atom a constant-predicate filter (optionally with a numeric value-range test
//! against the combined range dictionary), delivering resolved result columns. Every atom's
//! source is the one authenticated `present` relation, and every sub-proof shares the
//! reconciliation transcript, so the whole BGP answer is bound to the trusted head root. Also
//! holds the duplicate-freeness assertions (bounded canonical-triple key + combined-catalog
//! hash uniqueness) that make `present` a genuine set.

use super::*;
use crate::proof::commitment::build_resolution_openings;
use crate::proof::lexdistinct::{
    prove_lex_strictly_sorted_with_context, verify_lex_strictly_sorted, LexDistinctProof,
};

// ===========================================================================
// Wave 6 — GENERAL n-atom conjunctive BGP over the reconciled `present` relation.
//
// This generalizes the single-atom filter and 2-atom join above to the SAME read-only
// BGP fragment the base-layer path (`crate::proof::query::prove_layer_bgp`) supports over a
// materialized layer: an arbitrary left-deep conjunctive chain of k atoms, constants in
// any position, and repeated-variable same-row equality — now over a DELTA STACK, with
// every input column authenticated to the trusted head root by the reconciliation above.
//
// It is built by REUSE, not new crypto. `present` is authenticated (by the reconciliation)
// as three canonical HASH columns `[hash_object(S), hash_term(P), hash_object(O)]` at 2^K,
// plus (via [`prove_id_auth`]) its subject/object COMBINED-DICT id columns pinned to those
// hashes. Each atom is then the SAME audited composition the base RANGED atom uses:
//   1. an authenticated EQUALITY SELECTION over `present`'s root-bound HASH columns
//      ([`crate::proof::filter::prove_selection_only`]) — predicate/subject/object CONSTANTS are
//      public canonical hashes compared against the authenticated hash columns, and a
//      repeated variable is a same-row hash equality; this pins `sel` to EXACTLY the
//      matching rows (complete AND sound), and an absent constant simply selects nothing;
//   2. a COMPACTION ([`crate::proof::filter::prove_compaction_only`]) of the atom's KEPT variable
//      columns — the reconciliation-authenticated combined-dict id columns (`present_id_s`
//      / `present_id_o`, each pinned row-for-row to the selected term by [`verify_id_auth`])
//      — by that same `sel`, yielding the atom's filtered relation as small joinable ids.
// The k filtered relations feed the audited left-deep join chain ([`crate::proof::bgp`]) verbatim,
// so >= 3-atom joins on shared node variables work by reuse. Every result column is a
// combined-dict node/value id, resolved to canonical IRIs/typed literals by the chain
// catalog resolver ([`resolve_result_columns`]) exactly as the 2-atom join is.
//
// NUMERIC RANGE over `present`: an atom may carry an object-value RANGE constraint
// (`Triple(?s, p, ?v) [?v op c]`, single- or paired-bound / InRange). It reuses the audited
// object-range gadget ([`crate::proof::range::prove_object_range_select`]) verbatim, authenticating
// each `present` object id's ORDER-PRESERVING VALUE + DATATYPE against a COMBINED
// `(id, value, datatype)` range dictionary bag-unioned from every layer's incremental
// `node_value_dict_range_commitments` (all root-bound). The range selection is ANDed with the
// atom's equality selection (a boolean product) before compaction — the SAME composition the
// base ranged atom uses — so `present`'s object id column (already at the working domain)
// needs no relocation. Supported datatypes are exactly the base range's aligned fixed-width
// numerics (Boolean/UInt32/Int64/Float64, `b_bits = 64`); string/decimal/date range is out
// (same limits as the base path).
//
// SCOPE (this pass): the predicate of each atom is a CONSTANT IRI (its hash); every kept
// output/join column is therefore a node/value id (uniform, joinable, resolvable). Each
// atom keeps >= 1 variable. Predicate VARIABLES and fully-ground atoms reject-closed
// (a later pass authenticates `present`'s predicate ids the same way as its node ids).
// ===========================================================================

/// One subject/object position of a `present` BGP atom: a kept query VARIABLE (an output
/// combined-dict id) or a CONSTANT given as its public canonical `hash_object(...)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresentTerm {
    Variable,
    /// The public `hash_object(canonical object)` of the constant (a node IRI in the
    /// subject or object position, or a typed literal in the object position).
    Constant(ArkFr),
}

/// One atom of a `present` BGP: a CONSTANT predicate (its public `hash_term` IRI hash),
/// subject and object each a [`PresentTerm`], an optional same-row subject==object
/// equality (both positions must then be variables), and an optional numeric object-value
/// RANGE (the object must then be a variable, and `subject==object` must be false). Mirrors
/// the base [`crate::proof::query::LayerBgpAtom`] restricted to this pass's supported shape.
#[derive(Clone, Debug)]
pub struct PresentBgpAtom {
    pub subject: PresentTerm,
    /// `hash_term(predicate IRI)` — the predicate is a constant this pass.
    pub predicate_hash: ArkFr,
    pub object: PresentTerm,
    pub subject_object_equal: bool,
    /// A numeric range constraint on the (variable) object, or `None` for a plain atom.
    pub object_range: Option<ObjectRangeSpec>,
}

/// The extra authenticated selection + table-domain intersection for an atom's SECOND
/// range bound (an `InRange` / half-open interval). Mirrors the base `SecondRangeSubproof`.
#[derive(Clone, Debug)]
pub struct PresentSecondRange {
    pub object_range: ObjectRangeProof,
    pub com_range_sel: ArkGT,
    pub intersect: BooleanProductProof,
    pub com_intersection_sel: ArkGT,
}

/// The DATATYPE-HOMOGENEITY guard for a `present` ranged atom (fail-closed fidelity). Same as
/// the base path's `RangeDatatypeGuardProof`, minus the relocation: `present`'s object column
/// and equality selection already share the working domain, so `(1 − dt_match)` feeds the
/// `equality_sel ⊙ (1 − dt_match) == 0` product directly. A datatype-heterogeneous compared
/// column reject-closes (we refuse rather than under-report TerminusDB's cross-datatype
/// comparison); homogeneous data is unaffected.
#[derive(Clone, Debug)]
pub struct PresentRangeDatatypeGuardProof {
    /// `equality_sel ⊙ (1 − dt_match)`; verification asserts its commitment is `Com(0)`.
    pub bad: BooleanProductProof,
    pub com_bad: ArkGT,
}

/// A ranged atom's sub-proofs over `present`: the authenticated object-value range selection
/// (optionally intersected with a second bound), the constant-equality selection, their
/// boolean product, and the compaction of the kept id columns by that combined selection.
#[derive(Clone, Debug)]
pub struct PresentRangeSubproof {
    pub object_range: ObjectRangeProof,
    pub com_range_sel: ArkGT,
    pub second: Option<PresentSecondRange>,
    pub equality: SelectionOnlyProof,
    pub com_equality_sel: ArkGT,
    /// DATATYPE-HOMOGENEITY guard (fail-closed fidelity). `present`'s object id column and
    /// equality selection share the working domain (no relocation): asserts
    /// `Com(equality_sel ⊙ (1 − dt_match)) == Com(0)` over the object-range gadget's committed
    /// `dt_match` — every predicate-matched present row carries the pinned datatype, else the
    /// heterogeneous column is reject-closed (we refuse rather than under-report).
    pub datatype_guard: PresentRangeDatatypeGuardProof,
    pub combine: BooleanProductProof,
    pub com_combined_sel: ArkGT,
    pub compaction: CompactionOnlyProof,
}

/// The per-atom filter: a PLAIN equality atom (authenticated selection over `present`'s hash
/// columns + compaction) or a RANGED object atom (equality AND an authenticated object-value
/// range). Which one is used is fixed by the PUBLIC atom (`object_range`), not the proof.
#[derive(Clone, Debug)]
pub enum PresentAtomFilter {
    Plain {
        selection: SelectionOnlyProof,
        com_sel: ArkGT,
        compaction: CompactionOnlyProof,
    },
    Range(PresentRangeSubproof),
}

/// The per-atom half of a `present` BGP proof: the atom's filter (plain or ranged) plus the
/// compacted kept-column commitments (S,O order; predicate is constant) and result length.
#[derive(Clone, Debug)]
pub struct PresentBgpAtomProof {
    pub filter: PresentAtomFilter,
    /// One commitment per kept (variable) column, in S,O order (predicate is constant).
    pub com_kept: Vec<ArkGT>,
    pub m: usize,
}

/// A general n-atom BGP proof over `present`: the two present-id authentications shared by
/// every atom, the combined `(id, value, datatype)` range dictionary (present iff some atom
/// carries a range), one per-atom (plain/ranged) proof, and the left-deep join chain over
/// the filtered relations when `k >= 2`. All under the reconciliation transcript.
#[derive(Clone, Debug)]
pub struct PresentBgpProof {
    pub auth_s: PresentIdAuth,
    pub auth_o: PresentIdAuth,
    pub auth_p: PresentIdAuth,
    /// Wave 7 hardening: present is a SET (no duplicate canonical triple), over the bounded id
    /// key. Sound in TERM space GIVEN the combined-dict hash-uniqueness the verifier asserts
    /// (term ↔ id is a bijection). `None` only when present has `< 2` rows (trivially a set).
    pub present_distinct: Option<ColumnDistinctProof>,
    pub range_dict: Option<CombinedRangeDictProof>,
    pub atoms: Vec<PresentBgpAtomProof>,
    pub chain: Option<BgpChainProof>,
    pub result_len: usize,
    pub result_log_domain: usize,
    pub result_arity: usize,
    /// C1b/E3: UNCOMMITTED prover-shipped Merkle resolution witness for a TERM-LESS reconciled
    /// (materialized-present) BGP artifact. One [`DictOpening`] per DISTINCT resolved result id,
    /// re-verified per id against the root-bound `nv_merkle_root` at resolve time. `None` on the
    /// prover default and on term-bearing artifacts (whose resolver rebuilds from the clear
    /// catalogs); the term-less producer attaches it via [`attach_termless_reconcile_openings`].
    /// NEVER folded into `commitment_root`/`resolution_root` — bound solely by `verify_opening`,
    /// so a forged opening is rejected and a missing one fails closed. C4: this witness IS now
    /// wire-serialized as part of `DeltaReadProof::encode` (it rides the `Option<ResolutionOpenings>`
    /// `Wire` impl), so it survives the byte boundary; it stays an UNCOMMITTED witness, re-verified
    /// per id at resolve time. Mirrors [`DeltaReadProof::openings`].
    pub openings: Option<ResolutionOpenings>,
    /// C2a: UNCOMMITTED prover-shipped REAL present object-id column (the same `present_id_o`
    /// prefix `auth_o.com_id` commits), used by `verify_dict_value_terms_chain` to recommit the
    /// touched object column and pin it to `auth_o.com_id` when a numeric object-value RANGE is
    /// verified over a TERM-LESS delta / reconciled view. Empty on the prover default and on
    /// term-bearing artifacts; the term-less producer attaches it via
    /// [`attach_termless_present_range_openings`]. C4: wire-serialized (rides the `Vec<u64>` `Wire`
    /// impl in `DeltaReadProof::encode`), like `openings`; still an uncommitted, resolve-time witness.
    pub range_object_oid: Vec<u64>,
    /// C2a: UNCOMMITTED prover-shipped Merkle value↔term openings for a TERM-LESS object-value
    /// RANGE — one [`crate::proof::commitment::DictOpening`] per DISTINCT present object id,
    /// re-verified per id against the root-bound `nv_merkle_root` by the chain resolver at verify
    /// time. `None` on the prover default and on term-bearing artifacts (whose committed
    /// value/datatype columns are tied by `validate()`'s catalog re-MSM); a term-less range with
    /// this `None` fails closed (the D2 default). NEVER folded into `commitment_root` /
    /// `resolution_root`. C4: wire-serialized (rides the `Option<ResolutionOpenings>` `Wire` impl in
    /// `DeltaReadProof::encode`); still an uncommitted, resolve-time witness. Mirrors `openings`.
    pub range_value_openings: Option<ResolutionOpenings>,
}

/// Build the hash-space [`AtomFilterSpec`] (predicate/subject/object constants +
/// subject==object equality) and the KEPT variable positions (S then O; predicate is a
/// constant so it is never kept) for a `present` atom. Rejects the unsupported shapes.
pub(super) fn present_atom_spec(atom: &PresentBgpAtom) -> io::Result<(AtomFilterSpec, Vec<Position>)> {
    if let Some(range) = &atom.object_range {
        if !matches!(atom.object, PresentTerm::Variable) {
            return Err(invalid("present BGP range: the ranged object must be a variable"));
        }
        if atom.subject_object_equal {
            return Err(invalid("present BGP range: subject==object with an object range is unsupported"));
        }
        if range.b_bits != OBJECT_RANGE_B_BITS {
            return Err(invalid(
                "present BGP range: b_bits must be 64 (order-preserving dict values are < 2^64)",
            ));
        }
    }
    let mut constants = vec![(Position::P, atom.predicate_hash)];
    if let PresentTerm::Constant(h) = atom.subject {
        constants.push((Position::S, h));
    }
    if let PresentTerm::Constant(h) = atom.object {
        constants.push((Position::O, h));
    }
    let mut kept = Vec::new();
    if matches!(atom.subject, PresentTerm::Variable) {
        kept.push(Position::S);
    }
    if matches!(atom.object, PresentTerm::Variable) {
        kept.push(Position::O);
    }
    let equalities = if atom.subject_object_equal {
        if !matches!(atom.subject, PresentTerm::Variable) || !matches!(atom.object, PresentTerm::Variable) {
            return Err(invalid("present BGP: subject==object equality requires two variable positions"));
        }
        // The object is aliased to the subject: keep only the subject column.
        kept.retain(|p| *p != Position::O);
        vec![(Position::S, Position::O)]
    } else {
        Vec::new()
    };
    if kept.is_empty() {
        return Err(invalid(
            "present BGP: each atom must keep at least one variable this pass (fully-ground atoms unsupported)",
        ));
    }
    // The spec is a valid S/P/O cover by construction; `prove/verify_selection_only`
    // re-check it internally (`AtomFilterSpec::validate`).
    let spec = AtomFilterSpec { constants, kept: kept.clone(), equalities };
    Ok((spec, kept))
}

/// The kept-column combined-dict id commitments for `kept`, in order (S -> `present_id_s`,
/// O -> `present_id_o`). Both are authenticated to the reconciled hashes by [`verify_id_auth`].
pub(super) fn present_kept_source_coms(
    kept: &[Position],
    auth_s: &PresentIdAuth,
    auth_o: &PresentIdAuth,
) -> io::Result<Vec<ArkGT>> {
    kept.iter()
        .map(|pos| match pos {
            Position::S => Ok(auth_s.com_id),
            Position::O => Ok(auth_o.com_id),
            Position::P => Err(invalid("present BGP: predicate variable is unsupported this pass")),
        })
        .collect()
}

/// The kept-column combined-dict id COLUMNS for `kept` (prover side), in the same S,O order.
pub(super) fn present_kept_sources(kept: &[Position], artifacts: &ReconcileArtifacts) -> Vec<Vec<ArkFr>> {
    kept.iter()
        .map(|pos| match pos {
            Position::S => artifacts.present_id_s.clone(),
            Position::O => artifacts.present_id_o.clone(),
            Position::P => unreachable!("predicate is a constant this pass"),
        })
        .collect()
}

/// Whether any atom carries an object-value range (drives building the combined range dict).
pub(super) fn bgp_needs_range_dict(atoms: &[PresentBgpAtom]) -> bool {
    atoms.iter().any(|atom| atom.object_range.is_some())
}

// ===========================================================================
// Wave 7 — TRUST-BOUNDARY HARDENING: cryptographically assert the two store invariants the
// reconcile verifier's soundness relied on out of band, instead of trusting the store.
//
//   (1) present DUPLICATE-FREENESS: present is a SET — no two rows share the same canonical
//       triple. Asserted over the BOUNDED combined-dict-id key `(id_s, id_p, id_o)` — the id
//       space the join/BGP operates in — because a 254-bit canonical-HASH key cannot be
//       strict-sorted (adjacent gaps of near-modulus field elements exceed the 253-bit range
//       the sort gadget bounds). A duplicated present row (a store that smuggled a duplicated
//       addition into a bag) has a duplicated id-triple and is rejected.
//   (2) combined-dict ID UNIQUENESS: the combined node/value AND predicate dictionaries assign
//       DISTINCT ids (no two dict rows share an id), so the `id → hash` lookup every group-hash /
//       BGP step performs is a FUNCTION and the id space present is joined in is well-formed. This
//       is PROVED over COMMITTED data by `prove_remint` / `verify_remint` (the id side, below) —
//       NOT by (1)'s bounded strict-sort — so a store that reused/duplicated a dict id is rejected
//       on the fully term-less path too.
//
// Assertion (1) reuses the audited FULL-FIELD permutation ([`crate::proof::permutation`]) + BOUNDED
// strict-sort ([`crate::proof::sort`]) as a distinctness argument over the packed id-triple key:
// commit the sorted column U, prove U is a permutation of the key column (reciprocal LogUp,
// field-native, no ordering cap), then prove U STRICTLY increases (each adjacent gap in
// `[1, 2^bits)`, forcing all entries distinct). The key columns are the reconciliation-authenticated
// id commitments (bound to the trusted head root); the whole argument runs on the shared
// reconcile/BGP transcript.
//
// NO-RE-MINT / NO-RE-USE (I4): dictionary HASH-uniqueness (no term re-minted with two DIFFERENT ids
// — a duplicate HASH sitting at two ids) AND dictionary ID-uniqueness (assertion (2): no id re-used
// by two rows — a duplicate ID carrying two hashes) are both PROVED over COMMITTED data by
// `prove_remint` / `verify_remint` (below), run by both reconcile cores. Detecting a duplicate among
// 254-bit canonical hashes needs a full-field distinctness primitive (limb-wise lexicographic strict
// sort) — I4-a's `prove_column_distinct_fullfield` — because the magnitude strict-sort caps keys
// at 253 bits and `hash_object`/`hash_term` values sit near the ~254-bit field modulus, so their
// sorted adjacent gaps overflow the range check on honest stacks; the id columns reuse that SAME
// full-field gadget (their exact mirror). `prove_remint` runs it over each combined `Com(dict_hash)`
// AND `Com(dict_id)` column (the bag union of every layer's root-bound dict rows), so the old
// clear-catalog recompute guard is retired. Assertion (2) — the id-side distinctness — closes
// duplicate/malformed IDS; hash collision-resistance (Blake2b-512) closes distinct-terms-same-hash;
// the same-hash-two-distinct-ids case is closed by the hash-side `verify_remint`; the
// same-id-two-distinct-terms (id re-use) case is closed by its id-side.
// ===========================================================================

/// `2^w` as a field element (`w` is small — a dictionary-id bit-width, `<= 84`).
pub(super) fn pow2_fr(w: usize) -> ArkFr {
    let mut value = f1();
    for _ in 0..w {
        value = value + value;
    }
    value
}

/// The smallest `w` with `2^w > max_val` (so ids in `[0, max_val]` pack injectively at radix
/// `2^w`). `max_val = 0` gives `w = 1` (a `>= 1` width for the sort domain).
pub(super) fn id_pack_width(max_val: usize) -> usize {
    let mut w = 1usize;
    while (1u128 << w) <= max_val as u128 {
        w += 1;
    }
    w
}

/// A distinctness proof for one committed column: the sorted permutation of the column plus
/// the strict-sortedness of that permutation (⇒ the real prefix `[0, n)` is ALL-DISTINCT).
#[derive(Clone, Debug)]
pub struct ColumnDistinctProof {
    pub perm: RelationPermutationProof,
    pub com_sorted: ArkGT,
    pub sorted: SortedProof,
}

/// Prove the real prefix `[0, n)` of the column committed by `com` (values `column`, padded
/// to `2^working_log`) is ALL-DISTINCT, given every real entry is a nonnegative integer
/// `< 2^bits`. Sorts the prefix, proves the sort is a permutation of the committed column,
/// and proves it strictly increases. Requires `n >= 2` (a `<= 1`-row column is trivially a set).
pub(super) fn prove_column_distinct(
    context: &StoreProverContext<'_>,
    column: &[ArkFr],
    com: ArkGT,
    n: usize,
    working_log: usize,
    bits: usize,
    transcript: &mut Transcript254,
) -> io::Result<ColumnDistinctProof> {
    use ark_ff::PrimeField;
    if n < 2 {
        return Err(invalid("column-distinct requires at least two rows"));
    }
    if bits == 0 || bits > 253 {
        return Err(invalid("column-distinct key bit-width out of range (need 1..=253)"));
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| column[a].0.into_bigint().cmp(&column[b].0.into_bigint()));
    let input = RelationWitness {
        columns: vec![column.to_vec()],
        commitments: vec![com],
        log_domain: working_log,
        n,
    };
    let perm = prove_relation_permutation_with_context(context, &input, &order, transcript)?;
    let com_sorted = perm.commitments[0];
    let sorted = prove_strictly_sorted_bits_with_context(
        context, &perm.columns[0], n, bits, &com_sorted, transcript,
    )?;
    Ok(ColumnDistinctProof { perm: perm.proof, com_sorted, sorted })
}

pub(super) fn verify_column_distinct(
    com: ArkGT,
    n: usize,
    working_log: usize,
    bits: usize,
    proof: &ColumnDistinctProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    if n < 2 {
        return Err(invalid("column-distinct requires at least two rows"));
    }
    if bits == 0 || bits > 253 {
        return Err(invalid("column-distinct key bit-width out of range (need 1..=253)"));
    }
    let input = RelationCommit { commitments: vec![com], log_domain: working_log, n };
    verify_relation_permutation(
        &input, std::slice::from_ref(&proof.com_sorted), n, working_log, &proof.perm, transcript,
    )?;
    verify_strictly_sorted_bits(&proof.com_sorted, n, bits, &proof.sorted, transcript)
}

/// Full-field variant of [`ColumnDistinctProof`]: proves the real prefix `[0, n)` of the
/// column committed by `com` is ALL-DISTINCT over the ENTIRE field (each entry a canonical
/// `Fr < p`) — for ~254-bit hash columns the bounded strict-sort's 253-bit gap cannot cover.
/// Sorts the prefix, proves the sort is a permutation of the committed column, and proves the
/// sorted candidate strictly increases LEXICOGRAPHICALLY over 2×127-bit limbs (I4-a).
#[derive(Clone, Debug)]
pub(crate) struct ColumnDistinctFullFieldProof {
    pub perm: RelationPermutationProof,
    pub com_sorted: ArkGT,
    pub lex: LexDistinctProof,
}

/// Prove the real prefix `[0, n)` of the column committed by `com` (values `column`, padded
/// to `2^working_log`) is ALL-DISTINCT over the ENTIRE field — every real entry is a canonical
/// `Fr < p` by construction, so there is NO bit-width bound (unlike [`prove_column_distinct`]).
/// Sorts the prefix, proves the sort is a permutation of the committed column, and proves the
/// sorted candidate strictly increases LEXICOGRAPHICALLY (I4-a). Requires `n >= 1` (a single-row
/// column is a trivially-distinct set).
pub(super) fn prove_column_distinct_fullfield(
    context: &StoreProverContext<'_>,
    column: &[ArkFr],
    com: ArkGT,
    n: usize,
    working_log: usize,
    transcript: &mut Transcript254,
) -> io::Result<ColumnDistinctFullFieldProof> {
    use ark_ff::PrimeField;
    if n < 1 {
        return Err(invalid("full-field column-distinct requires at least one row"));
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| column[a].0.into_bigint().cmp(&column[b].0.into_bigint()));
    let input = RelationWitness {
        columns: vec![column.to_vec()],
        commitments: vec![com],
        log_domain: working_log,
        n,
    };
    let perm = prove_relation_permutation_with_context(context, &input, &order, transcript)?;
    let com_sorted = perm.commitments[0];
    let lex =
        prove_lex_strictly_sorted_with_context(context, &perm.columns[0], n, &com_sorted, transcript)?;
    Ok(ColumnDistinctFullFieldProof { perm: perm.proof, com_sorted, lex })
}

pub(super) fn verify_column_distinct_fullfield(
    com: ArkGT,
    n: usize,
    working_log: usize,
    proof: &ColumnDistinctFullFieldProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    if n < 1 {
        return Err(invalid("full-field column-distinct requires at least one row"));
    }
    let input = RelationCommit { commitments: vec![com], log_domain: working_log, n };
    verify_relation_permutation(
        &input, std::slice::from_ref(&proof.com_sorted), n, working_log, &proof.perm, transcript,
    )?;
    verify_lex_strictly_sorted(&proof.com_sorted, n, &proof.lex, transcript)
}

/// The no-re-mint / no-re-use carrier: full-field distinctness proofs for the COMBINED node/value
/// and predicate dictionary HASH columns AND their ID columns (each the bag union of every chain
/// layer's root-bound `Com(dict_hash)` / `Com(dict_id)`, relocated to the working domain).
///
/// * HASH distinctness (`nv`, `pred`) ⇔ no canonical term is re-MINTED at two ids across the chain
///   (the committed-data replacement for the old clear-catalog `assert_combined_catalog_hash_unique`).
/// * ID distinctness (`nv_id`, `pred_id`) ⇔ no dict id is re-USED by two rows across the chain, so
///   the `id → hash` lookup every group-hash / BGP step performs is a FUNCTION. Without it a
///   malicious head could present two combined-dict rows sharing one id — e.g. `(5, hash(ex:A))`
///   and `(5, hash(ex:Evil))`, all HASHES still distinct so `nv`/`pred` pass — and then resolve a
///   delivered id `5` to the FORGED term `ex:Evil`. This is the id-side mirror of the hash proof.
///
/// Works term-less: a term-less view drops its clear catalog but RETAINS its dict commitments, so
/// both the hash and id columns still contribute to the combined columns and are covered.
#[derive(Clone, Debug)]
pub(crate) struct RemintProof {
    pub nv: ColumnDistinctFullFieldProof,
    pub pred: ColumnDistinctFullFieldProof,
    /// Full-field distinctness of the combined node/value dict ID column (`Com(dict_id)`), the id
    /// side of the no-re-use guarantee (mirrors `nv` on the hash column).
    pub nv_id: ColumnDistinctFullFieldProof,
    /// Full-field distinctness of the combined predicate dict ID column (mirrors `pred`).
    pub pred_id: ColumnDistinctFullFieldProof,
}

/// Prove the combined node/value + predicate dictionary HASH columns are each all-distinct
/// (no re-mint) AND their ID columns are each all-distinct (no re-use). Runs right after
/// `prove_combined_dict` for both dicts, before any group-hash / BGP step trusts the term↔id
/// bijection.
pub(super) fn prove_remint(
    context: &StoreProverContext<'_>,
    combined_nv: &CombinedDictBuilt,
    combined_pred: &CombinedDictBuilt,
    working_log: usize,
    transcript: &mut Transcript254,
) -> io::Result<RemintProof> {
    let nv = prove_column_distinct_fullfield(
        context, &combined_nv.hash_at_k, combined_nv.com_hash_at_k, combined_nv.n, working_log, transcript,
    )?;
    let pred = prove_column_distinct_fullfield(
        context, &combined_pred.hash_at_k, combined_pred.com_hash_at_k, combined_pred.n, working_log, transcript,
    )?;
    // ID side (no re-use): each combined dict's ID column is all-distinct too — no dict id shared by
    // two rows — proved over the SAME root-bound `Com(dict_id)` (`com_id_at_k`) the group-hash
    // lookups resolve against. Exact mirror of the hash proofs above; runs AFTER them so honest
    // hash-side transcript positions are unchanged (both sides append in this same order).
    let nv_id = prove_column_distinct_fullfield(
        context, &combined_nv.id_at_k, combined_nv.com_id_at_k, combined_nv.n, working_log, transcript,
    )?;
    let pred_id = prove_column_distinct_fullfield(
        context, &combined_pred.id_at_k, combined_pred.com_id_at_k, combined_pred.n, working_log, transcript,
    )?;
    Ok(RemintProof { nv, pred, nv_id, pred_id })
}

/// Verify both combined-dict HASH columns are all-distinct (no re-mint) AND both combined-dict ID
/// columns are all-distinct (no re-use). Runs right after `verify_combined_dict` for both dicts,
/// mirroring `prove_remint`'s transcript position. FAILS CLOSED if any of the four columns has a
/// duplicate: the hash proofs bind to `com_hash_at_k`, the id proofs to `com_id_at_k`, both
/// root-bound.
pub(super) fn verify_remint(
    combined_nv: &CombinedDictProof,
    combined_pred: &CombinedDictProof,
    working_log: usize,
    proof: &RemintProof,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    verify_column_distinct_fullfield(combined_nv.com_hash_at_k, combined_nv.n, working_log, &proof.nv, transcript)?;
    verify_column_distinct_fullfield(combined_pred.com_hash_at_k, combined_pred.n, working_log, &proof.pred, transcript)?;
    // ID side (no re-use), bound to the root-committed `Com(dict_id)` columns — same order as the prover.
    verify_column_distinct_fullfield(combined_nv.com_id_at_k, combined_nv.n, working_log, &proof.nv_id, transcript)?;
    verify_column_distinct_fullfield(combined_pred.com_id_at_k, combined_pred.n, working_log, &proof.pred_id, transcript)
}

/// The bounded canonical-triple key column `id_s + 2^w*id_p + 2^(2w)*id_o` for present (an
/// injective pack of the three authenticated dict ids, so distinct keys ⇔ distinct triples).
/// Returns the packed column and the pack bit-width `w`; the caller derives its commitment
/// homomorphically from the id-column commitments.
pub(super) fn present_triple_key(
    artifacts: &ReconcileArtifacts,
    head_nv: usize,
    head_pred: usize,
) -> io::Result<(Vec<ArkFr>, usize)> {
    let w = id_pack_width(head_nv.max(head_pred));
    if 3 * w > 253 {
        return Err(invalid(
            "present dedup: dictionary too large to pack a bounded canonical-triple key (need 3*width <= 253)",
        ));
    }
    let wshift = pow2_fr(w);
    let w2shift = wshift * wshift;
    let size = artifacts.present_id_s.len();
    let key: Vec<ArkFr> = (0..size)
        .map(|i| {
            artifacts.present_id_s[i] + wshift * artifacts.present_id_p[i] + w2shift * artifacts.present_id_o[i]
        })
        .collect();
    Ok((key, w))
}

/// The present canonical-triple key COMMITMENT, derived homomorphically from the authenticated
/// id-column commitments (Dory is linear): `com(id_s) + 2^w*com(id_p) + 2^(2w)*com(id_o)`.
pub(super) fn present_triple_key_commitment(auth_s: &PresentIdAuth, auth_p: &PresentIdAuth, auth_o: &PresentIdAuth, w: usize) -> ArkGT {
    let wshift = pow2_fr(w);
    let w2shift = wshift * wshift;
    auth_s.com_id + wshift * auth_p.com_id + w2shift * auth_o.com_id
}

/// Prove one RANGED atom over `present`: the authenticated object-value range selection
/// (optionally intersected with a second bound), the constant-equality selection, their
/// product, and the compaction of the kept id columns by the combined selection. `present`'s
/// object id column and the combined range dict already live at the working domain, so no
/// cross-domain relocation is needed (unlike the base ranged atom). Returns the subproof, the
/// kept commitments, the dense kept columns (chain witness), and the result length.
#[allow(clippy::too_many_arguments)]
pub(super) fn prove_present_range_atom(
    context: &StoreProverContext<'_>,
    artifacts: &ReconcileArtifacts,
    spec: &AtomFilterSpec,
    kept: &[Position],
    range: &ObjectRangeSpec,
    range_dict: &CombinedRangeDictBuilt,
    head_nv: usize,
    auth_s: &PresentIdAuth,
    auth_o: &PresentIdAuth,
    transcript: &mut Transcript254,
) -> io::Result<(PresentRangeSubproof, Vec<ArkGT>, Vec<Vec<ArkFr>>, usize)> {
    let m = artifacts.present_len;
    // (1) Authenticated object-value range selection over present's object id column.
    let (object_range, com_range_sel) = prove_object_range_select_with_context(
        context, &artifacts.present_id_o, &auth_o.com_id,
        &range_dict.id_at_k, &range_dict.value_at_k, &range_dict.datatype_at_k,
        &range_dict.com_id_at_k, &range_dict.com_value_at_k, &range_dict.com_datatype_at_k,
        head_nv, range.datatype, range.op, range.c, range.b_bits, m, transcript,
    )?;
    let range_sel = object_range_sel_column(
        &artifacts.present_id_o, &range_dict.id_at_k, &range_dict.value_at_k, &range_dict.datatype_at_k,
        head_nv, range.datatype, range.op, range.c, range.b_bits, m,
    );

    // (1b) Optional second bound → intersection of the two masks (InRange / half-open).
    let (effective_range_sel, com_effective, second) = if let Some(bound) = &range.second {
        let (second_object_range, second_com) = prove_object_range_select_with_context(
            context, &artifacts.present_id_o, &auth_o.com_id,
            &range_dict.id_at_k, &range_dict.value_at_k, &range_dict.datatype_at_k,
            &range_dict.com_id_at_k, &range_dict.com_value_at_k, &range_dict.com_datatype_at_k,
            head_nv, range.datatype, bound.op, bound.c, range.b_bits, m, transcript,
        )?;
        let second_sel = object_range_sel_column(
            &artifacts.present_id_o, &range_dict.id_at_k, &range_dict.value_at_k, &range_dict.datatype_at_k,
            head_nv, range.datatype, bound.op, bound.c, range.b_bits, m,
        );
        let (intersect, com_intersection_sel) = prove_boolean_product_with_context(
            context, &range_sel, &second_sel, &com_range_sel, &second_com, transcript,
        )?;
        let intersection: Vec<ArkFr> = range_sel.iter().zip(&second_sel).map(|(x, y)| *x * *y).collect();
        (
            intersection,
            com_intersection_sel,
            Some(PresentSecondRange {
                object_range: second_object_range,
                com_range_sel: second_com,
                intersect,
                com_intersection_sel,
            }),
        )
    } else {
        (range_sel.clone(), com_range_sel, None)
    };

    // (2) Constant-equality selection (predicate + any subject constant) over present's hashes.
    let (equality, com_equality_sel, eq_sel) = prove_selection_only_with_context(
        context, spec,
        &artifacts.present_s, &artifacts.present_p, &artifacts.present_o, m,
        &artifacts.com_present_s, &artifacts.com_present_p, &artifacts.com_present_o, transcript,
    )?;

    // (2h) DATATYPE-HOMOGENEITY guard (fail-closed fidelity). The object-range gadget masks out
    // every matched present row whose object datatype != range.datatype; we CANNOT faithfully
    // answer such a query (rational siblings compared by value / rational-vs-float errors /
    // non-numeric rows reaching Prolog structural `compare/3`, which can INCLUDE a string/node
    // the mask drops = under-report). So we FAIL CLOSED. `present`'s object id column and
    // equality selection share the working domain (no relocation): require
    // `Com(equality_sel ⊙ (1 − dt_match)) == Com(0)`.
    let dt_match = object_range_dtmatch_column(
        &artifacts.present_id_o, &range_dict.id_at_k, &range_dict.value_at_k,
        &range_dict.datatype_at_k, head_nv, range.datatype, m,
    );
    let one_minus_dt: Vec<ArkFr> = dt_match.iter().map(|d| f1() - *d).collect();
    for i in 0..m {
        if eq_sel[i] != f0() && one_minus_dt[i] != f0() {
            return Err(invalid(
                "RangeDatatypeNotHomogeneous: a predicate-matched present row's object has a \
                 datatype other than the range constant's; a numeric range/order/aggregate over \
                 a datatype-heterogeneous column cannot faithfully match TerminusDB's \
                 cross-datatype value+structural comparison — reject-closed (fail-closed)",
            ));
        }
    }
    let domain = artifacts.present_id_o.len();
    let com_ones = context.commit_column(&vec![f1(); domain])?;
    let com_one_minus_dt = com_ones - object_range.com_dt_match;
    let (bad, com_bad) = prove_boolean_product_with_context(
        context, &eq_sel, &one_minus_dt, &com_equality_sel, &com_one_minus_dt, transcript,
    )?;
    let com_zero = context.commit_column(&vec![f0(); domain])?;
    if com_bad != com_zero {
        return Err(invalid(
            "RangeDatatypeNotHomogeneous: the compared present column is not datatype-homogeneous \
             over the predicate-matched rows — reject-closed (fail-closed)",
        ));
    }
    let datatype_guard = PresentRangeDatatypeGuardProof { bad, com_bad };

    // (3) Combined selection = equality_sel · range_sel (both boolean, chi_n-masked).
    let (combine, com_combined_sel) = prove_boolean_product_with_context(
        context, &eq_sel, &effective_range_sel, &com_equality_sel, &com_effective, transcript,
    )?;
    let combined_sel: Vec<ArkFr> = eq_sel.iter().zip(&effective_range_sel).map(|(x, y)| *x * *y).collect();

    // (4) Compact the kept id columns by the combined selection.
    let source_coms = present_kept_source_coms(kept, auth_s, auth_o)?;
    let sources = present_kept_sources(kept, artifacts);
    let (compaction, com_kept, kept_out, mm) = prove_compaction_only_with_context(
        context, &sources, &source_coms, &combined_sel, &com_combined_sel, transcript,
    )?;
    Ok((
        PresentRangeSubproof {
            object_range,
            com_range_sel,
            second,
            equality,
            com_equality_sel,
            datatype_guard,
            combine,
            com_combined_sel,
            compaction,
        },
        com_kept,
        kept_out,
        mm,
    ))
}

pub(crate) fn prove_present_bgp(
    context: &StoreProverContext<'_>,
    layers: &[&LayerCommitment],
    artifacts: &ReconcileArtifacts,
    atoms: &[PresentBgpAtom],
    joins: &[JoinStep],
    transcript: &mut Transcript254,
) -> io::Result<(PresentBgpProof, PresentReadResult)> {
    if atoms.is_empty() {
        return Err(invalid("present BGP needs at least one atom"));
    }
    let m = artifacts.present_len;
    let head_nv = layers.last().unwrap().state.node_value_count as usize;
    let head_pred = layers.last().unwrap().state.predicate_count as usize;
    // (1) Authenticate present's subject/object/predicate combined-dict id columns.
    let auth_s = prove_id_auth(context, &artifacts.present_id_s, m, &artifacts.nv_dict, transcript)?;
    let auth_o = prove_id_auth(context, &artifacts.present_id_o, m, &artifacts.nv_dict, transcript)?;
    let auth_p = prove_id_auth(context, &artifacts.present_id_p, m, &artifacts.pred_dict, transcript)?;

    // (1h) TRUST-BOUNDARY HARDENING (Wave 7): present is a SET — no duplicate canonical triple —
    // proved over the bounded combined-dict-id key `(id_s, id_p, id_o)` on the shared transcript.
    // (Its soundness in TERM space rests on the combined-dict hash-uniqueness the VERIFIER checks
    // via `verify_remint` over the root-bound combined `Com(dict_hash)` columns — see `prove_remint`,
    // run in both reconcile cores right after the combined dicts and before this BGP.)
    let present_distinct = if m >= 2 {
        let (key, w) = present_triple_key(artifacts, head_nv, head_pred)?;
        let com_key = present_triple_key_commitment(&auth_s, &auth_p, &auth_o, w);
        Some(prove_column_distinct(context, &key, com_key, m, artifacts.working_log, 3 * w, transcript)?)
    } else {
        None
    };

    // (1r) The combined (id, value, datatype) range dictionary — built ONLY when some atom
    // ranges (so plain plans keep the same transcript as before), authenticated across the chain.
    let range_dict = if bgp_needs_range_dict(atoms) {
        let branches = range_dict_branches(layers)?;
        Some(prove_combined_range_dict(context, &branches, artifacts.working_log, transcript)?)
    } else {
        None
    };

    // (2) Per-atom: a plain equality selection + compaction, or a ranged atom, both yielding
    //     the kept combined-dict id columns as the atom's filtered relation.
    let mut atom_proofs = Vec::with_capacity(atoms.len());
    let mut relations: Vec<RelationWitness> = Vec::with_capacity(atoms.len());
    for atom in atoms {
        let (spec, kept) = present_atom_spec(atom)?;
        let (filter, com_kept, kept_out, mm) = if let Some(range) = &atom.object_range {
            let dict = range_dict
                .as_ref()
                .ok_or_else(|| invalid("present BGP: ranged atom without a combined range dict"))?;
            let (sub, com_kept, kept_out, mm) = prove_present_range_atom(
                context, artifacts, &spec, &kept, range, dict, head_nv, &auth_s, &auth_o, transcript,
            )?;
            (PresentAtomFilter::Range(sub), com_kept, kept_out, mm)
        } else {
            let (selection, com_sel, sel) = prove_selection_only_with_context(
                context, &spec,
                &artifacts.present_s,
                &artifacts.present_p,
                &artifacts.present_o,
                m,
                &artifacts.com_present_s,
                &artifacts.com_present_p,
                &artifacts.com_present_o,
                transcript,
            )?;
            let source_coms = present_kept_source_coms(&kept, &auth_s, &auth_o)?;
            let sources = present_kept_sources(&kept, artifacts);
            let (compaction, com_kept, kept_out, mm) = prove_compaction_only_with_context(
                context, &sources, &source_coms, &sel, &com_sel, transcript,
            )?;
            (PresentAtomFilter::Plain { selection, com_sel, compaction }, com_kept, kept_out, mm)
        };
        relations.push(RelationWitness {
            columns: kept_out,
            commitments: com_kept.clone(),
            log_domain: artifacts.working_log,
            n: mm,
        });
        atom_proofs.push(PresentBgpAtomProof { filter, com_kept, m: mm });
    }

    // (3) The left-deep chain over the filtered relations (k = 1 is already the result).
    let (chain, result_columns, result_com_count, result_log_domain, result_m) = if relations.len() == 1 {
        let r = &relations[0];
        (None, r.columns.clone(), r.commitments.len(), r.log_domain, r.n)
    } else {
        let res = prove_bgp_chain_with_result_and_context(context, &relations, joins, transcript)?;
        let arity = res.commitments.len();
        (Some(res.proof), res.columns, arity, res.log_domain, res.n)
    };

    let columns: Vec<Vec<ArkFr>> = result_columns.iter().map(|c| c[..result_m].to_vec()).collect();
    Ok((
        PresentBgpProof {
            auth_s,
            auth_o,
            auth_p,
            present_distinct,
            range_dict: range_dict.map(|rd| rd.proof),
            atoms: atom_proofs,
            chain,
            result_len: result_m,
            result_log_domain,
            result_arity: result_com_count,
            // Prover default: the term-less producer attaches the openings witness separately via
            // `attach_termless_reconcile_openings`; a term-bearing artifact leaves this `None`.
            openings: None,
            // C2a prover default: the term-less range producer attaches these separately via
            // `attach_termless_present_range_openings`; a term-bearing artifact leaves them empty.
            range_object_oid: Vec::new(),
            range_value_openings: None,
        },
        // BGP output columns are combined-dict ids (resolved directly as ids); no I2a id-hints.
        PresentReadResult { columns, len: result_m, id_hints: Vec::new() },
    ))
}

/// Verify one RANGED atom over `present`, mirroring [`prove_present_range_atom`] on the
/// shared transcript in prover order, concluding `ap.com_kept[0..ap.m]` is EXACTLY the
/// range-filtered kept relation.
#[allow(clippy::too_many_arguments)]
pub(super) fn verify_present_range_atom(
    verified: &ReconcileVerified,
    range_dict: &CombinedRangeDictProof,
    head_nv: usize,
    spec: &AtomFilterSpec,
    source_coms: &[ArkGT],
    range: &ObjectRangeSpec,
    sub: &PresentRangeSubproof,
    com_kept: &[ArkGT],
    m_kept: usize,
    auth_o: &PresentIdAuth,
    transcript: &mut Transcript254,
) -> io::Result<()> {
    let m = verified.present_len;
    // (1) Object-value range selection over present's authenticated object id column.
    verify_object_range_select(
        &auth_o.com_id,
        &range_dict.com_id_at_k, &range_dict.com_value_at_k, &range_dict.com_datatype_at_k,
        head_nv, range.datatype, range.op, range.c, range.b_bits, m, &sub.com_range_sel,
        &sub.object_range, transcript,
    )?;
    // (1b) Optional second bound → intersection.
    let com_effective = match (&range.second, &sub.second) {
        (Some(bound), Some(second)) => {
            verify_object_range_select(
                &auth_o.com_id,
                &range_dict.com_id_at_k, &range_dict.com_value_at_k, &range_dict.com_datatype_at_k,
                head_nv, range.datatype, bound.op, bound.c, range.b_bits, m, &second.com_range_sel,
                &second.object_range, transcript,
            )?;
            verify_boolean_product(
                &sub.com_range_sel, &second.com_range_sel, &second.com_intersection_sel,
                &second.intersect, transcript,
            )?;
            second.com_intersection_sel
        }
        (None, None) => sub.com_range_sel,
        _ => return Err(invalid("present BGP range: second-bound presence disagrees with the plan")),
    };
    // (2) Constant-equality selection.
    verify_selection_only(
        spec, m, &verified.com_present_s, &verified.com_present_p, &verified.com_present_o,
        &sub.com_equality_sel, &sub.equality, transcript,
    )?;
    // (2h) DATATYPE-HOMOGENEITY guard (fail-closed fidelity): require
    // `Com(equality_sel ⊙ (1 − dt_match)) == Com(0)` over the object-range gadget's committed
    // `dt_match` (working domain) — every predicate-matched present row carries the pinned
    // datatype, else the heterogeneous column is reject-closed (we refuse rather than
    // under-report TerminusDB's cross-datatype comparison).
    let domain = 1usize << verified.working_log;
    let com_ones = commit_column(&vec![f1(); domain])?;
    let com_one_minus_dt = com_ones - sub.object_range.com_dt_match;
    verify_boolean_product(
        &sub.com_equality_sel, &com_one_minus_dt, &sub.datatype_guard.com_bad,
        &sub.datatype_guard.bad, transcript,
    )?;
    let com_zero = commit_column(&vec![f0(); domain])?;
    if sub.datatype_guard.com_bad != com_zero {
        return Err(invalid(
            "RangeDatatypeNotHomogeneous: the compared present column is not datatype-homogeneous \
             over the predicate-matched rows — reject-closed (fail-closed)",
        ));
    }
    // (3) Combined selection = equality_sel · range_sel.
    verify_boolean_product(
        &sub.com_equality_sel, &com_effective, &sub.com_combined_sel, &sub.combine, transcript,
    )?;
    // (4) Compaction of the kept id columns by the combined selection.
    verify_compaction_only(source_coms, &sub.com_combined_sel, com_kept, m_kept, &sub.compaction, transcript)
}

pub(crate) fn verify_present_bgp(
    views: &[VerifierCommitmentRef<'_>],
    combined_nv: &CombinedDictProof,
    combined_pred: &CombinedDictProof,
    verified: &ReconcileVerified,
    head_nv: usize,
    atoms: &[PresentBgpAtom],
    joins: &[JoinStep],
    proof: &PresentBgpProof,
    transcript: &mut Transcript254,
) -> io::Result<(Vec<ArkGT>, usize, usize)> {
    if atoms.is_empty() {
        return Err(invalid("present BGP needs at least one atom"));
    }
    if proof.atoms.len() != atoms.len() {
        return Err(invalid("present BGP atom count mismatch"));
    }
    let m = verified.present_len;
    let head_pred = views.last().unwrap().state.predicate_count as usize;
    // (1) present id authentication (subject/object vs combined node/value dict, predicate vs
    //     combined predicate dict) against the reconciled hashes.
    verify_id_auth(combined_nv, m, head_nv, verified.com_present_s, &proof.auth_s, transcript)?;
    verify_id_auth(combined_nv, m, head_nv, verified.com_present_o, &proof.auth_o, transcript)?;
    verify_id_auth(combined_pred, m, head_pred, verified.com_present_p, &proof.auth_p, transcript)?;

    // (1h) TRUST-BOUNDARY HARDENING (Wave 7): assert the store invariant `present`'s SET proof rests
    // on. The sibling HASH-UNIQUENESS invariant (the no-re-mint guard: the combined node/value +
    // predicate dictionaries assign a UNIQUE id per TERM, no hash at two ids across the chain) is
    // PROVED once per reconciled entry by `verify_remint` (I4), run by `verify_reconcile_core` over the
    // root-bound combined `Com(dict_hash)` columns (and by `verify_delta_read_core` on the delta-read
    // path, which reuses this BGP core without going through `verify_reconcile_core`), so it already
    // holds here — no second check.
    //   present is a SET (no duplicate canonical triple) over the bounded id key; the key commitment
    //   is DERIVED from the authenticated id commitments, so a fabricated key is rejected. Given
    //   hash-uniqueness (term ↔ id bijective), id-triple distinctness ⇒ TERM-triple distinctness.
    //   Presence is fixed by the public present length (a `< 2`-row present is trivially a set).
    let present_needs_distinct = m >= 2;
    if present_needs_distinct != proof.present_distinct.is_some() {
        return Err(invalid("present BGP: present-distinct presence disagrees with the plan"));
    }
    if let Some(distinct) = &proof.present_distinct {
        let w = id_pack_width(head_nv.max(head_pred));
        if 3 * w > 253 {
            return Err(invalid(
                "present dedup: dictionary too large to pack a bounded canonical-triple key (need 3*width <= 253)",
            ));
        }
        let com_key = present_triple_key_commitment(&proof.auth_s, &proof.auth_p, &proof.auth_o, w);
        verify_column_distinct(com_key, m, verified.working_log, 3 * w, distinct, transcript)?;
    }

    // (1r) The combined range dictionary — required iff (and only iff) some atom ranges.
    let needs_range = bgp_needs_range_dict(atoms);
    // (1r-tl) Merkle-dict I2b / E2a-D2 → C2a: a numeric object-value RANGE authenticates each
    // object id's ORDER-PRESERVING VALUE + DATATYPE against the committed
    // `node_value_dict_range_commitments` columns, whose tie to the REAL terms rests on the catalog
    // re-MSM a TERM-LESS artifact drops. When EVERY view is term-bearing that re-MSM (in each
    // `validate()`) restores the tie and the term-bearing range path below is UNCHANGED. When ANY
    // view is term-less the tie is instead OPENED per range atom via prover-shipped value↔term
    // Merkle openings ([`verify_dict_value_terms_chain`], in the RANGE arm below); a range atom
    // shipping NO such openings fails CLOSED there (the old blanket-reject default, now atom-local
    // so the openings path can relax it precisely).
    let all_terms_present = views.iter().all(|v| v.terms_present);
    if needs_range != proof.range_dict.is_some() {
        return Err(invalid("present BGP: range-dict presence disagrees with the plan"));
    }
    if let Some(range_dict) = &proof.range_dict {
        let branches = range_dict_branch_commits(views)?;
        verify_combined_range_dict(&branches, head_nv, verified.working_log, range_dict, transcript)?;
    }

    // (2) Per-atom filter (plain or ranged), rebuilding each filtered relation from the
    //     authenticated kept commitments (never trusting a fresh relation commitment).
    let mut relations: Vec<RelationCommit> = Vec::with_capacity(atoms.len());
    for (atom, ap) in atoms.iter().zip(proof.atoms.iter()) {
        let (spec, kept) = present_atom_spec(atom)?;
        let source_coms = present_kept_source_coms(&kept, &proof.auth_s, &proof.auth_o)?;
        if ap.com_kept.len() != source_coms.len() {
            return Err(invalid("present BGP: kept commitment arity does not match the atom's variables"));
        }
        match (&atom.object_range, &ap.filter) {
            (Some(range), PresentAtomFilter::Range(sub)) => {
                let range_dict = proof
                    .range_dict
                    .as_ref()
                    .ok_or_else(|| invalid("present BGP: ranged atom without a combined range dict"))?;
                verify_present_range_atom(
                    verified, range_dict, head_nv, &spec, &source_coms, range, sub,
                    &ap.com_kept, ap.m, &proof.auth_o, transcript,
                )?;
                // C2a: over a TERM-LESS view the committed order-preserving value/datatype columns
                // the range gadget just authenticated are NOT tied to the real terms (the catalog
                // re-MSM is dropped). Restore the tie: recommit the present object-id column to
                // `auth_o.com_id` and re-derive value/datatype from the CHAIN-opened Merkle terms
                // (`verify_dict_value_terms_chain`), then run the NaN gate over the opened touched
                // terms (MANDATORY — present range supports Float64). No openings ⇒ fail CLOSED (the
                // D2 default). Term-bearing: unchanged (each `validate()`'s re-MSM already ties them).
                if !all_terms_present {
                    let openings = proof.range_value_openings.as_ref().ok_or_else(|| {
                        invalid("term-less present range requires prover-shipped value<->term openings")
                    })?;
                    let domain = 1usize << verified.working_log;
                    // Primary object-range lookup: tie its committed value/datatype to the terms.
                    let touched = verify_dict_value_terms_chain(
                        views,
                        &proof.auth_o.com_id,
                        &sub.object_range.lookup.com_v_val,
                        &sub.object_range.lookup.com_v_dt,
                        m,
                        domain,
                        &proof.range_object_oid,
                        openings,
                    )?;
                    reject_nan_objects(&touched, range.datatype)?;
                    // A second bound ranges over the SAME object column (same `oid`/openings), but
                    // its own separately-committed lookup value/datatype columns MUST be tied too —
                    // else `InRange` re-opens the value forge on the second half.
                    if let Some(second) = &sub.second {
                        let touched2 = verify_dict_value_terms_chain(
                            views,
                            &proof.auth_o.com_id,
                            &second.object_range.lookup.com_v_val,
                            &second.object_range.lookup.com_v_dt,
                            m,
                            domain,
                            &proof.range_object_oid,
                            openings,
                        )?;
                        reject_nan_objects(&touched2, range.datatype)?;
                    }
                } else {
                    // #49 term-BEARING present range: mirror the base path's fidelity guard
                    // (`query/range_atom.rs`, the `layer.terms_present` arm runs
                    // `reject_nan_catalog`). A Float64/Float32 range reads the order-preserving
                    // VALUE encoding, which assigns every IEEE NaN a total-order position — so a
                    // catalog NaN of the queried float datatype would give a wrong-but-verifying
                    // answer (Prolog treats NaN as unordered). The term-less arm above already
                    // NaN-guards over the opened touched terms; the term-bearing arm (the C2a
                    // re-pin runs ONLY when `!all_terms_present`) previously did NOT. Scan each
                    // term-bearing view's clear node/value catalog, keyed on `range.datatype`
                    // (`range.datatype` covers BOTH bounds — the second bound shares it). No-op for
                    // a non-float range (`reject_nan_catalog` returns early) and for the term-less
                    // path (handled above).
                    for view in views {
                        reject_nan_catalog(view.node_value_catalog, range.datatype)?;
                    }
                }
            }
            (None, PresentAtomFilter::Plain { selection, com_sel, compaction }) => {
                verify_selection_only(
                    &spec, m, &verified.com_present_s, &verified.com_present_p, &verified.com_present_o,
                    com_sel, selection, transcript,
                )?;
                verify_compaction_only(&source_coms, com_sel, &ap.com_kept, ap.m, compaction, transcript)?;
            }
            _ => return Err(invalid("present BGP: atom filter kind disagrees with the plan (plain vs range)")),
        }
        relations.push(RelationCommit {
            commitments: ap.com_kept.clone(),
            log_domain: verified.working_log,
            n: ap.m,
        });
    }

    // (3) The left-deep chain over the filtered relations.
    let (result_coms, result_m, result_log_domain) = if relations.len() == 1 {
        if proof.chain.is_some() {
            return Err(invalid("single-atom present BGP carries a spurious join chain"));
        }
        (relations[0].commitments.clone(), relations[0].n, verified.working_log)
    } else {
        let chain = proof
            .chain
            .as_ref()
            .ok_or_else(|| invalid("multi-atom present BGP is missing its join chain"))?;
        let (coms, mm) = verify_bgp_chain(&relations, joins, chain, transcript)?;
        let log = chain
            .stages
            .last()
            .ok_or_else(|| invalid("present BGP join chain has no stages"))?
            .join
            .num_vars();
        (coms, mm, log)
    };

    if proof.result_len != result_m
        || proof.result_arity != result_coms.len()
        || proof.result_log_domain != result_log_domain
    {
        return Err(invalid("present BGP proof result shape disagrees with the verified chain"));
    }
    Ok((result_coms, result_m, result_log_domain))
}

/// Prove a general n-atom conjunctive BGP over the reconciled N-layer `present`. `atoms`
/// are the conjuncts (constant predicate, variable/constant subject+object, optional
/// subject==object); `joins` is the left-deep plan over the kept id columns (empty for a
/// single atom). Reconciliation + BGP share ONE transcript, so the result is bound to the
/// reconciliation-authenticated relation. The delivered result columns are combined-dict
/// node/value ids (resolution deferred to [`verify_and_resolve_bgp`]).
pub fn prove_read_bgp(
    layers: &[&LayerCommitment],
    atoms: &[PresentBgpAtom],
    joins: &[JoinStep],
) -> io::Result<(ReconcileProof, PresentBgpProof, PresentReadResult)> {
    prove_read_bgp_with_context(&StoreProverContext::cpu(), layers, atoms, joins)
}

/// [`prove_read_bgp`] with a caller-supplied prover context (for GPU/Blitzar MSM offload). The
/// context threads to the reconcile + present-BGP MSMs; identical bytes and proof to the CPU
/// convenience.
pub fn prove_read_bgp_with_context(
    context: &StoreProverContext<'_>,
    layers: &[&LayerCommitment],
    atoms: &[PresentBgpAtom],
    joins: &[JoinStep],
) -> io::Result<(ReconcileProof, PresentBgpProof, PresentReadResult)> {
    let mut transcript = Transcript254::new(b"parture-delta-reconcile-envelope");
    let (reconcile, artifacts) = prove_reconcile_core(context, layers, &mut transcript)?;
    let (bgp, result) = prove_present_bgp(context, layers, &artifacts, atoms, joins, &mut transcript)?;
    Ok((reconcile, bgp, result))
}

/// C1b/E3 (producer): build the UNCOMMITTED term-less resolution witness for a proven reconciled
/// (materialized-present) BGP, so its results can be RESOLVED to real IRIs/typed literals over a
/// TERM-LESS head view. Every BGP kept result column is a combined-dict node/value id (predicates
/// are constants, matched in committed hash space — there is NO catalog dict-membership step to
/// open), so this opens ONLY node/value ids: it maps each column value via
/// `canonical_dictionary_id`, drops the reserved null id 0, and routes each distinct id to its
/// introducing layer via `build_resolution_openings` over the term-bearing prover views (the
/// prover still holds the full catalogs). BODY-IDENTICAL to [`build_termless_delta_openings`]. The
/// witness is re-verified per id at the verifier by `verify_opening` against the root-bound
/// `nv_merkle_root` (a forged opening is rejected, a missing one fails closed) and is NEVER folded
/// into `commitment_root`/`resolution_root`.
pub fn build_termless_reconcile_openings(
    layers: &[&LayerCommitment],
    result_columns: &[Vec<ArkFr>],
) -> io::Result<ResolutionOpenings> {
    let mut nv_ids: Vec<u64> = Vec::new();
    for column in result_columns {
        for value in column {
            let raw = canonical_dictionary_id(value)?;
            // The reserved null id 0 never indexes a dict row (never opened; masked out).
            if raw != 0 {
                nv_ids.push(raw);
            }
        }
    }
    // Term-bearing prover views: the prover holds the clear catalogs and rebuilds each id-indexed
    // tree. Empty predicate id list — BGP result constants carry no dict-membership opening.
    let views: Vec<VerifierCommitmentRef<'_>> = layers.iter().map(|l| l.verifier_view()).collect();
    build_resolution_openings(&views, &nv_ids, &[])
}

/// C1b/E3 (producer): attach the term-less resolution witness to a proven [`PresentBgpProof`], so
/// it can ship alongside a TERM-LESS reconciled artifact. `result_columns` are the prover's
/// [`PresentReadResult::columns`] returned by [`prove_read_bgp`]. Mirrors
/// [`attach_termless_delta_openings`]. Over-inclusion is safe (the verifier only looks up the ids
/// it needs, each independently checked); under-inclusion fails closed. A term-bearing verifier
/// ignores the witness entirely (it rebuilds from the clear catalogs).
pub fn attach_termless_reconcile_openings(
    proof: &mut PresentBgpProof,
    layers: &[&LayerCommitment],
    result_columns: &[Vec<ArkFr>],
) -> io::Result<()> {
    proof.openings = Some(build_termless_reconcile_openings(layers, result_columns)?);
    Ok(())
}

/// C2a (producer): mint the TERM-LESS value↔term witness for a numeric object-value RANGE over a
/// delta / reconciled `present`. `present_object_ids` is the FULL REAL present object-id column —
/// the SAME `artifacts.present_id_o[..present_len]` prefix `auth_o.com_id` commits (on the delta
/// path the head group's object ids; on the reconcile path the reconciled present's object ids, in
/// reconcile order). It records those ids as `oid` (for the verifier's `Com(oid) == auth_o.com_id`
/// recommit), drops the reserved null id 0 for the opening set, and routes each DISTINCT id to its
/// introducing layer via `build_resolution_openings` over the term-bearing prover views (empty
/// predicate list — a range reads only node/value terms). Mirrors [`build_termless_delta_openings`]
/// / [`build_termless_reconcile_openings`], except it opens the PRESENT OBJECT column (not the
/// result columns) — the column the range gadget authenticated `(value, datatype)` against.
///
/// The witness is UNCOMMITTED (never folded into `commitment_root` / `resolution_root`): at verify,
/// `verify_dict_value_terms_chain` recommits `oid` to `auth_o.com_id` (binding the touched set)
/// and re-derives value/datatype from the chain-opened terms, failing closed on any recommit
/// mismatch or missing/forged opening. Over-inclusion is safe (the recommit pins the exact column);
/// a term-bearing verifier ignores the witness.
pub fn build_termless_present_range_openings(
    layers: &[&LayerCommitment],
    present_object_ids: &[ArkFr],
) -> io::Result<(Vec<u64>, ResolutionOpenings)> {
    let mut oid: Vec<u64> = Vec::with_capacity(present_object_ids.len());
    let mut nv_ids: Vec<u64> = Vec::new();
    for value in present_object_ids {
        let raw = canonical_dictionary_id(value)?;
        oid.push(raw);
        // The reserved null id 0 never indexes a dict row (never opened; masked out).
        if raw != 0 {
            nv_ids.push(raw);
        }
    }
    // Term-bearing prover views: the prover holds the clear catalogs and rebuilds each id-indexed
    // tree. Empty predicate id list — a range reads only node/value object terms.
    let views: Vec<VerifierCommitmentRef<'_>> = layers.iter().map(|l| l.verifier_view()).collect();
    let openings = build_resolution_openings(&views, &nv_ids, &[])?;
    Ok((oid, openings))
}

/// C2a (producer): attach the term-less value↔term range witness to a proven [`PresentBgpProof`],
/// so an object-value RANGE resolves+verifies over a TERM-LESS delta / reconciled artifact.
/// `present_object_ids` is the prover's real present object-id column (see
/// [`build_termless_present_range_openings`]). On the delta path this is
/// `proof.bgp` (pass `&mut proof.bgp`); on the reconcile path it is the BGP proof directly. A
/// term-bearing verifier ignores both fields (its committed value/datatype are catalog-tied);
/// leaving them unset keeps a term-less range fail-closed (the D2 default).
pub fn attach_termless_present_range_openings(
    proof: &mut PresentBgpProof,
    layers: &[&LayerCommitment],
    present_object_ids: &[ArkFr],
) -> io::Result<()> {
    let (oid, openings) = build_termless_present_range_openings(layers, present_object_ids)?;
    proof.range_object_oid = oid;
    proof.range_value_openings = Some(openings);
    Ok(())
}

/// Verify a general n-atom BGP read over `present`. On success returns the authenticated
/// result row count. Thin wrapper over [`verify_read_bgp_from_views`] that builds term-bearing
/// views from the full sidecars.
pub fn verify_read_bgp(
    layers: &[&LayerCommitment],
    trusted_head_root: ProofHash,
    atoms: &[PresentBgpAtom],
    joins: &[JoinStep],
    reconcile: &ReconcileProof,
    proof: &PresentBgpProof,
) -> io::Result<usize> {
    let views: Vec<VerifierCommitmentRef<'_>> = layers.iter().map(|l| l.verifier_view()).collect();
    verify_read_bgp_from_views(&views, trusted_head_root, atoms, joins, reconcile, proof)
}

/// Verify a general n-atom BGP read over the reconciled `present` FROM verifier VIEWS (C1/E3).
/// The views-based core entry: it accepts the borrowed [`VerifierCommitmentRef`] slice directly,
/// so a TERM-LESS reconcile view (built via [`LayerCommitment::verifier_commitment_termless`] on
/// the head, term-bearing elsewhere) can be passed for the term-less remote verifier. Plain
/// conjunctive BGP is sound term-less; an object-value RANGE atom fails closed inside
/// `verify_present_bgp` (E2a-D2). Transcript init is identical to [`verify_read_bgp`].
pub fn verify_read_bgp_from_views(
    views: &[VerifierCommitmentRef<'_>],
    trusted_head_root: ProofHash,
    atoms: &[PresentBgpAtom],
    joins: &[JoinStep],
    reconcile: &ReconcileProof,
    proof: &PresentBgpProof,
) -> io::Result<usize> {
    let mut transcript = Transcript254::new(b"parture-delta-reconcile-envelope");
    let verified = verify_reconcile_core(views, trusted_head_root, reconcile, &mut transcript)?;
    let head_nv = views.last().unwrap().state.node_value_count as usize;
    let (_coms, result_m, _log) = verify_present_bgp(
        views, &reconcile.combined_nv, &reconcile.combined_pred, &verified, head_nv, atoms, joins, proof,
        &mut transcript,
    )?;
    Ok(result_m)
}

/// Verify a general n-atom BGP read AND resolve its result bindings to `CanonicalObject`s.
/// Every result column is a combined-dict node/value id (predicates are constants), so
/// each resolves against the chain node/value catalogs. Thin wrapper over
/// [`verify_and_resolve_bgp_from_views`] that builds term-bearing views.
pub fn verify_and_resolve_bgp(
    layers: &[&LayerCommitment],
    trusted_head_root: ProofHash,
    atoms: &[PresentBgpAtom],
    joins: &[JoinStep],
    reconcile: &ReconcileProof,
    proof: &PresentBgpProof,
    claimed_columns: &[Vec<ArkFr>],
) -> io::Result<Vec<Vec<CanonicalObject>>> {
    let views: Vec<VerifierCommitmentRef<'_>> = layers.iter().map(|l| l.verifier_view()).collect();
    verify_and_resolve_bgp_from_views(
        &views, trusted_head_root, atoms, joins, reconcile, proof, claimed_columns,
    )
}

/// Verify a general n-atom BGP read AND resolve its result bindings FROM verifier VIEWS (C1/E3).
/// Views-based analogue of [`verify_and_resolve_bgp`]. Resolution over a TERM-LESS view consumes
/// the prover-shipped Merkle `proof.openings` (E2b, minted by
/// [`attach_termless_reconcile_openings`]): each id is re-verified against the root-bound
/// `nv_merkle_root`, so a forged opening is rejected and a MISSING/absent one fails closed. A
/// term-bearing artifact leaves `proof.openings` `None` and the resolver rebuilds from catalogs.
pub fn verify_and_resolve_bgp_from_views(
    views: &[VerifierCommitmentRef<'_>],
    trusted_head_root: ProofHash,
    atoms: &[PresentBgpAtom],
    joins: &[JoinStep],
    reconcile: &ReconcileProof,
    proof: &PresentBgpProof,
    claimed_columns: &[Vec<ArkFr>],
) -> io::Result<Vec<Vec<CanonicalObject>>> {
    let mut transcript = Transcript254::new(b"parture-delta-reconcile-envelope");
    let verified = verify_reconcile_core(views, trusted_head_root, reconcile, &mut transcript)?;
    let head_nv = views.last().unwrap().state.node_value_count as usize;
    let (result_coms, result_m, result_log_domain) = verify_present_bgp(
        views, &reconcile.combined_nv, &reconcile.combined_pred, &verified, head_nv, atoms, joins, proof,
        &mut transcript,
    )?;
    let kinds = vec![ResolveKind::NodeValueId; result_coms.len()];
    // C1b: pass the prover-shipped term-less resolution openings (BGP results are combined-dict
    // ids with no hash kinds, so `id_hints` stays `None`). `None` on a term-bearing artifact,
    // whose resolver rebuilds from the clear catalogs instead.
    resolve_result_columns(
        views, claimed_columns, &result_coms, result_m, result_log_domain, &kinds, None,
        proof.openings.as_ref(),
        // No fully-term-less RENDER belt on the BGP path (walk-only).
        None,
    )
}


// ===========================================================================
// C4-slice-2: WIRE-SHIPPABLE reconciled materialized-present read + the DB-free byte-in verifier.
// ===========================================================================

/// The shippable reconciled MATERIALIZED-present plain-BGP read proof: the [`ReconcileProof`]
/// (proving `present` IS the head triple set) plus the [`PresentBgpProof`] (the n-atom BGP over
/// that `present`). Unlike a delta-read (a single `DeltaReadProof`), a reconciled present read
/// carries BOTH structs, so this carrier bundles them into ONE framed blob. Its `encode`/`decode`
/// (in `envelope::wire`, framed MAGIC ‖ VERSION ‖ body ‖ SHA-256 with the independent
/// `RECONCILE_READ_*` constants) is transparent to verification: the decoded proof re-verifies
/// through the SAME [`verify_read_bgp_from_views`].
#[derive(Clone, Debug)]
pub struct ReconciledReadBgpProof {
    pub reconcile: ReconcileProof,
    pub bgp: PresentBgpProof,
}

/// C4-slice-2 byte-in remote-verifier entry: verify a reconciled materialized-present plain-BGP
/// read shipped as BYTES against a trusted head root, holding ONLY the encoded proof + the encoded
/// term-less [`VerifierCommitment`](crate::proof::commitment::VerifierCommitment) chain (no
/// `LayerCommitment` sidecars, no DB). Decodes each layer commitment and the read proof, then
/// re-verifies through the audited [`verify_read_bgp_from_views`] — serialization does NOT change
/// verification. `atoms`/`joins`/`trusted_head_root` are the VERIFIER's OWN parameters
/// (anti-substitution): they are NEVER parsed from the proof bytes.
pub fn decode_and_verify_read_bgp(
    read_proof_bytes: &[u8],
    verifier_commitment_bytes: &[&[u8]],
    trusted_head_root: ProofHash,
    atoms: &[PresentBgpAtom],
    joins: &[JoinStep],
) -> io::Result<usize> {
    // Own the decoded commitments FIRST, so the borrowed views below cannot dangle.
    let owned: Vec<crate::proof::commitment::VerifierCommitment> = verifier_commitment_bytes
        .iter()
        .map(|b| crate::proof::commitment::VerifierCommitment::decode(b))
        .collect::<io::Result<_>>()?;
    let views: Vec<VerifierCommitmentRef<'_>> = owned.iter().map(|c| c.verifier_view()).collect();
    let proof = ReconciledReadBgpProof::decode(read_proof_bytes)?;
    verify_read_bgp_from_views(
        &views,
        trusted_head_root,
        atoms,
        joins,
        &proof.reconcile,
        &proof.bgp,
    )
}
