//! `GroupBy` aggregates over the reconciled `present` relation via VERIFIER-RECOMPUTE.
//!
//! The delta-stack analogue of the base path's group aggregation
//! (`crates/woql2/.../columns.rs::expected_group_count_columns` etc.): the group aggregate is a
//! DETERMINISTIC function of the root-authenticated child relation, so it needs NO new crypto —
//! the verifier RE-COMPUTES it over the authenticated present rows.
//!
//! The child BGP result relation is authenticated + resolved by [`verify_and_resolve_bgp`] (the
//! SAME reconcile + present-BGP + resolve path a plain read uses: each kept id column is bound to
//! the head root and resolved to a `CanonicalObject` via the root-bound chain dictionaries, incl.
//! the C2a value↔term tie for value columns). The aggregate then groups the AUTHENTICATED resolved
//! rows and recomputes per group; a fabricated / mis-grouped result is impossible because the
//! verifier never trusts a prover-claimed group — it derives every group from the authenticated
//! rows itself (a tampered child row fails the resolution re-commitment inside
//! [`verify_and_resolve_bgp`] and rejects before any grouping).
//!
//! GROUPING is bound to the AUTHENTICATED key column: groups are keyed by each row's resolved
//! key `CanonicalObject` (a Node IRI). The combined chain dictionary is injective (an IRI is minted
//! in exactly one layer, so it has ONE id across the chain), so grouping by the resolved IRI equals
//! grouping by the authenticated key id — and matches TerminusDB's `group_by/4` (which groups
//! solutions by the group-key binding). This slice covers `GroupCount` (cardinality per group, no
//! value-authentication); `GroupSum` / group `Min`/`Max` extend it with a numeric value column.

use std::collections::BTreeMap;

use super::*;

/// One recomputed group: the group's resolved key objects (one Node IRI per key column, in
/// `key_columns` order) and the number of authenticated child rows in the group.
pub type GroupCountEntry = (Vec<CanonicalObject>, u64);

/// Prove `GroupCount(BGP)` over an N-layer delta stack: reconcile + present-BGP over the child
/// relation (IDENTICAL to [`prove_read_bgp`] — the child rows are the delivered/authenticated
/// data; the groups are recomputed at verify). Returns the reconcile proof, the BGP proof, and the
/// child result relation (its kept id columns).
pub fn prove_read_group_count_bgp(
    layers: &[&LayerCommitment],
    atoms: &[PresentBgpAtom],
    joins: &[JoinStep],
) -> io::Result<(ReconcileProof, PresentBgpProof, PresentReadResult)> {
    prove_read_bgp(layers, atoms, joins)
}

/// Recompute the `GroupCount` output from the root-authenticated resolved child rows: group the
/// rows by their key `CanonicalObject` tuple (each must be a Node IRI) and count per group. The
/// groups are returned sorted by the canonical key-IRI-bytes tuple (deterministic, prover-blind).
pub(super) fn recompute_group_count(
    resolved: &[Vec<CanonicalObject>],
    key_columns: &[usize],
) -> io::Result<Vec<GroupCountEntry>> {
    if key_columns.is_empty() {
        return Err(invalid("reconciled group-count requires at least one key column"));
    }
    let n = resolved.first().map_or(0, Vec::len);
    for &kc in key_columns {
        let column = resolved
            .get(kc)
            .ok_or_else(|| invalid("reconciled group-count key column out of range"))?;
        if column.len() != n {
            return Err(invalid("reconciled group-count key column length mismatch"));
        }
    }
    // Key each group by the canonical IRI-bytes tuple (sorted), carrying the resolved key objects
    // for the readable output. A non-node key (a value literal) reject-closes — TerminusDB groups
    // node-valued keys here, and a value key has no authenticated node identity in this path.
    let mut groups: BTreeMap<Vec<Vec<u8>>, (Vec<CanonicalObject>, u64)> = BTreeMap::new();
    for row in 0..n {
        let mut key_bytes = Vec::with_capacity(key_columns.len());
        let mut key_objects = Vec::with_capacity(key_columns.len());
        for &kc in key_columns {
            match &resolved[kc][row] {
                object @ CanonicalObject::Node(iri) => {
                    key_bytes.push(iri.clone());
                    key_objects.push(object.clone());
                }
                _ => {
                    return Err(invalid(
                        "reconciled group-count key is not a node IRI (value-keyed grouping is unsupported)",
                    ))
                }
            }
        }
        let entry = groups.entry(key_bytes).or_insert((key_objects, 0));
        entry.1 = entry
            .1
            .checked_add(1)
            .ok_or_else(|| invalid("reconciled group-count exceeds u64"))?;
    }
    Ok(groups.into_values().collect())
}

/// Verify + resolve a reconciled `GroupCount(BGP)` read against the trusted head root, then RECOMPUTE
/// the per-group counts over the authenticated resolved child rows. `child_columns` are the prover's
/// delivered child id columns (bound to the head root by [`verify_and_resolve_bgp`] — a tampered /
/// fabricated child row rejects there). `key_columns` index the child schema (== the store result
/// column order). Returns the groups, sorted by key.
#[allow(clippy::too_many_arguments)]
pub fn verify_and_resolve_group_count_bgp(
    layers: &[&LayerCommitment],
    trusted_head_root: ProofHash,
    atoms: &[PresentBgpAtom],
    joins: &[JoinStep],
    key_columns: &[usize],
    reconcile: &ReconcileProof,
    bgp: &PresentBgpProof,
    child_columns: &[Vec<ArkFr>],
) -> io::Result<Vec<GroupCountEntry>> {
    let resolved = verify_and_resolve_bgp(
        layers,
        trusted_head_root,
        atoms,
        joins,
        reconcile,
        bgp,
        child_columns,
    )?;
    recompute_group_count(&resolved, key_columns)
}

// ============================================================================================
// GROUP-SUM — `GroupBy(keys) + Sum(value)` over the reconciled materialized present.
//
// Same verifier-recompute skeleton as GroupCount, but over the C2a-authenticated numeric VALUE
// column: `verify_and_resolve_bgp` binds each value id to its typed literal against the head root
// (the value↔term tie), so summing the RESOLVED numeric value — never the raw object id — is
// value-authenticated. The per-group sum mirrors the base path's FIXED-WIDTH bounded accumulator
// (`crates/woql2/.../columns.rs::GroupSumValue`): `u64` for the unsigned datatypes and `i128`
// (narrowed to `i64` at output) for the signed ones, with `checked_add`/`try_from` REJECT-CLOSING
// an out-of-range sum (TerminusDB sums at arbitrary rational precision; we prove the exact result
// only within this bound and fail closed on overflow — never wrap, never under-report).
//
// STRICT datatype-homogeneity: the value column's datatype is pinned by the child BGP's numeric
// RANGE atom, whose store range-path enforces `Com(equality_sel ⊙ (1 − dt_match)) == Com(0)` — a
// mixed / non-numeric value column reject-closes inside `verify_and_resolve_bgp` before any sum.
// The per-value decode below is a belt: a resolved value whose exact datatype != the planned one
// (or a noncanonical width / non-integral type) reject-closes. Float is excluded entirely (the
// order/range float exclusion: IEEE total-order diverges from WOQL numeric order on -0.0/NaN).

use std::convert::{TryFrom, TryInto};

use crate::proof::CanonicalDatatype;

/// One recomputed group-sum: the group's resolved key objects and the exact per-group sum, rendered
/// as a canonical integer value (`UInt64` for unsigned datatypes, `Int64` for signed).
pub type GroupSumEntry = (Vec<CanonicalObject>, CanonicalObject);

/// The fixed-width bounded sum accumulator — mirrors the base `GroupSumValue` bound EXACTLY.
#[derive(Clone, Copy)]
enum SumTerm {
    Unsigned(u64),
    Signed(i128),
}

impl SumTerm {
    fn zero(datatype: CanonicalDatatype) -> io::Result<Self> {
        match datatype {
            CanonicalDatatype::UInt8
            | CanonicalDatatype::UInt16
            | CanonicalDatatype::UInt32
            | CanonicalDatatype::UInt64 => Ok(Self::Unsigned(0)),
            CanonicalDatatype::Int8
            | CanonicalDatatype::Int16
            | CanonicalDatatype::Int32
            | CanonicalDatatype::Int64 => Ok(Self::Signed(0)),
            _ => Err(invalid(
                "reconciled group-sum requires a fixed-width integral datatype (float/other reject-closed)",
            )),
        }
    }

    fn checked_add(self, rhs: Self) -> io::Result<Self> {
        match (self, rhs) {
            (Self::Unsigned(a), Self::Unsigned(b)) => a
                .checked_add(b)
                .map(Self::Unsigned)
                .ok_or_else(|| invalid("reconciled group-sum unsigned result exceeds u64")),
            (Self::Signed(a), Self::Signed(b)) => a
                .checked_add(b)
                .map(Self::Signed)
                .ok_or_else(|| invalid("reconciled group-sum signed accumulator exceeds i128")),
            _ => Err(invalid("reconciled group-sum mixes signed and unsigned values")),
        }
    }

    /// Render the final sum as a canonical integer value: `UInt64` (plain big-endian) for an
    /// unsigned sum; `Int64` (offset-binary sign-flipped big-endian, the store's signed lexical
    /// convention) for a signed sum, narrowed to `i64` (out of range reject-closes).
    fn into_object(self) -> io::Result<CanonicalObject> {
        match self {
            Self::Unsigned(v) => Ok(CanonicalObject::Value {
                datatype: CanonicalDatatype::UInt64,
                lexical: v.to_be_bytes().to_vec(),
            }),
            Self::Signed(v) => {
                let n = i64::try_from(v)
                    .map_err(|_| invalid("reconciled group-sum signed result exceeds i64"))?;
                Ok(CanonicalObject::Value {
                    datatype: CanonicalDatatype::Int64,
                    lexical: ((n as u64) ^ (1u64 << 63)).to_be_bytes().to_vec(),
                })
            }
        }
    }

    /// Compare two terms by TRUE numeric value (not raw offset-binary lexical) — a negative signed
    /// value orders below a positive one. Both operands share the group's single (homogeneous)
    /// datatype, so they are both `Unsigned` or both `Signed`; a mix reject-closes. Mirrors the base
    /// `GroupSumValue::cmp_same_domain`.
    fn cmp_same_domain(self, rhs: Self) -> io::Result<Ordering> {
        match (self, rhs) {
            (Self::Unsigned(a), Self::Unsigned(b)) => Ok(a.cmp(&b)),
            (Self::Signed(a), Self::Signed(b)) => Ok(a.cmp(&b)),
            _ => Err(invalid("reconciled group-extrema mixes signed and unsigned values")),
        }
    }
}

/// Decode ONE resolved value to its exact integer term, asserting its datatype matches the planned
/// one (homogeneity belt) and its lexical is the canonical fixed width. Mirrors the base
/// `integral_group_sum_value`: unsigned datatypes read plain big-endian; signed read offset-binary
/// (sign-bit-flipped) big-endian.
fn integral_value(object: &CanonicalObject, datatype: CanonicalDatatype) -> io::Result<SumTerm> {
    let CanonicalObject::Value { datatype: actual, lexical } = object else {
        return Err(invalid("reconciled group-sum value is not a typed Store literal"));
    };
    if *actual != datatype {
        return Err(invalid("reconciled group-sum value has the wrong exact datatype"));
    }
    let l = lexical.as_slice();
    Ok(match (datatype, l.len()) {
        (CanonicalDatatype::UInt8, 1) => SumTerm::Unsigned(u64::from(l[0])),
        (CanonicalDatatype::UInt16, 2) => {
            SumTerm::Unsigned(u64::from(u16::from_be_bytes(l.try_into().unwrap())))
        }
        (CanonicalDatatype::UInt32, 4) => {
            SumTerm::Unsigned(u64::from(u32::from_be_bytes(l.try_into().unwrap())))
        }
        (CanonicalDatatype::UInt64, 8) => {
            SumTerm::Unsigned(u64::from_be_bytes(l.try_into().unwrap()))
        }
        (CanonicalDatatype::Int8, 1) => SumTerm::Signed(i128::from((l[0] ^ 0x80) as i8)),
        (CanonicalDatatype::Int16, 2) => {
            let e = u16::from_be_bytes(l.try_into().unwrap());
            SumTerm::Signed(i128::from((e ^ (1u16 << 15)) as i16))
        }
        (CanonicalDatatype::Int32, 4) => {
            let e = u32::from_be_bytes(l.try_into().unwrap());
            SumTerm::Signed(i128::from((e ^ (1u32 << 31)) as i32))
        }
        (CanonicalDatatype::Int64, 8) => {
            let e = u64::from_be_bytes(l.try_into().unwrap());
            SumTerm::Signed(i128::from((e ^ (1u64 << 63)) as i64))
        }
        _ => {
            return Err(invalid(
                "reconciled group-sum value has a noncanonical width or non-integral datatype",
            ))
        }
    })
}

/// Recompute the `GroupSum` output from the root-authenticated resolved child rows: group by the
/// key `CanonicalObject` tuple (each a Node IRI) and sum the authenticated numeric VALUE column per
/// group under the fixed-width bounded accumulator. Groups sorted by the canonical key-IRI-bytes.
pub(super) fn recompute_group_sum(
    resolved: &[Vec<CanonicalObject>],
    key_columns: &[usize],
    value_column: usize,
    datatype: CanonicalDatatype,
) -> io::Result<Vec<GroupSumEntry>> {
    // Float / non-integral datatypes reject-close (IEEE total-order != WOQL numeric order).
    SumTerm::zero(datatype)?;
    if key_columns.is_empty() {
        return Err(invalid("reconciled group-sum requires at least one key column"));
    }
    let n = resolved.first().map_or(0, Vec::len);
    let value_col = resolved
        .get(value_column)
        .ok_or_else(|| invalid("reconciled group-sum value column out of range"))?;
    if value_col.len() != n {
        return Err(invalid("reconciled group-sum value column length mismatch"));
    }
    for &kc in key_columns {
        if kc == value_column {
            return Err(invalid("reconciled group-sum key column aliases the value column"));
        }
        let column = resolved
            .get(kc)
            .ok_or_else(|| invalid("reconciled group-sum key column out of range"))?;
        if column.len() != n {
            return Err(invalid("reconciled group-sum key column length mismatch"));
        }
    }
    let mut groups: BTreeMap<Vec<Vec<u8>>, (Vec<CanonicalObject>, SumTerm)> = BTreeMap::new();
    for row in 0..n {
        let value = integral_value(&value_col[row], datatype)?;
        let mut key_bytes = Vec::with_capacity(key_columns.len());
        let mut key_objects = Vec::with_capacity(key_columns.len());
        for &kc in key_columns {
            match &resolved[kc][row] {
                object @ CanonicalObject::Node(iri) => {
                    key_bytes.push(iri.clone());
                    key_objects.push(object.clone());
                }
                _ => {
                    return Err(invalid(
                        "reconciled group-sum key is not a node IRI (value-keyed grouping is unsupported)",
                    ))
                }
            }
        }
        let entry = groups
            .entry(key_bytes)
            .or_insert((key_objects, SumTerm::zero(datatype)?));
        entry.1 = entry.1.checked_add(value)?;
    }
    groups
        .into_values()
        .map(|(keys, sum)| Ok((keys, sum.into_object()?)))
        .collect()
}

/// Prove `GroupSum(BGP)` over an N-layer delta stack: reconcile + present-BGP over the child
/// relation (IDENTICAL to [`prove_read_bgp`] — the child rows are delivered/authenticated; the sums
/// are recomputed at verify over the C2a-authenticated value column).
pub fn prove_read_group_sum_bgp(
    layers: &[&LayerCommitment],
    atoms: &[PresentBgpAtom],
    joins: &[JoinStep],
) -> io::Result<(ReconcileProof, PresentBgpProof, PresentReadResult)> {
    prove_read_bgp(layers, atoms, joins)
}

/// Verify + resolve a reconciled `GroupSum(BGP)` read against the trusted head root, then RECOMPUTE
/// the per-group sums over the root-authenticated resolved child rows. The value column is the
/// C2a-authenticated numeric literal (a tampered value / mixed datatype rejects inside
/// `verify_and_resolve_bgp`, resp. the per-value decode). `value_column`/`key_columns` index the
/// child schema (== the store result column order). Returns the groups, sorted by key.
#[allow(clippy::too_many_arguments)]
pub fn verify_and_resolve_group_sum_bgp(
    layers: &[&LayerCommitment],
    trusted_head_root: ProofHash,
    atoms: &[PresentBgpAtom],
    joins: &[JoinStep],
    key_columns: &[usize],
    value_column: usize,
    datatype: CanonicalDatatype,
    reconcile: &ReconcileProof,
    bgp: &PresentBgpProof,
    child_columns: &[Vec<ArkFr>],
) -> io::Result<Vec<GroupSumEntry>> {
    let resolved = verify_and_resolve_bgp(
        layers,
        trusted_head_root,
        atoms,
        joins,
        reconcile,
        bgp,
        child_columns,
    )?;
    recompute_group_sum(&resolved, key_columns, value_column, datatype)
}

// ============================================================================================
// GROUP MIN/MAX (extrema) — `GroupBy(keys) + RangeMin/RangeMax(value)` over the reconciled present.
//
// Same verifier-recompute skeleton as GroupSum over the C2a value↔term-authenticated VALUE column,
// but the per-group answer is the group's numeric MINIMUM (resp. MAXIMUM) — one of the group's own
// authenticated values, returned in its ORIGINAL resolved datatype. There is NO overflow concern
// (the extremum is a member value, not an accumulation). Ordering is by TRUE numeric value via
// [`SumTerm::cmp_same_domain`] over the decoded integer (`integral_value` decodes signed offset-
// binary lexicals to their real i128), so a negative signed value orders below a positive one —
// never by the raw offset-binary lexical. Same integral + STRICT datatype-homogeneity guards as
// GroupSum: a mixed / non-numeric / float value column reject-closes.

use std::cmp::Ordering;

use crate::proof::group::GroupExtremaKind;

/// One recomputed group extremum: the group's resolved key objects and the min/max VALUE object
/// (in its original resolved datatype — the extremum is one of the group's authenticated values).
pub type GroupExtremumEntry = (Vec<CanonicalObject>, CanonicalObject);

/// Recompute the `GroupBy + RangeMin/RangeMax` output from the root-authenticated resolved child
/// rows: group by the key `CanonicalObject` tuple (each a Node IRI) and pick the per-group numeric
/// minimum (`Min`) or maximum (`Max`) of the authenticated VALUE column, compared by true numeric
/// value. Groups sorted by the canonical key-IRI-bytes.
pub(super) fn recompute_group_extrema(
    resolved: &[Vec<CanonicalObject>],
    key_columns: &[usize],
    value_column: usize,
    datatype: CanonicalDatatype,
    kind: GroupExtremaKind,
) -> io::Result<Vec<GroupExtremumEntry>> {
    // Float / non-integral datatypes reject-close (IEEE total-order != WOQL numeric order).
    SumTerm::zero(datatype)?;
    if key_columns.is_empty() {
        return Err(invalid("reconciled group-extrema requires at least one key column"));
    }
    let n = resolved.first().map_or(0, Vec::len);
    let value_col = resolved
        .get(value_column)
        .ok_or_else(|| invalid("reconciled group-extrema value column out of range"))?;
    if value_col.len() != n {
        return Err(invalid("reconciled group-extrema value column length mismatch"));
    }
    for &kc in key_columns {
        if kc == value_column {
            return Err(invalid("reconciled group-extrema key column aliases the value column"));
        }
        let column = resolved
            .get(kc)
            .ok_or_else(|| invalid("reconciled group-extrema key column out of range"))?;
        if column.len() != n {
            return Err(invalid("reconciled group-extrema key column length mismatch"));
        }
    }
    // Per group: (resolved key objects, best decoded numeric value, best VALUE object).
    let mut groups: BTreeMap<Vec<Vec<u8>>, (Vec<CanonicalObject>, SumTerm, CanonicalObject)> =
        BTreeMap::new();
    for row in 0..n {
        let decoded = integral_value(&value_col[row], datatype)?;
        let object = value_col[row].clone();
        let mut key_bytes = Vec::with_capacity(key_columns.len());
        let mut key_objects = Vec::with_capacity(key_columns.len());
        for &kc in key_columns {
            match &resolved[kc][row] {
                keyobj @ CanonicalObject::Node(iri) => {
                    key_bytes.push(iri.clone());
                    key_objects.push(keyobj.clone());
                }
                _ => {
                    return Err(invalid(
                        "reconciled group-extrema key is not a node IRI (value-keyed grouping is unsupported)",
                    ))
                }
            }
        }
        match groups.entry(key_bytes) {
            std::collections::btree_map::Entry::Vacant(slot) => {
                slot.insert((key_objects, decoded, object));
            }
            std::collections::btree_map::Entry::Occupied(mut slot) => {
                let entry = slot.get_mut();
                let ordering = decoded.cmp_same_domain(entry.1)?;
                let replace = match kind {
                    GroupExtremaKind::Min => ordering == Ordering::Less,
                    GroupExtremaKind::Max => ordering == Ordering::Greater,
                };
                if replace {
                    entry.1 = decoded;
                    entry.2 = object;
                }
            }
        }
    }
    Ok(groups
        .into_values()
        .map(|(keys, _best, object)| (keys, object))
        .collect())
}

/// Prove `GroupBy + RangeMin/RangeMax(BGP)` over an N-layer delta stack: reconcile + present-BGP
/// over the child relation (IDENTICAL to [`prove_read_bgp`]); the extrema are recomputed at verify
/// over the C2a-authenticated value column.
pub fn prove_read_group_extrema_bgp(
    layers: &[&LayerCommitment],
    atoms: &[PresentBgpAtom],
    joins: &[JoinStep],
) -> io::Result<(ReconcileProof, PresentBgpProof, PresentReadResult)> {
    prove_read_bgp(layers, atoms, joins)
}

/// Verify + resolve a reconciled `GroupBy + RangeMin/RangeMax(BGP)` read against the trusted head
/// root, then RECOMPUTE the per-group min/max over the root-authenticated resolved child rows. A
/// tampered value / mixed datatype rejects inside `verify_and_resolve_bgp` (resp. the per-value
/// decode). Returns the groups, sorted by key.
#[allow(clippy::too_many_arguments)]
pub fn verify_and_resolve_group_extrema_bgp(
    layers: &[&LayerCommitment],
    trusted_head_root: ProofHash,
    atoms: &[PresentBgpAtom],
    joins: &[JoinStep],
    key_columns: &[usize],
    value_column: usize,
    datatype: CanonicalDatatype,
    kind: GroupExtremaKind,
    reconcile: &ReconcileProof,
    bgp: &PresentBgpProof,
    child_columns: &[Vec<ArkFr>],
) -> io::Result<Vec<GroupExtremumEntry>> {
    let resolved = verify_and_resolve_bgp(
        layers,
        trusted_head_root,
        atoms,
        joins,
        reconcile,
        bgp,
        child_columns,
    )?;
    recompute_group_extrema(&resolved, key_columns, value_column, datatype, kind)
}

// ============================================================================================
// C4 byte-in: DB-free reconciled GroupBy-aggregate reads over the TERM-LESS VerifierCommitment chain.
//
// A group aggregate is a VERIFIER-RECOMPUTE over the root-authenticated resolved child rows (see the
// module doc), so — unlike a plain present read, which delivers only the row COUNT — the byte-in
// group path must ship the reconcile + present-BGP proof PLUS the delivered CHILD result columns
// (the grouped-over rows), then re-run the SAME `verify_and_resolve_bgp_from_views` + recompute over
// verifier VIEWS. The GroupSpec (key columns / aggregate kind / value column / datatype) and the BGP
// atoms/joins are the VERIFIER's OWN re-lowered query parameters — NEVER parsed from the proof bytes
// (anti-substitution). The child columns ARE in the bytes (they are the delivered answer data), but
// they are root-authenticated inside `verify_and_resolve_bgp_from_views`, so a tampered / fabricated
// child row rejects there before any grouping.
// ============================================================================================

/// Prover-side: mint ALL term-less resolution witnesses a byte-in GROUP read needs into a proven
/// child [`PresentBgpProof`], so the SAME proof also verifies + RESOLVES DB-free over a TERM-LESS
/// `VerifierCommitment` chain. A group aggregate is a verifier-recompute over the RESOLVED child
/// rows, so — unlike a plain present/count read (which delivers only the row count and needs no
/// openings) — the group byte-in must resolve, which term-less requires the prover-shipped Merkle
/// witnesses:
///   * the result-id openings (always — the node/value child result columns);
///   * for a ranged child (an object-value range atom, i.e. the sum/extrema numeric VALUE column),
///     the value↔term range openings over the reconciled present object column.
/// A term-bearing verifier ignores every witness (it rebuilds from the clear catalogs); leaving them
/// unattached keeps a term-less RESOLVE fail-closed (the D2 default). Call once after proving.
pub fn attach_termless_group_openings(
    bgp: &mut PresentBgpProof,
    layers: &[&LayerCommitment],
    atoms: &[PresentBgpAtom],
    result_columns: &[Vec<ArkFr>],
) -> io::Result<()> {
    attach_termless_reconcile_openings(bgp, layers, result_columns)?;
    // A numeric object-value RANGE atom (the sum/extrema value column is authenticated through the
    // child BGP's `>= 0` range) needs the value↔term openings over the reconciled present object
    // column, in reconcile order (the SAME `present_id_o` prefix `auth_o.com_id` commits).
    if atoms.iter().any(|a| a.object_range.is_some()) {
        let poids: Vec<ArkFr> =
            reconcile_present(layers)?.iter().map(|r| ark(r.object_id)).collect();
        attach_termless_present_range_openings(bgp, layers, &poids)?;
    }
    Ok(())
}

/// Views-based `GroupCount` verify+recompute (the byte-in remote core). Analogue of
/// [`verify_and_resolve_group_count_bgp`] over borrowed [`VerifierCommitmentRef`]s (so a TERM-LESS
/// head view can be passed for the DB-free remote verifier).
#[allow(clippy::too_many_arguments)]
pub fn verify_and_resolve_group_count_bgp_from_views(
    views: &[VerifierCommitmentRef<'_>],
    trusted_head_root: ProofHash,
    atoms: &[PresentBgpAtom],
    joins: &[JoinStep],
    key_columns: &[usize],
    reconcile: &ReconcileProof,
    bgp: &PresentBgpProof,
    child_columns: &[Vec<ArkFr>],
) -> io::Result<Vec<GroupCountEntry>> {
    let resolved = verify_and_resolve_bgp_from_views(
        views, trusted_head_root, atoms, joins, reconcile, bgp, child_columns,
    )?;
    recompute_group_count(&resolved, key_columns)
}

/// Views-based `GroupSum` verify+recompute (the byte-in remote core). Analogue of
/// [`verify_and_resolve_group_sum_bgp`] over borrowed [`VerifierCommitmentRef`]s.
#[allow(clippy::too_many_arguments)]
pub fn verify_and_resolve_group_sum_bgp_from_views(
    views: &[VerifierCommitmentRef<'_>],
    trusted_head_root: ProofHash,
    atoms: &[PresentBgpAtom],
    joins: &[JoinStep],
    key_columns: &[usize],
    value_column: usize,
    datatype: CanonicalDatatype,
    reconcile: &ReconcileProof,
    bgp: &PresentBgpProof,
    child_columns: &[Vec<ArkFr>],
) -> io::Result<Vec<GroupSumEntry>> {
    let resolved = verify_and_resolve_bgp_from_views(
        views, trusted_head_root, atoms, joins, reconcile, bgp, child_columns,
    )?;
    recompute_group_sum(&resolved, key_columns, value_column, datatype)
}

/// Views-based group `Min`/`Max` verify+recompute (the byte-in remote core). Analogue of
/// [`verify_and_resolve_group_extrema_bgp`] over borrowed [`VerifierCommitmentRef`]s.
#[allow(clippy::too_many_arguments)]
pub fn verify_and_resolve_group_extrema_bgp_from_views(
    views: &[VerifierCommitmentRef<'_>],
    trusted_head_root: ProofHash,
    atoms: &[PresentBgpAtom],
    joins: &[JoinStep],
    key_columns: &[usize],
    value_column: usize,
    datatype: CanonicalDatatype,
    kind: GroupExtremaKind,
    reconcile: &ReconcileProof,
    bgp: &PresentBgpProof,
    child_columns: &[Vec<ArkFr>],
) -> io::Result<Vec<GroupExtremumEntry>> {
    let resolved = verify_and_resolve_bgp_from_views(
        views, trusted_head_root, atoms, joins, reconcile, bgp, child_columns,
    )?;
    recompute_group_extrema(&resolved, key_columns, value_column, datatype, kind)
}

/// Which per-group aggregate a wire group read was PROVED for. Carried in the proof only as an
/// ANTI-SUBSTITUTION WITNESS (the byte-in verifier compares it to its OWN re-lowered query spec and
/// reject-closes on mismatch — the byte-level analogue of the in-memory `expected.group ==
/// proof.group` check). The actual recompute ALWAYS uses the VERIFIER's params, never these bytes.
/// Mirrors the compiler's `GroupAgg`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GroupReadAgg {
    /// Per-group cardinality (`GroupCount`).
    Count,
    /// Per-group numeric sum over `value_column` (pinned fixed-width integral `datatype`).
    Sum { value_column: usize, datatype: CanonicalDatatype },
    /// Per-group numeric min/max over `value_column` (pinned fixed-width integral `datatype`).
    Extrema { value_column: usize, datatype: CanonicalDatatype, kind: GroupExtremaKind },
}

/// The `GroupBy` spec a wire group read was PROVED for (key columns + aggregate). Carried ONLY as an
/// anti-substitution witness (see [`GroupReadAgg`]); the recompute uses the verifier's own params.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupReadSpec {
    pub key_columns: Vec<usize>,
    pub agg: GroupReadAgg,
}

/// The shippable reconciled `GroupBy`-aggregate read: the [`ReconcileProof`] + the child
/// [`PresentBgpProof`] + the delivered child result columns (the grouped-over rows) + the
/// [`GroupReadSpec`] the proof was produced for. Its `encode`/`decode` (in `envelope::wire`, framed
/// MAGIC ‖ VERSION ‖ body ‖ SHA-256 with the independent `GROUP_READ_*` constants) is transparent to
/// verification: the decoded proof re-verifies through the SAME
/// `verify_and_resolve_group_*_bgp_from_views`, and its carried `spec` is COMPARED to the verifier's
/// re-lowered spec (reject on mismatch) — it is a witness, never the source of the recompute params.
#[derive(Clone, Debug)]
pub struct ReconciledGroupReadProof {
    pub reconcile: ReconcileProof,
    pub bgp: PresentBgpProof,
    pub child_columns: Vec<Vec<ArkFr>>,
    pub spec: GroupReadSpec,
}

/// Decode the term-less commitment chain, own it, and decode the group read bundle. Shared by the
/// three byte-in group verifiers below (the returned owned commitments must outlive the views).
fn decode_group_bundle(
    group_proof_bytes: &[u8],
    verifier_commitment_bytes: &[&[u8]],
) -> io::Result<(Vec<crate::proof::commitment::VerifierCommitment>, ReconciledGroupReadProof)> {
    let owned: Vec<crate::proof::commitment::VerifierCommitment> = verifier_commitment_bytes
        .iter()
        .map(|b| crate::proof::commitment::VerifierCommitment::decode(b))
        .collect::<io::Result<_>>()?;
    let proof = ReconciledGroupReadProof::decode(group_proof_bytes)?;
    Ok((owned, proof))
}

/// C4 byte-in remote-verifier entry: verify a reconciled `GroupCount(BGP)` shipped as BYTES against
/// a trusted head root, holding ONLY the encoded proof + the encoded term-less `VerifierCommitment`
/// chain. `atoms`/`joins`/`key_columns`/`trusted_head_root` are the VERIFIER's OWN parameters
/// (anti-substitution): NEVER parsed from the proof bytes.
pub fn decode_and_verify_group_count(
    group_proof_bytes: &[u8],
    verifier_commitment_bytes: &[&[u8]],
    trusted_head_root: ProofHash,
    atoms: &[PresentBgpAtom],
    joins: &[JoinStep],
    key_columns: &[usize],
) -> io::Result<Vec<GroupCountEntry>> {
    let (owned, proof) = decode_group_bundle(group_proof_bytes, verifier_commitment_bytes)?;
    // Anti-substitution: the wire-carried GroupSpec must equal the VERIFIER's re-lowered spec (agg
    // kind + key columns). A proof produced for a different aggregate / different keys reject-closes
    // here — the byte-level analogue of the in-memory `expected.group == proof.group` check. The
    // recompute below uses the VERIFIER's params, never the bytes.
    let expected = GroupReadSpec { key_columns: key_columns.to_vec(), agg: GroupReadAgg::Count };
    if proof.spec != expected {
        return Err(invalid(
            "reconciled group-count byte-in: the proof's group spec does not match the verifier's query (anti-substitution)",
        ));
    }
    let views: Vec<VerifierCommitmentRef<'_>> = owned.iter().map(|c| c.verifier_view()).collect();
    verify_and_resolve_group_count_bgp_from_views(
        &views, trusted_head_root, atoms, joins, key_columns, &proof.reconcile, &proof.bgp,
        &proof.child_columns,
    )
}

/// C4 byte-in remote-verifier entry for a reconciled `GroupSum(BGP)`. See
/// [`decode_and_verify_group_count`]; `value_column`/`datatype` are the verifier's re-lowered spec.
#[allow(clippy::too_many_arguments)]
pub fn decode_and_verify_group_sum(
    group_proof_bytes: &[u8],
    verifier_commitment_bytes: &[&[u8]],
    trusted_head_root: ProofHash,
    atoms: &[PresentBgpAtom],
    joins: &[JoinStep],
    key_columns: &[usize],
    value_column: usize,
    datatype: CanonicalDatatype,
) -> io::Result<Vec<GroupSumEntry>> {
    let (owned, proof) = decode_group_bundle(group_proof_bytes, verifier_commitment_bytes)?;
    // Anti-substitution: the wire-carried GroupSpec must equal the VERIFIER's re-lowered spec (agg
    // kind + key columns + value column + datatype). See `decode_and_verify_group_count`.
    let expected = GroupReadSpec {
        key_columns: key_columns.to_vec(),
        agg: GroupReadAgg::Sum { value_column, datatype },
    };
    if proof.spec != expected {
        return Err(invalid(
            "reconciled group-sum byte-in: the proof's group spec does not match the verifier's query (anti-substitution)",
        ));
    }
    let views: Vec<VerifierCommitmentRef<'_>> = owned.iter().map(|c| c.verifier_view()).collect();
    verify_and_resolve_group_sum_bgp_from_views(
        &views, trusted_head_root, atoms, joins, key_columns, value_column, datatype,
        &proof.reconcile, &proof.bgp, &proof.child_columns,
    )
}

/// C4 byte-in remote-verifier entry for a reconciled group `Min`/`Max`. See
/// [`decode_and_verify_group_count`]; `value_column`/`datatype`/`kind` are the verifier's spec.
#[allow(clippy::too_many_arguments)]
pub fn decode_and_verify_group_extrema(
    group_proof_bytes: &[u8],
    verifier_commitment_bytes: &[&[u8]],
    trusted_head_root: ProofHash,
    atoms: &[PresentBgpAtom],
    joins: &[JoinStep],
    key_columns: &[usize],
    value_column: usize,
    datatype: CanonicalDatatype,
    kind: GroupExtremaKind,
) -> io::Result<Vec<GroupExtremumEntry>> {
    let (owned, proof) = decode_group_bundle(group_proof_bytes, verifier_commitment_bytes)?;
    // Anti-substitution: the wire-carried GroupSpec must equal the VERIFIER's re-lowered spec (agg
    // kind + key columns + value column + datatype + min/max kind). See `decode_and_verify_group_count`.
    let expected = GroupReadSpec {
        key_columns: key_columns.to_vec(),
        agg: GroupReadAgg::Extrema { value_column, datatype, kind },
    };
    if proof.spec != expected {
        return Err(invalid(
            "reconciled group-extrema byte-in: the proof's group spec does not match the verifier's query (anti-substitution)",
        ));
    }
    let views: Vec<VerifierCommitmentRef<'_>> = owned.iter().map(|c| c.verifier_view()).collect();
    verify_and_resolve_group_extrema_bgp_from_views(
        &views, trusted_head_root, atoms, joins, key_columns, value_column, datatype, kind,
        &proof.reconcile, &proof.bgp, &proof.child_columns,
    )
}
