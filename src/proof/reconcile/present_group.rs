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
