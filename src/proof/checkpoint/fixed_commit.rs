//! Append-only Dory column commitments at a FIXED matrix layout — the homomorphic append
//! ("Phase 1c") the checkpoint dictionary columns need.
//!
//! An ordinary proof column is committed at the layout its own padded length implies
//! (`matrix_dims(log2 len)`): a column that grows past a power of two changes layout, so its old
//! commitment cannot be extended. A checkpoint dictionary column only ever grows at the end
//! (chain ids are stable), so it is committed at ONE layout for every length: the largest PF4
//! domain's, `2^SETUP_MAX_LOG_N` coefficients laid out as `2^FIXED_NU` rows of `2^FIXED_SIGMA`
//! columns over the tier-`SETUP_MAX_LOG_N` setup. Position `p` sits in row `p >> FIXED_SIGMA`,
//! column `p & (2^FIXED_SIGMA - 1)`, and
//!
//! ```text
//! Com(v) = Π_rows e( Σ_cols v[row, col] · Γ1[col], Γ2[row] )
//! ```
//!
//! is linear in `v` with zero rows contributing the identity. Hence for a segment `s` written at
//! the global offset `o`,
//!
//! ```text
//! Com(old ‖ s at o) = Com(old) + Com(s at o)            (GT written additively)
//! ```
//!
//! and computing `Com(s at o)` costs `O(|s|)` G1 work plus one pairing per touched row, never
//! `O(o)`. The commitment of a column `v` of any length equals `commit_column(v zero-padded to
//! 2^SETUP_MAX_LOG_N)` (whose standard dims are exactly `(FIXED_NU, FIXED_SIGMA)` on the same
//! setup), so it is an ordinary Dory commitment openable at a `SETUP_MAX_LOG_N`-variable point.

use std::io;

use dory_pcs::backends::arkworks::{ArkFr, ArkGT, G1Routines, BN254};
use dory_pcs::primitives::arithmetic::{DoryRoutines, Group, PairingCurve};

use crate::proof::commitment::{prover_setup, SETUP_MAX_LOG_N};
use crate::proof::invalid;

/// Column count (log) of the fixed layout: `ceil(SETUP_MAX_LOG_N / 2)`.
pub const FIXED_SIGMA: usize = SETUP_MAX_LOG_N.div_ceil(2);
/// Row count (log) of the fixed layout.
pub const FIXED_NU: usize = SETUP_MAX_LOG_N - FIXED_SIGMA;
/// Longest column the fixed layout holds.
pub const FIXED_MAX_LEN: usize = 1 << SETUP_MAX_LOG_N;

/// `Com(values written at positions offset..offset+len)` in the fixed layout; the identity for an
/// empty segment.
pub fn commit_fixed_segment(offset: usize, values: &[ArkFr]) -> io::Result<ArkGT> {
    if values.is_empty() {
        return Ok(<ArkGT as Group>::identity());
    }
    let end = offset
        .checked_add(values.len())
        .ok_or_else(|| invalid("checkpoint dictionary segment overflows"))?;
    if end > FIXED_MAX_LEN {
        return Err(invalid("checkpoint dictionary column exceeds the fixed Dory layout"));
    }
    let setup = prover_setup(SETUP_MAX_LOG_N)?;
    let width = 1usize << FIXED_SIGMA;
    let mut rows = Vec::new();
    let mut g2 = Vec::new();
    let mut position = offset;
    let mut taken = 0usize;
    while taken < values.len() {
        let row = position >> FIXED_SIGMA;
        let column = position & (width - 1);
        let count = (width - column).min(values.len() - taken);
        rows.push(G1Routines::msm(
            &setup.g1_vec[column..column + count],
            &values[taken..taken + count],
        ));
        g2.push(setup.g2_vec[row]);
        position += count;
        taken += count;
    }
    Ok(<BN254 as PairingCurve>::multi_pair(&rows, &g2))
}

/// `Com(column)` of a whole column written from position 0.
pub fn commit_fixed_column(values: &[ArkFr]) -> io::Result<ArkGT> {
    commit_fixed_segment(0, values)
}

/// `Com(old) + Com(segment at offset)`: the append the prover performs and the verifier
/// recomputes from the clear new entries.
pub fn append_fixed(old: &ArkGT, offset: usize, segment: &[ArkFr]) -> io::Result<ArkGT> {
    Ok(old.add(&commit_fixed_segment(offset, segment)?))
}

// ---------------------------------------------------------------------------------------------
// Segmented columns: no cap below 2^32 entries.
//
// One fixed layout holds `2^SETUP_MAX_LOG_N` entries. A checkpoint dictionary column of any
// length is a LIST of such fixed-layout commitments, segment `k` committing rows
// `k·DICT_SEGMENT_LEN .. (k+1)·DICT_SEGMENT_LEN` at local offsets `0..DICT_SEGMENT_LEN`. An
// append fills the last segment and opens new ones as needed — each touched segment gains the
// commitment of its part at its local offset, so the cost stays `O(new entries)` plus one pairing
// per touched row. The segment count is `ceil(len / DICT_SEGMENT_LEN)` (none for an empty
// column), so the list is a function of the length and the checkpoint root binds it entirely.
// ---------------------------------------------------------------------------------------------

/// Rows per dictionary segment: one full fixed layout.
pub const DICT_SEGMENT_LEN: usize = FIXED_MAX_LEN;

/// Largest dictionary a checkpoint commits: the term trees' id space (`2^32 - 1` ids).
pub const MAX_DICT_LEN: u64 = (1u64 << super::term_tree::TERM_TREE_DEPTH) - 1;

/// Number of segments of a column of `len` rows.
pub fn segment_count(len: u64, segment_len: usize) -> usize {
    len.div_ceil(segment_len as u64) as usize
}

/// The segment commitments of a whole column written from row 0.
pub fn commit_segmented(values: &[ArkFr], segment_len: usize) -> io::Result<Vec<ArkGT>> {
    append_segmented(&[], 0, values, segment_len)
}

/// Append `values` at row `offset` of a segmented column whose commitments are `old` (which must
/// be exactly the segments of an `offset`-row column), returning the segments of the
/// `offset + values.len()`-row column.
pub fn append_segmented(
    old: &[ArkGT],
    offset: usize,
    values: &[ArkFr],
    segment_len: usize,
) -> io::Result<Vec<ArkGT>> {
    if segment_len == 0 || segment_len > FIXED_MAX_LEN {
        return Err(invalid("checkpoint dictionary segment length is outside the fixed layout"));
    }
    if old.len() != segment_count(offset as u64, segment_len) {
        return Err(invalid("checkpoint dictionary segments do not match the column length"));
    }
    let end = offset
        .checked_add(values.len())
        .filter(|end| *end as u64 <= MAX_DICT_LEN)
        .ok_or_else(|| invalid("checkpoint dictionary column exceeds its id space"))?;
    let mut out = old.to_vec();
    out.resize(segment_count(end as u64, segment_len), <ArkGT as Group>::identity());
    let mut position = offset;
    let mut taken = 0usize;
    while taken < values.len() {
        let segment = position / segment_len;
        let local = position % segment_len;
        let count = (segment_len - local).min(values.len() - taken);
        out[segment] = append_fixed(&out[segment], local, &values[taken..taken + count])?;
        position += count;
        taken += count;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dory_pcs::backends::arkworks::ArkworksPolynomial;
    use dory_pcs::primitives::arithmetic::Field;
    use dory_pcs::primitives::poly::Polynomial;
    use dory_pcs::Transparent;

    fn column(len: usize, seed: u64) -> Vec<ArkFr> {
        (0..len as u64).map(|i| ArkFr::from_u64(seed.wrapping_mul(7919) + i * 31 + 1)).collect()
    }

    #[test]
    fn append_at_offset_equals_from_scratch_commitment() {
        // Segments crossing row boundaries (width 4096) and starting mid-row.
        let full = column(9_000, 3);
        let scratch = commit_fixed_column(&full).unwrap();
        for split in [0usize, 1, 4095, 4096, 4097, 8191, 8999, 9000] {
            let old = commit_fixed_column(&full[..split]).unwrap();
            let appended = append_fixed(&old, split, &full[split..]).unwrap();
            assert_eq!(appended, scratch, "split {split}");
        }
        // Several appends in sequence.
        let mut acc = commit_fixed_column(&[]).unwrap();
        let mut at = 0;
        for len in [1usize, 2, 300, 4000, 1, 4696] {
            acc = append_fixed(&acc, at, &full[at..at + len]).unwrap();
            at += len;
        }
        assert_eq!(at, full.len());
        assert_eq!(acc, scratch);
    }

    #[test]
    fn segmented_appends_fill_the_last_segment_then_open_new_ones() {
        // A small segment length exercises the boundary logic; production uses 2^24.
        let seg = 4096 + 7;
        let full = column(3 * seg + 11, 4);
        let scratch = commit_segmented(&full, seg).unwrap();
        assert_eq!(scratch.len(), 4);
        for (k, chunk) in full.chunks(seg).enumerate() {
            assert_eq!(scratch[k], commit_fixed_column(chunk).unwrap(), "segment {k}");
        }
        // Appends across every boundary, one of them spanning two whole segments.
        let mut acc = Vec::new();
        let mut at = 0;
        for len in [1usize, seg - 2, 1, 2 * seg + 3, 0, full.len() - (1 + seg - 2 + 1 + 2 * seg + 3)] {
            acc = append_segmented(&acc, at, &full[at..at + len], seg).unwrap();
            at += len;
            assert_eq!(acc.len(), segment_count(at as u64, seg));
        }
        assert_eq!(acc, scratch);
        // The old segment list must belong to the offset.
        assert!(append_segmented(&scratch[..3], full.len(), &[ArkFr::one()], seg).is_err());
        assert!(append_segmented(&scratch, 3 * seg, &[ArkFr::one()], seg).is_err());
        assert_eq!(segment_count(0, DICT_SEGMENT_LEN), 0);
        assert_eq!(segment_count(1 << 24, DICT_SEGMENT_LEN), 1);
        assert_eq!(segment_count((1 << 24) + 1, DICT_SEGMENT_LEN), 2);
        assert_eq!(segment_count(MAX_DICT_LEN, DICT_SEGMENT_LEN), 256);
    }

    #[test]
    fn wrong_offset_or_value_changes_the_commitment() {
        let full = column(100, 9);
        let scratch = commit_fixed_column(&full).unwrap();
        let old = commit_fixed_column(&full[..60]).unwrap();
        assert_ne!(append_fixed(&old, 61, &full[60..]).unwrap(), scratch);
        let mut wrong = full[60..].to_vec();
        wrong[3] = wrong[3] + ArkFr::one();
        assert_ne!(append_fixed(&old, 60, &wrong).unwrap(), scratch);
    }

    #[test]
    fn fixed_layout_is_the_dory_commitment_at_the_max_tier_layout() {
        // Two full rows: Dory's own commit with nu = 1, sigma = FIXED_SIGMA on the max-tier setup.
        let values = column(2 << FIXED_SIGMA, 5);
        let poly = ArkworksPolynomial::new(values.clone());
        let (dory, _, _) = poly
            .commit::<BN254, Transparent, G1Routines>(1, FIXED_SIGMA, prover_setup(SETUP_MAX_LOG_N).unwrap())
            .unwrap();
        assert_eq!(commit_fixed_column(&values).unwrap(), dory);
        // A short column is the same commitment as its zero-padded single row.
        let short = column(10, 6);
        let mut row = short.clone();
        row.resize(1 << FIXED_SIGMA, ArkFr::zero());
        let (dory_row, _, _) = ArkworksPolynomial::new(row)
            .commit::<BN254, Transparent, G1Routines>(0, FIXED_SIGMA, prover_setup(SETUP_MAX_LOG_N).unwrap())
            .unwrap();
        assert_eq!(commit_fixed_column(&short).unwrap(), dory_row);
    }
}
