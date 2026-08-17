//! Residual LogUp helper: batch modular inversion.
//!
//! HISTORY: this module was the shared LogUp (log-derivative) reciprocal toolkit —
//! `rlc_fold` (the `beta` row fingerprint), `reciprocal_witness` (`1/(alpha - fold)`),
//! `reciprocal_claim` (the Identity pin), and `multiset_link_claim` /
//! `masked_multiset_link_claim` (the masked/signed ZeroSum multiset links). Every one of
//! those consumers has been migrated to the GKR fraction tree ([`super::logup_gkr`]) — the
//! reconcile link (the last, masked/signed one) included — so the committed-reciprocal gadget
//! is fully retired. What remains is the one arithmetic helper that outlived it:
//!
//!   * [`pseudo_inverse`] — Montgomery's trick batch inverse (nonzero entries inverted, zeros
//!     passed through) — still used by the `filter` IsZero selector witness.
//!
//! Clean-room from the log-derivative math (cf. p3-lookup, MIT/Apache); no COSL code was
//! consulted or copied.

use dory_pcs::backends::arkworks::ArkFr;
use dory_pcs::primitives::arithmetic::Field;

#[inline]
fn f0() -> ArkFr {
    ArkFr::zero()
}
#[inline]
fn f1() -> ArkFr {
    ArkFr::one()
}

/// Batch-invert (Montgomery's trick over the nonzero subset): returns
/// `vals[i]^{-1}` for each nonzero entry and `0` for zero entries.
pub(crate) fn pseudo_inverse(vals: &[ArkFr]) -> Vec<ArkFr> {
    let indices: Vec<usize> = (0..vals.len()).filter(|&i| !vals[i].is_zero()).collect();
    let m = indices.len();
    let mut out = vec![f0(); vals.len()];
    if m == 0 {
        return out;
    }
    let mut prefix = vec![f1(); m + 1];
    for (t, &i) in indices.iter().enumerate() {
        prefix[t + 1] = prefix[t] * vals[i];
    }
    let mut running = match prefix[m].inv() {
        Some(inv) => inv,
        None => return out, // product of nonzero values is nonzero; unreachable
    };
    for t in (0..m).rev() {
        let i = indices[t];
        out[i] = prefix[t] * running;
        running = running * vals[i];
    }
    out
}
