//! Transparent (trustless) Dory public-parameter setup.
//!
//! The commitment scheme needs public "keys" (generator vectors) to turn a column of
//! numbers into a succinct commitment and to check an opening. For the trust model to be
//! trustless, EVERY node must derive BYTE-IDENTICAL parameters with nobody holding a secret
//! trapdoor — otherwise a party who knew how the generators were made could forge openings.
//! So this module derives them deterministically from a fixed public "nothing-up-my-sleeve"
//! label ([`setup_seed`]) and caches, per size tier, a prover setup (used while committing)
//! and its matching verifier setup (used while checking). It also holds the small size
//! accounting — how big a domain a given row count needs, which tier serves it, and the
//! per-tier memory budget — so no layer can ask for parameters beyond the format's cap.

use super::*;

/// Domain-separated 32-byte seed for the transparent setup. Derived by hashing
/// a fixed public label ("nothing up my sleeve"), so it is fully reproducible.
pub(super) fn setup_seed() -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"terminus-store/proof-of-woql/dory-transparent-setup/v0");
    hasher.finalize().into()
}

/// Deterministically derive the transparent Dory prover setup.
///
/// Determinism is load-bearing for the trustless model: every node must
/// derive byte-identical public parameters so a commitment is reproducible.
/// `dory_pcs::setup()` samples its generators from `OsRng` (non-deterministic),
/// so we bypass it and seed a portable `ChaCha20Rng` from a fixed hashed label,
/// then sample the G1/G2 generators via arkworks' `UniformRand`.
///
/// In the resolved ark-ec 0.5 backend, `UniformRand<Projective>` samples a
/// base-field x-coordinate and sign, maps them to the curve, and clears the
/// cofactor. It does *not* expose a sampled scalar relative to the canonical
/// generator. Thus the public seed reveals candidate coordinates, but no known
/// discrete-log relation between accepted points. This remains a nonstandard
/// seeded sampler rather than an RFC 9380 BN254 G1/G2 suite; ark-bn254 0.5 does
/// not provide SWU/WB map configurations for either group. Replacing it requires
/// external cryptographic review and a commitment-format migration, not an
/// unaudited local G2 map.
pub(super) fn deterministic_prover_setup(max_log_n: usize) -> ProverSetup<BN254> {
    use ark_ff::UniformRand;
    let mut rng = ChaCha20Rng::from_seed(setup_seed());
    let n = 1usize << max_log_n.div_ceil(2);
    fn g1(rng: &mut impl rand::Rng) -> ArkG1 {
        loop {
            let point = G1Projective::rand(&mut *rng);
            if point != G1Projective::default() {
                return ArkG1(point);
            }
        }
    }
    fn g2(rng: &mut impl rand::Rng) -> ArkG2 {
        loop {
            let point = G2Projective::rand(&mut *rng);
            if point != G2Projective::default() {
                return ArkG2(point);
            }
        }
    }
    let g1_vec: Vec<ArkG1> = (0..n).map(|_| g1(&mut rng)).collect();
    let g2_vec: Vec<ArkG2> = (0..n).map(|_| g2(&mut rng)).collect();
    let h1 = g1(&mut rng);
    let h2 = g2(&mut rng);
    let ht = <BN254 as PairingCurve>::pair(&h1, &h2);
    ProverSetup {
        g1_vec,
        g2_vec,
        h1,
        h2,
        ht,
    }
}

pub(super) fn setup_tier_for_num_vars(num_vars: usize) -> io::Result<usize> {
    SETUP_TIER_LOG_N
        .iter()
        .copied()
        .find(|tier| num_vars <= *tier)
        .ok_or_else(|| {
            invalid(format!(
                "proof domain 2^{num_vars} exceeds PF4's maximum 2^{SETUP_MAX_LOG_N}"
            ))
        })
}

pub(super) fn setup_generator_count(tier: usize) -> io::Result<usize> {
    1usize
        .checked_shl(tier.div_ceil(2) as u32)
        .ok_or_else(|| invalid("Dory setup generator count overflow"))
}

pub(super) fn setup_retained_bytes(tier: usize) -> io::Result<usize> {
    setup_generator_count(tier)?
        .checked_mul(std::mem::size_of::<ArkG1>() + std::mem::size_of::<ArkG2>())
        .and_then(|bytes| {
            bytes.checked_add(
                std::mem::size_of::<ArkG1>()
                    + std::mem::size_of::<ArkG2>()
                    + std::mem::size_of::<ArkGT>(),
            )
        })
        .ok_or_else(|| invalid("Dory setup retained-byte estimate overflow"))
}

pub(super) fn preflight_setup_tier(tier: usize) -> io::Result<()> {
    let retained = setup_retained_bytes(tier)?;
    if retained > MAX_SETUP_RETAINED_BYTES {
        return Err(invalid(format!(
            "Dory setup tier 2^{tier} requires {retained} retained bytes, above the PF4 setup budget"
        )));
    }
    Ok(())
}

pub(super) fn padded_domain_for_rows(rows: usize) -> io::Result<(usize, usize)> {
    if rows == 0 {
        return Ok((0, 0));
    }
    if rows > MAX_RECORDS {
        return Err(invalid(format!(
            "proof table has {rows} records, above the PF4 limit of {MAX_RECORDS}"
        )));
    }
    let padded = rows
        .checked_next_power_of_two()
        .ok_or_else(|| invalid("proof column length overflow"))?
        .max(2);
    let num_vars = padded.trailing_zeros() as usize;
    setup_tier_for_num_vars(num_vars)?;
    Ok((padded, num_vars))
}

/// Peak transient allocation retained by the optimized storage-side
/// commitment builder: one padded field column plus Dory's row commitments.
/// Columns are constructed sequentially, so only the largest table counts.
pub(crate) fn construction_commitment_scratch_bytes(row_counts: &[usize]) -> io::Result<usize> {
    row_counts.iter().try_fold(0usize, |peak, rows| {
        let (padded, num_vars) = padded_domain_for_rows(*rows)?;
        if padded == 0 {
            return Ok(peak);
        }
        let (nu, _) = matrix_dims(num_vars);
        let row_commitments = 1usize
            .checked_shl(nu as u32)
            .and_then(|count| count.checked_mul(std::mem::size_of::<ArkG1>()))
            .ok_or_else(|| invalid("Dory row-commitment scratch estimate overflow"))?;
        let bytes = padded
            .checked_mul(std::mem::size_of::<ArkFr>())
            .and_then(|value| value.checked_add(row_commitments))
            .ok_or_else(|| invalid("Dory commitment scratch estimate overflow"))?;
        Ok(peak.max(bytes))
    })
}

/// Hash the SERIALIZED generator vectors of a tier's deterministic setup. Folded into
/// [`setup_parameter_id`] so the transcript-bound parameter id pins the ACTUAL public
/// parameters, not merely the recipe (label + seed + tier + layout string) that produced them.
/// Without this, a dependency/sampler bump that silently changed how `UniformRand` maps the seed
/// to curve points would leave two builds agreeing on `parameter_id` while holding DIFFERENT
/// generators — cross-verifier ambiguity. Generators are serialized with the same arkworks
/// compressed encoding the rest of the commitment code uses. Cached per tier: the serialization
/// is not free, and `setup_parameter_id` is called on every column-opening prove/verify.
fn setup_generators_hash(tier: usize) -> io::Result<ProofHash> {
    static HASH_20: OnceLock<ProofHash> = OnceLock::new();
    static HASH_22: OnceLock<ProofHash> = OnceLock::new();
    static HASH_24: OnceLock<ProofHash> = OnceLock::new();
    let cell = match tier {
        LEGACY_SETUP_LOG_N => &HASH_20,
        22 => &HASH_22,
        SETUP_MAX_LOG_N => &HASH_24,
        _ => return Err(invalid("Dory setup tier selection returned an unknown tier")),
    };
    if let Some(hash) = cell.get() {
        return Ok(*hash);
    }
    let setup = dory_prover_setup(tier)?;
    let hash = hash_prover_setup_generators(setup)?;
    Ok(*cell.get_or_init(|| hash))
}

/// SHA-256 over the domain-separated, length-framed compressed serialization of every generator
/// the setup holds (`g1_vec`, `g2_vec`, then the blinding `h1`/`h2`/`ht`), in that fixed order.
fn hash_prover_setup_generators(setup: &ProverSetup<BN254>) -> io::Result<ProofHash> {
    fn absorb<T: CanonicalSerialize>(hasher: &mut Sha256, value: &T) -> io::Result<()> {
        let mut bytes = Vec::new();
        value.serialize_compressed(&mut bytes).map_err(|error| {
            invalid(format!("Dory setup generator serialization failed: {error}"))
        })?;
        hasher.update((bytes.len() as u64).to_be_bytes());
        hasher.update(&bytes);
        Ok(())
    }
    let mut hasher = Sha256::new();
    hasher.update(b"terminus-store/proof-of-woql/dory-parameter-generators/v0");
    hasher.update((setup.g1_vec.len() as u64).to_be_bytes());
    for g in &setup.g1_vec {
        absorb(&mut hasher, g)?;
    }
    hasher.update((setup.g2_vec.len() as u64).to_be_bytes());
    for g in &setup.g2_vec {
        absorb(&mut hasher, g)?;
    }
    absorb(&mut hasher, &setup.h1)?;
    absorb(&mut hasher, &setup.h2)?;
    absorb(&mut hasher, &setup.ht)?;
    Ok(hasher.finalize().into())
}

/// Stable public identity of the exact parameter tier selected for a domain.
/// The legacy tier retains its old generator stream; larger tiers are distinct
/// deterministic parameter sets. Query transcripts bind this identity. The id folds a hash of the
/// tier's ACTUAL serialized generators (see [`setup_generators_hash`]), so it binds the real
/// parameters rather than only the label/seed/tier/layout recipe that derived them.
pub(crate) fn setup_parameter_id(num_vars: usize) -> io::Result<ProofHash> {
    let tier = setup_tier_for_num_vars(num_vars)?;
    let generators_hash = setup_generators_hash(tier)?;
    let mut hasher = Sha256::new();
    hasher.update(b"terminus-store/proof-of-woql/dory-parameter-set/v0");
    hasher.update(setup_seed());
    hasher.update((tier as u64).to_be_bytes());
    hasher.update(b"matrix:sigma-ceil-nu-floor;coeff:row-major-lsb-first");
    hasher.update(generators_hash);
    Ok(hasher.finalize().into())
}

/// Lazily derive only the deterministic tier required by this domain. Keeping
/// 2^20 as its own tier preserves all existing v4 commitments byte-for-byte.
pub(super) fn dory_prover_setup(num_vars: usize) -> io::Result<&'static ProverSetup<BN254>> {
    static SETUP_20: OnceLock<ProverSetup<BN254>> = OnceLock::new();
    static SETUP_22: OnceLock<ProverSetup<BN254>> = OnceLock::new();
    static SETUP_24: OnceLock<ProverSetup<BN254>> = OnceLock::new();
    let tier = setup_tier_for_num_vars(num_vars)?;
    preflight_setup_tier(tier)?;
    Ok(match tier {
        20 => SETUP_20.get_or_init(|| deterministic_prover_setup(20)),
        22 => SETUP_22.get_or_init(|| deterministic_prover_setup(22)),
        24 => SETUP_24.get_or_init(|| deterministic_prover_setup(24)),
        _ => unreachable!("setup tier selection returned an unknown tier"),
    })
}

/// The single source of truth for the row/column matrix layout of a padded
/// column of `2^num_vars` scalars: `sigma = ceil(num_vars/2)` columns-bits,
/// `nu = num_vars - sigma` row-bits (so `nu <= sigma`). The flat coefficient
/// index is `row * 2^sigma + col`, folded LSB-first, so an evaluation point's
/// first `sigma` variables are the column bits and the last `nu` are the row
/// bits. The query argument (proof::argument) reuses this so its sumcheck
/// point convention matches the committed columns exactly.
pub(crate) fn matrix_dims(num_vars: usize) -> (usize, usize) {
    let sigma = num_vars.div_ceil(2);
    let nu = num_vars - sigma;
    (nu, sigma)
}

/// The deterministic prover parameter tier selected for this proof domain.
pub(crate) fn prover_setup(num_vars: usize) -> io::Result<&'static ProverSetup<BN254>> {
    dory_prover_setup(num_vars)
}

/// The verifier setup derived once from the global prover setup.
pub(crate) fn verifier_setup(num_vars: usize) -> io::Result<&'static VerifierSetup<BN254>> {
    static SETUP_20: OnceLock<VerifierSetup<BN254>> = OnceLock::new();
    static SETUP_22: OnceLock<VerifierSetup<BN254>> = OnceLock::new();
    static SETUP_24: OnceLock<VerifierSetup<BN254>> = OnceLock::new();
    let tier = setup_tier_for_num_vars(num_vars)?;
    let prover = dory_prover_setup(num_vars)?;
    Ok(match tier {
        20 => SETUP_20.get_or_init(|| prover.to_verifier_setup()),
        22 => SETUP_22.get_or_init(|| prover.to_verifier_setup()),
        24 => SETUP_24.get_or_init(|| prover.to_verifier_setup()),
        _ => unreachable!("setup tier selection returned an unknown tier"),
    })
}

