//! Store-wide switch for writing intrinsic proof state.

use std::io;

/// The environment variable a TerminusDB server reads to choose its
/// store's proof-state mode: `on`, or `off` (the default when unset or empty).
pub const PROOF_STATE_ENV: &str = "TERMINUSDB_PROOF_STATE";

/// Whether commits and pack imports of a store write intrinsic proof state.
///
/// With `On`, every commit prepares its layer's proof sidecar (except commits
/// that ask for none, such as the system graph's) and pack imports compute
/// the sidecar of a layer that arrives without one. With `Off`, commits and
/// imports write none: such a layer has the same on-disk shape as a legacy
/// layer, and `migrate_proof_chain` prepares its proof state later.
///
/// `Off` is the default, so proving is opt-in: a store that is never told
/// otherwise behaves exactly like an unforked one, paying no per-commit
/// commitment cost and never failing a commit onto an unmigrated chain. A
/// deployment that proves reads turns it on, either per process through
/// [`PROOF_STATE_ENV`] (which every newly constructed store reads) or per
/// store in code with
/// [`Store::with_proof_state`](crate::store::Store::with_proof_state), which
/// overrides the environment either way.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ProofStateMode {
    On,
    #[default]
    Off,
}

impl ProofStateMode {
    /// True when the store writes proof state.
    pub fn is_on(self) -> bool {
        self == ProofStateMode::On
    }

    /// Parse `on` or `off` (case-insensitive, surrounding whitespace ignored).
    /// Empty means the default, `Off`.
    pub fn parse(value: &str) -> io::Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "on" => Ok(ProofStateMode::On),
            "" | "off" => Ok(ProofStateMode::Off),
            other => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{PROOF_STATE_ENV} must be `on` or `off`, not `{other}`"),
            )),
        }
    }

    /// The process-wide mode from [`PROOF_STATE_ENV`], for a context that has
    /// no store to consult: a store under construction, or a pack import on a
    /// bare layer store. A malformed setting is a configuration error that
    /// must not quietly become "not proving", so it panics.
    pub fn from_env_or_panic() -> Self {
        // (2018 edition: `panic!` does not capture format arguments implicitly.)
        Self::from_env().unwrap_or_else(|error| panic!("{}", error))
    }

    /// Read the mode from [`PROOF_STATE_ENV`]; unset means `Off`.
    pub fn from_env() -> io::Result<Self> {
        match std::env::var(PROOF_STATE_ENV) {
            Ok(value) => Self::parse(&value),
            Err(std::env::VarError::NotPresent) => Ok(ProofStateMode::Off),
            Err(std::env::VarError::NotUnicode(_)) => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{PROOF_STATE_ENV} is not valid unicode"),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_on_and_off_only() {
        assert_eq!(ProofStateMode::parse("").unwrap(), ProofStateMode::Off);
        assert_eq!(ProofStateMode::parse("on").unwrap(), ProofStateMode::On);
        assert_eq!(ProofStateMode::parse(" OFF ").unwrap(), ProofStateMode::Off);
        assert!(ProofStateMode::parse("false").is_err());
        assert!(ProofStateMode::parse("1").is_err());
    }

    #[test]
    fn default_is_off_so_proving_is_opt_in() {
        assert_eq!(ProofStateMode::default(), ProofStateMode::Off);
        assert!(!ProofStateMode::default().is_on());
    }
}
