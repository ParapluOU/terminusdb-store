//! Store-wide switch for writing intrinsic proof state.

use std::io;

/// The environment variable a TerminusDB server reads to choose its
/// store's proof-state mode: `on` (the default when unset or empty) or `off`.
pub const PROOF_STATE_ENV: &str = "TERMINUSDB_PROOF_STATE";

/// Whether commits and pack imports of a store write intrinsic proof state.
///
/// With `On`, every commit prepares its layer's proof sidecar (except commits
/// that ask for none, such as the system graph's) and pack imports compute
/// the sidecar of a layer that arrives without one. With `Off`, commits and
/// imports write none: such a layer has the same on-disk shape as a legacy
/// layer, and `migrate_proof_chain` prepares its proof state later.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ProofStateMode {
    #[default]
    On,
    Off,
}

impl ProofStateMode {
    /// True when the store writes proof state.
    pub fn is_on(self) -> bool {
        self == ProofStateMode::On
    }

    /// Parse `on` or `off` (case-insensitive, surrounding whitespace ignored).
    /// Empty means the default, `On`.
    pub fn parse(value: &str) -> io::Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "on" => Ok(ProofStateMode::On),
            "off" => Ok(ProofStateMode::Off),
            other => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{PROOF_STATE_ENV} must be `on` or `off`, not `{other}`"),
            )),
        }
    }

    /// Read the mode from [`PROOF_STATE_ENV`]; unset means `On`.
    pub fn from_env() -> io::Result<Self> {
        match std::env::var(PROOF_STATE_ENV) {
            Ok(value) => Self::parse(&value),
            Err(std::env::VarError::NotPresent) => Ok(ProofStateMode::On),
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
        assert_eq!(ProofStateMode::parse("").unwrap(), ProofStateMode::On);
        assert_eq!(ProofStateMode::parse("on").unwrap(), ProofStateMode::On);
        assert_eq!(ProofStateMode::parse(" OFF ").unwrap(), ProofStateMode::Off);
        assert!(ProofStateMode::parse("false").is_err());
        assert!(ProofStateMode::parse("1").is_err());
    }
}
