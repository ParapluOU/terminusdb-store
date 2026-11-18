//! A golden commitment root over a fixed layer.
//!
//! The root binds the dictionary-encoded triple columns, the dictionary hashes
//! of the canonical terms, and the order-preserving value/datatype columns, so
//! it moves if any of that encoding changes — including a change that arrives
//! through tdb-succinct rather than through this crate. A ledger that recorded
//! a checkpoint, or a verifier holding a root, would stop verifying silently;
//! this test makes such a change fail loudly instead, with the new root to
//! paste in once it is a deliberate, dev-reset-ed decision.
//!
//! Recorded on the move to tdb-succinct v0.2.1 (TerminusDB 12.1.0-rc3), which
//! left every encoding this touches identical to the 0.1.x rev before it: this
//! exact value was produced by both.

use terminus_store::storage::ProofStateMode;
use terminus_store::store::sync::open_sync_memory_store;
use terminus_store::ValueTriple;

/// The layer covers the term shapes the dictionaries encode differently:
/// a node IRI, a plain string, an integer, a decimal and a dateTime.
const EXPECTED_ROOT: &str =
    "f5b8619cdca36b33a9dc7a902a487346f3b7df83619aac6e3f5445fe978af420";

#[test]
fn fixed_layer_has_its_recorded_commitment_root() {
    let store = open_sync_memory_store().with_proof_state(ProofStateMode::On);
    let builder = store.create_base_layer().unwrap();
    builder
        .add_value_triple(ValueTriple::new_node(
            "http://ex/subject",
            "http://ex/knows",
            "http://ex/object",
        ))
        .unwrap();
    builder
        .add_value_triple(ValueTriple::new_string_value(
            "http://ex/subject",
            "http://ex/label",
            "a string",
        ))
        .unwrap();
    builder
        .add_value_triple(ValueTriple::new_value(
            "http://ex/subject",
            "http://ex/count",
            tdb_succinct::TdbDataType::make_entry(&42i64),
        ))
        .unwrap();
    builder
        .add_value_triple(ValueTriple::new_value(
            "http://ex/subject",
            "http://ex/when",
            tdb_succinct::TdbDataType::make_entry(
                &chrono::NaiveDate::from_ymd_opt(2026, 11, 18)
                    .unwrap()
                    .and_hms_opt(12, 0, 0)
                    .unwrap(),
            ),
        ))
        .unwrap();
    let layer = builder.commit().unwrap();
    let root = layer.proof_commitment().unwrap().state.commitment_root;
    let hex = hex::encode(root);
    assert_eq!(
        hex, EXPECTED_ROOT,
        "the committed encoding changed: every recorded checkpoint root is stale"
    );
}
