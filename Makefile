.PHONY: help build build-bare test test-bare test-unit test-integration clippy check fmt check-fmt package clean pr

# `proof` is opt-in (see Cargo.toml [features]), and this crate's test suite is
# mostly the proof system, so the default targets here ask for it. The *-bare
# targets are the other configuration — what a consumer that only reads and
# writes triples compiles — and are worth running before a release, since
# nothing else notices a `#[cfg(feature = "proof")]` that should not be there.
FEATURES ?= --features proof

help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

build: ## Build the library (with the proof stack)
	cargo build $(FEATURES)

build-bare: ## Build without the proof stack, as a triples-only consumer does
	cargo build

test: ## Run the full test suite (with the proof stack)
	cargo test $(FEATURES)

test-bare: ## Run the suite without the proof stack
	TERMINUSDB_PROOF_STATE=off cargo test

test-unit: ## Run unit tests (all tests currently live in-module)
	cargo test $(FEATURES) --lib

test-integration: ## Run integration-level tests (pack import/export suites)
	cargo test $(FEATURES) --lib storage::pack

clippy: ## Run clippy on lib and tests with all features (benches need nightly)
	cargo clippy --lib --tests --all-features

check: ## Run cargo check, both feature configurations
	cargo check $(FEATURES)
	cargo check

fmt: ## Format all Rust code
	cargo fmt

check-fmt: ## Check formatting without changing files
	cargo fmt --check

package: ## Package the crate, like CI
	cargo package --allow-dirty --no-verify

clean: ## Remove build artifacts
	cargo clean

pr: check-fmt clippy build test test-bare ## Run pre-PR checks (fmt, clippy, build, both test configurations)
