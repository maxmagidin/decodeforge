.DEFAULT_GOAL := help

.PHONY: help setup format lint test test-profile check check-pytorch-pin test-native test-bridge-cdylib \
	validate-contracts verify-bundle fixture-check rust-fixture-check \
	check-rust-toolchain repair-rust-toolchain repair-rust-toolchain-apply \
	capture-g0-evidence verify-g0-repository verify-g0-result test-g1-tools \
	prepare-g1-input prepare-g1-cases run-g1-session analyze-g1 verify-g1-result \
	test-g3 test-g3-adapter-real build-g3-bridge run-g3-session run-g3-demo analyze-g3 verify-g3-result \
	prepare-g3-assets prepare-g3-assets-timed verify-g3-assets verify-evaluation-result \
	render-results-visual verify-results-visual

UV := uv
RUST_VERSION := 1.98.0
PYTHON_VERSION := 3.12.14
UV_VERSION := 0.12.5
# Command-line variables are recursive by default. Capture public path inputs as
# raw simple values before expansion; recipes pass them through the shell
# environment so Make cannot reinterpret embedded syntax.
ifeq ($(origin CARGO_TARGET_DIR), undefined)
unexport CARGO_TARGET_DIR
else
override CARGO_TARGET_DIR := $(value CARGO_TARGET_DIR)
export CARGO_TARGET_DIR
endif
override WEIGHTS := $(value WEIGHTS)
override OUTPUT := $(value OUTPUT)
override RECEIPT := $(value RECEIPT)
override ASSETS := $(value ASSETS)
override SPEC := $(value SPEC)
override SESSION_ID := $(value SESSION_ID)
override SESSION_INDEX := $(value SESSION_INDEX)
override MODEL_DIR := $(value MODEL_DIR)
override LIBRARY := $(value LIBRARY)
override LIBRARY_SHA256 := $(value LIBRARY_SHA256)
override PREPARATION_RECEIPT := $(value PREPARATION_RECEIPT)
override SESSION_1 := $(value SESSION_1)
override SESSION_2 := $(value SESSION_2)
override SESSION_3 := $(value SESSION_3)
override OUTPUT_DIR := $(value OUTPUT_DIR)
override BUNDLE := $(value BUNDLE)
override CASES := $(value CASES)
override PREPARED_WEIGHTS := $(value PREPARED_WEIGHTS)
override CHECKOUT := $(value CHECKOUT)
export WEIGHTS OUTPUT RECEIPT ASSETS SPEC SESSION_ID SESSION_INDEX MODEL_DIR
export LIBRARY LIBRARY_SHA256 PREPARATION_RECEIPT SESSION_1 SESSION_2 SESSION_3
export OUTPUT_DIR BUNDLE CASES PREPARED_WEIGHTS CHECKOUT
CARGO := PATH="$$(dirname "$$(rustup which --toolchain $(RUST_VERSION) cargo)"):$$PATH" cargo
G0_RESULT := results/g0/apple-m4-primary/sha256-311053f53efd9c28ab3e4338ca83e78e53acf8c969d9f8a76c6e56f7c2d79d86
G1_RESULT := results/g1/apple-m4-primary
G3_RESULT := results/g3/apple-m4-primary

help:
	@printf '%s\n' \
		'DecodeForge commands' \
		'' \
		'Setup and development:' \
		'  make setup                     Install pinned tools and dependencies' \
		'  make check                     Run the complete development suite' \
		'  make format                    Format Rust and Python sources' \
		'  make lint                      Run static, docs, and schema checks' \
		'  make test-profile              Run the decode-profiler tests' \
		'' \
		'Checked-in results (no model download):' \
		'  make verify-g1-result          Recompute the Apple M4 kernel report' \
		'  make verify-g3-result          Validate the saved model-integration bundle' \
		'  make verify-evaluation-result  Recompute the broader evaluation summary' \
		'' \
		'Run `make -n <target>` to preview a recipe.'

setup:
	@command -v rustup >/dev/null 2>&1 || { echo "setup: rustup is required" >&2; exit 2; }
	@command -v $(UV) >/dev/null 2>&1 || { echo "setup: uv is required" >&2; exit 2; }
	@command -v clang >/dev/null 2>&1 || { echo "setup: Clang is required" >&2; exit 2; }
	@command -v git >/dev/null 2>&1 || { echo "setup: Git is required" >&2; exit 2; }
	@test "$$($(UV) --version | awk '{print $$2}')" = "$(UV_VERSION)" || { \
		echo "setup: expected uv $(UV_VERSION); found $$($(UV) --version)" >&2; exit 2; }
	@rustup toolchain list | grep -Eq '^$(RUST_VERSION)(-|$$)' || \
		rustup toolchain install $(RUST_VERSION) --profile minimal --component rustfmt --component clippy
	@rustup component list --toolchain $(RUST_VERSION) | grep -q '^rustfmt.*(installed)' || \
		rustup component add --toolchain $(RUST_VERSION) rustfmt
	@rustup component list --toolchain $(RUST_VERSION) | grep -q '^clippy.*(installed)' || \
		rustup component add --toolchain $(RUST_VERSION) clippy
	@$(UV) python install $(PYTHON_VERSION)
	@$(UV) sync --locked
	@$(CARGO) fetch --locked
	@echo "setup: ok (Rust $(RUST_VERSION), Python $(PYTHON_VERSION), uv $(UV_VERSION), Clang detected)"

check-rust-toolchain:
	$(UV) run --frozen python scripts/check_rust_toolchain.py --rust-version "$(RUST_VERSION)"

# Inspection is the default. The apply target is an explicit opt-in to a
# narrowly guarded layout repair; ordinary setup does not modify that layout.
repair-rust-toolchain:
	$(UV) run --frozen python scripts/repair_rust_toolchain.py --rust-version "$(RUST_VERSION)"

repair-rust-toolchain-apply:
	$(UV) run --frozen python scripts/repair_rust_toolchain.py --rust-version "$(RUST_VERSION)" --apply

format:
	$(CARGO) fmt --all
	$(UV) run --frozen ruff format python scripts

lint:
	$(UV) run --frozen python scripts/check_docs.py
	$(UV) run --frozen python scripts/render_results_visual.py --verify
	$(CARGO) fmt --all --check
	$(CARGO) clippy --workspace --all-targets --all-features --locked -- -D warnings
	RUSTDOCFLAGS="-D warnings" $(CARGO) doc --workspace --no-deps --locked
	$(UV) lock --check
	$(UV) run --frozen ruff format --check python scripts
	$(UV) run --frozen ruff check python scripts
	$(UV) run --frozen --extra g1-benchmark --extra g3-generation mypy
	$(UV) run --frozen python scripts/check_workspace.py
	$(UV) run --frozen python scripts/check_headers.py
	$(UV) run --frozen python scripts/validate_schemas.py --all

test: test-bridge-cdylib
	$(CARGO) build --workspace --all-features --locked
	$(CARGO) test --workspace --all-features --locked
	$(CARGO) test --workspace --all-features --locked --release
	$(UV) run --frozen --extra g1-benchmark --extra g3-generation python -m pytest -q
	$(UV) run --frozen python scripts/generate_q8_fixtures.py --check
	$(MAKE) rust-fixture-check
	$(CARGO) run --quiet --locked -p decodeforge -- --version

test-profile:
	$(UV) run --frozen --extra g3-generation python -m pytest -q \
		python/tests/test_decode_profile.py python/tests/test_profile_capture.py

check: lint test verify-g1-result
	$(MAKE) verify-g3-result BUNDLE="$(G3_RESULT)"
	$(MAKE) verify-evaluation-result

verify-evaluation-result:
	$(UV) run --frozen python scripts/analyze_evaluation.py \
		--correctness results/evaluation/apple-m4-v1/correctness-v1.json \
		--performance results/evaluation/apple-m4-v1/performance-0.json \
			results/evaluation/apple-m4-v1/performance-1.json \
			results/evaluation/apple-m4-v1/performance-2.json \
		--spec benchmarks/evaluation-v1/spec.json \
		--verify-summary results/evaluation/apple-m4-v1/summary.json

render-results-visual:
	$(UV) run --frozen python scripts/render_results_visual.py

verify-results-visual:
	$(UV) run --frozen python scripts/render_results_visual.py --verify

check-pytorch-pin:
	$(UV) run --frozen --extra pytorch-cpu python -c 'import platform, torch; assert torch.__version__.split("+")[0] == "2.13.0"; print(f"pytorch-pin: ok (torch={torch.__version__}, host={platform.system()}:{platform.machine()})")'

test-native:
	@test "$$(uname -s):$$(uname -m)" = "Darwin:arm64" || { \
		echo "test-native: requires an Apple-arm64 macOS host" >&2; exit 2; }
	$(UV) run --frozen python scripts/check_headers.py
	$(CARGO) test --locked --all-features -p decodeforge-runtime -p decodeforge-compiler -p decodeforge-bridge
	$(CARGO) test --locked --all-features --release -p decodeforge-runtime -p decodeforge-compiler -p decodeforge-bridge

test-bridge-cdylib:
	$(CARGO) build --quiet --release --locked -p decodeforge-bridge
	@set -eu; \
	case "$$(uname -s)" in \
		Darwin) library="$${CARGO_TARGET_DIR:-target}/release/libdecodeforge_bridge.dylib"; uv_args="--extra pytorch-cpu" ;; \
		Linux) library="$${CARGO_TARGET_DIR:-target}/release/libdecodeforge_bridge.so"; uv_args="" ;; \
		*) echo "test-bridge-cdylib: unsupported host $$(uname -s)" >&2; exit 2 ;; \
	esac; \
	test -f "$$library"; \
	$(CARGO) run --quiet --release --locked -p decodeforge-bridge --example export_ffi_fixture | \
	$(UV) run --frozen $$uv_args python scripts/check_bridge_cdylib.py --library "$$library"

test-g1-tools:
	$(CARGO) test --locked -p decodeforge-compiler --bin decodeforge-g1-bench
	$(UV) run --frozen --extra g1-benchmark python -m pytest -q \
		python/tests/test_prepare_g1_inputs.py python/tests/test_g1_evidence.py

prepare-g1-input:
	@test -n "$${WEIGHTS}" || { echo "prepare-g1-input: WEIGHTS=<full model.safetensors> is required" >&2; exit 2; }
	@test -n "$${OUTPUT}" || { echo "prepare-g1-input: OUTPUT=<one-tensor safetensors> is required" >&2; exit 2; }
	$(UV) run --frozen --extra g1-benchmark python scripts/prepare_g1_inputs.py \
		--weights "$${WEIGHTS}" --output "$${OUTPUT}"

prepare-g1-cases:
	@test -n "$${PREPARED_WEIGHTS}" || { echo "prepare-g1-cases: PREPARED_WEIGHTS=<one-tensor safetensors> is required" >&2; exit 2; }
	@test -n "$${OUTPUT}" || { echo "prepare-g1-cases: OUTPUT=<case directory> is required" >&2; exit 2; }
	$(CARGO) run --quiet --release --locked -p decodeforge-compiler \
		--bin decodeforge-g1-bench -- prepare-cases \
		--weights "$${PREPARED_WEIGHTS}" --output "$${OUTPUT}"

run-g1-session:
	@test "$$(uname -s):$$(uname -m)" = "Darwin:arm64" || { \
		echo "run-g1-session: requires an Apple-arm64 macOS host" >&2; exit 2; }
	@test -n "$${CASES}" || { echo "run-g1-session: CASES=<case manifest> is required" >&2; exit 2; }
	@test -n "$${OUTPUT}" || { echo "run-g1-session: OUTPUT=<session JSON> is required" >&2; exit 2; }
	@test -n "$${SESSION_ID}" || { echo "run-g1-session: SESSION_ID=<unique ID> is required" >&2; exit 2; }
	$(CARGO) build --quiet --release --locked -p decodeforge-compiler \
		--bin decodeforge-g1-bench
	"$${CARGO_TARGET_DIR:-target}/release/decodeforge-g1-bench" run-session --cases "$${CASES}" \
		--output "$${OUTPUT}" --session-id "$${SESSION_ID}"

analyze-g1:
	@test -n "$${SESSION_1}" -a -n "$${SESSION_2}" -a -n "$${SESSION_3}" || { \
		echo "analyze-g1: SESSION_1, SESSION_2, and SESSION_3 are required" >&2; exit 2; }
	@test -n "$${OUTPUT_DIR}" || { echo "analyze-g1: OUTPUT_DIR=<directory> is required" >&2; exit 2; }
	$(UV) run --frozen --extra g1-benchmark python scripts/analyze_g1_benchmark.py \
		--sessions "$${SESSION_1}" "$${SESSION_2}" "$${SESSION_3}" \
		--output-dir "$${OUTPUT_DIR}"

verify-g1-result:
	@set -eu; \
	output="$$(mktemp -d "$${TMPDIR:-/tmp}/decodeforge-g1-result.XXXXXX")"; \
	trap 'test -z "$$output" || rm -r -- "$$output"' EXIT; \
	$(UV) run --frozen --extra g1-benchmark python scripts/analyze_g1_benchmark.py \
		--sessions "$(G1_RESULT)/session-01.json" "$(G1_RESULT)/session-02.json" \
			"$(G1_RESULT)/session-03.json" --output-dir "$$output"; \
	diff -u "$(G1_RESULT)/report.json" "$$output/report.json"; \
	diff -u "$(G1_RESULT)/report.md" "$$output/report.md"; \
	echo "verify-g1-result: ok"

prepare-g3-assets: check-rust-toolchain
	@test -n "$${WEIGHTS}" || { echo "prepare-g3-assets: WEIGHTS=<model.safetensors> is required" >&2; exit 2; }
	@test -n "$${OUTPUT}" || { echo "prepare-g3-assets: OUTPUT=<new asset directory> is required" >&2; exit 2; }
	$(CARGO) run --quiet --release --locked -p decodeforge-compiler \
		--bin decodeforge-prepare-qproj -- --source "$${WEIGHTS}" --output "$${OUTPUT}"

prepare-g3-assets-timed: check-rust-toolchain
	@test -n "$${WEIGHTS}" || { echo "prepare-g3-assets-timed: WEIGHTS=<model.safetensors> is required" >&2; exit 2; }
	@test -n "$${OUTPUT}" || { echo "prepare-g3-assets-timed: OUTPUT=<new asset directory> is required" >&2; exit 2; }
	@test -n "$${RECEIPT}" || { echo "prepare-g3-assets-timed: RECEIPT=<new receipt JSON outside OUTPUT> is required" >&2; exit 2; }
	$(CARGO) build --quiet --release --locked -p decodeforge-compiler \
		--bin decodeforge-prepare-qproj
	$(UV) run --frozen python scripts/prepare_g3_assets_timed.py \
		--checkout . --source "$${WEIGHTS}" --output "$${OUTPUT}" \
		--receipt "$${RECEIPT}" --prepare-tool "$${CARGO_TARGET_DIR:-target}/release/decodeforge-prepare-qproj"

verify-g3-assets:
	@test -n "$${ASSETS}" || { echo "verify-g3-assets: ASSETS=<prepared asset directory> is required" >&2; exit 2; }
	$(CARGO) run --quiet --release --locked -p decodeforge-compiler \
		--bin decodeforge-prepare-qproj -- --verify "$${ASSETS}"

test-g3:
	$(CARGO) test --locked -p decodeforge-compiler --lib model_assets::tests::
	$(CARGO) test --locked -p decodeforge-compiler --bin decodeforge-prepare-qproj
	$(UV) run --frozen python scripts/validate_schemas.py --all
	$(UV) run --frozen --extra g3-generation python -m pytest -q \
		python/tests/test_contracts.py \
		python/tests/test_torch_bridge.py \
		python/tests/test_qproj_adapter.py \
		python/tests/test_qproj_model.py \
		python/tests/test_g3_evidence.py \
		python/tests/test_g3_preparation.py \
		python/tests/test_g3_session.py \
		python/tests/test_g3_session_cli.py \
		python/tests/test_g3_results.py

test-g3-adapter-real: check-rust-toolchain verify-g3-assets
	@test "$$(uname -s):$$(uname -m)" = "Darwin:arm64" || { \
		echo "test-g3-adapter-real: requires an Apple-arm64 macOS host" >&2; exit 2; }
	$(CARGO) build --quiet --release --locked -p decodeforge-bridge
	$(UV) run --frozen --extra pytorch-cpu python scripts/check_qproj_adapter_real.py \
		--library "$${CARGO_TARGET_DIR:-target}/release/libdecodeforge_bridge.dylib" \
		--assets "$${ASSETS}" --spec "$${SPEC:-benchmarks/g3/spec.json}"

build-g3-bridge: check-rust-toolchain
	$(CARGO) build --quiet --release --locked -p decodeforge-bridge

run-g3-session:
	@test -n "$${SESSION_ID}" || { echo "run-g3-session: SESSION_ID is required" >&2; exit 2; }
	@test -n "$${SESSION_INDEX}" || { echo "run-g3-session: SESSION_INDEX is required" >&2; exit 2; }
	@test -n "$${MODEL_DIR}" || { echo "run-g3-session: MODEL_DIR is required" >&2; exit 2; }
	@test -n "$${ASSETS}" || { echo "run-g3-session: ASSETS is required" >&2; exit 2; }
	@test -n "$${LIBRARY}" || { echo "run-g3-session: LIBRARY is required" >&2; exit 2; }
	@test -n "$${LIBRARY_SHA256}" || { echo "run-g3-session: LIBRARY_SHA256 is required" >&2; exit 2; }
	@test -n "$${PREPARATION_RECEIPT}" || { echo "run-g3-session: PREPARATION_RECEIPT is required" >&2; exit 2; }
	@test -n "$${OUTPUT}" || { echo "run-g3-session: OUTPUT is required" >&2; exit 2; }
	$(UV) run --frozen --extra g3-generation python scripts/run_g3_session.py \
		--session-id "$${SESSION_ID}" --session-index "$${SESSION_INDEX}" \
		--model-dir "$${MODEL_DIR}" --assets "$${ASSETS}" \
		--library "$${LIBRARY}" --library-sha256 "$${LIBRARY_SHA256}" \
		--preparation-receipt "$${PREPARATION_RECEIPT}" --output "$${OUTPUT}" \
		--spec "$${SPEC:-benchmarks/g3/spec.json}"

run-g3-demo: run-g3-session

analyze-g3:
	@test -n "$${SESSION_1}" -a -n "$${SESSION_2}" -a -n "$${SESSION_3}" || { \
		echo "analyze-g3: SESSION_1, SESSION_2, and SESSION_3 are required" >&2; exit 2; }
	@test -n "$${RECEIPT}" || { echo "analyze-g3: RECEIPT=<preparation receipt JSON> is required" >&2; exit 2; }
	@test -n "$${OUTPUT_DIR}" || { echo "analyze-g3: OUTPUT_DIR=<new directory> is required" >&2; exit 2; }
	$(UV) run --frozen python scripts/analyze_g3_result.py \
		--sessions "$${SESSION_1}" "$${SESSION_2}" "$${SESSION_3}" \
		--preparation-receipt "$${RECEIPT}" \
		--output-dir "$${OUTPUT_DIR}"

verify-g3-result:
	$(UV) run --frozen python scripts/verify_g3_result.py --bundle "$${BUNDLE:-$(G3_RESULT)}"

validate-contracts:
	$(UV) run --frozen python scripts/validate_schemas.py --all

verify-bundle:
	@test -n "$${BUNDLE}" || { echo "verify-bundle: BUNDLE=<path> is required" >&2; exit 2; }
	$(UV) run --frozen python scripts/validate_schemas.py --bundle "$${BUNDLE}"

fixture-check:
	$(UV) run --frozen python scripts/generate_q8_fixtures.py --check

rust-fixture-check:
	$(CARGO) run --quiet --offline --locked -p decodeforge -- q8 verify
	$(CARGO) run --quiet --offline --locked --release -p decodeforge -- q8 verify

capture-g0-evidence:
	@test -n "$${OUTPUT}" || { echo "capture-g0-evidence: OUTPUT=<path> is required" >&2; exit 2; }
	@test -n "$${CHECKOUT}" || { echo "capture-g0-evidence: CHECKOUT=<path> is required" >&2; exit 2; }
	UV_OFFLINE=true CARGO_NET_OFFLINE=true $(UV) run --frozen python scripts/capture_g0_evidence.py --output "$${OUTPUT}" --checkout "$${CHECKOUT}"

verify-g0-repository:
	@test -n "$${BUNDLE}" || { echo "verify-g0-repository: BUNDLE=<path> is required" >&2; exit 2; }
	@test -n "$${CHECKOUT}" || { echo "verify-g0-repository: CHECKOUT=<path> is required" >&2; exit 2; }
	UV_OFFLINE=true $(UV) run --frozen python scripts/verify_g0_repository.py --bundle "$${BUNDLE}" --checkout "$${CHECKOUT}"

verify-g0-result:
	UV_OFFLINE=true $(UV) run --frozen python scripts/validate_schemas.py --bundle "$(G0_RESULT)"
	UV_OFFLINE=true $(UV) run --frozen python scripts/verify_g0_repository.py --bundle "$(G0_RESULT)" --checkout .
