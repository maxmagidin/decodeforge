.PHONY: setup format lint test check check-pytorch-pin test-native \
	validate-contracts verify-bundle fixture-check rust-fixture-check \
	capture-g0-evidence verify-g0-repository verify-g0-result test-g1-tools \
	prepare-g1-input prepare-g1-cases run-g1-session analyze-g1 verify-g1-result

UV := uv
RUST_VERSION := 1.98.0
PYTHON_VERSION := 3.12.14
UV_VERSION := 0.12.5
CARGO := PATH="$$(dirname "$$(rustup which --toolchain $(RUST_VERSION) cargo)"):$$PATH" cargo
G1_BENCH := $(if $(CARGO_TARGET_DIR),$(CARGO_TARGET_DIR),target)/release/decodeforge-g1-bench
G0_RESULT := results/g0/apple-m4-primary/sha256-311053f53efd9c28ab3e4338ca83e78e53acf8c969d9f8a76c6e56f7c2d79d86
G1_RESULT := results/g1/apple-m4-primary

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

format:
	$(CARGO) fmt --all
	$(UV) run --frozen ruff format python scripts

lint:
	$(CARGO) fmt --all --check
	$(CARGO) clippy --workspace --all-targets --all-features --locked -- -D warnings
	RUSTDOCFLAGS="-D warnings" $(CARGO) doc --workspace --no-deps --locked
	$(UV) lock --check
	$(UV) run --frozen ruff format --check python scripts
	$(UV) run --frozen ruff check python scripts
	$(UV) run --frozen --extra g1-benchmark mypy
	$(UV) run --frozen python scripts/check_workspace.py
	$(UV) run --frozen python scripts/check_headers.py
	$(UV) run --frozen python scripts/validate_schemas.py --all

test:
	$(CARGO) build --workspace --all-features --locked
	$(CARGO) test --workspace --all-features --locked
	$(CARGO) test --workspace --all-features --locked --release
	$(UV) run --frozen --extra g1-benchmark python -m pytest -q
	$(UV) run --frozen python scripts/generate_q8_fixtures.py --check
	$(MAKE) rust-fixture-check
	$(CARGO) run --quiet --locked -p decodeforge -- --version

check: lint test verify-g1-result

check-pytorch-pin:
	$(UV) run --frozen --extra pytorch-cpu python -c 'import platform, torch; assert torch.__version__.split("+")[0] == "2.13.0"; print(f"pytorch-pin: ok (torch={torch.__version__}, host={platform.system()}:{platform.machine()})")'

test-native:
	@test "$$(uname -s):$$(uname -m)" = "Darwin:arm64" || { \
		echo "test-native: requires an Apple-arm64 macOS host" >&2; exit 2; }
	$(UV) run --frozen python scripts/check_headers.py
	$(CARGO) test --locked --all-features -p decodeforge-runtime -p decodeforge-compiler
	$(CARGO) test --locked --all-features --release -p decodeforge-runtime -p decodeforge-compiler

test-g1-tools:
	$(CARGO) test --locked -p decodeforge-compiler --bin decodeforge-g1-bench
	$(UV) run --frozen --extra g1-benchmark python -m pytest -q \
		python/tests/test_prepare_g1_inputs.py python/tests/test_g1_evidence.py

prepare-g1-input:
	@test -n "$(WEIGHTS)" || { echo "prepare-g1-input: WEIGHTS=<full model.safetensors> is required" >&2; exit 2; }
	@test -n "$(OUTPUT)" || { echo "prepare-g1-input: OUTPUT=<one-tensor safetensors> is required" >&2; exit 2; }
	$(UV) run --frozen --extra g1-benchmark python scripts/prepare_g1_inputs.py \
		--weights "$(WEIGHTS)" --output "$(OUTPUT)"

prepare-g1-cases:
	@test -n "$(PREPARED_WEIGHTS)" || { echo "prepare-g1-cases: PREPARED_WEIGHTS=<one-tensor safetensors> is required" >&2; exit 2; }
	@test -n "$(OUTPUT)" || { echo "prepare-g1-cases: OUTPUT=<case directory> is required" >&2; exit 2; }
	$(CARGO) run --quiet --release --locked -p decodeforge-compiler \
		--bin decodeforge-g1-bench -- prepare-cases \
		--weights "$(PREPARED_WEIGHTS)" --output "$(OUTPUT)"

run-g1-session:
	@test "$$(uname -s):$$(uname -m)" = "Darwin:arm64" || { \
		echo "run-g1-session: requires an Apple-arm64 macOS host" >&2; exit 2; }
	@test -n "$(CASES)" || { echo "run-g1-session: CASES=<case manifest> is required" >&2; exit 2; }
	@test -n "$(OUTPUT)" || { echo "run-g1-session: OUTPUT=<session JSON> is required" >&2; exit 2; }
	@test -n "$(SESSION_ID)" || { echo "run-g1-session: SESSION_ID=<unique ID> is required" >&2; exit 2; }
	$(CARGO) build --quiet --release --locked -p decodeforge-compiler \
		--bin decodeforge-g1-bench
	"$(G1_BENCH)" run-session --cases "$(CASES)" \
		--output "$(OUTPUT)" --session-id "$(SESSION_ID)"

analyze-g1:
	@test -n "$(SESSION_1)" -a -n "$(SESSION_2)" -a -n "$(SESSION_3)" || { \
		echo "analyze-g1: SESSION_1, SESSION_2, and SESSION_3 are required" >&2; exit 2; }
	@test -n "$(OUTPUT_DIR)" || { echo "analyze-g1: OUTPUT_DIR=<directory> is required" >&2; exit 2; }
	$(UV) run --frozen --extra g1-benchmark python scripts/analyze_g1_benchmark.py \
		--sessions "$(SESSION_1)" "$(SESSION_2)" "$(SESSION_3)" \
		--output-dir "$(OUTPUT_DIR)"

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

validate-contracts:
	$(UV) run --frozen python scripts/validate_schemas.py --all

verify-bundle:
	@test -n "$(BUNDLE)" || { echo "verify-bundle: BUNDLE=<path> is required" >&2; exit 2; }
	$(UV) run --frozen python scripts/validate_schemas.py --bundle "$(BUNDLE)"

fixture-check:
	$(UV) run --frozen python scripts/generate_q8_fixtures.py --check

rust-fixture-check:
	$(CARGO) run --quiet --offline --locked -p decodeforge -- q8 verify
	$(CARGO) run --quiet --offline --locked --release -p decodeforge -- q8 verify

capture-g0-evidence:
	@test -n "$(OUTPUT)" || { echo "capture-g0-evidence: OUTPUT=<path> is required" >&2; exit 2; }
	@test -n "$(CHECKOUT)" || { echo "capture-g0-evidence: CHECKOUT=<path> is required" >&2; exit 2; }
	UV_OFFLINE=true CARGO_NET_OFFLINE=true $(UV) run --frozen python scripts/capture_g0_evidence.py --output "$(OUTPUT)" --checkout "$(CHECKOUT)"

verify-g0-repository:
	@test -n "$(BUNDLE)" || { echo "verify-g0-repository: BUNDLE=<path> is required" >&2; exit 2; }
	@test -n "$(CHECKOUT)" || { echo "verify-g0-repository: CHECKOUT=<path> is required" >&2; exit 2; }
	UV_OFFLINE=true $(UV) run --frozen python scripts/verify_g0_repository.py --bundle "$(BUNDLE)" --checkout "$(CHECKOUT)"

verify-g0-result:
	UV_OFFLINE=true $(UV) run --frozen python scripts/validate_schemas.py --bundle "$(G0_RESULT)"
	UV_OFFLINE=true $(UV) run --frozen python scripts/verify_g0_repository.py --bundle "$(G0_RESULT)" --checkout .
