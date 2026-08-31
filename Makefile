.PHONY: setup format lint test check check-pytorch-pin test-native \
	validate-contracts verify-bundle fixture-check rust-fixture-check \
	capture-g0-evidence verify-g0-repository

UV := uv
RUST_VERSION := 1.98.0
PYTHON_VERSION := 3.12.14
UV_VERSION := 0.12.5
CARGO := PATH="$$(dirname "$$(rustup which --toolchain $(RUST_VERSION) cargo)"):$$PATH" cargo

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
	$(UV) run --frozen mypy
	$(UV) run --frozen python scripts/check_workspace.py
	$(UV) run --frozen python scripts/check_headers.py
	$(UV) run --frozen python scripts/validate_schemas.py --all

test:
	$(CARGO) build --workspace --all-features --locked
	$(CARGO) test --workspace --all-features --locked
	$(CARGO) test --workspace --all-features --locked --release
	$(UV) run --frozen python -m pytest -q
	$(UV) run --frozen python scripts/generate_q8_fixtures.py --check
	$(MAKE) rust-fixture-check
	$(CARGO) run --quiet --locked -p decodeforge -- --version

check: lint test

check-pytorch-pin:
	$(UV) run --frozen --extra pytorch-cpu python -c 'import platform, torch; assert torch.__version__.split("+")[0] == "2.13.0"; print(f"pytorch-pin: ok (torch={torch.__version__}, host={platform.system()}:{platform.machine()})")'

test-native:
	$(UV) run --frozen python scripts/check_headers.py

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
