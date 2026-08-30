.PHONY: setup format lint test check check-pytorch-pin test-native

UV := uv
RUST_VERSION := 1.98.0
PYTHON_VERSION := 3.12.14
UV_VERSION := 0.12.5

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
	@cargo fetch --locked
	@echo "setup: ok (Rust $(RUST_VERSION), Python $(PYTHON_VERSION), uv $(UV_VERSION), Clang detected)"

format:
	cargo fmt --all
	$(UV) run --frozen ruff format python scripts

lint:
	cargo fmt --all --check
	cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
	RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked
	$(UV) lock --check
	$(UV) run --frozen ruff format --check python scripts
	$(UV) run --frozen ruff check python scripts
	$(UV) run --frozen mypy
	$(UV) run --frozen python scripts/check_workspace.py
	$(UV) run --frozen python scripts/check_headers.py

test:
	cargo build --workspace --all-features --locked
	cargo test --workspace --all-features --locked
	$(UV) run --frozen python -m pytest -q
	cargo run --quiet --locked -p decodeforge -- --version

check: lint test

check-pytorch-pin:
	$(UV) run --frozen --extra pytorch-cpu python -c 'import platform, torch; assert torch.__version__.split("+")[0] == "2.13.0"; print(f"pytorch-pin: ok (torch={torch.__version__}, host={platform.system()}:{platform.machine()})")'

test-native:
	$(UV) run --frozen python scripts/check_headers.py
