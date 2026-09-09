# Contributing to DecodeForge

DecodeForge welcomes focused contributions, bug reports, and questions. The
project is developed in small, independently testable changes. Its first target
is Apple Silicon; portable workspace checks also run on Linux.

Before changing numerical behavior, ABI surfaces, benchmark claims, or retained
evidence, read the [technical design](docs/DESIGN.md) and the relevant contract:
[Q8 format](docs/Q8_FORMAT_V1.md),
[benchmark methodology](docs/BENCHMARKS.md), or
[model evaluation](docs/EVALUATION_V1.md). For a first contribution,
documentation, diagnostics, rejection tests, and small reproducibility fixes
are safer entry points than expanding compiler scope.

## Prerequisites

- macOS ARM64 or Linux x86-64;
- Clang, Git, and Make;
- `rustup`;
- `uv` 0.12.5.

On macOS, install `rustup` with Homebrew and install the repository's exact `uv`
version with its versioned installer:

```sh
brew install rustup
curl -LsSf https://astral.sh/uv/0.12.5/install.sh | sh
export PATH="$(brew --prefix rustup)/bin:$PATH"
```

The Homebrew `uv` formula can move ahead of the pinned version; `make setup`
checks for `uv` 0.12.5 so the lockfile is interpreted consistently. Modifying a
shell profile for the `rustup` path is optional.

## Clean setup and checks

```sh
make setup
make format
make check
make check-pytorch-pin
git diff --check
```

`make setup` installs the pinned Rust and Python versions and synchronizes the
checked-in locks. Once caches are populated, the portable checks must also run
without network access:

```sh
CARGO_NET_OFFLINE=true UV_OFFLINE=true make setup
CARGO_NET_OFFLINE=true UV_OFFLINE=true make check
```

`make check` intentionally contains no performance threshold. The separate
PyTorch command only verifies that the pinned CPU wheel imports; it does not
claim framework integration.

Useful focused commands include:

```sh
uv run --frozen python scripts/check_docs.py
uv run --frozen python -m pytest -q python/tests/test_q8.py
rustup run 1.98.0 cargo test --locked -p decodeforge-core
make verify-g0-result
make verify-g1-result
make verify-g3-result
make verify-evaluation-result
make test-bridge-cdylib
```

The verification targets recompute or validate checked-in evidence; they do not
silently rerun the model or convert CI into performance evidence. Run plain
`make` at any time to see the short command guide.

## Fresh macOS Rust loader readiness

Before a real G3 capture, run `make check-rust-toolchain`. The pinned Rust macOS
distribution can omit the library link expected by `rust-objcopy`, producing
`@rpath/libLLVM.dylib` loader errors. Successful Rust compilation alone does not
prove this preflight passed; stripping failures may only be warnings.

`make repair-rust-toolchain` inspects the pinned installation without changing
it. If it identifies the known missing link, explicitly opt in with
`make repair-rust-toolchain-apply`. The repair only creates the guarded missing
link to that same toolchain's LLVM library; it refuses unexpected existing
entries and reruns the normal loader check. No `DYLD_*` workaround, compiler pin
change, or stripping bypass is applied. Ordinary `make setup` does not perform
this repair. Hosted macOS CI opts in explicitly on its disposable runner, then
runs the strict preflight before the foundation tests and again offline.

## Results and generated data

Commit small correctness fixtures, generated source, assembly, and raw samples
needed to reproduce published claims. Do not commit model checkpoints, scratch
benchmark output, secrets, hostnames, usernames, serial numbers, or absolute
developer paths.

Every performance change must name the baseline and timed boundary, run
correctness before timing, retain unfavorable observations, and state whether
it changes kernel-only or integrated-model behavior. Use the pull request
template to record validation, claim boundaries, and limitations.

## License

DecodeForge's original code is licensed under [Apache 2.0](LICENSE), selected
explicitly by the maintainer. Contributions are governed by its contribution
terms. Preserve existing third-party copyright, attribution, and license notices;
dependencies and model artifacts retain their own licenses.
