# Contributing to DecodeForge

DecodeForge is developed in small, independently testable changes. The initial
product target is Apple Silicon; portable workspace checks also run on Linux so
target-independent code stays honest.

## Prerequisites

- macOS ARM64 or Linux x86-64;
- Clang, Git, and Make;
- `rustup`;
- `uv` 0.12.5.

On Homebrew, `brew install rustup uv` installs the required bootstrap tools.
Because the `rustup` formula is keg-only, add `$(brew --prefix rustup)/bin` to
the command's `PATH`; modifying a shell profile is optional.

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

## License

DecodeForge's original code is licensed under [Apache 2.0](LICENSE), selected
explicitly by the maintainer. Contributions are governed by its contribution
terms. Preserve existing third-party copyright, attribution, and license notices;
dependencies and model artifacts retain their own licenses.
