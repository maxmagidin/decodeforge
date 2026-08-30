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

## Results and generated data

Commit small correctness fixtures, generated source, assembly, and raw samples
needed to reproduce published claims. Do not commit model checkpoints, scratch
benchmark output, secrets, hostnames, usernames, serial numbers, or absolute
developer paths.

The repository does not yet declare a license. Contributors must not infer or
add one without an explicit maintainer decision.
