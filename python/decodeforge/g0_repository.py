"""Checked-in Git provenance checks for an already portable-verified G0 bundle."""

from __future__ import annotations

import os
import re
import selectors
import signal
import subprocess
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from decodeforge.g0_evidence import (
    G0_FILE_LIMITS,
    G0_PROVENANCE_BASELINE,
    RUN_MANIFEST_NAME,
    JsonObject,
    _verified_g0_bundle,
    _VerifiedG0Bundle,
)

_GIT_EXECUTABLE: Final = "git"
_GIT_GLOBAL_OPTIONS: Final[tuple[str, ...]] = (
    "--no-replace-objects",
    "--literal-pathspecs",
    "-c",
    "core.fsmonitor=false",
    "-c",
    "core.untrackedCache=false",
    "-c",
    "core.filemode=true",
    "-c",
    "core.symlinks=true",
    "-c",
    "core.checkStat=default",
    "-c",
    "core.ignoreStat=false",
    "-c",
    "core.trustctime=true",
)
_GIT_TIMEOUT_SECONDS: Final = 5.0
_GIT_OUTPUT_LIMIT: Final = 64 * 1024
_FIXTURE_MANIFEST_PATH: Final = "tests/fixtures/v1/manifest.json"
_FULL_REVISION: Final = re.compile(r"^[0-9a-f]{40}$")
_GIT_ENV: Final[dict[str, str]] = {
    "GIT_CONFIG_COUNT": "0",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_NO_LAZY_FETCH": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_PAGER": "cat",
    "GIT_TERMINAL_PROMPT": "0",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": os.defpath,
}


@dataclass(frozen=True)
class _GitResult:
    """One bounded, non-interactive Git invocation result."""

    returncode: int | None
    stdout: bytes
    failure: str | None


@dataclass(frozen=True)
class _GitCheckout:
    """One validated worktree root and its explicit Git directory."""

    work_tree: Path
    git_dir: Path


def _git_environment() -> dict[str, str]:
    """Return a minimal Git environment independent of the caller's shell."""

    return dict(_GIT_ENV)


def _terminate(process: subprocess.Popen[bytes]) -> None:
    """Stop and reap a bounded Git process group without exposing stderr."""

    # The direct process can exit after spawning a descendant which still owns
    # stdout.  Its session/process group remains addressable in that case, so
    # always try the group first rather than keying this on ``poll()``.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        if process.poll() is None:
            with suppress(OSError):
                process.kill()
    with suppress(OSError, subprocess.TimeoutExpired):
        process.wait(timeout=1)


def _run_git_process(
    cwd: Path, binding_options: tuple[str, ...], arguments: tuple[str, ...]
) -> _GitResult:
    """Run a fixed Git argv with bounded stdout and silent stderr."""

    try:
        process = subprocess.Popen(
            [
                _GIT_EXECUTABLE,
                *_GIT_GLOBAL_OPTIONS,
                *binding_options,
                *arguments,
            ],
            cwd=cwd,
            env=_git_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            start_new_session=True,
        )
    except OSError:
        return _GitResult(None, b"", "execution")

    assert process.stdout is not None
    assert process.stderr is not None
    selector = selectors.DefaultSelector()
    output: list[bytes] = []
    output_size = 0
    deadline = time.monotonic() + _GIT_TIMEOUT_SECONDS
    stdout_fd = process.stdout.fileno()
    stderr_fd = process.stderr.fileno()
    try:
        selector.register(stdout_fd, selectors.EVENT_READ, "stdout")
        selector.register(stderr_fd, selectors.EVENT_READ, "stderr")
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _terminate(process)
                return _GitResult(None, b"", "timeout")
            events = selector.select(remaining)
            if not events:
                # If the direct process already exited but a descendant still
                # holds stdout, the descriptor remains registered without an
                # EOF event.  That is still a timed-out process group, not a
                # successful Git invocation.
                _terminate(process)
                return _GitResult(None, b"", "timeout")
            for key, _ in events:
                if key.data == "stderr":
                    # A fixed Git query should be silent. Read one byte only
                    # to drain the readiness event, then fail closed without
                    # retaining or exposing the warning/error text.
                    if os.read(stderr_fd, 1):
                        _terminate(process)
                        return _GitResult(None, b"", "stderr")
                    selector.unregister(stderr_fd)
                    continue
                chunk = os.read(
                    stdout_fd,
                    min(8192, _GIT_OUTPUT_LIMIT + 1 - output_size),
                )
                if not chunk:
                    selector.unregister(stdout_fd)
                    continue
                output.append(chunk)
                output_size += len(chunk)
                if output_size > _GIT_OUTPUT_LIMIT:
                    _terminate(process)
                    return _GitResult(None, b"", "output-limit")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _terminate(process)
            return _GitResult(None, b"", "timeout")
        return _GitResult(process.wait(timeout=remaining), b"".join(output), None)
    except subprocess.TimeoutExpired:
        _terminate(process)
        return _GitResult(None, b"", "timeout")
    except OSError:
        _terminate(process)
        return _GitResult(None, b"", "execution")
    finally:
        selector.close()
        process.stdout.close()
        process.stderr.close()
        if process.poll() is None:
            _terminate(process)


def _run_git_discovery(checkout: Path, arguments: tuple[str, ...]) -> _GitResult:
    """Run the one bootstrap query before an exact Git binding exists."""

    return _run_git_process(checkout, (), arguments)


def _run_git(checkout: _GitCheckout, arguments: tuple[str, ...]) -> _GitResult:
    """Run Git bound explicitly to the validated worktree and Git directory."""

    return _run_git_process(
        checkout.work_tree,
        (
            f"--git-dir={checkout.git_dir}",
            f"--work-tree={checkout.work_tree}",
        ),
        arguments,
    )


def _result_lines(result: _GitResult, expected_count: int) -> tuple[bytes, ...] | None:
    """Decode exactly the bounded newline-delimited bootstrap result shape."""

    if result.failure is not None or result.returncode != 0:
        return None
    if not result.stdout.endswith(b"\n"):
        return None
    lines = tuple(result.stdout[:-1].split(b"\n"))
    if len(lines) != expected_count or any(not line or b"\0" in line for line in lines):
        return None
    return lines


def _checkout_binding(checkout: Path) -> _GitCheckout | None:
    """Resolve one exact non-bare worktree before running provenance queries."""

    try:
        work_tree = checkout.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if not work_tree.is_dir():
        return None
    bootstrap = _run_git_discovery(
        work_tree,
        ("rev-parse", "--show-toplevel", "--absolute-git-dir", "--is-bare-repository"),
    )
    lines = _result_lines(bootstrap, 3)
    if lines is None or lines[2] != b"false":
        return None
    try:
        reported_root = Path(os.fsdecode(lines[0])).resolve(strict=True)
        git_dir = Path(os.fsdecode(lines[1])).resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if reported_root != work_tree or not git_dir.is_dir():
        return None
    resolved = _GitCheckout(work_tree=work_tree, git_dir=git_dir)
    core_worktree = _run_git(
        resolved,
        ("config", "--null", "--get-all", "core.worktree"),
    )
    if core_worktree.failure is not None:
        return None
    if core_worktree.returncode == 1 and core_worktree.stdout == b"":
        return resolved
    return None


def _repository_diagnostic(field: str, summary: str) -> JsonObject:
    """Return a schema-valid provenance diagnostic without stderr or paths."""

    return {
        "schema_version": 1,
        "code": "DFE-BUNDLE-009",
        "severity": "error",
        "component": "bundle",
        "summary": summary,
        "context": {"artifact": RUN_MANIFEST_NAME, "field": field},
    }


def _single_revision(result: _GitResult) -> str | None:
    """Decode exactly one full lowercase Git revision from bounded stdout."""

    if result.failure is not None or result.returncode != 0:
        return None
    try:
        text = result.stdout.decode("ascii")
    except UnicodeDecodeError:
        return None
    if not text.endswith("\n"):
        return None
    revision = text.removesuffix("\n")
    if _FULL_REVISION.fullmatch(revision) is None:
        return None
    return revision


def _is_commit(checkout: _GitCheckout, revision: str) -> bool | None:
    """Return commit/non-commit, or ``None`` when Git could not answer safely."""

    result = _run_git(
        checkout,
        (
            "rev-parse",
            "--verify",
            "--quiet",
            "--end-of-options",
            f"{revision}^{{commit}}",
        ),
    )
    if result.failure is not None:
        return None
    if result.returncode == 0:
        return _single_revision(result) == revision
    if result.returncode == 1 and result.stdout == b"":
        return False
    return None


def _is_ancestor(checkout: _GitCheckout, ancestor: str, descendant: str) -> bool | None:
    """Return Git ancestry, or ``None`` when the isolated command failed."""

    result = _run_git(
        checkout,
        ("merge-base", "--is-ancestor", ancestor, descendant),
    )
    if result.failure is not None or result.stdout:
        return None
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    return None


def _fixture_blob(checkout: _GitCheckout, revision: str) -> bytes | None:
    """Read one bounded fixture-manifest blob from the recorded commit only."""

    object_name = f"{revision}:{_FIXTURE_MANIFEST_PATH}"
    size_result = _run_git(checkout, ("cat-file", "-s", object_name))
    if size_result.failure is not None or size_result.returncode != 0:
        return None
    try:
        size_text = size_result.stdout.decode("ascii")
    except UnicodeDecodeError:
        return None
    if not size_text.endswith("\n") or not size_text[:-1].isdigit():
        return None
    expected_size = int(size_text[:-1])
    if expected_size > G0_FILE_LIMITS["fixture-manifest.json"]:
        return None
    blob_result = _run_git(checkout, ("cat-file", "blob", object_name))
    if blob_result.failure is not None or blob_result.returncode != 0:
        return None
    if len(blob_result.stdout) != expected_size:
        return None
    return blob_result.stdout


def _recorded_revision(verified: _VerifiedG0Bundle) -> str | None:
    """Read the already portable-validated project revision without paths."""

    project = verified.manifest.get("project")
    if not isinstance(project, dict):
        return None
    revision = project.get("revision")
    if type(revision) is not str or _FULL_REVISION.fullmatch(revision) is None:
        return None
    return revision


def _checkout_head(checkout: _GitCheckout) -> str | None:
    """Resolve the checkout HEAD only when it names one full commit."""

    return _single_revision(
        _run_git(checkout, ("rev-parse", "--verify", "HEAD^{commit}"))
    )


def _checkout_clean(checkout: _GitCheckout) -> bool | None:
    """Return clean/dirty, or ``None`` if hardened status could not run."""

    status = _run_git(
        checkout,
        (
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignored=no",
            "--ignore-submodules=none",
        ),
    )
    if status.failure is not None or status.returncode != 0:
        return None
    return not status.stdout


def _checkout_index_unflagged(checkout: _GitCheckout) -> bool | None:
    """Reject index entries that hide worktree checks from ``git status``."""

    result = _run_git(checkout, ("ls-files", "-v", "-z", "--full-name"))
    if result.failure is not None or result.returncode != 0:
        return None
    if not result.stdout:
        return True
    if not result.stdout.endswith(b"\0"):
        return None
    permitted_tags = {
        b"H",
        b"h",
        b"S",
        b"s",
        b"M",
        b"m",
        b"R",
        b"r",
        b"C",
        b"c",
        b"K",
        b"k",
        b"?",
    }
    for record in result.stdout[:-1].split(b"\0"):
        if len(record) < 3 or record[1:2] != b" " or not record[2:]:
            return None
        tag = record[:1]
        if tag not in permitted_tags:
            return None
        if tag.lower() == b"s" or tag.islower():
            return False
    return True


def _cleanliness_diagnostic(clean: bool | None) -> JsonObject:
    """Explain an unavailable or non-clean checkout without Git output."""

    if clean is None:
        summary = "The explicit checkout cleanliness could not be determined."
    else:
        summary = "The explicit checkout has staged, unstaged, or untracked changes."
    return _repository_diagnostic("repository.cleanliness", summary)


def _index_flags_diagnostic(unflagged: bool | None) -> JsonObject:
    """Explain unavailable or hiding index flags without exposing Git output."""

    if unflagged is None:
        summary = "The explicit checkout index flags could not be determined safely."
    else:
        summary = "The explicit checkout has assume-unchanged or skip-worktree entries."
    return _repository_diagnostic("repository.index_flags", summary)


def verify_g0_repository_bundle(
    bundle: Path, checkout: Path
) -> list[JsonObject] | None:
    """Verify Git provenance after the portable snapshot verifier has succeeded.

    This gate intentionally requires an explicit checkout. It never invokes a
    recorded reproduction argv, never consults a source-tree default, and uses
    only the immutable snapshots returned by the portable verifier for bundle
    contents. ``None`` preserves the non-G0 foundation dispatch signal.
    """

    verified, diagnostics = _verified_g0_bundle(bundle)
    if diagnostics is not None:
        return diagnostics
    if verified is None:
        return None
    bound_checkout = _checkout_binding(checkout)
    if bound_checkout is None:
        return [
            _repository_diagnostic(
                "repository.checkout",
                "The explicit checkout is not one exact non-bare Git worktree.",
            )
        ]
    revision = _recorded_revision(verified)
    if revision is None:
        return [
            _repository_diagnostic(
                "project.revision",
                "The portable G0 revision is unavailable for repository provenance.",
            )
        ]

    head = _checkout_head(bound_checkout)
    if head is None:
        return [
            _repository_diagnostic(
                "repository.head",
                "The explicit checkout does not expose one full commit HEAD.",
            )
        ]
    baseline_commit = _is_commit(bound_checkout, G0_PROVENANCE_BASELINE)
    if baseline_commit is None:
        return [
            _repository_diagnostic(
                "repository.baseline",
                "The G0 provenance baseline could not be determined safely.",
            )
        ]
    if not baseline_commit:
        return [
            _repository_diagnostic(
                "repository.baseline",
                "The G0 provenance baseline is not a commit in the checkout.",
            )
        ]
    revision_commit = _is_commit(bound_checkout, revision)
    if revision_commit is None:
        return [
            _repository_diagnostic(
                "project.revision",
                "The recorded G0 revision could not be determined safely.",
            )
        ]
    if not revision_commit:
        return [
            _repository_diagnostic(
                "project.revision",
                "The recorded G0 revision is not a commit in the checkout.",
            )
        ]
    head_commit = _is_commit(bound_checkout, head)
    if head_commit is None:
        return [
            _repository_diagnostic(
                "repository.head",
                "The explicit checkout HEAD could not be determined safely.",
            )
        ]
    if not head_commit:
        return [
            _repository_diagnostic(
                "repository.head",
                "The explicit checkout HEAD is not a commit object.",
            )
        ]
    baseline_ancestor = _is_ancestor(bound_checkout, G0_PROVENANCE_BASELINE, revision)
    if baseline_ancestor is None:
        return [
            _repository_diagnostic(
                "repository.ancestry.baseline",
                "The baseline ancestry could not be determined safely.",
            )
        ]
    if not baseline_ancestor:
        return [
            _repository_diagnostic(
                "repository.ancestry.baseline",
                "The recorded G0 revision does not descend from the baseline.",
            )
        ]
    revision_ancestor = _is_ancestor(bound_checkout, revision, head)
    if revision_ancestor is None:
        return [
            _repository_diagnostic(
                "repository.ancestry.head",
                "The checkout ancestry could not be determined safely.",
            )
        ]
    if not revision_ancestor:
        return [
            _repository_diagnostic(
                "repository.ancestry.head",
                "The checkout HEAD does not descend from the recorded G0 revision.",
            )
        ]

    initial_clean = _checkout_clean(bound_checkout)
    if initial_clean is not True:
        return [_cleanliness_diagnostic(initial_clean)]
    initial_index_unflagged = _checkout_index_unflagged(bound_checkout)
    if initial_index_unflagged is not True:
        return [_index_flags_diagnostic(initial_index_unflagged)]

    expected_fixture = _fixture_blob(bound_checkout, revision)
    if expected_fixture is None:
        return [
            _repository_diagnostic(
                "fixture-manifest.json.repository_blob",
                "The recorded fixture-manifest blob could not be read safely.",
            )
        ]
    if expected_fixture != verified.snapshots["fixture-manifest.json"].content:
        return [
            _repository_diagnostic(
                "fixture-manifest.json.repository_blob",
                "The copied fixture manifest does not match the recorded revision.",
            )
        ]

    # These are deliberately the final Git operations. Git cannot give this
    # read-only gate an atomic repository snapshot, so repeat the mutable
    # checkout observations after every ancestry/object/blob query. Checking
    # HEAD last rejects a branch movement after the final clean/index pass.
    final_clean = _checkout_clean(bound_checkout)
    if final_clean is not True:
        return [_cleanliness_diagnostic(final_clean)]
    final_index_unflagged = _checkout_index_unflagged(bound_checkout)
    if final_index_unflagged is not True:
        return [_index_flags_diagnostic(final_index_unflagged)]
    if _checkout_head(bound_checkout) != head:
        return [
            _repository_diagnostic(
                "repository.head",
                "The explicit checkout HEAD changed during provenance verification.",
            )
        ]
    return []
