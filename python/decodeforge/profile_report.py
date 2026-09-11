"""Readable Markdown for a freshly validated diagnostic analysis result.

This renderer is not a verifier for independently supplied saved reports.
Its caller must run the analyzer first. It deliberately omits prompts, command
lines, environment strings, and local file paths from the shareable document.
"""

from __future__ import annotations

import html
import re
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any, Final

_MODES: Final = (
    ("hybrid_native", "Hybrid native"),
    ("same_q8_reference", "Same-Q8 reference"),
)
_BUCKET_LABELS: Final = {
    "input_preparation": "Input preparation",
    "output_validation": "Output validation",
    "token_selection": "Token selection",
    "bookkeeping": "Token bookkeeping",
    "step_remainder": "Step remainder",
    "model_remainder": "Other model work + observer overhead",
    "qproj_remainder": "Query-projection remainder",
    "adapter_storage_guard": "Adapter storage checks",
    "fallback_remainder": "Fallback remainder",
    "fallback_clone": "Fallback weight cloning",
    "fallback_hash": "Fallback weight hashing",
    "fallback_linear": "Fallback linear operation",
    "native_operator_remainder": "Native operator remainder",
    "guarded_binding_run": "Guarded native binding",
    "token_embedding": "Token embedding",
    "layer_norm": "Layer norms",
    "attention_k_proj": "Attention key projections",
    "attention_v_proj": "Attention value projections",
    "attention_o_proj": "Attention output projections",
    "mlp": "Feed-forward blocks (MLPs)",
    "lm_head": "Language-model head",
}


def _markdown(value: Any) -> str:
    """Render metadata as one literal text line, including inside table cells."""
    text = " ".join(str(value).split())
    escaped = html.escape(text, quote=False)
    return re.sub(r"([\\`*_{}\[\]()#+.!|\-])", r"\\\1", escaped)


def _median(values: Sequence[Fraction]) -> Fraction:
    # The analyzer supplies exactly three sessions. Retain exact arithmetic for
    # table ordering; display rounding must never manufacture a winning bucket.
    return sorted(values)[1]


def _ms(nanoseconds: int | Fraction) -> str:
    value = Fraction(nanoseconds) / 1_000_000
    if 0 < value < Fraction(1, 1_000):
        return "<0.001"
    return f"{float(value):.3f}"


def _share(amount: int, total: int) -> str:
    value = Fraction(amount, total) * 100
    if 0 < value < Fraction(1, 100):
        return "<0.01%"
    return f"{float(value):.2f}%"


def _names(names: Sequence[str]) -> str:
    return " = ".join(_BUCKET_LABELS[name] for name in names)


def _stability(
    state: Mapping[str, Any], sessions: Sequence[Mapping[str, Any]], mode: str
) -> str:
    if state["top_group_stable"]:
        group = state["stable_top_group"]
        if len(group) == 1:
            top = f"Largest observed cost: {_names(group)} in all three sessions."
        else:
            top = (
                f"Largest observed cost: exact tie between {_names(group)} "
                "in all three sessions; no single leader."
            )
    else:
        observed = "; ".join(
            f"s{session['session_index']}: "
            f"{_names(session['paths'][mode]['cached_decode']['ranking_groups'][0])}"
            for session in sessions
        )
        top = f"The top group changed across sessions ({observed}); no stable leader."
    full = "stable" if state["ordering_stable"] else "unstable"
    return f"{top} Full category ordering: {full} across all three sessions."


def _ties(sessions: Sequence[Mapping[str, Any]], mode: str) -> str:
    groups: dict[tuple[str, ...], list[int]] = {}
    all_zero: dict[tuple[str, ...], bool] = {}
    omitted = _omitted(sessions, mode)
    for session in sessions:
        phase = session["paths"][mode]["cached_decode"]
        for group in phase["ranking_groups"]:
            group = [name for name in group if name not in omitted]
            if len(group) < 2:
                continue
            key = tuple(sorted(group))
            groups.setdefault(key, []).append(session["session_index"])
            all_zero[key] = all_zero.get(key, True) and all(
                phase["buckets_ns"][name] == 0 for name in key
            )
    if not groups:
        return "No exact ties were recorded among the displayed categories."
    descriptions = []
    for group, indexes in sorted(groups.items()):
        scope = (
            "all sessions"
            if len(indexes) == 3
            else ", ".join(f"s{index}" for index in indexes)
        )
        zero = " (zero recorded time)" if all_zero[group] else ""
        descriptions.append(f"{scope}: {_names(group)}{zero}")
    return "Exact ties: " + "; ".join(descriptions) + "."


def _omitted(sessions: Sequence[Mapping[str, Any]], mode: str) -> set[str]:
    return {
        name
        for name in _BUCKET_LABELS
        if all(
            session["paths"][mode]["cached_decode"]["buckets_ns"][name] == 0
            for session in sessions
        )
    }


def _cost_table(sessions: Sequence[Mapping[str, Any]], mode: str) -> list[str]:
    phases = [session["paths"][mode]["cached_decode"] for session in sessions]
    omitted = _omitted(sessions, mode)
    medians = {
        name: _median(
            [Fraction(phase["buckets_ns"][name], phase["steps"]) for phase in phases]
        )
        for name in _BUCKET_LABELS
        if name not in omitted
    }
    rows = [
        "| Cost | Median ms/cached step | Share s0 | Share s1 | Share s2 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in sorted(medians, key=lambda key: (-medians[key], _BUCKET_LABELS[key])):
        shares = " | ".join(
            _share(phase["buckets_ns"][name], phase["total_ns"]) for phase in phases
        )
        rows.append(f"| {_BUCKET_LABELS[name]} | {_ms(medians[name])} | {shares} |")
    return rows


def render_profile_report(report: Mapping[str, Any]) -> str:
    """Render fresh analyzer output, optionally with CLI-added capture digests.

    Statistics use the three observed session averages, not pooled token
    latencies. Missing input digests are disclosed when called directly on the
    analyzer API result. This function never prints prompts or local paths.
    """
    sessions = sorted(report["sessions"], key=lambda session: session["session_index"])
    lines = [
        "# Decode profile report",
        "",
        "Three diagnostic sessions show where instrumented cached decoding spent time. "
        "These observations do not establish a speedup, a completed optimization, "
        "or completion of P0.",
        "",
        "A cached step generates one token after the prompt is processed (prefill). "
        "Hybrid native uses the native query-projection operation where eligible. "
        "Same-Q8 reference uses the reference operation with the same "
        "quantized weights.",
        "",
        "The tables show nonoverlapping costs; categories with zero recorded time "
        "in all three sessions are omitted. The JSON keeps all 21 categories. "
        "Median ms/cached step "
        "is the median of the three session averages, not a median of individual "
        "token latencies. Shares partition each session's cached-step time; separately "
        "computed medians need not add to the median total. Display rounding does "
        "not establish an exact tie. Tables are sorted by median cost; stability "
        "statements compare each session's exact ranking.",
    ]
    for mode, label in _MODES:
        phases = [session["paths"][mode]["cached_decode"] for session in sessions]
        total = _median(
            [Fraction(phase["total_ns"], phase["steps"]) for phase in phases]
        )
        steps = " / ".join(str(phase["steps"]) for phase in phases)
        lines.extend(
            [
                "",
                f"## {label}",
                "",
                _stability(report["stability"][mode], sessions, mode),
                "",
                f"Median cached-step total: {_ms(total)} ms. "
                f"Observed cached steps (s0 / s1 / s2): {steps}.",
                "",
                *_cost_table(sessions, mode),
                "",
                _ties(sessions, mode),
            ]
        )
        omitted = _omitted(sessions, mode)
        if omitted:
            lines.extend(
                [
                    "",
                    f"Omitted {len(omitted)} categories with zero recorded time "
                    "in all three sessions; remaining shares are not renormalized.",
                ]
            )
    lines.extend(
        [
            "",
            "Remainders include observer overhead and work outside their measured "
            "children. Query-projection remainder includes the outer native "
            "eligibility guard; other model work includes unselected transformer "
            "operations. Native operator remainder excludes the nested binding call. "
            "Guarded native binding includes locks, pointer checks, the ABI call, "
            "and status handling; "
            "neither native category is kernel-only time.",
            "",
            "## Prefill and observer context",
            "",
            "Prefill includes prompt processing and first-token selection. The table "
            "below uses medians of whole-session observations. Profile/control ratios "
            "are shown separately for s0 / s1 / s2 and compare whole calls, including "
            "profile setup and removal; they are not cached-only overhead "
            "or a correction.",
            "",
            "| Mode | Prefill ms | Control call ms | Profile call ms | "
            "Profile/control s0 / s1 / s2 |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    remainder = []
    for mode, label in _MODES:
        paths = [session["paths"][mode] for session in sessions]
        prefill = _ms(
            _median([Fraction(path["prefill"]["total_ns"]) for path in paths])
        )
        control = _ms(
            _median([Fraction(path["observer"]["control_ns"]) for path in paths])
        )
        profiled = _ms(
            _median([Fraction(path["observer"]["profile_ns"]) for path in paths])
        )
        ratios = " / ".join(f"{path['observer']['ratio']:.3f}" for path in paths)
        lines.append(f"| {label} | {prefill} | {control} | {profiled} | {ratios} |")
        value = _ms(
            _median([Fraction(path["generation_remainder_ns"]) for path in paths])
        )
        remainder.append(f"{label}: {value} ms")
    lines.extend(
        [
            "",
            "Median generation remainder outside prefill/cached-step spans — "
            + "; ".join(remainder)
            + ".",
            "",
            "## Reproducibility",
            "",
            "Producer revision: "
            + _markdown(report["matched_inputs"]["source"]["git_revision"])
            + ".",
        ]
    )
    captures = report.get("input_captures")
    if captures:
        lines.extend(["", "| Session | Input capture SHA-256 |", "| --- | --- |"])
        for capture in sorted(captures, key=lambda value: str(value["session_index"])):
            lines.append(
                f"| {_markdown(capture['session_index'])} | "
                f"{_markdown(capture['sha256'])} |"
            )
    else:
        lines.extend(["", "Input capture SHA-256 digests were not supplied."])
    lines.extend(
        [
            "",
            "Matching identities and distinct declared process IDs are consistency "
            "checks; they do not authenticate captures or prove independent execution.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = ["render_profile_report"]
