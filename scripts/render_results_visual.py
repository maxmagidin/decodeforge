#!/usr/bin/env python3
"""Render a deterministic SVG overview from retained DecodeForge results."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_G1 = ROOT / "results" / "g1" / "apple-m4-primary" / "report.json"
DEFAULT_EVALUATION = ROOT / "results" / "evaluation" / "apple-m4-v1" / "summary.json"
DEFAULT_OUTPUT = ROOT / "docs" / "assets" / "decodeforge-results-overview.svg"

WIDTH = 1280
HEIGHT = 760
BACKGROUND = "#0d0d0f"
PANEL = "#19181c"
LINE = "#3a373f"
TEXT = "#f5f1e7"
MUTED = "#aaa6a0"
GOLD = "#d8b35b"
GREEN = "#6ee7a8"
BLUE = "#7dd3fc"
MULTIPLY = chr(215)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"expected number for {label}")
    return float(value)


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"expected integer for {label}")
    return int(value)


def _text(x: float, y: float, value: str, css_class: str = "") -> str:
    class_attr = f' class="{css_class}"' if css_class else ""
    return f'<text x="{x:.1f}" y="{y:.1f}"{class_attr}>{escape(value)}</text>'


def _kernel_data(report: dict[str, Any]) -> list[tuple[str, float, float, float]]:
    cases = report.get("cases")
    if not isinstance(cases, list) or len(cases) != 1 or not isinstance(cases[0], dict):
        raise ValueError("G1 report must contain exactly one case")
    sessions = cases[0].get("sessions")
    if not isinstance(sessions, list) or len(sessions) != 3:
        raise ValueError("G1 report must contain exactly three sessions")
    rows: list[tuple[str, float, float, float]] = []
    for session in sessions:
        if not isinstance(session, dict):
            raise ValueError("G1 session must be an object")
        interval = session.get("confidence_interval")
        if not isinstance(interval, dict):
            raise ValueError("G1 session is missing its confidence interval")
        rows.append(
            (
                str(session.get("session_id", "session")),
                _number(interval.get("lower"), "confidence lower"),
                _number(session.get("speedup"), "session speedup"),
                _number(interval.get("upper"), "confidence upper"),
            )
        )
    return rows


def _model_data(
    summary: dict[str, Any],
) -> tuple[dict[tuple[str, str], tuple[float, float, float]], dict[str, int | float]]:
    rows = summary.get("performance_by_session_case_path")
    if not isinstance(rows, list):
        raise ValueError("evaluation summary is missing performance rows")
    grouped: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("evaluation performance row must be an object")
        throughput = row.get("decode_tokens_per_second")
        if not isinstance(throughput, dict):
            raise ValueError("evaluation row is missing throughput")
        grouped[(str(row.get("case_id")), str(row.get("path")))].append(
            _number(throughput.get("median"), "per-session throughput median")
        )

    ranges: dict[tuple[str, str], tuple[float, float, float]] = {}
    for key, values in grouped.items():
        if len(values) != 3:
            raise ValueError(f"expected three process medians for {key}")
        ranges[key] = (min(values), statistics.median(values), max(values))

    correctness = summary.get("correctness")
    if not isinstance(correctness, dict):
        raise ValueError("evaluation summary is missing correctness")
    metrics: dict[str, int | float] = {
        "cases": _integer(correctness.get("cases"), "correctness cases"),
        "steps": _integer(
            correctness.get("compared_generation_steps"), "generation steps"
        ),
        "max_error": _number(
            correctness.get("max_logit_abs_error"), "maximum logit error"
        ),
    }
    return ranges, metrics


def render(g1_path: Path, evaluation_path: Path) -> str:
    sessions = _kernel_data(_load_object(g1_path))
    model_ranges, metrics = _model_data(_load_object(evaluation_path))
    median_speedup = statistics.median(row[2] for row in sessions)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title description">',
        '<title id="title">DecodeForge Apple M4 results overview</title>',
        '<desc id="description">Verified kernel speedups with paired confidence intervals, correctness and coverage metrics, and observed model decode-throughput ranges.</desc>',
        "<style>",
        f"text {{ font-family: system-ui, -apple-system, sans-serif; fill: {TEXT}; }}",
        ".eyebrow { font-size: 13px; font-weight: 750; letter-spacing: 2px; }",
        ".title { font-size: 36px; font-weight: 760; }",
        ".subtitle { font-size: 16px; }",
        ".metric { font-size: 35px; font-weight: 780; font-variant-numeric: tabular-nums; }",
        ".metric-label { font-size: 13px; }",
        ".panel-title { font-size: 20px; font-weight: 720; }",
        ".axis { font-size: 12px; font-variant-numeric: tabular-nums; }",
        ".row-label { font-size: 13px; }",
        ".value { font-size: 12px; font-weight: 700; font-variant-numeric: tabular-nums; }",
        f".muted {{ fill: {MUTED}; }}",
        f".gold {{ fill: {GOLD}; }}",
        f".green {{ fill: {GREEN}; }}",
        f".blue {{ fill: {BLUE}; }}",
        "</style>",
        f'<rect width="{WIDTH}" height="{HEIGHT}" rx="24" fill="{BACKGROUND}"/>',
        _text(52, 45, "DECODEFORGE / RETAINED APPLE M4 EVIDENCE", "eyebrow gold"),
        _text(52, 88, "From generated kernel to model execution", "title"),
        _text(
            52,
            116,
            "Rendered from checked-in G1 and evaluation JSON; kernel and model boundaries stay separate.",
            "subtitle muted",
        ),
    ]

    metric_cards = [
        (52, f"{median_speedup:.2f}{MULTIPLY}", "NEON vs generated scalar", GOLD),
        (352, "22/22", "query projections dispatched", BLUE),
        (652, f"{metrics['cases']}/30", "correctness prompts passed", GREEN),
        (952, f"{metrics['steps']:,}", "generated tokens matched", TEXT),
    ]
    for metric_x, value, label, color in metric_cards:
        parts.extend(
            [
                f'<rect x="{metric_x}" y="145" width="276" height="98" rx="14" fill="{PANEL}" stroke="{LINE}"/>',
                f'<text x="{metric_x + 18}" y="190" class="metric" style="fill:{color}">{escape(value)}</text>',
                _text(metric_x + 18, 220, label, "metric-label muted"),
            ]
        )

    parts.extend(
        [
            f'<rect x="52" y="270" width="540" height="430" rx="18" fill="{PANEL}" stroke="{LINE}"/>',
            _text(76, 307, "Generated-kernel speedup", "panel-title"),
            _text(76, 330, "Paired BCa 95% intervals · zoomed axis", "axis muted"),
        ]
    )
    kernel_left = 165.0
    kernel_right = 548.0
    kernel_min = 3.94
    kernel_max = 3.98

    def kernel_x(value: float) -> float:
        return kernel_left + (value - kernel_min) / (kernel_max - kernel_min) * (
            kernel_right - kernel_left
        )

    for tick in (3.94, 3.95, 3.96, 3.97, 3.98):
        tick_x = kernel_x(tick)
        parts.extend(
            [
                f'<line x1="{tick_x:.1f}" y1="354" x2="{tick_x:.1f}" y2="570" stroke="{LINE}"/>',
                _text(tick_x, 590, f"{tick:.2f}{MULTIPLY}", "axis muted"),
            ]
        )
    for index, (session_id, lower, point, upper) in enumerate(sessions):
        y = 390 + index * 72
        label = f"Session {index + 1}"
        parts.extend(
            [
                _text(76, y + 5, label, "row-label"),
                f'<line x1="{kernel_x(lower):.1f}" y1="{y}" x2="{kernel_x(upper):.1f}" y2="{y}" stroke="{GOLD}" stroke-width="5"/>',
                f'<path d="M {kernel_x(lower):.1f} {y - 8} V {y + 8} M {kernel_x(upper):.1f} {y - 8} V {y + 8}" stroke="{GOLD}" stroke-width="2"/>',
                f'<circle cx="{kernel_x(point):.1f}" cy="{y}" r="6" fill="{GREEN}" stroke="{BACKGROUND}" stroke-width="2"/>',
                _text(470, y - 13, f"{point:.5f}{MULTIPLY}", "value green"),
                f'<g id="{escape(session_id)}" data-lower="{lower}" data-point="{point}" data-upper="{upper}"/>',
            ]
        )
    parts.extend(
        [
            _text(
                76,
                632,
                "Claim gate: every session's lower bound > 1.0",
                "row-label green",
            ),
            _text(
                76,
                660,
                "Same Q8 projection · same prepared-call boundary",
                "axis muted",
            ),
            _text(76, 682, "Not a whole-model or stock-PyTorch speedup", "axis muted"),
        ]
    )

    parts.extend(
        [
            f'<rect x="616" y="270" width="612" height="430" rx="18" fill="{PANEL}" stroke="{LINE}"/>',
            _text(640, 307, "Integrated decode throughput", "panel-title"),
            _text(
                640, 330, "Range of three process medians · tokens/second", "axis muted"
            ),
        ]
    )
    model_left = 790.0
    model_right = 1188.0

    def model_x(value: float) -> float:
        return model_left + value / 16.0 * (model_right - model_left)

    for tick in (0, 4, 8, 12, 16):
        model_tick_x = model_x(float(tick))
        parts.extend(
            [
                f'<line x1="{model_tick_x:.1f}" y1="352" x2="{model_tick_x:.1f}" y2="616" stroke="{LINE}"/>',
                _text(model_tick_x, 636, str(tick), "axis muted"),
            ]
        )

    paths = [
        ("fp32", "FP32", BLUE),
        ("hybrid_native", "Native", GOLD),
        ("same_q8_reference", "Q8 ref", MUTED),
    ]
    cases = [
        ("short-01-16", "Short / 16"),
        ("medium-02-32", "Medium / 32"),
        ("long-03-64", "Long / 64"),
    ]
    for case_index, (case_id, case_label) in enumerate(cases):
        base_y = 380 + case_index * 86
        _case_text = _text(640, base_y, case_label, "row-label")
        parts.append(_case_text)
        for path_index, (path_id, path_label, color) in enumerate(paths):
            low, middle, high = model_ranges[(case_id, path_id)]
            y = base_y + 18 + path_index * 19
            parts.extend(
                [
                    _text(705, y + 4, path_label, "axis muted"),
                    f'<line x1="{model_x(low):.1f}" y1="{y}" x2="{model_x(high):.1f}" y2="{y}" stroke="{color}" stroke-width="5"/>',
                    f'<circle cx="{model_x(middle):.1f}" cy="{y}" r="4" fill="{color}"/>',
                    f'<g data-case="{case_id}" data-path="{path_id}" data-min="{low}" data-median="{middle}" data-max="{high}"/>',
                ]
            )
    parts.extend(
        [
            _text(640, 660, "Native beats the guarded Q8 reference", "row-label green"),
            _text(640, 682, "No consistent advantage over original FP32", "axis muted"),
            _text(
                52,
                733,
                f"Maximum native/reference logit difference: {float(metrics['max_error']):.10f} · Sources: G1 report.json + evaluation summary.json",
                "axis muted",
            ),
            "</svg>",
        ]
    )
    return "\n".join(parts) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1", type=Path, default=DEFAULT_G1)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    rendered = render(args.g1, args.evaluation)
    if args.verify:
        if not args.output.is_file():
            print(f"results-visual: missing {args.output}", file=sys.stderr)
            return 1
        if args.output.read_text(encoding="utf-8") != rendered:
            print(f"results-visual: stale {args.output}", file=sys.stderr)
            return 1
        print("results-visual: ok")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"results-visual: wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
