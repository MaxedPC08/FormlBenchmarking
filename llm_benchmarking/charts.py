from collections import defaultdict
from pathlib import Path

from llm_benchmarking.cases import CaseResult


COLORS = [
    "#0f766e",
    "#7c3aed",
    "#b45309",
    "#2563eb",
    "#be123c",
    "#4d7c0f",
    "#9333ea",
    "#0e7490",
]


def write_mean_score_chart(
    path: str | Path,
    results: list[CaseResult],
    title: str = "Mean Score by Model and Benchmark",
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    chart = build_chart_data(results)
    target.write_text(render_svg(chart, title), encoding="utf-8")


def build_chart_data(results: list[CaseResult]) -> dict[str, object]:
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for result in results:
        if result.metadata.get("skipped"):
            continue
        model_label = str(result.metadata.get("chart_model_label") or result.model)
        buckets[model_label][result.benchmark].append(float(result.score))

    models = sorted(buckets)
    benchmarks = sorted(
        {benchmark for by_benchmark in buckets.values() for benchmark in by_benchmark}
    )
    scores = {
        model: {
            benchmark: _mean(buckets[model].get(benchmark, []))
            for benchmark in benchmarks
        }
        for model in models
    }
    return {"models": models, "benchmarks": benchmarks, "scores": scores}


def render_svg(chart: dict[str, object], title: str) -> str:
    models: list[str] = chart["models"]  # type: ignore[assignment]
    benchmarks: list[str] = chart["benchmarks"]  # type: ignore[assignment]
    scores: dict[str, dict[str, float | None]] = chart["scores"]  # type: ignore[assignment]

    margin_left = 72
    margin_right = 28
    margin_top = 92
    margin_bottom = 150
    plot_height = 360
    bar_width = 14
    bar_gap = 4
    group_gap = 36
    group_width = len(benchmarks) * bar_width + max(0, len(benchmarks) - 1) * bar_gap
    width = (
        margin_left
        + margin_right
        + len(models) * group_width
        + max(0, len(models) - 1) * group_gap
    )
    height = margin_top + plot_height + margin_bottom
    axis_bottom = margin_top + plot_height

    elements = [
        _text(width / 2, 34, title, size=22, weight=700, anchor="middle"),
        _text(
            width / 2,
            58,
            "Grouped bars show each benchmark's mean score within a model.",
            size=12,
            fill="#667085",
            anchor="middle",
        ),
    ]

    for tick in range(0, 6):
        value = tick / 5
        y = axis_bottom - value * plot_height
        elements.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        elements.append(
            _text(
                margin_left - 12,
                y + 4,
                f"{value:.1f}",
                size=11,
                fill="#667085",
                anchor="end",
            )
        )

    elements.append(
        f'<line x1="{margin_left}" y1="{axis_bottom}" x2="{width - margin_right}" y2="{axis_bottom}" stroke="#111827" stroke-width="1.2"/>'
    )
    elements.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{axis_bottom}" stroke="#111827" stroke-width="1.2"/>'
    )

    for model_index, model in enumerate(models):
        group_x = margin_left + model_index * (group_width + group_gap)
        for benchmark_index, benchmark in enumerate(benchmarks):
            score = scores[model].get(benchmark)
            if score is None:
                continue
            x = group_x + benchmark_index * (bar_width + bar_gap)
            bar_height = max(0.0, min(1.0, score)) * plot_height
            y = axis_bottom - bar_height
            color = COLORS[benchmark_index % len(COLORS)]
            elements.append(
                f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{color}" rx="2"/>'
            )
        label = _wrap_label(model, 18)
        label_x = group_x + group_width / 2
        for line_index, line in enumerate(label):
            elements.append(
                _text(label_x, axis_bottom + 24 + line_index * 14, line, size=11, anchor="middle")
            )

    legend_x = margin_left
    legend_y = height - 42
    for index, benchmark in enumerate(benchmarks):
        x = legend_x + (index % 4) * 190
        y = legend_y + (index // 4) * 20
        color = COLORS[index % len(COLORS)]
        elements.append(f'<rect x="{x}" y="{y - 10}" width="12" height="12" fill="{color}" rx="2"/>')
        elements.append(_text(x + 18, y, benchmark, size=12))

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{_escape(title)}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  {"\n  ".join(elements)}
</svg>
"""


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _wrap_label(label: str, max_chars: int) -> list[str]:
    parts = []
    for chunk in label.split("/"):
        if parts:
            parts[-1] = f"{parts[-1]}/"
        parts.extend(_split_model_chunk(chunk, max_chars))

    lines: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current} {part}".strip()
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = part
    if current:
        lines.append(current)
    return lines[:4]


def _split_model_chunk(chunk: str, max_chars: int) -> list[str]:
    if len(chunk) <= max_chars:
        return [chunk]
    parts = []
    current = ""
    for piece in chunk.split("-"):
        next_piece = piece if not current else f"{current}-{piece}"
        if len(next_piece) <= max_chars:
            current = next_piece
            continue
        if current:
            parts.append(current)
        current = piece
    if current:
        parts.append(current)
    return parts


def _text(
    x: float,
    y: float,
    text: str,
    size: int = 12,
    weight: int = 400,
    fill: str = "#111827",
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}">{_escape(text)}</text>'
    )


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
