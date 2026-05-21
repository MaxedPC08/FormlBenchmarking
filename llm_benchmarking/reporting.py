from html import escape
from pathlib import Path
import re
from typing import Any

from llm_benchmarking.cases import CaseResult


def write_html_report(
    path: str | Path,
    results: list[CaseResult],
    summary: dict[str, Any],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    html = _render_report(results, summary)
    target.write_text(html, encoding="utf-8")


def _render_report(results: list[CaseResult], summary: dict[str, Any]) -> str:
    title = "LLM Benchmark Report"
    rows = _flatten_summary(summary)
    body = "\n".join(
        [
            _summary_cards(results, rows),
            _bar_chart("Mean Score", rows, "mean_score", "{:.2f}"),
            _bar_chart("Pass Rate", rows, "pass_rate", "{:.0%}"),
            _bar_chart("Cases Run", rows, "cases", "{:.0f}"),
            _bar_chart("Input Tokens", rows, "input_tokens", "{:.0f}"),
            _bar_chart("Output Tokens", rows, "output_tokens", "{:.0f}"),
            _case_table(results),
        ]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --ink: #171b21;
      --muted: #667085;
      --line: #d9dee7;
      --accent: #0f766e;
      --accent-2: #7c3aed;
      --warn: #b42318;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      width: min(1180px, calc(100vw - 32px));
      margin: 28px auto 48px;
    }}
    header {{
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 24px;
      margin-bottom: 20px;
    }}
    h1, h2 {{
      margin: 0;
      letter-spacing: 0;
    }}
    h1 {{ font-size: 28px; line-height: 1.1; }}
    h2 {{ font-size: 17px; margin-bottom: 14px; }}
    .muted {{ color: var(--muted); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .card, section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }}
    .card {{ padding: 14px 16px; }}
    .metric {{ font-size: 24px; font-weight: 700; margin-top: 5px; }}
    section {{ padding: 16px; margin: 14px 0; overflow: hidden; }}
    svg {{ display: block; width: 100%; height: auto; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }}
    th, td {{
      border-top: 1px solid var(--line);
      padding: 10px 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
      text-transform: uppercase;
    }}
    td pre {{
      white-space: pre-wrap;
      word-break: break-word;
      margin: 0;
      max-height: 180px;
      overflow: auto;
      font: 12px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
    .status-pass {{ color: var(--accent); font-weight: 700; }}
    .status-fail {{ color: var(--warn); font-weight: 700; }}
    .status-skip {{ color: var(--muted); font-weight: 700; }}
    @media (max-width: 800px) {{
      header {{ display: block; }}
      .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      table {{ min-width: 860px; }}
      .table-wrap {{ overflow-x: auto; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>{title}</h1>
        <div class="muted">Generated from the latest benchmark run.</div>
      </div>
      <div class="muted">{len(results)} case result(s)</div>
    </header>
    {body}
  </main>
</body>
</html>
"""


def _flatten_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for model, benchmarks in summary.items():
        for benchmark, metrics in benchmarks.items():
            rows.append(
                {
                    "model": model,
                    "benchmark": benchmark,
                    "label": f"{model} / {benchmark}",
                    "cases": float(metrics["cases"]),
                    "attempted_cases": _metric_float(metrics.get("attempted_cases")),
                    "skipped_cases": _metric_float(metrics.get("skipped_cases")),
                    "mean_score": float(metrics["mean_score"]),
                    "pass_rate": float(metrics["pass_rate"]),
                    "input_tokens": _metric_float(metrics.get("input_tokens")),
                    "output_tokens": _metric_float(metrics.get("output_tokens")),
                    "total_tokens": _metric_float(metrics.get("total_tokens")),
                    "avg_input_tokens": _metric_float(metrics.get("avg_input_tokens")),
                    "avg_output_tokens": _metric_float(metrics.get("avg_output_tokens")),
                    "avg_total_tokens": _metric_float(metrics.get("avg_total_tokens")),
                }
            )
    return sorted(rows, key=lambda row: (row["model"], row["benchmark"]))


def _summary_cards(results: list[CaseResult], rows: list[dict[str, Any]]) -> str:
    model_count = len({result.model for result in results})
    benchmark_count = len({result.benchmark for result in results})
    attempted = [result for result in results if not result.metadata.get("skipped")]
    denominator = attempted or results
    mean_score = sum(result.score for result in denominator) / len(denominator) if denominator else 0.0
    pass_rate = sum(result.passed for result in denominator) / len(denominator) if denominator else 0.0
    input_tokens = _sum_result_tokens(attempted, "input_tokens")
    output_tokens = _sum_result_tokens(attempted, "output_tokens")
    total_tokens = _sum_result_tokens(attempted, "total_tokens")
    skipped_count = len(results) - len(attempted)
    cards = [
        ("Models", str(model_count)),
        ("Benchmarks", str(benchmark_count)),
        ("Attempted Cases", _format_number(len(attempted))),
        ("Skipped Cases", _format_number(skipped_count)),
        ("Overall Mean", f"{mean_score:.2f}"),
        ("Overall Pass Rate", f"{pass_rate:.0%}"),
        ("Input Tokens", _format_number(input_tokens)),
        ("Output Tokens", _format_number(output_tokens)),
        ("Total Tokens", _format_number(total_tokens)),
        ("Avg Tokens / Case", _format_number(_avg_total_tokens(results))),
    ]
    items = "\n".join(
        f'<div class="card"><div class="muted">{label}</div><div class="metric">{value}</div></div>'
        for label, value in cards
    )
    return f'<div class="grid">{items}</div>'


def _bar_chart(
    title: str,
    rows: list[dict[str, Any]],
    metric: str,
    formatter: str,
) -> str:
    if not rows:
        return f"<section><h2>{escape(title)}</h2><div class=\"muted\">No data.</div></section>"

    label_width = 260
    value_width = 70
    chart_width = 760
    row_height = 34
    top = 18
    height = top + row_height * len(rows) + 10
    values = [row.get(metric) for row in rows if row.get(metric) is not None]
    max_value = max(values) if values else 1.0
    if metric in {"mean_score", "pass_rate"}:
        max_value = max(1.0, max_value)

    bars = []
    for index, row in enumerate(rows):
        y = top + index * row_height
        value = row.get(metric)
        width = int(chart_width * ((value or 0.0) / max_value))
        fill = "#0f766e" if index % 2 == 0 else "#7c3aed"
        display_value = formatter.format(value) if value is not None else "n/a"
        bars.append(
            f"""
      <text x="0" y="{y + 18}" fill="#171b21" font-size="12">{escape(row["label"])}</text>
      <rect x="{label_width}" y="{y}" width="{chart_width}" height="22" rx="4" fill="#eef2f6"></rect>
      <rect x="{label_width}" y="{y}" width="{width}" height="22" rx="4" fill="{fill}"></rect>
      <text x="{label_width + chart_width + 12}" y="{y + 16}" fill="#171b21" font-size="12" font-weight="700">{display_value}</text>
"""
        )

    width = label_width + chart_width + value_width
    return f"""
<section>
  <h2>{escape(title)}</h2>
  <svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)} chart">
    {"".join(bars)}
  </svg>
</section>
"""


def _case_table(results: list[CaseResult]) -> str:
    if not results:
        return '<section><h2>Case Details</h2><div class="muted">No case results.</div></section>'

    rows = []
    for result in sorted(results, key=lambda item: (item.model, item.benchmark, item.case_id)):
        if result.metadata.get("skipped"):
            status_class = "status-skip"
            status = "skip"
        else:
            status_class = "status-pass" if result.passed else "status-fail"
            status = "pass" if result.passed else "fail"
        display_prediction = _display_prediction(result)
        input_tokens = _metadata_int(result, "input_tokens")
        output_tokens = _metadata_int(result, "output_tokens")
        total_tokens = _metadata_int(result, "total_tokens")
        rows.append(
            f"""
      <tr>
        <td>{escape(result.model)}</td>
        <td>{escape(result.benchmark)}</td>
        <td>{escape(result.case_id)}</td>
        <td>{result.score:.3f}</td>
        <td class="{status_class}">{status}</td>
        <td>{_format_number(input_tokens)}</td>
        <td>{_format_number(output_tokens)}</td>
        <td>{_format_number(total_tokens)}</td>
        <td><pre>{escape(display_prediction)}</pre></td>
      </tr>
"""
        )

    return f"""
<section>
  <h2>Case Details</h2>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th style="width: 150px;">Model</th>
          <th style="width: 140px;">Benchmark</th>
          <th style="width: 180px;">Case</th>
          <th style="width: 80px;">Score</th>
          <th style="width: 80px;">Status</th>
          <th style="width: 95px;">Input Tok</th>
          <th style="width: 95px;">Output Tok</th>
          <th style="width: 95px;">Total Tok</th>
          <th>Output</th>
        </tr>
      </thead>
      <tbody>
        {"".join(rows)}
      </tbody>
    </table>
  </div>
</section>
"""


def _metric_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metadata_int(result: CaseResult, key: str) -> int | None:
    value = result.metadata.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sum_result_tokens(results: list[CaseResult], key: str) -> int | None:
    values = [_metadata_int(result, key) for result in results]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values)


def _avg_total_tokens(results: list[CaseResult]) -> float | None:
    values = [_metadata_int(result, "total_tokens") for result in results]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _format_number(value: int | float | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and not value.is_integer():
        return f"{value:,.1f}"
    return f"{int(value):,}"


def _display_prediction(result: CaseResult) -> str:
    metadata = result.metadata or {}
    if metadata.get("skipped"):
        return (
            "Skipped: input too long\n"
            f"Estimated input tokens: {_format_number(_metadata_int(result, 'estimated_input_tokens'))}\n"
            f"Available input tokens: {_format_number(_metadata_int(result, 'available_input_tokens'))}"
        )
    if metadata.get("error"):
        return f"Provider error:\n{metadata['error']}"

    predicted_choice = metadata.get("predicted_choice")
    if predicted_choice:
        return str(predicted_choice)

    if result.benchmark in {"longbench_v2", "truthfulqa"}:
        return _no_parseable_output(result)

    cleaned = _strip_thinking(result.prediction)

    if result.benchmark == "humaneval":
        code = _extract_code(cleaned)
        return code or _no_parseable_output(result)

    answers = metadata.get("answers")
    if isinstance(answers, list):
        for answer in answers:
            answer_text = str(answer).strip()
            if answer_text and answer_text.lower() in cleaned.lower():
                return answer_text

    final = _extract_final_answer(cleaned)
    return final or _no_parseable_output(result)


def _strip_thinking(prediction: str) -> str:
    text = prediction.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<analysis>.*?</analysis>", "", text, flags=re.IGNORECASE | re.DOTALL)

    for marker in ("</think>", "</analysis>"):
        if marker in text.lower():
            index = text.lower().rfind(marker)
            text = text[index + len(marker) :]

    return text.strip()


def _extract_code(text: str) -> str:
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    lines = text.splitlines()
    code_lines = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("def ", "class ", "return ", "if ", "for ", "while ")):
            in_code = True
        if in_code:
            if stripped and not _looks_like_explanation(stripped):
                code_lines.append(line.rstrip())
    return "\n".join(code_lines).strip()


def _extract_final_answer(text: str) -> str:
    markers = [
        "final answer:",
        "answer:",
        "the answer is",
        "therefore,",
        "so,",
    ]
    lower = text.lower()
    for marker in markers:
        index = lower.rfind(marker)
        if index != -1:
            candidate = text[index + len(marker) :].strip()
            return _first_nonempty_line(candidate)
    return _first_nonempty_line(text)


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _looks_like_explanation(line: str) -> bool:
    explanation_starts = (
        "approach",
        "example",
        "however",
        "note:",
        "steps:",
        "we ",
        "why?",
    )
    return line.lower().startswith(explanation_starts)


def _no_parseable_output(result: CaseResult) -> str:
    detail = result.metadata.get("detail") if result.metadata else None
    if detail:
        return f"No parseable final output.\n\nScorer detail:\n{detail}"
    return "No parseable final output."
