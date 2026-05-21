#!/usr/bin/env python3
"""Create a static grouped bar chart from multiple benchmark JSONL runs."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_benchmarking.cases import CaseResult
from llm_benchmarking.charts import write_mean_score_chart


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one grouped bar chart from multiple run JSONL files. "
            "By default, rows with the same model name are merged."
        )
    )
    parser.add_argument("run_paths", nargs="+", type=Path, help="One or more results/run_*.jsonl files.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("charts/combined_runs_mean_scores.svg"),
        help="Output SVG path.",
    )
    parser.add_argument(
        "--title",
        default="Mean Score by Model and Benchmark",
        help="Chart title.",
    )
    parser.add_argument(
        "--split-by-run",
        action="store_true",
        help="Show the same model from different files as separate model groups.",
    )
    args = parser.parse_args()

    results = []
    for run_path in args.run_paths:
        run_label = _run_label(run_path)
        for result in _read_results_or_summary(run_path):
            if args.split_by_run:
                metadata = {
                    **result.metadata,
                    "chart_model_label": f"{result.model} ({run_label})",
                }
                result = replace(result, metadata=metadata)
            results.append(result)

    write_mean_score_chart(args.output, results, args.title)
    print(f"Wrote {args.output}")


def _read_results_or_summary(path: Path) -> list[CaseResult]:
    if not path.exists():
        raise SystemExit(f"Input file does not exist: {path}")
    if path.suffix.lower() == ".json":
        return _read_summary(path)
    if path.suffix.lower() != ".jsonl":
        raise SystemExit(
            f"Unsupported input file: {path}\n"
            "Expected a run_*.jsonl raw results file or summary_*.json summary file."
        )
    return _read_jsonl_results(path)


def _read_jsonl_results(path: Path) -> list[CaseResult]:
    results = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"Could not parse {path} as JSONL at line {line_number}: {exc}\n"
                    "If this is a summary file, pass the .json file. If this is a report "
                    "or chart file, use the matching run_*.jsonl instead."
                ) from exc
            results.append(CaseResult(**row))
    return results


def _read_summary(path: Path) -> list[CaseResult]:
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Could not parse summary JSON {path}: {exc}") from exc

    results = []
    for model, benchmarks in summary.items():
        if not isinstance(benchmarks, dict):
            raise SystemExit(f"Summary file has an unexpected shape: {path}")
        for benchmark, metrics in benchmarks.items():
            if not isinstance(metrics, dict) or "mean_score" not in metrics:
                raise SystemExit(f"Summary file has an unexpected shape: {path}")
            results.append(
                CaseResult(
                    benchmark=benchmark,
                    model=model,
                    case_id=f"{path.stem}:{benchmark}",
                    prompt="",
                    prediction="",
                    score=float(metrics["mean_score"]),
                    passed=float(metrics["mean_score"]) >= 1.0,
                    metadata={"source_summary": str(path)},
                )
            )
    return results


def _run_label(path: Path) -> str:
    stem = path.stem
    if stem.startswith("run_"):
        return stem.removeprefix("run_")
    return stem


if __name__ == "__main__":
    main()
