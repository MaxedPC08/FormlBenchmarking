#!/usr/bin/env python3
"""Create a static grouped bar chart from a benchmark JSONL run."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_benchmarking.cases import CaseResult
from llm_benchmarking.charts import write_mean_score_chart


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a static grouped bar chart SVG from a run JSONL."
    )
    parser.add_argument("run_path", type=Path, help="Path to results/run_*.jsonl")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output SVG path. Defaults to charts/<run_stem>_mean_scores.svg",
    )
    parser.add_argument(
        "--title",
        default="Mean Score by Model and Benchmark",
        help="Chart title.",
    )
    args = parser.parse_args()

    results = _read_results(args.run_path)
    output = args.output or Path("charts") / f"{args.run_path.stem}_mean_scores.svg"
    write_mean_score_chart(output, results, args.title)
    print(f"Wrote {output}")


def _read_results(path: Path) -> list[CaseResult]:
    results = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"Could not parse {path} as JSONL at line {line_number}: {exc}\n"
                    "Use a run_*.jsonl raw results file, not an HTML report or SVG chart."
                ) from exc
            results.append(CaseResult(**row))
    return results


if __name__ == "__main__":
    main()
