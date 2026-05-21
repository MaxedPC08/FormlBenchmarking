#!/usr/bin/env python3
"""Combine two benchmark runs by replacing selected benchmarks."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_benchmarking.cases import CaseResult
from llm_benchmarking.json_utils import write_jsonl
from llm_benchmarking.reporting import write_html_report
from llm_benchmarking.runner import summarize


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Keep most results from a base run, then replace selected benchmark "
            "rows with rows from another run."
        )
    )
    parser.add_argument("base_run", type=Path)
    parser.add_argument("replacement_run", type=Path)
    parser.add_argument(
        "--replace",
        nargs="+",
        required=True,
        help="Benchmark names to replace from the replacement run.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Output name stem. Defaults to base stem plus '_combined'.",
    )
    args = parser.parse_args()

    replace = set(args.replace)
    base_results = _read_results(args.base_run)
    replacement_results = _read_results(args.replacement_run)

    combined = [
        result for result in base_results if result.benchmark not in replace
    ] + [
        result for result in replacement_results if result.benchmark in replace
    ]

    output_dir = args.base_run.parent
    stem = args.name or f"{args.base_run.stem}_combined"
    run_path = output_dir / f"{stem}.jsonl"
    summary_path = output_dir / f"{stem.replace('run_', 'summary_')}.json"
    report_path = output_dir / f"{stem.replace('run_', 'report_')}.html"

    summary = summarize(combined)
    write_jsonl(run_path, [asdict(result) for result in combined])
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_html_report(report_path, combined, summary)

    print(f"Wrote {run_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")


def _read_results(path: Path) -> list[CaseResult]:
    results = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            results.append(CaseResult(**json.loads(line)))
    return results


if __name__ == "__main__":
    main()
