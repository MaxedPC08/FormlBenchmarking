#!/usr/bin/env python3
"""Recompute benchmark scores from a saved JSONL run."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_benchmarking.benchmarks import BENCHMARK_REGISTRY
from llm_benchmarking.cases import BenchmarkCase, CaseResult
from llm_benchmarking.json_utils import write_jsonl
from llm_benchmarking.runner import summarize
from llm_benchmarking.reporting import write_html_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore an existing benchmark run.")
    parser.add_argument("run_path", type=Path)
    parser.add_argument(
        "--suffix",
        default="rescored",
        help="Suffix to append to rewritten result, summary, and report files.",
    )
    args = parser.parse_args()

    results = []
    for line in args.run_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        result = CaseResult(**json.loads(line))
        results.append(_rescore(result))

    stem = args.run_path.stem
    output_dir = args.run_path.parent
    run_path = output_dir / f"{stem}_{args.suffix}.jsonl"
    summary_path = output_dir / f"{stem.replace('run_', 'summary_')}_{args.suffix}.json"
    report_path = output_dir / f"{stem.replace('run_', 'report_')}_{args.suffix}.html"

    summary = summarize(results)
    write_jsonl(run_path, [asdict(result) for result in results])
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_html_report(report_path, results, summary)

    print(f"Wrote {run_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")


def _rescore(result: CaseResult) -> CaseResult:
    benchmark_cls = BENCHMARK_REGISTRY.get(result.benchmark)
    if not benchmark_cls:
        return result

    metadata = dict(result.metadata)
    case = BenchmarkCase(
        id=result.case_id,
        prompt=result.prompt,
        answers=[str(answer) for answer in metadata.get("answers", [])],
        metadata=metadata,
    )

    try:
        score, score_metadata = benchmark_cls().score(case, result.prediction)
    except Exception:
        return result

    merged_metadata = {
        **metadata,
        **score_metadata,
        "input_tokens": metadata.get("input_tokens"),
        "output_tokens": metadata.get("output_tokens"),
        "total_tokens": metadata.get("total_tokens"),
    }
    return CaseResult(
        benchmark=result.benchmark,
        model=result.model,
        case_id=result.case_id,
        prompt=result.prompt,
        prediction=result.prediction,
        score=score,
        passed=score >= 1.0,
        metadata=merged_metadata,
    )


if __name__ == "__main__":
    main()
