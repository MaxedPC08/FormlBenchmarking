import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_benchmarking import config
from llm_benchmarking.benchmarks import BENCHMARK_REGISTRY
from llm_benchmarking.cases import CaseResult
from llm_benchmarking.charts import write_mean_score_chart
from llm_benchmarking.json_utils import write_jsonl
from llm_benchmarking.providers import build_provider
from llm_benchmarking.reporting import write_html_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark LLMs on configured tasks.")
    parser.add_argument("--list", action="store_true", help="List enabled models and benchmarks.")
    parser.add_argument("--dry-run", action="store_true", help="Load cases without calling models.")
    parser.add_argument("--limit", type=int, default=None, help="Override per-benchmark case limit.")
    parser.add_argument("--only", nargs="*", default=None, help="Run only these benchmark names.")
    parser.add_argument(
        "--context-window",
        type=int,
        default=None,
        help="Override every model context window for this run.",
    )
    args = parser.parse_args()

    selected = _selected_benchmarks(args.only)
    if args.list:
        _print_config(selected)
        return

    results = run(
        selected,
        dry_run=args.dry_run,
        limit_override=args.limit,
        context_window_override=args.context_window,
    )
    if args.dry_run:
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(config.RESULTS_DIR)
    result_path = output_dir / f"run_{timestamp}.jsonl"
    summary_path = output_dir / f"summary_{timestamp}.json"
    report_path = output_dir / f"report_{timestamp}.html"
    chart_path = output_dir / f"chart_{timestamp}.svg"
    write_jsonl(result_path, [asdict(result) for result in results])
    summary = summarize(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_html_report(report_path, results, summary)
    write_mean_score_chart(chart_path, results)
    print(f"Wrote {len(results)} case results to {result_path}")
    print(f"Wrote summary to {summary_path}")
    print(f"Wrote report to {report_path}")
    print(f"Wrote chart to {chart_path}")
    print(json.dumps(summary, indent=2))


def run(
    selected: dict[str, dict[str, Any]],
    dry_run: bool = False,
    limit_override: int | None = None,
    context_window_override: int | None = None,
) -> list[CaseResult]:
    providers = [build_provider(model) for model in config.MODELS]
    all_results: list[CaseResult] = []

    for benchmark_name, benchmark_config in selected.items():
        benchmark_cls = BENCHMARK_REGISTRY[benchmark_name]
        limit = limit_override if limit_override is not None else benchmark_config.get("limit")
        benchmark = benchmark_cls(
            data_path=benchmark_config.get("data_path"),
            limit=limit,
        )
        cases = benchmark.cases()
        print(f"{benchmark_name}: loaded {len(cases)} case(s)")
        if dry_run:
            continue

        for provider in providers:
            for case in cases:
                estimated_input_tokens = _estimate_tokens(case.prompt)
                available_input_tokens = _available_input_tokens(
                    provider.model,
                    context_window_override,
                )
                if available_input_tokens is not None and estimated_input_tokens > available_input_tokens:
                    metadata = {
                        "skipped": True,
                        "skip_reason": "input_too_long",
                        "estimated_input_tokens": estimated_input_tokens,
                        "available_input_tokens": available_input_tokens,
                        "context_window": _context_window(
                            provider.model,
                            context_window_override,
                        ),
                        "max_output_tokens": provider.model.max_tokens,
                        "input_tokens": None,
                        "output_tokens": None,
                        "total_tokens": None,
                    }
                    all_results.append(
                        CaseResult(
                            benchmark=benchmark_name,
                            model=provider.model.name,
                            case_id=case.id,
                            prompt=case.prompt,
                            prediction="",
                            score=0.0,
                            passed=False,
                            metadata=metadata,
                        )
                    )
                    print(
                        f"{benchmark_name}/{provider.model.name}/{case.id}: skipped "
                        f"estimated_input_tokens={estimated_input_tokens} "
                        f"available_input_tokens={available_input_tokens}"
                    )
                    continue

                try:
                    generation = provider.generate(case.prompt)
                    prediction = generation.text
                    score, metadata = benchmark.score(case, prediction)
                    metadata = {
                        **metadata,
                        "input_tokens": generation.input_tokens,
                        "output_tokens": generation.output_tokens,
                        "total_tokens": generation.total_tokens,
                    }
                except Exception as exc:
                    prediction = ""
                    score = 0.0
                    metadata = {
                        "error": str(exc),
                        "input_tokens": None,
                        "output_tokens": None,
                        "total_tokens": None,
                    }
                all_results.append(
                    CaseResult(
                        benchmark=benchmark_name,
                        model=provider.model.name,
                        case_id=case.id,
                        prompt=case.prompt,
                        prediction=prediction,
                        score=score,
                        passed=score >= 1.0,
                        metadata=metadata,
                    )
                )
                print(
                    f"{benchmark_name}/{provider.model.name}/{case.id}: "
                    f"score={score:.3f} "
                    f"tokens={_format_tokens(metadata.get('input_tokens'), metadata.get('output_tokens'))}"
                )
                if metadata.get("error"):
                    print(f"  error: {metadata['error']}")
    return all_results


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    buckets: dict[str, list[CaseResult]] = {}
    for result in results:
        key = f"{result.model}::{result.benchmark}"
        buckets.setdefault(key, []).append(result)

    summary = {}
    for key, bucket in buckets.items():
        model, benchmark = key.split("::", 1)
        attempted = [item for item in bucket if not item.metadata.get("skipped")]
        denominator = attempted or bucket
        summary.setdefault(model, {})[benchmark] = {
            "cases": len(bucket),
            "attempted_cases": len(attempted),
            "skipped_cases": len(bucket) - len(attempted),
            "mean_score": sum(item.score for item in denominator) / len(denominator),
            "pass_rate": sum(item.passed for item in denominator) / len(denominator),
            "input_tokens": _sum_metadata_int(attempted, "input_tokens"),
            "output_tokens": _sum_metadata_int(attempted, "output_tokens"),
            "total_tokens": _sum_metadata_int(attempted, "total_tokens"),
            "avg_input_tokens": _avg_metadata_int(attempted, "input_tokens"),
            "avg_output_tokens": _avg_metadata_int(attempted, "output_tokens"),
            "avg_total_tokens": _avg_metadata_int(attempted, "total_tokens"),
        }
    return summary


def _sum_metadata_int(results: list[CaseResult], key: str) -> int | None:
    values = [_metadata_int(result, key) for result in results]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values)


def _avg_metadata_int(results: list[CaseResult], key: str) -> float | None:
    values = [_metadata_int(result, key) for result in results]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _metadata_int(result: CaseResult, key: str) -> int | None:
    value = result.metadata.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_tokens(input_tokens: int | None, output_tokens: int | None) -> str:
    if input_tokens is None and output_tokens is None:
        return "n/a"
    input_display = str(input_tokens) if input_tokens is not None else "?"
    output_display = str(output_tokens) if output_tokens is not None else "?"
    return f"in:{input_display} out:{output_display}"


def _estimate_tokens(text: str) -> int:
    chars_per_token = float(getattr(config, "CHARS_PER_TOKEN_ESTIMATE", 4.0))
    chars_per_token = max(chars_per_token, 1.0)
    return max(1, int(len(text) / chars_per_token))


def _context_window(model: Any, override: int | None = None) -> int | None:
    if override is not None:
        return override if override > 0 else None
    value = getattr(model, "context_window", None)
    if value is None:
        value = getattr(config, "DEFAULT_CONTEXT_WINDOW_TOKENS", None)
    if value is None:
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _available_input_tokens(model: Any, override: int | None = None) -> int | None:
    context_window = _context_window(model, override)
    if context_window is None:
        return None
    safety_margin = int(getattr(config, "CONTEXT_WINDOW_SAFETY_MARGIN_TOKENS", 256))
    return max(0, context_window - int(model.max_tokens) - safety_margin)


def _selected_benchmarks(only: list[str] | None) -> dict[str, dict[str, Any]]:
    enabled = {
        name: settings
        for name, settings in config.BENCHMARKS.items()
        if settings.get("enabled", False)
    }
    if only:
        missing = sorted(set(only) - set(BENCHMARK_REGISTRY))
        if missing:
            raise SystemExit(f"Unknown benchmark(s): {', '.join(missing)}")
        return {name: enabled[name] for name in only if name in enabled}
    return enabled


def _print_config(selected: dict[str, dict[str, Any]]) -> None:
    print("Models:")
    for model in config.MODELS:
        print(f"  - {model['provider']}:{model['name']}")
    print("Benchmarks:")
    for name, settings in selected.items():
        data_path = settings.get("data_path") or "built-in smoke sample"
        print(f"  - {name}: {data_path}")
