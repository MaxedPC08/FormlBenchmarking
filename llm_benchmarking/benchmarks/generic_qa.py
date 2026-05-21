from typing import Any

from llm_benchmarking.benchmarks.base import Benchmark
from llm_benchmarking.cases import BenchmarkCase
from llm_benchmarking.json_utils import read_json_or_jsonl
from llm_benchmarking.scoring import contains_answer, exact_match, token_f1


class GenericQABenchmark(Benchmark):
    scoring = "contains"
    sample_cases: list[BenchmarkCase] = []

    def load_cases(self) -> list[BenchmarkCase]:
        if not self.data_path:
            return self.sample_cases

        rows = read_json_or_jsonl(self.data_path)
        cases = []
        for index, row in enumerate(rows):
            prompt = _first(row, "prompt", "input", "question", "context")
            if "context" in row and "question" in row:
                prompt = f"Context:\n{row['context']}\n\nQuestion: {row['question']}\nAnswer:"
            answers = _answers(row)
            case_id = str(row.get("id", row.get("uid", index)))
            cases.append(
                BenchmarkCase(
                    id=case_id,
                    prompt=str(prompt),
                    answers=answers,
                    metadata={k: v for k, v in row.items() if k not in {"prompt", "input"}},
                )
            )
        return cases

    def score(self, case: BenchmarkCase, prediction: str) -> tuple[float, dict[str, Any]]:
        if self.scoring == "exact":
            score = exact_match(prediction, case.answers)
        elif self.scoring == "f1":
            score = token_f1(prediction, case.answers)
        else:
            score = contains_answer(prediction, case.answers)
        return score, {"answers": case.answers, "scoring": self.scoring}


def _first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    raise ValueError(f"Could not find any of {keys} in row: {row}")


def _answers(row: dict[str, Any]) -> list[str]:
    for key in ("answers", "answer", "target", "targets", "output", "outputs"):
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            return [str(item) for item in value]
        return [str(value)]
    return []
