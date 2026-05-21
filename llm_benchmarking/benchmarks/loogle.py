from typing import Any

from llm_benchmarking.benchmarks.base import Benchmark
from llm_benchmarking.cases import BenchmarkCase
from llm_benchmarking.json_utils import read_json_or_jsonl
from llm_benchmarking.scoring import contains_answer, exact_match, token_f1


class LooGLEBenchmark(Benchmark):
    sample_cases = [
        BenchmarkCase(
            id="loogle_sample",
            prompt=(
                "Answer the question using only the provided long context. Keep the "
                "answer concise.\n\n"
                "Title: Sample city notes\n\n"
                "Context:\n"
                "In 2001, Mira founded Northstar Labs in Seattle. Years later, the "
                "company opened a second office in Lisbon. The Seattle office stayed "
                "focused on hardware, while the Lisbon office focused on translation "
                "tools.\n\n"
                "Question:\n"
                "Which city hosted Northstar Labs' second office?"
            ),
            answers=["Lisbon"],
            metadata={"task": "sample", "title": "Sample city notes"},
        )
    ]

    def load_cases(self) -> list[BenchmarkCase]:
        if not self.data_path:
            return self.sample_cases

        rows = read_json_or_jsonl(self.data_path)
        return [self._row_to_case(index, row) for index, row in enumerate(rows)]

    def score(self, case: BenchmarkCase, prediction: str) -> tuple[float, dict[str, Any]]:
        answers = case.answers
        if not answers:
            return 0.0, {"answers": answers, "scoring": "missing_answer"}
        exact = exact_match(prediction, answers)
        contains = contains_answer(prediction, answers)
        f1 = token_f1(prediction, answers)
        score = max(exact, contains, f1)
        return score, {
            "exact": exact,
            "contains": contains,
            "token_f1": f1,
            "answers": answers,
            "scoring": "max_exact_contains_token_f1",
        }

    def _row_to_case(self, index: int, row: dict[str, Any]) -> BenchmarkCase:
        title = str(row.get("title", ""))
        context = str(row.get("context", ""))
        question = str(row.get("question", ""))
        task = str(row.get("task", ""))
        answer = row.get("answer", row.get("answers", ""))
        answers = [str(item) for item in answer] if isinstance(answer, list) else [str(answer)]

        prompt = (
            "Answer the question using only the provided long context. Keep the "
            "answer concise, and do not use outside knowledge.\n\n"
            f"Title:\n{title}\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}"
        )
        return BenchmarkCase(
            id=str(row.get("id", index)),
            prompt=prompt,
            answers=[answer for answer in answers if answer],
            metadata={
                "task": task,
                "title": title,
                "doc_id": row.get("doc_id"),
                "evidence": row.get("evidence", []),
                "type": row.get("type"),
                "raw": row,
            },
        )
