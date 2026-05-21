from typing import Any

from llm_benchmarking.benchmarks.generic_qa import GenericQABenchmark
from llm_benchmarking.cases import BenchmarkCase
from llm_benchmarking.json_utils import read_json_or_jsonl
from llm_benchmarking.scoring import multiple_choice_letter


class LongBenchV2Benchmark(GenericQABenchmark):
    name = "longbench_v2"
    scoring = "multiple_choice"
    sample_cases = [
        BenchmarkCase(
            id="longbench_v2_sample",
            prompt=(
                "Read the passage and choose the correct answer. Respond with only A, B, C, or D.\n\n"
                "Passage: Mira filed the blue contract in cabinet C after moving the "
                "red invoice to cabinet A. The green memo remained in cabinet B.\n\n"
                "Question: Which cabinet contains the blue contract?\n"
                "A. cabinet A\nB. cabinet B\nC. cabinet C\nD. none of the above"
            ),
            answers=["C"],
            metadata={"choices": ["A", "B", "C", "D"]},
        )
    ]

    def load_cases(self) -> list[BenchmarkCase]:
        if not self.data_path:
            return self.sample_cases

        rows = read_json_or_jsonl(self.data_path)
        return [self._row_to_case(index, row) for index, row in enumerate(rows)]

    def score(self, case: BenchmarkCase, prediction: str) -> tuple[float, dict[str, Any]]:
        valid_letters = set(case.metadata.get("choices", ["A", "B", "C", "D"]))
        predicted = multiple_choice_letter(prediction, valid_letters)
        score = float(predicted in case.answers)
        return score, {"predicted_choice": predicted, "answers": case.answers}

    def _row_to_case(self, index: int, row: dict[str, Any]) -> BenchmarkCase:
        if {"choice_A", "choice_B", "choice_C", "choice_D"} <= set(row):
            choices = [row["choice_A"], row["choice_B"], row["choice_C"], row["choice_D"]]
            letters = ["A", "B", "C", "D"]
            options = "\n".join(
                f"{letter}. {choice}" for letter, choice in zip(letters, choices)
            )
            prompt = (
                "Read the context and choose the correct answer. "
                "Respond with only A, B, C, or D.\n\n"
                f"Context:\n{row.get('context', '')}\n\n"
                f"Question: {row.get('question', '')}\n"
                f"{options}"
            )
            return BenchmarkCase(
                id=str(row.get("_id", row.get("id", index))),
                prompt=prompt,
                answers=[str(row.get("answer", "")).strip().upper()],
                metadata={"choices": letters, "raw": row},
            )

        choices = row.get("choices", [])
        letters = [chr(ord("A") + i) for i in range(len(choices))]
        options = "\n".join(f"{letter}. {choice}" for letter, choice in zip(letters, choices))
        prompt = row.get("prompt")
        if not prompt:
            prompt = (
                "Read the context and choose the correct answer. Respond with only the letter.\n\n"
                f"Context:\n{row.get('context', '')}\n\n"
                f"Question: {row.get('question', '')}\n"
                f"{options}"
            )
        answer = str(row.get("answer", row.get("target", ""))).strip().upper()
        return BenchmarkCase(
            id=str(row.get("id", row.get("_id", index))),
            prompt=str(prompt),
            answers=[answer] if answer else [],
            metadata={"choices": letters or ["A", "B", "C", "D"], "raw": row},
        )
