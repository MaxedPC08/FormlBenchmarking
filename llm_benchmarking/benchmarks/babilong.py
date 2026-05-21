import re
from typing import Any

from llm_benchmarking.benchmarks.generic_qa import GenericQABenchmark
from llm_benchmarking.cases import BenchmarkCase
from llm_benchmarking.scoring import contains_answer, exact_match, normalize_text


LOCATIONS = {
    "bathroom",
    "bedroom",
    "garden",
    "hallway",
    "kitchen",
    "office",
}


class BABILongBenchmark(GenericQABenchmark):
    name = "babilong"
    scoring = "location"
    sample_cases = [
        BenchmarkCase(
            id="babilong_sample",
            prompt=(
                "Answer the question from the story. Reply with only the location word.\n\n"
                "Daniel went to the hallway. Sandra travelled to the kitchen. "
                "Daniel picked up the football. Daniel went to the garden.\n\n"
                "Question: Where is the football?"
            ),
            answers=["garden", "the garden"],
        )
    ]

    def load_cases(self) -> list[BenchmarkCase]:
        cases = super().load_cases()
        return [
            BenchmarkCase(
                id=case.id,
                prompt=_tighten_prompt(case.prompt),
                answers=case.answers,
                metadata=case.metadata,
            )
            for case in cases
        ]

    def score(self, case: BenchmarkCase, prediction: str) -> tuple[float, dict[str, Any]]:
        extracted = _extract_location(prediction)
        if extracted:
            score = exact_match(extracted, case.answers)
        else:
            score = contains_answer(prediction, case.answers)
        return score, {
            "answers": case.answers,
            "extracted_answer": extracted,
            "scoring": self.scoring,
        }


def _tighten_prompt(prompt: str) -> str:
    instruction = "Answer with only one location word."
    if instruction.lower() in prompt.lower():
        return prompt
    return f"{instruction}\n\n{prompt}"


def _extract_location(prediction: str) -> str | None:
    text = normalize_text(prediction)
    if text in LOCATIONS:
        return text

    patterns = [
        r"\b(?:is|in|at|to)\s+(?:the\s+)?(bathroom|bedroom|garden|hallway|kitchen|office)\b",
        r"\b(bathroom|bedroom|garden|hallway|kitchen|office)\b",
    ]
    matches = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, text))
        if matches:
            break
    return matches[-1] if matches else None
