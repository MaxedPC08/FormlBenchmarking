import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from llm_benchmarking.benchmarks.base import Benchmark
from llm_benchmarking.cases import BenchmarkCase
from llm_benchmarking.json_utils import read_json_or_jsonl


class HumanEvalBenchmark(Benchmark):
    name = "humaneval"

    def load_cases(self) -> list[BenchmarkCase]:
        if not self.data_path:
            return [
                BenchmarkCase(
                    id="humaneval_sample",
                    prompt=(
                        "Complete the Python function. Return only code, with no markdown.\n\n"
                        "def add(a: int, b: int) -> int:\n"
                    ),
                    metadata={
                        "prompt_code": "def add(a: int, b: int) -> int:\n",
                        "test": "assert add(2, 3) == 5\nassert add(-1, 1) == 0",
                        "entry_point": "add",
                    },
                )
            ]

        rows = read_json_or_jsonl(self.data_path)
        cases = []
        for index, row in enumerate(rows):
            prompt = row.get("prompt")
            if not prompt:
                raise ValueError("HumanEval rows need a 'prompt' field.")
            cases.append(
                BenchmarkCase(
                    id=str(row.get("task_id", row.get("id", index))),
                    prompt=(
                        "Complete the Python function. Return only code, with no markdown.\n\n"
                        f"{prompt}"
                    ),
                    metadata={
                        "prompt_code": prompt,
                        "test": row.get("test", ""),
                        "entry_point": row.get("entry_point"),
                        "raw": row,
                    },
                )
            )
        return cases

    def score(self, case: BenchmarkCase, prediction: str) -> tuple[float, dict[str, Any]]:
        code = _strip_markdown(prediction)
        prompt_code = str(case.metadata.get("prompt_code", ""))
        entry_point = str(case.metadata.get("entry_point", ""))
        if prompt_code and entry_point and f"def {entry_point}" not in code:
            code = f"{prompt_code}{code}"
        passed, detail = _run_python_test(code, str(case.metadata.get("test", "")))
        return float(passed), {"passed": passed, "detail": detail}


def _strip_markdown(prediction: str) -> str:
    text = prediction.strip("\n")
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.rstrip()


def _run_python_test(code: str, test: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="humaneval_") as tmpdir:
        path = Path(tmpdir) / "candidate.py"
        path.write_text(f"{code}\n\n{test}\n", encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    if result.returncode == 0:
        return True, "passed"
    detail = (result.stderr or result.stdout).strip()
    return False, detail[-1000:]
