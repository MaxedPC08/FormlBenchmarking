from abc import ABC, abstractmethod
from typing import Any

from llm_benchmarking.cases import BenchmarkCase


class Benchmark(ABC):
    name: str

    def __init__(self, data_path: str | None = None, limit: int | None = None) -> None:
        self.data_path = data_path
        self.limit = limit

    def cases(self) -> list[BenchmarkCase]:
        cases = self.load_cases()
        if self.limit is not None:
            return cases[: self.limit]
        return cases

    @abstractmethod
    def load_cases(self) -> list[BenchmarkCase]:
        raise NotImplementedError

    @abstractmethod
    def score(self, case: BenchmarkCase, prediction: str) -> tuple[float, dict[str, Any]]:
        raise NotImplementedError
