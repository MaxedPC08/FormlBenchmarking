from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    prompt: str
    answers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CaseResult:
    benchmark: str
    model: str
    case_id: str
    prompt: str
    prediction: str
    score: float
    passed: bool
    metadata: dict[str, Any] = field(default_factory=dict)
