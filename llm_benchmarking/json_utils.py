import json
import gzip
from pathlib import Path
from typing import Any, Iterable


def read_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {source}")

    if source.suffix.lower() == ".gz" and source.name.endswith(".jsonl.gz"):
        rows = []
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    rows.append(json.loads(stripped))
        return rows

    if source.suffix.lower() == ".jsonl":
        rows = []
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    rows.append(json.loads(stripped))
        return rows

    with source.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "examples", "instances", "rows"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"Expected a JSON list or JSONL records in {source}")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
