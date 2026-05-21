import json
import re
from typing import Any

from llm_benchmarking.benchmarks.base import Benchmark
from llm_benchmarking.cases import BenchmarkCase
from llm_benchmarking.json_utils import read_json_or_jsonl


ENTITY_TYPES = {"PER", "ORG", "LOC"}
TAG_NAMES = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


class UniversalNERBenchmark(Benchmark):
    def load_cases(self) -> list[BenchmarkCase]:
        if not self.data_path:
            return _sample_cases()

        cases = []
        for index, row in enumerate(read_json_or_jsonl(self.data_path)):
            tokens = [str(token) for token in row.get("tokens", [])]
            text = str(row.get("text") or " ".join(tokens))
            tags = [_tag_name(tag) for tag in row.get("ner_tags", [])]
            if len(tokens) != len(tags):
                raise ValueError(
                    "UniversalNER rows must contain equal-length tokens and ner_tags "
                    f"(row {index} has {len(tokens)} tokens and {len(tags)} tags)."
                )
            case_id = str(row.get("id", row.get("idx", index)))
            cases.append(
                BenchmarkCase(
                    id=case_id,
                    prompt=_build_prompt(tokens, text),
                    metadata={
                        **{k: v for k, v in row.items() if k not in {"prompt"}},
                        "text": text,
                        "tokens": tokens,
                        "ner_tags": tags,
                    },
                )
            )
        return cases

    def score(self, case: BenchmarkCase, prediction: str) -> tuple[float, dict[str, Any]]:
        tokens = [str(token) for token in case.metadata.get("tokens", [])]
        tags = [_tag_name(tag) for tag in case.metadata.get("ner_tags", [])]
        gold = set(_bio_entities(tokens, tags))
        predicted, parse_error = _prediction_entities(prediction, tokens)

        true_positives = len(gold & predicted)
        false_positives = len(predicted - gold)
        false_negatives = len(gold - predicted)
        precision = true_positives / len(predicted) if predicted else float(not gold)
        recall = true_positives / len(gold) if gold else float(not predicted)
        score = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        return score, {
            "scoring": "entity_f1",
            "precision": precision,
            "recall": recall,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "gold_entities": [_entity_dict(entity, tokens) for entity in sorted(gold)],
            "predicted_entities": [
                _entity_dict(entity, tokens) for entity in sorted(predicted)
            ],
            "parse_error": parse_error,
        }


def _build_prompt(tokens: list[str], text: str) -> str:
    numbered_tokens = "\n".join(
        f"{index}: {token}" for index, token in enumerate(tokens)
    )
    return (
        "Extract all named entities from the sentence using only these labels: "
        "PER, ORG, LOC.\n"
        "Return only a JSON array. Each item must be an object with exactly these "
        'keys: "text", "type", "start_token", "end_token". Use zero-based token '
        "indices and make end_token exclusive. Return [] if there are no entities.\n\n"
        f"Sentence:\n{text}\n\n"
        f"Tokens:\n{numbered_tokens}"
    )


def _sample_cases() -> list[BenchmarkCase]:
    tokens = [
        "Several",
        "analysts",
        "have",
        "suggested",
        "Huawei",
        "is",
        "best",
        "placed",
        "to",
        "benefit",
        "from",
        "Samsung",
        "'s",
        "setback",
        ".",
    ]
    text = "Several analysts have suggested Huawei is best placed to benefit from Samsung's setback."
    return [
        BenchmarkCase(
            id="sample_en",
            prompt=_build_prompt(tokens, text),
            metadata={
                "text": text,
                "tokens": tokens,
                "ner_tags": [
                    "O",
                    "O",
                    "O",
                    "O",
                    "B-ORG",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "B-ORG",
                    "O",
                    "O",
                    "O",
                ],
            },
        )
    ]


def _tag_name(tag: Any) -> str:
    if isinstance(tag, int) and 0 <= tag < len(TAG_NAMES):
        return TAG_NAMES[tag]
    tag_text = str(tag)
    if tag_text.isdigit() and 0 <= int(tag_text) < len(TAG_NAMES):
        return TAG_NAMES[int(tag_text)]
    if "OTH" in tag_text or tag_text == "B-O":
        return "O"
    return tag_text


def _bio_entities(tokens: list[str], tags: list[str]) -> list[tuple[str, int, int]]:
    entities: list[tuple[str, int, int]] = []
    current_type: str | None = None
    start: int | None = None

    for index, tag in enumerate(tags + ["O"]):
        prefix, entity_type = _split_tag(tag)
        continues = prefix == "I" and entity_type == current_type
        if current_type is not None and not continues:
            assert start is not None
            entities.append((current_type, start, index))
            current_type = None
            start = None
        if prefix == "B" or (prefix == "I" and entity_type in ENTITY_TYPES and not continues):
            current_type = entity_type
            start = index

    return entities


def _split_tag(tag: str) -> tuple[str, str | None]:
    if "-" not in tag:
        return tag, None
    prefix, entity_type = tag.split("-", 1)
    if entity_type not in ENTITY_TYPES:
        return "O", None
    return prefix, entity_type


def _prediction_entities(
    prediction: str, tokens: list[str]
) -> tuple[set[tuple[str, int, int]], str | None]:
    try:
        payload = json.loads(_json_payload(prediction))
    except json.JSONDecodeError as exc:
        return set(), str(exc)

    if isinstance(payload, dict):
        for key in ("entities", "named_entities", "predictions"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        return set(), "Expected a JSON array of entities."

    entities = set()
    errors = []
    for item in payload:
        if not isinstance(item, dict):
            errors.append(f"Ignored non-object item: {item!r}")
            continue
        entity_type = _normalize_type(
            item.get("type") or item.get("label") or item.get("entity_type")
        )
        if entity_type not in ENTITY_TYPES:
            errors.append(f"Ignored unknown entity type: {item!r}")
            continue
        start = _as_int(item.get("start_token"))
        end = _as_int(item.get("end_token"))
        if start is None or end is None:
            matched = _match_entity_text(str(item.get("text", "")), tokens)
            if matched is None:
                errors.append(f"Ignored entity without usable token span: {item!r}")
                continue
            start, end = matched
        if end == start and 0 <= start < len(tokens):
            end += 1
        if start < 0 or end <= start or end > len(tokens):
            errors.append(f"Ignored out-of-range token span: {item!r}")
            continue
        entities.add((entity_type, start, end))

    return entities, "; ".join(errors) if errors else None


def _json_payload(prediction: str) -> str:
    stripped = prediction.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.I)
    if fenced:
        return fenced.group(1).strip()
    start_positions = [position for position in (stripped.find("["), stripped.find("{")) if position >= 0]
    if start_positions:
        return stripped[min(start_positions) :]
    return stripped


def _normalize_type(value: Any) -> str:
    text = str(value or "").upper().strip()
    aliases = {
        "PERSON": "PER",
        "PEOPLE": "PER",
        "ORGANIZATION": "ORG",
        "ORGANISATION": "ORG",
        "LOCATION": "LOC",
        "PLACE": "LOC",
    }
    return aliases.get(text, text)


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _match_entity_text(entity_text: str, tokens: list[str]) -> tuple[int, int] | None:
    normalized = _normalize_entity_text(entity_text)
    if not normalized:
        return None
    for start in range(len(tokens)):
        for end in range(start + 1, len(tokens) + 1):
            if _normalize_entity_text(_join_entity_text(tokens[start:end])) == normalized:
                return start, end
    return None


def _join_entity_text(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,!?;:%])", r"\1", text)
    text = re.sub(r"\s+(['’]s)\b", r"\1", text)
    text = re.sub(r"([([{])\s+", r"\1", text)
    text = re.sub(r"\s+([])}])", r"\1", text)
    return text


def _normalize_entity_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _entity_dict(entity: tuple[str, int, int], tokens: list[str]) -> dict[str, Any]:
    entity_type, start, end = entity
    return {
        "type": entity_type,
        "start_token": start,
        "end_token": end,
        "text": _join_entity_text(tokens[start:end]),
    }
