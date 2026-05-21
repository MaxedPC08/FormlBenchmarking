import re
import string
from collections import Counter


def normalize_text(value: str) -> str:
    value = value.lower()
    value = value.translate(str.maketrans("", "", string.punctuation))
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def exact_match(prediction: str, answers: list[str]) -> float:
    normalized_prediction = normalize_text(prediction)
    return float(any(normalized_prediction == normalize_text(answer) for answer in answers))


def contains_answer(prediction: str, answers: list[str]) -> float:
    normalized_prediction = normalize_text(prediction)
    return float(any(normalize_text(answer) in normalized_prediction for answer in answers))


def token_f1(prediction: str, answers: list[str]) -> float:
    prediction_tokens = normalize_text(prediction).split()
    if not prediction_tokens:
        return 0.0

    best = 0.0
    for answer in answers:
        answer_tokens = normalize_text(answer).split()
        if not answer_tokens:
            continue
        common = Counter(prediction_tokens) & Counter(answer_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            continue
        precision = overlap / len(prediction_tokens)
        recall = overlap / len(answer_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def multiple_choice_letter(prediction: str, valid_letters: set[str]) -> str | None:
    match = re.search(r"\b([A-Z])\b", prediction.upper())
    if match and match.group(1) in valid_letters:
        return match.group(1)
    match = re.search(r"(?:answer is|answer:)\s*([A-Z])", prediction, flags=re.IGNORECASE)
    if match and match.group(1).upper() in valid_letters:
        return match.group(1).upper()
    return None
