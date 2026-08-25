from __future__ import annotations

import re


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def token_f1(prediction: str, reference: str) -> float:
    predicted = _tokens(prediction)
    expected = _tokens(reference)
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum(min(predicted.count(token), expected.count(token)) for token in set(predicted) & set(expected))
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def reciprocal_rank(retrieved_ids: list[str], reference_ids: list[str]) -> float:
    references = set(reference_ids)
    for index, source_id in enumerate(retrieved_ids, start=1):
        if source_id in references:
            return 1.0 / index
    return 0.0
