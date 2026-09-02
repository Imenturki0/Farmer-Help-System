from __future__ import annotations

import math
import re
from typing import Sequence


def normalize_id(value) -> str:
    """
    Normalize chunk IDs so evaluation does not care
    whether the dataset contains 2 or "2".
    """
    return str(value)


def normalize_ids(values: Sequence) -> list[str]:
    return [normalize_id(value) for value in values]


# ============================================================
# RETRIEVAL
# ============================================================

def recall_at_k(
    retrieved: Sequence,
    relevant: Sequence,
    k: int,
) -> float:

    retrieved = normalize_ids(retrieved[:k])
    relevant = set(normalize_ids(relevant))

    if not relevant:
        return 0.0

    return len(set(retrieved) & relevant) / len(relevant)


def precision_at_k(
    retrieved: Sequence,
    relevant: Sequence,
    k: int,
) -> float:

    retrieved = normalize_ids(retrieved[:k])
    relevant = set(normalize_ids(relevant))

    if not retrieved:
        return 0.0

    return len(set(retrieved) & relevant) / len(retrieved)


def hit_at_k(
    retrieved: Sequence,
    relevant: Sequence,
    k: int,
) -> int:

    retrieved = set(normalize_ids(retrieved[:k]))
    relevant = set(normalize_ids(relevant))

    return int(bool(retrieved & relevant))


def reciprocal_rank(
    retrieved: Sequence,
    relevant: Sequence,
) -> float:

    relevant = set(normalize_ids(relevant))

    for rank, doc_id in enumerate(
        normalize_ids(retrieved),
        start=1,
    ):
        if doc_id in relevant:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    retrieved: Sequence,
    relevant: Sequence,
    k: int,
) -> float:

    retrieved = normalize_ids(retrieved[:k])
    relevant = set(normalize_ids(relevant))

    if not relevant:
        return 0.0

    dcg = 0.0

    for rank, doc_id in enumerate(retrieved, start=1):

        if doc_id in relevant:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant), k)

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg


# ============================================================
# TEXT
# ============================================================

def normalize_text(text: str) -> str:

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def token_f1(
    prediction: str,
    reference: str,
) -> float:

    prediction_tokens = set(
        normalize_text(prediction).split()
    )

    reference_tokens = set(
        normalize_text(reference).split()
    )

    if not prediction_tokens or not reference_tokens:
        return 0.0

    common = (
        prediction_tokens
        & reference_tokens
    )

    precision = len(common) / len(prediction_tokens)
    recall = len(common) / len(reference_tokens)

    if precision + recall == 0:
        return 0.0

    return (
        2 * precision * recall
        / (precision + recall)
    )


def exact_match(
    prediction: str,
    reference: str,
) -> float:

    return float(
        normalize_text(prediction)
        ==
        normalize_text(reference)
    )