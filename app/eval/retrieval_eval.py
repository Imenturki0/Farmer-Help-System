from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from app.services.rag import rag

from app.eval.config import CONFIG
from app.eval.metrics import (
    recall_at_k,
    precision_at_k,
    hit_at_k,
    reciprocal_rank,
    ndcg_at_k,
    normalize_ids,
)


# ============================================================
# RETRIEVERS
# ============================================================

def dense_search(
    question: str,
    k: int,
):
    return rag._vector_search(
        question,
        k,
    )


def bm25_search(
    question: str,
    k: int,
):
    return rag.bm25.search(
        question,
        k,
    )


def hybrid_search(
    question: str,
    k: int,
):
    dense = rag._vector_search(
        question,
        k,
    )

    bm25 = rag.bm25.search(
        question,
        k,
    )

    fused = rag.rrf_fusion(
        dense,
        bm25,
    )

    return sorted(
        fused,
        key=lambda item: item.get(
            "rrf_score",
            0.0,
        ),
        reverse=True,
    )[:k]


def production_search(
    question: str,
    k: int,
):
    results, _best_score = rag.search(
        question,
        k=CONFIG.retrieval_k,
        final_k=k,
    )

    return results


RETRIEVERS: dict[
    str,
    Callable,
] = {
    "bm25": bm25_search,
    "dense": dense_search,
    "hybrid": hybrid_search,
    "production": production_search,
}


# ============================================================
# HELPERS
# ============================================================

def extract_ids(
    results: list[dict[str, Any]],
) -> list[str]:

    return normalize_ids(
        [
            result["chunk_id"]
            for result in results
            if "chunk_id" in result
        ]
    )


# ============================================================
# SINGLE QUERY
# ============================================================

def evaluate_query(
    item: dict[str, Any],
    mode: str,
) -> dict[str, Any]:

    question = item["question"]

    relevant = normalize_ids(
        item["relevant_chunks"]
    )

    retriever = RETRIEVERS[mode]

    start = time.perf_counter()

    results = retriever(
        question,
        CONFIG.retrieval_k,
    )

    latency_ms = (
        time.perf_counter()
        - start
    ) * 1000

    retrieved_ids = extract_ids(
        results
    )

    result = {
        "id": item["id"],
        "question": question,
        "question_type": item.get(
            "question_type"
        ),

        "relevant_chunks": relevant,

        "retrieved_chunks": retrieved_ids,

        "correct_chunks": list(
            set(retrieved_ids)
            & set(relevant)
        ),

        "latency_ms": latency_ms,

        "_results": results,
    }

    for k in CONFIG.evaluation_ks:

        result[f"recall@{k}"] = (
            recall_at_k(
                retrieved_ids,
                relevant,
                k,
            )
        )

        result[f"precision@{k}"] = (
            precision_at_k(
                retrieved_ids,
                relevant,
                k,
            )
        )

        result[f"hit@{k}"] = (
            hit_at_k(
                retrieved_ids,
                relevant,
                k,
            )
        )

        result[f"ndcg@{k}"] = (
            ndcg_at_k(
                retrieved_ids,
                relevant,
                k,
            )
        )

    result["mrr"] = reciprocal_rank(
        retrieved_ids,
        relevant,
    )

    return result


# ============================================================
# EVALUATE RETRIEVER
# ============================================================

def evaluate_retriever(
    dataset: list[dict[str, Any]],
    mode: str,
) -> tuple[
    dict[str, float],
    list[dict[str, Any]],
]:

    if mode not in RETRIEVERS:
        raise ValueError(
            f"Unknown retrieval mode: {mode}"
        )

    details = []

    for index, item in enumerate(dataset):

        print(
            f"[RETRIEVAL] "
            f"{mode}: "
            f"{index + 1}/{len(dataset)}"
        )

        try:

            details.append(
                evaluate_query(
                    item,
                    mode,
                )
            )

        except Exception as exc:

            details.append(
                {
                    "id": item.get("id"),
                    "question": item.get(
                        "question"
                    ),
                    "error": str(exc),
                }
            )

    valid = [
        result
        for result in details
        if "error" not in result
    ]

    if not valid:
        return {}, details

    metrics = {}

    for k in CONFIG.evaluation_ks:

        metrics[f"recall@{k}"] = float(
            np.mean(
                [
                    result[
                        f"recall@{k}"
                    ]
                    for result in valid
                ]
            )
        )

        metrics[f"precision@{k}"] = float(
            np.mean(
                [
                    result[
                        f"precision@{k}"
                    ]
                    for result in valid
                ]
            )
        )

        metrics[f"hit@{k}"] = float(
            np.mean(
                [
                    result[
                        f"hit@{k}"
                    ]
                    for result in valid
                ]
            )
        )

        metrics[f"ndcg@{k}"] = float(
            np.mean(
                [
                    result[
                        f"ndcg@{k}"
                    ]
                    for result in valid
                ]
            )
        )

    metrics["mrr"] = float(
        np.mean(
            [
                result["mrr"]
                for result in valid
            ]
        )
    )

    metrics["avg_latency_ms"] = float(
        np.mean(
            [
                result["latency_ms"]
                for result in valid
            ]
        )
    )

    metrics["questions"] = len(valid)

    metrics["failed_questions"] = (
        len(details) - len(valid)
    )

    return metrics, details