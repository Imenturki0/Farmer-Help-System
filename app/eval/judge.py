from __future__ import annotations

import time
from typing import Any

from rouge_score import rouge_scorer

from app.services.rag import rag
from app.services.llm import generate_answer

from app.eval.config import CONFIG
from app.eval.metrics import (
    token_f1,
    exact_match,
)


def build_context(
    results: list[dict[str, Any]],
) -> list[str]:

    return [
        result["text"]
        for result in results[:CONFIG.context_k]
    ]


def build_prompt(
    question: str,
    contexts: list[str],
) -> str:

    context = "\n\n".join(contexts)

    return f"""
You are a farming assistant.

Answer the user's question using ONLY the provided context.

Rules:
- Do not use outside knowledge.
- Do not invent facts.
- If the context does not contain enough information, say that you do not have enough information.
- Give a concise and useful answer.
- Preserve quantities, rates, dates, and recommendations accurately.

Question:
{question}

Context:
{context}

Answer:
"""


def evaluate_generation(
    item: dict[str, Any],
    retrieval_result: dict[str, Any],
) -> dict[str, Any]:

    question = item["question"]

    ground_truth = item["ground_truth"]

    # We need the actual retrieval objects.
    retrieved_ids = retrieval_result["retrieved_chunks"]

    # Get actual documents corresponding to retrieved IDs.
    # The production retriever result is stored by the runner.
    results = retrieval_result["_results"]

    contexts = build_context(results)

    prompt = build_prompt(
        question,
        contexts,
    )

    start = time.perf_counter()

    answer = generate_answer(prompt)

    latency_ms = (
        time.perf_counter() - start
    ) * 1000

    scorer = rouge_scorer.RougeScorer(
        ["rougeL"],
        use_stemmer=True,
    )

    rouge_l = scorer.score(
        ground_truth,
        answer,
    )["rougeL"].fmeasure

    return {
        "id": item["id"],
        "question": question,

        "topic": item.get("topic"),
        "question_type": item.get("question_type"),

        "ground_truth": ground_truth,
        "answer": answer,

        "retrieved_chunks": retrieved_ids,
        "context": contexts,

        "rouge_l": float(rouge_l),

        "token_f1": token_f1(
            answer,
            ground_truth,
        ),

        "exact_match": exact_match(
            answer,
            ground_truth,
        ),

        "generation_latency_ms": latency_ms,
    }