import json
import time
import os
import re
import numpy as np

from tqdm import tqdm

from app.services.rag import rag
from app.services.llm import generate_answer


# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = "data/eval/qa_dataset.json"

RESULT_DIR = "data/eval/generation_results"

DOC_PATH = "data/processed/chunks.json"

TOP_K = 20
FINAL_K = 5
CONTEXT_K = 3


# ============================================================
# DATASET
# ============================================================

def load_dataset(path=DATASET_PATH):

    with open(
        path,
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# PRODUCTION RETRIEVAL
# ============================================================

def retrieve_production(question):

    start = time.perf_counter()

    results, best_score = rag.search(
        question,
        k=TOP_K,
        final_k=FINAL_K
    )

    latency_ms = (
        time.perf_counter() - start
    ) * 1000

    return results, best_score, latency_ms


# ============================================================
# CONTEXT
# ============================================================

def build_context(results):

    return "\n\n".join(
        r["text"]
        for r in results[:CONTEXT_K]
    )


# ============================================================
# GENERATION PROMPT
# ============================================================

def build_prompt(
        question,
        context
):

    return f"""
You are a farming assistant.

Answer the user's question using ONLY the provided context.

Rules:
- Do not use outside knowledge.
- Do not invent facts.
- If the context does not contain enough information, say that you do not have enough information.
- Give a concise and useful answer.
- When the context contains a specific quantity, rate, date, or recommendation, preserve it accurately.

Question:
{question}

Context:
{context}

Answer:
"""


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# LEXICAL OVERLAP
# ============================================================

def lexical_f1(
        answer,
        ground_truth
):

    answer_tokens = set(
        normalize_text(answer).split()
    )

    truth_tokens = set(
        normalize_text(ground_truth).split()
    )

    if not answer_tokens or not truth_tokens:
        return 0.0

    common = (
        answer_tokens
        &
        truth_tokens
    )

    precision = len(common) / len(answer_tokens)

    recall = len(common) / len(truth_tokens)

    if precision + recall == 0:
        return 0.0

    return (
        2 * precision * recall
        /
        (precision + recall)
    )


# ============================================================
# LLM JUDGE
# ============================================================

def evaluate_with_llm(
        question,
        ground_truth,
        answer,
        context
):

    prompt = f"""
You are evaluating a RAG-based farming assistant.

Evaluate the generated answer using ONLY the information
provided below.

QUESTION:
{question}

REFERENCE ANSWER:
{ground_truth}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}

Evaluate three dimensions.

1. CORRECTNESS
Does the generated answer correctly answer the question
and agree with the reference answer?

2. FAITHFULNESS
Are the claims in the generated answer supported by
the retrieved context?

3. RELEVANCE
Does the answer directly address the user's question
without unnecessary information?

Give each score from 0 to 1.

0 = completely wrong
0.25 = mostly wrong
0.5 = partially correct
0.75 = mostly correct
1 = fully correct

Return EXACTLY:

CORRECTNESS: <score>
FAITHFULNESS: <score>
RELEVANCE: <score>

Do not include any explanation.
"""

    response = generate_answer(prompt)

    correctness = 0.0
    faithfulness = 0.0
    relevance = 0.0

    patterns = {
        "correctness": r"CORRECTNESS:\s*([01](?:\.\d+)?)",
        "faithfulness": r"FAITHFULNESS:\s*([01](?:\.\d+)?)",
        "relevance": r"RELEVANCE:\s*([01](?:\.\d+)?)"
    }

    for key, pattern in patterns.items():

        match = re.search(
            pattern,
            response,
            re.IGNORECASE
        )

        if match:

            value = float(
                match.group(1)
            )

            value = max(
                0.0,
                min(1.0, value)
            )

            if key == "correctness":
                correctness = value

            elif key == "faithfulness":
                faithfulness = value

            elif key == "relevance":
                relevance = value

    return {
        "correctness": correctness,
        "faithfulness": faithfulness,
        "relevance": relevance
    }


# ============================================================
# ANSWER SUPPORT CHECK
# ============================================================

def context_coverage(
        answer,
        context
):

    """
    Simple additional signal.

    Measures how much of the generated answer's
    content overlaps with retrieved context.

    This is NOT the main faithfulness metric.
    """

    answer_words = set(
        normalize_text(answer).split()
    )

    context_words = set(
        normalize_text(context).split()
    )

    if not answer_words:
        return 0.0

    return len(
        answer_words & context_words
    ) / len(answer_words)


# ============================================================
# SINGLE QUESTION
# ============================================================

def evaluate_question(item):

    question = item["question"]

    ground_truth = item["ground_truth"]

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    retrieval_start = time.perf_counter()

    results, best_score = rag.search(
        question,
        k=TOP_K,
        final_k=FINAL_K
    )

    retrieval_latency = (
        time.perf_counter()
        -
        retrieval_start
    ) * 1000

    # --------------------------------------------------------
    # CONTEXT
    # --------------------------------------------------------

    context = build_context(
        results
    )

    retrieved_ids = [
        r["chunk_id"]
        for r in results
    ]

    expected_ids = item[
        "expected_chunks"
    ]

    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------

    prompt = build_prompt(
        question,
        context
    )

    generation_start = time.perf_counter()

    answer = generate_answer(
        prompt
    )

    generation_latency = (
        time.perf_counter()
        -
        generation_start
    ) * 1000

    total_latency = (
        retrieval_latency
        +
        generation_latency
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    lexical_score = lexical_f1(
        answer,
        ground_truth
    )

    coverage = context_coverage(
        answer,
        context
    )

    judge_scores = evaluate_with_llm(
        question,
        ground_truth,
        answer,
        context
    )

    # --------------------------------------------------------
    # RETRIEVAL HIT
    # --------------------------------------------------------

    retrieved_top5 = set(
        retrieved_ids[:5]
    )

    expected_set = set(
        expected_ids
    )

    hit_at_5 = int(
        bool(
            retrieved_top5
            &
            expected_set
        )
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    return {

        "id": item.get("id"),

        "topic": item.get(
            "topic",
            "unknown"
        ),

        "question_type": item.get(
            "question_type",
            "unknown"
        ),

        "question": question,

        "ground_truth": ground_truth,

        "answer": answer,

        "expected_chunks": expected_ids,

        "retrieved_chunks": retrieved_ids,

        "correct_retrieved_chunks": list(
            set(retrieved_ids)
            &
            set(expected_ids)
        ),

        "best_rerank_score": float(
            best_score
        ),

        # -------------------------
        # Answer metrics
        # -------------------------

        "correctness": judge_scores[
            "correctness"
        ],

        "faithfulness": judge_scores[
            "faithfulness"
        ],

        "relevance": judge_scores[
            "relevance"
        ],

        "lexical_f1": lexical_score,

        "context_coverage": coverage,

        # -------------------------
        # Retrieval
        # -------------------------

        "retrieval_hit@5": hit_at_5,

        # -------------------------
        # Latency
        # -------------------------

        "retrieval_latency_ms":
            retrieval_latency,

        "generation_latency_ms":
            generation_latency,

        "total_latency_ms":
            total_latency,

        # -------------------------
        # Context
        # -------------------------

        "context": context
    }


# ============================================================
# SUMMARY
# ============================================================

def calculate_summary(
        details
):

    if not details:

        return {}

    def avg(key):

        values = [
            d[key]
            for d in details
        ]

        return float(
            np.mean(values)
        )

    summary = {

        # -------------------------
        # Answer quality
        # -------------------------

        "correctness":
            avg("correctness"),

        "faithfulness":
            avg("faithfulness"),

        "relevance":
            avg("relevance"),

        "lexical_f1":
            avg("lexical_f1"),

        "context_coverage":
            avg("context_coverage"),

        # -------------------------
        # Retrieval
        # -------------------------

        "retrieval_hit@5":
            avg("retrieval_hit@5"),

        # -------------------------
        # Latency
        # -------------------------

        "average_retrieval_latency_ms":
            avg(
                "retrieval_latency_ms"
            ),

        "average_generation_latency_ms":
            avg(
                "generation_latency_ms"
            ),

        "average_total_latency_ms":
            avg(
                "total_latency_ms"
            ),

        # -------------------------
        # Dataset
        # -------------------------

        "num_questions":
            len(details)
    }

    return summary


# ============================================================
# TOPIC BREAKDOWN
# ============================================================

def calculate_topic_results(
        details
):

    topics = {}

    for item in details:

        topic = item.get(
            "topic",
            "unknown"
        )

        if topic not in topics:
            topics[topic] = []

        topics[topic].append(
            item
        )

    results = {}

    for topic, items in topics.items():

        def avg(key):

            return float(
                np.mean(
                    [
                        x[key]
                        for x in items
                    ]
                )
            )

        results[topic] = {

            "questions":
                len(items),

            "correctness":
                avg("correctness"),

            "faithfulness":
                avg("faithfulness"),

            "relevance":
                avg("relevance"),

            "lexical_f1":
                avg("lexical_f1"),

            "latency_ms":
                avg("total_latency_ms")
        }

    return results


# ============================================================
# QUESTION TYPE BREAKDOWN
# ============================================================

def calculate_question_type_results(
        details
):

    types = {}

    for item in details:

        question_type = item.get(
            "question_type",
            "unknown"
        )

        if question_type not in types:
            types[question_type] = []

        types[question_type].append(
            item
        )

    results = {}

    for question_type, items in types.items():

        def avg(key):

            return float(
                np.mean(
                    [
                        x[key]
                        for x in items
                    ]
                )
            )

        results[question_type] = {

            "questions":
                len(items),

            "correctness":
                avg("correctness"),

            "faithfulness":
                avg("faithfulness"),

            "relevance":
                avg("relevance"),

            "latency_ms":
                avg("total_latency_ms")
        }

    return results


# ============================================================
# SAVE
# ============================================================

def save_results(
        summary,
        details,
        topics,
        question_types
):

    os.makedirs(
        RESULT_DIR,
        exist_ok=True
    )

    # Main summary

    with open(
        f"{RESULT_DIR}/production_summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    # Per-question results

    with open(
        f"{RESULT_DIR}/production_details.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            details,
            f,
            indent=4,
            ensure_ascii=False
        )

    # Topic results

    with open(
        f"{RESULT_DIR}/by_topic.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            topics,
            f,
            indent=4,
            ensure_ascii=False
        )

    # Question type results

    with open(
        f"{RESULT_DIR}/by_question_type.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            question_types,
            f,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# PRINT
# ============================================================

def print_results(
        summary
):

    print()
    print("=" * 70)
    print("PRODUCTION GENERATION EVALUATION")
    print("=" * 70)

    print(
        f"Questions: "
        f"{summary['num_questions']}"
    )

    print()

    print(
        f"Correctness: "
        f"{summary['correctness']:.4f}"
    )

    print(
        f"Faithfulness: "
        f"{summary['faithfulness']:.4f}"
    )

    print(
        f"Relevance: "
        f"{summary['relevance']:.4f}"
    )

    print(
        f"Lexical F1: "
        f"{summary['lexical_f1']:.4f}"
    )

    print(
        f"Context coverage: "
        f"{summary['context_coverage']:.4f}"
    )

    print()

    print(
        f"Retrieval Hit@5: "
        f"{summary['retrieval_hit@5']:.4f}"
    )

    print()

    print(
        f"Retrieval latency: "
        f"{summary['average_retrieval_latency_ms']:.2f} ms"
    )

    print(
        f"Generation latency: "
        f"{summary['average_generation_latency_ms']:.2f} ms"
    )

    print(
        f"Total latency: "
        f"{summary['average_total_latency_ms']:.2f} ms"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    dataset = load_dataset()

    print(
        "Dataset size:",
        len(dataset)
    )

    # --------------------------------------------------------
    # LOAD DOCUMENTS
    # --------------------------------------------------------

    rag.load_docs(
        DOC_PATH
    )

    # --------------------------------------------------------
    # LOAD BM25
    # --------------------------------------------------------

    rag.bm25.load(
        DOC_PATH
    )

    # --------------------------------------------------------
    # QDRANT
    # --------------------------------------------------------

    if not rag.vector_db.collection_exists():

        print(
            "Building Qdrant collection..."
        )

        rag.build_index()

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    details = []

    for item in tqdm(
        dataset,
        desc="Evaluating answers"
    ):

        try:

            result = evaluate_question(
                item
            )

            details.append(
                result
            )

        except Exception as e:

            print(
                f"\nEvaluation failed for "
                f"question {item.get('id')}: {e}"
            )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = calculate_summary(
        details
    )

    topics = calculate_topic_results(
        details
    )

    question_types = (
        calculate_question_type_results(
            details
        )
    )

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print_results(
        summary
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_results(
        summary,
        details,
        topics,
        question_types
    )

    print()
    print(
        "Saved results to:",
        RESULT_DIR
    )