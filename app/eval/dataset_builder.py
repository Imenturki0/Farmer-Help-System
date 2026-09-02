from __future__ import annotations

import json
import random
from collections import defaultdict
from typing import Any

from app.services.llm import generate_answer
from app.eval.config import CONFIG


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks() -> list[dict[str, Any]]:
    """Load canonical chunks produced by ingestion."""

    if not CONFIG.chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CONFIG.chunks_path}"
        )

    with CONFIG.chunks_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "chunks.json must contain a list."
        )

    return data


# ============================================================
# QUALITY FILTER
# ============================================================

def is_good_chunk(
    chunk: dict[str, Any],
) -> bool:
    """
    Filter obviously bad chunks from evaluation dataset creation.

    This does NOT modify production chunks.
    It only prevents poor chunks from being used to
    generate evaluation questions.
    """

    text = chunk.get("text", "")

    if not isinstance(text, str):
        return False

    words = text.split()

    if len(words) < CONFIG.min_eval_chunk_words:
        return False

    if len(words) > CONFIG.max_eval_chunk_words:
        return False

    alpha_ratio = (
        sum(character.isalpha() for character in text)
        / max(len(text), 1)
    )

    return alpha_ratio >= 0.60


# ============================================================
# QUESTION TYPE
# ============================================================

def classify_question(
    question: str,
) -> str:
    """Classify generated questions into broad evaluation types."""

    q = question.lower()

    if any(
        word in q
        for word in (
            "why",
            "cause",
            "reason",
        )
    ):
        return "explanation"

    if any(
        word in q
        for word in (
            "symptom",
            "disease",
            "infection",
            "pest",
        )
    ):
        return "diagnosis"

    if any(
        word in q
        for word in (
            "recommend",
            "should",
            "best",
            "how to",
        )
    ):
        return "recommendation"

    return "factual"


# ============================================================
# DOCUMENT GROUPING
# ============================================================

def group_by_source(
    chunks: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group chunks by source document."""

    documents = defaultdict(list)

    for chunk in chunks:

        source = chunk.get(
            "source",
            "unknown",
        )

        documents[source].append(chunk)

    return documents


# ============================================================
# CHUNK ORDER
# ============================================================

def chunk_sort_key(
    chunk: dict[str, Any],
):
    """
    Sort chunks in document order.

    Your chunk IDs are strings such as:
        document.pdf:10

    Therefore we should NOT do int(chunk_id).
    """

    pages = chunk.get("pages", [])

    first_page = (
        min(pages)
        if pages
        else 0
    )

    chunk_id = str(
        chunk.get(
            "chunk_id",
            "",
        )
    )

    try:
        index = int(
            chunk_id.rsplit(":", 1)[1]
        )
    except (
        ValueError,
        IndexError,
    ):
        index = 0

    return first_page, index


# ============================================================
# CONTEXT WINDOWS
# ============================================================

def create_context_windows(
    chunks: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """
    Create contiguous context windows.

    Example:

        chunk 10
        chunk 11
        chunk 12

    becomes one evaluation context.

    We do not randomly combine unrelated chunks.
    """

    documents = group_by_source(chunks)

    windows = []

    size = CONFIG.context_window_size

    for source, doc_chunks in documents.items():

        doc_chunks.sort(
            key=chunk_sort_key
        )

        if len(doc_chunks) < size:
            continue

        for index in range(
            len(doc_chunks) - size + 1
        ):

            window = doc_chunks[
                index:index + size
            ]

            windows.append(window)

    return windows


# ============================================================
# GENERATE QA
# ============================================================

def generate_qa(
    context_chunks: list[dict[str, Any]],
) -> tuple[str | None, str | None]:

    context = "\n\n".join(
        (
            f"CHUNK ID: {chunk['chunk_id']}\n"
            f"SECTION: {chunk.get('section')}\n"
            f"TEXT:\n{chunk['text']}"
        )
        for chunk in context_chunks
    )

    prompt = f"""
You are creating a high-quality evaluation dataset
for a farming RAG system.

Use ONLY the information contained in the context.

Create ONE realistic question that a farmer could ask.

The question should require information from at least
TWO different chunks.

The question must be answerable from the provided context.

The answer must:
- use ONLY the provided context
- contain no outside knowledge
- preserve important quantities, units, dates,
  ingredients, procedures, and recommendations
- be concise but complete

Do NOT mention:
- chunks
- documents
- context
- evaluation
- the dataset

Return exactly:

QUESTION:
<question>

ANSWER:
<answer>

CONTEXT:
{context}
"""

    response = generate_answer(prompt)

    if not response:
        return None, None

    if "QUESTION:" not in response:
        return None, None

    if "ANSWER:" not in response:
        return None, None

    question = (
        response
        .split("QUESTION:", 1)[1]
        .split("ANSWER:", 1)[0]
        .strip()
    )

    answer = (
        response
        .split("ANSWER:", 1)[1]
        .split("CONTEXT:", 1)[0]
        .strip()
    )

    if not question or not answer:
        return None, None

    return question, answer


# ============================================================
# VALIDATE GENERATED QA
# ============================================================

def is_valid_qa(
    question: str,
    answer: str,
) -> bool:
    """Basic validation of generated evaluation examples."""

    if len(question.split()) < 5:
        return False

    if len(answer.split()) < 5:
        return False

    return True


# ============================================================
# BUILD DATASET
# ============================================================

def build_dataset() -> None:
    """
    Build the RAG evaluation dataset.

    Pipeline:

        chunks.json
            ↓
        quality filtering
            ↓
        contiguous document windows
            ↓
        LLM-generated QA
            ↓
        qa_dataset.json
    """

    print(
        "\n=============================================="
    )
    print(
        "[EVAL DATASET] Building evaluation dataset"
    )
    print(
        "=============================================="
    )

    chunks = load_chunks()

    print(
        f"[EVAL DATASET] Loaded chunks: {len(chunks)}"
    )

    good_chunks = [
        chunk
        for chunk in chunks
        if is_good_chunk(chunk)
    ]

    print(
        f"[EVAL DATASET] Usable chunks: "
        f"{len(good_chunks)}"
    )

    windows = create_context_windows(
        good_chunks
    )

    print(
        f"[EVAL DATASET] Context windows: "
        f"{len(windows)}"
    )

    random.shuffle(windows)

    dataset = []

    source_counts = defaultdict(int)

    for window in windows:

        if len(dataset) >= CONFIG.max_total_samples:
            break

        source = window[0].get(
            "source",
            "unknown",
        )

        # Avoid generating too many questions
        # from one document.
        if (
            source_counts[source]
            >= CONFIG.samples_per_source
        ):
            continue

        print(
            f"[EVAL DATASET] Generating "
            f"{len(dataset) + 1}/"
            f"{CONFIG.max_total_samples}"
        )

        question, answer = generate_qa(
            window
        )

        if not question or not answer:
            continue

        if not is_valid_qa(
            question,
            answer,
        ):
            continue

        dataset.append(
            {
                "id": f"eval_{len(dataset):05d}",

                "question": question,

                "question_type":
                    classify_question(
                        question
                    ),

                "relevant_chunks": [
                    chunk["chunk_id"]
                    for chunk in window
                ],

                "ground_truth": answer,

                "reference_context": [
                    chunk["text"]
                    for chunk in window
                ],

                "sources": list(
                    dict.fromkeys(
                        chunk["source"]
                        for chunk in window
                    )
                ],
            }
        )

        source_counts[source] += 1

    CONFIG.dataset_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with CONFIG.dataset_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            dataset,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "\n=============================================="
    )
    print(
        "[EVAL DATASET] Complete"
    )
    print(
        f"[EVAL DATASET] Created: {len(dataset)}"
    )
    print(
        f"[EVAL DATASET] Saved: {CONFIG.dataset_path}"
    )
    print(
        "=============================================="
    )


if __name__ == "__main__":
    build_dataset()