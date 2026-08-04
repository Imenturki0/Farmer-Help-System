import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import NarrativeText, ListItem, Table
from app.services.llm import generate_answer
import requests
import json
# =========================
# MODEL
# =========================
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

OLLAMA_URL = "http://localhost:11434/api/generate"
CLASSIFIER_MODEL = "llama3.2:1b"
# =========================
# EXTRACTION WITH UNSTRUCTURED
# =========================
def extract_elements(pdf_path: str) -> list[dict]:
    """
    Use unstructured to extract only real content elements.
    It automatically handles:
      - Header/footer removal
      - Page numbers
      - Author blocks
      - Titles vs body text

    We keep: NarrativeText, ListItem, Table
    """

    elements = partition_pdf(
        filename=str(pdf_path),
        strategy="fast",
        include_page_breaks=False,
    )

    texts = []

    for el in elements:

        if not isinstance(el, (NarrativeText, ListItem, Table)):
            continue

        text = str(el).strip()

        if len(text.split()) < 15:
            continue

        texts.append(
            {
                "text": text,
                "page": el.metadata.page_number,
            }
        )

    return texts


# =========================
# EMBEDDINGS
# =========================
def embed(elements: list[dict]) -> np.ndarray:
    texts = [item["text"] for item in elements]

    return model.encode(
        texts,
        normalize_embeddings=True,
    )


# =========================
# SEMANTIC CHUNKING
# =========================
def chunk_texts(
    texts: list[dict],
    threshold: float = 0.78,
    max_words: int = 250,
) -> list[dict]:

    if not texts:
        return []

    embeddings = embed(texts)

    chunks = []

    current_texts = [texts[0]["text"]]
    current_pages = [texts[0]["page"]]

    centroid = embeddings[0]
    size = len(texts[0]["text"].split())

    for i in range(1, len(texts)):

        emb = embeddings[i]

        words = len(texts[i]["text"].split())

        similarity = float(np.dot(centroid, emb))

        if similarity < threshold or size + words > max_words:

            chunks.append(
                {
                    "text": " ".join(current_texts),
                    "pages": sorted(set(current_pages)),
                }
            )

            current_texts = [texts[i]["text"]]
            current_pages = [texts[i]["page"]]

            centroid = emb
            size = words

        else:

            current_texts.append(texts[i]["text"])
            current_pages.append(texts[i]["page"])

            centroid = (centroid * len(current_texts) + emb) / (
                len(current_texts) + 1
            )
            centroid /= np.linalg.norm(centroid) + 1e-8

            size += words

    if current_texts:

        chunks.append(
            {
                "text": " ".join(current_texts),
                "pages": sorted(set(current_pages)),
            }
        )

    return chunks


# =========================
# LLM CATEGORIZATION
# =========================
def categorize_with_llm(text: str) -> dict:

    prompt = f"""
You are a farming document classifier.

Given this text, identify:
1. The crop it talks about (e.g. tomato, wheat, sesame, cabbage, rice, corn)
2. The main topic

Return ONLY JSON:

{{
  "crop":"the crop name or general",
  "topic":"fertilizer|pest|irrigation|soil|harvest|composting|nutrients|disease|general"
}}

Text:
{text[:500]}
"""

    
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": CLASSIFIER_MODEL,
            "prompt": prompt,
            "stream": False,
            "temperature": 0,
        },
        timeout=120,
    )

    response.raise_for_status()

    result = response.json()["response"].strip()

    try:
        return json.loads(result)
    except json.JSONDecodeError:
        return {
            "crop": "general",
            "topic": "general",
        }


# =========================
# PIPELINE
# =========================
def process_pdf(pdf_path: str) -> list[dict]:

    print(f"  Extracting: {Path(pdf_path).name}")

    texts = extract_elements(pdf_path)

    if not texts:
        print(f"  WARNING: No content extracted from {pdf_path}")
        return []

    print(f"  Extracted {len(texts)} elements → chunking...")

    chunks = chunk_texts(texts)

    print(f"  Got {len(chunks)} chunks")

    return chunks


# =========================
# MAIN
# =========================
def main():

    pdf_folder = Path("data/raw/pdfs")

    output = []

    gid = 0
    skipped = 0

    for pdf in sorted(pdf_folder.glob("*.pdf")):

        print(f"\nProcessing: {pdf.name}")

        chunks = process_pdf(str(pdf))

        for chunk in chunks:

            metadata = categorize_with_llm(chunk["text"])

            output.append(
                {
                    "chunk_id": gid,
                    "text": chunk["text"],
                    "pages": chunk["pages"],
                    "source": pdf.name,
                    "crop": metadata["crop"],
                    "topic": metadata["topic"],
                }
            )

            gid += 1

    Path("data/processed").mkdir(exist_ok=True)

    with open(
        "data/processed/chunks.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nDone. Saved: {len(output)} chunks | Skipped (not farming): {skipped}"
    )


if __name__ == "__main__":
    main()