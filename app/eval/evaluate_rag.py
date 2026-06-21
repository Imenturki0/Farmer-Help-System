import json
import numpy as np
import time
from tqdm import tqdm
from rouge_score import rouge_scorer

import faiss
from sentence_transformers import SentenceTransformer

from app.services.llm import generate_answer


# =========================
# LOAD DATASET
# =========================
def load_dataset(path="data/eval/qa_dataset.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# METRICS
# =========================
def recall_at_k(retrieved_ids, true_id, k):
    return int(true_id in retrieved_ids[:k])


def mrr(retrieved_ids, true_id):
    if true_id in retrieved_ids:
        return 1 / (retrieved_ids.index(true_id) + 1)
    return 0


# =========================
# BUILD INDEX (LOCAL ONLY)
# =========================
def build_faiss_index(chunks):
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    texts = [c["text"] for c in chunks]
    metadata = chunks

    embeddings = model.encode(texts, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return model, index, texts, metadata


# =========================
# SEARCH
# =========================
def search(query, model, index, texts, metadata, k=5):

    q_vec = model.encode([query], normalize_embeddings=True)
    q_vec = np.array(q_vec).astype("float32")

    scores, indices = index.search(q_vec, k)

    results = []

    for idx in indices[0]:
        if idx == -1:
            continue

        results.append({
            "text": texts[idx],
            "chunk_id": metadata[idx]["chunk_id"]
        })

    return results


# =========================
# PROMPT
# =========================
def build_prompt(question, contexts):
    context_text = "\n\n".join(contexts)

    return f"""
    You are a friendly farming assistant.

    RULES:
    - You are ONLY allowed to answer farming and agriculture-related questions
    - If the question is not about farming, politely say:
    "I can only help with farming-related questions."
    - Be friendly and helpful
    - Do NOT hallucinate or guess
    - Use context only if relevant

    If the user greets you (hi, hello), respond

    User Question:
    {question}


    Context:
    {context_text if context_text else "No relevant farming context found."}

    """


# =========================
# EVALUATION
# =========================
def evaluate(dataset):

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    retrieval_results = []
    generation_results = []

    # Load chunks
    with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print("Building FAISS index...")

    model, index, texts, metadata = build_faiss_index(chunks)

    print("Index ready ✔")

    for item in tqdm(dataset):

        question = item["question"]
        true_chunk = item["chunk_id"]
        gold = item["ground_truth"]

        # =====================
        # RETRIEVAL
        # =====================

        start = time.time()

        retrieved = search(question, model, index, texts, metadata, k=5)

        retrieval_time = time.time() - start

        retrieved_ids = [r["chunk_id"] for r in retrieved]
        contexts = [r["text"] for r in retrieved]

        # metrics
        retrieval_results.append({
            "recall@1": recall_at_k(retrieved_ids, true_chunk, 1),
            "recall@3": recall_at_k(retrieved_ids, true_chunk, 3),
            "recall@5": recall_at_k(retrieved_ids, true_chunk, 5),
            "mrr": mrr(retrieved_ids, true_chunk),
            "time": retrieval_time
        })

        # =====================
        # GENERATION
        # =====================

        prompt = build_prompt(question, contexts[:3])

        answer = generate_answer(prompt)

        generation_results.append({
            "rouge": rouge.score(answer, gold)["rougeL"].fmeasure,
            "exact_match": int(answer.strip().lower() == gold.strip().lower())
        })

    # =====================
    # FINAL REPORT
    # =====================

    print("\n===== RETRIEVAL =====")
    print("Recall@1:", np.mean([x["recall@1"] for x in retrieval_results]))
    print("Recall@3:", np.mean([x["recall@3"] for x in retrieval_results]))
    print("Recall@5:", np.mean([x["recall@5"] for x in retrieval_results]))
    print("MRR:", np.mean([x["mrr"] for x in retrieval_results]))
    print("Avg time:", np.mean([x["time"] for x in retrieval_results]))

    print("\n===== GENERATION =====")
    print("ROUGE-L:", np.mean([x["rouge"] for x in generation_results]))
    print("Exact match:", np.mean([x["exact_match"] for x in generation_results]))


# =========================
# RUN
# =========================
if __name__ == "__main__":

    dataset = load_dataset()

    print("Samples:", len(dataset))

    evaluate(dataset)