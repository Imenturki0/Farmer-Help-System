import json
import time
import os
import numpy as np
from tqdm import tqdm

from app.services.rag import rag
from app.services.llm import generate_answer


# =====================================================
# CONFIG
# =====================================================

DATASET_PATH = "data/eval/qa_dataset.json"

RESULT_DIR = "data/eval/generation_results"



# =====================================================
# LOAD DATASET
# =====================================================

def load_dataset():

    with open(
        DATASET_PATH,
        encoding="utf-8"
    ) as f:

        return json.load(f)



# =====================================================
# RETRIEVAL
# Hybrid Retrieval WITHOUT Reranker
# =====================================================


def retrieve_hybrid(
        question,
        k=10,
        final_k=5
):

    # Dense retrieval
    vector_results = rag._vector_search(
        question,
        k
    )


    # Keyword retrieval
    bm25_results = rag.bm25.search(
        question,
        k
    )


    # RRF fusion
    candidates = rag.rrf_fusion(
        vector_results,
        bm25_results
    )


    candidates = sorted(
        candidates,
        key=lambda x: x["rrf_score"],
        reverse=True
    )


    return candidates[:final_k]



# =====================================================
# CONTEXT
# =====================================================


def build_context(results):

    return "\n\n".join(
        r["text"]
        for r in results[:3]
    )



# =====================================================
# PROMPT
# =====================================================


def build_prompt(
        question,
        context
):

    return f"""

You are a farming assistant.

Rules:
- Answer only using the provided context.
- Do not hallucinate.
- If the context does not contain the answer, say you don't know.


Question:

{question}


Context:

{context}


Answer:

"""



# =====================================================
# ANSWER METRIC
# Simple lexical similarity
# =====================================================


def keyword_score(
        answer,
        ground_truth
):

    answer = answer.lower()

    truth = ground_truth.lower()


    words = truth.split()


    if not words:
        return 0


    matched = sum(
        1
        for w in words
        if w in answer
    )


    return matched / len(words)



# =====================================================
# EVALUATION
# =====================================================


def evaluate(
        dataset
):


    print()
    print("="*60)
    print("MODE: hybrid_without_reranker")
    print("="*60)


    details=[]

    scores=[]

    latencies=[]



    for item in tqdm(dataset):


        question = item["question"]

        ground_truth = item["ground_truth"]



        start=time.time()



        # -----------------------------
        # RETRIEVAL
        # -----------------------------

        docs = retrieve_hybrid(
            question
        )


        context = build_context(
            docs
        )


        # -----------------------------
        # GENERATION
        # -----------------------------

        prompt = build_prompt(
            question,
            context
        )


        answer = generate_answer(
            prompt
        )


        latency = (
            time.time()-start
        )*1000



        # -----------------------------
        # SCORE
        # -----------------------------

        score = keyword_score(
            answer,
            ground_truth
        )


        scores.append(score)

        latencies.append(latency)



        details.append({

            "question": question,

            "ground_truth": ground_truth,

            "answer": answer,

            "context": context,


            "retrieved_chunks":[

                r["chunk_id"]

                for r in docs

            ],


            "score": score,


            "latency_ms": latency

        })



    summary={


        "answer_score":

            float(
                np.mean(scores)
            ),



        "average_latency_ms":

            float(
                np.mean(latencies)
            )

    }



    return summary, details



# =====================================================
# SAVE RESULTS
# =====================================================


def save_results(
        summary,
        details
):

    os.makedirs(
        RESULT_DIR,
        exist_ok=True
    )


    with open(
        f"{RESULT_DIR}/hybrid_summary.json",
        "w",
        encoding="utf-8"
    ) as f:


        json.dump(
            summary,
            f,
            indent=4
        )



    with open(
        f"{RESULT_DIR}/hybrid_details.json",
        "w",
        encoding="utf-8"
    ) as f:


        json.dump(
            details,
            f,
            indent=4,
            ensure_ascii=False
        )



# =====================================================
# MAIN
# =====================================================


if __name__=="__main__":


    dataset = load_dataset()


    print(
        "Dataset size:",
        len(dataset)
    )


    DOC_PATH="data/processed/chunks.json"



    # Load documents

    rag.load_docs(
        DOC_PATH
    )


    # Load BM25

    rag.bm25.load(
        DOC_PATH
    )



    summary,details = evaluate(
        dataset
    )



    print("\nRESULTS")
    print("-"*40)


    print(
        "Answer score:",
        summary["answer_score"]
    )


    print(
        "Average latency:",
        summary["average_latency_ms"],
        "ms"
    )



    save_results(
        summary,
        details
    )