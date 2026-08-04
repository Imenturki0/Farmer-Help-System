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
# =====================================================


# -------------------------------
# WITHOUT RERANKER
# -------------------------------

def retrieve_without_reranker(
        question,
        k=10,
        final_k=5
):

    vector_results = rag._vector_search(
        question,
        k
    )


    bm25_results = rag.bm25.search(
        question,
        k
    )


    candidates = rag.rrf_fusion(
        vector_results,
        bm25_results
    )


    candidates = sorted(
        candidates,
        key=lambda x:x["rrf_score"],
        reverse=True
    )


    return candidates[:final_k]



# -------------------------------
# WITH RERANKER
# -------------------------------


def retrieve_with_reranker(
        question,
        k=10,
        final_k=5
):

    results, score = rag.search(
        question,
        k=k,
        final_k=final_k
    )

    return results



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
- Answer only using the context.
- Do not invent information.
- If context is insufficient say you don't know.

Question:

{question}


Context:

{context}

Answer:

"""



# =====================================================
# SIMPLE ANSWER METRICS
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


def evaluate_mode(
        dataset,
        mode
):

    print()
    print("="*60)
    print("MODE:", mode)
    print("="*60)


    results=[]


    scores=[]

    latencies=[]


    for item in tqdm(dataset):


        question = item["question"]

        ground_truth = item["ground_truth"]



        start=time.time()


        # --------------------------
        # RETRIEVAL
        # --------------------------

        if mode=="without_reranker":

            docs = retrieve_without_reranker(
                question
            )


        elif mode=="with_reranker":

            docs = retrieve_with_reranker(
                question
            )


        else:

            raise ValueError(mode)



        context = build_context(
            docs
        )


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


        score = keyword_score(
            answer,
            ground_truth
        )


        scores.append(
            score
        )


        latencies.append(
            latency
        )


        results.append({

            "question":question,

            "ground_truth":ground_truth,

            "answer":answer,
            
            "context": context,   

            "retrieved_chunks":[
                r["chunk_id"]
                for r in docs
            ],

            "score":score,

            "latency_ms":latency

        })


    summary={

        "answer_score":
            float(np.mean(scores)),

        "average_latency_ms":
            float(np.mean(latencies))

    }


    return summary, results



# =====================================================
# SAVE
# =====================================================


def save_results(
        mode,
        summary,
        details
):

    os.makedirs(
        RESULT_DIR,
        exist_ok=True
    )


    with open(
        f"{RESULT_DIR}/{mode}_summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )


    with open(
        f"{RESULT_DIR}/{mode}_details.json",
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
# COMPARE
# =====================================================


def compare(all_results):

    print()
    print("="*70)
    print("FINAL COMPARISON")
    print("="*70)


    print(
        f"{'Mode':20}"
        f"{'Answer Score':20}"
        f"{'Latency(ms)':20}"
    )


    print("-"*70)


    for mode,result in all_results.items():

        print(

            f"{mode:20}"

            f"{result['answer_score']:<20.4f}"

            f"{result['average_latency_ms']:<20.2f}"

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


    rag.load_docs(
        DOC_PATH
    )


    rag.bm25.load(
        DOC_PATH
    )


    modes=[

        "without_reranker",

        "with_reranker"

    ]


    all_results={}


    for mode in modes:


        summary,details = evaluate_mode(
            dataset,
            mode
        )


        print(summary)


        save_results(
            mode,
            summary,
            details
        )


        all_results[mode]=summary



    compare(
        all_results
    )