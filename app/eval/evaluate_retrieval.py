import json
import numpy as np
from tqdm import tqdm

from sentence_transformers import CrossEncoder

from app.services.rag import rag
from app.services.bm25_retriever import BM25Retriever
import time


# ==========================
# LOAD DATASET
# ==========================

def load_dataset(path="data/eval/qa_dataset.json"):

    with open(path,"r",encoding="utf-8") as f:
        return json.load(f)



# ==========================
# METRICS
# ==========================

def recall_at_k(ids, true_id, k):

    return int(true_id in ids[:k])


def mrr(ids, true_id):

    if true_id in ids:
        return 1/(ids.index(true_id)+1)

    return 0



# ==========================
# FAISS SEARCH ONLY
# ==========================

def faiss_search(question,k=20):

    q = rag.model.encode(
        [question],
        normalize_embeddings=True
    )

    q = np.array(q).astype("float32")


    scores,indices = rag.index.search(
        q,k
    )


    results=[]


    for idx,score in zip(
        indices[0],
        scores[0]
    ):

        results.append(
            {
                "chunk_id":
                    rag.metadata[idx]["chunk_id"],

                "text":
                    rag.metadata[idx]["text"],

                "score":
                    float(score)
            }
        )


    return results



# ==========================
# EVALUATION
# ==========================

def evaluate(dataset, mode):


    bm25=None
    reranker=None


    if "bm25" in mode:

        bm25=BM25Retriever()

        bm25.load(
            "data/processed/chunks.json"
        )


    if "reranker" in mode:

        reranker=CrossEncoder(
            "BAAI/bge-reranker-base"
        )



    recalls=[]
    mrrs=[]
    latencies = []


    for item in tqdm(dataset):


        question=item["question"]
        true_id=item["chunk_id"]



        # --------------------
        # FAISS
        # --------------------
        # ==========================
        # RETRIEVAL LATENCY START
        # ==========================
        t0 = time.time()

        faiss_results=faiss_search(
            question,
            k=20
        )


        results=faiss_results



        # --------------------
        # BM25 + FAISS
        # --------------------

        if "bm25" in mode:


            bm25_results=bm25.search(
                question,
                k=20
            )


            merged={}


            for r in faiss_results:

                merged[r["chunk_id"]]=r



            for r in bm25_results:

                if r["chunk_id"] not in merged:
                    merged[r["chunk_id"]]=r



            results=list(
                merged.values()
            )
       
        retrieval_time = time.time() - t0
        # --------------------
        # RERANKER
        # --------------------
        rerank_time = 0
        if "reranker" in mode:
            t1 = time.time()

            pairs=[
                (
                    question,
                    r["text"]
                )
                for r in results
            ]


            scores=reranker.predict(
                pairs,
                batch_size=8
            )


            for r,s in zip(
                results,
                scores
            ):
                r["score"]=float(s)


            results=sorted(
                results,
                key=lambda x:x["score"],
                reverse=True
            )
            rerank_time = time.time() - t1

        ids=[
            r["chunk_id"]
            for r in results
        ]


        recalls.append(
            recall_at_k(
                ids,
                true_id,
                5
            )
        )


        mrrs.append(
            mrr(
                ids,
                true_id
            )
        )

        # ==========================
        # STORE LATENCY (IMPORTANT)
        # ==========================
        latencies.append({
        "retrieval_time": retrieval_time,
        "rerank_time": rerank_time,
        "total_time": retrieval_time + rerank_time
    })


    print("\n===================")
    print(mode)
    print("===================")

    print(
        "Recall@5:",
        np.mean(recalls)
    )

    print(
        "MRR:",
        np.mean(mrrs)
    )
    print("\nLatency Results")
    print("Retrieval time:", np.mean([x["retrieval_time"] for x in latencies]))
    print("Rerank time:", np.mean([x["rerank_time"] for x in latencies]))
    print("Total time:", np.mean([x["total_time"] for x in latencies]))



# ==========================
# RUN
# ==========================


if __name__=="__main__":


    dataset=load_dataset()


    rag.load_docs(
        "data/processed/chunks.json"
    )

    rag.build_index()



    evaluate(
        dataset,
        "faiss"
    )


    evaluate(
        dataset,
        "faiss_bm25"
    )


    evaluate(
        dataset,
        "faiss_reranker"
    )


    evaluate(
        dataset,
        "faiss_bm25_reranker"
    )