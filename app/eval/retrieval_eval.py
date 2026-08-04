
import json
import time
import numpy as np

from tqdm import tqdm

from app.services.rag import rag
# from flashrank import RerankRequest

# =====================================================
# CONFIG
# =====================================================



DATASET_PATH = "data/eval/qa_dataset.json"

RESULT_DIR = "data/eval/results"

# =====================================================
# LOAD DATASET
# =====================================================

def load_dataset(path=DATASET_PATH):

    with open(
        path,
        encoding="utf-8"
    ) as f:

        return json.load(f)


# =====================================================
# METRICS
# =====================================================


def recall_at_k(
        retrieved,
        expected,
        k
):

    retrieved = set(
        retrieved[:k]
    )

    expected = set(
        expected
    )

    if not expected:
        return 0


    return len(
        retrieved.intersection(expected)
    ) / len(expected)



def precision_at_k(
        retrieved,
        expected,
        k
):

    retrieved = retrieved[:k]

    if not retrieved:
        return 0


    correct = len(
        set(retrieved)
        &
        set(expected)
    )


    return correct / k



def hit_at_k(
        retrieved,
        expected,
        k
):

    return int(
        len(
            set(retrieved[:k])
            &
            set(expected)
        )
        > 0
    )



def mrr(
        retrieved,
        expected
):

    expected=set(expected)


    for rank,cid in enumerate(
        retrieved,
        start=1
    ):

        if cid in expected:

            return 1/rank


    return 0



def dcg_at_k(
        retrieved,
        expected,
        k
):

    score=0

    expected=set(expected)


    for i,cid in enumerate(
        retrieved[:k],
        start=1
    ):

        if cid in expected:

            score += 1 / np.log2(i+1)


    return score



def ndcg_at_k(
        retrieved,
        expected,
        k
):

    actual = dcg_at_k(
        retrieved,
        expected,
        k
    )


    ideal = dcg_at_k(
        expected,
        expected,
        k
    )


    if ideal == 0:
        return 0


    return actual / ideal
# =====================================================
# RETRIEVAL MODES
# =====================================================


def qdrant_search(
        question,
        k=20
):

    return rag._vector_search(
        question,
        k
    )



def bm25_search(
        question,
        k=20
):

    return rag.bm25.search(
        question,
        k
    )



def hybrid_search(
        question,
        k=20
):

    vector_results = rag._vector_search(
        question,
        k
    )


    bm25_results = rag.bm25.search(
        question,
        k
    )


    fused = rag.rrf_fusion(
        vector_results,
        bm25_results
    )


    fused = sorted(
        fused,
        key=lambda x:x["rrf_score"],
        reverse=True
    )


    return fused[:k]



def production_search(
        question
):

    results, best_score = rag.search(
        question,
        k=20,
        final_k=5
    )


    return results



# =====================================================
# RESULT EXTRACTION
# =====================================================


def extract_ids(results):

    return [
        r["chunk_id"]
        for r in results
    ]



# =====================================================
# EVALUATION
# =====================================================


def evaluate(
        dataset,
        mode
):


    print("\n")
    print("="*60)
    print("MODE:", mode)
    print("="*60)



    metrics = {

        "recall@1":[],
        "recall@3":[],
        "recall@5":[],
        "recall@10":[],

        "precision@5":[],

        "hit@5":[],

        "mrr":[],

        "ndcg@5":[],

        "latency":[]

    }


    query_results=[]



    for item in tqdm(dataset):


        question = item["question"]

        expected = item["expected_chunks"]



        start=time.time()



        # -----------------------------
        # SELECT RETRIEVER
        # -----------------------------


        if mode=="qdrant":

            results = qdrant_search(
                question
            )


        elif mode=="bm25":

            results = bm25_search(
                question
            )


        elif mode=="hybrid":

            results = hybrid_search(
                question
            )


        elif mode=="production":

            results = production_search(
                question
            )


        else:

            raise ValueError(
                f"Unknown mode {mode}"
            )



        latency = (
            time.time()-start
        ) * 1000



        ids = extract_ids(
            results
        )



        # -----------------------------
        # METRICS
        # -----------------------------


        metrics["recall@1"].append(
            recall_at_k(
                ids,
                expected,
                1
            )
        )


        metrics["recall@3"].append(
            recall_at_k(
                ids,
                expected,
                3
            )
        )


        metrics["recall@5"].append(
            recall_at_k(
                ids,
                expected,
                5
            )
        )


        metrics["recall@10"].append(
            recall_at_k(
                ids,
                expected,
                10
            )
        )


        metrics["precision@5"].append(
            precision_at_k(
                ids,
                expected,
                5
            )
        )


        metrics["hit@5"].append(
            hit_at_k(
                ids,
                expected,
                5
            )
        )


        metrics["mrr"].append(
            mrr(
                ids,
                expected
            )
        )


        metrics["ndcg@5"].append(
            ndcg_at_k(
                ids,
                expected,
                5
            )
        )


        metrics["latency"].append(
            latency
        )



        # -----------------------------
        # SAVE QUERY RESULT
        # -----------------------------


        query_results.append({

            "question": question,

            "expected_chunks": expected,

            "retrieved_chunks": ids,

            "correct_chunks":
                list(
                    set(ids)
                    &
                    set(expected)
                ),

            "latency_ms": latency

        })



    return metrics, query_results

# =====================================================
# PRINT RESULTS
# =====================================================


def print_results(
        mode,
        metrics
):

    print("\nRESULTS")
    print("-"*40)


    for key,value in metrics.items():

        print(
            f"{key}: {np.mean(value):.4f}"
        )



# =====================================================
# SAVE RESULTS
# =====================================================


def save_results(
        mode,
        metrics,
        queries
):

    import os

    os.makedirs(
        RESULT_DIR,
        exist_ok=True
    )


    summary = {

        key: float(
            np.mean(value)
        )

        for key,value in metrics.items()

    }


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
        f"{RESULT_DIR}/{mode}_queries.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            queries,
            f,
            indent=4,
            ensure_ascii=False
        )



# =====================================================
# COMPARE ALL MODES
# =====================================================


def compare_results(
        all_results
):

    print("\n")
    print("="*80)
    print("FINAL COMPARISON")
    print("="*80)


    headers = [

        "Mode",
        "Recall@5",
        "Hit@5",
        "MRR",
        "nDCG@5",
        "Latency(ms)"

    ]


    print(
        f"{headers[0]:15}"
        f"{headers[1]:12}"
        f"{headers[2]:12}"
        f"{headers[3]:12}"
        f"{headers[4]:12}"
        f"{headers[5]:15}"
    )


    print("-"*80)



    for mode,metrics in all_results.items():


        print(

            f"{mode:15}"

            f"{np.mean(metrics['recall@5']):<12.4f}"

            f"{np.mean(metrics['hit@5']):<12.4f}"

            f"{np.mean(metrics['mrr']):<12.4f}"

            f"{np.mean(metrics['ndcg@5']):<12.4f}"

            f"{np.mean(metrics['latency']):<15.2f}"

        )



# =====================================================
# MAIN
# =====================================================


if __name__ == "__main__":



    dataset = load_dataset()


    print(
        "Dataset size:",
        len(dataset)
    )


    # ---------------------------------
    # Load metadata
    # ---------------------------------

    DOC_PATH = "data/processed/chunks.json"


    rag.load_docs(
        DOC_PATH
    )


    # load BM25 index
    rag.bm25.load(
        DOC_PATH
    )

    # ---------------------------------
    # Qdrant must already exist
    # ---------------------------------

    if not rag.vector_db.collection_exists():

        print("Building Qdrant collection...")
    
        rag.build_index()



    modes = [

        "bm25",

        "qdrant",

        "hybrid",

        "production"

    ]



    all_results = {}



    for mode in modes:


        metrics, queries = evaluate(
            dataset,
            mode
        )


        print_results(
            mode,
            metrics
        )


        save_results(
            mode,
            metrics,
            queries
        )


        all_results[mode] = metrics



    compare_results(
        all_results
    )