from sentence_transformers import SentenceTransformer,CrossEncoder
import faiss
import numpy as np
import json
import os
from app.services.bm25_retriever import BM25Retriever

class FAISSRAG:

    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):
        self.model = SentenceTransformer(model_name)

          # 🔥 RERANKER ADDED
        self.reranker = CrossEncoder("BAAI/bge-reranker-base")

        self.docs = []
        self.metadata = []
        self.index = None

        self.bm25 = BM25Retriever()

        emb = self.model.encode(["hello world"])
        print("Embedding shape:", emb.shape)

    def load_docs(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.docs = [item["text"] for item in data]
        self.metadata = data
       
        print(f"Loaded {len(self.docs)} clean chunks")

    def build_index(self, index_path="vector_db/faiss.index"):

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            print("Loaded FAISS index from disk")

            self.bm25.load("data/processed/chunks.json")

            return

        print("Creating embeddings...")

        embeddings = self.model.encode(
            self.docs,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        embeddings = np.array(embeddings).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        os.makedirs("vector_db", exist_ok=True)
        faiss.write_index(self.index, index_path)

        print("Built and saved FAISS index")

    # def search(self, query, k=20, final_k=3):  
    #     # -------------------------
    #     # 1. FAISS retrieval (FAST)
    #     # -------------------------
    #     q_vec = self.model.encode([query], normalize_embeddings=True)
    #     q_vec = np.array(q_vec).astype("float32")

    #     scores, indices = self.index.search(q_vec, k)

    #     candidates = []

    #     for idx, score in zip(indices[0], scores[0]):

    #         if idx == -1:
    #             continue

    #         candidates.append({
    #             "text": self.docs[idx],
    #             "source": self.metadata[idx]["source"],
    #             "faiss_score": float(score)
    #         })


    #     if not candidates:
    #         return [],0.0
        
    #     # -------------------------
    #     # 2. RERANKING (ACCURACY)
    #     # -------------------------
        
    #     pairs = [(query, c["text"]) for c in candidates]
    #     rerank_scores = self.reranker.predict(pairs)

    #      # attach rerank score
    #     for i in range(len(candidates)):
    #         candidates[i]["rerank_score"] = float(rerank_scores[i])

    #     # sort by reranker (IMPORTANT)
    #     candidates = sorted(
    #         candidates,
    #         key=lambda x: x["rerank_score"],
    #         reverse=True
    #     )

    #      # -------------------------
    #     # 3. FINAL FILTER
    #     # -------------------------
    #     top_results = candidates[:final_k]

    #     best_score = top_results[0]["rerank_score"]

    #     return top_results, best_score

    def normalize(self,scores):
        mn = min(scores)
        mx = max(scores)

        if mx == mn:
            return [0]*len(scores)

        return [
            (x-mn)/(mx-mn)
            for x in scores
        ]
    
    def search(self, query, k=20, final_k=3):
        
        # --------------------
        # 1. FAISS
        # --------------------
        q_vec = self.model.encode([query], normalize_embeddings=True)
        q_vec = np.array(q_vec).astype("float32")

        scores, indices = self.index.search(q_vec, k)

        faiss_results = []

        for idx, score in zip(indices[0], scores[0]):

            if idx == -1:
                continue

            faiss_results.append({
                "text": self.docs[idx],
                "source": self.metadata[idx]["source"],
                "faiss_score": float(score)
            })

        # --------------------
        # 2. BM25
        # --------------------
        bm25_results = self.bm25.search(query, k=k)
        # --------------------
        # NORMALIZE BM25
        # --------------------

        if bm25_results:
            bm_scores=[
                r["bm25_score"]
                for r in bm25_results
            ]
            normalized_scores = self.normalize(bm_scores)

            for r, score in zip(bm25_results, normalized_scores):
                r["bm25_score"] = score 


        # --------------------
        # 3. MERGE (HYBRID)
        # --------------------
        merged = {}

        for r in faiss_results:
            merged[r["text"]] = {
                **r,
                "hybrid_score":  0.7 * r["faiss_score"]
            }
        print("\n=== BM25 DEBUG ===")
        for r in bm25_results:
            print(r["text"][:80], " | score:", r["bm25_score"])

            if r["text"] in merged:
                merged[r["text"]]["hybrid_score"] += (
                    0.3 * r["bm25_score"]
                )
            else:
                merged[r["text"]] = {
                    **r,
                    "hybrid_score": 0.3 * r["bm25_score"]
                }

        candidates = list(merged.values())

        # hybrid ranking first
        candidates = sorted(
            candidates,
            key=lambda x: x["hybrid_score"],
            reverse=True
        )
        print("\n=== HYBRID DEBUG ===")
        for c in candidates[:5]:
            print(c["text"][:80], " | hybrid:", c["hybrid_score"])


        # --------------------
        # 4. RERANKER
        # --------------------
        pairs = [(query, c["text"]) for c in candidates]

        rerank_scores = self.reranker.predict(pairs, batch_size=32)

        for c, s in zip(candidates, rerank_scores):
            c["rerank_score"] = float(s)

        # sort by reranker
        candidates = sorted(
            candidates,
            key=lambda x: x["rerank_score"],
            reverse=True
        )
        print("\n=== RERANK DEBUG ===")
        for c in candidates[:5]:
            print(c["text"][:80], " | rerank:", c["rerank_score"])

        top = candidates[:final_k]

        best_score = top[0]["rerank_score"] if top else -999

        return top, best_score
    
rag = FAISSRAG()