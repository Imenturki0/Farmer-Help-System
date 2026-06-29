from sentence_transformers import SentenceTransformer,CrossEncoder
import faiss
import numpy as np
import json
import os
from app.services.bm25_retriever import BM25Retriever
from flashrank import Ranker,RerankRequest
from concurrent.futures import ThreadPoolExecutor
import time

class FAISSRAG:

    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):
        self.model = SentenceTransformer(model_name)

          # 🔥 RERANKER ADDED
        self.reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

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
       
        # print(f"Loaded {len(self.docs)} clean chunks")

    def build_index(self, index_path="vector_db/faiss.index"):
        self.bm25.load("data/processed/chunks.json")
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            print("Loaded FAISS index from disk")
           
            

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

   
    def normalize(self,scores):
        mn = min(scores)
        mx = max(scores)

        if mx == mn:
            return [0]*len(scores)

        return [
            (x-mn)/(mx-mn)
            for x in scores
        ]
    
    def rrf_fusion(self,faiss_results,bm25_results,k=60):
        fused ={}

        #FAISS ranking
        for rank, r in enumerate(faiss_results):
            cid =r["chunk_id"]

            fused[cid]={
                "text": r["text"],
                "chunk_id": cid,
                "source": r["source"],
                "faiss_score": r["faiss_score"],
                "bm25_score": 0.0,
                "rrf_score": 1 / (k + rank + 1)
            }
            # BM25 ranking
        for rank, r in enumerate(bm25_results):

            cid = r["chunk_id"]

            if cid in fused:

                fused[cid]["bm25_score"] = r["bm25_score"]

                fused[cid]["rrf_score"] += (
                    1 / (k + rank + 1)
                )

            else:

                fused[cid] = {
                    "text": r["text"],
                    "chunk_id": cid,
                    "source": r["source"],
                    "faiss_score": 0.0,
                    "bm25_score": r["bm25_score"],
                    "rrf_score": 1 / (k + rank + 1)
                }


        return list(fused.values())

    
    def _faiss_search(self,query,k):
        q_vec =self.model.encode([query],normalize_embeddings=True)
        q_vec= np.array(q_vec).astype("float32")

        scores, indices = self.index.search(q_vec,k)

        results=[]

        for idx, score in zip(indices[0], scores[0]):
            if idx== -1:
                continue
            results.append({
                "text": self.docs[idx],
                "chunk_id": self.metadata[idx]["chunk_id"],
                "source": self.metadata[idx]["source"],
                "faiss_score": float(score)
            })
        return results    
    

    def search(self, query, k=20, final_k=3):
        t0 = time.time()
        
        with ThreadPoolExecutor() as executor:

            faiss_future= executor.submit(self._faiss_search ,query, k)
            bm25_future = executor.submit(self.bm25.search,query ,k)

            faiss_results = faiss_future.result()
            bm25_results = bm25_future.result()
        t1 = time.time()
        print(f"[TIMING] Retrieval (FAISS+BM25): {(t1 - t0)*1000:.2f} ms")
      
        # # --------------------
        # # NORMALIZE BM25
        # # --------------------

        # if bm25_results:
        #     bm_scores=[
        #         r["bm25_score"]
        #         for r in bm25_results
        #     ]
        #     normalized_scores = self.normalize(bm_scores)

        #     for r, score in zip(bm25_results, normalized_scores):
        #         r["bm25_score"] = score 
            
        # t2 = time.time()
        # print(f"[TIMING] BM25 normalization: {(t2 - t1)*1000:.2f} ms")

        # # --------------------
        # #  3. HYBRID MERGE (FAISS + BM25)
        # # --------------------

        # all_candidates = {}

        # #1. FAISS results 
        # for r in faiss_results:
        #     cid = r["chunk_id"]
        #     all_candidates[cid] ={
        #         "text": r["text"],
        #         "chunk_id": cid,
        #         "source": r["source"],
        #         "faiss_score": r["faiss_score"],
        #         "bm25_score": 0.0
        #     }
        # # 2. BM25 results
        # for r in bm25_results:
        #     cid = r["chunk_id"]
        #     if r["text"] in all_candidates:
        #         all_candidates[cid]["bm25_score"]=r["bm25_score"]

        #     else:
        #         all_candidates[cid] = {
        #             "text": r["text"],
        #             "chunk_id": cid,
        #             "source": r["source"],
        #             "faiss_score": 0.0,
        #             "bm25_score": r["bm25_score"]
        #         }
       

        # # 4. FINAL CANDIDATES
        # candidates = list(all_candidates.values())   

        # # 3. HYBRID SCORE
        # for c in all_candidates.values():
        #     c["hybrid_score"] = (
        #         0.6 * c["faiss_score"] +
        #         0.4 * c["bm25_score"]
        #     )
        # # sort by hybrid score BEFORE reranker
        # candidates = sorted(
        #     candidates,
        #     key=lambda x: x["hybrid_score"],
        #     reverse=True
        # )
        # t3 = time.time()
        # print(f"[TIMING] Merge + hybrid scoring: {(t3 - t2)*1000:.2f} ms")

        # --------------------
        # RRF FUSION
        # --------------------

        candidates =self.rrf_fusion(
            faiss_results,
            bm25_results
        )
        candidates = sorted(
            candidates,
            key=lambda x: x["rrf_score"],
            reverse=True
        )
                
        docs = [
            {
            "id": i,
            "text": c["text"],
            "chunk_id": c["chunk_id"],
            "source": c["source"],
            "faiss_score": c["faiss_score"],
            "bm25_score": c["bm25_score"]
            }
            for i, c in enumerate(candidates)
        ]
        request = RerankRequest(
            query=query,
            passages=docs
        )

        t4 = time.time()

        results = self.reranker.rerank(request)

        t5 = time.time()
        print(f"[TIMING] FlashRank rerank: {(t5 - t4)*1000:.2f} ms")
  
        reranked_candidates = []
        for r in results:
            idx = r["id"]
            original = candidates[idx]
            reranked_candidates.append({
                "text": original["text"],
                "chunk_id": original["chunk_id"],
                "source": original["source"],
                "faiss_score": float(original["faiss_score"]),
                "bm25_score": float(original["bm25_score"]),
                "rerank_score": float(r["score"])
            })
        # sort by reranker
        reranked_candidates = sorted(
            reranked_candidates,
            key=lambda x: x["rerank_score"],
            reverse=True
        )
        t6 = time.time()
        print(f"[TIMING] Final formatting: {(t6 - t5)*1000:.2f} ms")

        print(f"[TOTAL TIME]: {(t6 - t0)*1000:.2f} ms")
        
        top = reranked_candidates[:final_k]

        best_score = top[0]["rerank_score"] if top else -999

        return top, best_score
    
rag = FAISSRAG()