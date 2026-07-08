from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os

from app.services.bm25_retriever import BM25Retriever
from flashrank import Ranker,RerankRequest
from concurrent.futures import ThreadPoolExecutor
import time
from app.services.vector_db import QdrantVectorDB

class FAISSRAG:

    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):
        self.model = SentenceTransformer(model_name)

          # 🔥 RERANKER ADDED
        self.reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

        self.docs = []
        self.metadata = []
        # self.index = None
        self.vector_db = QdrantVectorDB()

        self.bm25 = BM25Retriever()

        emb = self.model.encode(["hello world"])
        print("Embedding shape:", emb.shape)

    def load_docs(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.docs = [item["text"] for item in data]
        self.metadata = data
       
        # print(f"Loaded {len(self.docs)} clean chunks")

    # def build_index(self, index_path="vector_db/faiss.index"):
    #     self.bm25.load("data/processed/chunks.json")
    #     if os.path.exists(index_path):
    #         # self.index = faiss.read_index(index_path)
    #         print("Loaded FAISS index from disk")
           
            

    #         return

    #     print("Creating embeddings...")

    #     embeddings = self.model.encode(
    #         self.docs,
    #         normalize_embeddings=True,
    #         show_progress_bar=True
    #     )

    #     embeddings = np.array(embeddings).astype("float32")

    #     dim = embeddings.shape[1]
    #     # self.index = faiss.IndexFlatIP(dim)
    #     self.index.add(embeddings)

    #     os.makedirs("vector_db", exist_ok=True)
    #     # faiss.write_index(self.index, index_path)

       

    #     print("Built and saved FAISS index")

    def build_index(self):
        self.bm25.load(
        "data/processed/chunks.json"
    )
        if self.vector_db.collection_exists():
            print("Qdrant collection already exists.")
            return 


        embeddings = self.model.encode(
        self.docs,
        normalize_embeddings=True
    )

        embeddings = np.array(
        embeddings
    ).astype("float32")
        
        self.vector_db.create_collection(
        embeddings.shape[1]
    )


        self.vector_db.add_documents(
        embeddings,
        self.metadata
    )


        print("Qdrant index ready")
   
    def normalize(self,scores):
        mn = min(scores)
        mx = max(scores)

        if mx == mn:
            return [0]*len(scores)

        return [
            (x-mn)/(mx-mn)
            for x in scores
        ]
    
    def rrf_fusion(self,vector_results,bm25_results,k=60):
        fused ={}

        #FAISS ranking
        for rank, r in enumerate(vector_results):
            cid =r["chunk_id"]

            fused[cid]={
                "text": r["text"],
                "chunk_id": cid,
                "source": r["source"],
                "vector_score": r["vector_score"],
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
                    "vector_score": 0.0,
                    "bm25_score": r["bm25_score"],
                    "rrf_score": 1 / (k + rank + 1)
                }


        return list(fused.values())

    
    
    def _vector_search(self,query, k):

        q_vec = self.model.encode(
            [query],
            normalize_embeddings=True
        )[0]


        return self.vector_db.search(q_vec,k )
    
    def search(self, query, k=20, final_k=3):
        t0 = time.time()
        
        with ThreadPoolExecutor() as executor:

            vector_future= executor.submit(self._vector_search ,query, k)
            bm25_future = executor.submit(self.bm25.search,query ,k)

            vector_results = vector_future.result()
            bm25_results = bm25_future.result()
        t1 = time.time()
        print(f"[TIMING] Retrieval (Qdrant+BM25): {(t1 - t0)*1000:.2f} ms")
      
        
        # --------------------
        # RRF FUSION
        # --------------------

        candidates =self.rrf_fusion(
            vector_results,
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
            "vector_score": c["vector_score"],
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
                "vector_score": float(original["vector_score"]),
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