from sentence_transformers import SentenceTransformer
import numpy as np
import json
import time
from typing import Tuple, List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from app.services.bm25_retriever import BM25Retriever
from app.services.vector_db import QdrantVectorDB
from app.core.logger import RequestLogger
from app.config.settings import settings

class ProductionRAGPipeline:
    """
    Production-grade RAG with:
    - Full instrumentation
    - Error handling & graceful degradation
    - Quality metrics
    - Request tracing
    """
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.rag.embedding_model
        self.logger = logging.getLogger(__name__)
        
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(f"Embedding model failed to load: {e}")
        
     
        self.vector_db = QdrantVectorDB()
        self.bm25 = BM25Retriever()
        
        # Embedding cache for queries
        self.embedding_cache = {}
        
        # Load existing BM25 index
        self.bm25.load(
            "data/processed/chunks.json"
        )

        self.logger.info(
            "BM25 index loaded"
        )
        self.logger.info(f"RAG Pipeline initialized with model: {self.model_name}")
    
    
    def _vector_search(self, query: str, k: int, request_logger: Optional[RequestLogger] = None) -> List[Dict]:
        """
        Dense vector search with caching
        """
        try:
            t0 = time.time()
            
            # Check cache
            if query in self.embedding_cache:
                q_vec = self.embedding_cache[query]
            else:
                q_vec = self.model.encode(
                    [query],
                    normalize_embeddings=True
                )[0]
                self.embedding_cache[query] = q_vec
            
            results = self.vector_db.search(q_vec, k)
            
            latency = (time.time() - t0) * 1000
            if request_logger:
                request_logger.logger.debug(
                    f"Vector search: {len(results)} results in {latency:.2f}ms"
                )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            if request_logger:
                request_logger.log_error("vector_search_failed", str(e))
            return []
    
    def _bm25_search(self, query: str, k: int, request_logger: Optional[RequestLogger] = None) -> List[Dict]:
        """
        Sparse BM25 search
        """
        try:
            t0 = time.time()
            results = self.bm25.search(query, k)
            latency = (time.time() - t0) * 1000
            
            if request_logger:
                request_logger.logger.debug(
                    f"BM25 search: {len(results)} results in {latency:.2f}ms"
                )
            return results
            
        except Exception as e:
            self.logger.error(f"BM25 search failed: {e}")
            if request_logger:
                request_logger.log_error("bm25_search_failed", str(e))
            return []
    
    def rrf_fusion(self, vector_results: List[Dict], bm25_results: List[Dict], k: int = 60) -> List[Dict]:
        """
        Reciprocal Rank Fusion - combine dense and sparse results
        """
        fused = {}
        
        # Vector results
        for rank, r in enumerate(vector_results):
            cid = r["chunk_id"]
            fused[cid] = {
                "text": r["text"],
                "chunk_id": cid,
                "source": r["source"],
                "vector_score": float(r["vector_score"]),
                "bm25_score": 0.0,
                "rrf_score": 1 / (k + rank + 1)
            }
        
        # BM25 results
        for rank, r in enumerate(bm25_results):
            cid = r["chunk_id"]
            if cid in fused:
                fused[cid]["bm25_score"] = float(r["bm25_score"])
                fused[cid]["rrf_score"] += 1 / (k + rank + 1)
            else:
                fused[cid] = {
                    "text": r["text"],
                    "chunk_id": cid,
                    "source": r["source"],
                    "vector_score": 0.0,
                    "bm25_score": float(r["bm25_score"]),
                    "rrf_score": 1 / (k + rank + 1)
                }
        
        return list(fused.values())
    
    def search(
        self,
        query: str,
        k: int = None,
        final_k: int = None,
        request_logger: Optional[RequestLogger] = None
    ) -> Tuple[List[Dict], float]:
        """
        PRODUCTION SEARCH with full tracing
        
        Args:
            query: Search query
            k: Top-K retrieval
            final_k: Final context chunks
            request_logger: Optional request logger for tracing
        
        Returns:
            (results: List[Dict], best_score: float)
        """
        
        k = k or settings.rag.retrieval_k
        final_k = final_k or settings.rag.final_k
        
        try:
            t0 = time.time()
            
            # Parallel retrieval
            with ThreadPoolExecutor(max_workers=2) as executor:
                vector_future = executor.submit(self._vector_search, query, k, request_logger)
                bm25_future = executor.submit(self._bm25_search, query, k, request_logger)
                
                vector_results = vector_future.result(timeout=5)
                bm25_results = bm25_future.result(timeout=5)
            
            retrieval_latency = (time.time() - t0) * 1000
            
            # RRF Fusion
            candidates = self.rrf_fusion(vector_results, bm25_results)
            candidates = sorted(
                candidates,
                key=lambda x: x["rrf_score"],
                reverse=True
            )
            
            # Select top-K
            top = candidates[:final_k]
            
            results = [
                {
                    "text": c["text"],
                    "chunk_id": c["chunk_id"],
                    "source": c["source"],
                    "vector_score": c["vector_score"],
                    "bm25_score": c["bm25_score"],
                    "rrf_score": c["rrf_score"]
                }
                for c in top
            ]
            
            best_score = results[0]["rrf_score"] if results else -999
            
            # Log retrieval
            if request_logger:
                request_logger.log_retrieval(
                    route="rag",
                    query=query,
                    results_count=len(results),
                    best_score=best_score,
                    latency_ms=retrieval_latency,
                    chunks_retrieved=results
                )
            
            return results, best_score
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            if request_logger:
                request_logger.log_error("search_failed", str(e))
            
            # Graceful degradation: return empty results
            return [], -999


# Global instance
rag = ProductionRAGPipeline()