
import pytest

from app.services.bm25_retriever import BM25Retriever
from app.services.vector_db import QdrantVectorDB
from app.services.rag import ProductionRAGPipeline


@pytest.mark.integration
def test_bm25_loads_real_documents():
    retriever = BM25Retriever()

    retriever.load("data/processed/chunks.json")

    results = retriever.search("tomato fertilizer", k=3)

    assert retriever.bm25 is not None
    assert len(retriever.docs) > 0
    assert len(results) > 0

    for result in results:
        assert "text" in result
        assert "chunk_id" in result
        assert "bm25_score" in result


@pytest.mark.integration
def test_qdrant_real_collection_search():
    db = QdrantVectorDB()

    assert db.collection_exists()

    # Small deterministic test vector.
    vector = [0.0] * 768

    results = db.search(vector, k=3)

    assert isinstance(results, list)

    for result in results:
        assert "text" in result
        assert "chunk_id" in result
        assert "source" in result
        assert "vector_score" in result


@pytest.mark.integration
def test_real_rag_search():
    pipeline = ProductionRAGPipeline()

    results, score = pipeline.search(
        "How should tomatoes be fertilized?",
        k=5,
        final_k=3,
    )

    assert isinstance(results, list)
    assert len(results) > 0

    assert isinstance(score, float)

    for result in results:
        assert "text" in result
        assert "chunk_id" in result
        assert "source" in result
        assert "rrf_score" in result
