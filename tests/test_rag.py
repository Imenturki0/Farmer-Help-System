import pytest
from unittest.mock import Mock, patch

from app.services.rag import ProductionRAGPipeline


@pytest.fixture
def rag_pipeline():
    """Create RAG pipeline without loading external services."""
    pipeline = ProductionRAGPipeline.__new__(ProductionRAGPipeline)

    pipeline.model = Mock()
    pipeline.vector_db = Mock()
    pipeline.bm25 = Mock()
    pipeline.embedding_cache = {}
    pipeline.logger = Mock()

    return pipeline


class TestRRFFusion:


    def test_fuses_results_from_both_retrievers(self, rag_pipeline):
        vector_results = [
            {
                "chunk_id": "chunk-1",
                "text": "Vector result",
                "source": "guide.pdf",
                "vector_score": 0.90,
            },
            {
                "chunk_id": "chunk-2",
                "text": "Second vector result",
                "source": "guide.pdf",
                "vector_score": 0.80,
            },
        ]

        bm25_results = [
            {
                "chunk_id": "chunk-1",
                "text": "Vector result",
                "source": "guide.pdf",
                "bm25_score": 12.0,
            },
            {
                "chunk_id": "chunk-3",
                "text": "BM25-only result",
                "source": "other.pdf",
                "bm25_score": 10.0,
            },
        ]

        results = rag_pipeline.rrf_fusion(
            vector_results,
            bm25_results,
            k=60,
        )

        assert len(results) == 3

        chunk_ids = {result["chunk_id"] for result in results}

        assert chunk_ids == {
            "chunk-1",
            "chunk-2",
            "chunk-3",
        }

    def test_chunk_in_both_retrievers_gets_higher_rrf_score(self, rag_pipeline):
        vector_results = [
            {
                "chunk_id": "shared",
                "text": "Shared result",
                "source": "guide.pdf",
                "vector_score": 0.90,
            }
        ]

        bm25_results = [
            {
                "chunk_id": "shared",
                "text": "Shared result",
                "source": "guide.pdf",
                "bm25_score": 15.0,
            }
        ]

        results = rag_pipeline.rrf_fusion(
            vector_results,
            bm25_results,
            k=60,
        )

        assert len(results) == 1

        result = results[0]

        expected_score = (1 / 61) + (1 / 61)

        assert result["rrf_score"] == pytest.approx(expected_score)
        assert result["vector_score"] == 0.90
        assert result["bm25_score"] == 15.0

    def test_vector_only_result_is_kept(self, rag_pipeline):
        vector_results = [
            {
                "chunk_id": "vector-only",
                "text": "Vector result",
                "source": "guide.pdf",
                "vector_score": 0.85,
            }
        ]

        results = rag_pipeline.rrf_fusion(
            vector_results,
            [],
            k=60,
        )

        assert len(results) == 1
        assert results[0]["chunk_id"] == "vector-only"
        assert results[0]["bm25_score"] == 0.0
        assert results[0]["rrf_score"] == pytest.approx(1 / 61)

    def test_bm25_only_result_is_kept(self, rag_pipeline):
        bm25_results = [
            {
                "chunk_id": "bm25-only",
                "text": "BM25 result",
                "source": "guide.pdf",
                "bm25_score": 10.0,
            }
        ]

        results = rag_pipeline.rrf_fusion(
            [],
            bm25_results,
            k=60,
        )

        assert len(results) == 1
        assert results[0]["chunk_id"] == "bm25-only"
        assert results[0]["vector_score"] == 0.0
        assert results[0]["bm25_score"] == 10.0
        assert results[0]["rrf_score"] == pytest.approx(1 / 61)

class TestRAGSearch:
    
    def test_search_combines_and_ranks_results(self, rag_pipeline):
        vector_results = [
            {
                "chunk_id": "chunk-1",
                "text": "Shared result",
                "source": "guide.pdf",
                "vector_score": 0.95,
            },
            {
                "chunk_id": "chunk-2",
                "text": "Vector result",
                "source": "guide.pdf",
                "vector_score": 0.80,
            },
        ]

        bm25_results = [
            {
                "chunk_id": "chunk-1",
                "text": "Shared result",
                "source": "guide.pdf",
                "bm25_score": 15.0,
            },
            {
                "chunk_id": "chunk-3",
                "text": "BM25 result",
                "source": "other.pdf",
                "bm25_score": 10.0,
            },
        ]

        with patch.object(
            rag_pipeline,
            "_vector_search",
            return_value=vector_results,
        ) as mock_vector:

            with patch.object(
                rag_pipeline,
                "_bm25_search",
                return_value=bm25_results,
            ) as mock_bm25:

                results, best_score = rag_pipeline.search(
                    query="How do I grow tomatoes?",
                    k=10,
                    final_k=2,
                )

        mock_vector.assert_called_once()
        mock_bm25.assert_called_once()

        assert len(results) == 2

        # chunk-1 appears first in BOTH retrievers,
        # so it should have the highest RRF score.
        assert results[0]["chunk_id"] == "chunk-1"

        assert best_score == results[0]["rrf_score"]


    def test_search_respects_final_k(self, rag_pipeline):
        vector_results = [
            {
                "chunk_id": "chunk-1",
                "text": "Result 1",
                "source": "guide.pdf",
                "vector_score": 0.90,
            },
            {
                "chunk_id": "chunk-2",
                "text": "Result 2",
                "source": "guide.pdf",
                "vector_score": 0.80,
            },
            {
                "chunk_id": "chunk-3",
                "text": "Result 3",
                "source": "guide.pdf",
                "vector_score": 0.70,
            },
        ]

        with patch.object(
            rag_pipeline,
            "_vector_search",
            return_value=vector_results,
        ), patch.object(
            rag_pipeline,
            "_bm25_search",
            return_value=[],
        ):

            results, _ = rag_pipeline.search(
                query="test",
                k=10,
                final_k=2,
            )

        assert len(results) == 2


    def test_search_handles_no_results(self, rag_pipeline):

        with patch.object(
            rag_pipeline,
            "_vector_search",
            return_value=[],
        ), patch.object(
            rag_pipeline,
            "_bm25_search",
            return_value=[],
        ):

            results, best_score = rag_pipeline.search(
                query="unknown question",
                k=10,
                final_k=5,
            )

        assert results == []
        assert best_score == -999

class TestRetrieverFailures:
    
    def test_vector_search_returns_empty_list_on_failure(self, rag_pipeline):
        with patch.object(
            rag_pipeline.model,
            "encode",
            side_effect=Exception("Embedding model failed"),
        ):
            results = rag_pipeline._vector_search(
                query="test query",
                k=5,
            )

        assert results == []

    def test_bm25_search_returns_empty_list_on_failure(self, rag_pipeline):
        with patch.object(
            rag_pipeline.bm25,
            "search",
            side_effect=Exception("BM25 failed"),
        ):
            results = rag_pipeline._bm25_search(
                query="test query",
                k=5,
            )

        assert results == []

class TestSearchResilience:
    
    def test_search_continues_when_vector_search_fails(self, rag_pipeline):
        bm25_results = [
            {
                "chunk_id": "bm25-1",
                "text": "BM25 result",
                "source": "guide.pdf",
                "bm25_score": 10.0,
            }
        ]

        with patch.object(
            rag_pipeline,
            "_vector_search",
            return_value=[],
        ), patch.object(
            rag_pipeline,
            "_bm25_search",
            return_value=bm25_results,
        ):
            results, best_score = rag_pipeline.search(
                query="test query",
                k=10,
                final_k=5,
            )

        assert len(results) == 1
        assert results[0]["chunk_id"] == "bm25-1"
        assert best_score == results[0]["rrf_score"]

    def test_search_continues_when_bm25_search_fails(self, rag_pipeline):
        vector_results = [
            {
                "chunk_id": "vector-1",
                "text": "Vector result",
                "source": "guide.pdf",
                "vector_score": 0.90,
            }
        ]

        with patch.object(
            rag_pipeline,
            "_vector_search",
            return_value=vector_results,
        ), patch.object(
            rag_pipeline,
            "_bm25_search",
            return_value=[],
        ):
            results, best_score = rag_pipeline.search(
                query="test query",
                k=10,
                final_k=5,
            )

        assert len(results) == 1
        assert results[0]["chunk_id"] == "vector-1"
        assert best_score == results[0]["rrf_score"]