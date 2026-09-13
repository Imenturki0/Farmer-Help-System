
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from fastapi.responses import StreamingResponse

from app.main import app


client = TestClient(app)


class TestHealth:

    def test_health_returns_200(self):
        response = client.get("/health")

        assert response.status_code == 200

    def test_health_returns_healthy_status(self):
        response = client.get("/health")

        data = response.json()

        assert data["status"] == "healthy"


class TestDeepHealth:

    @patch("app.main.rag")
    def test_deep_health_returns_healthy(self, mock_rag):
        mock_rag.vector_db = Mock()
        mock_rag.bm25 = Mock()
        mock_rag.model = Mock()

        response = client.get("/health/deep")

        assert response.status_code == 200

        data = response.json()

        assert data["status"] == "healthy"
        assert data["components"]["api"] == "operational"
        assert data["components"]["rag"] == "operational"
        assert data["components"]["llm"] == "operational"

    @patch("app.main.rag")
    def test_deep_health_detects_degraded_rag(self, mock_rag):
        mock_rag.vector_db = None
        mock_rag.bm25 = Mock()
        mock_rag.model = Mock()

        response = client.get("/health/deep")

        assert response.status_code == 200

        data = response.json()

        assert data["status"] == "degraded"
        assert data["components"]["rag"] == "degraded"
        assert data["components"]["llm"] == "operational"


class TestAsk:

    @patch("app.main.orchestrator")
    def test_ask_returns_answer(self, mock_orchestrator):
        mock_orchestrator.handle_question.return_value = (
        "Plant tomatoes in well-drained soil."
        )

        response = client.post(
            "/ask",
            json={
                "text": "How should I plant tomatoes?",
                "session_id": "test-session",
            },
        )

        assert response.status_code == 200
        assert "tomatoes" in response.text.lower()

    @patch("app.main.orchestrator")
    def test_ask_calls_orchestrator(self, mock_orchestrator):
        mock_orchestrator.handle_question.return_value = "Test answer"

        response = client.post(
            "/ask",
            json={
                "text": "Hello",
                "session_id": "test-session",
            },
        )

        assert response.status_code == 200
        mock_orchestrator.handle_question.assert_called_once()
    
   
    @patch("app.main.orchestrator")
    def test_ask_handles_orchestrator_error(self, mock_orchestrator):
        mock_orchestrator.handle_question.side_effect = Exception(
            "Test orchestrator failure"
        )

        response = client.post(
            "/ask",
            json={
                "text": "How should I plant tomatoes?",
                "session_id": "test-session",
            },
        )

        assert response.status_code == 200
        assert "error occurred" in response.text.lower()


class TestAskStream:

    @patch("app.main.orchestrator")
    def test_ask_stream_returns_stream(self, mock_orchestrator):
        mock_orchestrator.handle_question_stream.return_value = (
            StreamingResponse(
                iter(["Hello ", "farmer!"]),
                media_type="text/plain",
            )
        )

        response = client.post(
            "/ask-stream",
            json={
                "text": "How do I grow tomatoes?",
                "session_id": "test-session",
            },
        )

        assert response.status_code == 200
        assert "Hello farmer!" in response.text

    @patch("app.main.orchestrator")
    def test_ask_stream_calls_orchestrator(self, mock_orchestrator):
        mock_orchestrator.handle_question_stream.return_value = (
            StreamingResponse(
                iter(["Test answer"]),
                media_type="text/plain",
            )
        )

        response = client.post(
            "/ask-stream",
            json={
                "text": "How do I grow tomatoes?",
                "session_id": "test-session",
            },
        )

        assert response.status_code == 200
        mock_orchestrator.handle_question_stream.assert_called_once()

    @patch("app.main.orchestrator")
    def test_ask_stream_handles_error(self, mock_orchestrator):
        mock_orchestrator.handle_question_stream.side_effect = Exception(
            "Test streaming failure"
        )

        response = client.post(
            "/ask-stream",
            json={
                "text": "How do I grow tomatoes?",
                "session_id": "test-session",
            },
        )

        assert response.status_code == 500
        assert response.json()["detail"] == "Streaming failed"




class TestValidation:


    def test_ask_rejects_missing_text(self):
        response = client.post(
            "/ask",
            json={
                "session_id": "test-session",
            },
        )

        assert response.status_code == 422

    def test_ask_rejects_empty_body(self):

        response = client.post(
            "/ask",
            json={},
        )

        assert response.status_code == 422


class TestInfo:

    def test_info_returns_system_information(self):
        response = client.get("/info")

        assert response.status_code == 200

        data = response.json()

        assert "name" in data
        assert "version" in data
        assert "environment" in data
        assert "rag" in data
        assert "llm" in data

        assert "embedding_model" in data["rag"]
        assert "retrieval_k" in data["rag"]
        assert "final_k" in data["rag"]

        assert "model" in data["llm"]
        assert "temperature" in data["llm"]


#  python -m pytest tests/test_api.py -q
#all together python -m pytest -q
