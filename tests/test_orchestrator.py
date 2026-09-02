import pytest
from unittest.mock import Mock, patch, MagicMock
from app.core.orchestrator import ProductionOrchestrator
from app.schemas import Question
from app.core.logger import RequestLogger

@pytest.fixture
def orchestrator():
    return ProductionOrchestrator()

@pytest.fixture
def sample_question():
    return Question(
        text="How do I treat nitrogen deficiency?",
        session_id="test-user-123",
        lat=35.0,
        lon=10.0
    )

class TestOrchestrator:
    """Test production orchestrator"""
    
    def test_route_determination(self, orchestrator):
        """Test routing logic"""
        request_logger = RequestLogger()
        
        # RAG question
        route, conf = orchestrator._determine_route(
            "What's the best way to grow tomatoes?",
            request_logger
        )
        assert route in ["rag", "chat", "weather", "unknown"]
        assert 0 <= conf <= 1
    
    def test_low_confidence_defaults_to_rag(self, orchestrator):
        """Low confidence should default to RAG"""
        request_logger = RequestLogger()
        
        with patch("app.core.orchestrator.llm_route") as mock_route:
            mock_route.return_value = ("unknown", 0.3)  # Low confidence
            
            route, conf = orchestrator._determine_route("test", request_logger)
            
            # Should default to RAG despite "unknown" route
            assert route == "rag"
            assert conf == 0.3
    
    @patch("app.services.llm.generate_answer")
    def test_chat_handler(self, mock_llm, orchestrator):
        """Test chat route handler"""
        mock_llm.return_value = "That sounds like a good farming question!"
        
        request_logger = RequestLogger()
        
        answer = orchestrator.handle_chat(
            text="Hello!",
            session_id="test",
            request_logger=request_logger
        )
        
        assert answer == "That sounds like a good farming question!"
        mock_llm.assert_called_once()
    
    @patch("app.services.weather.get_weather")
    @patch("app.services.llm.generate_answer")
    def test_weather_handler(self, mock_llm, mock_weather, orchestrator):
        """Test weather route handler"""
        mock_weather.return_value = {
            "current_weather": {
                "temperature": 25,
                "windspeed": 15,
                "precipitation": 0
            }
        }
        mock_llm.return_value = "Based on the 25°C weather..."
        
        request_logger = RequestLogger()
        
        answer = orchestrator.handle_weather(
            text="Is it good weather for planting?",
            lat=35.0,
            lon=10.0,
            session_id="test",
            request_logger=request_logger
        )
        
        assert answer == "Based on the 25°C weather..."
        mock_weather.assert_called_once_with(35.0, 10.0)
    
    @patch("app.services.rag.rag.search")
    @patch("app.core.citations.CitationEnforcer.enforce_citations")
    @patch("app.services.llm.generate_answer")
    def test_rag_handler(self, mock_llm, mock_citations, mock_search, orchestrator):
        """Test RAG route handler"""
        mock_search.return_value = (
            [
                {
                    "text": "Nitrogen deficiency causes yellowing of leaves",
                    "chunk_id": "1",
                    "source": "guide.pdf"
                }
            ],
            0.85
        )
        mock_llm.return_value = "Nitrogen deficiency is indicated by yellowing"
        mock_citations.return_value = ("Nitrogen deficiency is indicated by yellowing", False)
        
        request_logger = RequestLogger()
        
        answer, score = orchestrator.handle_rag(
            text="What's nitrogen deficiency?",
            session_id="test",
            request_logger=request_logger
        )
        
        assert answer == "Nitrogen deficiency is indicated by yellowing"
        assert score == 0.85
        mock_search.assert_called_once()
    
    @patch("app.services.llm.generate_answer")
    def test_unknown_handler(self, mock_llm, orchestrator):
        """Test unknown route handler"""
        mock_llm.return_value = "I only help with farming topics..."
        
        request_logger = RequestLogger()
        answer = orchestrator.handle_unknown(request_logger)
        
        assert "farming" in answer.lower()
    
    @patch("app.core.orchestrator.orchestrator._determine_route")
    @patch("app.core.orchestrator.orchestrator.handle_chat")
    @patch("app.core.memory.memory.add")
    def test_full_question_flow(self, mock_memory, mock_chat, mock_route, orchestrator, sample_question):
        """Test full question handling flow"""
        mock_route.return_value = ("chat", 0.9)
        mock_chat.return_value = "That's a great question!"
        
        answer = orchestrator.handle_question(sample_question)
        
        assert answer == "That's a great question!"
        mock_memory.assert_called()
    
    @patch("app.core.orchestrator.orchestrator.handle_rag")
    def test_rag_error_recovery(self, mock_rag, orchestrator, sample_question):
        """Test graceful degradation on RAG error"""
        mock_rag.side_effect = Exception("RAG failure")
        
        with patch("app.core.orchestrator.orchestrator._determine_route") as mock_route:
            mock_route.return_value = ("rag", 0.8)
            
            answer = orchestrator.handle_question(sample_question)
            
            # Should return error message, not crash
            assert "error" in answer.lower() or "unexpected" in answer.lower()


class TestCitationEnforcement:
    """Test citation enforcement"""
    
    @pytest.fixture
    def citation_enforcer(self):
        from app.core.citations import CitationEnforcer
        return CitationEnforcer(None)
    
    @patch("app.services.llm.generate_answer")
    def test_citation_check_supported(self, mock_llm, citation_enforcer):
        """Test citation check when answer is supported"""
        mock_llm.return_value = json.dumps({
            "is_supported": True,
            "confidence": 0.95,
            "reasoning": "Answer is well-supported by documents"
        })
        
        is_supported, confidence, reasoning = citation_enforcer.check_citations(
            question="How to treat nitrogen deficiency?",
            answer="Apply nitrogen fertilizer in spring",
            retrieved_chunks=[
                {"text": "Nitrogen should be applied in spring...", "source": "guide.pdf"}
            ]
        )
        
        assert is_supported == True
        assert confidence == 0.95
    
    @patch("app.services.llm.generate_answer")
    def test_citation_check_unsupported(self, mock_llm, citation_enforcer):
        """Test citation check when answer is unsupported"""
        mock_llm.return_value = json.dumps({
            "is_supported": False,
            "confidence": 0.1,
            "reasoning": "Answer contradicts the retrieved documents"
        })
        
        is_supported, confidence, reasoning = citation_enforcer.check_citations(
            question="Can I grow bananas in frozen soil?",
            answer="Yes, bananas grow well in frozen soil",
            retrieved_chunks=[
                {"text": "Bananas require warm tropical climate", "source": "tropical-crops.pdf"}
            ]
        )
        
        assert is_supported == False
        assert confidence == 0.1
    
    def test_empty_chunks_not_supported(self, citation_enforcer):
        """No chunks = answer cannot be supported"""
        is_supported, confidence, _ = citation_enforcer.check_citations(
            question="Test?",
            answer="Test answer",
            retrieved_chunks=[]
        )
        
        assert is_supported == False
        assert confidence == 0.0
    
    def test_enforce_citations_fallback(self, citation_enforcer):
        """Test fallback when citation fails"""
        with patch.object(citation_enforcer, "check_citations") as mock_check:
            mock_check.return_value = (False, 0.2, "Not supported")
            
            final_answer, is_hallucination = citation_enforcer.enforce_citations(
                question="Test",
                answer="Original answer",
                retrieved_chunks=[{"text": "Some content", "source": "doc.pdf"}],
                best_score=0.5
            )
            
            assert is_hallucination == True
            assert "don't have reliable" in final_answer.lower()


class TestErrorHandling:
    """Test error handling"""
    
    @patch("app.core.orchestrator.orchestrator.handle_question")
    def test_orchestrator_handles_exceptions(self, mock_handle):
        """Test orchestrator handles exceptions gracefully"""
        mock_handle.side_effect = RuntimeError("Unexpected error")
        
        orchestrator = ProductionOrchestrator()
        question = Question(
            text="test",
            session_id="test",
            lat=0,
            lon=0
        )
        
        # Should not crash, should return error message
        answer = orchestrator.handle_question(question)
        assert isinstance(answer, str)


# Run tests with: python -m pytest tests/test_orchestrator.