import time
from typing import Optional, Tuple, Generator
from fastapi.responses import StreamingResponse

from app.services.weather import get_weather
from app.services.rag import rag
from app.services.llm import generate_answer, generate_stream
from app.core.router import llm_route
from app.core.memory import memory
from app.core.logger import RequestLogger, app_logger
from app.core.citations import CitationEnforcer
from app.config.settings import settings, prompts
from app.schemas import Question


class ProductionOrchestrator:
    """
    PRODUCTION ORCHESTRATOR:
    - Single source of truth for request handling
    - Full instrumentation and logging
    - Citation enforcement
    - Graceful error handling
    - Metrics collection
    """
    
    def __init__(self):
        self.citation_enforcer = CitationEnforcer(None)  # Will set per-request
    
    def _get_conversation_history(self, session_id: str) -> str:
        """Get formatted conversation history"""
        history = memory.get(session_id)
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
    
    def _determine_route(self, text: str, request_logger: RequestLogger) -> Tuple[str, float]:
        """
        Route the question to appropriate handler
        
        Returns:
            (route: str, confidence: float)
        """
        try:
            route, confidence = llm_route(text)
            
            # If confidence is low, default to RAG
            if confidence < 0.6:
                route = "rag"
                confidence = 0.5
            
            request_logger.logger.debug(
                f"Route determined: {route} (confidence: {confidence:.2f})"
            )
            
            return route, confidence
            
        except Exception as e:
            request_logger.log_error("routing_failed", str(e))
            # Default to RAG on routing failure
            return "rag", 0.3
    
    def handle_chat(
        self,
        text: str,
        session_id: str,
        request_logger: RequestLogger
    ) -> str:
        """Handle casual conversation"""
        try:
            history = self._get_conversation_history(session_id)
            
            prompt = prompts["chat_prompt"].format(
                history=history,
                text=text
            )
            
            t0 = time.time()
            answer = generate_answer(prompt)
            latency = (time.time() - t0) * 1000
            
            request_logger.log_generation(
                latency_ms=latency,
                tokens_used=len(answer.split()),
                model=settings.llm.model_name
            )
            
            return answer
            
        except Exception as e:
            request_logger.log_error("chat_handler_failed", str(e))
            return "I'm having trouble right now. Please try again later."
    
    def handle_weather(
        self,
        text: str,
        lat: float,
        lon: float,
        session_id: str,
        request_logger: RequestLogger
    ) -> str:
        """Handle weather-related questions"""
        try:
            weather_data = get_weather(lat, lon)
            current = weather_data.get("current_weather", {})
            
            weather_text = f"""
Temperature: {current.get('temperature', 0)}°C
Wind: {current.get('windspeed', 0)} km/h
Precipitation: {current.get('precipitation', 0)}mm
"""
            
            history = self._get_conversation_history(session_id)
            
            prompt = prompts["weather_prompt"].format(
                history=history,
                weather_data=weather_text,
                question=text
            )
            
            t0 = time.time()
            answer = generate_answer(prompt)
            latency = (time.time() - t0) * 1000
            
            request_logger.log_generation(
                latency_ms=latency,
                tokens_used=len(answer.split()),
                model=settings.llm.model_name
            )
            
            return answer
            
        except Exception as e:
            request_logger.log_error("weather_handler_failed", str(e))
            return "I couldn't retrieve weather information. Please try again."
    
    def handle_rag(
        self,
        text: str,
        session_id: str,
        request_logger: RequestLogger
    ) -> Tuple[str, float]:
        """Handle knowledge base questions"""
        try:
            # Retrieve
            results, best_score = rag.search(
                text,
                k=settings.rag.retrieval_k,
                final_k=settings.rag.final_k,
                request_logger=request_logger
            )
            
            # Build context
            context = ""
            if results and best_score > settings.rag.retrieval_threshold:
                context = "\n\n".join([r["text"] for r in results[:3]])
            
            # Generate answer
            history = self._get_conversation_history(session_id)
            
            prompt = prompts["rag_prompt"].format(
                history=history,
                question=text,
                context=context if context else "No relevant context found."
            )
            
            t0 = time.time()
            answer = generate_answer(prompt)
            latency = (time.time() - t0) * 1000
            
            request_logger.log_generation(
                latency_ms=latency,
                tokens_used=len(answer.split()),
                model=settings.llm.model_name
            )
            
            # ⭐ PRODUCTION PATTERN: Enforce citations
            self.citation_enforcer.logger = request_logger
            final_answer, is_hallucination = self.citation_enforcer.enforce_citations(
                question=text,
                answer=answer,
                retrieved_chunks=results,
                best_score=best_score,
                threshold=0.7
            )
            
            return final_answer, best_score
            
        except Exception as e:
            request_logger.log_error("rag_handler_failed", str(e))
            return (
                "I encountered an error retrieving information. Please try again.",
                -999
            )
    
    def handle_unknown(self, request_logger: RequestLogger) -> str:
        """Handle out-of-scope questions"""
        prompt = prompts["unknown_prompt"]
        answer = generate_answer(prompt)
        return answer
    
    def handle_question(
        self,
        question: Question
    ) -> str:
        """
        MAIN PRODUCTION HANDLER
        Coordinates routing, retrieval, generation, and quality checks
        """
        
        # Create request-scoped logger
        request_logger = RequestLogger()
        request_logger.log_request_start(
            session_id=question.session_id,
            question=question.text
        )
        
        t_start = time.time()
        
        try:
            # Route request
            route, confidence = self._determine_route(question.text, request_logger)
            
            best_score = -1
            answer = ""
            
            # Handle by route
            if route == "chat":
                answer = self.handle_chat(
                    question.text,
                    question.session_id,
                    request_logger
                )
            
            elif route == "weather":
                answer = self.handle_weather(
                    question.text,
                    question.lat,
                    question.lon,
                    question.session_id,
                    request_logger
                )
            
            elif route == "rag":
                answer, best_score = self.handle_rag(
                    question.text,
                    question.session_id,
                    request_logger
                )
            
            elif route == "unknown":
                answer = self.handle_unknown(request_logger)
            
            else:
                answer = "I didn't understand that. Could you rephrase?"
            
            # Store in memory
            memory.add(question.session_id, "User", question.text)
            memory.add(question.session_id, "Assistant", answer)
            
            # Log completion
            total_latency = (time.time() - t_start) * 1000
            request_logger.log_request_end(
                total_latency_ms=total_latency,
                status="success"
            )
            
            return answer
            
        except Exception as e:
            request_logger.log_error(
                error_type="orchestrator_error",
                error_msg=str(e),
                traceback=str(e)
            )
            
            total_latency = (time.time() - t_start) * 1000
            request_logger.log_request_end(
                total_latency_ms=total_latency,
                status="error",
                error=str(e)
            )
            
            return "An unexpected error occurred. Please try again."
    
    def handle_question_stream(
        self,
        question: Question
    ) -> StreamingResponse:
        """
        STREAMING HANDLER
        Same as handle_question but with streaming response
        """
        
        request_logger = RequestLogger()
        request_logger.log_request_start(
            session_id=question.session_id,
            question=question.text
        )
        
        async def stream_generator():
            try:
                t_start = time.time()
                
                # Route
                route, _ = self._determine_route(question.text, request_logger)
                
                history = self._get_conversation_history(question.session_id)
                
                # Build prompt based on route
                if route == "chat":
                    prompt = prompts["chat_prompt"].format(
                        history=history,
                        text=question.text
                    )
                
                elif route == "weather":
                    weather_data = get_weather(question.lat, question.lon)
                    current = weather_data.get("current_weather", {})
                    weather_text = f"Temperature: {current.get('temperature', 0)}°C\nWind: {current.get('windspeed', 0)} km/h"
                    
                    prompt = prompts["weather_prompt"].format(
                        history=history,
                        weather_data=weather_text,
                        question=question.text
                    )
                
                elif route == "rag":
                    results, best_score = rag.search(question.text, request_logger=request_logger)
                    context = "\n\n".join([r["text"] for r in results[:3]]) if results else ""
                    
                    prompt = prompts["rag_prompt_streaming"].format(
                        context=context,
                        question=question.text
                    )
                
                else:
                    prompt = prompts["unknown_prompt"]
                
                # Stream generation
                full_answer = ""
                for token in generate_stream(prompt):
                    full_answer += token
                    yield token
                
                # Store and log
                memory.add(question.session_id, "User", question.text)
                memory.add(question.session_id, "Assistant", full_answer)
                
                total_latency = (time.time() - t_start) * 1000
                request_logger.log_request_end(
                    total_latency_ms=total_latency,
                    status="success"
                )
                
            except Exception as e:
                request_logger.log_error("streaming_handler_failed", str(e))
                yield f"\n\n[ERROR] {str(e)}"
        
        return StreamingResponse(stream_generator(), media_type="text/plain")


# Global instance
orchestrator = ProductionOrchestrator()