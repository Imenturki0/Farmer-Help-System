import logging
import json
from datetime import datetime
from typing import Any, Dict, Optional
import uuid
from pathlib import Path

# Ensure logs directory exists
Path("logs").mkdir(exist_ok=True)

class StructuredLogger:
    """Production-grade structured logging with JSON format"""
    
    def __init__(self, name: str = "farm-ai"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # JSON file handler (for structured logs)
        json_handler = logging.FileHandler("logs/app.json")
        json_formatter = logging.Formatter('%(message)s')
        json_handler.setFormatter(json_formatter)
        self.logger.addHandler(json_handler)
        
        # Console handler (for development)
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def _structured_log(self, level: str, data: Dict[str, Any]):
        """Log as structured JSON"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            **data
        }
        self.logger.log(
            getattr(logging, level),
            json.dumps(log_entry)
        )
    
    def info(self, message: str, **kwargs):
        self._structured_log("INFO", {"message": message, **kwargs})
    
    def error(self, message: str, **kwargs):
        self._structured_log("ERROR", {"message": message, **kwargs})
    
    def warning(self, message: str, **kwargs):
        self._structured_log("WARNING", {"message": message, **kwargs})
    
    def debug(self, message: str, **kwargs):
        self._structured_log("DEBUG", {"message": message, **kwargs})


class RequestLogger:
    """Track request through entire pipeline"""
    
    def __init__(self):
        self.logger = StructuredLogger("request-pipeline")
        self.request_id = str(uuid.uuid4())[:8]
    
    def log_request_start(self, session_id: str, question: str, route: Optional[str] = None):
        """Log when request arrives"""
        self.logger.info(
            "Request received",
            request_id=self.request_id,
            session_id=session_id,
            question=question,
            route=route
        )
    
    def log_retrieval(
        self,
        route: str,
        query: str,
        results_count: int,
        best_score: float,
        latency_ms: float,
        chunks_retrieved: list
    ):
        """Log retrieval results"""
        self.logger.info(
            "Retrieval completed",
            request_id=self.request_id,
            route=route,
            query=query,
            results_count=results_count,
            best_score=best_score,
            latency_ms=f"{latency_ms:.2f}",
            chunk_ids=[c.get("chunk_id") for c in chunks_retrieved],
            sources=[c.get("source") for c in chunks_retrieved]
        )
    
    def log_generation(
        self,
        latency_ms: float,
        tokens_used: int,
        model: str
    ):
        """Log LLM generation"""
        self.logger.info(
            "Generation completed",
            request_id=self.request_id,
            latency_ms=f"{latency_ms:.2f}",
            tokens_used=tokens_used,
            model=model
        )
    
    def log_quality_check(
        self,
        is_citation_supported: bool,
        confidence: float,
        reasoning: str
    ):
        """Log citation/quality check results"""
        self.logger.info(
            "Quality check completed",
            request_id=self.request_id,
            is_citation_supported=is_citation_supported,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def log_request_end(
        self,
        total_latency_ms: float,
        status: str = "success",
        error: Optional[str] = None
    ):
        """Log request completion"""
        self.logger.info(
            "Request completed",
            request_id=self.request_id,
            total_latency_ms=f"{total_latency_ms:.2f}",
            status=status,
            error=error
        )
    
    def log_error(self, error_type: str, error_msg: str, traceback: Optional[str] = None):
        """Log errors"""
        self.logger.error(
            "Error occurred",
            request_id=self.request_id,
            error_type=error_type,
            error_message=error_msg,
            traceback=traceback
        )


# Global logger instance
app_logger = StructuredLogger()

def get_request_logger() -> RequestLogger:
    """Factory for request-scoped loggers"""
    return RequestLogger()