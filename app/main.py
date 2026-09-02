from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.exceptions import RequestValidationError
# import logging
import time
import uuid
from contextlib import asynccontextmanager

from app.schemas import Question
from app.core.orchestrator import orchestrator
from app.services.rag import rag
from app.core.logger import app_logger
from app.config.settings import settings

# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown."""

    app_logger.info("🚀 Starting Farmer AI Assistant...")
    app_logger.info("✅ RAG Pipeline initialized")

    yield

    app_logger.info("🛑 Shutting down Farmer AI Assistant...")


# ============================================================================
# APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title="Farmer AI Assistant",
    description="Production RAG system for agricultural guidance",
    version=settings.app_name,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


# ============================================================================
# MIDDLEWARE
# ============================================================================

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ "http://localhost:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request ID middleware - trace requests through system
class RequestIDMiddleware:
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request_id = str(uuid.uuid4())[:8]
            scope["request_id"] = request_id
        
        await self.app(scope, receive, send)


app.add_middleware(RequestIDMiddleware)


# Logging middleware
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log all requests"""
    t0 = time.time()
    
    try:
        response = await call_next(request)
    except Exception as e:
        app_logger.error(
            f"Request failed",
            path=request.url.path,
            method=request.method,
            error=str(e)
        )
        raise
    
    latency_ms = (time.time() - t0) * 1000
    app_logger.info(
        f"Request completed",
        path=request.url.path,
        method=request.method,
        status_code=response.status_code,
        latency_ms=f"{latency_ms:.2f}"
    )
    
    return response


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors"""
    app_logger.error(
        "Validation error",
        path=request.url.path,
        errors=exc.errors()
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Invalid request parameters"}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors"""
    app_logger.error(
        "Unhandled exception",
        path=request.url.path,
        error=str(exc),
        error_type=type(exc).__name__
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )


# ============================================================================
# HEALTH CHECK ENDPOINTS
# ============================================================================

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Simple health check
    Returns: 200 if service is healthy
    """
    return {
        "status": "healthy",
        "service": "Farmer AI Assistant",
        "version": settings.app_name
    }


@app.get("/health/deep", tags=["Health"])
async def deep_health_check():
    """
    Deep health check - verifies all components
    """
    status_info = {
        "status": "healthy",
        "components": {
            "api": "operational",
            "rag": "unknown",
            "llm": "unknown"
        }
    }
    
    # Check RAG
    try:
        if rag.vector_db and rag.bm25:
            status_info["components"]["rag"] = "operational"
        else:
            status_info["components"]["rag"] = "degraded"
            status_info["status"] = "degraded"
    except Exception as e:
        status_info["components"]["rag"] = f"error: {str(e)}"
        status_info["status"] = "unhealthy"
    
    # Check LLM (simplified - just verify model loaded)
    try:
        if rag.model:
            status_info["components"]["llm"] = "operational"
        else:
            status_info["components"]["llm"] = "degraded"
            status_info["status"] = "degraded"
    except Exception as e:
        status_info["components"]["llm"] = f"error: {str(e)}"
        status_info["status"] = "unhealthy"
    
    return status_info


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.post("/ask", tags=["Query"], response_class=PlainTextResponse)
async def ask_farm(question: Question):
    """
    Ask a farming question and get an answer.
    
    The system will:
    1. Route your question to the appropriate handler
    2. Retrieve relevant documents if needed
    3. Generate an answer grounded in evidence
    4. Return citations for transparency
    
    Args:
        question: Question object with text, session_id, lat, lon
    
    Returns:
        Plain text answer
    
    Examples:
        ```
        {
            "text": "How do I treat nitrogen deficiency?",
            "session_id": "user-123",
            "lat": 35.0,
            "lon": 10.0
        }
        ```
    """
    try:
        answer = orchestrator.handle_question(question)
        return answer
    except Exception as e:
        app_logger.error(f"Ask endpoint failed: {e}")
        return "An error occurred. Please try again."


@app.post("/ask-stream", tags=["Query"])
async def ask_stream_route(question: Question):
    """
    Ask a farming question with streaming response.
    
    Returns tokens in real-time for better UX.
    
    Args:
        question: Question object
    
    Returns:
        Server-Sent Events stream of response tokens
    """
    try:
        return orchestrator.handle_question_stream(question)
    except Exception as e:
        app_logger.error(f"Stream endpoint failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Streaming failed"}
        )


# ============================================================================
# METADATA ENDPOINTS
# ============================================================================

@app.get("/info", tags=["Info"])
async def get_info():
    """Get system information"""
    return {
        "name": settings.app_name,
        "version": settings.version,
        "environment": settings.environment,
        "rag": {
            "embedding_model": settings.rag.embedding_model,
            "retrieval_k": settings.rag.retrieval_k,
            "final_k": settings.rag.final_k
        },
        "llm": {
            "model": settings.llm.model_name,
            "temperature": settings.llm.temperature
        }
    }


@app.get("/metrics", tags=["Metrics"])
async def get_metrics():
    """
    Get system metrics (placeholder for future integration)
    
    In production, this would return:
    - Request count
    - Average latency
    - Error rate
    - Cache hit rate
    - Quality metrics
    """
    return {
        "note": "Metrics collection not yet integrated",
        "status": "pending implementation",
        "description": "Use /health/deep for component status"
    }


# ============================================================================
# DEVELOPMENT/DEBUG ENDPOINTS
# ============================================================================

if settings.environment == "development":
    
    @app.get("/debug/config", tags=["Debug"])
    async def debug_config():
        """Show current configuration (dev only)"""
        return {
            "rag": settings.rag.model_dump(),
            "llm": settings.llm.model_dump(),
            "api": settings.api.model_dump()
        }
    
    @app.get("/debug/prompts", tags=["Debug"])
    async def debug_prompts():
        """Show current prompts (dev only)"""
        from app.config.settings import prompts
        return prompts


# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """API root endpoint"""
    return {
        "message": "Farmer AI Assistant API",
        "status": "operational",
        "docs": "/docs",
        "health": "/health"
    }


# ============================================================================
# STARTUP LOGGING
# ============================================================================

app_logger.info("✅ Farmer AI Assistant API initialized")
app_logger.info(f"Environment: {settings.environment}")
app_logger.info(f"LLM Model: {settings.llm.model_name}")
app_logger.info(f"Embedding Model: {settings.rag.embedding_model}")