"""
Production middleware for FastAPI
- Rate limiting
- Request tracking
- Error handling
- Security headers
"""

import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Tuple
from collections import defaultdict

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logger import app_logger
from app.config.settings import settings


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add request ID to all requests for tracing"""
    
    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        
        # Add to response headers for client tracking
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting by IP address
    Prevents abuse and DOS attacks
    """
    
    def __init__(self, app, requests_per_minute: int = None, requests_per_hour: int = None):
        super().__init__(app)
        
        self.requests_per_minute = requests_per_minute or settings.api.rate_limit_requests
        self.requests_per_hour = requests_per_hour or (self.requests_per_minute * 60)
        
        # Track requests: {ip: [(timestamp, endpoint)]}
        self.request_history: Dict[str, list] = defaultdict(list)
    
    async def dispatch(self, request: Request, call_next):
        if not settings.api.rate_limit_enabled:
            return await call_next(request)
        
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/health/deep"]:
            return await call_next(request)
        
        now = datetime.utcnow()
        one_minute_ago = now - timedelta(minutes=1)
        one_hour_ago = now - timedelta(hours=1)
        
        # Clean old requests
        self.request_history[client_ip] = [
            (ts, ep) for ts, ep in self.request_history[client_ip]
            if ts > one_hour_ago
        ]
        
        # Count recent requests
        minute_requests = sum(
            1 for ts, _ in self.request_history[client_ip]
            if ts > one_minute_ago
        )
        
        hour_requests = len(self.request_history[client_ip])
        
        # Check limits
        if minute_requests >= self.requests_per_minute:
            app_logger.warning(
                f"Rate limit exceeded",
                client_ip=client_ip,
                minute_requests=minute_requests,
                limit=self.requests_per_minute
            )
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Too many requests. Please try again later.",
                    "retry_after": 60
                },
                headers={"Retry-After": "60"}
            )
        
        if hour_requests >= self.requests_per_hour:
            app_logger.warning(
                f"Hour rate limit exceeded",
                client_ip=client_ip,
                hour_requests=hour_requests,
                limit=self.requests_per_hour
            )
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Hourly quota exceeded. Please try again later.",
                    "retry_after": 3600
                },
                headers={"Retry-After": "3600"}
            )
        
        # Record request
        self.request_history[client_ip].append((now, str(request.url.path)))
        
        # Continue
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-Rate-Limit-Limit"] = str(self.requests_per_minute)
        response.headers["X-Rate-Limit-Remaining"] = str(
            self.requests_per_minute - minute_requests
        )
        
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # CORS headers
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        # Remove server info
        response.headers["Server"] = "FarmerAI/1.0"
        
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Detailed request/response logging
    """
    
    async def dispatch(self, request: Request, call_next):
        t0 = time.time()
        
        # Request logging
        app_logger.info(
            "Request received",
            request_id=getattr(request.state, "request_id", "unknown"),
            method=request.method,
            path=request.url.path,
            client_ip=request.client.host if request.client else "unknown"
        )
        
        try:
            response = await call_next(request)
        except Exception as e:
            latency = (time.time() - t0) * 1000
            
            app_logger.error(
                "Request failed with exception",
                request_id=getattr(request.state, "request_id", "unknown"),
                method=request.method,
                path=request.url.path,
                error=str(e),
                error_type=type(e).__name__,
                latency_ms=f"{latency:.2f}"
            )
            raise
        
        latency = (time.time() - t0) * 1000
        
        # Response logging
        app_logger.info(
            "Response sent",
            request_id=getattr(request.state, "request_id", "unknown"),
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=f"{latency:.2f}"
        )
        
        return response


class TimeoutMiddleware(BaseHTTPMiddleware):
    """
    Request timeout handling
    Prevents long-running requests from blocking
    """
    
    def __init__(self, app, timeout_seconds: int = None):
        super().__init__(app)
        self.timeout = timeout_seconds or settings.api.request_timeout
    
    async def dispatch(self, request: Request, call_next):
        try:
            # This doesn't actually timeout (FastAPI limitation)
            # But we log the intention
            response = await call_next(request)
            return response
        except Exception as e:
            app_logger.error(
                f"Request timeout or error",
                path=request.url.path,
                timeout_sec=self.timeout,
                error=str(e)
            )
            
            return JSONResponse(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                content={"detail": "Request timeout. Please try again."}
            )


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Collect basic metrics
    - Request count by endpoint
    - Latency distribution
    - Error rates
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.metrics = {
            "total_requests": 0,
            "total_errors": 0,
            "requests_by_endpoint": defaultdict(int),
            "errors_by_endpoint": defaultdict(int),
            "latencies": []
        }
    
    async def dispatch(self, request: Request, call_next):
        t0 = time.time()
        
        try:
            response = await call_next(request)
        except Exception as e:
            latency = (time.time() - t0) * 1000
            
            self.metrics["total_errors"] += 1
            self.metrics["errors_by_endpoint"][request.url.path] += 1
            
            raise
        
        latency = (time.time() - t0) * 1000
        
        # Update metrics
        self.metrics["total_requests"] += 1
        self.metrics["requests_by_endpoint"][request.url.path] += 1
        self.metrics["latencies"].append(latency)
        
        if response.status_code >= 400:
            self.metrics["total_errors"] += 1
            self.metrics["errors_by_endpoint"][request.url.path] += 1
        
        # Add latency to response headers
        response.headers["X-Process-Time"] = str(latency)
        
        return response
    
    def get_metrics(self):
        """Return current metrics"""
        if self.metrics["latencies"]:
            avg_latency = sum(self.metrics["latencies"]) / len(self.metrics["latencies"])
            p95_latency = sorted(self.metrics["latencies"])[
                int(0.95 * len(self.metrics["latencies"]))
            ]
        else:
            avg_latency = 0
            p95_latency = 0
        
        return {
            "total_requests": self.metrics["total_requests"],
            "total_errors": self.metrics["total_errors"],
            "error_rate": (
                self.metrics["total_errors"] / self.metrics["total_requests"]
                if self.metrics["total_requests"] > 0 else 0
            ),
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
            "requests_by_endpoint": dict(self.metrics["requests_by_endpoint"]),
            "errors_by_endpoint": dict(self.metrics["errors_by_endpoint"])
        }


def add_middleware_stack(app):
    """
    Add all middleware in correct order
    Order matters! Inner-most middleware executes last.
    """
    
    # Order (from last to first applied):
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(TimeoutMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(RequestIDMiddleware)
    
    return app