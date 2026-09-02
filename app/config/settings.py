
from typing import Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ============================================================================
# RAG
# ============================================================================

class RAGSettings(BaseModel):
    """RAG pipeline configuration."""

    embedding_model: str = "BAAI/bge-base-en-v1.5"

    # Retrieval
    retrieval_threshold: float = 0.3
    retrieval_k: int = 10
    final_k: int = 5

    # Quality gates
    min_recall_k: float = 0.60
    min_mrr: float = 0.50


# ============================================================================
# LLM
# ============================================================================

class LLMSettings(BaseModel):
    """LLM configuration."""

    model_name: str = "llama3"
    model_path: str = "http://localhost:11434/api/generate"

    # Generation
    temperature: float = 0.3
    top_k: int = 20
    top_p: float = 0.9
    max_tokens: int = 500

    # Connection
    keep_alive: str = "5m"
    inference_timeout: int = 30


# ============================================================================
# QDRANT
# ============================================================================

class QdrantSettings(BaseModel):
    """Qdrant vector database configuration."""

    host: str = "localhost"
    port: int = 6333
    api_key: Optional[str] = None
    collection_name: str = "farming-docs"


# ============================================================================
# OBSERVABILITY
# ============================================================================

class ObservabilitySettings(BaseModel):
    """Monitoring and observability configuration."""

    langfuse_enabled: bool = True
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: Optional[str] = None

    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/app.log"

    # Metrics
    enable_metrics: bool = True
    metrics_port: int = 8001


# ============================================================================
# API
# ============================================================================

class APISettings(BaseModel):
    """API configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False

    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_period: int = 60

    # Request timeout
    request_timeout: int = 30


# ============================================================================
# ROOT SETTINGS
# ============================================================================

class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Nested configuration
    rag: RAGSettings = Field(default_factory=RAGSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    observability: ObservabilitySettings = Field(
        default_factory=ObservabilitySettings
    )
    api: APISettings = Field(default_factory=APISettings)

    # Application
    app_name: str = "Farmer AI"
    version: str = "1.0.0"
    environment: str = "development"


# ============================================================================
# PROMPTS
# ============================================================================

def load_prompts():
    """Load prompts from YAML configuration."""
    with open("app/config/prompts.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================================
# GLOBAL SETTINGS
# ============================================================================

settings = Settings()
prompts = load_prompts()

