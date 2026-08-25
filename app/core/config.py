from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Enterprise RAG Pipeline"
    environment: str = "development"
    log_level: str = "INFO"
    api_prefix: str = "/v1"
    api_key: str | None = None
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    database_url: str = "postgresql+asyncpg://rag:rag_password@localhost:5432/ragdb"
    database_url_sync: str = "postgresql+psycopg://rag:rag_password@localhost:5432/ragdb"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32

    llm_provider: str = "extractive"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.0
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"

    retrieval_top_k: int = 5
    retrieval_candidate_k: int = 40
    rrf_k: int = 60
    vector_weight: float = 1.0
    keyword_weight: float = 1.0

    cache_enabled: bool = True
    cache_similarity_threshold: float = 0.93
    cache_ttl_seconds: int = 3600
    cache_namespace: str = "default"
    prompt_version: str = "v1"

    chunk_size_chars: int = 1200
    chunk_overlap_chars: int = 200
    max_context_chars: int = 12000
    max_upload_mb: int = 20

    ragas_enabled: bool = True
    ragas_llm_model: str = "gpt-4o-mini"
    ragas_embedding_model: str = "text-embedding-3-small"
    ragas_max_concurrency: int = 4

    auto_seed_synthetic_data: bool = False
    synthetic_data_path: Path = Path("synthetic_data")

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, value: str) -> str:
        normalized = value.lower().strip()
        if normalized not in {"extractive", "openai", "ollama"}:
            raise ValueError("LLM_PROVIDER must be one of: extractive, openai, ollama")
        return normalized

    @field_validator("cache_similarity_threshold")
    @classmethod
    def validate_cache_threshold(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("CACHE_SIMILARITY_THRESHOLD must be between 0 and 1")
        return value

    @field_validator("chunk_overlap_chars")
    @classmethod
    def validate_overlap(cls, value: int, info) -> int:  # type: ignore[no-untyped-def]
        size = info.data.get("chunk_size_chars", 1200)
        if value < 0 or value >= size:
            raise ValueError("CHUNK_OVERLAP_CHARS must be non-negative and smaller than chunk size")
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
