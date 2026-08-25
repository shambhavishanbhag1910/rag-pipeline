from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DocumentInput(BaseModel):
    source_id: str = Field(min_length=1, max_length=255)
    title: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1, max_length=5_000_000)
    source_uri: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    tenant_id: str = Field(default="demo", min_length=1, max_length=100)
    documents: list[DocumentInput] = Field(min_length=1, max_length=100)


class IngestedDocument(BaseModel):
    source_id: str
    title: str
    chunks: int
    status: str


class IngestResponse(BaseModel):
    tenant_id: str
    processed: int
    documents: list[IngestedDocument]


class QueryFilters(BaseModel):
    category: str | None = None
    department: str | None = None


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=5000)
    tenant_id: str = Field(default="demo", min_length=1, max_length=100)
    top_k: int | None = Field(default=None, ge=1, le=20)
    use_cache: bool = True
    filters: QueryFilters = Field(default_factory=QueryFilters)


class SourceItem(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    source_id: str
    title: str
    source_uri: str | None = None
    content: str
    score: float
    vector_score: float
    keyword_score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    tenant_id: str
    cached: bool
    cache_similarity: float | None = None
    model: str
    prompt_version: str
    sources: list[SourceItem]
    latency_ms: float
    trace_id: str


class CacheStats(BaseModel):
    tenant_id: str
    entries: int
    total_hits: int


class EvaluationRequest(BaseModel):
    tenant_id: str = "demo"
    dataset_path: str = "synthetic_data/evaluation/golden_dataset.jsonl"
    include_llm_metrics: bool = True


class EvaluationResponse(BaseModel):
    run_id: uuid.UUID
    status: str
    metrics: dict[str, float | int | str | None]
    details: list[dict[str, Any]]
    created_at: datetime


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    database: str | None = None


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    detail: str
    trace_id: str | None = None
