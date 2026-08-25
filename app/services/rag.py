from __future__ import annotations

import time

import structlog

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.metrics import CACHE_LOOKUPS, GENERATION_LATENCY, RAG_QUERIES, RETRIEVAL_LATENCY
from app.schemas import QueryRequest, QueryResponse, SourceItem
from app.services.cache import SemanticCacheService
from app.services.embeddings import EmbeddingService, get_embedding_service
from app.services.llm import BaseLLMProvider, get_llm_provider
from app.services.retrieval import HybridRetriever

logger = structlog.get_logger(__name__)


class RAGService:
    def __init__(
        self,
        embeddings: EmbeddingService | None = None,
        retriever: HybridRetriever | None = None,
        cache: SemanticCacheService | None = None,
        llm: BaseLLMProvider | None = None,
    ) -> None:
        self.settings = get_settings()
        self.embeddings = embeddings or get_embedding_service()
        self.retriever = retriever or HybridRetriever()
        self.cache = cache or SemanticCacheService()
        self.llm = llm or get_llm_provider()

    async def answer(
        self,
        session: AsyncSession,
        request: QueryRequest,
        trace_id: str,
    ) -> QueryResponse:
        started = time.perf_counter()
        query_embedding = await self.embeddings.embed_query(request.question)
        top_k = request.top_k or self.settings.retrieval_top_k
        namespace = (
            f"{self.settings.cache_namespace}:{self.llm.model_name}:top{top_k}:"
            f"{request.filters.category or '*'}:{request.filters.department or '*'}"
        )

        if request.use_cache:
            cache_hit = await self.cache.lookup(
                session,
                tenant_id=request.tenant_id,
                namespace=namespace,
                query_embedding=query_embedding,
            )
            if cache_hit:
                CACHE_LOOKUPS.labels("hit").inc()
                RAG_QUERIES.labels("hit", cache_hit.model_name).inc()
                sources = [SourceItem.model_validate(item) for item in cache_hit.sources]
                logger.info("rag_cache_hit", trace_id=trace_id, similarity=cache_hit.similarity)
                return QueryResponse(
                    answer=cache_hit.answer,
                    tenant_id=request.tenant_id,
                    cached=True,
                    cache_similarity=cache_hit.similarity,
                    model=cache_hit.model_name,
                    prompt_version=self.settings.prompt_version,
                    sources=sources,
                    latency_ms=round((time.perf_counter() - started) * 1000, 2),
                    trace_id=trace_id,
                )

        if request.use_cache:
            CACHE_LOOKUPS.labels("miss").inc()
        retrieval_started = time.perf_counter()
        retrieval = await self.retriever.search(
            session,
            tenant_id=request.tenant_id,
            query=request.question,
            query_embedding=query_embedding,
            top_k=top_k,
            filters=request.filters,
        )
        RETRIEVAL_LATENCY.observe(time.perf_counter() - retrieval_started)
        generation_started = time.perf_counter()
        answer = await self.llm.generate(request.question, retrieval.sources)
        GENERATION_LATENCY.labels(self.llm.model_name).observe(
            time.perf_counter() - generation_started
        )
        RAG_QUERIES.labels("miss", self.llm.model_name).inc()

        if request.use_cache and retrieval.sources:
            await self.cache.store(
                session,
                tenant_id=request.tenant_id,
                namespace=namespace,
                query=request.question,
                query_embedding=query_embedding,
                answer=answer,
                sources=[source.model_dump(mode="json") for source in retrieval.sources],
                model_name=self.llm.model_name,
            )

        logger.info(
            "rag_answer_generated",
            trace_id=trace_id,
            source_count=len(retrieval.sources),
            model=self.llm.model_name,
        )
        return QueryResponse(
            answer=answer,
            tenant_id=request.tenant_id,
            cached=False,
            model=self.llm.model_name,
            prompt_version=self.settings.prompt_version,
            sources=retrieval.sources,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            trace_id=trace_id,
        )
