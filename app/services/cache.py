from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import delete, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models import SemanticCache
from app.services.retrieval import vector_literal


@dataclass(frozen=True)
class CacheHit:
    cache_id: UUID
    answer: str
    sources: list[dict[str, Any]]
    similarity: float
    model_name: str


class SemanticCacheService:
    def __init__(self) -> None:
        self.settings = get_settings()

    async def lookup(
        self,
        session: AsyncSession,
        *,
        tenant_id: str,
        namespace: str,
        query_embedding: list[float],
    ) -> CacheHit | None:
        if not self.settings.cache_enabled:
            return None

        sql = text(
            """
            SELECT id, answer, sources, model_name,
                   1 - (query_embedding <=> CAST(:embedding AS vector)) AS similarity
            FROM semantic_cache
            WHERE tenant_id = :tenant_id
              AND namespace = :namespace
              AND prompt_version = :prompt_version
              AND expires_at > now()
              AND 1 - (query_embedding <=> CAST(:embedding AS vector)) >= :threshold
            ORDER BY query_embedding <=> CAST(:embedding AS vector)
            LIMIT 1
            """
        )
        row = (
            await session.execute(
                sql,
                {
                    "embedding": vector_literal(query_embedding),
                    "tenant_id": tenant_id,
                    "namespace": namespace,
                    "prompt_version": self.settings.prompt_version,
                    "threshold": self.settings.cache_similarity_threshold,
                },
            )
        ).mappings().first()
        if row is None:
            return None

        cache_id = UUID(str(row["id"]))
        await session.execute(
            update(SemanticCache)
            .where(SemanticCache.id == cache_id)
            .values(hit_count=SemanticCache.hit_count + 1, last_hit_at=func.now())
        )
        await session.commit()
        return CacheHit(
            cache_id=cache_id,
            answer=row["answer"],
            sources=list(row["sources"] or []),
            similarity=float(row["similarity"]),
            model_name=row["model_name"],
        )

    async def store(
        self,
        session: AsyncSession,
        *,
        tenant_id: str,
        namespace: str,
        query: str,
        query_embedding: list[float],
        answer: str,
        sources: list[dict[str, Any]],
        model_name: str,
    ) -> None:
        if not self.settings.cache_enabled:
            return
        expires_at = datetime.now(UTC) + timedelta(seconds=self.settings.cache_ttl_seconds)
        session.add(
            SemanticCache(
                tenant_id=tenant_id,
                namespace=namespace,
                query=query,
                query_embedding=query_embedding,
                answer=answer,
                sources=sources,
                model_name=model_name,
                prompt_version=self.settings.prompt_version,
                expires_at=expires_at,
            )
        )
        await session.commit()

    async def clear(self, session: AsyncSession, tenant_id: str) -> int:
        result = await session.execute(
            delete(SemanticCache).where(SemanticCache.tenant_id == tenant_id)
        )
        await session.commit()
        return int(result.rowcount or 0)

    async def stats(self, session: AsyncSession, tenant_id: str) -> tuple[int, int]:
        row = (
            await session.execute(
                select(func.count(SemanticCache.id), func.coalesce(func.sum(SemanticCache.hit_count), 0))
                .where(SemanticCache.tenant_id == tenant_id)
                .where(SemanticCache.expires_at > func.now())
            )
        ).one()
        return int(row[0]), int(row[1])
