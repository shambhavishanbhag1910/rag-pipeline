from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.schemas import QueryFilters, SourceItem


@dataclass(frozen=True)
class RetrievalResult:
    sources: list[SourceItem]


def vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"


class HybridRetriever:
    def __init__(self) -> None:
        self.settings = get_settings()

    async def search(
        self,
        session: AsyncSession,
        *,
        tenant_id: str,
        query: str,
        query_embedding: list[float],
        top_k: int,
        filters: QueryFilters,
    ) -> RetrievalResult:
        candidate_k = max(self.settings.retrieval_candidate_k, top_k * 4)
        sql = text(
            """
            WITH semantic AS (
                SELECT c.id,
                       row_number() OVER (ORDER BY c.embedding <=> CAST(:embedding AS vector)) AS rank
                FROM chunks c
                WHERE c.tenant_id = :tenant_id
                  AND (:category IS NULL OR c.metadata->>'category' = :category)
                  AND (:department IS NULL OR c.metadata->>'department' = :department)
                ORDER BY c.embedding <=> CAST(:embedding AS vector)
                LIMIT :candidate_k
            ),
            keyword AS (
                SELECT c.id,
                       row_number() OVER (
                         ORDER BY ts_rank_cd(c.search_vector, websearch_to_tsquery('english', :query)) DESC
                       ) AS rank
                FROM chunks c
                WHERE c.tenant_id = :tenant_id
                  AND c.search_vector @@ websearch_to_tsquery('english', :query)
                  AND (:category IS NULL OR c.metadata->>'category' = :category)
                  AND (:department IS NULL OR c.metadata->>'department' = :department)
                ORDER BY ts_rank_cd(c.search_vector, websearch_to_tsquery('english', :query)) DESC
                LIMIT :candidate_k
            ),
            fused AS (
                SELECT COALESCE(s.id, k.id) AS id,
                       COALESCE(:vector_weight / (:rrf_k + s.rank), 0.0) +
                       COALESCE(:keyword_weight / (:rrf_k + k.rank), 0.0) AS score
                FROM semantic s
                FULL OUTER JOIN keyword k ON s.id = k.id
            )
            SELECT c.id AS chunk_id,
                   c.document_id,
                   d.source_id,
                   d.title,
                   d.source_uri,
                   c.content,
                   c.metadata,
                   fused.score,
                   1 - (c.embedding <=> CAST(:embedding AS vector)) AS vector_score,
                   COALESCE(ts_rank_cd(c.search_vector, websearch_to_tsquery('english', :query)), 0.0)
                     AS keyword_score
            FROM fused
            JOIN chunks c ON c.id = fused.id
            JOIN documents d ON d.id = c.document_id
            ORDER BY fused.score DESC, vector_score DESC
            LIMIT :top_k
            """
        )
        rows = (
            await session.execute(
                sql,
                {
                    "embedding": vector_literal(query_embedding),
                    "tenant_id": tenant_id,
                    "query": query,
                    "candidate_k": candidate_k,
                    "top_k": top_k,
                    "rrf_k": self.settings.rrf_k,
                    "vector_weight": self.settings.vector_weight,
                    "keyword_weight": self.settings.keyword_weight,
                    "category": filters.category,
                    "department": filters.department,
                },
            )
        ).mappings()

        sources = [
            SourceItem(
                chunk_id=UUID(str(row["chunk_id"])),
                document_id=UUID(str(row["document_id"])),
                source_id=row["source_id"],
                title=row["title"],
                source_uri=row["source_uri"],
                content=row["content"],
                score=float(row["score"]),
                vector_score=float(row["vector_score"]),
                keyword_score=float(row["keyword_score"]),
                metadata=dict(row["metadata"] or {}),
            )
            for row in rows
        ]
        return RetrievalResult(sources=sources)
