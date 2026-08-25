from __future__ import annotations

import hashlib
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.metrics import INGESTED_CHUNKS, INGESTED_DOCUMENTS
from app.models import Chunk, Document, SemanticCache
from app.schemas import DocumentInput, IngestedDocument
from app.services.chunking import chunk_text
from app.services.embeddings import EmbeddingService, get_embedding_service


class IngestionService:
    def __init__(self, embeddings: EmbeddingService | None = None) -> None:
        self.embeddings = embeddings or get_embedding_service()

    async def ingest_documents(
        self,
        session: AsyncSession,
        tenant_id: str,
        documents: list[DocumentInput],
    ) -> list[IngestedDocument]:
        results: list[IngestedDocument] = []
        for document in documents:
            result = await self.ingest_document(session, tenant_id, document)
            results.append(result)
            INGESTED_DOCUMENTS.labels(result.status).inc()
            if result.status != "unchanged":
                INGESTED_CHUNKS.inc(result.chunks)
        if any(result.status != "unchanged" for result in results):
            await session.execute(delete(SemanticCache).where(SemanticCache.tenant_id == tenant_id))
        await session.commit()
        return results

    async def ingest_document(
        self,
        session: AsyncSession,
        tenant_id: str,
        document: DocumentInput,
    ) -> IngestedDocument:
        content_hash = hashlib.sha256(document.content.encode("utf-8")).hexdigest()
        existing = await session.scalar(
            select(Document).where(
                Document.tenant_id == tenant_id,
                Document.source_id == document.source_id,
            )
        )

        if existing and existing.content_hash == content_hash:
            count = len(chunk_text(document.content))
            return IngestedDocument(
                source_id=document.source_id,
                title=document.title,
                chunks=count,
                status="unchanged",
            )

        chunks = chunk_text(document.content)
        if not chunks:
            raise ValueError(f"Document {document.source_id!r} contains no indexable text")

        vectors = await self.embeddings.embed_documents([chunk.content for chunk in chunks])

        status = "created"
        if existing:
            status = "updated"
            await session.execute(delete(Chunk).where(Chunk.document_id == existing.id))
            existing.title = document.title
            existing.source_uri = document.source_uri
            existing.content_hash = content_hash
            existing.metadata_ = document.metadata
            db_document = existing
        else:
            db_document = Document(
                tenant_id=tenant_id,
                source_id=document.source_id,
                title=document.title,
                source_uri=document.source_uri,
                content_hash=content_hash,
                metadata_=document.metadata,
            )
            session.add(db_document)
            await session.flush()

        for chunk, vector in zip(chunks, vectors, strict=True):
            metadata: dict[str, Any] = {
                **document.metadata,
                "source_id": document.source_id,
                "title": document.title,
                "chunk_index": chunk.index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
            }
            session.add(
                Chunk(
                    document_id=db_document.id,
                    tenant_id=tenant_id,
                    chunk_index=chunk.index,
                    content=chunk.content,
                    metadata_=metadata,
                    embedding=vector,
                )
            )

        await session.flush()
        return IngestedDocument(
            source_id=document.source_id,
            title=document.title,
            chunks=len(chunks),
            status=status,
        )
