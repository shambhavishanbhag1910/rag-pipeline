from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Document
from app.schemas import DocumentInput
from app.services.ingestion import IngestionService


async def seed_synthetic_data(
    session: AsyncSession,
    *,
    base_path: str | Path,
    tenant_id: str = "demo",
    reset: bool = False,
) -> list[Any]:
    root = Path(base_path)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Synthetic manifest not found: {manifest_path}")

    if reset:
        await session.execute(delete(Document).where(Document.tenant_id == tenant_id))
        await session.commit()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents: list[DocumentInput] = []
    for item in manifest["documents"]:
        path = root / item["path"]
        documents.append(
            DocumentInput(
                source_id=item["source_id"],
                title=item["title"],
                content=path.read_text(encoding="utf-8"),
                source_uri=f"synthetic://{item['path']}",
                metadata=item.get("metadata", {}),
            )
        )
    return await IngestionService().ingest_documents(session, tenant_id, documents)
