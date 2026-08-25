from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import require_api_key
from app.core.config import get_settings
from app.db import get_session
from app.models import EvaluationRun
from app.schemas import (
    CacheStats,
    EvaluationRequest,
    EvaluationResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryResponse,
    QueryRequest,
)
from app.services.cache import SemanticCacheService
from app.services.evaluation import EvaluationService
from app.services.ingestion import IngestionService
from app.services.loaders import load_file_bytes
from app.services.rag import RAGService
from app.services.synthetic import seed_synthetic_data

settings = get_settings()
router = APIRouter()


@router.get("/health/live", response_model=HealthResponse, tags=["health"])
async def liveness() -> HealthResponse:
    return HealthResponse(status="ok", service=settings.app_name, version="1.0.0")


@router.get("/health/ready", response_model=HealthResponse, tags=["health"])
async def readiness(session: AsyncSession = Depends(get_session)) -> HealthResponse:
    try:
        await session.execute(text("SELECT 1"))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}") from exc
    return HealthResponse(
        status="ready",
        service=settings.app_name,
        version="1.0.0",
        database="ok",
    )


@router.post(
    f"{settings.api_prefix}/documents/ingest",
    response_model=IngestResponse,
    dependencies=[Depends(require_api_key)],
    tags=["documents"],
)
async def ingest_documents(
    request: IngestRequest,
    session: AsyncSession = Depends(get_session),
) -> IngestResponse:
    results = await IngestionService().ingest_documents(
        session, request.tenant_id, request.documents
    )
    return IngestResponse(
        tenant_id=request.tenant_id,
        processed=len(results),
        documents=results,
    )


@router.post(
    f"{settings.api_prefix}/documents/upload",
    response_model=IngestResponse,
    dependencies=[Depends(require_api_key)],
    tags=["documents"],
)
async def upload_documents(
    files: list[UploadFile] = File(...),
    tenant_id: str = Form(default="demo"),
    category: str | None = Form(default=None),
    department: str | None = Form(default=None),
    session: AsyncSession = Depends(get_session),
) -> IngestResponse:
    from app.schemas import DocumentInput

    documents: list[DocumentInput] = []
    for file in files:
        payload = await file.read()
        if len(payload) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File {file.filename!r} exceeds {settings.max_upload_mb} MB",
            )
        try:
            content, metadata = load_file_bytes(file.filename or "upload.txt", payload)
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        metadata.update({"category": category, "department": department})
        metadata = {key: value for key, value in metadata.items() if value is not None}
        documents.append(
            DocumentInput(
                source_id=f"upload:{file.filename}:{uuid.uuid4().hex[:8]}",
                title=file.filename or "Uploaded document",
                content=content,
                source_uri=f"upload://{file.filename}",
                metadata=metadata,
            )
        )

    results = await IngestionService().ingest_documents(session, tenant_id, documents)
    return IngestResponse(tenant_id=tenant_id, processed=len(results), documents=results)


@router.post(
    f"{settings.api_prefix}/query",
    response_model=QueryResponse,
    dependencies=[Depends(require_api_key)],
    tags=["rag"],
)
async def query(
    request: QueryRequest,
    http_request: Request,
    session: AsyncSession = Depends(get_session),
) -> QueryResponse:
    trace_id = getattr(http_request.state, "trace_id", uuid.uuid4().hex)
    return await RAGService().answer(session, request, trace_id=trace_id)


@router.get(
    f"{settings.api_prefix}/cache/stats",
    response_model=CacheStats,
    dependencies=[Depends(require_api_key)],
    tags=["cache"],
)
async def cache_stats(
    tenant_id: str = "demo",
    session: AsyncSession = Depends(get_session),
) -> CacheStats:
    entries, total_hits = await SemanticCacheService().stats(session, tenant_id)
    return CacheStats(tenant_id=tenant_id, entries=entries, total_hits=total_hits)


@router.delete(
    f"{settings.api_prefix}/cache",
    dependencies=[Depends(require_api_key)],
    tags=["cache"],
)
async def clear_cache(
    tenant_id: str = "demo",
    session: AsyncSession = Depends(get_session),
) -> dict[str, int | str]:
    deleted = await SemanticCacheService().clear(session, tenant_id)
    return {"tenant_id": tenant_id, "deleted": deleted}


@router.post(
    f"{settings.api_prefix}/admin/seed-synthetic",
    dependencies=[Depends(require_api_key)],
    tags=["admin"],
)
async def seed_synthetic(
    tenant_id: str = "demo",
    reset: bool = False,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    results = await seed_synthetic_data(
        session,
        base_path=settings.synthetic_data_path,
        tenant_id=tenant_id,
        reset=reset,
    )
    return {
        "tenant_id": tenant_id,
        "documents": [result.model_dump() for result in results],
    }


@router.post(
    f"{settings.api_prefix}/evaluations/run",
    response_model=EvaluationResponse,
    dependencies=[Depends(require_api_key)],
    tags=["evaluation"],
)
async def run_evaluation(
    request: EvaluationRequest,
    session: AsyncSession = Depends(get_session),
) -> EvaluationResponse:
    project_root = Path.cwd().resolve()
    dataset_path = Path(request.dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = project_root / dataset_path
    dataset_path = dataset_path.resolve()
    if dataset_path.suffix.lower() != ".jsonl":
        raise HTTPException(status_code=400, detail="Evaluation dataset must be JSONL")
    if project_root != dataset_path and project_root not in dataset_path.parents:
        raise HTTPException(status_code=400, detail="Evaluation dataset must be inside the project directory")
    resolved_request = request.model_copy(update={"dataset_path": str(dataset_path)})
    run = await EvaluationService().run(session, resolved_request)
    return EvaluationResponse(
        run_id=run.id,
        status=run.status,
        metrics=run.metrics,
        details=run.details,
        created_at=run.created_at,
    )


@router.get(
    f"{settings.api_prefix}/evaluations/{{run_id}}",
    response_model=EvaluationResponse,
    dependencies=[Depends(require_api_key)],
    tags=["evaluation"],
)
async def get_evaluation(
    run_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> EvaluationResponse:
    run = await session.scalar(select(EvaluationRun).where(EvaluationRun.id == run_id))
    if run is None:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return EvaluationResponse(
        run_id=run.id,
        status=run.status,
        metrics=run.metrics,
        details=run.details,
        created_at=run.created_at,
    )
