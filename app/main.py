from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import text

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.metrics import HTTP_LATENCY, HTTP_REQUESTS
from app.db import AsyncSessionLocal
from app.services.synthetic import seed_synthetic_data

settings = get_settings()
configure_logging(settings.log_level)
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    async with AsyncSessionLocal() as session:
        await session.execute(text("SELECT 1"))
        if settings.auto_seed_synthetic_data:
            results = await seed_synthetic_data(
                session,
                base_path=settings.synthetic_data_path,
                tenant_id="demo",
                reset=False,
            )
            logger.info("synthetic_data_seeded", documents=len(results))
    yield


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description=(
        "Enterprise RAG API with PostgreSQL/pgvector hybrid retrieval, semantic caching, "
        "multi-tenant isolation, and automated Ragas evaluation."
    ),
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def observability_middleware(request: Request, call_next: Any) -> Response:
    trace_id = request.headers.get("X-Request-ID", uuid.uuid4().hex)
    request.state.trace_id = trace_id
    started = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception:
        logger.exception(
            "request_failed",
            trace_id=trace_id,
            method=request.method,
            path=request.url.path,
        )
        raise
    finally:
        elapsed = time.perf_counter() - started
        route = request.scope.get("route")
        metric_path = getattr(route, "path", request.url.path)
        HTTP_REQUESTS.labels(request.method, metric_path, str(status_code)).inc()
        HTTP_LATENCY.labels(request.method, metric_path).observe(elapsed)
        logger.info(
            "request_complete",
            trace_id=trace_id,
            method=request.method,
            path=request.url.path,
            status=status_code,
            latency_ms=round(elapsed * 1000, 2),
        )
    response.headers["X-Request-ID"] = trace_id
    return response


@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {
        "service": settings.app_name,
        "docs": "/docs",
        "health": "/health/ready",
    }


app.include_router(router)
