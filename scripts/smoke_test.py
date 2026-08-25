from __future__ import annotations

import asyncio
import json
import uuid

from app.core.config import get_settings
from app.db import AsyncSessionLocal
from app.schemas import QueryRequest
from app.services.rag import RAGService
from app.services.synthetic import seed_synthetic_data


async def run() -> None:
    settings = get_settings()
    async with AsyncSessionLocal() as session:
        await seed_synthetic_data(
            session,
            base_path=settings.synthetic_data_path,
            tenant_id="demo",
            reset=False,
        )
        response = await RAGService().answer(
            session,
            QueryRequest(
                question="How quickly must a Sev-1 security incident be acknowledged?",
                tenant_id="demo",
                use_cache=False,
            ),
            trace_id=f"smoke-{uuid.uuid4().hex}",
        )
    print(json.dumps(response.model_dump(mode="json"), indent=2))


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
