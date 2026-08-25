import os

import pytest
from sqlalchemy import text

from app.db import AsyncSessionLocal


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("RUN_INTEGRATION_TESTS"), reason="Set RUN_INTEGRATION_TESTS=1")
async def test_pgvector_extension_available() -> None:
    async with AsyncSessionLocal() as session:
        version = await session.scalar(text("SELECT extversion FROM pg_extension WHERE extname='vector'"))
    assert version is not None
