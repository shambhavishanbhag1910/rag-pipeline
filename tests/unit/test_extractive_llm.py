import uuid

import pytest

from app.schemas import SourceItem
from app.services.llm import ExtractiveLLMProvider


@pytest.mark.asyncio
async def test_extractive_provider_returns_cited_answer() -> None:
    source = SourceItem(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        source_id="IR-001",
        title="Incident Response",
        content="A Severity 1 incident must be acknowledged within 15 minutes.",
        score=0.1,
        vector_score=0.9,
        keyword_score=0.5,
    )
    answer = await ExtractiveLLMProvider().generate(
        "How quickly is a Severity 1 incident acknowledged?", [source]
    )
    assert "15 minutes" in answer
    assert "[S1]" in answer
