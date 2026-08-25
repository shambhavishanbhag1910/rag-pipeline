from __future__ import annotations

import asyncio
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings


class EmbeddingService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._model: SentenceTransformer | None = None
        self._lock = asyncio.Lock()

    async def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            async with self._lock:
                if self._model is None:
                    self._model = await asyncio.to_thread(
                        SentenceTransformer,
                        self.settings.embedding_model,
                        device=self.settings.embedding_device,
                    )
                    dimension = self._model.get_sentence_embedding_dimension()
                    if dimension != self.settings.embedding_dimension:
                        raise RuntimeError(
                            f"Embedding dimension mismatch: model={dimension}, "
                            f"configured={self.settings.embedding_dimension}"
                        )
        return self._model

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = await self._get_model()
        embeddings = await asyncio.to_thread(
            model.encode,
            texts,
            batch_size=self.settings.embedding_batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32).tolist()

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
