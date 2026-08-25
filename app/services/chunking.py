from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.config import get_settings


@dataclass(frozen=True)
class TextChunk:
    index: int
    content: str
    start_char: int
    end_char: int


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[TextChunk]:
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size_chars
    overlap = settings.chunk_overlap_chars if overlap is None else overlap
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    normalized = _normalize_text(text)
    if not normalized:
        return []

    chunks: list[TextChunk] = []
    start = 0
    index = 0
    length = len(normalized)

    while start < length:
        target_end = min(start + chunk_size, length)
        end = target_end
        if target_end < length:
            search_start = min(start + max(chunk_size // 2, 1), target_end)
            boundaries = [
                normalized.rfind("\n\n", search_start, target_end),
                normalized.rfind(". ", search_start, target_end),
                normalized.rfind("; ", search_start, target_end),
                normalized.rfind(" ", search_start, target_end),
            ]
            best = max(boundaries)
            if best > start:
                end = best + (2 if normalized[best : best + 2] in {". ", "; "} else 0)

        content = normalized[start:end].strip()
        if content:
            chunks.append(TextChunk(index=index, content=content, start_char=start, end_char=end))
            index += 1

        if end >= length:
            break
        next_start = max(end - overlap, start + 1)
        while next_start < length and normalized[next_start].isspace():
            next_start += 1
        start = next_start

    return chunks
