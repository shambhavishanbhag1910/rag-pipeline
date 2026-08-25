from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

from pypdf import PdfReader


def load_file_bytes(filename: str, content: bytes) -> tuple[str, dict[str, Any]]:
    suffix = Path(filename).suffix.lower()
    metadata: dict[str, Any] = {"filename": filename, "file_type": suffix.lstrip(".") or "text"}

    if suffix == ".pdf":
        reader = PdfReader(BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
        metadata["page_count"] = len(pages)
        return "\n\n".join(pages).strip(), metadata

    decoded = content.decode("utf-8", errors="replace")
    if suffix == ".json":
        parsed = json.loads(decoded)
        return json.dumps(parsed, indent=2, ensure_ascii=False), metadata

    if suffix not in {".txt", ".md", ".markdown", ".csv", ".json", ""}:
        raise ValueError(f"Unsupported file type: {suffix}")
    return decoded.strip(), metadata
