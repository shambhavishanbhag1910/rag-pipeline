from app.services.chunking import chunk_text


def test_chunk_text_respects_size_and_overlap() -> None:
    text = "Paragraph one. " * 80 + "\n\n" + "Paragraph two. " * 80
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 2
    assert all(chunk.content for chunk in chunks)
    assert all(len(chunk.content) <= 305 for chunk in chunks)
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


def test_chunk_text_empty() -> None:
    assert chunk_text("   \n\n ") == []
