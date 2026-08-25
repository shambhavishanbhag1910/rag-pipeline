from app.services.loaders import load_file_bytes


def test_markdown_loader() -> None:
    text, metadata = load_file_bytes("policy.md", b"# Policy\nContent")
    assert "Policy" in text
    assert metadata["file_type"] == "md"


def test_json_loader_formats_json() -> None:
    text, metadata = load_file_bytes("data.json", b'{"a":1}')
    assert '"a": 1' in text
    assert metadata["file_type"] == "json"
