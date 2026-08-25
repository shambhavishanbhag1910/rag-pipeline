from app.services.metrics import reciprocal_rank, token_f1


def test_token_f1_exact_match() -> None:
    assert token_f1("alpha beta", "alpha beta") == 1.0


def test_token_f1_partial_match() -> None:
    score = token_f1("alpha beta gamma", "alpha beta delta")
    assert 0.6 < score < 0.8


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["A", "B", "C"], ["B"]) == 0.5
    assert reciprocal_rank(["A", "B"], ["Z"]) == 0.0
