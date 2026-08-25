import pytest


def test_ragas_api_surface() -> None:
    pytest.importorskip("ragas")
    from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall
    from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

    assert IDBasedContextPrecision
    assert IDBasedContextRecall
    assert AnswerRelevancy
    assert ContextPrecision
    assert ContextRecall
    assert Faithfulness


def test_pgvector_sqlalchemy_surface() -> None:
    pytest.importorskip("pgvector")
    from pgvector.sqlalchemy import VECTOR

    assert VECTOR
