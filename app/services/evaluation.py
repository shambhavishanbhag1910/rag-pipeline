from __future__ import annotations

import json
import math
import uuid
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models import EvaluationRun
from app.schemas import EvaluationRequest, QueryRequest
from app.services.embeddings import EmbeddingService, get_embedding_service
from app.services.metrics import reciprocal_rank, token_f1
from app.services.rag import RAGService



def _safe_float(value: Any) -> float | None:
    try:
        number = float(value.value if hasattr(value, "value") else value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class EvaluationService:
    def __init__(
        self,
        rag: RAGService | None = None,
        embeddings: EmbeddingService | None = None,
    ) -> None:
        self.settings = get_settings()
        self.rag = rag or RAGService()
        self.embeddings = embeddings or get_embedding_service()

    @staticmethod
    def load_dataset(path: str | Path) -> list[dict[str, Any]]:
        dataset_path = Path(path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")
        with dataset_path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if not rows:
            raise ValueError("Evaluation dataset is empty")
        return rows

    async def run(
        self,
        session: AsyncSession,
        request: EvaluationRequest,
    ) -> EvaluationRun:
        run = EvaluationRun(
            status="running",
            dataset_name=Path(request.dataset_path).name,
            metrics={},
            details=[],
        )
        session.add(run)
        await session.commit()
        await session.refresh(run)

        try:
            rows = self.load_dataset(request.dataset_path)
            details: list[dict[str, Any]] = []
            answer_pairs: list[tuple[str, str]] = []

            for row in rows:
                response = await self.rag.answer(
                    session,
                    QueryRequest(
                        question=row["question"],
                        tenant_id=request.tenant_id,
                        top_k=int(row.get("top_k", self.settings.retrieval_top_k)),
                        use_cache=False,
                    ),
                    trace_id=f"eval-{uuid.uuid4()}",
                )
                retrieved_ids = [source.source_id for source in response.sources]
                reference_ids = list(row.get("reference_context_ids", []))
                answer_pairs.append((response.answer, row["reference_answer"]))
                detail: dict[str, Any] = {
                    "question": row["question"],
                    "reference_answer": row["reference_answer"],
                    "answer": response.answer,
                    "retrieved_context_ids": retrieved_ids,
                    "reference_context_ids": reference_ids,
                    "retrieved_contexts": [source.content for source in response.sources],
                    "source_hit": float(bool(set(retrieved_ids) & set(reference_ids))),
                    "mrr": reciprocal_rank(retrieved_ids, reference_ids),
                    "token_f1": token_f1(response.answer, row["reference_answer"]),
                    "latency_ms": response.latency_ms,
                }
                details.append(detail)

            await self._add_ragas_id_metrics(details)
            await self._add_semantic_answer_similarity(details, answer_pairs)

            llm_metrics_status = "skipped"
            if request.include_llm_metrics and self.settings.ragas_enabled:
                if self.settings.openai_api_key:
                    await self._add_ragas_llm_metrics(details)
                    llm_metrics_status = "completed"
                else:
                    llm_metrics_status = "skipped_missing_openai_api_key"

            metric_names = [
                "source_hit",
                "mrr",
                "token_f1",
                "semantic_answer_similarity",
                "ragas_id_context_precision",
                "ragas_id_context_recall",
                "ragas_faithfulness",
                "ragas_answer_relevancy",
                "ragas_context_precision",
                "ragas_context_recall",
                "ragas_factual_correctness",
                "latency_ms",
            ]
            metrics: dict[str, float | int | str | None] = {
                "samples": len(details),
                "ragas_llm_metrics_status": llm_metrics_status,
            }
            for name in metric_names:
                values = [float(item[name]) for item in details if item.get(name) is not None]
                if values:
                    metrics[f"mean_{name}"] = round(mean(values), 6)

            run.status = "completed"
            run.metrics = metrics
            run.details = details
            run.completed_at = datetime.now(UTC)
            await session.commit()
            await session.refresh(run)
            return run
        except Exception as exc:
            run.status = "failed"
            run.error = str(exc)
            run.completed_at = datetime.now(UTC)
            await session.commit()
            raise

    async def _add_ragas_id_metrics(self, details: list[dict[str, Any]]) -> None:
        from ragas import SingleTurnSample
        from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall

        precision_metric = IDBasedContextPrecision()
        recall_metric = IDBasedContextRecall()
        for detail in details:
            sample = SingleTurnSample(
                retrieved_context_ids=detail["retrieved_context_ids"],
                reference_context_ids=detail["reference_context_ids"],
            )
            precision = await precision_metric.single_turn_ascore(sample)
            recall = await recall_metric.single_turn_ascore(sample)
            detail["ragas_id_context_precision"] = _safe_float(precision)
            detail["ragas_id_context_recall"] = _safe_float(recall)

    async def _add_semantic_answer_similarity(
        self,
        details: list[dict[str, Any]],
        answer_pairs: list[tuple[str, str]],
    ) -> None:
        flat = [text for pair in answer_pairs for text in pair]
        vectors = await self.embeddings.embed_documents(flat)
        for index, detail in enumerate(details):
            predicted = np.asarray(vectors[index * 2], dtype=np.float32)
            reference = np.asarray(vectors[index * 2 + 1], dtype=np.float32)
            similarity = float(np.dot(predicted, reference))
            detail["semantic_answer_similarity"] = max(-1.0, min(1.0, similarity))

    async def _add_ragas_llm_metrics(self, details: list[dict[str, Any]]) -> None:
        from openai import AsyncOpenAI
        from ragas.embeddings.base import embedding_factory
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            FactualCorrectness,
            Faithfulness,
        )

        client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        llm = llm_factory(self.settings.ragas_llm_model, client=client)
        embeddings = embedding_factory(
            "openai",
            model=self.settings.ragas_embedding_model,
            client=client,
        )
        metrics = {
            "ragas_faithfulness": Faithfulness(llm=llm),
            "ragas_answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
            "ragas_context_precision": ContextPrecision(llm=llm),
            "ragas_context_recall": ContextRecall(llm=llm),
            "ragas_factual_correctness": FactualCorrectness(llm=llm),
        }

        for detail in details:
            common = {
                "user_input": detail["question"],
                "response": detail["answer"],
                "reference": detail["reference_answer"],
                "retrieved_contexts": detail["retrieved_contexts"],
            }
            calls = {
                "ragas_faithfulness": {
                    "user_input": common["user_input"],
                    "response": common["response"],
                    "retrieved_contexts": common["retrieved_contexts"],
                },
                "ragas_answer_relevancy": {
                    "user_input": common["user_input"],
                    "response": common["response"],
                },
                "ragas_context_precision": {
                    "user_input": common["user_input"],
                    "reference": common["reference"],
                    "retrieved_contexts": common["retrieved_contexts"],
                },
                "ragas_context_recall": {
                    "user_input": common["user_input"],
                    "reference": common["reference"],
                    "retrieved_contexts": common["retrieved_contexts"],
                },
                "ragas_factual_correctness": {
                    "response": common["response"],
                    "reference": common["reference"],
                },
            }
            for name, scorer in metrics.items():
                try:
                    result = await scorer.ascore(**calls[name])
                    detail[name] = _safe_float(result)
                except Exception as exc:  # Evaluation should continue and expose per-metric failures.
                    detail[name] = None
                    detail[f"{name}_error"] = str(exc)
