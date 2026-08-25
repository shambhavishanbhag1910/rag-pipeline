from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from app.db import AsyncSessionLocal
from app.schemas import EvaluationRequest
from app.services.evaluation import EvaluationService


async def run(
    tenant_id: str,
    dataset: str,
    include_llm_metrics: bool,
    output_dir: str,
    thresholds_path: str | None,
) -> None:
    async with AsyncSessionLocal() as session:
        result = await EvaluationService().run(
            session,
            EvaluationRequest(
                tenant_id=tenant_id,
                dataset_path=dataset,
                include_llm_metrics=include_llm_metrics,
            ),
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = output / f"evaluation_{timestamp}.json"
    csv_path = output / f"evaluation_{timestamp}.csv"
    payload = {
        "run_id": str(result.id),
        "status": result.status,
        "dataset": result.dataset_name,
        "metrics": result.metrics,
        "details": result.details,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(result.details).to_csv(csv_path, index=False)

    print(json.dumps(result.metrics, indent=2))
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")

    if thresholds_path:
        thresholds = json.loads(Path(thresholds_path).read_text(encoding="utf-8"))
        failures: list[str] = []
        for metric, minimum in thresholds.items():
            actual = result.metrics.get(metric)
            if not isinstance(actual, (int, float)) or float(actual) < float(minimum):
                failures.append(f"{metric}: actual={actual!r}, required>={minimum}")
        if failures:
            print("Evaluation threshold failures:")
            for failure in failures:
                print(f"- {failure}")
            raise SystemExit(2)
        print("All configured evaluation thresholds passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run automated Ragas evaluation")
    parser.add_argument("--tenant-id", default="demo")
    parser.add_argument(
        "--dataset",
        default="synthetic_data/evaluation/golden_dataset.jsonl",
    )
    parser.add_argument("--output-dir", default="artifacts/evaluations")
    parser.add_argument("--no-llm-metrics", action="store_true")
    parser.add_argument("--thresholds", default=None)
    args = parser.parse_args()
    asyncio.run(
        run(
            args.tenant_id,
            args.dataset,
            not args.no_llm_metrics,
            args.output_dir,
            args.thresholds,
        )
    )


if __name__ == "__main__":
    main()
