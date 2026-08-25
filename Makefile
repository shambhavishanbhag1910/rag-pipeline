.PHONY: help install dev up down logs seed evaluate evaluate-ci test lint format smoke clean

help:
	@echo "install   Install project and dev dependencies"
	@echo "up        Build and start PostgreSQL + API"
	@echo "down      Stop the stack"
	@echo "seed      Ingest bundled synthetic documents"
	@echo "evaluate  Run automated Ragas + deterministic evaluation"
	@echo "evaluate-ci Run evaluation with quality thresholds"
	@echo "smoke     Seed and execute a sample query"
	@echo "test      Run unit tests"
	@echo "lint      Run Ruff"

install:
	python -m pip install -e ".[dev]"

dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f api

seed:
	docker compose exec api python -m scripts.seed --reset

evaluate:
	docker compose exec api python -m scripts.evaluate

evaluate-ci:
	docker compose exec api python -m scripts.evaluate --no-llm-metrics --thresholds synthetic_data/evaluation/thresholds.json

smoke:
	docker compose exec api python -m scripts.smoke_test

test:
	python -m pytest -q

lint:
	ruff check .

format:
	ruff format .

clean:
	docker compose down -v --remove-orphans
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
