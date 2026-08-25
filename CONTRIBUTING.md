# Contributing

1. Create a feature branch from `main`.
2. Install development dependencies with `python -m pip install -e ".[dev]"`.
3. Run `ruff check .` and `pytest -q tests/unit`.
4. Add or update tests for behavior changes.
5. Never commit credentials, customer data, model secrets, or generated evaluation artifacts.
6. Open a pull request describing the change, operational impact, and rollback approach.
