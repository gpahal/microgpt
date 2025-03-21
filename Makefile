.PHONY: sync
sync:
	uv sync --all-extras --all-packages --group dev

.PHONY: lint
lint:
	uv run ruff check src
	uv run ruff format src --check
	uv run mypy src

.PHONY: lint-fix
lint-fix:
	uv run ruff check src --fix
	uv run ruff format src

.PHONY: tests
tests:
	uv run pytest tests

.PHONY: jupyter
jupyter:
	uv run jupyter lab
