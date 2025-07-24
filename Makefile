.PHONY: sync
sync:
	uv sync --all-extras --all-packages --group dev

.PHONY: lint
lint:
	uv run ruff check src tests
	uv run ruff format src tests --check
	uv run mypy src tests

.PHONY: lint-fix
lint-fix:
	uv run ruff check src tests --fix
	uv run ruff format src tests
	uv run mypy src tests

.PHONY: test
test:
	uv run pytest tests

.PHONY: jupyter
jupyter:
	uv run jupyter lab
