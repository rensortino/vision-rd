.PHONY: install lint format test help notebook

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies and set up notebook output stripping
	uv sync
	uv run nbstripout --install

lint: ## Run ruff linter on code and notebooks
	uv run ruff check .

format: ## Format code and notebooks with ruff
	uv run ruff format .

test: ## Run tests with pytest
	uv run pytest tests/ -v

notebook: ## Launch JupyterLab for notebooks
	uv run jupyter lab notebooks/
