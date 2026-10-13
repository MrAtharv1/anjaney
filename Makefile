.PHONY: setup check test format clean reproduce

setup:
	pip install -e ".[dev,ml]"
	pre-commit install

check:
	ruff check .
	ruff format --check .
	pytest

test:
	pytest --cov=src/anjaney tests/

format:
	ruff check --fix .
	ruff format .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf build dist *.egg-info

reproduce:
	@echo "Reproduction pipeline will be wired in Phase 7."