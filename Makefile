.PHONY: setup check test test-python test-web format clean reproduce parity demo help

PYTHON ?= python
PIP ?= $(PYTHON) -m pip
PYTEST ?= $(PYTHON) -m pytest
RUFF ?= $(PYTHON) -m ruff

help:
	@echo "ANJANEY Monorepo Command Suite"
	@echo "--------------------------------------------------------"
	@echo "make setup        - Install editable package and dependencies"
	@echo "make check        - Run linters, format checks, and test suites"
	@echo "make test         - Run both Python and TypeScript test suites"
	@echo "make format       - Auto-fix code formatting with Ruff"
	@echo "make reproduce    - Execute full deterministic research pipeline"
	@echo "make parity       - Validate TypeScript in-browser parity"
	@echo "make demo         - Build and start full containerized demo"
	@echo "make clean        - Remove caches and build artifacts"

setup:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev,ml,api]"
	pre-commit install
	cd apps/web && npm install

check:
	$(RUFF) check .
	$(RUFF) format --check .
	$(PYTEST)
	cd apps/web && npm test

format:
	$(RUFF) check --fix .
	$(RUFF) format .

test: test-python test-web

test-python:
	$(PYTEST)

test-web:
	cd apps/web && npm test

parity:
	$(PYTHON) -m anjaney.cli export bundle --features-file data/processed/features.parquet --output-dir data/processed/bundle
	cd apps/web && npm test

reproduce:
	anjaney data generate-synthetic --n-participants 50 --output-dir data/synthetic
	anjaney data build --source synthetic --output-dir data/processed --window-size 5
	anjaney features build --windows-file data/processed/windows.parquet --output-file data/processed/features.parquet --groups text_embed,timing,lexical
	anjaney train baseline --features-file data/processed/features.parquet --splits-file data/processed/splits.json
	anjaney rq1 run --features-file data/processed/features.parquet --prereg-file configs/prereg.yaml
	anjaney conformal run --features-file data/processed/features.parquet --splits-file data/processed/splits.json --alpha 0.10
	anjaney eval run --features-file data/processed/features.parquet --splits-file data/processed/splits.json
	anjaney export bundle --features-file data/processed/features.parquet --output-dir data/processed/bundle
	anjaney export benchmark --output-dir docs

demo:
	docker compose up --build

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf build dist *.egg-info .coverage