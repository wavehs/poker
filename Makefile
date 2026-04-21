.PHONY: install dev test lint typecheck api ui synth docker clean benchmark-capture benchmark-pipeline train-yolo dataset-gen

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v

lint:
	ruff check .

typecheck:
	mypy libs/ services/ apps/

api:
	uvicorn apps.api.main:app --reload --host 0.0.0.0 --port 8000

ui:
	python -m http.server 3000 --directory apps/ui

synth:
	python -m data.synthetic_tables.generator

docker:
	docker-compose up --build

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov/

# ─── Phase 2 targets ─────────────────────────────────────────────────────

benchmark-capture:
	python -m services.capture_agent.benchmarks

benchmark-pipeline:
	python -m evals.benchmark_pipeline

train-yolo:
	python -m services.vision_core.train train --data data/labeled_frames/data.yaml --epochs 50

dataset-gen:
	python -m data.dataset generate --count 100 --output data/labeled_frames

dataset-validate:
	python -m data.dataset validate --path data/labeled_frames

dataset-split:
	python -m data.dataset split --path data/labeled_frames --ratio 0.7 0.2 0.1
