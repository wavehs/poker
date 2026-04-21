.PHONY: install dev test lint typecheck api ui synth docker clean

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
