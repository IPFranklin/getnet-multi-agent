.PHONY: install run test cov lint evals ingest docker-build docker-up docker-down docker-test demo

install:        ## install dev dependencies
	pip install -r requirements-dev.txt

run:            ## run the API locally with reload
	uvicorn app.main:app --reload --port 8000

test:           ## unit + integration tests (offline)
	LLM_PROVIDER=fake EMBEDDING_BACKEND=hash pytest -q

cov:
	LLM_PROVIDER=fake EMBEDDING_BACKEND=hash pytest -q --cov=app --cov-report=term-missing

lint:
	ruff check app tests evals

evals:          ## evaluation harness (uses .env: real LLM if configured)
	python -m evals.run_evals

ingest:         ## rebuild the knowledge index crawling live sources
	python -m app.rag.ingest --crawl

docker-build:
	docker compose build

docker-up:
	docker compose up -d && docker compose logs -f api

docker-down:
	docker compose down

docker-test:
	docker compose run --rm tests

demo:           ## send the challenge's example questions to a running API
	./scripts/demo.sh
