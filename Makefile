PY ?= python3

.PHONY: install data train eval serve test lint docker all

install:            ## install runtime + dev dependencies
	$(PY) -m pip install -r requirements-dev.txt

data:               ## download Bitext dataset, de-duplicate, split
	$(PY) scripts/prepare_data.py

train: data         ## train intent classifier + write reports/classifier_metrics.json
	$(PY) scripts/train_classifier.py

eval:               ## retrieval + end-to-end benchmarks → reports/
	$(PY) scripts/evaluate_retrieval.py
	$(PY) scripts/evaluate_pipeline.py

serve:              ## run the API + dashboard on http://localhost:8000
	PYTHONPATH=src $(PY) -m uvicorn supportpilot.api:app --reload --port 8000

test:
	$(PY) -m pytest -q

lint:
	ruff check src scripts tests

docker:
	docker compose up --build

all: install train eval test
