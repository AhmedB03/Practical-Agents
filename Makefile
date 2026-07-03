PY ?= python

.PHONY: setup ingest demo eval ablate test clean

setup:
	$(PY) -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

ingest:
	$(PY) -m src.practical_agents.ingest

demo:
	$(PY) scripts/demo.py

eval:
	$(PY) -m eval.run_eval

ablate:
	$(PY) -m eval.run_eval --ablate

test:
	$(PY) -m pytest -q

clean:
	rm -rf eval/index eval/results .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
