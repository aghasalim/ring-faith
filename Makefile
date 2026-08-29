PY := .venv/bin/python
.DEFAULT_GOAL := help

help:  ## show this help
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-12s %s\n", $$1, $$2}'

venv:  ## create the virtualenv and install the package
	python3 -m venv .venv && $(PY) -m pip install -q -U pip && $(PY) -m pip install -q -e ".[dev]"

test:  ## run the test suite
	$(PY) -m pytest tests/ -q

sweep:  ## run the full topology x camouflage sweep (~15 min)
	$(PY) experiments/run_sweep.py --seeds 5 --n-explain 25 --out reports

demo:  ## reproduce the degenerate-split finding
	$(PY) experiments/degenerate_split_demo.py

report:  ## rebuild the README tables from reports/*.csv
	$(PY) experiments/make_tables.py

all: test sweep demo report  ## everything

figures:  ## redraw the README figures from reports/*.csv
	$(PY) experiments/make_figures.py

clean:  ## remove caches
	rm -rf .pytest_cache **/__pycache__ src/*.egg-info

.PHONY: help venv test sweep demo report all clean report-check

report-check:  ## fail if reports/tables.md no longer matches the generator
	$(PY) experiments/make_tables.py --check
