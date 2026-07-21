PYTHON ?= python3
VENV_PYTHON := .venv/bin/python
PYTHONPATH_VALUE := src

.PHONY: setup test solve-known analyze-problem2 result verify all

setup:
	$(PYTHON) -m venv .venv
	$(VENV_PYTHON) -m pip install -r requirements.txt

test:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) -m unittest discover -s tests -v

solve-known:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(VENV_PYTHON) scripts/solve_known_weather.py 1 2 \
		--time-limit 180 \
		--output output/problem1/known_weather_exact.json

analyze-problem2:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) scripts/analyze_problem2.py

result:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) scripts/fill_result.py \
		B题/Result.xlsx \
		output/problem1/known_weather_exact.json \
		output/final/Result.xlsx

verify: test
	unzip -t output/final/Result.xlsx

all: test solve-known analyze-problem2 result verify
