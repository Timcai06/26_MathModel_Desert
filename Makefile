PYTHON ?= python3
LATEXMK ?= latexmk
VENV_PYTHON := .venv/bin/python
PYTHONPATH_VALUE := src

.PHONY: setup test solve-known analyze-problem2 optimize-level4 analyze-problem3 analyze-robustness figures sync-paper-code paper-assets paper result verify deliver all

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

optimize-level4:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) scripts/optimize_level4.py

analyze-problem3:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) scripts/analyze_problem3.py

analyze-robustness:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) scripts/analyze_robustness.py

figures:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(VENV_PYTHON) scripts/generate_figures.py

sync-paper-code:
	mkdir -p paper/code
	cp src/desert/optimizer.py paper/code/optimizer.py
	cp src/desert/level3_mdp.py paper/code/level3_mdp.py
	cp src/desert/level5_game.py paper/code/level5_game.py
	cp src/desert/multiplayer.py paper/code/multiplayer.py
	cp src/desert/evaluation.py paper/code/evaluation.py
	cp src/desert/weather.py paper/code/weather.py
	cp src/desert/level6_oracle.py paper/code/level6_oracle.py
	cp scripts/analyze_robustness.py paper/code/analyze_robustness.py

paper-assets: figures
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) scripts/generate_paper_tables.py

paper: paper-assets sync-paper-code
	cd paper && $(LATEXMK) -norc -xelatex -interaction=nonstopmode -halt-on-error -outdir=build main.tex

result:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) scripts/fill_result.py \
		B题/Result.xlsx \
		output/problem1/known_weather_exact.json \
		output/final/Result.xlsx

verify: test result
	unzip -t output/final/Result.xlsx

deliver: paper result
	mkdir -p output/final output/pdf
	cp paper/build/main.pdf "output/final/穿越沙漠策略研究_完整稿.pdf"
	cp paper/build/main.pdf "output/pdf/穿越沙漠策略研究_完整稿.pdf"

all: solve-known analyze-problem2 optimize-level4 analyze-problem3 analyze-robustness verify deliver
