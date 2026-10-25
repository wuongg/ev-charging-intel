# EV Fleet Charging Intelligence
# Day 1 live targets: install, data, audit, test.
# Day 2: forecast runs the full stack.
# Day 3: scenarios builds the K=50 bank. optimize runs P0 versus P3.

ifeq ($(OS),Windows_NT)
SHELL := cmd.exe
.SHELLFLAGS := /C
endif

.PHONY: install data audit forecast scenarios optimize evaluate chaos reproduce plots serve test

install:
	uv sync --frozen

data:
	uv run python scripts/download_data.py
	uv run python scripts/run_audit.py

audit:
	uv run python scripts/run_audit.py

forecast:
	uv run python scripts/train_forecasts.py

scenarios:
	uv run python scripts/build_scenarios.py

optimize:
	uv run python scripts/run_policies.py

evaluate:
	uv run python scripts/evaluate.py

chaos:
	uv run python scripts/run_chaos.py

reproduce:
	uv run python scripts/reproduce.py

plots:
	uv run python scripts/make_report.py

serve:
	uv run python -c "print('not implemented yet (week 4)')"

test:
	uv run pytest -q
