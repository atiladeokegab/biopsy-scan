.PHONY: smoke install test data summary

install:
	uv sync --dev

test:
	uv run pytest -q

# The event smoke: offline, tiny invented fixture, no real data.
smoke: install test

data:
	./scripts/fetch_data.sh

summary:
	uv run python -I core/summary.py
