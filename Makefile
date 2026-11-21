.PHONY: smoke install test data summary demo tissue

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

# The demo page: validate its data files, then serve it (static; open http://localhost:8000, or `make demo PORT=8001` if 8000 is taken).
PORT ?= 8000
demo:
	python3 -I demo/check_data.py
	python3 -m http.server $(PORT) -d demo

# Regenerate demo/data/tissue.json from real Charville cells (fetches the data first; seeded, so deterministic).
tissue: data
	uv run python -I scripts/export_tissue.py
