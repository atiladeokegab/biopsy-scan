.PHONY: smoke install test data summary demo tissue deploy modal-gpu-tests modal-ladder

install:
	uv sync --dev

test:
	uv run pytest -q

# The event smoke: offline, tiny invented fixture, no real data. Plus the demo's race regression (Node).
smoke: install test
	node demo/race.test.mjs

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

# The public demo: .github/workflows/pages.yml publishes demo/ to GitHub Pages on every push to
# integration. This target checks the data the workflow will publish and prints the URL.
PAGES_URL ?= https://atiladeokegab.github.io/sota-parallelism-nov21/
deploy:
	python3 -I demo/check_data.py demo/data
	@echo "Deployed by .github/workflows/pages.yml on push to integration: $(PAGES_URL)"

# Real GPUs on Modal (docs/GPU.md). Needs `uv run modal setup` once; GPU=A100|H100|L4.
GPU ?= A100
CONTAINERS ?= 4
STUDIES ?= 16
REPS ?= 5
modal-gpu-tests: install
	uv run modal run gpu/modal_app.py::run_tests --gpu $(GPU)

modal-ladder: install
	uv run modal run gpu/modal_app.py::run_ladder --gpu $(GPU) --reps $(REPS) --containers $(CONTAINERS) --studies $(STUDIES)
