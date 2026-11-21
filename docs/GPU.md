# Real GPU runs on Modal

The CUDA kernels in `reimage/` are checked under the numba CUDA simulator in `make smoke`; real GPU
numbers come from [Modal](https://modal.com) through `gpu/modal_app.py`.

**Once:** `uv run modal setup` (browser login; the token stays in `~/.modal.toml`, never in the repo).

| Command | What runs | Output |
|---|---|---|
| `make modal-gpu-tests GPU=A100` | `reimage/test_backends.py`, `test_expression.py`, `test_fleet.py` on a real GPU, so the CUDA-gated tests run instead of skipping | pass/fail/skip counts, nvidia-smi model, memory, driver |
| `make modal-ladder GPU=A100 CONTAINERS=4 STUDIES=16 REPS=5` | `bench/real_cpu.py`'s Charville design point, NumPy reference vs GPU, same seeds, exact parity; the same studies on 1 and N GPU containers, merged bit-identical; R-15's `stats.screen` on the GPU when it is importable | one file |

`GPU` is `A100` (default), `H100` or `L4`. The first `modal-ladder` downloads the Zenodo cohort into
the `preflight-data` Modal volume (CPU only, once).

Each run writes `bench/results/modal-<gpu>-<utc>.json`, status `MEASURED`, with the GPU model,
driver, CUDA driver and runtime versions, the commit (and whether the tree was dirty), and container
counts. Timing rules:

- Warm-up (JIT, adjacency upload) is excluded and reported as `first_call_s`; samples report median, min and max.
- `modal_wall_s` is the call as seen from your machine; `in_function_s` is inside the container.
  Their difference, `startup_and_transfer_s`, is container start-up and transfer, reported on its own and
  never part of a speed-up. Data load and adjacency are under `preparation_s`.
- The NumPy reference runs in the same container as the GPU row. The EPYC rungs in
  `bench/results/cpu-ladder.json` use the same design on another host.
