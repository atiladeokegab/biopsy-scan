# Real GPU runs on Modal

The CUDA kernels in `reimage/` are checked under the numba CUDA simulator in `make smoke`; real GPU
numbers come from [Modal](https://modal.com) through `gpu/modal_app.py`.

**Once:** `uv run modal setup` (browser login; the token stays in `~/.modal.toml`, never in the repo).

| Command | What runs | Output |
|---|---|---|
| `make modal-gpu-tests GPU=A100-40GB` | `reimage/test_backends.py`, `test_expression.py`, `test_fleet.py` in a container with a real GPU and the CUDA simulator off (it refuses to run otherwise). Three tests start their own `NUMBA_ENABLE_CUDASIM=1` subprocess; they are counted as simulator coverage, not device coverage. Only the skip that needs CUDA to be absent is expected; any other skip exits non-zero | pass/fail/skip counts, simulator vs in-process coverage, nvidia-smi model, memory, driver |
| `make modal-ab GPU=A100-40GB` | R-14 full Cox fits vs R-15 `stats.screen` on CPU and GPU, interleaved, on one 16-core GPU container; input fingerprints, seed and B recorded | B vs C is equal work only if every screen output matches (U, V, z, n, maxT, p, adjusted_p, null_maxT) |
| `make modal-gpu-vs STUDIES=1000` | the same studies on one A100-40GB and one H100, block by block | seconds and dollars per study, bit-identical across GPUs |
| `make modal-ladder GPU=A100-40GB CONTAINERS=4 STUDIES=16 REPS=5` | `bench/real_cpu.py`'s Charville design point, NumPy reference vs GPU, same seeds, exact parity; the same studies on 1 and N GPU containers, merged bit-identical; R-15's `stats.screen` on the GPU when it is importable | one file |

`GPU` is `A100-40GB` (default), `A100-80GB`, `H100` or `L4`: an exact type, since plain `A100` mixed 40 GB and 80 GB cards in one curve. The cohort lives in the `preflight-data` Modal volume.
Zenodo answers 403 to Modal's IPs, so upload it once from a machine that has `make data`'s files:
`uv run modal volume put preflight-data data/charville_raw_data.zip /charville_raw_data.zip` (and
`charville_labels.csv`). Runs stop with that hint when the files are missing.

Each run writes `bench/results/modal-<gpu>-<utc>.json`, status `MEASURED`, with the GPU model,
driver, CUDA driver and runtime versions, the commit (and whether the tree was dirty), and container
counts. Timing rules:

- Warm-up (JIT, adjacency upload) is excluded and reported as `first_call_s`; samples report median, min and max.
- `modal_wall_s` is the call as seen from your machine; `in_function_s` is inside the container.
  Their difference, `startup_and_transfer_s`, is container start-up and transfer, reported on its own and
  never part of a speed-up. Data load and adjacency are under `preparation_s`.
- The NumPy reference runs in the same container as the GPU row. The EPYC rungs in
  `bench/results/cpu-ladder.json` use the same design on another host.
- Fail-closed: any parity, bit-identity, seed-coverage or distinct-container check that is false makes the
  run `FAILED`; its file then holds no timing or ratio, and the command exits non-zero.
- Fleet scaling uses the measured compute span: first start to last end across containers, from host wall
  clocks (NTP-synchronised; skew not measured). `studies / max(shard compute)` assumes perfect overlap and is
  labelled `ESTIMATE`. Only git-tracked files are uploaded to Modal.
