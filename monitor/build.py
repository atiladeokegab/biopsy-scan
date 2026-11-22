"""monitor/data.json: measured CPU and GPU utilisation for the usage monitor, replayed on the race's clock.

    python3 -I monitor/build.py          # write monitor/data.json
    python3 -I monitor/build.py --check  # exit 1 if a fresh build differs from the committed file

Current is the squidpy + lifelines arm of the usage-traces run. Ours is the newest
bench/results/gpu-fused-usage-h100-*.json when one exists, else the CPU-fused arm of the same usage-traces run.
Times are seconds from the start of each arm's compute window; nothing is smoothed or interpolated.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "monitor" / "data.json"
TRACES = "bench/results/usage-traces-20261122T132125Z.json"
RACE = "demo/app/data.json"
H100_GLOB = "gpu-fused-usage-h100-*.json"
PAD = 0.5  # seconds either side of the compute window, so the idle-to-busy edge shows


def read(rel: str) -> dict:
    return json.loads((REPO / rel).read_text())


def lane(a: dict, name: str, gpu: str, cores: float) -> dict:
    """One arm in usage-traces shape: cpu [[t, [per-core %]]] or [[t, cores busy]], gpu [[t, util %, MiB]]."""
    lo, hi = a["compute_window_s"]
    win = lambda t: lo - PAD <= t <= hi + PAD  # noqa: E731
    busy = lambda c: math.fsum(c) / 100 if isinstance(c, list) else c  # noqa: E731  gVisor spreads use over visible cores; fsum: same digits on every Python
    return {"name": name, "gpu": gpu, "studies": a["studies"], "cores": cores, "wall_s": round(hi - lo, 3),
            "cpu": [[round(t - lo, 2), round(busy(c), 2)] for t, c in a["cpu"] if win(t)],
            "gpu_util": [[round(g[0] - lo, 2), g[1]] for g in a["gpu"] if win(g[0])]}


def newest_h100(results: Path) -> str | None:
    files = sorted(results.glob(H100_GLOB))  # the UTC stamp in the name sorts by time
    return str(files[-1].relative_to(REPO)) if files else None


def h100_arm(r: dict) -> dict:
    """The timed gpu_fused launch of a gpu-fused-usage file (flat 100 ms samples tagged by arm), in usage-traces shape."""
    a, dt = r["arms"]["gpu_fused"], r["sample_period_s"]
    s = [x for x in r["samples"] if x["arm"] == "gpu_fused"]  # warm-up and setup samples are not the timed launch
    lo = s[0]["t_s"] - dt  # each sample closes its 100 ms interval
    return {"compute_window_s": [lo, lo + a["wall_s"]], "studies": a["studies_per_launch"], "host": r["host"],
            "cpu": [[x["t_s"], x["busy_cores"]] for x in s],
            "gpu": [[x["t_s"], x["gpu_util_pct"], x["gpu_mem_used_mib"]] for x in s]}


def build(results: Path = REPO / "bench" / "results") -> dict:
    tr, race = read(TRACES), read(RACE)
    assert tr["status"] == "MEASURED", TRACES
    gpu_name = lambda a: (a["host"].get("gpu") or "").split(",")[0].replace("NVIDIA ", "") or None  # noqa: E731
    cur = tr["arms"]["current"]
    current = lane(cur, "squidpy + lifelines", gpu_name(cur), cur["host"]["cores_reserved"])
    h100 = newest_h100(results)
    if h100:
        r = json.loads((REPO / h100).read_text())
        assert r["status"] == "MEASURED" and r["parity"]["all"] is True, h100
        a = h100_arm(r)
        ours = lane(a, "H100 fused kernel", gpu_name(a), None)
    else:
        a = tr["arms"]["fused"]
        ours = lane(a, "CPU fused kernel", gpu_name(a), a["host"]["cores_reserved"])
    return {"schema": "preflight-monitor/1", "sources": [TRACES, h100 or TRACES, RACE],
            "ours_source": h100 or TRACES, "sample_interval_s": tr["sampling"]["interval_s"],
            "race": {"seconds": race["race"]["seconds"], "race_s_per_real_s": race["race"]["race_s_per_real_s"]},
            "lanes": {"current": current, "ours": ours}}


def main(argv: list[str]) -> int:
    fresh = json.dumps(build(), separators=(",", ":")) + "\n"
    if "--check" in argv:
        ok = OUT.exists() and OUT.read_text() == fresh
        print(f"monitor/data.json: {'ok' if ok else 'STALE, run python3 -I monitor/build.py'}")
        return 0 if ok else 1
    OUT.write_text(fresh)
    print(f"wrote {OUT.relative_to(REPO)} ({len(fresh):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
