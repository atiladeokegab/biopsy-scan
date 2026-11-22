"""Build demo/plan/data.json for the study-planning calculator from the committed power-atlas production runs.

    uv run python -I demo/plan/build_data.py

Every number is copied from atlas/results/production/<cohort>/grid.json (SIMULATED: model-based Monte Carlo,
synthetic outcomes on the real cell tables) or computed by bench/vs_schoenfeld.schoenfeld_power (ESTIMATE: textbook
formula). The page interpolates only in patients, only between two simulated points of one design; nothing else.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.vs_schoenfeld import schoenfeld_power  # noqa: E402

COHORTS = ("charville", "upmc")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def cohort(name: str) -> dict:
    base = f"atlas/results/production/{name}"
    grid = json.loads((ROOT / base / "grid.json").read_text())
    meta = json.loads((ROOT / base / "metadata.json").read_text())
    audit_path = f"audit/results/{name}.json"
    m = int(json.loads((ROOT / audit_path).read_text())["n_features_tested"])
    execution = json.loads((ROOT / "atlas/results/production/modal-execution.json").read_text())
    run = next(r for r in execution["runs"] if r.get("kind") == "finished" and r.get("cohort") == name)
    events, n, hr = int(meta["source_events"]), int(meta["source_n"]), float(meta["hr_per_source_sd"])
    alpha = 0.05  # the family-wise alpha in every row of summary.parquet (column alpha)
    points = []
    for p in grid["points"]:
        assert p["power_lo"] <= p["power"] <= p["power_hi"], p["design_id"]
        points.append({
            "patients": p["patients"], "regions": p["regions"], "fov_um": p["fov_um"], "miscall": p["miscall"],
            "power": p["power"], "lo": p["power_lo"], "hi": p["power_hi"], "n": grid["studies_per_point"],
            "identification": p["identification_rate"], "fwer": p["fwer"], "design_id": p["design_id"],
        })
    return {
        "cohort": grid["cohort"]["name"], "source_patients": n, "source_events": events, "features_m": m,
        "hr_per_sd": hr, "alpha": alpha, "axes": grid["axes"], "studies_per_point": grid["studies_per_point"],
        "source": f"{base}/grid.json", "metadata": f"{base}/metadata.json", "features_source": audit_path,
        "source_commit": git("log", "-1", "--format=%H", "--", base),
        "atlas_code_sha": run["source_sha"],
        "hardware": f"Modal CPU container, {run['cpu']} cores, {run['memory_mib'] // 1024} GiB (modal-execution.json)",
        "points": points,
        # Formula power at every whole patient count inside the simulated range: formula[i] is at patients lo + i.
        "formula": [schoenfeld_power(k, events / n, hr, alpha, m)
                    for k in range(min(grid["axes"]["patients"]), max(grid["axes"]["patients"]) + 1)],
    }


def main() -> None:
    translate = json.loads((ROOT / "results/upmc-translate.json").read_text())
    data = {
        "schema": "preflight-plan/1",
        "labels": {
            "sim": "SIMULATED: Monte Carlo on the real cell tables, synthetic outcomes",
            "interp": "ESTIMATE: straight line between two simulated patient counts",
            "formula": "ESTIMATE: textbook formula (Schoenfeld 1983), assumes perfect measurement of the whole tissue",
        },
        "cohorts": {c: cohort(c) for c in COHORTS},
        "stains": {"status": translate["step1_panels"]["status"], "reason": translate["step1_panels"]["reason"],
                   "source": "results/upmc-translate.json", "code": "translate/panels.py"},
    }
    out = ROOT / "demo/plan/data.json"
    out.write_text(json.dumps(data, indent=1, allow_nan=False) + "\n")
    print(f"wrote {out.relative_to(ROOT)}: " + ", ".join(f"{c} {len(v['points'])} points" for c, v in data["cohorts"].items()))


if __name__ == "__main__":
    main()
