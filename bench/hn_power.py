"""Power on head and neck tissue: the #83 power-at-scale design on UPMC discovery tissue only.

    DATA_DIR=<dir with upmc_labels.csv, upmc_raw_data.zip> uv run python -I -m bench.hn_power OUT_DIR

The source is the 34 UPMC discovery patients (verified case identity) and their cell positions and types,
resampled with replacement to 50, 100, 200, 400 and 600 patients. Outcomes are SYNTHETIC: planted Weibull
outcomes at HR 2 per source SD on the atlas's own plant, the #83 event fraction, administrative censoring at
one common horizon. Same imaging design and budget as #83 (2 regions, 400 um field, 10% mis-calls; 200
calibration, 100 null, 300 planted studies per point; seed 20261121; 64-draw acquisition bank). The Modal
launcher gpu/hn_power.py runs 200 planted studies per point to stay inside its $5 cap.

No real outcome is read: the labels file is read for two columns only (region_id, sample_label_visualizer),
and no locked patient's region, cells or files are touched: the patient list is checked disjoint from
core/upmc_locked.json, and cells come from core.cohorts.load_cells(split="discovery"), which never opens a
locked region's files. Writes bench/results/hn-power-upmc-discovery.json, labelled ESTIMATE.
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "core" / "upmc_locked.json"
RESULT = ROOT / "bench" / "results" / "hn-power-upmc-discovery.json"
PATIENTS = (50, 100, 200, 400, 600)
PLANNED = (50, 100, 200, 300, 400, 600)  # the published curve: the five-point run plus Zeus's P=300 point
# one value across every point of a curve, in each row and its container's metadata
PROVENANCE = ("source_hash", "grammar_hash", "seed", "regions_per_patient", "fov_um", "eps", "alpha",
              "hr_per_source_sd", "shape", "target_event_fraction", "calibration_n", "plant_feature", "bank_hash",
              "measurement_model")
META_KEEP = ("source_n", "source_regions", "source_cells", "source_hash", "grammar_hash", "budget",
             "hr_per_source_sd", "seed", "backend", "bank_draws", "measurement_model", "mc_interval_scope")
LABEL_COLUMNS = ["region_id", "sample_label_visualizer"]  # identity only; every outcome column stays unread
OUTCOME_COLUMNS = {"status", "primary_outcome", "recurred", "hpvstatus", "survival_status", "survival_day"}
# The #83 design's planted event fraction (Charville 38/159, bench/power_at_scale.FROZEN_MODEL): the same
# target here, so the design matches #83 and nothing about UPMC outcomes is used.
TARGET_EVENT_FRACTION = 0.2389937106918239
DESIGN = ["--cohorts", "upmc", "--regions", "2", "--fov-um", "400", "--eps", "0.1", "--hr", "2.0",
          "--seed", "20261121", "--calibration", "200", "--null", "100", "--alternative", "300", "--nested", "0",
          "--permutations", "999", "--batch", "8"]
LABEL = "ESTIMATE; simulated cohorts resampled from 34 UPMC discovery patients' tissue; outcomes SYNTHETIC"
CAVEATS = {
    "outcomes": "SYNTHETIC: planted Weibull outcomes, HR 2 per source SD; no real UPMC outcome was read",
    "source": "34 discovery patients resampled with replacement up to 600: every simulated cohort reuses the "
              "same 34 patients' tissue, so the simulation sees far less tissue variety than 600 real patients "
              "would bring; the answer is conditional on these 34",
    "censoring": "administrative censoring at one common horizon, tuned to the #83 event fraction (0.239); "
                 "#83 used Charville's reverse Kaplan-Meier censoring, which needs real follow-up",
    "plant": "the atlas's plant rule (first finite feature with within-stratum variation), on UPMC's own grammar",
    "scope": "a design estimate for planning; no clinical claim",
}


def discovery_patients(data_dir: Path) -> pd.DataFrame:
    """Discovery regions with a verified case number: region_id, case, group. Reads no outcome column."""
    lock = json.loads(LOCK.read_text())
    out = case_of_region(data_dir)
    out = out[(out.case != "") & ~out.region_id.isin(set(lock["locked_regions"]))].reset_index(drop=True)
    out["group"] = (out.region_id.str.split("_").str[1].str[1:].astype(int) >= 4).astype(int)  # core.cohorts._group
    check_disjoint(out, lock)
    if (out.groupby("case").group.nunique() > 1).any():
        raise ValueError("a discovery patient spans both coverslip groups")
    return out


def case_of_region(data_dir: Path) -> pd.DataFrame:
    """region_id -> case number ("" for NA), from the two identity columns only."""
    lab = pd.read_csv(Path(data_dir) / "upmc_labels.csv", usecols=LABEL_COLUMNS, dtype=str, keep_default_na=False)
    assert set(lab.columns) == set(LABEL_COLUMNS) and not OUTCOME_COLUMNS & set(lab.columns), "outcome column read"
    case = lab.sample_label_visualizer.where(lab.sample_label_visualizer != "NA", "").str.split(" ").str[0]
    return pd.DataFrame({"region_id": lab.region_id, "case": case})


def check_cache(data_dir: Path) -> None:
    """The discovery cell cache on disk holds no locked region and no locked patient's cells."""
    lock = json.loads(LOCK.read_text())
    with np.load(Path(data_dir) / "upmc_cells.discovery.npz", allow_pickle=False) as z:
        ids = pd.DataFrame({"region_id": z["region_ids"].astype(str)})
    ids["case"] = ids.region_id.map(case_of_region(data_dir).set_index("region_id").case).fillna("")
    check_disjoint(ids, lock)


def check_disjoint(regions: pd.DataFrame, lock: dict) -> None:
    """No locked region and no locked patient on the discovery side."""
    bad_r = set(regions.region_id) & set(lock["locked_regions"])
    bad_p = set(regions.case) & {str(p) for p in lock["locked_patients"]}
    if bad_r or bad_p:
        raise PermissionError(f"locked set on the discovery side: {len(bad_r)} regions, {len(bad_p)} patients")


def source(data_dir: Path) -> dict:
    """The atlas's cells dict from discovery tissue, with placeholder outcomes that no model step reads."""
    from core.cohorts import load_cells

    reg = discovery_patients(data_dir)
    names = sorted(reg.case.unique())
    pidx = {p: i for i, p in enumerate(names)}
    c = load_cells("upmc", list(reg.region_id), Path(data_dir), "discovery")
    check_cache(data_dir)
    assert list(c["region_ids"]) == list(reg.region_id), "cells beyond the 34 patients' regions"
    rid = np.repeat(np.arange(len(reg), dtype=np.int32), np.diff(c["region_ptr"]))
    n = len(names)
    return {"cell_type": c["cell_type"].astype(np.int8), "x": c["xy_um"][:, 0].astype(np.float32),
            "y": c["xy_um"][:, 1].astype(np.float32), "region_id": rid,
            "patient_of_region": reg.case.map(pidx).to_numpy(np.int32),
            # placeholders: everyone followed to one horizon, none failed; fit_model gets the target explicitly
            "time": np.ones(n), "event": np.zeros(n, bool),
            "strata": reg.groupby("case").group.first().reindex(names).to_numpy(np.int32),
            "type_names": c["type_names"], "region_names": list(reg.region_id), "patient_names": names}


def run(out: Path, cache: Path, cells: dict, patients=PATIENTS, extra: list[str] = ()) -> None:
    """The atlas CLI on the outcome-free discovery source (from source()), with the #83 event-fraction target."""
    import atlas.__main__ as cli
    import atlas.runner as runner
    from atlas.simulation import fit_model

    cli.load = lambda cohort: cells if cohort == "upmc" else (_ for _ in ()).throw(ValueError(cohort))
    runner.fit_model = partial(fit_model, target=TARGET_EVENT_FRACTION)
    cli.main([*DESIGN, "--patients", *map(str, patients), "--out", str(out), "--cache", str(cache), *extra])


def read_run(out: Path) -> tuple[list[dict], dict]:
    """The atlas's summary rows and metadata for one run directory."""
    rows = pd.read_parquet(out / "upmc" / "summary.parquet").to_dict("records")
    return rows, json.loads((out / "upmc" / "metadata.json").read_text())


def stage(repo: Path, sha: str, dest: Path) -> Path:
    """sha's tracked files in dest, without the locked manifest: the remote image never holds a locked ID."""
    archive = subprocess.run(["git", "-C", str(repo), "archive", sha], capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(dest, filter="data")
    (Path(dest) / "core" / "upmc_locked.json").unlink(missing_ok=True)
    leaked = [f for f in Path(dest).rglob("*") if "upmc_locked" in f.name]
    if leaked:
        raise PermissionError(f"locked manifest staged for upload: {leaked[0]}")
    return Path(dest)


def producer(repo: Path) -> str:
    """HEAD, refusing tracked changes, so the payload is built by exactly the commit the receipt records."""
    git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True).stdout  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise RuntimeError("tracked changes in the working tree: commit them before building the payload")
    return git("rev-parse", "HEAD").strip()


def collect(receipts: list[dict], planned=PLANNED) -> tuple[list[dict], dict, list[int]]:
    """Validated rows, common metadata and the planned points with no result, from launcher receipts.

    Each finished point has exactly one row for its own patient count; a point seen twice must be identical; every
    row and metadata share one source, grammar, design, seed, budget and bank; unplanned points are refused."""
    done = {}
    for d in receipts:
        for p in d["points"]:
            n = p["patients"]
            if n not in planned:
                raise ValueError(f"P={n} is not a planned point")
            if "rows" not in p:
                continue
            if len(p["rows"]) != 1 or int(p["rows"][0]["patients"]) != n:
                raise ValueError(f"P={n}: expected one row for P={n}")
            if n in done and done[n]["rows"] != p["rows"]:
                raise ValueError(f"P={n}: conflicting duplicate results")
            done[n] = p
    if not done:
        raise ValueError("no finished point")
    for k in PROVENANCE:
        seen = {json.dumps(p["rows"][0].get(k)) for p in done.values()}
        seen |= {json.dumps(p["metadata"][k]) for p in done.values() if k in p["metadata"]}
        if len(seen) != 1:
            raise ValueError(f"points disagree on {k}: {sorted(seen)}")
    if len({json.dumps(p["metadata"]["budget"], sort_keys=True) for p in done.values()}) != 1:
        raise ValueError("points disagree on budget")
    meta = next(iter(done.values()))["metadata"]
    return [done[n]["rows"][0] for n in sorted(done)], meta, [n for n in planned if n not in done]


def validate(summary: dict) -> dict:
    """A published curve re-derived from its own points: counts, intervals, monotone power and the 80% answer."""
    from bench.power_at_scale import answer, clopper_pearson

    pts = summary["points"]
    if [p["patients"] for p in pts] != sorted({p["patients"] for p in pts}):
        raise ValueError("points not strictly increasing in patients")
    for p in pts:
        k, n = p["rejections"], p["planted_studies"]
        if not 0 <= k <= n or abs(p["power"] - k / n) > 1e-12 or \
                max(abs(a - b) for a, b in zip((p["power_lo"], p["power_hi"]), clopper_pearson(k, n))) > 1e-12:
            raise ValueError(f"P={p['patients']}: power or interval does not match {k}/{n}")
    if any(b["rejections"] / b["planted_studies"] < a["rejections"] / a["planted_studies"] for a, b in zip(pts, pts[1:])):
        raise ValueError("power is not monotone in patients: the interpolated crossing is ambiguous")
    if summary["answer"] != answer(pts, summary["answer"]["missing_patients"]):
        raise ValueError("the 80% answer does not follow from the points")
    return summary


def summarize(rows: list[dict], meta: dict, missing: list[int] = ()) -> dict:
    from bench.power_at_scale import answer, clopper_pearson

    points = []
    for r in sorted(rows, key=lambda r: r["patients"]):
        k, n = int(r["power_count"]), int(r["power_n"])
        lo, hi = clopper_pearson(k, n)
        points.append({"patients": int(r["patients"]), "power": k / n, "power_lo": lo, "power_hi": hi,
                       "planted_studies": n, "rejections": k, "fwer": float(r["fwer"]),
                       "fwer_count": int(r["fwer_count"]), "fwer_n": int(r["fwer_n"]),
                       "fwer_lo": float(r["fwer_lo"]), "fwer_hi": float(r["fwer_hi"]),
                       "mean_events": float(r["alternative_mean_events"]), "threshold": float(r["threshold"])})
    return {"label": LABEL, "outcomes": "SYNTHETIC", "source": "simulated cohorts of 50-600 resampled with replacement from 34 UPMC discovery patients' tissue",
            "interval": "per point: Clopper-Pearson 95% on rejections / planted studies (200 per point on Modal)",
            "cohort": "UPMC head and neck, discovery split only", "caveats": CAVEATS, "points": points,
            "answer": answer(points, list(missing)), "metadata": {k: meta.get(k) for k in META_KEEP}}


def from_raw(*raws: Path, planned=PLANNED) -> dict:
    """The validated summary re-derived from gpu/hn_power.py receipts (their rows are the measured record)."""
    ds = [json.loads(Path(r).read_text()) for r in raws]
    out = validate(summarize(*collect(ds, planned)))
    return {**out, "raw": [f"bench/results/{Path(r).name}" for r in raws], "commit": [d["commit"] for d in ds]}


if __name__ == "__main__" and sys.argv[1] == "--from-raw":
    print(json.dumps(from_raw(*map(Path, sys.argv[2:])), indent=1))
elif __name__ == "__main__":
    out = Path(sys.argv[1])
    run(out, out / "cache", source(Path(__import__("os").environ["DATA_DIR"])), extra=sys.argv[2:])
    print(json.dumps(summarize(*read_run(out))["answer"]["statement"]))
