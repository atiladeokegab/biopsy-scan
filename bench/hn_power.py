"""Power on head and neck tissue: the #83 power-at-scale design on UPMC discovery tissue only.

    DATA_DIR=<dir with upmc_labels.csv, upmc_raw_data.zip> uv run python -I -m bench.hn_power OUT_DIR

The source is the 34 UPMC discovery patients (verified case identity) and their cell positions and types,
resampled with replacement to 50, 100, 200, 400 and 600 patients. Outcomes are SYNTHETIC: planted Weibull
outcomes at HR 2 per source SD on the atlas's own plant, the #83 event fraction, administrative censoring at
one common horizon. Same imaging design and budget as #83 (2 regions, 400 um field, 10% mis-calls; 200
calibration, 100 null, 300 planted studies per point; seed 20261121; 64-draw acquisition bank).

No real outcome is read: the labels file is read for two columns only (region_id, sample_label_visualizer),
and no locked patient's region, cells or files are touched: the patient list is checked disjoint from
core/upmc_locked.json, and cells come from core.cohorts.load_cells(split="discovery"), which never opens a
locked region's files. Writes bench/results/hn-power-upmc-discovery.json, labelled ESTIMATE.
"""
from __future__ import annotations

import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "core" / "upmc_locked.json"
RESULT = ROOT / "bench" / "results" / "hn-power-upmc-discovery.json"
PATIENTS = (50, 100, 200, 400, 600)
LABEL_COLUMNS = ["region_id", "sample_label_visualizer"]  # identity only; every outcome column stays unread
OUTCOME_COLUMNS = {"status", "primary_outcome", "recurred", "hpvstatus", "survival_status", "survival_day"}
# The #83 design's planted event fraction (Charville 38/159, bench/power_at_scale.FROZEN_MODEL): the same
# target here, so the design matches #83 and nothing about UPMC outcomes is used.
TARGET_EVENT_FRACTION = 0.2389937106918239
DESIGN = ["--cohorts", "upmc", "--regions", "2", "--fov-um", "400", "--eps", "0.1", "--hr", "2.0",
          "--seed", "20261121", "--calibration", "200", "--null", "100", "--alternative", "300", "--nested", "0",
          "--permutations", "999", "--batch", "8"]
LABEL = "ESTIMATE"
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
    lab = pd.read_csv(Path(data_dir) / "upmc_labels.csv", usecols=LABEL_COLUMNS, dtype=str, keep_default_na=False)
    assert set(lab.columns) == set(LABEL_COLUMNS) and not OUTCOME_COLUMNS & set(lab.columns), "outcome column read"
    case = lab.sample_label_visualizer.where(lab.sample_label_visualizer != "NA", "").str.split(" ").str[0]
    out = pd.DataFrame({"region_id": lab.region_id, "case": case})
    out = out[(out.case != "") & ~out.region_id.isin(set(lock["locked_regions"]))].reset_index(drop=True)
    out["group"] = (out.region_id.str.split("_").str[1].str[1:].astype(int) >= 4).astype(int)  # core.cohorts._group
    check_disjoint(out, lock)
    if (out.groupby("case").group.nunique() > 1).any():
        raise ValueError("a discovery patient spans both coverslip groups")
    return out


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
    rid = np.repeat(np.arange(len(reg), dtype=np.int32), np.diff(c["region_ptr"]))
    n = len(names)
    return {"cell_type": c["cell_type"].astype(np.int8), "x": c["xy_um"][:, 0].astype(np.float32),
            "y": c["xy_um"][:, 1].astype(np.float32), "region_id": rid,
            "patient_of_region": reg.case.map(pidx).to_numpy(np.int32),
            # placeholders: everyone followed to one horizon, none failed; fit_model gets the target explicitly
            "time": np.ones(n), "event": np.zeros(n, bool),
            "strata": reg.groupby("case").group.first().reindex(names).to_numpy(np.int32),
            "type_names": c["type_names"], "region_names": list(reg.region_id), "patient_names": names}


def run(out: Path, cache: Path, data_dir: Path, patients=PATIENTS, extra: list[str] = ()) -> None:
    """The atlas CLI on the outcome-free discovery source, with the #83 event-fraction target."""
    import atlas.__main__ as cli
    import atlas.runner as runner
    from atlas.simulation import fit_model

    cells = source(data_dir)
    cli.load = lambda cohort: cells if cohort == "upmc" else (_ for _ in ()).throw(ValueError(cohort))
    runner.fit_model = partial(fit_model, target=TARGET_EVENT_FRACTION)
    cli.main([*DESIGN, "--patients", *map(str, patients), "--out", str(out), "--cache", str(cache), *extra])


def summarize(out: Path) -> dict:
    from bench.power_at_scale import answer, clopper_pearson

    rows = pd.read_parquet(out / "upmc" / "summary.parquet").sort_values("patients")
    meta = json.loads((out / "upmc" / "metadata.json").read_text())
    points = []
    for r in rows.to_dict("records"):
        k, n = int(r["power_count"]), int(r["power_n"])
        lo, hi = clopper_pearson(k, n)
        points.append({"patients": int(r["patients"]), "power": k / n, "power_lo": lo, "power_hi": hi,
                       "planted_studies": n, "rejections": k, "fwer": float(r["fwer"]),
                       "mean_events": float(r["alternative_mean_events"]), "threshold": float(r["threshold"])})
    return {"label": LABEL, "outcomes": "SYNTHETIC", "source": "34 discovery patients resampled",
            "cohort": "UPMC head and neck, discovery split only", "caveats": CAVEATS, "points": points,
            "answer": answer(points), "metadata": {k: meta.get(k) for k in
                                                   ("source_n", "source_regions", "source_cells", "grammar_hash",
                                                    "budget", "hr_per_source_sd", "seed", "backend", "bank_draws")}}


if __name__ == "__main__":
    out = Path(sys.argv[1])
    run(out, out / "cache", Path(__import__("os").environ["DATA_DIR"]), extra=sys.argv[2:])
    print(json.dumps(summarize(out)["answer"]["statement"]))
