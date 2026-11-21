"""Cohort loaders: labels, patient identity, coverslip strata, survival endpoints, cells.

Data: Zenodo record 13179600 (CC-BY-4.0), fetched by scripts/fetch_data.sh into data/.
Everything read here is untrusted data: only parsed as CSV, never executed.
"""
from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
COHORTS = ("upmc", "charville")

# shortcut: CODEX 0.377 um per pixel (Wu et al. 2022 acquisition); confirm against the paper's methods if a scale check disagrees.
PIXEL_UM = 0.377

UPMC_TUPLE = ["status", "primary_outcome", "recurred", "hpvstatus", "survival_status", "survival_day"]

# The locked held-out UPMC patients (core/heldout.py). None only for invented test fixtures.
LOCK_FILE: Path | None = ROOT / "core" / "upmc_locked.json"
# "all" reads both sides: only the power/design-grid path (scripts/export_grid.py) may pass it.
SPLITS = ("discovery", "locked", "all")


class LockedPatientError(PermissionError):
    """A discovery-side loader was asked for a locked held-out region."""


def manifest_hash(m: dict) -> str:
    """sha256 of the manifest's canonical JSON, without its own sha256 field."""
    body = json.dumps({k: v for k, v in m.items() if k != "sha256"}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()


def locked_regions(cohort: str = "upmc", data_dir: Path = DATA_DIR) -> set[str]:
    """Region IDs of the locked held-out patients. Raises if the manifest is missing or edited by hand,
    or if data_dir's labels file is not the one the split was locked on."""
    if cohort != "upmc" or LOCK_FILE is None:
        return set()
    m = json.loads(Path(LOCK_FILE).read_text())
    if m.get("sha256") != manifest_hash(m):
        raise ValueError(f"{LOCK_FILE}: sha256 does not match its content")
    labels = Path(data_dir) / "upmc_labels.csv"
    if hashlib.sha256(labels.read_bytes()).hexdigest() != m["labels_sha256"]:
        raise ValueError(f"{labels}: sha256 differs from the labels the held-out split was locked on")
    return set(m["locked_regions"])


def _check_split(cohort: str, split: str) -> None:
    if split not in SPLITS or (cohort != "upmc" and split == "locked"):
        raise ValueError(f"unknown split {split!r} for {cohort}")


def upmc_patient_ids(lab: pd.DataFrame) -> tuple[pd.Series, list[dict]]:
    """The UPMC patient-grouping rule: one patient_id per region, "" when unmapped, plus the conflicts.

    A case number (first word of sample_label_visualizer) is the patient. An NA region is the patient
    "T:<clinical tuple>" when no case-numbered region shares its tuple. Unmapped and reported, never
    guessed: a case whose regions disagree on the tuple; an NA region whose tuple also holds a case
    number; an NA tuple spanning both coverslip groups (verified patients nest inside one group).
    """
    case = lab.sample_label_visualizer.where(lab.sample_label_visualizer != "NA", "").str.split(" ").str[0]
    tup = lab[UPMC_TUPLE].agg("|".join, axis=1)
    grp = _group("upmc", _coverslip(lab.region_id))
    has = case != ""
    pid = case.where(has, "T:" + tup)
    cases_in_tuple = case.where(has).groupby(tup).transform("nunique")
    rules = {
        "case with conflicting outcomes": has & (tup.groupby(case).transform("nunique") > 1),
        "NA region in a tuple holding a case number": ~has & (cases_in_tuple > 0),
        "NA tuple spanning both coverslip groups": ~has & (cases_in_tuple == 0) & (grp.groupby(tup).transform("nunique") > 1),
    }
    conflicts = []
    for kind, bad in rules.items():
        for key in sorted(set(pid[bad])):
            m = bad & (pid == key)
            conflicts.append({"kind": kind, "group": key, "regions": sorted(lab.region_id[m]),
                              "coverslips": sorted(set(_coverslip(lab.region_id[m])))})
        pid = pid.where(~bad, "")
    return pid, conflicts


def _coverslip(region_id: pd.Series) -> pd.Series:
    return region_id.str.split("_").str[1]  # UPMC_c001_v001_r001_reg001 -> c001


def _group(cohort: str, coverslip: pd.Series) -> pd.Series:
    """Coverslip group used as the stratum: UPMC 0 = c001-c003, 1 = c004-c007; Charville c001-c004 -> 0-3."""
    n = coverslip.str[1:].astype(int)
    return (n >= 4).astype(int) if cohort == "upmc" else n - 1


def load_regions(cohort: str, data_dir: Path = DATA_DIR, identity: str = "case", split: str = "discovery") -> pd.DataFrame:
    """One row per region: region_id, coverslip, group, case, patient_id, time_days, event.

    identity "case" (default): verified identity = the case number in the sample label; UPMC regions
    labelled NA are dropped. Only this identity is used for patient-level inference and the
    regions-per-patient axis; Charville is the patient-level cohort.
    UPMC only, provisional and descriptive only: "tuple" = one patient per clinical tuple (79);
    "tuple+case" = also split the tuples holding two case numbers (81). One all-NA tuple spans
    both coverslip groups. "patient" = upmc_patient_ids: the case number, else the clinical tuple of
    NA regions; unmapped regions are dropped.
    split (UPMC): "discovery" (default) never returns a locked held-out region; "locked" returns only those;
    "all" returns both, for the power/design grid only.
    """
    if cohort not in COHORTS:
        raise ValueError(f"unknown cohort {cohort!r}")
    _check_split(cohort, split)
    if identity not in ("case", "tuple", "tuple+case", "patient") or (cohort != "upmc" and identity != "case"):
        raise ValueError(f"unknown identity {identity!r} for {cohort}")
    lab = pd.read_csv(Path(data_dir) / f"{cohort}_labels.csv", dtype=str, keep_default_na=False)
    out = pd.DataFrame({"region_id": lab.region_id, "coverslip": _coverslip(lab.region_id)})
    out["group"] = _group(cohort, out.coverslip)
    if cohort == "upmc":
        out["case"] = lab.sample_label_visualizer.where(lab.sample_label_visualizer != "NA", "").str.split(" ").str[0]
        key = lab[UPMC_TUPLE].agg("|".join, axis=1)
        if identity == "tuple+case":
            n_cases = out.case.where(out.case != "").groupby(key).transform("nunique")
            key = key + np.where(n_cases > 1, "|" + out.case, "")
        out["patient_id"] = {"case": out.case, "patient": upmc_patient_ids(lab)[0]}.get(identity, key)
        out["time_days"] = lab.survival_day.astype(float)
        out["event"] = lab.survival_status.astype(float)  # any-cause death
        if identity in ("case", "patient"):
            out = out[out.patient_id != ""]
        if split != "all":
            locked = out.region_id.isin(locked_regions(cohort, data_dir))
            out = out[locked if split == "locked" else ~locked].reset_index(drop=True)
    else:
        out["case"] = lab.sample_label_visualizer.str.split("_").str[0]
        out["patient_id"] = out.case
        surg = pd.to_datetime(lab.surgery_date, format="%m/%d/%Y", errors="coerce")
        last = pd.to_datetime(lab.last_contact_date, format="%m/%d/%Y", errors="coerce")
        out["time_days"] = (last - surg).dt.days.astype(float)  # NaN when a date is missing
        out["event"] = (lab.alive_or_deceased == "Dead").astype(float).where(out.time_days.notna())
    return out


def load_patients(regions: pd.DataFrame) -> pd.DataFrame:
    """One row per patient, in sorted patient_id order.

    stratum: the patient's coverslip group, or -1 when its regions span groups (ambiguous);
    usable: has an endpoint and an unambiguous stratum.
    """
    g = regions.groupby("patient_id", sort=True)
    if (g[["time_days", "event"]].nunique() > 1).any().any():
        raise ValueError("a patient has conflicting endpoints across regions")
    pts = g[["time_days", "event"]].first()  # first non-null per column
    pts["stratum"] = g.group.first().where(g.group.nunique() == 1, -1).astype(int)
    pts["n_coverslips"] = g.coverslip.nunique()
    pts["n_regions"] = g.size()
    pts["usable"] = pts.time_days.notna() & pts.event.notna() & (pts.stratum >= 0)
    return pts.reset_index()


def load(cohort: str, data_dir: Path = DATA_DIR, split: str = "discovery") -> dict:
    """The patient-level cohort (docs/CONTRACT.md): usable patients with a verified identity only."""
    reg = load_regions(cohort, data_dir, split=split)
    pts = load_patients(reg)
    pts = pts[pts.usable].reset_index(drop=True)
    pidx = {p: i for i, p in enumerate(pts.patient_id)}
    reg = reg[reg.patient_id.isin(pidx)].reset_index(drop=True)
    c = load_cells(cohort, list(reg.region_id), data_dir, split)
    rid = np.repeat(np.arange(len(reg), dtype=np.int32), np.diff(c["region_ptr"]))
    return {"cell_type": c["cell_type"].astype(np.int8), "x": c["xy_um"][:, 0].astype(np.float32),
            "y": c["xy_um"][:, 1].astype(np.float32), "region_id": rid,
            "patient_of_region": reg.patient_id.map(pidx).to_numpy(np.int32),
            "time": pts.time_days.to_numpy(np.float64), "event": pts.event.to_numpy(bool),
            "strata": pts.stratum.to_numpy(np.int32), "type_names": c["type_names"],
            "region_names": list(reg.region_id), "patient_names": list(pts.patient_id)}


def load_cells(cohort: str, region_ids: list[str] | None = None, data_dir: Path = DATA_DIR,
               split: str = "discovery") -> dict:
    """Cells of the given regions (default: all of the split, in label order), concatenated region by region.

    Raises LockedPatientError when asked for a region outside the split, e.g. a locked held-out region
    from the discovery side.

    Returns xy_um float64 (N,2), cell_type int16 (N,), region_ptr int64 (R+1,), region_ids, type_names.
    type_names is the sorted list of every type in the cohort, so codes are stable across calls.
    """
    data_dir = Path(data_dir)
    all_regions = list(pd.read_csv(data_dir / f"{cohort}_labels.csv", usecols=["region_id"], dtype=str).region_id)
    _check_split(cohort, split)
    locked = locked_regions(cohort, data_dir)
    on_side = lambda r: split == "all" or (r in locked) == (split == "locked")  # noqa: E731
    region_ids = [r for r in all_regions if on_side(r)] if region_ids is None else list(region_ids)
    wrong = [r for r in region_ids if not on_side(r)]
    if wrong:
        raise LockedPatientError(f"{len(wrong)} region(s) outside the {split} split, e.g. {wrong[0]}")
    cache = data_dir / f"{cohort}_cells.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        names, ptr = list(z["region_ids"]), z["region_ptr"]
        pos = {r: i for i, r in enumerate(names)}
        idx = [pos[r] for r in region_ids]
        sl = [np.arange(ptr[i], ptr[i + 1]) for i in idx]
        take = np.concatenate(sl) if sl else np.zeros(0, int)
        return {"xy_um": z["xy_um"][take], "cell_type": z["cell_type"][take],
                "region_ptr": np.concatenate([[0], np.cumsum([len(s) for s in sl])]).astype(np.int64),
                "region_ids": region_ids, "type_names": list(z["type_names"])}
    xy, ty = [], []
    with zipfile.ZipFile(data_dir / f"{cohort}_raw_data.zip") as z:
        for r in all_regions:
            d = pd.read_csv(z.open(f"raw_data/{r}.cell_data.csv"))
            t = pd.read_csv(z.open(f"raw_data/{r}.cell_types.csv"))
            m = d.merge(t, on="CELL_ID", how="inner", validate="one_to_one")
            if len(m) != len(d):
                raise ValueError(f"{r}: {len(d) - len(m)} cells without a type")
            xy.append(m[["X", "Y"]].to_numpy(np.float64) * PIXEL_UM)
            ty.append(m.CELL_TYPE.to_numpy(str))
    type_names = sorted(set(np.concatenate(ty)))
    code = {n: i for i, n in enumerate(type_names)}
    np.savez(cache, xy_um=np.concatenate(xy), type_names=np.array(type_names),
             cell_type=np.array([code[n] for n in np.concatenate(ty)], np.int16),
             region_ptr=np.concatenate([[0], np.cumsum([len(a) for a in xy])]).astype(np.int64),
             region_ids=np.array(all_regions))
    return load_cells(cohort, region_ids, data_dir, split)
