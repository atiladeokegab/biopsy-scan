"""Cohort loaders: labels, patient identity, coverslip strata, survival endpoints, cells.

Data: Zenodo record 13179600 (CC-BY-4.0), fetched by scripts/fetch_data.sh into data/.
Everything read here is untrusted data: only parsed as CSV, never executed.
"""
from __future__ import annotations

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


def _coverslip(region_id: pd.Series) -> pd.Series:
    return region_id.str.split("_").str[1]  # UPMC_c001_v001_r001_reg001 -> c001


def load_regions(cohort: str, data_dir: Path = DATA_DIR, identity: str = "tuple") -> pd.DataFrame:
    """One row per region: region_id, coverslip, case, patient_id, time_days, event.

    identity (UPMC only): "tuple" = one patient per clinical tuple (79 patients);
    "tuple+case" = also split a tuple that holds two different case numbers (sensitivity).
    """
    if cohort not in COHORTS:
        raise ValueError(f"unknown cohort {cohort!r}")
    lab = pd.read_csv(Path(data_dir) / f"{cohort}_labels.csv", dtype=str, keep_default_na=False)
    out = pd.DataFrame({"region_id": lab.region_id, "coverslip": _coverslip(lab.region_id)})
    if cohort == "upmc":
        if identity not in ("tuple", "tuple+case"):
            raise ValueError(f"unknown identity {identity!r}")
        out["case"] = lab.sample_label_visualizer.where(lab.sample_label_visualizer != "NA", "").str.split(" ").str[0]
        key = lab[UPMC_TUPLE].agg("|".join, axis=1)
        if identity == "tuple+case":
            n_cases = out.case.where(out.case != "").groupby(key).transform("nunique")
            key = key + np.where(n_cases > 1, "|" + out.case, "")
        out["patient_id"] = key
        out["time_days"] = lab.survival_day.astype(float)
        out["event"] = lab.survival_status.astype(float)  # any-cause death
    else:
        out["case"] = lab.sample_label_visualizer.str.split("_").str[0]
        out["patient_id"] = out.case
        surg = pd.to_datetime(lab.surgery_date, format="%m/%d/%Y", errors="coerce")
        last = pd.to_datetime(lab.last_contact_date, format="%m/%d/%Y", errors="coerce")
        out["time_days"] = (last - surg).dt.days.astype(float)  # NaN when a date is missing
        out["event"] = (lab.alive_or_deceased == "Dead").astype(float).where(out.time_days.notna())
    return out


def load_patients(regions: pd.DataFrame) -> pd.DataFrame:
    """One row per patient, in sorted patient_id order; patient index = row number.

    stratum: the patient's modal coverslip (most regions, ties to the lowest), because UPMC
    patients span coverslips; n_coverslips says how many they span.
    """
    g = regions.groupby("patient_id", sort=True)
    ends = g[["time_days", "event"]].agg(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
    if (g[["time_days", "event"]].nunique() > 1).any().any():
        raise ValueError("a patient has conflicting endpoints across regions")
    modal = regions.groupby(["patient_id", "coverslip"]).size().reset_index(name="n")
    modal = modal.sort_values(["patient_id", "n", "coverslip"], ascending=[True, False, True]).drop_duplicates("patient_id")
    pts = ends.join(modal.set_index("patient_id").coverslip.rename("stratum"))
    pts["n_coverslips"] = g.coverslip.nunique()
    pts["n_regions"] = g.size()
    return pts.reset_index()


def load_cells(cohort: str, region_ids: list[str] | None = None, data_dir: Path = DATA_DIR) -> dict:
    """Cells of the given regions (default: all, in label order), concatenated region by region.

    Returns xy_um float64 (N,2), cell_type int16 (N,), region_ptr int64 (R+1,), region_ids, type_names.
    type_names is the sorted list of every type in the cohort, so codes are stable across calls.
    """
    data_dir = Path(data_dir)
    all_regions = list(load_regions(cohort, data_dir).region_id)
    region_ids = all_regions if region_ids is None else list(region_ids)
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
    return load_cells(cohort, region_ids, data_dir)
