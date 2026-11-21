"""Source provenance, Parquet evidence and standalone scientific figures."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def source_metadata(cells):
    rid = np.asarray(cells["region_id"])
    por = np.asarray(cells["patient_of_region"])
    n = len(cells["time"])
    spans = []
    ptr = np.searchsorted(rid, np.arange(len(por) + 1))
    for r in range(len(por)):
        if ptr[r + 1] > ptr[r]:
            spans.append([np.ptp(cells[key][ptr[r]:ptr[r + 1]]) for key in ("x", "y")])
    if not spans:
        raise ValueError("source tissue has no nonempty region")
    spans = np.asarray(spans, float)
    digest = hashlib.sha256()
    for key in ("cell_type", "x", "y", "region_id", "patient_of_region", "time", "event", "strata"):
        a = np.ascontiguousarray(cells[key])
        digest.update(f"{key}:{a.dtype}:{a.shape}".encode())
        digest.update(a.view(np.uint8))
    digest.update(json.dumps(cells.get("type_names"), sort_keys=True).encode())
    return {"source_n": n, "source_events": int(np.sum(cells["event"])), "source_regions": len(por),
            "source_cells": len(rid), "source_hash": digest.hexdigest(),
            "region_um": float(spans.max()),
            "region_width_median_um": float(np.median(spans[:, 0])),
            "region_height_median_um": float(np.median(spans[:, 1])),
            "region_um_definition": "maximum x/y bounding-box span across source regions",
            "max_regions_per_patient": int(np.bincount(por, minlength=n).max())}


def save_tables(summaries, records, metadata, out_dir):
    """Atomically replace completed-point evidence; no raw tissue is written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("summary", summaries), ("studies", records)):
        if rows is None:
            continue
        target = out_dir / f"{name}.parquet"
        temporary = target.with_suffix(".tmp.parquet")
        pd.DataFrame(rows).to_parquet(temporary, index=False)
        temporary.replace(target)
    target = out_dir / "metadata.json"
    temporary = target.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    temporary.replace(target)


def save_point(summary, records, out_dir):
    """Write one evidence partition, then its completion marker, atomically."""
    out_dir = Path(out_dir)
    point = summary["design_id"]
    for folder, frame in (("studies", pd.DataFrame(records)), ("points", pd.DataFrame([summary]))):
        if folder == "studies":
            for key in ("image_seed", "survival_seed", "censor_seed", "permutation_seed"):
                frame[key] = np.asarray([row[key] for row in records], dtype=np.uint64)
            for key in ("conditional_p", "direct_maxT", "omnibus_p"):
                frame[key] = pd.array(frame.get(key, [None] * len(frame)), dtype="Float64")
            for key in ("maxT_reject", "conditional_reject", "direct_reject_at_bank_cutoff", "omnibus_reject"):
                frame[key] = pd.array(frame.get(key, [None] * len(frame)), dtype="boolean")
        target = out_dir / folder / f"{point}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{point}.tmp.parquet")
        frame.to_parquet(temporary, index=False)
        temporary.replace(target)


def charts(summaries, out_dir, *, cohort, fov_um=None, eps=0.1, hr=2.0):
    """Power/identification and held-out global-null error rates with 95% CIs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame(summaries)
    selected = (df.cohort == cohort) & (df.eps == eps) & (df.hr_per_source_sd == hr)
    selected &= df.fov_um.isna() if fov_um is None else df.fov_um == fov_um
    df = df[selected].sort_values("patients")
    if df.empty:
        raise ValueError("no completed design points in the requested chart slice")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fov = "full region" if fov_um is None else f"{fov_um:g} µm"
    source_n = int(df.source_n.iloc[0])
    source_events = int(df.source_events.iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True, sharey=True, layout="constrained")
    for ax, metric, title in zip(axes, ("power", "identification_rate"), ("Discovery power", "Correct identification")):
        keys = sorted(df.regions_per_patient.dropna().unique())
        if df.regions_per_patient.isna().any():
            keys.append(None)
        for i, k in enumerate(keys):
            rows = df[df.regions_per_patient.isna() if k is None else df.regions_per_patient == k]
            x, y = rows.patients.to_numpy(), rows[metric].to_numpy()
            ax.errorbar(x, y, yerr=[y - rows[metric + "_lo"], rows[metric + "_hi"] - y],
                        marker="o", capsize=2, label="All available" if k is None else f"{int(k)} regions/patient", color=f"C{i}")
        ax.axvline(source_n, color="0.4", linestyle=":", linewidth=1)
        ax.set(title=title, xlabel="Simulated patients", ylim=(-0.02, 1.02))
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Probability under the source model")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"{cohort.title()}: HR {hr:g} per source SD; {fov}; mis-call {eps:.0%}")
    method = df.measurement_model.iloc[0] if "measurement_model" in df else "direct"
    scope = "Descriptive simulation only. " if cohort == "upmc" else ""
    fig.supxlabel(f"SYNTHETIC; {method}. {scope}Source: {source_n} verified patients, {source_events} events.\n95% binomial intervals conditional on source/bank/calibration. Region caps apply.", fontsize=8)
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"{cohort}-power.{ext}", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(max(7, len(df) * 0.4), 4.5), layout="constrained")
    for offset, metric, label in ((-0.12, "fwer", "maxT"), (0.12, "omnibus_fwer", "Joint omnibus")):
        x, y = np.arange(len(df)) + offset, df[metric].to_numpy()
        ax.errorbar(x, y, yerr=[y - df[metric + "_lo"], df[metric + "_hi"] - y],
                    fmt="o", capsize=2, label=label)
    ax.axhline(float(df.alpha.iloc[0]), color="0.4", linestyle="--", label="Nominal alpha")
    labels = [f"{r.patients}/" + ("all" if pd.isna(r.regions_per_patient) else str(int(r.regions_per_patient)))
              for r in df.itertuples()]
    ax.set(xticks=np.arange(len(df)), xticklabels=labels, xlabel="Design: patients / regions per patient",
           ylabel="Held-out global-null rejection rate", ylim=(-0.01, max(0.2, df.omnibus_fwer_hi.max(), df.fwer_hi.max()) + 0.03),
           title=f"SYNTHETIC — {cohort.title()}: null validation; {fov}; mis-call {eps:.0%} (95% intervals)")
    ax.tick_params(axis="x", rotation=60)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"{cohort}-calibration.{ext}", dpi=180)
    plt.close(fig)


def comparison_charts(summaries, out_dir, *, cohort):
    """Paired shortcut/direct-imaging gaps, including finite-sample MC bounds."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame(summaries)
    df = df[df.cohort == cohort]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    any_data = False
    for ax, stage in zip(axes, ("null", "alternative")):
        checked = df[df.get(stage + "_nested_shared_minus_conditional", pd.Series(np.nan, index=df.index)).notna()
                     | df.get(stage + "_bank_minus_direct_rejection", pd.Series(np.nan, index=df.index)).notna()]
        positions = {point: i for i, point in enumerate(checked.design_id.unique())}
        for j, (kind, mean, label) in enumerate((
            ("nested", "nested_shared_minus_conditional", "Shared cutoff − conditional permutations"),
            ("bank_direct", "bank_minus_direct_rejection", "Bank − direct imaging, same cutoff"),
        )):
            key = stage + "_" + mean
            if key not in df:
                continue
            rows = df[df[key].notna()]
            if rows.empty:
                continue
            any_data = True
            y = rows[key].to_numpy()
            lo, hi = (rows[f"{stage}_{kind}_difference_{end}"].to_numpy() for end in ("lo", "hi"))
            x = np.array([positions[point] for point in rows.design_id])
            ax.errorbar(x + j * 0.12, y, yerr=[y - lo, hi - y], fmt="o", capsize=3, label=label)
        checked = checked.drop_duplicates("design_id")
        ax.set_xticks(np.arange(len(checked)))
        ax.set_xticklabels([f"{r.patients}/{r.regions_per_patient}/{r.fov_um}/{r.eps}" for r in checked.itertuples()], rotation=30, ha="right", fontsize=8)
        ax.axhline(0, color="0.4", linestyle="--")
        ax.set(title=stage.title(), xlabel="Patients / regions / FOV / mis-call", ylabel="Paired rejection-rate difference")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=7, frameon=False)
    if any_data:
        fig.suptitle(f"SYNTHETIC — {cohort.title()}: paired method checks, conservative 95% Monte Carlo bounds")
        for ext in ("png", "svg"):
            fig.savefig(Path(out_dir) / f"{cohort}-method-checks.{ext}", dpi=180)
    plt.close(fig)
