"""Textbook power against Preflight's simulated power on the same design points.

    uv run python -I bench/vs_schoenfeld.py ATLAS_OUT_DIR [--cohort charville] [--name vs-schoenfeld]

ATLAS_OUT_DIR is an atlas output directory (`summary.parquet` plus `metadata.json`, as `python -m atlas`
writes them). Writes bench/results/<name>.json and a one-page <name>.svg beside it.

ANALYTIC: Schoenfeld's events formula for a Cox model (Schoenfeld 1983, Biometrics 39:499-503), in its
continuous-covariate form (Hsieh and Lavori 2000, Control Clin Trials 21:552-560): with D expected events
and a log hazard ratio b per SD, a two-sided test at level a' has power
    Phi(sqrt(D) |b| - z_{1-a'/2}) + Phi(-sqrt(D) |b| - z_{1-a'/2}).
Here D = patients x the cohort's event fraction, and a' = alpha / m (Bonferroni over the m tested features).
It assumes the planted feature is measured exactly, on the whole tissue, with no imaging error.

MEASURED (sim): the atlas's model-based Monte Carlo power: studies re-imaged cell by cell from the real
cohort at each design, planted Weibull outcomes, and the maxT test at one calibrated threshold per design.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from html import escape
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[1]
N = NormalDist()


def schoenfeld_power(patients: float, event_fraction: float, hr_per_sd: float, alpha: float, m: int) -> float:
    """Two-sided Cox power for a per-SD hazard ratio, Bonferroni over m features."""
    z = N.inv_cdf(1 - alpha / (2 * m))
    s = math.sqrt(patients * event_fraction) * abs(math.log(hr_per_sd))
    return N.cdf(s - z) + N.cdf(-s - z)


def compare(summary: list[dict], meta: dict, m: int, *, cohort: str) -> dict:
    rows = [r for r in summary if r["cohort"] == cohort]
    if not rows:
        raise ValueError(f"no {cohort} rows in the atlas summary")
    one = lambda k: {r[k] for r in rows}  # noqa: E731
    for k in ("hr_per_source_sd", "alpha", "regions_per_patient", "fov_um", "eps"):
        if len(one(k)) != 1:
            raise ValueError(f"one curve per design: rows mix {k} {sorted(map(str, one(k)))}")
    if len({r["patients"] for r in rows}) != len(rows):
        raise ValueError("rows repeat a patient count")
    hr, alpha = rows[0]["hr_per_source_sd"], float(rows[0]["alpha"])
    events, n = int(meta["source_events"]), int(meta["source_n"])
    f = events / n
    points = []
    for r in sorted(rows, key=lambda r: r["patients"]):
        a = schoenfeld_power(r["patients"], f, hr, alpha, m)
        points.append({"patients": int(r["patients"]), "analytic_power": a,
                       "sim_power": float(r["power"]), "sim_lo": float(r["power_lo"]), "sim_hi": float(r["power_hi"]),
                       "sim_n": int(r["power_n"]), "sim_identification": float(r["identification_rate"]),
                       "sim_mean_events": float(r["alternative_mean_events"]), "sim_fwer": float(r["fwer"]),
                       "sim_fwer_lo": float(r["fwer_lo"]), "sim_fwer_hi": float(r["fwer_hi"]),
                       "formula_minus_sim": a - float(r["power"]),
                       "formula_outside_sim_interval": not (float(r["power_lo"]) <= a <= float(r["power_hi"]))})
    over = [p for p in points if p["formula_outside_sim_interval"] and p["formula_minus_sim"] > 0]
    under = [p for p in points if p["formula_outside_sim_interval"] and p["formula_minus_sim"] < 0]
    if over and not under:
        verdict = f"The formula overstates power at {len(over)} of {len(points)} patient counts (outside the simulation's 95% interval)."
    elif under and not over:
        verdict = f"The formula understates power at {len(under)} of {len(points)} patient counts (outside the simulation's 95% interval)."
    elif over and under:
        verdict = f"Mixed: the formula overstates at {len(over)} and understates at {len(under)} of {len(points)} patient counts."
    else:
        verdict = f"The formula falls inside the simulation's 95% interval at all {len(points)} patient counts."
    inside = len(points) - len(over) - len(under)
    if inside and (over or under):
        verdict = verdict[:-1] + f"; inside it at the other {inside}."
    design = {k: rows[0][k] for k in ("regions_per_patient", "fov_um", "eps", "backend")}
    return {
        "schema": "preflight-vs-schoenfeld/1", "cohort": cohort, "verdict": verdict,
        "hr_per_sd": float(hr), "alpha": alpha, "features_m": m, "bonferroni_alpha": alpha / m,
        "event_fraction": f, "events": events, "patients_source": n,
        "design": {**design, "fov_um": "full core" if design["fov_um"] is None else design["fov_um"],
                   "regions_per_patient": "all" if design["regions_per_patient"] is None else design["regions_per_patient"]},
        "analytic": {"label": "ANALYTIC (formula)",
                     "method": "Schoenfeld (1983) events formula, continuous covariate (Hsieh and Lavori 2000), "
                               "D = patients x event fraction, two-sided, Bonferroni over m features; "
                               "assumes the feature is measured exactly on the whole tissue"},
        "simulated": {"label": "MEASURED (sim)",
                      "method": meta.get("inference", "model-based Monte Carlo"),
                      "budget": meta.get("budget"), "seed": meta.get("seed"), "source_hash": meta.get("source_hash"),
                      "grammar_hash": meta.get("grammar_hash"), "atlas_code": meta.get("atlas_code")},
        "points": points,
    }


def svg(res: dict) -> str:
    W, H, L, R, T, B = 900, 600, 80, 220, 70, 160
    pts = res["points"]
    x0, x1 = min(p["patients"] for p in pts), max(p["patients"] for p in pts)
    X = lambda v: L + (W - L - R) * (v - x0) / max(1, x1 - x0)  # noqa: E731
    Y = lambda v: T + (H - T - B) * (1 - v)  # noqa: E731
    d = res["design"]
    fov = d["fov_um"] if isinstance(d["fov_um"], str) else f'{d["fov_um"]:g} µm'
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="13">',
           f'<rect width="{W}" height="{H}" fill="#fff"/>',
           f'<text x="12" y="26" font-size="17" font-weight="bold">Textbook power formula vs Preflight, {escape(res["cohort"].title())}</text>',
           (f'<text x="12" y="48" fill="#444">HR {res["hr_per_sd"]:g} per SD planted · alpha {res["alpha"]:g}, '
            f'Bonferroni over {res["features_m"]} features · event fraction {res["events"]}/{res["patients_source"]} · '
            f'design: {escape(str(d["regions_per_patient"]))} region(s), {escape(fov)} field, '
            f'{float(d["eps"]):.0%} mis-called</text>')]
    for v in (0, 0.25, 0.5, 0.75, 1):
        out += [f'<line x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#ddd"/>',
                f'<text x="{L - 8}" y="{Y(v) + 4:.1f}" text-anchor="end" fill="#555">{v:.0%}</text>']
    for p in pts:
        out.append(f'<text x="{X(p["patients"]):.1f}" y="{H - B + 20}" text-anchor="middle" fill="#555">{p["patients"]}</text>')
    out += [f'<text x="{(L + W - R) / 2:.0f}" y="{H - B + 42}" text-anchor="middle">patients</text>',
            f'<text x="18" y="{(T + H - B) / 2:.0f}" transform="rotate(-90 18 {(T + H - B) / 2:.0f})" text-anchor="middle">power</text>']
    band = [f"{X(p['patients']):.1f},{Y(p['sim_hi']):.1f}" for p in pts] + \
           [f"{X(p['patients']):.1f},{Y(p['sim_lo']):.1f}" for p in reversed(pts)]
    out.append(f'<polygon points="{" ".join(band)}" fill="#2f7d4f" fill-opacity="0.18"/>')
    for key, colour, dash in (("analytic_power", "#b5462f", ' stroke-dasharray="7 5"'), ("sim_power", "#2f7d4f", "")):
        line = " ".join(f"{X(p['patients']):.1f},{Y(p[key]):.1f}" for p in pts)
        out.append(f'<polyline points="{line}" fill="none" stroke="{colour}" stroke-width="2.5"{dash}/>')
        out += [f'<circle cx="{X(p["patients"]):.1f}" cy="{Y(p[key]):.1f}" r="3.5" fill="{colour}"/>' for p in pts]
    ya, ys = Y(pts[-1]["analytic_power"]) + 4, Y(pts[-1]["sim_power"]) + 4
    if abs(ya - ys) < 36:  # end labels would overlap: the sim label (two lines) goes below the formula's, or above
        ys = ya + 20 if ys >= ya else ya - 36
    out += [f'<text x="{W - R + 10}" y="{ya:.1f}" fill="#b5462f" font-weight="bold">ANALYTIC (formula)</text>',
            f'<text x="{W - R + 10}" y="{ys:.1f}" fill="#2f7d4f" font-weight="bold">MEASURED (sim)</text>',
            f'<text x="{W - R + 10}" y="{ys + 16:.1f}" fill="#2f7d4f" font-size="11">band: 95% interval</text>']
    fw = [p["sim_fwer"] for p in pts]
    n = pts[0]["sim_n"]
    foot = [f"MEASURED (sim) = model-based Monte Carlo: {n:,} planted-effect studies per point, re-imaged cell by cell",
            f"from real tissue, maxT at one calibrated threshold per design; null-study FWER {min(fw):.3f} to {max(fw):.3f}.",
            "ANALYTIC = Schoenfeld (1983) events formula for a per-SD covariate (Hsieh and Lavori 2000),",
            "D = patients x event fraction, Bonferroni; it assumes exact, whole-tissue measurement."]
    out += [f'<text x="12" y="{H - 94}" font-weight="bold">{escape(res["verdict"])}</text>']
    out += [f'<text x="12" y="{H - 70 + 17 * i}" fill="#555" font-size="12">{escape(t)}</text>' for i, t in enumerate(foot)]
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main(argv=None):
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("atlas_out", type=Path)
    ap.add_argument("--cohort", default="charville", choices=["charville", "upmc"])
    ap.add_argument("--name", default="vs-schoenfeld")
    a = ap.parse_args(argv)
    meta = json.loads((a.atlas_out / "metadata.json").read_text())
    audit = json.loads((ROOT / "audit" / "results" / f"{a.cohort}.json").read_text())
    summary = pd.read_parquet(a.atlas_out / "summary.parquet").astype(object).where(lambda d: d.notna(), None)
    res = compare(summary.to_dict("records"), meta, int(audit["n_features_tested"]), cohort=a.cohort)
    out = ROOT / "bench" / "results" / f"{a.name}.json"
    out.write_text(json.dumps(res, indent=1, allow_nan=False) + "\n")
    out.with_suffix(".svg").write_text(svg(res))
    print(f"{out}: {res['verdict']}")


if __name__ == "__main__":
    sys.exit(main())
