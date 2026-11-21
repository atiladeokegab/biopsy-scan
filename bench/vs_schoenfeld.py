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

ESTIMATE (sim): the atlas's model-based Monte Carlo power: studies re-imaged cell by cell from the real
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


def compare(summary: list[dict], meta: dict, m: int, *, cohort: str, control: list[dict] | None = None,
            status: str | None = None) -> dict:
    """control: an optional second atlas design (e.g. full core, no mis-calls) drawn as a reference line."""
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
    design = _design(rows[0])
    extra = {}
    if control is not None:
        c = compare(control, meta, m, cohort=cohort)
        # a run stopped early may leave the control with fewer points; it may not add new ones
        if not {p["patients"] for p in c["points"]} <= {p["patients"] for p in points} or c["hr_per_sd"] != float(hr):
            raise ValueError("the control must use the main design's patient counts and planted effect")
        extra["control"] = {"label": "ESTIMATE (sim), control design", "design": c["design"],
                            "points": [{k: p[k] for k in ("patients", "sim_power", "sim_lo", "sim_hi", "sim_n",
                                                          "sim_identification", "sim_fwer", "sim_fwer_lo", "sim_fwer_hi")} for p in c["points"]]}
    return {
        "schema": "preflight-vs-schoenfeld/1", "cohort": cohort, **({"status": status} if status else {}), "verdict": verdict,
        "hr_per_sd": float(hr), "alpha": alpha, "features_m": m, "bonferroni_alpha": alpha / m,
        "event_fraction": f, "events": events, "patients_source": n,
        "design": design,
        "analytic": {"label": "ANALYTIC (formula)",
                     "method": "Schoenfeld (1983) events formula, continuous covariate (Hsieh and Lavori 2000), "
                               "D = patients x event fraction, two-sided, Bonferroni over m features; "
                               "assumes the feature is measured exactly on the whole tissue"},
        "simulated": {"label": "ESTIMATE (sim)",
                      "method": "model-based Monte Carlo (" + meta.get("inference", "atlas") + ")",
                      "budget": meta.get("budget"), "seed": meta.get("seed"), "source_hash": meta.get("source_hash"),
                      "grammar_hash": meta.get("grammar_hash"), "atlas_code": meta.get("atlas_code")},
        "points": points, **extra,
    }


def _design(r: dict) -> dict:
    return {"regions_per_patient": "all" if r["regions_per_patient"] is None else r["regions_per_patient"],
            "fov_um": "full core" if r["fov_um"] is None else r["fov_um"], "eps": r["eps"], "backend": r["backend"]}


def _fov(d: dict) -> str:
    return d["fov_um"] if isinstance(d["fov_um"], str) else f'{d["fov_um"]:g} µm'


def _fwer(name: str, points: list[dict]) -> str:
    worst = max(points, key=lambda p: p["sim_fwer"])
    ci = (f", 95% CI {worst['sim_fwer_lo']:.3f} to {worst['sim_fwer_hi']:.3f}"
          if worst.get("sim_fwer_lo") is not None else "")
    return (f"{name} {min(p['sim_fwer'] for p in points):.3f} to {worst['sim_fwer']:.3f} "
            f"(highest at {worst['patients']} patients{ci})")


def svg(res: dict) -> str:
    W, H, L, R, T, B = 900, 617, 80, 220, 70, 177
    pts = res["points"]
    x0, x1 = min(p["patients"] for p in pts), max(p["patients"] for p in pts)
    X = lambda v: L + (W - L - R) * (v - x0) / max(1, x1 - x0)  # noqa: E731
    Y = lambda v: T + (H - T - B) * (1 - v)  # noqa: E731
    d = res["design"]
    fov = _fov(d)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="13">',
           f'<rect width="{W}" height="{H}" fill="#fff"/>',
           f'<text x="12" y="26" font-size="17" font-weight="bold">{escape(res["status"] + " · ") if "status" in res else ""}Textbook power formula vs Preflight, {escape(res["cohort"].title())}</text>',
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
    labels = [(Y(pts[-1]["analytic_power"]) + 4, "#b5462f", ["ANALYTIC (formula)"]),
              (Y(pts[-1]["sim_power"]) + 4, "#2f7d4f", ["ESTIMATE (sim)", "band: 95% interval"])]
    if "control" in res:
        c = res["control"]
        line = " ".join(f"{X(p['patients']):.1f},{Y(p['sim_power']):.1f}" for p in c["points"])
        out.append(f'<polyline points="{line}" fill="none" stroke="#777" stroke-width="1.5" stroke-dasharray="2 4"/>')
        # markers keep a one-point (stopped) control visible: a polyline of one point draws nothing
        out += [f'<circle cx="{X(p["patients"]):.1f}" cy="{Y(p["sim_power"]):.1f}" r="4.5" fill="#fff" stroke="#555" stroke-width="2"/>' for p in c["points"]]
        cd = c["design"]
        labels.append((Y(c["points"][-1]["sim_power"]) + 4, "#666",
                       ["ESTIMATE (sim), control", f"{_fov(cd)}, {float(cd['eps']):.0%} mis-called"
                        + (f", {len(c['points'])} of {len(pts)} points" if len(c["points"]) < len(pts) else "")]))
    floor = T - 40
    for y, colour, lines in sorted(labels):  # stack end labels so none overlap
        y = max(y, floor)
        for i, t in enumerate(lines):
            weight = ' font-weight="bold"' if i == 0 else ' font-size="11"'
            out.append(f'<text x="{W - R + 10}" y="{y + 16 * i:.1f}" fill="{colour}"{weight}>{escape(t)}</text>')
        floor = y + 16 * len(lines) + 6
    n = pts[0]["sim_n"]
    code = res["simulated"].get("atlas_code")
    snap = f' (frozen snapshot of the atlas code, {code["base_commit"]} + working tree)' if isinstance(code, dict) else ""
    foot = [f"ESTIMATE (sim) = model-based Monte Carlo{snap}: {n:,} planted-effect studies per point, re-imaged cell",
            "by cell from real tissue, maxT at one calibrated threshold per design.",
            "Null-study FWER: " + "; ".join(_fwer(name, ps) for name, ps in
                                             (("main", pts), ("control", res.get("control", {}).get("points")))
                                             if ps) + ".",
            "ANALYTIC = Schoenfeld (1983) events formula for a per-SD covariate (Hsieh and Lavori 2000),",
            "D = patients x event fraction, Bonferroni; it assumes exact, whole-tissue measurement."]
    out += [f'<text x="12" y="{H - 111}" font-weight="bold">{escape(res["verdict"])}</text>']
    out += [f'<text x="12" y="{H - 87 + 17 * i}" fill="#555" font-size="12">{escape(t)}</text>' for i, t in enumerate(foot)]
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main(argv=None):
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("atlas_out", type=Path)
    ap.add_argument("--cohort", default="charville", choices=["charville", "upmc"])
    ap.add_argument("--control", type=Path, help="a second atlas output dir drawn as a reference design")
    ap.add_argument("--name", default="vs-schoenfeld")
    ap.add_argument("--status", help="a label for the whole result, e.g. ESTIMATE")
    a = ap.parse_args(argv)
    meta = json.loads((a.atlas_out / "metadata.json").read_text())
    audit = json.loads((ROOT / "audit" / "results" / f"{a.cohort}.json").read_text())
    read = lambda d: pd.read_parquet(d / "summary.parquet").astype(object).where(lambda t: t.notna(), None).to_dict("records")  # noqa: E731
    res = compare(read(a.atlas_out), meta, int(audit["n_features_tested"]), cohort=a.cohort,
                  control=read(a.control) if a.control else None, status=a.status)
    out = ROOT / "bench" / "results" / f"{a.name}.json"
    out.write_text(json.dumps(res, indent=1, allow_nan=False) + "\n")
    out.with_suffix(".svg").write_text(svg(res))
    print(f"{out}: {res['verdict']}")


if __name__ == "__main__":
    sys.exit(main())
