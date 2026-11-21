"""Charville's best real signal against its shuffled-outcome null: one MEASURED chart.

    DATA_DIR=... uv run python -I bench/null_cloud.py            # recompute the null, write JSON + SVG
    uv run python -I bench/null_cloud.py --svg-only              # redraw the SVG from the committed JSON

Recomputes the audit's permutation test (audit/runner.py: same stats backend, same seed, same 9,999
within-stratum outcome shuffles), keeps every shuffle's maxT, and checks that the p it gets matches
audit/results/charville.json before writing bench/results/charville-null-cloud.{json,svg}.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from html import escape
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench" / "results" / "charville-null-cloud.json"
AUDIT = ROOT / "audit" / "results" / "charville.json"


def summarise(null_max: np.ndarray, observed: float, audit: dict, *, seed: int, bins: int = 40) -> dict:
    null_max = np.asarray(null_max, dtype=np.float64)
    tol = 16 * np.finfo(np.float64).eps * max(1.0, observed)
    at_least = int(np.sum(null_max >= observed - tol))
    p = (1 + at_least) / (1 + len(null_max))  # observed row counted, as in the audit
    if abs(p - audit["maxT"]["p"]) > 1e-9 or abs(observed - audit["maxT"]["statistic"]) > 1e-9:
        raise ValueError(f"recomputed maxT {observed:.6f}, p {p:.6f} do not match the audit "
                         f"({audit['maxT']['statistic']:.6f}, {audit['maxT']['p']:.6f})")
    lo, hi = float(min(null_max.min(), observed)), float(max(null_max.max(), observed))
    edges = np.linspace(math.floor(lo * 4) / 4, math.ceil(hi * 4) / 4, bins + 1)
    counts, _ = np.histogram(null_max, bins=edges)
    return {
        "schema": "preflight-null-cloud/1", "cohort": "charville", "label": "MEASURED",
        "what": "maxT (largest |z| across all tested features) of the real cohort, against the same statistic "
                "after shuffling outcomes within coverslip strata",
        "exploratory": True, "already_seen": True, "clinical_claims": False,
        "observed_maxT": observed, "observed_feature": audit["maxT"]["top_hits"][0]["feature"],
        "p": p, "null_at_least_observed": at_least, "n_resamples": int(len(null_max)), "seed": seed,
        "cutoff_alpha_05": float(audit["maxT"]["conditional_cutoff_alpha_05"]),
        "null_quantiles": {q: float(np.quantile(null_max, float(q))) for q in ("0.5", "0.9", "0.95", "0.99")},
        "features_tested": audit["n_features_tested"], "patients": audit["n_patients"], "events": audit["n_events"],
        "histogram": {"edges": [float(e) for e in edges], "counts": [int(c) for c in counts]},
        "source": "audit/results/charville.json (p, statistic and cutoff checked equal); null maxima recomputed",
    }


def svg(res: dict) -> str:
    W, H, L, R, T, B = 900, 560, 70, 40, 92, 110
    edges, counts = res["histogram"]["edges"], res["histogram"]["counts"]
    X = lambda v: L + (W - L - R) * (v - edges[0]) / (edges[-1] - edges[0])  # noqa: E731
    top = max(counts)
    Y = lambda c: H - B - (H - T - B) * c / top  # noqa: E731
    obs, cut, p = res["observed_maxT"], res["cutoff_alpha_05"], res["p"]
    one_in = round(1 / p)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="13">',
           f'<rect width="{W}" height="{H}" fill="#fff"/>',
           '<text x="12" y="28" font-size="18" font-weight="bold">Chance explains the best pattern in real Charville</text>',
           (f'<text x="12" y="52" fill="#444">MEASURED · {res["patients"]} patients, {res["events"]} events, '
            f'{res["features_tested"]} spatial features · exploratory, already-seen data, no clinical claim</text>'),
           (f'<text x="12" y="72" fill="#444">Grey: the best score in each of {res["n_resamples"]:,} studies with the '
            'outcomes shuffled within coverslip (no real signal)</text>')]
    for a, b, c in zip(edges, edges[1:], counts):
        if c:
            out.append(f'<rect x="{X(a) + 0.5:.1f}" y="{Y(c):.1f}" width="{X(b) - X(a) - 1:.1f}" '
                       f'height="{H - B - Y(c):.1f}" fill="#b9b9b9"/>')
    out.append(f'<line x1="{L}" x2="{W - R}" y1="{H - B}" y2="{H - B}" stroke="#555"/>')
    v = math.ceil(edges[0] * 2) / 2
    while v <= edges[-1] + 1e-9:
        out += [f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{H - B}" y2="{H - B + 5}" stroke="#555"/>',
                f'<text x="{X(v):.1f}" y="{H - B + 20}" text-anchor="middle" fill="#555">{v:g}</text>']
        v += 0.5
    out.append(f'<text x="{(L + W - R) / 2:.0f}" y="{H - B + 42}" text-anchor="middle">'
               'best score across all features in one study (maxT)</text>')
    out += [f'<line x1="{X(cut):.1f}" x2="{X(cut):.1f}" y1="{T}" y2="{H - B}" stroke="#777" stroke-dasharray="5 4"/>',
            f'<text x="{X(cut) + 6:.1f}" y="{T + 12}" fill="#555" font-size="12">{cut:.2f}: needed for p &lt; 0.05</text>']
    # the real study: a plain dark point on the axis, deliberately not styled as a discovery
    halo = 'stroke="#fff" stroke-width="4" paint-order="stroke"'
    out += [f'<line x1="{X(obs):.1f}" x2="{X(obs):.1f}" y1="{T + 48}" y2="{H - B}" stroke="#222" stroke-width="1.5"/>',
            f'<circle cx="{X(obs):.1f}" cy="{H - B}" r="6" fill="#222"/>',
            f'<text x="{X(obs) + 6:.1f}" y="{T + 60}" font-weight="bold" {halo}>Real Charville: {obs:.2f}</text>',
            f'<text x="{X(obs) + 6:.1f}" y="{T + 78}" fill="#333" {halo}>p = {p:.2f}: 1 in {one_in} shuffles score this high</text>']
    out += [(f'<text x="12" y="{H - 38}" fill="#555" font-size="12">Best feature: {escape(res["observed_feature"])}. '
             f'{res["null_at_least_observed"]:,} of {res["n_resamples"]:,} shuffled studies scored at least as high.</text>'),
            (f'<text x="12" y="{H - 20}" fill="#555" font-size="12">Single-step maxT permutation test, seed {res["seed"]}; '
             'the same run as audit/results/charville.json.</text>'),
            "</svg>"]
    return "\n".join(out) + "\n"


def compute(seed: int = 0, n_resamples: int = 9999) -> dict:
    from audit.runner import load_default_core, load_default_stats
    from stats.reference import maxT

    core, stats = load_default_core(), load_default_stats()
    cells = core.load("charville")
    x = np.asarray(core.truth(cells), dtype=np.float64)
    perm = stats.permutation_test(x, np.asarray(cells["time"], np.float64), np.asarray(cells["event"], np.float64),
                                  np.asarray(cells["strata"], np.int32), n_resamples=n_resamples, seed=seed)
    return summarise(maxT(perm["null_z"]), float(perm["maxT"]), json.loads(AUDIT.read_text()), seed=seed)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--svg-only", action="store_true")
    a = ap.parse_args(argv)
    if a.svg_only:
        res = json.loads(OUT.read_text())
    else:
        sys.path.insert(0, str(ROOT))
        res = compute()
        OUT.write_text(json.dumps(res, indent=1, allow_nan=False) + "\n")
    OUT.with_suffix(".svg").write_text(svg(res))
    print(f"{OUT.with_suffix('.svg')}: maxT {res['observed_maxT']:.3f}, p {res['p']:.4f}")


if __name__ == "__main__":
    sys.exit(main())
