"""Fill demo/data/preflight.json (the "Preflight your study" panel) from two committed power files, nothing else.

Run: python3 -I demo/collect_preflight.py [--check]. --check exits 1 if the committed file differs from a fresh build.
  bench/results/power-at-scale.json  the cropped design (2 x 400 um fields, 10% mis-called), 40 to 1,000 patients
  bench/results/vs-schoenfeld-300.json  the control design (whole core, no mis-calls), 40 to 300 patients, and the
                                        Schoenfeld formula's inputs
The formula is evaluated here for every slider stop and must reproduce the file's own analytic_power at each of its
points, so the page never computes a number the files do not pin. Interpolation happens on the page, labelled.
"""
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[1]
CURVE = "bench/results/power-at-scale.json"
VS = "bench/results/vs-schoenfeld-300.json"
OUT = ROOT / "demo/data/preflight.json"
PATIENTS = range(50, 601, 10)  # the slider: 50 to 600 in steps of 10


def formula(n, vs):
    """Schoenfeld events formula, as the file states it: D = patients x event fraction, two-sided, Bonferroni over m."""
    z = NormalDist().inv_cdf(1 - vs["bonferroni_alpha"] / 2)
    s = math.sqrt(n * vs["event_fraction"]) * math.log(vs["hr_per_sd"])
    return NormalDist().cdf(s - z) + NormalDist().cdf(-s - z)


def build():
    raw = {p: (ROOT / p).read_bytes() for p in (CURVE, VS)}
    curve, vs = (json.loads(raw[p]) for p in (CURVE, VS))
    assert vs["bonferroni_alpha"] == vs["alpha"] / vs["features_m"], "Bonferroni alpha is not alpha / m"
    for p in vs["points"]:  # the formula must reproduce the file before the page may show it
        assert abs(formula(p["patients"], vs) - p["analytic_power"]) < 1e-9, f"formula drifts at {p['patients']}"
    cropped = [{"patients": p["patients"], "power": p["power"], "lo": p["power_lo"], "hi": p["power_hi"],
                "n": p["power_n"]} for p in curve["points"]]
    for p in vs["points"]:  # the two files share the cropped points up to 300 patients; they must agree
        c = next(q for q in cropped if q["patients"] == p["patients"])
        assert (c["power"], c["lo"], c["hi"], c["n"]) == (p["sim_power"], p["sim_lo"], p["sim_hi"], p["sim_n"]), \
            f"the two files disagree at {p['patients']} patients"
    d, cd = curve["design"], vs["control"]["design"]
    assert (d["fov_um"], d["eps"], d["regions_per_patient"]) == (vs["design"]["fov_um"], vs["design"]["eps"], 2)
    assert (cd["fov_um"], cd["eps"]) == ("full core", 0.0)
    return {
        "schema": "preflight-panel/1",
        "note": "Built by demo/collect_preflight.py from the two files below; do not edit by hand.",
        "sources": {p: hashlib.sha256(raw[p]).hexdigest() for p in (CURVE, VS)},
        "cohort": curve["cohort"], "hr_per_sd": curve["hr_per_sd"], "alpha": curve["alpha"],
        "designs": [
            {"id": "cropped", "window": "400um", "miscall": d["eps"], "regions": d["regions_per_patient"],
             "label": curve["label"], "source": CURVE, "points": cropped},
            {"id": "control", "window": "core", "miscall": cd["eps"], "regions": cd["regions_per_patient"],
             "label": vs["control"]["label"], "source": VS,
             "points": [{"patients": p["patients"], "power": p["sim_power"], "lo": p["sim_lo"], "hi": p["sim_hi"],
                         "n": p["sim_n"]} for p in vs["control"]["points"]]},
        ],
        "formula": {"label": vs["analytic"]["label"], "method": vs["analytic"]["method"], "source": VS,
                    "features_m": vs["features_m"], "event_fraction": vs["event_fraction"],
                    "power": {str(n): formula(n, vs) for n in PATIENTS}},
    }


def main(argv):
    text = json.dumps(build(), indent=1) + "\n"
    if "--check" in argv:
        ok = OUT.exists() and OUT.read_text() == text
        print(f"{OUT.relative_to(ROOT)}: {'ok' if ok else 'STALE, run python3 -I demo/collect_preflight.py'}")
        return 0 if ok else 1
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
