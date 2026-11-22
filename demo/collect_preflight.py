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


# The two designs the panel offers, exactly as the source files must state them; anything else is refused.
CROPPED = {"regions_per_patient": 2, "fov_um": 400.0, "eps": 0.1}
CONTROL = {"regions_per_patient": 2, "fov_um": "full core", "eps": 0.0}
SHARED = ("cohort", "hr_per_sd", "alpha", "patients_source")  # both files must simulate the same scenario
SHARED_SIM = ("method", "budget", "seed", "source_hash", "grammar_hash")


class Bad(ValueError):
    pass


def need(ok, msg):
    if not ok:
        raise Bad(msg)


def rate(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 0 <= v <= 1


def series(points, where):
    """Nonempty, strictly increasing positive patient counts; finite rates in [0, 1]; lo <= value <= hi; n > 0."""
    need(isinstance(points, list) and points, f"{where}: no points")
    ns = [p.get("patients") for p in points]
    need(all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in ns), f"{where}: patients must be positive integers")
    need(all(a < b for a, b in zip(ns, ns[1:])), f"{where}: patients must be strictly increasing and unique")
    for p in points:
        at = f"{where} at {p['patients']} patients"
        need(all(rate(p.get(k)) for k in ("power", "lo", "hi", "fwer", "fwer_lo", "fwer_hi")), f"{at}: a rate is not finite in [0, 1]")
        need(p["lo"] <= p["power"] <= p["hi"] and p["fwer_lo"] <= p["fwer"] <= p["fwer_hi"], f"{at}: value outside its interval")
        need(isinstance(p.get("n"), int) and not isinstance(p["n"], bool) and p["n"] > 0, f"{at}: n must be a positive integer")
    return points


def build(root=ROOT):
    raw = {p: (root / p).read_bytes() for p in (CURVE, VS)}
    curve, vs = (json.loads(raw[p]) for p in (CURVE, VS))
    need({k: curve["design"].get(k) for k in CROPPED} == CROPPED, f"{CURVE}: design is not {CROPPED}")
    need({k: vs["design"].get(k) for k in CROPPED} == CROPPED, f"{VS}: design is not {CROPPED}")
    need({k: vs["control"]["design"].get(k) for k in CONTROL} == CONTROL, f"{VS}: control design is not {CONTROL}")
    for k in SHARED:  # absent or null in both is not agreement
        need(curve.get(k) is not None and vs.get(k) is not None, f"missing {k}")
        need(curve.get(k) == vs.get(k), f"the two files disagree on {k}")
    for k in SHARED_SIM:
        need(curve["simulated"].get(k) is not None and vs["simulated"].get(k) is not None, f"missing simulated.{k}")
        need(curve["simulated"].get(k) == vs["simulated"].get(k), f"the two files disagree on simulated.{k}")
    ns = [p.get("patients") for p in vs.get("points") or []]  # an empty comparison would skip every check below
    need(ns, "comparison: no points")
    need(all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in ns) and all(a < b for a, b in zip(ns, ns[1:])),
         "comparison: patients must be positive, strictly increasing and unique")
    need(abs(vs["event_fraction"] - vs["events"] / vs["patients_source"]) < 1e-12, "event fraction is not events / source patients")
    need(vs["bonferroni_alpha"] == vs["alpha"] / vs["features_m"], "Bonferroni alpha is not alpha / m")
    for p in vs["points"]:  # the formula must reproduce the file before the page may show it
        need(abs(formula(p["patients"], vs) - p["analytic_power"]) < 1e-9, f"formula drifts at {p['patients']}")
    cropped = series([{"patients": p.get("patients"), "power": p.get("power"), "lo": p.get("power_lo"),
                       "hi": p.get("power_hi"), "n": p.get("power_n"), "fwer": p.get("fwer"),
                       "fwer_lo": p.get("fwer_lo"), "fwer_hi": p.get("fwer_hi")} for p in curve["points"]], CURVE)
    control = series([{"patients": p.get("patients"), "power": p.get("sim_power"), "lo": p.get("sim_lo"),
                       "hi": p.get("sim_hi"), "n": p.get("sim_n"), "fwer": p.get("sim_fwer"),
                       "fwer_lo": p.get("sim_fwer_lo"), "fwer_hi": p.get("sim_fwer_hi")}
                      for p in vs["control"]["points"]], f"{VS} control")
    for p in vs["points"]:  # the two files share the cropped points up to 300 patients; they must agree
        c = next((q for q in cropped if q["patients"] == p["patients"]), None)
        need(c is not None and (c["power"], c["lo"], c["hi"], c["n"]) == (p["sim_power"], p["sim_lo"], p["sim_hi"], p["sim_n"]),
             f"the two files disagree at {p['patients']} patients")
    return {
        "schema": "preflight-panel/2",
        "note": "Built by demo/collect_preflight.py from the two files below; do not edit by hand.",
        "sources": {p: hashlib.sha256(raw[p]).hexdigest() for p in (CURVE, VS)},
        "scenario": {"cohort": curve["cohort"], "patients_source": curve["patients_source"],
                     "events_source": vs["events"], "hr_per_sd": curve["hr_per_sd"], "features_m": vs["features_m"],
                     "alpha": curve["alpha"], "null_studies": curve["simulated"]["budget"]["null"], "method": curve["simulated"]["method"], "resampling": curve["resampling"]},
        "designs": [
            {"id": "cropped", "window": "400um", "miscall": CROPPED["eps"], "label": curve["label"], "source": CURVE,
             "points": cropped},
            {"id": "control", "window": "core", "miscall": CONTROL["eps"], "label": vs["control"]["label"], "source": VS,
             "points": control},
        ],
        "formula": {"label": vs["analytic"]["label"], "method": vs["analytic"]["method"], "source": VS,
                    "power": {str(n): formula(n, vs) for n in PATIENTS}},
    }


def same(a, b):
    """Equal, except floats may differ in the last bits: math.erf is the platform's libm, so CI and a laptop differ."""
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= 1e-12
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(map(same, a, b))
    return type(a) is type(b) and a == b


def main(argv):
    try:
        data = build()
    except Bad as e:
        print(f"refused: {e}")
        return 1
    text = json.dumps(data, indent=1, allow_nan=False) + "\n"
    if "--check" in argv:
        ok = OUT.exists() and same(json.loads(OUT.read_text()), data)
        print(f"{OUT.relative_to(ROOT)}: {'ok' if ok else 'STALE, run python3 -I demo/collect_preflight.py'}")
        return 0 if ok else 1
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
