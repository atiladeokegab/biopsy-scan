"""Fill demo/data/baseline.json (the speed race) from bench/results/*.json; the page never needs a code change.

Run: python3 -I demo/collect_race.py [extra.json ...]. Pages runs it before every deploy.
A bench file feeds the race in one of two ways:
  Only a file with "status": "MEASURED" and "studies": 1000 (the declared design point) may fill a race row;
  anything else is refused with a message. Either
  0. build-modal's "kind": "design_point" file (slots a_lab_all_cores, b_ours_cpu_all_cores, c_ours_1_gpu,
     d_ours_n_gpu.runs["16"]), or
  1. a top-level "race" list, one entry per row of the equal-work unit (one design point, 1,000 studies):
     [{"id": "gpu-1", "seconds": 4.2, "startup_s": 90.0, "label": "MEASURED", "hardware": "Modal", "gpu": "1x NVIDIA H100"}]
     ids: lab-cpu, preflight-cpu, gpu-1, gpu-16. "seconds" is compute time; start-up goes in "startup_s",
     shown apart and never counted. Or
  2. R-14's impl-a shape (cpu_model, cohorts[].timings[]): the Charville 3,780-candidate Cox screen fills the
     "workflow" footnote only. It is different work, so it is never a race bar and never part of a ratio.
  3. the Modal runner's "kind": "ladder" file (#75): fills "interim", two measured pairs shown until the
     design point is timed: one study on 1 GPU against NumPy in the same container, and 16 studies on 1 against
     4 containers (compute only; start-up in startup_s). Each group races on its own scale.
A row no file feeds keeps its committed value (PENDING until measured). check_data.py then validates the result.
It also writes demo/data/claims.json, the power and honesty numbers the page shows, from
bench/results/vs-schoenfeld-300.json and results/charville-v2.json, so the page replays committed results offline.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo" / "data" / "baseline.json"
CLAIMS = ROOT / "demo" / "data" / "claims.json"
PLANNED_PATIENTS = 200  # the demo's study: "they plan 200 patients"
DESIGN_STUDIES = 1000  # the race's declared unit: one design point, 1,000 simulated studies (baseline.json task)
TIMED = ("MEASURED", "ESTIMATE")
SLOTS = {"lab-cpu": "a_lab_all_cores", "preflight-cpu": "b_ours_cpu_all_cores", "gpu-1": "c_ours_1_gpu"}
KEEP = ("seconds", "startup_s", "label", "hardware", "cores", "gpu", "note")


def refused(d):
    """Why a file may not fill a race row: only a MEASURED run of exactly the declared design point may."""
    if d.get("status") != "MEASURED":
        return f"status {d.get('status')!r}, not MEASURED"
    if d.get("studies") != DESIGN_STUDIES:
        return f"{d.get('studies')!r} studies, not the design point's {DESIGN_STUDIES}"
    return None


def design_point(name, d):
    """build-modal's "kind": "design_point" file: slots a-d become the four race rows."""
    s = d["slots"]
    for rid, key in SLOTS.items():
        r = s.get(key, {})
        if r.get("label") not in TIMED or not r.get("seconds"):
            continue
        gpu = r.get("gpu")
        yield rid, {"seconds": round(r["seconds"], 2), "label": r["label"], "hardware": "Modal",
                    "cores": None if gpu else r.get("workers"), "gpu": f"1x {gpu[0]}" if gpu else None, "source": name}
    run = s.get("d_ours_n_gpu", {}).get("runs", {}).get("16")
    if s.get("d_ours_n_gpu", {}).get("label") in TIMED and run:
        yield "gpu-16", {"seconds": round(run["compute_s"], 2), "startup_s": round(run["startup_and_transfer_s"], 1),
                         "label": s["d_ours_n_gpu"]["label"], "hardware": "Modal", "cores": None,
                         "gpu": f"{run['containers_seen']}x {' / '.join(sorted(set(run['gpus'])))}", "source": name}


def rows_from(path, d):
    name = f"bench/results/{path.name}"
    if d.get("race") or d.get("kind") == "design_point":
        why = refused(d)
        if why:
            print(f"{name}: refused for the race: {why}", file=sys.stderr)
        elif d.get("kind") == "design_point":
            yield from design_point(name, d)
        else:
            for r in d["race"]:
                yield r["id"], {**{k: r[k] for k in KEEP if k in r}, "source": r.get("source", name)}
    if d.get("kind") == "ladder" and d.get("status") == "MEASURED":
        yield "interim", interim(name, d)
    if "cpu_model" in d:
        for c in d.get("cohorts", []):
            if c.get("cohort") == "charville" and c.get("candidates") == 3780 and c.get("timings"):
                t = max(c["timings"], key=lambda t: t["workers"])
                one = [x for x in c["timings"] if x["workers"] == 1]
                yield "workflow", {"method": "Standard Cox screen of all 3,780 candidates (statsmodels PHReg)",
                                   "seconds": round(t["wall_s"], 2), "label": t["label"], "hardware": d["cpu_model"],
                                   "cores": t["workers"], "gpu": None, "source": name,
                                   "fingerprints": c.get("fingerprints"),
                                   **({"note": f"{one[0]['wall_s']:.1f} s on 1 core"} if one else {})}


def interim(name, d):
    """The ladder file's kernel pair and fleet pair, copied as measured; no ratio is kept."""
    k, f, lab = d["kernel"], d["fleet"], d["status"]
    rows = k["rows"]
    fleet = []
    for n, r in sorted(f["runs"].items(), key=lambda kv: int(kv[0])):
        g = sorted(set(r["gpus"]))
        each = r["studies_per_container"][0]
        fleet.append({"id": f"fleet-{n}", "method": f"{n} GPU container{'s' if int(n) > 1 else ''}, "
                      + (f"{each} studies each" if int(n) > 1 else f"all {each} studies"),
                      "role": "baseline" if n == "1" else "screen",
                      "label": lab, "seconds": round(r["compute_s_max"], 2), "startup_s": round(r["startup_and_transfer_s"], 1),
                      "hardware": "Modal", "gpu": f"{r['containers_seen']}x {' / '.join(g)}", "cores": None, "source": name})
    return [
        {"title": "One study, feature generation only (" + k["unit"].split(": ", 1)[-1] + ")", "unit": k["unit"],
         "same_scope": True, "rows": [
            {"id": "kernel-numpy", "method": "NumPy reference, same container", "role": "baseline", "label": lab,
             "seconds": round(rows["numpy_reference"]["median_s"], 3), "hardware": "Modal GPU container, on its CPU",
             "cores": None, "gpu": None, "note": f"median of {len(rows['numpy_reference']['samples_s'])}", "source": name},
            {"id": "kernel-gpu", "method": "The same step on 1 GPU", "role": "screen", "label": lab,
             "seconds": round(rows["gpu"]["median_s"], 3), "hardware": "Modal", "cores": None, "gpu": k["gpu"]["model"],
             "note": f"median of {len(rows['gpu']['samples_s'])}, exact parity", "source": name}]},
        {"title": f"{f['studies']} studies ({f['design']}) on a fleet, compute only", "unit": f["note"],
         "same_scope": True, "rows": fleet}]


def main(extra):
    race = json.loads(OUT.read_text())
    by_id = {r["id"]: r for r in race["rows"]}
    for path in sorted((ROOT / "bench" / "results").glob("*.json")) + [Path(p) for p in extra]:
        try:
            d = json.loads(path.read_text())
        except (OSError, ValueError) as e:
            sys.exit(f"{path}: {e}")
        if not isinstance(d, dict):
            continue
        for rid, fields in rows_from(path, d):
            if rid == "interim":
                race["interim"] = fields
                print(f"interim: {len(fields)} groups from {fields[0]['rows'][0]['source']}")
                continue
            if rid == "workflow":
                race["workflow"] = fields
                print(f"workflow footnote: {fields['seconds']} s {fields['label']} from {fields['source']}")
                continue
            if rid not in by_id:
                sys.exit(f"{path}: race id {rid!r} is not a row in {OUT.name}")
            by_id[rid].update(fields)
            print(f"{rid}: {fields.get('seconds')} s {fields.get('label')} from {fields['source']}")
    OUT.write_text(json.dumps(race, indent=2, ensure_ascii=False) + "\n")


def claims(write=True):
    vs_path, mt_path = "bench/results/vs-schoenfeld-300.json", "results/charville-v2.json"
    vs, mt = (json.loads((ROOT / p).read_text()) for p in (vs_path, mt_path))
    p = next(x for x in vs["points"] if x["patients"] == PLANNED_PATIENTS)
    n_null = vs["simulated"]["budget"]["null"]
    hits = round(sum(x["sim_fwer"] * n_null for x in vs["points"]))
    top = mt["maxT"]["top_hits"][0]
    out = {
        "schema": "preflight-claims/1",
        "power": {"patients": PLANNED_PATIENTS, "sim": p["sim_power"], "lo": p["sim_lo"], "hi": p["sim_hi"],
                  "n": p["sim_n"], "label": vs["status"], "formula": p["analytic_power"], "formula_label": "ANALYTIC",
                  "source": vs_path},
        "false_alarms": {"hits": hits, "n": n_null * len(vs["points"]), "target": vs["alpha"], "label": vs["status"],
                         "source": vs_path},
        "maxT": {"feature": top["feature"], "z": top["z"], "p": mt["maxT"]["p"], "tests": mt["n_features_tested"],
                 "label": mt["labels"]["maxT"], "verdict": "exploratory, not a discovery", "source": mt_path},
    }
    if not write:
        return out
    CLAIMS.write_text(json.dumps(out, indent=2) + "\n")
    print(f"claims: power {p['sim_power']:.2f} vs formula {p['analytic_power']:.2f}; false alarms {hits}/{out['false_alarms']['n']}")


if __name__ == "__main__":
    main(sys.argv[1:])
    claims()
