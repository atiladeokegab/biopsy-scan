"""Fill demo/data/baseline.json (the speed race) from bench/results/*.json; the page never needs a code change.

Run: python3 -I demo/collect_race.py [extra.json ...]. Pages runs it before every deploy.
A bench file feeds the race in one of two ways:
  Only a file with "status": "MEASURED" and "studies": 1000 (the declared design point) may fill a race row;
  anything else is refused with a message. Either
  0. bench/modal_design.py's four-arm file (arms lab, numba, gpu, hybrid side by side, one container class): fills
     the four lanes lab-cpu, preflight-cpu, gpu-1 and hybrid, only when MEASURED at 1,000 studies per stream with
     its parity gate passed (X sha256 on every shared study; z in full for gpu, via per-study maxT for hybrid). A file without all four arms (the three-arm run) feeds nothing. There is no 16-GPU lane.
     The same file plus a fused arm (a --compare-with run) fills only fused-cpu, and only when the fused X
     sha256 equals every other arm's on every shared study. Or
  1. build-modal's "kind": "design_point" file (slots a_lab_all_cores, b_ours_cpu_all_cores, c_ours_1_gpu), or
     It must carry its "scope" (WORKFLOW comparison, EXPLORATORY calibration), which the page prints; GPU lanes
     need e_gpu_parity.exact_all_containers; the threshold-and-calls step is kept apart in "calls_s".
     A generic "race" list feeds nothing. Or
  2. R-14's impl-a shape (cpu_model, cohorts[].timings[]): the Charville 3,780-candidate Cox screen fills the
     "workflow" footnote only. It is different work, so it is never a race bar and never part of a ratio.
  3. the Modal runner's "kind": "ladder" file (#75), only the one Atlas cleared (CLEARED_LADDER): fills "interim",
     two measured pairs shown until the design point is timed: one study's feature step on 1 GPU against NumPy in
     the same container, and 1,000 studies on 1 A100 against 8 (compute span; start-up in startup_s). Each group
     races on its own scale, with Atlas's cleared scope line, numbers from the file, beside its bars.
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
CLEARED_LADDER = "modal-a100-20261121T183709Z.json"  # Atlas cleared this run (lead, 2026-11-21 19:00Z)
SLOTS = {"lab-cpu": "a_lab_all_cores", "preflight-cpu": "b_ours_cpu_all_cores", "gpu-1": "c_ours_1_gpu"}
ARMS = {"lab-cpu": "lab", "preflight-cpu": "numba", "gpu-1": "gpu", "hybrid": "hybrid"}  # the four declared lanes
FUSED = {"fused-cpu": "fused"}  # the fifth lane: a --compare-with run that adds the fused arm to the four-arm file


def four_arms(name, d):
    """bench/modal_design.py's file: each arm's compute_wall_s; load, prep, adjacency and warm-up apart in startup_s."""
    why = ("not MEASURED" if d.get("status") != "MEASURED" else
           "not 1,000 studies per stream" if d.get("design", {}).get("studies_per_stream") != DESIGN_STUDIES else
           "parity gate not passed" if d.get("parity", {}).get("all") is not True else
           "not the four declared arms" if sorted(set(d["arms"]) - {"fused"}) != sorted(ARMS.values()) else
           "fused X not bit-identical to every other arm" if "fused" in d["arms"] and not fused_x_parity(d) else None)
    if why:
        print(f"{name}: refused for the race: {why}", file=sys.stderr)
        return
    lab = d["labels"]
    yield "scope", {"comparison": f"Compute {lab['compute']} on one container class ({d['arms']['lab']['container_class']}); "
                                  f"outcomes {lab['outcomes']}; FWER and power are {lab['rates'].split(' ')[0]}. "
                                  "Seconds: in-container study and scoring wall time; load, outcome prep, adjacency "
                                  "and warm-up apart in start-up; the cutoff and rate summary not timed. The lab and "
                                  "batched lanes run different test and calibration procedures, not equal statistical "
                                  "work. Parity: X sha256 on every shared study; z in full for the A100 lane, through "
                                  "per-study maxT for the hybrid"
                                  + ("; the fused lane: same features, same top score (X sha256 on every shared "
                                     "study, maxT through each study)" if "fused" in d["arms"] else ""),
                    "calibration": "Lab: 1,000 within-strata shuffles in every study. Preflight lanes: one threshold "
                                   "from 1,000 independent calibration nulls; evaluating those studies is timed inside "
                                   "each lane, computing the cutoff from them is not"}
    lanes = {**ARMS, **FUSED} if "fused" in d["arms"] else ARMS
    for rid, a in lanes.items():
        v = d["arms"][a]
        if v.get("source", "this run") != "this run":
            continue  # copied from the four-arm file, which fills that lane itself
        st, gpu = v["stages"], a in ("gpu", "hybrid")
        yield rid, {"seconds": round(st["compute_wall_s"], 2), "label": lab["compute"],
                    "hardware": v["container_class"], "cores": None if a == "gpu" else v["workers"],
                    "gpu": f"1x {v['host']['gpu'].split(',')[0]}" if gpu else None, "source": name,
                    "startup_s": round(st["end_to_end_in_container_s"] - st["compute_wall_s"], 1),
                    "note": f"{st['evaluated_studies']:,} studies; re-imaging {100 * v['split']['reimage_share']:.1f}% "
                            + ("of elapsed stage wall time" if gpu else "of summed worker wall time")
                            + ("; same features, same top score; run on a different host of the same container class, about 2 h "
                               "after the lab, Numba and A100 lanes and 70 min after the hybrid; one sweep" if a == "fused" else "")}


def fused_x_parity(d):
    """Full X parity for the fused arm: sha256 equal to each other arm on every shared study (none skipped)."""
    prior = d["parity"].get("against_prior", {}).get("fused", {})
    return all(prior.get(a, {}).get("x_bitwise") is True and prior[a].get("x_compared", 0) > 0
               for a in ARMS.values())


def refused(d):
    """Why a file may not fill a race row: only a MEASURED run of exactly the declared design point may."""
    if d.get("status") != "MEASURED":
        return f"status {d.get('status')!r}, not MEASURED"
    if d.get("studies") != DESIGN_STUDIES:
        return f"{d.get('studies')!r} studies, not the design point's {DESIGN_STUDIES}"
    return None


def design_point(name, d):
    """build-modal's "kind": "design_point" file: slots a-d become the four race rows."""
    s, scope = d["slots"], d.get("scope") or {}
    if not all(isinstance(scope.get(k), str) and scope[k] for k in ("comparison", "calibration")):
        print(f"{name}: refused for the race: no WORKFLOW/EXPLORATORY scope to print", file=sys.stderr)
        return
    yield "scope", {k: scope[k] for k in ("comparison", "calibration")}
    parity = d.get("e_gpu_parity", {}).get("exact_all_containers") is True
    for rid, key in SLOTS.items():
        r = s.get(key, {})
        if r.get("label") not in TIMED or not r.get("seconds"):
            continue
        gpu = r.get("gpu")
        if gpu and not parity:  # a GPU lane must give the CPU's answer exactly
            continue
        yield rid, {"seconds": round(r["seconds"], 2), "label": r["label"], "hardware": "Modal",
                    "cores": None if gpu else r.get("workers"), "gpu": f"1x {gpu[0]}" if gpu else None, "source": name,
                    **({"calls_s": round(r["plus_threshold_and_calls_s"], 2)} if "plus_threshold_and_calls_s" in r else {}),
                    **({"note": r["basis"]} if r.get("basis") not in (None, "measured") else {})}


def rows_from(path, d):
    name = f"bench/results/{path.name}"
    if isinstance(d.get("arms"), dict) and "labels" in d:
        yield from four_arms(name, d)
    if d.get("kind") == "design_point":
        why = refused(d)
        if why:
            print(f"{name}: refused for the race: {why}", file=sys.stderr)
        else:
            yield from design_point(name, d)
    if d.get("kind") == "ladder" and d.get("status") == "MEASURED":
        why = None if path.name == CLEARED_LADDER else "not the ladder run Atlas cleared"
        groups = None if why else interim(name, d)
        if groups:
            yield "interim", groups
        else:
            print(f"{name}: refused for the race: {why or 'its 1-vs-8 A100 pair is not clean'}", file=sys.stderr)
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
    """The cleared ladder file's kernel pair and 1-vs-8 fleet pair, copied as measured, or None if either is not clean."""
    k, f, lab = d["kernel"], d["fleet"], d["status"]
    rows, runs = k["rows"], f["runs"]
    one, eight = runs.get("1"), runs.get("8")
    clean = lambda r, n: (r and r["containers_seen"] == n and r["gpus"] == one["gpus"] and len(r["gpus"]) == 1
                          and r["all_overlapped"] and r["seed_coverage_exact"] and r["bit_identical_to_1"])
    if not (k.get("parity_exact") is True and f["studies"] == DESIGN_STUDIES and clean(one, 1) and clean(eight, 8)):
        return None
    gpu = one["gpus"][0].removeprefix("NVIDIA ")
    fleet = [{"id": f"fleet-{n}", "method": "1 A100, all 1,000 studies" if n == 1 else f"{n} A100s, {r['studies_per_container'][0]} studies each",
              "role": "baseline" if n == 1 else "screen", "label": lab, "seconds": round(r["compute_span_s"], 2),
              "startup_s": round(r["startup_and_transfer_s"], 1), "note": "compute span", "hardware": "Modal",
              "gpu": f"{n}x {r['gpus'][0]}", "cores": None, "source": name} for n, r in ((1, one), (8, eight))]
    return [
        {"title": "One study", "unit": k["unit"], "same_scope": True,
         "scope_line": f"{lab}, {k['speedup_median']:.2f}× faster: feature step only, NumPy vs GPU, same container, "
                  f"median of {len(rows['gpu']['samples_s'])}, exact parity", "rows": [
            {"id": "kernel-numpy", "method": "NumPy reference, same container", "role": "baseline", "label": lab,
             "seconds": round(rows["numpy_reference"]["median_s"], 4), "hardware": "Modal GPU container, on its CPU",
             "cores": None, "gpu": None, "note": f"median of {len(rows['numpy_reference']['samples_s'])}", "source": name},
            {"id": "kernel-gpu", "method": "The same step on 1 GPU", "role": "screen", "label": lab,
             "seconds": round(rows["gpu"]["median_s"], 4), "hardware": "Modal", "cores": None, "gpu": k["gpu"]["model"],
             "note": f"median of {len(rows['gpu']['samples_s'])}, exact parity", "source": name}]},
        {"title": "1,000 studies: 1 A100 against 8", "unit": f["note"], "same_scope": True,
         "scope_line": f"{lab}, observed {one['compute_span_s'] / eight['compute_span_s']:.1f}× compute-span reduction "
                  f"{one['compute_span_s']:.1f}→{eight['compute_span_s']:.1f} s for {f['studies']:,} Charville "
                  f"re-imaging/feature-generation studies, 8 {gpu} vs 1, one sweep; outputs bit-identical; "
                  "inference outside timing", "rows": fleet}]


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
            if rid == "scope":
                race["scope"] = fields
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
                  "model": f"a planted effect of HR {vs['hr_per_sd']:g} per source SD among {vs['features_m']} candidate "
                           f"features, {vs['design']['regions_per_patient']} fields of {vs['design']['fov_um']:g} µm per "
                           f"patient, {vs['design']['eps']:.0%} cells mis-called (simulation: calibrated maxT; formula: "
                           "Bonferroni); not the PD1 pattern's power",
                  "source": vs_path},
        "false_alarms": {"hits": hits, "n": n_null * len(vs["points"]), "designs": len(vs["points"]),
                         "target": vs["alpha"], "label": vs["status"],
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
