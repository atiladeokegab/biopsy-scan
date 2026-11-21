"""A-11.1@2: docs/SUBMISSION.md states a 5-minute pitch with a live demo and the demo video as backup."""
import re
from pathlib import Path

SUBMISSION = Path(__file__).resolve().parent.parent / "docs" / "SUBMISSION.md"


def format_problems(text):
    """What the submission doc's format statement is missing; empty when it is right."""
    problems = []
    if not re.search(r"\b5-minute pitch\b", text):
        problems.append("no '5-minute pitch'")
    if not re.search(r"\blive demo\b", text, re.I):
        problems.append("no live demo")
    if not re.search(r"demo video as backup", text, re.I):
        problems.append("no demo video as backup")
    if re.search(r"2\s*[-–]\s*3[ -]minute", text):
        problems.append("still says 2-3 minutes")
    return problems


def test_submission_states_five_minute_pitch_with_live_demo_and_backup_video():
    assert format_problems(SUBMISSION.read_text(encoding="utf-8")) == []


def test_check_rejects_the_old_two_to_three_minute_copy():
    old = "A live 2-3 minute pitch to the judges, with the demo video as backup if the live demo fails."
    assert "no '5-minute pitch'" in format_problems(old)
    assert "still says 2-3 minutes" in format_problems(old)
    assert "still says 2-3 minutes" in format_problems("a live 2–3 minute pitch")


ROOT = SUBMISSION.parent.parent
JUDGING = ROOT / "docs" / "JUDGING.md"


def merged_result_rows():
    """Each JUDGING row (by its opening words) -> its source file and the figures computed from that file."""
    import json

    def load(path):
        return json.loads((ROOT / path).read_text(encoding="utf-8"))

    pct = lambda v: f"{100 * v:.1f}%"
    pp = lambda v: f"{100 * v:+.1f}".replace("-", "−")  # percentage points, typographic minus
    design = "bench/results/modal-design-20261121T194458Z.json"
    d = load(design)
    lab, numba = d["arms"]["lab"], d["arms"]["numba"]
    two = "bench/results/event-count-calibration.json"
    alt = load(two)["methods"]["two_bins"]["paired_vs_pooled"]["alternative"]
    lo, hi = alt["paired_bootstrap95"]
    matched = "bench/results/matched-e2e-20261121T201925Z.json"
    e = load(matched)
    n = e["wall"]["studies"]
    return {
        "| The race (#113)": (design, [
            f"{lab['stages']['wall_per_study_s'] / numba['stages']['wall_per_study_s']:.1f}×",
            f"{lab['stages']['studies_wall_s'] / numba['stages']['studies_wall_s']:.2f}×",
        ]),
        "| Matched workflow comparison through scoring (#123)": (matched, [
            f"{e['wall']['ratio']:.1f}×",
            f"{e['wall']['with_setup']['ratio']:.1f}×",
            f"{e['decisions']['agree']} of {e['decisions']['compared']}",
            f"{n['lab']:,} lab studies",
            f"{n['preflight_evaluated']:,} evaluated",
            f"{n['preflight_decided']:,} decided",
            f"{n['preflight_calibration']:,} calibration",
            "excludes data load, the threshold calculation and the final calls",
        ]),
        "| Testing every candidate becomes one matrix product": (design, [
            f"{d['arms']['gpu']['stages']['scoring_wall_s']:.2f} s",
            f"{d['notes']['x_shape']['features']} features",
        ]),
        "| The shared-threshold procedure had lower estimated power": (design, [
            pct(numba["summary"]["power"]["rate"]), pct(lab["summary"]["power"]["rate"]),
            pct(numba["summary"]["fwer"]["rate"]), pct(lab["summary"]["fwer"]["rate"]),
        ]),
        "| Event-count calibration does not close the power gap": (two, [f"{pp(alt['difference'])} points [{pp(lo)}, {pp(hi)}]"]),
    }


def binding_problems(text):
    """Figures missing from their own row, or rows without their source; empty when every figure sits in its row."""
    lines = text.splitlines()
    problems = []
    for start, (source, quotes) in merged_result_rows().items():
        rows = [line for line in lines if line.startswith(start)]
        if len(rows) != 1:
            problems.append(f"{start}: {len(rows)} rows")
            continue
        problems += [f"{start}: {q}" for q in [source, *quotes] if q not in rows[0]]
    return problems


def test_judging_binds_each_merged_figure_to_its_own_row_and_source():
    assert binding_problems(JUDGING.read_text(encoding="utf-8")) == []


def test_binding_check_catches_swapped_race_and_matched_ratios():
    text = JUDGING.read_text(encoding="utf-8")
    race = merged_result_rows()["| The race (#113)"][1][0]
    swapped = text.replace(race, "@").replace("42.1×", race).replace("@", "42.1×")
    assert any("7.0×" in p for p in binding_problems(swapped)) and any("42.1×" in p for p in binding_problems(swapped))


def test_no_rate_equivalence_wording():
    for doc in (JUDGING, SUBMISSION):
        text = doc.read_text(encoding="utf-8")
        for phrase in ("same FWER and power answer", "false alarms match", "speed costs power", "like-for-like"):
            assert phrase not in text, (doc.name, phrase)


def test_submission_does_not_claim_every_speed_up_is_against_rung_0():
    assert "Every speed-up is quoted against rung 0" not in SUBMISSION.read_text(encoding="utf-8")


SIM_VS_REAL = "the difference interval contains zero at all three sizes; outcome model only, not imaging noise"


def sim_vs_real_quotes():
    """#119: the per-size difference and its paired interval, in points, from the committed file."""
    import json

    data = json.loads((ROOT / "bench/results/real-check-charville.json").read_text(encoding="utf-8"))
    assert data["labels"] == {"counts": "MEASURED", "outcomes": "SYNTHETIC", "power": "ESTIMATE"}
    assert len(data["points"]) == 3 and all(p["difference_interval_contains_zero"] for p in data["points"])
    pp = lambda v: f"{100 * v:+.2f}".replace("-", "−")
    return [f"{pp(p['difference'])} points [{pp(p['difference_lo'])}, {pp(p['difference_hi'])}]" for p in data["points"]]


def test_both_docs_carry_sim_vs_real_with_its_labels_and_no_equivalence_claim():
    judging, submission = JUDGING.read_text(encoding="utf-8"), SUBMISSION.read_text(encoding="utf-8")
    row = next(line for line in judging.splitlines() if "real-check-charville.json" in line)
    assert [q for q in sim_vs_real_quotes() if q not in row] == []
    for text in (row, next(line for line in submission.splitlines() if "#119" in line)):
        assert SIM_VS_REAL in text.lower()
        assert "counts MEASURED" in text and "outcomes SYNTHETIC" in text and "power ESTIMATE" in text
        assert "equivalen" not in text.lower()


def test_matched_run_is_not_divided_per_core():
    assert "per core" not in JUDGING.read_text(encoding="utf-8")
