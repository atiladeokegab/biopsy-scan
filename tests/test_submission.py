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


def merged_result_quotes():
    """The figures docs/JUDGING.md must quote, each computed from its committed results file (#113, #117, #123)."""
    import json

    def load(path):
        return json.loads((ROOT / path).read_text(encoding="utf-8"))

    race = load("bench/results/modal-design-20261121T194458Z.json")["arms"]
    lab, numba = race["lab"]["stages"], race["numba"]["stages"]
    two_bin = load("bench/results/event-count-calibration.json")["methods"]["two_bins"]["paired_vs_pooled"]["alternative"]
    lo, hi = two_bin["paired_bootstrap95"]
    e2e = load("bench/results/matched-e2e-20261121T201925Z.json")
    pp = lambda v: f"{100 * v:+.1f}".replace("-", "−")  # percentage points, typographic minus
    return [
        f"{lab['wall_per_study_s'] / numba['wall_per_study_s']:.1f}×",
        f"{lab['studies_wall_s'] / numba['studies_wall_s']:.2f}×",
        f"{pp(two_bin['difference'])} points [{pp(lo)}, {pp(hi)}]",
        f"{e2e['wall']['ratio']:.1f}×",
        f"{e2e['wall']['with_setup']['ratio']:.1f}×",
        f"{e2e['decisions']['agree']} of {e2e['decisions']['compared']}",
    ]


def test_judging_quotes_the_merged_race_two_bin_and_matched_results_from_their_files():
    text = JUDGING.read_text(encoding="utf-8")
    assert [q for q in merged_result_quotes() if q not in text] == []


def test_submission_does_not_claim_every_speed_up_is_against_rung_0():
    assert "Every speed-up is quoted against rung 0" not in SUBMISSION.read_text(encoding="utf-8")
