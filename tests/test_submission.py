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
