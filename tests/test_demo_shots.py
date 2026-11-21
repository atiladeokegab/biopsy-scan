"""scripts/demo_shots.py: video defaults and the optional scroll, with no browser (playwright is not a dependency)."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("demo_shots", ROOT / "scripts" / "demo_shots.py")
ds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ds)

RACE_S = 10  # demo/race.js RACE_MS


def test_default_video_reaches_the_last_still():
    a = ds.parse_args([])
    assert a.tail == 10
    assert a.hold + RACE_S + a.tail >= max(ds.STILLS_S)  # load, click and start-up add more (MEASURED: 31-34 s)


def test_scroll_is_optional_and_repeatable():
    assert ds.parse_args([]).scroll == []
    assert ds.parse_args(["--scroll", "#power", "--scroll", "#chart"]).scroll == ["#power", "#chart"]


class FakePage:
    def __init__(self, present):
        self.present, self.calls = present, []

    def evaluate(self, js, sel):
        self.calls.append(("scroll", sel))
        return sel in self.present

    def screenshot(self):
        self.calls.append(("frame",))

    def wait_for_timeout(self, ms):
        pass


def test_scroll_shows_each_target_in_order_and_reports_a_missing_one():
    page = FakePage({"#power"})
    problems = ds._scroll(page, ["#power", "#gone"], dwell_s=0.6)
    assert [c[1] for c in page.calls if c[0] == "scroll"] == ["#power", "#gone"]
    assert ("frame",) in page.calls  # the dwell keeps feeding frames to the recording
    assert problems == ["scroll target not found: #gone"]
