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


def test_cursor_is_off_by_default():
    assert ds.parse_args([]).cursor is False
    assert ds.parse_args(["--cursor"]).cursor is True


class FakeTarget:
    def __init__(self, calls):
        self.calls = calls

    def bounding_box(self):
        return {"x": 100, "y": 200, "width": 40, "height": 20}

    def click(self):
        self.calls.append(("click",))


class FakeClickPage(FakePage):
    def __init__(self):
        super().__init__(set())

    def locator(self, sel):
        self.calls.append(("locate", sel))
        return FakeTarget(self.calls)

    def evaluate(self, js, arg):
        self.calls.append(("glide", arg))


def test_click_with_cursor_glides_to_the_target_centre_first():
    page = FakeClickPage()
    ds._click(page, "#run", cursor=True)
    glides = [c for c in page.calls if c[0] == "glide"]
    assert glides and glides[0][1][:2] == [120, 210]  # centre of the box
    assert page.calls.index(glides[0]) < page.calls.index(("click",))


def test_click_without_cursor_draws_nothing():
    page = FakeClickPage()
    ds._click(page, "#run", cursor=False)
    assert ("click",) in page.calls and not [c for c in page.calls if c[0] == "glide"]


def test_click_with_cursor_on_a_hidden_target_stops_with_a_clear_message():
    import pytest

    class Hidden(FakeTarget):
        def bounding_box(self):
            return None  # what playwright returns for an element that is not visible

    page = FakeClickPage()
    page.locator = lambda sel: Hidden(page.calls)
    with pytest.raises(SystemExit, match="click target not visible: #run"):
        ds._click(page, "#run", cursor=True)
    assert ("click",) not in page.calls


def test_race_scroll_is_off_by_default_and_takes_one_target():
    assert ds.parse_args([]).race_scroll is None
    assert ds.parse_args(["--race-scroll", "#speed-interim"]).race_scroll == "#speed-interim"
