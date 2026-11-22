"""The guided tour leaves form controls their arrow keys and never reopens itself after a Skip (#141)."""
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

DEMO = Path(__file__).resolve().parents[1] / "demo"


class _Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def page():
    # shortcut: playwright is not a project dependency, so CI skips this file, as with test_demo_offline.py.
    sync_api = pytest.importorskip("playwright.sync_api")
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_Quiet, directory=str(DEMO)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with sync_api.sync_playwright() as pw:
        browser = pw.chromium.launch()
        p = browser.new_page(viewport={"width": 1280, "height": 720})
        p.base = f"http://127.0.0.1:{server.server_port}/index.html"
        yield p
        browser.close()
    server.shutdown()


def test_arrow_keys_on_a_slider_move_the_slider_not_the_tour(page):
    page.goto(page.base + "?tour=1", wait_until="networkidle")
    page.wait_for_selector(".tour-card")
    page.locator("#pf-patients").focus()
    page.keyboard.press("ArrowRight")
    assert page.locator("#pf-patients").input_value() == "210"
    assert page.locator("#tour-count").text_content().startswith("1 of")


def test_an_early_skip_cancels_the_pending_auto_start(page):
    page.goto(page.base, wait_until="networkidle")
    page.evaluate("localStorage.clear()")
    # A first visit in a browser that is not automated, with the clock held so auto-start stays pending.
    page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => false})")
    page.clock.install(time=0)
    page.clock.pause_at(1000)
    page.goto(page.base, wait_until="domcontentloaded")
    assert page.locator(".tour-card").count() == 0
    page.locator("#tour-btn").click()
    page.keyboard.press("Escape")
    assert page.locator(".tour-card").count() == 0
    page.clock.run_for(2000)
    assert page.locator(".tour-card").count() == 0
