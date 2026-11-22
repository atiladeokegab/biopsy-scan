"""The demo page runs with no network: it references no remote URL and, served locally, requests nothing off-box."""
import re
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import pytest

DEMO = Path(__file__).resolve().parents[1] / "demo"
# XML namespace identifiers are names, not requests.
NAMESPACES = {"http://www.w3.org/2000/svg", "http://www.w3.org/1999/xlink", "http://www.w3.org/1999/xhtml"}
URL = re.compile(r"https?://[^\s\"'`)<>]+")


def remote_urls(text: str) -> list[str]:
    return [u for u in URL.findall(text) if u not in NAMESPACES]


def test_remote_url_scan_flags_a_url_and_skips_namespaces():
    assert remote_urls('const ns = "http://www.w3.org/2000/svg";') == []
    assert remote_urls('<script src="https://cdn.example.com/x.js"></script>') == ["https://cdn.example.com/x.js"]


def test_demo_sources_name_no_remote_url():
    files = [p for ext in ("html", "js", "css") for p in DEMO.glob(f"*.{ext}")] + list(DEMO.glob("workflow/*.js"))
    assert files
    found = {str(p.relative_to(DEMO)): remote_urls(p.read_text(encoding="utf-8")) for p in files}
    assert not {k: v for k, v in found.items() if v}


class _Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def test_served_demo_requests_nothing_off_box():
    # shortcut: playwright is not a project dependency, so CI skips this half; run it with
    # `uv run --with playwright==1.64.0 pytest tests/test_demo_offline.py` (needs its chromium installed).
    sync_api = pytest.importorskip("playwright.sync_api")
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_Quiet, directory=str(DEMO)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    blocked, local_errors = [], []

    def route(r):
        if urlsplit(r.request.url).hostname in ("127.0.0.1", "localhost"):
            r.continue_()
        else:
            blocked.append(r.request.url)
            r.abort()

    try:
        with sync_api.sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page()
            page.route("**/*", route)
            # bench.json is optional (the page shows PENDING on 404); anything else missing is a fault.
            page.on("response", lambda r: r.status >= 400 and not r.url.endswith("/data/bench.json")
                    and local_errors.append(f"{r.status} {r.url}"))
            page.goto(f"http://127.0.0.1:{server.server_port}/index.html", wait_until="networkidle")
            if page.locator("#run").is_enabled():
                page.locator("#run").click()
                page.wait_for_load_state("networkidle")
            browser.close()
    finally:
        server.shutdown()
    assert blocked == [], f"off-box requests: {blocked}"
    assert local_errors == [], f"failed local requests: {local_errors}"
