"""Screenshot the demo's first screen at desktop and phone size, as a pre-recorded fallback.

    uv run --frozen --with playwright==1.64.0 python -I scripts/demo_shots.py

Serves demo/ on 127.0.0.1, blocks every off-box request, waits for the page to finish its first render,
and writes demo/shots/first-screen-<w>x<h>.png. Exits 1 if the page shows an error or anything goes off-box.
"""
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright

DEMO = Path(__file__).resolve().parents[1] / "demo"
SIZES = [(1280, 720), (390, 844)]
READY = """() => {
  const t = document.getElementById('verdict').textContent, c = document.getElementById('tissue-caption').textContent;
  return t && t !== 'Evaluating design...' && c && c !== 'Preparing tissue slabs...';
}"""


class _Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def main(out: Path = DEMO / "shots") -> int:
    out.mkdir(exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_Quiet, directory=str(DEMO)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    blocked, problems = [], []

    def route(r):
        if urlsplit(r.request.url).hostname in ("127.0.0.1", "localhost"):
            r.continue_()
        else:
            blocked.append(r.request.url)
            r.abort()

    try:
        with sync_playwright() as pw:
            # software WebGL so the 3D tissue renders on a box with no GPU
            browser = pw.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
            for w, h in SIZES:
                page = browser.new_page(viewport={"width": w, "height": h}, reduced_motion="reduce")
                page.route("**/*", route)
                page.goto(f"http://127.0.0.1:{server.server_port}/index.html", wait_until="networkidle")
                page.wait_for_function(READY, timeout=30_000)
                page.wait_for_timeout(500)  # one more frame for the WebGL canvas
                if page.locator("#error").is_visible():
                    problems.append(f"{w}x{h}: error shown: {page.locator('#error').inner_text()}")
                if "could not start WebGL" in page.locator("#tissue-caption").inner_text():
                    problems.append(f"{w}x{h}: WebGL did not start")
                path = out / f"first-screen-{w}x{h}.png"
                page.screenshot(path=str(path))
                print(f"{path.relative_to(DEMO.parent)}: {w}x{h}")
                page.close()
            browser.close()
    finally:
        server.shutdown()
    problems += [f"off-box request: {u}" for u in blocked]
    for p in problems:
        print(p, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
