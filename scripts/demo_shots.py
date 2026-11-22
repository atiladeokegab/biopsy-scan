"""Screenshot the demo's first screen at desktop and phone size, or record the backup demo video.

    uv run --frozen --with playwright==1.64.0 python -I scripts/demo_shots.py
    uv run --frozen --with playwright==1.64.0 python -I scripts/demo_shots.py --video --out DIR

Serves demo/ on 127.0.0.1, blocks every off-box request, waits for the page to finish its first render,
and writes demo/shots/first-screen-<w>x<h>.png. Exits 1 if the page shows an error or anything goes off-box.

--video records 1280x720 from the moment the page is ready: the first screen for --hold seconds, then "Run the studies", until every timed race
lane is done (--race-scroll scrolls one element into view as the race starts), plus --tail seconds (default 10, so the 30 s still exists); then each --scroll target, smoothly
scrolled into view and held 5 s. --cursor draws a pointer that glides to each click target first, so viewers
see what is clicked. It writes demo.mp4, stills at 0, 10, 20 and 30 s cut from that file, and timeline.json
to --out (keep video out of git). Needs ffmpeg and ffprobe on PATH.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


DEMO = Path(__file__).resolve().parents[1] / "demo"
SIZES = [(1280, 720), (390, 844)]
READY = """() => {
  const t = document.getElementById('verdict').textContent, c = document.getElementById('tissue-caption').textContent;
  return t && t !== 'Evaluating design...' && c && c !== 'Preparing tissue slabs...';
}"""
RACE_DONE = """() => {
  const lanes = [...document.querySelectorAll('#speed-race li, #speed-interim li')].filter((li) => !li.classList.contains('pending'));
  return lanes.length > 0 && lanes.every((li) => li.classList.contains('done'));
}"""
STILLS_S = (0, 10, 20, 30)
GLIDE_MS = 900
# A drawn pointer (headless Chromium shows none), eased on requestAnimationFrame so it moves smoothly however
# slow the round trips are. Arg: [x, y, ms]; the first call draws it at (x, y) when ms is 0.
CURSOR_JS = """([x, y, ms]) => new Promise((done) => {
  let c = document.getElementById('shots-cursor');
  if (!c) {
    c = document.createElement('div');
    c.id = 'shots-cursor';
    c.dataset.x = x; c.dataset.y = y;
    c.style.cssText = 'position:fixed;left:0;top:0;z-index:2147483647;pointer-events:none';
    c.innerHTML = '<svg width="22" height="30" viewBox="0 0 22 30"><path d="M2 2 L2 24 L8 18 L12 28 L16 26 L12 16 L20 16 Z" '
      + 'fill="#fff" stroke="#000" stroke-width="1.6" stroke-linejoin="round"/></svg>';
    document.body.appendChild(c);
  }
  const x0 = +c.dataset.x, y0 = +c.dataset.y, t0 = performance.now();
  const ease = (u) => (u < 0.5 ? 2 * u * u : 1 - (-2 * u + 2) ** 2 / 2);
  const f = (now) => {
    const u = ms > 0 ? Math.min(1, (now - t0) / ms) : 1, e = ease(u);
    const cx = x0 + (x - x0) * e, cy = y0 + (y - y0) * e;
    c.style.transform = `translate(${cx - 2}px, ${cy - 2}px)`;
    c.dataset.x = cx; c.dataset.y = cy;
    if (u < 1) requestAnimationFrame(f); else done();
  };
  requestAnimationFrame(f);
})"""
SCROLL_DWELL_S = 5.0


class _Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def _page_problems(page, tag: str) -> list[str]:
    out = []
    if page.locator("#error").is_visible():
        out.append(f"{tag}: error shown: {page.locator('#error').inner_text()}")
    if "could not start WebGL" in page.locator("#tissue-caption").inner_text():
        out.append(f"{tag}: WebGL did not start")
    return out


def _shots(browser, url: str, out: Path, route) -> list[str]:
    problems = []
    for w, h in SIZES:
        page = browser.new_page(viewport={"width": w, "height": h}, reduced_motion="reduce")
        page.route("**/*", route)
        page.goto(url, wait_until="networkidle")
        page.wait_for_function(READY, timeout=30_000)
        page.wait_for_timeout(500)  # one more frame for the WebGL canvas
        problems += _page_problems(page, f"{w}x{h}")
        path = out / f"first-screen-{w}x{h}.png"
        page.screenshot(path=str(path))
        print(f"{path}: {w}x{h}")
        page.close()
    return problems


def _hold(page, seconds: float) -> None:
    # The recording only gets a frame when the page paints, and it dropped the race's final repaint (MEASURED:
    # 2.87 s on video, 2.88 s in the DOM); a screenshot forces a fresh frame, so take one each half second.
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        page.screenshot()
        page.wait_for_timeout(500)


def _scroll(page, selectors: list[str], dwell_s: float, block: str = "start") -> list[str]:
    problems = []
    for sel in selectors:
        found = page.evaluate("([s, b]) => { const e = document.querySelector(s); "
                              "if (e) e.scrollIntoView({behavior: 'smooth', block: b}); return !!e; }", [sel, block])
        if not found:
            problems.append(f"scroll target not found: {sel}")
            continue
        _hold(page, dwell_s)
    return problems


def _click(page, selector: str, cursor: bool) -> None:
    target = page.locator(selector)
    if cursor:
        b = target.bounding_box()
        if b is None:  # hidden or detached: there is nowhere to glide to
            raise SystemExit(f"click target not visible: {selector}")
        page.evaluate(CURSOR_JS, [round(b["x"] + b["width"] / 2), round(b["y"] + b["height"] / 2), GLIDE_MS])
        page.wait_for_timeout(300)  # let the eye land before the click
    target.click()


def _video(browser, url: str, out: Path, route, hold_s: float, tail_s: float, scroll: list[str], cursor: bool,
           race_scroll: str | None = None) -> list[str]:
    w, h = SIZES[0]
    with tempfile.TemporaryDirectory() as tmp:
        ctx = browser.new_context(viewport={"width": w, "height": h}, record_video_dir=tmp,
                                  record_video_size={"width": w, "height": h})
        page = ctx.new_page()
        t0 = time.monotonic()  # the recording starts with the page
        page.route("**/*", route)
        page.goto(url, wait_until="networkidle")
        page.wait_for_function(READY, timeout=30_000)
        ready_s = time.monotonic() - t0  # the video is cut to start here, so it never opens on a blank page
        problems = _page_problems(page, "video")
        if cursor:
            page.evaluate(CURSOR_JS, [w // 2, round(h * 0.6), 0])
        page.wait_for_timeout(hold_s * 1000)
        _click(page, "#run", cursor)
        click_s = time.monotonic() - t0 - ready_s
        if race_scroll:  # bottom-aligned: the last lane's caption ends at the frame's edge; if the target is taller
            problems += _scroll(page, [race_scroll], 0, block="end")  # than 720 px, its top text is what scrolls off
        page.wait_for_function(RACE_DONE, timeout=60_000)
        done_s = time.monotonic() - t0 - ready_s
        _hold(page, tail_s)
        if cursor:  # nothing more is clicked, and a resting pointer would cover labels in the scroll views
            page.evaluate("document.getElementById('shots-cursor')?.remove()")
        problems += _scroll(page, scroll, SCROLL_DWELL_S)
        problems += _page_problems(page, "video, after the race")
        webm = Path(page.video.path())
        ctx.close()  # finalises the recording
        mp4 = out / "demo.mp4"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{ready_s:.2f}", "-i", str(webm), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)], check=True)
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                str(mp4)], check=True, capture_output=True, text=True).stdout)
    stills = []
    for s in STILLS_S:
        if s >= dur:
            problems.append(f"no still at {s} s: the video is {dur:.1f} s")
            continue
        png = out / f"still-{s:02d}s.png"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(s), "-i", str(mp4), "-frames:v", "1",
                        str(png)], check=True)
        stills.append(png.name)
    timeline = {"source": "MEASURED", "video": mp4.name, "width": w, "height": h, "duration_s": round(dur, 2),
                "load_cut_s": round(ready_s, 2), "run_clicked_s": round(click_s, 2), "race_done_s": round(done_s, 2), "stills": stills,
                "sha256": hashlib.sha256(mp4.read_bytes()).hexdigest()}
    (out / "timeline.json").write_text(json.dumps(timeline, indent=1) + "\n")
    print(json.dumps(timeline))
    return problems


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--demo", type=Path, default=DEMO, help="the demo folder to serve (default: this repo's demo/)")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default: <demo>/shots)")
    ap.add_argument("--video", action="store_true", help="record the backup video instead of the first-screen shots")
    ap.add_argument("--hold", type=float, default=10.0, help="seconds on the first screen before Run (video only)")
    ap.add_argument("--tail", type=float, default=10.0, help="seconds after the race finishes (video only)")
    ap.add_argument("--scroll", action="append", default=[], metavar="SELECTOR",
                    help="after the tail, scroll this element into view and hold it (video only; repeatable, in order)")
    ap.add_argument("--cursor", action="store_true",
                    help="draw a pointer that glides to each click target before clicking (video only)")
    ap.add_argument("--race-scroll", metavar="SELECTOR",
                    help="right after the click, smoothly scroll this element's bottom to the frame's bottom for the race (video only)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    from playwright.sync_api import sync_playwright  # here, so tests import this module without playwright

    a = parse_args(argv)
    out = a.out or a.demo / "shots"
    out.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_Quiet, directory=str(a.demo)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    blocked, problems = [], []

    def route(r):
        if urlsplit(r.request.url).hostname in ("127.0.0.1", "localhost"):
            r.continue_()
        else:
            blocked.append(r.request.url)
            r.abort()

    url = f"http://127.0.0.1:{server.server_port}/index.html"
    try:
        with sync_playwright() as pw:
            # Software WebGL lets the 3D tissue render with no GPU, but with it the screencast that records video
            # froze the race panel (MEASURED: the bars never filled in the recording), so video mode goes without.
            args = [] if a.video else ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
            browser = pw.chromium.launch(args=args)
            problems += _video(browser, url, out, route, a.hold, a.tail, a.scroll, a.cursor, a.race_scroll) if a.video else _shots(browser, url, out, route)
            browser.close()
    finally:
        server.shutdown()
    problems += [f"off-box request: {u}" for u in blocked]
    for p in problems:
        print(p, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
