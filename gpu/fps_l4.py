"""Frame rate and stills of the demo's 3D re-imaging view on a real NVIDIA L4 (Modal), headless Chrome.

    uv run modal run gpu/fps_l4.py             # writes demo/shots/l4/ and bench/results/fps-l4-<utc>.json

Chrome must render on the NVIDIA GPU, not a software rasteriser: each launch flag set is tried and the first whose
WebGL renderer names NVIDIA is kept; with none, the run fails rather than time SwiftShader. Frames are timed with
requestAnimationFrame over the scene's full script (fly-in, windows, flicker: 0 to 15 s) at 1280x720, while the
rest of the page (the race's idle state and the six-core tissue view) runs as it does for a visitor.
A data-centre L4 is not a laptop GPU: the number says the scene holds on real GPU hardware, not on any one laptop.
Headless Chrome is vsync-capped at 60 Hz, so the figure is the observed frame cadence, not throughput. A broken scene
(no step reached, any page or console error, a short or stalled sample) is written as FAILED with its reason, never MEASURED.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import modal

DEMO = Path(__file__).resolve().parents[1] / "demo"
app = modal.App("preflight-fps-l4")
image = (
    modal.Image.from_registry("mcr.microsoft.com/playwright/python:v1.48.0-jammy")
    .apt_install("libvulkan1", "vulkan-tools", "libegl1", "libgl1", "libgles2")
    .pip_install("playwright==1.48.0")
    .add_local_dir(str(DEMO), "/demo", ignore=["shots", "*.py", "*.mjs"])
)

ATTEMPTS = [
    ["--use-angle=vulkan", "--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan", "--disable-vulkan-surface"],
    ["--use-gl=angle", "--use-angle=gl-egl"],
    ["--use-gl=egl"],
]
BASE = ["--headless=new", "--ignore-gpu-blocklist", "--enable-gpu", "--no-sandbox", "--disable-dev-shm-usage",
        "--disable-background-timer-throttling", "--disable-renderer-backgrounding"]
RENDERER = """() => { const g = document.createElement('canvas').getContext('webgl2');
  const e = g && g.getExtension('WEBGL_debug_renderer_info'); return g ? (e ? g.getParameter(e.UNMASKED_RENDERER_WEBGL) : g.getParameter(g.RENDERER)) : 'no webgl' }"""
DURATION_MS = 15500  # the declared sample: the whole 15 s take plus half a second
MIN_FRAMES = 300  # under about 20 fps over the take the scene has stalled: no cadence is reported
LAST_STEP = 3  # the take's fourth and last step
# rAF timestamps from the moment the section is on screen (the scene's clock starts there) to DURATION_MS later.
TIMED = """() => new Promise((done) => { const ts = []; const t0 = performance.now();
  document.querySelector('.reimage').scrollIntoView({ block: 'start' });
  const f = (t) => { ts.push(t); if (t - t0 < DURATION_MS) requestAnimationFrame(f); else done(ts); }; requestAnimationFrame(f); })"""


@app.function(image=image, gpu="L4", cpu=4, memory=16384, timeout=900)
def measure() -> dict:
    import glob
    import os
    import subprocess
    import time

    from playwright.sync_api import sync_playwright

    # NVIDIA's EGL and Vulkan ICDs, so ANGLE and EGL can find the driver the runtime mounts.
    lib = next((os.path.dirname(p) for p in glob.glob("/usr/**/libEGL_nvidia.so*", recursive=True)), None)
    # /usr/share is read-only on Modal: write the ICDs to /tmp and point the loaders at them.
    icd = {k: glob.glob(f"{d}/*") for k, d in (("egl", "/usr/share/glvnd/egl_vendor.d"), ("vk", "/etc/vulkan/icd.d"),
                                                ("vk_usr", "/usr/share/vulkan/icd.d"))}
    os.makedirs("/tmp/icd", exist_ok=True)
    Path("/tmp/icd/10_nvidia.json").write_text(
        '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}')
    Path("/tmp/icd/nvidia_icd.json").write_text(
        '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.0"}}')
    os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = "/tmp/icd/10_nvidia.json"
    os.environ["VK_ICD_FILENAMES"] = "/tmp/icd/nvidia_icd.json"
    if lib:
        os.environ["LD_LIBRARY_PATH"] = f"{lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    server = subprocess.Popen(["python3", "-m", "http.server", "8000", "-d", "/demo"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)
    out = {"gpu": smi, "nvidia_lib_dir": lib, "icd_present": icd, "tried": []}
    with sync_playwright() as p:
        browser = None
        for extra in ATTEMPTS:
            b = p.chromium.launch(headless=False, args=BASE + extra)
            pg = b.new_page()
            r = pg.evaluate(RENDERER)
            pg.close()
            out["tried"].append([extra, r])
            if "NVIDIA" in r.upper():
                browser, out["renderer"], out["flags"] = b, r, BASE + extra
                break
            b.close()
        if browser is None:
            server.kill()
            return out  # no GPU renderer: the caller fails the run
        errors = []
        ctx = browser.new_context(viewport={"width": 1280, "height": 720}, device_scale_factor=1)
        pg = ctx.new_page()
        pg.on("pageerror", lambda e: errors.append(str(e)))
        # Console errors, and the warning the page logs when the 3D view falls back (WebGL failed).
        pg.on("console", lambda m: (m.type == "error" or "3D view off" in m.text) and errors.append(f"console: {m.text} {m.location.get('url', '')}".strip()))
        pg.goto("http://localhost:8000/", wait_until="networkidle")
        pg.wait_for_timeout(1000)
        out["timestamps_ms"] = pg.evaluate(TIMED.replace("DURATION_MS", str(DURATION_MS)))
        # The take reached its last step, and the controls appeared because WebGL started.
        out["last_step"] = pg.evaluate("[...document.querySelectorAll('#reimage-steps li')].findIndex((li) => li.hasAttribute('aria-current'))")
        # At the end of the take only Replay is offered (the scene stops, so there is nothing to pause).
        out["end_controls"] = pg.evaluate("[...document.querySelectorAll('#reimage-replay, #reimage-pause')].filter((b) => !b.hidden).map((b) => b.id)")
        out["errors"] = errors
        ctx.close()
        # Stills on the GPU at each camera height, both sizes.
        out["stills"] = {}
        for name, vp, dpr in (("1280x720", (1280, 720), 1), ("390x844", (390, 844), 2)):
            # The phone context is a mobile device (overlay scrollbars, touch), as a real phone would be.
            ctx = browser.new_context(viewport={"width": vp[0], "height": vp[1]}, device_scale_factor=dpr, is_mobile=dpr > 1, has_touch=dpr > 1)
            for t in (0, 2.6, 5.4, 9.8, 12.6, 13, 15):
                pg = ctx.new_page()
                pg.goto(f"http://localhost:8000/?t3d={t}", wait_until="networkidle")
                pg.evaluate("document.querySelector('.reimage').scrollIntoView({block: 'start'})")
                pg.wait_for_timeout(1500)
                out["stills"][f"{name}-t{t}"] = pg.screenshot()
                if t == 15:  # the no-WebGL fallback's still at each size: the canvas alone, last frame
                    b = pg.locator("#reimage").bounding_box()  # 2 px off the top: the card's border sits on that edge
                    out[f"end_jpg_{name}"] = pg.screenshot(type="jpeg", quality=82, clip={"x": b["x"], "y": b["y"] + 2, "width": b["width"], "height": b["height"] - 2})
                pg.close()
            ctx.close()
        browser.close()
    server.kill()
    return out


def summary(ts: list[float]) -> dict:
    """Frame intervals over the whole script -> median fps and 5th-percentile fps (the slow tail)."""
    d = sorted(b - a for a, b in zip(ts, ts[1:]))
    q = lambda f: d[min(len(d) - 1, int(f * len(d)))]  # noqa: E731
    return {"frames": len(d), "span_s": round((ts[-1] - ts[0]) / 1000, 2),
            "median_fps": round(1000 / q(0.5), 1), "p5_fps": round(1000 / q(0.95), 1),
            "median_frame_ms": round(q(0.5), 2), "p95_frame_ms": round(q(0.95), 2), "max_frame_ms": round(d[-1], 2)}


def result(out: dict, ts: list[float], wall: float, stamp: str) -> dict:
    """The run's record: MEASURED cadence only from a whole, working take; otherwise FAILED with the reason."""
    span = (ts[-1] - ts[0]) if len(ts) > 1 else 0.0
    why = [w for bad, w in (
        (out.get("last_step", -1) < LAST_STEP, f"the take did not reach its last step (last_step {out.get('last_step')})"),
        (bool(out.get("errors")), f"page or console error: {out.get('errors')}"),
        (span < DURATION_MS - 100, f"sample shorter than the declared duration ({span:.0f} of {DURATION_MS} ms)"),
        (len(ts) - 1 < MIN_FRAMES, f"too few frames ({max(len(ts) - 1, 0)} of at least {MIN_FRAMES})"),
        (out.get("end_controls") != ["reimage-replay"], f"take did not end: end controls were {out.get('end_controls')}"),
    ) if bad]
    base = {"utc": stamp, "modal_wall_s": round(wall, 1), "declared_ms": DURATION_MS, "min_frames": MIN_FRAMES, **out}
    if why:
        return {"status": "FAILED", "reason": "; ".join(why), **base}
    return {"status": "MEASURED",
            "label": "observed frame cadence on an L4, headless, vsync-capped (MEASURED); a data-centre L4, not a laptop GPU",
            "what": "rAF frame intervals over the 3D re-imaging script (fly-in, windows, flicker), 1280x720, whole page live",
            **base, **summary(ts)}


@app.local_entrypoint()
def main():
    t0 = datetime.now(timezone.utc)
    out = measure.remote()
    wall = (datetime.now(timezone.utc) - t0).total_seconds()
    if "renderer" not in out:
        raise SystemExit(f"no NVIDIA WebGL renderer: {out['tried']}")
    stamp = t0.strftime("%Y%m%dT%H%M%SZ")
    stills, ends = out.pop("stills"), {k: out.pop(k) for k in ("end_jpg_1280x720", "end_jpg_390x844")}
    res = result(out, out.pop("timestamps_ms"), wall, stamp)
    path = DEMO.parent / "bench" / "results" / f"fps-l4-{stamp}.json"
    path.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "tried"}, indent=1))
    print("wrote", path)
    if res["status"] != "MEASURED":
        raise SystemExit(f"FAILED: {res['reason']}")  # a broken scene keeps the committed stills
    shots = DEMO / "shots" / "l4"
    shots.mkdir(parents=True, exist_ok=True)
    for k, png in stills.items():
        (shots / f"reimage3d-{k}.png").write_bytes(png)
    (DEMO / "img").mkdir(exist_ok=True)
    (DEMO / "img" / "reimage-end.jpg").write_bytes(ends["end_jpg_1280x720"])
    (DEMO / "img" / "reimage-end-phone.jpg").write_bytes(ends["end_jpg_390x844"])  # framed for the phone, not the desktop frame shrunk
