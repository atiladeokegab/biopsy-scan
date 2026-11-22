"""Frame rate and stills of the demo's 3D re-imaging view on a real NVIDIA L4 (Modal), headless Chrome.

    uv run modal run gpu/fps_l4.py             # writes demo/shots/l4/ and bench/results/fps-l4-<utc>.json

Chrome must render on the NVIDIA GPU, not a software rasteriser: each launch flag set is tried and the first whose
WebGL renderer names NVIDIA is kept; with none, the run fails rather than time SwiftShader. Frames are timed with
requestAnimationFrame over the scene's full script (fly-in, windows, flicker: 0 to 15 s) at 1280x720, while the
rest of the page (the race's idle state and the six-core tissue view) runs as it does for a visitor.
A data-centre L4 is not a laptop GPU: the number says the scene holds on real GPU hardware, not on any one laptop.
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
# rAF timestamps from the moment the section is on screen (the scene's clock starts there) to 15.5 s later.
TIMED = """() => new Promise((done) => { const ts = []; const t0 = performance.now();
  document.querySelector('.reimage').scrollIntoView({ block: 'start' });
  const f = (t) => { ts.push(t); if (t - t0 < 15500) requestAnimationFrame(f); else done(ts); }; requestAnimationFrame(f); })"""


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
        pg.goto("http://localhost:8000/", wait_until="networkidle")
        pg.wait_for_timeout(1000)
        out["timestamps_ms"] = pg.evaluate(TIMED)
        # The take reached its last step, and the controls appeared because WebGL started.
        out["last_step"] = pg.evaluate("[...document.querySelectorAll('#reimage-steps li')].findIndex((li) => li.hasAttribute('aria-current'))")
        out["controls_shown"] = pg.evaluate("!document.getElementById('reimage-pause').hidden && !document.getElementById('reimage').hidden")
        out["errors"] = errors
        ctx.close()
        # Stills on the GPU at each camera height, both sizes.
        out["stills"] = {}
        for name, vp, dpr in (("1280x720", (1280, 720), 1), ("390x844", (390, 844), 2)):
            ctx = browser.new_context(viewport={"width": vp[0], "height": vp[1]}, device_scale_factor=dpr)
            for t in (0, 2.6, 5.4, 9.8, 12.6, 13, 15):
                pg = ctx.new_page()
                pg.goto(f"http://localhost:8000/?t3d={t}", wait_until="networkidle")
                pg.evaluate("document.querySelector('.reimage').scrollIntoView({block: 'start'})")
                pg.wait_for_timeout(1500)
                out["stills"][f"{name}-t{t}"] = pg.screenshot()
                if name == "1280x720" and t == 15:  # the no-WebGL fallback's still: the canvas alone, last frame
                    b = pg.locator("#reimage").bounding_box()  # 2 px off the top: the card's border sits on that edge
                    out["end_jpg"] = pg.screenshot(type="jpeg", quality=82, clip={"x": b["x"], "y": b["y"] + 2, "width": b["width"], "height": b["height"] - 2})
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


@app.local_entrypoint()
def main():
    t0 = datetime.now(timezone.utc)
    out = measure.remote()
    wall = (datetime.now(timezone.utc) - t0).total_seconds()
    if "renderer" not in out:
        raise SystemExit(f"no NVIDIA WebGL renderer: {out['tried']}")
    stamp = t0.strftime("%Y%m%dT%H%M%SZ")
    shots = DEMO / "shots" / "l4"
    shots.mkdir(parents=True, exist_ok=True)
    for k, png in out.pop("stills").items():
        (shots / f"reimage3d-{k}.png").write_bytes(png)
    (DEMO / "img").mkdir(exist_ok=True)
    (DEMO / "img" / "reimage-end.jpg").write_bytes(out.pop("end_jpg"))
    ts = out.pop("timestamps_ms")
    res = {"status": "MEASURED", "label": "MEASURED (L4, headless Chrome); a data-centre L4, not a laptop GPU",
           "what": "rAF frame intervals over the 3D re-imaging script (fly-in, windows, flicker), 1280x720, whole page live",
           "utc": stamp, "modal_wall_s": round(wall, 1), **out, **summary(ts)}
    path = DEMO.parent / "bench" / "results" / f"fps-l4-{stamp}.json"
    path.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "tried"}, indent=1))
    print("wrote", path)
