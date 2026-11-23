"""Proof take of the 3D view on 1 x L4: gpu/fps_l4.py's own measure() and result(), unchanged, with Playwright
recording the timed 1280x720 context. Scratch recorder, not product code.

    cd <worktree> && timeout 900 uv run --with modal modal run <this file> --take DIR
"""
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

WT = Path(__file__).resolve().parents[3]  # repo root
sys.path.insert(0, str(WT / "gpu"))
import fps_l4 as f  # noqa: E402

import modal  # noqa: E402

app = modal.App("preflight-proof-l4")
image = f.image.add_local_python_source("fps_l4")


@app.function(image=image, gpu="L4", cpu=4, memory=16384, timeout=900)
def record() -> dict:
    import glob

    from playwright.sync_api._generated import Browser, Page

    marks, win = {}, {}
    # Runs 2-3: the screencast showed 633 px of page over an 87 px grey band (--window-size=1280,807 did not change it).
    # Grow the recorded tab's window by the band before the take, over CDP, and record the bounds before and after.
    new_context, evaluate = Browser.new_context, Page.evaluate

    def ctx_with_video(self, **kw):
        if "video" not in marks and kw.get("viewport") == {"width": 1280, "height": 720} and not kw.get("is_mobile"):
            marks["video"] = time.monotonic()  # the timed take's context: the only one recorded
            kw.update(record_video_dir="/tmp/vid", record_video_size={"width": 1280, "height": 720})
        return new_context(self, **kw)

    def timed_evaluate(self, expr, *a, **kw):
        if "requestAnimationFrame(f)" in expr:
            win["sizes"] = evaluate(self, "[outerWidth, outerHeight, innerWidth, innerHeight]")
            cdp = self.context.new_cdp_session(self)
            w = cdp.send("Browser.getWindowForTarget")
            win["before"] = w["bounds"]
            cdp.send("Browser.setWindowBounds", {"windowId": w["windowId"], "bounds": {"width": 1280, "height": w["bounds"]["height"] + 87}})
            win["after"] = cdp.send("Browser.getWindowBounds", {"windowId": w["windowId"]})["bounds"]
            self.wait_for_timeout(500)
            marks["take_start"] = time.monotonic()
            r = evaluate(self, expr, *a, **kw)
            marks["take_end"] = time.monotonic()
            # Runs 4-5: after the resize the screencast lags the page by about 2 s (run 5's webm ran 22.2 s for
            # about 20.4 s of wall, and Pause was still drawn at its end). Hold 6 s so the video reaches the ended
            # scene. The sample is already taken, so the cadence is unaffected.
            self.wait_for_timeout(6000)
            return r
        return evaluate(self, expr, *a, **kw)

    Browser.new_context, Page.evaluate = ctx_with_video, timed_evaluate
    out = f.measure.local()
    vids = glob.glob("/tmp/vid/*.webm")
    out["video_webm"] = Path(vids[0]).read_bytes() if vids else None
    out["window"] = win
    out["video_marks_s"] = {k: round(v - marks["video"], 2) for k, v in marks.items() if "video" in marks}
    return out


@app.local_entrypoint()
def main(take: str):
    dest = Path(take)
    dest.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now(timezone.utc)
    out = record.remote()
    wall = (datetime.now(timezone.utc) - t0).total_seconds()
    if "renderer" not in out:
        raise SystemExit(f"no NVIDIA WebGL renderer: {out['tried']}")
    stamp = t0.strftime("%Y%m%dT%H%M%SZ")
    webm, marks = out.pop("video_webm"), out.pop("video_marks_s")
    for k in ("stills", "end_jpg_1280x720", "end_jpg_390x844"):
        out.pop(k)
    res = f.result(out, out.pop("timestamps_ms"), wall, stamp)
    res["recorded"] = "Playwright screencast of this timed context; the cadence is with recording on"
    (dest / f"fps-l4-{stamp}.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "tried"}, indent=1))
    if webm is None:
        raise SystemExit("no video recorded")
    (dest / "take.webm").write_bytes(webm)
    # Cut from the scroll to the 3D view (the take's clock start) to the end of the recording (the 6 s hold).
    ss = marks["take_start"]
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{ss:.2f}", "-i", str(dest / "take.webm"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", str(dest / "proof-3d.mp4")], check=True)
    (dest / "marks.json").write_text(json.dumps({"stamp": stamp, "wall_s": round(wall, 1), **marks}, indent=1) + "\n")
    # Framecheck's failing item: no grey band. The bottom 80 rows of the page here are its dark background.
    mid = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", "8", "-i", str(dest / "proof-3d.mp4"), "-frames:v", "1", "-vf",
                          "crop=1280:80:0:640,format=gray", "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    print("bottom-80-row mean luma", round(sum(mid) / len(mid), 1), "window", res.get("window"))
    if sum(mid) / len(mid) > 60:
        raise SystemExit("grey band at the bottom of the take")
    print("wrote", dest, "status", res["status"])
    if res["status"] != "MEASURED":
        raise SystemExit(f"FAILED: {res['reason']}")
