# proof-3d-e (00841e4): re-record after framecheck FAIL on proof-3d

- Page: integration 00841e4 (#131 merged); gpu/ and demo/ at 00841e4. Recorded on 1 x Modal L4, headless Chrome, ANGLE/Vulkan on the NVIDIA L4.
- Recorder: recorder-rec_l4.py (scratch, not in the repo). It runs gpu/fps_l4.py's own measure() and result() unchanged, with Playwright video on the timed 1280x720 context.
- The fix for proof-3d's one failing item (an 87 px grey band at y=633..719 in every frame): before the take, the recorded tab's window is grown by 87 px over CDP (Browser.setWindowBounds 720 -> 807 high). The page's viewport stays 1280x720 (innerWidth/innerHeight 1280x720). --window-size=1280,807 alone did not work (run 3). The recorder now refuses a take whose bottom 80 rows are not dark (run 2's take: mean luma 128; this take: 40.8 to 41.3 in every frame).
- After the resize, the screencast lags the page by about 2 s, so a 2.5 s hold still ended on Pause drawn (run 5). This take holds 6 s after the sample. Pause disappears between 16 and 17 s, and Replay alone holds to the end.
- proof-3d.mp4: 1280x720, 25 fps (the screencast's rate), 23.32 s, cut from the scroll to the 3D view (the take's clock start) to the end of the recording.
- The bottom 87 px now show what a 1280x720 visitor sees under the card: the tops of the next three cards.
- fps line (fps-l4-20261122T092716Z.json): MEASURED, observed frame cadence on an L4, headless, vsync-capped, with video capture on. Median 59.9 fps, p5 59.5 fps, 924 frames over 15.5 s, last_step 3, end controls [reimage-replay], no errors. Not the cited figure; run 8 stands.
- Not frame-checked by the recorder; Hephaestus checks it.
