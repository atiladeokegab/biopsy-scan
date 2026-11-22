# proof-3d (00841e4)

- Page: integration 00841e4 (#131 merged), demo/ as served; recorded on 1 x Modal L4, headless Chrome, ANGLE/Vulkan on the NVIDIA L4.
- Recorder: scratch wrapper around gpu/fps_l4.py at 00841e4. It runs that file's own measure() and result() unchanged, with Playwright video on the timed 1280x720 context.
- proof-3d.mp4: 1280x720, 25 fps (Playwright screencast rate), 17.32 s, cut from the scroll to the 3D view (take start) to the end of the recording, about 1.8 s after the 15.5 s sample. take.webm is the uncut recording (19.12 s).
- fps line (fps-l4-20261122T080914Z.json): MEASURED, observed frame cadence on an L4, headless, vsync-capped, with video capture on. Median 59.9 fps, p5 59.5 fps, 924 frames over 15.5 s, last_step 3, end controls [reimage-replay], no errors. Not the cited figure; run 8 stands.
- Not frame-checked by the recorder; Hephaestus checks it.
