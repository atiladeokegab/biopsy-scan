# Atlas's frozen #53 snapshot

These 7 files are Atlas's atlas code exactly as frozen at 2026-11-21T12:56:20Z (branch `5-study-atlas`,
base commit `ab0ae72` plus its uncommitted working tree), byte-for-byte unchanged. They produced
`bench/results/vs-schoenfeld-300.json` (#53), whose `simulated.atlas_code.sha256_12` records them, and
`gpu/power_at_scale.py` runs them (with the repo at `1f1bc74` for `core/`, `reimage/`, `stats/`) for
`bench/results/power-at-scale.json`. The container refuses to run if any hash differs. Their own tests are
not collected here (`conftest.py`): they belong to that frozen tree, not to integration.

```
fc5b891418e40482a36aa86658a00b1d9f4771edc6935fd3144bd0f74bbd7818  atlas/__init__.py
f4ad4e3ecb9b71595c8cdb3539acacc5108e1d49e10172b2b96feff8fd3b3bf7  atlas/__main__.py
1801872f7b8788370c275c49e43e039ab2a5fe1d02812cc00c5a0b756dffcacc  atlas/artifacts.py
c380b0abfeb2dab7f2d948238156b33d8cc21c09d063d4b36110b692dc964e2f  atlas/runner.py
b922388d90fff2e79e6aaa80d39eca5b45bccd2e03f73fee3da096517e38132f  atlas/simulation.py
2137b05d4f8d782c7de5fc8556e50edd3d5fc61915030e4b3c663615f90f7b1e  atlas/test_runner.py
f37574145a4ddb2f33ca1916e1743fec491e65eba48a22c18d44d27d1604cd49  atlas/test_simulation.py
```

Check: `cd bench/atlas_snapshot && sha256sum -c` with the block above.
