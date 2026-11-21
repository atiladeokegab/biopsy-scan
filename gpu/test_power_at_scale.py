import importlib.util
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("modal", reason="needs the modal client: uv run --with modal pytest gpu/")
_spec = importlib.util.spec_from_file_location("gpu_power_at_scale", Path(__file__).with_name("power_at_scale.py"))
pas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pas)
pinned = subprocess.run(["git", "-C", str(pas.REPO), "cat-file", "-e", pas.PIN + "^{commit}"]).returncode == 0


@pytest.mark.skipif(not pinned, reason="the pinned commit is not in this clone")
def test_staging_is_fresh_and_verified(tmp_path):
    src = pas.stage(tmp_path / "a")
    assert (src / "pyproject.toml").exists() and not (src / "atlas").exists()
    (src / "core" / "truth.py").write_text("tampered\n")
    with pytest.raises(RuntimeError, match="core/truth.py does not match"):
        pas.verify(src)
    (src / "core" / "extra.py").write_text("")
    with pytest.raises(RuntimeError, match="staged files differ"):
        pas.verify(src)


def test_snapshot_upload_is_the_committed_seven_files_only(tmp_path):
    stray = pas.SNAPSHOT / "untracked_stray.py"
    stray.write_text("print('not committed')\n")
    try:
        staged = pas.stage_snapshot(tmp_path / "snap")
    finally:
        stray.unlink()
    assert sorted(f.name for f in staged.iterdir()) == sorted(n.split("/")[1] for n in pas.snapshot_sha256())
    (staged / "runner.py").write_text("tampered\n")
    with pytest.raises(RuntimeError, match="snapshot"):
        pas.verify_snapshot(staged)
