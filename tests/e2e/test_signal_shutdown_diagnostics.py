"""Real children exercise shutdown evidence without deliberately crashing Python."""

from __future__ import annotations

import fnmatch
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.e2e.signal_shutdown_diagnostics import CapturedChild


@pytest.mark.parametrize("code", [0, 7])
def test_shutdown_capture_preserves_only_abnormal_evidence(tmp_path: Path, code: int) -> None:
    config = tmp_path / "config"
    artifacts = tmp_path / "artifacts"
    child = CapturedChild(
        [
            sys.executable,
            "-c",
            f"import sys; sys.stderr.write('shutdown witness\\n'); sys.exit({code})",
        ],
        config=config,
    )
    assert child.wait(timeout=10) == code
    stall = config / "logs" / f"runtime-stall-{child.pid}.log"
    stall_body = "stall header\n" + "frame\n" * 1000 + "stall witness\n"
    stall.write_text(stall_body)
    evidence = child.shutdown_evidence(artifacts)
    assert child.stderr_path.is_file(), "owned child's stderr witness was discarded"
    assert child.stderr_path.read_text() == "shutdown witness\n"
    if code == 0:
        assert evidence == ""
        assert not artifacts.exists()
    else:
        assert "shutdown witness" in evidence
        assert "stall witness" in evidence
        payload = json.loads(next(artifacts.glob("*.json")).read_text())
        assert payload["pid"] == child.pid
        assert payload["returncode"] == code
        assert payload["stderr_tail"] == "shutdown witness\n"
        assert payload["stall_tail"].endswith("stall witness\n")
        assert "stall header" not in payload["stall_tail"]
        attachments = artifacts / "attachments" / str(child.pid)
        assert (attachments / child.stderr_path.name).read_text() == "shutdown witness\n"
        assert (attachments / stall.name).read_text() == stall_body
        print(evidence)


def test_shutdown_capture_is_bounded_and_missing_stall_is_explicit(tmp_path: Path) -> None:
    child = CapturedChild(
        [
            sys.executable,
            "-c",
            "import sys; sys.stderr.write('x' * 200000 + 'FINAL WITNESS'); sys.exit(9)",
        ],
        config=tmp_path,
    )
    assert child.wait(timeout=10) == 9
    evidence = child.shutdown_evidence(tmp_path / "artifacts")
    assert child.stderr_path.stat().st_size <= child.STDERR_BYTES
    assert child.stderr_path.read_text().endswith("FINAL WITNESS")
    assert "FINAL WITNESS" in evidence
    assert "unavailable" in evidence
    assert next((tmp_path / "artifacts").glob("*.json")).stat().st_size < 96 * 1024


def test_default_shutdown_artifacts_match_ci_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise default staging, not just a custom artifact directory in tests."""
    workflow = Path(__file__).resolve().parents[2] / ".github/workflows/full-suite.yml"
    upload = workflow.read_text().split("name: junit-timings-", 1)[1]
    upload = upload.split("retention-days:", 1)[0]
    globs = [line.strip() for line in upload.splitlines() if "test-artifacts/" in line]
    monkeypatch.chdir(tmp_path)
    child = CapturedChild(
        [sys.executable, "-c", "import sys; sys.stderr.write('CI witness'); sys.exit(7)"],
        config=tmp_path / "config",
    )
    assert child.wait(timeout=10) == 7
    stall = child.config / "logs" / f"runtime-stall-{child.pid}.log"
    stall.write_text("complete CI stall witness")
    evidence = child.shutdown_evidence()
    files = sorted(path for path in Path("test-artifacts").rglob("*") if path.is_file())
    assert len(files) == 3
    assert all(any(fnmatch.fnmatch(path.as_posix(), glob) for glob in globs) for path in files)
    assert any(path.name == stall.name for path in files)
    assert "CI witness" in evidence
    print("CI upload matches:", ", ".join(str(path) for path in files))


def test_without_capture_negative_control_loses_stderr(tmp_path: Path) -> None:
    """The old DEVNULL wiring cannot supply the discriminator's witness."""
    child = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stderr.write('shutdown witness'); sys.exit(7)"],
        stderr=subprocess.DEVNULL,
    )
    assert child.wait(timeout=10) == 7
    assert child.stderr is None
    assert not list(tmp_path.rglob("*.stderr"))
