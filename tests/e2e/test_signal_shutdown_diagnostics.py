"""Real children exercise shutdown evidence without deliberately crashing Python."""

from __future__ import annotations

import fnmatch
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.e2e import signal_shutdown_diagnostics as diag
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
    jobs = yaml.safe_load(workflow.read_text())["jobs"]
    # Jobs have separate working trees: unit-job allowlists cannot preserve
    # files produced by the signal cases in tui-e2e, even if their names match.
    upload = next(
        step for step in jobs["tui-e2e"]["steps"] if step.get("name") == "Upload e2e junit timings"
    )
    assert upload["uses"].startswith("actions/upload-artifact@")
    assert upload["if"] == "always()"
    globs = upload["with"]["path"].splitlines()
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


# --- native macOS crash report: pure units over SYNTHETIC .ips text --------------
# Nothing below reads the real DiagnosticReports or crashes anything; the directory
# is a tmp dir injected into ``find_native_report``.

_PID = 4242
_SECRET = "ACCESS_TOKEN_MUST_NOT_LEAVE"


def _ips(pid: int = _PID, *, bug_type: str = "309", frames: int = 3, proc: str = "python3.12"):
    header = {"bug_type": bug_type, "incident_id": "INC-1", "timestamp": "2026-11-15 10:00:00"}
    body = {
        "pid": pid,
        "procName": proc,
        "exception": {"type": "EXC_BAD_ACCESS", "signal": "SIGSEGV", "subtype": "KERN_INVALID"},
        "faultingThread": 1,
        "registers_and_env": _SECRET,  # must never be copied out
        "usedImages": [{"name": "python3.12", "path": "/Users/runner/secret/python"}],
        "threads": [
            {"frames": []},
            {
                "frames": [
                    {"imageIndex": 0, "symbol": f"sym{n}-{pid}" + "x" * 300, "imageOffset": n}
                    for n in range(frames)
                ]
            },
        ],
    }
    return json.dumps(header) + "\n" + json.dumps(body)


def _find(directory: Path, started: float = 0.0, pid: int = _PID) -> dict[str, Any]:
    return diag.find_native_report(
        directory, pid=pid, started_wall=started, poll_budget_s=0.0, sleep=lambda _s: None
    )


def test_native_report_positive_is_allow_listed_and_bounded(tmp_path: Path) -> None:
    (tmp_path / "python3.12-2026.ips").write_text(_ips(frames=60))
    found = _find(tmp_path)
    blob = json.dumps(found, ensure_ascii=True)
    assert found["found"] is True
    assert found["signal"] == "SIGSEGV" and found["exception_type"] == "EXC_BAD_ACCESS"
    assert found["incident_id"] == "INC-1"
    assert found["frames_total"] == 60 and found["frames_kept"] <= diag.NATIVE_MAX_FRAMES
    assert found["faulting_thread_frames"][0]["image"] == "python3.12"
    assert len(blob) <= diag.NATIVE_MAX_OUTPUT_BYTES
    assert _SECRET not in blob and "/Users/runner" not in blob


@pytest.mark.parametrize(
    "case",
    ["wrong-pid", "stale", "torn", "oversize", "not-a-crash", "other-process"],
)
def test_native_report_negatives_never_return_another_processs_data(
    tmp_path: Path, case: str
) -> None:
    import os

    path = tmp_path / "python3.12-2026.ips"
    text = {
        "wrong-pid": _ips(pid=_PID + 1),
        "stale": _ips(),
        "torn": _ips()[:-40],
        "oversize": _ips() + " " * diag.NATIVE_MAX_FILE_BYTES,
        "not-a-crash": _ips(bug_type="308"),
        "other-process": _ips(proc="Safari"),
    }[case]
    path.write_text(text)
    started = 1_000_000.0
    mtime = started - 100 if case == "stale" else started + 5
    os.utime(path, (mtime, mtime))  # exactly (atime, mtime): the stub wants a fixed pair
    found = _find(tmp_path, started=started)
    assert found["found"] is False and found["reason"]
    assert "sym0" not in json.dumps(found) and "SIGSEGV" not in json.dumps(found)


def test_native_report_refuses_symlinks_and_missing_directory(tmp_path: Path) -> None:
    real = tmp_path / "real.ips"
    real.write_text(_ips())
    scan = tmp_path / "scan"
    scan.mkdir()
    (scan / "link.ips").symlink_to(real)
    assert _find(scan)["found"] is False
    assert _find(tmp_path / "missing")["found"] is False
    link_dir = tmp_path / "linkdir"
    link_dir.symlink_to(scan)
    assert _find(link_dir)["found"] is False


@pytest.mark.parametrize(
    ("code", "system", "environ", "expected"),
    [
        (-11, "Darwin", {"CI": "true", "GITHUB_ACTIONS": "true"}, True),
        (-11, "Darwin", {}, False),  # a developer machine: never read the real home
        (-11, "Darwin", {"CI": "true"}, False),
        (-11, "Linux", {"CI": "true", "GITHUB_ACTIONS": "true"}, False),
        (-9, "Darwin", {"CI": "true", "GITHUB_ACTIONS": "true"}, False),
        (7, "Darwin", {"CI": "true", "GITHUB_ACTIONS": "true"}, False),
    ],
)
def test_native_report_guard(
    code: int, system: str, environ: dict[str, str], expected: bool
) -> None:
    assert diag.native_report_enabled(code, system=system, environ=environ) is expected


def test_a_non_native_exit_never_consults_the_real_reports_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom() -> Path:
        raise AssertionError("the real DiagnosticReports must not be resolved")

    monkeypatch.setattr(diag, "os_user_reports_dir", _boom)
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    child = CapturedChild([sys.executable, "-c", "import sys; sys.exit(7)"], config=tmp_path / "c")
    assert child.wait(timeout=10) == 7
    child.shutdown_evidence(tmp_path / "artifacts")
    payload = json.loads(next((tmp_path / "artifacts").glob("*.json")).read_text())
    assert "native_report" not in payload
