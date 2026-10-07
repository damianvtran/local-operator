"""The Slice-0 POC's isolation instruments must be able to say no.

WHY THIS FILE EXISTS. The POC's probe (`infra/remote-agents-poc/image/probes.py`) and
its driver live outside the package and outside `tests/`, so nothing else in the suite
executes them — and probe 4e, the watcher, is the only instrument that can observe the
model key sitting in a process's LAUNCH environment, which is the exposure agent review
round 1 raised as SEC-1. An instrument whose RED case has never been produced is a
decoration: the cases below drive its scanning and verdict logic with the
process-environment source stubbed, because a real one needs procfs.

The container run that produced the real red reading (and the five that produce the
green one) is recorded in `docs/design/remote-cloud-agents-poc-results.md`.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROBES_PATH = ROOT / "infra/remote-agents-poc/image/probes.py"

#: A value shaped like a key, and deliberately not one. This is the fixture the
#: watcher searches for; it is never a real credential and never derived from one.
TEST_KEY = "LOP-POC-TEST-KEY-0123456789abcdef"


@pytest.fixture(scope="module")
def probes() -> ModuleType:
    """`probes.py` loaded by path — the directory is not an importable package."""
    spec = importlib.util.spec_from_file_location("poc_probes", PROBES_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _watch(
    probes: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    blobs: list[tuple[str, bytes, str]],
    *,
    key: str = TEST_KEY,
    prefix_chars: int = 8,
) -> tuple[int, dict[str, Any]]:
    """Run one watcher sample over a stubbed process list."""
    monkeypatch.setattr(probes, "_environ_blobs", lambda: blobs)
    monkeypatch.setattr(probes, "_procfs_available", lambda: True)
    monkeypatch.setattr(probes, "_find_agent_pid", lambda: None)
    stop = tmp_path / "stop"
    stop.write_text("", encoding="utf-8")
    out = tmp_path / "watch.json"
    rc = probes.watch_environ(out, key, prefix_chars, stop, 1000, 5.0, 1)
    return rc, json.loads(out.read_text(encoding="utf-8"))


def test_the_watcher_goes_red_when_a_process_environment_carries_the_key(
    probes: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The whole point of probe 4e: it must FAIL on the delivery path it replaced."""
    blobs = [
        ("7", b"HOME=/workspace\x00PATH=/usr/bin", "lop"),
        ("42", f"LOP_POC_MODEL_KEY={TEST_KEY}\x00PATH=/usr/bin".encode(), "sleep"),
    ]
    rc, report = _watch(probes, monkeypatch, tmp_path, blobs)
    assert rc == 1, "a process whose environment carries the key must make the watcher RED"
    assert report["pass"] is False
    assert report["matches_by_needle"]["value"] == 1
    assert {"pid": 42, "comm": "sleep", "needle": "value"} in report["matching_processes"]
    assert TEST_KEY not in json.dumps(report), "the report must never carry the value"


def test_the_watcher_goes_green_when_no_environment_carries_it(
    probes: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    blobs = [
        ("7", b"HOME=/workspace\x00PATH=/usr/bin", "lop"),
        ("8", b"HOME=/root\x00PATH=/usr/sbin", "init"),
    ]
    rc, report = _watch(probes, monkeypatch, tmp_path, blobs)
    assert rc == 0
    assert report["pass"] is True
    assert set(report["matches_by_needle"].values()) == {0}
    assert report["processes_scanned_max"] == 2


def test_the_prefix_needle_is_counted_separately_from_the_value(
    probes: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A leaked prefix is worth detecting on its own, and must not be reported AS the
    value: the two counts answer different questions about the same run."""
    blobs = [("9", b"SOME_VAR=" + TEST_KEY[:8].encode(), "lop-agent")]
    rc, report = _watch(probes, monkeypatch, tmp_path, blobs)
    assert rc == 1
    assert report["matches_by_needle"]["prefix"] == 1
    assert report["matches_by_needle"]["value"] == 0


def test_the_watcher_blocks_instead_of_passing_without_procfs(
    probes: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No procfs means it cannot observe — which is not the same as "clean"."""
    monkeypatch.setattr(probes, "_procfs_available", lambda: False)
    out = tmp_path / "watch.json"
    rc = probes.watch_environ(out, TEST_KEY, 8, tmp_path / "stop", 1000, 5.0, 1)
    assert rc == 2
    assert json.loads(out.read_text(encoding="utf-8"))["blocked"] is True


def test_probe_4f_is_green_without_a_ps_and_red_with_one(
    probes: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """4f is the instrument behind the "no `ps` in the image" condition.

    Both directions, and both shapes the condition can fail in: a `ps` reachable by NAME
    and one reachable only by an ABSOLUTE PATH (which is what this Dockerfile's own
    `/usr/local/bin` would be), plus a multiplexer that implements `ps` without a `ps`
    file. A condition asserted by a probe that cannot fail is the defect the rest of this
    file exists to prevent.
    """
    monkeypatch.setattr(probes, "shutil", SimpleNamespace(which=lambda name: None))
    monkeypatch.setattr(probes, "_PS_PATHS", ())
    green = probes.probe_ps_absent()
    assert green["pass"] is True
    assert green["detail"]["which_ps"] is None

    monkeypatch.setattr(
        probes,
        "shutil",
        SimpleNamespace(which=lambda name: "/usr/bin/ps" if name == "ps" else None),
    )
    red = probes.probe_ps_absent()
    assert red["pass"] is False
    assert red["detail"]["which_ps"] == "/usr/bin/ps"

    # A `ps` that only an ABSOLUTE PATH finds, via a real file: monkeypatching
    # `os.path.exists` would patch it for every module in the interpreter.
    ps_file = tmp_path / "ps"
    ps_file.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(probes, "shutil", SimpleNamespace(which=lambda name: None))
    monkeypatch.setattr(probes, "_PS_PATHS", (str(ps_file),))
    path_red = probes.probe_ps_absent()
    assert path_red["pass"] is False
    assert path_red["detail"]["paths_present"] == [str(ps_file)]

    monkeypatch.setattr(probes, "_PS_PATHS", ())
    monkeypatch.setattr(
        probes,
        "shutil",
        SimpleNamespace(which=lambda name: "/bin/busybox" if name == "busybox" else None),
    )
    multiplexer_red = probes.probe_ps_absent()
    assert multiplexer_red["pass"] is False
    assert multiplexer_red["detail"]["multiplexers_present"] == ["busybox"]


def _scan(directory: Path, key: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(PROBES_PATH),
            "--scan-dir",
            str(directory),
            "--key-fd",
            "0",
            "--key-prefix-chars",
            "8",
        ],
        input=key,
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_rescan_refuses_an_empty_key_rather_than_reporting_clean(tmp_path: Path) -> None:
    """rc 0 from a scan that inspected nothing is the "dead instrument" shape.

    The three cases together: an empty key is an ERROR (2), a key present in a file is
    a hit (1), and an absent key over a real file is the only thing that may be 0.
    """
    (tmp_path / "clean.txt").write_text("nothing to see here", encoding="utf-8")
    empty = _scan(tmp_path, "")
    assert empty.returncode == 2, empty.stdout + empty.stderr
    (tmp_path / "leak.txt").write_text(f"leaked {TEST_KEY}", encoding="utf-8")
    hit = _scan(tmp_path, TEST_KEY)
    assert hit.returncode == 1
    # The file carries the value and therefore its prefix: TWO needles, two counts, one
    # file. Per-needle counts are the point — a single number could not say which probe
    # found it, and the value's presence is strictly stronger than the prefix's.
    counts = json.loads(hit.stdout)["matches_by_needle"]
    assert counts["value"] == 1
    assert counts["prefix"] == 1
    (tmp_path / "leak.txt").unlink()
    clean = _scan(tmp_path, TEST_KEY)
    assert clean.returncode == 0
    assert json.loads(clean.stdout)["scanned_files"] >= 1
