"""``run_bounded.py --start-by``: refuse a run that would start outside its window.

Why these cells exist: node hosts grant heavy-test windows with a latest start
("start by 08:05Z"). A run that queued behind a slot or lease and woke up late
must exit with the distinct late-start code, never spawn its child, and leave a
completion record with ``cause=late-start``. Without the flag nothing changes.
The CLI cells drive the real script as a separate process; the deadline-passes-
during-a-wait cell drives ``run()`` with an injected clock so it is exact.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import pytest

from local_operator import bounded_contract, memory_guard

RUNNER = Path(__file__).resolve().parents[3] / "scripts" / "run_bounded.py"


def _invoke(tmp_path: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("CMUX_", "LOCAL_OPERATOR_", "LOP_"))
    }
    env.update(
        HOME=str(tmp_path),
        LOCAL_OPERATOR_CONFIG_DIR=str(tmp_path / "config"),
        LOCAL_OPERATOR_SCRATCHPAD=str(tmp_path),
    )
    return subprocess.run(
        [sys.executable, str(RUNNER), "--contract=command-status", "--memory-mb", "0", *args],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _records(tmp_path: Path) -> list[dict[str, object]]:
    return [
        json.loads(p.read_text()) for p in (tmp_path / "run-bounded-runs").glob("*.completion.json")
    ]


def _touch(marker: Path) -> list[str]:
    return ["--", sys.executable, "-c", f"open({str(marker)!r}, 'w').write('spawned')"]


@pytest.mark.parametrize(
    "deadline",
    [
        lambda: f"{time.time() - 60:.3f}",
        lambda: "2020-01-01T00:00:00Z",
        lambda: "2020-01-01T09:00:00+09:00",
    ],
    ids=["epoch", "iso-z", "iso-offset"],
)
def test_past_deadline_exits_late_start_and_never_spawns(
    tmp_path: Path, deadline: Callable[[], str]
) -> None:
    marker = tmp_path / "must-not-exist"
    done = _invoke(tmp_path, ["--start-by", deadline(), *_touch(marker)])
    assert done.returncode == bounded_contract.EXIT_LATE_START == 79, done.stderr
    assert not marker.exists()
    assert "[run_bounded] late start: start-by " in done.stderr
    assert "command not started" in done.stderr
    assert "cause=late-start" in done.stderr and "spawned=no" in done.stderr
    (row,) = _records(tmp_path)
    assert row["termination_cause"] == "late-start"
    assert row["wrapper_exit_code"] == 79
    assert row["command_exit_code"] is None


def test_future_deadline_runs_the_command(tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    done = _invoke(tmp_path, ["--start-by", "2999-01-01T00:00:00Z", *_touch(marker)])
    assert done.returncode == 0, done.stderr
    assert marker.read_text() == "spawned"
    (row,) = _records(tmp_path)
    assert row["termination_cause"] == "child-exit"


def test_without_the_flag_behaviour_is_unchanged(tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    done = _invoke(tmp_path, _touch(marker))
    assert done.returncode == 0, done.stderr
    assert marker.exists()
    assert "late start" not in done.stderr


@pytest.mark.parametrize(
    # -1e300/1e300 are finite floats datetime cannot represent: refused at
    # parse time (2), not left to crash utc_stamp at the post-wait breach.
    "bad", ["soon", "2026-11-19T08:05:00", "nan", "2026-13-40T00:00Z", "-1e300", "1e300"]
)
def test_unparseable_start_by_is_a_usage_error_before_anything(tmp_path: Path, bad: str) -> None:
    marker = tmp_path / "must-not-exist"
    # "=" form: a bare "-1e300" would be read by argparse as an option and
    # refused for the wrong reason, which would not exercise parse_start_by.
    done = _invoke(tmp_path, [f"--start-by={bad}", *_touch(marker)])
    assert done.returncode == 2, done.stderr
    assert "argument --start-by: " in done.stderr
    assert "expected one argument" not in done.stderr
    assert not marker.exists()
    # argparse refused before run(): no completion record either.
    assert _records(tmp_path) == []


def test_parse_start_by_accepts_epoch_and_iso() -> None:
    assert bounded_contract.parse_start_by("1763539500") == 1763539500.0
    assert bounded_contract.parse_start_by("2025-11-19T08:05:00Z") == 1763539500.0
    assert bounded_contract.parse_start_by("2025-11-19T09:05:00+01:00") == 1763539500.0
    with pytest.raises(ValueError):
        bounded_contract.parse_start_by("2025-11-19T08:05:00")  # naive: zone unknown


def _load_runner():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("run_bounded_start_by_under_test", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_deadline_that_passes_during_pre_spawn_work_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Admitted in time, late by the spawn point: the clock read at the gate is
    # what decides, not the time the flag was parsed.
    monkeypatch.setenv("LOCAL_OPERATOR_SCRATCHPAD", str(tmp_path))
    # An inherited bash-tool grant would refuse memory_mb=0 (125) first.
    monkeypatch.delenv(memory_guard.RESOLVED_MEMORY_MB_ENV, raising=False)
    runner = _load_runner()
    spawned: list[object] = []
    monkeypatch.setattr(runner.bounded_contract, "execute", lambda *a, **k: spawned.append(a) or 0)
    stream = io.StringIO()
    rc = runner.run(
        ["true"],
        timeout=None,
        grace=1.0,
        memory_mb=0,
        stream=stream,
        contract="command-status",
        start_by=1000.0,
        _clock=lambda: 1000.5,
    )
    assert rc == 79
    assert spawned == []
    assert "0.5s late" in stream.getvalue()
    (row,) = _records(tmp_path)
    assert row["termination_cause"] == "late-start"


def test_late_start_record_carries_the_resolved_memory_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The budget is resolved before the gate, so the late-start record names
    # the grant the run would have had, as the spawn-failure record does.
    monkeypatch.setenv("LOCAL_OPERATOR_SCRATCHPAD", str(tmp_path))
    monkeypatch.delenv(memory_guard.RESOLVED_MEMORY_MB_ENV, raising=False)
    runner = _load_runner()
    monkeypatch.setattr(runner.bounded_contract, "execute", lambda *a, **k: 0)
    rc = runner.run(
        ["true"],
        timeout=None,
        grace=1.0,
        memory_mb=512,
        stream=io.StringIO(),
        contract="command-status",
        start_by=1000.0,
        _clock=lambda: 1001.0,
    )
    assert rc == 79
    (row,) = _records(tmp_path)
    assert row["termination_cause"] == "late-start"
    assert row["memory_ceiling_mb"] == 512
    assert row["memory_budget_source"] == "explicit"
