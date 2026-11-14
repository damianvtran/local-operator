"""A slow daemon restart is handed off to a watched background worker, not killed.

THE MEASURED FAILURE (operator's Mac, relayed 2026-11-14): `lop network restart`
took 35 s and `lop mobile restart` 54 s, both succeeding, while the update's
daemon-refresh stage gave each refresh child at most 30 s and SIGKILLed it at that
bound. Every update reported the service daemons and mobile "failed" and left the
relay and the mobile daemon on the OLD build until a hand restart.

These tests drive REAL processes \u2014 parent ``subprocess.run`` \u2192 refresh child \u2192
detached watchdog \u2192 worker \u2014 with the bounds scaled down by monkeypatching, and a
fake refresh body that sleeps longer than the parent's bound. What they pin:

* a slow but healthy restart is reported ``pending`` / "still restarting", never
  ``failed``, and the background worker finishes and records success;
* a real non-zero exit is still ``failed``;
* a wedged worker is still killed \u2014 at its own (background) bound \u2014 and reported.

The launchd half (a real ``kickstart -k`` of a real LaunchAgent) cannot run on a
Linux host; the worker here stands in for it.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pytest

from local_operator import update as update_mod
from local_operator.update import DaemonStatus, MobileRefresh

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The fake refresh child. It goes through the REAL ``_run_refresh_child`` gate:
#: ``-c`` keeps the trailing ``--refresh-*`` flag in ``sys.orig_argv``, which is
#: what marks a process as a refresh child, and the worker re-executes this same
#: argv with the worker role set, so the body below is what runs in the background.
FAKE_CHILD = """
import os, sys, time
from local_operator import update

def body():
    unit = os.environ["FAKE_UNIT"]
    if unit == "services":
        print("refreshing: network relay :: lop network install", flush=True)
    time.sleep(float(os.environ["FAKE_SLEEP_S"]))
    rc = int(os.environ["FAKE_RC"])
    if rc:
        print("error: boom", file=sys.stderr)
        return rc
    if unit == "services":
        print('daemon_report: {"name":"network relay","status":"refreshed"}')
        print("network relay: restarted on the new build")
    return 0

sys.exit(update._run_refresh_child(
    body, unit=os.environ["FAKE_UNIT"], bound_s=float(os.environ["FAKE_BOUND_S"])
))
"""


def _fake_child(flag: str) -> tuple[list[str], str]:
    return [sys.executable, "-c", FAKE_CHILD, flag], sys.executable


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Isolated config dir (the run records land there) and the scaled bounds."""
    config = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    monkeypatch.setenv("PYTHONPATH", str(REPO_ROOT))
    monkeypatch.delenv(update_mod._REFRESH_ROLE_ENV, raising=False)
    monkeypatch.delenv(update_mod._REFRESH_HANDOFF_ENV, raising=False)
    # Scaled-down stand-ins for the 60 s/30 s parent bounds and the 5 s margin:
    # the child is told to hand off 2 s before the parent's 5 s kill.
    monkeypatch.setattr(update_mod, "_DAEMON_REFRESH_TIMEOUT_S", 5.0)
    monkeypatch.setattr(update_mod, "_MOBILE_RESTART_TIMEOUT_S", 5.0)
    monkeypatch.setattr(update_mod, "_REFRESH_HANDOFF_MARGIN_S", 2.0)
    yield config / "update" / "refresh"


def _set_fake(monkeypatch: pytest.MonkeyPatch, *, unit: str, sleep: float, rc: int, bound: float):
    monkeypatch.setenv("FAKE_UNIT", unit)
    monkeypatch.setenv("FAKE_SLEEP_S", str(sleep))
    monkeypatch.setenv("FAKE_RC", str(rc))
    monkeypatch.setenv("FAKE_BOUND_S", str(bound))


def _wait_for_record(root: Path, unit: str, within: float) -> dict[str, object]:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        results = list(root.glob(f"{unit}-*/result.json"))
        if results:
            return json.loads(results[0].read_text(encoding="utf-8"))
        time.sleep(0.1)
    raise AssertionError(f"no background record for {unit} within {within}s")


def _services_refresh() -> update_mod.DaemonRefresh:
    with (
        patch.object(update_mod, "_installed_daemon_plists", return_value=[Path("/tmp/x")]),
        patch.object(
            update_mod, "_daemon_refresh_invocation", return_value=_fake_child("--refresh-daemons")
        ),
    ):
        return update_mod.refresh_service_daemons_after_upgrade()


class _Plist:
    def exists(self) -> bool:
        return True


def _mobile_refresh() -> MobileRefresh:
    with (
        patch.object(update_mod, "_mobile_plist_path", return_value=_Plist()),
        patch.object(update_mod, "_repair_refusal", return_value=None),
        patch.object(
            update_mod, "_mobile_restart_invocation", return_value=_fake_child("--refresh-mobile")
        ),
    ):
        return update_mod.refresh_mobile_after_upgrade()


def test_a_slow_service_restart_is_still_restarting_and_finishes_in_the_background(
    rig: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The relay's 35 s, scaled: a body slower than the parent's bound is not killed."""
    _set_fake(monkeypatch, unit="services", sleep=6.0, rc=0, bound=30.0)
    started = time.monotonic()
    refresh = _services_refresh()
    elapsed = time.monotonic() - started
    assert elapsed < 5.0, f"the parent's 5 s bound must not be what ended the wait ({elapsed:.2f}s)"
    assert refresh.statuses == (DaemonStatus("network relay", "pending"),)
    assert refresh.warnings == (), "slow is not a warning, and never 'failed'"
    assert any("still restarting" in line for line in refresh.lines), refresh.lines
    assert not any("refreshing:" in line for line in refresh.lines)
    record = _wait_for_record(rig, "services", within=15.0)
    assert record["rc"] == 0 and record["timed_out"] is False
    log = next(rig.glob("services-*/stdout.log")).read_text(encoding="utf-8")
    assert '"status":"refreshed"' in log, "the background worker finished the restart"
    assert update_mod.background_refresh_lines()[0].startswith(
        "background services refresh: finished"
    )


def test_a_slow_mobile_bounce_is_restarting_not_failed(
    rig: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mobile 54 s, scaled: ``restarting`` in the parent, ``pending`` in the report."""
    _set_fake(monkeypatch, unit="mobile", sleep=6.0, rc=0, bound=30.0)
    result = _mobile_refresh()
    assert result.kind == "restarting"
    assert any("mobile: still restarting" in line for line in result.lines), result.lines
    entry = update_mod._mobile_daemon_refresh(result)
    assert entry.statuses == (DaemonStatus("mobile", "pending"),)
    assert entry.warnings == ()
    record = _wait_for_record(rig, "mobile", within=15.0)
    assert record["rc"] == 0 and record["timed_out"] is False


def test_a_fast_restart_reads_exactly_as_before(rig: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Finishing inside the hand-off relays the worker's output unchanged."""
    _set_fake(monkeypatch, unit="services", sleep=0.0, rc=0, bound=30.0)
    refresh = _services_refresh()
    assert refresh.statuses == (DaemonStatus("network relay", "refreshed"),)
    assert refresh.lines == ("network relay: restarted on the new build",)
    assert refresh.warnings == ()


def test_a_real_nonzero_exit_is_still_failed(rig: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_fake(monkeypatch, unit="services", sleep=0.0, rc=1, bound=30.0)
    refresh = _services_refresh()
    assert refresh.warnings == ("warning: could not refresh installed daemons: error: boom",)
    assert DaemonStatus("network relay", "failed") in refresh.statuses
    _set_fake(monkeypatch, unit="mobile", sleep=0.0, rc=1, bound=30.0)
    assert _mobile_refresh() == MobileRefresh(kind="failed", error="error: boom")


def test_a_wedged_worker_is_killed_at_its_own_bound_and_reported(
    rig: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hang protection is kept: the watchdog kills the worker and records the failure."""
    _set_fake(monkeypatch, unit="services", sleep=120.0, rc=0, bound=4.0)
    refresh = _services_refresh()
    assert refresh.statuses == (DaemonStatus("network relay", "pending"),)
    record = _wait_for_record(rig, "services", within=15.0)
    assert record["timed_out"] is True and record["rc"] is None
    pid = record["worker_pid"]
    assert isinstance(pid, int)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    (line,) = update_mod.background_refresh_lines()
    assert "FAILED" in line and "did not finish within 4s" in line
    assert "`lop services restart`" in line


def test_the_hand_off_strips_announcements_for_a_pre_upgrade_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parent that installs this release reads an announced, unreported daemon as
    KILLED and words it ``failed``; it does not know ``pending``. So the hand-off must
    not leave that daemon's ``refreshing:`` line in its stdout."""
    monkeypatch.delenv(update_mod._REFRESH_ROLE_ENV, raising=False)
    worker = (
        "import time\nprint('refreshing: network relay :: lop network install', flush=True)\n"
        "time.sleep(3)\n"
    )
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = update_mod._hand_off_when_slow(
            unit="services",
            argv=[sys.executable, "-c", worker],
            executable=sys.executable,
            bound_s=10.0,
            handoff_at=time.time() + 1.0,
            runs_root=tmp_path,
        )
    assert rc == 0
    text = out.getvalue()
    assert "refreshing:" not in text
    # Read with the PRE-upgrade vocabulary: no status at all, so no "failed".
    old_kinds = frozenset({"refreshed", "already", "failed", "unsupervised"})
    with patch.object(update_mod, "_STATUS_KINDS", old_kinds):
        assert update_mod._child_daemon_statuses(text) == ()
    assert update_mod._child_daemon_statuses(text) == (DaemonStatus("network relay", "pending"),)
    _wait_for_record(tmp_path, "services", within=10.0)


def test_the_parent_passes_its_hand_off_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(update_mod, "_REFRESH_HANDOFF_MARGIN_S", 5.0)
    completed = subprocess.CompletedProcess([], 0, stdout="", stderr="")
    with (
        patch.object(update_mod, "_installed_daemon_plists", return_value=[Path("/tmp/x")]),
        patch("subprocess.run", return_value=completed) as run,
    ):
        before = time.time()
        update_mod.refresh_service_daemons_after_upgrade()
    env = run.call_args.kwargs["env"]
    handoff = float(env[update_mod._REFRESH_HANDOFF_ENV])
    timeout = run.call_args.kwargs["timeout"]
    # The env carries three decimals, so allow the rounding either way.
    assert before + timeout - 5.0 - 0.01 <= handoff <= time.time() + timeout - 5.0 + 0.01


def test_outside_a_refresh_child_the_body_runs_in_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pytest's argv names no refresh flag, so nothing is ever spawned here."""
    with patch.object(update_mod, "_hand_off_when_slow") as handoff:
        assert update_mod._run_refresh_child(lambda: 7, unit="services", bound_s=1.0) == 7
    handoff.assert_not_called()


def test_an_unwritable_run_directory_falls_back_to_the_in_process_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(update_mod._REFRESH_ROLE_ENV, raising=False)
    monkeypatch.setattr(update_mod, "_started_as_refresh_child", lambda: True)
    with patch.object(update_mod, "_hand_off_when_slow", side_effect=PermissionError("ro")):
        assert update_mod._run_refresh_child(lambda: 3, unit="mobile", bound_s=1.0) == 3
