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


# --- PR #4004 round-1 remediation -------------------------------------------------

_SLOW_RELAY_WORKER = (
    "import time\nprint('refreshing: network relay :: lop network install', flush=True)\n"
    "time.sleep({sleep})\n"
    'print(\'daemon_report: {{"name":"network relay","status":"refreshed"}}\', flush=True)\n'
)


def _hand_off_in_process(
    tmp_path: Path, worker: str, *, unit: str, bound: float, handoff_in: float, **kwargs: object
) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = update_mod._hand_off_when_slow(
            unit=unit,
            argv=[sys.executable, "-c", worker],
            executable=sys.executable,
            bound_s=bound,
            handoff_at=time.time() + handoff_in,
            runs_root=tmp_path,
            **kwargs,  # type: ignore[arg-type]
        )
    return rc, out.getvalue(), err.getvalue()


def _reap_watchdog(run: Path) -> None:
    """The in-process hand-off makes the watchdog OUR child: reap it, by exact pid."""
    pid = update_mod._read_json(run / "spec.json").get("watchdog_pid")
    if isinstance(pid, int):
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass


def test_an_old_parent_hears_from_the_mobile_child_that_it_is_not_done_yet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1-1 / Q1: the pre-hand-off parent (no hand-off time passed) words rc 0 as
    "mobile daemon restarted" \u2014 measured against the real v0.103.0 parent in review
    round 1 \u2014 so the child's own line must say the bounce is still running."""
    monkeypatch.delenv(update_mod._REFRESH_ROLE_ENV, raising=False)
    worker = "import time\ntime.sleep(2)\n"
    rc, old, _ = _hand_off_in_process(
        tmp_path, worker, unit="mobile", bound=10.0, handoff_in=0.5, parent_knows_pending=False
    )
    assert rc == 0
    assert "mobile: still restarting" in old
    assert "an older updater may report it as restarted before it has" in old
    assert "lop services status" in old
    rc, new, _ = _hand_off_in_process(tmp_path, worker, unit="mobile", bound=10.0, handoff_in=0.5)
    assert rc == 0 and "mobile: still restarting" in new and "older updater" not in new
    for run in update_mod._refresh_runs(tmp_path, "mobile"):
        _reap_watchdog(run)
        assert (run / "result.json").is_file()


def test_only_a_parent_that_passed_no_hand_off_time_is_treated_as_old(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(update_mod._REFRESH_ROLE_ENV, raising=False)
    monkeypatch.setattr(update_mod, "_started_as_refresh_child", lambda: True)
    seen: list[object] = []
    with patch.object(
        update_mod,
        "_hand_off_when_slow",
        side_effect=lambda **kw: seen.append(kw["parent_knows_pending"]) or 0,
    ):
        monkeypatch.delenv(update_mod._REFRESH_HANDOFF_ENV, raising=False)
        update_mod._run_refresh_child(lambda: 9, unit="mobile", bound_s=1.0)
        monkeypatch.setenv(update_mod._REFRESH_HANDOFF_ENV, f"{time.time() + 5:.3f}")
        update_mod._run_refresh_child(lambda: 9, unit="mobile", bound_s=1.0)
    assert seen == [False, True]


def test_a_dead_watchdog_reads_failed_and_its_run_is_pruned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1-2: SIGKILL the watchdog (and its worker) mid-run: no record will ever come,
    so status must not say "still running" forever, and pruning must remove it."""
    import signal

    monkeypatch.setattr(update_mod, "_refresh_runs_root", lambda: tmp_path)
    monkeypatch.delenv(update_mod._REFRESH_ROLE_ENV, raising=False)
    rc, _, _ = _hand_off_in_process(
        tmp_path, "import time\ntime.sleep(60)\n", unit="services", bound=120.0, handoff_in=0.3
    )
    assert rc == 0
    (run,) = update_mod._refresh_runs(tmp_path, "services")
    pid = update_mod._read_json(run / "spec.json")["watchdog_pid"]
    assert isinstance(pid, int)
    # Its own session, so its pgid is its pid: the worker goes with it.
    os.killpg(pid, signal.SIGKILL)
    os.waitpid(pid, 0)
    (line,) = update_mod.background_refresh_lines()
    assert line.startswith("background services refresh: FAILED"), line
    assert "no record" in line and "`lop services restart`" in line
    monkeypatch.setattr(update_mod, "_REFRESH_RUNS_KEPT", 0)
    update_mod._prune_refresh_runs(tmp_path, "services")
    assert not run.exists()


def test_a_run_past_its_bound_with_a_live_pid_is_abandoned_but_a_fresh_one_is_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1-2's clock backstop: a pid the OS reused looks alive, so age decides."""
    monkeypatch.setattr(update_mod, "_refresh_runs_root", lambda: tmp_path)
    now = time.time()
    old = tmp_path / f"services-{update_mod._run_stamp(now - 1000)}-a"
    fresh = tmp_path / f"services-{update_mod._run_stamp(now)}-b"
    for run, started in ((old, now - 1000), (fresh, now)):
        run.mkdir()
        meta = {"unit": "services", "started_at": started, "bound_s": 240.0}
        update_mod._write_run_meta(run, {**meta, "watchdog_pid": os.getpid()})
    assert update_mod._run_abandoned(old) is True
    assert update_mod._run_abandoned(fresh) is False
    assert update_mod.background_refresh_lines() == [
        f"background services refresh: still running ({fresh})"
    ]
    monkeypatch.setattr(update_mod, "_REFRESH_RUNS_KEPT", 0)
    update_mod._prune_refresh_runs(tmp_path, "services")
    assert not old.exists() and fresh.exists(), "a running run is never pruned"


def _ended_run(path: Path) -> Path:
    """A directory that looks like a FINISHED run (so pruning would remove it)."""
    path.mkdir(parents=True)
    (path / "result.json").write_text("{}", encoding="utf-8")
    (path / "precious.txt").write_text("keep", encoding="utf-8")
    return path


def test_run_removal_refuses_every_path_that_is_not_a_run_of_this_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CI census (PR #4004): the hand-off's one ``rmtree`` must not be steerable.

    Each crafted target is a FINISHED run in shape (``result.json`` present), so
    only the guard in ``_remove_refresh_run`` stands between it and deletion.
    """
    root = tmp_path / "update" / "refresh"
    root.mkdir(parents=True)
    stamp = update_mod._run_stamp(time.time())
    outside = _ended_run(tmp_path / "sessions" / f"services-{stamp}-x1")
    nested = _ended_run(root / "nested" / f"services-{stamp}-x2")
    wrong_unit = _ended_run(root / f"mobile-{stamp}-x3")
    not_a_run = _ended_run(root / "services-notastamp")
    link = root / f"services-{stamp}-x4"
    link.symlink_to(outside, target_is_directory=True)
    climbing = root / ".." / ".." / "sessions" / outside.name

    for target in (outside, nested, wrong_unit, not_a_run, link, climbing):
        assert update_mod._remove_refresh_run(root, "services", target) is False, target
    # The whole prune path, too: the symlink and the misnamed dir match the glob.
    monkeypatch.setattr(update_mod, "_REFRESH_RUNS_KEPT", 0)
    update_mod._prune_refresh_runs(root, "services")
    for kept in (outside, nested, wrong_unit, not_a_run):
        assert (kept / "precious.txt").read_text(encoding="utf-8") == "keep", kept
    assert link.is_symlink()

    # The positive control: a genuine ended run of this unit IS removed.
    real = _ended_run(root / f"services-{stamp}-ok_1")
    assert update_mod._remove_refresh_run(root, "services", real) is True
    assert not real.exists()


def test_the_latest_run_is_the_latest_started_not_the_latest_finished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1-3: run A starts, run B starts, A finishes (bumping its dir mtime) \u2014 B is
    still the latest run."""
    monkeypatch.setattr(update_mod, "_refresh_runs_root", lambda: tmp_path)
    now = time.time()
    a = tmp_path / f"services-{update_mod._run_stamp(now - 0.5)}-zzz"
    b = tmp_path / f"services-{update_mod._run_stamp(now)}-aaa"
    for run, started in ((a, now - 0.5), (b, now)):
        run.mkdir()
        update_mod._write_run_meta(
            run,
            {
                "unit": "services",
                "started_at": started,
                "bound_s": 240.0,
                "watchdog_pid": os.getpid(),
            },
        )
    time.sleep(0.05)
    (a / "result.json").write_text(json.dumps({"rc": 0, "finished_at": time.time()}))
    assert a.stat().st_mtime > b.stat().st_mtime, "the precondition of the old bug"
    assert update_mod.background_refresh_lines() == [
        f"background services refresh: still running ({b})"
    ]
    # Stamps sort as text in start order, across a second boundary too.
    assert update_mod._run_stamp(1.999999) < update_mod._run_stamp(2.0)


def test_a_worker_finishing_while_the_logs_are_read_is_never_relayed_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1-4: the verdict is taken BEFORE the logs. Force the race: the stdout read
    returns the announcement, then waits for the worker to finish. Read in the old
    order, that stale announcement was relayed as a FINISHED run, and the parent
    words an announced-but-unreported daemon ``failed``."""
    monkeypatch.delenv(update_mod._REFRESH_ROLE_ENV, raising=False)
    real_read = update_mod._read_text

    def racing_read(path: Path) -> str:
        text = real_read(path)
        if path.name == "stdout.log":
            deadline = time.monotonic() + 15.0
            while not (path.parent / "result.json").exists() and time.monotonic() < deadline:
                time.sleep(0.05)
        return text

    monkeypatch.setattr(update_mod, "_read_text", racing_read)
    rc, out, _ = _hand_off_in_process(
        tmp_path, _SLOW_RELAY_WORKER.format(sleep=1.5), unit="services", bound=30.0, handoff_in=0.5
    )
    assert rc == 0
    statuses = update_mod._child_daemon_statuses(out)
    assert DaemonStatus("network relay", "failed") not in statuses, out
    assert statuses == (DaemonStatus("network relay", "pending"),)
    (run,) = update_mod._refresh_runs(tmp_path, "services")
    _reap_watchdog(run)


def test_a_nested_refresh_child_never_inherits_the_worker_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R1-5 / Q2: the services worker of a hand run bounces mobile through a nested
    child; with the role inherited it ran in-process and was killed at 30 s."""
    monkeypatch.setenv(update_mod._REFRESH_ROLE_ENV, "worker")
    env = update_mod._refresh_child_env(30.0)
    assert update_mod._REFRESH_ROLE_ENV not in env
    assert update_mod._REFRESH_HANDOFF_ENV in env
    assert os.environ[update_mod._REFRESH_ROLE_ENV] == "worker", "only the child's copy"


@pytest.mark.parametrize(
    ("mobile", "expected"),
    [
        (MobileRefresh(kind="failed", error="timed out"), 1),
        (MobileRefresh(kind="restarting"), 0),
        (MobileRefresh(kind="restarted"), 0),
    ],
)
def test_a_hand_run_fails_when_its_mobile_half_failed(
    mobile: MobileRefresh, expected: int, capsys: pytest.CaptureFixture[str]
) -> None:
    """R1-5 / Q2: the hand run's exit code is the background record, so a mobile
    timeout must not exit 0 (status said ``ok``)."""
    with (
        patch.object(update_mod, "daemons_refresh_command", return_value=0),
        patch.object(update_mod, "_repair_refusal", return_value=None),
        patch.object(update_mod, "refresh_mobile_after_upgrade", return_value=mobile),
    ):
        assert update_mod._run_daemon_repair(services_only=False) == expected
    capsys.readouterr()


def test_status_names_the_workers_last_word_on_a_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q2: a mobile timeout inside a services run carries no daemon report; status
    names it from the worker's stderr rather than a bare ``exit 1``."""
    monkeypatch.setattr(update_mod, "_refresh_runs_root", lambda: tmp_path)
    run = tmp_path / f"services-{update_mod._run_stamp(time.time())}-x"
    run.mkdir()
    (run / "stderr.log").write_text(
        "warning: mobile daemon did not restart: timed out; run lop mobile restart\n"
    )
    (run / "result.json").write_text(json.dumps({"rc": 1, "finished_at": time.time()}))
    (line,) = update_mod.background_refresh_lines()
    assert "FAILED" in line and "exit 1 (warning: mobile daemon did not restart: timed out" in line


def test_a_worker_stopped_at_its_bound_is_one_warning_with_its_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q3: the child's line is quoted inside the parent's own ``warning:`` sentence,
    so it carries no prefix of its own, and it names the in-flight daemon's command."""
    monkeypatch.delenv(update_mod._REFRESH_ROLE_ENV, raising=False)
    rc, _, err = _hand_off_in_process(
        tmp_path, _SLOW_RELAY_WORKER.format(sleep=30), unit="services", bound=1.0, handoff_in=8.0
    )
    assert rc == 1
    last = err.strip().splitlines()[-1]
    assert not last.startswith("warning:"), last
    assert "did not finish within 1s and was stopped" in last
    assert "`lop network install`" in last
    (run,) = update_mod._refresh_runs(tmp_path, "services")
    _reap_watchdog(run)
    completed = subprocess.CompletedProcess([], 1, stdout="", stderr=err)
    with (
        patch.object(update_mod, "_installed_daemon_plists", return_value=[Path("/tmp/x")]),
        patch("subprocess.run", return_value=completed),
    ):
        (warning,) = update_mod.refresh_service_daemons_after_upgrade().warnings
    assert warning.count("warning:") == 1, warning
