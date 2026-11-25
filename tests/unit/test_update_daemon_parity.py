"""Update parity: every background daemon moves, and a missed serve move retries.

Two defects measured 2026-11-17, each pinned here with stubs for the process
inventory and systemd so no test touches a live manager or daemon:

(A) On a Linux node ``lop update`` to 0.106.7 reported
    ``central_services_complete: true`` while the wakes supervisor and the
    secret broker ran 0.102.2 and the browser bridge 0.105.3 — the Linux step
    table carried the relay alone, and every other refresher answered
    ``unsupported`` (→ ``already``). ``lop services status`` read unit files,
    which name the stable shim, so it listed none of them.
(B) On a loaded Mac the serve move missed its 45 s budget and the update left
    serve on the old build until a manual ``lop services restart``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator import services, update

OLD = Path("/x/.local/share/lop/generations/20261113T022148Z-0.102.2")


class FakeSystemctl:
    """``systemctl --user`` with a scripted MainPID sequence and KillMode."""

    def __init__(self, pids: list[int], kill_mode: str = "process", restart_rc: int = 0) -> None:
        self.pids = list(pids)
        self.kill_mode = kill_mode
        self.restart_rc = restart_rc
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, *args: str, timeout: float = 30) -> Any:
        self.calls.append(args)
        if args[0] == "show":
            pid = self.pids[0] if self.pids else 0
            return SimpleNamespace(
                returncode=0, stdout=f"MainPID={pid}\nKillMode={self.kill_mode}\n", stderr=""
            )
        if args[0] == "restart":
            if self.restart_rc == 0 and len(self.pids) > 1:
                self.pids.pop(0)
            return SimpleNamespace(returncode=self.restart_rc, stdout="", stderr="boom")
        raise AssertionError(args)

    @property
    def restarts(self) -> int:
        return sum(1 for call in self.calls if call[0] == "restart")


def _move(
    fake: FakeSystemctl,
    stale_pids: set[int],
    *,
    keeps_children: bool,
    child_census: Any = lambda: 0,
    verify_of: Any = None,
) -> Any:
    return update._systemd_restart_if_build_moved(
        name="wakes supervisor",
        unit="local-operator-wakes.service",
        recovery="lop wake install",
        keeps_children=keeps_children,
        systemctl=fake,
        stale_of=lambda pid: OLD if pid in stale_pids else None,
        unit_installed=lambda unit: True,
        addressable=lambda unit: True,
        child_census=child_census,
        # The tri-state post-restart reader, consistent with ``stale_pids``.
        verify_of=verify_of or (lambda pid: pid in stale_pids),
    )


@pytest.fixture(autouse=True)
def _no_live_wakes_unit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep status off the developer's real user manager (``systemctl show``).

    ``update.wakes_unit_kill_mode`` reads the LOADED KillMode of the real
    wakes unit; ``None`` is "the gate does not apply", the macOS/no-unit
    answer. Tests about that gate inject their own reader.
    """
    monkeypatch.setattr(update, "wakes_unit_kill_mode", lambda: None)


# --------------------------------------------------------------------------- A


def test_a_stale_generation_daemon_is_restarted_onto_the_current_build() -> None:
    fake = FakeSystemctl([3106, 4200])
    outcome = _move(fake, {3106}, keeps_children=True)
    assert outcome.kind == "restarted"
    assert outcome.detail == OLD.name
    assert fake.restarts == 1
    assert update._plist_status(outcome.kind) == "refreshed"


def test_a_current_daemon_is_left_alone() -> None:
    fake = FakeSystemctl([4200])
    assert _move(fake, set(), keeps_children=False).kind == "current"
    assert fake.restarts == 0


def test_a_restart_that_brings_back_an_old_build_is_a_failure() -> None:
    fake = FakeSystemctl([3106, 4200])
    outcome = _move(fake, {3106, 4200}, keeps_children=False)
    assert outcome.kind == "failed"
    assert "not running the new build" in outcome.detail


def test_a_deferral_is_respected_and_reported_with_its_reason() -> None:
    """A unit that would stop its conversations is NOT restarted — and says so."""
    fake = FakeSystemctl([3106, 4200], kill_mode="control-group")
    outcome = _move(fake, {3106}, keeps_children=True)
    assert outcome.kind == "stale"
    assert fake.restarts == 0
    assert "would stop the conversations" in outcome.detail
    assert "still on an older build" in outcome.warning()
    # The child's machine line carries the reason through to the parent.
    status = update._plist_status(outcome.kind)
    assert status == "pending"
    line = update._daemon_report_line(outcome.name, status, outcome.detail)
    (parsed,) = update._parse_daemon_reports(line)
    assert parsed == update.DaemonStatus("wakes supervisor", "pending", outcome.detail)


def test_a_healthy_report_line_is_byte_identical_to_the_old_shape() -> None:
    assert update._daemon_report_line("tunnel", "already") == (
        'daemon_report: {"name":"tunnel","status":"already"}'
    )


def test_a_linux_step_table_carries_every_installed_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator import supervisors

    for unit in ("local-operator-wakes.service", "local-operator-browser.service"):
        (tmp_path / unit).write_text("[Service]\n")
    monkeypatch.setattr(supervisors, "systemd_unit_path", lambda unit: tmp_path / unit)
    monkeypatch.setattr(update, "_linux_relay_unit_installed", lambda: False)
    monkeypatch.setattr(update.sys, "platform", "linux")
    names = [name for name, _, _ in update._linux_refresh_steps()]
    # The secret broker's roll step is unconditional (it is not a unit).
    assert names == ["browser bridge", "wakes supervisor", "secret broker"]
    assert update._linux_restart_units_installed() is True


def _inventory_table() -> list[tuple[int, str]]:
    gen = "/h/.local/share/lop/generations"
    return [
        (
            3106,
            f"{gen}/20261113T022148Z-0.102.2/tools/local-operator/bin/python3 "
            "-m local_operator.wakes.supervisor",
        ),
        (
            3663,
            f"{gen}/20261113T022148Z-0.102.2/tools/local-operator/bin/python3 "
            "-m local_operator.secrets.brokerd",
        ),
        (
            3105,
            f"{gen}/20261117T152926Z-0.106.7/tools/local-operator/bin/python3 "
            "-P -m local_operator.cli serve",
        ),
        (
            36276,
            f"{gen}/20261117T103123Z-0.106.6/tools/local-operator/bin/python3 "
            "-P -m local_operator.session.runtime.process",
        ),
    ]


def _stale_inventory() -> list[services.StaleDaemon]:
    old = {3106, 3663, 36276}
    return services.stale_background_daemons(
        table=_inventory_table,
        staleness=lambda pid: pid in old,
        generation=lambda pid: OLD,
        wake_census=lambda: 0,
    )


def test_a_inventory_reads_running_processes_not_unit_files() -> None:
    found = _stale_inventory()
    # A runtime (36276) is a conversation, never a daemon; serve has its own rows.
    assert [(d.name, d.pid) for d in found] == [("wakes supervisor", 3106), ("secret broker", 3663)]
    assert all(d.generation == OLD.name for d in found)


def test_a_non_movable_daemon_is_stale_in_the_update_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statuses = (
        update.DaemonStatus("network relay", "refreshed"),
        # The refresher's own word was "already" (the old Linux "unsupported").
        update.DaemonStatus("wakes supervisor", "already"),
    )
    out = update._reconcile_with_processes(statuses, inventory=_stale_inventory)
    by_name = {s.name: s for s in out}
    assert by_name["network relay"].status == "refreshed"
    assert by_name["wakes supervisor"].status == "pending"
    assert by_name["secret broker"].status == "pending"
    assert "lop secret broker restart" in by_name["secret broker"].reason
    assert OLD.name in by_name["secret broker"].reason
    monkeypatch.setattr(update, "_live_serve_records", lambda: {}, raising=False)
    receipt = update._rolling_receipt(None, update.StageReport(daemons=out))
    assert receipt["central_services_complete"] is False
    assert (
        "secret broker is pending (still running" in receipt["central_services_incomplete_reason"]
    )
    rows = {row["name"]: row for row in receipt["services"] if "name" in row}
    assert rows["secret broker"]["state"] == "pending"
    assert "lop secret broker restart" in rows["secret broker"]["reason"]


def test_a_non_movable_daemon_is_stale_in_services_status(monkeypatch: pytest.MonkeyPatch) -> None:
    # The REAL inventory, with only its two readers stubbed: the ps table and
    # the per-pid generation comparison.
    old = {3106, 3663, 36276}
    monkeypatch.setattr(services, "_own_process_table", _inventory_table)
    monkeypatch.setattr(update, "generation_staleness_of_process", lambda pid: pid in old)
    monkeypatch.setattr(update, "generation_of_process", lambda pid: OLD)
    monkeypatch.setattr(update, "current_generation", lambda: OLD.parent / "current-gen")
    monkeypatch.setattr(update, "_wake_owned_runtime_count", lambda: 0)
    lines = services.status_lines()
    broker = [line for line in lines if line.startswith("STALE daemon: secret broker")]
    assert broker and "pid 3663" in broker[0] and OLD.name in broker[0]
    assert any(line.startswith("STALE daemon: wakes supervisor") for line in lines)


# --------------------------------------------------------------------------- B


def _record(pid: int, instance: str, version: str = "0.106.6") -> Any:
    return SimpleNamespace(pid=pid, instance_id=instance, version=version, port=8977, host="h")


def test_b_a_move_timeout_schedules_a_retry_and_reports_pending() -> None:
    clock = iter(float(n) for n in range(1000))
    asked: list[dict[str, Any]] = []

    def retry(**kwargs: Any) -> bool:
        asked.append(kwargs)
        return True

    out = services._await_relocations(
        [_record(3105, "i-1")],
        wait_s=3.0,
        sleep=lambda s: None,
        scan=lambda: [_record(3105, "i-1")],
        monotonic=lambda: next(clock),
        retry=retry,
        rolling=True,
    )
    assert asked == [{"rolling": True}]
    (refresh,) = out
    assert "move pending, retrying in the background" in refresh.warnings[0]
    assert "lop services restart" not in refresh.warnings[0]
    (status,) = refresh.statuses
    assert (status.status, status.reason) == ("pending", "move pending, retrying in the background")


def test_b_no_retry_keeps_the_manual_sentence() -> None:
    clock = iter(float(n) for n in range(1000))
    out = services._await_relocations(
        [_record(3105, "i-1")],
        wait_s=3.0,
        sleep=lambda s: None,
        scan=lambda: [_record(3105, "i-1")],
        monotonic=lambda: next(clock),
    )
    assert "next `lop services restart` will try again" in out[0].warnings[0]


def test_b_a_later_success_clears_the_retry() -> None:
    stale_reads = iter([[_record(3105, "i-1")], []])
    reloads: list[dict[str, Any]] = []
    said: list[str] = []
    rc = services.retry_serve_move(
        rolling=True,
        delays=(1.0, 2.0, 4.0),
        sleep=lambda s: None,
        stale=lambda: next(stale_reads),
        reload=lambda **kw: reloads.append(kw) or [],
        out=said.append,
    )
    assert rc == 0
    assert reloads == [{"rolling": True}]  # one attempt; the second read found it moved
    assert said[-1] == "serve move: every serve daemon is on the current build"


def test_b_retry_gives_up_bounded_and_names_the_recovery() -> None:
    reloads: list[dict[str, Any]] = []
    said: list[str] = []
    rc = services.retry_serve_move(
        delays=(1.0, 2.0),
        sleep=lambda s: None,
        stale=lambda: [_record(3105, "i-1")],
        reload=lambda **kw: reloads.append(kw) or [],
        out=said.append,
        err=said.append,
    )
    assert rc == 1 and len(reloads) == 2
    assert "run `lop services restart`" in said[-1]


def _fake_start(runs: list[dict[str, Any]]) -> Any:
    def start(**kwargs: Any) -> Path:
        runs.append(kwargs)
        root: Path = kwargs["runs_root"]
        root.mkdir(parents=True, exist_ok=True)
        run = root / f"serve-{update._run_stamp(time.time())}-x{len(runs)}"
        run.mkdir()
        spec = {
            "started_at": time.time(),
            "bound_s": kwargs["bound_s"],
            "watchdog_pid": os.getpid(),
        }
        (run / "spec.json").write_text(json.dumps(spec))
        return run

    return start


def test_b_schedule_starts_one_bounded_worker_and_status_tracks_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs: list[dict[str, Any]] = []
    invocation = lambda label, tail, module: ([label, "-m", module, *tail], None)  # noqa: E731
    root = tmp_path / "refresh"
    kwargs = dict(runs_root=root, start=_fake_start(runs), invocation=invocation)
    assert update.schedule_serve_move_retry(rolling=True, **kwargs) is True
    # A second miss while the first retry runs joins it rather than doubling it.
    assert update.schedule_serve_move_retry(rolling=True, **kwargs) is True
    assert len(runs) == 1
    assert runs[0]["unit"] == "serve"
    assert runs[0]["argv"][-3:] == ["local_operator.services", "--retry-serve-move", "--rolling"]
    assert runs[0]["bound_s"] == update._SERVE_RETRY_BACKGROUND_BOUND_S

    monkeypatch.setattr(update, "_refresh_runs_root", lambda: root)
    (pending,) = [line for line in update.background_refresh_lines() if "serve" in line]
    assert pending.startswith("serve move: pending, retrying in the background")

    run = update._refresh_runs(root, "serve")[-1]
    (run / "result.json").write_text(json.dumps({"rc": 0, "finished_at": time.time()}))
    (done,) = [line for line in update.background_refresh_lines() if "serve" in line]
    assert done.startswith("background serve refresh: finished") and done.endswith("ok")


def test_b_update_stage_hands_the_scheduler_to_the_serve_move(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def restart_services(**options: Any) -> list[Any]:
        seen.update(options)
        return []

    monkeypatch.setattr(services, "restart_services", restart_services)
    monkeypatch.setattr(update, "_services_refusal", lambda: None)
    monkeypatch.setattr(update, "_live_serve_records", lambda: {})
    monkeypatch.setattr(update, "_reconcile_with_processes", lambda statuses: statuses)
    update._services_stage()
    assert seen.get("schedule_retry") is update.schedule_serve_move_retry


def test_a_no_generation_layout_means_no_process_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pip/pipx host or a redirected HOME has no pointer: nothing is scanned."""
    monkeypatch.setattr(update, "current_generation", lambda: None)
    scanned: list[bool] = []
    assert services.stale_background_daemons(table=lambda: scanned.append(True) or []) == []
    assert scanned == []


# ------------------------------------------------------------- review round 1


def test_r1_1_a_wake_started_conversation_defers_the_wakes_restart() -> None:
    """KillMode=process is not enough: the supervisor's stop sweeps its runtimes."""
    fake = FakeSystemctl([3106, 4200], kill_mode="process")
    outcome = _move(fake, {3106}, keeps_children=True, child_census=lambda: 1)
    assert fake.restarts == 0
    assert outcome.kind == "stale"
    assert "1 wake-started conversation running" in outcome.detail
    assert update._plist_status(outcome.kind) == "pending"


def test_r1_1_an_uncountable_census_defers_like_a_positive_one() -> None:
    fake = FakeSystemctl([3106, 4200], kill_mode="process")
    outcome = _move(fake, {3106}, keeps_children=True, child_census=lambda: None)
    assert fake.restarts == 0
    assert outcome.kind == "stale"
    assert "could not be counted" in outcome.detail


def test_r1_1_only_the_wakes_unit_carries_a_child_census() -> None:
    """The bridge connector's stop signals nothing (see the KILLMODE block)."""
    assert set(update._UNIT_CHILD_CENSUS) == {update._WAKES_UNIT}
    assert update._UNIT_CHILD_CENSUS[update._WAKES_UNIT] is update._wake_owned_runtime_count


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process census")
def test_r1_1_a_live_wake_owned_runtime_survives_the_refresh(tmp_path: Path) -> None:
    """A REAL process carrying the wake-owner marker is counted and left running."""
    from local_operator.paths import CONFIG_DIR_ENV
    from local_operator.session.runtime.types import RUNTIME_MODULE
    from local_operator.wakes import owned_runtimes

    root = tmp_path / "lop-config"
    root.mkdir()
    env = {
        "PATH": os.environ.get("PATH", ""),
        owned_runtimes.OWNER_ENV: "r1-1-synthetic-token",
        CONFIG_DIR_ENV: str(root),
        "LOP_MOBILE_CHILD_RESUME": "r1-1-synthetic-session",
    }
    # ``-c`` runs the sleep; the trailing ``-m <runtime module>`` words are only
    # argv, which is exactly what the census's spawn-contract match reads.
    child = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(120)", "-m", RUNTIME_MODULE],
        env=env,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 10
        while owned_runtimes.count_owned(root) != 1 and time.monotonic() < deadline:
            time.sleep(0.05)  # until exec has replaced the forked argv
        fake = FakeSystemctl([3106, 4200], kill_mode="process")
        outcome = _move(
            fake,
            {3106},
            keeps_children=True,
            child_census=lambda: owned_runtimes.count_owned(root),
        )
        assert child.poll() is None, "the wake-owned runtime must still be running"
        assert fake.restarts == 0
        assert outcome.kind == "stale"
        assert "1 wake-started conversation running" in outcome.detail
        # The production census (every root) sees it too, so it would defer as well.
        unscoped = update._wake_owned_runtime_count()
        assert unscoped is not None and unscoped >= 1
        assert re.search(r"\b[1-9]\d* wake-started", update.wake_children_deferral(unscoped))
    finally:
        child.kill()
        child.wait(timeout=10)


def test_r1_2_an_unreadable_new_build_is_unverified_not_moved() -> None:
    fake = FakeSystemctl([3106, 4200])
    outcome = _move(fake, {3106}, keeps_children=False, verify_of=lambda pid: None)
    assert fake.restarts == 1
    assert outcome.kind == "unverified"
    assert update._plist_status(outcome.kind) == "pending"
    assert "MainPID 4200" in outcome.detail
    assert "unverified" in outcome.warning()
    assert outcome.summary() == ""


def test_r1_3_status_says_why_the_wakes_supervisor_was_left() -> None:
    def inventory(count: int | None) -> list[services.StaleDaemon]:
        return services.stale_background_daemons(
            table=_inventory_table,
            staleness=lambda pid: pid == 3106,
            generation=lambda pid: OLD,
            wake_census=lambda: count,
        )

    (busy,) = inventory(2)
    assert busy.reason == update.wake_children_deferral(2)
    assert "2 wake-started conversations running" in busy.reason
    (idle,) = inventory(0)
    assert "`lop services restart` moves it now" in idle.reason


def _failed_serve_run(root: Path) -> None:
    run = root / f"serve-{update._run_stamp(time.time())}-x1"
    run.mkdir(parents=True)
    (run / "spec.json").write_text(json.dumps({"started_at": time.time(), "bound_s": 1.0}))
    (run / "result.json").write_text(json.dumps({"rc": 1, "finished_at": time.time()}))


def test_r1_4_a_later_manual_move_clears_a_failed_serve_retry_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "refresh"
    _failed_serve_run(root)
    monkeypatch.setattr(update, "_refresh_runs_root", lambda: root)

    monkeypatch.setattr(services, "stale_serve_daemons", lambda records=None: [object()])
    (failed,) = [line for line in update.background_refresh_lines() if "serve" in line]
    assert "FAILED" in failed

    # No stamp to compare against: not a proof of a move, so the failure stays.
    monkeypatch.setattr(services, "stale_serve_daemons", lambda records=None: None)
    assert any("FAILED" in line for line in update.background_refresh_lines())

    # ``lop services restart`` moved serve and wrote no run record.
    monkeypatch.setattr(services, "stale_serve_daemons", lambda records=None: [])
    assert [line for line in update.background_refresh_lines() if "serve" in line] == []


def test_the_daemon_status_line_names_only_a_proven_stale_daemon_of_that_name() -> None:
    """`lop mobile status` / `lop tunnel status` reuse the census (2026-11-18).

    The field finding: both daemons sat on the old build after an upgrade and
    neither status said so. The line is the census filtered to ONE name, so it
    cannot disagree with `lop services status`; a name the census did not prove
    stale prints nothing.
    """
    rows = [
        services.StaleDaemon(
            name="tunnel", pid=1206, generation="20261113T090000Z-old", reason="run x"
        )
    ]

    line = services.stale_daemon_status_line("tunnel", "lop tunnel restart", daemons=rows)

    assert "20261113T090000Z-old" in line
    assert "lop tunnel restart" in line
    assert services.stale_daemon_status_line("mobile", "lop mobile restart", daemons=rows) == ""
    assert services.stale_daemon_status_line("tunnel", "lop tunnel restart", daemons=[]) == ""


def _install_check(
    fake: FakeSystemctl,
    *,
    stale_pids: set[int] = frozenset(),  # type: ignore[assignment]
    verdict: Any = None,
    restart_when_unproven: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    """``systemd_install_build_line`` against a fake user manager, never a live one."""
    from local_operator import supervisors

    monkeypatch.setattr(supervisors, "systemd_unit_path", lambda unit: Path(__file__))
    monkeypatch.setattr(supervisors, "systemd_unit_is_addressable", lambda unit: True)
    return update.systemd_install_build_line(
        name="mobile",
        unit="local-operator-mobile.service",
        recovery="lop mobile restart",
        restart_when_unproven=restart_when_unproven,
        systemctl=fake,
        stale_of=lambda pid: OLD if pid in stale_pids else None,
        verdict_of=verdict or (lambda pid: pid in stale_pids),
    )


def test_install_check_restarts_a_proven_stale_unit_with_keyed_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The proven path is #4175's mover; the line is the shared keyed step (R1-2)."""
    from local_operator.i18n.keys import cli_daemon_build as _k
    from local_operator.i18n.text import tr

    fake = FakeSystemctl([3106, 4200])
    check = _install_check(
        fake,
        stale_pids={3106},
        verdict=lambda pid: None if pid == 4200 else pid == 3106,
        restart_when_unproven=False,
        monkeypatch=monkeypatch,
    )
    assert (check.kind, check.generation) == ("restarted", OLD.name)
    assert check.step() == tr("cli.daemon_build")(_k.install_restarted(generation=OLD.name))
    assert fake.restarts == 1


def test_install_check_a_proven_current_unit_is_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeSystemctl([4200])
    check = _install_check(
        fake, verdict=lambda pid: False, restart_when_unproven=True, monkeypatch=monkeypatch
    )
    assert (check.kind, check.step(), fake.restarts) == ("current", "", 0)


def test_install_check_unknown_build_restarts_mobile_onto_a_new_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R1-1: the mover reads "unreadable" as current; the wrapper must not."""
    fake = FakeSystemctl([777, 900])
    check = _install_check(
        fake, verdict=lambda pid: None, restart_when_unproven=True, monkeypatch=monkeypatch
    )
    assert check.kind == "restarted-unconfirmed"
    assert check.step()
    assert fake.restarts == 1


def test_install_check_unknown_build_restart_without_a_new_pid_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeSystemctl([777])  # restart "succeeds" but MainPID never changes
    check = _install_check(
        fake, verdict=lambda pid: None, restart_when_unproven=True, monkeypatch=monkeypatch
    )
    assert check.kind == "failed"
    assert "lop mobile restart" in check.step()


def test_install_check_unknown_build_is_named_not_bounced_for_tunnel_and_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_operator.i18n.keys import cli_daemon_build as _k
    from local_operator.i18n.text import tr

    fake = FakeSystemctl([777, 900])
    check = _install_check(
        fake, verdict=lambda pid: None, restart_when_unproven=False, monkeypatch=monkeypatch
    )
    assert check.kind == "unconfirmed"
    assert check.step() == tr("cli.daemon_build")(
        _k.install_unconfirmed(command="lop mobile restart")
    )
    assert fake.restarts == 0


def test_install_check_a_refused_proven_restart_fails_the_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeSystemctl([3106, 4200], restart_rc=1)
    check = _install_check(
        fake, stale_pids={3106}, restart_when_unproven=False, monkeypatch=monkeypatch
    )
    assert check.kind == "failed"


def test_install_check_never_asks_the_mover_to_keep_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#4175's mover is reused unchanged and never told to spare children."""
    from local_operator import launchd

    seen: list[dict[str, object]] = []

    def mover(**kwargs: object) -> launchd.PlistRefresh:
        seen.append(kwargs)
        return launchd.PlistRefresh(name="tunnel", kind="restarted", detail="g-old")

    monkeypatch.setattr(update, "_systemd_restart_if_build_moved", mover)
    check = update.systemd_install_build_line(
        name="tunnel",
        unit="lop-tunnel.service",
        recovery="lop tunnel restart",
        restart_when_unproven=False,
        systemctl=FakeSystemctl([1]),
    )
    assert (check.kind, check.generation) == ("restarted", "g-old")
    assert seen[0]["keeps_children"] is False


# ----------------------------------------------- #4175 deferred follow-ups


def _census_run(table: str, rows: dict[int, str]) -> Any:
    """A ``ps`` runner: ``-eo`` answers ``table``, ``-p <pid>`` answers ``rows[pid]``."""

    def run(command: Any, timeout_s: float) -> str:
        if "-eo" in command:
            return table
        if "-p" in command:
            return rows.get(int(command[command.index("-p") + 1]), "")
        return ""

    return run


def _runtime_line(pid: int) -> str:
    from local_operator.session.runtime.types import RUNTIME_MODULE

    return f"{pid} 1 01:00 00:00:01 /usr/bin/python3 -m {RUNTIME_MODULE}"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process census")
def test_r2_1_an_alive_runtime_whose_probe_fails_makes_the_census_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Alive but unreadable is not "zero owned": the census answers ``None``.

    The pid is THIS test process, so ``kill(pid, 0)`` really says alive; the
    per-pid ``ps -p`` answers ``""`` as the runner does on a timeout.
    """
    from local_operator.session.runtime import reclaim
    from local_operator.wakes import owned_runtimes

    me = os.getpid()
    run = _census_run(_runtime_line(me) + "\n", {})
    monkeypatch.setattr(reclaim, "_run_command", run)
    assert owned_runtimes.count_owned(None) is None
    # The gate this feeds defers on it rather than restarting.
    fake = FakeSystemctl([3106, 4200], kill_mode="process")
    outcome = _move(
        fake, {3106}, keeps_children=True, child_census=update._wake_owned_runtime_count
    )
    assert fake.restarts == 0 and outcome.kind == "stale"
    assert "could not be counted" in outcome.detail


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process census")
def test_r2_1_an_alive_runtime_with_no_readable_environment_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_operator.session.runtime import reclaim
    from local_operator.wakes import owned_runtimes

    me = os.getpid()
    run = _census_run(_runtime_line(me) + "\n", {me: _runtime_line(me)})
    monkeypatch.setattr(reclaim, "_run_command", run)
    monkeypatch.setattr(reclaim, "pid_environment", lambda pid, **_: "")
    assert owned_runtimes.count_owned(None) is None


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process census")
def test_r2_1_a_runtime_that_exited_or_belongs_to_another_user_is_not_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a pid our stop could still reach holds the restart off."""
    from local_operator.session.runtime import reclaim
    from local_operator.wakes import owned_runtimes

    gone, foreign = 2_000_001, 2_000_002
    table = _runtime_line(gone) + "\n" + _runtime_line(foreign) + "\n"
    monkeypatch.setattr(reclaim, "_run_command", _census_run(table, {}))

    def kill(pid: int, sig: int) -> None:
        assert sig == 0
        raise ProcessLookupError if pid == gone else PermissionError

    monkeypatch.setattr(owned_runtimes.os, "kill", kill)
    assert owned_runtimes.count_owned(None) == 0


def _running_serve_run(root: Path) -> None:
    run = root / f"serve-{update._run_stamp(time.time())}-x1"
    run.mkdir(parents=True)
    (run / "spec.json").write_text(json.dumps({"started_at": time.time(), "bound_s": 1500.0}))


def test_q1_a_running_retry_is_not_pending_once_serve_has_moved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "refresh"
    _running_serve_run(root)
    monkeypatch.setattr(update, "_refresh_runs_root", lambda: root)
    monkeypatch.setattr(update, "_run_abandoned", lambda run: False)

    monkeypatch.setattr(services, "stale_serve_daemons", lambda records=None: [object()])
    (pending,) = [line for line in update.background_refresh_lines() if "serve" in line]
    assert pending.startswith("serve move: pending, retrying in the background")

    # No build stamp: not a proof of a move, so the retry row stays.
    monkeypatch.setattr(services, "stale_serve_daemons", lambda records=None: None)
    assert any("pending" in line for line in update.background_refresh_lines())

    # The retry's attempt moved serve; the worker has not written its result yet.
    monkeypatch.setattr(services, "stale_serve_daemons", lambda records=None: [])
    assert [line for line in update.background_refresh_lines() if "serve" in line] == []


def test_q2_status_names_the_killmode_for_a_control_group_wakes_unit() -> None:
    def inventory(count: int | None, kill_mode: str | None) -> services.StaleDaemon:
        (row,) = services.stale_background_daemons(
            table=_inventory_table,
            staleness=lambda pid: pid == 3106,
            generation=lambda pid: OLD,
            wake_census=lambda: count,
            wake_kill_mode=lambda: kill_mode,
        )
        return row

    busy = inventory(2, "control-group")
    assert "KillMode=control-group" in busy.reason
    assert "`lop wake install`" in busy.reason
    assert "moves when they end" not in busy.reason
    # Byte-identical to what the update report says for the same unit.
    fake = FakeSystemctl([3106, 4200], kill_mode="control-group")
    report = _move(fake, {3106}, keeps_children=True, child_census=lambda: 2)
    assert fake.restarts == 0
    assert report.detail == f"left on {OLD.name}: {busy.reason}"
    # An unanswered manager is refused as unknown, as the gate refuses it.
    assert "KillMode=unknown" in inventory(0, "").reason
    # KillMode=process (and no unit at all) keep the census sentence.
    assert inventory(2, "process").reason == update.wake_children_deferral(2)
    assert inventory(2, None).reason == update.wake_children_deferral(2)


# ------------------------------------------------- daemon freshness disclosure


def _emit(stage: update.StageReport, monkeypatch: pytest.MonkeyPatch, capsys: Any) -> Any:
    # No ``rolling`` block: this pins the base payload, not the generation receipt.
    monkeypatch.setattr(update, "_is_generation_install", lambda _root=None: False)
    update._emit_update_report(install_version="0.2.0", target="0.2.0", stage=stage)
    captured = capsys.readouterr()
    return captured, json.loads(captured.out.strip().splitlines()[-1])["update_report"]


def test_freshness_a_stale_daemon_cannot_render_as_plain_success(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A row the census proved stale must surface its remedy, in both halves.

    Before this, the reconciled ``pending`` row was dropped from ``daemons[]``
    and nothing was printed: the run read as a clean success over a relay
    still on the old generation.
    """
    statuses = (
        update.DaemonStatus("network relay", "already"),
        update.DaemonStatus("tunnel", "refreshed"),
    )
    stale = lambda: [  # noqa: E731
        services.StaleDaemon("network relay", 4242, OLD.name, "run `lop network restart`")
    ]
    stage = update.StageReport(daemons=update._reconcile_with_processes(statuses, inventory=stale))
    captured, report = _emit(stage, monkeypatch, capsys)
    # The frozen keys did not move: ``ok`` still means "the install succeeded".
    assert report["ok"] is True and report["schema"] == "update.report.v1"
    assert report["daemons_complete"] is False
    (row,) = report["daemons_incomplete"]
    assert row["name"] == "network relay" and row["status"] == "pending"
    assert "lop network restart" in row["remedy"]
    assert "still on the old build" in captured.err
    assert "network relay:" in captured.err and "`lop network restart`" in captured.err
    assert "tunnel" not in captured.err


def test_freshness_a_reasonless_failed_row_takes_the_census_remedy(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No second command table: a failed row borrows ``_DAEMON_MODULES``'s sentence."""
    stage = update.StageReport(
        daemons=(
            update.DaemonStatus("mobile", "failed"),
            update.DaemonStatus("service daemons", "failed"),
        )
    )
    captured, report = _emit(stage, monkeypatch, capsys)
    remedies = {row["name"]: row["remedy"] for row in report["daemons_incomplete"]}
    assert remedies["mobile"] == services.daemon_remedy("mobile") == "run `lop mobile restart`"
    # Unknown to the census: no invented command, the human row points at status.
    assert remedies["service daemons"] == ""
    assert "service daemons: run `lop services status`" in captured.err


def test_freshness_a_reasonless_unsupervised_row_gets_no_per_daemon_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An ``unsupervised`` row's own warning already names the fix.

    Regression (merge gate run 45523620065): the census remedy for ``mobile``
    is ``lop mobile restart``, which is launchd-only and wrong for a daemon
    running outside its supervisor — ``test_update_command_unsupervised_warns``
    pins that it never appears. A reasonless ``pending`` row, which had no
    earlier warning, still takes the generic pointer at ``lop services status``.

    Owner constraint (PR #4562): the summary must not add a THIRD instruction
    beside the warning's ``lop mobile serve`` — so the unsupervised row prints
    no per-daemon line at all, while ``daemons_incomplete`` keeps it truthfully.
    """
    stage = update.StageReport(
        daemons=(
            update.DaemonStatus("mobile", "unsupervised"),
            update.DaemonStatus("network relay", "pending"),
        )
    )
    captured, report = _emit(stage, monkeypatch, capsys)
    rows = {row["name"]: row for row in report["daemons_incomplete"]}
    assert rows["mobile"] == {"name": "mobile", "status": "unsupervised", "remedy": ""}
    assert rows["network relay"]["remedy"] == ""
    assert report["daemons_complete"] is False
    assert "lop mobile restart" not in captured.err
    assert "lop network restart" not in captured.err
    assert "mobile:" not in captured.err
    assert "network relay: run `lop services status`" in captured.err


def test_freshness_an_unsupervised_mobile_summary_repeats_only_the_warning(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End to end through the refresher's warning AND the summary (owner, PR #4562).

    The warning names ``lop mobile serve``; the summary must neither contradict
    it (``lop mobile restart``) nor add a per-daemon ``mobile:`` instruction.
    """
    refresh = update._mobile_daemon_refresh(update.MobileRefresh(kind="unsupervised"))
    for line in refresh.warnings:
        print(line, file=sys.stderr)
    stage = update.StageReport(daemons=refresh.statuses)
    update._print_incomplete_daemons(update._incomplete_daemons(stage))
    err = capsys.readouterr().err
    assert "lop mobile serve" in err
    assert "lop mobile restart" not in err
    assert "mobile:" not in err
    assert "lop services status" not in err


def test_freshness_a_fully_refreshed_report_still_reads_clean(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    statuses = (
        update.DaemonStatus("network relay", "refreshed"),
        update.DaemonStatus("tunnel", "already"),
    )
    stage = update.StageReport(daemons=update._reconcile_with_processes(statuses, inventory=list))
    captured, report = _emit(stage, monkeypatch, capsys)
    assert captured.err == ""
    assert "daemons_complete" not in report and "daemons_incomplete" not in report
    assert captured.out == (
        '{"update_report":{"schema":"update.report.v1","ok":true,"install_version":"0.2.0",'
        '"target":"0.2.0","serve":[],"daemons":[{"name":"network relay","status":"refreshed",'
        '"version":"0.2.0"},{"name":"tunnel","status":"already","version":"0.2.0"}]}}\n'
    )
