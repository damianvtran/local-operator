"""``lop services heavy {wait,run,status}`` (``local_operator/heavy_cli.py``).

Why these cells exist: before these verbs, sessions on mesh nodes coordinated
the machine-wide heavy-job slot (#4104) by peer messages and ``pgrep -f``
loops. The verbs are only worth having if they agree with the REAL lease, so
the cells that matter drive the real CLI as a separate OS process against a
real flock'd slot held by another process, and check the slot's state from
outside: held while the child runs, free after it exits or is signalled.
Every cell uses an isolated config dir, one slot, and short bounds.
"""

from __future__ import annotations

import io
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from local_operator import heavy_cli, heavy_lease

pytestmark = pytest.mark.skipif(not heavy_lease.supported(), reason="flock is POSIX-only")

ROOT = Path(__file__).resolve().parents[2]

#: Holds slot 0 of ``argv[1]`` for up to ``argv[3]`` seconds, then writes the
#: release time to ``argv[2]`` (a ``.held`` marker appears once the slot is
#: taken). It releases early once a ``.release`` file appears next to the log,
#: so a cell can free the slot on an observed event instead of a timer.
_HOLDER = textwrap.dedent("""
    import sys, time
    from pathlib import Path
    from local_operator import heavy_lease
    directory, log, hold = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
    release = log.parent / (log.name + ".release")
    while (lease := heavy_lease.try_acquire(directory, 1, command="pnpm build")) is None:
        time.sleep(0.02)
    (log.parent / (log.name + ".held")).write_text("1")
    deadline = time.monotonic() + hold
    while time.monotonic() < deadline and not release.exists():
        time.sleep(0.02)
    end = time.time()
    lease.release()
    log.write_text(f"{end}\\n")
""")


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, str]:
    config = tmp_path / "config"
    config.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    monkeypatch.setenv(heavy_lease.HEAVY_JOBS_SLOTS_ENV, "1")
    # English pinned: the cells assert the reused English lease messages.
    monkeypatch.setenv("LOP_I18N", "0")
    clean = {k: v for k, v in os.environ.items() if not k.startswith("CMUX_")}
    return {**clean, "PYTHONPATH": str(ROOT)}


def _lop(env: dict[str, str], *args: str, **kw: object) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-m", "local_operator.cli", "services", "heavy", *args],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        **kw,  # type: ignore[arg-type]
    )


def _hold(env: dict[str, str], log: Path, seconds: float) -> subprocess.Popen[bytes]:
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(heavy_lease.lease_dir()), str(log), str(seconds)],
        env=env,
        start_new_session=True,
    )
    _wait_for(lambda: log.with_name(log.name + ".held").exists())
    return proc


def _wait_for(condition, timeout: float = 15.0) -> None:  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition never became true"
        time.sleep(0.02)


def _reap(proc: subprocess.Popen) -> None:  # type: ignore[type-arg]
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGKILL)
    proc.wait(timeout=10)


def _slot_free() -> bool:
    lease = heavy_lease.try_acquire(heavy_lease.lease_dir(), 1)
    if lease is None:
        return False
    lease.release()
    return True


def _disable_queue() -> None:
    from local_operator import settings_io
    from local_operator.config import ConfigManager
    from local_operator.paths import config_dir

    settings_io.write_setting(
        ConfigManager(config_dir()), settings_io.BY_KEY["bash.heavy_jobs.enabled"], False
    )


# -- wait ---------------------------------------------------------------------


def test_wait_returns_once_a_held_slot_frees(env: dict[str, str], tmp_path: Path) -> None:
    log = tmp_path / "holder.log"
    # The slot is freed only AFTER the waiter has reported itself queued, never
    # on a fixed timer: a ``python -m local_operator.cli`` start costs ~0.7 s
    # idle and can pass 2 s on a loaded runner, so a 2 s hold could expire
    # before the waiter's first poll -- it then saw a free slot, printed
    # nothing and exited 0 (py3.12 shard 4, job 135487900771; 30/30 locally
    # with the cell pinned to one oversubscribed core). 60 s is only the
    # holder's backstop; the waiter's own --timeout 30 bounds the readline.
    holder = _hold(env, log, 60.0)
    try:
        waiter = _lop(env, "wait", "--timeout", "30")
        assert waiter.stderr is not None
        first = waiter.stderr.readline()  # the queued note prints on the first poll
        log.with_name(log.name + ".release").write_text("1")
        out, rest = waiter.communicate(timeout=40)
        err = first + rest
        returned = time.time()
        # Let the holder write its release time; _reap would SIGKILL it mid-write.
        holder.wait(timeout=30)
    finally:
        _reap(holder)
    assert waiter.returncode == 0, err
    assert "queued behind pid" in err and "(pnpm build)" in err
    # It returned AFTER the holder released, not on a guess.
    assert returned >= float(log.read_text())
    # And it did not take the slot on its way out.
    assert _slot_free()


def test_wait_times_out_non_zero_with_the_refusal(env: dict[str, str], tmp_path: Path) -> None:
    holder = _hold(env, tmp_path / "holder.log", 60.0)
    try:
        waiter = _lop(env, "wait", "--timeout", "1")
        _, err = waiter.communicate(timeout=30)
    finally:
        _reap(holder)
    assert waiter.returncode == heavy_cli.EXIT_TIMED_OUT
    assert "HEAVY JOB QUEUE FULL" in err and "(pnpm build)" in err


def test_wait_notes_every_five_seconds_not_every_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [0.0]
    out = io.StringIO()

    def sleep(seconds: float) -> None:
        clock[0] += seconds

    # Busy for 12 s of fake time, then free: notes at 0, 5 and 10 s only.
    holder = heavy_lease.Holder(slot=0, pid=1, session_id="", command="x", started_at=0.0)
    monkeypatch.setattr(heavy_cli, "_free_slots", lambda d, s: (clock[0] >= 12, [holder]))
    monkeypatch.setattr(heavy_cli, "_queue_off_line", lambda: None)
    rc = heavy_cli.wait_command(60, out=out, sleep=sleep, clock=lambda: clock[0])
    assert rc == 0
    assert out.getvalue().count("queued behind") == 3


# -- run ----------------------------------------------------------------------

#: Run as the wrapped command: reports, from inside the child, whether the slot
#: is free and what its record says about the holder.
_PROBE = textwrap.dedent("""
    import os, shlex, sys
    from local_operator import heavy_lease
    import time
    d = heavy_lease.lease_dir()
    taken = heavy_lease.try_acquire(d, 1)
    # The parent records the child's group just after spawning it.
    while (h := heavy_lease.read_holders(d)[0]).pgid is None:
        time.sleep(0.02)
    # The record names the program by its audit label, never the argv (#4264 Q3).
    cmd_ok = h.command == os.path.basename(sys.executable)
    print(f"free={taken is not None} pid={h.pid} parent={os.getppid()} "
          f"pgid_ok={h.pgid == os.getpgrp()} cmd_ok={cmd_ok}")
""")


def test_run_holds_the_slot_while_the_child_runs_then_frees_it(env: dict[str, str]) -> None:
    proc = _lop(env, "run", "--", sys.executable, "-c", _PROBE)
    out, err = proc.communicate(timeout=60)
    assert proc.returncode == 0, err
    fields = dict(item.split("=") for item in out.split())
    assert fields["free"] == "False"  # the child ran while holding the slot
    assert fields["pid"] == fields["parent"]  # the record names this lop process
    assert fields["pgid_ok"] == "True"  # ...and the child's own group
    assert fields["cmd_ok"] == "True"  # ...and the program's label
    assert _slot_free()


def test_run_propagates_the_exit_code(env: dict[str, str]) -> None:
    proc = _lop(env, "run", "--", "sh", "-c", "exit 7")
    proc.communicate(timeout=60)
    assert proc.returncode == 7
    assert _slot_free()


def test_slot_frees_even_when_the_command_leaves_a_process_behind(
    env: dict[str, str], tmp_path: Path
) -> None:
    # The leftover inherits the lock descriptor; only the explicit release
    # (LOCK_UN) frees the slot for every duplicate, not just closing ours.
    pid_file = tmp_path / "leftover.pid"
    proc = _lop(env, "run", "--", "sh", "-c", f"sleep 30 >/dev/null 2>&1 & echo $! > {pid_file}")
    proc.communicate(timeout=60)
    leftover = int(pid_file.read_text())
    try:
        assert proc.returncode == 0
        assert _slot_free()
    finally:
        os.kill(leftover, signal.SIGKILL)


def test_sigterm_releases_the_slot_and_ends_the_child(env: dict[str, str]) -> None:
    proc = _lop(env, "run", "--", "sleep", "60")
    try:
        _wait_for(lambda: any(h.pgid for h in heavy_lease.read_holders(heavy_lease.lease_dir())))
        (holder,) = heavy_lease.read_holders(heavy_lease.lease_dir())
        assert holder.pid == proc.pid
        child_group = holder.pgid
        assert child_group is not None
        os.kill(proc.pid, signal.SIGTERM)
        proc.communicate(timeout=30)
    finally:
        _reap(proc)
    assert proc.returncode == 128 + signal.SIGTERM
    assert _slot_free()
    with pytest.raises(ProcessLookupError):
        os.killpg(child_group, 0)


def test_run_times_out_without_running(env: dict[str, str], tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    holder = _hold(env, tmp_path / "holder.log", 60.0)
    try:
        proc = _lop(env, "run", "--timeout", "1", "--", "touch", str(marker))
        _, err = proc.communicate(timeout=30)
    finally:
        _reap(holder)
    assert proc.returncode == heavy_cli.EXIT_TIMED_OUT
    assert "HEAVY JOB QUEUE FULL" in err
    assert not marker.exists()


def test_run_without_a_command_is_a_usage_error_and_takes_no_slot(env: dict[str, str]) -> None:
    proc = _lop(env, "run")
    _, err = proc.communicate(timeout=60)
    assert proc.returncode == 2
    assert "put the command after `--`" in err
    assert not list(heavy_lease.lease_dir().glob("slot-*.json"))


def test_the_wrapper_is_not_itself_classified_heavy() -> None:
    # If the bash tool classified `lop services heavy run -- pnpm build` as
    # heavy it would take the slot first and the CLI would queue behind its own
    # parent until the bound ran out. Guard the classifier against that.
    assert heavy_lease.is_heavy("lop services heavy run -- pnpm build", None, cwd=None) is None


# -- status -------------------------------------------------------------------


def test_status_lists_each_holder(env: dict[str, str]) -> None:
    lease = heavy_lease.try_acquire(heavy_lease.lease_dir(), 1, command="make -j8")
    assert lease is not None
    out = io.StringIO()
    try:
        rc = heavy_cli.status_command(out=out, now=time.time() + 65)
    finally:
        lease.release()
    lines = out.getvalue().splitlines()
    assert rc == 0
    assert lines[0].startswith("heavy jobs: 1/1 slots held")
    assert lines[1] == f"  slot 0: pid {os.getpid()}, running 1m05s: make"  # label, not argv


# -- off and unsupported --------------------------------------------------------


def test_queue_off_wait_and_status_say_so_and_run_runs(env: dict[str, str], tmp_path: Path) -> None:
    _disable_queue()
    holder = _hold(env, tmp_path / "holder.log", 60.0)
    try:
        waiter = _lop(env, "wait", "--timeout", "30")
        _, wait_err = waiter.communicate(timeout=30)
        status = _lop(env, "status")
        status_out, _ = status.communicate(timeout=30)
        runner = _lop(env, "run", "--", "sh", "-c", "echo ran; exit 3")
        run_out, run_err = runner.communicate(timeout=30)
    finally:
        _reap(holder)
    off = "heavy jobs: queue off (bash.heavy_jobs.enabled=false)"
    assert waiter.returncode == 0 and off in wait_err
    assert status.returncode == 0 and status_out.strip() == off
    # Ran beside the held slot, unqueued, with the one-line note.
    assert runner.returncode == 3 and run_out.strip() == "ran" and off in run_err


def test_unsupported_platform_says_so(
    env: dict[str, str], monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(heavy_lease, "supported", lambda: False)
    line = "heavy jobs: queue unavailable on this platform"
    out = io.StringIO()
    assert heavy_cli.wait_command(5, out=out) == 0
    assert heavy_cli.status_command(out=out) == 0
    assert heavy_cli.run_command(["sh", "-c", "exit 4"], None, out=out) == 4
    assert out.getvalue().splitlines() == [line, line, line]


def _reload_heavy_cli_without(monkeypatch: pytest.MonkeyPatch, *missing: tuple[object, str]):
    """Reload ``heavy_cli`` with POSIX-only names removed, as on Windows.

    Regression for the ``xplat-probe-windows`` failure: a module-level
    ``signal.SIGHUP`` made the module unimportable there. The reload is undone
    (names restored, module reloaded again) so later cells see the real tuple.
    """
    import importlib

    for owner, name in missing:
        monkeypatch.delattr(owner, name)
    try:
        return tuple(importlib.reload(heavy_cli)._FORWARDED)
    finally:
        monkeypatch.undo()
        importlib.reload(heavy_cli)


def test_imports_without_sighup_and_does_not_forward_it(monkeypatch: pytest.MonkeyPatch) -> None:
    forwarded = _reload_heavy_cli_without(monkeypatch, (signal, "SIGHUP"))
    assert forwarded == (signal.SIGINT, signal.SIGTERM)


def test_without_killpg_nothing_is_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    # Windows shape: no SIGHUP and no process groups, so ``run`` must install
    # no handler that could reach ``os.killpg``.
    forwarded = _reload_heavy_cli_without(monkeypatch, (signal, "SIGHUP"), (os, "killpg"))
    assert forwarded == ()


def test_posix_forwarded_signals_are_unchanged() -> None:
    assert heavy_cli._FORWARDED == (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
