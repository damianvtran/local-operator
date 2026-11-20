"""The heavy-job lease is re-entrant for a holder's descendants, and only for them.

Why these cells exist (2026-11-20): the bash tool leased a whole line because it
contained ``go test -race`` and passed the lock descriptor to it; a ``lop
services heavy run --timeout 3300`` INSIDE that line then queued for the slot its
own ancestor held. With one slot on the host it waited out its whole timeout,
and every other heavy command on the machine was refused at 900 s. The holder
was alive the whole time, so liveness or pid-reuse checks could not see it.

The fix exports ``LOCAL_OPERATOR_HEAVY_LEASE=<slot>:<fd>`` beside the inherited
descriptor and lets ``heavy run``/``wait`` and ``acquire`` run under the
ancestor's slot once that descriptor is PROVED to be the holder's own lock
(``heavy_lease.inherited_lease``). The cells that matter therefore use real OS
processes and real flocks (nothing here mocks ``flock``): a leased bash-tool
command spawning the real CLI, a forged marker queueing behind a real holder,
an unrelated process still excluded, and the slot freed afterwards. Every cell
uses an isolated config dir, one slot, short bounds, and reaps what it spawns.
Module-level code names no new API, so the incident cells also run (and fail)
against the pre-fix tree as the base reproduction.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator import heavy_audit, heavy_lease

pytestmark = pytest.mark.skipif(not heavy_lease.supported(), reason="flock is POSIX-only")

ROOT = Path(__file__).resolve().parents[2]
MARKER = "LOCAL_OPERATOR_HEAVY_LEASE"
EXIT_TIMED_OUT = 75

#: Holds slot 0 of ``argv[1]`` until a ``<argv[2]>.release`` file appears (or
#: ``argv[3]`` seconds pass); writes ``<argv[2]>.held`` once it holds it.
_HOLDER = textwrap.dedent("""
    import sys, time
    from pathlib import Path
    from local_operator import heavy_lease
    directory, log, hold = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
    while (lease := heavy_lease.try_acquire(directory, 1, command="pnpm build")) is None:
        time.sleep(0.02)
    (log.parent / (log.name + ".held")).write_text("1")
    deadline = time.monotonic() + hold
    while time.monotonic() < deadline and not (log.parent / (log.name + ".release")).exists():
        time.sleep(0.02)
    lease.release()
""")


@pytest.fixture
def lease_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, str]:
    """Isolated config, ONE slot, a 1 s bash-tool queue; returns a child env."""
    config = tmp_path / "config"
    config.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    monkeypatch.setenv(heavy_lease.HEAVY_JOBS_SLOTS_ENV, "1")
    # 1 s is the smallest accepted bound; the bash tool's own queue refuses fast.
    monkeypatch.setenv(heavy_lease.HEAVY_JOBS_WAIT_SECONDS_ENV, "1")
    monkeypatch.setenv("LOP_I18N", "0")
    monkeypatch.delenv(MARKER, raising=False)
    clean = {k: v for k, v in os.environ.items() if not k.startswith("CMUX_")}
    return {**clean, "PYTHONPATH": str(ROOT)}


def _cli(*args: str) -> str:
    """A shell-quoted ``lop services heavy …`` that cannot reach the real config.

    The isolation is spelled ON the line, not only inherited, so even a bash env
    policy that dropped variables could not send the nested CLI to the
    machine's real lease directory.
    """
    pinned = (
        f"LOCAL_OPERATOR_CONFIG_DIR={shlex.quote(os.environ['LOCAL_OPERATOR_CONFIG_DIR'])}"
        f" {heavy_lease.HEAVY_JOBS_SLOTS_ENV}=1 LOP_I18N=0 PYTHONPATH={shlex.quote(str(ROOT))}"
    )
    argv = [sys.executable, "-m", "local_operator.cli", "services", "heavy", *args]
    return f"{pinned} {shlex.join(argv)}"


def _lop(env: dict[str, str], *args: str, **kw: Any) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-m", "local_operator.cli", "services", "heavy", *args],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        **kw,
    )


def _hold(env: dict[str, str], log: Path) -> subprocess.Popen[bytes]:
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(heavy_lease.lease_dir()), str(log), "60"],
        env=env,
        start_new_session=True,
    )
    _wait_for(lambda: log.with_name(log.name + ".held").exists())
    return proc


def _wait_for(condition: Any, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition never became true"
        time.sleep(0.02)


def _reap(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.wait(timeout=10)


def _slot_free() -> bool:
    lease = heavy_lease.try_acquire(heavy_lease.lease_dir(), 1)
    if lease is None:
        return False
    lease.release()
    return True


def _text(result: Any) -> str:
    return "".join(getattr(block, "text", "") for block in result.content)


async def _until_free(timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not _slot_free():
        assert time.monotonic() < deadline, "slot never released"
        await asyncio.sleep(0.05)


#: ``false && go test ./...`` makes the bash tool classify the line heavy (the
#: incident's ``go test``) without ever running go, so the call takes the slot.
_HEAVY_TAIL = "; false && go test ./..."


# -- the incident: a leased bash command running the CLI ------------------------


def test_heavy_run_inside_a_leased_bash_command_runs_at_once(
    lease_env: dict[str, str], tmp_path: Path
) -> None:
    """(a) + (d) + (e): the 2026-11-20 shape, end to end, with one slot.

    The real ``execute_bash`` leases ``… heavy run -- <inner>; false && go
    test``. The nested ``heavy run`` must start its child at once under the
    inherited slot (before the fix it queued behind its own ancestor until its
    ``--timeout`` and exited 75). While the inner command runs, an UNRELATED
    ``heavy run`` with no inherited lock is still refused, and the refusal
    names the holder by its heavy segment (``go test``), not ``cd``/``rm``.
    Once the bash command ends the slot is free and its record gone.
    """
    from local_operator.tools import builtin

    held, release = tmp_path / "inner.held", tmp_path / "inner.release"
    inner = (
        f"touch {shlex.quote(str(held))};"
        f" while [ ! -e {shlex.quote(str(release))} ]; do sleep 0.05; done"
    )
    command = (
        f"cd {shlex.quote(str(tmp_path))} && rm -rf probe && "
        + _cli("run", "--timeout", "10", "--", "sh", "-c", inner)
        + "; echo nested-rc=$?"
        + _HEAVY_TAIL
    )
    updates: list[Any] = []

    async def scenario() -> tuple[Any, subprocess.CompletedProcess[str]]:
        task = asyncio.ensure_future(
            builtin.execute_bash(
                "t", {"command": command, "timeout": 60}, None, updates.append, None
            )
        )
        try:
            deadline = time.monotonic() + 20
            # Before the fix the inner command never starts: the nested run
            # queues for 10 s, so this deadline would trip first only if the
            # whole interpreter start took longer than that.
            while not held.exists():
                assert not task.done(), _text(task.result())
                assert time.monotonic() < deadline, "the nested run never started its child"
                await asyncio.sleep(0.05)
            # (d) an unrelated process is still excluded while the slot is held.
            unrelated = await asyncio.to_thread(
                subprocess.run,
                [
                    sys.executable,
                    "-m",
                    "local_operator.cli",
                    "services",
                    "heavy",
                    "run",
                    "--timeout",
                    "1",
                    "--",
                    "true",
                ],
                env=lease_env,
                capture_output=True,
                text=True,
                timeout=60,
            )
        finally:
            release.write_text("1")
        result = await task
        await _until_free()
        return result, unrelated

    result, unrelated = asyncio.run(scenario())
    text = _text(result)
    assert not result.is_error, text
    assert "nested-rc=0" in text, text
    assert "running under heavy-job slot 0, already held by an ancestor" in text, text
    assert unrelated.returncode == EXIT_TIMED_OUT, unrelated.stderr
    assert "HEAVY JOB QUEUE FULL" in unrelated.stderr
    # The holder is labelled by the segment that made the line heavy.
    assert "(go test)" in unrelated.stderr, unrelated.stderr
    # (e) released after the parent ended, and no record left behind.
    assert not (heavy_lease.lease_dir() / "slot-0.json").exists()
    events = heavy_audit.tail(heavy_lease.lease_dir(), 50)
    # ONE take/release pair for the whole line; the nested run is a `nested`
    # row inside it and the unrelated run a `timeout`. (_until_free's own
    # probe appends a later take/release, hence the prefix.)
    assert [e["event"] for e in events][:4] == ["take", "nested", "timeout", "release"], events
    take, nested = events[0], events[1]
    assert take["label"] == "go test" and take["pid"] == os.getpid()
    assert nested["slot"] == 0 and nested["holder"] == os.getpid() and nested["label"] == "sh"


def test_heavy_wait_inside_a_leased_bash_command_returns_at_once(lease_env: dict[str, str]) -> None:
    """(b): ``heavy wait`` under its own ancestor's slot returns 0 at once."""
    from local_operator.tools import builtin

    command = _cli("wait", "--timeout", "10") + "; echo wait-rc=$?" + _HEAVY_TAIL

    async def scenario() -> Any:
        started = time.monotonic()
        result = await builtin.execute_bash(
            "t", {"command": command, "timeout": 60}, None, None, None
        )
        await _until_free()
        return result, time.monotonic() - started

    result, elapsed = asyncio.run(scenario())
    text = _text(result)
    assert "wait-rc=0" in text and "already held by an ancestor" in text, text
    # Structural, not a race: queued, it would have taken its full 10 s bound.
    assert elapsed < 10, elapsed
    nested = [e for e in heavy_audit.tail(heavy_lease.lease_dir(), 50) if e["event"] == "nested"]
    assert len(nested) == 1 and nested[0]["label"] == "heavy wait", nested


def test_run_passes_the_slot_down_through_nested_runs(lease_env: dict[str, str]) -> None:
    """``heavy run`` exports the marker too, and a nested run passes it on.

    ``run -- sh -c 'run -- wait'``: the outer run takes the slot, the middle
    one runs under it (InheritedLease), and the innermost ``wait`` still
    verifies, which needs the middle run to have passed the fd AND the marker.
    """
    inner = _cli("run", "--timeout", "10", "--", *shlex.split(_cli("wait", "--timeout", "10"))[4:])
    proc = _lop(lease_env, "run", "--timeout", "10", "--", "sh", "-c", inner)
    try:
        _, err = proc.communicate(timeout=90)
    finally:
        _reap(proc)
    assert proc.returncode == 0, err
    assert err.count("already held by an ancestor") == 2, err
    assert _slot_free()


# -- a forged or stale marker is never authority --------------------------------


def _forged(env: dict[str, str], marker: str, **kw: Any) -> subprocess.Popen[str]:
    return _lop({**env, MARKER: marker}, "wait", "--timeout", "1", **kw)


def test_marker_without_an_inherited_descriptor_still_queues(
    lease_env: dict[str, str], tmp_path: Path
) -> None:
    """(c) bare env var: no such fd in the child, so it queues and is refused."""
    holder = _hold(lease_env, tmp_path / "holder.log")
    try:
        waiter = _forged(lease_env, "0:987")
        _, err = waiter.communicate(timeout=60)
        runner = _lop(
            {**lease_env, MARKER: "0:987"},
            "run",
            "--timeout",
            "1",
            "--",
            "touch",
            str(tmp_path / "ran"),
        )
        _, run_err = runner.communicate(timeout=60)
    finally:
        (tmp_path / "holder.log.release").write_text("1")
        _reap(holder)
    assert waiter.returncode == EXIT_TIMED_OUT, err
    assert "HEAVY JOB QUEUE FULL" in err and "ancestor" not in err
    assert runner.returncode == EXIT_TIMED_OUT, run_err
    assert not (tmp_path / "ran").exists()


def test_marker_naming_an_fd_to_another_file_still_queues(
    lease_env: dict[str, str], tmp_path: Path
) -> None:
    """(c) an fd that IS inherited but is some other file: inode mismatch."""
    holder = _hold(lease_env, tmp_path / "holder.log")
    other = tmp_path / "decoy.lock"
    other.write_text("")
    fd = os.open(other, os.O_RDONLY)
    try:
        waiter = _forged(lease_env, f"0:{fd}", pass_fds=(fd,))
        _, err = waiter.communicate(timeout=60)
    finally:
        os.close(fd)
        (tmp_path / "holder.log.release").write_text("1")
        _reap(holder)
    assert waiter.returncode == EXIT_TIMED_OUT, err
    assert "HEAVY JOB QUEUE FULL" in err


def test_marker_naming_a_fresh_open_of_the_real_lock_still_queues(
    lease_env: dict[str, str], tmp_path: Path
) -> None:
    """(c) the strongest forgery: the right file, opened separately.

    Same device and inode as ``slot-0.lock`` and a live holder record, so only
    the flock probe can tell it apart: it is a different open file description
    and the real holder's lock refuses it.
    """
    holder = _hold(lease_env, tmp_path / "holder.log")
    fd = os.open(heavy_lease.lease_dir() / "slot-0.lock", os.O_RDONLY)
    try:
        waiter = _forged(lease_env, f"0:{fd}", pass_fds=(fd,))
        _, err = waiter.communicate(timeout=60)
    finally:
        os.close(fd)
        (tmp_path / "holder.log.release").write_text("1")
        _reap(holder)
    assert waiter.returncode == EXIT_TIMED_OUT, err
    assert "HEAVY JOB QUEUE FULL" in err


@pytest.mark.parametrize("value", ["", "abc", "0:", ":3", "-1:3", "0:3:4", "0:99999999999", "1:3"])
def test_malformed_or_wrong_slot_markers_are_ignored(lease_env: dict[str, str], value: str) -> None:
    lease = heavy_lease.try_acquire(heavy_lease.lease_dir(), 1)
    assert lease is not None
    try:
        assert heavy_lease.inherited_lease(heavy_lease.lease_dir(), {MARKER: value}) is None
    finally:
        lease.release()


def test_a_released_lease_is_not_inherited_and_is_not_retaken(lease_env: dict[str, str]) -> None:
    """A marker outliving its lease (released, fd still open) must not re-lock.

    ``Lease.release`` unlinks the record before unlocking, so the missing
    record refuses the marker BEFORE any flock probe; the slot stays free.
    """
    directory = heavy_lease.lease_dir()
    lease = heavy_lease.try_acquire(directory, 1)
    assert lease is not None
    # A second descriptor on the SAME description, as a child would inherit it.
    fd = os.dup(lease.fileno())  # type: ignore[arg-type]
    try:
        lease.release()
        assert heavy_lease.inherited_lease(directory, {MARKER: f"0:{fd}"}) is None
        assert _slot_free()
    finally:
        os.close(fd)


def test_a_release_racing_the_probe_unlocks_what_the_probe_retook(
    lease_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Step 5: the record vanished between the check and the probe.

    The record read is stubbed to say "held, then gone" (the real flock is
    not): the probe re-locked the shared description, so the verifier must
    unlock it again, never leaving a slot held with no record.
    """
    directory = heavy_lease.lease_dir()
    lease = heavy_lease.try_acquire(directory, 1)
    assert lease is not None
    fd = lease.fileno()
    assert fd is not None
    real = heavy_lease._slot_holder(directory, 0)
    answers = iter([real, None])
    try:
        # A scoped context, never monkeypatch.undo(): undo would also revert
        # lease_env's isolated config dir and aim _slot_free at the real one.
        with monkeypatch.context() as patch:
            patch.setattr(heavy_lease, "_slot_holder", lambda _d, _s: next(answers))
            assert heavy_lease.inherited_lease(directory, {MARKER: f"0:{fd}"}) is None
        assert heavy_lease.lease_dir() == directory
        assert _slot_free()
    finally:
        lease.release()


# -- the bash tool of a runtime that itself runs under a lease -------------------


def test_bash_tool_under_an_inherited_lease_runs_and_never_frees_it(
    lease_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A nested lop runtime's bash tool runs under the ancestor's slot.

    This process plays the ancestor: it holds slot 0 and names its own
    descriptor in the marker, as an inherited one would be. The heavy command
    must run (a queued one would be refused after the 1 s bound), say so, and
    leave the slot HELD: the InheritedLease's release must never unlock the
    ancestor's description.
    """
    from local_operator.tools import builtin

    directory = heavy_lease.lease_dir()
    lease = heavy_lease.try_acquire(directory, 1, command="pnpm build")
    assert lease is not None
    monkeypatch.setenv(MARKER, f"{lease.slot}:{lease.fileno()}")
    updates: list[Any] = []
    try:

        async def scenario() -> Any:
            result = await builtin.execute_bash(
                "t",
                {"command": "printf ran:$" + "{" + MARKER + "}" + _HEAVY_TAIL},
                None,
                updates.append,
                None,
            )
            await asyncio.sleep(0.2)  # let the release watcher run
            return result

        result = asyncio.run(scenario())
        assert not result.is_error, _text(result)
        # The child saw the marker for the fd it was passed.
        assert f"ran:0:{lease.fileno()}" in _text(result), _text(result)
        notes = [u.details.get("heavy_job_queue") for u in updates if u.details]
        assert any(n and "already held by an ancestor" in n for n in notes), notes
        assert heavy_lease.try_acquire(directory, 1) is None, "the ancestor's slot was freed"
    finally:
        lease.release()
    assert _slot_free()


def test_non_leased_bash_commands_do_not_inherit_a_marker(
    lease_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A command that is passed no descriptor is passed no marker either."""
    from local_operator.tools import builtin

    monkeypatch.setenv(MARKER, "0:987")

    async def scenario() -> Any:
        return await builtin.execute_bash(
            "t", {"command": "printf seen:${" + MARKER + ":-none}"}, None, None, None
        )

    assert "seen:none" in _text(asyncio.run(scenario()))


# -- labels -----------------------------------------------------------------------


def test_holder_label_names_the_heavy_segment_not_the_first_word() -> None:
    line = "cd /x && rm -rf tmp/probe && SP=1 && go test -race ./... ; go test ./..."
    heavy = heavy_lease.is_heavy(line)
    assert heavy is not None
    assert heavy_audit.command_label(line) == "rm"  # what the 2026-11-20 record said
    assert heavy.label(line) == "go test"
    # A reason that is not a command falls back to the line's own label.
    by_memory = heavy_lease.is_heavy("make test", 4096)
    assert by_memory is not None and by_memory.reason == "memory_mb=4096"
    assert by_memory.label("make test") == "make test"
