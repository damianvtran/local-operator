"""The wake supervisor's ownership-scoped SIGTERM sweep (#3028 Q-S1)."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.session.runtime.reclaim import RUNTIME_MODULE
from local_operator.wakes import owned_runtimes


@pytest.fixture(autouse=True)
def _published_records(monkeypatch):
    """Give every candidate a PUBLISHED record, as a real store does.

    The sweep now builds its marker from ``reclaim.read_fleet`` (the published run
    record) rather than the 2-s-rounded census age, because coverage is strict to
    1.0 s and an age-derived key can miss it — and a missed key means the runtime
    refuses the supervisor's SIGTERM, which is #3028 exactly. A unit cell that faked
    only the census would exercise a tree no real store has, so the fleet is stubbed
    once, here, for every cell in this module.
    """
    from local_operator.session.runtime import reclaim

    class _Records(dict[Any, Any]):
        def get(self, pid, default=None):  # noqa: ANN001
            row = reclaim.process_row(pid)
            age = float(getattr(row, "age_s", 0.0) or 30.0)
            return SimpleNamespace(
                pid=pid,
                session_id="",
                started_at=time.time() - age,
            )

    monkeypatch.setattr(
        reclaim,
        "read_fleet",
        lambda root=None, **kw: SimpleNamespace(records=_Records(), boots={}),
    )


def _row(pid: int, root: Path, owner: str, session_id: str, age: float = 30.0):
    env = (
        f"LOCAL_OPERATOR_CONFIG_DIR={root} LOP_MOBILE_CHILD_RESUME={session_id} "
        f"{owned_runtimes.OWNER_ENV}={owner}"
    )
    return SimpleNamespace(
        pid=pid,
        age_s=age,
        command=f"{sys.executable} -m {RUNTIME_MODULE} {env}",
    )


def test_two_roots_only_the_sweeping_roots_runtime_is_signalled(
    tmp_path: Path, monkeypatch
) -> None:
    """A rig supervisor must not terminate the live install's runtimes.

    The ownership token and root are both required; equal session ids do not
    make the other store's process ours. The signal-time process row is also
    rechecked, not just the census row.
    """
    ours, theirs = tmp_path / "ours", tmp_path / "theirs"
    rows = [_row(111, ours, "owner-a", "shared-id"), _row(222, theirs, "owner-b", "shared-id")]
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr("local_operator.session.runtime.reclaim.runtime_processes", lambda: rows)
    monkeypatch.setattr(
        "local_operator.session.runtime.reclaim.process_row",
        lambda pid: next(row for row in rows if row.pid == pid),
    )
    monkeypatch.setattr(owned_runtimes.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    assert owned_runtimes.sweep(ours, owner="owner-a") == [111]
    assert sent == [(111, signal.SIGTERM)]


def test_a_changed_pid_token_or_younger_process_is_never_signalled(
    tmp_path: Path, monkeypatch
) -> None:
    """The exact-pid signal-time guard refuses a recycled or substituted target."""
    root = tmp_path / "root"
    old = _row(111, root, "owner-a", "s1", age=30)
    monkeypatch.setattr("local_operator.session.runtime.reclaim.runtime_processes", lambda: [old])
    sent: list[int] = []
    monkeypatch.setattr(owned_runtimes.os, "kill", lambda pid, sig: sent.append(pid))

    monkeypatch.setattr(
        "local_operator.session.runtime.reclaim.process_row",
        lambda pid: _row(pid, root, "owner-a", "s1", age=1),
    )
    assert owned_runtimes.sweep(root, owner="owner-a") == []
    monkeypatch.setattr(
        "local_operator.session.runtime.reclaim.process_row",
        lambda pid: _row(pid, root, "different-owner", "s1"),
    )
    assert owned_runtimes.sweep(root, owner="owner-a") == []
    assert sent == []


def test_a_sweep_without_a_token_or_permission_targets_nothing(tmp_path: Path, monkeypatch) -> None:
    """No token and no ``any_owner`` is NO target — never an unnamed sweep."""
    rows = [_row(111, tmp_path, "owner-a", "s1")]
    monkeypatch.setattr("local_operator.session.runtime.reclaim.runtime_processes", lambda: rows)
    monkeypatch.setattr(
        "local_operator.session.runtime.reclaim.process_row",
        lambda pid: rows[0] if pid == 111 else None,
    )
    sent: list[int] = []
    monkeypatch.setattr(owned_runtimes.os, "kill", lambda pid, sig: sent.append(pid))
    assert owned_runtimes.sweep(tmp_path) == []
    assert sent == []
    # The uninstall fallback sweeps EVERY wake-owned runtime of THIS store.
    assert owned_runtimes.sweep(tmp_path, any_owner=True) == [111]


def test_a_real_owned_process_is_signalled_by_exact_pid(tmp_path: Path, monkeypatch) -> None:
    """A real process proves the census/argv/env parsing, not just the mocks."""
    root = tmp_path / "root"
    # The host-wide census is a separately tested reclaim seam; use the one-pid
    # production probe here so this cell measures ownership and SIGTERM, not
    # the loaded host's ps latency.
    from local_operator.session.runtime import reclaim

    monkeypatch.setattr(
        reclaim,
        "runtime_processes",
        lambda: [row for pid in [child.pid] if (row := reclaim.process_row(pid)) is not None],
    )
    env = os.environ.copy()
    env.update(
        LOCAL_OPERATOR_CONFIG_DIR=str(root),
        LOP_MOBILE_CHILD_RESUME="r1gs1gnal01",
        **{owned_runtimes.OWNER_ENV: "owner-real"},
    )
    # A tiny real ``-m`` module with the runtime's argv shape; the production
    # runtime is exercised by the isolated SIGTERM rig, not by this unit cell.
    child = subprocess.Popen(
        # ``-m <runtime>`` is only the argv identity: no model runtime boots.
        # The child SLEEPS rather than SIGSTOPping itself. A stopped process
        # holds SIGTERM pending until SIGCONT, so whenever the stop landed
        # before the sweep (Linux CI) the wait below timed out and the cell
        # hung (#3028 shard 0); whether it landed first was a startup race.
        [
            sys.executable,
            "-c",
            "import time; time.sleep(120)",
            "-m",
            RUNTIME_MODULE,
        ],
        env={**env, "PYTHONPATH": str(tmp_path)},
        cwd=tmp_path,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 15
        signalled: list[int] = []
        while time.monotonic() < deadline and not signalled:
            signalled = owned_runtimes.sweep(root, owner="owner-real")
            if not signalled:
                time.sleep(0.1)
        assert signalled == [child.pid]
        try:
            returncode = child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pytest.fail(f"pid {child.pid} was signalled but did not exit within 15 s")
        assert returncode == -signal.SIGTERM
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


# ---------------------------------------------------------------------------
# #3028 review r3 MAJOR-1: a stop reaches EVERY invocation's runtimes in this
# root, and nothing without the marker. Real processes, real signals, driven
# through serve()'s own stop handler (not a direct sweep() call).
# ---------------------------------------------------------------------------


def _spawn_stand_in(root: Path, session_id: str, *, owner: str | None) -> subprocess.Popen[bytes]:
    """A real process carrying the runtime's argv identity and, optionally, a marker.

    It SLEEPS (never SIGSTOPs — a stopped process holds SIGTERM pending, the
    shard-0 hang). ``owner=None`` stands in for a `lop serve`-owned or
    interactive runtime: same root, same argv shape, no supervisor marker.
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in (owned_runtimes.OWNER_ENV, "LOP_WAKE_SUPERVISOR_ACTIVE")
    }
    env.update(LOCAL_OPERATOR_CONFIG_DIR=str(root), LOP_MOBILE_CHILD_RESUME=session_id)
    if owner is not None:
        env[owned_runtimes.OWNER_ENV] = owner
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)", "-m", RUNTIME_MODULE],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _await_probe(children: list[subprocess.Popen[bytes]]) -> None:
    """Wait until every child's environment is readable through the real probe."""
    from local_operator.session.runtime import reclaim

    deadline = time.monotonic() + 15
    pending = {child.pid for child in children}
    while pending and time.monotonic() < deadline:
        for pid in list(pending):
            row = reclaim.process_row(pid)
            if row is not None and "LOP_MOBILE_CHILD_RESUME=" in row.command:
                pending.discard(pid)
        if pending:
            time.sleep(0.05)
    assert not pending, f"probe never saw the environment of {sorted(pending)}"


def _stop_a_real_serve(root: Path, monkeypatch, children: list[subprocess.Popen[bytes]]) -> None:
    """Run the real serve() on ``root`` and fire the stop handler it installs."""
    import asyncio

    from local_operator.session.runtime import reclaim
    from local_operator.wakes import supervisor
    from local_operator.wakes.store import write_entry

    monkeypatch.setattr(
        reclaim,
        "runtime_processes",
        lambda: [row for c in children if (row := reclaim.process_row(c.pid)) is not None],
    )
    handlers: list[Any] = []
    monkeypatch.setattr(
        supervisor,
        "_install_stop_handlers",
        lambda loop, on_stop: handlers.append(on_stop) or [],
    )
    # A wake an hour out keeps the loop asleep: only the stop ends it.
    write_entry(
        root,
        "stopanchor01",
        cwd=str(root),
        schedules=[
            {
                "id": "w1",
                "message": "later",
                "next_due_at": int(time.time() * 1000) + 3_600_000,
                "created_at": 1,
            }
        ],
    )

    async def run() -> None:
        task = asyncio.create_task(supervisor.serve(root))
        for _ in range(500):
            if handlers:
                break
            await asyncio.sleep(0.01)
        assert handlers, "serve() never installed its stop handler"
        handlers[0]()
        try:
            await asyncio.wait_for(task, timeout=30)
        except asyncio.CancelledError:
            pass

    asyncio.run(run())


def _exited(child: subprocess.Popen[bytes], timeout: float = 15.0) -> int | None:
    try:
        return child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


def test_a_stop_sweeps_an_earlier_invocations_runtime_and_spares_the_unmarked(
    tmp_path: Path, monkeypatch
) -> None:
    """(a) an EARLIER invocation's child, same root: SIGTERMed by this stop.
    (b) a same-root runtime with NO marker (serve-owned / interactive): untouched.
    (c) a marked runtime in ANOTHER root: untouched.

    Fails on 18a4374e03a for (a): the stop matched only the current
    invocation's fresh token, so the long-lived runtime a stop exists for —
    spawned by an earlier invocation and outliving it — survived the unit stop.
    """
    monkeypatch.delenv(owned_runtimes.OWNER_ENV, raising=False)
    monkeypatch.delenv("LOP_WAKE_SUPERVISOR_ACTIVE", raising=False)
    root, other = tmp_path / "root", tmp_path / "other"
    root.mkdir()
    other.mkdir()
    earlier = _spawn_stand_in(root, "s3earlier01", owner="token-from-an-earlier-invocation")
    unmarked = _spawn_stand_in(root, "s3unmarked1", owner=None)
    foreign = _spawn_stand_in(other, "s3foreign01", owner="token-from-an-earlier-invocation")
    children = [earlier, unmarked, foreign]
    try:
        _await_probe(children)
        _stop_a_real_serve(root, monkeypatch, children)

        assert _exited(earlier) == -signal.SIGTERM, "(a) earlier invocation's runtime"
        assert unmarked.poll() is None, "(b) an unmarked same-root runtime was signalled"
        assert foreign.poll() is None, "(c) another root's marked runtime was signalled"
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)


# ---------------------------------------------------------------------------
# #3028 review r4 MAJOR-2: the marker must not travel past the runtimes the
# supervisor forks itself. TWO HOPS through the real env builder
# (``launch._spawn_runtime``), with only the final spawn call captured, then a
# LATER invocation's real stop.
# ---------------------------------------------------------------------------

#: Every marker variable, spelled out so this cell runs unchanged on the
#: pre-r4 tree (which has no ``MARKER_ENVS``) and fails there for the real reason.
_MARKERS = (
    owned_runtimes.OWNER_ENV,
    "LOP_WAKE_SUPERVISOR_PID",
    "LOP_WAKE_SUPERVISOR_ACTIVE",
)

#: Runs inside hop 1 (a real separate process, so ``os.getpid()`` differs from
#: the supervisor's): boot the way a runtime does (``forget_inherited_marker``),
#: then build hop 2's environment through the real ``_spawn_runtime`` and print
#: it as JSON. ``spawn_disclaimed`` is the one call captured; nothing is launched.
_HOP_TWO = r"""
import json, sys
from local_operator.session.runtime import launch
try:
    from local_operator.wakes.owned_runtimes import forget_inherited_marker
except ImportError:  # the pre-r4 tree: a runtime kept its inherited marker
    pass
else:
    forget_inherited_marker()
captured = {}


class _Stop(Exception):
    pass


def _capture(*args, **kwargs):
    captured.update(kwargs.get("env") or {})
    raise _Stop


launch.spawn_disclaimed = _capture
try:
    launch._spawn_runtime("s4grandkid01", sys.argv[1], defer_materialise=False)
except _Stop:
    pass
print(json.dumps(captured))
"""


def _env_built_by_spawn_runtime(monkeypatch, session_id: str, cwd: Path) -> dict[str, str]:
    """The child environment ``_spawn_runtime`` builds in THIS process."""
    from local_operator.session.runtime import launch

    captured: dict[str, str] = {}

    class _Stop(Exception):
        pass

    def _capture(*args: Any, **kwargs: Any) -> Any:
        captured.update(kwargs.get("env") or {})
        raise _Stop

    monkeypatch.setattr(launch, "spawn_disclaimed", _capture)
    with pytest.raises(_Stop):
        launch._spawn_runtime(session_id, str(cwd), defer_materialise=False)
    return captured


def _run_with_env(env: dict[str, str], session_id: str) -> subprocess.Popen[bytes]:
    """A real process with exactly ``env`` and the runtime's argv identity."""
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)", "-m", RUNTIME_MODULE],
        env={**env, "LOP_MOBILE_CHILD_RESUME": session_id},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def test_a_wake_runtimes_grandchild_is_never_swept_by_a_later_stop(
    tmp_path: Path, monkeypatch
) -> None:
    """A session a wake runtime starts is not the supervisor's, so a stop spares it.

    Hop 1: the supervisor (this process, with its token and its own pid) builds
    a wake runtime's environment through the real ``_spawn_runtime``: stamped.
    Hop 2: that runtime — a real separate process — builds a grandchild's
    environment through the same builder: on 4a221c04 the token was copied
    whenever the caller's env had it, so the grandchild was marked and a LATER
    invocation's ``any_owner`` stop SIGTERMed it. Now the grandchild carries no
    marker and survives, while hop 1's runtime is still swept.
    """
    import json

    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    for name in _MARKERS:
        monkeypatch.delenv(name, raising=False)
    # This process plays the EARLIER supervisor invocation.
    monkeypatch.setenv(owned_runtimes.OWNER_ENV, "tok-earlier-invocation")
    monkeypatch.setenv("LOP_WAKE_SUPERVISOR_PID", str(os.getpid()))
    # The pre-r4 gate, so this cell is red on 4a221c04 for the right reason.
    monkeypatch.setenv("LOP_WAKE_SUPERVISOR_ACTIVE", "1")

    hop_one_env = _env_built_by_spawn_runtime(monkeypatch, "s4wakert001", root)
    assert hop_one_env.get(owned_runtimes.OWNER_ENV) == "tok-earlier-invocation"

    built = subprocess.run(
        [sys.executable, "-c", _HOP_TWO, str(root)],
        env={**hop_one_env, "LOP_MOBILE_CHILD_RESUME": "s4wakert001"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert built.returncode == 0, built.stderr[-2000:]
    hop_two_env = json.loads(built.stdout.strip().splitlines()[-1])

    for name in _MARKERS:
        monkeypatch.delenv(name, raising=False)
    wake_runtime = _run_with_env(hop_one_env, "s4wakert001")
    grandchild = _run_with_env(hop_two_env, "s4grandkid01")
    children = [wake_runtime, grandchild]
    try:
        _await_probe(children)
        # A LATER invocation's real stop (a fresh token, through serve()).
        _stop_a_real_serve(root, monkeypatch, children)

        assert _exited(wake_runtime) == -signal.SIGTERM, "hop 1 is the supervisor's"
        # The outcome first, so the pre-r4 tree fails on the sweep itself.
        assert _exited(grandchild, timeout=3.0) is None, (
            "the grandchild was swept as the supervisor's: "
            f"marker={hop_two_env.get(owned_runtimes.OWNER_ENV)!r}"
        )
        assert owned_runtimes.OWNER_ENV not in hop_two_env
        for name in _MARKERS[1:]:
            assert name not in hop_one_env, f"{name} must not be passed on"
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)


def test_only_the_supervisor_process_itself_stamps(monkeypatch) -> None:
    """The pid gate, at the unit level: a matching token in a NON-supervisor strips."""
    for name in _MARKERS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(owned_runtimes.OWNER_ENV, "tok")
    monkeypatch.setenv(owned_runtimes.SUPERVISOR_PID_ENV, str(os.getpid() + 1))
    child = {owned_runtimes.OWNER_ENV: "tok", owned_runtimes.SUPERVISOR_PID_ENV: "1"}
    owned_runtimes.stamp_child_env(child)
    assert child == {}

    monkeypatch.setenv(owned_runtimes.SUPERVISOR_PID_ENV, str(os.getpid()))
    owned_runtimes.stamp_child_env(child)
    assert child == {owned_runtimes.OWNER_ENV: "tok"}


# ---------------------------------------------------------------------------
# The victim-side sanction gate (docs/design-surface-independence.md, slice 1)
#
# The sweep was the ONE in-tree sender of a terminating signal to a runtime that
# staged nothing. A runtime now REFUSES such a signal, so a bare sweep would
# silently stop working — the exact opposite of what this unit's stop is for.
# These cells pin the marker, its ORDER (before the signal), and its withdrawal
# when the signal never lands.
# ---------------------------------------------------------------------------


def _seed_conversation(root: Path, session_id: str) -> Path:
    """The conversation directory a staged marker is written into."""
    from local_operator.session.runtime.types import session_dir

    directory = session_dir(root, session_id)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def test_a_sweep_stages_a_marker_for_each_candidate_before_it_signals(
    tmp_path: Path, monkeypatch
) -> None:
    """ORDERING IS THE CONTRACT: the marker is on disk when the signal is sent.

    The kill double reads the marker INSIDE the signal call, so a sweep that
    staged afterwards (or not at all) fails here rather than passing on a file
    written a moment later.
    """
    from local_operator.session.runtime import registry
    from local_operator.session.runtime.types import session_dir

    root = tmp_path / "root"
    rows = [_row(111, root, "owner-a", "s1"), _row(222, root, "owner-a", "s2")]
    for session_id in ("s1", "s2"):
        _seed_conversation(root, session_id)
    monkeypatch.setattr("local_operator.session.runtime.reclaim.runtime_processes", lambda: rows)
    monkeypatch.setattr(
        "local_operator.session.runtime.reclaim.process_row",
        lambda pid: next(row for row in rows if row.pid == pid),
    )

    seen: list[tuple[int, dict[str, Any] | None]] = []

    def _kill(pid: int, sig: int) -> None:
        # Read at signal time, from the victim's own conversation directory.
        session_id = "s1" if pid == 111 else "s2"
        seen.append((pid, registry.read_stop_marker(session_dir(root, session_id))))

    monkeypatch.setattr(owned_runtimes.os, "kill", _kill)

    assert owned_runtimes.sweep(root, any_owner=True) == [111, 222]
    assert [pid for pid, _ in seen] == [111, 222]
    for pid, marker in seen:
        assert marker is not None, f"pid {pid} was signalled with nothing staged"
        assert marker["mechanism"] == owned_runtimes.SWEEP_MECHANISM
        assert marker["deliberate"] is False, "a sweep must not read as a user's stop"
        assert marker["actor"] == owned_runtimes.SWEEP_ATTEST_ACTOR
        assert marker["pid"] == pid


def test_a_signal_that_never_landed_takes_its_marker_back(tmp_path: Path, monkeypatch) -> None:
    """A refused-by-the-kernel signal must not leave a marker narrating a death.

    ``os.kill`` raising means the act was NOT taken, so the marker staged for it is
    the only artifact left saying otherwise, and a later unrelated death of that
    run would be narrated as this sweep's.
    """
    from local_operator.session.runtime import registry
    from local_operator.session.runtime.types import session_dir

    root = tmp_path / "root"
    rows = [_row(111, root, "owner-a", "s1")]
    _seed_conversation(root, "s1")
    monkeypatch.setattr("local_operator.session.runtime.reclaim.runtime_processes", lambda: rows)
    monkeypatch.setattr("local_operator.session.runtime.reclaim.process_row", lambda pid: rows[0])

    def _kill(pid: int, sig: int) -> None:
        raise ProcessLookupError(pid)

    monkeypatch.setattr(owned_runtimes.os, "kill", _kill)

    assert owned_runtimes.sweep(root, owner="owner-a") == []
    assert registry.read_stop_marker(session_dir(root, "s1")) is None


def test_an_unwritable_conversation_does_not_stop_the_sweep(tmp_path: Path, monkeypatch) -> None:
    """The unit's stop outranks the paperwork: no directory, still signalled.

    A runtime whose conversation directory is gone is exactly the one ``lop stop``
    cannot reach by record; leaving it alive because a sidecar could not be written
    would be the failure this sweep exists to prevent.
    """
    root = tmp_path / "root"
    rows = [_row(111, root, "owner-a", "no-such-session")]
    monkeypatch.setattr("local_operator.session.runtime.reclaim.runtime_processes", lambda: rows)
    monkeypatch.setattr("local_operator.session.runtime.reclaim.process_row", lambda pid: rows[0])
    sent: list[int] = []
    monkeypatch.setattr(owned_runtimes.os, "kill", lambda pid, sig: sent.append(pid))

    assert owned_runtimes.sweep(root, owner="owner-a") == [111]
    assert sent == [111]


def test_a_candidate_with_no_published_record_is_still_stopped_with_an_unstamped_marker(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    """Q3/R-3: a supervisor stop must still stop a runtime that has no published record.

    The first cut skipped it ("no run key, no sanction"), which left the runtime
    running where main signalled it. The candidate is one THIS unit started and
    promised to stop, so the stop goes out with a marker staged without a start
    stamp (session + pid + staging window still pair it, ``marker_covers_signal``),
    and the gap is said in the log. The ``fleet.boots`` fallback the first cut also
    had is gone: boot values are plain dicts (so ``getattr`` found nothing) and
    their ``started_at`` differs from the runtime's own by 0.15-16 s.
    """
    import logging

    from local_operator.session.runtime import reclaim, registry, signal_receipt
    from local_operator.session.runtime.types import session_dir

    root = tmp_path / "root"
    rows = [_row(111, root, "owner-a", "s1")]
    _seed_conversation(root, "s1")
    monkeypatch.setattr(reclaim, "runtime_processes", lambda: rows)
    monkeypatch.setattr(reclaim, "process_row", lambda pid: rows[0])
    monkeypatch.setattr(
        reclaim,
        "read_fleet",
        lambda root=None, **kw: SimpleNamespace(
            records={}, boots={111: {"pid": 111, "started_at": 12345.0}}
        ),
    )
    seen: list[dict[str, Any] | None] = []

    def _kill(pid: int, sig: int) -> None:
        seen.append(registry.read_stop_marker(session_dir(root, "s1")))

    monkeypatch.setattr(owned_runtimes.os, "kill", _kill)
    caplog.set_level(logging.WARNING, logger=owned_runtimes.logger.name)

    assert owned_runtimes.sweep(root, owner="owner-a") == [111]
    assert len(seen) == 1 and seen[0] is not None
    marker = seen[0]
    assert marker["started_at"] is None, "a boot record's start was used as the run key"
    assert any("no published start" in r.getMessage() for r in caplog.records)
    assert signal_receipt.marker_covers_signal(
        marker, session_id="s1", pid=111, started_at=1.0, at=float(marker["at"])
    )


def test_a_candidate_with_only_a_run_record_is_keyed_from_it(tmp_path: Path, monkeypatch) -> None:
    """No ``SessionRecord`` but a ``runtime-run.json`` for this pid: that start is the key."""
    from local_operator.session.runtime import journal, reclaim, registry
    from local_operator.session.runtime.types import session_dir

    root = tmp_path / "root"
    rows = [_row(111, root, "owner-a", "s1")]
    directory = _seed_conversation(root, "s1")
    assert journal.write_run_record(
        "s1", conversation_dir=directory, pid=111, started_at=1_799_990_123.5
    )
    monkeypatch.setattr(reclaim, "runtime_processes", lambda: rows)
    monkeypatch.setattr(reclaim, "process_row", lambda pid: rows[0])
    monkeypatch.setattr(
        reclaim, "read_fleet", lambda root=None, **kw: SimpleNamespace(records={}, boots={})
    )
    seen: list[dict[str, Any] | None] = []
    monkeypatch.setattr(
        owned_runtimes.os,
        "kill",
        lambda pid, sig: seen.append(registry.read_stop_marker(session_dir(root, "s1"))),
    )

    assert owned_runtimes.sweep(root, owner="owner-a") == [111]
    assert seen[0] is not None and seen[0]["started_at"] == 1_799_990_123.5
