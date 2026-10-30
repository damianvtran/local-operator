"""The stale-lock reaper's proofs, one refusal at a time.

THE INCIDENT. Connector pid 48311 (generation 20261023T142851Z-0.86.1, ppid 1,
seven days old, idle, no sockets) held ``<config>/team_bridge/connector.lock``,
so the supervised job com.local-operator.bridge-connector was refused 2,826 times
and nothing inbound was admitted for days. ``flock`` is released only when the
holder dies, and that holder was frozen: no graceful exit was ever coming.

THE DIRECTION IS REFUSAL, and that is what most of this file pins. A wrong reap
stops a working receiver — a live current-generation connector, another user's,
another store's — so every proof must hold and the first one that does not stops
the attempt. A missed reap costs one more supervised start; a wrong one costs
someone's process.

The world here is injected at the module's own seams (``_read_process``,
``_lock_holder_pids``, ``_signal``, ``_clock``/``_pause``), so no test signals a
real process and none waits twenty seconds: the clock is a number the pause
advances.
"""

from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path

import pytest

from local_operator.team_bridge.connector import reap

HOLDER = 48311
OTHER = 999
CONFIG = Path("/nonexistent/store-a")
OTHER_CONFIG = Path("/nonexistent/store-b")

#: A ``ps``-shaped argv for a connector on ``config``. The image path carries the
#: generation, which is what ``update.generation_in_argv`` reads.
_OLD_ARGV = (
    "/home/x/.local/share/lop/generations/20261023T142851Z-0.86.1/tools/local-operator/bin/"
    "Local Operator -m local_operator.team_bridge.connector --config-dir {config}"
)
_CURRENT_ARGV = (
    "/home/x/.local/share/lop/generations/20261030T145827Z-0.92.2/tools/local-operator/bin/"
    "Local Operator -m local_operator.team_bridge.connector --config-dir {config}"
)


def _proc(
    pid: int = HOLDER,
    *,
    argv: str | None = None,
    uid: int | None = None,
    start: str = "Fri Oct 23 14:28:51 2026",
    cpu: float = 1.0,
) -> reap._Process:
    return reap._Process(
        pid=pid,
        uid=os.getuid() if uid is None else uid,
        start=start,
        cpu_s=cpu,
        argv=_OLD_ARGV.format(config=CONFIG) if argv is None else argv,
    )


class _World:
    """The seams as one object: reads queue up, signals are recorded, time is free."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, config: Path = CONFIG) -> None:
        self.config = config
        self.holders: list[int] | None = [HOLDER]
        self.reads: dict[int, list[reap._Process | None]] = {HOLDER: [_proc()]}
        self.signals: list[tuple[int, int]] = []
        self.waiter_current = True
        self.holder_stale = True
        self.now = 0.0
        self.pauses = 0
        monkeypatch.setattr(reap, "_is_posix", lambda: True)
        monkeypatch.setattr(reap, "_lock_holder_pids", lambda path: self.holders)
        monkeypatch.setattr(reap, "_read_process", self.read)
        monkeypatch.setattr(reap, "_waiter_is_current", lambda: self.waiter_current)
        monkeypatch.setattr(reap, "_holder_is_stale", lambda pid: self.holder_stale)
        monkeypatch.setattr(reap, "_holder_generation", lambda pid: "20261023T142851Z-0.86.1")
        monkeypatch.setattr(reap, "_signal", self.signal)
        monkeypatch.setattr(reap, "_clock", lambda: self.now)
        monkeypatch.setattr(reap, "_pause", self.pause)

    def read(self, pid: int) -> reap._Process | None:
        queue = self.reads.get(pid)
        if not queue:
            return None
        return queue.pop(0) if len(queue) > 1 else queue[0]

    def signal(self, pid: int, number: int) -> bool:
        self.signals.append((pid, number))
        # The default action: a holder that does not handle SIGTERM is gone.
        if number == signal.SIGTERM:
            self.reads[pid] = [None]
        return True

    async def pause(self, seconds: float) -> None:
        self.pauses += 1
        self.now += seconds
        await asyncio.sleep(0)


def _reap(world: _World) -> reap.ReapOutcome:
    return asyncio.run(reap.reap_stale_holder(world.config))


def test_a_stale_generation_holder_is_terminated_with_sigterm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE INCIDENT'S SHAPE: every proof holds, one SIGTERM, and the lock is free.

    The holder is the one missing thing the caller needs, so this also pins the
    outcome kind the caller retries its acquire on.
    """

    world = _World(monkeypatch)
    outcome = _reap(world)
    assert outcome.kind == "reaped", outcome
    assert outcome.reason == "sigterm"
    assert world.signals == [(HOLDER, signal.SIGTERM)], world.signals


def test_a_current_generation_holder_is_never_signalled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A LIVE CURRENT CONNECTOR IS NEVER TOUCHED, whatever it is doing.

    ``_holder_is_stale`` is the tri-state answer (``None`` and ``False`` both
    arrive here as False), so a holder whose build could not be established takes
    the same silent route as one that is current — the direction that leaves a
    working receiver working.
    """

    world = _World(monkeypatch)
    world.reads[HOLDER] = [_proc(argv=_CURRENT_ARGV.format(config=CONFIG))]
    world.holder_stale = False
    outcome = _reap(world)
    assert outcome.kind == "refused"
    assert outcome.reason == "holder_generation_unproven"
    assert world.signals == []


def test_a_holder_on_another_store_is_left_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same uid, same module, DIFFERENT ``--config-dir``: not our lock to break.

    The singleton is per store, and a supervisor's label is per user: a process
    serving another store is another installation's connector, however it came to
    hold this file.
    """

    world = _World(monkeypatch)
    world.reads[HOLDER] = [_proc(argv=_OLD_ARGV.format(config=OTHER_CONFIG))]
    outcome = _reap(world)
    assert outcome.kind == "refused"
    assert outcome.reason == "different_config"
    assert world.signals == []


def test_a_holder_from_another_uid_is_left_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Another account's connector is not ours to signal, even on our lock file."""

    world = _World(monkeypatch)
    world.reads[HOLDER] = [_proc(uid=os.getuid() + 1)]
    outcome = _reap(world)
    assert outcome.kind == "refused"
    assert outcome.reason == "different_uid"
    assert world.signals == []


def test_a_holder_that_is_not_a_connector_is_left_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The file could be held by anything; only the connector is ever signalled.

    Also pins the whole-word match: a module named ``..._v2`` is a different
    program and must not borrow this rule.
    """

    world = _World(monkeypatch)
    world.reads[HOLDER] = [
        _proc(
            argv="/usr/bin/python3 -m local_operator.team_bridge.connector_v2 "
            f"--config-dir {CONFIG}"
        )
    ]
    outcome = _reap(world)
    assert outcome.kind == "refused"
    assert outcome.reason == "not_a_connector"
    assert world.signals == []


def test_the_waiter_must_be_provably_current_before_it_judges_anyone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proof 1, and it is asked FIRST: a superseded waiter may not pick a winner.

    ``None`` (our own argv names no generation, or the pointer is unreadable) is
    the same answer as ``True`` here — unproven means no reap.
    """

    world = _World(monkeypatch)
    world.waiter_current = False
    outcome = _reap(world)
    assert outcome.kind == "refused"
    assert outcome.reason == "waiter_generation_unproven"
    assert world.signals == []
    assert world.pauses == 0


@pytest.mark.parametrize("holders", [[HOLDER, OTHER], [], [0]])
def test_only_exactly_one_other_holder_may_be_reaped(
    holders: list[int], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Zero, two, or a pid that is not a process: refuse rather than pick one.

    Two holders cannot both be the stale singleton, so the reading is wrong about
    at least one of them; picking would be a guess about which process to stop.
    """

    world = _World(monkeypatch)
    world.holders = holders
    outcome = _reap(world)
    assert outcome.kind == "refused"
    assert outcome.reason == "holder_count_not_one"
    assert world.signals == []


def test_a_holder_discovery_that_cannot_answer_is_a_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No ``lsof``, a timeout, garbage: an unanswered question is not a holder."""

    world = _World(monkeypatch)
    world.holders = None
    outcome = _reap(world)
    assert outcome.kind == "refused"
    assert outcome.reason == "holders_unknown"
    assert world.signals == []


def test_a_pid_reused_between_the_proofs_and_the_signal_is_not_signalled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proof 5. Pids are reused, and the process we would signal is not the one
    we proved about — so the start time and argv are re-read first."""

    world = _World(monkeypatch)
    world.reads[HOLDER] = [
        _proc(),
        _proc(start="Fri Oct 30 09:00:00 2026"),
    ]
    outcome = _reap(world)
    assert outcome.kind == "refused"
    assert outcome.reason == "holder_identity_moved"
    assert world.signals == []


def test_a_wedged_holder_is_killed_after_the_graceful_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SIGTERM, twenty seconds, flat CPU, then SIGKILL — the frozen process.

    The CPU sample is what separates "wedged" from "working": the incident holder
    answered ``ps`` and nothing else, so its time never moved.
    """

    world = _World(monkeypatch)
    world.reads[HOLDER] = [_proc(cpu=1.0)]
    # The holder ignores SIGTERM: this world's signal() must not clear the read.
    monkeypatch.setattr(
        reap, "_signal", lambda pid, number: world.signals.append((pid, number)) or True
    )
    world.signals = []

    outcome = _reap(world)

    assert outcome.kind == "reaped"
    assert outcome.reason == "sigkill"
    assert world.signals == [(HOLDER, signal.SIGTERM), (HOLDER, signal.SIGKILL)], world.signals
    assert world.now >= reap.STOP_WAIT_S


def test_a_holder_that_ignores_sigterm_but_burns_cpu_is_not_killed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Busy is not wedged: real CPU inside the window buys the benefit of the doubt.

    A connector finishing an in-flight iteration looks exactly like this, and
    SIGKILLing it is how a reap would corrupt the work it was completing.
    """

    world = _World(monkeypatch)
    reads: list[reap._Process | None] = [_proc(cpu=1.0), _proc(cpu=4.0)]
    world.reads[HOLDER] = reads
    monkeypatch.setattr(
        reap, "_read_process", lambda pid: reads[-1] if len(reads) == 1 else reads.pop(0)
    )
    monkeypatch.setattr(
        reap, "_signal", lambda pid, number: world.signals.append((pid, number)) or True
    )
    world.signals = []

    outcome = _reap(world)

    assert outcome.kind == "reaped"
    assert outcome.reason == "sigterm_busy"
    assert world.signals == [(HOLDER, signal.SIGTERM)], world.signals


def test_windows_never_reaps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No flock, no generations: the probe there is the installer's own."""

    world = _World(monkeypatch)
    monkeypatch.setattr(reap, "_is_posix", lambda: False)
    outcome = _reap(world)
    assert outcome.kind == "refused"
    assert outcome.reason == "platform"
    assert world.signals == []


def test_a_refusal_logs_one_content_free_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The reason is a FIELD, not a sentence with a path or an argv in it: the
    connector's log is persisted, and this line is read by a person looking for
    why nothing inbound was admitted."""

    world = _World(monkeypatch)
    world.holder_stale = False
    with caplog.at_level("WARNING", logger="local_operator.team_bridge.connector"):
        _reap(world)
    messages = [record.getMessage() for record in caplog.records]
    expected = f"bridge connector reap refused: proof=holder_generation_unproven pid={HOLDER}"
    assert messages == [expected], messages
    assert str(CONFIG) not in "\n".join(messages)


# -- the caller's contract -----------------------------------------------------


def _commit(config_dir: Path) -> None:
    """A committed owner, so ``_main_async`` reaches the LOCK instead of idling."""

    from local_operator.team_bridge.store import OwnerReceipt, Store
    from local_operator.team_bridge.types import OwnerSnapshot

    owner = OwnerSnapshot(
        credential_id=1,
        account_id="acc_1",
        tenant_id="tnt_1",
        endpoint_id="bep_1",
        canonical_base_url="http://127.0.0.1:1/v1",
        enrollment_generation=1,
    )
    Store(config_dir).commit_enrollment(owner, OwnerReceipt("bridge_consent_v1", "bep_1"))


def test_a_busy_lock_makes_exactly_one_attempt_then_exits_75(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE CALLER'S CONTRACT. One reap attempt per process start, and the same
    exit 75 as before when it refuses — a supervisor retries a failure, not a
    clean exit, so 75 is what keeps the job coming back."""

    import argparse

    from local_operator.team_bridge.connector import __main__ as main_mod
    from local_operator.team_bridge.connector.lock import (
        ConnectorLock,
        ConnectorLockBusy,
    )

    _commit(tmp_path)
    attempts: list[Path] = []

    def busy(self: ConnectorLock) -> None:
        raise ConnectorLockBusy("held elsewhere")

    async def refuse(config_dir: Path) -> reap.ReapOutcome:
        attempts.append(config_dir)
        return reap.ReapOutcome(kind="refused", reason="holder_generation_unproven")

    monkeypatch.setattr(ConnectorLock, "acquire", busy)
    monkeypatch.setattr(reap, "reap_stale_holder", refuse)
    args = argparse.Namespace(
        config_dir=str(tmp_path),
        credential_id=None,
        account_id=None,
        tenant_id=None,
        endpoint_id=None,
        base_url="http://127.0.0.1:1/v1",
    )

    assert asyncio.run(main_mod._main_async(args, None)) == 75
    assert attempts == [tmp_path]


def test_a_successful_reap_retries_the_lock_and_serves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a reap the acquire is retried ONCE, and a second refusal still exits 75
    rather than leaving the process in a state the supervisor cannot read."""

    import argparse

    from local_operator.team_bridge.connector import __main__ as main_mod
    from local_operator.team_bridge.connector.lock import (
        ConnectorLock,
        ConnectorLockBusy,
    )

    _commit(tmp_path)
    acquires: list[int] = []

    def busy(self: ConnectorLock) -> None:
        acquires.append(1)
        raise ConnectorLockBusy("still held")

    async def reaped(config_dir: Path) -> reap.ReapOutcome:
        return reap.ReapOutcome(kind="reaped", reason="sigterm", pid=HOLDER)

    monkeypatch.setattr(ConnectorLock, "acquire", busy)
    monkeypatch.setattr(reap, "reap_stale_holder", reaped)
    args = argparse.Namespace(
        config_dir=str(tmp_path),
        credential_id=None,
        account_id=None,
        tenant_id=None,
        endpoint_id=None,
        base_url="http://127.0.0.1:1/v1",
    )

    assert asyncio.run(main_mod._main_async(args, None)) == 75
    # The first acquire, then exactly one retry after the reap — never a loop.
    assert len(acquires) == 2
