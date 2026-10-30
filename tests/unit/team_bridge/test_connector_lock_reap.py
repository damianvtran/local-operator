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
        self.lock_free = False
        self.lock_probes = 0
        self.now = 0.0
        self.pauses = 0
        monkeypatch.setattr(reap, "_is_posix", lambda: True)
        monkeypatch.setattr(reap, "_lock_holder_pids", lambda path: self.holders)
        monkeypatch.setattr(reap, "_lock_free", self.probe_lock)
        monkeypatch.setattr(reap, "_read_process", self.read)
        monkeypatch.setattr(reap, "_waiter_is_current", lambda: self.waiter_current)
        monkeypatch.setattr(reap, "_holder_is_stale", lambda pid: self.holder_stale)
        monkeypatch.setattr(reap, "_holder_generation", lambda pid: "20261023T142851Z-0.86.1")
        monkeypatch.setattr(reap, "_signal", self.signal)
        monkeypatch.setattr(reap, "_clock", lambda: self.now)
        monkeypatch.setattr(reap, "_pause", self.pause)

    def probe_lock(self, config: Path) -> bool:
        self.lock_probes += 1
        return self.lock_free

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


def test_a_lock_proved_free_is_never_signalled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE OPENERS GAP: a lock a non-blocking acquire can take needs no signal.

    On macOS the holder probe names OPENERS, and the process selected above may
    merely have the file open. The last proof asks the lock itself; when it is
    free the outcome is ``lock_free`` and nothing is signalled — signalling an
    opener because ``lsof`` could not tell it apart from the holder is the wrong
    reap this direction exists to prevent.
    """

    world = _World(monkeypatch)
    world.lock_free = True
    outcome = _reap(world)
    assert outcome.kind == "reaped", outcome
    assert outcome.reason == "lock_free"
    assert world.lock_probes == 1
    assert world.signals == []


def test_a_lock_probe_that_cannot_answer_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Doubt is not freedom: an unreadable lock file refuses before any signal."""

    world = _World(monkeypatch)

    def boom(config: Path) -> bool:
        raise OSError("unreadable")

    monkeypatch.setattr(reap, "_lock_free", boom)
    outcome = _reap(world)
    assert outcome.kind == "refused"
    assert outcome.reason == "lock_probe_failed"
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


def test_a_failed_sigkill_is_not_reported_as_reaped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A kill the kernel refuses stopped nobody, so it is a refusal, not a reap.

    ``_signal`` answers False for a refusal and for a process that vanished
    between the identity check and the call; either way this call delivered
    nothing, and claiming ``reaped/sigkill`` would report a stop that did not
    happen.
    """

    world = _World(monkeypatch)
    monkeypatch.setattr(
        reap,
        "_signal",
        lambda pid, number: world.signals.append((pid, number)) or number != signal.SIGKILL,
    )
    world.signals = []

    outcome = _reap(world)

    assert outcome.kind == "refused", outcome
    assert outcome.reason == "sigkill_refused"
    assert world.signals == [(HOLDER, signal.SIGTERM), (HOLDER, signal.SIGKILL)]


def test_a_graceful_reap_writes_one_reaped_line_with_sigterm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """``stopping`` says SIGTERM was sent; one ``reaped`` line says how it ended."""

    world = _World(monkeypatch)
    with caplog.at_level("WARNING", logger="local_operator.team_bridge.connector"):
        outcome = _reap(world)

    assert outcome.reason == "sigterm"
    messages = [record.getMessage() for record in caplog.records]
    stopping = [m for m in messages if "stopping stale holder" in m]
    reaped = [m for m in messages if "reaped stale holder" in m]
    assert len(stopping) == 1, messages
    assert len(reaped) == 1, messages
    assert reaped[0].endswith("evidence=same_uid,same_config,older_generation,sigterm"), reaped


def test_a_wedged_reap_writes_one_reaped_line_with_sigkill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """ONE grep-able claim per attempt, written after the outcome is known.

    The old code wrote a ``reaped`` line right after SIGTERM and a second after
    SIGKILL, so one attempt produced two claims about one process — and the
    first was written before the holder had necessarily died.
    """

    world = _World(monkeypatch)
    monkeypatch.setattr(
        reap, "_signal", lambda pid, number: world.signals.append((pid, number)) or True
    )
    world.signals = []
    with caplog.at_level("WARNING", logger="local_operator.team_bridge.connector"):
        outcome = _reap(world)

    assert outcome.reason == "sigkill"
    messages = [record.getMessage() for record in caplog.records]
    stopping = [m for m in messages if "stopping stale holder" in m]
    reaped = [m for m in messages if "reaped stale holder" in m]
    assert len(stopping) == 1, messages
    assert len(reaped) == 1, messages
    assert reaped[0].endswith("evidence=same_uid,same_config,older_generation,sigkill"), reaped


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


# -- the parsers, from fixture strings ----------------------------------------
#
# These run on EVERY host, including a macOS one with no /proc and no live
# holder: the /proc/locks table and the ps/lsof answers are fed as strings, so the
# parsers are exercised here rather than first on CI's Linux shard.


class _FakeRun:
    """What ``reap._run`` answers: the two fields the parsers read."""

    def __init__(self, returncode: int, stdout: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout


_PS_LINE = (
    "501 Fri Oct 23 14:28:51 2026 1:02.34 "
    "/img -m local_operator.team_bridge.connector --config-dir /store"
)


def _lock_identity(path: Path) -> str:
    """The ``major:minor:inode`` a ``/proc/locks`` row carries, in hex."""

    sample = path.stat()
    return f"{os.major(sample.st_dev):x}:{os.minor(sample.st_dev):x}:{sample.st_ino}"


def _table_row(pid: int, identity: str, kind: str = "FLOCK") -> str:
    return f"1: {kind}  ADVISORY  WRITE {pid} {identity} 0 EOF"


def test_the_proc_locks_parser_reads_a_fixture_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only FLOCK rows for THIS inode count, once per pid."""

    path = tmp_path / "connector.lock"
    path.write_bytes(b"")
    identity = _lock_identity(path)
    table = "\n".join(
        [
            _table_row(HOLDER, identity),
            _table_row(OTHER, identity),
            _table_row(777, identity, kind="POSIX"),  # a different lock kind
            _table_row(888, "00:1b:999999"),  # another inode
            _table_row(HOLDER, identity),  # the same holder, listed twice
            "garbage",
        ]
    )
    monkeypatch.setattr(reap, "_proc_locks_text", lambda: table)
    assert reap._proc_locks_holder_pids(path) == [HOLDER, OTHER]


def test_the_proc_locks_parser_answers_none_when_the_table_is_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "connector.lock"
    path.write_bytes(b"")
    monkeypatch.setattr(reap, "_proc_locks_text", lambda: None)
    assert reap._proc_locks_holder_pids(path) is None


def test_the_proc_locks_parser_answers_none_when_the_file_is_gone(tmp_path: Path) -> None:
    assert reap._proc_locks_holder_pids(tmp_path / "absent.lock") is None


def test_the_lsof_parser_reads_a_fixture_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reap, "_run", lambda argv: _FakeRun(0, "p123\np456\np123\n"))
    assert reap._lsof_holder_pids(tmp_path / "lock") == [123, 456]


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (_FakeRun(1, ""), []),  # lsof's "no matches": the one non-zero kept
        (_FakeRun(1, "p123\n"), None),  # non-zero AND output: unanswered
        (_FakeRun(0, ""), []),
        (_FakeRun(0, "pabc\n"), None),  # unparseable
        (None, None),  # no lsof / timeout
    ],
)
def test_the_lsof_parser_refuses_what_it_cannot_read(
    result: _FakeRun | None,
    expected: list[int] | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reap, "_run", lambda argv: result)
    assert reap._lsof_holder_pids(tmp_path / "lock") == expected


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ("/img -m local_operator.team_bridge.connector --config-dir /store", "/store"),
        ("/img --config-dir=/store", "/store"),
        ("/img --config-dir", None),
        ("/img --config-dir --base-url http://x", None),  # next token is a flag
        ("/img --config-dir=", None),
        ("/img -m mod", None),
    ],
)
def test_the_option_parser_reads_both_spellings(argv: str, expected: str | None) -> None:
    assert reap._option(argv, "--config-dir") == expected


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (f"/img --config-dir {CONFIG}", True),
        (f"/img --config-dir {OTHER_CONFIG}", False),
        # n2: a relative path resolves against the HOLDER's cwd, not ours.
        ("/img --config-dir store-a", False),
        ("/img -m mod", False),
    ],
)
def test_the_config_probe_refuses_what_it_cannot_compare(argv: str, expected: bool) -> None:
    assert reap._same_config(argv, CONFIG) is expected


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("0:01.23", 1.23),
        ("1:02:03", 3723.0),
        # m3: both ps flavours switch to ``dd-`` past 24 h of cumulative CPU.
        ("1-02:03:04", 93784.0),
        ("0-00:00.50", 0.5),
        ("12", None),
        ("1:2:3:4", None),
        ("junk", None),
        ("-1:00", None),
    ],
)
def test_the_cpu_parser_reads_every_ps_time_shape(token: str, expected: float | None) -> None:
    assert reap._cpu_seconds(token) == expected


def test_the_process_reader_parses_one_ps_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def fake_run(argv: list[str]) -> _FakeRun:
        seen.append(argv)
        return _FakeRun(0, _PS_LINE + "\n")

    monkeypatch.setattr(reap, "_run", fake_run)
    proc = reap._read_process(HOLDER)

    assert proc is not None
    assert (proc.pid, proc.uid, proc.start) == (HOLDER, 501, "Fri Oct 23 14:28:51 2026")
    assert proc.cpu_s == 62.34
    assert proc.argv.endswith("--config-dir /store")
    # ``-ww`` is load-bearing: a piped ps truncates argv on Linux.
    assert "-ww" in seen[0]


@pytest.mark.parametrize(
    "result",
    [
        None,
        _FakeRun(1),
        _FakeRun(0, ""),
        _FakeRun(0, "501 Fri Oct 23 14:28:51 2026 1:02.34"),  # argv missing
        _FakeRun(0, "notauid Fri Oct 23 14:28:51 2026 1:02.34 /img -m mod"),
        _FakeRun(0, "501 Fri Oct 23 14:28:51 2026 nonsense /img -m mod"),
    ],
)
def test_the_process_reader_refuses_an_unreadable_answer(
    result: _FakeRun | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reap, "_run", lambda argv: result)
    assert reap._read_process(HOLDER) is None


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
