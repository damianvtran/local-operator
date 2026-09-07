"""Arbitration proofs for ``engage_runtime``: at most one runtime, ever.

Two runtimes on one transcript is a FORKED TRAJECTORY — both append, neither
sees the other's rows, and the conversation silently splits. That is the
failure this module's tests exist to make impossible, and the ones that matter
are concurrency proofs rather than API checks: the race is real, it is between
processes, and a check-then-spawn passes every single-threaded test while
losing it.

The spawn itself is stubbed here (a real one costs ~1.2 s of session
construction and a provider) and replaced with a recorder that reproduces the
part of the contract arbitration depends on: a candidate takes the transcript
lease, and only the winner publishes a record. What is NOT stubbed is the
decision — the lease, the record scan, and the loop that reads them are the
production ones.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

import local_operator.session.runtime.launch as launch_module
from local_operator.session.runtime.launch import (
    _CONSTRUCTING_POLL_S,
    _CONSTRUCTING_WINDOW_S,
    _MAX_SPAWNS,
    _POLL_CAP_S,
    _POLL_FACTOR,
    _POLL_INITIAL_S,
    PeerMessageErrand,
    PromptErrand,
    RuntimeStartupError,
    WarmErrand,
    _poll_delay,
    engage_runtime,
)
from local_operator.session.runtime.types import SessionRecord
from local_operator.session_lease import SessionLeaseHeldError, acquire_session_lease

SESSION_ID = "sessionaaa01"


class FakeRuntimeFleet:
    """Stands in for the spawned runtime processes of one machine.

    Each "spawn" contends for the real transcript lease exactly as
    ``process.py`` does through ``spawn_owned_session``. The winner publishes a
    real ``SessionRecord``; losers raise ``SessionLeaseHeldError`` and record
    themselves as exits, which is what ``process.py`` turns into ``return 0``.
    """

    def __init__(self, config_dir: Path, *, construction_delay_s: float = 0.0) -> None:
        self.config_dir = config_dir
        self.construction_delay_s = construction_delay_s
        # ``engage_runtime`` spawns from a worker thread (the spawn blocks, and
        # it must not sit on the caller's loop — #401), so a fake standing in
        # for a process is called OFF the loop too and has to hop back.
        self.loop: asyncio.AbstractEventLoop | None = None
        self.spawns = 0
        self.winners = 0
        self.losers = 0
        self.deferred: list[bool] = []
        #: Set when a candidate wins, so the welcome frame can name the
        #: session the attach client is arbitrating against.
        self.session_id = ""
        self._lease: Any = None
        self._server: asyncio.Server | None = None
        self.delivered: list[dict[str, Any]] = []

    # -- the spawn contract -------------------------------------------------

    def spawn(self, session_id: str, cwd: str, *, defer_materialise: bool) -> None:
        self.spawns += 1
        self.deferred.append(defer_materialise)
        directory = self.config_dir / "sessions" / session_id
        directory.mkdir(parents=True, exist_ok=True)
        try:
            lease = acquire_session_lease(directory)
        except SessionLeaseHeldError:
            # The designed outcome for every contender but one. process.py
            # logs this and exits 0 — losing a race is not an error.
            self.losers += 1
            return
        self._lease = lease
        self.session_id = session_id
        self.winners += 1
        # Construction takes real time in production, and the window between
        # taking the lease and publishing a record is exactly the STARTING
        # state the arbitration loop must wait through rather than spawn into.
        loop = self.loop
        assert loop is not None, "fleet.loop must be bound before a spawn"
        asyncio.run_coroutine_threadsafe(self._publish_after_construction(session_id), loop)

    async def _publish_after_construction(self, session_id: str) -> None:
        if self.construction_delay_s:
            await asyncio.sleep(self.construction_delay_s)
        await self._serve(session_id)

    async def _serve(self, session_id: str) -> None:
        """Publish a record backed by a socket that acks like a runtime."""
        from local_operator.session.runtime import registry

        server = await asyncio.start_server(self._handle, host="127.0.0.1", port=0)
        self._server = server
        port = server.sockets[0].getsockname()[1]
        record = SessionRecord(
            pid=os.getpid(),
            kind="daemon",
            session_id=session_id,
            conversation_name="fake",
            cwd=str(self.config_dir),
            model_label="test/model",
            control_port=port,
            control_key="k" * 16,
        )
        registry.publish(record, self.config_dir)
        # The lease marker the record scan consults for liveness.
        marker = self.config_dir / "sessions" / session_id / ".session.pid"
        marker.write_text(str(os.getpid()), encoding="utf-8")

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        auth = await reader.readline()
        if not auth:
            return
        # Both dial classes expect an unsolicited welcome first, and an attach
        # client VALIDATES it (identity arbitration), so it must be a real
        # projection naming this session rather than an empty object.
        writer.write(
            json.dumps(
                {
                    "op": "welcome",
                    "data": {
                        "session_id": self.session_id,
                        "pid": os.getpid(),
                        "cwd": str(self.config_dir),
                    },
                }
            ).encode()
            + b"\n"
        )
        await writer.drain()
        while True:
            line = await reader.readline()
            if not line:
                return
            frame = json.loads(line.decode())
            self.delivered.append(frame)
            writer.write(
                json.dumps(
                    {
                        "op": "ack",
                        "req": frame.get("req"),
                        "detail": "delivered",
                        "duplicate": False,
                    }
                ).encode()
                + b"\n"
            )
            await writer.drain()

    def close(self) -> None:
        if self._server is not None:
            self._server.close()
        if self._lease is not None:
            self._lease.release()


@pytest.fixture
def fleet(tmp_path: Path, monkeypatch):  # noqa: ANN201
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions").mkdir(parents=True, exist_ok=True)
    instance = FakeRuntimeFleet(tmp_path)

    def _spawn(session_id: str, cwd: str, *, defer_materialise: bool) -> None:
        instance.spawn(session_id, cwd, defer_materialise=defer_materialise)

    monkeypatch.setattr("local_operator.session.runtime.launch._spawn_runtime", _spawn)
    try:
        yield instance
    finally:
        instance.close()


@pytest.mark.asyncio
async def test_ten_concurrent_engages_start_exactly_one_runtime(
    fleet: FakeRuntimeFleet, tmp_path: Path
) -> None:
    """The headline proof: N contenders, one runtime, N deliveries.

    Ten callers engage the same cold session simultaneously. Several may spawn
    a candidate — that is allowed and is what makes the design robust — but the
    lease admits exactly ONE, every loser exits, and all ten messages land on
    that one runtime.
    """
    fleet.loop = asyncio.get_running_loop()
    outcomes = await asyncio.gather(
        *(
            engage_runtime(
                SESSION_ID,
                str(tmp_path),
                PeerMessageErrand(text=f"note {index}", sender={"pid": index}),
                config_dir=tmp_path,
            )
            for index in range(10)
        )
    )

    assert fleet.winners == 1, "the lease admitted more than one runtime"
    assert fleet.losers == fleet.spawns - 1
    assert len({outcome.session_id for outcome in outcomes}) == 1
    peer_frames = [f for f in fleet.delivered if f.get("op") == "peer_message"]
    assert len(peer_frames) == 10, "every engagement must reach the one runtime"
    assert {f["text"] for f in peer_frames} == {f"note {i}" for i in range(10)}


@pytest.mark.asyncio
async def test_engaging_a_running_session_spawns_nothing(
    fleet: FakeRuntimeFleet, tmp_path: Path
) -> None:
    """The common case must cost no process at all."""
    fleet.loop = asyncio.get_running_loop()
    await engage_runtime(SESSION_ID, str(tmp_path), WarmErrand(), config_dir=tmp_path)
    assert fleet.spawns == 1
    spawns_after_start = fleet.spawns

    for _ in range(5):
        outcome = await engage_runtime(
            SESSION_ID,
            str(tmp_path),
            PeerMessageErrand(text="hello", sender={}),
            config_dir=tmp_path,
        )
        assert outcome.spawned is False
    assert fleet.spawns == spawns_after_start, "a live runtime was engaged with a new spawn"


@pytest.mark.asyncio
async def test_engaging_during_construction_waits_instead_of_spawning(
    tmp_path: Path, monkeypatch
) -> None:
    """STARTING is a state, and engaging into it must not spawn a doomed twin.

    A contender that has taken the lease but not yet published its record is
    mid-construction — ~1.2 s in production. Callers arriving in that window
    see the lease, wait, and attach to the record when it appears. Without that
    branch each of them spawns a candidate that can only lose.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions").mkdir(parents=True, exist_ok=True)
    instance = FakeRuntimeFleet(tmp_path, construction_delay_s=2.0)
    instance.loop = asyncio.get_running_loop()
    monkeypatch.setattr(
        "local_operator.session.runtime.launch._spawn_runtime",
        lambda session_id, cwd, *, defer_materialise: instance.spawn(
            session_id, cwd, defer_materialise=defer_materialise
        ),
    )
    try:
        first = asyncio.ensure_future(
            engage_runtime(SESSION_ID, str(tmp_path), WarmErrand(), config_dir=tmp_path)
        )
        # Let the first caller take the lease and enter construction.
        while instance.winners == 0:
            await asyncio.sleep(0.01)

        # Everyone arriving now sees a lease with no record: the STARTING case.
        others = await asyncio.gather(
            *(
                engage_runtime(
                    SESSION_ID,
                    str(tmp_path),
                    PeerMessageErrand(text=f"late {i}", sender={}),
                    config_dir=tmp_path,
                )
                for i in range(5)
            )
        )
        await first

        assert instance.winners == 1
        assert instance.losers == 0, "a doomed candidate was spawned into the STARTING window"
        assert len(others) == 5
    finally:
        instance.close()


@pytest.mark.asyncio
async def test_warm_errand_defers_materialisation_and_others_do_not(
    fleet: FakeRuntimeFleet, tmp_path: Path
) -> None:
    """A speculative engage must not commit a directory for an abandoned draft."""
    fleet.loop = asyncio.get_running_loop()
    await engage_runtime(SESSION_ID, str(tmp_path), WarmErrand(), config_dir=tmp_path)
    assert fleet.deferred == [True]

    fleet.deferred.clear()
    other = FakeRuntimeFleet(tmp_path)
    other.loop = asyncio.get_running_loop()
    # A prompt-carrying engage against a DIFFERENT cold session, so it spawns.
    import local_operator.session.runtime.launch as launch_module

    original = launch_module._spawn_runtime
    launch_module._spawn_runtime = lambda session_id, cwd, *, defer_materialise: other.spawn(
        session_id, cwd, defer_materialise=defer_materialise
    )
    try:
        await engage_runtime(
            "sessionbbb02",
            str(tmp_path),
            PromptErrand(text="do the thing"),
            config_dir=tmp_path,
        )
        assert other.deferred == [False], "real work must materialise the session"
    finally:
        launch_module._spawn_runtime = original
        other.close()


@pytest.mark.asyncio
async def test_every_errand_carries_an_identity_even_when_the_caller_omits_one(
    fleet: FakeRuntimeFleet, tmp_path: Path
) -> None:
    """Identity is what makes a retry safe, so it is never optional."""
    fleet.loop = asyncio.get_running_loop()
    await engage_runtime(
        SESSION_ID,
        str(tmp_path),
        PromptErrand(text="no id supplied"),
        config_dir=tmp_path,
    )
    prompts = [f for f in fleet.delivered if f.get("op") == "prompt"]
    assert len(prompts) == 1
    assert prompts[0]["command_id"], "engage must mint a command_id when the caller has none"


@pytest.mark.asyncio
async def test_a_retried_prompt_is_delivered_under_the_same_identity(
    fleet: FakeRuntimeFleet, tmp_path: Path
) -> None:
    """A caller's own id survives, which is what the runtime dedupes against."""
    fleet.loop = asyncio.get_running_loop()
    errand = PromptErrand(text="idempotent", command_id="cmd-stable-1")
    await engage_runtime(SESSION_ID, str(tmp_path), errand, config_dir=tmp_path)
    await engage_runtime(SESSION_ID, str(tmp_path), errand, config_dir=tmp_path)

    prompts = [f for f in fleet.delivered if f.get("op") == "prompt"]
    assert [f["command_id"] for f in prompts] == ["cmd-stable-1", "cmd-stable-1"]


@pytest.mark.asyncio
async def test_engage_times_out_rather_than_spinning_forever(tmp_path: Path, monkeypatch) -> None:
    """A runtime that never appears must fail the caller, not hang it."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "local_operator.session.runtime.launch._spawn_runtime",
        lambda session_id, cwd, *, defer_materialise: None,  # a spawn that never starts
    )
    with pytest.raises(TimeoutError):
        await engage_runtime(
            SESSION_ID,
            str(tmp_path),
            PromptErrand(text="nobody home"),
            config_dir=tmp_path,
            deadline_s=0.5,
        )


@pytest.mark.asyncio
async def test_a_candidate_that_dies_during_construction_is_respawned(
    tmp_path: Path, monkeypatch
) -> None:
    """The inverse failure: no runtime when one is owed.

    A candidate that wins the lease and then dies mid-construction (`process.py`
    returning 2, an OOM kill, a bad credential) leaves no record and no lease
    holder. Round 1 (R1) measured the caller sitting in the `spawned=True`
    branch for the FULL 30 s deadline before failing, while a fresh candidate
    would have acquired the lease immediately.

    Distinguishing this from the STARTING case is what makes the respawn safe:
    STARTING has a live lease holder, this has none. The test asserts the
    respawn happens AND that it is bounded, since a session that can never
    construct must report its error rather than loop.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions").mkdir(parents=True, exist_ok=True)

    attempts: list[str] = []

    class _DeadPopen:
        """The engage loop's death signal is ``poll() is not None`` — a live
        constructor is by definition still constructing (round 2, Q8)."""

        returncode = 2

        def poll(self) -> int:
            return self.returncode

    def dies_during_construction(
        session_id: str, cwd: str, *, defer_materialise: bool
    ) -> _DeadPopen:
        """Take the lease, then vanish — publishing no record."""
        from local_operator.session_lease import acquire_session_lease

        attempts.append(session_id)
        lease = acquire_session_lease(tmp_path / "sessions" / session_id)
        lease.release()
        return _DeadPopen()

    monkeypatch.setattr(
        "local_operator.session.runtime.launch._spawn_runtime", dies_during_construction
    )

    # RuntimeStartupError, not TimeoutError, and it must arrive well inside the
    # deadline. Waiting out the full deadline for a session whose every
    # candidate has already died is what made the first message in a new chat
    # look hung for 30 seconds before failing (QA Q1); a timeout here would
    # mean that behaviour had returned.
    started = time.monotonic()
    with pytest.raises(RuntimeStartupError):
        await engage_runtime(
            SESSION_ID, str(tmp_path), WarmErrand(), config_dir=tmp_path, deadline_s=6.0
        )
    elapsed = time.monotonic() - started
    assert elapsed < 5.0, f"engage burned {elapsed:.1f}s of a 6s deadline instead of failing fast"

    assert len(attempts) > 1, "a dead candidate was never respawned; R1 has regressed"
    assert len(attempts) <= _MAX_SPAWNS, "respawning is unbounded; a broken session crash-loops"


@pytest.mark.asyncio
async def test_a_child_that_dies_reports_its_own_reason_not_a_generic_failure(
    tmp_path: Path, monkeypatch
) -> None:
    """The child's fatal error must reach the caller, in words a user can act on.

    Both spawn streams used to go to DEVNULL, so a candidate that could never
    construct produced no traceback, no message and no exit reason anywhere the
    parent could see -- the whole reason a misconfigured session reported only
    "owner unavailable" (QA Q1).
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions").mkdir(parents=True, exist_ok=True)

    class _DeadPopen:
        returncode = 2

        def poll(self) -> int:
            return self.returncode

    def dies_with_a_reason(session_id: str, cwd: str, *, defer_materialise: bool) -> _DeadPopen:
        from local_operator.session_lease import acquire_session_lease

        acquire_session_lease(tmp_path / "sessions" / session_id).release()
        process = _DeadPopen()
        capture = tmp_path / f"capture-{len(list(tmp_path.glob('capture-*')))}.log"
        # Exactly the shape `process.py` now writes to stderr on a fatal
        # construction failure.
        capture.write_text(
            "Traceback (most recent call last):\n"
            '  File "x.py", line 1, in y\n'
            "local_operator.session_factory.HostingNotConfiguredError: "
            "Hosting platform is not configured.\n"
        )
        setattr(process, "lop_capture_path", capture)
        return process

    monkeypatch.setattr("local_operator.session.runtime.launch._spawn_runtime", dies_with_a_reason)

    with pytest.raises(RuntimeStartupError) as raised:
        await engage_runtime(
            SESSION_ID, str(tmp_path), WarmErrand(), config_dir=tmp_path, deadline_s=6.0
        )

    # The vetted, user-facing sentence -- not the raw exception text, which can
    # carry endpoint URLs and filesystem paths.
    assert "Settings > Providers" in raised.value.actionable
    assert "Traceback" not in raised.value.actionable
    assert "x.py" not in raised.value.actionable


def test_spawn_capture_is_private_and_anonymous(tmp_path, monkeypatch) -> None:
    """The child's stdio capture must not be world-readable, or name the session.

    It receives the runtime's ENTIRE stdout+stderr -- tracebacks, provider error
    bodies, config echoes -- and lives in the shared temp directory. Created
    through `Path.open("wb")` it took the process umask (measured 0o644 here),
    so any local user could read another user's provider errors; and because the
    name embedded `session_id`, a directory listing alone disclosed live session
    ids without opening anything (review round 2, MAJOR-2).
    """
    import os
    import stat

    from local_operator.session.runtime import launch as launch_module

    monkeypatch.setattr(os, "umask", lambda _mask: 0o022)
    spawned: list[Any] = []

    class _Popen:
        returncode = None

        def poll(self):
            return None

        def kill(self):
            return None

    def fake_popen(*args: Any, **kwargs: Any):
        # Record the handle the spawn opened, then hand back an inert process:
        # the file's PERMISSIONS are what is under test, not the child.
        spawned.append(kwargs["stdout"])
        return _Popen()

    monkeypatch.setattr(launch_module.subprocess, "Popen", fake_popen)

    session_id = "secret-session-id-abc123"
    process = launch_module._spawn_runtime(session_id, str(tmp_path), defer_materialise=True)
    capture = getattr(process, "lop_capture_path")
    try:
        mode = stat.S_IMODE(capture.stat().st_mode)
        assert not mode & stat.S_IROTH, f"world-readable: {oct(mode)}"
        assert not mode & stat.S_IRGRP, f"group-readable: {oct(mode)}"
        assert mode == 0o600, oct(mode)
        assert session_id not in capture.name, capture.name
        # The four-unlink lifecycle still needs a real path on the process.
        assert capture.exists()
    finally:
        capture.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Poll shape
#
# WHY THESE ARE SCHEDULE-SHAPE ASSERTIONS AND NOT TIME BOUNDS
# ===========================================================
# The property under test is "the parent does not sleep past a runtime that is
# already serving". The tempting way to assert that is to spawn a runtime and
# bound the wall time, and AGENTS.md ("Timing, flakes, and how to assert that
# something is fast") is emphatic that this repo has burned multiple PRs doing
# exactly that: a bound calibrated on a dev box is not a bound on CI, and the
# only honest conclusion at several sites was that no portable number exists.
#
# So these tests assert the SHAPE OF THE SCHEDULE the loop computes -- the
# actual sleep durations it requests, captured from a stubbed `asyncio.sleep`.
# That is a fact about what the code decided, not about how fast the machine
# ran it, so it cannot flake under load: the recorded values are identical on
# an idle laptop and a saturated 4-vCPU runner. It also fails deterministically
# if someone restores the old single-grid constants, which is the regression
# these guard.
# ---------------------------------------------------------------------------


class _sleep_recorder:
    """A stand-in for ``launch``'s ``asyncio`` that records what it was asked to sleep.

    WHY A PROXY OBJECT AND NOT ``setattr(launch_module.asyncio, "sleep", ...)``
    ==========================================================================
    ``launch_module.asyncio`` is not a per-module copy -- it is THE shared
    ``asyncio`` module object, so setting an attribute on it patches
    ``asyncio.sleep`` for every module in the process. Under ``-n auto`` the
    recorder then captures sleeps from whatever else is running in the same
    worker, and the assertion reports another test's timings as this one's
    (observed: 0.2 and 1.0 leaking in from ``tests/unit/tui/test_stop_command.py``,
    which failed a poll-shape test that passes in isolation).

    Rebinding the NAME on the module under test keeps the patch where the test
    can reason about it: only ``launch``'s own ``asyncio.sleep`` calls are
    recorded, and nothing outside this module is affected.
    """

    def __init__(self, slept: list[float], clock: dict[str, float] | None = None) -> None:
        self._slept = slept
        self._clock = clock

    async def sleep(self, duration: float) -> None:
        self._slept.append(duration)
        # When a virtual clock is in play, the loop's own deadline arithmetic
        # advances by exactly the sleeps it asked for -- time is DATA here,
        # never a measurement, so a 3 s window costs no real seconds.
        if self._clock is not None:
            self._clock["now"] += duration

    def __getattr__(self, name: str) -> Any:
        # Anything the module reaches for other than `sleep` is the real thing.
        return getattr(asyncio, name)


class _virtual_clock:
    """A stand-in for ``launch``'s ``time``, driven by the sleeps it requests.

    Scoped for the same reason as :class:`_sleep_recorder`: ``time`` is a shared
    module object, and a process-wide ``monotonic`` would be considerably worse
    than a process-wide ``sleep``.
    """

    def __init__(self, clock: dict[str, float]) -> None:
        self._clock = clock

    def monotonic(self) -> float:
        return self._clock["now"]

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)


def test_poll_delay_is_dense_while_a_construction_is_known_to_be_in_flight() -> None:
    """The unit fact: a known construction polls flat and dense, and the
    open-ended backoff is preserved rather than decayed underneath it.

    Carrying the backoff forward untouched is what makes the fallback correct:
    when the dense window expires the open-ended wait must start from the top,
    not from a value that decayed while we were watching a healthy child.
    """
    backoff = _POLL_INITIAL_S
    for elapsed in (0.0, 0.5, 1.2, _CONSTRUCTING_WINDOW_S - 0.001):
        delay, backoff = _poll_delay(backoff, elapsed)
        assert delay == _CONSTRUCTING_POLL_S
        assert backoff == _POLL_INITIAL_S, "the open-ended backoff decayed during a dense window"


def test_poll_delay_falls_back_to_the_open_ended_backoff() -> None:
    """With nothing known to be constructing, the exponential shape is intact.

    The backoff is what keeps a 30-second deadline from becoming a spin when
    the wait is genuinely open-ended (a contender's contended lease, a session
    that will never appear), so the dense regime must not have replaced it.
    """
    delay, backoff = _poll_delay(_POLL_INITIAL_S, None)
    assert delay == _POLL_INITIAL_S
    assert backoff == pytest.approx(_POLL_INITIAL_S * _POLL_FACTOR)

    # It must still climb, and still cap.
    for _ in range(40):
        delay, backoff = _poll_delay(backoff, None)
    assert delay == _POLL_CAP_S, "the open-ended backoff no longer reaches its cap"

    # A construction that outlives the dense window is no longer 'about to
    # publish', so it rejoins the open-ended shape rather than spinning.
    delay, _ = _poll_delay(_POLL_INITIAL_S, _CONSTRUCTING_WINDOW_S)
    assert delay == _POLL_INITIAL_S


@pytest.mark.asyncio
async def test_engage_polls_densely_while_its_own_candidate_constructs(
    tmp_path: Path, monkeypatch
) -> None:
    """The integration fact, and the regression guard for the ~150 ms of dead time.

    A candidate that we spawned and that is still alive is by definition still
    constructing. Under the old single grid the loop slept 50, 135, 280, 525 ms
    while waiting for it, so a child publishing at ~400 ms went unnoticed until
    525 ms. Here the whole recorded schedule must be the dense interval.

    The sleeps are RECORDED, not endured: `asyncio.sleep` is stubbed to a
    no-op, so this test asserts the schedule the loop computed without waiting
    for any of it.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions").mkdir(parents=True, exist_ok=True)

    class _LiveCandidate:
        """A spawned child that is alive and has published nothing yet."""

        returncode = None

        def poll(self) -> None:
            return None

    monkeypatch.setattr(
        "local_operator.session.runtime.launch._spawn_runtime",
        lambda session_id, cwd, *, defer_materialise: _LiveCandidate(),
    )

    slept: list[float] = []

    # Scoped to a PRIVATE asyncio proxy for this module, never to the shared
    # `asyncio` module object. `launch_module.asyncio` IS the global module, so
    # `setattr(launch_module.asyncio, "sleep", ...)` patches it for the whole
    # process and this recorder then captures unrelated sleeps from any test
    # running beside it (observed: 0.2 and 1.0 leaking in from the TUI suite).
    monkeypatch.setattr(launch_module, "asyncio", _sleep_recorder(slept))

    with pytest.raises(TimeoutError):
        await engage_runtime(
            SESSION_ID, str(tmp_path), WarmErrand(), config_dir=tmp_path, deadline_s=0.4
        )

    assert slept, "the loop never polled"
    # No wake exceeds the dense interval. Stated as a ceiling rather than as
    # equality because the loop clamps each sleep to the time left on the
    # deadline (`min(delay, deadline - now)`), so the last few wakes of a
    # short-deadline run are legitimately SMALLER than the interval.
    assert max(slept) <= _CONSTRUCTING_POLL_S, f"not a dense grid: {sorted(set(slept))}"
    # The unclamped body of the schedule is exactly the dense interval, and
    # the old grid's signature is absent: under 0.05/x1.7 the loop would have
    # slept 50, 85, 145, 246 ms and noticed a ~400 ms child only at 525 ms.
    # Those values are precisely what this guards against.
    assert _CONSTRUCTING_POLL_S in slept, "the dense interval was never requested"
    assert not [d for d in slept if d > _CONSTRUCTING_POLL_S], "an exponential wake survived"


@pytest.mark.asyncio
async def test_engage_uses_the_coarse_grid_when_nothing_is_constructing(
    tmp_path: Path, monkeypatch
) -> None:
    """The other half: an open-ended wait must NOT be polled densely.

    Once every spawn we are allowed has been made and no candidate is alive,
    there is no prior on when (or whether) a record appears. Polling that at
    100 Hz for the rest of a 30-second deadline is the spin the backoff exists
    to prevent, so the schedule here must climb rather than stay flat.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions").mkdir(parents=True, exist_ok=True)
    # A spawn that starts nothing: no candidate handle, no lease, no record --
    # so the loop can observe no construction in flight.
    monkeypatch.setattr(
        "local_operator.session.runtime.launch._spawn_runtime",
        lambda session_id, cwd, *, defer_materialise: None,
    )

    slept: list[float] = []

    # Scoped to a PRIVATE asyncio proxy for this module, never to the shared
    # `asyncio` module object. `launch_module.asyncio` IS the global module, so
    # `setattr(launch_module.asyncio, "sleep", ...)` patches it for the whole
    # process and this recorder then captures unrelated sleeps from any test
    # running beside it (observed: 0.2 and 1.0 leaking in from the TUI suite).
    monkeypatch.setattr(launch_module, "asyncio", _sleep_recorder(slept))

    with pytest.raises(TimeoutError):
        await engage_runtime(
            SESSION_ID, str(tmp_path), WarmErrand(), config_dir=tmp_path, deadline_s=5.0
        )

    assert len(slept) > 2
    assert slept[0] == _POLL_INITIAL_S
    # Strictly increasing until the cap: the exponential shape is intact.
    assert slept[1] == pytest.approx(_POLL_INITIAL_S * _POLL_FACTOR)
    assert _CONSTRUCTING_POLL_S not in slept, "an open-ended wait was polled at the dense interval"


@pytest.mark.asyncio
async def test_a_dense_window_does_not_outlive_its_welcome(tmp_path: Path, monkeypatch) -> None:
    """A construction that never finishes must stop being polled densely.

    The dense regime is a bet on a prior ("a record is about to appear"). A
    candidate still alive and recordless after `_CONSTRUCTING_WINDOW_S` has
    falsified that bet -- it is wedged or pathologically slow, not imminent --
    and continuing at 100 Hz for the remaining ~27 s of the deadline would be
    the spin the backoff exists to prevent. The schedule must therefore leave
    the dense interval behind, on its own, with the child still alive.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions").mkdir(parents=True, exist_ok=True)

    class _WedgedCandidate:
        returncode = None

        def poll(self) -> None:
            return None

    monkeypatch.setattr(
        "local_operator.session.runtime.launch._spawn_runtime",
        lambda session_id, cwd, *, defer_materialise: _WedgedCandidate(),
    )

    slept: list[float] = []
    # A virtual clock: the loop's own deadline arithmetic advances by exactly
    # the sleeps it asks for, so the dense window can expire without the test
    # waiting out three real seconds. Time is DATA here, never a measurement.
    clock = {"now": 0.0}

    # Same scoping as above, for both the sleep and the clock: `time` is a
    # shared module object too, and a process-wide `monotonic` is far worse.
    monkeypatch.setattr(launch_module, "asyncio", _sleep_recorder(slept, clock))
    monkeypatch.setattr(launch_module, "time", _virtual_clock(clock))

    with pytest.raises(TimeoutError):
        await engage_runtime(
            SESSION_ID, str(tmp_path), WarmErrand(), config_dir=tmp_path, deadline_s=10.0
        )

    dense = [d for d in slept if d == _CONSTRUCTING_POLL_S]
    coarse = [d for d in slept if d > _CONSTRUCTING_POLL_S]
    assert dense, "the dense regime never engaged"
    assert coarse, "the dense window never expired; a wedged child is polled forever"
    # The window is bounded by design, so the dense phase is ~_CONSTRUCTING_WINDOW_S
    # of virtual time and not one poll more.
    assert len(dense) == pytest.approx(_CONSTRUCTING_WINDOW_S / _CONSTRUCTING_POLL_S, rel=0.05)
    # ...and the fallback restarts the backoff from the top rather than from a
    # value that decayed while the child looked healthy.
    assert coarse[0] == _POLL_INITIAL_S


@pytest.mark.asyncio
async def test_a_published_record_that_refuses_the_dial_is_not_polled_densely(
    tmp_path: Path, monkeypatch
) -> None:
    """A live runtime that will not answer must stay on the SLOW path.

    The lease is held for a session's whole life -- ``session_factory``
    registers its release as a dispose hook -- so ``_lease_holder`` names a
    live pid for a fully constructed, happily serving runtime, not merely one
    that is starting. A runtime that has published a record but refuses the
    dial (wedged control socket, port exhaustion) therefore takes the loop's
    ``ConnectionError`` retry path with its lease still held.

    Reading that as "constructing" turned a slow retry into a hot spin:
    measured at 301 dial attempts inside a 3 s window against 8 under the
    open-ended grid (review round 1, MAJOR-1). The guard is the shape of the
    schedule -- no dense wake may be requested while a record exists -- which
    is a fact about what the loop decided and cannot flake under load.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions").mkdir(parents=True, exist_ok=True)

    record = SessionRecord(
        pid=os.getpid(),
        kind="daemon",
        session_id=SESSION_ID,
        conversation_name="wedged",
        cwd=str(tmp_path),
        model_label="test/model",
        control_port=1,
        control_key="k" * 16,
    )
    # A record IS present and the lease IS held by a live pid: construction is
    # over and this runtime is simply unreachable. `find_owner_record` is a
    # function-local import in the loop, so it is patched at its source module.
    monkeypatch.setattr(
        "local_operator.mobile.attach_client.find_owner_record",
        lambda config_dir, session_id: (record, os.getpid()),
    )
    monkeypatch.setattr(launch_module, "_lease_holder", lambda config_dir, session_id: os.getpid())

    dials = 0

    async def refuses_the_dial(*args: Any, **kwargs: Any) -> tuple[str, bool]:
        nonlocal dials
        dials += 1
        raise ConnectionError("control socket refuses")

    monkeypatch.setattr(launch_module, "_deliver", refuses_the_dial)
    # Nothing may be spawned: a record exists, so the loop must only be
    # retrying the dial. A spawn here would be a separate (arbitration) bug.
    monkeypatch.setattr(
        "local_operator.session.runtime.launch._spawn_runtime",
        lambda session_id, cwd, *, defer_materialise: pytest.fail("spawned despite a live record"),
    )

    slept: list[float] = []
    clock = {"now": 0.0}
    monkeypatch.setattr(launch_module, "asyncio", _sleep_recorder(slept, clock))
    monkeypatch.setattr(launch_module, "time", _virtual_clock(clock))

    with pytest.raises(TimeoutError):
        await engage_runtime(
            SESSION_ID, str(tmp_path), WarmErrand(), config_dir=tmp_path, deadline_s=3.0
        )

    assert dials > 0, "the dial was never attempted; the repro no longer exercises the retry path"
    assert (
        _CONSTRUCTING_POLL_S not in slept
    ), f"a published record was polled at the dense interval: {sorted(set(slept))}"
    # The open-ended shape is what should be running here, from the top.
    assert slept[0] == _POLL_INITIAL_S
    assert max(slept) > _CONSTRUCTING_POLL_S, "the schedule never backed off"
    # Retries are bounded by the backoff, not by the poll floor: an order of
    # magnitude fewer than the ~300 the dense regime produced over 3 s.
    assert dials < 30, f"{dials} dial attempts in a 3 s window is a retry storm"
