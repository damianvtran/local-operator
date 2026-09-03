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
from pathlib import Path
from typing import Any

import pytest

from local_operator.session.runtime.launch import (
    PeerMessageErrand,
    PromptErrand,
    WarmErrand,
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
