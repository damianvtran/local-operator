"""A COLD viewer can join a runtime that has latched a drain (memo §7.2, R1).

THE CELL THE JOINTABILITY FIX NEEDS. Round 1's fix made the engage answer a
draining record with a detail instead of raising; the arm offered for it dialled
the record *directly* (``AttachedSession.connect(record)``), which never runs the
engage at all — so that arm passed on a tree where the defect was present, and
the regression the fix removes stayed unpinned (agent review round 2, MAJOR-1).

This one drives the path R1 named: ``AttachedSession.cold(...)._ensure_bound()``
— the engage, then the dial — against a record that is LIVE and carrying a
drain's own phrase. The measurement is the reviewer's: a real published record
whose pid is this process, a real session lease, and a control port that counts
connections and then says nothing. The count is the discriminator, because it
answers exactly the question R1 is about — did the bind reach the live runtime?

On the pre-fix tree the engage withholds the join: ``ConnectionError("the runtime
is reconnecting")`` with **zero** dials, measured from the same rig. Here it must
reach the socket. The stub closes each connection, so the bind still ends in an
error — that error is not the point; the dial count is.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from local_operator.session.runtime import launch, registry
from local_operator.session.runtime.types import (
    LEAVING_FOR_BUILD,
    PROTOCOL_VERSION,
    SessionRecord,
)
from local_operator.session_lease import acquire_session_lease

SESSION_ID = "coldjoindrain"


class _CountingControlPort:
    """A control port that counts dials and then hangs up."""

    def __init__(self) -> None:
        self.dials = 0
        self._server: asyncio.AbstractServer | None = None
        self.port = 0

    async def start(self) -> None:
        async def on_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            self.dials += 1
            try:
                await asyncio.sleep(0.2)
            finally:
                writer.close()

        self._server = await asyncio.start_server(on_client, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()


@pytest.mark.asyncio
async def test_a_draining_session_is_joinable_from_a_cold_facade(tmp_path: Path) -> None:
    """The cold bind reaches a live, draining runtime instead of withholding."""
    from local_operator.session.attached import AttachedSession

    config = tmp_path / ".local-operator"
    session_dir = config / "sessions" / SESSION_ID
    session_dir.mkdir(parents=True)
    # A REAL lease held by this process, so the owner the registry reports is a
    # live pid — the state the incident's session was in for 1 h 40 m.
    acquire_session_lease(session_dir, pid=os.getpid())

    port = _CountingControlPort()
    await port.start()
    try:
        record = SessionRecord(
            pid=os.getpid(),
            kind="tui",
            session_id=SESSION_ID,
            conversation_name="cold-join",
            cwd=str(tmp_path),
            model_label="test/model",
            control_port=port.port,
            control_key="k" * 16,
            protocol=PROTOCOL_VERSION,
            capabilities=["tui_state_v1"],
            leaving=LEAVING_FOR_BUILD,
            started=True,
        )
        registry.publish(record, config)

        async def never_take_over() -> None:
            raise AssertionError("a cold viewer never takes over a session")

        facade = await AttachedSession.cold(
            SESSION_ID, config_dir=config, cwd=str(tmp_path), takeover_factory=never_take_over
        )
        started = time.monotonic()
        failure: BaseException | None = None
        try:
            await asyncio.wait_for(facade._ensure_bound(), timeout=45)
        except BaseException as exc:  # noqa: BLE001 — the dial, not success, is the assertion
            failure = exc
        finally:
            await facade.dispose()
    finally:
        await port.stop()

    # THE DISCRIMINATOR FIRST, so the red on a pre-fix tree is the BEHAVIOUR and
    # not a name that tree has never heard of. On `b6cdeec3` this reads
    # `dials=0 ... ConnectionError: the runtime is reconnecting`, because
    # `_deliver` raised before the bind's dial loop was ever entered.
    assert port.dials >= 1, (
        "the cold bind never reached the live record: "
        f"dials={port.dials} after {time.monotonic() - started:.1f}s, failure={failure!r}"
    )
    assert not (
        failure is not None and "reconnecting" in str(failure)
    ), f"the bind reported a live session as unreachable: {failure!r}"
    # And the state the engage reports for a leaving record, named once, so a
    # future edit that drops the detail fails here rather than in the field.
    assert launch.LEAVING_DETAIL == "owner-leaving", launch.LEAVING_DETAIL


@pytest.mark.asyncio
async def test_a_draining_session_hands_a_cold_facade_its_canonical_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sync LANDED, not merely the dial — the operator's symptom, on a real owner.

    The cell above stops at the dial count, which is what made it the reviewer's
    discriminator for R1 (a pre-fix tree never dialled at all). A dial is not the
    operator's problem, though: the reported failure was
    ``RuntimeUnresponsiveError`` from an attach whose socket was open, authenticated,
    and still missing its canonical state, on a session that had latched a drain and
    was keeping three subagent lanes stepping. So this arm drives the same cold
    facade against a REAL ``RuntimeServer`` that has genuinely latched a drain
    (``serving.begin_drain`` + the record the announce publishes), and asserts the
    one fact only the wire can establish: the client learned the owner's canonical
    EPOCH and SEQUENCE, which the pump sets from a ``frontend_sync`` frame and from
    nothing else (``AttachClient``'s handler for that op raises on a malformed one and
    is the sole writer of both fields — the deltas that follow are refused unless they
    continue that exact pair).

    Kept as a SECOND arm rather than folded into the first because the two fail for
    different reasons and a merged one would report the wrong one: a withheld engage
    (R1) fails the dial count, while a bind that misses its envelope fails here with
    the counts green — which is precisely how the incident presented in the field.
    """
    from dataclasses import replace as _replace

    from local_operator.session.attached import AttachedSession
    from local_operator.session.frontend_state import FrontendStateStore
    from local_operator.session.runtime.server import RuntimeServer
    from local_operator.session.runtime.serving import ServingSessionHandle
    from tests.unit.session.runtime.test_server import FakeHandle, _wait_record

    config = tmp_path / ".local-operator"
    session_dir = config / "sessions" / SESSION_ID
    session_dir.mkdir(parents=True)
    # A REAL lease held by this process: the owner the registry reports is a live pid,
    # which is the state the incident's session was in for 1 h 40 m.
    acquire_session_lease(session_dir, pid=os.getpid())
    # The runtime below publishes its own discovery record, and this is the root it
    # publishes INTO — the same one the facade is told to read, and the ambient root
    # as well, so nothing here can reach a store outside this tmp_path.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))

    handle = FakeHandle()
    # ONE HANDLE, ONE SESSION: the reduced handle names itself ``s1``, and a facade
    # that joins under another id is refused by the owner as "another conversation"
    # before any state is sent — so the double is aligned with the session under test
    # rather than left with the stub's own id.
    handle._projection = _replace(handle._projection, session_id=SESSION_ID)
    handle._frontend = FrontendStateStore(
        handle._frontend.state.model_copy(update={"session_id": SESSION_ID})
    )

    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    facade: object | None = None
    try:
        # THE DRAIN IS REAL: the production latch on the handle, and the phrase its
        # announcer publishes on the record (``announce_retiring`` writes both in one
        # call; this is the record half, so the facade's engage meets a live, DRAINING
        # owner rather than a hand-written record that no runtime stands behind).
        assert ServingSessionHandle.begin_drain(handle, "stale-build", "0.62.9 -> 0.62.12")
        runtime.note_leaving(LEAVING_FOR_BUILD)
        record = await _wait_record()
        assert record.leaving == LEAVING_FOR_BUILD, record.leaving
        assert handle._draining is True, "the owner has to be draining for this cell"

        async def never_take_over() -> None:
            raise AssertionError("a cold viewer never takes over a session")

        facade = await AttachedSession.cold(
            SESSION_ID, config_dir=config, cwd=str(tmp_path), takeover_factory=never_take_over
        )
        await asyncio.wait_for(facade._ensure_bound(), timeout=45)

        client = facade._client
        owner = handle._frontend.state
        assert client._frontend_epoch == owner.epoch, (
            "the cold facade never learned the owner's canonical epoch, so no "
            "frontend_sync frame reached it: dialling is not joining"
        )
        assert client._frontend_sequence == owner.sequence, (
            f"the facade holds sequence {client._frontend_sequence} against the "
            f"owner's {owner.sequence}: its canonical state is not the owner's"
        )
    finally:
        if facade is not None:
            await facade.dispose()
        runtime.close()
