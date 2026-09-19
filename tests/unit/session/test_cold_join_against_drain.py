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
