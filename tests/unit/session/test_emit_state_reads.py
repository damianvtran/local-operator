"""The per-event fold gate must read one scalar, not deep-copy the state.

``Session._emit`` runs on EVERY emitted event on a session with any
subscriber -- and a parked sidebar source is a subscriber -- to keep the
canonical ``history_generation`` in step. It used to ask for that one int
through ``FrontendStateStore.state``, the property that deep-copies every job,
usage row and trajectory so no caller can mutate the store's own instance:
measured at ~0.3-1 ms of loop CPU per event on a modest state, scaling with
the state's size, on the busiest path a streaming session has.

The guard is structural, not a stopwatch: with the gate open (a real
subscriber), emitting a plain event must not read ``state`` at all. If a
future change reintroduces any full-state read on this path, this fails.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from local_operator.harness.types import Message, MessageUpdateEvent
from local_operator.session.frontend_state import FrontendStateStore
from tests.e2e.harness import ScriptedStream, build_session


@pytest.mark.asyncio
async def test_emit_reads_history_generation_without_cloning_the_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = build_session(tmp_path / "sessions" / "emit-gate", ScriptedStream([]), cwd=tmp_path)

    # Open the fold gate the way production does: a real subscriber on the
    # canonical store. Without this the gate short-circuits and the test would
    # pass vacuously.
    session.subscribe_frontend(lambda update: None)
    store = session._frontend_state_store
    assert store is not None and store.has_subscribers

    reads: list[int] = []
    original = FrontendStateStore.state.fget
    assert original is not None

    def counted(self: FrontendStateStore):  # type: ignore[no-untyped-def]
        reads.append(len(reads))
        return original(self)

    monkeypatch.setattr(FrontendStateStore, "state", property(counted))

    message = Message.assistant("gate probe")
    message.id = "emit-gate-1"
    await session._emit(MessageUpdateEvent(message=message, delta="gate probe"))
    await asyncio.sleep(0)

    assert reads == [], (
        "emitting one message_update read FrontendStateStore.state, which deep-copies "
        "every job, usage row and trajectory: the history_generation compare must go "
        "through read_field (the allow-listed scalar accessor)"
    )

    await session.dispose()
