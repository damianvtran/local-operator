"""The session seam: ``Session.prompt`` expands ``@path`` before the turn.

This is the R5 guarantee under test — one call site is what gives the CLI,
headless, server, scheduler, mobile and subagent surfaces the feature without
per-surface work. The ordering test is the important one: expansion has to run
BEFORE ``_turn_lock`` is acquired, or an approval parks a human on the lock
that compaction also needs.
"""

from __future__ import annotations

import pytest

from local_operator.harness.types import StreamEndEvent
from tests.unit.session.test_session import ScriptedStream, make_session


def _sent_text(stream: ScriptedStream) -> str:
    """The text of the user message the model would have received."""
    return stream.requests[0].messages[-1].text


class SpyGate:
    """Records each ask and answers ``reply``; may assert while it is asked."""

    def __init__(self, reply: bool = True, on_ask=None) -> None:
        self.reply = reply
        self.asks: list[str] = []
        self._on_ask = on_ask

    async def __call__(self, tool_name: str, description: str) -> bool:
        if self._on_ask is not None:
            self._on_ask()
        self.asks.append(description)
        return self.reply


@pytest.mark.asyncio
async def test_prompt_expands_references_before_the_turn(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.md").write_text("THE FILE BODY\n", encoding="utf-8")
    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream, cwd=str(workspace))

    await session.prompt("what does @notes.md do?")

    sent = _sent_text(stream)
    assert "<operator-references>" in sent
    assert "THE FILE BODY" in sent
    # The operator's own sentence still leads the message: the block is
    # appended, never substituted in place.
    assert sent.startswith("what does @notes.md do?")
    await session.dispose()


@pytest.mark.asyncio
async def test_prompt_consults_the_approval_gate_for_an_outside_path(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "elsewhere.txt").write_text("OUTSIDE BODY", encoding="utf-8")
    gate = SpyGate(reply=False)
    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream, cwd=str(workspace), request_approval=gate)

    await session.prompt(f"read @{outside / 'elsewhere.txt'}")

    assert len(gate.asks) == 1
    assert str(outside / "elsewhere.txt") in gate.asks[0]
    assert "OUTSIDE BODY" not in _sent_text(stream)
    await session.dispose()


@pytest.mark.asyncio
async def test_yolo_bypasses_the_approval_gate(tmp_path):
    """Pins the ``None if self._yolo else self._request_approval`` branch at the
    call site — the same form ``_build_tool_context`` passes."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "elsewhere.txt").write_text("OUTSIDE BODY", encoding="utf-8")
    gate = SpyGate(reply=False)
    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream, cwd=str(workspace), request_approval=gate, yolo=True)

    await session.prompt(f"read @{outside / 'elsewhere.txt'}")

    assert gate.asks == []
    assert "OUTSIDE BODY" in _sent_text(stream)
    await session.dispose()


@pytest.mark.asyncio
async def test_expansion_runs_before_the_turn_lock_is_acquired(tmp_path):
    """The R3 deadlock guard, asserted structurally rather than by timing.

    An approval can park on a human indefinitely, and in the TUI the app
    awaiting this prompt is the same one drawing the approval card; holding
    ``_turn_lock`` across that also blocks the compaction that shares it. This
    is the test that catches the mitigation being implemented in the wrong
    place — moving the call inside the lock makes it fail deterministically.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "elsewhere.txt").write_text("body", encoding="utf-8")
    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    observed: list[bool] = []
    session = make_session(tmp_path, stream, cwd=str(workspace))
    gate = SpyGate(reply=True, on_ask=lambda: observed.append(session._turn_lock.locked()))
    session._request_approval = gate

    await session.prompt(f"read @{outside / 'elsewhere.txt'}")

    assert observed == [False]
    await session.dispose()


@pytest.mark.asyncio
async def test_a_declined_reference_still_completes_the_turn(tmp_path):
    """A declined approval is not an error: the token stays verbatim and the
    turn runs, because losing the user's message is the worse failure."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "elsewhere.txt").write_text("body", encoding="utf-8")
    typed = f"read @{outside / 'elsewhere.txt'}"
    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    session = make_session(
        tmp_path, stream, cwd=str(workspace), request_approval=SpyGate(reply=False)
    )

    await session.prompt(typed)

    assert len(stream.requests) == 1
    assert _sent_text(stream) == typed
    await session.dispose()
