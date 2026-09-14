"""The ``variables`` control op across the wire: refusal, framing, pass-through.

Two mixed-version facts meet at this op and both are asserted here rather than
left to the reader:

* an owner whose build predates the op answers ``unknown op: 'variables'`` — the
  same string the transport raises for an op it does not list at all — and the
  viewer turns exactly that into ``unsupported``. A different sentence would
  reach the panel as a 503, telling the user to reconnect when the honest answer
  is "update the backend";
* a frame that is not shaped like a code-memory request is REFUSED before the
  handle sees it. The payload path does not run ``validate_control_frame`` for
  every op, so this arm does it itself: the op writes into a live interpreter
  namespace, and an unknown action silently resolving to a write is a
  wrong-semantics bug on the one verb that mutates user state.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.session.attached import AttachedSession
from local_operator.session.runtime.server import RuntimeServer
from tests.unit.session.runtime.test_server import FakeHandle


class RecordingHandle(FakeHandle):
    """FakeHandle plus the capability, recording what it was asked for."""

    def __init__(self) -> None:
        super().__init__()
        self.variables_calls: list[tuple[Any, ...]] = []
        self.answer: dict[str, Any] = {"ok": True, "state": "ok"}

    async def variables_op(self, action: str, key: str, value: str, value_type: str):
        self.variables_calls.append((action, key, value, value_type))
        return self.answer


def _server(handle: Any) -> RuntimeServer:
    """A runtime around ``handle`` — no socket, no registration, no record.

    ``_dispatch_payload`` reads only ``self._handle``, and starting the whole
    runtime would test the registry instead of the dispatch arm.
    """
    return RuntimeServer(handle, kind="tui")


@pytest.mark.asyncio
async def test_an_old_handle_answers_the_unknown_op_error() -> None:
    """A handle that cannot answer must not be reported as an empty namespace."""
    server = _server(FakeHandle())

    with pytest.raises(ValueError, match="unknown op: 'variables'"):
        await server._dispatch_payload("variables", {"op": "variables", "action": "list"})


@pytest.mark.asyncio
async def test_the_verb_reaches_the_handle_with_the_callers_arguments() -> None:
    handle = RecordingHandle()
    handle.answer = {"ok": True, "state": "observed", "kernel": "resident", "variables": []}
    server = _server(handle)

    data = await server._dispatch_payload(
        "variables",
        {"op": "variables", "action": "set", "key": "n", "value": "42", "type": "int"},
    )

    assert handle.variables_calls == [("set", "n", "42", "int")]
    assert data == handle.answer


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "frame",
    [
        {"op": "variables"},
        {"op": "variables", "action": "explode"},
        {"op": "variables", "action": "list", "key": 42},
        {"op": "variables", "action": "set", "key": "n", "value": ["not", "text"]},
        {"op": "variables", "action": "set", "key": "n", "value": "1", "type": None},
    ],
)
async def test_a_malformed_frame_is_refused_before_the_handle(frame) -> None:
    handle = RecordingHandle()
    server = _server(handle)

    with pytest.raises(ValueError):
        await server._dispatch_payload("variables", frame)

    assert handle.variables_calls == [], "a malformed frame must not reach the session"


@pytest.mark.asyncio
async def test_the_viewer_reports_an_old_owner_as_unsupported() -> None:
    """The mapping that makes a build skew a STATE rather than a transport error."""

    class OldClient:
        connected = True

        async def variables(self, *args: Any):
            raise RuntimeError("unknown op: 'variables'")

    # Annotated ``Any`` on purpose: the double is deliberately NOT a real
    # ``AttachedSession`` (these two attributes are the whole contract the
    # method reads), and a structural annotation would make pyright police
    # ~110 members the method never touches.
    stub: Any = SimpleNamespace(_client=OldClient(), _recovering=False)

    assert await AttachedSession.variables_op(stub, "list") == {"state": "unsupported"}


@pytest.mark.asyncio
async def test_the_viewer_does_not_swallow_a_real_owner_failure() -> None:
    """Only the unknown-op sentence is a build fact; anything else must surface."""

    class BrokenClient:
        connected = True

        async def variables(self, *args: Any):
            raise ValueError("the owner's store is unreadable")

    stub: Any = SimpleNamespace(_client=BrokenClient(), _recovering=False)

    with pytest.raises(ValueError, match="unreadable"):
        await AttachedSession.variables_op(stub, "list")
