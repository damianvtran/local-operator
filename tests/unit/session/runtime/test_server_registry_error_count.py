"""The unreadable-agent COUNT across the attach transport.

``ProfileRegistryUnavailable`` gained a count so the message tells the operator
how many definitions to go and repair. That detail only helps if it reaches the
surface that motivated it: ``/team <name>`` over an attach client renders the
error the client RECONSTRUCTS from the frame, not the owner's string, so a
count that stops at the socket leaves that surface showing the original
unactionable sentence.

The integer travels in its own ``error_count`` field rather than inside the
message. ``session/errors.py`` admits an enumerated category across this
boundary precisely because arbitrary prose may carry paths, socket addresses or
another conversation's identity; an integer carries none of those, so the
wording is still rebuilt locally from the code.
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest

from local_operator.session.errors import (
    AttachmentUnavailable,
    ProfileRegistryUnavailable,
    admission_error,
)
from local_operator.session.runtime.server import RuntimeServer, _ClientConn
from tests.unit.session.runtime.test_server import FakeHandle


def _rig(exc: Exception) -> tuple[RuntimeServer, list[dict[str, Any]], _ClientConn]:
    """A server whose dispatch raises ``exc``, with its socket writes captured."""
    server = RuntimeServer(FakeHandle(), kind="tui")
    sent: list[dict[str, Any]] = []

    async def capture(target, frame):  # noqa: ANN001
        sent.append(frame)

    async def failing_dispatch(op, frame):  # noqa: ANN001
        raise exc

    server._send_to = capture  # type: ignore[assignment]
    server._dispatch = failing_dispatch  # type: ignore[assignment]
    conn = _ClientConn(writer=cast(Any, object()), kind=cast(Any, "attach"))
    server._clients[id(conn.writer)] = conn
    return server, sent, conn


async def _error_frame(exc: Exception) -> dict[str, Any]:
    server, sent, conn = _rig(exc)
    await server._on_request({"op": "prompt", "req": 1}, conn)
    errors = [f for f in sent if f.get("op") == "error"]
    assert errors, f"expected an error frame, got {sent}"
    # Round-trip through JSON: the frame really goes over a socket.
    return cast(dict[str, Any], json.loads(json.dumps(errors[0])))


@pytest.mark.asyncio
async def test_registry_error_count_reaches_the_attach_client() -> None:
    """End to end: raise on the owner, decode on the client, keep the count."""
    frame = await _error_frame(ProfileRegistryUnavailable(count=3))

    assert frame["error_code"] == ProfileRegistryUnavailable.code
    assert frame["error_count"] == 3

    # Exactly what attach_client.py does with the reply.
    known = admission_error(str(frame.get("error_code", "")), frame.get("error_count"))
    assert isinstance(known, ProfileRegistryUnavailable)
    assert "3 agent definitions could not be read" in str(known)


@pytest.mark.asyncio
async def test_countless_registry_error_sends_no_count_field() -> None:
    """A scan that died on OSError has no number, and must not invent one."""
    frame = await _error_frame(ProfileRegistryUnavailable())

    assert frame["error_code"] == ProfileRegistryUnavailable.code
    assert "error_count" not in frame

    known = admission_error(str(frame.get("error_code", "")), frame.get("error_count"))
    assert "could not be read." not in str(known)


@pytest.mark.asyncio
async def test_other_admission_categories_carry_no_count() -> None:
    """``error_count`` belongs to one category, not to error frames generally."""
    frame = await _error_frame(AttachmentUnavailable())

    assert frame["error_code"] == AttachmentUnavailable.code
    assert "error_count" not in frame


@pytest.mark.asyncio
async def test_registry_error_frame_leaks_no_path() -> None:
    """The offending paths stay in the local log; only the integer crosses."""
    frame = await _error_frame(ProfileRegistryUnavailable(count=2))

    wire = json.dumps(frame)
    for token in ("/Users", "/private", "/tmp", ".yml", "agents/"):
        assert token not in wire, f"{token} crossed the transport boundary"
