"""``retire_now`` on the RuntimeServer — `/move`'s side of the wire.

A sibling of ``refresh_if_idle`` (see ``test_server_refresh``) differing in one
term, and the tests here exist to pin exactly that difference: a refresh is
owed only when the build on disk has moved, whereas a move is owed whenever the
user asked for it. Everything after that decision is deliberately shared, so
these also check that a move leaves by the same ``retiring`` announcement — the
frame a viewer already knows means "a successor is owed; engage one".

Every uncertain answer is ``kept``, the asymmetry the whole file family
records: a wrong "retire" costs a cold start nobody asked for, a wrong "keep"
costs one more check.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from local_operator import update as update_mod
from local_operator.session.runtime.server import RuntimeServer, _ClientConn
from local_operator.update import BuildStamp
from tests.unit.session.runtime.test_server import FakeHandle

BOOT = BuildStamp(version="0.51.0", source_ref="abc1234567890")


class MovableHandle(FakeHandle):
    def __init__(self, *, reason: str = "") -> None:
        super().__init__()
        self.reason = reason
        self.stopped = False
        self.probes = 0

    def may_refresh(self) -> str:
        self.probes += 1
        return self.reason

    def request_stop(self) -> None:
        self.stopped = True


def _conn(kind: str = "attach") -> _ClientConn:
    return _ClientConn(writer=cast(Any, object()), kind=cast(Any, kind))


def _rig(handle: Any) -> tuple[RuntimeServer, list[dict[str, Any]]]:
    server = RuntimeServer(handle, kind="tui")
    server._boot_build = BOOT
    sent: list[dict[str, Any]] = []

    async def capture(target, frame):  # noqa: ANN001
        sent.append({"_recipient": target.kind, **frame})

    server._send_to = capture  # type: ignore[assignment]
    return server, sent


async def _ask(server: RuntimeServer, sent: list[dict[str, Any]], conn: _ClientConn) -> str:
    await server._on_request({"op": "retire_now", "req": 1}, conn)
    replies = [f for f in sent if f.get("op") in ("ack", "error")]
    assert replies, "the op never replied"
    reply = replies[-1]
    assert reply.get("op") == "ack", f"unexpected reply: {reply}"
    return str(reply.get("detail", ""))


@pytest.mark.asyncio
async def test_an_idle_runtime_announces_retiring_and_stops() -> None:
    handle = MovableHandle(reason="")
    server, sent = _rig(handle)
    viewer = _conn("attach")
    server._clients[id(viewer.writer)] = viewer

    assert await _ask(server, sent, viewer) == "retiring"
    assert handle.stopped is True
    announced = [f for f in sent if f.get("op") == "retiring"]
    assert [f["_recipient"] for f in announced] == ["attach"]
    assert announced[0]["reason"] == "moved"


@pytest.mark.asyncio
async def test_a_move_does_NOT_require_a_newer_build_on_disk(monkeypatch) -> None:
    """The one term this op drops, and the whole reason it is not a reuse of
    ``refresh_if_idle`` — which would answer "kept: build on disk matches"."""
    monkeypatch.setattr(update_mod, "installed_build", lambda *_a, **_k: BOOT)
    handle = MovableHandle(reason="")
    server, sent = _rig(handle)
    viewer = _conn("attach")
    server._clients[id(viewer.writer)] = viewer

    assert await _ask(server, sent, viewer) == "retiring"
    assert handle.stopped is True


@pytest.mark.asyncio
async def test_a_busy_runtime_keeps_itself_and_says_why() -> None:
    handle = MovableHandle(reason="busy")
    server, sent = _rig(handle)
    viewer = _conn("attach")
    server._clients[id(viewer.writer)] = viewer

    assert await _ask(server, sent, viewer) == "kept: busy"
    assert handle.stopped is False
    assert not [f for f in sent if f.get("op") == "retiring"]


@pytest.mark.asyncio
async def test_work_arriving_while_retiring_was_announced_keeps_the_runtime() -> None:
    """The re-check after the one await between decision and stop — the same
    guard ``_retire_if_pristine`` documents, and the race the viewer's own
    idle read cannot close."""

    class Racing(MovableHandle):
        def may_refresh(self) -> str:
            self.probes += 1
            # Idle on the first probe, busy by the second: a turn opened while
            # the announcement was draining.
            return "" if self.probes == 1 else "busy"

    handle = Racing()
    server, sent = _rig(handle)
    viewer = _conn("attach")
    server._clients[id(viewer.writer)] = viewer

    detail = await _ask(server, sent, viewer)
    assert detail == "kept: busy (arrived while retiring was announced)"
    assert handle.stopped is False


@pytest.mark.asyncio
async def test_a_handle_that_cannot_judge_itself_is_kept() -> None:
    """Unknown state is not an invitation to stop a runtime."""
    server, sent = _rig(FakeHandle())
    viewer = _conn("attach")
    server._clients[id(viewer.writer)] = viewer

    assert (await _ask(server, sent, viewer)).startswith("kept:")


@pytest.mark.asyncio
async def test_a_failing_idle_probe_keeps_the_runtime() -> None:
    class Broken(MovableHandle):
        def may_refresh(self) -> str:
            raise RuntimeError("probe exploded")

    handle = Broken()
    server, sent = _rig(handle)
    viewer = _conn("attach")
    server._clients[id(viewer.writer)] = viewer

    assert "idle probe failed" in await _ask(server, sent, viewer)
    assert handle.stopped is False
