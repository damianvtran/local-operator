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

The exclusivity fence (review R3) lives here too, because it is the OWNER that
has to enforce it: a move retires the runtime, and every facade attached at that
moment engages a successor from its OWN cwd, so two facades with different
directories make contradictory successor requests and the loser can win the
race with the old path. Propagating the target to arbitrary siblings is a
cross-viewer protocol this release deliberately does not ship, so the bounded
answer is to REFUSE while another actual attach is registered — enforced at the
latch, announced by a capability, and fail-closed on an owner that would ignore
the flag.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from local_operator import update as update_mod
from local_operator.session.runtime.server import RuntimeServer, _ClientConn
from local_operator.session.runtime.types import EXCLUSIVE_MOVE_CAPABILITY
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


async def _ask(
    server: RuntimeServer,
    sent: list[dict[str, Any]],
    conn: _ClientConn,
    **fields: Any,
) -> str:
    await server._on_request({"op": "retire_now", "req": 1, **fields}, conn)
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
    monkeypatch.setattr(update_mod, "disk_build", lambda *_a, **_k: BOOT)
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


class LatchingHandle(MovableHandle):
    """``MovableHandle`` with the safe retirement latch the fence promises.

    ``EXCLUSIVE_MOVE_CAPABILITY`` is advertised only by a handle that carries
    ``begin_retire``, because that latch is where the fence re-checks itself —
    so this double is what a real ``ServingSessionHandle`` looks like to
    ``RuntimeServer`` on that seam, and ``MovableHandle`` above is the REDUCED
    handle the capability must stay absent from. ``commit`` drives the two
    answers a real latch gives: it commits when the runtime is still idle, and
    returns ``False`` when work arrived inside the announcement window.
    """

    def __init__(self, *, reason: str = "", commit: bool = True) -> None:
        super().__init__(reason=reason)
        self.commit = commit
        self.retirements: list[tuple[str, str]] = []

    def begin_retire(self, cause: str, detail: str = "") -> bool:
        self.retirements.append((cause, detail))
        return self.commit


@pytest.mark.asyncio
async def test_an_exclusive_move_refuses_while_another_attach_is_registered() -> None:
    """The bounded answer to contradictory successors (review R3).

    A sibling facade would engage the successor from its OWN cwd, so the move is
    refused while another ACTUAL attach is registered. Refused BEFORE the
    announcement, nothing is stopped, and the fence is released: a runtime that
    stayed alive must admit viewers again, or a refused move would leave the
    session unreachable for the rest of its life.
    """
    handle = LatchingHandle()
    server, sent = _rig(handle)
    viewer, sibling = _conn("attach"), _conn("attach")
    server._clients[id(viewer.writer)] = viewer
    server._clients[id(sibling.writer)] = sibling

    detail = await _ask(server, sent, viewer, exclusive=True)

    assert detail.startswith("kept: This session is open in another terminal or attached client.")
    assert handle.stopped is False
    assert handle.retirements == [], "the latch was reached after the refusal"
    assert [f for f in sent if f.get("op") == "retiring"] == []
    assert server._exclusive_move_fence is None


@pytest.mark.asyncio
async def test_an_exclusive_move_retires_and_holds_the_fence_when_it_is_alone() -> None:
    """One actual attach, and the fence is RETAINED once retirement commits.

    Retaining it is the point: after this commit the runtime is going away and a
    facade admitted in the meantime would engage a successor from its own cwd,
    which is exactly the contradictory-successor race the fence exists to close.
    """
    handle = LatchingHandle()
    server, sent = _rig(handle)
    viewer = _conn("attach")
    server._clients[id(viewer.writer)] = viewer

    assert await _ask(server, sent, viewer, exclusive=True) == "retiring"
    assert handle.stopped is True
    assert server._exclusive_move_fence is viewer


@pytest.mark.asyncio
async def test_a_viewer_arriving_during_the_announcement_refuses_the_move(monkeypatch) -> None:
    """The re-check at the latch, not a second sample of the same count.

    Modelled here as the connection that the registration fence cannot catch: a
    dial whose frame was already read — so it passed ``_on_connection``'s check
    before the fence existed — and whose registration lands after the observer
    count. Neither the count nor the gate can see it, which is why the fence is
    re-checked at the latch; a runtime with a sibling already registered must
    keep itself rather than commit behind it.
    """
    handle = LatchingHandle()
    server, sent = _rig(handle)
    viewer, late = _conn("attach"), _conn("attach")
    server._clients[id(viewer.writer)] = viewer
    announce = server.announce_retiring

    async def announce_then_register(reason_label: str, *, to: str = "") -> None:
        await announce(reason_label, to=to)
        server._clients[id(late.writer)] = late

    monkeypatch.setattr(server, "announce_retiring", announce_then_register)

    detail = await _ask(server, sent, viewer, exclusive=True)

    assert detail.startswith("kept: This session is open in another terminal or attached client.")
    assert handle.stopped is False
    assert handle.retirements == []
    assert server._exclusive_move_fence is None


@pytest.mark.asyncio
async def test_a_latch_less_owner_neither_advertises_nor_honours_the_fence() -> None:
    """Fail CLOSED on an old owner, and never fall back to a plain retire.

    An owner that ignores the unknown ``exclusive`` field would retire
    unguarded, so the desktop only sends it after reading
    ``EXCLUSIVE_MOVE_CAPABILITY`` off the record — and a caller that sends it
    anyway gets a refusal naming the update. The capability itself is withheld
    from a REDUCED handle, because the fence's promise includes a re-check at
    ``begin_retire`` and a handle without that latch cannot keep it. The legacy
    op still works on the same handle, which is what makes this a refusal to
    move exclusively rather than a broken retirement.
    """
    reduced = MovableHandle()
    server, sent = _rig(reduced)
    assert EXCLUSIVE_MOVE_CAPABILITY not in server._record.capabilities
    viewer = _conn("attach")
    server._clients[id(viewer.writer)] = viewer

    detail = await _ask(server, sent, viewer, exclusive=True)

    assert detail == "kept: this runtime cannot move exclusively; /reload first"
    assert reduced.stopped is False
    assert [f for f in sent if f.get("op") == "retiring"] == []
    assert await _ask(server, sent, viewer) == "retiring"
    assert reduced.stopped is True

    latching, _ = _rig(LatchingHandle())
    assert EXCLUSIVE_MOVE_CAPABILITY in latching._record.capabilities


@pytest.mark.asyncio
async def test_a_stop_that_raises_after_the_latch_commits_keeps_the_fence() -> None:
    """Review round 2, N6: ``committed`` cannot be derived from the return only.

    ``request_stop`` runs AFTER ``begin_retire`` has already committed the
    retirement latch, so a raise there must not read as "nothing committed": the
    runtime is going away and a facade admitted now would engage the successor
    from its own cwd, which is the one state the retained fence exists for. The
    caller is still told the op failed — the error frame is the reply.
    """

    class Raising(LatchingHandle):
        def request_stop(self) -> None:
            raise RuntimeError("stop failed after the latch committed")

    handle = Raising()
    server, sent = _rig(handle)
    viewer = _conn("attach")
    server._clients[id(viewer.writer)] = viewer

    await server._on_request({"op": "retire_now", "req": 1, "exclusive": True}, viewer)

    assert handle.retirements, "the latch was never reached"
    assert server._retirement_committed is True
    assert server._exclusive_move_fence is viewer, "the fence was released while retiring"
    assert [f for f in sent if f.get("op") == "ack"] == []
    errors = [f for f in sent if f.get("op") == "error"]
    assert errors and "stop failed" in str(errors[-1].get("message"))
