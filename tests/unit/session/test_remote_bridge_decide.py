"""``AttachedSession.bridge_decide`` — the decision door on a paired device (U1).

WHY THIS FILE EXISTS. The dock looks the ask ops up on whichever session holds
the ask (``tui/app.py::_deliver_owner_answer``), so a viewer that lacked
``bridge_decide`` left an agent-messages request undecidable from the desktop —
and that is the surface the plain door's refusal copy names ("the window that
runs this session or a paired device"). CI caught the absence on the owner fix
that made the dock call the name at all
(``tests/unit/session/test_remote_registries.py`` and
``tests/unit/session/test_viewer_protocol.py``), which pins PRESENCE; this pins
the BEHAVIOUR those guards cannot: the triple the wire op receives, and the
verdict shape the dock renders.

The owner's half of the door (the grant it consumes, the refusals it words) is
pinned in ``tests/unit/team_bridge/test_tb02_p1_provenance.py``; this is the
proxy, not a second decision path.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.approval import (
    mint_bridge_decide_grant,
    reset_bridge_decide_grants_for_tests,
)
from local_operator.session.attached import AttachedSession
from tests.unit.session.test_session import wait_for
from tests.unit.team_bridge.test_h5_session_glue import _tasks_pending


@pytest.fixture(autouse=True)
def _grants():
    """The grant table is process-global memory; every cell starts empty."""
    reset_bridge_decide_grants_for_tests()
    yield
    reset_bridge_decide_grants_for_tests()


@pytest.fixture
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Home, config and cwd out of the way of the developer's real ones."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def cold_session(tmp_path):
    async def _build() -> AttachedSession:
        return await AttachedSession.cold(
            "session-under-test",
            config_dir=tmp_path,
            cwd=str(tmp_path),
            takeover_factory=lambda *_a, **_k: None,
        )

    return _build


class FakeClient:
    """The attach client's decision surface, and nothing else."""

    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        #: Every call's exact arguments: the triple is what the far side
        #: validates against the request's own closed set, so the argument
        #: ORDER is part of the contract, not an implementation detail.
        self.decided: list[tuple[str, str, str]] = []
        # ``is_cold`` reads this — a bound facade is one whose client is up.
        self.connected = True

    async def bridge_decide(self, ask_id: str, question_id: str, word: str) -> str:
        self.decided.append((ask_id, question_id, word))
        if self.error is not None:
            raise self.error
        return "decided"


def _bind(session: AttachedSession, client: object) -> None:
    """Make ``session`` look bound, the way ``is_cold`` actually reads it.

    No ``dispose()`` in these cells: it walks the socket and waiter state a cold
    facade with a double never built (``client.close`` among them), which is why
    ``tests/unit/session/test_remote_move.py`` does not dispose either — the
    double stands in for a live client, and there is no process behind it.
    """
    session._client = client  # type: ignore[assignment]
    session._ready_for_events = True


@pytest.mark.asyncio
async def test_the_viewer_forwards_the_decision_triple_and_reports_the_owners_verdict(
    cold_session,
) -> None:
    """``(ask_id, question_id, word)`` verbatim, and ``{"ok": True}`` back.

    The word is the request's OPTION ID (``approve``/``deny``), never a drawn
    label: the door validates it against the request's own closed set, so a
    proxy that reshaped any of the three would refuse a real human decision.
    """
    session = await cold_session()
    client = FakeClient()
    _bind(session, client)
    assert await session.bridge_decide("b1", "bmsg_1", "approve") == {"ok": True}
    assert client.decided == [("b1", "bmsg_1", "approve")]


@pytest.mark.asyncio
async def test_a_wire_refusal_is_a_verdict_not_an_exception(cold_session) -> None:
    """The far side's sentence is what the surface renders, as on every sibling.

    A refusal that crossed the wire arrives as a ``RuntimeError`` (the client
    raises the owner's copy); the dock needs ``{"ok": False, "error": ...}`` so
    the card reports the request's own words instead of a traceback — the same
    contract ``respond_ask``/``decline_ask``/``dismiss_ask`` keep.
    """
    session = await cold_session()
    client = FakeClient(
        error=RuntimeError("that answer expired before it could be sent — please answer again")
    )
    _bind(session, client)
    outcome = await session.bridge_decide("b1", "bmsg_1", "approve")
    assert outcome["ok"] is False
    assert "expired before it could be sent" in outcome["error"]
    assert client.decided == [("b1", "bmsg_1", "approve")]


# -- the whole path on REAL objects: facade -> wire op -> door -> the POST -----------------


def _post_recorder(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Patch the decide TRANSPORT (the one external seam) and return its log.

    ``decide_from_human_answer`` is where a decision becomes an HTTP POST; the
    fence is patched because a real runtime is not standing behind these
    sessions. Same two patches ``test_tb02_p1_provenance`` uses, for the same
    reason: everything else in the path below is production.
    """
    from local_operator.team_bridge import authority as fence_mod
    from local_operator.team_bridge import decide as decide_mod

    posted: list[str] = []

    async def recording_decide(client: Any, decision: Any, **kwargs: Any) -> Any:
        posted.append(str(getattr(decision, "word", "")))
        return {"request": {}}

    monkeypatch.setattr(decide_mod, "decide_from_human_answer", recording_decide)
    monkeypatch.setattr(fence_mod, "authority_fence", lambda runtime: None)
    return posted


async def _bind_viewer_to_the_handle(tmp_path: Path, handle: Any) -> AttachedSession:
    """A viewer whose client is the wire op itself, bound the way ``is_cold`` reads it."""

    class WireClient:
        """The attach client's decision surface: one op, the socket removed."""

        def __init__(self) -> None:
            self.calls: list[tuple[str, str, str]] = []
            self.connected = True

        async def bridge_decide(self, ask_id: str, question_id: str, word: str) -> str:
            self.calls.append((ask_id, question_id, word))
            return await handle.bridge_decide(ask_id, question_id, word)

    viewer = await AttachedSession.cold(
        "viewer-under-test",
        config_dir=tmp_path,
        cwd=str(tmp_path),
        takeover_factory=lambda *_a, **_k: None,
    )
    _bind(viewer, WireClient())
    return viewer


async def _deliver_the_response_row(session: Any, ask_id: str) -> None:
    """Run the production delivery step the runtime's reconcile hands the row to."""
    queue = session.ask_queue()
    record = [row for row in queue.records() if str(row.get("ask_id") or "") == ask_id][-1]
    await session._append_injected_message(
        queue.refresh_delivery_message(queue._response_message(record))
    )


@pytest.mark.asyncio
async def test_the_viewer_decides_a_real_request_through_the_wire_op_and_posts_once(
    tmp_path: Path, isolated_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M1's whole path on REAL objects: facade → wire op → door → exactly ONE POST.

    The three cells above stop at the facade (a double), which is why the
    facade gap reached CI: they can see the forwarding but not whether it reaches
    a door that decides. This one runs the real path. A real ``Session`` with a
    real queued agent-messages request, its ``ServingSessionHandle`` — the seam
    every wire client crosses, and the reason the frame's arguments are the
    door's own — and the grant minted where ADMISSION mints it
    (``runtime/server.py``'s authority seam, called by its own function), so the
    POST counted below is production minus the socket.
    """
    from local_operator.session.runtime.serving import ServingSessionHandle
    from tests.unit.team_bridge.test_tb02_p1_provenance import (
        MESSAGE_ID,
        _enqueue_bridge_ask,
        _session,
    )

    posted = _post_recorder(monkeypatch)
    session = await _session(tmp_path)
    ask_id = _enqueue_bridge_ask(session, issuance_key="iss-u1-viewer")
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    viewer = await _bind_viewer_to_the_handle(tmp_path, handle)

    mint_bridge_decide_grant(
        session_id=session.session_id,
        ask_id=ask_id,
        question_id=MESSAGE_ID,
        word="approve",
    )
    assert await viewer.bridge_decide(ask_id, MESSAGE_ID, "approve") == {"ok": True}

    await _deliver_the_response_row(session, ask_id)
    await wait_for(lambda: len(posted) == 1 and not _tasks_pending(session), 5.0)
    assert posted == ["approve"], posted
    await session.dispose()


@pytest.mark.asyncio
async def test_without_the_admission_the_viewer_decides_nothing(
    tmp_path: Path, isolated_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No admitted frame → the door records, and the transport NEVER posts.

    This is the safety property the facade member must not weaken: the decision
    door answers ``{"ok": True}`` on an open row even with no grant, because the
    grant is consumed at the DISPATCHER rather than at the write
    (``session.py``: the designed §1.4 shape, and the reason the POST below is
    zero while the verdict is not a refusal). R2-1 — whether a holder that can
    have no admitted frame should instead be refused in words the human can read
    — is recorded as a deliberate follow-up: that sentence is user-visible copy
    for the design round, not a string to invent here.
    """
    from local_operator.session.runtime.serving import ServingSessionHandle
    from tests.unit.team_bridge.test_tb02_p1_provenance import (
        MESSAGE_ID,
        _enqueue_bridge_ask,
        _session,
    )

    posted = _post_recorder(monkeypatch)
    session = await _session(tmp_path)
    ask_id = _enqueue_bridge_ask(session, issuance_key="iss-u1-no-admission")
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    viewer = await _bind_viewer_to_the_handle(tmp_path, handle)

    # NO mint: nothing was admitted over the wire seam for this target.
    assert await viewer.bridge_decide(ask_id, MESSAGE_ID, "approve") == {"ok": True}

    await _deliver_the_response_row(session, ask_id)
    await wait_for(lambda: not _tasks_pending(session), 5.0)
    assert posted == [], "an unadmitted decision reached the decide transport"
    await session.dispose()
