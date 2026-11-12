"""Team Bridge latch vs a card parked BEFORE the latch (security r7 lead on #3796).

The lead, reproduced on a real spawned runtime before this fix: a viewer card
raised while the session was clean was never redrawn after the latch, and its
``G`` routed ``approvals default auto`` with no ``bridge_ack``. The owner
applied it (``tool_approval_mode: auto`` written), and global auto covers a
latched session's cards by design, so the next egress call ran with no card.

Rows pinned here:
- the owner refuses an unacked routed ``default auto`` while latched, on both
  owner hosts (the detached runtime and a TUI-hosted owner), and writes nothing;
- the same command WITH ``bridge_ack`` is honoured (typed / acked desktop);
- the owner re-publishes parked cards as latched, on their same request ids,
  and a viewer redraws from that re-publish (its mirror latch fires);
- a non-bool ``bridge_ack`` (the string ``"false"``) is never sent as true
  (S-R7-1) from either client seam.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.config import ConfigManager
from local_operator.config_watch import _reset_for_tests
from local_operator.harness.bridge_egress import BridgeTaintLatch, bridge_card_notice
from local_operator.mobile.tui_handle import TuiSessionHandle
from local_operator.mobile.types import PendingRequest
from local_operator.session.attached import AttachedSession
from local_operator.session.frontend_state import SlashResult
from local_operator.session.runtime.server import RuntimeServer
from tests.unit.session.runtime.test_server import FakeHandle
from tests.unit.session.runtime.test_serving_approvals_live import _handle
from tests.unit.session.test_remote import _never_take_over, _wait_record


@pytest.fixture(autouse=True)
def _fresh_registry():
    _reset_for_tests()
    yield
    _reset_for_tests()


def _mode(config_dir: Path) -> Any:
    return ConfigManager(config_dir).get_config_value("tool_approval_mode")


@pytest.mark.asyncio
@pytest.mark.parametrize("argument", ["default auto", "default yolo", "default  off"])
async def test_routed_default_auto_is_refused_while_latched_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, argument: str
) -> None:
    handle, session, watcher, _emitted = _handle(tmp_path, auto_approve=False)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(watcher.config_dir))
    setattr(session, "bridge_latched", True)

    refused = handle._approvals_slash(session, argument, SlashResult)

    assert refused.data.get("ok") is False
    assert refused.data.get("approval_reason") == "bridge_latched"
    # The kind-selected ``ack_refused`` copy, the same receipt as session auto.
    assert refused.text == handle._approvals_slash(session, "auto", SlashResult).text
    assert _mode(watcher.config_dir) == "ask"
    assert handle._auto_approve is False
    await handle.dispose()


@pytest.mark.asyncio
async def test_routed_default_auto_with_the_ack_is_honoured_while_latched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handle, session, watcher, _emitted = _handle(tmp_path, auto_approve=False)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(watcher.config_dir))
    setattr(session, "bridge_latched", True)

    receipt = handle._approvals_slash(session, "default auto", SlashResult, bridge_ack=True)

    assert receipt.data.get("ok") is True, receipt
    assert _mode(watcher.config_dir) == "auto"
    # Tightening never needs the ack, latched or not.
    back = handle._approvals_slash(session, "default ask", SlashResult)
    assert back.data.get("ok") is True and _mode(watcher.config_dir) == "ask"
    await handle.dispose()


@pytest.mark.asyncio
async def test_unlatched_routed_default_auto_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control: the refusal is the latch's, not a new rule for everyone."""
    handle, session, watcher, _emitted = _handle(tmp_path, auto_approve=False)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(watcher.config_dir))
    setattr(session, "bridge_latched", False)

    receipt = handle._approvals_slash(session, "default auto", SlashResult)

    assert receipt.data.get("ok") is True and _mode(watcher.config_dir) == "auto"
    await handle.dispose()


@pytest.mark.parametrize(
    ("args", "ack", "refused"),
    [
        ("default auto", False, True),
        ("default yolo", False, True),
        ("auto", False, True),
        ("default auto", True, False),
        ("default ask", False, False),
    ],
)
def test_a_tui_hosted_owner_refuses_unacked_default_auto_while_latched(
    args: str, ack: bool, refused: bool
) -> None:
    owner = SimpleNamespace(_session=lambda: SimpleNamespace(bridge_latched=True))
    refusal = TuiSessionHandle._bridge_ack_refusal
    receipt = refusal(owner, "approvals", args, ack)  # type: ignore[arg-type]
    if not refused:
        assert receipt is None
        return
    assert receipt is not None
    assert receipt["data"] == {"ok": False, "approval_reason": "bridge_latched"}


@pytest.mark.asyncio
async def test_the_latch_republishes_a_parked_card_as_latched_on_its_request_id(
    tmp_path: Path,
) -> None:
    handle, session, _watcher, _emitted = _handle(tmp_path, auto_approve=False)
    latch = BridgeTaintLatch()
    setattr(session, "bridge_taint_latch", latch)
    setattr(session, "bridge_latched", False)
    handle._watch_bridge_latch()
    parked = asyncio.ensure_future(handle._approval_gate("write", "write: parked.txt"))
    await asyncio.sleep(0)
    clean = handle._fold.projection.pending
    assert clean is not None and clean.bridge_latched is False

    setattr(session, "bridge_latched", True)
    latch.mark()

    relatched = handle._fold.projection.pending
    assert relatched is not None
    assert relatched.request_id == clean.request_id
    assert relatched.bridge_latched is True
    assert relatched.detail == f"{bridge_card_notice(latch)}\n{clean.detail}"
    # The re-publish never answers the parked gate.
    assert not parked.done()
    await handle.approval_answer(clean.request_id, False, False)
    assert await parked is False
    await handle.dispose()


class _AckRecordingHandle(FakeHandle):
    def __init__(self) -> None:
        super().__init__()
        self.acks: list[Any] = []

    async def run_slash_authoritative(
        self, command, args, images, *, bridge_ack=False
    ):  # noqa: ANN001, ANN202
        self.acks.append(bridge_ack)
        return {"kind": "notice", "text": f"owner ran /{command}", "style": "info"}


@pytest.mark.asyncio
async def test_viewer_redraws_a_parked_card_when_the_owner_republishes_it_latched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    registrant = RuntimeServer(FakeHandle(), kind="tui")
    registrant.start()
    viewer = None
    try:
        record = await _wait_record(tmp_path)
        viewer = await AttachedSession.connect(
            record, "s1", config_dir=tmp_path, takeover_factory=_never_take_over
        )
        latch = viewer.bridge_taint_latch
        assert isinstance(latch, BridgeTaintLatch) and latch.mirror is True
        fired: list[bool] = []
        latch.on_latched.append(lambda: fired.append(True))
        clean = PendingRequest(request_id="r1", kind="approval", title="write", detail="d")
        viewer._apply_pending_gate(clean)
        assert viewer.bridge_latched is False and fired == []

        # Same request id, re-published latched: the identity check would drop
        # it, so the mirror must fire anyway -- that is what redraws the card.
        notice = bridge_card_notice()
        viewer._apply_pending_gate(
            PendingRequest(
                request_id="r1",
                kind="approval",
                title="write",
                detail=f"{notice}\nd",
                bridge_latched=True,
            )
        )
        assert fired == [True]
        assert viewer.bridge_latched is True
    finally:
        if viewer is not None:
            await viewer.dispose()
        registrant.close()


@pytest.mark.asyncio
async def test_a_non_bool_bridge_ack_is_not_sent_as_true_from_the_viewer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S-R7-1: ``attached.py``'s forwarding seam tests identity, like the owner."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    owner = _AckRecordingHandle()
    registrant = RuntimeServer(owner, kind="tui")
    registrant.start()
    viewer = None
    try:
        record = await _wait_record(tmp_path)
        viewer = await AttachedSession.connect(
            record, "s1", config_dir=tmp_path, takeover_factory=_never_take_over
        )
        # A non-loosening command, so the attach seam's operator-authority gate
        # (which refuses a loosening from this socket) does not stand in front
        # of the forwarding being measured; the ack rides every slash alike.
        await viewer.route_shared_slash("goal", "x", bridge_ack="false")  # type: ignore[arg-type]
        await viewer.route_shared_slash("goal", "x", bridge_ack=True)
        assert owner.acks == [False, True]
    finally:
        if viewer is not None:
            await viewer.dispose()
        registrant.close()


@pytest.mark.asyncio
async def test_a_non_bool_bridge_ack_is_not_sent_as_true_from_the_attach_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S-R7-1: ``attach_client.slash_result`` omits a non-``True`` ack."""
    from local_operator.mobile.attach_client import AttachClient

    seen: list[dict[str, Any]] = []

    async def payload(self, op, **fields):  # noqa: ANN001, ANN202
        seen.append(fields)
        return {"kind": "notice", "text": "applied"}

    monkeypatch.setattr(AttachClient, "_request_payload", payload)
    client = AttachClient(lambda p: None, lambda reason: None)

    await client.slash_result("approvals", "auto", bridge_ack="false")  # type: ignore[arg-type]
    assert "bridge_ack" not in seen[-1]
    await client.slash_result("approvals", "auto", bridge_ack=True)
    assert seen[-1]["bridge_ack"] is True
