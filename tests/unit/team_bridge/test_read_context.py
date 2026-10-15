"""The READ-ONLY read context: the two standings, the grammars, and no writes.

WHY THIS FILE EXISTS BESIDE THE ROUTE CELLS. The route tests drive the read
through the ASGI app, which is what a caller does; these pin the decisions the
route cannot show — WHICH standing the read resolved (two DISTINCT context
types, so a surface can never present retained history as a live inbox), and the
ID/cursor grammars that keep a caller-supplied value out of an upstream path.

The store here is the REAL one (``commit_enrollment`` is the seam H4 commits
through): the two standings differ by the DOCUMENT on disk, not by an argument,
and a fixture that faked either would test nothing about the difference.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from local_operator.team_bridge.read_context import (
    CURSOR_PATTERN,
    MESSAGE_ID_PATTERN,
    HistoryReadContext,
    ReadContext,
    ReadInbox,
    ReadRefused,
    ReceiveReadContext,
    inbox_view,
    resolve_read_context,
)
from local_operator.team_bridge.store import OwnerReceipt, OwnerSnapshot, Store
from tests.unit.server.test_desktop_team_bridge import _access, _owner


def _enroll(root: Path, *, enabled: bool) -> OwnerSnapshot:
    owner = _owner()
    Store(root).commit_enrollment(owner, OwnerReceipt("bridge_consent_v1", "ref-1"))
    (root / "config.yml").write_text(
        yaml.safe_dump({"values": {"team_bridge": {"enabled": enabled}}})
    )
    return owner


def _patch_client(monkeypatch: pytest.MonkeyPatch) -> None:
    from local_operator.team_bridge import client as tb_client
    from tests.unit.server.test_desktop_team_bridge import FakeClient

    FakeClient.behavior = {}
    FakeClient.instances = []
    FakeClient.behavior["access"] = _access()
    monkeypatch.setattr(tb_client, "TeamBridgeClient", FakeClient)


@pytest.mark.asyncio
async def test_an_enabled_machine_resolves_receive_standing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Enabled + consent + own credential row = the receipt/receive context."""

    _patch_client(monkeypatch)
    owner = _enroll(tmp_path, enabled=True)

    context = await resolve_read_context(tmp_path, expected_account_id="acc_1")

    assert isinstance(context, ReceiveReadContext)
    assert context.mode == "receive"
    assert context.read_only is False
    assert context.owner == owner
    assert not hasattr(context, "reason")


@pytest.mark.asyncio
async def test_a_paused_machine_resolves_retained_history_standing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The SAME namespace, a weaker standing: reads only, and it SAYS so.

    The distinction is a type, not a flag a caller may forget to read: a surface
    that receives a ``HistoryReadContext`` cannot render it as a live inbox, and
    the view names the reason so the honest sentence is available to it.
    """

    _patch_client(monkeypatch)
    _enroll(tmp_path, enabled=False)

    context = await resolve_read_context(tmp_path, expected_account_id="acc_1")

    assert isinstance(context, HistoryReadContext)
    assert isinstance(context, ReadContext)
    assert context.mode == "history"
    assert context.read_only is True
    assert context.reason == "bridge_off"
    view = inbox_view(ReadInbox(context=context))
    assert view["mode"] == "history"
    assert view["read_only"] is True
    assert view["history_reason"] == "bridge_off"
    assert view["body_included"] is False


@pytest.mark.asyncio
async def test_a_declared_account_this_machine_is_not_signed_into_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The captured-account guard is the RESOLVER's, before any wire call."""

    _patch_client(monkeypatch)
    _enroll(tmp_path, enabled=True)
    from tests.unit.server.test_desktop_team_bridge import FakeClient

    FakeClient.behavior["access"] = _access("acc_other")

    with pytest.raises(ReadRefused) as refusal:
        await resolve_read_context(tmp_path, expected_account_id="acc_1")
    assert refusal.value.code == "account_mismatch"
    assert FakeClient.behavior.get("last_inbox_kwargs") is None


def test_the_id_and_cursor_grammars_reject_injection_shapes() -> None:
    """The probe's shapes: ``..``, ``?``, ``/`` and a fragment never match.

    These values are interpolated into an upstream path or query by the
    transport, so this is the gate that keeps a caller-supplied id from
    rewriting the request — before any path is built.
    """

    accepted: dict[str, list[str]] = {
        MESSAGE_ID_PATTERN: ["bmsg_1", "bmsg_AbC-9_0"],
        CURSOR_PATTERN: ["MjU=", "b64+/_-="],
    }
    refused: dict[str, list[str]] = {
        MESSAGE_ID_PATTERN: ["bmsg_1/../x", "bmsg_1?y=1", "../../etc", "bmsg_1#f", "msg_1"],
        CURSOR_PATTERN: ["a b", "a?b=1", "a#b", "../x"],
    }
    for pattern, values in accepted.items():
        for value in values:
            assert re.match(pattern, value), (pattern, value)
    for pattern, values in refused.items():
        for value in values:
            assert re.match(pattern, value) is None, (pattern, value)


def test_the_view_carries_no_field_a_body_could_ride() -> None:
    """The view's own contract: metadata flags, stated, and no content key."""

    context = HistoryReadContext(owner=_owner(), access=_access(), reason="bridge_off")
    view: Any = inbox_view(ReadInbox(context=context))
    assert view["screening_detail_included"] is False
    assert view["local_events"] is False
    assert "items" in view and view["items"] == []
