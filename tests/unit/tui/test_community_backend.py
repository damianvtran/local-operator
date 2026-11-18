"""``LiveCommunityBackend`` — the page's production adapter — over the REAL client.

The view tests drive the widget with ``FakeBackend``; this file closes the gap
underneath it: the adapter's calls go through ``CommunityClient`` and the fake
hub's HTTP transport, so a wrong path, body or outcome mapping fails here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from local_operator.tui.community_backend import LiveCommunityBackend
from tests.unit.community.fakehub import COMMUNITY, JOINED, FakeHub, make_client, outcome

SUBMIT = "/community/v1/communities/local-operator/conversations/conv_general/submissions"


def _backend(tmp_path: Path, *, submit: tuple[int, Any] | None = None) -> tuple[Any, FakeHub]:
    hub = FakeHub()

    def override(sent: Any) -> httpx.Response | None:
        if sent.method == "GET" and sent.path == "/community/v1/communities":
            return httpx.Response(200, json={"joined": [COMMUNITY], "featured": []})
        if sent.path.endswith("/bootstrap"):
            return httpx.Response(200, json=JOINED)
        if sent.path == SUBMIT and submit is not None:
            return httpx.Response(submit[0], json=submit[1])
        return None

    hub.override = override
    return LiveCommunityBackend(tmp_path, client_factory=lambda _cfg: make_client(hub)), hub


@pytest.mark.asyncio
async def test_home_history_and_submit_through_the_client(tmp_path: Path) -> None:
    backend, hub = _backend(tmp_path)
    home = await backend.home()
    assert home is not None and home.joined and home.accepted_version == "2026-11-01"
    assert [c.name for c in home.channels] == ["general"]
    rows = await backend.history("local-operator", "conv_general")
    assert rows[0].text == "hello"
    result = await backend.submit("local-operator", "conv_general", "hi", "k" * 32, "2026-11-01")
    assert result.kind == "posted"
    (sent,) = [s for s in hub.sent if s.path == SUBMIT]
    # Submissions are multipart (the contract's SubmissionForm).
    assert b'name="text"' in sent.body and b"\r\n\r\nhi\r\n" in sent.body


@pytest.mark.asyncio
async def test_refusal_and_too_late_cancel_map_to_sender_only_outcomes(tmp_path: Path) -> None:
    refused = {"status": "refused", "category": "private_information", "reference": "MOD-AAAA-BBBB"}
    backend, _ = _backend(tmp_path, submit=(422, refused))
    result = await backend.submit("local-operator", "conv_general", "x", "k" * 32, None)
    assert (result.kind, result.category, result.reference) == (
        "refused",
        "private_information",
        "MOD-AAAA-BBBB",
    )
    backend, hub = _backend(tmp_path)
    base = hub.override

    def published_cancel(sent: Any) -> httpx.Response | None:
        if sent.path.endswith(":cancel"):
            return httpx.Response(200, json=outcome("published"))
        return base(sent)

    hub.override = published_cancel
    assert (await backend.cancel("local-operator", "k" * 32)).kind == "too_late"


def test_last_selection_round_trips_and_ignores_garbage(tmp_path: Path) -> None:
    backend, _ = _backend(tmp_path)
    assert backend.last_selection() is None
    backend.remember_selection("t_tran")
    assert backend.last_selection() == "t_tran"
    (tmp_path / "community" / "org_selection.json").write_text("{not json")
    assert backend.last_selection() is None
