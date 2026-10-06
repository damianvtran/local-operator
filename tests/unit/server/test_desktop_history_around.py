"""The anchored history read over the REAL route (design §D4/§D9).

The reader's own contract is pinned in ``tests/unit/session/test_transcript.py``
(the forward oracle) and the cache's in ``test_page_cache.py``; this file is the
half a client actually meets: the query parameters, the statuses for a request
that names no window or an out-of-range bound, the reconciliation answer for an
absent journal, and the peer-session branch that v1 answers unsupported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.resume import SessionRow
from local_operator.server.routes import desktop_sessions
from local_operator.server.utils.desktop_sessions import (
    DesktopSessionBridge,
    DesktopSessions,
)
from local_operator.session.transcript import (
    ENTRY_MESSAGE,
    TRANSCRIPT_FILENAME,
    TranscriptEntry,
)

TOKEN = "synthetic-desktop-token"


@pytest.fixture(autouse=True)
def _desktop_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every row runs against a synthetic HOME, not the operator's."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))


def _seed_journal(directory: Path, rows: int) -> list[str]:
    """``rows`` message rows in journal order; returns their ids, oldest first."""
    directory.mkdir(parents=True, exist_ok=True)
    ids: list[str] = []
    lines: list[str] = []
    for index in range(rows):
        entry = TranscriptEntry(
            f"{index:012x}",
            float(index),
            ENTRY_MESSAGE,
            {"role": "user", "content": f"message {index}"},
        )
        ids.append(entry.id)
        lines.append(entry.to_json())
    (directory / TRANSCRIPT_FILENAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ids


class _Harness:
    """One pool, one real router app, one session with a seeded journal."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.app = FastAPI()
        self.app.include_router(desktop_sessions.router)
        self.pool = DesktopSessions(root)
        self.app.state.desktop_sessions = self.pool
        # The route ladder resolves some arms through app state; an isolated
        # manager keeps those off the operator's own config.
        self.app.state.config_manager = ConfigManager(config_dir=root)
        self.session_id = ""
        self.ids: list[str] = []
        self.client: AsyncClient | None = None

    async def __aenter__(self) -> "_Harness":
        self.session_id = await self.pool.create(str(self.root))
        self.ids = _seed_journal(self.root / "sessions" / self.session_id, rows=8)
        self.client = AsyncClient(
            transport=ASGITransport(app=self.app),
            base_url="http://localhost",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self.client is not None:
            await self.client.aclose()


@pytest.mark.asyncio
async def test_history_around_returns_the_window_and_both_edges(tmp_path: Path) -> None:
    """The anchored read, end to end: rows oldest→newest, flags on the wire."""
    async with _Harness(tmp_path) as harness:
        assert harness.client is not None
        base = f"/v1/desktop/sessions/{harness.session_id}"

        response = await harness.client.get(
            f"{base}/history",
            params={"around_id": harness.ids[3], "before": 1, "after": 2},
        )
        assert response.status_code == 200, response.text
        page = response.json()["result"]
        assert [row["id"] for row in page["entries"]] == harness.ids[2:6]
        assert page["has_more"] is True
        assert page["has_newer"] is True
        assert page["cursor_missing"] is False

        at_tail = await harness.client.get(
            f"{base}/history",
            params={"around_id": harness.ids[-1], "before": 1, "after": 1},
        )
        assert at_tail.status_code == 200, at_tail.text
        tail_page = at_tail.json()["result"]
        assert [row["id"] for row in tail_page["entries"]] == harness.ids[-2:]
        assert tail_page["has_newer"] is False
        assert tail_page["has_more"] is True


@pytest.mark.asyncio
async def test_history_around_defaults_centre_the_window_on_the_id(tmp_path: Path) -> None:
    """An around request without counts still answers a ``limit``-sized page."""
    async with _Harness(tmp_path) as harness:
        assert harness.client is not None
        response = await harness.client.get(
            f"/v1/desktop/sessions/{harness.session_id}/history",
            params={"around_id": harness.ids[4], "limit": 4},
        )
        assert response.status_code == 200, response.text
        assert [row["id"] for row in response.json()["result"]["entries"]] == harness.ids[2:6]


@pytest.mark.asyncio
async def test_history_around_unknown_session_is_the_shared_404(tmp_path: Path) -> None:
    """THE 404 BELONGS TO THE LOOKUP: an id nobody holds is unknown, not empty."""
    async with _Harness(tmp_path) as harness:
        assert harness.client is not None
        response = await harness.client.get(
            "/v1/desktop/sessions/ffffffffffff/history", params={"around_id": "deadbeef"}
        )
        assert response.status_code == 404, response.text


@pytest.mark.asyncio
async def test_history_around_rejects_malformed_requests(tmp_path: Path) -> None:
    """A request that describes no window fails the same way at every door.

    The combinations go through the reader's own validator (409, the reader's
    sentence — the same refusal the sidecar tests pin), while the numeric
    bounds are the wire's (422 from the query declaration).
    """
    async with _Harness(tmp_path) as harness:
        assert harness.client is not None
        base = f"/v1/desktop/sessions/{harness.session_id}"

        both = await harness.client.get(
            f"{base}/history",
            params={"around_id": harness.ids[1], "before_id": harness.ids[0]},
        )
        assert both.status_code == 409, both.text
        assert "choose at most one" in both.json()["detail"]

        counts = await harness.client.get(f"{base}/history", params={"before": 2})
        assert counts.status_code == 409, counts.text
        assert "only meaningful with around_id" in counts.json()["detail"]

        negative = await harness.client.get(
            f"{base}/history", params={"around_id": harness.ids[1], "before": -1}
        )
        assert negative.status_code == 422, negative.text

        over = await harness.client.get(
            f"{base}/history", params={"around_id": harness.ids[1], "after": 501}
        )
        assert over.status_code == 422, over.text


@pytest.mark.asyncio
async def test_history_around_on_an_absent_journal_reconciles(tmp_path: Path) -> None:
    """A session with no journal answers the empty reconciled page, not a 404."""
    async with _Harness(tmp_path) as harness:
        assert harness.client is not None
        other = await harness.pool.create(str(tmp_path))
        response = await harness.client.get(
            f"/v1/desktop/sessions/{other}/history", params={"around_id": "deadbeef"}
        )
        assert response.status_code == 200, response.text
        page = response.json()["result"]
        assert page["entries"] == []
        assert page["cursor_missing"] is True
        assert page["has_newer"] is None


@pytest.mark.asyncio
async def test_history_around_on_a_peer_session_is_an_unsupported_empty_page(
    tmp_path: Path,
) -> None:
    """V1 cannot anchor a peer's wire window; the answer is the empty reconciled page.

    Pinned at the facade rather than the route because a real peer bridge needs
    the mesh: the branch under test is the facade's ``remote_row`` switch. The
    plain remote read beside it is the one that CHANGED with D5-core (design
    ``docs/design/mesh-cold-read-stored-history.md``): a cold peer page is now
    read from the OWNER's stored journal, and this bridge has no relay and no
    owner at all, so the page is UNSERVABLE -- and an unservable page says so
    (``cursor_missing: True``) rather than answering the empty triple a
    conversation with no rows produces. That distinction is the point of the
    change: an empty page on a peer now means the owner has no rows. What a
    served cold page looks like is pinned over two real relays in
    ``tests/unit/network/test_remote_viewer.py``.
    """
    row = SessionRow(
        id="ffffffffffff",
        mtime=1789400000.0,
        name="peer chat",
        live_state="idle",
        locality="remote",
        owner_device="peer-device",
        owner_device_name="build-box",
        reachable=True,
        unreachable_reason="",
    )
    bridge = DesktopSessionBridge(tmp_path, row.id, "", remote_row=row)

    anchored = await bridge.history(around_id="deadbeef", before=2, after=2)
    assert anchored == {
        "entries": [],
        "has_more": False,
        "cursor_missing": True,
        "has_newer": None,
    }

    plain = await bridge.history(limit=10)
    assert plain == {
        "entries": [],
        "has_more": False,
        # CHANGED BY DESIGN (D5-core): the reader could not fetch the owner's
        # stored page (no relay on this device at all), so the page is marked
        # untrustworthy rather than published as "this conversation is empty".
        "cursor_missing": True,
        "has_newer": None,
    }

    # Peer parity (remediation round 1, R1-1): a malformed request must refuse
    # the SAME way here as on a local session — the remote branch is reached
    # only by requests that pass the shared validator first.
    with pytest.raises(ValueError, match="only meaningful with around_id"):
        await bridge.history(before=2)
    with pytest.raises(ValueError, match="choose at most one"):
        await bridge.history(around_id="deadbeef", before_id="deadbeef")


@pytest.mark.asyncio
async def test_history_before_id_keeps_has_newer_null(tmp_path: Path) -> None:
    """The unchanged reads carry the null on the wire: an unasked question
    is not answered False (remediation round 1, R1-3).

    The anchored read is the only one whose contract bounds its newer side, so
    a paging read must serialize ``has_newer: null`` — a client that branches
    on the key must see "not asked", never a claim.
    """
    async with _Harness(tmp_path) as harness:
        assert harness.client is not None
        response = await harness.client.get(
            f"/v1/desktop/sessions/{harness.session_id}/history",
            params={"before_id": harness.ids[4], "limit": 2},
        )
        assert response.status_code == 200, response.text
        page = response.json()["result"]
        assert [row["id"] for row in page["entries"]] == harness.ids[2:4]
        assert page["has_newer"] is None
        assert page["cursor_missing"] is False
