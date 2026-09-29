"""``GET /v1/desktop/sessions/{id}/find``: the wire shape, ranking, refusals.

Drives the REAL router app over an isolated config root — a session minted by
the pool, its journal written by the real ``Transcript`` writer, and the find
route reading the index derived from that journal. Nothing here touches the
operator's store.

The reference implementation of the pipeline is
``tests/unit/session/test_transcript_find.py``; this file pins the route: the
D9 answer shape, the query/limit bounds, the unknown-session refusal, the peer
degradation, and that a draft-era request (a session with nothing written yet)
answers the empty ready rather than a 404.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, AsyncIterator, cast

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.harness.message_types import PEER_MESSAGE_MESSAGE_TYPE
from local_operator.harness.types import CustomMessage, Message
from local_operator.server.routes import desktop_sessions
from local_operator.server.utils.desktop_sessions import DesktopSessions
from local_operator.session import transcript_find, transcript_index
from local_operator.session.transcript import Transcript

TOKEN = "synthetic-desktop-token"


@pytest.fixture(autouse=True)
def _desktop_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir(parents=True, exist_ok=True)


@pytest.fixture(autouse=True)
def _clean_module_state():
    """The index and find modules keep process-wide state; start clean."""
    transcript_index._reset_for_tests()
    transcript_find._reset_for_tests()
    yield
    transcript_index._reset_for_tests()
    transcript_find._reset_for_tests()


@pytest.fixture()
def hide_cross_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """``display.hide_cross_session``, written through the real reader path.

    A dedicated config dir, the same shape the mobile fold suite's fixture
    uses: the write goes through ``settings_io``, so the flat-dotted key the
    reader actually looks up is the one exercised, and no test inherits a
    flip another made.
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    from local_operator import settings_io
    from local_operator.tui.settings import settings_reload

    settings_reload()

    def set_hidden(on: bool) -> None:
        settings_io.write_setting(
            ConfigManager(config_dir), settings_io.BY_KEY["display.hide_cross_session"], on
        )

    yield set_hidden
    settings_reload()


async def _seed(root: Path, session_id: str, messages: list[Message | CustomMessage]) -> None:
    """Write ``messages`` through the real journal writer, as a live session does."""
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    transcript = Transcript(directory)
    for message in messages:
        await transcript.append_message(message)
    transcript.flush()


class _Harness:
    """One isolated config root, one real router app, one minted session."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.app = FastAPI()
        self.app.include_router(desktop_sessions.router)
        self.pool = DesktopSessions(root)
        self.app.state.desktop_sessions = self.pool
        self.app.state.config_manager = ConfigManager(config_dir=root)
        self.inputs = root / "workspace"
        self.inputs.mkdir(parents=True, exist_ok=True)
        self.client: AsyncClient | None = None

    async def __aenter__(self) -> "_Harness":
        self.client = AsyncClient(
            transport=ASGITransport(app=self.app),
            base_url="http://localhost",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self.client is not None:
            await self.client.aclose()

    async def create_session(self) -> str:
        return await self.pool.create(str(self.inputs))


@pytest_asyncio.fixture
async def harness(tmp_path: Path) -> AsyncIterator[_Harness]:
    async with _Harness(tmp_path / "root") as live:
        yield live


async def _find(harness: _Harness, session_id: str, **params: Any) -> Any:
    assert harness.client is not None
    return await harness.client.get(f"/v1/desktop/sessions/{session_id}/find", params=params)


@pytest.mark.asyncio
async def test_find_returns_ranked_hits_oldest_first(harness: _Harness) -> None:
    session_id = await harness.create_session()
    await _seed(
        harness.root,
        session_id,
        [
            Message.user("deploy the staging target"),
            Message.assistant("I will deploy it now"),
            Message.user("unrelated chatter"),
        ],
    )
    response = await _find(harness, session_id, q="deploy")
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["query"] == "deploy"
    assert result["state"] == "ready"
    assert result["partial"] is False
    assert result["truncated"] is False
    assert [hit["role"] for hit in result["hits"]] == ["user", "agent"]
    assert result["hits"][0]["snippet"] == "deploy the staging target"
    assert result["hits"][0]["ranges"] == [[0, 6]]
    assert result["hits"][0]["tier"] == "exact"


@pytest.mark.asyncio
async def test_find_soft_hit_carries_empty_ranges(harness: _Harness) -> None:
    session_id = await harness.create_session()
    await _seed(harness.root, session_id, [Message.user("the classifier needs work")])
    response = await _find(harness, session_id, q="classifer")
    assert response.status_code == 200, response.text
    hits = response.json()["result"]["hits"]
    assert [(hit["tier"], hit["ranges"]) for hit in hits] == [("soft", [])]


@pytest.mark.asyncio
async def test_find_reports_truncation_at_the_limit(harness: _Harness) -> None:
    session_id = await harness.create_session()
    await _seed(
        harness.root,
        session_id,
        [Message.user(f"needle number {index}") for index in range(3)],
    )
    response = await _find(harness, session_id, q="needle", limit=2)
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert len(result["hits"]) == 2
    assert result["truncated"] is True


@pytest.mark.asyncio
async def test_find_with_nothing_written_is_ready_empty(harness: _Harness) -> None:
    """A session the pane just created has no journal; the empty answer is correct.

    This is also the draft-era shape the overlay can reach: there are no
    messages to search, which is why it must render "No matches" rather than an
    error.
    """
    session_id = await harness.create_session()
    response = await _find(harness, session_id, q="deploy")
    assert response.status_code == 200, response.text
    assert response.json()["result"] == {
        "query": "deploy",
        "state": "ready",
        "partial": False,
        "hits": [],
        "truncated": False,
    }


@pytest.mark.asyncio
async def test_find_unknown_session_is_404(harness: _Harness) -> None:
    response = await _find(harness, "0123456789ab", q="deploy")
    assert response.status_code == 404, response.text


@pytest.mark.asyncio
async def test_find_peer_conversation_answers_unsupported(harness: _Harness) -> None:
    """The adapter's D4 branch: the journal is on the peer, not here.

    Driven at the bridge (the route's own resolution of a peer needs a mesh),
    which is where the branch lives.
    """
    session_id = await harness.create_session()
    async with harness.pool.session(session_id, read=True, allow_draft=True) as bridge:
        bridge.remote_row = cast(Any, object())
        try:
            result = await bridge.find("deploy", 10)
        finally:
            bridge.remote_row = None
    assert result == {
        "query": "deploy",
        "state": "unsupported",
        "partial": False,
        "hits": [],
        "truncated": False,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "params",
    [
        pytest.param({"q": "deploy", "limit": 0}, id="limit-below-range"),
        pytest.param({"q": "deploy", "limit": 201}, id="limit-above-range"),
        pytest.param({}, id="q-missing"),
        pytest.param({"q": ""}, id="q-empty"),
        pytest.param({"q": "x" * 257}, id="q-too-long"),
    ],
)
async def test_find_rejects_out_of_bounds_input(harness: _Harness, params: dict[str, Any]) -> None:
    session_id = await harness.create_session()
    response = await _find(harness, session_id, **params)
    assert response.status_code == 422, response.text


@pytest.mark.asyncio
async def test_find_bounds_are_the_declared_ones(harness: _Harness) -> None:
    """256 characters and a limit of 200 are ACCEPTED; 257 and 201 are not.

    The boundary itself, not just the far side of it — an off-by-one in the
    declarations is exactly what the parametrised refusals above would miss.
    """
    session_id = await harness.create_session()
    await _seed(harness.root, session_id, [Message.user("x" * 300 + " needle")])
    at_max = await _find(harness, session_id, q="n" + "e" * 255, limit=200)
    assert at_max.status_code == 200, at_max.text
    over = await _find(harness, session_id, q="n" + "e" * 256, limit=200)
    assert over.status_code == 422, over.text


@pytest.mark.asyncio
async def test_find_response_model_matches_the_d9_shape(harness: _Harness) -> None:
    """Every key D9 names is present, and nothing rides outside it."""
    session_id = await harness.create_session()
    await _seed(harness.root, session_id, [Message.user("deploy")])
    response = await _find(harness, session_id, q="deploy")
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert set(result) == {"query", "state", "partial", "hits", "truncated"}
    assert set(result["hits"][0]) == {"id", "role", "ts", "snippet", "ranges", "tier"}


@pytest.mark.asyncio
async def test_hidden_cross_session_hides_peer_rows_from_the_route(
    harness: _Harness, hide_cross_session: Any
) -> None:
    """``display.hide_cross_session`` on: a peer row is not a find result.

    The route contract (design §5.3): with the flag on no surface paints a
    peer row, and the overlay reveals a hit by jumping to its id — so a
    returned peer hit would be a content leak AND a jump to a row that is not
    there. OFF is the byte-identical default and keeps the row findable; the
    pair is the pin. Seeded through the REAL writer, peer-shaped the way the
    relay persists one (`details.body` — the shape the find suite's fixtures
    pin).
    """
    session_id = await harness.create_session()
    await _seed(
        harness.root,
        session_id,
        [
            Message.user("deploy the staging target", id="u1"),
            CustomMessage(
                id="p1",
                custom_type=PEER_MESSAGE_MESSAGE_TYPE,
                attribution="user",
                details={"body": "peer says the deploy is green", "sender": {"pid": 999}},
            ),
        ],
    )

    response = await _find(harness, session_id, q="deploy")
    assert response.status_code == 200, response.text
    assert [hit["id"] for hit in response.json()["result"]["hits"]] == ["u1", "p1"]

    hide_cross_session(True)
    result = (await _find(harness, session_id, q="deploy")).json()["result"]
    assert [hit["id"] for hit in result["hits"]] == ["u1"]
    # Peer-only text: nothing under any tier, and the envelope stays the D9
    # ready shape rather than an error.
    result = (await _find(harness, session_id, q="green")).json()["result"]
    assert result["hits"] == [] and result["state"] == "ready"

    hide_cross_session(False)
    result = (await _find(harness, session_id, q="green")).json()["result"]
    assert [hit["id"] for hit in result["hits"]] == ["p1"]
