"""The desktop ``/asides`` route: the streamed answer, and the tool-call refusal.

Three contracts are wire-visible on this route and all three are pinned here:

* the POST's response text stays AUTHORITATIVE, and the turns the route stores
  (and returns from ``GET /asides/{aside_id}``) are the RAW ones the user typed
  — the aside instruction is applied by the runtime seam, never by the route;
* the same answer is streamed to the panel as live-only ``aside_delta`` frames
  on the session's events stream while the POST is still in flight;
* a provider answer that is a bare tool call is retried once (inside the shared
  primitive) and a second one is the typed ``409 aside_unanswered`` rather than
  a 500 — the refusal the app can render.

The bridge is real (``DesktopSessionBridge``) so ``publish`` and its
``replay=False`` path are the production ones; only the session ``remote`` and
the pool are doubles, because the runtime is the subject of the tests next door.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace
from typing import Any, cast

import pytest
from httpx import ASGITransport, AsyncClient

from local_operator.server.utils.desktop_sessions import DesktopSessionBridge
from local_operator.session.errors import AsideUnanswered

SESSION_ID = "0123456789ab"
REQUEST_ID = "01234567-89ab-cdef-0123-456789abcdef"
SECOND_ID = "fedcba98-7654-3210-fedc-ba9876543210"
TOKEN = "aside-route-token"


@pytest.fixture
def restore_app_state() -> Iterator[None]:
    """Give the process the ``app`` state it had before this test.

    ``app`` is a module-level singleton and these tests write
    ``desktop_sessions``/``desktop_asides`` onto it; the same fixture (and the
    same reasoning) as ``test_serve_lifecycle`` and ``test_serve_retire``, which
    a hand-written list of keys would let go stale as the lifespan grows.
    """
    from local_operator.server.app import app

    saved = {key: app.state[key] for key in app.state}
    state = app.state
    try:
        yield
    finally:
        for key in list(state):
            del state[key]
        for key, value in saved.items():
            state[key] = value


class _FakeRemote:
    """The bridge's ``remote``: records the turns and streams what it is told."""

    def __init__(
        self,
        *,
        chunks: tuple[str, ...] = (),
        answer: str = "answer.",
        error: Exception | None = None,
    ) -> None:
        self.chunks = chunks
        self.answer = answer
        self.error = error
        self.turns: list[list[Any]] = []
        self.bound = 0
        self.finished = 0.0

    async def bind_runtime(self) -> None:
        self.bound += 1

    async def complete_aside(self, turns: list[Any], *, on_delta: Any = None, **_kw: Any) -> str:
        self.turns.append(list(turns))
        for chunk in self.chunks:
            if on_delta is not None:
                on_delta(chunk)
        self.finished = time.monotonic()
        if self.error is not None:
            raise self.error
        return self.answer


class _FakePool:
    """Hands the route one prepared bridge, standing in for the desktop pool."""

    def __init__(self, bridge: DesktopSessionBridge) -> None:
        self._bridge = bridge

    @contextlib.asynccontextmanager
    async def session(self, session_id: str, *, read: bool = False) -> AsyncIterator[Any]:
        yield self._bridge


def _install(monkeypatch: pytest.MonkeyPatch, tmp_path, remote: _FakeRemote):
    """Wire the shared app to a real bridge over ``remote``; return both."""
    from local_operator.server.app import app

    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    app.state.config_manager = SimpleNamespace(config_dir=tmp_path)
    bridge = DesktopSessionBridge(tmp_path, SESSION_ID, str(tmp_path))
    # ``cast`` because the route only ever uses this through the AttachedSession
    # surface (``bind_runtime`` + ``complete_aside``); a real session is the
    # subject of the runtime tests, not of these.
    bridge.remote = cast(Any, remote)
    app.state.desktop_sessions = _FakePool(bridge)
    app.state.desktop_asides = {}
    return app, bridge


def _drained(bridge: DesktopSessionBridge, sub: Any) -> list[dict[str, Any]]:
    frames = []
    while not sub.queue.empty():
        item = sub.queue.get_nowait()
        if item is not None:
            frames.append(item[0])
    return frames


@pytest.mark.asyncio
async def test_the_aside_streams_live_only_deltas_and_stores_the_raw_question(
    tmp_path, monkeypatch, restore_app_state: None
) -> None:
    """The stream, the frozen response shape, and the raw turns — one pass.

    The deltas are published BEFORE the response is read (the route publishes
    them from inside the request), they carry the frozen payload, and they are
    NOT retained for replay: a late subscriber must never receive an earlier
    chunk, and no chunk ever joins the transcript.
    """
    remote = _FakeRemote(chunks=("part one. ", "part two."), answer="part one. part two.")
    app, bridge = _install(monkeypatch, tmp_path, remote)
    sub = bridge.subscribe()

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        response = await client.post(
            f"/v1/desktop/sessions/{SESSION_ID}/asides",
            json={"request_id": REQUEST_ID, "text": "why this model?"},
        )

    assert response.status_code == 200, response.text
    body = response.json()["result"]["data"]
    assert body == {"aside_id": REQUEST_ID, "text": "part one. part two.", "off_record": True}

    frames = _drained(bridge, sub)
    assert [(f["type"], f["payload"]) for f in frames] == [
        ("aside_delta", {"aside_id": REQUEST_ID, "delta": "part one. "}),
        ("aside_delta", {"aside_id": REQUEST_ID, "delta": "part two."}),
    ]
    assert [f["seq"] for f in frames] == sorted(f["seq"] for f in frames)
    assert all(f["session_id"] == SESSION_ID and f["epoch"] == bridge.epoch for f in frames)
    # LIVE ONLY: the deltas advanced the sequence but were never retained, so a
    # reconnect cannot replay a chunk of an answer that is already settled.
    assert list(bridge.replay) == []

    # The route hands the runtime the RAW question: the instruction is the
    # runtime seam's job (see ``session/runtime/serving.py``), and a stored or
    # rendered ``<aside>`` block would leak scaffolding into the user's panel.
    (sent,) = remote.turns
    assert [m.text for m in sent] == ["why this model?"]
    assert sent[0].role == "user"

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        recovered = await client.get(f"/v1/desktop/sessions/{SESSION_ID}/asides/{REQUEST_ID}")

    turns = recovered.json()["result"]["data"]["turns"]
    assert [t["content"][0]["text"] for t in turns] == ["why this model?", "part one. part two."]
    assert "<aside>" not in recovered.text


@pytest.mark.asyncio
async def test_a_continuation_keeps_only_raw_turns_and_streams_again(
    tmp_path, monkeypatch, restore_app_state: None
) -> None:
    """A second question re-uses the aside's prefix, still unwrapped.

    The prefix is what a continuation OWNS (so two panels cannot adopt it), and
    it is also what the model sees on the second request — with only the new
    question added, and no instruction baked into either turn.
    """
    remote = _FakeRemote(chunks=("first.",), answer="first.")
    app, bridge = _install(monkeypatch, tmp_path, remote)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        assert (
            await client.post(
                f"/v1/desktop/sessions/{SESSION_ID}/asides",
                json={"request_id": REQUEST_ID, "text": "first question?"},
            )
        ).status_code == 200
        remote.answer, remote.chunks = "second.", ()
        again = await client.post(
            f"/v1/desktop/sessions/{SESSION_ID}/asides",
            json={
                "request_id": SECOND_ID,
                "text": "and now?",
                "aside_id": REQUEST_ID,
            },
        )

    assert again.status_code == 200, again.text
    assert again.json()["result"]["data"] == {
        "aside_id": SECOND_ID,
        "text": "second.",
        "off_record": True,
    }
    first, second = remote.turns
    assert [m.text for m in first] == ["first question?"]
    assert [m.text for m in second] == ["first question?", "first.", "and now?"]


@pytest.mark.asyncio
async def test_a_model_that_will_not_answer_in_text_is_a_named_409(
    tmp_path, monkeypatch, restore_app_state: None
) -> None:
    """``AsideUnanswered`` reaches the client as a typed refusal, not a 500.

    The route ladder maps it through the ``ValueError`` arm, so the app gets a
    code it can key on and the sentence the error object built locally (never
    the provider's own words). The aside entry is un-claimed exactly as it is on
    any other failure, so the panel can be closed rather than left running.
    """
    remote = _FakeRemote(error=AsideUnanswered())
    app, _bridge = _install(monkeypatch, tmp_path, remote)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        response = await client.post(
            f"/v1/desktop/sessions/{SESSION_ID}/asides",
            json={"request_id": REQUEST_ID, "text": "why?"},
        )

    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "aside_unanswered"
    assert detail["message"] == str(AsideUnanswered())
    entry = app.state.desktop_asides[REQUEST_ID]
    assert entry.running is False, "a failed aside must not stay claimed as running"
