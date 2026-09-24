"""The desktop ``/asides`` route: the streamed answer, and its refusals.

Four contracts are wire-visible on this route and all four are pinned here:

* the POST's response text stays AUTHORITATIVE, and the turns the route stores
  (and returns from ``GET /asides/{aside_id}``) are the RAW ones the user typed
  — the aside instruction is applied by the runtime seam, never by the route;
* the same answer is streamed to the ASKING VIEWER as live-only ``aside_delta``
  frames while the POST is still in flight, and to no other subscriber of the
  same session — an aside is off the record, so a fan-out here would be the
  leak the runtime seam already refuses one hop earlier;
* a provider answer that is a bare tool call is retried once (inside the shared
  primitive) and a second one is the typed ``409 aside_unanswered`` rather than
  a 500 — the refusal the app can render;
* an answer that settles with NO text is the typed ``409 aside_empty_answer``
  and leaves no entry, rather than a 200 storing an empty, adoptable turn.

A REFUSED ASIDE LEAVES NOTHING BEHIND, which is why the refusal tests read the
store rather than only the status: the entry a raise would otherwise leave is a
question with no answer, which no surface can continue or adopt.

The bridge is real (``DesktopSessionBridge``) so the targeted publish and its
accounting are the production ones; only the session ``remote`` and the pool are
doubles, because the runtime is the subject of the tests next door.
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
from local_operator.session.errors import AsideEmptyAnswer, AsideUnanswered

SESSION_ID = "0123456789ab"
REQUEST_ID = "01234567-89ab-cdef-0123-456789abcdef"
SECOND_ID = "fedcba98-7654-3210-fedc-ba9876543210"
THIRD_ID = "00112233-4455-6677-8899-aabbccddeeff"
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
    chunk, and no chunk ever joins the transcript. The asker names its own
    subscription — the ``open`` frame's ``subscription_id`` — because that is
    what addresses the frames to it.
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
            json={
                "request_id": REQUEST_ID,
                "text": "why this model?",
                "subscription_id": sub.id,
            },
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
async def test_another_viewer_of_the_session_receives_no_deltas(
    tmp_path, monkeypatch, restore_app_state: None
) -> None:
    """An aside streams to the subscription that ASKED, and to no other.

    The measured defect: a second ``/events`` stream on the same session got
    every frame of an aside it never asked for, which contradicts the whole
    point of an off-record exchange (and the runtime seam's own rule one hop
    earlier — ``_aside_delta_sink`` targets the asking connection). Publishing
    to one subscription is what makes "the requesting viewer" true on this hop.
    """
    remote = _FakeRemote(chunks=("part one. ", "part two."), answer="part one. part two.")
    app, bridge = _install(monkeypatch, tmp_path, remote)
    asker = bridge.subscribe()
    other = bridge.subscribe()

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        response = await client.post(
            f"/v1/desktop/sessions/{SESSION_ID}/asides",
            json={
                "request_id": REQUEST_ID,
                "text": "off the record?",
                "subscription_id": asker.id,
            },
        )

    assert response.status_code == 200, response.text
    assert [f["payload"]["delta"] for f in _drained(bridge, asker)] == [
        "part one. ",
        "part two.",
    ]
    assert _drained(bridge, other) == [], "a viewer that did not ask must receive nothing"


@pytest.mark.asyncio
async def test_an_ask_that_names_no_subscription_publishes_nothing(
    tmp_path, monkeypatch, restore_app_state: None
) -> None:
    """No subscription id means NO frames — not a broadcast to everyone.

    The fallback is the load-bearing half of the fix: a broadcast is exactly
    the leak being closed, and the POST's ``text`` already settles the answer
    for a caller that named no subscription (an older client, or the CLI).
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
            json={"request_id": REQUEST_ID, "text": "no subscription here"},
        )

    assert response.status_code == 200, response.text
    assert response.json()["result"]["data"]["text"] == "part one. part two."
    assert _drained(bridge, sub) == []
    # The sequence does not move for a request that published nothing, so a
    # client's cursor is not advanced over frames that were never sent.
    assert bridge.sequence == 0


@pytest.mark.asyncio
async def test_an_unknown_subscription_id_publishes_nothing_and_does_not_500(
    tmp_path, monkeypatch, restore_app_state: None
) -> None:
    """A subscription that closed mid-aside costs the stream, not the answer.

    ``publish_to_subscription`` reports the miss instead of raising, and the
    route ignores it: the aside's answer is what the request is for, and a
    renderer that reconnected between the POST and its first token has the
    settled text waiting in the response.
    """
    remote = _FakeRemote(chunks=("part one.",), answer="part one.")
    app, bridge = _install(monkeypatch, tmp_path, remote)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        response = await client.post(
            f"/v1/desktop/sessions/{SESSION_ID}/asides",
            json={
                "request_id": REQUEST_ID,
                "text": "gone?",
                "subscription_id": "f" * 32,
            },
        )

    assert response.status_code == 200, response.text
    assert response.json()["result"]["data"]["text"] == "part one."
    assert bridge.subscribers == {}


@pytest.mark.asyncio
async def test_a_malformed_subscription_id_is_refused_before_the_aside_runs(
    tmp_path, monkeypatch, restore_app_state: None
) -> None:
    """A malformed id is a 422 from the body, never a silent no-op.

    Typed for the reason the events route types it: an id this route cannot
    compare is an id whose frames would go nowhere, and answering 200 would
    report a stream that was never addressed to anyone.
    """
    remote = _FakeRemote(answer="answer.")
    app, _bridge = _install(monkeypatch, tmp_path, remote)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        response = await client.post(
            f"/v1/desktop/sessions/{SESSION_ID}/asides",
            json={"request_id": REQUEST_ID, "text": "hi", "subscription_id": "not-hex"},
        )

    assert response.status_code == 422, response.text
    assert remote.turns == []


@pytest.mark.asyncio
async def test_a_continuation_keeps_only_raw_turns_and_streams_again(
    tmp_path, monkeypatch, restore_app_state: None
) -> None:
    """A second question re-uses the aside's prefix, still unwrapped.

    The prefix is what a continuation OWNS (so two panels cannot adopt it), and
    it is also what the model sees on the second request — with only the new
    question added, and no instruction baked into either turn. The second answer
    streams to the viewer that asked for IT, exactly as the first one did.
    """
    remote = _FakeRemote(chunks=("first.",), answer="first.")
    app, bridge = _install(monkeypatch, tmp_path, remote)
    sub = bridge.subscribe()
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        assert (
            await client.post(
                f"/v1/desktop/sessions/{SESSION_ID}/asides",
                json={
                    "request_id": REQUEST_ID,
                    "text": "first question?",
                    "subscription_id": sub.id,
                },
            )
        ).status_code == 200
        remote.answer, remote.chunks = "second.", ("second.",)
        again = await client.post(
            f"/v1/desktop/sessions/{SESSION_ID}/asides",
            json={
                "request_id": SECOND_ID,
                "text": "and now?",
                "aside_id": REQUEST_ID,
                "subscription_id": sub.id,
            },
        )

    assert again.status_code == 200, again.text
    assert again.json()["result"]["data"] == {
        "aside_id": SECOND_ID,
        "text": "second.",
        "off_record": True,
    }
    assert [f["payload"] for f in _drained(bridge, sub)] == [
        {"aside_id": REQUEST_ID, "delta": "first."},
        {"aside_id": SECOND_ID, "delta": "second."},
    ]
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
    the provider's own words). The sentence is true of EVERY arm that raises it
    — QA reproduced a tool call once and then SILENCE on the corrected retry,
    which the previous copy ("a tool call … both times it was asked") described
    wrongly.
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
    # The sentence must not claim a tool call on BOTH attempts: the reachable
    # second arm is an empty answer, and only the user can see which one they got.
    assert "both times" not in detail["message"]
    assert "tool call" in detail["message"] and "ask again" in detail["message"]


@pytest.mark.asyncio
async def test_a_refused_aside_leaves_the_store_clean_and_the_retry_works(
    tmp_path, monkeypatch, restore_app_state: None
) -> None:
    """A refusal drops its entry, so "ask again" is a thing the user can DO.

    The route claims the entry before the provider is asked, so a raise used to
    leave a question with no answer: ``turns`` odd, which every reader of the
    store treats as neither completable nor adoptable, so the panel's own
    remedy (ask again) hit a dead end that did not say why — and the entry held
    one of the 64 panel slots for the store's full hour.
    """
    remote = _FakeRemote(error=AsideUnanswered())
    app, _bridge = _install(monkeypatch, tmp_path, remote)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        refused = await client.post(
            f"/v1/desktop/sessions/{SESSION_ID}/asides",
            json={"request_id": REQUEST_ID, "text": "why?"},
        )
        # The refused entry is GONE, not merely un-claimed: nothing can read it
        # back and it no longer occupies a slot.
        recovered = await client.get(f"/v1/desktop/sessions/{SESSION_ID}/asides/{REQUEST_ID}")
        # The retry the shipped panel sends (a failed FIRST question leaves it
        # with no aside_id, so it starts a fresh one) works and stores a clean,
        # complete exchange.
        remote.error = None
        remote.answer, remote.chunks = "a real answer.", ()
        retried = await client.post(
            f"/v1/desktop/sessions/{SESSION_ID}/asides",
            json={"request_id": SECOND_ID, "text": "why?"},
        )

    assert refused.status_code == 409, refused.text
    assert recovered.status_code == 404, recovered.text
    assert retried.status_code == 200, retried.text
    assert retried.json()["result"]["data"]["text"] == "a real answer."
    entries = app.state.desktop_asides
    assert list(entries) == [SECOND_ID], "the refusal must not leave a dead entry behind"
    assert len(entries[SECOND_ID].turns) % 2 == 0


@pytest.mark.asyncio
async def test_a_retry_naming_the_dropped_entry_id_is_refused(
    tmp_path, monkeypatch, restore_app_state: None
) -> None:
    """The dropped entry is gone for CONTINUATIONS too, not only for reads.

    "Ask again" is only actionable if the id the refusal released is really
    released. What works is a FRESH ask (the shipped panel holds no ``aside_id``
    after a failed first question) or a retry naming a LIVE panel's id; what is
    refused is a client carrying the failed ask's own id forward as ``aside_id``
    — the same 409 a closed, expired or foreign panel gets, because the store
    no longer holds that entry (``docs/DESKTOP_CONTROLS.md``).
    """
    remote = _FakeRemote(error=AsideUnanswered())
    app, _bridge = _install(monkeypatch, tmp_path, remote)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        refused = await client.post(
            f"/v1/desktop/sessions/{SESSION_ID}/asides",
            json={"request_id": REQUEST_ID, "text": "why?"},
        )
        # The provider stays broken on purpose: a refusal at the DOOR never
        # reaches it, so ``remote.turns`` below measures the whole request.
        stale = await client.post(
            f"/v1/desktop/sessions/{SESSION_ID}/asides",
            json={"request_id": SECOND_ID, "text": "why?", "aside_id": REQUEST_ID},
        )

    assert refused.status_code == 409, refused.text
    assert stale.status_code == 409, stale.text
    assert stale.json()["detail"] == "This aside is no longer available"
    assert [m.text for m in remote.turns[0]] == ["why?"]
    assert app.state.desktop_asides == {}, "the dropped entry must not come back"


@pytest.mark.asyncio
async def test_a_refused_continuation_leaves_the_panel_it_continued_usable(
    tmp_path, monkeypatch, restore_app_state: None
) -> None:
    """The prefix a refusal borrowed must come back exactly as it was found.

    A continuation claims the panel's prefix (``previous.adopted = True``) so two
    requests cannot promote the same exchange. When the ask then fails, the entry
    for the FAILED request is dropped — and the panel's own entry has to become
    continuable again, or the refusal would have broken the aside the user is
    still looking at, which is the one thing the refusal's copy ("ask again")
    cannot repair.
    """
    remote = _FakeRemote(chunks=("first.",), answer="first.")
    app, _bridge = _install(monkeypatch, tmp_path, remote)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        first = await client.post(
            f"/v1/desktop/sessions/{SESSION_ID}/asides",
            json={"request_id": REQUEST_ID, "text": "first question?"},
        )
        remote.error = AsideUnanswered()
        refused = await client.post(
            f"/v1/desktop/sessions/{SESSION_ID}/asides",
            json={"request_id": SECOND_ID, "text": "and now?", "aside_id": REQUEST_ID},
        )
        remote.error = None
        retried = await client.post(
            f"/v1/desktop/sessions/{SESSION_ID}/asides",
            json={"request_id": THIRD_ID, "text": "and now?", "aside_id": REQUEST_ID},
        )

    assert first.status_code == 200, first.text
    assert refused.status_code == 409, refused.text
    assert refused.json()["detail"]["code"] == "aside_unanswered"
    assert retried.status_code == 200, retried.text
    entries = app.state.desktop_asides
    assert sorted(entries) == [THIRD_ID, REQUEST_ID]
    # The retry re-used the prefix: the panel's exchange plus the new question.
    assert [m.text for m in entries[THIRD_ID].turns] == [
        "first question?",
        "first.",
        "and now?",
        "first.",
    ]
    assert len(entries[REQUEST_ID].turns) == 2


@pytest.mark.asyncio
async def test_a_settled_answer_with_no_text_is_a_named_409_and_stores_nothing(
    tmp_path, monkeypatch, restore_app_state: None
) -> None:
    """An empty settled answer is a refusal, not a finished exchange.

    A 200 with ``text: ""`` stored an EMPTY assistant turn marked complete and
    adoptable: the renderer branches on truthiness, so the user saw no answer
    and no error, and the panel's next "Ask again" continued that empty
    exchange rather than starting over. The refusal belongs on this route, one
    layer above the primitive, because this is where the durable entry is
    written — ``Session.complete_aside``'s empty answer is deliberate for the
    goal-loop judge, which reads it as "no verdict".
    """
    remote = _FakeRemote(answer="   ")
    app, _bridge = _install(monkeypatch, tmp_path, remote)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        response = await client.post(
            f"/v1/desktop/sessions/{SESSION_ID}/asides",
            json={"request_id": REQUEST_ID, "text": "anything at all"},
        )
        recovered = await client.get(f"/v1/desktop/sessions/{SESSION_ID}/asides/{REQUEST_ID}")

    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "aside_empty_answer"
    assert detail["message"] == str(AsideEmptyAnswer())
    assert "Ask again" in detail["message"]
    # NOTHING STORED: no entry, so no empty turn to adopt and no "complete"
    # exchange for the next question to continue.
    assert app.state.desktop_asides == {}
    assert recovered.status_code == 404, recovered.text
    # The provider WAS asked — the refusal is about its answer, not a door.
    assert [m.text for m in remote.turns[0]] == ["anything at all"]
