"""The bulk receipt (issue #2016): ONE gesture clears the pile the badge enumerates.

WHAT THIS FILE PINS, and why each part is load-bearing:

* **The happy path, through the real route.** Two conversations with unread
  completions clear in one call; the answer's ``read`` bucket carries the
  store's own post-write states, and the badge read that follows needs NO
  test-side refresh -- the route's own cache invalidation is what makes the
  next read see it, which is the half a client's receipt repaint hangs on.
* **The shape contract.** An empty batch is REFUSED (422) rather than answered
  as a no-op: the desktop contract's ``SeenMany`` says 1..500, and the single
  route's 422-for-shape rule is the house precedent. Missing, mistyped and
  oversized batches share the refusal; unauthenticated callers meet the gate's
  401 before any of it.
* **Per-item, never per-call.** A dead id answers ``unknown`` FOR ITS ITEM --
  one stale row in a batch must not cost the other receipts their clear -- and
  a batch that clears nothing is still 200, because the three verdict buckets
  ARE the answer.
* **NOT A SWEEP.** A completion published after the caller's render is not in
  the batch and stays unread: its superseded token answers ``superseded``,
  moves no watermark, and the caller's remedy is in its own hands -- re-read,
  then post the token the state now names. This is the property that lets the
  phone's gesture inherit the desktop plane's safety story instead of
  inventing a watermark sweep.
* **The parity pin.** The clear lands in the store the DESKTOP plane opens --
  checked through the desk's own read path (``DesktopSessions.list`` ->
  ``AttentionStore(root / "attention.db")``), not a mobile-local shortcut --
  so this reddens if the mobile route ever stops writing the shared store.
* **S6's advisory half.** A ``device_id`` this machine knows is nudged once per
  conversation the batch actually cleared, with the receipt's own watermark;
  without the field the ack still lands and nobody is nudged.

Isolated config dir per test (the suite's autouse HOME isolation, plus the
config-dir monkeypatch below), like the badge's own file next door; nothing
here touches a live daemon or store.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from local_operator.mobile import push_devices
from local_operator.mobile.daemon import MobileDaemon, build_app
from local_operator.server.utils.desktop_sessions import DesktopSessions
from local_operator.session.attention import AttentionStore
from tests.unit.session.test_catalog_read_failures import _store

SEEN_MANY = "/api/attention/seen"


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *session_ids: str
) -> tuple[Path, MobileDaemon]:
    """An isolated config root with the named listable sessions, and a daemon.

    Sessions are built through the catalogue suite's own helper so this file
    and the listing/badge tests agree about what a listable conversation IS.
    """
    cfg = tmp_path / "config"
    if session_ids:
        _store(cfg, *session_ids)
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)
    return cfg, MobileDaemon(port=0, password="pw123")


def _logged_in(daemon: MobileDaemon) -> TestClient:
    client = TestClient(build_app(daemon), follow_redirects=False)
    client.post("/login", data={"password": "pw123"})
    return client


def _publish(session_id: str) -> str:
    """One completion for ``session/<id>``; returns its token."""
    token = str(uuid.uuid4())
    AttentionStore().publish(f"session/{session_id}", token, "result", "complete")
    return token


def _unread(client: TestClient) -> dict[str, Any]:
    response = client.get("/api/attention/unread")
    assert response.status_code == 200
    return response.json()


def _items(client: TestClient) -> list[dict[str, str]]:
    """The rendered pairs, as the phone reads them off the badge payload."""
    return [
        {"session_id": row["session_id"], "completion_token": row["completion_token"]}
        for row in _unread(client)["conversations"]
    ]


def _seen_many(client: TestClient, items: list[dict[str, str]], **extra: Any):
    return client.post(SEEN_MANY, json={"items": items, **extra})


def test_bulk_mark_clears_every_current_completion_in_one_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, daemon = _fixture(tmp_path, monkeypatch, "aaaaaaaaaaaa", "bbbbbbbbbbbb")
    client = _logged_in(daemon)
    _publish("aaaaaaaaaaaa")
    _publish("bbbbbbbbbbbb")
    assert _unread(client)["count"] == 2

    response = _seen_many(client, _items(client))
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert sorted(state["conversation_id"] for state in body["read"]) == [
        "session/aaaaaaaaaaaa",
        "session/bbbbbbbbbbbb",
    ]
    assert all(state["unseen"] is False for state in body["read"])
    assert body["superseded"] == []
    assert body["unknown"] == []

    # NO test-side refresh: the route's own invalidation is what the next read
    # sees, which is the half a client's receipt repaint hangs on. A route
    # that forgot it would serve the cached build and this stays at 2.
    assert _unread(client)["count"] == 0
    rows = client.get("/api/sessions").json()["sessions"]
    assert all(
        row["unseen"] is False
        for row in rows
        if row["session_id"] in ("aaaaaaaaaaaa", "bbbbbbbbbbbb")
    )


def test_the_shape_contract_refuses_empty_and_malformed_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, daemon = _fixture(tmp_path, monkeypatch, "aaaaaaaaaaaa")
    client = _logged_in(daemon)

    assert client.post(SEEN_MANY).status_code == 422  # no body at all
    empty = client.post(SEEN_MANY, json={"items": []})
    assert empty.status_code == 422
    assert "non-empty" in empty.json()["error"]
    assert client.post(SEEN_MANY, json={}).status_code == 422
    assert client.post(SEEN_MANY, json={"items": "nope"}).status_code == 422
    one_missing = client.post(SEEN_MANY, json={"items": [{"session_id": "aaaaaaaaaaaa"}]})
    assert one_missing.status_code == 422
    assert (
        client.post(
            SEEN_MANY, json={"items": [{"session_id": 7, "completion_token": "t"}]}
        ).status_code
        == 422
    )
    oversized = {
        "items": [
            {"session_id": "aaaaaaaaaaaa", "completion_token": str(uuid.uuid4())}
            for _ in range(501)
        ]
    }
    limited = client.post(SEEN_MANY, json=oversized)
    assert limited.status_code == 422
    assert "500" in limited.json()["error"]

    # The gate answers first: an unauthenticated caller gets 401, not a parse.
    bare = TestClient(build_app(daemon), follow_redirects=False)
    assert (
        bare.post(
            SEEN_MANY,
            json={"items": [{"session_id": "aaaaaaaaaaaa", "completion_token": "t"}]},
        ).status_code
        == 401
    )


def test_a_malformed_pair_is_refused_at_the_shape_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed identity is a 422, exactly as the desktop ``SeenItem`` answers it.

    Agent NIT-2 and QA Q2/Q3: the route used to accept any string and let the
    store answer ``unknown``, so a non-hex id or a non-UUID token diverged from
    the contract this route claims to mirror. The accepted shapes are the
    desktop's own (12 lowercase hex; the canonical UUID), and the id pattern is
    the constant the store's receipt path uses -- so the two surfaces answer the
    same malformed input the same way. A well-formed pair for a conversation
    this machine does not have is still the per-item ``unknown`` verdict, never
    a refusal for the call (the cell below pins that half).
    """
    cfg, daemon = _fixture(tmp_path, monkeypatch, "aaaaaaaaaaaa")
    client = _logged_in(daemon)
    token = str(uuid.uuid4())

    def refuse(session_id: str, completion_token: str) -> int:
        return _seen_many(
            client, [{"session_id": session_id, "completion_token": completion_token}]
        ).status_code

    assert refuse("zzz-not-hex", token) == 422
    assert refuse("ABCDEF012345", token) == 422, "uppercase hex is not a session id"
    assert refuse("aaaaaaaaaaa", token) == 422, "11 characters is not a session id"
    assert refuse("aaaaaaaaaaaa", "not-a-uuid") == 422
    assert refuse("aaaaaaaaaaaa", "") == 422
    assert refuse("aaaaaaaaaaaa", token.upper()) == 422, "uuids are lowercase"

    # A WELL-FORMED pair this machine cannot acknowledge keeps the per-item
    # verdict: the shape layer refuses malformed identity, not foreign identity.
    unknown = _seen_many(
        client, [{"session_id": "deadbeef1234", "completion_token": str(uuid.uuid4())}]
    )
    assert unknown.status_code == 200
    assert unknown.json() == {
        "ok": True,
        "read": [],
        "superseded": [],
        "unknown": ["deadbeef1234"],
    }


def test_a_dead_or_foreign_item_is_unknown_for_that_item_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, daemon = _fixture(tmp_path, monkeypatch, "aaaaaaaaaaaa")
    client = _logged_in(daemon)
    real = _publish("aaaaaaaaaaaa")

    response = _seen_many(
        client,
        [
            {"session_id": "aaaaaaaaaaaa", "completion_token": real},
            {"session_id": "deadbeef1234", "completion_token": str(uuid.uuid4())},
        ],
    )
    assert response.status_code == 200
    body = response.json()
    assert [state["conversation_id"] for state in body["read"]] == ["session/aaaaaaaaaaaa"]
    assert body["superseded"] == []
    assert body["unknown"] == ["deadbeef1234"], "one stale row cost the batch a clear"
    assert _unread(client)["count"] == 0

    # A known conversation with a token the store does not hold for it is the
    # SAME verdict, and a batch that clears nothing is still 200: the buckets
    # ARE the answer, so a client never throws away a partial result.
    noop = _seen_many(
        client, [{"session_id": "aaaaaaaaaaaa", "completion_token": str(uuid.uuid4())}]
    )
    assert noop.status_code == 200
    assert noop.json() == {
        "ok": True,
        "read": [],
        "superseded": [],
        "unknown": ["aaaaaaaaaaaa"],
    }


def test_a_completion_published_after_the_render_is_never_cleared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NOT A SWEEP: the batch names what the caller rendered, and the store's
    own compare runs against each conversation's CURRENT token inside the one
    write, so a newer result survives a stale receipt."""
    cfg, daemon = _fixture(tmp_path, monkeypatch, "aaaaaaaaaaaa", "bbbbbbbbbbbb")
    client = _logged_in(daemon)
    rendered_a = _publish("aaaaaaaaaaaa")
    rendered_b = _publish("bbbbbbbbbbbb")
    newer_a = _publish("aaaaaaaaaaaa")  # lands AFTER a phone rendered rendered_a

    # The receipt state BEFORE the call: a superseded item must move NOTHING,
    # so the strongest assertion is that the state is byte-equal afterwards.
    # (``sequence`` is global INSERT order, so an absolute value in a test that
    # publishes for two conversations would pin the wrong number.)
    before = AttentionStore().state("session/aaaaaaaaaaaa")
    response = _seen_many(
        client,
        [
            {"session_id": "aaaaaaaaaaaa", "completion_token": rendered_a},
            {"session_id": "bbbbbbbbbbbb", "completion_token": rendered_b},
        ],
    )
    assert response.status_code == 200
    body = response.json()
    assert [state["conversation_id"] for state in body["read"]] == ["session/bbbbbbbbbbbb"]
    assert body["superseded"] == ["aaaaaaaaaaaa"]
    assert body["unknown"] == []

    state = AttentionStore().state("session/aaaaaaaaaaaa")
    assert state == before, "a superseded item moved the receipt"
    assert state["unseen"] is True, "a stale receipt cleared a newer completion"
    assert state["completion_token"] == newer_a
    assert _unread(client)["count"] == 1

    # The caller's remedy is in its own hands: re-read, then post the token the
    # state now names.
    caught_up = _seen_many(client, [{"session_id": "aaaaaaaaaaaa", "completion_token": newer_a}])
    assert caught_up.json()["read"][0]["unseen"] is False
    assert _unread(client)["count"] == 0


def test_the_clear_is_a_write_the_desktop_plane_read_path_sees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PARITY PIN: the mobile clear is visible to the desktop's OWN read path.

    The desk serves each row's attention from the same construction the mobile
    daemon writes through -- ``AttentionStore(root / "attention.db")`` inside
    ``DesktopSessions.list`` -- so this cell drives the real mobile route and
    then asks the DESKTOP's read path, not a mobile-local shortcut. If mobile
    ever stops writing the shared store (its own legacy file, another root),
    the desk's read still says ``unseen`` and this reddens.
    """
    cfg, daemon = _fixture(tmp_path, monkeypatch, "aaaaaaaaaaaa", "bbbbbbbbbbbb")
    client = _logged_in(daemon)
    _publish("aaaaaaaaaaaa")
    _publish("bbbbbbbbbbbb")
    assert _unread(client)["count"] == 2

    response = _seen_many(client, _items(client))
    assert response.status_code == 200
    body = response.json()
    assert len(body["read"]) == 2 and not body["superseded"] and not body["unknown"]

    async def desktop_read() -> dict[str, dict[str, Any]]:
        page = await DesktopSessions(cfg).list(50)
        return {row["id"]: row["attention"] for row in page.rows}

    attention = asyncio.run(desktop_read())
    assert attention["aaaaaaaaaaaa"]["unseen"] is False
    assert attention["bbbbbbbbbbbb"]["unseen"] is False


def test_a_cleared_batch_nudges_the_worker_once_per_conversation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S6's advisory half, mirrored from the single route: the receipt is the
    route's write; the nudge only says which device acted and on what."""
    cfg, daemon = _fixture(tmp_path, monkeypatch, "aaaaaaaaaaaa", "bbbbbbbbbbbb")
    device_id = str(
        push_devices.register(
            cfg,
            {
                "platform": "ios",
                "token": "apns-token-bulk-ack",
                "environment": "production",
                "app_version": "1.0.0 (12)",
                "install_id": "3f2f0a5e-1c3b-4d6e-8a90-2b7c4d1e5f60",
                "name": "Bulkphone",
            },
        )["device_id"]
    )

    class Recorder:
        """The ``note_ack`` surface, and nothing else."""

        def __init__(self) -> None:
            self.nudged: list[tuple[str, str, int]] = []

        def note_ack(self, *, device_id: str, conversation: str, acknowledged: int) -> None:
            self.nudged.append((device_id, conversation, acknowledged))

    recorder = Recorder()
    daemon.push_worker = recorder  # type: ignore[assignment]
    client = _logged_in(daemon)
    _publish("aaaaaaaaaaaa")
    _publish("bbbbbbbbbbbb")

    response = _seen_many(client, _items(client), device_id=device_id)
    assert response.status_code == 200
    assert sorted(entry[:2] for entry in recorder.nudged) == [
        (device_id, "session/aaaaaaaaaaaa"),
        (device_id, "session/bbbbbbbbbbbb"),
    ]
    # Each nudge must carry the acknowledged watermark the receipt just moved
    # that conversation to -- the value the worker's pass matches hints against.
    assert {conversation: ack for _device, conversation, ack in recorder.nudged} == {
        state["conversation_id"]: state["revision"][1] for state in response.json()["read"]
    }, "the nudge did not carry the receipt's watermark"

    # Advisory only: without the field the ack still happens and nudges nobody.
    _publish("aaaaaaaaaaaa")
    assert _seen_many(client, _items(client)).status_code == 200
    assert len(recorder.nudged) == 2
