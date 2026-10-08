"""``/api/sessions/{id}/checkpoints`` on the wire: the rail's ticks, whole-conversation.

WHY THIS FILE EXISTS. ``GET /api/sessions/{session_id}/checkpoints`` is the
relay's half of the desktop rail's manifest
(``GET /v1/desktop/sessions/{session_id}/checkpoints``, design D9): the phone's
transcript rail reads it, so these assertions are about what THE PHONE IS TOLD,
over the real app and a real journal -- not about the derivation
(``tests/unit/session/test_transcript_index.py`` owns that). Where a cell
matters on both surfaces it is asserted here against the desktop's shape,
because "field for field the desktop's" is only true while it is tested.

The three rules worth pinning here, because each is invisible in a green unit
suite and load-bearing on the client:

* The manifest covers EVERY turn of the journal, derived with NO runtime: the
  phone's own projection is a bounded tail window, so a rail built from the
  frames a phone holds would silently mark only the tail it carries.
* ``index.state`` keeps the empty answer and the failed answer apart: a
  journal that fails to read after a successful stat answers ``error``, and
  must never render as "no checkpoints" (the cell below pins the ``chmod 000``
  file; the un-``stat``-able case is the shared derivation's -- deferred, see
  PR #2068).
* The rows are the desktop rail's own wire models; a declared-field move on
  those models reds the pin below before a phone can read a default the
  desktop never serves.

Nothing here touches a real session: the config root is the isolated one
conftest installs, and every journal written is this test's own fixture.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from local_operator.mobile.daemon import MobileDaemon, build_app
from local_operator.paths import config_dir
from local_operator.session import transcript_index as ti

SID = "cafe00112233"


@pytest.fixture(autouse=True)
def _clean_module_state():
    """The manifest module keeps process-wide loop state; no test may inherit another's."""
    ti._reset_for_tests()
    yield
    ti._reset_for_tests()


def _client() -> TestClient:
    """A logged-in client over the real daemon app, under the test's HOME."""
    client = TestClient(build_app(MobileDaemon(port=0, password="pw123")), follow_redirects=False)
    assert client.post("/login", data={"password": "pw123"}).status_code in (200, 303)
    return client


def _journal(session_id: str, rows: list[dict[str, Any]]) -> Path:
    """Write a journal in the journal's own row format (the session suite's shapes)."""
    path = config_dir() / "sessions" / session_id / "transcript.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    return path


def _user(id_: str, ts: float, text: str = "hello") -> dict[str, Any]:
    return {
        "id": id_,
        "ts": ts,
        "type": "message",
        "payload": {"kind": "message", "role": "user", "content": [{"text": text}]},
    }


def _assistant(id_: str, ts: float, text: str = "answer") -> dict[str, Any]:
    return {
        "id": id_,
        "ts": ts,
        "type": "message",
        "payload": {"kind": "message", "role": "assistant", "content": [{"text": text}]},
    }


def _inject(id_: str, ts: float, text: str = "peer says hi") -> dict[str, Any]:
    """A cross-session notification: chrome, never a turn of its own."""
    return {
        "id": id_,
        "ts": ts,
        "type": "message",
        "payload": {"kind": "custom", "custom_type": "hub_message", "details": {"text": text}},
    }


def _start(id_: str, ts: float, token: str) -> dict[str, Any]:
    return {
        "id": id_,
        "ts": ts,
        "type": "custom",
        "payload": {
            "custom_type": "attention_started",
            "details": {"conversation_id": f"session/{SID}", "token": token},
        },
    }


def _marker(
    id_: str, ts: float, token: str, *, kind: str | None = "complete", eligible: bool = True
) -> dict[str, Any]:
    details: dict[str, Any] = {
        "conversation_id": f"session/{SID}",
        "token": token,
        "eligible": eligible,
    }
    if kind is not None and eligible:
        details["kind"] = kind
        details["anchor"] = "a-anchor"
    return {
        "id": id_,
        "ts": ts,
        "type": "custom",
        "payload": {"custom_type": "completion_attention", "details": details},
    }


def _turn(index: int, *, outcome: str | None = "complete") -> list[dict[str, Any]]:
    """One turn, in the journal's own order: start, user, answer, marker.

    ``outcome=None`` leaves the turn UNCLOSED (no marker row), which is what
    makes it the live open tail on the rail."""
    base = float(index * 10)
    rows = [
        _start(f"s{index}", base, f"t{index}"),
        _user(f"u{index}", base + 1, f"question {index}"),
        _assistant(f"a{index}", base + 2, f"answer {index}"),
    ]
    if outcome is not None:
        rows.append(_marker(f"m{index}", base + 3, f"t{index}", kind=outcome))
    return rows


def _body(client: TestClient, session_id: str = SID) -> dict[str, Any]:
    response = client.get(f"/api/sessions/{session_id}/checkpoints")
    assert response.status_code == 200, response.text
    return response.json()


def _settled(client: TestClient, session_id: str = SID) -> dict[str, Any]:
    """Poll until the manifest leaves ``building`` (small journals settle within
    the first-paint wait; the poll only makes the assertion load-proof)."""
    body: dict[str, Any] = {}
    for _ in range(60):
        body = _body(client, session_id)
        if body["index"]["state"] != "building":
            return body
        time.sleep(0.05)
    raise AssertionError(f"manifest never settled: {body}")


def test_the_gate_holds_and_login_unlocks_the_surface() -> None:
    _journal(SID, _turn(0))
    client = TestClient(build_app(MobileDaemon(port=0, password="pw123")), follow_redirects=False)
    refused = client.get(f"/api/sessions/{SID}/checkpoints")
    assert refused.status_code == 401
    assert refused.json()["error"] == "authentication required"

    assert client.post("/login", data={"password": "pw123"}).status_code in (200, 303)
    body = _settled(client)
    assert set(body) == {"session_id", "index", "checkpoints"}


def test_an_unknown_session_is_refused_like_the_reads_beside_it() -> None:
    client = _client()
    response = client.get("/api/sessions/nosuchsession/checkpoints")
    assert response.status_code == 404
    assert response.json() == {"error": "unknown session"}


def test_a_conversation_with_no_user_turns_answers_the_empty_ready_manifest() -> None:
    """A notification with no turn around it is not a rail tick: nothing to mark
    is the honest ``ready`` answer, and it exists beside (not instead of) the
    unreadable case below."""
    _journal(SID, [_inject("n1", 1.0)])

    body = _settled(_client())

    assert body["index"]["state"] == "ready"
    assert body["checkpoints"] == []


def test_the_manifest_covers_every_turn_with_its_own_outcome() -> None:
    """Three turns -- one completed, one failed, one open tail -- come back whole
    and in order, in the desktop rail's own wire shape."""
    _journal(SID, [*_turn(0), *_turn(1, outcome="error"), *_turn(2, outcome=None)])

    body = _settled(_client())

    assert body["session_id"] == SID
    assert body["index"]["state"] == "ready"
    assert isinstance(body["index"]["built_at"], float)

    entries = body["checkpoints"]
    assert [(entry["kind"], entry["id"]) for entry in entries] == [
        ("user", "u0"),
        ("completion", "a0"),
        ("user", "u1"),
        ("completion", "a1"),
        ("user", "u2"),
        ("completion", "a2"),
    ]
    assert [entry["turn"] for entry in entries] == [1, 1, 2, 2, 3, 3]
    # ``seq`` is the journal row ordinal (0-based), which is what places a tick
    # proportionally in the rail without loading the row.
    assert [entry["seq"] for entry in entries] == [1, 2, 5, 6, 9, 10]
    assert [entry["outcome"] for entry in entries] == [
        None,
        "complete",
        None,
        "error",
        None,
        "open",
    ]
    assert [entry["text"] for entry in entries] == [
        "question 0",
        "answer 0",
        "question 1",
        "answer 1",
        "question 2",
        "answer 2",
    ]
    for entry in entries:
        assert set(entry) >= {"id", "kind", "turn", "ts", "seq", "text", "outcome", "naming"}
        if entry["kind"] == "user":
            # Nothing to name on a user tick; the keys are null, not absent.
            assert entry["naming"] is None
        else:
            # No naming section yet: the state the rail polls on.
            assert entry["naming"] == {"state": "pending", "name": None, "summary": None}


def test_the_manifest_covers_a_conversation_larger_than_one_phone_page() -> None:
    """45 turns -- past the phone's page budget and its tail window -- must come
    back whole: coverage cannot depend on which frames the phone happens to
    hold, which is the whole reason this route exists."""
    turns = 45
    rows: list[dict[str, Any]] = []
    for index in range(turns):
        rows.extend(_turn(index))
    _journal(SID, rows)

    body = _settled(_client())

    entries = body["checkpoints"]
    assert len(entries) == turns * 2
    assert [entry["id"] for entry in entries] == [
        id_ for index in range(turns) for id_ in (f"u{index}", f"a{index}")
    ]


def test_an_unreadable_journal_answers_error_never_an_empty_ready_manifest() -> None:
    """The one case where "no checkpoints" would be a claim this process has
    not earned: the journal is there and cannot be read. The answer is the
    manifest's own ``error`` state -- a rail must hide (or say so), never
    render nothing over a failed read."""
    path = _journal(SID, _turn(0))
    client = _client()
    path.chmod(0)
    seen: list[str] = []
    body: dict[str, Any] = {}
    try:
        for _ in range(80):
            body = _body(client)
            seen.append(body["index"]["state"])
            if body["index"]["state"] == "error":
                break
            time.sleep(0.05)
    finally:
        path.chmod(0o600)

    assert body["index"]["state"] == "error", f"states seen: {seen}"
    assert body["checkpoints"] == []
    assert "ready" not in seen


def test_the_shared_wire_models_field_sets_are_pinned() -> None:
    """The mirror guarantee in ``local_operator/mobile/checkpoints.py``, made
    checkable: the rows on this wire are built as the desktop rail's own models,
    and this cell is what makes a declared field ADDED, RENAMED or REMOVED on
    those models fail here -- construction alone cannot, because an extra is
    ignored (Pydantic's default) and a rename lands as a default.

    A red here means the shared wire moved: the emitter
    (``session/transcript_index.py``) and the phone's client contract must be
    mirrored before a phone can read a default the desktop never serves."""
    from local_operator.server.models.desktop_sessions import (
        CheckpointEntry,
        CheckpointIndex,
        CheckpointManifest,
        CheckpointNaming,
    )

    assert set(CheckpointNaming.model_fields) == {"state", "name", "summary"}
    assert set(CheckpointEntry.model_fields) == {
        "id",
        "kind",
        "turn",
        "ts",
        "seq",
        "text",
        "outcome",
        "naming",
    }
    assert set(CheckpointIndex.model_fields) == {"state", "built_at"}
    assert set(CheckpointManifest.model_fields) == {"session_id", "index", "checkpoints"}
