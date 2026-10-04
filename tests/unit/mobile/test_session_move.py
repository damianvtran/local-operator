"""The phone's one-tap new session and its in-composer directory move.

WHY THIS FILE EXISTS. ``POST /api/sessions/start`` with no ``cwd``, the
``default`` field ``GET /api/directories`` publishes, and
``POST /api/sessions/{id}/directory`` are one feature with three surfaces: the
phone starts a conversation in "where the user has been working lately" without
a picker screen, and lets the user point a session that has not been used at a
different directory without losing the conversation's identity. The rules that
are easy to break silently and expensive to find later, and which these cells
pin:

* the DEFAULT is resolved by ONE function, so the directory a no-cwd start lands
  in is always a directory ``/api/directories`` also offers;
* the RECENTS list is filtered by the spawn gate, so no suggestion is a tap
  that 400s;
* a move RIDES THE RELAY'S OWN VIEWER CONNECTION, because the runtime refuses a
  pristine retire while any other attach client is registered and a fresh dial
  would be counted against itself;
* a move WAITS for the retiring owner to unpublish before spawning, or the
  successor adopts the dying runtime and answers with a pid that is about to
  exit;
* the refusals are the three stable codes the native app branches on, with the
  copy it renders verbatim -- never the runtime's own ``kept: …`` prose.

Nothing here touches the operator's machine: the transport is a fake viewer, so
no runtime is dialled and no child is spawned (``spawn_session`` is recorded,
never called for real).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from local_operator.mobile import daemon as daemon_mod
from local_operator.mobile.daemon import (
    MobileDaemon,
    SessionEntry,
    _default_start_cwd,
    _directory_refusal_code,
    build_app,
)
from local_operator.mobile.types import SessionProjection, SessionRecord

SESSION = "abcdef123456"
PID = 4242


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _record(session_id: str = SESSION, pid: int = PID, cwd: str = "/synthetic") -> SessionRecord:
    return SessionRecord(
        pid=pid,
        kind="daemon",
        session_id=session_id,
        conversation_name="",
        cwd=cwd,
        model_label="fixture",
        control_port=1,
        control_key="fixture",
    )


def _client_and_daemon(
    *, session_id: str | None = SESSION, streaming: bool = False, cwd: str = "/synthetic"
) -> tuple[TestClient, MobileDaemon]:
    """A logged-in client over the real app, plus the daemon and (optionally) a
    registered live entry for ``session_id``."""
    daemon = MobileDaemon(port=0, password="pw123")
    if session_id is not None:
        record = _record(session_id=session_id, cwd=cwd)
        entry = SessionEntry(record)
        entry.projection = SessionProjection(
            session_id=session_id, pid=record.pid, kind="daemon", cwd=cwd, streaming=streaming
        )
        daemon.table.entries[record.pid] = entry
    client = TestClient(build_app(daemon), follow_redirects=False)
    assert client.post("/login", data={"password": "pw123"}).status_code in (200, 303)
    return client, daemon


class FakeViewer:
    """Stands in for the relay's own viewer connection (``_phone_attaches``).

    Deliberately NOT an AttachClient: the assertion that matters is that the
    route ASKS THIS ONE, so a fake that records the ask and answers with a
    chosen ``kept: …`` detail is the whole instrument.
    """

    def __init__(self, detail: str = "retired") -> None:
        self.detail = detail
        self.asked = 0
        self.connected = True

    async def retire_if_pristine(self) -> str:
        self.asked += 1
        return self.detail

    def close(self) -> None:
        self.connected = False


def _prepare_move(
    monkeypatch: pytest.MonkeyPatch,
    daemon: MobileDaemon,
    *,
    detail: str = "retired",
    unpublishes: bool = True,
    new_pid: int = 9090,
) -> tuple[FakeViewer, list[tuple[str, str | None]]]:
    """Wire the route's transport to fakes and return the viewer and the spawn
    log (``(cwd, resume)`` per call)."""
    viewer = FakeViewer(detail)
    monkeypatch.setattr(daemon, "phone_viewer", lambda _session_id: viewer)
    spawns: list[tuple[str, str | None]] = []

    async def fake_spawn(
        cwd: str,
        provider: str | None = None,
        model_id: str | None = None,
        resume: str | None = None,
    ) -> int:
        spawns.append((cwd, resume))
        return new_pid

    monkeypatch.setattr(daemon, "spawn_session", fake_spawn)

    async def fake_wait(_session_id: str, _pid: int, _timeout: float) -> bool:
        return unpublishes

    monkeypatch.setattr(daemon_mod, "_wait_for_owner_to_unpublish", fake_wait)
    return viewer, spawns


# --------------------------------------------------------------------------
# Default resolution
# --------------------------------------------------------------------------


def test_default_start_cwd_prefers_the_first_admitted_recent(tmp_path, monkeypatch) -> None:
    """A recent directory the spawn gate refuses (outside home/tmp) must not
    become the default -- the answer has to stay spawnable."""
    home = tmp_path / "home"
    outside = tmp_path / "elsewhere"
    for path in (home, outside):
        path.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    # The fake scratch root matters: pytest's ``tmp_path`` already lives under
    # the REAL system temp dir, so an unpatched ``_tmp_dir`` would admit
    # ``outside`` as a tmp child and the gate would look broken (the same trap
    # ``test_spawn_dir_gate_allows_home_and_tmp_only`` records).
    monkeypatch.setattr(daemon_mod, "_tmp_dir", lambda: str(tmp_path / "scratch"))
    monkeypatch.setattr(
        daemon_mod, "_recent_directories", lambda limit=8: [str(outside), str(home / "proj")]
    )
    (home / "proj").mkdir()
    assert _default_start_cwd() == str(home / "proj")

    # Nothing admitted: home is the fallback, and home is always admitted.
    monkeypatch.setattr(daemon_mod, "_recent_directories", lambda limit=8: [str(outside)])
    assert _default_start_cwd() == str(home)


def test_recent_directories_drops_a_directory_the_spawn_gate_refuses(tmp_path, monkeypatch) -> None:
    """THE DEAD-TAP RULE: a recent outside home/tmp would be offered as a
    one-tap suggestion and then 400 at start, so the list is filtered with the
    same predicate the start route applies."""
    home = tmp_path / "home"
    home.mkdir()
    inside = home / "proj"
    inside.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    monkeypatch.setattr(daemon_mod, "_tmp_dir", lambda: str(tmp_path / "scratch"))

    class FakeRegistry:
        def __init__(self, **_kwargs) -> None:
            pass

        def list_agents(self):
            class Agent:
                def __init__(self, cwd: str) -> None:
                    self.current_working_directory = cwd
                    self.last_message_datetime = "2026-01-01"

            return [Agent(str(outside)), Agent(str(inside))]

    monkeypatch.setattr("local_operator.agents.AgentRegistry", FakeRegistry)
    assert daemon_mod._recent_directories() == [str(inside)]


def test_directories_default_is_the_same_function_the_start_route_uses(
    tmp_path, monkeypatch
) -> None:
    """Pin the equality: the picker's ``default`` and the cwd a no-cwd start
    hands the spawner must be one answer, or the phone lands somewhere the
    picker never offered."""
    sentinel = tmp_path / "sentinel"
    sentinel.mkdir()
    monkeypatch.setattr(daemon_mod, "_default_start_cwd", lambda: str(sentinel))

    client, daemon = _client_and_daemon(session_id=None)
    assert client.get("/api/directories").json()["default"] == str(sentinel)

    spawns: list[tuple[str, str | None]] = []

    async def fake_spawn(cwd, provider=None, model_id=None, resume=None) -> int:
        spawns.append((cwd, resume))
        return 1234

    monkeypatch.setattr(daemon, "spawn_session", fake_spawn)
    reply = client.post("/api/sessions/start", json={})
    assert reply.status_code == 200
    assert len(spawns) == 1
    assert spawns[0][0] == str(sentinel)
    # The start route mints the durable conversation id and hands it to the
    # child as its resume handle (the route is an identity, never a pid).
    assert reply.json()["session_id"] == spawns[0][1]
    assert reply.json()["pid"] == 1234


# --------------------------------------------------------------------------
# The move
# --------------------------------------------------------------------------


def test_move_starts_a_successor_in_the_new_directory_under_the_same_id(
    tmp_path, monkeypatch
) -> None:
    """The happy path: the session id is PRESERVED (it is the published
    identity), the successor is a new generation, and the ask rode the relay's
    own viewer connection."""
    target = tmp_path / "target"
    target.mkdir()
    client, daemon = _client_and_daemon()
    viewer, spawns = _prepare_move(monkeypatch, daemon, new_pid=777)

    reply = client.post(f"/api/sessions/{SESSION}/directory", json={"cwd": str(target)})
    assert reply.status_code == 200
    assert reply.json() == {"ok": True, "pid": 777, "session_id": SESSION}
    assert spawns == [(str(target), SESSION)]
    assert viewer.asked == 1


def test_move_dials_a_throwaway_only_when_no_viewer_is_attached(tmp_path, monkeypatch) -> None:
    """With no phone watching there is no connection to be counted against, so
    the route dials its own -- and must close it again."""
    target = tmp_path / "target"
    target.mkdir()
    client, daemon = _client_and_daemon()
    monkeypatch.setattr(daemon, "phone_viewer", lambda _session_id: None)

    dialled: list[FakeViewer] = []

    class FakeAttachClient(FakeViewer):
        def __init__(self, *_args, **_kwargs) -> None:
            super().__init__("retired")
            dialled.append(self)

        async def connect(self, _record, _session_id) -> None:
            return None

    monkeypatch.setattr("local_operator.mobile.attach_client.AttachClient", FakeAttachClient)

    async def fake_spawn(cwd, provider=None, model_id=None, resume=None) -> int:
        return 4321

    monkeypatch.setattr(daemon, "spawn_session", fake_spawn)
    monkeypatch.setattr(
        daemon_mod, "_wait_for_owner_to_unpublish", lambda *_a, **_k: asyncio.sleep(0, True)
    )

    reply = client.post(f"/api/sessions/{SESSION}/directory", json={"cwd": str(target)})
    assert reply.status_code == 200
    assert len(dialled) == 1
    assert dialled[0].asked == 1
    assert dialled[0].connected is False


@pytest.mark.parametrize(
    ("detail", "code", "sentence"),
    [
        (
            "kept: session has work or history",
            "session_has_history",
            "This session already has messages, so its working directory can't change.",
        ),
        (
            "kept: work arrived while stopping was announced",
            "session_busy",
            "This session is busy right now, so its working directory can't change.",
        ),
        (
            "kept: 1 viewer(s) still attached",
            "move_unavailable",
            "The working directory can't change right now.",
        ),
        (
            "kept: this runtime cannot stop itself gracefully",
            "move_unavailable",
            "The working directory can't change right now.",
        ),
        (
            "kept: pristine probe failed (boom)",
            "move_unavailable",
            "The working directory can't change right now.",
        ),
    ],
)
def test_move_maps_the_runtimes_own_refusals_onto_the_phones_codes(
    tmp_path, monkeypatch, detail: str, code: str, sentence: str
) -> None:
    """The runtime's ``kept: …`` prose is for a log; the phone gets one of three
    stable codes and a sentence it renders verbatim. Nothing moves on a
    refusal."""
    target = tmp_path / "target"
    target.mkdir()
    client, daemon = _client_and_daemon()
    viewer, spawns = _prepare_move(monkeypatch, daemon, detail=detail)

    reply = client.post(f"/api/sessions/{SESSION}/directory", json={"cwd": str(target)})
    assert reply.status_code == 409
    assert reply.json() == {"error": sentence, "code": code}
    assert spawns == []
    assert viewer.asked == 1


def test_move_is_refused_while_a_turn_is_running(tmp_path, monkeypatch) -> None:
    """A live turn is BUSY, not "has history": the sentence has to say the
    thing the user can act on (try again when it lands) rather than the
    permanent one. The runtime is never asked."""
    target = tmp_path / "target"
    target.mkdir()
    client, daemon = _client_and_daemon(streaming=True)
    viewer, spawns = _prepare_move(monkeypatch, daemon)

    reply = client.post(f"/api/sessions/{SESSION}/directory", json={"cwd": str(target)})
    assert reply.status_code == 409
    assert reply.json()["code"] == "session_busy"
    assert viewer.asked == 0
    assert spawns == []


def test_move_does_not_adopt_a_retiring_owner(tmp_path, monkeypatch) -> None:
    """THE ADOPTION GUARD: if the retiring owner never unpublishes, the route
    refuses rather than spawning a successor that ``spawn_session`` would
    resolve to the DYING runtime's record."""
    target = tmp_path / "target"
    target.mkdir()
    client, daemon = _client_and_daemon()
    _viewer, spawns = _prepare_move(monkeypatch, daemon, unpublishes=False)

    reply = client.post(f"/api/sessions/{SESSION}/directory", json={"cwd": str(target)})
    assert reply.status_code == 409
    assert reply.json()["code"] == "move_unavailable"
    assert spawns == []


def test_move_refuses_a_directory_the_spawn_gate_would_reject(tmp_path, monkeypatch) -> None:
    """The move applies the START route's admission to the new path, and does
    not ask the owner before it has a directory it could spawn in."""
    home = tmp_path / "home"
    home.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    monkeypatch.setattr(daemon_mod, "_tmp_dir", lambda: str(tmp_path / "scratch"))

    client, daemon = _client_and_daemon()
    viewer, spawns = _prepare_move(monkeypatch, daemon)

    reply = client.post(f"/api/sessions/{SESSION}/directory", json={"cwd": str(outside)})
    assert reply.status_code == 400
    assert "error" in reply.json()
    assert "code" not in reply.json()
    assert viewer.asked == 0
    assert spawns == []


def test_move_refuses_an_unknown_session(tmp_path, monkeypatch) -> None:
    target = tmp_path / "target"
    target.mkdir()
    client, daemon = _client_and_daemon(session_id=None)
    viewer, spawns = _prepare_move(monkeypatch, daemon)

    reply = client.post("/api/sessions/deadbeef0000/directory", json={"cwd": str(target)})
    assert reply.status_code == 409
    assert reply.json() == {"error": "session not connected"}
    assert viewer.asked == 0


def test_move_requires_a_directory_and_authentication(tmp_path) -> None:
    client, _daemon = _client_and_daemon()
    assert client.post(f"/api/sessions/{SESSION}/directory", json={}).status_code == 400

    target = tmp_path / "target"
    target.mkdir()
    anonymous = TestClient(
        build_app(MobileDaemon(port=0, password="pw123")), follow_redirects=False
    )
    reply = anonymous.post(f"/api/sessions/{SESSION}/directory", json={"cwd": str(target)})
    assert reply.status_code == 401

    # The SAME-ORIGIN GATE the other mutation routes carry (a sibling origin on
    # the owner's tunnel must not be able to move a session, exactly as it may
    # not start one) -- read from the shared ``gate`` helper, but pinned here so
    # a future route written without it fails a test rather than an audit.
    sibling = TestClient(
        build_app(MobileDaemon(port=0, password="pw123")),
        base_url="https://owner-lop.radienthq.com",
        follow_redirects=False,
    )
    assert sibling.post("/login", data={"password": "pw123"}).status_code == 303
    refused = sibling.post(
        f"/api/sessions/{SESSION}/directory",
        content=json.dumps({"cwd": str(target)}),
        headers={"origin": "https://other-lop.radienthq.com", "content-type": "text/plain"},
    )
    assert refused.status_code == 403


def test_directory_refusal_code_mapping_is_closed() -> None:
    """Anything the mapping does not recognise must fall to the generic
    refusal, never to a success."""
    assert _directory_refusal_code("kept: session has work or history") == "session_has_history"
    assert (
        _directory_refusal_code("kept: work arrived while stopping was announced") == "session_busy"
    )
    assert _directory_refusal_code("kept: something nobody has seen before") == "move_unavailable"
    assert _directory_refusal_code("retired") == "move_unavailable"
