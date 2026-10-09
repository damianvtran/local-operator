"""The relay's mesh half on the wire: ``include_peers`` and the transfer verb.

WHY THIS FILE EXISTS. "Sessions and delegation" parity: ``GET /api/sessions``
gains ``include_peers`` (the sessions OTHER devices hold, appended after the
local rows) and ``POST /api/sessions/{id}/transfer`` gains the phone plane's
move verb. Both halves run through ``mobile/mesh.py``, which dials THIS
device's network relay exactly as the desktop plane and the CLI do — so these
cells stand up a REAL control socket (a fake relay speaking the JSON-lines
protocol) and assert what the phone is told, because a mocked dial would not
prove the envelope unwrap, the auth frame, or the zero-peer short circuit.

The rules worth pinning here, each invisible in a green unit suite otherwise:

* The remote rows are the DESKTOP's flat-field contract, and the nested
  transport ``peer`` block must NOT be published (a client grouping by it
  filed every remote row under one heading — Addendum 2 B).
* The birth stamp of an OLD-BUILD peer can arrive as a non-number; it must
  read as the no-claim zero and sort last inside its bin, never crash and
  never be special-cased.
* ``request_id`` is at-most-once: a replay returns the recorded receipt with
  ``replayed: true`` and does NOT dial again; a same-id DIFFERENT body is a
  conflict; an unconfirmed refusal is recorded (a retry replays it) while a
  plain refusal releases the id so a retry may really run.
* The unconfirmed set is the desktop route's, pinned equal here so the two
  planes cannot drift.

Nothing here touches a real session or the operator's live machine: the config
root is the isolated one conftest installs, the relay record is this test's
own, and the fake socket answers only what a cell sets it to answer.
"""

from __future__ import annotations

import json
import os
import socket
import threading
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from starlette.requests import Request
from starlette.testclient import TestClient

from local_operator.mobile import mesh as mobile_mesh
from local_operator.mobile.auth import COOKIE_NAME, sign_cookie
from local_operator.mobile.daemon import MobileDaemon, build_app
from local_operator.paths import config_dir

PASSWORD = "pw123"
SID = "abc123def456"
DEVICE = "d_9c02aabb"
REQUEST_ID = "1234abcd-1234-1234-1234-1234abcd1234"
UNCONFIRMED = "peer_unreachable"


# ---------------------------------------------------------------------------
# The fake relay: a real loopback socket speaking the control protocol
# ---------------------------------------------------------------------------


class _FakeRelay:
    """This device's network relay, reduced to the protocol ``control_request`` speaks.

    One connection per op: a hello line (carrying the record's control key),
    one op line, one ``ack`` reply. Every frame is recorded, and
    ``connections`` counts dials so a test can prove a read opened NOTHING
    (the zero-peer property is "no dial", not "no rows").
    """

    def __init__(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.sock.settimeout(0.1)
        self.port = int(self.sock.getsockname()[1])
        self.hellos: list[dict[str, Any]] = []
        self.ops: list[dict[str, Any]] = []
        self.connections = 0
        self.detail: Callable[[dict[str, Any]], dict[str, Any]] = lambda op: {}
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def _accept_loop(self) -> None:
        while not self._closed.is_set():
            try:
                conn, _ = self.sock.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            self.connections += 1
            threading.Thread(target=self._serve_op, args=(conn,), daemon=True).start()

    def _serve_op(self, conn: socket.socket) -> None:
        try:
            stream = conn.makefile("rwb")
            hello_line = stream.readline()
            op_line = stream.readline()
            if not hello_line or not op_line:
                return
            self.hellos.append(json.loads(hello_line))
            op = json.loads(op_line)
            self.ops.append(op)
            reply = {"op": "ack", "req": op.get("req", 1), "detail": self.detail(op)}
            stream.write((json.dumps(reply) + "\n").encode())
            stream.flush()
        except (OSError, ValueError):
            pass
        finally:
            conn.close()

    def close(self) -> None:
        self._closed.set()
        try:
            self.sock.close()
        except OSError:
            pass


#: The control key the published record carries; the hello frame must present it.
CONTROL_KEY = "c0ffee" * 10 + "abcd"


def _publish_relay(relay: _FakeRelay) -> None:
    """Publish this test's relay record under the isolated config root."""
    from local_operator.network.store import publish_peer_record
    from local_operator.network.types import PeerRecord

    record = PeerRecord(pid=os.getpid(), control_port=relay.port, control_key=CONTROL_KEY)
    publish_peer_record(record)


def _catalog_detail(
    sessions: list[dict[str, Any]],
    *,
    device: str = DEVICE,
    name: str = "devon-laptop",
    network: str = "n_4a1c",
) -> dict[str, Any]:
    """A ``net_catalog`` reply: the peer block runs through both halves."""
    return {
        "peers": {
            device: {
                "name": name,
                "network_id": network,
                "reachable": True,
                "reason": "",
                "age_s": 1.5,
            }
        },
        "sessions": [{"peer": {"device_id": device, "name": name}, **row} for row in sessions],
    }


def _peer_session(
    *,
    session_id: str,
    started: Any = 1789400000.0,
    state: str = "idle",
    name: str = "port the parser",
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "conversation_name": name,
        "state": state,
        "busy": state == "busy",
        "pending": None,
        "detached": False,
        "started": started,
    }


def _move_detail(*, mode: str = "move", **over: Any) -> dict[str, Any]:
    detail: dict[str, Any] = {
        "ok": True,
        "session_id": SID,
        "new_session_id": SID,
        "mode": mode,
        "to_device": {"device_id": DEVICE, "name": "devon-laptop"},
        "phases": [
            {"phase": "prepared"},
            {"phase": "handing_off"},
            {"phase": "committed"},
            {"phase": "done"},
        ],
    }
    detail.update(over)
    return detail


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_module_state() -> Iterator[None]:
    """The federated projection keeps a process-wide TTL cache; no test inherits another's."""
    from local_operator.session import peer_rows

    peer_rows.clear_cache()
    yield
    peer_rows.clear_cache()


@pytest.fixture()
def relay() -> Iterator[_FakeRelay]:
    fake = _FakeRelay()
    try:
        yield fake
    finally:
        fake.close()


@pytest.fixture()
def client() -> TestClient:
    """A logged-in client over the real daemon app, under the test's config root."""
    app = build_app(MobileDaemon(port=0, password=PASSWORD, dial_registrants=False))
    client = TestClient(app, follow_redirects=False)
    assert client.post("/login", data={"password": PASSWORD}).status_code in (200, 303)
    return client


def _remote_rows(client: TestClient, query: str = "include_peers=true") -> list[dict[str, Any]]:
    response = client.get(f"/api/sessions?{query}")
    assert response.status_code == 200, response.text
    return list(response.json()["sessions"])


# ---------------------------------------------------------------------------
# The read half: include_peers
# ---------------------------------------------------------------------------


def test_include_peers_appends_the_desktops_flat_rows(
    client: TestClient, relay: _FakeRelay
) -> None:
    """The remote row carries the desktop's flat fields — and NOT the nested peer block."""
    started = 1789400000.0
    relay.detail = lambda op: _catalog_detail(
        [_peer_session(session_id=SID, started=started, state="busy")]
    )
    _publish_relay(relay)

    rows = _remote_rows(client)
    remote = [row for row in rows if row.get("locality") == "remote"]
    assert len(remote) == 1
    row = remote[0]
    assert row["session_id"] == SID
    assert row["conversation_name"] == "port the parser"
    assert row["created_at"] == started
    assert row["section"] == "active"  # a live_state is ACTIVE, the shared binning rule
    assert row["live_state"] == "busy"
    assert row["pending"] is None
    # The desktop's flat locality fields, same names and meanings.
    assert row["owner_device"] == DEVICE
    assert row["owner_device_name"] == "devon-laptop"
    assert row["reachable"] is True
    assert row["unreachable_reason"] == ""
    assert row["placement"] is None
    assert row["origin"] is None
    # NEVER the nested transport block (Addendum 2 B: a client grouping by it
    # filed every remote row under one heading).
    assert "peer" not in row
    # The auth frame carried this install's record: hello first, then the op.
    assert relay.hellos and relay.hellos[0]["key"] == CONTROL_KEY
    # The catalogue reads BOTH halves (the peer table and the rows) over this
    # one op; what matters here is that only the catalogue op was asked for.
    assert {op["op"] for op in relay.ops} == {"peer_session_rows"}


def test_without_include_peers_the_answer_is_unchanged_and_no_socket_opens(
    client: TestClient, relay: _FakeRelay
) -> None:
    relay.detail = lambda op: _catalog_detail([_peer_session(session_id=SID)])
    _publish_relay(relay)

    rows = _remote_rows(client, query="")
    assert all(row.get("locality") != "remote" for row in rows)
    assert relay.connections == 0


def test_a_machine_with_no_relay_opens_no_socket(client: TestClient, relay: _FakeRelay) -> None:
    """The zero-peer short circuit: no record published, so no dial at all."""
    relay.detail = lambda op: _catalog_detail([_peer_session(session_id=SID)])
    # NOTE: nothing published.

    rows = _remote_rows(client)
    assert rows == []
    assert relay.connections == 0


def test_the_non_number_birth_stamp_falls_within_the_bins(
    client: TestClient, relay: _FakeRelay
) -> None:
    """An old-build peer's ``started`` is a non-number: no crash, no special case.

    The claim reads as the no-claim zero (already handled inside
    ``session.peer_rows``), so the row lands in the ordinary bins and sorts
    LAST among its section's rows — the honest direction for an unknown birth.
    """
    relay.detail = lambda op: _catalog_detail(
        [
            _peer_session(session_id="old-build-1", started="not-a-number", state="stored"),
            _peer_session(session_id="real-stamp-1", started=1789400000.0, state="stored"),
        ]
    )
    _publish_relay(relay)

    rows = _remote_rows(client)
    remote = [row for row in rows if row.get("locality") == "remote"]
    assert [row["session_id"] for row in remote] == ["real-stamp-1", "old-build-1"]
    old = remote[1]
    assert old["created_at"] == 0.0
    assert old["section"] == "previous"
    assert remote[0]["section"] == "previous"


def test_the_unconfirmed_set_is_the_desktop_routes() -> None:
    """Drift guard: the two planes must branch on the SAME refusal family."""
    from local_operator.server.routes.desktop_mesh import _MOVE_UNCONFIRMED_CODES

    assert mobile_mesh.MOVE_UNCONFIRMED_CODES == _MOVE_UNCONFIRMED_CODES


# ---------------------------------------------------------------------------
# Auth: the new surface refuses exactly like its neighbours
# ---------------------------------------------------------------------------


def test_unauthenticated_calls_are_refused_like_the_neighbouring_routes() -> None:
    app = build_app(MobileDaemon(port=0, password=PASSWORD, dial_registrants=False))
    anonymous = TestClient(app, follow_redirects=False)

    neighbour = anonymous.get("/api/sessions")
    peers = anonymous.get("/api/sessions?include_peers=true")
    transfer = anonymous.post(f"/api/sessions/{SID}/transfer", json={"to": DEVICE})
    for response in (neighbour, peers, transfer):
        assert response.status_code == 401
        assert response.json() == {"error": "authentication required"}


# ---------------------------------------------------------------------------
# The write half: the transfer route
# ---------------------------------------------------------------------------


def _transfer(client: TestClient, body: dict[str, Any], *, session_id: str = SID) -> Any:
    return client.post(f"/api/sessions/{session_id}/transfer", json=body)


def test_transfer_shapes_the_desktop_receipt(client: TestClient, relay: _FakeRelay) -> None:
    relay.detail = lambda op: _move_detail()
    _publish_relay(relay)

    response = _transfer(client, {"to": DEVICE, "keep": False, "wait_s": 0})
    assert response.status_code == 200, response.text
    receipt = response.json()
    assert receipt["locality"] == "remote"
    assert receipt["owner_device"] == DEVICE
    assert receipt["source_retired"] is True
    assert receipt["session_id"] == SID
    assert receipt["new_session_id"] == SID
    assert receipt["mode"] == "move"
    assert receipt["replayed"] is False
    assert receipt["phases"] == [
        {"phase": "prepared", "peer": DEVICE, "progress": 0.25},
        {"phase": "handing_off", "peer": DEVICE, "progress": 0.5},
        {"phase": "committed", "peer": DEVICE, "progress": 0.75},
        {"phase": "done", "peer": DEVICE, "progress": 1.0},
    ]
    frame = relay.ops[-1]
    assert frame["op"] == "session_move"
    assert frame["session_id"] == SID
    assert frame["action"] == "offload"
    assert frame["to"] == DEVICE
    assert frame["keep"] is False
    assert frame["wait_s"] == 0.0


def test_a_keep_copy_says_so(client: TestClient, relay: _FakeRelay) -> None:
    relay.detail = lambda op: _move_detail(mode="keep")
    _publish_relay(relay)

    response = _transfer(client, {"to": DEVICE, "keep": True, "wait_s": 0})
    assert response.status_code == 200
    receipt = response.json()
    assert receipt["mode"] == "keep"
    assert receipt["source_retired"] is False
    assert relay.ops[-1]["keep"] is True


def test_request_id_replay_returns_the_recorded_receipt_without_dialling_again(
    client: TestClient, relay: _FakeRelay
) -> None:
    relay.detail = lambda op: _move_detail()
    _publish_relay(relay)

    first = _transfer(client, {"to": DEVICE, "request_id": REQUEST_ID})
    assert first.status_code == 200
    assert first.json()["replayed"] is False
    ops_after_first = len(relay.ops)

    second = _transfer(client, {"to": DEVICE, "request_id": REQUEST_ID})
    assert second.status_code == 200
    assert second.json()["replayed"] is True
    assert second.json()["session_id"] == first.json()["session_id"]
    # The replay came from the journal: the relay was not asked a second time.
    assert len(relay.ops) == ops_after_first == 1


def test_same_request_id_with_different_input_is_a_conflict(
    client: TestClient, relay: _FakeRelay
) -> None:
    relay.detail = lambda op: _move_detail()
    _publish_relay(relay)

    assert _transfer(client, {"to": DEVICE, "request_id": REQUEST_ID}).status_code == 200
    conflict = _transfer(client, {"to": "local", "request_id": REQUEST_ID})
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "receipt_conflict"
    assert "different input" in conflict.json()["error"]
    assert len(relay.ops) == 1


def test_the_journal_is_durable_across_daemon_instances(
    client: TestClient, relay: _FakeRelay
) -> None:
    """A restart does not re-open a recorded id: the file IS the receipt."""
    relay.detail = lambda op: _move_detail()
    _publish_relay(relay)
    assert _transfer(client, {"to": DEVICE, "request_id": REQUEST_ID}).status_code == 200

    # A second daemon (a restart) over the SAME config root.
    second_app = build_app(MobileDaemon(port=0, password=PASSWORD, dial_registrants=False))
    second = TestClient(second_app, follow_redirects=False)
    assert second.post("/login", data={"password": PASSWORD}).status_code in (200, 303)
    replay = second.post(
        f"/api/sessions/{SID}/transfer", json={"to": DEVICE, "request_id": REQUEST_ID}
    )
    assert replay.status_code == 200
    assert replay.json()["replayed"] is True
    assert len(relay.ops) == 1


def test_wait_s_is_bounded_to_the_desktop_range(client: TestClient, relay: _FakeRelay) -> None:
    relay.detail = lambda op: _move_detail()
    _publish_relay(relay)

    for bad in (300.5, -1, "30", True):
        response = _transfer(client, {"to": DEVICE, "wait_s": bad})
        assert response.status_code == 422, (bad, response.text)
        assert "wait_s" in response.json()["error"]
    assert len(relay.ops) == 0

    ok = _transfer(client, {"to": DEVICE, "wait_s": 300})
    assert ok.status_code == 200
    assert relay.ops[-1]["wait_s"] == 300.0


def test_malformed_bodies_are_refused_by_name(client: TestClient, relay: _FakeRelay) -> None:
    _publish_relay(relay)

    for body, needle in (
        ({"to": "bad id!"}, "'to'"),
        ({"to": DEVICE, "keep": "yes"}, "'keep'"),
        ({"to": DEVICE, "request_id": "not-a-uuid"}, "'request_id'"),
        ({"to": DEVICE, "keeep": True}, "keeep"),
    ):
        response = _transfer(client, body)
        assert response.status_code == 422, (body, response.text)
        assert needle in response.json()["error"]
    assert relay.connections == 0


def test_a_source_that_does_not_exist_refuses_and_releases_the_id(
    client: TestClient, relay: _FakeRelay
) -> None:
    refusal = {
        "ok": False,
        "code": "unknown_session",
        "message": f"this device does not hold a session {SID}",
        "changed": False,
    }
    relay.detail = lambda op: refusal
    _publish_relay(relay)

    first = _transfer(client, {"to": DEVICE, "request_id": REQUEST_ID})
    assert first.status_code == 409
    assert first.json() == {
        "error": f"this device does not hold a session {SID}",
        "code": "unknown_session",
    }
    # Nothing was moved, so the id was RELEASED: the retry may really run.
    second = _transfer(client, {"to": DEVICE, "request_id": REQUEST_ID})
    assert second.status_code == 409
    assert len(relay.ops) == 2


def test_an_unconfirmed_refusal_is_recorded_and_replayed(
    client: TestClient, relay: _FakeRelay
) -> None:
    """The request may be in flight: a retry replays 503 instead of trying again."""
    relay.detail = lambda op: {
        "ok": False,
        "code": UNCONFIRMED,
        "message": "the peer stopped replying after the request arrived",
        "changed": True,
    }
    _publish_relay(relay)

    first = _transfer(client, {"to": DEVICE, "request_id": REQUEST_ID})
    assert first.status_code == 503
    assert first.json()["code"] == UNCONFIRMED
    second = _transfer(client, {"to": DEVICE, "request_id": REQUEST_ID})
    assert second.status_code == 503
    assert second.json()["code"] == UNCONFIRMED
    assert len(relay.ops) == 1


def test_a_corrupt_receipt_store_refuses_rather_than_resetting(
    client: TestClient, relay: _FakeRelay
) -> None:
    """An unreadable journal must never read as "no recorded requests"."""
    relay.detail = lambda op: _move_detail()
    _publish_relay(relay)
    config_dir().mkdir(parents=True, exist_ok=True)
    (config_dir() / "mobile-transfer-receipts.json").write_text("{not json", encoding="utf-8")

    response = _transfer(client, {"to": DEVICE, "request_id": REQUEST_ID})
    assert response.status_code == 503
    assert response.json()["code"] == "receipt_store_unreadable"
    assert relay.connections == 0


# ---------------------------------------------------------------------------
# The SSE half: the list stream takes the same flag
# ---------------------------------------------------------------------------


async def _first_sessions_frame(app: Any, query: bytes) -> dict[str, Any]:
    endpoint = next(
        r for r in app.routes if getattr(r, "path", "") == "/api/sessions/events"
    ).endpoint
    response = await endpoint(
        Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/api/sessions/events",
                "query_string": query,
                "headers": [
                    (b"host", b"fixture"),
                    (b"cookie", f"{COOKIE_NAME}={sign_cookie(PASSWORD)}".encode()),
                ],
                "scheme": "http",
                "server": ("fixture", 80),
            }
        )
    )
    try:
        frame = await anext(response.body_iterator)
    finally:
        await response.body_iterator.aclose()
    line = next(line for line in frame.splitlines() if line.startswith("data:"))
    return json.loads(line[len("data:") :].strip())


@pytest.mark.asyncio
async def test_the_list_stream_carries_peers_only_when_asked(relay: _FakeRelay) -> None:
    relay.detail = lambda op: _catalog_detail([_peer_session(session_id=SID, state="idle")])
    _publish_relay(relay)

    app = build_app(MobileDaemon(port=0, password=PASSWORD, dial_registrants=False))
    asked = await _first_sessions_frame(app, b"include_peers=true")
    assert [row["session_id"] for row in asked["sessions"]] == [SID]

    from local_operator.session import peer_rows

    peer_rows.clear_cache()
    plain = await _first_sessions_frame(app, b"")
    assert plain["sessions"] == []
