"""The session plane, driven across two real relays and a real runtime socket.

WHAT MAKES THIS EVIDENCE. Two config roots, two device identities, two relays on
real loopback sockets with the real handshake, and — on the owning side — a REAL
``RuntimeServer`` serving its control socket, publishing a real discovery record,
and recording what actually reached it (``FakeHandle.calls``). Nothing here stubs
the transport or the session plane: the ops under test are the ones the CLI
drives, framed as the CLI frames them.

The properties this file exists for, each from ``mesh-session-mobility.md``:

* ``test_a_peer_can_create_...`` — R8/§5.3: a peer creates a session ON another
  device, the id is minted by the owner, the stamp names the owner, and the first
  prompt is admitted there.
* ``test_the_federated_listing_...`` — R6/§9.2: one list, each row carrying its
  locality and its peer block.
* ``test_the_stream_...`` — §3.2/R-IF-1: a viewer connection becomes a
  pass-through, so the frames it sees are the runtime's own.
* ``test_a_forwarded_op_for_a_session_this_device_does_not_own_is_refused`` —
  INV-1/§7.2: two devices never both hold a live claim on one id.
* ``test_quitting_the_viewer_leaves_the_peer_runtime_running`` — the headline
  guarantee: closing the local viewer must not stop a remote runtime.
"""

from __future__ import annotations

import asyncio
import json
import socket
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import dial as session_dial
from local_operator.network import projection, relay, store
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _init_network,
    _pair,
    devices,
)
from tests.unit.session.runtime.test_server import FakeHandle

SESSION = "9f3ac1e0b7d2"

Devices = tuple[relay.RelayServer, relay.RelayServer, str, int]


@pytest.fixture()
def peer_pair(request: pytest.FixtureRequest) -> Devices:
    """The shared two-relay fixture, under a name my tests can take.

    Requested by NAME rather than imported into a test signature: pytest
    registers ``test_relay_e2e``'s fixture here by importing it, and a test
    parameter with the same name as that module-level import is a redefinition
    flake8 refuses (F811) — so every test below takes this alias instead, and
    the fixture itself is the shared one, not a second copy of it.
    """
    pair: Devices = request.getfixturevalue("devices")
    return pair


def _listen(server: relay.RelayServer) -> tuple[str, int]:
    """Give a device a listener, so the OTHER device can dial it.

    Called AFTER pairing, deliberately: ``bind_control`` publishes this device's
    relay record, and a join that runs with one already published takes a
    different path than the one the shared fixture exercises.
    """
    host, port = server.bind()
    server.bind_control()
    server.start()
    return str(host), int(port)


def _viewer(server: relay.RelayServer) -> None:
    """Run this device's own relay: the viewer dials ITS control socket.

    ``bind_control`` publishes the peer record the viewer's client reads to find
    the port and key, and ``start`` serves it — the same two steps
    ``lop network start`` performs. A device with no relay running cannot be a
    viewer at all, which is the refusal ``RemoteOwner.engage`` reports.
    """
    server.bind_control()
    server.start()


def _dial_to(server: relay.RelayServer, record: Any, host: str, port: int) -> relay.PeerLink:
    link, reason = server.dial(record.network_id, host=f"{host}:{port}", epoch=record.epoch)
    assert link is not None, reason
    return link


def _own_locally(root: Path, session_id: str, device_id: str) -> None:
    """Make ``root`` the owner of a seeded session, without any peer involved."""
    from local_operator.session.placement import (
        MeshStamp,
        SessionPlacement,
        write_stamp,
    )

    _seed(root, session_id)
    write_stamp(
        root,
        MeshStamp(
            session_id=session_id,
            network_id="n_local",
            home_device=device_id,
            placement=SessionPlacement(
                mode="peer", network_id="n_local", home_device=device_id, stamp_revision=1
            ),
            origin={"kind": "user", "source_device": "", "source_session_id": ""},
        ),
    )


# ---------------------------------------------------------------------------
# The owning side's runtime: a real RuntimeServer over a real socket
# ---------------------------------------------------------------------------


class _Handle(FakeHandle):
    """``FakeHandle`` for a session id this test mints, not its hardcoded one."""

    def __init__(self, session_id: str) -> None:
        super().__init__()
        self._projection = replace(self._projection, session_id=session_id)
        from local_operator.session.frontend_state import FrontendStateStore

        self._frontend = FrontendStateStore(
            self._frontend.state.model_copy(update={"session_id": session_id})
        )

    def prompts(self) -> list[str]:
        return [str(call[1][0]) for call in self.calls if call[0] == "prompt"]


class _Served:
    """A runtime this test started, and the record it published."""

    def __init__(self, handle: _Handle, runtime: Any) -> None:
        self.handle = handle
        self.runtime = runtime

    def stop(self) -> None:
        try:
            self.runtime.close()
        except Exception:  # noqa: BLE001 — teardown must not mask a failure
            pass


def _serve(monkeypatch: pytest.MonkeyPatch, root: Path) -> dict[str, _Served]:
    """Make ``engage_runtime`` on ``root`` start a real runtime, and record it.

    The relay calls ``launch.engage_runtime`` inside its own thread, exactly as
    the CLI path does; this stands in for the spawned PROCESS with an in-process
    ``RuntimeServer`` publishing a REAL registry record — so everything
    downstream (the relay's dial, the record lookup, the welcome identity check)
    is the production code path.
    """
    served: dict[str, _Served] = {}
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))

    async def engage(session_id: str, cwd: str, work: Any, **kwargs: Any) -> Any:
        # The runtime publishes into the ambient config root, so the fake pins it
        # here rather than trusting whatever the test last set: the relay calls
        # this from its OWN thread, where a stray ambient value would publish the
        # record into the wrong device's store and the poll below would wait for
        # something that can never arrive.
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
        if session_id not in served:
            from local_operator.session.runtime.server import RuntimeServer

            handle = _Handle(session_id)
            runtime = RuntimeServer(handle, kind="tui")
            runtime.start()
            served[session_id] = _Served(handle, runtime)
        from local_operator.session.runtime import registry

        async with asyncio.timeout(20):
            while not any(
                record.session_id == session_id and status == "live"
                for record, status in registry.scan(root)
            ):
                await asyncio.sleep(0.01)
        return None

    monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", engage)
    return served


def _seed(root: Path, session_id: str) -> None:
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text("", encoding="utf-8")


def _warm(root: Path, session_id: str) -> None:
    """Start the owning runtime for a seeded session, the relay's own way."""
    from local_operator.session.runtime.launch import WarmErrand, engage_runtime

    asyncio.run(engage_runtime(session_id, str(root), WarmErrand(), config_dir=root))


def _call(root: Path, op: str, **fields: Any) -> dict[str, Any]:
    """One local op against this device's relay, as the CLI issues it."""
    record = store.find_own_relay(root)
    assert record is not None, "no relay is running for this device"
    reply = relay.control_request(record, op, **fields)
    assert reply is not None, f"no reply to {op}"
    return reply


def _dial(server: relay.RelayServer, record: Any, host: str, port: int) -> relay.PeerLink:
    link, reason = server.dial(record.network_id, host=f"{host}:{port}", epoch=record.epoch)
    assert link is not None, reason
    return link


def _stop_all(served: dict[str, _Served]) -> None:
    for entry in served.values():
        entry.stop()


# ---------------------------------------------------------------------------
# R8 — create, engage, prompt, and list
# ---------------------------------------------------------------------------


def test_a_peer_can_create_a_session_on_this_device_and_prompt_it(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    server_a, server_b, _host_a, _port_a = peer_pair
    record, _host, _port = _pair(peer_pair, monkeypatch, role="drive")
    host_b, port_b = _listen(server_b)
    served = _serve(monkeypatch, server_b.root)
    try:
        link = _dial_to(server_a, record, host_b, port_b)
        reply = link.request(
            {
                "op": "net_session_create",
                "req": 7,
                "locality": "remote",
                "cwd": str(server_b.root),
                "name": "mesh design",
                "prompt": "port the parser",
            }
        )
        assert reply is not None and reply["op"] == "ack", reply
        detail = reply["detail"]
        session_id = detail["session_id"]
        assert session_id, "the owner must mint the id"
        assert detail["admitted"] is True, detail

        # THE OWNER MINTED IT, AND THE STAMP SAYS SO (§5.3 step 3).
        from local_operator.session.placement import read_stamp

        stamp = read_stamp(server_b.root, session_id)
        assert stamp is not None, "a created session must carry its ownership stamp"
        assert stamp.home_device == server_b.identity.device_id
        assert stamp.placement.mode == "peer"
        assert stamp.origin["source_device"] == server_a.identity.device_id

        # THE PROMPT REACHED THE RUNTIME, not merely the relay.
        assert served[session_id].handle.prompts() == ["port the parser"]

        # AND THE RELAY LEFT NO LEASE BEHIND. ``claim_session`` writes the
        # CLAIMING process's pid into the session's live marker, and for this
        # slice the claimer is the relay — whose pid is alive, so the runtime it
        # then spawns reads the marker as "somebody else is constructing this"
        # and waits out the whole engage deadline. The create must therefore
        # release the claim it took for the mkdir window; this is the assertion
        # that would have caught that, and it is why the release exists.
        from local_operator.session.retention import LIVE_MARKER_NAME

        assert not (
            server_b.root / "sessions" / session_id / LIVE_MARKER_NAME
        ).exists(), "the peer's relay left its own pid in the new session's lease"

        # The creator sees it, as a REMOTE row filed under B (§9.2).
        rows = _call(server_a.root, "peer_session_rows")["detail"]
        remote = [row for row in rows["sessions"] if row["session_id"] == session_id]
        assert remote, rows
        assert remote[0]["locality"] == "remote"
        assert remote[0]["peer"]["device_id"] == server_b.identity.device_id
        assert rows["peers"][server_b.identity.device_id]["reachable"] is True

        # And the owning device sees it as its OWN, with its own stamp.
        own = [row for row in server_b.local_session_rows() if row["session_id"] == session_id]
        assert own and own[0]["placement"]["home_device"] == server_b.identity.device_id
        # The link stays open to the end of the test: `_fan_out_catalog` reaches a
        # peer over a live link (or the member's own endpoints, which the shared
        # fixture advertises as the DIALER's address — so a closed link is one this
        # device cannot necessarily re-open, and the listing is not what closes it).
        link.close("test")
    finally:
        _stop_all(served)


def test_a_create_frame_that_names_a_session_id_is_refused(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The device that will OWN the session mints its id (§5.3, §6.2)."""
    server_a, server_b, _host_a, _port_a = peer_pair
    record, _host, _port = _pair(peer_pair, monkeypatch, role="drive")
    host_b, port_b = _listen(server_b)
    link = _dial_to(server_a, record, host_b, port_b)
    reply = link.request(
        {
            "op": "net_session_create",
            "req": 8,
            "locality": "remote",
            "session_id": "already-mine",
            "prompt": "hi",
        }
    )
    assert reply is not None and reply["op"] == "error"
    assert "mints its id" in str(reply["message"])
    link.close("test")


def test_the_federated_listing_carries_locality_and_peer_for_both_halves(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R6/§9.2: one list, and no surface infers remoteness from an id's shape."""
    server_a, server_b, _host_a, _port_a = peer_pair
    record, _host, _port = _pair(peer_pair, monkeypatch, role="drive")
    host_b, port_b = _listen(server_b)
    _seed(server_a.root, "local-one")
    _seed(server_b.root, SESSION)
    served = _serve(monkeypatch, server_b.root)
    try:
        _warm(server_b.root, SESSION)
        _dial_to(server_a, record, host_b, port_b)
        payload = _call(server_a.root, "peer_session_rows")["detail"]
        by_id = {row["session_id"]: row for row in payload["sessions"]}
        assert by_id["local-one"]["locality"] == "local"
        assert by_id["local-one"]["peer"] is None
        assert by_id[SESSION]["locality"] == "remote"
        assert by_id[SESSION]["peer"]["name"] == server_b.identity.name
        # And a COLD session on a peer is still a row: a synchronous engage is
        # what makes an idle remote session usable, and it needs a row to name.
        empty = [row for row in payload["sessions"] if row["session_id"] == "no-such"]
        assert empty == []
    finally:
        _stop_all(served)


def test_a_cold_session_on_a_peer_still_lists_and_can_be_engaged(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§1.2: an idle session on a peer must still list as remote, and be warmable."""
    server_a, server_b, _host_a, _port_a = peer_pair
    record, _host, _port = _pair(peer_pair, monkeypatch, role="drive")
    host_b, port_b = _listen(server_b)
    _seed(server_b.root, SESSION)
    served = _serve(monkeypatch, server_b.root)
    try:
        _dial_to(server_a, record, host_b, port_b)
        payload = _call(server_a.root, "peer_session_rows")["detail"]
        row = next(item for item in payload["sessions"] if item["session_id"] == SESSION)
        assert row["state"] == "stored", row
        assert row["locality"] == "remote"

        engaged = _call(
            server_a.root,
            "peer_session_engage",
            peer=server_b.identity.device_id,
            session_id=SESSION,
            cwd=str(server_b.root),
        )
        assert engaged["op"] == "ack", engaged
        assert engaged["detail"]["engaged"] is True
        live = [
            item
            for item in server_b.local_session_rows()
            if item["session_id"] == SESSION and item["state"] == "live"
        ]
        assert live, "engage did not warm the session on its owner"
    finally:
        _stop_all(served)


def test_a_session_lifecycle_op_is_refused_by_name(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§8 routes delete/archive to the owner's implementation; this build has none.

    Refused WITH the module that owns it rather than half-implemented: a second
    ``rmtree`` of a session directory is what
    ``tests/unit/session/test_no_session_deletion.py`` exists to prevent.
    """
    server_a, server_b, _host_a, _port_a = peer_pair
    record, _host, _port = _pair(peer_pair, monkeypatch, role="admin")
    host_b, port_b = _listen(server_b)
    _seed(server_b.root, SESSION)
    link = _dial_to(server_a, record, host_b, port_b)
    reply = link.request(
        {
            "op": "net_session_lifecycle",
            "req": 9,
            "locality": "remote",
            "action": "delete",
            "session_id": SESSION,
            "confirmed": True,
        }
    )
    assert reply is not None and reply["op"] == "error"
    assert "session/archived.py" in str(reply["message"])
    link.close("test")


def test_an_op_the_member_is_not_granted_is_refused_by_the_chokepoint(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read-only member cannot create a session: capability ``prompt``."""
    server_a, server_b, host_a, port_a = peer_pair
    record, _host, _port = _pair(peer_pair, monkeypatch, role="read")
    _serve(monkeypatch, server_a.root)
    link = _dial_to(server_b, record, host_a, port_a)
    reply = link.request(
        {
            "op": "net_session_create",
            "req": 12,
            "locality": "remote",
            "cwd": str(server_a.root),
            "prompt": "let me in",
        }
    )
    assert reply is not None and reply["op"] == "error", reply
    assert "prompt" in str(reply["message"])
    link.close("test")


# ---------------------------------------------------------------------------
# INV-1 — one writer, always
# ---------------------------------------------------------------------------


def test_a_forwarded_op_for_a_session_this_device_does_not_own_is_refused(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The negative half of the ownership guard (§1.1, §7.2).

    B holds the session; A is asked to act on it. A must refuse, because the
    alternative is two devices both believing they may turn a turn on one id. The
    id is B's own session, owned there and nowhere else.
    """
    server_a, server_b, host_a, port_a = peer_pair
    record, _host, _port = _pair(peer_pair, monkeypatch, role="drive")
    _own_locally(server_b.root, SESSION, server_b.identity.device_id)
    try:
        # B holds the session; B asks A (over B's own link) to act on it, and A
        # must refuse: this is "two devices never both hold a claim on one id"
        # seen from the side that is being asked to overstep.
        link = _dial_to(server_b, record, host_a, port_a)
        reply = link.request(
            {
                "op": "net_forward",
                "req": 11,
                "locality": "remote",
                "frame": {"op": "prompt", "req": 1, "session_id": SESSION, "text": "hi"},
            }
        )
        assert reply is not None and reply["op"] == "error", reply
        assert "does not live on this device" in str(reply["message"])
        link.close("test")
    finally:
        pass


def test_the_owner_answers_for_a_session_it_holds_and_refuses_one_it_does_not(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``net_session_facts`` is answered from ownership, not from liveness.

    A COLD session is still owned (that is what makes ``net_session_engage`` able
    to warm it), and a session this device has never heard of is answered "not
    owned" rather than confused with one that is merely stopped — two different
    answers a resolver must be able to tell apart (§3.4).
    """
    server_a, server_b, _host_a, _port_a = peer_pair
    record, _host, _port = _pair(peer_pair, monkeypatch, role="drive")
    host_b, port_b = _listen(server_b)
    _seed(server_b.root, SESSION)
    _dial_to(server_a, record, host_b, port_b)
    cold = _call(
        server_a.root,
        "peer_session_facts",
        peer=server_b.identity.device_id,
        session_id=SESSION,
    )["detail"]
    assert cold["owned"] is True and cold["published"] is False, cold
    unknown = _call(
        server_a.root,
        "peer_session_facts",
        peer=server_b.identity.device_id,
        session_id="nobody-has-this",
    )["detail"]
    assert unknown == {"owned": False, "published": False, "pid": None, "record": None}


def test_resolve_owner_answers_the_local_store_first_and_never_guesses_local(
    tmp_path: Path,
) -> None:
    """§2.1's order, including the arm that must NOT be ``local``."""
    from local_operator.session.placement import (
        MeshStamp,
        SessionPlacement,
        write_stamp,
    )

    # 4. Nothing at all → unknown, with the sentence §2.1 writes out.
    answer = projection.resolve_owner("nope", config_dir=tmp_path)
    assert answer.kind == "unknown"
    assert answer.reason == "No device in this network holds that conversation."

    # 1. A directory and no stamp → local (exactly today's behaviour).
    _seed(tmp_path, SESSION)
    assert projection.resolve_owner(SESSION, config_dir=tmp_path).kind == "local"

    # 1b. A directory stamped to ANOTHER device → NOT local. This is the arm that
    # keeps a handoff's leftovers from resurrecting a moved session here.
    write_stamp(
        tmp_path,
        MeshStamp(
            session_id=SESSION,
            network_id="n_1",
            home_device="dev_somebody_else",
            placement=SessionPlacement(mode="peer", network_id="n_1", home_device="dev_x"),
        ),
    )
    assert projection.resolve_owner(SESSION, config_dir=tmp_path).kind == "unknown"

    # 2. A tombstone naming a device we do not know → unknown, and it says why.
    projection.write_tombstone(
        SESSION, device_id="dev_gone", device_name="build-box", config_dir=tmp_path
    )
    answer = projection.resolve_owner(SESSION, config_dir=tmp_path)
    assert answer.kind == "unknown"
    assert "no longer in the network" in answer.reason


# ---------------------------------------------------------------------------
# Zero peers — the regression that must be provable
# ---------------------------------------------------------------------------


def test_the_local_path_runs_no_network_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R16 topology 0, asserted with a SPY rather than by reading comments.

    No network is configured and no peer is named, so the resolver and the facade
    must reach nothing in the mesh package. The spy is on the relay's one entry
    point from a local caller — if any path below dials the relay, this fails with
    the call in hand.
    """
    from local_operator.session.attached import AttachedSession
    from local_operator.session.owner import LocalOwner

    calls: list[str] = []

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append("relay.control_request")
        return None

    monkeypatch.setattr(relay, "control_request", spy)
    monkeypatch.setattr(
        projection, "_relay_call", lambda *a, **k: calls.append("projection._relay_call")
    )
    _seed(tmp_path, SESSION)

    answer = projection.resolve_owner(SESSION, config_dir=tmp_path, catalog=None)
    assert answer.kind == "local"
    assert calls == [], f"the local resolver touched the network: {calls}"

    async def _never() -> Any:  # pragma: no cover - never awaited here
        raise AssertionError("no takeover")

    session = AttachedSession(
        config_dir=tmp_path, session_id=SESSION, takeover_factory=_never, surface="terminal"
    )
    assert isinstance(session._owner, LocalOwner)
    assert session._owner.placement.mode == "local"
    assert session.runtime_locality == "this-machine"
    assert session._can_go_cold is False
    assert session._owner.locate() == (None, None)
    assert calls == [], f"the local facade touched the network: {calls}"


def test_a_remote_owner_refuses_closed_when_there_is_no_relay(tmp_path: Path) -> None:
    """A refusal carries a machine code and a sentence, and refuses CLOSED."""
    row = projection.PeerRow(session_id=SESSION, device_id="dev_peer", device_name="build-box")
    owner = projection.remote_owner_for(SESSION, config_dir=tmp_path, row=row)
    with pytest.raises(projection.ProjectionRefusal) as caught:
        asyncio.run(owner.engage(cwd="", warm=None))
    assert caught.value.code == projection.CODE_RELAY_UNAVAILABLE
    assert owner.placement.mode == "peer"
    assert owner.placement.home_device == "dev_peer"
    assert owner.seed().device_name == "build-box"


# ---------------------------------------------------------------------------
# The stream — pass-through, refusals, and what a quit does
# ---------------------------------------------------------------------------


class _StreamClient:
    """A viewer connection: LOCAL relay control auth, then session frames."""

    def __init__(self, root: Path) -> None:
        record = store.find_own_relay(root)
        assert record is not None
        self.sock = socket.create_connection(("127.0.0.1", record.control_port), timeout=10)
        self.reader = session_dial.LineReader(self.sock)
        self.sock.sendall(
            (json.dumps({"key": record.control_key, "client": "cli"}) + "\n").encode()
        )

    def send(self, frame: dict[str, Any]) -> None:
        self.sock.sendall((json.dumps(frame) + "\n").encode())

    def recv(self, timeout: float = 20.0) -> dict[str, Any] | None:
        return self.reader.read_frame(timeout)

    def open_stream(self, peer: str, session_id: str, **auth: Any) -> dict[str, Any]:
        self.send(
            {
                "op": "stream_open",
                "req": 1,
                "peer": peer,
                "session_id": session_id,
                "auth": auth or {"frontend_state": True},
            }
        )
        opened = self.recv()
        assert opened is not None, "no answer to stream_open"
        return opened

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


def _wait_for(predicate: Any, timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def test_the_stream_is_a_pass_through_and_quitting_it_leaves_the_peer_running(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§3.2 + the quit guarantee, on one real socket each side.

    The viewer dials ITS OWN relay, opens a stream to the session on the peer, and
    from then on speaks the runtime's own vocabulary: the welcome arrives
    untranslated, a prompt is acked, and the peer's runtime records it. Then the
    viewer socket is closed the way a quitting TUI closes it — and the peer's
    runtime must still be alive, still listed, and still reachable by a new viewer.
    """
    server_a, server_b, host_a, port_a = peer_pair
    record, _host, _port = _pair(peer_pair, monkeypatch, role="drive")
    # The SESSION lives on A (the owner); the VIEWER is B, whose relay dials A.
    _seed(server_a.root, SESSION)
    served = _serve(monkeypatch, server_a.root)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_b.root))
    try:
        _warm(server_a.root, SESSION)
        _viewer(server_b)
        assert _dial_to(server_b, record, host_a, port_a) is not None

        client = _StreamClient(server_b.root)
        opened = client.open_stream(
            server_a.identity.device_id,
            SESSION,
            events=True,
            frontend_state=True,
            surface="terminal",
        )
        assert opened["op"] == "ack", opened
        assert opened["detail"]["stream"].startswith("s")

        # The owner's own welcome, forwarded with no translation.
        welcome = client.recv()
        assert welcome is not None and welcome["op"] == "projection", welcome
        assert welcome["data"]["session_id"] == SESSION

        client.send(
            {
                "op": "prompt",
                "req": 2,
                "command_id": str(uuid.uuid4()),
                "text": "hello from B",
            }
        )
        acked = None
        for _ in range(40):
            frame = client.recv()
            assert frame is not None, "the stream went quiet before the prompt was acked"
            if frame.get("req") == 2:
                acked = frame
                break
        assert acked is not None and acked["op"] == "ack", acked
        assert _wait_for(lambda: served[SESSION].handle.prompts() == ["hello from B"]), served[
            SESSION
        ].handle.calls

        # QUIT: the viewer socket goes away, exactly as a quitting TUI leaves it.
        client.close()
        assert _wait_for(
            lambda: not any(stream.session_id == SESSION for stream in server_b._streams.values())
        ), "the opening relay did not notice the viewer leaving"

        # A'S RUNTIME SURVIVED: its record is still live, its handle untouched,
        # and a SECOND viewer can reach the same session.
        from local_operator.session.runtime import registry

        calls_after_quit = len(served[SESSION].handle.calls)
        assert _wait_for(
            lambda: bool(
                [
                    rec
                    for rec, status in registry.scan(server_a.root)
                    if status == "live" and rec.session_id == SESSION
                ]
            )
        ), "closing a remote viewer stopped the peer's runtime"

        client2 = _StreamClient(server_b.root)
        reopened = client2.open_stream(server_a.identity.device_id, SESSION)
        assert reopened["op"] == "ack", reopened
        assert client2.recv() is not None, "the session was no longer reachable"
        assert len(served[SESSION].handle.calls) == calls_after_quit
        client2.close()
    finally:
        _stop_all(served)


def test_a_read_only_member_may_open_a_stream_and_may_not_prompt_through_it(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``net_stream`` opens on ``view``; every frame down it needs its own grant.

    This is what keeps a carrier from being an authorisation bypass: the read-only
    member gets its transcript (the welcome arrives) and its prompt is refused AT
    THE LINK, with nothing reaching the runtime's handle.
    """
    server_a, server_b, host_a, port_a = peer_pair
    record, _host, _port = _pair(peer_pair, monkeypatch, role="read")
    _seed(server_a.root, SESSION)
    served = _serve(monkeypatch, server_a.root)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_b.root))
    try:
        _warm(server_a.root, SESSION)
        _viewer(server_b)
        _dial_to(server_b, record, host_a, port_a)
        client = _StreamClient(server_b.root)
        opened = client.open_stream(server_a.identity.device_id, SESSION)
        assert opened["op"] == "ack", opened
        assert (client.recv() or {}).get("op") == "projection"

        before = list(served[SESSION].handle.calls)
        client.send(
            {"op": "prompt", "req": 3, "command_id": str(uuid.uuid4()), "text": "let me in"}
        )
        refusal = None
        for _ in range(40):
            frame = client.recv()
            if frame is None:
                break
            if frame.get("op") == "error":
                refusal = frame
                break
        assert refusal is not None, "a read-only member's prompt was not refused"
        assert "prompt" in str(refusal["message"])
        assert (
            served[SESSION].handle.calls == before
        ), "the refused frame reached the runtime anyway"
        client.close()
    finally:
        _stop_all(served)


# ---------------------------------------------------------------------------
# Q-R5-2 — `--force` is the owner's own force, and `mode` is how it travels
# ---------------------------------------------------------------------------


def test_the_forwarded_stop_mode_is_the_owners_own_force(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mesh verb's `--stop --force` must BE the owner's `lop stop --force`.

    WHY THIS TEST EXISTS. The owner's ladder declines to signal a target that
    reports a turn in flight, and the refusal it composes NAMES ``--force`` as the
    way past it. That sentence is painted by whichever front end asked, so the
    mesh viewer has to accept the flag it offers — a surface offering an action it
    cannot accept is the defect UX round 2 called U7, and here the machine holding
    the session may be one the operator cannot sit down at (QA round 5, Q-R5-2).

    WHAT IS TESTED WHERE. The flag's MEANING is one place and one test —
    ``control.stop_session(force=True)``, "signal the target the plain stop left
    alone" (``tests/unit/session/runtime/test_control.py``). This test owns the ONE
    hop between the wire and that flag: ``mode`` is the ladder's own spelling
    (``relay._op_session_stop``), the frame travels a REAL paired link, and the two
    ends of the pair are asserted so the mapping cannot silently become a no-op —
    ``graceful`` skips the busy target, ``immediate`` signals it. The signal seams
    are the ones the ladder's own test uses, because no unit test may send a real
    signal (see that module's header).
    """
    import signal as signal_mod

    from local_operator.session.runtime import control as control_mod
    from tests.unit.session.runtime.test_control import _bare_record

    server_a, server_b, _host_a, _port_a = peer_pair
    record, _host, _port = _pair(peer_pair, monkeypatch, role="drive")
    host_b, port_b = _listen(server_b)
    link = _dial_to(server_a, record, host_b, port_b)

    sent: list[tuple[int, int]] = []
    #: A record for a runtime that is busy and whose socket does NOT answer: the
    #: one shape that reaches the busy skip, which is the skip `--force` bypasses.
    target = _bare_record(busy=True, control_port=1)
    # The session must LIVE on B before its relay will act on it: the authoriser
    # refuses a session-scoped op for an id this device does not own (§7.2/INV-1).
    _own_locally(server_b.root, target.session_id, server_b.identity.device_id)
    monkeypatch.setattr(server_b, "_session_record", lambda _sid: target)
    monkeypatch.setattr(control_mod, "_identity_by_record", lambda _r: (True, ""))
    monkeypatch.setattr(control_mod.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    monkeypatch.setattr(control_mod.registry, "pid_alive", lambda _pid, **_: not sent)
    plain = link.request(
        {
            "op": "net_session_stop",
            "req": 11,
            "locality": "remote",
            "session_id": target.session_id,
            "mode": "graceful",
        }
    )
    assert plain is not None and plain["op"] == "ack", plain
    assert plain["detail"]["outcome"] == "skipped", plain["detail"]
    assert plain["detail"]["rung"] == "busy"
    assert sent == [], "a plain stop must not signal a target mid-turn"

    forced = link.request(
        {
            "op": "net_session_stop",
            "req": 12,
            "locality": "remote",
            "session_id": target.session_id,
            "mode": "immediate",
        }
    )
    assert forced is not None and forced["op"] == "ack", forced
    assert [sig for _pid, sig in sent] == [signal_mod.SIGTERM]
    assert forced["detail"]["outcome"] == "stopped", forced["detail"]
    assert forced["detail"]["rung"] == "sigterm"
    link.close("test")
