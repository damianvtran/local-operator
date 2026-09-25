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

import argparse
import asyncio
import json
import os
import socket
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import dial as session_dial
from local_operator.network import projection, relay, store
from local_operator.session.cleanup import mark_store
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
# A peer's runtime with a REAL session in it
# ---------------------------------------------------------------------------
#
# ``_serve`` above stands in for the spawned runtime with a ``FakeHandle``, and
# that is the right rig for everything the relay's own transport does: the
# create frame, the engage, the prompt hand-off, the listing. It is blind to the
# one thing two rounds filed against (QA round 13, Q13-1 / UX round 4, U24)
# because there is no SESSION in it — the conversation name a running session
# keeps is decided by the session's own naming errand, on its first real turn,
# and a fake handle has neither. So the turn has to be real for the claim to be
# testable at all; a rig that fakes it is what let the previous head ship a pin
# that passed while the name was still being replaced.
#
# ``hosting: test`` is the production shape for a real turn with no provider:
# ``providers.registry`` maps provider id ``test`` (wire ``mock``) to
# ``MockClient``, ``tests.e2e.harness.build_session``'s ``TEST_MODEL`` is that
# provider, and the mock's reply is the same deterministic string QA's and UX's
# meshes produced — which is why the generated title in the failing direction is
# literally ``Hello from the mock provider``.


class _RealServed:
    """A real ``Session`` on the peer, on its own loop, served by a real runtime."""

    def __init__(
        self,
        session: Any,
        handle: Any,
        runtime: Any,
        loop: asyncio.AbstractEventLoop,
        thread: threading.Thread,
    ) -> None:
        self.session = session
        self.handle = handle
        self.runtime = runtime
        self._loop = loop
        self._thread = thread

    def on_session_loop(self, coro: Any, *, timeout: float = 30.0) -> Any:
        """Run ``coro`` on the session's own loop and return its result."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    def transcript_entries(self) -> list[dict[str, Any]]:
        """Every journalled line of this session's transcript, as parsed dicts."""
        path = Path(self.session.transcript.path)
        if not path.exists():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def naming_state(self) -> dict[str, Any]:
        """The newest ``conversation_name`` entry's payload — the transcript's OWN
        answer to "what is this conversation called, and did the user name it".

        Read from disk rather than from the live holder for the reason the two
        rounds' findings are about: the holder is memory, the journal is the
        record every later reader (a resume, a fork, the title backfill) trusts.
        """
        seen = [
            (entry.get("payload") or {}).get("details") or {}
            for entry in self.transcript_entries()
            if (entry.get("payload") or {}).get("custom_type") == "conversation_name"
        ]
        return seen[-1] if seen else {}

    def wait_for_turn(self, *, timeout: float = 60.0) -> None:
        """Block until the admitted prompt has RUN, then until naming has settled.

        Two waits, and the second is not decoration: on the unfixed tree the
        auto-namer runs concurrently with the turn (``_maybe_name_conversation``
        dispatches it at admission, ``serving.ServingSessionHandle.prompt``), so
        a test that read the sidecar the instant the turn landed could pass on
        the bug by winning a race. Draining the handle's naming tasks after the
        turn makes the failing direction deterministic rather than a coin flip —
        and a test that cannot fail on the code it was written against is not a
        pin at all.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if any(
                (entry.get("payload") or {}).get("role") == "assistant"
                for entry in self.transcript_entries()
            ):
                break
            time.sleep(0.05)
        else:
            raise AssertionError(
                "the peer's session never finished its turn: nothing in the transcript "
                "carries an assistant row, so this test would be asserting about a turn "
                "that did not happen"
            )
        while time.monotonic() < deadline and self.handle._background_tasks:
            time.sleep(0.05)

    def stop(self) -> None:
        """Close the runtime, dispose the session, and reap the loop it ran on."""
        try:
            self.on_session_loop(self.runtime.aclose(), timeout=20.0)
        finally:
            try:
                self.on_session_loop(self.session.dispose(), timeout=20.0)
            finally:
                self._loop.call_soon_threadsafe(self._loop.stop)
                self._thread.join(timeout=5.0)
                self._loop.close()


def _start_real_session(root: Path, session_id: str, cwd: str) -> _RealServed:
    """Boot one real session + runtime on a dedicated loop, and hand it back.

    A LOOP OF ITS OWN, not the caller's: ``ServingSessionHandle`` publishes the
    loop it was built on (``session_loop``) and the runtime hops to it for every
    handle call, so the loop has to outlive the ``asyncio.run`` the relay wraps
    its engage in — exactly as the spawned process's loop does in production.
    """
    from local_operator.providers.clients import MockClient
    from local_operator.session.runtime.server import RuntimeServer
    from local_operator.session.runtime.serving import ServingSessionHandle
    from tests.e2e.harness import build_session

    loop = asyncio.new_event_loop()
    booted: list[Any] = []
    buried: list[BaseException] = []
    gate = threading.Event()

    def _boot() -> None:
        asyncio.set_event_loop(loop)
        try:
            # BUILT ON THE RUNNING LOOP, exactly as production builds it
            # (``process.amain``, the TUI's adoption worker): a session
            # constructed OUTSIDE a loop cannot start its own background writes
            # (``_spawn_conversation_name_write``
            # catches the missing loop and defers to the dispose flush), and the
            # journal write a booting session owes the transcript is one of
            # them. A rig that built it on the thread would be testing a
            # session shape no `lop` ever constructs.
            async def _build() -> tuple[Any, Any, Any]:
                session = build_session(
                    root / "sessions" / session_id, MockClient().stream, cwd=Path(cwd)
                )
                handle = ServingSessionHandle(session, loop, cwd=cwd)
                runtime = RuntimeServer(handle, kind="tui")
                await runtime.start_in_process()
                return session, handle, runtime

            booted.extend(loop.run_until_complete(_build()))
            gate.set()
            loop.run_forever()
        except BaseException as exc:  # noqa: BLE001 — re-raised on the caller's thread
            buried.append(exc)
            gate.set()

    thread = threading.Thread(target=_boot, name=f"mesh-real-session-{session_id}", daemon=True)
    thread.start()
    if not gate.wait(timeout=30.0):
        raise AssertionError(f"the runtime for {session_id} never booted")
    if buried:
        raise buried[0]
    session, handle, runtime = booted
    return _RealServed(session, handle, runtime, loop, thread)


def _serve_real_sessions(monkeypatch: pytest.MonkeyPatch, root: Path) -> dict[str, _RealServed]:
    """``_serve``'s stand-in for the runtime start, serving REAL sessions.

    The relay's ``_engage_locally`` is only the trigger: it calls
    ``launch.engage_runtime`` and every interesting thing happens in whatever
    that returns. In production it spawns a process; here it boots a real
    session in this one, which is the same contract the fake rig substitutes a
    ``FakeHandle`` for.
    """
    served: dict[str, _RealServed] = {}
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))

    async def engage(session_id: str, cwd: str, *_args: Any, **_kwargs: Any) -> None:
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
        if session_id not in served:
            served[session_id] = _start_real_session(root, session_id, cwd)

    monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", engage)
    return served


def _stop_real(served: dict[str, _RealServed]) -> None:
    for entry in served.values():
        entry.stop()


def _cli_listing_line(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    peer_token: str,
    capsys: pytest.CaptureFixture[str],
    session_id: str,
) -> str:
    """The one line `lop network sessions --peer <token>` prints for a session.

    THE COMMAND'S OWN FUNCTION, not a re-derivation of its logic: QA read the
    empty name column on this surface, so this drives ``cli._cmd_sessions`` —
    the handler argv dispatches to — over the real relay control socket with the
    ambient config dir pointed at the device that owns the CLI, exactly as a
    user's shell does it. Called in-process rather than as a subprocess because
    the path under test is the listing, and a second interpreter would only add
    a spawn to time out.
    """
    from local_operator.network import cli as network_cli

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    capsys.readouterr()
    assert (
        network_cli._cmd_sessions(argparse.Namespace(peer=peer_token, all_peers=False, json=False))
        == 0
    )
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith(session_id)]
    assert lines, "`network sessions` listed nothing for the session under test"
    return lines[0]


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


def test_a_promptless_create_is_still_a_row_on_both_devices(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q-R10-3: the bare ``/new remote <peer>`` must leave a listable session.

    The prompted create above works because its turn writes the activity file
    the ordinary catalogue ranks on. The PROMPTLESS create — the first thing a
    user types — has no turn, so the peer's runtime finds no work and no
    viewer, and (on a real device) idle-exits: the session then carried neither
    activity file, which makes it "never worked in" and outside the ranked set
    entirely. The id the user was handed by ``net_session_create`` was therefore
    a row on NEITHER machine, while the receipt they read was a bare boolean.

    So this test drives the whole shape: create with no prompt, stop the
    runtime (the idle-exit), and then require the row from the owner's own
    listing and from the creator's sidebar producer.
    """
    server_a, server_b, _host_a, _port_a = peer_pair
    record, _host, _port = _pair(peer_pair, monkeypatch, role="drive")
    host_b, port_b = _listen(server_b)
    served = _serve(monkeypatch, server_b.root)
    try:
        link = _dial_to(server_a, record, host_b, port_b)
        reply = link.request(
            {
                "op": "net_session_create",
                "req": 9,
                "locality": "remote",
                "cwd": str(server_b.root),
                "name": "bare create",
                "prompt": "",
            }
        )
        assert reply is not None and reply["op"] == "ack", reply
        detail = reply["detail"]
        session_id = detail["session_id"]
        assert session_id
        # ``admitted`` is the relay's word for "the first prompt landed", and
        # with no prompt there is none to land. The CLI's RECEIPT is what may
        # not print this alone (UX round 1, U2); the wire keeps the field.
        assert detail["admitted"] is False
        assert detail["detail"] == ""

        session_dir = server_b.root / "sessions" / session_id
        assert session_dir.is_dir(), "the peer minted no directory at all"
        assert not any(
            (session_dir / name).exists() for name in ("transcript.jsonl", "inbox.jsonl")
        ), (
            "this test is about a session with no activity: if a create now writes one, "
            "the membership rule below is no longer what carries the row"
        )

        # THE IDLE EXIT, as the real device performs it: the runtime goes, its
        # discovery record goes with it, and the only thing left is the
        # directory and its stamp. It waits for the join first: a PROMPTLESS create
        # answers BEFORE its runtime is up (``relay._warm_after_create``), so reading
        # ``served`` straight out of the ack is a race, and the "idle exit" this test
        # simulates starts from a runtime that is actually running.
        assert _wait_for(
            lambda: session_id in served, 60.0
        ), "the peer's runtime never came up for the created session"
        served[session_id].stop()
        assert not [
            row
            for row in server_b.local_session_rows()
            if row["session_id"] == session_id and row["state"] != "stored"
        ]

        own = [row for row in server_b.local_session_rows() if row["session_id"] == session_id]
        assert own, "the owner itself cannot list the session it just minted"
        assert own[0]["state"] == "stored"
        assert own[0]["placement"]["home_device"] == server_b.identity.device_id
        assert own[0]["conversation_name"] == "bare create"

        rows = _call(server_a.root, "peer_session_rows")["detail"]
        remote = [row for row in rows["sessions"] if row["session_id"] == session_id]
        assert remote, rows
        assert remote[0]["locality"] == "remote"

        # ...and through the SIDEBAR's producer, which is the surface the user
        # actually reads (Q-R10-1's whole subject).
        from local_operator.session.peer_rows import clear_cache, peer_session_rows

        clear_cache()
        produced = peer_session_rows(server_a.root)
        assert [row.id for row in produced] == [session_id]
        assert produced[0].owner_device == server_b.identity.device_id
        link.close("test")
    finally:
        _stop_all(served)


def test_a_named_create_writes_the_sidecar_the_product_reads(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The name in a create frame must land in the PRODUCT'S sidecar format.

    QA round 12's major, pinned at the file rather than at a UI: the relay used
    to build ``title.json`` inline under a ``title`` key while every product
    reader (``resume._read_title_sidecar``) reads ``text``, so the name a user
    typed was present on disk and invisible to every surface that lists a
    session. This asserts the three keys the product's writer produces, so the
    writer and the reader cannot drift apart again without a red test — a test
    that only asserted ``stored_session_title`` would pass on the broken writer
    once the reader learned the legacy key, which is exactly the loophole this
    closes.
    """
    server_a, server_b, _host_a, _port_a = peer_pair
    record, _host, _port = _pair(peer_pair, monkeypatch, role="drive")
    host_b, port_b = _listen(server_b)
    served = _serve(monkeypatch, server_b.root)
    try:
        link = _dial_to(server_a, record, host_b, port_b)
        reply = link.request(
            {
                "op": "net_session_create",
                "req": 11,
                "locality": "remote",
                "cwd": str(server_b.root),
                "name": "receipt-naming",
                "prompt": "",
            }
        )
        assert reply is not None and reply["op"] == "ack", reply
        session_id = reply["detail"]["session_id"]
        session_dir = server_b.root / "sessions" / session_id

        payload = json.loads((session_dir / "title.json").read_text(encoding="utf-8"))
        assert payload.get("text") == "receipt-naming", (
            "the sidecar must carry the product's own key: "
            f"resume._read_title_sidecar reads 'text', the payload is {payload}"
        )
        assert payload.get("names") == ["receipt-naming"], payload
        # ``--name`` is a name the USER typed; the product's birth-title
        # analogue (desktop_wakes._birth_title) sets the flag on exactly that
        # condition. Recorded here so the relay's claim is auditable rather
        # than merely written.
        assert payload.get("user_set") is True, payload

        # AND THE PRODUCT READS IT, which is the user-visible half.
        from local_operator.resume import session_name, stored_session_title

        assert stored_session_title(session_dir) == "receipt-naming"
        assert session_name(session_dir) == "receipt-naming"
        link.close("test")
    finally:
        _stop_all(served)


def test_the_name_survives_the_hand_over_to_the_catalogue(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Name, then let the CATALOGUE take the row over — the name is still there.

    QA round 12's other shape: a promptless create is listed off the relay's own
    empty-mint half, which reads the sidecar directly, and the moment the
    session has an activity file the CATALOGUE ranks it and the name comes from
    ``resume.session_name`` instead. On the code QA measured, that hand-over is
    WHERE THE NAME VANISHED: the sidecar held ``title``, the catalogue read
    ``text``, and the row painted ``Untitled conversation``. Both halves and
    BOTH devices are asserted here, because the user reads the creator's
    listing, not the owner's.

    WHAT THIS DOES NOT COVER, stated because the previous head claimed it did
    (UX round 4, U24): NO TURN RUNS HERE. The ``_serve`` rig's runtime is a
    ``FakeHandle``, so there is no session and no auto-namer, and the transcript
    this test writes is empty text. It pins the sidecar's SPELLING and the
    catalogue hand-over, and nothing about what a live session does with the
    name. That claim is
    ``test_a_named_create_keeps_its_name_through_a_real_turn``'s job, on a rig
    with a real session in it.
    """
    server_a, server_b, _host_a, _port_a = peer_pair
    record, _host, _port = _pair(peer_pair, monkeypatch, role="drive")
    host_b, port_b = _listen(server_b)
    served = _serve(monkeypatch, server_b.root)
    try:
        link = _dial_to(server_a, record, host_b, port_b)
        reply = link.request(
            {
                "op": "net_session_create",
                "req": 12,
                "locality": "remote",
                "cwd": str(server_b.root),
                "name": "receipt-naming",
                "prompt": "",
            }
        )
        assert reply is not None and reply["op"] == "ack", reply
        session_id = reply["detail"]["session_id"]

        # THE TURN, in QA's own shape: an activity file appears, so the
        # session leaves the empty-mint half and the catalogue takes over.
        transcript = server_b.root / "sessions" / session_id / "transcript.jsonl"
        transcript.write_text("", encoding="utf-8")
        # THE RUNTIME JOINS IN THE BACKGROUND for a promptless create, so this waits for
        # the one it is about to stop rather than assuming the ack proved it was up.
        assert _wait_for(
            lambda: session_id in served, 60.0
        ), "the peer's runtime never came up for the created session"
        served[session_id].stop()

        own = [row for row in server_b.local_session_rows() if row["session_id"] == session_id]
        assert own, "the owner cannot list the session it minted"
        assert own[0]["conversation_name"] == "receipt-naming", (
            "the owner's own listing lost the name the user typed: "
            f"{own[0]['conversation_name']!r}"
        )

        rows = _call(server_a.root, "peer_session_rows")["detail"]
        remote = [row for row in rows["sessions"] if row["session_id"] == session_id]
        assert remote, rows
        assert remote[0]["conversation_name"] == "receipt-naming", remote[0]

        # ...and through the SIDEBAR's producer, which is what the user reads.
        from local_operator.session.peer_rows import clear_cache, peer_session_rows

        clear_cache()
        produced = [row for row in peer_session_rows(server_a.root) if row.id == session_id]
        assert produced, "the created session is not on the creator's sidebar"
        assert produced[0].name == "receipt-naming", produced[0]
        link.close("test")
    finally:
        _stop_all(served)


class _NamedRemoteCreate:
    """One named create on a REAL peer, with a real session behind it.

    Shared by the two tests below because the RIG is the expensive part — two
    relays, a paired mesh, a session booted on its own loop — and the two
    findings that need it differ only in WHEN they read the surfaces: QA round
    13's Q13-2 is about the window while the runtime is live, Q13-1/U24 about
    what survives the turn. ``stop`` is the only teardown the caller owes.
    """

    def __init__(
        self,
        *,
        server_a: relay.RelayServer,
        server_b: relay.RelayServer,
        served: dict[str, _RealServed],
        session_id: str,
        link: relay.PeerLink,
    ) -> None:
        self.server_a = server_a
        self.server_b = server_b
        self.served = served
        self.session_id = session_id
        self.link = link

    @property
    def owner(self) -> _RealServed:
        return self.served[self.session_id]

    @property
    def session_dir(self) -> Path:
        return self.server_b.root / "sessions" / self.session_id

    @property
    def peer_token(self) -> str:
        """The token ``--peer`` takes: the device id, which no name can shadow."""
        return self.server_b.identity.device_id

    def owner_rows(self) -> list[dict[str, Any]]:
        """The OWNER's listing rows for this session — live half or stored half."""
        return [
            row
            for row in self.server_b.local_session_rows()
            if row["session_id"] == self.session_id
        ]

    def sidebar_rows(self) -> list[Any]:
        """The rows the CREATOR's sidebar producer yields — what the user reads."""
        from local_operator.session.peer_rows import clear_cache, peer_session_rows

        clear_cache()
        return [row for row in peer_session_rows(self.server_a.root) if row.id == self.session_id]

    def federated_rows(self) -> list[dict[str, Any]]:
        """The creator's federated listing, as the relay answers it."""
        payload = _call(self.server_a.root, "peer_session_rows")["detail"]
        return [row for row in payload["sessions"] if row["session_id"] == self.session_id]

    def stop(self) -> None:
        try:
            self.link.close("test")
        finally:
            _stop_real(self.served)


def _create_named_session_on_a_real_peer(
    peer_pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
    *,
    name: str,
    prompt: str,
) -> _NamedRemoteCreate:
    """Pair, dial, and have the CREATOR create a named session on the peer.

    The frame is the product's own ``net_session_create`` over a real link, with
    a real session booted on the peer's side of it — the create QA and UX both
    typed as ``lop network sessions --peer <dev> --create --name …``.
    """
    server_a, server_b, _host_a, _port_a = peer_pair
    record, _host, _port = _pair(peer_pair, monkeypatch, role="drive")
    host_b, port_b = _listen(server_b)
    served = _serve_real_sessions(monkeypatch, server_b.root)
    try:
        link = _dial_to(server_a, record, host_b, port_b)
        reply = link.request(
            {
                "op": "net_session_create",
                "req": 21,
                "locality": "remote",
                "cwd": str(server_b.root),
                "name": name,
                "prompt": prompt,
            }
        )
        assert reply is not None and reply["op"] == "ack", reply
        assert reply["detail"]["admitted"] is bool(prompt), reply["detail"]
        created = _NamedRemoteCreate(
            server_a=server_a,
            server_b=server_b,
            served=served,
            session_id=reply["detail"]["session_id"],
            link=link,
        )
        assert created.session_id not in ("", None), reply["detail"]
        if prompt:
            # THE PROMPT IS ADMITTED BEFORE THE ACK, so the runtime is up already.
            assert (
                created.session_id in served
            ), "the relay brought up no runtime for the new session"
        else:
            # A PROMPTLESS CREATE ANSWERS BEFORE ITS RUNTIME IS UP
            # (``relay._warm_after_create``: the spawn measured 15.9-22.1 s against a
            # front end whose own window for this call is 20 s, so the answer no longer
            # waits for it — that wait is what made the desktop time out on a
            # conversation it had just created). The property these tests assert is what
            # the surfaces paint WHILE the runtime is live, so they wait for the join
            # instead of reading it out of the ack. A warm that FAILS leaves the session
            # cold and records ``session.create.warm_failed`` in the peer's audit log,
            # which is what this failure names rather than waiting forever.
            assert _wait_for(lambda: created.session_id in served, 60.0), (
                "the relay never brought up a runtime for the new session; a warm that "
                "fails says so in the peer's audit record `session.create.warm_failed`"
            )
    except BaseException:
        # The rig is what is expensive here, not the assertion: a failure between
        # the boot and the hand-off would otherwise leave a session, a runtime and
        # a loop behind for the rest of the run.
        _stop_real(served)
        raise
    return created


def test_a_named_create_paints_the_name_while_the_runtime_is_live(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Q13-2/U26: no window in which a NAMED session reads as nameless.

    QA round 13 measured ~4–5 s after a create in which the sidebar literally
    painted ``Untitled conversation`` and the CLI a nameless ``live`` row; UX
    round 4 filed the same row for one poll cycle, and their own note says the
    producer already carried the name when they looked directly — i.e. the first
    paint was the placeholder, not a wrong name. Both surfaces read the RUNTIME'S
    record for a resident session, so the name was missing from the record, which
    is the same missing seed as Q13-1 one step earlier: the record is published
    from the live session's projection, so it is named the moment the boot is.

    A PROMPTLESS create, which is QA's own repro and the sharper form of it: with
    no turn there is no auto-namer that could move the name afterwards, so what
    this asserts is only ever about the window itself.
    """
    created = _create_named_session_on_a_real_peer(
        peer_pair, monkeypatch, name="live-name-probe", prompt=""
    )
    try:
        rows = created.owner_rows()
        assert rows, "the owner cannot list the session it minted"
        assert rows[0]["state"] != "stored", (
            "this is the STORED row, not the live one, so the window Q13-2 is "
            f"about does not exist here ({rows[0]['state']!r}) and the assertion "
            "below would prove nothing"
        )
        assert rows[0]["conversation_name"] == "live-name-probe", (
            "while the runtime is live the owner's row paints "
            f"{rows[0]['conversation_name']!r} for a session the user named"
        )
        assert _cli_listing_line(
            monkeypatch, created.server_a.root, created.peer_token, capsys, created.session_id
        ).endswith("live-name-probe"), "the live row `network sessions` prints carries no name"
    finally:
        created.stop()


def test_a_named_create_keeps_its_name_through_a_real_turn(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """THE BLOCKING ONE, on a rig where the turn is real (QA Q13-1 / UX U24).

    ``lop network sessions --peer <dev> --create --name "Field notes" --prompt …``
    leaves the user's name in the title sidecar, and the session the peer then
    runs takes a real turn. What two independent rounds measured on this branch
    is that the name is gone afterwards — replaced, on the sidebar, in the
    federated listing, in the CLI's own output and in the transcript, by the
    auto-namer's title, with the typed name demoted into ``names[]`` and
    ``user_set`` rewritten to false.

    The mechanism is in the BOOT, not in the relay's write: ``--name`` reaches
    the sidecar with ``user_set=True``, and the runtime that then takes the turn
    is a new process whose transcript holds no ``conversation_name`` entry — so
    the holder the naming gates read started EMPTY, and ``user_set``, the flag
    whose whole job is "a generated title never displaces this", had no reader on
    that path at all. See ``Session._load_conversation_name``.

    EVERY SURFACE THE FINDINGS NAME IS ASSERTED, because "the sidecar is right"
    is exactly what the pin this replaces proved while the session still lost the
    name. The turn itself is asserted to have HAPPENED
    (``_RealServed.wait_for_turn``, which reads the mock's own reply out of the
    transcript): a rig that fakes the turn is what let the previous head ship a
    pin that passed while the feature was broken.
    """
    created = _create_named_session_on_a_real_peer(
        peer_pair, monkeypatch, name="Field notes", prompt="hello from my keyboard"
    )
    try:
        # THE REAL TURN, and the assertion that it happened: the mock's reply on
        # the wire this branch's `hosting: test` resolves to.
        created.owner.wait_for_turn()
        assert any(
            "Hello from the mock provider" in json.dumps(entry)
            for entry in created.owner.transcript_entries()
        ), "the peer's session never took the mocked turn this test is about"

        # 1. THE SIDECAR, which is the record the relay wrote.
        from local_operator.resume import read_title_state, stored_session_title

        state = read_title_state(created.session_dir)
        assert state is not None, "the create wrote no title sidecar"
        assert state.text == "Field notes", (
            "the name the user typed was replaced in the sidecar by "
            f"{state.text!r} once the session took a turn"
        )
        assert state.user_set is True, "the generated title took the user's precedence flag"
        assert state.names == ("Field notes",), state.names
        assert stored_session_title(created.session_dir) == "Field notes"

        # 2. THE TRANSCRIPT'S OWN NAMING STATE — what a later resume, a fork and
        # the title backfill read — and the live holder the naming gates compare
        # against, so the claim is not merely on disk.
        assert (
            created.owner.naming_state().get("text") == "Field notes"
        ), created.owner.naming_state()
        assert created.owner.naming_state().get("user_set") is True, created.owner.naming_state()
        assert created.owner.session.conversation_name == "Field notes"
        assert created.owner.session.conversation_name_state.user_set is True

        # 3. THE OWNER'S OWN ROW, 4. the federated listing, 5. the sidebar's
        # producer (the row a person actually reads) and 6. the CLI's output.
        own = created.owner_rows()
        assert own and own[0]["conversation_name"] == "Field notes", own
        remote = created.federated_rows()
        assert remote, "the creator's listing does not carry the session at all"
        assert remote[0]["conversation_name"] == "Field notes", remote[0]
        produced = created.sidebar_rows()
        assert produced, "the created session is not on the creator's sidebar"
        assert produced[0].name == "Field notes", produced[0]
        assert _cli_listing_line(
            monkeypatch, created.server_a.root, created.peer_token, capsys, created.session_id
        ).endswith("Field notes"), "`network sessions` does not print the name the user typed"
    finally:
        created.stop()


def _mesh_name(server: relay.RelayServer, name: str) -> None:
    """Give this device's own row in every network the name a person typed.

    ``MemberRecord`` is what every viewer-side label resolves from
    (``relay._fan_out_catalog``, so the sidebar heading and the picker too), and
    the IDENTITY is a separate record — which is precisely the state a device
    that started its relay before the join is in, because
    ``cli.load_or_mint(name=…)`` keeps an existing identity's name.
    """
    for network in store.list_networks(server.root):
        for member in network.active_members():
            if member.device_id == server.identity.device_id:
                member.name = name
        store.save(network, server.root)


def test_a_devices_own_receipts_use_the_name_the_mesh_knows(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UX round 4, U25: one device, one name, on every surface a person reads.

    The owner's own sentence read ``… is not running on damians-MacBook-Pro``
    about a session every other surface in the same session heads ``⇄ pixel-8``
    — a receipt that invents a second device. ``relay._own_label`` answered from
    the identity while the member table is what the sidebar, the listing, the
    picker and the create receipt all resolve, and the two diverge exactly when
    the relay was already running when the device joined.

    Asserted on the two surfaces that put a name on THIS device for a person:
    the sentence the stop op returns (the receipt UX filed) and the device block
    the CLI prints in its holder column. The identity is deliberately left alone
    — the divergence is the subject, so the fixture reproduces it rather than
    removing it.
    """
    server_a, server_b, _host_a, _port_a = peer_pair
    record, _host, _port = _pair(peer_pair, monkeypatch, role="drive")
    host_b, port_b = _listen(server_b)
    link = _dial_to(server_a, record, host_b, port_b)
    try:
        _mesh_name(server_b, "pixel-8")
        assert server_b.identity.name != "pixel-8", (
            "the fixture must keep the hostname identity and the mesh name apart, "
            "or this test would pass without the fix"
        )
        # The session has to LIVE on B or the ownership chokepoint refuses the op
        # before any receipt is composed (§7.2/INV-1); it is simply not running,
        # which is the branch whose sentence names the device.
        _own_locally(server_b.root, "ffffffffffff", server_b.identity.device_id)
        receipt = link.request(
            {
                "op": "net_session_stop",
                "req": 31,
                "locality": "remote",
                "session_id": "ffffffffffff",
                "mode": "graceful",
            }
        )
        assert receipt is not None and receipt["op"] == "ack", receipt
        detail = str(receipt["detail"]["detail"])
        assert "pixel-8" in detail, f"the owner's own receipt names it {detail!r}"
        assert server_b.identity.name not in detail, detail
        assert _call(server_b.root, "peer_session_rows")["detail"]["device_name"] == "pixel-8"
    finally:
        link.close("test")


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


def test_a_mesh_hosted_session_the_catalogue_ranks_is_listed_once(
    peer_pair: Devices,
) -> None:
    """ONE row per session, whichever pass of the listing produced it.

    ``local_session_rows`` composes three passes — the live registry, the
    ordinary catalogue, and the mesh-hosted half — and the last two OVERLAP by
    construction: a session this device hosts for the mesh carries a
    ``placement.mode="peer"`` stamp (what ``_mesh_hosted_rows`` filters on) and,
    once it has had a turn, the catalogue ranks it too (what ``_stored_rows``
    reads). The stored pass appended without recording what it appended, so the
    mesh-hosted pass re-emitted every stamped session the catalogue had already
    listed — three sessions on a peer painted as five sidebar rows, two of them
    phantoms (design round 2 D17 and UX round 2 U11, the same fact found twice).

    The PHANTOM'S NAME is why this is not cosmetic: the catalogue half and the
    mesh-hosted half name one session differently (a stored title versus a bare
    id), so the second row read as a session the user never created.
    """
    _server_a, server_b, _host, _port = peer_pair
    _own_locally(server_b.root, SESSION, server_b.identity.device_id)

    rows = server_b.local_session_rows()
    ids = [row["session_id"] for row in rows]
    assert ids.count(SESSION) == 1, (
        "a mesh-hosted session the catalogue already ranked was listed twice: " f"{rows}"
    )


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
        # NO ID TWICE: the merge is a UNION of two device listings, so a repeated
        # id here means one pass re-emitted what another had already produced —
        # and the dict comprehension below would have collapsed it silently
        # (design round 2 D17 / UX round 2 U11).
        listed = [row["session_id"] for row in payload["sessions"]]
        assert len(listed) == len(set(listed)), listed
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


def test_the_session_plane_listing_has_a_header_and_says_the_state_in_words(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """UX round 5, U29: a header, and the state in the words the app uses.

    The listing was four bare CLI columns with no header, so three of them were
    for the reader to infer from shape alone, and the STATE it printed was the
    catalogue's own token — ``stored`` — which is a third vocabulary for a state
    the rest of the app already shows (the sidebar paints that session under
    ``⇄`` with a row mark; ``resume.session_state_words`` owns the words, and
    ``--json`` keeps the token).

    Driven through ``cli._cmd_sessions`` over a real peer link, because the header
    and the row are ONE printed listing: a re-derived format string would pin
    nothing about what the command prints.
    """
    from local_operator.network import cli as network_cli

    server_a, server_b, _host_a, _port_a = peer_pair
    record, _host, _port = _pair(peer_pair, monkeypatch, role="drive")
    host_b, port_b = _listen(server_b)
    _seed(server_b.root, SESSION)
    link = _dial_to(server_a, record, host_b, port_b)
    try:
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
        capsys.readouterr()
        assert (
            network_cli._cmd_sessions(  # noqa: SLF001 - the command's own handler
                argparse.Namespace(peer=server_b.identity.device_id, all_peers=False, json=False)
            )
            == 0
        )
        out = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
        # The header is one line of the listing's own grid (review round 9, NIT), so
        # it is asserted where it LANDS rather than as a literal: a reworded label
        # fails here, and so does a column that stops lining up with its values.
        header = out[0]
        row = next(line for line in out if line.startswith(SESSION))
        assert header.index("SESSION") == row.index(SESSION), (header, row)
        assert header.index("STATE") == row.index("not running"), (header, row)
        # The words, and not the peer's 34-character id in place of its name.
        assert "not running" in row, row
        assert "stored" not in row, row
        assert server_b.identity.device_id not in row, row
    finally:
        link.close("test")


def test_a_session_lifecycle_op_deletes_on_the_owner(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§8 routes delete to the OWNER's implementation, and it lands there.

    The op is served by the mobility slice now, so what this pins is the routing
    rather than a by-name refusal: the requester asks, the OWNER deletes out of its
    own store (through the same ``remove_session_dir`` every other delete uses), and
    the requester's own root is untouched. A second ``rmtree`` of a session directory
    anywhere is what ``tests/unit/session/test_no_session_deletion.py`` prevents.
    """
    server_a, server_b, _host_a, _port_a = peer_pair
    record, _host, _port = _pair(peer_pair, monkeypatch, role="admin")
    host_b, port_b = _listen(server_b)
    _seed(server_b.root, SESSION)
    mark_store(server_b.root / "sessions")
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
    assert reply is not None and reply["op"] == "ack", reply
    assert reply["detail"]["deleted"] is True, reply["detail"]
    assert not (server_b.root / "sessions" / SESSION).exists()
    assert not (server_a.root / "sessions" / SESSION).exists()
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

        # The owner's own welcome, forwarded with no translation. Its op is the
        # OWNER's choice for this connection shape, and the pin proves the relay
        # did not rewrite it: a connection that asked for events AND the
        # canonical frontend (this one) is welcomed with the identity-only
        # ``welcome`` frame (runtime/server.py ``_welcome_frame``, commit c69d04b2),
        # which ``RemoteSessionClient.connect`` accepts beside ``projection``.
        welcome = client.recv()
        assert welcome is not None and welcome["op"] == "welcome", welcome
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


# ---------------------------------------------------------------------------
# Q-R7-2 — the forced stop's hop budget
# ---------------------------------------------------------------------------


def test_a_forced_stop_hop_outlasts_the_ladder_that_answers_it() -> None:
    """Q-R7-2: the requester's budget is at least the owner's own worst case.

    STRUCTURAL, NOT A MEASUREMENT, deliberately: the defect was a relation between
    two constants, so a test that waited the difference out would spend three
    minutes re-proving what an inequality proves in a microsecond — and this host
    cannot afford three-minute rigs.

    The ladder's bound is ``SIGTERM_GRACE_S + SIGKILL_CONFIRM_S``, the only two
    rungs a FORCED stop can reach when a socket is silent, and the hop's old budget
    (``max(op_wait_s, ENGAGE_DEADLINE_S)`` — a SPAWN's budget) sat below it. That
    was the whole bug: the owner's receipt was thrown away by a caller that had
    stopped listening, so an operator was told a peer "stopped answering" about a
    peer that was working, with no outcome, no rung and no pid.
    """
    from local_operator.session.runtime import control

    ladder = control.SIGTERM_GRACE_S + control.SIGKILL_CONFIRM_S
    assert relay.forced_stop_deadline_s() > ladder, "the hop must outlast the ladder it awaits"
    # AND THE DEFAULT IS NOT ENOUGH, which is why the special case exists: if this
    # ever stops holding, the forced-stop budget is dead code and should go.
    assert max(relay.wire.OP_WAIT_S, relay.ENGAGE_DEADLINE_S) < ladder


def test_only_the_forced_stop_hop_gets_the_ladder_budget(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The forced mode passes the derived budget; every other mode keeps the default.

    The LINK is the only thing stubbed: ``_ctl_peer_stop`` → ``_local_peer_call``
    run for real, so what is asserted is the hop this code actually issues, and the
    receipt is passed through untouched — the rung and the pid are what make "it
    acted" distinguishable from "the caller gave up" (QA round 7, Q-R7-2).
    """
    server_a, _server_b, _host, _port = peer_pair
    seen: list[float | None] = []
    detail = {"outcome": "stopped", "rung": "sigterm", "pid": 4242}

    class _RecordingLink:
        """A peer that answers at once, whatever budget it was given."""

        def request(self, frame: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
            seen.append(timeout)
            return {"op": "ack", "req": frame.get("req"), "detail": dict(detail)}

    monkeypatch.setattr(server_a, "_resolve_peer", lambda peer: peer)
    monkeypatch.setattr(server_a, "_ensure_link", lambda _peer: _RecordingLink())

    def _stop(mode: str) -> dict[str, Any]:
        reply = server_a.control_dispatch(
            "peer_session_stop",
            {"peer": "d_" + "c" * 32, "session_id": SESSION, "mode": mode, "req": 1},
        )
        assert reply["op"] == "ack", reply
        return reply["detail"]

    forced = _stop("immediate")
    passive = _stop("graceful")

    assert forced["rung"] == "sigterm" and forced["pid"] == 4242, forced
    # The receipt is the OWNER's, and it crosses unchanged in either mode: the
    # budget decides how long this side waits, never what it reports.
    assert passive == forced, passive
    assert seen == [
        relay.forced_stop_deadline_s(),
        max(server_a.settings.op_wait_s, relay.ENGAGE_DEADLINE_S),
    ]


def test_the_pending_field_reads_as_one_vocabulary() -> None:
    """Q-R7-1/N5: the wire value is a string or NO CLAIM — never ``"True"``.

    The field is the string a NEEDS column renders, so a peer's legacy boolean has
    to be translated into the vocabulary rather than stringified: ``True`` is the
    pre-fix producer's spelling of an unread completion, and it reads as that
    claim; ``"True"`` is a value no reader understands.
    """
    from local_operator.network.types import (
        NEEDS_APPROVAL,
        NEEDS_ASK,
        normalise_pending,
    )

    assert normalise_pending(NEEDS_APPROVAL) == NEEDS_APPROVAL
    assert normalise_pending(NEEDS_ASK) == NEEDS_ASK
    assert normalise_pending("  ask  ") == NEEDS_ASK
    assert normalise_pending(True) == NEEDS_ASK
    assert normalise_pending(False) is None
    assert normalise_pending(None) is None
    assert normalise_pending("") is None
    assert normalise_pending(1) is None

    row = projection.PeerRow.from_json(
        {"session_id": "s", "pending": True}, device_id="d_" + "e" * 32
    )
    assert row.pending == NEEDS_ASK


# ---------------------------------------------------------------------------
# QA round 1 (the desktop round that drove a REAL backend): the backend's half
# ---------------------------------------------------------------------------


def test_a_peer_in_two_networks_is_asked_once_and_contributes_one_row_per_session(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q1 at the source: the fan-out asks a DEVICE once, however many networks it shares.

    Measured on three real paired devices: B is in two networks with A, so the loop
    over ``this device's networks × each network's members`` reached B twice — it
    dialled ``net_catalog`` twice and appended B's rows twice, which is how two
    conversations became four rows in the listing, the search and the count. A device's
    catalogue does not depend on which network carried the question, so one answer is
    the whole answer.
    """
    from tests.unit.network.test_refusals import _admit_a_member_that_cannot_answer

    server = relay.RelayServer(
        root=root, settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1")
    )
    peer = "d_" + "b" * 32
    # TWO networks, the SAME member in both: the shape QA built with two real joins.
    _admit_a_member_that_cannot_answer(server, name="qa-laptop-b", endpoints=["127.0.0.1:1"])
    _admit_a_member_that_cannot_answer(server, name="qa-laptop-b", endpoints=["127.0.0.1:1"])

    dials: list[str] = []
    rows = [
        {"session_id": "c7c74407768f", "state": "idle", "started": 1.0, "conversation_name": "One"},
        {"session_id": "0835f0d1a2b3", "state": "idle", "started": 2.0, "conversation_name": "Two"},
    ]

    class _Link:
        def request(self, frame: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
            return {"op": "ack", "detail": {"sessions": [dict(row) for row in rows]}}

    def _ensure(device_id: str, probe_timeout_s: float | None = None) -> tuple[Any, str]:
        dials.append(device_id)
        return _Link(), ""

    monkeypatch.setattr(server, "_ensure_link_with_reason", _ensure)
    peers, sessions = server._fan_out_catalog()

    assert dials == [peer], "a device sharing two networks was asked once per network"
    assert [row["session_id"] for row in sessions] == ["c7c74407768f", "0835f0d1a2b3"]
    assert len(peers) == 1 and peers[peer]["reachable"] is True


def test_a_create_for_a_folder_the_peer_does_not_have_is_refused_by_the_peer(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q9: the peer validates the working directory the create frame names.

    The field names a path on the PEER'S disk — the requesting device cannot see that
    disk — and the desktop's own hint promises as much ("Must exist on <peer>"). The
    relay dropped the value instead, so ``/nonexistent/on/this/mac`` was accepted with
    a 200 and the conversation ran in the peer's home directory: for an agent that runs
    commands, silently the wrong place.
    """
    server_a, server_b, _host_a, _port_a = peer_pair
    record, _host, _port = _pair(peer_pair, monkeypatch, role="drive")
    host_b, port_b = _listen(server_b)
    link = _dial_to(server_a, record, host_b, port_b)
    try:
        missing = link.request(
            {
                "op": "net_session_create",
                "req": 31,
                "locality": "remote",
                "cwd": "/nonexistent/on/this/mac",
                "name": "nowhere",
                "prompt": "",
            }
        )
        assert missing is not None and missing["op"] == "error", missing
        # THE SENTENCE IS WHAT CROSSES THE PEER BOUNDARY (``wire.refusal_frame``: a peer
        # is told the refusal, never the code, so the cause stays in the local audit
        # log). What this asserts is therefore what the requesting device — and through
        # it the user — is told.
        sentence = str(missing["message"])
        assert "/nonexistent/on/this/mac" in sentence
        assert (
            "device-b" in sentence
        ), f"the refusal must name the peer that checked it: {sentence!r}"
        assert not list((server_b.root / "sessions").glob("*")), (
            "a refused create must mint nothing: a half-created session would be a row "
            "with no runtime and no way to ask for one"
        )

        # A RELATIVE PATH IS REFUSED TOO, and it is the same class of silent
        # somewhere-else: it would resolve against the RELAY's working directory, a
        # directory the user never named and cannot see from the request.
        relative = link.request(
            {
                "op": "net_session_create",
                "req": 32,
                "locality": "remote",
                "cwd": "some/folder",
                "name": "relative",
                "prompt": "",
            }
        )
        assert relative is not None and relative["op"] == "error", relative
        assert "not a full path" in str(relative["message"]), relative
        assert "device-b" in str(relative["message"]), relative
        assert not list((server_b.root / "sessions").glob("*"))
    finally:
        link.close("test")


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="root can enter any directory, so a mode bit says nothing about it",
)
def test_a_folder_the_peer_cannot_enter_is_refused_before_anything_is_minted(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review round 1, MINOR 1: ``is_dir()`` is not "this device's user can open it".

    ``stat`` needs search permission on the PARENTS only, so a ``chmod 000`` directory
    passed the Q9 check: the create answered 200 and the failure surfaced later as a
    spawn error whose only trace was a ``session.create.warm_failed`` audit record, so
    the user was left a row that could never warm and the 200 was not the signal this
    route claims it is. Measured in the same process: ``os.chdir`` on such a directory
    raises ``PermissionError`` while ``is_dir()`` is True. The mode is restored in a
    ``finally`` so the test root can still be cleaned up.
    """
    server_a, server_b, _host_a, _port_a = peer_pair
    record, _host, _port = _pair(peer_pair, monkeypatch, role="drive")
    host_b, port_b = _listen(server_b)
    locked = server_b.root / "locked"
    locked.mkdir()
    locked.chmod(0o000)
    link = _dial_to(server_a, record, host_b, port_b)
    try:
        reply = link.request(
            {
                "op": "net_session_create",
                "req": 41,
                "locality": "remote",
                "cwd": str(locked),
                "name": "locked",
                "prompt": "",
            }
        )
        assert reply is not None and reply["op"] == "error", reply
        sentence = str(reply["message"])
        # The sentence names the path and the device that checked it, the same two
        # things the missing-folder refusal names — the remedy is the same and the
        # cause is what differs.
        assert str(locked) in sentence, sentence
        assert "device-b" in sentence, sentence
        assert "cannot be entered" in sentence, sentence
        assert not list((server_b.root / "sessions").glob("*")), (
            "a create refused for an unenterable folder must mint nothing: the row a "
            "half-create leaves can never warm"
        )
    finally:
        locked.chmod(0o700)
        link.close("test")


def test_a_promptless_create_answers_before_its_runtime_has_joined(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q4b: the answer is the id, and the warm-up is not on the caller's path.

    The spawn plus its discovery record measured 15.9-22.1 s on LOOPBACK on this fleet
    while the desktop's own window for this call is 20 s, so the create timed out on a
    conversation the peer HAD created, the app reported the deadline as "refused", and
    a retry would have minted a second one. The engage is held here until the test
    releases it, so the ack arriving at all is the proof that it did not wait.
    """
    server_a, server_b, _host_a, _port_a = peer_pair
    record, _host, _port = _pair(peer_pair, monkeypatch, role="drive")
    host_b, port_b = _listen(server_b)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_b.root))
    entered = threading.Event()
    released = threading.Event()

    async def engage(session_id: str, cwd: str, work: Any, **kwargs: Any) -> None:
        entered.set()
        assert released.wait(30.0), "the test never released the warm-up"

    monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", engage)
    link = _dial_to(server_a, record, host_b, port_b)
    try:
        started = time.monotonic()
        reply = link.request(
            {
                "op": "net_session_create",
                "req": 33,
                "locality": "remote",
                "cwd": str(server_b.root),
                "name": "fast",
                "prompt": "",
            }
        )
        elapsed = time.monotonic() - started
        assert reply is not None and reply["op"] == "ack", reply
        detail = reply["detail"]
        assert detail["warming"] is True, detail
        assert detail["detail"] == "", "a join in progress is not a complaint"
        # THE ACK ARRIVED WHILE THE JOIN WAS STILL BLOCKED: ``engage`` cannot finish
        # until this test releases it, so an ack that had waited for the warm-up would
        # have taken the full 30 s (or answered an error) rather than a round trip. That
        # the worker has STARTED by now is expected — it is scheduled the moment the
        # create returns — and it is what makes the next line a real statement about
        # ordering rather than about a thread that never ran.
        assert entered.wait(5.0), "the background warm never started"
        assert not released.is_set(), "the runtime joined before the create answered"
        assert elapsed < 5.0, f"the create held the caller for {elapsed:.1f}s"
        # THE SESSION EXISTS ALREADY, which is what the id is for.
        assert (server_b.root / "sessions" / detail["session_id"]).is_dir()

        # ...and the JOIN IS REAL: releasing it lets the background warm finish.
        released.set()
        assert entered.wait(30.0), (
            "the create never warmed the runtime in the background, so the session would "
            "stay cold (a failed warm says so in the peer's audit record)"
        )
    finally:
        released.set()
        link.close("test")
