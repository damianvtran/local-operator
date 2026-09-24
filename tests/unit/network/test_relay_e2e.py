"""The relay end to end: two "devices" on one host, over real TCP.

This is the test that makes the unit invariants mean something: two config roots,
two identities, two relays, real sockets, and the whole protocol driven the way the
CLI drives it. It covers R1 (a relay per install that owns no session), R3 (the
human-confirmed pairing), R4 (zero trust: an unauthorised frame is refused) and R5
(revocation without visiting the revoked device).
"""

from __future__ import annotations

import threading
import time
from argparse import Namespace
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from local_operator.network import audit as audit_mod
from local_operator.network import cli as net_cli
from local_operator.network import identity
from local_operator.network import invite as invite_mod
from local_operator.network import relay, store, types, wire
from local_operator.network.handshake import (
    Credential,
    Handshake,
    pair_abort_frame,
    pair_timeout_seconds,
    sas_matches,
)

NETWORK_NAME = "home-net"


@pytest.fixture()
def devices(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[relay.RelayServer, relay.RelayServer, str, int]]:
    """Two relays on loopback: A listening, B able to dial it."""
    root_a = root / "a"
    root_b = root / "b"
    identity_a = identity.mint(root_a, name="device-a")
    identity_b = identity.mint(root_b, name="device-b")
    server_a = relay.RelayServer(
        root=root_a,
        settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1"),
        identity=identity_a,
        audit=audit_mod.AuditLog(root_a),
    )
    server_b = relay.RelayServer(
        root=root_b,
        settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1"),
        identity=identity_b,
        audit=audit_mod.AuditLog(root_b),
    )
    host, port = server_a.bind()
    server_a.bind_control()
    server_a.start()
    # The CLI's joining half resolves the config dir from the AMBIENT environment
    # (that is how a user runs it), so the test points the ambient dir at B's root:
    # otherwise the join would write into the isolated HOME and the test would be
    # asserting about a file the code never touched.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root_b))
    try:
        yield server_a, server_b, host, port
    finally:
        server_a.stop()
        server_b.stop()


def _init_network(server: relay.RelayServer, *, role: str = "admin") -> types.NetworkRecord:
    """Create a network on A and persist it, exactly as `lop network init` does."""
    record = types.NetworkRecord(
        network_id=store.new_network_id(),
        name=NETWORK_NAME,
        created_by=server.identity.device_id,
        self_device_id=server.identity.device_id,
        self_role=role,
        self_capabilities=sorted(types.capabilities_for_role(role)),
        listen={"address": "127.0.0.1", "port": server.settings.port, "advertised": []},
    )
    from secrets import token_bytes

    state = types.SecretState(
        network_id=record.network_id, epoch=1, secret=wire.b64u(token_bytes(32))
    )
    relay.admit(
        record,
        device_id=server.identity.device_id,
        public_key=server.identity.public_key,
        name=server.identity.name,
        role=role,
        capabilities=sorted(types.capabilities_for_role(role)),
        added_by=server.identity.device_id,
        added_via="self",
        root=server.root,
        persist=False,
    )
    store.save(record, server.root)
    store.save_secrets(state, server.root)
    return record


def _mint_invite(server: relay.RelayServer, record: types.NetworkRecord) -> tuple[str, Any]:
    state = store.load_secrets(record.network_id, server.root)
    minted = invite_mod.mint(record, state.secret, role="drive", ttl_s=600.0)
    record.invites.append(minted.record)
    store.save(record, server.root)
    path = store.save_invite_token(minted.record.invite_id, minted.token, server.root)
    assert path.exists()
    return minted.token, minted.envelope


def _join(
    server_b: relay.RelayServer,
    *,
    host: str,
    port: int,
    token: str,
    envelope: Any,
    typed_code: str = "000000",
    settings: relay.NetworkSettings | None = None,
) -> Any:
    """Drive the joining side's ceremony, with the human step supplied by the test.

    The human on the INVITER side is stubbed in the tests below (the relay asks a
    person whether their screen shows the transcribed code); the human on THIS side
    is the ``typed_code`` argument. Both halves of the ceremony are therefore real
    frames over a real socket, and only the two people are simulated — which is what
    a machine is allowed to simulate and nothing more.
    """
    args = Namespace(
        sas_stdin=True,
        verify=False,
        emit_sas=True,
        name=server_b.identity.name,
        json=True,
    )
    return net_cli._join_one(  # noqa: SLF001 — the CLI's own driver, exercised as the CLI runs it
        host=f"{host}:{port}",
        token=token,
        envelope=envelope,
        identity=server_b.identity,
        settings=settings or relay.NetworkSettings(port=0, listen_address="127.0.0.1"),
        args=args,
        wire=wire,
        Handshake=Handshake,
        Credential=Credential,
        pair_abort_frame=pair_abort_frame,
        pair_timeout_seconds=pair_timeout_seconds,
        sas_matches=sas_matches,
        invite_mod=invite_mod,
        store=store,
        relay_mod=relay,
    )


def _type_the_code(monkeypatch: pytest.MonkeyPatch, code: str | None = None) -> None:
    """The JOINER's person. ``None`` means "read the right digits off the other
    screen" — `_read_code` is handed the derived value, so this models a correct
    human rather than bypassing the check. A literal ``code`` models a wrong one."""

    def fake_read_code(args: Any, derived: str, fingerprint: str) -> str:
        del args, fingerprint
        return derived if code is None else code

    monkeypatch.setattr(net_cli, "_read_code", fake_read_code)


def _answer_confirmation(
    server: relay.RelayServer, *, admit: bool = True, timeout: float = 20.0
) -> dict[str, Any] | None:
    """The INVITER's person, standing at `lop network confirm`.

    The relay has no terminal here (a daemon, and a test process besides), so it
    parks the pairing in a 0600 pending record carrying BOTH codes; this waits for
    that record and answers it through the relay's own control op — the same path
    the CLI takes, audit record included. Returns the parked row it answered, or
    ``None`` when none appeared, which is what the refusal cases assert.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        rows = server._ctl_pair_pending({})  # noqa: SLF001 — the CLI's own control op
        if rows:
            server._ctl_pair_confirm(  # noqa: SLF001
                {
                    "invite_id": rows[0]["invite_id"],
                    "decision": "admit" if admit else "decline",
                    "matched": admit,
                    "reason": "" if admit else "declined",
                    "answered_by": "harness",
                }
            )
            return dict(rows[0])
        time.sleep(0.05)
    return None


# ---------------------------------------------------------------------------
# R3 — pairing
# ---------------------------------------------------------------------------


def test_pairing_admits_the_joiner_after_both_people_confirm(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole ceremony with a human on EACH side, driven as the CLI drives it.

    A parks the pairing and a person on A answers it (``_pair``); B's person
    transcribes the code B's own screen shows (``_type_the_code``). Both halves are
    real frames and real files, and the only thing simulated is the two people.
    """
    server_a, server_b, _host, _port = devices
    record, _host, _port = _pair(devices, monkeypatch)

    # A admitted B: the row is written BEFORE the frame that announces it.
    member = record.member(server_b.identity.device_id)
    assert member is not None and member.active
    assert member.added_via == "invite"
    assert set(member.capabilities) == set(types.capabilities_for_role("drive"))
    assert record.invites[0].state == "consumed"

    # B holds the network, the secret and — the gap this slice closed — the
    # INVITER's public key, without which no later handshake could be verified.
    joined = store.load(record.network_id, server_b.root)
    assert joined.epoch == 1
    assert len(joined.active_members()) == 2
    inviter_row = joined.member(server_a.identity.device_id)
    assert inviter_row is not None
    assert inviter_row.public_key == server_a.identity.public_key
    assert store.load_secrets(record.network_id, server_b.root).secret

    # THE QUESTION THAT WAS ASKED names both codes: the inviter's own derivation and
    # the joiner's transcription, which is what makes the inviter's human the
    # comparator rather than a spectator. §5.3: "B transcribes, A compares".
    parked = _pair_answered[record.network_id]
    assert parked["sas"] and len(str(parked["sas"])) == 6
    assert parked["transcribed"] == parked["sas"], "the two sides must derive the same code"
    assert parked["joiner_device_id"] == server_b.identity.device_id
    assert wire.sas_display(str(parked["sas"])) in str(parked["prompt"])
    assert "YOUR screen shows" in str(parked["prompt"])

    events = _events(server_a)
    assert "pairing_awaiting_confirmation" in events
    assert "pairing_confirmed" in events
    assert "member_admitted" in events


def test_a_wrong_transcription_burns_the_invite_and_admits_nothing(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The negative half of the joiner's step: the codes disagreed, so the join is
    refused AND the token is consumed, which is what makes an attacker's next
    attempt need a fresh invite (i.e. another human action).

    The inviter's human is never asked — the wrong transcription is refused before
    the pairing is parked, which is the ordering that keeps a bad code from
    becoming a question somebody can answer yes to.
    """
    server_a, server_b, host, port = devices
    record = _init_network(server_a)
    token, envelope = _mint_invite(server_a, record)
    _type_the_code(monkeypatch, code="999999")

    with pytest.raises(types.PairingRefusal) as excinfo:
        _join(server_b, host=host, port=port, token=token, envelope=envelope)

    # THE JOINER'S OWN HALF OF THE REFUSAL. This assertion is the one that was
    # missing: the inviter's invite outcome was checked below and the joiner's
    # path was not, so a call to a helper that lives in a different module
    # (``invite.sas_mismatch_sentence``) reached production and turned every wrong
    # transcription — the normal user path — into an AttributeError instead of a
    # refusal sentence (QA round 1, F-1).
    assert excinfo.value.code == "sas_mismatch"
    assert "digit" in excinfo.value.sentence or "code" in excinfo.value.sentence

    refreshed = store.load(record.network_id, server_a.root)
    assert refreshed.member(server_b.identity.device_id) is None
    assert refreshed.invites[0].state == "consumed"
    assert refreshed.invites[0].outcome == "sas_mismatch"
    # WAITED FOR, NOT READ ONCE. This row is written by the relay's own accept-loop
    # thread AFTER it has already flushed the abort frame to the joiner, so the
    # joiner learns it was refused while the inviter's durable row is still one
    # flush away. `_await_event` has the measurement: the gap is ~90 us in the
    # tightest run on this host, which is inside a single scheduler quantum, and it
    # inverted on CI (two heads, two shards) as `'pairing_refused' in []`. The
    # invite-state assertions above are unaffected: the inviter saves those BEFORE
    # it answers the joiner, which is why only this line could see the window.
    events = _await_event(server_a, "pairing_refused")
    assert "pairing_refused" in events
    assert "pairing_awaiting_confirmation" not in events
    assert not store.record_path(record.network_id, server_b.root).exists()


def test_a_declined_confirmation_admits_nobody_and_burns_the_invite(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The inviter's human says no — the case the inviter's prompt exists for.

    B transcribed correctly, so the joiner's half PASSED; the refusal comes from the
    person on A, which is exactly the interlock: a correct code is not sufficient if
    the other screen did not show the same digits.
    """
    server_a, server_b, _host, _port = devices
    with pytest.raises(Exception):
        _pair(devices, monkeypatch, admit=False)
    record = store.load(_init_network(server_a).network_id, server_a.root)
    assert record.member(server_b.identity.device_id) is None
    events = _events(server_a)
    assert "pairing_awaiting_confirmation" in events
    assert "pairing_refused" in events
    assert "member_admitted" not in events
    assert not store.record_path(record.network_id, server_b.root).exists()


def test_an_unanswered_confirmation_times_out_and_admits_nobody(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nobody answers. The window is the invite's own remaining life, so a short
    invite makes this an ordinary, fast test — and the point is that silence
    refuses: an unanswered question must never admit a device.

    No answering thread is started, which is the whole difference from the tests
    above; the relay is left to its own timeout.
    """
    server_a, server_b, host, port = devices
    record = _init_network(server_a)
    state = store.load_secrets(record.network_id, server_a.root)
    minted = invite_mod.mint(record, state.secret, role="drive", ttl_s=1.0)
    record.invites.append(minted.record)
    store.save(record, server_a.root)
    store.save_invite_token(minted.record.invite_id, minted.token, server_a.root)
    _type_the_code(monkeypatch)
    # Either shape is a refusal and both are honest: the joiner times out reading the
    # answer (``None``), or it reads the abort frame the inviter sends and raises.
    # Which one wins is a race between two timers of the same length, so the test
    # accepts both — and asserts the PROPERTY, that nobody was admitted.
    joined: Any
    try:
        joined = _join(server_b, host=host, port=port, token=minted.token, envelope=minted.envelope)
    except Exception as exc:  # noqa: BLE001 — a refusal, asserted below
        joined = exc

    refreshed = store.load(record.network_id, server_a.root)
    assert (
        refreshed.member(server_b.identity.device_id) is None
    ), "an unanswered confirmation admitted a device"
    assert not isinstance(joined, tuple), f"the join reported success with no answer: {joined!r}"
    # The inviter finishes its own window a moment after the joiner's read gives up
    # (both are the invite's remaining life), and the consume is what must land.
    deadline = time.time() + 5.0
    while time.time() < deadline:
        refreshed = store.load(record.network_id, server_a.root)
        if refreshed.invites[0].state == "consumed":
            break
        time.sleep(0.1)
    assert refreshed.invites[0].state == "consumed", "an abandoned pairing left the token redeemed"
    assert "pairing_refused" in _events(server_a)
    # And the parked question was cleaned up rather than left behind holding a code.
    assert store.pending_pairings(server_a.root) == []


def test_a_replayed_invite_is_refused(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server_a, server_b, host, port = devices
    record = _init_network(server_a)
    token, envelope = _mint_invite(server_a, record)
    _type_the_code(monkeypatch)

    def _answer_and_record() -> None:
        _answer_confirmation(server_a)

    thread = threading.Thread(target=_answer_and_record, daemon=True)
    thread.start()
    try:
        assert _join(server_b, host=host, port=port, token=token, envelope=envelope) is not None
    finally:
        thread.join(10)
    # Redeem the SAME token again: the record says consumed, so the listener refuses
    # BEFORE the challenge — silently, so the second attempt does not even learn
    # whether the invite was ever valid. The joiner reports WHAT HAPPENED AT THE
    # HOST rather than a bare failure, which is the best it can do without an
    # oracle: a sentence, not a success (QA round 1, F-4).
    second = _join(server_b, host=host, port=port, token=token, envelope=envelope)
    assert isinstance(second, str), second
    assert not isinstance(second, tuple)
    refreshed = store.load(record.network_id, server_a.root)
    assert len(refreshed.active_members()) == 2
    assert refreshed.invites[0].state == "consumed"
    assert "handshake_refused" in _events(server_a)


def test_an_admission_does_not_revert_what_landed_during_the_human_step(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The listener's admission is a read-modify-write that SPANS the human step.

    THE WIDEST WINDOW IN THE PACKAGE, and the one this cell exists for: the listener
    reads its record, shows a code, waits for a person to compare it, and then writes
    the admission from the copy it read. Everything another writer landed in those
    seconds used to be reverted by that write — and nothing reported it, because the
    file that landed was a well-formed record stamped with a HIGHER sequence than the
    one it clobbered. The writers that land there are the ones whose loss is
    expensive: the heartbeat's endpoint sync, a membership pull, and a peer's epoch
    rotation with the member list that travels with it.

    WHAT THE CELL DOES: the human step is where the concurrent write lands, because
    that IS the window, and the write is the relay's own control op (mint a second
    invite) rather than a store poke — so two real write sites are exercised, not a
    simulation of one. Both must be on disk afterwards, beside the member the pairing
    admitted. Pre-fix the second invite is gone: the admission wrote back the record
    the listener had read before it ever showed the code.
    """
    server_a, server_b, host, port = devices
    record = _init_network(server_a)
    token, envelope = _mint_invite(server_a, record)
    _type_the_code(monkeypatch)
    minted_in_the_window: list[str] = []

    def human_step(self: relay.RelayServer, **kwargs: Any) -> types.PairDecision:
        # WHILE THE HUMANS TALK, another writer lands on the same record.
        minted = self._ctl_invite({"network": record.network_id})  # noqa: SLF001
        minted_in_the_window.append(minted["invite_id"])
        return types.PairDecision(
            invite_id=kwargs["invite_id"],
            decision="admit",
            matched=True,
            reason="",
            answered_by="human",
        )

    monkeypatch.setattr(relay.RelayServer, "_inviter_human_step", human_step)
    joined = _join(server_b, host=host, port=port, token=token, envelope=envelope)
    assert joined is not None and not isinstance(joined, str), joined

    assert minted_in_the_window, "the concurrent writer never ran: no window was exercised"
    after = store.load(record.network_id, server_a.root)
    assert after.invite(minted_in_the_window[0]) is not None, (
        "the admission wrote back the copy it read before the human step, and the "
        "invite another writer minted in that window is gone"
    )
    assert after.member(server_b.identity.device_id) is not None, "the pairing admitted nobody"
    assert len(after.active_members()) == 2
    assert after.invites[0].state == "consumed"


def _drop_after_the_hello(
    server: relay.RelayServer, *, host: str, port: int, envelope: Any
) -> None:
    """A join handshake that sends its hello, takes the challenge, and vanishes.

    It stops at ``read_challenge`` on purpose: that frame is proof the listener got
    PAST ``accept_hello`` and past the invite's own checks, which is the state the
    round-1 defect needed. A helper that returned before the challenge would leave
    this test unable to tell "the invite was not burned" from "the listener never
    looked".
    """
    import socket

    from local_operator.network.identity import mint_instance_id

    sock = socket.create_connection((host, port), timeout=5)
    try:
        handshake = Handshake.new(
            role="dialer",
            identity=server.identity,
            network_id=envelope.network_id,
            epoch=envelope.epoch,
            instance_id=mint_instance_id(),
            session_protocol=net_cli._session_protocol(),  # noqa: SLF001 — the CLI's own value
            mode="join",
            capabilities=list(wire.LINK_CAPABILITIES),
            build={},
        )
        handshake.join_block = {
            "invite_id": envelope.invite_id,
            "joiner_public_key": server.identity.public_key,
            "joiner_name": server.identity.name,
        }
        handshake.send_hello(sock)
        handshake.read_challenge(wire.FrameReader(sock), wire.deadline_in(5.0))
    finally:
        sock.close()


def test_a_hello_that_never_authenticates_does_not_burn_the_invite(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round-1 MAJOR 3: an unauthenticated hello must not consume a valid invite.

    ``redeemed`` used to be written from the hello alone, so a connection that
    dropped before its auth frame left the invite unusable and the honest device's
    retry was refused ``invite_in_use``. Driven as the wire drives it: a real join
    hello (and a real challenge back, so the listener is past its invite checks),
    then a close, then the SAME token joined properly.
    """
    server_a, server_b, host, port = devices
    record = _init_network(server_a)
    token, envelope = _mint_invite(server_a, record)

    _drop_after_the_hello(server_b, host=host, port=port, envelope=envelope)
    refreshed = store.load(record.network_id, server_a.root)
    assert (
        refreshed.invites[0].state == "minted"
    ), "an unauthenticated hello consumed a valid invite"

    # The honest retry, on the same token, succeeds — which is the whole property.
    _type_the_code(monkeypatch)
    thread = threading.Thread(target=lambda: _answer_confirmation(server_a), daemon=True)
    thread.start()
    try:
        joined = _join(server_b, host=host, port=port, token=token, envelope=envelope)
    finally:
        thread.join(15)
    assert joined is not None and not isinstance(joined, str), joined
    assert store.load(record.network_id, server_a.root).invites[0].state == "consumed"
    assert len(store.load(record.network_id, server_a.root).active_members()) == 2


def test_silent_connections_are_capped_before_authentication(root: Path) -> None:
    """Round-1 MAJOR 1: the pre-auth phase is BOUNDED, so a silent connection
    costs a slot and not an unbounded thread.

    The finding's shape: N connections that send nothing produced N live handshake
    threads and 0 links, because ``max_links`` counts established links and an
    unauthenticated connection has none. Asserted on the wire rather than on a
    thread count: past the cap the socket is closed at the accept, so a read on it
    returns EOF immediately, while a slot-holder's read times out because nothing
    has been written to it and it is still open.
    """
    import socket

    root_a = root / "cap"
    server = relay.RelayServer(
        root=root_a,
        settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1", max_handshakes=2),
        identity=identity.mint(root_a, name="cap-device"),
        audit=audit_mod.AuditLog(root_a),
    )
    host, port = server.bind()
    server.bind_control()
    server.start()
    held: list[socket.socket] = []
    try:
        for _ in range(6):
            held.append(socket.create_connection((host, port), timeout=5))
        # Accept order is the connect order on one loopback listener, so the first
        # `max_handshakes` are the ones holding slots.
        for quiet in held[2:]:
            quiet.settimeout(5.0)
            assert quiet.recv(1) == b"", "a connection past the pre-auth cap was held open"
        for busy in held[:2]:
            busy.settimeout(1.0)
            with pytest.raises(TimeoutError):
                busy.recv(1)
    finally:
        for sock in held:
            sock.close()
        server.stop()


def _refusal_was_delivered(sock: Any) -> bool:
    """Has the relay closed this connection? Asked only of a socket ``select`` called readable.

    A refused connection is closed with NO frame — that silence is what keeps the port
    from being a probe oracle — so EOF is the only thing a dropped peer ever observes,
    and it is therefore the event a test can wait on instead of polling the log.
    """
    try:
        return sock.recv(1) == b""
    except OSError:
        return True


def test_a_cap_drop_names_itself_in_the_local_audit(root: Path) -> None:
    """The cap's refusal is SILENT to the peer, so the operator's log is where it has
    to be named.

    A dropped connection gets no reply and no frame — that is what keeps the port
    from being a probe oracle (`_accept_loop`) — so a saturated relay used to leave
    no trace at all on the device that was saturated: the symptom was a peer that
    could not connect while this device reported nothing. ONE record per window, not
    one per connection: the flood below drops four connections and must NOT produce
    four records, because an unauthenticated stranger may not churn the log an
    incident is reconstructed from.
    """
    import select
    import socket

    root_a = root / "cap-audit"
    server = relay.RelayServer(
        root=root_a,
        settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1", max_handshakes=2),
        identity=identity.mint(root_a, name="cap-audit-device"),
        audit=audit_mod.AuditLog(root_a),
    )
    host, port = server.bind()
    server.bind_control()
    server.start()
    held: list[socket.socket] = []
    try:
        for _ in range(6):
            held.append(socket.create_connection((host, port), timeout=5))
        # WAIT FOR THE FLOOD TO HAVE HAPPENED, NOT FOR A RECORD TO APPEAR. The old
        # loop polled the audit and broke at the FIRST record it saw, which reads a
        # SNAPSHOT of a flood still in progress: a per-connection implementation
        # passes whenever the read happens to land after one drop, and the assertion
        # could not tell that from a coalescer doing its job (measured — deleting the
        # coalescer left this test GREEN). What the relay actually does to a dropped
        # connection is CLOSE it, silently and with no frame (`_accept_loop`), so EOF
        # on the four sockets that never got a pre-auth slot IS "the drops have been
        # delivered" — and ``_note_handshake_cap`` runs BEFORE that close, so a single
        # read once the four are down sees everything the flood had to say. The two
        # sockets that did get slots stay open: a silent peer holds its slot for
        # ``handshake_timeout_s`` (10 s), well past this wait.
        refused = 0
        deadline = time.time() + 10.0
        while refused < 4 and time.time() < deadline:
            ready, _, _ = select.select(held, [], [], 0.05)
            refused = sum(1 for sock in ready if _refusal_was_delivered(sock))
        # A BACKSTOP, NOT THE ASSERTION: a relay that never delivered the four drops
        # in ten seconds is wedged, and a wedged flood must not be read as a coalescer
        # that behaved.
        assert refused >= 4, f"the server closed only {refused} of the four dropped connections"
        rows: list[dict[str, Any]] = [
            row for row in server.audit.tail(limit=200) if row.get("event") == "handshake_refused"
        ]
        # THE CONTRACT IS A BOUND, NOT A COUNT. The window is
        # ``HANDSHAKE_CAP_NOTICE_S`` wide, so a flood that straddles a boundary is
        # told in two records — but never one per connection, which is the churn this
        # record exists to prevent: FOUR drops, so a count approaching that is that
        # bug. The ``== 1`` this replaces was reading a DIFFERENT defect as a count
        # (below) and so punished a correct coalescer for it.
        assert 1 <= len(rows) < 4, rows
        # AND EVERY ROW MUST BE A DIFFERENT ROW. ``seq`` is stamped once per recorded
        # event, so two rows sharing one is not the coalescer failing at all: it is
        # the WRITER publishing ONE record twice. Measured on this head, that is what
        # the flake was — 3 of 30 isolated runs, both rows identical in ``seq`` and
        # ``ts`` — and it is why the bound above cannot be the only assertion: a
        # duplicated line passes every bound. ``AuditLog.flush`` now takes the
        # payload off the buffer before it opens the file, so a second flusher in
        # that window (the heartbeat, or a reader's ``tail``) finds nothing to
        # re-publish; ``tests/unit/network/test_audit.py`` pins that directly.
        assert len({row["seq"] for row in rows}) == len(rows), rows
        for row in rows:
            assert row["cause"] == "handshake_cap", row
            assert row["outcome"] == "refused", row
            assert row["detail"] == {"cause": "handshake_cap", "mode": "unauthenticated"}, row
    finally:
        for sock in held:
            sock.close()
        server.stop()


def test_a_member_cannot_send_into_or_close_another_members_stream(root: Path) -> None:
    """Round-1 MINOR 5: the stream-ownership rule holds on send and close, not
    only on push.

    A stream id is unpredictable (``os.urandom``), but unpredictable is not
    unforgeable — a member that learns one from a log, a traceback or a bug must not
    be able to write into, or end, another member's stream. The PUSH path always
    checked ``stream.link is link``; the request path (``net_stream`` with
    ``send``/``close``) did not, so knowing the id was enough.
    """
    root_a = root / "streams"
    server = relay.RelayServer(
        root=root_a,
        settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1"),
        identity=identity.mint(root_a, name="stream-device"),
        audit=audit_mod.AuditLog(root_a),
    )
    owner = cast(relay.PeerLink, object())
    other = cast(relay.PeerLink, object())
    stream = relay._Stream(  # noqa: SLF001 — the object the ownership rule protects
        stream_id="s_" + "a" * 16,
        session_id="sess-1",
        peer_device_id="d_owner",
        link=owner,
    )
    server._streams[stream.stream_id] = stream  # noqa: SLF001

    assert server._stream_for(owner, stream.stream_id) is stream  # noqa: SLF001
    assert server._stream_for(other, stream.stream_id) is None  # noqa: SLF001
    # The push and closed routes answer "not mine" for a link that does not own it.
    assert server.route_stream_push(other, {"stream": stream.stream_id, "frame": {}}) is False
    assert server.route_stream_closed(other, {"stream": stream.stream_id}) is False
    # A close REQUEST from the wrong link is refused by name rather than obeyed, and
    # with the same refusal an unknown id gets — the answer never confirms whether
    # the id exists on this device.
    with pytest.raises(types.MeshRefusal) as excinfo:
        server._op_stream(  # noqa: SLF001
            other, {"op": "net_stream", "action": "close", "stream": stream.stream_id}
        )
    assert excinfo.value.code == "unknown_stream"
    assert stream.closed is False, "a member closed another member's stream"


# ---------------------------------------------------------------------------
# R1/R4 — a member link, a catalogue, and a refusal
# ---------------------------------------------------------------------------


#: The parked pairing `_pair`'s inviter answered, keyed by network id, so a test can
#: assert on the question that was actually asked and on the codes it carried.
_pair_answered: dict[str, dict[str, Any]] = {}


def _pair(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
    *,
    role: str = "drive",
    ttl_s: float = 600.0,
    admit: bool = True,
    settings: relay.NetworkSettings | None = None,
) -> tuple[types.NetworkRecord, str, int]:
    server_a, server_b, host, port = devices
    record = _init_network(server_a)
    state = store.load_secrets(record.network_id, server_a.root)
    minted = invite_mod.mint(record, state.secret, role=role, ttl_s=ttl_s)
    record.invites.append(minted.record)
    store.save(record, server_a.root)
    store.save_invite_token(minted.record.invite_id, minted.token, server_a.root)
    _type_the_code(monkeypatch)
    answered: dict[str, Any] = {}
    failures: list[BaseException] = []

    def _answer_and_record() -> None:
        try:
            row = _answer_confirmation(server_a, admit=admit)
        except BaseException as exc:  # noqa: BLE001 — reported below, not swallowed
            failures.append(exc)
            return
        if row:
            answered.update(row)

    # The two humans run CONCURRENTLY — two people at two keyboards — and the joiner
    # blocks until the inviter's answer reaches it, so answering after the call
    # returns would deadlock on the joiner's own wait.
    thread = threading.Thread(target=_answer_and_record, daemon=True)
    thread.start()
    try:
        joined = _join(
            server_b,
            host=host,
            port=port,
            token=minted.token,
            envelope=minted.envelope,
            settings=settings,
        )
    finally:
        thread.join(10)
    # An exception inside the answering thread would otherwise present as the JOINER
    # timing out minutes later, pointing at the wrong side of the ceremony.
    assert not failures, f"the inviter's human step raised: {failures[0]!r}"
    assert joined is not None
    _pair_answered[record.network_id] = answered
    return store.load(record.network_id, server_a.root), host, port


def test_a_member_dials_and_gets_a_catalogue_reply(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point of the transport: one relay asks another a question and
    gets an answer, over mutually-authenticated records."""
    server_a, server_b, host, port = devices
    record, _host, _port = _pair(devices, monkeypatch)
    link, reason = server_b.dial(record.network_id, host=f"{host}:{port}", epoch=record.epoch)
    assert link is not None, f"the member handshake failed: {reason}"
    reply = link.request({"op": "net_catalog", "req": 41, "locality": "remote"})
    assert reply is not None, "no reply to net_catalog"
    # A reply is ALWAYS an ack carrying the payload in `detail` (design §10.2) — and
    # that shape is load-bearing: a reply bearing the request's own op name would be
    # read as a new request by the peer, and the two would answer each other forever.
    assert reply["op"] == "ack"
    assert reply["req"] == 41
    detail = reply["detail"]
    assert detail["device"]["device_id"] == server_a.identity.device_id
    assert isinstance(detail["sessions"], list)
    # The catalogue is the READ-THROUGH: it lists session rows, and the relay holds
    # no transcript of its own.
    assert "transcript" not in str(reply)
    assert link.stray_replies == 0, "the link answered something nobody asked"
    link.close("test")


def test_a_read_member_cannot_prompt_through_the_link(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server_a, server_b, host, port = devices
    record, _host, _port = _pair(devices, monkeypatch, role="read")
    link, reason = server_b.dial(record.network_id, host=f"{host}:{port}", epoch=record.epoch)
    assert link is not None, reason
    reply = link.request(
        {
            "op": "net_forward",
            "req": 42,
            "locality": "remote",
            "frame": {"op": "prompt", "command_id": "c1", "text": "do something"},
        }
    )
    assert reply is not None
    assert reply["op"] == "error"
    assert "prompt" in str(reply["message"])
    assert "authorisation_refused" in _events(server_a)
    link.close("test")


def test_a_frame_claiming_local_locality_is_refused_over_the_link(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server_a, server_b, host, port = devices
    record, _host, _port = _pair(devices, monkeypatch)
    link, reason = server_b.dial(record.network_id, host=f"{host}:{port}", epoch=record.epoch)
    assert link is not None, reason
    reply = link.request({"op": "net_catalog", "req": 43, "locality": "local"})
    assert reply is not None and reply["op"] == "error"
    link.close("test")


def test_a_forwarded_session_op_is_refused_with_a_named_sentence(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The session plane belongs to another slice, and this build says so rather than
    silently pretending to have carried the frame."""
    server_a, server_b, host, port = devices
    record, _host, _port = _pair(devices, monkeypatch, role="admin")
    link, reason = server_b.dial(record.network_id, host=f"{host}:{port}", epoch=record.epoch)
    assert link is not None, reason
    reply = link.request(
        {
            "op": "net_forward",
            "req": 44,
            "locality": "remote",
            "frame": {"op": "prompt", "command_id": "c1", "text": "hi", "session_id": "s1"},
        }
    )
    assert reply is not None and reply["op"] == "error"
    assert "session" in str(reply["message"])
    link.close("test")


# ---------------------------------------------------------------------------
# R5 — revocation without visiting the revoked device
# ---------------------------------------------------------------------------


def test_revocation_rotates_the_epoch_and_refuses_the_removed_device_afterwards(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server_a, server_b, host, port = devices
    record, _host, _port = _pair(devices, monkeypatch)
    removed_id = server_b.identity.device_id
    before = store.load_secrets(record.network_id, server_a.root).secret

    result = server_a._ctl_member_rm(  # noqa: SLF001 — the CLI's own control op
        {"network": record.network_id, "device_id": removed_id}
    )
    assert result["epoch"] == 2
    after = store.load_secrets(record.network_id, server_a.root)
    assert after.secret != before
    assert after.previous_epoch == 1

    # CONVERGENCE RULE at the queue: the removed device is offline, so the rotation
    # would have been queued for it — and it is NOT, because it carries the secret.
    assert store.queued_frames(removed_id, server_a.root) == []

    # A FRESH connection from the removed device is refused, which is the whole of
    # R5: no visit, no cooperation, no warning.
    # The refusal is SILENT by design: a closed socket, no error frame, and the
    # reason only in the removing device's own audit record — WHICH IS WHY THE
    # DIALER'S OWN REASON IS NAMED FOR WHAT IT OBSERVED. It used to say
    # ``handshake_failed:ConnectionError``, a transport fact that a removed device's
    # operator could not act on; it now says ``handshake_refused:ConnectionError``,
    # because a peer that accepts the connection and closes it during the handshake
    # is what this protocol's refusal looks like (QA round 3, Q-R3-2). The transport
    # class stays in the suffix: `handshake_refused:TimeoutError` is a peer that never
    # answered, which is a different incident.
    link, reason = server_b.dial(record.network_id, host=f"{host}:{port}", epoch=1)
    assert link is None
    assert reason.startswith("handshake_refused") or reason in ("not_a_member", "epoch_stale")
    assert "member_removed" in _events(server_a)
    assert "handshake_refused" in _events(server_a)


def test_the_removed_device_does_not_learn_the_new_secret(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other half of revocation: a device that cannot authenticate must not be
    handed the key it would need to read new traffic."""
    server_a, server_b, host, port = devices
    record, _host, _port = _pair(devices, monkeypatch)
    old_secret = store.load_secrets(record.network_id, server_b.root).secret
    server_a._ctl_member_rm(  # noqa: SLF001
        {"network": record.network_id, "device_id": server_b.identity.device_id}
    )
    new_secret = store.load_secrets(record.network_id, server_a.root).secret
    assert store.load_secrets(record.network_id, server_b.root).secret == old_secret
    assert new_secret != old_secret


def test_an_untrusted_network_refuses_a_fresh_connection(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A panic is a STATE, not a one-off action: a connection that arrives afterwards
    is refused at handshake step 3."""
    server_a, server_b, host, port = devices
    record, _host, _port = _pair(devices, monkeypatch)
    server_a._ctl_panic({"network": record.network_id})  # noqa: SLF001
    assert store.load(record.network_id, server_a.root).trust == "untrusted"
    link, reason = server_b.dial(record.network_id, host=f"{host}:{port}", epoch=record.epoch)
    assert link is None
    assert "untrusted" not in reason  # never told why: the refusal is a closed socket
    assert "handshake_refused" in _events(server_a)
    # Recovery is explicit and local.
    server_a._ctl_trust({"network": record.network_id, "trust": "active"})  # noqa: SLF001
    assert store.load(record.network_id, server_a.root).trust == "active"


def test_a_second_live_claim_on_one_device_id_evicts_and_audits(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
) -> None:
    """Copied-key detection: the copy IS the device at the crypto layer, so what the
    design does instead is make it visible — a second live claim outside the restart
    grace window evicts the first, audits it, and flags the member.

    The tracker is driven directly because the wire path would need a second process
    holding a copied key; the FENCE is what is under test.
    """
    server_a = devices[0]
    tracker = server_a.identity_use
    assert tracker.observe("d_copy", instance_id="i_one", link_id="l1", now=1000.0).kind == "new"
    verdict = tracker.observe("d_copy", instance_id="i_two", link_id="l2", now=1060.0)
    assert verdict.kind == "duplicate"
    assert verdict.evicted is not None
    server_a._note_duplicate("d_copy", "i_two")  # noqa: SLF001 — the relay's own audit path
    assert "duplicate_identity" in _events(server_a)


# ---------------------------------------------------------------------------
# F-2 — an endpoint a PEER can dial, without being told the address
# ---------------------------------------------------------------------------


def test_a_paired_device_is_dialable_from_its_record_alone(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each side records an address for the other that `_ensure_link` can dial.

    QA round 1's blocker (F-2), as a property. After a real join each side must
    hold an endpoint for the other that IT never observed, and a relay with nothing
    in memory — the production shape, where the process that pairs is not the
    process that later lists — must form the link by dialling it. Before this, the
    self row was written `endpoints: []` and the only other address ever recorded
    was the observed source address (an ephemeral NAT port), so a paired peer was
    permanently unreachable and every remote-session verb refused.
    """
    server_a, server_b, host, port = devices
    # B IS BOUND AND LISTENING so the reciprocal dial can be proven, not asserted.
    host_b, port_b = server_b.bind()
    server_b.bind_control()
    server_b.start()
    record, _host_a, _port_a = _pair(
        devices,
        monkeypatch,
        settings=relay.NetworkSettings(port=port_b, listen_address="127.0.0.1"),
    )

    # B holds A's endpoint even though B never dialled A: it came from A's welcome.
    joined = store.load(record.network_id, server_b.root)
    inviter_row = joined.member(server_a.identity.device_id)
    assert inviter_row is not None and inviter_row.endpoints
    # F-7: B's durable record advertises B, NOT the address B happened to dial.
    assert joined.listen.get("advertised") == [f"127.0.0.1:{port_b}"]

    # A FRESH RELAY on B's saved root, with nothing in memory, dials A from the row.
    # This is the production path: the relay that pairs is not the relay that lists.
    fresh = relay.RelayServer(
        root=server_b.root,
        settings=relay.NetworkSettings(port=port_b, listen_address="127.0.0.1"),
        identity=server_b.identity,
        audit=audit_mod.AuditLog(server_b.root),
    )
    link, reason = fresh._ensure_link_with_reason(  # noqa: SLF001 — the production dial path
        server_a.identity.device_id
    )
    assert link is not None, f"the joiner could not dial the inviter from its record: {reason}"
    link.close("test")

    # ...and the reciprocal: A holds B's DECLARED endpoint, from B's hello.
    a_record = store.load(record.network_id, server_a.root)
    b_row = a_record.member(server_b.identity.device_id)
    assert b_row is not None and b_row.endpoints == [f"{host_b}:{port_b}"], b_row
    back, back_reason = server_a._ensure_link_with_reason(  # noqa: SLF001
        server_b.identity.device_id
    )
    assert back is not None, back_reason
    back.close("test")


# ---------------------------------------------------------------------------
# The relay is not a session owner
# ---------------------------------------------------------------------------


def test_the_relay_never_becomes_a_session_owner() -> None:
    """R1's structural half.

    A relay that owned sessions would allow two writers of one transcript, would
    make `lop network stop` a way to lose work, and would orphan every session it
    owned on a crash. The import graph is where that is enforced, because intention
    is not a barrier.
    """
    forbidden = (
        "session.runtime.serving",
        "session.runtime.process",
        "session.session_factory",
        "local_operator.session.session",
        "session_lease",
    )
    # THE ONE READ THE RELAY MAY MAKE OF THE LEASE, by exact import line.
    # ``session_lease.lease_holder`` reads the claim file WITHOUT acquiring —
    # that non-acquisition is its whole contract (see its docstring) — and the
    # move's source-side retire needs it to refuse moving a session a record-less
    # process is writing. Reading who holds a lease is not holding one. Every
    # OTHER name in that module (``acquire_session_lease``, the reapers) is still
    # forbidden: the carve-out is the literal line, so importing anything beside
    # ``lease_holder`` on it, or importing the module, is an offender again.
    read_only = "from local_operator.session_lease import lease_holder\n"
    offenders: list[str] = []
    for path in Path(__file__).resolve().parents[3].joinpath("local_operator/network").glob("*.py"):
        text = path.read_text(encoding="utf-8").replace(read_only, "")
        for name in forbidden:
            if f"import {name}" in text or f"from local_operator.{name}" in text:
                offenders.append(f"{path.name}: {name}")
    assert offenders == [], f"the relay imports a session owner: {offenders}"


#: The modules the MOVE owns, exempt from the scan below BY NAME.
#:
#: Moving a session is the one thing the relay does that writes session state, and
#: that is the design rather than a leak: the destination writes verified bytes into
#: ``network/staging/`` (outside ``sessions/``) and adopts them with ONE
#: ``os.replace``; a replica is recovered into a NEW session directory. So the guard
#: that matters for these two files is a PER-CALL one, and it exists and is
#: stronger than this scan: ``tests/unit/session/test_no_session_deletion.py``
#: allow-lists every rename, replace and rmtree in them at the call site with a
#: reason, and ``tests/unit/network/test_mobility.py`` pins that the only path into
#: ``sessions/`` is a promote of a copy whose every byte was verified against the
#: owner's manifest.
_MOVE_WRITERS: frozenset[str] = frozenset({"mobility.py", "sync.py"})


def test_the_relay_writes_no_session_state() -> None:
    """R1's second structural half, stated as a rule a reviewer can check.

    The relay READS the session plane (``registry.scan`` — a read-through cache) and
    must never write it: no transcript, no lease, no session record. The imports are
    guarded by ``test_the_relay_never_becomes_a_session_owner``; this asserts the
    write side, which an import guard cannot see.
    """
    offenders: list[str] = []
    for path in sorted(
        Path(__file__).resolve().parents[3].joinpath("local_operator/network").glob("*.py")
    ):
        if path.name in _MOVE_WRITERS:
            continue
        text = path.read_text(encoding="utf-8")
        if "transcript.jsonl" in text:
            offenders.append(f"{path.name}: names a transcript")
        for write_call in ('open("w', "open('w", "write_text(", "write_bytes("):
            for chunk in text.split(write_call)[:-1]:
                tail = chunk[-400:]
                if "sessions" in tail and "run" not in tail.rsplit("sessions", 1)[1][:8]:
                    offenders.append(f"{path.name}: writes near a sessions/ path")
    assert offenders == [], offenders


def _events(server: relay.RelayServer) -> list[str]:
    return [str(record.get("event")) for record in server.audit.tail(limit=500)]


def _await_event(server: relay.RelayServer, event: str, *, timeout: float = 5.0) -> list[str]:
    """The relay's audit events, waiting for ``event`` to land first.

    Several of these cells drive the OTHER device's ceremony through a real socket
    and then assert about the row the RELAY's own thread writes — which it writes
    after it has already answered the joiner, so the cell can observe the refusal
    before the inviter's durable record exists. A relay-backed write is concurrent
    with the test thread by construction; a single read is therefore a read of a
    window the cell does not own.

    Bounded, and it does not weaken the assertion it guards: an event that never
    lands still fails the caller's ``in`` check, five seconds later.

    What each cell's margin actually is, measured on 2026-09-22 while the host ran
    at load ~48 (the probe and its transcripts are not committed; the numbers are
    from `AuditLog.record` and `_events` timestamps):

    * the wrong-transcription refusal (this module's ``pairing_refused``):
      min -0.09 ms over 32 runs, i.e. the row beat the joiner's raise by 90 us at
      its tightest, and the cell reads ~0.2 ms after the raise. That is the one
      that failed in CI, and the reason is here: nothing separates the two sides
      but scheduling.
    * the declined confirmation: -2.5 ms at its worst over 4 runs.
    * the unanswered confirmation: the inviter records 117-329 ms AFTER the joiner
      gives up, and the cell's read landed ~150 ms after the row.

    So only the first one needs the wait to be honest; the two slower paths are
    left as they were rather than converted for symmetry.
    """
    deadline = time.time() + timeout
    events = _events(server)
    while event not in events and time.time() < deadline:
        time.sleep(0.02)
        events = _events(server)
    return events


def test_a_stale_peer_record_is_reaped_and_the_relay_reports_its_links(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
) -> None:
    server_a = devices[0]
    record = server_a.peer_record()
    assert record.protocol == types.MESH_PROTOCOL_VERSION
    assert record.device_id == server_a.identity.device_id
    assert record.control_port == server_a._control_port  # noqa: SLF001
    assert record.networks == []  # publishing always, even with nothing to publish
    published = store.publish_peer_record(record, server_a.root)
    assert published.exists()
    scanned = store.scan_peer_records(server_a.root)
    assert any(row.pid == record.pid for row, _state in scanned)
    status = server_a.status()
    assert status["device_id"] == server_a.identity.device_id
    assert isinstance(status["links"], list)
