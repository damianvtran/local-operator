"""The endpoint probe and the member-table pull: the two round-2 mesh defects.

WHAT IS TESTABLE HERE, AND WHAT NEEDS THE RIG. These tests pin the properties a
unit test can pin honestly:

* the row's ORDER does not decide who is reachable — every declared address gets
  its own bounded attempt, and an address declared LAST can win;
* a candidate nobody dialled is reported as ``not_attempted``, never as
  unreachable — the distinction an operator acts on;
* addresses that failed differently are each named, so a dead lease and a wrong
  port do not read as one failure;
* a socket the probe does not keep is CLOSED, not leaked;
* a peer's member table is adopted under the epoch rules — new rows are learned,
  a tombstone is never revived, a local row keeps its authority, and a peer with
  an older table can never shrink ours.

A REAL BLACK HOLE — an address whose packets are dropped, so a dial HANGS instead
of refusing — cannot be built portably in a unit test: it needs a non-routable
address, which behaves differently on a laptop and in a CI container. That case is
covered by the three-relay run over loopback in the evidence directory, which is
where the end-to-end claim belongs. What is pinned here is the code path that made
that hang fatal (QA round 2, Q-R2-2).

THREE WAYS THIS FILE ITSELF WENT FLAKY UNDER XDIST, all fixed at the source rather
than by loosening an assertion, because a test that cannot fail is not evidence:

* an address made "refused" by binding port 0 and closing it can be re-bound by a
  PARALLEL worker's own ``bind(0)``, which turns it into a live listener and makes
  ``winner`` nondeterministic — so the unraceable fixture is a privileged port
  nothing unprivileged can claim (``_refused_endpoint``);
* ``attempts`` on the EARLY-RETURN path is only what the collector had heard when
  the first address answered, so a candidate can read ``no_answer`` while being
  perfectly reachable — the test that asserts every address's own answer therefore
  asks for ``wait_all=True`` and asserts ``complete`` before reading it; and
* a candidate "still in flight" cannot be ordered up by handing the probe a
  deadline 100 µs out (QA round 14, Q14-1): on warm loopback the dial itself
  frequently reports inside that window — measured in one process, 430 of 600
  runs (71.7%) answered ``connect_failed:ConnectionRefusedError`` with
  ``complete=True``, against 12/12 ``no_answer`` in a cold process — so the cell
  that names an unreported candidate hands the probe an ALREADY-EXPIRED deadline
  instead, which reaches the same naming pass with one deterministic answer.
"""

from __future__ import annotations

import socket
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import identity, relay, store, types, wire
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _pair,
    devices,
)

SELF = "d_" + "a" * 32
PEER = "d_" + "b" * 32
NEWCOMER = "d_" + "c" * 32

Devices = tuple[relay.RelayServer, relay.RelayServer, str, int]


@pytest.fixture()
def peer_pair(request: pytest.FixtureRequest) -> Devices:
    """The shared two-relay fixture, under a name these tests can take."""
    pair: Devices = request.getfixturevalue("devices")
    return pair


# ---------------------------------------------------------------------------
# The probe's own behaviour
# ---------------------------------------------------------------------------


def _live_endpoint() -> tuple[socket.socket, str]:
    """A real listener on loopback, and the address that reaches it."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(8)
    host, port = server.getsockname()[:2]
    return server, f"{host}:{port}"


def _refused_endpoint() -> str:
    """An address NOTHING CAN BE LISTENING ON, so "refused" is deterministic.

    Port 1 is in the privileged range: no unprivileged process — this suite, an
    xdist worker beside it, or anything else on the host — can bind it, and the
    ephemeral allocator never hands it out. That is the whole point. The obvious
    alternative is to bind port 0, read the port, close the socket and dial that
    address, and it carries a window in which a PARALLEL test's own ``bind(0)``
    takes the port: the address this test calls refused then becomes a live
    listener, the probe has two successes racing for ``winner``, and the failure is
    load-sensitive and unreproducible at ``-n0``. Prose is not a guarantee here, so
    the address is one nothing can claim rather than one that merely is not claimed
    yet.

    The FAILURE CLASS is deliberately not pinned by the caller: the kernel answers
    a connect to a closed loopback port with ``ConnectionRefusedError`` on the
    usual path and ``ConnectionResetError`` when it resets instead, and both are
    the platform's own name for the same fact. What this file asserts is that the
    address DID NOT CONNECT and that the probe named the failure — which is the
    property the report exists for.
    """
    return "127.0.0.1:1"


def test_every_declared_address_is_dialled_and_the_last_one_can_win() -> None:
    """The defect, in one assertion: the reachable address is LAST and still wins.

    A member row is written by the DECLARING device, so its order is not something
    a receiver can fix. A probe that walked the row and spent a shared deadline
    would report this member unreachable with its one good address untouched.

    ``wait_all=True`` DELIBERATELY. With a single candidate's answer being enough,
    the probe returns the moment an address connects, and a candidate that has not
    REPORTED by then is labelled ``no_answer`` — honest (“not heard from yet”) but
    not this test's claim, which is about what each of the three addresses did. That
    distinction is exactly what `complete` reports, so the test asks for the full
    answer and asserts it got one: reading a partial list as if it were complete is
    how this test first went flaky under xdist (and why the field exists).
    """
    server, live = _live_endpoint()
    refused = _refused_endpoint()
    try:
        probe = relay.probe_candidates(
            ["not-an-address", refused, live],
            deadline=time.monotonic() + 10.0,
            connect_cap=2.0,
            wait_all=True,
        )
        assert probe.winner == live, probe.attempts
        assert probe.sock is not None
        assert probe.complete is True, probe.attempts
        assert probe.reason == "", "a probe with a winner has no reason to give"
        probe.sock.close()
        by_endpoint = {row.endpoint: row for row in probe.attempts}
        assert by_endpoint["not-an-address"].detail == "bad_endpoint"
        assert by_endpoint[refused].connected is False
        assert by_endpoint[refused].detail.startswith("connect_failed:"), by_endpoint[refused]
        assert by_endpoint[live].detail == "ok"
    finally:
        server.close()


def test_a_candidate_the_deadline_never_reached_is_reported_as_not_attempted() -> None:
    """``not_attempted`` is its own answer, and it is not ``unreachable``.

    The two lead an operator to different actions — one is "that device is off",
    the other is "this listing gave up" — and reporting the second as the first is
    the dead-instrument failure in a reachability field (Q-R2-2's second half).
    """
    probe = relay.probe_candidates(
        ["127.0.0.1:1", "127.0.0.1:2"],
        deadline=time.monotonic() - 1.0,
        connect_cap=2.0,
    )
    assert probe.sock is None
    assert probe.complete is False, "nothing was dialled, so nothing reported"
    assert [row.detail for row in probe.attempts] == ["not_attempted", "not_attempted"]
    assert probe.reason.startswith("not_attempted")


def test_a_candidate_that_never_reported_is_named_rather_than_dropped() -> None:
    """Every declared address appears in the result, even one that reported nothing.

    A missing row reads as "fine" to whoever renders the table, which is the one
    thing a probe result must never say about an address nobody got an answer from.
    The row asserted here is the one the prober SYNTHESISES in its closing pass for
    an endpoint that produced no result of its own, and the claim is that the row
    exists, is spelled as the member declared it, and carries a reason pointing at
    that missing answer rather than at a dial that never happened.

    THE DEADLINE IS ALREADY SPENT, DELIBERATELY (QA round 14, Q14-1). Asking for an
    attempt "still in flight" by handing this probe ``deadline=time.monotonic() +
    0.0001`` does not ask for that state — it asks for whichever of three real
    answers wins a 100 µs window, measured in one process as 430 of 600 runs
    (71.7%) reporting ``connect_failed:ConnectionRefusedError`` — whose ``complete``
    is True, so the fast refusal broke the ``complete is False`` assertion too —
    against 169 ``no_answer``, so the cell was a coin flip in any warm shard.
    An expired deadline is what makes the answer single: nothing is dialled, so the
    closing pass names the endpoint ``not_attempted`` — the same pass, deterministic
    — and that is the production shape the neighbouring cell exists for as well (a
    listing whose budget the earlier members spent before this one's turn).
    """
    probe = relay.probe_candidates(
        ["127.0.0.1:1"], deadline=time.monotonic() - 1.0, connect_cap=30.0
    )
    assert len(probe.attempts) == 1
    assert probe.attempts[0].endpoint == "127.0.0.1:1"
    assert probe.attempts[0].connected is False
    assert probe.attempts[0].detail == "not_attempted"
    assert probe.complete is False
    assert probe.reason == probe.attempts[0].detail


def test_the_probe_keeps_only_the_winner_and_closes_the_rest() -> None:
    """A probe that returns one socket and leaks the others eats the peer's
    connection budget one probe at a time."""
    first, first_address = _live_endpoint()
    second, second_address = _live_endpoint()
    servers = [first, second]
    try:
        probe = relay.probe_candidates(
            [first_address, second_address],
            deadline=time.monotonic() + 10.0,
            connect_cap=2.0,
            wait_all=True,
        )
        assert probe.sock is not None
        assert probe.complete is True, probe.attempts
        assert [row.detail for row in probe.attempts] == ["ok", "ok"]
        accepted = []
        for server in servers:
            server.settimeout(5.0)
            accepted.append(server.accept()[0])
        # The probe closed everything it did not keep: one of the two accepted
        # connections reaches EOF, and the winner's does not.
        closed = 0
        for sock in accepted:
            if _sees_eof(sock):
                closed += 1
        assert closed == 1, "the probe must close every socket but the winner"
        for sock in accepted:
            sock.close()
        probe.sock.close()
    finally:
        for server in servers:
            server.close()


def _sees_eof(sock: socket.socket, *, seconds: float = 10.0) -> bool:
    """Whether this accepted connection has been closed by its peer."""
    sock.settimeout(0.2)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            if sock.recv(1) == b"":
                return True
        except (TimeoutError, BlockingIOError):
            continue
        except OSError:
            return True
    return False


def test_one_failure_code_is_reported_as_itself_and_several_are_named() -> None:
    """The common single-address row keeps its documented one-code answer; a mixed
    row names every address, so a diagnosis is not averaged into a shrug."""
    single = [relay.CandidateAttempt("10.0.0.1:7", False, "connect_failed:TimeoutError")]
    assert relay.probe_reason(single) == "connect_failed:TimeoutError"
    same = [
        relay.CandidateAttempt("10.0.0.1:7", False, "connect_failed:TimeoutError"),
        relay.CandidateAttempt("10.0.0.2:7", False, "connect_failed:TimeoutError"),
    ]
    assert relay.probe_reason(same) == "connect_failed:TimeoutError"
    mixed = [
        relay.CandidateAttempt("10.0.0.1:7", False, "connect_failed:TimeoutError"),
        relay.CandidateAttempt("lan-box:7", False, "bad_endpoint"),
    ]
    assert relay.probe_reason(mixed) == (
        "unreachable: 10.0.0.1:7 connect_failed:TimeoutError; lan-box:7 bad_endpoint"
    )
    # And an address that ANSWERED is not part of a failure sentence: a reason
    # beside a live socket would read "unreachable: ... ok".
    assert relay.probe_reason([relay.CandidateAttempt("10.0.0.1:7", True, "ok")]) == ""
    assert relay.probe_reason([]) == ""


def test_every_detail_the_probe_writes_is_a_code_the_human_gloss_can_read() -> None:
    """Round 10, MAJOR-2 and MINOR-2, made structural at the producer.

    A human surface tells a compound ``unreachable: <endpoint> <detail>; …`` reason
    from the relay's prose by asking whether EVERY segment ends in one of these codes
    (``resume._carries_wire_tokens`` reads :func:`relay.is_probe_detail`), and it
    reads each code through ``resume.PEER_REASON_WORDS``. That makes this vocabulary
    a CONTRACT between the probe and every reader: a detail written outside the set
    hands the whole list back to the reader as prose — endpoints and exception class
    names included — and a code with no entry prints itself.

    Both halves are asserted over the codes the REAL producer writes on the three
    deterministic paths a unit test can drive: a budget that expired before this
    candidate's turn came, an endpoint whose port cannot be parsed, and an address
    that refuses the connection.
    """
    import local_operator.resume as resume

    expired = relay.probe_candidates(
        ["127.0.0.1:1"], deadline=time.monotonic() - 1.0, connect_cap=1.0
    )
    unparseable = relay.probe_candidates(
        ["not-a-host-port"], deadline=time.monotonic() + 10.0, connect_cap=1.0
    )
    refused = relay.probe_candidates(
        [_refused_endpoint()], deadline=time.monotonic() + 10.0, connect_cap=10.0
    )
    written = {row.detail for probe in (expired, unparseable, refused) for row in probe.attempts}
    assert {"not_attempted", "bad_endpoint"} <= written, written
    for detail in sorted(written):
        assert relay.is_probe_detail(detail), detail
        # The reader's half: no code reaches a person as itself.
        assert resume.peer_reason_words(detail) != detail, detail
    # The set the recognition reads is also the set the writer writes: the open
    # exception family is admitted by its prefix, and a bare word is not admitted
    # at all — that admission is what the recognition turns on (MINOR-2).
    assert relay.is_probe_detail(f"{relay.CONNECT_FAILED_PREFIX}OSError")
    assert relay.is_probe_detail("refused") is False
    assert relay.is_probe_detail("host-a") is False


# ---------------------------------------------------------------------------
# The member table, learned from a peer
# ---------------------------------------------------------------------------


def _record(device_id: str) -> types.NetworkRecord:
    record = types.NetworkRecord(
        network_id="n_0123456789abcdef01234567",
        name="home-net",
        epoch=1,
        self_device_id=device_id,
        self_role="admin",
        self_capabilities=sorted(types.capabilities_for_role("admin")),
    )
    record.members.append(
        types.MemberRecord(
            device_id=device_id,
            public_key=wire.b64u(b"a" * 32),
            role="admin",
            capabilities=sorted(types.capabilities_for_role("admin")),
            added_via="self",
        )
    )
    return record


def _row(
    device_id: str, *, role: str = "drive", endpoints: list[str] | None = None
) -> dict[str, Any]:
    return types.MemberRecord(
        device_id=device_id,
        public_key=wire.b64u(b"b" * 32),
        name="laptop",
        role=role,
        capabilities=sorted(types.capabilities_for_role(role)),
        endpoints=list(endpoints or []),
        added_via="invite",
    ).to_json()


def test_a_member_a_peer_knows_is_adopted() -> None:
    record = _record(SELF)
    changed, added = relay.adopt_members(record, [_row(PEER)])
    assert changed is True
    assert added == [PEER]
    assert record.member(PEER) is not None


def test_a_tombstoned_device_is_never_revived_by_a_peers_table() -> None:
    """Removal is the epoch path's decision. A peer holding a stale snapshot must
    not be able to undo it — otherwise `member rm` lasts until the removed device
    next asks somebody for their table."""
    record = _record(SELF)
    record.removed_ids.append(PEER)
    changed, added = relay.adopt_members(record, [_row(PEER)])
    assert (changed, added) == (False, [])
    assert record.member(PEER) is None


def test_a_row_we_already_hold_keeps_its_local_authority() -> None:
    """Authority is not a claim a peer gets to make about a third device. What a
    peer CAN relay is where that device said it could be reached."""
    record = _record(SELF)
    record.members.append(
        types.MemberRecord(
            device_id=PEER,
            public_key=wire.b64u(b"x" * 32),
            role="read",
            capabilities=sorted(types.capabilities_for_role("read")),
        )
    )
    changed, added = relay.adopt_members(
        record, [_row(PEER, role="admin", endpoints=["10.0.0.9:9", "10.0.0.9:10"])]
    )
    held = record.member(PEER)
    assert held is not None
    assert (changed, added) == (True, [])
    assert held.role == "read"
    assert set(held.capabilities) == set(types.capabilities_for_role("read"))
    assert held.public_key == wire.b64u(b"x" * 32)
    assert held.endpoints == ["10.0.0.9:9", "10.0.0.9:10"]


def test_a_peers_older_table_never_shrinks_ours() -> None:
    """A device that has not seen the newcomer is the normal case, and shrinking
    the table on the weaker evidence is how a mesh loses members that still are
    members."""
    record = _record(SELF)
    record.members.append(
        types.MemberRecord(device_id=PEER, public_key=wire.b64u(b"b" * 32), role="drive")
    )
    changed, added = relay.adopt_members(record, [_row(SELF)])
    assert (changed, added) == (False, [])
    assert record.member(PEER) is not None


# ---------------------------------------------------------------------------
# The one row an incoming table may RETIRE — a proven key rotation
# ---------------------------------------------------------------------------


def _rotated(
    record: types.NetworkRecord, root: Path
) -> tuple[types.MemberRecord, dict[str, Any], dict[str, Any], Any, Any]:
    """A real rotation, held the way a peer holds it.

    Returns the row this record already holds for the device, the row the ROTATED
    DEVICE publishes for itself, the statement that joins them, and the two
    identities — because a made-up signature would pin the shape of the check rather
    than the check itself, and a test that needs a second statement signed for
    another network needs the old private half to build one.

    Built from the product's own identity code: a minted keypair, a real ``rotate()``,
    and the statement signed by the OLD key.
    """
    lane = root / "rotating"
    old = identity.mint(lane, name="laptop")
    new, _previous = identity.rotate(lane, name="laptop")
    held = types.MemberRecord(
        device_id=old.device_id,
        public_key=old.public_key,
        name="laptop",
        role="read",
        capabilities=sorted(types.capabilities_for_role("read")),
        added_via="invite",
    )
    record.members.append(held)
    published = {
        # The row the ROTATED DEVICE publishes: its own id and key, the ids it came
        # from, the statement that proves the hop — and a role it does not hold here.
        "device_id": new.device_id,
        "public_key": new.public_key,
        "name": "laptop",
        "role": "admin",
        "capabilities": sorted(types.capabilities_for_role("admin")),
        "previous_ids": [old.device_id],
        "rotation_proof": identity.rotation_statement(old, new, record.network_id),
        "rotated_at": time.time(),
        "added_via": "invite",
    }
    return held, published, dict(published["rotation_proof"]), old, new


def test_a_row_carrying_the_rotation_statement_retires_the_row_it_succeeded(root: Path) -> None:
    """ONE DEVICE, ONE ROW: the row is retired, not left beside the row it succeeded.

    The statement is what makes the retirement lawful — it is signed by the key THIS
    record holds for the row being retired — and it is the same check the live
    ``net_identity_rotate`` frame makes, applied to a statement that arrived as part
    of a member table. Without the retirement the peer keeps two active rows for one
    device and reports one more member than that device does (QA round 16, Q16-1).

    What is asserted beyond the count is that the retirement is a SUPERSESSION and
    not a removal: the old id still resolves, nothing is burned, no row is
    tombstoned, and the standing this record resolved at admission stands.
    """
    record = _record(SELF)
    held, published, statement, _old, _new = _rotated(record, root)

    changed, added = relay.adopt_members(record, [published])

    assert (changed, added) == (True, [published["device_id"]])
    surviving = record.member(published["device_id"])
    assert surviving is not None
    assert record.member(held.device_id) is surviving, "the retired id must still resolve"
    assert [row.device_id for row in record.active_members()] == [SELF, published["device_id"]]
    assert held.device_id not in record.removed_ids, "a rotation is not a removal"
    assert all(row.removed_at is None for row in record.members)
    assert surviving.role == "read", "a peer does not get to promote the device it relays"
    assert surviving.added_via == "invite"
    assert surviving.rotation_proof == statement


@pytest.mark.parametrize("tamper", ["no_statement", "another_network", "edited_signature"])
def test_a_rotation_claim_this_record_cannot_verify_retires_nothing(
    root: Path, tamper: str
) -> None:
    """A re-identification this record cannot verify must not move a row.

    THE MECHANISM THIS GUARDS. The merge's rows are relayed claims, and a row that
    names one of OUR rows as its past self is the one case where a merge could retire
    a member. Believed on a bare claim, any member could hide a third device: the row
    goes, and since ``record.member()`` resolves the retired id through the successor,
    that id would answer with the claimant's key — so the member's own frames stop
    matching its own row. So the claim is refused in BOTH directions, nothing retired
    and nothing adopted, because adopting it unretired is the two-active-rows
    divergence this rule exists to close.

    The three tampered forms are the ones that look like proof: none at all, a real
    statement signed for ANOTHER network (a statement names the network it rotates
    within, so it is not usable anywhere), and a real statement whose signature has
    been edited.
    """
    record = _record(SELF)
    held, published, statement, old, new = _rotated(record, root)
    if tamper == "no_statement":
        published.pop("rotation_proof")
    elif tamper == "another_network":
        published["rotation_proof"] = identity.rotation_statement(
            old, new, "n_ffffffffffffffffffffffff"
        )
    else:
        published["rotation_proof"] = {**statement, "sig_old": wire.b64u(b"0" * 64)}

    changed, added = relay.adopt_members(record, [published])

    assert (changed, added) == (False, []), f"{tamper} moved a row"
    assert record.member(held.device_id) is held
    assert [row.device_id for row in record.active_members()] == [SELF, held.device_id]
    assert record.member(published["device_id"]) is None, f"{tamper} was adopted anyway"


# ---------------------------------------------------------------------------
# The pull, across two real relays
# ---------------------------------------------------------------------------


def _add_member(root: Path, network_id: str, device_id: str) -> None:
    """Give ``root`` a member row for a device, as an admission would."""
    record = store.load(network_id, root)
    relay.admit(
        record,
        device_id=device_id,
        public_key=wire.b64u(b"n" * 32),
        name="newcomer",
        role="drive",
        capabilities=sorted(types.capabilities_for_role("drive")),
        added_by=record.self_device_id,
        added_via="invite",
        root=root,
        persist=True,
    )


def test_the_dialler_learns_the_member_table_from_the_peer_it_dialled(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q-R2-1, dialling half: a member admitted after this device joined becomes
    visible to it as soon as it contacts anybody — no restart, no epoch."""
    server_a, server_b, host_a, port_a = peer_pair
    record, _host, _port = _pair(peer_pair, monkeypatch, role="admin")
    _add_member(server_a.root, record.network_id, NEWCOMER)
    assert store.load(record.network_id, server_b.root).member(NEWCOMER) is None

    link, reason = server_b.dial(record.network_id, host=f"{host_a}:{port_a}", epoch=record.epoch)
    assert link is not None, reason
    learned = store.load(record.network_id, server_b.root)
    newcomer_row = learned.member(NEWCOMER)
    assert newcomer_row is not None
    assert newcomer_row.name == "newcomer"
    link.close("test")


def test_the_listener_learns_the_member_table_from_the_peer_that_dialled_it(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q-R2-1, listening half: this device was dialled, so nothing here ASKED.

    Without this the pull only runs on the side that dialled, and a dial-only
    device — the one that must be dialled to be reached at all — would never
    learn anything."""
    server_a, server_b, host_a, port_a = peer_pair
    record, _host, _port = _pair(peer_pair, monkeypatch, role="admin")
    _add_member(server_b.root, record.network_id, NEWCOMER)
    assert store.load(record.network_id, server_a.root).member(NEWCOMER) is None

    link, reason = server_b.dial(record.network_id, host=f"{host_a}:{port_a}", epoch=record.epoch)
    assert link is not None, reason
    # The listener's pull runs on its own handshake thread — `dial` returns when
    # THIS side's handshake is done, and the peer may not have finished its end — so
    # this waits on the EVENT (the row appearing), not on a fixed sleep, and the
    # bound is generous because the thing being waited for is a loopback round trip
    # that competes with every other worker. The assertion below it is the test: if
    # the row never lands, the pull did not happen.
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if store.load(record.network_id, server_a.root).member(NEWCOMER) is not None:
            break
        time.sleep(0.05)
    learned = store.load(record.network_id, server_a.root)
    assert learned.member(NEWCOMER) is not None, "the listener never pulled the table"
    link.close("test")


def test_learning_a_member_is_audited(peer_pair: Devices, monkeypatch: pytest.MonkeyPatch) -> None:
    """A membership change an operator cannot see is a membership change they
    discover by surprise: `lop network log` is where this one has to land."""
    server_a, server_b, host_a, port_a = peer_pair
    record, _host, _port = _pair(peer_pair, monkeypatch, role="admin")
    _add_member(server_a.root, record.network_id, NEWCOMER)

    link, reason = server_b.dial(record.network_id, host=f"{host_a}:{port_a}", epoch=record.epoch)
    assert link is not None, reason
    events = [
        row
        for row in server_b.audit.tail(50, network_id=record.network_id)
        if row.get("event") == "membership_learned"
    ]
    assert events, "the pull recorded nothing"
    assert NEWCOMER in events[0]["detail"]["added"]
    link.close("test")
