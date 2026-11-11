"""Membership convergence, and the three surfaces that reported it wrong.

THIS FILE EXISTS BECAUSE ROUND 3 PROVED CONVERGENCE IN A TOPOLOGY THAT COULD NOT
EXPRESS THE BUG. Round 2's fix re-pulled the member table at link establishment,
and a three-relay loopback proof showed agreement — because in that proof a link
was always being established. On real hardware (QA round 3) the links were up
before the newcomer was admitted, a healthy pair holds its link open indefinitely
(``wire.KEEPALIVE_S`` against ``LINK_IDLE_S``), and so no trigger ever fired:
`members: 3` against `4` for four minutes on the dial-only Mac AND on an in-VPC
peer that the admitting device could reach directly.

So the shape below is the real one, and it is deliberately hostile to the fix that
failed:

* FOUR relays: ``c`` is the only one that accepts inbound links; ``a`` is dial-only
  (it advertises loopback and is never dialled), ``b`` and ``d`` declare nothing at
  all — the NAT'd pair, exactly as qa-mac2 was.
* EVERY LINK THAT EXISTS IS ESTABLISHED BEFORE THE LATE ADMISSION, and none is
  established afterwards, so a fix that only fires on establishment cannot pass.
* ``d`` is admitted by ``b`` and is reachable only through ``b``. ``b`` has no link
  to ``a`` (neither can dial the other), and ``c`` — ``a``'s only neighbour — has no
  link to ``d``. So ``a``'s only route to the knowledge that ``d`` exists is a
  two-hop table transfer: ``b`` → ``c`` → ``a``.
* ``a`` is a leaf that can only ever ask ``c``, so its own report must say that it
  verified with one of three peers rather than presenting its count as checked.

The tests also pin the three smaller round-3 findings on the same code path: a
removed device learning it was removed, a pairing refusal that names its remedy,
and a wedged relay that is not reported as an absent one.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from argparse import Namespace
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

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
    refusal_from_pairing,
    sas_matches,
)
from tests.unit.network import conftest as net_fixtures

NETWORK_NAME = "mesh-r4"


# ---------------------------------------------------------------------------
# The four-device harness, driven the way the CLI drives it
# ---------------------------------------------------------------------------

#: HOW A ONE-HOST RIG EXPRESSES THE TWO KINDS OF UNREACHABLE DEVICE, because
#: `advertised_endpoints` always names a real port for a loopback listener and this
#: host resolves no non-loopback address for its own hostname. What matters to the
#: mesh is the observable: whether a peer has an address it can dial.
#:
#: * ``hub`` — dialable. The device every other device joins through.
#: * ``dial_only`` — the documented `--listen-address 127.0.0.1` device, with the
#:   address it declares made undialable. That is the Mac from the QA run, viewed
#:   from a peer: the row said `127.0.0.1:4300`, which on the PEER's host is itself.
#: * ``silent`` — declares NOTHING, so its row carries no endpoint at all and every
#:   peer's dial gets `no_endpoint`. That is qa-mac2, behind a NAT, whose observed
#:   address in the real run (`66.23.24.219:50821`) answered nobody.
#:
#: ``silent`` DECLARES NOTHING BY CONSTRUCTION RATHER THAN BY ASSUMPTION ABOUT THE
#: HOST. `advertise_endpoints` answers for a ``0.0.0.0`` listener with the resolved
#: addresses of ``socket.gethostname()``, so whether this mode's row carried a
#: dialable address depended on the runner: on CI's name resolution it did, here it
#: does not (`getaddrinfo` raises, and the `OSError` branch returns no host). A
#: sibling that reads that row dials it, the link forms, and "this leaf can ask only
#: one of its peers" stops being the state under test — the CI shard 0 failure in
#: `test_a_member_count_reports_what_it_could_and_could_not_verify`. The `devices`
#: fixture below enforces the absence at the source instead of reading the host.
HUB = "hub"
DIAL_ONLY = "dial_only"
SILENT = "silent"


class Device:
    """One install: its own root, identity, relay and record."""

    def __init__(self, tmp_path: Path, name: str, *, mode: str) -> None:
        assert mode in (HUB, DIAL_ONLY, SILENT)
        self.name = name
        self.mode = mode
        self.root = tmp_path / name
        self.identity = identity.mint(self.root, name=name)
        listen = "0.0.0.0" if mode == SILENT else "127.0.0.1"
        self.server = relay.RelayServer(
            root=self.root,
            settings=relay.NetworkSettings(port=0, listen_address=listen),
            identity=self.identity,
            audit=audit_mod.AuditLog(self.root),
        )
        self.host, self.port = self.server.bind()
        if mode == DIAL_ONLY:
            # THE LISTENER STAYS BOUND AND THE DECLARED PORT BECOMES 0, which is how
            # this rig says "the address you can read in my row is not one you can
            # dial" — the single-host analogue of a loopback address seen from
            # another machine. Applied to the settings BEFORE `start()`, so the self
            # row and every peer's copy are written with the undialable address.
            self.server.settings = replace(self.server.settings, port=0)
        self.server.bind_control()
        self.server.start()

    @property
    def device_id(self) -> str:
        return self.identity.device_id

    def stop(self) -> None:
        self.server.stop()


@pytest.fixture()
def devices(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The relays a test builds, with each mode's DOCUMENTED reachability ENFORCED.

    A mode is a claim about where a peer can reach a device, and this rig may not
    leave that claim to the machine it runs on. ``advertise_endpoints`` answers
    "where do peers reach us" from a device that listens on ``0.0.0.0`` by
    resolving ``socket.gethostname()`` — runner-dependent by nature — so the mode
    the module docstring calls ``silent`` ("declares NOTHING ... every peer's dial
    gets ``no_endpoint``") carried a dialable address on a host whose name
    resolves, and none on a host where it does not. The failure that produces is
    not a fixture detail: the peer that reads the row dials it, the link forms, and
    the topology the test is ABOUT (a leaf that can ask only one of its two peers)
    is gone before the test reads it — which is exactly the CI shard 0 failure in
    ``test_a_member_count_reports_what_it_could_and_could_not_verify``.

    So the meaning is enforced at the source of the answer rather than inferred
    from the hostname: a ``silent`` device declares nothing on EVERY path that
    publishes an endpoint (its own member row, the hello it sends a peer, and the
    copy that peer stores for it), because all of them go through this function.
    Every other mode gets the real answer unchanged.
    """
    built: list[Device] = []
    original = relay.advertise_endpoints

    def _advertise(settings: relay.NetworkSettings, *, declared: Any = ()) -> list[str]:
        for device in built:
            if device.mode == SILENT and device.server.settings is settings:
                return []
        return original(settings, declared=declared)

    monkeypatch.setattr(relay, "advertise_endpoints", _advertise)
    try:
        yield built, tmp_path
    finally:
        for device in built:
            device.stop()


def _make(devices: Any, name: str, *, mode: str = HUB) -> Device:
    built, tmp_path = devices
    device = Device(tmp_path, name, mode=mode)
    built.append(device)
    return device


def _init_network(server: relay.RelayServer) -> types.NetworkRecord:
    """Create a network on the hub, exactly as `lop network init` does."""
    from secrets import token_bytes

    record = types.NetworkRecord(
        network_id=store.new_network_id(),
        name=NETWORK_NAME,
        created_by=server.identity.device_id,
        self_device_id=server.identity.device_id,
        self_role="admin",
        self_capabilities=sorted(types.capabilities_for_role("admin")),
        listen={"address": "127.0.0.1", "port": server.settings.port, "advertised": []},
    )
    state = types.SecretState(
        network_id=record.network_id, epoch=1, secret=wire.b64u(token_bytes(32))
    )
    relay.admit(
        record,
        device_id=server.identity.device_id,
        public_key=server.identity.public_key,
        name=server.identity.name,
        role="admin",
        capabilities=sorted(types.capabilities_for_role("admin")),
        added_by=server.identity.device_id,
        added_via="self",
        root=server.root,
        persist=False,
    )
    store.save(record, server.root)
    store.save_secrets(state, server.root)
    return record


def _mint_invite(
    server: relay.RelayServer, record: types.NetworkRecord, *, role: str = "admin"
) -> tuple[str, Any]:
    state = store.load_secrets(record.network_id, server.root)
    minted = invite_mod.mint(record, state.secret, role=role, ttl_s=600.0)
    record.invites.append(minted.record)
    store.save(record, server.root)
    store.save_invite_token(minted.record.invite_id, minted.token, server.root)
    return minted.token, minted.envelope


def _type_the_code(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(net_cli, "_read_code", lambda args, derived, fingerprint: derived)


def _answer_confirmation(
    server: relay.RelayServer, stop: threading.Event, answered: list[str], *, interval: float = 0.05
) -> None:
    """Answer the inviter's parked prompt for as long as the ceremony is running.

    EVENT-DRIVEN, NOT CLOCK-DRIVEN, and that is the fix for a real failure rather than
    a test convenience: this used to give up silently after 20 s, and when it did — a
    loaded host starving the thread — the invite's human step ran on to its own expiry
    and the ceremony ended with ``timeout``. A CORRECT REFUSAL REPORTED AS THE WRONG
    ONE is worse than a slow test: the round-4 review measured it as
    ``assert 'timeout' == 'device_id_conflict'`` in 2 runs of 5. Polling until the
    caller says the ceremony is over means a slow host costs latency and nothing else.
    """
    while not stop.is_set():
        rows = server._ctl_pair_pending({})  # noqa: SLF001 — the CLI's own control op
        for row in rows:
            server._ctl_pair_confirm(  # noqa: SLF001
                {
                    "invite_id": row["invite_id"],
                    "decision": "admit",
                    "matched": True,
                    "reason": "",
                    "answered_by": "harness",
                }
            )
            answered.append(str(row["invite_id"]))
        stop.wait(interval)


def _join(
    device: Device,
    *,
    inviter: Device,
    monkeypatch: pytest.MonkeyPatch,
    settings: relay.NetworkSettings | None = None,
    confirm: bool = True,
    role: str = "admin",
) -> dict[str, Any]:
    """The whole pairing: the joiner's code typed, and optionally the inviter's human.

    ``LOCAL_OPERATOR_CONFIG_DIR`` is pointed at the JOINER's root for the call,
    because the joining half of the CLI resolves its config root from the ambient
    environment (that is how a user runs it) and would otherwise write the record
    into the isolated HOME instead of the device's own store.

    ``confirm=False`` runs it with NOBODY at the inviting device at all. That is how a
    refusal the inviter can decide LOCALLY is tested: it must arrive without a human
    being asked, so a host that starves this process cannot turn it into the window's
    expiry instead of the reason.
    """
    record = store.load(store.list_networks(inviter.root)[0].network_id, inviter.root)
    token, envelope = _mint_invite(inviter.server, record, role=role)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(device.root))
    _type_the_code(monkeypatch)
    stop = threading.Event()
    answered: list[str] = []
    thread = threading.Thread(
        target=_answer_confirmation,
        args=(inviter.server, stop, answered),
        daemon=True,
    )
    if confirm:
        thread.start()
    try:
        args = Namespace(sas_stdin=True, verify=False, emit_sas=True, name=device.name, json=True)
        answer = net_cli._join_one(  # noqa: SLF001 — the CLI's own driver
            host=f"{inviter.host}:{inviter.port}",
            token=token,
            envelope=envelope,
            identity=device.identity,
            settings=settings or device.server.settings,
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
    finally:
        stop.set()
        if confirm:
            thread.join(30)
    # AN ADMITTED PAIRING WENT THROUGH THE HUMAN STEP: the local refusals bypass a
    # question nobody needs to answer, and a legitimate join must not.
    if confirm and isinstance(answer, tuple):
        assert answered, "the pairing was admitted without a confirmation ever being parked"
    # A join that failed at the transport gives back a SENTENCE, not a payload: fail
    # with that sentence rather than with a KeyError on a dict that was never built.
    assert isinstance(answer, tuple), answer
    _lines, payload = answer
    return payload


def _members(device: Device) -> set[str]:
    record = store.load(store.list_networks(device.root)[0].network_id, device.root)
    return {row.device_id for row in record.active_members()}


def _events(device: Device) -> list[str]:
    return [str(row.get("event")) for row in device.server.audit.tail(limit=500)]


def _recorded(
    device: Device, word: str, *, timeout_s: float = net_fixtures.NET_WAIT_BUDGET_S
) -> net_fixtures.WaitResult:
    """Wait until ``device``'s own trail carries ``word``; report whether it arrived.

    THE TRAIL IS WRITTEN BY WHICHEVER THREAD DID THE ACT, AND AFTER THE ACT. The
    listener that refuses a pairing seals and sends its abort frame BEFORE it records
    ``pairing_refused`` (relay.py, the refusal branch) — so the joiner's exception,
    which is the effect the test already holds, arrives strictly first. A single-shot
    read of the trail right after it therefore assumes a scheduling gap of ZERO; a
    shard runner failed this cell on 1 ms of one (CI, head 3d19009e6: the events list
    ended at ``trust_changed`` and no ``pairing_refused`` ever appeared), and 1 ms is
    well inside the 525-668 ms starvation gaps this fleet measures. The bound is the
    backstop for a row that genuinely never arrives — which then fails the assertion
    below with the whole trail in the message, and it is the SHARED backstop
    (``net_fixtures.NET_WAIT_BUDGET_S``): the row is not shorter-latency by construction,
    so this helper must not carry a private 15 s of its own.
    """
    return net_fixtures.wait_for(lambda: word in _events(device), timeout_s=timeout_s)


def _link_peers(device: Device) -> set[str]:
    return {
        link.device_id
        for link in device.server.links.values()  # noqa: SLF001 — the relay's own table
        if link.alive
    }


# ---------------------------------------------------------------------------
# Q-R2-1 — the shape that broke round 3's fix
# ---------------------------------------------------------------------------


def test_membership_converges_over_links_that_were_already_up(
    devices: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The blocker: a late member becomes visible with nothing restarted.

    The links are A–C and B–C, both established before D is admitted, and neither is
    established again. The convergence that follows therefore cannot be a
    link-establishment side effect: ``a`` never dials anything after D joins (the
    test asserts that, rather than hoping), and ``a``'s only neighbour never holds a
    link to D at all.
    """
    # A COMPRESSED CADENCE, NOT A STORM. The product's interval is 15 s; asking for
    # a table every 200 ms is what this test can afford to WAIT, not a load any real
    # mesh sees. Ten pulls a second, from four relays in one process on a host that
    # runs ~25 sessions, starves the loops and makes the window a coin toss (measured:
    # every relay silent for >4 s at a time, which is the host, not the code) — and a
    # test whose verdict depends on the host's load decides nothing.
    monkeypatch.setattr(relay, "MEMBERSHIP_PULL_MIN_INTERVAL_S", 1.0)
    # ...and the CEILING too: the backoff would otherwise climb to the real 30 s
    # inside this test and its own window would expire waiting for a pull.
    monkeypatch.setattr(relay, "MEMBERSHIP_PULL_MAX_INTERVAL_S", 2.0)
    monkeypatch.setattr(relay, "MEMBERSHIP_PUSH_PULL_INTERVAL_S", 2.0)
    monkeypatch.setattr(relay, "MEMBERSHIP_PULL_PASS_S", 0.25)

    # This case proves CORE membership convergence over a fixed graph. Redial
    # and sibling slices may legitimately replace links and repair a stale table
    # through a fresh handshake, which would make this causal proof meaningless.
    # Gate those unrelated producers BEFORE any relay starts; the real membership
    # pull/push loops, authenticated sockets, store and audit remain unchanged.
    def no_redial(self: relay.RelayServer) -> None:
        self._stop.wait()

    monkeypatch.setattr(relay.RelayServer, "_redial_loop", no_redial)
    monkeypatch.setattr(relay.RelayServer, "_install_slices", lambda self: None)

    a = _make(devices, "a", mode=DIAL_ONLY)
    b = _make(devices, "b", mode=SILENT)
    c = _make(devices, "c", mode=HUB)
    d = _make(devices, "d", mode=SILENT)

    # C is the hub every device joins through, and the only one that accepts inbound.
    record = _init_network(c.server)
    for joiner in (b, a):
        assert _join(joiner, inviter=c, monkeypatch=monkeypatch)["device_id"] == joiner.device_id

    # THE LINKS THAT WILL STILL EXIST WHEN THE LATE MEMBER ARRIVES: A–C and B–C.
    link, reason = a.server.dial(record.network_id, host=f"{c.host}:{c.port}", epoch=1)
    assert link is not None, reason
    link, reason = b.server.dial(record.network_id, host=f"{c.host}:{c.port}", epoch=1)
    assert link is not None, reason
    # A CONTACT ON A LINK THAT ALREADY EXISTS ESTABLISHES NOTHING, which is the
    # mechanism round 2's fix rested on and the reason it could never run again: it
    # pulled the table when a link was REGISTERED, and with every link already up
    # there was no registration left to trigger. Asserted rather than assumed.
    live_links = [item.link_id for item in a.server.links.values() if item.alive]  # noqa: SLF001
    same, _why = a.server._ensure_link_with_reason(c.device_id)  # noqa: SLF001
    assert same is not None and same.link_id in live_links
    after = [item.link_id for item in a.server.links.values() if item.alive]  # noqa: SLF001
    assert after == live_links

    initial = {a.device_id, b.device_id, c.device_id}

    def initial_settled() -> bool:
        if not all(_members(device) == initial for device in (a, b, c)):
            return False
        # Link admission precedes its first table read. Finish those reads before
        # admitting D so an in-flight registration pull cannot prove convergence.
        return all(
            (peer_link := owner.server._link_for(peer_id)) is not None
            and peer_link.member_pulled_at > 0
            for owner, peer_id in (
                (a, c.device_id),
                (b, c.device_id),
                (c, a.device_id),
                (c, b.device_id),
            )
        )

    assert net_fixtures.wait_for(initial_settled), {
        device.name: _members(device) for device in (a, b, c)
    }
    assert _members(a) == initial
    assert _link_peers(a) == {c.device_id}
    assert _link_peers(b) == {c.device_id}

    # FROM HERE NOTHING IS TOUCHED, NOBODY IS RESTARTED, AND A IS WATCHED FOR DIALS.
    dials: list[str] = []
    original = a.server.dial

    def _spy(network_id: str, **fields: Any) -> Any:
        dials.append(str(fields.get("host") or ""))
        return original(network_id, **fields)

    # Cached-link lookups and no-endpoint checks are NOT socket dials. Observe
    # actual connection attempts, including ones made during the late admission.
    monkeypatch.setattr(a.server, "dial", _spy)

    # B admits D after the existing topology is settled and the dial spy is
    # installed. A/C may learn D BEFORE join() returns: early convergence is not
    # a failure and must not be asserted away as an old three-member snapshot.
    assert _join(d, inviter=b, monkeypatch=monkeypatch)["device_id"] == d.device_id
    expected = initial | {d.device_id}
    agreement = net_fixtures.wait_for(lambda: _members(a) == _members(c) == _members(b) == expected)
    actual_a, actual_c = _members(a), _members(c)
    assert actual_a == actual_c == expected and agreement, {
        "agreement_wait": agreement,
        "a": actual_a,
        "c": actual_c,
        "expected": expected,
    }
    # NOTHING WAS DIALED TO MAKE THIS HAPPEN — the refresh runs on the links that
    # exist, which is the property that makes it independent of a mesh's shape.
    assert dials == [], dials
    current_link = a.server._link_for(c.device_id)
    assert current_link is not None and current_link.link_id in live_links
    assert _link_peers(a) == {c.device_id}
    assert d.device_id not in _link_peers(c)

    # And the device that learned it says so in its audit, naming its source.
    #
    # THIS READS THE ROW, SO IT WAITS FOR THE ROW. ``refresh_membership`` saves the
    # merged record and only THEN records ``membership_learned``, so the merge the loop
    # above polls is visible while the row is still pending on the pulling thread — and
    # the predicate above is that merge, not this record. Measured with a delay injected
    # at ``AuditLog.record``: 1 s reds this cell at the old ``learned[-1]`` with
    # ``assert 'd_…' in []``, against the 525-668 ms starvation gaps this fleet records
    # with the loop idle.
    #
    # The predicate is the ASSERTION'S OWN condition — the row that names D as added —
    # rather than "some row appeared", which a pull that changed nothing also satisfies
    # (``adopt_members`` reports ``changed`` without ``added``).
    def _rows_naming_d() -> list[dict[str, Any]]:
        return [
            row
            for row in a.server.audit.tail(limit=500)  # noqa: SLF001 — the relay's own log
            if row.get("event") == "membership_learned"
            and d.device_id in (row.get("detail") or {}).get("added", [])
        ]

    assert net_fixtures.wait_for(
        lambda: bool(_rows_naming_d())
    ), f"the device that learned the late member recorded no row naming it: {_events(a)}"
    learned = _rows_naming_d()
    # The trail is read AGAIN rather than reused from the predicate: the pulls here run
    # on a compressed cadence (1 s), so a row describing a later no-op merge can land
    # between the wait and this assertion, and the assertion is about D's row.
    assert learned[-1]["detail"]["source"] == c.device_id


def test_a_member_count_reports_what_it_could_and_could_not_verify(
    devices: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of Q-R2-1: the count is never bare.

    A leaf that can only ask one of its two peers must say exactly that, and it
    must name the peers it could not ask — an incomplete table presented as
    authoritative is worse than an error, because an operator acts on it.
    """
    monkeypatch.setattr(relay, "MEMBERSHIP_PULL_MIN_INTERVAL_S", 1.0)
    # ...and the CEILING too: the backoff would otherwise climb to the real 30 s
    # inside this test and its own window would expire waiting for a pull.
    monkeypatch.setattr(relay, "MEMBERSHIP_PULL_MAX_INTERVAL_S", 2.0)
    monkeypatch.setattr(relay, "MEMBERSHIP_PUSH_PULL_INTERVAL_S", 2.0)
    monkeypatch.setattr(relay, "MEMBERSHIP_PULL_PASS_S", 0.25)

    a = _make(devices, "a", mode=DIAL_ONLY)
    b = _make(devices, "b", mode=SILENT)
    c = _make(devices, "c", mode=HUB)

    # THERE MUST BE NO LINK BETWEEN A AND B, IN EITHER DIRECTION, and the topology
    # this test is ABOUT rests on that rather than on an optimisation: "this leaf
    # can ask only one of its two peers" is a claim about which links exist, so a
    # link that forms on its own makes the assertions below read the race instead of
    # the rule.
    #
    # Direction A → B is forced by the `devices` fixture, which makes `b` declare no
    # endpoint at all, so `a`'s contact path (`_ctl_ls` → `contact_peers`) dials the
    # row it holds for `b` and gets `no_endpoint` — never a link. That is the
    # direction CI failed in, and it is why the row `a` reads for `b`, not `b`'s
    # behaviour, is what had to become deterministic.
    #
    # Direction B → A is held at the dial path, which is the one choke point every
    # route to a link shares. It is the belt to the fixture's braces (`a` advertises
    # only loopback): it makes "b cannot reach a either" a written fact rather than
    # an inference from an address that happens to carry port 0. Installed before any
    # network exists, so there is no window in which the link could already be up.
    original_dial = b.server._ensure_link_with_reason  # noqa: SLF001 — the dialling path

    def _hold_a(device_id: str, **fields: Any) -> Any:
        if device_id == a.device_id:
            return None, "test_fixture: this device is held unreachable"
        return original_dial(device_id, **fields)

    b.server._ensure_link_with_reason = _hold_a  # type: ignore[method-assign]  # noqa: SLF001

    record = _init_network(c.server)
    assert _join(b, inviter=c, monkeypatch=monkeypatch)["device_id"] == b.device_id
    assert _join(a, inviter=c, monkeypatch=monkeypatch)["device_id"] == a.device_id
    link, reason = a.server.dial(record.network_id, host=f"{c.host}:{c.port}", epoch=1)
    assert link is not None, reason

    rows = a.server._ctl_ls({})  # noqa: SLF001 — the CLI's own control op
    row = next(item for item in rows if item["network_id"] == record.network_id)
    assert row["members"] == 3
    assert row["membership_state"] == "active"

    # THE FIXTURE'S PRECONDITION, ASSERTED RATHER THAN ASSUMED. If a link to `b`
    # exists by the time the report is read, then "b is the peer a cannot verify
    # with" is not the state under test — and the accidental dial that forms it is
    # `a`'s own, over the row it holds for `b`, so that is the line the fixtures
    # above have to close. It shows up here, by name, instead of as a cryptically
    # longer ``answered`` list three assertions later.
    assert (
        a.server._link_for(b.device_id) is None
    ), "the leaf gained a link to its unverifiable peer"

    # A lone leaf CANNOT verify with the peer it has no link to, and says so.
    reports = a.server.refresh_membership()  # noqa: SLF001
    report = reports[record.network_id]
    assert report.answered == [c.device_id]
    assert [entry["device_id"] for entry in report.silent] == [b.device_id]
    assert report.complete is False
    assert "verified with 1 of 2 peer(s)" in report.sentence()

    # ASKING AGAIN IMMEDIATELY IS NOT A GAP: a link that answered inside the cadence
    # is reported as answered-with-an-age, not as "no peer answered" — a `show` one
    # second after a refresh must not claim the table was never checked.
    again = a.server.refresh_membership()[record.network_id]  # noqa: SLF001
    assert again.answered == [c.device_id]
    assert again.not_due and again.not_due[0]["device_id"] == c.device_id
    assert again.complete is False  # b still has no link to ask over

    # The whole point of the block: the count and its provenance travel together.
    row = next(
        item
        for item in a.server._ctl_ls({})  # noqa: SLF001
        if item["network_id"] == record.network_id
    )
    assert row["membership"]["table"]["answered"] == [c.device_id]
    assert row["membership"]["table"]["complete"] is False


# ---------------------------------------------------------------------------
# Q-R3-2 — a removed device learns it was removed
# ---------------------------------------------------------------------------


def test_the_removal_frame_is_applied_by_the_device_it_removes(
    devices: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The frame that removes a device must be applicable BY that device.

    It carries the sender's new epoch, the full member list and our own tombstone.
    It used to be refused as ``self_absent_from_members`` — the consistency rule that
    stops a peer quietly ageing us out of our own network was applied to the one
    frame that says so out loud in its own ``removed`` array, so the removed device
    kept its old epoch, its old four-member table and ``trust: active`` while every
    attempt to reach the network failed as a transport error.
    """
    a = _make(devices, "a", mode=HUB)
    b = _make(devices, "b", mode=HUB)
    record = _init_network(a.server)
    assert _join(b, inviter=a, monkeypatch=monkeypatch)["device_id"] == b.device_id

    # A live link, so the rotation has somewhere to be delivered.
    link, reason = b.server.dial(record.network_id, host=f"{a.host}:{a.port}", epoch=1)
    assert link is not None, reason

    before = store.load_secrets(record.network_id, b.root).secret
    outcome = a.server._ctl_member_rm(  # noqa: SLF001 — the CLI's own control op
        {"network": record.network_id, "device_id": b.device_id}
    )
    assert outcome["epoch"] == 2
    assert net_fixtures.wait_for(
        lambda: store.load(record.network_id, b.root).epoch == 2, timeout_s=5.0
    ), "the removed device never applied the rotation that removed it"

    removed = store.load(record.network_id, b.root)
    assert removed.epoch == 2, "the removed device never applied the rotation that removed it"
    assert b.device_id in removed.removed_ids
    own_row = removed.member(b.device_id)
    assert own_row is not None
    assert own_row.active is False
    # ``removed_at`` is stamped only by the tombstone path, so its presence IS the
    # assertion — a removed row without the stamp is the defect, not a detail.
    assert own_row.removed_at is not None
    assert own_row.removed_at > 0
    # THE SECRET IS STILL WITHHELD — the ordering fix does not leak key material to
    # the device being evicted (§8.1, `epoch_secret_withheld_from_removed`).
    assert store.load_secrets(record.network_id, b.root).secret == before

    # And its OWN surface says it, by name, with a remedy — not "trust: active" and a
    # full member list beside a transport error.
    standing = relay.membership_state(removed)
    assert standing["state"] == "removed"
    assert "no longer a member" in standing["sentence"]
    assert any("identity rotate" in remedy for remedy in standing["remedies"])
    row = b.server._ctl_ls({})[0]  # noqa: SLF001
    assert row["membership_state"] == "removed"
    assert "active member" not in row["membership"]["sentence"]


def test_a_silently_refused_handshake_is_named_locally(
    devices: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The offline half of Q-R3-2: the refusal is silent on the wire by design.

    A peer that explains every refusal is an oracle, so a refused handshake closes
    the socket with no reply. The refused device therefore has to say what it OBSERVED
    — the peer accepted the connection and closed it mid-handshake — mark the network
    `refused_by_peers` with the design's own value, and name the states that look like
    this instead of reporting a bare transport error beside a healthy member list.
    """
    a = _make(devices, "a", mode=HUB)
    b = _make(devices, "b", mode=HUB)
    record = _init_network(a.server)
    assert _join(b, inviter=a, monkeypatch=monkeypatch)["device_id"] == b.device_id
    # THE DEVICE IS AWAY WHEN THE REMOVAL HAPPENS, and that is part of the scenario,
    # not setup colour. ``member rm`` broadcasts the rotation over every live link
    # (``_broadcast_epoch``: "Send net_epoch to every live link"), and the admitting
    # device owes the rest of the network a post-pairing contact (``contact_peers`` at
    # the tail of ``_run_pair_listener``) whose dial runs on the listener thread and
    # races the very next statement here. When that dial lands first the joiner is
    # reachable at broadcast time, the epoch frame reaches it, ``apply_epoch`` takes
    # its ``removing_us`` branch (a path this file pins in its own right), and
    # ``membership_state`` answers 'removed' by design — removed outranks refused —
    # so the assertion below would be about a device that was TOLD. Measured: CI shard
    # (3.12, 4), run 37851665431 first attempt, ``'removed' == 'refused'``; reproduced
    # on this fleet with a probe trace — contact_peers dial → broadcast send → b's
    # APPLY ``removed_by_this_rotation``. Q-R3-2's story is the device that was away
    # when the removal happened and came back to a silent refusal, so the device goes
    # away first: with the relay stopped no link can exist to receive the rotation
    # (``stop``'s post-condition closes every link and refuses a late dialer), which
    # makes the refusal the ONLY account of the removal b can hold.
    b.server.stop()
    a.server._ctl_member_rm(  # noqa: SLF001 — the CLI's own control op
        {"network": record.network_id, "device_id": b.device_id}
    )
    # ...AND IT COMES BACK: a fresh relay on the existing root — the durable state is
    # on disk, the process is not (the process that paired is not the process that
    # later dials). Dial-only, per the rule the fresh-relay helper in
    # ``test_relay_e2e`` states: nothing below needs a listener, and a test that
    # starts one more listener than it stops is a leak.
    b.server = relay.RelayServer(
        root=b.root,
        settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1"),
        identity=b.identity,
        audit=audit_mod.AuditLog(b.root),
    )

    refused = store.load(record.network_id, b.root)
    refused.stale = ""
    store.save(refused, b.root)

    link, reason = b.server.dial(record.network_id, host=f"{a.host}:{a.port}", epoch=1)
    assert link is None
    assert reason.startswith("handshake_refused:"), reason

    events = [
        row
        for row in b.server.audit.tail(limit=500)  # noqa: SLF001
        if row.get("event") == "handshake_refused"
    ]
    assert events, _events(b)
    assert events[-1]["detail"]["cause"] == "peer_closed_silently"

    marked = store.load(record.network_id, b.root)
    assert marked.stale == "refused_by_peers"
    standing = relay.membership_state(marked)
    assert standing["state"] == "refused"
    assert "refusing this device's handshakes" in standing["sentence"]
    assert any("trust" in remedy for remedy in standing["remedies"])

    # A COMPLETED HANDSHAKE CLEARS IT, so a peer that restarted mid-handshake does not
    # leave a permanent accusation. (The transition is a direct call because the
    # refusal that set it is a removal on the peer's side here, and a removed id
    # cannot be re-admitted — which is the next test.) It takes the network id, not
    # ``marked``: the mark is one field of a record other writers edit, so the helper
    # re-reads inside the store's lock rather than writing this copy back.
    b.server._clear_refusal_mark(marked.network_id)  # noqa: SLF001
    assert store.load(record.network_id, b.root).stale == ""


# ---------------------------------------------------------------------------
# Q-R3-3 — a refusal names its remedy
# ---------------------------------------------------------------------------


def test_a_pairing_refusal_carries_the_sentence_and_the_remedy(
    devices: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`device_id_conflict` on its own is a dead end.

    Measured: a removed device that re-joins with its existing identity is refused,
    and the joiner was told ``the pairing was refused (device_id_conflict)`` — the
    admitting device's own sentence (which named the burned id and the way out) was
    dropped one line before the wire. The way out is also NOT
    `lop network trust --active`, assert the second half of this test says so on the
    device that would run it: a removed id is burned and no trust change revives it,
    while a rotated identity is admitted.
    """
    a = _make(devices, "a", mode=HUB)
    b = _make(devices, "b", mode=HUB)
    record = _init_network(a.server)
    assert _join(b, inviter=a, monkeypatch=monkeypatch)["device_id"] == b.device_id

    a.server._ctl_member_rm(  # noqa: SLF001
        {"network": record.network_id, "device_id": b.device_id}
    )
    assert store.load(record.network_id, a.root).is_burned(b.device_id)

    # THE CLAIM QA'S REPORT MADE, TESTED: `trust --active` does not un-burn an id.
    a.server._ctl_trust(  # noqa: SLF001
        {"network": record.network_id, "trust": "active", "reason": "harness"}
    )
    assert store.load(record.network_id, a.root).is_burned(b.device_id)

    # NO HUMAN ON THE INVITING DEVICE AT ALL, and no clock to outlive: whether this
    # id is burned is a LOCAL fact the admitting device holds, so the refusal must
    # arrive on its own. This is what makes the case robust under load — the round-4
    # review measured `assert 'timeout' == 'device_id_conflict'` in 2 runs of 5 when
    # the refusal waited behind a confirmation window (and it is why the listener now
    # decides it before it asks: nobody should compare six digits for a pairing that
    # cannot succeed).
    started = time.monotonic()
    with pytest.raises(types.PairingRefusal) as excinfo:
        _join(b, inviter=a, monkeypatch=monkeypatch, confirm=False)
    elapsed = time.monotonic() - started
    sentence = excinfo.value.sentence
    assert excinfo.value.code == "device_id_conflict"
    assert b.device_id in sentence, sentence
    assert "identity rotate" in sentence, sentence
    assert "does not restore a removed member" in sentence, sentence
    # The refusal did not wait on the invite's confirmation window (180 s here).
    assert elapsed < 60.0, f"the refusal took {elapsed:.1f}s, which is a window expiring"
    # AND NOBODY WAS ASKED: the inviter parked no prompt for this refusal. Exactly one
    # confirmation was parked in this test — the admitted pairing above — so a burned
    # joiner reaching the human step (which is what the round-4 review caught, via its
    # expiry) shows up here as a second one.
    #
    # THE REFUSAL'S OWN ROW IS AWAITED, NOT ASSUMED: see `_recorded`. The frame that
    # carried this refusal reached the joiner above, and the inviter records the row
    # after sending it, so the read races the write by design.
    assert _recorded(a, "pairing_refused"), _events(a)
    events = _events(a)
    assert events.count("pairing_awaiting_confirmation") == 1, events
    assert "pairing_refused" in events, events

    # AND THE REMEDY WORKS: a fresh identity is admitted, with a fresh invite.
    new_identity, _old = identity.rotate(b.root)
    b.identity = new_identity
    b.server.identity = new_identity
    payload = _join(b, inviter=a, monkeypatch=monkeypatch)
    assert payload["device_id"] == new_identity.device_id
    assert payload["device_id"] == new_identity.device_id


def test_the_remedy_map_covers_every_reason_it_names() -> None:
    """Cheap, and it is the contract the joiner's screen depends on: an operator who
    is told a code and nothing else retries in the dark."""
    from local_operator.network.handshake import PAIRING_REMEDIES

    for reason, remedy in PAIRING_REMEDIES.items():
        refusal = refusal_from_pairing(reason, detail="")
        assert remedy in refusal.sentence
        assert refusal.code == reason


# ---------------------------------------------------------------------------
# The transport underneath all of it
# ---------------------------------------------------------------------------


def test_a_frame_queued_immediately_before_a_close_reaches_the_peer(
    devices: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``send`` then ``close`` must mean SENT then closed.

    ``send`` is asynchronous on purpose, so the writer drains the queue on its own
    schedule — and the close used to shut the socket down underneath it, discarding
    whatever was queued. The two frames that matter most are exactly the ones sent a
    line before a close: the ``net_epoch`` rotation `_rehandshake_network` pushes down
    every link to change keys, and the ``net_bye`` it queues above the close it
    announces. Measured: with no delay inserted, a rotation to epoch 2 never reached
    the peer; with a one-second delay inserted before the close it did. That is a
    membership revocation being lost in the socket buffer.
    """
    a = _make(devices, "a", mode=HUB)
    b = _make(devices, "b", mode=HUB)
    record = _init_network(a.server)
    assert _join(b, inviter=a, monkeypatch=monkeypatch)["device_id"] == b.device_id
    link, reason = b.server.dial(record.network_id, host=f"{a.host}:{a.port}", epoch=1)
    assert link is not None, reason

    # WATCHED ON THE RECEIVING LINK'S OWN DISPATCH, not on a class attribute: the
    # handler table is bound at construction, so patching the class would observe
    # nothing and the test would pass on a lost frame for the wrong reason.
    a_link = next(  # noqa: SLF001
        item for item in a.server.links.values() if item.link_id == link.link_id
    )
    seen: list[tuple[str, Any]] = []
    original = a_link._handle  # noqa: SLF001

    def _spy(frame: dict[str, Any]) -> None:
        seen.append((str(frame.get("op")), frame.get("req")))
        return original(frame)

    a_link._handle = _spy  # type: ignore[method-assign]  # noqa: SLF001
    link.send({"op": "net_member_list", "req": 424242, "locality": "remote"})
    link.close("we-closed")
    assert net_fixtures.wait_for(
        # Other membership pulls can already be in flight on this link. Only
        # this request proves the frame queued immediately before close arrived.
        lambda: ("net_member_list", 424242) in seen,
        timeout_s=3.0,
    ), "the frame queued immediately before the close never arrived: close beat the writer"

    assert (
        seen
    ), "the frame queued immediately before the close never arrived: close beat the writer"


# ---------------------------------------------------------------------------
# Q-R3-4 — a diagnostic may not contradict itself
# ---------------------------------------------------------------------------


def test_a_wedged_relay_is_running_and_says_it_is_not_answering(
    devices: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live pid with a stale heartbeat is a WEDGED relay, not an absent one.

    Measured (QA round 3): with the relay SIGSTOPped, `status --json` reported
    ``relay_running: false`` in a payload whose own ``record`` block carried
    ``pid 21094``, because the running-ness came from the control socket alone and
    the record from a live-only scan. This device IS the live pid, and the record's
    heartbeat is old — the same reading, without needing to stop a process.
    """
    from local_operator.network.types import PeerRecord

    _built, tmp_path = devices
    root = tmp_path / "wedged"
    root.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    path = store.publish_peer_record(
        PeerRecord(
            pid=os.getpid(),
            device_id="d_" + "a" * 32,
            control_port=1,  # nothing answers there, which is the wedge
            control_key=wire.b64u(b"0" * 32),
        ),
        root,
    )
    # ``publish`` re-stamps the heartbeat of the process that owns the record, which is
    # this one — so the stale stamp a WEDGED relay would have is written afterwards,
    # into the record file itself.
    stale = json.loads(path.read_text())
    stale["heartbeat_at"] = time.time() - 3600.0
    path.write_text(json.dumps(stale))

    record, state = store.scan_own_relay(root)
    assert record is not None and state == "wedged"
    # The live-only reader is the one a DIALER needs, and it is right to stay silent.
    assert store.find_own_relay(root) is None

    payload = relay.status()
    assert payload["record"] is not None
    assert payload["record"]["pid"] == os.getpid()
    assert payload["relay_running"] is True, "a live pid beside relay_running: false"
    assert payload["relay_answering"] is False
    assert payload["relay_state"] == "wedged"

    line, up = net_cli._relay_state()  # noqa: SLF001 — what `doctor` prints
    assert up is True
    assert f"pid {os.getpid()}" in line and "did not answer" in line

    message = net_cli._relay_unavailable_message()  # noqa: SLF001
    assert "is not running" not in message
    assert f"pid {os.getpid()}" in message


# ---------------------------------------------------------------------------
# Q16-1 — a rotation learned through the TABLE, not through the frame
# ---------------------------------------------------------------------------


def test_a_pulled_rotation_table_leaves_one_row_for_the_rotated_device(
    devices: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A peer that pulls AFTER a rotation counts that member ONCE.

    THE ROTATION IS DRIVEN BY THE CLI'S OWN VERB, and deliberately driven the way it
    behaves when nothing is answering: ``lop network identity rotate`` rewrites the
    record locally and announces nothing, so a member table is the ONLY route by
    which a peer can learn the new id. That ordering — pull before the queued
    ``net_identity_rotate`` frame — is what QA round 16 reproduced on a real device:
    the merge appended the rotated row BESIDE the pre-rotation row it had never been
    told to retire, so the peer held two active rows for one device and reported one
    more member than the rotated device itself did.

    What makes the retire lawful is not the merge's judgement: it is the statement
    the rotation produced, which now rides on the row. So this cell also pins that
    the peer still resolves the OLD id (a link that authenticated at it is not cut)
    and that nothing about the retirement is a removal — no tombstone, no burned id.
    """
    hub = _make(devices, "hub")
    peer = _make(devices, "peer")
    record = _init_network(hub.server)
    _join(peer, inviter=hub, monkeypatch=monkeypatch)
    assert _members(hub) == _members(peer) == {hub.device_id, peer.device_id}

    old_id = hub.device_id
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(hub.root))
    # ``health`` is the verb's own "is a relay answering" probe; None is its
    # no-answer half, which writes the record itself and announces to nobody.
    monkeypatch.setattr(relay, "health", lambda *args, **kwargs: None)
    assert net_cli._cmd_identity_rotate(Namespace(json=True)) == 0  # noqa: SLF001
    rotated = store.load(record.network_id, hub.root)
    new_id = rotated.self_device_id
    assert new_id != old_id, "the verb did not rotate the device id"

    link, reason = peer.server.dial(record.network_id, host=f"{hub.host}:{hub.port}", epoch=1)
    assert link is not None, reason
    assert peer.server._pull_members(link) == ""  # noqa: SLF001
    view = store.load(record.network_id, peer.root)
    assert _members(peer) == _members(hub)
    assert [row.device_id for row in view.active_members() if row.device_id != peer.device_id] == [
        new_id
    ]
    assert view.member(old_id) is view.member(new_id), "the old id must still resolve"
    assert old_id not in view.removed_ids, "a rotation is not a removal"
    assert all(row.removed_at is None for row in view.members)
    # The proof came across the wire WITH the row, so the next device this one
    # serves can make the same check rather than trusting this peer's word for it.
    surviving = view.member(new_id)
    assert surviving is not None
    assert surviving.rotation_proof.get("sig_old")


def test_the_announced_rotation_carries_its_statement_on_the_row(
    devices: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ANNOUNCE half writes the proof too, not just the offline half.

    The frame is the DELIVERY; the row is the route for a member the verb cannot
    reach, and a peer that dials in afterwards reads the table rather than the frame.
    Both halves of ``lop network identity rotate`` therefore leave the statement on
    the row (``MemberRecord.rotation_proof``), and this is the half a live relay runs
    — through ``announce_identity_rotation``, on a ``RelayServer`` the verb builds
    for itself, which is also why that verb's ``sent`` counter reads 0 while its
    ``queued`` does not (QA round 16, Q16-2).

    The proof is verified the way a PEER verifies it — against the public key on the
    row it already holds for the old id — because a statement that only round-trips
    through our own writer proves nothing about what a peer can check.
    """
    hub = _make(devices, "hub")
    peer = _make(devices, "peer")
    record = _init_network(hub.server)
    _join(peer, inviter=hub, monkeypatch=monkeypatch)
    old_id = hub.device_id
    peer_view = store.load(record.network_id, peer.root)
    old_row = peer_view.member(old_id)
    assert old_row is not None

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(hub.root))
    # A truthy ``health`` is the verb's "a relay is answering" half, which announces
    # instead of rewriting the record itself.
    monkeypatch.setattr(relay, "health", lambda *args, **kwargs: {"ok": True})
    assert net_cli._cmd_identity_rotate(Namespace(json=True)) == 0  # noqa: SLF001

    rotated = store.load(record.network_id, hub.root)
    assert rotated.self_device_id != old_id
    self_row = rotated.member(rotated.self_device_id)
    assert self_row is not None
    proof = self_row.rotation_proof
    assert proof["network_id"] == record.network_id
    assert proof["old_device_id"] == old_id
    assert proof["new_device_id"] == rotated.self_device_id
    # Verified from the OTHER device's copy of the old key: what makes the row
    # admissible to a peer that never saw the frame.
    identity.verify_rotation_statement(proof, old_row.public_key)


# ---------------------------------------------------------------------------
# The third way to get a wrong address onto a row
# ---------------------------------------------------------------------------


def test_a_member_that_declares_nothing_gets_no_endpoint_not_the_observed_one(
    devices: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row's ``endpoints`` is what ``_ensure_link`` DIALS, so a value that cannot be
    dialled is worse than no value — it is read as an address.

    The pair ceremony used to fall back to the OBSERVED source address of the pairing
    connection when the joiner declared nothing: ``peer_addr``, the peer's ephemeral
    port on this one socket, gone the moment the pairing link drops. On a loopback rig
    that is ``127.0.0.1:<ephemeral>`` — a loopback address AND a port nothing listens
    on (``127.0.0.1:52315`` in the audit that reported it) — and on a real pair it is
    whatever the NAT mapped for that connection. Both are confidently wrong, where the
    honest answer already has vocabulary everywhere it is read: ``endpoints: []``,
    reported as ``no_endpoint`` ("no address published for it") and ``reachable:
    false``, with the member kept.

    The silent device is silent BY CONSTRUCTION — the ``devices`` fixture's ``SILENT``
    mode answers nothing on every path that publishes an endpoint — so this cannot pass
    because some host happened to advertise, and the precondition is asserted rather
    than assumed. What this does NOT claim is that a DECLARED loopback endpoint is
    wrong: `--listen-address 127.0.0.1` is the documented dial-only mode saying "you
    cannot reach me", and ``test_relay_e2e``'s F-2 case pins that it stays.
    """
    hub = _make(devices, "hub")
    silent = _make(devices, "silent-joiner", mode=SILENT)
    _init_network(hub.server)
    # THE PRECONDITION: nothing declared, so the row has an OBSERVATION and no
    # declaration to be built from. Without this, a rig that advertised would satisfy
    # the assertion below for entirely the wrong reason.
    assert relay.advertise_endpoints(silent.server.settings) == []

    _join(silent, inviter=hub, monkeypatch=monkeypatch)

    record = store.load(store.list_networks(hub.root)[0].network_id, hub.root)
    row = record.member(silent.identity.device_id)
    assert row is not None and row.active, _members(hub)
    assert row.endpoints == [], row.endpoints

    # AND THE HONEST SENTENCE. "Nothing to dial" must read as nothing declared rather
    # than as a failed dial: the dial's refusal is a claim about the peer, and the peer
    # here has said nothing about itself at all.
    #
    # THE CEREMONY'S LINK IS CLOSED FIRST, and that is the fix for a race this assertion
    # used to lose: the join has just paired over a real socket, so this device's link to
    # the joiner is still ALIVE — `_ensure_link_with_reason` answers the link cache before
    # it dials anything, and `_link_for` counts any link whose `_closed` is unset — and
    # whether its reader had noticed the close yet was a coin flip. CI shard 0 caught it
    # (`assert <PeerLink> is None`); 25 local runs did not. Closing HERE, synchronously,
    # makes the subject the product's answer for a member with no live link — the state
    # `no_endpoint` is vocabulary for — instead of the teardown's timing. The alternative,
    # branching on "the link may still be alive", would assert two different claims and
    # pass either way, which is the shape this file has already been burned by.
    found = hub.server._link_for(silent.identity.device_id)  # noqa: SLF001
    if found is not None:
        found.close("test-closed")
    assert hub.server._link_for(silent.identity.device_id) is None, "the link must be gone"
    link, reason = hub.server._ensure_link_with_reason(  # noqa: SLF001 — the surface's reader
        silent.identity.device_id
    )
    assert link is None
    assert reason == "no_endpoint", reason


def test_an_invite_names_the_live_listener_not_only_the_records_stale_port(
    devices: Any,
) -> None:
    """An invite's hosts are where a joiner DIALS, so the LIVE listener must be among them.

    ``record.listen["advertised"]`` is written once, by ``init``, from the config as it
    stood then. A listener bound to a port the record does not name used to mint invites
    naming that dead port ALONE — the QA rig saw the invite read ``47774`` while ``lsof``
    showed the listener on ``47778``, and a joiner following it dialed nothing (review
    round 1, Q-1). The relay answers from what it actually bound
    (``_own_relay_listen``/``advertised_endpoints``), and the record's own entries are
    KEPT ahead of it because that list is also where a deliberate ``--advertise-host``
    declaration lives — so the property is that the live port is always named, not that
    the record is ignored.
    """
    hub = _make(devices, "hub")
    record = _init_network(hub.server)
    live_port = int(record.listen["port"])
    stale_port = live_port + 17
    # The stale half: what a record written by an earlier launch would carry.
    record.listen["port"] = stale_port
    record.listen["advertised"] = [f"127.0.0.1:{stale_port}"]
    store.save(record, hub.root)
    assert hub.server.settings.port == live_port, "the fixture's bind is the live port"

    minted = hub.server._ctl_invite(
        {  # noqa: SLF001 — the CLI's own control op
            "network": record.name,
            "role": "read",
            "ttl_s": 600.0,
        }
    )

    assert f"127.0.0.1:{live_port}" in minted["hosts"], minted["hosts"]
    # THE PRECEDENCE THE DOCSTRING NAMES, and a claim the line above does NOT make: the
    # record's OWN entry is kept beside the live one, because that list is also where a
    # deliberate `--advertise-host` declaration lives. The clause this replaces — "not the
    # stale entry alone" — could not fail where the line above passed (`stale_port` is
    # `live_port + 17`, so a live entry already rules that singleton out) and passed where
    # the line above failed (an empty list is not that singleton): it passed either way,
    # which is the shape this file's own comment says it has been burned by (review round
    # 3, MINOR). Dropping the record half of the mint fails THIS line and leaves the one
    # above green, which is the teeth the old clause only looked like it had.
    assert f"127.0.0.1:{stale_port}" in minted["hosts"], minted["hosts"]


def test_an_invite_falls_back_to_the_record_when_nothing_is_detected(devices: Any) -> None:
    """The fallback is load-bearing: a device that detects nothing still hands out its row.

    The silent rig answers ``[]`` for every path that publishes an endpoint, so the live
    half of ``_invite_hosts`` is empty BY CONSTRUCTION — the same shape as a real device
    behind a NAT that holds nothing dialable. The record is then what a joiner gets, and
    without it such a device would mint ``hosts: []`` and send the joiner to
    ``--host host:port``, which is the state this change exists to remove.
    """
    quiet = _make(devices, "silent-joiner", mode=SILENT)
    record = _init_network(quiet.server)
    recorded = int(record.listen["port"])
    record.listen["advertised"] = [f"127.0.0.1:{recorded}"]
    store.save(record, quiet.root)
    assert relay.advertise_endpoints(quiet.server.settings) == [], "the rig must detect nothing"

    minted = quiet.server._ctl_invite(
        {  # noqa: SLF001
            "network": record.name,
            "role": "read",
            "ttl_s": 600.0,
        }
    )

    assert minted["hosts"] == [f"127.0.0.1:{recorded}"], minted["hosts"]


# ---------------------------------------------------------------------------
# The unanswered read: an age, a retry, and a status read that asks
# ---------------------------------------------------------------------------


def test_an_unanswered_read_carries_its_age_and_the_retry() -> None:
    """The "contradiction" class: a failed read must name WHEN and say it is retried.

    `lop network ls` printed `[members NOT verified: no peer answered]` for a read
    that runs again within seconds (a peer that accepts and then does not answer
    its table read costs one four-second pull, ``MEMBERSHIP_PULL_TIMEOUT_S``),
    while `lop network peers` reported the same peer reachable — different
    questions at different instants, made to look contradictory by a wording with
    no date and no next step. The sentence (`show`) and the marker (`ls`, the
    agent digest) now carry the read's age and "— retrying"; a row NO read has
    fed says so instead of borrowing the failure's words, and the failure
    headline itself claims no ask (design round 1, D1) because a member with no
    live link was never asked.
    """
    device = "d_" + "1" * 32
    report = relay.MembershipReport(network_id="n_" + "a" * 22, refreshed_at=time.time() - 42.0)
    report.silent.append({"device_id": device, "reason": "no_table:no_answer"})
    sentence = report.sentence()
    assert "no table came back in the last read" in sentence, sentence
    assert "no peer answered" not in sentence, sentence
    assert ("42s ago" in sentence) or ("43s ago" in sentence), sentence
    assert sentence.endswith(
        "— retrying: d_1111111111 (it did not answer the table read)"
    ), sentence

    row = {"members": 2, "membership": {"table": report.to_json()}}
    marker = relay.membership_marker(row)
    assert "no table came back in the last read" in marker, marker
    assert ("42s ago" in marker) or ("43s ago" in marker), marker
    assert marker.endswith("— retrying: d_1111111111 (it did not answer the table read)]"), marker

    # THE HEADLINE CLAIMS NO ASK (design round 1, D1): a member with NO LIVE LINK
    # was never asked, so "no peer answered" would contradict its own reason
    # ("nothing is connected to it") — the placeholder arm's defect, one case
    # over. Both surfaces say only what every silent case shares.
    stalled = relay.MembershipReport(network_id="n_" + "a" * 22, refreshed_at=time.time() - 8.0)
    stalled.silent.append({"device_id": device, "reason": "no_live_link"})
    stalled_sentence = stalled.sentence()
    assert "no table came back in the last read" in stalled_sentence, stalled_sentence
    assert "no peer answered" not in stalled_sentence, stalled_sentence
    assert stalled_sentence.endswith(
        "— retrying: d_1111111111 (nothing is connected to it)"
    ), stalled_sentence

    # A row built by an older build (no stamps) OMITS the age rather than inventing
    # one, and still says the retry the cadence performs.
    bare = relay.membership_marker(
        {
            "members": 2,
            "membership": {
                "table": {
                    "complete": False,
                    "answered": [],
                    "not_answered": [{"device_id": device, "reason": "no_live_link"}],
                }
            },
        }
    )
    assert "no table came back in the last read — retrying" in bare, bare
    assert "no peer answered" not in bare, bare
    assert "s ago" not in bare and "just now" not in bare, bare

    # A row NO read has fed says that — "no peer answered" about a read nobody ran
    # is the same lie in the other direction, and it is what every `status` row
    # said before the read learned to ask.
    unread = relay.membership_marker(
        {
            "members": 2,
            "membership": {"table": {"complete": False, "answered": [], "not_answered": []}},
        }
    )
    assert unread == "  [members NOT verified: no table read has completed yet — retrying]", unread

    # THE PARTIAL ARM NAMES THE MISSING PEER TOO (design round 1, D4): "1 of 2" hid
    # WHICH member was missing, while the no-answer arm and the long form both name
    # theirs — the list now says who, and why, on both short and long forms.
    partial = relay.membership_marker(
        {
            "members": 3,
            "membership": {
                "table": {
                    "complete": False,
                    "answered": [device],
                    "not_answered": [{"device_id": "d_" + "2" * 32, "reason": "no_live_link"}],
                    "oldest_answer_age_s": 3.2,
                }
            },
        }
    )
    assert partial == (
        "  [members verified with 1 of 2 peer(s) (3s ago); "
        "NOT verified with d_2222222222 (nothing is connected to it)]"
    ), partial

    # AGES STOP BEING SECONDS once they are stale — the case the age exists for
    # (design round 1, N3).
    assert relay._age_words(3600.0) == "1h ago", relay._age_words(3600.0)
    assert relay._age_words(7200.0) == "2h ago", relay._age_words(7200.0)
    assert relay._age_words(172800.0) == "2d ago", relay._age_words(172800.0)

    # THE SOLO-NETWORK ARM: a pass that had nobody to ask says so — its `--json`
    # sentence used to claim "no peer answered" about a network with no peers.
    lonely = relay.MembershipReport(network_id="n_" + "a" * 22, refreshed_at=time.time())
    assert lonely.sentence() == "members verified: no other members to ask", lonely.sentence()

    # The VERIFIED arms carry the oldest answer's age too (it was already in the
    # report; it is now on the line). A row without it keeps the pre-age bytes, so
    # a fixture from an older relay renders exactly what it always did.
    aged = relay.membership_marker(
        {
            "members": 2,
            "membership": {
                "table": {
                    "complete": True,
                    "answered": ["d_1"],
                    "not_answered": [],
                    "oldest_answer_age_s": 3.2,
                }
            },
        }
    )
    assert aged == "  [members verified with all 1 peer(s) (3s ago)]", aged
    defunct = relay.membership_marker(
        {
            "members": 2,
            "membership": {"table": {"complete": True, "answered": ["d_1"], "not_answered": []}},
        }
    )
    assert defunct == "  [members verified with all 1 peer(s)]", defunct


def test_a_status_read_asks_for_the_pass_it_reports(
    devices: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`status` must not answer from the last completed pass: it asks for a read.

    The discriminator is the COMPLETION SEQUENCE, not the emptiness of the
    cache: a `status` that merely read ``_membership_reports`` would leave
    ``_membership_seq`` untouched while claiming a fresh row, so the refresh
    read must bump it and the plain read must not.

    A blank baseline cannot be demanded. The admit-time ``contact_peers`` pass
    starts on a handshake thread of its own once the joiner's result frame is
    sent and completes whenever it gets to run — publishing a pass that answers
    the link. The old version captured `before` immediately after the dial and
    asserted an empty table; on a starved runner that pass landed first and the
    assertion failed on state that was merely later (CI shard 4, 2026-10-12;
    causally reproduced by delaying the admit pass past the dial). So the
    fixture now QUIESCES every unprompted writer (the sibling test below
    carries the full derivation of each wait), then FORCES the completed pass
    it asserts against, and proves the freshness contract structurally.
    """
    monkeypatch.setattr(relay, "MEMBERSHIP_PULL_PASS_S", 30.0)
    a = _make(devices, "a", mode=HUB)
    b = _make(devices, "b", mode=HUB)
    record = _init_network(a.server)
    admit_seq = a.server._membership_seq  # noqa: SLF001 — "no pass completed yet"
    assert _join(b, inviter=a, monkeypatch=monkeypatch)["device_id"] == b.device_id
    link, reason = b.server.dial(record.network_id, host=f"{a.host}:{a.port}", epoch=1)
    assert link is not None, reason

    def _row(payload: dict[str, Any]) -> dict[str, Any]:
        return next(row for row in payload["networks"] if row["network_id"] == record.network_id)

    a_link = a.server._link_for(b.device_id)  # noqa: SLF001 — the fixture's link
    assert a_link is not None

    # QUIESCE EVERY UNPROMPTED WRITER BEFORE THE BASELINE IS TAKEN: a's
    # establishment pull for the inbound link, the admit-time contact_peers
    # pass, and any inbound handshake still returning (each stage's derivation
    # lives in the sibling test below). Afterwards the only publisher left is
    # a pass this fixture or the read itself runs — the cadence is parked for
    # the whole test lifetime.
    assert net_fixtures.wait_for(
        lambda: (current := a.server._link_for(b.device_id)) is not None  # noqa: SLF001
        and current.member_pulled_at > 0.0
    ), "a's establishment pull never stamped the current link"
    assert net_fixtures.wait_for(
        lambda: a.server._membership_seq > admit_seq  # noqa: SLF001
    ), "a's admit-time contact_peers pass never completed"

    def _all_inbound_done() -> bool:
        return a.server._handshake_slots._value == a.server.settings.max_handshakes  # noqa: SLF001

    assert net_fixtures.wait_for(
        _all_inbound_done
    ), "an inbound handshake can still restamp the link"
    # Resolve AFTER quiescence: an admit-time dial can replace the link captured
    # above. Re-arming that retired object would not make the live pull due.
    a_link = a.server._link_for(b.device_id)  # noqa: SLF001
    assert a_link is not None, "a holds no live link after its handshakes settled"

    # THE BASELINE, made deterministic instead of assumed: one completed pass
    # that really answers the link. The due window is re-opened first so this
    # pass enters the pull rather than answering from the not-due gate.
    a_link.member_pulled_at = 0.0
    baseline = a.server.refresh_membership()[record.network_id]  # noqa: SLF001
    assert baseline.answered == [b.device_id], baseline
    seq_before = a.server._membership_seq  # noqa: SLF001

    # Plain `status` reports the completed pass and asks NOTHING: the sequence
    # is the receipt — a cached answer that never asked cannot have bumped it.
    before = _row(a.server.status())["membership"]["table"]  # noqa: SLF001
    assert before["answered"] == [b.device_id], before
    assert before["complete"] is True, before
    assert a.server._membership_seq == seq_before, "a plain status ran a pass"  # noqa: SLF001

    # The read asks, and the pass it asked for is the one it reports: the
    # completion sequence moves, and the table is the pass that moved it.
    after = _row(a.server.status(refresh=True))["membership"]["table"]  # noqa: SLF001
    assert (
        a.server._membership_seq > seq_before
    ), "the read reported a cached table without asking"  # noqa: SLF001
    assert after["answered"] == [b.device_id], after
    assert after["complete"] is True
    assert isinstance(after["oldest_answer_age_s"], float), after
    assert "verified with all 1 peer(s)" in after["sentence"]


def test_a_status_read_does_not_wait_out_a_hung_pass(
    devices: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The read's wait is BOUNDED: a pass stuck in a pull must not hold the command.

    The fallback is the last completed pass — reported with its own age, never
    dressed as fresh — and the call returns while the kicked pass is STILL IN
    FLIGHT (asserted on the completion sequence, not on a clock), instead of
    waiting out the peer's pull. A release in ``finally`` leaves nothing waiting.
    """
    monkeypatch.setattr(relay, "MEMBERSHIP_PULL_PASS_S", 30.0)
    a = _make(devices, "a", mode=HUB)
    b = _make(devices, "b", mode=HUB)
    record = _init_network(a.server)
    # THE BASELINE FOR THE SECOND WAIT BELOW, taken BEFORE the join: a's admit-time
    # ``contact_peers`` pass starts on a handshake thread of its own once the joiner's
    # result frame is sent, and may still be in flight when this test is already past
    # the dial.
    admit_seq = a.server._membership_seq  # noqa: SLF001 — "no pass completed yet"
    assert _join(b, inviter=a, monkeypatch=monkeypatch)["device_id"] == b.device_id
    link, reason = b.server.dial(record.network_id, host=f"{a.host}:{a.port}", epoch=1)
    assert link is not None, reason

    a_link = a.server._link_for(b.device_id)  # noqa: SLF001 — the fixture's link
    assert a_link is not None

    # LET a's OWN ESTABLISHMENT PULL SETTLE BEFORE THE DUE WINDOW IS ARMED. b's dial
    # returns when b is satisfied, but a's listener side runs its own ``net_member_list``
    # pull for this link on a handshake thread of its own (``register_link``), and that
    # pull STAMPS ``member_pulled_at`` when its answer lands. Armed too early, the arm
    # below (``= 0.0``) is overwritten by that late stamp, the link reads as pulled
    # moments ago, the kicked pass finds it NOT DUE and answers from the cadence
    # without ever entering ``_pull_members`` — "the kicked pass never entered the
    # pull" (CI shards 2026-10-09/10; the margin between the two threads is ~1 ms on a
    # quiet host, measured, and a starved runner turns it into a coin flip). The stamp
    # is the event: it is written exactly once per establishment pull, so waiting for
    # it leaves no other writer racing the arm.
    assert net_fixtures.wait_for(
        lambda: (current := a.server._link_for(b.device_id)) is not None  # noqa: SLF001
        and current.member_pulled_at > 0.0
    ), "a's establishment pull never stamped the current link"

    # AND LET THE ADMIT-TIME PASS COMPLETE TOO, or one writer the test does not
    # control can still restamp the link behind the arm below (agent review round 1,
    # MINOR — the residual behind QA round 1's one green miss): the same listener
    # runs a ``contact_peers`` pass after sending the result frame, and starved
    # across the dial it reads the REPLACEMENT link while that link is due
    # (``member_pulled_at == 0.0``), pulling it: "a second stamp on the same link",
    # landing after the arm and reinstating the not-due skip. The pass bumps
    # ``_membership_seq`` exactly once, at completion, after every pull it made has
    # stamped, so waiting for the bump leaves no unprompted pass in flight when the
    # due window is armed.
    assert net_fixtures.wait_for(
        lambda: a.server._membership_seq > admit_seq
    ), "a's admit-time contact_peers pass never completed"

    # AND NO INBOUND HANDSHAKE MAY STILL BE IN FLIGHT EITHER, closing the sibling
    # shape a probe for this change found: the establishment wait above can observe
    # the ADMIT pass's restamp first while ``register_link``'s own establishment
    # pull is still in flight — and THAT pull's stamp can then land behind the arm
    # below, the same not-due skip from the other side. A ``_handshake_slots`` slot
    # is the right signal because it is acquired once per inbound connection
    # (``_accept_loop``) and released only in ``_handshake_inbound``'s ``finally`` —
    # i.e. when the WHOLE inbound work returns: ``register_link``'s establishment
    # pull (the stamp) and a pair listener's ``contact_peers`` (the admit restamp)
    # included — so "all slots free" is exactly "no unprompted inbound writer can
    # still stamp this link". The two inbound connections on a here are the join's
    # and the dial's. ``_value`` is read because the semaphore exposes nothing else,
    # and is only ever compared against the cap that built it.
    def _all_inbound_done() -> bool:
        return a.server._handshake_slots._value == a.server.settings.max_handshakes  # noqa: SLF001

    assert net_fixtures.wait_for(
        _all_inbound_done
    ), "an inbound handshake can still restamp the link"
    # Resolve AFTER quiescence: an admit-time dial can replace the link captured
    # above. Re-arming that retired object would not make the live pull due.
    a_link = a.server._link_for(b.device_id)  # noqa: SLF001
    assert a_link is not None, "a holds no live link after its handshakes settled"

    # A completed pass first: the fallback must BE this one, so make it answer with
    # b's table (a real pull — the link is up, and the zero makes it due).
    a_link.member_pulled_at = 0.0
    first = a.server.refresh_membership()[record.network_id]  # noqa: SLF001
    assert first.answered == [b.device_id], first

    # Make the NEXT pull hang, and re-open the link's due window so the pass the
    # read triggers actually enters the hung pull.
    a_link.member_pulled_at = 0.0
    release = threading.Event()

    entered = threading.Event()

    def _hang(link: Any) -> str:
        entered.set()
        release.wait(10.0)
        return "no_answer"

    monkeypatch.setattr(a.server, "_pull_members", _hang)
    seq_before = a.server._membership_seq  # noqa: SLF001 — "no pass completed yet"
    inflight = False
    try:
        payload = a.server.status(refresh=True)  # noqa: SLF001
        # STRUCTURAL, NOT A CLOCK (agent review round 1, NIT): the hung pass cannot
        # have completed, so an unchanged completion sequence at the moment the
        # read returns proves it returned WITH the pass still in flight — whatever
        # the scheduler did to wall times.
        inflight = a.server._membership_seq == seq_before  # noqa: SLF001
    finally:
        release.set()  # let the loop's pass finish; nothing may be left waiting

    table = next(row for row in payload["networks"] if row["network_id"] == record.network_id)[
        "membership"
    ]["table"]
    # The fallback is the last COMPLETED pass, with its own age on the line.
    assert table["answered"] == [b.device_id], table
    assert table["complete"] is True
    assert ("just now" in table["sentence"]) or ("s ago" in table["sentence"]), table["sentence"]
    assert net_fixtures.wait_for(entered.is_set), "the kicked pass never entered the pull"
    assert inflight, "the read waited the hung pass out instead of returning with it in flight"


def test_a_status_read_with_a_pass_still_running_says_so_not_the_last_failure(
    devices: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bounded read whose kicked pass is still running must SAY SO.

    The last completed pass below FAILED ("no table came back"), while the peer
    is reachable on the live probe. Re-presenting that failure headline beside
    `peers` saying reachable reads as a contradiction: the truth at the moment
    the bounded wait expires is that a NEW read is in flight, neither verified
    nor failed. After the pull is released, the next read reports verified.
    """
    monkeypatch.setattr(relay, "MEMBERSHIP_PULL_PASS_S", 30.0)
    a = _make(devices, "a", mode=HUB)
    b = _make(devices, "b", mode=HUB)
    record = _init_network(a.server)
    admit_seq = a.server._membership_seq  # noqa: SLF001
    assert _join(b, inviter=a, monkeypatch=monkeypatch)["device_id"] == b.device_id
    link, reason = b.server.dial(record.network_id, host=f"{a.host}:{a.port}", epoch=1)
    assert link is not None, reason
    a_link = a.server._link_for(b.device_id)  # noqa: SLF001
    assert a_link is not None
    # Quiesce unprompted writers (same derivation as the hung-pass sibling):
    # establishment stamp, admit-time pass, and every inbound handshake.
    assert net_fixtures.wait_for(
        lambda: (current := a.server._link_for(b.device_id)) is not None  # noqa: SLF001
        and current.member_pulled_at > 0.0
    ), "a's establishment pull never stamped the current link"
    assert net_fixtures.wait_for(
        lambda: a.server._membership_seq > admit_seq  # noqa: SLF001
    ), "a's admit-time contact_peers pass never completed"

    def _all_inbound_done() -> bool:
        return a.server._handshake_slots._value == a.server.settings.max_handshakes  # noqa: SLF001

    assert net_fixtures.wait_for(
        _all_inbound_done
    ), "an inbound handshake can still restamp the link"
    # Resolve AFTER quiescence: an admit-time dial can replace the link captured
    # above. Re-arming that retired object would not make the live pull due.
    a_link = a.server._link_for(b.device_id)  # noqa: SLF001
    assert a_link is not None, "a holds no live link after its handshakes settled"

    def _row(payload: dict[str, Any]) -> dict[str, Any]:
        return next(row for row in payload["networks"] if row["network_id"] == record.network_id)

    # The last COMPLETED pass fails: its headline is the stale verdict.
    monkeypatch.setattr(a.server, "_pull_members", lambda link: "no_answer")
    a_link.member_pulled_at = 0.0
    failed = a.server.refresh_membership()[record.network_id]  # noqa: SLF001
    assert failed.answered == [], failed
    assert "no table came back" in failed.sentence(), failed.sentence()

    # The next pull blocks past the bounded status wait, then really answers.
    entered = threading.Event()
    release = threading.Event()

    def _blocked(link: Any) -> str:
        entered.set()
        release.wait(10.0)
        return relay.RelayServer._pull_members(a.server, link)  # noqa: SLF001

    monkeypatch.setattr(a.server, "_pull_members", _blocked)
    a_link.member_pulled_at = 0.0
    seq_before = a.server._membership_seq  # noqa: SLF001
    try:
        payload = a.server.status(refresh=True)  # noqa: SLF001
        inflight = a.server._membership_seq == seq_before  # noqa: SLF001
        peers = a.server.peer_status()  # noqa: SLF001
    finally:
        release.set()
    assert entered.is_set(), "the kicked pass never entered the pull"
    assert inflight, "the read waited the blocked pull out instead of returning bounded"
    assert peers and peers[0]["reachable"] is True, peers
    row = _row(payload)
    table = row["membership"]["table"]
    sentence = table["sentence"]
    assert "table read still running" in sentence, sentence
    assert "retrying" in sentence, sentence
    assert "no table came back" not in sentence, sentence
    assert "members verified" not in sentence, sentence
    assert "still running" in relay.membership_marker(row), relay.membership_marker(row)

    # After the pull lands, the next read is the verified completed pass.
    assert net_fixtures.wait_for(
        lambda: a.server._membership_seq > seq_before  # noqa: SLF001
    ), "the blocked pass never completed after release"
    after = _row(a.server.status(refresh=True))["membership"]["table"]  # noqa: SLF001
    assert after["complete"] is True, after
    assert "verified with all 1 peer(s)" in after["sentence"], after["sentence"]


def test_a_row_no_read_has_fed_says_so_on_every_surface(
    devices: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A count nobody read says THAT, and never that a peer failed to answer.

    The placeholder rows (``RelayServer.network_summary`` without a report, and
    ``cli._summarise`` when the relay itself did not answer) fed the marker an
    empty `not_answered`, which the old marker rendered as "no peer answered" — a
    claim about an ask that never happened, the same defect as the un-dated
    failure in the other direction. The empty list is now read for what it is.
    """
    a = _make(devices, "a", mode=HUB)
    b = _make(devices, "b", mode=HUB)
    record = _init_network(a.server)
    assert _join(b, inviter=a, monkeypatch=monkeypatch)["device_id"] == b.device_id

    row = a.server.network_summary(store.load(record.network_id, a.root))
    table = row["membership"]["table"]
    assert "no table read has completed yet" in table["sentence"], table
    assert "no peer answered" not in table["sentence"], table
    marker = relay.membership_marker(row)
    assert marker == "  [members NOT verified: no table read has completed yet — retrying]", marker

    # The CLI's own fallback (the relay did not answer `net_ls`/`net_show`) says
    # the same about its half: the read never completed, because the relay was
    # silent — and no peer was asked.
    summary = net_cli._summarise(store.load(record.network_id, a.root))  # noqa: SLF001
    sentence = summary["membership"]["table"]["sentence"]
    assert "no table read has completed yet" in sentence, sentence
    assert "relay did not answer" in sentence, sentence


#: How long the relay's SELF-TIMED membership pass is parked for the status-verb
#: cells. The cadence (``MEMBERSHIP_PULL_PASS_S``) is a second, unprompted
#: publisher of ``_membership_reports``: left at the 30 s the sibling cells use,
#: a runner starved past it would let a cadence tick publish behind the cell and
#: satisfy "a fresh pass happened" for a request that asked for nothing. The wake
#: a fresh read sets (``_fresh_membership_read_state``) and ``stop()`` both break
#: the loop's wait independently of this number, so parking it removes only the
#: unprompted publisher and leaves the path under test fully functional. The cell's
#: own lifetime is unchanged: the number only has to exceed it.
_CADENCE_PARKED_S = 3600.0


def _relay_source_identity() -> dict[str, str]:
    """Path and sha256 of the relay module under test (this cell runs in-process).

    Recorded so the artifact says WHICH bytes were exercised, the in-process analogue
    of the module identity the store-walk child reports.
    """
    path = Path(str(relay.__file__)).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _first_line(error: BaseException, limit: int = 300) -> str:
    """The first line of an assertion's message: the guard's own sentence, bounded."""
    lines = str(error).splitlines() or [""]
    return lines[0][:limit]


def _status_pair_after_admit(
    devices: Any, monkeypatch: pytest.MonkeyPatch
) -> tuple[Device, Device, types.NetworkRecord, Any]:
    """Two joined, linked relays with every unprompted publisher on ``a`` drained.

    The pairing makes ``a`` (the inviter) run ``contact_peers`` on a handshake
    thread once the joiner's result frame is sent: a LEGITIMATE publication into
    ``_membership_reports`` that lands whenever that thread gets to run. The old
    form of this cell read the cache straight after the dial and asserted it was
    still empty, i.e. it asserted that this pass had not landed yet — a race the
    sibling cells above already close, each for the same reason. A publication is
    not evidence of anything wrong here, so the cell waits for it instead of
    betting against it:

    * ``a``'s establishment pull has stamped the link;
    * the admit-time pass has completed (the sequence moved past the value read
      BEFORE the join);
    * every inbound handshake has returned (the semaphore is back at its cap), so
      neither writer can restamp or replace the link later.

    THE LINK RETURNED IS RESOLVED AFTER THAT QUIESCENCE. One captured before the
    handshakes settle can be the link a later handshake replaced, and re-arming a
    stale object would arm nothing the relay still reads.
    """
    monkeypatch.setattr(relay, "MEMBERSHIP_PULL_PASS_S", _CADENCE_PARKED_S)
    a = _make(devices, "a", mode=HUB)
    b = _make(devices, "b", mode=HUB)
    record = _init_network(a.server)
    admit_seq = a.server._membership_seq  # noqa: SLF001 — "no pass completed yet"
    assert _join(b, inviter=a, monkeypatch=monkeypatch)["device_id"] == b.device_id
    link, reason = b.server.dial(record.network_id, host=f"{a.host}:{a.port}", epoch=1)
    assert link is not None, reason

    # The ambient root is what `health()` scans for the relay's record; a's own
    # store holds one (published at start, re-stamped by its heartbeat).
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(a.root))

    def _pulled() -> bool:
        current = a.server._link_for(b.device_id)  # noqa: SLF001
        return current is not None and current.member_pulled_at > 0.0

    assert net_fixtures.wait_for(_pulled), "a's establishment pull never stamped"
    assert net_fixtures.wait_for(
        lambda: a.server._membership_seq > admit_seq  # noqa: SLF001
    ), "a's admit-time contact_peers pass never completed"
    assert net_fixtures.wait_for(
        lambda: a.server._handshake_slots._value  # noqa: SLF001
        == a.server.settings.max_handshakes,
    ), "an inbound handshake can still restamp the link"
    settled = a.server._link_for(b.device_id)  # noqa: SLF001 — resolved AFTER quiescence
    assert settled is not None, "a holds no live link to b after the handshakes settled"
    return a, b, record, settled


def _observe_status_verb(
    devices: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    drop_refresh_field: bool,
    publication_wait_s: float,
) -> dict[str, Any]:
    """Drive plain then fresh `status` over the real control socket; return receipts.

    EVERYTHING IS A LOGICAL RECEIPT, NOT A CLOCK: the completion sequence
    (``_membership_seq``), the link's pull stamp going from the re-armed ``0.0`` to a
    real value, the NAME of the thread that ran each pass, and how many times the
    relay's own fresh read was entered. No wall-clock ordering is asserted (two reads
    of ``time.time()`` may tie on some platforms).

    THE ORDER IS THE POINT. The cached baseline is read FIRST, with no direct
    refresh; only then is the due window re-armed (``member_pulled_at = 0.0``) on the
    link the relay currently holds, and nothing that publishes runs between the
    re-arm and the fresh request. A baseline refresh AFTER the re-arm would consume
    the due pull, and the fresh request would then (correctly) answer from the
    not-due gate without a pull.

    ``drop_refresh_field`` is the negative control and it intervenes at the WIRE: the
    client's control request loses its ``refresh`` field on the way to the relay,
    which is what a client or a handler that dropped it would look like. The relay's
    own code is untouched, so the positive case's single entry into the fresh read
    against this case's none is direct evidence of what the field does.
    """
    a, b, record, a_link = _status_pair_after_admit(devices, monkeypatch)
    network_id = record.network_id
    link_id = a_link.link_id

    def _table(payload: dict[str, Any]) -> dict[str, Any]:
        row = next(row for row in payload["networks"] if row["network_id"] == network_id)
        return row["membership"]["table"]

    # The already-published pass, read through the same socket, with no refresh.
    baseline = _table(relay.status())
    seq0 = a.server._membership_seq  # noqa: SLF001

    publishers: list[str] = []
    real_refresh = a.server.refresh_membership

    def _recording_refresh(*args: Any, **kwargs: Any) -> Any:
        publishers.append(threading.current_thread().name)
        return real_refresh(*args, **kwargs)

    # THE RELAY'S OWN FRESH READ, observed rather than replaced: how often the handler
    # entered it, on which thread, and what sequence it saw on the way in and out.
    fresh_calls: list[dict[str, Any]] = []
    real_fresh = a.server._fresh_membership_read_state  # noqa: SLF001

    def _recording_fresh(*args: Any, **kwargs: Any) -> Any:
        entry: dict[str, Any] = {
            "thread": threading.current_thread().name,
            "seq_at_call": a.server._membership_seq,  # noqa: SLF001
        }
        reports, in_flight = real_fresh(*args, **kwargs)
        entry["seq_returned"] = a.server._membership_seq  # noqa: SLF001
        entry["in_flight"] = in_flight
        fresh_calls.append(entry)
        return reports, in_flight

    monkeypatch.setattr(a.server, "refresh_membership", _recording_refresh)
    monkeypatch.setattr(a.server, "_fresh_membership_read_state", _recording_fresh)

    dropped: list[str] = []
    if drop_refresh_field:
        real_control_request = relay.control_request

        def _without_refresh(control_record: Any, op: str, **fields: Any) -> Any:
            if op == "net_status" and fields.get("refresh"):
                fields.pop("refresh")
                dropped.append(op)
            return real_control_request(control_record, op, **fields)

        monkeypatch.setattr(relay, "control_request", _without_refresh)

    # Re-arm the link the relay holds NOW, and prove it is still the one resolved
    # after quiescence: a replaced link is reported by identity, never accepted.
    current = a.server._link_for(b.device_id)  # noqa: SLF001
    link_id_at_rearm = current.link_id if current is not None else None
    if current is not None:
        current.member_pulled_at = 0.0
    plain = _table(relay.status())
    observed: dict[str, Any] = {
        "peer": b.device_id,
        "link_id": link_id,
        "link_id_at_rearm": link_id_at_rearm,
        "seq0": seq0,
        "baseline_refreshed_at": baseline["refreshed_at"],
        "plain_seq": a.server._membership_seq,  # noqa: SLF001
        "plain_refreshed_at": plain["refreshed_at"],
        "plain_stamp": current.member_pulled_at if current is not None else None,
        "test_thread": threading.current_thread().name,
    }

    # ONE fresh request; then wait for the SPECIFIC pass it asked for, not for
    # whichever report happens to look favourable. The read's own wait is bounded
    # and may return ``in_flight`` — correct behaviour, so the reply is recorded and
    # never asserted; the receipt is the sequence and the plain read after it.
    initial_fresh = _table(relay.status(refresh=True))
    observed["initial_fresh_in_flight"] = initial_fresh["in_flight"]
    observed["published"] = net_fixtures.wait_for(
        lambda: a.server._membership_seq > seq0,  # noqa: SLF001
        timeout_s=publication_wait_s,
    )
    observed["seq_after"] = a.server._membership_seq  # noqa: SLF001
    observed["publishers"] = list(publishers)
    observed["fresh_calls"] = [dict(call) for call in fresh_calls]
    observed["dropped_refresh_fields"] = len(dropped)
    after = a.server._link_for(b.device_id)  # noqa: SLF001
    observed["link_id_after"] = after.link_id if after is not None else None
    observed["pulled_stamp"] = after.member_pulled_at if after is not None else None
    final = _table(relay.status())
    observed["final"] = {
        key: final[key] for key in ("answered", "complete", "in_flight", "not_due")
    }
    observed["final_seq"] = a.server._membership_seq  # noqa: SLF001
    return observed


def _assert_status_verb_asked_for_a_fresh_pass(observed: dict[str, Any]) -> None:
    """The oracle: plain status asks nothing, the fresh verb causes ONE real pass.

    Each message opens with the guard that fired — the controls below assert on it,
    so a control cannot pass because something else broke first.
    """
    # THE LINK IS THE ONE THAT WAS RE-ARMED, from the quiescent state to the pull. A
    # replacement is a publisher this cell did not control, named rather than excused.
    assert observed["link_id_at_rearm"] == observed["link_id"], (
        "link identity: the link changed between quiescence and the re-arm "
        f"({observed['link_id']} -> {observed['link_id_at_rearm']})"
    )
    # PLAIN STATUS reports the completed pass and asks for nothing.
    assert observed["plain_seq"] == observed["seq0"], "a plain status ran a membership pass"
    assert (
        observed["plain_refreshed_at"] == observed["baseline_refreshed_at"]
    ), "a plain status replaced the cached report"
    assert observed["plain_stamp"] == 0.0, "a plain status pulled a member table"

    # THE FRESH VERB reaches the relay's fresh read exactly once, from the control
    # handler (not from this test's thread), with nothing published since the
    # baseline — the request's intended target is the very next sequence.
    calls = observed["fresh_calls"]
    assert len(calls) == 1, (
        f"the fresh request reached the relay's fresh read {len(calls)} time(s), expected 1"
        f" (dropped refresh fields: {observed['dropped_refresh_fields']})"
    )
    assert calls[0]["thread"] != observed["test_thread"], "the fresh read ran on the test thread"
    assert calls[0]["seq_at_call"] == observed["seq0"], (
        "a pass published between the baseline and the fresh request "
        f"(sequence {observed['seq0']} -> {calls[0]['seq_at_call']}, "
        f"publishers={observed['publishers']})"
    )

    # ... and causes a NEW completed pass: the sequence moved, exactly once, run by
    # the membership loop that the wake released — not by a cadence tick (parked)
    # and not by any other caller.
    assert observed["published"], (
        "the fresh request produced no new completed pass: sequence stayed at "
        f"{observed['seq_after']} (was {observed['seq0']}), publishers={observed['publishers']}"
    )
    assert observed["seq_after"] == observed["seq0"] + 1, observed
    assert observed["publishers"] == ["mesh-membership"], observed["publishers"]
    # ... that pass did a REAL pull on the SAME link (the stamp went 0.0 -> set), so
    # the table was read for this request rather than answered from the not-due gate.
    assert observed["link_id_after"] == observed["link_id"], (
        "link identity: the link changed during the fresh pass "
        f"({observed['link_id']} -> {observed['link_id_after']})"
    )
    assert observed["pulled_stamp"] > 0.0, "the fresh pass did not pull the member table"
    final = observed["final"]
    assert final["answered"] == [observed["peer"]], final
    assert final["complete"] is True, final
    assert final["in_flight"] is False, final
    assert final["not_due"] == [], final
    assert observed["final_seq"] == observed["seq_after"], "reading the result ran another pass"


def _judge_status_verb(
    observed: dict[str, Any],
    record_property: Callable[[str, object], None],
    *,
    expect: str | None,
) -> None:
    """Record the receipts, run the oracle, and record WHICH guard spoke.

    The receipts are scalars and synthetic device/link ids only: sequence numbers,
    stamps, thread names, the cached table's four boolean/list fields — never a
    frame, an argument, an environment or a key. ``expect`` is ``None`` for a cell
    whose oracle must pass, or the text the failing guard must open with.
    """
    record_property("status_receipts", json.dumps(observed, sort_keys=True, default=str))
    record_property("status_relay_source", json.dumps(_relay_source_identity(), sort_keys=True))
    if expect is None:
        try:
            _assert_status_verb_asked_for_a_fresh_pass(observed)
        except AssertionError as error:
            record_property("status_oracle", "failed: " + _first_line(error))
            raise
        record_property("status_oracle", "passed")
        return
    with pytest.raises(AssertionError) as caught:
        _assert_status_verb_asked_for_a_fresh_pass(observed)
    record_property("status_oracle", _first_line(caught.value))
    assert expect in _first_line(caught.value, 1000), (
        f"the oracle failed, but not on the guard this control is about ({expect!r}): "
        f"{_first_line(caught.value, 1000)}"
    )


def test_the_status_verb_asks_over_the_control_socket_for_a_fresh_pass(
    devices: Any,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    """The whole hop: `relay.status(refresh=True)` → health → control frame → pass.

    The relay-level cells above drive ``RelayServer.status`` directly; this one
    proves the FIELD crosses the socket (the `net_status` handler reads `refresh`
    off the frame) and that the module-level read the CLI's `status` verb calls
    waits long enough for the answer — by reading a real relay through its real
    control socket.

    WHAT THIS CELL DOES NOT ASSERT: that the cache is empty before the verb. It
    used to, and a legitimate admit-time publication made that false on a loaded
    runner (``answered == [b]`` — CI main, 10 heads). The contract is the
    sequence: a plain read leaves it alone, a fresh read enters the relay's fresh
    read once and moves the sequence by exactly one real pull. The controls below
    prove the oracle rejects the two ways that could go wrong.
    """
    observed = _observe_status_verb(
        devices, monkeypatch, drop_refresh_field=False, publication_wait_s=10.0
    )
    _judge_status_verb(observed, record_property, expect=None)


def test_status_verb_control_a_dropped_refresh_field_is_rejected_for_not_asking(
    devices: Any,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    """Control: when the client's request loses ``refresh`` the oracle fails — for it.

    The intervention is at the wire (the control request is sent without the field),
    so the relay's handler sees a plain status: its fresh read is entered ZERO times
    where the positive case enters it once. The cached answer still lists the peer
    (``answered == [b]``), so an oracle that read that list would pass here; this one
    reads the handler entry, the sequence, the publisher and the pull stamp. The
    absence wait is short because it can only ever be a bound: the cadence is
    parked, so nothing else is going to publish into it.
    """
    observed = _observe_status_verb(
        devices, monkeypatch, drop_refresh_field=True, publication_wait_s=2.0
    )
    assert observed["dropped_refresh_fields"] == 1, observed
    assert observed["fresh_calls"] == [], observed
    assert observed["seq_after"] == observed["seq0"], observed
    assert observed["publishers"] == [], observed
    assert observed["pulled_stamp"] == 0.0, observed
    _judge_status_verb(
        observed,
        record_property,
        expect="the fresh request reached the relay's fresh read 0 time(s)",
    )


def test_status_verb_control_the_old_empty_assertion_fails_after_a_legitimate_pass(
    devices: Any,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    """Control: the old ``answered == []`` precondition is false once a REAL pass publishes.

    The pass is a real ``refresh_membership`` over the live link the relay holds,
    re-armed so it actually pulls: it stands in for the admit-time pass (same
    function, same link, same report) without depending on a thread's timing. It is
    a contract counterexample, NOT the pass that failed in CI — that publisher is
    still unidentified. The old assertion is then applied to the REAL data a plain
    status returns over the socket, and the pass's own receipts are checked so the
    data is known to be a pull and not a not-due answer.
    """
    a, b, record, _settled = _status_pair_after_admit(devices, monkeypatch)
    live = a.server._link_for(b.device_id)  # noqa: SLF001 — the link the relay holds now
    assert live is not None
    live.member_pulled_at = 0.0
    seq_before = a.server._membership_seq  # noqa: SLF001
    stamp_before = live.member_pulled_at
    publisher = threading.current_thread().name
    reports = a.server.refresh_membership()
    assert a.server._membership_seq == seq_before + 1  # noqa: SLF001
    assert live.member_pulled_at > 0.0, "the stand-in pass did not pull"
    assert reports[record.network_id].not_due == [], "the stand-in pass answered from not-due"
    # What the two asserts above established, kept in the artifact: this control's
    # pass is a direct call from the test thread (a stand-in), not the admit-time
    # publisher, so the receipts name the caller instead of implying a CI cause.
    record_property(
        "status_old_empty_receipts",
        json.dumps(
            {
                "seq_before": seq_before,
                "seq_after": a.server._membership_seq,  # noqa: SLF001
                "stamp_before": stamp_before,
                "stamp_after": live.member_pulled_at,
                "not_due": reports[record.network_id].not_due,
                "link_id": live.link_id,
                "publisher": publisher,
            },
            sort_keys=True,
        ),
    )

    row = next(r for r in relay.status()["networks"] if r["network_id"] == record.network_id)
    stale = row["membership"]["table"]
    record_property(
        "status_old_empty_table",
        json.dumps({key: stale[key] for key in ("answered", "complete", "in_flight", "not_due")}),
    )
    with pytest.raises(AssertionError) as caught:
        assert stale["answered"] == [], stale
    record_property("status_oracle", "old-empty: " + _first_line(caught.value))
    assert b.device_id in str(caught.value)
    assert stale["complete"] is True and stale["in_flight"] is False, stale
    assert stale["not_due"] == [], stale
    assert (
        a.server._membership_seq == seq_before + 1  # noqa: SLF001
    ), "reading the report ran another pass"


# ---------------------------------------------------------------------------
# The conditional member pull (``wire.MEMBER_LIST_DIGEST_V1``)
# ---------------------------------------------------------------------------
#
# WHY THESE EXIST. Every live link re-pulled the WHOLE member table every 15 s in
# each direction, and on a settled mesh that is the same table every time (measured
# 2026-11-08 on the two-relay rig: 1.43 KB per reply with 2 members, ~0.79 KB
# compressed with 6). The conditional pull answers "unchanged" instead — and the
# cells below pin the two ways that could go wrong: a table change hidden behind an
# "unchanged" answer, and an older build being sent a key it does not know.


def _without_digest_advertisement(monkeypatch: pytest.MonkeyPatch, device: Device) -> None:
    """Model an old build before its handshake can start any membership pulls.

    Capabilities decide the request at frame construction, not when the answer
    spy appends its receipt. Editing a live link therefore leaves already-built
    conditional requests in flight. Filter the advertisement instead: every link
    learns the old build's immutable capability set before it can send a frame.
    """
    original = Handshake.new

    def _new(**fields: Any) -> Handshake:
        if fields["identity"].device_id == device.device_id:
            fields["capabilities"] = [
                cap
                for cap in fields.get("capabilities", wire.LINK_CAPABILITIES)
                if cap != wire.MEMBER_LIST_DIGEST_V1
            ]
        return original(**fields)

    monkeypatch.setattr(Handshake, "new", _new)


def _spy_member_list(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record every ``net_member_list`` a relay ANSWERS: who, what was asked, what went back.

    Patched on the class BEFORE any relay is built: the op table binds the handler in
    ``RelayServer.__init__``, so a relay built first would keep the unpatched one.
    """
    seen: list[dict[str, Any]] = []
    original = relay.RelayServer._op_member_list  # noqa: SLF001 — the handler under test

    def _spy(self: relay.RelayServer, link: Any, frame: dict[str, Any]) -> dict[str, Any] | None:
        reply = original(self, link, frame)
        if reply is None:
            return None  # Change notifications are not answered pulls.
        detail = reply.get("detail") or {}
        seen.append(
            {
                "server": self.identity.device_id,
                "asker": link.device_id,
                "if_digest": "if_digest" in frame,
                "unchanged": detail.get("unchanged") is True,
                "rows": len(detail.get("members") or []),
            }
        )
        return reply

    monkeypatch.setattr(relay.RelayServer, "_op_member_list", _spy)
    return seen


def _synthetic_row(
    template: types.MemberRecord | None, tag: str, **fields: Any
) -> types.MemberRecord:
    """A real row's shape under a new id and key: what a third member looks like."""
    assert template is not None
    return replace(
        template,
        device_id=tag + template.device_id[len(tag) :],
        public_key=tag + template.public_key[len(tag) :],
        name=f"device-{tag}",
        **fields,
    )


def _row(device: Device, device_id: str) -> types.MemberRecord | None:
    record = store.load(store.list_networks(device.root)[0].network_id, device.root)
    return record.member(device_id)


def _changed(value: Any, name: str) -> Any:
    """A value of the same shape as ``value`` that differs from it."""
    if name == "rotation_proof":
        return {"old_device_id": "zz-claimed-predecessor"}
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float) or value is None:
        return 4242.0
    if isinstance(value, str):
        return value + "-moved"
    if isinstance(value, list):
        return [*value, "zz-extra"]
    if isinstance(value, dict):
        return {**value, "zz": "extra"}
    raise AssertionError(f"no changed-value rule for {name}: {type(value)}")


def test_the_sync_digest_moves_exactly_on_what_the_merge_can_act_on(tmp_path: Path) -> None:
    """Walk EVERY ``MemberRecord`` field against the rule, in both directions.

    A field the merge takes from a peer's row must move the digest (else an
    ``unchanged`` answer would withhold it); a field the merge keeps local must NOT
    (else two devices that legitimately disagree on it never settle — QA round 1, Q1:
    ``capabilities`` after a ``credential share``). A new field fails this cell until
    it is classified in ``relay.SYNC_DIGEST_*_FIELDS``.
    """
    import dataclasses

    server = relay.RelayServer(
        root=tmp_path / "x",
        settings=relay.NetworkSettings(port=0),
        identity=identity.mint(tmp_path / "x", name="x"),
        audit=audit_mod.AuditLog(tmp_path / "x"),
    )
    record = _init_network(server)
    sender = record.members[0].device_id
    record.members.append(_synthetic_row(record.members[0], "ff", endpoints=["10.0.0.1:1"]))
    adopted = relay.SYNC_DIGEST_ADOPTED_FIELDS | relay.SYNC_DIGEST_PARTIAL_FIELDS
    for index in (0, len(record.members) - 1):
        row = record.members[index]
        is_sender = row.device_id == sender
        base = relay.member_sync_digest(record, sender_device_id=sender)
        for field in dataclasses.fields(types.MemberRecord):
            moved = replace(row, **{field.name: _changed(getattr(row, field.name), field.name)})
            record.members[index] = moved
            try:
                after = relay.member_sync_digest(record, sender_device_id=sender)
            finally:
                record.members[index] = row
            # ``name`` is taken only from the SENDER's own row (rule N1).
            expect = field.name in adopted and (field.name != "name" or is_sender)
            assert (after != base) is expect, (field.name, "sender" if is_sender else "third")
    # Capabilities, the Q1 field, in the order a writer could leave them: no effect.
    row = record.members[-1]
    record.members[-1] = replace(row, capabilities=list(reversed(row.capabilities)))
    assert relay.member_sync_digest(record, sender_device_id=sender) == base


def test_a_settled_link_pulls_no_rows_and_every_change_still_lands(
    devices: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A settled link answers "unchanged"; an added row AND a rename still cross.

    Fails on a build without the conditional pull (no answer is ever ``unchanged``),
    and on one keyed on the authority digest (the rename never lands).
    """
    monkeypatch.setattr(relay, "MEMBERSHIP_PULL_MIN_INTERVAL_S", 1.0)
    # ...and the CEILING too: the backoff would otherwise climb to the real 30 s
    # inside this test and its own window would expire waiting for a pull.
    monkeypatch.setattr(relay, "MEMBERSHIP_PULL_MAX_INTERVAL_S", 2.0)
    monkeypatch.setattr(relay, "MEMBERSHIP_PUSH_PULL_INTERVAL_S", 2.0)
    monkeypatch.setattr(relay, "MEMBERSHIP_PULL_PASS_S", 0.25)
    seen = _spy_member_list(monkeypatch)
    a = _make(devices, "a", mode=DIAL_ONLY)
    c = _make(devices, "c", mode=HUB)
    record = _init_network(c.server)
    assert _join(a, inviter=c, monkeypatch=monkeypatch)["device_id"] == a.device_id
    link, reason = a.server.dial(record.network_id, host=f"{c.host}:{c.port}", epoch=1)
    assert link is not None, reason
    assert wire.MEMBER_LIST_DIGEST_V1 in link.capabilities

    def _c_answered_a(**match: Any) -> bool:
        return any(
            row["server"] == c.device_id
            and row["asker"] == a.device_id
            and all(row[k] == v for k, v in match.items())
            for row in list(seen)
        )

    assert net_fixtures.wait_for(
        lambda: _c_answered_a(unchanged=True, rows=0)
    ), f"a settled link never answered unchanged: {seen}"
    # The first pull of a link stays FULL whatever the digest says.
    assert any(not row["if_digest"] and row["rows"] for row in seen), seen

    # A ROW ADDED on C crosses: the digests now differ, so C answers with the table.
    with store.mutate(record.network_id, c.root) as copy:
        copy.members.append(
            _synthetic_row(copy.member(c.device_id), "ee", endpoints=["10.9.9.9:1"])
        )
        store.save(copy, c.root)
        late_id = copy.members[-1].device_id
    assert net_fixtures.wait_for(lambda: late_id in _members(a)), _members(a)

    # A RENAME of an existing row crosses too — a field the authority digest does
    # not cover and ``adopt_members`` does take (from the device the row names, so
    # C renames ITSELF; a third party's rename would be refused by design).
    with store.mutate(record.network_id, c.root) as copy:
        own = copy.member(c.device_id)
        assert own is not None
        own.name = "hub-renamed"
        store.save(copy, c.root)
    assert net_fixtures.wait_for(
        lambda: (row := _row(a, c.device_id)) is not None and row.name == "hub-renamed"
    ), _row(a, c.device_id)
    # ...and the link settles again afterwards.
    count = len(seen)
    assert net_fixtures.wait_for(
        lambda: any(row["unchanged"] and row["asker"] == a.device_id for row in seen[count:])
    ), seen[count:]

    # A RE-ESTABLISHED LINK STARTS FULL (review R1-m3): every connection is a new
    # ``PeerLink`` whose first pull re-reads the peer's rows whole, digest or not.
    for server in (a.server, c.server):
        for item in list(server.links.values()):  # noqa: SLF001
            item.close()
    count = len(seen)
    relinked, reason = a.server.dial(record.network_id, host=f"{c.host}:{c.port}", epoch=1)
    assert relinked is not None, reason
    assert net_fixtures.wait_for(
        lambda: any(r["asker"] == a.device_id and r["server"] == c.device_id for r in seen[count:])
    ), seen[count:]
    first = next(
        r for r in seen[count:] if r["asker"] == a.device_id and r["server"] == c.device_id
    )
    assert not first["if_digest"] and first["rows"] and not first["unchanged"], first
    assert net_fixtures.wait_for(
        lambda: any(r["unchanged"] and r["asker"] == a.device_id for r in seen[count:])
    ), seen[count:]


def test_a_share_to_a_drive_device_still_settles_to_unchanged(
    devices: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QA round 1, Q1: a LOCAL capability edit must not park a link on full pulls.

    ``credential share`` grants ``broker_credential`` on the OWNER's copy of the
    holder's row only (``relay.set_member_capabilities``, called by
    ``finite_share``); ``adopt_members`` never relays it, so the two ends disagree
    on that row's capabilities for good. A digest that hashed them never matched
    again and both directions fell back to main's full tables (measured: 0/4
    ``unchanged`` each way). Fails on a digest that covers ``capabilities``.
    """
    monkeypatch.setattr(relay, "MEMBERSHIP_PULL_MIN_INTERVAL_S", 1.0)
    # ...and the CEILING too: the backoff would otherwise climb to the real 30 s
    # inside this test and its own window would expire waiting for a pull.
    monkeypatch.setattr(relay, "MEMBERSHIP_PULL_MAX_INTERVAL_S", 2.0)
    monkeypatch.setattr(relay, "MEMBERSHIP_PUSH_PULL_INTERVAL_S", 2.0)
    monkeypatch.setattr(relay, "MEMBERSHIP_PULL_PASS_S", 0.25)
    seen = _spy_member_list(monkeypatch)
    a = _make(devices, "a", mode=DIAL_ONLY)
    c = _make(devices, "c", mode=HUB)
    record = _init_network(c.server)
    joined = _join(a, inviter=c, monkeypatch=monkeypatch, role="drive")
    assert joined["device_id"] == a.device_id
    link, reason = a.server.dial(record.network_id, host=f"{c.host}:{c.port}", epoch=1)
    assert link is not None, reason

    with store.mutate(record.network_id, c.root) as copy:
        relay.set_member_capabilities(copy, device_id=a.device_id, grant=["broker_credential"])
        store.save(copy, c.root)
    held_by_c = _row(c, a.device_id)
    held_by_a = _row(a, a.device_id)
    assert held_by_c is not None and held_by_a is not None
    assert "broker_credential" in held_by_c.capabilities
    assert "broker_credential" not in held_by_a.capabilities, "the rig did not diverge"

    count = len(seen)
    for asker, server in ((a, c), (c, a)):
        assert net_fixtures.wait_for(
            lambda asker=asker, server=server: any(
                r["unchanged"] and r["asker"] == asker.device_id and r["server"] == server.device_id
                for r in seen[count:]
            )
        ), (asker.name, seen[count:])


def test_an_unchanged_answer_for_another_digest_is_not_believed(tmp_path: Path) -> None:
    """Review R1-m2: "unchanged" counts only when it echoes the digest we sent.

    Anything else (a server bug, a skewed digest build) is not stamped as an answer:
    it falls through to the full-table checks and is reported as a bad answer, so
    the next pass asks again instead of the link silently claiming a fresh table.
    """
    from types import SimpleNamespace

    server = relay.RelayServer(
        root=tmp_path / "x",
        settings=relay.NetworkSettings(port=0),
        identity=identity.mint(tmp_path / "x", name="x"),
        audit=audit_mod.AuditLog(tmp_path / "x"),
    )
    record = _init_network(server)
    sent: list[dict[str, Any]] = []

    def _link(echo: str) -> Any:
        def request(frame: dict[str, Any], timeout: float) -> dict[str, Any]:
            sent.append(frame)
            digest = frame["if_digest"] if echo == "same" else "0" * 64
            return {
                "op": "ack",
                "req": frame["req"],
                "detail": {"unchanged": True, "digest": digest},
            }

        return SimpleNamespace(
            alive=True,
            network_id=record.network_id,
            device_id="peer-device",
            capabilities=frozenset({wire.MEMBER_LIST_DIGEST_V1}),
            member_pulled_at=1.0,
            member_pull_added=["stale"],
            local_capabilities=frozenset({wire.MEMBER_LIST_DIGEST_V1}),
            member_snapshot_sequence=-1,
            member_table_lock=threading.Lock(),
            request=request,
        )

    wrong = _link("other")
    assert server._pull_members(wrong) == "bad_answer"  # noqa: SLF001
    assert wrong.member_pulled_at == 1.0, "a mismatched unchanged was stamped as an answer"
    right = _link("same")
    assert server._pull_members(right) == ""  # noqa: SLF001
    assert right.member_pulled_at > 1.0 and right.member_pull_added == []
    assert all("if_digest" in frame for frame in sent)


def test_a_mixed_build_mesh_converges_and_the_old_side_never_sees_the_digest(
    devices: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One end of a link predates ``member-list-digest-v1``; the mesh still converges.

    Modelled at the negotiation, which is the only place the build difference is
    visible: A's handshake omits digest and change-push support, so C must keep
    asking full tables (an old A would not know ``if_digest``), while this test's
    client implementation can still ask the capable C conditionally. Capabilities
    are fixed BEFORE the handshake, not reassigned while an already-prepared
    conditional request may still be in flight. A row added on EITHER side arrives.
    """
    monkeypatch.setattr(relay, "MEMBERSHIP_PULL_MIN_INTERVAL_S", 1.0)
    # ...and the CEILING too: the backoff would otherwise climb to the real 30 s
    # inside this test and its own window would expire waiting for a pull.
    monkeypatch.setattr(relay, "MEMBERSHIP_PULL_MAX_INTERVAL_S", 2.0)
    monkeypatch.setattr(relay, "MEMBERSHIP_PUSH_PULL_INTERVAL_S", 2.0)
    monkeypatch.setattr(relay, "MEMBERSHIP_PULL_PASS_S", 0.25)
    seen = _spy_member_list(monkeypatch)
    a = _make(devices, "a", mode=DIAL_ONLY)
    c = _make(devices, "c", mode=HUB)
    new_handshake = Handshake.new
    old_features = {wire.MEMBER_LIST_DIGEST_V1, wire.MEMBER_LIST_PUSH_V1}

    def handshake_for_build(**kwargs: Any) -> Handshake:
        if kwargs["identity"].device_id == a.device_id:
            kwargs["capabilities"] = [
                cap for cap in kwargs["capabilities"] if cap not in old_features
            ]
        return new_handshake(**kwargs)

    monkeypatch.setattr(Handshake, "new", staticmethod(handshake_for_build))
    pushes: list[dict[str, Any]] = []
    send = relay.PeerLink.send

    def record_send(self: relay.PeerLink, frame: dict[str, Any], **kwargs: Any) -> bool:
        if "changed_members" in frame:
            pushes.append(frame)
        return send(self, frame, **kwargs)

    monkeypatch.setattr(relay.PeerLink, "send", record_send)
    record = _init_network(c.server)
    assert _join(a, inviter=c, monkeypatch=monkeypatch)["device_id"] == a.device_id
    link, reason = a.server.dial(record.network_id, host=f"{c.host}:{c.port}", epoch=1)
    assert link is not None, reason
    assert net_fixtures.wait_for(lambda: c.server._link_for(a.device_id) is not None)
    reverse = c.server._link_for(a.device_id)
    assert reverse is not None
    assert not old_features.intersection(reverse.capabilities)
    assert not old_features.intersection(link.local_capabilities)
    count = len(seen)

    for owner, tag in ((c, "e1"), (a, "e2")):
        with store.mutate(record.network_id, owner.root) as copy:
            copy.members.append(_synthetic_row(copy.member(c.device_id), tag))
            store.save(copy, owner.root)
    both = {row.device_id for row in store.load(record.network_id, c.root).active_members()}
    both |= {row.device_id for row in store.load(record.network_id, a.root).active_members()}
    assert net_fixtures.wait_for(lambda: _members(a) == both and _members(c) == both), (
        _members(a),
        _members(c),
        both,
    )
    # And after convergence A's conditional pulls settle to "unchanged" (the new side
    # still saves), while every pull C made of the "old" A stayed full.
    assert net_fixtures.wait_for(
        lambda: any(r["unchanged"] and r["asker"] == a.device_id for r in seen[count:])
    ), seen[count:]
    # Check EVERY C request, including first contact: an old negotiated receiver
    # must never see a digest or a push, not merely after a mutable test cutoff.
    from_c = [r for r in seen if r["asker"] == c.device_id]
    assert from_c and all(not r["if_digest"] and r["rows"] for r in from_c), from_c
    assert not pushes
    assert reverse.member_pull_interval_s <= relay.MEMBERSHIP_PULL_MAX_INTERVAL_S


# ---------------------------------------------------------------------------
# Pull FREQUENCY: a settled link backs off, and nothing goes stale
# ---------------------------------------------------------------------------
#
# #3608 made every pull small (297 B on the wire). What was left was the RATE:
# both ends of every link asked four times a minute for a table that had not
# moved. These cells pin the backoff that answers it, and — the part that
# matters — that convergence and the freshness sentence survive it.

#: THE COMPRESSED CLOCK THE BACKOFF RUNS ON, and why it takes THREE numbers the
#: way production takes three: the floor and the ceiling define the doubling
#: (0.2 -> 0.4 -> 0.8 s) and the cadence is the granularity of "due". Small
#: enough to run, large enough that "backed off" and "due" are separated by an
#: assertion instead of a race.
_BACKOFF_FLOOR_S = 0.2
_BACKOFF_CEILING_S = 0.8
_BACKOFF_PASS_S = 0.05


@pytest.fixture()
def backoff_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(relay, "MEMBERSHIP_PULL_MIN_INTERVAL_S", _BACKOFF_FLOOR_S)
    monkeypatch.setattr(relay, "MEMBERSHIP_PULL_MAX_INTERVAL_S", _BACKOFF_CEILING_S)
    # Both negotiated ceilings follow this fixture's compressed clock; production
    # push-only convergence and the real five-minute ceiling have separate tests.
    monkeypatch.setattr(relay, "MEMBERSHIP_PUSH_PULL_INTERVAL_S", _BACKOFF_CEILING_S)
    monkeypatch.setattr(relay, "MEMBERSHIP_PULL_PASS_S", _BACKOFF_PASS_S)


def _settled_pair(devices: Any, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any, str, Any]:
    """A dial-only leaf joined to a hub and linked to it: the two-relay baseline."""
    a = _make(devices, "a", mode=DIAL_ONLY)
    c = _make(devices, "c", mode=HUB)
    record = _init_network(c.server)
    assert _join(a, inviter=c, monkeypatch=monkeypatch)["device_id"] == a.device_id
    link, reason = a.server.dial(record.network_id, host=f"{c.host}:{c.port}", epoch=1)
    assert link is not None, reason
    return a, c, record.network_id, link


def _backed_off(a: Any, c: Any) -> bool:
    """Both ends of the pair have settled: each of their OWN pull intervals is at the ceiling.

    BOTH, because each end schedules its own asks — a bound proved for one
    direction says nothing about the other, and every change that crosses this
    pull crosses it in the direction of the device that asks.
    """
    mine = a.server._link_for(c.device_id)  # noqa: SLF001
    theirs = c.server._link_for(a.device_id)  # noqa: SLF001
    return (
        mine is not None
        and theirs is not None
        and mine.member_pull_interval_s >= _BACKOFF_CEILING_S
        and theirs.member_pull_interval_s >= _BACKOFF_CEILING_S
    )


def _rename_own_row(device: Any, network_id: str, name: str) -> None:
    """Rename ``device``'s own row, the change that travels ONLY by this pull."""
    with store.mutate(network_id, device.root) as copy:
        own = copy.member(device.device_id)
        assert own is not None
        own.name = name
        store.save(copy, device.root)


def test_a_settled_link_backs_off_and_a_rename_still_lands_within_the_bound(
    devices: Any, monkeypatch: pytest.MonkeyPatch, backoff_clock: None
) -> None:
    """The frequency half: a quiet link asks less, and a rename still crosses.

    THE BOUND THIS PINS IS THE WHOLE OF THE CONCESSION. A rename of the sender's
    own row travels by this pull alone (``member_sync_digest`` hashes ``name`` for
    the sender's row only), so on a backed-off link its worst case is the ceiling
    plus one cadence of scheduling slop. Measured here rather than recomputed from
    the constants, because the constants are the only thing the test supplies.
    """
    seen = _spy_member_list(monkeypatch)
    a, c, network_id, link = _settled_pair(devices, monkeypatch)
    assert net_fixtures.wait_for(lambda: _backed_off(a, c)), (
        link.member_pull_interval_s,
        getattr(c.server._link_for(a.device_id), "member_pull_interval_s", None),  # noqa: SLF001
        seen,
    )

    # A LINK AT THE CEILING IS NOT ASKED AT THE FLOOR RATE. An age past the floor
    # and short of the ceiling costs a walk and no traffic at all.
    link.member_pulled_at = time.time() - (_BACKOFF_FLOOR_S + 0.1)
    before = len(seen)
    report = a.server.refresh_membership()[network_id]
    assert not [row for row in seen[before:] if row["asker"] == a.device_id], seen[before:]
    assert [row["device_id"] for row in report.not_due] == [c.device_id], report.not_due

    _rename_own_row(c, network_id, "hub-renamed")
    started = time.monotonic()
    # Slop is the cadence plus this host's own scheduling, not a second ceiling.
    bound = _BACKOFF_CEILING_S + _BACKOFF_PASS_S + 1.0
    assert net_fixtures.wait_for(
        lambda: (row := _row(a, c.device_id)) is not None and row.name == "hub-renamed",
        timeout_s=bound,
    ), _row(a, c.device_id)
    elapsed = time.monotonic() - started
    assert elapsed <= bound, (elapsed, bound)


def test_a_change_resets_the_interval_to_the_floor_immediately(
    devices: Any, monkeypatch: pytest.MonkeyPatch, backoff_clock: None
) -> None:
    """A change does not wait out the backoff it just invalidated.

    Two halves, and the second is the one that would rot: the interval goes back to
    the floor on the pull that brought the change, AND the schedule then honours
    it — a backdated stamp past the floor is asked at once, which is exactly what
    the backed-off link above does NOT do at that age.
    """
    seen = _spy_member_list(monkeypatch)
    a, c, network_id, link = _settled_pair(devices, monkeypatch)
    assert net_fixtures.wait_for(lambda: _backed_off(a, c)), link.member_pull_interval_s

    _rename_own_row(c, network_id, "hub-renamed")
    link.member_pulled_at = time.time() - (_BACKOFF_CEILING_S + 0.05)
    a.server.refresh_membership()
    row = _row(a, c.device_id)
    assert row is not None and row.name == "hub-renamed", row
    assert link.member_pull_interval_s == _BACKOFF_FLOOR_S, link.member_pull_interval_s

    # THE RESET IS THE SCHEDULE, NOT A FIELD: at an age past the floor the peer is
    # asked again, where a link still on the ceiling would not be.
    link.member_pulled_at = time.time() - (_BACKOFF_FLOOR_S + 0.05)
    before = len(seen)
    a.server.refresh_membership()
    assert [
        row for row in seen[before:] if row["asker"] == a.device_id and row["server"] == c.device_id
    ], seen[before:]


def test_a_removal_lands_while_the_link_is_backed_off(
    devices: Any, monkeypatch: pytest.MonkeyPatch, backoff_clock: None
) -> None:
    """A removal is pushed, so the backoff costs it nothing.

    The half of the bound that a removal does NOT pay: it travels by ``net_epoch``
    / ``removed_ids``, not by this pull, so it lands faster than even one
    backed-off interval — the reason the ceiling may be chosen for the rename.
    """
    a, c, network_id, _link = _settled_pair(devices, monkeypatch)
    b = _make(devices, "b", mode=HUB)
    assert _join(b, inviter=c, monkeypatch=monkeypatch)["device_id"] == b.device_id
    assert net_fixtures.wait_for(lambda: _backed_off(a, c))

    started = time.monotonic()
    outcome = c.server._ctl_member_rm(  # noqa: SLF001 — the CLI's own control op
        {"network": network_id, "device_id": b.device_id}
    )
    assert outcome.get("ok") is not False, outcome
    assert net_fixtures.wait_for(
        lambda: (row := _row(a, b.device_id)) is not None and row.active is False,
        timeout_s=_BACKOFF_CEILING_S + _BACKOFF_PASS_S + 1.0,
    ), _row(a, b.device_id)
    elapsed = time.monotonic() - started
    assert elapsed < _BACKOFF_CEILING_S, (elapsed, "a removal must not wait out a pull")


def test_a_peer_without_the_digest_capability_never_backs_off(
    devices: Any, monkeypatch: pytest.MonkeyPatch, backoff_clock: None
) -> None:
    """Mixed versions: a link whose peer cannot answer ``unchanged`` keeps today's rate.

    The capability is a HANDSHAKE property: C never advertises it, including on
    the establishment pull. Every ask is unconditional, every answer is a full
    table, and nothing the peer says proves the two tables agree — so the interval
    may not leave the floor. This is "an older peer is unaffected" in one place:
    the backoff is unreachable without a digest-verified ``unchanged``.
    """
    seen = _spy_member_list(monkeypatch)
    a = _make(devices, "a", mode=DIAL_ONLY)
    c = _make(devices, "c", mode=HUB)
    _without_digest_advertisement(monkeypatch, c)
    record = _init_network(c.server)
    network_id = record.network_id
    assert _join(a, inviter=c, monkeypatch=monkeypatch)["device_id"] == a.device_id
    link, reason = a.server.dial(network_id, host=f"{c.host}:{c.port}", epoch=1)
    assert link is not None, reason
    assert wire.MEMBER_LIST_DIGEST_V1 not in link.capabilities
    assert net_fixtures.wait_for(
        lambda: link.member_pulled_at > 0.0 and link.member_pull_interval_s == _BACKOFF_FLOOR_S
    ), "the old peer's establishment pull never completed at the floor interval"
    mark = 0  # All requests, not merely answers recorded after a live capability edit.

    for _ in range(3):
        # Past every interval, so "due" is never the question this loop is asking.
        link.member_pulled_at = time.time() - (_BACKOFF_CEILING_S + 0.05)
        a.server.refresh_membership()
        assert link.member_pull_interval_s == _BACKOFF_FLOOR_S, link.member_pull_interval_s

    asked = [
        row for row in seen[mark:] if row["asker"] == a.device_id and row["server"] == c.device_id
    ]
    assert asked, seen[mark:]
    assert all(
        not row["if_digest"] and row["rows"] and not row["unchanged"] for row in asked
    ), asked


def test_a_backed_off_link_is_still_reported_as_answered_with_its_age(
    devices: Any, monkeypatch: pytest.MonkeyPatch, backoff_clock: None
) -> None:
    """`lop network status` stays truthful while a link is backed off.

    NOT ASKED IS NOT UNANSWERED. The digest answer already stamped "this peer's
    table answered"; a pass that skips the ask must keep reporting the peer as
    answered, with the TRUE age of that answer, so the sentence it renders says
    "members verified with all 1 peer(s) (N ago)" rather than "no table came back".
    The age is the assertion that matters: a report that quietly presented a
    backed-off answer as fresh would be the same bug from the other side.
    """
    seen = _spy_member_list(monkeypatch)
    a, c, network_id, link = _settled_pair(devices, monkeypatch)
    assert net_fixtures.wait_for(lambda: _backed_off(a, c)), link.member_pull_interval_s

    link.member_pulled_at = time.time() - (_BACKOFF_FLOOR_S + 0.1)
    before = len(seen)
    report = a.server.refresh_membership()[network_id]
    assert not [row for row in seen[before:] if row["asker"] == a.device_id], seen[before:]
    assert report.answered == [c.device_id], report
    assert report.silent == [] and report.complete is True, report
    age = report.oldest_answer_age_s
    assert age is not None and age >= _BACKOFF_FLOOR_S, age
    payload = report.to_json()
    assert payload["not_answered"] == [] and payload["answered"] == [c.device_id], payload
    assert payload["oldest_answer_age_s"] is not None
    assert payload["sentence"].startswith("members verified with all 1 peer(s)"), payload[
        "sentence"
    ]


def test_every_member_field_is_classified_for_the_sync_digest() -> None:
    """Review R2-m1: the three digest sets PARTITION ``fields(MemberRecord)``.

    WHY THE PARTITION AND NOT JUST THE BEHAVIOUR: the per-field cell above proves
    every field this build has behaves as the rule says, and passes just as happily
    for a NEW field nobody hashed — it simply falls outside both hashed sets, so
    ``expect=False`` holds and a field the merge might be taught to adopt is
    silently classed local. That is the one direction that withholds a change.
    Spelling the local set out turns "a field was added" into a failure until
    somebody classifies it.
    """
    import dataclasses

    every = {field.name for field in dataclasses.fields(types.MemberRecord)}
    adopted = relay.SYNC_DIGEST_ADOPTED_FIELDS
    partial = relay.SYNC_DIGEST_PARTIAL_FIELDS
    local = relay.SYNC_DIGEST_LOCAL_FIELDS
    assert not (adopted & partial), adopted & partial
    assert not (adopted & local), adopted & local
    assert not (partial & local), partial & local
    assert adopted | partial | local == every, {
        "unclassified": every - (adopted | partial | local),
        "invented": (adopted | partial | local) - every,
    }
