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

import json
import os
import threading
import time
from argparse import Namespace
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


def _mint_invite(server: relay.RelayServer, record: types.NetworkRecord) -> tuple[str, Any]:
    state = store.load_secrets(record.network_id, server.root)
    minted = invite_mod.mint(record, state.secret, role="admin", ttl_s=600.0)
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
    token, envelope = _mint_invite(inviter.server, record)
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


def _recorded(device: Device, word: str, *, timeout_s: float = 15.0) -> bool:
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
    below with the whole trail in the message.
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
    monkeypatch.setattr(relay, "MEMBERSHIP_PULL_PASS_S", 0.25)

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

    assert _members(a) == {a.device_id, b.device_id, c.device_id}
    assert _link_peers(a) == {c.device_id}
    assert _link_peers(b) == {c.device_id}

    # D IS ADMITTED LATE, by B, and holds no link to anybody afterwards: the pairing
    # socket is the CLI's, and it closes when the ceremony ends. B's record knows D;
    # nobody else's does yet.
    assert _join(d, inviter=b, monkeypatch=monkeypatch)["device_id"] == d.device_id
    assert _members(b) == {a.device_id, b.device_id, c.device_id, d.device_id}
    assert _members(a) == {a.device_id, b.device_id, c.device_id}
    assert _members(c) == {a.device_id, b.device_id, c.device_id}

    # FROM HERE NOTHING IS TOUCHED, NOBODY IS RESTARTED, AND A IS WATCHED FOR DIALS.
    dials: list[str] = []
    original = a.server._ensure_link_with_reason  # noqa: SLF001 — the dialling path

    def _spy(device_id: str, **fields: Any) -> Any:
        dials.append(device_id)
        return original(device_id, **fields)

    a.server._ensure_link_with_reason = _spy  # type: ignore[method-assign]  # noqa: SLF001

    deadline = time.time() + 30.0
    while time.time() < deadline and d.device_id not in _members(a):
        time.sleep(0.1)

    assert _members(a) == {
        a.device_id,
        b.device_id,
        c.device_id,
        d.device_id,
    }, "the late member never became visible over a link that was already up"
    assert _members(c) == {a.device_id, b.device_id, c.device_id, d.device_id}
    # NOTHING WAS DIALED TO MAKE THIS HAPPEN — the refresh runs on the links that
    # exist, which is the property that makes it independent of a mesh's shape.
    assert dials == [], dials
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
    deadline = time.time() + 5.0
    while time.time() < deadline:
        removed = store.load(record.network_id, b.root)
        if removed.epoch == 2:
            break
        time.sleep(0.05)

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
    seen: list[str] = []
    original = a_link._handle  # noqa: SLF001

    def _spy(frame: dict[str, Any]) -> None:
        seen.append(str(frame.get("op")))
        return original(frame)

    a_link._handle = _spy  # type: ignore[method-assign]  # noqa: SLF001
    link.send({"op": "net_member_list", "req": 424242, "locality": "remote"})
    link.close("we-closed")
    deadline = time.time() + 3.0
    while time.time() < deadline and not seen:
        time.sleep(0.05)

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
    """`status` must not answer from the last cadence tick: it asks for a read.

    With the background cadence parked well past this test's lifetime, the only
    pass that can complete between the dial and the read is the one the read asks
    for (``RelayServer._fresh_membership_read``); a `status` that merely read
    ``_membership_reports`` would keep reporting the pre-dial pass, whose table
    never saw this link.
    """
    monkeypatch.setattr(relay, "MEMBERSHIP_PULL_PASS_S", 30.0)
    a = _make(devices, "a", mode=HUB)
    b = _make(devices, "b", mode=HUB)
    record = _init_network(a.server)
    assert _join(b, inviter=a, monkeypatch=monkeypatch)["device_id"] == b.device_id
    link, reason = b.server.dial(record.network_id, host=f"{a.host}:{a.port}", epoch=1)
    assert link is not None, reason

    def _row(payload: dict[str, Any]) -> dict[str, Any]:
        return next(row for row in payload["networks"] if row["network_id"] == record.network_id)

    # Before the read asks: the last completed pass predates the link, so b is not
    # in `answered` — there is no fresh table for the read to lean on yet.
    before = _row(a.server.status())["membership"]["table"]  # noqa: SLF001
    assert before["answered"] == [], before
    assert before["complete"] is False

    # The read asks, and the pass it asked for lands inside the bound.
    after = _row(a.server.status(refresh=True))["membership"]["table"]  # noqa: SLF001
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
        lambda: a_link.member_pulled_at > 0.0, timeout_s=10.0
    ), "a's establishment pull never stamped the link"

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
    assert net_fixtures.wait_for(
        entered.is_set, timeout_s=10.0
    ), "the kicked pass never entered the pull"
    assert inflight, "the read waited the hung pass out instead of returning with it in flight"


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


def test_the_status_verb_asks_over_the_control_socket_for_a_fresh_pass(
    devices: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole hop: `relay.status(refresh=True)` → health → control frame → pass.

    The relay-level cells above drive ``RelayServer.status`` directly; this one
    proves the FIELD crosses the socket (the `net_status` handler reads `refresh`
    off the frame) and that the module-level read the CLI's `status` verb calls
    waits long enough for the answer — by reading a real relay through its real
    control socket.
    """
    monkeypatch.setattr(relay, "MEMBERSHIP_PULL_PASS_S", 30.0)
    a = _make(devices, "a", mode=HUB)
    b = _make(devices, "b", mode=HUB)
    record = _init_network(a.server)
    assert _join(b, inviter=a, monkeypatch=monkeypatch)["device_id"] == b.device_id
    link, reason = b.server.dial(record.network_id, host=f"{a.host}:{a.port}", epoch=1)
    assert link is not None, reason

    # The ambient root is what `health()` scans for the relay's record; a's own
    # store holds one (published at start, re-stamped by its heartbeat).
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(a.root))

    def _row(payload: dict[str, Any]) -> dict[str, Any]:
        return next(row for row in payload["networks"] if row["network_id"] == record.network_id)

    stale = _row(relay.status())["membership"]["table"]
    assert stale["answered"] == [], stale
    fresh = _row(relay.status(refresh=True))["membership"]["table"]
    assert fresh["answered"] == [b.device_id], fresh
    assert fresh["complete"] is True
