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
    learned = [
        row
        for row in a.server.audit.tail(limit=500)  # noqa: SLF001 — the relay's own log
        if row.get("event") == "membership_learned"
    ]
    assert learned, _events(a)
    assert d.device_id in learned[-1]["detail"]["added"]
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
    a.server._ctl_member_rm(  # noqa: SLF001 — the CLI's own control op
        {"network": record.network_id, "device_id": b.device_id}
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
    # cannot be re-admitted — which is the next test.)
    b.server._clear_refusal_mark(marked)  # noqa: SLF001
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
