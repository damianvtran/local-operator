"""A ``credential share`` reaches a DEFAULT-INVITED borrower, over two real relays.

WHY THIS FILE EXISTS (mesh perf matrix, share-propagation finding, 5/5 >120 s). After
``lop network invite --role drive`` and ``lop network credential share … --with B``, B
never received the share: B's cadence gated its placement pull on its OWN admission-time
``self_capabilities`` (no ``broker_credential`` — ``drive`` does not carry it), while the
share grants that capability only on the OWNER's row for B, which is the row the owner's
authoriser actually reads. A manual pull worked in one round trip, so the gate was the
whole defect.

The four properties pinned here, each failing on the pre-fix build:

* the cadence pulls past the stale self-view, so the share arrives on B's own tick;
* the share verb pushes the new row to B, so it lands before the command returns;
* an UNSHARED default-invited device is still refused by the owner, and the refusal
  parks B's cadence instead of becoming a 15 s refusal loop on the owner's audit log;
* a borrower build without the push is told nothing new and still converges by pull.

Fixtures are the real-link suite's: two serve-shaped relays on loopback, the shipped
parser, a real ``AuthStore`` on the owner. The ambient config root is the OWNER's
(``AuthStore`` reads it); the borrower is driven only through root-explicit APIs.
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

from local_operator.network import store, wire
from local_operator.network.credentials import placement as placement_mod
from local_operator.network.credentials import sync as cred_sync
from local_operator.network.credentials.client import MeshCredentialClient
from local_operator.network.credentials.types import Grant
from tests.unit.network import conftest as net_fixtures
from tests.unit.network.test_credentials_owner import (  # noqa: F401 — fixtures by import
    STUB_PROVIDER,
    RotatingIdP,
    idp,
    stub_provider,
)
from tests.unit.network.test_credentials_real_link import _lop_network
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _pair,
    devices,
)


@pytest.fixture()
def drive_mesh(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Owner A (admin, relay up, a LIVE login) and borrower B admitted with ``--role drive``.

    Both relays listen and know each other's address, so either side can dial: the push
    is A dialling B, the cadence pull is B dialling A.
    """
    request.getfixturevalue("stub_provider")
    stub_idp: RotatingIdP = request.getfixturevalue("idp")
    server_a, server_b, host, port = request.getfixturevalue("devices")
    record, _host, _port = _pair((server_a, server_b, host, port), monkeypatch, role="drive")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
    host_b, port_b = server_b.bind()
    server_b.bind_control()
    with store.mutate(record.network_id, server_b.root) as copy:
        owner_row = copy.member(server_a.identity.device_id)
        assert owner_row is not None
        owner_row.endpoints = [f"{host}:{port}"]
        store.save(copy, server_b.root)
    with store.mutate(record.network_id, server_a.root) as copy:
        borrower_row = copy.member(server_b.identity.device_id)
        assert borrower_row is not None
        borrower_row.endpoints = [f"{host_b}:{port_b}"]
        store.save(copy, server_a.root)
    from local_operator.providers.auth_store import AuthStore

    auth = AuthStore(config_dir=server_a.root)
    auth.upsert_credential(
        STUB_PROVIDER,
        {
            "type": "oauth",
            "access": "owner-live-access",
            "expires": int(time.time() * 1000) + 3_600_000,
            "refresh": stub_idp.current_refresh,
            "email": "owner@example.test",
        },
    )
    auth.close()
    yield type(
        "DriveMesh",
        (),
        {
            "a": server_a,
            "b": server_b,
            "network_id": record.network_id,
            "owner": server_a.identity.device_id,
            "borrower": server_b.identity.device_id,
            "idp": stub_idp,
            "b_endpoint": f"{host_b}:{port_b}",
            "a_endpoint": f"{host}:{port}",
        },
    )
    server_b.stop()


def _start_b(mesh: Any) -> None:
    mesh.b.start()


def _b_holds(mesh: Any) -> bool:
    document = placement_mod.PlacementDocument.load(
        mesh.network_id, mesh.b.root, self_device=mesh.borrower
    )
    return document.is_holder(STUB_PROVIDER, mesh.borrower)


def _share_json(mesh: Any, capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    capsys.readouterr()
    code = _lop_network(
        "credential", "share", STUB_PROVIDER, "--with", mesh.b.identity.name, "--json"
    )
    assert code == 0
    return json.loads(capsys.readouterr().out)


def _refusals_on_owner(mesh: Any) -> int:
    mesh.a.audit.flush()
    return sum(
        1
        for row in mesh.a.audit.tail(2_000)
        if row.get("event") == "authorisation_refused"
        and row.get("actor") == mesh.borrower
        and (row.get("detail") or {}).get("op") == "net_broker"
    )


def test_the_default_invite_really_lacks_the_capability_on_its_own_row(drive_mesh: Any) -> None:
    """The precondition the defect rests on, asserted so the cells below are not vacuous."""
    from local_operator.network import definitions

    _start_b(drive_mesh)
    record = store.load(drive_mesh.network_id, drive_mesh.b.root)
    assert "broker_credential" not in record.self_capabilities
    assert (
        definitions.unholdable_capability(drive_mesh.b, drive_mesh.owner, "net_broker")
        == "broker_credential"
    )


def test_a_default_invite_share_arrives_on_the_borrowers_own_cadence(
    drive_mesh: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Push disabled: B's tick step alone must deliver the share (pre-fix: never)."""
    # Imported here, not at module level, so the pre-fix build (no such module) fails
    # this cell on the cadence assertion below rather than at collection.
    try:
        from local_operator.network.credentials import share_push
    except ImportError:
        pass
    else:
        monkeypatch.setattr(
            share_push,
            "push_share",
            lambda *a, **k: {"delivered": False, "outcome": "skipped:test_disabled"},
        )
    _start_b(drive_mesh)
    payload = _share_json(drive_mesh, capsys)
    # ``.get``: the pre-fix build has no push and no ``delivery`` field, and this cell
    # must fail there on the CADENCE, which is the defect it pins.
    assert payload.get("delivery", {}).get("delivered") is not True
    assert not _b_holds(drive_mesh), "the share arrived without the cadence under test"

    cred_sync.credentials_sync_step(drive_mesh.b, drive_mesh.owner)

    assert _b_holds(drive_mesh), "B's cadence never pulled the share past its stale self-view"
    client = MeshCredentialClient.for_this_device(drive_mesh.b.root)
    assert client is not None and client.should_borrow(STUB_PROVIDER)


def test_a_share_lands_within_one_round_trip_of_the_push(
    drive_mesh: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """B's cadence is switched off: only the push can deliver, before the verb returns."""
    monkeypatch.setattr(
        cred_sync,
        "_pull_past_stale_self_view",
        lambda *a, **k: "skipped:test_disabled",
        raising=False,
    )
    _start_b(drive_mesh)
    started = time.monotonic()
    payload = _share_json(drive_mesh, capsys)
    elapsed = time.monotonic() - started

    assert payload["delivery"] == {"delivered": True, "outcome": "pushed"}, payload
    # No wait, no tick: the row is on B's disk when the command returns.
    assert _b_holds(drive_mesh)
    # A real borrow through it (the owner still decides; this is not a local grant).
    grant = MeshCredentialClient.for_this_device(drive_mesh.b.root)
    assert grant is not None
    served = grant.request_grant_sync(STUB_PROVIDER, session_id="sess-1")
    assert isinstance(served, Grant), served
    # A loose structural bound, not a latency budget: well under one 15 s tick.
    assert elapsed < 10.0, elapsed


def test_the_human_receipt_says_the_share_was_delivered(
    drive_mesh: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    _start_b(drive_mesh)
    capsys.readouterr()
    assert (
        _lop_network("credential", "share", STUB_PROVIDER, "--with", drive_mesh.b.identity.name)
        == 0
    )
    out = capsys.readouterr().out
    assert "usable there: now" in out, out
    # The name is said once, on the receipt's first line (design review D3).
    assert out.count(drive_mesh.b.identity.name) == 1, out


def test_an_unshared_default_invite_is_still_refused_and_parks_its_cadence(
    drive_mesh: Any,
) -> None:
    """No share: the owner refuses B's pull at its chokepoint; B learns nothing and backs off."""
    _start_b(drive_mesh)
    before = _refusals_on_owner(drive_mesh)

    first = cred_sync.credentials_sync_step(drive_mesh.b, drive_mesh.owner)
    assert first == "skipped:no_broker_credential"
    assert net_fixtures.wait_for(lambda: _refusals_on_owner(drive_mesh) == before + 1)
    assert not _b_holds(drive_mesh)
    assert MeshCredentialClient.for_this_device(drive_mesh.b.root) is None
    # And the OWNER refuses a borrow outright, whatever B believes: the ungated pull
    # widened nothing, the owner's row for B is still the enforcement point.
    asker = MeshCredentialClient.for_relay(drive_mesh.b)
    assert asker is not None
    reply = asker._ask_owner_directly(  # noqa: SLF001 — the leg-2 seam, by hand
        drive_mesh.owner,
        {
            "op": "net_broker",
            "kind": "grant",
            "network_id": drive_mesh.network_id,
            "from_device": drive_mesh.borrower,
            "key": STUB_PROVIDER,
        },
    )
    assert isinstance(reply, dict) and reply.get("op") == "error", reply
    assert "access" not in json.dumps(reply)
    before = _refusals_on_owner(drive_mesh)

    # Parked: the next ticks ask nothing, so the owner's audit log does not grow.
    for _ in range(3):
        assert (
            cred_sync.credentials_sync_step(drive_mesh.b, drive_mesh.owner)
            == "skipped:no_broker_credential"
        )
    time.sleep(0.5)
    assert _refusals_on_owner(drive_mesh) == before, "a parked cadence asked the owner again"


def test_a_share_push_clears_the_parked_refusal(
    drive_mesh: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    _start_b(drive_mesh)
    cred_sync.credentials_sync_step(drive_mesh.b, drive_mesh.owner)
    assert drive_mesh.owner in cred_sync._owner_refusals(drive_mesh.b)  # noqa: SLF001
    assert _share_json(drive_mesh, capsys)["delivery"]["delivered"] is True
    assert drive_mesh.owner not in cred_sync._owner_refusals(drive_mesh.b)  # noqa: SLF001


def test_a_borrower_without_the_push_is_not_pushed_to_and_still_converges(
    drive_mesh: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Mixed versions: B's build predates ``credential-share-push-v1``.

    The owner must not send the frame (an old B would refuse it by name or merge it
    through the legacy path), must say the share is not yet delivered, and B's own pull
    must still converge.
    """
    from local_operator.network.credentials import share_push

    monkeypatch.setattr(
        wire,
        "LINK_CAPABILITIES",
        tuple(c for c in wire.LINK_CAPABILITIES if c != wire.CREDENTIAL_SHARE_PUSH_V1),
    )
    _start_b(drive_mesh)
    sent: list[dict[str, Any]] = []
    from local_operator.network import relay as relay_mod

    original = relay_mod.PeerLink.request

    def _record(self: Any, frame: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        sent.append(dict(frame))
        return original(self, frame, *args, **kwargs)

    monkeypatch.setattr(relay_mod.PeerLink, "request", _record)
    payload = _share_json(drive_mesh, capsys)
    assert payload["delivery"] == {
        "delivered": False,
        "outcome": "skipped:peer_lacks_share_push",
    }, payload
    assert not any(f.get("origin") == share_push.SHARE_PUSH_ORIGIN for f in sent)
    assert not _b_holds(drive_mesh)
    cred_sync.credentials_sync_step(drive_mesh.b, drive_mesh.owner)
    assert _b_holds(drive_mesh)


# ---------------------------------------------------------------------------
# The receiver's narrowing, without a link: what a share push may NOT do
# ---------------------------------------------------------------------------


class _Link:
    def __init__(self, device_id: str, network_id: str) -> None:
        self.device_id = device_id
        self.network_id = network_id
        self.capabilities: tuple[str, ...] = (wire.CREDENTIAL_SHARE_PUSH_V1,)
        self.local_capabilities: tuple[str, ...] = (wire.CREDENTIAL_SHARE_PUSH_V1,)


def _frame(network_id: str, sender: str, holder: str, owner: str = "") -> dict[str, Any]:
    from local_operator.network.credentials import share_push

    return {
        "op": "net_broker",
        "kind": "placement",
        "origin": share_push.SHARE_PUSH_ORIGIN,
        "from_device": sender,
        "document": {
            "schema": placement_mod.PLACEMENT_SCHEMA,
            "network_id": network_id,
            "epoch": 0,
            "written_by": sender,
            "credentials": [
                {
                    "key": STUB_PROVIDER,
                    "owner_device": owner or sender,
                    "provider": STUB_PROVIDER,
                    "kind": "oauth-rotating",
                    "doc_rev": 1,
                    "holders": [{"device": holder, "scope": "device"}],
                }
            ],
        },
    }


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("third_party_holder", "not_a_holder"),
        ("third_party_owner", "not_owner"),
        ("not_negotiated", "unsupported"),
    ],
)
def test_the_receiver_refuses_what_a_share_push_may_not_carry(
    drive_mesh: Any, case: str, expected: str
) -> None:
    from local_operator.network.credentials import share_push

    _start_b(drive_mesh)
    link = _Link(drive_mesh.owner, drive_mesh.network_id)
    holder = drive_mesh.borrower
    owner = ""
    if case == "third_party_holder":
        holder = "device-somebody-else"
    elif case == "third_party_owner":
        owner = "device-somebody-else"
    else:
        link.capabilities = ()
    detail = share_push.member_share_push(
        drive_mesh.b, link, _frame(drive_mesh.network_id, drive_mesh.owner, holder, owner)
    )
    assert detail.get("kind") == "error" and detail.get("code") == expected, detail
    assert not _b_holds(drive_mesh)


# ---------------------------------------------------------------------------
# Round 1 remediation: honest receipts, missed pushes, and a key another device owns
# ---------------------------------------------------------------------------


LONG_NAME = "studio-macbook-pro-m4-max-upstairs-office-2026"


@pytest.mark.parametrize(
    ("outcome", "first", "follow_up"),
    [
        ("pushed", "usable there: now", None),
        ("offline", "not yet — offline; it arrives when that device reconnects", None),
        ("failed", "not yet — that device did not answer", "lop network credentials"),
        ("refused:malformed_frame", "turned the update down", "lop network credentials"),
        ("refused:placement_conflict", "no — that device borrows this", "lop network credentials"),
        ("refused:not_a_holder", "no — the share was removed", "run the share again"),
        ("local_relay_outdated", "this device's network service is older", "lop network restart"),
        ("local_refused:busy", "this device's network service refused", "lop network status"),
        ("no_relay_answer", "this device's network service", "lop network status"),
        ("skipped:peer_lacks_share_push", "runs an older lop", "lop network update"),
        ("skipped:not_a_holder", "no — the share was removed", "run the share again"),
        ("skipped:no_dialer", "arrives on that device's next sync", None),
    ],
)
def test_every_delivery_outcome_reads_as_what_is_true(
    outcome: str, first: str, follow_up: str | None
) -> None:
    """Design review D1/D2/D3: the answer leads, no wire code in prose, fits 80 columns."""
    from local_operator.network import cli as network_cli

    lines = network_cli._delivery_lines(  # noqa: SLF001 — the receipt's one map
        {"delivered": outcome == "pushed", "outcome": outcome}, LONG_NAME
    )
    assert lines[0].startswith("usable there: ") and first in lines[0], lines
    assert not any(":" in line.split(" — ", 1)[-1] and "_" in line for line in lines), lines
    if follow_up is None:
        assert len(lines) == 1, lines
    else:
        assert any(follow_up in line for line in lines[1:]), lines
        assert all(line.startswith("  ") for line in lines[1:]), lines
    # Prose never repeats the name; only a command that needs it may.
    assert all(len(line) < 80 for line in lines if LONG_NAME not in line), lines


def _partition(mesh: Any) -> None:
    """Neither side can reach the other: both addresses dead, every live link dropped.

    BOTH directions: B's own cadences redial A, and a link B opened is one A's push
    would reuse — so a one-way partition delivered the push under load.

    "NO LIVE LINK" IS NOT "UNREACHABLE", and waiting on the former reddened this cell
    on loaded CI shards (jobs 129062385990, 129535685754: ``pushed != offline``). Each
    relay's ``mesh-redial`` loop may already be mid-dial when the rows are rewritten —
    B's start-up seed comes due one tick after ``_start_b``, A's redial of the pairing
    link it lost in setup ~1 s after that loss — having resolved the peer's REAL address
    before the rewrite. If that handshake is starved past the no-live-link check, it
    admits a fresh link after it, and the admission's ``_note_peer_endpoints`` writes the
    real address back onto both rows: the share then finds a live, genuine link to a
    running relay and delivering over it is correct product behaviour. So the cut is
    re-applied until both sides' OWN redial loops have tried the other and failed since
    the last loss. A redial loop dials serially on one thread, so a failure recorded
    there proves any dial it had in flight has finished; and a dial that finished by
    connecting left a live link, which the next pass closes (resetting that side's
    attempts).

    THE ACCEPTING SIDE IS A SECOND WRITER, and neither redial tracker sees it (review
    R1-1, reproduced by holding the acceptor's write until this wait first passed). An
    inbound connection completes on the acceptor's own ``mesh-handshake`` thread:
    ``send_welcome`` -> ``register_link`` -> ``_note_peer_endpoints`` /
    ``_note_observed_endpoint``, and that last step writes the dialer's REAL address
    onto the acceptor's row — after the dialer already holds its welcome, so after its
    redial loop may have moved on and recorded a failure. So the cut also waits until
    neither relay has an inbound handshake in flight. ``reload_blocker()`` reads that
    count from the real bracket: incremented on the accept thread BEFORE the
    handshake thread starts, decremented in ``_handshake_inbound``'s ``finally`` after
    the endpoint write (and after the pairing listener's ``contact_peers``, which runs
    on that same thread). ORDER MATTERS within a poll: attempts, then the in-flight
    count, then the cut. A dial the redial loop has finished either got its welcome
    (so its acceptor thread was already counted) or never will; a handshake counted
    as idle and finished before the cut is visible to the cut (a healed row is
    rewritten, a live link closed and the poll fails).

    No other dialer can hold a pre-cut address across the cut. Every other dial
    starts from ``_link_for``, which found the sync link live until the cut closed it,
    after the rows were already dead; ``contact_peers`` dials only from the listing
    surfaces (never invoked here) and the pairing listener, whose run is inside the
    counted handshake above. ``peer_status`` also calls ``RedialTracker.failed`` (via
    ``lop network peers``), but it dials from the rows too and is never invoked here,
    so every failure counted below is the redial loop's."""

    def _cut_and_confirmed_unreachable() -> bool:
        # Read BEFORE the cut (see the docstring's ordering): ``attempts`` counts
        # redial failures since this side's last loss (``connected`` clears it; of
        # its other caller, ``peer_status``, see the docstring) ...
        dialers_failed = (
            int(mesh.a.redial.snapshot(mesh.borrower).get("attempts", 0)) >= 1
            and int(mesh.b.redial.snapshot(mesh.owner).get("attempts", 0)) >= 1
        )
        # ... and no acceptor is still between ``send_welcome`` and its endpoint write.
        acceptors_idle = mesh.a.reload_blocker() is None and mesh.b.reload_blocker() is None
        # Idempotent cut, re-applied on every poll: a late admission heals the rows.
        for server, other in ((mesh.a, mesh.borrower), (mesh.b, mesh.owner)):
            with store.mutate(mesh.network_id, server.root) as copy:
                row = copy.member(other)
                assert row is not None
                if row.endpoints != ["127.0.0.1:9"]:
                    row.endpoints = ["127.0.0.1:9"]
                    store.save(copy, server.root)
        live = [x for x in [*mesh.a.links.values(), *mesh.b.links.values()] if x.alive]
        for link in live:
            link.close()
        if live:
            return False
        return dialers_failed and acceptors_idle

    assert net_fixtures.wait_for(_cut_and_confirmed_unreachable)


def _heal(mesh: Any) -> None:
    for server, other, endpoint in (
        (mesh.a, mesh.borrower, mesh.b_endpoint),
        (mesh.b, mesh.owner, mesh.a_endpoint),
    ):
        with store.mutate(mesh.network_id, server.root) as copy:
            row = copy.member(other)
            assert row is not None
            row.endpoints = [endpoint]
            store.save(copy, server.root)


def test_a_missed_push_is_resent_on_the_owners_first_sync_after_reconnect(
    drive_mesh: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """QA Q1: B's relay stayed up but was unreachable at share time, AND B was already
    parked by an earlier refusal (every fresh drive device is). Before the fix the
    share arrived only when the 30-min park expired (measured 1814-1843 s)."""
    from local_operator.network.credentials import share_push

    _start_b(drive_mesh)
    cred_sync.credentials_sync_step(drive_mesh.b, drive_mesh.owner)
    assert drive_mesh.owner in cred_sync._owner_refusals(drive_mesh.b)  # noqa: SLF001
    _partition(drive_mesh)

    payload = _share_json(drive_mesh, capsys)
    assert payload["delivery"] == {"delivered": False, "outcome": "offline"}, payload
    assert not _b_holds(drive_mesh)

    _heal(drive_mesh)
    # ONE owner sync after reconnect: the retry step the tick runs for this member.
    assert share_push.retry_step(drive_mesh.a, drive_mesh.borrower) == "pushed"
    assert _b_holds(drive_mesh)
    assert share_push.retry_step(drive_mesh.a, drive_mesh.borrower) is None, "not forgotten"


def _relink(mesh: Any) -> Any:
    """Drop every link between A and B, then let B dial A again."""
    for server in (mesh.a, mesh.b):
        for link in list(server.links.values()):
            link.close()
    assert net_fixtures.wait_for(
        lambda: not any(x.alive for x in [*mesh.a.links.values(), *mesh.b.links.values()])
    )
    link, why = mesh.b._ensure_link_with_reason(mesh.owner)  # noqa: SLF001
    assert link is not None, why
    return link


def test_an_owner_relay_restart_ends_the_borrowers_parked_refusal(
    drive_mesh: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """QA Q1b: the owner's relay restarted after a missed push and forgot its retry. B's
    first sync that sees the NEW relay instance pulls the share (was ~1814 s)."""
    from local_operator.network import relay as relay_mod
    from local_operator.network.credentials import share_push

    monkeypatch.setattr(
        share_push,
        "push_share",
        lambda *a, **k: {"delivered": False, "outcome": "skipped:test_disabled"},
    )
    _start_b(drive_mesh)
    cred_sync.credentials_sync_step(drive_mesh.b, drive_mesh.owner)
    assert drive_mesh.owner in cred_sync._owner_refusals(drive_mesh.b)  # noqa: SLF001
    _share_json(drive_mesh, capsys)
    cred_sync.credentials_sync_step(drive_mesh.b, drive_mesh.owner)
    assert not _b_holds(drive_mesh), "the park did not hold on an undisturbed owner"

    # A restarted relay process: a fresh instance id in its handshake, no retry memory.
    drive_mesh.a.instance_id = relay_mod.mint_instance_id()
    started = time.monotonic()
    link = _relink(drive_mesh)
    assert link.instance_id == drive_mesh.a.instance_id

    cred_sync.credentials_sync_step(drive_mesh.b, drive_mesh.owner)
    assert _b_holds(drive_mesh), "a new owner instance did not end the park"
    assert time.monotonic() - started < 60.0


def test_a_flapping_link_to_the_same_owner_does_not_multiply_refusals(
    drive_mesh: Any,
) -> None:
    """QA Q3: an UNSHARED B on a link dropped 15 times must not cost the owner one
    audited refusal per new link (measured 16 in 300 s); the park holds per instance."""
    _start_b(drive_mesh)
    before = _refusals_on_owner(drive_mesh)
    cred_sync.credentials_sync_step(drive_mesh.b, drive_mesh.owner)
    assert net_fixtures.wait_for(lambda: _refusals_on_owner(drive_mesh) == before + 1)
    for _ in range(15):
        _relink(drive_mesh)
        cred_sync.credentials_sync_step(drive_mesh.b, drive_mesh.owner)
    time.sleep(0.5)
    assert _refusals_on_owner(drive_mesh) - before <= 2, "a re-link re-asked a parked owner"
    assert not _b_holds(drive_mesh)


def _refusals_on_borrower(mesh: Any) -> int:
    mesh.b.audit.flush()
    return sum(
        1
        for row in mesh.b.audit.tail(2_000)
        if row.get("event") == "authorisation_refused" and row.get("actor") == mesh.owner
    )


def test_an_authoriser_refusal_on_the_holder_is_not_retried(
    drive_mesh: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """Review R2-M1: B's row for A lacks ``broker_credential`` (a non-admin lender). B's
    authoriser refuses the push without a code; that is an answer, never ``failed``, so
    the owner's tick must not re-send it (was one refusal row on B per tick for 24 h)."""
    from local_operator.network.credentials import share_push

    _start_b(drive_mesh)
    with store.mutate(drive_mesh.network_id, drive_mesh.b.root) as copy:
        row = copy.member(drive_mesh.owner)
        assert row is not None
        row.capabilities = [c for c in row.capabilities if c != "broker_credential"]
        store.save(copy, drive_mesh.b.root)
    _relink(drive_mesh)
    before = _refusals_on_borrower(drive_mesh)

    payload = _share_json(drive_mesh, capsys)
    assert payload["delivery"] == {
        "delivered": False,
        "outcome": "refused:not_authorised",
    }, payload
    for _ in range(5):
        assert share_push.retry_step(drive_mesh.a, drive_mesh.borrower) is None
    time.sleep(0.5)
    assert _refusals_on_borrower(drive_mesh) - before == 1


def test_a_push_that_loses_to_another_owners_row_is_not_reported_usable(
    drive_mesh: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """Review m2: B already borrows this key from device C. The merge keeps C's row, so
    the push must not ack as delivered, and the receipt must say why."""
    third = "d_third_owner_000000000000000000"
    _start_b(drive_mesh)
    placement_mod.merge_from_peer(
        drive_mesh.network_id,
        {
            "schema": placement_mod.PLACEMENT_SCHEMA,
            "network_id": drive_mesh.network_id,
            "epoch": 0,
            "written_by": third,
            "credentials": [
                {
                    "key": STUB_PROVIDER,
                    "owner_device": third,
                    "provider": STUB_PROVIDER,
                    "kind": "oauth-rotating",
                    "doc_rev": 1,
                    "holders": [{"device": drive_mesh.borrower, "scope": "device"}],
                }
            ],
        },
        from_device=third,
        self_device=drive_mesh.borrower,
        root=drive_mesh.b.root,
    )

    payload = _share_json(drive_mesh, capsys)
    assert payload["delivery"] == {
        "delivered": False,
        "outcome": "refused:placement_conflict",
        "other_owner": third,
    }, payload
    stored = placement_mod.PlacementDocument.load(
        drive_mesh.network_id, drive_mesh.b.root, self_device=drive_mesh.borrower
    )
    assert stored.owner_of(STUB_PROVIDER) == third
    capsys.readouterr()
    assert (
        _lop_network("credential", "share", STUB_PROVIDER, "--with", drive_mesh.b.identity.name)
        == 0
    )
    # The receipt names the device to revoke on (design round 2, D6).
    assert f"revoke it on {third}" in capsys.readouterr().out
