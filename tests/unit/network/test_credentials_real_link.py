"""Credential brokering END TO END over two real relays — the seam CI never drove.

WHY THIS FILE EXISTS (QA round 1, Q2). Every leg-2 frame the borrower's relay sent to
the owner carried no ``req``. ``PeerLink.request`` registers a waiter only for a frame
that has one, so the call returned ``None`` at once and the borrower reported "owner
offline" — while the owner really did refresh, audit and serve the grant into a reply
nobody was waiting for. CI was 24/24 green because no test drove ``serve_grant``,
``serve_report`` or ``pull_placement`` over a link: the transport tests built their own
frames WITH a ``req``, and the in-process tests called the owner's handler directly.

So nothing here builds a broker frame by hand except the one probe that must lack a
``req``. Every step is the product's own path:

* the SHARE and the REVOKE are ``lop network credential …`` through the real parser
  (so an argument the parser never defines fails here — Q1);
* the borrower asks ITS OWN relay over its loopback control socket (leg 1), exactly as
  a session's store does, and that relay asks the owner's relay over a real,
  mutually-authenticated TCP link (leg 2);
* the owner resolves against a real ``AuthStore`` whose refresh POSTs to a rotating
  loopback IdP that refuses a spent token, so the POST count is a fact.

ONE PROCESS, TWO ROOTS — and the one constraint that shapes the test. ``AuthStore``
derives its database from ``LOCAL_OPERATOR_CONFIG_DIR``, not from its argument, so the
ambient root is kept on the OWNER for the whole test and the borrower is driven only
through APIs that take its root explicitly (its relay, its client). The borrower never
constructs an ``AuthStore`` here; the brokered turn through a borrower's store is
covered separately, and QA drives the whole thing across real processes.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import pytest

from local_operator.network import relay, store
from local_operator.network.credentials import placement as placement_mod
from local_operator.network.credentials.client import MeshCredentialClient
from local_operator.network.credentials.types import BrokerError, Grant
from tests.unit.network.test_credentials_owner import (  # noqa: F401 — fixtures by import
    STUB_PROVIDER,
    RotatingIdP,
    idp,
    stub_provider,
)
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _pair,
    devices,
)

Devices = tuple[relay.RelayServer, relay.RelayServer, str, int]


def _parser() -> argparse.ArgumentParser:
    """The shipped ``lop network`` parser, built the way ``lop`` builds it."""
    from local_operator.network import cli as network_cli

    parser = argparse.ArgumentParser(prog="lop")
    network_cli.add_parser(parser.add_subparsers(dest="command"))
    return parser


def _lop_network(*argv: str) -> int:
    """Run one ``lop network …`` command exactly as typed: parse, then dispatch."""
    from local_operator.network import cli as network_cli

    return network_cli.main(_parser().parse_args(["network", *argv]))


def _wait_for(predicate: Any, *, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return bool(predicate())


@pytest.fixture()
def mesh(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Owner A (admin, relay running, an EXPIRED login) and borrower B (drive, relay running).

    The owner's login is expired on purpose, so the first grant must refresh: a test
    that served a live token would prove nothing about the one-POST property or about
    a refresh being spent on a reply nobody receives.
    """
    # ``getfixturevalue`` by name, as ``test_credentials_transport`` does: a parameter of
    # the same name would shadow the import that makes each fixture reachable.
    request.getfixturevalue("stub_provider")
    stub_idp: RotatingIdP = request.getfixturevalue("idp")
    pair_devices: Devices = request.getfixturevalue("devices")
    server_a, server_b, host, port = pair_devices
    record, _host, _port = _pair(pair_devices, monkeypatch, role="drive")
    # THE AMBIENT ROOT IS THE OWNER'S from here on: see the module docstring.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
    # B must be able to DIAL A: the pairing leaves A's row on B with no endpoint, and a
    # relay started later syncs only its own row.
    with store.mutate(record.network_id, server_b.root) as copy:
        owner_row = copy.member(server_a.identity.device_id)
        assert owner_row is not None
        owner_row.endpoints = [f"{host}:{port}"]
        store.save(copy, server_b.root)
    server_b.start()

    from local_operator.providers.auth_store import AuthStore

    auth = AuthStore(config_dir=server_a.root)
    auth.upsert_credential(
        STUB_PROVIDER,
        {
            "type": "oauth",
            "access": "-".join(("stale", "access")),
            "expires": int(time.time() * 1000) - 60_000,
            "refresh": stub_idp.current_refresh,
            "email": "owner@example.test",
        },
    )
    auth.close()
    try:
        yield type(
            "Mesh",
            (),
            {
                "a": server_a,
                "b": server_b,
                "network_id": record.network_id,
                "owner": server_a.identity.device_id,
                "borrower": server_b.identity.device_id,
                "idp": stub_idp,
            },
        )
    finally:
        server_b.stop()


def _share(mesh: Any) -> None:
    # No ``--network``: this device is in exactly one, which is how an operator types it.
    assert _lop_network("credential", "share", STUB_PROVIDER, "--with", mesh.b.identity.name) == 0


def _pull(mesh: Any) -> dict[str, Any]:
    """``lop network credentials`` on B, as far as its relay: the leg-1 placement op."""
    found = store.find_own_relay(mesh.b.root)
    assert found is not None, "the borrower's relay published no record"
    reply = relay.control_request(found, "credential_placement", timeout=30.0)
    assert isinstance(reply, dict), reply
    detail = reply.get("detail")
    return detail if isinstance(detail, dict) else reply


def _borrower_client(mesh: Any) -> MeshCredentialClient:
    client = MeshCredentialClient.for_this_device(mesh.b.root)
    assert client is not None, "the borrower cannot borrow: the share never reached it"
    return client


def _audit(server: relay.RelayServer, event: str) -> list[dict[str, Any]]:
    server.audit.flush()
    return [row for row in server.audit.tail(500) if row.get("event") == event]


def test_a_share_reaches_the_borrower_and_a_borrow_is_served_over_the_link(
    mesh: Any,
) -> None:
    """Q2's headline, inverted: pull learns the share, a grant arrives, ONE POST.

    Before the fix both halves failed on this exact path — the pull came back with
    ``changed: []`` and no file, and the grant came back ``owner_offline`` while the
    owner's IdP logged a POST.
    """
    _share(mesh)
    pulled = _pull(mesh)
    assert STUB_PROVIDER in pulled.get("changed", []), pulled
    document = placement_mod.PlacementDocument.load(
        mesh.network_id, mesh.b.root, self_device=mesh.borrower
    )
    assert document.owner_of(STUB_PROVIDER) == mesh.owner
    assert document.is_holder(STUB_PROVIDER, mesh.borrower)

    grant = _borrower_client(mesh).request_grant_sync(STUB_PROVIDER, session_id="sess-1")
    assert isinstance(grant, Grant), grant
    assert grant.credential_ref.owner_device == mesh.owner
    assert grant.access_token and grant.access_token != "-".join(("stale", "access"))
    assert len(mesh.idp.posts) == 1, "the owner did not refresh exactly once"
    # The owner served it and the borrower received it: nothing arrived as a stray.
    assert all(link.stray_replies == 0 for link in mesh.b.links.values())


def test_a_frame_that_cannot_be_answered_costs_the_owner_nothing(mesh: Any) -> None:
    """The ORDERING half of Q2: a request whose reply cannot be matched is refused FIRST.

    Sent with ``link.send`` because ``request`` is exactly the call that, for a frame
    with no ``req``, gives up without waiting. The owner answers with a refusal the
    borrower can only count as a stray — and it must not have spent a token POST, or
    written a grant into its audit, on the way.
    """
    _share(mesh)
    _pull(mesh)
    link, _reason = mesh.b._ensure_link_with_reason(mesh.owner)  # noqa: SLF001 — the dial seam
    assert link is not None
    strays = link.stray_replies
    sent = link.send(
        {
            "op": "net_broker",
            "kind": "grant",
            "key": STUB_PROVIDER,
            "provider": STUB_PROVIDER,
            "from_device": mesh.borrower,
            "for_session": "sess-noreq",
            "model_id": "stub-model",
        }
    )
    assert sent
    assert _wait_for(lambda: link.stray_replies > strays), "the owner never answered at all"
    assert mesh.idp.posts == [], "an unmatchable request spent the owner's refresh token"
    assert _audit(mesh.a, "credential.grant") == []


def test_a_failure_report_reaches_the_owners_broker(mesh: Any) -> None:
    """Q7: a real report carries the borrower's session and must still be HEARD.

    The report is sent with a session id, as ``rotate_sibling`` always sends it. Before
    the fix that id travelled as ``session_id``, which the owner's chokepoint reads as
    "act on a session I own" and refuses, so the owner's report arm never ran from a real
    turn. The owner's own ``credential.report`` audit record is the proof it ran.
    """
    _share(mesh)
    _pull(mesh)
    found = store.find_own_relay(mesh.b.root)
    assert found is not None
    reply = relay.control_request(
        found,
        "credential_report",
        timeout=30.0,
        credential_key=STUB_PROVIDER,
        kind="unavailable",
        session_id="sess-7",
        model_id="stub-model",
    )
    assert isinstance(reply, dict)
    detail = reply.get("detail")
    assert isinstance(detail, dict) and detail.get("action") == "noted", reply
    records = _audit(mesh.a, "credential.report")
    assert records, "the report never reached the owner's broker"
    assert records[-1]["detail"]["sub"] == mesh.borrower
    assert records[-1].get("session_id") == "sess-7"


def test_a_revoked_borrower_is_refused_by_name(
    mesh: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """Q4 on the real path: the refusal names the share, not the owner's build.

    ``credential revoke`` also takes ``broker_credential`` off B's member row, so the
    owner's TRANSPORT refuses before its broker can say ``not_a_holder`` — and that
    refusal used to reach B as "runs a build that cannot lend credentials", cached for
    five minutes as ``unsupported``.
    """
    _share(mesh)
    _pull(mesh)
    assert isinstance(
        _borrower_client(mesh).request_grant_sync(STUB_PROVIDER, session_id="sess-1"), Grant
    )
    capsys.readouterr()
    assert _lop_network("credential", "revoke", STUB_PROVIDER, "--from", mesh.b.identity.name) == 0
    # Q5: the receipt states the three latencies, including the one TTL does not bound.
    receipt = capsys.readouterr().out
    assert "new borrows by" in receipt and "refused now" in receipt, receipt
    assert "stays valid at the provider until the token expires" in receipt, receipt
    # B's own copy of the sharing list still names B (it has not pulled), so B really
    # asks: the refusal under test is the OWNER's answer, not a local short-circuit.
    refused = _borrower_client(mesh).request_grant_sync(STUB_PROVIDER, session_id="sess-2")
    assert isinstance(refused, BrokerError), refused
    assert refused.code == "not_a_holder", refused
    assert "does not share" in refused.message and "cannot lend" not in refused.message


def test_an_offline_owner_is_named_with_when_it_was_last_seen(mesh: Any) -> None:
    """Q6: "last seen never" for an owner that had served a grant seconds earlier.

    ``member.last_seen_at`` is stamped only by a rotation, so on a real row it is empty;
    the grant this device was just served is the sighting that counts.
    """
    _share(mesh)
    _pull(mesh)
    assert isinstance(
        _borrower_client(mesh).request_grant_sync(STUB_PROVIDER, session_id="sess-1"), Grant
    )
    mesh.a.stop()
    # WAIT OUT THE SHUTDOWN GAP rather than race it (review round 3, F2): ``stop``
    # shuts the owner's worker pool ~50 ms before it closes links, and this test is
    # about an owner that is GONE. The gap itself is its own test below.
    assert _wait_for(lambda: not any(link.alive for link in mesh.b.links.values()))
    offline = _borrower_client(mesh).request_grant_sync(STUB_PROVIDER, session_id="sess-2")
    assert isinstance(offline, BrokerError), offline
    assert offline.code == "owner_offline", offline
    assert "last seen just now" in offline.message, offline.message


def test_an_owner_that_is_shutting_down_reads_as_offline(mesh: Any) -> None:
    """F2 (review round 3): the owner mid-``stop`` answered ``internal`` in its own voice.

    ``RelayServer.stop`` shuts the slow-op pool FIRST and closes links ~50 ms later. A
    borrower whose link is still up in that gap gets "this device's relay is stopping",
    which was classified ``internal``, cached, and shown to the borrower as if its own
    relay were stopping. This holds the gap open deterministically by doing exactly
    ``stop``'s first step and nothing else.
    """
    _share(mesh)
    _pull(mesh)
    client = _borrower_client(mesh)
    assert isinstance(client.request_grant_sync(STUB_PROVIDER, session_id="sess-1"), Grant)
    pool, _slots = mesh.a._slow_executor()  # noqa: SLF001 — ``stop``'s own first step
    pool.shutdown(wait=False, cancel_futures=True)
    assert any(link.alive for link in mesh.b.links.values()), "the gap was not held open"
    stopping = _borrower_client(mesh).request_grant_sync(STUB_PROVIDER, session_id="sess-2")
    assert isinstance(stopping, BrokerError), stopping
    assert stopping.code == "owner_offline", stopping
    assert "relay is stopping" not in stopping.message, stopping.message


def test_a_static_key_revoke_says_the_copy_never_expires(
    mesh: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """F3 (review round 3): the receipt promised "until the token expires" for a key
    that never expires. A static API key shared and then revoked must say the only true
    thing: a copy lives until the key is rotated at the provider — in the lines AND in
    the ``--json`` payload an incident script would read.
    """
    import json

    from local_operator.providers.auth_store import AuthStore

    auth = AuthStore(config_dir=mesh.a.root)
    auth.upsert_credential(
        "deepseek", {"type": "api_key", "key": "-".join(("static", "fixture", "value"))}
    )
    auth.close()
    peer = mesh.b.identity.name
    assert _lop_network("credential", "share", "deepseek", "--with", peer) == 0
    capsys.readouterr()
    assert _lop_network("credential", "revoke", "deepseek", "--from", peer) == 0
    receipt = capsys.readouterr().out
    assert "never expires" in receipt and "rotate the 'deepseek' key" in receipt, receipt
    assert "until the token expires" not in receipt, receipt

    assert _lop_network("credential", "share", "deepseek", "--with", peer) == 0
    capsys.readouterr()
    assert _lop_network("credential", "revoke", "deepseek", "--from", peer, "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert "never expires" in payload["revocation"]["copied_bearer"], payload


_MISSING = object()


def test_a_report_with_a_garbled_retry_value_is_answered_not_crashed(mesh: Any) -> None:
    """QA round 2: a non-numeric ``retry_after_ms`` crashed the BORROWER's leg-1 handler.

    ``_LocalOps.report`` read it with a bare ``int(...)``: ``"abc"`` raised ``ValueError``
    and the control reply came back ``null`` — the same hole m1 closed on the owner's
    side, in the other direction. Each shape a peer or a skewed build could send —
    non-numeric, negative, absurdly large, ``NaN``, a container, missing — must reach the
    owner and come back with a DEFINED answer, over the real socket and the real link.
    """
    _share(mesh)
    _pull(mesh)
    found = store.find_own_relay(mesh.b.root)
    assert found is not None
    for value in ("abc", -5, 10**18, "nan", [1], _MISSING):
        extra = {} if value is _MISSING else {"retry_after_ms": value}
        reply = relay.control_request(
            found,
            "credential_report",
            timeout=30.0,
            credential_key=STUB_PROVIDER,
            kind="unavailable",
            session_id="sess-garbled",
            **extra,
        )
        assert isinstance(reply, dict), (value, reply)
        detail = reply.get("detail")
        assert isinstance(detail, dict) and detail.get("action") == "noted", (value, reply)
