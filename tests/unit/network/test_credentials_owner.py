"""The broker: authorisation, coalescing, and the safety of a peer's report.

WHAT MAKES THIS EVIDENCE. Every test here drives the REAL owner-side objects: a real
``AuthStore`` over its own SQLite file, the real ``MeshCredentialBroker`` on its own
long-lived event loop, and a real loopback HTTP token endpoint that ROTATES the
refresh token on every exchange and refuses a token it has already spent — so a
second POST is not counted, it FAILS. Nothing is stubbed except the peer link, and
that is a two-attribute object (a link is only ever read for its device id here).

THE FOUR PROPERTIES, each from the design and the build plan:

* exactly ONE token POST when several borrowers ask at once (§3.4 — the whole reason
  the broker owns one event loop rather than calling ``asyncio.run`` per request);
* a borrower's refresh attempt costs ZERO POSTs when the owner is away (§2.4), proved
  in a SEPARATE PROCESS so "off the owner" is a fact rather than a same-process
  assumption;
* a peer's 401 cannot disable or rotate the owner's row (finding 8, cut line unsafe
  item 2 — the failure this requirement exists to prevent);
* the refusals that are decided before any work: not a holder, device-bound, forced
  refresh from a non-admin.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from local_operator.network.credentials import owner as owner_mod
from local_operator.network.credentials import placement as placement_mod
from local_operator.network.credentials import store as mesh_store
from local_operator.network.credentials.types import (
    PEER_BROKER_OP,
    is_mcp_key,
    synthetic_credential_id,
)
from local_operator.providers import registry

#: A provider id that exists only in this module. It is registered by inserting into
#: the registry's id index (there is no public registration seam, and a real provider
#: would drag its own token URL and CLI shape in with it).
STUB_PROVIDER = "meshtest"

OWNER_DEVICE = "d_00000000000000000000000000000001"
BORROWER_DEVICE = "d_00000000000000000000000000000002"
BORROWER_TWO = "d_00000000000000000000000000000003"


# ---------------------------------------------------------------------------
# A rotating stub identity provider, on a real loopback socket
# ---------------------------------------------------------------------------


class RotatingIdP:
    """A token endpoint that rotates on every exchange and refuses a spent token.

    THE ROTATION IS THE INSTRUMENT, not decoration. A stub that returned the same
    refresh token would let two POSTs both succeed, so a test could not tell "one
    exchange" from "two exchanges that happened to agree" — which is exactly the
    distinction the whole slice is about. Here the second POST of a spent refresh
    token gets ``invalid_grant`` and the borrow FAILS, so the assertion
    ``posts == 1`` is a claim about the code and not about the stub's tolerance.
    """

    def __init__(self) -> None:
        self.posts: list[str] = []  # the refresh token each POST presented
        self.access_seq = 0
        self.current_refresh = "refresh-0"
        self.invalid = False
        idp = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            # Signature matches ``BaseHTTPRequestHandler``'s own (a keyword ``format``
            # parameter), because a checker reads an override's shape as a contract and
            # ``*args`` is not that shape. Bodies are suppressed: one line per request
            # would bury the pytest output this test is read for.
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def do_POST(self) -> None:  # noqa: N802 — http.server's spelling
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length).decode("utf-8")
                presented = urllib.parse.parse_qs(body).get("refresh_token", [""])[0]
                idp.posts.append(presented)
                if idp.invalid:
                    payload = {"error": "invalid_grant"}
                    self._reply(400, payload)
                    return
                if presented != idp.current_refresh:
                    # A SPENT TOKEN. This is what a real reuse-detecting IdP does, and
                    # it is why two POSTs of one rotating token cannot both work.
                    self._reply(400, {"error": "invalid_grant"})
                    return
                idp.access_seq += 1
                idp.current_refresh = f"refresh-{idp.access_seq}"
                self._reply(
                    200,
                    {
                        "access_token": f"access-{idp.access_seq}",
                        "refresh_token": idp.current_refresh,
                        "expires_in": 3600,
                    },
                )

            def _reply(self, status: int, payload: dict[str, Any]) -> None:
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/token"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)


@pytest.fixture()
def idp() -> Any:
    stub = RotatingIdP()
    try:
        yield stub
    finally:
        stub.close()


@pytest.fixture()
def stub_provider(monkeypatch: pytest.MonkeyPatch, idp: RotatingIdP) -> str:
    """Register the test-only provider, with a refresh fn that really POSTs.

    ``_BY_ID`` is the registry's own index and the only thing ``get_provider_definition``
    reads, so this is what makes the provider exist for the store's refresh lookup.
    ``monkeypatch.setitem`` restores it, so no other test can see this provider.
    """
    posted: list[str] = []

    async def refresh(creds: dict[str, Any], **_: Any) -> dict[str, Any]:
        import httpx

        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                idp.url,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": str(creds.get("refresh") or ""),
                },
            )
        posted.append(str(creds.get("refresh") or ""))
        if response.status_code != 200:
            from local_operator.providers.auth_store import AuthStoreError

            raise AuthStoreError(f"stub IdP refused the exchange: {response.status_code}")
        token = response.json()
        merged = dict(creds)
        merged["access"] = token["access_token"]
        merged["refresh"] = token["refresh_token"]
        merged["expires"] = int(time.time() * 1000) + int(token["expires_in"]) * 1000
        return merged

    definition = registry.ProviderDefinition(
        id=STUB_PROVIDER,
        name="Mesh stub",
        refresh_token=refresh,
        get_api_key=lambda creds: str(creds.get("access") or ""),
        store_credentials_as=STUB_PROVIDER,
    )
    monkeypatch.setitem(registry._BY_ID, STUB_PROVIDER, definition)  # noqa: SLF001 — no seam
    return STUB_PROVIDER


class _Link:
    """The only part of a peer link this module reads: who is on the far end."""

    def __init__(self, device_id: str) -> None:
        self.device_id = device_id
        self.network_id = "n_owner"
        self.epoch = 1


@pytest.fixture()
def owner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_provider: str) -> Any:
    """An owner device: its own config root, its own ``auth.db``, its own broker.

    ``LOCAL_OPERATOR_CONFIG_DIR`` is pointed at the root because ``AuthStore``'s
    database path derives from ``paths.config_dir()`` rather than from the
    ``config_dir`` argument (which the store passes through to the env tier only) —
    so a test that set only the argument would be reading the operator's REAL
    ``auth.db``. That is the same hazard ``AGENTS.md`` isolates whole runs against.
    """
    root = tmp_path / "owner"
    root.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    from local_operator.providers.auth_store import AuthStore

    auth = AuthStore(config_dir=root)
    row = auth.upsert_credential(
        STUB_PROVIDER,
        {
            "access": "access-0",
            # ALREADY EXPIRED, so the first borrow must refresh. Without this the
            # test would prove that a live cached token is served once, which is a
            # different (and much weaker) claim.
            "expires": int(time.time() * 1000) - 60_000,
            "refresh": "refresh-0",
            "email": "owner@example.test",
        },
    )
    document = placement_mod.PlacementDocument("n_owner", root=root, written_by=OWNER_DEVICE)
    document.declare(
        STUB_PROVIDER,
        owner_device=OWNER_DEVICE,
        owner_device_name="owner-laptop",
        provider=STUB_PROVIDER,
        identity_label="owner@example.test",
        by=OWNER_DEVICE,
    )
    document.grant(STUB_PROVIDER, BORROWER_DEVICE, scope="session", by=OWNER_DEVICE)
    document.grant(STUB_PROVIDER, BORROWER_TWO, scope="session", by=OWNER_DEVICE)
    document.save()
    # A REAL audit log, because the `credential.grant` record is part of the
    # deliverable and its fields are whitelisted: a key missing from
    # ``audit.DETAIL_KEYS`` is dropped SILENTLY, so an assertion against a fake
    # recorder would pass while the file on disk carried nothing.
    from local_operator.network.audit import AuditLog

    audit = AuditLog(root=root)
    broker = owner_mod.MeshCredentialBroker(
        root=root,
        self_device=OWNER_DEVICE,
        self_device_name="owner-laptop",
        network_id="n_owner",
        auth_store=auth,
        audit=audit,
    )
    try:
        yield type(
            "Owner",
            (),
            {
                "root": root,
                "auth": auth,
                "row": row,
                "broker": broker,
                "document": document,
                "audit": audit,
            },
        )
    finally:
        broker.close()
        audit.close()
        auth.close()


def _grant_frame(
    device: str,
    *,
    session: str = "sess-1",
    force: bool = False,
    key: str = STUB_PROVIDER,
    provider: str = "",
) -> dict[str, Any]:
    return {
        "op": PEER_BROKER_OP,
        "kind": "grant",
        "key": key,
        "provider": provider or key,
        "from_device": device,
        "from_device_name": device[-4:],
        "for_session": session,
        "model_id": "stub-model",
        "force_refresh": force,
    }


def _detail(reply: dict[str, Any]) -> dict[str, Any]:
    """The ``detail`` of a handler's reply frame, asserting the frame's shape.

    Tests drive ``on_broker`` — the registered handler — rather than ``grant``/
    ``report`` directly, so the ``{"op": "ack", "req": …, "detail": …}`` envelope the
    relay actually sends is part of what is under test: a refusal that arrived as a
    bare dict would satisfy a protocol-level check and fail a real peer.
    """
    assert reply.get("op") == "ack", reply
    detail = reply.get("detail")
    assert isinstance(detail, dict), reply
    return detail


def _ask_grant(owner: Any, device: str, **kwargs: Any) -> dict[str, Any]:
    return _detail(owner.broker.on_broker(_Link(device), _grant_frame(device, **kwargs)))


def _lease_rows(auth: Any) -> int:
    return int(
        auth._conn.execute("SELECT COUNT(*) FROM auth_credential_refresh_leases").fetchone()[0]
    )


# ---------------------------------------------------------------------------
# Coalescing: several borrowers, one POST
# ---------------------------------------------------------------------------


def test_concurrent_borrowers_cost_exactly_one_token_post(owner: Any, idp: RotatingIdP) -> None:
    """Four simultaneous borrows, two sessions and two devices ⇒ ONE exchange.

    THE HEADLINE PROPERTY. The one POST is what makes brokering safe at all: the
    owner's rotating refresh token is presented once, so no device can be logged out
    by a sibling's refresh — the PR-24 failure class, moved off-host.

    Both coalescing mechanisms are exercised at once and both must hold: the
    ``(key, for_session, force)`` map joins the callers that arrive together, and the
    store's own per-credential lock is what protects the two that asked under
    DIFFERENT session ids (they cannot join, so they must serialise inside the store
    and the second must find the token already fresh).
    """
    owner.broker._loop.loop()  # start the loop once, so the first call is not the race
    replies: list[dict[str, Any]] = []
    errors: list[BaseException] = []
    gate = threading.Barrier(4)

    def ask(device: str, session: str) -> None:
        try:
            gate.wait(timeout=10)
            replies.append(_ask_grant(owner, device, session=session))
        except BaseException as exc:  # noqa: BLE001 — reported by the assertion below
            errors.append(exc)

    threads = [
        threading.Thread(target=ask, args=(BORROWER_DEVICE, "sess-1")),
        threading.Thread(target=ask, args=(BORROWER_DEVICE, "sess-1")),
        threading.Thread(target=ask, args=(BORROWER_TWO, "sess-2")),
        threading.Thread(target=ask, args=(BORROWER_TWO, "sess-2")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, errors
    assert len(replies) == 4
    assert idp.posts == ["refresh-0"], f"expected one exchange, saw {idp.posts!r}"
    for detail in replies:
        assert detail["kind"] == "grant", detail
        assert detail["access_token"] == "access-1"
    # ONE EXCHANGE, ONE TOKEN, EVERYONE SERVED FROM IT. The POST count is the proof
    # (`["refresh-0"]` above, and the stub REFUSES a spent token, so a second exchange
    # would have failed rather than gone uncounted). ``refreshed`` is an observation —
    # "the owner's row was rewritten while this request was being served" — and is
    # deliberately NOT asserted as a count: a joiner whose resolve overlapped the
    # exchange correctly reports True too, and whether any given asker overlapped is
    # timing, not behaviour. What must hold is that all four hold the SAME freshly
    # minted bearer.
    assert any(detail["refreshed"] for detail in replies), replies
    assert {detail["access_token"] for detail in replies} == {"access-1"}
    # The owner's own row is the NEW token, so the next borrow serves it without a POST.
    assert owner.auth.get_credential(owner.row.id).data["refresh"] == "refresh-1"
    assert _lease_rows(owner.auth) == 0, "a refresh lease outlived its exchange"


def test_a_second_borrow_seconds_later_serves_the_live_token(owner: Any, idp: RotatingIdP) -> None:
    """A token with life left is SERVED, not refreshed (§3.5). No gratuitous POSTs."""
    first = _ask_grant(owner, BORROWER_DEVICE)
    second = _ask_grant(owner, BORROWER_DEVICE)
    assert first["access_token"] == second["access_token"] == "access-1"
    assert second["refreshed"] is False
    assert idp.posts == ["refresh-0"]


def test_the_grant_never_outlives_the_token_or_the_ttl(owner: Any, idp: RotatingIdP) -> None:
    """``min(token_expiry, now + grant_ttl_s)`` — §3.3's narrowing rule, on the wire."""
    detail = _ask_grant(owner, BORROWER_DEVICE)
    now_ms = time.time() * 1000
    assert detail["grant_expires_at_ms"] <= int(now_ms) + int(900 * 1000) + 2_000
    assert detail["grant_expires_at_ms"] <= detail["token_expires_at_ms"]
    assert detail["scope"] == {"kind": "session", "session_id": "sess-1"}
    assert detail["credential_ref"]["owner_device"] == OWNER_DEVICE
    assert detail["credential_ref"]["provider"] == STUB_PROVIDER


# ---------------------------------------------------------------------------
# The refusals decided before any work
# ---------------------------------------------------------------------------


def test_a_device_that_is_not_a_holder_is_refused_without_a_post(
    owner: Any, idp: RotatingIdP
) -> None:
    stranger = "d_00000000000000000000000000000009"
    detail = _ask_grant(owner, stranger)
    assert detail["kind"] == "error"
    assert detail["code"] == "not_a_holder"
    assert idp.posts == []


def test_a_revoked_holder_stops_being_served(owner: Any, idp: RotatingIdP) -> None:
    with placement_mod.mutate("n_owner", owner.root, self_device=OWNER_DEVICE) as document:
        document.revoke(STUB_PROVIDER, BORROWER_TWO, by=OWNER_DEVICE)
    detail = _ask_grant(owner, BORROWER_TWO)
    assert detail["code"] == "not_a_holder"
    assert idp.posts == []


def test_a_device_bound_provider_is_refused_by_name(
    owner: Any, idp: RotatingIdP, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kimi is refused before authorisation is even consulted (finding 7).

    Refused by NAME, not by a rule that could miss: a borrower replaying a device-bound
    grant presents the owner's token with ITS OWN fingerprint, which is a different
    device using that token — the case the design itself calls never.
    """
    detail = _ask_grant(owner, BORROWER_DEVICE, key="kimi", provider="kimi")
    assert detail["code"] == "device_bound"
    assert "bound to the device that made them" in detail["message"]
    assert idp.posts == []


def test_a_forced_refresh_from_a_non_admin_is_refused(owner: Any, idp: RotatingIdP) -> None:
    """Cut line unsafe item 3: a peer cannot make the owner spend a refresh on demand.

    A forced refresh is the cheap way to provoke an IdP's reuse detection, which
    revokes the whole token family. The owner's member table names this device a
    ``drive`` member, so the answer is no.
    """
    detail = _ask_grant(owner, BORROWER_DEVICE, force=True)
    assert detail["kind"] == "error"
    assert detail["code"] == "not_authorised"
    assert idp.posts == []


def test_an_ask_for_a_key_this_device_does_not_own_is_refused(owner: Any, idp: RotatingIdP) -> None:
    detail = _ask_grant(owner, BORROWER_DEVICE, key="deepseek", provider="deepseek")
    assert detail["code"] == "not_owner"
    assert idp.posts == []


# ---------------------------------------------------------------------------
# A peer's report can never disable or rotate the owner's login
# ---------------------------------------------------------------------------


def _report(owner: Any, device: str, failure: str, **extra: Any) -> dict[str, Any]:
    frame = {
        "op": PEER_BROKER_OP,
        "kind": "report",
        "key": STUB_PROVIDER,
        "provider": STUB_PROVIDER,
        "from_device": device,
        "from_device_name": device[-4:],
        "failure": failure,
        **extra,
    }
    return _detail(owner.broker.on_broker(_Link(device), frame))


def test_the_report_fixture_can_actually_see_a_rotate_sibling_regression(
    owner: Any, idp: RotatingIdP
) -> None:
    """CONTROL for the test below: this fixture state CAN be disabled.

    The reviewer's P7 was that the old assertion (``disabled_cause == [None]``) passed
    even with ``rotate_sibling`` swapped in for the report arm, because
    ``rotate_sibling`` found no failing row: there was no sticky pointer for the
    borrower's session and no ``api_key`` to match. That made the test unable to fail
    for the regression it is named after. So the row here is made FINDABLE the way the
    failover driver leaves it — a pinned session and the bearer that actually failed —
    and then the exact call the mutation would make really does disable it.
    """
    from local_operator.providers.failover import ProviderError

    grant = _ask_grant(owner, BORROWER_DEVICE)
    bearer = str(grant["access_token"])
    owner.auth.pin_session_credential(STUB_PROVIDER, "sess-1", owner.row.id)
    assert owner.auth.session_credential_id(STUB_PROVIDER, "sess-1") == owner.row.id
    assert owner.auth.credential_id_for_key(STUB_PROVIDER, bearer) == owner.row.id, (
        "the row is not reachable from the bearer the borrower used, so a "
        "rotate_sibling regression would be invisible to this fixture"
    )
    owner.auth.rotate_sibling(STUB_PROVIDER, "sess-1", ProviderError(401, "invalid_grant"), bearer)
    rows = owner.auth.list_credentials(STUB_PROVIDER, include_disabled=True)
    assert [row.disabled_cause for row in rows] == ["invalidated-token"], rows


def test_a_peer_report_cannot_disable_or_delete_the_owners_login(
    owner: Any, idp: RotatingIdP
) -> None:
    """THE REQUIREMENT'S NAMED FAILURE, on a fixture that CAN be disabled.

    A 401, an ``invalid_grant`` and a 429 from a borrower, all against the owner's
    live row. Afterwards every row this device holds is still ENABLED — no
    ``disabled_cause``, nothing deleted — because acting on a peer's observation is how
    one bad 401 on one peer logs the operator out of every device.

    THE PRECONDITION IS THE DISCRIMINATOR (review round 1, F6): the session is pinned
    and the served bearer is a key that matches the owner's row, which is the state the
    control above shows a swapped-in ``rotate_sibling`` WOULD disable. Without it this
    test passed for a mutation it exists to catch.
    """
    grant = _ask_grant(owner, BORROWER_DEVICE)
    owner.auth.pin_session_credential(STUB_PROVIDER, "sess-1", owner.row.id)
    assert owner.auth.session_credential_id(STUB_PROVIDER, "sess-1") == owner.row.id
    assert (
        owner.auth.credential_id_for_key(STUB_PROVIDER, str(grant["access_token"])) == owner.row.id
    ), "the fixture cannot see a rotate_sibling regression; see the control above"

    _report(owner, BORROWER_DEVICE, "invalid")
    _report(owner, BORROWER_DEVICE, "invalid")
    _report(owner, BORROWER_DEVICE, "quota", model_id="stub-model", retry_after_ms=1_000)
    rows = owner.auth.list_credentials(STUB_PROVIDER, include_disabled=True)
    assert rows, "the credential disappeared"
    assert [row.disabled_cause for row in rows] == [None]
    assert owner.auth.get_credential(owner.row.id) is not None


def test_the_owner_side_refresh_a_report_can_provoke_is_rate_limited(
    owner: Any, idp: RotatingIdP
) -> None:
    """At most ONE coalesced owner refresh per credential per report window (§2.3).

    The refresh itself is the owner's own call through ``_ensure_oauth_fresh``, and the
    bound is what stops a borrower that keeps failing from driving the owner's token
    endpoint. The count is the POST count: the stub refuses a spent token, so a second
    exchange in the window would be visible as a refusal rather than as a number.
    """
    first = _report(owner, BORROWER_DEVICE, "invalid")
    posts_after_first = len(idp.posts)
    second = _report(owner, BORROWER_DEVICE, "invalid")
    assert first["action"] == "refreshed"
    assert second["action"] == "coalesced"
    assert len(idp.posts) == posts_after_first, "a second report spent another exchange"
    assert posts_after_first == 1


def test_a_report_from_a_device_that_is_not_a_holder_is_refused(
    owner: Any, idp: RotatingIdP
) -> None:
    stranger = "d_00000000000000000000000000000009"
    detail = _report(owner, stranger, "invalid")
    assert detail["code"] == "not_a_holder"


# ---------------------------------------------------------------------------
# Zero POSTs from a borrower whose owner is away, proved in another process
# ---------------------------------------------------------------------------

_BORROWER_CHILD = '''
"""Borrow with the owner away, and say what it cost. Run as a child process."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])
owner_device = sys.argv[2]
self_device = sys.argv[3]
# ``relay`` runs the leg-1 half against a REAL relay on this root too: the
# owner-unreachable path is a different code path from "no relay at all", and the
# reviewer's F7 was that only the latter was covered.
mode = sys.argv[4] if len(sys.argv) > 4 else "norelay"

from local_operator.network.credentials import placement as placement_mod
from local_operator.network.credentials import store as mesh_store
from local_operator.network.identity import identity_dir
from local_operator.providers import registry

# A refresh function that would record a call. The assertion is that it is NEVER
# called: with no refresh token, a brokered credential must have nothing to refresh
# with, so a fallback to "refresh it here" would show up as a non-zero count.
calls = []


async def refresh(creds, **_):
    calls.append(dict(creds))
    raise AssertionError("the borrower must never refresh")


definition = registry.ProviderDefinition(
    id="meshtest",
    name="Mesh stub",
    refresh_token=refresh,
    get_api_key=lambda creds: str(creds.get("access") or ""),
    store_credentials_as="meshtest",
)
registry._BY_ID["meshtest"] = definition

document = placement_mod.PlacementDocument("n_owner", root=root, written_by=owner_device)
document.declare(
    "meshtest",
    owner_device=owner_device,
    owner_device_name="owner-laptop",
    provider="meshtest",
    by=owner_device,
)
document.grant("meshtest", self_device, scope="session", by=owner_device)
document.save()

store = mesh_store.build_auth_store(root)
kind = type(store).__name__
relay_code = "no_relay"
if mode == "relay":
    # A REAL relay on this root, in its own process, with a network record that lists
    # the owner at an address nothing is listening on: the borrower's own relay is up
    # and cannot reach the owner, which is the case a "no relay" test cannot see.
    import socket
    from secrets import token_bytes

    from local_operator.network import relay as relay_mod
    from local_operator.network import store as net_store
    from local_operator.network import wire
    from local_operator.network.identity import load as load_identity
    from local_operator.network.types import NetworkRecord, SecretState

    dead = socket.socket()
    dead.bind(("127.0.0.1", 0))
    dead_port = dead.getsockname()[1]
    dead.close()
    identity = load_identity(root)
    record = NetworkRecord(
        network_id="n_owner",
        name="relay-net",
        created_by=self_device,
        self_device_id=self_device,
        self_role="drive",
        self_capabilities=["drive"],
    )
    for device_id, public_key, name in (
        (self_device, identity.public_key, identity.name),
        (owner_device, "a" * 43, "owner-laptop"),
    ):
        relay_mod.admit(
            record,
            device_id=device_id,
            public_key=public_key,
            name=name,
            role="admin" if device_id == owner_device else "drive",
            capabilities=["admin"] if device_id == owner_device else ["drive"],
            added_by=self_device,
            root=root,
            persist=False,
            endpoints=[f"127.0.0.1:{dead_port}"] if device_id == owner_device else [],
        )
    net_store.save(record, root)
    net_store.save_secrets(
        SecretState(network_id="n_owner", epoch=1, secret=wire.b64u(token_bytes(32))), root
    )
    server = relay_mod.RelayServer(
        root=root, settings=relay_mod.NetworkSettings(port=0, listen_address="127.0.0.1")
    )
    server.start()
    try:
        own = net_store.find_own_relay(root)
        assert own is not None, "the relay published no record to dial"
        # THE LEG-1 OP THE RUNTIME USES, over the real control socket.
        reply = relay_mod.control_request(
            own,
            "credential_grant",
            timeout=30.0,
            credential_key="meshtest",
            provider="meshtest",
            session_id="sess-1",
            model_id="",
        )
        detail = (reply or {}).get("detail") or {}
        relay_code = str(detail.get("code") or detail.get("kind") or "no_detail")
    finally:
        server.stop()

key = asyncio.run(mesh_store.build_auth_store(root).get_api_key("meshtest", "sess-1"))
print(json.dumps({"kind": kind, "key": key, "refresh_calls": len(calls), "relay_code": relay_code}))
'''


@pytest.mark.parametrize("mode", ["norelay", "relay"])
def test_a_borrower_with_the_owner_away_makes_zero_token_posts(
    mode: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_provider: str,
    idp: RotatingIdP,
) -> None:
    """§2.4 in a SEPARATE PROCESS, in both shapes of "the owner is away".

    Two things this proves that a same-process test could not. First, the borrower
    really is another process with its own config root, so "off the owner" is a fact
    about the runtime rather than an assumption about which object was called.
    Second, the refresh function it WOULD have to call to fall back to a local refresh
    is installed in that child and counts — and it stays at zero, which is the
    structural claim ("the borrower holds no refresh token") measured rather than
    asserted.

    BOTH MODES, because one is not the other (review round 1, F7). ``norelay`` is
    ``find_own_relay -> None``; ``relay`` starts a REAL relay on the child's own root,
    with the owner listed at an address nothing listens on, so the request travels the
    child's whole leg-1 path and comes back ``owner_offline`` — a path on which the
    first version raised ``TypeError`` out of the dial seam and answered ``internal``.

    THE CHILD'S REFRESH COUNT IS THE MEASUREMENT, and it is the only one kept: the
    parent-side ``idp.posts`` check the first version carried could never fail, because
    the child is a separate process with no knowledge of this IdP's address.
    """
    root = tmp_path / ("borrower-relay" if mode == "relay" else "borrower")
    root.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    from local_operator.network.identity import mint

    identity = mint(root, name="borrower-laptop")
    script = tmp_path / f"borrow_child_{mode}.py"
    script.write_text(_BORROWER_CHILD, encoding="utf-8")

    # EVERY INHERITED CMUX_* IS UNSET: an inherited workspace id once let a headless
    # test rename the operator's real workspaces.
    env = {k: v for k, v in os.environ.items() if not k.startswith("CMUX_")}
    env["LOCAL_OPERATOR_CONFIG_DIR"] = str(root)
    completed = subprocess.run(
        [sys.executable, str(script), str(root), OWNER_DEVICE, identity.device_id, mode],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["kind"] == "MeshAwareAuthStore", result
    assert result["key"] is None, "a credential appeared with no owner to lend one"
    assert result["refresh_calls"] == 0, "the borrower tried to refresh locally"
    if mode == "relay":
        # The refusals here are the same ones the operator reads: the borrower's own
        # relay was up and could not reach the owner, so the answer is the offline
        # sentence — never a silent local refresh.
        assert result["relay_code"] == "owner_offline", result
    else:
        assert result["relay_code"] == "no_relay", result
    # The owner's IdP was never part of the child's reach, so it recorded nothing.
    assert idp.posts == []


# ---------------------------------------------------------------------------
# The store's own surface: the synthetic id, and local-first
# ---------------------------------------------------------------------------


def test_the_wrapper_covers_the_methods_the_session_path_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_provider: str
) -> None:
    """The ~20-method surface (finding 5), checked by NAME rather than by hand.

    The design's three protocols were too narrow for the real call sites, so this
    asserts the exact set ``model/configure.py`` reaches for — a wrapper missing one
    of these raises ``AttributeError`` on a live turn, and the ``__getattr__``
    pass-through would hide that until then.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    from local_operator.providers.auth_store import AuthStore

    wrapper = mesh_store.MeshAwareAuthStore(
        AuthStore(config_dir=tmp_path), mesh=None, config_dir=tmp_path
    )
    try:
        needed = (
            "get_api_key",
            "get_oauth_access",
            "rotate_sibling",
            "list_credentials",
            "get_credential",
            "block_credential",
            "is_blocked_for_model",
            "is_blocked",
            "session_credential_id",
            "pin_session_credential",
            "release_session_credential",
            "list_oauth_accesses",
            "ensure_oauth_fresh",
            "ensure_oauth_fresh_or_raise",
            "deprioritize_credential",
            "clear_blocks_for_model",
            "upsert_credential",
            "delete_credential",
            "grant_is_dead",
            "send_unconfirmed",
            "_set_sticky",
            "db_path",
            "close",
        )
        missing = [name for name in needed if getattr(wrapper, name, None) is None]
        assert not missing, missing
    finally:
        wrapper.close()


def test_a_brokered_credential_gets_a_negative_synthetic_id() -> None:
    """The id no real row can have (finding 5's consequence), checked directly.

    SQLite assigns ``INTEGER PRIMARY KEY`` from 1 upward, so every negative integer is
    unreachable — which is what makes it safe to key blocks, stickiness and the
    failover driver's bookkeeping on a borrowed credential without any risk of
    colliding with a local login.
    """
    first = synthetic_credential_id("openai", OWNER_DEVICE)
    assert first < 0
    assert first == synthetic_credential_id("openai", OWNER_DEVICE), "not deterministic"
    assert first != synthetic_credential_id("openai", BORROWER_DEVICE)
    assert first != synthetic_credential_id("deepseek", OWNER_DEVICE)
    assert not is_mcp_key("openai")
    assert is_mcp_key("mcp:https://example.test/mcp")


def test_a_peers_request_does_not_move_the_owners_own_routing(owner: Any, idp: RotatingIdP) -> None:
    """``read_only=True`` on the owner (§2.1), measured rather than asserted.

    The whole design rests on the borrower holding a DELEGATION and not a second copy
    of the account. A resolve that was not read-only would let a peer's request write
    the owner's sticky pointer — this device's own next turn would then be pinned to an
    account chosen by another machine — and could block the owner's row on a failure
    the peer observed. So: no stickiness is written for the borrower's session, and no
    block exists.
    """
    _ask_grant(owner, BORROWER_DEVICE, session="peer-sess")
    assert owner.auth.session_credential_id(STUB_PROVIDER, "peer-sess") is None
    assert (
        owner.auth._conn.execute("SELECT COUNT(*) FROM auth_credential_blocks").fetchone()[0] == 0
    )


def test_the_grant_carries_no_refresh_material(owner: Any, idp: RotatingIdP) -> None:
    """The wire shape, checked for the ONE field that would break the model.

    A grant with a refresh token in it would let a borrower rotate the owner's token
    family — the failure the whole slice exists to make impossible — so the absence is
    asserted against the serialised grant rather than eyeballed in the builder. The
    owner's own stored token is the instrument: it must not appear anywhere in the
    detail.
    """
    detail = _ask_grant(owner, BORROWER_DEVICE)
    blob = json.dumps(detail)

    def _keys(payload: Any) -> list[str]:
        if isinstance(payload, dict):
            found: list[str] = []
            for name, value in payload.items():
                found.append(str(name))
                found.extend(_keys(value))
            return found
        if isinstance(payload, list):
            out: list[str] = []
            for item in payload:
                out.extend(_keys(item))
            return out
        return []

    # BY KEY, not by substring: ``refreshed`` is the flag that says the bearer was
    # minted during this request, and a substring test would confuse it with the
    # material whose absence is the point.
    assert not [name for name in _keys(detail) if "refresh" in name and name != "refreshed"]
    assert owner.auth.get_credential(owner.row.id).data["refresh"] not in blob
    assert detail["token_kind"] == "bearer"


def test_a_grant_is_audited_on_the_owner_with_act_and_sub(owner: Any, idp: RotatingIdP) -> None:
    """The delegation record, read back from the real ``audit.jsonl``.

    ``act`` is the owing device (the broker) and ``sub`` is the device the grant was
    lent to — the RFC 8693 delegation markers ``mesh-credentials.md`` §1.4 names and
    ``mesh-incident-response.md`` §4.3 fields. Read from the FILE rather than from a
    fake recorder because the writer keeps a per-event detail whitelist: a field
    missing from ``audit.DETAIL_KEYS`` is dropped without complaint, so a recorder
    would agree with a record that never reached disk.
    """
    detail = _ask_grant(owner, BORROWER_DEVICE, session="sess-1")
    assert detail["kind"] == "grant"
    records = [
        row
        for row in owner.audit.tail(50, network_id="n_owner")
        if row.get("event") == "credential.grant"
    ]
    assert len(records) == 1, records
    record = records[0]
    assert record["actor"] == OWNER_DEVICE
    assert record["subject"] == BORROWER_DEVICE
    fields = record["detail"]
    assert fields["act"] == OWNER_DEVICE
    assert fields["sub"] == BORROWER_DEVICE
    assert fields["credential_key"] == STUB_PROVIDER
    assert fields["scope"] == "session"
    assert isinstance(fields["latency_ms"], int)
    assert isinstance(fields["refreshed"], bool)
    # THE BEARER IS NOT IN THE RECORD, and its key is refused by name rather than
    # being remembered not to be added.
    assert detail["access_token"] not in json.dumps(record)
    assert "access_token" not in fields


def test_a_refused_grant_is_audited_with_its_code(owner: Any, idp: RotatingIdP) -> None:
    """A refusal the operator will be asked about is on the owner's own log too."""
    stranger = "d_00000000000000000000000000000009"
    _ask_grant(owner, stranger)
    records = [
        row
        for row in owner.audit.tail(50, network_id="n_owner")
        if row.get("event") == "credential.grant_refused"
    ]
    assert len(records) == 1, records
    assert records[0]["detail"]["code"] == "not_a_holder"
    assert records[0]["detail"]["sub"] == stranger


# ---------------------------------------------------------------------------
# Who is asking comes from the TRANSPORT (review round 1, F1)
# ---------------------------------------------------------------------------
#
# THE DEFECT THESE CLOSE, reproduced by the reviewer against this fixture: the
# caller was read from ``frame["from_device"]``, a field the peer writes. A member
# holding ``broker_credential`` could therefore put the OWNER'S id in a frame and be
# served a grant, have a forced refresh honoured, and — through the placement arm —
# rewrite the owner's own sharing list on disk, cutting off every real borrower.
# Nothing below is about the parse: the whole point is which device the link says is
# on the far end.

STRANGER = "d_00000000000000000000000000000009"
MCP_URL = "https://mcp.example.test/mcp"
MCP_KEY = "mcp:" + MCP_URL


def _placement_file(owner: Any) -> Path:
    return placement_mod.placement_path("n_owner", owner.root)


def _holder_ids(owner: Any, key: str = STUB_PROVIDER) -> list[str]:
    """The holders ON DISK. Read from the file, never from a live object."""
    document = placement_mod.PlacementDocument.load("n_owner", owner.root, self_device=OWNER_DEVICE)
    entry = document.entry(key)
    assert entry is not None, key
    return [row.device for row in entry.holders]


def _push(device: str, document: dict[str, Any], *, want: str = "push") -> dict[str, Any]:
    return {
        "op": PEER_BROKER_OP,
        "kind": "placement",
        "want": want,
        "network_id": "n_owner",
        "from_device": device,
        "document": document,
    }


def _row_json(key: str, *, owner: str, holders: list[str], rev: int = 2) -> dict[str, Any]:
    return {
        "key": key,
        "provider": key,
        "kind": "oauth-rotating",
        "owner_device": owner,
        "owner_device_name": owner,
        "identity_label": "",
        "holders": [
            {"device": device, "scope": "session", "granted_at": 1.0, "granted_by": owner}
            for device in holders
        ],
        "doc_rev": rev,
    }


def test_every_arm_refuses_a_frame_that_claims_another_devices_id(
    owner: Any, idp: RotatingIdP
) -> None:
    """A stranger's link claiming the OWNER'S id gets a named refusal, and no work.

    All three arms, because all three read an identity: a grant (the holder check, the
    admin check and the audit subject), a report (the refresh arm and the block arm) and
    a placement push (the merge rule). `identity_mismatch` is closed and named, so the
    peer is told what happened rather than being handed a generic internal error.
    """
    frames = [
        _grant_frame(OWNER_DEVICE),
        {
            "op": PEER_BROKER_OP,
            "kind": "report",
            "key": STUB_PROVIDER,
            "provider": STUB_PROVIDER,
            "from_device": OWNER_DEVICE,
            "failure": "quota",
            "model_id": "stub-model",
            "retry_after_ms": 10_000_000,
        },
        _push(
            OWNER_DEVICE,
            {"credentials": [_row_json(STUB_PROVIDER, owner=OWNER_DEVICE, holders=[STRANGER])]},
        ),
        {"op": PEER_BROKER_OP, "kind": "placement", "want": "pull", "from_device": OWNER_DEVICE},
    ]
    for frame in frames:
        detail = _detail(owner.broker.on_broker(_Link(STRANGER), frame))
        assert detail["code"] == "identity_mismatch", (frame["kind"], detail)
        assert "nothing was lent or changed" in detail["message"]
    # Nothing was spent, nothing was written, and nobody was blocked.
    assert idp.posts == []
    assert owner.auth.get_credential(owner.row.id).disabled_cause is None
    assert (
        owner.auth._conn.execute("SELECT COUNT(*) FROM auth_credential_blocks").fetchone()[0] == 0
    )
    assert _holder_ids(owner) == [OWNER_DEVICE, BORROWER_DEVICE, BORROWER_TWO]
    # The forced-refresh arm would have cost a POST through the admin check; the
    # forged frame never reaches it, so the count stays at zero above.


def test_a_forged_placement_push_cannot_rewrite_the_sharing_list(
    owner: Any, idp: RotatingIdP
) -> None:
    """THE REVIEWER'S P1c, on the file: the owner's entry survived a stranger's push.

    ``[owner, B, B2] -> [owner, stranger]`` is what the reviewer measured, and the
    consequence was that every real borrower was cut off and the stranger was admitted
    afterwards. The file is compared BYTE FOR BYTE, so a rewrite that happened to keep
    the same holder set would fail this too.
    """
    before = _placement_file(owner).read_bytes()
    forged = _push(
        OWNER_DEVICE,
        {
            "epoch": 9,
            "credentials": [
                _row_json(STUB_PROVIDER, owner=OWNER_DEVICE, holders=[STRANGER], rev=99)
            ],
        },
    )
    detail = _detail(owner.broker.on_broker(_Link(STRANGER), forged))
    assert detail["code"] == "identity_mismatch"
    assert _placement_file(owner).read_bytes() == before, "a forged push rewrote the document"
    assert _holder_ids(owner) == [OWNER_DEVICE, BORROWER_DEVICE, BORROWER_TWO]
    # And the stranger is still refused on the honest path.
    assert _ask_grant(owner, STRANGER)["code"] == "not_a_holder"


def test_the_same_push_from_the_device_that_owns_the_row_is_still_accepted(owner: Any) -> None:
    """THE POSITIVE CONTROL for the refusal above.

    Without this, "the file did not change" would be satisfied by a merge path that
    never ran. An honest borrower pushing a row IT owns is merged and lands on disk —
    so the refusal above is attributable to the forged identity and not to pushes
    being ignored.
    """
    honest = _push(
        BORROWER_DEVICE,
        {
            "epoch": 3,
            "credentials": [_row_json("deepseek", owner=BORROWER_DEVICE, holders=[OWNER_DEVICE])],
        },
    )
    detail = _detail(owner.broker.on_broker(_Link(BORROWER_DEVICE), honest))
    assert detail["kind"] == "ack", detail
    document = placement_mod.PlacementDocument.load("n_owner", owner.root, self_device=OWNER_DEVICE)
    assert document.owner_of("deepseek") == BORROWER_DEVICE
    # A claim about a THIRD device's row is still dropped, and so is a row that names
    # this device as its owner.
    third = _push(
        BORROWER_DEVICE,
        {
            "epoch": 4,
            "credentials": [
                _row_json("moonshot", owner=BORROWER_TWO, holders=[BORROWER_DEVICE]),
                _row_json("openai", owner=OWNER_DEVICE, holders=[BORROWER_DEVICE], rev=99),
            ],
        },
    )
    _detail(owner.broker.on_broker(_Link(BORROWER_DEVICE), third))
    document = placement_mod.PlacementDocument.load("n_owner", owner.root, self_device=OWNER_DEVICE)
    assert document.entry("moonshot") is None, "a third device's row was taken on a rumour"
    assert document.entry("openai") is None, "a peer's row naming THIS device was accepted"


def test_a_document_that_names_this_device_is_refused_by_merge_itself(tmp_path: Path) -> None:
    """The second lock on the same door, at the merge rule rather than the transport.

    ``merge`` takes ``from_device`` as the caller the TRANSPORT authenticated, so a
    merge "from" this device is a forgery by construction — refused even if a caller
    ever reaches it with the wrong argument.
    """
    document = placement_mod.PlacementDocument("n_net", root=tmp_path, written_by=OWNER_DEVICE)
    incoming = {
        "credentials": [_row_json("openai", owner=OWNER_DEVICE, holders=[STRANGER], rev=99)]
    }
    assert document.merge(incoming, from_device=OWNER_DEVICE, self_device=OWNER_DEVICE) == []
    assert document.entries == {}
    assert document.merge(incoming, from_device="", self_device=OWNER_DEVICE) == []


# ---------------------------------------------------------------------------
# A revoke reaches a running broker, and is never written back (review F2)
# ---------------------------------------------------------------------------


def test_a_revoke_made_while_the_broker_runs_stops_the_service(
    owner: Any, idp: RotatingIdP
) -> None:
    """THE REVIEWER'S P2: revoke through ``mutate``, exactly as the CLI does it.

    The broker was built before the revoke, so a document held on the object would
    still name B2 a holder. ``grant_ttl_s`` does NOT bound this: it bounds a grant
    already lent out, and an ungranted request is decided now.
    """
    assert _ask_grant(owner, BORROWER_TWO)["kind"] == "grant"
    idp.posts.clear()
    with placement_mod.mutate("n_owner", owner.root, self_device=OWNER_DEVICE) as document:
        document.revoke(STUB_PROVIDER, BORROWER_TWO, by=OWNER_DEVICE)
    detail = _ask_grant(owner, BORROWER_TWO, session="sess-2")
    assert detail["code"] == "not_a_holder", detail
    assert idp.posts == []
    assert _holder_ids(owner) == [OWNER_DEVICE, BORROWER_DEVICE]


def test_a_merge_never_writes_a_revoked_holder_back(owner: Any) -> None:
    """THE REVIEWER'S P2b: a merge after a revoke used to resurrect the revoked row.

    The trigger needs no attacker: any peer notification that changed anything made the
    broker save its stale in-memory copy, and the revoke was reverted on disk. So this
    revokes, then performs a merge that really does change the document, and asserts
    BOTH that the change landed and that the revoked holder stayed out.
    """
    with placement_mod.mutate("n_owner", owner.root, self_device=OWNER_DEVICE) as document:
        document.revoke(STUB_PROVIDER, BORROWER_TWO, by=OWNER_DEVICE)
    before = _placement_file(owner).read_bytes()
    assert BORROWER_TWO not in before.decode()

    push = _push(
        BORROWER_DEVICE,
        {
            "epoch": 5,
            "credentials": [_row_json("deepseek", owner=BORROWER_DEVICE, holders=[OWNER_DEVICE])],
        },
    )
    detail = _detail(owner.broker.on_broker(_Link(BORROWER_DEVICE), push))
    assert detail["kind"] == "ack"
    document = placement_mod.PlacementDocument.load("n_owner", owner.root, self_device=OWNER_DEVICE)
    assert document.entry("deepseek") is not None, "the merge did not land, so nothing was proved"
    assert _holder_ids(owner) == [OWNER_DEVICE, BORROWER_DEVICE], "the revoked holder came back"
    assert _ask_grant(owner, BORROWER_TWO, session="sess-3")["code"] == "not_a_holder"


def test_a_share_made_after_the_relay_started_is_served(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_provider: str
) -> None:
    """THE OTHER DIRECTION OF F2, and it needs no restart either.

    A relay that starts before the first ``credential share`` owns nothing to lend yet.
    The first version decided that ONCE, at relay start, so it refused every borrow until
    it was restarted; the handler now decides per request, from the document on disk.
    """
    import asyncio
    from types import SimpleNamespace

    root = tmp_path / "later"
    root.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    from local_operator.network import store as network_store
    from local_operator.network.types import MeshRefusal, NetworkRecord
    from local_operator.providers.auth_store import AuthStore

    record = NetworkRecord(
        network_id="n_owner",
        name="later-net",
        created_by=OWNER_DEVICE,
        self_device_id=OWNER_DEVICE,
        self_role="admin",
        self_capabilities=["admin"],
    )
    network_store.save(record, root)
    server = SimpleNamespace(
        root=root,
        identity=SimpleNamespace(device_id=OWNER_DEVICE, name="owner-laptop"),
        audit=None,
    )
    handler = owner_mod.MeshCredentialBroker.relay_handler(server)
    frame = {
        "op": PEER_BROKER_OP,
        "kind": "grant",
        "key": STUB_PROVIDER,
        "provider": STUB_PROVIDER,
        "from_device": BORROWER_DEVICE,
        "for_session": "sess-1",
        "model_id": "stub-model",
    }
    # Nothing is declared yet: the same by-name refusal a relay with no broker gives.
    with pytest.raises(MeshRefusal) as excinfo:
        handler(_Link(BORROWER_DEVICE), frame)
    assert excinfo.value.code == "not_implemented"

    auth = AuthStore(config_dir=root)
    auth.upsert_credential(
        STUB_PROVIDER,
        {
            # ``type: oauth`` explicitly: without it the store infers an ``api_key``
            # row, and ``get_oauth_access`` — the read the owner's own resolve uses —
            # answers ``None``, so the test would fail on its own fixture.
            "type": "oauth",
            "access": "access-later",
            "expires": int(time.time() * 1000) + 3_600_000,
            "refresh": "",
            "email": "later@example.test",
        },
    )
    try:
        with placement_mod.mutate("n_owner", root, self_device=OWNER_DEVICE) as document:
            document.declare(
                STUB_PROVIDER,
                owner_device=OWNER_DEVICE,
                owner_device_name="owner-laptop",
                provider=STUB_PROVIDER,
                by=OWNER_DEVICE,
            )
            document.grant(STUB_PROVIDER, BORROWER_DEVICE, scope="session", by=OWNER_DEVICE)
        detail = _detail(handler(_Link(BORROWER_DEVICE), frame))
        assert detail["kind"] == "grant", detail
        assert detail["access_token"] == "access-later"
        assert asyncio.run(auth.get_api_key(STUB_PROVIDER, "owner-sess")) == "access-later"
    finally:
        auth.close()


# ---------------------------------------------------------------------------
# What a peer's report may do to the owner's OWN login (review round 1, F3)
# ---------------------------------------------------------------------------
#
# THE DEFECT THESE CLOSE, reproduced by the reviewer: one ``quota`` report with no
# model and a huge retry time wrote an ACCOUNT-WIDE, one-hour block on the owner's
# credential, and the owner's own session then got no key at all. Honest borrowers
# send this report: a provider 529 overload was classified as quota exhaustion, and a
# family-scoped cap lost its scope in transit and arrived account-wide. So every test
# here asserts the OUTCOME the operator cares about — the owner can still get a key
# and complete a turn — and not merely that a column is NULL.


def _block_rows(owner: Any) -> list[tuple[str, int]]:
    return list(
        owner.auth._conn.execute(
            "SELECT block_scope, blocked_until_ms FROM auth_credential_blocks"
        ).fetchall()
    )


def _owner_resolves(owner: Any, model_id: str = "") -> bool:
    import asyncio

    key = asyncio.run(owner.auth.get_api_key(STUB_PROVIDER, "owner-own", model_id=model_id))
    return bool(key)


def test_one_unscoped_quota_report_cannot_lock_the_owner_out_of_its_own_login(
    owner: Any, idp: RotatingIdP
) -> None:
    """THE REVIEWER'S P3, as the operator would notice it: the owner still works.

    ``retry_after_ms`` is a peer's claim, it arrives with no model, and the cap it used
    to reach was the owner's own one-hour ceiling. A report that cannot be scoped
    faithfully is now NOTED and writes nothing at all.
    """
    detail = _report(owner, BORROWER_DEVICE, "quota", retry_after_ms=10_000_000)
    assert detail["action"] == "noted" and detail.get("reason") == "unscoped", detail
    assert _block_rows(owner) == [], "an unscoped peer report wrote a block on the owner"
    assert _owner_resolves(owner), "the owner's own session cannot get a key"


def test_a_scoped_report_blocks_only_that_family_and_only_briefly(
    owner: Any, idp: RotatingIdP
) -> None:
    """A family-scoped 429 is carried faithfully — and bounded by the OWNER's rules.

    The scope is written as the owner's own block READ understands it
    (``model:<family>``), the duration is the owner's shortest backoff rather than the
    peer's claim, and another family on the same account still resolves — which is the
    under-block direction ``rotate_sibling`` argues for.
    """
    detail = _report(
        owner, BORROWER_DEVICE, "quota", model_id="claude-fable-5", retry_after_ms=10_000_000
    )
    assert detail["action"] == "blocked" and detail["scope"] == "model:fable", detail
    assert detail["block_ms"] <= 60_000, detail
    rows = _block_rows(owner)
    assert [row[0] for row in rows] == ["model:fable"], rows
    remaining_ms = rows[0][1] - int(time.time() * 1000)
    assert 0 < remaining_ms <= 60_000, remaining_ms
    # The family the report named is out of rotation; every other model still resolves,
    # and the owner's own session still completes on the account.
    assert owner.auth.is_blocked_for_model(owner.row.id, STUB_PROVIDER, "claude-fable-5") is True
    assert owner.auth.is_blocked_for_model(owner.row.id, STUB_PROVIDER, "claude-opus-5") is False
    assert _owner_resolves(owner, model_id="claude-opus-5a"), "the account was taken out entirely"


def test_a_second_scoped_report_in_the_window_writes_nothing_new(
    owner: Any, idp: RotatingIdP
) -> None:
    """The per-holder rate limit: a broken borrower cannot keep the block alive.

    Without it a holder that reports on every provider call renews its own verdict
    forever, which is the same lock-out by a slower route.
    """
    first = _report(owner, BORROWER_DEVICE, "quota", model_id="claude-fable-5")
    before = _block_rows(owner)
    second = _report(owner, BORROWER_DEVICE, "quota", model_id="claude-fable-5")
    assert first["action"] == "blocked"
    assert second["action"] == "coalesced", second
    assert _block_rows(owner) == before, "the second report extended the block"
    # A DIFFERENT holder is its own slot: the limit is per holder, not global.
    third = _report(owner, BORROWER_TWO, "quota", model_id="claude-fable-5")
    assert third["action"] == "blocked", third


def test_a_report_whose_scope_cannot_be_carried_faithfully_widens_nothing(
    owner: Any, idp: RotatingIdP
) -> None:
    """Unknown family, short slug, or a claimed scope that is not one: write NOTHING.

    The block READ matches a scope by SUBSTRING, so accepting whatever a peer sent would
    let a slug like ``a`` — or an empty one — stop every model on the account, which is
    the over-block this whole arm exists to avoid.
    """
    for extra in (
        {"model_id": "gpt-5-mini"},  # a model this registry parses no family from
        {"block_scope": "a", "model_id": "claude-fable-5"},
        {"block_scope": "model:", "model_id": "claude-fable-5"},
        {"block_scope": "model:not-a-family", "model_id": "claude-fable-5"},
    ):
        detail = _report(owner, BORROWER_DEVICE, "quota", retry_after_ms=10_000_000, **extra)
        assert detail["action"] == "noted" and detail.get("reason") == "unscoped", (extra, detail)
    assert _block_rows(owner) == []
    assert _owner_resolves(owner)


def test_a_provider_overload_is_not_reported_as_quota_exhaustion(
    owner: Any, idp: RotatingIdP
) -> None:
    """THE REVIEWER'S P6: a 529 was mapped to ``quota`` and blocked the account.

    Both halves: the classifier a borrower uses, and what the owner does with the kind
    it produces. ``rotate_sibling`` deprioritises on a provider fault and blocks on a
    usage limit; a peer report must not be more destructive than the owner's own
    reaction, so an ``unavailable`` report changes nothing at all.
    """
    from local_operator.network.credentials.store import report_kind_for
    from local_operator.providers.failover import ProviderError

    assert report_kind_for(ProviderError(529, "overloaded")) == "unavailable"
    assert report_kind_for(ProviderError(503, "service unavailable")) == "unavailable"
    assert report_kind_for(ProviderError(429, "rate limit exceeded")) == "quota"
    assert report_kind_for(ProviderError(401, "invalid_grant")) == "invalid"
    assert report_kind_for(ProviderError(401, "unauthorized")) == "unauthorized"

    detail = _report(owner, BORROWER_DEVICE, "unavailable", model_id="claude-fable-5")
    assert detail == {"kind": "ack", "key": STUB_PROVIDER, "action": "noted"}, detail
    assert _block_rows(owner) == [], "a provider overload wrote a block on the owner"
    assert owner.auth.get_credential(owner.row.id).disabled_cause is None
    assert _owner_resolves(owner)


# ---------------------------------------------------------------------------
# MCP: a real row id, and a bearer that stops at its grant (review F4)
# ---------------------------------------------------------------------------
#
# THE DEFECTS THESE CLOSE, reproduced by the reviewer: ``_resolve_mcp`` converted
# ``McpTokenStorage.credential_id`` — the STRING ``mcp_oauth:<url>`` — to an int, so
# every MCP grant answered ``internal ValueError`` and every MCP report crashed out of
# the handler, on a path no test touched. And a borrowed MCP bearer was kept on the
# auth object until the server rejected it rather than until the grant expired.

MCP_BEARER = "-".join(("mcp", "borrowed", "bearer"))


def _declare_mcp(owner: Any, *, holders: tuple[str, ...] = (BORROWER_DEVICE,)) -> Any:
    """A real ``mcp-oauth`` row on the owner, plus the placement entry for it."""
    row = owner.auth.upsert_credential(
        "mcp-oauth",
        {
            "type": "oauth",
            "tokens": {"access_token": MCP_BEARER, "token_type": "Bearer", "expires_in": 3600},
            "project_id": MCP_URL,
        },
    )
    assert row.identity_key == MCP_URL, row.identity_key
    with placement_mod.mutate("n_owner", owner.root, self_device=OWNER_DEVICE) as document:
        document.declare(
            MCP_KEY,
            owner_device=OWNER_DEVICE,
            owner_device_name="owner-laptop",
            provider="mcp-oauth",
            kind="mcp-rotating",
            by=OWNER_DEVICE,
        )
        for device in holders:
            document.grant(MCP_KEY, device, scope="session", by=OWNER_DEVICE)
    return row


@pytest.fixture()
def no_mcp_refresh(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The owner's own MCP refresh, stubbed: it needs the network, not a broker."""
    calls: list[str] = []

    async def _refresh(url: str, cfg: Any, store: Any = None) -> None:
        calls.append(url)
        return None

    monkeypatch.setattr("local_operator.mcp.auth.ensure_mcp_oauth_fresh", _refresh)
    return calls


def test_an_mcp_grant_serves_the_owners_access_token(owner: Any, no_mcp_refresh: list[str]) -> None:
    """The whole MCP brokering path, which had NO test at all before this round.

    The grant must carry the row's OWN integer id (a synthetic or string id would make
    the borrower's descriptor and the owner's audit disagree about which row was lent),
    and the token must be the owner's stored access token.
    """
    row = _declare_mcp(owner)
    detail = _detail(
        owner.broker.on_broker(
            _Link(BORROWER_DEVICE),
            {
                "op": PEER_BROKER_OP,
                "kind": "grant",
                "key": MCP_KEY,
                "provider": MCP_KEY,
                "from_device": BORROWER_DEVICE,
                "from_session": "",
                "for_session": "sess-1",
                "model_id": "",
            },
        )
    )
    assert detail["kind"] == "grant", detail
    assert detail["access_token"] == MCP_BEARER
    assert detail["credential_ref"]["credential_id"] == row.id
    assert detail["credential_ref"]["credential_id"] > 0
    assert detail["credential_ref"]["kind"] == "mcp-oauth"
    assert no_mcp_refresh == [MCP_URL], "the owner's own refresh is the one that runs"


def test_an_mcp_report_is_answered_and_blocks_nothing(
    owner: Any, no_mcp_refresh: list[str]
) -> None:
    """The report arm was dead for MCP: it raised ``ValueError`` out of the handler.

    A used-up provider error must not be able to reach it either way, which is why
    ``quota`` for an MCP key is NOTED rather than scoped — the owner's provider-side
    block vocabulary is about provider rows, and an MCP server has none.
    """
    _declare_mcp(owner)
    for failure in ("quota", "invalid", "unavailable"):
        reply = owner.broker.on_broker(
            _Link(BORROWER_DEVICE),
            {
                "op": PEER_BROKER_OP,
                "kind": "report",
                "key": MCP_KEY,
                "provider": MCP_KEY,
                "from_device": BORROWER_DEVICE,
                "failure": failure,
                "model_id": "",
                "retry_after_ms": 10_000_000,
            },
        )
        assert reply["op"] == "ack", (failure, reply)
        detail = _detail(reply)
        assert detail["kind"] == "ack" and detail["action"] in ("noted", "refreshed"), (
            failure,
            detail,
        )
    assert _block_rows(owner) == []
    # Exactly ONE owner-side refresh for the three reports: only the ``invalid`` arm
    # asks for one, and the per-credential window coalesces a second ask. A refresh the
    # owner did not offer here is the retry storm the coalescing exists to stop.
    assert no_mcp_refresh == [MCP_URL]


def test_a_borrowed_mcp_bearer_is_re_borrowed_when_its_grant_expires() -> None:
    """F4's second half: the bearer is the GRANT's lifetime, not the connection's.

    The first version cached the bearer on the auth object and re-borrowed only after a
    401, so a borrower whose grant had expired kept presenting the old token until the
    server happened to refuse it — a revocation that never arrived. Driven through the
    real ``async_auth_flow``, with the real ``GrantCache`` applying the real
    ``grant_expires_at_ms``: while the grant is live, no borrow is issued at all, and
    once it is not, the next request borrows again rather than reusing it.
    """
    import asyncio

    import httpx2

    from local_operator.network.credentials.client import GrantCache
    from local_operator.network.credentials.mcp_bearer import BrokeredBearerAuth
    from local_operator.network.credentials.types import CredentialRef, Grant

    def _grant(token: str, *, expires_in_ms: int) -> Grant:
        now_ms = int(time.time() * 1000)
        return Grant(
            access_token=token,
            kind="bearer",
            token_expires_at_ms=now_ms + expires_in_ms,
            grant_expires_at_ms=now_ms + expires_in_ms,
            credential_ref=CredentialRef(
                owner_device=OWNER_DEVICE,
                owner_device_name="owner-laptop",
                provider="mcp-oauth",
                kind="mcp-oauth",
                credential_id=1,
            ),
            served_by=OWNER_DEVICE,
        )

    class _Source:
        def __init__(self) -> None:
            self.grants = GrantCache()
            self.borrows = 0
            self.reports: list[str] = []

        def request_grant_sync(self, key: str, **_kwargs: Any) -> Grant:
            self.borrows += 1
            return _grant(f"fresh-{self.borrows}", expires_in_ms=900_000)

        def report_sync(self, key: str, *, kind: str, **_kwargs: Any) -> None:
            self.reports.append(kind)

    async def _authorization(auth: BrokeredBearerAuth, status: int = 200) -> list[str | None]:
        """Every ``Authorization`` header one request through the flow puts on the wire."""
        generator = auth.async_auth_flow(httpx2.Request("POST", MCP_URL))
        seen: list[str | None] = []
        request = await generator.__anext__()
        seen.append(request.headers.get("Authorization"))
        while True:
            try:
                request = await generator.asend(httpx2.Response(status, request=request))
            except StopAsyncIteration:
                return seen
            seen.append(request.headers.get("Authorization"))

    source = _Source()
    auth = BrokeredBearerAuth(url=MCP_URL, key=MCP_KEY, client=source)

    async def _run() -> tuple[list[str | None], list[str | None]]:
        # A LIVE grant: served from the cache, so the MCP server costs no borrow.
        source.grants.put(MCP_KEY, "", _grant("live-grant", expires_in_ms=600_000))
        served = await _authorization(auth)
        assert source.borrows == 0, "a live grant was re-borrowed for every request"
        # THE GRANT HAS EXPIRED (the owner's task expiry or a revoke), so the next
        # request must borrow again rather than present the token it still holds.
        source.grants.put(MCP_KEY, "", _grant("stale-grant", expires_in_ms=-1_000))
        after = await _authorization(auth)
        return served, after

    served, after = asyncio.run(_run())
    assert served == ["Bearer live-grant"], served
    assert after == ["Bearer fresh-1"], after
    assert source.borrows == 1, source.borrows


def test_the_remote_block_ceiling_is_the_owners_own_shortest_backoff() -> None:
    """The peer's claimed retry time is bounded by the OWNER's rule, and pinned to it.

    Restated rather than imported (``owner.py`` must stay importable on the relay's
    construction path, which cannot pull ``providers.auth_store``), so the relation is
    asserted here instead: a remote quota report buys the shortest block the owner would
    have written on its own evidence, and the owner's own usage probe decides anything
    longer.
    """
    from local_operator.network.credentials.owner import REMOTE_QUOTA_BLOCK_MAX_MS
    from local_operator.providers.auth_store import (
        DEFAULT_BLOCK_MS,
        MAX_CREDENTIAL_BLOCK_MS,
    )

    assert REMOTE_QUOTA_BLOCK_MAX_MS == DEFAULT_BLOCK_MS
    assert REMOTE_QUOTA_BLOCK_MAX_MS < MAX_CREDENTIAL_BLOCK_MS
