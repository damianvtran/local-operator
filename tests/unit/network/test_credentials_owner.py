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

            def log_message(self, *args: Any) -> None:  # keep pytest output clean
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
    broker = owner_mod.MeshCredentialBroker(
        root=root,
        self_device=OWNER_DEVICE,
        self_device_name="owner-laptop",
        network_id="n_owner",
        placement=document,
        auth_store=auth,
    )
    try:
        yield type(
            "Owner",
            (),
            {"root": root, "auth": auth, "row": row, "broker": broker, "document": document},
        )
    finally:
        broker.close()
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
    owner.document.revoke(STUB_PROVIDER, BORROWER_TWO, by=OWNER_DEVICE)
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


def test_a_peer_report_cannot_disable_or_delete_the_owners_login(
    owner: Any, idp: RotatingIdP
) -> None:
    """THE REQUIREMENT'S NAMED FAILURE, refused three ways.

    A 401, an ``invalid_grant`` and a 429 from a borrower, all against the owner's
    live row. Afterwards every row this device holds is still ENABLED — no
    ``disabled_cause``, nothing deleted — because acting on a peer's observation is
    how one bad 401 on one peer logs the operator out of every device.
    """
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
key = asyncio.run(store.get_api_key("meshtest", "sess-1"))
print(json.dumps({"kind": kind, "key": key, "refresh_calls": len(calls)}))
'''


def test_a_borrower_with_the_owner_away_makes_zero_token_posts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_provider: str, idp: RotatingIdP
) -> None:
    """§2.4 in a SEPARATE PROCESS: no relay, no owner, no grant, no POST.

    Two things this proves that a same-process test could not. First, the borrower
    really is another process with its own config root, so "off the owner" is a fact
    about the runtime rather than an assumption about which object was called.
    Second, the refresh function it WOULD have to call to fall back to a local refresh
    is installed in that child and counts — and it stays at zero, which is the
    structural claim ("the borrower holds no refresh token") measured rather than
    asserted.
    """
    root = tmp_path / "borrower"
    root.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    from local_operator.network.identity import mint

    identity = mint(root, name="borrower-laptop")
    script = tmp_path / "borrow_child.py"
    script.write_text(_BORROWER_CHILD, encoding="utf-8")

    env = {k: v for k, v in os.environ.items() if not k.startswith("CMUX_")}
    env["LOCAL_OPERATOR_CONFIG_DIR"] = str(root)
    completed = subprocess.run(
        [sys.executable, str(script), str(root), OWNER_DEVICE, identity.device_id],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["kind"] == "MeshAwareAuthStore", result
    assert result["key"] is None, "a credential appeared with no owner to lend one"
    assert result["refresh_calls"] == 0, "the borrower tried to refresh locally"
    assert idp.posts == [], "the borrower reached a token endpoint"


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
