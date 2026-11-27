"""Radient Guard member consent on the host (Guard RFC v0.9 §5, slice L2a).

Every test drives the REAL code paths against a local fixture HTTP server that
stands in for agent-server's (not yet built, A2) ``/v1/guard/*`` routes and
``/v1/me/memberships``, with the real ``RadientClient`` as the transport. The
install key always goes to a counting or 0600-file store: no test reaches the
macOS Keychain (a Keychain prompt on the operator's screen is the failure the
host rules forbid).
"""

from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest
from pydantic import SecretStr

from local_operator.clients.radient import RadientClient
from local_operator.guard import consent as C
from local_operator.guard import install_key
from local_operator.hub_org import eligibility as el
from local_operator.hub_org import enrol

TERMS = "sha256:" + "ab" * 32
TERMS_2 = "sha256:" + "cd" * 32
TENANT = "ten_acme"


# ---------------------------------------------------------------------------
# Fixture server (the A2 contract this slice codes against)
# ---------------------------------------------------------------------------


class _Fixture:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.memberships: list[dict[str, Any]] = []
        self.status: dict[str, Any] = {"status": "sharing"}
        self.grant_reply: tuple[int, dict[str, Any]] = (201, {"result": {"consent": {}}})
        self.revoke_reply: tuple[int, dict[str, Any]] = (200, {"result": {"revoked": True}})
        self.lock = threading.Lock()

    def guard_requests(self) -> list[dict[str, Any]]:
        with self.lock:
            return [r for r in self.requests if r["path"].startswith("/v1/guard/")]


def _handler(fx: _Fixture) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:  # quiet
            return

        def _reply(self, status: int, body: dict[str, Any]) -> None:
            raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _record(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            parsed = urlparse(self.path)
            entry = {
                "method": self.command,
                "path": parsed.path,
                "query": parse_qs(parsed.query),
                "headers": {k.lower(): v for k, v in self.headers.items()},
                "body": json.loads(body) if body else None,
            }
            with fx.lock:
                fx.requests.append(entry)
            return entry

        def do_GET(self) -> None:
            entry = self._record()
            if entry["path"] == "/v1/me/memberships":
                self._reply(200, {"msg": "ok", "result": {"memberships": fx.memberships}})
            elif entry["path"] == "/v1/guard/status":
                self._reply(200, {"msg": "ok", "result": fx.status})
            else:
                self._reply(404, {"error": "refused", "code": "not_found"})

        def do_POST(self) -> None:
            entry = self._record()
            if entry["path"] == "/v1/guard/consents":
                self._reply(*fx.grant_reply)
            elif entry["path"] == "/v1/guard/consents/revoke":
                self._reply(*fx.revoke_reply)
            else:
                self._reply(404, {"error": "refused", "code": "not_found"})

    return Handler


@pytest.fixture
def fixture_server():
    fx = _Fixture()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(fx))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    fx.base = f"http://127.0.0.1:{server.server_address[1]}/v1"  # type: ignore[attr-defined]
    try:
        yield fx
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class CountingKeyStore:
    """An in-memory credential store that counts writes (stands in for Keychain)."""

    name = "counting"

    def __init__(self) -> None:
        self.items: dict[str, str] = {}
        self.writes = 0
        self.reads = 0

    def load(self, install_id: str) -> str | None:
        self.reads += 1
        return self.items.get(install_id)

    def store(self, install_id: str, secret_hex: str) -> None:
        self.writes += 1
        self.items[install_id] = secret_hex

    def delete(self, install_id: str) -> None:
        self.items.pop(install_id, None)


@pytest.fixture
def config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from local_operator.operator import trust

    root = tmp_path / "config"
    root.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    # Belt and braces: even a code path that ignores the injected store must
    # never reach the real Keychain from a test.
    monkeypatch.setenv(install_key.BACKEND_ENV, "file")

    def absent(uid: int | str | None = None) -> Any:
        return trust.AnchorLoad(
            anchor=None,
            path=trust.anchor_path(uid),
            root_owned=False,
            reason="pinned absent by the test",
            exists=False,
        )

    monkeypatch.setattr(trust, "load_anchor", absent)
    el.reset_memo()
    yield root
    el.reset_memo()


def _offer(**over: Any) -> dict[str, Any]:
    body = {
        "tenant": TENANT,
        "org_display_name": "Acme",
        "collection_rev": 1,
        "revocation_gen": 0,
        "scopes": ["decisions", "task_time"],
        "added_scopes": [],
        "rendered_terms_digest": TERMS,
        "reason": "new_org",
    }
    body.update(over)
    return body


def _client(fx: _Fixture) -> RadientClient:
    return RadientClient(api_key=SecretStr("jwt-person"), base_url=fx.base)


def _host(root: Path, fx: _Fixture, store: Any) -> C.GuardConsentHost:
    return C.GuardConsentHost(root, transport=_client(fx), key_store=store)


def _membership(plan_status: str, tenant: str = TENANT) -> dict[str, Any]:
    return {
        "tenant_id": tenant,
        "tenant_name": "Acme",
        "role": "member",
        "status": "active",
        "is_home": False,
        "plan": {"status": plan_status, "seats": 5},
    }


@pytest.fixture
def boundary_rig(config_dir: Path, fixture_server: _Fixture, monkeypatch: pytest.MonkeyPatch):
    """``hub_org.enrol.run_boundary`` -- the real sign-in / start / 6 h tick body --
    against the fixture server, as the signed-in person."""

    monkeypatch.setattr(el, "_configured_base_url", lambda _c: fixture_server.base)
    monkeypatch.setattr(
        el, "person_client", lambda _c, base_url=None: (_client(fixture_server), "acct_1")
    )
    monkeypatch.setattr(enrol, "_refresh_vault_gate", lambda *a, **k: None)
    monkeypatch.setattr(enrol, "auto_enrol_enabled", lambda _c: False)
    return config_dir, fixture_server


# ---------------------------------------------------------------------------
# Non-Team: zero /v1/guard/* calls and zero credential-store writes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("plan_status", ["none", "canceled"])
def test_non_team_sign_in_restart_and_idle_make_zero_guard_calls(
    boundary_rig: Any, monkeypatch: pytest.MonkeyPatch, plan_status: str
) -> None:
    root, fx = boundary_rig
    fx.memberships = [_membership(plan_status)]
    fx.status = {"status": "consent_required", "prompt": _offer()}  # would queue if asked
    store = CountingKeyStore()
    monkeypatch.setattr(install_key, "default_store", lambda _c: store)
    # Sign-in, daemon start ("restart"), then four 6 h ticks ("idle"), each the
    # real boundary body with a fresh process memo, as a restart has.
    for boundary in ["sign-in", "daemon-start", "tick", "tick", "tick", "tick"]:
        el.reset_memo()
        enrol.run_boundary(root, boundary, force=True)
    memberships = [r for r in fx.requests if r["path"] == "/v1/me/memberships"]
    assert len(memberships) == 6  # the boundaries really ran against the server
    assert fx.guard_requests() == []
    assert store.writes == 0 and store.reads == 0
    assert not (root / "guard").exists()  # no Guard file of any kind


def test_team_member_status_read_queues_one_prompt_without_any_key_write(
    boundary_rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, fx = boundary_rig
    fx.memberships = [_membership("active")]
    fx.status = {"status": "consent_required", "prompt": _offer()}
    store = CountingKeyStore()
    monkeypatch.setattr(install_key, "default_store", lambda _c: store)
    enrol.run_boundary(root, "sign-in", force=True)
    guard = fx.guard_requests()
    assert [(r["method"], r["path"]) for r in guard] == [("GET", "/v1/guard/status")]
    assert guard[0]["query"] == {"tenant": [TENANT]}
    assert guard[0]["headers"]["authorization"] == "Bearer jwt-person"
    assert "origin" not in guard[0]["headers"]
    pending = C.GuardConsentHost(root).pending()
    assert [(p.tenant, p.collection_rev, p.state) for p in pending] == [(TENANT, 1, "pending")]
    # A status read is never enrolment: no key exists before a person answers.
    assert store.writes == 0
    assert not install_key.key_file(root).exists()


def test_team_status_404_before_a2_is_a_quiet_no_op(boundary_rig: Any) -> None:
    """Until A2 ships, production answers 404 for /v1/guard/status: nothing is queued."""
    root, fx = boundary_rig
    fx.memberships = [_membership("active")]
    el.reset_memo()
    enrol.run_boundary(root, "sign-in", force=True)  # writes the Team gate record
    outcomes = C.check_point(
        root,
        boundary="tick",
        # ``/missing/guard/status`` is an unknown path on the fixture: a 404.
        transport_factory=lambda: RadientClient(
            api_key=SecretStr("jwt"), base_url=fx.base + "/missing"
        ),
    )
    assert outcomes == []
    assert any(r["path"] == "/v1/missing/guard/status" for r in fx.requests)
    assert C.GuardConsentHost(root).pending() == []


# ---------------------------------------------------------------------------
# Admission: nothing but an admitted answer sends anything
# ---------------------------------------------------------------------------


#: Each non-admitted source maps to the two facts ``admit_increasing`` reads:
#: none of them holds the spawn capability or a verified signature.
NOT_ADMITTED = {
    "yolo": (False, None),
    "approvals-auto": (False, None),
    "bridge_decide": (False, None),
    "peer-send": (False, None),
    "mesh-unsigned": (False, None),
    "mesh-bad-signature": (False, False),
    "pane-without-capability": (False, None),
}


@pytest.mark.parametrize("source", sorted(NOT_ADMITTED))
def test_unadmitted_answers_leave_the_card_pending_and_send_nothing(
    config_dir: Path, fixture_server: _Fixture, source: str
) -> None:
    store = CountingKeyStore()
    host = _host(config_dir, fixture_server, store)
    assert host.offer(C.ConsentOffer.from_wire(_offer())).status == "queued"
    capability, signature = NOT_ADMITTED[source]
    outcome = host.accept(TENANT, 1, capability=capability, signature=signature, source=source)
    assert outcome.status == "refused" and outcome.code == "guard_answer_not_admitted"
    assert fixture_server.guard_requests() == []
    assert store.writes == 0 and store.reads == 0
    assert [p.state for p in host.pending()] == ["pending"]
    assert [(r["source"], r["code"]) for r in host.refusals()] == [
        (source, "guard_answer_not_admitted")
    ]


def test_admitted_accept_sends_exactly_one_signed_request_and_enrols_lazily(
    config_dir: Path, fixture_server: _Fixture
) -> None:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    store = CountingKeyStore()
    host = _host(config_dir, fixture_server, store)
    host.offer(C.ConsentOffer.from_wire(_offer()))
    assert store.writes == 0
    outcome = host.accept(TENANT, 1, capability=True, signature=None, source="tui")
    assert outcome.status == "granted"
    grants = fixture_server.guard_requests()
    assert [(r["method"], r["path"]) for r in grants] == [("POST", "/v1/guard/consents")]
    envelope = grants[0]["body"]
    payload = envelope["payload"]
    assert payload["scopes"] == ["decisions", "task_time"]
    assert payload["tenant"] == TENANT and payload["terms_digest"] == TERMS
    assert "origin" not in grants[0]["headers"]
    raw_pub = base64.urlsafe_b64decode(payload["enroll"]["public_key"] + "==")
    sig = base64.urlsafe_b64decode(envelope["signature"] + "==")
    Ed25519PublicKey.from_public_bytes(raw_pub).verify(
        sig, C.SIGNING_DOMAIN + C.canonical_json(payload)
    )
    assert store.writes == 1  # created inside the admitted accept, once
    # A second answer to the same card sends nothing more.
    again = host.accept(TENANT, 1, capability=True, signature=None, source="tui")
    assert again.status == "refused" and len(fixture_server.guard_requests()) == 1
    assert host.consent(TENANT) == {
        "tenant": TENANT,
        "collection_rev": 1,
        "scopes": ["decisions", "task_time"],
        "state": "live",
    }
    assert [n["kind"] for n in host.notices()] == ["grant"]


def test_signature_true_is_admitted_without_capability(
    config_dir: Path, fixture_server: _Fixture
) -> None:
    host = _host(config_dir, fixture_server, CountingKeyStore())
    host.offer(C.ConsentOffer.from_wire(_offer()))
    outcome = host.accept(TENANT, 1, capability=False, signature=True, source="phone")
    assert outcome.status == "granted"
    assert len(fixture_server.guard_requests()) == 1


def test_queue_survives_restart_and_second_grant_does_not_re_enrol(
    config_dir: Path, fixture_server: _Fixture
) -> None:
    store = CountingKeyStore()
    _host(config_dir, fixture_server, store).offer(C.ConsentOffer.from_wire(_offer()))
    # "Restart": a brand-new host over the same config root.
    restarted = _host(config_dir, fixture_server, store)
    assert [(p.tenant, p.state) for p in restarted.pending()] == [(TENANT, "pending")]
    assert restarted.accept(TENANT, 1, capability=True, signature=None, source="tui").status == (
        "granted"
    )
    # A later widening, after another restart, reuses the key and omits enroll.
    later = _host(config_dir, fixture_server, store)
    later.offer(
        C.ConsentOffer.from_wire(
            _offer(
                collection_rev=2,
                scopes=["decisions", "review_cycles", "task_time"],
                added_scopes=["review_cycles"],
                reason="widened",
                rendered_terms_digest=TERMS_2,
            )
        )
    )
    assert later.accept(TENANT, 2, capability=True, signature=None, source="tui").status == (
        "granted"
    )
    bodies = [r["body"]["payload"] for r in fixture_server.guard_requests()]
    assert "enroll" in bodies[0] and "enroll" not in bodies[1]
    assert bodies[1]["scopes"] == ["decisions", "review_cycles", "task_time"]
    assert store.writes == 1
    assert [n["kind"] for n in later.notices()] == ["grant", "widen"]


def test_server_refusal_keeps_the_card_pending(config_dir: Path, fixture_server: _Fixture) -> None:
    fixture_server.grant_reply = (403, {"error": "refused", "code": "guard_not_entitled"})
    host = _host(config_dir, fixture_server, CountingKeyStore())
    host.offer(C.ConsentOffer.from_wire(_offer()))
    outcome = host.accept(TENANT, 1, capability=True, signature=None, source="tui")
    assert outcome.status == "failed"
    assert [p.state for p in host.pending()] == ["pending"]
    assert host.consent(TENANT) is None


# ---------------------------------------------------------------------------
# Sticky decline, per-scope widening decline (U10), revoke
# ---------------------------------------------------------------------------


def test_decline_is_sticky_until_share_again(config_dir: Path, fixture_server: _Fixture) -> None:
    host = _host(config_dir, fixture_server, CountingKeyStore())
    host.offer(C.ConsentOffer.from_wire(_offer()))
    host.decline(TENANT, 1, surface="tui")
    for rev in (1, 2, 3):
        assert host.offer(C.ConsentOffer.from_wire(_offer(collection_rev=rev))).status == (
            "suppressed_sticky"
        )
    assert host.pending() == [] and fixture_server.guard_requests() == []
    host.share_again(TENANT)
    assert host.offer(C.ConsentOffer.from_wire(_offer(collection_rev=4))).status == "queued"


def test_widening_keep_current_suppresses_same_scopes_and_prompts_only_the_difference(
    config_dir: Path, fixture_server: _Fixture
) -> None:
    host = _host(config_dir, fixture_server, CountingKeyStore())
    host.offer(C.ConsentOffer.from_wire(_offer()))
    host.accept(TENANT, 1, capability=True, signature=None, source="tui")
    widened = _offer(
        collection_rev=2,
        scopes=["decisions", "posture", "task_time"],
        added_scopes=["posture"],
        reason="widened",
    )
    assert host.offer(C.ConsentOffer.from_wire(widened)).status == "queued"
    host.decline(TENANT, 2, surface="tui")  # "Keep current"
    assert host.declined_scopes(TENANT) == {"posture"}
    assert host.sticky(TENANT) is None  # sharing the old scopes continues
    # The same widening on the next three status calls: no row, no notice.
    for _ in range(3):
        assert host.offer(C.ConsentOffer.from_wire(widened)).status in {
            "duplicate",
            "suppressed_declined_scopes",
        }
    for rev in (3, 4, 5):
        same = dict(widened, collection_rev=rev)
        assert host.offer(C.ConsentOffer.from_wire(same)).status == "suppressed_declined_scopes"
    assert host.pending() == []
    # A later widening adding one more scope asks for that scope only.
    more = _offer(
        collection_rev=6,
        scopes=["decisions", "posture", "review_cycles", "task_time"],
        added_scopes=["posture", "review_cycles"],
        reason="widened",
    )
    queued = host.offer(C.ConsentOffer.from_wire(more))
    assert queued.status == "queued"
    assert queued.row is not None and queued.row.asked_scopes == ("review_cycles",)
    assert queued.row.grant_scopes == ("decisions", "review_cycles", "task_time")
    assert host.consent(TENANT)["state"] == "live"
    assert [n["kind"] for n in host.notices()] == ["grant"]


def test_revoke_is_jwt_only_sticky_and_noticed(config_dir: Path, fixture_server: _Fixture) -> None:
    host = _host(config_dir, fixture_server, CountingKeyStore())
    host.offer(C.ConsentOffer.from_wire(_offer()))
    host.accept(TENANT, 1, capability=True, signature=None, source="tui")
    assert host.revoke(TENANT).status == "granted"
    revoke = fixture_server.guard_requests()[-1]
    assert (revoke["method"], revoke["path"], revoke["body"]) == (
        "POST",
        "/v1/guard/consents/revoke",
        {"tenant": TENANT},
    )
    assert "signature" not in json.dumps(revoke["body"])
    assert host.consent(TENANT)["state"] == "revoked"
    assert host.sticky(TENANT) == "revoked"
    assert host.offer(C.ConsentOffer.from_wire(_offer(collection_rev=2))).status == (
        "suppressed_sticky"
    )
    assert [n["kind"] for n in host.notices()] == ["grant", "revoke"]


def test_revoke_that_cannot_reach_radient_stays_pending_and_retries(
    config_dir: Path, fixture_server: _Fixture
) -> None:
    host = _host(config_dir, fixture_server, CountingKeyStore())
    host.offer(C.ConsentOffer.from_wire(_offer()))
    host.accept(TENANT, 1, capability=True, signature=None, source="tui")
    fixture_server.revoke_reply = (503, {"error": "refused", "code": "unavailable"})
    assert host.revoke(TENANT).status == "failed"
    assert host.consent(TENANT)["state"] == "revoke_pending"
    fixture_server.revoke_reply = (200, {"result": {}})
    host.retry_pending_revokes()
    assert host.consent(TENANT)["state"] == "revoked"


def test_console_revoke_observed_on_accept_is_sticky(
    config_dir: Path, fixture_server: _Fixture
) -> None:
    fixture_server.grant_reply = (409, {"error": "refused", "code": "guard_revocation_stale"})
    host = _host(config_dir, fixture_server, CountingKeyStore())
    host.offer(C.ConsentOffer.from_wire(_offer()))
    outcome = host.accept(TENANT, 1, capability=True, signature=None, source="tui")
    assert outcome.status == "revoked_first"
    assert host.sticky(TENANT) == "revoked" and host.pending() == []
    assert [n["kind"] for n in host.notices()] == ["console_revoke"]


@pytest.mark.parametrize(
    "bad",
    [
        {"scopes": ["decisions", "keystrokes"]},
        {"collection_rev": True},
        {"reason": "because"},
        {"rendered_terms_digest": "md5:x"},
        {"reason": "widened", "added_scopes": []},
        {"added_scopes": ["posture"]},
    ],
)
def test_malformed_offers_are_refused(bad: dict[str, Any]) -> None:
    with pytest.raises(C.GuardConsentRefused):
        C.ConsentOffer.from_wire(_offer(**bad))


def test_damaged_store_is_refused_not_recreated(config_dir: Path) -> None:
    path = C.guard_consent_path(config_dir)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"this is not sqlite" * 100)
    with pytest.raises(C.GuardConsentRefused) as refused:
        C.GuardConsentHost(config_dir).offer(C.ConsentOffer.from_wire(_offer()))
    assert refused.value.code == "consent_store_damaged"
    assert path.read_bytes().startswith(b"this is not sqlite")


def test_store_permissions_are_owner_only(config_dir: Path, fixture_server: _Fixture) -> None:
    host = _host(config_dir, fixture_server, CountingKeyStore())
    host.offer(C.ConsentOffer.from_wire(_offer()))
    assert (C.guard_consent_path(config_dir).stat().st_mode & 0o777) == 0o600
    assert (C.guard_consent_path(config_dir).parent.stat().st_mode & 0o777) == 0o700


def test_file_key_store_is_0600_and_never_created_by_load(config_dir: Path) -> None:
    store = install_key.FileKeyStore(config_dir)
    assert install_key.load(store, "inst") is None
    assert not install_key.key_file(config_dir).exists()
    key, created = install_key.get_or_create(store, "inst")
    assert created and (install_key.key_file(config_dir).stat().st_mode & 0o777) == 0o600
    again, created_again = install_key.get_or_create(store, "inst")
    assert not created_again
    assert install_key.public_raw(again) == install_key.public_raw(key)
