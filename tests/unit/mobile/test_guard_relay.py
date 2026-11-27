"""The phone relay's Radient Guard twin (``/api/guard*``, slice L2d).

Driven over the REAL relay app (``mobile.daemon.build_app``) with the real paired-device
cookie, the REAL consent engine and typed approval engine on a real temporary root, a REAL
file-only operator key (the CTO's no-hardware decision is what lets that level sign this op),
and the real ``RadientClient`` talking to the A2 fixture server of ``test_guard_consent`` --
whose ``guard_requests()`` is what "zero ``/v1/guard`` calls" and "exactly one signed grant"
are counted from. Nothing reaches the real Keychain (``LOCAL_OPERATOR_GUARD_KEY_STORE=file``).

WHAT IS PINNED (and, for the load-bearing rows, MUTATED -- see the ``mutation`` tests at the
bottom: each removes one guard and asserts the matching probe then sees the hole, so the
probe is known to be able to fail):

* the route SET is exactly the contract; no path is a grant, an approve or a confirm;
* every route needs the paired-device cookie and answers 401 before any engine call;
* a non-Team account makes zero ``/v1/guard`` requests and builds no Guard file; every write
  and the consent routes refuse ``404 guard_unavailable``;
* ``accept`` is the host-signed typed ``guard_consent`` accept: it needs the card's own two
  digests, refuses a wrong one before the signer, refuses a non-pending card, sends exactly
  one grant, and stamps the answer ``mobile``;
* the relay is not a general approvals proxy (any other card kind is ``404``);
* ``share-again`` is person-gated by the daemon's start-time capture;
* ``features.guard`` is advertised if and only if the twin is mounted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from local_operator.guard import consent as C
from local_operator.guard import install_key, surface
from local_operator.hub_org import eligibility as el
from local_operator.hub_org.eligibility import OrgRef
from local_operator.mobile import daemon as mobile_daemon
from local_operator.mobile import guard_relay
from local_operator.mobile.daemon import MobileDaemon, build_app
from local_operator.network import approvals as A
from local_operator.network.credentials import desktop_approval
from tests.unit.guard.test_guard_consent import (
    TENANT,
    CountingKeyStore,
    _client,
    _offer,
)
from tests.unit.guard.test_guard_consent import (
    fixture_server as _consent_fixture_server,
)
from tests.unit.network.test_approvals_store import _device_request, _make_key
from tests.unit.network.test_approvals_store import root as approvals_root

root = approvals_root
fixture_server = _consent_fixture_server
FILE_ONLY = ("file_only", "operator-file-only")
ZERO = "sha256:" + "0" * 64
BASE = "/api/guard"


@pytest.fixture(autouse=True)
def _a_person_backend(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any
):  # noqa: F811
    """An isolated root, an operator key, a file key store, and no agent-driven markers.

    pytest itself may run inside an agent's shell, which would make ``can_answer`` false for
    every row; rows that need that set the marker themselves BEFORE building the app (the
    relay reads it once, when the app is built, like the desktop's start-time capture).
    """

    _make_key(root)
    monkeypatch.setenv(install_key.BACKEND_ENV, "file")
    for name in surface._AGENT_DRIVEN_MARKERS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv(surface.HOST_CAN_ANSWER_ENV, raising=False)
    surface.reset_serve_capture()
    monkeypatch.setattr(desktop_approval, "assurance", lambda: FILE_ONLY)
    desktop_approval._ATTEMPTS.clear()
    desktop_approval._CANCELLED.clear()
    monkeypatch.setattr(C, "default_transport", lambda _c: _client(fixture_server))
    el.reset_memo()
    yield
    el.reset_memo()
    desktop_approval._ATTEMPTS.clear()
    desktop_approval._CANCELLED.clear()
    surface.reset_serve_capture()


def _team(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(el, "eligible_orgs", lambda _c: [OrgRef(TENANT, "Acme", "member")])


def _non_team(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(el, "eligible_orgs", lambda _c: [])


def _paired() -> TestClient:
    """A phone that signed in: the relay's own cookie, as ``/login`` mints it."""

    client = TestClient(build_app(MobileDaemon(port=0, password="pw123")), follow_redirects=False)
    assert client.post("/login", data={"password": "pw123"}).status_code in (200, 303)
    return client


def _anonymous() -> TestClient:
    return TestClient(build_app(MobileDaemon(port=0, password="pw123")), follow_redirects=False)


def _host(root: Path, fx: Any) -> C.GuardConsentHost:
    return C.GuardConsentHost(root, transport=_client(fx), key_store=CountingKeyStore())


def _queued(root: Path, fx: Any, **over: Any) -> tuple[C.GuardConsentHost, C.PromptRow]:
    host = _host(root, fx)
    outcome = host.offer(C.ConsentOffer.from_wire(_offer(**over)))
    assert outcome.status == "queued" and outcome.row is not None
    assert outcome.row.approval_id, "an anchored host files the desktop mirror"
    return host, outcome.row


def _sharing(root: Path, fx: Any) -> C.GuardConsentHost:
    host = _host(root, fx)
    assert host.offer(C.ConsentOffer.from_wire(_offer())).status == "queued"
    assert host.accept(TENANT, 1, capability=True, signature=None, source="tui").status == "granted"
    return host


def _calls(fx: Any) -> int:
    with fx.lock:
        return len(fx.requests)


def _signer(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    real = A.sign_decision
    counters = {"signer": 0}

    def counted(**kw: Any) -> str:
        counters["signer"] += 1
        return real(**kw)

    monkeypatch.setattr(A, "sign_decision", counted)
    return counters


def _digests(client: TestClient, approval_id: str) -> dict[str, str]:
    card = client.get(f"{BASE}/consents/{approval_id}").json()
    return {
        "request_digest": card["request_digest"],
        "disclosure_digest": card["disclosure_digest"],
    }


# ---------------------------------------------------------------------------
# The route set, the capability and the gate
# ---------------------------------------------------------------------------

_TWIN = (
    ("GET", BASE),
    ("GET", f"{BASE}/non-team-card"),
    ("POST", f"{BASE}/orgs/{{tenant}}/revoke"),
    ("POST", f"{BASE}/orgs/{{tenant}}/share-again"),
    ("GET", f"{BASE}/consents/{{approval_id}}"),
    ("POST", f"{BASE}/consents/{{approval_id}}/accept"),
    ("POST", f"{BASE}/consents/{{approval_id}}/decline"),
)


def _mounted() -> set[tuple[str, str]]:
    """Method + path as the table declares them, minus Starlette's implicit HEAD and the
    ``:str`` converter spelling (``{tenant:str}`` is ``{tenant}``)."""

    app = build_app(MobileDaemon(port=0, password="pw123"))
    return {
        (method, route.path.replace(":str}", "}"))  # type: ignore[attr-defined]
        for route in app.routes
        for method in getattr(route, "methods", set())
        if route.path.startswith(BASE) and method != "HEAD"  # type: ignore[attr-defined]
    }


def test_the_route_set_is_exactly_the_contract_and_has_no_grant_route() -> None:
    assert _mounted() == set(_TWIN)
    # A grant is answered ONLY by the typed accept of the pending guard_consent card; no path
    # may be a generic approve, a scope setter, a report confirm or a notification.
    for _method, path in _TWIN:
        assert not any(word in path for word in ("approve", "grant", "confirm", "scope", "push"))


def _features(client: TestClient) -> dict[str, Any]:
    return client.get("/api/sessions").json()["capabilities"]["features"]


def test_features_guard_is_advertised_because_the_twin_is_mounted() -> None:
    client = _paired()
    assert _features(client)["guard"] == 1
    # ... and the routes it promises answer (a 404 route-not-found would have no `code`).
    assert client.get(BASE).status_code == 200


def test_features_guard_is_withheld_when_the_twin_could_not_mount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A relay that failed to load the twin must not point a phone at routes that 404."""

    def broken(*_a: Any, **_k: Any) -> Any:
        raise ImportError("guard twin unavailable")

    monkeypatch.setattr(guard_relay, "routes", broken)
    client = _paired()
    assert "guard" not in _features(client)
    # The phone's calm "unavailable" line: the route is a plain not-found, the relay is up.
    assert client.get(BASE).status_code == 404
    assert client.get("/api/sessions").status_code == 200


@pytest.mark.parametrize(("method", "path"), _TWIN)
def test_every_route_needs_the_paired_cookie_and_reaches_no_engine(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_server: Any,  # noqa: F811
    method: str,
    path: str,
) -> None:
    _team(monkeypatch)
    host, row = _queued(root, fixture_server)
    counters = _signer(monkeypatch)
    anonymous = _anonymous()
    url = path.format(tenant=TENANT, approval_id=row.approval_id)
    body = {"request_digest": ZERO, "disclosure_digest": ZERO}
    response = anonymous.request(method, url, json=body if "accept" in url else None)
    assert response.status_code == 401, (url, response.text)
    assert response.json() == {"error": "authentication required"}
    # Refused at the gate: nothing signed, nothing sent, the prompt still waiting.
    assert counters["signer"] == 0 and fixture_server.guard_requests() == []
    assert host.prompt(TENANT, 1).state == "pending"  # type: ignore[union-attr]


def test_a_cross_origin_post_is_refused_before_the_engine(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any  # noqa: F811
) -> None:
    """The relay's exact-Origin rule (sibling tunnel hosts share cookies) covers the twin too."""

    _team(monkeypatch)
    _host_, row = _queued(root, fixture_server)
    counters = _signer(monkeypatch)
    client = _paired()
    response = client.post(
        f"{BASE}/consents/{row.approval_id}/accept",
        json=_digests(client, row.approval_id),
        headers={"Origin": "https://evil.example"},
    )
    assert response.status_code == 403
    assert counters["signer"] == 0 and fixture_server.guard_requests() == []


# ---------------------------------------------------------------------------
# Non-Team and Guard-unavailable: zero calls, zero files, 404 guard_unavailable
# ---------------------------------------------------------------------------


def test_a_non_team_account_makes_zero_requests_and_builds_no_guard_file(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any  # noqa: F811
) -> None:
    _non_team(monkeypatch)
    client = _paired()
    response = client.get(BASE)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mode"] == "non_team" and body["orgs"] == [] and body["non_team"]["url"]
    assert _calls(fixture_server) == 0
    assert not (root / "guard").exists()


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("POST", f"{BASE}/orgs/{TENANT}/revoke", None),
        ("POST", f"{BASE}/orgs/{TENANT}/share-again", None),
        ("GET", f"{BASE}/consents/ap_0000000000000", None),
        (
            "POST",
            f"{BASE}/consents/ap_0000000000000/accept",
            {"request_digest": ZERO, "disclosure_digest": ZERO},
        ),
        ("POST", f"{BASE}/consents/ap_0000000000000/decline", None),
    ],
)
def test_every_write_and_consent_route_is_404_guard_unavailable_for_non_team(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_server: Any,  # noqa: F811
    method: str,
    path: str,
    body: Any,
) -> None:
    _non_team(monkeypatch)
    counters = _signer(monkeypatch)
    response = _paired().request(method, path, json=body)
    assert response.status_code == 404, response.text
    assert response.json()["code"] == "guard_unavailable"
    assert counters["signer"] == 0 and _calls(fixture_server) == 0
    assert not (root / "guard").exists()


def test_a_team_account_with_no_org_raising_guard_is_unavailable_too(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any  # noqa: F811
) -> None:
    """Team membership alone is not Guard: no org turned it on, so there is no consent store."""

    _team(monkeypatch)
    client = _paired()
    assert client.get(BASE).json()["mode"] == "none"
    response = client.post(f"{BASE}/orgs/{TENANT}/revoke")
    assert response.status_code == 404 and response.json()["code"] == "guard_unavailable"
    assert _calls(fixture_server) == 0


def test_the_non_team_card_is_the_one_read_that_leaves_the_machine(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any  # noqa: F811
) -> None:
    _non_team(monkeypatch)
    monkeypatch.setattr(
        surface,
        "load_trial_eligibility",
        lambda _c: {"eligible": True, "trial_days": 14, "tenant_name": "Acme\n\x1b[31mCorp"},
    )
    response = _paired().get(f"{BASE}/non-team-card")
    assert response.status_code == 200, response.text
    assert response.json() == {"trial_days": 14, "org": "Acme [31mCorp", "url": surface.team_url()}
    assert fixture_server.guard_requests() == []


def test_the_non_team_card_is_not_for_a_team_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _team(monkeypatch)
    response = _paired().get(f"{BASE}/non-team-card")
    assert response.status_code == 404 and response.json()["code"] == "guard_not_applicable"


# ---------------------------------------------------------------------------
# Status and the engine parity
# ---------------------------------------------------------------------------


def test_status_is_the_desktop_engines_answer_for_the_same_store(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any  # noqa: F811
) -> None:
    from local_operator.server.models.desktop_guard import GuardStatus
    from local_operator.server.utils import desktop_guard as engine

    _team(monkeypatch)
    _host_, row = _queued(root, fixture_server)
    before = _calls(fixture_server)
    body = _paired().get(BASE).json()
    expected = GuardStatus.model_validate(
        engine.status(root, can_answer=True, listed=set())
    ).model_dump(mode="json")
    assert body == expected
    org = body["orgs"][0]
    assert body["badge"] == "waiting" and org["state"] == "waiting"
    assert org["prompt"]["approval_id"] == row.approval_id
    assert _calls(fixture_server) == before  # local facts only: zero network


# ---------------------------------------------------------------------------
# The consent card
# ---------------------------------------------------------------------------


def test_the_card_carries_the_engines_own_digests_and_only_what_the_phone_needs(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any  # noqa: F811
) -> None:
    _team(monkeypatch)
    _host_, row = _queued(root, fixture_server)
    response = _paired().get(f"{BASE}/consents/{row.approval_id}")
    assert response.status_code == 200, response.text
    card = response.json()
    typed = desktop_approval.row_fields(A.load_record(row.approval_id, root=root))
    assert typed is not None
    assert card == {
        "approval_id": row.approval_id,
        "state": "requested",
        "tenant": TENANT,
        "org": "Acme",
        "reason": "new_org",
        "widening": False,
        "asked_scopes": ["decisions", "task_time"],
        "request_digest": typed["request_digest"],
        "disclosure_digest": typed["disclosure_digest"],
        "signable": True,
    }
    # No key beyond the contract: a grant_scopes/signature/principal field would be a leak.
    assert "refusal" not in card


def test_a_card_this_host_would_not_sign_says_so_and_names_why(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any  # noqa: F811
) -> None:
    _team(monkeypatch)
    _host_, row = _queued(root, fixture_server)
    monkeypatch.setattr(desktop_approval, "assurance", lambda: (None, "operator-unavailable"))
    card = _paired().get(f"{BASE}/consents/{row.approval_id}").json()
    assert card["signable"] is False and card["refusal"] == "credential_authority_unavailable"


@pytest.mark.parametrize("approval_id", ["ap_doesnotexist0", "x" * 129, "has space", "bad$id"])
def test_a_missing_or_malformed_id_is_404_approval_not_found(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any, approval_id: str  # noqa: F811
) -> None:
    _team(monkeypatch)
    _queued(root, fixture_server)  # Guard is available: only the id is wrong
    client = _paired()
    for method, suffix in (("GET", ""), ("POST", "/decline")):
        response = client.request(method, f"{BASE}/consents/{approval_id}{suffix}")
        assert response.status_code == 404, (approval_id, response.text)
        assert response.json()["code"] == "approval_not_found"
    accept = client.post(
        f"{BASE}/consents/{approval_id}/accept",
        json={"request_digest": ZERO, "disclosure_digest": ZERO},
    )
    assert accept.status_code == 404 and accept.json()["code"] == "approval_not_found"


def _other_card(root: Path) -> str:
    """A real approval card of ANOTHER kind (a device onboarding request) in the same store."""

    record = A.create_request(**{**_device_request("req_other_kind_01"), "root": root})
    assert record["kind"] != "guard_consent"
    return str(record["approval_id"])


def test_the_relay_is_not_an_approvals_proxy_another_kind_is_indistinguishable_from_none(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any  # noqa: F811
) -> None:
    _team(monkeypatch)
    _queued(root, fixture_server)
    other = _other_card(root)
    counters = _signer(monkeypatch)
    client = _paired()
    answers = [
        client.get(f"{BASE}/consents/{other}"),
        client.post(f"{BASE}/consents/{other}/decline"),
        client.post(
            f"{BASE}/consents/{other}/accept",
            json={"request_digest": ZERO, "disclosure_digest": ZERO},
        ),
    ]
    missing = client.get(f"{BASE}/consents/ap_doesnotexist0")
    for answer in answers:
        assert answer.status_code == 404, answer.text
        assert answer.json() == missing.json()
    assert counters["signer"] == 0
    assert A.load_record(other, root=root)["state"] == "requested"  # untouched, not denied


# ---------------------------------------------------------------------------
# accept: the host-signed typed approve-operation
# ---------------------------------------------------------------------------


def test_accept_with_the_cards_own_digests_signs_once_and_sends_exactly_one_grant(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any  # noqa: F811
) -> None:
    _team(monkeypatch)
    host, row = _queued(root, fixture_server)
    counters = _signer(monkeypatch)
    audits: list[tuple[str, dict[str, Any]]] = []
    real_audit = desktop_approval._audit
    monkeypatch.setattr(
        desktop_approval,
        "_audit",
        lambda event, approval_id, root_, detail: (
            audits.append((event, dict(detail))),
            real_audit(event, approval_id, root_, detail),
        )[1],
    )
    client = _paired()
    response = client.post(
        f"{BASE}/consents/{row.approval_id}/accept", json=_digests(client, row.approval_id)
    )
    assert response.status_code == 200, response.text
    assert response.json() == {
        "approval_id": row.approval_id,
        "state": "granted",
        "tenant": TENANT,
    }
    assert counters["signer"] == 1
    grants = fixture_server.guard_requests()
    assert [(r["method"], r["path"]) for r in grants] == [("POST", "/v1/guard/consents")]
    assert grants[0]["body"]["payload"]["scopes"] == ["decisions", "task_time"]
    consent = host.consent(TENANT)
    assert consent is not None and consent["state"] == "live"
    # The person answered on the phone: the queue and the audit rows both say so.
    org = client.get(BASE).json()["orgs"][0]
    assert (org["state"], org["answered_via"], org["scopes_shared"]) == (
        "sharing",
        "mobile",
        ["decisions", "task_time"],
    )
    assert {detail["surface"] for event, detail in audits if event.startswith("credential.")} == {
        "mobile"
    }
    assert any(event == "credential.desktop_signed" for event, _d in audits)


@pytest.mark.parametrize("stale", ["request_digest", "disclosure_digest"])
def test_accept_with_a_wrong_digest_is_refused_before_the_signer(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_server: Any,  # noqa: F811
    stale: str,
) -> None:
    _team(monkeypatch)
    host, row = _queued(root, fixture_server)
    counters = _signer(monkeypatch)
    client = _paired()
    body = {**_digests(client, row.approval_id), stale: ZERO}
    response = client.post(f"{BASE}/consents/{row.approval_id}/accept", json=body)
    assert response.status_code == 409, response.text
    answer = response.json()
    assert answer["code"] == "guard_consent_expected_mismatch"
    assert answer["engine_code"] in {"credential_request_changed", "credential_disclosure_changed"}
    assert counters["signer"] == 0 and fixture_server.guard_requests() == []
    assert host.prompt(TENANT, 1).state == "pending"  # type: ignore[union-attr]
    # NEGATIVE CONTROL: the identical request with the card's own digests is the success path.
    control = client.post(
        f"{BASE}/consents/{row.approval_id}/accept", json=_digests(client, row.approval_id)
    )
    assert control.status_code == 200 and counters["signer"] == 1


@pytest.mark.parametrize(
    "body",
    [
        None,
        [],
        {},
        {"request_digest": ZERO},
        {"disclosure_digest": ZERO},
        {"request_digest": ZERO, "disclosure_digest": ZERO, "scopes": ["decisions"]},
        {"request_digest": ZERO, "disclosure_digest": ZERO, "operation": "guard_consent"},
        {"request_digest": ZERO, "disclosure_digest": ZERO, "tenant": "ten_other"},
        {"request_digest": "sha256:abc", "disclosure_digest": ZERO},
        {"request_digest": 1, "disclosure_digest": ZERO},
    ],
)
def test_accept_body_is_closed_two_digests_and_nothing_else(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_server: Any,  # noqa: F811
    body: Any,
) -> None:
    """All-or-nothing by construction: the body has no scope, operation or tenant field."""

    _team(monkeypatch)
    host, row = _queued(root, fixture_server)
    counters = _signer(monkeypatch)
    client = _paired()
    if body is None:
        response = client.post(f"{BASE}/consents/{row.approval_id}/accept")
    else:
        response = client.post(f"{BASE}/consents/{row.approval_id}/accept", json=body)
    assert response.status_code == 422, response.text
    assert response.json()["code"] == "invalid_body"
    assert counters["signer"] == 0 and fixture_server.guard_requests() == []
    assert host.prompt(TENANT, 1).state == "pending"  # type: ignore[union-attr]


def test_accept_of_a_card_that_is_no_longer_pending_is_refused_and_signs_nothing_more(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any  # noqa: F811
) -> None:
    _team(monkeypatch)
    _host_, row = _queued(root, fixture_server)
    counters = _signer(monkeypatch)
    client = _paired()
    digests = _digests(client, row.approval_id)
    assert client.post(f"{BASE}/consents/{row.approval_id}/accept", json=digests).status_code == 200
    again = client.post(f"{BASE}/consents/{row.approval_id}/accept", json=digests)
    assert again.status_code == 409, again.text
    assert again.json()["code"] == "approval_decision_conflict"
    assert counters["signer"] == 1 and len(fixture_server.guard_requests()) == 1
    # ... and the card now reads as answered, which is what the phone re-reads to say so.
    card = client.get(f"{BASE}/consents/{row.approval_id}").json()
    assert card["state"] != "requested" and card["signable"] is False


def test_accept_of_a_declined_card_is_refused(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any  # noqa: F811
) -> None:
    _team(monkeypatch)
    _host_, row = _queued(root, fixture_server)
    counters = _signer(monkeypatch)
    client = _paired()
    digests = _digests(client, row.approval_id)
    assert client.post(f"{BASE}/consents/{row.approval_id}/decline").status_code == 200
    response = client.post(f"{BASE}/consents/{row.approval_id}/accept", json=digests)
    assert response.status_code == 409, response.text
    assert response.json()["code"] == "approval_decision_conflict"
    assert counters["signer"] == 0 and fixture_server.guard_requests() == []


def test_a_grant_that_did_not_reach_radient_is_a_409_and_the_retry_signs_nothing_new(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any  # noqa: F811
) -> None:
    """Signed but not applied: never reported as granted; a re-send is the desktop's own retry."""

    _team(monkeypatch)
    host, row = _queued(root, fixture_server)
    counters = _signer(monkeypatch)
    fixture_server.grant_reply = (503, {"error": "refused", "code": "unavailable"})
    client = _paired()
    digests = _digests(client, row.approval_id)
    first = client.post(f"{BASE}/consents/{row.approval_id}/accept", json=digests)
    assert first.status_code == 409, first.text
    assert first.json()["code"], first.text
    assert host.consent(TENANT) is None  # nothing is shared
    assert host.prompt(TENANT, 1).state == "pending"  # type: ignore[union-attr]
    assert counters["signer"] == 1
    fixture_server.grant_reply = (201, {"result": {"consent": {}}})
    retry = client.post(f"{BASE}/consents/{row.approval_id}/accept", json=digests)
    assert retry.status_code == 200, retry.text
    assert retry.json()["state"] == "granted"
    assert counters["signer"] == 1  # the SAME signature was re-sent; no second gesture
    assert host.consent(TENANT)["state"] == "live"  # type: ignore[index]


# ---------------------------------------------------------------------------
# decline
# ---------------------------------------------------------------------------


def test_decline_is_the_hosts_ordinary_deny_sticky_stamped_mobile_and_sends_nothing(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any  # noqa: F811
) -> None:
    _team(monkeypatch)
    host, row = _queued(root, fixture_server)
    counters = _signer(monkeypatch)
    client = _paired()
    response = client.post(f"{BASE}/consents/{row.approval_id}/decline")
    assert response.status_code == 200, response.text
    assert response.json() == {"approval_id": row.approval_id, "state": "denied", "tenant": TENANT}
    assert counters["signer"] == 0 and fixture_server.guard_requests() == []
    assert host.sticky(TENANT) == "declined" and host.pending() == []
    org = client.get(BASE).json()["orgs"][0]
    assert (org["state"], org["answered_via"], org["can_share_again"]) == (
        "not_sharing",
        "mobile",
        True,
    )
    again = client.post(f"{BASE}/consents/{row.approval_id}/decline")
    assert again.status_code == 409  # a decision is write-once


# ---------------------------------------------------------------------------
# revoke and share-again
# ---------------------------------------------------------------------------


def test_decline_of_a_card_whose_grant_already_landed_is_refused_and_sharing_stays(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any  # noqa: F811
) -> None:
    """The approvals store would let a signed card be denied after its grant landed; the relay
    must not write "denied" beside a live consent (the phone would then read "not shared")."""

    _team(monkeypatch)
    host, row = _queued(root, fixture_server)
    client = _paired()
    digests = _digests(client, row.approval_id)
    assert client.post(f"{BASE}/consents/{row.approval_id}/accept", json=digests).status_code == 200
    card_before = A.load_record(row.approval_id, root=root)["state"]
    response = client.post(f"{BASE}/consents/{row.approval_id}/decline")
    assert response.status_code == 409, response.text
    assert response.json()["code"] == "approval_decision_conflict"
    assert A.load_record(row.approval_id, root=root)["state"] == card_before  # not "denied"
    assert host.consent(TENANT)["state"] == "live"  # type: ignore[index]
    assert host.sticky(TENANT) is None
    assert client.get(BASE).json()["orgs"][0]["state"] == "sharing"


def test_revoke_stops_sharing_sends_one_revoke_and_a_second_is_409(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any  # noqa: F811
) -> None:
    _team(monkeypatch)
    host = _sharing(root, fixture_server)
    before = len(fixture_server.guard_requests())
    client = _paired()
    response = client.post(f"{BASE}/orgs/{TENANT}/revoke")
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["status"] == "revoked" and result["org"]["state"] == "paused"
    sent = fixture_server.guard_requests()[before:]
    assert [(r["method"], r["path"], r["body"]) for r in sent] == [
        ("POST", "/v1/guard/consents/revoke", {"tenant": TENANT})
    ]
    assert host.sticky(TENANT) == "revoked"
    again = client.post(f"{BASE}/orgs/{TENANT}/revoke")
    assert again.status_code == 409 and again.json()["code"] == "guard_not_sharing"
    unknown = client.post(f"{BASE}/orgs/ten_other/revoke")
    assert unknown.status_code == 404 and unknown.json()["code"] == "guard_org_unknown"


def test_share_again_lifts_a_sticky_no_and_grants_nothing(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any  # noqa: F811
) -> None:
    _team(monkeypatch)
    host = _host(root, fixture_server)
    host.offer(C.ConsentOffer.from_wire(_offer()))
    host.decline(TENANT, 1, surface="tui")
    fixture_server.status = {"status": "consent_required", "prompt": _offer()}
    client = _paired()
    response = client.post(f"{BASE}/orgs/{TENANT}/share-again")
    assert response.status_code == 200, response.text
    org = response.json()["org"]
    assert org["state"] == "waiting" and org["prompt"] is not None
    assert host.sticky(TENANT) is None and host.consent(TENANT) is None
    # ONE read of the offer and NOT ONE grant: a grant is the host-signed card.
    assert [(r["method"], r["path"]) for r in fixture_server.guard_requests()] == [
        ("GET", "/v1/guard/status")
    ]


@pytest.mark.parametrize("marker", ["LOCAL_OPERATOR_AGENT_SHELL", "LOCAL_OPERATOR_CONSOLE_SURFACE"])
def test_share_again_is_person_gated_by_how_the_relay_daemon_was_started(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_server: Any,  # noqa: F811
    marker: str,
) -> None:
    _team(monkeypatch)
    host = _host(root, fixture_server)
    host.offer(C.ConsentOffer.from_wire(_offer()))
    host.decline(TENANT, 1, surface="tui")
    # The markers are read when the app is BUILT (the daemon's start-time fact); clearing them
    # afterwards must not lift the refusal (a request cannot change how the daemon started).
    monkeypatch.setenv(marker, "1")
    agent_started = _paired()
    monkeypatch.delenv(marker)
    status = agent_started.get(BASE).json()
    assert status["can_answer"] is False and status["orgs"][0]["can_share_again"] is False
    response = agent_started.post(f"{BASE}/orgs/{TENANT}/share-again")
    assert response.status_code == 409 and response.json()["code"] == "guard_answer_not_admitted"
    assert host.sticky(TENANT) == "declined" and fixture_server.guard_requests() == []
    # NEGATIVE CONTROL: the identical request on a daemon a person started succeeds.
    control = _paired().post(f"{BASE}/orgs/{TENANT}/share-again")
    assert control.status_code == 200 and host.sticky(TENANT) is None


# ---------------------------------------------------------------------------
# Mutations: each removes ONE guard and shows the matching probe sees the hole, so the
# probe above is known to be able to fail. (No product code is edited: the guard is
# replaced at its seam for one test.)
# ---------------------------------------------------------------------------


def test_mutation_without_the_kind_check_the_proxy_probe_sees_another_cards_existence(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any  # noqa: F811
) -> None:
    _team(monkeypatch)
    _queued(root, fixture_server)
    other = _other_card(root)
    real = A.load_record

    def no_kind_check(root_: Path, approval_id: str) -> dict[str, Any]:
        return real(approval_id, root=root_)

    monkeypatch.setattr(guard_relay, "_consent_record", no_kind_check)
    response = _paired().post(f"{BASE}/consents/{other}/decline")
    # The mutated relay DENIED a card that is not Guard's (or at least stopped answering
    # 404): exactly what the kind check exists to prevent.
    assert response.status_code != 404 or A.load_record(other, root=root)["state"] != "requested"


def test_mutation_an_open_body_would_let_a_scope_ride_the_accept(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any  # noqa: F811
) -> None:
    from local_operator.server.models import desktop_guard as models

    monkeypatch.setitem(models.GuardConsentAcceptRequest.model_config, "extra", "ignore")
    models.GuardConsentAcceptRequest.model_rebuild(force=True)
    try:
        accepted = models.GuardConsentAcceptRequest.model_validate(
            {"request_digest": ZERO, "disclosure_digest": ZERO, "scopes": ["decisions"]}
        )
    finally:
        monkeypatch.undo()
        models.GuardConsentAcceptRequest.model_rebuild(force=True)
    assert accepted.request_digest == ZERO  # the probe CAN see an open body
    with pytest.raises(Exception):  # noqa: B017 - ValidationError; restored model is closed
        models.GuardConsentAcceptRequest.model_validate(
            {"request_digest": ZERO, "disclosure_digest": ZERO, "scopes": ["decisions"]}
        )


def test_mutation_without_the_pre_signer_digest_check_the_signer_runs_and_a_second_layer_holds(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any  # noqa: F811
) -> None:
    """Neuter the engine's pre-signer digest check: the wrong-digest probe now REACHES the signer.

    That is the hole the check closes (a signature, and for a presence key a Touch ID prompt,
    spent on a card nobody reviewed), so the probe in
    ``test_accept_with_a_wrong_digest_is_refused_before_the_signer`` can see it. The store's
    own write-time ``expected_digest`` check is an INDEPENDENT second layer: it still refuses,
    nothing is recorded and no grant is sent, which is why the mutation answers 409, not 200.
    """

    _team(monkeypatch)
    host, row = _queued(root, fixture_server)
    counters = _signer(monkeypatch)
    real = desktop_approval._validated

    def trusting(root_: Path, approval_id: str, **kw: Any) -> Any:
        record = A._load_raw(approval_id, root_)
        kw["request_digest"] = record["request_digest"]
        kw["disclosure_digest_"] = desktop_approval.disclosure_digest(
            desktop_approval._op(kw["op"]).disclosure(dict(record))
        )
        return real(root_, approval_id, **kw)

    monkeypatch.setattr(desktop_approval, "_validated", trusting)
    response = _paired().post(
        f"{BASE}/consents/{row.approval_id}/accept",
        json={"request_digest": ZERO, "disclosure_digest": ZERO},
    )
    assert counters["signer"] == 1, response.text  # the hole: the signer was reached
    assert response.status_code == 409  # ... and the second layer still refused
    assert fixture_server.guard_requests() == []
    assert host.prompt(TENANT, 1).state == "pending"  # type: ignore[union-attr]


def test_mutation_a_relay_that_ignored_the_start_time_capture_would_lift_the_no(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any  # noqa: F811
) -> None:
    _team(monkeypatch)
    host = _host(root, fixture_server)
    host.offer(C.ConsentOffer.from_wire(_offer()))
    host.decline(TENANT, 1, surface="tui")
    monkeypatch.setenv("LOCAL_OPERATOR_AGENT_SHELL", "1")
    monkeypatch.setattr(surface, "served_host_can_answer", lambda: True)
    client = _paired()
    response = client.post(f"{BASE}/orgs/{TENANT}/share-again")
    assert response.status_code == 200 and host.sticky(TENANT) is None


def test_mutation_a_capability_that_ignores_the_mount_is_a_promise_of_404s(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Force the flag the way a build-time-only advertisement would: key up, routes absent.

    The probe pair (``features.guard`` present, ``GET /api/guard`` is a route-not-found) is
    exactly what ``test_features_guard_is_withheld_when_the_twin_could_not_mount`` forbids, so
    it is known to see the inconsistency when the gate is missing.
    """

    monkeypatch.setattr(guard_relay, "routes", lambda *_a, **_k: [])
    client = _paired()
    assert "guard" not in _features(client)  # the real gate withholds it
    monkeypatch.setattr(mobile_daemon, "_GUARD_TWIN_MOUNTED", True)  # the mutation
    assert _features(client)["guard"] == 1 and client.get(BASE).status_code == 404
