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


def test_an_unwrapped_typed_approval_still_audits_desktop_and_an_unknown_surface_reads_desktop(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any  # noqa: F811
) -> None:
    """The ``mobile`` stamp is opt-in: every caller that does not wrap its call, and any value
    outside the allow-list, keeps writing ``desktop`` (MINOR-2: the "desktop unchanged" claim)."""

    _team(monkeypatch)
    _host_, row = _queued(root, fixture_server)
    audits: list[dict[str, Any]] = []
    real_audit = desktop_approval._audit
    monkeypatch.setattr(
        desktop_approval,
        "_audit",
        lambda event, approval_id, root_, detail: (
            audits.append(dict(detail)),
            real_audit(event, approval_id, root_, detail),
        )[1],
    )
    assert desktop_approval.answer_surface() == "desktop"
    with desktop_approval.answering_as("evil"):
        assert desktop_approval.answer_surface() == "desktop"
    with desktop_approval.answering_as("mobile"):
        assert desktop_approval.answer_surface() == "mobile"
    assert desktop_approval.answer_surface() == "desktop"  # reset on exit, no leak
    typed = desktop_approval.row_fields(A.load_record(row.approval_id, root=root)) or {}
    phases = desktop_approval.approve_and_apply(
        root,
        row.approval_id,
        op="guard_consent",
        request_digest=typed["request_digest"],
        disclosure_digest_=typed["disclosure_digest"],
    )
    assert phases["applied"]["ok"], phases
    assert audits and {detail["surface"] for detail in audits} == {"desktop"}
    assert _paired().get(BASE).json()["orgs"][0]["answered_via"] == "desktop"


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


def test_a_grant_that_did_not_reach_radient_is_a_409_and_the_retry_records_no_new_approval(
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
    # The consent engine's own code, as measured: the server's ``unavailable`` rides through;
    # ``guard_consent_not_granted`` is only the fallback when Radient gives none (MINOR-1).
    assert first.json()["code"] == "unavailable", first.text
    assert host.consent(TENANT) is None  # nothing is shared
    assert host.prompt(TENANT, 1).state == "pending"  # type: ignore[union-attr]
    assert counters["signer"] == 1
    fixture_server.grant_reply = (201, {"result": {"consent": {}}})
    retry = client.post(f"{BASE}/consents/{row.approval_id}/accept", json=digests)
    assert retry.status_code == 200, retry.text
    assert retry.json()["state"] == "granted"
    # The approval is not re-signed (no second gesture); the GRANT envelope is a fresh
    # install-key signature with a new nonce, which is why this says "approval" and not "signs
    # nothing" (QA observation).
    assert counters["signer"] == 1
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


def test_decline_of_a_signed_card_whose_grant_did_not_land_denies_it_and_sends_nothing(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any  # noqa: F811
) -> None:
    """The queue still holds the prompt (``pending``), so the decline wins; the signed card is
    then denied to match. The store's guard must NOT block this order (prompt ``declined``)."""

    _team(monkeypatch)
    host, row = _queued(root, fixture_server)
    fixture_server.grant_reply = (503, {"error": "refused", "code": "unavailable"})
    client = _paired()
    first = client.post(
        f"{BASE}/consents/{row.approval_id}/accept", json=_digests(client, row.approval_id)
    )
    assert first.status_code == 409
    assert A.load_record(row.approval_id, root=root)["state"] == "approved"
    response = client.post(f"{BASE}/consents/{row.approval_id}/decline")
    assert response.status_code == 200, response.text
    assert response.json()["state"] == "denied"
    assert A.load_record(row.approval_id, root=root)["state"] == "denied"
    assert host.consent(TENANT) is None and host.sticky(TENANT) == "declined"


def test_a_plain_decline_leaves_the_card_settled_by_the_queue(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any  # noqa: F811
) -> None:
    """Documented shape (design §2): the queue's decline withdraws the unanswered card."""

    _team(monkeypatch)
    _host_, row = _queued(root, fixture_server)
    assert _paired().post(f"{BASE}/consents/{row.approval_id}/decline").status_code == 200
    assert A.load_record(row.approval_id, root=root)["state"] in ("withdrawn", "denied")


def test_decline_loses_to_an_accept_that_lands_inside_the_gap_and_never_says_denied(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any  # noqa: F811
) -> None:
    """S-1, deterministic. Inject a complete accept at the exact point the old code checked the
    prompt and then wrote the deny. The old order answered ``200 denied`` beside a live consent
    (reproduced by the security review); the queue-first order answers 409 and sharing stays."""

    _team(monkeypatch)
    host, row = _queued(root, fixture_server)
    client = _paired()
    digests = _digests(client, row.approval_id)
    real = guard_relay._prompt_waiting
    injected: list[int] = []

    def waiting_then_accept(*args: Any, **kwargs: Any) -> bool:
        waiting = real(*args, **kwargs)
        if not injected:
            injected.append(
                _paired()
                .post(f"{BASE}/consents/{row.approval_id}/accept", json=digests)
                .status_code
            )
        return waiting

    monkeypatch.setattr(guard_relay, "_prompt_waiting", waiting_then_accept)
    response = client.post(f"{BASE}/consents/{row.approval_id}/decline")
    assert injected == [200]  # the grant landed inside the gap
    assert response.status_code == 409, response.text
    assert response.json()["code"] == "approval_decision_conflict"
    assert A.load_record(row.approval_id, root=root)["state"] != "denied"
    assert host.consent(TENANT)["state"] == "live"  # type: ignore[index]
    assert host.sticky(TENANT) is None
    assert len(fixture_server.guard_requests()) == 1
    assert client.get(BASE).json()["orgs"][0]["state"] == "sharing"


def test_decline_whose_queue_step_raises_untyped_after_an_accept_claimed_the_row_is_409_not_denied(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any  # noqa: F811
) -> None:
    """S-4, by injection. The queue decline fails with an UNTYPED error (an unreadable queue)
    at the moment an accept has already claimed and landed the row. The store's per-kind deny
    guard then refuses the card deny, and the old code read that refusal as the expected
    "queue already declined" shape and answered ``200 denied`` beside a live consent. A refused
    deny is only expected once the queue decline SUCCEEDED, so this must now be a 409."""

    _team(monkeypatch)
    host, row = _queued(root, fixture_server)
    client = _paired()
    digests = _digests(client, row.approval_id)
    injected: list[int] = []

    def accept_then_fail(self: Any, *args: Any, **kwargs: Any) -> None:
        injected.append(
            _paired().post(f"{BASE}/consents/{row.approval_id}/accept", json=digests).status_code
        )
        raise RuntimeError("queue unreadable")

    monkeypatch.setattr(C.GuardConsentHost, "decline", accept_then_fail)
    response = client.post(f"{BASE}/consents/{row.approval_id}/decline")
    assert injected == [200]  # the grant landed before the decline's queue step failed
    assert response.status_code == 409, response.text
    assert response.json()["code"] == "approval_decision_conflict"
    assert A.load_record(row.approval_id, root=root)["state"] != "denied"
    assert host.consent(TENANT)["state"] == "live"  # type: ignore[index]
    assert len(fixture_server.guard_requests()) == 1


def test_decline_whose_queue_step_raises_untyped_but_nothing_claimed_still_denies(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any  # noqa: F811
) -> None:
    """The other half of S-4: an unreadable queue with NO accept in flight keeps the old,
    safe behaviour (the card deny stands), so the fix does not turn every queue hiccup into a
    refused "Don't share"."""

    _team(monkeypatch)
    _host_, row = _queued(root, fixture_server)

    def unreadable(self: Any, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("queue unreadable")

    monkeypatch.setattr(C.GuardConsentHost, "decline", unreadable)
    response = _paired().post(f"{BASE}/consents/{row.approval_id}/decline")
    assert response.status_code == 200, response.text
    assert response.json()["state"] == "denied"
    assert A.load_record(row.approval_id, root=root)["state"] == "denied"
    assert fixture_server.guard_requests() == []


def test_a_deny_landing_between_the_cards_verification_and_the_claim_stops_the_grant(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any  # noqa: F811
) -> None:
    """S-5, the 2 ms interleave, deterministic. ``verify_for_run`` has read the signed card
    (approved); then a deny lands while the queue row is still ``pending`` -- exactly when the
    store's per-kind guard lets it through -- and only then does the accept's claim win. The
    old code sent the grant beside a card that says "denied". The post-claim re-read under the
    record lock must refuse, hand the claim back and send nothing."""

    from local_operator.server.utils import desktop_approvals as plane

    _team(monkeypatch)
    host, row = _queued(root, fixture_server)
    client = _paired()
    digests = _digests(client, row.approval_id)
    real_claim = C.GuardConsentHost._claim  # noqa: SLF001
    denied: list[str] = []

    def deny_then_claim(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not denied:
            denied.append(plane.deny(root, row.approval_id)["state"])
        return real_claim(self, *args, **kwargs)

    monkeypatch.setattr(C.GuardConsentHost, "_claim", deny_then_claim)
    response = client.post(f"{BASE}/consents/{row.approval_id}/accept", json=digests)
    assert denied == ["denied"]  # the deny won the card before the claim
    assert response.status_code == 409, response.text
    assert response.json()["code"] == "approval_decision_conflict"
    assert fixture_server.guard_requests() == []  # nothing was sent
    assert A.load_record(row.approval_id, root=root)["state"] == "denied"
    assert host.consent(TENANT) is None and host.sticky(TENANT) is None
    pending = host.prompt(TENANT, 1)
    # The claim was handed back (never stranded ``committing``).
    assert pending is not None and pending.state == C.PROMPT_PENDING


def test_the_stores_deny_refuses_a_guard_card_whose_grant_is_in_flight_or_landed(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any  # noqa: F811
) -> None:
    """The desktop plane's ``deny`` route is the same store write: the guard lives there, so it
    closes S-1 for every surface, and it is NARROWER than "not pending" (a declined row may
    still have its card denied: see the signed-card decline test above)."""

    from local_operator.network.types import MeshRefusal
    from local_operator.server.utils import desktop_approvals as plane

    _team(monkeypatch)
    host, row = _queued(root, fixture_server)
    # ``committing``: claimed, the send in progress. A deny now would sit beside the grant.
    claimed = host._claim(TENANT, 1)  # noqa: SLF001 - the claim IS the in-flight state
    assert claimed is not None
    with pytest.raises(MeshRefusal) as refused:
        plane.deny(root, row.approval_id)
    assert refused.value.code == "approval_decision_conflict"
    assert A.load_record(row.approval_id, root=root)["state"] == "requested"


def test_the_stores_deny_refuses_after_the_grant_landed_on_the_desktop_route_too(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any  # noqa: F811
) -> None:
    from local_operator.network.types import MeshRefusal
    from local_operator.server.utils import desktop_approvals as plane

    _team(monkeypatch)
    host, row = _queued(root, fixture_server)
    client = _paired()
    assert (
        client.post(
            f"{BASE}/consents/{row.approval_id}/accept", json=_digests(client, row.approval_id)
        ).status_code
        == 200
    )
    with pytest.raises(MeshRefusal) as refused:
        plane.deny(root, row.approval_id)
    assert refused.value.code == "approval_decision_conflict"
    assert A.load_record(row.approval_id, root=root)["state"] != "denied"
    assert host.consent(TENANT)["state"] == "live"  # type: ignore[index]


def test_accept_and_decline_released_together_never_end_denied_beside_a_live_consent(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any  # noqa: F811
) -> None:
    """S-1, the real race through the real relay: both requests released from one barrier, many
    iterations (one tenant each, so no sticky or live row carries over). The security review
    measured the old order inconsistent in 1 of 12; the invariants below hold every time:

    * a decline answered ``200 denied`` means NO consent is live and the sticky "no" is set;
    * a live consent means the decline was answered 409 and the card is not ``denied``;
    * exactly one of the two answers won; a grant is sent at most once per tenant.
    """

    import threading
    import time

    iterations = 100
    tenants = [f"org_race{index:02d}" for index in range(iterations)]
    monkeypatch.setattr(
        el, "eligible_orgs", lambda _c: [OrgRef(t, "Acme", "member") for t in tenants]
    )
    host = _host(root, fixture_server)
    cards: dict[str, str] = {}
    for tenant in tenants:
        outcome = host.offer(C.ConsentOffer.from_wire(_offer(tenant=tenant)))
        assert outcome.status == "queued" and outcome.row is not None
        cards[tenant] = outcome.row.approval_id
    accepter, decliner = _paired(), _paired()
    wins = {"accept": 0, "decline": 0}
    for index, tenant in enumerate(tenants):
        approval_id = cards[tenant]
        digests = _digests(accepter, approval_id)
        # The decline starts 0..~40 ms after the accept in a sweep, so some iterations land the
        # decline's check/deny INSIDE the accept's sign-and-send (the gap the old code lost in);
        # a pure same-instant release almost always lets the cheaper decline finish first.
        stagger = (index % 40) * 0.001
        barrier = threading.Barrier(2)
        results: dict[str, Any] = {}

        def run(name: str, call: Any) -> None:
            barrier.wait()
            if name == "decline":
                time.sleep(stagger)
            results[name] = call()

        threads = [
            threading.Thread(
                target=run,
                args=(
                    "accept",
                    lambda: accepter.post(f"{BASE}/consents/{approval_id}/accept", json=digests),
                ),
            ),
            threading.Thread(
                target=run,
                args=("decline", lambda: decliner.post(f"{BASE}/consents/{approval_id}/decline")),
            ),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        accept_r, decline_r = results["accept"], results["decline"]
        consent = host.consent(tenant)
        live = consent is not None and consent["state"] == "live"
        card = A.load_record(approval_id, root=root)["state"]
        context = (
            tenant,
            accept_r.status_code,
            accept_r.text,
            decline_r.status_code,
            decline_r.text,
            card,
        )
        if decline_r.status_code == 200:
            assert decline_r.json()["state"] == "denied", context
            assert not live and host.sticky(tenant) == "declined", context
            assert accept_r.status_code == 409, context
            wins["decline"] += 1
        else:
            assert decline_r.status_code == 409, context
            assert accept_r.status_code == 200 and live, context
            assert card != "denied", context
            wins["accept"] += 1
        assert not (card == "denied" and live), context
    grants = [r for r in fixture_server.guard_requests() if r["path"] == "/v1/guard/consents"]
    assert len(grants) == wins["accept"]
    assert sum(wins.values()) == iterations


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
