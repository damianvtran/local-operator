"""``/v1/desktop/guard``: the TUI's ``/guard`` screen over the desktop HTTP plane (slice L2c).

Driven over the REAL router mounted behind the real ``require_desktop`` bearer, against
the REAL consent engine on a real temporary root and the real ``RadientClient`` talking to
the A2 fixture server of ``test_guard_consent`` (whose ``guard_requests()`` is what
"zero ``/v1/guard`` calls" is counted from). Nothing here reaches the real Keychain
(``LOCAL_OPERATOR_GUARD_KEY_STORE=file`` and an injected counting store).

WHAT IS PINNED, in the order the contract states it:

* every route is bearer-gated; the route SET is exactly the contract's (no grant, no
  report-confirm route exists);
* a non-Team account makes zero requests of any kind and builds no Guard file, on every
  route, and every write answers ``404 guard_unavailable`` (with a Team account as the
  negative control that proves the counter can see a call);
* the routes call the TUI's own engine: the same predicates, badge and revoke;
* ``share-again`` is refused wherever the TUI's person-only gate refuses and grants nothing;
* revoke commits the local stop even when Radient cannot be reached.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.guard import consent as C
from local_operator.guard import install_key, surface
from local_operator.guard.reports import GuardReports
from local_operator.hub_org import eligibility as el
from local_operator.hub_org.eligibility import OrgRef
from local_operator.server import desktop
from local_operator.server.features import DESKTOP_ONLY_FEATURES, feature_flags
from local_operator.server.routes import desktop_approvals as approval_routes
from local_operator.server.routes import desktop_guard as guard_routes
from local_operator.server.utils.desktop_sessions import DesktopSessions
from tests.unit.guard.test_guard_consent import (
    TENANT,
    CountingKeyStore,
    _client,
    _offer,
)
from tests.unit.guard.test_guard_consent import (
    fixture_server as _consent_fixture_server,
)
from tests.unit.network.test_approvals_store import _make_key
from tests.unit.network.test_approvals_store import root as approvals_root

root = approvals_root
fixture_server = _consent_fixture_server
TOKEN = "g" * 43
BASE = "/v1/desktop/guard"


@pytest.fixture(autouse=True)
def _a_person_backend(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any
):  # noqa: F811
    """An isolated root, an operator key, a file key store, and no agent-driven markers.

    pytest itself may run inside an agent's shell, which would make ``can_answer`` false for
    every row; rows that need that set the marker themselves.
    """

    _make_key(root)
    monkeypatch.setenv(install_key.BACKEND_ENV, "file")
    for name in surface._AGENT_DRIVEN_MARKERS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(C, "default_transport", lambda _c: _client(fixture_server))
    el.reset_memo()
    yield
    el.reset_memo()


def _team(monkeypatch: pytest.MonkeyPatch, *orgs: OrgRef) -> None:
    """The cached membership read (``hub_org.eligibility``), the one input both planes share."""

    listed = list(orgs) or [OrgRef(TENANT, "Acme", "member")]
    monkeypatch.setattr(el, "eligible_orgs", lambda _c: list(listed))


def _non_team(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(el, "eligible_orgs", lambda _c: [])


@asynccontextmanager
async def _http(
    root: Path, monkeypatch: pytest.MonkeyPatch, *, can_answer: bool | None = None
) -> AsyncIterator[AsyncClient]:
    monkeypatch.setattr(desktop, "_SPAWN_BEARER", None)
    monkeypatch.setattr(desktop, "_CLAIMED", None)
    monkeypatch.setenv(desktop.TOKEN_ENV, TOKEN)
    app = FastAPI()
    app.state.config_manager = ConfigManager(root)
    app.state.desktop_sessions = DesktopSessions(root)
    if can_answer is not None:
        # What the lifespan publishes from the start-time capture on a real ``lop serve``.
        app.state.guard_host_can_answer = can_answer
    app.include_router(approval_routes.router)
    app.include_router(guard_routes.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield client


def _host(root: Path, fx: Any) -> C.GuardConsentHost:
    return C.GuardConsentHost(root, transport=_client(fx), key_store=CountingKeyStore())


def _sharing(root: Path, fx: Any, **over: Any) -> C.GuardConsentHost:
    """An org that raised Guard and a person who said yes in the terminal (the engine path)."""

    host = _host(root, fx)
    assert host.offer(C.ConsentOffer.from_wire(_offer(**over))).status == "queued"
    assert (
        host.accept(
            TENANT, over.get("collection_rev", 1), capability=True, signature=None, source="tui"
        ).status
        == "granted"
    )
    return host


def _body(response: Any) -> Any:
    return response.json()["result"]


def _code(response: Any) -> str:
    return response.json()["detail"]["code"]


def _calls(fx: Any) -> int:
    with fx.lock:
        return len(fx.requests)


# ---------------------------------------------------------------------------
# The route set and the gate
# ---------------------------------------------------------------------------

_WRITES = (
    ("POST", f"{BASE}/orgs/{TENANT}/revoke", None),
    ("POST", f"{BASE}/orgs/{TENANT}/share-again", None),
    ("POST", f"{BASE}/reports/rep_1/discard", None),
    ("POST", f"{BASE}/notices/1/shown", None),
    ("POST", f"{BASE}/notice-due", {"surface": "desktop"}),
)
_ALL = (("GET", BASE, None), ("GET", f"{BASE}/non-team-card", None), *_WRITES)


def test_the_route_set_is_exactly_the_contract_and_has_no_grant_route() -> None:
    served = {
        (method, getattr(route, "path", ""))
        for route in guard_routes.router.routes
        for method in getattr(route, "methods", set())
    }
    assert served == {
        ("GET", BASE),
        ("GET", f"{BASE}/non-team-card"),
        ("POST", f"{BASE}/orgs/{{tenant}}/revoke"),
        ("POST", f"{BASE}/orgs/{{tenant}}/share-again"),
        ("POST", f"{BASE}/reports/{{draft_id}}/discard"),
        ("POST", f"{BASE}/notices/{{notice_id}}/shown"),
        ("POST", f"{BASE}/notice-due"),
    }
    # A grant is answered only by the typed guard_consent card; a report is never
    # confirmed from this plane. No path may even be NAMED like either.
    for _method, path in served:
        assert not any(
            word in path for word in ("accept", "approve", "grant", "confirm", "consent")
        )


def test_the_capability_is_advertised_and_the_relay_may_republish_it() -> None:
    assert feature_flags()["guard"] == 1
    # L2d: the phone relay serves its own twin at /api/guard*, so the key is no longer
    # desktop-only. The relay still withholds it unless the twin is mounted
    # (tests/unit/mobile/test_guard_relay.py pins both directions).
    assert "guard" not in DESKTOP_ONLY_FEATURES


@pytest.mark.asyncio
@pytest.mark.parametrize(("method", "path", "body"), _ALL)
async def test_every_route_is_bearer_gated(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_server: Any,
    method: str,
    path: str,
    body: Any,
) -> None:
    _team(monkeypatch)
    async with _http(root, monkeypatch) as client:
        for headers in ({"Authorization": ""}, {"Authorization": "Bearer " + "x" * 43}):
            response = await client.request(method, path, json=body, headers=headers)
            assert response.status_code == 401, (path, response.text)
    # Refused before any engine call: nothing was read from Radient, nothing was built.
    assert _calls(fixture_server) == 0
    assert not (root / "guard").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path", ["/orgs/ten_acme/accept", "/orgs/ten_acme/grant", "/reports/r/confirm"]
)
async def test_no_grant_or_report_confirm_route_exists(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any, path: str
) -> None:
    _team(monkeypatch)
    host = _host(root, fixture_server)
    host.offer(C.ConsentOffer.from_wire(_offer()))
    async with _http(root, monkeypatch) as client:
        response = await client.post(BASE + path, json={"scopes": ["decisions"]})
        assert response.status_code in (404, 405)
    assert host.prompt(TENANT, 1).state == "pending"  # type: ignore[union-attr]
    assert fixture_server.guard_requests() == []


# ---------------------------------------------------------------------------
# Non-Team: zero calls, zero files, 404 on every write
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_team_status_is_the_card_with_zero_requests_and_no_guard_file(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any
) -> None:
    _non_team(monkeypatch)
    async with _http(root, monkeypatch) as client:
        response = await client.get(BASE)
    assert response.status_code == 200, response.text
    body = _body(response)
    assert body["mode"] == "non_team" and body["orgs"] == [] and body["badge"] == ""
    assert body["non_team"] == {"trial_days": None, "org": None, "url": surface.team_url()}
    assert body["backup_days"] == 31
    # THE ASSERTION THIS SLICE EXISTS FOR: no request of any kind (so no /v1/guard/*) ...
    assert _calls(fixture_server) == 0
    # ... and not one Guard file built for an account Guard is not on.
    assert not (root / "guard").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(("method", "path", "body"), _WRITES)
async def test_non_team_writes_are_404_guard_unavailable_and_touch_nothing(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_server: Any,
    method: str,
    path: str,
    body: Any,
) -> None:
    _non_team(monkeypatch)
    async with _http(root, monkeypatch) as client:
        response = await client.request(method, path, json=body)
    assert response.status_code == 404, response.text
    assert _code(response) == "guard_unavailable"
    assert _calls(fixture_server) == 0
    assert not (root / "guard").exists()  # the cadence file in particular is never created


@pytest.mark.asyncio
async def test_team_without_a_raised_prompt_is_mode_none_and_writes_are_404(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any
) -> None:
    _team(monkeypatch)
    async with _http(root, monkeypatch) as client:
        status = await client.get(BASE)
        write = await client.post(f"{BASE}/orgs/{TENANT}/revoke")
    assert _body(status)["mode"] == "none" and _body(status)["non_team"] is None
    assert write.status_code == 404 and _code(write) == "guard_unavailable"
    assert not (root / "guard").exists() and _calls(fixture_server) == 0


@pytest.mark.asyncio
async def test_non_team_card_reads_eligibility_and_sanitises_the_org_name(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any
) -> None:
    _non_team(monkeypatch)
    monkeypatch.setattr(
        surface,
        "load_trial_eligibility",
        lambda _c: {"eligible": True, "trial_days": 14, "tenant_name": "Acme\n\x1b[31mCorp"},
    )
    async with _http(root, monkeypatch) as client:
        response = await client.get(f"{BASE}/non-team-card")
    assert _body(response) == {"trial_days": 14, "org": "Acme [31mCorp", "url": surface.team_url()}
    assert fixture_server.guard_requests() == []


@pytest.mark.asyncio
async def test_non_team_card_degrades_to_see_team_when_the_read_fails(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _non_team(monkeypatch)

    def boom(_c: Any) -> Any:
        raise OSError("offline")

    monkeypatch.setattr(surface, "load_trial_eligibility", boom)
    async with _http(root, monkeypatch) as client:
        response = await client.get(f"{BASE}/non-team-card")
    assert response.status_code == 200
    assert _body(response)["trial_days"] is None and _body(response)["org"] is None


@pytest.mark.asyncio
async def test_non_team_card_is_not_for_a_team_account(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _team(monkeypatch)
    async with _http(root, monkeypatch) as client:
        response = await client.get(f"{BASE}/non-team-card")
    assert response.status_code == 404 and _code(response) == "guard_not_applicable"


# ---------------------------------------------------------------------------
# Status: the TUI's states, predicates and badge, as data
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_waiting_prompt_is_display_only_and_names_its_approvals_card(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any
) -> None:
    _team(monkeypatch)
    host = _host(root, fixture_server)
    row = host.offer(C.ConsentOffer.from_wire(_offer())).row
    assert row is not None and row.approval_id
    async with _http(root, monkeypatch) as client:
        body = _body(await client.get(BASE))
        cards = await client.get("/v1/desktop/approvals")
    assert body["mode"] == "orgs" and body["badge"] == "waiting"
    org = body["orgs"][0]
    assert (org["tenant"], org["org"], org["state"]) == (TENANT, "Acme", "waiting")
    assert org["scopes_shared"] == [] and not org["can_revoke"] and not org["can_share_again"]
    assert org["prompt"]["approval_id"] == row.approval_id
    assert org["prompt"]["asked_scopes"] == ["decisions", "task_time"]
    # The id points at a real guard_consent card on the approvals plane...
    assert row.approval_id in {r["approval_id"] for r in _body(cards)["approvals"]}
    # ... and reading status answered nothing: the prompt is exactly as pending as before.
    assert host.prompt(TENANT, 1).state == "pending"  # type: ignore[union-attr]
    assert fixture_server.guard_requests() == []


@pytest.mark.asyncio
async def test_sharing_state_scopes_and_predicates_come_from_the_engine(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any
) -> None:
    _team(monkeypatch)
    _sharing(root, fixture_server)
    async with _http(root, monkeypatch) as client:
        body = _body(await client.get(BASE))
    org = body["orgs"][0]
    assert org["state"] == "sharing" and body["badge"] == "sharing"
    assert org["scopes_shared"] == ["decisions", "task_time"]
    state = surface.org_states(root)[0]
    assert (org["can_revoke"], org["can_share_again"]) == (
        surface.is_sharing(state),
        surface.is_stopped(state),
    )
    assert org["can_revoke"] and not org["can_share_again"]
    assert [n["kind"] for n in body["notices"]] == ["grant"]
    assert body["can_answer"] is True and body["this_computer"]


@pytest.mark.asyncio
async def test_a_pending_widening_is_sharing_pending_never_paused(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any
) -> None:
    _team(monkeypatch)
    host = _sharing(root, fixture_server)
    host.offer(
        C.ConsentOffer.from_wire(
            _offer(
                collection_rev=2,
                scopes=["decisions", "task_time", "review_cycles"],
                added_scopes=["review_cycles"],
                reason="widened",
            )
        )
    )
    async with _http(root, monkeypatch) as client:
        body = _body(await client.get(BASE))
    org = body["orgs"][0]
    assert org["state"] == "sharing_pending" and body["badge"] == "sharing_pending"
    assert org["prompt"]["widening"] is True and org["can_revoke"] is True
    assert org["scopes_shared"] == ["decisions", "task_time"]  # what is shared NOW


@pytest.mark.asyncio
async def test_a_damaged_store_is_mode_unavailable_not_an_exception(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _team(monkeypatch)
    path = C.guard_consent_path(root)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"this is not sqlite" * 100)
    async with _http(root, monkeypatch) as client:
        status = await client.get(BASE)
        write = await client.post(f"{BASE}/orgs/{TENANT}/revoke")
    assert status.status_code == 200 and _body(status)["mode"] == "unavailable"
    assert write.status_code == 409 and _code(write) == "consent_store_damaged"
    assert path.read_bytes().startswith(b"this is not sqlite")  # never recreated


# ---------------------------------------------------------------------------
# Revoke
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_revoke_sends_one_jwt_only_request_and_the_org_reads_stopped(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any
) -> None:
    _team(monkeypatch)
    host = _sharing(root, fixture_server)
    before = len(fixture_server.guard_requests())
    async with _http(root, monkeypatch) as client:
        response = await client.post(f"{BASE}/orgs/{TENANT}/revoke")
        again = await client.post(f"{BASE}/orgs/{TENANT}/revoke")
        status = _body(await client.get(BASE))
    assert response.status_code == 200, response.text
    result = _body(response)
    assert result["status"] == "revoked" and result["code"] == ""
    assert (result["org"]["state"], result["org"]["reason"]) == ("paused", "scope_revoked")
    sent = fixture_server.guard_requests()[before:]
    assert [(r["method"], r["path"], r["body"]) for r in sent] == [
        ("POST", "/v1/guard/consents/revoke", {"tenant": TENANT})
    ]
    assert host.sticky(TENANT) == "revoked"
    # A second stop is a refusal, not a second request.
    assert again.status_code == 409 and _code(again) == "guard_not_sharing"
    assert len(fixture_server.guard_requests()) == before + 1
    assert status["orgs"][0]["can_share_again"] is True and status["orgs"][0]["can_revoke"] is False


@pytest.mark.asyncio
async def test_revoke_that_cannot_reach_radient_is_a_200_revoke_pending_and_not_sharing(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any
) -> None:
    _team(monkeypatch)
    host = _sharing(root, fixture_server)
    fixture_server.revoke_reply = (503, {"error": "refused", "code": "unavailable"})
    async with _http(root, monkeypatch) as client:
        response = await client.post(f"{BASE}/orgs/{TENANT}/revoke")
    assert response.status_code == 200, response.text
    result = _body(response)
    assert result["status"] == "revoke_pending" and result["code"]
    assert (result["org"]["state"], result["org"]["reason"]) == ("paused", "revoke_pending")
    # The local stop is committed: nothing more may be collected, and no second stop is offered.
    assert host.consent(TENANT)["state"] == "revoke_pending"  # type: ignore[index]
    assert result["org"]["can_revoke"] is False and result["org"]["can_share_again"] is False
    # The next admission point retries it, exactly as for the TUI.
    fixture_server.revoke_reply = (200, {"result": {}})
    host.retry_pending_revokes()
    assert host.consent(TENANT)["state"] == "revoked"  # type: ignore[index]


@pytest.mark.asyncio
async def test_revoke_refusals_unknown_org_not_sharing_and_malformed_id(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any
) -> None:
    _team(monkeypatch)
    host = _host(root, fixture_server)
    host.offer(C.ConsentOffer.from_wire(_offer()))  # waiting: nothing is shared yet
    async with _http(root, monkeypatch) as client:
        unknown = await client.post(f"{BASE}/orgs/ten_other/revoke")
        waiting = await client.post(f"{BASE}/orgs/{TENANT}/revoke")
        malformed = await client.post(f"{BASE}/orgs/{'a' * 300}/revoke")
        spaced = await client.post(f"{BASE}/orgs/ten%20acme/revoke")
    assert (unknown.status_code, _code(unknown)) == (404, "guard_org_unknown")
    assert (waiting.status_code, _code(waiting)) == (409, "guard_not_sharing")
    assert malformed.status_code == 422 and spaced.status_code == 422
    assert fixture_server.guard_requests() == []


# ---------------------------------------------------------------------------
# Share again
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_share_again_lifts_the_decline_and_the_prompt_returns_ungranted(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any
) -> None:
    _team(monkeypatch)
    host = _host(root, fixture_server)
    host.offer(C.ConsentOffer.from_wire(_offer()))
    host.decline(TENANT, 1, surface="tui")
    assert host.sticky(TENANT) == "declined"
    # The org's offer, as the server restates it at the next admission point.
    fixture_server.status = {"status": "consent_required", "prompt": _offer()}
    async with _http(root, monkeypatch) as client:
        before = _body(await client.get(BASE))["orgs"][0]
        response = await client.post(f"{BASE}/orgs/{TENANT}/share-again")
    assert (before["state"], before["can_share_again"]) == ("not_sharing", True)
    assert response.status_code == 200, response.text
    org = _body(response)["org"]
    assert host.sticky(TENANT) is None
    assert org["state"] == "waiting" and org["prompt"] is not None
    # ONE read of the offer (the admission point) and NOT ONE grant: a grant is the signed card.
    paths = [(r["method"], r["path"]) for r in fixture_server.guard_requests()]
    assert paths == [("GET", "/v1/guard/status")]
    assert host.consent(TENANT) is None


@pytest.mark.asyncio
async def test_share_again_is_refused_where_the_tui_hides_it_and_changes_nothing(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any
) -> None:
    _team(monkeypatch)
    host = _host(root, fixture_server)
    host.offer(C.ConsentOffer.from_wire(_offer()))
    host.decline(TENANT, 1, surface="tui")
    # The TUI's person-only gate: a backend started inside an agent's shell / console pane.
    monkeypatch.setenv("LOCAL_OPERATOR_AGENT_SHELL", "1")
    assert surface.host_can_answer() is False
    async with _http(root, monkeypatch) as client:
        status = _body(await client.get(BASE))
        response = await client.post(f"{BASE}/orgs/{TENANT}/share-again")
    assert status["can_answer"] is False and status["orgs"][0]["can_share_again"] is False
    assert response.status_code == 409 and _code(response) == "guard_answer_not_admitted"
    assert host.sticky(TENANT) == "declined"  # the sticky "no" stands
    assert fixture_server.guard_requests() == []
    # NEGATIVE CONTROL: the identical request from a person's backend succeeds.
    monkeypatch.delenv("LOCAL_OPERATOR_AGENT_SHELL")
    async with _http(root, monkeypatch) as client:
        control = await client.post(f"{BASE}/orgs/{TENANT}/share-again")
    assert control.status_code == 200 and host.sticky(TENANT) is None


@pytest.mark.asyncio
async def test_console_pane_marker_also_refuses_share_again(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any
) -> None:
    _team(monkeypatch)
    host = _host(root, fixture_server)
    host.offer(C.ConsentOffer.from_wire(_offer()))
    host.decline(TENANT, 1, surface="tui")
    monkeypatch.setenv("LOCAL_OPERATOR_CONSOLE_SURFACE", "1")
    async with _http(root, monkeypatch) as client:
        response = await client.post(f"{BASE}/orgs/{TENANT}/share-again")
    assert _code(response) == "guard_answer_not_admitted" and host.sticky(TENANT) == "declined"


@pytest.mark.asyncio
async def test_share_again_for_an_org_that_is_not_stopped_is_409(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any
) -> None:
    _team(monkeypatch)
    _sharing(root, fixture_server)
    async with _http(root, monkeypatch) as client:
        response = await client.post(f"{BASE}/orgs/{TENANT}/share-again")
        unknown = await client.post(f"{BASE}/orgs/ten_other/share-again")
    assert response.status_code == 409 and _code(response) == "guard_not_stopped"
    assert unknown.status_code == 404 and _code(unknown) == "guard_org_unknown"


# ---------------------------------------------------------------------------
# Reports and receipts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_report_draft_is_listed_and_discard_is_the_only_answer(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any
) -> None:
    _team(monkeypatch)
    _sharing(root, fixture_server)
    draft, _new = GuardReports(root).draft(TENANT, "credential_exposure")
    async with _http(root, monkeypatch) as client:
        listed = _body(await client.get(BASE))
        discarded = await client.post(f"{BASE}/reports/{draft.id}/discard")
        again = await client.post(f"{BASE}/reports/{draft.id}/discard")
        missing = await client.post(f"{BASE}/reports/rep_nope/discard")
        after = _body(await client.get(BASE))
    assert listed["badge"] == "report_waiting"
    assert [(r["id"], r["kind"], r["org"]) for r in listed["reports"]] == [
        (draft.id, "credential_exposure", "Acme")
    ]
    assert _body(discarded) == {"id": draft.id, "state": "discarded"}
    assert again.status_code == 404 and _code(again) == "guard_report_unknown"
    assert missing.status_code == 404
    assert after["reports"] == [] and after["badge"] == "sharing"
    saved = GuardReports(root).get(draft.id)
    assert saved is not None and (saved.state, saved.decided_via) == ("discarded", "desktop")


@pytest.mark.asyncio
async def test_receipts_are_listed_once_marked_shown_and_a_stop_carries_none(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any
) -> None:
    _team(monkeypatch)
    host = _sharing(root, fixture_server)
    host.revoke(TENANT)  # writes a ``revoke`` notice: a stop is not a receipt
    grant_id = next(n["id"] for n in host.notices() if n["kind"] == "grant")
    revoke_id = next(n["id"] for n in host.notices() if n["kind"] == "revoke")
    async with _http(root, monkeypatch) as client:
        # S-2: a real, unshown receipt id that THIS daemon has not listed is not accepted, so a
        # bearer holder cannot dismiss "Not you? Stop sharing" by guessing small integers.
        unlisted = await client.post(f"{BASE}/notices/{grant_id}/shown")
        still_unshown = [n["id"] for n in host.notices(unshown_only=True)]
        first = _body(await client.get(BASE))
        grant = first["notices"][0]
        refused = await client.post(f"{BASE}/notices/{revoke_id}/shown")
        guessed = await client.post(f"{BASE}/notices/{grant['id'] + 1000}/shown")
        shown = await client.post(f"{BASE}/notices/{grant['id']}/shown")
        second = _body(await client.get(BASE))
        replay = await client.post(f"{BASE}/notices/{grant['id']}/shown")
    assert unlisted.status_code == 404 and _code(unlisted) == "guard_notice_unknown"
    assert grant_id in still_unshown, "the refused call must not have flipped the receipt"
    assert [n["kind"] for n in first["notices"]] == ["grant"]
    assert grant["org"] == "Acme" and grant["scopes"] == ["decisions", "task_time"]
    assert refused.status_code == 404 and _code(refused) == "guard_notice_unknown"
    assert guessed.status_code == 404 and _code(guessed) == "guard_notice_unknown"
    assert _body(shown) == {"id": grant["id"], "shown": True}
    assert second["notices"] == []
    # Idempotent: the same acknowledgement again is the same answer, not an error.
    assert replay.status_code == 200 and _body(replay) == {"id": grant["id"], "shown": True}
    assert [n["kind"] for n in host.notices(unshown_only=True)] == ["revoke"]  # not a receipt


@pytest.mark.asyncio
async def test_wire_timestamps_are_floored_to_the_minute_like_the_tui(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any
) -> None:
    """S-4: a renderer cannot show (and a relay twin cannot forward) finer timing than the TUI."""

    _team(monkeypatch)
    _sharing(root, fixture_server)
    draft, _created = GuardReports(root).draft(TENANT, "other")
    async with _http(root, monkeypatch) as client:
        body = _body(await client.get(BASE))
    notice_at = body["notices"][0]["at"]
    created = body["reports"][0]["created_at"]
    assert notice_at > 0 and notice_at % 60 == 0
    assert created > 0 and created % 60 == 0
    # The wire value is the engine's value floored (never rounded up into the future).
    assert created == surface.to_minute(draft.created_at) <= draft.created_at
    assert surface.to_minute(1795817427.449125) == 1795817400.0
    # A non-numeric epoch is a wire-boundary guard, so feed it as an untyped value.
    junk: Any = "junk"
    assert surface.to_minute(junk) == 0.0


@pytest.mark.asyncio
async def test_notice_due_is_once_per_day_and_bound_to_the_desktop_surface(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any
) -> None:
    _team(monkeypatch)
    _host(root, fixture_server).offer(C.ConsentOffer.from_wire(_offer()))
    async with _http(root, monkeypatch) as client:
        first = await client.post(f"{BASE}/notice-due", json={"surface": "desktop"})
        second = await client.post(f"{BASE}/notice-due", json={"surface": "desktop"})
        # S-3: the plane has one bearer and no caller identity, so another surface's day
        # cannot be consumed from here at all.
        phone = await client.post(f"{BASE}/notice-due", json={"surface": "mobile"})
        bad = await client.post(f"{BASE}/notice-due", json={"surface": "tui"})
        extra = await client.post(f"{BASE}/notice-due", json={"surface": "desktop", "scopes": []})
    assert (_body(first)["due"], _body(second)["due"]) == (True, False)
    assert phone.status_code == bad.status_code == extra.status_code == 422
    assert surface.should_show_notice(root, surface="mobile") is True  # the day was never spent


@pytest.mark.asyncio
async def test_the_routes_read_the_start_time_capture_from_app_state(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any
) -> None:
    """S-1: ``app.state.guard_host_can_answer`` (set by the lifespan) wins over the environment."""

    _team(monkeypatch)
    host = _host(root, fixture_server)
    host.offer(C.ConsentOffer.from_wire(_offer()))
    host.decline(TENANT, 1, surface="tui")
    assert surface.host_can_answer() is True  # a clean environment, as a served daemon has
    async with _http(root, monkeypatch, can_answer=False) as client:
        status = _body(await client.get(BASE))
        refused = await client.post(f"{BASE}/orgs/{TENANT}/share-again")
    assert status["can_answer"] is False and status["orgs"][0]["can_share_again"] is False
    assert refused.status_code == 409 and _code(refused) == "guard_answer_not_admitted"
    assert host.sticky(TENANT) == "declined"


# ---------------------------------------------------------------------------
# One engine, one measured X
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_desktop_badge_and_the_tui_badge_are_one_function(
    root: Path, monkeypatch: pytest.MonkeyPatch, fixture_server: Any
) -> None:
    from local_operator.tui.widgets import guard_view

    _team(monkeypatch)
    _sharing(root, fixture_server)
    GuardReports(root).draft(TENANT, "other")
    states = surface.org_states(root)
    async with _http(root, monkeypatch) as client:
        body = _body(await client.get(BASE))
    assert body["badge"] == surface.badge_kind(states, reports_waiting=1) == "report_waiting"
    # The TUI's sentence is derived from the same enum (a draft waiting, then plain sharing).
    assert guard_view.badge_text(states, reports_waiting=1) != guard_view.badge_text(states)
    assert surface.badge_kind(states) == "sharing"


def test_the_measured_backup_days_are_the_one_value_both_surfaces_read() -> None:
    from local_operator.i18n.keys import tui_guard as keys
    from local_operator.i18n.text import tr

    assert surface.BACKUP_DAYS == 31
    sentence = tr("tui.guard")(keys.revoke_body_trace(backup_days=str(surface.BACKUP_DAYS)))
    assert "31 days" in sentence and "{" not in sentence
    # Measuring X does NOT lift the engine's refusal of the scopes whose revoke copy needs it.
    assert C.UNMEASURED_SCOPES == {"trace_detail", "admin_share"}
