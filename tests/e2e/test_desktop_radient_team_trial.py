"""``team_trial.*`` on the Radient proxy, over real loopback HTTP.

WHY THIS EXISTS. The trial start is a mutation with legal weight -- it records
the user's Terms acceptance and renames their home organization -- so the
properties that matter are ones only the real route shows: the upstream sees
EXACTLY the designed body with ``source`` pinned to ``lop`` (never the
renderer's), a replay under the same request id is answered from the receipt
without a second upstream call, and each designed refusal arrives at the
renderer under its own ``code`` with its ``details`` intact instead of being
folded into ``radient_credential_refused`` ("sign in again"), which is the wrong
remedy for every one of them.

Nothing contacts Radient: the host is a stub on loopback and the credential is
fabricated in the test's own isolated config dir.
"""

import json
from typing import Any

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from tests.e2e.test_desktop_controls import request_id
from tests.e2e.test_desktop_radient_resend import BEARER, ResendStub, proxy

pytestmark = [pytest.mark.e2e, pytest.mark.asyncio]

TENANT = "ten_home1"
TERMS = {"terms_version": "2026-07-29", "terms_accepted": True}
TRIAL = {
    "state": "active",
    "trial_start": "2026-11-03T14:05:00Z",
    "trial_end": "2026-12-03T14:05:00Z",
    "days_left": 30,
}


class TrialStub(ResendStub):
    """The three designed routes; records every call with its JSON body.

    A ``ResendStub`` only so the shared ``proxy`` fixture (typed on it) serves
    it unchanged: the fixture reads nothing but ``.app``, which is replaced.
    """

    def __init__(self, status: int = 201, body: dict[str, Any] | None = None) -> None:
        self.status = status
        self.body = (
            body if body is not None else {"trial": TRIAL, "tenant": {"id": TENANT, "name": "Acme"}}
        )
        self.requests: list[tuple[str, str, str, Any]] = []
        self.app = FastAPI()

        @self.app.get("/v1/team-trial/eligibility")
        async def eligibility(request: Request) -> JSONResponse:
            self.requests.append(
                ("GET", request.url.path, request.headers.get("authorization", ""), None)
            )
            return JSONResponse(
                {
                    "eligible": True,
                    "reason": None,
                    "tenant_id": TENANT,
                    "tenant_name": "Ada's Organization",
                    "trial_days": 30,
                    "would_end_at": TRIAL["trial_end"],
                    "trial": None,
                    "terms_version": "2026-07-29",
                    "terms_url": "https://radienthq.com/terms",
                    "subscribe_url": "https://console.radienthq.com/dashboard/team",
                }
            )

        @self.app.api_route("/v1/tenants/{tenant}/team-trial", methods=["GET", "POST"])
        async def trial(tenant: str, request: Request) -> JSONResponse:
            raw = await request.body()
            self.requests.append(
                (
                    request.method,
                    request.url.path,
                    request.headers.get("authorization", ""),
                    json.loads(raw) if raw else None,
                )
            )
            return JSONResponse(self.body, status_code=self.status)


def _start(payload: dict[str, Any] | None = None, rid: str | None = None) -> dict[str, Any]:
    return {
        "operation": "team_trial.start",
        "tenant_id": TENANT,
        "request_id": rid or request_id(),
        "payload": payload or {**TERMS, "org_name": "Acme"},
    }


async def test_eligibility_reads_the_home_tenant_with_the_stored_bearer(
    headless_tui_env, monkeypatch
) -> None:
    stub = TrialStub()
    async with proxy(headless_tui_env, monkeypatch, stub) as client:
        response = await client.post(
            "/v1/desktop/radient", json={"operation": "team_trial.eligibility"}
        )
        assert response.status_code == 200, response.text
        data = response.json()["result"]["data"]
        assert data["eligible"] is True and data["trial"] is None
        assert stub.requests == [("GET", "/v1/team-trial/eligibility", "Bearer " + BEARER, None)]


async def test_start_sends_the_designed_body_once_and_replays_from_the_receipt(
    headless_tui_env, monkeypatch
) -> None:
    stub = TrialStub()
    async with proxy(headless_tui_env, monkeypatch, stub) as client:
        body = _start()
        response = await client.post("/v1/desktop/radient", json=body)
        assert response.status_code == 200, response.text
        assert response.json()["result"]["data"]["trial"]["trial_end"] == TRIAL["trial_end"]
        assert stub.requests == [
            (
                "POST",
                f"/v1/tenants/{TENANT}/team-trial",
                "Bearer " + BEARER,
                {**TERMS, "org_name": "Acme", "source": "lop"},
            )
        ]
        again = await client.post("/v1/desktop/radient", json=body)
        assert again.status_code == 200
        assert len(stub.requests) == 1  # a lost response never re-submits consent


async def test_status_reads_the_trial_block(headless_tui_env, monkeypatch) -> None:
    stub = TrialStub(
        200,
        {
            "state": "ended",
            "trial_end": TRIAL["trial_end"],
            "ended_at": TRIAL["trial_end"],
            "days_left": 0,
            "entitlement_source": "none",
        },
    )
    async with proxy(headless_tui_env, monkeypatch, stub) as client:
        response = await client.post(
            "/v1/desktop/radient", json={"operation": "team_trial.status", "tenant_id": TENANT}
        )
        assert response.status_code == 200, response.text
        assert response.json()["result"]["data"]["state"] == "ended"
        assert stub.requests[0][:2] == ("GET", f"/v1/tenants/{TENANT}/team-trial")


@pytest.mark.parametrize(
    "status,envelope,http,code,detail_key",
    [
        (
            403,
            {"code": "trial_not_eligible", "details": {"reason": "trial_used"}},
            403,
            "trial_not_eligible",
            "reason",
        ),
        (
            409,
            {"code": "terms_outdated", "details": {"terms_version": "2026-12-01"}},
            409,
            "terms_outdated",
            "terms_version",
        ),
        (400, {"code": "terms_required"}, 400, "terms_required", None),
        # The bug the scoped carry exists for: NOT radient_credential_refused.
        (429, {"code": "rate_limited"}, 429, "rate_limited", None),
        (403, {"code": "not_a_member"}, 403, "not_a_member", None),
        (
            403,
            {
                "code": "team_trial_ended",
                "details": {
                    "trial_ended_at": "2026-12-03T14:05:00Z",
                    "subscribe_url": "https://console.radienthq.com/dashboard/team",
                },
            },
            403,
            "team_trial_ended",
            "trial_ended_at",
        ),
        # An uncoded 401 keeps its sign-in-again meaning.
        (401, {"error": "unauthorized"}, 401, "radient_credential_refused", None),
    ],
)
async def test_start_refusals_reach_the_renderer_under_their_own_code(
    headless_tui_env, monkeypatch, status, envelope, http, code, detail_key
) -> None:
    stub = TrialStub(status, envelope)
    async with proxy(headless_tui_env, monkeypatch, stub) as client:
        response = await client.post("/v1/desktop/radient", json=_start())
        assert response.status_code == http, response.text
        detail = response.json()["detail"]
        assert detail["code"] == code
        assert detail["details"]["upstream_status"] == status
        if detail_key:
            assert detail["details"][detail_key] == envelope["details"][detail_key]
        assert len(stub.requests) == 1


@pytest.mark.parametrize(
    "payload",
    [{"terms_version": "2026-07-29", "terms_accepted": False}, {**TERMS, "source": "site"}],
)
async def test_an_invalid_start_is_refused_before_any_upstream_call(
    headless_tui_env, monkeypatch, payload
) -> None:
    stub = TrialStub()
    async with proxy(headless_tui_env, monkeypatch, stub) as client:
        response = await client.post("/v1/desktop/radient", json=_start(payload))
        assert response.status_code == 422, response.text
        assert stub.requests == []
