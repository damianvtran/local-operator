"""``signup.resend`` on the Radient proxy, over real loopback HTTP.

WHY THIS EXISTS. The op is the one on this transport that SENDS MAIL, and its
upstream answers (200/429/409/503/401) mean different things to a user than the
generic reading gives them: ``_upstream_refusal`` would render the upstream's
429 -- "you asked for a link a moment ago" -- as ``radient_credential_refused``,
i.e. "sign in again". Each case drives the real route against a labelled stub
host and asserts three things the unit tests cannot: the response's ``code``,
that EXACTLY ONE upstream call was made (a mail-sending op must never retry
or fan out), and that the stored bearer -- not the app's own token -- was what
the stub saw.

Nothing contacts Radient: the host is a stub on loopback and the credential is
fabricated in the test's own isolated config dir.
"""

import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from local_operator.server.app import app
from tests.e2e.test_desktop_controls import request_id
from tests.e2e.test_desktop_radient import serve

pytestmark = [pytest.mark.e2e, pytest.mark.asyncio]

BEARER = "resend-access-token-fixture"


class ResendStub:
    """A fake ``POST /v1/auth/signup/resend`` that records every call."""

    def __init__(self, status: int, error: str | None = None) -> None:
        self.status = status
        self.error = error
        #: (method, path, authorization header) per request that arrived.
        self.calls: list[tuple[str, str, str]] = []
        self.app = FastAPI()

        @self.app.post("/v1/auth/signup/resend")
        async def resend(request: Request) -> JSONResponse:
            auth = request.headers.get("authorization", "")
            self.calls.append(("POST", "/v1/auth/signup/resend", auth))
            body: dict[str, object] = {"status": self.status, "msg": "stub"}
            if self.error:
                body["error"] = self.error
            return JSONResponse(body, status_code=self.status)


@asynccontextmanager
async def proxy(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, stub: ResendStub
) -> AsyncIterator[httpx.AsyncClient]:
    from local_operator.server.routes import desktop_radient

    token = secrets.token_hex(32)
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", token)
    (config_dir / "config.yml").write_text("version: 0.0.0\nvalues: {}\n")
    async with serve(stub.app) as upstream_url, serve(app) as desktop_url:
        monkeypatch.setattr(desktop_radient, "base_url", lambda: upstream_url + "/v1")
        async with httpx.AsyncClient(
            base_url=desktop_url, timeout=30, headers={"Authorization": "Bearer " + token}
        ) as client:
            await client.get("/v1/auth/status")  # materialises app.state.desktop_auth
            app.state.desktop_auth.store.upsert_credential(
                "radient",
                {
                    "type": "oauth",
                    "access": BEARER,
                    "refresh": "refresh-fixture",
                    "expires": int(time.time() * 1000) + 3_600_000,
                },
            )
            yield client


def _request(rid: str | None = None) -> dict[str, str]:
    return {"operation": "signup.resend", "request_id": rid or request_id()}


async def test_resend_success_spends_the_stored_bearer_exactly_once(
    headless_tui_env, monkeypatch
) -> None:
    stub = ResendStub(200)
    async with proxy(headless_tui_env, monkeypatch, stub) as client:
        body = _request()
        response = await client.post("/v1/desktop/radient", json=body)
        assert response.status_code == 200, response.text
        assert response.json()["result"]["data"]["status"] == 200
        assert stub.calls == [("POST", "/v1/auth/signup/resend", "Bearer " + BEARER)]

        # A lost response retried under the SAME request id replays the receipt:
        # no second email.
        again = await client.post("/v1/desktop/radient", json=body)
        assert again.status_code == 200
        assert len(stub.calls) == 1


@pytest.mark.parametrize(
    "status,error,http,code",
    [
        # THE BUG THIS OP'S OWN MAPPING EXISTS FOR: not radient_credential_refused.
        (429, "rate_limited", 429, "signup_resend_rate_limited"),
        # Already verified / disposable domain / nothing ever issued: no grant.
        (409, "nothing_to_resend", 409, "signup_resend_nothing_to_resend"),
        # Grant claiming switched off upstream: an outage, retry later. 503 is not
        # in the proxy's passthrough set, so it reads as the generic 502 outage.
        (503, "grant_unavailable", 502, "radient_upstream_failed"),
        (500, "internal_error", 502, "radient_upstream_failed"),
        # A refused credential keeps its generic sign-in-again meaning.
        (401, None, 401, "radient_credential_refused"),
        (403, None, 403, "radient_credential_refused"),
    ],
)
async def test_resend_upstream_answers_map_to_distinct_codes(
    headless_tui_env, monkeypatch, status, error, http, code
) -> None:
    stub = ResendStub(status, error)
    async with proxy(headless_tui_env, monkeypatch, stub) as client:
        response = await client.post("/v1/desktop/radient", json=_request())
        assert response.status_code == http, response.text
        detail = response.json()["detail"]
        assert detail["code"] == code
        assert detail["details"]["upstream_status"] == status
        # Exactly one call, with the stored bearer; nothing retried.
        assert stub.calls == [("POST", "/v1/auth/signup/resend", "Bearer " + BEARER)]


async def test_resend_without_a_request_id_is_refused_before_any_upstream_call(
    headless_tui_env, monkeypatch
) -> None:
    stub = ResendStub(200)
    async with proxy(headless_tui_env, monkeypatch, stub) as client:
        response = await client.post("/v1/desktop/radient", json={"operation": "signup.resend"})
        assert response.status_code == 422
        assert stub.calls == []


async def test_resend_refuses_a_caller_supplied_address(headless_tui_env, monkeypatch) -> None:
    """The upstream reads the address from the account; the proxy must not even
    let a caller try to name one."""
    stub = ResendStub(200)
    async with proxy(headless_tui_env, monkeypatch, stub) as client:
        response = await client.post(
            "/v1/desktop/radient",
            json={**_request(), "payload": {"email": "someone@example.com"}},
        )
        assert response.status_code == 422
        assert stub.calls == []


async def test_resend_with_no_stored_credential_is_the_no_credential_class(
    headless_tui_env, monkeypatch
) -> None:
    stub = ResendStub(200)
    async with proxy(headless_tui_env, monkeypatch, stub) as client:
        app.state.desktop_auth.store.delete_credential(
            app.state.desktop_auth.store.list_credentials("radient")[0].id
        )
        response = await client.post("/v1/desktop/radient", json=_request())
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "radient_no_credential"
        assert stub.calls == []
