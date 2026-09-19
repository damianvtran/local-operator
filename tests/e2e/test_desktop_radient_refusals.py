"""What a refused or stale Radient credential becomes at a client, over real HTTP.

The proxy's failure classes are the whole point of this module: a caller has to
be able to tell "this account's Radient sign-in was refused" (sign in again)
from "this app cannot authenticate to its own server" (restart or re-pair it)
from "Radient is unhappy" (retry), and today only the batch op says which. The
second half is that a bearer the store already considers due for a refresh is
not spent at all: ``AuthStore._ensure_oauth_fresh``'s last resort hands the
caller the stored row when a PEER holds the refresh lease, and relaying that
token can only produce Radient's own 401 — the failure that names no remedy.

Nothing here contacts Radient: the host is a labelled stub on loopback and the
credential is fabricated in the test's own isolated config dir.
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

from local_operator.providers.auth_store import AuthStore
from local_operator.server.app import app
from tests.e2e.test_desktop_radient import serve

pytestmark = [pytest.mark.e2e, pytest.mark.asyncio]

STALE = "stale-access-token-fixture"
FRESH = "fresh-access-token-fixture"
REFRESH = "refresh-token-fixture"
#: The sentence a dead grant and a missing sign-in share, because they share the
#: remedy: the words ``errors``-style copy has always used for the 409.
SIGN_IN = "Sign in to Radient to access your account"


class Stub:
    """A fake Radient: /token, /me, /agents, /prices, and a log of what arrived.

    ``/agents`` answers 200 for the SAME expired bearer ``/me`` refuses under,
    which is what the operator's install did on 2026-09-19 — the reason a stale
    bearer was worth spending at all, and why "serve it and see" was tempting.
    """

    def __init__(self, *, refresh: str = "ok", me_status: int = 200) -> None:
        self.refresh = refresh
        self.me_status = me_status
        #: (method, path, bearer-or-"" ) for every request that reached the stub.
        self.calls: list[tuple[str, str, str]] = []
        self.app = self._build()

    def _build(self) -> FastAPI:
        fake = FastAPI()

        def bearer(request: Request) -> str:
            header = request.headers.get("authorization", "")
            return header.removeprefix("Bearer ") if header else "<none>"

        @fake.post("/token")
        async def token(request: Request) -> JSONResponse:
            body = await request.json()
            self.calls.append(("POST", "/token", ""))
            if self.refresh == "invalid_grant":
                return JSONResponse({"error": "invalid_grant"}, status_code=400)
            if self.refresh == "500":
                return JSONResponse({"error": "server_error"}, status_code=500)
            assert body["grant_type"] == "refresh_token"
            return JSONResponse(
                {"access_token": FRESH, "refresh_token": REFRESH, "expires_in": 3600}
            )

        @fake.get("/v1/me")
        async def me(request: Request) -> JSONResponse:
            self.calls.append(("GET", "/v1/me", bearer(request)))
            if self.me_status != 200:
                return JSONResponse({"detail": "refused"}, status_code=self.me_status)
            return JSONResponse({"status": 200, "result": {"email": "fixture@example.com"}})

        @fake.get("/v1/agents")
        async def agents(request: Request) -> JSONResponse:
            self.calls.append(("GET", "/v1/agents", bearer(request)))
            return JSONResponse({"status": 200, "result": {"agents": [], "total": 21}})

        @fake.get("/v1/prices")
        async def prices(request: Request) -> JSONResponse:
            self.calls.append(("GET", "/v1/prices", bearer(request)))
            return JSONResponse({"status": 200, "result": {"tiers": []}})

        return fake

    def paths(self) -> list[str]:
        """Every request that reached upstream, in order, with the bearer it carried."""
        return [f"{method} {path} {bearer or '<none>'}" for method, path, bearer in self.calls]

    def spent(self) -> list[str]:
        """The bearers that actually reached upstream, in order — "<none>" excluded."""
        return [bearer for _method, _path, bearer in self.calls if bearer and bearer != "<none>"]


@asynccontextmanager
async def proxy(
    config_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    refresh: str = "ok",
    me_status: int = 200,
    seed: str = "expired",
    lease: bool = False,
) -> AsyncIterator[tuple[httpx.AsyncClient, Stub]]:
    """A drivable daemon whose Radient host is the stub, with one seeded credential.

    ``lease`` reproduces the measured live state: a SECOND store — a second
    process, from this one's point of view — holds the refresh lease while the
    stored access token is already expired.
    """
    from local_operator.providers.oauth import radient as oauth
    from local_operator.server.routes import desktop_radient

    token = secrets.token_hex(32)
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", token)
    (config_dir / "config.yml").write_text("version: 0.0.0\nvalues: {}\n")
    stub = Stub(refresh=refresh, me_status=me_status)
    peer: AuthStore | None = None
    async with serve(stub.app) as upstream_url, serve(app) as desktop_url:
        monkeypatch.setattr(desktop_radient, "base_url", lambda: upstream_url + "/v1")
        monkeypatch.setattr(oauth, "TOKEN_URL", upstream_url + "/token")
        async with httpx.AsyncClient(
            base_url=desktop_url, timeout=30, headers={"Authorization": "Bearer " + token}
        ) as client:
            if seed == "expired":
                # `/v1/auth/status` is what materialises `app.state.desktop_auth`;
                # the store it builds is the same one the proxy resolves through.
                await client.get("/v1/auth/status")
                store = app.state.desktop_auth.store
                row = store.upsert_credential(
                    "radient",
                    {
                        "type": "oauth",
                        "access": STALE,
                        "refresh": REFRESH,
                        "expires": int(time.time() * 1000) - 3_600_000,
                    },
                )
                if lease:
                    peer = AuthStore(config_dir / "auth.db")
                    assert peer._try_refresh_lease(row.id)
            try:
                yield client, stub
            finally:
                if peer is not None:
                    peer.close()


async def test_a_dead_grant_answers_the_sign_in_class_and_spends_nothing(
    headless_tui_env, monkeypatch
) -> None:
    """invalid_grant: only a re-login revives it, so that is what the client is told."""
    async with proxy(headless_tui_env, monkeypatch, refresh="invalid_grant") as (client, stub):
        response = await client.post("/v1/desktop/radient", json={"operation": "account"})
        assert response.status_code == 401
        assert response.json()["detail"] == {
            "code": "radient_credential_refused",
            "message": SIGN_IN,
            "details": {"reason": "grant_invalid"},
        }
        # Nothing was spent on a credential the store had already been told is dead.
        assert "GET /v1/me" not in stub.paths()

        batch = await client.post(
            "/v1/desktop/radient",
            json={"operation": "agents.statuses", "query": {"agent_ids": "a1"}},
        )
        assert batch.status_code == 401
        assert batch.json()["detail"]["code"] == "radient_credential_refused"


async def test_a_transient_refresh_failure_answers_the_retry_class(
    headless_tui_env, monkeypatch
) -> None:
    """A 5xx from the token endpoint is not the account's fault: retry, do not re-sign-in."""
    async with proxy(headless_tui_env, monkeypatch, refresh="500") as (client, stub):
        response = await client.post("/v1/desktop/radient", json={"operation": "account"})
        assert response.status_code == 502
        assert response.json()["detail"]["code"] == "radient_upstream_failed"
        assert response.json()["detail"]["details"]["reason"] == "credential_unavailable"
        assert "GET /v1/me" not in stub.paths()


async def test_a_peer_leased_refresh_never_spends_the_stale_bearer(
    headless_tui_env, monkeypatch
) -> None:
    """THE MEASURED DEFECT: a stale bearer the store knows is due must not go upstream.

    Before this change the account op sent it and relayed Radient's 401, while
    `agents.list` answered 200 off the same token because that upstream path does
    not check expiry — one credential, two verdicts, and a client that could not
    tell either from a re-pair problem.
    """
    async with proxy(headless_tui_env, monkeypatch, lease=True) as (client, stub):
        for body in (
            {"operation": "account"},
            {"operation": "agents.list", "query": {"page": 1, "per_page": 1}},
        ):
            response = await client.post("/v1/desktop/radient", json=body)
            assert response.status_code == 502, body
            detail = response.json()["detail"]
            assert detail["code"] == "radient_upstream_failed"
            assert detail["details"]["reason"] == "refresh_did_not_land"
        # Not one upstream read was attempted, and no bearer of any kind was spent.
        assert stub.calls == []

        # The credentialless op still answers: it needs no sign-in, so a credential
        # it cannot freshen must not become a failure it never needed — and it goes
        # bare rather than spending the stale bearer.
        prices = await client.post("/v1/desktop/radient", json={"operation": "prices"})
        assert prices.status_code == 200
        assert "GET /v1/prices <none>" in stub.paths()


async def test_an_upstream_refusal_of_a_live_bearer_is_the_credential_class(
    headless_tui_env, monkeypatch
) -> None:
    """A FRESH bearer Radient still refuses keeps the upstream status, plus the code."""
    async with proxy(headless_tui_env, monkeypatch, me_status=401) as (client, stub):
        response = await client.post("/v1/desktop/radient", json={"operation": "account"})
        assert response.status_code == 401
        assert response.json()["detail"] == {
            "code": "radient_credential_refused",
            "message": "Radient could not complete this operation",
            "details": {"upstream_status": 401},
        }
        # The fresh token WAS spent here, which is what separates this case from the
        # lease case above: the refusal came from Radient, not from the store.
        assert stub.spent() == [FRESH]


async def test_an_upstream_outage_is_the_retry_class(headless_tui_env, monkeypatch) -> None:
    async with proxy(headless_tui_env, monkeypatch, me_status=500) as (client, stub):
        response = await client.post("/v1/desktop/radient", json={"operation": "account"})
        assert response.status_code == 502
        assert response.json()["detail"] == {
            "code": "radient_upstream_failed",
            "message": "Radient could not complete this operation",
            "details": {"upstream_status": 500},
        }
        assert stub.spent() == [FRESH]


async def test_no_credential_at_all_answers_the_sign_in_class_with_a_code(
    headless_tui_env, monkeypatch
) -> None:
    async with proxy(headless_tui_env, monkeypatch, seed="none") as (client, stub):
        response = await client.post("/v1/desktop/radient", json={"operation": "account"})
        assert response.status_code == 409
        assert response.json()["detail"] == {
            "code": "radient_no_credential",
            "message": SIGN_IN,
            "details": {},
        }
        assert stub.calls == []


async def test_the_apps_own_bearer_refusal_carries_no_radient_code(
    headless_tui_env, monkeypatch
) -> None:
    """The other half of the disambiguation: the PLANE's refusal must not look like ours.

    A client tells "restart or re-pair the app" from "sign in to Radient" by the
    code, so the plane's own 401 has to arrive without one.
    """
    async with proxy(headless_tui_env, monkeypatch) as (client, _stub):
        client.headers.pop("Authorization")
        response = await client.post("/v1/desktop/radient", json={"operation": "account"})
        assert response.status_code == 401
        detail = response.json()["detail"]
        assert not isinstance(detail, dict) or "radient_" not in str(detail.get("code", ""))
