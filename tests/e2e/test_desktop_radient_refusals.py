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

import asyncio
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

    def __init__(
        self, *, refresh: str = "ok", me_status: int = 200, hold_token: bool = False
    ) -> None:
        self.refresh = refresh
        self.me_status = me_status
        #: Set the moment a refresh POST ARRIVES, before it is answered.
        #:
        #: The join case needs a peer to be provably MID-REFRESH when the proxied
        #: request arrives, and a sleep cannot establish that: under load the request
        #: slides to one side of the window and the case stops testing what it names.
        #: So the test waits on this instead of on the clock.
        self.token_started = asyncio.Event()
        #: When set, the endpoint does not ANSWER until the test says so, which lets
        #: the test hold a peer inside its refresh for as long as the case needs.
        self.token_gate = asyncio.Event() if hold_token else None
        #: (method, path, bearer-or-"" ) for every request that reached the stub.
        self.calls: list[tuple[str, str, str]] = []
        self.app = self._build()

    def release_token(self) -> None:
        """Answer the refresh POST a gate is holding; a no-op without one."""
        if self.token_gate is not None:
            self.token_gate.set()

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
            self.token_started.set()
            if self.token_gate is not None:
                await self.token_gate.wait()
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

    def token_posts(self) -> int:
        """How many refresh POSTs the token endpoint received.

        A COUNT, not an assertion about the count: what the store paid to answer a
        request is invisible to the client, and this change is what makes it visible
        — one refresh per failure was the base's cost, one per FAILING REQUEST was the
        regression, and one per block window is what the memo buys back.
        """
        return sum(
            1 for method, path, _bearer in self.calls if method == "POST" and path == "/token"
        )


@asynccontextmanager
async def proxy(
    config_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    refresh: str = "ok",
    me_status: int = 200,
    hold_token: bool = False,
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
    stub = Stub(refresh=refresh, me_status=me_status, hold_token=hold_token)
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
        # Six sequential refusals, the shape QA measured the cost in (R31): the base spent
        # ONE refresh per FAILING REQUEST — seven for six — and the memo brought that to one
        # per block window. It is now ONE FOR THE WHOLE RUN, because the store records the
        # refusal ON THE ROW: the diagnosis that names it reads the persisted verdict
        # instead of re-earning it with a token-endpoint POST against a grant the endpoint
        # had already refused (review round 1: the tombstone saves a POST). The count still
        # discriminates — a regression that went back to one POST per request would be 6.
        for _ in range(4):
            again = await client.post("/v1/desktop/radient", json={"operation": "account"})
            assert again.status_code == 401
            assert again.json()["detail"]["details"]["reason"] == "grant_invalid"
        assert stub.token_posts() == 1, stub.calls


async def test_a_transient_refresh_failure_answers_the_retry_class(
    headless_tui_env, monkeypatch
) -> None:
    """A 5xx from the token endpoint is not the account's fault: retry, do not re-sign-in."""
    async with proxy(headless_tui_env, monkeypatch, refresh="500") as (client, stub):
        for _ in range(2):
            response = await client.post("/v1/desktop/radient", json={"operation": "account"})
            assert response.status_code == 502
            assert response.json()["detail"]["code"] == "radient_upstream_failed"
            assert response.json()["detail"]["details"]["reason"] == "credential_unavailable"
        assert "GET /v1/me" not in stub.paths()
        # ONE POST FOR THE PAIR OF REQUESTS. The base paid two: the first request's
        # cascade spent one and the diagnosis that named the failure spent another. This
        # change arms a write-ahead send marker before the POST, and a failure that came
        # back with an ANSWER (a 5xx proves nothing about our token, so the marker is kept)
        # defers the deferred token instead of re-presenting it — so the diagnosis pays
        # nothing. The cost is stated rather than implied: the marker is re-bounded to
        # ``ANSWERED_SEND_TTL_S`` (one ``DEFAULT_BLOCK_MS``), the window the cascade ALREADY
        # refuses this credential for, so the account heals on the cadence it always had —
        # and the token is not re-presented inside that window, which is what a provider
        # that committed a rotation and then failed the response needs (review round 1, R3).
        assert stub.token_posts() == 1, stub.calls


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


async def test_a_peer_refresh_that_lands_is_joined_not_refused(
    headless_tui_env, monkeypatch
) -> None:
    """THE LIVING HALF OF THE PEER CASE, and the one this suite was missing.

    A peer holding the lease is not a refusal: it is a refresh that is going to
    succeed. This test drives a REAL one — a genuine `get_oauth_access` in a second
    store on the same `auth.db` — and holds its refresh POST open, so our request is
    provably inside the peer's window rather than racing it, then lets the peer land
    while our request is waiting.

    The join is then asserted in the two directions that matter: our request has NOT
    refused while the peer is still in flight, and once the peer's token is written
    the same request answers 200 off THAT token. Before the bounded join it answered
    502 `refresh_did_not_land` for the whole peer window, where the base answered 200
    off the stale bearer (review round 1: a 500 ms endpoint, a request issued 50 ms
    in, measured 502, then 200 once the peer landed). The lease-holds-and-never-lands
    case above passes either way, which is exactly why this one has to exist.
    """
    async with proxy(headless_tui_env, monkeypatch, hold_token=True) as (client, stub):
        peer = AuthStore(headless_tui_env / "auth.db")
        try:
            # The peer takes the cross-process lease and blocks inside the token
            # endpoint. `token_started` is set by the endpoint itself, so when it fires
            # the lease IS held — no sleep is asked to prove that.
            peer_refresh = asyncio.create_task(peer.get_oauth_access("radient"))
            await asyncio.wait_for(stub.token_started.wait(), timeout=60)
            request = asyncio.create_task(
                client.post("/v1/desktop/radient", json={"operation": "account"})
            )
            # The credentialless op, asked inside the SAME window, must not be made to wait
            # for a credential it never needed: it falls back to running bare at once, so
            # its line reaches the ledger before the join above can finish, and it spends
            # no bearer even once the peer's token exists — the proof for that is below,
            # in the ledger's ORDER rather than in a clock reading.
            prices = await asyncio.wait_for(
                client.post("/v1/desktop/radient", json={"operation": "prices"}), timeout=30
            )
            assert prices.status_code == 200
            # Long past the one 0.05 s peer slice the store's own path waits, and past
            # everything a refusal would have needed: a proxy that refused here would
            # have answered already, and the window is still open because the peer's
            # POST is gated.
            await asyncio.sleep(0.7)
            assert not request.done(), "the proxy refused while a peer's refresh was in flight"
            stub.release_token()
            response = await asyncio.wait_for(request, timeout=30)
            assert response.status_code == 200, response.text
            assert await peer_refresh is not None
        finally:
            stub.release_token()
            peer.close()
        # Joined, and proven joined by what the endpoint was asked: ONE refresh for the
        # pair of us — the peer's — and the bearer we spent is the token it minted. A
        # request that had raced the peer would have POSTed a second time, and one that
        # had refused would have spent nothing at all.
        assert stub.token_posts() == 1, stub.calls
        # The peer's refresh POST, then the credentialless read that did not wait for it,
        # then OUR read — which is on the ledger only because the peer's token exists.
        assert stub.paths() == [
            "POST /token <none>",
            "GET /v1/prices <none>",
            f"GET /v1/me {FRESH}",
        ], stub.paths()


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
        # Asserted as a SHAPE and as the sentence, not as "not a dict carrying a
        # radient_ code": that form also passes for a dict with no code at all, so it
        # would not have pinned the thing its name claims (review n1). This matters
        # because the desktop client keys on `code` — a refusal that quietly became
        # typed here would move this class into Radient's on the client for no reason.
        detail = response.json()["detail"]
        assert isinstance(detail, str)
        assert detail == "Desktop authorization is required."
