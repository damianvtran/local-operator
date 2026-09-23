"""The desktop's read-only view of this machine's tunnel and its login verdict.

Reported to the desktop app the way the operator hit it: a Radient sign-in that
looked configured in the account list while the connector on the machine was
parked and the phone was unreachable. Three things are pinned here — the route's
shape, that it answers about THIS MACHINE (a cached cloud record, never an
upstream call), and that a park outranks any probe of a gateway that is not
running.

The route is exercised through the ROUTER, not by calling its body: the desktop
gate is part of what the UI sees.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.server.routes import auth, desktop_tunnel

TOKEN = "desktop-tunnel-test-token"
pytestmark = pytest.mark.asyncio


def _tunnel_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "tunnel"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _configure(tmp_path: Path, *, stopped: bool = False, credential_id: int = 7) -> None:
    _tunnel_dir(tmp_path).joinpath("config.json").write_text(
        json.dumps(
            {
                "tunnel_id": "tunnel-1",
                "credential_id": credential_id,
                "gateway_port": 4100,
                "stopped": stopped,
                "record": {"id": "tunnel-1", "status": "active"},
            }
        )
    )


def _park(tmp_path: Path, *, reason: str = "login_required") -> None:
    _tunnel_dir(tmp_path).joinpath("state.json").write_text(
        json.dumps(
            {
                "state": "parked",
                "reason": reason,
                "detail": "The connector's Radient login is no longer valid.",
                "remedy": {"command": "lop login radient", "url": "https://console.invalid"},
                "credential_id": 7,
                "at": 1_800_000_000,
                "first_at": 1_800_000_000,
                "attempts": 1,
                "logged_at": 1_800_000_000,
                "logged_attempts": 1,
            }
        )
    )


@pytest_asyncio.fixture
async def desktop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    app = FastAPI()
    app.include_router(auth.router)
    app.include_router(desktop_tunnel.router)
    app.state.config_manager = ConfigManager(tmp_path)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield client, tmp_path, app
    if getattr(app.state, "desktop_auth", None):
        await app.state.desktop_auth.close()


async def test_the_route_reports_the_park_rather_than_a_dead_gateway(desktop) -> None:
    """The parked connector is the answer, and the cloud is honestly cached.

    A parked connector is not running, so the loopback probe can only ever
    answer "nothing is listening here" — reporting that as the state would send
    the UI hunting for a service problem when the fix is a sign-in.
    """
    client, tmp_path, _app = desktop
    _configure(tmp_path)
    _park(tmp_path)

    body = (await client.get("/v1/desktop/tunnel")).json()
    assert body["status"] == 200
    result = body["result"]

    assert result["configured"] is True
    assert result["tunnel_id"] == "tunnel-1"
    assert result["connector"]["state"] == "parked"
    assert result["connector"]["reason"] == "login_required"
    assert result["connector"]["since"] == 1_800_000_000
    # The remedy is a terminal command, reported rather than run: this route has
    # no write half, so the UI shows it and the operator runs it.
    assert result["remedy"] == {"command": "lop login radient", "url": "https://console.invalid"}
    # Never a live claim: this route answers about the machine without calling
    # Radient, which is also the only answer available when the login is dead.
    assert result["cloud"] == {"status": "active", "source": "cached", "reason": ""}
    # No credential row exists in this fixture, so the login verdict is the
    # honest "sign in": there is nothing stored for this tunnel to refresh, which
    # is why no call is made at all. (A row that is merely STALE does make one —
    # bounded and memoised, see REFRESH_WAIT_S — which the tests at the end of
    # this file drive.)
    assert result["login"] == {"credential_id": 7, "state": "login_required"}


async def test_a_machine_with_no_tunnel_says_so_instead_of_stopped(desktop) -> None:
    """`stopped` would describe a connector someone stopped; nothing was.

    The UI needs the difference to decide whether to offer a sign-in at all.
    """
    client, _tmp_path, _app = desktop
    result = (await client.get("/v1/desktop/tunnel")).json()["result"]

    assert result["configured"] is False
    assert result["connector"]["state"] == "not configured"
    assert result["login"]["state"] == "unknown"
    assert result["remedy"] is None


async def test_a_deliberately_stopped_tunnel_is_not_reported_as_parked(desktop) -> None:
    """A stopped tunnel is "not using the tunnel", as `lop tunnel status` says
    too: the park behind it is not a condition anyone has to act on."""
    client, tmp_path, _app = desktop
    _configure(tmp_path, stopped=True)
    _park(tmp_path)

    result = (await client.get("/v1/desktop/tunnel")).json()["result"]
    assert result["connector"]["state"] == "stopped"
    assert result["remedy"] is None


async def test_the_account_status_carries_the_login_verdict(desktop) -> None:
    """The account section's honesty: a stored row can be `configured` with an
    unexpired token and still be refused, which is the incident's own shape."""
    client, tmp_path, _app = desktop
    _configure(tmp_path)
    _park(tmp_path)

    result = (await client.get("/v1/auth/status")).json()["result"]
    assert result["accounts"] == []
    assert result["radient_login"] == {"credential_id": 7, "state": "login_required"}
    assert result["tunnel_remedy"] == {
        "command": "lop login radient",
        "url": "https://console.invalid",
    }


async def test_a_hanging_token_endpoint_cannot_hold_the_poll_open(desktop, monkeypatch) -> None:
    """M1: the login verdict is BOUNDED, so a partitioned network cannot stall the poll.

    Measured on this branch before the fix: **30.9 s** for one
    `GET /v1/desktop/tunnel` — the store's own client timeout for the refresh
    POST, spent on a call whose answer is "I could not check", on the loop that
    serves every other request.

    The assertion is structural rather than a stopwatch: the stubbed endpoint
    parks for several times the bound and records whether it ever finished, so a
    passing test means the refresh was CANCELLED at the bound. An unbounded call
    would have completed it, and `finished` is what says which happened — a
    duration would only say how fast the machine is.
    """
    import asyncio
    from contextlib import closing

    from local_operator.providers import auth_store
    from local_operator.tunnels import report

    client, tmp_path, _app = desktop
    with closing(auth_store.AuthStore()) as store:
        row = store.upsert_credential(
            "radient",
            {
                "type": "oauth",
                "account_id": "qa",
                "access": "stale-access",
                "refresh": "stored-refresh",
                # Outside the refresh skew, so the verdict has to ASK: a fresh
                # token is served without a call and this test would prove
                # nothing.
                "expires": 1,
            },
        )
    _configure(tmp_path, credential_id=row.id)

    started = asyncio.Event()
    finished = asyncio.Event()

    async def hanging(credentials):  # noqa: ANN001 — the store's own refresh fn
        started.set()
        await asyncio.sleep(report.REFRESH_WAIT_S * 3)
        finished.set()

    monkeypatch.setattr(auth_store.AuthStore, "_refresh_fn", lambda self, provider: hanging)

    result = (await client.get("/v1/desktop/tunnel")).json()["result"]

    assert started.is_set(), "the refresh was never attempted: the fixture proves nothing"
    assert not finished.is_set(), "the route waited for the token endpoint instead of bounding it"
    # The honest answer for a check that did not finish, and NOT `login_required`:
    # sending an operator whose network is down to a login is the misdirection
    # this surface exists to remove.
    assert result["login"] == {"credential_id": row.id, "state": "unknown"}


async def test_a_verdict_that_cost_a_call_is_not_re_asked_on_every_poll(
    desktop, monkeypatch
) -> None:
    """M1: a negative verdict is reused for the window the store blocks the row for.

    The same treatment #1340 gave the desktop's own side of this question
    (`DIAGNOSIS_TTL_S`): the answer needed a token-endpoint POST, and asking
    again inside the window changes nothing while paying again.

    The row's own write stamp is in the memo's key, so a login that LANDS in
    between is decided again — that half is asserted, because a memo that masked a
    re-login would turn this fix into a worse bug than the cost it removes. The
    store's clock is stepped rather than slept past: it stamps writes in whole
    milliseconds, and two writes in one millisecond are indistinguishable to a
    key built from that stamp.
    """
    import itertools
    from contextlib import closing

    from local_operator.providers import auth_store

    client, tmp_path, _app = desktop
    clock = itertools.count(start=1_800_000_000_000, step=1_000)
    monkeypatch.setattr(auth_store.AuthStore, "_now_ms", staticmethod(lambda: next(clock)))
    attempts: list[str] = []

    async def offline(credentials):  # noqa: ANN001 — the store's own refresh fn
        attempts.append(str(credentials.get("refresh")))
        raise httpx.ConnectError("network is unreachable")

    monkeypatch.setattr(auth_store.AuthStore, "_refresh_fn", lambda self, provider: offline)
    with closing(auth_store.AuthStore()) as store:
        row = store.upsert_credential(
            "radient",
            {
                "type": "oauth",
                "account_id": "qa",
                "access": "stale-access",
                "refresh": "first-refresh",
                "expires": 1,
            },
        )
    _configure(tmp_path, credential_id=row.id)

    first = (await client.get("/v1/desktop/tunnel")).json()["result"]
    assert first["login"] == {"credential_id": row.id, "state": "unknown"}
    assert len(attempts) == 1

    # The same poll again inside the window: the verdict stands and the token
    # endpoint is not asked a second time.
    second = (await client.get("/v1/desktop/tunnel")).json()["result"]
    assert second["login"] == first["login"]
    assert len(attempts) == 1, "the poll paid for the same answer twice"

    # A login lands (the same identity, written again): the row CHANGED, so the
    # verdict is decided again whatever the clock says.
    with closing(auth_store.AuthStore()) as store:
        store.upsert_credential(
            "radient",
            {
                "type": "oauth",
                "account_id": "qa",
                "access": "fresh-access",
                "refresh": "second-refresh",
                "expires": 1,
            },
        )
    third = (await client.get("/v1/desktop/tunnel")).json()["result"]
    assert len(attempts) == 2, "a landed login was masked by the memo"
    assert third["login"] == first["login"]
