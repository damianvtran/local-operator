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

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.credentials import CredentialManager
from local_operator.server.routes import auth, desktop_tunnel

TOKEN = "desktop-tunnel-test-token"
pytestmark = pytest.mark.asyncio


def _tunnel_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "tunnel"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _configure(tmp_path: Path, *, stopped: bool = False) -> None:
    _tunnel_dir(tmp_path).joinpath("config.json").write_text(
        json.dumps(
            {
                "tunnel_id": "tunnel-1",
                "credential_id": 7,
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
    app.state.credential_manager = CredentialManager(tmp_path)
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
    # honest "sign in" — decided on this device, without a network call.
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
