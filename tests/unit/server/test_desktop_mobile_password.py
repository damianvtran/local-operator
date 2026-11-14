"""``GET /v1/desktop/mobile/password``: the desktop's "Show phone password".

What is pinned, and why each matters:

* the desktop gate (bearer, then Origin) refuses before the store is read -- a
  page or a stray local process must not reach the secret;
* an unset password is a stable ``mobile_password_unset`` code the UI keys its
  guidance off, never prose;
* the value leaves with ``no-store`` / ``Pragma: no-cache`` and is read at
  request time (a password rotated between two clicks shows the new one);
* nothing the server logs -- any logger, any level -- carries the value;
* the capability key is advertised to the desktop but stripped from the phone
  relay's map, which serves no such route.

The route is exercised through the ROUTER, so the desktop gate is part of what
is tested. The store is the real Linux file fallback under ``tmp_path``: the
platform and ``secret-tool`` lookup are pinned so a developer's Keychain or
Secret Service is never read or written by this file.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.mobile import auth as mobile_auth
from local_operator.server.routes import desktop_mobile_password

TOKEN = "desktop-mobile-password-test-token"
ORIGIN = "http://localhost:5173"
PASSWORD = "phone-portal-secret-7f3a9c"  # noqa: S105 -- a synthetic fixture value
pytestmark = pytest.mark.asyncio

ROUTE = "/v1/desktop/mobile/password"


@pytest_asyncio.fixture
async def desktop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for name in ("HOME", "LOCAL_OPERATOR_CONFIG_DIR"):
        monkeypatch.setenv(name, str(tmp_path))
    for name in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", ORIGIN)
    monkeypatch.delenv(mobile_auth.PASSWORD_ENV, raising=False)
    for name in [key for key in os.environ if key.startswith("CMUX_")]:
        monkeypatch.delenv(name)
    # The file fallback, on every host: never the developer's real keyring.
    monkeypatch.setattr(mobile_auth, "_PLATFORM", "linux")
    monkeypatch.setattr(mobile_auth, "_IS_WINDOWS", False)
    monkeypatch.setattr(mobile_auth, "_secret_tool", lambda: None)

    app = FastAPI()
    app.include_router(desktop_mobile_password.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield client, tmp_path


async def test_no_bearer_is_refused(desktop) -> None:
    client, _ = desktop
    mobile_auth.store_password(PASSWORD)

    response = await client.get(ROUTE, headers={"Authorization": ""})

    assert response.status_code == 401
    assert PASSWORD not in response.text


async def test_a_wrong_bearer_is_refused(desktop) -> None:
    client, _ = desktop
    mobile_auth.store_password(PASSWORD)

    response = await client.get(ROUTE, headers={"Authorization": "Bearer not-the-token"})

    assert response.status_code == 401
    assert PASSWORD not in response.text


async def test_a_wrong_origin_is_refused_even_with_the_bearer(desktop) -> None:
    client, _ = desktop
    mobile_auth.store_password(PASSWORD)

    response = await client.get(ROUTE, headers={"Origin": "https://evil.example"})

    assert response.status_code == 403
    assert PASSWORD not in response.text


async def test_unset_is_a_stable_404_code(desktop) -> None:
    client, tmp_path = desktop
    assert not mobile_auth.password_file().exists()

    response = await client.get(ROUTE, headers={"Origin": ORIGIN})

    assert response.status_code == 404
    assert response.json() == {"detail": {"code": "mobile_password_unset"}}
    assert response.headers["cache-control"] == "no-store"


async def test_set_returns_the_value_with_no_store(desktop) -> None:
    client, tmp_path = desktop
    mobile_auth.store_password(PASSWORD)
    # Proves the store under test is the isolated one, not a home-level file.
    assert mobile_auth.password_file().is_relative_to(tmp_path)

    response = await client.get(ROUTE, headers={"Origin": ORIGIN})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == 200
    assert body["result"] == {"password": PASSWORD}
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["pragma"] == "no-cache"


async def test_the_store_is_read_per_request_not_cached(desktop) -> None:
    """A rotation between two clicks shows the new value: nothing is memoised."""
    client, _ = desktop
    mobile_auth.store_password(PASSWORD)
    first = (await client.get(ROUTE)).json()["result"]["password"]
    mobile_auth.store_password("rotated-value-11b2")
    second = (await client.get(ROUTE)).json()["result"]["password"]

    assert (first, second) == (PASSWORD, "rotated-value-11b2")


async def test_a_200_logs_nothing_carrying_the_password(
    desktop, caplog: pytest.LogCaptureFixture
) -> None:
    """Every record at every logger, DEBUG and up, across a successful read."""
    client, tmp_path = desktop
    mobile_auth.store_password(PASSWORD)

    with caplog.at_level(logging.DEBUG):
        logging.getLogger().setLevel(logging.DEBUG)
        response = await client.get(ROUTE, headers={"Origin": ORIGIN})

    assert response.status_code == 200
    assert response.json()["result"]["password"] == PASSWORD
    rendered = [record.getMessage() for record in caplog.records]
    rendered += [repr(record.args) for record in caplog.records]
    assert not [line for line in rendered if PASSWORD in line]
    # Nor any log file the server may have written under the isolated root
    # (the password file itself is the store, not a log).
    store = mobile_auth.password_file()
    leaked = [
        path
        for path in tmp_path.rglob("*")
        if path.is_file()
        and path != store
        and PASSWORD in path.read_text(encoding="utf-8", errors="ignore")
    ]
    assert leaked == []


async def test_the_capability_is_advertised_to_the_desktop_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_operator.mobile import daemon
    from local_operator.server.features import feature_flags

    assert feature_flags()["mobile_password_read"] == 1
    # The phone relay republishes the map minus DESKTOP_ONLY_FEATURES; it
    # serves no password route, so it must not advertise one.
    monkeypatch.setattr(daemon, "_FEATURE_FLAGS_MEMO", None)
    assert "mobile_password_read" not in daemon._feature_flags()
