"""``POST /v1/desktop/usage/reset`` -- the desktop's half of ``usage.reset``.

WHY THESE EXIST
---------------
The route spends a vendor usage-reset credit, which cannot be refunded, and it
is the contract the desktop UI's ``usage.reset`` op is written against. These
tests hold the ROUTE's half: what a body may say, that a body naming no account
never reaches the controller, that a definitive outcome forces exactly ONE live
read so the desktop cannot paint stale numbers, that a ``failed``/``refused``
outcome leaves the rows it already holds alone, and that the rows this route
returns are the SAME shape ``GET /v1/desktop/usage`` builds (one builder, so the
two cannot drift).

Everything is synthetic and every controller is a double: the vendor is never
called, and no real credential store is opened. Single flight is exercised on
the REAL controller in ``tests/unit/providers/test_controller.py``
(``TestConsumeUsageReset``), because a double would only prove the route calls
it.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.providers.usage import (
    ResetConsumeResult,
    UsageAmount,
    UsageLimit,
    UsageReport,
)
from local_operator.server.routes import auth, desktop_catalogues
from local_operator.server.utils.desktop_auth import DesktopAuth

TOKEN = "desktop-usage-reset-test-token"

#: A client-minted idempotency id in the shape ``RequestID`` accepts.
REDEEM_ID = "8f14e45f-ceea-467a-9a3a-1b2c3d4e5f60"

#: Outcomes the vendor ANSWERED: the spend's fate is settled, the numbers moved,
#: and the desktop must be handed the post-spend rows.
DEFINITIVE = ("reset", "already_redeemed", "no_credit", "nothing_to_reset")

pytestmark = pytest.mark.asyncio


class _Controller:
    """The provider controller this route talks to, recorded rather than real."""

    def __init__(self, *, outcome: str, reports: list[UsageReport], refused: bool = False):
        self.outcome = outcome
        self.refused = refused
        self.reports = reports
        self.consume_calls: list[tuple[Any, ...]] = []
        self.fetch_calls: list[tuple[Any, ...]] = []
        self.closed = 0

    def provider(self, provider_id: str) -> object | None:
        return object() if provider_id == "openai" else None

    async def consume_usage_reset(
        self,
        provider: str,
        identity: str | None,
        redeem_request_id: str,
        credential_ref: str | None = None,
    ) -> ResetConsumeResult:
        self.consume_calls.append((provider, identity, redeem_request_id, credential_ref))
        return ResetConsumeResult(
            outcome=self.outcome, windows_reset=2, detail="", refused=self.refused
        )

    async def fetch_usage(
        self, providers: list[str] | None = None, force_refresh: bool = False
    ) -> list[UsageReport]:
        self.fetch_calls.append((providers, force_refresh))
        return list(self.reports)

    def cached_usage_reports(self, provider: str | None = None) -> list[UsageReport]:
        return list(self.reports)

    def close(self) -> None:
        self.closed += 1


def _reports() -> list[UsageReport]:
    """One report per account, in the shape the row builder consumes.

    ``fetched_at=0`` on purpose: ``age_ms`` is then ``None`` on both routes, so
    the row-shape comparison is about the SHAPE and not about how many
    milliseconds passed between the two requests.
    """
    return [
        UsageReport(
            provider="openai",
            fetched_at=0,
            limits=[
                UsageLimit(
                    id="codex-5h",
                    label="5 hour",
                    amount=UsageAmount(used=20.0, limit=100.0),
                )
            ],
            resets_available=1,
            identity="me@example.com",
            credential_ref="cred-1",
        )
    ]


@pytest_asyncio.fixture
async def desktop(tmp_path, monkeypatch):
    """The two usage routes on an isolated HOME/config, with a controller double."""
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    controller = _Controller(outcome="reset", reports=_reports())
    monkeypatch.setattr(DesktopAuth, "controller", lambda self: controller)
    app = FastAPI()
    app.include_router(auth.router)
    app.include_router(desktop_catalogues.router)
    app.state.config_manager = ConfigManager(tmp_path)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield client, controller
    if getattr(app.state, "desktop_auth", None):
        await app.state.desktop_auth.close()


def _body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "provider": "openai",
        "credential_ref": None,
        "identity": "me@example.com",
        "redeem_request_id": REDEEM_ID,
    }
    body.update(overrides)
    return body


async def test_a_reset_without_the_desktop_token_is_refused(desktop) -> None:
    """The same auth guard as every sibling route, on a route that spends.

    Worth its own case rather than trusting the router-level dependency: this is
    the ONE write on this module's surface whose effect cannot be undone, so an
    unauthenticated caller reaching the controller would be a spend on someone
    else's machine.
    """
    client, controller = desktop

    response = await client.post(
        "/v1/desktop/usage/reset",
        json=_body(),
        headers={"Authorization": "Bearer wrong"},
    )

    assert response.status_code == 401, response.text
    assert controller.consume_calls == []


async def test_an_unknown_provider_is_the_sibling_422_envelope(desktop) -> None:
    """The registry decides "unknown" BEFORE any spend can be attempted."""
    client, controller = desktop

    response = await client.post("/v1/desktop/usage/reset", json=_body(provider="nonesuch"))

    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == "wire.errors.desktop.unknown_provider"
    assert controller.consume_calls == []
    assert controller.fetch_calls == []


async def test_a_body_naming_no_account_never_reaches_the_controller(desktop) -> None:
    """Neither identifier is a BODY-shaped error, not a reset attempt.

    Letting it through would make the controller answer ``refused`` and the
    desktop paint the failed band for a request it should not have sent -- and
    the fallback a guessing server would take is the current cascade account,
    which is exactly the login the user did not name.
    """
    client, controller = desktop

    response = await client.post(
        "/v1/desktop/usage/reset", json=_body(identity=None, credential_ref=None)
    )

    assert response.status_code == 422, response.text
    assert "credential_ref or identity is required" in response.text
    assert controller.consume_calls == []
    assert controller.fetch_calls == []


async def test_an_unknown_body_field_is_refused(desktop) -> None:
    """``extra=\"forbid\"``: a misspelt field must be told, not silently dropped."""
    client, controller = desktop

    response = await client.post("/v1/desktop/usage/reset", json=_body(redeem="x"))

    assert response.status_code == 422, response.text
    assert controller.consume_calls == []


@pytest.mark.parametrize(
    "body, expected",
    [
        # Exactly ONE identifier is sent; the route forwards what it was given
        # and never invents the other one.
        (
            {"identity": "me@example.com", "credential_ref": None},
            ("openai", "me@example.com", REDEEM_ID, None),
        ),
        ({"identity": None, "credential_ref": "cred-1"}, ("openai", None, REDEEM_ID, "cred-1")),
        (
            {"identity": "me@example.com", "credential_ref": "cred-1"},
            ("openai", "me@example.com", REDEEM_ID, "cred-1"),
        ),
    ],
)
async def test_the_identifiers_reach_the_controller_unchanged(desktop, body, expected) -> None:
    client, controller = desktop

    response = await client.post("/v1/desktop/usage/reset", json=_body(**body))

    assert response.status_code == 200, response.text
    assert controller.consume_calls == [expected]


@pytest.mark.parametrize("outcome", DEFINITIVE)
async def test_a_definitive_outcome_returns_one_forced_refresh(desktop, outcome) -> None:
    """Every answer the vendor GAVE settles the spend, so the numbers moved.

    ONE fetch, forced, for that provider -- a second read or an unforced (cached)
    one would hand the desktop the row written before the spend and the card
    would report a reset that had not landed yet.
    """
    client, controller = desktop
    controller.outcome = outcome

    response = await client.post("/v1/desktop/usage/reset", json=_body())

    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["reset"] == {
        "outcome": outcome,
        "windows_reset": 2,
        "detail": "",
        "refused": False,
    }
    assert controller.fetch_calls == [(["openai"], True)]
    assert result["usage"] is not None
    assert [row["provider"] for row in result["usage"]["reports"]] == ["openai"]
    assert result["usage"]["source"] == "live"


@pytest.mark.parametrize("refused", [False, True])
async def test_failed_and_refused_never_refresh(desktop, refused) -> None:
    """A ``failed`` verdict says the credit's fate is UNKNOWN; replacing the
    numbers under that sentence would contradict it. ``refused`` (nothing was
    sent) adds nothing to refresh for, and ``usage`` stays ``None`` in both."""
    client, controller = desktop
    controller.outcome = "failed"
    controller.refused = refused

    response = await client.post("/v1/desktop/usage/reset", json=_body())

    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["reset"] == {
        "outcome": "failed",
        "windows_reset": 2,
        "detail": "",
        "refused": refused,
    }
    assert controller.fetch_calls == []
    assert result["usage"] is None


async def test_the_reset_and_get_routes_build_identical_rows(desktop) -> None:
    """ONE row builder: the reset's rows are the GET's rows, key for key.

    They are read back to back, so a field one route derived and the other did
    not (``state``, ``age_ms``, a normalized ``notes``) is exactly the drift this
    test exists to catch -- the desktop paints the reset's rows immediately.

    The GET is asked for ``live`` so the two payloads state the same ``source``:
    the only difference left is the clock, which is what this test then excuses.
    """
    client, controller = desktop

    get_body = (await client.get("/v1/desktop/usage?provider=openai&live=true")).json()["result"]
    post_body = (await client.post("/v1/desktop/usage/reset", json=_body())).json()["result"]

    # Key for key, top level included: ``fetched_at`` is the clock and is the
    # only field allowed to differ.
    assert set(post_body["usage"]) == set(get_body)
    assert {k: v for k, v in post_body["usage"].items() if k != "fetched_at"} == {
        k: v for k, v in get_body.items() if k != "fetched_at"
    }
    # And the row is the dataclass plus the three derived keys -- not a subset
    # that happens to agree on what both emitted.
    row = post_body["usage"]["reports"][0]
    assert {"provider", "limits", "resets_available", "credential_ref", "age_ms", "state"} <= set(
        row
    )
    assert set(row) == set(dataclasses.asdict(UsageReport(provider="openai"))) | {"age_ms", "state"}
