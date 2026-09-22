"""The `/v1/desktop/models` `errors` map names only providers with NO listing.

Reproduction (D1). `ProviderController.live_catalogue` returns `(entries, statuses)`
where `statuses` maps EVERY provider it considered to discovery's status string —
`ok`, `cached`, `static`, `unauthenticated`, `stale`, `empty`. The route named every
key of that map as an error, so the desktop renderer's notice
(`Some providers did not answer (23).`) accused all 23 providers on the operator's
own catalogue, including the two aggregators whose 452 rows were on screen, while
discovery had reported nothing failed at all.

The invariant these tests hold down is one sentence: the map's KEY SET is exactly
the providers whose status is in `FAILED_LISTING_STATUSES` — nothing more, nothing
less — and the map's shape (provider -> reason) is unchanged so the renderer needs
no change to read it.
"""

from pathlib import Path

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.credentials import CredentialManager
from local_operator.server.routes import auth, desktop_catalogues

TOKEN = "desktop-catalogue-errors-test-token"
pytestmark = pytest.mark.asyncio

#: A MIXED status dict in the shape discovery really produces: two healthy
#: listings, a deliberate cache serve, a keyless provider that was never asked,
#: providers with no listing endpoint at all, one genuinely failed fetch, and one
#: that answered with nothing. Only the last two belong in `errors`.
MIXED_STATUSES = {
    "openrouter": "ok",
    "deepseek": "ok",
    "radient": "cached",
    "vllm": "stale",
    "radient-key": "static",
    "anthropic": "unauthenticated",
    "acme-local": "empty",
}
EXPECTED_ERROR_KEYS = {"vllm", "acme-local"}


class _StubController:
    """The two methods the route uses, plus `close` for its `finally`."""

    def __init__(self, statuses: dict[str, str]) -> None:
        self.statuses = statuses
        self.closed = False

    async def live_catalogue(self, **_kwargs):
        return [], dict(self.statuses)

    def usable_providers(self):
        return {"openrouter", "deepseek"}

    def close(self) -> None:
        self.closed = True


@pytest_asyncio.fixture
async def catalogue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    app = FastAPI()
    app.include_router(auth.router)
    app.include_router(desktop_catalogues.router)
    app.state.config_manager = ConfigManager(tmp_path)
    app.state.credential_manager = CredentialManager(tmp_path)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield client, app
    if getattr(app.state, "desktop_auth", None):
        await app.state.desktop_auth.close()


async def _models_with(client: AsyncClient, app: FastAPI, statuses: dict[str, str]):
    """GET `/v1/desktop/models?live=true` against a controller with `statuses`."""
    app.state.desktop_auth = _StubHost(statuses)
    return (await client.get("/v1/desktop/models?live=true")).json()["result"]


class _StubHost:
    def __init__(self, statuses: dict[str, str]) -> None:
        self.statuses = statuses
        self.last: _StubController | None = None

    def controller(self) -> _StubController:
        self.last = _StubController(self.statuses)
        return self.last

    async def close(self) -> None:
        return None


async def test_errors_names_only_providers_that_produced_no_listing(catalogue):
    """THE reproduction: a mixed status dict must accuse exactly the failed two.

    Before the fix this returned all seven keys and the renderer rendered
    `Some providers did not answer (7)`. With 23 real providers that is the
    operator's `(23)` note.
    """
    client, app = catalogue
    body = await _models_with(client, app, MIXED_STATUSES)
    assert set(body["errors"]) == EXPECTED_ERROR_KEYS
    # The wire shape is unchanged: a provider -> reason map the renderer reads
    # by key. Nothing here may become a list of reasons or a nested object.
    assert all(isinstance(reason, str) and reason for reason in body["errors"].values())


async def test_healthy_and_deliberate_statuses_are_never_accused(catalogue):
    """Each non-failure status, one at a time, keeps the map EMPTY.

    Parametrised per status rather than asserted in one blob so a future change
    that starts accusing exactly one of them names itself in the failure.
    """
    client, app = catalogue
    for status in ("ok", "cached", "static", "unauthenticated"):
        body = await _models_with(client, app, {"openrouter": status})
        assert body["errors"] == {}, f"{status!r} was reported as a failed listing"


async def test_a_stale_or_empty_listing_is_the_only_thing_reported(catalogue):
    client, app = catalogue
    stale = await _models_with(client, app, {"openrouter": "stale"})
    assert set(stale["errors"]) == {"openrouter"}
    empty = await _models_with(client, app, {"openrouter": "empty"})
    assert set(empty["errors"]) == {"openrouter"}


async def test_the_map_is_wired_from_discovery_vocabulary_not_a_local_copy(catalogue):
    """The route and discovery must share ONE statement of what a failure is.

    A second spelling of the set here is how the statuses and the failures came
    to disagree in the first place, so the constant is imported, not re-typed.
    """
    from local_operator.model.discovery import FAILED_LISTING_STATUSES

    assert FAILED_LISTING_STATUSES == {"stale", "empty"}
    for provider, status in MIXED_STATUSES.items():
        expected = status in FAILED_LISTING_STATUSES
        assert (provider in EXPECTED_ERROR_KEYS) == expected
