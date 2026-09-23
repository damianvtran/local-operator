"""The `/v1/desktop/models` `errors` map names only providers with NO listing.

Reproduction (D1). `ProviderController.live_catalogue` returns `(entries, statuses)`
where `statuses` maps EVERY provider it considered to discovery's status string —
`ok`, `cached`, `static`, `unauthenticated`, `stale`, `empty`. The route named every
key of that map as an error, so the desktop renderer's notice
(`Some providers did not answer (23).`) accused all 23 providers on the operator's
own catalogue, including the two aggregators whose 452 rows were on screen, while
discovery had reported nothing failed at all.

The rule the route now holds, in one sentence: report a provider when it produced no
listing AND contributed no rows. `stale` and `empty` are failures outright; `static`
is a failure ONLY when the provider also contributed zero entries — a provider
bundling registry rows and a provider whose fetch died with nothing cached both
report `static`, and only the second is a failure (R1-4/Q1). `cached`,
`unauthenticated` and a row-ful `static` stay silent. The map's shape
(provider -> reason) is unchanged, so the renderer needs no change to read it.
"""

from pathlib import Path

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.server.routes import auth, desktop_catalogues

TOKEN = "desktop-catalogue-errors-test-token"
pytestmark = pytest.mark.asyncio

#: A MIXED status dict in the shape discovery really produces: two healthy
#: listings, a deliberate cache serve, a provider with no listing endpoint but
#: bundled rows, a keyless provider that was never asked, one genuinely failed
#: fetch, and one that answered with nothing. Only the failed fetch and the empty
#: answer belong in `errors` — the `static` provider CONTRIBUTES ROWS here, which
#: is the case that must stay silent.
MIXED_STATUSES = {
    "openrouter": "ok",
    "deepseek": "ok",
    "radient": "cached",
    "vllm": "stale",
    "radient-key": "static",
    "anthropic": "unauthenticated",
    "acme-local": "empty",
}
CONTRIBUTING = {"openrouter", "deepseek", "radient", "radient-key"}
EXPECTED_ERROR_KEYS = {"vllm", "acme-local"}


class _StubController:
    """The two methods the route uses, plus `close` for its `finally`.

    `contributing` is the set of providers that returned at least one entry — the
    second fact the route needs to tell a row-ful `static` from a row-less one
    (R1-4/Q1).
    """

    def __init__(self, statuses: dict[str, str], contributing: set[str]) -> None:
        self.statuses = statuses
        self.contributing = contributing
        self.closed = False

    async def live_catalogue(self, **_kwargs):
        return (
            [_entry(provider) for provider in sorted(self.contributing)],
            dict(self.statuses),
        )

    def usable_providers(self):
        return {"openrouter", "deepseek"}

    def close(self) -> None:
        self.closed = True


def _entry(provider: str):
    """A minimal CatalogueEntry, distinguishable by its provider."""
    from local_operator.providers.controller import CatalogueEntry

    return CatalogueEntry(
        provider=provider,
        model_id="a-model",
        label="a-model",
        context_window=1000,
        input_price=0.0,
        output_price=0.0,
        connected=True,
    )


@pytest_asyncio.fixture
async def catalogue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    app = FastAPI()
    app.include_router(auth.router)
    app.include_router(desktop_catalogues.router)
    app.state.config_manager = ConfigManager(tmp_path)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield client, app
    if getattr(app.state, "desktop_auth", None):
        await app.state.desktop_auth.close()


async def _models_with(
    client: AsyncClient,
    app: FastAPI,
    statuses: dict[str, str],
    contributing: set[str] | None = None,
):
    """GET `/v1/desktop/models?live=true` against a controller with `statuses`."""
    app.state.desktop_auth = _StubHost(statuses, contributing or set())
    return (await client.get("/v1/desktop/models?live=true")).json()["result"]


class _StubHost:
    def __init__(self, statuses: dict[str, str], contributing: set[str]) -> None:
        self.statuses = statuses
        self.contributing = contributing
        self.last: _StubController | None = None

    def controller(self) -> _StubController:
        self.last = _StubController(self.statuses, self.contributing)
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
    body = await _models_with(client, app, MIXED_STATUSES, CONTRIBUTING)
    assert set(body["errors"]) == EXPECTED_ERROR_KEYS
    # The wire shape is unchanged: a provider -> reason map the renderer reads
    # by key. Nothing here may become a list of reasons or a nested object.
    assert all(isinstance(reason, str) and reason for reason in body["errors"].values())


async def test_a_static_provider_that_contributed_rows_is_never_accused(catalogue):
    """`static` with rows on screen is an ANSWER, not a failure.

    A provider that bundles registry rows reports `static` whenever its live
    listing is unavailable, and those rows are exactly what the user asked to
    see. Accusing it would be the D1 over-report in miniature.
    """
    client, app = catalogue
    body = await _models_with(client, app, {"openrouter": "static"}, contributing={"openrouter"})
    assert body["errors"] == {}


async def test_a_static_provider_that_contributed_nothing_is_an_error(catalogue):
    """R1-4/Q1: the fetch failed with nothing cached on a row-less provider.

    Ten registry providers bundle ZERO static rows (the aggregators, ``typesafe``,
    ``test``, and the five local_setup ones), so a first-run 401 or dead host
    yields `static` with no rows — the operator's original "refresh failed"
    state. The route must name it; `FAILED_LISTING_STATUSES` alone would go
    silent here where the base named the provider.
    """
    client, app = catalogue
    body = await _models_with(client, app, {"openrouter": "static"}, contributing=set())
    assert set(body["errors"]) == {"openrouter"}


async def test_a_cached_or_unauthenticated_provider_is_never_accused(catalogue):
    """Each non-failure status, one at a time, keeps the map EMPTY.

    `static` is deliberately absent from this list: its verdict needs the second
    fact (did it contribute rows), and it is covered by the two tests above.
    """
    client, app = catalogue
    for status in ("ok", "cached", "unauthenticated"):
        body = await _models_with(client, app, {"openrouter": status}, contributing=set())
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
    `static` is excluded from this cross-check deliberately: the route resolves it
    with a second fact, so only the unambiguous statuses map straight through.
    """
    from local_operator.model.discovery import FAILED_LISTING_STATUSES

    assert FAILED_LISTING_STATUSES == {"stale", "empty"}
    for provider, status in MIXED_STATUSES.items():
        if status == "static":
            continue
        expected = status in FAILED_LISTING_STATUSES
        assert (provider in EXPECTED_ERROR_KEYS) == expected
