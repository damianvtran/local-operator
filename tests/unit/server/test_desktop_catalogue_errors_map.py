"""The `/v1/desktop/models` `errors` map, through the real rule that builds it.

Reproduction (D1). `ProviderController.live_catalogue` returns `(entries, statuses)`
where `statuses` maps EVERY provider it considered to discovery's status string —
`ok`, `cached`, `static`, `unauthenticated`, `stale`, `empty`. The route named every
key of that map as an error, so the desktop renderer's notice
(`Some providers did not answer (23).`) accused all 23 providers on the operator's
own catalogue, including the two aggregators whose 452 rows were on screen, while
discovery had reported nothing failed at all.

The rule that decides which statuses become errors now lives in
`ProviderController.catalogue_failures` (its own cases are in
`tests/unit/providers/test_catalogue_failures.py`). These tests keep the ROUTE's
half of the contract — the map's shape on the wire, and that the route publishes
exactly what the rule returned — which is why `_StubController` DELEGATES to a real
controller over this test's isolated config and credential store instead of
re-implementing the rule: a second spelling of it here is how the statuses and the
failures came to disagree in the first place.

Two silences in particular are the route's to keep. The app's own mock host
(`test`) has no listing transport at all, so it cannot have failed to list; and a
local PRESET nobody configured (`ollama`) is the app's own default port with
nobody home — the operator's machine reported five of those, plus `test`, as six
providers that "did not answer".
"""

from pathlib import Path

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.providers.auth_store import AuthStore
from local_operator.providers.controller import (
    CATALOGUE_FAILURE_REASON,
    CatalogueEntry,
    ProviderController,
)
from local_operator.server.routes import auth, desktop_catalogues

TOKEN = "desktop-catalogue-errors-test-token"
pytestmark = pytest.mark.asyncio

#: Every environment key that would make one of these providers "engaged" through
#: ``resolve_env_key``, and so decide a case for a reason the case is not about.
_CREDENTIAL_ENV = (
    "OPENROUTER_API_KEY",
    "DEEPSEEK_API_KEY",
    "RADIENT_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OLLAMA_API_KEY",
)

#: Seeded credentials, so the ENGAGEMENT axis is satisfied for the cloud cases: a
#: provider the user never connected is excused before the status rule is reached,
#: and a case about the status rule must not pass for that reason instead.
CREDENTIALED = ("openrouter", "deepseek")

#: The local endpoint the user pointed at. `vllm`'s failures are the user's to see
#: BECAUSE of this; `ollama`'s below are not, and that difference is the fix.
CONFIGURED_LOCAL = ("vllm", "http://127.0.0.1:8000/v1")

#: A MIXED status dict in the shape discovery really produces: two healthy
#: listings, a deliberate cache serve, a configured local server that failed, a
#: local preset nobody configured, the mock host, and a provider that was never
#: asked. Only the failed cloud fetch and the configured local failure belong in
#: `errors`.
MIXED_STATUSES = {
    "openrouter": "ok",
    "deepseek": "stale",
    "radient": "cached",
    "vllm": "stale",
    "ollama": "static",
    "test": "static",
    "anthropic": "unauthenticated",
}
CONTRIBUTING = {"openrouter", "radient"}
EXPECTED_ERROR_KEYS = {"deepseek", "vllm"}


class _StubController:
    """A REAL controller with only the live listing stubbed.

    ``catalogue_failures`` and ``usable_providers`` are the real methods, bound to
    a controller over this test's isolated config and credential store, so the
    route is exercised against the rule the desktop actually runs. ``live_catalogue``
    is the one thing worth stubbing: its statuses are what each case is CHOOSING,
    and obtaining them for real costs a provider round trip each.
    """

    def __init__(self, statuses: dict[str, str], contributing: set[str], store: AuthStore) -> None:
        self._real = ProviderController(store)
        self.statuses = statuses
        self.contributing = contributing
        self.closed = False

    async def live_catalogue(self, **_kwargs):
        return (
            [_entry(provider) for provider in sorted(self.contributing)],
            dict(self.statuses),
        )

    def catalogue_failures(
        self, entries: list[CatalogueEntry], statuses: dict[str, str]
    ) -> dict[str, str]:
        return self._real.catalogue_failures(entries, statuses)

    def usable_providers(self):
        return self._real.usable_providers()

    def close(self) -> None:
        self.closed = True
        self._real.close()


def _entry(provider: str) -> CatalogueEntry:
    """A minimal CatalogueEntry, distinguishable by its provider."""
    return CatalogueEntry(
        provider=provider,
        model_id="a-model",
        label="a-model",
        context_window=1000,
        input_price=0.0,
        output_price=0.0,
        connected=True,
    )


class _StubHost:
    def __init__(self, statuses: dict[str, str], contributing: set[str], store: AuthStore) -> None:
        self.statuses = statuses
        self.contributing = contributing
        self.store = store
        self.last: _StubController | None = None

    def controller(self) -> _StubController:
        self.last = _StubController(self.statuses, self.contributing, self.store)
        return self.last

    async def close(self) -> None:
        return None


@pytest_asyncio.fixture
async def catalogue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The route, with a fresh credential store and the one configured local server."""
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in _CREDENTIAL_ENV:
        monkeypatch.delenv(name, raising=False)
    store = AuthStore(tmp_path / "auth.db")
    for provider in CREDENTIALED:
        store.upsert_credential(provider, {"key": f"key-for-{provider}", "source": "login"})
    manager = ConfigManager(tmp_path)
    manager.update_config({"providers": {CONFIGURED_LOCAL[0]: {"base_url": CONFIGURED_LOCAL[1]}}})
    app = FastAPI()
    app.include_router(auth.router)
    app.include_router(desktop_catalogues.router)
    app.state.config_manager = manager
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield client, app, store
    if getattr(app.state, "desktop_auth", None):
        await app.state.desktop_auth.close()
    store.close()


async def _models_with(
    client: AsyncClient,
    app: FastAPI,
    store: AuthStore,
    statuses: dict[str, str],
    contributing: set[str] | None = None,
):
    """GET `/v1/desktop/models?live=true` against a controller with `statuses`."""
    app.state.desktop_auth = _StubHost(statuses, contributing or set(), store)
    return (await client.get("/v1/desktop/models?live=true")).json()["result"]


async def test_errors_names_only_providers_that_produced_no_listing(catalogue):
    """THE reproduction: a mixed status dict must accuse exactly the failed two.

    Before the fix this returned all seven keys and the renderer rendered
    `Some providers did not answer (7)`. With 23 real providers that is the
    operator's `(23)` note.
    """
    client, app, store = catalogue
    body = await _models_with(client, app, store, MIXED_STATUSES, CONTRIBUTING)
    assert set(body["errors"]) == EXPECTED_ERROR_KEYS
    # The wire shape is unchanged: a provider -> reason map the renderer reads
    # by key. Nothing here may become a list of reasons or a nested object.
    assert all(reason == CATALOGUE_FAILURE_REASON for reason in body["errors"].values())


async def test_an_unconfigured_local_preset_is_never_accused(catalogue):
    """The app's own default port is not a provider the user has (the 6-name bug).

    `ollama`'s preset is `http://localhost:11434/v1` and a machine without Ollama
    running reports `static`/0 rows for it. That is not a listing that failed —
    nothing was ever asked of the user's Ollama, because the user never pointed
    the app at one.
    """
    client, app, store = catalogue
    body = await _models_with(client, app, store, {"ollama": "static"})
    assert body["errors"] == {}


async def test_a_provider_with_no_listing_transport_is_never_accused(catalogue):
    """`test` is the mock wire: the registry says it cannot be listed at all."""
    client, app, store = catalogue
    body = await _models_with(client, app, store, {"test": "static"})
    assert body["errors"] == {}


async def test_a_configured_local_server_that_failed_is_accused(catalogue):
    """The other half of the axis: once pointed somewhere, its failure is the user's."""
    client, app, store = catalogue
    body = await _models_with(client, app, store, {"vllm": "empty"})
    assert set(body["errors"]) == {"vllm"}


async def test_a_static_provider_that_contributed_rows_is_never_accused(catalogue):
    """`static` with rows on screen is an ANSWER, not a failure.

    A provider that bundles registry rows reports `static` whenever its live
    listing is unavailable, and those rows are exactly what the user asked to
    see. Accusing it would be the D1 over-report in miniature.
    """
    client, app, store = catalogue
    body = await _models_with(client, app, store, {"openrouter": "static"}, {"openrouter"})
    assert body["errors"] == {}


async def test_a_static_provider_that_contributed_nothing_is_an_error(catalogue):
    """R1-4/Q1: the fetch failed with nothing cached on a row-less provider.

    Ten registry providers bundle ZERO static rows (the aggregators, ``typesafe``,
    ``test``, and the five local_setup ones), so a first-run 401 or dead host
    yields `static` with no rows — the operator's original "refresh failed"
    state. The rule must name it; `FAILED_LISTING_STATUSES` alone would go
    silent here where the base named the provider.
    """
    client, app, store = catalogue
    body = await _models_with(client, app, store, {"openrouter": "static"}, set())
    assert set(body["errors"]) == {"openrouter"}


async def test_a_cached_or_unauthenticated_provider_is_never_accused(catalogue):
    """Each non-failure status, one at a time, keeps the map EMPTY.

    `static` is deliberately absent from this list: its verdict needs the second
    fact (did it contribute rows), and it is covered by the two tests above.
    """
    client, app, store = catalogue
    for status in ("ok", "cached", "unauthenticated"):
        body = await _models_with(client, app, store, {"openrouter": status}, set())
        assert body["errors"] == {}, f"{status!r} was reported as a failed listing"


async def test_a_stale_or_empty_listing_is_the_only_thing_reported(catalogue):
    client, app, store = catalogue
    stale = await _models_with(client, app, store, {"openrouter": "stale"})
    assert set(stale["errors"]) == {"openrouter"}
    empty = await _models_with(client, app, store, {"openrouter": "empty"})
    assert set(empty["errors"]) == {"openrouter"}


async def test_the_map_is_wired_from_discovery_vocabulary_not_a_local_copy(catalogue):
    """The rule and discovery must share ONE statement of what a failure is.

    A second spelling of the set is how the statuses and the failures came to
    disagree in the first place, so the constant is imported, not re-typed.
    `static` is excluded from this cross-check deliberately: the rule resolves it
    with a second fact, so only the unambiguous statuses map straight through.
    """
    from local_operator.model.discovery import FAILED_LISTING_STATUSES

    assert FAILED_LISTING_STATUSES == {"stale", "empty"}
    for provider, status in MIXED_STATUSES.items():
        if status == "static":
            continue
        expected = status in FAILED_LISTING_STATUSES
        assert (provider in EXPECTED_ERROR_KEYS) == expected
