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
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.model.discovery import DiscoveredModel
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


class _RealHost:
    """A host whose controller runs the REAL live read, with only the fetch stubbed.

    The host above stubs ``live_catalogue``, which is right for cases about the
    status rule but blind to what the live read did with a CREDENTIAL — the rows'
    ``connected`` flag, and whether the provider's own listing failed at all, are
    both products of it. Stubbing the transport is the narrowest way to reach
    those two answers without a provider round trip.
    """

    def __init__(self, store: AuthStore) -> None:
        self.store = store
        self.last: ProviderController | None = None

    def controller(self) -> ProviderController:
        self.last = ProviderController(self.store)
        return self.last

    async def close(self) -> None:
        if self.last is not None:
            self.last.close()


def _stub_transport(
    monkeypatch: pytest.MonkeyPatch,
    rows: dict[str, list[Any]],
    failed: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Stub discovery per provider AND the price projection (no network at all).

    ``rows`` are the providers whose listing answered; ``failed`` are the ones whose
    fetch fails with nothing cached (discovery's ``static``/0-row shape); every
    other provider answers ``unauthenticated`` — asked, no credential, silent by
    design — so a case is never decided by an unrelated status.

    ``seen`` is the unit-level instrument the credential question needs: the value
    handed to the fetch. A keyless request and a bogus-key request are
    indistinguishable on the wire — both 401 — so nothing downstream of this
    boundary can tell a working key from a missing one.
    """
    seen: dict[str, str | None] = {}

    def fake(provider_id: str, **kwargs: Any):
        seen[provider_id] = kwargs.get("api_key")
        if provider_id in rows:
            return list(rows[provider_id]), "ok"
        if provider_id in failed:
            return [], "static"
        return [], "unauthenticated"

    monkeypatch.setattr("local_operator.providers.controller.available_models", fake)
    monkeypatch.setattr("local_operator.model.prices.models_dev_providers", lambda **_kw: {})
    return seen


def _damage_secret_store(root: Path, payload: bytes) -> None:
    """A REAL secret store, then its database replaced by a non-store file.

    Order matters, and not for tidiness: damaging a store the rig never created
    only reaches ``open_store``'s "no secret store found" path, which the reader
    has always caught — so a rig that skips the initialization proves nothing and
    would pass on the broken revision too.
    """
    from local_operator.providers.registry import store_provider_key
    from local_operator.secrets.keys import store_path

    store_provider_key("OPENROUTER_API_KEY", "sk-or-from-settings", base=root)
    store_path(root).write_bytes(payload)


async def test_a_store_first_key_reaches_the_live_fetch_and_its_rows(
    catalogue, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """R2-1 on the wire: the key saved in Settings is USED, so the rows are connected.

    Before this fix the provider was fetched with ``api_key=None`` while the rule
    named it as failing — on every live read, indefinitely, with the working key
    resolvable on disk. The two halves asserted here are the half the wire can
    show (the rows carry ``connected: true`` where they carried ``false``) and the
    half only the boundary can (``seen`` — the key actually handed to the fetch).
    """
    from local_operator.providers.registry import store_provider_key

    client, app, store = catalogue
    store_provider_key("RADIENT_API_KEY", "sk-rad-from-settings", base=tmp_path)
    seen = _stub_transport(
        monkeypatch,
        {"radient": [DiscoveredModel(id="radient-1", name="Radient 1", context_window=8_000)]},
    )
    app.state.desktop_auth = _RealHost(store)

    body = (await client.get("/v1/desktop/models?live=true")).json()["result"]

    rows = [row for row in body["models"] if row["provider"] == "radient"]
    assert seen["radient"] == "sk-rad-from-settings"
    assert rows, "a provider whose listing answered must contribute its rows"
    assert all(
        row["connected"] for row in rows
    ), "a key the store knows about must make the rows connected, not anonymous"
    assert "radient" not in body["errors"], "it answered — there is nothing to report"


async def test_a_store_first_key_that_failed_is_named_for_a_real_reason(
    catalogue, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The other half: the same key, a listing that really failed, IS named.

    The report is about a failure the user can act on, and here the app did try —
    with their key — and the provider refused. That is the honest case the rule
    exists for, and it must not be silenced by the fix above.
    """
    from local_operator.providers.registry import store_provider_key

    client, app, store = catalogue
    store_provider_key("RADIENT_API_KEY", "sk-rad-from-settings", base=tmp_path)
    seen = _stub_transport(monkeypatch, {}, failed=("radient",))
    app.state.desktop_auth = _RealHost(store)

    body = (await client.get("/v1/desktop/models?live=true")).json()["result"]

    assert seen["radient"] == "sk-rad-from-settings", "the key was used, so the failure is real"
    assert body["errors"].get("radient") == CATALOGUE_FAILURE_REASON
    assert all(
        reason == CATALOGUE_FAILURE_REASON for reason in body["errors"].values()
    ), "the wire shape is a provider -> generic reason map"


async def test_a_damaged_secret_store_still_answers_the_live_read(
    catalogue, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Q2-1: an unreadable store degrades, it does not 500 the picker's Refresh.

    With ``secrets/store.db`` — a store this rig really created, holding a
    provider row — replaced by a non-store file, this route answered 500: the
    reader the union reaches (``stored_provider_env_keys`` through
    ``persisted_providers``) caught neither ``sqlite3.DatabaseError`` nor
    ``sqlite3.OperationalError``, and the route has no handler but ``finally:
    controller.close()``. The body below is what "cannot be read" must look like:
    a normal answer, with the store contributing nothing (exactly as if it held no
    provider rows at all) while the CONFIG axis keeps its verdicts — an
    unconfigured local preset stays silent even now, because the config is
    readable. The AUTH store here is untouched, so ``credentials_known`` stays
    true: this is the secret store's failure mode, not the auth store's.
    """
    from local_operator.providers.registry import stored_provider_env_keys

    client, app, store = catalogue
    _damage_secret_store(tmp_path, b"not-a-store-at-all")
    _stub_transport(monkeypatch, {})
    app.state.desktop_auth = _RealHost(store)

    response = await client.get("/v1/desktop/models?live=true")

    assert response.status_code == 200, "a damaged store must not become a 500"
    # The identity that IS the degradation: the reader answers exactly what it
    # answers for a store holding no provider rows, rather than raising.
    assert stored_provider_env_keys(tmp_path) == set()
    body = response.json()["result"]
    assert isinstance(body["errors"], dict)
    assert all(reason == CATALOGUE_FAILURE_REASON for reason in body["errors"].values())
    assert body["credentials_known"] is True, "the AUTH store is readable and untouched"
    assert (
        "ollama" not in body["errors"]
    ), "the config axis is still readable, so a preset nobody configured stays silent"
