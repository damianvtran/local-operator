"""``GET /v1/desktop/models?scope=usable|all``: the access filter on the wire.

The desktop picker asks this route for the models a user can actually run, and
before this change the route had no way to say "these rows are the ones with a
credential" — the renderer grouped by ``connected`` and never filtered, so a
sign-in-less user's list led with ~445 Radient rows they could not use. The
filter itself is the picker's own predicate (``providers.catalogue``); these
tests keep the ROUTE's half of the contract: the parameter's default, the two
additive response fields, the current-model exemption's interaction with the
count, and the unreadable-store degradation that must never read as "you own no
models".

Everything is synthetic: credentials are written through the real
``PUT /v1/auth/providers/<id>/key`` route into this test's isolated config dir,
never a real user store.
"""

from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.server.routes import auth, desktop_catalogues

TOKEN = "desktop-catalogue-scope-test-token"
pytestmark = pytest.mark.asyncio

#: Providers that are usable with no credential at all. They carry no STATIC
#: model rows — a local server enumerates its models live — so the catalogue
#: this route serves from the registry never contains them; the set is here to
#: document why ``scope=usable`` on an empty store is legitimately empty rather
#: than suspicious.
_KEYLESS = {"ollama", "lmstudio", "vllm", "llamacpp", "openai-compatible", "test", "mock"}


@pytest_asyncio.fixture
async def catalogue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    # An ambient provider key would decide these cases from the developer's
    # shell: ``usable_providers`` reads the environment as well as the store.
    for name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "XAI_API_KEY",
        "OPENROUTER_API_KEY",
        "RADIENT_API_KEY",
        "OLLAMA_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
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


async def _models(client: AsyncClient, query: str = "") -> dict[str, Any]:
    response = await client.get(f"/v1/desktop/models{query}")
    assert response.status_code == 200, response.text
    return response.json()["result"]


async def test_the_parameter_defaults_to_all_so_old_clients_see_no_change(catalogue):
    """``scope`` absent — every shipped client today — must be byte-compatible.

    The filter is opt-in, and the two new fields are additive; a client that
    never sends the parameter keeps the whole catalogue and can ignore them.
    """
    client, _ = catalogue
    body = await _models(client)
    assert body["scope"] == "all"
    assert body["hidden"] == 0
    # Unfiltered: with no credentials stored, the majority of rows are
    # unconnected, and all of them are still listed.
    assert any(not row["connected"] for row in body["models"])
    assert len({row["provider"] for row in body["models"]}) > len(_KEYLESS)


async def test_scope_all_is_the_same_view_as_the_default(catalogue):
    client, _ = catalogue
    default = await _models(client)
    explicit = await _models(client, "?scope=all")
    assert explicit["models"] == default["models"]
    assert explicit["scope"] == "all"
    assert explicit["hidden"] == 0


async def test_scope_usable_keeps_only_rows_with_a_credential(catalogue):
    """One stored credential, and the filter admits exactly its rows — with the
    count of what it withheld."""
    client, _ = catalogue
    assert (
        await client.put("/v1/auth/providers/deepseek/key", json={"value": "sk-test-not-real"})
    ).status_code == 200

    everything = await _models(client)
    usable = await _models(client, "?scope=usable")

    assert usable["scope"] == "usable"
    assert usable["credentials_known"] is True
    providers = {row["provider"] for row in usable["models"]}
    assert providers == {"deepseek"}, providers
    assert all(
        row["connected"] for row in usable["models"]
    ), "a filtered row the user cannot run on would be the bug this filter exists to fix"
    assert usable["hidden"] == len(everything["models"]) - len(usable["models"])
    assert usable["hidden"] > 0


async def test_scope_usable_keeps_the_current_model_even_when_unusable(catalogue):
    """The session's own model is exempt, and the exemption is not counted.

    Dropping it would make a session running an unreachable model look
    unconfigured on the one surface that answers "what am I on" (the same
    exemption the TUI applies). ``current`` is the session's ``provider/id``.
    """
    client, _ = catalogue
    assert (
        await client.put("/v1/auth/providers/deepseek/key", json={"value": "sk-test-not-real"})
    ).status_code == 200
    everything = await _models(client)
    kept_without = await _models(client, "?scope=usable")
    # Derived from the unfiltered listing rather than hard-coded, so the case
    # survives a registry whose anthropic ids move.
    stranded = next(
        row["selector"]
        for row in everything["models"]
        if row["provider"] not in {"deepseek"} and row["provider"] not in _KEYLESS
    )
    kept_with = await _models(client, f"?scope=usable&current={stranded.replace('/', '%2F')}")

    assert stranded in {row["selector"] for row in kept_with["models"]}
    assert stranded not in {row["selector"] for row in kept_without["models"]}
    # ``hidden`` counts what was DROPPED, so the exempted row is not a hidden
    # row: the count is one lower with it kept, and the arithmetic matches the
    # rows returned either way (the picker's own convention).
    assert kept_with["hidden"] == kept_without["hidden"] - 1
    assert kept_with["hidden"] == len(everything["models"]) - len(kept_with["models"])


async def test_scope_usable_on_an_empty_store_withholds_every_registry_row(catalogue):
    """No credentials at all: the answer is empty, and it is a real answer.

    The static registry only carries cloud providers (a keyless local
    enumerates its models live and has no rows here), so with no credential
    stored there is genuinely nothing this user can run — the empty list is the
    filter working, not the unreadable-store degradation, which is what
    ``credentials_known: true`` beside it distinguishes.
    """
    client, _ = catalogue
    everything = await _models(client)
    body = await _models(client, "?scope=usable")
    assert body["models"] == []
    assert body["credentials_known"] is True
    assert body["hidden"] == len(everything["models"]) > 0


async def test_an_unreadable_store_filters_nothing_and_says_so(catalogue):
    """``usable=None`` is "cannot tell", and the wire must carry that honestly.

    An empty list would claim the user owns no models — precisely what the app
    failed to establish — so the degradation shows everything, ``hidden`` stays
    0 (nothing was withheld), and ``credentials_known: false`` is what tells the
    renderer not to badge the rows.
    """
    import sqlite3

    client, app = catalogue
    # The host is built lazily by the first authenticated request, so take the
    # unfiltered listing FIRST and break the store only after it exists.
    everything = await _models(client)

    def locked(provider=None, include_disabled=False):
        raise sqlite3.OperationalError("database is locked")

    app.state.desktop_auth.store.list_credentials = locked  # type: ignore[method-assign]
    body = await _models(client, "?scope=usable")
    assert body["credentials_known"] is False
    assert body["hidden"] == 0
    # Same rows, same order. ``connected`` is deliberately NOT compared: with
    # the store unreadable every row degrades to connected=True ("cannot tell"
    # must not present as "you own nothing"), which is the documented
    # listing degradation and not this filter's business.
    assert [row["selector"] for row in body["models"]] == [
        row["selector"] for row in everything["models"]
    ]


async def test_an_unknown_scope_is_refused(catalogue):
    """A typo must not silently mean one of the two views.

    ``Literal`` validation answers 422; a permissive parse would have made
    ``scope=usabl`` behave as ``all`` without saying anything.
    """
    client, _ = catalogue
    response = await client.get("/v1/desktop/models?scope=usabl")
    assert response.status_code == 422, response.text
