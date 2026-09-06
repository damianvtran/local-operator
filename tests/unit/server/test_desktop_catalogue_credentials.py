"""Credential state on the desktop catalogue route, across thread boundaries.

Design round 2 (D18) reported `/v1/desktop/models` returning 117 models all
`connected: true` on a backend whose `/v1/auth/providers` reported a single
credential. The two surfaces already shared one authority
(`ProviderController.usable_providers`), so a "unify the predicate" fix would
have changed nothing: the divergence was that the route called the catalogue
through `asyncio.to_thread`, and the AuthStore's sqlite connection belongs to
the thread that created it. From a worker it raised `sqlite3.ProgrammingError`,
which `usable_providers()` swallowed as "store unreadable" -- and an unreadable
store deliberately degrades to "show everything as connected", so a bug wearing
the costume of a designed degradation reached a design review.

These tests exercise the ROUTE. A controller-level test on the main thread
passed throughout and proved nothing.
"""

from pathlib import Path

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.credentials import CredentialManager
from local_operator.server.routes import auth, desktop_catalogues

TOKEN = "desktop-catalogue-test-token"
pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def catalogue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    # A provider env key in the ambient environment would make every assertion
    # here depend on the developer's shell: `usable_providers` reads the
    # environment as well as the store.
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "XAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
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


async def _connected_providers(client: AsyncClient) -> set[str]:
    body = (await client.get("/v1/desktop/models")).json()["result"]
    return {row["provider"] for row in body["models"] if row["connected"]}


async def test_catalogue_reports_only_providers_with_a_credential(catalogue):
    """One stored credential must yield exactly one connected non-local provider."""
    client, _ = catalogue
    assert (
        await client.put("/v1/auth/providers/deepseek/key", json={"value": "sk-test-not-real"})
    ).status_code == 200

    body = (await client.get("/v1/desktop/models")).json()["result"]
    connected = {row["provider"] for row in body["models"] if row["connected"]}
    # Local providers need no credential at all, so they are legitimately usable;
    # the claim under test is that no OTHER provider is called connected.
    remote_connected = {
        provider
        for provider in connected
        if provider not in {"ollama", "lmstudio", "vllm", "llamacpp", "openai-compatible"}
    }
    assert remote_connected == {"deepseek"}, remote_connected
    assert any(not row["connected"] for row in body["models"]), (
        "every model is connected on a store with one credential; the catalogue "
        "is reporting the unreadable-store degradation instead of real state."
    )

    # The other surface, same backend, same store: the two must agree.
    providers = (await client.get("/v1/auth/providers")).json()["result"]["providers"]
    assert {row["id"] for row in providers if row["has_credential"]} == {"deepseek"}


async def test_no_credentials_means_no_remote_provider_is_connected(catalogue):
    client, _ = catalogue
    connected = await _connected_providers(client)
    assert not (
        connected - {"ollama", "lmstudio", "vllm", "llamacpp", "openai-compatible", "test", "mock"}
    ), connected


async def test_credentials_known_cannot_contradict_the_rows_beside_it(catalogue):
    """`credentials_known: true` alongside all-connected is an impossible pairing.

    The two were computed on different threads from one store: the flag on the
    event loop (where the connection works, so it saw a real set) and the rows
    in a worker (where the same read raised and degraded to "everything is
    connected"). No single store state can produce both, which makes this a
    cheap invariant to hold the whole class of defect down.
    """
    client, _ = catalogue
    body = (await client.get("/v1/desktop/models")).json()["result"]
    everything_connected = all(row["connected"] for row in body["models"])
    assert body["models"]
    assert not (body["credentials_known"] and everything_connected), (
        "credentials_known is true while every model claims to be connected; "
        "these were read from one store and cannot both be true."
    )


async def test_catalogue_is_identical_on_a_worker_thread(catalogue):
    """The invariant that survives someone reintroducing a thread hop.

    Pins the property rather than the current call shape: whatever thread the
    catalogue is built on, it must report the same connected set.
    """
    import asyncio

    client, app = catalogue
    assert (
        await client.put("/v1/auth/providers/deepseek/key", json={"value": "sk-test-not-real"})
    ).status_code == 200

    controller = app.state.desktop_auth.controller()
    try:
        on_loop = controller.initial_catalogue()
        on_worker = await asyncio.to_thread(controller.initial_catalogue)
    finally:
        controller.close()

    assert {row.provider for row in on_loop if row.connected} == {
        row.provider for row in on_worker if row.connected
    }


async def test_a_programming_error_is_not_reported_as_an_unreadable_store():
    """`usable_providers` degrades for a locked store, but never for API misuse.

    The catch-all that returned None for any exception is what let the thread
    bug masquerade as a designed degradation. A locked store must still degrade;
    a ProgrammingError must surface.
    """
    import sqlite3
    import tempfile

    from local_operator.providers.auth_store import AuthStore
    from local_operator.providers.controller import ProviderController

    def raising(error: Exception):
        def boom(provider=None, include_disabled=False):
            raise error

        return boom

    root = Path(tempfile.mkdtemp())
    store = AuthStore(root / "auth.db")
    try:
        controller = ProviderController(store)
        # A locked database is an environment fact; it must still degrade.
        store.list_credentials = raising(  # type: ignore[assignment]
            sqlite3.OperationalError("database is locked")
        )
        assert controller.usable_providers() is None

        # Misuse of the sqlite API is a BUG. Reported as "unreadable store" it
        # became "every provider is connected" on a machine with no credentials.
        store.list_credentials = raising(  # type: ignore[assignment]
            sqlite3.ProgrammingError(
                "SQLite objects created in a thread can only be used in that same thread."
            )
        )
        with pytest.raises(sqlite3.ProgrammingError):
            controller.usable_providers()
    finally:
        store.close()
