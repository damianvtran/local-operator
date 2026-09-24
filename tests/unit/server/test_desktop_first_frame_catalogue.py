"""The composer's inline ``/model `` list must offer a cached, provider-listed id.

WHY THIS TEST IS AT THE ROUTE AND NOT AT THE CONTROLLER.
``GET /v1/desktop/sessions/{id}/command-entities?command=model`` is the surface
behind the composer's ``/model `` argument list, and unlike the dialog it NEVER
goes live: its rows are ``ProviderController.initial_catalogue()`` and nothing
else. A model that exists only in the provider's own listing -- present in
Anthropic's ``/v1/models``, absent from the shipped registry -- therefore could
not be offered there even after the dialog had refetched it. That is the same
reported symptom on a sibling surface, so it is pinned on the wire, on an
isolated HOME and config dir, with the network POISONED: the id has to appear
with no fetch at all, since the whole point of a first frame is that it paints
before the network answers.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, AsyncIterator

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.model import discovery
from local_operator.server.routes import desktop_catalogues
from local_operator.server.utils.desktop_sessions import DesktopSessions

TOKEN = "desktop-first-frame-token"

#: An id the shipped registry does not carry, in the shape a provider listing
#: would hand it over. The registry's own anthropic rows are asserted below, so
#: the test cannot pass by the fixture having emptied them.
UNSHIPPED_ID = "claude-fable-6"

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def composer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[AsyncClient, str, Path]]:
    """The real route on an isolated HOME/config/cache, plus a synthetic session."""
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    # HOME is what redirects the CACHE root: `catalogue.default_cache_dir()`
    # derives it from the home directory, not from the config dir, so a run that
    # sets only LOCAL_OPERATOR_CONFIG_DIR plants its document where the route
    # will never look (AGENTS.md, "Isolating a run").
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "XAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    app = FastAPI()
    app.include_router(desktop_catalogues.router)
    app.state.config_manager = ConfigManager(tmp_path)
    pool = DesktopSessions(tmp_path)
    app.state.desktop_sessions = pool
    inputs = tmp_path / "workspace"
    inputs.mkdir(parents=True, exist_ok=True)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        session_id = await pool.create(str(inputs))
        yield client, session_id, tmp_path
    await pool.close()
    if getattr(app.state, "desktop_auth", None):
        await app.state.desktop_auth.close()


def _plant(cache_dir: Path, provider: str, *, ids: list[str]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{provider}.listing.json").write_text(
        json.dumps(
            {
                "fetched_at": time.time(),
                "payload": {
                    # The module's own capture version, so a capture bump reads
                    # as an unusable document instead of passing silently.
                    "capture": discovery.listing_capture_version(provider),
                    "models": [{"id": model_id, "context_window": 1_000_000} for model_id in ids],
                },
            }
        ),
        encoding="utf-8",
    )


def _poison_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def exploding(*_args: Any, **_kwargs: Any) -> list[Any]:
        raise AssertionError("the inline /model list is a first frame: it must not fetch")

    monkeypatch.setattr(discovery, "fetch_models", exploding)


async def test_the_inline_model_list_offers_a_cached_provider_listed_id(composer) -> None:
    client, session_id, root = composer
    _plant(root / ".local-operator" / "cache", "anthropic", ids=["claude-opus-5", UNSHIPPED_ID])

    response = await client.get(
        f"/v1/desktop/sessions/{session_id}/command-entities", params={"command": "model"}
    )

    assert response.status_code == 200, response.text
    body = response.json()["result"]
    assert body["command"] == "model"
    ids = {entity["model_id"] for entity in body["entities"]}
    assert UNSHIPPED_ID in ids, (
        "the composer's `/model ` list is built from initial_catalogue(), which "
        "read the shipped registry for a direct provider; a cached listing id "
        "must reach it without the dialog having to go live first"
    )
    # The registry rows ride along: an anthropic listing is a union, not an
    # authoritative set, so a cold-cache regression that dropped them would be a
    # different bug than the one this test is about -- and it would be invisible
    # without this line.
    assert "claude-opus-5" in ids


async def test_the_inline_model_list_fetches_nothing(composer, monkeypatch) -> None:
    """No network on this path, cached listing or not: it is a first frame."""
    client, session_id, root = composer
    _poison_network(monkeypatch)
    _plant(root / ".local-operator" / "cache", "anthropic", ids=[UNSHIPPED_ID])

    response = await client.get(
        f"/v1/desktop/sessions/{session_id}/command-entities", params={"command": "model"}
    )

    assert response.status_code == 200, response.text
    assert UNSHIPPED_ID in {entity["model_id"] for entity in response.json()["result"]["entities"]}
