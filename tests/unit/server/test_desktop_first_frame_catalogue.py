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
import yaml
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.model import discovery
from local_operator.server.routes import desktop_catalogues
from local_operator.server.utils.desktop_sessions import DesktopSessions

TOKEN = "desktop-first-frame-token"

#: An id the shipped registry does not carry, in the shape a provider listing
#: would hand it over. No shipped id is planted with it, so the registry-union
#: assertion below can only pass for the reason it states (review round 2, R2-5).
UNSHIPPED_ID = "claude-fable-6"

#: A shipped anthropic id the listing never mentions -- the union's proof.
SHIPPED_ID = "claude-opus-5"

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


def _hand_edit_local_endpoint(root: Path, provider: str, base_url: str) -> None:
    """A local provider's endpoint, written the way a person's editor writes it.

    Reachable on the shipped app: the settings editor validates through
    ``validate_endpoint_setting`` and the config FILE does not, which is how a
    ``localhost:notaport`` value gets in. The ``values`` wrapper is the real file
    shape -- a top-level ``providers`` block is silently ignored by
    ``ConfigManager._load_config``, and a test that wrote one would measure the
    cold path while claiming to measure this one.
    """
    (root / "config.yml").write_text(
        yaml.safe_dump({"values": {"providers": {provider: {"base_url": base_url}}}}),
        encoding="utf-8",
    )


async def test_the_inline_model_list_offers_a_cached_provider_listed_id(composer) -> None:
    client, session_id, root = composer
    _plant(root / ".local-operator" / "cache", "anthropic", ids=[UNSHIPPED_ID])

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
    # The listing never mentions this id, so its presence is the registry union
    # and nothing else: a regression that dropped the registry's rows on a cached
    # listing would fail HERE.
    assert SHIPPED_ID in ids


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


async def test_a_hand_edited_local_endpoint_still_answers_the_inline_list(composer) -> None:
    """The R2-1 regression, on the surface it was reported against.

    A local provider's stored endpoint that ``normalize_base_url`` rejects used to
    raise straight out of ``initial_catalogue`` -- reachable only once the frame
    began reading every provider through the cache reader -- and this route then
    answered **409 with zero rows** where it had answered 200 before, i.e. the
    composer's ``/model `` list went empty. The endpoint stays rejected (asserted
    below through ``resolve_base_url``), so the frame's rows are produced DESPITE
    it rather than because the config was never read.
    """
    from local_operator.providers.local import resolve_base_url

    client, session_id, root = composer
    _hand_edit_local_endpoint(root, "lmstudio", "http://localhost:notaport")
    with pytest.raises(ValueError):
        resolve_base_url("lmstudio")

    response = await client.get(
        f"/v1/desktop/sessions/{session_id}/command-entities", params={"command": "model"}
    )

    assert response.status_code == 200, response.text
    body = response.json()["result"]
    assert body["entities"], "the catalogue came back empty for one unconfigured local server"
    assert SHIPPED_ID in {entity["model_id"] for entity in body["entities"]}
