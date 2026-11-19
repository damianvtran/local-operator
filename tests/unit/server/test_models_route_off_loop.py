"""``GET /v1/models`` must not hold the event loop while a provider listing hangs.

The OpenRouter and Radient branches called the clients' blocking
``list_models()`` (a bare ``requests.get``) directly inside the ``async`` route.
One upstream that accepted the connection and never answered -- a TLS-inspecting
proxy or antivirus, a flaky network -- froze every route on the server, ``/health``
included, until the OS dropped the socket. The desktop app polls ``/health`` and
refreshes models right after sign-in, so users saw "Server is offline" for minutes.

These tests park a fake ``list_models`` on a ``threading.Event`` and assert that
``/health`` is answered WHILE the listing is still parked. They wait on events,
not the clock: the only timeout is the fake's own safety release, which exists so
that a regression (the call back on the loop) fails with a clear assertion instead
of hanging the worker forever.
"""

import asyncio
import threading
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from local_operator.clients.openrouter import (
    OpenRouterClient,
    OpenRouterListModelsResponse,
)
from local_operator.clients.radient import RadientClient, RadientListModelsResponse
from local_operator.env import get_env_config
from local_operator.server.app import app

pytestmark = pytest.mark.asyncio

# Long enough that a loop-blocking regression is unambiguous (``/health`` cannot
# answer before it), short enough that such a regression fails the test promptly.
SAFETY_RELEASE_S = 10.0


class ParkedListing:
    """A ``list_models`` stand-in that blocks its calling thread until released."""

    def __init__(self, empty: Any) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.returned = threading.Event()
        self.thread_name: str | None = None
        self._empty = empty

    def __call__(self, *_args: Any, **_kwargs: Any) -> Any:
        self.thread_name = threading.current_thread().name
        self.entered.set()
        self.release.wait(SAFETY_RELEASE_S)
        self.returned.set()
        return self._empty


@pytest.fixture
def env_config_state():
    previous = getattr(app.state, "env_config", None)
    app.state.env_config = get_env_config()
    try:
        yield
    finally:
        app.state.env_config = previous


async def _health_answers_while_listing_parked(provider: str, parked: ParkedListing) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://127.0.0.1:1111"
    ) as client:
        models_task = asyncio.create_task(client.get(f"/v1/models?provider={provider}"))
        try:
            # Wait for the route to actually reach the upstream call. to_thread runs
            # it on a worker, so poll the Event from a thread rather than the loop.
            entered = await asyncio.to_thread(parked.entered.wait, SAFETY_RELEASE_S)
            assert entered, "the route never called list_models()"

            health = await client.get("/health")

            assert health.status_code == 200
            # The discriminating assertion: /health was answered while the listing
            # was still parked. On the loop, /health could only run after the fake
            # gave up waiting, by which point `returned` is set.
            assert not parked.returned.is_set(), (
                "/health was only answered after list_models() returned: the "
                "blocking call is running on the event loop again"
            )
            assert parked.thread_name != threading.main_thread().name
        finally:
            parked.release.set()
        models = await models_task
    assert models.status_code == 200


async def test_radient_listing_does_not_block_health(
    monkeypatch, env_config_state, mock_credential_manager
):
    parked = ParkedListing(RadientListModelsResponse(data=[]))
    monkeypatch.setattr(RadientClient, "list_models", parked)

    async def credential(*_args: Any, **_kwargs: Any) -> SecretStr:
        return SecretStr("synthetic-radient-key")

    monkeypatch.setattr(
        "local_operator.providers.radient_credentials.resolve_radient_credential", credential
    )

    await _health_answers_while_listing_parked("radient", parked)


async def test_openrouter_listing_does_not_block_health(
    monkeypatch, env_config_state, mock_credential_manager
):
    parked = ParkedListing(OpenRouterListModelsResponse(data=[]))
    monkeypatch.setattr(OpenRouterClient, "list_models", parked)
    monkeypatch.setattr(
        "local_operator.providers.registry.provider_env_key",
        lambda provider, **_kw: "synthetic-openrouter-key" if provider == "openrouter" else None,
    )

    await _health_answers_while_listing_parked("openrouter", parked)
