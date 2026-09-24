"""The desktop LIVE read must join the picker cadence, not discovery's 24 h TTL.

WHY THESE EXIST
---------------
``GET /v1/desktop/models?live=true`` sits behind the desktop picker's "Refresh
from providers" control: it is the user asking NOW. It nevertheless read the
disk-cached listing with discovery's default hard TTL (24 h), so a document
fetched in the morning answered an afternoon refresh without issuing a single
request -- and a model the provider published in between could not appear on the
very click meant to find it. The other two picker surfaces already ask for
:data:`PICKER_TTL_S` (15 minutes) for this reason, and this file pins the third
to the same choice:

* ``tui/app.py::_refresh_catalogue`` -> ``live_catalogue(ttl_s=PICKER_TTL_S)``
* ``mobile/daemon.py`` -> ``live_catalogue(ttl_s=PICKER_TTL_S, providers=...)``
* ``server/routes/desktop_catalogues.py::models`` -> now the same.

The document planted below is aged 16 minutes on purpose: it is YOUNGER than the
24 h default (so the route served it as-is, with zero requests) and OLDER than
the 15-minute cadence (so a route honouring the cadence must refetch it
synchronously). The companion cases hold the other two edges down: a document
INSIDE the cadence is still served without a request (the cadence bounds the
refetch, it does not remove it), and the non-live initial read never fetches at
all (it is the paint-on-open path and must stay I/O-free).

Everything is driven through the ROUTE with an isolated HOME, because the cache
root is derived from the home directory independently of
``LOCAL_OPERATOR_CONFIG_DIR`` -- see AGENTS.md "Isolating a run". A controller- or
discovery-level test would not see the TTL the route chose pass.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.model import discovery, registry
from local_operator.model.discovery import DiscoveredModel
from local_operator.server.routes import auth, desktop_catalogues

TOKEN = "desktop-catalogue-cadence-token"

#: Older than ``PICKER_TTL_S`` (15 min), younger than discovery's 24 h default.
PAST_THE_CADENCE_S = 16 * 60

#: Inside ``PICKER_TTL_S``: the document is fresh by the cadence's own standard.
INSIDE_THE_CADENCE_S = 5 * 60

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def catalogue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The desktop catalogue route on an isolated HOME, config dir and cache."""
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    # Redirecting the cache too: ``catalogue.default_cache_dir()`` follows HOME,
    # not LOCAL_OPERATOR_CONFIG_DIR, so the planted document is only visible to
    # the route when this is set.
    monkeypatch.setenv("HOME", str(tmp_path))
    # An ambient provider key would make the connected set depend on the
    # developer's shell (``usable_providers`` reads the environment as well as
    # the store), and the route only fetches providers with a credential.
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "XAI_API_KEY"):
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
        # Exactly one provider carries a credential, so exactly one listing can
        # be fetched and the fetch log has one possible entry.
        response = await client.put(
            "/v1/auth/providers/anthropic/key", json={"value": "sk-ant-test-not-real"}
        )
        assert response.status_code == 200, response.text
        yield client, tmp_path
    if getattr(app.state, "desktop_auth", None):
        await app.state.desktop_auth.close()


def _plant_anthropic_document(tmp_path: Path, *, age_s: float, ids: list[str]) -> None:
    """An Anthropic listing document of the given age, listing exactly ``ids``.

    The capture-2 shape the reader expects, under the document name
    ``discovery`` derives for this provider -- the same fixture the discovery
    tests use, planted in the HOME-derived cache root the route will read.
    """
    cache_dir = tmp_path / ".local-operator" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "anthropic.listing.json").write_text(
        json.dumps(
            {
                "fetched_at": time.time() - age_s,
                "payload": {
                    "capture": discovery.listing_capture_version("anthropic"),
                    "models": [{"id": model_id, "context_window": 1_000_000} for model_id in ids],
                },
            }
        ),
        encoding="utf-8",
    )


#: A model that exists only in the FETCH's answer: it is what the user is looking
#: for, and it cannot appear by accident, because the planted document omits it.
#:
#: It must ALSO be an id the shipped registry does not carry, and that is not
#: decoration: ``live_catalogue`` layers a provider's listing OVER the shipped
#: rows, so a curated id reaches the served set whether or not a fetch happened
#: -- the inside-the-cadence case would then pass for the wrong reason, and the
#: past-the-cadence case could not fail. This id was ``claude-opus-5-5`` while
#: that was unshipped; adding it as a curated row for the suggested-defaults
#: work silently converted the assertion into a claim about the registry. The
#: guard test at the end of this file holds the constraint open, and
#: ``test_desktop_first_frame_catalogue.UNSHIPPED_ID`` is the same shape for the
#: same reason.
RELEASED_SINCE_THE_DOCUMENT = "claude-fable-6"


class _Fetches:
    """Stands in for ``discovery.fetch_models``: records calls, serves the wire.

    Only ``anthropic`` is given rows. The route lists every provider whose
    credential is usable, and the keyless ones (local servers, the public
    aggregators) are fetched with no credential at all, so a stub that answered
    every provider with the same rows would leak
    :data:`RELEASED_SINCE_THE_DOCUMENT` into the catalogue through a provider
    this test says nothing about. An empty list for the rest is the honest
    stand-in: "this provider answered and listed nothing".
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, provider_id: str, **kwargs: object) -> list[DiscoveredModel]:
        self.calls.append(provider_id)
        if provider_id != "anthropic":
            return []
        return [
            DiscoveredModel(id="claude-opus-5", name="Claude Opus 5", context_window=1_000_000),
            DiscoveredModel(
                id=RELEASED_SINCE_THE_DOCUMENT,
                name="Claude Fable 6",
                context_window=1_000_000,
            ),
        ]


@pytest.fixture
def fetched(monkeypatch: pytest.MonkeyPatch) -> _Fetches:
    stub = _Fetches()
    monkeypatch.setattr(discovery, "fetch_models", stub)
    return stub


async def _live_ids(client: AsyncClient) -> tuple[set[str], dict[str, Any]]:
    body = (await client.get("/v1/desktop/models?live=true")).json()["result"]
    return {row["model_id"] for row in body["models"]}, body


async def test_a_live_read_refetches_a_document_past_the_picker_cadence(
    catalogue, fetched: _Fetches
) -> None:
    """The incident: a 16-minute document answered a user-requested refresh.

    Under the 24 h default this document is inside the SOFT TTL as well, so it
    was served with no request at all -- not even a background refresh -- and the
    model released since it was written was unreachable from the picker for the
    rest of the day. Asserting the fetch happened AND that its new id is in the
    answer covers both halves: a route could otherwise refetch and still drop the
    row.
    """
    client, tmp_path = catalogue
    _plant_anthropic_document(tmp_path, age_s=PAST_THE_CADENCE_S, ids=["claude-opus-5"])

    ids, body = await _live_ids(client)

    assert "anthropic" in fetched.calls, (
        "a 16-minute-old document answered ?live=true without a request; the "
        "live read is the user asking now and must not be served from a document "
        "older than the picker cadence"
    )
    assert RELEASED_SINCE_THE_DOCUMENT in ids
    assert body["source"] == "live"


async def test_a_live_read_inside_the_cadence_is_served_without_a_fetch(
    catalogue, fetched: _Fetches
) -> None:
    """The cadence BOUNDS the refetch; it must not remove it.

    A five-minute document is fresh by the cadence's own standard, so opening the
    picker repeatedly (or hammering refresh) must not re-list every provider on
    each click.
    """
    client, tmp_path = catalogue
    _plant_anthropic_document(tmp_path, age_s=INSIDE_THE_CADENCE_S, ids=["claude-opus-5"])

    ids, _body = await _live_ids(client)

    assert "anthropic" not in fetched.calls, (
        "a five-minute document is fresh by the cadence's own standard, so the "
        "live read must serve it: the cadence bounds the refetch, it does not "
        "turn every refresh into nine provider requests"
    )
    assert "claude-opus-5" in ids
    # Absent because no fetch happened AND because the registry can never supply
    # it; the sentinel's comment and the guard test below are what keep the
    # second half of that true.
    assert RELEASED_SINCE_THE_DOCUMENT not in ids


async def test_the_initial_read_never_fetches_even_a_document_past_the_cadence(
    catalogue, fetched: _Fetches
) -> None:
    """``live=false`` is the paint-on-open path: registry rows, disk only, no I/O.

    It is deliberately NOT given the cadence: it is the first frame the picker
    paints, and a synchronous listing there would put nine provider requests on
    the keystroke that opens it.
    """
    client, tmp_path = catalogue
    _plant_anthropic_document(tmp_path, age_s=PAST_THE_CADENCE_S, ids=["claude-opus-5"])

    body = (await client.get("/v1/desktop/models")).json()["result"]

    assert fetched.calls == []
    assert body["source"] == "initial"


async def test_the_fetch_only_sentinel_is_not_a_shipped_registry_row() -> None:
    """The sentinel must stay unshipped, or two assertions above go vacuous.

    A test rather than a comment because the failure is silent: a curated row
    puts the sentinel in the served set with no fetch at all, so the
    inside-the-cadence case keeps passing (for the registry's reason, not the
    cadence's) while the past-the-cadence case can no longer fail. That is
    exactly what shipping ``claude-opus-5-5`` did.
    """
    assert RELEASED_SINCE_THE_DOCUMENT not in registry.static_models("anthropic")
