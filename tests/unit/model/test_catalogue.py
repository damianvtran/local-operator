"""The model catalogue: real windows for aggregators, and never a hard failure.

Aggregators (OpenRouter, Radient) route hundreds of models, so their static
registry entry is a placeholder carrying ``context_window = -1``. Before the
catalogue existed, ``configure_model`` silently fell back to a 128,000 window
and zero prices for every one of them, which is wrong in a way that changes
runtime behaviour rather than just a display: auto compaction derives its
threshold from the window, so a 1M-context model summarised its history at
~102k instead of ~800k.

These tests pin both halves of the contract: the numbers come from the
catalogue when it is reachable, and NOTHING in the path can stop a session
from starting when it is not.
"""

from __future__ import annotations

import json
import time

import pytest

from local_operator.model import catalogue
from local_operator.model.catalogue import cached_listing


def _payload(model_id: str = "vendor/model", window: int = 1_000_000) -> dict:
    return {
        "data": [
            {
                "id": model_id,
                "name": model_id,
                "description": "d",
                "context_length": window,
                "pricing": {"prompt": "0.0000001", "completion": "0.0000002"},
            }
        ]
    }


# -- cache mechanics ---------------------------------------------------------


def test_a_fresh_cache_is_used_without_fetching(tmp_path) -> None:
    calls = []

    def fetch():
        calls.append(1)
        return _payload()

    first = cached_listing("openrouter", fetch, cache_dir=tmp_path)
    second = cached_listing("openrouter", fetch, cache_dir=tmp_path)
    assert first == second == _payload()
    assert len(calls) == 1, "the second call must be served from disk"


def test_an_expired_cache_refetches(tmp_path) -> None:
    cached_listing("openrouter", lambda: _payload(window=1), cache_dir=tmp_path)
    fresh = cached_listing("openrouter", lambda: _payload(window=2), ttl_s=-1, cache_dir=tmp_path)
    assert fresh["data"][0]["context_length"] == 2


def test_a_failed_fetch_prefers_a_stale_cache_over_nothing(tmp_path) -> None:
    """A window that is a week old beats one that is wrong by 8x."""
    cached_listing("openrouter", lambda: _payload(window=999), cache_dir=tmp_path)

    def boom():
        raise RuntimeError("no network")

    got = cached_listing("openrouter", boom, ttl_s=-1, cache_dir=tmp_path)
    assert got is not None
    assert got["data"][0]["context_length"] == 999


def test_a_failed_fetch_with_no_cache_returns_none(tmp_path) -> None:
    """The one case where the caller must keep its static fallback — and it
    must be a None, not an exception, or an offline start dies here."""

    def boom():
        raise RuntimeError("no network")

    assert cached_listing("openrouter", boom, cache_dir=tmp_path) is None


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        "[]",  # valid JSON, wrong shape
        json.dumps({"payload": {"data": []}}),  # no fetched_at
        json.dumps({"fetched_at": "nope", "payload": {}}),  # unparseable stamp
    ],
    ids=["truncated", "wrong-type", "no-timestamp", "bad-timestamp"],
)
def test_an_unusable_cache_is_treated_as_absent(tmp_path, content: str) -> None:
    """A half-written file from a killed process must not raise; this is an
    optimisation store, so the only correct response is to refetch."""
    path = tmp_path / "openrouter.models.json"
    path.write_text(content, encoding="utf-8")
    assert cached_listing("openrouter", lambda: _payload(), cache_dir=tmp_path) == _payload()


def test_the_cache_write_is_atomic(tmp_path) -> None:
    """Two sessions can start at once, so a reader must never observe a
    partial document. Assert the temp file is not left behind and the final
    document parses."""
    cached_listing("openrouter", lambda: _payload(), cache_dir=tmp_path)
    assert not list(tmp_path.glob("*.tmp")), "temp file leaked"
    raw = json.loads((tmp_path / "openrouter.models.json").read_text(encoding="utf-8"))
    assert raw["payload"] == _payload()
    assert time.time() - raw["fetched_at"] < 60


def test_an_unwritable_cache_dir_still_returns_the_payload(tmp_path, monkeypatch) -> None:
    """Losing the cache is a performance problem; failing the call is not."""

    def refuse(*_args, **_kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(catalogue.Path, "mkdir", refuse)
    assert cached_listing("openrouter", lambda: _payload(), cache_dir=tmp_path) == _payload()


# -- integration with configure_model ---------------------------------------


def test_configure_model_takes_the_window_from_the_catalogue(monkeypatch, tmp_path) -> None:
    """The defect this module exists for: an aggregator's real window reaching
    the ModelSpec instead of the 128k placeholder."""
    from local_operator.clients.openrouter import OpenRouterListModelsResponse
    from local_operator.model import configure as configure_mod

    class FakeClient:
        def list_models(self):
            return OpenRouterListModelsResponse.model_validate(
                _payload("vendor/big", window=1_000_000)
            )

    monkeypatch.setattr(
        configure_mod,
        "_catalogue_source",
        lambda _p: (FakeClient(), OpenRouterListModelsResponse),
    )
    monkeypatch.setattr(catalogue, "default_cache_dir", lambda: tmp_path)

    config = configure_mod.configure_model(hosting="openrouter", model_name="vendor/big")
    assert config.spec.context_window == 1_000_000
    assert config.info.input_price == pytest.approx(0.1)
    assert config.info.output_price == pytest.approx(0.2)


@pytest.mark.parametrize(
    "break_it",
    ["client-raises", "no-such-model", "payload-drift"],
)
def test_configure_model_never_fails_on_a_bad_catalogue(
    monkeypatch, tmp_path, break_it: str
) -> None:
    """Every catalogue failure mode degrades to the static fallback. A session
    MUST start with no network, an unknown model id, or a drifted schema."""
    from local_operator.clients.openrouter import OpenRouterListModelsResponse
    from local_operator.model import configure as configure_mod
    from local_operator.model.configure import UNKNOWN_CONTEXT_WINDOW

    class FakeClient:
        def list_models(self):
            if break_it == "client-raises":
                raise RuntimeError("no network")
            if break_it == "payload-drift":
                return OpenRouterListModelsResponse.model_validate({"data": []})
            return OpenRouterListModelsResponse.model_validate(_payload("other/model"))

    monkeypatch.setattr(
        configure_mod,
        "_catalogue_source",
        lambda _p: (FakeClient(), OpenRouterListModelsResponse),
    )
    monkeypatch.setattr(catalogue, "default_cache_dir", lambda: tmp_path)

    config = configure_mod.configure_model(hosting="openrouter", model_name="vendor/asked-for")
    assert config.spec.context_window == UNKNOWN_CONTEXT_WINDOW


def test_a_provider_with_a_real_static_entry_never_consults_the_catalogue(monkeypatch) -> None:
    """The catalogue is for placeholder entries only. Anthropic's window is in
    the registry, so touching the network for it would add latency to every
    session start for nothing."""
    from local_operator.model import configure as configure_mod

    def explode(_provider):
        raise AssertionError("the catalogue must not be consulted for anthropic")

    monkeypatch.setattr(configure_mod, "_catalogue_source", explode)
    config = configure_mod.configure_model(
        hosting="anthropic", model_name="claude-sonnet-4-20250514"
    )
    assert config.spec.context_window == 200_000


@pytest.mark.parametrize("provider", ["openrouter", "radient"])
def test_a_keyless_install_does_not_raise_building_a_client(monkeypatch, provider: str) -> None:
    """`OpenRouterClient` raises RuntimeError on an empty key IN ITS
    CONSTRUCTOR, and `RadientClient` requires a base_url. The first version of
    the catalogue got both wrong and turned a metadata optimisation into "the
    CLI will not start without a key" — caught by the existing default-names
    tests, which is exactly the failure this pins.
    """
    from local_operator.model import configure as configure_mod

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("RADIENT_API_KEY", raising=False)

    # Must answer, not raise: None (no client) or a usable (client, model) pair.
    source = configure_mod._catalogue_source(provider)
    assert source is None or len(source) == 2


@pytest.mark.parametrize(
    "hosting,model",
    [("openrouter", "google/gemini-2.0-flash-001"), ("radient", "auto")],
)
def test_configure_model_survives_a_keyless_install(monkeypatch, hosting: str, model: str) -> None:
    """End of the same defect: the whole call must still return a usable
    configuration with no credentials anywhere."""
    from local_operator.model.configure import UNKNOWN_CONTEXT_WINDOW, configure_model

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("RADIENT_API_KEY", raising=False)

    config = configure_model(hosting=hosting, model_name=model)
    assert config.spec.context_window >= UNKNOWN_CONTEXT_WINDOW
