"""Unit tests for ProviderController — the TUI's provider/model/usage facade.

Credential/login behavior is exercised against a fake auth store so no real
SQLite or network is needed; usage dispatch is tested against a canned
httpx transport.
"""

from __future__ import annotations

import types
from typing import Any

import pytest
import httpx

from local_operator.providers.controller import ProviderController


class FakeAuthStore:
    """Minimal stand-in for the AuthStore credential surface."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self._next_id = 1
        self.api_keys: dict[str, str] = {}
        self.oauth: dict[str, object] = {}

    def list_credentials(self, provider=None):
        rows = (
            list(self.rows)
            if provider is None
            else [r for r in self.rows if r["provider"] == provider]
        )
        return [types.SimpleNamespace(**r) for r in rows]

    def upsert_credential(self, provider, credential):
        row = {
            "id": self._next_id,
            "provider": provider,
            "credential_type": "api_key" if "refresh" not in credential else "oauth",
            "data": credential,
            "identity_key": credential.get("email"),
        }
        self._next_id += 1
        self.rows.append(row)
        return row

    def delete_credentials_for_provider(self, provider, disabled_cause="logged-out"):
        before = len(self.rows)
        self.rows = [r for r in self.rows if r["provider"] != provider]
        return before - len(self.rows)

    async def get_oauth_access(self, provider):
        return self.oauth.get(provider)

    async def get_api_key(self, provider):
        return self.api_keys.get(provider)


@pytest.fixture
def store() -> FakeAuthStore:
    return FakeAuthStore()


@pytest.fixture
def controller(store):
    return ProviderController(store, login_callbacks=None)


def test_login_provider_listing(controller) -> None:
    ids = {p.id for p in controller.login_providers()}
    assert {"openai", "anthropic", "openrouter", "alibaba", "google", "deepseek"} <= ids


def test_has_any_credential(controller, store) -> None:
    assert controller.has_any_credential("openrouter") is False
    store.upsert_credential("openrouter", {"key": "sk-or-1", "source": "login"})
    assert controller.has_any_credential("openrouter") is True


def test_credential_alias_resolves_storage_id(controller, store) -> None:
    # xai-oauth stores under xai; has_any_credential("xai-oauth") must see it.
    store.upsert_credential("xai", {"access": "tok", "refresh": "ref"})
    assert controller.has_any_credential("xai-oauth") is True


def test_resolve_model_openrouter(controller) -> None:
    spec = controller.resolve_model("openrouter", "deepseek/deepseek-chat")
    assert spec.provider == "openrouter"
    assert spec.model_id == "deepseek/deepseek-chat"


def test_resolve_model_unknown_provider_does_not_raise(controller) -> None:
    # build_model_spec tolerates an unknown provider (no definition): it
    # produces a spec with a null base_url rather than raising.
    spec = controller.resolve_model("nonsense", "x")
    assert spec.provider == "nonsense"


@pytest.mark.asyncio
async def test_logout_removes_and_reports(controller, store) -> None:
    store.upsert_credential("openrouter", {"key": "k", "source": "login"})
    msg = await controller.logout("openrouter")
    assert "1 credential" in msg
    assert store.rows == []


@pytest.mark.asyncio
async def test_logout_unknown_provider_raises(controller) -> None:
    with pytest.raises(ValueError):
        await controller.logout("nonsense")


@pytest.mark.asyncio
async def test_logout_no_credentials_raises(controller) -> None:
    with pytest.raises(ValueError):
        await controller.logout("deepseek")


@pytest.mark.asyncio
async def test_fetch_usage_never_raises(controller) -> None:
    # Unknown/unsupported provider id -> clean empty list, no exception.
    reports = await controller.fetch_usage(["deepseek", "nonsense"])
    assert reports == []


@pytest.mark.asyncio
async def test_fetch_one_no_credential_returns_none(controller) -> None:
    # No stored credential and no api key -> None, not a crash.
    async with httpx.AsyncClient() as client:
        result = await controller._fetch_one(client, "openrouter")
    assert result is None


def test_usage_enabled_provider_ids(controller) -> None:
    ids = controller.usage_enabled_providers()
    assert "openrouter" in ids
    assert "deepseek" in ids
    assert "zai" not in ids, "unreachable: no ProviderDefinition exists for it"
    assert ids == sorted(ids)


# -- catalogue ---------------------------------------------------------------


def test_an_env_key_counts_as_a_usable_credential(controller, monkeypatch) -> None:
    """A key in the environment is what the stream-time cascade resolves, so a
    session started that way runs perfectly. Reporting the provider as needing a
    login was both wrong and unactionable — there is no login to perform."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert not controller.is_usable("openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    assert controller.is_usable("openrouter")
    assert not controller.has_any_credential("openrouter"), "still nothing STORED"


def test_a_keyless_provider_is_usable_with_no_credential_at_all(controller) -> None:
    """Which is the whole point of running a local server."""
    assert controller.is_usable("ollama")


def test_the_static_catalogue_needs_no_network_and_carries_real_models(controller) -> None:
    """The picker paints from this on the keystroke that opens it, so it must be
    synchronous and must already be useful."""
    entries = controller.static_catalogue()
    assert entries
    selectors = {entry.selector for entry in entries}
    assert "anthropic/claude-opus-4-20250514" in selectors
    assert all(entry.provider and entry.model_id for entry in entries)


def test_an_unknown_price_is_not_reported_as_free(controller, monkeypatch) -> None:
    """The picker renders a genuine pair of zeroes as `free`, so an unknown price
    passed through as zero would advertise a paid model as costing nothing.
    Anthropic makes this immediate: its listing carries no pricing at all, so every
    model it discovers that we did not already ship would read `free`."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    entries = {entry.selector: entry for entry in controller.static_catalogue()}
    priced = entries["anthropic/claude-opus-4-20250514"]
    assert priced.input_price > 0

    # A provider whose registry rows carry no prices must report unknown (< 0),
    # never 0.0, unless it needs no credential at all.
    unknown = [e for e in entries.values() if e.input_price <= 0]
    for entry in unknown:
        definition = controller.provider(entry.provider)
        keyless = definition is not None and definition.allows_missing_api_key
        assert entry.input_price == (0.0 if keyless else -1.0), entry


def test_a_reseller_is_flagged_so_the_picker_can_prefer_the_direct_route(
    controller, monkeypatch
) -> None:
    """`openrouter/anthropic/claude-opus-5` and `anthropic/claude-opus-5` are the
    same model; the picker ranks the direct one first and needs this flag to know
    which is which."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    for entry in controller.static_catalogue():
        assert entry.aggregated == (entry.provider in ("openrouter", "radient")), entry
