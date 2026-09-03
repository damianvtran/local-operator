"""Unit tests for ProviderController — the TUI's provider/model/usage facade.

Credential/login behavior is exercised against a fake auth store so no real
SQLite or network is needed; usage dispatch is tested against a canned
httpx transport.
"""

from __future__ import annotations

import dataclasses
import types
from collections.abc import Iterator
from typing import Any

import httpx
import pytest

from local_operator.harness.types import ModelSpec
from local_operator.providers.controller import PICKER_TTL_S, ProviderController
from local_operator.providers.registry import get_provider_definition
from local_operator.providers.usage import UsageAmount, UsageLimit, UsageReport
from local_operator.providers.usage_cache import (
    USAGE_ACCOUNT_MAX_FAILURES,
    UsageCacheStore,
    account_backoff_ms,
)


class FakeAuthStore:
    """Minimal stand-in for the AuthStore credential surface."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self._next_id = 1
        self.api_keys: dict[str, str] = {}
        self.oauth: dict[str, object] = {}
        #: Every OAuth account per provider, as the real store now enumerates
        #: them. Distinct from `oauth`, which is the cascade's single pick.
        self.oauth_accounts: dict[str, list[object]] = {}
        #: Runtime/config override keys. When set, list_oauth_identities
        #: returns [] — same contract as AuthStore, because an override
        #: aims at a gateway and stored identity does not apply.
        self._runtime_overrides: dict[str, str] = {}

    def set_runtime_api_key(self, provider: str, api_key: str | None) -> None:
        if api_key:
            self._runtime_overrides[provider] = api_key
            self.api_keys[provider] = api_key
        else:
            self._runtime_overrides.pop(provider, None)

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

    async def list_oauth_accesses(self, provider):
        if self._runtime_overrides.get(provider):
            return []
        return list(self.oauth_accounts.get(provider, []))

    def list_oauth_identities(self, provider):
        # Reporting enumerator: stored identities, no bearer required.
        # An override short-circuits to [] — same as AuthStore — so
        # /usage takes the API-key route instead of naming stored logins.
        if self._runtime_overrides.get(provider):
            return []
        # Rows first (stable stored order), then any oauth_accounts the test
        # registered without also upserting a row. oauth_accounts is the
        # *bearer* list and may be a subset — using it alone would hide a
        # refresh-failed login, which is the defect under test.
        seen: set[str] = set()
        identities: list[object] = []
        for row in self.list_credentials(provider):
            if getattr(row, "credential_type", None) != "oauth":
                continue
            data = getattr(row, "data", None) or {}
            email = data.get("email") or getattr(row, "identity_key", None)
            account_id = data.get("account_id")
            label = email or account_id
            if label:
                seen.add(str(label))
            identities.append(
                types.SimpleNamespace(
                    access_token="",
                    credential_id=getattr(row, "id", 0),
                    account_id=account_id,
                    email=email,
                    org_id=data.get("org_id"),
                    api_endpoint=None,
                    kind="oauth",
                    raw=data,
                )
            )
        for access in self.oauth_accounts.get(provider, []):
            label = getattr(access, "email", None) or getattr(access, "account_id", None)
            if label and str(label) in seen:
                continue
            if label:
                seen.add(str(label))
            identities.append(access)
        return identities

    async def get_api_key(self, provider):
        return self.api_keys.get(provider)


@pytest.fixture
def store() -> FakeAuthStore:
    return FakeAuthStore()


@pytest.fixture
def usage_cache(tmp_path) -> Iterator[UsageCacheStore]:
    # Aim the shared cache at a temp file so controller tests never touch (or
    # are polluted by) the real ~/.local-operator/usage_cache.db.
    cache = UsageCacheStore(tmp_path / "usage_cache.db")
    yield cache
    cache.close()


@pytest.fixture
def controller(store, usage_cache):
    return ProviderController(store, login_callbacks=None, usage_cache=usage_cache)


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
        result = await controller._fetch_one(client, "openrouter", access=None)
    assert result is None


class TestUsageIsPerAccount:
    """Quota is per account, so a provider with two logins has two reports.

    The cascade (`get_oauth_access`) answers "which account will the next
    request run as" and can only ever name one — and with no session id its
    selection order round-robins, so the one account that got reported was not
    even stable between refreshes. A user with two Anthropic logins saw a
    single block and no sign the other existed.
    """

    @staticmethod
    def _account(email: str, account_id: str):
        return types.SimpleNamespace(
            access_token=f"tok-{account_id}",
            credential_id=0,
            account_id=account_id,
            email=email,
            org_id=None,
            api_endpoint=None,
            kind="oauth",
            raw=None,
        )

    @pytest.mark.asyncio
    async def test_every_account_gets_its_own_report(self, controller, store, monkeypatch) -> None:
        store.oauth_accounts["anthropic"] = [
            self._account("first@example.com", "acct-1"),
            self._account("second@example.com", "acct-2"),
        ]
        seen: list[tuple[str, str | None]] = []

        async def fake_fetch(
            client, provider, *, api_key, access_token, account_id, oauth_creds=None
        ):
            seen.append((provider, account_id))
            return UsageReport(provider=provider, limits=[])

        monkeypatch.setattr("local_operator.providers.controller.fetch_usage", fake_fetch)
        reports = await controller.fetch_usage(["anthropic"])

        assert [r.identity for r in reports] == ["first@example.com", "second@example.com"]
        # Each report was fetched with ITS OWN account, not the same one twice.
        assert seen == [("anthropic", "acct-1"), ("anthropic", "acct-2")]

    @pytest.mark.asyncio
    async def test_order_is_stable_across_refreshes(self, controller, store, monkeypatch) -> None:
        """Two refreshes must not reshuffle the list under the reader."""
        store.oauth_accounts["anthropic"] = [
            self._account("first@example.com", "acct-1"),
            self._account("second@example.com", "acct-2"),
        ]

        async def fake_fetch(
            client, provider, *, api_key, access_token, account_id, oauth_creds=None
        ):
            return UsageReport(provider=provider, limits=[])

        monkeypatch.setattr("local_operator.providers.controller.fetch_usage", fake_fetch)
        first = [r.identity for r in await controller.fetch_usage(["anthropic"])]
        second = [r.identity for r in await controller.fetch_usage(["anthropic"])]
        assert first == second == ["first@example.com", "second@example.com"]

    @pytest.mark.asyncio
    async def test_one_failing_account_does_not_hide_the_others(
        self, controller, store, monkeypatch
    ) -> None:
        store.oauth_accounts["anthropic"] = [
            self._account("broken@example.com", "acct-1"),
            self._account("fine@example.com", "acct-2"),
        ]

        async def fake_fetch(
            client, provider, *, api_key, access_token, account_id, oauth_creds=None
        ):
            if account_id == "acct-1":
                raise RuntimeError("quota endpoint exploded")
            return UsageReport(provider=provider, limits=[])

        monkeypatch.setattr("local_operator.providers.controller.fetch_usage", fake_fetch)
        reports = await controller.fetch_usage(["anthropic"])
        # The broken account stays on the panel (no last-good yet, so an
        # empty stub) — omitting it is how a 429 hid a logged-in login.
        assert [r.identity for r in reports] == ["broken@example.com", "fine@example.com"]
        broken = reports[0]
        assert broken.consecutive_failures == 1
        assert broken.limits == []
        assert reports[1].consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_api_key_route_reports_once(self, controller, store, monkeypatch) -> None:
        """No OAuth account means one report, not one per nothing.

        An API key is not an identity — the cascade resolves a single secret
        per provider — so fanning out there would print the same numbers twice.
        """
        store.api_keys["openrouter"] = "sk-or-1"
        calls = 0

        async def fake_fetch(
            client, provider, *, api_key, access_token, account_id, oauth_creds=None
        ):
            nonlocal calls
            calls += 1
            return UsageReport(provider=provider, limits=[])

        monkeypatch.setattr("local_operator.providers.controller.fetch_usage", fake_fetch)
        reports = await controller.fetch_usage(["openrouter"])
        assert len(reports) == 1
        assert calls == 1


@pytest.mark.asyncio
async def test_openai_listing_credential_carries_the_chatgpt_account_scope(
    controller, store
) -> None:
    store.oauth["openai"] = types.SimpleNamespace(
        kind="oauth",
        access_token="chatgpt-token",
        account_id="acct-42",
        org_id=None,
    )

    assert await controller._listing_credential("openai") == (
        "chatgpt-token",
        True,
        "acct-42",
    )


@pytest.mark.asyncio
async def test_a_login_flavour_finds_the_row_its_login_actually_wrote(controller, store) -> None:
    """``openai-device`` is a login flavour of ``openai``, not a second account.

    The ChatGPT device-code login writes ONE credential row, under the aliased
    name (``store_credentials_as``). Asking ``AuthStore`` for the literal id
    found nothing — its ``WHERE provider = ?`` is exact — so the flavour listed
    anonymously: no OAuth, no account scope, no account-scoped catalogue, and a
    logged-in account was offered the bundled ``gpt-4o``/``o3`` rows under that
    second prefix. Exactly the ids an authoritative listing exists to withdraw.
    """
    store.oauth["openai"] = types.SimpleNamespace(
        kind="oauth",
        access_token="chatgpt-token",
        account_id="acct-42",
        org_id=None,
    )

    assert await controller._listing_credential("openai-device") == (
        "chatgpt-token",
        True,
        "acct-42",
    )


def test_usage_enabled_provider_ids(controller) -> None:
    ids = controller.usage_enabled_providers()
    assert "openrouter" in ids
    assert "deepseek" in ids
    assert "zai" in ids, "reachable: ProviderDefinition, credential path and fetcher all exist"
    assert ids == sorted(ids)


#: Every env var that can make a usage provider look credentialed. Cleared first
#: so the test describes the install it sets up rather than the developer's shell.
_USAGE_ENV_VARS = (
    "OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_OAUTH_TOKEN",
    "OPENAI_API_KEY",
    "XAI_API_KEY",
    "KIMI_API_KEY",
    "DEEPSEEK_API_KEY",
)


def test_an_api_key_cannot_reach_an_oauth_only_usage_endpoint(controller, monkeypatch) -> None:
    """`is_usable` answers "is there any credential", which is too coarse here.

    Five of the eight usage providers are OAuth-only for USAGE — an API key cannot
    authenticate against their endpoint at all — so a user holding only
    `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `XAI_API_KEY` holds keys that run the
    model and cannot read the quota. Advertising them anyway is the `zai` defect
    one level finer: a provider no available credential can reach.
    """
    for name in _USAGE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai-test")
    monkeypatch.setenv("XAI_API_KEY", "xai-test")

    for provider in ("anthropic", "openai", "openai-device", "xai"):
        assert controller.is_usable(provider), f"{provider}: the key does run the model"
        assert not controller.can_report_usage(provider), f"{provider}: but not the quota"
    # `xai-oauth` has no env var of its own, but its base's DOES authenticate it
    # (same wire, same endpoint), so the stream-time cascade runs it and
    # `is_usable` must agree — the status surfaces used to say "needs login"
    # for a provider whose very next request succeeds. The finer usage check
    # still excludes it: an API key cannot read the OAuth-only quota endpoint.
    assert controller.is_usable("xai-oauth")
    assert not controller.can_report_usage("xai-oauth")
    assert controller.usage_reportable_providers() == []


def test_is_usable_agrees_with_the_cascade_on_env_keyed_flavours(controller, monkeypatch) -> None:
    """`is_usable` and the stream-time cascade are one question, one reader.

    With only the base provider's var set, `get_api_key(flavour)` resolves the
    env key, so every status surface built on `is_usable` (`/usage`'s warning,
    the welcome screen, the provider row) must say usable too — and must keep
    saying unusable when NO var is set, or the welcome screen would promise a
    login that the cascade then fails."""
    for name in _USAGE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    assert controller.is_usable("xai")
    assert controller.is_usable("xai-oauth")
    monkeypatch.delenv("XAI_API_KEY")
    assert not controller.is_usable("xai-oauth")


def test_an_env_api_key_does_reach_an_api_key_usage_endpoint(controller, monkeypatch) -> None:
    """The other half: where an API-key route EXISTS, an env key reaches it, because
    that is the tier the stream-time cascade resolves."""
    for name in _USAGE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("KIMI_API_KEY", "sk-moonshot")

    assert controller.usage_reportable_providers() == ["kimi", "openrouter"]
    assert not controller.has_any_credential("openrouter"), "nothing STORED, still reachable"


def test_an_oauth_login_unlocks_an_oauth_only_usage_endpoint(
    controller, store, monkeypatch
) -> None:
    for name in _USAGE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    assert controller.usage_reportable_providers() == []
    store.upsert_credential("anthropic", {"access": "tok", "refresh": "ref"})
    assert controller.can_report_usage("anthropic")
    assert controller.usage_reportable_providers() == ["anthropic"]


@pytest.mark.asyncio
async def test_the_default_fetch_target_list_is_the_advertised_list(
    controller, monkeypatch
) -> None:
    """Bare `/usage` must fetch exactly what `/provider` advertised. When the two
    filters were written separately, `/provider` listed anthropic, openai,
    openai-device and xai as reporting quota and bare `/usage` returned nothing."""
    for name in _USAGE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    seen: list[str] = []

    async def _record(client, provider):
        seen.append(provider)
        return None

    monkeypatch.setattr(controller, "_fetch_one", _record)
    assert await controller.fetch_usage() == []
    assert seen == controller.usage_reportable_providers() == []


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


def test_one_entry_can_be_rebuilt_for_the_model_a_session_is_running(controller) -> None:
    """A picker must offer the running model even when the catalogue withdrew it.

    An authoritative account-scoped listing is allowed to prune bundled ids, so
    the session's own model can be absent from the catalogue entirely. This is
    how the picker gets it back, with the same normalization every other entry
    goes through rather than a caller reaching into the registry itself.
    """
    entry = controller.entry_for("anthropic", "claude-opus-4-20250514")

    assert entry is not None
    assert entry.selector == "anthropic/claude-opus-4-20250514"
    assert entry.context_window > 0
    assert entry.input_price > 0
    # An id the registry does not describe is a real answer, not an error: an
    # operator may have configured a model by hand.
    assert controller.entry_for("anthropic", "claude-not-a-model") is None
    assert controller.entry_for("not-a-provider", "whatever") is None


def test_an_aggregator_current_model_is_rebuilt_from_the_resolved_spec(controller) -> None:
    """Aggregators deliberately have no enumerable static catalogue.

    ``static_models('openrouter')`` is empty even for a model the session is
    running, so a rescue that only reads static rows cannot restore the current
    marker when the live listing is unavailable. Session startup already
    resolved the exact model; the single-row rescue spends that spec rather than
    doing synchronous network/cache work on the TUI thread.
    """
    spec = ModelSpec(
        provider="openrouter",
        model_id="deepseek/deepseek-chat",
        display_name="DeepSeek Chat",
        context_window=64_000,
    )

    entry = controller.entry_for(
        "openrouter",
        "deepseek/deepseek-chat",
        spec=spec,
    )

    assert entry is not None
    assert entry.selector == "openrouter/deepseek/deepseek-chat"
    # Naming's honesty rule declines an ambiguous family name and spends the
    # selector instead — the rescue must use the SAME display decision as the
    # band, not force the raw metadata name through.
    assert entry.label == "openrouter/deepseek/deepseek-chat"
    assert entry.context_window == 64_000
    assert entry.input_price == -1.0
    assert entry.output_price == -1.0
    assert entry.aggregated is True

    # The supplied spec is only evidence for itself, never a generic bypass for
    # another selector or provider.
    assert controller.entry_for("openrouter", "other/model", spec=spec) is None
    assert controller.entry_for("radient", spec.model_id, spec=spec) is None


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


def test_usable_providers_agrees_with_is_usable_in_one_store_read(
    controller, store, monkeypatch
) -> None:
    """The picker asks about the whole registry on one keystroke, so it asks once.
    Two predicates answering the same question is how surfaces drift apart."""
    for name in _USAGE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    store.upsert_credential("anthropic", {"access": "tok", "refresh": "ref"})
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds-test")

    usable = controller.usable_providers()
    assert usable is not None
    assert "anthropic" in usable, "stored credential"
    assert "deepseek" in usable, "env key"
    assert "ollama" in usable, "needs no credential at all"
    assert "openrouter" not in usable, "neither"
    for definition in controller.login_providers():
        assert (definition.id in usable) == controller.is_usable(definition.id), definition.id


def test_an_unreadable_store_answers_none_rather_than_empty(controller, store) -> None:
    """ "You have no credentials" and "I could not look" are different answers, and
    a caller that filters a model list on the first would show an empty picker."""

    def boom(provider=None):
        raise RuntimeError("database is locked")

    store.list_credentials = boom  # type: ignore[assignment]
    assert controller.usable_providers() is None


def test_a_catalogue_survives_a_store_that_cannot_be_read(controller, store) -> None:
    """The catalogue is what the picker paints; a locked SQLite file must cost the
    auth ANNOTATION, never the list. Unknown reads as connected because the
    alternative marks every model as needing a login the app never checked for."""

    def boom(provider=None):
        raise RuntimeError("database is locked")

    store.list_credentials = boom  # type: ignore[assignment]
    entries = controller.static_catalogue()
    assert entries
    assert all(entry.connected for entry in entries)


class TestUsageCache:
    """`/usage` answers from the shared cache, not the network, whenever a row
    is warm. The cache is what makes the command instant across every lop
    session on the machine, so these pin the fetch path to it."""

    @staticmethod
    def _account(email: str, account_id: str):
        return types.SimpleNamespace(
            access_token=f"tok-{account_id}",
            credential_id=0,
            account_id=account_id,
            email=email,
            org_id=None,
            api_endpoint=None,
            kind="oauth",
            raw=None,
        )

    @pytest.mark.asyncio
    async def test_a_warm_row_is_served_without_crossing_the_network(
        self, controller, store, monkeypatch
    ) -> None:
        store.oauth_accounts["anthropic"] = [self._account("me@example.com", "acct-1")]
        calls = 0

        async def fake_fetch(
            client, provider, *, api_key, access_token, account_id, oauth_creds=None
        ):
            nonlocal calls
            calls += 1
            return UsageReport(provider=provider, limits=[])

        monkeypatch.setattr("local_operator.providers.controller.fetch_usage", fake_fetch)

        first = await controller.fetch_usage(["anthropic"])
        second = await controller.fetch_usage(["anthropic"])

        assert len(first) == 1 and len(second) == 1
        # The second read hit the cache: exactly one network round total.
        assert calls == 1

    @pytest.mark.asyncio
    async def test_force_refresh_bypasses_the_warm_row(
        self, controller, store, monkeypatch
    ) -> None:
        store.oauth_accounts["anthropic"] = [self._account("me@example.com", "acct-1")]
        calls = 0

        async def fake_fetch(
            client, provider, *, api_key, access_token, account_id, oauth_creds=None
        ):
            nonlocal calls
            calls += 1
            return UsageReport(provider=provider, limits=[])

        monkeypatch.setattr("local_operator.providers.controller.fetch_usage", fake_fetch)

        await controller.fetch_usage(["anthropic"])
        await controller.fetch_usage(["anthropic"], force_refresh=True)

        # `r` in the panel must actually re-fetch, not hand back the cache.
        assert calls == 2

    @pytest.mark.asyncio
    async def test_login_invalidates_the_cached_row(self, controller, store, monkeypatch) -> None:
        """The account set is folded into the cache key, so adding an account
        stops the old row from matching and forces a fresh fetch.

        The fingerprint is a synchronous projection of the STORED credential
        rows (the same source `list_oauth_accesses` reads in the real store),
        so the test populates `rows` as well as the fake's `oauth_accounts`.
        """
        store.upsert_credential(
            "anthropic", {"refresh": "r1", "access": "a1", "email": "me@example.com"}
        )
        store.oauth_accounts["anthropic"] = [self._account("me@example.com", "acct-1")]
        calls = 0

        async def fake_fetch(
            client, provider, *, api_key, access_token, account_id, oauth_creds=None
        ):
            nonlocal calls
            calls += 1
            return UsageReport(provider=provider, limits=[])

        monkeypatch.setattr("local_operator.providers.controller.fetch_usage", fake_fetch)

        await controller.fetch_usage(["anthropic"])
        # A second account logs in: the fingerprint changes, the cached row no
        # longer matches, and the fetch runs again.
        store.upsert_credential(
            "anthropic", {"refresh": "r2", "access": "a2", "email": "other@example.com"}
        )
        store.oauth_accounts["anthropic"].append(self._account("other@example.com", "acct-2"))
        await controller.fetch_usage(["anthropic"])
        # The second fetch re-ran for BOTH accounts (the row no longer matched),
        # so the total is 1 + 2, not a cache hit.
        assert calls == 3

    @pytest.mark.asyncio
    async def test_a_failed_refresh_serves_the_last_good_value(
        self, controller, store, monkeypatch
    ) -> None:
        """A DOWN endpoint must not blank (or negative-cache over) real data.

        HONEST failure mode: the real fetchers never raise — `_get_json`
        swallows transport errors, non-200s and bad JSON and returns None — so
        an outage reaches the cache layer as an EMPTY result. The disambiguator
        is history: a provider that had data a moment ago and reports none now
        keeps its last good value under a short cool-down.
        """
        store.oauth_accounts["anthropic"] = [self._account("me@example.com", "acct-1")]
        fail = False

        async def fake_fetch(
            client, provider, *, api_key, access_token, account_id, oauth_creds=None
        ):
            if fail:
                return None  # what a 429/outage actually looks like to callers
            return UsageReport(provider=provider, limits=[])

        monkeypatch.setattr("local_operator.providers.controller.fetch_usage", fake_fetch)

        warm = await controller.fetch_usage(["anthropic"])
        assert len(warm) == 1

        # Expire the row (a RECENT past expiry — an ancient one is pruned by the
        # retention cleanup on the next write), then fail the refresh: the stale
        # value must survive.
        key = controller._usage_cache_key("anthropic")
        cache = controller._usage_cache_store()
        assert cache is not None
        stale = cache.get(key, include_expired=True)
        assert stale is not None
        import time as _time

        cache.set(key, "anthropic", stale, expires_at_ms=int(_time.time() * 1000) - 1000)

        fail = True
        recovered = await controller.fetch_usage(["anthropic"])
        assert len(recovered) == 1, "a blip must not blank the report"
        # And the shared row still holds the data for every OTHER session.
        assert cache.get(key, include_expired=True), "last-good row was overwritten"

    @pytest.mark.asyncio
    async def test_a_provider_with_no_data_history_is_negative_cached(
        self, controller, store, monkeypatch
    ) -> None:
        """An API-key provider that reports nothing (and never has) is cached
        as empty, so the warmer stops re-hitting an endpoint with nothing to
        say. OAuth logins take the per-account path instead — they must stay
        on the panel even with no last-good."""
        store.api_keys["openrouter"] = "sk-or-1"
        calls = 0

        async def fake_fetch(
            client, provider, *, api_key, access_token, account_id, oauth_creds=None
        ):
            nonlocal calls
            calls += 1
            return None  # endpoint answers, but there is no quota to report

        monkeypatch.setattr("local_operator.providers.controller.fetch_usage", fake_fetch)

        assert await controller.fetch_usage(["openrouter"]) == []
        assert await controller.fetch_usage(["openrouter"]) == []
        # The empty answer was cached: exactly one network round total.
        assert calls == 1
        # And the row is visible to the warmer's age probe.
        assert controller.usage_cache_age_ms("openrouter") is not None

    @pytest.mark.asyncio
    async def test_alias_providers_share_one_cache_row(
        self, controller, store, monkeypatch
    ) -> None:
        """`openai-device` logs in under `openai`; both spellings must read the
        same cache row rather than hold one permanently-stale copy each."""
        assert controller._usage_cache_key("openai") == controller._usage_cache_key("openai-device")

    @pytest.mark.asyncio
    async def test_old_enough_data_lets_an_empty_answer_be_believed(
        self, controller, store, monkeypatch
    ) -> None:
        """A provider that GENUINELY went quota-less must eventually settle.

        The empty-over-data heuristic reads a blank answer over recent data as
        an outage — but each write_failure kept the old row alive, so a lapsed
        plan was re-fetched on every cool-down forever. Once the last real data
        is older than EMPTY_OVER_DATA_ACCEPT_MS, the empty answer is accepted
        and negative-cached at full TTL.
        """
        import time as _time

        from local_operator.providers.controller import EMPTY_OVER_DATA_ACCEPT_MS

        store.api_keys["openrouter"] = "sk-or-1"

        async def fake_fetch(
            client, provider, *, api_key, access_token, account_id, oauth_creds=None
        ):
            return None  # blank answer, endpoint reachable

        monkeypatch.setattr("local_operator.providers.controller.fetch_usage", fake_fetch)

        # Plant a last-good row whose data is OLDER than the acceptance window,
        # already expired so the refresh path runs. API-key route: no OAuth
        # identity set, so the empty-over-data heuristic still applies.
        key = controller._usage_cache_key("openrouter")
        cache = controller._usage_cache_store()
        assert cache is not None
        now_ms = int(_time.time() * 1000)
        old = UsageReport(provider="openrouter", limits=[])
        old.fetched_at = now_ms - EMPTY_OVER_DATA_ACCEPT_MS - 60_000
        cache.set(key, "openrouter", [old], expires_at_ms=now_ms - 1000)

        # The empty answer is BELIEVED: no stale serve, and the row is now a
        # full-TTL negative entry (fresh, empty).
        assert await controller.fetch_usage(["openrouter"]) == []
        assert cache.get(key) == []


class TestPerAccountLastKnown:
    """A 429 for one login must not erase that login — or its siblings.

    #277 cached the *list of reports that succeeded this fetch*. A partial
    success (3 of 4 Anthropic tokens 200, one 429) overwrote the last-good
    4-account snapshot with a 3-account payload, which is how
    damian@gominerva.com vanished from ``/usage`` while still logged in.
    """

    @staticmethod
    def _account(email: str, account_id: str):
        return types.SimpleNamespace(
            access_token=f"tok-{account_id}",
            credential_id=0,
            account_id=account_id,
            email=email,
            org_id=None,
            api_endpoint=None,
            kind="oauth",
            raw=None,
        )

    @staticmethod
    def _report(identity: str, percent: float, fetched_at: int | None = None) -> UsageReport:
        import time as _time

        return UsageReport(
            provider="anthropic",
            fetched_at=fetched_at if fetched_at is not None else int(_time.time() * 1000),
            identity=identity,
            limits=[
                UsageLimit(
                    id=f"{identity}:7d",
                    label="7 day",
                    amount=UsageAmount(
                        used=percent,
                        limit=100.0,
                        used_fraction=percent / 100.0,
                        unit="percent",
                    ),
                    window="7 day",
                    shared=True,
                )
            ],
        )

    def _four(self, store: FakeAuthStore) -> list[tuple[str, str]]:
        accounts = [
            ("damian@gominerva.com", "acct-gominerva"),
            ("damian@radienthq.com", "acct-radient"),
            ("damian@pergamonhq.com", "acct-pergamon"),
            ("damianvtran@gmail.com", "acct-gmail"),
        ]
        store.oauth_accounts["anthropic"] = [
            self._account(email, account_id) for email, account_id in accounts
        ]
        for email, _account_id in accounts:
            store.upsert_credential("anthropic", {"refresh": "r", "access": "a", "email": email})
        return accounts

    @pytest.mark.asyncio
    async def test_a_partial_fetch_keeps_the_failed_accounts_last_known(
        self, controller, store, monkeypatch
    ) -> None:
        """4 accounts; one fetch returns None for gominerva → still 4 identities,
        and that one keeps the previous weekly number."""
        self._four(store)
        fail_gominerva = False

        async def fake_fetch(
            client, provider, *, api_key, access_token, account_id, oauth_creds=None
        ):
            if fail_gominerva and account_id == "acct-gominerva":
                return None
            percent = {"acct-gominerva": 12.0, "acct-radient": 34.0}.get(account_id, 56.0)
            identity = {
                "acct-gominerva": "damian@gominerva.com",
                "acct-radient": "damian@radienthq.com",
                "acct-pergamon": "damian@pergamonhq.com",
                "acct-gmail": "damianvtran@gmail.com",
            }[account_id]
            return self._report(identity, percent)

        monkeypatch.setattr("local_operator.providers.controller.fetch_usage", fake_fetch)

        warm = await controller.fetch_usage(["anthropic"])
        assert [r.identity for r in warm] == [
            "damian@gominerva.com",
            "damian@radienthq.com",
            "damian@pergamonhq.com",
            "damianvtran@gmail.com",
        ]
        assert warm[0].limits[0].amount.used == 12.0

        key = controller._usage_cache_key("anthropic")
        cache = controller._usage_cache_store()
        assert cache is not None
        import time as _time

        cache.set(key, "anthropic", warm, expires_at_ms=int(_time.time() * 1000) - 1000)

        fail_gominerva = True
        recovered = await controller.fetch_usage(["anthropic"])
        assert [r.identity for r in recovered] == [
            "damian@gominerva.com",
            "damian@radienthq.com",
            "damian@pergamonhq.com",
            "damianvtran@gmail.com",
        ]
        gominerva = recovered[0]
        assert gominerva.limits[0].amount.used == 12.0
        assert gominerva.consecutive_failures == 1
        assert gominerva.usage_unavailable is False
        # Partial success must NOT shrink the cached anthropic payload.
        cached = cache.get(key, include_expired=True)
        assert cached is not None
        assert [r.identity for r in cached] == [r.identity for r in recovered]

    @pytest.mark.asyncio
    async def test_max_failures_with_no_last_good_still_lists_the_account(
        self, controller, store, monkeypatch
    ) -> None:
        store.oauth_accounts["anthropic"] = [self._account("new@example.com", "acct-new")]

        async def fake_fetch(
            client, provider, *, api_key, access_token, account_id, oauth_creds=None
        ):
            return None

        monkeypatch.setattr("local_operator.providers.controller.fetch_usage", fake_fetch)

        reports = None
        for _ in range(USAGE_ACCOUNT_MAX_FAILURES):
            key = controller._usage_cache_key("anthropic")
            cache = controller._usage_cache_store()
            assert cache is not None
            import time as _time

            existing = cache.get(key, include_expired=True)
            if existing:
                now_ms = int(_time.time() * 1000)
                for report in existing:
                    report.next_probe_at_ms = now_ms - 1
                cache.set(key, "anthropic", existing, expires_at_ms=now_ms - 1000)
            reports = await controller.fetch_usage(["anthropic"])
        assert reports is not None
        assert len(reports) == 1
        assert reports[0].identity == "new@example.com"
        assert reports[0].usage_unavailable is True
        assert reports[0].limits == []

    @pytest.mark.asyncio
    async def test_force_refresh_retries_a_maxed_out_account(
        self, controller, store, monkeypatch
    ) -> None:
        store.oauth_accounts["anthropic"] = [self._account("new@example.com", "acct-new")]
        calls = 0
        succeed = False

        async def fake_fetch(
            client, provider, *, api_key, access_token, account_id, oauth_creds=None
        ):
            nonlocal calls
            calls += 1
            if succeed:
                return self._report("new@example.com", 7.0)
            return None

        monkeypatch.setattr("local_operator.providers.controller.fetch_usage", fake_fetch)

        for _ in range(USAGE_ACCOUNT_MAX_FAILURES):
            key = controller._usage_cache_key("anthropic")
            cache = controller._usage_cache_store()
            assert cache is not None
            import time as _time

            existing = cache.get(key, include_expired=True)
            if existing:
                now_ms = int(_time.time() * 1000)
                for report in existing:
                    report.next_probe_at_ms = now_ms - 1
                cache.set(key, "anthropic", existing, expires_at_ms=now_ms - 1000)
            await controller.fetch_usage(["anthropic"])

        calls_before = calls
        succeed = True
        # Without force, the unavailable account is not re-probed.
        idle = await controller.fetch_usage(["anthropic"])
        assert calls == calls_before
        assert idle[0].usage_unavailable is True

        recovered = await controller.fetch_usage(["anthropic"], force_refresh=True)
        assert calls == calls_before + 1
        assert recovered[0].usage_unavailable is False
        assert recovered[0].consecutive_failures == 0
        assert recovered[0].limits[0].amount.used == 7.0

    @pytest.mark.asyncio
    async def test_backoff_skips_only_the_failed_account(
        self, controller, store, monkeypatch
    ) -> None:
        """A failed account is not re-fetched until its backoff elapses;
        siblings that are fresh still refresh."""
        store.oauth_accounts["anthropic"] = [
            self._account("fail@example.com", "acct-fail"),
            self._account("ok@example.com", "acct-ok"),
        ]
        seen: list[str] = []
        fail = False

        async def fake_fetch(
            client, provider, *, api_key, access_token, account_id, oauth_creds=None
        ):
            seen.append(account_id)
            if fail and account_id == "acct-fail":
                return None
            identity = "fail@example.com" if account_id == "acct-fail" else "ok@example.com"
            return self._report(identity, 10.0)

        monkeypatch.setattr("local_operator.providers.controller.fetch_usage", fake_fetch)

        await controller.fetch_usage(["anthropic"])
        key = controller._usage_cache_key("anthropic")
        cache = controller._usage_cache_store()
        assert cache is not None
        import time as _time

        cache.set(
            key,
            "anthropic",
            cache.get(key, include_expired=True) or [],
            expires_at_ms=int(_time.time() * 1000) - 1000,
        )
        fail = True
        seen.clear()
        first_fail = await controller.fetch_usage(["anthropic"])
        assert "acct-fail" in seen and "acct-ok" in seen
        assert first_fail[0].consecutive_failures == 1
        backoff = account_backoff_ms(1)
        assert backoff > 0

        # Expire the *provider* row so the lease/refresh path runs again, but
        # the failed account's next_probe_at is still in the future.
        cache.set(
            key,
            "anthropic",
            first_fail,
            expires_at_ms=int(_time.time() * 1000) - 1000,
        )
        seen.clear()
        second = await controller.fetch_usage(["anthropic"])
        assert "acct-fail" not in seen
        assert "acct-ok" in seen
        assert second[0].consecutive_failures == 1
        assert second[0].limits[0].amount.used == 10.0

    @pytest.mark.asyncio
    async def test_a_refresh_failed_identity_is_still_listed(
        self, controller, store, monkeypatch
    ) -> None:
        """list_oauth_accesses omitted the unrefreshable row; /usage must not."""
        store.upsert_credential(
            "anthropic",
            {
                "refresh": "r",
                "access": "a",
                "email": "stale@example.com",
                "account_id": "acct-stale",
            },
        )
        store.upsert_credential(
            "anthropic",
            {"refresh": "r", "access": "b", "email": "live@example.com", "account_id": "acct-live"},
        )
        # Only the live account can mint a bearer this cycle.
        store.oauth_accounts["anthropic"] = [self._account("live@example.com", "acct-live")]

        async def fake_fetch(
            client, provider, *, api_key, access_token, account_id, oauth_creds=None
        ):
            return self._report("live@example.com", 22.0)

        monkeypatch.setattr("local_operator.providers.controller.fetch_usage", fake_fetch)
        reports = await controller.fetch_usage(["anthropic"])
        assert [r.identity for r in reports] == ["stale@example.com", "live@example.com"]
        assert reports[0].limits == []
        assert reports[0].consecutive_failures == 1
        assert reports[1].limits[0].amount.used == 22.0

    @pytest.mark.asyncio
    async def test_an_override_fetches_the_api_key_not_stored_oauth_stubs(
        self, controller, store, monkeypatch
    ) -> None:
        """OAuth rows plus a runtime override must take the API-key route.

        ``list_oauth_identities`` returns [] when an override is set — that
        empty list is authoritative. Falling through to ``list_credentials``
        would name the stored emails, skip ``_fetch_one(access=None)``, and
        paint last-known / unavailable stubs for accounts the session is
        not using.
        """
        self._four(store)
        store.set_runtime_api_key("anthropic", "sk-ant-override")
        seen: list[tuple[str | None, str | None]] = []

        async def fake_fetch(
            client, provider, *, api_key, access_token, account_id, oauth_creds=None
        ):
            seen.append((api_key, account_id))
            return UsageReport(
                provider=provider,
                identity=None,
                limits=[
                    UsageLimit(
                        id="override:7d",
                        label="7 day",
                        amount=UsageAmount(
                            used=9.0, limit=100.0, used_fraction=0.09, unit="percent"
                        ),
                        window="7 day",
                        shared=True,
                    )
                ],
            )

        monkeypatch.setattr("local_operator.providers.controller.fetch_usage", fake_fetch)
        reports = await controller.fetch_usage(["anthropic"])
        assert seen == [("sk-ant-override", None)]
        assert len(reports) == 1
        assert reports[0].identity is None
        assert reports[0].usage_unavailable is False
        assert {r.identity for r in reports}.isdisjoint(
            {
                "damian@gominerva.com",
                "damian@radienthq.com",
                "damian@pergamonhq.com",
                "damianvtran@gmail.com",
            }
        )


# ---------------------------------------------------------------------------
# The picker's prices come from the same keyless chain as the status band
# ---------------------------------------------------------------------------
#
# A direct-provider model the shipped registry did not carry showed a BLANK
# price in the picker (``_price``'s unknown sentinel) while the status band said
# ``$10/50`` the moment it was selected: the rows were priced from
# ``merge_models(registry, listing)`` alone, and Anthropic's listing quotes no
# money. ``live_catalogue`` now fills those holes through ``prices.price_row``
# over ONE read of each document, so the two surfaces cannot drift.

from local_operator.model.discovery import DiscoveredModel  # noqa: E402

# The picker's own formatter, so these tests assert what a USER would read in
# the price column rather than re-spelling the sentinel convention themselves.
from local_operator.tui.widgets.model_picker import format_price_pair  # noqa: E402

#: The models.dev projection's ``providers`` map, as ``models_dev_providers``
#: returns it — one row per case the tests below exercise.
_PROJECTION = {
    "anthropic": {
        "claude-fable-5-1": {
            "name": "Claude Fable 5.1",
            "cost": {"input": 10, "output": 50, "cache_read": 0.25, "cache_write": 12.5},
            "limit": {"context": 1_000_000, "output": 128_000},
        },
    },
    "openai": {
        "gpt-5.4": {
            "name": "GPT-5.4",
            "cost": {"input": 2.5, "output": 15},
            "limit": {"context": 400_000, "output": 128_000},
        },
    },
}

_OPENROUTER_ROWS = [
    DiscoveredModel(
        id="anthropic/claude-fable-5.1",
        name="Anthropic: Claude Fable 5.1",
        context_window=1_000_000,
        max_tokens=128_000,
        input_price=10.0,
        output_price=50.0,
        cache_read_price=0.25,
        cache_write_price=12.5,
    ),
    DiscoveredModel(
        id="anthropic/claude-nova-9",
        name="Anthropic: Claude Nova 9",
        context_window=2_000_000,
        input_price=7.0,
        output_price=35.0,
    ),
    DiscoveredModel(id="openrouter/free-router", name="Free Router", context_window=8_000),
]


def _listing(monkeypatch, rows: dict[str, list[DiscoveredModel]], status: str = "ok"):
    """Stub discovery per provider: ``rows`` for the named ids, nothing elsewhere."""
    calls: list[str] = []

    def fake(provider_id, **kwargs):
        calls.append(provider_id)
        return list(rows.get(provider_id, [])), status if provider_id in rows else "static"

    monkeypatch.setattr("local_operator.providers.controller.available_models", fake)
    return calls


def _projection(monkeypatch, providers):
    reads: list[int] = []

    def fake(**kwargs):
        reads.append(1)
        return providers

    monkeypatch.setattr("local_operator.model.prices.models_dev_providers", fake)
    return reads


def _by_selector(entries):
    return {entry.selector: entry for entry in entries}


@pytest.mark.asyncio
async def test_an_unpriced_direct_row_is_priced_from_models_dev(
    controller, store, monkeypatch
) -> None:
    """The operator's screenshot: ``claude-fable-5-1`` blank, ``claude-fable-5``
    ``$10/50``. Anthropic's listing carries the window and no money; the row
    must leave with models.dev's price and KEEP the listing's window."""
    store.upsert_credential("anthropic", {"key": "sk-ant", "type": "api_key"})
    unpriced = DiscoveredModel(
        id="claude-fable-5-1", name="Claude Fable 5.1", context_window=999_000, max_tokens=128_000
    )
    _listing(monkeypatch, {"anthropic": [unpriced]})
    reads = _projection(monkeypatch, _PROJECTION)

    entries, statuses = await controller.live_catalogue()

    row = _by_selector(entries)["anthropic/claude-fable-5-1"]
    assert (row.input_price, row.output_price) == (10.0, 50.0)
    assert row.context_window == 999_000, "the provider's own window wins over the catalogue"
    assert statuses["anthropic"] == "ok"
    assert len(reads) == 1, "the projection is read once for the whole catalogue"


@pytest.mark.asyncio
async def test_a_price_the_listing_quoted_is_never_overridden(
    controller, store, monkeypatch
) -> None:
    """The provider's own number is authoritative; the chain fills holes only."""
    store.upsert_credential("anthropic", {"key": "sk-ant", "type": "api_key"})
    quoted = DiscoveredModel(
        id="claude-fable-5-1", context_window=1_000_000, input_price=8.0, output_price=40.0
    )
    _listing(monkeypatch, {"anthropic": [quoted]})
    _projection(monkeypatch, _PROJECTION)

    entries, _ = await controller.live_catalogue()

    row = _by_selector(entries)["anthropic/claude-fable-5-1"]
    assert (row.input_price, row.output_price) == (8.0, 40.0)


@pytest.mark.asyncio
async def test_a_stated_zero_reaches_the_picker_as_free_not_as_unknown(
    controller, store, monkeypatch
) -> None:
    """The ``:free`` routes, which rendered a BLANK price cell.

    ``_price`` maps ``0.0`` to its ``-1.0`` unknown sentinel for every provider
    that wants a credential, so a vendor's quoted ``$0`` was indistinguishable
    from silence and got the same empty cell — eighteen rows literally named
    ``:free`` among them. The flag the parser sets is what tells them apart.
    """
    store.upsert_credential("anthropic", {"key": "sk-ant", "type": "api_key"})
    stated = DiscoveredModel(id="gemma-free", context_window=32_000, free=True)
    _listing(monkeypatch, {"anthropic": [stated]})
    _projection(monkeypatch, _PROJECTION)

    entries, _ = await controller.live_catalogue()

    row = _by_selector(entries)["anthropic/gemma-free"]
    assert (row.input_price, row.output_price) == (0.0, 0.0), "the stated zero survived"
    assert format_price_pair(row.input_price, row.output_price) == "free"


@pytest.mark.asyncio
async def test_a_row_nobody_priced_still_renders_blank(controller, store, monkeypatch) -> None:
    """The other half of the same distinction, and the reason it cannot simply
    stop mapping zero to the sentinel: an UNPRICED row must keep its blank cell
    rather than gain a ``free`` it was never quoted."""
    store.upsert_credential("anthropic", {"key": "sk-ant", "type": "api_key"})
    silent = DiscoveredModel(id="claude-unpriced", context_window=200_000)
    _listing(monkeypatch, {"anthropic": [silent]})
    _projection(monkeypatch, {})

    entries, _ = await controller.live_catalogue()

    row = _by_selector(entries)["anthropic/claude-unpriced"]
    assert (row.input_price, row.output_price) == (-1.0, -1.0), "unknown, not free"
    assert format_price_pair(row.input_price, row.output_price) == ""


@pytest.mark.asyncio
async def test_a_plan_billed_row_stays_blank_rather_than_claiming_to_be_free(
    controller, store, monkeypatch
) -> None:
    """``alibaba-token-plan`` bills CREDITS, so models.dev quotes it 0/0 to mean
    "not priced in dollars" — a stated zero whose real cost is still unknowable.

    It must stop the chain exactly as before (never taking ``alibaba``'s
    pay-per-token rate) and must NOT print ``free``, which would be a lie the
    user could act on. This is the case that keeps the fix honest.
    """
    store.upsert_credential("alibaba-token-plan", {"key": "sk-plan", "type": "api_key"})
    plan_row = DiscoveredModel(id="glm-5.2", context_window=1_000_000)
    _listing(monkeypatch, {"alibaba-token-plan": [plan_row]})
    _projection(
        monkeypatch,
        {
            "alibaba-token-plan": {
                "glm-5.2": {"cost": {"input": 0, "output": 0}, "limit": {"context": 1_000_000}}
            },
            # A priced sibling under the pay-per-token key: if the plan's zero
            # stopped answering, this is the number that would wrongly appear.
            "alibaba": {"glm-5.2": {"cost": {"input": 0.6, "output": 2.2}}},
        },
    )

    entries, _ = await controller.live_catalogue()

    row = _by_selector(entries)["alibaba-token-plan/glm-5.2"]
    assert (row.input_price, row.output_price) == (-1.0, -1.0), "unknowable, so blank"
    assert format_price_pair(row.input_price, row.output_price) == ""


@pytest.mark.asyncio
async def test_a_models_dev_miss_falls_back_to_the_openrouter_rows_already_listed(
    controller, store, monkeypatch
) -> None:
    """The secondary leg for the picker is the ``openrouter`` provider's OWN rows
    from this same call — no second document, no second request. The projection
    lacks ``claude-nova-9``; OpenRouter prices it under ``anthropic/``."""
    store.upsert_credential("anthropic", {"key": "sk-ant", "type": "api_key"})
    unpriced = DiscoveredModel(id="claude-nova-9", context_window=0)
    _listing(monkeypatch, {"anthropic": [unpriced], "openrouter": list(_OPENROUTER_ROWS)})
    _projection(monkeypatch, _PROJECTION)

    entries, _ = await controller.live_catalogue()

    rows = _by_selector(entries)
    nova = rows["anthropic/claude-nova-9"]
    assert (nova.input_price, nova.output_price) == (7.0, 35.0)
    assert nova.context_window == 2_000_000, "a window the listing left at 0 is filled"
    # OpenRouter's own rows are untouched by the enrichment: an unpriced
    # ``openrouter/*`` row stays unknown (aggregator ⇒ never enriched).
    assert rows["openrouter/openrouter/free-router"].input_price == -1.0
    assert rows["openrouter/anthropic/claude-fable-5.1"].input_price == 10.0


@pytest.mark.asyncio
async def test_models_dev_beats_openrouter_when_both_price_a_picker_row(
    controller, store, monkeypatch
) -> None:
    store.upsert_credential("anthropic", {"key": "sk-ant", "type": "api_key"})
    _listing(
        monkeypatch,
        {
            "anthropic": [DiscoveredModel(id="claude-fable-5-1", context_window=1_000_000)],
            "openrouter": [
                dataclasses.replace(_OPENROUTER_ROWS[0], input_price=1.0, output_price=2.0)
            ],
        },
    )
    _projection(monkeypatch, _PROJECTION)

    entries, _ = await controller.live_catalogue()

    row = _by_selector(entries)["anthropic/claude-fable-5-1"]
    assert (row.input_price, row.output_price) == (10.0, 50.0)


@pytest.mark.asyncio
async def test_neither_document_leaves_the_unknown_sentinel_and_never_fetches(
    controller, store, monkeypatch
) -> None:
    """Offline picker: no projection on disk, no OpenRouter rows. The row keeps
    the ``-1`` unknown sentinel (never ``free``), and nothing raises."""
    store.upsert_credential("anthropic", {"key": "sk-ant", "type": "api_key"})
    _listing(monkeypatch, {"anthropic": [DiscoveredModel(id="claude-fable-5-1")]})
    _projection(monkeypatch, None)

    def no_network(*args, **kwargs):  # pragma: no cover - the assertion is that it is unused
        raise AssertionError("the picker path must not fetch")

    monkeypatch.setattr("httpx.get", no_network)
    monkeypatch.setattr("httpx.Client.send", no_network)

    entries, _ = await controller.live_catalogue()

    row = _by_selector(entries)["anthropic/claude-fable-5-1"]
    assert (row.input_price, row.output_price) == (-1.0, -1.0)


@pytest.mark.asyncio
async def test_a_login_flavour_is_priced_under_its_canonical_provider(
    controller, store, monkeypatch
) -> None:
    """``openai-device`` prices as ``openai`` — the same translation the resolver
    applies — so a ChatGPT account's live rows get the pay-per-token price the
    projection keys under ``openai``."""
    store.upsert_credential("openai", {"key": "sk", "type": "api_key"})
    _listing(monkeypatch, {"openai-device": [DiscoveredModel(id="gpt-5.4", context_window=0)]})
    _projection(monkeypatch, _PROJECTION)

    entries, _ = await controller.live_catalogue()

    row = _by_selector(entries)["openai-device/gpt-5.4"]
    assert (row.input_price, row.output_price) == (2.5, 15.0)
    assert row.context_window == 400_000


@pytest.mark.asyncio
async def test_a_keyless_provider_stays_free_rather_than_unknown(controller, monkeypatch) -> None:
    """Ollama really is free per token; the chain has no mapping for it and the
    ``_price`` rule keeps a genuine zero visible."""
    _listing(monkeypatch, {"ollama": [DiscoveredModel(id="qwen3:8b", context_window=32_000)]})
    _projection(monkeypatch, _PROJECTION)

    entries, _ = await controller.live_catalogue()

    row = _by_selector(entries)["ollama/qwen3:8b"]
    assert (row.input_price, row.output_price) == (0.0, 0.0)


def test_the_static_catalogue_reads_nothing_and_still_paints(controller, monkeypatch) -> None:
    """The first frame is registry-only by contract: no document read, no thread."""

    def not_here(**kwargs):  # pragma: no cover - the assertion is that it is unused
        raise AssertionError("static_catalogue must not read the price documents")

    monkeypatch.setattr("local_operator.model.prices.models_dev_providers", not_here)
    assert controller.static_catalogue()


# ---------------------------------------------------------------------------
# Credential-change invalidation (ported from bbqben's #535)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_logging_in_drops_the_cached_listing(controller, store, monkeypatch) -> None:
    """Re-authing is what a user does when a model is missing. The credential
    row is written AND the listing document — fetched anonymously, or under
    whatever account came before — is dropped, so the next picker open lists
    under the new credential instead of serving the old catalogue."""
    dropped: list[str] = []
    monkeypatch.setattr(
        "local_operator.providers.controller.invalidate_listing",
        lambda provider_id: dropped.append(provider_id) or 1,
    )

    async def fake_login(_callbacks):
        return {"access_token": "t", "refresh_token": "r", "email": "you@example.com"}

    # ProviderDefinition is a frozen dataclass, so the login is swapped by
    # replacing the definition the controller resolves rather than the field.
    definition = controller.provider("anthropic")
    assert definition is not None
    monkeypatch.setattr(
        "local_operator.providers.controller.get_provider_definition",
        lambda provider_id: (
            dataclasses.replace(definition, login=fake_login)
            if provider_id == "anthropic"
            else get_provider_definition(provider_id)
        ),
    )

    await controller.login("anthropic")

    assert dropped == ["anthropic"]


@pytest.mark.asyncio
async def test_an_api_key_login_drops_the_listing_under_the_storage_id(
    controller, store, monkeypatch
) -> None:
    """The paste-a-key path stores under ``store_credentials_as``; the document
    is named the same way, so that is the id to invalidate."""
    dropped: list[str] = []
    monkeypatch.setattr(
        "local_operator.providers.controller.invalidate_listing",
        lambda provider_id: dropped.append(provider_id) or 1,
    )

    async def fake_login(_callbacks):
        return "xai-key"

    definition = controller.provider("xai-oauth")
    assert definition is not None and definition.store_credentials_as == "xai"
    monkeypatch.setattr(
        "local_operator.providers.controller.get_provider_definition",
        lambda provider_id: (
            dataclasses.replace(definition, login=fake_login)
            if provider_id == "xai-oauth"
            else get_provider_definition(provider_id)
        ),
    )

    await controller.login("xai-oauth")

    assert dropped == ["xai"]


@pytest.mark.asyncio
async def test_logging_out_clears_the_listing_the_next_credential_must_not_inherit(
    controller, store, monkeypatch
) -> None:
    """A catalogue fetched under the credential just removed must not decide what
    the next account can select."""
    dropped: list[str] = []
    monkeypatch.setattr(
        "local_operator.providers.controller.invalidate_listing",
        lambda provider_id: dropped.append(provider_id) or 1,
    )
    store.upsert_credential("anthropic", {"key": "sk-ant", "type": "api_key"})

    await controller.logout("anthropic")

    assert "anthropic" in dropped


@pytest.mark.asyncio
async def test_a_failed_invalidation_never_fails_a_successful_login(
    controller, store, monkeypatch
) -> None:
    def boom(provider_id):
        raise OSError("read-only cache")

    monkeypatch.setattr("local_operator.providers.controller.invalidate_listing", boom)

    async def fake_login(_callbacks):
        return "sk-ant"

    definition = controller.provider("anthropic")
    assert definition is not None
    monkeypatch.setattr(
        "local_operator.providers.controller.get_provider_definition",
        lambda provider_id: (
            dataclasses.replace(definition, login=fake_login)
            if provider_id == "anthropic"
            else get_provider_definition(provider_id)
        ),
    )

    assert "Stored API key" in await controller.login("anthropic")


@pytest.mark.asyncio
async def test_login_and_logout_drop_the_in_process_model_info_memo(
    controller, store, monkeypatch
) -> None:
    """A status-band resolution that degraded before the login (no credential
    → registry-only numbers) is memoised per TTL bucket; without this drop a
    long-lived TUI keeps the stale answer for the rest of the bucket. The
    server's credential route already pairs the two invalidations for exactly
    this event; the controller hook now matches."""
    cleared: list[str] = []
    monkeypatch.setattr("local_operator.providers.controller.invalidate_listing", lambda pid: 1)
    monkeypatch.setattr(
        "local_operator.model.configure.invalidate_model_info_cache",
        lambda: cleared.append("memo"),
    )

    async def fake_login(_callbacks):
        return "sk-ant"

    definition = controller.provider("anthropic")
    assert definition is not None
    monkeypatch.setattr(
        "local_operator.providers.controller.get_provider_definition",
        lambda provider_id: (
            dataclasses.replace(definition, login=fake_login)
            if provider_id == "anthropic"
            else get_provider_definition(provider_id)
        ),
    )

    await controller.login("anthropic")
    assert cleared == ["memo"]

    store.upsert_credential("anthropic", {"key": "sk-ant", "type": "api_key"})
    await controller.logout("anthropic")
    assert cleared == ["memo", "memo"]


@pytest.mark.asyncio
async def test_logout_invalidates_once_per_storage_id(controller, store, monkeypatch) -> None:
    """``zai-oauth`` and ``zai`` resolve to the SAME document set, so iterating
    both would glob the cache dir twice per logout."""
    dropped: list[str] = []
    monkeypatch.setattr(
        "local_operator.providers.controller.invalidate_listing",
        lambda provider_id: dropped.append(provider_id) or 1,
    )
    store.upsert_credential("zai", {"key": "sk-zai", "type": "api_key"})

    await controller.logout("zai-oauth")

    assert dropped == ["zai"]


def _spy_available_models(monkeypatch, *, live: dict[str, list[str]] | None = None):
    """Record every ``available_models`` call and the TTL it was given.

    Returns the call log. ``live`` names the model ids a provider answers with;
    anything absent answers as an unauthenticated provider, which is what the
    registry's two dozen unconfigured entries look like in a real run.
    """
    calls: list[tuple[str, float | None]] = []
    live = live or {}

    def fake(provider_id: str, **kwargs: Any):
        calls.append((provider_id, kwargs.get("ttl_s")))
        ids = live.get(provider_id)
        if ids is None:
            return [], "unauthenticated"
        return [DiscoveredModel(id=model_id, name=model_id) for model_id in ids], "ok"

    monkeypatch.setattr("local_operator.providers.controller.available_models", fake)
    return calls


@pytest.mark.asyncio
async def test_the_picker_ttl_is_the_only_ttl_override(controller, store, monkeypatch) -> None:
    """``live_catalogue`` passes the caller's TTL through untouched and adds none of
    its own, so ``PICKER_TTL_S`` remains the single statement of picker freshness."""
    store.upsert_credential("anthropic", {"key": "sk-ant", "type": "api_key"})
    calls = _spy_available_models(monkeypatch, live={"anthropic": ["claude-opus-5"]})

    await controller.live_catalogue()
    assert [ttl for _, ttl in calls if ttl is not None] == [], "no override by default"

    calls.clear()
    await controller.live_catalogue(ttl_s=PICKER_TTL_S)
    assert {ttl for _, ttl in calls} == {PICKER_TTL_S}
