"""Unit tests for ProviderController — the TUI's provider/model/usage facade.

Credential/login behavior is exercised against a fake auth store so no real
SQLite or network is needed; usage dispatch is tested against a canned
httpx transport.
"""

from __future__ import annotations

import dataclasses
import sqlite3
import time
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
    USAGE_REPORT_TTL_MS,
    UsageCacheStore,
    account_backoff_ms,
)

#: Why every ``fake_login`` double in this file takes ``**_kwargs``.
#:
#: ``ProviderController.login`` forwards ``signal=`` to EVERY login callable so
#: a host can offer a cancel without knowing which flavour of provider it is
#: talking to. A double that pins the older, narrower signature therefore
#: raises ``TypeError`` inside the controller — and the surfaces above it
#: report that as an ordinary "login failed", so the assertion misreads as a
#: real provider failure rather than as a stale test. Stated once here and
#: referenced from each double (agent review round 1, nit-2).
LOGIN_DOUBLE_SIGNATURE_NOTE = (
    "ProviderController.login forwards signal= to every login callable; a "
    "double with a narrower signature raises TypeError and misreads as a "
    "provider failure."
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
        return [types.SimpleNamespace(**r) for r in rows if not r.get("disabled_cause")]

    def upsert_credential(self, provider, credential):
        declared = credential.get("type")
        cred_type = (
            declared
            if declared in ("oauth", "api_key")
            else ("oauth" if credential.get("refresh") and credential.get("access") else "api_key")
        )
        row = {
            "id": self._next_id,
            "provider": provider,
            "credential_type": cred_type,
            "data": credential,
            "identity_key": credential.get("email"),
        }
        self._next_id += 1
        self.rows.append(row)
        return row

    def active_local_credential(self, provider, endpoint):
        for row in reversed(self.rows):
            if row["provider"] == provider and row["data"].get("endpoint") == endpoint:
                return None if row.get("disabled_cause") else types.SimpleNamespace(**row)
        return None

    def disable_credential(self, credential_id, cause):
        for row in self.rows:
            if row["id"] == credential_id:
                row["disabled_cause"] = cause

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
        # `credential_invalid` is a real `OAuthAccess` field (default False),
        # and the controller reads it directly; a double that omits it is the
        # only shape that would ever need a `getattr` fallback (#618 R12).
        return types.SimpleNamespace(
            access_token=f"tok-{account_id}",
            credential_id=0,
            account_id=account_id,
            email=email,
            org_id=None,
            api_endpoint=None,
            kind="oauth",
            raw=None,
            credential_invalid=False,
        )

    @pytest.mark.asyncio
    async def test_every_account_gets_its_own_report(self, controller, store, monkeypatch) -> None:
        store.oauth_accounts["anthropic"] = [
            self._account("first@example.com", "acct-1"),
            self._account("second@example.com", "acct-2"),
        ]
        seen: list[tuple[str, str | None]] = []

        async def fake_fetch(
            client,
            provider,
            *,
            api_key,
            access_token,
            account_id,
            oauth_creds=None,
            extra_creds=None,
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
            client,
            provider,
            *,
            api_key,
            access_token,
            account_id,
            oauth_creds=None,
            extra_creds=None,
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
            client,
            provider,
            *,
            api_key,
            access_token,
            account_id,
            oauth_creds=None,
            extra_creds=None,
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
            client,
            provider,
            *,
            api_key,
            access_token,
            account_id,
            oauth_creds=None,
            extra_creds=None,
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


def test_the_rescue_entry_labels_a_router_but_only_on_an_aggregator(controller) -> None:
    """The rescue path decides ``routed`` from the ID, and nothing else covered it.

    This is the branch the reported bug actually takes: a user on
    ``radient/auto`` whose live listing could not be had gets their current
    model rebuilt here, with no listing row to read a ``-1`` off. So the label
    has to come from the id — and that is bespoke logic no other construction
    site shares, which is why it needs its own assertion rather than riding on
    the parser's tests.

    The negative case is the R1 scoping fix at this call site: ``ollama/auto``
    reaches this same branch, and a local model the user happened to name
    ``auto`` must keep its genuine ``free`` rather than being relabelled.
    """
    router_spec = ModelSpec(
        provider="radient",
        model_id="auto",
        display_name="Automatic",
        context_window=1_048_576,
    )
    entry = controller.entry_for("radient", "auto", spec=router_spec)
    assert entry is not None
    assert entry.routed is True, "the router must carry its label without a listing"

    # A non-router id on the SAME aggregator: the flag is about this endpoint,
    # not about the provider being an aggregator.
    other_spec = ModelSpec(
        provider="radient",
        model_id="vendor/model",
        display_name="Vendor Model",
        context_window=32_000,
    )
    other = controller.entry_for("radient", "vendor/model", spec=other_spec)
    assert other is not None
    assert other.routed is False

    # R1: the id leg is aggregator-scoped. Ollama's listing is the user's own
    # filesystem, so `auto` is a name a user can simply give a local model, and
    # ollama is the one provider whose zero price is a REAL free.
    ollama_spec = ModelSpec(
        provider="ollama",
        model_id="auto",
        display_name="auto",
        context_window=8_192,
    )
    local = controller.entry_for("ollama", "auto", spec=ollama_spec)
    assert local is not None
    assert local.routed is False, "a local model named `auto` is not a meta-route"


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
    a caller that filters a model list on the first would show an empty picker.

    Raises the error sqlite ACTUALLY raises for a locked database. It used to
    raise ``RuntimeError``, which no sqlite call produces, and that mattered: it
    let the test pass against a catch-all ``except Exception`` that also
    swallowed ``ProgrammingError`` from cross-thread use, reporting a bug as
    this designed degradation (D18).
    """

    def boom(provider=None):
        raise sqlite3.OperationalError("database is locked")

    store.list_credentials = boom  # type: ignore[assignment]
    assert controller.usable_providers() is None


def test_a_catalogue_survives_a_store_that_cannot_be_read(controller, store) -> None:
    """The catalogue is what the picker paints; a locked SQLite file must cost the
    auth ANNOTATION, never the list. Unknown reads as connected because the
    alternative marks every model as needing a login the app never checked for."""

    def boom(provider=None):
        raise sqlite3.OperationalError("database is locked")

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
        # `credential_invalid` is a real `OAuthAccess` field (default False),
        # and the controller reads it directly; a double that omits it is the
        # only shape that would ever need a `getattr` fallback (#618 R12).
        return types.SimpleNamespace(
            access_token=f"tok-{account_id}",
            credential_id=0,
            account_id=account_id,
            email=email,
            org_id=None,
            api_endpoint=None,
            kind="oauth",
            raw=None,
            credential_invalid=False,
        )

    @pytest.mark.asyncio
    async def test_a_warm_row_is_served_without_crossing_the_network(
        self, controller, store, monkeypatch
    ) -> None:
        store.oauth_accounts["anthropic"] = [self._account("me@example.com", "acct-1")]
        calls = 0

        async def fake_fetch(
            client,
            provider,
            *,
            api_key,
            access_token,
            account_id,
            oauth_creds=None,
            extra_creds=None,
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
            client,
            provider,
            *,
            api_key,
            access_token,
            account_id,
            oauth_creds=None,
            extra_creds=None,
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
            client,
            provider,
            *,
            api_key,
            access_token,
            account_id,
            oauth_creds=None,
            extra_creds=None,
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
            client,
            provider,
            *,
            api_key,
            access_token,
            account_id,
            oauth_creds=None,
            extra_creds=None,
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
            client,
            provider,
            *,
            api_key,
            access_token,
            account_id,
            oauth_creds=None,
            extra_creds=None,
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
            client,
            provider,
            *,
            api_key,
            access_token,
            account_id,
            oauth_creds=None,
            extra_creds=None,
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
        # `credential_invalid` is a real `OAuthAccess` field (default False),
        # and the controller reads it directly; a double that omits it is the
        # only shape that would ever need a `getattr` fallback (#618 R12).
        return types.SimpleNamespace(
            access_token=f"tok-{account_id}",
            credential_id=0,
            account_id=account_id,
            email=email,
            org_id=None,
            api_endpoint=None,
            kind="oauth",
            raw=None,
            credential_invalid=False,
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
            client,
            provider,
            *,
            api_key,
            access_token,
            account_id,
            oauth_creds=None,
            extra_creds=None,
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
            client,
            provider,
            *,
            api_key,
            access_token,
            account_id,
            oauth_creds=None,
            extra_creds=None,
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
        """An unavailable account is no longer latched: it re-probes on its own
        once the retry cadence elapses, and ``r`` still retries immediately.

        Updated for the recovery-discoverability contract: reaching the ceiling
        schedules a jittered ``USAGE_UNAVAILABLE_RETRY_MS`` probe instead of
        setting ``next_probe_at_ms=None``, so the background warmer (and any
        non-forced fetch after the cadence) discovers a recovered provider
        without ``r``.
        """
        import time as _time

        store.oauth_accounts["anthropic"] = [self._account("new@example.com", "acct-new")]
        calls = 0
        succeed = False

        async def fake_fetch(
            client,
            provider,
            *,
            api_key,
            access_token,
            account_id,
            oauth_creds=None,
            extra_creds=None,
        ):
            nonlocal calls
            calls += 1
            if succeed:
                return self._report("new@example.com", 7.0)
            return None

        monkeypatch.setattr("local_operator.providers.controller.fetch_usage", fake_fetch)

        def _expire_and_backdate() -> None:
            """Expire the cache row and put every next_probe in the past, so the
            next fetch re-probes instead of serving the fresh payload."""
            key = controller._usage_cache_key("anthropic")
            cache = controller._usage_cache_store()
            assert cache is not None
            existing = cache.get(key, include_expired=True)
            if existing:
                now_ms = int(_time.time() * 1000)
                for report in existing:
                    report.next_probe_at_ms = now_ms - 1
                cache.set(key, "anthropic", existing, expires_at_ms=now_ms - 1000)

        async def _drive_to_unavailable() -> None:
            for _ in range(USAGE_ACCOUNT_MAX_FAILURES):
                _expire_and_backdate()
                await controller.fetch_usage(["anthropic"])

        await _drive_to_unavailable()
        calls_before = calls
        succeed = True

        # 1. No probe BEFORE the cadence: the cache row is fresh for
        #    ~USAGE_UNAVAILABLE_RETRY_MS, so a non-forced fetch serves it.
        idle = await controller.fetch_usage(["anthropic"])
        assert calls == calls_before
        assert idle[0].usage_unavailable is True

        # 2. Force still retries immediately, regardless of the schedule.
        forced = await controller.fetch_usage(["anthropic"], force_refresh=True)
        assert calls == calls_before + 1
        assert forced[0].usage_unavailable is False
        assert forced[0].consecutive_failures == 0

        # 3. Probe AFTER the cadence, WITHOUT force: re-drive to unavailable,
        #    then expire the row and backdate next_probe past the cadence. A
        #    non-forced fetch now probes and a 200 clears the latch.
        succeed = False
        await _drive_to_unavailable()
        calls_before = calls
        succeed = True
        _expire_and_backdate()
        auto = await controller.fetch_usage(["anthropic"])  # NOT forced
        assert calls == calls_before + 1
        assert auto[0].usage_unavailable is False
        assert auto[0].consecutive_failures == 0
        assert auto[0].limits[0].amount.used == 7.0

    @pytest.mark.asyncio
    async def test_a_latched_account_recovers_on_its_own_after_the_retry_cadence(
        self, controller, store, monkeypatch
    ) -> None:
        """Recovery is discoverable without ``r``: a latched account re-probes
        once the jittered ``USAGE_UNAVAILABLE_RETRY_MS`` cadence elapses, and a
        200 clears ``usage_unavailable``, ``consecutive_failures`` and the panel
        note end to end.

        This is the operator's stated need: accounts are out of quota now, and
        the panel must show on its own when quota becomes available again. The
        old latch set ``next_probe_at_ms=None`` and ``_account_in_backoff``
        returned True on the flag alone forever, so only a manual ``r`` could
        ever discover the recovery.
        """
        import time as _time

        from local_operator.tui.widgets.usage_panel import _account_status_note

        store.oauth_accounts["anthropic"] = [self._account("me@example.com", "acct-me")]
        succeed = False

        async def fake_fetch(
            client,
            provider,
            *,
            api_key,
            access_token,
            account_id,
            oauth_creds=None,
            extra_creds=None,
        ):
            if succeed:
                return self._report("me@example.com", 23.0)
            return None

        monkeypatch.setattr("local_operator.providers.controller.fetch_usage", fake_fetch)

        # Drive to the unavailable ceiling.
        latched: list[UsageReport] = []
        for _ in range(USAGE_ACCOUNT_MAX_FAILURES):
            key = controller._usage_cache_key("anthropic")
            cache = controller._usage_cache_store()
            assert cache is not None
            existing = cache.get(key, include_expired=True)
            if existing:
                now_ms = int(_time.time() * 1000)
                for report in existing:
                    report.next_probe_at_ms = now_ms - 1
                cache.set(key, "anthropic", existing, expires_at_ms=now_ms - 1000)
            latched = await controller.fetch_usage(["anthropic"])

        assert latched[0].usage_unavailable is True
        assert latched[0].consecutive_failures == USAGE_ACCOUNT_MAX_FAILURES
        # The latched row carries the panel note.
        assert _account_status_note(latched[0], int(_time.time() * 1000)) != ""

        # The cadence elapses: expire the row and backdate next_probe, the same
        # lever the loop above pulls. A NON-forced fetch now re-probes.
        succeed = True
        key = controller._usage_cache_key("anthropic")
        cache = controller._usage_cache_store()
        assert cache is not None
        now_ms = int(_time.time() * 1000)
        for report in latched:
            report.next_probe_at_ms = now_ms - 1
        cache.set(key, "anthropic", latched, expires_at_ms=now_ms - 1000)

        recovered = await controller.fetch_usage(["anthropic"])  # NOT forced
        assert recovered[0].usage_unavailable is False
        assert recovered[0].consecutive_failures == 0
        assert recovered[0].limits[0].amount.used == 23.0
        # And the panel note clears with it.
        assert _account_status_note(recovered[0], int(_time.time() * 1000)) == ""

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
            client,
            provider,
            *,
            api_key,
            access_token,
            account_id,
            oauth_creds=None,
            extra_creds=None,
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
            client,
            provider,
            *,
            api_key,
            access_token,
            account_id,
            oauth_creds=None,
            extra_creds=None,
        ):
            return self._report("live@example.com", 22.0)

        monkeypatch.setattr("local_operator.providers.controller.fetch_usage", fake_fetch)
        reports = await controller.fetch_usage(["anthropic"])
        assert [r.identity for r in reports] == ["stale@example.com", "live@example.com"]
        assert reports[0].limits == []
        assert reports[0].consecutive_failures == 1
        assert reports[1].limits[0].amount.used == 22.0

    @pytest.mark.asyncio
    async def test_a_dead_grant_is_not_a_transient_failure(
        self, controller, store, monkeypatch
    ) -> None:
        """The reported defect, at the controller grain.

        A permanently dead grant used to walk the same path as a network blip:
        bump the streak, back off, and after the ceiling render `usage
        unavailable`. It must instead be marked as needing a re-login, keep
        its last-known numbers, and take no retry budget at all.
        """
        store.upsert_credential(
            "anthropic",
            {"refresh": "r", "access": "a", "email": "dead@example.com", "account_id": "acct-dead"},
        )
        dead = self._account("dead@example.com", "acct-dead")
        dead.access_token = ""
        dead.credential_invalid = True
        store.oauth_accounts["anthropic"] = [dead]
        probed: list[str | None] = []

        async def fake_fetch(
            client,
            provider,
            *,
            api_key,
            access_token,
            account_id,
            oauth_creds=None,
            extra_creds=None,
        ):
            probed.append(account_id)
            raise AssertionError("a dead grant must never be probed")

        monkeypatch.setattr("local_operator.providers.controller.fetch_usage", fake_fetch)
        reports = await controller.fetch_usage(["anthropic"])

        assert probed == []  # no request is made with an empty bearer
        assert len(reports) == 1
        assert reports[0].credential_invalid is True
        # The transient machinery stays untouched: no streak, no ceiling, and
        # no scheduled retry that could never succeed.
        assert reports[0].usage_unavailable is False
        assert reports[0].consecutive_failures == 0
        assert reports[0].next_probe_at_ms is None

    @pytest.mark.asyncio
    async def test_a_dead_grant_keeps_its_last_known_numbers(
        self, controller, store, monkeypatch
    ) -> None:
        """The login is still real, so its final reading is still the truth
        about it — the panel shows the meters beside the re-login note."""
        store.upsert_credential(
            "anthropic",
            {"refresh": "r", "access": "a", "email": "dead@example.com", "account_id": "acct-dead"},
        )
        dead = self._account("dead@example.com", "acct-dead")
        dead.access_token = ""
        dead.credential_invalid = True
        store.oauth_accounts["anthropic"] = [dead]

        async def fake_fetch(
            client,
            provider,
            *,
            api_key,
            access_token,
            account_id,
            oauth_creds=None,
            extra_creds=None,
        ):
            return self._report("dead@example.com", 61.0)

        # One good round populates last-known...
        monkeypatch.setattr("local_operator.providers.controller.fetch_usage", fake_fetch)
        store.oauth_accounts["anthropic"] = [self._account("dead@example.com", "acct-dead")]
        first = await controller.fetch_usage(["anthropic"])
        assert first[0].limits[0].amount.used == 61.0

        # ...then the grant dies.
        store.oauth_accounts["anthropic"] = [dead]
        second = await controller.fetch_usage(["anthropic"], force_refresh=True)
        assert second[0].credential_invalid is True
        assert second[0].limits[0].amount.used == 61.0

    @pytest.mark.asyncio
    async def test_a_working_bearer_clears_the_dead_grant_flag(
        self, controller, store, monkeypatch
    ) -> None:
        """After `/login`, a 200 is what proves the new grant works, so the
        state must heal itself rather than needing a cache wipe."""
        store.upsert_credential(
            "anthropic",
            {"refresh": "r", "access": "a", "email": "dead@example.com", "account_id": "acct-dead"},
        )
        dead = self._account("dead@example.com", "acct-dead")
        dead.access_token = ""
        dead.credential_invalid = True
        store.oauth_accounts["anthropic"] = [dead]

        async def fake_fetch(
            client,
            provider,
            *,
            api_key,
            access_token,
            account_id,
            oauth_creds=None,
            extra_creds=None,
        ):
            return self._report("dead@example.com", 5.0)

        monkeypatch.setattr("local_operator.providers.controller.fetch_usage", fake_fetch)
        first = await controller.fetch_usage(["anthropic"])
        assert first[0].credential_invalid is True

        # The user re-logs in: the row mints a bearer again.
        store.oauth_accounts["anthropic"] = [self._account("dead@example.com", "acct-dead")]
        healed = await controller.fetch_usage(["anthropic"], force_refresh=True)
        assert healed[0].credential_invalid is False
        assert healed[0].limits[0].amount.used == 5.0

    @pytest.mark.asyncio
    async def test_the_verdict_clears_on_an_automatic_poll_not_only_on_r(
        self, controller, store, usage_cache, monkeypatch
    ) -> None:
        """Review R1/Q1: the panel's own advice must be enough to fix it.

        The user reads `sign-in expired — /login kimi`, runs `/login`, and then
        does nothing else. The next AUTOMATIC poll has to return the account to
        normal. This is deliberately NOT a `force_refresh` test: `r` bypasses
        the backoff gate, so the three healing tests above all passed while a
        `credential_invalid` report was permanently skipped by the ordinary
        cycle — the verdict was a one-way latch that outlived its own cause.
        """
        store.upsert_credential(
            "anthropic",
            {"refresh": "r", "access": "a", "email": "dead@example.com", "account_id": "acct-dead"},
        )
        dead = self._account("dead@example.com", "acct-dead")
        dead.access_token = ""
        dead.credential_invalid = True
        store.oauth_accounts["anthropic"] = [dead]

        async def fake_fetch(
            client,
            provider,
            *,
            api_key,
            access_token,
            account_id,
            oauth_creds=None,
            extra_creds=None,
        ):
            return self._report("dead@example.com", 7.0)

        monkeypatch.setattr("local_operator.providers.controller.fetch_usage", fake_fetch)
        first = await controller.fetch_usage(["anthropic"])
        assert first[0].credential_invalid is True

        # The grant is re-minted (the store hands back a bearer again), but
        # the cached row still carries the verdict. It is EXPIRED rather than
        # invalidated: the round-1 latch lived in `_account_in_backoff`, which
        # only sees a `prior` when the cache has a (stale) row to hand it.
        # Invalidating the row here — as this test originally did — emptied
        # `previous_by_id`, so `prior` was None and the gate short-circuited
        # before the latch was ever consulted; the test then passed with the
        # bug restored (#618 R10). Expiring the row is also the state the
        # panel's own poll cycle reaches: the login path drops the row, but a
        # row that merely aged out must heal on the next automatic poll too.
        store.oauth_accounts["anthropic"] = [self._account("dead@example.com", "acct-dead")]
        real_now = usage_cache._now_ms()
        monkeypatch.setattr(
            UsageCacheStore,
            "_now_ms",
            staticmethod(lambda: real_now + 2 * USAGE_REPORT_TTL_MS),
        )
        # Precondition for the assertion below to mean anything: the flagged
        # row is still on hand as `previous`, so the gate is actually reached.
        assert usage_cache.get(controller._usage_cache_key("anthropic")) is None
        stale = usage_cache.get(controller._usage_cache_key("anthropic"), include_expired=True)
        assert stale is not None and stale[0].credential_invalid is True

        # No `r`: this is the ordinary background poll.
        healed = await controller.fetch_usage(["anthropic"])
        assert healed[0].credential_invalid is False
        assert healed[0].limits[0].amount.used == 7.0

    @pytest.mark.asyncio
    async def test_a_transient_miss_does_not_latch_a_healthy_credential(
        self, controller, store, monkeypatch
    ) -> None:
        """Review R1/Q1, the worse half: a live grant must never acquire the
        note. Reaching `_mark_account_failure` means the bearer minted and the
        usage ENDPOINT failed — direct evidence the grant is alive — so a
        stale verdict has to be dropped rather than carried forward."""
        now = 1_000_000
        previous = self._report("dead@example.com", 12.0)
        previous.credential_invalid = True

        marked = controller._mark_account_failure(
            previous, provider="anthropic", identity="dead@example.com", now_ms=now
        )

        assert marked.credential_invalid is False
        assert marked.consecutive_failures == 1
        # And with the verdict gone it is an ordinary backoff, not a permanent
        # exclusion from the automatic cycle.
        assert controller._account_in_backoff(marked, now + 10_000_000, force=False) is False

    @pytest.mark.asyncio
    async def test_login_drops_the_cached_row_carrying_the_verdict(
        self, controller, store, usage_cache, monkeypatch
    ) -> None:
        """QA Q6 on #618: the third leg of the R1 fix, pinned at its call site.

        Re-authenticating an ALREADY-stored account keeps the account
        fingerprint, so the cache key does not move and the fingerprint-based
        self-invalidation never fires. `login()` has to drop the row itself,
        and until this test nothing exercised that wiring: the healing test
        above went through the helper by hand, so deleting the call from
        `login()` broke zero tests. Here the login path is driven end to end
        (a canned OAuth exchange over the same stored identity) and the row
        is read back from the shared cache.
        """
        store.upsert_credential(
            "anthropic",
            {"refresh": "r", "access": "a", "email": "dead@example.com", "account_id": "acct-dead"},
        )
        dead = self._account("dead@example.com", "acct-dead")
        dead.access_token = ""
        dead.credential_invalid = True
        store.oauth_accounts["anthropic"] = [dead]

        async def fake_fetch(
            client,
            provider,
            *,
            api_key,
            access_token,
            account_id,
            oauth_creds=None,
            extra_creds=None,
        ):
            raise AssertionError("a dead grant must not be probed")

        monkeypatch.setattr("local_operator.providers.controller.fetch_usage", fake_fetch)
        first = await controller.fetch_usage(["anthropic"])
        assert first[0].credential_invalid is True
        key = controller._usage_cache_key("anthropic")
        assert usage_cache.get(key) is not None, "the verdict is cached and fresh"

        # The same identity logs in again: the fake's upsert appends rather
        # than replacing, so pin the fingerprint to prove the key is unchanged
        # and the drop below is the login's own doing, not a key move.
        # ``**_kwargs``: see LOGIN_DOUBLE_SIGNATURE_NOTE.
        async def fake_login(_callbacks, **_kwargs):
            return {
                "access_token": "t2",
                "refresh_token": "r2",
                "email": "dead@example.com",
                "account_id": "acct-dead",
            }

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
        monkeypatch.setattr(
            "local_operator.providers.controller.invalidate_listing", lambda provider_id: 1
        )
        monkeypatch.setattr(controller, "_account_fingerprint", lambda provider: "pinned")
        # Re-key under the pinned fingerprint so the row `login()` must drop is
        # the one that would otherwise be served ahead of the next fetch.
        reports = usage_cache.get(key, include_expired=True)
        assert reports is not None
        usage_cache.set(
            controller._usage_cache_key("anthropic"),
            "anthropic",
            reports,
            expires_at_ms=usage_cache._now_ms() + USAGE_REPORT_TTL_MS,
        )
        pinned_key = controller._usage_cache_key("anthropic")
        assert usage_cache.get(pinned_key) is not None

        await controller.login("anthropic")

        assert controller._usage_cache_key("anthropic") == pinned_key, "key did not move"
        assert usage_cache.get(pinned_key, include_expired=True) is None, "login dropped the row"
        assert controller.usage_cache_age_ms("anthropic") is None

    @pytest.mark.asyncio
    async def test_a_dead_grant_is_not_excluded_from_the_automatic_cycle(
        self, controller, store, monkeypatch
    ) -> None:
        """The gate saved no network call and cost the only path that heals.

        `list_oauth_accesses` refreshes every row before this gate is reached,
        and the usage probe is short-circuited separately for a dead row, so
        skipping the cycle spent nothing — it only prevented the fetch that
        clears the flag.
        """
        report = UsageReport(provider="kimi", fetched_at=1_000, identity="cred:8")
        report.credential_invalid = True
        assert controller._account_in_backoff(report, 2_000, force=False) is False

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
            client,
            provider,
            *,
            api_key,
            access_token,
            account_id,
            oauth_creds=None,
            extra_creds=None,
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


class TestQwenCloudConsoleRoute:
    """The console route for an account whose OAuth grant can never refresh.

    This route failed SILENTLY in production before it existed, and the
    symptom was an empty ``/usage`` table indistinguishable from "this
    provider has no quota endpoint" — the failure mode usage.py:36-44 says
    this module has already been bitten by four times. The cause is that the
    two identity enumerators disagree for ``alibaba-token-plan``:
    ``list_oauth_identities`` names the stored login while
    ``list_oauth_accesses`` mints no bearer (the row's ``expires`` is in the
    past and no ``ProviderDefinition`` declares a refresh token, so
    ``_ensure_oauth_fresh`` can never revive it). ``expected`` is therefore
    non-empty, ``_fetch_provider`` takes the ``if expected:`` branch, every
    identity hits ``access is None``, and the API-key route below it is
    unreachable.

    These tests pin the MECHANISM, not just the outcome: reinstating a bare
    ``continue`` at the ``access is None`` point must turn them red.
    """

    #: The console gateway's own host, so a test can assert it was — or was
    #: never — contacted without matching on a path substring.
    CONSOLE_HOST = "cs-data.qwencloud.com"

    #: A placeholder session cookie. The real credential is a full-account
    #: console ticket and never appears in this repository.
    TICKET = "fake-console-ticket"

    def _dead_grant(self, store) -> str:
        """Store the real account's shape: a login that mints no bearer.

        The row makes ``list_oauth_identities`` non-empty while
        ``oauth_accounts`` stays unset, which is exactly the disagreement
        that made the route unreachable.
        """
        email = "fake@example.test"
        store.upsert_credential(
            "alibaba-token-plan",
            {
                "type": "oauth",
                "access": "fake-mgmt",
                "email": email,
                "expires": 1,
                "account_id": "fake-acct",
            },
        )
        return email

    def _ticket(self, store) -> None:
        store.upsert_credential(
            "qwencloud-console",
            {"ticket": self.TICKET, "project_id": "qwencloud-console:personal"},
        )

    @staticmethod
    def _spy(monkeypatch, controller, report=None):
        """Record every ``_fetch_one`` call, mirroring the double at :541."""
        calls: list[dict[str, Any]] = []
        original = type(controller)._fetch_one

        async def _record(client, provider, *, access=None, extra_creds=None):
            calls.append({"provider": provider, "access": access, "extra_creds": extra_creds})
            if report is not None:
                return report
            return await original(
                controller, client, provider, access=access, extra_creds=extra_creds
            )

        monkeypatch.setattr(controller, "_fetch_one", _record)
        return calls

    @pytest.mark.asyncio
    async def test_a_stored_ticket_reaches_the_console_route(
        self, controller, store, monkeypatch
    ) -> None:
        """The point of the slice: ``_fetch_one`` RUNS, carrying the ticket.

        Asserting only on the returned report would still pass if the report
        arrived by some other route, so this pins the call itself.
        """
        email = self._dead_grant(store)
        self._ticket(store)
        expected = UsageReport(provider="alibaba-token-plan", limits=[])
        calls = self._spy(monkeypatch, controller, report=expected)

        reports = await controller.fetch_usage(["alibaba-token-plan"])

        assert len(calls) == 1, "a bare `continue` here is the defect under test"
        assert calls[0]["provider"] == "alibaba-token-plan"
        # The console credential is not an OAuth account: it arrives beside
        # `access=None`, which is what makes the api-key early-return skip it.
        assert calls[0]["access"] is None
        assert calls[0]["extra_creds"] is not None
        assert calls[0]["extra_creds"]["ticket"] == self.TICKET
        # And the report reaches the panel under the stored identity.
        assert [r.identity for r in reports] == [email]

    @pytest.mark.asyncio
    async def test_the_ticket_reaches_the_usage_dispatcher(
        self, controller, store, monkeypatch
    ) -> None:
        """The other half of reachability: ``_fetch_one`` FORWARDS the ticket.

        Patching ``_fetch_one`` proves it is called but says nothing about
        what it does, so dropping ``extra_creds`` from the ``fetch_usage``
        call would leave that test green while the fetcher goes unreachable
        again — the dead-code defect this slice exists to avoid. This runs
        the real ``_fetch_one`` and pins the dispatcher's arguments instead.
        """
        self._dead_grant(store)
        self._ticket(store)
        seen: list[dict[str, Any] | None] = []

        async def fake_fetch(
            client,
            provider,
            *,
            api_key,
            access_token,
            account_id,
            oauth_creds=None,
            extra_creds=None,
        ):
            seen.append(extra_creds)
            return UsageReport(provider=provider, limits=[])

        monkeypatch.setattr("local_operator.providers.controller.fetch_usage", fake_fetch)

        await controller.fetch_usage(["alibaba-token-plan"])

        assert len(seen) == 1, "the dispatcher must be reached exactly once"
        assert seen[0] is not None, "dropping extra_creds here makes the fetcher dead code"
        assert seen[0]["ticket"] == self.TICKET

    @pytest.mark.asyncio
    async def test_no_ticket_means_no_console_request(self, controller, store, monkeypatch) -> None:
        """Without a stored ticket the route must not fire at all.

        The guard against the opposite defect: an unconditional console
        attempt would contact the gateway for every dead grant on the box.
        """
        self._dead_grant(store)
        calls = self._spy(monkeypatch, controller)

        def no_network(request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError(f"no ticket stored, yet {request.url} was contacted")

        # Bound before patching: the replacement builds a real client, so
        # reading the name through the module would recurse into itself.
        real_client = httpx.AsyncClient

        def _mock_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
            return real_client(transport=httpx.MockTransport(no_network))

        monkeypatch.setattr("local_operator.providers.controller.httpx.AsyncClient", _mock_client)

        reports = await controller.fetch_usage(["alibaba-token-plan"])

        assert calls == [], "no credential to spend, so nothing to fetch"
        # Unchanged from before the console route existed: the expected
        # identity is still named, with no numbers behind it.
        assert [r.identity for r in reports] == ["fake@example.test"]
        assert reports[0].limits == []

    @pytest.mark.asyncio
    async def test_the_ticket_reaches_the_api_key_route_with_no_oauth_row(
        self, controller, store, monkeypatch
    ) -> None:
        """The console route must not require a DEAD OAuth row to exist.

        Spending the ticket only at the ``access is None`` point inside
        ``if expected:`` made the whole feature depend on a present-but-dead
        grant. The real store holds an ``alibaba-token-plan`` api_key row, so
        ``lop logout alibaba-token-plan`` drops the OAuth row, leaves the
        api_key row, and ``/usage`` rendered NOTHING for a valid ticket --
        the silent-empty-table symptom this feature exists to fix (QA D3).
        """
        store.upsert_credential(
            "alibaba-token-plan", {"key": "fake-inference-key", "type": "api_key"}
        )
        self._ticket(store)
        assert (
            controller._expected_oauth_identities("alibaba-token-plan") == []
        ), "no OAuth row: this is the API-key route, not the dead-grant one"
        expected = UsageReport(provider="alibaba-token-plan", limits=[])
        calls = self._spy(monkeypatch, controller, report=expected)

        await controller.fetch_usage(["alibaba-token-plan"])

        assert len(calls) == 1
        assert calls[0]["access"] is None
        assert calls[0]["extra_creds"] is not None, "the API-key route dropped the ticket"
        assert calls[0]["extra_creds"]["ticket"] == self.TICKET

    @pytest.mark.asyncio
    async def test_the_api_key_route_passes_no_ticket_for_other_providers(
        self, controller, store, monkeypatch
    ) -> None:
        """Threading the ticket through the API-key route must not widen it.

        That route is shared by every provider with no OAuth row, so the
        storage-id guard inside ``_qwencloud_console_creds`` is the only thing
        keeping a full-account console cookie off another provider's fetch.
        Asserted by EXECUTION over the whole registry rather than by reading
        the guard.
        """
        self._ticket(store)
        store.upsert_credential("deepseek", {"key": "fake-deepseek-key", "type": "api_key"})
        calls = self._spy(monkeypatch, controller, report=UsageReport(provider="deepseek"))

        await controller.fetch_usage(["deepseek"])

        assert len(calls) == 1
        assert calls[0]["extra_creds"] is None, "deepseek must never carry the console cookie"

        from local_operator.providers.registry import PROVIDER_REGISTRY

        resolved = []
        for definition in PROVIDER_REGISTRY:
            creds, _ = await controller._qwencloud_console_creds(definition.id)
            if creds is not None:
                resolved.append(definition.id)
        resolved.sort()
        assert resolved == ["alibaba-token-plan", "alibaba-token-plan-oauth"]

    @pytest.mark.asyncio
    async def test_another_provider_never_queries_the_ticket_namespace(
        self, controller, store, monkeypatch
    ) -> None:
        """``_qwencloud_console_creds`` is guarded on the storage id.

        Without the guard every provider's fetch would query the
        ``qwencloud-console`` namespace on every cycle, and any provider with
        a dead grant would try to spend a QwenCloud cookie on its own API.
        """
        self._ticket(store)
        store.upsert_credential(
            "anthropic",
            {"type": "oauth", "access": "tok", "email": "other@example.test", "expires": 1},
        )
        asked: list[str | None] = []
        real_list = store.list_credentials

        def _watch(provider=None):
            asked.append(provider)
            return real_list(provider)

        monkeypatch.setattr(store, "list_credentials", _watch)
        calls = self._spy(monkeypatch, controller)

        reports = await controller.fetch_usage(["anthropic"])

        assert "qwencloud-console" not in asked, "the ticket namespace is QwenCloud's alone"
        assert calls == [], "anthropic's dead grant must not reach the console route"
        assert [r.identity for r in reports] == ["other@example.test"]
        assert await controller._qwencloud_console_creds("anthropic") == (None, None)

    @pytest.mark.asyncio
    async def test_the_console_report_merges_into_one_credits_row(
        self, controller, store, monkeypatch
    ) -> None:
        """Exactly ONE ``credits-7d`` survives at the panel's own grain.

        ``usage_panel`` flattens ``[limit for report in reports for limit in
        report.limits]`` with no dedup by id, and ``_merge_account_reports``
        merges at ACCOUNT grain without concatenating limits. So a console
        report landing under a DIFFERENT identity key than the stored login
        renders the same window twice. Measured the way the panel measures it.
        """
        email = self._dead_grant(store)
        self._ticket(store)
        console_report = UsageReport(
            provider="alibaba-token-plan",
            limits=[
                UsageLimit(
                    id="credits-7d",
                    label="Credits (7d)",
                    amount=UsageAmount(used=24.05, limit=100.0, unit="percent"),
                )
            ],
        )
        self._spy(monkeypatch, controller, report=console_report)

        reports = await controller.fetch_usage(["alibaba-token-plan"])

        limit_ids = [limit.id for report in reports for limit in report.limits]
        assert limit_ids == ["credits-7d"], "a second row means two identity keys"
        assert [r.identity for r in reports] == [email]

    @pytest.mark.asyncio
    async def test_two_dead_grants_spend_the_one_ticket_once(
        self, controller, store, monkeypatch
    ) -> None:
        """One console session for the account, not one per login.

        The ticket is account-wide, so attempting the route per identity
        would send a request each and land a report under each. The panel
        flattens limits with no dedup by id (usage_panel.py:407), so the same
        7-day window would render TWICE. This is the duplicate-row failure
        the ``live[...]`` key mirroring prevents for ONE report, arriving
        through a second door: mirroring cannot help when two identities each
        produce their own console report.
        """
        emails = []
        for name in ("one", "two"):
            email = f"{name}@example.test"
            store.upsert_credential(
                "alibaba-token-plan",
                {
                    "type": "oauth",
                    "access": "fake-mgmt",
                    "email": email,
                    "expires": 1,
                    "account_id": f"acct-{name}",
                },
            )
            emails.append(email)
        self._ticket(store)
        assert controller._expected_oauth_identities("alibaba-token-plan") == emails

        def _report() -> UsageReport:
            # A fresh object per call: one shared instance would collapse the
            # duplicate by identity rather than by the fix under test.
            return UsageReport(
                provider="alibaba-token-plan",
                limits=[
                    UsageLimit(
                        id="credits-7d",
                        label="Credits (7d)",
                        amount=UsageAmount(used=24.05, limit=100.0, unit="percent"),
                    )
                ],
            )

        calls: list[dict[str, Any] | None] = []

        async def _record(client, provider, *, access=None, extra_creds=None):
            calls.append(extra_creds)
            return _report()

        monkeypatch.setattr(controller, "_fetch_one", _record)

        reports = await controller.fetch_usage(["alibaba-token-plan"])

        assert len(calls) == 1, "one account-wide ticket, so one request per cycle"
        limit_ids = [limit.id for report in reports for limit in report.limits]
        assert limit_ids == ["credits-7d"], "the same window must not render twice"
        # The second login keeps its row on the panel; it simply carries no
        # numbers, which is what it did before the console route existed.
        assert [r.identity for r in reports] == emails

    @pytest.mark.asyncio
    async def test_a_failing_ticket_is_not_retried_per_identity(
        self, controller, store, monkeypatch
    ) -> None:
        """The attempt is counted, not the success.

        A ticket that answers ``None`` (expired -- the gateway returns 200
        with an errorCode) is a property of the TICKET, not of the identity
        that happened to reach it. Counting only successes would retry the
        same dead cookie once per expected login, turning one useless
        request into N on every refresh of a multi-login account.
        """
        for name in ("one", "two"):
            store.upsert_credential(
                "alibaba-token-plan",
                {
                    "type": "oauth",
                    "access": "fake-mgmt",
                    "email": f"{name}@example.test",
                    "expires": 1,
                    "account_id": f"acct-{name}",
                },
            )
        self._ticket(store)
        calls: list[str] = []

        async def _record(client, provider, *, access=None, extra_creds=None):
            calls.append(provider)
            return None

        monkeypatch.setattr(controller, "_fetch_one", _record)

        await controller.fetch_usage(["alibaba-token-plan"])

        assert len(calls) == 1, "a dead ticket costs one request, not one per login"

    @pytest.mark.asyncio
    async def test_the_oauth_flavour_alias_reaches_the_route_too(
        self, controller, store, monkeypatch
    ) -> None:
        """``alibaba-token-plan-oauth`` stores under ``alibaba-token-plan``.

        The guard compares the STORAGE id (``credential_provider_id``), not
        the spelling the caller used, so both ids find the same ticket. The
        alias resolution itself is covered at
        ``test_credential_alias_resolves_storage_id``; this pins that the
        console guard honours it rather than matching a literal string.
        """
        self._ticket(store)

        assert (await controller._qwencloud_console_creds("alibaba-token-plan-oauth"))[
            0
        ] is not None
        assert await controller._qwencloud_console_creds(
            "alibaba-token-plan-oauth"
        ) == await controller._qwencloud_console_creds("alibaba-token-plan")

    @pytest.mark.asyncio
    async def test_a_ticketless_row_is_not_a_credential(self, controller, store) -> None:
        """A row in the namespace with no ``ticket`` must not count.

        The CLI writes the row before the capture completes, so an empty
        ticket is a real state — and returning it would send a request with
        no cookie, which reads as a generic auth failure rather than as
        "no ticket stored yet".
        """
        store.upsert_credential("qwencloud-console", {"ticket": "", "project_id": "p"})

        assert await controller._qwencloud_console_creds("alibaba-token-plan") == (None, None)


class TestQwenCloudTicketFromSecretStore:
    """The ticket VALUE now lives in the encrypted store, not in ``auth.db``.

    The defect these pin is not "the value moved" — it is that the read path
    used to answer every failure with the same silent ``None``. A locked
    hardened store then rendered identically to "this provider has no quota
    endpoint": the window vanished and the panel showed its generic empty
    string, which is the "bug that dresses itself as a plausible degraded
    state" failure ``controller.py``'s own 268-278 names. Four outcomes, four
    distinguishable results, and the two the user can act on carry the command
    that fixes them.
    """

    #: A placeholder value. The real credential is a full-account console
    #: ticket and never appears in this repository.
    TICKET = "fake-console-ticket"

    def _dead_grant(self, store) -> str:
        email = "fake@example.test"
        store.upsert_credential(
            "alibaba-token-plan",
            {
                "type": "oauth",
                "access": "fake-mgmt",
                "email": email,
                "expires": 1,
                "account_id": "fake-acct",
            },
        )
        return email

    def _migrated_row(self, store) -> None:
        """The post-migration metadata row: a POINTER, never the value.

        Mirrors ``store_ticket``'s write exactly — no ``ticket`` key, ever.
        """
        store.upsert_credential(
            "qwencloud-console",
            {
                "project_id": "qwencloud-console:personal",
                "captured_at": 1,
                "secret_name": "QWENCLOUD_CONSOLE_TICKET",
                "length": len(self.TICKET),
            },
        )

    @staticmethod
    def _store_exists(monkeypatch, tmp_path, exists: bool = True) -> None:
        """Make ``store_path(...).exists()`` answer ``exists``, via the config dir.

        **Do not patch ``store_path`` itself — that leak is not cosmetic.**
        ``local_operator.secrets.store`` does ``from ...keys import store_path``
        at module scope (``store.py``:54-59), so whichever value is installed
        the FIRST time that module is imported is bound there PERMANENTLY:
        monkeypatch reverts the attribute on ``keys`` and never the copy
        ``store`` already holds. An earlier version of this helper patched the
        attribute with a duck-typed stub, and the stub escaped this class and
        broke every later test that opened a real store — 33 failures in
        ``test_qwencloud_console.py`` reading ``TypeError: expected str, bytes
        or os.PathLike object``, with a traceback blaming ``secrets/store.py``,
        a file neither slice touches. Pointing the same patch at a real ``Path``
        does NOT fix it: measured, it still leaks and merely trades the
        ``TypeError`` for ``sqlite3.OperationalError: unable to open database
        file`` (35 failed). The capture is the defect; the stub's type was only
        how it announced itself.

        CI hid all of this because ``shard_tests.py --total 5`` happens to deal
        the two files into different shards, and that split is rebalanced by
        measured duration — so any new or retimed test can re-deal them
        together.

        So this steers ``LOCAL_OPERATOR_CONFIG_DIR`` instead, which
        :func:`~local_operator.paths.config_dir` re-reads on every call for
        exactly this reason (its own docstring says a module constant would
        freeze whatever the first importer saw). Nothing is captured, the
        patched state dies with the env var, and ``C1`` already drives the
        guard this way.
        """
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
        if exists:
            target = tmp_path / "secrets" / "store.db"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch()

    @staticmethod
    def _retrieval(monkeypatch, result):
        """Point ``access.retrieve_secret`` at ``result``; record every call."""
        from local_operator.secrets import access

        calls: list[str] = []

        def _retrieve(name, base=None):
            calls.append(name)
            if isinstance(result, BaseException):
                raise result
            return result

        monkeypatch.setattr(access, "retrieve_secret", _retrieve)
        return calls

    @staticmethod
    def _spy(monkeypatch, controller, report=None):
        calls: list[dict[str, Any]] = []

        async def _record(client, provider, *, access=None, extra_creds=None):
            calls.append({"provider": provider, "access": access, "extra_creds": extra_creds})
            return report

        monkeypatch.setattr(controller, "_fetch_one", _record)
        return calls

    # -- the value ---------------------------------------------------------

    @pytest.mark.asyncio
    async def test_the_value_reaches_the_fetcher_unchanged(
        self, controller, store, monkeypatch, tmp_path
    ) -> None:
        """C2. The decrypted value arrives in the shape ``usage.py`` expects.

        Asserting the report came back would pass on a fetcher that never saw
        the cookie, so this pins ``extra_creds`` itself — and pins that the
        metadata keys (``secret_name``, ``length``) do NOT ride along into a
        dict the fetcher interpolates into a header.
        """
        self._dead_grant(store)
        self._migrated_row(store)
        self._store_exists(monkeypatch, tmp_path)
        self._retrieval(monkeypatch, self.TICKET.encode())
        calls = self._spy(
            monkeypatch, controller, report=UsageReport(provider="alibaba-token-plan", limits=[])
        )

        await controller.fetch_usage(["alibaba-token-plan"])

        assert len(calls) == 1
        creds = calls[0]["extra_creds"]
        assert creds is not None
        assert creds["ticket"] == self.TICKET
        assert "secret_name" not in creds
        assert "length" not in creds

    @pytest.mark.asyncio
    async def test_a_legacy_plaintext_row_still_works(self, controller, store, monkeypatch) -> None:
        """C7. A user who has not migrated keeps a working ``/usage``.

        And reaches it WITHOUT the secret store: the legacy branch returns
        before the import, so an un-migrated user on a host with no store
        still never spawns a daemon.
        """
        self._dead_grant(store)
        store.upsert_credential(
            "qwencloud-console",
            {"ticket": self.TICKET, "project_id": "qwencloud-console:personal"},
        )
        retrievals = self._retrieval(monkeypatch, self.TICKET.encode())
        calls = self._spy(
            monkeypatch, controller, report=UsageReport(provider="alibaba-token-plan", limits=[])
        )

        await controller.fetch_usage(["alibaba-token-plan"])

        assert calls[0]["extra_creds"]["ticket"] == self.TICKET
        assert retrievals == [], "a legacy row must not touch the secret store at all"

    # -- the four outcomes -------------------------------------------------

    @pytest.mark.asyncio
    async def test_a_locked_store_paints_a_note_not_an_empty_panel(
        self, controller, store, monkeypatch, tmp_path
    ) -> None:
        """C3. The whole point: a locked store is VISIBLE and ACTIONABLE.

        Pins three separate things, because each fails independently: a report
        survives at all (without it the block vanishes), the note names the
        remedy, and the store's raw wire text — which says nothing a user can
        act on — never reaches the panel.
        """
        from local_operator.secrets.client import BrokerDenied

        email = self._dead_grant(store)
        self._migrated_row(store)
        self._store_exists(monkeypatch, tmp_path)
        self._retrieval(monkeypatch, BrokerDenied("no lop session is registered with the broker"))
        self._spy(monkeypatch, controller)

        reports = await controller.fetch_usage(["alibaba-token-plan"])

        assert len(reports) == 1, "a locked store must not make the block disappear"
        report = reports[0]
        assert report.identity == email
        assert report.notes is not None
        assert "lop secret unlock" in report.notes
        assert "registered with the broker" not in report.notes

    @pytest.mark.asyncio
    async def test_a_locked_store_never_crashes_the_gather(
        self, controller, store, monkeypatch, tmp_path
    ) -> None:
        """C4. ``fetch_usage`` fans out under ``asyncio.gather``.

        A raise here would take out every OTHER provider in the same fan-out,
        turning one locked ticket into a blank panel. Driven through the real
        ``fetch_usage`` rather than the helper for exactly that reason.
        """
        from local_operator.secrets.client import BrokerLocked

        self._dead_grant(store)
        self._migrated_row(store)
        store.upsert_credential("deepseek", {"key": "fake-deepseek-key", "type": "api_key"})
        self._store_exists(monkeypatch, tmp_path)
        self._retrieval(monkeypatch, BrokerLocked("locked"))
        self._spy(monkeypatch, controller, report=UsageReport(provider="deepseek", limits=[]))

        reports = await controller.fetch_usage(["alibaba-token-plan", "deepseek"])

        assert {r.provider for r in reports} >= {"alibaba-token-plan", "deepseek"}

    @pytest.mark.asyncio
    async def test_a_missing_secret_names_the_repair(
        self, controller, store, monkeypatch, tmp_path
    ) -> None:
        """C5. Metadata present, value gone — a real state with its own fix.

        Distinct from locked: unlocking cannot help, the ticket has to be
        re-captured, so the note must name a DIFFERENT command.
        """
        from local_operator.secrets.errors import SecretNotFound

        self._dead_grant(store)
        self._migrated_row(store)
        self._store_exists(monkeypatch, tmp_path)
        self._retrieval(monkeypatch, SecretNotFound("QWENCLOUD_CONSOLE_TICKET"))
        self._spy(monkeypatch, controller)

        reports = await controller.fetch_usage(["alibaba-token-plan"])

        assert len(reports) == 1
        assert reports[0].notes is not None
        assert "qwencloud-ticket set" in reports[0].notes
        assert "secret unlock" not in reports[0].notes

    @pytest.mark.asyncio
    async def test_an_unreadable_store_stays_silent(
        self, controller, store, monkeypatch, tmp_path
    ) -> None:
        """The fourth outcome, and the one that must NOT paint a note.

        A note is a promise that the user can act. A corrupt store offers no
        command that fixes it, so inventing one would send them at a remedy
        that cannot work — the opposite failure to the silent None.
        """
        from local_operator.secrets.errors import SecretStoreError

        self._dead_grant(store)
        self._migrated_row(store)
        self._store_exists(monkeypatch, tmp_path)
        self._retrieval(monkeypatch, SecretStoreError("db is corrupt"))
        self._spy(monkeypatch, controller)

        reports = await controller.fetch_usage(["alibaba-token-plan"])

        assert [r.notes for r in reports] == [] or all(r.notes is None for r in reports)

    @pytest.mark.asyncio
    async def test_a_note_on_a_successful_api_key_report_still_reschedules(
        self, controller, store, monkeypatch, tmp_path
    ) -> None:
        """The note must expire on the API-key route too — the third branch.

        The polarity defect this pins is the one the whole re-probe exists to
        prevent, re-armed on the one path no test covered. Both branches that
        return a note for a MISSING report schedule the 10 s re-probe; this
        third one attaches the note to a LIVE 200 and then handed it to
        ``_mark_account_success``, which sets ``next_probe_at_ms = None``. The
        note then outlived ``lop secret unlock`` for the full jittered TTL
        (measured: 358890 ms, ~6 minutes) unless the user pressed ``r``.
        ``_settle_live_report``'s guard does not cover this path.

        Reachable by any user with a valid ``api_key`` row plus a
        migrated-but-locked ticket — they read the remedy, run it, and the
        panel keeps telling them to run it. Every pre-existing ``report.notes``
        assertion sits on the dead-grant path, which is how it shipped.
        """
        from local_operator.providers.usage_cache import USAGE_FAILURE_BACKOFF_MS
        from local_operator.secrets.client import BrokerLocked

        store.upsert_credential(
            "alibaba-token-plan", {"key": "fake-inference-key", "type": "api_key"}
        )
        self._migrated_row(store)
        assert (
            controller._expected_oauth_identities("alibaba-token-plan") == []
        ), "no OAuth row: this is the API-key route, not the dead-grant one"
        self._store_exists(monkeypatch, tmp_path)
        self._retrieval(monkeypatch, BrokerLocked("locked"))
        # A LIVE report: the fetch succeeds on the api_key, and only the ticket
        # is locked. `report is not None` plus a note is the uncovered case.
        self._spy(
            monkeypatch, controller, report=UsageReport(provider="alibaba-token-plan", limits=[])
        )

        now_ms = int(time.time() * 1000)
        reports = await controller.fetch_usage(["alibaba-token-plan"])

        assert len(reports) == 1
        report = reports[0]
        assert report.notes is not None and "lop secret unlock" in report.notes
        # The assertion that discriminates. `is not None` alone passes against
        # the defect only if the success path left a stamp, and it leaves None;
        # the bound pins it to the SHORT re-probe rather than to any schedule,
        # so a value drawn from the ~5-minute TTL fails here.
        assert report.next_probe_at_ms is not None, "the note was left with no re-probe scheduled"
        assert (
            report.next_probe_at_ms <= now_ms + USAGE_FAILURE_BACKOFF_MS + 1_000
        ), "the note is scheduled on the full TTL, so it outlives `lop secret unlock`"

    @pytest.mark.asyncio
    async def test_exposed_file_modes_are_reported_not_swallowed(
        self, controller, store, monkeypatch, tmp_path
    ) -> None:
        """A world-readable store is the one state the user MUST be told about.

        ``InsecurePermissions``'s own docstring calls it "a condition to stop
        on, not one to quietly repair — the exposure already happened and the
        operator needs to know". Returning the silent ``None`` this clause used
        to give it does the repairing-by-hiding it forbids: the window simply
        vanishes and nothing anywhere says the ticket is readable by another
        account.

        The remedy was RUN, not assumed — ``lop secret status`` against a 0644
        throwaway store prints the offending path and its ``chmod 0600`` fix.
        """
        from local_operator.secrets.errors import InsecurePermissions

        email = self._dead_grant(store)
        self._migrated_row(store)
        self._store_exists(monkeypatch, tmp_path)
        self._retrieval(monkeypatch, InsecurePermissions("store.db has mode 0644"))
        self._spy(monkeypatch, controller)

        reports = await controller.fetch_usage(["alibaba-token-plan"])

        assert len(reports) == 1, "an exposed store must not make the block disappear"
        assert reports[0].identity == email
        assert reports[0].notes is not None
        assert "lop secret status" in reports[0].notes
        # Distinct from the other three: a user who reads `secret unlock` or
        # `qwencloud-ticket set` here runs a command that cannot fix the mode
        # bits and leaves the exposure in place.
        assert "unlock" not in reports[0].notes
        assert "qwencloud-ticket" not in reports[0].notes

    @pytest.mark.asyncio
    async def test_a_version_skewed_broker_is_reported_not_swallowed(
        self, controller, store, monkeypatch, tmp_path
    ) -> None:
        """A daemon left running across a runtime update names its own fix.

        ``BrokerIncompatible`` exists BECAUSE collapsing "live but unusable"
        into "unreachable" silently disarmed a safety property (its round-4 Q4
        note). Answering it with a silent ``None`` here is that same collapse
        one layer up — and it arms itself precisely at a runtime update, which
        ``AGENTS.md`` calls routine on this machine because ``lop-update`` runs
        under live sessions.
        """
        from local_operator.secrets.errors import BrokerIncompatible

        self._dead_grant(store)
        self._migrated_row(store)
        self._store_exists(monkeypatch, tmp_path)
        self._retrieval(monkeypatch, BrokerIncompatible("protocol 2", pid=123, protocol=2))
        self._spy(monkeypatch, controller)

        reports = await controller.fetch_usage(["alibaba-token-plan"])

        assert len(reports) == 1
        assert reports[0].notes is not None
        assert "lop secret broker restart" in reports[0].notes
        # The pid and protocol ride on the exception for the CLI's benefit;
        # a panel note is 40 cells and they would cost the remedy its room.
        assert "123" not in reports[0].notes

    @pytest.mark.asyncio
    async def test_an_unreachable_broker_stays_silent(
        self, controller, store, monkeypatch, tmp_path
    ) -> None:
        """``BrokerUnavailable`` is transient, so it gets no note — deliberately.

        The sibling of ``test_an_unreadable_store_stays_silent``, pinned
        separately because the reasoning differs: a corrupt record has no
        repair verb at all, while "nothing answered" is usually a race the next
        auto-refresh wins — the daemon starts lazily and exits on its own idle
        timer. A note here would ask the user to act on something that has
        already fixed itself.

        Pinned so that a later round cannot quietly give this class a note on
        the grounds that the other two got one.
        """
        from local_operator.secrets.errors import BrokerUnavailable

        self._dead_grant(store)
        self._migrated_row(store)
        self._store_exists(monkeypatch, tmp_path)
        self._retrieval(monkeypatch, BrokerUnavailable("socket absent"))
        self._spy(monkeypatch, controller)

        reports = await controller.fetch_usage(["alibaba-token-plan"])

        assert all(r.notes is None for r in reports)

    # -- the note fits the panel -------------------------------------------

    @pytest.mark.asyncio
    async def test_the_note_survives_truncation_at_narrow_widths(self) -> None:
        """C6. The REMEDY survives the panel's truncation across the WIDTH RANGE.

        Three things this deliberately does NOT do, each of which produces a
        green test over a note the user cannot act on:

        1. It does not assert ``len(note) <= N``. That pins the wrong
           quantity — it passes on a note that still loses its trailing
           command, which is the only part that carries the fix.
        2. It does not RE-DERIVE the width from the ``PANEL_*`` constants.
           That chain yields 49 cells at a 60-column terminal and the real
           panel yields 47: ``overlay.screen_size`` reports the app's CONTENT
           box (58 for a 60-column terminal), so a derived budget overflows by
           two. The derived version of this test went green against a note
           that visibly truncated in a rendered frame.
        3. It does not test ONE width. The previous version rendered only at
           60 and passed, while the shipped locked note lost its remedy at
           every width from ``PANEL_MIN_WIDTH`` (32) to 59 — most of the range
           a split pane actually gets. Rendering at the widest supported size
           is the same class of error as deriving the budget: both check the
           case that cannot fail.

        So it renders the real panel at each width down to the floor and reads
        the painted line back. ``_lowest`` is the narrowest terminal each note
        is claimed to survive, measured from these frames, and the assertion
        runs at every width at or above it.
        """
        from local_operator.providers.controller import (
            QWENCLOUD_TICKET_BROKER_NOTE,
            QWENCLOUD_TICKET_EXPOSED_NOTE,
            QWENCLOUD_TICKET_LOCKED_NOTE,
            QWENCLOUD_TICKET_ORPHAN_NOTE,
        )
        from local_operator.tui.app import OperatorApp
        from tests.unit.tui.test_app_pilot import FakeSession, _factory

        # (note, remedy, narrowest terminal it survives). The floor is 40 for
        # the locked note rather than `PANEL_MIN_WIDTH`: at 36 and below the
        # card's budget is 25 cells and `lop secret unlock` alone is 19, so no
        # phrasing that keeps the command runnable as printed fits beside a
        # state word. Notes are painted raw by the body builder — only the
        # per-account note routes through `_fit_status_note`'s shortening
        # ladder — so going lower needs that seam, which is `usage_panel.py`'s.
        cases = (
            (QWENCLOUD_TICKET_LOCKED_NOTE, "lop secret unlock", 40),
            (QWENCLOUD_TICKET_ORPHAN_NOTE, "lop qwencloud-ticket set", 50),
            (QWENCLOUD_TICKET_EXPOSED_NOTE, "lop secret status", 45),
            (QWENCLOUD_TICKET_BROKER_NOTE, "lop secret broker restart", 55),
        )

        for note, remedy, lowest in cases:
            # The note's own head, so the line is located by what is being
            # asserted rather than by a word ("ticket") that two of these four
            # notes do not contain — a filter that matches nothing makes the
            # `painted` assertion the only thing standing between a silently
            # skipped case and a green run.
            head = note.split(" ")[0]
            for columns in (60, 58, 55, 50, 45, 40, 36, 32):
                if columns < lowest:
                    continue
                app = OperatorApp(lambda: _factory(FakeSession()))
                async with app.run_test(size=(columns, 30)) as pilot:
                    await pilot.pause()
                    panel = app._usage_panel()
                    assert panel is not None
                    panel.start_fetch()
                    panel.show_reports(
                        [UsageReport(provider="alibaba-token-plan", limits=[], notes=note)]
                    )
                    await pilot.pause()
                    painted = [
                        line.plain
                        for line in panel._body().lines
                        if line.plain.strip().startswith(head)
                    ]

                assert painted, f"{note!r} never reached the panel at {columns} columns"
                assert (
                    remedy in painted[0]
                ), f"the remedy is truncated at {columns} columns: {painted[0]!r}"
                assert "\u2026" not in painted[0]

    # -- the guards --------------------------------------------------------

    @pytest.mark.asyncio
    async def test_no_secret_store_spawns_no_daemon(
        self, controller, store, monkeypatch, tmp_path
    ) -> None:
        """C1. The store-less host — the common case — starts no daemon.

        ``retrieve_secret`` against a base with no store SPAWNS A BROKER and
        leaves it behind before failing, so ``/usage`` would start one on every
        refresh for a user who has never run ``lop secret set``. Checked by
        ARTIFACT and by SPY: a test that only asserted the return value passes
        with the guard deleted, because the failure still returns None.
        """
        import local_operator.secrets.client as client_mod

        self._dead_grant(store)
        self._migrated_row(store)
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))

        spawned: list[Any] = []
        monkeypatch.setattr(client_mod, "_spawn_broker", lambda base: spawned.append(base))
        self._spy(monkeypatch, controller)

        await controller.fetch_usage(["alibaba-token-plan"])

        assert spawned == [], "a host with no secret store must never spawn a broker"
        assert not (tmp_path / "secrets").exists()
        assert not (tmp_path / "secrets" / "broker.sock").exists()
        assert not (tmp_path / "secrets" / "broker.lock").exists()

    @pytest.mark.asyncio
    async def test_no_other_provider_reaches_the_secret_store(
        self, controller, store, monkeypatch, tmp_path
    ) -> None:
        """C9. The storage-id guard, asserted by EXECUTION over the registry.

        Without it every provider's refresh would try to retrieve a QwenCloud
        credential — and the value is a full-account console ticket, so the
        blast radius of widening this guard is the whole account.
        """
        from local_operator.providers.registry import PROVIDER_REGISTRY

        self._migrated_row(store)
        self._store_exists(monkeypatch, tmp_path)
        retrievals = self._retrieval(monkeypatch, self.TICKET.encode())

        reached = []
        for definition in PROVIDER_REGISTRY:
            creds, _ = await controller._qwencloud_console_creds(definition.id)
            if creds is not None:
                reached.append(definition.id)
        reached.sort()

        assert reached == ["alibaba-token-plan", "alibaba-token-plan-oauth"]
        assert len(retrievals) == len(reached), "no other provider may query the namespace"

    @pytest.mark.asyncio
    async def test_the_ticket_is_spent_at_most_once_per_cycle(
        self, controller, store, monkeypatch, tmp_path
    ) -> None:
        """C10. One ticket, one retrieval, however many dead grants.

        The flag is set on the ATTEMPT, not on success, and this is the test
        that pins the difference. A locked store returns no creds, so a flag
        set only on the success path leaves every later identity re-running
        the retrieval — and on a broker-down store that is a measured 10 s of
        blocked event loop EACH, inside the ``asyncio.gather`` that paints the
        panel. Three dead grants froze the TUI for 30 s.
        """
        from local_operator.secrets.client import BrokerDenied

        for index in range(3):
            store.upsert_credential(
                "alibaba-token-plan",
                {
                    "type": "oauth",
                    "access": "fake-mgmt",
                    "email": f"fake{index}@example.test",
                    "expires": 1,
                    "account_id": f"fake-acct-{index}",
                },
            )
        self._migrated_row(store)
        self._store_exists(monkeypatch, tmp_path)
        retrievals = self._retrieval(
            monkeypatch, BrokerDenied("no lop session is registered with the broker")
        )
        self._spy(monkeypatch, controller)

        await controller.fetch_usage(["alibaba-token-plan"])

        assert len(retrievals) == 1, (
            "a failed retrieval is a property of the TICKET, not of the identity: "
            "re-attempting it per identity multiplies a 10 s broker stall"
        )

    @pytest.mark.asyncio
    async def test_a_slow_retrieval_does_not_block_the_event_loop(
        self, controller, store, monkeypatch, tmp_path
    ) -> None:
        """C12. The retrieval runs off the loop, so the TUI keeps painting.

        ``retrieve_secret`` is SYNCHRONOUS and can take a measured 10 s when
        the broker will not start (``ensure_broker`` polls to
        ``STARTUP_TIMEOUT_S`` twice). This runs inside the ``asyncio.gather``
        that paints the usage panel, on an auto-refresh the user never asked
        for, so on the loop it freezes the whole TUI for that time —
        ``client.py``'s own #401 note records this codebase already freezing
        the TUI with exactly that shape of blocking call.

        Pinned by OBSERVABLE, not by asserting ``asyncio.to_thread`` appears in
        the source: a concurrent task is started and must get its turn WHILE
        the retrieval is still in flight. A test that only checked the return
        value passes with the hop deleted, because the answer is the same
        either way — only the latency of everything else changes.

        The retrieval sleeps 200 ms rather than 10 s: the property is "the loop
        advanced at all during it", which any blocking interval demonstrates,
        and a real stall would make this suite unrunnable.
        """
        import asyncio as _asyncio

        self._dead_grant(store)
        self._migrated_row(store)
        self._store_exists(monkeypatch, tmp_path)

        from local_operator.secrets import access

        def _slow_retrieve(name, base=None):
            time.sleep(0.2)
            return self.TICKET.encode()

        monkeypatch.setattr(access, "retrieve_secret", _slow_retrieve)
        self._spy(
            monkeypatch, controller, report=UsageReport(provider="alibaba-token-plan", limits=[])
        )

        ticks = 0

        async def _heartbeat() -> None:
            nonlocal ticks
            while True:
                await _asyncio.sleep(0.01)
                ticks += 1

        beat = _asyncio.create_task(_heartbeat())
        try:
            await controller.fetch_usage(["alibaba-token-plan"])
        finally:
            beat.cancel()

        # On the loop the heartbeat cannot run at all while the retrieval
        # blocks; off it, ~20 ticks fit in the 200 ms. The bound is deliberately
        # far below that so ordinary scheduler jitter on a contended box cannot
        # fail it, while zero-or-one tick — the blocking signature — still does.
        assert ticks >= 5, (
            f"the event loop advanced only {ticks} times during a 200 ms retrieval: "
            "the blocking call is back on the loop"
        )

    # -- the note clears itself --------------------------------------------

    @pytest.mark.asyncio
    async def test_unlocking_clears_the_note_without_a_manual_refresh(
        self, controller, store, monkeypatch, tmp_path
    ) -> None:
        """The note must not outlive the remedy the user just ran.

        A locked report is non-empty, so it is CACHED. With no scheduled
        re-probe the payload expires on the full jittered 5-minute TTL, and
        ``lop secret unlock`` — run in another terminal, as the note instructs
        — left the panel insisting the store was locked until the TTL lapsed
        or the user pressed ``r``. That is the "permanent message the user
        cannot act their way out of" polarity defect ``_account_in_backoff``
        documents this codebase having already paid for once.

        Measured before the fix: the note survived a non-forced refresh.
        """
        import time as time_mod

        from local_operator.secrets.client import BrokerDenied

        self._dead_grant(store)
        self._migrated_row(store)
        self._store_exists(monkeypatch, tmp_path)

        locked = {"value": True}
        from local_operator.secrets import access

        def _retrieve(name, base=None):
            if locked["value"]:
                raise BrokerDenied("no lop session is registered with the broker")
            return self.TICKET.encode()

        monkeypatch.setattr(access, "retrieve_secret", _retrieve)
        self._spy(
            monkeypatch, controller, report=UsageReport(provider="alibaba-token-plan", limits=[])
        )

        locked_reports = await controller.fetch_usage(["alibaba-token-plan"])
        assert "lop secret unlock" in (locked_reports[0].notes or "")
        assert (
            locked_reports[0].next_probe_at_ms is not None
        ), "the note has to schedule its own re-probe or it cannot clear itself"

        # The user runs the remedy, then the panel auto-refreshes -- NOT `r`.
        locked["value"] = False
        real_time = time_mod.time
        monkeypatch.setattr(time_mod, "time", lambda: real_time() + 15)

        cleared = await controller.fetch_usage(["alibaba-token-plan"])

        assert [r.notes for r in cleared] == [
            None
        ], "the locked note outlived `lop secret unlock` on a non-forced refresh"

    # -- the import graph --------------------------------------------------

    @pytest.mark.parametrize(
        ("module", "forbidden"),
        [
            # `qwencloud_console` is stdlib-only: `providers.usage` is a leak
            # there too, because the fetcher must never reach a credential
            # store.
            (
                "local_operator.providers.qwencloud_console",
                ("local_operator.secrets", "local_operator.tui", "local_operator.providers.usage"),
            ),
            # `controller` legitimately imports `providers.usage` at module
            # scope — it constructs `UsageReport`, and has since before this
            # slice. Only the SECRET/TUI stack is forbidden here.
            (
                "local_operator.providers.controller",
                ("local_operator.secrets", "local_operator.tui"),
            ),
        ],
    )
    def test_the_secret_stack_stays_off_the_import_graph(
        self, module: str, forbidden: tuple[str, ...]
    ) -> None:
        """C11. Neither module may drag the secret stack in at import time.

        ``qwencloud_console`` is stdlib-only precisely so ``controller.py`` can
        import it at module scope without a cycle; ``access.py`` pulls in
        ``cryptography`` and the client pulls in ``socket``/``fcntl``, which is
        why THIS module's four secret-store imports sit inside
        ``_qwencloud_console_creds`` rather than at the top of the file.

        ``controller`` is parametrized in deliberately: a probe of
        ``qwencloud_console`` alone cannot see a module-scope import added
        here, so the brief's single-module version went green against the very
        regression it was written to catch. Verified by moving one import to
        module scope — that turns the ``controller`` case red and leaves the
        ``qwencloud_console`` case green.

        Run in a FRESH interpreter: a same-process check passes trivially
        because this file's other tests already loaded the modules.
        """
        import subprocess
        import sys

        probe = (
            f"import sys; import {module}; "
            f"print(sorted(m for m in sys.modules if m.startswith({forbidden!r})))"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True
        )

        assert (
            result.stdout.strip() == "[]"
        ), f"{module} leaks a forbidden module at import: {result.stdout.strip()}"


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

    # ``**_kwargs``: see LOGIN_DOUBLE_SIGNATURE_NOTE.
    async def fake_login(_callbacks, **_kwargs):
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

    # ``**_kwargs``: see LOGIN_DOUBLE_SIGNATURE_NOTE.
    async def fake_login(_callbacks, **_kwargs):
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

    # ``**_kwargs``: see LOGIN_DOUBLE_SIGNATURE_NOTE.
    async def fake_login(_callbacks, **_kwargs):
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

    # ``**_kwargs``: see LOGIN_DOUBLE_SIGNATURE_NOTE.
    async def fake_login(_callbacks, **_kwargs):
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


@pytest.mark.asyncio
async def test_usable_providers_suppresses_secondary_flavour_when_oauth_active(
    controller, store
) -> None:
    """When a base provider has an active OAuth credential, secondary alias flavours
    (like `radient-key` for `radient`) must be omitted from usable_providers so they
    do not pollute /model suggestions."""
    store.upsert_credential("radient", {"access": "token", "type": "oauth"})
    usable = controller.usable_providers()
    assert usable is not None
    assert "radient" in usable
    assert "radient-key" not in usable


def test_initial_catalogue_layers_cached_aggregators_without_network(
    controller, store, tmp_path, monkeypatch
) -> None:
    """The first frame paints shipped registry models layered with cached aggregators.

    Aggregators return {} from static_models(). When a listing was previously cached
    on disk, initial_catalogue() includes those models synchronously so openrouter
    and radient models appear on the first frame rather than popping in only after
    live_catalogue().
    """
    import json
    import time

    # Plant a cached listing on disk for openrouter in tmp_path
    cache_file = tmp_path / "openrouter.listing.json"
    cache_file.write_text(
        json.dumps(
            {
                "fetched_at": time.time(),
                "payload": {
                    "capture": 6,
                    "models": [
                        {
                            "id": "meta/llama-3.3-70b",
                            "name": "Meta Llama 3.3 70B",
                            "context_window": 131072,
                            "input_price": 0.12,
                            "output_price": 0.30,
                            "free": False,
                        }
                    ],
                },
            }
        )
    )

    initial = controller.initial_catalogue(cache_dir=tmp_path)
    by_sel = {entry.selector: entry for entry in initial}

    # OpenRouter model is present on the initial frame
    assert "openrouter/meta/llama-3.3-70b" in by_sel
    entry = by_sel["openrouter/meta/llama-3.3-70b"]
    assert entry.aggregated is True
    assert entry.input_price == 0.12
    assert entry.output_price == 0.30
    assert entry.context_window == 131072

    # Direct shipped models are still present
    assert "anthropic/claude-opus-5" in by_sel

    # static_catalogue remains strictly static (no aggregators)
    static = {entry.selector: entry for entry in controller.static_catalogue()}
    assert "openrouter/meta/llama-3.3-70b" not in static
    assert "anthropic/claude-opus-5" in static


@pytest.mark.asyncio
async def test_live_catalogue_fetches_providers_in_parallel(controller, store, monkeypatch) -> None:
    """live_catalogue runs provider available_models checks concurrently with gather."""
    import time

    concurrency = 0
    max_concurrency = 0

    def slow_available_models(provider_id, **kwargs):
        nonlocal concurrency, max_concurrency
        concurrency += 1
        if concurrency > max_concurrency:
            max_concurrency = concurrency
        time.sleep(0.01)
        concurrency -= 1
        return [], "static"

    monkeypatch.setattr(
        "local_operator.providers.controller.available_models", slow_available_models
    )

    entries, statuses = await controller.live_catalogue()
    # Concurrency should be greater than 1 since providers are gathered
    assert max_concurrency > 1, f"Expected concurrent calls, got max_concurrency={max_concurrency}"


def test_persisted_providers_excludes_an_env_only_provider(controller, monkeypatch) -> None:
    """The rung that separates this from ``usable_providers``, and why it exists.

    An env key is a working credential for a LOCAL turn — the stream-time cascade
    resolves it — which is why ``usable_providers`` counts it. It is not consent
    for a REMOTE picker: the mobile daemon is launched by a service manager whose
    environment the phone's user never chose and cannot see, so an inherited
    ``DEEPSEEK_API_KEY`` would silently add an account to a sheet reachable over
    a tunnel.
    """
    for name in _USAGE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds-test")

    usable = controller.usable_providers()
    persisted = controller.persisted_providers()
    assert usable is not None and persisted is not None
    assert "deepseek" in usable, "the local cascade runs on it"
    assert "deepseek" not in persisted, "nobody persisted it"


def test_persisted_providers_excludes_a_keyless_local_provider(controller, monkeypatch) -> None:
    """``allows_missing_api_key`` means usable with no credential at all, which is
    the whole point of running a local Ollama — and exactly why it must not be
    advertised remotely: keyless readiness is not evidence a server is running,
    and the phone cannot reach the owner's loopback anyway."""
    for name in _USAGE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    usable = controller.usable_providers()
    persisted = controller.persisted_providers()
    assert usable is not None and persisted is not None
    assert "ollama" in usable
    assert "ollama" not in persisted


def test_persisted_providers_includes_a_stored_login(controller, store, monkeypatch) -> None:
    for name in _USAGE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    store.upsert_credential("anthropic", {"access": "tok", "refresh": "ref", "type": "oauth"})
    persisted = controller.persisted_providers()
    assert persisted is not None
    assert "anthropic" in persisted


@pytest.mark.parametrize(
    ("key_name", "provider_id"),
    [
        # The plain-string ``env_keys`` form.
        ("OPENROUTER_API_KEY", "openrouter"),
        ("OPENAI_API_KEY", "openai"),
        # The CALLABLE form. Parametrizing over both forms is the point of this
        # case rather than tidiness: a reader built on ``env_key_name`` alone
        # resolves every string provider and silently drops the callable one, so
        # a single string-keyed case passes over a reader that loses the only
        # provider using the other form.
        ("ANTHROPIC_API_KEY", "anthropic"),
    ],
)
def test_persisted_providers_includes_a_legacy_credential_manager_key(
    controller, monkeypatch, tmp_path, key_name, provider_id
) -> None:
    """``lop credential update`` writes the legacy file, ``/login`` writes auth.db.

    A reader consulting only one of the two hides every provider configured
    through the other, and both are sanctioned flows.
    """
    from local_operator.credentials import CredentialManager

    for name in _USAGE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    manager = CredentialManager(tmp_path)
    manager.set_credential(key_name, "sk-persisted")
    controller.credential_manager = manager

    persisted = controller.persisted_providers()
    assert persisted is not None
    assert provider_id in persisted


def test_persisted_providers_ignores_an_empty_legacy_value(
    controller, monkeypatch, tmp_path
) -> None:
    """A key present but blank is not a credential; the legacy file keeps such
    rows, and treating one as a login sends the picker fetching anonymously."""
    from local_operator.credentials import CredentialManager

    for name in _USAGE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    manager = CredentialManager(tmp_path)
    manager.set_credential("OPENROUTER_API_KEY", "")
    controller.credential_manager = manager

    persisted = controller.persisted_providers()
    assert persisted is not None
    assert "openrouter" not in persisted


def test_persisted_providers_suppresses_a_secondary_flavour_when_oauth_is_active(
    controller, store
) -> None:
    """Same suppression ``usable_providers`` performs: one account, offered once."""
    store.upsert_credential("radient", {"access": "token", "type": "oauth"})
    persisted = controller.persisted_providers()
    assert persisted is not None
    assert "radient" in persisted
    assert "radient-key" not in persisted


def test_persisted_providers_answers_none_on_an_unreadable_store(controller, store) -> None:
    """ "I could not look" is not "you have nothing" — see ``picker_rows``, which
    shows every model on ``None`` rather than claiming an empty inventory."""

    def boom(provider=None):
        raise sqlite3.OperationalError("database is locked")

    store.list_credentials = boom  # type: ignore[assignment]
    assert controller.persisted_providers() is None


def test_persisted_providers_reraises_cross_thread_misuse(controller, store) -> None:
    """A connection used from the wrong thread is a BUG in the caller, not an
    environment fact. It must not dress itself as the unreadable-store
    degradation, which would silently label every model connected (D18)."""

    def boom(provider=None):
        raise sqlite3.ProgrammingError("SQLite objects created in a thread...")

    store.list_credentials = boom  # type: ignore[assignment]
    with pytest.raises(sqlite3.ProgrammingError):
        controller.persisted_providers()


@pytest.mark.asyncio
async def test_live_catalogue_without_a_providers_argument_is_unchanged(
    controller, store, monkeypatch
) -> None:
    """The default has to stay byte-identical: every existing caller passes no
    ``providers``, and the narrowing was added for one new caller only."""
    store.upsert_credential("anthropic", {"key": "sk-ant", "type": "api_key"})
    calls = _spy_available_models(monkeypatch, live={"anthropic": ["claude-opus-5"]})

    from local_operator.providers.registry import PROVIDER_REGISTRY

    entries, statuses = await controller.live_catalogue()
    fetched = {provider for provider, _ttl in calls}
    assert len(fetched) > 1, "the whole registry is still enumerated"
    assert "anthropic" in {entry.provider for entry in entries}
    # Every provider reports a status, including the local ones that resolve a
    # base URL and return before any listing call — so ``statuses`` is the
    # CHAT registry, not the subset that made a request. A decision-only
    # provider (``typesafe``) is not in it, and that is the flag working: it is
    # filtered before the enumeration, so it has neither rows nor a status line
    # in a catalogue it can never appear in.
    assert set(statuses) == {
        definition.id for definition in PROVIDER_REGISTRY if not definition.decision_only
    }


@pytest.mark.asyncio
async def test_live_catalogue_narrows_to_the_named_providers(
    controller, store, monkeypatch
) -> None:
    """A caller that has already decided which accounts it may speak for says so,
    and nothing outside that set is contacted or reported."""
    store.upsert_credential("anthropic", {"key": "sk-ant", "type": "api_key"})
    store.upsert_credential("openrouter", {"key": "sk-or", "type": "api_key"})
    calls = _spy_available_models(
        monkeypatch, live={"anthropic": ["claude-opus-5"], "openrouter": ["vendor/model"]}
    )

    entries, statuses = await controller.live_catalogue(providers={"anthropic"})
    assert {provider for provider, _ttl in calls} == {"anthropic"}
    assert set(statuses) == {"anthropic"}
    assert {entry.provider for entry in entries} == {"anthropic"}


@pytest.mark.asyncio
async def test_an_admitted_empty_set_fetches_nothing(controller, store, monkeypatch) -> None:
    """``set()`` is honoured literally. An owner logged in to nothing is a real
    state with a real answer; falling back to the whole registry would turn the
    strictest case into the loosest one."""
    store.upsert_credential("anthropic", {"key": "sk-ant", "type": "api_key"})
    calls = _spy_available_models(monkeypatch, live={"anthropic": ["claude-opus-5"]})

    entries, statuses = await controller.live_catalogue(providers=set())
    assert calls == []
    assert entries == []
    assert statuses == {}


@pytest.mark.asyncio
async def test_a_narrowed_provider_lists_with_its_credential_not_anonymously(
    controller, store, monkeypatch
) -> None:
    """``connected`` follows the CALLER's determination for an admitted id.

    ``usable_providers`` has no legacy ``credentials.env`` rung, so a provider
    configured with ``lop credential update`` came back unconnected, listed
    anonymously, and the phone's sheet showed it empty — with a credential on
    disk the whole time. Narrowing must not be the reason rows disappear.
    """
    from local_operator.credentials import CredentialManager

    for name in _USAGE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    calls = _spy_available_models(monkeypatch, live={"openrouter": ["vendor/model"]})

    entries, _statuses = await controller.live_catalogue(providers={"openrouter"})
    assert [provider for provider, _ttl in calls] == ["openrouter"]
    assert [entry.selector for entry in entries] == ["openrouter/vendor/model"]
    assert all(entry.connected for entry in entries)
    assert isinstance(CredentialManager, type)
