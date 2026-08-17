"""Unit tests for ProviderController — the TUI's provider/model/usage facade.

Credential/login behavior is exercised against a fake auth store so no real
SQLite or network is needed; usage dispatch is tested against a canned
httpx transport.
"""

from __future__ import annotations

import types
from typing import Any

import httpx
import pytest

from local_operator.providers.controller import ProviderController
from local_operator.providers.usage import UsageReport


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
        return list(self.oauth_accounts.get(provider, []))

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
        assert [r.identity for r in reports] == ["fine@example.com"]

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


def test_usage_enabled_provider_ids(controller) -> None:
    ids = controller.usage_enabled_providers()
    assert "openrouter" in ids
    assert "deepseek" in ids
    assert "zai" not in ids, "unreachable: no ProviderDefinition exists for it"
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
    # `xai-oauth` has no env var of its own and nothing stored under `xai`, so it
    # never even reaches the finer check — it is unusable outright.
    assert not controller.is_usable("xai-oauth")
    assert not controller.can_report_usage("xai-oauth")
    assert controller.usage_reportable_providers() == []


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
