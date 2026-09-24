"""`ProviderController.catalogue_failures`: which listings a user can act on.

Reproduction. The desktop picker's notice counts `Object.keys(errors)` from
`GET /v1/desktop/models`, and that map was built from every key of
`live_catalogue`'s status map — so the operator's own machine reported SIX
providers as having "not answered": `ollama`, `lmstudio`, `llamacpp`, `vllm`,
`openai-compatible` (the app's own preset local ports; `config.yml` has no
`providers:` section at all) and `test` (the app's own mock host). A FRESH
INSTALL reported SEVEN (the same six plus the keyless aggregator `radient`, whose
public listing answers 401 and which has no bundled rows). No cloud provider was
failing.

The rule under test is therefore not "is it local" but "could the app have listed
this provider on the user's behalf, and did not": a listing transport at all, an
ENGAGED provider (config for a `local_setup` preset, a credential for everyone
else), and a status that means the listing failed. The cases below are the two
axes independently, because each one alone re-breaks one of the two reports.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from local_operator.config import ConfigManager
from local_operator.providers.auth_store import AuthStore
from local_operator.providers.controller import (
    CATALOGUE_FAILURE_REASON,
    CatalogueEntry,
    ProviderController,
)
from local_operator.providers.registry import store_provider_key

#: Any environment key that would make one of these providers "engaged" through
#: ``resolve_env_key`` and so change what a case asserts. Deleted per test rather
#: than per case so a new case cannot forget one.
_CREDENTIAL_ENV = (
    "OPENROUTER_API_KEY",
    "RADIENT_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "DEEPSEEK_API_KEY",
    "OLLAMA_API_KEY",
    "LMSTUDIO_API_KEY",
    "LLAMACPP_API_KEY",
    "VLLM_API_KEY",
)


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated config dir AND home: `config_dir()` follows the first, the
    catalogue cache the second, and the rule reads the first."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in _CREDENTIAL_ENV:
        monkeypatch.delenv(name, raising=False)
    return tmp_path


@pytest.fixture
def store(tmp_path: Path):
    instance = AuthStore(tmp_path / "auth.db")
    try:
        yield instance
    finally:
        instance.close()


@pytest.fixture
def controller(store: AuthStore):
    instance = ProviderController(store)
    try:
        yield instance
    finally:
        instance.close()


def _configure_local(root: Path, provider: str, base_url: str) -> None:
    """Point a local preset at an endpoint, the way `_configure_local` writes it."""
    ConfigManager(root).update_config({"providers": {provider: {"base_url": base_url}}})


def _entry(provider: str) -> CatalogueEntry:
    return CatalogueEntry(
        provider=provider,
        model_id="a-model",
        label="a-model",
        context_window=1000,
        input_price=0.0,
        output_price=0.0,
        connected=True,
    )


def test_a_store_first_credential_engages_the_provider(
    controller: ProviderController, isolated: Path
) -> None:
    """A key saved through Settings engages a provider auth.db knows nothing about.

    `usable_providers()` reads auth rows and the environment. The provider-class
    rows of the encrypted secret store — what `PATCH /v1/credentials`, `lop
    credential update` and the desktop Settings / onboarding flows write — are
    invisible to it, which is why `persisted_providers()` exists and why the
    mobile daemon passes `providers=`. Engaging on `usable_providers()` alone made
    this rule go SILENT for a provider the user HAD engaged, on exactly the ids
    `live_catalogue` fetches anonymously (a keyless aggregator's listing 401s, so
    the failure is real).
    """
    store_provider_key("OPENROUTER_API_KEY", "sk-or-from-settings", base=isolated)
    # The gap this test is about, asserted rather than assumed.
    assert "openrouter" not in (controller.usable_providers() or set())
    assert "openrouter" in (controller.persisted_providers() or set())
    assert controller.catalogue_failures([], {"openrouter": "stale"}) == {
        "openrouter": CATALOGUE_FAILURE_REASON
    }


def test_the_mock_host_is_never_named(controller: ProviderController) -> None:
    """`test` has no listing transport at all, so it cannot have failed to list.

    `NO_LISTING_PROVIDERS` is the registry's own statement about these ids, and a
    request to the mock wire has nowhere to go — the app must not report its own
    fixture as a broken provider.
    """
    assert controller.catalogue_failures([], {"test": "static"}) == {}


def test_an_unconfigured_local_preset_is_not_a_failure(controller: ProviderController) -> None:
    """The app's own default port with nobody home is not a provider that went away.

    This is the operator's machine: `ollama`'s `static`/0 rows means no Ollama app
    is running, which is the normal state of every install that never used it.
    """
    assert controller.catalogue_failures([], {"ollama": "static"}) == {}
    assert controller.catalogue_failures([], {"vllm": "stale"}) == {}


def test_a_configured_local_provider_that_failed_is_named(
    controller: ProviderController, isolated: Path
) -> None:
    """Once the user points a preset somewhere, a failure IS theirs to see."""
    _configure_local(isolated, "vllm", "http://127.0.0.1:8000/v1")
    assert controller.catalogue_failures([], {"vllm": "stale"}) == {
        "vllm": CATALOGUE_FAILURE_REASON
    }
    # ...and the configured preset's `static`-with-no-rows shape too.
    assert controller.catalogue_failures([], {"vllm": "static"}) == {
        "vllm": CATALOGUE_FAILURE_REASON
    }


def test_a_configured_local_provider_that_answered_is_never_named(
    controller: ProviderController, isolated: Path
) -> None:
    _configure_local(isolated, "vllm", "http://127.0.0.1:8000/v1")
    assert controller.catalogue_failures([_entry("vllm")], {"vllm": "ok"}) == {}


def test_a_cloud_provider_with_a_credential_that_failed_is_named(
    controller: ProviderController, store: AuthStore
) -> None:
    store.upsert_credential("openrouter", {"key": "sk-or-1", "source": "login"})
    assert controller.catalogue_failures([], {"openrouter": "stale"}) == {
        "openrouter": CATALOGUE_FAILURE_REASON
    }


def test_a_cloud_provider_without_a_credential_is_not_named(
    controller: ProviderController,
) -> None:
    """Nothing to list with is not a listing that failed.

    A provider the user never connected is reported by the picker's own row state
    (its registry rows, and the TUI footer's access note), not as an outage.
    """
    assert controller.catalogue_failures([], {"openrouter": "stale"}) == {}


def test_a_keyless_aggregator_is_named_only_once_it_has_a_credential(
    controller: ProviderController, store: AuthStore
) -> None:
    """MEASURED: this is the seventh name a fresh install used to show.

    `radient` bundles no rows and its public listing answers 401, so `static`/0
    rows named it on an empty credential store. With a credential the same status
    is a real failure the user can act on, so the engagement axis — not the
    local/cloud split — is what has to excuse it.
    """
    assert controller.catalogue_failures([], {"radient": "static"}) == {}
    store.upsert_credential("radient", {"key": "rc-1", "source": "login"})
    assert controller.catalogue_failures([], {"radient": "static"}) == {
        "radient": CATALOGUE_FAILURE_REASON
    }


def test_an_unreadable_store_engages_cloud_but_not_an_unconfigured_local(
    controller: ProviderController, store: AuthStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`usable_providers() is None` must narrow nothing it cannot know.

    Unknown is not unengaged: the code's existing degradation reads an unreadable
    store as "everything connected", and this method follows it. The local axis is
    read from CONFIG, so it is still known — and still excuses a preset nobody
    configured.
    """

    def _unreadable(**_kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(store, "list_credentials", _unreadable)
    assert controller.catalogue_failures([], {"openrouter": "stale"}) == {
        "openrouter": CATALOGUE_FAILURE_REASON
    }
    assert controller.catalogue_failures([], {"ollama": "static"}) == {}


def test_only_a_failed_listing_is_named(controller: ProviderController, store: AuthStore) -> None:
    """`ok`/`cached`/`unauthenticated` are answers, not failures (D1).

    `cached` served a stored document on purpose and `unauthenticated` means the
    provider was never asked — naming either told the user their working
    providers had "not answered".
    """
    store.upsert_credential("openrouter", {"key": "sk-or-1", "source": "login"})
    for status in ("ok", "cached", "unauthenticated"):
        assert controller.catalogue_failures([], {"openrouter": status}) == {}


def test_an_empty_answer_is_a_failure(controller: ProviderController, store: AuthStore) -> None:
    store.upsert_credential("openrouter", {"key": "sk-or-1", "source": "login"})
    assert controller.catalogue_failures([], {"openrouter": "empty"}) == {
        "openrouter": CATALOGUE_FAILURE_REASON
    }


def test_rows_excuse_only_a_static_provider(
    controller: ProviderController, store: AuthStore
) -> None:
    """The row count resolves `static`'s ambiguity and nothing else (R1-4/Q1).

    `static` means "the registry is all there is", which for a provider that
    bundles rows is a fine answer; a failed fetch with nothing cached reports the
    same word and is the operator's original "refresh failed" state. A `stale`
    document is stale whatever rows it also contributed.
    """
    store.upsert_credential("openrouter", {"key": "sk-or-1", "source": "login"})
    assert controller.catalogue_failures([_entry("openrouter")], {"openrouter": "static"}) == {}
    assert controller.catalogue_failures([_entry("openrouter")], {"openrouter": "stale"}) == {
        "openrouter": CATALOGUE_FAILURE_REASON
    }


def test_the_reported_value_is_the_one_generic_sentence(
    controller: ProviderController, store: AuthStore
) -> None:
    """Never the provider's own error: the surface reads only the keys, and a
    provider exception can carry a response body or a credential URL."""
    store.upsert_credential("openrouter", {"key": "sk-or-1", "source": "login"})
    store.upsert_credential("deepseek", {"key": "sk-ds-1", "source": "login"})
    failures = controller.catalogue_failures([], {"openrouter": "stale", "deepseek": "empty"})
    assert set(failures) == {"openrouter", "deepseek"}
    assert set(failures.values()) == {CATALOGUE_FAILURE_REASON}


def test_the_statuses_map_is_read_only(controller: ProviderController) -> None:
    """The caller's dict is the live catalogue's; a rule may not consume it."""
    statuses = {"ollama": "static", "test": "static"}
    controller.catalogue_failures([], statuses)
    assert statuses == {"ollama": "static", "test": "static"}
