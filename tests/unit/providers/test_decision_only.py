"""``decision_only`` providers: storable, but never offerable as a chat model.

Why this file exists: TypeSafe's Jev (the ``typesafe`` row) is a REAL
provider — it has a shipped login, a base URL and an env var — and its wire
rejects ``chat/completions`` on every host we reach it through
(``docs/design/classification-layer.md`` §9). Every surface that decides what a
session may RUN ON therefore has to skip it, and each of those surfaces is a
separate place the flag can be forgotten:

1. **Discovery and listing** — ``providers.controller``'s catalogue builders,
   which every model sheet (TUI picker, phone, desktop) reads.
2. **The ``/model`` catalogue ranking** — ``model.ranking.rank_rows``, the last
   gate before a row becomes a choice.
3. **Session model resolution** — ``session_factory.resolve_hosting_model_with_source``,
   which decides what a booting session runs on.
4. **The failover chain** — ``providers.failover.expand_fallback_targets``, the
   one place a configured chain entry becomes a route.

A provider that slipped through any one of them produces a session that cannot
answer a single turn, and the user has no way to read that as "that model was
never offerable" — it arrives as an ordinary provider error. So each surface
gets its own test here, and the four are asserted to agree.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

from local_operator.config import ConfigManager
from local_operator.model.ranking import ModelRow, rank_rows
from local_operator.providers import registry
from local_operator.providers.controller import ProviderController
from local_operator.providers.failover import (
    expand_fallback_candidates,
    expand_fallback_targets,
)
from local_operator.providers.usage_cache import UsageCacheStore
from local_operator.session_factory import (
    HostingNotChatError,
    HostingNotConfiguredError,
    HostingUnknownError,
    resolve_hosting_model_with_source,
)


class _Row:
    """One stored credential row, as the controller's store returns them."""

    def __init__(self, provider: str, credential_type: str = "api_key") -> None:
        self.provider = provider
        self.credential_type = credential_type


class _FakeAuthStore:
    """The credential-store slice ``ProviderController`` reads a catalogue with."""

    def __init__(self, rows: list[_Row] | None = None) -> None:
        self.rows = list(rows or [])

    def list_credentials(self, provider: str | None = None) -> list[_Row]:
        if provider is None:
            return list(self.rows)
        return [row for row in self.rows if row.provider == provider]

    async def get_api_key(self, provider: str) -> str | None:
        return None

    def active_local_credential(self, provider: str, endpoint: str) -> Any:
        return None


def _controller(tmp_path: Path) -> ProviderController:
    # ``type: ignore[arg-type]``: the double implements the READ slice a catalogue
    # needs (``list_credentials``); the write/oauth halves of ``ControllerAuthStore``
    # are unreachable from these paths, and implementing them would be four
    # methods of dead code pretending to be coverage.
    return ProviderController(
        _FakeAuthStore(),  # type: ignore[arg-type]
        login_callbacks=None,
        usage_cache=UsageCacheStore(tmp_path / "usage_cache.db"),
    )


# ---------------------------------------------------------------------------
# The registry row itself
# ---------------------------------------------------------------------------


def test_the_typesafe_row_is_login_capable_and_decision_only() -> None:
    """The key has to be storable; the provider must be un-selectable."""
    definition = registry.get_provider_definition("typesafe")

    assert definition is not None
    assert definition.name == "TypeSafe (Jev)"
    assert definition.base_url == "https://api.typesafe.ai/v1"
    assert definition.decision_only is True
    # The paste-a-key path (§9): `create_api_key_login` tags itself, so every
    # host that attaches a prompt can complete this login.
    assert definition.login_kind == "api_key"
    assert definition.paste_prompt_required is True
    # And it is offered by the login surfaces, which is the half that has to
    # keep working.
    assert "typesafe" in {provider.id for provider in registry.list_login_providers()}
    assert "jev" in definition.search_aliases


def test_the_env_keys_are_read_primary_first() -> None:
    """``TYPESAFE_API_KEY`` then ``JEV_API_KEY`` — the tuple form's contract."""
    assert registry.env_key_names("typesafe") == ("TYPESAFE_API_KEY", "JEV_API_KEY")
    # The display name is the FIRST one; both are credential-file rungs, because
    # an operator who set the older spelling is configured either way.
    assert registry.env_key_name("typesafe") == "TYPESAFE_API_KEY"
    assert registry.credential_file_names("typesafe") == ["TYPESAFE_API_KEY", "JEV_API_KEY"]


def test_resolve_env_key_prefers_the_primary_and_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    assert registry.resolve_env_key("typesafe") is None

    monkeypatch.setenv("JEV_API_KEY", "jev-only")
    assert registry.resolve_env_key("typesafe") == "jev-only"

    monkeypatch.setenv("TYPESAFE_API_KEY", "typesafe-key")
    assert registry.resolve_env_key("typesafe") == "typesafe-key"


def test_only_the_typesafe_row_is_decision_only() -> None:
    """The flag is not a broad brush: every other row stays selectable."""
    flagged = {row.id for row in registry.PROVIDER_REGISTRY if row.decision_only}

    assert flagged == {"typesafe"}
    assert registry.is_decision_only("typesafe") is True
    assert registry.is_decision_only("openai") is False
    assert registry.is_decision_only(None) is False
    assert registry.is_decision_only("not-a-provider") is False


# ---------------------------------------------------------------------------
# 1. Discovery and listing
# ---------------------------------------------------------------------------


def test_the_static_catalogue_offers_no_typesafe_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even when the shipped registry WOULD hand it a model, the catalogue drops it.

    ``static_models`` is patched to answer for every provider id, which is the
    only shape of this test that can fail: a provider with no static models is
    absent from the catalogue whether or not the filter exists.
    """
    monkeypatch.setattr(
        "local_operator.providers.controller.static_models",
        lambda provider_id: {"jev-1.13": _Info()},
    )
    controller = _controller(tmp_path)

    entries = controller.static_catalogue()

    providers = {entry.provider for entry in entries}
    assert "typesafe" not in providers
    assert "openai" in providers, "the filter must not empty the catalogue"


def test_the_first_frame_catalogue_offers_no_typesafe_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "local_operator.providers.controller.static_models",
        lambda provider_id: {"jev-1.13": _Info()},
    )
    controller = _controller(tmp_path)

    entries = controller.initial_catalogue(cache_dir=tmp_path)

    assert "typesafe" not in {entry.provider for entry in entries}


@pytest.mark.asyncio
async def test_the_live_catalogue_never_asks_for_typesafe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No listing call at all: the decision endpoint is never asked for chat models.

    The recorder is the point — it proves the provider is filtered BEFORE the
    fetch, so the run does not even spend a round trip on a list it must not
    show.
    """
    asked: list[str] = []

    def recording_available_models(provider_id: str, **kwargs: Any) -> tuple[list[Any], str]:
        asked.append(provider_id)
        return [], "static"

    monkeypatch.setattr(
        "local_operator.providers.controller.available_models", recording_available_models
    )
    controller = _controller(tmp_path)

    entries, _statuses = await controller.live_catalogue()

    assert "typesafe" not in asked
    assert "openai" in asked
    assert "typesafe" not in {entry.provider for entry in entries}


@pytest.mark.asyncio
async def test_an_explicit_providers_set_still_gets_no_typesafe_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller may narrow the catalogue; it may not widen it to a non-chat model."""
    asked: list[str] = []

    def recording_available_models(provider_id: str, **kwargs: Any) -> tuple[list[Any], str]:
        asked.append(provider_id)
        return [], "static"

    monkeypatch.setattr(
        "local_operator.providers.controller.available_models", recording_available_models
    )
    controller = _controller(tmp_path)

    entries, _statuses = await controller.live_catalogue(providers={"typesafe", "openai"})

    assert "typesafe" not in asked
    assert asked == ["openai"]
    assert "typesafe" not in {entry.provider for entry in entries}


class _Info:
    """Minimal stand-in for the model registry's per-model metadata."""

    name = "Jev"
    context_window = 64000
    default_context_window = None
    max_context_window = None
    input_price = 0.0
    output_price = 0.0
    time_of_use = False


# ---------------------------------------------------------------------------
# 2. The ``/model`` catalogue ranking
# ---------------------------------------------------------------------------


def test_rank_rows_drops_a_decision_only_provider() -> None:
    """Ranking is the last gate before a row becomes a choice: filter here too."""
    rows = [
        ModelRow(provider="typesafe", model_id="jev-1.13", label="TypeSafe (Jev)/jev-1.13"),
        ModelRow(provider="openai", model_id="gpt-4o", label="OpenAI/gpt-4o"),
    ]

    # Empty query (the picker's resting frame): everything that survives, i.e.
    # the chat model and nothing else.
    assert [row.provider for row in rank_rows(list(rows), "")] == ["openai"]
    # A query that NAMES the provider: no match at all, rather than the one model
    # that cannot answer. (``openai/gpt-4o`` matches neither `typesafe` nor
    # `jev`, so an empty list here is the correct answer, not a lost row.)
    for query in ("typesafe", "jev"):
        assert rank_rows(list(rows), query) == [], query


# ---------------------------------------------------------------------------
# 3. Session model resolution
# ---------------------------------------------------------------------------


def _args(**kwargs: Any) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


def test_a_decision_only_hosting_is_refused_at_boot(tmp_path: Path) -> None:
    """The boot resolver refuses it, recoverably, naming the value and the way out."""
    manager = ConfigManager(tmp_path)

    with pytest.raises(HostingNotChatError) as caught:
        resolve_hosting_model_with_source(
            None, _args(hosting="typesafe", model="jev-1.13"), manager
        )

    error = caught.value
    assert error.hosting == "typesafe"
    assert error.source == "flag"
    assert "decision-model calls, not chat completions" in str(error)
    assert "--hosting" in str(error), "the remedy must be the one that can work"
    # Recoverable, like every other hosting fault: the TUI opens in the setup
    # state where `/model` can fix it, rather than a dead "session failed to
    # start" with no way to reach the command that repairs it.
    assert isinstance(error, HostingNotConfiguredError)
    # ...but NOT the unknown-provider class: this provider is known and has a
    # login, so "not a known provider" and its `/login` remedy would be false.
    assert not isinstance(error, HostingUnknownError)


def test_a_decision_only_hosting_from_config_names_the_config_remedy(tmp_path: Path) -> None:
    manager = ConfigManager(tmp_path)
    manager.set_config_value("hosting", "typesafe")

    with pytest.raises(HostingNotChatError) as caught:
        resolve_hosting_model_with_source(None, _args(), manager)

    assert caught.value.source == "config"
    assert "/model" in str(caught.value)


def test_a_chat_hosting_still_resolves(tmp_path: Path) -> None:
    """The guard is narrow: the ordinary path is untouched."""
    manager = ConfigManager(tmp_path)

    hosting, model, source = resolve_hosting_model_with_source(
        None, _args(hosting="openai", model="gpt-4o"), manager
    )

    assert (hosting, model, source) == ("openai", "gpt-4o", "flag")


# ---------------------------------------------------------------------------
# 4. The failover chain
# ---------------------------------------------------------------------------


def test_expand_fallback_targets_drops_a_decision_only_entry() -> None:
    """A chain that names Jev must not route a failing turn onto it."""
    chain = ["typesafe/jev-1.13", "anthropic/claude-sonnet-4"]

    targets = expand_fallback_targets("openai/gpt-4o", chain)

    assert [target.selector for target in targets] == ["anthropic/claude-sonnet-4"]


def test_a_wildcard_decision_only_entry_is_dropped_too() -> None:
    """The provider prefix is read before the ``/*`` expansion inherits the id."""
    chain = ["typesafe/*", "openai/*"]

    targets = expand_fallback_targets("deepseek/deepseek-chat", chain)

    assert [target.selector for target in targets] == ["openai/deepseek-chat"]


def test_an_all_decision_only_chain_has_no_fallback() -> None:
    """The honest answer: no route, rather than a route that is guaranteed to fail."""
    assert expand_fallback_targets("openai/gpt-4o", ["typesafe/jev-1.13"]) == []
    assert expand_fallback_candidates("openai/gpt-4o", ["typesafe/jev-1.13"]) == []


# ---------------------------------------------------------------------------
# The surfaces are not the only thing that must agree: `--hosting` too
# ---------------------------------------------------------------------------


def test_typesafe_is_not_an_offered_chat_hosting_on_the_cli() -> None:
    """``--hosting`` selects a CHAT hosting, and Jev is not one.

    The flag's choices are a hardcoded list of chat hostings (the registry is
    not consulted), so this is a decision not to add one — pinned here so a
    later "the registry has a login for it, let's add it" edit fails here
    instead of at the user's first turn.
    """
    from local_operator.cli import build_cli_parser

    parser = build_cli_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--hosting", "typesafe"])
    assert parser.parse_args(["--hosting", "openai"]).hosting == "openai"
