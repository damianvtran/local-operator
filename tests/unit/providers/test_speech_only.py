"""``speech_only`` providers: storable, but never offerable as a chat model.

Why this file exists: ElevenLabs (the ``elevenlabs`` row) is a REAL provider —
shipped login, base URL, env var — and its wire serves speech-to-text and no
chat route at all, while the mobile voice path REQUIRES the key to be storable.
That combination is the same shape ``decision_only`` exists for, one flag over
(``tests/unit/providers/test_decision_only.py``): every surface that decides
what a session may RUN ON has to skip it, and each surface is a separate place
the flag can be forgotten —

1. **Discovery and listing** — ``providers.controller``'s catalogue builders.
2. **The ``/model`` catalogue ranking** — ``model.ranking.rank_rows``.
3. **Session model resolution** — ``session_factory``'s preflight and the
   stored-selection validator both owner-side readers share.
4. **The failover chain** — ``providers.failover.expand_fallback_targets``.

— plus the two doors a NON-picker surface reaches (the live-switch spec
builder, the peer/desktop model validator) and the login planner, which must
store the key WITHOUT writing a chat hosting. A provider that slipped through
any one of them produces a session that cannot answer a turn, and the user has
no way to read that as "that model was never offerable" — it arrives as an
ordinary provider error.
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
    return ProviderController(
        _FakeAuthStore(),  # type: ignore[arg-type]
        login_callbacks=None,
        usage_cache=UsageCacheStore(tmp_path / "usage_cache.db"),
    )


class _Info:
    """Minimal stand-in for the model registry's per-model metadata."""

    name = "Scribe v2"
    context_window = 0
    default_context_window = None
    max_context_window = None
    input_price = 0.0
    output_price = 0.0
    time_of_use = False


# ---------------------------------------------------------------------------
# The registry row itself
# ---------------------------------------------------------------------------


def test_the_elevenlabs_row_is_login_capable_and_speech_only() -> None:
    """The key has to be storable; the provider must be un-selectable as chat."""
    definition = registry.get_provider_definition("elevenlabs")

    assert definition is not None
    assert definition.name == "ElevenLabs"
    assert definition.base_url == "https://api.elevenlabs.io/v1"
    assert definition.speech_only is True
    assert definition.decision_only is False, "the flags are different classes"
    # The paste-a-key path: `create_api_key_login` tags itself, so every host
    # that attaches a prompt can complete this login.
    assert definition.login_kind == "api_key"
    assert definition.paste_prompt_required is True
    # And it is offered by the login surfaces, which is the half that has to
    # keep working — the whole point is that the key CAN be stored.
    assert "elevenlabs" in {provider.id for provider in registry.list_login_providers()}
    assert "eleven" in definition.search_aliases


def test_the_env_key_is_the_primary_name() -> None:
    assert registry.env_key_names("elevenlabs") == ("ELEVENLABS_API_KEY",)
    assert registry.env_key_name("elevenlabs") == "ELEVENLABS_API_KEY"
    assert registry.credential_file_names("elevenlabs") == ["ELEVENLABS_API_KEY"]


def test_only_the_speech_rows_are_speech_only() -> None:
    """The flag is not a broad brush: every other row stays selectable."""
    flagged = {row.id for row in registry.PROVIDER_REGISTRY if row.speech_only}

    assert flagged == {"elevenlabs", "openai-key"}
    assert registry.is_speech_only("elevenlabs") is True
    # Normalisation matches the sibling predicate's contract: case, padding,
    # aliases; None and unknown ids answer False.
    assert registry.is_speech_only("ElevenLabs") is True
    assert registry.is_speech_only("  elevenlabs  ") is True
    assert registry.is_speech_only("openai") is False
    assert registry.is_speech_only("typesafe") is False, "decision-only is the other flag"
    assert registry.is_speech_only(None) is False
    assert registry.is_speech_only("not-a-provider") is False


def test_the_speech_only_message_stops_at_the_fact() -> None:
    message = registry.speech_only_message("elevenlabs")

    assert "serves speech-to-text, not chat completions" in message
    assert "elevenlabs" in message
    # The sibling's sentence says something FALSE about this provider; keeping
    # them distinct is the reason two message functions exist.
    assert message != registry.decision_only_message("elevenlabs")


# ---------------------------------------------------------------------------
# 1. Discovery and listing
# ---------------------------------------------------------------------------


def test_the_static_catalogue_offers_no_elevenlabs_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even when the shipped registry WOULD hand it a model, the catalogue drops it.

    ``static_models`` is patched to answer for every provider id, which is the
    only shape of this test that can fail: a provider with no static models is
    absent from the catalogue whether or not the filter exists.
    """
    monkeypatch.setattr(
        "local_operator.providers.controller.static_models",
        lambda provider_id: {"scribe_v2": _Info()},
    )
    controller = _controller(tmp_path)

    entries = controller.static_catalogue()

    providers = {entry.provider for entry in entries}
    assert "elevenlabs" not in providers
    assert "openai" in providers, "the filter must not empty the catalogue"


def test_the_first_frame_catalogue_offers_no_elevenlabs_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "local_operator.providers.controller.static_models",
        lambda provider_id: {"scribe_v2": _Info()},
    )
    controller = _controller(tmp_path)

    entries = controller.initial_catalogue(cache_dir=tmp_path)

    assert "elevenlabs" not in {entry.provider for entry in entries}


@pytest.mark.asyncio
async def test_the_live_catalogue_never_asks_for_elevenlabs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No listing call at all: a speech provider is never asked for chat models.

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

    assert "elevenlabs" not in asked
    assert "openai" in asked
    assert "elevenlabs" not in {entry.provider for entry in entries}


@pytest.mark.asyncio
async def test_an_explicit_providers_set_still_gets_no_elevenlabs_rows(
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

    entries, _statuses = await controller.live_catalogue(providers={"elevenlabs", "openai"})

    assert "elevenlabs" not in asked
    assert asked == ["openai"]
    assert "elevenlabs" not in {entry.provider for entry in entries}


# ---------------------------------------------------------------------------
# 2. The /model ranking
# ---------------------------------------------------------------------------


def test_rank_rows_drops_a_speech_only_provider() -> None:
    """Ranking is the last gate before a row becomes a choice: filter here too."""
    rows = [
        ModelRow(provider="elevenlabs", model_id="scribe_v2", label="ElevenLabs/scribe_v2"),
        ModelRow(provider="openai", model_id="gpt-4o", label="OpenAI/gpt-4o"),
    ]

    # Empty query (the picker's resting frame): everything that survives, i.e.
    # the chat model and nothing else.
    assert [row.provider for row in rank_rows(list(rows), "")] == ["openai"]
    # A query that NAMES the provider: no match at all, rather than the one model
    # that cannot answer.
    for query in ("elevenlabs", "scribe"):
        assert rank_rows(list(rows), query) == [], query


# ---------------------------------------------------------------------------
# 3. Session model resolution
# ---------------------------------------------------------------------------


def _args(**kwargs: Any) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


def test_a_speech_only_hosting_is_refused_at_boot(tmp_path: Path) -> None:
    """The boot resolver refuses it, recoverably, naming the value and the way out."""
    manager = ConfigManager(tmp_path)

    with pytest.raises(HostingNotChatError) as caught:
        resolve_hosting_model_with_source(
            None, _args(hosting="elevenlabs", model="scribe_v2"), manager
        )

    error = caught.value
    assert error.hosting == "elevenlabs"
    assert error.source == "flag"
    assert "speech-to-text, not chat completions" in str(error)
    assert "--hosting" in str(error), "the remedy must be the one that can work"
    # Recoverable, like every other hosting fault: the TUI opens in the setup
    # state where `/model` can fix it, rather than a dead "session failed to
    # start" with no way to reach the command that repairs it.
    assert isinstance(error, HostingNotConfiguredError)
    # ...but NOT the unknown-provider class: this provider is known and has a
    # login, so "not a known provider" and its `/login` remedy would be false.
    assert not isinstance(error, HostingUnknownError)


def test_a_speech_only_hosting_from_config_names_the_config_remedy(tmp_path: Path) -> None:
    manager = ConfigManager(tmp_path)
    manager.set_config_value("hosting", "elevenlabs")

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


def test_a_stored_speech_only_selection_is_not_handed_out(tmp_path: Path) -> None:
    """The journal's own identity: the shared reader refuses it like Jev's."""
    import asyncio

    from local_operator.session.model_selection import read_model_selection
    from local_operator.session.transcript import Transcript

    directory = tmp_path / "sessions" / "speech-only"
    transcript = Transcript(directory)
    asyncio.run(
        transcript.append_custom(
            "selected_model", {"version": 2, "selector": "elevenlabs/scribe_v2"}
        )
    )

    assert read_model_selection(directory) is None, "the reader must not hand it out"


def test_a_stored_speech_only_selection_is_named_by_the_refusal_reader(
    tmp_path: Path,
) -> None:
    """The refusal is not silent: the resolver names the provider it refused.

    The counterpart of the test above. The shared reader refuses the row, and
    the second reader — the probe the resolver runs before it can fall back to
    the configured hosting — must name it for BOTH flag classes the validator
    can refuse, or a speech-only row resumes silently onto a model its own
    journal never named (agent review round 1, R1-3).
    """
    import asyncio

    from local_operator.session.model_selection import (
        read_model_selection,
        refused_decision_only_selection,
    )
    from local_operator.session.transcript import Transcript

    directory = tmp_path / "sessions" / "speech-only"
    transcript = Transcript(directory)
    asyncio.run(
        transcript.append_custom(
            "selected_model", {"version": 2, "selector": "elevenlabs/scribe_v2"}
        )
    )

    assert read_model_selection(directory) is None, "the reader must not hand it out"
    assert refused_decision_only_selection(directory) == "elevenlabs"

    # A perfectly good chat hosting in the config does not change the answer:
    # the conversation's own stored identity is the thing that cannot chat, and
    # falling through to the config would hide it.
    manager = ConfigManager(tmp_path)
    manager.update_config({"hosting": "openai", "model_name": "gpt-4o"}, write=False)

    with pytest.raises(HostingNotChatError) as caught:
        resolve_hosting_model_with_source(
            None, _args(resume="speech-only", hosting=None, model=None), manager
        )

    assert caught.value.hosting == "elevenlabs"
    assert caught.value.source == "resume"
    assert "speech-to-text, not chat completions" in str(caught.value)


# ---------------------------------------------------------------------------
# The last door: a live switch builds a spec, or it refuses
# ---------------------------------------------------------------------------


def test_build_model_spec_refuses_a_speech_only_hosting() -> None:
    from local_operator.model.configure import build_model_spec

    with pytest.raises(ValueError) as caught:
        build_model_spec("elevenlabs", "scribe_v2")

    assert "speech-to-text, not chat completions" in str(caught.value)


def test_validate_model_selection_refuses_with_a_typed_code() -> None:
    from local_operator.model.configure import (
        ModelSelectionRefused,
        validate_model_selection,
    )

    with pytest.raises(ModelSelectionRefused) as caught:
        validate_model_selection("elevenlabs", "scribe_v2")

    assert caught.value.code == "provider_speech_only"
    assert "speech-to-text" in caught.value.message


# ---------------------------------------------------------------------------
# 4. The failover chain
# ---------------------------------------------------------------------------


def test_expand_fallback_targets_drops_a_speech_only_entry() -> None:
    """A chain that names the voice provider must not route a turn onto it."""
    chain = ["elevenlabs/scribe_v2", "anthropic/claude-sonnet-4"]

    targets = expand_fallback_targets("openai/gpt-4o", chain)

    assert [target.selector for target in targets] == ["anthropic/claude-sonnet-4"]


def test_a_wildcard_speech_only_entry_is_dropped_too() -> None:
    """The provider prefix is read before the ``/*`` expansion inherits the id."""
    chain = ["elevenlabs/*", "openai/*"]

    targets = expand_fallback_targets("deepseek/deepseek-chat", chain)

    assert [target.selector for target in targets] == ["openai/deepseek-chat"]


def test_an_all_speech_only_chain_has_no_fallback() -> None:
    """The honest answer: no route, rather than a route that is guaranteed to fail."""
    assert expand_fallback_targets("openai/gpt-4o", ["elevenlabs/scribe_v2"]) == []
    assert expand_fallback_candidates("openai/gpt-4o", ["elevenlabs/scribe_v2"]) == []


# ---------------------------------------------------------------------------
# The fifth door: `login elevenlabs` stores the key, never a hosting
# ---------------------------------------------------------------------------


def test_a_speech_only_provider_is_never_adopted_as_hosting() -> None:
    """``login elevenlabs`` is how the VOICE path gets its key; routing stays put.

    A plan that adopted it as hosting — as the ordinary cases would — writes a
    config whose very next session cannot answer a turn, and ElevenLabs is the
    provider where the exemption is not just a guard but the ONLY correct
    outcome: storing the key is the entire point of its login.
    """
    from local_operator.providers.login_defaults import plan_login_defaults

    configured = plan_login_defaults("elevenlabs", "deepseek", "deepseek-chat")
    assert configured.hosting is None
    assert configured.model_name is None
    assert configured.receipt is not None
    assert configured.receipt == "Nothing changed — chats keep running on deepseek/deepseek-chat."

    for empty in ("", None):
        plan = plan_login_defaults("elevenlabs", empty, None)
        assert plan.hosting is None, empty
        assert plan.model_name is None
        assert plan.repairing is False
        assert plan.receipt == "Nothing changed — pick a chat model with /model first."

    # Case 3: hosting set but unusable — not repaired onto a provider that
    # cannot serve a session.
    repaired = plan_login_defaults("elevenlabs", "anthropicxyq", "claude-sonnet-4-5")
    assert repaired.hosting is None
    assert repaired.model_name is None
