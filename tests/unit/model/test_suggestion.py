"""``local_operator.model.suggestion``: the fail-over rule and its reason table.

What these pin, and why each shape is what it is:

- The five reasons, first match wins, each produced by the authority the design
  names — and NONE of them raising: a suggestion can be anything a hand-edited
  zip or ``team.yml`` carries, so the verdict is data, never an error.
- The uncertainty rule (§4.1): a check that cannot be answered — an unreadable
  credential store, an unreadable catalogue, ``offered_model_ids``' documented
  ``None`` — ACCEPTS the pair. Only positive evidence (a usable provider whose
  enumerable catalogue lacks the id) fails over; anything else would drop a
  suggestion because a database was briefly locked.
- Id spelling: the match runs through ``normalised_id`` on BOTH sides, so the
  Gemini-style ``models/`` prefix and case differences cannot refuse an id the
  picker itself would have offered.

The authorities are monkeypatched at the module the resolver imports from
(function-local imports, so patching ``local_operator.model.discovery`` and
``local_operator.providers.local`` attributes is what takes effect), and no
test touches the network or the operator's home.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.model.suggestion import (
    REASON_INVALID,
    REASON_LOCAL_NOT_CONFIGURED,
    REASON_PROVIDER_UNAVAILABLE,
    REASON_UNKNOWN_MODEL,
    REASON_UNKNOWN_PROVIDER,
    ModelNotice,
    SuggestionVerdict,
    resolve_model_suggestion,
)


class _FakeAuthStore:
    """The only surface ``ProviderController.is_usable`` reads from a store."""

    def __init__(self, providers: tuple[str, ...] = ()) -> None:
        self._providers = providers

    def list_credentials(self, provider: str | None = None) -> list[Any]:
        return [SimpleNamespace(provider=name) for name in self._providers]


class _UnreadableAuthStore:
    """A store whose read fails — the uncertainty case that must ACCEPT."""

    def list_credentials(self, provider: str | None = None) -> list[Any]:
        raise OSError("database is locked")


@pytest.fixture(autouse=True)
def _no_operator_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let the operator's shell decide what these tests mean.

    ``resolve_env_key`` reads the process environment, so a developer with
    ``OPENROUTER_API_KEY`` exported would see a different reason than CI does.
    The providers used here are exactly the env-keyed ones the tests exercise.
    """

    for name in ("OPENROUTER_API_KEY", "RADIANT_API_KEY", "MISTRAL_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def _offered(monkeypatch: pytest.MonkeyPatch, value: set[str] | None) -> list[Any]:
    """Patch the catalogue authority; return the recorded calls."""

    calls: list[Any] = []

    def fake(provider_id: str, *, cache_dir: Any = None) -> set[str] | None:
        calls.append((provider_id, cache_dir))
        return value

    monkeypatch.setattr("local_operator.model.discovery.offered_model_ids", fake)
    return calls


def _configured_local(monkeypatch: pytest.MonkeyPatch, value: frozenset[str]) -> None:
    monkeypatch.setattr(
        "local_operator.providers.local.configured_local_providers",
        lambda values=None: value,
    )


# --- shape: invalid ---------------------------------------------------------------


@pytest.mark.parametrize(
    "suggestion",
    [
        None,
        "openrouter",
        {"hosting": "openrouter"},
        {"model": "x"},
        {"hosting": "", "model": "x"},
        {"hosting": "  ", "model": "x"},
        {"hosting": "openrouter", "model": ""},
        {"hosting": 5, "model": "x"},
        {"model": "x", "hosting": ["openrouter"]},
    ],
)
def test_a_malformed_suggestion_is_invalid_never_an_error(suggestion: Any) -> None:
    verdict = resolve_model_suggestion(suggestion)

    assert verdict.available is False
    assert verdict.reason == REASON_INVALID


def test_an_invalid_suggestion_echoes_whatever_is_readable() -> None:
    verdict = resolve_model_suggestion({"hosting": "  openrouter ", "model": ""})

    assert verdict.reason == REASON_INVALID
    assert verdict.hosting == "openrouter"
    assert verdict.model == ""


# --- unknown provider -------------------------------------------------------------


def test_an_unknown_provider_id_fails_over_by_name() -> None:
    verdict = resolve_model_suggestion({"hosting": "nope-not-a-provider", "model": "m"})

    assert verdict.reason == REASON_UNKNOWN_PROVIDER
    assert verdict.hosting == "nope-not-a-provider"


# --- known provider, nothing to run it --------------------------------------------


def test_a_known_provider_with_no_credential_is_not_logged_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _offered(monkeypatch, None)

    verdict = resolve_model_suggestion(
        {"hosting": "openrouter", "model": "anthropic/claude-opus-5.5"},
        auth_store=_FakeAuthStore(()),
    )

    assert verdict.reason == REASON_PROVIDER_UNAVAILABLE
    assert verdict.available is False


def test_a_stored_credential_makes_the_provider_usable(monkeypatch: pytest.MonkeyPatch) -> None:
    _offered(monkeypatch, None)

    verdict = resolve_model_suggestion(
        {"hosting": "openrouter", "model": "anthropic/claude-opus-5.5"},
        auth_store=_FakeAuthStore(("openrouter",)),
    )

    assert verdict.available is True
    assert verdict.reason is None


def test_a_provider_that_needs_no_credential_is_usable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The local presets and the mock: ``allows_missing_api_key`` IS the credential."""

    _offered(monkeypatch, None)
    _configured_local(monkeypatch, frozenset({"lmstudio"}))

    verdict = resolve_model_suggestion(
        {"hosting": "lmstudio", "model": "qwen2.5-coder-32b-instruct-q8_0"},
        auth_store=_FakeAuthStore(()),
    )

    assert verdict.available is True


# --- local realm: never pointed anywhere ------------------------------------------


def test_a_local_preset_never_configured_is_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configured_local(monkeypatch, frozenset())
    calls = _offered(monkeypatch, None)

    verdict = resolve_model_suggestion(
        {"hosting": "lmstudio", "model": "m"}, auth_store=_FakeAuthStore(())
    )

    assert verdict.reason == REASON_LOCAL_NOT_CONFIGURED
    # The model question is not even asked: the realm itself is absent.
    assert calls == []


def test_a_local_preset_configured_with_nothing_cached_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cached-only (OQ3): ``None`` from the catalogue means "cannot say" ⇒ accept."""

    _configured_local(monkeypatch, frozenset({"ollama"}))
    _offered(monkeypatch, None)

    verdict = resolve_model_suggestion(
        {"hosting": "ollama", "model": "some-local-id"}, auth_store=_FakeAuthStore(())
    )

    assert verdict.available is True


def test_a_configured_local_preset_with_a_cached_listing_checks_the_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configured_local(monkeypatch, frozenset({"ollama"}))
    _offered(monkeypatch, {"llama3.2"})

    missing = resolve_model_suggestion(
        {"hosting": "ollama", "model": "not-pulled"}, auth_store=_FakeAuthStore(())
    )
    present = resolve_model_suggestion(
        {"hosting": "ollama", "model": "llama3.2"}, auth_store=_FakeAuthStore(())
    )

    assert missing.reason == REASON_UNKNOWN_MODEL
    assert present.available is True


# --- enumerable catalogue lacks the id --------------------------------------------


def test_an_enumerable_catalogue_missing_the_id_is_unknown_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _offered(monkeypatch, {"provider/one", "provider/two"})

    verdict = resolve_model_suggestion(
        {"hosting": "openrouter", "model": "provider/three"},
        auth_store=_FakeAuthStore(("openrouter",)),
    )

    assert verdict.reason == REASON_UNKNOWN_MODEL


# --- id spelling ------------------------------------------------------------------


def test_ids_match_through_normalisation_on_both_sides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Gemini ``models/`` prefix and case fold cannot refuse a picker's offer."""

    _offered(monkeypatch, {"gemini-2.5-pro", "anthropic/claude-opus-5.5"})

    prefixed = resolve_model_suggestion(
        {"hosting": "google", "model": "models/Gemini-2.5-Pro"},
        auth_store=_FakeAuthStore(("google",)),
    )
    cased = resolve_model_suggestion(
        {"hosting": "openrouter", "model": "Anthropic/Claude-Opus-5.5"},
        auth_store=_FakeAuthStore(("openrouter",)),
    )

    assert prefixed.available is True
    assert cased.available is True


# --- uncertainty accepts -----------------------------------------------------------


def test_an_unreadable_store_accepts_the_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    """A locked database is not evidence of "no credential" (the stated rule)."""

    _offered(monkeypatch, None)

    verdict = resolve_model_suggestion(
        {"hosting": "openrouter", "model": "m"}, auth_store=_UnreadableAuthStore()
    )

    assert verdict.available is True


def test_an_unreadable_catalogue_accepts_the_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(provider_id: str, *, cache_dir: Any = None) -> set[str] | None:
        raise OSError("cache directory vanished")

    monkeypatch.setattr("local_operator.model.discovery.offered_model_ids", broken)

    verdict = resolve_model_suggestion(
        {"hosting": "openrouter", "model": "m"}, auth_store=_FakeAuthStore(("openrouter",))
    )

    assert verdict.available is True


def test_none_from_the_catalogue_accepts_the_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    """``offered_model_ids``' own contract: ``None`` means "cannot be enumerated"."""

    _offered(monkeypatch, None)

    verdict = resolve_model_suggestion(
        {"hosting": "openrouter", "model": "m"}, auth_store=_FakeAuthStore(("openrouter",))
    )

    assert verdict.available is True


# --- the verdict's carried notice ---------------------------------------------------


def test_the_verdict_names_the_trimmed_pair_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _offered(monkeypatch, {"m"})

    verdict = resolve_model_suggestion(
        {"hosting": " openrouter ", "model": " m "},
        auth_store=_FakeAuthStore(("openrouter",)),
    )

    assert verdict == SuggestionVerdict(True, None, "openrouter", "m")
    assert verdict.notice() is None


def test_the_notice_payload_and_line_carry_reason_and_request() -> None:
    verdict = resolve_model_suggestion({"hosting": "definitely-not-a-provider", "model": "m"})
    notice = verdict.notice()

    assert notice is not None
    assert notice.as_payload() == {
        "reason": REASON_UNKNOWN_PROVIDER,
        "requested": {"hosting": "definitely-not-a-provider", "model": "m"},
    }
    line = notice.describe()
    assert "definitely-not-a-provider" in line
    assert "not applied" in line
    # Wording shape only; the copy round owns the sentence.
    assert (
        ModelNotice(REASON_INVALID, "", "").describe().endswith("Using your default model instead.")
    )
