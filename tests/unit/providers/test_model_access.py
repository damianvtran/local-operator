"""``model_access``: which providers a user can run on, and which models are stranded.

Why these tests exist at all: this module decides whether a sign-in may rewrite a
user's DEFAULT and move their live sessions. Every term in it is a claim about
someone's machine — a stored row, an env key, a local server they pointed
somewhere, a credential borrowed from a paired device — so the matrix is the
contract, and the two directions (``is_stranded`` for the source, ``is_accessible``
for the target) are pinned together so they cannot drift apart.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest


class _Cred:
    """A stored credential row, in the slice ``usable_providers`` reads."""

    def __init__(self, provider: str, credential_type: str = "api_key") -> None:
        self.provider = provider
        self.credential_type = credential_type


class _Store:
    """A credential store double: rows in, rows out, close recorded."""

    instances: list["_Store"] = []

    def __init__(self, rows: list[_Cred] | None = None) -> None:
        self.rows = list(rows or [])
        self.closed = False
        _Store.instances.append(self)

    def list_credentials(self, provider: Any = None) -> list[_Cred]:
        return list(self.rows)

    def close(self) -> None:
        self.closed = True


class _Controller:
    """The slice of ``ProviderController`` the wrapper uses."""

    def __init__(self, usable: set[str] | None, config_dir: Path | None = None) -> None:
        self._usable = usable
        self.config_dir = config_dir

    def usable_providers(self) -> set[str] | None:
        return self._usable


def test_an_unknown_store_is_none_and_never_an_empty_set() -> None:
    """``None`` is "cannot tell": a caller must move nothing, not "nothing is signed in"."""
    from local_operator.providers.model_access import credentialed_chat_providers

    assert credentialed_chat_providers(_Controller(None)) is None


def test_decision_speech_and_media_providers_are_not_chat_capable() -> None:
    """A stored FAL or ElevenLabs key is not a model to run a turn on."""
    from local_operator.providers.model_access import credentialed_chat_providers

    usable = {"openai", "typesafe", "fal", "elevenlabs"}
    assert credentialed_chat_providers(_Controller(usable)) == {"openai"}


def test_keyless_locals_count_only_when_the_user_pointed_one_somewhere() -> None:
    """The design's own line: the picker offers them, "did they have anything" does not.

    ``providers.<id>.base_url`` is the app's record of a deliberate opt-in
    (``configure_local_providers``), so a preset-only ``ollama`` is dropped while
    a configured one stays — and the generic ``openai-compatible`` gateway can
    ONLY appear there by being configured, which is the same fact reached the same
    way.
    """
    from local_operator.providers.model_access import credentialed_chat_providers

    usable = {"ollama", "lmstudio", "openai-compatible", "openai", "test"}
    assert credentialed_chat_providers(_Controller(usable)) == {"openai"}
    configured = credentialed_chat_providers(
        _Controller(usable),
        config_values={"providers": {"ollama": {"base_url": "http://192.168.1.9:11434/v1"}}},
    )
    assert configured == {"openai", "ollama"}


def test_the_test_provider_is_never_credentialed(monkeypatch: pytest.MonkeyPatch) -> None:
    """``test`` needs no key, so it can never be evidence that a user was set up."""
    from local_operator.providers.model_access import credentialed_chat_providers

    monkeypatch.setenv("TEST_API_KEY", "x")
    assert credentialed_chat_providers(_Controller({"test"})) == set()


def test_a_borrowed_credential_counts_as_accessible(monkeypatch: pytest.MonkeyPatch) -> None:
    """A paired device that lends ``radient`` has no local row, and still runs turns.

    Without this rung the planner would call that device signed out of Radient and
    the re-home would move a session that works — the design's explicit warning.
    The placement read is stubbed at its own seam because the real document needs a
    joined network; what is pinned here is that the wrapper UNIONS what it returns.
    """
    from local_operator.providers import model_access

    monkeypatch.setattr(model_access, "_borrowed_provider_keys", lambda config_dir: {"radient"})
    assert model_access.credentialed_chat_providers(_Controller({"openai"})) == {
        "openai",
        "radient",
    }


def test_borrowed_keys_are_empty_when_there_is_no_placement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The cheap predicate: a device that never joined a network borrows nothing."""
    from local_operator.providers.model_access import _borrowed_provider_keys

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / ".local-operator"))
    assert _borrowed_provider_keys(tmp_path / ".local-operator") == set()


@pytest.mark.parametrize(
    ("provider", "accessible", "stranded", "accessible_target"),
    [
        # Signed in: neither stranded nor a candidate target needing proof.
        ("openai", {"openai"}, False, True),
        # Not in the set: stranded, and not a legal target either.
        ("openai", {"deepseek"}, True, False),
        # A flavour's storage id answers for the base provider, both directions.
        ("anthropic", {"anthropic-key"}, False, True),
        ("anthropic-key", {"anthropic"}, False, True),
        # Unknowable: never moves anything, in either direction.
        ("openai", None, False, False),
        # Empty and unregistered ids are the planner's other case, not this one.
        ("", {"openai"}, False, False),
        ("not-a-provider", {"openai"}, False, False),
    ],
)
def test_the_two_directions_are_exact_mirrors(
    provider: str, accessible: set[str] | None, stranded: bool, accessible_target: bool
) -> None:
    from local_operator.providers.model_access import is_accessible, is_stranded

    assert is_stranded(provider, accessible) is stranded
    assert is_accessible(provider, accessible) is accessible_target
    if accessible is None:
        # The invariant that matters: an unreadable store moves nothing, so the
        # two predicates can never both fire on one provider.
        assert not (stranded and accessible_target)


def test_a_real_store_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The real path, not a double: store a row, read the set back through a controller.

    Also pins the DB path: ``config_dir`` alone does NOT set ``AuthStore``'s
    database, so the helper has to spell ``<root>/auth.db`` itself or an explicit
    root would silently read the ambient store.
    """
    from local_operator.providers.auth_store import AuthStore
    from local_operator.providers.controller import ProviderController
    from local_operator.providers.model_access import (
        credentialed_chat_providers,
        credentialed_chat_providers_here,
    )

    root = tmp_path / "cfg"
    root.mkdir()
    store = AuthStore(root / "auth.db", config_dir=root)
    try:
        store.upsert_credential("deepseek", {"type": "api_key", "source": "login", "key": "k"})
        controller = ProviderController(store, root)
        assert credentialed_chat_providers(controller) == {"deepseek"}
    finally:
        store.close()
    here = credentialed_chat_providers_here(config_dir=root)
    assert here == {"deepseek"}
    # And the ambient store is untouched by that call: the helper reads the root
    # it was given.
    other = credentialed_chat_providers_here(config_dir=tmp_path / "elsewhere")
    assert other == set(), other


def test_an_unreadable_store_is_none_not_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corrupt database must degrade to "cannot tell", like ``usable_providers``."""
    import local_operator.providers.auth_store as auth_store_mod
    from local_operator.providers import model_access

    root = tmp_path / "cfg"
    root.mkdir()
    (root / "auth.db").write_bytes(b"not a database at all")

    def _boom(self: Any, provider: Any = None) -> list[Any]:
        raise sqlite3.DatabaseError("file is not a database")

    monkeypatch.setattr(auth_store_mod.AuthStore, "list_credentials", _boom)
    assert model_access.credentialed_chat_providers_here(config_dir=root) is None


# ---------------------------------------------------------------------------
# The FIRST-LOGIN rule (operator refinement, round 1): the session re-home is
# scoped to the login that turns "nothing" into "something"
# ---------------------------------------------------------------------------


def test_an_empty_set_is_not_a_first_login() -> None:
    """The provider just added must BE in the set — an empty read proves nothing.

    A credential write precedes every call, so an empty set means the read is
    wrong rather than the user has nothing; moving sessions on it would be a
    guess, and every doubt here answers "no move".
    """
    from local_operator.providers.model_access import is_first_provider_login

    assert is_first_provider_login(set(), "openai") is False
    assert is_first_provider_login(None, "openai") is False


def test_a_single_provider_login_is_the_first() -> None:
    """One credentialed chat provider, the one just added: the repair's whole case."""
    from local_operator.providers.model_access import is_first_provider_login

    assert is_first_provider_login({"openai"}, "openai") is True
    # A login FLAVOUR is the same account: ``openai-device`` stores under
    # ``openai``, and signing in with it while only ``openai`` is credentialed is
    # still that account's login, not a second one.
    assert is_first_provider_login({"openai"}, "openai-device") is True
    assert is_first_provider_login({"anthropic"}, "anthropic-key") is True


def test_a_second_provider_login_never_moves_sessions() -> None:
    """The refinement's core: N>1 credentialed providers means "not the first".

    A user who can already run turns has conversations of their own; a later
    sign-in adds a provider to switch models with, and must not re-point them.
    Radient counts like any other provider, deliberately: a prior Radient/web
    sign-in means "already logged in", so a later OpenAI login is not a first.
    """
    from local_operator.providers.model_access import is_first_provider_login

    assert is_first_provider_login({"openai", "deepseek"}, "openai") is False
    assert is_first_provider_login({"radient", "openai"}, "openai") is False


def test_a_configured_local_counts_as_an_earlier_login() -> None:
    """A pointed-at local server is a working choice, so it makes the next login non-first.

    ``credentialed_chat_providers`` includes a configured ``ollama`` precisely
    because the user opted in; the rule reads that same set, so the two cannot
    disagree about whether this user already had somewhere to run.
    """
    from local_operator.providers.model_access import is_first_provider_login

    assert is_first_provider_login({"ollama", "openai"}, "openai") is False
