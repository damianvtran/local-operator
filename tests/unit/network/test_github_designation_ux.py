"""Deferred GitHub UX and account-hint boundaries; no live login or GitHub API."""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

from local_operator.i18n import catalogues
from local_operator.network import readiness
from local_operator.network.credentials import github as github_mod
from tests.unit.network.test_github_pushable_designation import (
    TOKEN,
    _cli,
    _config,
    _repo,
    _token_arm,
)
from tests.unit.network.test_github_pushable_designation import api as _api_fixture
from tests.unit.network.test_github_pushable_designation import (
    app_keypair as _app_keypair_fixture,
)
from tests.unit.network.test_github_pushable_designation import env as _env_fixture

# Canonical fixture exports, distinct from the parameters that request them.
api = _api_fixture
app_keypair = _app_keypair_fixture
env = _env_fixture


def test_readiness_full_list_pointer_does_not_stack_closing_parentheses() -> None:
    phrase = readiness._designation_phrase(
        {
            "repositories": ["octo-org/app"],
            "repositories_count": 1,
            "repositories_origin": "enumerated",
            "repositories_age_s": 0,
        },
        "owner-laptop",
    )
    assert ") — full list: " in phrase
    assert "credential github repos on owner-laptop" in phrase


def test_github_readiness_owns_its_conjunction_instead_of_using_the_global_english_join(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(names: Any) -> str:
        raise AssertionError("the owned GitHub clause still uses the English-only join")

    monkeypatch.setattr(readiness, "_name_list", forbidden)
    for names, expected in [
        (["octo-org/a"], "octo-org/a"),
        (["octo-org/a", "octo-org/b"], "octo-org/a and octo-org/b"),
        (["octo-org/a", "octo-org/b", "octo-org/c"], "octo-org/a, octo-org/b and octo-org/c"),
    ]:
        phrase = readiness._designation_phrase(
            {
                "repositories": names,
                "repositories_count": len(names),
                "repositories_origin": "enumerated",
                "repositories_age_s": 0,
            },
            "owner",
        )
        assert phrase.startswith(expected + " (")


def test_settings_help_explains_clear_fence_and_the_existing_default_reset() -> None:
    help_text = catalogues.load_catalogue("en", "wire.settings")[
        "wire.settings.network_credentials_github_repositories.help"
    ]
    assert "When never set" in help_text and "minus opt-outs" in help_text
    assert "Clearing a previously shared" in help_text and "stops lending" in help_text
    assert "lop network credential github repos --use-default" in help_text
    assert "empty: none" not in help_text


@pytest.mark.parametrize("code", ["github_repositories_unset", "github_repositories_overflow"])
def test_localized_refusal_round_trip_uses_code_bound_provenance_not_english(
    monkeypatch: pytest.MonkeyPatch,
    code: str,
) -> None:
    from types import SimpleNamespace

    from local_operator.network.credentials import messages
    from local_operator.network.credentials import types as credential_types
    from local_operator.network.credentials.client import MeshCredentialClient

    context: Any = SimpleNamespace(
        owner_label=lambda key: "owner",
        owner_of=lambda key: "d_owner",
        owner_last_seen_s=lambda owner: None,
    )
    calls: list[Any] = []

    def localized(message: Any) -> str:
        calls.append(message)
        return "localized-owned-message-with-no-English-wrapper-marker"

    monkeypatch.setattr(messages, "_github_text", localized, raising=False)
    error = credential_types.BrokerError(code=code, message="owned producer detail")
    first = MeshCredentialClient._render(context, error, "github", "GitHub")
    assert first.reason == code + "_rendered"
    decoded = credential_types.BrokerError.from_detail(first.to_detail())
    second = MeshCredentialClient._render(context, decoded, "github", "GitHub")
    assert second.message == first.message and len(calls) == 1


@pytest.mark.parametrize("existing", ["holder_excluded", "pool_depleted", "other-existing-reason"])
def test_github_rendering_never_overwrites_an_existing_semantic_reason(existing: str) -> None:
    from types import SimpleNamespace

    from local_operator.network.credentials.client import MeshCredentialClient
    from local_operator.network.credentials.types import BrokerError

    context: Any = SimpleNamespace(
        owner_label=lambda key: "owner",
        owner_of=lambda key: "d_owner",
        owner_last_seen_s=lambda owner: None,
    )
    error = BrokerError(code="github_repositories_unset", reason=existing)
    assert MeshCredentialClient._render(context, error, "github", "GitHub").reason == existing


def test_presentation_tags_cannot_change_quota_auth_retry_or_legacy_absent_reasons() -> None:
    from local_operator.network.credentials import types as credential_types

    for code in credential_types.BROKER_ERROR_TTL_MS:
        for marker in [
            "github_repositories_unset_rendered",
            "github_repositories_overflow_rendered",
        ]:
            plain = credential_types.BrokerError.from_detail({"code": code})
            tagged = credential_types.BrokerError.from_detail({"code": code, "reason": marker})
            assert plain.reason == ""
            expected = marker if marker == code + "_rendered" else ""
            assert tagged.reason == expected
            assert tagged.cache_ttl_ms == plain.cache_ttl_ms
            assert tagged.is_holder_exclusion == plain.is_holder_exclusion
    for reason in ["holder_excluded", "pool_depleted"]:
        result = credential_types.BrokerError.from_detail(
            {"code": "quota_blocked", "reason": reason}
        )
        assert result.reason == reason
        assert result.is_holder_exclusion == (reason == "holder_excluded")


def _gh_cache(holder: Any, *, hint: str) -> None:
    _config(holder.root, source="gh")
    github_mod._safe_write_cache(
        holder.root,
        {
            "arm": github_mod.SOURCE_GH,
            "identity": github_mod.source_identity(github_mod.SOURCE_GH, token=TOKEN),
            "hint": hint,
            "fetched_at": time.time(),
            "repositories": ["account-one/private"],
        },
    )


def test_an_unreadable_account_hint_is_unknown_not_an_earlier_login(
    env: Any,
    monkeypatch: pytest.MonkeyPatch,  # noqa: F811 — fixture by import
) -> None:
    _gh_cache(env, hint="")
    monkeypatch.setattr(github_mod, "_read_gh_hosts", lambda home=None: {})
    result = github_mod.designation(
        env.root, source=github_mod.SOURCE_GH, allow_fetch=False, home=env.home
    )
    assert result.repositories == ()
    assert "unknown" in result.error and "earlier" not in result.error


@pytest.mark.parametrize("variable", ["GH_TOKEN", "GITHUB_TOKEN"])
def test_an_environment_login_does_not_reuse_the_hosts_accounts_hint_or_list(
    env: Any,
    monkeypatch: pytest.MonkeyPatch,
    variable: str,  # noqa: F811
) -> None:
    _gh_cache(env, hint="gh:account-one")
    monkeypatch.setattr(
        github_mod, "_read_gh_hosts", lambda home=None: {"github.com": {"user": "account-one"}}
    )
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    # Positive control: without an override the current metadata matches.
    same = github_mod.designation(
        env.root, source=github_mod.SOURCE_GH, allow_fetch=False, home=env.home
    )
    assert same.repositories == ("account-one/private",)
    monkeypatch.setenv(variable, "synthetic-different-account-credential")
    assert github_mod.source_hint(github_mod.SOURCE_GH, env.root, home=env.home) != "gh:account-one"
    other = github_mod.designation(
        env.root, source=github_mod.SOURCE_GH, allow_fetch=False, home=env.home
    )
    assert other.repositories == ()
    assert "unknown" in other.error and "earlier" not in other.error
    # The authorization match remains credential-bound, not a metadata hint.
    assert github_mod.source_identity(
        github_mod.SOURCE_GH, token=TOKEN
    ) != github_mod.source_identity(
        github_mod.SOURCE_GH, token="synthetic-different-account-credential"
    )


def test_including_a_mistyped_opt_out_is_harmless_cleanup_not_another_spelling_warning(
    env: Any,
    api: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],  # noqa: F811
) -> None:
    _token_arm(env, api, [_repo("octo-org/app")])
    monkeypatch.setattr(github_mod, "default_minter", api.minter)
    assert _cli("credential", "github", "exclude", "octo-org/typo") == 0
    capsys.readouterr()
    assert _cli("credential", "github", "include", "octo-org/typo") == 0
    text = capsys.readouterr().out
    assert "octo-org/typo" in text and "nothing else changes" in text
    assert "check the spelling" not in text


@pytest.mark.parametrize("as_json", [False, True], ids=["text", "json"])
def test_explicit_repos_refresh_wires_delayed_progress_only_for_text(
    env: Any,
    api: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],  # noqa: F811
    as_json: bool,
) -> None:
    import threading

    _token_arm(env, api, [_repo("octo-org/app")])
    monkeypatch.setattr(github_mod, "default_minter", api.minter)

    # Exercise the existing callback without wall-clock sleeps. The sibling N5
    # test independently covers its real delayed timer and fast-call cancellation.
    class ExpiredTimer:
        def __init__(self, delay: float, callback: Any) -> None:
            self.callback = callback

        def start(self) -> None:
            self.callback()

        def cancel(self) -> None:
            pass

    monkeypatch.setattr(threading, "Timer", ExpiredTimer)
    args = ["credential", "github", "repos", "--refresh"] + (["--json"] if as_json else [])
    assert _cli(*args) == 0
    captured = capsys.readouterr()
    assert captured.err.count("listing the repositories") == (0 if as_json else 1)
    if as_json:
        assert json.loads(captured.out)["repositories"] == ["octo-org/app"]
