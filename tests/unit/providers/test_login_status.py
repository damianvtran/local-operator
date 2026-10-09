"""``login status`` / ``/login status``: the CLI half (the TUI half is in tests/unit/tui)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

from local_operator.providers import auth_cli
from local_operator.providers.auth_store import AuthStore
from local_operator.providers.registry import get_provider_definition


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[AuthStore]:
    # A root with no provider-class rows, and no ambient provider vars: the
    # point of every test below is what the probe does WITHOUT them.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "empty-config"))
    for var in ("OPENAI_API_KEY", "ELEVENLABS_API_KEY", "RADIENT_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    auth = AuthStore(db_path=tmp_path / "auth.db", config_dir=tmp_path / "config")
    yield auth
    auth.close()


# ---------------------------------------------------------------------------
# /login status (CLI half; the TUI half is in tests/unit/tui)
# ---------------------------------------------------------------------------


def test_login_status_is_not_a_provider_id() -> None:
    assert get_provider_definition(auth_cli.LOGIN_STATUS_WORD) is None


def test_run_login_status_prints_the_listing_and_no_secret(
    store: AuthStore, capsys: pytest.CaptureFixture[str]
) -> None:
    store.upsert_credential(
        "openai-key", {"type": "api_key", "source": "login", "key": "sk-never-printed"}
    )
    assert auth_cli.run_login("status", None, store) == 0
    out = capsys.readouterr().out
    assert "openai-key" in out and "api_key (login)" in out
    assert "sk-never-printed" not in out


def test_format_logins_names_env_vars_not_values(
    store: AuthStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "ambient-secret")
    lines = auth_cli.format_logins(store, None)
    text = "\n".join(lines)
    assert "ELEVENLABS_API_KEY=<set>" in text
    assert "ambient-secret" not in text
    assert "No stored credentials." in text


def test_the_status_word_is_case_insensitive_like_the_tui(
    store: AuthStore, capsys: pytest.CaptureFixture[str]
) -> None:
    """QA Q2: ``/login STATUS`` worked in the TUI, ``lop login STATUS`` did not."""
    assert auth_cli.run_login(" Status ", None, store) == 0
    assert "No stored credentials." in capsys.readouterr().out


def test_the_url_header_matches_the_flow_it_belongs_to(capsys) -> None:
    """Review round 1, Q3: a paste-key row's URL is where the key is MADE.

    ``on_auth_url`` printed "Open this URL to authorize:" for every login, so a
    row whose whole flow is "copy a key off a dashboard" told the user to
    authorize a page that only offers "Create new secret key" — the same
    mismatch the prompt text was already fixed for ("API key", never "code").
    """
    from local_operator.providers.registry import get_provider_definition

    url = "https://platform.openai.com/api-keys"

    paste = get_provider_definition("openai-api-key")
    assert paste is not None and paste.paste_prompt_required is True
    # ``LoginCallbacks.on_auth_url`` is Optional (a host may not publish on the
    # URL at all); bind the object and narrow before the call so the type gate
    # sees the same non-None the assertion pins.
    paste_callbacks = auth_cli._callbacks_interactive(paste)
    assert paste_callbacks.on_auth_url is not None
    paste_callbacks.on_auth_url(url)
    assert "create a key" in capsys.readouterr().out

    browser = get_provider_definition("openai")
    assert browser is not None and browser.paste_prompt_required is False
    browser_callbacks = auth_cli._callbacks_interactive(browser)
    assert browser_callbacks.on_auth_url is not None
    browser_callbacks.on_auth_url(url)
    out = capsys.readouterr().out
    assert "authorize" in out
    assert "create a key" not in out
