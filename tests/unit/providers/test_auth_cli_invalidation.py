"""``lop login`` / ``lop logout`` drop the provider's cached model listing.

The CLI half of the credential-change invalidation ported from bbqben's #535.
The TUI half lives on ``ProviderController.login``/``logout`` and is covered in
``test_controller.py``; this file drives the REAL ``run_login``/``run_logout``
functions with the provider's login coroutine swapped for a canned answer, so
the hook is exercised where it is called rather than in isolation.

No network, no real ``~/.local-operator``: the cache directory is redirected to
``tmp_path`` through ``catalogue.default_cache_dir``, which is what the
no-argument ``invalidate_listing`` the CLI calls resolves through.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.model import catalogue
from local_operator.providers import auth_cli
from local_operator.providers.registry import get_provider_definition


class _Store:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    def upsert_credential(self, provider, credential):
        self.rows[provider] = dict(credential)
        return dataclasses.make_dataclass("Row", ["provider"])(provider)

    def delete_credentials_for_provider(self, provider, disabled_cause="logged-out"):
        return 1 if self.rows.pop(provider, None) is not None else 0


def _plant(cache: Path, name: str) -> Path:
    document = cache / name
    document.write_text(
        json.dumps({"fetched_at": time.time(), "payload": {"capture": 2, "models": []}}),
        encoding="utf-8",
    )
    return document


@pytest.fixture
def cache(tmp_path, monkeypatch) -> Path:
    monkeypatch.setattr(catalogue, "default_cache_dir", lambda: tmp_path)
    return tmp_path


def _swap_login(monkeypatch, provider_id: str, answer):
    definition = get_provider_definition(provider_id)
    assert definition is not None

    async def fake_login(_callbacks):
        return answer

    monkeypatch.setattr(
        auth_cli,
        "get_provider_definition",
        lambda pid: (
            dataclasses.replace(definition, login=fake_login)
            if pid == provider_id
            else get_provider_definition(pid)
        ),
    )


def test_an_api_key_login_drops_the_planted_listing(cache, monkeypatch, capsys) -> None:
    monkeypatch.setattr(auth_cli, "_apply_login_defaults", lambda provider_id: None)
    document = _plant(cache, "anthropic.listing.json")
    untouched = _plant(cache, "openrouter.listing.json")
    _swap_login(monkeypatch, "anthropic", "sk-ant-pasted")

    assert auth_cli.run_login("anthropic", None, _Store()) == 0  # type: ignore[arg-type]

    assert "Stored API key for 'anthropic'" in capsys.readouterr().out
    assert not document.exists(), "the listing fetched before the login is gone"
    assert untouched.exists(), "another provider's document is not"


def test_an_oauth_login_drops_every_document_of_the_storage_id(cache, monkeypatch) -> None:
    """``xai-oauth`` stores as ``xai``; the documents are named the same way."""
    monkeypatch.setattr(auth_cli, "_apply_login_defaults", lambda provider_id: None)
    plain = _plant(cache, "xai.listing.json")
    scoped = _plant(cache, "xai.oauth.listing.json")
    prefix_sibling = _plant(cache, "xai-other.listing.json")
    _swap_login(monkeypatch, "xai-oauth", {"access_token": "t", "refresh_token": "r"})

    assert auth_cli.run_login("xai-oauth", None, _Store()) == 0  # type: ignore[arg-type]

    assert not plain.exists() and not scoped.exists()
    assert prefix_sibling.exists(), "the dot separator keeps a prefix-sharing id apart"


def test_logout_drops_the_listing_and_a_failed_drop_never_fails_the_logout(
    cache, monkeypatch, capsys
) -> None:
    store = _Store()
    store.upsert_credential("anthropic", {"key": "k"})
    document = _plant(cache, "anthropic.listing.json")

    assert auth_cli.run_logout("anthropic", store) == 0  # type: ignore[arg-type]
    assert not document.exists()
    assert "Removed 1 credential(s)" in capsys.readouterr().out

    # Best-effort by construction: an unreadable cache is a stale list, not a
    # failed logout.
    store.upsert_credential("anthropic", {"key": "k"})

    def boom(provider_id):
        raise OSError("read-only cache")

    monkeypatch.setattr("local_operator.model.discovery.invalidate_listing", boom)
    assert auth_cli.run_logout("anthropic", store) == 0  # type: ignore[arg-type]
