"""The delegation seed stays off every NAME listing, and a Vault value keeps its
provenance: labelled as Vault, and never promoted into the long-term store.

Review round 1 on #3558: R2/D1 (the ``/credential`` label), R3 (three listings
still enumerated ``LOP_VAULT_DELEGATION_*``), S-4 (``--persist`` of a shared value
defeated the teammate's rotation and revocation).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from local_operator.secrets.access import open_store
from local_operator.secrets.promote import promote_session_credential
from local_operator.secrets.runtime import SecretsMapping
from local_operator.variables import VariableStore, format_credential_list
from local_operator.vault_agent import keys

from .conftest import ENDPOINT


def _seeded(config_dir: Path) -> None:
    keys.load_or_create_seed(ENDPOINT, config_dir)
    open_store(config_dir, create=True).set("GITHUB_TOKEN", b"ghp_x")


def test_every_source_has_its_own_label() -> None:
    store = VariableStore()
    store.store_credential("GH", "a", "command")
    store.store_credential("ASKED", "b", "ask")
    store.store_credential("RADIENT_VAULT_DEPLOY", "c", "vault", trim=False)
    rows = format_credential_list(store.list_credentials()).splitlines()
    assert any("GH" in r and r.endswith("from /credential") for r in rows)
    assert any("ASKED" in r and r.endswith("from ask") for r in rows)
    assert any("RADIENT_VAULT_DEPLOY" in r and r.endswith("from Radient Vault") for r in rows)


def test_a_vault_value_is_never_promoted_to_the_long_term_store(config_dir) -> None:
    store = VariableStore()
    store.store_credential("RADIENT_VAULT_DEPLOY", "shared-by-a-teammate", "vault", trim=False)
    result = promote_session_credential(store, "RADIENT_VAULT_DEPLOY", base=config_dir)
    assert not result.ok and "session-only" in result.message
    assert "shared-by-a-teammate" not in result.message
    # Nothing was written, and the session copy still works.
    assert not (config_dir / "secrets").exists() or open_store(config_dir).list() == []
    assert store.credential_env() == {"RADIENT_VAULT_DEPLOY": "shared-by-a-teammate"}


def test_the_cell_secrets_mapping_does_not_list_the_seed(config_dir) -> None:
    _seeded(config_dir)
    assert list(SecretsMapping(config_dir)) == ["GITHUB_TOKEN"]


def test_the_info_panel_does_not_list_the_seed(config_dir) -> None:
    from local_operator.info.collect import _credential_key_names

    _seeded(config_dir)
    assert _credential_key_names(config_dir) == ("GITHUB_TOKEN",)


def test_the_credentials_route_does_not_list_the_seed(config_dir, monkeypatch) -> None:
    from local_operator.providers import registry
    from local_operator.server.routes import credentials

    _seeded(config_dir)
    monkeypatch.setattr(registry, "stored_provider_env_keys", lambda _root: [])
    manager: Any = SimpleNamespace(config_dir=config_dir)
    response = asyncio.run(credentials.list_credentials(manager))
    assert response.result is not None
    assert response.result["keys"] == ["GITHUB_TOKEN"]


def test_the_remote_viewer_wire_keeps_the_vault_source() -> None:
    """Review round 2, M2: the remote-viewer path relabelled a Vault row."""
    from local_operator.variables import SessionCredential, credential_source_from_wire

    assert credential_source_from_wire("vault") == "vault"
    assert credential_source_from_wire("ask") == "ask"
    assert credential_source_from_wire("command") == "command"
    assert credential_source_from_wire("something-new") == "command"
    rows = [SessionCredential(key="RADIENT_VAULT_X", source=credential_source_from_wire("vault"))]
    assert format_credential_list(rows).splitlines()[1].endswith("from Radient Vault")
