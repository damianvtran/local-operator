"""H2: the Radient Vault delegation seed lives in a namespace closed both ways.

The seed (``LOP_VAULT_DELEGATION_<endpoint>``) signs agent proofs for as long as
a grant lives, so an agent-class or provider-class verb must never reach it, and
the delegation-class role must reach nothing else. The read and delete halves
matter most: a hostile caller wants the row that already exists.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from local_operator.secrets.errors import InvalidSecretName
from local_operator.secrets.store import (
    DELEGATION_SECRET_PREFIX,
    SecretStore,
    is_delegation_secret_name,
    is_reserved_secret_name,
    secret_class,
)

NAME = DELEGATION_SECRET_PREFIX + "bep_x"
SEED = b"00" * 32


@pytest.fixture
def seeded(store: SecretStore) -> SecretStore:
    store.set(NAME, SEED, role="delegation")
    return store


def test_predicates_and_display_class() -> None:
    assert is_delegation_secret_name(NAME)
    assert is_delegation_secret_name("  " + NAME)
    assert not is_delegation_secret_name("LOP_PROVIDER_X")
    assert is_reserved_secret_name(NAME)
    assert is_reserved_secret_name("LOP_PROVIDER_X")
    assert not is_reserved_secret_name("GITHUB_TOKEN")
    assert secret_class(NAME) == "delegation"


@pytest.mark.parametrize("role", ["agent", "provider"])
def test_other_roles_cannot_create_a_delegation_row(store: SecretStore, role: str) -> None:
    with pytest.raises(InvalidSecretName):
        store.set(NAME, SEED, role=role)


@pytest.mark.parametrize("role", ["agent", "provider"])
def test_other_roles_cannot_read_describe_update_or_delete_it(
    seeded: SecretStore, role: str
) -> None:
    with pytest.raises(InvalidSecretName):
        seeded.get(NAME, role=role)
    with pytest.raises(InvalidSecretName):
        seeded.describe(NAME, role=role)
    with pytest.raises(InvalidSecretName):
        seeded.update(NAME, b"11" * 32, role=role)
    with pytest.raises(InvalidSecretName):
        seeded.delete(NAME, role=role)
    # Still there, unchanged, for its owner.
    assert seeded.get(NAME, role="delegation") == SEED


def test_the_agent_role_cannot_delete_it_by_record_id(seeded: SecretStore) -> None:
    record = seeded.describe(NAME, role="delegation")
    with pytest.raises(InvalidSecretName):
        seeded.delete_record_id(record.record_id)
    assert seeded.get(NAME, role="delegation") == SEED


@pytest.mark.parametrize("name", ["GITHUB_TOKEN", "LOP_PROVIDER_OPENROUTER_API_KEY"])
def test_the_delegation_role_is_refused_outside_its_prefix(store: SecretStore, name: str) -> None:
    with pytest.raises(InvalidSecretName):
        store.set(name, b"v", role="delegation")
    store.set(name, b"v", role="provider" if name.startswith("LOP_PROVIDER_") else "agent")
    with pytest.raises(InvalidSecretName):
        store.get(name, role="delegation")
    with pytest.raises(InvalidSecretName):
        store.delete(name, role="delegation")


def test_the_delegation_role_round_trips_and_deletes_its_own_row(seeded: SecretStore) -> None:
    assert seeded.describe(NAME, role="delegation").name == NAME
    seeded.delete(NAME, role="delegation")
    assert NAME not in [r.name for r in seeded.list()]


def test_lop_secret_get_refuses_the_row_and_prints_no_bytes(tmp_path: Path) -> None:
    """A real subprocess: "exits non-zero and prints nothing" is a process property."""

    from local_operator.secrets.access import open_store

    home = tmp_path / "home"
    config = tmp_path / "config"
    home.mkdir()
    config.mkdir()
    environment = {k: v for k, v in os.environ.items() if not k.startswith(("CMUX_", "LOP_"))}
    environment.update(
        HOME=str(home),
        LOCAL_OPERATOR_CONFIG_DIR=str(config),
        PYTHONPATH=str(Path(__file__).resolve().parents[3]),
        TERM="xterm-256color",
    )
    canary = "ab" * 32
    store = open_store(config, create=True)
    store.set(NAME, canary.encode(), role="delegation")
    result = subprocess.run(
        [sys.executable, "-m", "local_operator.cli", "secret", "get", NAME],
        capture_output=True,
        env=environment,
        timeout=180,
    )
    assert result.returncode != 0
    assert result.stdout == b""
    assert canary.encode() not in result.stderr
