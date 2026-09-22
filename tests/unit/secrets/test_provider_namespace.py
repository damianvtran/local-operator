"""The provider/agent namespace boundary, and the ``migrate-env`` verb.

Two things are pinned here and they are one story: a reserved name prefix that
keeps provider-owned rows and agent secrets from ever colliding (design §3), and
the verb that moves the operator's plaintext ``credentials.env`` into the store
under those names (design §4 step 4, §5).

The boundary is enforced in the STORE, on both the write and the read path, so
these tests drive the store for that half — a check spelled in a CLI verb is one
an agent surface can walk around. ``migrate-env`` is driven through the real
subprocess, because the properties that matter for it (never creating the source
file, never printing a value) are properties of the PROCESS, and an in-process
call cannot show them.

Isolation follows ``tests/unit/secrets/conftest.py``: ``HOME`` and
``LOCAL_OPERATOR_CONFIG_DIR`` both point inside the per-test root, so nothing
here can reach the operator's live store or their real ``credentials.env``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from local_operator.credentials import CREDENTIALS_FILE_NAME, CredentialManager
from local_operator.secrets.errors import InvalidSecretName
from local_operator.secrets.store import (
    PROVIDER_SECRET_PREFIX,
    SecretStore,
    is_provider_secret_name,
    provider_secret_name,
    secret_class,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

#: The operator's real keys as of the design (design §5), split exactly as the
#: mapping table splits them. Kept as data rather than spelled inline so the
#: expectation and the assertion cannot drift apart.
PROVIDER_KEYS = ["RADIENT_API_KEY", "OPENROUTER_API_KEY"]
AGENT_KEYS = [
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "OSWORLD_EVAL_MODEL_API_KEY",
    "GOOGLE_ACCESS_TOKEN",
    "GOOGLE_REFRESH_TOKEN",
    "GOOGLE_TOKEN_EXPIRY_TIMESTAMP",
]


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    """A private root for one test's config dir and HOME.

    Named locally rather than reaching into the shared conftest so the
    subprocess-facing tests below can pass the SAME root to both a store read
    and a child process's ``LOCAL_OPERATOR_CONFIG_DIR``.
    """
    return tmp_path


# --- the primitive -----------------------------------------------------------


def test_provider_secret_name_prefixes_the_env_key() -> None:
    assert provider_secret_name("ANTHROPIC_API_KEY") == "LOP_PROVIDER_ANTHROPIC_API_KEY"
    assert provider_secret_name("RADIENT_API_KEY") == PROVIDER_SECRET_PREFIX + "RADIENT_API_KEY"


def test_provider_secret_name_refuses_an_empty_env_key() -> None:
    """A blank env key would name the bare prefix, which is a namespace, not a row."""
    with pytest.raises(InvalidSecretName):
        provider_secret_name("   ")


def test_is_provider_secret_name_is_the_prefix_test() -> None:
    assert is_provider_secret_name("LOP_PROVIDER_ANYTHING")
    assert not is_provider_secret_name("ANYTHING")
    # Whitespace is stripped, because the STORE normalises the name before it
    # checks — a predicate that disagreed with the store would be the one place
    # a padded name slipped through.
    assert is_provider_secret_name(f"  {PROVIDER_SECRET_PREFIX}X  ")


def test_secret_class_is_derived_from_the_name_not_stored() -> None:
    assert secret_class("LOP_PROVIDER_X") == "provider"
    assert secret_class("X") == "secret"


# --- the boundary, in the store ---------------------------------------------


def test_an_agent_cannot_create_a_provider_named_row(store: SecretStore) -> None:
    """The default role is the restrictive one, so the omission is refused too."""
    with pytest.raises(InvalidSecretName) as excinfo:
        store.set(provider_secret_name("RADIENT_API_KEY"), b"value")

    assert PROVIDER_SECRET_PREFIX in str(excinfo.value)
    # And nothing was written: a refusal that still created the row would leave
    # the name occupied by a record no provider reader could authenticate.
    assert store.list() == []


def test_a_provider_must_use_the_prefix(store: SecretStore) -> None:
    with pytest.raises(InvalidSecretName):
        store.set("RADIENT_API_KEY", b"value", role="provider")
    assert store.list() == []


def test_an_unknown_role_is_refused(store: SecretStore) -> None:
    with pytest.raises(InvalidSecretName):
        store.set("SOMETHING", b"value", role="wizard")


def test_a_provider_row_round_trips_under_its_prefixed_name(store: SecretStore) -> None:
    name = provider_secret_name("RADIENT_API_KEY")
    store.set(name, b"provider-key", role="provider", description="migrated")

    assert store.get(name, role="provider") == b"provider-key"
    records = store.list()
    assert [record.name for record in records] == [name]
    # The stored ``kind`` is untouched — provenance lives in the name (design §3),
    # and extending KINDS would have been a format change for every writer.
    assert records[0].kind == "string"


def test_an_agent_read_of_a_provider_row_is_refused(store: SecretStore) -> None:
    """The half that actually contains a value.

    The write guard stops an agent CREATING a ``LOP_PROVIDER_*`` record; the
    value worth having is the one that already exists, so an unrestricted read
    would hand it to exactly the caller the prefix exists to keep it from.
    """
    name = provider_secret_name("OPENROUTER_API_KEY")
    store.set(name, b"provider-key", role="provider")

    with pytest.raises(InvalidSecretName):
        store.get(name)

    assert store.get(name, role="provider") == b"provider-key"


def test_an_agent_update_of_a_provider_row_is_refused(store: SecretStore) -> None:
    """Otherwise a caller who knows a name could RE-SEAL it to a value of theirs."""
    name = provider_secret_name("RADIENT_API_KEY")
    store.set(name, b"real-key", role="provider")

    with pytest.raises(InvalidSecretName):
        store.update(name, b"attacker-value")

    assert store.get(name, role="provider") == b"real-key"


def test_the_two_namespaces_are_disjoint_so_shadowing_is_impossible(
    store: SecretStore,
) -> None:
    """Why a naming rule is enough: ``name_index`` is UNIQUE, and on disjoint sets.

    An agent value can never be served for a provider name and vice versa, so
    the failure the design calls out — a shadowed provider key authenticating
    against an attacker-supplied value — cannot be constructed at all.
    """
    store.set(provider_secret_name("RADIENT_API_KEY"), b"provider-value", role="provider")
    store.set("RADIENT_API_KEY", b"agent-value")

    assert store.get(provider_secret_name("RADIENT_API_KEY"), role="provider") == b"provider-value"
    assert store.get("RADIENT_API_KEY") == b"agent-value"


# --- the read-only credential source ----------------------------------------


def test_read_credentials_does_not_create_the_file(sandbox: Path) -> None:
    """``__init__`` would recreate the very file the migration is retiring."""
    root = sandbox / "config"
    # The config dir EXISTS but the plaintext store does not — the state after an
    # operator has migrated and deleted the file. Reading must leave it exactly
    # that way rather than resurrecting an empty ``credentials.env``.
    root.mkdir()

    assert CredentialManager.read_credentials(root) == {}
    assert not (root / CREDENTIALS_FILE_NAME).exists()
    assert list(root.iterdir()) == []


def test_read_credentials_returns_values_and_drops_blanks(sandbox: Path) -> None:
    root = sandbox / "config"
    root.mkdir()
    (root / CREDENTIALS_FILE_NAME).write_text("DEEPSEEK_API_KEY=sekrit\nEMPTY_KEY=\n")

    values = CredentialManager.read_credentials(root)
    assert {key: secret.get_secret_value() for key, secret in values.items()} == {
        "DEEPSEEK_API_KEY": "sekrit"
    }


# --- migrate-env, through the real process ----------------------------------


def _migrate_cli(sandbox: Path, *arguments: str) -> tuple[subprocess.CompletedProcess[bytes], Path]:
    """Run ``lop secret migrate-env`` with HOME and the config dir redirected.

    A real subprocess for the same reason ``test_cli.py`` uses one: "creates no
    file" and "prints no value" are process properties. ``CMUX_*`` is scrubbed
    per the team's QA rules — an inherited ``CMUX_WORKSPACE_ID`` has previously
    let headless tests rename the operator's real cmux workspaces.
    """
    home = sandbox / "home"
    config = sandbox / "config"
    home.mkdir(exist_ok=True)
    config.mkdir(exist_ok=True)
    environment = {key: value for key, value in os.environ.items() if not key.startswith("CMUX_")}
    environment.update(
        HOME=str(home),
        LOCAL_OPERATOR_CONFIG_DIR=str(config),
        PYTHONPATH=str(REPO_ROOT),
        TERM="xterm-256color",
    )
    environment.pop("NO_COLOR", None)
    result = subprocess.run(
        [sys.executable, "-m", "local_operator.cli", "secret", "migrate-env", *arguments],
        capture_output=True,
        env=environment,
        timeout=180,
    )
    return result, config


def _write_source(config: Path, keys: list[str]) -> None:
    (config / CREDENTIALS_FILE_NAME).write_text(
        "\n".join(f"{key}=value-of-{key}" for key in keys) + "\n"
    )


def _stored_names(config: Path) -> list[str]:
    """The store's names, read through the real store API under the same root."""
    from local_operator.secrets.access import open_store

    return sorted(record.name for record in open_store(config).list())


def test_migrate_env_dry_run_writes_nothing_and_prints_names_only(sandbox: Path) -> None:
    _, config = _migrate_cli(sandbox)
    _write_source(config, PROVIDER_KEYS + AGENT_KEYS)

    result, config = _migrate_cli(sandbox, "--dry-run")

    assert result.returncode == 0, result.stderr
    output = result.stdout.decode()
    # Names and destinations, one line per key.
    assert "RADIENT_API_KEY -> LOP_PROVIDER_RADIENT_API_KEY (provider) [dry-run]" in output
    assert "AWS_ACCESS_KEY_ID -> AWS_ACCESS_KEY_ID (secret) [dry-run]" in output
    # No value anywhere, on either stream.
    for stream in (result.stdout, result.stderr):
        assert b"value-of-" not in stream
    # And no store was created, nor the source touched.
    assert sorted(path.name for path in config.iterdir()) == [CREDENTIALS_FILE_NAME]
    assert (config / CREDENTIALS_FILE_NAME).read_text().count("value-of-") == 8


def test_migrate_env_stores_the_right_names_and_roles_and_is_idempotent(sandbox: Path) -> None:
    _, config = _migrate_cli(sandbox)
    _write_source(config, PROVIDER_KEYS + AGENT_KEYS)

    result, config = _migrate_cli(sandbox)
    assert result.returncode == 0, result.stderr
    assert result.stdout.decode().count("[stored]") == 8

    # Read the store back through the real store API, not through the CLI's own
    # report — a migration that claimed success while writing nothing would pass
    # a stdout-only assertion.
    names = _stored_names(config)
    assert names == sorted([provider_secret_name(key) for key in PROVIDER_KEYS] + AGENT_KEYS)

    # The source survives; the operator deletes it once they have verified.
    assert (config / CREDENTIALS_FILE_NAME).exists()

    # Re-running converges: nothing stored twice, and no value is overwritten.
    again, _ = _migrate_cli(sandbox)
    assert again.returncode == 0, again.stderr
    assert again.stdout.decode().count("[already present]") == 8
    assert again.stdout.decode().count("[stored]") == 0
    assert _stored_names(config) == names


def test_migrate_env_on_an_absent_file_creates_nothing(sandbox: Path) -> None:
    """The END state of the whole exercise must be a no-op, not a resurrection."""
    result, config = _migrate_cli(sandbox)

    assert result.returncode == 0, result.stderr
    assert b"no credentials to migrate" in result.stderr
    assert sorted(path.name for path in config.iterdir()) == []


# --- the readers and writers that replace the plaintext file -----------------


def test_provider_secret_value_reads_only_the_provider_namespace(sandbox: Path) -> None:
    """The reader resolves a provider row, and never an agent secret of that name.

    The namespace check is the point: a hostile or mistaken call asking for a key
    that exists only as an AGENT secret must not be served it, because the whole
    reserved-prefix design exists to keep provider keys from being readable by
    the agent surfaces and vice versa.
    """
    from local_operator.providers.registry import provider_secret_value
    from local_operator.secrets import access

    store = access.open_store(sandbox, create=True)
    store.set(provider_secret_name("OPENROUTER_API_KEY"), b"provider-value", role="provider")
    # An agent secret under the same env-key spelling, which the provider reader
    # must NOT find: it only ever looks under LOP_PROVIDER_.
    store.set("OPENROUTER_API_KEY", b"agent-value", role="agent")

    assert provider_secret_value("OPENROUTER_API_KEY", base=sandbox) == "provider-value"


def test_store_provider_key_writes_and_remove_provider_key_deletes(sandbox: Path) -> None:
    from local_operator.providers.registry import (
        provider_secret_value,
        remove_provider_key,
        store_provider_key,
    )

    store_provider_key("ANTHROPIC_API_KEY", "first", base=sandbox)
    assert provider_secret_value("ANTHROPIC_API_KEY", base=sandbox) == "first"
    # A re-run updates in place rather than failing on the existing name.
    store_provider_key("ANTHROPIC_API_KEY", "second", base=sandbox)
    assert provider_secret_value("ANTHROPIC_API_KEY", base=sandbox) == "second"
    assert remove_provider_key("ANTHROPIC_API_KEY", base=sandbox) is True
    assert provider_secret_value("ANTHROPIC_API_KEY", base=sandbox) is None
    # Deleting an absent row is the desired end state, not an error.
    assert remove_provider_key("ANTHROPIC_API_KEY", base=sandbox) is False


def test_provider_env_key_prefers_the_store_over_the_environment(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Store-first is the whole resolution order; a stale export must lose."""
    from local_operator.providers.registry import provider_env_key, store_provider_key

    monkeypatch.setenv("OPENROUTER_API_KEY", "from-the-environment")
    assert provider_env_key("openrouter") == "from-the-environment"

    store_provider_key("OPENROUTER_API_KEY", "from-the-store", base=sandbox)
    # The store row wins even with the environment variable still set.
    assert provider_env_key("openrouter", base=sandbox) == "from-the-store"


def test_provider_env_key_falls_back_to_the_legacy_file_during_the_transition(
    sandbox: Path,
) -> None:
    """Mid-migration installs keep resolving: the file leg is still read LAST.

    It is read, not created — the non-creating reader is what lets a host that has
    never run ``lop secret migrate-env`` keep working without the read resurrecting
    a file it had deleted.
    """
    from local_operator.providers.registry import provider_env_key

    (sandbox / CREDENTIALS_FILE_NAME).write_text("OPENROUTER_API_KEY=from-the-file\n")
    assert provider_env_key("openrouter", base=sandbox) == "from-the-file"


def test_no_plaintext_writer_outside_the_retired_manager() -> None:
    """The plaintext file must have NO writers left outside ``credentials.py``.

    ``credentials.py`` itself keeps ``set_credential``/``write_to_file`` until PR2
    deletes the module, but every OTHER module in the package must have stopped
    writing the file: a single surviving call re-creates the greppable plaintext
    copy the whole consolidation exists to retire. ``prompt_for_credential`` is
    deliberately NOT in the set — it is a UI prompt whose WRITE now goes to the
    store, so calling it is fine; it is the file-write primitives that must have
    no callers. Read at the AST level so a commented-out or string-literal mention
    is not a false hit.
    """
    import ast
    from pathlib import Path

    package = Path(__file__).resolve().parents[3] / "local_operator"
    writers = {"set_credential", "write_to_file"}
    offenders: list[str] = []
    for path in sorted(package.rglob("*.py")):
        if path.name == "credentials.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in writers:
                offenders.append(f"{path.relative_to(package)}:{node.lineno} .{node.attr}")
    assert offenders == [], offenders
