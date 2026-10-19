"""Adoption of literal ``mcp.json`` credentials: naming, rewrite, conflicts, safety.

The measured defect this file is written against: a head node held
``GOOGLE_OAUTH_CLIENT_ID``/``GOOGLE_OAUTH_CLIENT_SECRET`` (env) and an
``Authorization: Bearer …`` header as LITERALS. The definitions channel carried
only the references it could see, so the peer's mirror said
``${GOOGLE_OAUTH_CLIENT_ID}`` and no store anywhere held that name. Adoption is
the source-side fix: store row first, exact ``${NAME}`` reference second.

Every cell drives the real encrypted store under a private root and the real
user-scope document — no stubs for either writer. Values are sentinels, so a
leak assertion is meaningful.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

import pytest

from local_operator.mcp.credential_sources import (
    CredentialAdoptionError,
    adopt_literal_credentials,
    literal_credential_keys,
)

SENTINEL = "SENTINEL-ADOPT-VALUE"


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    """A private config root for one test (the network suite's fixture name)."""
    return tmp_path


@pytest.fixture(autouse=True)
def _no_broker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never reach a broker another test (or the operator) may be running."""
    monkeypatch.setattr("local_operator.secrets.client.ensure_broker", lambda *a, **k: False)


def _write(root: Path, servers: dict[str, Any]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "mcp.json"
    path.write_text(json.dumps({"mcpServers": servers}, indent=2) + "\n", encoding="utf-8")
    return path


def _servers(root: Path) -> dict[str, Any]:
    return json.loads((root / "mcp.json").read_text(encoding="utf-8"))["mcpServers"]


def _seed_store(root: Path, values: dict[str, str]) -> None:
    from local_operator.secrets import access

    store = access.open_store(root, create=True)
    for name, value in values.items():
        store.set(name, value.encode("utf-8"))


def _store_value(root: Path, name: str) -> str:
    from local_operator.secrets import access

    store = access.open_store(root)
    return store.get(name).decode("utf-8")


def _sentinel_free(value: object) -> None:
    text = json.dumps(value, default=str) + repr(value)
    assert SENTINEL not in text


# ---------------------------------------------------------------------------
# Naming and the rewrite
# ---------------------------------------------------------------------------


def test_an_env_key_that_is_free_becomes_its_own_secret_name(root: Path) -> None:
    _write(
        root,
        {
            "google-workspace": {
                "type": "stdio",
                "command": "npx",
                "env": {
                    "GOOGLE_OAUTH_CLIENT_ID": "client-id-sentinel",
                    "GOOGLE_OAUTH_CLIENT_SECRET": "client-secret-sentinel",
                },
            }
        },
    )

    receipt = adopt_literal_credentials(root)

    assert [row["secret"] for row in receipt["adopted"]] == [
        "GOOGLE_OAUTH_CLIENT_ID",
        "GOOGLE_OAUTH_CLIENT_SECRET",
    ]
    assert _servers(root)["google-workspace"]["env"] == {
        "GOOGLE_OAUTH_CLIENT_ID": "${GOOGLE_OAUTH_CLIENT_ID}",
        "GOOGLE_OAUTH_CLIENT_SECRET": "${GOOGLE_OAUTH_CLIENT_SECRET}",
    }
    assert _store_value(root, "GOOGLE_OAUTH_CLIENT_ID") == "client-id-sentinel"
    assert _store_value(root, "GOOGLE_OAUTH_CLIENT_SECRET") == "client-secret-sentinel"


def test_a_header_always_uses_the_prefixed_generated_name_and_keeps_the_whole_value(
    root: Path,
) -> None:
    """``Authorization: Bearer …`` must become EXACTLY ``${…}`` — scheme included.

    ``mcpdefs.environment_state`` only classifies an exact single reference, so
    leaving ``Bearer`` outside would keep the literal off the wire and never
    authenticate the header: the silent half-fix this cell exists to refuse.
    """
    _write(
        root,
        {
            "slack": {
                "type": "http",
                "url": "https://slack.test/mcp",
                "headers": {"Authorization": f"Bearer {SENTINEL}"},
            }
        },
    )

    receipt = adopt_literal_credentials(root)

    assert receipt["adopted"] == [
        {
            "server": "slack",
            "field": "headers",
            "key": "Authorization",
            "secret": "MCP_SLACK_AUTHORIZATION",
        }
    ]
    assert _servers(root)["slack"]["headers"] == {"Authorization": "${MCP_SLACK_AUTHORIZATION}"}
    # The WHOLE value, scheme included, is what the reference resolves to.
    assert _store_value(root, "MCP_SLACK_AUTHORIZATION") == f"Bearer {SENTINEL}"
    # Every other field survives untouched.
    assert _servers(root)["slack"]["type"] == "http"
    assert _servers(root)["slack"]["url"] == "https://slack.test/mcp"


def test_a_second_run_adopts_nothing_and_rewrites_nothing(root: Path) -> None:
    _write(root, {"gl": {"type": "stdio", "command": "x", "env": {"TOKEN": "v"}}})
    adopt_literal_credentials(root)
    before = (root / "mcp.json").read_bytes()

    receipt = adopt_literal_credentials(root)

    assert receipt["adopted"] == []
    assert receipt["reused"] == []
    assert receipt["conflicts"] == []
    assert (root / "mcp.json").read_bytes() == before


def test_an_existing_identical_store_row_is_reused_not_rewritten(root: Path) -> None:
    _seed_store(root, {"TOKEN": "same"})
    _write(root, {"gl": {"type": "stdio", "command": "x", "env": {"TOKEN": "same"}}})

    receipt = adopt_literal_credentials(root)

    assert receipt["adopted"] == []
    assert receipt["reused"] == [
        {"server": "gl", "field": "env", "key": "TOKEN", "secret": "TOKEN"}
    ]
    assert _servers(root)["gl"]["env"] == {"TOKEN": "${TOKEN}"}


def test_a_dashed_env_key_falls_back_to_a_reference_safe_generated_name(root: Path) -> None:
    """``MY-TOKEN`` is a legal env var but not a legal ``${NAME}``.

    Using it verbatim would rewrite the config to ``${MY-TOKEN}``, which the
    reference parser does not publish — the same invisible-literal failure this
    module exists to fix, wearing a rewrite.
    """
    _write(root, {"gl": {"type": "stdio", "command": "x", "env": {"MY-TOKEN": "v"}}})

    receipt = adopt_literal_credentials(root)

    assert [row["secret"] for row in receipt["adopted"]] == ["MCP_GL_MY_TOKEN"]
    assert _servers(root)["gl"]["env"] == {"MY-TOKEN": "${MCP_GL_MY_TOKEN}"}


def test_an_occupied_env_key_falls_back_to_the_generated_name(root: Path) -> None:
    _seed_store(root, {"TOKEN": "someone-else"})
    _write(root, {"gl": {"type": "stdio", "command": "x", "env": {"TOKEN": "mine"}}})

    receipt = adopt_literal_credentials(root)

    assert receipt["adopted"] == [
        {"server": "gl", "field": "env", "key": "TOKEN", "secret": "MCP_GL_TOKEN"}
    ]
    assert _servers(root)["gl"]["env"] == {"TOKEN": "${MCP_GL_TOKEN}"}
    assert _store_value(root, "TOKEN") == "someone-else"
    assert _store_value(root, "MCP_GL_TOKEN") == "mine"


def test_a_generated_name_collision_is_refused_and_never_overwritten(root: Path) -> None:
    _seed_store(root, {"MCP_SLACK_AUTHORIZATION": "someone-else"})
    _write(
        root,
        {
            "slack": {
                "type": "http",
                "url": "https://slack.test/mcp",
                "headers": {"Authorization": f"Bearer {SENTINEL}"},
            }
        },
    )
    before = (root / "mcp.json").read_bytes()

    receipt = adopt_literal_credentials(root)

    assert receipt["adopted"] == []
    assert receipt["conflicts"] == [
        {
            "server": "slack",
            "field": "headers",
            "key": "Authorization",
            "secret": "MCP_SLACK_AUTHORIZATION",
            "reason": "name_holds_a_different_value",
        }
    ]
    assert (root / "mcp.json").read_bytes() == before
    assert _store_value(root, "MCP_SLACK_AUTHORIZATION") == "someone-else"


# ---------------------------------------------------------------------------
# Boundaries
# ---------------------------------------------------------------------------


def test_only_the_user_scope_document_is_read(root: Path) -> None:
    _write(root, {"mine": {"type": "stdio", "command": "x", "env": {"MINE": "a"}}})
    project = root / ".local-operator" / "mcp.json"
    project.parent.mkdir(parents=True)
    project.write_text(
        json.dumps({"mcpServers": {"p": {"type": "stdio", "env": {"P": "project"}}}}),
        encoding="utf-8",
    )
    foreign = root / ".mcp.json"
    foreign.write_text(
        json.dumps({"mcpServers": {"f": {"type": "stdio", "env": {"F": "foreign"}}}}),
        encoding="utf-8",
    )
    project_before, foreign_before = project.read_bytes(), foreign.read_bytes()

    adopt_literal_credentials(root)

    assert project.read_bytes() == project_before
    assert foreign.read_bytes() == foreign_before
    assert _servers(root)["mine"]["env"] == {"MINE": "${MINE}"}


def test_a_store_failure_leaves_the_document_untouched(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(root, {"gl": {"type": "stdio", "command": "x", "env": {"TOKEN": SENTINEL}}})
    before = (root / "mcp.json").read_bytes()

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise OSError("the store is not available")

    monkeypatch.setattr("local_operator.secrets.access.open_store", _boom)

    with pytest.raises(CredentialAdoptionError) as caught:
        adopt_literal_credentials(root)

    assert (root / "mcp.json").read_bytes() == before
    assert SENTINEL not in str(caught.value)


def test_the_receipt_and_the_rewritten_file_never_carry_a_value(root: Path) -> None:
    _write(
        root,
        {
            "env-server": {"type": "stdio", "command": "x", "env": {"E": SENTINEL}},
            "header-server": {
                "type": "http",
                "url": "https://h.test/mcp",
                "headers": {"Authorization": f"Bearer {SENTINEL}"},
            },
        },
    )

    receipt = adopt_literal_credentials(root)

    _sentinel_free(receipt)
    assert SENTINEL not in (root / "mcp.json").read_text(encoding="utf-8")


def test_the_document_ends_at_mode_0600(root: Path) -> None:
    path = _write(root, {"gl": {"type": "stdio", "command": "x", "env": {"TOKEN": "v"}}})
    path.chmod(0o600)

    adopt_literal_credentials(root)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_dry_run_writes_neither_the_store_nor_the_document(root: Path) -> None:
    path = _write(root, {"gl": {"type": "stdio", "command": "x", "env": {"TOKEN": "v"}}})
    before = path.read_bytes()

    receipt = adopt_literal_credentials(root, dry_run=True)

    assert receipt["adopted"] == [
        {"server": "gl", "field": "env", "key": "TOKEN", "secret": "TOKEN"}
    ]
    assert path.read_bytes() == before
    assert not (root / "secrets").exists()


def test_literal_credential_keys_reports_bindings_without_values(root: Path) -> None:
    _write(
        root,
        {
            "gl": {
                "type": "stdio",
                "command": "x",
                "env": {"TOKEN": SENTINEL, "REF": "${ALREADY}"},
                "headers": {"X-Empty": ""},
            }
        },
    )

    rows = literal_credential_keys(root)

    assert rows == [
        {"server": "gl", "field": "env", "key": "TOKEN", "has_value": True},
        {"server": "gl", "field": "headers", "key": "X-Empty", "has_value": False},
    ]
    _sentinel_free(rows)


def test_a_missing_document_is_an_empty_noop(root: Path) -> None:
    assert adopt_literal_credentials(root) == {
        "adopted": [],
        "reused": [],
        "conflicts": [],
        "skipped": [],
    }
    assert literal_credential_keys(root) == []
