"""Real encrypted-store and child-stderr regressions, synthetic values only."""

import asyncio
import logging
import sys
from types import SimpleNamespace

import pytest

from local_operator.credentials import CredentialManager
from local_operator.mcp.config import MCPHttpServerConfig, MCPStdioServerConfig
from local_operator.mcp.credentials import MCPCredentials, store_credentials
from local_operator.mcp.manager import McpManager
from local_operator.mcp.secret_refs import (
    McpSecretRefError,
    public_secret_refs,
    resolve_config_secrets,
)
from local_operator.secrets import access
from local_operator.secrets.errors import SecretCorrupt
from local_operator.variables import VariableStore


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    for key in list(__import__("os").environ):
        if key.startswith("CMUX_"):
            monkeypatch.delenv(key)
    base = tmp_path / "config"
    base.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(base))
    # Exercise the real encrypted store without launching a persistent broker
    # for every test. Broker denial is independently tested at the access seam.
    monkeypatch.setattr("local_operator.secrets.client.ensure_broker", lambda *a, **kw: False)
    return base


def test_metadata_is_pristine_deduplicated_and_cold():
    cfg = MCPHttpServerConfig(
        url="https://example.invalid/mcp",
        headers={
            "Authorization": "Bearer ${HUBSPOT_TOKEN}",
            "Other": "${HUBSPOT_TOKEN}",
            "Literal": "$${NOT_A_KEY}",
            "Region": "us-east-1",
        },
    )
    assert public_secret_refs(cfg) == [
        {
            "id": "HUBSPOT_TOKEN",
            "bindings": [
                {"field": "headers", "key": "Authorization"},
                {"field": "headers", "key": "Other"},
            ],
        }
    ]


def test_encrypted_precedence_and_readonly_legacy(isolated):
    legacy = CredentialManager(isolated)
    legacy.set_credential("TOKEN", "legacy-synthetic")
    before = (isolated / "credentials.env").read_bytes()
    cfg = MCPStdioServerConfig(command="unused", env={"API_KEY": "${TOKEN}"})
    assert resolve_config_secrets("test", cfg).env["API_KEY"] == "legacy-synthetic"
    access.open_store(isolated, create=True).set("TOKEN", b"encrypted-synthetic")
    assert resolve_config_secrets("test", cfg).env["API_KEY"] == "encrypted-synthetic"
    assert (isolated / "credentials.env").read_bytes() == before
    assert cfg.env["API_KEY"] == "${TOKEN}"


@pytest.mark.parametrize("failure", [PermissionError("denied"), SecretCorrupt("corrupt"), b""])
def test_encrypted_failure_never_downgrades(isolated, monkeypatch, failure):
    CredentialManager(isolated).set_credential("TOKEN", "legacy-synthetic")
    access.open_store(isolated, create=True).set("TOKEN", b"encrypted-synthetic")

    def retrieve(*args):
        if isinstance(failure, Exception):
            raise failure
        return failure

    monkeypatch.setattr(access, "retrieve_secret", retrieve)
    with pytest.raises(McpSecretRefError, match="unavailable"):
        resolve_config_secrets(
            "test", MCPStdioServerConfig(command="unused", env={"X": "${TOKEN}"})
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("encrypted", [False, True])
async def test_real_split_child_stderr_is_scrubbed(isolated, encrypted, caplog):
    sentinel = "synthetic-split-canary-91730"
    if encrypted:
        access.open_store(isolated, create=True).set("TOKEN", sentinel.encode())
    else:
        CredentialManager(isolated).set_credential("TOKEN", sentinel)
    cfg = MCPStdioServerConfig(
        command=sys.executable,
        timeout=3000,
        env={"X": "${TOKEN}"},
        args=[
            "-c",
            'import os,sys,time; x=os.environ["X"]; sys.stderr.write("a"*8190+x[:7]); '
            'sys.stderr.flush(); time.sleep(.02); sys.stderr.write(x[7:]+"\\n"); sys.exit(1)',
        ],
    )
    variables = VariableStore(env={})
    manager = McpManager(
        isolated, secret_base=isolated, register_secret=variables.register_redaction
    )
    caplog.set_level(logging.DEBUG, logger="local_operator.mcp")
    with pytest.raises(Exception) as caught:
        await asyncio.wait_for(manager._connect_server("canary", cfg), 15)
    await manager.disconnect_all()
    assert sentinel not in str(caught.value)
    assert sentinel not in caplog.text
    assert sentinel[:7] not in caplog.text
    assert not variables.credential_env()
    assert variables.redact(sentinel) == "[redacted]"


@pytest.mark.asyncio
async def test_metadata_probe_never_creates_the_legacy_file(isolated):
    """A cold metadata read must not write the store it is describing.

    ``CredentialManager.__init__`` creates ``credentials.env``, so a probe that
    constructs it turns a read into a write of the plaintext file this change
    promises to leave alone. Reproduced by the assembled desktop probe.
    """
    from local_operator.mcp.credentials import credential_source

    assert credential_source("TOKEN", isolated) == "missing"
    assert not (isolated / "credentials.env").exists()
    assert not (isolated / "secrets" / "store.db").exists()


@pytest.mark.asyncio
async def test_store_validates_all_ids_and_confirms_replacement(isolated, monkeypatch):
    cfg = MCPHttpServerConfig(
        url="https://example.invalid", headers={"Authorization": "Bearer ${TOKEN}"}
    )
    monkeypatch.setattr(
        "local_operator.mcp.credentials.load_all_mcp_configs", lambda cwd: ({"api": cfg}, {})
    )
    owner = SimpleNamespace(
        session_id="synthetic-owner",
        config_dir=isolated,
        variables=VariableStore(env={}),
        mcp_manager=McpManager(isolated),
    )

    async def save(values, confirmed=()):
        return await store_credentials(
            owner, MCPCredentials(name="api", values=values, confirmed_replace=list(confirmed))
        )

    assert (await save({"TOKEN": "one", "Authorization": "wrong"}))["code"] == "invalid_target"
    assert not (isolated / "secrets" / "store.db").exists()
    assert (await save({"TOKEN": "one"}))["code"] == "saved"
    assert (await save({"TOKEN": "two"}))["code"] == "replace_confirmation_required"
    assert access.open_store(isolated).get("TOKEN") == b"one"
    assert (await save({"TOKEN": "two"}, ["TOKEN"]))["code"] == "saved"
    assert access.open_store(isolated).get("TOKEN") == b"two"
    assert not owner.variables.credential_env()
    assert not (isolated / "credentials.env").exists()
