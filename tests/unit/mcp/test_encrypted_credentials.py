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


def test_the_encrypted_store_is_the_only_leg_read(isolated):
    """The plaintext fallback is GONE: only the encrypted store resolves a ref.

    The legacy ``credentials.env`` has no writers left, so a value that lives
    only there must NOT be served — it reads as missing until ``lop secret
    migrate-env`` moves it, which is the honest state of an install that has not
    migrated rather than a silent downgrade to a store nothing maintains.
    """
    CredentialManager(isolated).set_credential("TOKEN", "legacy-synthetic")
    cfg = MCPStdioServerConfig(command="unused", env={"API_KEY": "${TOKEN}"})
    with pytest.raises(McpSecretRefError):
        resolve_config_secrets("test", cfg)
    access.open_store(isolated, create=True).set("TOKEN", b"encrypted-synthetic")
    assert resolve_config_secrets("test", cfg).env["API_KEY"] == "encrypted-synthetic"
    # The config keeps the reference it was written with.
    assert cfg.env["API_KEY"] == "${TOKEN}"


@pytest.mark.parametrize("failure", [PermissionError("denied"), SecretCorrupt("corrupt"), b""])
def test_encrypted_failure_never_downgrades(isolated, monkeypatch, failure):
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
async def test_real_split_child_stderr_is_scrubbed(isolated, caplog):
    sentinel = "synthetic-split-canary-91730"
    access.open_store(isolated, create=True).set("TOKEN", sentinel.encode())
    # The split has to land INSIDE ``STDERR_LINE_LIMIT`` (2000): everything a
    # child writes past it is truncated before any sink sees it, so a probe that
    # splits at 8190 passes with the whole scrub removed — it was measuring the
    # truncation, not the redaction (agent review R-2). 37 characters of banner
    # leave the credential's own first fragment inside the retained line.
    cfg = MCPStdioServerConfig(
        command=sys.executable,
        timeout=3000,
        env={"X": "${TOKEN}"},
        args=[
            "-c",
            'import os,sys,time; x=os.environ["X"]; sys.stderr.write("a"*37+x[:7]); '
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
    # The child's own reason reached the raised error, so the assertion below is
    # about the QUOTED TAIL and not about a transport error that happens to be
    # secret-free: the banner is the only place those 37 characters exist.
    assert "a" * 37 in str(caught.value), str(caught.value)
    assert sentinel not in str(caught.value)
    assert sentinel not in _chain_text(caught.value)
    assert sentinel not in caplog.text
    assert sentinel[:7] not in caplog.text
    assert not variables.credential_env()
    assert variables.redact(sentinel) == "[redacted]"


def _chain_text(exc: BaseException) -> str:
    """``str()`` of an exception and everything it is chained to.

    ``str()`` deliberately, and not ``repr()``/``args``: an ``MCPError`` keeps
    its text in ``error.message`` and builds ``__str__`` from that field, so a
    walk over ``repr`` — or over ``args``, which the scrub DOES rewrite — reads a
    clean tuple while ``str()``, the string every sink renders, still carries the
    credential. That gap is why the earlier guard could not see this regression
    (agent review R-2).
    """
    parts: list[str] = []
    seen: set[int] = set()
    node: BaseException | None = exc
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        parts.append(f"{type(node).__name__}: {node}")
        node = node.__cause__ or node.__context__
    return " | ".join(parts)


#: A stdio child that answers ``initialize`` with a JSON-RPC ERROR whose message
#: echoes the credential it was handed. That is the shape a badly behaved
#: third-party server has, and the one an SDK ``MCPError`` keeps in
#: ``error.message`` — so it is the shape a sink publishes verbatim unless the
#: value was registered for scrubbing. ``tail`` also writes one line to stderr,
#: which is what makes ``explain()`` quote a tail; without it ``explain()``
#: hands the exception itself back, which is the case for every HTTP transport
#: (they spawn nothing) and for a stdio child that stays quiet.
_ECHO_CHILD = (
    "import json,os,sys;"
    "tail = sys.argv[1] == 'tail';"
    "sys.stderr.write('booting canary\\n') if tail else None;"
    "sys.stderr.flush();"
    "req = json.loads(sys.stdin.readline());"
    "sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': req['id'], 'error':"
    " {'code': -32000, 'message': 'rejected credential ' + os.environ['X']}}) + '\\n');"
    "sys.stdout.flush();"
    "sys.exit(1)"
)


@pytest.mark.asyncio
@pytest.mark.parametrize("tail", [True, False], ids=["with-tail", "no-tail"])
async def test_a_server_echoed_credential_never_reaches_a_sink(isolated, tail):
    """A credential echoed in a JSON-RPC error is scrubbed on all three paths.

    The three paths are the finding (agent review R-1): the raised exception,
    the CAUSE it deliberately keeps as evidence, and the startup-failure text
    that becomes ``McpStartupOutcome.failures`` — which the toast, the transcript
    notice, ``/mcp`` and the desktop projection all render. Both arms are
    exercised, because they fail differently: with a stderr tail the raised error
    is rebuilt from scrubbed text and only the CHAIN leaked; without one the raw
    exception went out, and that is every HTTP connect.

    Fails with redaction removed, in every arm — that is what makes it a guard.
    """
    sentinel = "synthetic-echo-canary-40711"
    access.open_store(isolated, create=True).set("TOKEN", sentinel.encode())
    cfg = MCPStdioServerConfig(
        command=sys.executable,
        timeout=3000,
        env={"X": "${TOKEN}"},
        args=["-c", _ECHO_CHILD, "tail" if tail else "quiet"],
    )
    variables = VariableStore(env={})
    manager = McpManager(
        isolated, secret_base=isolated, register_secret=variables.register_redaction
    )
    manager._configs = {"canary": cfg}
    manager._sources = {"canary": "test"}
    with pytest.raises(Exception) as caught:
        await asyncio.wait_for(manager._connect_server("canary", cfg), 15)
    assert sentinel not in str(caught.value)
    assert sentinel not in _chain_text(caught.value)
    # The published surface: one round is what fills `startup_failures()`, which
    # `session_factory` turns into the outcome the front ends read. Wait for it to
    # SETTLE rather than assuming which arm the gate took: a spawn that takes
    # longer than the 250 ms gate is deferred, and its failure is then recorded by
    # the continuation — the same single write path, just later.
    await manager._connect_round({"canary": cfg}, {"canary": "test"})
    deadline = asyncio.get_running_loop().time() + 10
    while manager.startup_settling() and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.05)
    assert manager.startup_settling() is False, "the startup round never settled"
    failures = manager.startup_failures()
    await manager.disconnect_all()
    # Not vacuous: the SERVER's own sentence is in there, with only the value
    # replaced — so the round recorded this failure and not some other one.
    assert failures["canary"].startswith("rejected credential [redacted]"), failures
    assert sentinel not in str(failures)
    assert variables.redact(sentinel) == "[redacted]"
    assert not variables.credential_env()


@pytest.mark.asyncio
async def test_metadata_probe_never_creates_the_legacy_file(isolated):
    """A cold metadata read must not write the store it is describing.

    A probe that constructs a store (``CredentialManager.__init__`` writes
    ``credentials.env``; opening the encrypted store initialises it) turns a read
    into a write of state the caller never asked for. Reproduced by the assembled
    desktop probe.
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
