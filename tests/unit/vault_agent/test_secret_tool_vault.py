"""H3 (the schema gate) and H4 (no value in the transcript) on the real ``secret`` tool."""

from __future__ import annotations

import logging

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
    assert "vault_resolve" not in tool.parameters["properties"]["op"]["enum"]


async def _call(context: ToolContext, **kwargs):
    return await execute_secret("call-v", {"op": "vault_resolve", **kwargs}, AbortSignal(), None, context)


@pytest.mark.asyncio
async def test_execution_refuses_before_any_http_when_the_gate_is_closed(config_dir, monkeypatch) -> None:
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
        answers=[(200, mint_answer()), (200, {"value": CANARY, "username": "deploy", "expires_at": None})]
    )
    # The tool calls resolve_item with its default factory; swap the default so
    # the REAL flow (gate, seed, grant, mint, sign, route 15) runs on the stub.
    monkeypatch.setitem(
        resolve.resolve_item.__kwdefaults__, "client_factory", lambda _root: make_client(transport)
    )
    store = VariableStore()
    caplog.set_level(logging.DEBUG)
    result = await _call(ToolContext(variables=store), vault_id=VAULT, item_id=ITEM, name="deploy key")

    assert not result.is_error, result.text
    assert store.credential_env() == {"DEPLOY_KEY": CANARY}  # byte for byte, not trimmed
    assert [c.source for c in store.list_credentials()] == ["vault"]
    core = CANARY.strip()
    # Not in the receipt, its details, the log, or anything the model replays.
    assert core not in result.text
    assert core not in repr(result.details)
    assert core not in caplog.text
    assert "deploy" not in result.text  # the username is not returned either
    assert "$DEPLOY_KEY" in result.text
    # Output that prints the value without its trailing newline is still scrubbed.
    assert core in store.redaction_values()
    assert core not in store.redact(f"echo {core}")


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
        "c", {"op": "retrieve", "name": DELEGATION_SECRET_PREFIX + "bep_x"}, AbortSignal(), None, ToolContext()
    )
    assert refused.is_error


def test_the_module_keeps_one_gating_convention() -> None:
    # The vault drop mirrors agent_tool._org_schema; no second mechanism.
    assert secret_tool._VAULT_OPS == ("vault_resolve",)
    assert secret_tool._VAULT_PROPERTIES == ("vault_id", "item_id")
