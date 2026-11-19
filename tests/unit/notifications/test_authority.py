"""Explicit notification policy must not be inferred from specialist identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from local_operator.cli import sessions_notifications_command
from local_operator.harness.types import (
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
)
from local_operator.notifications.authority import GRANT_NAME, granted, set_grant
from local_operator.notifications.store import NotificationStore
from local_operator.resume import ORIGIN_AGENT_WORKSTREAM, mark_session_origin
from tests.unit.session.test_session import ScriptedStream, make_session


def _workstream(directory: Path) -> bytes:
    mark_session_origin(directory, ORIGIN_AGENT_WORKSTREAM, opened_by={"session": "parent"})
    return (directory / "origin.json").read_bytes()


def test_grant_revoke_preserves_provenance_and_binds_the_session(tmp_path: Path) -> None:
    directory = tmp_path / "synthetic-specialist"
    original = _workstream(directory)
    assert not granted(directory)
    decision = set_grant(directory, allow=True, reason="Operator requested standalone updates")
    assert decision["source"] == "operator-cli"
    assert granted(directory)
    assert (directory / "origin.json").read_bytes() == original
    copied = tmp_path / "other-session"
    _workstream(copied)
    (copied / GRANT_NAME).write_bytes((directory / GRANT_NAME).read_bytes())
    assert not granted(copied)
    set_grant(directory, allow=False, reason="Operator revoked updates")
    assert not granted(directory)
    assert (directory / "origin.json").read_bytes() == original


@pytest.mark.parametrize("case", ["subagent", "agent-shell", "remote", "blank-reason"])
def test_invalid_grants_are_refused(tmp_path: Path, case: str) -> None:
    directory = tmp_path / "synthetic-specialist"
    _workstream(directory)
    if case in {"subagent", "agent-shell"}:
        mark_session_origin(directory, case)
    if case == "remote":
        (directory / "mesh.json").write_text("{}")
    with pytest.raises(ValueError):
        set_grant(
            directory, allow=True, reason="" if case == "blank-reason" else "operator request"
        )
    assert not (directory / GRANT_NAME).exists()


@pytest.mark.parametrize(
    "change", [{"version": 2}, {"granted": "true"}, {"source": "profile"}, {"reason": ""}]
)
def test_malformed_grant_fails_closed(tmp_path: Path, change: dict) -> None:
    _workstream(tmp_path)
    grant = set_grant(tmp_path, allow=True, reason="operator request")
    grant.update(change)
    (tmp_path / GRANT_NAME).write_text(json.dumps(grant))
    assert not granted(tmp_path)


def test_cli_resolves_and_records_explicit_policy(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    directory = tmp_path / "sessions" / "abcd1234abcd"
    original = _workstream(directory)
    (directory / "transcript.jsonl").write_text("{}\n")
    args = argparse.Namespace(session="abcd", grant=True, revoke=False, reason="operator request")
    assert sessions_notifications_command(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["granted"] is True
    assert result["decision"]["reason"] == "operator request"
    assert (directory / "origin.json").read_bytes() == original
    args.grant, args.revoke = False, True
    assert sessions_notifications_command(args) == 0
    assert json.loads(capsys.readouterr().out)["granted"] is False


@pytest.mark.asyncio
async def test_granted_specialist_can_notify_and_revoke_closes_stale_tool(tmp_path: Path) -> None:
    session = make_session(tmp_path, ScriptedStream([]))
    session._config_dir = tmp_path / "config"
    _workstream(session.transcript.directory)
    try:
        # Self registration/proactivity/attachment alone cannot promote it.
        session._agent_id = "synthetic-specialist"
        assert session._notify_tool_callable() is None
        set_grant(session.transcript.directory, allow=True, reason="operator request")
        door = session._notify_tool_callable()
        assert door is not None
        await door(
            title="Specialist memo ready",
            body="memo",
            priority="normal",
            key="memo",
            links=[],
            anchor_id="a",
        )
        items, _ = NotificationStore(session._config_dir).list_items()
        assert len(items) == 1
        set_grant(session.transcript.directory, allow=False, reason="operator revoked")
        assert session._notify_tool_callable() is None
        result = await door(
            title="Another specialist memo",
            body="memo",
            priority="normal",
            key="other",
            links=[],
            anchor_id="b",
        )
        assert "unavailable" in result
        assert len(NotificationStore(session._config_dir).list_items()[0]) == 1
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_cli_grant_reaches_next_live_provider_request_and_executes(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import local_operator.cli as cli

    monkeypatch.setattr(cli, "config_dir", lambda: tmp_path)
    root = tmp_path / "sessions"
    directory = root / "sess"
    _workstream(directory)
    stream = ScriptedStream(
        [
            [StreamTextDelta(delta="not granted"), StreamEndEvent(stop_reason="stop")],
            [
                StreamToolCallDelta(
                    index=0,
                    id="notify-memo",
                    name="notify",
                    argument_delta=json.dumps(
                        {"title": "Memo ready", "body": "Synthetic memo", "key": "memo"}
                    ),
                ),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="memo published"), StreamEndEvent(stop_reason="stop")],
            [StreamTextDelta(delta="revoked"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    session = make_session(root, stream)
    session._config_dir = tmp_path
    args = argparse.Namespace(session="sess", grant=True, revoke=False, reason="operator request")
    try:
        await session.prompt("Before authorization")
        assert not any(tool.name == "notify" for tool in stream.requests[0].tools)
        assert sessions_notifications_command(args) == 0
        assert json.loads(capsys.readouterr().out)["granted"] is True
        # A grant does not rewrite an already-published turn's cache prefix.
        assert not any(tool.name == "notify" for tool in session._wire_tools())
        await session.prompt("Publish the memo")
        assert any(tool.name == "notify" for tool in stream.requests[1].tools)
        items, _ = NotificationStore(tmp_path).list_items()
        assert len(items) == 1 and items[0]["key"] == "memo"
        assert session._run_notified is True
        cached_tool = next(tool for tool in session._tools if tool.name == "notify")
        cached_context = session._build_tool_context()
        args.grant, args.revoke = False, True
        assert sessions_notifications_command(args) == 0
        assert json.loads(capsys.readouterr().out)["granted"] is False
        assert any(tool.name == "notify" for tool in session._wire_tools())
        # An already-resolved tool/context must deny immediately, even before
        # the next turn reconciles the provider-visible inventory.
        denied = await cached_tool.execute(
            "stale-notify",
            {"title": "Stale", "body": "Must not publish", "key": "stale"},
            context=cached_context,
        )
        assert "unavailable" in str(denied)
        assert len(NotificationStore(tmp_path).list_items()[0]) == 1
        await session.prompt("After revocation")
        assert not any(tool.name == "notify" for tool in stream.requests[3].tools)
    finally:
        await session.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("declared", [[], ["no_reply"], ["notify"]])
async def test_live_grant_respects_declared_inventory(tmp_path: Path, declared: list[str]) -> None:
    directory = tmp_path / "sess"
    _workstream(directory)
    stream = ScriptedStream([[StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream)
    try:
        session.set_tool_inventory(declared)
        set_grant(directory, allow=True, reason="operator request")
        await session.prompt("Check bounded inventory")
        names = {tool.name for tool in stream.requests[0].tools}
        assert ("notify" in names) == ("notify" in declared)
        assert names <= set(declared)
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_grant_survives_a_fresh_session_owner(tmp_path: Path) -> None:
    directory = tmp_path / "sess"
    _workstream(directory)
    set_grant(directory, allow=True, reason="operator request")
    session = make_session(tmp_path, ScriptedStream([]))
    try:
        assert session._notify_tool_callable() is not None
        assert any(tool.name == "notify" for tool in session._tools)
    finally:
        await session.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exclusion", ["child", "hidden", "one-shot", "contract", "remote", "kill-switch"]
)
async def test_grant_never_overrides_hard_exclusions(
    tmp_path: Path, monkeypatch, exclusion: str
) -> None:
    session = make_session(tmp_path, ScriptedStream([]))
    _workstream(session.transcript.directory)
    set_grant(session.transcript.directory, allow=True, reason="operator request")
    if exclusion == "child":
        session._job_id = "synthetic-child"
    elif exclusion == "hidden":
        mark_session_origin(session.transcript.directory, "agent-shell")
    elif exclusion == "one-shot":
        session._one_shot_exit = True
    elif exclusion == "contract":
        session._output_contract = object()
    elif exclusion == "remote":
        (session.transcript.directory / "mesh.json").write_text("{}")
    else:
        from local_operator.tools import builtin

        monkeypatch.setattr(builtin, "NOTIFY_ENABLED", False)
    try:
        assert session._notify_tool_callable() is None
        session._reconcile_notify_tool()
        assert not any(tool.name == "notify" for tool in session._tools)
    finally:
        await session.dispose()
