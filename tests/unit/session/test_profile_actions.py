"""Real Session/registry receipt actions; synthetic keys, no OS signing."""

import json
import os
from pathlib import Path
from typing import Any, Literal, overload

import pytest

from local_operator.harness.types import Message, TextContent, ToolResult
from local_operator.session.session import Session
from local_operator.tools.agent_tool import execute_agent
from local_operator.tools.team_tool import execute_team
from tests.unit.session.test_account_profile_consent import (
    SEED_INSTRUCTIONS,
    TAINT,
    _context,
    _session,
    _start_run,
)
from tests.unit.session.test_account_profile_consent import stores as consent_stores

stores = consent_stores
_EVIDENCE_ROOT = os.environ.get("LOCAL_OPERATOR_SCRATCHPAD")


async def deliberate_rewrite(execute, name, context) -> dict[str, Any]:
    """Read ``name`` through the real ``op='show'`` and bind a whole-field rewrite.

    Fixtures here replace a stored brief wholesale ON PURPOSE: they exercise
    receipts, proposals, backups and revert of a landed save, not the
    text-edit path. The lossless-edit guard (kept real, never mocked) refuses
    a whole-field write that drops stored lines unless the caller proves it
    read the current text, so the fixture does exactly what a model must: a
    real show, then ``replace=True`` with that show's printed revision.
    """
    shown = await execute("show-call", {"op": "show", "name": name}, None, None, context)
    assert not shown.is_error, shown.content
    assert shown.details is not None and shown.details.get("complete") is True
    text = "".join(part.text for part in shown.content if isinstance(part, TextContent))
    revisions = [
        line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("revision: ")
    ]
    assert len(revisions) == 1 and revisions[0].startswith("definition-revision:"), text
    return {"replace": True, "base_revision": revisions[0]}


@overload
async def saved(
    stores, kind="agent", *, receipt: Literal[True]
) -> tuple[Session, dict[str, Any], str, ToolResult]: ...


@overload
async def saved(
    stores, kind="agent", *, receipt: Literal[False] = False
) -> tuple[Session, dict[str, Any], str]: ...


async def saved(stores, kind="agent", *, receipt=False):
    session = _session(stores)
    context = _context(session, stores)
    context.register_profile_action = session._profile_actions.register
    name = "reviewer" if kind == "agent" else "lopdev"
    _start_run(session, Message.user(f"Update {name} instructions to demand tests."))
    execute = execute_agent if kind == "agent" else execute_team
    result = await execute(
        "trusted-call",
        {
            "op": "update",
            "name": name,
            "instructions": "Demand tests.",
            **await deliberate_rewrite(execute, name, context),
        },
        None,
        None,
        context,
    )
    assert not result.is_error, result.content
    assert result.details is not None
    value = (session, result.details["profile_action"], result.details["backup_id"])
    return (*value, result) if receipt else value


def identity(descriptor, call="trusted-call"):
    return {
        key: descriptor[key] for key in ("action_id", "receipt_id", "profile_kind", "profile_name")
    } | {"tool_call_id": call}


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["agent", "team"])
async def test_real_saved_receipt_reverts_once_and_reports_cached_success(stores, kind):
    session, action, _ = await saved(stores, kind)
    try:
        owner = session._profile_actions
        assert owner.status(**identity(action))["state"] == "available"
        first = owner.activate(action["action_id"], approved=True, remember=False)
        assert first["state"] == "succeeded"
        assert owner.activate(action["action_id"], approved=True, remember=False) == first
        assert owner.status(**identity(action))["state"] == "succeeded"
        if kind == "agent":
            row = stores["account"].get_agent_by_name("reviewer")
            assert stores["account"].get_agent_system_prompt(row.id) == SEED_INSTRUCTIONS
        else:
            # Another executor owns the write; inspect fresh persisted briefs.
            from local_operator.teams import TeamRegistry

            stored = TeamRegistry(stores["teams"].config_dir).get_team_by_name("lopdev")
            assert stored is not None
            assert stored.instructions == "Ship."
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_forged_identity_stale_target_and_restart_disable_actions(stores):
    from local_operator.profile_actions import ProfileActions

    session, action, _ = await saved(stores)
    try:
        owner = session._profile_actions
        assert owner.status(**identity(action, "other-call"))["state"] == "unavailable"
        assert (
            ProfileActions(session._session_id).status(**identity(action))["state"] == "unavailable"
        )
        row = stores["account"].get_agent_by_name("reviewer")
        stores["account"].set_agent_system_prompt(row.id, "Later edit.")
        assert owner.status(**identity(action))["state"] == "stale"
        assert (
            owner.activate(action["action_id"], approved=True, remember=False)["state"] == "stale"
        )
        assert stores["account"].get_agent_system_prompt(row.id) == "Later edit."
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_substituted_backup_and_remember_never_restore(stores):
    session, action, backup = await saved(stores)
    try:
        owner = session._profile_actions
        with pytest.raises(ValueError, match="never enable"):
            owner.activate(action["action_id"], approved=True, remember=True)
        path = stores["session_dir"] / "profile-backups" / (backup + ".json")
        body = json.loads(path.read_text())
        body["instructions"] = "Substituted."
        path.write_text(json.dumps(body))
        assert (
            owner.activate(action["action_id"], approved=True, remember=False)["state"] == "failed"
        )
        row = stores["account"].get_agent_by_name("reviewer")
        assert stores["account"].get_agent_system_prompt(row.id) == "Demand tests."
    finally:
        await session.dispose()


async def _wire_action_probe(stores, tmp_path, monkeypatch, kind="agent"):
    from local_operator.harness.approval import mint_operator_cap
    from local_operator.paths import config_dir
    from tests.unit.session.runtime.test_approval_authority_seam import (
        _dial,
        _send,
        _serve,
    )

    session, action, _, saved_result = await saved(stores, kind, receipt=True)

    cap = mint_operator_cap()
    from tests.unit.session.runtime import test_approval_authority_seam as seam

    # Use the real Session from handle construction onward, including the
    # trusted session identity published by the production runtime itself.
    monkeypatch.setattr(seam, "_AttachableSession", lambda: session)
    live = await _serve(tmp_path, operator_cap=cap)
    bare = await _dial(live.record)
    owner = None
    try:
        status = await _send(bare, None, {"op": "profile_action_status", **identity(action)})
        assert status.get("detail", {}).get("state") == "available", status
        refused = await _send(
            bare,
            None,
            {
                "op": "approval_answer",
                "request_id": action["action_id"],
                "approved": True,
                "remember": False,
            },
        )
        assert refused.get("op") == "error", refused
        # A non-viewer control socket is the runtime's single daemon seat.
        # Connecting the owner first would legitimately replace the bare seat.
        bare.close()
        owner = await _dial(live.record, cap=cap)
        frame = {
            "op": "approval_answer",
            "request_id": action["action_id"],
            "approved": True,
            "remember": False,
            "operator_cap": owner.proof(cap),
        }
        result = await _send(owner, None, frame)
        assert result.get("detail", {}).get("state") == "succeeded", result
        duplicate = await _send(owner, None, frame)
        assert duplicate.get("detail", {}).get("state") == "succeeded", duplicate
        # Treat the first result as dropped: the read reconciles completion
        # without resending the inverse or trusting a persisted verified flag.
        reconciled = await _send(owner, None, {"op": "profile_action_status", **identity(action)})
        assert reconciled.get("detail", {}).get("state") == "succeeded", reconciled
        return {
            "available": status,
            "refused": refused,
            "applied": result,
            "duplicate": duplicate,
            "reconciled": reconciled,
            "saved_tool_result": saved_result.model_dump(mode="json"),
            "source_session_id": live.record.session_id,
        }
    finally:
        bare.close()
        if owner is not None:
            owner.close()
        await live.close(config_dir())


@pytest.mark.asyncio
async def test_actual_socket_status_and_authority_gated_duplicate_activation(
    stores, tmp_path, monkeypatch
):
    await _wire_action_probe(stores, tmp_path, monkeypatch)


@pytest.mark.asyncio
async def test_export_golden_profile_wires(stores, tmp_path, monkeypatch):
    """Function serialization is not HTTP proof; ACK cases use the real socket."""
    import shlex
    import subprocess
    import sys

    from local_operator import profile_proposals as PP

    session = _session(stores)
    context = _context(session, stores)
    context.register_profile_action = session._profile_actions.register
    try:
        # TAINT (untrusted peer input): since the 2026-11 ruling a clean wake run
        # writes its own profiles directly, so only a run that read untrusted
        # input still files the Agents proposal this test pins.
        _start_run(session, TAINT)
        queued = await execute_agent(
            "queued-call",
            {
                "op": "update",
                "name": "reviewer",
                "instructions": "Public queued fixture guidance.",
                **await deliberate_rewrite(execute_agent, "reviewer", context),
            },
            None,
            None,
            context,
        )
        # An unrequested (wake-run) edit is an Agents proposal, accepted here
        # the way the desktop Agents route accepts it (no mesh signature).
        assert queued.details is not None
        assert queued.details["status"] == "proposed"
        root = stores["account"].config_dir
        rows = [PP.summary(row) for row in PP.list_proposals(root)]
        signed = PP.apply(root, queued.details["proposal_id"], actor="desktop")
        _start_run(
            session, Message.user("Update reviewer instructions to public fixture guidance.")
        )
        agent = await execute_agent(
            "saved-agent",
            {
                "op": "update",
                "name": "reviewer",
                "instructions": "Public saved agent fixture guidance.",
                **await deliberate_rewrite(execute_agent, "reviewer", context),
            },
            None,
            None,
            context,
        )
        _start_run(session, Message.user("Update lopdev instructions to public team guidance."))
        team = await execute_team(
            "saved-team",
            {
                "op": "update",
                "name": "lopdev",
                "instructions": "Public saved team fixture guidance.",
                **await deliberate_rewrite(execute_team, "lopdev", context),
            },
            None,
            None,
            context,
        )
        assert agent.details is not None and team.details is not None
        assert agent.details.get("profile_action") and team.details.get("profile_action")
        read_agent = await execute_agent(
            "show-agent", {"op": "show", "name": "reviewer"}, None, None, context
        )
        read_team = await execute_team(
            "show-team", {"op": "show", "name": "lopdev"}, None, None, context
        )
        assert not read_agent.is_error and not read_team.is_error
        assert not (read_agent.details or {}).get("profile_action")
        assert not (read_team.details or {}).get("profile_action")
        git_env = dict(os.environ, GIT_CONFIG_SYSTEM="/dev/null", GIT_TERMINAL_PROMPT="0")
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, env=git_env).strip()
        golden = {
            "meta": {
                "source_sha": sha,
                "uncommitted_source": bool(
                    subprocess.check_output(
                        ["git", "status", "--porcelain"], text=True, env=git_env
                    ).strip()
                ),
                "generator_command": (
                    f"PYTHONPATH=. {shlex.quote(sys.executable)} -m pytest -n0 -q "
                    "-p no:cacheprovider tests/unit/session/test_profile_actions.py "
                    "-k export_golden_profile_wires"
                ),
                "list_row_proof": (
                    "actual profile_proposals.list_proposals(root) summaries, "
                    "function serialization only; not HTTP"
                ),
            },
            "saved_agent_tool_result": agent.model_dump(mode="json"),
            "saved_team_tool_result": team.model_dump(mode="json"),
            "queued_tool_result": queued.model_dump(mode="json"),
            "readonly_agent_tool_result": read_agent.model_dump(mode="json"),
            "readonly_team_tool_result": read_team.model_dump(mode="json"),
            "profile_proposal_rows": rows,
            "accepted_proposal_receipt": signed,
        }
    finally:
        await session.dispose()
    golden["live_action_ack"] = {}
    for kind in ("agent", "team"):
        wire = await _wire_action_probe(stores, tmp_path, monkeypatch, kind)
        # Keep the EXACT saved result that minted this socket action: a fixture
        # pairing a later action's ACK with an earlier result is not live proof.
        golden[f"saved_{kind}_tool_result"] = wire.pop("saved_tool_result")
        golden["meta"][f"{kind}_source_session_id"] = wire.pop("source_session_id")
        descriptor = golden[f"saved_{kind}_tool_result"]["details"]["profile_action"]
        assert wire["available"]["detail"]["action_id"] == descriptor["action_id"]
        golden["live_action_ack"][kind] = wire
    from local_operator.harness.types import ToolExecutionEndEvent, ToolResult

    golden["tool_execution_end_events"] = {}
    for key in (
        "saved_agent_tool_result",
        "saved_team_tool_result",
        "readonly_agent_tool_result",
        "readonly_team_tool_result",
    ):
        value = ToolResult.model_validate(golden[key])
        event = ToolExecutionEndEvent(
            tool_call_id=value.tool_call_id,
            tool_name=value.tool_name,
            result=value,
            duration_s=value.duration_s,
        )
        golden["tool_execution_end_events"][key] = event.model_dump(mode="json")
    golden["meta"]["event_carrier_proof"] = (
        "Native ToolExecutionEndEvent.model_dump(mode=json) around real executed results; "
        "constructed carrier, not recorded harness dispatch"
    )
    path = Path(_EVIDENCE_ROOT or tmp_path) / "profile-action-golden.json"
    path.write_text(json.dumps(golden, indent=2, sort_keys=True))
    assert golden["live_action_ack"]["agent"]["available"]["detail"]["state"] == "available"


@pytest.mark.asyncio
async def test_later_writer_between_save_and_receipt_cannot_become_the_revert_base(
    stores, monkeypatch
):
    from local_operator.tools import agent_tool

    original = agent_tool._account_save_receipt

    def receipt(context, registry, name, plan, details, tool_call_id=""):
        row = registry.get_agent_by_name(name)
        registry.set_agent_system_prompt(row.id, "Later writer before receipt.")
        return original(context, registry, name, plan, details, tool_call_id)

    monkeypatch.setattr(agent_tool, "_account_save_receipt", receipt)
    session, action, _ = await saved(stores)
    try:
        assert session._profile_actions.status(**identity(action))["state"] == "stale"
        result = session._profile_actions.activate(
            action["action_id"], approved=True, remember=False
        )
        assert result["state"] == "stale"
        row = stores["account"].get_agent_by_name("reviewer")
        assert stores["account"].get_agent_system_prompt(row.id) == "Later writer before receipt."
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_status_failure_never_discloses_private_instruction_values(stores, monkeypatch):
    from local_operator import profile_approval

    session, action, _ = await saved(stores)

    def broken(*args, **kwargs):
        raise ValueError("private instruction fixture must not be returned")

    monkeypatch.setattr(profile_approval, "snapshot", broken)
    try:
        status = session._profile_actions.status(**identity(action))
        assert status == {"action_id": action["action_id"], "state": "unavailable"}
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_real_metadata_write_failure_settles_terminal_without_retry(stores, monkeypatch):
    from local_operator import agents

    session, action, _ = await saved(stores)
    original = agents._write_text_atomically
    calls = []

    def fail(path, *args, **kwargs):
        if path.name == "agent.yml":
            calls.append(path)
            raise OSError("synthetic actual atomic metadata failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(agents, "_write_text_atomically", fail)
    try:
        result = session._profile_actions.activate(
            action["action_id"], approved=True, remember=False
        )
        assert result["state"] == "failed"
        assert session._profile_actions.status(**identity(action))["state"] == "failed"
        count = len(calls)
        assert count == 1
        assert (
            session._profile_actions.activate(action["action_id"], approved=True, remember=False)[
                "state"
            ]
            == "failed"
        )
        assert len(calls) == count
        row = stores["account"].get_agent_by_name("reviewer")
        assert stores["account"].get_agent_system_prompt(row.id) == "Demand tests."
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_actual_queued_receipt_has_one_canonical_diff(stores):
    from local_operator.tools.agent_tool import execute_agent
    from local_operator.tui.widgets.tool_card import ToolCard

    session = _session(stores)
    try:
        # TAINT (untrusted peer input): since the 2026-11 ruling a clean wake run
        # writes its own profiles directly, so only a run that read untrusted
        # input still files the Agents proposal this test pins.
        _start_run(session, TAINT)
        context = _context(session, stores)
        result = await execute_agent(
            "queued-diff",
            {
                "op": "update",
                "name": "reviewer",
                "instructions": SEED_INSTRUCTIONS.replace("Be terse.", "Demand tests."),
                **await deliberate_rewrite(execute_agent, "reviewer", context),
            },
            None,
            None,
            context,
        )
        assert result.details is not None
        # Unrequested (wake-run) edits are filed as Agents proposals now.
        assert result.details["status"] == "proposed"
        card = ToolCard("queued-diff", "agent", {"op": "update", "name": "reviewer"})
        card.mark_done(
            "\n".join(item.text for item in result.content if isinstance(item, TextContent)),
            details=result.details,
        )
        card._expanded = True
        body = card._build_content(110).plain
        assert body.count("+Demand tests.") == 1
        # The receipt line now uses the approval card's own summary vocabulary.
        assert "Proposed" in body and "instructions: +1 −1 lines" in body
    finally:
        await session.dispose()
