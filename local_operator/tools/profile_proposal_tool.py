"""Conversation door for proposals; authority is always the host callback.

Deferred by default: filing receipts name this tool, so its schema need not tax
sessions that never edit profiles. It never accepts an authority flag on wire.
"""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from local_operator.harness.types import AgentTool, ToolContext
from local_operator.tools.builtin import _error, _text, _validation_error


class ProfileProposalParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal["list", "show", "apply", "reject"]
    id: str | None = None
    name: str | None = None


async def execute_profile_proposal(tool_call_id, args, signal=None, on_update=None, context=None):
    from local_operator import profile_proposals as P
    from local_operator.profile_scope import profile_diff_summary

    try:
        params = ProfileProposalParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, "profile_proposal", exc)
    registry = (context.agent_registry or context.team_registry) if context else None
    if context is None or registry is None:
        return _error(tool_call_id, "profile_proposal", "No profile store is attached.")
    root = Path(registry.config_dir)
    # Named in errors raised after the record is read ("Agents → marketer"):
    # apply/reject carry only an id, so ``params.name`` is usually empty.
    target_name = params.name or "the profile"
    try:
        if params.op == "list":
            rows = P.list_proposals(root, name=params.name)
            return _text(
                tool_call_id,
                "profile_proposal",
                "\n".join(
                    f"{r['id']}: {r['kind']} {r['name']!r} ({', '.join(r['fields'])}); "
                    f"review in Agents → {r['name']}."
                    for r in rows
                )
                or "No pending profile proposals.",
                details={"proposals": [P.summary(r) for r in rows]},
            )
        if not params.id:
            return _error(tool_call_id, "profile_proposal", "This operation requires id.")
        proposal = P.get(root, params.id)
        if proposal is None:
            return _error(tool_call_id, "profile_proposal", "Proposal not found.")
        target_name = str(proposal.get("name") or target_name)
        if params.op == "show":
            if context.decide_profile_proposal:
                await context.decide_profile_proposal(params.id, "show")
            from local_operator.profile_text_edits import stats_heading

            detail = P.detail(root, proposal)
            # Long-text fields carry the canonical whole-field counts (the same
            # ``field_line_stats`` the write receipt uses) and go LAST, which is
            # how ``profile_diff_summary`` keeps their bodies out of the count.
            long_text = {"instructions", "project"}
            rows = sorted(detail["changes"], key=lambda c: c["field"] in long_text)
            diff = "\n".join(
                (
                    stats_heading(c["field"], c["before_text"], c["after_text"])
                    if c["field"] in long_text
                    else c["label"]
                )
                + ":\n"
                + "\n".join(
                    difflib.unified_diff(
                        c["before_text"].splitlines(),
                        c["after_text"].splitlines(),
                        fromfile="before",
                        tofile="after",
                        lineterm="",
                    )
                )
                for c in rows
            )
            return _text(
                tool_call_id,
                "profile_proposal",
                f"{params.id}: {proposal['state']}. Agents → {proposal['name']}\n"
                + profile_diff_summary(diff),
                details={
                    "proposal_id": params.id,
                    "profile_kind": proposal["kind"],
                    "profile_name": proposal["name"],
                    "state": proposal["state"],
                    "fields": proposal["fields"],
                    "diff": diff.splitlines(),
                },
            )
        decide = context.decide_profile_proposal
        decision = await decide(params.id, params.op) if decide else "refused"
        if decision not in {"granted", "card", "autonomous"}:
            return _error(
                tool_call_id,
                "profile_proposal",
                f"Not applied: owner confirmation is required. The user can review it "
                f"in Agents → {proposal['name']}.",
            )
        # The audit actor says who decided: ``agent:`` for a clean local agent
        # applying a trusted proposal itself, ``conversation:`` for the person.
        prefix = "agent" if decision == "autonomous" else "conversation"
        actor = f"{prefix}:{context.session_id}"
        if params.op == "reject":
            result = P.reject(root, params.id, actor=actor)
            return _text(
                tool_call_id,
                "profile_proposal",
                f"Rejected {params.id}; profile unchanged.",
                details={"proposal": P.summary(result)},
            )
        result = P.apply(root, params.id, actor=actor)
        backup = result["backup_id"]
        fields = ", ".join(proposal["fields"])
        text = f"Applied {params.id} to {proposal['kind']} {proposal['name']!r} ({fields})."
        if backup:
            text += (
                f" Backup {backup}; revert with {proposal['kind']} op=revert backup_id={backup}."
            )
        return _text(
            tool_call_id,
            "profile_proposal",
            text,
            details={
                "proposal": P.summary(result["proposal"]),
                "backup_id": backup,
                "backup_path": result.get("backup_path"),
            },
        )
    except P.ProposalError as exc:
        return _error(
            tool_call_id,
            "profile_proposal",
            f"Not applied: {str(exc).rstrip('. ')}. Review in Agents → {target_name}.",
        )


def build_profile_proposal_tool(context: ToolContext) -> AgentTool | None:
    """Only hosts with profile registries can inspect or decide proposals."""
    if context.agent_registry is None and context.team_registry is None:
        return None
    return AgentTool(
        name="profile_proposal",
        label="Profile proposals",
        description=(
            "List/show proposed account agent or team changes in Agents. After the user "
            "agrees, apply the specific displayed proposal; reject unwanted proposals. "
            "The host verifies consent; child sessions cannot apply."
        ),
        parameters=ProfileProposalParams.model_json_schema(),
        approval_tier="read",
        concurrency="exclusive",
        interruptible=False,
        execute=execute_profile_proposal,
    )
