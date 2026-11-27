"""The ``guard`` agent tool: Radient Guard status, report drafts, admin insights link.

Guard RFC v0.9 §5.2.4 / §9.2' / §5.6, slice L2b.

GATE (rung 3, createIf; AGENTS.md "tool-surface footprint ladder"). Built only
where :func:`guard.surface.guard_available` holds: the account is on Radient Team
(the cached, zero-network membership record) AND an org has turned Guard on for
this member, which this computer knows because L2a's admission point queued a
prompt. Everyone else -- signed out, a non-Team account, a Team org that never
turned Guard on -- gets ``None``: no schema, no description, zero ``/v1/guard/*``
calls, no file. DEFERRED (``tools/deferral.py``): even where it exists, the schema
costs only the inventory phrase until ``read tool://guard`` or a first call.

WHAT THE MODEL CAN DO, AND WHAT IT CANNOT.

* ``status`` -- the passive indicator per org ("sharing", "waiting for your
  choice", "paused"), from local facts only. No figures exist on this computer.
* ``report_draft`` -- DRAFT a one-shot report of a closed ``kind``
  (``credential_exposure``, ``excessive_guardrail``, ``other``). The schema has no
  free-text field at all, so a secret cannot ride a report even by mistake. The
  draft is a local row; the person sends or discards it in ``/guard``
  (``guard/reports.py``: ``confirm`` is admitted only through
  ``admit_increasing``, which no tool call carries). The tool never sends.
* ``open_insights`` -- ONLY in the schema of an owner/admin of a Team org, and
  re-checked at call time. It returns an opaque first-party Console URL, never
  counts or reports: org insights are k-anonymous releases rendered in Console,
  and keeping them out of tool results keeps them out of transcripts and peers.

There is NO consent verb: the consent prompt is answered only by a person at a
host surface (§5.2.4). ``status`` points at ``/guard`` and stops there.

OUTPUT is model-facing and therefore English by design (AGENTS.md i18n
exceptions); the person-facing strings live in the ``tui.guard`` catalogue.
"""

from __future__ import annotations

import json
import logging
import unicodedata
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from local_operator.harness.types import AbortSignal, AgentTool, ToolContext, ToolResult

logger = logging.getLogger(__name__)

TOOL_NAME = "guard"

ADMIN_ACTIONS = ("open_insights",)


class GuardParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["status", "report_draft", "open_insights"] = Field(
        description="status | report_draft (a person sends it in /guard) | open_insights"
    )
    kind: Literal["credential_exposure", "excessive_guardrail", "other"] | None = Field(
        default=None, description="report_draft: what you observed. No other detail is sent."
    )
    tenant: str | None = Field(
        default=None, description="Org tenant id; omit when there is one Guard org."
    )


#: Recommends self-reporting (operator ask 2): every clause changes behaviour.
#: "Never put the value anywhere" matters because the natural next step after
#: spotting a leaked key is quoting it; the report needs only the kind.
_DESCRIPTION = (
    "Radient Guard for this member's Team org. When you notice a credential exposed "
    "in your work (a key in output, a file, a log) or a guardrail blocking legitimate "
    "work, draft a report with report_draft: kind only, never the value or any detail. "
    "A person sends or discards it in /guard; you cannot send it. status shows whether "
    "this computer shares Guard stats. See guide://guard."
)

#: Added only for an owner/admin (operator ask 3): a member's tool never offers
#: the insights link, so the model cannot send someone to a page they cannot
#: open. (The guide and the consent prompt do say that admins see aggregate
#: stats -- that is the disclosure the member consents to -- but neither names a
#: route to them; QA round 1, Q2.)
_ADMIN_CLAUSE = (
    " open_insights: your org's Guard insights are in Console; this returns the "
    "link to give the user. It returns no figures; never invent any."
)


#: Cap on an org-supplied name in model-facing text (security round 1, S-3).
#: The server allows 256 characters (``guard/consent.py``); 80 holds every real
#: org name while leaving no room for a paragraph of instructions.
_NAME_CAP = 80


def _org_label(name: str) -> str:
    """An org-SUPPLIED name, made inert for model context (security round 1, S-3).

    The name is set by the org -- the party that gains when members opt in --
    and lands in the member's agent context. So it is one line (control and
    format characters, newlines included, become spaces), capped at
    :data:`_NAME_CAP` characters, JSON-quoted so it reads as data, and labelled
    as the org's own text. The TUI renders names with markup off, so this is
    the model-facing half only.
    """

    flat = "".join(" " if unicodedata.category(ch)[0] == "C" else ch for ch in str(name or ""))
    flat = " ".join(flat.split())
    if len(flat) > _NAME_CAP:
        flat = flat[: _NAME_CAP - 1] + "…"
    return f"org-supplied name {json.dumps(flat, ensure_ascii=False)}"


def _tenant_label(tenant: str) -> str:
    """A tenant id, server-issued but still org-adjacent: quoted and capped too."""

    flat = "".join(ch for ch in str(tenant or "") if unicodedata.category(ch)[0] != "C")
    return json.dumps(flat[:128], ensure_ascii=False)


def _config_dir() -> Path | None:
    try:
        from local_operator.paths import config_dir

        return Path(config_dir())
    except Exception:  # noqa: BLE001 - no config root, no tool
        return None


def _schema(admin: bool) -> dict[str, Any]:
    schema = GuardParams.model_json_schema()
    if not admin:
        action = (schema.get("properties") or {}).get("action") or {}
        if isinstance(action.get("enum"), list):
            action["enum"] = [a for a in action["enum"] if a not in ADMIN_ACTIONS]
        action["description"] = "status | report_draft (a person sends it in /guard)"
    return schema


def _gate() -> tuple[bool, bool]:
    """``(guard available, owner/admin on a Team org)`` for this machine. Never raises.

    A module-level seam so the real-surface measurement
    (``scripts/real_tool_surface.py``) and the registry tests can hold the tool
    as a Guard member would, without a membership record on the runner.
    """

    from local_operator.guard import surface

    root = _config_dir()
    if not surface.guard_available(root):
        return False, False
    return True, bool(surface.guard_insights_tenants(root))


def build_guard_tool(context: ToolContext) -> AgentTool | None:
    """createIf: Guard is on for this member (Team + an org-raised prompt)."""

    del context  # the gate is the machine's membership record, not session state
    available, admin = _gate()
    if not available:
        return None
    return AgentTool(
        name=TOOL_NAME,
        label="Guard",
        description=_DESCRIPTION + (_ADMIN_CLAUSE if admin else ""),
        parameters=_schema(admin),
        # read tier: status and open_insights read local facts; report_draft
        # writes one inert local row that a person must still confirm, which is
        # the real gate. Prompting for the draft would add a second, weaker gate
        # in front of the one that matters.
        approval_tier="read",
        concurrency="exclusive",
        interruptible=False,
        execute=execute_guard,
    )


def _pick_tenant(requested: str | None, candidates: list[tuple[str, str]]) -> tuple[str, str] | str:
    """The (tenant, name) to act on, or a refusal sentence."""

    if requested:
        for tenant, name in candidates:
            if tenant == requested:
                return tenant, name
        return f"tenant {_tenant_label(requested)} is not one of this member's Guard orgs."
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return "no Team org has turned on Radient Guard for this member."
    names = ", ".join(
        f"{_tenant_label(tenant)} ({_org_label(name)})" for tenant, name in candidates
    )
    return f"several Guard orgs; pass tenant= one of: {names}."


async def execute_guard(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Any = None,
    context: ToolContext | None = None,
) -> ToolResult:
    from local_operator.tools.builtin import _error, _validation_error

    del signal, on_update, context
    try:
        params = GuardParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, TOOL_NAME, exc)

    from local_operator.guard import surface

    root = _config_dir()
    # Re-checked per call: the schema was built at session start, and a member
    # whose org left Team since then must not keep a working tool.
    if root is None or not surface.guard_available(root):
        return _error(tool_call_id, TOOL_NAME, "Radient Guard is not available for this account.")

    from local_operator.guard.consent import GuardConsentRefused
    from local_operator.guard.reports import GuardReportRefused

    try:
        return _execute(tool_call_id, params, root)
    except (GuardConsentRefused, GuardReportRefused) as exc:
        # A damaged or newer-generation store (review R1-3): say so, never raise
        # into the turn. Nothing was drafted or sent.
        damaged = exc.code.endswith("_damaged")
        return _error(
            tool_call_id,
            TOOL_NAME,
            (
                "Radient Guard's local store on this computer can't be read right now; "
                "nothing was drafted or sent."
                if damaged
                else str(exc)
            ),
            details={"action": params.action, "code": exc.code},
        )


def _execute(tool_call_id: str, params: GuardParams, root: Path) -> ToolResult:
    from local_operator.guard import surface
    from local_operator.tools.builtin import _error, _text

    if params.action == "open_insights":
        admins = surface.guard_insights_tenants(root)
        if not admins:
            return _error(
                tool_call_id,
                TOOL_NAME,
                "open_insights is for org owners and admins on Radient Team.",
                details={"action": "open_insights", "code": "guard_insights_role"},
            )
        picked = _pick_tenant(params.tenant, admins)
        if isinstance(picked, str):
            return _error(tool_call_id, TOOL_NAME, picked)
        tenant, name = picked
        url = surface.insights_url(tenant)
        return _text(
            tool_call_id,
            TOOL_NAME,
            f"Guard insights for {_org_label(name)} are in Console: {url}\n"
            "Give the user this link. Figures are shown there only, after sign-in.",
            details={"action": "open_insights", "tenant": tenant, "url": url},
        )

    states = surface.org_states(root)
    if params.action == "status":
        if not states:
            return _text(tool_call_id, TOOL_NAME, "No Guard org has asked this member to share.")
        lines = [_status_line(state) for state in states]
        return _text(
            tool_call_id,
            TOOL_NAME,
            "\n".join(lines),
            details={
                "action": "status",
                "orgs": [
                    {"tenant": s.tenant, "state": s.state, "reason": s.reason} for s in states
                ],
            },
        )

    # report_draft
    if params.kind is None:
        return _error(
            tool_call_id,
            TOOL_NAME,
            "report_draft needs kind: credential_exposure, excessive_guardrail or other.",
        )
    picked = _pick_tenant(params.tenant, [(s.tenant, s.org) for s in states])
    if isinstance(picked, str):
        return _error(tool_call_id, TOOL_NAME, picked)
    tenant, name = picked
    from local_operator.guard.reports import GuardReports

    draft, created = GuardReports(root).draft(tenant, params.kind)
    lead = "Drafted" if created else "Already drafted"
    return _text(
        tool_call_id,
        TOOL_NAME,
        f"{lead} a {params.kind} report for {_org_label(name)}. NOT sent: the user sends or "
        "discards it in /guard. Tell them so; do not include the credential or any "
        "detail anywhere.",
        details={
            "action": "report_draft",
            "tenant": tenant,
            "kind": params.kind,
            "draft_id": draft.id,
            "sent": False,
        },
    )


#: Plain reasons for a paused state (copy round 1, C7): §15.4' C9 says never
#: the enum, and the agent repeats this text to the user.
_PAUSED_PLAIN = {
    "entitlement_lapsed": "the Team plan is inactive",
    "org_off": "the org turned Guard off",
    "membership_ended": "the user is no longer in the org",
    "scope_revoked": "the user stopped sharing",
    "console_revoked": "the user stopped sharing from Console",
    "revoke_pending": "stopping; waiting for Radient to confirm",
}


def _status_line(state: Any) -> str:
    from local_operator.guard import surface

    text = {
        surface.STATE_SHARING: "sharing Guard stats from this computer",
        surface.STATE_SHARING_PENDING: (
            "sharing; the org asked for more and the user has not chosen yet (/guard)"
        ),
        surface.STATE_WAITING: "waiting for the user's choice in /guard; nothing shared yet",
        surface.STATE_NOT_SHARING: "the user chose not to share",
        surface.STATE_WILL_ASK: (
            "not sharing; the user asked to be asked again, and the org's next prompt "
            "will appear in /guard"
        ),
        surface.STATE_PAUSED: f"paused ({_PAUSED_PLAIN.get(state.reason, 'paused')})",
    }.get(state.state, "unknown")
    return f"{_org_label(state.org)} (tenant {_tenant_label(state.tenant)}): {text}"


__all__ = ["GuardParams", "build_guard_tool", "execute_guard"]
