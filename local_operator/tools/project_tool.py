"""The ``project`` tool: track a workstream across sessions.

A project is a named row (status, description, progress snippet, tags, linked
sessions, dates, estimate, milestones) in the operator's own config root. It is
deliberately not an agent, not a schedule and not a second team — see
:mod:`local_operator.projects` for storage and ``guide://projects`` for how to
use the primitive well.

Shape: ONE tool with ops (``list | show | create | update | link | unlink |
milestone``) plus ``project_delete`` as a separate write-tier tool, exactly the
``team`` / ``team_delete`` split — the destructive split is the existing
convention so that deletion always asks for write approval.

CONTEXT DISCIPLINE
==================

The registry is NEVER enumerated into the prompt. ``list`` returns one compact
line per project and only when called; ``show`` is the only op that returns the
full record plus the aggregated view of its linked sessions. Long output is
spilled through the same helper the other tools use.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from local_operator.harness.types import (
    AbortSignal,
    AgentTool,
    AgentToolUpdate,
    ToolContext,
    ToolResult,
)
from local_operator.projects import (
    _SESSION_ID_RE,
    MILESTONES_MAX,
    SESSIONS_MAX,
    MilestoneEdit,
    Project,
    ProjectEdit,
    ProjectMilestone,
    ProjectNameConflictError,
    ProjectRegistry,
    ProjectRegistryLockTimeout,
    ProjectSchemaGuardError,
    build_project_view,
    milestone_status,
    progress_is_stale,
    readable_error,
)
from local_operator.tools.builtin import (
    _error,
    _guard,
    _text,
    _validation_error,
    spill_truncate,
)

logger = logging.getLogger(__name__)

#: One listing row stays scannable in a transcript; the same discipline as the
#: team tool's row cap.
_ROW_CAP = 160

_STATUS_WORDS = ("active", "paused", "done", "archived")


class ProjectParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["list", "show", "create", "update", "link", "unlink", "milestone"] = Field(
        description=(
            "list/show: read; create/update: author or fix; link/unlink: attach or "
            "detach a session; milestone: add-or-update or remove ONE. "
            "project_delete removes irreversibly."
        )
    )
    name: str | None = Field(default=None, description="Project name (all ops but list).")
    description: str | None = Field(
        default=None, description="create/update: one line on what this workstream is."
    )
    status: Literal["active", "paused", "done", "archived"] | None = Field(
        default=None,
        description=(
            "create/update. 'done' with completed_at omitted stamps today; an "
            "explicit '' clears it; moving away from 'done' leaves it."
        ),
    )
    progress: str | None = Field(
        default=None,
        description=(
            "update: one dated line, not a transcript. Re-sending the same line on "
            "a stale record refreshes its timestamp; on a fresh record it writes "
            "nothing."
        ),
    )
    tags: list[str] | None = Field(
        default=None,
        description="create/update: replace-set; each [a-z0-9][a-z0-9_-]{0,23}.",
    )
    session_id: str | None = Field(
        default=None, description="link/unlink: defaults to THIS session."
    )
    start_date: str | None = Field(
        default=None, description="create/update: ISO YYYY-MM-DD; '' clears it."
    )
    target_date: str | None = Field(
        default=None,
        description="create/update: ISO YYYY-MM-DD, never before start_date; '' clears it.",
    )
    completed_at: str | None = Field(
        default=None,
        description="create/update: ISO YYYY-MM-DD the work finished; '' clears it.",
    )
    estimate: float | None = Field(
        default=None, description="create/update: > 0 and <= 1000 (fractional allowed)."
    )
    estimate_unit: Literal["points", "days"] | None = Field(
        default=None, description="create/update: 'points' (default) or 'days'."
    )
    milestones: list[ProjectMilestone] | None = Field(
        default=None,
        description=(
            "create/update: FULL replace (<= 20, names unique case-insensitively). "
            "Prefer op='milestone' for one edit."
        ),
    )
    milestone: str | None = Field(
        default=None, description="milestone: its name (add-or-update by name)."
    )
    milestone_target_date: str | None = Field(
        default=None,
        description="milestone: ISO YYYY-MM-DD; '' clears it.",
    )
    milestone_completed: bool | None = Field(
        default=None,
        description="milestone: True sets completed_at to today, False clears it.",
    )
    remove: bool = Field(default=False, description="milestone: remove the named milestone.")


class ProjectDeleteParams(BaseModel):
    """Separate schema so irreversible removal can carry a write-tier gate."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(description="Existing project name to delete permanently.")


def _registry(context: ToolContext | None) -> ProjectRegistry | None:
    raw = getattr(context, "project_registry", None) if context else None
    return raw if isinstance(raw, ProjectRegistry) else None


def _calling_session_id(context: ToolContext | None) -> str | None:
    """This session's id, when it has the shape the row can store.

    ``None`` means the caller has no session identity (or one off the contract's
    shape): ``create`` then makes an unlinked project and says so rather than
    storing a junk id that would fail validation.
    """
    candidate = str(getattr(context, "session_id", "") or "")
    return candidate if _SESSION_ID_RE.fullmatch(candidate) else None


def _reported_age(project: Project, *, now: float | None = None) -> str | None:
    """``2h``-style age of the progress snippet, or ``None`` when none is stored.

    The caller composes the sentence, so one implementation of the age arithmetic
    serves the listing row, the ``show`` block and the update receipt without any
    of them disagreeing about how old the snippet is.
    """
    if not project.progress or project.progress_updated_at is None:
        return None
    moment = time.time() if now is None else now
    age = max(0.0, moment - project.progress_updated_at)
    if age < 90:
        return f"{int(age)}s"
    if age < 5400:
        return f"{int(age // 60)}m"
    if age < 172800:
        return f"{int(age // 3600)}h"
    return f"{age / 86400:.0f}d"


def _milestone_counts(project: Project) -> str:
    done = sum(1 for m in project.milestones if m.completed_at)
    return f"M {done}/{len(project.milestones)}"


def _estimate_text(project: Project) -> str:
    if project.estimate is None:
        return "no estimate"
    suffix = "pt" if project.estimate_unit == "points" else "d"
    number = f"{project.estimate:g}"
    return f"est {number}{suffix}"


def _row(project: Project, *, now: float | None = None) -> str:
    """One scannable listing line, ``_ROW_CAP``-bounded."""
    parts = [
        f"- {project.name} [{project.status}]",
        _estimate_text(project),
    ]
    if project.target_date:
        parts.append(f"→{project.target_date}")
    if project.milestones:
        parts.append(_milestone_counts(project))
    sessions = len(project.sessions)
    parts.append(f"{sessions} session" + ("" if sessions == 1 else "s"))
    age = _reported_age(project, now=now)
    if age is None:
        parts.append("no progress")
    else:
        stale = " (stale)" if progress_is_stale(project, now=now) else ""
        parts.append(f"progress {age} ago{stale}")
    summary = (project.description or "").strip()
    row = " · ".join(parts)
    if summary:
        # QUOTED: an unquoted tail read as if it were the progress text (the
        # first cut of this row shipped exactly that ambiguity).
        row += f' · "{summary}"'
    return row if len(row) <= _ROW_CAP else row[: _ROW_CAP - 1].rstrip() + "…"


def _field_lines(project: Project) -> list[str]:
    """The ``show`` header block: every stored field, derived status where one exists."""
    lines = [
        f"{project.name} [{project.status}]",
        f"description: {project.description or '(unstated)'}",
        f"estimate: {_estimate_text(project)}",
        f"dates: start {project.start_date or '—'} · target {project.target_date or '—'}"
        f" · completed {project.completed_at or '—'}",
        f"tags: {', '.join(project.tags) if project.tags else '(none)'}",
    ]
    age = _reported_age(project)
    stale = ", stale" if progress_is_stale(project) else ""
    freshness = f"reported {age} ago{stale}" if age is not None else "none recorded"
    lines.append(f"progress ({freshness}): {project.progress or '—'}")
    if project.progress and project.progress_reported_by:
        lines.append(f"progress reported by: {project.progress_reported_by}")
    if project.milestones:
        lines.append(f"milestones ({len(project.milestones)}/{MILESTONES_MAX}):")
        for milestone in project.milestones:
            state = milestone_status(milestone)
            target = milestone.target_date or "—"
            lines.append(f"  - {milestone.name} [{state}] target {target}")
    else:
        lines.append("milestones: (none)")
    return lines


def _session_lines(view: dict[str, Any]) -> list[str]:
    """Per-session rollup from the ONE shared composition (``build_project_view``)."""
    rows = view.get("sessions") or []
    if not rows:
        return ["linked sessions: (none)"]
    lines: list[str] = [f"linked sessions ({len(rows)}/{SESSIONS_MAX}):"]
    for row in rows:
        runtime = row.get("runtime") or {}
        state = runtime.get("state") or "stopped"
        busy = ", busy" if runtime.get("busy") else ""
        session_id = row["session_id"]
        if not row.get("exists"):
            lines.append(f"  - {session_id} [missing] — no session directory")
            continue
        title = row.get("title") or "(untitled)"
        bits = [state + busy]
        agents = row.get("subagents")
        if agents is not None:
            bits.append(f"{agents['running']} running · {agents['settled']} settled subagents")
        todos = row.get("todos")
        if todos is not None:
            bits.append(f"todos {todos['open']} open / {todos['total']}")
        if row.get("archived"):
            bits.append("archived")
        lines.append(f"  - {session_id} [{' · '.join(bits)}] {title}")
    return lines


def _project_edit(params: ProjectParams, *, creating: bool) -> ProjectEdit:
    """A ``ProjectEdit`` carrying exactly the fields the caller supplied.

    ``model_fields_set`` is the whole point: an update must not clobber a field
    the caller never mentioned, and an explicit ``""``/``None`` must stay
    distinguishable from an absent one (the clear-vs-untouched vocabulary).
    """
    payload: dict[str, Any] = {}
    # The params only THIS op dispatches on; everything else is a row field the
    # edit vocabulary owns.
    dispatch_only = {
        "op",
        "name",
        "session_id",
        "milestone",
        "milestone_target_date",
        "milestone_completed",
        "remove",
    }
    for field in params.model_fields_set:
        if field in dispatch_only:
            continue
        payload[field] = getattr(params, field)
    if creating:
        # An explicit name must be present for `validate_project_name` inside
        # the store; the tool has already refused a blank one.
        payload["name"] = params.name
    return ProjectEdit(**payload)


async def _op_list(context: ToolContext | None, tool_call_id: str) -> ToolResult:
    registry = _registry(context)
    if registry is None:
        return _error(tool_call_id, "project", "no project registry attached to this session.")
    try:
        projects = registry.list_projects()
    except ProjectRegistryLockTimeout as exc:
        return _error(tool_call_id, "project", str(exc))
    if not projects:
        body = (
            "no projects yet. Create one with op='create' (it links this "
            "session automatically) once the workstream is worth tracking. "
            "Read guide://projects first."
        )
        return _text(tool_call_id, "project", body)
    body = "projects:\n" + "\n".join(_row(project) for project in projects)
    body += "\n\nop='show' name='<name>' has the full record plus its linked sessions."
    text, spill = spill_truncate(body, "project", context)
    return _text(tool_call_id, "project", text, details=spill or None)


async def _op_show(context: ToolContext | None, tool_call_id: str, name: str) -> ToolResult:
    registry = _registry(context)
    if registry is None:
        return _error(tool_call_id, "project", "no project registry attached to this session.")
    try:
        project = registry.get_project_by_name(name)
    except ProjectRegistryLockTimeout as exc:
        return _error(tool_call_id, "project", str(exc))
    if project is None:
        return _error(tool_call_id, "project", f"no project named {name!r} (try op='list')")
    lines = _field_lines(project)
    try:
        lines.extend(_session_lines(build_project_view(project, config_dir=registry.config_dir)))
    except Exception as exc:  # noqa: BLE001 — the record is the point; the view degrades
        logger.warning("could not compose project view for %s: %s", project.name, exc)
        lines.append("linked sessions: (view unavailable this call)")
    body = "\n".join(lines)
    text, spill = spill_truncate(body, "project", context)
    return _text(tool_call_id, "project", text, details=spill or None)


async def _op_create(
    context: ToolContext | None, tool_call_id: str, params: ProjectParams
) -> ToolResult:
    registry = _registry(context)
    if registry is None:
        return _error(
            tool_call_id, "project", "no project registry attached to this session; cannot save."
        )
    session_id = _calling_session_id(context)
    try:
        fields = _project_edit(params, creating=True)
        project = registry.create_project(fields, sessions=[session_id] if session_id else ())
    except ProjectRegistryLockTimeout as exc:
        return _error(tool_call_id, "project", str(exc))
    except ProjectSchemaGuardError as exc:
        # A row written by a newer build is READABLE but never mutable; the
        # remedy in the message is the whole answer the model should relay.
        return _error(tool_call_id, "project", str(exc))
    except ProjectNameConflictError as exc:
        # The teams tool's sentence, for the same reason: the model's next move
        # is op='update', so the refusal names it rather than leaving it to
        # guess ("already exists" alone reads as a dead end).
        return _error(tool_call_id, "project", f"{exc}; use op='update' to change it.")
    except (ValueError, ValidationError) as exc:
        return _error(tool_call_id, "project", readable_error(exc))
    except Exception as exc:  # noqa: BLE001
        return _error(tool_call_id, "project", f"could not save project: {exc}")
    if session_id:
        receipt = (
            f"created project {project.name!r} [{project.status}] and linked this session "
            f"(session {session_id}). "
        )
    else:
        receipt = (
            f"created project {project.name!r} [{project.status}] with no session link "
            "(this host has no session id). "
        )
    return _text(
        tool_call_id,
        "project",
        receipt
        + "Update status/progress with op='update' as the work moves; read guide://projects.",
    )


async def _op_update(
    context: ToolContext | None, tool_call_id: str, params: ProjectParams
) -> ToolResult:
    registry = _registry(context)
    if registry is None:
        return _error(tool_call_id, "project", "no project registry attached to this session.")
    name = (params.name or "").strip()
    try:
        project = registry.get_project_by_name(name)
    except ProjectRegistryLockTimeout as exc:
        return _error(tool_call_id, "project", str(exc))
    if project is None:
        return _error(
            tool_call_id, "project", f"no project named {name!r} to update (try op='list')."
        )
    reporter = _calling_session_id(context) or "operator"
    try:
        fields = _project_edit(params, creating=False)
        outcome = registry.update_project(project.id, fields, reporter=reporter)
    except ProjectRegistryLockTimeout as exc:
        return _error(tool_call_id, "project", str(exc))
    except ProjectSchemaGuardError as exc:
        return _error(tool_call_id, "project", str(exc))
    except (ValueError, ValidationError) as exc:
        return _error(tool_call_id, "project", readable_error(exc))
    except Exception as exc:  # noqa: BLE001
        return _error(tool_call_id, "project", f"could not update project: {exc}")
    updated = outcome.project
    if not outcome.changed:
        return _text(
            tool_call_id,
            "project",
            f"project {updated.name!r} already held those values — nothing written.",
        )
    if outcome.refreshed:
        return _text(
            tool_call_id,
            "project",
            f"refreshed project {updated.name!r} — the progress line is unchanged "
            f"and now dated today ({reporter}).",
        )
    age = _reported_age(updated)
    detail = f"progress {age} ago" if age is not None else "no progress recorded"
    if "progress" not in params.model_fields_set:
        # Only mention the snippet when the call itself touched it: a
        # status-only update that says "progress 3h ago" invites the model to
        # think it reported something.
        detail = f"status {updated.status}"
    return _text(
        tool_call_id,
        "project",
        f"updated project {updated.name!r} [{updated.status}] — {detail}.",
    )


def _link_target(context: ToolContext | None, params: ProjectParams) -> str | None:
    explicit = (params.session_id or "").strip()
    if explicit:
        return explicit
    return _calling_session_id(context)


async def _op_link(
    context: ToolContext | None, tool_call_id: str, params: ProjectParams, *, linking: bool
) -> ToolResult:
    registry = _registry(context)
    if registry is None:
        return _error(tool_call_id, "project", "no project registry attached to this session.")
    name = (params.name or "").strip()
    session_id = _link_target(context, params)
    if not session_id:
        return _error(
            tool_call_id,
            "project",
            "no session to link: this host has no session id and 'session_id' was not given.",
        )
    try:
        project = registry.get_project_by_name(name)
    except ProjectRegistryLockTimeout as exc:
        return _error(tool_call_id, "project", str(exc))
    if project is None:
        return _error(tool_call_id, "project", f"no project named {name!r} (try op='list').")
    try:
        if linking:
            project, changed = registry.link_session(project.id, session_id)
        else:
            project, changed = registry.unlink_session(project.id, session_id)
    except ProjectRegistryLockTimeout as exc:
        return _error(tool_call_id, "project", str(exc))
    except (ValueError, ValidationError) as exc:
        return _error(tool_call_id, "project", readable_error(exc))
    except Exception as exc:  # noqa: BLE001
        return _error(tool_call_id, "project", f"could not update the link set: {exc}")
    verb = "linked" if linking else "unlinked"
    if linking and not changed:
        return _text(
            tool_call_id,
            "project",
            f"session {session_id} was already linked to {project.name!r} "
            f"({len(project.sessions)} linked).",
        )
    if not linking and not changed:
        return _error(
            tool_call_id,
            "project",
            f"session {session_id} is not linked to {project.name!r}; "
            "op='show' lists the linked sessions.",
        )
    return _text(
        tool_call_id,
        "project",
        f"{verb} session {session_id} {'to' if linking else 'from'} {project.name!r} "
        f"({len(project.sessions)} linked now).",
    )


async def _op_milestone(
    context: ToolContext | None, tool_call_id: str, params: ProjectParams
) -> ToolResult:
    registry = _registry(context)
    if registry is None:
        return _error(tool_call_id, "project", "no project registry attached to this session.")
    name = (params.name or "").strip()
    milestone_name = (params.milestone or "").strip()
    if not milestone_name:
        return _error(tool_call_id, "project", "op='milestone' needs 'milestone' (the name).")
    try:
        project = registry.get_project_by_name(name)
    except ProjectRegistryLockTimeout as exc:
        return _error(tool_call_id, "project", str(exc))
    if project is None:
        return _error(tool_call_id, "project", f"no project named {name!r} (try op='list').")
    payload: dict[str, Any] = {"name": milestone_name, "remove": params.remove}
    if "milestone_target_date" in params.model_fields_set:
        payload["target_date"] = params.milestone_target_date
    if "milestone_completed" in params.model_fields_set:
        payload["completed"] = params.milestone_completed
    try:
        edit = MilestoneEdit(**payload)
        project, action = registry.set_milestone(project.id, edit)
    except ProjectRegistryLockTimeout as exc:
        return _error(tool_call_id, "project", str(exc))
    except KeyError as exc:
        missing = exc.args[0] if exc.args else exc
        return _error(tool_call_id, "project", f"{missing} (try op='show')")
    except (ValueError, ValidationError) as exc:
        return _error(tool_call_id, "project", readable_error(exc))
    except Exception as exc:  # noqa: BLE001
        return _error(tool_call_id, "project", f"could not update the milestone: {exc}")
    if action == "unchanged":
        return _text(
            tool_call_id,
            "project",
            f"milestone {milestone_name!r} on {project.name!r} already read that way "
            "— nothing written.",
        )
    return _text(
        tool_call_id,
        "project",
        f"{action} milestone {milestone_name!r} on {project.name!r} "
        f"({_milestone_counts(project)}).",
    )


@_guard("project")
async def execute_project(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Track workstreams with the project primitive."""
    try:
        params = ProjectParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, "project", exc)

    needs_name = {"show", "create", "update", "link", "unlink", "milestone"}
    if params.op in needs_name and not (params.name or "").strip():
        return _error(tool_call_id, "project", f"op={params.op!r} needs 'name'.")
    if params.op == "list":
        return await _op_list(context, tool_call_id)
    if params.op == "show":
        return await _op_show(context, tool_call_id, str(params.name))
    if params.op == "create":
        return await _op_create(context, tool_call_id, params)
    if params.op == "update":
        return await _op_update(context, tool_call_id, params)
    if params.op in {"link", "unlink"}:
        return await _op_link(context, tool_call_id, params, linking=params.op == "link")
    return await _op_milestone(context, tool_call_id, params)


def build_project_tool(context: ToolContext) -> AgentTool | None:
    """createIf: the tool exists only where a registry can back it."""
    if getattr(context, "project_registry", None) is None:
        return None
    return AgentTool(
        name="project",
        label="Projects",
        description=(
            "Track multi-session workstreams: create a project (auto-linked to "
            "this session), report honest progress, link sessions, set "
            "dates/estimate/milestones, and read the aggregated view of its "
            "sessions (runtime state, subagents, todos). Read guide://projects "
            "before first use."
        ),
        parameters=ProjectParams.model_json_schema(),
        approval_tier="read",
        concurrency="exclusive",
        interruptible=False,
        execute=execute_project,
    )


@_guard("project_delete")
async def execute_project_delete(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Delete one project after the loop has passed the write approval gate."""
    try:
        params = ProjectDeleteParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, "project_delete", exc)
    name = params.name.strip()
    if not name:
        return _error(tool_call_id, "project_delete", "name must not be empty.")
    registry = _registry(context)
    if registry is None:
        return _error(
            tool_call_id, "project_delete", "no project registry attached to this session."
        )
    try:
        project = registry.get_project_by_name(name)
    except ProjectRegistryLockTimeout as exc:
        return _error(tool_call_id, "project_delete", str(exc))
    if project is None:
        return _error(tool_call_id, "project_delete", f"no project named {name!r} to delete.")
    try:
        registry.delete_project(project.id)
    except Exception as exc:  # noqa: BLE001
        return _error(tool_call_id, "project_delete", f"could not delete project: {exc}")
    return _text(tool_call_id, "project_delete", f"deleted project {name!r}.")


def build_project_delete_tool(context: ToolContext) -> AgentTool | None:
    """createIf: destructive deletion exists only beside a real registry."""
    if getattr(context, "project_registry", None) is None:
        return None
    return AgentTool(
        name="project_delete",
        label="Delete project",
        description=(
            "Permanently remove a saved project row. This is separate from the "
            "project tool so deletion always asks for write approval; session "
            "directories are never touched."
        ),
        parameters=ProjectDeleteParams.model_json_schema(),
        approval_tier="write",
        concurrency="exclusive",
        interruptible=False,
        describe_approval=lambda args, _cwd: (
            f"Delete project {str(args.get('name') or '').strip()!r} permanently"
        ),
        execute=execute_project_delete,
    )
